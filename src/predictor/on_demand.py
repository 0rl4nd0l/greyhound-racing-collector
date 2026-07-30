"""Safety core for isolated, on-demand pre-jump research predictions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import stat
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MODEL_ALIASES = {
    "latest-research": "market_form_residual_v1",
    "market-form-residual-v1": "market_form_residual_v1",
    "market_form_residual_v1": "market_form_residual_v1",
    "market-only": "market_only_v1",
    "market-only-implied": "market_only_v1",
    "market_only_implied": "market_only_v1",
    "market_only_v1": "market_only_v1",
}
MODEL_FILES = {
    "market_only_v1": None,
    "market_form_residual_v1": ROOT / "artifacts/frozen_models/market_form_residual_v1",
}
SCHEMA_FILES = {
    "market_only_v1": ROOT / "configs/prediction/schemas/market_only_v1.schema.json",
    "market_form_residual_v1": ROOT
    / "configs/prediction/schemas/market_form_residual_v1.schema.json",
}
OUTCOME_KEYS = {
    "actual_win",
    "finish_position",
    "official_result",
    "outcome",
    "placing",
    "result",
    "winner",
    "winner_name",
}


class PredictionBlocked(RuntimeError):
    """Fail-closed operator result with one stable blocker code."""

    def __init__(self, code: str, **details: Any) -> None:
        super().__init__(code)
        self.code = code
        self.details = details


@dataclass(frozen=True)
class ModelIdentity:
    requested: str
    resolved: str
    alias: bool
    model_path: Path | None
    manifest_path: Path | None
    model_sha256: str | None
    manifest_sha256: str | None
    schema_path: Path
    schema_sha256: str


@dataclass
class Dependencies:
    schedule: Callable[
        [datetime, float, Path, Path, int], Sequence[Mapping[str, Any]]
    ]
    seal_features: Callable[..., Mapping[str, Path]]
    score_residual: Callable[..., Mapping[str, Any]]
    now: Callable[[], datetime]
    capture_one: Callable[..., Mapping[str, Any]] | None = None
    monotonic: Callable[[], float] = time.monotonic


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _contains_outcome(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in OUTCOME_KEYS or _contains_outcome(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_outcome(item) for item in value)
    return False


def _write_canonical(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise PredictionBlocked("WRITE_TARGET_EXISTS", path=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        try:
            handle = os.fdopen(descriptor, "wb")
        except Exception:
            os.close(descriptor)
            raise
        with handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _copy_exact(source: Path, target: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise PredictionBlocked("SOURCE_FILE_UNSAFE", path=str(source))
    if target.exists() or target.is_symlink():
        raise PredictionBlocked("WRITE_TARGET_EXISTS", path=str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o600)
    try:
        try:
            writer = os.fdopen(descriptor, "wb")
        except Exception:
            os.close(descriptor)
            raise
        with source.open("rb") as reader, writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
    except Exception:
        target.unlink(missing_ok=True)
        raise


def write_exact_bytes(target: Path, value: bytes) -> None:
    if target.exists() or target.is_symlink():
        raise PredictionBlocked("WRITE_TARGET_EXISTS", path=str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o600)
    try:
        try:
            handle = os.fdopen(descriptor, "wb")
        except Exception:
            os.close(descriptor)
            raise
        with handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        target.unlink(missing_ok=True)
        raise


def create_bundle(output_root: Path, now: datetime) -> Path:
    root = output_root.resolve()
    if output_root.is_symlink() or (root.exists() and not root.is_dir()):
        raise PredictionBlocked("OUTPUT_ROOT_UNSAFE", path=str(output_root))
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    mode = stat.S_IMODE(root.stat().st_mode)
    if mode & 0o022:
        raise PredictionBlocked("OUTPUT_ROOT_WRITABLE_BY_OTHERS", path=str(root))
    bundle = (
        root / f"prediction_{now.strftime('%Y%m%dT%H%M%S%f%z')}_{uuid.uuid4().hex[:12]}"
    )
    bundle.mkdir(mode=0o700)
    return bundle


def resolve_model(requested: str) -> ModelIdentity:
    normalized = requested.strip().lower()
    resolved = MODEL_ALIASES.get(normalized)
    if resolved is None:
        raise PredictionBlocked("MODEL_UNSUPPORTED", requested=requested)
    schema_path = SCHEMA_FILES[resolved]
    if not schema_path.is_file():
        raise PredictionBlocked("MODEL_SCHEMA_MISSING", path=str(schema_path))
    artifact_dir = MODEL_FILES[resolved]
    model_path = artifact_dir / "model.json" if artifact_dir else None
    manifest_path = artifact_dir / "manifest.json" if artifact_dir else None
    if artifact_dir and (not model_path.is_file() or not manifest_path.is_file()):
        raise PredictionBlocked("MODEL_ARTIFACT_MISSING", model=resolved)
    return ModelIdentity(
        requested=requested,
        resolved=resolved,
        alias=normalized != resolved,
        model_path=model_path,
        manifest_path=manifest_path,
        model_sha256=sha256_file(model_path) if model_path else None,
        manifest_sha256=sha256_file(manifest_path) if manifest_path else None,
        schema_path=schema_path,
        schema_sha256=sha256_file(schema_path),
    )


def _validate_simple_schema(
    value: Any, schema: Mapping[str, Any], label: str = "config"
) -> None:
    expected_type = schema.get("type")
    type_map = {
        "object": dict,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
    }
    if expected_type in type_map and (
        not isinstance(value, type_map[expected_type])
        or isinstance(value, bool)
        and expected_type in {"integer", "number"}
    ):
        raise PredictionBlocked(
            "CONFIG_SCHEMA_MISMATCH", field=label, reason=f"type:{expected_type}"
        )
    if isinstance(value, Mapping):
        required = schema.get("required") or []
        missing = sorted(set(required) - set(value))
        if missing:
            raise PredictionBlocked(
                "CONFIG_SCHEMA_MISMATCH",
                field=label,
                reason=f"missing:{','.join(missing)}",
            )
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise PredictionBlocked(
                    "CONFIG_SCHEMA_MISMATCH",
                    field=label,
                    reason=f"extra:{','.join(extra)}",
                )
        for key, child in properties.items():
            if key in value:
                _validate_simple_schema(value[key], child, f"{label}.{key}")
    if "const" in schema and value != schema["const"]:
        raise PredictionBlocked("CONFIG_SCHEMA_MISMATCH", field=label, reason="const")
    if "enum" in schema and value not in schema["enum"]:
        raise PredictionBlocked("CONFIG_SCHEMA_MISMATCH", field=label, reason="enum")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise PredictionBlocked(
                "CONFIG_SCHEMA_MISMATCH", field=label, reason="minimum"
            )
        if "maximum" in schema and value > schema["maximum"]:
            raise PredictionBlocked(
                "CONFIG_SCHEMA_MISMATCH", field=label, reason="maximum"
            )


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON numeric constant: {value}")


def load_config(path: Path, model: ModelIdentity) -> tuple[dict[str, Any], str, bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw, parse_constant=_reject_nonfinite_json_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PredictionBlocked("CONFIG_INVALID_JSON", path=str(path)) from exc
    if not isinstance(value, dict) or canonical_bytes(value) != raw:
        raise PredictionBlocked("CONFIG_NOT_CANONICAL", path=str(path))
    if value.get("model") != model.resolved:
        raise PredictionBlocked(
            "MODEL_CONFIG_MISMATCH",
            requested=model.resolved,
            configured=value.get("model"),
        )
    try:
        schema = json.loads(model.schema_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictionBlocked(
            "MODEL_SCHEMA_INVALID", path=str(model.schema_path)
        ) from exc
    _validate_simple_schema(value, schema)
    return value, sha256_bytes(raw), raw


def runner_set_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    keys = sorted(
        f"{int(row['box_number'])}:{str(row.get('identity') or row.get('dog_name') or '').strip().upper()}"
        for row in rows
    )
    if len(keys) < 2 or len(keys) != len(set(keys)):
        raise PredictionBlocked("RUNNER_SET_AMBIGUOUS")
    return sha256_bytes(canonical_bytes(keys))


def normalize_validation_receipt(
    *,
    race_id: str,
    captured_at: datetime,
    validation: Mapping[str, Any],
    source_kind: str,
) -> dict[str, Any]:
    if validation.get("status") != "PASS" or validation.get("reasons") not in (
        None,
        [],
    ):
        raise PredictionBlocked(
            "MARKET_UNAVAILABLE", reasons=validation.get("reasons") or []
        )
    markets: dict[str, list[dict[str, Any]]] = {}
    for market, key in (("win", "accepted_rows"), ("place", "accepted_place_rows")):
        raw_rows = validation.get(key)
        if not isinstance(raw_rows, list) or not raw_rows:
            raise PredictionBlocked("MARKET_UNAVAILABLE", market=market)
        rows: list[dict[str, Any]] = []
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                raise PredictionBlocked("RUNNER_SET_AMBIGUOUS", market=market)
            try:
                box = int(raw.get("box_number"))
                odds = float(raw.get("odds_decimal"))
            except (TypeError, ValueError) as exc:
                raise PredictionBlocked("RUNNER_SET_AMBIGUOUS", market=market) from exc
            identity = (
                str(raw.get("identity") or raw.get("dog_name") or "").strip().upper()
            )
            dog_name = str(
                raw.get("dog_name") or raw.get("dog_clean_name") or ""
            ).strip()
            if (
                not 1 <= box <= 10
                or not identity
                or not dog_name
                or not math.isfinite(odds)
                or odds <= 1
            ):
                raise PredictionBlocked("RUNNER_SET_AMBIGUOUS", market=market)
            rows.append(
                {
                    "box_number": box,
                    "dog_name": dog_name,
                    "identity": identity,
                    "odds_decimal": odds,
                }
            )
        rows.sort(key=lambda row: (row["box_number"], row["identity"]))
        if len({(row["box_number"], row["identity"]) for row in rows}) != len(rows):
            raise PredictionBlocked("RUNNER_SET_AMBIGUOUS", market=market)
        markets[market] = rows
    win_keys = {(row["box_number"], row["identity"]) for row in markets["win"]}
    place_keys = {(row["box_number"], row["identity"]) for row in markets["place"]}
    if win_keys != place_keys:
        raise PredictionBlocked(
            "RUNNER_SET_AMBIGUOUS", reason="win_place_runner_mismatch"
        )
    return {
        "schema_version": "on_demand_odds_receipt_v1",
        "race_id": race_id,
        "captured_at": captured_at.isoformat(),
        "source_kind": source_kind,
        "source_url": validation.get("source_url"),
        "markets": markets,
        "runner_set_sha256": runner_set_sha256(markets["win"]),
    }


def receipt_from_handoff(
    receipt: Mapping[str, Any], *, current_time: datetime, max_age_seconds: int
) -> tuple[dict[str, Any], bytes, bytes, bytes]:
    try:
        captured_at = datetime.fromisoformat(str(receipt["append_timestamp"]))
        report_raw = bytes(receipt["_report_bytes"])
        form_raw = bytes(receipt["_form_bytes"])
        sidecar_raw = bytes(receipt["_sidecar_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PredictionBlocked("RECEIPT_INVALID") from exc
    if captured_at.tzinfo is None or captured_at.utcoffset() is None:
        raise PredictionBlocked("RECEIPT_INVALID", reason="timestamp_timezone_missing")
    age = (current_time - captured_at).total_seconds()
    if age < 0 or age > max_age_seconds:
        raise PredictionBlocked("RECEIPT_STALE", age_seconds=age)
    try:
        report = json.loads(report_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PredictionBlocked("RECEIPT_INVALID") from exc
    if _contains_outcome(report):
        raise PredictionBlocked("RECEIPT_CONTAINS_OUTCOME")
    attempts = report.get("attempts") if isinstance(report, Mapping) else None
    matches = [
        row
        for row in attempts or []
        if isinstance(row, Mapping)
        and row.get("race_id") == receipt.get("race_id")
        and row.get("status") == "APPENDED"
        and isinstance(row.get("validation"), Mapping)
        and row["validation"].get("status") == "PASS"
    ]
    if len(matches) != 1:
        raise PredictionBlocked("RECEIPT_AMBIGUOUS")
    normalized = normalize_validation_receipt(
        race_id=str(receipt.get("race_id")),
        captured_at=captured_at,
        validation=matches[0]["validation"],
        source_kind="verified_autonomous_receipt",
    )
    expected_hashes = {
        "source_report_sha256": sha256_bytes(report_raw),
        "source_form_sha256": sha256_bytes(form_raw),
        "source_sidecar_sha256": sha256_bytes(sidecar_raw),
    }
    if any(receipt.get(key) != value for key, value in expected_hashes.items()):
        raise PredictionBlocked("RECEIPT_TAMPERED")
    normalized["source_hashes"] = expected_hashes
    normalized["handoff"] = {
        key: value for key, value in receipt.items() if not str(key).startswith("_")
    }
    return normalized, report_raw, form_raw, sidecar_raw


def _table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')]


def seal_history_database(
    *,
    source: Path,
    target: Path,
    target_race_id: str,
    cutoff: datetime,
    runner_names: Sequence[str],
) -> dict[str, Any]:
    if not source.is_file() or target.exists():
        raise PredictionBlocked("HISTORY_DATABASE_UNAVAILABLE")
    source_uri = f"file:{source.resolve()}?mode=ro"
    source_db = sqlite3.connect(source_uri, uri=True)
    source_db.row_factory = sqlite3.Row
    source_db.execute("PRAGMA query_only=ON")
    source_db.execute("BEGIN")
    target.parent.mkdir(parents=True, exist_ok=True)
    target_db = sqlite3.connect(target)
    try:
        if source_db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise PredictionBlocked("HISTORY_DATABASE_INTEGRITY_FAILED")
        schemas: dict[str, str] = {}
        for table in ("race_metadata", "dog_race_data"):
            row = source_db.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if row is None or not row[0]:
                raise PredictionBlocked("HISTORY_SCHEMA_MISSING", table=table)
            schemas[table] = str(row[0])
            target_db.execute(schemas[table])
        race_columns = _table_columns(source_db, "race_metadata")
        dog_columns = _table_columns(source_db, "dog_race_data")
        if (
            not {"race_id", "race_date"}.issubset(race_columns)
            or "race_id" not in dog_columns
        ):
            raise PredictionBlocked("HISTORY_SCHEMA_AMBIGUOUS")
        duplicate_ids = source_db.execute(
            "SELECT race_id FROM race_metadata WHERE race_id IS NOT NULL GROUP BY race_id HAVING COUNT(*) != 1 LIMIT 1"
        ).fetchone()
        if duplicate_ids:
            raise PredictionBlocked(
                "HISTORY_IDENTITY_AMBIGUOUS", race_id=duplicate_ids[0]
            )
        cutoff_date = cutoff.date().isoformat()
        metadata_rows = [
            dict(row)
            for row in source_db.execute(
                'SELECT "race_id", "race_date" FROM race_metadata'
            )
        ]
        safe_ids: set[str] = set()
        excluded_target = excluded_at_or_after = ambiguous_dates = 0
        for row in metadata_rows:
            race_id = str(row.get("race_id") or "")
            raw_date = str(row.get("race_date") or "")
            if race_id == target_race_id:
                excluded_target += 1
                continue
            try:
                parsed = date.fromisoformat(raw_date[:10])
            except ValueError:
                ambiguous_dates += 1
                continue
            if parsed.isoformat() >= cutoff_date:
                excluded_at_or_after += 1
                continue
            if not race_id:
                ambiguous_dates += 1
                continue
            safe_ids.add(race_id)

        def rows_for_safe_ids(table: str) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            ordered = sorted(safe_ids)
            for offset in range(0, len(ordered), 500):
                chunk = ordered[offset : offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(
                    dict(row)
                    for row in source_db.execute(
                        f'SELECT * FROM "{table}" WHERE "race_id" IN ({placeholders})',
                        tuple(chunk),
                    )
                )
            return rows

        safe_metadata = rows_for_safe_ids("race_metadata")
        normalized_names = {
            name.strip().upper() for name in runner_names if name.strip()
        }
        name_column = (
            "dog_clean_name"
            if "dog_clean_name" in dog_columns
            else "dog_name"
            if "dog_name" in dog_columns
            else None
        )
        if name_column is None:
            raise PredictionBlocked(
                "HISTORY_SCHEMA_AMBIGUOUS", reason="dog_name_missing"
            )
        relevant_ambiguous: list[str] = []
        if normalized_names:
            relevant_history = source_db.execute(
                f'''SELECT DISTINCT dr.race_id, rm.race_date
                    FROM dog_race_data dr
                    LEFT JOIN race_metadata rm ON rm.race_id = dr.race_id
                    WHERE UPPER(TRIM(COALESCE(dr."{name_column}", ''))) IN ({",".join("?" for _ in normalized_names)})''',
                tuple(sorted(normalized_names)),
            )
            for history_row in relevant_history:
                history_race_id = str(history_row[0] or "")
                raw_history_date = str(history_row[1] or "")
                try:
                    date.fromisoformat(raw_history_date[:10])
                except ValueError:
                    relevant_ambiguous.append(history_race_id or "<missing>")
        if relevant_ambiguous:
            raise PredictionBlocked(
                "HISTORY_CUTOFF_AMBIGUOUS",
                race_ids=sorted(relevant_ambiguous),
                row_count=len(relevant_ambiguous),
            )

        def insert_rows(
            table: str, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
        ) -> None:
            if not rows:
                return
            column_sql = ",".join(f'"{column}"' for column in columns)
            placeholders = ",".join("?" for _ in columns)
            target_db.executemany(
                f'INSERT INTO "{table}" ({column_sql}) VALUES ({placeholders})',
                [tuple(row.get(column) for column in columns) for row in rows],
            )

        insert_rows("race_metadata", race_columns, safe_metadata)
        dog_rows = rows_for_safe_ids("dog_race_data")
        insert_rows("dog_race_data", dog_columns, dog_rows)
        target_db.commit()
        target_db.execute("PRAGMA optimize")
    finally:
        target_db.close()
        source_db.close()
    return {
        "schema_version": "sealed_prediction_history_v1",
        "source_sha256": sha256_file(source),
        "sealed_sha256": sha256_file(target),
        "target_race_id": target_race_id,
        "cutoff_timestamp": cutoff.isoformat(),
        "cutoff_basis": "race_date_strictly_before_target_jump_date",
        "safe_race_count": len(safe_ids),
        "safe_dog_row_count": len(dog_rows),
        "excluded_target_metadata_rows": excluded_target,
        "excluded_at_or_after_cutoff_metadata_rows": excluded_at_or_after,
        "excluded_ambiguous_date_metadata_rows": ambiguous_dates,
        "target_rows_materialized": 0,
        "at_or_after_cutoff_rows_materialized": 0,
    }


def market_only_prediction(receipt: Mapping[str, Any]) -> dict[str, Any]:
    win_rows = list((receipt.get("markets") or {}).get("win") or [])
    inverse = [1.0 / float(row["odds_decimal"]) for row in win_rows]
    total = sum(inverse)
    if len(win_rows) < 2 or not math.isfinite(total) or total <= 0:
        raise PredictionBlocked("MARKET_UNAVAILABLE")
    predictions = [
        {
            "box_number": int(row["box_number"]),
            "dog_name": str(row["dog_name"]),
            "probability": value / total,
            "win_odds": float(row["odds_decimal"]),
        }
        for row, value in zip(win_rows, inverse)
    ]
    predictions.sort(key=lambda row: (-row["probability"], row["box_number"]))
    for rank, row in enumerate(predictions, start=1):
        row["rank"] = rank
    return {
        "adapter": "market_only_v1",
        "variant": "market_only_implied",
        "probability_sum": sum(row["probability"] for row in predictions),
        "predictions": predictions,
    }


def bundle_manifest(
    bundle: Path, *, exclude: Sequence[str] = ("bundle_manifest.json",)
) -> dict[str, Any]:
    excluded = set(exclude)
    files = {
        path.relative_to(bundle).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(bundle.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.relative_to(bundle).as_posix() not in excluded
    }
    return {"schema_version": "on_demand_prediction_bundle_manifest_v1", "files": files}


def verify_bundle(bundle: Path) -> dict[str, Any]:
    if any(path.is_symlink() for path in bundle.rglob("*")):
        raise PredictionBlocked("REPLAY_TAMPERED")
    manifest_path = bundle / "bundle_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictionBlocked("REPLAY_MANIFEST_INVALID") from exc
    expected = manifest.get("files") if isinstance(manifest, Mapping) else None
    if not isinstance(expected, Mapping):
        raise PredictionBlocked("REPLAY_MANIFEST_INVALID")
    actual = bundle_manifest(bundle)["files"]
    if actual != expected:
        raise PredictionBlocked("REPLAY_TAMPERED")
    return dict(manifest)
