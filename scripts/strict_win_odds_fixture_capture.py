#!/usr/bin/env python3
"""Build sealed report-only WIN odds replay fixtures for future races.

This script consumes a future-race capture plan and a raw fetch-result payload,
writes immutable replay fixtures and normalized projections under a report-only
artifact directory, and validates strict WIN odds provenance. It never appends
to SQLite and never updates registries, model pointers, snapshots, manifests,
TGR, EV, betting, odds-capture state, or official results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from accuracy_program.odds_coverage import normalize_dog_name  # noqa: E402
from accuracy_program.snapshots import RESULT_FIELD_NAMES  # noqa: E402


DEFAULT_OUTPUT_PARENT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = (
    "artifacts/full_evidence_orchestration_20260525/strict_win_odds_fixture_capture_"
)
RAW_FIXTURE_SCHEMA = "strict_win_odds_raw_replay_fixture_v2"
NORMALIZED_PROJECTION_SCHEMA = "strict_win_odds_normalized_projection_v2"
MANIFEST_SCHEMA = "strict_win_odds_fixture_manifest_v2"
PACKET_REPORT_SCHEMA = "strict_win_odds_fixture_packet_report_v2"
VALIDATION_SCHEMA = "strict_win_odds_fixture_validation_v2"
PRESEAL_VALIDATION_SCHEMA = "strict_win_odds_fixture_packet_validation_preseal_v2"
PACKET_VALIDATOR_SCHEMA = "strict_win_odds_fixture_packet_validator_v2"
CAPTURE_MODE = "strict_win_fixture_v2"
COLLECTOR_PATH = "scripts/strict_win_odds_fixture_capture.py"
FINAL_PLAN_ONLY_DONE = "STRICT_WIN_FUTURE_COLLECTION_PLAN_ONLY_DONE"
FINAL_SEALED_NO_DB_APPEND = "STRICT_WIN_FIXTURE_PACKET_SEALED_NO_DB_APPEND"
FINAL_BLOCKED_VALIDATION_FAILED = "STRICT_WIN_FIXTURE_PACKET_BLOCKED_VALIDATION_FAILED"
FINAL_NO_READY_RACES = "STRICT_WIN_FIXTURE_COLLECTION_NO_READY_RACES"
POST_RACE_SOURCE_URL_MARKERS = ("result", "results", "dividend", "payout", "sp-only")
NO_WRITE_GUARANTEES = {
    "db_write": False,
    "registry_mutation": False,
    "model_pointer_update": False,
    "snapshot_write": False,
    "protected_manifest_write": False,
    "tgr_write": False,
    "ev_output": False,
    "betting_action": False,
    "runtime_odds_capture_write": False,
    "official_result_write": False,
    "training": False,
}
FORBIDDEN_FIELD_NAMES = frozenset(
    RESULT_FIELD_NAMES
    | {
        "outcome",
        "outcomes",
        "result",
        "results",
        "official_outcome",
        "official_outcomes",
        "winner_box",
        "winning_box",
        "finish_order",
        "finish",
        "finishes",
        "finishing_order",
        "finishing_position",
        "dividend",
        "dividends",
        "is_placer",
        "is_winner",
        "margin",
        "margins",
        "official_position",
        "official_winner",
        "payout",
        "payouts",
        "result_position",
        "starting_price",
    }
)
MANIFEST_PATH = "strict_win_fixture_manifest.json"
PRESEAL_PATH = "strict_win_fixture_validation_preseal.json"
PACKET_REPORT_PATH = "strict_win_fixture_packet_report.json"
FINAL_STATUS_PATH = "final_status.json"
SINGLETON_MANIFEST_ROLES = {
    "preseal_validation": PRESEAL_PATH,
    "packet_report": PACKET_REPORT_PATH,
    "final_status": FINAL_STATUS_PATH,
}


def now_id(now: datetime | None = None) -> str:
    return (now or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")


def serialized_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, default=str)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_payload(payload: object) -> str:
    return sha256_bytes(canonical_bytes(payload))


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_file_bytes(path: Path) -> bytes:
    return path.read_bytes()


def parse_json_bytes(payload: bytes, *, context: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context}_json_invalid") from exc


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        logical = path if path.is_absolute() else ROOT / path
        return logical.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def resolve_repo_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def resolve_repo_file(value: Any) -> Path | None:
    """Resolve an existing regular file without permitting repository escape."""

    candidate = resolve_repo_path(value)
    if candidate is None:
        return None
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(ROOT.resolve())
    except (FileNotFoundError, RuntimeError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def read_json(path: Path) -> Any:
    return parse_json_bytes(read_file_bytes(path), context="input")


def write_bytes_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def write_json_new(path: Path, payload: object) -> bytes:
    encoded = serialized_json_bytes(payload)
    write_bytes_new(path, encoded)
    return encoded


def normalized_field_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def assert_outcome_free(value: Any, *, context: str) -> None:
    """Reject outcome-bearing keys recursively using one contract-wide check."""

    def visit(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                key_name = normalized_field_name(key)
                if key_name in FORBIDDEN_FIELD_NAMES:
                    raise ValueError(f"forbidden_outcome_field:{context}:{path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(node, (list, tuple)):
            for index, child in enumerate(node):
                visit(child, f"{path}[{index}]")

    visit(value, context)


def assert_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    repo_root = ROOT.resolve()
    try:
        resolved = logical.resolve(strict=False)
        relative = resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    except RuntimeError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if not relative.as_posix().startswith(
        "artifacts/full_evidence_orchestration_20260525/"
    ):
        raise ValueError("output_dir_must_be_under_artifacts")
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_strict_win_fixture_artifact:{relative}")
    return resolved


def parse_current_time(value: str | None) -> datetime:
    if not value:
        return datetime.now().astimezone()
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    if len(text) >= 5 and text[-5] in {"+", "-"} and text[-4:].isdigit():
        text = f"{text[:-5]}{text[-5:-2]}:{text[-2:]}"
    parsed = datetime.fromisoformat(text)
    return parsed.astimezone() if parsed.tzinfo is None else parsed


def parse_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        if len(text) >= 5 and text[-5] in {"+", "-"} and text[-4:].isdigit():
            text = f"{text[:-5]}{text[-5:-2]}:{text[-2:]}"
        parsed = None
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            for fmt in (
                "%Y-%m-%d %I:%M %p",
                "%Y-%m-%d %I:%M:%S %p",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d %H:%M:%S",
            ):
                try:
                    parsed = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
        if parsed is None:
            return None
    return parsed


def has_timezone(value: datetime | None) -> bool:
    return (
        value is not None and value.tzinfo is not None and value.utcoffset() is not None
    )


def safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(str(value).strip())
    except Exception:
        return None


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def first_present(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def source_url_is_trusted_sportsbet(url: Any) -> bool:
    try:
        parsed = urlparse(str(url or ""))
        host = (parsed.hostname or "").lower()
    except (TypeError, ValueError):
        return False
    return parsed.scheme.lower() == "https" and (
        host == "sportsbet.com.au" or host.endswith(".sportsbet.com.au")
    )


def source_url_looks_post_race(url: Any) -> bool:
    text = str(url or "").lower()
    return any(marker in text for marker in POST_RACE_SOURCE_URL_MARKERS)


def source_url_from_fetch_result(fetch_result: Mapping[str, Any]) -> str | None:
    race_info = fetch_result.get("race_info")
    if not isinstance(race_info, Mapping):
        race_info = {}
    value = (
        fetch_result.get("source_url")
        or fetch_result.get("sportsbet_url")
        or fetch_result.get("venue_url")
        or race_info.get("source_url")
        or race_info.get("venue_url")
        or race_info.get("sportsbet_url")
        or race_info.get("url")
    )
    return str(value) if value not in (None, "") else None


def normalized_source_url(value: Any) -> str | None:
    if not source_url_is_trusted_sportsbet(value):
        return None
    try:
        parsed = urlparse(str(value).strip())
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except (TypeError, ValueError):
        return None
    netloc = host if port in (None, 443) else f"{host}:{port}"
    return (
        parsed._replace(scheme="https", netloc=netloc, fragment="").geturl().rstrip("/")
    )


def normalized_venue(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())


def runner_is_active(row: Mapping[str, Any]) -> bool | None:
    active = row.get("active")
    scratched = row.get("scratched", row.get("is_scratched"))
    status = str(row.get("runner_status") or row.get("status") or "").strip().lower()
    if isinstance(active, bool) and isinstance(scratched, bool):
        return active if active == (not scratched) else None
    if isinstance(active, bool):
        return active
    if isinstance(scratched, bool):
        return not scratched
    if status in {"active", "runner", "open"}:
        return True
    if status in {"scratched", "scratch", "inactive", "withdrawn"}:
        return False
    return True


def normalize_runner(dog_name: Any, box_number: Any) -> dict[str, Any] | None:
    box = safe_int(box_number)
    name = str(dog_name or "").strip()
    identity = normalize_dog_name(name)
    if box is None or not identity:
        return None
    return {"box_number": box, "dog_name": name, "identity": identity}


def expected_runner_rows(plan_item: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in plan_item.get("expected_runners") or []:
        if not isinstance(item, Mapping):
            continue
        runner = normalize_runner(item.get("dog_name"), item.get("box_number"))
        if runner:
            runner["active"] = runner_is_active(item)
            rows.append(runner)
    return rows


def raw_win_rows(fetch_result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    race_info = fetch_result.get("race_info")
    if not isinstance(race_info, Mapping):
        race_info = {}
    for value in (
        fetch_result.get("odds_data"),
        fetch_result.get("win_odds"),
        fetch_result.get("runner_rows"),
        race_info.get("odds_data"),
        race_info.get("win_odds"),
    ):
        if isinstance(value, list):
            return [row for row in value if isinstance(row, Mapping)]
    return []


def raw_win_row_container(fetch_result: Mapping[str, Any]) -> list[Any] | None:
    race_info = fetch_result.get("race_info")
    if not isinstance(race_info, Mapping):
        race_info = {}
    for value in (
        fetch_result.get("odds_data"),
        fetch_result.get("win_odds"),
        fetch_result.get("runner_rows"),
        race_info.get("odds_data"),
        race_info.get("win_odds"),
    ):
        if isinstance(value, list):
            return value
    return None


def fetch_result_count_only_without_raw(fetch_result: Mapping[str, Any]) -> bool:
    win_count = safe_int(fetch_result.get("win_count"))
    return bool(win_count and not raw_win_rows(fetch_result))


def normalized_fetch_runner_row(
    row: Mapping[str, Any],
    *,
    expected_by_box: Mapping[int, str],
) -> dict[str, Any]:
    box = safe_int(first_present(row, ("box_number", "box", "runner_number")))
    dog_name = str(
        first_present(row, ("dog_name", "dog_clean_name", "runner_name", "name")) or ""
    ).strip()
    identity = normalize_dog_name(dog_name)
    odds_decimal = safe_float(
        first_present(row, ("odds_decimal", "decimal_odds", "price", "win_odds"))
    )
    raw_runner_text = first_present(
        row,
        (
            "sportsbet_raw_runner_text",
            "raw_runner_text",
            "runner_text",
            "raw_text",
        ),
    )
    if box is None:
        match_status = "box_missing"
    elif box not in expected_by_box:
        match_status = "unexpected_box"
    elif expected_by_box[box] == identity:
        match_status = "box_name_exact"
    else:
        match_status = "box_name_mismatch"
    return {
        "box_number": box,
        "dog_name": dog_name,
        "identity": identity,
        "odds_decimal": odds_decimal,
        "active": runner_is_active(row),
        "raw_runner_text": str(raw_runner_text or "").strip(),
        "source_list_position": safe_int(
            first_present(row, ("sportsbet_list_position", "list_position", "position"))
        ),
        "box_source": str(
            first_present(row, ("sportsbet_box_source", "box_source")) or "unknown"
        ),
        "match_status": match_status,
        "raw": dict(row),
    }


def safe_filename(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text[:160] or "unknown"


def fixture_source_sidecar(plan_item: Mapping[str, Any]) -> dict[str, Any]:
    sidecar = resolve_repo_file(plan_item.get("sidecar_path"))
    if sidecar is None:
        return {"path": None, "bytes": None, "payload": None, "sha256": None}
    try:
        sidecar_bytes = read_file_bytes(sidecar)
        sidecar_payload = parse_json_bytes(sidecar_bytes, context="source_sidecar")
    except (OSError, ValueError):
        return {
            "path": relpath(sidecar),
            "bytes": None,
            "payload": None,
            "sha256": None,
        }
    return {
        "path": relpath(sidecar),
        "bytes": sidecar_bytes,
        "payload": sidecar_payload,
        "sha256": sha256_bytes(sidecar_bytes),
    }


def build_raw_fixture(
    *,
    plan_item: Mapping[str, Any],
    fetch_result: Mapping[str, Any],
    sidecar_source: Mapping[str, Any] | None = None,
    collector_bytes: bytes | None = None,
) -> dict[str, Any]:
    expected = expected_runner_rows(plan_item)
    expected_by_box = {row["box_number"]: row["identity"] for row in expected}
    source_url = source_url_from_fetch_result(fetch_result)
    raw_capture_timestamp = fetch_result.get("capture_timestamp") or fetch_result.get(
        "timestamp"
    )
    capture_dt = parse_timestamp(raw_capture_timestamp)
    capture_timestamp = (
        capture_dt.isoformat()
        if capture_dt is not None
        else str(raw_capture_timestamp or "")
    )
    runner_rows = [
        normalized_fetch_runner_row(row, expected_by_box=expected_by_box)
        for row in raw_win_rows(fetch_result)
    ]
    sidecar = sidecar_source or fixture_source_sidecar(plan_item)
    if collector_bytes is None:
        collector_bytes = read_file_bytes(Path(__file__).resolve())
    accepted_source_url = first_present(
        plan_item, ("sportsbet_url", "source_url", "accepted_source_url")
    )
    base = {
        "schema_version": RAW_FIXTURE_SCHEMA,
        "prior_states": {
            "denominator": "DATA_LINEAGE_BLOCKER_STOP",
            "odds_provenance_audit": "ODDS_CAPTURE_PROVENANCE_AUDIT_DONE",
            "future_plan": FINAL_PLAN_ONLY_DONE,
        },
        "race_id": plan_item.get("canonical_race_identity"),
        "venue": plan_item.get("venue"),
        "race_number": safe_int(plan_item.get("race_number")),
        "race_date": str(plan_item.get("race_date") or "")[:10] or None,
        "source_url": source_url,
        "accepted_source_url": accepted_source_url,
        "market_type": "win",
        "capture_timestamp": capture_timestamp,
        "jump_datetime": plan_item.get("jump_datetime"),
        "expected_runners": expected,
        "runner_rows": runner_rows,
        "raw_fetch_result": dict(fetch_result),
        "provenance": {
            "fetch_method": fetch_result.get("discovery_method"),
            "source_host": urlparse(str(source_url or "")).netloc.lower() or None,
            "source_sidecar_path": sidecar.get("path"),
            "source_sidecar_sha256": sidecar.get("sha256"),
            "plan_schema_version": plan_item.get("schema_version"),
            "plan_capture_mode": plan_item.get("capture_mode"),
            "collector": COLLECTOR_PATH,
            "collector_sha256": sha256_bytes(collector_bytes),
            "report_only": True,
            "append_approved": False,
            "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        },
    }
    fixture_id = sha256_payload(base)
    return {"fixture_id": fixture_id, **base}


def normalized_projection_from_fixture(fixture: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for row in fixture.get("runner_rows") or []:
        if not isinstance(row, Mapping):
            continue
        rows.append(
            {
                "race_id": fixture.get("race_id"),
                "dog_name": row.get("dog_name"),
                "dog_clean_name": row.get("dog_name"),
                "box_number": row.get("box_number"),
                "odds_decimal": row.get("odds_decimal"),
                "active": row.get("active"),
                "market_type": "win",
                "source": "sportsbet",
                "source_url": fixture.get("source_url"),
                "timestamp": fixture.get("capture_timestamp"),
                "capture_timestamp": fixture.get("capture_timestamp"),
                "capture_mode": CAPTURE_MODE,
                "odds_level": "dog",
                "sportsbet_box_source": row.get("box_source"),
                "sportsbet_list_position": row.get("source_list_position"),
                "sportsbet_raw_runner_text": row.get("raw_runner_text"),
                "fixture_id": fixture.get("fixture_id"),
                "match_status": row.get("match_status"),
            }
        )
    return {
        "schema_version": NORMALIZED_PROJECTION_SCHEMA,
        "fixture_id": fixture.get("fixture_id"),
        "race_id": fixture.get("race_id"),
        "venue": fixture.get("venue"),
        "race_number": fixture.get("race_number"),
        "race_date": fixture.get("race_date"),
        "market_type": "win",
        "capture_timestamp": fixture.get("capture_timestamp"),
        "jump_datetime": fixture.get("jump_datetime"),
        "source_url": fixture.get("source_url"),
        "accepted_source_url": fixture.get("accepted_source_url"),
        "append_allowed_without_owner_approval": False,
        "rows": rows,
    }


def duplicate_values(values: Sequence[Any]) -> list[Any]:
    counts = Counter(values)
    return sorted(
        value
        for value, count in counts.items()
        if value not in (None, "") and count > 1
    )


def fixture_content_id(fixture: Mapping[str, Any]) -> str:
    content = dict(fixture)
    content.pop("fixture_id", None)
    return sha256_payload(content)


def validate_source_sidecar(
    provenance: Any,
    *,
    sidecar_source: Mapping[str, Any] | None = None,
) -> tuple[list[str], Mapping[str, Any] | None]:
    if not isinstance(provenance, Mapping):
        return ["source_sidecar_missing_or_untrusted"], None
    sidecar_path = provenance.get("source_sidecar_path")
    if not isinstance(sidecar_path, str) or not sidecar_path.strip():
        return ["source_sidecar_missing_or_untrusted"], None
    logical_path = Path(sidecar_path)
    if logical_path.is_absolute() or ".." in logical_path.parts:
        return ["source_sidecar_missing_or_untrusted"], None
    loaded = sidecar_source
    if loaded is None:
        loaded = fixture_source_sidecar({"sidecar_path": sidecar_path})
    sidecar_bytes = loaded.get("bytes") if isinstance(loaded, Mapping) else None
    sidecar_payload = loaded.get("payload") if isinstance(loaded, Mapping) else None
    if not isinstance(sidecar_bytes, bytes) or not isinstance(sidecar_payload, Mapping):
        return ["source_sidecar_missing_or_untrusted"], None
    expected_hash = provenance.get("source_sidecar_sha256")
    if not isinstance(expected_hash, str) or expected_hash != sha256_bytes(
        sidecar_bytes
    ):
        return ["source_sidecar_sha256_mismatch"], sidecar_payload
    return [], sidecar_payload


def nested_metadata_value(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    value = first_present(payload, keys)
    if value not in (None, ""):
        return value
    race_info = payload.get("race_info")
    if isinstance(race_info, Mapping):
        return first_present(race_info, keys)
    return None


def record_runner_container(payload: Mapping[str, Any]) -> list[Any] | None:
    for key in ("expected_runners", "runners", "runner_rows", "odds_data"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    race_info = payload.get("race_info")
    if isinstance(race_info, Mapping):
        return record_runner_container(race_info)
    return None


def record_runner_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    container = record_runner_container(payload)
    if container is None:
        return []
    return [row for row in container if isinstance(row, Mapping)]


def normalized_roster(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, tuple[str, bool | None, float | None]]:
    roster: dict[int, tuple[str, bool | None, float | None]] = {}
    for row in rows:
        runner = normalize_runner(
            first_present(row, ("dog_name", "dog_clean_name", "runner_name", "name")),
            first_present(row, ("box_number", "box", "runner_number")),
        )
        if runner is None:
            continue
        odds = safe_float(
            first_present(row, ("odds_decimal", "decimal_odds", "price", "win_odds"))
        )
        roster[runner["box_number"]] = (
            runner["identity"],
            runner_is_active(row),
            odds,
        )
    return roster


def metadata_mismatch_reasons(
    fixture: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    prefix: str,
) -> list[str]:
    reasons: list[str] = []
    identities = fetch_record_keys(payload)
    if str(fixture.get("race_id") or "") not in identities:
        reasons.append(f"{prefix}_race_identity_mismatch")
    comparisons = (
        ("venue", normalized_venue, ("venue", "track")),
        ("race_number", safe_int, ("race_number", "race_no", "race")),
        ("race_date", lambda value: str(value or "")[:10], ("race_date", "date")),
    )
    for field, normalizer, keys in comparisons:
        expected = normalizer(fixture.get(field))
        actual = normalizer(nested_metadata_value(payload, keys))
        if not actual or actual != expected:
            reasons.append(f"{prefix}_{field}_mismatch")
    fixture_jump = parse_timestamp(fixture.get("jump_datetime"))
    payload_jump = parse_timestamp(
        nested_metadata_value(
            payload, ("jump_datetime", "start_datetime", "race_start", "jump_time")
        )
    )
    if (
        fixture_jump is None
        or payload_jump is None
        or not has_timezone(fixture_jump)
        or not has_timezone(payload_jump)
        or fixture_jump != payload_jump.astimezone(fixture_jump.tzinfo)
    ):
        reasons.append(f"{prefix}_jump_datetime_mismatch")
    source_url = normalized_source_url(source_url_from_fetch_result(payload))
    accepted_url = normalized_source_url(fixture.get("accepted_source_url"))
    if source_url is None or accepted_url is None or source_url != accepted_url:
        reasons.append(f"{prefix}_source_url_mismatch")
    market_type = (
        str(nested_metadata_value(payload, ("market_type", "market", "bet_type")) or "")
        .strip()
        .lower()
    )
    if market_type != "win":
        reasons.append(f"{prefix}_market_type_not_win")
    return reasons


def roster_mismatch_reasons(
    fixture: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    prefix: str,
) -> list[str]:
    expected = normalized_roster(
        [
            row
            for row in fixture.get("expected_runners") or []
            if isinstance(row, Mapping)
        ]
    )
    actual_rows = record_runner_rows(payload)
    actual_container = record_runner_container(payload)
    actual = normalized_roster(actual_rows)
    reasons: list[str] = []
    if actual_container is not None and any(
        not isinstance(row, Mapping) for row in actual_container
    ):
        reasons.append(f"{prefix}_runner_duplicate_or_malformed")
    if set(actual) != set(expected):
        reasons.append(f"{prefix}_runner_box_set_mismatch")
        return reasons
    if len(actual_rows) != len(actual):
        reasons.append(f"{prefix}_runner_duplicate_or_malformed")
    for box in sorted(expected):
        expected_identity, expected_active, _ = expected[box]
        actual_identity, actual_active, actual_odds = actual[box]
        if actual_identity != expected_identity:
            reasons.append(f"{prefix}_runner_{box}_identity_mismatch")
        if actual_active is None or actual_active != expected_active:
            reasons.append(f"{prefix}_runner_{box}_active_state_mismatch")
        if prefix == "source":
            fixture_row = next(
                (
                    row
                    for row in fixture.get("runner_rows") or []
                    if isinstance(row, Mapping)
                    and safe_int(row.get("box_number")) == box
                ),
                None,
            )
            fixture_odds = (
                safe_float(fixture_row.get("odds_decimal")) if fixture_row else None
            )
            if actual_odds != fixture_odds:
                reasons.append(f"{prefix}_runner_{box}_odds_mismatch")
        elif actual_odds is not None:
            fixture_row = next(
                (
                    row
                    for row in fixture.get("runner_rows") or []
                    if isinstance(row, Mapping)
                    and safe_int(row.get("box_number")) == box
                ),
                None,
            )
            fixture_odds = (
                safe_float(fixture_row.get("odds_decimal")) if fixture_row else None
            )
            if actual_odds != fixture_odds:
                reasons.append(f"{prefix}_runner_{box}_odds_mismatch")
    return reasons


def validate_fixture_payload(
    fixture: Mapping[str, Any],
    projection: Mapping[str, Any] | None = None,
    *,
    sidecar_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    for context, payload in (("fixture", fixture), ("projection", projection)):
        if payload is None:
            continue
        try:
            assert_outcome_free(payload, context=context)
        except ValueError as exc:
            reasons.append(str(exc))
    if fixture.get("schema_version") != RAW_FIXTURE_SCHEMA:
        reasons.append("fixture_schema_version_invalid")
    if not fixture.get("fixture_id"):
        reasons.append("fixture_id_missing")
    if fixture.get("fixture_id") != fixture_content_id(fixture):
        reasons.append("fixture_id_content_mismatch")
    if not fixture.get("race_id"):
        reasons.append("race_id_missing")
    if fixture.get("market_type") != "win":
        reasons.append("market_type_not_win")

    accepted_source_url = fixture.get("accepted_source_url")
    if normalized_source_url(accepted_source_url) is None:
        reasons.append("accepted_sportsbet_source_url_missing_or_untrusted")

    source_url = fixture.get("source_url")
    if not source_url:
        reasons.append("sportsbet_source_url_missing")
    elif not source_url_is_trusted_sportsbet(source_url):
        reasons.append("sportsbet_source_url_untrusted")
    elif source_url_looks_post_race(source_url):
        reasons.append("sportsbet_source_url_post_race")

    capture_dt = parse_timestamp(fixture.get("capture_timestamp"))
    jump_dt = parse_timestamp(fixture.get("jump_datetime"))
    if capture_dt is None:
        reasons.append("capture_timestamp_missing_or_invalid")
    elif not has_timezone(capture_dt):
        reasons.append("capture_timestamp_not_timezone_aware")
    if jump_dt is None:
        reasons.append("jump_datetime_missing_or_invalid")
    elif not has_timezone(jump_dt):
        reasons.append("jump_datetime_not_timezone_aware")
    if capture_dt is not None and jump_dt is not None:
        if has_timezone(capture_dt) and has_timezone(jump_dt):
            comparable_jump = jump_dt.astimezone(capture_dt.tzinfo)
            if capture_dt >= comparable_jump:
                reasons.append("capture_timestamp_not_before_jump")
        elif not has_timezone(capture_dt) and not has_timezone(jump_dt):
            if capture_dt >= jump_dt:
                reasons.append("capture_timestamp_not_before_jump")

    expected_container = fixture.get("expected_runners") or []
    runner_container = fixture.get("runner_rows") or []
    if not isinstance(expected_container, list) or any(
        not isinstance(row, Mapping) for row in expected_container
    ):
        reasons.append("expected_runner_records_malformed")
        expected_container = []
    if not isinstance(runner_container, list) or any(
        not isinstance(row, Mapping) for row in runner_container
    ):
        reasons.append("runner_records_malformed")
        runner_container = []
    expected = list(expected_container)
    runner_rows = list(runner_container)
    expected_by_box: dict[int, str] = {}
    expected_boxes_in_order: list[int] = []
    expected_identities_in_order: list[str] = []
    for row in expected:
        runner = normalize_runner(row.get("dog_name"), row.get("box_number"))
        if runner:
            expected_boxes_in_order.append(runner["box_number"])
            expected_identities_in_order.append(runner["identity"])
            expected_by_box.setdefault(runner["box_number"], runner["identity"])
    if not expected_by_box:
        reasons.append("expected_runners_missing")
    duplicate_expected_boxes = duplicate_values(expected_boxes_in_order)
    duplicate_expected_identities = duplicate_values(expected_identities_in_order)
    if duplicate_expected_boxes:
        reasons.append("duplicate_expected_runner_boxes")
    if duplicate_expected_identities:
        reasons.append("duplicate_expected_runner_identities")
    if not runner_rows:
        raw_fetch = fixture.get("raw_fetch_result")
        if isinstance(raw_fetch, Mapping) and fetch_result_count_only_without_raw(
            raw_fetch
        ):
            reasons.append("count_only_capture_no_raw_runner_odds")
        else:
            reasons.append("raw_runner_rows_missing")

    expected_boxes = set(expected_by_box)
    actual_boxes = {safe_int(row.get("box_number")) for row in runner_rows}
    actual_boxes.discard(None)
    missing_boxes = sorted(expected_boxes - actual_boxes)
    extra_boxes = sorted(actual_boxes - expected_boxes)
    if missing_boxes:
        reasons.append("missing_expected_runner_boxes")
    if extra_boxes:
        reasons.append("extra_unexpected_runner_boxes")

    duplicate_boxes = duplicate_values(
        [safe_int(row.get("box_number")) for row in runner_rows]
    )
    duplicate_identities = duplicate_values(
        [row.get("identity") for row in runner_rows]
    )
    if duplicate_boxes:
        reasons.append("duplicate_runner_boxes")
    if duplicate_identities:
        reasons.append("duplicate_runner_identities")

    for index, row in enumerate(runner_rows):
        prefix = f"runner_{index}"
        box = safe_int(row.get("box_number"))
        identity = str(row.get("identity") or "")
        if box is None:
            reasons.append(f"{prefix}_box_missing")
        elif box in expected_by_box and expected_by_box[box] != identity:
            reasons.append(f"{prefix}_identity_mismatch")
        if not str(row.get("raw_runner_text") or "").strip():
            reasons.append(f"{prefix}_raw_runner_text_missing")
        active = row.get("active")
        if not isinstance(active, bool):
            reasons.append(f"{prefix}_active_state_invalid")
        odds_decimal = safe_float(row.get("odds_decimal"))
        if active is True and (odds_decimal is None or odds_decimal <= 1.0):
            reasons.append(f"{prefix}_invalid_odds_decimal")
        if active is False and odds_decimal is not None:
            reasons.append(f"{prefix}_scratched_runner_has_price")
        if row.get("match_status") != "box_name_exact":
            reasons.append(f"{prefix}_match_status_not_exact")

    if projection is not None:
        if projection.get("schema_version") != NORMALIZED_PROJECTION_SCHEMA:
            reasons.append("projection_schema_version_invalid")
        if projection.get("fixture_id") != fixture.get("fixture_id"):
            reasons.append("projection_fixture_id_mismatch")
        if projection.get("market_type") != "win":
            reasons.append("projection_market_type_not_win")
        if projection.get("append_allowed_without_owner_approval") is not False:
            reasons.append("projection_owner_approval_flag_invalid")
        projection_container = projection.get("rows") or []
        if not isinstance(projection_container, list) or any(
            not isinstance(row, Mapping) for row in projection_container
        ):
            reasons.append("projection_runner_records_malformed")
            projection_container = []
        projection_rows = list(projection_container)
        if len(projection_rows) != len(runner_rows):
            reasons.append("projection_runner_row_count_mismatch")
        expected_projection = normalized_projection_from_fixture(fixture)
        if canonical_bytes(projection) != canonical_bytes(expected_projection):
            reasons.append("projection_content_mismatch")

    raw_fetch = fixture.get("raw_fetch_result")
    if not isinstance(raw_fetch, Mapping):
        reasons.append("raw_fetch_result_missing_or_invalid")
    else:
        if raw_fetch.get("success") is not True:
            reasons.append("source_fetch_not_successful")
        reasons.extend(metadata_mismatch_reasons(fixture, raw_fetch, prefix="source"))
        reasons.extend(roster_mismatch_reasons(fixture, raw_fetch, prefix="source"))
        raw_capture = parse_timestamp(
            nested_metadata_value(raw_fetch, ("capture_timestamp", "timestamp"))
        )
        if (
            raw_capture is None
            or capture_dt is None
            or not has_timezone(raw_capture)
            or not has_timezone(capture_dt)
            or raw_capture != capture_dt.astimezone(raw_capture.tzinfo)
        ):
            reasons.append("source_capture_timestamp_mismatch")

    sidecar_reasons, sidecar_payload = validate_source_sidecar(
        fixture.get("provenance"), sidecar_source=sidecar_source
    )
    reasons.extend(sidecar_reasons)
    if sidecar_payload is not None:
        try:
            assert_outcome_free(sidecar_payload, context="source_sidecar")
        except ValueError as exc:
            reasons.append(str(exc))
        reasons.extend(
            metadata_mismatch_reasons(fixture, sidecar_payload, prefix="sidecar")
        )
        reasons.extend(
            roster_mismatch_reasons(fixture, sidecar_payload, prefix="sidecar")
        )

    return {
        "schema_version": VALIDATION_SCHEMA,
        "status": "PASS" if not reasons else "BLOCKED",
        "reasons": sorted(set(reasons)),
        "race_id": fixture.get("race_id"),
        "fixture_id": fixture.get("fixture_id"),
        "market_type": fixture.get("market_type"),
        "expected_runner_count": len(expected_by_box),
        "runner_row_count": len(runner_rows),
        "missing_expected_boxes": missing_boxes,
        "extra_unexpected_boxes": extra_boxes,
        "duplicate_boxes": duplicate_boxes,
        "duplicate_identities": duplicate_identities,
        "duplicate_expected_boxes": duplicate_expected_boxes,
        "duplicate_expected_identities": duplicate_expected_identities,
        "db_append_eligible": False,
        "owner_approval_required_before_append": True,
    }


def fetch_records_from_payload(payload: Any) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    errors: list[str] = []
    if isinstance(payload, list):
        for index, row in enumerate(payload):
            if isinstance(row, Mapping):
                records.append(row)
            else:
                errors.append(f"fetch_record_malformed:list:{index}")
    elif not isinstance(payload, Mapping):
        raise ValueError("fetch_payload_must_be_object_or_list")
    else:
        direct_candidate = bool(
            raw_win_rows(payload)
            or fetch_result_count_only_without_raw(payload)
            or fetch_record_keys(payload)
        )
        declared_containers = [
            key
            for key in ("fetch_results", "items", "attempts")
            if isinstance(payload.get(key), list)
        ]
        if direct_candidate and declared_containers:
            raise ValueError("fetch_payload_competing_direct_and_record_container")
        if direct_candidate:
            records.append(payload)
        else:
            records.extend(_fetch_records_from_containers(payload, errors))
    if not records:
        errors.append("fetch_records_missing")
    for index, record in enumerate(records):
        if not fetch_record_keys(record):
            errors.append(f"fetch_record_identity_missing:{index}")
        runner_container = raw_win_row_container(record)
        if runner_container is not None:
            for runner_index, runner in enumerate(runner_container):
                if not isinstance(runner, Mapping):
                    errors.append(
                        f"fetch_runner_record_malformed:{index}:{runner_index}"
                    )
    if errors:
        raise ValueError(";".join(sorted(set(errors))))
    return records


def _fetch_records_from_containers(
    payload: Mapping[str, Any], errors: list[str]
) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    containers: list[tuple[str, list[Any]]] = []
    for key in ("fetch_results", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            containers.append((key, value))
    attempts = payload.get("attempts")
    if isinstance(attempts, list):
        extracted_attempts: list[Any] = []
        for index, attempt in enumerate(attempts):
            if not isinstance(attempt, Mapping):
                errors.append(f"fetch_record_malformed:attempts:{index}")
                continue
            candidates = [
                attempt[key]
                for key in ("fetch_result", "raw_fetch_result")
                if key in attempt
            ]
            if len(candidates) != 1 or not isinstance(candidates[0], Mapping):
                errors.append(f"fetch_record_malformed:attempts:{index}")
                continue
            extracted_attempts.append(candidates[0])
        containers.append(("attempts", extracted_attempts))
    populated = [(key, value) for key, value in containers if value]
    if len(populated) > 1:
        errors.append("fetch_payload_competing_record_containers")
    for key, values in populated:
        for index, row in enumerate(values):
            if isinstance(row, Mapping):
                records.append(row)
            else:
                errors.append(f"fetch_record_malformed:{key}:{index}")
    return records


def assert_fetch_records_outcome_free(payload: Any) -> None:
    try:
        assert_outcome_free(payload, context="fetch_payload")
    except ValueError as exc:
        raise ValueError("raw_fetch_result_contains_outcome_fields") from exc


def fetch_record_keys(record: Mapping[str, Any]) -> set[str]:
    keys = {
        str(record.get(key) or "").strip()
        for key in ("canonical_race_identity", "alias_race_id", "race_id")
    }
    race_info = record.get("race_info")
    if isinstance(race_info, Mapping):
        keys.update(
            str(race_info.get(key) or "").strip()
            for key in ("canonical_race_identity", "alias_race_id", "race_id")
        )
    return {key for key in keys if key}


def index_fetch_records(payload: Any) -> dict[str, list[Mapping[str, Any]]]:
    index: dict[str, list[Mapping[str, Any]]] = {}
    records = fetch_records_from_payload(payload)
    for record in records:
        for key in fetch_record_keys(record):
            bucket = index.setdefault(key, [])
            if all(existing is not record for existing in bucket):
                bucket.append(record)
    duplicate_keys = sorted(key for key, rows in index.items() if len(rows) > 1)
    if duplicate_keys:
        raise ValueError(
            "duplicate_fetch_record_identities:" + ",".join(duplicate_keys)
        )
    return index


def matching_fetch_result(
    *,
    plan_item: Mapping[str, Any],
    fetch_index: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Mapping[str, Any] | None:
    matches: list[Mapping[str, Any]] = []
    for key in (
        plan_item.get("canonical_race_identity"),
        plan_item.get("race_id"),
    ):
        text = str(key or "").strip()
        if text and text in fetch_index:
            for record in fetch_index[text]:
                if all(existing is not record for existing in matches):
                    matches.append(record)
    if len(matches) > 1:
        raise ValueError(
            "competing_fetch_records_for_ready_race:"
            + str(plan_item.get("canonical_race_identity") or "")
        )
    return matches[0] if matches else None


def ready_plan_items(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    items = plan.get("items") or []
    if not isinstance(items, list):
        raise ValueError("plan_items_must_be_list")
    ready: list[Mapping[str, Any]] = []
    identities: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"plan_item_malformed:{index}")
        if item.get("status") != "READY_TO_CAPTURE":
            continue
        identity = str(item.get("canonical_race_identity") or "").strip()
        if not identity:
            raise ValueError(f"ready_plan_item_identity_missing:{index}")
        identities.append(identity)
        expected_runners = item.get("expected_runners")
        if not isinstance(expected_runners, list) or any(
            not isinstance(runner, Mapping) for runner in expected_runners
        ):
            raise ValueError(f"ready_plan_item_expected_runners_malformed:{index}")
        ready.append(item)
    duplicates = duplicate_values(identities)
    if duplicates:
        raise ValueError("duplicate_ready_plan_race_identities:" + ",".join(duplicates))
    return ready


def manifest_file_entry(
    *,
    packet_dir: Path,
    path: Path,
    role: str,
    payload_bytes: bytes,
    fixture_id: str | None = None,
) -> dict[str, Any]:
    return {
        "path": path.relative_to(packet_dir).as_posix(),
        "role": role,
        "fixture_id": fixture_id,
        "sha256": sha256_bytes(payload_bytes),
        "bytes": len(payload_bytes),
    }


def packet_status_from_validations(validations: Sequence[Mapping[str, Any]]) -> str:
    if not validations:
        return FINAL_NO_READY_RACES
    return (
        FINAL_SEALED_NO_DB_APPEND
        if all(row.get("status") == "PASS" for row in validations)
        else FINAL_BLOCKED_VALIDATION_FAILED
    )


def build_fixture_packet(
    *,
    plan: Mapping[str, Any],
    fetch_payload: Any,
    output_dir: Path,
    current_time: datetime,
    plan_path: Path | None = None,
    fetch_result_path: Path | None = None,
    plan_bytes: bytes | None = None,
    fetch_result_bytes: bytes | None = None,
) -> dict[str, Any]:
    if plan_path is not None:
        plan_bytes = (
            plan_bytes if plan_bytes is not None else read_file_bytes(plan_path)
        )
        parsed_plan = parse_json_bytes(plan_bytes, context="plan")
        if not isinstance(parsed_plan, Mapping):
            raise ValueError("plan_must_be_json_object")
        plan = parsed_plan
    if fetch_result_path is not None:
        fetch_result_bytes = (
            fetch_result_bytes
            if fetch_result_bytes is not None
            else read_file_bytes(fetch_result_path)
        )
        fetch_payload = parse_json_bytes(fetch_result_bytes, context="fetch_result")

    assert_outcome_free(plan, context="plan")
    assert_fetch_records_outcome_free(fetch_payload)
    ready_items = ready_plan_items(plan)
    fetch_index = index_fetch_records(fetch_payload)
    prepared: list[
        tuple[Mapping[str, Any], Mapping[str, Any] | None, Mapping[str, Any]]
    ] = []
    collector_bytes = read_file_bytes(Path(__file__).resolve())
    selected_fetch_races: dict[int, str] = {}
    for item in ready_items:
        sidecar_source = fixture_source_sidecar(item)
        sidecar_payload = sidecar_source.get("payload")
        if isinstance(sidecar_payload, Mapping):
            assert_outcome_free(sidecar_payload, context="source_sidecar")
        fetch_result = matching_fetch_result(plan_item=item, fetch_index=fetch_index)
        if fetch_result is not None:
            race_id = str(item.get("canonical_race_identity") or "")
            prior_race_id = selected_fetch_races.get(id(fetch_result))
            if prior_race_id is not None:
                competing = ",".join(sorted((prior_race_id, race_id)))
                raise ValueError(
                    f"fetch_record_matches_multiple_ready_races:{competing}"
                )
            selected_fetch_races[id(fetch_result)] = race_id
        prepared.append((item, fetch_result, sidecar_source))

    output_dir = assert_output_dir_safe(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output_dir_already_exists:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    validations: list[dict[str, Any]] = []
    fixture_results: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []

    for item, fetch_result, sidecar_source in prepared:
        race_id = str(item.get("canonical_race_identity") or "")
        if fetch_result is None:
            validation = {
                "schema_version": VALIDATION_SCHEMA,
                "status": "BLOCKED",
                "reasons": ["raw_fetch_result_missing_for_ready_race"],
                "race_id": race_id,
                "fixture_id": None,
                "market_type": "win",
                "expected_runner_count": len(expected_runner_rows(item)),
                "runner_row_count": 0,
                "db_append_eligible": False,
                "owner_approval_required_before_append": True,
            }
            validations.append(validation)
            fixture_results.append(
                {
                    "race_id": race_id,
                    "status": validation["status"],
                    "reasons": validation["reasons"],
                    "fixture_path": None,
                    "projection_path": None,
                }
            )
            continue

        fixture = build_raw_fixture(
            plan_item=item,
            fetch_result=fetch_result,
            sidecar_source=sidecar_source,
            collector_bytes=collector_bytes,
        )
        projection = normalized_projection_from_fixture(fixture)
        validation = validate_fixture_payload(
            fixture, projection, sidecar_source=sidecar_source
        )
        validations.append(validation)

        base_name = (
            f"{safe_filename(fixture.get('race_id'))}_{fixture['fixture_id'][:12]}"
        )
        fixture_path = output_dir / "fixtures" / f"{base_name}_raw_fixture.json"
        projection_path = output_dir / "normalized" / f"{base_name}_projection.json"
        fixture_bytes = write_json_new(fixture_path, fixture)
        projection_bytes = write_json_new(projection_path, projection)
        files.append(
            manifest_file_entry(
                packet_dir=output_dir,
                path=fixture_path,
                role="raw_fixture",
                payload_bytes=fixture_bytes,
                fixture_id=str(fixture.get("fixture_id") or ""),
            )
        )
        files.append(
            manifest_file_entry(
                packet_dir=output_dir,
                path=projection_path,
                role="normalized_projection",
                payload_bytes=projection_bytes,
                fixture_id=str(fixture.get("fixture_id") or ""),
            )
        )
        fixture_results.append(
            {
                "race_id": fixture.get("race_id"),
                "fixture_id": fixture.get("fixture_id"),
                "status": validation["status"],
                "reasons": validation["reasons"],
                "fixture_path": fixture_path.relative_to(output_dir).as_posix(),
                "projection_path": projection_path.relative_to(output_dir).as_posix(),
            }
        )

    preseal_validation = {
        "schema_version": PRESEAL_VALIDATION_SCHEMA,
        "generated_at": current_time.isoformat(),
        "status": packet_status_from_validations(validations),
        "validations": validations,
    }
    assert_outcome_free(preseal_validation, context="preseal_validation")
    validation_path = output_dir / "strict_win_fixture_validation_preseal.json"
    validation_bytes = write_json_new(validation_path, preseal_validation)
    files.append(
        manifest_file_entry(
            packet_dir=output_dir,
            path=validation_path,
            role="preseal_validation",
            payload_bytes=validation_bytes,
        )
    )

    status = str(preseal_validation["status"])
    report = {
        "schema_version": PACKET_REPORT_SCHEMA,
        "generated_at": current_time.isoformat(),
        "status": status,
        "plan_path": relpath(plan_path) if plan_path else None,
        "fetch_result_path": relpath(fetch_result_path) if fetch_result_path else None,
        "candidate_race_count": plan.get("candidate_race_count"),
        "ready_plan_item_count": len(ready_items),
        "fixture_count": sum(1 for row in fixture_results if row.get("fixture_id")),
        "validation_pass_count": sum(
            1 for row in validations if row.get("status") == "PASS"
        ),
        "validation_blocked_count": sum(
            1 for row in validations if row.get("status") != "PASS"
        ),
        "fixture_results": fixture_results,
        "db_append_performed": False,
        "db_append_approved": False,
        "owner_approval_required_before_append": True,
        "denominator_floor_complete_valid_prejump_market_races": 300,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }
    final_status = {"status": status}
    assert_outcome_free(report, context="packet_report")
    assert_outcome_free(final_status, context="final_status")
    report_path = output_dir / PACKET_REPORT_PATH
    final_status_path = output_dir / FINAL_STATUS_PATH
    report_bytes = write_json_new(report_path, report)
    final_status_bytes = write_json_new(final_status_path, final_status)
    files.extend(
        [
            manifest_file_entry(
                packet_dir=output_dir,
                path=report_path,
                role="packet_report",
                payload_bytes=report_bytes,
            ),
            manifest_file_entry(
                packet_dir=output_dir,
                path=final_status_path,
                role="final_status",
                payload_bytes=final_status_bytes,
            ),
        ]
    )

    manifest_base = {
        "schema_version": MANIFEST_SCHEMA,
        "generated_at": current_time.isoformat(),
        "prior_states": {
            "denominator": "DATA_LINEAGE_BLOCKER_STOP",
            "odds_provenance_audit": "ODDS_CAPTURE_PROVENANCE_AUDIT_DONE",
            "future_plan": FINAL_PLAN_ONLY_DONE,
        },
        "status": preseal_validation["status"],
        "plan_path": relpath(plan_path) if plan_path else None,
        "plan_sha256": sha256_bytes(plan_bytes) if plan_bytes is not None else None,
        "fetch_result_path": relpath(fetch_result_path) if fetch_result_path else None,
        "fetch_result_sha256": (
            sha256_bytes(fetch_result_bytes) if fetch_result_bytes is not None else None
        ),
        "ready_plan_item_count": len(ready_items),
        "fixture_count": sum(1 for row in fixture_results if row.get("fixture_id")),
        "files": files,
        "manifest_role": {
            "path": MANIFEST_PATH,
            "role": "manifest",
            "self_hash": None,
        },
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }
    manifest = {**manifest_base, "manifest_sha256": sha256_payload(manifest_base)}
    assert_outcome_free(manifest, context="manifest")
    manifest_path = output_dir / MANIFEST_PATH
    manifest_bytes = write_json_new(manifest_path, manifest)

    packet_validation = validate_packet(output_dir)
    report["manifest_path"] = relpath(manifest_path)
    report["manifest_sha256"] = sha256_bytes(manifest_bytes)
    report["packet_validation"] = packet_validation
    if packet_validation["status"] != "PASS":
        report["status"] = packet_validation["status"]
    return report


def validate_manifest_integrity(manifest: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    try:
        assert_outcome_free(manifest, context="manifest")
    except ValueError as exc:
        reasons.append(str(exc))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        reasons.append("manifest_schema_version_invalid")
    expected = manifest.get("manifest_sha256")
    manifest_base = dict(manifest)
    manifest_base.pop("manifest_sha256", None)
    if expected != sha256_payload(manifest_base):
        reasons.append("manifest_sha256_mismatch")
    if manifest.get("manifest_role") != {
        "path": MANIFEST_PATH,
        "role": "manifest",
        "self_hash": None,
    }:
        reasons.append("manifest_role_invalid")
    return reasons


def normalized_packet_relative_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        return None
    normalized = pure.as_posix()
    return normalized if normalized == value else None


def resolve_packet_file(packet_dir: Path, value: Any) -> Path | None:
    normalized = normalized_packet_relative_path(value)
    if normalized is None:
        return None
    try:
        packet_root = packet_dir.resolve(strict=True)
        logical = packet_root / normalized
        if logical.is_symlink():
            return None
        resolved = logical.resolve(strict=False)
        resolved.relative_to(packet_root)
    except (FileNotFoundError, RuntimeError, ValueError):
        return None
    return resolved


def validate_packet(packet_dir: Path) -> dict[str, Any]:
    manifest_path = resolve_packet_file(packet_dir, MANIFEST_PATH)
    reasons: list[str] = []
    fixture_validations: list[dict[str, Any]] = []
    if manifest_path is None:
        return {
            "schema_version": PACKET_VALIDATOR_SCHEMA,
            "status": FINAL_BLOCKED_VALIDATION_FAILED,
            "reasons": ["manifest_path_outside_packet"],
            "fixture_validations": [],
        }
    if not manifest_path.exists():
        return {
            "schema_version": PACKET_VALIDATOR_SCHEMA,
            "status": FINAL_BLOCKED_VALIDATION_FAILED,
            "reasons": ["manifest_missing"],
            "fixture_validations": [],
        }
    try:
        manifest_bytes = read_file_bytes(manifest_path)
        manifest = parse_json_bytes(manifest_bytes, context="manifest")
    except (OSError, ValueError):
        return {
            "schema_version": PACKET_VALIDATOR_SCHEMA,
            "status": FINAL_BLOCKED_VALIDATION_FAILED,
            "reasons": ["manifest_json_invalid"],
            "fixture_validations": [],
        }
    if not isinstance(manifest, Mapping):
        return {
            "schema_version": PACKET_VALIDATOR_SCHEMA,
            "status": FINAL_BLOCKED_VALIDATION_FAILED,
            "reasons": ["manifest_not_object"],
            "fixture_validations": [],
        }
    reasons.extend(validate_manifest_integrity(manifest))
    packet_root = packet_dir.resolve(strict=True)

    declared_entries = manifest.get("files")
    if not isinstance(declared_entries, list):
        declared_entries = []
        reasons.append("manifest_files_not_list")
    declared_paths: set[str] = set()
    role_keys: set[tuple[str, str]] = set()
    file_payloads: dict[str, Mapping[str, Any]] = {}
    file_bytes_by_path: dict[str, bytes] = {}
    entries_by_path: dict[str, Mapping[str, Any]] = {}
    actual_files: set[str] = set()
    for path in sorted(packet_root.rglob("*")):
        rel = path.relative_to(packet_root).as_posix()
        try:
            mode = path.lstat().st_mode
        except OSError:
            reasons.append(f"packet_entry_unreadable:{rel}")
            continue
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            reasons.append(f"packet_non_regular_entry:{rel}")
            continue
        actual_files.add(rel)

    for entry in declared_entries:
        if not isinstance(entry, Mapping):
            reasons.append("manifest_file_entry_invalid")
            continue
        rel = normalized_packet_relative_path(entry.get("path"))
        raw_rel = entry.get("path")
        if rel is None:
            reasons.append(f"manifest_file_path_outside_packet:{raw_rel}")
            continue
        if rel in declared_paths:
            reasons.append(f"manifest_file_path_duplicate:{rel}")
            if entry.get("role") == "normalized_projection":
                reasons.append("manifest_projection_fixture_id_duplicate")
            if entry.get("role") == "raw_fixture":
                reasons.append("manifest_raw_fixture_id_duplicate")
            continue
        declared_paths.add(rel)
        entries_by_path[rel] = entry
        role = entry.get("role")
        fixture_id = entry.get("fixture_id")
        role_fixture = str(fixture_id or "")
        if not isinstance(role, str):
            reasons.append(f"manifest_file_role_invalid:{role}")
            continue
        role_key = (role, role_fixture)
        if role_key in role_keys:
            reasons.append(f"manifest_role_conflict:{role}:{role_fixture}")
        role_keys.add(role_key)
        if role in SINGLETON_MANIFEST_ROLES:
            if rel != SINGLETON_MANIFEST_ROLES[role] or fixture_id not in (None, ""):
                reasons.append(f"manifest_role_path_invalid:{role}:{rel}")
        elif role == "raw_fixture":
            if (
                not isinstance(fixture_id, str)
                or not fixture_id
                or not rel.startswith("fixtures/")
                or not rel.endswith("_raw_fixture.json")
            ):
                reasons.append(f"manifest_role_path_invalid:{role}:{rel}")
        elif role == "normalized_projection":
            if (
                not isinstance(fixture_id, str)
                or not fixture_id
                or not rel.startswith("normalized/")
                or not rel.endswith("_projection.json")
            ):
                reasons.append(f"manifest_role_path_invalid:{role}:{rel}")
        else:
            reasons.append(f"manifest_file_role_invalid:{role}")
        file_path = resolve_packet_file(packet_root, rel)
        if file_path is None:
            reasons.append(f"manifest_file_path_outside_packet:{rel}")
            continue
        try:
            mode = file_path.lstat().st_mode
        except OSError:
            reasons.append(f"manifest_file_missing:{rel}")
            continue
        if not stat.S_ISREG(mode):
            reasons.append(f"manifest_file_not_regular:{rel}")
            continue
        try:
            payload_bytes = read_file_bytes(file_path)
        except OSError:
            reasons.append(f"manifest_file_unreadable:{rel}")
            continue
        file_bytes_by_path[rel] = payload_bytes
        if sha256_bytes(payload_bytes) != entry.get("sha256"):
            reasons.append(f"manifest_file_sha256_mismatch:{rel}")
        declared_bytes = entry.get("bytes")
        if type(declared_bytes) is not int or declared_bytes != len(payload_bytes):
            reasons.append(f"manifest_file_size_mismatch:{rel}")
        try:
            payload = parse_json_bytes(payload_bytes, context=f"manifest_file:{rel}")
        except ValueError:
            reasons.append(f"manifest_file_json_invalid:{rel}")
            continue
        if not isinstance(payload, Mapping):
            reasons.append(f"manifest_file_json_not_object:{rel}")
            continue
        file_payloads[rel] = payload
        try:
            assert_outcome_free(payload, context=f"manifest_file:{rel}")
        except ValueError as exc:
            reasons.append(str(exc))

    expected_actual_files = declared_paths | {MANIFEST_PATH}
    for undeclared in sorted(actual_files - expected_actual_files):
        reasons.append(f"packet_file_undeclared:{undeclared}")
    for missing in sorted(expected_actual_files - actual_files):
        reasons.append(f"packet_file_missing:{missing}")

    source_payloads: dict[str, Any] = {}
    for field, hash_field, context in (
        ("plan_path", "plan_sha256", "plan"),
        ("fetch_result_path", "fetch_result_sha256", "fetch_result"),
    ):
        source_path_value = manifest.get(field)
        expected_hash = manifest.get(hash_field)
        if source_path_value is None and expected_hash is None:
            continue
        source_path = resolve_repo_file(source_path_value)
        if source_path is None:
            reasons.append(f"{context}_source_missing_or_untrusted")
            continue
        try:
            source_bytes = read_file_bytes(source_path)
            source_payload = parse_json_bytes(source_bytes, context=context)
        except (OSError, ValueError):
            reasons.append(f"{context}_source_json_invalid")
            continue
        if sha256_bytes(source_bytes) != expected_hash:
            reasons.append(f"{context}_source_sha256_mismatch")
        if context == "plan" and not isinstance(source_payload, Mapping):
            reasons.append("plan_source_not_object")
        source_payloads[context] = source_payload
        try:
            assert_outcome_free(source_payload, context=context)
        except ValueError as exc:
            reasons.append(str(exc))

    projections_by_fixture: dict[str, Mapping[str, Any]] = {}
    fixture_files: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    raw_fixture_ids: set[str] = set()
    preseal_entry_count = 0
    sidecar_cache: dict[str, Mapping[str, Any]] = {}
    replay_plan_by_race: dict[str, Mapping[str, Any]] = {}
    replay_fetch_index: dict[str, list[Mapping[str, Any]]] = {}
    replay_plan = source_payloads.get("plan")
    if isinstance(replay_plan, Mapping):
        try:
            replay_plan_by_race = {
                str(item.get("canonical_race_identity")): item
                for item in ready_plan_items(replay_plan)
            }
        except ValueError as exc:
            reasons.append(f"replay_plan_invalid:{exc}")
    replay_fetch = source_payloads.get("fetch_result")
    if replay_fetch is not None:
        try:
            replay_fetch_index = index_fetch_records(replay_fetch)
        except ValueError as exc:
            reasons.append(f"replay_fetch_invalid:{exc}")
    for rel, entry in entries_by_path.items():
        payload = file_payloads.get(rel)
        if payload is None:
            continue
        role = entry.get("role")
        if role == "raw_fixture":
            expected_path = (
                "fixtures/"
                f"{safe_filename(payload.get('race_id'))}_"
                f"{str(payload.get('fixture_id') or '')[:12]}_raw_fixture.json"
            )
            if rel != expected_path:
                reasons.append(f"manifest_role_path_invalid:{role}:{rel}")
            fixture_files.append((rel, entry, payload))
        elif role == "normalized_projection":
            expected_path = (
                "normalized/"
                f"{safe_filename(payload.get('race_id'))}_"
                f"{str(payload.get('fixture_id') or '')[:12]}_projection.json"
            )
            if rel != expected_path:
                reasons.append(f"manifest_role_path_invalid:{role}:{rel}")
            fixture_id = str(payload.get("fixture_id") or "")
            if entry.get("fixture_id") != fixture_id:
                reasons.append("manifest_projection_fixture_id_mismatch")
            if fixture_id in projections_by_fixture:
                reasons.append("manifest_projection_fixture_id_duplicate")
            else:
                projections_by_fixture[fixture_id] = payload
        elif role == "preseal_validation":
            preseal_entry_count += 1

    for fixture_rel, fixture_entry, fixture in fixture_files:
        fixture_id = str(fixture.get("fixture_id") or "")
        if fixture_entry.get("fixture_id") != fixture_id:
            reasons.append("manifest_raw_fixture_id_mismatch")
        if fixture_id in raw_fixture_ids:
            reasons.append("manifest_raw_fixture_id_duplicate")
        raw_fixture_ids.add(fixture_id)
        projection = projections_by_fixture.get(fixture_id)
        provenance = fixture.get("provenance")
        sidecar_source: Mapping[str, Any] | None = None
        if isinstance(provenance, Mapping):
            sidecar_path = provenance.get("source_sidecar_path")
            if isinstance(sidecar_path, str):
                if sidecar_path not in sidecar_cache:
                    sidecar_cache[sidecar_path] = fixture_source_sidecar(
                        {"sidecar_path": sidecar_path}
                    )
                sidecar_source = sidecar_cache[sidecar_path]
        validation = validate_fixture_payload(
            fixture, projection, sidecar_source=sidecar_source
        )
        race_id = str(fixture.get("race_id") or "")
        if replay_plan_by_race:
            replay_plan_item = replay_plan_by_race.get(race_id)
            if replay_plan_item is None:
                validation["reasons"] = sorted(
                    set(
                        list(validation.get("reasons") or [])
                        + ["replay_plan_item_missing"]
                    )
                )
                validation["status"] = "BLOCKED"
            else:
                replay_reasons = metadata_mismatch_reasons(
                    fixture, replay_plan_item, prefix="plan"
                ) + roster_mismatch_reasons(fixture, replay_plan_item, prefix="plan")
                if replay_reasons:
                    validation["reasons"] = sorted(
                        set(list(validation.get("reasons") or []) + replay_reasons)
                    )
                    validation["status"] = "BLOCKED"
        if replay_fetch_index:
            try:
                source_record = matching_fetch_result(
                    plan_item={
                        "canonical_race_identity": race_id,
                        "race_id": race_id,
                    },
                    fetch_index=replay_fetch_index,
                )
            except ValueError as exc:
                source_record = None
                reasons.append(f"replay_fetch_invalid:{exc}")
            if source_record is None:
                validation["reasons"] = sorted(
                    set(
                        list(validation.get("reasons") or [])
                        + ["replay_fetch_record_missing"]
                    )
                )
                validation["status"] = "BLOCKED"
            elif canonical_bytes(source_record) != canonical_bytes(
                fixture.get("raw_fetch_result")
            ):
                validation["reasons"] = sorted(
                    set(
                        list(validation.get("reasons") or [])
                        + ["replay_fetch_fixture_content_mismatch"]
                    )
                )
                validation["status"] = "BLOCKED"
        validation["fixture_path"] = fixture_rel
        if projection is None:
            validation["status"] = "BLOCKED"
            validation["reasons"] = sorted(
                set(list(validation.get("reasons") or []) + ["projection_missing"])
            )
        fixture_validations.append(validation)

    if any(row.get("status") != "PASS" for row in fixture_validations):
        reasons.append("fixture_payload_validation_failed")
    declared_fixture_count = manifest.get("fixture_count")
    if (
        type(declared_fixture_count) is not int
        or declared_fixture_count < 0
        or declared_fixture_count != len(fixture_files)
        or declared_fixture_count != len(projections_by_fixture)
    ):
        reasons.append("manifest_fixture_count_mismatch")
    if raw_fixture_ids != set(projections_by_fixture):
        reasons.append("manifest_fixture_id_set_mismatch")
    if preseal_entry_count != 1:
        reasons.append("manifest_preseal_validation_entry_count_invalid")
    for role in ("packet_report", "final_status"):
        role_count = sum(
            1
            for entry in declared_entries
            if isinstance(entry, Mapping) and entry.get("role") == role
        )
        if role_count != 1:
            reasons.append(f"manifest_{role}_entry_count_invalid")

    preseal_payload = file_payloads.get(PRESEAL_PATH)
    validations_without_paths = []
    for validation in fixture_validations:
        normalized_validation = dict(validation)
        normalized_validation.pop("fixture_path", None)
        validations_without_paths.append(normalized_validation)
    expected_preseal = {
        "schema_version": PRESEAL_VALIDATION_SCHEMA,
        "generated_at": manifest.get("generated_at"),
        "status": packet_status_from_validations(validations_without_paths),
        "validations": validations_without_paths,
    }
    if not isinstance(preseal_payload, Mapping) or canonical_bytes(
        preseal_payload
    ) != canonical_bytes(expected_preseal):
        reasons.append("preseal_validation_content_mismatch")
    if manifest.get("status") != expected_preseal["status"]:
        reasons.append("manifest_status_mismatch")
    final_status_payload = file_payloads.get(FINAL_STATUS_PATH)
    if final_status_payload != {"status": manifest.get("status")}:
        reasons.append("final_status_content_mismatch")

    return {
        "schema_version": PACKET_VALIDATOR_SCHEMA,
        "status": "PASS" if not reasons else FINAL_BLOCKED_VALIDATION_FAILED,
        "reasons": sorted(set(reasons)),
        "manifest_path": relpath(manifest_path),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "fixture_validations": fixture_validations,
        "db_append_performed": False,
        "owner_approval_required_before_append": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.validate_packet:
        return validate_packet(args.validate_packet)
    current_time = parse_current_time(args.current_time)
    plan_path = args.plan
    fetch_result_path = args.fetch_result
    output_dir = assert_output_dir_safe(
        args.output_dir
        or DEFAULT_OUTPUT_PARENT
        / f"strict_win_odds_fixture_capture_{now_id(current_time)}_report_only"
    )
    plan_bytes = read_file_bytes(plan_path)
    fetch_result_bytes = read_file_bytes(fetch_result_path)
    plan = parse_json_bytes(plan_bytes, context="plan")
    fetch_payload = parse_json_bytes(fetch_result_bytes, context="fetch_result")
    if not isinstance(plan, Mapping):
        raise ValueError("plan_must_be_json_object")
    return build_fixture_packet(
        plan=plan,
        fetch_payload=fetch_payload,
        output_dir=output_dir,
        current_time=current_time,
        plan_path=plan_path,
        fetch_result_path=fetch_result_path,
        plan_bytes=plan_bytes,
        fetch_result_bytes=fetch_result_bytes,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--fetch-result", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--current-time")
    parser.add_argument("--validate-packet", type=Path)
    args = parser.parse_args(argv)
    if args.validate_packet:
        return args
    missing = [
        name
        for name, value in (
            ("--plan", args.plan),
            ("--fetch-result", args.fetch_result),
        )
        if value is None
    ]
    if missing:
        parser.error("required unless --validate-packet is used: " + ", ".join(missing))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("status") in {"PASS", FINAL_SEALED_NO_DB_APPEND} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
