#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "beautifulsoup4==4.13.4",
#   "numpy==1.26.4",
#   "requests==2.32.4",
#   "selenium==4.34.2",
#   "webdriver-manager==4.0.2",
# ]
# ///
"""Build one isolated, research-only prediction for an exact pre-jump race."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import socket
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.refresh_prejump_upcoming import (  # noqa: E402
    _parse_race_jump_datetime,
    parse_current_time,
    stable_race_id,
    stable_race_id_variants,
    venue_exclusion_aliases,
)
from scripts.predict_market_form_residual import (  # noqa: E402
    DEFAULT_EVIDENCE_ROOT,
    DEFAULT_RETAINED_EVIDENCE_ROOTS,
    FEATURE_GENERATOR_FILES,
    ManualPredictionError as CaptureHandoffError,
    discover_race_artifacts,
    score_from_artifacts,
)
from src.predictor.on_demand import (  # noqa: E402
    Dependencies,
    PredictionBlocked,
    _copy_exact,
    _write_canonical,
    bundle_manifest,
    canonical_bytes,
    create_bundle,
    load_config,
    market_only_prediction,
    normalize_validation_receipt,
    receipt_from_handoff,
    resolve_model,
    seal_history_database,
    sha256_bytes,
    sha256_file,
    verify_bundle,
    write_exact_bytes,
)
from utils.csv_metadata import canonical_thedogs_race_identity  # noqa: E402


DEFAULT_DB = ROOT / "greyhound_racing_data.db"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/on_demand_prediction_runs"
DEFAULT_CAPTURE_EVIDENCE_ROOTS = (
    DEFAULT_EVIDENCE_ROOT,
    *DEFAULT_RETAINED_EVIDENCE_ROOTS,
)
DEFAULT_LOCK = (
    ROOT
    / "artifacts/full_evidence_orchestration_20260525/shadow_autopilot_daemon_runtime/shadow_autopilot.lock"
)
LOCK_RELATIVE_PATH = Path(
    "artifacts/full_evidence_orchestration_20260525/"
    "shadow_autopilot_daemon_runtime/shadow_autopilot.lock"
)
PRECURRENT_PACKET_REJECTION_REASONS = frozenset(
    {
        "sidecar_target_grade_context_schema_missing",
        "sidecar_target_grade_exact_value_missing",
        "sidecar_target_grade_equivalence_key_missing",
        "sidecar_target_grade_race_url_missing",
        "sidecar_target_grade_source_url_missing",
        "sidecar_target_grade_source_sha256_missing",
        "sidecar_target_grade_race_date_missing",
        "sidecar_target_grade_race_number_missing",
        "sidecar_target_grade_venue_missing",
        "feature_generator_implementation_hash_mismatch",
    }
)


class CollectorLockBusy(RuntimeError):
    """An existing collector lock is always busy; it is never reclaimed here."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        super().__init__("collector_lock_busy_no_steal")
        self.payload = dict(payload)


@dataclass(frozen=True)
class OwnedCollectorLock:
    path: Path
    run_id: str
    device: int
    inode: int


def _token(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())


def _authoritative_meeting_aliases(race: Mapping[str, Any]) -> set[str] | None:
    """Keep same-code Murray Bridge meetings distinct by their TheDogs slug."""

    identity = canonical_thedogs_race_identity(race.get("url") or race.get("race_url"))
    if identity is None or identity["venue_slug"] not in {
        "murray-bridge",
        "murray-bridge-straight",
    }:
        return None
    return {
        value
        for value in (
            race.get("venue"),
            race.get("venue_name"),
            identity["venue_slug"],
        )
        if value
    }


def _race_query_tokens(race: Mapping[str, Any]) -> set[str]:
    venue = race.get("venue") or race.get("venue_name")
    number = race.get("race_number")
    name = race.get("race_name") or race.get("name")
    values = {
        stable_race_id(race),
        name,
        f"{venue} race {number}",
        f"race {number} {venue}",
        f"{venue} r{number}",
    }
    authoritative_aliases = _authoritative_meeting_aliases(race)
    aliases = authoritative_aliases or venue_exclusion_aliases(
        venue, source_url=race.get("url") or race.get("race_url")
    )
    if authoritative_aliases is None:
        values.update(stable_race_id_variants(race))
    for alias in aliases:
        values.update(
            {
                f"{alias} race {number}",
                f"race {number} {alias}",
                f"{alias} r{number}",
                f"Race {number} - {alias} - {race.get('date') or race.get('race_date')}",
            }
        )
    return {_token(value) for value in values if value}


def resolve_target_race(
    races: Sequence[Mapping[str, Any]],
    *,
    race_id: str | None,
    race_query: str | None,
) -> tuple[str, Mapping[str, Any] | None, list[str]]:
    """Resolve exactly one current master race identity without guessing."""

    if bool(race_id) == bool(race_query):
        raise ValueError("exactly_one_of_race_id_or_race_required")
    query = _token(race_id or race_query)
    matches = [race for race in races if query in _race_query_tokens(race)]
    identities = sorted(
        {value for race in matches if (value := stable_race_id(race)) is not None}
    )
    if not matches:
        return "BLOCKED_RACE_NOT_FOUND", None, []
    if len(matches) != 1:
        return "BLOCKED_RACE_AMBIGUOUS", None, identities
    return "RESOLVED", matches[0], identities


def _read_lock_for_diagnostics(lock_path: Path) -> Mapping[str, Any] | None:
    if lock_path.is_symlink():
        return None
    try:
        value = json.loads(lock_path.read_bytes())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _unlink_owned_lock_inode(lock: OwnedCollectorLock) -> bool:
    try:
        current = lock.path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    if lock.path.is_symlink() or (current.st_dev, current.st_ino) != (
        lock.device,
        lock.inode,
    ):
        return False
    lock.path.unlink()
    return True


def _acquire_collector_lock_no_steal(
    lock_path: Path, *, run_id: str, output_dir: Path
) -> OwnedCollectorLock:
    if lock_path.is_symlink() or not lock_path.parent.is_dir():
        raise PredictionBlocked("LOCK_PATH_UNSAFE", path=str(lock_path))
    payload = {
        "schema_version": "shadow_autopilot_daemon_lock_v1",
        "run_id": run_id,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at": datetime.now().astimezone().isoformat(),
        "output_dir": str(output_dir.resolve()),
        "acquisition_policy": "on_demand_no_steal_v1",
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as exc:
        raise CollectorLockBusy(
            {
                "lock_path": str(lock_path),
                "existing_lock": _read_lock_for_diagnostics(lock_path),
                "reason": "existing_lock_present_no_steal",
            }
        ) from exc
    except OSError as exc:
        raise PredictionBlocked(
            "LOCK_ACQUIRE_FAILED", error=type(exc).__name__
        ) from exc
    try:
        opened = os.fstat(descriptor)
    except OSError as exc:
        try:
            opened = os.stat(descriptor)
        except OSError:
            os.close(descriptor)
            raise PredictionBlocked(
                "LOCK_ACQUIRE_FAILED",
                reason="descriptor_identity_unavailable",
                cleanup="not_safe",
            ) from exc
        os.close(descriptor)
        failed_lock = OwnedCollectorLock(
            path=lock_path,
            run_id=run_id,
            device=opened.st_dev,
            inode=opened.st_ino,
        )
        _unlink_owned_lock_inode(failed_lock)
        raise PredictionBlocked(
            "LOCK_ACQUIRE_FAILED", reason="descriptor_stat_failed"
        ) from exc
    lock = OwnedCollectorLock(
        path=lock_path,
        run_id=run_id,
        device=opened.st_dev,
        inode=opened.st_ino,
    )
    try:
        try:
            handle = os.fdopen(descriptor, "wb")
        except Exception:
            os.close(descriptor)
            raise
        with handle:
            handle.write(canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
    except Exception as exc:
        _unlink_owned_lock_inode(lock)
        raise PredictionBlocked(
            "LOCK_ACQUIRE_FAILED", error=type(exc).__name__
        ) from exc
    return lock


def _release_owned_collector_lock(lock: OwnedCollectorLock) -> None:
    payload = _read_lock_for_diagnostics(lock.path)
    if payload is None or payload.get("run_id") != lock.run_id:
        raise PredictionBlocked("LOCK_RELEASE_FAILED", reason="ownership_unverified")
    if not _unlink_owned_lock_inode(lock):
        raise PredictionBlocked("LOCK_RELEASE_FAILED", reason="inode_changed")


def _default_schedule(days_ahead: int) -> Sequence[Mapping[str, Any]]:
    from upcoming_race_browser import UpcomingRaceBrowser

    previous = os.environ.get("UPCOMING_RACES_DIR")
    try:
        with tempfile.TemporaryDirectory(prefix="greyhound_on_demand_schedule_") as tmp:
            os.environ["UPCOMING_RACES_DIR"] = tmp
            with contextlib.redirect_stdout(sys.stderr):
                races = UpcomingRaceBrowser().get_upcoming_races(days_ahead=days_ahead)
            return list(races)
    finally:
        if previous is None:
            os.environ.pop("UPCOMING_RACES_DIR", None)
        else:
            os.environ["UPCOMING_RACES_DIR"] = previous


@contextlib.contextmanager
def _isolated_runtime_cwd(bundle: Path):
    """Contain legacy import-time relative writes inside transient bundle scratch."""

    bundle = bundle.resolve()
    previous_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix=".runtime_", dir=bundle) as runtime:
        try:
            os.chdir(runtime)
            yield Path(runtime)
        finally:
            os.chdir(previous_cwd)


def _default_refresh(
    target: Mapping[str, Any], bundle: Path, current_time: datetime, days_ahead: int
) -> tuple[Path, Path]:
    del days_ahead
    bundle = bundle.resolve()
    upcoming_dir = bundle / "source" / "upcoming"
    upcoming_dir.mkdir(parents=True, exist_ok=False)
    previous = os.environ.get("UPCOMING_RACES_DIR")
    try:
        os.environ["UPCOMING_RACES_DIR"] = str(upcoming_dir)
        with _isolated_runtime_cwd(bundle):
            from scripts.predict_market_form_residual import (
                _sidecar_context,
                _validate_form_binding,
            )
            from upcoming_race_browser import UpcomingRaceBrowser

            with contextlib.redirect_stdout(sys.stderr):
                result = UpcomingRaceBrowser().download_race_csv(
                    str(target.get("url") or target.get("race_url") or ""),
                    race_info_hint=target,
                )
    except Exception as exc:
        raise PredictionBlocked(
            "EXACT_METADATA_UNAVAILABLE", error=type(exc).__name__
        ) from exc
    finally:
        if previous is None:
            os.environ.pop("UPCOMING_RACES_DIR", None)
        else:
            os.environ["UPCOMING_RACES_DIR"] = previous
    if not isinstance(result, Mapping) or result.get("success") is not True:
        raise PredictionBlocked(
            "EXACT_METADATA_UNAVAILABLE",
            reason=result.get("error") if isinstance(result, Mapping) else None,
        )
    form = Path(str(result.get("filepath") or ""))
    if not form.is_absolute():
        form = ROOT / form
    sidecar = form.with_name(form.name + ".metadata.json")
    if (
        form.is_symlink()
        or sidecar.is_symlink()
        or not form.is_file()
        or not sidecar.is_file()
        or not form.resolve().is_relative_to(upcoming_dir.resolve())
        or not sidecar.resolve().is_relative_to(upcoming_dir.resolve())
        or sidecar != form.with_name(form.name + ".metadata.json")
    ):
        raise PredictionBlocked("EXACT_METADATA_UNAVAILABLE")
    try:
        form_raw = form.read_bytes()
        sidecar_value = json.loads(sidecar.read_bytes())
        if not isinstance(sidecar_value, Mapping):
            raise TypeError("sidecar_not_mapping")
        context = _sidecar_context(sidecar_value)
        _validate_form_binding(
            sidecar_value,
            form_csv_path=form,
            form_raw=form_raw,
            form_sha=sha256_bytes(form_raw),
        )
    except (OSError, TypeError, json.JSONDecodeError, CaptureHandoffError) as exc:
        raise PredictionBlocked("EXACT_METADATA_UNAVAILABLE", reason=str(exc)) from exc
    expected_race_id = stable_race_id(target)
    expected_jump = _parse_race_jump_datetime(target, now=current_time)
    if (
        context.get("expected_race_id") != expected_race_id
        or expected_jump is None
        or context.get("jump_timestamp") != expected_jump
    ):
        raise PredictionBlocked(
            "EXACT_METADATA_UNAVAILABLE", reason="identity_mismatch"
        )
    return form, sidecar


def discover_capture_handoff(
    *,
    evidence_roots: Sequence[Path],
    db_path: Path,
    race_id: str,
    jump_datetime: datetime,
    capture_window_minutes: int,
    current_time: datetime,
) -> dict[str, Any] | None:
    """Adapt one master-verified sealed packet to the reuse-only receipt seam."""

    del db_path, capture_window_minutes
    match = re.fullmatch(
        r"Race\s+([0-9]{1,2})\s+-\s+(.+?)\s+-\s+[0-9]{4}-[0-9]{2}-[0-9]{2}",
        race_id,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise CaptureHandoffError("capture_packet_race_id_invalid")
    race_query = f"{match.group(2)} r{match.group(1)}"
    available_roots = [Path(root) for root in evidence_roots if Path(root).is_dir()]
    if not available_roots:
        return None
    try:
        packet = discover_race_artifacts(
            race_query=race_query,
            exact_race_id=race_id,
            evidence_roots=available_roots,
            score_timestamp=current_time,
        )
    except CaptureHandoffError as exc:
        if str(exc) in {
            "race_feature_packet_not_found",
            "race_capture_report_not_found",
        }:
            return None
        raise
    if str(packet.get("race_id")) != race_id:
        raise CaptureHandoffError("capture_packet_identity_mismatch")
    artifact_prediction = score_from_artifacts(
        race_id=race_id,
        form_csv_path=Path(packet["form_csv_path"]),
        sidecar_path=Path(packet["sidecar_path"]),
        feature_rows_path=Path(packet["feature_rows_path"]),
        feature_manifest_path=Path(packet["feature_manifest_path"]),
        implementation_manifest_path=Path(packet["implementation_manifest_path"]),
        capture_path=Path(packet["capture_path"]),
        model_path=ROOT / "artifacts/frozen_models/market_form_residual_v1/model.json",
        manifest_path=ROOT
        / "artifacts/frozen_models/market_form_residual_v1/manifest.json",
        score_timestamp=current_time,
    )
    try:
        packet_jump = datetime.fromisoformat(str(artifact_prediction["jump_timestamp"]))
        append_timestamp = str(artifact_prediction["odds_append_timestamp"])
        input_hashes = artifact_prediction["input_hashes"]
    except (KeyError, TypeError, ValueError) as exc:
        raise CaptureHandoffError("capture_packet_prediction_invalid") from exc
    if packet_jump != jump_datetime:
        raise CaptureHandoffError("capture_packet_jump_mismatch")

    raw_by_label: dict[str, bytes] = {}
    for label, path_key, hash_key in (
        ("report", "capture_path", "capture_artifact_sha256"),
        ("form", "form_csv_path", "form_csv_sha256"),
        ("sidecar", "sidecar_path", "sidecar_sha256"),
    ):
        path = Path(packet[path_key])
        if path.is_symlink() or not path.is_file():
            raise CaptureHandoffError(f"capture_packet_{label}_unsafe")
        raw = path.read_bytes()
        if sha256_bytes(raw) != input_hashes.get(hash_key):
            raise CaptureHandoffError(f"capture_packet_{label}_changed_after_score")
        raw_by_label[label] = raw

    form_name = Path(packet["form_csv_path"]).name
    return {
        "schema_version": "on_demand_verified_master_packet_v1",
        "race_id": race_id,
        "append_timestamp": append_timestamp,
        "source_report_sha256": sha256_bytes(raw_by_label["report"]),
        "source_form_sha256": sha256_bytes(raw_by_label["form"]),
        "source_sidecar_sha256": sha256_bytes(raw_by_label["sidecar"]),
        "packet_record_schema_version": artifact_prediction["record_schema_version"],
        "packet_record_checksum_sha256": artifact_prediction["record_checksum_sha256"],
        "packet_effective_state_schema_version": artifact_prediction[
            "effective_state_schema_version"
        ],
        "packet_effective_state_sha256": artifact_prediction["effective_state_sha256"],
        "_report_bytes": raw_by_label["report"],
        "_form_bytes": raw_by_label["form"],
        "_sidecar_bytes": raw_by_label["sidecar"],
        "_form_name": form_name,
    }


def seal_live_features(
    *,
    form_csv: Path,
    db_path: Path,
    output_dir: Path,
    current_time: datetime,
) -> dict[str, Path]:
    """Build a current-master feature packet inside the isolated run bundle."""

    from scripts.run_feature_recovery_execution_v1 import DEFAULT_SCHEMA, load_json
    from scripts.run_shadow_non_tgr_rf_evaluation import (
        build_live_feature_rows,
        same_distance_same_grade_history_provenance_report,
        shadow_relpath,
        validate_schema_contract,
    )

    schema = load_json(DEFAULT_SCHEMA)
    audit = validate_schema_contract(schema)
    if audit.get("status") != "PASS":
        raise RuntimeError(f"schema_contract_failed:{audit.get('fail_reasons')}")
    rows = build_live_feature_rows(
        input_paths=[form_csv], schema=schema, db_path=db_path
    )
    if not rows:
        raise RuntimeError("feature_rows_missing")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "shadow_feature_rows.json"
    manifest_path = output_dir / "shadow_manifest.json"
    history_path = output_dir / "same_distance_same_grade_history_provenance.json"
    implementation_path = output_dir / "implementation_file_manifest.json"
    _write_canonical(rows_path, rows)
    _write_canonical(
        history_path,
        same_distance_same_grade_history_provenance_report(rows),
    )
    manifest = {
        "schema_version": "shadow_live_scoring_manifest_v1",
        "generated_at": current_time.isoformat(),
        "run_started_at": current_time.isoformat(),
        "feature_freeze_timestamp": current_time.isoformat(),
        "output_mode": "shadow_only",
        "input_files": [shadow_relpath(form_csv)],
        "prediction_rows": 0,
        "feature_rows": shadow_relpath(rows_path),
        "tgr_enabled": False,
        "registry_mutation": False,
        "production_prediction_write": False,
        "odds_used_for_shadow_scoring": False,
        "betting_output": False,
        "ev_output": False,
    }
    _write_canonical(manifest_path, manifest)
    artifacts = {
        shadow_relpath(path): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in (rows_path, manifest_path, history_path)
    }
    implementation = {
        "schema_version": "shadow_implementation_file_manifest_v1",
        "output_dir": shadow_relpath(output_dir),
        "git_head": "on_demand_isolated_runtime",
        "git_branch": "on_demand_isolated_runtime",
        "implementation_files": list(FEATURE_GENERATOR_FILES),
        "implementation_file_hashes": {
            relative: sha256_file(ROOT / relative)
            for relative in FEATURE_GENERATOR_FILES
        },
        "artifact_files": artifacts,
    }
    _write_canonical(implementation_path, implementation)
    return {
        "feature_rows": rows_path,
        "feature_manifest": manifest_path,
        "implementation_manifest": implementation_path,
    }


def _default_fetch(
    context: Mapping[str, Any], isolated_db: Path, timeout: float
) -> Mapping[str, Any]:
    from scripts import autonomous_live_odds_capture as capture

    form_csv = Path(str(context["form_csv"]))
    current_time = datetime.now().astimezone()
    jump = context["jump_timestamp"]
    if current_time >= jump:
        raise PredictionBlocked("POST_JUMP")
    item = capture.build_plan_item(form_csv, current_time)
    target_race_id = str(context["race_id"])
    blockers = list(item.get("blockers") or [])
    if item.get("race_id") != target_race_id:
        raise PredictionBlocked("EXACT_METADATA_UNAVAILABLE", reasons=blockers)
    if blockers:
        raise PredictionBlocked("CAPTURE_WINDOW_UNAVAILABLE", reasons=blockers)
    with contextlib.redirect_stdout(sys.stderr):
        fetched = capture.fetch_odds_for_target_race_with_timeout(
            str(isolated_db),
            item.get("venue"),
            item.get("race_number"),
            item.get("race_date"),
            allow_auto_scrape_odds=True,
            timeout_seconds=timeout,
        )
    validation = capture.validate_fetched_odds(item, fetched)
    captured_at = datetime.now().astimezone()
    if captured_at >= jump:
        raise PredictionBlocked("POST_JUMP")
    return {
        "captured_at": captured_at.isoformat(),
        "plan_item": item,
        "validation": validation,
        "fetch_result": {
            key: value
            for key, value in fetched.items()
            if key not in {"odds_data", "odds_data_place", "race_info"}
        },
    }


def default_dependencies(args: argparse.Namespace) -> Dependencies:
    from scripts.predict_market_form_residual import score_from_artifacts

    run_id = f"on_demand_{uuid.uuid4().hex}"

    def acquire_lock() -> Any:
        expected_lock = Path(args.db).resolve().parent / LOCK_RELATIVE_PATH
        if Path(args.lock_path).resolve() != expected_lock.resolve():
            raise PredictionBlocked(
                "LOCK_PATH_DB_ROOT_MISMATCH",
                expected_lock_path=str(expected_lock),
            )
        return _acquire_collector_lock_no_steal(
            Path(args.lock_path),
            run_id=run_id,
            output_dir=Path(args.lock_output_dir),
        )

    def release_lock(handle: Any) -> None:
        if not isinstance(handle, OwnedCollectorLock):
            raise PredictionBlocked("LOCK_RELEASE_FAILED", reason="handle_invalid")
        _release_owned_collector_lock(handle)

    return Dependencies(
        schedule=_default_schedule,
        refresh=_default_refresh,
        discover_receipt=discover_capture_handoff,
        fetch_odds=_default_fetch,
        acquire_lock=acquire_lock,
        release_lock=release_lock,
        lock_busy_type=CollectorLockBusy,
        seal_features=seal_live_features,
        score_residual=score_from_artifacts,
        now=lambda: datetime.now().astimezone(),
    )


def _public_model(model: Any) -> dict[str, Any]:
    return {
        "requested": model.requested,
        "resolved": model.resolved,
        "alias_resolved": model.alias,
        "model_sha256": model.model_sha256,
        "manifest_sha256": model.manifest_sha256,
        "schema_sha256": model.schema_sha256,
    }


def _capture_attempt(
    *, race_id: str, captured_at: datetime, validation: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "autonomous_live_odds_capture_attempt_v1",
        "race_id": race_id,
        "status": "APPENDED",
        "reasons": [],
        "fetch_time": captured_at.isoformat(),
        "append_time": captured_at.isoformat(),
        "persistence_scope": "isolated_research_bundle_only",
        "validation": dict(validation),
    }


def _selected_variant(prediction: Mapping[str, Any], variant: str) -> dict[str, Any]:
    probability_key = {
        "full_strength": "full_probability",
        "half_strength": "half_probability",
    }[variant]
    rows = [
        {
            "box_number": int(row["box_number"] if "box_number" in row else row["box"]),
            "dog_name": str(row["dog_name"] if "dog_name" in row else row["dog"]),
            "probability": float(row[probability_key]),
            "market_probability": float(row["market_probability"]),
            "win_odds": float(row["win_odds"]),
        }
        for row in prediction.get("predictions") or []
    ]
    rows.sort(key=lambda row: (-row["probability"], row["box_number"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {
        "adapter": "market_form_residual_v1",
        "variant": variant,
        "probability_sum": sum(row["probability"] for row in rows),
        "predictions": rows,
        "artifact_prediction": dict(prediction),
    }


def _bundle_file(bundle: Path, path: Path, *, label: str) -> tuple[Path, str]:
    """Return one regular, non-symlink bundle file and its stable relative path."""

    bundle_root = bundle.resolve()
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise PredictionBlocked(
            "BUNDLE_SOURCE_UNSAFE", source=label, path=str(candidate)
        )
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(bundle_root)
    except (OSError, ValueError) as exc:
        raise PredictionBlocked(
            "BUNDLE_SOURCE_UNSAFE", source=label, path=str(candidate)
        ) from exc
    return resolved, relative.as_posix()


def _validated_refreshed_sources(
    bundle: Path, form_csv: Path, sidecar: Path
) -> tuple[Path, Path]:
    form, _ = _bundle_file(bundle, form_csv, label="refreshed_form_csv")
    metadata, _ = _bundle_file(bundle, sidecar, label="refreshed_sidecar")
    if metadata != form.with_name(form.name + ".metadata.json"):
        raise PredictionBlocked(
            "BUNDLE_SOURCE_UNSAFE",
            source="refreshed_sidecar",
            reason="sidecar_not_exactly_adjacent",
        )
    return form, metadata


def _persist_blocked_bundle(bundle: Path, exc: PredictionBlocked) -> None:
    """Seal the smallest post-bundle blocker as a research-only result."""

    bundle_path = str(bundle.resolve())
    blocker = {"code": exc.code, **exc.details}
    result = {
        "schema_version": "on_demand_race_prediction_v1",
        "status": exc.code,
        "research_only": True,
        "production_persisted": False,
        "betting_output": False,
        "blockers": [blocker],
        "bundle": bundle_path,
    }
    _write_canonical(bundle / "result.json", result)
    _write_canonical(bundle / "bundle_manifest.json", bundle_manifest(bundle))
    exc.details["bundle"] = bundle_path


def _discover(
    dependencies: Dependencies,
    *,
    evidence_roots: Sequence[Path],
    db_path: Path,
    race_id: str,
    jump: datetime,
    current_time: datetime,
) -> Mapping[str, Any] | None:
    from scripts.autonomous_live_odds_capture import due_capture_window

    capture_window, _ = due_capture_window((jump - current_time).total_seconds() / 60.0)
    if capture_window is None:
        return None
    try:
        return dependencies.discover_receipt(
            evidence_roots=evidence_roots,
            db_path=db_path,
            race_id=race_id,
            jump_datetime=jump,
            capture_window_minutes=capture_window,
            current_time=current_time,
        )
    except CaptureHandoffError as exc:
        reason = str(exc)
        code = "RECEIPT_AMBIGUOUS" if "ambiguous" in reason else "RECEIPT_INVALID"
        raise PredictionBlocked(code, reason=reason) from exc


def _discover_for_auto(
    dependencies: Dependencies,
    *,
    evidence_roots: Sequence[Path],
    db_path: Path,
    race_id: str,
    jump: datetime,
    current_time: datetime,
    rejected_receipts: list[dict[str, Any]],
) -> Mapping[str, Any] | None:
    """Reject only proven pre-current packets before trying fresh acquisition."""

    try:
        return _discover(
            dependencies,
            evidence_roots=evidence_roots,
            db_path=db_path,
            race_id=race_id,
            jump=jump,
            current_time=current_time,
        )
    except PredictionBlocked as exc:
        reason = str(exc.details.get("reason") or "")
        if (
            exc.code != "RECEIPT_INVALID"
            or reason not in PRECURRENT_PACKET_REJECTION_REASONS
        ):
            raise
        rejection = {"code": exc.code, "reason": reason}
        if rejection not in rejected_receipts:
            rejected_receipts.append(rejection)
        return None


def _acquire_or_reuse(
    dependencies: Dependencies,
    *,
    odds_source: str,
    evidence_roots: Sequence[Path],
    db_path: Path,
    race_id: str,
    jump: datetime,
    current_time: datetime,
    wait_seconds: float,
    poll_seconds: float,
) -> tuple[Any | None, Mapping[str, Any] | None, list[dict[str, Any]]]:
    rejected_receipts: list[dict[str, Any]] = []
    if odds_source == "receipt":
        receipt = _discover(
            dependencies,
            evidence_roots=evidence_roots,
            db_path=db_path,
            race_id=race_id,
            jump=jump,
            current_time=current_time,
        )
        if receipt is not None:
            return None, receipt, rejected_receipts
        raise PredictionBlocked("RECEIPT_UNAVAILABLE")
    if odds_source == "auto":
        receipt = _discover_for_auto(
            dependencies,
            evidence_roots=evidence_roots,
            db_path=db_path,
            race_id=race_id,
            jump=jump,
            current_time=current_time,
            rejected_receipts=rejected_receipts,
        )
        if receipt is not None:
            return None, receipt, rejected_receipts
    started = dependencies.monotonic()
    deadline = started + wait_seconds
    last_busy: BaseException | None = None
    while True:
        observed = dependencies.monotonic()
        elapsed = max(0.0, observed - started)
        observed_time = current_time + timedelta(seconds=elapsed)
        if observed_time >= jump:
            raise PredictionBlocked(
                "POST_JUMP",
                race_id=race_id,
                jump_timestamp=jump.isoformat(),
                lock_wait_elapsed_seconds=elapsed,
                lock_wait_limit_seconds=wait_seconds,
                lock_details=getattr(last_busy, "payload", None),
            )
        if last_busy is not None and observed >= deadline:
            raise PredictionBlocked(
                "BUSY",
                lock_details=getattr(last_busy, "payload", None),
                lock_wait_elapsed_seconds=elapsed,
                lock_wait_limit_seconds=wait_seconds,
            ) from last_busy
        try:
            return dependencies.acquire_lock(), None, rejected_receipts
        except dependencies.lock_busy_type as exc:
            last_busy = exc
            observed = dependencies.monotonic()
            elapsed = max(0.0, observed - started)
            observed_time = current_time + timedelta(seconds=elapsed)
            if observed_time >= jump:
                raise PredictionBlocked(
                    "POST_JUMP",
                    race_id=race_id,
                    jump_timestamp=jump.isoformat(),
                    lock_wait_elapsed_seconds=elapsed,
                    lock_wait_limit_seconds=wait_seconds,
                    lock_details=getattr(exc, "payload", None),
                ) from exc
            remaining = deadline - observed
            if remaining <= 0:
                details = getattr(exc, "payload", None)
                raise PredictionBlocked(
                    "BUSY",
                    lock_details=details,
                    lock_wait_elapsed_seconds=elapsed,
                    lock_wait_limit_seconds=wait_seconds,
                ) from exc
            if odds_source == "auto":
                receipt = _discover_for_auto(
                    dependencies,
                    evidence_roots=evidence_roots,
                    db_path=db_path,
                    race_id=race_id,
                    jump=jump,
                    current_time=observed_time,
                    rejected_receipts=rejected_receipts,
                )
                if receipt is not None:
                    return None, receipt, rejected_receipts
            until_jump = (jump - observed_time).total_seconds()
            dependencies.sleep(min(max(poll_seconds, 0.01), remaining, until_jump))


def _run_prediction(
    args: argparse.Namespace, dependencies: Dependencies, state: dict[str, Path]
) -> dict[str, Any]:
    current_time = (
        parse_current_time(args.current_time)
        if args.current_time
        else dependencies.now()
    )
    if current_time.tzinfo is None or current_time.utcoffset() is None:
        raise PredictionBlocked("CURRENT_TIME_TIMEZONE_MISSING")
    model = resolve_model(args.model)
    config, config_sha, config_raw = load_config(Path(args.config), model)
    races = dependencies.schedule(int(args.days_ahead))
    resolved, target, matches = resolve_target_race(
        races, race_id=None, race_query=args.race
    )
    if resolved != "RESOLVED" or target is None:
        raise PredictionBlocked(resolved, matching_race_ids=matches)
    race_id = str(stable_race_id(target) or "")
    jump = _parse_race_jump_datetime(target, now=current_time)
    if not race_id or jump is None:
        raise PredictionBlocked("EXACT_RACE_IDENTITY_UNAVAILABLE")
    if jump <= current_time:
        raise PredictionBlocked(
            "POST_JUMP", race_id=race_id, jump_timestamp=jump.isoformat()
        )

    bundle = create_bundle(Path(args.output_root), current_time)
    state["bundle"] = bundle
    request = {
        "schema_version": "on_demand_prediction_request_v1",
        "race_query": args.race,
        "race_id": race_id,
        "jump_timestamp": jump.isoformat(),
        "request_timestamp": current_time.isoformat(),
        "odds_source": args.odds_source,
        "model": _public_model(model),
        "config_sha256": config_sha,
        "research_only": True,
    }
    _write_canonical(bundle / "request.json", request)
    write_exact_bytes(bundle / "config.json", config_raw)
    _copy_exact(model.schema_path, bundle / "model" / "config.schema.json")
    if model.model_path and model.manifest_path:
        _copy_exact(model.model_path, bundle / "model" / "model.json")
        _copy_exact(model.manifest_path, bundle / "model" / "manifest.json")

    lock_handle: Any | None = None
    try:
        lock_handle, handoff, rejected_receipts = _acquire_or_reuse(
            dependencies,
            odds_source=args.odds_source,
            evidence_roots=tuple(Path(path) for path in args.capture_evidence_root),
            db_path=Path(args.db),
            race_id=race_id,
            jump=jump,
            current_time=current_time,
            wait_seconds=float(config["bundle"]["lock_wait_seconds"]),
            poll_seconds=float(config["bundle"]["poll_seconds"]),
        )
        if rejected_receipts:
            _write_canonical(
                bundle / "source" / "rejected_receipts.json",
                {
                    "schema_version": "on_demand_rejected_receipts_v1",
                    "rejections": rejected_receipts,
                },
            )
        if handoff is not None:
            receipt, capture_raw, form_raw, sidecar_raw = receipt_from_handoff(
                handoff,
                current_time=current_time,
                max_age_seconds=int(config["bundle"]["receipt_max_age_seconds"]),
            )
            form_name = str(handoff.get("_form_name") or "form.csv")
            if Path(form_name).name != form_name:
                raise PredictionBlocked("RECEIPT_INVALID")
            form_csv = bundle / "source" / form_name
            sidecar = form_csv.with_name(form_csv.name + ".metadata.json")
            capture_path = bundle / "source" / "capture.json"
            write_exact_bytes(form_csv, form_raw)
            write_exact_bytes(sidecar, sidecar_raw)
            write_exact_bytes(capture_path, capture_raw)
        else:
            form_csv, sidecar = dependencies.refresh(
                target, bundle, current_time, int(args.days_ahead)
            )
            form_csv, sidecar = _validated_refreshed_sources(bundle, form_csv, sidecar)
            plan_context = {
                "race_id": race_id,
                "form_csv": str(form_csv),
                "current_time": dependencies.now(),
                "jump_timestamp": jump,
            }
            provisional_names: list[str] = []
            try:
                sidecar_value = json.loads(sidecar.read_bytes())
                participants = (
                    sidecar_value.get("participants")
                    or sidecar_value.get("runners")
                    or []
                )
                provisional_names = [
                    str(row.get("dog_name") or row.get("name") or "")
                    for row in participants
                    if isinstance(row, Mapping)
                ]
            except (OSError, json.JSONDecodeError):
                pass
            sealed_db = bundle / "features" / "sealed_history.db"
            history = seal_history_database(
                source=Path(args.db),
                target=sealed_db,
                target_race_id=race_id,
                cutoff=jump,
                runner_names=provisional_names,
            )
            _write_canonical(bundle / "features" / "history_seal.json", history)
            try:
                fetched = dependencies.fetch_odds(
                    plan_context, sealed_db, float(args.fetch_timeout_seconds)
                )
            except PredictionBlocked:
                raise
            except Exception as exc:
                raise PredictionBlocked(
                    "MARKET_UNAVAILABLE", error=type(exc).__name__
                ) from exc
            try:
                captured_at = datetime.fromisoformat(str(fetched["captured_at"]))
                validation = fetched["validation"]
            except (KeyError, TypeError, ValueError) as exc:
                raise PredictionBlocked("MARKET_UNAVAILABLE") from exc
            if not isinstance(validation, Mapping):
                raise PredictionBlocked("MARKET_UNAVAILABLE")
            receipt = normalize_validation_receipt(
                race_id=race_id,
                captured_at=captured_at,
                validation=validation,
                source_kind="isolated_immediate_capture",
            )
            attempt = _capture_attempt(
                race_id=race_id, captured_at=captured_at, validation=validation
            )
            capture_path = bundle / "source" / "capture.json"
            _write_canonical(capture_path, attempt)
            _write_canonical(bundle / "source" / "capture_provenance.json", fetched)
    finally:
        if lock_handle is not None:
            dependencies.release_lock(lock_handle)

    try:
        odds_captured_at = datetime.fromisoformat(str(receipt["captured_at"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise PredictionBlocked("ODDS_TIMESTAMP_AMBIGUOUS") from exc
    if odds_captured_at.tzinfo is None or odds_captured_at.utcoffset() is None:
        raise PredictionBlocked("ODDS_TIMESTAMP_AMBIGUOUS")
    if odds_captured_at >= jump:
        raise PredictionBlocked(
            "POST_JUMP", race_id=race_id, odds_captured_at=odds_captured_at.isoformat()
        )

    _write_canonical(bundle / "odds_receipt.json", receipt)
    runner_names = [str(row["dog_name"]) for row in receipt["markets"]["win"]]
    sealed_db = bundle / "features" / "sealed_history.db"
    history_path = bundle / "features" / "history_seal.json"
    if not sealed_db.exists():
        history = seal_history_database(
            source=Path(args.db),
            target=sealed_db,
            target_race_id=race_id,
            cutoff=jump,
            runner_names=runner_names,
        )
        _write_canonical(history_path, history)
    else:
        history = json.loads(history_path.read_bytes())
    if (
        history.get("target_rows_materialized") != 0
        or history.get("at_or_after_cutoff_rows_materialized") != 0
    ):
        raise PredictionBlocked("TARGET_EXCLUSION_WEAK")

    score_time = dependencies.now()
    if score_time >= jump:
        raise PredictionBlocked("POST_JUMP", race_id=race_id)
    if model.resolved == "market_only_v1":
        prediction = market_only_prediction(receipt)
        feature_identity: dict[str, Any] = {"required": False}
    else:
        feature_dir = bundle / "features" / "sealed"
        try:
            sealed = dependencies.seal_features(
                form_csv=form_csv,
                db_path=sealed_db,
                output_dir=feature_dir,
                current_time=score_time,
            )
        except PredictionBlocked:
            raise
        except Exception as exc:
            raise PredictionBlocked(
                "FEATURE_SEAL_FAILED", error=type(exc).__name__
            ) from exc
        feature_rows = json.loads(Path(sealed["feature_rows"]).read_bytes())
        unsafe_rows = [
            row
            for row in feature_rows
            if row.get("same_distance_same_grade_target_race_rows_used")
            not in (0, None)
            or row.get("same_distance_same_grade_post_outcome_rows_used")
            not in (0, None)
        ]
        if unsafe_rows:
            raise PredictionBlocked("TARGET_EXCLUSION_WEAK")
        try:
            artifact_prediction = dependencies.score_residual(
                race_id=race_id,
                form_csv_path=form_csv,
                sidecar_path=sidecar,
                feature_rows_path=Path(sealed["feature_rows"]),
                feature_manifest_path=Path(sealed["feature_manifest"]),
                implementation_manifest_path=Path(sealed["implementation_manifest"]),
                capture_path=capture_path,
                model_path=bundle / "model" / "model.json",
                manifest_path=bundle / "model" / "manifest.json",
                score_timestamp=score_time,
            )
        except PredictionBlocked:
            raise
        except Exception as exc:
            raise PredictionBlocked(
                "RESIDUAL_SCORER_FAILED", error=type(exc).__name__
            ) from exc
        if (
            artifact_prediction.get("model_sha256") != model.model_sha256
            or artifact_prediction.get("manifest_sha256") != model.manifest_sha256
        ):
            raise PredictionBlocked("FROZEN_MODEL_DRIFT")
        prediction = _selected_variant(artifact_prediction, str(config["variant"]))
        replay_paths = {
            "form_csv": _bundle_file(bundle, form_csv, label="form_csv")[1],
            "sidecar": _bundle_file(bundle, sidecar, label="sidecar")[1],
            "feature_rows": _bundle_file(
                bundle, Path(sealed["feature_rows"]), label="feature_rows"
            )[1],
            "feature_manifest": _bundle_file(
                bundle, Path(sealed["feature_manifest"]), label="feature_manifest"
            )[1],
            "implementation_manifest": _bundle_file(
                bundle,
                Path(sealed["implementation_manifest"]),
                label="implementation_manifest",
            )[1],
            "capture": _bundle_file(bundle, capture_path, label="capture")[1],
            "model": _bundle_file(
                bundle, bundle / "model" / "model.json", label="model"
            )[1],
            "manifest": _bundle_file(
                bundle, bundle / "model" / "manifest.json", label="manifest"
            )[1],
        }
        feature_identity = {
            "required": True,
            "feature_rows_sha256": sha256_file(Path(sealed["feature_rows"])),
            "feature_manifest_sha256": sha256_file(Path(sealed["feature_manifest"])),
            "implementation_manifest_sha256": sha256_file(
                Path(sealed["implementation_manifest"])
            ),
            "replay_paths": replay_paths,
        }

    completed_time = dependencies.now()
    if completed_time.tzinfo is None or completed_time.utcoffset() is None:
        raise PredictionBlocked("CURRENT_TIME_TIMEZONE_MISSING")
    if completed_time >= jump:
        raise PredictionBlocked(
            "POST_JUMP", race_id=race_id, completed_at=completed_time.isoformat()
        )

    result = {
        "schema_version": "on_demand_race_prediction_v1",
        "status": "PREDICTION_READY",
        "research_only": True,
        "production_persisted": False,
        "betting_output": False,
        "race": {
            "query": args.race,
            "race_id": race_id,
            "jump_timestamp": jump.isoformat(),
        },
        "score_timestamp": score_time.isoformat(),
        "odds_source": receipt["source_kind"],
        "rejected_receipts": rejected_receipts,
        "runner_set_sha256": receipt["runner_set_sha256"],
        "model": _public_model(model),
        "config": {"sha256": config_sha, "value": config},
        "history_seal": history,
        "source_identity": {
            "form_csv_sha256": sha256_file(form_csv),
            "sidecar_sha256": sha256_file(sidecar),
            "capture_sha256": sha256_file(capture_path),
            "sealed_history_sha256": sha256_file(sealed_db),
            "odds_captured_at": receipt["captured_at"],
        },
        "feature_identity": feature_identity,
        "prediction": prediction,
        "blockers": [],
        "bundle": str(bundle.resolve()),
    }
    _write_canonical(bundle / "result.json", result)
    manifest = bundle_manifest(bundle)
    _write_canonical(bundle / "bundle_manifest.json", manifest)
    return result


def run_prediction(
    args: argparse.Namespace, dependencies: Dependencies
) -> dict[str, Any]:
    state: dict[str, Path] = {}
    try:
        return _run_prediction(args, dependencies, state)
    except PredictionBlocked as exc:
        bundle = state.get("bundle")
        if bundle is not None:
            try:
                _persist_blocked_bundle(bundle, exc)
            except (OSError, TypeError, ValueError, PredictionBlocked) as persist_exc:
                # Never replace the smallest operational blocker with a reporting failure.
                exc.details["bundle"] = str(bundle.resolve())
                exc.details["bundle_persistence_error"] = type(persist_exc).__name__
        raise
    except Exception as exc:
        blocked = PredictionBlocked(
            "PREDICTION_INTERNAL_ERROR", error=type(exc).__name__
        )
        bundle = state.get("bundle")
        if bundle is not None:
            try:
                _persist_blocked_bundle(bundle, blocked)
            except (OSError, TypeError, ValueError, PredictionBlocked) as persist_exc:
                blocked.details["bundle"] = str(bundle.resolve())
                blocked.details["bundle_persistence_error"] = type(persist_exc).__name__
        raise blocked from exc


def replay_bundle(
    bundle: Path,
    score_residual: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    verify_bundle(bundle)
    try:
        result = json.loads((bundle / "result.json").read_bytes())
        receipt = json.loads((bundle / "odds_receipt.json").read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictionBlocked("REPLAY_INPUT_INVALID") from exc
    adapter = result.get("prediction", {}).get("adapter")
    if adapter == "market_only_v1":
        replayed = market_only_prediction(receipt)
        if canonical_bytes(replayed) != canonical_bytes(result["prediction"]):
            raise PredictionBlocked("REPLAY_NONDETERMINISTIC")
    else:
        feature_identity = result.get("feature_identity") or {}
        replay_paths = feature_identity.get("replay_paths")
        if not isinstance(replay_paths, Mapping):
            raise PredictionBlocked(
                "REPLAY_INPUT_INVALID", reason="replay_paths_missing"
            )

        def replay_path(label: str) -> Path:
            raw = replay_paths.get(label)
            if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
                raise PredictionBlocked("REPLAY_INPUT_INVALID", source=label)
            path, relative = _bundle_file(bundle, bundle / raw, label=label)
            if relative != raw:
                raise PredictionBlocked("REPLAY_INPUT_INVALID", source=label)
            return path

        inputs = {label: replay_path(label) for label in replay_paths}
        required = {
            "form_csv",
            "sidecar",
            "feature_rows",
            "feature_manifest",
            "implementation_manifest",
            "capture",
            "model",
            "manifest",
        }
        if set(inputs) != required:
            raise PredictionBlocked("REPLAY_INPUT_INVALID", reason="replay_path_set")
        expected = {
            "feature_rows_sha256": sha256_file(inputs["feature_rows"]),
            "feature_manifest_sha256": sha256_file(inputs["feature_manifest"]),
            "implementation_manifest_sha256": sha256_file(
                inputs["implementation_manifest"]
            ),
        }
        if any(feature_identity.get(key) != value for key, value in expected.items()):
            raise PredictionBlocked("REPLAY_TAMPERED")
        try:
            score_time = datetime.fromisoformat(str(result["score_timestamp"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise PredictionBlocked(
                "REPLAY_INPUT_INVALID", reason="score_timestamp"
            ) from exc
        if score_time.tzinfo is None or score_time.utcoffset() is None:
            raise PredictionBlocked(
                "REPLAY_INPUT_INVALID", reason="score_timestamp_timezone"
            )
        if score_residual is None:
            from scripts.predict_market_form_residual import score_from_artifacts

            score_residual = score_from_artifacts
        try:
            artifact_prediction = score_residual(
                race_id=str(result["race"]["race_id"]),
                form_csv_path=inputs["form_csv"],
                sidecar_path=inputs["sidecar"],
                feature_rows_path=inputs["feature_rows"],
                feature_manifest_path=inputs["feature_manifest"],
                implementation_manifest_path=inputs["implementation_manifest"],
                capture_path=inputs["capture"],
                model_path=inputs["model"],
                manifest_path=inputs["manifest"],
                score_timestamp=score_time,
            )
            replayed = _selected_variant(
                artifact_prediction, str(result["prediction"]["variant"])
            )
        except PredictionBlocked:
            raise
        except Exception as exc:
            raise PredictionBlocked(
                "REPLAY_SCORER_FAILED", error=type(exc).__name__
            ) from exc
        if canonical_bytes(replayed) != canonical_bytes(result["prediction"]):
            raise PredictionBlocked("REPLAY_NONDETERMINISTIC")
    return result


def list_configs() -> dict[str, Any]:
    """Return the finite checked-in config catalog with resolved immutable IDs."""

    catalog = []
    for name, selector, relative in (
        (
            "market-form-residual-v1",
            "latest-research",
            Path("configs/prediction/manual-default.json"),
        ),
        (
            "market-only",
            "market-only",
            Path("configs/prediction/market-only.json"),
        ),
    ):
        model = resolve_model(selector)
        config, config_sha, _ = load_config(ROOT / relative, model)
        catalog.append(
            {
                "name": name,
                "selector": selector,
                "config": relative.as_posix(),
                "config_sha256": config_sha,
                "resolved_config": config,
                "model": _public_model(model),
            }
        )
    return {
        "schema_version": "on_demand_prediction_config_catalog_v1",
        "status": "CONFIGS_AVAILABLE",
        "configs": catalog,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--race", help="Exact named upcoming race, e.g. 'gunnedah r5'")
    target.add_argument("--replay-bundle", type=Path)
    target.add_argument("--list-configs", action="store_true")
    parser.add_argument("--model", default="latest-research")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/prediction/manual-default.json"
    )
    parser.add_argument(
        "--odds-source", choices=("auto", "receipt", "capture"), default="auto"
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--days-ahead", type=int, default=1)
    parser.add_argument("--current-time")
    parser.add_argument("--fetch-timeout-seconds", type=float, default=45.0)
    parser.add_argument(
        "--capture-evidence-root",
        action="append",
        type=Path,
        default=list(DEFAULT_CAPTURE_EVIDENCE_ROOTS),
    )
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--lock-output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.list_configs:
            output = list_configs()
        elif args.replay_bundle:
            output = replay_bundle(args.replay_bundle.resolve())
        else:
            output = run_prediction(args, default_dependencies(args))
    except PredictionBlocked as exc:
        output = {
            "schema_version": "on_demand_race_prediction_v1",
            "status": exc.code,
            "research_only": True,
            "production_persisted": False,
            "blockers": [{"code": exc.code, **exc.details}],
        }
        if "bundle" in exc.details:
            output["bundle"] = exc.details["bundle"]
    print(canonical_bytes(output).decode(), end="")
    return 0 if output.get("status") in {"PREDICTION_READY", "CONFIGS_AVAILABLE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
