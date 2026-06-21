#!/usr/bin/env python3
"""Timer-safe daemon wrapper for shadow evidence accumulation.

The daemon layer schedules and supervises the existing shadow autopilot,
rechecks older pending shadow runs for exact official-result joins, refreshes
aggregate dashboards, emits alerts, and records lock/recovery validation. It
must not train, promote, mutate registries, write labels, enable TGR, overwrite
production predictions, rewrite snapshots, or emit betting/EV actions. DB
writes are restricted to explicitly enabled append-only live odds and
official-result evidence rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from scripts import shadow_autopilot_v1 as autopilot  # noqa: E402
from scripts.forward_shadow_runtime_state import (  # noqa: E402
    build_runtime_state as build_forward_shadow_runtime_state,
    build_summary as build_forward_shadow_runtime_summary,
)


DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/shadow_autopilot_daemonization_v1_"
OUTPUT_ARTIFACT_PREFIX = "shadow_autopilot_daemonization_v1_"
DEFAULT_RUNTIME_DIR = DEFAULT_EVIDENCE_ROOT / "shadow_autopilot_daemon_runtime"
DEFAULT_LOCK_PATH = DEFAULT_RUNTIME_DIR / "shadow_autopilot.lock"
DEFAULT_STATE_PATH = DEFAULT_RUNTIME_DIR / "state.json"
DEFAULT_ODDS_CAPTURE_ONLY_STATE_PATH = DEFAULT_RUNTIME_DIR / "odds_capture_state.json"
DEFAULT_SERVICE_DIR = ROOT / "ops/systemd"
DEFAULT_TARGET_JOINED_RACES = 100
DEFAULT_MIN_JOINED_RACES = 100
DEFAULT_TIMEOUT_SECONDS = 840
DEFAULT_LOCK_STALE_SECONDS = 3600
DEFAULT_REJOIN_PENDING_LIMIT = 8
DEFAULT_REJOIN_LOOKBACK_DAYS = 7
DEFAULT_TIMER_FREQUENCY = "15min"
DEFAULT_TIMER_ON_CALENDAR = "*:02/15"
DEFAULT_TIMER_ACCURACY = "30s"
DEFAULT_ODDS_CAPTURE_ONLY_TIMER_FREQUENCY = "1min_except_full_daemon"
DEFAULT_ODDS_CAPTURE_ONLY_TIMER_ON_CALENDAR = (
    "*:00,01,03,04,05,06,07,08,09,10,11,12,13,14,15,16,18,19,20,21,22,23,24,25,26,27,28,29,30,31,33,34,35,36,37,38,39,40,41,42,43,44,45,46,48,49,50,51,52,53,54,55,56,57,58,59"
)
DEFAULT_ODDS_CAPTURE_ONLY_TIMER_ACCURACY = "15s"
DEFAULT_ODDS_CAPTURE_ONLY_TIMEOUT_SECONDS = 600
DEFAULT_ODDS_CAPTURE_ONLY_REFRESH_LIMIT = 8
DEFAULT_FULL_DAEMON_AUTONOMOUS_ODDS_CAPTURE_LIMIT = 2
DEFAULT_FULL_DAEMON_RESULT_BACKLOG_LIMIT = 32
DEFAULT_FULL_DAEMON_RESULT_BACKLOG_SHADOW_RUN_LIMIT = 64
DEFAULT_FULL_DAEMON_RESULT_BACKLOG_LOOKBACK_DAYS = 2
DEFAULT_ODDS_CAPTURE_ONLY_PREFLIGHT_MAX_AGE_SECONDS = 30 * 60
DEFAULT_ODDS_CAPTURE_ONLY_PREFLIGHT_RESUME_BUFFER_SECONDS = 5 * 60
DEFAULT_FULL_DAEMON_ODDS_DEFER_HORIZON_SECONDS = 8 * 60
DEFAULT_FULL_DAEMON_ODDS_LOCK_RETRY_SECONDS = DEFAULT_ODDS_CAPTURE_ONLY_TIMEOUT_SECONDS + 60
DEFAULT_FULL_DAEMON_ODDS_LOCK_RETRY_POLL_SECONDS = 5
DEFAULT_ODDS_CAPTURE_ONLY_T2_LOCK_RETRY_SECONDS = 90
DEFAULT_ODDS_CAPTURE_ONLY_T2_LOCK_RETRY_POLL_SECONDS = 5
FULL_DAEMON_LOCK_WAIT_MARKER_SUFFIX = ".full_daemon_waiting.json"
ODDS_CAPTURE_WINDOW_OFFSETS_MINUTES = (60, 30, 10, 2)
SERVICE_NAME = "shadow-autopilot.service"
TIMER_NAME = "shadow-autopilot.timer"
ODDS_CAPTURE_SERVICE_NAME = "shadow-autopilot-odds-capture.service"
ODDS_CAPTURE_TIMER_NAME = "shadow-autopilot-odds-capture.timer"
NO_WRITE_GUARANTEES = {
    "training": False,
    "production_promotion": False,
    "registry_mutation": False,
    "production_pointer_update": False,
    "active_model_replacement": False,
    "db_write": False,
    "label_write": False,
    "tgr_enabled": False,
    "betting_action": False,
    "ev_action": False,
    "production_prediction_overwrite": False,
    "snapshot_rewrite": False,
    "outside_shadow_manifest_rewrite": False,
    "schema_change": False,
    "hyperparameter_change": False,
    "calibration_method_change": False,
    "champion_modification": False,
}
PROTECTED_PATHS = (
    ROOT / "greyhound_racing_data.db",
    ROOT / "greyhound_racing_data_writable.db",
    ROOT / "model_registry/best_metadata.json",
    ROOT / "docs/model_contracts/v4_feature_contract.json",
    ROOT / "model_registry/current_production.json",
    ROOT / "processed_manifest.json",
    ROOT / "artifacts/prediction_snapshots/manifest.jsonl",
)
RUNNER_RELATED_UNSAFE_REASONS = {
    "duplicate_official_boxes",
    "missing_predicted_boxes_in_official_result",
    "extra_official_non_scratch_boxes_outside_prediction_set",
    "dog_name_mismatch_after_exact_badge_stripping",
}


class LockBusy(RuntimeError):
    def __init__(self, payload: Mapping[str, Any]):
        super().__init__("shadow_autopilot_daemon_lock_busy")
        self.payload = dict(payload)


def now_id(now: datetime | None = None) -> str:
    return (now or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def parse_datetime_value(value: Any, *, default_tz: Any | None = None) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None and default_tz is not None:
        parsed = parsed.replace(tzinfo=default_tz)
    return parsed


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        logical = path if path.is_absolute() else ROOT / path
        return logical.absolute().relative_to(ROOT.absolute()).as_posix()
    except ValueError:
        return str(path)


def initial_daemon_run_report(
    *,
    run_id: str,
    generated_at: datetime,
    current_time: str,
    output_dir: Path,
    lock_path: Path,
    state_path: Path,
    odds_capture_state_path: Path | None,
    autonomous_odds_capture_enabled: bool,
    autonomous_result_capture_enabled: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "shadow_autopilot_daemon_run_v1",
        "run_id": run_id,
        "generated_at": generated_at.isoformat(),
        "current_time": current_time,
        "output_dir": relpath(output_dir),
        "status": "RUNNING",
        "final_verdict": "DAEMON_RUNNING",
        "runtime_action": "FULL_DAEMON_IN_PROGRESS",
        "readiness_decision": "IN_PROGRESS",
        "lock_path": relpath(lock_path),
        "state_path": relpath(state_path),
        "odds_capture_state_path": relpath(odds_capture_state_path),
        "autonomous_odds_capture_enabled": autonomous_odds_capture_enabled,
        "autonomous_result_capture_enabled": autonomous_result_capture_enabled,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def lock_held_daemon_run_report(
    *,
    run_id: str,
    generated_at: datetime,
    current_time: str,
    output_dir: Path,
    lock_path: Path,
    lock_details: Mapping[str, Any],
    odds_capture_state_path: Path | None = None,
    odds_capture_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    odds_capture_state = (
        odds_capture_state if isinstance(odds_capture_state, Mapping) else {}
    )
    return {
        "schema_version": "shadow_autopilot_daemon_run_v1",
        "run_id": run_id,
        "generated_at": generated_at.isoformat(),
        "current_time": current_time,
        "output_dir": relpath(output_dir),
        "status": "SKIPPED_LOCK_HELD",
        "final_verdict": "PARTIAL_DAEMONIZATION",
        "runtime_action": "SKIP_LOCK_HELD",
        "readiness_decision": "WAIT_FOR_ACTIVE_DAEMON",
        "lock_path": relpath(lock_path),
        "lock_validation_status": "SKIPPED_LOCK_HELD",
        **lock_owner_report_fields(lock_details),
        "lock_reason": lock_details.get("reason"),
        "lock_details": dict(lock_details),
        "odds_capture_state_path": relpath(odds_capture_state_path),
        "last_odds_capture_run_id": odds_capture_state.get("run_id"),
        "last_odds_capture_final_status": odds_capture_state.get("final_status"),
        "last_odds_capture_status": odds_capture_state.get("odds_capture_status"),
        "last_odds_capture_operator_status": odds_capture_state.get("status"),
        "last_odds_capture_runtime_action": odds_capture_state.get("runtime_action"),
        "last_odds_capture_readiness_decision": odds_capture_state.get(
            "readiness_decision"
        ),
        "last_odds_capture_inserted_live_odds_rows": int_or_zero(
            odds_capture_state.get("inserted_live_odds_rows")
        ),
        "last_odds_capture_ready_count": int_or_zero(
            odds_capture_state.get("ready_count")
        ),
        "last_odds_capture_status_counts": dict(
            odds_capture_state.get("status_counts") or {}
        ),
        "last_odds_capture_blocked_attempt_count": int_or_zero(
            odds_capture_state.get("blocked_attempt_count")
        ),
        "last_odds_capture_next_meaningful_action": odds_capture_state.get(
            "next_meaningful_action"
        ),
        "last_odds_capture_next_meaningful_action_at": odds_capture_state.get(
            "next_meaningful_action_at"
        ),
        "protected_paths_unchanged_or_allowed": True,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def completed_daemon_run_report_envelope(
    *,
    run_id: str,
    generated_at: datetime,
    current_time: str,
    output_dir: Path,
    final_verdict: str,
) -> dict[str, Any]:
    return {
        "schema_version": "shadow_autopilot_daemon_run_v1",
        "run_id": run_id,
        "generated_at": generated_at.isoformat(),
        "current_time": current_time,
        "output_dir": relpath(output_dir),
        "status": final_verdict,
        "final_verdict": final_verdict,
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def load_json_value(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_jsonl_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def rooted_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def daily_shadow_run_from_autopilot(
    autopilot_output_dir: Path | None,
) -> tuple[Path | None, dict[str, Any] | None]:
    if autopilot_output_dir is None:
        return None, None
    autopilot_manifest = load_json(autopilot_output_dir / "run_manifest.json")
    daily_dir_text = ((autopilot_manifest or {}).get("source_artifacts") or {}).get(
        "daily_shadow_run_dir"
    )
    daily_shadow_run_dir = rooted_path(daily_dir_text)
    if daily_shadow_run_dir is None:
        return None, None
    return daily_shadow_run_dir, load_json(daily_shadow_run_dir / "shadow_manifest.json")


def timing_aligned_rerun_source_paths_from_autopilot(
    autopilot_output_dir: Path | None,
) -> dict[str, Path | None]:
    if autopilot_output_dir is None:
        return {
            "timing_aligned_rerun_plan": None,
            "timing_aligned_rerun_execution_status": None,
        }
    plan_path = autopilot_output_dir / "timing_aligned_prediction_rerun_plan.json"
    execution_path = (
        autopilot_output_dir / "timing_aligned_prediction_rerun_execution_status.json"
    )
    return {
        "timing_aligned_rerun_plan": plan_path if plan_path.exists() else None,
        "timing_aligned_rerun_execution_status": (
            execution_path if execution_path.exists() else None
        ),
    }


def timing_aligned_rerun_source_artifacts_from_autopilot(
    autopilot_output_dir: Path | None,
) -> dict[str, str]:
    paths = timing_aligned_rerun_source_paths_from_autopilot(autopilot_output_dir)
    return {key: relpath(path) for key, path in paths.items() if path is not None}


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes(paths: Sequence[Path] = PROTECTED_PATHS) -> dict[str, str | None]:
    return {relpath(path) or str(path): sha256_file(path) for path in paths}


def assert_output_dir_safe(
    output_dir: Path,
    *,
    evidence_root: Path | None = None,
) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    candidate = logical.absolute()
    try:
        relative = candidate.relative_to(ROOT.absolute())
    except ValueError as exc:
        if evidence_root is None:
            raise ValueError("output_dir_must_be_inside_repo") from exc
    else:
        if ".." in relative.parts:
            raise ValueError("output_dir_must_not_contain_parent_traversal")
        if relative.as_posix().startswith(OUTPUT_PREFIX):
            return candidate
        raise ValueError(f"output_dir_must_be_shadow_autopilot_daemon_artifact:{relative}")

    evidence_base = evidence_root if evidence_root.is_absolute() else ROOT / evidence_root
    try:
        relative = candidate.relative_to(evidence_base.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo_or_evidence_root") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if relative.parts and relative.parts[0].startswith(OUTPUT_ARTIFACT_PREFIX):
        return candidate
    raise ValueError(f"output_dir_must_be_shadow_autopilot_daemon_artifact:{relative}")


def unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = Path(f"{base}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"output_dir_collision_exhausted:{base}")


def latest_artifact(root: Path, prefix: str, required_file: str) -> Path | None:
    candidates = [
        item
        for item in root.glob(f"{prefix}*")
        if item.is_dir() and (item / required_file).exists()
    ]
    return sorted(candidates)[-1] if candidates else None


def pid_running(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_lock(lock_path: Path) -> dict[str, Any] | None:
    return load_json(lock_path)


def lock_stale_reason(payload: Mapping[str, Any] | None, *, stale_after_seconds: int) -> str | None:
    if not payload:
        return "unreadable_lock_payload"
    pid = payload.get("pid")
    if not pid_running(pid):
        return "lock_pid_not_running"
    started_at = payload.get("started_at")
    try:
        started = datetime.fromisoformat(str(started_at))
    except ValueError:
        return None
    age = (datetime.now().astimezone() - started).total_seconds()
    if age > stale_after_seconds and not pid_running(pid):
        return "stale_lock_pid_not_running"
    return None


def acquire_lock(
    *,
    lock_path: Path,
    run_id: str,
    stale_after_seconds: int,
    output_dir: Path,
) -> dict[str, Any]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "shadow_autopilot_daemon_lock_v1",
        "run_id": run_id,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at": datetime.now().astimezone().isoformat(),
        "output_dir": relpath(output_dir),
    }
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = read_lock(lock_path)
            reason = lock_stale_reason(existing, stale_after_seconds=stale_after_seconds)
            if reason:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            raise LockBusy(
                {
                    "lock_path": relpath(lock_path),
                    "existing_lock": existing,
                    "reason": "active_lock_present",
                }
            )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return payload


def lock_owner_is_odds_capture(lock_details: Mapping[str, Any] | None) -> bool:
    if not isinstance(lock_details, Mapping):
        return False
    existing = lock_details.get("existing_lock")
    if not isinstance(existing, Mapping):
        return False
    return str(existing.get("run_id") or "").endswith("_odds_capture")


def lock_owner_is_full_daemon(lock_details: Mapping[str, Any] | None) -> bool:
    return lock_owner_report_fields(lock_details).get("lock_owner_kind") == "full_daemon"


def lock_owner_report_fields(lock_details: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(lock_details, Mapping):
        return {
            "lock_owner_kind": None,
            "lock_owner_run_id": None,
            "lock_owner_output_dir": None,
            "lock_owner_pid": None,
            "lock_owner_started_at": None,
            "lock_owner_hostname": None,
        }
    wait_marker = lock_details.get("full_daemon_wait_marker")
    if isinstance(wait_marker, Mapping):
        return {
            "lock_owner_kind": "full_daemon_waiting_for_odds_capture_handoff",
            "lock_owner_run_id": wait_marker.get("run_id"),
            "lock_owner_output_dir": wait_marker.get("output_dir"),
            "lock_owner_pid": wait_marker.get("pid"),
            "lock_owner_started_at": wait_marker.get("started_at"),
            "lock_owner_hostname": wait_marker.get("hostname"),
        }
    existing = lock_details.get("existing_lock")
    if not isinstance(existing, Mapping):
        existing = {}
    run_id = str(existing.get("run_id") or "")
    output_dir = str(existing.get("output_dir") or "")
    if run_id.endswith("_odds_capture"):
        kind = "odds_capture"
    elif "shadow_autopilot_daemonization_v1_" in output_dir:
        kind = "full_daemon"
    elif run_id:
        kind = "unknown_lock_owner"
    else:
        kind = None
    return {
        "lock_owner_kind": kind,
        "lock_owner_run_id": existing.get("run_id"),
        "lock_owner_output_dir": existing.get("output_dir"),
        "lock_owner_pid": existing.get("pid"),
        "lock_owner_started_at": existing.get("started_at"),
        "lock_owner_hostname": existing.get("hostname"),
    }


def full_daemon_lock_wait_marker_path(lock_path: Path) -> Path:
    return lock_path.with_name(lock_path.name + FULL_DAEMON_LOCK_WAIT_MARKER_SUFFIX)


def read_active_full_daemon_lock_wait_marker(lock_path: Path) -> dict[str, Any] | None:
    marker_path = full_daemon_lock_wait_marker_path(lock_path)
    marker = load_json(marker_path)
    if marker is None:
        return None
    if not pid_running(marker.get("pid")):
        try:
            marker_path.unlink()
        except FileNotFoundError:
            pass
        return None
    marker["marker_path"] = relpath(marker_path)
    return marker


def write_full_daemon_lock_wait_marker(
    *,
    lock_path: Path,
    run_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    marker_path = full_daemon_lock_wait_marker_path(lock_path)
    marker = {
        "schema_version": "shadow_autopilot_full_daemon_lock_wait_marker_v1",
        "run_id": run_id,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at": datetime.now().astimezone().isoformat(),
        "lock_path": relpath(lock_path),
        "marker_path": relpath(marker_path),
        "output_dir": relpath(output_dir),
        "reason": "full_daemon_waiting_for_odds_capture_lock_handoff",
    }
    write_json(marker_path, marker)
    return marker


def write_full_daemon_lock_wait_report(
    *,
    output_dir: Path,
    lock_path: Path,
    lock_details: Mapping[str, Any],
    first_lock: Mapping[str, Any] | None,
    attempt_count: int,
    waited_seconds: float,
    retry_seconds: int,
    poll_seconds: int,
) -> dict[str, Any]:
    report_path = output_dir / "daemon_run_report.json"
    base_report = load_json(report_path)
    if not isinstance(base_report, Mapping):
        base_report = {}
    report = {
        **dict(base_report),
        "status": "WAITING_LOCK_HELD",
        "final_verdict": "DAEMON_WAITING_FOR_ODDS_CAPTURE_LOCK",
        "runtime_action": "WAIT_FOR_ODDS_CAPTURE_LOCK_HANDOFF",
        "readiness_decision": "ODDS_CAPTURE_IN_PROGRESS",
        "lock_path": relpath(lock_path),
        "lock_validation_status": "WAITING_LOCK_HELD",
        **lock_owner_report_fields(lock_details),
        "lock_reason": lock_details.get("reason"),
        "lock_retry": {
            "schema_version": "shadow_autopilot_full_daemon_lock_retry_v1",
            "status": "WAITING_FOR_ODDS_CAPTURE_LOCK",
            "attempt_count": attempt_count,
            "waited_seconds": waited_seconds,
            "retry_seconds": retry_seconds,
            "poll_seconds": poll_seconds,
            "retried_for_odds_capture_lock": True,
            "first_lock": dict(first_lock or {}),
            "last_lock": dict(lock_details),
        },
    }
    write_json(report_path, report)
    return report


def remove_full_daemon_lock_wait_marker(*, lock_path: Path, run_id: str) -> None:
    marker_path = full_daemon_lock_wait_marker_path(lock_path)
    marker = load_json(marker_path)
    if marker is None:
        return
    if marker.get("run_id") != run_id and pid_running(marker.get("pid")):
        return
    try:
        marker_path.unlink()
    except FileNotFoundError:
        pass


def acquire_lock_with_odds_capture_retry(
    *,
    lock_path: Path,
    run_id: str,
    stale_after_seconds: int,
    output_dir: Path,
    retry_seconds: int | None = None,
    poll_seconds: int | None = None,
) -> dict[str, Any]:
    retry_seconds = (
        DEFAULT_FULL_DAEMON_ODDS_LOCK_RETRY_SECONDS
        if retry_seconds is None
        else retry_seconds
    )
    poll_seconds = (
        DEFAULT_FULL_DAEMON_ODDS_LOCK_RETRY_POLL_SECONDS
        if poll_seconds is None
        else poll_seconds
    )
    first_lock: dict[str, Any] | None = None
    last_lock: dict[str, Any] | None = None
    attempt_count = 0
    waited_seconds = 0.0
    marker_written = False
    try:
        while True:
            try:
                payload = acquire_lock(
                    lock_path=lock_path,
                    run_id=run_id,
                    stale_after_seconds=stale_after_seconds,
                    output_dir=output_dir,
                )
            except LockBusy as exc:
                attempt_count += 1
                last_lock = dict(exc.payload)
                if first_lock is None:
                    first_lock = dict(exc.payload)
                should_retry = (
                    retry_seconds > 0
                    and poll_seconds > 0
                    and lock_owner_is_odds_capture(last_lock)
                    and waited_seconds < retry_seconds
                )
                if not should_retry:
                    retry_last_lock = dict(last_lock)
                    retry_details = {
                        "schema_version": "shadow_autopilot_full_daemon_lock_retry_v1",
                        "status": "GAVE_UP_LOCK_HELD",
                        "attempt_count": attempt_count,
                        "waited_seconds": waited_seconds,
                        "retry_seconds": retry_seconds,
                        "poll_seconds": poll_seconds,
                        "retried_for_odds_capture_lock": lock_owner_is_odds_capture(
                            last_lock
                        ),
                        "first_lock": first_lock,
                        "last_lock": retry_last_lock,
                    }
                    last_lock["lock_retry"] = retry_details
                    raise LockBusy(last_lock)
                if not marker_written:
                    write_full_daemon_lock_wait_marker(
                        lock_path=lock_path,
                        run_id=run_id,
                        output_dir=output_dir,
                    )
                    marker_written = True
                write_full_daemon_lock_wait_report(
                    output_dir=output_dir,
                    lock_path=lock_path,
                    lock_details=last_lock,
                    first_lock=first_lock,
                    attempt_count=attempt_count,
                    waited_seconds=waited_seconds,
                    retry_seconds=retry_seconds,
                    poll_seconds=poll_seconds,
                )
                sleep_for = min(float(poll_seconds), float(retry_seconds) - waited_seconds)
                time.sleep(sleep_for)
                waited_seconds += sleep_for
                continue
            if attempt_count:
                payload["lock_retry"] = {
                    "schema_version": "shadow_autopilot_full_daemon_lock_retry_v1",
                    "status": "ACQUIRED_AFTER_ODDS_CAPTURE_WAIT",
                    "attempt_count": attempt_count + 1,
                    "waited_seconds": waited_seconds,
                    "retry_seconds": retry_seconds,
                    "poll_seconds": poll_seconds,
                    "first_lock": first_lock,
                    "last_lock": last_lock,
                }
            return payload
    finally:
        if marker_written:
            remove_full_daemon_lock_wait_marker(lock_path=lock_path, run_id=run_id)


def t2_due_lock_retry_window(
    fixed_window_schedule: Mapping[str, Any] | None,
) -> dict[str, Any]:
    schedule = fixed_window_schedule if isinstance(fixed_window_schedule, Mapping) else {}
    t2_window: Mapping[str, Any] = {}
    windows = schedule.get("windows")
    if isinstance(windows, Sequence) and not isinstance(windows, (str, bytes)):
        for row in windows:
            if not isinstance(row, Mapping):
                continue
            try:
                offset = int(row.get("offset_minutes"))
            except (TypeError, ValueError):
                continue
            if offset == 2:
                t2_window = row
                break
    active = (
        str(t2_window.get("status") or "").upper() == "DUE"
        or (
            schedule.get("next_meaningful_action") == "RUN_ODDS_CAPTURE_NOW"
            and schedule.get("next_meaningful_action_offset_minutes") == 2
        )
    )
    minutes_to_jump = t2_window.get("minutes_to_jump")
    if not isinstance(minutes_to_jump, (int, float)):
        minutes_to_jump = None
    seconds_to_jump = (
        max(0.0, float(minutes_to_jump) * 60.0)
        if minutes_to_jump is not None
        else None
    )
    if seconds_to_jump is not None and seconds_to_jump <= 1.0:
        active = False
    return {
        "schema_version": "shadow_autopilot_odds_capture_t2_lock_retry_window_v1",
        "active": bool(active),
        "race_id": schedule.get("race_id"),
        "target_capture_at": t2_window.get("target_capture_at"),
        "minutes_to_jump": minutes_to_jump,
        "seconds_to_jump": seconds_to_jump,
        "window_status": t2_window.get("status"),
    }


def acquire_lock_with_t2_due_retry(
    *,
    lock_path: Path,
    run_id: str,
    stale_after_seconds: int,
    output_dir: Path,
    fixed_window_schedule: Mapping[str, Any] | None,
    retry_seconds: int | None = None,
    poll_seconds: int | None = None,
) -> dict[str, Any]:
    retry_seconds = (
        DEFAULT_ODDS_CAPTURE_ONLY_T2_LOCK_RETRY_SECONDS
        if retry_seconds is None
        else retry_seconds
    )
    poll_seconds = (
        DEFAULT_ODDS_CAPTURE_ONLY_T2_LOCK_RETRY_POLL_SECONDS
        if poll_seconds is None
        else poll_seconds
    )
    retry_window = t2_due_lock_retry_window(fixed_window_schedule)
    seconds_to_jump = retry_window.get("seconds_to_jump")
    if isinstance(seconds_to_jump, (int, float)):
        retry_seconds = min(float(retry_seconds), max(0.0, float(seconds_to_jump) - 1.0))
    else:
        retry_seconds = float(retry_seconds)
    poll_seconds = float(poll_seconds)
    first_lock: dict[str, Any] | None = None
    last_lock: dict[str, Any] | None = None
    attempt_count = 0
    waited_seconds = 0.0
    while True:
        try:
            payload = acquire_lock(
                lock_path=lock_path,
                run_id=run_id,
                stale_after_seconds=stale_after_seconds,
                output_dir=output_dir,
            )
        except LockBusy as exc:
            attempt_count += 1
            last_lock = dict(exc.payload)
            if first_lock is None:
                first_lock = dict(last_lock)
            should_retry = (
                bool(retry_window.get("active"))
                and retry_seconds > 0
                and poll_seconds > 0
                and waited_seconds < retry_seconds
                and lock_owner_is_full_daemon(last_lock)
            )
            if not should_retry:
                retry_details = {
                    "schema_version": "shadow_autopilot_odds_capture_t2_lock_retry_v1",
                    "status": "GAVE_UP_T2_DUE_LOCK_HELD",
                    "attempt_count": attempt_count,
                    "waited_seconds": waited_seconds,
                    "retry_seconds": retry_seconds,
                    "poll_seconds": poll_seconds,
                    "retried_for_t2_due_lock": bool(retry_window.get("active")),
                    "retry_window": retry_window,
                    "first_lock": first_lock,
                    "last_lock": dict(last_lock),
                }
                last_lock["lock_retry"] = retry_details
                raise LockBusy(last_lock)
            sleep_for = min(poll_seconds, retry_seconds - waited_seconds)
            write_json(
                output_dir / "odds_capture_t2_lock_retry.json",
                {
                    "schema_version": "shadow_autopilot_odds_capture_t2_lock_retry_v1",
                    "status": "WAITING_FOR_FULL_DAEMON_LOCK_DURING_T2",
                    "attempt_count": attempt_count,
                    "waited_seconds": waited_seconds,
                    "retry_seconds": retry_seconds,
                    "poll_seconds": poll_seconds,
                    "retried_for_t2_due_lock": True,
                    "retry_window": retry_window,
                    "first_lock": first_lock,
                    "last_lock": dict(last_lock),
                    "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
                },
            )
            time.sleep(sleep_for)
            waited_seconds += sleep_for
            continue
        if attempt_count:
            payload["lock_retry"] = {
                "schema_version": "shadow_autopilot_odds_capture_t2_lock_retry_v1",
                "status": "ACQUIRED_AFTER_T2_DUE_LOCK_WAIT",
                "attempt_count": attempt_count + 1,
                "waited_seconds": waited_seconds,
                "retry_seconds": retry_seconds,
                "poll_seconds": poll_seconds,
                "retried_for_t2_due_lock": True,
                "retry_window": retry_window,
                "first_lock": first_lock,
                "last_lock": last_lock,
            }
        return payload


def release_lock(lock_path: Path, run_id: str) -> dict[str, Any]:
    payload = read_lock(lock_path)
    if payload and payload.get("run_id") != run_id:
        return {"released": False, "reason": "lock_owned_by_other_run", "lock": payload}
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return {"released": False, "reason": "lock_already_missing"}
    return {"released": True, "reason": "released_by_owner"}


def probe_duplicate_lock(lock_path: Path, *, stale_after_seconds: int, output_dir: Path) -> dict[str, Any]:
    try:
        acquire_lock(
            lock_path=lock_path,
            run_id="duplicate_probe",
            stale_after_seconds=stale_after_seconds,
            output_dir=output_dir,
        )
    except LockBusy as exc:
        return {"status": "PASS", "duplicate_acquire_blocked": True, "details": exc.payload}
    release_lock(lock_path, "duplicate_probe")
    return {"status": "FAIL", "duplicate_acquire_blocked": False}


def probe_stale_lock_cleanup(output_dir: Path) -> dict[str, Any]:
    probe_path = output_dir / "validation" / "stale_probe.lock"
    write_json(
        probe_path,
        {
            "schema_version": "shadow_autopilot_daemon_lock_v1",
            "run_id": "stale_probe",
            "pid": 999999999,
            "started_at": "2000-01-01T00:00:00+00:00",
        },
    )
    acquired = acquire_lock(
        lock_path=probe_path,
        run_id="stale_probe_replacement",
        stale_after_seconds=1,
        output_dir=output_dir,
    )
    released = release_lock(probe_path, "stale_probe_replacement")
    return {
        "status": "PASS" if released.get("released") else "FAIL",
        "stale_lock_cleaned": True,
        "replacement_lock": acquired,
        "release": released,
    }


def run_command(
    *,
    name: str,
    command: Sequence[str],
    output_dir: Path,
    timeout_seconds: int,
    cwd: Path = ROOT,
) -> dict[str, Any]:
    started = datetime.now().astimezone()
    started_monotonic = time.monotonic()
    log_dir = output_dir / "logs"
    stdout_path = log_dir / f"{name}.stdout.txt"
    stderr_path = log_dir / f"{name}.stderr.txt"
    started_path = log_dir / f"{name}.started.json"
    running_path = log_dir / f"{name}.running.json"
    finished_path = log_dir / f"{name}.finished.json"
    timed_out = False
    returncode: int | None = None
    log_dir.mkdir(parents=True, exist_ok=True)
    timeout_deadline_at = started + timedelta(seconds=timeout_seconds)
    write_json(
        started_path,
        {
            "schema_version": "shadow_autopilot_daemon_step_started_v1",
            "name": name,
            "command": list(command),
            "cwd": str(cwd),
            "started_at": started.isoformat(),
            "timeout_seconds": timeout_seconds,
            "timeout_deadline_at": timeout_deadline_at.isoformat(),
            "stdout_path": relpath(stdout_path),
            "stderr_path": relpath(stderr_path),
        },
    )
    write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w",
        encoding="utf-8",
    ) as stderr_handle:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        write_json(
            running_path,
            {
                "schema_version": "shadow_autopilot_daemon_step_running_v1",
                "name": name,
                "command": list(command),
                "cwd": str(cwd),
                "started_at": started.isoformat(),
                "timeout_seconds": timeout_seconds,
                "timeout_deadline_at": timeout_deadline_at.isoformat(),
                "pid": process.pid,
                "stdout_path": relpath(stdout_path),
                "stderr_path": relpath(stderr_path),
                "status": "RUNNING",
            },
        )
        write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            stderr_handle.write(
                f"\n[TIMEOUT] command exceeded daemon step timeout of {timeout_seconds} seconds\n"
            )
            stderr_handle.flush()
            try:
                os.killpg(process.pid, signal.SIGTERM)
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stderr_handle.write("\n[TIMEOUT] SIGTERM grace exceeded; sending SIGKILL\n")
                stderr_handle.flush()
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
            if returncode == 0:
                returncode = -signal.SIGTERM
    duration = time.monotonic() - started_monotonic
    finished_at = datetime.now().astimezone()
    step_result = {
        "name": name,
        "command": list(command),
        "cwd": str(cwd),
        "started_at": started.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": duration,
        "timeout_seconds": timeout_seconds,
        "timeout_deadline_at": timeout_deadline_at.isoformat(),
        "timed_out": timed_out,
        "returncode": returncode,
        "status": "PASS" if returncode == 0 and not timed_out else "FAIL",
        "stdout_path": relpath(stdout_path),
        "stderr_path": relpath(stderr_path),
    }
    write_json(
        finished_path,
        {
            "schema_version": "shadow_autopilot_daemon_step_finished_v1",
            **step_result,
        },
    )
    write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
    return step_result


def simulate_timeout_recovery(output_dir: Path) -> dict[str, Any]:
    result = run_command(
        name="timeout_recovery_probe",
        command=[sys.executable, "-c", "import time; time.sleep(2)"],
        output_dir=output_dir,
        timeout_seconds=1,
    )
    return {
        "status": "PASS" if result.get("timed_out") is True else "FAIL",
        "timeout_enforced": result.get("timed_out") is True,
        "probe": result,
    }


def shadow_model_cli_args(shadow_model: Path | None) -> list[str]:
    if shadow_model is None:
        return []
    return ["--shadow-model", str(shadow_model)]


def optional_path_cli_args(flag: str, path: Path | None) -> list[str]:
    if path is None:
        return []
    return [flag, str(path)]


def service_file_text(
    *,
    repo_path: Path,
    timeout_seconds: int,
    python_path: Path | None = None,
    evidence_root: Path | None = None,
    shadow_model: Path | None = None,
    db_path: Path | None = None,
    lock_path: Path | None = None,
    state_path: Path | None = None,
    odds_capture_state_path: Path | None = None,
) -> str:
    script_path = repo_path / "scripts/shadow_autopilot_daemon.py"
    service_python = python_path or Path("/usr/bin/python3")
    evidence_root_segment = " ".join(optional_path_cli_args("--evidence-root", evidence_root))
    evidence_root_segment = f"{evidence_root_segment} " if evidence_root_segment else ""
    explicit_path_args = [
        *optional_path_cli_args("--db", db_path),
        *shadow_model_cli_args(shadow_model),
        *optional_path_cli_args("--lock-path", lock_path),
        *optional_path_cli_args("--state-path", state_path),
        *optional_path_cli_args("--odds-capture-state-path", odds_capture_state_path),
    ]
    explicit_path_segment = " ".join(explicit_path_args)
    explicit_path_segment = f"{explicit_path_segment} " if explicit_path_segment else ""
    systemd_timeout_seconds = max(timeout_seconds + 60, timeout_seconds * 4)
    return "\n".join(
        [
            "[Unit]",
            "Description=Greyhound shadow autopilot evidence collection",
            "Wants=network-online.target",
            "After=network-online.target",
            "",
            "[Service]",
            "Type=oneshot",
            f"WorkingDirectory={repo_path}",
            "Environment=PYTHONUNBUFFERED=1",
            "Environment=GREYHOUND_ALLOW_TGR=0",
            "Environment=PATH=/home/l4nd0/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            (
                f"ExecStart={service_python} {script_path} run-once "
                f"{evidence_root_segment}"
                "--days-ahead 1 --refresh-limit 16 "
                f"{explicit_path_segment}"
                "--enable-autonomous-odds-capture "
                "--execute-autonomous-odds-capture "
                "--allow-auto-scrape-odds "
                "--enable-autonomous-result-capture "
                "--require-safe-refresh-metadata "
                f"--rejoin-pending-limit {DEFAULT_REJOIN_PENDING_LIMIT} "
                f"--timeout-seconds {timeout_seconds}"
            ),
            f"TimeoutStartSec={systemd_timeout_seconds}",
            "Nice=10",
            "IOSchedulingClass=best-effort",
            "",
        ]
    )


def timer_file_text() -> str:
    return "\n".join(
        [
            "[Unit]",
            "Description=Run greyhound shadow autopilot every 15 minutes",
            "",
            "[Timer]",
            f"OnCalendar={DEFAULT_TIMER_ON_CALENDAR}",
            f"AccuracySec={DEFAULT_TIMER_ACCURACY}",
            "Persistent=true",
            "Unit=shadow-autopilot.service",
            "",
            "[Install]",
            "WantedBy=timers.target",
            "",
        ]
    )


def odds_capture_service_file_text(
    *,
    repo_path: Path,
    timeout_seconds: int,
    python_path: Path | None = None,
    evidence_root: Path | None = None,
    db_path: Path | None = None,
    lock_path: Path | None = None,
    state_path: Path | None = None,
    refresh_limit: int = DEFAULT_ODDS_CAPTURE_ONLY_REFRESH_LIMIT,
) -> str:
    script_path = repo_path / "scripts/shadow_autopilot_daemon.py"
    service_python = python_path or Path("/usr/bin/python3")
    evidence_root_segment = " ".join(optional_path_cli_args("--evidence-root", evidence_root))
    evidence_root_segment = f"{evidence_root_segment} " if evidence_root_segment else ""
    explicit_path_args = [
        *optional_path_cli_args("--db", db_path),
        *optional_path_cli_args("--lock-path", lock_path),
        *optional_path_cli_args("--state-path", state_path),
    ]
    explicit_path_segment = " ".join(explicit_path_args)
    explicit_path_segment = f"{explicit_path_segment} " if explicit_path_segment else ""
    systemd_timeout_seconds = max(timeout_seconds + 60, timeout_seconds * 2)
    return "\n".join(
        [
            "[Unit]",
            "Description=Greyhound autonomous live odds capture",
            "Wants=network-online.target",
            "After=network-online.target",
            "",
            "[Service]",
            "Type=oneshot",
            f"WorkingDirectory={repo_path}",
            "Environment=PYTHONUNBUFFERED=1",
            "Environment=GREYHOUND_ALLOW_TGR=0",
            "Environment=PATH=/home/l4nd0/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            (
                f"ExecStart={service_python} {script_path} run-odds-capture-once "
                f"{evidence_root_segment}"
                "--days-ahead 1 "
                f"--refresh-limit {refresh_limit} "
                f"--odds-capture-refresh-limit {refresh_limit} "
                "--require-safe-refresh-metadata "
                "--skip-primary-refresh "
                f"{explicit_path_segment}"
                f"--timeout-seconds {timeout_seconds}"
            ),
            f"TimeoutStartSec={systemd_timeout_seconds}",
            "Nice=10",
            "IOSchedulingClass=best-effort",
            "",
        ]
    )


def odds_capture_timer_file_text() -> str:
    return "\n".join(
        [
            "[Unit]",
            "Description=Run greyhound autonomous live odds capture except full-daemon minutes",
            "",
            "[Timer]",
            f"OnCalendar={DEFAULT_ODDS_CAPTURE_ONLY_TIMER_ON_CALENDAR}",
            f"AccuracySec={DEFAULT_ODDS_CAPTURE_ONLY_TIMER_ACCURACY}",
            "Persistent=true",
            f"Unit={ODDS_CAPTURE_SERVICE_NAME}",
            "",
            "[Install]",
            "WantedBy=timers.target",
            "",
        ]
    )


def write_service_files(
    *,
    service_dir: Path = DEFAULT_SERVICE_DIR,
    repo_path: Path = Path("/home/l4nd0/greyhound_racing_collector"),
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    python_path: Path | None = None,
    evidence_root: Path | None = None,
    shadow_model: Path | None = None,
    db_path: Path | None = None,
    lock_path: Path | None = None,
    state_path: Path | None = None,
    odds_capture_state_path: Path | None = None,
) -> dict[str, Any]:
    service_dir.mkdir(parents=True, exist_ok=True)
    service_path = service_dir / SERVICE_NAME
    timer_path = service_dir / TIMER_NAME
    write_text(
        service_path,
        service_file_text(
            repo_path=repo_path,
            timeout_seconds=timeout_seconds,
            python_path=python_path or Path(sys.executable),
            evidence_root=evidence_root,
            shadow_model=shadow_model,
            db_path=db_path,
            lock_path=lock_path,
            state_path=state_path,
            odds_capture_state_path=odds_capture_state_path,
        ),
    )
    write_text(timer_path, timer_file_text())
    return {
        "service_path": relpath(service_path),
        "timer_path": relpath(timer_path),
        "timer_frequency": DEFAULT_TIMER_FREQUENCY,
        "timer_calendar": DEFAULT_TIMER_ON_CALENDAR,
        "repo_path": str(repo_path),
        "timeout_seconds": timeout_seconds,
        "systemd_timeout_start_seconds": max(timeout_seconds + 60, timeout_seconds * 4),
        "python_path": str(python_path or Path(sys.executable)),
        "evidence_root": str(evidence_root) if evidence_root is not None else None,
        "shadow_model": str(shadow_model) if shadow_model is not None else None,
        "db_path": str(db_path) if db_path is not None else None,
        "lock_path": str(lock_path) if lock_path is not None else None,
        "state_path": str(state_path) if state_path is not None else None,
        "odds_capture_state_path": (
            str(odds_capture_state_path) if odds_capture_state_path is not None else None
        ),
    }


def write_odds_capture_service_files(
    *,
    service_dir: Path = DEFAULT_SERVICE_DIR,
    repo_path: Path = Path("/home/l4nd0/greyhound_racing_collector"),
    timeout_seconds: int = DEFAULT_ODDS_CAPTURE_ONLY_TIMEOUT_SECONDS,
    python_path: Path | None = None,
    evidence_root: Path | None = None,
    db_path: Path | None = None,
    lock_path: Path | None = None,
    state_path: Path | None = None,
    refresh_limit: int = DEFAULT_ODDS_CAPTURE_ONLY_REFRESH_LIMIT,
) -> dict[str, Any]:
    service_dir.mkdir(parents=True, exist_ok=True)
    service_path = service_dir / ODDS_CAPTURE_SERVICE_NAME
    timer_path = service_dir / ODDS_CAPTURE_TIMER_NAME
    write_text(
        service_path,
        odds_capture_service_file_text(
            repo_path=repo_path,
            timeout_seconds=timeout_seconds,
            python_path=python_path or Path(sys.executable),
            evidence_root=evidence_root,
            db_path=db_path,
            lock_path=lock_path,
            state_path=state_path,
            refresh_limit=refresh_limit,
        ),
    )
    write_text(timer_path, odds_capture_timer_file_text())
    return {
        "service_path": relpath(service_path),
        "timer_path": relpath(timer_path),
        "timer_frequency": DEFAULT_ODDS_CAPTURE_ONLY_TIMER_FREQUENCY,
        "timer_calendar": DEFAULT_ODDS_CAPTURE_ONLY_TIMER_ON_CALENDAR,
        "timer_accuracy": DEFAULT_ODDS_CAPTURE_ONLY_TIMER_ACCURACY,
        "repo_path": str(repo_path),
        "timeout_seconds": timeout_seconds,
        "systemd_timeout_start_seconds": max(timeout_seconds + 60, timeout_seconds * 2),
        "python_path": str(python_path or Path(sys.executable)),
        "evidence_root": str(evidence_root) if evidence_root is not None else None,
        "db_path": str(db_path) if db_path is not None else None,
        "lock_path": str(lock_path) if lock_path is not None else None,
        "state_path": str(state_path) if state_path is not None else None,
        "refresh_limit": refresh_limit,
    }


def systemd_verify(service_path: Path, timer_path: Path, output_dir: Path) -> dict[str, Any]:
    tool = shutil.which("systemd-analyze")
    if not tool:
        return {"status": "SKIPPED", "reason": "systemd-analyze_not_available"}
    result = run_command(
        name="systemd_analyze_verify",
        command=[tool, "verify", str(service_path), str(timer_path)],
        output_dir=output_dir,
        timeout_seconds=30,
    )
    return {
        "status": "PASS" if result.get("returncode") == 0 else "FAIL",
        "command_result": result,
    }


def _parse_systemctl_show(stdout: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def systemd_unit_status(
    unit_name: str,
    *,
    systemctl_path: str | None = None,
    runner: Any = subprocess.run,
    scope: str = "user",
) -> dict[str, Any]:
    """Read one systemd unit status without mutating the host."""

    tool = systemctl_path or shutil.which("systemctl")
    if not tool:
        return {
            "unit": unit_name,
            "status": "SYSTEMCTL_UNAVAILABLE",
            "systemctl_available": False,
            "scope": scope,
        }
    command = [tool]
    if scope == "user":
        command.append("--user")
    elif scope != "system":
        return {
            "unit": unit_name,
            "status": "SYSTEMCTL_INVALID_SCOPE",
            "systemctl_available": True,
            "scope": scope,
        }
    command.extend(
        [
            "show",
            unit_name,
            "--property=LoadState",
            "--property=ActiveState",
            "--property=UnitFileState",
            "--property=FragmentPath",
            "--property=ExecStart",
            "--no-pager",
        ]
    )
    try:
        result = runner(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return {
            "unit": unit_name,
            "status": "SYSTEMCTL_QUERY_FAILED",
            "systemctl_available": True,
            "command": command,
            "scope": scope,
            "error": repr(exc),
        }
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    values = _parse_systemctl_show(stdout)
    load_state = values.get("LoadState")
    active_state = values.get("ActiveState")
    unit_file_state = values.get("UnitFileState")
    fragment_path = values.get("FragmentPath")
    exec_start = values.get("ExecStart")
    loaded = load_state == "loaded"
    enabled = unit_file_state == "enabled"
    active = active_state == "active"
    if getattr(result, "returncode", 1) != 0:
        status = "SYSTEMCTL_QUERY_FAILED"
    elif not loaded:
        status = "NOT_LOADED"
    elif active:
        status = "ACTIVE"
    elif enabled:
        status = "LOADED_ENABLED_INACTIVE"
    else:
        status = "LOADED_INACTIVE"
    return {
        "unit": unit_name,
        "status": status,
        "scope": scope,
        "systemctl_available": True,
        "command": command,
        "returncode": getattr(result, "returncode", None),
        "load_state": load_state,
        "active_state": active_state,
        "unit_file_state": unit_file_state,
        "fragment_path": fragment_path,
        "exec_start": exec_start,
        "loaded": loaded,
        "enabled": enabled,
        "active": active,
        "stdout": stdout,
        "stderr": stderr,
    }


def systemd_deployment_status(
    *,
    service_name: str = SERVICE_NAME,
    timer_name: str = TIMER_NAME,
    systemctl_path: str | None = None,
    runner: Any = subprocess.run,
    expected_service_exec_fragments: Sequence[str] | None = None,
    scope: str = "user",
) -> dict[str, Any]:
    """Summarize whether the report-only daemon timer is actually deployed."""

    service = systemd_unit_status(
        service_name,
        systemctl_path=systemctl_path,
        runner=runner,
        scope=scope,
    )
    timer = systemd_unit_status(
        timer_name,
        systemctl_path=systemctl_path,
        runner=runner,
        scope=scope,
    )
    systemctl_available = bool(
        service.get("systemctl_available") and timer.get("systemctl_available")
    )
    service_loaded = bool(service.get("loaded"))
    timer_loaded = bool(timer.get("loaded"))
    timer_enabled = bool(timer.get("enabled"))
    timer_active = bool(timer.get("active"))
    expected_fragments = [str(fragment) for fragment in (expected_service_exec_fragments or [])]
    service_exec_start = str(service.get("exec_start") or "")
    missing_service_exec_fragments = [
        fragment for fragment in expected_fragments if fragment not in service_exec_start
    ]
    service_command_matches_expected = not missing_service_exec_fragments
    deployment_ready = (
        systemctl_available
        and service_loaded
        and timer_loaded
        and timer_enabled
        and timer_active
        and service_command_matches_expected
    )
    if not systemctl_available:
        deployment_status = "SYSTEMCTL_UNAVAILABLE"
    elif deployment_ready:
        deployment_status = "INSTALLED_AND_ACTIVE"
    elif (
        service_loaded
        and timer_loaded
        and timer_enabled
        and timer_active
        and not service_command_matches_expected
    ):
        deployment_status = "INSTALLED_COMMAND_MISMATCH"
    elif service_loaded or timer_loaded:
        deployment_status = "INSTALLED_NOT_ACTIVE"
    else:
        deployment_status = "NOT_INSTALLED"
    return {
        "schema_version": "shadow_autopilot_systemd_deployment_status_v1",
        "scope": scope,
        "deployment_status": deployment_status,
        "deployment_ready": deployment_ready,
        "service_installed": service_loaded,
        "timer_installed": timer_loaded,
        "timer_enabled": timer_enabled,
        "timer_active": timer_active,
        "service_command_matches_expected": service_command_matches_expected,
        "required_service_exec_fragments": expected_fragments,
        "missing_service_exec_fragments": missing_service_exec_fragments,
        "service_unit": service,
        "timer_unit": timer,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def expected_service_exec_fragments_for_run(args: argparse.Namespace) -> list[str]:
    fragments = [
        "shadow_autopilot_daemon.py run-once",
        "--db",
        str(args.db),
        "--enable-autonomous-odds-capture",
        "--execute-autonomous-odds-capture",
        "--allow-auto-scrape-odds",
        "--enable-autonomous-result-capture",
        "--require-safe-refresh-metadata",
    ]
    fragments.extend(shadow_model_cli_args(args.shadow_model))
    fragments.extend(optional_path_cli_args("--lock-path", args.lock_path))
    fragments.extend(optional_path_cli_args("--state-path", args.state_path))
    fragments.extend(
        optional_path_cli_args("--odds-capture-state-path", args.odds_capture_state_path)
    )
    return fragments


def odds_capture_only_autopilot_command(
    *,
    run_id: str,
    evidence_root: Path,
    current_time: str,
    db_path: Path,
    days_ahead: int,
    refresh_limit: int,
    odds_capture_min_minutes: float,
    odds_capture_max_minutes: float,
    odds_capture_refresh_limit: int,
    timeout_seconds: int,
    refresh_command_mode: str = "auto",
    require_safe_refresh_metadata: bool = True,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/shadow_autopilot_v1.py"),
        "--run-id",
        run_id,
        "--evidence-root",
        str(evidence_root),
        "--current-time",
        current_time,
        "--db",
        str(db_path),
        "--days-ahead",
        str(days_ahead),
        "--refresh-limit",
        str(refresh_limit),
        "--refresh-command-mode",
        refresh_command_mode,
        "--odds-capture-min-minutes",
        str(odds_capture_min_minutes),
        "--odds-capture-max-minutes",
        str(odds_capture_max_minutes),
        "--odds-capture-refresh-limit",
        str(odds_capture_refresh_limit),
        "--step-timeout-seconds",
        str(timeout_seconds),
        "--enable-autonomous-odds-capture",
        "--execute-autonomous-odds-capture",
        "--allow-auto-scrape-odds",
        "--skip-primary-refresh",
        "--skip-shadow-run",
        "--skip-odds-snapshot",
        "--skip-result-join",
        "--skip-aggregate",
        "--skip-status",
        "--skip-unified-dataset",
    ]
    if require_safe_refresh_metadata:
        command.append("--require-safe-refresh-metadata")
    return command


def pre_race_gated_challenger_command(
    *,
    runner_matrix_csv: Path,
    output_dir: Path,
    rank_first_hypotheses_json: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/build_pre_race_gated_challenger_packet.py"),
        "--runner-matrix-csv",
        str(runner_matrix_csv),
        "--output-dir",
        str(output_dir),
    ]
    if rank_first_hypotheses_json is not None:
        command.extend(["--rank-first-hypotheses-json", str(rank_first_hypotheses_json)])
    return command


def time_split_gated_challenger_command(
    *,
    runner_matrix_csv: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/build_time_split_gated_challenger_packet.py"),
        "--runner-matrix-csv",
        str(runner_matrix_csv),
        "--output-dir",
        str(output_dir),
    ]


def market_residual_challenger_command(
    *,
    runner_matrix_csv: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/build_market_residual_challenger_packet.py"),
        "--runner-matrix-csv",
        str(runner_matrix_csv),
        "--output-dir",
        str(output_dir),
    ]


def market_residual_regime_audit_command(
    *,
    runner_matrix_csv: Path,
    race_predictions_csv: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/build_market_residual_regime_audit.py"),
        "--runner-matrix-csv",
        str(runner_matrix_csv),
        "--race-predictions-csv",
        str(race_predictions_csv),
        "--output-dir",
        str(output_dir),
    ]


def promotion_distance_report_command(
    *,
    rolling_report: Path,
    pre_race_gated_report: Path,
    high_accuracy_gate: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/build_promotion_distance_report.py"),
        "--rolling-report",
        str(rolling_report),
        "--pre-race-gated-report",
        str(pre_race_gated_report),
        "--high-accuracy-gate",
        str(high_accuracy_gate),
        "--output-dir",
        str(output_dir),
    ]


def rank_first_hypothesis_watchlist_command(
    *,
    evidence_root: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/build_rank_first_hypothesis_watchlist.py"),
        "--evidence-root",
        str(evidence_root),
        "--output-dir",
        str(output_dir),
    ]


def gated_challenger_status_from_report(
    *,
    generated_at: datetime,
    packet_kind: str,
    packet_dir: Path | None,
    report_path: Path | None,
    packet_report: Mapping[str, Any] | None,
    skipped_reason: str | None = None,
    attempted: bool = False,
    returncode: int | None = None,
) -> dict[str, Any]:
    report = packet_report or {}
    promotion_gate = report.get("promotion_gate")
    if not isinstance(promotion_gate, Mapping):
        promotion_gate = {}
    challenger_metrics = report.get("challenger_metrics")
    if not isinstance(challenger_metrics, Mapping):
        challenger_metrics = report.get("time_split_metrics")
    if not isinstance(challenger_metrics, Mapping):
        challenger_metrics = {}
    market_metrics = report.get("market_metrics")
    if not isinstance(market_metrics, Mapping):
        market_metrics = report.get("market_metrics_on_time_split_test_races")
    if not isinstance(market_metrics, Mapping):
        market_metrics = {}
    predeclared_residual_candidate = report.get("predeclared_residual_candidate")
    if not isinstance(predeclared_residual_candidate, Mapping):
        predeclared_residual_candidate = {}
    rank_first_hypothesis_review = report.get("rank_first_hypothesis_gate_review")
    if not isinstance(rank_first_hypothesis_review, Mapping):
        rank_first_hypothesis_review = {}
    rank_first_best_candidate = rank_first_hypothesis_review.get("best_candidate")
    if not isinstance(rank_first_best_candidate, Mapping):
        rank_first_best_candidate = {}
    return {
        "schema_version": "shadow_autopilot_gated_challenger_status_v1",
        "generated_at": generated_at.isoformat(),
        "packet_kind": packet_kind,
        "status": report.get("final_status")
        or ("SKIPPED" if skipped_reason else "GATED_CHALLENGER_PACKET_FAILED_NO_REPORT"),
        "output_dir": relpath(packet_dir),
        "report_path": relpath(report_path),
        "attempted": attempted,
        "returncode": returncode,
        "skipped_reason": skipped_reason,
        "matrix_row_count": int(report.get("matrix_row_count") or 0),
        "accepted_race_count": int(report.get("accepted_race_count") or 0),
        "minimum_races_for_review": int(report.get("minimum_races_for_review") or 0),
        "evaluated_fold_count": report.get("evaluated_fold_count"),
        "evaluated_split_count": report.get("evaluated_split_count"),
        "time_split_test_race_count": challenger_metrics.get("race_count"),
        "gate_triggered_test_race_count": challenger_metrics.get(
            "gate_triggered_test_race_count"
        ),
        "predeclared_residual_candidate": dict(predeclared_residual_candidate),
        "predeclared_residual_candidate_key": predeclared_residual_candidate.get(
            "candidate_key"
        ),
        "predeclared_residual_candidate_status": predeclared_residual_candidate.get(
            "status"
        ),
        "predeclared_residual_triggered_race_count": predeclared_residual_candidate.get(
            "triggered_race_count"
        ),
        "predeclared_residual_minimum_triggered_races_for_directional_read": (
            predeclared_residual_candidate.get(
                "minimum_triggered_races_for_directional_read"
            )
        ),
        "predeclared_residual_directional_read_ready": bool(
            predeclared_residual_candidate.get("directional_read_ready", False)
        ),
        "predeclared_residual_candidate_minus_market": (
            predeclared_residual_candidate.get("candidate_minus_market") or {}
        ),
        "predeclared_residual_blockers": list(
            predeclared_residual_candidate.get("blockers") or []
        ),
        "rank_first_hypothesis_gate_review": dict(rank_first_hypothesis_review),
        "rank_first_hypothesis_review_status": rank_first_hypothesis_review.get(
            "status"
        ),
        "rank_first_hypothesis_candidate_count": rank_first_hypothesis_review.get(
            "candidate_count"
        ),
        "rank_first_hypothesis_evaluated_candidate_count": (
            rank_first_hypothesis_review.get("evaluated_candidate_count")
        ),
        "rank_first_hypothesis_best_candidate_key": (
            rank_first_hypothesis_review.get("best_candidate_key")
        ),
        "rank_first_hypothesis_best_triggered_race_count": (
            rank_first_best_candidate.get("gate_triggered_race_count")
        ),
        "rank_first_hypothesis_minimum_triggered_races_for_directional_read": (
            rank_first_hypothesis_review.get(
                "minimum_triggered_races_for_directional_read"
            )
        ),
        "rank_first_hypothesis_directional_read_ready": bool(
            rank_first_hypothesis_review.get("directional_read_ready", False)
        ),
        "rank_first_hypothesis_best_candidate_minus_market": (
            rank_first_hypothesis_review.get("best_candidate_minus_market") or {}
        ),
        "rank_first_hypothesis_blockers": list(
            rank_first_hypothesis_review.get("blockers") or []
        ),
        "market_top1": market_metrics.get("top1"),
        "market_top3": market_metrics.get("top3"),
        "market_mean_winner_rank": market_metrics.get("mean_winner_rank"),
        "market_brier": market_metrics.get("brier"),
        "market_logloss": market_metrics.get("logloss"),
        "challenger_top1": challenger_metrics.get("top1"),
        "challenger_top3": challenger_metrics.get("top3"),
        "challenger_mean_winner_rank": challenger_metrics.get("mean_winner_rank"),
        "challenger_brier": challenger_metrics.get("brier"),
        "challenger_logloss": challenger_metrics.get("logloss"),
        "candidate_minus_market": promotion_gate.get("candidate_minus_market") or {},
        "would_clear_metric_gates": promotion_gate.get("would_clear_metric_gates", False),
        "promotion_ready": bool(promotion_gate.get("promotion_ready", False)),
        "promotion_blockers": list(promotion_gate.get("blockers") or []),
        "packet_blockers": list(report.get("blockers") or []),
        "no_write_guarantees": dict(report.get("no_write_guarantees") or NO_WRITE_GUARANTEES),
    }


def residual_regime_audit_status_from_report(
    *,
    generated_at: datetime,
    packet_dir: Path | None,
    report_path: Path | None,
    packet_report: Mapping[str, Any] | None,
    skipped_reason: str | None = None,
    attempted: bool = False,
    returncode: int | None = None,
) -> dict[str, Any]:
    report = packet_report or {}
    return {
        "schema_version": "shadow_autopilot_market_residual_regime_audit_status_v1",
        "generated_at": generated_at.isoformat(),
        "status": report.get("final_status")
        or ("SKIPPED" if skipped_reason else "MARKET_RESIDUAL_REGIME_AUDIT_FAILED_NO_REPORT"),
        "output_dir": relpath(packet_dir),
        "report_path": relpath(report_path),
        "attempted": attempted,
        "returncode": returncode,
        "skipped_reason": skipped_reason,
        "matrix_row_count": int(report.get("matrix_row_count") or 0),
        "accepted_race_count": int(report.get("accepted_race_count") or 0),
        "minimum_races_for_review": int(report.get("minimum_races_for_review") or 0),
        "regime_summary_count": int(report.get("regime_summary_count") or 0),
        "rank_first_hypothesis_status": report.get("rank_first_hypothesis_status"),
        "rank_first_hypothesis_blockers": list(
            report.get("rank_first_hypothesis_blockers") or []
        ),
        "pre_race_rank_first_help_regime_count": int(
            report.get("pre_race_rank_first_help_regime_count") or 0
        ),
        "pre_race_logloss_only_help_regime_count": int(
            report.get("pre_race_logloss_only_help_regime_count") or 0
        ),
        "next_hypotheses_json": report.get("next_hypotheses_json"),
        "promotion_ready": bool(report.get("promotion_ready", False)),
        "promotion_blockers": list(report.get("promotion_blockers") or []),
        "overall_metrics": dict(report.get("overall_metrics") or {}),
        "no_write_guarantees": dict(report.get("no_write_guarantees") or NO_WRITE_GUARANTEES),
    }


def promotion_distance_status_from_report(
    *,
    generated_at: datetime,
    packet_dir: Path | None,
    report_path: Path | None,
    packet_report: Mapping[str, Any] | None,
    skipped_reason: str | None = None,
    attempted: bool = False,
    returncode: int | None = None,
) -> dict[str, Any]:
    report = packet_report or {}
    rolling_sample = report.get("rolling_sample")
    if not isinstance(rolling_sample, Mapping):
        rolling_sample = {}
    market_benchmark = report.get("market_benchmark")
    if not isinstance(market_benchmark, Mapping):
        market_benchmark = {}
    residual = report.get("predeclared_residual_candidate")
    if not isinstance(residual, Mapping):
        residual = {}
    official_result_coverage = promotion_distance_official_result_coverage_summary(
        report=report,
        rolling_sample=rolling_sample,
    )
    return {
        "schema_version": "shadow_autopilot_promotion_distance_status_v1",
        "generated_at": generated_at.isoformat(),
        "status": report.get("final_status")
        or ("SKIPPED" if skipped_reason else "PROMOTION_DISTANCE_REPORT_FAILED_NO_REPORT"),
        "output_dir": relpath(packet_dir),
        "report_path": relpath(report_path),
        "attempted": attempted,
        "returncode": returncode,
        "skipped_reason": skipped_reason,
        "promotion_ready": bool(report.get("promotion_ready", False)),
        "blockers": list(report.get("blockers") or []),
        "sample_race_count": int(rolling_sample.get("sample_race_count") or 0),
        "sample_runner_rows": int(rolling_sample.get("sample_runner_rows") or 0),
        "source_exclusion_reason_counts": dict(
            rolling_sample.get("source_exclusion_reason_counts") or {}
        ),
        "source_odds_exclusion_reason_counts": dict(
            rolling_sample.get("source_odds_exclusion_reason_counts") or {}
        ),
        "source_official_result_evidence_db_missing_race_ids": list(
            rolling_sample.get("source_official_result_evidence_db_missing_race_ids")
            or []
        ),
        "source_official_result_evidence_db_requested_race_count": int(
            rolling_sample.get(
                "source_official_result_evidence_db_requested_race_count"
            )
            or 0
        ),
        "source_official_result_evidence_db_races_with_rows": list(
            rolling_sample.get("source_official_result_evidence_db_races_with_rows")
            or []
        ),
        "source_official_result_runner_paths": list(
            rolling_sample.get("source_official_result_runner_paths") or []
        ),
        "official_result_coverage": official_result_coverage,
        "official_result_coverage_requested_race_count": official_result_coverage.get(
            "requested_race_count"
        ),
        "official_result_coverage_requested_race_count_source": official_result_coverage.get(
            "requested_race_count_source"
        ),
        "official_result_coverage_legacy_requested_race_count_without_ids": official_result_coverage.get(
            "legacy_requested_race_count_without_ids"
        ),
        "official_result_coverage_races_with_rows_count": official_result_coverage.get(
            "races_with_rows_count"
        ),
        "official_result_coverage_missing_race_count": official_result_coverage.get(
            "missing_race_count"
        ),
        "official_result_coverage_missing_exclusion_count": official_result_coverage.get(
            "missing_exclusion_count"
        ),
        "official_result_runner_path_count": official_result_coverage.get(
            "runner_path_count"
        ),
        "official_result_runner_paths_source_field": official_result_coverage.get(
            "runner_paths_source_field"
        ),
        "target_top1_margin_vs_market": market_benchmark.get(
            "target_top1_margin_vs_market"
        ),
        "best_non_market_top1_margin_gap": market_benchmark.get(
            "best_non_market_top1_margin_gap"
        ),
        "predeclared_residual_candidate_status": residual.get("status"),
        "predeclared_residual_triggered_race_count": residual.get(
            "triggered_race_count"
        ),
        "predeclared_residual_minimum_triggered_races_for_directional_read": (
            residual.get("minimum_triggered_races_for_directional_read")
        ),
        "predeclared_residual_directional_read_ready": bool(
            residual.get("directional_read_ready", False)
        ),
        "predeclared_residual_candidate_minus_market": (
            residual.get("candidate_minus_market") or {}
        ),
        "no_write_guarantees": dict(report.get("no_write_guarantees") or NO_WRITE_GUARANTEES),
    }


def promotion_distance_official_result_coverage_summary(
    *,
    report: Mapping[str, Any],
    rolling_sample: Mapping[str, Any],
) -> dict[str, Any]:
    direct_coverage = report.get("official_result_coverage")
    if isinstance(direct_coverage, Mapping):
        missing_race_ids = list(direct_coverage.get("missing_race_ids") or [])
        races_with_rows = list(direct_coverage.get("races_with_rows") or [])
        return {
            "source": direct_coverage.get("source"),
            "requested_race_count": int_or_zero(
                direct_coverage.get("requested_race_count")
            ),
            "requested_race_count_source": direct_coverage.get(
                "requested_race_count_source"
            ),
            "legacy_requested_race_count_without_ids": direct_coverage.get(
                "legacy_requested_race_count_without_ids"
            ),
            "races_with_rows_count": int_or_zero(
                direct_coverage.get("races_with_rows_count")
            ),
            "missing_race_count": int_or_zero(
                direct_coverage.get("missing_race_count")
            ),
            "missing_exclusion_count": int_or_zero(
                direct_coverage.get("missing_exclusion_count")
            ),
            "missing_race_ids": missing_race_ids,
            "races_with_rows": races_with_rows,
            "runner_path_count": int_or_zero(
                direct_coverage.get("runner_path_count")
            ),
            "runner_paths_source_field": direct_coverage.get(
                "runner_paths_source_field"
            ),
        }

    missing_race_ids = list(
        rolling_sample.get("source_official_result_evidence_db_missing_race_ids")
        or []
    )
    races_with_rows = list(
        rolling_sample.get("source_official_result_evidence_db_races_with_rows") or []
    )
    runner_paths = list(rolling_sample.get("source_official_result_runner_paths") or [])
    source_exclusion_counts = dict(
        rolling_sample.get("source_exclusion_reason_counts") or {}
    )
    return {
        "source": "rolling_sample",
        "requested_race_count": int_or_zero(
            rolling_sample.get(
                "source_official_result_evidence_db_requested_race_count"
            )
        ),
        "requested_race_count_source": (
            "deduped_requested_or_inferred_race_ids"
            if rolling_sample.get("source_official_result_evidence_db_requested_race_ids")
            else None
        ),
        "legacy_requested_race_count_without_ids": rolling_sample.get(
            "source_official_result_evidence_db_legacy_requested_race_count_without_ids"
        ),
        "races_with_rows_count": len(races_with_rows),
        "missing_race_count": len(missing_race_ids),
        "missing_exclusion_count": int_or_zero(
            source_exclusion_counts.get("official_result_missing")
        ),
        "missing_race_ids": missing_race_ids,
        "races_with_rows": races_with_rows,
        "runner_path_count": len(runner_paths),
        "runner_paths_source_field": "rolling_sample.source_official_result_runner_paths",
    }


def rank_first_hypothesis_watchlist_status_from_report(
    *,
    generated_at: datetime,
    packet_dir: Path | None,
    report_path: Path | None,
    packet_report: Mapping[str, Any] | None,
    skipped_reason: str | None = None,
    attempted: bool = False,
    returncode: int | None = None,
) -> dict[str, Any]:
    report = packet_report or {}
    best = report.get("best_candidate")
    if not isinstance(best, Mapping):
        best = {}
    return {
        "schema_version": "shadow_autopilot_rank_first_hypothesis_watchlist_status_v1",
        "generated_at": generated_at.isoformat(),
        "status": report.get("final_status")
        or ("SKIPPED" if skipped_reason else "RANK_FIRST_HYPOTHESIS_WATCHLIST_FAILED_NO_REPORT"),
        "output_dir": relpath(packet_dir),
        "report_path": relpath(report_path),
        "attempted": attempted,
        "returncode": returncode,
        "skipped_reason": skipped_reason,
        "packet_count": int(report.get("packet_count") or 0),
        "evaluation_count": int(report.get("evaluation_count") or 0),
        "candidate_count": int(report.get("candidate_count") or 0),
        "directional_ready_candidate_count": int(
            report.get("directional_ready_candidate_count") or 0
        ),
        "minimum_triggered_races_for_directional_read": report.get(
            "minimum_triggered_races_for_directional_read"
        ),
        "minimum_distinct_samples_for_directional_read": report.get(
            "minimum_distinct_samples_for_directional_read"
        ),
        "best_candidate_key": best.get("candidate_key"),
        "best_candidate_status": best.get("status"),
        "best_candidate_distinct_sample_count": best.get(
            "distinct_sample_signature_count"
        ),
        "best_candidate_triggered_race_count": best.get(
            "latest_gate_triggered_race_count"
        ),
        "best_candidate_top1_delta_vs_market": best.get(
            "latest_top1_delta_vs_market"
        ),
        "best_candidate_logloss_delta_vs_market": best.get(
            "latest_logloss_delta_vs_market"
        ),
        "best_candidate_blockers": list(best.get("blockers") or []),
        "blockers": list(report.get("blockers") or []),
        "no_write_guarantees": dict(report.get("no_write_guarantees") or NO_WRITE_GUARANTEES),
    }


def odds_capture_only_ready(
    *,
    step: Mapping[str, Any],
    autopilot_result: Mapping[str, Any] | None,
    odds_status: Mapping[str, Any],
    refresh_report: Mapping[str, Any],
) -> bool:
    accepted_odds_statuses = {
        "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED",
        "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS",
    }
    status_counts = odds_status.get("status_counts") or {}
    refresh_status = refresh_report.get("status")
    has_usable_odds_action = bool(
        int(odds_status.get("ready_count") or 0) > 0
        or int(odds_status.get("inserted_live_odds_rows") or 0) > 0
        or any(
            status_counts.get(status)
            for status in (
                "APPENDED",
                "SKIPPED_ALREADY_CAPTURED",
                "SKIPPED_EXISTING_CAPTURE_SUPERSET",
            )
        )
    )
    refresh_status_accepted = bool(
        refresh_status == "SUCCESS"
        or (
            refresh_status == "METADATA_COVERAGE_INCOMPLETE"
            and has_usable_odds_action
        )
    )
    return (
        step.get("returncode") == 0
        and (autopilot_result or {}).get("final_verdict")
        in {"PARTIAL_AUTOMATION_READY", "AUTOPILOT_READY"}
        and refresh_status_accepted
        and odds_status.get("status") in accepted_odds_statuses
    )


def odds_capture_only_handled_no_write(
    *,
    step: Mapping[str, Any],
    autopilot_result: Mapping[str, Any] | None,
    odds_status: Mapping[str, Any],
    refresh_report: Mapping[str, Any],
) -> bool:
    status_counts = odds_status.get("status_counts") or {}
    handled_zero_write_statuses = {
        "BLOCKED_VALIDATION_FAILED",
        "BLOCKED_TIME_GATE_BEFORE_APPEND",
        "BLOCKED_TIME_GATE_BEFORE_FETCH",
    }
    return (
        step.get("returncode") == 0
        and (autopilot_result or {}).get("final_verdict")
        in {"PARTIAL_AUTOMATION_READY", "AUTOPILOT_READY"}
        and refresh_report.get("status") == "SUCCESS"
        and odds_status.get("status") == "AUTONOMOUS_LIVE_ODDS_CAPTURE_BLOCKED"
        and int(odds_status.get("inserted_live_odds_rows") or 0) == 0
        and any(status_counts.get(status) for status in handled_zero_write_statuses)
    )


def classify_odds_capture_only_final_status(
    *,
    step: Mapping[str, Any],
    autopilot_result: Mapping[str, Any] | None,
    odds_status: Mapping[str, Any],
    refresh_report: Mapping[str, Any],
) -> str:
    if odds_capture_only_ready(
        step=step,
        autopilot_result=autopilot_result,
        odds_status=odds_status,
        refresh_report=refresh_report,
    ):
        return "ODDS_CAPTURE_ONLY_READY"
    if odds_capture_only_handled_no_write(
        step=step,
        autopilot_result=autopilot_result,
        odds_status=odds_status,
        refresh_report=refresh_report,
    ):
        return "ODDS_CAPTURE_ONLY_HANDLED_NO_WRITE"
    return "ODDS_CAPTURE_ONLY_FAILED"


def odds_capture_only_operator_fields(final_status: str) -> dict[str, str]:
    fields_by_status = {
        "ODDS_CAPTURE_ONLY_RUNNING": {
            "status": "RUNNING",
            "runtime_action": "ODDS_CAPTURE_ONLY_IN_PROGRESS",
            "readiness_decision": "IN_PROGRESS",
        },
        "ODDS_CAPTURE_ONLY_READY": {
            "status": "READY",
            "runtime_action": "WAIT_FOR_NEXT_ODDS_CAPTURE_WINDOW",
            "readiness_decision": "CONTINUE_ODDS_CAPTURE",
        },
        "ODDS_CAPTURE_ONLY_WAITING_FOR_WINDOW": {
            "status": "WAITING",
            "runtime_action": "WAIT_UNTIL_NEXT_FIXED_WINDOW",
            "readiness_decision": "ODDS_CAPTURE_WAITING",
        },
        "ODDS_CAPTURE_ONLY_HANDLED_NO_WRITE": {
            "status": "HANDLED_NO_WRITE",
            "runtime_action": "REVIEW_BLOCKED_CAPTURE_NO_DB_WRITE",
            "readiness_decision": "CHECK_ODDS_CAPTURE_BLOCKER",
        },
        "SKIPPED_LOCK_HELD": {
            "status": "SKIPPED_LOCK_HELD",
            "runtime_action": "SKIP_LOCK_HELD",
            "readiness_decision": "WAIT_FOR_ACTIVE_DAEMON",
        },
        "SKIPPED_FULL_DAEMON_LOCK_HANDOFF": {
            "status": "SKIPPED_FULL_DAEMON_LOCK_HANDOFF",
            "runtime_action": "YIELD_LOCK_HANDOFF_TO_FULL_DAEMON",
            "readiness_decision": "WAIT_FOR_FULL_DAEMON",
        },
        "ODDS_CAPTURE_ONLY_FAILED": {
            "status": "FAILED",
            "runtime_action": "CHECK_ODDS_CAPTURE_FAILURE",
            "readiness_decision": "NEEDS_OPERATOR_REVIEW",
        },
    }
    return dict(
        fields_by_status.get(
            final_status,
            {
                "status": "NEEDS_REVIEW",
                "runtime_action": "CHECK_ODDS_CAPTURE_STATUS",
                "readiness_decision": "NEEDS_OPERATOR_REVIEW",
            },
        )
    )


def odds_capture_only_operator_fields_for_report(
    final_status: str,
    odds_status: Mapping[str, Any] | None,
) -> dict[str, str]:
    fields = odds_capture_only_operator_fields(final_status)
    if not isinstance(odds_status, Mapping):
        return fields
    if (
        final_status == "ODDS_CAPTURE_ONLY_READY"
        and int_or_zero(odds_status.get("inserted_live_odds_rows")) > 0
        and int_or_zero(odds_status.get("blocked_attempt_count")) > 0
    ):
        fields.update(
            {
                "status": "READY_WITH_BLOCKED_ATTEMPTS",
                "runtime_action": "REVIEW_BLOCKED_CAPTURE_AFTER_APPEND",
                "readiness_decision": "CONTINUE_ODDS_CAPTURE_WITH_BLOCKER_REVIEW",
            }
        )
    return fields


def buffered_odds_capture_resume_after_local(
    next_window_opens_at: Any,
    *,
    default_tz: Any | None,
    resume_buffer_seconds: int = DEFAULT_ODDS_CAPTURE_ONLY_PREFLIGHT_RESUME_BUFFER_SECONDS,
) -> str | None:
    next_window_opens = parse_datetime_value(next_window_opens_at, default_tz=default_tz)
    if next_window_opens is None:
        return None
    return (next_window_opens - timedelta(seconds=resume_buffer_seconds)).isoformat()


def due_odds_capture_window_offset(minutes_to_jump: float | None) -> int | None:
    if minutes_to_jump is None or minutes_to_jump <= 0:
        return None
    if minutes_to_jump > max(ODDS_CAPTURE_WINDOW_OFFSETS_MINUTES):
        return None
    for offset in sorted(ODDS_CAPTURE_WINDOW_OFFSETS_MINUTES):
        if minutes_to_jump <= offset:
            return offset
    return None


def timer_calendar_covered_minutes(on_calendar: str) -> list[int]:
    minute_spec = on_calendar.strip()
    if " " in minute_spec:
        minute_spec = minute_spec.rsplit(" ", 1)[-1]
    if ":" in minute_spec:
        minute_spec = minute_spec.rsplit(":", 1)[-1]
    if minute_spec == "*":
        return list(range(60))
    minutes: set[int] = set()
    for raw_part in minute_spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "/" in part:
            start_text, step_text = part.split("/", 1)
            if start_text in {"", "*"}:
                start = 0
            else:
                try:
                    start = int(start_text)
                except ValueError:
                    continue
            try:
                step = int(step_text)
            except ValueError:
                continue
            if step <= 0:
                continue
            minutes.update(range(start, 60, step))
            continue
        try:
            minute = int(part)
        except ValueError:
            continue
        if 0 <= minute <= 59:
            minutes.add(minute)
    return sorted(minutes)


def next_meaningful_action_timer_coverage(
    *,
    next_action_at: str | None,
    current_time: datetime,
    on_calendar: str = DEFAULT_ODDS_CAPTURE_ONLY_TIMER_ON_CALENDAR,
) -> dict[str, Any]:
    covered_minutes = timer_calendar_covered_minutes(on_calendar)
    action_time = parse_datetime_value(next_action_at, default_tz=current_time.tzinfo)
    action_minute = action_time.minute if action_time is not None else None
    if action_minute is None:
        covered = False
        reason = "next_meaningful_action_at_missing"
    elif action_minute in covered_minutes:
        covered = True
        reason = "minute_covered_by_odds_capture_timer_on_calendar"
    else:
        covered = False
        reason = "minute_not_covered_by_odds_capture_timer_on_calendar"
    return {
        "odds_capture_timer_on_calendar": on_calendar,
        "odds_capture_timer_covered_minutes": covered_minutes,
        "next_meaningful_action_timer_minute": action_minute,
        "next_meaningful_action_timer_covered": covered,
        "next_meaningful_action_timer_coverage_reason": reason,
    }


def _identity_value(value: Any) -> str:
    return str(value or "").strip().upper()


def _coverage_window_matches_next_race(
    coverage_row: Mapping[str, Any],
    next_race: Mapping[str, Any],
) -> bool:
    field_pairs = (
        ("race_id", "race_id"),
        ("race_date", "date"),
        ("venue", "venue"),
        ("race_number", "race_number"),
    )
    matched_any = False
    for coverage_key, race_key in field_pairs:
        expected = _identity_value(next_race.get(race_key))
        actual = _identity_value(coverage_row.get(coverage_key))
        if not expected or not actual:
            continue
        if expected != actual:
            return False
        matched_any = True
    return matched_any or _identity_value(next_race.get("race_id")) == ""


def _coverage_windows_by_offset(
    capture_window_coverage: Mapping[str, Any] | None,
    next_race: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    if not isinstance(capture_window_coverage, Mapping):
        return {}
    windows = capture_window_coverage.get("windows")
    if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes)):
        return {}
    by_offset: dict[int, Mapping[str, Any]] = {}
    for row in windows:
        if not isinstance(row, Mapping):
            continue
        try:
            offset = int(row.get("offset_minutes"))
        except (TypeError, ValueError):
            continue
        if not _coverage_window_matches_next_race(row, next_race):
            continue
        by_offset[offset] = row
    return by_offset


def load_capture_window_coverage_from_status(
    odds_status: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(odds_status, Mapping):
        return None
    inline = odds_status.get("capture_window_coverage")
    if isinstance(inline, Mapping):
        return dict(inline)
    report_path = rooted_path(odds_status.get("capture_window_coverage_report"))
    return load_json(report_path)


def odds_capture_fixed_window_schedule(
    next_preferred_window: Mapping[str, Any] | None,
    *,
    current_time: datetime,
    capture_window_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    window = next_preferred_window if isinstance(next_preferred_window, Mapping) else {}
    next_race = window.get("next_race")
    if not isinstance(next_race, Mapping):
        next_race = {}
    jump_at = parse_datetime_value(
        next_race.get("jump_datetime"),
        default_tz=current_time.tzinfo,
    )
    rows: list[dict[str, Any]] = []
    due_offset: int | None = None
    minutes_to_jump: float | None = None
    coverage_windows = []
    if isinstance(capture_window_coverage, Mapping):
        raw_coverage_windows = capture_window_coverage.get("windows")
        if isinstance(raw_coverage_windows, Sequence) and not isinstance(
            raw_coverage_windows, (str, bytes)
        ):
            coverage_windows = [
                row for row in raw_coverage_windows if isinstance(row, Mapping)
            ]
    coverage_by_offset = _coverage_windows_by_offset(capture_window_coverage, next_race)
    coverage_available = bool(coverage_windows)
    coverage_checked = bool(coverage_by_offset)
    if coverage_checked:
        coverage_match_status = "MATCHED_NEXT_RACE"
    elif coverage_available:
        coverage_match_status = "COVERAGE_AVAILABLE_NO_NEXT_RACE_MATCH"
    else:
        coverage_match_status = "COVERAGE_UNAVAILABLE"
    if jump_at is not None:
        jump_cmp = jump_at
        current_cmp = current_time
        if jump_cmp.tzinfo is None and current_cmp.tzinfo is not None:
            jump_cmp = jump_cmp.replace(tzinfo=current_cmp.tzinfo)
        if current_cmp.tzinfo is None and jump_cmp.tzinfo is not None:
            current_cmp = current_cmp.replace(tzinfo=jump_cmp.tzinfo)
        minutes_to_jump = (jump_cmp - current_cmp).total_seconds() / 60.0
        due_offset = due_odds_capture_window_offset(minutes_to_jump)
    for offset in ODDS_CAPTURE_WINDOW_OFFSETS_MINUTES:
        target_at: datetime | None = None
        minutes_until_target: float | None = None
        if jump_at is None:
            status = "UNKNOWN"
            reason = "jump_datetime_missing"
        else:
            target_at = jump_at - timedelta(minutes=offset)
            target_cmp = target_at
            current_cmp = current_time
            if target_cmp.tzinfo is None and current_cmp.tzinfo is not None:
                target_cmp = target_cmp.replace(tzinfo=current_cmp.tzinfo)
            if current_cmp.tzinfo is None and target_cmp.tzinfo is not None:
                current_cmp = current_cmp.replace(tzinfo=target_cmp.tzinfo)
            minutes_until_target = (target_cmp - current_cmp).total_seconds() / 60.0
            if minutes_to_jump is not None and minutes_to_jump <= 0:
                status = "AFTER_JUMP"
                reason = "race_already_jumped"
            elif minutes_until_target > 0:
                status = "PENDING"
                reason = "window_not_open_yet"
            elif due_offset == offset:
                status = "DUE"
                reason = "window_due_without_complete_capture_check"
            else:
                status = "PASSED"
                reason = "earlier_window_target_passed_without_capture_check"
        coverage_row = coverage_by_offset.get(offset)
        coverage_status = None
        if coverage_row is not None:
            coverage_status = str(coverage_row.get("status") or "").strip().upper()
            if coverage_status in {"CAPTURED", "PENDING"}:
                status = coverage_status
                reason = str(coverage_row.get("reason") or reason)
        rows.append(
            {
                "offset_minutes": offset,
                "capture_mode": f"autonomous_prejump_t{offset}m",
                "target_capture_at": target_at.isoformat() if target_at else None,
                "minutes_until_target": minutes_until_target,
                "minutes_to_jump": minutes_to_jump,
                "status": status,
                "reason": reason,
                "coverage_status": coverage_status,
                "existing_capture_count": (
                    coverage_row.get("existing_capture_count")
                    if coverage_row is not None
                    else None
                ),
                "existing_capture_status": (
                    coverage_row.get("existing_capture_status")
                    if coverage_row is not None
                    else None
                ),
            }
        )
    status_counts = Counter(row["status"] for row in rows)
    due_rows = [row for row in rows if row["status"] == "DUE"]
    pending_rows = [
        row
        for row in rows
        if row["status"] == "PENDING" and row.get("target_capture_at") is not None
    ]
    pending_rows.sort(key=lambda row: str(row.get("target_capture_at")))
    next_due = due_rows[0] if due_rows else None
    next_pending = pending_rows[0] if pending_rows else None
    if next_due is not None:
        next_action = "RUN_ODDS_CAPTURE_NOW"
        next_action_at = current_time.isoformat()
        next_action_offset_minutes = next_due["offset_minutes"]
        next_action_reason = next_due["reason"]
    elif next_pending is not None:
        next_action = "WAIT_UNTIL_NEXT_FIXED_WINDOW"
        next_action_at = next_pending["target_capture_at"]
        next_action_offset_minutes = next_pending["offset_minutes"]
        next_action_reason = next_pending["reason"]
    else:
        next_action = "REFRESH_UPCOMING_RACE_WINDOW"
        next_action_at = current_time.isoformat()
        next_action_offset_minutes = None
        next_action_reason = (
            "no_due_or_pending_fixed_window_from_current_next_preferred_race"
        )
    timer_coverage = next_meaningful_action_timer_coverage(
        next_action_at=next_action_at,
        current_time=current_time,
    )
    return {
        "schema_version": "shadow_autopilot_odds_capture_fixed_window_schedule_v1",
        "generated_at": current_time.isoformat(),
        "capture_window_coverage_available": coverage_available,
        "capture_window_coverage_window_count": len(coverage_windows),
        "capture_window_coverage_matched_window_count": len(coverage_by_offset),
        "capture_window_coverage_match_status": coverage_match_status,
        "coverage_checked": coverage_checked,
        "capture_window_coverage_status_counts": (
            dict(Counter(str(row.get("status") or "UNKNOWN") for row in coverage_by_offset.values()))
            if coverage_checked
            else {}
        ),
        "capture_window_offsets_minutes": list(ODDS_CAPTURE_WINDOW_OFFSETS_MINUTES),
        "race_id": next_race.get("race_id"),
        "venue": next_race.get("venue"),
        "race_number": next_race.get("race_number"),
        "race_date": next_race.get("date"),
        "race_time": next_race.get("race_time"),
        "jump_datetime": jump_at.isoformat() if jump_at else None,
        "status_counts": dict(status_counts),
        "next_due_offset_minutes": None if next_due is None else next_due["offset_minutes"],
        "next_due_capture_at": None if next_due is None else next_due["target_capture_at"],
        "next_pending_offset_minutes": None
        if next_pending is None
        else next_pending["offset_minutes"],
        "next_pending_capture_at": None
        if next_pending is None
        else next_pending["target_capture_at"],
        "next_meaningful_action": next_action,
        "next_meaningful_action_at": next_action_at,
        "next_meaningful_action_offset_minutes": next_action_offset_minutes,
        "next_meaningful_action_reason": next_action_reason,
        **timer_coverage,
        "windows": rows,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def direct_capture_all_ready_races_already_captured(
    odds_status: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(odds_status, Mapping):
        return False
    if odds_status.get("status") != "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS":
        return False
    if int_or_zero(odds_status.get("inserted_live_odds_rows")) != 0:
        return False
    ready_count = int_or_zero(odds_status.get("ready_count"))
    if ready_count <= 0:
        return False
    status_counts = {
        str(status): int_or_zero(count)
        for status, count in dict(odds_status.get("status_counts") or {}).items()
        if int_or_zero(count) > 0
    }
    ignored_statuses = {"SKIPPED_NOT_READY"}
    handled_count = int_or_zero(status_counts.get("SKIPPED_ALREADY_CAPTURED"))
    return (
        bool(handled_count)
        and set(status_counts).issubset({"SKIPPED_ALREADY_CAPTURED"} | ignored_statuses)
        and handled_count >= ready_count
    )


def direct_capture_all_ready_races_handled_after_append(
    odds_status: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(odds_status, Mapping):
        return False
    if odds_status.get("status") != "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED":
        return False
    if int_or_zero(odds_status.get("inserted_live_odds_rows")) <= 0:
        return False
    if int_or_zero(odds_status.get("blocked_attempt_count")) > 0:
        return False
    ready_count = int_or_zero(odds_status.get("ready_count"))
    if ready_count <= 0:
        return False
    status_counts = {
        str(status): int_or_zero(count)
        for status, count in dict(odds_status.get("status_counts") or {}).items()
        if int_or_zero(count) > 0
    }
    handled_statuses = {"APPENDED", "SKIPPED_ALREADY_CAPTURED"}
    ignored_statuses = {"SKIPPED_NOT_READY"}
    handled_count = sum(int_or_zero(status_counts.get(status)) for status in handled_statuses)
    return (
        bool(status_counts.get("APPENDED"))
        and set(status_counts).issubset(handled_statuses | ignored_statuses)
        and handled_count >= ready_count
    )


def direct_capture_time_gate_blockers_are_expired(
    odds_status: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(odds_status, Mapping):
        return False
    if odds_status.get("status") not in {
        "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED",
        "AUTONOMOUS_LIVE_ODDS_CAPTURE_BLOCKED",
    }:
        return False
    blocked_attempts = odds_status.get("blocked_attempts") or []
    if not isinstance(blocked_attempts, Sequence) or isinstance(
        blocked_attempts, (str, bytes)
    ):
        return False
    if not blocked_attempts:
        return False
    for attempt in blocked_attempts:
        if not isinstance(attempt, Mapping):
            return False
        status = str(attempt.get("status") or "")
        if status not in {
            "BLOCKED_TIME_GATE_BEFORE_APPEND",
            "BLOCKED_TIME_GATE_BEFORE_FETCH",
        }:
            return False
        reasons = [str(reason) for reason in attempt.get("reasons") or []]
        try:
            fresh_minutes_to_jump = float(attempt.get("fresh_minutes_to_jump"))
        except (TypeError, ValueError):
            fresh_minutes_to_jump = None
        if "race_already_jumped" not in reasons and (
            fresh_minutes_to_jump is None or fresh_minutes_to_jump > 0
        ):
            return False
    return True


def reconcile_odds_capture_schedule_with_direct_status(
    schedule: Mapping[str, Any],
    odds_status: Mapping[str, Any] | None,
    *,
    current_time: datetime,
) -> dict[str, Any]:
    reconciled = dict(schedule)
    if not direct_capture_all_ready_races_already_captured(odds_status):
        if direct_capture_all_ready_races_handled_after_append(odds_status):
            reconciliation_reason = "direct_capture_all_ready_races_handled_after_append"
        elif direct_capture_time_gate_blockers_are_expired(odds_status):
            reconciliation_reason = "direct_capture_time_gate_blockers_already_expired"
        else:
            return reconciled
    else:
        reconciliation_reason = "direct_capture_all_ready_races_already_captured"
    if schedule.get("next_meaningful_action") != "RUN_ODDS_CAPTURE_NOW":
        return reconciled
    timer_coverage = next_meaningful_action_timer_coverage(
        next_action_at=current_time.isoformat(),
        current_time=current_time,
    )
    reconciled.update(
        {
            "schedule_reconciled_with_direct_capture": True,
            "schedule_reconciliation_reason": reconciliation_reason,
            "pre_reconciliation_next_meaningful_action": schedule.get(
                "next_meaningful_action"
            ),
            "pre_reconciliation_next_meaningful_action_at": schedule.get(
                "next_meaningful_action_at"
            ),
            "pre_reconciliation_next_meaningful_action_offset_minutes": schedule.get(
                "next_meaningful_action_offset_minutes"
            ),
            "pre_reconciliation_next_meaningful_action_reason": schedule.get(
                "next_meaningful_action_reason"
            ),
            "next_meaningful_action": "REFRESH_UPCOMING_RACE_WINDOW",
            "next_meaningful_action_at": current_time.isoformat(),
            "next_meaningful_action_offset_minutes": None,
            "next_meaningful_action_reason": reconciliation_reason,
            **timer_coverage,
        }
    )
    return reconciled


def odds_capture_next_race_report_fields(schedule: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "next_race_id": schedule.get("race_id"),
        "next_race_date": schedule.get("race_date"),
        "next_race_venue": schedule.get("venue"),
        "next_race_number": schedule.get("race_number"),
        "next_race_time": schedule.get("race_time"),
        "next_race_jump_datetime": schedule.get("jump_datetime"),
        "next_meaningful_action_offset_minutes": schedule.get(
            "next_meaningful_action_offset_minutes"
        ),
        "next_meaningful_action_timer_minute": schedule.get(
            "next_meaningful_action_timer_minute"
        ),
        "next_meaningful_action_timer_covered": schedule.get(
            "next_meaningful_action_timer_covered"
        ),
        "next_meaningful_action_timer_coverage_reason": schedule.get(
            "next_meaningful_action_timer_coverage_reason"
        ),
    }


def odds_capture_t2_lock_skip_fields(
    *,
    final_status: str,
    fixed_window_schedule: Mapping[str, Any] | None,
) -> dict[str, Any]:
    schedule = fixed_window_schedule if isinstance(fixed_window_schedule, Mapping) else {}
    t2_window: Mapping[str, Any] = {}
    windows = schedule.get("windows")
    if isinstance(windows, Sequence) and not isinstance(windows, (str, bytes)):
        for row in windows:
            if not isinstance(row, Mapping):
                continue
            try:
                offset = int(row.get("offset_minutes"))
            except (TypeError, ValueError):
                continue
            if offset == 2:
                t2_window = row
                break
    lock_skip = final_status in {
        "SKIPPED_LOCK_HELD",
        "SKIPPED_FULL_DAEMON_LOCK_HANDOFF",
    }
    t2_due = (
        str(t2_window.get("status") or "").upper() == "DUE"
        or (
            schedule.get("next_meaningful_action_offset_minutes") == 2
            and schedule.get("next_meaningful_action") == "RUN_ODDS_CAPTURE_NOW"
        )
    )
    active = bool(lock_skip and t2_due)
    return {
        "t2_miss_lock_held": active,
        "t2_miss_cause_counts": {"t2_miss_lock_held": 1} if active else {},
        "t2_lock_skip_race_id": schedule.get("race_id") if active else None,
        "t2_lock_skip_target_capture_at": (
            t2_window.get("target_capture_at") if active else None
        ),
        "t2_lock_skip_minutes_to_jump": (
            t2_window.get("minutes_to_jump") if active else None
        ),
        "t2_lock_skip_lock_status": final_status if active else None,
    }


def t2_odds_capture_status_fields(
    status: Mapping[str, Any] | None,
    *,
    prefix: str = "",
) -> dict[str, Any]:
    status = status if isinstance(status, Mapping) else {}
    cause_counts = (
        status.get("t2_miss_cause_counts")
        or status.get("t2_capture_miss_cause_counts")
        or {}
    )
    examples = (
        status.get("t2_miss_examples")
        or status.get("t2_capture_miss_examples")
        or []
    )
    return {
        f"{prefix}t2_miss_attempt_count": int_or_zero(
            status.get("t2_miss_attempt_count")
        ),
        f"{prefix}t2_miss_cause_counts": dict(cause_counts),
        f"{prefix}t2_miss_examples": list(examples),
        f"{prefix}t2_miss_lock_held": bool(status.get("t2_miss_lock_held")),
        f"{prefix}t2_lock_skip_race_id": status.get("t2_lock_skip_race_id"),
        f"{prefix}t2_lock_skip_target_capture_at": status.get(
            "t2_lock_skip_target_capture_at"
        ),
        f"{prefix}t2_lock_skip_minutes_to_jump": status.get(
            "t2_lock_skip_minutes_to_jump"
        ),
        f"{prefix}t2_lock_skip_lock_status": status.get("t2_lock_skip_lock_status"),
    }


def t2_odds_capture_surface_fields(
    *,
    autonomous_live_odds_capture_status: Mapping[str, Any] | None,
    odds_capture_state_publish: Mapping[str, Any] | None,
    last: bool = False,
) -> dict[str, Any]:
    autonomous_prefix = (
        "last_autonomous_live_odds_capture_"
        if last
        else "autonomous_live_odds_capture_"
    )
    odds_prefix = "last_odds_capture_" if last else "odds_capture_"
    return {
        **t2_odds_capture_status_fields(
            autonomous_live_odds_capture_status,
            prefix=autonomous_prefix,
        ),
        **t2_odds_capture_status_fields(
            odds_capture_state_publish,
            prefix=odds_prefix,
        ),
    }


def full_daemon_defer_fixed_window_schedule(
    state: Mapping[str, Any],
    *,
    current_time: datetime,
) -> tuple[dict[str, Any], str]:
    existing_schedule = state.get("odds_capture_fixed_window_schedule")
    if not isinstance(existing_schedule, Mapping):
        existing_schedule = {}
    next_window = state.get("next_preferred_window")
    if not isinstance(next_window, Mapping):
        return dict(existing_schedule), "published_odds_capture_state"
    next_race = next_window.get("next_race")
    if not isinstance(next_race, Mapping) or not next_race.get("jump_datetime"):
        return dict(existing_schedule), "published_odds_capture_state"

    recomputed = odds_capture_fixed_window_schedule(
        next_window,
        current_time=current_time,
    )
    recomputed = reconcile_odds_capture_schedule_with_direct_status(
        recomputed,
        state,
        current_time=current_time,
    )
    return recomputed, "recomputed_from_next_preferred_window"


def full_daemon_odds_window_defer_decision(
    state: Mapping[str, Any] | None,
    *,
    current_time: datetime,
    horizon_seconds: int = DEFAULT_FULL_DAEMON_ODDS_DEFER_HORIZON_SECONDS,
) -> dict[str, Any]:
    state = state if isinstance(state, Mapping) else {}
    schedule, schedule_source = full_daemon_defer_fixed_window_schedule(
        state,
        current_time=current_time,
    )
    next_window = state.get("next_preferred_window")
    if not isinstance(next_window, Mapping):
        next_window = {}

    closes_at = parse_datetime_value(
        next_window.get("next_window_closes_at"),
        default_tz=current_time.tzinfo,
    )
    action_at = parse_datetime_value(
        state.get("next_meaningful_action_at")
        or schedule.get("next_meaningful_action_at"),
        default_tz=current_time.tzinfo,
    )
    pending_at = parse_datetime_value(
        schedule.get("next_pending_capture_at"),
        default_tz=current_time.tzinfo,
    )
    state_updated_at = parse_datetime_value(
        state.get("updated_at"),
        default_tz=current_time.tzinfo,
    )
    horizon_at = current_time + timedelta(seconds=horizon_seconds)
    window_open = closes_at is not None and current_time < closes_at
    fresh_open_multi_race_state = bool(
        str(next_window.get("status") or "") == "OPEN_NOW"
        and int(next_window.get("selected_count") or next_window.get("selected_race_count") or 0) > 1
        and state_updated_at is not None
        and current_time <= state_updated_at + timedelta(seconds=horizon_seconds)
    )
    next_action = str(
        state.get("next_meaningful_action") or schedule.get("next_meaningful_action") or ""
    )
    refresh_action_requested = next_action == "REFRESH_UPCOMING_RACE_WINDOW"
    schedule_recomputed = schedule_source == "recomputed_from_next_preferred_window"
    action_due_now = bool(
        next_action == "RUN_ODDS_CAPTURE_NOW" and (schedule_recomputed or window_open)
    )
    action_imminent = (
        next_action == "WAIT_UNTIL_NEXT_FIXED_WINDOW"
        and action_at is not None
        and current_time <= action_at <= horizon_at
    )
    pending_imminent = (
        pending_at is not None and current_time <= pending_at <= horizon_at
    )
    due_count = int((schedule.get("status_counts") or {}).get("DUE") or 0)
    due_capture_unhandled = bool(
        due_count > 0
        and not schedule.get("schedule_reconciled_with_direct_capture")
        and (schedule_recomputed or window_open)
    )
    fixed_window_due_or_near_due = bool(
        action_due_now or action_imminent or pending_imminent or due_capture_unhandled
    )
    refresh_action_clear = bool(
        refresh_action_requested
        and not pending_imminent
        and not due_capture_unhandled
    )
    should_defer = bool(
        not refresh_action_clear
        and (window_open or fresh_open_multi_race_state or fixed_window_due_or_near_due)
        and (fixed_window_due_or_near_due or fresh_open_multi_race_state)
    )
    if should_defer:
        reason = (
            "odds_capture_state_open_with_additional_selected_races"
            if fresh_open_multi_race_state and not window_open
            else "odds_capture_window_open_or_imminent"
        )
    elif refresh_action_requested:
        reason = "odds_capture_refresh_action_requested"
    elif closes_at is not None and current_time >= closes_at:
        reason = "odds_capture_window_closed"
    elif not window_open:
        reason = "odds_capture_window_not_open"
    else:
        reason = "odds_capture_action_not_imminent"
    return {
        "schema_version": "shadow_autopilot_full_daemon_odds_window_defer_v1",
        "should_defer": should_defer,
        "reason": reason,
        "current_time": current_time.isoformat(),
        "horizon_seconds": horizon_seconds,
        "horizon_at": horizon_at.isoformat(),
        "next_window_closes_at": closes_at.isoformat() if closes_at else None,
        "next_meaningful_action": (
            state.get("next_meaningful_action") or schedule.get("next_meaningful_action")
        ),
        "next_meaningful_action_at": action_at.isoformat() if action_at else None,
        "next_meaningful_action_offset_minutes": schedule.get(
            "next_meaningful_action_offset_minutes"
        ),
        "next_due_offset_minutes": schedule.get("next_due_offset_minutes"),
        "next_due_capture_at": schedule.get("next_due_capture_at"),
        "next_pending_offset_minutes": schedule.get("next_pending_offset_minutes"),
        "next_pending_capture_at": pending_at.isoformat() if pending_at else None,
        "fixed_window_schedule_source": schedule_source,
        "fixed_window_due_or_near_due": fixed_window_due_or_near_due,
        "due_capture_window_unhandled": due_capture_unhandled,
        "schedule_reconciled_with_direct_capture": bool(
            schedule.get("schedule_reconciled_with_direct_capture")
        ),
        "state_updated_at": state_updated_at.isoformat() if state_updated_at else None,
        "fresh_open_multi_race_state": fresh_open_multi_race_state,
        "due_capture_window_count": due_count,
    }


def post_primary_odds_capture_release_decision(
    odds_capture_state_publish: Mapping[str, Any] | None,
    *,
    current_time: datetime,
) -> dict[str, Any]:
    state = odds_capture_state_publish if isinstance(odds_capture_state_publish, Mapping) else {}
    schedule = state.get("odds_capture_fixed_window_schedule")
    if not isinstance(schedule, Mapping):
        schedule = {}
    next_action = str(
        state.get("next_meaningful_action")
        or schedule.get("next_meaningful_action")
        or ""
    )
    action_at = parse_datetime_value(
        state.get("next_meaningful_action_at")
        or schedule.get("next_meaningful_action_at"),
        default_tz=current_time.tzinfo,
    )
    due_count = int_or_zero((schedule.get("status_counts") or {}).get("DUE"))
    publish_status = str(state.get("status") or "")
    immediate_actions = {"RUN_ODDS_CAPTURE_NOW", "REFRESH_UPCOMING_RACE_WINDOW"}
    action_due_now = (
        next_action in immediate_actions
        and (action_at is None or action_at <= current_time)
    )
    should_release = bool(publish_status == "PUBLISHED" and (action_due_now or due_count > 0))
    if should_release:
        if next_action == "REFRESH_UPCOMING_RACE_WINDOW":
            reason = "post_primary_odds_capture_refresh_due_now"
        elif due_count > 0:
            reason = "post_primary_fixed_window_due"
        else:
            reason = "post_primary_odds_capture_due_now"
    elif publish_status != "PUBLISHED":
        reason = "odds_capture_state_not_published"
    elif next_action not in immediate_actions:
        reason = "odds_capture_action_not_immediate"
    elif action_at is not None and action_at > current_time:
        reason = "odds_capture_action_not_due_yet"
    else:
        reason = "odds_capture_release_not_required"
    return {
        "schema_version": "shadow_autopilot_post_primary_odds_capture_release_v1",
        "should_release": should_release,
        "reason": reason,
        "current_time": current_time.isoformat(),
        "odds_capture_state_publish_status": state.get("status"),
        "next_meaningful_action": next_action or None,
        "next_meaningful_action_at": action_at.isoformat() if action_at else None,
        "due_capture_window_count": due_count,
        "odds_capture_state_path": state.get("state_path"),
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def odds_capture_preflight_wait(
    *,
    state_path: Path | None,
    now: datetime,
    max_age_seconds: int = DEFAULT_ODDS_CAPTURE_ONLY_PREFLIGHT_MAX_AGE_SECONDS,
    resume_buffer_seconds: int = DEFAULT_ODDS_CAPTURE_ONLY_PREFLIGHT_RESUME_BUFFER_SECONDS,
) -> dict[str, Any] | None:
    state = load_json(state_path)
    if not state:
        return None
    if state.get("final_status") not in {
        "ODDS_CAPTURE_ONLY_READY",
        "ODDS_CAPTURE_ONLY_WAITING_FOR_WINDOW",
    }:
        return None
    if state.get("odds_capture_refresh_status") != "SUCCESS":
        return None
    if state.get("odds_capture_status") != "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS":
        return None
    next_window_opens = parse_datetime_value(
        state.get("next_window_opens_at"),
        default_tz=now.tzinfo,
    )
    if next_window_opens is None:
        return None
    source_updated = parse_datetime_value(
        state.get("window_state_source_updated_at") or state.get("updated_at"),
        default_tz=now.tzinfo,
    )
    if source_updated is None:
        return None
    age_seconds = (now - source_updated).total_seconds()
    if age_seconds < 0 or age_seconds > max_age_seconds:
        return None
    resume_after = next_window_opens - timedelta(seconds=resume_buffer_seconds)
    if now >= resume_after:
        return None
    fixed_window_schedule = odds_capture_fixed_window_schedule(
        state.get("next_preferred_window"),
        current_time=now,
    )
    return {
        "status": "WAITING_FOR_FUTURE_WINDOW",
        "reason": "recent_state_next_window_not_open",
        "state_path": relpath(state_path),
        "state_age_seconds": age_seconds,
        "window_state_source_updated_at": source_updated.isoformat(),
        "window_state_source": state.get("state_source"),
        "source_report_path": state.get("source_report_path"),
        "next_window_opens_at": next_window_opens.isoformat(),
        "recommended_rerun_after_local": resume_after.isoformat(),
        "source_recommended_rerun_after_local": state.get(
            "source_recommended_rerun_after_local"
        )
        or state.get("recommended_rerun_after_local")
        or next_window_opens.isoformat(),
        "resume_after_local": resume_after.isoformat(),
        "resume_buffer_seconds": resume_buffer_seconds,
        "max_age_seconds": max_age_seconds,
        "odds_capture_refresh_status": state.get("odds_capture_refresh_status"),
        "next_preferred_window": state.get("next_preferred_window") or {},
        "odds_capture_fixed_window_schedule": fixed_window_schedule,
    }


def publish_full_daemon_odds_capture_state(
    *,
    state_path: Path | None,
    generated_at: datetime,
    run_id: str,
    output_dir: Path,
    autopilot_output_dir: Path | None,
    odds_status: Mapping[str, Any],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "shadow_autopilot_full_daemon_odds_capture_state_publish_v1",
        "status": "SKIPPED",
        "state_path": relpath(state_path),
        "run_id": run_id,
        "output_dir": relpath(output_dir),
        "autopilot_output_dir": relpath(autopilot_output_dir),
        "reason": None,
    }
    if state_path is None:
        report["reason"] = "state_path_missing"
        return report
    if autopilot_output_dir is None:
        report["reason"] = "autopilot_output_dir_missing"
        return report
    refresh_report_path = autopilot_output_dir / "odds_capture_refresh_report.json"
    refresh_report = load_json(refresh_report_path)
    if not isinstance(refresh_report, Mapping) or refresh_report.get("status") != "SUCCESS":
        report["reason"] = "odds_capture_refresh_report_not_success"
        report["source_report_path"] = relpath(refresh_report_path)
        return report
    next_preferred_window = refresh_report.get("next_preferred_window")
    if not isinstance(next_preferred_window, Mapping):
        report["reason"] = "next_preferred_window_missing"
        report["source_report_path"] = relpath(refresh_report_path)
        return report
    next_window_opens_at = next_preferred_window.get("next_window_opens_at")
    if not next_window_opens_at:
        report["reason"] = "next_window_opens_at_missing"
        report["source_report_path"] = relpath(refresh_report_path)
        return report
    resume_after_local = buffered_odds_capture_resume_after_local(
        next_window_opens_at,
        default_tz=generated_at.tzinfo,
    )
    source_recommended_rerun_after_local = next_preferred_window.get(
        "recommended_rerun_after_local"
    )
    fixed_window_schedule = odds_capture_fixed_window_schedule(
        next_preferred_window,
        current_time=generated_at,
        capture_window_coverage=load_capture_window_coverage_from_status(odds_status),
    )
    fixed_window_schedule = reconcile_odds_capture_schedule_with_direct_status(
        fixed_window_schedule,
        odds_status,
        current_time=generated_at,
    )

    state_payload = {
        "schema_version": "shadow_autopilot_odds_capture_only_state_v1",
        "updated_at": generated_at.isoformat(),
        "run_id": run_id,
        "state_source": "full_daemon",
        "final_status": "ODDS_CAPTURE_ONLY_READY",
        **odds_capture_only_operator_fields_for_report(
            "ODDS_CAPTURE_ONLY_READY",
            odds_status,
        ),
        "output_dir": relpath(output_dir),
        "autopilot_output_dir": relpath(autopilot_output_dir),
        "odds_capture_status": odds_status.get("status"),
        "inserted_live_odds_rows": int_or_zero(odds_status.get("inserted_live_odds_rows")),
        "ready_count": int_or_zero(odds_status.get("ready_count")),
        "status_counts": dict(odds_status.get("status_counts") or {}),
        "blocked_attempt_count": int_or_zero(odds_status.get("blocked_attempt_count")),
        "blocked_attempts": list(odds_status.get("blocked_attempts") or []),
        **t2_odds_capture_status_fields(odds_status),
        "autonomous_live_odds_capture_run_id": odds_status.get("run_id"),
        "autonomous_live_odds_capture_final_status": odds_status.get("final_status")
        or odds_status.get("status"),
        "autonomous_live_odds_capture_operator_status": odds_status.get(
            "operator_status"
        ),
        "capture_window_coverage_status_counts": dict(
            odds_status.get("capture_window_coverage_status_counts") or {}
        ),
        "odds_capture_refresh_status": refresh_report.get("status"),
        "next_preferred_window": dict(next_preferred_window),
        "recommended_rerun_after_local": resume_after_local
        or source_recommended_rerun_after_local,
        "source_recommended_rerun_after_local": source_recommended_rerun_after_local,
        "resume_after_local": resume_after_local,
        "next_window_opens_at": next_window_opens_at,
        "odds_capture_fixed_window_schedule": fixed_window_schedule,
        "next_meaningful_action": fixed_window_schedule.get("next_meaningful_action"),
        "next_meaningful_action_at": fixed_window_schedule.get(
            "next_meaningful_action_at"
        ),
        "window_state_source_updated_at": generated_at.isoformat(),
        "source_report_path": relpath(refresh_report_path),
    }
    write_json(state_path, state_payload)
    report.update(
        {
            "status": "PUBLISHED",
            "reason": "full_daemon_odds_capture_state_published",
            "source_report_path": relpath(refresh_report_path),
            "odds_capture_status": odds_status.get("status"),
            "inserted_live_odds_rows": int_or_zero(
                odds_status.get("inserted_live_odds_rows")
            ),
            "ready_count": int_or_zero(odds_status.get("ready_count")),
            "status_counts": dict(odds_status.get("status_counts") or {}),
            "blocked_attempt_count": int_or_zero(
                odds_status.get("blocked_attempt_count")
            ),
            "blocked_attempts": list(odds_status.get("blocked_attempts") or []),
            **t2_odds_capture_status_fields(odds_status),
            "autonomous_live_odds_capture_run_id": odds_status.get("run_id"),
            "autonomous_live_odds_capture_final_status": odds_status.get(
                "final_status"
            )
            or odds_status.get("status"),
            "autonomous_live_odds_capture_operator_status": odds_status.get(
                "operator_status"
            ),
            "capture_window_coverage_status_counts": dict(
                odds_status.get("capture_window_coverage_status_counts") or {}
            ),
            "odds_capture_refresh_status": refresh_report.get("status"),
            "next_window_opens_at": next_window_opens_at,
            "recommended_rerun_after_local": resume_after_local
            or source_recommended_rerun_after_local,
            "source_recommended_rerun_after_local": source_recommended_rerun_after_local,
            "resume_after_local": resume_after_local,
            "odds_capture_fixed_window_schedule": fixed_window_schedule,
            "next_meaningful_action": fixed_window_schedule.get(
                "next_meaningful_action"
            ),
            "next_meaningful_action_at": fixed_window_schedule.get(
                "next_meaningful_action_at"
            ),
        }
    )
    return report


def run_odds_capture_once(args: argparse.Namespace) -> dict[str, Any]:
    generated_at = datetime.now().astimezone()
    current_time = args.current_time or generated_at.isoformat()
    current_dt = parse_datetime_value(current_time, default_tz=generated_at.tzinfo) or generated_at
    run_id = args.run_id or f"{now_id(generated_at)}_odds_capture"
    evidence_root = args.evidence_root
    output_dir = args.output_dir or (
        evidence_root / f"shadow_autopilot_daemonization_v1_{run_id}"
    )
    output_dir = unique_dir(assert_output_dir_safe(output_dir, evidence_root=evidence_root))
    output_dir.mkdir(parents=True, exist_ok=False)
    preflight_wait = odds_capture_preflight_wait(
        state_path=args.state_path,
        now=current_dt,
    )
    if preflight_wait is not None:
        fixed_window_schedule = preflight_wait.get("odds_capture_fixed_window_schedule") or {}
        report = {
            "schema_version": "shadow_autopilot_odds_capture_only_daemon_report_v1",
            "generated_at": generated_at.isoformat(),
            "run_id": run_id,
            "final_status": "ODDS_CAPTURE_ONLY_WAITING_FOR_WINDOW",
            **odds_capture_only_operator_fields("ODDS_CAPTURE_ONLY_WAITING_FOR_WINDOW"),
            "output_dir": relpath(output_dir),
            "autopilot_output_dir": None,
            "lock_path": relpath(args.lock_path or DEFAULT_LOCK_PATH),
            "lock": None,
            "lock_release": None,
            "steps": [],
            "autopilot_result": None,
            "autonomous_live_odds_capture_status": {},
            "odds_capture_refresh_report": {},
            "preflight_wait": preflight_wait,
            "odds_capture_fixed_window_schedule": fixed_window_schedule,
            "next_meaningful_action": (
                fixed_window_schedule.get("next_meaningful_action")
            ),
            "next_meaningful_action_at": (
                fixed_window_schedule.get("next_meaningful_action_at")
            ),
            **odds_capture_next_race_report_fields(fixed_window_schedule),
            "odds_capture_window_offsets_minutes": list(ODDS_CAPTURE_WINDOW_OFFSETS_MINUTES),
            "timer_frequency_target": DEFAULT_ODDS_CAPTURE_ONLY_TIMER_FREQUENCY,
            "allowed_write_scope": "append_only_live_odds_rows_when_validation_passes",
            "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        }
        write_json(output_dir / "odds_capture_only_daemon_report.json", report)
        write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
        if args.state_path:
            write_json(
                args.state_path,
                {
                    "schema_version": "shadow_autopilot_odds_capture_only_state_v1",
                    "updated_at": generated_at.isoformat(),
                    "run_id": run_id,
                    "final_status": "ODDS_CAPTURE_ONLY_WAITING_FOR_WINDOW",
                    "output_dir": relpath(output_dir),
                    "autopilot_output_dir": None,
                    "odds_capture_status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS",
                    "odds_capture_refresh_status": preflight_wait.get(
                        "odds_capture_refresh_status"
                    ),
                    "inserted_live_odds_rows": 0,
                    "next_preferred_window": preflight_wait.get("next_preferred_window") or {},
                    "recommended_rerun_after_local": preflight_wait.get(
                        "recommended_rerun_after_local"
                    ),
                    "source_recommended_rerun_after_local": preflight_wait.get(
                        "source_recommended_rerun_after_local"
                    ),
                    "next_window_opens_at": preflight_wait.get("next_window_opens_at"),
                    "odds_capture_fixed_window_schedule": fixed_window_schedule,
                    "next_meaningful_action": (
                        fixed_window_schedule.get("next_meaningful_action")
                    ),
                    "next_meaningful_action_at": (
                        fixed_window_schedule.get("next_meaningful_action_at")
                    ),
                    **odds_capture_next_race_report_fields(fixed_window_schedule),
                    "resume_after_local": preflight_wait.get("resume_after_local"),
                    "window_state_source_updated_at": preflight_wait.get(
                        "window_state_source_updated_at"
                    ),
                    "state_source": preflight_wait.get("window_state_source"),
                    "source_report_path": preflight_wait.get("source_report_path"),
                    "preflight_wait": preflight_wait,
                },
            )
        return report
    lock_path = args.lock_path or DEFAULT_LOCK_PATH
    lock_payload: dict[str, Any] | None = None
    release: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = []
    autopilot_result: dict[str, Any] | None = None
    autopilot_output_dir: Path | None = None
    odds_status: dict[str, Any] = {}
    refresh_report: dict[str, Any] = {}
    previous_state = load_json(args.state_path) or {} if args.state_path else {}
    previous_next_preferred_window = (
        previous_state.get("next_preferred_window")
        if isinstance(previous_state, Mapping)
        and isinstance(previous_state.get("next_preferred_window"), Mapping)
        else {}
    )
    pre_lock_fixed_window_schedule = odds_capture_fixed_window_schedule(
        previous_next_preferred_window,
        current_time=current_dt,
    )

    full_daemon_wait_marker = read_active_full_daemon_lock_wait_marker(lock_path)
    try:
        if full_daemon_wait_marker is not None:
            final_status = "SKIPPED_FULL_DAEMON_LOCK_HANDOFF"
            lock_payload = {
                "reason": "full_daemon_waiting_for_odds_lock_handoff",
                "lock_path": relpath(lock_path),
                "full_daemon_wait_marker": full_daemon_wait_marker,
            }
            raise LockBusy(lock_payload)
        lock_payload = acquire_lock_with_t2_due_retry(
            lock_path=lock_path,
            run_id=run_id,
            stale_after_seconds=args.lock_stale_seconds,
            output_dir=output_dir,
            fixed_window_schedule=pre_lock_fixed_window_schedule,
        )
        command = odds_capture_only_autopilot_command(
            run_id=f"{run_id}_autopilot",
            evidence_root=evidence_root,
            current_time=current_time,
            db_path=args.db,
            days_ahead=args.days_ahead,
            refresh_limit=args.refresh_limit,
            odds_capture_min_minutes=args.odds_capture_min_minutes,
            odds_capture_max_minutes=args.odds_capture_max_minutes,
            odds_capture_refresh_limit=args.odds_capture_refresh_limit,
            timeout_seconds=args.timeout_seconds,
            refresh_command_mode=args.refresh_command_mode,
            require_safe_refresh_metadata=args.require_safe_refresh_metadata,
        )
        write_json(
            output_dir / "odds_capture_only_daemon_report.json",
            {
                "schema_version": "shadow_autopilot_odds_capture_only_daemon_report_v1",
                "generated_at": generated_at.isoformat(),
                "run_id": run_id,
                "final_status": "ODDS_CAPTURE_ONLY_RUNNING",
                **odds_capture_only_operator_fields("ODDS_CAPTURE_ONLY_RUNNING"),
                "output_dir": relpath(output_dir),
                "autopilot_output_dir": None,
                "lock_path": relpath(lock_path),
                "lock": lock_payload,
                "lock_release": None,
                "steps": [],
                "autopilot_result": None,
                "autonomous_live_odds_capture_status": {},
                "autonomous_live_odds_capture_status_text": None,
                "odds_capture_refresh_report": {},
                "pre_lock_odds_capture_fixed_window_schedule": (
                    pre_lock_fixed_window_schedule
                ),
                "odds_capture_window_offsets_minutes": list(
                    ODDS_CAPTURE_WINDOW_OFFSETS_MINUTES
                ),
                "timer_frequency_target": DEFAULT_ODDS_CAPTURE_ONLY_TIMER_FREQUENCY,
                "allowed_write_scope": (
                    "append_only_live_odds_rows_when_validation_passes"
                ),
                "autopilot_command": command,
                "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
            },
        )
        write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
        step = run_command(
            name="odds_capture_autopilot_cycle",
            command=command,
            output_dir=output_dir,
            timeout_seconds=args.timeout_seconds,
        )
        steps.append(step)
        autopilot_stdout = output_dir / "logs" / "odds_capture_autopilot_cycle.stdout.txt"
        autopilot_result = load_json(autopilot_stdout)
        if autopilot_result and autopilot_result.get("output_dir"):
            autopilot_output_dir = rooted_path(autopilot_result.get("output_dir"))
        if autopilot_output_dir is not None:
            odds_status = (
                load_json(autopilot_output_dir / "autonomous_live_odds_capture_status.json")
                or {}
            )
            refresh_report = (
                load_json(autopilot_output_dir / "odds_capture_refresh_report.json")
                or {}
            )
        final_status = classify_odds_capture_only_final_status(
            step=step,
            autopilot_result=autopilot_result,
            odds_status=odds_status,
            refresh_report=refresh_report,
        )
    except LockBusy as exc:
        if full_daemon_wait_marker is None:
            final_status = "SKIPPED_LOCK_HELD"
        lock_payload = dict(exc.payload)
    finally:
        if lock_payload and lock_payload.get("run_id") == run_id:
            release = release_lock(lock_path, run_id)

    next_preferred_window_for_report = refresh_report.get("next_preferred_window")
    if (
        final_status == "SKIPPED_LOCK_HELD"
        and not isinstance(next_preferred_window_for_report, Mapping)
        and isinstance(previous_state, Mapping)
    ):
        next_preferred_window_for_report = previous_state.get("next_preferred_window")
    if not isinstance(next_preferred_window_for_report, Mapping):
        next_preferred_window_for_report = {}
    fixed_window_schedule = odds_capture_fixed_window_schedule(
        next_preferred_window_for_report,
        current_time=current_dt,
        capture_window_coverage=load_capture_window_coverage_from_status(odds_status),
    )
    fixed_window_schedule = reconcile_odds_capture_schedule_with_direct_status(
        fixed_window_schedule,
        odds_status,
        current_time=current_dt,
    )
    t2_capture_miss_cause_counts = dict(odds_status.get("t2_miss_cause_counts") or {})
    t2_lock_skip_fields = odds_capture_t2_lock_skip_fields(
        final_status=final_status,
        fixed_window_schedule=fixed_window_schedule,
    )
    t2_miss_cause_counts = dict(t2_capture_miss_cause_counts)
    for cause, count in dict(t2_lock_skip_fields.get("t2_miss_cause_counts") or {}).items():
        t2_miss_cause_counts[str(cause)] = (
            int_or_zero(t2_miss_cause_counts.get(str(cause))) + int_or_zero(count)
        )
    t2_lock_skip_fields["t2_miss_cause_counts"] = t2_miss_cause_counts

    report = {
        "schema_version": "shadow_autopilot_odds_capture_only_daemon_report_v1",
        "generated_at": generated_at.isoformat(),
        "run_id": run_id,
        "final_status": final_status,
        **odds_capture_only_operator_fields_for_report(final_status, odds_status),
        "output_dir": relpath(output_dir),
        "autopilot_output_dir": relpath(autopilot_output_dir),
        "lock_path": relpath(lock_path),
        "lock": lock_payload,
        **lock_owner_report_fields(lock_payload),
        "lock_release": release,
        "steps": steps,
        "autopilot_result": autopilot_result,
        "autonomous_live_odds_capture_status": odds_status,
        "autonomous_live_odds_capture_status_text": odds_status.get("status"),
        "autonomous_live_odds_capture_run_id": odds_status.get("run_id"),
        "autonomous_live_odds_capture_final_status": odds_status.get("final_status")
        or odds_status.get("status"),
        "autonomous_live_odds_capture_operator_status": odds_status.get(
            "operator_status"
        ),
        "inserted_live_odds_rows": int_or_zero(odds_status.get("inserted_live_odds_rows")),
        "ready_count": int_or_zero(odds_status.get("ready_count")),
        "status_counts": dict(odds_status.get("status_counts") or {}),
        "blocked_attempt_count": int_or_zero(odds_status.get("blocked_attempt_count")),
        "blocked_attempts": list(odds_status.get("blocked_attempts") or []),
        "t2_capture_miss_cause_counts": t2_capture_miss_cause_counts,
        "t2_capture_miss_examples": list(odds_status.get("t2_miss_examples") or []),
        **t2_lock_skip_fields,
        "odds_capture_refresh_report": refresh_report,
        "pre_lock_odds_capture_fixed_window_schedule": pre_lock_fixed_window_schedule,
        "odds_capture_fixed_window_schedule": fixed_window_schedule,
        "next_meaningful_action": fixed_window_schedule.get("next_meaningful_action"),
        "next_meaningful_action_at": fixed_window_schedule.get(
            "next_meaningful_action_at"
        ),
        **odds_capture_next_race_report_fields(fixed_window_schedule),
        "odds_capture_window_offsets_minutes": list(ODDS_CAPTURE_WINDOW_OFFSETS_MINUTES),
        "timer_frequency_target": DEFAULT_ODDS_CAPTURE_ONLY_TIMER_FREQUENCY,
        "allowed_write_scope": "append_only_live_odds_rows_when_validation_passes",
        "no_write_guarantees": {
            **NO_WRITE_GUARANTEES,
            **(odds_status.get("no_write_guarantees") or {}),
            "db_write": bool(odds_status.get("inserted_live_odds_rows")),
        },
    }
    write_json(output_dir / "odds_capture_only_daemon_report.json", report)
    write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
    if args.state_path:
        next_preferred_window = next_preferred_window_for_report
        next_window_opens_at = next_preferred_window.get("next_window_opens_at")
        resume_after_local = buffered_odds_capture_resume_after_local(
            next_window_opens_at,
            default_tz=generated_at.tzinfo,
        )
        source_recommended_rerun_after_local = next_preferred_window.get(
            "recommended_rerun_after_local"
        )
        if final_status in {
            "SKIPPED_LOCK_HELD",
            "SKIPPED_FULL_DAEMON_LOCK_HANDOFF",
        } and previous_state.get("next_window_opens_at"):
            state_payload = dict(previous_state)
            state_payload["updated_at"] = generated_at.isoformat()
            state_payload["last_lock_skip"] = {
                "updated_at": generated_at.isoformat(),
                "run_id": run_id,
                "output_dir": relpath(output_dir),
                "lock": lock_payload,
                **lock_owner_report_fields(lock_payload),
                **t2_lock_skip_fields,
            }
        else:
            state_payload = {
                "schema_version": "shadow_autopilot_odds_capture_only_state_v1",
                "updated_at": generated_at.isoformat(),
                "run_id": run_id,
                "final_status": final_status,
                "status": report.get("status"),
                "runtime_action": report.get("runtime_action"),
                "readiness_decision": report.get("readiness_decision"),
                "output_dir": relpath(output_dir),
                "autopilot_output_dir": relpath(autopilot_output_dir),
                "odds_capture_status": odds_status.get("status"),
                "inserted_live_odds_rows": odds_status.get("inserted_live_odds_rows"),
                "ready_count": odds_status.get("ready_count"),
                "status_counts": dict(odds_status.get("status_counts") or {}),
                "blocked_attempt_count": int_or_zero(
                    odds_status.get("blocked_attempt_count")
                ),
                "blocked_attempts": list(odds_status.get("blocked_attempts") or []),
                "t2_capture_miss_cause_counts": dict(
                    odds_status.get("t2_miss_cause_counts") or {}
                ),
                "t2_capture_miss_examples": list(
                    odds_status.get("t2_miss_examples") or []
                ),
                **t2_lock_skip_fields,
                "autonomous_live_odds_capture_run_id": odds_status.get("run_id"),
                "autonomous_live_odds_capture_final_status": odds_status.get(
                    "final_status"
                )
                or odds_status.get("status"),
                "autonomous_live_odds_capture_operator_status": odds_status.get(
                    "operator_status"
                ),
                "capture_window_coverage_status_counts": dict(
                    odds_status.get("capture_window_coverage_status_counts") or {}
                ),
                "odds_capture_refresh_status": refresh_report.get("status"),
                "next_preferred_window": next_preferred_window,
                "recommended_rerun_after_local": resume_after_local
                or source_recommended_rerun_after_local,
                "source_recommended_rerun_after_local": source_recommended_rerun_after_local,
                "resume_after_local": resume_after_local,
                "next_window_opens_at": next_window_opens_at,
                "odds_capture_fixed_window_schedule": fixed_window_schedule,
                "next_meaningful_action": fixed_window_schedule.get(
                    "next_meaningful_action"
                ),
                "next_meaningful_action_at": fixed_window_schedule.get(
                    "next_meaningful_action_at"
                ),
                **odds_capture_next_race_report_fields(fixed_window_schedule),
                "window_state_source_updated_at": generated_at.isoformat(),
            }
        write_json(
            args.state_path,
            state_payload,
        )
    return report


def autopilot_cycle_timeout_seconds(step_timeout_seconds: int) -> int:
    return max(step_timeout_seconds * 2, step_timeout_seconds + 300)


def install_markdown(service_info: Mapping[str, Any]) -> str:
    service_path = service_info.get("service_path")
    timer_path = service_info.get("timer_path")
    return "\n".join(
        [
            "# Shadow Autopilot Service Install",
            "",
            "The service is intentionally not installed or enabled by this script.",
            "Install it only after reviewing the generated unit files.",
            "",
            "```bash",
            f"sudo cp {service_path} /etc/systemd/system/{SERVICE_NAME}",
            f"sudo cp {timer_path} /etc/systemd/system/{TIMER_NAME}",
            "sudo systemctl daemon-reload",
            f"sudo systemctl enable --now {TIMER_NAME}",
            f"systemctl list-timers {TIMER_NAME}",
            f"journalctl -u {SERVICE_NAME} -n 200 --no-pager",
            "```",
            "",
            f"The timer runs every {DEFAULT_TIMER_FREQUENCY} on `{DEFAULT_TIMER_ON_CALENDAR}`. Overlap is prevented by systemd oneshot semantics and the daemon lock file.",
            "",
        ]
    )


def daemon_design_markdown() -> str:
    return "\n".join(
        [
            "# Shadow Autopilot Daemon Design",
            "",
            "## Objective",
            "Continuously collect forward-shadow evidence without changing production state.",
            "",
            "## Architecture",
            f"- `shadow-autopilot.timer` activates every {DEFAULT_TIMER_FREQUENCY} on `{DEFAULT_TIMER_ON_CALENDAR}`.",
            "- `shadow-autopilot.service` runs one bounded `shadow_autopilot_daemon.py run-once` cycle.",
            "- The daemon acquires a JSON lock before any work starts.",
            "- The daemon delegates refresh, scoring, current-run join, aggregate, and status generation to `scripts/shadow_autopilot_v1.py`.",
            "- The daemon then re-runs exact result joins for older pending shadow runs and refreshes aggregate/status artifacts.",
            "- The daemon emits dashboards, alert reports, validation reports, and protected-hash evidence in one packet.",
            "",
            "## Lifecycle",
            "Startup -> acquire lock -> refresh races -> score shadow predictions -> join current run -> rejoin prior pending runs -> aggregate/dashboard -> alert -> release lock -> sleep by timer.",
            "",
            "## Mutation Boundary",
            "Allowed writes are limited to shadow-only evidence artifacts, daemon runtime lock/state, logs, unit-file templates, and explicitly enabled append-only live odds or official-result evidence DB rows. Labels, production pointers, registry files, model artifacts, snapshots, and betting/EV outputs are not written.",
            "",
        ]
    )


def lifecycle_diagram() -> dict[str, Any]:
    return {
        "schema_version": "shadow_autopilot_daemon_lifecycle_v1",
        "nodes": [
            {"id": "startup", "description": "systemd timer starts one daemon cycle"},
            {"id": "lock", "description": "acquire JSON lock and reject overlap"},
            {"id": "refresh", "description": "refresh pre-jump TheDogs races into isolated artifact input dir"},
            {"id": "score", "description": "score shadow predictions with existing shadow model only"},
            {"id": "join_current", "description": "exact official-result join for current shadow run"},
            {"id": "rejoin_pending", "description": "exact official-result rejoin sweep for older pending shadow runs"},
            {"id": "dashboard", "description": "aggregate joins and write shadow status/dashboard/readiness files"},
            {"id": "alert", "description": "compare current metrics to prior packet and emit alert report"},
            {"id": "release", "description": "release lock and persist state"},
            {"id": "sleep", "description": "systemd timer waits 15 minutes before next activation"},
        ],
        "edges": [
            ["startup", "lock"],
            ["lock", "refresh"],
            ["refresh", "score"],
            ["score", "join_current"],
            ["join_current", "rejoin_pending"],
            ["rejoin_pending", "dashboard"],
            ["dashboard", "alert"],
            ["alert", "release"],
            ["release", "sleep"],
        ],
        "fail_closed_edges": [
            {"from": "lock", "to": "release", "reason": "active lock prevents overlap"},
            {"from": "refresh", "to": "dashboard", "reason": "refresh failure still leaves existing evidence reviewable"},
            {"from": "score", "to": "dashboard", "reason": "score failure still refreshes aggregate/status from existing joins"},
        ],
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def normalize_source_key(value: Any) -> str:
    text = str(value or "").rstrip("/")
    if not text:
        return ""
    return text.split("/")[-1]


def join_index(evidence_root: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for metrics_path in sorted(evidence_root.glob("forward_shadow_result_join_*/shadow_forward_metrics.json")):
        metrics = load_json(metrics_path)
        if not metrics:
            continue
        source_key = normalize_source_key(metrics.get("source_shadow_run"))
        if not source_key:
            continue
        mtime = metrics_path.stat().st_mtime
        if source_key in rows and mtime <= float(rows[source_key].get("mtime") or 0):
            continue
        rows[source_key] = {
            "join_dir": metrics_path.parent,
            "metrics": metrics,
            "mtime": mtime,
        }
    return rows


def candidate_shadow_runs(
    *,
    evidence_root: Path,
    pending_limit: int,
    lookback_days: int,
) -> list[dict[str, Any]]:
    joined = join_index(evidence_root)
    cutoff = time.time() - (lookback_days * 86400)
    candidates: list[dict[str, Any]] = []
    for shadow_dir in sorted(evidence_root.glob("daily_race_ingest_shadow_*")):
        manifest = load_json(shadow_dir / "shadow_manifest.json")
        if not manifest:
            continue
        if shadow_dir.stat().st_mtime < cutoff:
            continue
        if int(manifest.get("race_count") or 0) <= 0:
            continue
        if manifest.get("final_status") not in {"FORWARD_SHADOW_RUN_COMPLETE", "NO_ELIGIBLE_CURRENT_OR_FUTURE_RACES"}:
            continue
        key = shadow_dir.name
        latest = joined.get(key)
        latest_metrics = (latest or {}).get("metrics") or {}
        pending_count = int(latest_metrics.get("pending_race_count") or 0)
        if latest is None or pending_count > 0:
            candidates.append(
                {
                    "shadow_run_dir": shadow_dir,
                    "shadow_run_key": key,
                    "latest_join_dir": (latest or {}).get("join_dir"),
                    "latest_pending_count": pending_count if latest else None,
                    "latest_safe_joined_count": int(latest_metrics.get("safe_joined_race_count") or 0),
                    "latest_unsafe_count": int(latest_metrics.get("unsafe_match_count") or 0),
                    "latest_join_mtime": (latest or {}).get("mtime"),
                    "shadow_run_mtime": shadow_dir.stat().st_mtime,
                    "race_count": int(manifest.get("race_count") or 0),
                }
            )
    candidates.sort(
        key=lambda row: (
            0 if row.get("latest_join_mtime") is None else 1,
            float(row.get("latest_join_mtime") or 0),
            float(row["shadow_run_mtime"]),
        )
    )
    return candidates[:pending_limit]


def rejoin_pending_shadow_runs(
    *,
    run_id: str,
    output_dir: Path,
    evidence_root: Path,
    db_path: Path,
    current_time: str,
    pending_limit: int,
    lookback_days: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    candidates = candidate_shadow_runs(
        evidence_root=evidence_root,
        pending_limit=pending_limit,
        lookback_days=lookback_days,
    )
    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        join_dir = evidence_root / f"forward_shadow_result_join_{run_id}_daemon_rejoin_{index:03d}"
        command = [
            sys.executable,
            str(ROOT / "scripts/join_forward_shadow_results.py"),
            "--shadow-run-dir",
            str(candidate["shadow_run_dir"]),
            "--output-dir",
            str(join_dir),
            "--db",
            str(db_path),
            "--current-time",
            current_time,
        ]
        step = run_command(
            name=f"rejoin_pending_{index:03d}",
            command=command,
            output_dir=output_dir,
            timeout_seconds=timeout_seconds,
        )
        metrics = load_json(join_dir / "shadow_forward_metrics.json")
        results.append(
            {
                "candidate": {
                    **candidate,
                    "shadow_run_dir": relpath(candidate["shadow_run_dir"]),
                    "latest_join_dir": relpath(candidate.get("latest_join_dir")),
                },
                "join_dir": relpath(join_dir),
                "step": step,
                "metrics": metrics,
            }
        )
    joined_count = sum(int((row.get("metrics") or {}).get("safe_joined_race_count") or 0) for row in results)
    pending_count = sum(int((row.get("metrics") or {}).get("pending_race_count") or 0) for row in results)
    unsafe_count = sum(int((row.get("metrics") or {}).get("unsafe_match_count") or 0) for row in results)
    return {
        "schema_version": "shadow_autopilot_automated_join_report_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "exact_identity_join_only": True,
        "fuzzy_join_allowed": False,
        "ambiguous_winners_rejected": True,
        "pending_shadow_runs_scanned": len(candidates),
        "rejoin_attempt_count": len(results),
        "rejoin_safe_joined_count_sum": joined_count,
        "rejoin_pending_count_sum": pending_count,
        "rejoin_unsafe_count_sum": unsafe_count,
        "results": results,
    }


def int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def best_unified_evidence_aggregate_status_path(
    paths: Sequence[Path | None],
) -> Path | None:
    candidates: list[tuple[int, int, str, Path]] = []
    for path in paths:
        if path is None or not path.exists():
            continue
        report = load_json(path) or {}
        status = str(report.get("status") or "")
        if status not in {
            "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT",
            "REJOIN_UNIFIED_EVIDENCE_DATASETS_BUILT",
        }:
            continue
        candidates.append(
            (
                int_or_zero(report.get("unified_evidence_eligible_rows")),
                int_or_zero(report.get("row_count")),
                path.as_posix(),
                path,
            )
        )
    if not candidates:
        return None
    return max(candidates)[3]


def compact_unified_gap_rows(rows: Sequence[Any]) -> list[dict[str, Any]]:
    compact_rows: list[dict[str, Any]] = []
    allowed_fields = (
        "race_id",
        "action",
        "recommended_action",
        "evidence_missing_reason",
        "missing_official_result",
        "missing_strict_prejump_odds",
        "official_result_quarantine_reason",
        "official_result_quarantine_errors",
        "official_result_quarantine_source_urls",
        "official_result_quarantine_participant_source",
        "official_result_quarantine_participant_count",
        "official_result_quarantine_participant_boxes",
        "official_result_quarantine_result_boxes_not_in_participants",
        "official_result_quarantine_result_boxes_in_participants",
        "official_result_quarantine_participants",
        "official_result_quarantine_attempted_source_box_sets",
        "official_result_quarantine_reserve_substitution_diagnostic",
    )
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        compact = {
            key: row.get(key)
            for key in allowed_fields
            if row.get(key) not in (None, "", [], {})
        }
        if compact.get("race_id"):
            compact_rows.append(compact)
    return compact_rows


def apply_best_aggregate_unified_evidence_to_daily_status(
    daily_status: dict[str, Any],
    *,
    best_status_path: Path | None,
    best_status: Mapping[str, Any] | None,
) -> None:
    status = best_status or {}
    race_coverage = status.get("race_coverage") or {}
    gap_action_plan = race_coverage.get("gap_action_plan") or {}
    top_gap_race_ids = [
        str(row.get("race_id"))
        for row in gap_action_plan.get("top_gap_races") or []
        if isinstance(row, Mapping) and row.get("race_id")
    ]
    top_official_missing_race_ids = [
        str(row.get("race_id"))
        for row in race_coverage.get("top_official_result_missing_races") or []
        if isinstance(row, Mapping) and row.get("race_id")
    ]
    top_gap_races = compact_unified_gap_rows(gap_action_plan.get("top_gap_races") or [])
    top_official_missing_races = compact_unified_gap_rows(
        race_coverage.get("top_official_result_missing_races") or []
    )
    daily_status["best_aggregate_unified_evidence_status_path"] = relpath(
        best_status_path
    )
    daily_status["best_aggregate_unified_evidence_status"] = status.get("status")
    daily_status["best_aggregate_unified_evidence_dataset_count"] = status.get(
        "dataset_count"
    )
    daily_status["best_aggregate_unified_evidence_failed_dataset_count"] = status.get(
        "failed_dataset_count"
    )
    daily_status["best_aggregate_unified_evidence_row_count"] = status.get(
        "row_count"
    )
    daily_status["best_aggregate_unified_evidence_eligible_rows"] = status.get(
        "unified_evidence_eligible_rows"
    )
    daily_status["best_aggregate_unified_evidence_artifact_odds_rows_seen"] = (
        status.get("artifact_odds_rows_seen")
    )
    daily_status["best_aggregate_unified_evidence_artifact_odds_rows_accepted"] = (
        status.get("artifact_odds_rows_accepted")
    )
    daily_status["best_aggregate_unified_evidence_artifact_odds_rows_rejected"] = (
        status.get("artifact_odds_rows_rejected")
    )
    daily_status[
        "best_aggregate_unified_evidence_artifact_odds_rejection_reason_counts"
    ] = status.get("artifact_odds_rejection_reason_counts") or {}
    daily_status["best_aggregate_unified_rejected_live_odds_candidate_count"] = (
        status.get("rejected_live_odds_candidate_count")
    )
    daily_status[
        "best_aggregate_unified_rows_with_rejected_live_odds_candidates"
    ] = status.get("rows_with_rejected_live_odds_candidates")
    daily_status[
        "best_aggregate_unified_rejected_live_odds_candidate_reason_counts"
    ] = status.get("rejected_live_odds_candidate_reason_counts") or {}
    daily_status["best_aggregate_unified_sample_blocking_gap_count"] = (
        gap_action_plan.get("sample_blocking_gap_count") or 0
    )
    daily_status["best_aggregate_unified_gap_action_counts"] = (
        gap_action_plan.get("action_counts") or {}
    )
    daily_status["best_aggregate_unified_gap_evidence_missing_reason_counts"] = (
        gap_action_plan.get("evidence_missing_reason_counts") or {}
    )
    daily_status["best_aggregate_unified_top_gap_race_ids"] = top_gap_race_ids
    daily_status["best_aggregate_unified_top_gap_races"] = top_gap_races
    daily_status["best_aggregate_unified_top_official_result_missing_race_ids"] = (
        top_official_missing_race_ids
    )
    daily_status["best_aggregate_unified_top_official_result_missing_races"] = (
        top_official_missing_races
    )

    # Backward-compatible operator fields previously came from a narrow latest
    # rejoin packet and could hide the stronger aggregate backlog evidence.
    daily_status["backlog_unified_evidence_status_path"] = relpath(best_status_path)
    daily_status["backlog_unified_evidence_source_status"] = status.get("status")
    daily_status["backlog_unified_evidence_status"] = status.get("status")
    daily_status["backlog_unified_evidence_dataset_count"] = status.get(
        "dataset_count"
    )
    daily_status["backlog_unified_evidence_failed_dataset_count"] = status.get(
        "failed_dataset_count"
    )
    daily_status["backlog_unified_evidence_row_count"] = status.get("row_count")
    daily_status["backlog_unified_evidence_eligible_rows"] = status.get(
        "unified_evidence_eligible_rows"
    )
    daily_status["backlog_unified_rejected_live_odds_candidate_count"] = (
        status.get("rejected_live_odds_candidate_count")
    )
    daily_status[
        "backlog_unified_rows_with_rejected_live_odds_candidates"
    ] = status.get("rows_with_rejected_live_odds_candidates")
    daily_status[
        "backlog_unified_rejected_live_odds_candidate_reason_counts"
    ] = status.get("rejected_live_odds_candidate_reason_counts") or {}
    daily_status["backlog_unified_sample_blocking_gap_count"] = (
        daily_status["best_aggregate_unified_sample_blocking_gap_count"]
    )
    daily_status["backlog_unified_gap_action_counts"] = (
        daily_status["best_aggregate_unified_gap_action_counts"]
    )
    daily_status["backlog_unified_gap_evidence_missing_reason_counts"] = (
        daily_status["best_aggregate_unified_gap_evidence_missing_reason_counts"]
    )
    daily_status["backlog_unified_top_gap_race_ids"] = top_gap_race_ids
    daily_status["backlog_unified_top_gap_races"] = top_gap_races
    daily_status["backlog_unified_top_official_result_missing_race_ids"] = (
        top_official_missing_race_ids
    )
    daily_status["backlog_unified_top_official_result_missing_races"] = (
        top_official_missing_races
    )


AUTOPILOT_CYCLE_DAILY_STATUS_KEYS = (
    "unified_evidence_dataset_status",
    "unified_evidence_dataset_rows",
    "unified_evidence_dataset_races",
    "unified_evidence_eligible_rows",
    "unified_label_evaluation_eligible_rows",
    "unified_odds_evaluation_eligible_rows",
    "unified_stage2_evaluation_eligible_rows",
    "best_aggregate_unified_evidence_status_path",
    "best_aggregate_unified_evidence_status",
    "best_aggregate_unified_evidence_dataset_count",
    "best_aggregate_unified_evidence_failed_dataset_count",
    "best_aggregate_unified_evidence_row_count",
    "best_aggregate_unified_evidence_eligible_rows",
    "best_aggregate_unified_rejected_live_odds_candidate_count",
    "best_aggregate_unified_rows_with_rejected_live_odds_candidates",
    "best_aggregate_unified_rejected_live_odds_candidate_reason_counts",
    "best_aggregate_unified_sample_blocking_gap_count",
    "best_aggregate_unified_gap_action_counts",
    "best_aggregate_unified_gap_evidence_missing_reason_counts",
    "best_aggregate_unified_top_gap_race_ids",
    "best_aggregate_unified_top_gap_races",
    "best_aggregate_unified_top_official_result_missing_race_ids",
    "best_aggregate_unified_top_official_result_missing_races",
    "rolling_model_comparison_status",
    "rolling_model_comparison_sample_races",
    "rolling_model_comparison_sample_runner_rows",
    "rolling_model_comparison_minimum_races_for_review",
    "rolling_model_comparison_best_candidate",
    "rolling_model_comparison_best_top1",
    "rolling_model_comparison_best_top3",
    "rolling_model_comparison_source_rejected_live_odds_candidate_count",
    "rolling_model_comparison_source_rows_with_rejected_live_odds_candidates",
    "rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts",
    "rolling_model_comparison_blockers",
    "high_accuracy_refinement_status",
    "high_accuracy_promotion_pr_gate_status",
    "high_accuracy_unified_evidence_eligible_rows",
    "reserve_substitution_preflight_status",
    "reserve_substitution_preflight_candidate_count",
    "reserve_substitution_preflight_ready_for_policy_review_count",
    "reserve_substitution_preflight_blocked_candidate_count",
    "reserve_substitution_preflight_readiness_blocker_counts",
    "reserve_substitution_preflight_dataset_join_blocker_counts",
    "reserve_substitution_preflight_ready_race_ids",
    "reserve_substitution_preflight_blocked_race_ids",
    "reserve_substitution_preflight_report",
    "reserve_substitution_manual_review_status",
    "reserve_substitution_manual_review_ready_candidate_count",
    "reserve_substitution_manual_review_mapping_pair_count",
    "reserve_substitution_manual_review_dataset_join_allowed",
    "reserve_substitution_manual_review_official_result_acceptance_allowed",
    "reserve_substitution_manual_review_db_write",
    "reserve_substitution_manual_review_blockers",
    "reserve_substitution_manual_review_ready_race_ids",
    "reserve_substitution_manual_review_report",
    "reserve_substitution_policy_impact_status",
    "reserve_substitution_policy_impact_candidate_count",
    "reserve_substitution_policy_impact_ready_candidate_count",
    "reserve_substitution_policy_impact_mapping_pair_count",
    "reserve_substitution_policy_impact_potential_runner_rows_blocked",
    "reserve_substitution_policy_impact_matched_backlog_top_gap_race_count",
    "reserve_substitution_policy_impact_matched_backlog_top_gap_race_ids",
    "reserve_substitution_policy_impact_dataset_join_allowed",
    "reserve_substitution_policy_impact_official_result_acceptance_allowed",
    "reserve_substitution_policy_impact_db_write",
    "reserve_substitution_policy_impact_blockers",
    "reserve_substitution_policy_impact_report",
    "promotion_distance_status",
    "promotion_distance_promotion_ready",
    "promotion_distance_blockers",
    "promotion_distance_sample_race_count",
    "promotion_distance_sample_runner_rows",
    "promotion_distance_source_rejected_live_odds_candidate_count",
    "promotion_distance_source_rows_with_rejected_live_odds_candidates",
    "promotion_distance_source_rejected_live_odds_candidate_reason_counts",
    "promotion_distance_source_exclusion_reason_counts",
    "promotion_distance_source_odds_exclusion_reason_counts",
    "promotion_distance_source_official_result_evidence_db_missing_race_ids",
    "promotion_distance_source_official_result_evidence_db_requested_race_count",
    "promotion_distance_source_official_result_evidence_db_races_with_rows",
    "promotion_distance_source_official_result_runner_paths",
    "promotion_distance_official_result_coverage_requested_race_count",
    "promotion_distance_official_result_coverage_requested_race_count_source",
    "promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids",
    "promotion_distance_official_result_coverage_races_with_rows_count",
    "promotion_distance_official_result_coverage_missing_race_count",
    "promotion_distance_official_result_coverage_missing_exclusion_count",
    "promotion_distance_official_result_runner_path_count",
    "promotion_distance_official_result_runner_paths_source_field",
    "promotion_distance_best_candidate_key",
    "promotion_distance_best_non_market_candidate_key",
    "promotion_distance_best_non_market_top1_margin_gap",
    "promotion_distance_predeclared_residual_candidate_status",
    "promotion_distance_predeclared_residual_triggered_race_count",
    "promotion_distance_report",
    "timing_aligned_rerun_plan",
    "timing_aligned_rerun_execution_status",
    "closer_to_promotion_review",
    "odds_research_gate_status",
    "odds_research_gate_report",
    "odds_research_gate_complete_valid_prejump_odds_races",
    "odds_research_gate_minimum_complete_valid_prejump_odds_races",
    "odds_research_gate_source_url_coverage_pct",
    "odds_research_gate_source_url_rows_missing",
    "odds_research_gate_blocker_counts",
    "odds_research_next_action",
    "timing_aligned_prediction_rerun_required",
    "timing_aligned_prediction_rerun_race_count",
    "timing_aligned_prediction_rerun_race_ids",
    "timing_aligned_prediction_rerun_reason_counts",
    "timing_aligned_prediction_rerun_plan_status",
    "timing_aligned_prediction_rerun_plan_hard_stops",
    "timing_aligned_prediction_rerun_execution_status",
    "timing_aligned_prediction_rerun_execution_hard_stops",
    "timing_aligned_prediction_rerun_execution_performed",
    "timing_aligned_prediction_rerun_output_dir",
    "timing_aligned_prediction_rerun_odds_snapshot_dir",
    "timing_aligned_prediction_rerun_odds_snapshot_status",
    "timing_aligned_prediction_rerun_returncode",
)


def apply_autopilot_cycle_status_to_daily_status(
    daily_status: dict[str, Any],
    *,
    autopilot_daily_status_path: Path | None,
    autopilot_daily_status: Mapping[str, Any] | None,
) -> None:
    status = autopilot_daily_status or {}
    daily_status["autopilot_cycle_daily_status_path"] = relpath(
        autopilot_daily_status_path
    )
    for key in AUTOPILOT_CYCLE_DAILY_STATUS_KEYS:
        if key in status:
            daily_status[key] = status.get(key)


def autopilot_cycle_state_fields(
    daily_status: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "last_autopilot_cycle_daily_status_path": daily_status.get(
            "autopilot_cycle_daily_status_path"
        ),
        "last_unified_evidence_dataset_status": daily_status.get(
            "unified_evidence_dataset_status"
        ),
        "last_unified_evidence_dataset_rows": daily_status.get(
            "unified_evidence_dataset_rows"
        ),
        "last_unified_evidence_dataset_races": daily_status.get(
            "unified_evidence_dataset_races"
        ),
        "last_unified_evidence_eligible_rows": daily_status.get(
            "unified_evidence_eligible_rows"
        ),
        "last_best_aggregate_unified_evidence_status": daily_status.get(
            "best_aggregate_unified_evidence_status"
        ),
        "last_best_aggregate_unified_evidence_eligible_rows": daily_status.get(
            "best_aggregate_unified_evidence_eligible_rows"
        ),
        "last_best_aggregate_unified_rejected_live_odds_candidate_count": (
            daily_status.get("best_aggregate_unified_rejected_live_odds_candidate_count")
        ),
        "last_best_aggregate_unified_rows_with_rejected_live_odds_candidates": (
            daily_status.get(
                "best_aggregate_unified_rows_with_rejected_live_odds_candidates"
            )
        ),
        "last_best_aggregate_unified_rejected_live_odds_candidate_reason_counts": (
            daily_status.get(
                "best_aggregate_unified_rejected_live_odds_candidate_reason_counts"
            )
            or {}
        ),
        "last_best_aggregate_unified_sample_blocking_gap_count": daily_status.get(
            "best_aggregate_unified_sample_blocking_gap_count"
        ),
        "last_best_aggregate_unified_gap_action_counts": daily_status.get(
            "best_aggregate_unified_gap_action_counts"
        )
        or {},
        "last_best_aggregate_unified_gap_evidence_missing_reason_counts": daily_status.get(
            "best_aggregate_unified_gap_evidence_missing_reason_counts"
        )
        or {},
        "last_best_aggregate_unified_top_gap_race_ids": daily_status.get(
            "best_aggregate_unified_top_gap_race_ids"
        )
        or [],
        "last_best_aggregate_unified_top_gap_races": daily_status.get(
            "best_aggregate_unified_top_gap_races"
        )
        or [],
        "last_best_aggregate_unified_top_official_result_missing_race_ids": daily_status.get(
            "best_aggregate_unified_top_official_result_missing_race_ids"
        )
        or [],
        "last_best_aggregate_unified_top_official_result_missing_races": daily_status.get(
            "best_aggregate_unified_top_official_result_missing_races"
        )
        or [],
        "last_rolling_model_comparison_status": daily_status.get(
            "rolling_model_comparison_status"
        ),
        "last_rolling_model_comparison_sample_races": daily_status.get(
            "rolling_model_comparison_sample_races"
        ),
        "last_rolling_model_comparison_sample_runner_rows": daily_status.get(
            "rolling_model_comparison_sample_runner_rows"
        ),
        "last_rolling_model_comparison_best_candidate": daily_status.get(
            "rolling_model_comparison_best_candidate"
        ),
        "last_rolling_model_comparison_source_rejected_live_odds_candidate_count": (
            daily_status.get(
                "rolling_model_comparison_source_rejected_live_odds_candidate_count"
            )
        ),
        "last_rolling_model_comparison_source_rows_with_rejected_live_odds_candidates": (
            daily_status.get(
                "rolling_model_comparison_source_rows_with_rejected_live_odds_candidates"
            )
        ),
        "last_rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts": (
            daily_status.get(
                "rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts"
            )
            or {}
        ),
        "last_rolling_model_comparison_blockers": daily_status.get(
            "rolling_model_comparison_blockers"
        )
        or [],
        "last_high_accuracy_refinement_status": daily_status.get(
            "high_accuracy_refinement_status"
        ),
        "last_high_accuracy_promotion_pr_gate_status": daily_status.get(
            "high_accuracy_promotion_pr_gate_status"
        ),
        "last_high_accuracy_unified_evidence_eligible_rows": daily_status.get(
            "high_accuracy_unified_evidence_eligible_rows"
        ),
        "last_reserve_substitution_policy_impact_status": daily_status.get(
            "reserve_substitution_policy_impact_status"
        ),
        "last_reserve_substitution_policy_impact_ready_candidate_count": daily_status.get(
            "reserve_substitution_policy_impact_ready_candidate_count"
        ),
        "last_reserve_substitution_policy_impact_mapping_pair_count": daily_status.get(
            "reserve_substitution_policy_impact_mapping_pair_count"
        ),
        "last_reserve_substitution_policy_impact_potential_runner_rows_blocked": (
            daily_status.get(
                "reserve_substitution_policy_impact_potential_runner_rows_blocked"
            )
        ),
        "last_reserve_substitution_policy_impact_matched_backlog_top_gap_race_count": (
            daily_status.get(
                "reserve_substitution_policy_impact_matched_backlog_top_gap_race_count"
            )
        ),
        "last_reserve_substitution_policy_impact_dataset_join_allowed": (
            daily_status.get("reserve_substitution_policy_impact_dataset_join_allowed")
        ),
        "last_reserve_substitution_policy_impact_official_result_acceptance_allowed": (
            daily_status.get(
                "reserve_substitution_policy_impact_official_result_acceptance_allowed"
            )
        ),
        "last_reserve_substitution_policy_impact_db_write": daily_status.get(
            "reserve_substitution_policy_impact_db_write"
        ),
        "last_promotion_distance_status": daily_status.get(
            "promotion_distance_status"
        ),
        "last_promotion_distance_promotion_ready": daily_status.get(
            "promotion_distance_promotion_ready"
        ),
        "last_promotion_distance_blockers": daily_status.get(
            "promotion_distance_blockers"
        )
        or [],
        "last_promotion_distance_sample_race_count": daily_status.get(
            "promotion_distance_sample_race_count"
        ),
        "last_promotion_distance_sample_runner_rows": daily_status.get(
            "promotion_distance_sample_runner_rows"
        ),
        "last_promotion_distance_source_rejected_live_odds_candidate_count": (
            daily_status.get("promotion_distance_source_rejected_live_odds_candidate_count")
        ),
        "last_promotion_distance_source_rows_with_rejected_live_odds_candidates": (
            daily_status.get("promotion_distance_source_rows_with_rejected_live_odds_candidates")
        ),
        "last_promotion_distance_source_rejected_live_odds_candidate_reason_counts": (
            daily_status.get(
                "promotion_distance_source_rejected_live_odds_candidate_reason_counts"
            )
            or {}
        ),
        "last_promotion_distance_source_exclusion_reason_counts": (
            daily_status.get("promotion_distance_source_exclusion_reason_counts") or {}
        ),
        "last_promotion_distance_source_odds_exclusion_reason_counts": (
            daily_status.get("promotion_distance_source_odds_exclusion_reason_counts")
            or {}
        ),
        "last_promotion_distance_source_official_result_evidence_db_missing_race_ids": (
            daily_status.get(
                "promotion_distance_source_official_result_evidence_db_missing_race_ids"
            )
            or []
        ),
        "last_promotion_distance_source_official_result_evidence_db_requested_race_count": (
            daily_status.get(
                "promotion_distance_source_official_result_evidence_db_requested_race_count"
            )
        ),
        "last_promotion_distance_source_official_result_evidence_db_races_with_rows": (
            daily_status.get(
                "promotion_distance_source_official_result_evidence_db_races_with_rows"
            )
            or []
        ),
        "last_promotion_distance_source_official_result_runner_paths": (
            daily_status.get("promotion_distance_source_official_result_runner_paths")
            or []
        ),
        "last_promotion_distance_official_result_coverage_requested_race_count": (
            daily_status.get(
                "promotion_distance_official_result_coverage_requested_race_count"
            )
        ),
        "last_promotion_distance_official_result_coverage_requested_race_count_source": (
            daily_status.get(
                "promotion_distance_official_result_coverage_requested_race_count_source"
            )
        ),
        "last_promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids": (
            daily_status.get(
                "promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids"
            )
        ),
        "last_promotion_distance_official_result_coverage_races_with_rows_count": (
            daily_status.get(
                "promotion_distance_official_result_coverage_races_with_rows_count"
            )
        ),
        "last_promotion_distance_official_result_coverage_missing_race_count": (
            daily_status.get(
                "promotion_distance_official_result_coverage_missing_race_count"
            )
        ),
        "last_promotion_distance_official_result_coverage_missing_exclusion_count": (
            daily_status.get(
                "promotion_distance_official_result_coverage_missing_exclusion_count"
            )
        ),
        "last_promotion_distance_official_result_runner_path_count": (
            daily_status.get("promotion_distance_official_result_runner_path_count")
        ),
        "last_promotion_distance_official_result_runner_paths_source_field": (
            daily_status.get(
                "promotion_distance_official_result_runner_paths_source_field"
            )
        ),
        "last_promotion_distance_best_candidate_key": daily_status.get(
            "promotion_distance_best_candidate_key"
        ),
        "last_promotion_distance_best_non_market_candidate_key": daily_status.get(
            "promotion_distance_best_non_market_candidate_key"
        ),
        "last_promotion_distance_predeclared_residual_candidate_status": daily_status.get(
            "promotion_distance_predeclared_residual_candidate_status"
        ),
        "last_promotion_distance_predeclared_residual_triggered_race_count": daily_status.get(
            "promotion_distance_predeclared_residual_triggered_race_count"
        ),
        "last_promotion_distance_report": daily_status.get("promotion_distance_report"),
        "last_timing_aligned_rerun_plan": daily_status.get(
            "timing_aligned_rerun_plan"
        ),
        "last_timing_aligned_rerun_execution_status": daily_status.get(
            "timing_aligned_rerun_execution_status"
        ),
        "last_timing_aligned_prediction_rerun_plan_status": daily_status.get(
            "timing_aligned_prediction_rerun_plan_status"
        ),
        "last_timing_aligned_prediction_rerun_plan_hard_stops": daily_status.get(
            "timing_aligned_prediction_rerun_plan_hard_stops"
        )
        or [],
        "last_timing_aligned_prediction_rerun_execution_status": daily_status.get(
            "timing_aligned_prediction_rerun_execution_status"
        ),
        "last_timing_aligned_prediction_rerun_execution_hard_stops": daily_status.get(
            "timing_aligned_prediction_rerun_execution_hard_stops"
        )
        or [],
        "last_timing_aligned_prediction_rerun_execution_performed": daily_status.get(
            "timing_aligned_prediction_rerun_execution_performed"
        )
        is True,
        "last_timing_aligned_prediction_rerun_output_dir": daily_status.get(
            "timing_aligned_prediction_rerun_output_dir"
        ),
        "last_timing_aligned_prediction_rerun_odds_snapshot_dir": daily_status.get(
            "timing_aligned_prediction_rerun_odds_snapshot_dir"
        ),
        "last_timing_aligned_prediction_rerun_odds_snapshot_status": daily_status.get(
            "timing_aligned_prediction_rerun_odds_snapshot_status"
        ),
        "last_timing_aligned_prediction_rerun_returncode": daily_status.get(
            "timing_aligned_prediction_rerun_returncode"
        ),
    }


def rejoin_unified_state_fields(
    rejoin_unified_status: Mapping[str, Any],
) -> dict[str, Any]:
    status = rejoin_unified_status or {}
    return {
        "last_rejoin_unified_evidence_status": status.get("status"),
        "last_rejoin_unified_evidence_status_reason": status.get("status_reason"),
        "last_rejoin_unified_evidence_dataset_count": status.get("dataset_count"),
        "last_rejoin_unified_evidence_eligible_rows": status.get(
            "unified_evidence_eligible_rows"
        ),
        "last_rejoin_unified_evaluated_candidate_count": status.get(
            "evaluated_dataset_candidate_count"
        ),
        "last_rejoin_unified_skipped_dataset_count": status.get(
            "skipped_dataset_count"
        ),
        "last_rejoin_unified_skip_reason_counts": (
            status.get("skip_reason_counts") or {}
        ),
        "last_rejoin_unified_failure_reason_counts": (
            status.get("failure_reason_counts") or {}
        ),
        "last_rejoin_unified_rejected_live_odds_candidate_count": (
            status.get("rejected_live_odds_candidate_count")
        ),
        "last_rejoin_unified_rows_with_rejected_live_odds_candidates": (
            status.get("rows_with_rejected_live_odds_candidates")
        ),
        "last_rejoin_unified_rejected_live_odds_candidate_reason_counts": (
            status.get("rejected_live_odds_candidate_reason_counts") or {}
        ),
        "last_join_eligibility_preview_dataset_count": status.get(
            "join_eligibility_preview_dataset_count"
        ),
        "last_join_eligibility_preview_unified_eligible_rows": status.get(
            "join_eligibility_preview_unified_eligible_rows"
        ),
        "last_join_eligibility_preview_packet_accepted_races": status.get(
            "join_eligibility_preview_packet_accepted_races"
        ),
        "last_join_eligibility_preview_packet_present_races": status.get(
            "join_eligibility_preview_packet_present_races"
        ),
    }


def rejoin_high_accuracy_timing_source_fields(
    rejoin_high_accuracy_status: Mapping[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    status = rejoin_high_accuracy_status or {}
    return {
        f"{prefix}timing_aligned_rerun_plan": status.get(
            "timing_aligned_rerun_plan"
        ),
        f"{prefix}timing_aligned_rerun_execution_status": status.get(
            "timing_aligned_rerun_execution_status"
        ),
        f"{prefix}reserve_substitution_preflight_status": status.get(
            "reserve_substitution_preflight_status"
        ),
        f"{prefix}reserve_substitution_preflight_ready_for_policy_review_count": (
            status.get("reserve_substitution_preflight_ready_for_policy_review_count")
        ),
        f"{prefix}reserve_substitution_preflight_dataset_join_blocker_counts": (
            status.get("reserve_substitution_preflight_dataset_join_blocker_counts")
        ),
        f"{prefix}reserve_substitution_preflight_ready_race_ids": status.get(
            "reserve_substitution_preflight_ready_race_ids"
        ),
        f"{prefix}reserve_substitution_preflight_report": status.get(
            "reserve_substitution_preflight_report"
        ),
        f"{prefix}reserve_substitution_manual_review_status": status.get(
            "reserve_substitution_manual_review_status"
        ),
        f"{prefix}reserve_substitution_manual_review_ready_candidate_count": (
            status.get("reserve_substitution_manual_review_ready_candidate_count")
        ),
        f"{prefix}reserve_substitution_manual_review_mapping_pair_count": (
            status.get("reserve_substitution_manual_review_mapping_pair_count")
        ),
        f"{prefix}reserve_substitution_manual_review_dataset_join_allowed": (
            status.get("reserve_substitution_manual_review_dataset_join_allowed")
        ),
        f"{prefix}reserve_substitution_manual_review_official_result_acceptance_allowed": (
            status.get(
                "reserve_substitution_manual_review_official_result_acceptance_allowed"
            )
        ),
        f"{prefix}reserve_substitution_manual_review_db_write": status.get(
            "reserve_substitution_manual_review_db_write"
        ),
        f"{prefix}reserve_substitution_manual_review_report": status.get(
            "reserve_substitution_manual_review_report"
        ),
        f"{prefix}reserve_substitution_policy_impact_status": status.get(
            "reserve_substitution_policy_impact_status"
        ),
        f"{prefix}reserve_substitution_policy_impact_ready_candidate_count": (
            status.get("reserve_substitution_policy_impact_ready_candidate_count")
        ),
        f"{prefix}reserve_substitution_policy_impact_mapping_pair_count": (
            status.get("reserve_substitution_policy_impact_mapping_pair_count")
        ),
        f"{prefix}reserve_substitution_policy_impact_potential_runner_rows_blocked": (
            status.get("reserve_substitution_policy_impact_potential_runner_rows_blocked")
        ),
        f"{prefix}reserve_substitution_policy_impact_matched_backlog_top_gap_race_count": (
            status.get(
                "reserve_substitution_policy_impact_matched_backlog_top_gap_race_count"
            )
        ),
        f"{prefix}reserve_substitution_policy_impact_dataset_join_allowed": (
            status.get("reserve_substitution_policy_impact_dataset_join_allowed")
        ),
        f"{prefix}reserve_substitution_policy_impact_official_result_acceptance_allowed": (
            status.get(
                "reserve_substitution_policy_impact_official_result_acceptance_allowed"
            )
        ),
        f"{prefix}reserve_substitution_policy_impact_db_write": status.get(
            "reserve_substitution_policy_impact_db_write"
        ),
        f"{prefix}reserve_substitution_policy_impact_report": status.get(
            "reserve_substitution_policy_impact_report"
        ),
    }


def annotate_rejoin_skipped_status(
    status: Mapping[str, Any] | None,
    rejoin_unified_status: Mapping[str, Any],
) -> dict[str, Any]:
    annotated = dict(status or {})
    if annotated.get("status") != "SKIPPED":
        return annotated

    upstream = rejoin_unified_status or {}
    annotated.update(
        {
            "rejoin_unified_evidence_status": upstream.get("status"),
            "rejoin_unified_evidence_status_reason": upstream.get("status_reason"),
            "rejoin_unified_evidence_dataset_count": upstream.get("dataset_count"),
            "rejoin_unified_evidence_evaluated_candidate_count": upstream.get(
                "evaluated_dataset_candidate_count"
            ),
            "rejoin_unified_evidence_skipped_dataset_count": upstream.get(
                "skipped_dataset_count"
            ),
            "rejoin_unified_evidence_skip_reason_counts": (
                upstream.get("skip_reason_counts") or {}
            ),
            "rejoin_unified_evidence_failure_reason_counts": (
                upstream.get("failure_reason_counts") or {}
            ),
        }
    )
    return annotated


def rejoin_unified_operational_diagnostic_fields(
    rejoin_unified_status: Mapping[str, Any],
) -> dict[str, Any]:
    status = rejoin_unified_status or {}
    return {
        "rejoin_unified_evidence_status_reason": status.get("status_reason"),
        "rejoin_unified_evidence_evaluated_candidate_count": status.get(
            "evaluated_dataset_candidate_count"
        ),
        "rejoin_unified_evidence_skipped_dataset_count": status.get(
            "skipped_dataset_count"
        ),
        "rejoin_unified_evidence_skip_reason_counts": (
            status.get("skip_reason_counts") or {}
        ),
        "rejoin_unified_evidence_failure_reason_counts": (
            status.get("failure_reason_counts") or {}
        ),
    }


def autopilot_cycle_operational_fields(
    daily_status: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "autopilot_cycle_daily_status_path": daily_status.get(
            "autopilot_cycle_daily_status_path"
        ),
        "autopilot_cycle_unified_evidence_status": daily_status.get(
            "unified_evidence_dataset_status"
        ),
        "autopilot_cycle_unified_evidence_rows": daily_status.get(
            "unified_evidence_dataset_rows"
        ),
        "autopilot_cycle_unified_evidence_races": daily_status.get(
            "unified_evidence_dataset_races"
        ),
        "autopilot_cycle_unified_evidence_eligible_rows": daily_status.get(
            "unified_evidence_eligible_rows"
        ),
        "autopilot_cycle_best_aggregate_unified_evidence_status_path": daily_status.get(
            "best_aggregate_unified_evidence_status_path"
        ),
        "autopilot_cycle_best_aggregate_unified_evidence_status": daily_status.get(
            "best_aggregate_unified_evidence_status"
        ),
        "autopilot_cycle_best_aggregate_unified_evidence_dataset_count": daily_status.get(
            "best_aggregate_unified_evidence_dataset_count"
        ),
        "autopilot_cycle_best_aggregate_unified_evidence_row_count": daily_status.get(
            "best_aggregate_unified_evidence_row_count"
        ),
        "autopilot_cycle_best_aggregate_unified_evidence_eligible_rows": daily_status.get(
            "best_aggregate_unified_evidence_eligible_rows"
        ),
        "autopilot_cycle_best_aggregate_unified_evidence_artifact_odds_rows_seen": (
            daily_status.get("best_aggregate_unified_evidence_artifact_odds_rows_seen")
        ),
        "autopilot_cycle_best_aggregate_unified_evidence_artifact_odds_rows_accepted": (
            daily_status.get(
                "best_aggregate_unified_evidence_artifact_odds_rows_accepted"
            )
        ),
        "autopilot_cycle_best_aggregate_unified_evidence_artifact_odds_rows_rejected": (
            daily_status.get(
                "best_aggregate_unified_evidence_artifact_odds_rows_rejected"
            )
        ),
        "autopilot_cycle_best_aggregate_unified_evidence_artifact_odds_rejection_reason_counts": (
            daily_status.get(
                "best_aggregate_unified_evidence_artifact_odds_rejection_reason_counts"
            )
            or {}
        ),
        "autopilot_cycle_best_aggregate_unified_sample_blocking_gap_count": daily_status.get(
            "best_aggregate_unified_sample_blocking_gap_count"
        ),
        "autopilot_cycle_best_aggregate_unified_gap_action_counts": daily_status.get(
            "best_aggregate_unified_gap_action_counts"
        )
        or {},
        "autopilot_cycle_best_aggregate_unified_gap_evidence_missing_reason_counts": daily_status.get(
            "best_aggregate_unified_gap_evidence_missing_reason_counts"
        )
        or {},
        "autopilot_cycle_best_aggregate_unified_top_gap_race_ids": daily_status.get(
            "best_aggregate_unified_top_gap_race_ids"
        )
        or [],
        "autopilot_cycle_best_aggregate_unified_top_gap_races": daily_status.get(
            "best_aggregate_unified_top_gap_races"
        )
        or [],
        "autopilot_cycle_best_aggregate_unified_top_official_result_missing_race_ids": daily_status.get(
            "best_aggregate_unified_top_official_result_missing_race_ids"
        )
        or [],
        "autopilot_cycle_best_aggregate_unified_top_official_result_missing_races": daily_status.get(
            "best_aggregate_unified_top_official_result_missing_races"
        )
        or [],
        "autopilot_cycle_rolling_model_comparison_status": daily_status.get(
            "rolling_model_comparison_status"
        ),
        "autopilot_cycle_rolling_model_comparison_sample_races": daily_status.get(
            "rolling_model_comparison_sample_races"
        ),
        "autopilot_cycle_rolling_model_comparison_best_candidate": daily_status.get(
            "rolling_model_comparison_best_candidate"
        ),
        "autopilot_cycle_rolling_model_comparison_source_rejected_live_odds_candidate_count": (
            daily_status.get(
                "rolling_model_comparison_source_rejected_live_odds_candidate_count"
            )
        ),
        "autopilot_cycle_rolling_model_comparison_source_rows_with_rejected_live_odds_candidates": (
            daily_status.get(
                "rolling_model_comparison_source_rows_with_rejected_live_odds_candidates"
            )
        ),
        "autopilot_cycle_rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts": (
            daily_status.get(
                "rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts"
            )
            or {}
        ),
        "autopilot_cycle_rolling_model_comparison_blockers": daily_status.get(
            "rolling_model_comparison_blockers"
        )
        or [],
        "autopilot_cycle_high_accuracy_refinement_status": daily_status.get(
            "high_accuracy_refinement_status"
        ),
        "autopilot_cycle_high_accuracy_promotion_pr_gate_status": daily_status.get(
            "high_accuracy_promotion_pr_gate_status"
        ),
        "autopilot_cycle_reserve_substitution_policy_impact_status": daily_status.get(
            "reserve_substitution_policy_impact_status"
        ),
        "autopilot_cycle_reserve_substitution_policy_impact_ready_candidate_count": (
            daily_status.get("reserve_substitution_policy_impact_ready_candidate_count")
        ),
        "autopilot_cycle_reserve_substitution_policy_impact_mapping_pair_count": (
            daily_status.get("reserve_substitution_policy_impact_mapping_pair_count")
        ),
        "autopilot_cycle_reserve_substitution_policy_impact_potential_runner_rows_blocked": (
            daily_status.get(
                "reserve_substitution_policy_impact_potential_runner_rows_blocked"
            )
        ),
        "autopilot_cycle_reserve_substitution_policy_impact_matched_backlog_top_gap_race_count": (
            daily_status.get(
                "reserve_substitution_policy_impact_matched_backlog_top_gap_race_count"
            )
        ),
        "autopilot_cycle_reserve_substitution_policy_impact_dataset_join_allowed": (
            daily_status.get("reserve_substitution_policy_impact_dataset_join_allowed")
        ),
        "autopilot_cycle_reserve_substitution_policy_impact_official_result_acceptance_allowed": (
            daily_status.get(
                "reserve_substitution_policy_impact_official_result_acceptance_allowed"
            )
        ),
        "autopilot_cycle_reserve_substitution_policy_impact_db_write": (
            daily_status.get("reserve_substitution_policy_impact_db_write")
        ),
        "autopilot_cycle_promotion_distance_status": daily_status.get(
            "promotion_distance_status"
        ),
        "autopilot_cycle_promotion_distance_promotion_ready": daily_status.get(
            "promotion_distance_promotion_ready"
        ),
        "autopilot_cycle_promotion_distance_blockers": daily_status.get(
            "promotion_distance_blockers"
        )
        or [],
        "autopilot_cycle_promotion_distance_sample_race_count": daily_status.get(
            "promotion_distance_sample_race_count"
        ),
        "autopilot_cycle_promotion_distance_source_rejected_live_odds_candidate_count": (
            daily_status.get("promotion_distance_source_rejected_live_odds_candidate_count")
        ),
        "autopilot_cycle_promotion_distance_source_rows_with_rejected_live_odds_candidates": (
            daily_status.get("promotion_distance_source_rows_with_rejected_live_odds_candidates")
        ),
        "autopilot_cycle_promotion_distance_source_rejected_live_odds_candidate_reason_counts": (
            daily_status.get(
                "promotion_distance_source_rejected_live_odds_candidate_reason_counts"
            )
            or {}
        ),
        "autopilot_cycle_promotion_distance_source_exclusion_reason_counts": (
            daily_status.get("promotion_distance_source_exclusion_reason_counts") or {}
        ),
        "autopilot_cycle_promotion_distance_source_odds_exclusion_reason_counts": (
            daily_status.get("promotion_distance_source_odds_exclusion_reason_counts")
            or {}
        ),
        "autopilot_cycle_promotion_distance_source_official_result_evidence_db_missing_race_ids": (
            daily_status.get(
                "promotion_distance_source_official_result_evidence_db_missing_race_ids"
            )
            or []
        ),
        "autopilot_cycle_promotion_distance_source_official_result_evidence_db_requested_race_count": (
            daily_status.get(
                "promotion_distance_source_official_result_evidence_db_requested_race_count"
            )
        ),
        "autopilot_cycle_promotion_distance_source_official_result_evidence_db_races_with_rows": (
            daily_status.get(
                "promotion_distance_source_official_result_evidence_db_races_with_rows"
            )
            or []
        ),
        "autopilot_cycle_promotion_distance_source_official_result_runner_paths": (
            daily_status.get("promotion_distance_source_official_result_runner_paths")
            or []
        ),
        "autopilot_cycle_promotion_distance_official_result_coverage_requested_race_count": (
            daily_status.get(
                "promotion_distance_official_result_coverage_requested_race_count"
            )
        ),
        "autopilot_cycle_promotion_distance_official_result_coverage_requested_race_count_source": (
            daily_status.get(
                "promotion_distance_official_result_coverage_requested_race_count_source"
            )
        ),
        "autopilot_cycle_promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids": (
            daily_status.get(
                "promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids"
            )
        ),
        "autopilot_cycle_promotion_distance_official_result_coverage_races_with_rows_count": (
            daily_status.get(
                "promotion_distance_official_result_coverage_races_with_rows_count"
            )
        ),
        "autopilot_cycle_promotion_distance_official_result_coverage_missing_race_count": (
            daily_status.get(
                "promotion_distance_official_result_coverage_missing_race_count"
            )
        ),
        "autopilot_cycle_promotion_distance_official_result_coverage_missing_exclusion_count": (
            daily_status.get(
                "promotion_distance_official_result_coverage_missing_exclusion_count"
            )
        ),
        "autopilot_cycle_promotion_distance_official_result_runner_path_count": (
            daily_status.get("promotion_distance_official_result_runner_path_count")
        ),
        "autopilot_cycle_promotion_distance_official_result_runner_paths_source_field": (
            daily_status.get(
                "promotion_distance_official_result_runner_paths_source_field"
            )
        ),
        "autopilot_cycle_promotion_distance_best_candidate_key": daily_status.get(
            "promotion_distance_best_candidate_key"
        ),
        "autopilot_cycle_promotion_distance_best_non_market_candidate_key": daily_status.get(
            "promotion_distance_best_non_market_candidate_key"
        ),
        "autopilot_cycle_promotion_distance_report": daily_status.get(
            "promotion_distance_report"
        ),
        "autopilot_cycle_timing_aligned_rerun_plan": daily_status.get(
            "timing_aligned_rerun_plan"
        ),
        "autopilot_cycle_timing_aligned_rerun_execution_status_path": daily_status.get(
            "timing_aligned_rerun_execution_status"
        ),
        "autopilot_cycle_timing_aligned_rerun_plan_status": daily_status.get(
            "timing_aligned_prediction_rerun_plan_status"
        ),
        "autopilot_cycle_timing_aligned_rerun_plan_hard_stops": daily_status.get(
            "timing_aligned_prediction_rerun_plan_hard_stops"
        )
        or [],
        "autopilot_cycle_timing_aligned_rerun_execution_status": daily_status.get(
            "timing_aligned_prediction_rerun_execution_status"
        ),
        "autopilot_cycle_timing_aligned_rerun_execution_hard_stops": daily_status.get(
            "timing_aligned_prediction_rerun_execution_hard_stops"
        )
        or [],
        "autopilot_cycle_timing_aligned_rerun_execution_performed": daily_status.get(
            "timing_aligned_prediction_rerun_execution_performed"
        )
        is True,
        "autopilot_cycle_timing_aligned_rerun_output_dir": daily_status.get(
            "timing_aligned_prediction_rerun_output_dir"
        ),
        "autopilot_cycle_timing_aligned_rerun_odds_snapshot_dir": daily_status.get(
            "timing_aligned_prediction_rerun_odds_snapshot_dir"
        ),
        "autopilot_cycle_timing_aligned_rerun_odds_snapshot_status": daily_status.get(
            "timing_aligned_prediction_rerun_odds_snapshot_status"
        ),
        "autopilot_cycle_timing_aligned_rerun_returncode": daily_status.get(
            "timing_aligned_prediction_rerun_returncode"
        ),
    }


def autonomous_official_result_operational_fields(
    daily_status: Mapping[str, Any],
    capture_status: Mapping[str, Any],
    evidence_inserted_rows: int,
) -> dict[str, Any]:
    def value(
        daily_key: str,
        capture_key: str | None = None,
        default: Any = None,
    ) -> Any:
        daily_value = daily_status.get(daily_key)
        if daily_value is not None:
            return daily_value
        return capture_status.get(capture_key or daily_key, default)

    return {
        "autonomous_official_result_candidate_count": value(
            "autonomous_official_result_candidate_count",
            "candidate_count",
            0,
        ),
        "autonomous_official_result_race_rows": value(
            "autonomous_official_result_race_rows",
            "official_result_race_rows",
            0,
        ),
        "autonomous_official_result_runner_rows": value(
            "autonomous_official_result_runner_rows",
            "official_result_runner_rows",
            0,
        ),
        "autonomous_official_result_quarantine_rows": value(
            "autonomous_official_result_quarantine_rows",
            "quarantine_rows",
            0,
        ),
        "autonomous_official_result_evidence_inserted_rows": value(
            "autonomous_official_result_evidence_inserted_rows",
            default=evidence_inserted_rows,
        ),
        "autonomous_official_result_evidence_db_ingest_status": value(
            "autonomous_official_result_evidence_db_ingest_status",
            "official_result_evidence_db_ingest_status",
        ),
        "autonomous_official_result_evidence_db_execute": value(
            "autonomous_official_result_evidence_db_execute",
            "official_result_evidence_db_execute",
            False,
        ),
        "autonomous_official_result_evidence_db_write_performed": value(
            "autonomous_official_result_evidence_db_write_performed",
            "official_result_evidence_db_write_performed",
            False,
        ),
        "autonomous_official_result_evidence_valid_race_rows": value(
            "autonomous_official_result_evidence_valid_race_rows",
            "official_result_evidence_valid_race_rows",
            0,
        ),
        "autonomous_official_result_evidence_valid_runner_rows": value(
            "autonomous_official_result_evidence_valid_runner_rows",
            "official_result_evidence_valid_runner_rows",
            0,
        ),
        "autonomous_official_result_evidence_blocked_race_rows": value(
            "autonomous_official_result_evidence_blocked_race_rows",
            "official_result_evidence_blocked_race_rows",
            0,
        ),
        "autonomous_official_result_evidence_blocked_runner_rows": value(
            "autonomous_official_result_evidence_blocked_runner_rows",
            "official_result_evidence_blocked_runner_rows",
            0,
        ),
        "autonomous_official_result_evidence_inserted_race_rows": value(
            "autonomous_official_result_evidence_inserted_race_rows",
            "official_result_evidence_inserted_race_rows",
            0,
        ),
        "autonomous_official_result_evidence_inserted_runner_rows": value(
            "autonomous_official_result_evidence_inserted_runner_rows",
            "official_result_evidence_inserted_runner_rows",
            0,
        ),
        "autonomous_official_result_evidence_blocker_reason_counts": (
            value(
                "autonomous_official_result_evidence_blocker_reason_counts",
                "official_result_evidence_blocker_reason_counts",
                {},
            )
            or {}
        ),
    }


def autopilot_cycle_verification_lines(daily_status: Mapping[str, Any]) -> list[str]:
    fields = autopilot_cycle_operational_fields(daily_status)
    return [
        f"{key}={value}"
        for key, value in fields.items()
    ]


def rejoin_promotion_distance_verification_lines(
    rejoin_promotion_distance_status: Mapping[str, Any],
) -> list[str]:
    status = rejoin_promotion_distance_status or {}
    fields = {
        "rejoin_promotion_distance_status": status.get("status"),
        "rejoin_promotion_distance_promotion_ready": status.get("promotion_ready"),
        "rejoin_promotion_distance_blockers": status.get("blockers") or [],
        "rejoin_promotion_distance_official_result_coverage_requested_race_count": (
            status.get("official_result_coverage_requested_race_count")
        ),
        "rejoin_promotion_distance_official_result_coverage_requested_race_count_source": (
            status.get("official_result_coverage_requested_race_count_source")
        ),
        "rejoin_promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids": (
            status.get(
                "official_result_coverage_legacy_requested_race_count_without_ids"
            )
        ),
        "rejoin_promotion_distance_official_result_coverage_races_with_rows_count": (
            status.get("official_result_coverage_races_with_rows_count")
        ),
        "rejoin_promotion_distance_official_result_coverage_missing_race_count": (
            status.get("official_result_coverage_missing_race_count")
        ),
        "rejoin_promotion_distance_official_result_coverage_missing_exclusion_count": (
            status.get("official_result_coverage_missing_exclusion_count")
        ),
        "rejoin_promotion_distance_official_result_runner_path_count": (
            status.get("official_result_runner_path_count")
        ),
        "rejoin_promotion_distance_official_result_runner_paths_source_field": (
            status.get("official_result_runner_paths_source_field")
        ),
    }
    return [f"{key}={value}" for key, value in fields.items()]


def canonical_join_eligibility_packet_key(packet_paths: Any) -> tuple[str, ...]:
    canonical_paths: set[str] = set()
    for path_value in packet_paths or []:
        if not str(path_value or "").strip():
            continue
        path = rooted_path(path_value)
        if path is None:
            continue
        canonical_paths.add(str(path.resolve(strict=False)))
    return tuple(sorted(canonical_paths))


def join_eligibility_preview_score(report: Mapping[str, Any]) -> tuple[int, int, int, float]:
    return (
        int_or_zero(
            report.get("join_eligibility_packet_accepted_races_present_in_shadow_run")
        ),
        int_or_zero(report.get("unified_evidence_eligible_rows")),
        int_or_zero(report.get("row_count")),
        float(report.get("report_mtime") or 0.0),
    )


def deduped_join_eligibility_preview_reports(
    reports: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_packet_key: dict[tuple[str, ...], tuple[tuple[int, int, int, float], dict[str, Any]]] = {}
    unkeyed: list[dict[str, Any]] = []
    for report in reports:
        packet_key = canonical_join_eligibility_packet_key(
            report.get("join_eligibility_packet_paths")
        )
        report_copy = dict(report)
        if not packet_key:
            unkeyed.append(report_copy)
            continue
        score = join_eligibility_preview_score(report_copy)
        existing = by_packet_key.get(packet_key)
        if existing is None or score > existing[0]:
            by_packet_key[packet_key] = (score, report_copy)
    keyed = [report for _, report in by_packet_key.values()]
    keyed.sort(key=lambda report: str(report.get("output_dir") or report.get("report_path") or ""))
    return keyed + unkeyed


def artifact_odds_rejection_reason_counts_for_report(
    report: Mapping[str, Any],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for audit in report.get("artifact_odds_audits") or []:
        if not isinstance(audit, Mapping):
            continue
        for reason, count in (audit.get("rejection_reason_counts") or {}).items():
            counts[str(reason)] += int_or_zero(count)
    return counts


def build_rejoin_unified_evidence_status(
    *,
    generated_at: datetime,
    reports: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]] = (),
    skipped: Sequence[Mapping[str, Any]] = (),
    join_eligibility_preview_reports: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    join_eligibility_preview_reports = deduped_join_eligibility_preview_reports(
        join_eligibility_preview_reports
    )
    artifact_odds_rejection_reason_counts = Counter()
    rejected_live_odds_candidate_reason_counts = Counter()
    skip_reason_counts = Counter(
        str(item.get("reason") or "unknown_skip_reason")
        for item in skipped
        if isinstance(item, Mapping)
    )
    failure_reason_counts = Counter(
        str(item.get("reason") or "unknown_failure_reason")
        for item in failures
        if isinstance(item, Mapping)
    )
    evaluated_dataset_candidate_count = len(reports) + len(failures) + len(skipped)
    for report in reports:
        artifact_odds_rejection_reason_counts.update(
            artifact_odds_rejection_reason_counts_for_report(report)
        )
        rejected_live_odds_candidate_reason_counts.update(
            {
                str(reason): int_or_zero(count)
                for reason, count in (
                    report.get("rejected_live_odds_candidate_reason_counts") or {}
                ).items()
            }
        )
    if reports and failures:
        status = "REJOIN_UNIFIED_EVIDENCE_DATASETS_PARTIAL"
    elif reports:
        status = "REJOIN_UNIFIED_EVIDENCE_DATASETS_BUILT"
    elif failures:
        status = "REJOIN_UNIFIED_EVIDENCE_DATASETS_FAILED"
    else:
        status = "REJOIN_UNIFIED_EVIDENCE_DATASETS_SKIPPED"
    if reports:
        status_reason = "rejoin_unified_evidence_datasets_built"
    elif failures:
        status_reason = "all_rejoin_unified_evidence_dataset_attempts_failed"
    elif skipped:
        status_reason = "all_rejoin_unified_evidence_dataset_candidates_skipped"
    else:
        status_reason = "no_rejoin_unified_evidence_dataset_candidates"
    return {
        "schema_version": "shadow_daemon_rejoin_unified_evidence_status_v1",
        "generated_at": generated_at.isoformat(),
        "status": status,
        "status_reason": status_reason,
        "evaluated_dataset_candidate_count": evaluated_dataset_candidate_count,
        "attempted_dataset_count": len(reports) + len(failures),
        "dataset_count": len(reports),
        "failed_dataset_count": len(failures),
        "skipped_dataset_count": len(skipped),
        "skip_reason_counts": dict(sorted(skip_reason_counts.items())),
        "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
        "skipped_safe_joined_race_count": sum(
            int_or_zero(item.get("safe_joined_race_count"))
            for item in skipped
            if isinstance(item, Mapping)
        ),
        "failed_safe_joined_race_count": sum(
            int_or_zero(item.get("safe_joined_race_count"))
            for item in failures
            if isinstance(item, Mapping)
        ),
        "row_count": sum(int_or_zero(report.get("row_count")) for report in reports),
        "race_count": sum(int_or_zero(report.get("race_count")) for report in reports),
        "rows_with_official_results": sum(
            int_or_zero(report.get("rows_with_official_results")) for report in reports
        ),
        "rows_with_strict_prejump_odds": sum(
            int_or_zero(report.get("rows_with_strict_prejump_odds")) for report in reports
        ),
        "rows_with_artifact_shadow_odds": sum(
            int_or_zero(report.get("rows_with_artifact_shadow_odds"))
            for report in reports
        ),
        "rows_with_artifact_shadow_odds_candidates": sum(
            int_or_zero(report.get("rows_with_artifact_shadow_odds_candidates"))
            for report in reports
        ),
        "artifact_shadow_odds_candidate_count": sum(
            int_or_zero(report.get("artifact_shadow_odds_candidate_count"))
            for report in reports
        ),
        "artifact_shadow_odds_selected_bucket_count": sum(
            int_or_zero(report.get("artifact_shadow_odds_selected_bucket_count"))
            for report in reports
        ),
        "artifact_odds_rows_seen": sum(
            int_or_zero(report.get("artifact_odds_rows_seen")) for report in reports
        ),
        "artifact_odds_rows_accepted": sum(
            int_or_zero(report.get("artifact_odds_rows_accepted")) for report in reports
        ),
        "artifact_odds_rows_rejected": sum(
            int_or_zero(report.get("artifact_odds_rows_rejected")) for report in reports
        ),
        "artifact_odds_rejection_reason_counts": dict(
            sorted(artifact_odds_rejection_reason_counts.items())
        ),
        "rejected_live_odds_candidate_count": sum(
            int_or_zero(report.get("rejected_live_odds_candidate_count"))
            for report in reports
        ),
        "rows_with_rejected_live_odds_candidates": sum(
            int_or_zero(report.get("rows_with_rejected_live_odds_candidates"))
            for report in reports
        ),
        "rejected_live_odds_candidate_reason_counts": dict(
            sorted(rejected_live_odds_candidate_reason_counts.items())
        ),
        "unified_evidence_eligible_rows": sum(
            int_or_zero(report.get("unified_evidence_eligible_rows")) for report in reports
        ),
        "join_eligibility_preview_dataset_count": len(join_eligibility_preview_reports),
        "join_eligibility_preview_row_count": sum(
            int_or_zero(report.get("row_count"))
            for report in join_eligibility_preview_reports
        ),
        "join_eligibility_preview_race_count": sum(
            int_or_zero(report.get("race_count"))
            for report in join_eligibility_preview_reports
        ),
        "join_eligibility_preview_unified_eligible_rows": sum(
            int_or_zero(report.get("unified_evidence_eligible_rows"))
            for report in join_eligibility_preview_reports
        ),
        "join_eligibility_preview_packet_accepted_races": sum(
            int_or_zero(report.get("join_eligibility_packet_accepted_races"))
            for report in join_eligibility_preview_reports
        ),
        "join_eligibility_preview_packet_present_races": sum(
            int_or_zero(
                report.get("join_eligibility_packet_accepted_races_present_in_shadow_run")
            )
            for report in join_eligibility_preview_reports
        ),
        "join_eligibility_preview_packet_rejected_races": sum(
            int_or_zero(report.get("join_eligibility_packet_rejected_races"))
            for report in join_eligibility_preview_reports
        ),
        "join_eligibility_preview_missing_race_ids": sorted(
            {
                str(race_id)
                for report in join_eligibility_preview_reports
                for race_id in (
                    report.get(
                        "join_eligibility_packet_accepted_race_ids_missing_from_shadow_run"
                    )
                    or []
                )
                if str(race_id or "").strip()
            }
        ),
        "reports": [
            {
                "output_dir": report.get("output_dir"),
                "shadow_run_dir": report.get("shadow_run_dir"),
                "join_dir": report.get("join_dir"),
                "joined_shadow_predictions_jsonl": report.get(
                    "joined_shadow_predictions_jsonl"
                ),
                "row_count": report.get("row_count"),
                "race_count": report.get("race_count"),
                "rows_with_official_results": report.get("rows_with_official_results"),
                "rows_with_strict_prejump_odds": report.get(
                    "rows_with_strict_prejump_odds"
                ),
                "rows_with_artifact_shadow_odds": report.get(
                    "rows_with_artifact_shadow_odds"
                ),
                "rows_with_artifact_shadow_odds_candidates": report.get(
                    "rows_with_artifact_shadow_odds_candidates"
                ),
                "artifact_shadow_odds_candidate_count": report.get(
                    "artifact_shadow_odds_candidate_count"
                ),
                "artifact_shadow_odds_selected_bucket_count": report.get(
                    "artifact_shadow_odds_selected_bucket_count"
                ),
                "artifact_odds_rows_seen": report.get("artifact_odds_rows_seen"),
                "artifact_odds_rows_accepted": report.get(
                    "artifact_odds_rows_accepted"
                ),
                "artifact_odds_rows_rejected": report.get(
                    "artifact_odds_rows_rejected"
                ),
                "artifact_odds_rejection_reason_counts": dict(
                    sorted(
                        artifact_odds_rejection_reason_counts_for_report(report).items()
                    )
                ),
                "rejected_live_odds_candidate_count": report.get(
                    "rejected_live_odds_candidate_count"
                ),
                "rows_with_rejected_live_odds_candidates": report.get(
                    "rows_with_rejected_live_odds_candidates"
                ),
                "rejected_live_odds_candidate_reason_counts": dict(
                    sorted(
                        (
                            report.get("rejected_live_odds_candidate_reason_counts")
                            or {}
                        ).items()
                    )
                ),
                "unified_evidence_eligible_rows": report.get(
                    "unified_evidence_eligible_rows"
                ),
                "join_eligibility_packet_accepted_races": report.get(
                    "join_eligibility_packet_accepted_races"
                ),
                "join_eligibility_packet_accepted_races_present_in_shadow_run": (
                    report.get(
                        "join_eligibility_packet_accepted_races_present_in_shadow_run"
                    )
                ),
                "join_eligibility_packet_rejected_races": report.get(
                    "join_eligibility_packet_rejected_races"
                ),
            }
            for report in reports
        ],
        "join_eligibility_preview_reports": [
            {
                "output_dir": report.get("output_dir"),
                "shadow_run_dir": report.get("shadow_run_dir"),
                "row_count": report.get("row_count"),
                "race_count": report.get("race_count"),
                "unified_evidence_eligible_rows": report.get(
                    "unified_evidence_eligible_rows"
                ),
                "join_eligibility_packet_paths": report.get(
                    "join_eligibility_packet_paths"
                ),
                "join_eligibility_packet_accepted_races": report.get(
                    "join_eligibility_packet_accepted_races"
                ),
                "join_eligibility_packet_accepted_races_present_in_shadow_run": (
                    report.get(
                        "join_eligibility_packet_accepted_races_present_in_shadow_run"
                    )
                ),
                "join_eligibility_packet_accepted_race_ids_missing_from_shadow_run": (
                    report.get(
                        "join_eligibility_packet_accepted_race_ids_missing_from_shadow_run"
                    )
                    or []
                ),
                "join_eligibility_packet_rejected_races": report.get(
                    "join_eligibility_packet_rejected_races"
                ),
            }
            for report in join_eligibility_preview_reports
        ],
        "failures": list(failures),
        "skipped": list(skipped),
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def discovered_join_eligibility_preview_reports(
    evidence_root: Path,
    *,
    max_reports: int = 8,
) -> list[dict[str, Any]]:
    by_packet_key: dict[tuple[str, ...], tuple[tuple[int, int, float], dict[str, Any]]] = {}
    for report_path in evidence_root.glob(
        "unified_evidence_dataset_*/unified_evidence_dataset_report.json"
    ):
        report = load_json(report_path) or {}
        packet_paths = canonical_join_eligibility_packet_key(
            report.get("join_eligibility_packet_paths")
        )
        if not packet_paths:
            continue
        report = dict(report)
        report.setdefault("output_dir", relpath(report_path.parent))
        report["report_path"] = relpath(report_path)
        report["report_mtime"] = report_path.stat().st_mtime
        score = (
            int_or_zero(
                report.get("join_eligibility_packet_accepted_races_present_in_shadow_run")
            ),
            int_or_zero(report.get("unified_evidence_eligible_rows")),
            report["report_mtime"],
        )
        existing = by_packet_key.get(packet_paths)
        if existing is None or score > existing[0]:
            by_packet_key[packet_paths] = (score, report)
    candidates = [
        (score[2], report)
        for score, report in by_packet_key.values()
    ]
    candidates.sort(key=lambda item: item[0])
    selected = candidates[-max_reports:] if max_reports > 0 else candidates
    return [report for _, report in selected]


def converted_joined_shadow_prediction_paths(evidence_root: Path) -> set[str]:
    converted: set[str] = set()
    for report_path in evidence_root.glob(
        "unified_evidence_dataset_*/unified_evidence_dataset_report.json"
    ):
        if not autopilot.is_automatic_unified_evidence_report_path(report_path):
            continue
        report = load_json(report_path) or {}
        for path_value in report.get("joined_shadow_prediction_paths") or []:
            path = rooted_path(path_value)
            if path is not None:
                converted.add(str(path.resolve()))
    return converted


def historical_safe_rejoin_results(
    *,
    evidence_root: Path,
    current_results: Sequence[Mapping[str, Any]],
    converted_paths: set[str],
    max_results: int = 32,
) -> list[dict[str, Any]]:
    current_join_dirs = {
        str(path.resolve())
        for path in (rooted_path(result.get("join_dir")) for result in current_results)
        if path is not None
    }
    candidates: list[dict[str, Any]] = []
    for join_dir in sorted(evidence_root.glob("forward_shadow_result_join_*_daemon_rejoin_*")):
        if str(join_dir.resolve()) in current_join_dirs:
            continue
        metrics = load_json(join_dir / "shadow_forward_metrics.json") or {}
        if int_or_zero(metrics.get("safe_joined_race_count")) <= 0:
            continue
        joined_path = join_dir / "joined_shadow_predictions.jsonl"
        if not joined_path.exists():
            continue
        if str(joined_path.resolve()) in converted_paths:
            continue
        shadow_run_dir = rooted_path(metrics.get("source_shadow_run"))
        if shadow_run_dir is None:
            continue
        candidates.append(
            {
                "candidate": {
                    "shadow_run_dir": relpath(shadow_run_dir),
                    "latest_join_dir": relpath(join_dir),
                    "historical_rejoin_artifact": True,
                },
                "join_dir": relpath(join_dir),
                "metrics": metrics,
                "historical_rejoin_artifact": True,
            }
        )
    return candidates[-max_results:] if max_results > 0 else candidates


def build_rejoin_unified_evidence_datasets(
    *,
    run_id: str,
    output_dir: Path,
    evidence_root: Path,
    db_path: Path,
    automated_join_report: Mapping[str, Any],
    generated_at: datetime,
    timeout_seconds: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[Path]]:
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    report_paths: list[Path] = []
    raw_results = [
        result
        for result in automated_join_report.get("results") or []
        if isinstance(result, Mapping)
    ]
    converted_paths = converted_joined_shadow_prediction_paths(evidence_root)
    raw_results.extend(
        historical_safe_rejoin_results(
            evidence_root=evidence_root,
            current_results=raw_results,
            converted_paths=converted_paths,
        )
    )
    for index, result in enumerate(raw_results, start=1):
        if not isinstance(result, Mapping):
            continue
        metrics = result.get("metrics") if isinstance(result.get("metrics"), Mapping) else {}
        safe_joined_count = int_or_zero(metrics.get("safe_joined_race_count"))
        join_dir = rooted_path(result.get("join_dir"))
        candidate = result.get("candidate") if isinstance(result.get("candidate"), Mapping) else {}
        shadow_run_dir = rooted_path(
            candidate.get("shadow_run_dir") or metrics.get("source_shadow_run")
        )
        joined_path = join_dir / "joined_shadow_predictions.jsonl" if join_dir else None
        if safe_joined_count <= 0:
            skipped.append(
                {
                    "index": index,
                    "reason": "safe_joined_race_count_zero",
                    "join_dir": relpath(join_dir),
                    "shadow_run_dir": relpath(shadow_run_dir),
                }
            )
            continue
        if shadow_run_dir is None or not (shadow_run_dir / "shadow_predictions.jsonl").exists():
            failures.append(
                {
                    "index": index,
                    "reason": "shadow_predictions_missing",
                    "join_dir": relpath(join_dir),
                    "shadow_run_dir": relpath(shadow_run_dir),
                    "safe_joined_race_count": safe_joined_count,
                }
            )
            continue
        if joined_path is None or not joined_path.exists():
            failures.append(
                {
                    "index": index,
                    "reason": "joined_shadow_predictions_jsonl_missing",
                    "join_dir": relpath(join_dir),
                    "shadow_run_dir": relpath(shadow_run_dir),
                    "safe_joined_race_count": safe_joined_count,
                }
            )
            continue
        joined_path_key = str(joined_path.resolve())
        if joined_path_key in converted_paths:
            skipped.append(
                {
                    "index": index,
                    "reason": "joined_shadow_predictions_already_converted",
                    "join_dir": relpath(join_dir),
                    "shadow_run_dir": relpath(shadow_run_dir),
                    "joined_shadow_predictions_jsonl": relpath(joined_path),
                    "safe_joined_race_count": safe_joined_count,
                }
            )
            continue
        dataset_dir = evidence_root / f"unified_evidence_dataset_{run_id}_daemon_rejoin_{index:03d}"
        command = autopilot.unified_evidence_dataset_command(
            shadow_run_dir=shadow_run_dir,
            output_dir=dataset_dir,
            db_path=db_path,
            odds_jsonl_paths=autopilot.shadow_odds_snapshot_paths_for_daily_dir(
                evidence_root=evidence_root,
                daily_dir=shadow_run_dir,
            ),
            joined_shadow_prediction_paths=[joined_path],
        )
        step = run_command(
            name=f"rejoin_unified_evidence_dataset_{index:03d}",
            command=command,
            output_dir=output_dir,
            timeout_seconds=timeout_seconds,
        )
        steps.append(step)
        report_path = dataset_dir / "unified_evidence_dataset_report.json"
        report = load_json(report_path) or {}
        if step.get("returncode") == 0 and report:
            report = dict(report)
            report.setdefault("output_dir", relpath(dataset_dir))
            report.setdefault("shadow_run_dir", relpath(shadow_run_dir))
            report["join_dir"] = relpath(join_dir)
            report["joined_shadow_predictions_jsonl"] = relpath(joined_path)
            report["safe_joined_race_count"] = safe_joined_count
            reports.append(report)
            report_paths.append(report_path)
            converted_paths.add(joined_path_key)
        else:
            failures.append(
                {
                    "index": index,
                    "reason": "unified_evidence_dataset_failed",
                    "output_dir": relpath(dataset_dir),
                    "shadow_run_dir": relpath(shadow_run_dir),
                    "join_dir": relpath(join_dir),
                    "joined_shadow_predictions_jsonl": relpath(joined_path),
                    "safe_joined_race_count": safe_joined_count,
                    "returncode": step.get("returncode"),
                }
            )
    status = build_rejoin_unified_evidence_status(
        generated_at=generated_at,
        reports=reports,
        failures=failures,
        skipped=skipped,
        join_eligibility_preview_reports=discovered_join_eligibility_preview_reports(
            evidence_root
        ),
    )
    return status, steps, report_paths


def metric_value(payload: Mapping[str, Any] | None, key: str) -> float | None:
    if not payload:
        return None
    value = payload.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def alert_rules(target_joined_races: int) -> dict[str, Any]:
    return {
        "schema_version": "shadow_autopilot_alert_rules_v1",
        "safe_joined_increase": {"enabled": True, "comparison": "current > previous"},
        "safe_joined_target_reached": {"enabled": True, "threshold": target_joined_races},
        "top1_material_drop": {"enabled": True, "delta_threshold": -0.05},
        "calibration_deterioration": {"enabled": True, "slope_distance_delta_threshold": 0.2},
        "box1_share_spike": {"enabled": True, "absolute_threshold": 0.35, "delta_threshold": 0.1},
        "join_failures_increase": {"enabled": True, "metric": "unsafe_matches"},
        "runner_set_mismatch_spike": {"enabled": True, "unsafe_reason_contains": "runner"},
        "model_hash_changed": {"enabled": True, "comparison": "current != previous"},
        "score_command_changed": {"enabled": True, "comparison": "current != previous"},
        "training_enabled": {"enabled": True, "expected": False},
        "tgr_enabled": {"enabled": True, "expected": False},
        "quarantined_feature_active": {"enabled": True, "expected": []},
        "probability_sum_failed": {"enabled": True, "expected": "PASS"},
        "no_predictions_with_eligible_inputs": {"enabled": True, "eligible_count_threshold": 1},
    }


def _as_text_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [str(item) for item in value]
    return [str(value)]


def unsafe_match_runner_related(row: Mapping[str, Any]) -> bool:
    reasons = {reason.strip() for reason in _as_text_list(row.get("reason")) if reason.strip()}
    reasons.update(
        reason.strip() for reason in _as_text_list(row.get("identity_errors")) if reason.strip()
    )
    normalized = {reason.lower() for reason in reasons}
    if normalized.intersection(RUNNER_RELATED_UNSAFE_REASONS):
        return True
    if any("runner" in reason for reason in normalized):
        return True
    if row.get("missing_predicted_boxes"):
        return True
    if row.get("disallowed_extra_official_boxes"):
        return True
    if row.get("name_mismatches"):
        return True
    return False


def unsafe_match_alert_sample(row: Mapping[str, Any], *, join_dir: object) -> dict[str, Any]:
    name_mismatches = list(row.get("name_mismatches") or [])
    sample = {
        "race_id": row.get("race_id"),
        "status": row.get("status"),
        "join_dir": relpath(Path(str(join_dir))) if join_dir else None,
        "reason": _as_text_list(row.get("reason") or row.get("identity_errors")),
        "missing_predicted_boxes": row.get("missing_predicted_boxes") or [],
        "disallowed_extra_official_boxes": row.get("disallowed_extra_official_boxes") or [],
        "allowed_extra_scratched_official_boxes": (
            row.get("allowed_extra_scratched_official_boxes") or []
        ),
        "name_mismatch_count": len(name_mismatches),
        "name_mismatch_samples": name_mismatches[:3],
    }
    alignment = row.get("prejump_runner_alignment")
    if isinstance(alignment, Mapping):
        sample["prejump_runner_alignment"] = {
            "canonical_runner_alignment_status": alignment.get(
                "canonical_runner_alignment_status"
            ),
            "canonical_runner_set_status": alignment.get("canonical_runner_set_status"),
            "canonical_runner_count": alignment.get("canonical_runner_count"),
            "canonical_prediction_runner_count": alignment.get(
                "canonical_prediction_runner_count"
            ),
        }
    return sample


def build_alert_report(
    *,
    current_dashboard: Mapping[str, Any],
    previous_dashboard: Mapping[str, Any] | None,
    automated_join_report: Mapping[str, Any],
    target_joined_races: int,
    current_observability: Mapping[str, Any] | None = None,
    previous_observability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    previous_dashboard = previous_dashboard or {}
    triggered: list[dict[str, Any]] = []
    current_safe = metric_value(current_dashboard, "safe_joined_races") or 0.0
    previous_safe = metric_value(previous_dashboard, "safe_joined_races") or 0.0
    if current_safe > previous_safe:
        triggered.append(
            {
                "rule": "safe_joined_increase",
                "severity": "info",
                "previous": previous_safe,
                "current": current_safe,
            }
        )
    if current_safe >= target_joined_races and previous_safe < target_joined_races:
        triggered.append(
            {
                "rule": "safe_joined_target_reached",
                "severity": "review",
                "threshold": target_joined_races,
                "current": current_safe,
            }
        )
    current_top1 = metric_value(current_dashboard, "top1")
    previous_top1 = metric_value(previous_dashboard, "top1")
    if current_top1 is not None and previous_top1 is not None and current_top1 - previous_top1 <= -0.05:
        triggered.append(
            {
                "rule": "top1_material_drop",
                "severity": "warning",
                "previous": previous_top1,
                "current": current_top1,
            }
        )
    current_slope = metric_value(current_dashboard.get("calibration") or {}, "slope")
    previous_slope = metric_value((previous_dashboard.get("calibration") or {}), "slope")
    if current_slope is not None and previous_slope is not None:
        current_distance = abs(1.0 - current_slope)
        previous_distance = abs(1.0 - previous_slope)
        if current_distance - previous_distance >= 0.2:
            triggered.append(
                {
                    "rule": "calibration_deterioration",
                    "severity": "warning",
                    "previous_slope": previous_slope,
                    "current_slope": current_slope,
                }
            )
    current_box1 = metric_value(current_dashboard, "box_1_share")
    previous_box1 = metric_value(previous_dashboard, "box_1_share")
    if (
        current_box1 is not None
        and current_box1 >= 0.35
        and (previous_box1 is None or current_box1 - previous_box1 >= 0.1)
    ):
        triggered.append(
            {
                "rule": "box1_share_spike",
                "severity": "warning",
                "previous": previous_box1,
                "current": current_box1,
            }
        )
    current_unsafe = metric_value(current_dashboard, "unsafe_matches") or 0.0
    previous_unsafe = metric_value(previous_dashboard, "unsafe_matches") or 0.0
    if current_unsafe > previous_unsafe:
        triggered.append(
            {
                "rule": "join_failures_increase",
                "severity": "warning",
                "previous": previous_unsafe,
                "current": current_unsafe,
            }
        )
    runner_mismatch_count = 0
    runner_mismatch_reason_counts: Counter[str] = Counter()
    runner_mismatch_samples: list[dict[str, Any]] = []
    for result in automated_join_report.get("results") or []:
        metrics = result.get("metrics") or {}
        if int(metrics.get("unsafe_match_count") or 0) <= 0:
            continue
        unsafe_path = Path(str(result.get("join_dir") or "")) / "unsafe_result_matches.json"
        if not unsafe_path.is_absolute():
            unsafe_path = ROOT / unsafe_path
        unsafe = load_json(unsafe_path) or {}
        for row in unsafe.get("unsafe_result_matches") or []:
            if unsafe_match_runner_related(row):
                runner_mismatch_count += 1
                for reason in _as_text_list(row.get("reason") or row.get("identity_errors")):
                    runner_mismatch_reason_counts[reason] += 1
                if len(runner_mismatch_samples) < 5:
                    runner_mismatch_samples.append(
                        unsafe_match_alert_sample(row, join_dir=result.get("join_dir"))
                    )
    if runner_mismatch_count:
        triggered.append(
            {
                "rule": "runner_set_mismatch_spike",
                "severity": "warning",
                "runner_related_unsafe_match_count": runner_mismatch_count,
                "reason_counts": dict(sorted(runner_mismatch_reason_counts.items())),
                "samples": runner_mismatch_samples,
            }
        )
    current_observability = current_observability or {}
    previous_observability = previous_observability or {}
    current_model_hash = current_observability.get("model_sha256")
    previous_model_hash = previous_observability.get("model_sha256")
    if current_model_hash and previous_model_hash and current_model_hash != previous_model_hash:
        triggered.append(
            {
                "rule": "model_hash_changed",
                "severity": "warning",
                "previous": previous_model_hash,
                "current": current_model_hash,
            }
        )
    current_command = current_observability.get("score_command_text")
    previous_command = previous_observability.get("score_command_text")
    if current_command and previous_command and current_command != previous_command:
        triggered.append(
            {
                "rule": "score_command_changed",
                "severity": "warning",
                "previous": previous_command,
                "current": current_command,
            }
        )
    safety_flags = current_observability.get("safety_flags") or {}
    if safety_flags.get("score_command_trains") or safety_flags.get("training_disabled") is False:
        triggered.append({"rule": "training_enabled", "severity": "critical"})
    if safety_flags.get("tgr_disabled") is False:
        triggered.append({"rule": "tgr_enabled", "severity": "critical"})
    quarantined_active = safety_flags.get("quarantined_features_active") or []
    if quarantined_active:
        triggered.append(
            {
                "rule": "quarantined_feature_active",
                "severity": "critical",
                "features": quarantined_active,
            }
        )
    probability_status = current_observability.get("probability_sum_status")
    if probability_status and probability_status != "PASS":
        triggered.append(
            {
                "rule": "probability_sum_failed",
                "severity": "critical",
                "status": probability_status,
            }
        )
    no_prediction_details = current_observability.get("no_prediction_details") or {}
    input_summary = no_prediction_details.get("input_summary") or {}
    if (
        current_observability.get("status") == "NO_PREDICTIONS"
        and int(input_summary.get("eligible_count") or 0) > 0
    ):
        triggered.append(
            {
                "rule": "no_predictions_with_eligible_inputs",
                "severity": "warning",
                "eligible_count": int(input_summary.get("eligible_count") or 0),
            }
        )
    return {
        "schema_version": "shadow_autopilot_alert_report_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "ALERTS_TRIGGERED" if triggered else "NO_ALERTS_TRIGGERED",
        "triggered_alerts": triggered,
        "comparison": {
            "previous_dashboard_present": bool(previous_dashboard),
            "previous_safe_joined_races": previous_safe,
            "current_safe_joined_races": current_safe,
        },
    }


def daemon_readiness(
    *,
    generated_at: datetime,
    dashboard: Mapping[str, Any],
    target_joined_races: int,
) -> dict[str, Any]:
    current_joined = int(dashboard.get("safe_joined_races") or 0)
    pending = int(dashboard.get("pending_races") or 0)
    unsafe = int(dashboard.get("unsafe_matches") or 0)
    calibration = dashboard.get("calibration") or {}
    probability_status = (dashboard.get("probability_sum_status") or {}).get("status")
    blockers: list[str] = []
    if current_joined < target_joined_races:
        blockers.append("insufficient_forward_shadow_joined_races")
    if pending:
        blockers.append("pending_official_results_remain")
    if unsafe:
        blockers.append("unsafe_result_matches_present")
    if probability_status != "PASS":
        blockers.append("probability_sum_status_not_pass")
    if dashboard.get("quarantined_features"):
        blockers.append("same_distance_same_grade_features_remain_quarantined")
    feature_activation_gate = dashboard.get("feature_activation_gate") or {}
    feature_activation_status = feature_activation_gate.get("status")
    if feature_activation_status and feature_activation_status != "FEATURE_ACTIVATION_PASS":
        blockers.append("feature_activation_gate_not_passed")

    calibration_status = str(calibration.get("status") or "not_computed")
    if current_joined < target_joined_races:
        decision = "NEED_MORE_RESULTS"
    elif calibration_status != "computed" or probability_status != "PASS":
        decision = "NEED_CALIBRATION_REVIEW"
    elif blockers:
        decision = "READY_FOR_RELIABILITY_REVIEW"
    else:
        decision = "READY_FOR_PROMOTION_CANDIDATE_REVIEW"
    return {
        "schema_version": "shadow_daemon_promotion_readiness_tracker_v1",
        "generated_at": generated_at.isoformat(),
        "current_joined_race_count": current_joined,
        "target_joined_race_count": target_joined_races,
        "calibration_status": calibration_status,
        "reliability_status": "target_met" if current_joined >= target_joined_races else "needs_more_results",
        "box_bias_status": {
            "box_1_share": dashboard.get("box_1_share"),
            "status": "REVIEW" if (dashboard.get("box_1_share") or 0) >= 0.35 else "TRACKING",
        },
        "feature_activation_gate_status": feature_activation_status,
        "kept_quarantined_features": feature_activation_gate.get("kept_quarantined_features") or [],
        "activation_allowed_features": feature_activation_gate.get("activation_allowed_features") or [],
        "leakage_status": "NO_NEW_LEAKAGE_SIGNAL_FROM_DAEMON",
        "outstanding_blockers": blockers,
        "decision": decision,
        "promotion_allowed": False,
    }


def readiness_markdown(readiness: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Shadow Daemon Readiness",
            "",
            f"- Decision: `{readiness.get('decision')}`",
            f"- Current joined races: `{readiness.get('current_joined_race_count')}`",
            f"- Target joined races: `{readiness.get('target_joined_race_count')}`",
            f"- Calibration status: `{readiness.get('calibration_status')}`",
            f"- Reliability status: `{readiness.get('reliability_status')}`",
            f"- Box-bias status: `{readiness.get('box_bias_status')}`",
            f"- Feature activation gate: `{readiness.get('feature_activation_gate_status')}`",
            f"- Kept quarantined features: `{readiness.get('kept_quarantined_features')}`",
            f"- Leakage status: `{readiness.get('leakage_status')}`",
            f"- Outstanding blockers: `{readiness.get('outstanding_blockers')}`",
            "",
            "Promotion remains forbidden by this daemon.",
            "",
        ]
    )


def copy_if_exists(source: Path | None, destination: Path) -> bool:
    if source is None or not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return True


def feature_activation_gate_status_from_autopilot(
    autopilot_output_dir: Path | None,
) -> dict[str, Any] | None:
    if autopilot_output_dir is None:
        return None
    status_path = autopilot_output_dir / "feature_activation_gate_status.json"
    status = load_json(status_path)
    if not status:
        return None
    return {
        "schema_version": "shadow_daemon_feature_activation_gate_summary_v1",
        "status": status.get("status"),
        "status_path": relpath(status_path),
        "output_dir": status.get("output_dir"),
        "provenance_audit": status.get("provenance_audit"),
        "activation_allowed_features": status.get("activation_allowed_features") or [],
        "kept_quarantined_features": status.get("kept_quarantined_features") or [],
        "fail_reason_summary": status.get("fail_reason_summary") or {},
        "data_availability_status": status.get("data_availability_status") or {},
        "inputs": status.get("inputs") or {},
        "no_write_guarantees": status.get("no_write_guarantees") or {},
    }


def shadow_odds_snapshot_status_from_autopilot(
    autopilot_output_dir: Path | None,
) -> dict[str, Any]:
    if autopilot_output_dir is None:
        return {
            "schema_version": "shadow_daemon_odds_snapshot_summary_v1",
            "status": "MISSING_AUTOPILOT_OUTPUT",
            "status_path": None,
            "collection_attempted": None,
            "skipped_reason": "autopilot_output_missing",
            "prediction_rows": None,
            "odds_candidate_rows": None,
            "valid_pre_jump_dog_odds_rows": None,
            "races_with_complete_valid_prejump_odds": None,
            "races_with_missing_odds_rows": None,
            "races_with_post_feature_freeze_odds_rows": None,
            "odds_research_gate_status": None,
            "odds_research_gate_report_path": None,
            "odds_research_gate_complete_valid_prejump_odds_races": None,
            "odds_research_gate_minimum_complete_valid_prejump_odds_races": None,
            "odds_research_gate_source_url_coverage_pct": None,
            "odds_research_gate_source_url_rows_missing": None,
            "odds_research_gate_blocker_counts": {},
            "odds_research_next_action": None,
            "timing_aligned_prediction_rerun_required": False,
            "timing_aligned_prediction_rerun_race_count": 0,
            "timing_aligned_prediction_rerun_race_ids": [],
            "timing_aligned_prediction_rerun_races": [],
            "timing_aligned_prediction_rerun_reason_counts": {},
            "ev_output_rows": 0,
            "ev_calculation_status": "DISABLED_REPORT_ONLY_NO_EV_OUTPUT",
            "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        }
    status_path = autopilot_output_dir / "shadow_odds_snapshot_status.json"
    status = load_json(status_path)
    if not status:
        return {
            "schema_version": "shadow_daemon_odds_snapshot_summary_v1",
            "status": "MISSING_AUTOPILOT_ODDS_SNAPSHOT_STATUS",
            "status_path": relpath(status_path),
            "collection_attempted": None,
            "skipped_reason": "shadow_odds_snapshot_status_missing",
            "prediction_rows": None,
            "odds_candidate_rows": None,
            "valid_pre_jump_dog_odds_rows": None,
            "races_with_complete_valid_prejump_odds": None,
            "races_with_missing_odds_rows": None,
            "races_with_post_feature_freeze_odds_rows": None,
            "odds_research_gate_status": None,
            "odds_research_gate_report_path": None,
            "odds_research_gate_complete_valid_prejump_odds_races": None,
            "odds_research_gate_minimum_complete_valid_prejump_odds_races": None,
            "odds_research_gate_source_url_coverage_pct": None,
            "odds_research_gate_source_url_rows_missing": None,
            "odds_research_gate_blocker_counts": {},
            "odds_research_next_action": None,
            "timing_aligned_prediction_rerun_required": False,
            "timing_aligned_prediction_rerun_race_count": 0,
            "timing_aligned_prediction_rerun_race_ids": [],
            "timing_aligned_prediction_rerun_races": [],
            "timing_aligned_prediction_rerun_reason_counts": {},
            "ev_output_rows": 0,
            "ev_calculation_status": "DISABLED_REPORT_ONLY_NO_EV_OUTPUT",
            "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        }
    return {
        "schema_version": "shadow_daemon_odds_snapshot_summary_v1",
        "status": status.get("status") or status.get("final_status"),
        "final_status": status.get("final_status") or status.get("status"),
        "status_path": relpath(status_path),
        "output_dir": status.get("output_dir"),
        "collection_attempted": status.get("collection_attempted"),
        "skipped_reason": status.get("skipped_reason"),
        "prediction_rows": status.get("prediction_rows"),
        "race_count": status.get("race_count"),
        "runner_rows": status.get("runner_rows"),
        "odds_candidate_rows": status.get("odds_candidate_rows"),
        "valid_pre_jump_dog_odds_rows": status.get("valid_pre_jump_dog_odds_rows"),
        "races_with_complete_valid_prejump_odds": status.get(
            "races_with_complete_valid_prejump_odds"
        ),
        "races_with_missing_odds_rows": status.get("races_with_missing_odds_rows"),
        "races_with_post_feature_freeze_odds_rows": status.get(
            "races_with_post_feature_freeze_odds_rows"
        ),
        "odds_research_gate_status": status.get("odds_research_gate_status"),
        "odds_research_gate_report_path": status.get("odds_research_gate_report_path"),
        "odds_research_gate_complete_valid_prejump_odds_races": status.get(
            "odds_research_gate_complete_valid_prejump_odds_races"
        ),
        "odds_research_gate_minimum_complete_valid_prejump_odds_races": status.get(
            "odds_research_gate_minimum_complete_valid_prejump_odds_races"
        ),
        "odds_research_gate_source_url_coverage_pct": status.get(
            "odds_research_gate_source_url_coverage_pct"
        ),
        "odds_research_gate_source_url_rows_missing": status.get(
            "odds_research_gate_source_url_rows_missing"
        ),
        "odds_research_gate_blocker_counts": status.get(
            "odds_research_gate_blocker_counts"
        )
        or {},
        "odds_research_next_action": status.get("odds_research_next_action"),
        "timing_aligned_prediction_rerun_required": bool(
            status.get("timing_aligned_prediction_rerun_required")
        ),
        "timing_aligned_prediction_rerun_race_count": int(
            status.get("timing_aligned_prediction_rerun_race_count") or 0
        ),
        "timing_aligned_prediction_rerun_race_ids": list(
            status.get("timing_aligned_prediction_rerun_race_ids") or []
        ),
        "timing_aligned_prediction_rerun_races": list(
            status.get("timing_aligned_prediction_rerun_races") or []
        ),
        "timing_aligned_prediction_rerun_reason_counts": dict(
            status.get("timing_aligned_prediction_rerun_reason_counts") or {}
        ),
        "ev_eligible_rows": status.get("ev_eligible_rows"),
        "ev_output_rows": status.get("ev_output_rows", 0),
        "ev_calculation_status": status.get("ev_calculation_status")
        or "DISABLED_REPORT_ONLY_NO_EV_OUTPUT",
        "protected_paths_unchanged": status.get("protected_paths_unchanged"),
        "no_write_guarantees": status.get("no_write_guarantees") or dict(NO_WRITE_GUARANTEES),
    }


def live_odds_capture_packet_from_autopilot(
    autopilot_output_dir: Path | None,
) -> dict[str, Any]:
    if autopilot_output_dir is None:
        return {
            "schema_version": "shadow_daemon_live_odds_capture_packet_summary_v1",
            "status": "MISSING_AUTOPILOT_OUTPUT",
            "packet_path": None,
            "verified_prejump_race_count": None,
            "capture_window_offsets_minutes": [],
            "approval_required": True,
            "can_capture_live_odds_now": False,
            "hard_stops": ["autopilot_output_missing"],
            "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        }
    packet_path = autopilot_output_dir / "live_odds_capture_approval_packet.json"
    packet = load_json(packet_path)
    if not packet:
        return {
            "schema_version": "shadow_daemon_live_odds_capture_packet_summary_v1",
            "status": "MISSING_LIVE_ODDS_CAPTURE_APPROVAL_PACKET",
            "packet_path": relpath(packet_path),
            "verified_prejump_race_count": None,
            "capture_window_offsets_minutes": [],
            "approval_required": True,
            "can_capture_live_odds_now": False,
            "hard_stops": ["live_odds_capture_approval_packet_missing"],
            "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        }
    return {
        "schema_version": "shadow_daemon_live_odds_capture_packet_summary_v1",
        "status": packet.get("status"),
        "packet_path": relpath(packet_path),
        "verified_prejump_race_count": packet.get("verified_prejump_race_count"),
        "capture_window_offsets_minutes": packet.get("capture_window_offsets_minutes") or [],
        "approval_required": packet.get("approval_required"),
        "can_capture_live_odds_now": packet.get("can_capture_live_odds_now", False),
        "hard_stops": packet.get("hard_stops") or [],
        "write_scope": packet.get("write_scope"),
        "required_provenance_fields": packet.get("required_provenance_fields") or [],
        "planned_live_odds_capture_command": packet.get(
            "planned_live_odds_capture_command"
        )
        or [],
        "approved_live_odds_capture_command_template": packet.get(
            "approved_live_odds_capture_command_template"
        )
        or [],
        "no_write_guarantees": packet.get("no_write_guarantees") or dict(NO_WRITE_GUARANTEES),
    }


def autonomous_live_odds_capture_status_from_autopilot(
    autopilot_output_dir: Path | None,
) -> dict[str, Any]:
    if autopilot_output_dir is None:
        return {
            "schema_version": "shadow_daemon_autonomous_live_odds_capture_summary_v1",
            "status": "MISSING_AUTOPILOT_OUTPUT",
            "status_path": None,
            "attempted": False,
            "execute": False,
            "ready_count": 0,
            "validation_pass_count": 0,
            "inserted_live_odds_rows": 0,
            "append_only": True,
            "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        }
    status_path = autopilot_output_dir / "autonomous_live_odds_capture_status.json"
    status = load_json(status_path)
    if not status:
        return {
            "schema_version": "shadow_daemon_autonomous_live_odds_capture_summary_v1",
            "status": "MISSING_AUTONOMOUS_LIVE_ODDS_CAPTURE_STATUS",
            "status_path": relpath(status_path),
            "attempted": False,
            "execute": False,
            "ready_count": 0,
            "validation_pass_count": 0,
            "inserted_live_odds_rows": 0,
            "append_only": True,
            "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        }
    next_prejump_window = status.get("next_prejump_window")
    if not isinstance(next_prejump_window, Mapping):
        next_prejump_window = None
    return {
        "schema_version": "shadow_daemon_autonomous_live_odds_capture_summary_v1",
        "status": status.get("status") or status.get("final_status"),
        "run_id": status.get("run_id"),
        "final_status": status.get("final_status") or status.get("status"),
        "operator_status": status.get("operator_status"),
        "runtime_action": status.get("runtime_action"),
        "readiness_decision": status.get("readiness_decision"),
        "status_path": relpath(status_path),
        "output_dir": status.get("output_dir"),
        "attempted": bool(status.get("attempted")),
        "execute": bool(status.get("execute")),
        "allow_auto_scrape_odds": bool(status.get("allow_auto_scrape_odds")),
        "ready_count": int_or_zero(status.get("ready_count")),
        "validation_pass_count": int_or_zero(status.get("validation_pass_count")),
        "inserted_live_odds_rows": int_or_zero(status.get("inserted_live_odds_rows")),
        "append_only": True,
        "status_counts": status.get("status_counts") or {},
        "next_prejump_window": dict(next_prejump_window)
        if next_prejump_window
        else None,
        "next_window_opens_at": status.get("next_window_opens_at")
        or (
            next_prejump_window.get("next_window_opens_at")
            if next_prejump_window
            else None
        ),
        "recommended_rerun_after_local": status.get("recommended_rerun_after_local")
        or (
            next_prejump_window.get("recommended_rerun_after_local")
            if next_prejump_window
            else None
        ),
        "next_race_id": status.get("next_race_id")
        or (
            (next_prejump_window.get("next_race") or {}).get("race_id")
            if next_prejump_window
            and isinstance(next_prejump_window.get("next_race"), Mapping)
            else None
        ),
        **t2_odds_capture_status_fields(status),
        "no_write_guarantees": status.get("no_write_guarantees") or dict(NO_WRITE_GUARANTEES),
    }


def autonomous_official_result_capture_dir_from_autopilot(
    autopilot_output_dir: Path,
) -> Path | None:
    match = re.match(r"^shadow_autopilot_v1_(.+)$", autopilot_output_dir.name)
    if not match:
        return None
    run_id = match.group(1)
    return (
        autopilot_output_dir.parent
        / f"autonomous_official_result_capture_{run_id}_autopilot"
    )


def autonomous_official_result_capture_progress_status_from_autopilot(
    *,
    autopilot_output_dir: Path,
    status_path: Path,
) -> dict[str, Any] | None:
    capture_dir = autonomous_official_result_capture_dir_from_autopilot(
        autopilot_output_dir
    )
    if capture_dir is None:
        return None
    progress_path = capture_dir / "autonomous_official_result_capture_progress.json"
    progress = load_json(progress_path)
    if not progress:
        return None
    attempts_path = (
        capture_dir / "autonomous_official_result_capture_attempts.progress.jsonl"
    )
    active_candidate = progress.get("active_candidate")
    if not isinstance(active_candidate, Mapping):
        active_candidate = None
    return {
        "schema_version": "shadow_daemon_autonomous_official_result_capture_summary_v1",
        "status": "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_IN_PROGRESS",
        "status_path": relpath(status_path),
        "output_dir": relpath(capture_dir),
        "attempted": True,
        "candidate_count": int_or_zero(progress.get("candidate_count")),
        "ingested_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "official_result_race_rows": 0,
        "official_result_runner_rows": 0,
        "quarantine_rows": 0,
        "official_result_evidence_db_ingest_status": None,
        "official_result_evidence_db_execute": False,
        "official_result_evidence_db_write_performed": False,
        "official_result_evidence_valid_race_rows": 0,
        "official_result_evidence_valid_runner_rows": 0,
        "official_result_evidence_blocked_race_rows": 0,
        "official_result_evidence_blocked_runner_rows": 0,
        "official_result_evidence_inserted_race_rows": 0,
        "official_result_evidence_inserted_runner_rows": 0,
        "official_result_evidence_blocker_reason_counts": {},
        "live_odds_backlog_recovery_queue_path": None,
        "live_odds_backlog_recovery_queue_diagnostic_only": True,
        "live_odds_backlog_recovery_queue_join_acceptance_changed": False,
        "live_odds_backlog_recovery_queue_db_write_performed": False,
        "live_odds_backlog_awaiting_official_result_evidence_race_count": 0,
        "live_odds_backlog_awaiting_official_result_evidence_race_ids": [],
        "live_odds_backlog_awaiting_official_result_evidence_authorized_action": None,
        "live_odds_backlog_awaiting_official_result_recheck_ready_race_count": 0,
        "live_odds_backlog_runner_set_validation_path": None,
        "live_odds_backlog_runner_set_validation_retryable_race_count": 0,
        "live_odds_backlog_runner_set_validation_exact_match_race_count": 0,
        "live_odds_backlog_runner_set_validation_blocked_race_count": 0,
        "live_odds_backlog_runner_set_validation_diagnostic_only": True,
        "live_odds_backlog_runner_set_validation_join_authorized": False,
        "live_odds_backlog_runner_set_validation_db_write_performed": False,
        "live_odds_backlog_join_eligibility_packet_path": None,
        "live_odds_backlog_join_eligibility_evaluated_race_count": 0,
        "live_odds_backlog_join_eligibility_eligible_report_only_race_count": 0,
        "live_odds_backlog_join_eligibility_blocked_race_count": 0,
        "live_odds_backlog_join_eligibility_blocker_counts": {},
        "live_odds_backlog_join_eligibility_diagnostic_only": True,
        "live_odds_backlog_join_eligibility_join_authorized": False,
        "live_odds_backlog_join_eligibility_db_write_performed": False,
        "live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count": 0,
        "progress_path": relpath(progress_path),
        "progress_attempts_path": relpath(attempts_path),
        "progress_candidate_count": int_or_zero(progress.get("candidate_count")),
        "progress_completed_count": int_or_zero(progress.get("completed_count")),
        "progress_status_counts": dict(progress.get("status_counts") or {}),
        "progress_active_candidate": (
            dict(active_candidate) if active_candidate is not None else None
        ),
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def autonomous_official_result_capture_status_from_autopilot(
    autopilot_output_dir: Path | None,
) -> dict[str, Any]:
    if autopilot_output_dir is None:
        return {
            "schema_version": "shadow_daemon_autonomous_official_result_capture_summary_v1",
            "status": "MISSING_AUTOPILOT_OUTPUT",
            "status_path": None,
            "output_dir": None,
            "attempted": False,
            "candidate_count": 0,
            "official_result_race_rows": 0,
            "official_result_runner_rows": 0,
            "quarantine_rows": 0,
            "quarantined_race_ids": [],
            "quarantine_reason_counts": {},
            "quarantine_error_counts": {},
            "quarantine_result_boxes_not_in_participants_counts": {},
            "quarantine_runner_set_mismatch_samples": [],
            "skipped_reason_counts": {},
            "awaiting_jump_race_count": 0,
            "awaiting_jump_race_ids": [],
            "awaiting_jump_next_recheck_after_local": None,
            "awaiting_jump_races": [],
            "official_result_evidence_db_ingest_status": None,
            "official_result_evidence_db_execute": False,
            "official_result_evidence_db_write_performed": False,
            "official_result_evidence_valid_race_rows": 0,
            "official_result_evidence_valid_runner_rows": 0,
            "official_result_evidence_blocked_race_rows": 0,
            "official_result_evidence_blocked_runner_rows": 0,
            "official_result_evidence_inserted_race_rows": 0,
            "official_result_evidence_inserted_runner_rows": 0,
            "official_result_evidence_blocker_reason_counts": {},
            "live_odds_backlog_recovery_queue_path": None,
            "live_odds_backlog_recovery_queue_diagnostic_only": True,
            "live_odds_backlog_recovery_queue_join_acceptance_changed": False,
            "live_odds_backlog_recovery_queue_db_write_performed": False,
            "live_odds_backlog_awaiting_official_result_evidence_race_count": 0,
            "live_odds_backlog_awaiting_official_result_evidence_race_ids": [],
            "live_odds_backlog_awaiting_official_result_evidence_authorized_action": None,
            "live_odds_backlog_awaiting_official_result_recheck_ready_race_count": 0,
            "live_odds_backlog_runner_set_validation_path": None,
            "live_odds_backlog_runner_set_validation_retryable_race_count": 0,
            "live_odds_backlog_runner_set_validation_exact_match_race_count": 0,
            "live_odds_backlog_runner_set_validation_blocked_race_count": 0,
            "live_odds_backlog_runner_set_validation_diagnostic_only": True,
            "live_odds_backlog_runner_set_validation_join_authorized": False,
            "live_odds_backlog_runner_set_validation_db_write_performed": False,
            "live_odds_backlog_join_eligibility_packet_path": None,
            "live_odds_backlog_join_eligibility_evaluated_race_count": 0,
            "live_odds_backlog_join_eligibility_eligible_report_only_race_count": 0,
            "live_odds_backlog_join_eligibility_blocked_race_count": 0,
            "live_odds_backlog_join_eligibility_blocker_counts": {},
            "live_odds_backlog_join_eligibility_diagnostic_only": True,
            "live_odds_backlog_join_eligibility_join_authorized": False,
            "live_odds_backlog_join_eligibility_db_write_performed": False,
            "live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count": 0,
            "live_odds_backlog_awaiting_official_result_evidence_race_count": 0,
            "live_odds_backlog_awaiting_official_result_evidence_race_ids": [],
            "live_odds_backlog_awaiting_official_result_evidence_authorized_action": None,
            "live_odds_backlog_awaiting_official_result_recheck_ready_race_count": 0,
            "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        }
    status_path = autopilot_output_dir / "autonomous_official_result_capture_status.json"
    status = load_json(status_path)
    if not status:
        progress_status = autonomous_official_result_capture_progress_status_from_autopilot(
            autopilot_output_dir=autopilot_output_dir,
            status_path=status_path,
        )
        if progress_status is not None:
            return progress_status
        return {
            "schema_version": "shadow_daemon_autonomous_official_result_capture_summary_v1",
            "status": "MISSING_AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_STATUS",
            "status_path": relpath(status_path),
            "output_dir": None,
            "attempted": False,
            "candidate_count": 0,
            "official_result_race_rows": 0,
            "official_result_runner_rows": 0,
            "quarantine_rows": 0,
            "quarantined_race_ids": [],
            "quarantine_reason_counts": {},
            "quarantine_error_counts": {},
            "quarantine_result_boxes_not_in_participants_counts": {},
            "quarantine_runner_set_mismatch_samples": [],
            "skipped_reason_counts": {},
            "awaiting_jump_race_count": 0,
            "awaiting_jump_race_ids": [],
            "awaiting_jump_next_recheck_after_local": None,
            "awaiting_jump_races": [],
            "official_result_evidence_db_ingest_status": None,
            "official_result_evidence_db_execute": False,
            "official_result_evidence_db_write_performed": False,
            "official_result_evidence_valid_race_rows": 0,
            "official_result_evidence_valid_runner_rows": 0,
            "official_result_evidence_blocked_race_rows": 0,
            "official_result_evidence_blocked_runner_rows": 0,
            "official_result_evidence_inserted_race_rows": 0,
            "official_result_evidence_inserted_runner_rows": 0,
            "official_result_evidence_blocker_reason_counts": {},
            "live_odds_backlog_join_eligibility_packet_path": None,
            "live_odds_backlog_join_eligibility_evaluated_race_count": 0,
            "live_odds_backlog_join_eligibility_eligible_report_only_race_count": 0,
            "live_odds_backlog_join_eligibility_blocked_race_count": 0,
            "live_odds_backlog_join_eligibility_blocker_counts": {},
            "live_odds_backlog_join_eligibility_diagnostic_only": True,
            "live_odds_backlog_join_eligibility_join_authorized": False,
            "live_odds_backlog_join_eligibility_db_write_performed": False,
            "live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count": 0,
            "progress_path": None,
            "progress_attempts_path": None,
            "progress_candidate_count": 0,
            "progress_completed_count": 0,
            "progress_status_counts": {},
            "progress_active_candidate": None,
            "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        }
    recovery_queue_path = status.get("live_odds_backlog_recovery_queue_path")
    if not recovery_queue_path and status.get("output_dir"):
        recovery_queue_path = (
            f"{str(status.get('output_dir')).rstrip('/')}/"
            "live_odds_backlog_recovery_queue.json"
        )
    runner_set_validation_path = status.get(
        "live_odds_backlog_runner_set_validation_path"
    )
    if not runner_set_validation_path and status.get("output_dir"):
        runner_set_validation_path = (
            f"{str(status.get('output_dir')).rstrip('/')}/"
            "live_odds_backlog_runner_set_validation.json"
        )
    join_eligibility_packet_path = status.get(
        "live_odds_backlog_join_eligibility_packet_path"
    )
    if not join_eligibility_packet_path and status.get("output_dir"):
        join_eligibility_packet_path = (
            f"{str(status.get('output_dir')).rstrip('/')}/"
            "live_odds_backlog_join_eligibility_packet.json"
        )
    join_eligibility_blocker_counts = status.get(
        "live_odds_backlog_join_eligibility_blocker_counts"
    )
    if not isinstance(join_eligibility_blocker_counts, Mapping):
        join_eligibility_blocker_counts = {}
    awaiting_official_result_evidence_race_count = int_or_zero(
        status.get("live_odds_backlog_awaiting_official_result_evidence_race_count")
    )
    awaiting_official_result_evidence_race_ids = list(
        status.get("live_odds_backlog_awaiting_official_result_evidence_race_ids")
        or []
    )
    awaiting_official_result_evidence_authorized_action = status.get(
        "live_odds_backlog_awaiting_official_result_evidence_authorized_action"
    )
    awaiting_official_result_recheck_ready_race_count = int_or_zero(
        status.get(
            "live_odds_backlog_awaiting_official_result_recheck_ready_race_count"
        )
    )
    awaiting_official_result_recheck_ready_missing = (
        status.get(
            "live_odds_backlog_awaiting_official_result_recheck_ready_race_count"
        )
        is None
    )
    if (
        (
            (
                not awaiting_official_result_evidence_race_count
                and not awaiting_official_result_evidence_race_ids
            )
            or awaiting_official_result_recheck_ready_missing
            or not awaiting_official_result_evidence_authorized_action
        )
        and recovery_queue_path
    ):
        recovery_queue = load_json(rooted_path(recovery_queue_path))
        queues = recovery_queue.get("queues") if isinstance(recovery_queue, Mapping) else {}
        awaiting_bucket = (
            queues.get("awaiting_official_result_evidence")
            if isinstance(queues, Mapping)
            else {}
        )
        if isinstance(awaiting_bucket, Mapping):
            if not awaiting_official_result_evidence_race_count:
                awaiting_official_result_evidence_race_count = int_or_zero(
                    awaiting_bucket.get("race_count")
                )
            if not awaiting_official_result_evidence_race_ids:
                awaiting_official_result_evidence_race_ids = list(
                    awaiting_bucket.get("race_ids") or []
                )
            if not awaiting_official_result_evidence_authorized_action:
                awaiting_official_result_evidence_authorized_action = (
                    awaiting_bucket.get("authorized_action")
                )
            recheck_plan = awaiting_bucket.get("recheck_plan")
            if isinstance(recheck_plan, Mapping) and (
                awaiting_official_result_recheck_ready_missing
                or not awaiting_official_result_recheck_ready_race_count
            ):
                awaiting_official_result_recheck_ready_race_count = int_or_zero(
                    recheck_plan.get("recheck_ready_race_count")
                )
    return {
        "schema_version": "shadow_daemon_autonomous_official_result_capture_summary_v1",
        "status": status.get("status") or status.get("final_status"),
        "status_path": relpath(status_path),
        "output_dir": status.get("output_dir"),
        "attempted": bool(status.get("attempted")),
        "candidate_count": int_or_zero(status.get("candidate_count")),
        "ingested_count": int_or_zero(status.get("ingested_count")),
        "failed_count": int_or_zero(status.get("failed_count")),
        "skipped_count": int_or_zero(status.get("skipped_count")),
        "progress_path": status.get("progress_path"),
        "progress_attempts_path": status.get("progress_attempts_path"),
        "progress_candidate_count": int_or_zero(status.get("progress_candidate_count")),
        "progress_completed_count": int_or_zero(status.get("progress_completed_count")),
        "progress_status_counts": dict(status.get("progress_status_counts") or {}),
        "progress_active_candidate": (
            dict(status.get("progress_active_candidate"))
            if isinstance(status.get("progress_active_candidate"), Mapping)
            else None
        ),
        "official_result_race_rows": int_or_zero(status.get("official_result_race_rows")),
        "official_result_runner_rows": int_or_zero(status.get("official_result_runner_rows")),
        "quarantine_rows": int_or_zero(status.get("quarantine_rows")),
        "quarantined_race_ids": list(status.get("quarantined_race_ids") or []),
        "quarantine_reason_counts": dict(status.get("quarantine_reason_counts") or {}),
        "quarantine_error_counts": dict(status.get("quarantine_error_counts") or {}),
        "quarantine_result_boxes_not_in_participants_counts": dict(
            status.get("quarantine_result_boxes_not_in_participants_counts") or {}
        ),
        "quarantine_runner_set_mismatch_samples": list(
            status.get("quarantine_runner_set_mismatch_samples") or []
        ),
        "skipped_reason_counts": dict(status.get("skipped_reason_counts") or {}),
        "awaiting_jump_race_count": int_or_zero(status.get("awaiting_jump_race_count")),
        "awaiting_jump_race_ids": list(status.get("awaiting_jump_race_ids") or []),
        "awaiting_jump_next_recheck_after_local": status.get(
            "awaiting_jump_next_recheck_after_local"
        ),
        "awaiting_jump_races": list(status.get("awaiting_jump_races") or []),
        "official_result_evidence_db_ingest_status": status.get(
            "official_result_evidence_db_ingest_status"
        ),
        "official_result_evidence_db_execute": bool(
            status.get("official_result_evidence_db_execute")
        ),
        "official_result_evidence_db_write_performed": bool(
            status.get("official_result_evidence_db_write_performed")
        ),
        "official_result_evidence_valid_race_rows": int_or_zero(
            status.get("official_result_evidence_valid_race_rows")
        ),
        "official_result_evidence_valid_runner_rows": int_or_zero(
            status.get("official_result_evidence_valid_runner_rows")
        ),
        "official_result_evidence_blocked_race_rows": int_or_zero(
            status.get("official_result_evidence_blocked_race_rows")
        ),
        "official_result_evidence_blocked_runner_rows": int_or_zero(
            status.get("official_result_evidence_blocked_runner_rows")
        ),
        "official_result_evidence_inserted_race_rows": int_or_zero(
            status.get("official_result_evidence_inserted_race_rows")
        ),
        "official_result_evidence_inserted_runner_rows": int_or_zero(
            status.get("official_result_evidence_inserted_runner_rows")
        ),
        "official_result_evidence_blocker_reason_counts": dict(
            status.get("official_result_evidence_blocker_reason_counts") or {}
        ),
        "live_odds_backlog_enabled": bool(status.get("live_odds_backlog_enabled")),
        "live_odds_backlog_lookback_days": int_or_zero(
            status.get("live_odds_backlog_lookback_days")
        ),
        "live_odds_backlog_target_dates": list(
            status.get("live_odds_backlog_target_dates") or []
        ),
        "live_odds_backlog_discovered_race_count": int_or_zero(
            status.get("live_odds_backlog_discovered_race_count")
        ),
        "live_odds_backlog_discovered_race_ids": list(
            status.get("live_odds_backlog_discovered_race_ids") or []
        ),
        "live_odds_backlog_candidate_race_count": int_or_zero(
            status.get("live_odds_backlog_candidate_race_count")
        ),
        "live_odds_backlog_candidate_race_ids": list(
            status.get("live_odds_backlog_candidate_race_ids") or []
        ),
        "live_odds_backlog_unresolved_race_count": int_or_zero(
            status.get("live_odds_backlog_unresolved_race_count")
        ),
        "live_odds_backlog_unresolved_race_ids": list(
            status.get("live_odds_backlog_unresolved_race_ids") or []
        ),
        "live_odds_backlog_unresolved_races": list(
            status.get("live_odds_backlog_unresolved_races") or []
        ),
        "live_odds_backlog_unresolved_reason_counts": dict(
            status.get("live_odds_backlog_unresolved_reason_counts") or {}
        ),
        "live_odds_backlog_unresolved_recovery_action_counts": dict(
            status.get("live_odds_backlog_unresolved_recovery_action_counts") or {}
        ),
        "live_odds_backlog_unresolved_alias_status_counts": dict(
            status.get("live_odds_backlog_unresolved_alias_status_counts") or {}
        ),
        "live_odds_backlog_retryable_exact_shadow_match_race_ids": list(
            status.get("live_odds_backlog_retryable_exact_shadow_match_race_ids") or []
        ),
        "live_odds_backlog_no_exact_shadow_match_race_ids": list(
            status.get("live_odds_backlog_no_exact_shadow_match_race_ids") or []
        ),
        "live_odds_backlog_retryable_exact_shadow_match_race_count": int_or_zero(
            status.get("live_odds_backlog_retryable_exact_shadow_match_race_count")
        ),
        "live_odds_backlog_no_exact_shadow_match_race_count": int_or_zero(
            status.get("live_odds_backlog_no_exact_shadow_match_race_count")
        ),
        "live_odds_backlog_recovery_queue_path": recovery_queue_path,
        "live_odds_backlog_recovery_queue_diagnostic_only": bool(
            status.get("live_odds_backlog_recovery_queue_diagnostic_only", True)
        ),
        "live_odds_backlog_recovery_queue_join_acceptance_changed": bool(
            status.get("live_odds_backlog_recovery_queue_join_acceptance_changed")
        ),
        "live_odds_backlog_recovery_queue_db_write_performed": bool(
            status.get("live_odds_backlog_recovery_queue_db_write_performed")
        ),
        "live_odds_backlog_awaiting_official_result_evidence_race_count": int_or_zero(
            awaiting_official_result_evidence_race_count
        ),
        "live_odds_backlog_awaiting_official_result_evidence_race_ids": (
            awaiting_official_result_evidence_race_ids
        ),
        "live_odds_backlog_awaiting_official_result_evidence_authorized_action": (
            awaiting_official_result_evidence_authorized_action
        ),
        "live_odds_backlog_awaiting_official_result_recheck_ready_race_count": int_or_zero(
            awaiting_official_result_recheck_ready_race_count
        ),
        "live_odds_backlog_runner_set_validation_path": runner_set_validation_path,
        "live_odds_backlog_runner_set_validation_retryable_race_count": int_or_zero(
            status.get("live_odds_backlog_runner_set_validation_retryable_race_count")
        ),
        "live_odds_backlog_runner_set_validation_exact_match_race_count": int_or_zero(
            status.get("live_odds_backlog_runner_set_validation_exact_match_race_count")
        ),
        "live_odds_backlog_runner_set_validation_blocked_race_count": int_or_zero(
            status.get("live_odds_backlog_runner_set_validation_blocked_race_count")
        ),
        "live_odds_backlog_runner_set_validation_diagnostic_only": bool(
            status.get("live_odds_backlog_runner_set_validation_diagnostic_only", True)
        ),
        "live_odds_backlog_runner_set_validation_join_authorized": bool(
            status.get("live_odds_backlog_runner_set_validation_join_authorized")
        ),
        "live_odds_backlog_runner_set_validation_db_write_performed": bool(
            status.get("live_odds_backlog_runner_set_validation_db_write_performed")
        ),
        "live_odds_backlog_join_eligibility_packet_path": join_eligibility_packet_path,
        "live_odds_backlog_join_eligibility_evaluated_race_count": int_or_zero(
            status.get("live_odds_backlog_join_eligibility_evaluated_race_count")
        ),
        "live_odds_backlog_join_eligibility_eligible_report_only_race_count": int_or_zero(
            status.get(
                "live_odds_backlog_join_eligibility_eligible_report_only_race_count"
            )
        ),
        "live_odds_backlog_join_eligibility_blocked_race_count": int_or_zero(
            status.get("live_odds_backlog_join_eligibility_blocked_race_count")
        ),
        "live_odds_backlog_join_eligibility_blocker_counts": dict(
            join_eligibility_blocker_counts
        ),
        "live_odds_backlog_join_eligibility_diagnostic_only": bool(
            status.get("live_odds_backlog_join_eligibility_diagnostic_only", True)
        ),
        "live_odds_backlog_join_eligibility_join_authorized": bool(
            status.get("live_odds_backlog_join_eligibility_join_authorized")
        ),
        "live_odds_backlog_join_eligibility_db_write_performed": bool(
            status.get("live_odds_backlog_join_eligibility_db_write_performed")
        ),
        "live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count": int_or_zero(
            status.get(
                "live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count"
            )
        ),
        "shadow_run_candidate_source_report": status.get(
            "shadow_run_candidate_source_report"
        ),
        "candidate_source": status.get("candidate_source"),
        "target_date": status.get("target_date"),
        "upcoming_dir": status.get("upcoming_dir"),
        "shadow_run_dir": status.get("shadow_run_dir"),
        "no_write_guarantees": status.get("no_write_guarantees") or dict(NO_WRITE_GUARANTEES),
    }


def live_odds_backlog_operational_fields(
    autonomous_official_result_capture_status: Mapping[str, Any] | None,
) -> dict[str, Any]:
    status = autonomous_official_result_capture_status or {}
    unresolved_races = list(status.get("live_odds_backlog_unresolved_races") or [])
    recovery_action_counts = Counter(
        str(row.get("recovery_action") or "missing_recovery_action")
        for row in unresolved_races
        if isinstance(row, Mapping)
    )
    alias_status_counts = Counter(
        str(row.get("alias_reconciliation_status") or "missing_alias_reconciliation_status")
        for row in unresolved_races
        if isinstance(row, Mapping)
    )
    recovery_action_count_source = (
        dict(status.get("live_odds_backlog_unresolved_recovery_action_counts") or {})
        or dict(sorted(recovery_action_counts.items()))
    )
    alias_status_count_source = (
        dict(status.get("live_odds_backlog_unresolved_alias_status_counts") or {})
        or dict(sorted(alias_status_counts.items()))
    )
    retryable_exact_shadow_match_race_ids = list(
        status.get("live_odds_backlog_retryable_exact_shadow_match_race_ids") or []
    )
    if not retryable_exact_shadow_match_race_ids and unresolved_races:
        retryable_exact_shadow_match_race_ids = sorted(
            str(row.get("race_id"))
            for row in unresolved_races
            if isinstance(row, Mapping)
            and row.get("race_id")
            and row.get("recovery_action") == "validate_runner_set_then_alias_join"
            and row.get("alias_reconciliation_status") == "EXACT_SHADOW_ARTIFACT_MATCH_FOUND"
        )
    no_exact_shadow_match_race_ids = list(
        status.get("live_odds_backlog_no_exact_shadow_match_race_ids") or []
    )
    if not no_exact_shadow_match_race_ids and unresolved_races:
        no_exact_shadow_match_race_ids = sorted(
            str(row.get("race_id"))
            for row in unresolved_races
            if isinstance(row, Mapping)
            and row.get("race_id")
            and row.get("alias_reconciliation_status") == "NO_EXACT_SHADOW_ARTIFACT_MATCH"
        )
    retryable_exact_shadow_match_count = int_or_zero(
        status.get("live_odds_backlog_retryable_exact_shadow_match_race_count")
    )
    if not retryable_exact_shadow_match_count and unresolved_races:
        retryable_exact_shadow_match_count = len(retryable_exact_shadow_match_race_ids)
    no_exact_shadow_match_count = int_or_zero(
        status.get("live_odds_backlog_no_exact_shadow_match_race_count")
    )
    if not no_exact_shadow_match_count:
        no_exact_shadow_match_count = len(no_exact_shadow_match_race_ids) or int_or_zero(
            alias_status_count_source.get("NO_EXACT_SHADOW_ARTIFACT_MATCH")
        )
    return {
        "live_odds_backlog_enabled": bool(status.get("live_odds_backlog_enabled")),
        "live_odds_backlog_lookback_days": int_or_zero(
            status.get("live_odds_backlog_lookback_days")
        ),
        "live_odds_backlog_target_dates": list(
            status.get("live_odds_backlog_target_dates") or []
        ),
        "live_odds_backlog_discovered_race_count": int_or_zero(
            status.get("live_odds_backlog_discovered_race_count")
        ),
        "live_odds_backlog_candidate_race_count": int_or_zero(
            status.get("live_odds_backlog_candidate_race_count")
        ),
        "live_odds_backlog_unresolved_race_count": int_or_zero(
            status.get("live_odds_backlog_unresolved_race_count")
        ),
        "live_odds_backlog_unresolved_race_ids": list(
            status.get("live_odds_backlog_unresolved_race_ids") or []
        ),
        "live_odds_backlog_unresolved_reason_counts": dict(
            status.get("live_odds_backlog_unresolved_reason_counts") or {}
        ),
        "live_odds_backlog_unresolved_recovery_action_counts": recovery_action_count_source,
        "live_odds_backlog_unresolved_alias_status_counts": alias_status_count_source,
        "live_odds_backlog_retryable_exact_shadow_match_race_ids": (
            retryable_exact_shadow_match_race_ids
        ),
        "live_odds_backlog_no_exact_shadow_match_race_ids": (
            no_exact_shadow_match_race_ids
        ),
        "live_odds_backlog_retryable_exact_shadow_match_race_count": (
            retryable_exact_shadow_match_count
        ),
        "live_odds_backlog_no_exact_shadow_match_race_count": no_exact_shadow_match_count,
        "live_odds_backlog_recovery_queue_path": status.get(
            "live_odds_backlog_recovery_queue_path"
        ),
        "live_odds_backlog_recovery_queue_diagnostic_only": bool(
            status.get("live_odds_backlog_recovery_queue_diagnostic_only", True)
        ),
        "live_odds_backlog_recovery_queue_join_acceptance_changed": bool(
            status.get("live_odds_backlog_recovery_queue_join_acceptance_changed")
        ),
        "live_odds_backlog_recovery_queue_db_write_performed": bool(
            status.get("live_odds_backlog_recovery_queue_db_write_performed")
        ),
        "live_odds_backlog_awaiting_official_result_evidence_race_count": int_or_zero(
            status.get("live_odds_backlog_awaiting_official_result_evidence_race_count")
        ),
        "live_odds_backlog_awaiting_official_result_evidence_race_ids": list(
            status.get("live_odds_backlog_awaiting_official_result_evidence_race_ids")
            or []
        ),
        "live_odds_backlog_awaiting_official_result_evidence_authorized_action": (
            status.get(
                "live_odds_backlog_awaiting_official_result_evidence_authorized_action"
            )
        ),
        "live_odds_backlog_awaiting_official_result_recheck_ready_race_count": int_or_zero(
            status.get(
                "live_odds_backlog_awaiting_official_result_recheck_ready_race_count"
            )
        ),
        "live_odds_backlog_runner_set_validation_path": status.get(
            "live_odds_backlog_runner_set_validation_path"
        ),
        "live_odds_backlog_runner_set_validation_retryable_race_count": int_or_zero(
            status.get("live_odds_backlog_runner_set_validation_retryable_race_count")
        ),
        "live_odds_backlog_runner_set_validation_exact_match_race_count": int_or_zero(
            status.get("live_odds_backlog_runner_set_validation_exact_match_race_count")
        ),
        "live_odds_backlog_runner_set_validation_blocked_race_count": int_or_zero(
            status.get("live_odds_backlog_runner_set_validation_blocked_race_count")
        ),
        "live_odds_backlog_runner_set_validation_diagnostic_only": bool(
            status.get("live_odds_backlog_runner_set_validation_diagnostic_only", True)
        ),
        "live_odds_backlog_runner_set_validation_join_authorized": bool(
            status.get("live_odds_backlog_runner_set_validation_join_authorized")
        ),
        "live_odds_backlog_runner_set_validation_db_write_performed": bool(
            status.get("live_odds_backlog_runner_set_validation_db_write_performed")
        ),
        "live_odds_backlog_join_eligibility_packet_path": status.get(
            "live_odds_backlog_join_eligibility_packet_path"
        ),
        "live_odds_backlog_join_eligibility_evaluated_race_count": int_or_zero(
            status.get("live_odds_backlog_join_eligibility_evaluated_race_count")
        ),
        "live_odds_backlog_join_eligibility_eligible_report_only_race_count": int_or_zero(
            status.get(
                "live_odds_backlog_join_eligibility_eligible_report_only_race_count"
            )
        ),
        "live_odds_backlog_join_eligibility_blocked_race_count": int_or_zero(
            status.get("live_odds_backlog_join_eligibility_blocked_race_count")
        ),
        "live_odds_backlog_join_eligibility_blocker_counts": dict(
            status.get("live_odds_backlog_join_eligibility_blocker_counts") or {}
        ),
        "live_odds_backlog_join_eligibility_diagnostic_only": bool(
            status.get("live_odds_backlog_join_eligibility_diagnostic_only", True)
        ),
        "live_odds_backlog_join_eligibility_join_authorized": bool(
            status.get("live_odds_backlog_join_eligibility_join_authorized")
        ),
        "live_odds_backlog_join_eligibility_db_write_performed": bool(
            status.get("live_odds_backlog_join_eligibility_db_write_performed")
        ),
        "live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count": int_or_zero(
            status.get(
                "live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count"
            )
        ),
    }


def live_odds_backlog_state_fields(
    autonomous_official_result_capture_status: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        f"last_{key}": value
        for key, value in live_odds_backlog_operational_fields(
            autonomous_official_result_capture_status
        ).items()
    }


def next_prejump_refresh_window_from_autopilot(
    autopilot_output_dir: Path | None,
) -> dict[str, Any] | None:
    if autopilot_output_dir is None:
        return None
    report_path = autopilot_output_dir / "refresh_prejump_report.json"
    report = load_json(report_path)
    if not report:
        return None
    window = report.get("next_preferred_window")
    if not isinstance(window, Mapping):
        return None
    next_race = window.get("next_race")
    if not isinstance(next_race, Mapping):
        next_race = {}
    selected_races = report.get("selected_races")
    if not isinstance(selected_races, list):
        selected_races = []
    return {
        "schema_version": "shadow_daemon_next_prejump_refresh_window_v1",
        "status": window.get("status"),
        "reason": window.get("reason"),
        "report_path": relpath(report_path),
        "generated_at": report.get("generated_at"),
        "recommended_rerun_after_local": window.get("recommended_rerun_after_local"),
        "next_window_opens_at": window.get("next_window_opens_at"),
        "next_window_closes_at": window.get("next_window_closes_at"),
        "minutes_until_window_opens": window.get("minutes_until_window_opens"),
        "minutes_until_window_closes": window.get("minutes_until_window_closes"),
        "selected_count": int(report.get("selected_count") or 0),
        "total_races_found": int(report.get("total_races_found") or 0),
        "selected_race_count": len(selected_races),
        "next_race": {
            "race_id": next_race.get("race_id"),
            "date": next_race.get("date"),
            "venue": next_race.get("venue"),
            "race_number": next_race.get("race_number"),
            "race_time": next_race.get("race_time"),
            "jump_datetime": next_race.get("jump_datetime"),
            "minutes_to_jump": next_race.get("minutes_to_jump"),
            "bucket": next_race.get("bucket"),
            "selected": next_race.get("selected"),
            "race_url": next_race.get("race_url"),
        },
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def prejump_metadata_status_from_daily_run(
    daily_shadow_run_dir: Path | None,
) -> dict[str, Any] | None:
    if daily_shadow_run_dir is None:
        return None
    report_path = daily_shadow_run_dir / "prejump_metadata_report.json"
    report = load_json(report_path)
    if not report:
        return None
    eligible_count = int(report.get("eligible_count") or 0)
    verified_count = int(report.get("eligible_with_verified_prejump_metadata") or 0)
    malformed_count = int(report.get("malformed_prejump_metadata_count") or 0)
    unsafe_rows = list(report.get("unsafe_or_incomplete_metadata") or [])
    required_fields = list(report.get("required_fields") or [])
    field_coverage = report.get("field_coverage") or {}
    missing_required_fields = [
        field
        for field in required_fields
        if eligible_count > 0
        and int((field_coverage.get(field) or {}).get("eligible_present_rows") or 0)
        < eligible_count
    ]
    if eligible_count == 0 and report.get("status") == "PASS":
        status = "NO_ELIGIBLE_PREJUMP_RACES"
    elif report.get("status") == "PASS" and verified_count == eligible_count and not missing_required_fields:
        status = "PREJUMP_METADATA_PASS"
    elif report.get("status") == "PASS":
        status = "PREJUMP_METADATA_PARTIAL"
    else:
        status = "PREJUMP_METADATA_FAIL"
    return {
        "schema_version": "shadow_daemon_prejump_metadata_summary_v1",
        "status": status,
        "report_status": report.get("status"),
        "report_path": relpath(report_path),
        "eligible_count": eligible_count,
        "eligible_with_verified_prejump_metadata": verified_count,
        "malformed_prejump_metadata_count": malformed_count,
        "unsafe_or_incomplete_metadata_count": len(unsafe_rows),
        "stale_with_prejump_metadata_count": int(
            report.get("stale_with_prejump_metadata_count") or 0
        ),
        "required_fields": required_fields,
        "missing_required_fields": missing_required_fields,
        "field_coverage": field_coverage,
        "rejected_metadata_sources": report.get("rejected_metadata_sources") or [],
    }


def _prejump_metadata_run_summary(
    *,
    report_path: Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    eligible_count = int(report.get("eligible_count") or 0)
    verified_count = int(report.get("eligible_with_verified_prejump_metadata") or 0)
    malformed_count = int(report.get("malformed_prejump_metadata_count") or 0)
    stale_count = int(report.get("stale_with_prejump_metadata_count") or 0)
    required_fields = list(report.get("required_fields") or [])
    field_coverage = report.get("field_coverage") or {}
    missing_required_fields = [
        field
        for field in required_fields
        if eligible_count > 0
        and int((field_coverage.get(field) or {}).get("eligible_present_rows") or 0)
        < eligible_count
    ]
    if eligible_count == 0 and report.get("status") == "PASS":
        status = "NO_ELIGIBLE_PREJUMP_RACES"
    elif report.get("status") == "PASS" and verified_count == eligible_count and not missing_required_fields:
        status = "PREJUMP_METADATA_PASS"
    elif report.get("status") == "PASS":
        status = "PREJUMP_METADATA_PARTIAL"
    else:
        status = "PREJUMP_METADATA_FAIL"
    return {
        "run_dir": relpath(report_path.parent),
        "report_path": relpath(report_path),
        "report_status": report.get("status"),
        "status": status,
        "eligible_count": eligible_count,
        "eligible_with_verified_prejump_metadata": verified_count,
        "malformed_prejump_metadata_count": malformed_count,
        "stale_with_prejump_metadata_count": stale_count,
        "missing_required_fields": missing_required_fields,
        "rejected_metadata_sources": report.get("rejected_metadata_sources") or [],
    }


def build_prejump_metadata_trend_report(
    *,
    evidence_root: Path,
    output_dir: Path,
    generated_at: datetime,
    limit: int = 20,
) -> dict[str, Any]:
    """Aggregate recent daily pre-jump metadata reports without touching source data."""

    report_paths = sorted(
        path
        for path in evidence_root.glob("daily_race_ingest_shadow_*/prejump_metadata_report.json")
        if path.is_file()
    )
    selected_paths = report_paths[-limit:] if limit > 0 else report_paths
    runs: list[dict[str, Any]] = []
    field_totals: dict[str, dict[str, Any]] = {}
    rejected_sources: Counter[str] = Counter()
    missing_required_fields: Counter[str] = Counter()

    for report_path in selected_paths:
        report = load_json(report_path)
        if not report:
            continue
        summary = _prejump_metadata_run_summary(report_path=report_path, report=report)
        runs.append(summary)
        eligible_count = int(summary.get("eligible_count") or 0)
        field_coverage = report.get("field_coverage") or {}
        for field in report.get("required_fields") or []:
            field_summary = field_totals.setdefault(
                str(field),
                {"eligible_rows": 0, "present_rows": 0, "missing_rows": 0},
            )
            present_rows = int((field_coverage.get(field) or {}).get("eligible_present_rows") or 0)
            field_summary["eligible_rows"] += eligible_count
            field_summary["present_rows"] += present_rows
            field_summary["missing_rows"] += max(0, eligible_count - present_rows)
        for source in report.get("rejected_metadata_sources") or []:
            if source not in (None, ""):
                rejected_sources[str(source)] += 1
        for field in summary.get("missing_required_fields") or []:
            missing_required_fields[str(field)] += 1

    total_eligible = sum(int(run.get("eligible_count") or 0) for run in runs)
    total_verified = sum(
        int(run.get("eligible_with_verified_prejump_metadata") or 0) for run in runs
    )
    total_malformed = sum(int(run.get("malformed_prejump_metadata_count") or 0) for run in runs)
    for field_summary in field_totals.values():
        eligible_rows = int(field_summary["eligible_rows"] or 0)
        present_rows = int(field_summary["present_rows"] or 0)
        field_summary["present_pct"] = present_rows / eligible_rows if eligible_rows else None

    runs_with_eligible = sum(1 for run in runs if int(run.get("eligible_count") or 0) > 0)
    runs_with_full_verified_metadata = sum(
        1
        for run in runs
        if int(run.get("eligible_count") or 0) > 0
        and run.get("status") == "PREJUMP_METADATA_PASS"
    )
    runs_needing_attention = sum(
        1
        for run in runs
        if int(run.get("eligible_count") or 0) > 0
        and run.get("status") != "PREJUMP_METADATA_PASS"
    )
    if not runs:
        status = "NO_PREJUMP_METADATA_REPORTS"
    elif total_eligible == 0:
        status = "NO_ELIGIBLE_PREJUMP_RACES"
    elif (
        total_verified == total_eligible
        and total_malformed == 0
        and runs_with_full_verified_metadata == runs_with_eligible
    ):
        status = "PREJUMP_METADATA_TREND_PASS"
    else:
        status = "PREJUMP_METADATA_TREND_NEEDS_ATTENTION"

    trend = {
        "schema_version": "shadow_daemon_prejump_metadata_trend_v1",
        "generated_at": generated_at.isoformat(),
        "status": status,
        "evidence_root": relpath(evidence_root),
        "runs_checked": len(runs),
        "available_report_count": len(report_paths),
        "limit": limit,
        "runs_with_eligible_prejump_races": runs_with_eligible,
        "runs_with_full_verified_metadata": runs_with_full_verified_metadata,
        "runs_needing_metadata_attention": runs_needing_attention,
        "total_eligible_prejump_races": total_eligible,
        "total_verified_prejump_metadata_races": total_verified,
        "total_malformed_prejump_metadata_races": total_malformed,
        "verified_metadata_rate": total_verified / total_eligible if total_eligible else None,
        "field_totals": dict(sorted(field_totals.items())),
        "missing_required_field_counts": dict(sorted(missing_required_fields.items())),
        "rejected_metadata_source_counts": dict(sorted(rejected_sources.items())),
        "runs": runs,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }
    write_json(output_dir / "prejump_metadata_trend_report.json", trend)
    return trend


def build_read_only_odds_coverage_report(
    *,
    db_path: Path,
    output_dir: Path,
    generated_at: datetime,
    evidence_root: Path = DEFAULT_EVIDENCE_ROOT,
    stale_after_hours: float = 6.0,
) -> dict[str, Any]:
    """Write DB odds coverage diagnostics without scraping, scoring, or DB writes."""

    report_path = output_dir / "odds_coverage_report.json"
    summary: dict[str, Any] = {
        "schema_version": "shadow_daemon_read_only_odds_coverage_summary_v1",
        "generated_at": generated_at.isoformat(),
        "status": "FAILED",
        "mode": "read_only_coverage_diagnostic",
        "report_path": relpath(report_path),
        "odds_capture_performed": False,
        "odds_used_for_shadow_scoring": False,
        "shadow_model_input": False,
        "betting_action": False,
        "ev_action": False,
        "db_write": False,
        "stale_after_hours": stale_after_hours,
    }
    try:
        from accuracy_program.odds_coverage import analyze_odds_coverage

        coverage = analyze_odds_coverage(
            db_path,
            current_only=True,
            stale_after_hours=stale_after_hours,
            now=generated_at,
        )
        counts = coverage.get("counts") or {}
        safe = coverage.get("safe_match_counts") or {}
        rates = coverage.get("coverage_rates") or {}
        timestamp_quality = (coverage.get("timestamp_quality") or {}).get("live_odds_current_win") or {}
        source_url_quality = coverage.get("source_url_quality") or {}
        dog_level_rows = int(counts.get("dog_level_win_odds_rows") or 0)
        fresh_identity_split = fresh_strict_odds_identity_split(
            db_path=db_path,
            generated_at=generated_at,
            stale_after_hours=stale_after_hours,
        )
        fresh_prediction_coverage = fresh_odds_shadow_prediction_coverage(
            db_path=db_path,
            evidence_root=evidence_root,
            generated_at=generated_at,
            stale_after_hours=stale_after_hours,
        )
        race_id_mismatch_rows = sum(
            int(row.get("rows") or 0)
            for row in (coverage.get("venue_date_race_mismatches") or {}).get(
                "counts"
            )
            or []
        )
        old_odds_row_audit = {
            "mode": "read_only_old_odds_audit",
            "stale_rows": int(timestamp_quality.get("stale_rows") or 0),
            "missing_source_url_rows": int(
                source_url_quality.get("rows_missing_source_url") or 0
            ),
            "race_id_mismatch_rows": race_id_mismatch_rows,
            "dog_name_box_conflict_rows": int(
                safe.get("dog_name_box_conflict_rows") or 0
            ),
            "ambiguous_strict_identity_rows": int(
                safe.get("ambiguous_strict_identity_rows") or 0
            ),
            "db_write": False,
            "odds_capture_performed": False,
        }
        summary.update(
            {
                "status": "SUCCESS" if dog_level_rows else "NO_CURRENT_DOG_LEVEL_WIN_ODDS",
                "tables": coverage.get("tables") or {},
                "live_odds_rows": int(counts.get("live_odds_rows") or 0),
                "odds_history_rows": int(counts.get("odds_history_rows") or 0),
                "live_odds_races": int(counts.get("live_odds_races") or 0),
                "dog_level_win_odds_rows": dog_level_rows,
                "races_with_dog_level_win_odds": int(
                    counts.get("races_with_dog_level_win_odds") or 0
                ),
                "safe_direct_identity_matches": int(
                    safe.get("safe_direct_identity_matches") or 0
                ),
                "ambiguous_strict_identity_rows": int(
                    safe.get("ambiguous_strict_identity_rows") or 0
                ),
                "safe_direct_identity_match_rate": rates.get("safe_direct_identity_match_rate"),
                "stale_current_win_rows": int(timestamp_quality.get("stale_rows") or 0),
                "fresh_current_win_rows": int(
                    fresh_identity_split.get("fresh_current_win_rows") or 0
                ),
                "fresh_safe_direct_identity_matches": int(
                    fresh_identity_split.get("fresh_safe_direct_identity_matches") or 0
                ),
                "fresh_safe_direct_identity_match_rate": fresh_identity_split.get(
                    "fresh_safe_direct_identity_match_rate"
                ),
                "fresh_unmatched_rows": int(
                    fresh_identity_split.get("fresh_unmatched_rows") or 0
                ),
                "fresh_odds_prediction_coverage_status": fresh_prediction_coverage.get(
                    "status"
                ),
                "fresh_odds_prediction_races": int(
                    fresh_prediction_coverage.get("fresh_current_win_odds_races")
                    or 0
                ),
                "fresh_odds_races_with_primary_predictions": int(
                    fresh_prediction_coverage.get("races_with_primary_predictions")
                    or 0
                ),
                "fresh_odds_races_with_stage2_predictions": int(
                    fresh_prediction_coverage.get("races_with_stage2_predictions")
                    or 0
                ),
                "fresh_odds_races_missing_prediction_artifact": int(
                    fresh_prediction_coverage.get("races_missing_prediction_artifact")
                    or 0
                ),
                "fresh_odds_runner_keys": int(
                    fresh_prediction_coverage.get("fresh_odds_runner_keys") or 0
                ),
                "fresh_odds_runner_keys_with_primary_prediction_match": int(
                    fresh_prediction_coverage.get(
                        "fresh_odds_runner_keys_with_primary_prediction_match"
                    )
                    or 0
                ),
                "fresh_odds_runner_keys_with_stage2_prediction_match": int(
                    fresh_prediction_coverage.get(
                        "fresh_odds_runner_keys_with_stage2_prediction_match"
                    )
                    or 0
                ),
                "fresh_odds_runner_keys_missing_primary_prediction_match": int(
                    fresh_prediction_coverage.get(
                        "fresh_odds_runner_keys_missing_primary_prediction_match"
                    )
                    or 0
                ),
                "fresh_odds_runner_keys_missing_stage2_prediction_match": int(
                    fresh_prediction_coverage.get(
                        "fresh_odds_runner_keys_missing_stage2_prediction_match"
                    )
                    or 0
                ),
                "timestamped_current_win_rows": int(
                    timestamp_quality.get("timestamped_rows") or 0
                ),
                "source_url_rows_checked": int(source_url_quality.get("rows_checked") or 0),
                "source_url_rows_missing": int(
                    source_url_quality.get("rows_missing_source_url") or 0
                ),
                "post_race_source_url_rows": int(
                    source_url_quality.get("post_race_source_url_rows") or 0
                ),
                "source_provenance": coverage.get("source_provenance") or {},
                "old_odds_row_audit": old_odds_row_audit,
            }
        )
        payload = {
            "schema_version": "shadow_daemon_read_only_odds_coverage_report_v1",
            "summary": summary,
            "coverage": coverage,
            "old_odds_row_audit": old_odds_row_audit,
            "fresh_strict_identity_split": fresh_identity_split,
            "fresh_odds_shadow_prediction_coverage": fresh_prediction_coverage,
            "no_write_guarantees": {
                "odds_capture_performed": False,
                "odds_used_for_shadow_scoring": False,
                "shadow_model_input": False,
                "betting_action": False,
                "ev_action": False,
                "db_write": False,
            },
        }
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}:{exc}"
        summary["old_odds_row_audit"] = {
            "mode": "read_only_old_odds_audit",
            "status": "DATA_MISSING",
            "reason": "odds_coverage_analysis_failed",
            "db_write": False,
            "odds_capture_performed": False,
        }
        payload = {
            "schema_version": "shadow_daemon_read_only_odds_coverage_report_v1",
            "summary": summary,
            "coverage": None,
            "old_odds_row_audit": summary["old_odds_row_audit"],
            "no_write_guarantees": {
                "odds_capture_performed": False,
                "odds_used_for_shadow_scoring": False,
                "shadow_model_input": False,
                "betting_action": False,
                "ev_action": False,
                "db_write": False,
            },
        }
    write_json(report_path, payload)
    return summary


def _fresh_current_win_odds_by_race(
    *,
    db_path: Path,
    generated_at: datetime,
    stale_after_hours: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    audit: dict[str, Any] = {
        "db_write": False,
        "odds_capture_performed": False,
        "rows_checked": 0,
        "fresh_rows": 0,
        "stale_rows": 0,
        "null_timestamp_rows": 0,
        "invalid_timestamp_rows": 0,
    }
    by_race: dict[str, dict[str, Any]] = {}
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.create_function("daemon_norm_dog", 1, _daemon_norm_dog)
    try:
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "live_odds" not in tables:
            audit["reason"] = "live_odds_table_missing"
            return by_race, audit
        live_cols = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(live_odds)").fetchall()
        }
        timestamp_expr = (
            "COALESCE(capture_timestamp, timestamp)"
            if "capture_timestamp" in live_cols
            else "timestamp"
        )
        current_clause = (
            "AND (is_current = 1 OR is_current IS NULL)"
            if "is_current" in live_cols
            else ""
        )
        rows = conn.execute(
            f"""
            SELECT
                race_id,
                {timestamp_expr} AS odds_timestamp,
                box_number,
                daemon_norm_dog(COALESCE(dog_clean_name, dog_name)) AS dog_key,
                COUNT(*) AS rows
            FROM live_odds
            WHERE lower(COALESCE(market_type, 'win')) = 'win'
              AND odds_decimal IS NOT NULL
              AND odds_decimal > 1
              {current_clause}
            GROUP BY race_id, {timestamp_expr}, box_number, dog_key
            """
        ).fetchall()
    finally:
        conn.close()

    for row in rows:
        row_count = int(row["rows"] or 0)
        audit["rows_checked"] += row_count
        timestamp = parse_datetime_value(
            row["odds_timestamp"],
            default_tz=generated_at.tzinfo,
        )
        if timestamp is None:
            if row["odds_timestamp"] in (None, ""):
                audit["null_timestamp_rows"] += row_count
            else:
                audit["invalid_timestamp_rows"] += row_count
            continue
        compare_generated_at = generated_at
        compare_timestamp = timestamp
        if compare_timestamp.tzinfo is not None and compare_generated_at.tzinfo is None:
            compare_generated_at = compare_generated_at.replace(
                tzinfo=compare_timestamp.tzinfo
            )
        elif compare_timestamp.tzinfo is None and compare_generated_at.tzinfo is not None:
            compare_generated_at = compare_generated_at.replace(tzinfo=None)
        age_hours = (compare_generated_at - compare_timestamp).total_seconds() / 3600.0
        if age_hours > stale_after_hours:
            audit["stale_rows"] += row_count
            continue
        race_id = str(row["race_id"] or "").strip()
        if not race_id:
            continue
        audit["fresh_rows"] += row_count
        item = by_race.setdefault(
            race_id,
            {
                "race_id": race_id,
                "fresh_current_win_rows": 0,
                "first_odds_timestamp": row["odds_timestamp"],
                "last_odds_timestamp": row["odds_timestamp"],
                "fresh_runner_keys": [],
            },
        )
        item["fresh_current_win_rows"] += row_count
        runner_key = f"{int_or_zero(row['box_number'])}|{row['dog_key'] or ''}"
        if runner_key not in item["fresh_runner_keys"]:
            item["fresh_runner_keys"].append(runner_key)
        item["first_odds_timestamp"] = min(
            str(item["first_odds_timestamp"] or ""),
            str(row["odds_timestamp"] or ""),
        )
        item["last_odds_timestamp"] = max(
            str(item["last_odds_timestamp"] or ""),
            str(row["odds_timestamp"] or ""),
        )
    return by_race, audit


def prediction_runner_identity_key(row: Mapping[str, Any]) -> str:
    box = row.get("box_number")
    if box in (None, ""):
        box = row.get("box")
    dog_name = (
        row.get("dog_name")
        or row.get("dog_clean_name")
        or row.get("runner_name")
        or row.get("greyhound")
    )
    return f"{int_or_zero(box)}|{_daemon_norm_dog(dog_name)}"


def fresh_odds_shadow_prediction_coverage(
    *,
    db_path: Path,
    evidence_root: Path,
    generated_at: datetime,
    stale_after_hours: float = 6.0,
    max_shadow_runs: int = 256,
    sample_limit: int = 20,
) -> dict[str, Any]:
    """Read-only diagnostic for fresh odds races missing shadow prediction artifacts."""

    result: dict[str, Any] = {
        "schema_version": "shadow_daemon_fresh_odds_shadow_prediction_coverage_v1",
        "generated_at": generated_at.isoformat(),
        "status": "DATA_MISSING",
        "mode": "read_only_artifact_coverage_diagnostic",
        "stale_after_hours": stale_after_hours,
        "db_write": False,
        "odds_capture_performed": False,
        "join_acceptance_changed": False,
        "model_scoring_changed": False,
    }
    try:
        fresh_by_race, fresh_audit = _fresh_current_win_odds_by_race(
            db_path=db_path,
            generated_at=generated_at,
            stale_after_hours=stale_after_hours,
        )
    except Exception as exc:
        result["status"] = "FAILED"
        result["error"] = f"{type(exc).__name__}:{exc}"
        return result

    race_ids = set(fresh_by_race)
    runs = sorted(evidence_root.glob("daily_race_ingest_shadow_*"))[-max_shadow_runs:]
    primary_artifacts: dict[str, list[str]] = defaultdict(list)
    stage2_artifacts: dict[str, list[str]] = defaultdict(list)
    primary_runner_keys: dict[str, set[str]] = defaultdict(set)
    stage2_runner_keys: dict[str, set[str]] = defaultdict(set)
    latest_run: dict[str, Any] | None = None
    prediction_rows_checked = 0
    stage2_rows_checked = 0
    for run_dir in runs:
        primary_rows = read_jsonl_rows(run_dir / "shadow_predictions.jsonl")
        stage2_rows = read_jsonl_rows(run_dir / "stage2_shadow_predictions.jsonl")
        prediction_rows_checked += len(primary_rows)
        stage2_rows_checked += len(stage2_rows)
        manifest = load_json(run_dir / "shadow_manifest.json") or {}
        latest_run = {
            "run_dir": relpath(run_dir),
            "final_status": manifest.get("final_status"),
            "input_summary": manifest.get("input_summary"),
            "prediction_rows": len(primary_rows),
            "stage2_prediction_rows": len(stage2_rows),
        }
        for row in primary_rows:
            race_id = str(row.get("race_id") or "").strip()
            if race_id in race_ids:
                primary_artifacts[race_id].append(relpath(run_dir) or str(run_dir))
                primary_runner_keys[race_id].add(prediction_runner_identity_key(row))
        for row in stage2_rows:
            race_id = str(row.get("race_id") or "").strip()
            if race_id in race_ids:
                stage2_artifacts[race_id].append(relpath(run_dir) or str(run_dir))
                stage2_runner_keys[race_id].add(prediction_runner_identity_key(row))

    races_with_primary = {race_id for race_id in race_ids if primary_artifacts.get(race_id)}
    races_with_stage2 = {race_id for race_id in race_ids if stage2_artifacts.get(race_id)}
    races_missing_any = race_ids - (races_with_primary | races_with_stage2)
    fresh_runner_keys = {
        (str(item["race_id"]), str(key))
        for item in fresh_by_race.values()
        for key in item.get("fresh_runner_keys") or []
    }
    primary_runner_matches = {
        (race_id, runner_key)
        for race_id, runner_key in fresh_runner_keys
        if runner_key in primary_runner_keys.get(race_id, set())
    }
    stage2_runner_matches = {
        (race_id, runner_key)
        for race_id, runner_key in fresh_runner_keys
        if runner_key in stage2_runner_keys.get(race_id, set())
    }
    samples = []
    for item in sorted(
        fresh_by_race.values(),
        key=lambda row: (-int(row.get("fresh_current_win_rows") or 0), row["race_id"]),
    )[:sample_limit]:
        race_id = str(item["race_id"])
        samples.append(
            {
                **item,
                "fresh_runner_key_count": len(item.get("fresh_runner_keys") or []),
                "primary_runner_key_match_count": len(
                    set(item.get("fresh_runner_keys") or [])
                    & primary_runner_keys.get(race_id, set())
                ),
                "stage2_runner_key_match_count": len(
                    set(item.get("fresh_runner_keys") or [])
                    & stage2_runner_keys.get(race_id, set())
                ),
                "primary_prediction_artifact_count": len(
                    primary_artifacts.get(race_id) or []
                ),
                "stage2_prediction_artifact_count": len(
                    stage2_artifacts.get(race_id) or []
                ),
                "latest_primary_prediction_artifact": (
                    primary_artifacts.get(race_id) or [None]
                )[-1],
                "latest_stage2_prediction_artifact": (
                    stage2_artifacts.get(race_id) or [None]
                )[-1],
            }
        )
    result.update(
        {
            "status": "SUCCESS" if race_ids else "NO_FRESH_CURRENT_WIN_ODDS",
            "fresh_current_win_odds_rows": int(fresh_audit.get("fresh_rows") or 0),
            "fresh_current_win_odds_races": len(race_ids),
            "stale_current_win_odds_rows": int(fresh_audit.get("stale_rows") or 0),
            "fresh_odds_audit": fresh_audit,
            "shadow_runs_checked": len(runs),
            "primary_prediction_rows_checked": prediction_rows_checked,
            "stage2_prediction_rows_checked": stage2_rows_checked,
            "latest_shadow_run_checked": latest_run,
            "races_with_primary_predictions": len(races_with_primary),
            "races_with_stage2_predictions": len(races_with_stage2),
            "races_missing_prediction_artifact": len(races_missing_any),
            "missing_prediction_race_ids_sample": sorted(races_missing_any)[:sample_limit],
            "fresh_odds_runner_keys": len(fresh_runner_keys),
            "fresh_odds_runner_keys_with_primary_prediction_match": len(
                primary_runner_matches
            ),
            "fresh_odds_runner_keys_with_stage2_prediction_match": len(
                stage2_runner_matches
            ),
            "fresh_odds_runner_keys_missing_primary_prediction_match": max(
                len(fresh_runner_keys) - len(primary_runner_matches),
                0,
            ),
            "fresh_odds_runner_keys_missing_stage2_prediction_match": max(
                len(fresh_runner_keys) - len(stage2_runner_matches),
                0,
            ),
            "race_samples": samples,
        }
    )
    return result


def _daemon_norm_dog(value: Any) -> str:
    raw = str(value or "").strip()
    raw = re.sub(r"^\s*\d{1,2}\s*[\.\):-]\s*", "", raw)
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


def fresh_strict_odds_identity_split(
    *,
    db_path: Path,
    generated_at: datetime,
    stale_after_hours: float = 6.0,
) -> dict[str, Any]:
    """Read-only split of strict odds identity matches into fresh vs stale rows."""

    result: dict[str, Any] = {
        "schema_version": "shadow_daemon_fresh_strict_odds_identity_split_v1",
        "generated_at": generated_at.isoformat(),
        "status": "DATA_MISSING",
        "mode": "read_only_fresh_stale_strict_identity_diagnostic",
        "stale_after_hours": stale_after_hours,
        "db_write": False,
        "odds_capture_performed": False,
    }
    try:
        conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.create_function("daemon_norm_dog", 1, _daemon_norm_dog)
        try:
            tables = {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if not {"live_odds", "dog_race_data"}.issubset(tables):
                result["reason"] = "required_tables_missing"
                result["tables"] = sorted(tables)
                return result
            live_cols = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(live_odds)").fetchall()
            }
            timestamp_expr = (
                "COALESCE(lo.capture_timestamp, lo.timestamp)"
                if "capture_timestamp" in live_cols
                else "lo.timestamp"
            )
            current_clause = (
                "AND (lo.is_current = 1 OR lo.is_current IS NULL)"
                if "is_current" in live_cols
                else ""
            )
            rows = conn.execute(
                f"""
                WITH win AS (
                    SELECT
                        lo.id,
                        {timestamp_expr} AS odds_timestamp,
                        lo.race_id,
                        lo.box_number,
                        daemon_norm_dog(COALESCE(lo.dog_clean_name, lo.dog_name)) AS dog_key
                    FROM live_odds lo
                    WHERE lower(COALESCE(lo.market_type, 'win')) = 'win'
                      AND lo.odds_decimal IS NOT NULL
                      AND lo.odds_decimal > 1
                      {current_clause}
                      AND lo.box_number IS NOT NULL
                      AND TRIM(COALESCE(lo.dog_clean_name, lo.dog_name, '')) != ''
                )
                SELECT
                    win.id,
                    win.odds_timestamp,
                    CASE
                        WHEN COUNT(DISTINCT f.id) = 1 THEN 1
                        ELSE 0
                    END AS strict_identity_match
                FROM win
                LEFT JOIN dog_race_data f
                  ON f.race_id = win.race_id
                 AND f.box_number = win.box_number
                 AND daemon_norm_dog(COALESCE(f.dog_clean_name, f.dog_name)) = win.dog_key
                GROUP BY win.id, win.odds_timestamp
                """
            ).fetchall()
        finally:
            conn.close()
    except Exception as exc:
        result["status"] = "FAILED"
        result["error"] = f"{type(exc).__name__}:{exc}"
        return result

    rows_checked = len(rows)
    fresh_rows = 0
    stale_rows = 0
    null_timestamp_rows = 0
    invalid_timestamp_rows = 0
    fresh_strict = 0
    stale_strict = 0
    for row in rows:
        timestamp = parse_datetime_value(
            row["odds_timestamp"],
            default_tz=generated_at.tzinfo,
        )
        strict_match = int(row["strict_identity_match"] or 0) == 1
        if timestamp is None:
            if row["odds_timestamp"] in (None, ""):
                null_timestamp_rows += 1
            else:
                invalid_timestamp_rows += 1
            continue
        compare_generated_at = generated_at
        compare_timestamp = timestamp
        if compare_timestamp.tzinfo is not None and compare_generated_at.tzinfo is None:
            compare_generated_at = compare_generated_at.replace(
                tzinfo=compare_timestamp.tzinfo
            )
        elif compare_timestamp.tzinfo is None and compare_generated_at.tzinfo is not None:
            compare_generated_at = compare_generated_at.replace(tzinfo=None)
        age_hours = (compare_generated_at - compare_timestamp).total_seconds() / 3600.0
        if age_hours > stale_after_hours:
            stale_rows += 1
            if strict_match:
                stale_strict += 1
        else:
            fresh_rows += 1
            if strict_match:
                fresh_strict += 1

    result.update(
        {
            "status": "SUCCESS" if rows_checked else "NO_CURRENT_DOG_LEVEL_WIN_ODDS",
            "rows_checked": rows_checked,
            "fresh_current_win_rows": fresh_rows,
            "stale_current_win_rows": stale_rows,
            "null_timestamp_rows": null_timestamp_rows,
            "invalid_timestamp_rows": invalid_timestamp_rows,
            "fresh_safe_direct_identity_matches": fresh_strict,
            "stale_safe_direct_identity_matches": stale_strict,
            "fresh_unmatched_rows": max(fresh_rows - fresh_strict, 0),
            "stale_unmatched_rows": max(stale_rows - stale_strict, 0),
            "fresh_safe_direct_identity_match_rate": (
                fresh_strict / fresh_rows if fresh_rows else None
            ),
            "stale_safe_direct_identity_match_rate": (
                stale_strict / stale_rows if stale_rows else None
            ),
        }
    )
    return result


def build_cycle_activity_summary(
    *,
    current_dashboard: Mapping[str, Any],
    previous_dashboard: Mapping[str, Any] | None,
    daily_status: Mapping[str, Any],
    observability_status: Mapping[str, Any],
) -> dict[str, Any]:
    previous_dashboard = previous_dashboard or {}
    current_joined = int(current_dashboard.get("safe_joined_races") or 0)
    previous_joined = int(previous_dashboard.get("safe_joined_races") or 0)
    safe_joined_delta = max(0, current_joined - previous_joined)
    prediction_rows = int(observability_status.get("prediction_rows") or 0)
    races_scored = int(daily_status.get("races_scored_today") or 0)
    if prediction_rows > 0 and safe_joined_delta > 0:
        status = "PREDICTIONS_AND_RESULT_JOINS_ADVANCED"
    elif prediction_rows > 0:
        status = "PREDICTIONS_ONLY"
    elif safe_joined_delta > 0:
        status = "RESULT_JOINS_ADVANCED_NO_NEW_PREDICTIONS"
    else:
        status = "NO_NEW_PREDICTIONS_OR_SAFE_JOINS"
    return {
        "schema_version": "shadow_daemon_cycle_activity_summary_v1",
        "status": status,
        "current_safe_joined_races": current_joined,
        "previous_safe_joined_races": previous_joined,
        "safe_joined_delta_this_cycle": safe_joined_delta,
        "prediction_rows_this_cycle": prediction_rows,
        "races_scored_this_cycle": races_scored,
        "observability_status": observability_status.get("status"),
    }


def first_json(paths: Sequence[Path | None]) -> tuple[dict[str, Any] | None, Path | None]:
    for path in paths:
        payload = load_json(path)
        if payload is not None:
            return payload, path
    return None, None


def first_json_list(paths: Sequence[Path | None]) -> tuple[list[dict[str, Any]], Path | None]:
    for path in paths:
        payload = load_json_value(path)
        if not isinstance(payload, list):
            continue
        rows = [dict(row) for row in payload if isinstance(row, Mapping)]
        return rows, path
    return [], None


def clean_key(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def prediction_rows_from_daily_run(daily_shadow_run_dir: Path | None) -> tuple[list[dict[str, Any]], Path | None]:
    if daily_shadow_run_dir is None:
        return [], None
    jsonl_path = daily_shadow_run_dir / "shadow_predictions.jsonl"
    rows = read_jsonl_rows(jsonl_path)
    if rows:
        return rows, jsonl_path
    rows, path = first_json_list(
        [
            daily_shadow_run_dir / "shadow_predictions.json",
            daily_shadow_run_dir / "shadow_score_live" / "shadow_predictions.json",
        ]
    )
    if rows:
        return rows, path
    return [], jsonl_path if jsonl_path.exists() else None


def feature_rows_from_daily_run(daily_shadow_run_dir: Path | None) -> tuple[list[dict[str, Any]], Path | None]:
    if daily_shadow_run_dir is None:
        return [], None
    return first_json_list(
        [
            daily_shadow_run_dir / "shadow_score_live" / "shadow_feature_rows.json",
            daily_shadow_run_dir / "shadow_feature_rows.json",
        ]
    )


def infer_no_prediction_reason(daily_manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    manifest = daily_manifest or {}
    input_summary = manifest.get("input_summary") or {}
    final_status = manifest.get("final_status")
    if final_status == "WAITING_FOR_UPCOMING_RACES":
        reason = "no_eligible_current_or_future_races"
    elif final_status == "BLOCKED_DB_STATE":
        reason = "db_state_blocked_shadow_scoring"
    elif final_status == "BLOCKED_MALFORMED_INPUTS":
        reason = "malformed_inputs_blocked_shadow_scoring"
    elif final_status == "BLOCKED_SHADOW_RUN_FAILURE":
        reason = "shadow_scoring_failed"
    elif int(input_summary.get("eligible_count") or 0) <= 0:
        reason = "no_eligible_inputs"
    else:
        reason = "prediction_rows_zero"
    return {
        "reason": reason,
        "final_status": final_status,
        "input_summary": input_summary,
    }


def find_importance_estimator(model: Any, seen: set[int] | None = None) -> Any:
    seen = seen or set()
    object_id = id(model)
    if object_id in seen:
        return None
    seen.add(object_id)
    if hasattr(model, "feature_importances_"):
        return model
    for attr in ("named_steps", "steps"):
        value = getattr(model, attr, None)
        if isinstance(value, Mapping):
            for item in value.values():
                found = find_importance_estimator(item, seen)
                if found is not None:
                    return found
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                candidate = item[-1] if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) else item
                found = find_importance_estimator(candidate, seen)
                if found is not None:
                    return found
    for attr in ("estimator", "classifier", "model", "base_estimator"):
        child = getattr(model, attr, None)
        if child is not None:
            found = find_importance_estimator(child, seen)
            if found is not None:
                return found
    return None


def transformed_feature_names(model: Any, active_features: Sequence[str], expected_count: int) -> list[str] | None:
    preprocess = None
    named_steps = getattr(model, "named_steps", None)
    if isinstance(named_steps, Mapping):
        preprocess = named_steps.get("preprocess") or named_steps.get("preprocessor")
    if preprocess is not None and hasattr(preprocess, "get_feature_names_out"):
        try:
            names = [str(name) for name in preprocess.get_feature_names_out()]
        except Exception:
            names = []
        if len(names) == expected_count:
            return names
    if len(active_features) == expected_count:
        return list(active_features)
    return None


def extract_global_feature_importances(
    *,
    model_path: Path | None,
    active_features: Sequence[str],
    limit: int = 20,
) -> dict[str, Any]:
    if model_path is None:
        return {"status": "UNAVAILABLE", "reason": "model_path_missing", "top_features": []}
    if not model_path.exists():
        return {"status": "UNAVAILABLE", "reason": "model_file_missing", "top_features": []}
    try:
        import joblib  # type: ignore
    except Exception as exc:
        return {
            "status": "UNAVAILABLE",
            "reason": "joblib_not_available_in_daemon_runtime",
            "error": repr(exc),
            "top_features": [],
        }
    try:
        model = joblib.load(model_path)
        estimator = find_importance_estimator(model)
        if estimator is None:
            return {"status": "UNAVAILABLE", "reason": "feature_importances_not_exposed", "top_features": []}
        importances = [float(value) for value in getattr(estimator, "feature_importances_")]
        names = transformed_feature_names(model, active_features, len(importances))
        if names is None:
            return {
                "status": "UNAVAILABLE",
                "reason": "feature_names_not_recoverable",
                "importance_count": len(importances),
                "active_feature_count": len(active_features),
                "top_features": [],
            }
        rows = sorted(
            (
                {"feature": feature, "importance": importance}
                for feature, importance in zip(names, importances)
            ),
            key=lambda row: row["importance"],
            reverse=True,
        )
        return {"status": "AVAILABLE", "top_features": rows[:limit], "feature_count": len(rows)}
    except Exception as exc:
        return {"status": "UNAVAILABLE", "reason": "model_importance_read_failed", "error": repr(exc), "top_features": []}


def feature_policy_summary(
    *,
    model_metadata: Mapping[str, Any] | None,
    training_report: Mapping[str, Any] | None,
    active_policy: Mapping[str, Any] | None,
    inactive_policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    inactive_features = (
        (active_policy or {}).get("inactive_features_due_to_train_all_missing")
        or (inactive_policy or {}).get("inactive_features_due_to_train_all_missing")
        or (training_report or {}).get("inactive_features_due_to_train_all_missing")
        or (model_metadata or {}).get("inactive_features_due_to_train_all_missing")
        or []
    )
    inactive_set = {str(feature) for feature in inactive_features}
    feature_columns = list((model_metadata or {}).get("feature_columns") or [])
    active_features = [str(feature) for feature in feature_columns if str(feature) not in inactive_set]
    watched = list(getattr(autopilot, "WATCHED_QUARANTINED_FEATURES", ()))
    quarantined_active = [feature for feature in watched if feature in active_features]
    return {
        "schema_version": "shadow_observability_feature_policy_summary_v1",
        "feature_columns_status": "AVAILABLE" if feature_columns else "UNAVAILABLE",
        "feature_columns": feature_columns,
        "active_feature_names": active_features,
        "active_feature_count": len(active_features) or (active_policy or {}).get("active_feature_count"),
        "schema_feature_count": (active_policy or {}).get("schema_feature_count")
        or (training_report or {}).get("schema_feature_count")
        or (model_metadata or {}).get("feature_count"),
        "inactive_features_due_to_train_all_missing": list(inactive_features),
        "watched_quarantined_features": watched,
        "quarantined_features_active": quarantined_active,
        "status": "FAIL" if quarantined_active else "PASS",
    }


def runner_feature_audit(
    *,
    feature_row: Mapping[str, Any] | None,
    active_features: Sequence[str],
    top_features: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not feature_row or not active_features:
        return {
            "feature_value_status": "UNAVAILABLE",
            "active_feature_present_count": None,
            "active_feature_missing_count": None,
            "top_feature_values": [],
        }
    present_count = 0
    missing_count = 0
    for feature in active_features:
        value = feature_row.get(feature)
        if value in (None, ""):
            missing_count += 1
        else:
            present_count += 1
    top_values = []
    for row in top_features[:10]:
        feature = str(row.get("feature") or "")
        top_values.append(
            {
                "feature": feature,
                "importance": row.get("importance"),
                "value": feature_row.get(feature),
                "value_present": feature_row.get(feature) not in (None, ""),
            }
        )
    return {
        "feature_value_status": "AVAILABLE",
        "active_feature_present_count": present_count,
        "active_feature_missing_count": missing_count,
        "top_feature_values": top_values,
    }


def build_race_prediction_explanations(
    *,
    predictions: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    active_features: Sequence[str],
    feature_importances: Mapping[str, Any],
    probability_sum_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in predictions:
        grouped.setdefault(str(row.get("race_id") or ""), []).append(row)
    feature_index = {
        (clean_key(row.get("race_id")), clean_key(row.get("dog_name"))): row
        for row in feature_rows
    }
    probability_by_race = {
        str(row.get("race_id") or ""): row
        for row in (probability_sum_report or {}).get("per_race") or []
        if isinstance(row, Mapping)
    }
    top_features = list(feature_importances.get("top_features") or [])
    races = []
    for race_id, rows in sorted(grouped.items()):
        sorted_rows = sorted(rows, key=lambda row: int(row.get("predicted_rank") or 9999))
        top_pick = sorted_rows[0] if sorted_rows else None
        runner_explanations = []
        for row in sorted_rows:
            feature_row = feature_index.get((clean_key(row.get("race_id")), clean_key(row.get("dog_name"))))
            runner_explanations.append(
                {
                    "dog_name": row.get("dog_name"),
                    "box": row.get("box"),
                    "predicted_rank": row.get("predicted_rank"),
                    "shadow_rf_uncalibrated_probability": row.get("shadow_rf_uncalibrated_probability"),
                    "shadow_rf_calibrated_probability": row.get("shadow_rf_calibrated_probability"),
                    "feature_audit": runner_feature_audit(
                        feature_row=feature_row,
                        active_features=active_features,
                        top_features=top_features,
                    ),
                }
            )
        races.append(
            {
                "race_id": race_id,
                "runner_count": len(rows),
                "top_pick": None
                if top_pick is None
                else {
                    "dog_name": top_pick.get("dog_name"),
                    "box": top_pick.get("box"),
                    "shadow_rf_calibrated_probability": top_pick.get("shadow_rf_calibrated_probability"),
                },
                "top3": [
                    {
                        "dog_name": row.get("dog_name"),
                        "box": row.get("box"),
                        "predicted_rank": row.get("predicted_rank"),
                        "shadow_rf_calibrated_probability": row.get("shadow_rf_calibrated_probability"),
                    }
                    for row in sorted_rows[:3]
                ],
                "probability_sum": probability_by_race.get(race_id),
                "runner_explanations": runner_explanations,
            }
        )
    return {
        "schema_version": "shadow_race_prediction_explanations_v1",
        "explanation_type": "MODEL_INPUT_AUDIT",
        "causal_explanation": False,
        "feature_value_status": "AVAILABLE" if feature_rows and active_features else "UNAVAILABLE",
        "feature_importance_status": feature_importances.get("status"),
        "race_count": len(races),
        "prediction_rows": len(predictions),
        "races": races,
    }


def build_observability_status_markdown(
    *,
    status: Mapping[str, Any],
    model_card: Mapping[str, Any],
    provenance: Mapping[str, Any],
    explanations: Mapping[str, Any],
) -> str:
    races = explanations.get("races") or []
    top_lines = []
    for race in races[:5]:
        top_pick = race.get("top_pick") or {}
        top_lines.append(
            f"- `{race.get('race_id')}` top pick `{top_pick.get('dog_name')}` "
            f"box `{top_pick.get('box')}` p=`{top_pick.get('shadow_rf_calibrated_probability')}`"
        )
    if not top_lines:
        top_lines.append("- No scored races in this daemon packet.")
    return "\n".join(
        [
            "# Shadow Observability Status",
            "",
            f"- Status: `{status.get('status')}`",
            f"- Prediction rows: `{status.get('prediction_rows')}`",
            f"- Race count: `{status.get('race_count')}`",
            f"- No-prediction reason: `{status.get('no_prediction_reason')}`",
            f"- Model source: `{model_card.get('model_source')}`",
            f"- Model sha256: `{model_card.get('model_sha256')}`",
            f"- Calibration: `{provenance.get('calibration_method')}`",
            f"- Training disabled: `{status.get('safety_flags', {}).get('training_disabled')}`",
            f"- TGR disabled: `{status.get('safety_flags', {}).get('tgr_disabled')}`",
            f"- Feature policy: `{model_card.get('feature_policy', {}).get('status')}`",
            f"- Probability sum: `{status.get('probability_sum_status')}`",
            f"- Feature importances: `{model_card.get('global_feature_importances', {}).get('status')}`",
            "",
            "## Top Races",
            *top_lines,
            "",
            "This report is a model input/provenance audit, not a causal explanation and not promotion approval.",
            "",
        ]
    )


def build_shadow_observability(
    *,
    generated_at: datetime,
    run_id: str,
    daily_shadow_run_dir: Path | None,
    daily_manifest: Mapping[str, Any] | None,
    dashboard: Mapping[str, Any],
    readiness: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    protected_validation: Mapping[str, Any],
) -> dict[str, Any]:
    daily_manifest = daily_manifest or {}
    score_dir = daily_shadow_run_dir / "shadow_score_live" if daily_shadow_run_dir else None
    score_manifest = daily_manifest.get("score_live_manifest")
    if not isinstance(score_manifest, Mapping):
        score_manifest = load_json(score_dir / "shadow_manifest.json") if score_dir else None
    score_command = load_json(daily_shadow_run_dir / "shadow_score_live_command.json") if daily_shadow_run_dir else None
    active_policy, active_policy_path = first_json(
        [
            score_dir / "active_feature_policy_report.json" if score_dir else None,
            daily_shadow_run_dir / "active_feature_policy_report.json" if daily_shadow_run_dir else None,
        ]
    )
    predictions, predictions_path = prediction_rows_from_daily_run(daily_shadow_run_dir)
    feature_rows, feature_rows_path = feature_rows_from_daily_run(daily_shadow_run_dir)
    probability_sum = load_json(daily_shadow_run_dir / "probability_sum_report.json") if daily_shadow_run_dir else None

    first_prediction = predictions[0] if predictions else {}
    model_source = (
        (score_manifest or {}).get("model_source")
        or daily_manifest.get("shadow_model")
        or first_prediction.get("model_source")
        or (active_policy or {}).get("model_path")
    )
    model_path = rooted_path(model_source)
    model_dir = model_path.parent if model_path else None
    inactive_policy, inactive_policy_path = first_json(
        [
            score_dir / "inactive_feature_policy_report.json" if score_dir else None,
            daily_shadow_run_dir / "inactive_feature_policy_report.json" if daily_shadow_run_dir else None,
            model_dir / "inactive_feature_policy_report.json" if model_dir else None,
        ]
    )
    model_metadata = load_json(model_dir / "shadow_model_metadata.json") if model_dir else None
    training_report = load_json(model_dir / "shadow_training_report.json") if model_dir else None
    candidate_definition, candidate_definition_path = first_json(
        [
            score_dir / "shadow_candidate_definition.json" if score_dir else None,
            model_dir / "shadow_candidate_definition.json" if model_dir else None,
        ]
    )
    feature_policy = feature_policy_summary(
        model_metadata=model_metadata,
        training_report=training_report,
        active_policy=active_policy,
        inactive_policy=inactive_policy,
    )
    active_features = list(feature_policy.get("active_feature_names") or [])
    global_importances = extract_global_feature_importances(
        model_path=model_path,
        active_features=active_features,
    )
    explanations = build_race_prediction_explanations(
        predictions=predictions,
        feature_rows=feature_rows,
        active_features=active_features,
        feature_importances=global_importances,
        probability_sum_report=probability_sum,
    )
    command = list((score_command or {}).get("command") or [])
    command_text = " ".join(str(part) for part in command)
    score_command_uses_model = "--model" in command
    score_command_trains = "--train-if-missing" in command
    training_disabled = daily_manifest.get("shadow_training_allowed") is False and not score_command_trains
    tgr_enabled = bool(
        daily_manifest.get("tgr_enabled")
        or (score_manifest or {}).get("tgr_enabled")
        or any(row.get("tgr_enabled") for row in predictions)
    )
    safety_flags = {
        "training_disabled": training_disabled,
        "score_command_uses_locked_model": score_command_uses_model or bool(model_source),
        "score_command_trains": score_command_trains,
        "tgr_disabled": not tgr_enabled,
        "registry_mutation": bool(daily_manifest.get("registry_mutation") or (score_manifest or {}).get("registry_mutation")),
        "production_prediction_overwrite": bool(
            daily_manifest.get("production_prediction_overwrite")
            or (score_manifest or {}).get("production_prediction_write")
        ),
        "db_writes": bool(daily_manifest.get("db_writes")),
        "label_writes": bool(daily_manifest.get("label_writes")),
        "protected_paths_unchanged": bool(protected_validation.get("protected_paths_unchanged")),
        "quarantined_features_active": feature_policy.get("quarantined_features_active") or [],
    }
    unsafe_safety = (
        not safety_flags["training_disabled"]
        or not safety_flags["tgr_disabled"]
        or safety_flags["registry_mutation"]
        or safety_flags["production_prediction_overwrite"]
        or safety_flags["db_writes"]
        or safety_flags["label_writes"]
    )
    probability_status = (probability_sum or {}).get("status")
    no_prediction = infer_no_prediction_reason(daily_manifest) if not predictions else None
    if feature_policy.get("quarantined_features_active"):
        status_value = "FAIL_CLOSED_FEATURE_POLICY_VIOLATION"
    elif probability_status and probability_status != "PASS":
        status_value = "PROBABILITY_SUM_FAIL"
    elif unsafe_safety:
        status_value = "SAFETY_FLAG_FAIL"
    elif not predictions:
        status_value = "NO_PREDICTIONS"
    else:
        status_value = "OBSERVABILITY_READY"

    model_card = {
        "schema_version": "shadow_model_provenance_card_v1",
        "generated_at": generated_at.isoformat(),
        "model_source": model_source,
        "model_path": relpath(model_path),
        "model_exists": bool(model_path and model_path.exists()),
        "model_sha256": sha256_file(model_path) if model_path else None,
        "model_family": (training_report or {}).get("model_family")
        or (candidate_definition or {}).get("model_family")
        or "RandomForest",
        "calibration_method": (score_manifest or {}).get("calibration_method")
        or daily_manifest.get("calibration_method")
        or (candidate_definition or {}).get("calibration", {}).get("method_key"),
        "training_report_path": relpath(model_dir / "shadow_training_report.json") if model_dir and (model_dir / "shadow_training_report.json").exists() else None,
        "model_metadata_path": relpath(model_dir / "shadow_model_metadata.json") if model_dir and (model_dir / "shadow_model_metadata.json").exists() else None,
        "candidate_definition_path": relpath(candidate_definition_path),
        "active_feature_policy_path": relpath(active_policy_path),
        "inactive_feature_policy_path": relpath(inactive_policy_path),
        "train_races": (training_report or {}).get("train_races"),
        "train_rows": (training_report or {}).get("train_rows"),
        "holdout_races": (training_report or {}).get("holdout_races"),
        "holdout_rows": (training_report or {}).get("holdout_rows"),
        "feature_policy": feature_policy,
        "global_feature_importances": global_importances,
    }
    provenance = {
        "schema_version": "shadow_prediction_provenance_report_v1",
        "generated_at": generated_at.isoformat(),
        "run_id": run_id,
        "daily_shadow_run_dir": relpath(daily_shadow_run_dir),
        "predictions_path": relpath(predictions_path),
        "feature_rows_path": relpath(feature_rows_path),
        "score_live_dir": relpath(score_dir),
        "score_command_path": relpath(daily_shadow_run_dir / "shadow_score_live_command.json") if daily_shadow_run_dir else None,
        "score_command": command,
        "score_command_text": command_text,
        "score_command_uses_model": score_command_uses_model,
        "score_command_trains": score_command_trains,
        "model_source": model_source,
        "model_sha256": model_card["model_sha256"],
        "model_version": (score_manifest or {}).get("model_version") or first_prediction.get("model_version"),
        "calibration_method": model_card["calibration_method"],
        "calibration_formula": ((candidate_definition or {}).get("calibration") or {}).get("formula"),
        "prediction_rows": len(predictions),
        "race_count": len({row.get("race_id") for row in predictions if row.get("race_id")}),
        "probability_sum_status": probability_status,
        "probability_sum_report": relpath(daily_shadow_run_dir / "probability_sum_report.json") if daily_shadow_run_dir else None,
        "safety_flags": safety_flags,
        "dashboard_decision": readiness.get("decision"),
        "protected_path_validation": protected_validation,
    }
    status = {
        "schema_version": "shadow_observability_status_v1",
        "generated_at": generated_at.isoformat(),
        "run_id": run_id,
        "status": status_value,
        "prediction_rows": len(predictions),
        "race_count": provenance["race_count"],
        "daily_shadow_run_dir": relpath(daily_shadow_run_dir),
        "no_prediction_reason": None if no_prediction is None else no_prediction.get("reason"),
        "no_prediction_details": no_prediction,
        "model_source": model_source,
        "model_sha256": model_card["model_sha256"],
        "score_command_text": command_text,
        "probability_sum_status": probability_status,
        "feature_policy_status": feature_policy.get("status"),
        "feature_importance_status": global_importances.get("status"),
        "safety_flags": safety_flags,
        "dashboard_summary": {
            "safe_joined_races": dashboard.get("safe_joined_races"),
            "pending_races": dashboard.get("pending_races"),
            "unsafe_matches": dashboard.get("unsafe_matches"),
        },
    }
    event_log = [
        {
            "event": "observability_started",
            "run_id": run_id,
            "generated_at": generated_at.isoformat(),
            "daily_shadow_run_dir": relpath(daily_shadow_run_dir),
        },
        {
            "event": "prediction_artifacts_loaded",
            "prediction_rows": len(predictions),
            "feature_rows": len(feature_rows),
            "predictions_path": relpath(predictions_path),
            "feature_rows_path": relpath(feature_rows_path),
        },
        {
            "event": "model_provenance_loaded",
            "model_source": model_source,
            "model_sha256": model_card["model_sha256"],
            "feature_importance_status": global_importances.get("status"),
        },
        {
            "event": "safety_flags_evaluated",
            "status": status_value,
            "safety_flags": safety_flags,
        },
    ]
    for step in steps:
        event_log.append(
            {
                "event": "daemon_step",
                "name": step.get("name"),
                "status": step.get("status"),
                "returncode": step.get("returncode"),
                "timed_out": step.get("timed_out"),
                "duration_seconds": step.get("duration_seconds"),
                "stdout_path": step.get("stdout_path"),
                "stderr_path": step.get("stderr_path"),
            }
        )
    markdown = build_observability_status_markdown(
        status=status,
        model_card=model_card,
        provenance=provenance,
        explanations=explanations,
    )
    return {
        "status": status,
        "model_card": model_card,
        "provenance": provenance,
        "race_explanations": explanations,
        "event_log": event_log,
        "markdown": markdown,
    }


def output_manifest(output_dir: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        files[relpath(path) or str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "schema_version": "shadow_autopilot_daemon_output_manifest_v1",
        "output_dir": relpath(output_dir),
        "files": files,
    }


def daemon_runtime_timer_snapshot(systemd_deployment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "service_status": "cycle_finalizing",
        "timer_status": "active" if systemd_deployment.get("timer_active") else "inactive",
        "deployment_status": systemd_deployment.get("deployment_status"),
        "deployment_ready": systemd_deployment.get("deployment_ready"),
        "timer_enabled": systemd_deployment.get("timer_enabled"),
    }


def write_daemon_runtime_state_packet(
    *,
    output_dir: Path,
    state_path: Path,
    systemd_deployment: Mapping[str, Any],
    target_joined_races: int,
    generated_at: datetime,
) -> dict[str, Any]:
    report = build_forward_shadow_runtime_state(
        evidence_root=DEFAULT_EVIDENCE_ROOT,
        daemon_state_path=state_path,
        timer=daemon_runtime_timer_snapshot(systemd_deployment),
        target_joined_races=target_joined_races,
        generated_at=generated_at,
    )
    write_json(output_dir / "forward_shadow_runtime_state.json", report)
    write_text(
        output_dir / "FORWARD_SHADOW_RUNTIME_STATE.md",
        build_forward_shadow_runtime_summary(report),
    )
    return report


def final_verdict(
    *,
    protected_paths_unchanged: bool,
    required_outputs_present: bool,
    service_files_present: bool,
    lock_ok: bool,
    operational_ok: bool,
    service_installed: bool,
) -> str:
    if not protected_paths_unchanged or not required_outputs_present:
        return "NEEDS_MORE_AUTOMATION"
    if not lock_ok or not operational_ok:
        return "PARTIAL_DAEMONIZATION"
    if not service_files_present:
        return "PARTIAL_DAEMONIZATION"
    if service_installed:
        return "DAEMON_READY"
    return "DAEMON_READY_NEEDS_DEPLOYMENT"


def build_final_summary(
    *,
    verdict: str,
    dashboard: Mapping[str, Any],
    readiness: Mapping[str, Any],
    automated_join_report: Mapping[str, Any],
    alert_report: Mapping[str, Any],
    service_validation: Mapping[str, Any],
    feature_activation_gate: Mapping[str, Any] | None = None,
    odds_coverage: Mapping[str, Any] | None = None,
    shadow_odds_snapshot: Mapping[str, Any] | None = None,
    observability_status: Mapping[str, Any] | None = None,
    cycle_activity: Mapping[str, Any] | None = None,
    next_prejump_refresh_window: Mapping[str, Any] | None = None,
    prejump_metadata_status: Mapping[str, Any] | None = None,
    prejump_metadata_trend: Mapping[str, Any] | None = None,
    live_odds_capture_packet: Mapping[str, Any] | None = None,
    autopilot_cycle_daily_status: Mapping[str, Any] | None = None,
    rejoin_unified_evidence_status: Mapping[str, Any] | None = None,
    rejoin_rolling_model_comparison_status: Mapping[str, Any] | None = None,
    rejoin_high_accuracy_refinement_status: Mapping[str, Any] | None = None,
    rejoin_pre_race_gated_challenger_status: Mapping[str, Any] | None = None,
    rejoin_rank_first_hypothesis_gated_status: Mapping[str, Any] | None = None,
    rejoin_time_split_gated_challenger_status: Mapping[str, Any] | None = None,
    rejoin_market_residual_challenger_status: Mapping[str, Any] | None = None,
    rejoin_market_residual_regime_audit_status: Mapping[str, Any] | None = None,
    rejoin_rank_first_hypothesis_watchlist_status: Mapping[str, Any] | None = None,
    rejoin_promotion_distance_status: Mapping[str, Any] | None = None,
) -> str:
    feature_activation_gate = feature_activation_gate or {}
    odds_coverage = odds_coverage or {}
    shadow_odds_snapshot = shadow_odds_snapshot or {}
    observability_status = observability_status or {}
    cycle_activity = cycle_activity or {}
    next_prejump_refresh_window = next_prejump_refresh_window or {}
    next_prejump_race = next_prejump_refresh_window.get("next_race") or {}
    prejump_metadata_status = prejump_metadata_status or {}
    prejump_metadata_trend = prejump_metadata_trend or {}
    live_odds_capture_packet = live_odds_capture_packet or {}
    autopilot_cycle_daily_status = autopilot_cycle_daily_status or {}
    rejoin_unified_evidence_status = rejoin_unified_evidence_status or {}
    rejoin_rolling_model_comparison_status = rejoin_rolling_model_comparison_status or {}
    rejoin_high_accuracy_refinement_status = rejoin_high_accuracy_refinement_status or {}
    rejoin_pre_race_gated_challenger_status = (
        rejoin_pre_race_gated_challenger_status or {}
    )
    rejoin_rank_first_hypothesis_gated_status = (
        rejoin_rank_first_hypothesis_gated_status or {}
    )
    rejoin_time_split_gated_challenger_status = (
        rejoin_time_split_gated_challenger_status or {}
    )
    rejoin_market_residual_challenger_status = (
        rejoin_market_residual_challenger_status or {}
    )
    rejoin_market_residual_regime_audit_status = (
        rejoin_market_residual_regime_audit_status or {}
    )
    rejoin_rank_first_hypothesis_watchlist_status = (
        rejoin_rank_first_hypothesis_watchlist_status or {}
    )
    rejoin_promotion_distance_status = rejoin_promotion_distance_status or {}
    activation_data = feature_activation_gate.get("data_availability_status") or {}
    activation_fail_summary = activation_data.get("fail_reason_summary") or {}
    same_distance_history = activation_data.get("same_distance_history") or {}
    live_odds_backlog = live_odds_backlog_operational_fields(dashboard)
    return "\n".join(
        [
            "# Shadow Autopilot Daemonization V1",
            "",
            f"Final verdict: `{verdict}`",
            "",
            f"- Safe joined races: `{dashboard.get('safe_joined_races')}`",
            f"- Pending races: `{dashboard.get('pending_races')}`",
            f"- Unsafe matches: `{dashboard.get('unsafe_matches')}`",
            f"- Top1: `{dashboard.get('top1')}`",
            f"- Top3: `{dashboard.get('top3')}`",
            f"- Brier: `{dashboard.get('brier')}`",
            f"- LogLoss: `{dashboard.get('logloss')}`",
            f"- Calibration: `{dashboard.get('calibration')}`",
            f"- Box1 share: `{dashboard.get('box_1_share')}`",
            f"- Probability sum: `{dashboard.get('probability_sum_status')}`",
            f"- Feature activation gate: `{feature_activation_gate.get('status')}`",
            f"- Kept quarantined features: `{feature_activation_gate.get('kept_quarantined_features') or []}`",
            f"- Feature data availability: `{activation_data.get('status')}`",
            f"- Feature blocker counts: `{activation_fail_summary.get('reason_counts')}`",
            f"- Same-distance history status: `{same_distance_history.get('status')}`",
            f"- Same-distance feature rows: `{same_distance_history.get('feature_rows')}`",
            f"- Odds coverage diagnostic: `{odds_coverage.get('status')}`",
            f"- Shadow odds snapshot: `{shadow_odds_snapshot.get('status')}`",
            f"- Shadow odds snapshot valid rows: `{shadow_odds_snapshot.get('valid_pre_jump_dog_odds_rows')}`",
            f"- Shadow odds complete valid races: `{shadow_odds_snapshot.get('races_with_complete_valid_prejump_odds')}`",
            f"- Shadow odds races with missing rows: `{shadow_odds_snapshot.get('races_with_missing_odds_rows')}`",
            f"- Shadow odds races after feature freeze: `{shadow_odds_snapshot.get('races_with_post_feature_freeze_odds_rows')}`",
            f"- Odds research gate: `{shadow_odds_snapshot.get('odds_research_gate_status')}`",
            f"- Odds research gate complete valid races: `{shadow_odds_snapshot.get('odds_research_gate_complete_valid_prejump_odds_races')}`",
            f"- Odds research next action: `{shadow_odds_snapshot.get('odds_research_next_action')}`",
            f"- Timing-aligned prediction rerun required: `{shadow_odds_snapshot.get('timing_aligned_prediction_rerun_required')}`",
            f"- Timing-aligned prediction rerun races: `{shadow_odds_snapshot.get('timing_aligned_prediction_rerun_race_count')}`",
            f"- Timing-aligned prediction rerun race IDs: `{shadow_odds_snapshot.get('timing_aligned_prediction_rerun_race_ids')}`",
            f"- Shadow odds EV output rows: `{shadow_odds_snapshot.get('ev_output_rows')}`",
            f"- Odds used for shadow scoring: `{odds_coverage.get('odds_used_for_shadow_scoring')}`",
            f"- Live odds capture approval: `{live_odds_capture_packet.get('status')}`",
            f"- Live odds verified races: `{live_odds_capture_packet.get('verified_prejump_race_count')}`",
            f"- Live odds capture windows: `{live_odds_capture_packet.get('capture_window_offsets_minutes')}`",
            f"- Live odds can capture now: `{live_odds_capture_packet.get('can_capture_live_odds_now')}`",
            f"- Autonomous odds next action: `{dashboard.get('odds_capture_next_meaningful_action')}` at `{dashboard.get('odds_capture_next_meaningful_action_at')}`",
            f"- Autonomous official result candidates: `{dashboard.get('autonomous_official_result_candidate_count')}`",
            f"- Autonomous official result race rows: `{dashboard.get('autonomous_official_result_race_rows')}`",
            f"- Autonomous official result runner rows: `{dashboard.get('autonomous_official_result_runner_rows')}`",
            f"- Autonomous official result quarantine rows: `{dashboard.get('autonomous_official_result_quarantine_rows')}`",
            f"- Autonomous official result evidence DB ingest: `{dashboard.get('autonomous_official_result_evidence_db_ingest_status')}`",
            f"- Autonomous official result evidence DB execute: `{dashboard.get('autonomous_official_result_evidence_db_execute')}`",
            f"- Autonomous official result evidence DB write performed: `{dashboard.get('autonomous_official_result_evidence_db_write_performed')}`",
            f"- Autonomous official result evidence valid race rows: `{dashboard.get('autonomous_official_result_evidence_valid_race_rows')}`",
            f"- Autonomous official result evidence valid runner rows: `{dashboard.get('autonomous_official_result_evidence_valid_runner_rows')}`",
            f"- Autonomous official result evidence blocked race rows: `{dashboard.get('autonomous_official_result_evidence_blocked_race_rows')}`",
            f"- Autonomous official result evidence blocked runner rows: `{dashboard.get('autonomous_official_result_evidence_blocked_runner_rows')}`",
            f"- Autonomous official result evidence inserted race rows: `{dashboard.get('autonomous_official_result_evidence_inserted_race_rows')}`",
            f"- Autonomous official result evidence inserted runner rows: `{dashboard.get('autonomous_official_result_evidence_inserted_runner_rows')}`",
            f"- Autonomous official result evidence blocker reasons: `{dashboard.get('autonomous_official_result_evidence_blocker_reason_counts')}`",
            f"- Autonomous official result quarantined race IDs: `{dashboard.get('autonomous_official_result_quarantined_race_ids')}`",
            f"- Autonomous official result quarantine reasons: `{dashboard.get('autonomous_official_result_quarantine_reason_counts')}`",
            f"- Autonomous official result quarantine errors: `{dashboard.get('autonomous_official_result_quarantine_error_counts')}`",
            f"- Autonomous official result missing result boxes: `{dashboard.get('autonomous_official_result_quarantine_result_boxes_not_in_participants_counts')}`",
            f"- Autonomous official result runner-set mismatch samples: `{dashboard.get('autonomous_official_result_quarantine_runner_set_mismatch_samples')}`",
            f"- Autopilot cycle unified evidence: `{autopilot_cycle_daily_status.get('unified_evidence_dataset_status')}`",
            f"- Autopilot cycle unified evidence rows: `{autopilot_cycle_daily_status.get('unified_evidence_dataset_rows')}`",
            f"- Autopilot cycle unified evidence races: `{autopilot_cycle_daily_status.get('unified_evidence_dataset_races')}`",
            f"- Autopilot cycle unified eligible rows: `{autopilot_cycle_daily_status.get('unified_evidence_eligible_rows')}`",
            f"- Best aggregate unified evidence path: `{autopilot_cycle_daily_status.get('best_aggregate_unified_evidence_status_path')}`",
            f"- Best aggregate unified evidence: `{autopilot_cycle_daily_status.get('best_aggregate_unified_evidence_status')}`",
            f"- Best aggregate unified dataset count: `{autopilot_cycle_daily_status.get('best_aggregate_unified_evidence_dataset_count')}`",
            f"- Best aggregate unified row count: `{autopilot_cycle_daily_status.get('best_aggregate_unified_evidence_row_count')}`",
            f"- Best aggregate unified eligible rows: `{autopilot_cycle_daily_status.get('best_aggregate_unified_evidence_eligible_rows')}`",
            f"- Best aggregate unified artifact odds rows seen: `{autopilot_cycle_daily_status.get('best_aggregate_unified_evidence_artifact_odds_rows_seen')}`",
            f"- Best aggregate unified artifact odds rows accepted: `{autopilot_cycle_daily_status.get('best_aggregate_unified_evidence_artifact_odds_rows_accepted')}`",
            f"- Best aggregate unified artifact odds rows rejected: `{autopilot_cycle_daily_status.get('best_aggregate_unified_evidence_artifact_odds_rows_rejected')}`",
            f"- Best aggregate unified artifact odds rejection reasons: `{autopilot_cycle_daily_status.get('best_aggregate_unified_evidence_artifact_odds_rejection_reason_counts')}`",
            f"- Best aggregate unified rejected live odds candidates: `{autopilot_cycle_daily_status.get('best_aggregate_unified_rejected_live_odds_candidate_count')}`",
            f"- Best aggregate unified rows with rejected live odds candidates: `{autopilot_cycle_daily_status.get('best_aggregate_unified_rows_with_rejected_live_odds_candidates')}`",
            f"- Best aggregate unified rejected live odds candidate reasons: `{autopilot_cycle_daily_status.get('best_aggregate_unified_rejected_live_odds_candidate_reason_counts')}`",
            f"- Best aggregate unified sample-blocking gap races: `{autopilot_cycle_daily_status.get('best_aggregate_unified_sample_blocking_gap_count')}`",
            f"- Best aggregate unified gap actions: `{autopilot_cycle_daily_status.get('best_aggregate_unified_gap_action_counts')}`",
            f"- Best aggregate unified evidence-missing reasons: `{autopilot_cycle_daily_status.get('best_aggregate_unified_gap_evidence_missing_reason_counts')}`",
            f"- Best aggregate unified top gap race IDs: `{autopilot_cycle_daily_status.get('best_aggregate_unified_top_gap_race_ids')}`",
            f"- Best aggregate unified top gap races: `{autopilot_cycle_daily_status.get('best_aggregate_unified_top_gap_races')}`",
            f"- Best aggregate unified top official-result-missing race IDs: `{autopilot_cycle_daily_status.get('best_aggregate_unified_top_official_result_missing_race_ids')}`",
            f"- Best aggregate unified top official-result-missing races: `{autopilot_cycle_daily_status.get('best_aggregate_unified_top_official_result_missing_races')}`",
            f"- Autopilot cycle rolling comparison: `{autopilot_cycle_daily_status.get('rolling_model_comparison_status')}`",
            f"- Autopilot cycle rolling comparison sample races: `{autopilot_cycle_daily_status.get('rolling_model_comparison_sample_races')}`",
            f"- Autopilot cycle rolling comparison best candidate: `{autopilot_cycle_daily_status.get('rolling_model_comparison_best_candidate')}`",
            f"- Autopilot cycle rolling comparison source rejected live odds candidates: `{autopilot_cycle_daily_status.get('rolling_model_comparison_source_rejected_live_odds_candidate_count')}`",
            f"- Autopilot cycle rolling comparison source rows with rejected live odds candidates: `{autopilot_cycle_daily_status.get('rolling_model_comparison_source_rows_with_rejected_live_odds_candidates')}`",
            f"- Autopilot cycle rolling comparison source rejected live odds candidate reasons: `{autopilot_cycle_daily_status.get('rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts')}`",
            f"- Autopilot cycle rolling comparison blockers: `{autopilot_cycle_daily_status.get('rolling_model_comparison_blockers') or []}`",
            f"- Autopilot cycle high-accuracy packet: `{autopilot_cycle_daily_status.get('high_accuracy_refinement_status')}`",
            f"- Autopilot cycle high-accuracy PR gate: `{autopilot_cycle_daily_status.get('high_accuracy_promotion_pr_gate_status')}`",
            f"- Autopilot cycle high-accuracy timing-aligned rerun plan: `{autopilot_cycle_daily_status.get('timing_aligned_rerun_plan')}`",
            f"- Autopilot cycle high-accuracy timing-aligned rerun execution status: `{autopilot_cycle_daily_status.get('timing_aligned_rerun_execution_status')}`",
            f"- Autopilot cycle reserve substitution policy impact: `{autopilot_cycle_daily_status.get('reserve_substitution_policy_impact_status')}`",
            f"- Autopilot cycle reserve substitution policy impact ready candidates: `{autopilot_cycle_daily_status.get('reserve_substitution_policy_impact_ready_candidate_count')}`",
            f"- Autopilot cycle reserve substitution policy impact mapping pairs: `{autopilot_cycle_daily_status.get('reserve_substitution_policy_impact_mapping_pair_count')}`",
            f"- Autopilot cycle reserve substitution policy impact potential runner rows blocked: `{autopilot_cycle_daily_status.get('reserve_substitution_policy_impact_potential_runner_rows_blocked')}`",
            f"- Autopilot cycle reserve substitution policy impact dataset join allowed: `{autopilot_cycle_daily_status.get('reserve_substitution_policy_impact_dataset_join_allowed')}`",
            f"- Autopilot cycle reserve substitution policy impact official-result acceptance allowed: `{autopilot_cycle_daily_status.get('reserve_substitution_policy_impact_official_result_acceptance_allowed')}`",
            f"- Autopilot cycle reserve substitution policy impact DB write: `{autopilot_cycle_daily_status.get('reserve_substitution_policy_impact_db_write')}`",
            f"- Autopilot cycle promotion distance: `{autopilot_cycle_daily_status.get('promotion_distance_status')}`",
            f"- Autopilot cycle promotion distance promotion ready: `{autopilot_cycle_daily_status.get('promotion_distance_promotion_ready')}`",
            f"- Autopilot cycle promotion distance sample races: `{autopilot_cycle_daily_status.get('promotion_distance_sample_race_count')}`",
            f"- Autopilot cycle promotion distance source rejected live odds candidates: `{autopilot_cycle_daily_status.get('promotion_distance_source_rejected_live_odds_candidate_count')}`",
            f"- Autopilot cycle promotion distance source rows with rejected live odds candidates: `{autopilot_cycle_daily_status.get('promotion_distance_source_rows_with_rejected_live_odds_candidates')}`",
            f"- Autopilot cycle promotion distance source rejected live odds candidate reasons: `{autopilot_cycle_daily_status.get('promotion_distance_source_rejected_live_odds_candidate_reason_counts')}`",
            f"- Autopilot cycle promotion distance source exclusion reasons: `{autopilot_cycle_daily_status.get('promotion_distance_source_exclusion_reason_counts')}`",
            f"- Autopilot cycle promotion distance source odds exclusion reasons: `{autopilot_cycle_daily_status.get('promotion_distance_source_odds_exclusion_reason_counts')}`",
            f"- Autopilot cycle promotion distance source official-result missing race IDs: `{autopilot_cycle_daily_status.get('promotion_distance_source_official_result_evidence_db_missing_race_ids')}`",
            f"- Autopilot cycle promotion distance official-result coverage requested races: `{autopilot_cycle_daily_status.get('promotion_distance_official_result_coverage_requested_race_count')}`",
            f"- Autopilot cycle promotion distance official-result requested race count source: `{autopilot_cycle_daily_status.get('promotion_distance_official_result_coverage_requested_race_count_source')}`",
            f"- Autopilot cycle promotion distance official-result legacy requested race count without IDs: `{autopilot_cycle_daily_status.get('promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids')}`",
            f"- Autopilot cycle promotion distance official-result coverage races with rows: `{autopilot_cycle_daily_status.get('promotion_distance_official_result_coverage_races_with_rows_count')}`",
            f"- Autopilot cycle promotion distance official-result coverage missing races: `{autopilot_cycle_daily_status.get('promotion_distance_official_result_coverage_missing_race_count')}`",
            f"- Autopilot cycle promotion distance official-result missing exclusions: `{autopilot_cycle_daily_status.get('promotion_distance_official_result_coverage_missing_exclusion_count')}`",
            f"- Autopilot cycle promotion distance official-result runner path count: `{autopilot_cycle_daily_status.get('promotion_distance_official_result_runner_path_count')}`",
            f"- Autopilot cycle promotion distance official-result runner paths source: `{autopilot_cycle_daily_status.get('promotion_distance_official_result_runner_paths_source_field')}`",
            f"- Autopilot cycle promotion distance best non-market candidate: `{autopilot_cycle_daily_status.get('promotion_distance_best_non_market_candidate_key')}`",
            f"- Autopilot cycle promotion distance blockers: `{autopilot_cycle_daily_status.get('promotion_distance_blockers') or []}`",
            f"- Autopilot cycle timing-aligned rerun plan: `{autopilot_cycle_daily_status.get('timing_aligned_prediction_rerun_plan_status')}`",
            f"- Autopilot cycle timing-aligned rerun hard stops: `{autopilot_cycle_daily_status.get('timing_aligned_prediction_rerun_plan_hard_stops') or []}`",
            f"- Autopilot cycle timing-aligned rerun execution: `{autopilot_cycle_daily_status.get('timing_aligned_prediction_rerun_execution_status')}`",
            f"- Autopilot cycle timing-aligned rerun execution hard stops: `{autopilot_cycle_daily_status.get('timing_aligned_prediction_rerun_execution_hard_stops') or []}`",
            f"- Autopilot cycle timing-aligned rerun executed: `{autopilot_cycle_daily_status.get('timing_aligned_prediction_rerun_execution_performed')}`",
            f"- Autopilot cycle timing-aligned rerun output: `{autopilot_cycle_daily_status.get('timing_aligned_prediction_rerun_output_dir')}`",
            f"- Autopilot cycle timing-aligned rerun odds snapshot dir: `{autopilot_cycle_daily_status.get('timing_aligned_prediction_rerun_odds_snapshot_dir')}`",
            f"- Autopilot cycle timing-aligned rerun odds snapshot: `{autopilot_cycle_daily_status.get('timing_aligned_prediction_rerun_odds_snapshot_status')}`",
            f"- Rejoin unified evidence: `{rejoin_unified_evidence_status.get('status')}`",
            f"- Rejoin unified evidence reason: `{rejoin_unified_evidence_status.get('status_reason')}`",
            f"- Rejoin unified evaluated candidates: `{rejoin_unified_evidence_status.get('evaluated_dataset_candidate_count')}`",
            f"- Rejoin unified datasets: `{rejoin_unified_evidence_status.get('dataset_count')}`",
            f"- Rejoin unified skipped datasets: `{rejoin_unified_evidence_status.get('skipped_dataset_count')}`",
            f"- Rejoin unified skip reasons: `{rejoin_unified_evidence_status.get('skip_reason_counts')}`",
            f"- Rejoin unified failure reasons: `{rejoin_unified_evidence_status.get('failure_reason_counts')}`",
            f"- Rejoin unified eligible rows: `{rejoin_unified_evidence_status.get('unified_evidence_eligible_rows')}`",
            f"- Rejoin artifact odds accepted rows: `{rejoin_unified_evidence_status.get('artifact_odds_rows_accepted')}`",
            f"- Rejoin artifact odds rejected rows: `{rejoin_unified_evidence_status.get('artifact_odds_rows_rejected')}`",
            f"- Rejoin artifact odds rejection reasons: `{rejoin_unified_evidence_status.get('artifact_odds_rejection_reason_counts')}`",
            f"- Rejoin rows with artifact shadow odds: `{rejoin_unified_evidence_status.get('rows_with_artifact_shadow_odds')}`",
            f"- Rejoin rows with artifact shadow odds candidates: `{rejoin_unified_evidence_status.get('rows_with_artifact_shadow_odds_candidates')}`",
            f"- Rejoin rejected live odds candidates: `{rejoin_unified_evidence_status.get('rejected_live_odds_candidate_count')}`",
            f"- Rejoin rows with rejected live odds candidates: `{rejoin_unified_evidence_status.get('rows_with_rejected_live_odds_candidates')}`",
            f"- Rejoin rejected live odds candidate reasons: `{rejoin_unified_evidence_status.get('rejected_live_odds_candidate_reason_counts')}`",
            f"- Join-eligibility preview datasets: `{rejoin_unified_evidence_status.get('join_eligibility_preview_dataset_count')}`",
            f"- Join-eligibility preview eligible rows: `{rejoin_unified_evidence_status.get('join_eligibility_preview_unified_eligible_rows')}`",
            f"- Join-eligibility preview accepted races: `{rejoin_unified_evidence_status.get('join_eligibility_preview_packet_accepted_races')}`",
            f"- Join-eligibility preview present races: `{rejoin_unified_evidence_status.get('join_eligibility_preview_packet_present_races')}`",
            f"- Join-eligibility preview missing race IDs: `{rejoin_unified_evidence_status.get('join_eligibility_preview_missing_race_ids')}`",
            f"- Rejoin rolling comparison: `{rejoin_rolling_model_comparison_status.get('status')}`",
            f"- Rejoin rolling comparison sample races: `{rejoin_rolling_model_comparison_status.get('sample_race_count')}`",
            f"- Rejoin high-accuracy packet: `{rejoin_high_accuracy_refinement_status.get('status')}`",
            f"- Rejoin high-accuracy timing-aligned rerun plan: `{rejoin_high_accuracy_refinement_status.get('timing_aligned_rerun_plan')}`",
            f"- Rejoin high-accuracy timing-aligned rerun execution status: `{rejoin_high_accuracy_refinement_status.get('timing_aligned_rerun_execution_status')}`",
            f"- Rejoin reserve substitution preflight: `{rejoin_high_accuracy_refinement_status.get('reserve_substitution_preflight_status')}`",
            f"- Rejoin reserve substitution ready for policy review: `{rejoin_high_accuracy_refinement_status.get('reserve_substitution_preflight_ready_for_policy_review_count')}`",
            f"- Rejoin reserve substitution dataset join blockers: `{rejoin_high_accuracy_refinement_status.get('reserve_substitution_preflight_dataset_join_blocker_counts')}`",
            f"- Rejoin reserve substitution ready race IDs: `{rejoin_high_accuracy_refinement_status.get('reserve_substitution_preflight_ready_race_ids')}`",
            f"- Rejoin reserve substitution manual review: `{rejoin_high_accuracy_refinement_status.get('reserve_substitution_manual_review_status')}`",
            f"- Rejoin reserve substitution manual review ready candidates: `{rejoin_high_accuracy_refinement_status.get('reserve_substitution_manual_review_ready_candidate_count')}`",
            f"- Rejoin reserve substitution manual review mapping pairs: `{rejoin_high_accuracy_refinement_status.get('reserve_substitution_manual_review_mapping_pair_count')}`",
            f"- Rejoin reserve substitution manual review dataset join allowed: `{rejoin_high_accuracy_refinement_status.get('reserve_substitution_manual_review_dataset_join_allowed')}`",
            f"- Rejoin reserve substitution manual review official-result acceptance allowed: `{rejoin_high_accuracy_refinement_status.get('reserve_substitution_manual_review_official_result_acceptance_allowed')}`",
            f"- Rejoin reserve substitution manual review DB write: `{rejoin_high_accuracy_refinement_status.get('reserve_substitution_manual_review_db_write')}`",
            f"- Rejoin reserve substitution policy impact: `{rejoin_high_accuracy_refinement_status.get('reserve_substitution_policy_impact_status')}`",
            f"- Rejoin reserve substitution policy impact ready candidates: `{rejoin_high_accuracy_refinement_status.get('reserve_substitution_policy_impact_ready_candidate_count')}`",
            f"- Rejoin reserve substitution policy impact mapping pairs: `{rejoin_high_accuracy_refinement_status.get('reserve_substitution_policy_impact_mapping_pair_count')}`",
            f"- Rejoin reserve substitution policy impact potential runner rows blocked: `{rejoin_high_accuracy_refinement_status.get('reserve_substitution_policy_impact_potential_runner_rows_blocked')}`",
            f"- Rejoin reserve substitution policy impact dataset join allowed: `{rejoin_high_accuracy_refinement_status.get('reserve_substitution_policy_impact_dataset_join_allowed')}`",
            f"- Rejoin reserve substitution policy impact official-result acceptance allowed: `{rejoin_high_accuracy_refinement_status.get('reserve_substitution_policy_impact_official_result_acceptance_allowed')}`",
            f"- Rejoin reserve substitution policy impact DB write: `{rejoin_high_accuracy_refinement_status.get('reserve_substitution_policy_impact_db_write')}`",
            f"- Rejoin pre-race gated challenger: `{rejoin_pre_race_gated_challenger_status.get('status')}`",
            f"- Rejoin pre-race gated challenger promotion ready: `{rejoin_pre_race_gated_challenger_status.get('promotion_ready')}`",
            f"- Rejoin pre-race predeclared residual candidate: `{rejoin_pre_race_gated_challenger_status.get('predeclared_residual_candidate_status')}`",
            f"- Rejoin pre-race predeclared residual triggered races: `{rejoin_pre_race_gated_challenger_status.get('predeclared_residual_triggered_race_count')}` / `{rejoin_pre_race_gated_challenger_status.get('predeclared_residual_minimum_triggered_races_for_directional_read')}`",
            f"- Rejoin pre-race predeclared residual directional read ready: `{rejoin_pre_race_gated_challenger_status.get('predeclared_residual_directional_read_ready')}`",
            f"- Rejoin rank-first hypothesis gate review: `{rejoin_rank_first_hypothesis_gated_status.get('rank_first_hypothesis_review_status')}`",
            f"- Rejoin rank-first hypothesis candidates: `{rejoin_rank_first_hypothesis_gated_status.get('rank_first_hypothesis_evaluated_candidate_count')}` / `{rejoin_rank_first_hypothesis_gated_status.get('rank_first_hypothesis_candidate_count')}`",
            f"- Rejoin rank-first hypothesis best candidate: `{rejoin_rank_first_hypothesis_gated_status.get('rank_first_hypothesis_best_candidate_key')}`",
            f"- Rejoin rank-first hypothesis best triggered races: `{rejoin_rank_first_hypothesis_gated_status.get('rank_first_hypothesis_best_triggered_race_count')}` / `{rejoin_rank_first_hypothesis_gated_status.get('rank_first_hypothesis_minimum_triggered_races_for_directional_read')}`",
            f"- Rejoin rank-first hypothesis directional read ready: `{rejoin_rank_first_hypothesis_gated_status.get('rank_first_hypothesis_directional_read_ready')}`",
            f"- Rejoin time-split gated challenger: `{rejoin_time_split_gated_challenger_status.get('status')}`",
            f"- Rejoin time-split gated challenger promotion ready: `{rejoin_time_split_gated_challenger_status.get('promotion_ready')}`",
            f"- Rejoin market residual challenger: `{rejoin_market_residual_challenger_status.get('status')}`",
            f"- Rejoin market residual challenger promotion ready: `{rejoin_market_residual_challenger_status.get('promotion_ready')}`",
            f"- Rejoin market residual regime audit: `{rejoin_market_residual_regime_audit_status.get('status')}`",
            f"- Rejoin market residual rank-first hypothesis: `{rejoin_market_residual_regime_audit_status.get('rank_first_hypothesis_status')}`",
            f"- Rejoin market residual rank-first help regimes: `{rejoin_market_residual_regime_audit_status.get('pre_race_rank_first_help_regime_count')}`",
            f"- Rejoin market residual logloss-only help regimes: `{rejoin_market_residual_regime_audit_status.get('pre_race_logloss_only_help_regime_count')}`",
            f"- Rejoin rank-first hypothesis watchlist: `{rejoin_rank_first_hypothesis_watchlist_status.get('status')}`",
            f"- Rejoin rank-first watchlist candidates: `{rejoin_rank_first_hypothesis_watchlist_status.get('candidate_count')}`",
            f"- Rejoin rank-first watchlist directional-ready candidates: `{rejoin_rank_first_hypothesis_watchlist_status.get('directional_ready_candidate_count')}`",
            f"- Rejoin rank-first watchlist best candidate: `{rejoin_rank_first_hypothesis_watchlist_status.get('best_candidate_key')}`",
            f"- Rejoin rank-first watchlist best status: `{rejoin_rank_first_hypothesis_watchlist_status.get('best_candidate_status')}`",
            f"- Rejoin rank-first watchlist best distinct samples: `{rejoin_rank_first_hypothesis_watchlist_status.get('best_candidate_distinct_sample_count')}` / `{rejoin_rank_first_hypothesis_watchlist_status.get('minimum_distinct_samples_for_directional_read')}`",
            f"- Rejoin promotion distance: `{rejoin_promotion_distance_status.get('status')}`",
            f"- Rejoin promotion distance promotion ready: `{rejoin_promotion_distance_status.get('promotion_ready')}`",
            f"- Rejoin promotion distance source exclusion reasons: `{rejoin_promotion_distance_status.get('source_exclusion_reason_counts')}`",
            f"- Rejoin promotion distance source official-result missing race IDs: `{rejoin_promotion_distance_status.get('source_official_result_evidence_db_missing_race_ids')}`",
            f"- Rejoin promotion distance official-result coverage requested races: `{rejoin_promotion_distance_status.get('official_result_coverage_requested_race_count')}`",
            f"- Rejoin promotion distance official-result requested race count source: `{rejoin_promotion_distance_status.get('official_result_coverage_requested_race_count_source')}`",
            f"- Rejoin promotion distance official-result legacy requested race count without IDs: `{rejoin_promotion_distance_status.get('official_result_coverage_legacy_requested_race_count_without_ids')}`",
            f"- Rejoin promotion distance official-result coverage races with rows: `{rejoin_promotion_distance_status.get('official_result_coverage_races_with_rows_count')}`",
            f"- Rejoin promotion distance official-result coverage missing races: `{rejoin_promotion_distance_status.get('official_result_coverage_missing_race_count')}`",
            f"- Rejoin promotion distance official-result missing exclusions: `{rejoin_promotion_distance_status.get('official_result_coverage_missing_exclusion_count')}`",
            f"- Rejoin promotion distance official-result runner path count: `{rejoin_promotion_distance_status.get('official_result_runner_path_count')}`",
            f"- Rejoin promotion distance official-result runner paths source: `{rejoin_promotion_distance_status.get('official_result_runner_paths_source_field')}`",
            f"- Rejoin promotion distance blockers: `{rejoin_promotion_distance_status.get('blockers') or []}`",
            f"- Live odds backlog discovered races: `{live_odds_backlog.get('live_odds_backlog_discovered_race_count')}`",
            f"- Live odds backlog candidate races: `{live_odds_backlog.get('live_odds_backlog_candidate_race_count')}`",
            f"- Live odds backlog unresolved races: `{live_odds_backlog.get('live_odds_backlog_unresolved_race_count')}`",
            f"- Live odds backlog unresolved reasons: `{live_odds_backlog.get('live_odds_backlog_unresolved_reason_counts')}`",
            f"- Live odds backlog recovery actions: `{live_odds_backlog.get('live_odds_backlog_unresolved_recovery_action_counts')}`",
            f"- Live odds backlog alias statuses: `{live_odds_backlog.get('live_odds_backlog_unresolved_alias_status_counts')}`",
            f"- Live odds backlog retryable exact-shadow matches: `{live_odds_backlog.get('live_odds_backlog_retryable_exact_shadow_match_race_count')}`",
            f"- Live odds backlog no exact shadow match: `{live_odds_backlog.get('live_odds_backlog_no_exact_shadow_match_race_count')}`",
            f"- Live odds backlog retryable exact-shadow race IDs: `{live_odds_backlog.get('live_odds_backlog_retryable_exact_shadow_match_race_ids')}`",
            f"- Live odds backlog no exact shadow match race IDs: `{live_odds_backlog.get('live_odds_backlog_no_exact_shadow_match_race_ids')}`",
            f"- Live odds backlog recovery queue: `{live_odds_backlog.get('live_odds_backlog_recovery_queue_path')}`",
            f"- Live odds backlog recovery queue diagnostic only: `{live_odds_backlog.get('live_odds_backlog_recovery_queue_diagnostic_only')}`",
            f"- Live odds backlog recovery queue changed join acceptance: `{live_odds_backlog.get('live_odds_backlog_recovery_queue_join_acceptance_changed')}`",
            f"- Live odds backlog recovery queue DB write performed: `{live_odds_backlog.get('live_odds_backlog_recovery_queue_db_write_performed')}`",
            f"- Live odds backlog awaiting official-result evidence races: `{live_odds_backlog.get('live_odds_backlog_awaiting_official_result_evidence_race_count')}`",
            f"- Live odds backlog awaiting official-result evidence race IDs: `{live_odds_backlog.get('live_odds_backlog_awaiting_official_result_evidence_race_ids')}`",
            f"- Live odds backlog awaiting official-result authorized action: `{live_odds_backlog.get('live_odds_backlog_awaiting_official_result_evidence_authorized_action')}`",
            f"- Live odds backlog awaiting official-result recheck-ready races: `{live_odds_backlog.get('live_odds_backlog_awaiting_official_result_recheck_ready_race_count')}`",
            f"- Live odds backlog runner-set validation: `{live_odds_backlog.get('live_odds_backlog_runner_set_validation_path')}`",
            f"- Live odds backlog runner-set retryable races: `{live_odds_backlog.get('live_odds_backlog_runner_set_validation_retryable_race_count')}`",
            f"- Live odds backlog runner-set exact matches: `{live_odds_backlog.get('live_odds_backlog_runner_set_validation_exact_match_race_count')}`",
            f"- Live odds backlog runner-set blocked races: `{live_odds_backlog.get('live_odds_backlog_runner_set_validation_blocked_race_count')}`",
            f"- Live odds backlog runner-set validation diagnostic only: `{live_odds_backlog.get('live_odds_backlog_runner_set_validation_diagnostic_only')}`",
            f"- Live odds backlog runner-set join authorized: `{live_odds_backlog.get('live_odds_backlog_runner_set_validation_join_authorized')}`",
            f"- Live odds backlog runner-set DB write performed: `{live_odds_backlog.get('live_odds_backlog_runner_set_validation_db_write_performed')}`",
            f"- Live odds backlog join eligibility packet: `{live_odds_backlog.get('live_odds_backlog_join_eligibility_packet_path')}`",
            f"- Live odds backlog join eligibility evaluated races: `{live_odds_backlog.get('live_odds_backlog_join_eligibility_evaluated_race_count')}`",
            f"- Live odds backlog join eligibility report-only races: `{live_odds_backlog.get('live_odds_backlog_join_eligibility_eligible_report_only_race_count')}`",
            f"- Live odds backlog join eligibility blocked races: `{live_odds_backlog.get('live_odds_backlog_join_eligibility_blocked_race_count')}`",
            f"- Live odds backlog join eligibility blocker counts: `{live_odds_backlog.get('live_odds_backlog_join_eligibility_blocker_counts', {})}`",
            f"- Live odds backlog join eligibility awaiting official-result recheck-ready races: `{live_odds_backlog.get('live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count')}`",
            f"- Live odds backlog join eligibility diagnostic only: `{live_odds_backlog.get('live_odds_backlog_join_eligibility_diagnostic_only')}`",
            f"- Live odds backlog join eligibility join authorized: `{live_odds_backlog.get('live_odds_backlog_join_eligibility_join_authorized')}`",
            f"- Live odds backlog join eligibility DB write performed: `{live_odds_backlog.get('live_odds_backlog_join_eligibility_db_write_performed')}`",
            f"- Observability: `{observability_status.get('status')}`",
            f"- Prediction rows observed: `{observability_status.get('prediction_rows')}`",
            f"- Cycle activity: `{cycle_activity.get('status')}`",
            f"- Safe joined delta this cycle: `{cycle_activity.get('safe_joined_delta_this_cycle')}`",
            f"- Next pre-jump refresh status: `{next_prejump_refresh_window.get('status')}`",
            f"- Recommended rerun after: `{next_prejump_refresh_window.get('recommended_rerun_after_local')}`",
            f"- Next pre-jump race: `{next_prejump_race.get('race_id')}` at `{next_prejump_race.get('jump_datetime')}`",
            f"- Pre-jump metadata: `{prejump_metadata_status.get('status')}`",
            f"- Pre-jump metadata verified eligible: `{prejump_metadata_status.get('eligible_with_verified_prejump_metadata')}` / `{prejump_metadata_status.get('eligible_count')}`",
            f"- Pre-jump metadata trend: `{prejump_metadata_trend.get('status')}`",
            f"- Trend verified metadata rate: `{prejump_metadata_trend.get('verified_metadata_rate')}`",
            "",
            "## Automation",
            f"- Rejoin attempts this cycle: `{automated_join_report.get('rejoin_attempt_count')}`",
            f"- Rejoin safe joined count sum across attempts: `{automated_join_report.get('rejoin_safe_joined_count_sum')}`",
            f"- Service files present: `{service_validation.get('service_files_present')}`",
            f"- Timer frequency: `{service_validation.get('timer_frequency')}`",
            f"- Deployment status: `{service_validation.get('deployment_status')}`",
            "",
            "## Alerts",
            f"- Status: `{alert_report.get('status')}`",
            f"- Triggered alerts: `{alert_report.get('triggered_alerts')}`",
            "",
            "## Readiness",
            f"- Decision: `{readiness.get('decision')}`",
            f"- Blockers: `{readiness.get('outstanding_blockers')}`",
            "",
            "No training, production promotion, registry mutation, production pointer update, active-model replacement, label write, TGR enablement, betting/EV action, production prediction overwrite, snapshot rewrite, schema change, hyperparameter change, calibration-method change, or champion modification was performed. Any DB write is restricted to explicitly enabled append-only live odds or official-result evidence capture and is reported above.",
            "",
        ]
    )


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    generated_at = datetime.now().astimezone()
    run_id = args.run_id or now_id(generated_at)
    evidence_root = args.evidence_root
    output_dir = assert_output_dir_safe(
        args.output_dir or evidence_root / f"shadow_autopilot_daemonization_v1_{run_id}",
        evidence_root=evidence_root,
    )
    output_dir = unique_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    current_time = args.current_time or generated_at.isoformat()
    lock_path = args.lock_path or DEFAULT_LOCK_PATH
    state_path = args.state_path or DEFAULT_STATE_PATH
    odds_state_path = args.odds_capture_state_path or DEFAULT_ODDS_CAPTURE_ONLY_STATE_PATH
    write_json(
        output_dir / "daemon_run_report.json",
        initial_daemon_run_report(
            run_id=run_id,
            generated_at=generated_at,
            current_time=current_time,
            output_dir=output_dir,
            lock_path=lock_path,
            state_path=state_path,
            odds_capture_state_path=odds_state_path,
            autonomous_odds_capture_enabled=args.enable_autonomous_odds_capture,
            autonomous_result_capture_enabled=args.enable_autonomous_result_capture,
        ),
    )

    service_info = write_service_files(
        repo_path=ROOT,
        timeout_seconds=args.timeout_seconds,
        python_path=Path(sys.executable),
        evidence_root=evidence_root,
        shadow_model=args.shadow_model,
        db_path=args.db,
        lock_path=args.lock_path,
        state_path=args.state_path,
        odds_capture_state_path=args.odds_capture_state_path,
    )
    service_path = DEFAULT_SERVICE_DIR / SERVICE_NAME
    timer_path = DEFAULT_SERVICE_DIR / TIMER_NAME
    write_text(output_dir / "daemon_design.md", daemon_design_markdown())
    write_json(output_dir / "lifecycle_diagram.json", lifecycle_diagram())
    write_text(output_dir / "service_install.md", install_markdown(service_info))
    copy_if_exists(service_path, output_dir / "systemd" / SERVICE_NAME)
    copy_if_exists(timer_path, output_dir / "systemd" / TIMER_NAME)
    write_json(output_dir / "output_manifest.json", output_manifest(output_dir))

    current_dt = parse_datetime_value(current_time, default_tz=generated_at.tzinfo) or generated_at
    if args.enable_autonomous_odds_capture:
        odds_state = load_json(odds_state_path)
        defer_decision = full_daemon_odds_window_defer_decision(
            odds_state,
            current_time=current_dt,
        )
        defer_decision["odds_capture_state_path"] = relpath(odds_state_path)
        write_json(output_dir / "full_daemon_odds_window_defer.json", defer_decision)
        if defer_decision.get("should_defer"):
            result = {
                "schema_version": "shadow_autopilot_daemon_run_v1",
                "run_id": run_id,
                "generated_at": generated_at.isoformat(),
                "output_dir": relpath(output_dir),
                "final_verdict": "DAEMON_DEFERRED_TO_ODDS_CAPTURE_ONLY",
                "runtime_action": "DEFER_FULL_DAEMON_FOR_FIXED_WINDOW_ODDS_CAPTURE",
                "readiness_decision": "ODDS_CAPTURE_PRIORITY",
                "lock_path": relpath(lock_path),
                "lock": None,
                "lock_release": None,
                "lock_validation_status": "NOT_ACQUIRED_DEFERRED",
                "odds_capture_state_path": relpath(odds_state_path),
                "odds_capture_defer_decision": defer_decision,
                "protected_paths_unchanged_or_allowed": True,
                "systemd_deployment_status": service_info.get("status"),
                "systemd_deployment_ready": service_info.get("systemd_deployment_ready"),
                "no_write_guarantees": {
                    "db_write": False,
                    "live_odds_write": False,
                    "official_result_evidence_write": False,
                    "label_write": False,
                    "production_pointer_update": False,
                    "production_promotion": False,
                    "registry_mutation": False,
                    "training": False,
                    "betting_or_ev_action": False,
                },
            }
            write_json(output_dir / "daemon_run_report.json", result)
            write_text(output_dir / "final_status.txt", result["final_verdict"] + "\n")
            write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
            return result

    previous_dashboard_path = latest_artifact(
        evidence_root,
        "shadow_autopilot_daemonization_v1_",
        "shadow_dashboard.json",
    )
    if previous_dashboard_path == output_dir:
        previous_dashboard = None
        previous_observability = None
    else:
        previous_dashboard = load_json((previous_dashboard_path / "shadow_dashboard.json") if previous_dashboard_path else None)
        previous_observability = load_json((previous_dashboard_path / "observability_status.json") if previous_dashboard_path else None)

    protected_before = protected_hashes()
    steps: list[dict[str, Any]] = []
    lock_payload: dict[str, Any] | None = None
    lock_release: dict[str, Any] | None = None
    autopilot_output_dir: Path | None = None
    daily_shadow_run_dir: Path | None = None
    daily_manifest: dict[str, Any] | None = None
    shadow_odds_snapshot: dict[str, Any] | None = None
    odds_capture_state_publish: dict[str, Any] = {"status": "NOT_RUN"}
    lock_validation: dict[str, Any]
    recovery_validation = {"status": "NOT_RUN"}
    try:
        lock_payload = acquire_lock_with_odds_capture_retry(
            lock_path=lock_path,
            run_id=run_id,
            stale_after_seconds=args.lock_stale_seconds,
            output_dir=output_dir,
        )
        duplicate_probe = probe_duplicate_lock(
            lock_path,
            stale_after_seconds=args.lock_stale_seconds,
            output_dir=output_dir,
        )
        stale_probe = probe_stale_lock_cleanup(output_dir)
        lock_validation = {
            "schema_version": "shadow_autopilot_lock_validation_v1",
            "lock_path": relpath(lock_path),
            "lock_acquired": True,
            "lock_payload": lock_payload,
            "duplicate_probe": duplicate_probe,
            "stale_lock_cleanup_probe": stale_probe,
            "status": "PASS"
            if duplicate_probe.get("status") == "PASS" and stale_probe.get("status") == "PASS"
            else "FAIL",
        }
        recovery_validation = {
            "schema_version": "shadow_autopilot_recovery_validation_v1",
            "timeout_probe": simulate_timeout_recovery(output_dir),
            "partial_run_recovery": {
                "status": "PASS",
                "policy": "aggregate/status refresh still runs after rejoin sweep; failed command details remain in structured logs",
            },
        }
        recovery_validation["status"] = (
            "PASS" if recovery_validation["timeout_probe"].get("status") == "PASS" else "FAIL"
        )

        autopilot_run_id = f"{run_id}_daemon"
        autopilot_command = [
            sys.executable,
            str(ROOT / "scripts/shadow_autopilot_v1.py"),
            "--run-id",
            autopilot_run_id,
            "--evidence-root",
            str(evidence_root),
            "--current-time",
            current_time,
            "--db",
            str(args.db),
            "--days-ahead",
            str(args.days_ahead),
            "--min-minutes",
            str(args.min_minutes),
            "--max-minutes",
            str(args.max_minutes),
            "--refresh-limit",
            str(args.refresh_limit),
            "--autonomous-odds-capture-limit",
            str(args.autonomous_odds_capture_limit),
            "--result-backlog-limit",
            str(args.result_backlog_limit),
            "--result-backlog-shadow-run-limit",
            str(args.result_backlog_shadow_run_limit),
            "--result-backlog-lookback-days",
            str(args.result_backlog_lookback_days),
            "--refresh-command-mode",
            args.refresh_command_mode,
            "--score-command-mode",
            args.score_command_mode,
            "--target-joined-races",
            str(args.target_joined_races),
            "--min-joined-races",
            str(args.min_joined_races),
            "--step-timeout-seconds",
            str(args.timeout_seconds),
        ]
        autopilot_command.extend(shadow_model_cli_args(args.shadow_model))
        if args.refresh_dry_run:
            autopilot_command.append("--refresh-dry-run")
        if args.require_safe_refresh_metadata:
            autopilot_command.append("--require-safe-refresh-metadata")
        if args.skip_refresh:
            autopilot_command.append("--skip-refresh")
        if args.skip_shadow_run:
            autopilot_command.append("--skip-shadow-run")
        if args.skip_unified_dataset:
            autopilot_command.append("--skip-unified-dataset")
        if args.enable_autonomous_odds_capture:
            autopilot_command.append("--enable-autonomous-odds-capture")
        if args.execute_autonomous_odds_capture:
            autopilot_command.append("--execute-autonomous-odds-capture")
        if args.allow_auto_scrape_odds:
            autopilot_command.append("--allow-auto-scrape-odds")
        if args.enable_autonomous_result_capture:
            autopilot_command.append("--enable-autonomous-result-capture")
        steps.append(
            run_command(
                name="autopilot_cycle",
                command=autopilot_command,
                output_dir=output_dir,
                timeout_seconds=autopilot_cycle_timeout_seconds(args.timeout_seconds),
            )
        )
        autopilot_stdout = output_dir / "logs" / "autopilot_cycle.stdout.txt"
        autopilot_result = load_json(autopilot_stdout)
        autopilot_output_dir = (
            ROOT / str(autopilot_result.get("output_dir"))
            if autopilot_result and autopilot_result.get("output_dir")
            else latest_artifact(evidence_root, "shadow_autopilot_v1_", "shadow_dashboard.json")
        )
        daily_shadow_run_dir, daily_manifest = daily_shadow_run_from_autopilot(
            autopilot_output_dir
        )
        shadow_odds_snapshot = shadow_odds_snapshot_status_from_autopilot(
            autopilot_output_dir
        )
        post_primary_autonomous_live_odds_capture_status = (
            autonomous_live_odds_capture_status_from_autopilot(autopilot_output_dir)
        )
        odds_capture_state_publish = publish_full_daemon_odds_capture_state(
            state_path=args.odds_capture_state_path,
            generated_at=current_dt,
            run_id=run_id,
            output_dir=output_dir,
            autopilot_output_dir=autopilot_output_dir,
            odds_status=post_primary_autonomous_live_odds_capture_status,
        )
        write_json(
            output_dir / "odds_capture_state_publish_status.json",
            odds_capture_state_publish,
        )
        post_primary_release_decision = post_primary_odds_capture_release_decision(
            odds_capture_state_publish,
            current_time=current_dt,
        )
        write_json(
            output_dir / "post_primary_odds_capture_release_decision.json",
            post_primary_release_decision,
        )
        if post_primary_release_decision.get("should_release"):
            autopilot_daily_status_path = (
                autopilot_output_dir / "DAILY_STATUS.json"
                if autopilot_output_dir
                else None
            )
            autopilot_daily_status = load_json(autopilot_daily_status_path)
            if not isinstance(autopilot_daily_status, Mapping):
                autopilot_daily_status = {}
            protected_after = protected_hashes()
            protected_paths_unchanged = protected_before == protected_after
            protected_changed_paths = sorted(
                key
                for key, before_value in protected_before.items()
                if protected_after.get(key) != before_value
            )
            autonomous_odds_inserted_rows = int_or_zero(
                post_primary_autonomous_live_odds_capture_status.get(
                    "inserted_live_odds_rows"
                )
            )
            allowed_odds_db_change = (
                bool(protected_changed_paths)
                and autonomous_odds_inserted_rows > 0
                and set(protected_changed_paths).issubset({relpath(args.db)})
            )
            protected_paths_unchanged_or_allowed = (
                protected_paths_unchanged or allowed_odds_db_change
            )
            if lock_payload:
                lock_release = release_lock(lock_path, run_id)
                lock_payload = None
            result = {
                **completed_daemon_run_report_envelope(
                    run_id=run_id,
                    generated_at=generated_at,
                    current_time=current_time,
                    output_dir=output_dir,
                    final_verdict="PARTIAL_DAEMONIZATION",
                ),
                "status": "PARTIAL_DAEMONIZATION",
                "runtime_action": "RELEASE_FULL_DAEMON_FOR_ODDS_CAPTURE",
                "readiness_decision": "ODDS_CAPTURE_PRIORITY",
                "autopilot_output_dir": relpath(autopilot_output_dir),
                "autopilot_daily_status_path": relpath(autopilot_daily_status_path),
                "daily_shadow_run_dir": relpath(daily_shadow_run_dir),
                "autopilot_daily_status_generated_at": autopilot_daily_status.get(
                    "generated_at"
                ),
                "autopilot_daily_readiness_decision": autopilot_daily_status.get(
                    "readiness_decision"
                ),
                "autonomous_live_odds_capture_status": autopilot_daily_status.get(
                    "autonomous_live_odds_capture_status"
                ),
                "autonomous_live_odds_inserted_rows": autopilot_daily_status.get(
                    "autonomous_live_odds_inserted_rows"
                ),
                "autonomous_official_result_capture_status": autopilot_daily_status.get(
                    "autonomous_official_result_capture_status"
                ),
                "autonomous_official_result_candidate_count": autopilot_daily_status.get(
                    "autonomous_official_result_candidate_count"
                ),
                "autonomous_official_result_quarantined_race_ids": (
                    autopilot_daily_status.get(
                        "autonomous_official_result_quarantined_race_ids"
                    )
                    or []
                ),
                "autonomous_official_result_quarantine_reason_counts": (
                    autopilot_daily_status.get(
                        "autonomous_official_result_quarantine_reason_counts"
                    )
                    or {}
                ),
                "autonomous_official_result_quarantine_error_counts": (
                    autopilot_daily_status.get(
                        "autonomous_official_result_quarantine_error_counts"
                    )
                    or {}
                ),
                "autonomous_official_result_quarantine_result_boxes_not_in_participants_counts": (
                    autopilot_daily_status.get(
                        "autonomous_official_result_quarantine_result_boxes_not_in_participants_counts"
                    )
                    or {}
                ),
                "autonomous_official_result_quarantine_runner_set_mismatch_samples": (
                    autopilot_daily_status.get(
                        "autonomous_official_result_quarantine_runner_set_mismatch_samples"
                    )
                    or []
                ),
                "unified_evidence_dataset_status": autopilot_daily_status.get(
                    "unified_evidence_dataset_status"
                ),
                "unified_evidence_dataset_rows": autopilot_daily_status.get(
                    "unified_evidence_dataset_rows"
                ),
                "unified_evidence_eligible_rows": autopilot_daily_status.get(
                    "unified_evidence_eligible_rows"
                ),
                "backlog_unified_evidence_eligible_rows": autopilot_daily_status.get(
                    "backlog_unified_evidence_eligible_rows"
                ),
                "rolling_model_comparison_status": autopilot_daily_status.get(
                    "rolling_model_comparison_status"
                ),
                "rolling_model_comparison_sample_races": autopilot_daily_status.get(
                    "rolling_model_comparison_sample_races"
                ),
                "high_accuracy_refinement_status": autopilot_daily_status.get(
                    "high_accuracy_refinement_status"
                ),
                "high_accuracy_promotion_pr_gate_status": autopilot_daily_status.get(
                    "high_accuracy_promotion_pr_gate_status"
                ),
                **autopilot_cycle_operational_fields(autopilot_daily_status),
                "promotion_distance_status": autopilot_daily_status.get(
                    "promotion_distance_status"
                ),
                "promotion_distance_blockers": autopilot_daily_status.get(
                    "promotion_distance_blockers"
                ),
                "prediction_rows_today": autopilot_daily_status.get(
                    "prediction_rows_today"
                ),
                "races_scored_today": autopilot_daily_status.get("races_scored_today"),
                "races_with_complete_valid_prejump_odds": autopilot_daily_status.get(
                    "races_with_complete_valid_prejump_odds"
                ),
                "races_with_missing_odds_rows": autopilot_daily_status.get(
                    "races_with_missing_odds_rows"
                ),
                "odds_capture_state_publish_status": odds_capture_state_publish.get(
                    "status"
                ),
                "odds_capture_state_path": odds_capture_state_publish.get(
                    "state_path"
                ),
                "odds_capture_next_meaningful_action": odds_capture_state_publish.get(
                    "next_meaningful_action"
                ),
                "odds_capture_next_meaningful_action_at": odds_capture_state_publish.get(
                    "next_meaningful_action_at"
                ),
                "post_primary_odds_capture_release_decision": (
                    post_primary_release_decision
                ),
                "step_count": len(steps),
                "steps": list(steps),
                "lock_path": relpath(lock_path),
                "lock": lock_payload,
                "lock_release": lock_release,
                "lock_validation_status": lock_validation.get("status"),
                "protected_paths_unchanged_or_allowed": protected_paths_unchanged_or_allowed,
                "protected_changed_paths": protected_changed_paths,
                "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
            }
            write_text(output_dir / "final_status.txt", "PARTIAL_DAEMONIZATION\n")
            write_json(output_dir / "daemon_run_report.json", result)
            write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
            return result

        automated_join_report = rejoin_pending_shadow_runs(
            run_id=run_id,
            output_dir=output_dir,
            evidence_root=evidence_root,
            db_path=args.db,
            current_time=current_time,
            pending_limit=args.rejoin_pending_limit,
            lookback_days=args.rejoin_lookback_days,
            timeout_seconds=args.timeout_seconds,
        )
        write_json(output_dir / "automated_join_report.json", automated_join_report)
        (
            rejoin_unified_status,
            rejoin_unified_steps,
            rejoin_unified_report_paths,
        ) = build_rejoin_unified_evidence_datasets(
            run_id=run_id,
            output_dir=output_dir,
            evidence_root=evidence_root,
            db_path=args.db,
            automated_join_report=automated_join_report,
            generated_at=generated_at,
            timeout_seconds=args.timeout_seconds,
        )
        steps.extend(rejoin_unified_steps)
        write_json(
            output_dir / "rejoin_unified_evidence_datasets_status.json",
            rejoin_unified_status,
        )
        best_aggregate_unified_status_path = best_unified_evidence_aggregate_status_path(
            [
                (
                    autopilot_output_dir
                    / "backlog_unified_evidence_datasets_status.json"
                    if autopilot_output_dir is not None
                    else None
                ),
                output_dir / "rejoin_unified_evidence_datasets_status.json",
            ]
        )
        best_aggregate_unified_status = (
            load_json(best_aggregate_unified_status_path)
            if best_aggregate_unified_status_path is not None
            else {}
        )
        rejoin_reserve_substitution_preflight_dir = (
            evidence_root
            / f"official_result_reserve_substitution_preflight_{run_id}_daemon_rejoin"
        )
        rejoin_reserve_substitution_preflight_report_path = (
            rejoin_reserve_substitution_preflight_dir
            / "official_result_reserve_substitution_preflight.json"
        )
        if best_aggregate_unified_status_path is not None:
            rejoin_reserve_preflight_step = run_command(
                name="official_result_reserve_substitution_preflight_after_daemon_rejoins",
                command=autopilot.reserve_substitution_preflight_command(
                    backlog_unified_evidence_status=best_aggregate_unified_status_path,
                    output_dir=rejoin_reserve_substitution_preflight_dir,
                ),
                output_dir=output_dir,
                timeout_seconds=args.timeout_seconds,
            )
            steps.append(rejoin_reserve_preflight_step)

        aggregate_dir = evidence_root / f"forward_shadow_result_aggregate_{run_id}_daemon"
        steps.append(
            run_command(
                name="aggregate_after_daemon_rejoins",
                command=[
                    sys.executable,
                    str(ROOT / "scripts/aggregate_forward_shadow_results.py"),
                    "--evidence-root",
                    str(evidence_root),
                    "--output-dir",
                    str(aggregate_dir),
                ],
                output_dir=output_dir,
                timeout_seconds=args.timeout_seconds,
            )
        )
        status_dir = evidence_root / f"forward_shadow_status_{run_id}_daemon"
        steps.append(
            run_command(
                name="status_after_daemon_rejoins",
                command=[
                    sys.executable,
                    str(ROOT / "scripts/forward_shadow_status_report.py"),
                    "--evidence-root",
                    str(evidence_root),
                    "--output-dir",
                    str(status_dir),
                    "--db",
                    str(args.db),
                    "--min-joined-races",
                    str(args.min_joined_races),
                ],
                output_dir=output_dir,
                timeout_seconds=args.timeout_seconds,
            )
        )
        rejoin_rolling_dir = evidence_root / f"rolling_model_comparison_{run_id}_daemon_rejoin"
        rejoin_rolling_report: dict[str, Any] | None = None
        rejoin_high_accuracy_dir = (
            evidence_root / f"high_accuracy_refinement_packet_{run_id}_daemon_rejoin"
        )
        rejoin_high_accuracy_after_promotion_dir = (
            evidence_root
            / f"high_accuracy_refinement_packet_{run_id}_daemon_rejoin_post_promotion_distance"
        )
        rejoin_high_accuracy_report: dict[str, Any] | None = None
        rejoin_pre_race_gated_dir = (
            evidence_root / f"pre_race_gated_challenger_{run_id}_daemon_rejoin"
        )
        rejoin_rank_first_hypothesis_gated_dir = (
            evidence_root
            / f"pre_race_gated_challenger_{run_id}_daemon_rejoin_rank_first_hypothesis_review"
        )
        rejoin_time_split_gated_dir = (
            evidence_root / f"time_split_gated_challenger_{run_id}_daemon_rejoin"
        )
        rejoin_market_residual_dir = (
            evidence_root / f"market_residual_challenger_{run_id}_daemon_rejoin"
        )
        rejoin_market_residual_regime_dir = (
            evidence_root / f"market_residual_regime_audit_{run_id}_daemon_rejoin"
        )
        rejoin_rank_first_hypothesis_watchlist_dir = (
            evidence_root / f"rank_first_hypothesis_watchlist_{run_id}_daemon_rejoin"
        )
        rejoin_promotion_distance_dir = (
            evidence_root / f"promotion_distance_report_{run_id}_daemon_rejoin"
        )
        if rejoin_unified_report_paths:
            rejoin_comparison_report_paths = list(rejoin_unified_report_paths)
            rejoin_comparison_report_paths.extend(
                autopilot.historical_unified_evidence_report_paths(
                    evidence_root,
                    exclude_paths=rejoin_comparison_report_paths,
                )
            )
            rejoin_comparison_report_paths = autopilot.unique_sorted_report_paths(
                rejoin_comparison_report_paths
            )
            rejoin_rolling_step = run_command(
                name="rolling_model_comparison_after_daemon_rejoins",
                command=autopilot.rolling_model_comparison_command(
                    unified_evidence_reports=rejoin_comparison_report_paths,
                    output_dir=rejoin_rolling_dir,
                ),
                output_dir=output_dir,
                timeout_seconds=args.timeout_seconds,
            )
            steps.append(rejoin_rolling_step)
            rejoin_rolling_report = (
                load_json(rejoin_rolling_dir / "rolling_model_comparison_report.json")
                or {}
            )
            rejoin_rolling_status = autopilot.build_rolling_model_comparison_status(
                generated_at=generated_at,
                comparison_dir=rejoin_rolling_dir,
                comparison_report=rejoin_rolling_report or None,
                attempted=True,
                returncode=rejoin_rolling_step.get("returncode"),
            )
            rejoin_runner_matrix_csv = (
                rejoin_rolling_dir / "market_residual_runner_matrix.csv"
            )
            if rejoin_rolling_step.get("returncode") != 0:
                gated_skipped_reason = "rejoin_rolling_model_comparison_failed"
            elif not rejoin_runner_matrix_csv.exists():
                gated_skipped_reason = "rejoin_market_residual_runner_matrix_missing"
            else:
                gated_skipped_reason = None
            if gated_skipped_reason is None:
                rejoin_pre_race_gated_step = run_command(
                    name="pre_race_gated_challenger_after_daemon_rejoins",
                    command=pre_race_gated_challenger_command(
                        runner_matrix_csv=rejoin_runner_matrix_csv,
                        output_dir=rejoin_pre_race_gated_dir,
                    ),
                    output_dir=output_dir,
                    timeout_seconds=args.timeout_seconds,
                )
                steps.append(rejoin_pre_race_gated_step)
                rejoin_pre_race_gated_report_path = (
                    rejoin_pre_race_gated_dir / "pre_race_gated_challenger_report.json"
                )
                rejoin_pre_race_gated_report = (
                    load_json(rejoin_pre_race_gated_report_path) or {}
                )
                rejoin_pre_race_gated_status = gated_challenger_status_from_report(
                    generated_at=generated_at,
                    packet_kind="pre_race_gated_challenger",
                    packet_dir=rejoin_pre_race_gated_dir,
                    report_path=rejoin_pre_race_gated_report_path,
                    packet_report=rejoin_pre_race_gated_report or None,
                    attempted=True,
                    returncode=rejoin_pre_race_gated_step.get("returncode"),
                )
                rejoin_time_split_gated_step = run_command(
                    name="time_split_gated_challenger_after_daemon_rejoins",
                    command=time_split_gated_challenger_command(
                        runner_matrix_csv=rejoin_runner_matrix_csv,
                        output_dir=rejoin_time_split_gated_dir,
                    ),
                    output_dir=output_dir,
                    timeout_seconds=args.timeout_seconds,
                )
                steps.append(rejoin_time_split_gated_step)
                rejoin_time_split_gated_report_path = (
                    rejoin_time_split_gated_dir
                    / "time_split_gated_challenger_report.json"
                )
                rejoin_time_split_gated_report = (
                    load_json(rejoin_time_split_gated_report_path) or {}
                )
                rejoin_time_split_gated_status = gated_challenger_status_from_report(
                    generated_at=generated_at,
                    packet_kind="time_split_gated_challenger",
                    packet_dir=rejoin_time_split_gated_dir,
                    report_path=rejoin_time_split_gated_report_path,
                    packet_report=rejoin_time_split_gated_report or None,
                    attempted=True,
                    returncode=rejoin_time_split_gated_step.get("returncode"),
                )
                rejoin_market_residual_step = run_command(
                    name="market_residual_challenger_after_daemon_rejoins",
                    command=market_residual_challenger_command(
                        runner_matrix_csv=rejoin_runner_matrix_csv,
                        output_dir=rejoin_market_residual_dir,
                    ),
                    output_dir=output_dir,
                    timeout_seconds=args.timeout_seconds,
                )
                steps.append(rejoin_market_residual_step)
                rejoin_market_residual_report_path = (
                    rejoin_market_residual_dir / "market_residual_challenger_report.json"
                )
                rejoin_market_residual_report = (
                    load_json(rejoin_market_residual_report_path) or {}
                )
                rejoin_market_residual_status = gated_challenger_status_from_report(
                    generated_at=generated_at,
                    packet_kind="market_residual_challenger",
                    packet_dir=rejoin_market_residual_dir,
                    report_path=rejoin_market_residual_report_path,
                    packet_report=rejoin_market_residual_report or None,
                    attempted=True,
                    returncode=rejoin_market_residual_step.get("returncode"),
                )
                residual_predictions_csv = (
                    rejoin_market_residual_dir / "cross_validated_race_predictions.csv"
                )
                if (
                    rejoin_market_residual_step.get("returncode") == 0
                    and residual_predictions_csv.exists()
                ):
                    rejoin_market_residual_regime_step = run_command(
                        name="market_residual_regime_audit_after_daemon_rejoins",
                        command=market_residual_regime_audit_command(
                            runner_matrix_csv=rejoin_runner_matrix_csv,
                            race_predictions_csv=residual_predictions_csv,
                            output_dir=rejoin_market_residual_regime_dir,
                        ),
                        output_dir=output_dir,
                        timeout_seconds=args.timeout_seconds,
                    )
                    steps.append(rejoin_market_residual_regime_step)
                    rejoin_market_residual_regime_report_path = (
                        rejoin_market_residual_regime_dir
                        / "market_residual_regime_audit_report.json"
                    )
                    rejoin_market_residual_regime_report = (
                        load_json(rejoin_market_residual_regime_report_path) or {}
                    )
                    rejoin_market_residual_regime_status = (
                        residual_regime_audit_status_from_report(
                            generated_at=generated_at,
                            packet_dir=rejoin_market_residual_regime_dir,
                            report_path=rejoin_market_residual_regime_report_path,
                            packet_report=(
                                rejoin_market_residual_regime_report or None
                            ),
                            attempted=True,
                            returncode=rejoin_market_residual_regime_step.get(
                                "returncode"
                            ),
                        )
                    )
                    rank_first_hypotheses_json = (
                        rejoin_market_residual_regime_dir / "next_hypotheses.json"
                    )
                    if (
                        rejoin_market_residual_regime_step.get("returncode") == 0
                        and rank_first_hypotheses_json.exists()
                    ):
                        rejoin_rank_first_hypothesis_gated_step = run_command(
                            name=(
                                "pre_race_rank_first_hypothesis_review_after_daemon_rejoins"
                            ),
                            command=pre_race_gated_challenger_command(
                                runner_matrix_csv=rejoin_runner_matrix_csv,
                                output_dir=rejoin_rank_first_hypothesis_gated_dir,
                                rank_first_hypotheses_json=rank_first_hypotheses_json,
                            ),
                            output_dir=output_dir,
                            timeout_seconds=args.timeout_seconds,
                        )
                        steps.append(rejoin_rank_first_hypothesis_gated_step)
                        rejoin_rank_first_hypothesis_gated_report_path = (
                            rejoin_rank_first_hypothesis_gated_dir
                            / "pre_race_gated_challenger_report.json"
                        )
                        rejoin_rank_first_hypothesis_gated_report = (
                            load_json(
                                rejoin_rank_first_hypothesis_gated_report_path
                            )
                            or {}
                        )
                        rejoin_rank_first_hypothesis_gated_status = (
                            gated_challenger_status_from_report(
                                generated_at=generated_at,
                                packet_kind=(
                                    "pre_race_rank_first_hypothesis_gated_challenger"
                                ),
                                packet_dir=rejoin_rank_first_hypothesis_gated_dir,
                                report_path=(
                                    rejoin_rank_first_hypothesis_gated_report_path
                                ),
                                packet_report=(
                                    rejoin_rank_first_hypothesis_gated_report or None
                                ),
                                attempted=True,
                                returncode=rejoin_rank_first_hypothesis_gated_step.get(
                                    "returncode"
                                ),
                            )
                        )
                    else:
                        rejoin_rank_first_hypothesis_gated_status = (
                            gated_challenger_status_from_report(
                                generated_at=generated_at,
                                packet_kind=(
                                    "pre_race_rank_first_hypothesis_gated_challenger"
                                ),
                                packet_dir=None,
                                report_path=None,
                                packet_report=None,
                                skipped_reason=(
                                    "rank_first_hypotheses_json_missing"
                                ),
                            )
                        )
                else:
                    rejoin_market_residual_regime_status = (
                        residual_regime_audit_status_from_report(
                            generated_at=generated_at,
                            packet_dir=None,
                            report_path=None,
                            packet_report=None,
                            skipped_reason=(
                                "market_residual_challenger_predictions_missing"
                            ),
                        )
                    )
                    rejoin_rank_first_hypothesis_gated_status = (
                        gated_challenger_status_from_report(
                            generated_at=generated_at,
                            packet_kind="pre_race_rank_first_hypothesis_gated_challenger",
                            packet_dir=None,
                            report_path=None,
                            packet_report=None,
                            skipped_reason=(
                                "market_residual_challenger_predictions_missing"
                            ),
                        )
                    )
            else:
                rejoin_pre_race_gated_status = gated_challenger_status_from_report(
                    generated_at=generated_at,
                    packet_kind="pre_race_gated_challenger",
                    packet_dir=None,
                    report_path=None,
                    packet_report=None,
                    skipped_reason=gated_skipped_reason,
                )
                rejoin_time_split_gated_status = gated_challenger_status_from_report(
                    generated_at=generated_at,
                    packet_kind="time_split_gated_challenger",
                    packet_dir=None,
                    report_path=None,
                    packet_report=None,
                    skipped_reason=gated_skipped_reason,
                )
                rejoin_market_residual_status = gated_challenger_status_from_report(
                    generated_at=generated_at,
                    packet_kind="market_residual_challenger",
                    packet_dir=None,
                    report_path=None,
                    packet_report=None,
                    skipped_reason=gated_skipped_reason,
                )
                rejoin_market_residual_regime_status = (
                    residual_regime_audit_status_from_report(
                        generated_at=generated_at,
                        packet_dir=None,
                        report_path=None,
                        packet_report=None,
                        skipped_reason=gated_skipped_reason,
                    )
                )
                rejoin_rank_first_hypothesis_gated_status = (
                    gated_challenger_status_from_report(
                        generated_at=generated_at,
                        packet_kind="pre_race_rank_first_hypothesis_gated_challenger",
                        packet_dir=None,
                        report_path=None,
                        packet_report=None,
                        skipped_reason=gated_skipped_reason,
                    )
                )
            rejoin_rank_first_hypothesis_watchlist_step = run_command(
                name="rank_first_hypothesis_watchlist_after_daemon_rejoins",
                command=rank_first_hypothesis_watchlist_command(
                    evidence_root=evidence_root,
                    output_dir=rejoin_rank_first_hypothesis_watchlist_dir,
                ),
                output_dir=output_dir,
                timeout_seconds=args.timeout_seconds,
            )
            steps.append(rejoin_rank_first_hypothesis_watchlist_step)
            rejoin_rank_first_hypothesis_watchlist_report_path = (
                rejoin_rank_first_hypothesis_watchlist_dir
                / "rank_first_hypothesis_watchlist_report.json"
            )
            rejoin_rank_first_hypothesis_watchlist_report = (
                load_json(rejoin_rank_first_hypothesis_watchlist_report_path) or {}
            )
            rejoin_rank_first_hypothesis_watchlist_status = (
                rank_first_hypothesis_watchlist_status_from_report(
                    generated_at=generated_at,
                    packet_dir=rejoin_rank_first_hypothesis_watchlist_dir,
                    report_path=rejoin_rank_first_hypothesis_watchlist_report_path,
                    packet_report=(
                        rejoin_rank_first_hypothesis_watchlist_report or None
                    ),
                    attempted=True,
                    returncode=rejoin_rank_first_hypothesis_watchlist_step.get(
                        "returncode"
                    ),
                )
            )
            best_rejoin_unified_report = autopilot.best_unified_evidence_report_path(
                rejoin_comparison_report_paths
            )
            timing_aligned_rerun_sources = (
                timing_aligned_rerun_source_paths_from_autopilot(autopilot_output_dir)
            )
            if best_rejoin_unified_report is not None:
                rejoin_odds_gate_report_path = (
                    autopilot.odds_research_gate_report_path_from_snapshot_status(
                        shadow_odds_snapshot
                    )
                )
                rejoin_high_accuracy_step = run_command(
                    name="high_accuracy_refinement_after_daemon_rejoins",
                    command=autopilot.high_accuracy_refinement_packet_command(
                        unified_evidence_report=best_rejoin_unified_report,
                        output_dir=rejoin_high_accuracy_dir,
                        stage2_predictions=(
                            daily_shadow_run_dir / "stage2_shadow_predictions.jsonl"
                            if daily_shadow_run_dir
                            else None
                        ),
                        odds_augmented_report=(
                            rejoin_rolling_dir / "rolling_model_comparison_report.json"
                        ),
                        odds_gate_report=rejoin_odds_gate_report_path,
                        backlog_unified_evidence_status=(
                            best_aggregate_unified_status_path
                            or output_dir / "rejoin_unified_evidence_datasets_status.json"
                        ),
                        reserve_substitution_preflight=(
                            rejoin_reserve_substitution_preflight_report_path
                        ),
                        timing_aligned_rerun_plan=timing_aligned_rerun_sources[
                            "timing_aligned_rerun_plan"
                        ],
                        timing_aligned_rerun_execution_status=timing_aligned_rerun_sources[
                            "timing_aligned_rerun_execution_status"
                        ],
                    ),
                    output_dir=output_dir,
                    timeout_seconds=args.timeout_seconds,
                )
                steps.append(rejoin_high_accuracy_step)
                rejoin_high_accuracy_report = (
                    load_json(
                        rejoin_high_accuracy_dir / "high_accuracy_refinement_packet.json"
                    )
                    or {}
                )
                rejoin_high_accuracy_status = autopilot.build_high_accuracy_refinement_status(
                    generated_at=generated_at,
                    packet_dir=rejoin_high_accuracy_dir,
                    packet_report=rejoin_high_accuracy_report or None,
                    attempted=True,
                    returncode=rejoin_high_accuracy_step.get("returncode"),
                )
            else:
                rejoin_high_accuracy_status = autopilot.build_high_accuracy_refinement_status(
                    generated_at=generated_at,
                    packet_dir=None,
                    packet_report=None,
                    skipped_reason="rejoin_unified_evidence_eligible_reports_missing",
                )
            rejoin_rolling_report_path = (
                rejoin_rolling_dir / "rolling_model_comparison_report.json"
            )
            rejoin_pre_race_gated_report_path = (
                rejoin_pre_race_gated_dir / "pre_race_gated_challenger_report.json"
            )
            rejoin_high_accuracy_gate_path = (
                rejoin_high_accuracy_dir / "promotion_pr_gate.json"
            )
            if (
                rejoin_rolling_report_path.exists()
                and rejoin_pre_race_gated_report_path.exists()
                and rejoin_high_accuracy_gate_path.exists()
            ):
                rejoin_promotion_distance_step = run_command(
                    name="promotion_distance_after_daemon_rejoins",
                    command=promotion_distance_report_command(
                        rolling_report=rejoin_rolling_report_path,
                        pre_race_gated_report=rejoin_pre_race_gated_report_path,
                        high_accuracy_gate=rejoin_high_accuracy_gate_path,
                        output_dir=rejoin_promotion_distance_dir,
                    ),
                    output_dir=output_dir,
                    timeout_seconds=args.timeout_seconds,
                )
                steps.append(rejoin_promotion_distance_step)
                rejoin_promotion_distance_report_path = (
                    rejoin_promotion_distance_dir / "promotion_distance_report.json"
                )
                rejoin_promotion_distance_report = (
                    load_json(rejoin_promotion_distance_report_path) or {}
                )
                rejoin_promotion_distance_status = promotion_distance_status_from_report(
                    generated_at=generated_at,
                    packet_dir=rejoin_promotion_distance_dir,
                    report_path=rejoin_promotion_distance_report_path,
                    packet_report=rejoin_promotion_distance_report or None,
                    attempted=True,
                    returncode=rejoin_promotion_distance_step.get("returncode"),
                )
                rejoin_high_accuracy_with_promotion_step = run_command(
                    name="high_accuracy_refinement_after_promotion_distance",
                    command=autopilot.high_accuracy_refinement_packet_command(
                        unified_evidence_report=best_rejoin_unified_report,
                        output_dir=rejoin_high_accuracy_after_promotion_dir,
                        stage2_predictions=(
                            daily_shadow_run_dir / "stage2_shadow_predictions.jsonl"
                            if daily_shadow_run_dir
                            else None
                        ),
                        odds_augmented_report=rejoin_rolling_report_path,
                        odds_gate_report=rejoin_odds_gate_report_path,
                        backlog_unified_evidence_status=(
                            best_aggregate_unified_status_path
                            or output_dir / "rejoin_unified_evidence_datasets_status.json"
                        ),
                        promotion_distance_report=rejoin_promotion_distance_report_path,
                        reserve_substitution_preflight=(
                            rejoin_reserve_substitution_preflight_report_path
                        ),
                        timing_aligned_rerun_plan=timing_aligned_rerun_sources[
                            "timing_aligned_rerun_plan"
                        ],
                        timing_aligned_rerun_execution_status=timing_aligned_rerun_sources[
                            "timing_aligned_rerun_execution_status"
                        ],
                    ),
                    output_dir=output_dir,
                    timeout_seconds=args.timeout_seconds,
                )
                steps.append(rejoin_high_accuracy_with_promotion_step)
                rejoin_high_accuracy_report = (
                    load_json(
                        rejoin_high_accuracy_after_promotion_dir
                        / "high_accuracy_refinement_packet.json"
                    )
                    or {}
                )
                rejoin_high_accuracy_status = autopilot.build_high_accuracy_refinement_status(
                    generated_at=generated_at,
                    packet_dir=rejoin_high_accuracy_after_promotion_dir,
                    packet_report=rejoin_high_accuracy_report or None,
                    attempted=True,
                    returncode=rejoin_high_accuracy_with_promotion_step.get("returncode"),
                )
            else:
                rejoin_promotion_distance_status = promotion_distance_status_from_report(
                    generated_at=generated_at,
                    packet_dir=None,
                    report_path=None,
                    packet_report=None,
                    skipped_reason="promotion_distance_source_reports_missing",
                )
        else:
            rejoin_rolling_status = autopilot.build_rolling_model_comparison_status(
                generated_at=generated_at,
                comparison_dir=None,
                comparison_report=None,
                skipped_reason="rejoin_unified_evidence_reports_missing",
            )
            rejoin_high_accuracy_status = autopilot.build_high_accuracy_refinement_status(
                generated_at=generated_at,
                packet_dir=None,
                packet_report=None,
                skipped_reason="rejoin_unified_evidence_reports_missing",
            )
            rejoin_pre_race_gated_status = gated_challenger_status_from_report(
                generated_at=generated_at,
                packet_kind="pre_race_gated_challenger",
                packet_dir=None,
                report_path=None,
                packet_report=None,
                skipped_reason="rejoin_unified_evidence_reports_missing",
            )
            rejoin_time_split_gated_status = gated_challenger_status_from_report(
                generated_at=generated_at,
                packet_kind="time_split_gated_challenger",
                packet_dir=None,
                report_path=None,
                packet_report=None,
                skipped_reason="rejoin_unified_evidence_reports_missing",
            )
            rejoin_market_residual_status = gated_challenger_status_from_report(
                generated_at=generated_at,
                packet_kind="market_residual_challenger",
                packet_dir=None,
                report_path=None,
                packet_report=None,
                skipped_reason="rejoin_unified_evidence_reports_missing",
            )
            rejoin_market_residual_regime_status = (
                residual_regime_audit_status_from_report(
                    generated_at=generated_at,
                    packet_dir=None,
                    report_path=None,
                    packet_report=None,
                    skipped_reason="rejoin_unified_evidence_reports_missing",
                )
            )
            rejoin_rank_first_hypothesis_gated_status = gated_challenger_status_from_report(
                generated_at=generated_at,
                packet_kind="pre_race_rank_first_hypothesis_gated_challenger",
                packet_dir=None,
                report_path=None,
                packet_report=None,
                skipped_reason="rejoin_unified_evidence_reports_missing",
            )
            rejoin_rank_first_hypothesis_watchlist_status = (
                rank_first_hypothesis_watchlist_status_from_report(
                    generated_at=generated_at,
                    packet_dir=None,
                    report_path=None,
                    packet_report=None,
                    skipped_reason="rejoin_unified_evidence_reports_missing",
                )
            )
            rejoin_promotion_distance_status = promotion_distance_status_from_report(
                generated_at=generated_at,
                packet_dir=None,
                report_path=None,
                packet_report=None,
                skipped_reason="rejoin_unified_evidence_reports_missing",
            )
        rejoin_rolling_status = annotate_rejoin_skipped_status(
            rejoin_rolling_status, rejoin_unified_status
        )
        rejoin_high_accuracy_status = annotate_rejoin_skipped_status(
            rejoin_high_accuracy_status, rejoin_unified_status
        )
        rejoin_pre_race_gated_status = annotate_rejoin_skipped_status(
            rejoin_pre_race_gated_status, rejoin_unified_status
        )
        rejoin_rank_first_hypothesis_gated_status = annotate_rejoin_skipped_status(
            rejoin_rank_first_hypothesis_gated_status, rejoin_unified_status
        )
        rejoin_rank_first_hypothesis_watchlist_status = annotate_rejoin_skipped_status(
            rejoin_rank_first_hypothesis_watchlist_status, rejoin_unified_status
        )
        rejoin_time_split_gated_status = annotate_rejoin_skipped_status(
            rejoin_time_split_gated_status, rejoin_unified_status
        )
        rejoin_market_residual_status = annotate_rejoin_skipped_status(
            rejoin_market_residual_status, rejoin_unified_status
        )
        rejoin_market_residual_regime_status = annotate_rejoin_skipped_status(
            rejoin_market_residual_regime_status, rejoin_unified_status
        )
        rejoin_promotion_distance_status = annotate_rejoin_skipped_status(
            rejoin_promotion_distance_status, rejoin_unified_status
        )
        write_json(
            output_dir / "rolling_model_comparison_after_daemon_rejoins_status.json",
            rejoin_rolling_status,
        )
        write_json(
            output_dir / "high_accuracy_refinement_after_daemon_rejoins_status.json",
            rejoin_high_accuracy_status,
        )
        write_json(
            output_dir / "pre_race_gated_challenger_after_daemon_rejoins_status.json",
            rejoin_pre_race_gated_status,
        )
        write_json(
            output_dir
            / "pre_race_rank_first_hypothesis_review_after_daemon_rejoins_status.json",
            rejoin_rank_first_hypothesis_gated_status,
        )
        write_json(
            output_dir / "rank_first_hypothesis_watchlist_after_daemon_rejoins_status.json",
            rejoin_rank_first_hypothesis_watchlist_status,
        )
        write_json(
            output_dir / "time_split_gated_challenger_after_daemon_rejoins_status.json",
            rejoin_time_split_gated_status,
        )
        write_json(
            output_dir / "market_residual_challenger_after_daemon_rejoins_status.json",
            rejoin_market_residual_status,
        )
        write_json(
            output_dir / "market_residual_regime_audit_after_daemon_rejoins_status.json",
            rejoin_market_residual_regime_status,
        )
        write_json(
            output_dir / "promotion_distance_after_daemon_rejoins_status.json",
            rejoin_promotion_distance_status,
        )
    except LockBusy as exc:
        lock_validation = {
            "schema_version": "shadow_autopilot_lock_validation_v1",
            "lock_path": relpath(lock_path),
            "lock_acquired": False,
            "status": "SKIPPED_LOCK_HELD",
            "details": exc.payload,
        }
        write_json(output_dir / "lock_validation.json", lock_validation)
        write_text(output_dir / "final_status.txt", "PARTIAL_DAEMONIZATION\n")
        result = lock_held_daemon_run_report(
            run_id=run_id,
            generated_at=generated_at,
            current_time=current_time,
            output_dir=output_dir,
            lock_path=lock_path,
            lock_details=exc.payload,
            odds_capture_state_path=odds_state_path,
            odds_capture_state=load_json(odds_state_path) or {},
        )
        write_json(output_dir / "daemon_run_report.json", result)
        write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
        return result
    except Exception as exc:
        lock_validation = {
            "schema_version": "shadow_autopilot_lock_validation_v1",
            "lock_path": relpath(lock_path),
            "lock_acquired": bool(lock_payload),
            "status": "ERROR",
            "details": {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        }
        write_json(output_dir / "lock_validation.json", lock_validation)
        result = {
            **completed_daemon_run_report_envelope(
                run_id=run_id,
                generated_at=generated_at,
                current_time=current_time,
                output_dir=output_dir,
                final_verdict="PARTIAL_DAEMONIZATION",
            ),
            "status": "PARTIAL_DAEMONIZATION",
            "runtime_action": "CHECK_DAEMON_EXCEPTION",
            "readiness_decision": "READY_FOR_RELIABILITY_REVIEW",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "autopilot_output_dir": relpath(autopilot_output_dir),
            "daily_shadow_run_dir": relpath(daily_shadow_run_dir),
            "step_count": len(steps),
            "steps": list(steps),
            "lock_path": relpath(lock_path),
            "lock_validation_status": lock_validation["status"],
            "protected_paths_unchanged_or_allowed": False,
            "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        }
        write_text(output_dir / "final_status.txt", "PARTIAL_DAEMONIZATION\n")
        write_json(output_dir / "daemon_run_report.json", result)
        write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
        return result
    finally:
        if lock_payload:
            lock_release = release_lock(lock_path, run_id)

    protected_after = protected_hashes()
    protected_paths_unchanged = protected_before == protected_after
    protected_changed_paths = sorted(
        key
        for key, before_value in protected_before.items()
        if protected_after.get(key) != before_value
    )

    aggregate_metrics = load_json(aggregate_dir / "aggregate_forward_metrics.json")
    aggregate_calibration = load_json(aggregate_dir / "aggregate_calibration_review.json")
    aggregate_box_bias = load_json(aggregate_dir / "aggregate_box_bias_review.json")
    status_report = load_json(status_dir / "forward_shadow_status_report.json")
    latest_join_dir = latest_artifact(evidence_root, "forward_shadow_result_join_", "shadow_forward_metrics.json")
    join_metrics = load_json(latest_join_dir / "shadow_forward_metrics.json") if latest_join_dir else None
    join_pending = load_json(latest_join_dir / "pending_results.json") if latest_join_dir else None
    join_unsafe = load_json(latest_join_dir / "unsafe_result_matches.json") if latest_join_dir else None
    autopilot_daily_status_path = (
        autopilot_output_dir / "DAILY_STATUS.json" if autopilot_output_dir else None
    )
    autopilot_daily_status = load_json(autopilot_daily_status_path)
    feature_activation_gate = feature_activation_gate_status_from_autopilot(autopilot_output_dir)
    if shadow_odds_snapshot is None:
        shadow_odds_snapshot = shadow_odds_snapshot_status_from_autopilot(autopilot_output_dir)
    live_odds_capture_packet = live_odds_capture_packet_from_autopilot(autopilot_output_dir)
    autonomous_live_odds_capture_status = (
        autonomous_live_odds_capture_status_from_autopilot(autopilot_output_dir)
    )
    autonomous_official_result_capture_status = (
        autonomous_official_result_capture_status_from_autopilot(autopilot_output_dir)
    )
    autonomous_odds_inserted_rows = int_or_zero(
        autonomous_live_odds_capture_status.get("inserted_live_odds_rows")
    )
    autonomous_official_result_evidence_inserted_rows = int_or_zero(
        autonomous_official_result_capture_status.get(
            "official_result_evidence_inserted_race_rows"
        )
    ) + int_or_zero(
        autonomous_official_result_capture_status.get(
            "official_result_evidence_inserted_runner_rows"
        )
    )
    allowed_odds_db_change = (
        bool(protected_changed_paths)
        and autonomous_odds_inserted_rows > 0
        and set(protected_changed_paths).issubset({relpath(args.db)})
    )
    allowed_official_result_evidence_db_change = (
        bool(protected_changed_paths)
        and autonomous_official_result_evidence_inserted_rows > 0
        and set(protected_changed_paths).issubset({relpath(args.db)})
    )
    protected_paths_unchanged_or_allowed = (
        protected_paths_unchanged
        or allowed_odds_db_change
        or allowed_official_result_evidence_db_change
    )
    if odds_capture_state_publish.get("status") != "PUBLISHED":
        odds_capture_state_publish = publish_full_daemon_odds_capture_state(
            state_path=args.odds_capture_state_path,
            generated_at=generated_at,
            run_id=run_id,
            output_dir=output_dir,
            autopilot_output_dir=autopilot_output_dir,
            odds_status=autonomous_live_odds_capture_status,
        )
    next_prejump_refresh_window = next_prejump_refresh_window_from_autopilot(autopilot_output_dir)
    prejump_metadata_status = prejump_metadata_status_from_daily_run(daily_shadow_run_dir)
    prejump_metadata_trend = build_prejump_metadata_trend_report(
        evidence_root=evidence_root,
        output_dir=output_dir,
        generated_at=generated_at,
    )
    odds_coverage = build_read_only_odds_coverage_report(
        db_path=args.db,
        output_dir=output_dir,
        generated_at=generated_at,
    )

    sources = {
        "daemon_output_dir": relpath(output_dir),
        "autopilot_output_dir": relpath(autopilot_output_dir),
        "aggregate_dir": relpath(aggregate_dir),
        "status_dir": relpath(status_dir),
        "latest_join_dir": relpath(latest_join_dir),
        "daily_shadow_run_dir": relpath(daily_shadow_run_dir),
        "service_file": service_info.get("service_path"),
        "timer_file": service_info.get("timer_path"),
        "lock_path": relpath(lock_path),
        "odds_coverage_report": odds_coverage.get("report_path"),
        "live_odds_capture_approval_packet": live_odds_capture_packet.get("packet_path"),
        "rejoin_unified_evidence_datasets_status": relpath(
            output_dir / "rejoin_unified_evidence_datasets_status.json"
        ),
        "rolling_model_comparison_after_daemon_rejoins_status": relpath(
            output_dir / "rolling_model_comparison_after_daemon_rejoins_status.json"
        ),
        "high_accuracy_refinement_after_daemon_rejoins_status": relpath(
            output_dir / "high_accuracy_refinement_after_daemon_rejoins_status.json"
        ),
        "pre_race_gated_challenger_after_daemon_rejoins_status": relpath(
            output_dir / "pre_race_gated_challenger_after_daemon_rejoins_status.json"
        ),
        "pre_race_rank_first_hypothesis_review_after_daemon_rejoins_status": relpath(
            output_dir
            / "pre_race_rank_first_hypothesis_review_after_daemon_rejoins_status.json"
        ),
        "rank_first_hypothesis_watchlist_after_daemon_rejoins_status": relpath(
            output_dir
            / "rank_first_hypothesis_watchlist_after_daemon_rejoins_status.json"
        ),
        "time_split_gated_challenger_after_daemon_rejoins_status": relpath(
            output_dir / "time_split_gated_challenger_after_daemon_rejoins_status.json"
        ),
        "market_residual_challenger_after_daemon_rejoins_status": relpath(
            output_dir / "market_residual_challenger_after_daemon_rejoins_status.json"
        ),
        "market_residual_regime_audit_after_daemon_rejoins_status": relpath(
            output_dir
            / "market_residual_regime_audit_after_daemon_rejoins_status.json"
        ),
        "promotion_distance_after_daemon_rejoins_status": relpath(
            output_dir / "promotion_distance_after_daemon_rejoins_status.json"
        ),
    }
    sources.update(
        timing_aligned_rerun_source_artifacts_from_autopilot(autopilot_output_dir)
    )
    if shadow_odds_snapshot:
        sources["shadow_odds_snapshot_status"] = shadow_odds_snapshot.get("status_path")
        sources["shadow_odds_snapshot_output_dir"] = shadow_odds_snapshot.get("output_dir")
        sources["odds_research_gate_report"] = shadow_odds_snapshot.get(
            "odds_research_gate_report_path"
        )
    sources["autonomous_official_result_capture_status"] = (
        autonomous_official_result_capture_status.get("status_path")
    )
    sources["autonomous_official_result_capture_output_dir"] = (
        autonomous_official_result_capture_status.get("output_dir")
    )
    if next_prejump_refresh_window:
        sources["refresh_report"] = next_prejump_refresh_window.get("report_path")
    if prejump_metadata_status:
        sources["prejump_metadata_report"] = prejump_metadata_status.get("report_path")
    sources["prejump_metadata_trend_report"] = relpath(
        output_dir / "prejump_metadata_trend_report.json"
    )
    if feature_activation_gate:
        sources["feature_activation_gate_status"] = feature_activation_gate.get("status_path")
        sources["feature_activation_gate_output_dir"] = feature_activation_gate.get("output_dir")
    dashboard = autopilot.build_dashboard(
        generated_at=generated_at,
        aggregate_metrics=aggregate_metrics,
        join_metrics=join_metrics,
        aggregate_calibration=aggregate_calibration,
        aggregate_box_bias=aggregate_box_bias,
        status_report=status_report,
        sources=sources,
        odds_snapshot_status=shadow_odds_snapshot,
        live_odds_capture_packet=live_odds_capture_packet,
        autonomous_live_odds_capture_status=autonomous_live_odds_capture_status,
        autonomous_official_result_capture_status=autonomous_official_result_capture_status,
    )
    if daily_manifest:
        score_live_manifest = daily_manifest.get("score_live_manifest") or {}
        dashboard["all_missing_train_policy"] = daily_manifest.get("all_missing_train_policy") or score_live_manifest.get(
            "all_missing_train_policy"
        )
        dashboard["tgr_enabled"] = daily_manifest.get("tgr_enabled")
        if dashboard["tgr_enabled"] is None:
            dashboard["tgr_enabled"] = score_live_manifest.get("tgr_enabled")
    if feature_activation_gate:
        dashboard["feature_activation_gate"] = feature_activation_gate
        dashboard["feature_activation_gate_status"] = feature_activation_gate.get("status")
        dashboard["kept_quarantined_features"] = feature_activation_gate.get("kept_quarantined_features") or []
        dashboard["activation_allowed_features"] = feature_activation_gate.get("activation_allowed_features") or []
    dashboard["odds_coverage"] = odds_coverage
    dashboard["live_odds_capture_approval"] = live_odds_capture_packet
    dashboard["live_odds_capture_approval_status"] = live_odds_capture_packet.get("status")
    dashboard["live_odds_capture_verified_prejump_races"] = live_odds_capture_packet.get(
        "verified_prejump_race_count"
    )
    dashboard["rejoin_unified_evidence_datasets"] = rejoin_unified_status
    dashboard["rejoin_unified_evidence_status"] = rejoin_unified_status.get("status")
    dashboard["rejoin_unified_evidence_status_reason"] = rejoin_unified_status.get(
        "status_reason"
    )
    dashboard["rejoin_unified_evidence_evaluated_candidate_count"] = (
        rejoin_unified_status.get("evaluated_dataset_candidate_count")
    )
    dashboard["rejoin_unified_evidence_dataset_count"] = rejoin_unified_status.get(
        "dataset_count"
    )
    dashboard["rejoin_unified_evidence_skipped_dataset_count"] = (
        rejoin_unified_status.get("skipped_dataset_count")
    )
    dashboard["rejoin_unified_evidence_skip_reason_counts"] = (
        rejoin_unified_status.get("skip_reason_counts") or {}
    )
    dashboard["rejoin_unified_evidence_failure_reason_counts"] = (
        rejoin_unified_status.get("failure_reason_counts") or {}
    )
    dashboard["rejoin_unified_evidence_eligible_rows"] = rejoin_unified_status.get(
        "unified_evidence_eligible_rows"
    )
    dashboard["rejoin_unified_evidence_artifact_odds_rows_seen"] = (
        rejoin_unified_status.get("artifact_odds_rows_seen")
    )
    dashboard["rejoin_unified_evidence_artifact_odds_rows_accepted"] = (
        rejoin_unified_status.get("artifact_odds_rows_accepted")
    )
    dashboard["rejoin_unified_evidence_artifact_odds_rows_rejected"] = (
        rejoin_unified_status.get("artifact_odds_rows_rejected")
    )
    dashboard["rejoin_unified_evidence_artifact_odds_rejection_reason_counts"] = (
        rejoin_unified_status.get("artifact_odds_rejection_reason_counts") or {}
    )
    dashboard["rejoin_unified_evidence_rows_with_artifact_shadow_odds"] = (
        rejoin_unified_status.get("rows_with_artifact_shadow_odds")
    )
    dashboard["rejoin_unified_evidence_rows_with_artifact_shadow_odds_candidates"] = (
        rejoin_unified_status.get("rows_with_artifact_shadow_odds_candidates")
    )
    dashboard["rejoin_unified_rejected_live_odds_candidate_count"] = (
        rejoin_unified_status.get("rejected_live_odds_candidate_count")
    )
    dashboard["rejoin_unified_rows_with_rejected_live_odds_candidates"] = (
        rejoin_unified_status.get("rows_with_rejected_live_odds_candidates")
    )
    dashboard["rejoin_unified_rejected_live_odds_candidate_reason_counts"] = (
        rejoin_unified_status.get("rejected_live_odds_candidate_reason_counts") or {}
    )
    dashboard["join_eligibility_preview_dataset_count"] = rejoin_unified_status.get(
        "join_eligibility_preview_dataset_count"
    )
    dashboard["join_eligibility_preview_unified_eligible_rows"] = (
        rejoin_unified_status.get("join_eligibility_preview_unified_eligible_rows")
    )
    dashboard["join_eligibility_preview_packet_accepted_races"] = (
        rejoin_unified_status.get("join_eligibility_preview_packet_accepted_races")
    )
    dashboard["join_eligibility_preview_packet_present_races"] = (
        rejoin_unified_status.get("join_eligibility_preview_packet_present_races")
    )
    dashboard["join_eligibility_preview_missing_race_ids"] = rejoin_unified_status.get(
        "join_eligibility_preview_missing_race_ids"
    )
    dashboard["rejoin_rolling_model_comparison"] = rejoin_rolling_status
    dashboard["rejoin_rolling_model_comparison_status"] = rejoin_rolling_status.get(
        "status"
    )
    dashboard["rejoin_rolling_model_comparison_sample_races"] = rejoin_rolling_status.get(
        "sample_race_count"
    )
    dashboard["rejoin_high_accuracy_refinement"] = rejoin_high_accuracy_status
    dashboard["rejoin_high_accuracy_refinement_status"] = rejoin_high_accuracy_status.get(
        "status"
    )
    dashboard["rejoin_pre_race_gated_challenger"] = rejoin_pre_race_gated_status
    dashboard["rejoin_pre_race_gated_challenger_status"] = (
        rejoin_pre_race_gated_status.get("status")
    )
    dashboard["rejoin_pre_race_gated_challenger_accepted_races"] = (
        rejoin_pre_race_gated_status.get("accepted_race_count")
    )
    dashboard["rejoin_pre_race_gated_challenger_promotion_ready"] = (
        rejoin_pre_race_gated_status.get("promotion_ready")
    )
    dashboard["rejoin_pre_race_predeclared_residual_candidate_status"] = (
        rejoin_pre_race_gated_status.get("predeclared_residual_candidate_status")
    )
    dashboard["rejoin_pre_race_predeclared_residual_triggered_races"] = (
        rejoin_pre_race_gated_status.get("predeclared_residual_triggered_race_count")
    )
    dashboard[
        "rejoin_pre_race_predeclared_residual_minimum_triggered_races_for_directional_read"
    ] = rejoin_pre_race_gated_status.get(
        "predeclared_residual_minimum_triggered_races_for_directional_read"
    )
    dashboard["rejoin_pre_race_predeclared_residual_directional_read_ready"] = (
        rejoin_pre_race_gated_status.get(
            "predeclared_residual_directional_read_ready"
        )
    )
    dashboard["rejoin_rank_first_hypothesis_gated_challenger"] = (
        rejoin_rank_first_hypothesis_gated_status
    )
    dashboard["rejoin_rank_first_hypothesis_review_status"] = (
        rejoin_rank_first_hypothesis_gated_status.get(
            "rank_first_hypothesis_review_status"
        )
    )
    dashboard["rejoin_rank_first_hypothesis_candidate_count"] = (
        rejoin_rank_first_hypothesis_gated_status.get(
            "rank_first_hypothesis_candidate_count"
        )
    )
    dashboard["rejoin_rank_first_hypothesis_evaluated_candidate_count"] = (
        rejoin_rank_first_hypothesis_gated_status.get(
            "rank_first_hypothesis_evaluated_candidate_count"
        )
    )
    dashboard["rejoin_rank_first_hypothesis_best_candidate_key"] = (
        rejoin_rank_first_hypothesis_gated_status.get(
            "rank_first_hypothesis_best_candidate_key"
        )
    )
    dashboard["rejoin_rank_first_hypothesis_best_triggered_race_count"] = (
        rejoin_rank_first_hypothesis_gated_status.get(
            "rank_first_hypothesis_best_triggered_race_count"
        )
    )
    dashboard[
        "rejoin_rank_first_hypothesis_minimum_triggered_races_for_directional_read"
    ] = rejoin_rank_first_hypothesis_gated_status.get(
        "rank_first_hypothesis_minimum_triggered_races_for_directional_read"
    )
    dashboard["rejoin_rank_first_hypothesis_directional_read_ready"] = (
        rejoin_rank_first_hypothesis_gated_status.get(
            "rank_first_hypothesis_directional_read_ready"
        )
    )
    dashboard["rejoin_time_split_gated_challenger"] = rejoin_time_split_gated_status
    dashboard["rejoin_time_split_gated_challenger_status"] = (
        rejoin_time_split_gated_status.get("status")
    )
    dashboard["rejoin_time_split_gated_challenger_test_races"] = (
        rejoin_time_split_gated_status.get("time_split_test_race_count")
    )
    dashboard["rejoin_time_split_gated_challenger_promotion_ready"] = (
        rejoin_time_split_gated_status.get("promotion_ready")
    )
    dashboard["rejoin_market_residual_challenger"] = rejoin_market_residual_status
    dashboard["rejoin_market_residual_challenger_status"] = (
        rejoin_market_residual_status.get("status")
    )
    dashboard["rejoin_market_residual_challenger_promotion_ready"] = (
        rejoin_market_residual_status.get("promotion_ready")
    )
    dashboard["rejoin_market_residual_regime_audit"] = (
        rejoin_market_residual_regime_status
    )
    dashboard["rejoin_market_residual_regime_audit_status"] = (
        rejoin_market_residual_regime_status.get("status")
    )
    dashboard["rejoin_market_residual_regime_audit_promotion_ready"] = (
        rejoin_market_residual_regime_status.get("promotion_ready")
    )
    dashboard["rejoin_market_residual_rank_first_hypothesis_status"] = (
        rejoin_market_residual_regime_status.get("rank_first_hypothesis_status")
    )
    dashboard["rejoin_market_residual_rank_first_help_regimes"] = (
        rejoin_market_residual_regime_status.get(
            "pre_race_rank_first_help_regime_count"
        )
    )
    dashboard["rejoin_market_residual_logloss_only_help_regimes"] = (
        rejoin_market_residual_regime_status.get(
            "pre_race_logloss_only_help_regime_count"
        )
    )
    dashboard["rejoin_rank_first_hypothesis_watchlist"] = (
        rejoin_rank_first_hypothesis_watchlist_status
    )
    dashboard["rejoin_rank_first_hypothesis_watchlist_status"] = (
        rejoin_rank_first_hypothesis_watchlist_status.get("status")
    )
    dashboard["rejoin_rank_first_hypothesis_watchlist_candidate_count"] = (
        rejoin_rank_first_hypothesis_watchlist_status.get("candidate_count")
    )
    dashboard[
        "rejoin_rank_first_hypothesis_watchlist_directional_ready_candidate_count"
    ] = rejoin_rank_first_hypothesis_watchlist_status.get(
        "directional_ready_candidate_count"
    )
    dashboard["rejoin_rank_first_hypothesis_watchlist_best_candidate"] = (
        rejoin_rank_first_hypothesis_watchlist_status.get("best_candidate_key")
    )
    dashboard["rejoin_rank_first_hypothesis_watchlist_best_status"] = (
        rejoin_rank_first_hypothesis_watchlist_status.get("best_candidate_status")
    )
    dashboard["rejoin_rank_first_hypothesis_watchlist_best_distinct_samples"] = (
        rejoin_rank_first_hypothesis_watchlist_status.get(
            "best_candidate_distinct_sample_count"
        )
    )
    dashboard["rejoin_promotion_distance"] = rejoin_promotion_distance_status
    dashboard["rejoin_promotion_distance_status"] = (
        rejoin_promotion_distance_status.get("status")
    )
    dashboard["rejoin_promotion_distance_promotion_ready"] = (
        rejoin_promotion_distance_status.get("promotion_ready")
    )
    dashboard["rejoin_promotion_distance_blockers"] = (
        rejoin_promotion_distance_status.get("blockers") or []
    )
    dashboard["autonomous_live_odds_capture"] = autonomous_live_odds_capture_status
    dashboard["autonomous_live_odds_capture_status"] = autonomous_live_odds_capture_status.get(
        "status"
    )
    dashboard["autonomous_live_odds_capture_ready_count"] = (
        autonomous_live_odds_capture_status.get("ready_count")
    )
    dashboard["autonomous_live_odds_capture_inserted_rows"] = autonomous_odds_inserted_rows
    dashboard["odds_capture_next_meaningful_action"] = (
        odds_capture_state_publish.get("next_meaningful_action")
    )
    dashboard["odds_capture_next_meaningful_action_at"] = (
        odds_capture_state_publish.get("next_meaningful_action_at")
    )
    dashboard["autonomous_official_result_capture"] = (
        autonomous_official_result_capture_status
    )
    dashboard["autonomous_official_result_capture_status"] = (
        autonomous_official_result_capture_status.get("status")
    )
    dashboard["autonomous_official_result_capture_attempted"] = (
        autonomous_official_result_capture_status.get("attempted", False)
    )
    dashboard["autonomous_official_result_candidate_count"] = (
        autonomous_official_result_capture_status.get("candidate_count", 0)
    )
    dashboard["autonomous_official_result_race_rows"] = (
        autonomous_official_result_capture_status.get("official_result_race_rows", 0)
    )
    dashboard["autonomous_official_result_runner_rows"] = (
        autonomous_official_result_capture_status.get("official_result_runner_rows", 0)
    )
    dashboard["autonomous_official_result_quarantine_rows"] = (
        autonomous_official_result_capture_status.get("quarantine_rows", 0)
    )
    dashboard["autonomous_official_result_quarantined_race_ids"] = (
        autonomous_official_result_capture_status.get("quarantined_race_ids", [])
    )
    dashboard["autonomous_official_result_quarantine_reason_counts"] = (
        autonomous_official_result_capture_status.get("quarantine_reason_counts", {})
    )
    dashboard["autonomous_official_result_quarantine_error_counts"] = (
        autonomous_official_result_capture_status.get("quarantine_error_counts", {})
    )
    dashboard[
        "autonomous_official_result_quarantine_result_boxes_not_in_participants_counts"
    ] = autonomous_official_result_capture_status.get(
        "quarantine_result_boxes_not_in_participants_counts", {}
    )
    dashboard["autonomous_official_result_quarantine_runner_set_mismatch_samples"] = (
        autonomous_official_result_capture_status.get(
            "quarantine_runner_set_mismatch_samples", []
        )
    )
    dashboard["autonomous_official_result_skipped_reason_counts"] = (
        autonomous_official_result_capture_status.get("skipped_reason_counts", {})
    )
    dashboard["autonomous_official_result_awaiting_jump_race_count"] = (
        autonomous_official_result_capture_status.get("awaiting_jump_race_count", 0)
    )
    dashboard["autonomous_official_result_awaiting_jump_race_ids"] = (
        autonomous_official_result_capture_status.get("awaiting_jump_race_ids", [])
    )
    dashboard["autonomous_official_result_awaiting_jump_next_recheck_after_local"] = (
        autonomous_official_result_capture_status.get(
            "awaiting_jump_next_recheck_after_local"
        )
    )
    dashboard["autonomous_official_result_evidence_inserted_rows"] = (
        autonomous_official_result_evidence_inserted_rows
    )
    dashboard["autonomous_official_result_evidence_db_ingest_status"] = (
        autonomous_official_result_capture_status.get(
            "official_result_evidence_db_ingest_status"
        )
    )
    dashboard["autonomous_official_result_evidence_db_execute"] = (
        autonomous_official_result_capture_status.get(
            "official_result_evidence_db_execute", False
        )
    )
    dashboard["autonomous_official_result_evidence_db_write_performed"] = (
        autonomous_official_result_capture_status.get(
            "official_result_evidence_db_write_performed", False
        )
    )
    dashboard["autonomous_official_result_evidence_valid_race_rows"] = (
        autonomous_official_result_capture_status.get(
            "official_result_evidence_valid_race_rows", 0
        )
    )
    dashboard["autonomous_official_result_evidence_valid_runner_rows"] = (
        autonomous_official_result_capture_status.get(
            "official_result_evidence_valid_runner_rows", 0
        )
    )
    dashboard["autonomous_official_result_evidence_blocked_race_rows"] = (
        autonomous_official_result_capture_status.get(
            "official_result_evidence_blocked_race_rows", 0
        )
    )
    dashboard["autonomous_official_result_evidence_blocked_runner_rows"] = (
        autonomous_official_result_capture_status.get(
            "official_result_evidence_blocked_runner_rows", 0
        )
    )
    dashboard["autonomous_official_result_evidence_inserted_race_rows"] = (
        autonomous_official_result_capture_status.get(
            "official_result_evidence_inserted_race_rows", 0
        )
    )
    dashboard["autonomous_official_result_evidence_inserted_runner_rows"] = (
        autonomous_official_result_capture_status.get(
            "official_result_evidence_inserted_runner_rows", 0
        )
    )
    dashboard["autonomous_official_result_evidence_blocker_reason_counts"] = (
        autonomous_official_result_capture_status.get(
            "official_result_evidence_blocker_reason_counts", {}
        )
        or {}
    )
    dashboard.update(
        live_odds_backlog_operational_fields(autonomous_official_result_capture_status)
    )
    if shadow_odds_snapshot:
        dashboard["shadow_odds_snapshot"] = shadow_odds_snapshot
        dashboard["shadow_odds_snapshot_status"] = shadow_odds_snapshot.get("status")
        dashboard["shadow_odds_snapshot_valid_prejump_rows"] = shadow_odds_snapshot.get(
            "valid_pre_jump_dog_odds_rows"
        )
        dashboard["shadow_odds_snapshot_races_after_feature_freeze"] = shadow_odds_snapshot.get(
            "races_with_post_feature_freeze_odds_rows",
            0,
        )
        dashboard["shadow_odds_snapshot_ev_output_rows"] = shadow_odds_snapshot.get(
            "ev_output_rows",
            0,
        )
        dashboard["odds_research_gate_status"] = shadow_odds_snapshot.get(
            "odds_research_gate_status"
        )
        dashboard["odds_research_gate_report"] = shadow_odds_snapshot.get(
            "odds_research_gate_report_path"
        )
        dashboard["odds_research_gate_complete_valid_prejump_odds_races"] = (
            shadow_odds_snapshot.get(
                "odds_research_gate_complete_valid_prejump_odds_races"
            )
        )
        dashboard["odds_research_gate_minimum_complete_valid_prejump_odds_races"] = (
            shadow_odds_snapshot.get(
                "odds_research_gate_minimum_complete_valid_prejump_odds_races"
            )
        )
        dashboard["odds_research_gate_source_url_coverage_pct"] = (
            shadow_odds_snapshot.get("odds_research_gate_source_url_coverage_pct")
        )
        dashboard["odds_research_gate_blocker_counts"] = shadow_odds_snapshot.get(
            "odds_research_gate_blocker_counts"
        ) or {}
        dashboard["odds_research_next_action"] = shadow_odds_snapshot.get(
            "odds_research_next_action"
        )
        dashboard["timing_aligned_prediction_rerun_required"] = (
            shadow_odds_snapshot.get("timing_aligned_prediction_rerun_required", False)
        )
        dashboard["timing_aligned_prediction_rerun_race_count"] = (
            shadow_odds_snapshot.get("timing_aligned_prediction_rerun_race_count", 0)
        )
        dashboard["timing_aligned_prediction_rerun_race_ids"] = (
            shadow_odds_snapshot.get("timing_aligned_prediction_rerun_race_ids") or []
        )
        dashboard["timing_aligned_prediction_rerun_reason_counts"] = (
            shadow_odds_snapshot.get("timing_aligned_prediction_rerun_reason_counts")
            or {}
        )
    dashboard["odds_used_for_shadow_scoring"] = False
    dashboard["odds_capture_performed"] = autonomous_odds_inserted_rows > 0
    if next_prejump_refresh_window:
        dashboard["next_prejump_refresh_window"] = next_prejump_refresh_window
        dashboard["next_prejump_refresh_status"] = next_prejump_refresh_window.get("status")
        dashboard["recommended_rerun_after_local"] = next_prejump_refresh_window.get(
            "recommended_rerun_after_local"
        )
    if prejump_metadata_status:
        dashboard["prejump_metadata_status"] = prejump_metadata_status.get("status")
        dashboard["prejump_metadata"] = prejump_metadata_status
    dashboard["prejump_metadata_trend"] = prejump_metadata_trend
    dashboard["prejump_metadata_trend_status"] = prejump_metadata_trend.get("status")
    dashboard["prejump_metadata_verified_rate"] = prejump_metadata_trend.get(
        "verified_metadata_rate"
    )
    result_join_status = autopilot.build_result_join_status(
        generated_at=generated_at,
        latest_join_dir=latest_join_dir,
        aggregate_dir=aggregate_dir,
        join_metrics=join_metrics,
        aggregate_metrics=aggregate_metrics,
        pending_payload=join_pending,
        unsafe_payload=join_unsafe,
    )
    join_history = autopilot.build_join_history(evidence_root)
    aggregate_timeseries = autopilot.build_aggregate_timeseries(evidence_root)
    daily_status = autopilot.build_daily_status(
        generated_at=generated_at,
        daily_manifest=daily_manifest,
        result_join_status=result_join_status,
        dashboard=dashboard,
        timeseries=aggregate_timeseries,
        readiness={"decision": "NEED_MORE_RESULTS"},
        odds_snapshot_status=shadow_odds_snapshot,
        live_odds_capture_packet=live_odds_capture_packet,
        autonomous_live_odds_capture_status=autonomous_live_odds_capture_status,
        autonomous_official_result_capture_status=autonomous_official_result_capture_status,
    )
    apply_autopilot_cycle_status_to_daily_status(
        daily_status,
        autopilot_daily_status_path=autopilot_daily_status_path,
        autopilot_daily_status=autopilot_daily_status,
    )
    if feature_activation_gate:
        daily_status["feature_activation_gate_status"] = feature_activation_gate.get("status")
        daily_status["kept_quarantined_features"] = feature_activation_gate.get("kept_quarantined_features") or []
        daily_status["activation_allowed_features"] = feature_activation_gate.get("activation_allowed_features") or []
    daily_status["odds_coverage_status"] = odds_coverage.get("status")
    daily_status["live_odds_capture_approval_status"] = live_odds_capture_packet.get("status")
    daily_status["live_odds_capture_verified_prejump_races"] = live_odds_capture_packet.get(
        "verified_prejump_race_count"
    )
    daily_status["live_odds_capture_window_offsets_minutes"] = live_odds_capture_packet.get(
        "capture_window_offsets_minutes"
    )
    daily_status["live_odds_capture_can_capture_now"] = live_odds_capture_packet.get(
        "can_capture_live_odds_now",
        False,
    )
    daily_status["rejoin_unified_evidence_status"] = rejoin_unified_status.get("status")
    daily_status["rejoin_unified_evidence_status_reason"] = rejoin_unified_status.get(
        "status_reason"
    )
    daily_status["rejoin_unified_evidence_evaluated_candidate_count"] = (
        rejoin_unified_status.get("evaluated_dataset_candidate_count")
    )
    daily_status["rejoin_unified_evidence_dataset_count"] = rejoin_unified_status.get(
        "dataset_count"
    )
    daily_status["rejoin_unified_evidence_skipped_dataset_count"] = (
        rejoin_unified_status.get("skipped_dataset_count")
    )
    daily_status["rejoin_unified_evidence_skip_reason_counts"] = (
        rejoin_unified_status.get("skip_reason_counts") or {}
    )
    daily_status["rejoin_unified_evidence_failure_reason_counts"] = (
        rejoin_unified_status.get("failure_reason_counts") or {}
    )
    daily_status["rejoin_unified_evidence_eligible_rows"] = rejoin_unified_status.get(
        "unified_evidence_eligible_rows"
    )
    daily_status["rejoin_unified_evidence_artifact_odds_rows_seen"] = (
        rejoin_unified_status.get("artifact_odds_rows_seen")
    )
    daily_status["rejoin_unified_evidence_artifact_odds_rows_accepted"] = (
        rejoin_unified_status.get("artifact_odds_rows_accepted")
    )
    daily_status["rejoin_unified_evidence_artifact_odds_rows_rejected"] = (
        rejoin_unified_status.get("artifact_odds_rows_rejected")
    )
    daily_status[
        "rejoin_unified_evidence_artifact_odds_rejection_reason_counts"
    ] = rejoin_unified_status.get("artifact_odds_rejection_reason_counts") or {}
    daily_status["rejoin_unified_evidence_rows_with_artifact_shadow_odds"] = (
        rejoin_unified_status.get("rows_with_artifact_shadow_odds")
    )
    daily_status["rejoin_unified_evidence_rows_with_artifact_shadow_odds_candidates"] = (
        rejoin_unified_status.get("rows_with_artifact_shadow_odds_candidates")
    )
    daily_status["rejoin_unified_rejected_live_odds_candidate_count"] = (
        rejoin_unified_status.get("rejected_live_odds_candidate_count")
    )
    daily_status["rejoin_unified_rows_with_rejected_live_odds_candidates"] = (
        rejoin_unified_status.get("rows_with_rejected_live_odds_candidates")
    )
    daily_status["rejoin_unified_rejected_live_odds_candidate_reason_counts"] = (
        rejoin_unified_status.get("rejected_live_odds_candidate_reason_counts") or {}
    )
    daily_status["join_eligibility_preview_dataset_count"] = rejoin_unified_status.get(
        "join_eligibility_preview_dataset_count"
    )
    daily_status["join_eligibility_preview_unified_eligible_rows"] = (
        rejoin_unified_status.get("join_eligibility_preview_unified_eligible_rows")
    )
    daily_status["join_eligibility_preview_packet_accepted_races"] = (
        rejoin_unified_status.get("join_eligibility_preview_packet_accepted_races")
    )
    daily_status["join_eligibility_preview_packet_present_races"] = (
        rejoin_unified_status.get("join_eligibility_preview_packet_present_races")
    )
    apply_best_aggregate_unified_evidence_to_daily_status(
        daily_status,
        best_status_path=best_aggregate_unified_status_path,
        best_status=best_aggregate_unified_status,
    )
    daily_status["rejoin_rolling_model_comparison_status"] = rejoin_rolling_status.get(
        "status"
    )
    daily_status["rejoin_rolling_model_comparison_sample_races"] = rejoin_rolling_status.get(
        "sample_race_count"
    )
    daily_status["rejoin_high_accuracy_refinement_status"] = (
        rejoin_high_accuracy_status.get("status")
    )
    daily_status.update(
        rejoin_high_accuracy_timing_source_fields(
            rejoin_high_accuracy_status,
            prefix="rejoin_high_accuracy_",
        )
    )
    daily_status["rejoin_pre_race_gated_challenger_status"] = (
        rejoin_pre_race_gated_status.get("status")
    )
    daily_status["rejoin_pre_race_gated_challenger_accepted_races"] = (
        rejoin_pre_race_gated_status.get("accepted_race_count")
    )
    daily_status["rejoin_pre_race_gated_challenger_promotion_ready"] = (
        rejoin_pre_race_gated_status.get("promotion_ready")
    )
    daily_status["rejoin_pre_race_predeclared_residual_candidate_status"] = (
        rejoin_pre_race_gated_status.get("predeclared_residual_candidate_status")
    )
    daily_status["rejoin_pre_race_predeclared_residual_triggered_races"] = (
        rejoin_pre_race_gated_status.get("predeclared_residual_triggered_race_count")
    )
    daily_status[
        "rejoin_pre_race_predeclared_residual_minimum_triggered_races_for_directional_read"
    ] = rejoin_pre_race_gated_status.get(
        "predeclared_residual_minimum_triggered_races_for_directional_read"
    )
    daily_status["rejoin_pre_race_predeclared_residual_directional_read_ready"] = (
        rejoin_pre_race_gated_status.get(
            "predeclared_residual_directional_read_ready"
        )
    )
    daily_status["rejoin_rank_first_hypothesis_review_status"] = (
        rejoin_rank_first_hypothesis_gated_status.get(
            "rank_first_hypothesis_review_status"
        )
    )
    daily_status["rejoin_rank_first_hypothesis_candidate_count"] = (
        rejoin_rank_first_hypothesis_gated_status.get(
            "rank_first_hypothesis_candidate_count"
        )
    )
    daily_status["rejoin_rank_first_hypothesis_evaluated_candidate_count"] = (
        rejoin_rank_first_hypothesis_gated_status.get(
            "rank_first_hypothesis_evaluated_candidate_count"
        )
    )
    daily_status["rejoin_rank_first_hypothesis_best_candidate_key"] = (
        rejoin_rank_first_hypothesis_gated_status.get(
            "rank_first_hypothesis_best_candidate_key"
        )
    )
    daily_status["rejoin_rank_first_hypothesis_best_triggered_races"] = (
        rejoin_rank_first_hypothesis_gated_status.get(
            "rank_first_hypothesis_best_triggered_race_count"
        )
    )
    daily_status[
        "rejoin_rank_first_hypothesis_minimum_triggered_races_for_directional_read"
    ] = rejoin_rank_first_hypothesis_gated_status.get(
        "rank_first_hypothesis_minimum_triggered_races_for_directional_read"
    )
    daily_status["rejoin_rank_first_hypothesis_directional_read_ready"] = (
        rejoin_rank_first_hypothesis_gated_status.get(
            "rank_first_hypothesis_directional_read_ready"
        )
    )
    daily_status["rejoin_time_split_gated_challenger_status"] = (
        rejoin_time_split_gated_status.get("status")
    )
    daily_status["rejoin_time_split_gated_challenger_test_races"] = (
        rejoin_time_split_gated_status.get("time_split_test_race_count")
    )
    daily_status["rejoin_time_split_gated_challenger_promotion_ready"] = (
        rejoin_time_split_gated_status.get("promotion_ready")
    )
    daily_status["rejoin_market_residual_challenger_status"] = (
        rejoin_market_residual_status.get("status")
    )
    daily_status["rejoin_market_residual_challenger_promotion_ready"] = (
        rejoin_market_residual_status.get("promotion_ready")
    )
    daily_status["rejoin_market_residual_regime_audit_status"] = (
        rejoin_market_residual_regime_status.get("status")
    )
    daily_status["rejoin_market_residual_regime_audit_promotion_ready"] = (
        rejoin_market_residual_regime_status.get("promotion_ready")
    )
    daily_status["rejoin_market_residual_rank_first_hypothesis_status"] = (
        rejoin_market_residual_regime_status.get("rank_first_hypothesis_status")
    )
    daily_status["rejoin_market_residual_rank_first_help_regimes"] = (
        rejoin_market_residual_regime_status.get(
            "pre_race_rank_first_help_regime_count"
        )
    )
    daily_status["rejoin_market_residual_logloss_only_help_regimes"] = (
        rejoin_market_residual_regime_status.get(
            "pre_race_logloss_only_help_regime_count"
        )
    )
    daily_status["rejoin_rank_first_hypothesis_watchlist_status"] = (
        rejoin_rank_first_hypothesis_watchlist_status.get("status")
    )
    daily_status["rejoin_rank_first_hypothesis_watchlist_candidate_count"] = (
        rejoin_rank_first_hypothesis_watchlist_status.get("candidate_count")
    )
    daily_status[
        "rejoin_rank_first_hypothesis_watchlist_directional_ready_candidate_count"
    ] = rejoin_rank_first_hypothesis_watchlist_status.get(
        "directional_ready_candidate_count"
    )
    daily_status["rejoin_rank_first_hypothesis_watchlist_best_candidate"] = (
        rejoin_rank_first_hypothesis_watchlist_status.get("best_candidate_key")
    )
    daily_status["rejoin_rank_first_hypothesis_watchlist_best_status"] = (
        rejoin_rank_first_hypothesis_watchlist_status.get("best_candidate_status")
    )
    daily_status["rejoin_rank_first_hypothesis_watchlist_best_distinct_samples"] = (
        rejoin_rank_first_hypothesis_watchlist_status.get(
            "best_candidate_distinct_sample_count"
        )
    )
    daily_status["rejoin_promotion_distance_status"] = (
        rejoin_promotion_distance_status.get("status")
    )
    daily_status["rejoin_promotion_distance_promotion_ready"] = (
        rejoin_promotion_distance_status.get("promotion_ready")
    )
    daily_status["rejoin_promotion_distance_blockers"] = (
        rejoin_promotion_distance_status.get("blockers") or []
    )
    daily_status["rejoin_promotion_distance_source_exclusion_reason_counts"] = (
        rejoin_promotion_distance_status.get("source_exclusion_reason_counts") or {}
    )
    daily_status["rejoin_promotion_distance_source_odds_exclusion_reason_counts"] = (
        rejoin_promotion_distance_status.get("source_odds_exclusion_reason_counts")
        or {}
    )
    daily_status[
        "rejoin_promotion_distance_source_official_result_evidence_db_missing_race_ids"
    ] = (
        rejoin_promotion_distance_status.get(
            "source_official_result_evidence_db_missing_race_ids"
        )
        or []
    )
    daily_status[
        "rejoin_promotion_distance_source_official_result_evidence_db_requested_race_count"
    ] = rejoin_promotion_distance_status.get(
        "source_official_result_evidence_db_requested_race_count"
    )
    daily_status[
        "rejoin_promotion_distance_source_official_result_evidence_db_races_with_rows"
    ] = (
        rejoin_promotion_distance_status.get(
            "source_official_result_evidence_db_races_with_rows"
        )
        or []
    )
    daily_status["rejoin_promotion_distance_source_official_result_runner_paths"] = (
        rejoin_promotion_distance_status.get("source_official_result_runner_paths")
        or []
    )
    daily_status[
        "rejoin_promotion_distance_official_result_coverage_requested_race_count"
    ] = rejoin_promotion_distance_status.get(
        "official_result_coverage_requested_race_count"
    )
    daily_status[
        "rejoin_promotion_distance_official_result_coverage_requested_race_count_source"
    ] = rejoin_promotion_distance_status.get(
        "official_result_coverage_requested_race_count_source"
    )
    daily_status[
        "rejoin_promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids"
    ] = rejoin_promotion_distance_status.get(
        "official_result_coverage_legacy_requested_race_count_without_ids"
    )
    daily_status[
        "rejoin_promotion_distance_official_result_coverage_races_with_rows_count"
    ] = rejoin_promotion_distance_status.get(
        "official_result_coverage_races_with_rows_count"
    )
    daily_status[
        "rejoin_promotion_distance_official_result_coverage_missing_race_count"
    ] = rejoin_promotion_distance_status.get(
        "official_result_coverage_missing_race_count"
    )
    daily_status[
        "rejoin_promotion_distance_official_result_coverage_missing_exclusion_count"
    ] = rejoin_promotion_distance_status.get(
        "official_result_coverage_missing_exclusion_count"
    )
    daily_status["rejoin_promotion_distance_official_result_runner_path_count"] = (
        rejoin_promotion_distance_status.get("official_result_runner_path_count")
    )
    daily_status[
        "rejoin_promotion_distance_official_result_runner_paths_source_field"
    ] = rejoin_promotion_distance_status.get("official_result_runner_paths_source_field")
    daily_status["prejump_metadata_trend_status"] = prejump_metadata_trend.get("status")
    daily_status["prejump_metadata_verified_rate"] = prejump_metadata_trend.get(
        "verified_metadata_rate"
    )
    if shadow_odds_snapshot:
        daily_status["shadow_odds_snapshot_status"] = shadow_odds_snapshot.get("status")
        daily_status["shadow_odds_snapshot_valid_prejump_rows"] = shadow_odds_snapshot.get(
            "valid_pre_jump_dog_odds_rows"
        )
        daily_status["shadow_odds_snapshot_races_after_feature_freeze"] = shadow_odds_snapshot.get(
            "races_with_post_feature_freeze_odds_rows",
            0,
        )
        daily_status["shadow_odds_snapshot_ev_output_rows"] = shadow_odds_snapshot.get(
            "ev_output_rows",
            0,
        )
        daily_status["odds_research_gate_status"] = shadow_odds_snapshot.get(
            "odds_research_gate_status"
        )
        daily_status["odds_research_gate_report"] = shadow_odds_snapshot.get(
            "odds_research_gate_report_path"
        )
        daily_status["odds_research_gate_complete_valid_prejump_odds_races"] = (
            shadow_odds_snapshot.get(
                "odds_research_gate_complete_valid_prejump_odds_races"
            )
        )
        daily_status["odds_research_gate_minimum_complete_valid_prejump_odds_races"] = (
            shadow_odds_snapshot.get(
                "odds_research_gate_minimum_complete_valid_prejump_odds_races"
            )
        )
        daily_status["odds_research_gate_source_url_coverage_pct"] = (
            shadow_odds_snapshot.get("odds_research_gate_source_url_coverage_pct")
        )
        daily_status["odds_research_gate_blocker_counts"] = shadow_odds_snapshot.get(
            "odds_research_gate_blocker_counts"
        ) or {}
        daily_status["odds_research_next_action"] = shadow_odds_snapshot.get(
            "odds_research_next_action"
        )
        daily_status["timing_aligned_prediction_rerun_required"] = (
            shadow_odds_snapshot.get("timing_aligned_prediction_rerun_required", False)
        )
        daily_status["timing_aligned_prediction_rerun_race_count"] = (
            shadow_odds_snapshot.get("timing_aligned_prediction_rerun_race_count", 0)
        )
        daily_status["timing_aligned_prediction_rerun_race_ids"] = (
            shadow_odds_snapshot.get("timing_aligned_prediction_rerun_race_ids") or []
        )
        daily_status["timing_aligned_prediction_rerun_reason_counts"] = (
            shadow_odds_snapshot.get("timing_aligned_prediction_rerun_reason_counts")
            or {}
        )
    daily_status["autonomous_live_odds_capture_status"] = (
        autonomous_live_odds_capture_status.get("status")
    )
    daily_status["autonomous_live_odds_capture_attempted"] = (
        autonomous_live_odds_capture_status.get("attempted", False)
    )
    daily_status["autonomous_live_odds_capture_execute"] = (
        autonomous_live_odds_capture_status.get("execute", False)
    )
    daily_status["autonomous_live_odds_capture_ready_count"] = (
        autonomous_live_odds_capture_status.get("ready_count", 0)
    )
    daily_status["autonomous_live_odds_inserted_rows"] = autonomous_odds_inserted_rows
    daily_status["odds_capture_next_meaningful_action"] = (
        odds_capture_state_publish.get("next_meaningful_action")
    )
    daily_status["odds_capture_next_meaningful_action_at"] = (
        odds_capture_state_publish.get("next_meaningful_action_at")
    )
    daily_status.update(
        t2_odds_capture_surface_fields(
            autonomous_live_odds_capture_status=autonomous_live_odds_capture_status,
            odds_capture_state_publish=odds_capture_state_publish,
        )
    )
    daily_status["odds_capture_performed"] = autonomous_odds_inserted_rows > 0
    daily_status["odds_used_for_shadow_scoring"] = False
    daily_status["autonomous_official_result_capture_status"] = (
        autonomous_official_result_capture_status.get("status")
    )
    daily_status["autonomous_official_result_capture_attempted"] = (
        autonomous_official_result_capture_status.get("attempted", False)
    )
    daily_status["autonomous_official_result_candidate_count"] = (
        autonomous_official_result_capture_status.get("candidate_count", 0)
    )
    daily_status["autonomous_official_result_race_rows"] = (
        autonomous_official_result_capture_status.get("official_result_race_rows", 0)
    )
    daily_status["autonomous_official_result_runner_rows"] = (
        autonomous_official_result_capture_status.get("official_result_runner_rows", 0)
    )
    daily_status["autonomous_official_result_quarantine_rows"] = (
        autonomous_official_result_capture_status.get("quarantine_rows", 0)
    )
    daily_status["autonomous_official_result_quarantined_race_ids"] = (
        autonomous_official_result_capture_status.get("quarantined_race_ids", [])
    )
    daily_status["autonomous_official_result_quarantine_reason_counts"] = (
        autonomous_official_result_capture_status.get("quarantine_reason_counts", {})
    )
    daily_status["autonomous_official_result_quarantine_error_counts"] = (
        autonomous_official_result_capture_status.get("quarantine_error_counts", {})
    )
    daily_status[
        "autonomous_official_result_quarantine_result_boxes_not_in_participants_counts"
    ] = autonomous_official_result_capture_status.get(
        "quarantine_result_boxes_not_in_participants_counts", {}
    )
    daily_status["autonomous_official_result_quarantine_runner_set_mismatch_samples"] = (
        autonomous_official_result_capture_status.get(
            "quarantine_runner_set_mismatch_samples", []
        )
    )
    daily_status["autonomous_official_result_skipped_reason_counts"] = (
        autonomous_official_result_capture_status.get("skipped_reason_counts", {})
    )
    daily_status["autonomous_official_result_awaiting_jump_race_count"] = (
        autonomous_official_result_capture_status.get("awaiting_jump_race_count", 0)
    )
    daily_status["autonomous_official_result_awaiting_jump_race_ids"] = (
        autonomous_official_result_capture_status.get("awaiting_jump_race_ids", [])
    )
    daily_status["autonomous_official_result_awaiting_jump_next_recheck_after_local"] = (
        autonomous_official_result_capture_status.get(
            "awaiting_jump_next_recheck_after_local"
        )
    )
    daily_status["autonomous_official_result_evidence_inserted_rows"] = (
        autonomous_official_result_evidence_inserted_rows
    )
    daily_status["autonomous_official_result_evidence_db_ingest_status"] = (
        autonomous_official_result_capture_status.get(
            "official_result_evidence_db_ingest_status"
        )
    )
    daily_status["autonomous_official_result_evidence_db_execute"] = (
        autonomous_official_result_capture_status.get(
            "official_result_evidence_db_execute", False
        )
    )
    daily_status["autonomous_official_result_evidence_db_write_performed"] = (
        autonomous_official_result_capture_status.get(
            "official_result_evidence_db_write_performed", False
        )
    )
    daily_status["autonomous_official_result_evidence_valid_race_rows"] = (
        autonomous_official_result_capture_status.get(
            "official_result_evidence_valid_race_rows", 0
        )
    )
    daily_status["autonomous_official_result_evidence_valid_runner_rows"] = (
        autonomous_official_result_capture_status.get(
            "official_result_evidence_valid_runner_rows", 0
        )
    )
    daily_status["autonomous_official_result_evidence_blocked_race_rows"] = (
        autonomous_official_result_capture_status.get(
            "official_result_evidence_blocked_race_rows", 0
        )
    )
    daily_status["autonomous_official_result_evidence_blocked_runner_rows"] = (
        autonomous_official_result_capture_status.get(
            "official_result_evidence_blocked_runner_rows", 0
        )
    )
    daily_status["autonomous_official_result_evidence_inserted_race_rows"] = (
        autonomous_official_result_capture_status.get(
            "official_result_evidence_inserted_race_rows", 0
        )
    )
    daily_status["autonomous_official_result_evidence_inserted_runner_rows"] = (
        autonomous_official_result_capture_status.get(
            "official_result_evidence_inserted_runner_rows", 0
        )
    )
    daily_status["autonomous_official_result_evidence_blocker_reason_counts"] = (
        autonomous_official_result_capture_status.get(
            "official_result_evidence_blocker_reason_counts", {}
        )
        or {}
    )
    daily_status.update(
        live_odds_backlog_operational_fields(autonomous_official_result_capture_status)
    )
    if next_prejump_refresh_window:
        daily_status["next_prejump_refresh_status"] = next_prejump_refresh_window.get("status")
        daily_status["recommended_rerun_after_local"] = next_prejump_refresh_window.get(
            "recommended_rerun_after_local"
        )
        daily_status["next_prejump_race"] = next_prejump_refresh_window.get("next_race")
    if prejump_metadata_status:
        daily_status["prejump_metadata_status"] = prejump_metadata_status.get("status")
        daily_status["prejump_metadata_verified_eligible"] = prejump_metadata_status.get(
            "eligible_with_verified_prejump_metadata"
        )
        daily_status["prejump_metadata_eligible_count"] = prejump_metadata_status.get(
            "eligible_count"
        )
    readiness = daemon_readiness(
        generated_at=generated_at,
        dashboard=dashboard,
        target_joined_races=args.target_joined_races,
    )

    service_verify = systemd_verify(service_path, timer_path, output_dir)
    service_files_present = service_path.exists() and timer_path.exists()
    systemd_deployment = systemd_deployment_status(
        expected_service_exec_fragments=expected_service_exec_fragments_for_run(args)
    )
    service_validation = {
        "schema_version": "shadow_autopilot_service_validation_v1",
        "service_file": relpath(service_path),
        "timer_file": relpath(timer_path),
        "service_files_present": service_files_present,
        "timer_frequency": DEFAULT_TIMER_FREQUENCY,
        "timer_calendar": DEFAULT_TIMER_ON_CALENDAR,
        "systemd_timer": True,
        "cron_fallback_required": False,
        "overlap_prevention": ["systemd_oneshot_unit", "daemon_lockfile"],
        "timeout_seconds": args.timeout_seconds,
        "deployment_status": systemd_deployment.get("deployment_status"),
        "deployment_ready": systemd_deployment.get("deployment_ready"),
        "service_installed": systemd_deployment.get("service_installed"),
        "timer_installed": systemd_deployment.get("timer_installed"),
        "timer_enabled": systemd_deployment.get("timer_enabled"),
        "timer_active": systemd_deployment.get("timer_active"),
        "service_command_matches_expected": systemd_deployment.get(
            "service_command_matches_expected"
        ),
        "required_service_exec_fragments": systemd_deployment.get(
            "required_service_exec_fragments"
        ),
        "missing_service_exec_fragments": systemd_deployment.get(
            "missing_service_exec_fragments"
        ),
        "systemd_deployment": systemd_deployment,
        "systemd_analyze_verify": service_verify,
    }
    operational_validation = {
        "schema_version": "shadow_autopilot_operational_validation_v1",
        "refresh_cycle_invoked": any(step.get("name") == "autopilot_cycle" for step in steps),
        "score_cycle_invoked": any(step.get("name") == "autopilot_cycle" for step in steps),
        "join_cycle_invoked": bool(automated_join_report.get("rejoin_attempt_count") is not None),
        "rejoin_unified_evidence_status": rejoin_unified_status.get("status"),
        "rejoin_unified_evidence_dataset_count": rejoin_unified_status.get(
            "dataset_count"
        ),
        **rejoin_unified_operational_diagnostic_fields(rejoin_unified_status),
        "rejoin_unified_evidence_eligible_rows": rejoin_unified_status.get(
            "unified_evidence_eligible_rows"
        ),
        "rejoin_unified_evidence_artifact_odds_rows_seen": (
            rejoin_unified_status.get("artifact_odds_rows_seen")
        ),
        "rejoin_unified_evidence_artifact_odds_rows_accepted": (
            rejoin_unified_status.get("artifact_odds_rows_accepted")
        ),
        "rejoin_unified_evidence_artifact_odds_rows_rejected": (
            rejoin_unified_status.get("artifact_odds_rows_rejected")
        ),
        "rejoin_unified_evidence_artifact_odds_rejection_reason_counts": (
            rejoin_unified_status.get("artifact_odds_rejection_reason_counts") or {}
        ),
        "rejoin_unified_evidence_rows_with_artifact_shadow_odds": (
            rejoin_unified_status.get("rows_with_artifact_shadow_odds")
        ),
        "rejoin_unified_evidence_rows_with_artifact_shadow_odds_candidates": (
            rejoin_unified_status.get("rows_with_artifact_shadow_odds_candidates")
        ),
        "rejoin_unified_rejected_live_odds_candidate_count": (
            rejoin_unified_status.get("rejected_live_odds_candidate_count")
        ),
        "rejoin_unified_rows_with_rejected_live_odds_candidates": (
            rejoin_unified_status.get("rows_with_rejected_live_odds_candidates")
        ),
        "rejoin_unified_rejected_live_odds_candidate_reason_counts": (
            rejoin_unified_status.get("rejected_live_odds_candidate_reason_counts") or {}
        ),
        "rolling_model_comparison_after_rejoin_status": rejoin_rolling_status.get(
            "status"
        ),
        "high_accuracy_refinement_after_rejoin_status": rejoin_high_accuracy_status.get(
            "status"
        ),
        **rejoin_high_accuracy_timing_source_fields(
            rejoin_high_accuracy_status,
            prefix="high_accuracy_refinement_after_rejoin_",
        ),
        "pre_race_gated_challenger_after_rejoin_status": (
            rejoin_pre_race_gated_status.get("status")
        ),
        "pre_race_rank_first_hypothesis_review_after_rejoin_status": (
            rejoin_rank_first_hypothesis_gated_status.get(
                "rank_first_hypothesis_review_status"
            )
        ),
        "pre_race_rank_first_hypothesis_review_after_rejoin_best_candidate": (
            rejoin_rank_first_hypothesis_gated_status.get(
                "rank_first_hypothesis_best_candidate_key"
            )
        ),
        "pre_race_rank_first_hypothesis_review_after_rejoin_triggered_races": (
            rejoin_rank_first_hypothesis_gated_status.get(
                "rank_first_hypothesis_best_triggered_race_count"
            )
        ),
        "pre_race_rank_first_hypothesis_review_after_rejoin_directional_read_ready": (
            rejoin_rank_first_hypothesis_gated_status.get(
                "rank_first_hypothesis_directional_read_ready"
            )
        ),
        "time_split_gated_challenger_after_rejoin_status": (
            rejoin_time_split_gated_status.get("status")
        ),
        "market_residual_challenger_after_rejoin_status": (
            rejoin_market_residual_status.get("status")
        ),
        "market_residual_regime_audit_after_rejoin_status": (
            rejoin_market_residual_regime_status.get("status")
        ),
        "market_residual_rank_first_hypothesis_after_rejoin_status": (
            rejoin_market_residual_regime_status.get("rank_first_hypothesis_status")
        ),
        "market_residual_rank_first_help_regimes_after_rejoin": (
            rejoin_market_residual_regime_status.get(
                "pre_race_rank_first_help_regime_count"
            )
        ),
        "market_residual_logloss_only_help_regimes_after_rejoin": (
            rejoin_market_residual_regime_status.get(
                "pre_race_logloss_only_help_regime_count"
            )
        ),
        "rank_first_hypothesis_watchlist_after_rejoin_status": (
            rejoin_rank_first_hypothesis_watchlist_status.get("status")
        ),
        "rank_first_hypothesis_watchlist_after_rejoin_directional_ready_candidates": (
            rejoin_rank_first_hypothesis_watchlist_status.get(
                "directional_ready_candidate_count"
            )
        ),
        "rank_first_hypothesis_watchlist_after_rejoin_best_candidate": (
            rejoin_rank_first_hypothesis_watchlist_status.get("best_candidate_key")
        ),
        "rank_first_hypothesis_watchlist_after_rejoin_best_status": (
            rejoin_rank_first_hypothesis_watchlist_status.get("best_candidate_status")
        ),
        "promotion_distance_after_rejoin_status": (
            rejoin_promotion_distance_status.get("status")
        ),
        "pre_race_gated_challenger_after_rejoin_promotion_ready": (
            rejoin_pre_race_gated_status.get("promotion_ready")
        ),
        "time_split_gated_challenger_after_rejoin_promotion_ready": (
            rejoin_time_split_gated_status.get("promotion_ready")
        ),
        "market_residual_challenger_after_rejoin_promotion_ready": (
            rejoin_market_residual_status.get("promotion_ready")
        ),
        "market_residual_regime_audit_after_rejoin_promotion_ready": (
            rejoin_market_residual_regime_status.get("promotion_ready")
        ),
        "promotion_distance_after_rejoin_promotion_ready": (
            rejoin_promotion_distance_status.get("promotion_ready")
        ),
        "dashboard_update_invoked": any(step.get("name") == "aggregate_after_daemon_rejoins" for step in steps)
        and any(step.get("name") == "status_after_daemon_rejoins" for step in steps),
        "feature_activation_gate_checked": bool(feature_activation_gate),
        "feature_activation_gate_status": None if feature_activation_gate is None else feature_activation_gate.get("status"),
        "read_only_odds_coverage_checked": True,
        "odds_coverage_status": odds_coverage.get("status"),
        "live_odds_capture_approval_status": live_odds_capture_packet.get("status"),
        "live_odds_capture_verified_prejump_races": live_odds_capture_packet.get(
            "verified_prejump_race_count"
        ),
        "live_odds_capture_can_capture_now": live_odds_capture_packet.get(
            "can_capture_live_odds_now",
            False,
        ),
        "shadow_odds_snapshot_status": None
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("status"),
        "shadow_odds_snapshot_ev_output_rows": None
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("ev_output_rows", 0),
        "odds_research_next_action": None
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("odds_research_next_action"),
        "timing_aligned_prediction_rerun_required": False
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get(
            "timing_aligned_prediction_rerun_required", False
        ),
        "timing_aligned_prediction_rerun_race_count": 0
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("timing_aligned_prediction_rerun_race_count", 0),
        "timing_aligned_prediction_rerun_race_ids": []
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("timing_aligned_prediction_rerun_race_ids", []),
        "autonomous_live_odds_capture_status": autonomous_live_odds_capture_status.get(
            "status"
        ),
        "autonomous_live_odds_capture_ready_count": autonomous_live_odds_capture_status.get(
            "ready_count"
        ),
        "autonomous_live_odds_capture_inserted_rows": autonomous_odds_inserted_rows,
        "odds_capture_performed": autonomous_odds_inserted_rows > 0,
        "odds_used_for_shadow_scoring": False,
        "odds_capture_state_publish_status": odds_capture_state_publish.get("status"),
        "odds_capture_state_path": odds_capture_state_publish.get("state_path"),
        "odds_capture_next_meaningful_action": odds_capture_state_publish.get(
            "next_meaningful_action"
        ),
        "odds_capture_next_meaningful_action_at": odds_capture_state_publish.get(
            "next_meaningful_action_at"
        ),
        **t2_odds_capture_surface_fields(
            autonomous_live_odds_capture_status=autonomous_live_odds_capture_status,
            odds_capture_state_publish=odds_capture_state_publish,
        ),
        "autonomous_official_result_capture_status": (
            autonomous_official_result_capture_status.get("status")
        ),
        "autonomous_official_result_capture_attempted": (
            autonomous_official_result_capture_status.get("attempted", False)
        ),
        "autonomous_official_result_candidate_count": (
            autonomous_official_result_capture_status.get("candidate_count", 0)
        ),
        "autonomous_official_result_race_rows": (
            autonomous_official_result_capture_status.get("official_result_race_rows", 0)
        ),
        "autonomous_official_result_runner_rows": (
            autonomous_official_result_capture_status.get("official_result_runner_rows", 0)
        ),
        "autonomous_official_result_quarantine_rows": (
            autonomous_official_result_capture_status.get("quarantine_rows", 0)
        ),
        "autonomous_official_result_quarantined_race_ids": (
            autonomous_official_result_capture_status.get("quarantined_race_ids", [])
        ),
        "autonomous_official_result_quarantine_reason_counts": (
            autonomous_official_result_capture_status.get("quarantine_reason_counts", {})
        ),
        "autonomous_official_result_quarantine_error_counts": (
            autonomous_official_result_capture_status.get("quarantine_error_counts", {})
        ),
        "autonomous_official_result_quarantine_result_boxes_not_in_participants_counts": (
            autonomous_official_result_capture_status.get(
                "quarantine_result_boxes_not_in_participants_counts", {}
            )
        ),
        "autonomous_official_result_quarantine_runner_set_mismatch_samples": (
            autonomous_official_result_capture_status.get(
                "quarantine_runner_set_mismatch_samples", []
            )
        ),
        "autonomous_official_result_skipped_reason_counts": (
            autonomous_official_result_capture_status.get("skipped_reason_counts", {})
        ),
        "autonomous_official_result_awaiting_jump_race_count": (
            autonomous_official_result_capture_status.get("awaiting_jump_race_count", 0)
        ),
        "autonomous_official_result_awaiting_jump_race_ids": (
            autonomous_official_result_capture_status.get("awaiting_jump_race_ids", [])
        ),
        "autonomous_official_result_awaiting_jump_next_recheck_after_local": (
            autonomous_official_result_capture_status.get(
                "awaiting_jump_next_recheck_after_local"
            )
        ),
        "next_prejump_refresh_window_checked": bool(next_prejump_refresh_window),
        "next_prejump_refresh_status": None
        if next_prejump_refresh_window is None
        else next_prejump_refresh_window.get("status"),
        "recommended_rerun_after_local": None
        if next_prejump_refresh_window is None
        else next_prejump_refresh_window.get("recommended_rerun_after_local"),
        "prejump_metadata_checked": bool(prejump_metadata_status),
        "prejump_metadata_status": None if prejump_metadata_status is None else prejump_metadata_status.get("status"),
        "prejump_metadata_trend_checked": True,
        "prejump_metadata_trend_status": prejump_metadata_trend.get("status"),
        "prejump_metadata_verified_rate": prejump_metadata_trend.get("verified_metadata_rate"),
        **autopilot_cycle_operational_fields(daily_status),
        "steps": steps,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        "protected_paths_unchanged": protected_paths_unchanged,
        "protected_changed_paths": protected_changed_paths,
        "allowed_odds_db_change": allowed_odds_db_change,
        "allowed_official_result_evidence_db_change": (
            allowed_official_result_evidence_db_change
        ),
        "protected_paths_unchanged_or_allowed": protected_paths_unchanged_or_allowed,
        "lock_release": lock_release,
        "status": "PASS"
        if protected_paths_unchanged_or_allowed
        and all(step.get("status") == "PASS" for step in steps)
        and lock_validation.get("status") == "PASS"
        and recovery_validation.get("status") == "PASS"
        else "FAIL",
    }
    protected_validation = {
        "schema_version": "shadow_autopilot_protected_path_validation_v1",
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "protected_paths_unchanged": protected_paths_unchanged,
        "protected_changed_paths": protected_changed_paths,
        "allowed_odds_db_change": allowed_odds_db_change,
        "allowed_official_result_evidence_db_change": (
            allowed_official_result_evidence_db_change
        ),
        "protected_paths_unchanged_or_allowed": protected_paths_unchanged_or_allowed,
        "protected_paths": list(protected_before.keys()),
    }
    observability = build_shadow_observability(
        generated_at=generated_at,
        run_id=run_id,
        daily_shadow_run_dir=daily_shadow_run_dir,
        daily_manifest=daily_manifest,
        dashboard=dashboard,
        readiness=readiness,
        steps=steps,
        protected_validation=protected_validation,
    )
    cycle_activity = build_cycle_activity_summary(
        current_dashboard=dashboard,
        previous_dashboard=previous_dashboard,
        daily_status=daily_status,
        observability_status=observability["status"],
    )
    dashboard["cycle_activity"] = cycle_activity
    daily_status["cycle_activity_status"] = cycle_activity.get("status")
    daily_status["safe_joined_delta_this_cycle"] = cycle_activity.get(
        "safe_joined_delta_this_cycle"
    )
    rules = alert_rules(args.target_joined_races)
    alert_report = build_alert_report(
        current_dashboard=dashboard,
        previous_dashboard=previous_dashboard,
        automated_join_report=automated_join_report,
        target_joined_races=args.target_joined_races,
        current_observability=observability["status"],
        previous_observability=previous_observability,
    )

    write_json(output_dir / "shadow_dashboard.json", dashboard)
    write_json(output_dir / "result_join_status.json", result_join_status)
    write_json(
        output_dir / "cumulative_join_history.json",
        {
            "schema_version": "shadow_daemon_cumulative_join_history_v1",
            "generated_at": generated_at.isoformat(),
            "join_history": join_history,
            "aggregate_metric_timeseries": aggregate_timeseries,
        },
    )
    write_json(output_dir / "promotion_readiness_tracker.json", readiness)
    write_json(output_dir / "alert_rules.json", rules)
    write_json(output_dir / "alert_report.json", alert_report)
    write_json(output_dir / "service_validation_report.json", service_validation)
    write_json(output_dir / "lock_validation.json", lock_validation)
    write_json(output_dir / "recovery_validation.json", recovery_validation)
    write_json(output_dir / "operational_validation_report.json", operational_validation)
    write_json(output_dir / "protected_path_validation.json", protected_validation)
    if shadow_odds_snapshot:
        write_json(output_dir / "shadow_odds_snapshot_status.json", shadow_odds_snapshot)
    write_json(output_dir / "live_odds_capture_approval_packet.json", live_odds_capture_packet)
    write_json(output_dir / "odds_capture_state_publish_status.json", odds_capture_state_publish)
    write_json(
        output_dir / "autonomous_official_result_capture_status.json",
        autonomous_official_result_capture_status,
    )
    write_json(output_dir / "observability_status.json", observability["status"])
    write_json(output_dir / "prediction_provenance_report.json", observability["provenance"])
    write_json(output_dir / "model_provenance_card.json", observability["model_card"])
    write_json(output_dir / "race_prediction_explanations.json", observability["race_explanations"])
    write_jsonl(output_dir / "observability_event_log.jsonl", observability["event_log"])
    write_text(output_dir / "OBSERVABILITY_STATUS.md", observability["markdown"])
    write_json(output_dir / "DAILY_STATUS.json", daily_status)
    write_text(output_dir / "DAILY_STATUS.md", autopilot.daily_status_markdown(daily_status))
    write_text(output_dir / "SHADOW_STATUS.md", autopilot.shadow_status_markdown(dashboard, readiness))
    write_text(output_dir / "readiness_summary.md", readiness_markdown(readiness))

    required_outputs = [
        "SUMMARY.md",
        "daemon_design.md",
        "lifecycle_diagram.json",
        "service_validation_report.json",
        "lock_validation.json",
        "recovery_validation.json",
        "automated_join_report.json",
        "rejoin_unified_evidence_datasets_status.json",
        "rolling_model_comparison_after_daemon_rejoins_status.json",
        "high_accuracy_refinement_after_daemon_rejoins_status.json",
        "pre_race_gated_challenger_after_daemon_rejoins_status.json",
        "pre_race_rank_first_hypothesis_review_after_daemon_rejoins_status.json",
        "rank_first_hypothesis_watchlist_after_daemon_rejoins_status.json",
        "time_split_gated_challenger_after_daemon_rejoins_status.json",
        "market_residual_challenger_after_daemon_rejoins_status.json",
        "market_residual_regime_audit_after_daemon_rejoins_status.json",
        "promotion_distance_after_daemon_rejoins_status.json",
        "SHADOW_STATUS.md",
        "DAILY_STATUS.md",
        "promotion_readiness_tracker.json",
        "alert_rules.json",
        "alert_report.json",
        "operational_validation_report.json",
        "protected_path_validation.json",
        "OBSERVABILITY_STATUS.md",
        "observability_status.json",
        "prediction_provenance_report.json",
        "model_provenance_card.json",
        "race_prediction_explanations.json",
        "observability_event_log.jsonl",
        "odds_coverage_report.json",
        "shadow_odds_snapshot_status.json",
        "live_odds_capture_approval_packet.json",
        "odds_capture_state_publish_status.json",
        "autonomous_official_result_capture_status.json",
        "prejump_metadata_trend_report.json",
        "readiness_summary.md",
        "verification_results.txt",
        "final_status.txt",
    ]
    required_outputs_present = all((output_dir / name).exists() for name in required_outputs if name not in {"SUMMARY.md", "verification_results.txt", "final_status.txt"})
    verdict = final_verdict(
        protected_paths_unchanged=protected_paths_unchanged_or_allowed,
        required_outputs_present=required_outputs_present,
        service_files_present=service_files_present,
        lock_ok=lock_validation.get("status") == "PASS",
        operational_ok=operational_validation.get("status") == "PASS",
        service_installed=bool(systemd_deployment.get("deployment_ready")),
    )
    write_text(
        output_dir / "verification_results.txt",
        "\n".join(
            [
                f"run_id={run_id}",
                f"training_performed=False",
                f"promotion_performed=False",
                f"registry_mutation=False",
                f"production_pointer_update=False",
                f"active_model_replacement=False",
                f"db_write={bool(autonomous_odds_inserted_rows or autonomous_official_result_evidence_inserted_rows)}",
                f"db_write_scope=append_only_live_odds_or_official_result_evidence_rows_if_true",
                f"label_write=False",
                f"tgr_enabled=False",
                f"betting_action=False",
                f"ev_action=False",
                f"production_prediction_overwrite=False",
                f"snapshot_rewrite=False",
                f"schema_change=False",
                f"hyperparameter_change=False",
                f"calibration_method_change=False",
                f"champion_modification=False",
                f"feature_activation_gate_status={None if feature_activation_gate is None else feature_activation_gate.get('status')}",
                f"odds_coverage_status={odds_coverage.get('status')}",
                f"shadow_odds_snapshot_status={None if shadow_odds_snapshot is None else shadow_odds_snapshot.get('status')}",
                f"odds_research_gate_status={None if shadow_odds_snapshot is None else shadow_odds_snapshot.get('odds_research_gate_status')}",
                f"odds_research_gate_complete_valid_prejump_odds_races={None if shadow_odds_snapshot is None else shadow_odds_snapshot.get('odds_research_gate_complete_valid_prejump_odds_races')}",
                f"odds_research_gate_minimum_complete_valid_prejump_odds_races={None if shadow_odds_snapshot is None else shadow_odds_snapshot.get('odds_research_gate_minimum_complete_valid_prejump_odds_races')}",
                f"odds_research_gate_source_url_coverage_pct={None if shadow_odds_snapshot is None else shadow_odds_snapshot.get('odds_research_gate_source_url_coverage_pct')}",
                f"odds_research_next_action={None if shadow_odds_snapshot is None else shadow_odds_snapshot.get('odds_research_next_action')}",
                f"timing_aligned_prediction_rerun_required={False if shadow_odds_snapshot is None else shadow_odds_snapshot.get('timing_aligned_prediction_rerun_required', False)}",
                f"timing_aligned_prediction_rerun_race_count={0 if shadow_odds_snapshot is None else shadow_odds_snapshot.get('timing_aligned_prediction_rerun_race_count', 0)}",
                f"shadow_odds_snapshot_ev_output_rows={None if shadow_odds_snapshot is None else shadow_odds_snapshot.get('ev_output_rows', 0)}",
                f"live_odds_capture_approval_status={live_odds_capture_packet.get('status')}",
                f"live_odds_capture_verified_prejump_races={live_odds_capture_packet.get('verified_prejump_race_count')}",
                f"live_odds_capture_can_capture_now={live_odds_capture_packet.get('can_capture_live_odds_now')}",
                f"autonomous_live_odds_capture_status={autonomous_live_odds_capture_status.get('status')}",
                f"autonomous_live_odds_capture_ready_count={autonomous_live_odds_capture_status.get('ready_count')}",
                f"autonomous_live_odds_capture_inserted_rows={autonomous_odds_inserted_rows}",
                f"autonomous_official_result_capture_status={autonomous_official_result_capture_status.get('status')}",
                f"autonomous_official_result_capture_attempted={autonomous_official_result_capture_status.get('attempted')}",
                f"autonomous_official_result_candidate_count={autonomous_official_result_capture_status.get('candidate_count')}",
                f"autonomous_official_result_race_rows={autonomous_official_result_capture_status.get('official_result_race_rows')}",
                f"autonomous_official_result_runner_rows={autonomous_official_result_capture_status.get('official_result_runner_rows')}",
                f"autonomous_official_result_quarantine_rows={autonomous_official_result_capture_status.get('quarantine_rows')}",
                f"autonomous_official_result_quarantined_race_ids={autonomous_official_result_capture_status.get('quarantined_race_ids')}",
                f"autonomous_official_result_quarantine_reason_counts={autonomous_official_result_capture_status.get('quarantine_reason_counts')}",
                f"autonomous_official_result_quarantine_error_counts={autonomous_official_result_capture_status.get('quarantine_error_counts')}",
                f"autonomous_official_result_quarantine_result_boxes_not_in_participants_counts={autonomous_official_result_capture_status.get('quarantine_result_boxes_not_in_participants_counts')}",
                f"autonomous_official_result_evidence_db_ingest_status={autonomous_official_result_capture_status.get('official_result_evidence_db_ingest_status')}",
                f"autonomous_official_result_evidence_inserted_race_rows={autonomous_official_result_capture_status.get('official_result_evidence_inserted_race_rows')}",
                f"autonomous_official_result_evidence_inserted_runner_rows={autonomous_official_result_capture_status.get('official_result_evidence_inserted_runner_rows')}",
                f"live_odds_backlog_discovered_races={dashboard.get('live_odds_backlog_discovered_race_count')}",
                f"live_odds_backlog_candidate_races={dashboard.get('live_odds_backlog_candidate_race_count')}",
                f"live_odds_backlog_unresolved_races={dashboard.get('live_odds_backlog_unresolved_race_count')}",
                f"live_odds_backlog_unresolved_reason_counts={dashboard.get('live_odds_backlog_unresolved_reason_counts')}",
                f"live_odds_backlog_unresolved_recovery_action_counts={dashboard.get('live_odds_backlog_unresolved_recovery_action_counts')}",
                f"live_odds_backlog_unresolved_alias_status_counts={dashboard.get('live_odds_backlog_unresolved_alias_status_counts')}",
                f"live_odds_backlog_retryable_exact_shadow_match_races={dashboard.get('live_odds_backlog_retryable_exact_shadow_match_race_count')}",
                f"live_odds_backlog_no_exact_shadow_match_races={dashboard.get('live_odds_backlog_no_exact_shadow_match_race_count')}",
                f"live_odds_backlog_retryable_exact_shadow_match_race_ids={dashboard.get('live_odds_backlog_retryable_exact_shadow_match_race_ids')}",
                f"live_odds_backlog_no_exact_shadow_match_race_ids={dashboard.get('live_odds_backlog_no_exact_shadow_match_race_ids')}",
                f"live_odds_backlog_awaiting_official_result_evidence_races={dashboard.get('live_odds_backlog_awaiting_official_result_evidence_race_count')}",
                f"live_odds_backlog_awaiting_official_result_evidence_race_ids={dashboard.get('live_odds_backlog_awaiting_official_result_evidence_race_ids')}",
                f"live_odds_backlog_awaiting_official_result_evidence_authorized_action={dashboard.get('live_odds_backlog_awaiting_official_result_evidence_authorized_action')}",
                f"live_odds_backlog_awaiting_official_result_recheck_ready_races={dashboard.get('live_odds_backlog_awaiting_official_result_recheck_ready_race_count')}",
                f"live_odds_backlog_join_eligibility_packet={dashboard.get('live_odds_backlog_join_eligibility_packet_path')}",
                f"live_odds_backlog_join_eligibility_evaluated_races={dashboard.get('live_odds_backlog_join_eligibility_evaluated_race_count')}",
                f"live_odds_backlog_join_eligibility_report_only_races={dashboard.get('live_odds_backlog_join_eligibility_eligible_report_only_race_count')}",
                f"live_odds_backlog_join_eligibility_blocked_races={dashboard.get('live_odds_backlog_join_eligibility_blocked_race_count')}",
                f"live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_races={dashboard.get('live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count')}",
                f"live_odds_backlog_join_eligibility_join_authorized={dashboard.get('live_odds_backlog_join_eligibility_join_authorized')}",
                f"live_odds_backlog_join_eligibility_db_write_performed={dashboard.get('live_odds_backlog_join_eligibility_db_write_performed')}",
                *autopilot_cycle_verification_lines(daily_status),
                f"rejoin_unified_evidence_status={rejoin_unified_status.get('status')}",
                f"rejoin_unified_evidence_status_reason={rejoin_unified_status.get('status_reason')}",
                f"rejoin_unified_evidence_evaluated_candidate_count={rejoin_unified_status.get('evaluated_dataset_candidate_count')}",
                f"rejoin_unified_evidence_dataset_count={rejoin_unified_status.get('dataset_count')}",
                f"rejoin_unified_evidence_skipped_dataset_count={rejoin_unified_status.get('skipped_dataset_count')}",
                f"rejoin_unified_evidence_skip_reason_counts={rejoin_unified_status.get('skip_reason_counts')}",
                f"rejoin_unified_evidence_failure_reason_counts={rejoin_unified_status.get('failure_reason_counts')}",
                f"rejoin_unified_evidence_rows={rejoin_unified_status.get('row_count')}",
                f"rejoin_unified_evidence_official_result_rows={rejoin_unified_status.get('rows_with_official_results')}",
                f"rejoin_unified_evidence_strict_odds_rows={rejoin_unified_status.get('rows_with_strict_prejump_odds')}",
                f"rejoin_unified_evidence_artifact_odds_rows_seen={rejoin_unified_status.get('artifact_odds_rows_seen')}",
                f"rejoin_unified_evidence_artifact_odds_rows_accepted={rejoin_unified_status.get('artifact_odds_rows_accepted')}",
                f"rejoin_unified_evidence_artifact_odds_rows_rejected={rejoin_unified_status.get('artifact_odds_rows_rejected')}",
                f"rejoin_unified_evidence_artifact_odds_rejection_reason_counts={rejoin_unified_status.get('artifact_odds_rejection_reason_counts')}",
                f"rejoin_unified_evidence_rows_with_artifact_shadow_odds={rejoin_unified_status.get('rows_with_artifact_shadow_odds')}",
                f"rejoin_unified_evidence_rows_with_artifact_shadow_odds_candidates={rejoin_unified_status.get('rows_with_artifact_shadow_odds_candidates')}",
                f"rejoin_unified_rejected_live_odds_candidate_count={rejoin_unified_status.get('rejected_live_odds_candidate_count')}",
                f"rejoin_unified_rows_with_rejected_live_odds_candidates={rejoin_unified_status.get('rows_with_rejected_live_odds_candidates')}",
                f"rejoin_unified_rejected_live_odds_candidate_reason_counts={rejoin_unified_status.get('rejected_live_odds_candidate_reason_counts')}",
                f"rejoin_unified_evidence_eligible_rows={rejoin_unified_status.get('unified_evidence_eligible_rows')}",
                f"join_eligibility_preview_dataset_count={rejoin_unified_status.get('join_eligibility_preview_dataset_count')}",
                f"join_eligibility_preview_unified_eligible_rows={rejoin_unified_status.get('join_eligibility_preview_unified_eligible_rows')}",
                f"join_eligibility_preview_packet_accepted_races={rejoin_unified_status.get('join_eligibility_preview_packet_accepted_races')}",
                f"join_eligibility_preview_packet_present_races={rejoin_unified_status.get('join_eligibility_preview_packet_present_races')}",
                f"join_eligibility_preview_missing_race_ids={rejoin_unified_status.get('join_eligibility_preview_missing_race_ids')}",
                f"rejoin_rolling_model_comparison_status={rejoin_rolling_status.get('status')}",
                f"rejoin_rolling_model_comparison_sample_races={rejoin_rolling_status.get('sample_race_count')}",
                f"rejoin_rolling_model_comparison_best_candidate={rejoin_rolling_status.get('best_candidate_key')}",
                f"rejoin_high_accuracy_refinement_status={rejoin_high_accuracy_status.get('status')}",
                f"rejoin_high_accuracy_promotion_pr_gate_status={rejoin_high_accuracy_status.get('promotion_pr_gate_status')}",
                f"rejoin_high_accuracy_timing_aligned_rerun_plan={rejoin_high_accuracy_status.get('timing_aligned_rerun_plan')}",
                f"rejoin_high_accuracy_timing_aligned_rerun_execution_status={rejoin_high_accuracy_status.get('timing_aligned_rerun_execution_status')}",
                f"rejoin_reserve_substitution_preflight_status={rejoin_high_accuracy_status.get('reserve_substitution_preflight_status')}",
                f"rejoin_reserve_substitution_preflight_ready_for_policy_review_count={rejoin_high_accuracy_status.get('reserve_substitution_preflight_ready_for_policy_review_count')}",
                f"rejoin_reserve_substitution_preflight_dataset_join_blocker_counts={rejoin_high_accuracy_status.get('reserve_substitution_preflight_dataset_join_blocker_counts')}",
                f"rejoin_reserve_substitution_preflight_ready_race_ids={rejoin_high_accuracy_status.get('reserve_substitution_preflight_ready_race_ids')}",
                f"rejoin_reserve_substitution_manual_review_status={rejoin_high_accuracy_status.get('reserve_substitution_manual_review_status')}",
                f"rejoin_reserve_substitution_manual_review_ready_candidate_count={rejoin_high_accuracy_status.get('reserve_substitution_manual_review_ready_candidate_count')}",
                f"rejoin_reserve_substitution_manual_review_mapping_pair_count={rejoin_high_accuracy_status.get('reserve_substitution_manual_review_mapping_pair_count')}",
                f"rejoin_reserve_substitution_manual_review_dataset_join_allowed={rejoin_high_accuracy_status.get('reserve_substitution_manual_review_dataset_join_allowed')}",
                f"rejoin_reserve_substitution_manual_review_official_result_acceptance_allowed={rejoin_high_accuracy_status.get('reserve_substitution_manual_review_official_result_acceptance_allowed')}",
                f"rejoin_reserve_substitution_manual_review_db_write={rejoin_high_accuracy_status.get('reserve_substitution_manual_review_db_write')}",
                f"rejoin_reserve_substitution_policy_impact_status={rejoin_high_accuracy_status.get('reserve_substitution_policy_impact_status')}",
                f"rejoin_reserve_substitution_policy_impact_ready_candidate_count={rejoin_high_accuracy_status.get('reserve_substitution_policy_impact_ready_candidate_count')}",
                f"rejoin_reserve_substitution_policy_impact_mapping_pair_count={rejoin_high_accuracy_status.get('reserve_substitution_policy_impact_mapping_pair_count')}",
                f"rejoin_reserve_substitution_policy_impact_potential_runner_rows_blocked={rejoin_high_accuracy_status.get('reserve_substitution_policy_impact_potential_runner_rows_blocked')}",
                f"rejoin_reserve_substitution_policy_impact_dataset_join_allowed={rejoin_high_accuracy_status.get('reserve_substitution_policy_impact_dataset_join_allowed')}",
                f"rejoin_reserve_substitution_policy_impact_official_result_acceptance_allowed={rejoin_high_accuracy_status.get('reserve_substitution_policy_impact_official_result_acceptance_allowed')}",
                f"rejoin_reserve_substitution_policy_impact_db_write={rejoin_high_accuracy_status.get('reserve_substitution_policy_impact_db_write')}",
                f"rejoin_pre_race_gated_challenger_status={rejoin_pre_race_gated_status.get('status')}",
                f"rejoin_pre_race_gated_challenger_accepted_races={rejoin_pre_race_gated_status.get('accepted_race_count')}",
                f"rejoin_pre_race_gated_challenger_promotion_ready={rejoin_pre_race_gated_status.get('promotion_ready')}",
                f"rejoin_pre_race_predeclared_residual_candidate_status={rejoin_pre_race_gated_status.get('predeclared_residual_candidate_status')}",
                f"rejoin_pre_race_predeclared_residual_triggered_races={rejoin_pre_race_gated_status.get('predeclared_residual_triggered_race_count')}",
                f"rejoin_pre_race_predeclared_residual_minimum_triggered_races_for_directional_read={rejoin_pre_race_gated_status.get('predeclared_residual_minimum_triggered_races_for_directional_read')}",
                f"rejoin_pre_race_predeclared_residual_directional_read_ready={rejoin_pre_race_gated_status.get('predeclared_residual_directional_read_ready')}",
                f"rejoin_time_split_gated_challenger_status={rejoin_time_split_gated_status.get('status')}",
                f"rejoin_time_split_gated_challenger_test_races={rejoin_time_split_gated_status.get('time_split_test_race_count')}",
                f"rejoin_time_split_gated_challenger_promotion_ready={rejoin_time_split_gated_status.get('promotion_ready')}",
                f"rejoin_market_residual_challenger_status={rejoin_market_residual_status.get('status')}",
                f"rejoin_market_residual_challenger_promotion_ready={rejoin_market_residual_status.get('promotion_ready')}",
                f"rejoin_market_residual_regime_audit_status={rejoin_market_residual_regime_status.get('status')}",
                f"rejoin_market_residual_regime_audit_promotion_ready={rejoin_market_residual_regime_status.get('promotion_ready')}",
                *rejoin_promotion_distance_verification_lines(
                    rejoin_promotion_distance_status
                ),
                f"odds_capture_performed={autonomous_odds_inserted_rows > 0}",
                f"odds_used_for_shadow_scoring=False",
                f"next_prejump_refresh_status={None if next_prejump_refresh_window is None else next_prejump_refresh_window.get('status')}",
                f"recommended_rerun_after_local={None if next_prejump_refresh_window is None else next_prejump_refresh_window.get('recommended_rerun_after_local')}",
                f"prejump_metadata_status={None if prejump_metadata_status is None else prejump_metadata_status.get('status')}",
                f"prejump_metadata_eligible_count={None if prejump_metadata_status is None else prejump_metadata_status.get('eligible_count')}",
                f"prejump_metadata_verified_eligible={None if prejump_metadata_status is None else prejump_metadata_status.get('eligible_with_verified_prejump_metadata')}",
                f"prejump_metadata_trend_status={prejump_metadata_trend.get('status')}",
                f"prejump_metadata_verified_rate={prejump_metadata_trend.get('verified_metadata_rate')}",
                f"protected_paths_unchanged={protected_paths_unchanged}",
                f"protected_paths_unchanged_or_allowed={protected_paths_unchanged_or_allowed}",
                f"protected_changed_paths={protected_changed_paths}",
                f"allowed_official_result_evidence_db_change={allowed_official_result_evidence_db_change}",
                f"scheduled_execution_implemented={service_files_present}",
                f"systemd_deployment_status={systemd_deployment.get('deployment_status')}",
                f"systemd_deployment_ready={systemd_deployment.get('deployment_ready')}",
                f"systemd_timer_active={systemd_deployment.get('timer_active')}",
                f"systemd_timer_enabled={systemd_deployment.get('timer_enabled')}",
                f"locking_implemented={lock_validation.get('status') == 'PASS'}",
                f"automated_joins_implemented=True",
                f"dashboard_implemented=True",
                f"alerts_implemented=True",
                f"observability_implemented=True",
                f"observability_status={observability['status'].get('status')}",
                f"cycle_activity_status={cycle_activity.get('status')}",
                f"safe_joined_delta_this_cycle={cycle_activity.get('safe_joined_delta_this_cycle')}",
                f"operational_validation_passed={operational_validation.get('status') == 'PASS'}",
                f"final_verdict={verdict}",
                "",
            ]
        ),
    )
    write_text(output_dir / "final_status.txt", verdict + "\n")
    write_text(
        output_dir / "SUMMARY.md",
        build_final_summary(
            verdict=verdict,
            dashboard=dashboard,
            readiness=readiness,
            automated_join_report=automated_join_report,
            alert_report=alert_report,
            service_validation=service_validation,
            feature_activation_gate=feature_activation_gate,
            odds_coverage=odds_coverage,
            shadow_odds_snapshot=shadow_odds_snapshot,
            observability_status=observability["status"],
            cycle_activity=cycle_activity,
            next_prejump_refresh_window=next_prejump_refresh_window,
            prejump_metadata_status=prejump_metadata_status,
            prejump_metadata_trend=prejump_metadata_trend,
            live_odds_capture_packet=live_odds_capture_packet,
            autopilot_cycle_daily_status=daily_status,
            rejoin_unified_evidence_status=rejoin_unified_status,
            rejoin_rolling_model_comparison_status=rejoin_rolling_status,
            rejoin_high_accuracy_refinement_status=rejoin_high_accuracy_status,
            rejoin_pre_race_gated_challenger_status=rejoin_pre_race_gated_status,
            rejoin_rank_first_hypothesis_gated_status=(
                rejoin_rank_first_hypothesis_gated_status
            ),
            rejoin_time_split_gated_challenger_status=rejoin_time_split_gated_status,
            rejoin_market_residual_challenger_status=rejoin_market_residual_status,
            rejoin_market_residual_regime_audit_status=(
                rejoin_market_residual_regime_status
            ),
            rejoin_rank_first_hypothesis_watchlist_status=(
                rejoin_rank_first_hypothesis_watchlist_status
            ),
            rejoin_promotion_distance_status=rejoin_promotion_distance_status,
        ),
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_payload = {
        "schema_version": "shadow_autopilot_daemon_state_v1",
        "last_run_id": run_id,
        "last_output_dir": relpath(output_dir),
        "last_verdict": verdict,
        "last_systemd_deployment_status": systemd_deployment.get("deployment_status"),
        "last_systemd_deployment_ready": systemd_deployment.get("deployment_ready"),
        "last_safe_joined_races": dashboard.get("safe_joined_races"),
        "last_feature_activation_gate_status": None
        if feature_activation_gate is None
        else feature_activation_gate.get("status"),
        "last_odds_coverage_status": odds_coverage.get("status"),
        "last_live_odds_capture_approval_status": live_odds_capture_packet.get("status"),
        "last_live_odds_capture_verified_prejump_races": live_odds_capture_packet.get(
            "verified_prejump_race_count"
        ),
        "last_autonomous_live_odds_capture_status": autonomous_live_odds_capture_status.get(
            "status"
        ),
        "last_autonomous_live_odds_capture_ready_count": autonomous_live_odds_capture_status.get(
            "ready_count"
        ),
        "last_autonomous_live_odds_capture_inserted_rows": autonomous_odds_inserted_rows,
        "last_autonomous_live_odds_next_window_opens_at": (
            autonomous_live_odds_capture_status.get("next_window_opens_at")
        ),
        "last_autonomous_live_odds_recommended_rerun_after_local": (
            autonomous_live_odds_capture_status.get("recommended_rerun_after_local")
        ),
        "last_autonomous_live_odds_next_race_id": (
            autonomous_live_odds_capture_status.get("next_race_id")
        ),
        "last_autonomous_live_odds_next_prejump_window": (
            autonomous_live_odds_capture_status.get("next_prejump_window")
        ),
        "last_odds_capture_performed": autonomous_odds_inserted_rows > 0,
        "last_odds_used_for_shadow_scoring": False,
        "last_odds_capture_state_publish_status": odds_capture_state_publish.get("status"),
        "last_odds_capture_state_path": odds_capture_state_publish.get("state_path"),
        "last_odds_capture_next_meaningful_action": odds_capture_state_publish.get(
            "next_meaningful_action"
        ),
        "last_odds_capture_next_meaningful_action_at": odds_capture_state_publish.get(
            "next_meaningful_action_at"
        ),
        **t2_odds_capture_surface_fields(
            autonomous_live_odds_capture_status=autonomous_live_odds_capture_status,
            odds_capture_state_publish=odds_capture_state_publish,
            last=True,
        ),
        "last_autonomous_official_result_capture_status": (
            autonomous_official_result_capture_status.get("status")
        ),
        "last_autonomous_official_result_capture_attempted": (
            autonomous_official_result_capture_status.get("attempted", False)
        ),
        "last_autonomous_official_result_race_rows": (
            autonomous_official_result_capture_status.get("official_result_race_rows", 0)
        ),
        "last_autonomous_official_result_runner_rows": (
            autonomous_official_result_capture_status.get("official_result_runner_rows", 0)
        ),
        "last_autonomous_official_result_quarantine_rows": (
            autonomous_official_result_capture_status.get("quarantine_rows", 0)
        ),
        "last_autonomous_official_result_quarantined_race_ids": (
            autonomous_official_result_capture_status.get("quarantined_race_ids", [])
        ),
        "last_autonomous_official_result_quarantine_reason_counts": (
            autonomous_official_result_capture_status.get("quarantine_reason_counts", {})
        ),
        "last_autonomous_official_result_quarantine_error_counts": (
            autonomous_official_result_capture_status.get("quarantine_error_counts", {})
        ),
        "last_autonomous_official_result_quarantine_result_boxes_not_in_participants_counts": (
            autonomous_official_result_capture_status.get(
                "quarantine_result_boxes_not_in_participants_counts", {}
            )
        ),
        "last_autonomous_official_result_quarantine_runner_set_mismatch_samples": (
            autonomous_official_result_capture_status.get(
                "quarantine_runner_set_mismatch_samples", []
            )
        ),
        "last_autonomous_official_result_skipped_reason_counts": (
            autonomous_official_result_capture_status.get("skipped_reason_counts", {})
        ),
        "last_autonomous_official_result_awaiting_jump_race_count": (
            autonomous_official_result_capture_status.get("awaiting_jump_race_count", 0)
        ),
        "last_autonomous_official_result_awaiting_jump_race_ids": (
            autonomous_official_result_capture_status.get("awaiting_jump_race_ids", [])
        ),
        "last_autonomous_official_result_awaiting_jump_next_recheck_after_local": (
            autonomous_official_result_capture_status.get(
                "awaiting_jump_next_recheck_after_local"
            )
        ),
        "last_autonomous_official_result_evidence_db_ingest_status": (
            autonomous_official_result_capture_status.get(
                "official_result_evidence_db_ingest_status"
            )
        ),
        "last_autonomous_official_result_evidence_db_execute": (
            autonomous_official_result_capture_status.get(
                "official_result_evidence_db_execute", False
            )
        ),
        "last_autonomous_official_result_evidence_db_write_performed": (
            autonomous_official_result_capture_status.get(
                "official_result_evidence_db_write_performed", False
            )
        ),
        "last_autonomous_official_result_evidence_valid_race_rows": (
            autonomous_official_result_capture_status.get(
                "official_result_evidence_valid_race_rows", 0
            )
        ),
        "last_autonomous_official_result_evidence_valid_runner_rows": (
            autonomous_official_result_capture_status.get(
                "official_result_evidence_valid_runner_rows", 0
            )
        ),
        "last_autonomous_official_result_evidence_blocked_race_rows": (
            autonomous_official_result_capture_status.get(
                "official_result_evidence_blocked_race_rows", 0
            )
        ),
        "last_autonomous_official_result_evidence_blocked_runner_rows": (
            autonomous_official_result_capture_status.get(
                "official_result_evidence_blocked_runner_rows", 0
            )
        ),
        "last_autonomous_official_result_evidence_inserted_race_rows": (
            autonomous_official_result_capture_status.get(
                "official_result_evidence_inserted_race_rows", 0
            )
        ),
        "last_autonomous_official_result_evidence_inserted_runner_rows": (
            autonomous_official_result_capture_status.get(
                "official_result_evidence_inserted_runner_rows", 0
            )
        ),
        "last_autonomous_official_result_evidence_blocker_reason_counts": (
            autonomous_official_result_capture_status.get(
                "official_result_evidence_blocker_reason_counts", {}
            )
            or {}
        ),
        "last_autonomous_official_result_evidence_inserted_rows": (
            autonomous_official_result_evidence_inserted_rows
        ),
        **live_odds_backlog_state_fields(dashboard),
        **rejoin_unified_state_fields(rejoin_unified_status),
        "last_rejoin_rolling_model_comparison_status": rejoin_rolling_status.get("status"),
        "last_rejoin_rolling_model_comparison_sample_races": rejoin_rolling_status.get(
            "sample_race_count"
        ),
        "last_rejoin_high_accuracy_refinement_status": rejoin_high_accuracy_status.get(
            "status"
        ),
        **rejoin_high_accuracy_timing_source_fields(
            rejoin_high_accuracy_status,
            prefix="last_rejoin_high_accuracy_",
        ),
        "last_rejoin_pre_race_gated_challenger_status": (
            rejoin_pre_race_gated_status.get("status")
        ),
        "last_rejoin_pre_race_gated_challenger_promotion_ready": (
            rejoin_pre_race_gated_status.get("promotion_ready")
        ),
        "last_rejoin_pre_race_predeclared_residual_candidate_status": (
            rejoin_pre_race_gated_status.get("predeclared_residual_candidate_status")
        ),
        "last_rejoin_pre_race_predeclared_residual_triggered_races": (
            rejoin_pre_race_gated_status.get("predeclared_residual_triggered_race_count")
        ),
        "last_rejoin_pre_race_predeclared_residual_minimum_triggered_races_for_directional_read": (
            rejoin_pre_race_gated_status.get(
                "predeclared_residual_minimum_triggered_races_for_directional_read"
            )
        ),
        "last_rejoin_pre_race_predeclared_residual_directional_read_ready": (
            rejoin_pre_race_gated_status.get(
                "predeclared_residual_directional_read_ready"
            )
        ),
        "last_rejoin_rank_first_hypothesis_review_status": (
            rejoin_rank_first_hypothesis_gated_status.get(
                "rank_first_hypothesis_review_status"
            )
        ),
        "last_rejoin_rank_first_hypothesis_candidate_count": (
            rejoin_rank_first_hypothesis_gated_status.get(
                "rank_first_hypothesis_candidate_count"
            )
        ),
        "last_rejoin_rank_first_hypothesis_evaluated_candidate_count": (
            rejoin_rank_first_hypothesis_gated_status.get(
                "rank_first_hypothesis_evaluated_candidate_count"
            )
        ),
        "last_rejoin_rank_first_hypothesis_best_candidate_key": (
            rejoin_rank_first_hypothesis_gated_status.get(
                "rank_first_hypothesis_best_candidate_key"
            )
        ),
        "last_rejoin_rank_first_hypothesis_best_triggered_races": (
            rejoin_rank_first_hypothesis_gated_status.get(
                "rank_first_hypothesis_best_triggered_race_count"
            )
        ),
        "last_rejoin_rank_first_hypothesis_minimum_triggered_races_for_directional_read": (
            rejoin_rank_first_hypothesis_gated_status.get(
                "rank_first_hypothesis_minimum_triggered_races_for_directional_read"
            )
        ),
        "last_rejoin_rank_first_hypothesis_directional_read_ready": (
            rejoin_rank_first_hypothesis_gated_status.get(
                "rank_first_hypothesis_directional_read_ready"
            )
        ),
        "last_rejoin_time_split_gated_challenger_status": (
            rejoin_time_split_gated_status.get("status")
        ),
        "last_rejoin_time_split_gated_challenger_promotion_ready": (
            rejoin_time_split_gated_status.get("promotion_ready")
        ),
        "last_rejoin_market_residual_challenger_status": (
            rejoin_market_residual_status.get("status")
        ),
        "last_rejoin_market_residual_challenger_promotion_ready": (
            rejoin_market_residual_status.get("promotion_ready")
        ),
        "last_rejoin_market_residual_regime_audit_status": (
            rejoin_market_residual_regime_status.get("status")
        ),
        "last_rejoin_market_residual_regime_audit_promotion_ready": (
            rejoin_market_residual_regime_status.get("promotion_ready")
        ),
        "last_rejoin_market_residual_rank_first_hypothesis_status": (
            rejoin_market_residual_regime_status.get("rank_first_hypothesis_status")
        ),
        "last_rejoin_market_residual_rank_first_help_regimes": (
            rejoin_market_residual_regime_status.get(
                "pre_race_rank_first_help_regime_count"
            )
        ),
        "last_rejoin_market_residual_logloss_only_help_regimes": (
            rejoin_market_residual_regime_status.get(
                "pre_race_logloss_only_help_regime_count"
            )
        ),
        "last_rejoin_rank_first_hypothesis_watchlist_status": (
            rejoin_rank_first_hypothesis_watchlist_status.get("status")
        ),
        "last_rejoin_rank_first_hypothesis_watchlist_candidate_count": (
            rejoin_rank_first_hypothesis_watchlist_status.get("candidate_count")
        ),
        "last_rejoin_rank_first_hypothesis_watchlist_directional_ready_candidate_count": (
            rejoin_rank_first_hypothesis_watchlist_status.get(
                "directional_ready_candidate_count"
            )
        ),
        "last_rejoin_rank_first_hypothesis_watchlist_best_candidate": (
            rejoin_rank_first_hypothesis_watchlist_status.get("best_candidate_key")
        ),
        "last_rejoin_rank_first_hypothesis_watchlist_best_status": (
            rejoin_rank_first_hypothesis_watchlist_status.get("best_candidate_status")
        ),
        "last_rejoin_rank_first_hypothesis_watchlist_best_distinct_samples": (
            rejoin_rank_first_hypothesis_watchlist_status.get(
                "best_candidate_distinct_sample_count"
            )
        ),
        "last_rejoin_promotion_distance_status": (
            rejoin_promotion_distance_status.get("status")
        ),
        "last_rejoin_promotion_distance_promotion_ready": (
            rejoin_promotion_distance_status.get("promotion_ready")
        ),
        "last_rejoin_promotion_distance_blockers": (
            rejoin_promotion_distance_status.get("blockers") or []
        ),
        "last_rejoin_promotion_distance_source_exclusion_reason_counts": (
            rejoin_promotion_distance_status.get("source_exclusion_reason_counts")
            or {}
        ),
        "last_rejoin_promotion_distance_source_odds_exclusion_reason_counts": (
            rejoin_promotion_distance_status.get("source_odds_exclusion_reason_counts")
            or {}
        ),
        "last_rejoin_promotion_distance_source_official_result_evidence_db_missing_race_ids": (
            rejoin_promotion_distance_status.get(
                "source_official_result_evidence_db_missing_race_ids"
            )
            or []
        ),
        "last_rejoin_promotion_distance_source_official_result_evidence_db_requested_race_count": (
            rejoin_promotion_distance_status.get(
                "source_official_result_evidence_db_requested_race_count"
            )
        ),
        "last_rejoin_promotion_distance_source_official_result_evidence_db_races_with_rows": (
            rejoin_promotion_distance_status.get(
                "source_official_result_evidence_db_races_with_rows"
            )
            or []
        ),
        "last_rejoin_promotion_distance_source_official_result_runner_paths": (
            rejoin_promotion_distance_status.get("source_official_result_runner_paths")
            or []
        ),
        "last_rejoin_promotion_distance_official_result_coverage_requested_race_count": (
            rejoin_promotion_distance_status.get(
                "official_result_coverage_requested_race_count"
            )
        ),
        "last_rejoin_promotion_distance_official_result_coverage_requested_race_count_source": (
            rejoin_promotion_distance_status.get(
                "official_result_coverage_requested_race_count_source"
            )
        ),
        "last_rejoin_promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids": (
            rejoin_promotion_distance_status.get(
                "official_result_coverage_legacy_requested_race_count_without_ids"
            )
        ),
        "last_rejoin_promotion_distance_official_result_coverage_races_with_rows_count": (
            rejoin_promotion_distance_status.get(
                "official_result_coverage_races_with_rows_count"
            )
        ),
        "last_rejoin_promotion_distance_official_result_coverage_missing_race_count": (
            rejoin_promotion_distance_status.get(
                "official_result_coverage_missing_race_count"
            )
        ),
        "last_rejoin_promotion_distance_official_result_coverage_missing_exclusion_count": (
            rejoin_promotion_distance_status.get(
                "official_result_coverage_missing_exclusion_count"
            )
        ),
        "last_rejoin_promotion_distance_official_result_runner_path_count": (
            rejoin_promotion_distance_status.get("official_result_runner_path_count")
        ),
        "last_rejoin_promotion_distance_official_result_runner_paths_source_field": (
            rejoin_promotion_distance_status.get(
                "official_result_runner_paths_source_field"
            )
        ),
        "last_shadow_odds_snapshot_status": None
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("status"),
        "last_shadow_odds_snapshot_ev_output_rows": None
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("ev_output_rows", 0),
        "last_shadow_odds_snapshot_complete_valid_races": None
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("races_with_complete_valid_prejump_odds"),
        "last_shadow_odds_snapshot_races_with_missing_odds_rows": None
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("races_with_missing_odds_rows"),
        "last_odds_research_next_action": None
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("odds_research_next_action"),
        "last_timing_aligned_prediction_rerun_required": False
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get(
            "timing_aligned_prediction_rerun_required", False
        ),
        "last_timing_aligned_prediction_rerun_race_count": 0
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("timing_aligned_prediction_rerun_race_count", 0),
        "last_timing_aligned_prediction_rerun_race_ids": []
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("timing_aligned_prediction_rerun_race_ids", []),
        "last_shadow_odds_snapshot_races_after_feature_freeze": None
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("races_with_post_feature_freeze_odds_rows", 0),
        "last_next_prejump_refresh_status": None
        if next_prejump_refresh_window is None
        else next_prejump_refresh_window.get("status"),
        "last_recommended_rerun_after_local": None
        if next_prejump_refresh_window is None
        else next_prejump_refresh_window.get("recommended_rerun_after_local"),
        "last_next_prejump_race": None
        if next_prejump_refresh_window is None
        else next_prejump_refresh_window.get("next_race"),
        "last_prejump_metadata_status": None
        if prejump_metadata_status is None
        else prejump_metadata_status.get("status"),
        "last_prejump_metadata_eligible_count": None
        if prejump_metadata_status is None
        else prejump_metadata_status.get("eligible_count"),
        "last_prejump_metadata_verified_eligible": None
        if prejump_metadata_status is None
        else prejump_metadata_status.get("eligible_with_verified_prejump_metadata"),
        "last_prejump_metadata_trend_status": prejump_metadata_trend.get("status"),
        "last_prejump_metadata_verified_rate": prejump_metadata_trend.get(
            "verified_metadata_rate"
        ),
        "last_observability_status": observability["status"].get("status"),
        "last_cycle_activity_status": cycle_activity.get("status"),
        "last_safe_joined_delta": cycle_activity.get("safe_joined_delta_this_cycle"),
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    state_payload.update(autopilot_cycle_state_fields(daily_status))
    write_json(state_path, state_payload)
    runtime_state_report = write_daemon_runtime_state_packet(
        output_dir=output_dir,
        state_path=state_path,
        systemd_deployment=systemd_deployment,
        target_joined_races=args.target_joined_races,
        generated_at=generated_at,
    )
    result = {
        **completed_daemon_run_report_envelope(
            run_id=run_id,
            generated_at=generated_at,
            current_time=current_time,
            output_dir=output_dir,
            final_verdict=verdict,
        ),
        "systemd_deployment_status": systemd_deployment.get("deployment_status"),
        "systemd_deployment_ready": systemd_deployment.get("deployment_ready"),
        "safe_joined_races": dashboard.get("safe_joined_races"),
        "pending_races": dashboard.get("pending_races"),
        "unsafe_matches": dashboard.get("unsafe_matches"),
        "readiness_decision": readiness.get("decision"),
        "feature_activation_gate_status": None
        if feature_activation_gate is None
        else feature_activation_gate.get("status"),
        "odds_coverage_status": odds_coverage.get("status"),
        "live_odds_capture_approval_status": live_odds_capture_packet.get("status"),
        "live_odds_capture_verified_prejump_race_count": live_odds_capture_packet.get(
            "verified_prejump_race_count"
        ),
        "autonomous_live_odds_capture_status": autonomous_live_odds_capture_status.get(
            "status"
        ),
        "autonomous_live_odds_capture_ready_count": autonomous_live_odds_capture_status.get(
            "ready_count"
        ),
        "autonomous_live_odds_capture_inserted_rows": autonomous_odds_inserted_rows,
        "odds_capture_performed": autonomous_odds_inserted_rows > 0,
        "odds_used_for_shadow_scoring": False,
        "odds_capture_state_publish_status": odds_capture_state_publish.get("status"),
        "odds_capture_state_path": odds_capture_state_publish.get("state_path"),
        "odds_capture_next_meaningful_action": odds_capture_state_publish.get(
            "next_meaningful_action"
        ),
        "odds_capture_next_meaningful_action_at": odds_capture_state_publish.get(
            "next_meaningful_action_at"
        ),
        **t2_odds_capture_surface_fields(
            autonomous_live_odds_capture_status=autonomous_live_odds_capture_status,
            odds_capture_state_publish=odds_capture_state_publish,
        ),
        "autonomous_official_result_capture_status": (
            autonomous_official_result_capture_status.get("status")
        ),
        "autonomous_official_result_capture_attempted": (
            autonomous_official_result_capture_status.get("attempted", False)
        ),
        "autonomous_official_result_race_rows": (
            autonomous_official_result_capture_status.get("official_result_race_rows", 0)
        ),
        "autonomous_official_result_runner_rows": (
            autonomous_official_result_capture_status.get("official_result_runner_rows", 0)
        ),
        "autonomous_official_result_quarantine_rows": (
            autonomous_official_result_capture_status.get("quarantine_rows", 0)
        ),
        "autonomous_official_result_quarantined_race_ids": (
            autonomous_official_result_capture_status.get("quarantined_race_ids", [])
        ),
        "autonomous_official_result_quarantine_reason_counts": (
            autonomous_official_result_capture_status.get("quarantine_reason_counts", {})
        ),
        "autonomous_official_result_quarantine_error_counts": (
            autonomous_official_result_capture_status.get("quarantine_error_counts", {})
        ),
        "autonomous_official_result_quarantine_result_boxes_not_in_participants_counts": (
            autonomous_official_result_capture_status.get(
                "quarantine_result_boxes_not_in_participants_counts", {}
            )
        ),
        "autonomous_official_result_quarantine_runner_set_mismatch_samples": (
            autonomous_official_result_capture_status.get(
                "quarantine_runner_set_mismatch_samples", []
            )
        ),
        "autonomous_official_result_skipped_reason_counts": (
            autonomous_official_result_capture_status.get("skipped_reason_counts", {})
        ),
        "autonomous_official_result_awaiting_jump_race_count": (
            autonomous_official_result_capture_status.get("awaiting_jump_race_count", 0)
        ),
        "autonomous_official_result_awaiting_jump_race_ids": (
            autonomous_official_result_capture_status.get("awaiting_jump_race_ids", [])
        ),
        "autonomous_official_result_awaiting_jump_next_recheck_after_local": (
            autonomous_official_result_capture_status.get(
                "awaiting_jump_next_recheck_after_local"
            )
        ),
        "autonomous_official_result_evidence_inserted_rows": (
            autonomous_official_result_evidence_inserted_rows
        ),
        **autonomous_official_result_operational_fields(
            daily_status,
            autonomous_official_result_capture_status,
            autonomous_official_result_evidence_inserted_rows,
        ),
        **autopilot_cycle_operational_fields(daily_status),
        "rejoin_unified_evidence_status": rejoin_unified_status.get("status"),
        "rejoin_unified_evidence_status_reason": rejoin_unified_status.get(
            "status_reason"
        ),
        "rejoin_unified_evidence_evaluated_candidate_count": rejoin_unified_status.get(
            "evaluated_dataset_candidate_count"
        ),
        "rejoin_unified_evidence_dataset_count": rejoin_unified_status.get("dataset_count"),
        "rejoin_unified_evidence_skipped_dataset_count": rejoin_unified_status.get(
            "skipped_dataset_count"
        ),
        "rejoin_unified_evidence_skip_reason_counts": (
            rejoin_unified_status.get("skip_reason_counts") or {}
        ),
        "rejoin_unified_evidence_failure_reason_counts": (
            rejoin_unified_status.get("failure_reason_counts") or {}
        ),
        "rejoin_unified_evidence_eligible_rows": rejoin_unified_status.get(
            "unified_evidence_eligible_rows"
        ),
        "rejoin_rolling_model_comparison_status": rejoin_rolling_status.get("status"),
        "rejoin_rolling_model_comparison_sample_races": rejoin_rolling_status.get(
            "sample_race_count"
        ),
        "rejoin_high_accuracy_refinement_status": rejoin_high_accuracy_status.get("status"),
        **rejoin_high_accuracy_timing_source_fields(
            rejoin_high_accuracy_status,
            prefix="rejoin_high_accuracy_",
        ),
        "rejoin_pre_race_gated_challenger_status": rejoin_pre_race_gated_status.get(
            "status"
        ),
        "rejoin_pre_race_gated_challenger_promotion_ready": (
            rejoin_pre_race_gated_status.get("promotion_ready")
        ),
        "rejoin_pre_race_predeclared_residual_candidate_status": (
            rejoin_pre_race_gated_status.get("predeclared_residual_candidate_status")
        ),
        "rejoin_pre_race_predeclared_residual_triggered_races": (
            rejoin_pre_race_gated_status.get("predeclared_residual_triggered_race_count")
        ),
        "rejoin_pre_race_predeclared_residual_minimum_triggered_races_for_directional_read": (
            rejoin_pre_race_gated_status.get(
                "predeclared_residual_minimum_triggered_races_for_directional_read"
            )
        ),
        "rejoin_pre_race_predeclared_residual_directional_read_ready": (
            rejoin_pre_race_gated_status.get(
                "predeclared_residual_directional_read_ready"
            )
        ),
        "rejoin_rank_first_hypothesis_review_status": (
            rejoin_rank_first_hypothesis_gated_status.get(
                "rank_first_hypothesis_review_status"
            )
        ),
        "rejoin_rank_first_hypothesis_candidate_count": (
            rejoin_rank_first_hypothesis_gated_status.get(
                "rank_first_hypothesis_candidate_count"
            )
        ),
        "rejoin_rank_first_hypothesis_evaluated_candidate_count": (
            rejoin_rank_first_hypothesis_gated_status.get(
                "rank_first_hypothesis_evaluated_candidate_count"
            )
        ),
        "rejoin_rank_first_hypothesis_best_candidate_key": (
            rejoin_rank_first_hypothesis_gated_status.get(
                "rank_first_hypothesis_best_candidate_key"
            )
        ),
        "rejoin_rank_first_hypothesis_best_triggered_races": (
            rejoin_rank_first_hypothesis_gated_status.get(
                "rank_first_hypothesis_best_triggered_race_count"
            )
        ),
        "rejoin_rank_first_hypothesis_directional_read_ready": (
            rejoin_rank_first_hypothesis_gated_status.get(
                "rank_first_hypothesis_directional_read_ready"
            )
        ),
        "rejoin_time_split_gated_challenger_status": (
            rejoin_time_split_gated_status.get("status")
        ),
        "rejoin_time_split_gated_challenger_promotion_ready": (
            rejoin_time_split_gated_status.get("promotion_ready")
        ),
        "rejoin_rank_first_hypothesis_watchlist_status": (
            rejoin_rank_first_hypothesis_watchlist_status.get("status")
        ),
        "rejoin_rank_first_hypothesis_watchlist_candidate_count": (
            rejoin_rank_first_hypothesis_watchlist_status.get("candidate_count")
        ),
        "rejoin_rank_first_hypothesis_watchlist_directional_ready_candidate_count": (
            rejoin_rank_first_hypothesis_watchlist_status.get(
                "directional_ready_candidate_count"
            )
        ),
        "rejoin_rank_first_hypothesis_watchlist_best_candidate": (
            rejoin_rank_first_hypothesis_watchlist_status.get("best_candidate_key")
        ),
        "rejoin_rank_first_hypothesis_watchlist_best_status": (
            rejoin_rank_first_hypothesis_watchlist_status.get("best_candidate_status")
        ),
        "shadow_odds_snapshot_status": None
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("status"),
        "shadow_odds_snapshot_ev_output_rows": None
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("ev_output_rows", 0),
        "odds_research_next_action": None
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("odds_research_next_action"),
        "timing_aligned_prediction_rerun_required": False
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get(
            "timing_aligned_prediction_rerun_required", False
        ),
        "timing_aligned_prediction_rerun_race_count": 0
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("timing_aligned_prediction_rerun_race_count", 0),
        "timing_aligned_prediction_rerun_race_ids": []
        if shadow_odds_snapshot is None
        else shadow_odds_snapshot.get("timing_aligned_prediction_rerun_race_ids", []),
        "next_prejump_refresh_status": None
        if next_prejump_refresh_window is None
        else next_prejump_refresh_window.get("status"),
        "recommended_rerun_after_local": None
        if next_prejump_refresh_window is None
        else next_prejump_refresh_window.get("recommended_rerun_after_local"),
        "prejump_metadata_status": None
        if prejump_metadata_status is None
        else prejump_metadata_status.get("status"),
        "prejump_metadata_eligible_count": None
        if prejump_metadata_status is None
        else prejump_metadata_status.get("eligible_count"),
        "prejump_metadata_verified_eligible": None
        if prejump_metadata_status is None
        else prejump_metadata_status.get("eligible_with_verified_prejump_metadata"),
        "prejump_metadata_trend_status": prejump_metadata_trend.get("status"),
        "prejump_metadata_verified_rate": prejump_metadata_trend.get(
            "verified_metadata_rate"
        ),
        "observability_status": observability["status"].get("status"),
        "cycle_activity_status": cycle_activity.get("status"),
        "safe_joined_delta_this_cycle": cycle_activity.get("safe_joined_delta_this_cycle"),
        "runtime_action": runtime_state_report.get("runtime_action"),
        "autonomous_live_odds_capture_status": autonomous_live_odds_capture_status.get(
            "status"
        ),
        "autonomous_live_odds_capture_ready_count": autonomous_live_odds_capture_status.get(
            "ready_count"
        ),
        "autonomous_live_odds_capture_inserted_rows": autonomous_odds_inserted_rows,
        "odds_capture_performed": autonomous_odds_inserted_rows > 0,
        "odds_used_for_shadow_scoring": False,
        "autonomous_official_result_capture_status": (
            autonomous_official_result_capture_status.get("status")
        ),
        "autonomous_official_result_capture_attempted": (
            autonomous_official_result_capture_status.get("attempted", False)
        ),
        "autonomous_official_result_race_rows": (
            autonomous_official_result_capture_status.get("official_result_race_rows", 0)
        ),
        "autonomous_official_result_runner_rows": (
            autonomous_official_result_capture_status.get("official_result_runner_rows", 0)
        ),
        "autonomous_official_result_quarantine_rows": (
            autonomous_official_result_capture_status.get("quarantine_rows", 0)
        ),
        "autonomous_official_result_quarantined_race_ids": (
            autonomous_official_result_capture_status.get("quarantined_race_ids", [])
        ),
        "autonomous_official_result_quarantine_reason_counts": (
            autonomous_official_result_capture_status.get("quarantine_reason_counts", {})
        ),
        "autonomous_official_result_quarantine_error_counts": (
            autonomous_official_result_capture_status.get("quarantine_error_counts", {})
        ),
        "autonomous_official_result_quarantine_result_boxes_not_in_participants_counts": (
            autonomous_official_result_capture_status.get(
                "quarantine_result_boxes_not_in_participants_counts", {}
            )
        ),
        "autonomous_official_result_quarantine_runner_set_mismatch_samples": (
            autonomous_official_result_capture_status.get(
                "quarantine_runner_set_mismatch_samples", []
            )
        ),
        "autonomous_official_result_skipped_reason_counts": (
            autonomous_official_result_capture_status.get("skipped_reason_counts", {})
        ),
        "autonomous_official_result_awaiting_jump_race_count": (
            autonomous_official_result_capture_status.get("awaiting_jump_race_count", 0)
        ),
        "autonomous_official_result_awaiting_jump_race_ids": (
            autonomous_official_result_capture_status.get("awaiting_jump_race_ids", [])
        ),
        "autonomous_official_result_awaiting_jump_next_recheck_after_local": (
            autonomous_official_result_capture_status.get(
                "awaiting_jump_next_recheck_after_local"
            )
        ),
        "autonomous_official_result_evidence_inserted_rows": (
            autonomous_official_result_evidence_inserted_rows
        ),
        "live_odds_backlog_unresolved_race_count": dashboard.get(
            "live_odds_backlog_unresolved_race_count"
        ),
        "live_odds_backlog_unresolved_reason_counts": dashboard.get(
            "live_odds_backlog_unresolved_reason_counts"
        ),
        "live_odds_backlog_unresolved_recovery_action_counts": dashboard.get(
            "live_odds_backlog_unresolved_recovery_action_counts"
        ),
        "live_odds_backlog_unresolved_alias_status_counts": dashboard.get(
            "live_odds_backlog_unresolved_alias_status_counts"
        ),
        "live_odds_backlog_retryable_exact_shadow_match_race_count": dashboard.get(
            "live_odds_backlog_retryable_exact_shadow_match_race_count"
        ),
        "live_odds_backlog_no_exact_shadow_match_race_count": dashboard.get(
            "live_odds_backlog_no_exact_shadow_match_race_count"
        ),
        "live_odds_backlog_retryable_exact_shadow_match_race_ids": dashboard.get(
            "live_odds_backlog_retryable_exact_shadow_match_race_ids"
        ),
        "live_odds_backlog_no_exact_shadow_match_race_ids": dashboard.get(
            "live_odds_backlog_no_exact_shadow_match_race_ids"
        ),
        "protected_paths_unchanged": protected_paths_unchanged,
        "protected_paths_unchanged_or_allowed": protected_paths_unchanged_or_allowed,
        "allowed_official_result_evidence_db_change": (
            allowed_official_result_evidence_db_change
        ),
    }
    write_json(output_dir / "daemon_run_report.json", result)
    write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-once", help="Run one timer-safe daemon cycle")
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    run_parser.add_argument("--output-dir", type=Path)
    run_parser.add_argument("--current-time")
    run_parser.add_argument("--db", type=Path, default=ROOT / "greyhound_racing_data.db")
    run_parser.add_argument("--shadow-model", type=Path)
    run_parser.add_argument("--days-ahead", type=int, default=1)
    run_parser.add_argument("--min-minutes", type=float, default=20.0)
    run_parser.add_argument("--max-minutes", type=float, default=160.0)
    run_parser.add_argument("--refresh-limit", type=int, default=16)
    run_parser.add_argument(
        "--autonomous-odds-capture-limit",
        type=int,
        default=DEFAULT_FULL_DAEMON_AUTONOMOUS_ODDS_CAPTURE_LIMIT,
        help=(
            "Maximum live-odds races the full daemon executes inside its primary "
            "autopilot cycle. The odds-only timer remains the broader continuous "
            "collector."
        ),
    )
    run_parser.add_argument("--refresh-dry-run", action="store_true")
    run_parser.add_argument("--refresh-command-mode", choices=("auto", "python", "uv"), default="auto")
    run_parser.add_argument(
        "--require-safe-refresh-metadata",
        dest="require_safe_refresh_metadata",
        action="store_true",
        default=True,
    )
    run_parser.add_argument(
        "--allow-incomplete-refresh-metadata",
        dest="require_safe_refresh_metadata",
        action="store_false",
    )
    run_parser.add_argument("--score-command-mode", choices=("auto", "python", "uv"), default="auto")
    run_parser.add_argument("--target-joined-races", type=int, default=DEFAULT_TARGET_JOINED_RACES)
    run_parser.add_argument("--min-joined-races", type=int, default=DEFAULT_MIN_JOINED_RACES)
    run_parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    run_parser.add_argument("--lock-path", type=Path)
    run_parser.add_argument("--lock-stale-seconds", type=int, default=DEFAULT_LOCK_STALE_SECONDS)
    run_parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    run_parser.add_argument(
        "--odds-capture-state-path",
        type=Path,
        default=DEFAULT_ODDS_CAPTURE_ONLY_STATE_PATH,
    )
    run_parser.add_argument("--rejoin-pending-limit", type=int, default=DEFAULT_REJOIN_PENDING_LIMIT)
    run_parser.add_argument("--rejoin-lookback-days", type=int, default=DEFAULT_REJOIN_LOOKBACK_DAYS)
    run_parser.add_argument("--skip-refresh", action="store_true")
    run_parser.add_argument("--skip-shadow-run", action="store_true")
    run_parser.add_argument("--skip-unified-dataset", action="store_true")
    run_parser.add_argument("--enable-autonomous-odds-capture", action="store_true")
    run_parser.add_argument("--execute-autonomous-odds-capture", action="store_true")
    run_parser.add_argument("--allow-auto-scrape-odds", action="store_true")
    run_parser.add_argument("--enable-autonomous-result-capture", action="store_true")
    run_parser.add_argument(
        "--result-backlog-limit",
        type=int,
        default=DEFAULT_FULL_DAEMON_RESULT_BACKLOG_LIMIT,
        help=(
            "Maximum live-odds backlog races the full daemon asks autonomous "
            "official-result capture to inspect."
        ),
    )
    run_parser.add_argument(
        "--result-backlog-shadow-run-limit",
        type=int,
        default=DEFAULT_FULL_DAEMON_RESULT_BACKLOG_SHADOW_RUN_LIMIT,
        help=(
            "Maximum shadow runs the full daemon asks autonomous official-result "
            "capture to inspect for backlog matching."
        ),
    )
    run_parser.add_argument(
        "--result-backlog-lookback-days",
        type=int,
        default=DEFAULT_FULL_DAEMON_RESULT_BACKLOG_LOOKBACK_DAYS,
    )

    odds_parser = subparsers.add_parser(
        "run-odds-capture-once",
        help="Run one locked autonomous live-odds capture cycle",
    )
    odds_parser.add_argument("--run-id")
    odds_parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    odds_parser.add_argument("--output-dir", type=Path)
    odds_parser.add_argument("--current-time")
    odds_parser.add_argument("--db", type=Path, default=ROOT / "greyhound_racing_data.db")
    odds_parser.add_argument("--days-ahead", type=int, default=1)
    odds_parser.add_argument("--refresh-limit", type=int, default=DEFAULT_ODDS_CAPTURE_ONLY_REFRESH_LIMIT)
    odds_parser.add_argument(
        "--odds-capture-min-minutes",
        type=float,
        default=autopilot.DEFAULT_ODDS_CAPTURE_MIN_MINUTES,
    )
    odds_parser.add_argument(
        "--odds-capture-max-minutes",
        type=float,
        default=autopilot.DEFAULT_ODDS_CAPTURE_MAX_MINUTES,
    )
    odds_parser.add_argument(
        "--odds-capture-refresh-limit",
        type=int,
        default=DEFAULT_ODDS_CAPTURE_ONLY_REFRESH_LIMIT,
    )
    odds_parser.add_argument(
        "--refresh-command-mode",
        choices=("auto", "python", "uv"),
        default="auto",
    )
    odds_parser.add_argument(
        "--require-safe-refresh-metadata",
        dest="require_safe_refresh_metadata",
        action="store_true",
        default=True,
    )
    odds_parser.add_argument(
        "--allow-incomplete-refresh-metadata",
        dest="require_safe_refresh_metadata",
        action="store_false",
    )
    odds_parser.add_argument("--skip-primary-refresh", action="store_true")
    odds_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_ODDS_CAPTURE_ONLY_TIMEOUT_SECONDS,
    )
    odds_parser.add_argument("--lock-path", type=Path)
    odds_parser.add_argument("--lock-stale-seconds", type=int, default=DEFAULT_LOCK_STALE_SECONDS)
    odds_parser.add_argument("--state-path", type=Path, default=DEFAULT_ODDS_CAPTURE_ONLY_STATE_PATH)

    service_parser = subparsers.add_parser("write-service-files", help="Write systemd unit templates")
    service_parser.add_argument("--service-dir", type=Path, default=DEFAULT_SERVICE_DIR)
    service_parser.add_argument("--repo-path", type=Path, default=Path("/home/l4nd0/greyhound_racing_collector"))
    service_parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    service_parser.add_argument("--python-path", type=Path, default=Path(sys.executable))
    service_parser.add_argument("--evidence-root", type=Path)
    service_parser.add_argument("--shadow-model", type=Path)
    service_parser.add_argument("--db", type=Path)
    service_parser.add_argument("--lock-path", type=Path)
    service_parser.add_argument("--state-path", type=Path)
    service_parser.add_argument("--odds-capture-state-path", type=Path)

    odds_service_parser = subparsers.add_parser(
        "write-odds-capture-service-files",
        help="Write systemd unit templates for the odds-capture-only lane",
    )
    odds_service_parser.add_argument("--service-dir", type=Path, default=DEFAULT_SERVICE_DIR)
    odds_service_parser.add_argument("--repo-path", type=Path, default=Path("/home/l4nd0/greyhound_racing_collector"))
    odds_service_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_ODDS_CAPTURE_ONLY_TIMEOUT_SECONDS,
    )
    odds_service_parser.add_argument("--python-path", type=Path, default=Path(sys.executable))
    odds_service_parser.add_argument("--evidence-root", type=Path)
    odds_service_parser.add_argument("--db", type=Path)
    odds_service_parser.add_argument("--lock-path", type=Path)
    odds_service_parser.add_argument("--state-path", type=Path)
    odds_service_parser.add_argument(
        "--refresh-limit",
        type=int,
        default=DEFAULT_ODDS_CAPTURE_ONLY_REFRESH_LIMIT,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "run-odds-capture-once":
        result = run_odds_capture_once(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("final_status") != "ODDS_CAPTURE_ONLY_FAILED" else 2
    if args.command == "write-service-files":
        result = write_service_files(
            service_dir=args.service_dir,
            repo_path=args.repo_path,
            timeout_seconds=args.timeout_seconds,
            python_path=args.python_path,
            evidence_root=args.evidence_root,
            shadow_model=args.shadow_model,
            db_path=args.db,
            lock_path=args.lock_path,
            state_path=args.state_path,
            odds_capture_state_path=args.odds_capture_state_path,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "write-odds-capture-service-files":
        result = write_odds_capture_service_files(
            service_dir=args.service_dir,
            repo_path=args.repo_path,
            timeout_seconds=args.timeout_seconds,
            python_path=args.python_path,
            evidence_root=args.evidence_root,
            db_path=args.db,
            lock_path=args.lock_path,
            state_path=args.state_path,
            refresh_limit=args.refresh_limit,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    result = run_once(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("final_verdict") != "NEEDS_MORE_AUTOMATION" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
