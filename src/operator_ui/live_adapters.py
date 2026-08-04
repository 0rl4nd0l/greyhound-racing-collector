"""Read-only adapters for the repository's actual operational producers.

File names are symbolic keys installed by the server.  Unit bytes are passed as
server observations; the browser cannot provide a path, unit, schedule, or
deadline.  This module performs no discovery and never invokes a command.
"""

from __future__ import annotations

import hashlib
import math
import re
import base64
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from race_collection.synchronous_manual_capture import (
    CURRENT_RACE_INDEX_SCHEMA,
    MAX_CURRENT_INDEX_BYTES,
    CaptureOneRejected,
    VerifiedCurrentRaceIndex,
    bounded_current_race_index,
)
from src.predictor.on_demand import (
    PREDICTION_BUNDLE_INDEX_SCHEMA,
    PredictionBlocked,
    VerifiedPredictionBundleIndex,
    verify_indexed_prediction_bundle,
    verify_prediction_bundle_index,
)

from .api import APIObservation
from .foundation import (
    EvidenceEnvelope,
    OperatorEvidenceReader,
    _bind_path,
    _digest_regular_file,
    _new_envelope,
)

_DURATION = re.compile(
    r"^(?P<n>[+-]?(?:[0-9]+(?:\.[0-9]+)?|inf(?:inity)?|nan))\s*"
    r"(?P<u>us|usec|ms|msec|s|sec|second|seconds|m|min|minute|minutes|"
    r"h|hr|hour|hours|d|day|days)?$",
    re.IGNORECASE,
)
_HASH = re.compile(r"^[0-9a-f]{64}$")
_GIT_ID = re.compile(r"^[0-9a-f]{40}$")
_MARKET_FORM_RESIDUAL_MODEL_SHA256 = "624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d"
_MARKET_FORM_RESIDUAL_MANIFEST_SHA256 = "8537cbc3d843d106a1fe48793ef01197454ef092c0244025fd65685636a42080"
_FULL_SCHEMA = "shadow_autopilot_daemon_run_v1"
_ODDS_SCHEMA = "shadow_autopilot_odds_capture_only_daemon_report_v1"
_ODDS_STATE_SCHEMA = "shadow_autopilot_odds_capture_only_state_v1"
_FULL_ACTIVE = {
    ("RUNNING", "DAEMON_RUNNING"),
    ("WAITING_LOCK_HELD", "DAEMON_WAITING_FOR_ODDS_CAPTURE_LOCK"),
}
_FULL_COMPLETED = {
    ("DAEMON_READY", "DAEMON_READY"),
    ("DAEMON_READY_NEEDS_DEPLOYMENT", "DAEMON_READY_NEEDS_DEPLOYMENT"),
    ("PARTIAL_DAEMONIZATION", "PARTIAL_DAEMONIZATION"),
    ("SKIPPED_LOCK_HELD", "PARTIAL_DAEMONIZATION"),
    ("NEEDS_MORE_AUTOMATION", "NEEDS_MORE_AUTOMATION"),
    (None, "DAEMON_DEFERRED_TO_ODDS_CAPTURE_ONLY"),
}
_ODDS_ACTIVE = {"ODDS_CAPTURE_ONLY_RUNNING"}
_ODDS_COMPLETED = {"ODDS_CAPTURE_ONLY_READY", "ODDS_CAPTURE_ONLY_WAITING_FOR_WINDOW", "ODDS_CAPTURE_ONLY_HANDLED_NO_WRITE", "SKIPPED_LOCK_HELD", "SKIPPED_FULL_DAEMON_LOCK_HANDOFF", "ODDS_CAPTURE_ONLY_FAILED"}
_ODDS_ENVELOPE_STATUS = {
    "ODDS_CAPTURE_ONLY_RUNNING": {"RUNNING"},
    "ODDS_CAPTURE_ONLY_READY": {"READY", "READY_WITH_BLOCKED_ATTEMPTS"},
    "ODDS_CAPTURE_ONLY_WAITING_FOR_WINDOW": {"WAITING"},
    "ODDS_CAPTURE_ONLY_HANDLED_NO_WRITE": {"HANDLED_NO_WRITE"},
    "SKIPPED_LOCK_HELD": {"SKIPPED_LOCK_HELD"},
    "SKIPPED_FULL_DAEMON_LOCK_HANDOFF": {"SKIPPED_FULL_DAEMON_LOCK_HANDOFF"},
    "ODDS_CAPTURE_ONLY_FAILED": {"FAILED"},
}
_ODDS_RETAINED_STATE = {"SKIPPED_LOCK_HELD", "SKIPPED_FULL_DAEMON_LOCK_HANDOFF"}
_ODDS_REFRESH_REQUIRED = {
    "ODDS_CAPTURE_ONLY_READY",
    "ODDS_CAPTURE_ONLY_HANDLED_NO_WRITE",
}


class _UnitMissing(ValueError):
    pass


class _UnitConflict(ValueError):
    pass


@dataclass(frozen=True)
class InstalledUnits:
    """Exact installed unit observations, supplied only by server config."""

    full_timer: bytes
    full_service: bytes
    odds_timer: bytes
    odds_service: bytes
    observed_at: datetime
    working_directory: str | None = None
    full_active_state: str | None = None
    full_sub_state: str | None = None
    full_exec_main_pid: int | None = None
    full_unit_name: str | None = None
    odds_active_state: str | None = None
    odds_sub_state: str | None = None
    odds_exec_main_pid: int | None = None
    odds_unit_name: str | None = None
    full_timer_sha256: str | None = None
    full_service_sha256: str | None = None
    odds_timer_sha256: str | None = None
    odds_service_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class UpcomingRaceSource:
    index_path: Path
    evidence_root: Path
    timeout_seconds: float = 1.0


@dataclass(frozen=True, slots=True)
class PredictionBundleSource:
    root: Path


def _route_key(race_id: str) -> str:
    encoded = base64.urlsafe_b64encode(race_id.encode("utf-8")).decode("ascii").rstrip("=")
    key = f"r1.{encoded}"
    if len(key) > 128:
        raise ValueError("race identity cannot be represented by the bounded route grammar")
    return key


def _duration(value: str) -> float:
    match = _DURATION.fullmatch(value.strip())
    if not match:
        raise ValueError("unit duration missing or invalid")
    number = float(match.group("n"))
    unit = (match.group("u") or "s").lower()
    scale = {
        "us": 1e-6, "usec": 1e-6, "ms": 1e-3, "msec": 1e-3,
        "s": 1, "sec": 1, "second": 1, "seconds": 1,
        "m": 60, "min": 60, "minute": 60, "minutes": 60,
        "h": 3600, "hr": 3600, "hour": 3600, "hours": 3600,
        "d": 86400, "day": 86400, "days": 86400,
    }[unit]
    result = number * scale
    if not math.isfinite(result) or result <= 0:
        raise _UnitMissing("unit duration must be finite and positive")
    return result


def _unit(text: bytes, section: str) -> dict[str, list[str]]:
    try:
        lines = text.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("installed unit is not UTF-8") from exc
    current = None
    values: dict[str, list[str]] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            continue
        if current == section and "=" in line:
            key, value = line.split("=", 1)
            values.setdefault(key.strip(), []).append(value.strip())
    return values


def _one(values: Mapping[str, list[str]], name: str) -> str:
    found = values.get(name, [])
    if not found or not found[0]:
        raise _UnitMissing(f"installed unit is missing {name}")
    if len(found) != 1:
        raise _UnitConflict(f"installed unit has conflicting {name}")
    return found[0]


def _calendar_gap(expression: str) -> float:
    """Maximum gap for the generator's finite daily ``*-*-* HH:MM`` form."""
    events: set[int] = set()
    # systemd uses commas inside the hour/minute fields.  Multiple calendar
    # expressions are represented by repeated OnCalendar= directives and are
    # rejected by _one: silently reading only part of a repeat is unsafe.
    for clause in (expression,):
        token = clause.strip().split()
        if len(token) == 1:
            clock = token[0]  # exact clock-only form emitted by the producer
        elif len(token) == 2 and token[0] == "*-*-*":
            clock = token[1]
        else:
            raise ValueError("OnCalendar date form is unsupported or incomplete")
        if ":" not in clock:
            raise ValueError("OnCalendar repeat is unsupported or incomplete")
        clock_parts = clock.split(":")
        if len(clock_parts) not in {2, 3}:
            raise ValueError("OnCalendar clock is unsupported")
        hour_text, minute_text = clock_parts[:2]
        if len(clock_parts) == 3 and clock_parts[2] not in {"0", "00"}:
            raise ValueError("sub-minute OnCalendar repeat is unsupported")
        hours = range(24) if hour_text == "*" else _calendar_values(hour_text, 0, 23)
        minutes = range(60) if minute_text == "*" else _calendar_values(minute_text, 0, 59)
        events.update(hour * 60 + minute for hour in hours for minute in minutes)
    if not events:
        raise ValueError("OnCalendar repeat has no activations")
    ordered = sorted(events)
    gaps = [right - left for left, right in zip(ordered, ordered[1:])]
    gaps.append(1440 + ordered[0] - ordered[-1])
    return float(max(gaps) * 60)


def _calendar_values(text: str, lower: int, upper: int) -> list[int]:
    result: set[int] = set()
    for part in text.split(","):
        if "/" in part:
            base, step_text = part.split("/", 1)
            step = int(step_text)
            if step <= 0:
                raise ValueError("calendar step is invalid")
            start, end = (lower, upper) if base == "*" else _range(base)
            result.update(range(start, end + 1, step))
        elif ".." in part:
            start, end = _range(part)
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    if not result or min(result) < lower or max(result) > upper:
        raise ValueError("calendar value is out of range")
    return sorted(result)


def _range(text: str) -> tuple[int, int]:
    left, right = text.split("..", 1)
    return int(left), int(right)


def _time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("producer timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("producer timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("producer timestamp has no timezone")
    return parsed.astimezone(timezone.utc)


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value.encode()) > 1024:
        raise ValueError("producer identity is missing or unbounded")
    return value


def _sha(value: Any) -> str:
    value = _text(value)
    if not _HASH.fullmatch(value):
        raise ValueError("producer hash is invalid")
    return value


def _canonical_bytes(value: Any) -> bytes:
    import json
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _safe_relative(value: Any) -> str:
    value = _text(value)
    if value.startswith("/") or "\\" in value or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("producer locator is unsafe")
    return value


def _producer_locator(value: Any) -> str:
    """Validate an opaque producer locator without granting path authority."""
    value = _text(value)
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("producer locator contains a control character")
    return value


def _locator_named(value: Any, name: str) -> str:
    value = _producer_locator(value)
    if value.rsplit("/", 1)[-1] != name:
        raise ValueError("producer locator filename is contradictory")
    return value


def _exact_counts(value: Mapping[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, count in value.items():
        if not isinstance(name, str) or not name or type(count) is not int or count < 0:
            raise ValueError("producer count is invalid")
        result[name] = count
    if not result:
        raise ValueError("producer counts are missing")
    return result


_COUNT_LIMIT = 2**53 - 1
_SUMMARY_COUNT_KEYS = {
    "race_union_count", "shadow_prediction_race_count",
    "official_result_artifact_race_count", "official_result_evidence_db_race_count",
    "live_odds_race_count", "strict_prejump_odds_race_count",
    "shadow_races_with_official_result_evidence_db",
    "shadow_races_with_strict_prejump_odds",
    "shadow_races_complete_official_and_strict_odds", "action_counts",
}
_ACTIONS = {
    "not_shadow_scored", "append_official_result_evidence_backlog",
    "capture_official_result", "repair_official_result_runner_set_or_identity_join",
    "collect_future_strict_prejump_odds", "repair_strict_prejump_odds_runner_set",
    "ready_for_unified_evidence_evaluation",
}
_OFFICIAL_RESULT_GAP_ACTIONS = {
    "append_official_result_evidence_backlog", "capture_official_result",
    "repair_official_result_runner_set_or_identity_join",
}
_STRICT_ODDS_GAP_ACTIONS = {
    "collect_future_strict_prejump_odds", "repair_strict_prejump_odds_runner_set",
}
_DECISIONS = {
    "RUN_POST_BACKLOG_UNIFIED_EVALUATION", "RUN_BACKLOG_APPEND",
    "STRICT_PREJUMP_ODDS_COLLECTION_NEXT", "OFFICIAL_RESULT_CAPTURE_NEXT",
    "KEEP_COLLECTING_OR_DATA_MISSING",
}
_REPORT_KEYS = {
    "schema_version", "generated_at", "final_status", "recommended_decision",
    "output_dir", "artifact_roots", "db_path", "official_artifact_summary",
    "shadow_prediction_summary", "db_summary", "latest_backlog_append_report",
    "summary_counts", "scorecard_metrics", "top_gap_races", "inventory_csv",
    "inventory_jsonl", "scorecard_csv", "scorecard_jsonl", "no_write_guarantees",
}


def _bounded_count(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= _COUNT_LIMIT:
        raise ValueError("producer count is invalid")
    return value


def _count_map(value: Any, *, keys: set[str] | None = None) -> dict[str, int]:
    if not isinstance(value, Mapping) or (keys is not None and set(value) != keys):
        raise ValueError("producer count fields are invalid")
    result = {str(key): _bounded_count(count) for key, count in value.items()}
    if keys is None and any(key not in _ACTIONS for key in result):
        raise ValueError("producer action is invalid")
    return result


def _finite_metric(value: Any, *, lower: float | None = None, upper: float | None = None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("producer metric is invalid")
    parsed = float(value)
    if not math.isfinite(parsed) or (lower is not None and parsed < lower) or (upper is not None and parsed > upper):
        raise ValueError("producer metric is out of range")
    return parsed


def _bounded_identity(value: Any) -> str:
    value = _text(value)
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("producer identity contains a control character")
    return value


def _lock_metadata(value: Any, *, release: bool = False) -> None:
    """Validate non-disclosed producer lock metadata as a finite exact shape."""
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ValueError("inventory lock metadata is invalid")
    if release:
        base = {"released", "reason"}
        reason = value.get("reason")
        extra = (
            {"error"} if reason == "lock_unreadable" else
            {"lock"} if reason == "lock_owned_by_other_run" else set()
        )
        if set(value) != base | extra or type(value.get("released")) is not bool:
            raise ValueError("inventory lock release is invalid")
        if reason not in {"released_by_owner", "lock_already_missing", "lock_unreadable", "lock_owned_by_other_run"}:
            raise ValueError("inventory lock release reason is invalid")
        if value["released"] != (reason == "released_by_owner"):
            raise ValueError("inventory lock release closure is contradictory")
        if reason == "lock_unreadable":
            _bounded_identity(value["error"])
        elif reason == "lock_owned_by_other_run":
            lock = value["lock"]
            fields = {"schema_version", "run_id", "pid", "hostname", "started_at", "output_dir", "owner"}
            if not isinstance(lock, Mapping) or set(lock) != fields:
                raise ValueError("inventory lock release owner is invalid")
            if lock.get("schema_version") != "shadow_autopilot_daemon_lock_v1" or _bounded_count(lock.get("pid")) <= 0:
                raise ValueError("inventory lock release owner identity is invalid")
            for field in ("run_id", "hostname", "owner"):
                _bounded_identity(lock.get(field))
            _time(lock.get("started_at"))
            _producer_locator(lock.get("output_dir"))
        return
    required = {"schema_version", "lock_path", "status", "write_allowed"}
    optional = {"error", "pid", "lock", "owned_lock"}
    if not required <= set(value) or set(value) - required - optional:
        raise ValueError("inventory lock status fields are invalid")
    if value.get("schema_version") != "shared_lock_status_v1" or type(value.get("write_allowed")) is not bool:
        raise ValueError("inventory lock status schema is invalid")
    if value.get("status") not in {
        "not_configured", "missing", "unreadable", "invalid_payload",
        "present_without_pid", "stale_dead_pid", "present_pid_permission_unknown",
        "present_live_pid", "lock_path_missing_required", "stale_lock_unlink_failed",
        "lock_race_lost", "acquired_by_backlog_append",
    }:
        raise ValueError("inventory lock status is unknown")
    if value.get("lock_path") is not None:
        _producer_locator(value["lock_path"])
    if "error" in value:
        _bounded_identity(value["error"])
    if "pid" in value and _bounded_count(value["pid"]) <= 0:
        raise ValueError("inventory lock pid is invalid")
    lock_fields = {"schema_version", "run_id", "pid", "hostname", "started_at", "output_dir", "owner"}
    for name in ("lock", "owned_lock"):
        if name not in value:
            continue
        lock = value[name]
        if not isinstance(lock, Mapping) or set(lock) != lock_fields:
            raise ValueError("inventory lock owner fields are invalid")
        if lock.get("schema_version") != "shadow_autopilot_daemon_lock_v1" or _bounded_count(lock.get("pid")) <= 0:
            raise ValueError("inventory lock owner is invalid")
        for field in ("run_id", "hostname", "owner"):
            _bounded_identity(lock.get(field))
        _time(lock.get("started_at"))
        _producer_locator(lock.get("output_dir"))


def _inventory_semantics(report: Mapping[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    """Validate the exact finite fields emitted by build_packet.

    This deliberately does not turn source locators into authority.  The v1
    producer does not hash its scanned inputs or publish a closure manifest.
    """
    if set(report) != _REPORT_KEYS:
        raise ValueError("inventory report fields are invalid")
    summary = report.get("summary_counts")
    if not isinstance(summary, Mapping) or set(summary) != _SUMMARY_COUNT_KEYS:
        raise ValueError("inventory summary fields are invalid")
    counts = {key: _bounded_count(summary[key]) for key in _SUMMARY_COUNT_KEYS - {"action_counts"}}
    actions = _count_map(summary.get("action_counts"))
    total = counts["race_union_count"]
    shadow = counts["shadow_prediction_race_count"]
    if sum(actions.values()) != total or actions.get("not_shadow_scored", 0) != total - shadow:
        raise ValueError("inventory action totals are contradictory")
    for key, value in counts.items():
        if key != "race_union_count" and value > total:
            raise ValueError("inventory count exceeds population")
    for key in (
        "shadow_races_with_official_result_evidence_db",
        "shadow_races_with_strict_prejump_odds",
        "shadow_races_complete_official_and_strict_odds",
    ):
        if counts[key] > shadow:
            raise ValueError("inventory shadow count is contradictory")
    complete = counts["shadow_races_complete_official_and_strict_odds"]
    if complete != actions.get("ready_for_unified_evidence_evaluation", 0):
        raise ValueError("inventory closure count is contradictory")
    decision = report.get("recommended_decision")
    if decision not in _DECISIONS:
        raise ValueError("inventory decision is invalid")
    expected_decision = (
        "RUN_POST_BACKLOG_UNIFIED_EVALUATION" if complete > 0 else
        "RUN_BACKLOG_APPEND" if actions.get("append_official_result_evidence_backlog", 0) > 0 else
        "STRICT_PREJUMP_ODDS_COLLECTION_NEXT" if actions.get("collect_future_strict_prejump_odds", 0) > 0 else
        "OFFICIAL_RESULT_CAPTURE_NEXT" if actions.get("capture_official_result", 0) > 0 else
        "KEEP_COLLECTING_OR_DATA_MISSING"
    )
    if decision != expected_decision:
        raise ValueError("inventory decision contradicts counts")
    db_summary = report.get("db_summary")
    if not isinstance(db_summary, Mapping) or set(db_summary) != {"db_status", "table_status", "counts"}:
        raise ValueError("inventory DB summary is invalid")
    db_status = db_summary.get("db_status")
    if not isinstance(db_status, Mapping) or db_status.get("status") not in {"AVAILABLE", "DATA_MISSING"}:
        raise ValueError("inventory DB status is invalid")
    if db_status.get("db_path") != report.get("db_path"):
        raise ValueError("inventory DB identity is contradictory")
    final = report.get("final_status")
    expected_final = (
        "DATA_MISSING" if db_status["status"] != "AVAILABLE" else
        "RACE_EVIDENCE_INVENTORY_READY_FOR_EVALUATION"
        if decision == "RUN_POST_BACKLOG_UNIFIED_EVALUATION" else
        "RACE_EVIDENCE_INVENTORY_GAPS_FOUND"
    )
    if final != expected_final:
        raise ValueError("inventory final status contradicts closure")
    roots = report.get("artifact_roots")
    official = report.get("official_artifact_summary")
    if not isinstance(roots, list) or not roots or len(roots) > 1024:
        raise ValueError("inventory artifact roots are invalid")
    for root in roots:
        _producer_locator(root)
    official_keys = {
        "input_artifact_root_count", "artifact_discovery",
        "official_result_artifact_dir_count", "official_result_artifact_race_rows",
        "official_result_artifact_runner_rows", "official_result_artifact_race_count",
    }
    if not isinstance(official, Mapping) or set(official) != official_keys or _bounded_count(official.get("input_artifact_root_count")) != len(roots):
        raise ValueError("inventory artifact source identity is contradictory")
    for key in official_keys - {"artifact_discovery", "input_artifact_root_count"}:
        _bounded_count(official.get(key))
    discovery = official.get("artifact_discovery")
    if not isinstance(discovery, list) or len(discovery) != len(roots):
        raise ValueError("inventory artifact discovery is invalid")
    discovery_keys = {
        "input_artifact_dir", "mode", "direct_match",
        "discovered_child_artifact_count", "discovered_child_artifact_dirs",
        "discovered_child_artifact_dirs_truncated",
    }
    discovered_total = 0
    for root, row in zip(roots, discovery, strict=True):
        if not isinstance(row, Mapping) or set(row) != discovery_keys:
            raise ValueError("inventory artifact discovery row is invalid")
        if _producer_locator(row.get("input_artifact_dir")) != root:
            raise ValueError("inventory artifact discovery identity is contradictory")
        mode = row.get("mode")
        if mode not in {"direct_artifact_dir", "recursive_parent_discovery", "missing_artifact_dir"}:
            raise ValueError("inventory artifact discovery mode is invalid")
        if type(row.get("direct_match")) is not bool or type(row.get("discovered_child_artifact_dirs_truncated")) is not bool:
            raise ValueError("inventory artifact discovery flags are invalid")
        child_count = _bounded_count(row.get("discovered_child_artifact_count"))
        children = row.get("discovered_child_artifact_dirs")
        if not isinstance(children, list) or len(children) > 50:
            raise ValueError("inventory artifact discovery children are invalid")
        for child in children:
            _producer_locator(child)
        if len(children) != min(child_count, 50) or row["discovered_child_artifact_dirs_truncated"] != (child_count > 50):
            raise ValueError("inventory artifact discovery truncation is contradictory")
        if (mode == "direct_artifact_dir") != row["direct_match"] or (mode == "recursive_parent_discovery") != (child_count > 0):
            raise ValueError("inventory artifact discovery mode is contradictory")
        discovered_total += 1 if mode in {"direct_artifact_dir", "missing_artifact_dir"} else child_count
    if official["official_result_artifact_dir_count"] > discovered_total:
        raise ValueError("inventory artifact directory count is contradictory")
    shadow_summary = report.get("shadow_prediction_summary")
    shadow_keys = {
        "prediction_file_count", "prediction_file_limit", "prediction_file_scan_truncated",
        "prediction_file_kind_counts", "shadow_prediction_rows", "shadow_prediction_race_count",
    }
    if not isinstance(shadow_summary, Mapping) or set(shadow_summary) != shadow_keys:
        raise ValueError("inventory shadow summary is invalid")
    for key in ("prediction_file_count", "shadow_prediction_rows"):
        _bounded_count(shadow_summary.get(key))
    limit = shadow_summary.get("prediction_file_limit")
    if limit is not None:
        _bounded_count(limit)
    if type(shadow_summary.get("prediction_file_scan_truncated")) is not bool:
        raise ValueError("inventory shadow truncation status is invalid")
    kinds = shadow_summary.get("prediction_file_kind_counts")
    if not isinstance(kinds, Mapping) or any(key not in {"stage2_shadow_predictions.jsonl", "shadow_predictions.jsonl"} for key in kinds):
        raise ValueError("inventory prediction source kind is invalid")
    for value in kinds.values():
        _bounded_count(value)
    if sum(kinds.values()) != shadow_summary["prediction_file_count"]:
        raise ValueError("inventory prediction file counts are contradictory")
    if limit is not None and shadow_summary["prediction_file_count"] > limit:
        raise ValueError("inventory prediction file limit is contradictory")
    if shadow_summary["prediction_file_scan_truncated"] and limit is None:
        raise ValueError("inventory prediction truncation is contradictory")
    _bounded_count(shadow_summary.get("shadow_prediction_race_count"))
    if shadow_summary["shadow_prediction_race_count"] != shadow:
        raise ValueError("inventory shadow summary is contradictory")
    if official["official_result_artifact_race_count"] != counts["official_result_artifact_race_count"]:
        raise ValueError("inventory official summary is contradictory")

    table_names = {
        "autonomous_official_result_evidence_races",
        "autonomous_official_result_evidence_runners", "live_odds",
    }
    table_status = db_summary.get("table_status")
    db_counts = db_summary.get("counts")
    if not isinstance(table_status, Mapping) or not isinstance(db_counts, Mapping):
        raise ValueError("inventory DB tables or counts are invalid")
    if db_status["status"] == "DATA_MISSING":
        if set(db_status) != {"status", "reason", "db_path"} or db_status.get("reason") not in {"db_path_missing", "db_zero_byte"} or table_status or db_counts:
            raise ValueError("inventory unavailable DB shape is invalid")
    else:
        if set(db_status) != {"status", "db_path", "bytes"} or _bounded_count(db_status.get("bytes")) == 0:
            raise ValueError("inventory available DB shape is invalid")
        if set(table_status) != table_names:
            raise ValueError("inventory DB table status is incomplete")
        if any(not isinstance(item, Mapping) or set(item) != {"present"} or type(item["present"]) is not bool for item in table_status.values()):
            raise ValueError("inventory DB table status is invalid")
        count_keys = {
            "official_result_evidence_race_rows", "official_result_evidence_race_count",
            "official_result_evidence_runner_rows", "official_result_evidence_runner_race_count",
            "live_odds_rows", "live_odds_race_count", "strict_live_odds_rows",
            "strict_live_odds_race_count",
        }
        if set(db_counts) != count_keys:
            raise ValueError("inventory DB counts are incomplete")
        for value in db_counts.values():
            _bounded_count(value)
        for table, related in {
            "autonomous_official_result_evidence_races": ("official_result_evidence_race_rows", "official_result_evidence_race_count"),
            "autonomous_official_result_evidence_runners": ("official_result_evidence_runner_rows", "official_result_evidence_runner_race_count"),
            "live_odds": ("live_odds_rows", "live_odds_race_count", "strict_live_odds_rows", "strict_live_odds_race_count"),
        }.items():
            if not table_status[table]["present"] and any(db_counts[key] for key in related):
                raise ValueError("inventory absent DB table has counts")
        if db_counts["official_result_evidence_race_count"] != counts["official_result_evidence_db_race_count"] or db_counts["live_odds_race_count"] != counts["live_odds_race_count"] or db_counts["strict_live_odds_race_count"] != counts["strict_prejump_odds_race_count"]:
            raise ValueError("inventory DB and summary counts disagree")
        if db_counts["strict_live_odds_rows"] > db_counts["live_odds_rows"] or db_counts["strict_live_odds_race_count"] > db_counts["live_odds_race_count"]:
            raise ValueError("inventory strict odds counts are contradictory")
    backlog = report.get("latest_backlog_append_report")
    if not isinstance(backlog, Mapping):
        raise ValueError("inventory backlog report is invalid")
    if backlog.get("status") == "DATA_MISSING":
        if backlog != {"status": "DATA_MISSING", "reason": "no_backlog_append_report_found"}:
            raise ValueError("inventory backlog missing status is invalid")
    elif backlog.get("status") == "FOUND":
        backlog_keys = {
            "status", "path", "final_status", "artifact_count", "processed_count",
            "status_counts", "inserted_race_rows", "inserted_runner_rows",
            "db_write_performed", "shared_lock_status", "shared_lock_release",
        }
        if set(backlog) != backlog_keys:
            raise ValueError("inventory backlog fields are invalid")
        _locator_named(backlog.get("path"), "official_result_evidence_append_backlog_report.json")
        artifact_count = _bounded_count(backlog.get("artifact_count"))
        processed_count = _bounded_count(backlog.get("processed_count"))
        raw_status_counts = backlog.get("status_counts")
        if not isinstance(raw_status_counts, Mapping):
            raise ValueError("inventory backlog counts are invalid")
        status_counts = {str(key): _bounded_count(value) for key, value in raw_status_counts.items()}
        # Backlog item statuses are a different finite vocabulary from race
        # actions, so validate their bounded identities separately.
        if any(not isinstance(key, str) or not key or len(key) > 128 for key in backlog["status_counts"]):
            raise ValueError("inventory backlog item status is invalid")
        if sum(status_counts.values()) != processed_count or processed_count > artifact_count:
            raise ValueError("inventory backlog totals are contradictory")
        for key in ("inserted_race_rows", "inserted_runner_rows"):
            _bounded_count(backlog.get(key))
        final = backlog.get("final_status")
        allowed_finals = {
            "NO_ARTIFACTS", "APPENDED_OFFICIAL_RESULT_EVIDENCE_BACKLOG",
            "BLOCKED_SHARED_LOCK_HELD", "NOOP_ALREADY_PRESENT",
            "NO_DB_WRITE_PERFORMED", "READY_NOT_EXECUTED",
        }
        if final not in allowed_finals or type(backlog.get("db_write_performed")) is not bool:
            raise ValueError("inventory backlog closure status is invalid")
        if backlog["db_write_performed"] != (final == "APPENDED_OFFICIAL_RESULT_EVIDENCE_BACKLOG"):
            raise ValueError("inventory backlog closure disagrees with writes")
        _lock_metadata(backlog["shared_lock_status"])
        _lock_metadata(backlog["shared_lock_release"], release=True)
    else:
        raise ValueError("inventory backlog status is invalid")

    metrics = report.get("scorecard_metrics")
    metric_keys = {
        "schema_version", "evaluation_race_count", "model_top1_accuracy",
        "model_top3_accuracy", "model_mean_winner_rank", "model_logloss",
        "market_top1_accuracy", "market_top3_accuracy", "market_mean_winner_rank",
        "market_logloss", "skipped_race_reason_counts", "skipped_race_action_counts",
        "official_result_gap_action_counts", "strict_odds_gap_action_counts", "metric_notes",
    }
    if not isinstance(metrics, Mapping) or set(metrics) != metric_keys or metrics.get("schema_version") != "race_evidence_scorecard_metrics_v1":
        raise ValueError("inventory scorecard metrics are invalid")
    evaluated = _bounded_count(metrics.get("evaluation_race_count"))
    if evaluated != complete:
        raise ValueError("inventory evaluation and closure counts disagree")
    for name in ("model_top1_accuracy", "model_top3_accuracy", "market_top1_accuracy", "market_top3_accuracy"):
        value = _finite_metric(metrics.get(name), lower=0, upper=1)
        if (evaluated == 0) != (value is None):
            raise ValueError("inventory accuracy availability is contradictory")
    for name in ("model_mean_winner_rank", "market_mean_winner_rank"):
        value = _finite_metric(metrics.get(name), lower=1)
        if (evaluated == 0) != (value is None):
            raise ValueError("inventory rank availability is contradictory")
    for name in ("model_logloss", "market_logloss"):
        value = _finite_metric(metrics.get(name), lower=0)
        if (evaluated == 0) != (value is None):
            raise ValueError("inventory logloss availability is contradictory")
    reason_counts = metrics.get("skipped_race_reason_counts")
    if not isinstance(reason_counts, Mapping) or len(reason_counts) > 64:
        raise ValueError("inventory skipped reason counts are invalid")
    for key, value in reason_counts.items():
        _bounded_identity(key); _bounded_count(value)
    skipped = _count_map(metrics.get("skipped_race_action_counts"))
    skipped_population = total - evaluated
    if (
        sum(reason_counts.values()) != skipped_population
        or sum(skipped.values()) != skipped_population
    ):
        raise ValueError("inventory skipped race counts are contradictory")
    for field, allowed_actions in (
        ("official_result_gap_action_counts", _OFFICIAL_RESULT_GAP_ACTIONS),
        ("strict_odds_gap_action_counts", _STRICT_ODDS_GAP_ACTIONS),
    ):
        gap_counts = _count_map(metrics.get(field))
        if any(action not in allowed_actions for action in gap_counts):
            raise ValueError("inventory gap action is invalid")
        if sum(gap_counts.values()) > shadow:
            raise ValueError("inventory gap action counts are contradictory")
    notes = metrics.get("metric_notes")
    if notes != [
        "report_only_latest_shadow_prediction_per_race_box",
        "official_results_from_append_only_evidence_db",
        "market_baseline_from_latest_strict_sportsbet_odds_per_box",
        "scorecard_gap_action_counts_use_recommended_next_action",
    ]:
        raise ValueError("inventory metric notes are invalid")

    gaps = report.get("top_gap_races")
    gap_keys = {"race_id", "race_date", "venue", "race_number", "recommended_next_action", "shadow_box_count", "official_result_db_box_count", "strict_live_odds_box_count"}
    if not isinstance(gaps, list) or len(gaps) > 20:
        raise ValueError("inventory top gaps are invalid")
    seen: set[str] = set()
    for row in gaps:
        if not isinstance(row, Mapping) or set(row) != gap_keys:
            raise ValueError("inventory top gap row is invalid")
        race_id = _bounded_identity(row.get("race_id"))
        if race_id in seen or row.get("recommended_next_action") not in _ACTIONS - {"not_shadow_scored", "ready_for_unified_evidence_evaluation"}:
            raise ValueError("inventory top gap identity or action is invalid")
        seen.add(race_id)
        for key in ("shadow_box_count", "official_result_db_box_count", "strict_live_odds_box_count"):
            _bounded_count(row.get(key))
        if row["shadow_box_count"] == 0:
            raise ValueError("inventory top gap has no shadow boxes")
        if row.get("race_date") is not None: _bounded_identity(row["race_date"])
        if row.get("venue") is not None: _bounded_identity(row["venue"])
        if row.get("race_number") is not None: _bounded_count(row["race_number"])
    return counts, actions


def _status(envelope: EvidenceEnvelope, status: str, *, policy: str | None = None) -> EvidenceEnvelope:
    if status == "INVALID/INTEGRITY_FAILED":
        envelope = _invalid(envelope)
    values = envelope.to_dict()
    values["status"] = status
    if policy is not None:
        values["freshness_policy"] = policy
    values["reference_hashes"] = tuple(sorted(values["reference_hashes"].items()))
    identity = values["evidence_identity"]
    values["evidence_identity"] = tuple(identity.items()) if identity else None
    return _new_envelope(**values)


def _invalid(envelope: EvidenceEnvelope) -> EvidenceEnvelope:
    values = envelope.to_dict()
    values.update(
        status="INVALID/INTEGRITY_FAILED",
        schema_integrity="failed",
        source_at=None,
        generated_at=None,
        observed_at=None,
        age_seconds=None,
    )
    values["reference_hashes"] = tuple(sorted(values["reference_hashes"].items()))
    identity = values["evidence_identity"]
    values["evidence_identity"] = tuple(identity.items()) if identity else None
    return _new_envelope(**values)


def _missing_or_invalid(envelope: EvidenceEnvelope) -> str:
    return (
        "INVALID/INTEGRITY_FAILED"
        if envelope.status == "INVALID/INTEGRITY_FAILED"
        else "UNAVAILABLE/DATA_MISSING"
    )


def _path(value: Any) -> str:
    value = _producer_locator(value)
    parts = value.split("/")[1 if value.startswith("/") else 0:]
    if "\\" in value or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("producer path identity is unsafe")
    return value


def _same_optional(left: Mapping[str, Any], right: Mapping[str, Any], *fields: str) -> bool:
    """Compare relationships only when the producer supplies them on both sides."""
    return all(
        field not in left or field not in right or left[field] == right[field]
        for field in fields
    )


def _optional_time(value: Any) -> str | None:
    if value is None:
        return None
    return _time(value).isoformat().replace("+00:00", "Z")


def _optional_count(value: Any) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0 or value > 1_000_000_000:
        raise ValueError("producer count is invalid or unbounded")
    return value


def _optional_counts(value: Any) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or len(value) > 64:
        raise ValueError("producer status counts are invalid or unbounded")
    result: dict[str, int] = {}
    for name, count in value.items():
        if not isinstance(name, str) or not name or len(name.encode()) > 128:
            raise ValueError("producer status-count key is invalid or unbounded")
        parsed = _optional_count(count)
        if parsed is None:
            raise ValueError("producer status count is missing")
        result[name] = parsed
    return result


def _operational_context(report: Mapping[str, Any], *, odds: bool) -> dict[str, Any]:
    """Bounded public projection of exact producer lifecycle/lock/capture fields."""
    full_lock_held = (
        not odds
        and report.get("status") == "SKIPPED_LOCK_HELD"
        and report.get("final_verdict") == "PARTIAL_DAEMONIZATION"
    )
    if odds:
        action_field = "next_meaningful_action"
        action_at_field = "next_meaningful_action_at"
        inserted_rows_field = "inserted_live_odds_rows"
        ready_count_field = "ready_count"
        status_counts_field = "status_counts"
        blocked_attempt_count_field = "blocked_attempt_count"
    elif full_lock_held:
        action_field = "last_odds_capture_next_meaningful_action"
        action_at_field = "last_odds_capture_next_meaningful_action_at"
        inserted_rows_field = "last_odds_capture_inserted_live_odds_rows"
        ready_count_field = "last_odds_capture_ready_count"
        status_counts_field = "last_odds_capture_status_counts"
        blocked_attempt_count_field = "last_odds_capture_blocked_attempt_count"
    else:
        action_field = "odds_capture_next_meaningful_action"
        action_at_field = "odds_capture_next_meaningful_action_at"
        inserted_rows_field = "autonomous_live_odds_capture_inserted_rows"
        ready_count_field = "autonomous_live_odds_capture_ready_count"
        status_counts_field = None
        blocked_attempt_count_field = None

    action = report.get(action_field)
    if action is not None:
        action = _text(action)
    action_at = _optional_time(report.get(action_at_field))
    if action_at is not None and action is None:
        raise ValueError("producer action timestamp has no action")

    owner_values = (
        report.get("lock_owner_kind"),
        report.get("lock_owner_run_id"),
        report.get("lock_owner_started_at"),
    )
    if all(value is None for value in owner_values):
        owner = None
    else:
        kind, owner_run_id, started_at = owner_values
        if kind not in {
            "full_daemon", "odds_capture",
            "full_daemon_waiting_for_odds_capture_handoff",
        }:
            raise ValueError("producer lock owner kind is unknown")
        owner = {
            "kind": kind,
            "run_id": _text(owner_run_id),
            "started_at": _optional_time(started_at),
        }
        if owner["started_at"] is None:
            raise ValueError("producer lock owner start is missing")

    return {
        "final_status": report.get("final_status") if odds else None,
        "final_verdict": None if odds else report.get("final_verdict"),
        "status": report.get("status"),
        "next_meaningful_action": action,
        "next_meaningful_action_at": action_at,
        "lock_owner": owner,
        "recent_capture": {
            "inserted_live_odds_rows": _optional_count(report.get(inserted_rows_field)),
            "ready_count": _optional_count(report.get(ready_count_field)),
            "status_counts": _optional_counts(
                None if status_counts_field is None else report.get(status_counts_field)
            ),
            "blocked_attempt_count": _optional_count(
                None
                if blocked_attempt_count_field is None
                else report.get(blocked_attempt_count_field)
            ),
        },
    }


def _lane_data(
    lane: str,
    status: str,
    *,
    run_id: str = "unavailable",
    phase: str = "unavailable",
    cycle_state: str = "unavailable",
    deadline: datetime | None = None,
    age: float | None = None,
    references: Mapping[str, str | None] = {},
    identity: Mapping[str, str] = {},
    **extra: Any,
) -> dict[str, Any]:
    operational_context = extra.pop(
        "operational_context",
        {
            "final_status": None,
            "final_verdict": None,
            "status": None,
            "next_meaningful_action": None,
            "next_meaningful_action_at": None,
            "lock_owner": None,
            "recent_capture": {
                "inserted_live_odds_rows": None,
                "ready_count": None,
                "status_counts": None,
                "blocked_attempt_count": None,
            },
        },
    )
    return {
        "lane": lane,
        "status": status,
        "run_id": run_id,
        "phase": phase,
        "cycle_state": cycle_state,
        "deadline_utc": (
            None
            if deadline is None
            else deadline.isoformat().replace("+00:00", "Z")
        ),
        "state_age_seconds": None if age is None else max(0.0, age),
        "component_identity": dict(identity) or {"evidence": "unavailable"},
        "reference_hashes": {name: digest for name, digest in references.items() if digest is not None},
        "operational_context": operational_context,
        **extra,
    }


class LiveEvidenceAdapters:
    """Adapt finite, server-named real producer sources."""

    def __init__(
        self, reader: OperatorEvidenceReader, *, units: InstalledUnits,
        upcoming_races: UpcomingRaceSource | None = None,
        prediction_bundles: PredictionBundleSource | None = None,
    ):
        if not isinstance(reader, OperatorEvidenceReader) or not isinstance(units, InstalledUnits):
            raise TypeError("reader and installed units are required")
        self._reader = reader
        self._units = units
        self._upcoming_races = upcoming_races
        self._prediction_bundles = prediction_bundles

    @staticmethod
    def _verified_envelope(
        *, now: datetime, policy: str, identity: str, locator: str,
        status: str, source_at: datetime | None = None,
        content_sha256: str | None = None,
        references: Mapping[str, str] | None = None,
        evidence_identity: Mapping[str, str] | None = None,
        availability: str | None = None,
    ) -> EvidenceEnvelope:
        observed = now.astimezone(timezone.utc)
        stamp = observed.isoformat(timespec="microseconds").replace("+00:00", "Z")
        age = None if source_at is None else (observed - source_at.astimezone(timezone.utc)).total_seconds()
        present = status in {"AVAILABLE/FRESH", "STALE", "DIVERGENT"}
        valid = status in {"AVAILABLE/FRESH", "STALE", "DIVERGENT"}
        supported_claim = (
            "Exact producer-verified read-only evidence for this finite resource."
            if present and content_sha256 is not None and evidence_identity
            else "Integrity failed; no operational claim is supported."
            if status == "INVALID/INTEGRITY_FAILED"
            else "Evidence unavailable; no operational claim is supported."
        )
        return _new_envelope(
            source_kind="producer_verified_view", source_identity=identity,
            content_sha256=content_sha256, source_locator=locator,
            source_at=None if source_at is None else source_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            generated_at=None, observed_at=None, server_observed_at=stamp,
            age_seconds=age, freshness_policy=policy,
            availability=availability or ("present" if present else ("missing" if status == "UNAVAILABLE/DATA_MISSING" else "error")),
            schema_integrity="valid" if valid else ("failed" if status == "INVALID/INTEGRITY_FAILED" else "unknown"),
            reference_hashes=tuple(sorted((references or {}).items())),
            evidence_identity=None if evidence_identity is None else tuple(sorted(evidence_identity.items())),
            status=status,
            supported_claim=supported_claim,
        )

    @staticmethod
    def _collector_failure(code: str) -> tuple[str, str | None]:
        if code == "DISCOVERY_TIMEOUT":
            return "UNAVAILABLE/DATA_MISSING", "error"
        if code in {
            "CURRENT_INDEX_UNAVAILABLE", "CURRENT_INDEX_SOURCE_MISSING",
            "CURRENT_INDEX_PUBLICATION_MISSING", "CURRENT_INDEX_REPORT_MISSING",
        }:
            return "UNAVAILABLE/DATA_MISSING", "missing"
        if code == "CURRENT_INDEX_SOURCE_CHANGED":
            return "UNAVAILABLE/DATA_MISSING", "error"
        return "INVALID/INTEGRITY_FAILED", None

    @staticmethod
    def _prediction_failure(code: str) -> str:
        if code == "PREDICTION_BUNDLE_INDEX_UNAVAILABLE":
            return "UNAVAILABLE/DATA_MISSING"
        return "INVALID/INTEGRITY_FAILED"

    def _race_snapshot(self, now: datetime) -> tuple[EvidenceEnvelope, list[dict[str, Any]]]:
        source = self._upcoming_races
        if source is None:
            return self._verified_envelope(now=now, policy="P-UPCOMING-300-PREJUMP", identity=CURRENT_RACE_INDEX_SCHEMA, locator="operator_ui.current_race_index", status="UNAVAILABLE/DATA_MISSING"), []
        try:
            view = bounded_current_race_index(
                current_time=now, timeout_seconds=source.timeout_seconds,
                index_path=source.index_path, evidence_root=source.evidence_root,
                max_age_seconds=1200, return_verified_view=True,
            )
            if not isinstance(view, VerifiedCurrentRaceIndex):
                raise ValueError("producer did not return a v2 verified view")
            generated = _time(view.source_generated_at)
            age = (now.astimezone(timezone.utc) - generated).total_seconds()
            if age < 0:
                raise ValueError("future collector observation")
            races = []
            for row in view.races:
                jump = _time(row["jump_datetime"])
                if jump <= now.astimezone(timezone.utc):
                    continue
                race_id = _text(row["race_id"])
                races.append({
                    "route_id": _route_key(race_id), "race_id": race_id,
                    "source_race_id": race_id, "source_url": row["race_url"],
                    "racing_date": row["date"], "venue": row["venue"],
                    "meeting_slug": None, "race_number": row["race_number"],
                    "jump_utc": jump.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "source_zone": str(datetime.fromisoformat(row["jump_datetime"]).tzinfo),
                    "distance_metres": None, "grade": None,
                    "runners": [{
                        "runner_id": runner["identity"],
                        "source_runner_id": runner["source_native_runner_id"],
                        "box": runner["box_number"], "name": runner["dog_name"],
                        "scratch_state": runner["scratch_state"],
                    } for runner in row["runners"]],
                    "runner_set_sha256": row["runner_set_sha256"],
                })
            envelope = self._verified_envelope(
                now=now, policy="P-UPCOMING-300-PREJUMP", identity=view.schema_version,
                locator="operator_ui.current_race_index", status="AVAILABLE/FRESH" if age <= 300 else "STALE",
                source_at=generated, content_sha256=view.packet_sha256,
                references={"publication": view.publication_sha256, "refresh_report": view.source_refresh_report_sha256, "state": view.state_sha256, "daemon_report": view.report_sha256},
                evidence_identity={"run_id": view.run_id, "schema_version": view.schema_version},
            )
            return envelope, races
        except CaptureOneRejected as exc:
            status, availability = self._collector_failure(exc.code)
            content_sha256 = None
            if status == "INVALID/INTEGRITY_FAILED":
                try:
                    content_sha256, _ = _digest_regular_file(
                        _bind_path(source.index_path, source.evidence_root),
                        MAX_CURRENT_INDEX_BYTES,
                    )
                    availability = "present"
                except (FileNotFoundError, PermissionError, OSError, ValueError):
                    return self._verified_envelope(now=now, policy="P-UPCOMING-300-PREJUMP", identity=CURRENT_RACE_INDEX_SCHEMA, locator="operator_ui.current_race_index", status="UNAVAILABLE/DATA_MISSING", availability="error"), []
            return self._verified_envelope(now=now, policy="P-UPCOMING-300-PREJUMP", identity=CURRENT_RACE_INDEX_SCHEMA, locator="operator_ui.current_race_index", status=status, availability=availability, content_sha256=content_sha256), []
        except FileNotFoundError:
            return self._verified_envelope(now=now, policy="P-UPCOMING-300-PREJUMP", identity=CURRENT_RACE_INDEX_SCHEMA, locator="operator_ui.current_race_index", status="UNAVAILABLE/DATA_MISSING"), []
        except PermissionError:
            return self._verified_envelope(now=now, policy="P-UPCOMING-300-PREJUMP", identity=CURRENT_RACE_INDEX_SCHEMA, locator="operator_ui.current_race_index", status="UNAVAILABLE/DATA_MISSING", availability="unreadable"), []
        except (ValueError, TypeError, KeyError, OSError):
            return self._verified_envelope(now=now, policy="P-UPCOMING-300-PREJUMP", identity=CURRENT_RACE_INDEX_SCHEMA, locator="operator_ui.current_race_index", status="INVALID/INTEGRITY_FAILED"), []

    def upcoming(self, now: datetime) -> APIObservation:
        envelope, races = self._race_snapshot(now)
        return APIObservation(envelope, {"races": races} if envelope.status == "AVAILABLE/FRESH" else {})

    def race_detail(self, route_id: str, now: datetime) -> APIObservation:
        envelope, races = self._race_snapshot(now)
        if envelope.status != "AVAILABLE/FRESH":
            return APIObservation(envelope, {})
        race = next((item for item in races if item["route_id"] == route_id), None)
        if race is None:
            missing = self._verified_envelope(now=now, policy="P-UPCOMING-300-PREJUMP", identity=CURRENT_RACE_INDEX_SCHEMA, locator="operator_ui.current_race_index", status="UNAVAILABLE/DATA_MISSING")
            return APIObservation(missing, {})
        return APIObservation(envelope, {"race": race})

    @staticmethod
    def _prediction_record(bundle: Any) -> dict[str, Any]:
        result = bundle.result
        ready = result["status"] == "PREDICTION_READY"
        blocker = result["blocker"]
        evidence_names = ["bundle_manifest.json", *bundle.manifest["files"].keys()]
        return {
            "prediction_id": result["prediction_id"], "job_id": result["job_id"],
            "race_id": result["race"]["race_id"],
            "model_id": result["model"]["resolved"],
            "model_sha256": result["model"]["artifact_sha256"],
            "config_id": result["config"]["sha256"],
            "config_sha256": result["config"]["sha256"],
            "terminal_status": result["status"],
            "blocker_stage": result["blocker_stage"],
            "blocker_code": None if blocker is None else blocker["code"],
            "probabilities": None if not ready else [{"runner_id": row["identity"], "probability": row["probability"]} for row in result["prediction"]["predictions"]],
            "bundle_sha256": bundle.index_entry["logical_bundle_sha256"],
            "evidence_names": evidence_names,
            "evidence_identities": {"directory": bundle.directory, "runner_set_sha256": result["evidence"]["runner_set_sha256"]},
        }

    def _prediction_index(self, now: datetime) -> tuple[EvidenceEnvelope, VerifiedPredictionBundleIndex | None, list[dict[str, Any]]]:
        source = self._prediction_bundles
        if source is None:
            return self._verified_envelope(now=now, policy="P-BUNDLE-LIST-60", identity=PREDICTION_BUNDLE_INDEX_SCHEMA, locator="operator_ui.prediction_bundle_index", status="UNAVAILABLE/DATA_MISSING"), None, []
        try:
            view = verify_prediction_bundle_index(source.root, return_verified_view=True)
            if not isinstance(view, VerifiedPredictionBundleIndex):
                raise ValueError("producer did not return a verified index view")
            if view.published_at is None:
                raise ValueError("producer index has no publication time")
            published = _time(view.published_at)
            age = (now.astimezone(timezone.utc) - published).total_seconds()
            if age < 0:
                raise ValueError("future index publication")
            records = [self._prediction_record(verify_indexed_prediction_bundle(source.root, entry)) for entry in view.entries]
            status = "AVAILABLE/FRESH" if age <= 60 else "STALE"
            envelope = self._verified_envelope(now=now, policy="P-BUNDLE-LIST-60", identity=view.schema_version, locator="operator_ui.prediction_bundle_index", status=status, source_at=published, content_sha256=view.sha256, references={"index": view.sha256}, evidence_identity={"schema_version": view.schema_version})
            return envelope, view, records
        except FileNotFoundError:
            return self._verified_envelope(now=now, policy="P-BUNDLE-LIST-60", identity=PREDICTION_BUNDLE_INDEX_SCHEMA, locator="operator_ui.prediction_bundle_index", status="UNAVAILABLE/DATA_MISSING"), None, []
        except PermissionError:
            return self._verified_envelope(now=now, policy="P-BUNDLE-LIST-60", identity=PREDICTION_BUNDLE_INDEX_SCHEMA, locator="operator_ui.prediction_bundle_index", status="UNAVAILABLE/DATA_MISSING", availability="unreadable"), None, []
        except PredictionBlocked as exc:
            return self._verified_envelope(now=now, policy="P-BUNDLE-LIST-60", identity=PREDICTION_BUNDLE_INDEX_SCHEMA, locator="operator_ui.prediction_bundle_index", status=self._prediction_failure(exc.code)), None, []
        except (ValueError, TypeError, KeyError, OSError):
            return self._verified_envelope(now=now, policy="P-BUNDLE-LIST-60", identity=PREDICTION_BUNDLE_INDEX_SCHEMA, locator="operator_ui.prediction_bundle_index", status="INVALID/INTEGRITY_FAILED"), None, []

    def recent_predictions(self, now: datetime) -> APIObservation:
        envelope, _, records = self._prediction_index(now)
        return APIObservation(envelope, {"predictions": records} if envelope.status == "AVAILABLE/FRESH" else {})

    def prediction_detail(self, prediction_id: str, now: datetime) -> APIObservation:
        source = self._prediction_bundles
        if source is None:
            missing = self._verified_envelope(now=now, policy="P-IMMUTABLE-HISTORICAL", identity="on_demand_race_prediction_v2", locator="operator_ui.prediction_bundle", status="UNAVAILABLE/DATA_MISSING")
            return APIObservation(missing, {})
        try:
            view = verify_prediction_bundle_index(source.root, return_verified_view=True)
            if not isinstance(view, VerifiedPredictionBundleIndex):
                raise ValueError("producer did not return a verified index view")
            entry = next((item for item in view.entries if item["prediction_id"] == prediction_id), None)
            if entry is None:
                missing = self._verified_envelope(now=now, policy="P-IMMUTABLE-HISTORICAL", identity="on_demand_race_prediction_v2", locator="operator_ui.prediction_bundle", status="UNAVAILABLE/DATA_MISSING")
                return APIObservation(missing, {})
            record = self._prediction_record(
                verify_indexed_prediction_bundle(source.root, entry)
            )
            historical = self._verified_envelope(now=now, policy="P-IMMUTABLE-HISTORICAL", identity="on_demand_race_prediction_v2", locator="operator_ui.prediction_bundle", status="AVAILABLE/FRESH", source_at=_time(entry["generated_at"]), content_sha256=entry["logical_bundle_sha256"], references={"index": view.sha256, "manifest": entry["manifest_sha256"], "bundle": entry["logical_bundle_sha256"]}, evidence_identity={"prediction_id": prediction_id})
            return APIObservation(historical, {"prediction": record})
        except FileNotFoundError:
            missing = self._verified_envelope(now=now, policy="P-IMMUTABLE-HISTORICAL", identity="on_demand_race_prediction_v2", locator="operator_ui.prediction_bundle", status="UNAVAILABLE/DATA_MISSING")
            return APIObservation(missing, {})
        except PermissionError:
            unreadable = self._verified_envelope(now=now, policy="P-IMMUTABLE-HISTORICAL", identity="on_demand_race_prediction_v2", locator="operator_ui.prediction_bundle", status="UNAVAILABLE/DATA_MISSING", availability="unreadable")
            return APIObservation(unreadable, {})
        except PredictionBlocked as exc:
            status = self._prediction_failure(exc.code)
            failed = self._verified_envelope(now=now, policy="P-IMMUTABLE-HISTORICAL", identity="on_demand_race_prediction_v2", locator="operator_ui.prediction_bundle", status=status)
            return APIObservation(failed, {})
        except (ValueError, TypeError, KeyError, OSError):
            invalid = self._verified_envelope(now=now, policy="P-IMMUTABLE-HISTORICAL", identity="on_demand_race_prediction_v2", locator="operator_ui.prediction_bundle", status="INVALID/INTEGRITY_FAILED")
            return APIObservation(invalid, {})

    def _read(
        self, key: str, now: datetime
    ) -> tuple[EvidenceEnvelope, Mapping[str, Any] | None]:
        return self._reader.read_payload(key, server_observed_at=now)

    def _raw(
        self, key: str, now: datetime
    ) -> tuple[EvidenceEnvelope, bytes | None]:
        return self._reader.read_raw(key, server_observed_at=now)

    def _authenticated_raw(
        self, key: str, now: datetime
    ) -> tuple[EvidenceEnvelope, bytes | None, int | None]:
        return self._reader.read_raw_authenticated(key, server_observed_at=now)

    def _lane(self, *, lane: str, now: datetime) -> tuple[EvidenceEnvelope, dict[str, Any]]:
        odds = lane == "ODDS_ONLY"
        state_key, report_key = (("odds_state", "odds_report") if odds else ("full_state", "full_report"))
        state_env, state = self._read(state_key, now)
        report_env, report = self._read(report_key, now)
        if state is None or report is None:
            failed = state_env if state is None else report_env
            outer = _missing_or_invalid(failed)
            child = "INTEGRITY_FAILED" if outer.startswith("INVALID") else "DATA_MISSING"
            return _status(failed, outer), _lane_data(lane, child)
        try:
            timer = _unit(self._units.odds_timer if odds else self._units.full_timer, "Timer")
            service = _unit(self._units.odds_service if odds else self._units.full_service, "Service")
            gap = _calendar_gap(_one(timer, "OnCalendar")) if odds else _duration(_one(timer, "OnUnitInactiveSec"))
            accuracy = _duration(_one(timer, "AccuracySec"))
            timeout = _duration(_one(service, "TimeoutStartSec"))
        except _UnitMissing:
            return _status(report_env, "UNAVAILABLE/DATA_MISSING"), _lane_data(lane, "DATA_MISSING")
        except _UnitConflict:
            return _status(report_env, "DIVERGENT"), _lane_data(lane, "DIVERGENT")
        except (ValueError, OverflowError, TypeError):
            return _status(report_env, "INVALID/INTEGRITY_FAILED"), _lane_data(lane, "INTEGRITY_FAILED")
        try:
            expected_report = _ODDS_SCHEMA if odds else _FULL_SCHEMA
            expected_state = _ODDS_STATE_SCHEMA if odds else "shadow_autopilot_daemon_state_v1"
            if report.get("schema_version") != expected_report or state.get("schema_version") != expected_state:
                raise ValueError("daemon state/report schema is invalid")
            run_id = _text(report.get("run_id"))
            state_run_id = state.get("run_id") if odds else state.get("last_run_id")
            report_output = _path(report.get("output_dir"))
            state_output = _path(state.get("output_dir") if odds else state.get("last_output_dir"))
            report_at = _time(report.get("generated_at"))
            state_at = _time(state.get("updated_at")) if odds else None
        except (ValueError, OverflowError, TypeError):
            return _status(report_env, "INVALID/INTEGRITY_FAILED"), _lane_data(lane, "INTEGRITY_FAILED")
        observed = now.astimezone(timezone.utc)
        if report_at > observed or (state_at is not None and state_at > observed):
            return _status(report_env, "INVALID/INTEGRITY_FAILED"), _lane_data(lane, "INTEGRITY_FAILED", run_id=run_id)
        raw_status = report.get("status")
        report_verdict = report.get("final_status") if odds else report.get("final_verdict")
        if (raw_status is not None and not isinstance(raw_status, str)) or not isinstance(report_verdict, str):
            return _status(report_env, "INVALID/INTEGRITY_FAILED"), _lane_data(lane, "INTEGRITY_FAILED", run_id=run_id)
        if odds:
            if report_verdict not in _ODDS_ACTIVE | _ODDS_COMPLETED:
                return _status(report_env, "INVALID/INTEGRITY_FAILED"), _lane_data(lane, "INTEGRITY_FAILED", run_id=run_id)
            if raw_status not in _ODDS_ENVELOPE_STATUS[report_verdict]:
                return _status(report_env, "DIVERGENT"), _lane_data(lane, "DIVERGENT", run_id=run_id)
            active = report_verdict in _ODDS_ACTIVE
        else:
            if (raw_status, report_verdict) in _FULL_ACTIVE:
                active = True
            elif (raw_status, report_verdict) in _FULL_COMPLETED:
                active = False
            elif raw_status in {pair[0] for pair in _FULL_ACTIVE | _FULL_COMPLETED} or report_verdict in {pair[1] for pair in _FULL_ACTIVE | _FULL_COMPLETED}:
                return _status(report_env, "DIVERGENT"), _lane_data(lane, "DIVERGENT", run_id=run_id)
            else:
                return _status(report_env, "INVALID/INTEGRITY_FAILED"), _lane_data(lane, "INTEGRITY_FAILED", run_id=run_id)
        state_status = state.get("final_status") if odds else state.get("last_verdict")
        common_lifecycle_fields = (
            "status",
            "runtime_action",
            "readiness_decision",
            "next_meaningful_action",
            "next_meaningful_action_at",
            "inserted_live_odds_rows",
            "ready_count",
            "status_counts",
            "blocked_attempt_count",
        )
        retained = (
            odds and report_verdict in _ODDS_RETAINED_STATE
        ) or (
            not odds and (raw_status, report_verdict) in {
                ("SKIPPED_LOCK_HELD", "PARTIAL_DAEMONIZATION"),
                (None, "DAEMON_DEFERRED_TO_ODDS_CAPTURE_ONLY"),
            }
        )
        if (
            odds
            and not retained
            and state_run_id == run_id
            and not _same_optional(state, report, *common_lifecycle_fields)
        ):
            return _status(report_env, "DIVERGENT"), _lane_data(
                lane, "DIVERGENT", run_id=run_id
            )
        try:
            operational_context = _operational_context(report, odds=odds)
            owner_started_at = (
                operational_context["lock_owner"]["started_at"]
                if operational_context["lock_owner"] is not None else None
            )
            if owner_started_at is not None and _time(owner_started_at) > observed:
                raise ValueError("producer lock-owner timestamp is future")
        except (ValueError, OverflowError, TypeError):
            return _status(report_env, "INVALID/INTEGRITY_FAILED"), _lane_data(
                lane, "INTEGRITY_FAILED", run_id=run_id
            )
        lock_owner = operational_context["lock_owner"]
        if active and lock_owner is not None and lock_owner["run_id"] != run_id:
            return _status(report_env, "DIVERGENT"), _lane_data(
                lane, "DIVERGENT", run_id=run_id
            )
        lock = report.get("lock")
        release = report.get("lock_release")
        if odds and (
            isinstance(lock, Mapping)
            and lock.get("run_id") is not None
            and lock.get("run_id") != run_id
            or isinstance(release, Mapping)
            and release.get("run_id") is not None
            and release.get("run_id") != run_id
        ):
            return _status(report_env, "DIVERGENT"), _lane_data(
                lane, "DIVERGENT", run_id=run_id
            )
        if active:
            active_state = (
                self._units.odds_active_state if odds else self._units.full_active_state
            )
            sub_state = self._units.odds_sub_state if odds else self._units.full_sub_state
            pid = self._units.odds_exec_main_pid if odds else self._units.full_exec_main_pid
            if active_state is None or sub_state is None or pid is None:
                return _status(report_env, "UNAVAILABLE/DATA_MISSING"), _lane_data(
                    lane, "DATA_MISSING", run_id=run_id
                )
            if (
                active_state not in {"active", "activating"}
                or sub_state not in {"running", "start", "start-pre", "start-post"}
                or type(pid) is not int
                or pid <= 0
            ):
                return _status(report_env, "DIVERGENT"), _lane_data(
                    lane, "DIVERGENT", run_id=run_id
                )
        elif retained:
            pass
        elif (
            state_run_id != run_id
            or state_output != report_output
            or state_status != report_verdict
            or not _same_optional(state, report, *common_lifecycle_fields)
        ):
            return _status(report_env, "DIVERGENT"), _lane_data(
                lane, "DIVERGENT", run_id=run_id
            )
        if active:
            deadline = report_at + timedelta(seconds=timeout)
            lifecycle = "ACTIVE"
        else:
            deadline = report_at + timedelta(seconds=gap + accuracy)
            lifecycle = "COMPLETED"
        refresh_env = None
        if odds and not active and (
            report_verdict in _ODDS_REFRESH_REQUIRED
            or isinstance(report.get("odds_capture_refresh_report"), Mapping)
            and bool(report.get("odds_capture_refresh_report"))
        ):
            embedded = report.get("odds_capture_refresh_report")
            if not isinstance(embedded, Mapping) or not embedded:
                return _status(report_env, "INVALID/INTEGRITY_FAILED"), _lane_data(lane, "INTEGRITY_FAILED", run_id=run_id)
            autopilot_dir = report.get("autopilot_output_dir")
            state_autopilot_dir = state.get("autopilot_output_dir")
            try:
                autopilot_dir = _path(autopilot_dir)
                state_autopilot_dir = _path(state_autopilot_dir)
            except (ValueError, TypeError):
                return _status(report_env, "INVALID/INTEGRITY_FAILED"), _lane_data(lane, "INTEGRITY_FAILED", run_id=run_id)
            if state_autopilot_dir != autopilot_dir:
                return _status(report_env, "DIVERGENT"), _lane_data(lane, "DIVERGENT", run_id=run_id)
            if state.get("odds_capture_refresh_status") != embedded.get("status"):
                return _status(report_env, "DIVERGENT"), _lane_data(lane, "DIVERGENT", run_id=run_id)
            try:
                refresh_at = _time(embedded.get("generated_at"))
            except (ValueError, TypeError):
                return _status(report_env, "INVALID/INTEGRITY_FAILED"), _lane_data(lane, "INTEGRITY_FAILED", run_id=run_id)
            if refresh_at > report_at or refresh_at > observed:
                return _status(report_env, "DIVERGENT"), _lane_data(lane, "DIVERGENT", run_id=run_id)
            expected_locator = f"{autopilot_dir}/odds_capture_refresh_report.json"
            try:
                refresh_env, raw_refresh = self._reader.read_verified_payload(
                    "odds_refresh", expected_locator, server_observed_at=now
                )
            except (KeyError, ValueError):
                return _status(report_env, "DIVERGENT"), _lane_data(lane, "DIVERGENT", run_id=run_id)
            if raw_refresh is None:
                outer = _missing_or_invalid(refresh_env)
                child = "INTEGRITY_FAILED" if outer.startswith("INVALID") else "DATA_MISSING"
                return _status(refresh_env, outer), _lane_data(lane, child, run_id=run_id)
            if dict(raw_refresh) != dict(embedded):
                return _status(refresh_env, "DIVERGENT"), _lane_data(lane, "DIVERGENT", run_id=run_id)
            if refresh_env.age_seconds is None or refresh_env.generated_at != embedded.get("generated_at"):
                return _status(refresh_env, "INVALID/INTEGRITY_FAILED"), _lane_data(lane, "INTEGRITY_FAILED", run_id=run_id)
            deadline = min(deadline, refresh_at + timedelta(seconds=gap + accuracy))
        if observed > deadline:
            outer_status, lane_status = "STALE", "STALE"
        elif active:
            outer_status, lane_status = "AVAILABLE/FRESH", "ACTIVE"
        elif odds:
            lane_status = {
                "ODDS_CAPTURE_ONLY_READY": "RECEIPT_READY",
                "ODDS_CAPTURE_ONLY_WAITING_FOR_WINDOW": "CAPTURE_WINDOW_CLOSED",
                "ODDS_CAPTURE_ONLY_HANDLED_NO_WRITE": "CAPTURE_FAILED",
                "SKIPPED_LOCK_HELD": "CAPTURE_WINDOW_CLOSED",
                "SKIPPED_FULL_DAEMON_LOCK_HANDOFF": "CAPTURE_WINDOW_CLOSED",
                "ODDS_CAPTURE_ONLY_FAILED": "CAPTURE_FAILED",
            }[report_verdict]
            if raw_status == "READY_WITH_BLOCKED_ATTEMPTS":
                lane_status = "CAPTURE_FAILED"
            outer_status = "AVAILABLE/FRESH" if lane_status == "RECEIPT_READY" else "UNAVAILABLE/DATA_MISSING"
        else:
            if raw_status == "SKIPPED_LOCK_HELD":
                lane_status = "CAPTURE_WINDOW_CLOSED"
            else:
                lane_status = {
                    "DAEMON_READY": "RECEIPT_READY",
                    "DAEMON_READY_NEEDS_DEPLOYMENT": "DATA_MISSING",
                    "PARTIAL_DAEMONIZATION": "CAPTURE_FAILED",
                    "NEEDS_MORE_AUTOMATION": "CAPTURE_FAILED",
                    "DAEMON_DEFERRED_TO_ODDS_CAPTURE_ONLY": "CAPTURE_WINDOW_CLOSED",
                }[report_verdict]
            outer_status = "AVAILABLE/FRESH" if lane_status == "RECEIPT_READY" else "UNAVAILABLE/DATA_MISSING"
        references = {"state": state_env.content_sha256, "report": report_env.content_sha256}
        if refresh_env is not None:
            references["odds_refresh"] = refresh_env.content_sha256
        return _status(report_env, outer_status), _lane_data(
            lane, lane_status, run_id=run_id, phase=report_verdict,
            cycle_state=lifecycle, deadline=deadline,
            age=(observed - report_at).total_seconds(), references=references,
            identity={
                "state_schema": str(state.get("schema_version")),
                "report_schema": expected_report,
                "cadence_seconds": str(gap),
                "accuracy_seconds": str(accuracy),
                "timeout_seconds": str(timeout),
                **({"odds_refresh_age_seconds": str(refresh_env.age_seconds)} if refresh_env is not None else {}),
            },
            operational_context=operational_context,
        )

    def collector(self, now: datetime) -> APIObservation:
        full_env, full = self._lane(lane="FULL_DAEMON", now=now)
        odds_env, odds = self._lane(lane="ODDS_ONLY", now=now)
        # Collector health includes its installed/deployed identity.  Unit
        # activity is displayed by system(), but cannot turn either lane green.
        deployed = self.system(now)
        statuses = {full_env.status, odds_env.status, deployed.evidence.status}
        worst = next((value for value in ("INVALID/INTEGRITY_FAILED", "DIVERGENT", "UNAVAILABLE/DATA_MISSING", "STALE") if value in statuses), "AVAILABLE/FRESH")
        return APIObservation(_status(full_env, worst, policy="P-COLLECTOR-AGGREGATE"), {"lanes": [full, odds]})

    def system(self, now: datetime) -> APIObservation:
        envelope, payload = self._read("deployment_manifest", now)
        if payload is None:
            return APIObservation(_status(envelope, _missing_or_invalid(envelope)), {})
        manifest_fields = frozenset({
            "schema_version", "generated_at", "source_commit", "source_tree",
            "deployed_commit", "deployed_tree", "working_directory",
            "installed_unit_sha256",
        })
        required = ("source_commit", "source_tree", "deployed_commit", "deployed_tree")
        try:
            if (
                set(payload) != manifest_fields
                or payload["schema_version"] != "operator_ui_deployment_manifest_v1"
            ):
                raise ValueError("deployment manifest schema or fields are invalid")
            missing_identity = {name for name in required if payload.get(name) is None}
            if any(
                payload.get(name) is not None
                and not isinstance(payload.get(name), str)
                for name in required
            ):
                raise ValueError("deployment identity is invalid")
            if any(
                payload.get(name) is not None
                and not _GIT_ID.fullmatch(payload[name])
                for name in required
            ):
                raise ValueError("deployment Git identity is invalid")
            manifest_units = payload.get("installed_unit_sha256")
            missing_manifest_units = manifest_units is None
            if manifest_units is not None and not isinstance(manifest_units, Mapping):
                raise ValueError("deployment unit hashes are invalid")
            unit_names = frozenset(
                {"full_timer", "full_service", "odds_timer", "odds_service"}
            )
            if manifest_units is not None and set(manifest_units) != unit_names:
                raise ValueError("deployment unit hash fields are invalid")
            if manifest_units is not None and any(
                not isinstance(value, str) or not _HASH.fullmatch(value)
                for value in manifest_units.values()
            ):
                raise ValueError("deployment unit hash is invalid")
            if (
                self._units.observed_at.tzinfo is None
                or self._units.observed_at.utcoffset() is None
            ):
                raise ValueError("installed observation time is invalid")
            observed_now = now.astimezone(timezone.utc)
            unit_observed = self._units.observed_at.astimezone(timezone.utc)
            unit_age = (observed_now - unit_observed).total_seconds()
            envelope_age = envelope.age_seconds
            envelope_times = [
                value
                for value in (
                    envelope.source_at,
                    envelope.generated_at,
                    envelope.observed_at,
                )
                if value is not None
            ]
            if (
                envelope_age is None
                or len(envelope_times) != 1
                or not math.isfinite(unit_age)
                or not math.isfinite(envelope_age)
                or unit_age < 0
                or envelope_age < 0
            ):
                raise ValueError("deployment constituent time is invalid")
            envelope_observed = _time(envelope_times[0])
            if not math.isclose(
                envelope_age,
                (observed_now - envelope_observed).total_seconds(),
                abs_tol=1e-6,
            ):
                raise ValueError("deployment constituent age is inconsistent")
            observed = min(unit_observed, envelope_observed)
            age = (observed_now - observed).total_seconds()
            services = (
                _unit(self._units.full_service, "Service"),
                _unit(self._units.odds_service, "Service"),
            )
            working_directories = {_one(service, "WorkingDirectory") for service in services}
            if len(working_directories) != 1:
                raise ValueError("installed working directories conflict")
            installed_working_directory = working_directories.pop()
            if not installed_working_directory.startswith("/") or any(
                part in {"", ".", ".."}
                for part in installed_working_directory.split("/")[1:]
            ):
                raise ValueError("installed working directory is invalid")
            missing_working_directory = (
                self._units.working_directory is None
                or payload.get("working_directory") is None
            )
            statuses = {
                "full": (self._units.full_unit_name, self._units.full_active_state, self._units.full_sub_state, self._units.full_exec_main_pid),
                "odds": (self._units.odds_unit_name, self._units.odds_active_state, self._units.odds_sub_state, self._units.odds_exec_main_pid),
            }
            missing_service_status = any(
                value is None for values in statuses.values() for value in values
            )
            expected_units = {"full": "shadow-autopilot.service", "odds": "shadow-autopilot-odds-capture.service"}
            for lane, (unit_name, active_state, sub_state, pid) in statuses.items():
                if unit_name is not None and unit_name != expected_units[lane]:
                    raise ValueError("installed service unit identity is invalid")
                if active_state is not None:
                    _text(active_state)
                if sub_state is not None:
                    _text(sub_state)
                if pid is not None and (type(pid) is not int or pid < 0):
                    raise ValueError("installed service process observation is invalid")
        except (ValueError, OverflowError, TypeError):
            return APIObservation(_invalid(envelope), {})
        units = {name: hashlib.sha256(getattr(self._units, name)).hexdigest() for name in ("full_timer", "full_service", "odds_timer", "odds_service")}
        supplied_unit_hashes = {
            name: getattr(self._units, f"{name}_sha256") for name in units
        }
        try:
            if any(
                digest is not None
                and (not isinstance(digest, str) or not _HASH.fullmatch(digest))
                for digest in supplied_unit_hashes.values()
            ):
                raise ValueError("supplied unit hash is invalid")
        except (ValueError, TypeError):
            return APIObservation(_status(envelope, "INVALID/INTEGRITY_FAILED"), {})
        missing_supplied_unit_hash = any(
            digest is None for digest in supplied_unit_hashes.values()
        )
        incomplete = (
            missing_identity
            or missing_manifest_units
            or missing_working_directory
            or missing_supplied_unit_hash
            or missing_service_status
        )
        mismatch = (
            (
                payload.get("source_commit") is not None
                and payload.get("deployed_commit") is not None
                and payload["source_commit"] != payload["deployed_commit"]
            )
            or (
                payload.get("source_tree") is not None
                and payload.get("deployed_tree") is not None
                and payload["source_tree"] != payload["deployed_tree"]
            )
            or (
                payload.get("working_directory") is not None
                and payload["working_directory"] != installed_working_directory
            )
            or (
                self._units.working_directory is not None
                and self._units.working_directory != installed_working_directory
            )
            or (
                manifest_units is not None
                and any(manifest_units[name] != digest for name, digest in units.items())
            )
            or any(
                supplied_unit_hashes[name] is not None
                and supplied_unit_hashes[name] != digest
                for name, digest in units.items()
            )
        )
        status = (
            "DIVERGENT"
            if mismatch
            else "STALE"
            if age > 60
            else "DEGRADED"
            if incomplete
            else "HEALTHY"
        )
        outer = (
            "AVAILABLE/FRESH"
            if status == "HEALTHY"
            else "UNAVAILABLE/DATA_MISSING"
            if status == "DEGRADED"
            else status
        )
        return APIObservation(_status(envelope, outer), {"components": [{
            **{name: payload.get(name) for name in required}, "status": status,
            "component": "operator-ui-deployment", "version": str(payload.get("schema_version")),
            "observed_at": observed.isoformat().replace("+00:00", "Z"),
            "age_seconds": age, "reference_hashes": None if incomplete else units,
            "service_status": {
                lane: {"active_state": values[1], "sub_state": values[2], "exec_main_pid": values[3]}
                for lane, values in statuses.items()
            },
        }]})

    def corpus(self, now: datetime) -> APIObservation:
        envelope, report = self._read("corpus_report", now)
        manifest_env, manifest = self._read("corpus_manifest", now)
        if report is None or manifest is None:
            failed = envelope if report is None else manifest_env
            return APIObservation(_status(failed, _missing_or_invalid(failed)), {})
        try:
            if report.get("schema_version") != "race_evidence_inventory_report_v1" or manifest.get("schema_version") != "race_evidence_inventory_output_manifest_v1":
                raise ValueError("inventory producer schema is invalid")
            if set(manifest) != {"schema_version", "output_dir", "files"}:
                raise ValueError("inventory manifest envelope is invalid")
            if not isinstance(now, datetime) or now.tzinfo is None:
                raise ValueError("server time is invalid")
            generated = _time(report.get("generated_at"))
            age = (now.astimezone(timezone.utc) - generated).total_seconds()
            if not math.isfinite(age) or age < 0:
                raise ValueError("inventory producer time is inconsistent")
            output_dir = _producer_locator(report.get("output_dir"))
            if _producer_locator(manifest.get("output_dir")) != output_dir:
                return APIObservation(_status(envelope, "DIVERGENT"), {})
            names = {
                "inventory_csv": "race_evidence_inventory.csv",
                "inventory_jsonl": "race_evidence_inventory.jsonl",
                "scorecard_csv": "race_evidence_scorecard.csv",
                "scorecard_jsonl": "race_evidence_scorecard.jsonl",
            }
            locators = {key: _locator_named(report.get(key), name) for key, name in names.items()}
            if any(locator.rsplit("/", 1)[0] != output_dir.rstrip("/") for locator in locators.values()):
                return APIObservation(_status(envelope, "DIVERGENT"), {})
            files = manifest.get("files")
            if not isinstance(files, Mapping) or len(files) != 7:
                return APIObservation(_status(envelope, "DIVERGENT"), {})
            expected_by_name = {
                **{name: f"corpus_{key}" for key, name in names.items()},
                "race_evidence_inventory_report.json": "corpus_report_bytes",
                "SUMMARY.md": "corpus_summary", "final_status.txt": "corpus_final_status",
            }
            expected = {}
            for locator in files:
                checked = _producer_locator(locator)
                filename = checked.rsplit("/", 1)[-1]
                if filename not in expected_by_name or checked.rsplit("/", 1)[0] != output_dir.rstrip("/"):
                    return APIObservation(_status(envelope, "DIVERGENT"), {})
                expected[checked] = expected_by_name[filename]
            if set(expected.values()) != set(expected_by_name.values()):
                return APIObservation(_status(envelope, "DIVERGENT"), {})
            chain = {"report": envelope.content_sha256, "manifest": manifest_env.content_sha256}
            final_status_raw = None
            for locator, key in expected.items():
                item = files[locator]
                if not isinstance(item, Mapping) or set(item) != {"bytes", "sha256"} or type(item["bytes"]) is not int or item["bytes"] < 0:
                    raise ValueError("inventory manifest entry is invalid")
                declared = _sha(item.get("sha256"))
                raw_env, raw, byte_count = self._authenticated_raw(key, now)
                if raw is None and key not in {"corpus_inventory_csv", "corpus_inventory_jsonl"}:
                    return APIObservation(_status(raw_env, _missing_or_invalid(raw_env)), {})
                if raw_env.status == "DIVERGENT":
                    return APIObservation(_status(raw_env, "DIVERGENT"), {})
                raw_hash = raw_env.content_sha256
                if key == "corpus_report_bytes" and raw_hash != envelope.content_sha256:
                    return APIObservation(_status(envelope, "DIVERGENT"), {})
                if declared != raw_hash or item["bytes"] != byte_count:
                    return APIObservation(_status(envelope, "DIVERGENT"), {})
                if key == "corpus_final_status":
                    final_status_raw = raw
                chain[key.removeprefix("corpus_")] = raw_hash
            metrics = report.get("scorecard_metrics")
            if not isinstance(metrics, Mapping) or metrics.get("schema_version") != "race_evidence_scorecard_metrics_v1":
                raise ValueError("inventory counts are missing")
            _inventory_semantics(report)
            if final_status_raw != (str(report.get("final_status")) + "\n").encode():
                return APIObservation(_status(envelope, "DIVERGENT"), {})
            guarantees = report.get("no_write_guarantees")
            if guarantees != {
                "training": False, "production_promotion": False,
                "registry_mutation": False, "production_pointer_update": False,
                "active_model_replacement": False, "db_write": False,
                "label_write": False, "odds_write": False,
                "official_result_write": False, "daemon_control": False,
                "betting_or_ev_action": False, "snapshot_rewrite": False,
                "manifest_rewrite": False,
            }:
                raise ValueError("producer no-write guarantees are invalid")
        except (KeyError, TypeError, ValueError, OverflowError):
            return APIObservation(_invalid(envelope), {})
        if age > 86400:
            return APIObservation(_status(envelope, "STALE"), {})
        # v1 seals only its seven output files.  It does not declare hashes for
        # scanned prediction/result artifacts or the DB, publication/closure
        # evidence, or a hash-bound backlog report.  Consequently neither a
        # population identity nor producer counts may be disclosed as usable.
        return APIObservation(_status(envelope, "AVAILABLE/FRESH"), {
            "reports": [{
                "report_id": hashlib.sha256((envelope.content_sha256 or "").encode()).hexdigest(),
                "status": "UNAVAILABLE",
                "generated_at": report["generated_at"],
                "chain_hashes": chain,
                "admission_gap": (
                    "race_evidence_inventory_report_v1 has no hash-bound input population, "
                    "official-result publication/closure evidence, or referenced-report chain"
                ),
            }]
        })

    def models(self, now: datetime) -> APIObservation:
        envelope, catalog = self._read("model_catalog", now)
        if catalog is None:
            return APIObservation(_status(envelope, _missing_or_invalid(envelope)), {})
        expected = (
            ("market-form-residual-v1", "latest-research", "configs/prediction/manual-default.json", "market_form_residual_v1", "LATEST_RESEARCH", "model_latest"),
            ("market-only", "market-only", "configs/prediction/market-only.json", "market_only_v1", "BASELINE", "model_baseline"),
        )
        try:
            if catalog.get("schema_version") != "on_demand_prediction_config_catalog_v1" or catalog.get("status") != "CONFIGS_AVAILABLE" or set(catalog) != {"schema_version", "status", "configs"}:
                raise ValueError("predictor catalog schema is invalid")
            configs = catalog.get("configs")
            if not isinstance(configs, list) or len(configs) != 2:
                raise ValueError("predictor catalog is not the finite allowlist")
            observed = _time(envelope.observed_at)
            if not isinstance(now, datetime) or now.tzinfo is None:
                raise ValueError("server time is invalid")
            age = (now.astimezone(timezone.utc) - observed).total_seconds()
            if not math.isfinite(age) or age < 0:
                raise ValueError("catalog observation time is inconsistent")
            models = []
            for entry, spec in zip(configs, expected, strict=True):
                name, selector, config_path, resolved, role, prefix = spec
                if not isinstance(entry, Mapping) or set(entry) != {"name", "selector", "config", "config_sha256", "resolved_config", "model"}:
                    raise ValueError("catalog record schema is invalid")
                if (entry.get("name"), entry.get("selector"), entry.get("config")) != (name, selector, config_path):
                    raise ValueError("catalog choice or order is divergent")
                identity = entry.get("model")
                if not isinstance(identity, Mapping) or set(identity) != {"requested", "resolved", "alias_resolved", "model_sha256", "manifest_sha256", "schema_sha256"}:
                    raise ValueError("resolved model identity is invalid")
                if identity.get("requested") != selector or identity.get("resolved") != resolved or identity.get("alias_resolved") is not True:
                    raise ValueError("resolved model identity is divergent")
                config_env, config_raw = self._raw(f"{prefix}_config", now)
                schema_env, schema_raw = self._raw(f"{prefix}_schema", now)
                raw_observations = [(config_env, config_raw), (schema_env, schema_raw)]
                if resolved == "market_form_residual_v1":
                    model_env, model_raw = self._raw(f"{prefix}_artifact", now)
                    manifest_env, manifest_raw = self._raw(f"{prefix}_manifest", now)
                    raw_observations.extend([(model_env, model_raw), (manifest_env, manifest_raw)])
                else:
                    model_env = manifest_env = None
                    model_raw = manifest_raw = None
                missing = next((env for env, raw in raw_observations if raw is None), None)
                if missing is not None:
                    return APIObservation(_status(missing, _missing_or_invalid(missing)), {})
                if any(env.status == "DIVERGENT" for env, _ in raw_observations):
                    return APIObservation(_status(envelope, "DIVERGENT"), {})
                import json
                parsed_config = json.loads(config_raw.decode("utf-8"))
                parsed_schema = json.loads(schema_raw.decode("utf-8"))
                if parsed_config != entry.get("resolved_config") or _sha(entry.get("config_sha256")) != config_env.content_sha256 or _sha(identity.get("schema_sha256")) != schema_env.content_sha256:
                    return APIObservation(_status(envelope, "DIVERGENT"), {})
                expected_variant = "full_strength" if resolved == "market_form_residual_v1" else "market_only_implied"
                expected_bundle = {
                    "current_index_max_age_seconds": 1200,
                    "latency_budget": {"capture_seconds": 60, "discovery_seconds": 12, "lock_seconds": 1, "safety_seconds": 15, "scoring_seconds": 30, "validation_seconds": 8},
                    "receipt_max_age_seconds": 900,
                }
                if parsed_config != {"bundle": expected_bundle, "model": resolved, "schema_version": "on_demand_prediction_config_v1", "variant": expected_variant}:
                    raise ValueError("resolved prediction config semantics are invalid")
                if not isinstance(parsed_schema, Mapping) or set(parsed_schema) != {"type", "additionalProperties", "required", "properties"} or parsed_schema.get("type") != "object" or parsed_schema.get("additionalProperties") is not False or parsed_schema.get("required") != ["schema_version", "model", "variant", "bundle"]:
                    raise ValueError("prediction config schema envelope is invalid")
                properties = parsed_schema.get("properties")
                if not isinstance(properties, Mapping) or set(properties) != {"schema_version", "model", "variant", "bundle"}:
                    raise ValueError("prediction config schema properties are invalid")
                if properties["schema_version"] != {"const": "on_demand_prediction_config_v1", "type": "string"} or properties["model"] != {"const": resolved, "type": "string"}:
                    raise ValueError("prediction config schema identity is invalid")
                expected_variant_rule = ({"enum": ["full_strength", "half_strength"], "type": "string"} if resolved == "market_form_residual_v1" else {"const": "market_only_implied", "type": "string"})
                if properties["variant"] != expected_variant_rule:
                    raise ValueError("prediction config schema variant is invalid")
                bundle_rule = properties["bundle"]
                if not isinstance(bundle_rule, Mapping) or bundle_rule.get("type") != "object" or bundle_rule.get("additionalProperties") is not False or bundle_rule.get("required") != ["receipt_max_age_seconds", "current_index_max_age_seconds", "latency_budget"] or set(bundle_rule.get("properties", {})) != set(expected_bundle):
                    raise ValueError("prediction config schema bundle is invalid")
                bundle_properties = bundle_rule["properties"]
                bounded_integer = {"maximum": 3600, "minimum": 1, "type": "integer"}
                if bundle_properties["receipt_max_age_seconds"] != bounded_integer or bundle_properties["current_index_max_age_seconds"] != bounded_integer:
                    raise ValueError("prediction config schema age bounds are invalid")
                latency = bundle_properties["latency_budget"]
                latency_names = ["discovery_seconds", "lock_seconds", "capture_seconds", "validation_seconds", "scoring_seconds", "safety_seconds"]
                if not isinstance(latency, Mapping) or latency.get("type") != "object" or latency.get("additionalProperties") is not False or latency.get("required") != latency_names or set(latency.get("properties", {})) != set(latency_names) or any(rule != {"maximum": 300, "minimum": 0.01, "type": "number"} for rule in latency["properties"].values()):
                    raise ValueError("prediction config schema latency budget is invalid")
                declared_model, declared_manifest = identity.get("model_sha256"), identity.get("manifest_sha256")
                if resolved == "market_only_v1":
                    if declared_model is not None or declared_manifest is not None:
                        return APIObservation(_status(envelope, "DIVERGENT"), {})
                else:
                    if _sha(declared_model) != model_env.content_sha256 or _sha(declared_manifest) != manifest_env.content_sha256:
                        return APIObservation(_status(envelope, "DIVERGENT"), {})
                    if (
                        model_env.content_sha256 != _MARKET_FORM_RESIDUAL_MODEL_SHA256
                        or manifest_env.content_sha256 != _MARKET_FORM_RESIDUAL_MANIFEST_SHA256
                    ):
                        raise ValueError("frozen model artifact identity is not approved")
                    parsed_model = json.loads(model_raw.decode("utf-8"))
                    parsed_manifest = json.loads(manifest_raw.decode("utf-8"))
                    if not isinstance(parsed_model, Mapping) or parsed_model.get("schema_version") != "market_form_residual_frozen_model_v1" or parsed_model.get("artifact_role") != "shadow_only_not_activated" or parsed_model.get("model_family") != "race_conditional_logit_with_fixed_market_offset":
                        raise ValueError("frozen model identity is invalid")
                    if not isinstance(parsed_manifest, Mapping) or parsed_manifest.get("schema_version") != "market_form_residual_frozen_manifest_v1" or parsed_manifest.get("status") != "FROZEN_MODEL_READY_AWAITING_ACTIVATION" or parsed_manifest.get("model_path") != "artifacts/frozen_models/market_form_residual_v1/model.json" or parsed_manifest.get("model_sha256") != model_env.content_sha256 or parsed_manifest.get("model_schema_version") != parsed_model.get("schema_version"):
                        raise ValueError("frozen model manifest binding is invalid")
                models.append({
                    "model_id": resolved, "model_sha256": declared_model,
                    "config_id": name, "config_sha256": entry["config_sha256"],
                    "manifest_sha256": declared_manifest, "role": role,
                    "evaluation_status": "UNAVAILABLE", "evaluation_claim": None,
                    "slice_id": None, "evaluation_hashes": {},
                })
        except (KeyError, TypeError, ValueError, OverflowError, UnicodeDecodeError):
            return APIObservation(_invalid(envelope), {})
        status = "STALE" if age > 60 else "AVAILABLE/FRESH"
        return APIObservation(_status(envelope, status), {"models": models})
