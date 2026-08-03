#!/usr/bin/env python3
"""Plan and optionally execute provenance-safe pre-jump live odds captures.

The default mode is report-only. Execution requires both ``--execute`` and
``--allow-auto-scrape-odds``. The only execution write is append-only dog-level
WIN and PLACE rows in ``live_odds`` for races whose pre-jump TheDogs sidecar and
Sportsbet runner/box identity both validate exactly.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from race_collection.manual_prediction_collector_request import (  # noqa: E402
    ManualPredictionCollectorProtocol,
    ProtocolRejected,
)
from odds_auto_integrator import fetch_odds_for_target_race  # noqa: E402
from sportsbet_odds_integrator import parse_sportsbet_runner_box_from_text  # noqa: E402
from scripts.daily_race_ingest_shadow_orchestrator import (  # noqa: E402
    is_thedogs_source_url,
    looks_post_result_source_url,
    parse_int_value,
    parse_jump_datetime,
)
from utils.runner_completeness import (  # noqa: E402
    fetch_canonical_runner_set,
    normalise_runner_name,
    parse_runner_rows_from_csv,
)


DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/autonomous_live_odds_capture_"
OUTPUT_ARTIFACT_PREFIX = "autonomous_live_odds_capture_"
CAPTURE_WINDOWS_MINUTES = (60, 30, 10, 2)
REQUIRED_CAPTURE_MARKETS = ("win", "place")
DEFAULT_PLACE_TOPN = 3
ACCEPTED_SPORTSBET_BOX_SOURCES = {"explicit_dom", "runner_text"}
ACCEPTED_DOG_LEVEL_ODDS_LEVELS = {"dog", "runner"}
SCRATCHED_EXPECTED_RUNNER_STATUS_FIELDS = (
    "status",
    "runner_status",
    "participant_status",
    "scratch_status",
)
SCRATCHED_EXPECTED_RUNNER_BOOL_FIELDS = (
    "scratched",
    "is_scratched",
    "was_scratched",
)
SCRATCHED_EXPECTED_RUNNER_STATUSES = {
    "SCR",
    "SCRATCH",
    "SCRATCHED",
    "L/SCR",
    "LSCR",
    "LATE SCR",
    "LATE SCRATCH",
    "LATE SCRATCHED",
}
SCRATCHED_EXPECTED_RUNNER_COMPACT_STATUSES = {
    re.sub(r"[\s_/-]+", "", value) for value in SCRATCHED_EXPECTED_RUNNER_STATUSES
}
REQUIRED_EXISTING_CAPTURE_PROVENANCE_COLUMNS = {
    "capture_timestamp",
    "market_type",
    "odds_level",
    "source",
    "source_url",
    "sportsbet_box_source",
}
DEFAULT_FETCH_TIMEOUT_SECONDS = 45.0
NO_WRITE_GUARANTEES = {
    "training": False,
    "production_promotion": False,
    "registry_mutation": False,
    "production_pointer_update": False,
    "active_model_replacement": False,
    "label_write": False,
    "tgr_enabled": False,
    "betting_or_ev_action": False,
    "snapshot_rewrite": False,
    "odds_history_write": False,
    "race_metadata_write": False,
}
POST_RACE_URL_TOKENS = {"result", "results", "dividend", "dividends", "payout", "payouts"}


class FetchTimeoutError(TimeoutError):
    """Raised when an individual odds source fetch exceeds its fail-closed limit."""


def now_id(now: datetime | None = None) -> str:
    return (now or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        logical = path if path.is_absolute() else ROOT / path
        return logical.absolute().relative_to(ROOT.absolute()).as_posix()
    except ValueError:
        return str(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _fetch_timeout_handler(signum: int, frame: object) -> None:
    raise FetchTimeoutError("sportsbet_fetch_timeout")


def fetch_odds_for_target_race_with_timeout(
    db_path: str,
    venue: Any,
    race_number: Any,
    race_date: Any,
    *,
    allow_auto_scrape_odds: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        return fetch_odds_for_target_race(
            db_path,
            venue,
            race_number,
            race_date,
            allow_auto_scrape_odds=allow_auto_scrape_odds,
        )
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    signal.signal(signal.SIGALRM, _fetch_timeout_handler)
    try:
        return fetch_odds_for_target_race(
            db_path,
            venue,
            race_number,
            race_date,
            allow_auto_scrape_odds=allow_auto_scrape_odds,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def flush_attempt_progress(
    progress_dir: Path | None,
    *,
    attempts: Sequence[Mapping[str, Any]],
    active_attempt: Mapping[str, Any] | None = None,
) -> None:
    if progress_dir is None:
        return
    rows = [dict(row) for row in attempts]
    if active_attempt is not None:
        rows.append(dict(active_attempt))
    write_jsonl(progress_dir / "autonomous_live_odds_capture_attempts.progress.jsonl", rows)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1
    write_json(
        progress_dir / "autonomous_live_odds_capture_progress.json",
        {
            "schema_version": "autonomous_live_odds_capture_progress_v1",
            "generated_at": datetime.now().astimezone().isoformat(),
            "attempt_count": len(rows),
            "status_counts": status_counts,
            "active_attempt": dict(active_attempt) if active_attempt is not None else None,
            "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        },
    )


def blocked_attempt_summaries(
    attempts: Sequence[Mapping[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for attempt in attempts:
        status = str(attempt.get("status") or "")
        if not (status.startswith("BLOCKED") or status == "APPEND_FAILED"):
            continue
        validation = attempt.get("validation")
        if not isinstance(validation, Mapping):
            validation = {}
        fetch_result = attempt.get("fetch_result")
        if not isinstance(fetch_result, Mapping):
            fetch_result = {}
        summaries.append(
            {
                "race_id": attempt.get("race_id"),
                "capture_window_minutes": attempt.get("capture_window_minutes"),
                "status": status,
                "reasons": list(attempt.get("reasons") or []),
                "fetch_time": attempt.get("fetch_time"),
                "append_time": attempt.get("append_time"),
                "fresh_plan_status": attempt.get("fresh_plan_status"),
                "fresh_minutes_to_jump": attempt.get("fresh_minutes_to_jump"),
                "fetch_success": fetch_result.get("success"),
                "fetch_win_count": fetch_result.get("win_count"),
                "fetch_place_count": fetch_result.get("place_count"),
                "validation_status": validation.get("status"),
                "validation_source_url": validation.get("source_url"),
                "validation_expected_runner_count": validation.get(
                    "expected_runner_count"
                ),
                "validation_active_expected_runner_count": validation.get(
                    "active_expected_runner_count"
                ),
                "validation_accepted_row_count": validation.get("accepted_row_count"),
                "validation_failure_root_cause": validation.get("failure_root_cause"),
                "validation_failure_detail": validation.get("failure_detail"),
                "validation_missing_expected_runner_count": len(
                    validation.get("missing_expected_runners") or []
                ),
                "validation_extra_unexpected_runner_count": len(
                    validation.get("extra_unexpected_runners") or []
                ),
            }
        )
        if len(summaries) >= limit:
            break
    return summaries


def validation_failure_detail(
    *,
    active_expected_count: int,
    accepted_row_count: int,
    missing_count: int,
    extra_count: int,
    accepted_place_row_count: int | None = None,
    place_missing_count: int | None = None,
    place_extra_count: int | None = None,
    fetch_result: Mapping[str, Any],
) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "active_expected_runner_count": active_expected_count,
        "accepted_win_row_count": accepted_row_count,
        "missing_active_runner_count": missing_count,
        "extra_unexpected_runner_count": extra_count,
        "fetch_win_count": parse_int_value(fetch_result.get("win_count")),
        "fetch_place_count": parse_int_value(fetch_result.get("place_count")),
    }
    if accepted_place_row_count is not None:
        detail["accepted_place_row_count"] = accepted_place_row_count
    if place_missing_count is not None:
        detail["missing_place_runner_count"] = place_missing_count
    if place_extra_count is not None:
        detail["extra_place_unexpected_runner_count"] = place_extra_count
    root_cause = None
    if extra_count > 0 or (place_extra_count or 0) > 0:
        root_cause = "sportsbet_unexpected_runner_identity_mismatch"
    partial_win_rows = (
        missing_count > 0
        and accepted_row_count > 0
        and accepted_row_count < active_expected_count
    )
    partial_place_rows = (
        (place_missing_count or 0) > 0
        and (accepted_place_row_count or 0) > 0
        and (accepted_place_row_count or 0) < active_expected_count
    )
    if (
        root_cause is None
        and partial_win_rows
        and detail["fetch_place_count"] == active_expected_count
    ):
        root_cause = "sportsbet_win_market_partial_but_place_complete"
    elif root_cause is None and partial_win_rows:
        root_cause = "partial_same_race_win_market"
    elif root_cause is None and partial_place_rows:
        root_cause = "partial_same_race_place_market"
    elif (
        root_cause is None
        and active_expected_count > 0
        and accepted_row_count == active_expected_count
        and accepted_place_row_count == 0
    ):
        root_cause = "sportsbet_place_market_missing"
    if root_cause is not None:
        detail["root_cause"] = root_cause
    return detail


def validation_failure_reason(detail: Mapping[str, Any]) -> str | None:
    root_cause = detail.get("root_cause")
    if not root_cause:
        return None
    return (
        f"{root_cause}:"
        f"active_expected={detail.get('active_expected_runner_count')},"
        f"accepted_win={detail.get('accepted_win_row_count')},"
        f"accepted_place={detail.get('accepted_place_row_count')},"
        f"fetch_win={detail.get('fetch_win_count')},"
        f"fetch_place={detail.get('fetch_place_count')},"
        f"missing_active={detail.get('missing_active_runner_count')},"
        f"missing_place={detail.get('missing_place_runner_count')},"
        f"extra_unexpected={detail.get('extra_unexpected_runner_count')}"
    )


def t2_miss_cause_for_attempt(attempt: Mapping[str, Any]) -> str | None:
    try:
        capture_window_minutes = int(attempt.get("capture_window_minutes"))
    except (TypeError, ValueError):
        return None
    if capture_window_minutes != 2:
        return None
    status = str(attempt.get("status") or "")
    reasons = {str(reason) for reason in attempt.get("reasons") or []}
    if status in {
        "BLOCKED_TIME_GATE_BEFORE_FETCH",
        "BLOCKED_TIME_GATE_BEFORE_APPEND",
    } and "race_already_jumped" in reasons:
        return "t2_miss_late_time_gate"
    if status == "BLOCKED_VALIDATION_FAILED":
        return "t2_miss_validation_failed"
    if status == "BLOCKED_FETCH_TIMEOUT":
        return "t2_miss_fetch_timeout"
    if status == "BLOCKED_FETCH_EXCEPTION":
        return "t2_miss_fetch_exception"
    if status == "BLOCKED_EXISTING_CAPTURE_INCOMPLETE":
        return "t2_miss_existing_capture_incomplete"
    if status == "BLOCKED_EXISTING_CAPTURE_INVALID":
        return "t2_miss_existing_capture_invalid"
    if status == "APPEND_FAILED":
        return "t2_miss_append_failed"
    return None


def t2_miss_cause_summary(
    attempts: Sequence[Mapping[str, Any]],
    *,
    limit: int = 10,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    for attempt in attempts:
        cause = t2_miss_cause_for_attempt(attempt)
        if cause is None:
            continue
        counts[cause] = counts.get(cause, 0) + 1
        if len(examples) < limit:
            examples.append(
                {
                    "race_id": attempt.get("race_id"),
                    "capture_window_minutes": attempt.get("capture_window_minutes"),
                    "status": attempt.get("status"),
                    "cause": cause,
                    "reasons": list(attempt.get("reasons") or []),
                    "fetch_time": attempt.get("fetch_time"),
                    "append_time": attempt.get("append_time"),
                    "fresh_plan_status": attempt.get("fresh_plan_status"),
                    "fresh_minutes_to_jump": attempt.get("fresh_minutes_to_jump"),
                }
            )
    return {
        "schema_version": "autonomous_live_odds_capture_t2_miss_cause_summary_v1",
        "t2_miss_attempt_count": sum(counts.values()),
        "t2_miss_cause_counts": dict(sorted(counts.items())),
        "t2_miss_examples": examples,
    }


def capture_report_operator_fields(
    final_status: str,
    *,
    blocked_attempt_count: int = 0,
) -> dict[str, str]:
    if final_status == "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED":
        if blocked_attempt_count > 0:
            return {
                "status": "APPENDED_WITH_BLOCKED_ATTEMPTS",
                "runtime_action": "REVIEW_CAPTURE_BLOCKERS_AFTER_APPEND",
                "readiness_decision": "CONTINUE_ODDS_CAPTURE_WITH_BLOCKER_REVIEW",
            }
        return {
            "status": "APPENDED",
            "runtime_action": "WAIT_FOR_NEXT_ODDS_CAPTURE_WINDOW",
            "readiness_decision": "CONTINUE_ODDS_CAPTURE",
        }
    if final_status == "AUTONOMOUS_LIVE_ODDS_CAPTURE_READY_NOT_EXECUTED":
        return {
            "status": "READY_REPORT_ONLY",
            "runtime_action": "RUN_WITH_EXECUTE_AND_ALLOW_AUTO_SCRAPE_ODDS_TO_APPEND",
            "readiness_decision": "REPORT_ONLY_NO_WRITE",
        }
    if final_status == "AUTONOMOUS_LIVE_ODDS_CAPTURE_BLOCKED":
        return {
            "status": "BLOCKED",
            "runtime_action": "REVIEW_CAPTURE_BLOCKERS_BEFORE_RETRY",
            "readiness_decision": "CHECK_BLOCKED_ATTEMPTS",
        }
    return {
        "status": "READY",
        "runtime_action": "WAIT_FOR_NEXT_ODDS_CAPTURE_WINDOW",
        "readiness_decision": "CONTINUE_ODDS_CAPTURE",
    }


def parse_current_time(value: str | None) -> datetime:
    if not value:
        return datetime.now().astimezone()
    text = value.strip()
    if len(text) >= 5 and text[-5] in {"+", "-"} and text[-4:].isdigit():
        text = f"{text[:-5]}{text[-5:-2]}:{text[-2:]}"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.astimezone()
    return parsed


def parse_date_value(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value).strip()[:10]).date()
    except ValueError:
        return None


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
        raise ValueError(f"output_dir_must_be_autonomous_live_odds_capture_artifact:{relative}")

    evidence_base = evidence_root if evidence_root.is_absolute() else ROOT / evidence_root
    try:
        relative = candidate.relative_to(evidence_base.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo_or_evidence_root") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if relative.parts and relative.parts[0].startswith(OUTPUT_ARTIFACT_PREFIX):
        return candidate
    raise ValueError(f"output_dir_must_be_autonomous_live_odds_capture_artifact:{relative}")


def unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = Path(f"{base}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"output_dir_collision_exhausted:{base}")


def capture_report_identity_fields(output_dir: Path) -> dict[str, str]:
    name = output_dir.name
    prefix = "autonomous_live_odds_capture_"
    run_id = name[len(prefix) :] if name.startswith(prefix) else name
    return {
        "run_id": run_id,
        "output_dir": relpath(output_dir) or str(output_dir),
    }


def sidecar_path_for(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.name + ".metadata.json")


def load_sidecar(csv_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    sidecar = sidecar_path_for(csv_path)
    if not sidecar.exists():
        return None, "sidecar_metadata_missing"
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"sidecar_metadata_unreadable:{type(exc).__name__}"
    if not isinstance(payload, dict):
        return None, "sidecar_metadata_not_object"
    return payload, None


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def sidecar_section(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def sidecar_participants(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    shadow = sidecar_section(payload, "prejump_shadow_metadata")
    if isinstance(shadow.get("runner_box_name_list"), list):
        return [dict(row) for row in shadow["runner_box_name_list"] if isinstance(row, Mapping)]
    for key in ("runner_completeness_after_canonical_alignment", "runner_completeness"):
        section = sidecar_section(payload, key)
        if isinstance(section.get("participants"), list):
            return [dict(row) for row in section["participants"] if isinstance(row, Mapping)]
    return []


def canonical_alignment(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    shadow = sidecar_section(payload, "prejump_shadow_metadata")
    alignment = shadow.get("canonical_final_runner_alignment")
    if not isinstance(alignment, Mapping):
        alignment = payload.get("canonical_runner_alignment")
    return alignment if isinstance(alignment, Mapping) else {}


def participant_identity_map(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    output: dict[int, dict[str, Any]] = {}
    reasons: list[str] = []
    seen_identities: set[str] = set()
    for row in rows:
        box = parse_int_value(row.get("box_number") or row.get("box"))
        dog_name = str(row.get("dog_name") or row.get("name") or "").strip()
        identity = normalise_runner_name(dog_name)
        if box is None or not identity:
            reasons.append("runner_has_missing_box_or_name")
            continue
        if box in output:
            reasons.append(f"duplicate_runner_box:{box}")
        if identity in seen_identities:
            reasons.append(f"duplicate_runner_name:{identity}")
        seen_identities.add(identity)
        participant = {
            "box_number": box,
            "dog_name": dog_name,
            "identity": identity,
        }
        for field in (
            *SCRATCHED_EXPECTED_RUNNER_STATUS_FIELDS,
            *SCRATCHED_EXPECTED_RUNNER_BOOL_FIELDS,
        ):
            if field in row:
                participant[field] = row.get(field)
        output[box] = participant
    return output, sorted(set(reasons))


def csv_participant_identity_map(csv_path: Path) -> tuple[dict[int, dict[str, Any]], list[str]]:
    try:
        runner_rows = parse_runner_rows_from_csv(csv_path)
    except Exception as exc:
        return {}, [f"csv_target_runner_rows_unreadable:{type(exc).__name__}"]
    rows = [
        {"box_number": row.box_number, "dog_name": row.dog_name}
        for row in runner_rows
    ]
    output, reasons = participant_identity_map(rows)
    if not output:
        reasons.append("csv_target_runner_rows_missing")
    return output, sorted(set(reasons))


def runner_set_report(csv_path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    csv_by_box, csv_reasons = csv_participant_identity_map(csv_path)
    sidecar_by_box, sidecar_reasons = participant_identity_map(sidecar_participants(payload))
    reasons = list(csv_reasons) + list(sidecar_reasons)
    csv_boxes = set(csv_by_box)
    sidecar_boxes = set(sidecar_by_box)
    missing_csv_boxes = sorted(sidecar_boxes - csv_boxes)
    extra_csv_boxes = sorted(csv_boxes - sidecar_boxes)
    mismatches = []
    for box in sorted(csv_boxes & sidecar_boxes):
        if csv_by_box[box]["identity"] != sidecar_by_box[box]["identity"]:
            mismatches.append(
                {
                    "box_number": box,
                    "csv_dog_name": csv_by_box[box]["dog_name"],
                    "sidecar_dog_name": sidecar_by_box[box]["dog_name"],
                    "csv_identity": csv_by_box[box]["identity"],
                    "sidecar_identity": sidecar_by_box[box]["identity"],
                }
            )
    if missing_csv_boxes:
        reasons.append(
            "runner_box_name_list_missing_csv_boxes:"
            + ",".join(str(value) for value in missing_csv_boxes)
        )
    if extra_csv_boxes:
        reasons.append(
            "runner_box_name_list_extra_csv_boxes:"
            + ",".join(str(value) for value in extra_csv_boxes)
        )
    if mismatches:
        reasons.append("runner_box_name_list_name_mismatch")
    expected = [
        sidecar_by_box[box]
        for box in sorted(sidecar_by_box)
        if box in csv_by_box and csv_by_box[box]["identity"] == sidecar_by_box[box]["identity"]
    ]
    return {
        "status": "PASS" if not reasons and expected else "FAIL",
        "runner_count": len(expected),
        "expected_runners": expected,
        "csv_runner_count": len(csv_by_box),
        "sidecar_runner_count": len(sidecar_by_box),
        "missing_csv_boxes": missing_csv_boxes,
        "extra_csv_boxes": extra_csv_boxes,
        "name_mismatches": mismatches,
        "reasons": sorted(set(reasons)),
    }


def due_capture_window(minutes_to_jump: float | None) -> tuple[int | None, str]:
    if minutes_to_jump is None:
        return None, "jump_time_missing"
    if minutes_to_jump <= 0:
        return None, "race_already_jumped"
    if minutes_to_jump > max(CAPTURE_WINDOWS_MINUTES):
        return None, "before_first_capture_window"
    for offset in sorted(CAPTURE_WINDOWS_MINUTES):
        if minutes_to_jump <= offset:
            return offset, "due_now_or_passed_pre_jump"
    return None, "capture_window_unavailable"


def canonical_race_identity(venue: Any, race_number: Any, race_date: Any) -> str | None:
    number = parse_int_value(race_number)
    if not venue or number is None or not race_date:
        return None
    return f"Race {number} - {str(venue).strip().upper()} - {race_date}"


def race_time_from_datetime(value: datetime | None) -> str | None:
    return value.strftime("%H:%M") if value is not None else None


def build_plan_item(csv_path: Path, current_time: datetime) -> dict[str, Any]:
    payload, sidecar_error = load_sidecar(csv_path)
    reasons: list[str] = []
    if sidecar_error:
        reasons.append(sidecar_error)
        payload = {}
    assert payload is not None
    shadow = sidecar_section(payload, "prejump_shadow_metadata")
    race_info = sidecar_section(payload, "race_info")
    if shadow.get("status") != "PASS":
        reasons.append(f"prejump_shadow_metadata_status_not_pass:{shadow.get('status')}")
    leakage_safe = payload.get("metadata_is_leakage_safe") is True or (
        shadow.get("metadata_is_leakage_safe") is True
    )
    if not leakage_safe:
        reasons.append("metadata_is_leakage_safe_not_true")

    race_date_value = (
        parse_date_value(shadow.get("race_date"))
        or parse_date_value(race_info.get("date"))
        or parse_date_value(payload.get("race_date"))
        or parse_date_value(payload.get("date"))
    )
    race_date_text = race_date_value.isoformat() if race_date_value else None
    venue = shadow.get("venue") or race_info.get("venue") or payload.get("venue")
    venue_text = str(venue).strip().upper() if venue else None
    race_number = parse_int_value(
        shadow.get("race_number") or race_info.get("race_number") or payload.get("race_number")
    )
    jump_time = (
        shadow.get("jump_time")
        or race_info.get("race_time")
        or race_info.get("jump_time")
        or payload.get("jump_time")
        or payload.get("jump_datetime")
    )
    source_url = (
        shadow.get("source_url")
        or payload.get("metadata_source_url")
        or payload.get("race_url")
        or race_info.get("url")
    )
    source_url = str(source_url).strip() if source_url not in (None, "") else None
    if not race_date_value:
        reasons.append("race_date_missing")
    if not venue_text:
        reasons.append("venue_missing")
    if race_number is None:
        reasons.append("race_number_missing")
    if not jump_time:
        reasons.append("jump_time_missing")
    if not source_url:
        reasons.append("source_url_missing")
    elif not is_thedogs_source_url(source_url):
        reasons.append("source_url_not_thedogs")
    elif looks_post_result_source_url(source_url):
        reasons.append("source_url_looks_post_result")

    jump_dt = None
    jump_error = None
    if race_date_value and jump_time:
        jump_dt, jump_error = parse_jump_datetime(
            race_date=race_date_value,
            jump_time=jump_time,
            current_time=current_time,
        )
        if jump_error:
            reasons.append(jump_error)
    minutes_to_jump = None
    if jump_dt is not None:
        jump_cmp = jump_dt
        current_cmp = current_time
        if jump_cmp.tzinfo is None and current_cmp.tzinfo is not None:
            jump_cmp = jump_cmp.replace(tzinfo=current_cmp.tzinfo)
        if current_cmp.tzinfo is None and jump_cmp.tzinfo is not None:
            current_cmp = current_cmp.replace(tzinfo=jump_cmp.tzinfo)
        minutes_to_jump = (jump_cmp - current_cmp).total_seconds() / 60.0

    alignment = canonical_alignment(payload)
    if alignment.get("status") != "aligned":
        reasons.append("canonical_runner_alignment_not_aligned")
    if alignment.get("canonical_runner_set_status") != "available":
        reasons.append("canonical_runner_set_not_available")
    runner_report = runner_set_report(csv_path, payload)
    if runner_report.get("status") != "PASS":
        reasons.extend(runner_report.get("reasons") or ["runner_set_validation_failed"])

    capture_window, window_status = due_capture_window(minutes_to_jump)
    if capture_window is None:
        reasons.append(window_status)

    race_id = canonical_race_identity(venue_text, race_number, race_date_text)
    status = "READY_TO_CAPTURE" if not reasons else "BLOCKED"
    if status == "BLOCKED" and reasons == [window_status]:
        status = "NO_DUE_WINDOW"

    return {
        "schema_version": "autonomous_live_odds_capture_plan_item_v1",
        "status": status,
        "csv_path": relpath(csv_path),
        "sidecar_path": relpath(sidecar_path_for(csv_path)),
        "race_id": race_id,
        "venue": venue_text,
        "race_number": race_number,
        "race_date": race_date_text,
        "race_time": race_time_from_datetime(jump_dt),
        "jump_datetime": jump_dt.isoformat() if jump_dt else None,
        "minutes_to_jump": minutes_to_jump,
        "capture_window_minutes": capture_window,
        "window_status": window_status,
        "thedogs_source_url": source_url,
        "runner_set_validation": runner_report,
        "expected_runners": runner_report.get("expected_runners") or [],
        "blockers": sorted(set(reasons)),
    }


def _selected_race_key(row: Mapping[str, Any]) -> tuple[str, str]:
    race_url = str(row.get("race_url") or row.get("url") or "").strip()
    race_id = str(row.get("race_id") or "").strip()
    return race_url, race_id


def selected_races_by_key(report: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    selected = report.get("selected_races")
    if not isinstance(selected, list):
        return output
    for item in selected:
        if not isinstance(item, Mapping):
            continue
        row = dict(item)
        race_url, race_id = _selected_race_key(row)
        aliases = [str(value) for value in row.get("race_id_aliases") or [] if value]
        keys = [(race_url, race_id)]
        keys.extend((race_url, alias) for alias in aliases)
        keys.extend(("", alias) for alias in aliases)
        for key in keys:
            if key != ("", ""):
                output[key] = row
    return output


def refresh_report_for_input_dir(input_dir: Path) -> dict[str, Any]:
    candidates = [
        input_dir / "odds_capture_refresh_report.json",
        input_dir / "refresh_prejump_report.json",
        input_dir.parent / "odds_capture_refresh_report.json",
        input_dir.parent / "refresh_prejump_report.json",
    ]
    return next((load_json_object(path) for path in candidates if path.exists()), {})


def add_selected_race_aliases(
    plan_item: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    enriched = dict(plan_item)
    source_url = str(plan_item.get("thedogs_source_url") or "").strip()
    race_number = parse_int_value(plan_item.get("race_number"))
    race_date = str(plan_item.get("race_date") or "")
    selected = report.get("selected_races")
    if not source_url or not isinstance(selected, list):
        return enriched
    matches = [
        row
        for row in selected
        if isinstance(row, Mapping)
        and str(row.get("race_url") or row.get("url") or "").strip() == source_url
        and race_number_from_selected(row) == race_number
        and race_date_from_selected(row) == race_date
    ]
    if len(matches) != 1:
        return enriched
    aliases = {
        str(value)
        for value in [
            matches[0].get("race_id"),
            *(matches[0].get("race_id_aliases") or []),
        ]
        if isinstance(value, str) and value
    }
    if len(aliases) <= 16:
        enriched["race_id_aliases"] = sorted(aliases)
    return enriched


def race_number_from_selected(row: Mapping[str, Any]) -> int | None:
    return parse_int_value(row.get("race_number"))


def race_date_from_selected(row: Mapping[str, Any]) -> str | None:
    parsed = parse_date_value(row.get("date") or row.get("race_date"))
    return parsed.isoformat() if parsed else None


def long_race_id_from_selected(row: Mapping[str, Any]) -> str | None:
    race_number = race_number_from_selected(row)
    race_date = race_date_from_selected(row)
    aliases = [str(value) for value in row.get("race_id_aliases") or [] if value]
    if race_number is not None and race_date:
        suffix = f"Race {race_number} - "
        dated_suffix = f" - {race_date}"
        long_aliases = [
            alias
            for alias in aliases
            if alias.startswith(suffix) and alias.endswith(dated_suffix)
        ]
        if long_aliases:
            return sorted(long_aliases, key=len, reverse=True)[0]
    return str(row.get("race_id") or "").strip() or None


def expected_runners_from_participants(
    participants: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []
    for item in participants:
        if not isinstance(item, Mapping):
            continue
        box = parse_int_value(item.get("box_number") or item.get("box"))
        dog_name = str(item.get("dog_name") or item.get("name") or "").strip()
        identity = normalise_runner_name(dog_name)
        if box is None or not dog_name or not identity:
            continue
        expected.append(
            {
                "box_number": box,
                "dog_name": dog_name,
                "identity": identity,
            }
        )
    return expected


def canonical_expected_runners_from_download(
    download: Mapping[str, Any],
    *,
    source_url: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = download.get("result")
    if not isinstance(result, Mapping):
        return [], {
            "source": "canonical_thedogs_final_runner_set_fallback",
            "status": "SKIPPED",
            "reason": "download_result_missing",
        }
    normalization = result.get("normalization")
    if not isinstance(normalization, Mapping):
        return [], {
            "source": "canonical_thedogs_final_runner_set_fallback",
            "status": "SKIPPED",
            "reason": "normalization_missing",
        }
    alignment = normalization.get("canonical_runner_alignment")
    if not isinstance(alignment, Mapping):
        return [], {
            "source": "canonical_thedogs_final_runner_set_fallback",
            "status": "SKIPPED",
            "reason": "canonical_runner_alignment_missing",
        }
    if alignment.get("canonical_runner_set_status") != "available":
        return [], {
            "source": "canonical_thedogs_final_runner_set_fallback",
            "status": "SKIPPED",
            "reason": "canonical_runner_set_not_available",
            "canonical_runner_set_status": alignment.get("canonical_runner_set_status"),
        }
    canonical_url = str(
        alignment.get("canonical_source_url")
        or alignment.get("canonical_runner_source_url")
        or alignment.get("canonical_runner_set_source_url")
        or source_url
        or ""
    ).strip()
    if not canonical_url:
        return [], {
            "source": "canonical_thedogs_final_runner_set_fallback",
            "status": "SKIPPED",
            "reason": "canonical_source_url_missing",
        }
    if not is_thedogs_source_url(canonical_url):
        return [], {
            "source": "canonical_thedogs_final_runner_set_fallback",
            "status": "SKIPPED",
            "reason": "canonical_source_url_not_thedogs",
            "canonical_source_url": canonical_url,
        }
    if looks_post_result_source_url(canonical_url):
        return [], {
            "source": "canonical_thedogs_final_runner_set_fallback",
            "status": "SKIPPED",
            "reason": "canonical_source_url_looks_post_result",
            "canonical_source_url": canonical_url,
        }

    canonical = fetch_canonical_runner_set(canonical_url)
    participants = [
        row
        for row in canonical.get("final_runner_participants") or []
        if isinstance(row, Mapping)
    ]
    expected = expected_runners_from_participants(participants)
    status = str(canonical.get("canonical_runner_set_status") or "unavailable")
    return (
        expected if status == "available" else [],
        {
            "source": "canonical_thedogs_final_runner_set_fallback",
            "status": status,
            "canonical_source_url": canonical_url,
            "canonical_runner_count": len(participants),
            "expected_runner_count": len(expected),
            "reason": canonical.get("reason"),
            "ambiguous_reasons": list(canonical.get("ambiguous_reasons") or []),
        },
    )


def fallback_expected_runners_from_download(
    download: Mapping[str, Any],
    *,
    source_url: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    canonical_expected, canonical_report = canonical_expected_runners_from_download(
        download,
        source_url=source_url,
    )
    if canonical_expected:
        return canonical_expected, canonical_report

    result = download.get("result")
    if not isinstance(result, Mapping):
        return [], {
            "source": "downloaded_thedogs_form_csv_fallback",
            "status": "FAIL",
            "reason": "download_result_missing",
            "canonical_fallback": canonical_report,
        }
    completeness = result.get("runner_completeness")
    if not isinstance(completeness, Mapping):
        return [], {
            "source": "downloaded_thedogs_form_csv_fallback",
            "status": "FAIL",
            "reason": "runner_completeness_missing",
            "canonical_fallback": canonical_report,
        }
    if completeness.get("status") != "COMPLETE":
        return [], {
            "source": "downloaded_thedogs_form_csv_fallback",
            "status": "FAIL",
            "reason": "runner_completeness_not_complete",
            "runner_completeness_status": completeness.get("status"),
            "canonical_fallback": canonical_report,
        }
    participants = completeness.get("participants")
    if not isinstance(participants, list):
        return [], {
            "source": "downloaded_thedogs_form_csv_fallback",
            "status": "FAIL",
            "reason": "runner_completeness_participants_missing",
            "canonical_fallback": canonical_report,
        }
    expected = expected_runners_from_participants(
        [row for row in participants if isinstance(row, Mapping)]
    )
    return expected, {
        "source": "downloaded_thedogs_form_csv_fallback",
        "status": "PASS" if expected else "FAIL",
        "runner_count": len(expected),
        "canonical_fallback": canonical_report,
    }


def build_fallback_plan_item_from_download(
    *,
    selected: Mapping[str, Any],
    download: Mapping[str, Any],
    current_time: datetime,
) -> dict[str, Any]:
    reasons: list[str] = []
    race_number = race_number_from_selected(selected)
    race_date_text = race_date_from_selected(selected)
    venue_text = str(selected.get("venue") or "").strip().upper() or None
    race_id = long_race_id_from_selected(selected)
    source_url = str(selected.get("race_url") or selected.get("url") or "").strip() or None
    expected, expected_source_report = fallback_expected_runners_from_download(
        download,
        source_url=source_url,
    )
    if not expected:
        reasons.append("fallback_runner_rows_missing")
    jump_dt = parse_iso_datetime(selected.get("jump_datetime"))
    if jump_dt is None and race_date_text and selected.get("race_time"):
        race_date_value = parse_date_value(race_date_text)
        if race_date_value:
            jump_dt, jump_error = parse_jump_datetime(
                race_date=race_date_value,
                jump_time=selected.get("race_time"),
                current_time=current_time,
            )
            if jump_error:
                reasons.append(jump_error)

    if not race_id:
        reasons.append("race_id_missing")
    if not race_date_text:
        reasons.append("race_date_missing")
    if not venue_text:
        reasons.append("venue_missing")
    if race_number is None:
        reasons.append("race_number_missing")
    if not source_url:
        reasons.append("source_url_missing")
    elif not is_thedogs_source_url(source_url):
        reasons.append("source_url_not_thedogs")
    elif looks_post_result_source_url(source_url):
        reasons.append("source_url_looks_post_result")

    minutes_to_jump = None
    if jump_dt is not None:
        jump_cmp = jump_dt
        current_cmp = current_time
        if jump_cmp.tzinfo is None and current_cmp.tzinfo is not None:
            jump_cmp = jump_cmp.replace(tzinfo=current_cmp.tzinfo)
        if current_cmp.tzinfo is None and jump_cmp.tzinfo is not None:
            current_cmp = current_cmp.replace(tzinfo=jump_cmp.tzinfo)
        minutes_to_jump = (jump_cmp - current_cmp).total_seconds() / 60.0
    else:
        reasons.append("jump_datetime_missing")

    capture_window, window_status = due_capture_window(minutes_to_jump)
    if capture_window is None:
        reasons.append(window_status)

    result = download.get("result") if isinstance(download.get("result"), Mapping) else {}
    raw_export_path = result.get("raw_export_path") if isinstance(result, Mapping) else None
    status = "READY_TO_CAPTURE" if not reasons else "BLOCKED"
    if status == "BLOCKED" and reasons == [window_status]:
        status = "NO_DUE_WINDOW"
    return {
        "schema_version": "autonomous_live_odds_capture_plan_item_v1",
        "status": status,
        "csv_path": str(raw_export_path) if raw_export_path else None,
        "sidecar_path": None,
        "race_id": race_id,
        "race_id_aliases": list(selected.get("race_id_aliases") or []),
        "venue": venue_text,
        "race_number": race_number,
        "race_date": race_date_text,
        "race_time": race_time_from_datetime(jump_dt),
        "jump_datetime": jump_dt.isoformat() if jump_dt else None,
        "minutes_to_jump": minutes_to_jump,
        "capture_window_minutes": capture_window,
        "window_status": window_status,
        "thedogs_source_url": source_url,
        "runner_set_validation": {
            "status": "PASS" if expected else "FAIL",
            "runner_count": len(expected),
            "expected_runners": expected,
            **expected_source_report,
            "canonical_alignment_bypassed_for_odds_capture": True,
            "canonical_alignment_status": (
                result.get("normalization", {})
                .get("canonical_runner_alignment", {})
                .get("status")
                if isinstance(result.get("normalization"), Mapping)
                else None
            ),
        },
        "expected_runners": expected,
        "blockers": sorted(set(reasons)),
        "odds_capture_expected_runner_source": expected_source_report.get("source"),
    }


def fallback_plan_items_from_refresh_report(
    input_dir: Path,
    *,
    current_time: datetime,
) -> list[dict[str, Any]]:
    report = refresh_report_for_input_dir(input_dir)
    if not report:
        return []
    selected_by_key = selected_races_by_key(report)
    downloads = report.get("downloads")
    if not isinstance(downloads, list):
        return []
    rows: list[dict[str, Any]] = []
    for download in downloads:
        if not isinstance(download, Mapping):
            continue
        race_url = str(download.get("race_url") or "").strip()
        result = download.get("result")
        if not isinstance(result, Mapping):
            continue
        selected = selected_by_key.get((race_url, "")) or next(
            (
                row
                for (key_url, _), row in selected_by_key.items()
                if race_url and key_url == race_url
            ),
            None,
        )
        if not isinstance(selected, Mapping):
            continue
        if not result.get("raw_export_path"):
            continue
        rows.append(
            build_fallback_plan_item_from_download(
                selected=selected,
                download=download,
                current_time=current_time,
            )
        )
    return rows


def build_capture_plan(
    input_dirs: Sequence[Path],
    *,
    current_time: datetime,
    limit: int | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen_csv_paths: set[Path] = set()
    for input_dir in input_dirs:
        refresh_report = refresh_report_for_input_dir(Path(input_dir))
        input_dir_row_start = len(rows)
        for csv_path in sorted(Path(input_dir).rglob("*.csv")):
            if {"raw_exports", "quarantine"}.intersection(csv_path.parts):
                continue
            logical_path = csv_path.resolve()
            if logical_path in seen_csv_paths:
                continue
            seen_csv_paths.add(logical_path)
            rows.append(
                add_selected_race_aliases(
                    build_plan_item(csv_path, current_time),
                    refresh_report,
                )
            )
        input_dir_rows = rows[input_dir_row_start:]
        if not any(row.get("status") == "READY_TO_CAPTURE" for row in input_dir_rows):
            rows.extend(
                fallback_plan_items_from_refresh_report(
                    Path(input_dir),
                    current_time=current_time,
                )
            )
    rows = sorted(rows, key=capture_plan_priority_key)
    if limit is not None:
        rows = rows[:limit]
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "UNKNOWN")
        counts[status] = counts.get(status, 0) + 1
    return {
        "schema_version": "autonomous_live_odds_capture_plan_v1",
        "generated_at": current_time.isoformat(),
        "input_dirs": [relpath(Path(item)) for item in input_dirs],
        "capture_window_offsets_minutes": list(CAPTURE_WINDOWS_MINUTES),
        "accepted_sportsbet_box_sources": sorted(ACCEPTED_SPORTSBET_BOX_SOURCES),
        "limit": limit,
        "status_counts": counts,
        "ready_count": counts.get("READY_TO_CAPTURE", 0),
        "races": rows,
        "no_write_guarantees": {
            **NO_WRITE_GUARANTEES,
            "db_write": False,
        },
    }


def capture_plan_priority_key(row: Mapping[str, Any]) -> tuple[int, float, str]:
    minutes_to_jump = row.get("minutes_to_jump")
    minutes = (
        float(minutes_to_jump)
        if isinstance(minutes_to_jump, (int, float))
        else float("inf")
    )
    return (
        0 if row.get("status") == "READY_TO_CAPTURE" else 1,
        minutes,
        str(row.get("race_id") or ""),
    )


def plan_item_csv_path(plan_item: Mapping[str, Any]) -> Path | None:
    value = plan_item.get("csv_path")
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def plan_item_should_rebuild_from_csv(plan_item: Mapping[str, Any], csv_path: Path | None) -> bool:
    if csv_path is None or not csv_path.exists():
        return False
    if plan_item.get("odds_capture_expected_runner_source") == (
        "downloaded_thedogs_form_csv_fallback"
    ):
        return False
    return plan_item.get("sidecar_path") not in (None, "")


def refresh_plan_item_for_time(
    plan_item: Mapping[str, Any],
    current_time: datetime,
) -> dict[str, Any]:
    csv_path = plan_item_csv_path(plan_item)
    if plan_item_should_rebuild_from_csv(plan_item, csv_path):
        rebuilt = build_plan_item(csv_path, current_time)
        aliases = plan_item.get("race_id_aliases")
        if isinstance(aliases, list):
            rebuilt["race_id_aliases"] = list(aliases)
        return rebuilt

    refreshed = dict(plan_item)
    time_blockers = {
        "before_first_capture_window",
        "capture_window_unavailable",
        "jump_datetime_missing",
        "jump_datetime_unparseable",
        "jump_time_missing",
        "race_already_jumped",
    }
    reasons = [
        str(reason)
        for reason in refreshed.get("blockers") or []
        if str(reason) not in time_blockers
    ]
    jump_text = refreshed.get("jump_datetime")
    jump_dt = None
    if jump_text:
        try:
            jump_dt = datetime.fromisoformat(str(jump_text))
        except ValueError:
            reasons.append("jump_datetime_unparseable")
    else:
        reasons.append("jump_datetime_missing")

    minutes_to_jump = None
    if jump_dt is not None:
        jump_cmp = jump_dt
        current_cmp = current_time
        if jump_cmp.tzinfo is None and current_cmp.tzinfo is not None:
            jump_cmp = jump_cmp.replace(tzinfo=current_cmp.tzinfo)
        if current_cmp.tzinfo is None and jump_cmp.tzinfo is not None:
            current_cmp = current_cmp.replace(tzinfo=jump_cmp.tzinfo)
        minutes_to_jump = (jump_cmp - current_cmp).total_seconds() / 60.0
    capture_window, window_status = due_capture_window(minutes_to_jump)
    if capture_window is None:
        reasons.append(window_status)

    refreshed.update(
        {
            "status": "READY_TO_CAPTURE" if not reasons else "BLOCKED",
            "minutes_to_jump": minutes_to_jump,
            "capture_window_minutes": capture_window,
            "window_status": window_status,
            "blockers": sorted(set(reasons)),
        }
    )
    return refreshed


def is_sportsbet_source_url(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        parsed = urlparse(text)
    except Exception:
        return False
    host = parsed.netloc.lower().split("@")[-1].split(":")[0]
    return parsed.scheme in {"http", "https"} and (
        host == "sportsbet.com.au" or host.endswith(".sportsbet.com.au")
    )


def sportsbet_source_url_is_post_race(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    try:
        parsed = urlparse(text)
        searchable = " ".join(
            part for part in (parsed.path, parsed.query, parsed.fragment) if part
        )
    except Exception:
        searchable = text
    tokens = {token for token in re.split(r"[^a-z0-9]+", searchable) if token}
    return bool(tokens.intersection(POST_RACE_URL_TOKENS))


def fetched_source_url(fetch_result: Mapping[str, Any]) -> str | None:
    race_info = fetch_result.get("race_info")
    if not isinstance(race_info, Mapping):
        race_info = {}
    value = (
        race_info.get("venue_url")
        or race_info.get("sportsbet_url")
        or race_info.get("url")
        or fetch_result.get("source_url")
    )
    return str(value).strip() if value not in (None, "") else None


def fetched_market_odds_rows(
    fetch_result: Mapping[str, Any],
    market_type: str,
) -> list[dict[str, Any]]:
    market = str(market_type or "win").strip().lower()
    key = "odds_data_place" if market == "place" else "odds_data"
    rows = fetch_result.get(key)
    if not isinstance(rows, list):
        race_info = fetch_result.get("race_info")
        if isinstance(race_info, Mapping):
            rows = race_info.get(key)
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def fetched_odds_rows(fetch_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    return fetched_market_odds_rows(fetch_result, "win")


def fetched_place_odds_rows(fetch_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    return fetched_market_odds_rows(fetch_result, "place")


def normalize_fetched_row(row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    dog_name = str(row.get("dog_name") or row.get("dog_clean_name") or "").strip()
    identity = normalise_runner_name(dog_name)
    box = parse_int_value(row.get("box_number") or row.get("box"))
    box_source = str(row.get("sportsbet_box_source") or "").strip()
    raw_runner_text = row.get("sportsbet_raw_runner_text")
    if box_source == "runner_text":
        parsed_text_box = parse_sportsbet_runner_box_from_text(raw_runner_text)
        if parsed_text_box is not None:
            box = parsed_text_box
    try:
        odds = float(row.get("odds_decimal"))
    except Exception:
        odds = 0.0
    if box_source not in ACCEPTED_SPORTSBET_BOX_SOURCES:
        return None, f"unsupported_sportsbet_box_source:{box_source or 'missing'}"
    if box is None:
        return None, "sportsbet_box_number_missing"
    if not identity:
        return None, "sportsbet_dog_name_missing"
    if odds <= 1.0:
        return None, "sportsbet_odds_decimal_invalid"
    return {
        "dog_name": dog_name,
        "dog_clean_name": row.get("dog_clean_name") or dog_name,
        "box_number": box,
        "identity": identity,
        "odds_decimal": odds,
        "odds_fractional": row.get("odds_fractional", ""),
        "sportsbet_box_source": box_source,
        "sportsbet_list_position": row.get("sportsbet_list_position"),
        "sportsbet_raw_runner_text": raw_runner_text,
    }, None


def _market_reason_name(market_type: str, reason: str) -> str:
    market = str(market_type or "win").strip().lower()
    win_names = {
        "duplicate": "sportsbet_duplicate_runner_keys",
        "scratched": "sportsbet_odds_present_for_scratched_expected_runners",
        "missing": "sportsbet_missing_expected_runners",
        "extra": "sportsbet_extra_unexpected_runners",
        "rejected": "sportsbet_rejected_runner_rows",
        "zero": "sportsbet_accepted_runner_rows_zero",
    }
    if market == "win":
        return win_names[reason]
    place_names = {
        "duplicate": f"sportsbet_{market}_duplicate_runner_keys",
        "scratched": f"sportsbet_{market}_odds_present_for_scratched_expected_runners",
        "missing": f"sportsbet_{market}_missing_expected_runners",
        "extra": f"sportsbet_{market}_extra_unexpected_runners",
        "rejected": f"sportsbet_{market}_rejected_runner_rows",
        "zero": f"sportsbet_{market}_accepted_runner_rows_zero",
    }
    return place_names[reason]


def validate_fetched_market_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    market_type: str,
    expected_set: set[tuple[int, str]],
    active_expected_set: set[tuple[int, str]],
    scratched_expected_set: set[tuple[int, str]],
) -> dict[str, Any]:
    reasons: list[str] = []
    rejected_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    for raw_row in rows:
        normalized, reason = normalize_fetched_row(raw_row)
        if normalized is None:
            rejected_rows.append({"row": dict(raw_row), "reason": reason})
            continue
        accepted_rows.append(normalized)

    accepted_set = {(int(row["box_number"]), row["identity"]) for row in accepted_rows}
    duplicate_keys = sorted(
        key
        for key in accepted_set
        if sum(
            1
            for row in accepted_rows
            if (int(row["box_number"]), row["identity"]) == key
        )
        > 1
    )
    scratched_with_odds = sorted(scratched_expected_set & accepted_set)
    missing = sorted(active_expected_set - accepted_set)
    extra = sorted(accepted_set - expected_set)
    if duplicate_keys:
        reasons.append(
            _market_reason_name(market_type, "duplicate")
            + ":"
            + ",".join(f"{box}:{identity}" for box, identity in duplicate_keys)
        )
    if scratched_with_odds:
        reasons.append(
            _market_reason_name(market_type, "scratched")
            + ":"
            + ",".join(f"{box}:{identity}" for box, identity in scratched_with_odds)
        )
    if missing:
        reasons.append(
            _market_reason_name(market_type, "missing")
            + ":"
            + ",".join(f"{box}:{identity}" for box, identity in missing)
        )
    if extra:
        reasons.append(
            _market_reason_name(market_type, "extra")
            + ":"
            + ",".join(f"{box}:{identity}" for box, identity in extra)
        )
    if rejected_rows:
        reasons.append(
            f"{_market_reason_name(market_type, 'rejected')}:{len(rejected_rows)}"
        )
    if not accepted_rows:
        reasons.append(_market_reason_name(market_type, "zero"))

    return {
        "market_type": str(market_type or "win").strip().lower(),
        "accepted_rows": accepted_rows,
        "accepted_row_count": len(accepted_rows),
        "rejected_rows": rejected_rows,
        "missing_expected_runners": [
            {"box_number": box, "identity": identity} for box, identity in missing
        ],
        "extra_unexpected_runners": [
            {"box_number": box, "identity": identity} for box, identity in extra
        ],
        "scratched_expected_runners_with_odds": [
            {"box_number": box, "identity": identity}
            for box, identity in scratched_with_odds
        ],
        "duplicate_runner_keys": [
            {"box_number": box, "identity": identity}
            for box, identity in duplicate_keys
        ],
        "reasons": sorted(set(reasons)),
    }


def validate_fetched_odds(
    plan_item: Mapping[str, Any],
    fetch_result: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if fetch_result.get("success") is not True:
        reasons.append("sportsbet_fetch_not_successful")
    source_url = fetched_source_url(fetch_result)
    if not source_url:
        reasons.append("sportsbet_source_url_missing")
    elif not is_sportsbet_source_url(source_url):
        reasons.append("sportsbet_source_url_not_sportsbet")
    elif sportsbet_source_url_is_post_race(source_url):
        reasons.append("sportsbet_source_url_looks_post_result")

    race_info = fetch_result.get("race_info")
    if isinstance(race_info, Mapping):
        fetched_race_number = parse_int_value(race_info.get("race_number"))
        if fetched_race_number is not None and fetched_race_number != plan_item.get("race_number"):
            reasons.append("sportsbet_race_number_mismatch")

    expected_rows = [
        dict(row)
        for row in plan_item.get("expected_runners") or []
        if isinstance(row, Mapping)
    ]
    expected_sets = expected_runner_key_sets(expected_rows)
    expected_set = expected_sets["all"]
    active_expected_set = expected_sets["active"]
    scratched_expected_set = expected_sets["scratched"]
    win_validation = validate_fetched_market_rows(
        fetched_odds_rows(fetch_result),
        market_type="win",
        expected_set=expected_set,
        active_expected_set=active_expected_set,
        scratched_expected_set=scratched_expected_set,
    )
    place_validation = validate_fetched_market_rows(
        fetched_place_odds_rows(fetch_result),
        market_type="place",
        expected_set=expected_set,
        active_expected_set=active_expected_set,
        scratched_expected_set=scratched_expected_set,
    )
    reasons.extend(win_validation["reasons"])
    reasons.extend(place_validation["reasons"])
    failure_detail = validation_failure_detail(
        active_expected_count=len(active_expected_set),
        accepted_row_count=int(win_validation["accepted_row_count"]),
        missing_count=len(win_validation["missing_expected_runners"]),
        extra_count=len(win_validation["extra_unexpected_runners"]),
        accepted_place_row_count=int(place_validation["accepted_row_count"]),
        place_missing_count=len(place_validation["missing_expected_runners"]),
        place_extra_count=len(place_validation["extra_unexpected_runners"]),
        fetch_result=fetch_result,
    )
    failure_reason = validation_failure_reason(failure_detail)
    if failure_reason:
        reasons.append(failure_reason)
    return {
        "schema_version": "autonomous_live_odds_capture_validation_v1",
        "status": "PASS" if not reasons else "FAIL",
        "source_url": source_url,
        "accepted_rows": win_validation["accepted_rows"],
        "accepted_row_count": win_validation["accepted_row_count"],
        "rejected_rows": win_validation["rejected_rows"],
        "accepted_place_rows": place_validation["accepted_rows"],
        "accepted_place_row_count": place_validation["accepted_row_count"],
        "rejected_place_rows": place_validation["rejected_rows"],
        "market_validations": {
            "win": win_validation,
            "place": place_validation,
        },
        "expected_runner_count": len(expected_set),
        "active_expected_runner_count": len(active_expected_set),
        "scratched_expected_runner_count": len(scratched_expected_set),
        "scratched_expected_runners": [
            {"box_number": box, "identity": identity}
            for box, identity in sorted(scratched_expected_set)
        ],
        "scratched_expected_runners_with_odds": [
            dict(row)
            for row in win_validation["scratched_expected_runners_with_odds"]
        ],
        "missing_expected_runners": [
            dict(row) for row in win_validation["missing_expected_runners"]
        ],
        "extra_unexpected_runners": [
            dict(row) for row in win_validation["extra_unexpected_runners"]
        ],
        "place_missing_expected_runners": [
            dict(row) for row in place_validation["missing_expected_runners"]
        ],
        "place_extra_unexpected_runners": [
            dict(row) for row in place_validation["extra_unexpected_runners"]
        ],
        "failure_root_cause": failure_detail.get("root_cause"),
        "failure_detail": failure_detail,
        "reasons": sorted(set(reasons)),
    }


def explicit_expected_runner_scratched(row: Mapping[str, Any]) -> bool:
    for field in SCRATCHED_EXPECTED_RUNNER_BOOL_FIELDS:
        value = row.get(field)
        if value is True:
            return True
        if isinstance(value, (int, float)) and value == 1:
            return True
        if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "y"}:
            return True
    for field in SCRATCHED_EXPECTED_RUNNER_STATUS_FIELDS:
        status = str(row.get(field) or "").strip().upper()
        compact_status = re.sub(r"[\s_/-]+", "", status)
        if (
            status in SCRATCHED_EXPECTED_RUNNER_STATUSES
            or compact_status in SCRATCHED_EXPECTED_RUNNER_COMPACT_STATUSES
        ):
            return True
    return False


def expected_runner_key(row: Mapping[str, Any]) -> tuple[int, str] | None:
    box = parse_int_value(row.get("box_number"))
    identity = normalise_runner_name(row.get("dog_name") or row.get("identity"))
    if box is None or not identity:
        return None
    return (box, identity)


def expected_runner_key_sets(
    expected_runners: Sequence[Mapping[str, Any]],
) -> dict[str, set[tuple[int, str]]]:
    all_keys: set[tuple[int, str]] = set()
    active_keys: set[tuple[int, str]] = set()
    scratched_keys: set[tuple[int, str]] = set()
    for row in expected_runners:
        key = expected_runner_key(row)
        if key is None:
            continue
        all_keys.add(key)
        if explicit_expected_runner_scratched(row):
            scratched_keys.add(key)
        else:
            active_keys.add(key)
    return {"all": all_keys, "active": active_keys, "scratched": scratched_keys}


def expected_runner_key_set(expected_runners: Sequence[Mapping[str, Any]]) -> set[tuple[int, str]]:
    return expected_runner_key_sets(expected_runners)["active"]


def parse_iso_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def capture_window_bounds(
    *,
    jump_datetime: datetime | None,
    capture_window_minutes: int | None,
    tolerance_seconds: int = 180,
) -> tuple[datetime | None, datetime | None]:
    if jump_datetime is None or capture_window_minutes is None:
        return None, None
    target_at = jump_datetime - timedelta(minutes=capture_window_minutes)
    smaller_offsets = [
        offset for offset in CAPTURE_WINDOWS_MINUTES if offset < capture_window_minutes
    ]
    closes_at = (
        jump_datetime - timedelta(minutes=max(smaller_offsets))
        if smaller_offsets
        else jump_datetime
    )
    return target_at - timedelta(seconds=tolerance_seconds), closes_at


def capture_timestamp_in_window(
    value: Any,
    *,
    jump_datetime: datetime | None,
    capture_window_minutes: int | None,
) -> tuple[bool, str | None]:
    lower_bound, upper_bound = capture_window_bounds(
        jump_datetime=jump_datetime,
        capture_window_minutes=capture_window_minutes,
    )
    if lower_bound is None or upper_bound is None:
        return True, None
    captured_at = parse_iso_datetime(value)
    if captured_at is None:
        return False, "existing_capture_timestamp_missing_or_unparseable"
    if captured_at.tzinfo is None and lower_bound.tzinfo is not None:
        captured_at = captured_at.replace(tzinfo=lower_bound.tzinfo)
    if lower_bound.tzinfo is None and captured_at.tzinfo is not None:
        lower_bound = lower_bound.replace(tzinfo=captured_at.tzinfo)
    if upper_bound.tzinfo is None and captured_at.tzinfo is not None:
        upper_bound = upper_bound.replace(tzinfo=captured_at.tzinfo)
    if captured_at < lower_bound:
        return False, "existing_capture_before_fixed_window"
    if captured_at >= upper_bound:
        return False, "existing_capture_after_fixed_window"
    return True, None


def runner_group_status(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_set: set[tuple[int, str]],
    allowed_expected_set: set[tuple[int, str]] | None = None,
    expected_market_type: str = "win",
) -> dict[str, Any]:
    expected_market = str(expected_market_type or "win").strip().lower()
    observed_keys: list[tuple[int, str]] = []
    invalid_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        box = parse_int_value(row.get("box_number"))
        identity = normalise_runner_name(row.get("dog_name") or row.get("dog_clean_name"))
        reasons: list[str] = []
        if box is None:
            reasons.append("box_number_missing")
        if not identity:
            reasons.append("dog_name_missing")
        if "odds_decimal" in row:
            try:
                if float(row.get("odds_decimal") or 0) <= 1.0:
                    reasons.append("odds_decimal_invalid")
            except Exception:
                reasons.append("odds_decimal_invalid")
        if "source_url" in row:
            source_url = row.get("source_url")
            if not str(source_url or "").strip():
                reasons.append("source_url_missing")
            elif not is_sportsbet_source_url(source_url):
                reasons.append("source_url_not_sportsbet")
            elif sportsbet_source_url_is_post_race(source_url):
                reasons.append("source_url_post_race")
        if "capture_timestamp" in row and not str(row.get("capture_timestamp") or "").strip():
            reasons.append("capture_timestamp_missing")
        if (
            "market_type" in row
            and str(row.get("market_type") or "").strip().lower() != expected_market
        ):
            reasons.append(f"market_not_{expected_market}")
        if "source" in row and str(row.get("source") or "").strip().lower() != "sportsbet":
            reasons.append("source_not_sportsbet")
        if "odds_level" in row:
            odds_level = str(row.get("odds_level") or "").strip().lower()
            if not odds_level:
                reasons.append("odds_level_missing")
            elif odds_level not in ACCEPTED_DOG_LEVEL_ODDS_LEVELS:
                reasons.append("odds_level_not_dog")
        if "sportsbet_box_source" in row:
            box_source = str(row.get("sportsbet_box_source") or "").strip()
            if box_source not in ACCEPTED_SPORTSBET_BOX_SOURCES:
                reasons.append(f"unsupported_sportsbet_box_source:{box_source or 'missing'}")
        if reasons:
            invalid_rows.append(
                {
                    "row_index": index,
                    "box_number": box,
                    "identity": identity,
                    "reasons": sorted(set(reasons)),
                }
            )
            continue
        observed_keys.append((int(box), identity))

    observed_set = set(observed_keys)
    if allowed_expected_set is None:
        allowed_expected_set = expected_set
    inactive_expected_with_odds = sorted(observed_set & (allowed_expected_set - expected_set))
    duplicate_keys = sorted(
        key for key in observed_set if sum(1 for value in observed_keys if value == key) > 1
    )
    missing = sorted(expected_set - observed_set)
    extra = sorted(observed_set - allowed_expected_set)
    reasons: list[str] = []
    if inactive_expected_with_odds:
        reasons.append(
            "existing_capture_odds_present_for_inactive_expected_runners:"
            + ",".join(f"{box}:{identity}" for box, identity in inactive_expected_with_odds)
        )
    if missing:
        reasons.append(
            "existing_capture_missing_expected_runners:"
            + ",".join(f"{box}:{identity}" for box, identity in missing)
        )
    if extra:
        reasons.append(
            "existing_capture_extra_unexpected_runners:"
            + ",".join(f"{box}:{identity}" for box, identity in extra)
        )
    if duplicate_keys:
        reasons.append(
            "existing_capture_duplicate_runner_keys:"
            + ",".join(f"{box}:{identity}" for box, identity in duplicate_keys)
        )
    if invalid_rows:
        reasons.append(f"existing_capture_invalid_rows:{len(invalid_rows)}")

    if not reasons:
        existing_status = "COMPLETE"
    elif missing and not extra and not duplicate_keys and not invalid_rows:
        existing_status = "INCOMPLETE"
    else:
        existing_status = "INVALID"
    return {
        "status": existing_status,
        "market_type": expected_market,
        "observed_runner_count": len(observed_set),
        "missing_expected_runners": [
            {"box_number": box, "identity": identity} for box, identity in missing
        ],
        "inactive_expected_runners_with_odds": [
            {"box_number": box, "identity": identity}
            for box, identity in inactive_expected_with_odds
        ],
        "extra_unexpected_runners": [
            {"box_number": box, "identity": identity} for box, identity in extra
        ],
        "duplicate_runner_keys": [
            {"box_number": box, "identity": identity} for box, identity in duplicate_keys
        ],
        "invalid_rows": invalid_rows,
        "reasons": sorted(set(reasons)),
    }


def runner_multi_market_group_status(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_set: set[tuple[int, str]],
    allowed_expected_set: set[tuple[int, str]] | None = None,
) -> dict[str, Any]:
    rows_by_market: dict[str, list[Mapping[str, Any]]] = {
        market: [] for market in REQUIRED_CAPTURE_MARKETS
    }
    for row in rows:
        market = str(row.get("market_type") or "win").strip().lower()
        if market in rows_by_market:
            rows_by_market[market].append(row)

    market_statuses = {
        market: runner_group_status(
            rows_by_market[market],
            expected_set=expected_set,
            allowed_expected_set=allowed_expected_set,
            expected_market_type=market,
        )
        for market in REQUIRED_CAPTURE_MARKETS
    }
    missing_required_markets = [
        market for market, market_rows in rows_by_market.items() if not market_rows
    ]
    reasons: list[str] = []
    if missing_required_markets:
        reasons.append(
            "existing_capture_missing_required_markets:"
            + ",".join(missing_required_markets)
        )
    for market, market_status in market_statuses.items():
        for reason in market_status.get("reasons") or []:
            reasons.append(f"{market}:{reason}")

    complete = all(
        market_status.get("status") == "COMPLETE"
        for market_status in market_statuses.values()
    )
    invalid = any(
        market_status.get("status") == "INVALID"
        for market_status in market_statuses.values()
    )
    if complete and not missing_required_markets:
        existing_status = "COMPLETE"
    elif invalid:
        existing_status = "INVALID"
    else:
        existing_status = "INCOMPLETE"

    selected_market = next(
        (
            market_status
            for market_status in market_statuses.values()
            if market_status.get("status") != "COMPLETE"
        ),
        market_statuses[REQUIRED_CAPTURE_MARKETS[0]],
    )
    return {
        "status": existing_status,
        "required_markets": list(REQUIRED_CAPTURE_MARKETS),
        "observed_markets": [
            market for market, market_rows in rows_by_market.items() if market_rows
        ],
        "missing_required_markets": missing_required_markets,
        "market_statuses": market_statuses,
        "observed_runner_count": min(
            int(market_status.get("observed_runner_count") or 0)
            for market_status in market_statuses.values()
        ),
        "missing_expected_runners": list(
            selected_market.get("missing_expected_runners") or []
        ),
        "inactive_expected_runners_with_odds": list(
            selected_market.get("inactive_expected_runners_with_odds") or []
        ),
        "extra_unexpected_runners": list(
            selected_market.get("extra_unexpected_runners") or []
        ),
        "duplicate_runner_keys": list(
            selected_market.get("duplicate_runner_keys") or []
        ),
        "invalid_rows": [
            {**dict(row), "market_type": market}
            for market, market_status in market_statuses.items()
            for row in (market_status.get("invalid_rows") or [])
        ],
        "reasons": sorted(set(reasons)),
    }


def existing_capture_runner_status(
    db_path: Path,
    *,
    race_id: str,
    capture_mode: str,
    expected_runners: Sequence[Mapping[str, Any]],
    jump_datetime: datetime | None = None,
    capture_window_minutes: int | None = None,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "schema_version": "autonomous_live_odds_existing_capture_status_v1",
        "status": "NONE",
        "race_id": race_id,
        "capture_mode": capture_mode,
        "existing_row_count": 0,
        "expected_runner_count": 0,
        "observed_runner_count": 0,
        "active_expected_runner_count": 0,
        "scratched_expected_runner_count": 0,
        "scratched_expected_runners": [],
        "scratched_expected_runners_with_odds": [],
        "missing_expected_runners": [],
        "extra_unexpected_runners": [],
        "duplicate_runner_keys": [],
        "invalid_rows": [],
        "stale_capture_groups": [],
        "required_markets": list(REQUIRED_CAPTURE_MARKETS),
        "missing_required_markets": [],
        "market_statuses": {},
        "selected_capture_timestamp": None,
        "reasons": [],
    }
    if not db_path.exists():
        return status

    expected_sets = expected_runner_key_sets(expected_runners)
    expected_set = expected_sets["active"]
    allowed_expected_set = expected_sets["all"]
    scratched_expected_set = expected_sets["scratched"]
    status["expected_runner_count"] = len(allowed_expected_set)
    status["active_expected_runner_count"] = len(expected_set)
    status["scratched_expected_runner_count"] = len(scratched_expected_set)
    status["scratched_expected_runners"] = [
        {"box_number": box, "identity": identity}
        for box, identity in sorted(scratched_expected_set)
    ]
    if not allowed_expected_set:
        status["status"] = "INVALID"
        status["reasons"] = ["expected_runner_set_missing"]
        return status

    try:
        with sqlite3.connect(db_path) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(live_odds)")}
            if "race_id" not in columns or "capture_mode" not in columns:
                return status
            count_row = conn.execute(
                "SELECT COUNT(*) FROM live_odds WHERE race_id = ? AND capture_mode = ?",
                (race_id, capture_mode),
            ).fetchone()
            existing_count = int(count_row[0] or 0) if count_row else 0
            status["existing_row_count"] = existing_count
            if existing_count == 0:
                return status

            required_identity_columns = {"box_number", "dog_name", "dog_clean_name"}
            missing_identity_columns = sorted(required_identity_columns - columns)
            if "box_number" in missing_identity_columns or (
                "dog_name" in missing_identity_columns
                and "dog_clean_name" in missing_identity_columns
            ):
                status["status"] = "INVALID"
                status["reasons"] = [
                    "existing_capture_identity_columns_missing:"
                    + ",".join(missing_identity_columns)
                ]
                return status
            missing_provenance_columns = sorted(
                REQUIRED_EXISTING_CAPTURE_PROVENANCE_COLUMNS - columns
            )
            if missing_provenance_columns:
                status["status"] = "INVALID"
                status["reasons"] = [
                    "existing_capture_provenance_columns_missing:"
                    + ",".join(missing_provenance_columns)
                ]
                return status

            select_columns = [
                column
                for column in (
                    "box_number",
                    "dog_name",
                    "dog_clean_name",
                    "odds_decimal",
                    "source_url",
                    "capture_timestamp",
                    "market_type",
                    "source",
                    "odds_level",
                    "sportsbet_box_source",
                )
                if column in columns
            ]
            rows = conn.execute(
                f"SELECT {', '.join(select_columns)} "
                "FROM live_odds WHERE race_id = ? AND capture_mode = ?",
                (race_id, capture_mode),
            ).fetchall()
    except sqlite3.Error as exc:
        status["status"] = "INVALID"
        status["reasons"] = [f"existing_capture_read_failed:{type(exc).__name__}"]
        return status

    row_dicts = [dict(zip(select_columns, raw_row)) for raw_row in rows]
    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in row_dicts:
        grouped_rows.setdefault(str(row.get("capture_timestamp") or ""), []).append(row)
    group_reports: list[dict[str, Any]] = []
    stale_groups: list[dict[str, Any]] = []
    for capture_timestamp, group_rows in sorted(grouped_rows.items()):
        group = runner_multi_market_group_status(
            group_rows,
            expected_set=expected_set,
            allowed_expected_set=allowed_expected_set,
        )
        group["capture_timestamp"] = capture_timestamp or None
        group["existing_row_count"] = len(group_rows)
        in_window, temporal_reason = capture_timestamp_in_window(
            capture_timestamp,
            jump_datetime=jump_datetime,
            capture_window_minutes=capture_window_minutes,
        )
        if group["status"] == "COMPLETE" and not in_window:
            group["status"] = "STALE"
            group["reasons"] = sorted(set(group.get("reasons") or []) | {temporal_reason})
            stale_groups.append(group)
        group_reports.append(group)

    complete_groups = [group for group in group_reports if group.get("status") == "COMPLETE"]
    if complete_groups:
        selected = sorted(
            complete_groups, key=lambda group: str(group.get("capture_timestamp") or "")
        )[-1]
        existing_status = "COMPLETE"
        missing = []
        inactive_with_odds = []
        extra = []
        duplicate_keys = []
        invalid_rows = []
        reasons = []
        selected_capture_timestamp = selected.get("capture_timestamp")
        observed_runner_count = int(selected.get("observed_runner_count") or 0)
    else:
        actionable_groups = [group for group in group_reports if group.get("status") != "STALE"]
        if stale_groups and not actionable_groups:
            existing_status = "STALE"
            selected_capture_timestamp = None
            observed_runner_count = max(
                int(group.get("observed_runner_count") or 0) for group in stale_groups
            )
            missing = []
            inactive_with_odds = []
            extra = []
            duplicate_keys = []
            invalid_rows = []
            reasons = sorted(
                {
                    reason
                    for group in stale_groups
                    for reason in (group.get("reasons") or [])
                    if reason
                }
            )
        else:
            selected = sorted(
                actionable_groups or group_reports,
                key=lambda group: str(group.get("capture_timestamp") or ""),
            )[-1]
            existing_status = str(selected.get("status") or "INVALID")
            selected_capture_timestamp = selected.get("capture_timestamp")
            observed_runner_count = int(selected.get("observed_runner_count") or 0)
            missing = [
                (int(row["box_number"]), str(row["identity"]))
                for row in selected.get("missing_expected_runners") or []
            ]
            inactive_with_odds = [
                (int(row["box_number"]), str(row["identity"]))
                for row in selected.get("inactive_expected_runners_with_odds") or []
            ]
            extra = [
                (int(row["box_number"]), str(row["identity"]))
                for row in selected.get("extra_unexpected_runners") or []
            ]
            duplicate_keys = [
                (int(row["box_number"]), str(row["identity"]))
                for row in selected.get("duplicate_runner_keys") or []
            ]
            invalid_rows = list(selected.get("invalid_rows") or [])
            missing_required_markets = list(
                selected.get("missing_required_markets") or []
            )
            market_statuses = dict(selected.get("market_statuses") or {})
            reasons = list(selected.get("reasons") or [])
    if complete_groups:
        missing_required_markets = []
        market_statuses = dict(selected.get("market_statuses") or {})
    elif stale_groups and not actionable_groups:
        missing_required_markets = []
        market_statuses = {}
    status.update(
        {
            "status": existing_status,
            "observed_runner_count": observed_runner_count,
            "missing_expected_runners": [
                {"box_number": box, "identity": identity} for box, identity in missing
            ],
            "scratched_expected_runners_with_odds": [
                {"box_number": box, "identity": identity}
                for box, identity in inactive_with_odds
            ],
            "extra_unexpected_runners": [
                {"box_number": box, "identity": identity} for box, identity in extra
            ],
            "duplicate_runner_keys": [
                {"box_number": box, "identity": identity} for box, identity in duplicate_keys
            ],
            "invalid_rows": invalid_rows,
            "stale_capture_groups": stale_groups,
            "missing_required_markets": missing_required_markets,
            "market_statuses": market_statuses,
            "selected_capture_timestamp": selected_capture_timestamp,
            "reasons": sorted(set(reasons)),
        }
    )
    return status


def block_or_skip_existing_capture_attempt(
    attempt: dict[str, Any],
    existing_status: Mapping[str, Any],
) -> bool:
    status = existing_status.get("status")
    existing_count = int(existing_status.get("existing_row_count") or 0)
    if status == "NONE" or existing_count == 0:
        return False
    attempt["existing_capture"] = existing_status
    attempt["existing_capture_count"] = existing_count
    if status == "COMPLETE":
        attempt["status"] = "SKIPPED_ALREADY_CAPTURED"
        return True
    if status == "STALE":
        attempt["stale_existing_capture"] = existing_status
        return False
    if status == "INCOMPLETE":
        attempt["incomplete_existing_capture"] = existing_status
        return False
    if (
        status == "INVALID"
        and existing_status.get("extra_unexpected_runners")
        and not existing_status.get("missing_expected_runners")
        and not existing_status.get("duplicate_runner_keys")
        and not existing_status.get("invalid_rows")
    ):
        attempt["status"] = "SKIPPED_EXISTING_CAPTURE_SUPERSET"
        attempt["reasons"] = existing_status.get("reasons") or [
            "existing_capture_extra_unexpected_runners"
        ]
        return True
    attempt["status"] = "BLOCKED_EXISTING_CAPTURE_INVALID"
    attempt["reasons"] = existing_status.get("reasons") or [f"existing_capture_{status}"]
    return True


def capture_window_temporal_status(
    *,
    jump_datetime: datetime | None,
    current_time: datetime,
    offset_minutes: int,
) -> dict[str, Any]:
    if jump_datetime is None:
        return {
            "status": "UNKNOWN",
            "target_capture_at": None,
            "minutes_until_target": None,
            "minutes_to_jump": None,
            "reason": "jump_datetime_missing",
        }
    jump_cmp = jump_datetime
    current_cmp = current_time
    if jump_cmp.tzinfo is None and current_cmp.tzinfo is not None:
        jump_cmp = jump_cmp.replace(tzinfo=current_cmp.tzinfo)
    if current_cmp.tzinfo is None and jump_cmp.tzinfo is not None:
        current_cmp = current_cmp.replace(tzinfo=jump_cmp.tzinfo)
    target = jump_cmp - timedelta(minutes=offset_minutes)
    minutes_until_target = (target - current_cmp).total_seconds() / 60.0
    minutes_to_jump = (jump_cmp - current_cmp).total_seconds() / 60.0
    if minutes_to_jump <= 0:
        status = "AFTER_JUMP"
        reason = "race_already_jumped"
    elif minutes_until_target > 0:
        status = "PENDING"
        reason = "before_capture_window"
    else:
        status = "DUE_OR_PASSED_PRE_JUMP"
        reason = "capture_window_due_or_passed"
    return {
        "status": status,
        "target_capture_at": target.isoformat(),
        "minutes_until_target": minutes_until_target,
        "minutes_to_jump": minutes_to_jump,
        "reason": reason,
    }


def capture_window_coverage_for_item(
    *,
    db_path: Path,
    plan_item: Mapping[str, Any],
    current_time: datetime,
) -> list[dict[str, Any]]:
    race_id = str(plan_item.get("race_id") or "")
    expected_runners = [
        row for row in plan_item.get("expected_runners") or [] if isinstance(row, Mapping)
    ]
    jump_datetime = None
    jump_text = plan_item.get("jump_datetime")
    if jump_text:
        try:
            jump_datetime = datetime.fromisoformat(str(jump_text))
        except ValueError:
            jump_datetime = None
    rows: list[dict[str, Any]] = []
    for offset in CAPTURE_WINDOWS_MINUTES:
        capture_mode = f"autonomous_prejump_t{offset}m"
        temporal = capture_window_temporal_status(
            jump_datetime=jump_datetime,
            current_time=current_time,
            offset_minutes=offset,
        )
        existing = (
            existing_capture_runner_status(
                db_path,
                race_id=race_id,
                capture_mode=capture_mode,
                expected_runners=expected_runners,
                jump_datetime=jump_datetime,
                capture_window_minutes=offset,
            )
            if race_id
            else {
                "status": "INVALID",
                "existing_row_count": 0,
                "reasons": ["race_id_missing"],
            }
        )
        existing_status = existing.get("status")
        if existing_status == "COMPLETE":
            status = "CAPTURED"
            reason = "complete_existing_capture"
        elif existing_status == "INVALID":
            status = "BLOCKED_EXISTING_CAPTURE_INVALID"
            reason = "existing_capture_invalid"
        elif temporal["status"] == "AFTER_JUMP":
            status = "MISSED"
            reason = "window_passed_without_complete_capture"
        elif temporal["status"] == "DUE_OR_PASSED_PRE_JUMP":
            due_window, _window_status = due_capture_window(
                float(temporal.get("minutes_to_jump"))
                if isinstance(temporal.get("minutes_to_jump"), (int, float))
                else None
            )
            if due_window == offset:
                status = "DUE"
                reason = "window_due_without_complete_capture"
            else:
                status = "MISSED"
                reason = "earlier_window_passed_without_complete_capture"
        elif temporal["status"] == "PENDING":
            status = "PENDING"
            reason = "window_not_open_yet"
        else:
            status = "UNKNOWN"
            reason = temporal.get("reason") or "window_status_unknown"
        rows.append(
            {
                "schema_version": "autonomous_live_odds_capture_window_coverage_v1",
                "race_id": plan_item.get("race_id"),
                "venue": plan_item.get("venue"),
                "race_number": plan_item.get("race_number"),
                "race_date": plan_item.get("race_date"),
                "jump_datetime": plan_item.get("jump_datetime"),
                "offset_minutes": offset,
                "capture_mode": capture_mode,
                "target_capture_at": temporal.get("target_capture_at"),
                "minutes_until_target": temporal.get("minutes_until_target"),
                "minutes_to_jump": temporal.get("minutes_to_jump"),
                "temporal_status": temporal.get("status"),
                "existing_capture_status": existing_status,
                "existing_capture_count": int(existing.get("existing_row_count") or 0),
                "status": status,
                "reason": reason,
                "existing_capture_reasons": existing.get("reasons") or [],
            }
        )
    return rows


def build_capture_window_coverage(
    plan: Mapping[str, Any],
    *,
    db_path: Path,
    current_time: datetime,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in plan.get("races") or []:
        if isinstance(item, Mapping):
            rows.extend(
                capture_window_coverage_for_item(
                    db_path=db_path,
                    plan_item=item,
                    current_time=current_time,
                )
            )
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "schema_version": "autonomous_live_odds_capture_window_coverage_report_v1",
        "generated_at": current_time.isoformat(),
        "capture_window_offsets_minutes": list(CAPTURE_WINDOWS_MINUTES),
        "race_count": len(
            {
                str(row.get("race_id"))
                for row in rows
                if row.get("race_id") not in (None, "")
            }
        ),
        "window_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "windows": rows,
        "no_write_guarantees": {
            **NO_WRITE_GUARANTEES,
            "db_write": False,
        },
    }


def capture_window_next_action(
    coverage: Mapping[str, Any],
    *,
    current_time: datetime,
) -> dict[str, Any]:
    windows = [
        row for row in coverage.get("windows") or [] if isinstance(row, Mapping)
    ]
    due_windows = [row for row in windows if row.get("status") == "DUE"]
    pending_windows = [row for row in windows if row.get("status") == "PENDING"]
    if due_windows:
        return {
            "next_meaningful_action": "RUN_ODDS_CAPTURE_NOW",
            "next_meaningful_action_at": current_time.isoformat(),
            "next_meaningful_action_reason": "due_capture_windows_present",
            "next_due_capture_window_count": len(due_windows),
            "next_pending_capture_window_count": len(pending_windows),
        }

    def target_at(row: Mapping[str, Any]) -> str:
        return str(row.get("target_capture_at") or "")

    pending_with_targets = [row for row in pending_windows if target_at(row)]
    if pending_with_targets:
        next_window = sorted(pending_with_targets, key=target_at)[0]
        return {
            "next_meaningful_action": "WAIT_UNTIL_NEXT_FIXED_WINDOW",
            "next_meaningful_action_at": next_window.get("target_capture_at"),
            "next_meaningful_action_reason": "pending_capture_window",
            "next_due_capture_window_count": 0,
            "next_pending_capture_window_count": len(pending_windows),
        }

    return {
        "next_meaningful_action": "REFRESH_UPCOMING_RACE_WINDOW",
        "next_meaningful_action_at": current_time.isoformat(),
        "next_meaningful_action_reason": "no_due_or_pending_capture_windows",
        "next_due_capture_window_count": 0,
        "next_pending_capture_window_count": 0,
    }


def append_validated_capture(
    *,
    db_path: Path,
    plan_item: Mapping[str, Any],
    validation: Mapping[str, Any],
    current_time: datetime,
) -> dict[str, Any]:
    from sportsbet_odds_integrator import SportsbetOddsIntegrator

    race_info = {
        "race_id": plan_item.get("race_id"),
        "venue": plan_item.get("venue"),
        "race_number": plan_item.get("race_number"),
        "race_date": plan_item.get("race_date"),
        "race_time": plan_item.get("race_time"),
        "venue_url": validation.get("source_url"),
        "sportsbet_url": validation.get("source_url"),
        "preserve_race_id": True,
    }
    capture_mode = f"autonomous_prejump_t{plan_item.get('capture_window_minutes')}m"
    integrator = SportsbetOddsIntegrator(
        str(db_path),
        allow_auto_scrape_odds=True,
        setup_database=False,
    )
    place_report = integrator.append_pre_jump_odds_snapshot(
        race_info,
        list(validation.get("accepted_place_rows") or []),
        market_type="place",
        topN=DEFAULT_PLACE_TOPN,
        capture_mode=capture_mode,
        capture_timestamp=current_time.isoformat(),
        write_race_metadata=False,
    )
    if place_report.get("status") != "SUCCESS" or int(
        place_report.get("inserted_rows") or 0
    ) <= 0:
        return {
            "status": "FAILED",
            "race_id": plan_item.get("race_id"),
            "source_url": validation.get("source_url"),
            "capture_mode": capture_mode,
            "capture_timestamp": current_time.isoformat(),
            "market_types": list(REQUIRED_CAPTURE_MARKETS),
            "inserted_rows": int(place_report.get("inserted_rows") or 0),
            "win_inserted_rows": 0,
            "place_inserted_rows": int(place_report.get("inserted_rows") or 0),
            "warnings": [
                f"place:{warning}"
                for warning in (place_report.get("warnings") or [])
            ],
            "append_only": True,
            "market_reports": {
                "win": {
                    "status": "SKIPPED",
                    "warnings": ["place_append_failed_before_win_append"],
                    "inserted_rows": 0,
                },
                "place": place_report,
            },
        }
    win_report = integrator.append_pre_jump_odds_snapshot(
        race_info,
        list(validation.get("accepted_rows") or []),
        market_type="win",
        capture_mode=capture_mode,
        capture_timestamp=current_time.isoformat(),
        write_race_metadata=False,
    )
    warnings = [
        f"{market}:{warning}"
        for market, report in (("place", place_report), ("win", win_report))
        for warning in (report.get("warnings") or [])
    ]
    inserted_rows = int(place_report.get("inserted_rows") or 0) + int(
        win_report.get("inserted_rows") or 0
    )
    status = (
        "SUCCESS"
        if place_report.get("status") == "SUCCESS"
        and win_report.get("status") == "SUCCESS"
        and int(place_report.get("inserted_rows") or 0) > 0
        and int(win_report.get("inserted_rows") or 0) > 0
        else "FAILED"
    )
    return {
        "status": status,
        "race_id": plan_item.get("race_id"),
        "source_url": validation.get("source_url"),
        "capture_mode": capture_mode,
        "capture_timestamp": current_time.isoformat(),
        "market_types": list(REQUIRED_CAPTURE_MARKETS),
        "inserted_rows": inserted_rows,
        "win_inserted_rows": int(win_report.get("inserted_rows") or 0),
        "place_inserted_rows": int(place_report.get("inserted_rows") or 0),
        "warnings": warnings,
        "append_only": True,
        "market_reports": {
            "win": win_report,
            "place": place_report,
        },
    }


def execute_capture_plan(
    plan: Mapping[str, Any],
    *,
    db_path: Path,
    current_time: datetime,
    execute: bool,
    allow_auto_scrape_odds: bool,
    current_time_provider: Callable[[], datetime] | None = None,
    fetch_timeout_seconds: float = DEFAULT_FETCH_TIMEOUT_SECONDS,
    progress_dir: Path | None = None,
    receipt_publisher: Callable[..., Mapping[str, Any]] | None = None,
    forward_corpus_admitter: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    time_provider = current_time_provider or (lambda: datetime.now().astimezone())
    attempts: list[dict[str, Any]] = []
    inserted_rows = 0
    validation_pass_count = 0
    for item in plan.get("races") or []:
        if not isinstance(item, Mapping):
            continue
        attempt: dict[str, Any] = {
            "schema_version": "autonomous_live_odds_capture_attempt_v1",
            "race_id": item.get("race_id"),
            "status": "SKIPPED_NOT_READY",
            "plan_status": item.get("status"),
            "capture_window_minutes": item.get("capture_window_minutes"),
            "inserted_rows": 0,
            "reasons": [],
        }
        if item.get("status") != "READY_TO_CAPTURE":
            attempt["reasons"] = item.get("blockers") or ["plan_item_not_ready"]
            attempts.append(attempt)
            flush_attempt_progress(progress_dir, attempts=attempts)
            continue
        capture_mode = f"autonomous_prejump_t{item.get('capture_window_minutes')}m"
        if not execute:
            attempt["status"] = "PLANNED_NOT_EXECUTED"
            attempt["reasons"] = ["execute_flag_not_set"]
            attempts.append(attempt)
            flush_attempt_progress(progress_dir, attempts=attempts)
            continue
        if not allow_auto_scrape_odds:
            attempt["status"] = "BLOCKED_AUTO_SCRAPE_NOT_APPROVED"
            attempt["reasons"] = ["allow_auto_scrape_odds_flag_not_set"]
            attempts.append(attempt)
            flush_attempt_progress(progress_dir, attempts=attempts)
            continue
        fetch_time = time_provider()
        attempt["fetch_time"] = fetch_time.isoformat()
        executable_item = refresh_plan_item_for_time(item, fetch_time)
        if executable_item.get("status") != "READY_TO_CAPTURE":
            attempt["status"] = "BLOCKED_TIME_GATE_BEFORE_FETCH"
            attempt["reasons"] = executable_item.get("blockers") or ["fresh_time_gate_failed"]
            attempt["fresh_plan_status"] = executable_item.get("status")
            attempt["fresh_minutes_to_jump"] = executable_item.get("minutes_to_jump")
            attempts.append(attempt)
            flush_attempt_progress(progress_dir, attempts=attempts)
            continue
        item = executable_item
        attempt["capture_window_minutes"] = item.get("capture_window_minutes")
        capture_mode = f"autonomous_prejump_t{item.get('capture_window_minutes')}m"
        existing_status = existing_capture_runner_status(
            db_path,
            race_id=str(item.get("race_id")),
            capture_mode=capture_mode,
            expected_runners=[
                row for row in item.get("expected_runners") or [] if isinstance(row, Mapping)
            ],
            jump_datetime=parse_iso_datetime(item.get("jump_datetime")),
            capture_window_minutes=parse_int_value(item.get("capture_window_minutes")),
        )
        if block_or_skip_existing_capture_attempt(attempt, existing_status):
            if (
                attempt.get("status") == "SKIPPED_ALREADY_CAPTURED"
                and forward_corpus_admitter is not None
            ):
                try:
                    attempt["forward_corpus_admission"] = dict(
                        forward_corpus_admitter(
                            plan_item=item,
                            attempt=None,
                            receipt_publish=None,
                            emitted_at=time_provider(),
                        )
                    )
                except Exception as exc:
                    attempt["forward_corpus_admission"] = {
                        "schema_version": "scheduled-forward-corpus-admission-v1",
                        "status": "REJECTED",
                        "reason": type(exc).__name__,
                    }
            attempts.append(attempt)
            flush_attempt_progress(progress_dir, attempts=attempts)
            continue

        attempt["status"] = "FETCH_IN_PROGRESS"
        attempt["fetch_timeout_seconds"] = fetch_timeout_seconds
        flush_attempt_progress(progress_dir, attempts=attempts, active_attempt=attempt)
        try:
            fetch_result = fetch_odds_for_target_race_with_timeout(
                str(db_path),
                item.get("venue"),
                item.get("race_number"),
                item.get("race_date"),
                allow_auto_scrape_odds=True,
                timeout_seconds=fetch_timeout_seconds,
            )
        except FetchTimeoutError as exc:
            attempt["status"] = "BLOCKED_FETCH_TIMEOUT"
            attempt["reasons"] = [f"fetch_timeout:{fetch_timeout_seconds:g}s"]
            attempt["exception_message"] = str(exc)[:500]
            attempts.append(attempt)
            flush_attempt_progress(progress_dir, attempts=attempts)
            continue
        except Exception as exc:
            attempt["status"] = "BLOCKED_FETCH_EXCEPTION"
            attempt["reasons"] = [f"fetch_exception:{type(exc).__name__}"]
            attempt["exception_message"] = str(exc)[:500]
            attempts.append(attempt)
            flush_attempt_progress(progress_dir, attempts=attempts)
            continue
        validation = validate_fetched_odds(item, fetch_result)
        attempt["fetch_result"] = {
            key: value
            for key, value in fetch_result.items()
            if key not in {"odds_data", "odds_data_place", "race_info"}
        }
        attempt["validation"] = validation
        if validation.get("status") != "PASS":
            attempt["status"] = "BLOCKED_VALIDATION_FAILED"
            attempt["reasons"] = validation.get("reasons") or ["validation_failed"]
            attempts.append(attempt)
            flush_attempt_progress(progress_dir, attempts=attempts)
            continue
        validation_pass_count += 1
        append_time = time_provider()
        attempt["append_time"] = append_time.isoformat()
        append_item = refresh_plan_item_for_time(item, append_time)
        if append_item.get("status") != "READY_TO_CAPTURE":
            attempt["status"] = "BLOCKED_TIME_GATE_BEFORE_APPEND"
            attempt["reasons"] = append_item.get("blockers") or ["fresh_time_gate_failed"]
            attempt["fresh_plan_status"] = append_item.get("status")
            attempt["fresh_minutes_to_jump"] = append_item.get("minutes_to_jump")
            attempts.append(attempt)
            flush_attempt_progress(progress_dir, attempts=attempts)
            continue
        item = append_item
        attempt["capture_window_minutes"] = item.get("capture_window_minutes")
        capture_mode = f"autonomous_prejump_t{item.get('capture_window_minutes')}m"
        existing_status = existing_capture_runner_status(
            db_path,
            race_id=str(item.get("race_id")),
            capture_mode=capture_mode,
            expected_runners=[
                row for row in item.get("expected_runners") or [] if isinstance(row, Mapping)
            ],
            jump_datetime=parse_iso_datetime(item.get("jump_datetime")),
            capture_window_minutes=parse_int_value(item.get("capture_window_minutes")),
        )
        if block_or_skip_existing_capture_attempt(attempt, existing_status):
            if (
                attempt.get("status") == "SKIPPED_ALREADY_CAPTURED"
                and forward_corpus_admitter is not None
            ):
                try:
                    attempt["forward_corpus_admission"] = dict(
                        forward_corpus_admitter(
                            plan_item=item,
                            attempt=None,
                            receipt_publish=None,
                            emitted_at=time_provider(),
                        )
                    )
                except Exception as exc:
                    attempt["forward_corpus_admission"] = {
                        "schema_version": "scheduled-forward-corpus-admission-v1",
                        "status": "REJECTED",
                        "reason": type(exc).__name__,
                    }
            attempts.append(attempt)
            flush_attempt_progress(progress_dir, attempts=attempts)
            continue
        try:
            append_report = append_validated_capture(
                db_path=db_path,
                plan_item=item,
                validation=validation,
                current_time=append_time,
            )
        except Exception as exc:
            attempt["status"] = "APPEND_FAILED"
            attempt["reasons"] = [f"append_exception:{type(exc).__name__}"]
            attempt["exception_message"] = str(exc)[:500]
            attempts.append(attempt)
            flush_attempt_progress(progress_dir, attempts=attempts)
            continue
        attempt["append_report"] = append_report
        attempt["inserted_rows"] = int(append_report.get("inserted_rows") or 0)
        inserted_rows += int(attempt["inserted_rows"])
        attempt["status"] = (
            "APPENDED"
            if append_report.get("status") == "SUCCESS" and attempt["inserted_rows"] > 0
            else "APPEND_FAILED"
        )
        if attempt["status"] == "APPENDED" and receipt_publisher is not None:
            sealed_attempt = dict(attempt)
            try:
                attempt["collector_exact_receipt_publish"] = dict(
                    receipt_publisher(
                        plan_item=item,
                        attempt=sealed_attempt,
                        emitted_at=time_provider(),
                    )
                )
            except Exception as exc:
                attempt["collector_exact_receipt_publish"] = {
                    "schema_version": "collector_exact_capture_receipt_publish_v1",
                    "status": "REJECTED",
                    "reason": type(exc).__name__,
                }
            if (
                attempt["collector_exact_receipt_publish"].get("status")
                == "PUBLISHED"
                and forward_corpus_admitter is not None
            ):
                try:
                    attempt["forward_corpus_admission"] = dict(
                        forward_corpus_admitter(
                            plan_item=item,
                            attempt=sealed_attempt,
                            receipt_publish=attempt[
                                "collector_exact_receipt_publish"
                            ],
                            emitted_at=time_provider(),
                        )
                    )
                except Exception as exc:
                    attempt["forward_corpus_admission"] = {
                        "schema_version": "scheduled-forward-corpus-admission-v1",
                        "status": "REJECTED",
                        "reason": type(exc).__name__,
                    }
        attempts.append(attempt)
        flush_attempt_progress(progress_dir, attempts=attempts)

    status_counts: dict[str, int] = {}
    for attempt in attempts:
        status = str(attempt.get("status") or "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1
    blocked_attempts = blocked_attempt_summaries(attempts)
    t2_miss_summary = t2_miss_cause_summary(attempts)
    window_coverage = build_capture_window_coverage(
        plan,
        db_path=db_path,
        current_time=current_time,
    )
    next_action = capture_window_next_action(
        window_coverage,
        current_time=current_time,
    )
    ready_count = int(plan.get("ready_count") or 0)
    candidate_count = len([item for item in plan.get("races") or [] if isinstance(item, Mapping)])
    completed_count = len(attempts)
    receipt_publish_results = [
        attempt["collector_exact_receipt_publish"]
        for attempt in attempts
        if isinstance(attempt.get("collector_exact_receipt_publish"), Mapping)
    ]
    receipt_publish_count = sum(
        int(result.get("receipt_count") or 0)
        for result in receipt_publish_results
        if result.get("status") == "PUBLISHED"
    )
    receipt_publish_failure_count = sum(
        1 for result in receipt_publish_results if result.get("status") != "PUBLISHED"
    )
    forward_corpus_results = [
        attempt["forward_corpus_admission"]
        for attempt in attempts
        if isinstance(attempt.get("forward_corpus_admission"), Mapping)
    ]
    forward_corpus_success_count = sum(
        1
        for result in forward_corpus_results
        if result.get("status") in {"PREJUMP_CAPTURED", "EXACT_REPLAY"}
    )
    ready_race_ids = [
        str(item.get("race_id"))
        for item in plan.get("races") or []
        if isinstance(item, Mapping)
        and item.get("status") == "READY_TO_CAPTURE"
        and item.get("race_id") not in (None, "")
    ]
    if status_counts.get("APPENDED"):
        final_status = "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED"
    elif ready_count and not execute:
        final_status = "AUTONOMOUS_LIVE_ODDS_CAPTURE_READY_NOT_EXECUTED"
    elif any(status.startswith("BLOCKED") or status == "APPEND_FAILED" for status in status_counts):
        final_status = "AUTONOMOUS_LIVE_ODDS_CAPTURE_BLOCKED"
    else:
        final_status = "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS"

    operator_fields = capture_report_operator_fields(
        final_status,
        blocked_attempt_count=len(blocked_attempts),
    )
    return {
        "schema_version": "autonomous_live_odds_capture_report_v1",
        "generated_at": current_time.isoformat(),
        "final_status": final_status,
        **operator_fields,
        "execute": execute,
        "allow_auto_scrape_odds": allow_auto_scrape_odds,
        "db_path": str(db_path),
        "candidate_count": candidate_count,
        "completed_count": completed_count,
        "appended_attempt_count": int(status_counts.get("APPENDED") or 0),
        "skipped_already_captured_count": int(
            status_counts.get("SKIPPED_ALREADY_CAPTURED") or 0
        ),
        "status_counts": status_counts,
        "blocked_attempt_count": len(blocked_attempts),
        "blocked_attempts": blocked_attempts,
        **t2_miss_summary,
        "ready_count": ready_count,
        "ready_race_count": len(ready_race_ids),
        "ready_race_ids": ready_race_ids,
        "validation_pass_count": validation_pass_count,
        "inserted_live_odds_rows": inserted_rows,
        "collector_exact_receipt_publish_count": receipt_publish_count,
        "collector_exact_receipt_publish_failure_count": (
            receipt_publish_failure_count
        ),
        "forward_corpus_admission_success_count": forward_corpus_success_count,
        "forward_corpus_admission_failure_count": (
            len(forward_corpus_results) - forward_corpus_success_count
        ),
        "fetch_timeout_seconds": fetch_timeout_seconds,
        **next_action,
        "capture_window_coverage": window_coverage,
        "attempts": attempts,
        "no_write_guarantees": {
            **NO_WRITE_GUARANTEES,
            "db_write": bool(inserted_rows),
        },
    }


def apply_manual_request_to_plan(
    plan: Mapping[str, Any],
    *,
    protocol: ManualPredictionCollectorProtocol,
    request_id: str,
    collector_run_id: str,
    current_time: datetime,
    execute: bool,
    allow_auto_scrape_odds: bool,
) -> tuple[dict[str, Any], str]:
    """Prioritize one claimed request before the collector enters active capture."""

    context = protocol.claimed_request(request_id)
    try:
        prioritized = protocol.prioritize_capture_plan(
            context,
            plan,
            now=current_time,
        )
    except ProtocolRejected as exc:
        existing_response = protocol.read_response(request_id)
        if existing_response is not None:
            status = str(existing_response["status"])
            return {
                **dict(plan),
                "manual_request_id": request_id,
                "manual_request_prioritized": False,
                "manual_request_status": status,
            }, status
        status = (
            exc.code
            if exc.code
            in {"RACE_NOT_FOUND", "IDENTITY_MISMATCH", "CAPTURE_WINDOW_CLOSED"}
            else "CAPTURE_FAILED"
        )
        protocol.publish_terminal(
            context,
            status=status,
            now=current_time,
            reason=f"collector_plan_rejected:{exc.code}",
        )
        return {
            **dict(plan),
            "manual_request_id": request_id,
            "manual_request_prioritized": False,
            "manual_request_status": status,
        }, status
    if not execute or not allow_auto_scrape_odds:
        protocol.publish_terminal(
            context,
            status="CAPTURE_FAILED",
            now=current_time,
            reason="collector_capture_not_authorized",
        )
        return {
            **prioritized,
            "manual_request_status": "CAPTURE_FAILED",
        }, "CAPTURE_FAILED"
    try:
        protocol.begin_attempt(
            context,
            now=current_time,
            collector_run_id=collector_run_id,
        )
    except ProtocolRejected:
        existing_response = protocol.read_response(request_id)
        if existing_response is None:
            raise
        status = str(existing_response["status"])
        return {
            **prioritized,
            "manual_request_status": status,
        }, status
    return {
        **prioritized,
        "manual_request_status": "ATTEMPT_STARTED",
    }, "ATTEMPT_STARTED"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--db", type=Path, default=ROOT / "greyhound_racing_data.db")
    parser.add_argument("--current-time")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-auto-scrape-odds", action="store_true")
    parser.add_argument(
        "--fetch-timeout-seconds",
        type=float,
        default=DEFAULT_FETCH_TIMEOUT_SECONDS,
    )
    parser.add_argument("--manual-request-root", type=Path)
    parser.add_argument("--manual-request-id")
    parser.add_argument("--collector-receipt-root", type=Path)
    parser.add_argument("--collector-run-id")
    parser.add_argument("--forward-corpus-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    current_time = parse_current_time(args.current_time)
    evidence_root = args.evidence_root
    output_dir = assert_output_dir_safe(
        args.output_dir
        or evidence_root / f"autonomous_live_odds_capture_{now_id(current_time)}",
        evidence_root=evidence_root,
    )
    output_dir = unique_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    plan = build_capture_plan(
        args.input_dir,
        current_time=current_time,
        limit=args.limit,
    )
    manual_request_status = None
    if args.manual_request_id:
        if args.manual_request_root is None or not args.collector_run_id:
            raise ValueError("manual_request_collector_authority_missing")
        plan, manual_request_status = apply_manual_request_to_plan(
            plan,
            protocol=ManualPredictionCollectorProtocol(args.manual_request_root),
            request_id=args.manual_request_id,
            collector_run_id=args.collector_run_id,
            current_time=current_time,
            execute=args.execute,
            allow_auto_scrape_odds=args.allow_auto_scrape_odds,
        )
    receipt_publisher = None
    receipt_protocol = None
    if args.collector_receipt_root is not None:
        if (
            not args.collector_run_id
            or not args.execute
            or not args.allow_auto_scrape_odds
        ):
            raise ValueError("collector_receipt_authority_missing")
        from race_collection.synchronous_manual_capture import (
            publish_scheduled_capture_receipts,
        )

        receipt_protocol = ManualPredictionCollectorProtocol(
            args.collector_receipt_root
        )

        def receipt_publisher(**values: Any) -> Mapping[str, Any]:
            return publish_scheduled_capture_receipts(
                protocol=receipt_protocol,
                evidence_root=evidence_root,
                collector_run_id=args.collector_run_id,
                output_dir=output_dir,
                **values,
            )

    forward_corpus_admitter = None
    if args.forward_corpus_root is not None:
        if receipt_protocol is None or not args.collector_run_id:
            raise ValueError("forward_corpus_scheduled_receipt_authority_missing")
        from race_collection.scheduled_forward_corpus import (
            admit_scheduled_capture,
        )

        def forward_corpus_admitter(**values: Any) -> Mapping[str, Any]:
            return admit_scheduled_capture(
                protocol=receipt_protocol,
                evidence_root=evidence_root,
                corpus_root=args.forward_corpus_root,
                collector_run_id=args.collector_run_id,
                **values,
            )

    report = execute_capture_plan(
        plan,
        db_path=args.db,
        current_time=current_time,
        execute=args.execute,
        allow_auto_scrape_odds=args.allow_auto_scrape_odds,
        fetch_timeout_seconds=args.fetch_timeout_seconds,
        progress_dir=output_dir,
        receipt_publisher=receipt_publisher,
        forward_corpus_admitter=forward_corpus_admitter,
    )
    report = {
        **capture_report_identity_fields(output_dir),
        **report,
        "manual_request_id": args.manual_request_id,
        "manual_request_status": manual_request_status,
    }

    write_json(output_dir / "autonomous_live_odds_capture_plan.json", plan)
    write_jsonl(output_dir / "autonomous_live_odds_capture_attempts.jsonl", report["attempts"])
    write_json(
        output_dir / "autonomous_live_odds_capture_window_coverage.json",
        report["capture_window_coverage"],
    )
    write_json(output_dir / "autonomous_live_odds_capture_report.json", report)
    write_text(output_dir / "final_status.txt", str(report["final_status"]) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["final_status"] != "AUTONOMOUS_LIVE_ODDS_CAPTURE_BLOCKED" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
