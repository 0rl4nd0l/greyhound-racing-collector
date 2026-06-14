#!/usr/bin/env python3
"""Plan and optionally execute provenance-safe pre-jump live odds captures.

The default mode is report-only. Execution requires both ``--execute`` and
``--allow-auto-scrape-odds``. The only execution write is append-only dog-level
WIN rows in ``live_odds`` for races whose pre-jump TheDogs sidecar and
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

from odds_auto_integrator import fetch_odds_for_target_race  # noqa: E402
from sportsbet_odds_integrator import parse_sportsbet_runner_box_from_text  # noqa: E402
from scripts.daily_race_ingest_shadow_orchestrator import (  # noqa: E402
    is_thedogs_source_url,
    looks_post_result_source_url,
    parse_int_value,
    parse_jump_datetime,
)
from utils.runner_completeness import (  # noqa: E402
    normalise_runner_name,
    parse_runner_rows_from_csv,
)


DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/autonomous_live_odds_capture_"
CAPTURE_WINDOWS_MINUTES = (60, 30, 10, 2)
ACCEPTED_SPORTSBET_BOX_SOURCES = {"explicit_dom", "runner_text"}
ACCEPTED_DOG_LEVEL_ODDS_LEVELS = {"dog", "runner"}
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
                "validation_accepted_row_count": validation.get("accepted_row_count"),
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


def assert_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative = logical.absolute().relative_to(ROOT.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_autonomous_live_odds_capture_artifact:{relative}")
    return logical.absolute()


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
        output[box] = {
            "box_number": box,
            "dog_name": dog_name,
            "identity": identity,
        }
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


def build_capture_plan(
    input_dirs: Sequence[Path],
    *,
    current_time: datetime,
    limit: int | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for input_dir in input_dirs:
        for csv_path in sorted(Path(input_dir).glob("*.csv")):
            rows.append(build_plan_item(csv_path, current_time))
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


def refresh_plan_item_for_time(
    plan_item: Mapping[str, Any],
    current_time: datetime,
) -> dict[str, Any]:
    csv_path = plan_item_csv_path(plan_item)
    if csv_path is not None and csv_path.exists():
        return build_plan_item(csv_path, current_time)

    refreshed = dict(plan_item)
    reasons = list(refreshed.get("blockers") or [])
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


def fetched_odds_rows(fetch_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = fetch_result.get("odds_data")
    if not isinstance(rows, list):
        race_info = fetch_result.get("race_info")
        if isinstance(race_info, Mapping):
            rows = race_info.get("odds_data")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


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


def validate_fetched_odds(
    plan_item: Mapping[str, Any],
    fetch_result: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    rejected_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
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

    for raw_row in fetched_odds_rows(fetch_result):
        normalized, reason = normalize_fetched_row(raw_row)
        if normalized is None:
            rejected_rows.append({"row": raw_row, "reason": reason})
            continue
        accepted_rows.append(normalized)

    expected_rows = [
        dict(row)
        for row in plan_item.get("expected_runners") or []
        if isinstance(row, Mapping)
    ]
    expected_set = {
        (int(row["box_number"]), normalise_runner_name(row["dog_name"]))
        for row in expected_rows
        if row.get("box_number") not in (None, "") and row.get("dog_name")
    }
    accepted_set = {(int(row["box_number"]), row["identity"]) for row in accepted_rows}
    duplicate_keys = sorted(
        key
        for key in accepted_set
        if sum(1 for row in accepted_rows if (int(row["box_number"]), row["identity"]) == key) > 1
    )
    if duplicate_keys:
        reasons.append(
            "sportsbet_duplicate_runner_keys:"
            + ",".join(f"{box}:{identity}" for box, identity in duplicate_keys)
        )
    missing = sorted(expected_set - accepted_set)
    extra = sorted(accepted_set - expected_set)
    if missing:
        reasons.append(
            "sportsbet_missing_expected_runners:"
            + ",".join(f"{box}:{identity}" for box, identity in missing)
        )
    if extra:
        reasons.append(
            "sportsbet_extra_unexpected_runners:"
            + ",".join(f"{box}:{identity}" for box, identity in extra)
        )
    if rejected_rows:
        reasons.append(f"sportsbet_rejected_runner_rows:{len(rejected_rows)}")
    if not accepted_rows:
        reasons.append("sportsbet_accepted_runner_rows_zero")

    return {
        "schema_version": "autonomous_live_odds_capture_validation_v1",
        "status": "PASS" if not reasons else "FAIL",
        "source_url": source_url,
        "accepted_rows": accepted_rows,
        "accepted_row_count": len(accepted_rows),
        "rejected_rows": rejected_rows,
        "expected_runner_count": len(expected_set),
        "missing_expected_runners": [
            {"box_number": box, "identity": identity} for box, identity in missing
        ],
        "extra_unexpected_runners": [
            {"box_number": box, "identity": identity} for box, identity in extra
        ],
        "reasons": sorted(set(reasons)),
    }


def expected_runner_key_set(expected_runners: Sequence[Mapping[str, Any]]) -> set[tuple[int, str]]:
    keys: set[tuple[int, str]] = set()
    for row in expected_runners:
        box = parse_int_value(row.get("box_number"))
        identity = normalise_runner_name(row.get("dog_name") or row.get("identity"))
        if box is not None and identity:
            keys.add((box, identity))
    return keys


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
) -> dict[str, Any]:
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
        if "market_type" in row and str(row.get("market_type") or "").strip().lower() != "win":
            reasons.append("market_not_win")
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
    duplicate_keys = sorted(
        key for key in observed_set if sum(1 for value in observed_keys if value == key) > 1
    )
    missing = sorted(expected_set - observed_set)
    extra = sorted(observed_set - expected_set)
    reasons: list[str] = []
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
        "observed_runner_count": len(observed_set),
        "missing_expected_runners": [
            {"box_number": box, "identity": identity} for box, identity in missing
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
        "missing_expected_runners": [],
        "extra_unexpected_runners": [],
        "duplicate_runner_keys": [],
        "invalid_rows": [],
        "stale_capture_groups": [],
        "selected_capture_timestamp": None,
        "reasons": [],
    }
    if not db_path.exists():
        return status

    expected_set = expected_runner_key_set(expected_runners)
    status["expected_runner_count"] = len(expected_set)
    if not expected_set:
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
        group = runner_group_status(group_rows, expected_set=expected_set)
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
            extra = [
                (int(row["box_number"]), str(row["identity"]))
                for row in selected.get("extra_unexpected_runners") or []
            ]
            duplicate_keys = [
                (int(row["box_number"]), str(row["identity"]))
                for row in selected.get("duplicate_runner_keys") or []
            ]
            invalid_rows = list(selected.get("invalid_rows") or [])
            reasons = list(selected.get("reasons") or [])
    status.update(
        {
            "status": existing_status,
            "observed_runner_count": observed_runner_count,
            "missing_expected_runners": [
                {"box_number": box, "identity": identity} for box, identity in missing
            ],
            "extra_unexpected_runners": [
                {"box_number": box, "identity": identity} for box, identity in extra
            ],
            "duplicate_runner_keys": [
                {"box_number": box, "identity": identity} for box, identity in duplicate_keys
            ],
            "invalid_rows": invalid_rows,
            "stale_capture_groups": stale_groups,
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
    if status == "INCOMPLETE":
        attempt["status"] = "BLOCKED_EXISTING_CAPTURE_INCOMPLETE"
    else:
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
        elif existing_status == "INCOMPLETE":
            status = "BLOCKED_EXISTING_CAPTURE_INCOMPLETE"
            reason = "existing_capture_incomplete"
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
    return integrator.append_pre_jump_odds_snapshot(
        race_info,
        list(validation.get("accepted_rows") or []),
        capture_mode=capture_mode,
        capture_timestamp=current_time.isoformat(),
        write_race_metadata=False,
    )


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
            if key not in {"odds_data", "race_info"}
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
        "fetch_timeout_seconds": fetch_timeout_seconds,
        **next_action,
        "capture_window_coverage": window_coverage,
        "attempts": attempts,
        "no_write_guarantees": {
            **NO_WRITE_GUARANTEES,
            "db_write": bool(inserted_rows),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    current_time = parse_current_time(args.current_time)
    output_dir = assert_output_dir_safe(
        args.output_dir
        or DEFAULT_EVIDENCE_ROOT / f"autonomous_live_odds_capture_{now_id(current_time)}"
    )
    output_dir = unique_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    plan = build_capture_plan(
        args.input_dir,
        current_time=current_time,
        limit=args.limit,
    )
    report = execute_capture_plan(
        plan,
        db_path=args.db,
        current_time=current_time,
        execute=args.execute,
        allow_auto_scrape_odds=args.allow_auto_scrape_odds,
        fetch_timeout_seconds=args.fetch_timeout_seconds,
        progress_dir=output_dir,
    )
    report = {**capture_report_identity_fields(output_dir), **report}

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
