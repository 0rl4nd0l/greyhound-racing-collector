#!/usr/bin/env python3
"""Approval-gated pre-jump prediction improvement loop.

This is an operator planner/executor for the safe daily loop:
refresh, validate, dry-run capture, approved persist, approved odds capture,
official-first result dry-run, approved label write, read-only evaluation,
model-quality diagnosis, and report-only promotion control.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accuracy_program.snapshots import assert_no_result_fields  # noqa: E402
from utils.race_lifecycle import melbourne_now  # noqa: E402

PERSIST_DRY_RUN_REPORT_MAX_AGE_SECONDS = 10 * 60
RESULT_DRY_RUN_REPORT_MAX_AGE_SECONDS = 10 * 60
EVALUATION_REPORT_MAX_AGE_SECONDS = 10 * 60
MODEL_REVIEW_PACKET_MAX_AGE_SECONDS = 10 * 60
PERSIST_APPROVAL_WINDOW_CLOSING_SOON_SECONDS = 120
MIN_PROMOTION_CLEAN_OFFICIAL_RACES = 100
LOCAL_OPERATOR_TIMEZONE = "Australia/Melbourne"
LOCAL_OPERATOR_TZ = ZoneInfo(LOCAL_OPERATOR_TIMEZONE)
PREDICTION_PREVIEW_ALLOWED_FIELDS = (
    "predicted_rank",
    "box_number",
    "dog_name",
    "win_prob_norm",
    "odds_match_status",
    "market_odds_win",
    "ev_win",
    "quality_flags",
)

APPROVAL_GATES = {
    "live_persist": {
        "arg_name": "approve_live_persist",
        "cli_flag": "--approve-live-persist",
        "env_var": "APPROVE_LIVE_PERSIST",
    },
    "live_odds_capture": {
        "arg_name": "approve_live_odds_capture",
        "cli_flag": "--approve-live-odds-capture",
        "env_var": "APPROVE_LIVE_ODDS_CAPTURE",
    },
    "result_label_write": {
        "arg_name": "write_labels_approved",
        "cli_flag": "--write-labels-approved",
        "env_var": "APPROVE_RESULT_LABEL_WRITE",
    },
    "promotion": {
        "arg_name": "approve_promotion",
        "cli_flag": "--approve-promotion",
        "env_var": "APPROVE_MODEL_PROMOTION",
    },
}


def env_flag_enabled(name: str) -> bool:
    return str(os.environ.get(name) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "approved",
    }


def approval_state(args: argparse.Namespace) -> dict[str, bool]:
    return {
        name: details["approved"]
        for name, details in approval_provenance(args).items()
    }


def approval_provenance(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    details: dict[str, dict[str, Any]] = {}
    for name, config in APPROVAL_GATES.items():
        cli_approved = bool(getattr(args, str(config["arg_name"])))
        env_approved = env_flag_enabled(str(config["env_var"]))
        sources = []
        if cli_approved:
            sources.append("cli")
        if env_approved:
            sources.append("env")
        details[name] = {
            "approved": cli_approved or env_approved,
            "sources": sources,
            "cli_flag": config["cli_flag"],
            "cli_approved": cli_approved,
            "env_var": config["env_var"],
            "env_approved": env_approved,
        }
    return details


def _repo_python() -> str:
    candidate = ROOT / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _report_expiry_fields(
    gate: dict[str, Any],
    *,
    report_key: str,
) -> dict[str, Any]:
    expires_at_utc = None
    expires_at_local = None
    seconds_until_expiry = None
    max_age_seconds = gate.get("max_report_age_seconds")
    report_age_seconds = gate.get("report_age_seconds")
    if isinstance(max_age_seconds, (int, float)):
        if isinstance(report_age_seconds, (int, float)):
            seconds_until_expiry = round(
                max(0.0, float(max_age_seconds) - float(report_age_seconds)),
                3,
            )
        mtime_text = gate.get("report_mtime_utc")
        if mtime_text:
            try:
                expires_at = datetime.fromisoformat(str(mtime_text))
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
                expires_at = expires_at + timedelta(seconds=float(max_age_seconds))
                expires_at_utc = expires_at.astimezone(timezone.utc).isoformat(
                    timespec="seconds"
                )
                expires_at_local = expires_at.astimezone(LOCAL_OPERATOR_TZ).isoformat(
                    timespec="seconds"
                )
            except (TypeError, ValueError):
                expires_at_utc = None
                expires_at_local = None
    return {
        f"{report_key}_expires_at_utc": expires_at_utc,
        f"{report_key}_expires_at_local": expires_at_local,
        f"{report_key}_expires_at_local_timezone": LOCAL_OPERATOR_TIMEZONE,
        f"{report_key}_seconds_until_expiry": seconds_until_expiry,
        "approval_must_arrive_before_report_expiry": True,
        "rerun_required_after_expiry": True,
    }


def _persist_approval_window_urgency(
    window_status: str,
    seconds_until_expiry: Any,
) -> str:
    if window_status == "NOT_APPLICABLE_PERSISTED_CORPUS_PRESENT":
        return "NOT_APPLICABLE"
    if window_status == "WAITING_FOR_FUTURE_WINDOW":
        return "WAITING_FOR_WINDOW"
    if window_status == "REFRESH_REQUIRED":
        return "REFRESH_REQUIRED"
    if not window_status.startswith("OPEN"):
        return "UNKNOWN"
    try:
        seconds = float(seconds_until_expiry)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if seconds <= 0:
        return "REFRESH_REQUIRED"
    if seconds <= PERSIST_APPROVAL_WINDOW_CLOSING_SOON_SECONDS:
        return "CLOSING_SOON"
    return "OPEN"


def _command_with_required_flag(
    command: list[str],
    flag: str,
    *,
    insert_before: str | None = None,
) -> list[str]:
    approved_command = list(command)
    if flag in approved_command:
        return approved_command
    if insert_before and insert_before in approved_command:
        approved_command.insert(approved_command.index(insert_before), flag)
    else:
        approved_command.append(flag)
    return approved_command


def _race_id_command_args(
    race_ids: list[str] | tuple[str, ...] | None,
    *,
    flag: str = "--race-id",
) -> list[str]:
    args: list[str] = []
    for race_id in race_ids or []:
        text = str(race_id).strip()
        if text:
            args.extend([flag, text])
    return args


def _race_id_scope(race_ids: list[str] | tuple[str, ...] | None) -> list[str]:
    return sorted(
        {
            str(race_id).strip()
            for race_id in race_ids or []
            if str(race_id).strip()
        }
    )


def _approval_command_template_fields(
    *,
    command: list[str],
    flag: str,
    hard_stops: list[str],
    template_key: str,
    insert_before: str | None = None,
) -> dict[str, Any]:
    blocked = bool(hard_stops)
    return {
        template_key: None
        if blocked
        else _command_with_required_flag(
            command,
            flag,
            insert_before=insert_before,
        ),
        "approval_command_template_status": (
            "BLOCKED_BY_HARD_STOPS"
            if blocked
            else "READY_FOR_EXPLICIT_APPROVAL"
        ),
        "approval_command_template_blocked_reasons": list(hard_stops),
        "approval_command_requires_explicit_operator_confirmation": True,
    }


def _same_run_execute_ready_command(
    *,
    py: str,
    run_dir: Path,
    upcoming_dir: Path,
    snapshot_dir: Path,
    db_path: Path,
    date_text: str,
    min_minutes: float,
    max_minutes: float,
    limit: int,
    approval_flag: str,
    output_path: Path,
    result_race_ids: list[str] | tuple[str, ...] | None = None,
    report_only_calibration_design: Path | None = None,
) -> list[str]:
    command = [
        py,
        "scripts/prejump_prediction_loop.py",
        "--db",
        _rel(db_path),
        "--upcoming-dir",
        _rel(upcoming_dir),
        "--snapshot-dir",
        _rel(snapshot_dir),
        "--run-dir",
        _rel(run_dir),
        "--date",
        date_text,
        "--min-minutes",
        str(min_minutes),
        "--max-minutes",
        str(max_minutes),
        "--limit",
        str(limit),
        "--execute-ready",
        approval_flag,
        "--output",
        _rel(output_path),
    ]
    insert_at = command.index("--execute-ready")
    command[insert_at:insert_at] = _race_id_command_args(
        result_race_ids,
        flag="--result-race-id",
    )
    if report_only_calibration_design is not None:
        command[insert_at:insert_at] = [
            "--report-only-calibration-design",
            _rel(report_only_calibration_design),
        ]
    return command


def _same_run_execute_ready_command_template_fields(
    *,
    command: list[str] | None,
    rechecks: list[str],
) -> dict[str, Any]:
    return {
        "approved_same_run_execute_ready_command_template": list(command)
        if command
        else None,
        "same_run_execute_ready_command_template_status": (
            "READY_FOR_EXPLICIT_APPROVAL_AND_FRESH_RECHECK"
            if command
            else "DATA_MISSING_COMMAND_TEMPLATE"
        ),
        "same_run_execute_ready_command_requires_explicit_operator_confirmation": True,
        "same_run_execute_ready_rechecks": list(rechecks),
    }


def _prediction_snapshot_scan_for_date(
    snapshot_dir: Path,
    date_text: str,
) -> tuple[list[Path], list[dict[str, str]]]:
    date_dir = snapshot_dir if snapshot_dir.is_absolute() else ROOT / snapshot_dir
    date_dir = date_dir / date_text
    if not date_dir.exists():
        return [], []

    out: list[Path] = []
    rejected: list[dict[str, str]] = []
    for path in sorted(date_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        predictions = payload.get("predictions") if isinstance(payload, dict) else None
        if (
            isinstance(payload, dict)
            and payload.get("schema_version") == "prediction_snapshot_v1"
            and isinstance(predictions, list)
            and bool(predictions)
            and payload.get("is_pre_jump_snapshot") is True
            and payload.get("snapshot_state") == "pre_jump_feature_freeze"
            and isinstance(payload.get("snapshot_readiness"), dict)
            and payload["snapshot_readiness"].get("status") == "READY"
        ):
            try:
                assert_no_result_fields(payload)
            except ValueError as exc:
                rejected.append({"path": _rel(path), "reason": str(exc)})
                continue
            out.append(path)
    return out, rejected


def _prediction_snapshot_files_for_date(snapshot_dir: Path, date_text: str) -> list[Path]:
    files, _rejected = _prediction_snapshot_scan_for_date(snapshot_dir, date_text)
    return files


def _coerce_datetime_utc(
    value: Any,
    *,
    default_tz: ZoneInfo = LOCAL_OPERATOR_TZ,
) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=default_tz)
    return dt.astimezone(timezone.utc)


def _format_dt(dt: datetime | None, tz: ZoneInfo | timezone = timezone.utc) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(tz).isoformat(timespec="seconds")


def _persisted_snapshot_jump_status(
    snapshot_files: list[Path],
    *,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    if now_utc is None:
        now = datetime.now(timezone.utc)
    else:
        now = now_utc
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        now = now.astimezone(timezone.utc)

    future: list[tuple[datetime, dict[str, Any]]] = []
    due_or_past: list[tuple[datetime, dict[str, Any]]] = []
    unknown: list[dict[str, Any]] = []

    for path in snapshot_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            unknown.append(
                {
                    "path": _rel(path),
                    "reason": "snapshot_unreadable",
                }
            )
            continue
        if not isinstance(payload, dict):
            unknown.append(
                {
                    "path": _rel(path),
                    "reason": "snapshot_root_not_object",
                }
            )
            continue

        jump_text = payload.get("jump_datetime")
        jump_dt = _coerce_datetime_utc(jump_text)
        base = {
            "path": _rel(path),
            "race_id": payload.get("race_id"),
            "venue": payload.get("venue"),
            "race_number": payload.get("race_number"),
            "jump_datetime": jump_text,
        }
        if jump_dt is None:
            unknown.append(
                {
                    **base,
                    "reason": "jump_datetime_missing_or_unparseable",
                }
            )
            continue

        seconds_to_jump = round((jump_dt - now).total_seconds(), 3)
        entry = {
            **base,
            "jump_datetime_utc": _format_dt(jump_dt),
            "jump_datetime_local": _format_dt(jump_dt, LOCAL_OPERATOR_TZ),
        }
        if seconds_to_jump > 0:
            future.append(
                (
                    jump_dt,
                    {
                        **entry,
                        "seconds_to_jump": seconds_to_jump,
                    },
                )
            )
        else:
            due_or_past.append(
                (
                    jump_dt,
                    {
                        **entry,
                        "seconds_since_jump_or_due": round(abs(seconds_to_jump), 3),
                    },
                )
            )

    future.sort(key=lambda item: item[0])
    due_or_past.sort(key=lambda item: item[0], reverse=True)
    earliest_future = future[0][0] if future else None
    latest_future = future[-1][0] if future else None
    return {
        "schema_version": "persisted_snapshot_jump_status_v1",
        "evaluated_snapshot_count": len(snapshot_files),
        "now_utc": _format_dt(now),
        "now_local": _format_dt(now, LOCAL_OPERATOR_TZ),
        "known_jump_datetime_count": len(future) + len(due_or_past),
        "unknown_jump_datetime_count": len(unknown),
        "known_future_not_jumped_count": len(future),
        "known_jumped_or_due_count": len(due_or_past),
        "wait_for_known_future_jumps_before_result_dry_run": bool(future),
        "result_dry_run_wait_reason": (
            "known_persisted_races_not_jumped_yet" if future else None
        ),
        "earliest_known_future_jump_datetime_utc": _format_dt(earliest_future),
        "earliest_known_future_jump_datetime_local": _format_dt(
            earliest_future,
            LOCAL_OPERATOR_TZ,
        ),
        "latest_known_future_jump_datetime_utc": _format_dt(latest_future),
        "latest_known_future_jump_datetime_local": _format_dt(
            latest_future,
            LOCAL_OPERATOR_TZ,
        ),
        "known_future_not_jumped_examples": [item for _dt, item in future[:5]],
        "known_jumped_or_due_examples": [item for _dt, item in due_or_past[:5]],
        "unknown_jump_datetime_examples": unknown[:5],
    }


def _current_corpus_report(snapshot_dir: Path, date_text: str) -> dict[str, Any]:
    persisted_snapshot_files, rejected_snapshots = _prediction_snapshot_scan_for_date(
        snapshot_dir,
        date_text,
    )
    persisted_corpus_present = bool(persisted_snapshot_files)
    jump_status = _persisted_snapshot_jump_status(persisted_snapshot_files)
    return {
        "target_date": date_text,
        "snapshot_dir": _rel(snapshot_dir),
        "ready_persisted_prediction_snapshot_count_for_date": len(
            persisted_snapshot_files
        ),
        "ready_persisted_prediction_snapshot_examples": [
            _rel(path) for path in persisted_snapshot_files[:5]
        ],
        "result_contaminated_snapshot_rejection_count": len(rejected_snapshots),
        "result_contaminated_snapshot_rejection_examples": rejected_snapshots[:5],
        "persisted_snapshot_jump_status": jump_status,
        "known_future_not_jumped_snapshot_count_for_date": jump_status[
            "known_future_not_jumped_count"
        ],
        "result_dry_run_waiting_for_known_future_jumps": jump_status[
            "wait_for_known_future_jumps_before_result_dry_run"
        ],
        "status": "READY_PERSISTED_PREJUMP_SNAPSHOTS_PRESENT"
        if persisted_corpus_present
        else "NO_READY_PERSISTED_PREJUMP_SNAPSHOTS_FOR_DATE",
    }


def _persist_readiness_gate(
    report_path: Path,
    *,
    max_age_seconds: int | None = PERSIST_DRY_RUN_REPORT_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    path = report_path if report_path.is_absolute() else ROOT / report_path
    gate: dict[str, Any] = {
        "path": _rel(path),
        "clean_for_ready_subset_persist": False,
        "fresh_for_plan": False,
        "status": "DATA_MISSING",
        "reason": "dry_run_capture_report_missing",
        "ready_count": 0,
        "not_ready_count": 0,
        "capture_count": 0,
        "candidate_files": None,
        "ready_race_ids": [],
        "not_ready_race_ids": [],
        "report_mtime_utc": None,
        "report_age_seconds": None,
        "max_report_age_seconds": max_age_seconds,
    }
    if not path.exists():
        return gate
    stat = path.stat()
    report_mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
    report_age_seconds = max(
        0.0,
        (datetime.now(timezone.utc) - report_mtime).total_seconds(),
    )
    gate.update(
        {
            "report_mtime_utc": report_mtime.isoformat(timespec="seconds"),
            "report_age_seconds": round(report_age_seconds, 3),
            "fresh_for_plan": (
                max_age_seconds is None or report_age_seconds <= max_age_seconds
            ),
        }
    )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        gate["reason"] = f"dry_run_capture_report_unreadable:{type(exc).__name__}"
        return gate

    failures: list[str] = []
    if gate["fresh_for_plan"] is not True:
        failures.append("dry_run_capture_report_stale")
    captures = report.get("captures") if isinstance(report, dict) else None
    if not isinstance(report, dict):
        failures.append("report_root_not_object")
        captures = []
    else:
        if report.get("status") != "SUCCESS":
            failures.append("report_status_not_success")
        if report.get("dry_run") is not True:
            failures.append("report_is_not_dry_run")
        if report.get("persist_requested") is not False:
            failures.append("dry_run_precheck_should_not_request_persist")
        if report.get("persist_approved") is not False:
            failures.append("dry_run_precheck_should_not_be_persist_approved")
        if int(report.get("metadata_missing_count") or 0) != 0:
            failures.append("metadata_missing_count_nonzero")
        if int(report.get("metadata_unsafe_count") or 0) != 0:
            failures.append("metadata_unsafe_count_nonzero")
        if int(report.get("metadata_mismatch_count") or 0) != 0:
            failures.append("metadata_mismatch_count_nonzero")
        if not isinstance(captures, list):
            failures.append("captures_not_list")
            captures = []

    ready_race_ids: list[str] = []
    not_ready_race_ids: list[str] = []
    ev_readiness_counts: dict[str, int] = {}
    for capture in captures:
        if not isinstance(capture, dict):
            not_ready_race_ids.append("UNKNOWN_NON_OBJECT_CAPTURE")
            continue
        race_id = str(capture.get("race_id") or "UNKNOWN_RACE")
        readiness = capture.get("snapshot_readiness")
        readiness_status = (
            readiness.get("status")
            if isinstance(readiness, dict)
            else capture.get("snapshot_readiness_status")
        )
        if readiness_status == "READY":
            ready_race_ids.append(race_id)
        else:
            not_ready_race_ids.append(race_id)
        ev_readiness = capture.get("ev_readiness")
        ev_status = (
            ev_readiness.get("status")
            if isinstance(ev_readiness, dict)
            else capture.get("ev_readiness_status")
        )
        if ev_status:
            key = str(ev_status)
            ev_readiness_counts[key] = ev_readiness_counts.get(key, 0) + 1

    capture_count = len(captures)
    gate.update(
        {
            "ready_count": len(ready_race_ids),
            "not_ready_count": len(not_ready_race_ids),
            "capture_count": capture_count,
            "candidate_files": report.get("candidate_files")
            if isinstance(report, dict)
            else None,
            "ready_race_ids": ready_race_ids[:20],
            "not_ready_race_ids": not_ready_race_ids[:20],
            "lifecycle_counts": report.get("lifecycle_counts")
            if isinstance(report, dict)
            else None,
            "final_runner_set_counts": report.get("final_runner_set_counts")
            if isinstance(report, dict)
            else None,
            "target_metadata_counts": report.get("target_metadata_counts")
            if isinstance(report, dict)
            else None,
            "ev_readiness_counts": (
                report.get("ev_readiness_counts") or ev_readiness_counts
            )
            if isinstance(report, dict)
            else ev_readiness_counts,
        }
    )
    if capture_count <= 0:
        failures.append("capture_count_zero")
    if len(ready_race_ids) <= 0:
        failures.append("ready_count_zero")

    if failures:
        gate["status"] = "NOT_READY"
        gate["reason"] = ",".join(failures)
        return gate

    gate["clean_for_ready_subset_persist"] = True
    gate["fresh_for_plan"] = True
    gate["status"] = "READY" if len(ready_race_ids) == capture_count else "PARTIAL_READY"
    gate["reason"] = None
    return gate


def _sanitised_prediction_preview(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    preview: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        assert_no_result_fields(row)
        preview.append(
            {
                field: row.get(field)
                for field in PREDICTION_PREVIEW_ALLOWED_FIELDS
                if field in row
            }
        )
    return preview


def _dry_run_prediction_preview_report(
    report_path: Path,
    persist_readiness_gate: dict[str, Any],
    *,
    max_races: int = 20,
    max_runners_per_race: int = 8,
) -> dict[str, Any]:
    path = _rooted_path(report_path)
    report: dict[str, Any] = {
        "schema_version": "dry_run_prediction_preview_report_v1",
        "path": _rel(path),
        "status": "DATA_MISSING",
        "reason": "dry_run_capture_report_missing",
        "fresh_for_plan": persist_readiness_gate.get("fresh_for_plan") is True,
        "source_gate_status": persist_readiness_gate.get("status"),
        "source_gate_reason": persist_readiness_gate.get("reason"),
        "max_races": max_races,
        "max_runners_per_race": max_runners_per_race,
        "capture_count": 0,
        "preview_race_count": 0,
        "preview_runner_count": 0,
        "result_contaminated_capture_rejection_count": 0,
        "result_contaminated_capture_rejection_examples": [],
        "races": [],
    }
    if not path.exists():
        return report
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        report["status"] = "NOT_READY"
        report["reason"] = f"dry_run_capture_report_unreadable:{type(exc).__name__}"
        return report

    failures: list[str] = []
    warnings: list[str] = []
    if not isinstance(payload, dict):
        failures.append("report_root_not_object")
        payload = {}
    else:
        if payload.get("status") != "SUCCESS":
            failures.append("report_status_not_success")
        if payload.get("dry_run") is not True:
            failures.append("prediction_preview_requires_dry_run_report")
    if persist_readiness_gate.get("fresh_for_plan") is not True:
        warnings.append("dry_run_capture_report_stale")

    captures = payload.get("captures")
    if not isinstance(captures, list):
        failures.append("captures_not_list")
        captures = []

    races: list[dict[str, Any]] = []
    contaminated: list[dict[str, str]] = []
    preview_runner_count = 0
    for capture in captures:
        if not isinstance(capture, dict):
            continue
        race_id = str(capture.get("race_id") or "UNKNOWN_RACE")
        try:
            assert_no_result_fields(capture)
            preview = _sanitised_prediction_preview(capture.get("prediction_preview"))
        except ValueError as exc:
            contaminated.append({"race_id": race_id, "reason": str(exc)})
            continue
        if not preview:
            continue
        if len(races) >= max_races:
            continue
        preview = preview[:max_runners_per_race]
        preview_runner_count += len(preview)
        readiness = capture.get("snapshot_readiness")
        readiness_status = (
            readiness.get("status")
            if isinstance(readiness, dict)
            else capture.get("snapshot_readiness_status")
        )
        ev_readiness = capture.get("ev_readiness")
        races.append(
            {
                "race_id": race_id,
                "race_file": capture.get("race_file"),
                "snapshot_readiness_status": readiness_status,
                "lifecycle_status": capture.get("lifecycle_status"),
                "runner_count": capture.get("runner_count"),
                "ev_readiness_status": (
                    ev_readiness.get("status")
                    if isinstance(ev_readiness, dict)
                    else capture.get("ev_readiness_status")
                ),
                "probability_sum_check": capture.get("probability_sum_check"),
                "prediction_preview": preview,
            }
        )

    if contaminated:
        failures.append("result_field_leakage_detected")
    if not races and not contaminated:
        failures.append("prediction_preview_empty")

    report.update(
        {
            "capture_count": len(captures),
            "preview_race_count": 0 if contaminated else len(races),
            "preview_runner_count": 0 if contaminated else preview_runner_count,
            "result_contaminated_capture_rejection_count": len(contaminated),
            "result_contaminated_capture_rejection_examples": contaminated[:5],
            "races": [] if contaminated else races,
        }
    )
    if failures:
        report["status"] = "NOT_READY"
        report["reason"] = ",".join(failures)
        return report

    report["status"] = "STALE_AVAILABLE" if warnings else "READY"
    report["reason"] = ",".join(warnings) if warnings else None
    return report


def _normalise_path(path: Path) -> str:
    path = path.expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return str(path.resolve())


def _rooted_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def _int_count(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _count_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for key, raw_count in value.items():
        text = str(key)
        if not text:
            continue
        counts[text] = _int_count(raw_count)
    return counts


def _merge_counts(target: dict[str, int], value: Any) -> None:
    for key, count in _count_mapping(value).items():
        target[key] = target.get(key, 0) + count


def _ev_status_from_capture(capture: dict[str, Any]) -> str | None:
    ev_readiness = capture.get("ev_readiness")
    status = (
        ev_readiness.get("status")
        if isinstance(ev_readiness, dict)
        else capture.get("ev_readiness_status")
    )
    if status:
        return str(status)
    return None


def _ev_summary_from_persist_readiness_gate(
    persist_readiness_gate: dict[str, Any],
) -> dict[str, Any]:
    counts = _count_mapping(persist_readiness_gate.get("ev_readiness_counts"))
    return {
        "schema_version": "ev_readiness_summary_v1",
        "ev_summary_source": "dry_run_persist_readiness_gate",
        "ev_readiness_counts": counts,
        "ev_ready_count": counts.get("EV_READY", 0),
        "ev_not_ready_count": counts.get("EV_NOT_READY", 0),
        "priced_ev_runner_count": 0,
        "odds_exclusion_counts": {},
        "odds_exclusion_count": 0,
        "authoritative_capture_report_path": None,
        "ev_summary_consistency_check": "NOT_APPLICABLE_DRY_RUN_GATE",
        "ev_summary_failure_reason": None,
    }


def _authoritative_capture_report_ev_summary(report_path: Path) -> dict[str, Any]:
    path = _rooted_path(report_path)
    summary: dict[str, Any] = {
        "schema_version": "ev_readiness_summary_v1",
        "ev_summary_source": "DATA_MISSING",
        "ev_readiness_counts": {},
        "ev_ready_count": 0,
        "ev_not_ready_count": 0,
        "priced_ev_runner_count": 0,
        "odds_exclusion_counts": {},
        "odds_exclusion_count": 0,
        "authoritative_capture_report_path": _rel(path),
        "ev_summary_consistency_check": "NOT_CHECKED",
        "ev_summary_failure_reason": "authoritative_capture_report_missing",
    }
    if not path.exists():
        return summary
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        summary.update(
            {
                "ev_summary_source": "UNREADABLE_AUTHORITATIVE_CAPTURE_REPORT",
                "ev_summary_failure_reason": (
                    f"authoritative_capture_report_unreadable:{type(exc).__name__}"
                ),
            }
        )
        return summary

    if not isinstance(report, dict):
        summary.update(
            {
                "ev_summary_source": "INVALID_AUTHORITATIVE_CAPTURE_REPORT",
                "ev_summary_failure_reason": "authoritative_capture_report_root_not_object",
            }
        )
        return summary

    is_authoritative_persist = (
        report.get("status") == "SUCCESS"
        and report.get("dry_run") is False
        and report.get("persist_requested") is True
        and report.get("persist_approved") is True
    )
    if not is_authoritative_persist:
        summary.update(
            {
                "ev_summary_source": "NOT_AUTHORITATIVE_CAPTURE_REPORT",
                "ev_summary_failure_reason": "capture_report_is_not_approved_persist_report",
                "ev_summary_consistency_check": "REJECTED_NON_PERSISTED_REPORT",
            }
        )
        return summary

    captures = report.get("captures")
    captures = captures if isinstance(captures, list) else []
    report_counts = _count_mapping(report.get("ev_readiness_counts"))
    capture_counts: dict[str, int] = {}
    report_has_priced_ev_runner_count = report.get("priced_ev_runner_count") is not None
    report_priced_ev_runner_count = _int_count(report.get("priced_ev_runner_count"))
    capture_priced_ev_runner_count = 0
    report_has_odds_exclusion_counts = isinstance(
        report.get("odds_exclusion_counts"), dict
    )
    report_odds_exclusion_counts = _count_mapping(report.get("odds_exclusion_counts"))
    capture_odds_exclusion_counts: dict[str, int] = {}

    for capture in captures:
        if not isinstance(capture, dict):
            continue
        status = _ev_status_from_capture(capture)
        if status:
            capture_counts[status] = capture_counts.get(status, 0) + 1
        capture_priced_ev_runner_count += _int_count(
            capture.get("priced_ev_runner_count")
        )
        _merge_counts(
            capture_odds_exclusion_counts,
            capture.get("odds_exclusion_counts"),
        )

    counts = report_counts or capture_counts
    priced_ev_runner_count = (
        report_priced_ev_runner_count
        if report_has_priced_ev_runner_count
        else capture_priced_ev_runner_count
    )
    odds_exclusion_counts = (
        report_odds_exclusion_counts
        if report_has_odds_exclusion_counts
        else capture_odds_exclusion_counts
    )
    consistency_check = "NO_EV_STATUS_COUNTS"
    failure_reason = None
    if report_counts and capture_counts:
        if report_counts == capture_counts:
            consistency_check = "MATCH"
        else:
            consistency_check = "MISMATCH"
            failure_reason = "report_ev_readiness_counts_disagree_with_capture_statuses"
    elif report_counts:
        consistency_check = "SOURCE_COUNTS_USED"
    elif capture_counts:
        consistency_check = "CAPTURE_COUNTS_USED"
    else:
        failure_reason = "authoritative_capture_report_missing_ev_readiness_counts"

    summary.update(
        {
            "ev_summary_source": "authoritative_persist_capture_report",
            "ev_readiness_counts": counts,
            "ev_ready_count": counts.get("EV_READY", 0),
            "ev_not_ready_count": counts.get("EV_NOT_READY", 0),
            "priced_ev_runner_count": priced_ev_runner_count,
            "odds_exclusion_counts": odds_exclusion_counts,
            "odds_exclusion_count": sum(odds_exclusion_counts.values()),
            "ev_summary_consistency_check": consistency_check,
            "ev_summary_failure_reason": failure_reason,
        }
    )
    return summary


def _count_text_lines(path: Path) -> int | None:
    rooted = _rooted_path(path)
    if not rooted.exists():
        return None
    with rooted.open("r", encoding="utf-8") as handle:
        return sum(1 for _line in handle)


def _protected_resource_counters(
    *,
    snapshot_dir: Path,
    date_text: str,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    rooted_snapshot_dir = _rooted_path(snapshot_dir)
    manifest = _rooted_path(manifest_path or rooted_snapshot_dir / "manifest.jsonl")
    date_dir = rooted_snapshot_dir / date_text
    target_date_snapshot_json_count = (
        sum(1 for _path in date_dir.rglob("*.json"))
        if date_dir.exists()
        else 0
    )
    return {
        "schema_version": "protected_resource_counters_v1",
        "snapshot_dir": _rel(rooted_snapshot_dir),
        "target_date": date_text,
        "manifest_path": _rel(manifest),
        "manifest_line_count": _count_text_lines(manifest),
        "target_date_snapshot_json_count": target_date_snapshot_json_count,
    }


def _counter_delta(before: Any, after: Any) -> int | None:
    if isinstance(before, int) and isinstance(after, int):
        return after - before
    return None


def _execution_step_succeeded(
    execution_results: list[dict[str, Any]],
    step_name: str,
) -> bool:
    for result in execution_results:
        if result.get("name") != step_name:
            continue
        return result.get("returncode") == 0 and result.get("status") not in {
            "FAILED_REPORT_FRESHNESS",
            "SKIPPED",
        }
    return False


def _protected_resource_delta_report(
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    approvals: dict[str, bool],
    execution_results: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest_delta = _counter_delta(
        before.get("manifest_line_count"),
        after.get("manifest_line_count"),
    )
    snapshot_delta = _counter_delta(
        before.get("target_date_snapshot_json_count"),
        after.get("target_date_snapshot_json_count"),
    )
    changed = bool(
        (manifest_delta is not None and manifest_delta != 0)
        or (snapshot_delta is not None and snapshot_delta != 0)
    )
    persist_step_succeeded = _execution_step_succeeded(
        execution_results,
        "approved_persist_ready_subset",
    )
    live_persist_approved = approvals.get("live_persist") is True
    if changed and not live_persist_approved:
        status = "UNAPPROVED_PROTECTED_RESOURCE_CHANGE"
        reason = "snapshot manifest or target-date snapshot count changed without APPROVE_LIVE_PERSIST"
    elif changed and persist_step_succeeded:
        status = "CHANGED_AFTER_APPROVED_PERSIST"
        reason = None
    elif changed:
        status = "PROTECTED_RESOURCE_CHANGE_WITHOUT_SUCCESSFUL_PERSIST_STEP"
        reason = "protected resources changed, but approved persist step did not complete successfully"
    elif live_persist_approved and persist_step_succeeded:
        status = "NO_PROTECTED_RESOURCE_CHANGE_AFTER_APPROVED_PERSIST"
        reason = "approved persist step completed but protected counters did not change"
    elif live_persist_approved:
        status = "UNCHANGED_APPROVAL_PRESENT"
        reason = "APPROVE_LIVE_PERSIST was present, but no successful persist step changed protected counters"
    else:
        status = "UNCHANGED_NO_APPROVAL"
        reason = None
    return {
        "schema_version": "protected_resource_delta_v1",
        "status": status,
        "reason": reason,
        "protected_write_gate": "APPROVE_LIVE_PERSIST",
        "live_persist_approved": live_persist_approved,
        "persist_step_succeeded": persist_step_succeeded,
        "changed": changed,
        "snapshot_dir": after.get("snapshot_dir") or before.get("snapshot_dir"),
        "target_date": after.get("target_date") or before.get("target_date"),
        "manifest_path": after.get("manifest_path") or before.get("manifest_path"),
        "manifest_line_count_before": before.get("manifest_line_count"),
        "manifest_line_count_after": after.get("manifest_line_count"),
        "manifest_line_count_delta": manifest_delta,
        "target_date_snapshot_json_count_before": before.get(
            "target_date_snapshot_json_count"
        ),
        "target_date_snapshot_json_count_after": after.get(
            "target_date_snapshot_json_count"
        ),
        "target_date_snapshot_json_count_delta": snapshot_delta,
    }


def _persist_approval_packet(
    *,
    persist_readiness_gate: dict[str, Any],
    protected_resource_counters: dict[str, Any],
    approvals: dict[str, bool],
    approval_details: dict[str, dict[str, Any]],
    persist_command: list[str],
    same_run_execute_ready_command: list[str] | None = None,
) -> dict[str, Any]:
    live_persist_details = approval_details.get("live_persist") or {}
    required_cli_flag = (
        live_persist_details.get("cli_flag")
        or APPROVAL_GATES["live_persist"]["cli_flag"]
    )
    ready_count = int(persist_readiness_gate.get("ready_count") or 0)
    not_ready_count = int(persist_readiness_gate.get("not_ready_count") or 0)
    clean_ready_subset = (
        persist_readiness_gate.get("clean_for_ready_subset_persist") is True
    )
    fresh = persist_readiness_gate.get("fresh_for_plan") is True
    approved = approvals.get("live_persist") is True
    hard_stops: list[str] = []
    if not clean_ready_subset:
        hard_stops.append("persist_readiness_gate_not_clean")
    if not fresh:
        hard_stops.append("dry_run_capture_report_not_fresh")
    if ready_count <= 0:
        hard_stops.append("ready_count_zero")

    if hard_stops:
        status = "NOT_READY"
    elif approved:
        status = "APPROVAL_PRESENT_READY_TO_EXECUTE_READY_SUBSET"
    else:
        status = "AWAITING_EXPLICIT_APPROVAL_READY_SUBSET"

    return {
        "schema_version": "persist_approval_packet_v1",
        "status": status,
        "can_execute_persist_now": bool(approved and not hard_stops),
        "approval_required": not approved,
        "approval_gate": "APPROVE_LIVE_PERSIST",
        "approval_sources": list(live_persist_details.get("sources") or []),
        "required_cli_flag": required_cli_flag,
        "required_env_var": live_persist_details.get("env_var")
        or APPROVAL_GATES["live_persist"]["env_var"],
        "hard_stops": hard_stops,
        "dry_run_report_path": persist_readiness_gate.get("path"),
        "dry_run_report_fresh_for_plan": fresh,
        "dry_run_report_age_seconds": persist_readiness_gate.get(
            "report_age_seconds"
        ),
        "dry_run_report_max_age_seconds": persist_readiness_gate.get(
            "max_report_age_seconds"
        ),
        **_report_expiry_fields(
            persist_readiness_gate,
            report_key="dry_run_report",
        ),
        "persist_readiness_status": persist_readiness_gate.get("status"),
        "persist_readiness_reason": persist_readiness_gate.get("reason"),
        "ready_count": ready_count,
        "not_ready_count": not_ready_count,
        "capture_count": int(persist_readiness_gate.get("capture_count") or 0),
        "candidate_files": persist_readiness_gate.get("candidate_files"),
        "ready_race_ids": persist_readiness_gate.get("ready_race_ids") or [],
        "not_ready_race_ids": persist_readiness_gate.get("not_ready_race_ids") or [],
        "ready_subset_only": True,
        "planned_persist_command": persist_command,
        **_approval_command_template_fields(
            command=persist_command,
            flag=required_cli_flag,
            hard_stops=hard_stops,
            template_key="approved_persist_command_template",
            insert_before="--output",
        ),
        **_same_run_execute_ready_command_template_fields(
            command=same_run_execute_ready_command,
            rechecks=[
                "fresh_refresh_current_window",
                "validate_current_upcoming_contract",
                "dry_run_prejump_capture",
                "persist_readiness_gate",
                "protected_resource_delta",
            ],
        ),
        "protected_resource_counters_before": protected_resource_counters,
        "expected_protected_delta_upper_bounds": {
            "manifest_line_count_delta_max": ready_count,
            "target_date_snapshot_json_count_delta_max": ready_count,
        },
        "post_execute_delta_report": "post_execution_protected_resource_delta",
        "no_result_labels": True,
        "no_live_odds_capture": True,
        "no_model_retrain_or_promotion": True,
    }


def _live_odds_approval_packet(
    *,
    persist_readiness_gate: dict[str, Any],
    approvals: dict[str, bool],
    approval_details: dict[str, dict[str, Any]],
    odds_command: list[str],
    odds_report_path: Path,
    same_run_execute_ready_command: list[str] | None = None,
    ev_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    live_odds_details = approval_details.get("live_odds_capture") or {}
    required_cli_flag = (
        live_odds_details.get("cli_flag")
        or APPROVAL_GATES["live_odds_capture"]["cli_flag"]
    )
    ready_count = int(persist_readiness_gate.get("ready_count") or 0)
    clean_ready_subset = (
        persist_readiness_gate.get("clean_for_ready_subset_persist") is True
    )
    fresh = persist_readiness_gate.get("fresh_for_plan") is True
    approved = approvals.get("live_odds_capture") is True
    hard_stops: list[str] = []
    if not clean_ready_subset:
        hard_stops.append("persist_readiness_gate_not_clean")
    if not fresh:
        hard_stops.append("dry_run_capture_report_not_fresh")
    if ready_count <= 0:
        hard_stops.append("ready_count_zero")

    if hard_stops:
        status = "NOT_READY"
    elif approved:
        status = "APPROVAL_PRESENT_READY_TO_CAPTURE_LIVE_ODDS"
    else:
        status = "AWAITING_EXPLICIT_APPROVAL_READY_FOR_LIVE_ODDS"

    summary = ev_summary if isinstance(ev_summary, dict) else None
    if summary is None:
        summary = _ev_summary_from_persist_readiness_gate(persist_readiness_gate)
    ev_readiness_counts = _count_mapping(summary.get("ev_readiness_counts"))

    return {
        "schema_version": "live_odds_approval_packet_v1",
        "status": status,
        "can_capture_live_odds_now": bool(approved and not hard_stops),
        "approval_required": not approved,
        "approval_gate": "APPROVE_LIVE_ODDS_CAPTURE",
        "approval_sources": list(live_odds_details.get("sources") or []),
        "required_cli_flag": required_cli_flag,
        "required_env_var": live_odds_details.get("env_var")
        or APPROVAL_GATES["live_odds_capture"]["env_var"],
        "hard_stops": hard_stops,
        "dry_run_report_path": persist_readiness_gate.get("path"),
        "dry_run_report_fresh_for_plan": fresh,
        "dry_run_report_age_seconds": persist_readiness_gate.get(
            "report_age_seconds"
        ),
        "dry_run_report_max_age_seconds": persist_readiness_gate.get(
            "max_report_age_seconds"
        ),
        **_report_expiry_fields(
            persist_readiness_gate,
            report_key="dry_run_report",
        ),
        "ready_count": ready_count,
        "ready_race_ids": persist_readiness_gate.get("ready_race_ids") or [],
        "current_ev_readiness_counts": ev_readiness_counts,
        "ev_summary_source": summary.get("ev_summary_source"),
        "ev_ready_count": _int_count(summary.get("ev_ready_count")),
        "ev_not_ready_count": _int_count(summary.get("ev_not_ready_count")),
        "priced_ev_runner_count": _int_count(summary.get("priced_ev_runner_count")),
        "odds_exclusion_count": _int_count(summary.get("odds_exclusion_count")),
        "odds_exclusion_counts": _count_mapping(summary.get("odds_exclusion_counts")),
        "authoritative_capture_report_path": summary.get(
            "authoritative_capture_report_path"
        ),
        "ev_summary_consistency_check": summary.get("ev_summary_consistency_check"),
        "ev_summary_failure_reason": summary.get("ev_summary_failure_reason"),
        "odds_capture_report_path": _rel(_rooted_path(odds_report_path)),
        "planned_odds_command": odds_command,
        **_approval_command_template_fields(
            command=odds_command,
            flag=required_cli_flag,
            hard_stops=hard_stops,
            template_key="approved_odds_command_template",
            insert_before="--output",
        ),
        **_same_run_execute_ready_command_template_fields(
            command=same_run_execute_ready_command,
            rechecks=[
                "fresh_refresh_current_window",
                "validate_current_upcoming_contract",
                "dry_run_prejump_capture",
                "persist_readiness_gate",
                "live_odds_readiness_gate",
            ],
        ),
        "write_scope": "append_only_live_odds_rows",
        "same_run_dry_run_required": True,
        "no_snapshot_persist": True,
        "no_result_labels": True,
        "no_model_retrain_or_promotion": True,
        "ev_policy": {
            "ev_must_remain_null_unless_all_requirements_pass": True,
            "requirements": [
                "dog_level_win_odds",
                "source_url_backed",
                "timestamped",
                "captured_before_prediction",
                "captured_before_feature_freeze",
                "captured_before_jump_when_jump_time_available",
                "runner_aligned",
                "trusted_source",
            ],
        },
    }


def _result_label_approval_packet(
    *,
    current_corpus: dict[str, Any],
    result_dry_run_gate: dict[str, Any],
    label_write_readiness_gate: dict[str, Any],
    approvals: dict[str, bool],
    approval_details: dict[str, dict[str, Any]],
    label_write_command: list[str],
    label_write_readiness_command: list[str],
    label_write_preflight_command: list[str] | None = None,
    label_write_preflight_gate: dict[str, Any] | None = None,
    same_run_execute_ready_command: list[str] | None = None,
) -> dict[str, Any]:
    label_details = approval_details.get("result_label_write") or {}
    required_cli_flag = (
        label_details.get("cli_flag")
        or APPROVAL_GATES["result_label_write"]["cli_flag"]
    )
    persisted_count = int(
        current_corpus.get("ready_persisted_prediction_snapshot_count_for_date")
        or 0
    )
    approved = approvals.get("result_label_write") is True
    hard_stops: list[str] = []
    if persisted_count <= 0:
        hard_stops.append("persisted_prejump_corpus_missing")
    jump_status = current_corpus.get("persisted_snapshot_jump_status") or {}
    if jump_status.get("wait_for_known_future_jumps_before_result_dry_run") is True:
        hard_stops.append("persisted_prejump_races_not_jumped_yet")
    if result_dry_run_gate.get("status") == "DATA_MISSING":
        hard_stops.append("result_dry_run_report_missing")
    elif result_dry_run_gate.get("clean") is not True:
        hard_stops.append("result_dry_run_report_not_clean")
    if (
        result_dry_run_gate.get("status") != "DATA_MISSING"
        and result_dry_run_gate.get("fresh_for_plan") is not True
    ):
        hard_stops.append("result_dry_run_report_not_fresh")
    if not hard_stops and label_write_readiness_gate.get("status") != "READY":
        hard_stops.append("label_write_readiness_validation_not_ready")
    preflight_gate = label_write_preflight_gate or {}

    if hard_stops:
        status = "NOT_READY"
    elif approved:
        status = "APPROVAL_PRESENT_READY_TO_WRITE_OFFICIAL_LABELS"
    else:
        status = "AWAITING_EXPLICIT_APPROVAL_READY_FOR_LABEL_WRITE"

    return {
        "schema_version": "result_label_approval_packet_v1",
        "status": status,
        "can_write_labels_now": bool(approved and not hard_stops),
        "approval_required": not approved,
        "approval_gate": "APPROVE_RESULT_LABEL_WRITE",
        "approval_sources": list(label_details.get("sources") or []),
        "required_cli_flag": required_cli_flag,
        "required_env_var": label_details.get("env_var")
        or APPROVAL_GATES["result_label_write"]["env_var"],
        "hard_stops": hard_stops,
        "current_corpus_status": current_corpus.get("status"),
        "ready_persisted_prediction_snapshot_count_for_date": persisted_count,
        "persisted_snapshot_jump_status": jump_status,
        "known_future_not_jumped_snapshot_count_for_date": jump_status.get(
            "known_future_not_jumped_count"
        ),
        "result_dry_run_report_path": result_dry_run_gate.get("path"),
        "result_dry_run_report_status": result_dry_run_gate.get("status"),
        "result_dry_run_report_reason": result_dry_run_gate.get("reason"),
        "result_dry_run_report_clean": result_dry_run_gate.get("clean") is True,
        "result_dry_run_report_fresh_for_plan": result_dry_run_gate.get(
            "fresh_for_plan"
        )
        is True,
        "result_dry_run_report_age_seconds": result_dry_run_gate.get(
            "report_age_seconds"
        ),
        "result_dry_run_report_max_age_seconds": result_dry_run_gate.get(
            "max_report_age_seconds"
        ),
        **_report_expiry_fields(
            result_dry_run_gate,
            report_key="result_dry_run_report",
        ),
        "result_dry_run_status": result_dry_run_gate.get("status"),
        "result_dry_run_reason": result_dry_run_gate.get("reason"),
        "result_dry_run_clean": result_dry_run_gate.get("clean") is True,
        "result_dry_run_fresh_for_plan": result_dry_run_gate.get("fresh_for_plan")
        is True,
        "candidate_count": result_dry_run_gate.get("candidate_count"),
        "failed_count": result_dry_run_gate.get("failed_count"),
        "expected_scope": result_dry_run_gate.get("expected_scope"),
        "observed_scope": result_dry_run_gate.get("observed_scope"),
        "label_write_readiness_validation_report_path": label_write_readiness_gate.get(
            "path"
        ),
        "label_write_readiness_validation_status": label_write_readiness_gate.get(
            "status"
        ),
        "label_write_readiness_validation_reason": label_write_readiness_gate.get(
            "reason"
        ),
        "label_write_readiness_validation_fresh_for_plan": (
            label_write_readiness_gate.get("fresh_for_plan") is True
        ),
        "planned_label_write_readiness_validation_command": (
            label_write_readiness_command
        ),
        "label_write_preflight_packet_report_path": preflight_gate.get("path"),
        "label_write_preflight_packet_status": preflight_gate.get("status"),
        "label_write_preflight_packet_reason": preflight_gate.get("reason"),
        "label_write_preflight_packet_fresh_for_plan": (
            preflight_gate.get("fresh_for_plan") is True
        ),
        "planned_label_write_preflight_packet_command": (
            label_write_preflight_command
        ),
        "planned_label_write_command": label_write_command,
        **_approval_command_template_fields(
            command=label_write_command,
            flag=required_cli_flag,
            hard_stops=hard_stops,
            template_key="approved_label_write_command_template",
        ),
        **_same_run_execute_ready_command_template_fields(
            command=same_run_execute_ready_command,
            rechecks=[
                "current_persisted_prejump_corpus",
                "official_result_ingest_dry_run",
                "result_label_approval_gate",
                "official_first_scope_match",
            ],
        ),
        "write_scope": "official_result_label_rows_with_pre_write_backup",
        "same_run_result_dry_run_required": True,
        "official_first_policy": {
            "require_ready_prejump_snapshot": True,
            "dry_run_must_be_clean_and_fresh": True,
            "participant_alignment_required": True,
            "official_or_complete_result_required": True,
            "winner_only_or_partial_results_not_label_ready": True,
        },
        "no_snapshot_persist": True,
        "no_live_odds_capture": True,
        "no_model_retrain_or_promotion": True,
    }


def _clean_result_dry_run_report_for_scope(
    *,
    report_path: Path,
    expected_scope: dict[str, Any],
    max_age_seconds: int | None = RESULT_DRY_RUN_REPORT_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    path = report_path if report_path.is_absolute() else ROOT / report_path
    gate: dict[str, Any] = {
        "path": _rel(path),
        "clean": False,
        "fresh_for_plan": False,
        "status": "DATA_MISSING",
        "reason": "result_dry_run_report_missing",
        "report_mtime_utc": None,
        "report_age_seconds": None,
        "max_report_age_seconds": max_age_seconds,
        "expected_scope": expected_scope,
        "observed_scope": None,
        "candidate_count": None,
        "failed_count": None,
    }
    if not path.exists():
        return gate
    stat = path.stat()
    report_mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
    report_age_seconds = max(
        0.0,
        (datetime.now(timezone.utc) - report_mtime).total_seconds(),
    )
    gate.update(
        {
            "report_mtime_utc": report_mtime.isoformat(timespec="seconds"),
            "report_age_seconds": round(report_age_seconds, 3),
            "fresh_for_plan": (
                max_age_seconds is None or report_age_seconds <= max_age_seconds
            ),
        }
    )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        gate["reason"] = f"result_dry_run_report_unreadable:{type(exc).__name__}"
        return gate

    observed_scope = report.get("scope") if isinstance(report, dict) else None
    failures: list[str] = []
    if gate["fresh_for_plan"] is not True:
        failures.append("result_dry_run_report_stale")
    if not isinstance(report, dict):
        failures.append("report_root_not_object")
    else:
        if report.get("schema_version") != "official_result_ingest_report_v1":
            failures.append("schema_version_mismatch")
        if report.get("dry_run") is not True:
            failures.append("report_is_not_dry_run")
        if report.get("status") != "SUCCESS":
            failures.append("report_status_not_success")
        if report.get("clean_for_label_write") is not True:
            failures.append("report_not_clean_for_label_write")
        if observed_scope != expected_scope:
            failures.append("report_scope_mismatch")
        if int(report.get("failed_count") or 0) != 0:
            failures.append("failed_count_nonzero")
        if int(report.get("candidate_count") or 0) <= 0:
            failures.append("candidate_count_zero")
        if int(report.get("ingested_count") or 0) != int(report.get("candidate_count") or 0):
            failures.append("ingested_count_mismatch")

    gate.update(
        {
            "expected_scope": expected_scope,
            "observed_scope": observed_scope,
            "candidate_count": report.get("candidate_count") if isinstance(report, dict) else None,
            "failed_count": report.get("failed_count") if isinstance(report, dict) else None,
        }
    )
    if failures:
        gate["status"] = "NOT_READY"
        gate["reason"] = ",".join(failures)
        return gate
    gate["clean"] = True
    gate["fresh_for_plan"] = True
    gate["status"] = "READY"
    gate["reason"] = None
    return gate


def _clean_result_dry_run_report(
    *,
    report_path: Path,
    date_text: str,
    db_path: Path,
    upcoming_dir: Path,
    snapshot_dir: Path,
    race_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    expected_scope = {
        "db_path": _normalise_path(db_path),
        "date": date_text,
        "upcoming_dir": _normalise_path(upcoming_dir),
        "snapshot_dir": _normalise_path(snapshot_dir),
        "race_ids": _race_id_scope(race_ids),
        "require_ready_snapshot": True,
    }
    return _clean_result_dry_run_report_for_scope(
        report_path=report_path,
        expected_scope=expected_scope,
    )


def _label_write_readiness_validation_gate(
    *,
    report_path: Path,
    expected_scope: dict[str, Any],
    approved_dry_run_report: Path,
    max_age_seconds: int | None = RESULT_DRY_RUN_REPORT_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    path = _rooted_path(report_path)
    gate: dict[str, Any] = {
        "path": _rel(path),
        "status": "DATA_MISSING",
        "reason": "label_write_readiness_validation_missing",
        "fresh_for_plan": False,
        "report_mtime_utc": None,
        "report_age_seconds": None,
        "max_report_age_seconds": max_age_seconds,
        "expected_scope": expected_scope,
        "observed_scope": None,
        "approved_dry_run_report": _rel(_rooted_path(approved_dry_run_report)),
        "candidate_count_loaded_for_write_scope": None,
        "write_performed": None,
    }
    if not path.exists():
        return gate
    stat = path.stat()
    report_mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
    report_age_seconds = max(
        0.0,
        (datetime.now(timezone.utc) - report_mtime).total_seconds(),
    )
    gate.update(
        {
            "report_mtime_utc": report_mtime.isoformat(timespec="seconds"),
            "report_age_seconds": round(report_age_seconds, 3),
            "fresh_for_plan": (
                max_age_seconds is None or report_age_seconds <= max_age_seconds
            ),
        }
    )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        gate["reason"] = f"label_write_readiness_unreadable:{type(exc).__name__}"
        return gate

    dry_run_gate = (
        report.get("dry_run_report_gate") if isinstance(report, dict) else None
    )
    dry_run_gate = dry_run_gate if isinstance(dry_run_gate, dict) else {}
    failures: list[str] = []
    if gate["fresh_for_plan"] is not True:
        failures.append("label_write_readiness_report_stale")
    if not isinstance(report, dict):
        failures.append("report_root_not_object")
        report = {}
    else:
        if report.get("schema_version") != "result_label_write_readiness_validation_v1":
            failures.append("schema_version_mismatch")
        if report.get("status") != "READY_FOR_EXPLICIT_APPROVAL":
            failures.append("readiness_status_not_ready")
        if report.get("scope") != expected_scope:
            failures.append("readiness_scope_mismatch")
        if dry_run_gate.get("approved") is not True:
            failures.append("dry_run_report_gate_not_approved")
        resolved_dry_run_report = dry_run_gate.get("resolved_report_path")
        if resolved_dry_run_report and _normalise_path(
            Path(str(resolved_dry_run_report))
        ) != _normalise_path(approved_dry_run_report):
            failures.append("dry_run_report_path_mismatch")
        if report.get("approval_required") is not True:
            failures.append("approval_required_not_true")
        if report.get("required_cli_flag") != "--write-labels-approved":
            failures.append("required_cli_flag_mismatch")
        if report.get("required_env_var") != "APPROVE_RESULT_LABEL_WRITE":
            failures.append("required_env_var_mismatch")
        if report.get("write_performed") is not False:
            failures.append("readiness_report_performed_write")
        if int(report.get("candidate_count_loaded_for_write_scope") or 0) <= 0:
            failures.append("candidate_count_zero")

    gate.update(
        {
            "observed_scope": report.get("scope") if isinstance(report, dict) else None,
            "dry_run_report_gate": dry_run_gate,
            "candidate_count_loaded_for_write_scope": report.get(
                "candidate_count_loaded_for_write_scope"
            )
            if isinstance(report, dict)
            else None,
            "candidate_race_ids_loaded_for_write_scope": report.get(
                "candidate_race_ids_loaded_for_write_scope"
            )
            if isinstance(report, dict)
            else None,
            "planned_command_if_approved": report.get("planned_command_if_approved")
            if isinstance(report, dict)
            else None,
            "write_performed": report.get("write_performed")
            if isinstance(report, dict)
            else None,
        }
    )
    if failures:
        gate["status"] = "NOT_READY"
        gate["reason"] = ",".join(failures)
        return gate
    gate["status"] = "READY"
    gate["reason"] = None
    return gate


def _label_write_preflight_packet_gate(
    *,
    report_path: Path,
    expected_scope: dict[str, Any],
    label_readiness_report: Path,
    result_dry_run_report: Path,
    db_path: Path,
    max_age_seconds: int | None = RESULT_DRY_RUN_REPORT_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    path = _rooted_path(report_path)
    gate: dict[str, Any] = {
        "path": _rel(path),
        "status": "DATA_MISSING",
        "reason": "label_write_preflight_packet_missing",
        "fresh_for_plan": False,
        "report_mtime_utc": None,
        "report_age_seconds": None,
        "max_report_age_seconds": max_age_seconds,
        "expected_scope": expected_scope,
        "observed_scope": None,
        "label_readiness_report": _rel(_rooted_path(label_readiness_report)),
        "result_dry_run_report": _rel(_rooted_path(result_dry_run_report)),
        "db_path": _normalise_path(db_path),
        "write_performed": None,
        "approval_approved": None,
    }
    if not path.exists():
        return gate

    stat = path.stat()
    report_mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
    report_age_seconds = max(
        0.0,
        (datetime.now(timezone.utc) - report_mtime).total_seconds(),
    )
    gate.update(
        {
            "report_mtime_utc": report_mtime.isoformat(timespec="seconds"),
            "report_age_seconds": round(report_age_seconds, 3),
            "fresh_for_plan": (
                max_age_seconds is None or report_age_seconds <= max_age_seconds
            ),
        }
    )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        gate["reason"] = (
            f"label_write_preflight_packet_unreadable:{type(exc).__name__}"
        )
        return gate

    failures: list[str] = []
    if gate["fresh_for_plan"] is not True:
        failures.append("label_write_preflight_packet_stale")
    if not isinstance(report, dict):
        failures.append("report_root_not_object")
        report = {}
    else:
        if report.get("schema_version") != "label_write_preflight_packet_v1":
            failures.append("schema_version_mismatch")
        if report.get("status") != "READY_FOR_EXPLICIT_LABEL_WRITE_APPROVAL":
            failures.append("preflight_status_not_ready")
        if report.get("failures"):
            failures.append("preflight_report_contains_failures")
        race_scope = report.get("race_scope")
        if not isinstance(race_scope, dict):
            failures.append("race_scope_missing")
        else:
            observed_scope = {
                "db_path": race_scope.get("db_path"),
                "date": race_scope.get("date"),
                "upcoming_dir": race_scope.get("upcoming_dir"),
                "snapshot_dir": race_scope.get("snapshot_dir"),
                "race_ids": _race_id_scope(race_scope.get("race_ids") or []),
                "require_ready_snapshot": bool(
                    race_scope.get("require_ready_snapshot")
                ),
            }
            gate["observed_scope"] = observed_scope
            if observed_scope != expected_scope:
                failures.append("preflight_scope_mismatch")
        approval_gate = report.get("approval_gate")
        approval_gate = approval_gate if isinstance(approval_gate, dict) else {}
        if approval_gate.get("approved") is not False:
            failures.append("preflight_approval_already_present")
        if approval_gate.get("required_cli_flag") != "--write-labels-approved":
            failures.append("preflight_required_cli_flag_mismatch")
        if approval_gate.get("required_env_var") != "APPROVE_RESULT_LABEL_WRITE":
            failures.append("preflight_required_env_var_mismatch")
        writes = report.get("writes_performed")
        writes = writes if isinstance(writes, dict) else {}
        if any(value is not False for value in writes.values()):
            failures.append("preflight_report_performed_write")
        if report.get("no_write_preflight_only") is not True:
            failures.append("preflight_no_write_flag_missing")
        db_state = report.get("pre_write_db_state")
        db_state = db_state if isinstance(db_state, dict) else {}
        if db_state.get("quick_check") != "ok":
            failures.append("preflight_db_quick_check_not_ok")
        if db_state.get("result_free_before_write") is not True:
            failures.append("preflight_db_not_result_free")
        source_evidence = report.get("source_evidence")
        source_evidence = source_evidence if isinstance(source_evidence, dict) else {}
        if (
            _normalise_path(Path(str(source_evidence.get("label_readiness") or "")))
            != _normalise_path(label_readiness_report)
        ):
            failures.append("preflight_label_readiness_source_mismatch")
        if (
            _normalise_path(
                Path(str(source_evidence.get("result_dry_run_report") or ""))
            )
            != _normalise_path(result_dry_run_report)
        ):
            failures.append("preflight_result_dry_run_source_mismatch")
        if _normalise_path(Path(str(source_evidence.get("db") or ""))) != _normalise_path(
            db_path
        ):
            failures.append("preflight_db_source_mismatch")
        gate["write_performed"] = writes.get("result_label_write")
        gate["approval_approved"] = approval_gate.get("approved")

    if failures:
        gate["status"] = "NOT_READY"
        gate["reason"] = ",".join(failures)
        return gate
    gate["status"] = "READY"
    gate["reason"] = None
    return gate


def _count_jsonl_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _evaluation_report_gate(
    *,
    report_path: Path,
    dataset_path: Path | None,
    max_age_seconds: int | None = EVALUATION_REPORT_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    path = report_path if report_path.is_absolute() else ROOT / report_path
    dataset = None
    if dataset_path is not None:
        dataset = dataset_path if dataset_path.is_absolute() else ROOT / dataset_path
    gate: dict[str, Any] = {
        "path": _rel(path),
        "dataset_path": _rel(dataset) if dataset is not None else None,
        "status": "DATA_MISSING",
        "reason": "evaluation_report_missing",
        "dataset_ready": False,
        "clean_official_metrics_ready": False,
        "runner_rows_scored": 0,
        "evaluation_dataset_rows_written": 0,
        "clean_official_races_evaluated": 0,
        "snapshot_corpus_readiness_status": None,
        "model_quality_status": None,
        "retrain_gate_status": None,
        "promotion_gate_status": None,
        "fresh_for_plan": False,
        "report_mtime_utc": None,
        "report_age_seconds": None,
        "max_report_age_seconds": max_age_seconds,
    }
    if not path.exists():
        return gate
    stat = path.stat()
    report_mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
    report_age_seconds = max(
        0.0,
        (datetime.now(timezone.utc) - report_mtime).total_seconds(),
    )
    gate.update(
        {
            "report_mtime_utc": report_mtime.isoformat(timespec="seconds"),
            "report_age_seconds": round(report_age_seconds, 3),
            "fresh_for_plan": (
                max_age_seconds is None or report_age_seconds <= max_age_seconds
            ),
        }
    )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        gate["reason"] = f"evaluation_report_unreadable:{type(exc).__name__}"
        return gate

    failures: list[str] = []
    warnings: list[str] = []
    if gate["fresh_for_plan"] is not True:
        failures.append("evaluation_report_stale")
    if not isinstance(report, dict):
        failures.append("report_root_not_object")
        report = {}
    elif report.get("status") != "SUCCESS":
        failures.append("report_status_not_success")

    runner_rows_scored = int(report.get("runner_rows_scored") or 0)
    dataset_rows_written = int(report.get("evaluation_dataset_rows_written") or 0)
    observed_dataset_output = report.get("evaluation_dataset_output")
    corpus_readiness = report.get("snapshot_corpus_readiness")
    corpus_readiness_status = (
        corpus_readiness.get("status")
        if isinstance(corpus_readiness, dict)
        else None
    )
    clean_eval = report.get("clean_official_evaluation")
    clean_metrics_by_arm = (
        clean_eval.get("metrics_by_arm")
        if isinstance(clean_eval, dict)
        else None
    )
    clean_model_metrics = (
        clean_metrics_by_arm.get("model_only")
        if isinstance(clean_metrics_by_arm, dict)
        and isinstance(clean_metrics_by_arm.get("model_only"), dict)
        else {}
    )
    clean_official_races = (
        int(clean_eval.get("races_evaluated") or 0)
        if isinstance(clean_eval, dict)
        else 0
    )
    model_quality = report.get("model_quality_diagnosis")
    retrain_gate = (
        model_quality.get("retrain_gate")
        if isinstance(model_quality, dict)
        else {}
    )
    promotion_gate = (
        model_quality.get("promotion_gate")
        if isinstance(model_quality, dict)
        else {}
    )

    if runner_rows_scored <= 0:
        failures.append("runner_rows_scored_zero")
    if dataset is not None:
        if not dataset.exists():
            failures.append("evaluation_dataset_output_missing")
        line_count = _count_jsonl_rows(dataset)
        if line_count is not None and line_count != dataset_rows_written:
            failures.append("evaluation_dataset_row_count_mismatch")
        if dataset_rows_written <= 0:
            failures.append("evaluation_dataset_rows_written_zero")
        if observed_dataset_output is not None:
            observed_dataset = Path(str(observed_dataset_output))
            if not observed_dataset.is_absolute():
                observed_dataset = ROOT / observed_dataset
            if observed_dataset.resolve() != dataset.resolve():
                failures.append("evaluation_dataset_output_scope_mismatch")
    if observed_dataset_output is None:
        failures.append("evaluation_dataset_output_not_recorded")
    if corpus_readiness_status != "READY":
        warnings.append("snapshot_corpus_readiness_not_ready")
    expected_metrics = ("top1", "top3", "brier", "log_loss", "calibration")
    missing_metrics = [
        metric
        for metric in expected_metrics
        if clean_model_metrics.get(metric) is None
    ]
    if clean_official_races <= 0:
        failures.append("clean_official_races_evaluated_zero")
    if missing_metrics:
        failures.append("clean_official_metrics_missing:" + ",".join(missing_metrics))
    if not isinstance(model_quality, dict) or model_quality.get("status") != "SUCCESS":
        failures.append("model_quality_diagnosis_not_success")
    if retrain_gate.get("action_taken") not in (None, "none"):
        failures.append("retrain_gate_action_taken")
    if promotion_gate.get("action_taken") not in (None, "none"):
        failures.append("promotion_gate_action_taken")

    gate.update(
        {
            "runner_rows_scored": runner_rows_scored,
            "evaluation_dataset_rows_written": dataset_rows_written,
            "observed_dataset_output": observed_dataset_output,
            "clean_official_races_evaluated": clean_official_races,
            "snapshot_corpus_readiness_status": corpus_readiness_status,
            "model_quality_status": model_quality.get("status")
            if isinstance(model_quality, dict)
            else None,
            "retrain_gate_status": retrain_gate.get("status")
            if isinstance(retrain_gate, dict)
            else None,
            "promotion_gate_status": promotion_gate.get("status")
            if isinstance(promotion_gate, dict)
            else None,
            "missing_clean_official_metrics": missing_metrics,
            "warnings": warnings,
        }
    )
    if failures:
        gate["status"] = "NOT_READY"
        gate["reason"] = ",".join(failures)
        return gate

    gate["dataset_ready"] = True
    gate["clean_official_metrics_ready"] = True
    gate["fresh_for_plan"] = True
    gate["status"] = "PARTIAL_READY" if warnings else "READY"
    gate["reason"] = ",".join(warnings) if warnings else None
    return gate


def _model_review_packet_gate(
    *,
    packet_path: Path,
    evaluation_report_path: Path,
    dataset_path: Path,
    challenger_review_path: Path | None = None,
    max_age_seconds: int | None = MODEL_REVIEW_PACKET_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    path = _rooted_path(packet_path)
    report_path = _rooted_path(evaluation_report_path)
    dataset = _rooted_path(dataset_path)
    challenger_review = (
        _rooted_path(challenger_review_path) if challenger_review_path else None
    )
    gate: dict[str, Any] = {
        "path": _rel(path),
        "evaluation_report_path": _rel(report_path),
        "dataset_path": _rel(dataset),
        "challenger_review_path": _rel(challenger_review)
        if challenger_review
        else None,
        "status": "DATA_MISSING",
        "reason": "model_review_packet_missing",
        "packet_status": None,
        "packet_failures": [],
        "packet_warnings": [],
        "fresh_for_plan": False,
        "report_mtime_utc": None,
        "report_age_seconds": None,
        "max_report_age_seconds": max_age_seconds,
        "promotion_allowed": None,
        "registry_mutation_allowed": None,
        "promotion_action_taken": None,
        "required_promotion_gate": None,
        "challenger_review_gate_status": None,
        "challenger_review_gate_failures": [],
        "challenger_review_candidate_arm": None,
        "challenger_review_stability_status": None,
        "challenger_review_promotion_allowed": None,
    }
    if not path.exists():
        return gate
    stat = path.stat()
    report_mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
    report_age_seconds = max(
        0.0,
        (datetime.now(timezone.utc) - report_mtime).total_seconds(),
    )
    gate.update(
        {
            "report_mtime_utc": report_mtime.isoformat(timespec="seconds"),
            "report_age_seconds": round(report_age_seconds, 3),
            "fresh_for_plan": (
                max_age_seconds is None or report_age_seconds <= max_age_seconds
            ),
        }
    )
    try:
        packet = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        gate["reason"] = f"model_review_packet_unreadable:{type(exc).__name__}"
        return gate

    failures: list[str] = []
    if gate["fresh_for_plan"] is not True:
        failures.append("model_review_packet_stale")
    if not isinstance(packet, dict):
        failures.append("packet_root_not_object")
        packet = {}
    if packet.get("schema_version") != "model_review_packet_v1":
        failures.append("schema_version_mismatch")

    packet_status = packet.get("status")
    packet_failures = packet.get("failures")
    packet_warnings = packet.get("warnings")
    source_evidence = packet.get("source_evidence")
    promotion_control = packet.get("promotion_control")
    next_review_steps = packet.get("next_review_steps")
    challenger_review_gate = packet.get("challenger_review_gate")
    packet_failures = packet_failures if isinstance(packet_failures, list) else []
    packet_warnings = packet_warnings if isinstance(packet_warnings, list) else []
    source_evidence = source_evidence if isinstance(source_evidence, dict) else {}
    promotion_control = promotion_control if isinstance(promotion_control, dict) else {}
    next_review_steps = (
        next_review_steps if isinstance(next_review_steps, list) else []
    )
    challenger_review_gate = (
        challenger_review_gate if isinstance(challenger_review_gate, dict) else {}
    )

    required_promotion_gate = None
    for step in next_review_steps:
        if isinstance(step, dict) and step.get("name") == "promotion":
            required_promotion_gate = step.get("required_gate")
            break

    observed_report = source_evidence.get("evaluation_report")
    observed_dataset = source_evidence.get("evaluation_dataset")
    if observed_report and _normalise_path(Path(str(observed_report))) != _normalise_path(report_path):
        failures.append("evaluation_report_scope_mismatch")
    if observed_dataset and _normalise_path(Path(str(observed_dataset))) != _normalise_path(dataset):
        failures.append("evaluation_dataset_scope_mismatch")
    if source_evidence.get("evaluation_dataset_rows_observed") != source_evidence.get(
        "evaluation_dataset_rows_written"
    ):
        failures.append("evaluation_dataset_row_count_mismatch")
    if packet_status != "READY_FOR_CHALLENGER_REVIEW":
        failures.append("model_review_packet_not_ready")
    if packet_failures:
        failures.append("model_review_packet_failures:" + ",".join(map(str, packet_failures)))
    if promotion_control.get("action_taken") not in (None, "none"):
        failures.append("promotion_action_already_taken")
    if promotion_control.get("registry_mutation_allowed") is not False:
        failures.append("registry_mutation_not_blocked")
    if promotion_control.get("promotion_allowed") is not False:
        failures.append("promotion_not_blocked")
    if required_promotion_gate != "APPROVE_MODEL_PROMOTION":
        failures.append("promotion_required_gate_missing")
    if challenger_review:
        observed_challenger_review = challenger_review_gate.get("path")
        if challenger_review_gate.get("provided") is not True:
            failures.append("challenger_review_gate_missing")
        if (
            observed_challenger_review
            and _normalise_path(Path(str(observed_challenger_review)))
            != _normalise_path(challenger_review)
        ):
            failures.append("challenger_review_scope_mismatch")
        if not observed_challenger_review:
            failures.append("challenger_review_scope_missing")
        if challenger_review_gate.get("status") != "READY":
            failures.append("challenger_review_gate_not_ready")
        if challenger_review_gate.get("stability_status") != "STABLE_REPORT_ONLY":
            failures.append("challenger_review_not_stable_report_only")
        if challenger_review_gate.get("candidate_arm") != "power_calibrated_baseline":
            failures.append("challenger_review_candidate_mismatch")
        if challenger_review_gate.get("failed_split_count") not in (0, "0"):
            failures.append("challenger_review_failed_splits")
        if challenger_review_gate.get("all_log_loss_improved") is not True:
            failures.append("challenger_review_log_loss_not_improved")
        if challenger_review_gate.get("all_brier_improved") is not True:
            failures.append("challenger_review_brier_not_improved")
        if challenger_review_gate.get("all_ranking_preserved") is not True:
            failures.append("challenger_review_ranking_not_preserved")
        if challenger_review_gate.get("promotion_allowed") is not False:
            failures.append("challenger_review_promotion_not_blocked")
        if challenger_review_gate.get("registry_mutation_allowed") is not False:
            failures.append("challenger_review_registry_mutation_not_blocked")
        if challenger_review_gate.get("model_artifact_written") is not False:
            failures.append("challenger_review_model_artifact_written")
        challenger_failures = challenger_review_gate.get("failures")
        if challenger_failures:
            failures.append(
                "challenger_review_gate_failures:"
                + ",".join(map(str, challenger_failures))
            )

    gate.update(
        {
            "packet_status": packet_status,
            "packet_failures": packet_failures,
            "packet_warnings": packet_warnings,
            "source_evidence": source_evidence,
            "promotion_allowed": promotion_control.get("promotion_allowed"),
            "registry_mutation_allowed": promotion_control.get(
                "registry_mutation_allowed"
            ),
            "promotion_action_taken": promotion_control.get("action_taken"),
            "required_promotion_gate": required_promotion_gate,
            "challenger_review_gate_status": challenger_review_gate.get("status"),
            "challenger_review_gate_failures": challenger_review_gate.get("failures")
            if isinstance(challenger_review_gate.get("failures"), list)
            else [],
            "challenger_review_candidate_arm": challenger_review_gate.get(
                "candidate_arm"
            ),
            "challenger_review_stability_status": challenger_review_gate.get(
                "stability_status"
            ),
            "challenger_review_promotion_allowed": challenger_review_gate.get(
                "promotion_allowed"
            ),
        }
    )
    if failures:
        gate["status"] = "NOT_READY"
        gate["reason"] = ",".join(failures)
        return gate
    gate["status"] = "READY"
    gate["reason"] = None
    return gate


def _snapshot_challenger_review_gate(
    *,
    report_path: Path,
    dataset_path: Path,
    max_age_seconds: int | None = MODEL_REVIEW_PACKET_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    path = _rooted_path(report_path)
    dataset = _rooted_path(dataset_path)
    gate: dict[str, Any] = {
        "path": _rel(path),
        "dataset_path": _rel(dataset),
        "status": "DATA_MISSING",
        "reason": "snapshot_challenger_review_missing",
        "fresh_for_plan": False,
        "report_mtime_utc": None,
        "report_age_seconds": None,
        "max_report_age_seconds": max_age_seconds,
        "stability_status": None,
        "candidate_arm": None,
        "promotion_allowed": None,
        "registry_mutation_allowed": None,
        "model_artifact_written": None,
    }
    if not path.exists():
        return gate
    stat = path.stat()
    report_mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
    report_age_seconds = max(
        0.0,
        (datetime.now(timezone.utc) - report_mtime).total_seconds(),
    )
    gate.update(
        {
            "report_mtime_utc": report_mtime.isoformat(timespec="seconds"),
            "report_age_seconds": round(report_age_seconds, 3),
            "fresh_for_plan": (
                max_age_seconds is None or report_age_seconds <= max_age_seconds
            ),
        }
    )
    try:
        review = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        gate["reason"] = f"snapshot_challenger_review_unreadable:{type(exc).__name__}"
        return gate

    failures: list[str] = []
    if gate["fresh_for_plan"] is not True:
        failures.append("snapshot_challenger_review_stale")
    if not isinstance(review, dict):
        failures.append("review_root_not_object")
        review = {}
    if review.get("schema_version") != "snapshot_challenger_review_v1":
        failures.append("schema_version_mismatch")
    if review.get("status") != "SUCCESS":
        failures.append("snapshot_challenger_review_not_success")
    if review.get("failures"):
        failures.append("snapshot_challenger_review_contains_failures")

    source_evidence = review.get("source_evidence")
    stability = review.get("stability_review")
    promotion_control = review.get("promotion_control")
    challenger_training = review.get("challenger_training")
    source_evidence = source_evidence if isinstance(source_evidence, dict) else {}
    stability = stability if isinstance(stability, dict) else {}
    promotion_control = (
        promotion_control if isinstance(promotion_control, dict) else {}
    )
    challenger_training = (
        challenger_training if isinstance(challenger_training, dict) else {}
    )
    observed_dataset = source_evidence.get("evaluation_dataset")
    if (
        observed_dataset
        and _normalise_path(Path(str(observed_dataset))) != _normalise_path(dataset)
    ):
        failures.append("evaluation_dataset_scope_mismatch")
    if not observed_dataset:
        failures.append("evaluation_dataset_scope_missing")
    if stability.get("status") != "STABLE_REPORT_ONLY":
        failures.append("snapshot_challenger_not_stable_report_only")
    if stability.get("candidate_arm") != "power_calibrated_baseline":
        failures.append("snapshot_challenger_candidate_mismatch")
    if stability.get("failed_split_count") not in (0, "0"):
        failures.append("snapshot_challenger_failed_splits")
    if stability.get("all_log_loss_improved") is not True:
        failures.append("snapshot_challenger_log_loss_not_improved")
    if stability.get("all_brier_improved") is not True:
        failures.append("snapshot_challenger_brier_not_improved")
    if stability.get("all_ranking_preserved") is not True:
        failures.append("snapshot_challenger_ranking_not_preserved")
    if promotion_control.get("promotion_allowed") is not False:
        failures.append("snapshot_challenger_promotion_not_blocked")
    if promotion_control.get("registry_mutation_allowed") is not False:
        failures.append("snapshot_challenger_registry_mutation_not_blocked")
    if promotion_control.get("model_artifact_written") is not False:
        failures.append("snapshot_challenger_model_artifact_written")
    if challenger_training.get("model_artifact_written") is not False:
        failures.append("snapshot_challenger_training_model_artifact_written")

    gate.update(
        {
            "source_evidence": source_evidence,
            "stability_status": stability.get("status"),
            "candidate_arm": stability.get("candidate_arm"),
            "split_count": stability.get("split_count"),
            "failed_split_count": stability.get("failed_split_count"),
            "all_log_loss_improved": stability.get("all_log_loss_improved"),
            "all_brier_improved": stability.get("all_brier_improved"),
            "all_ranking_preserved": stability.get("all_ranking_preserved"),
            "promotion_allowed": promotion_control.get("promotion_allowed"),
            "registry_mutation_allowed": promotion_control.get(
                "registry_mutation_allowed"
            ),
            "model_artifact_written": promotion_control.get(
                "model_artifact_written"
            ),
        }
    )
    if failures:
        gate["status"] = "NOT_READY"
        gate["reason"] = ",".join(failures)
        return gate
    gate["status"] = "READY"
    gate["reason"] = None
    return gate


def _calibration_design_gate(
    *,
    report_path: Path,
    model_review_packet_path: Path,
    max_age_seconds: int | None = MODEL_REVIEW_PACKET_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    path = _rooted_path(report_path)
    packet_path = _rooted_path(model_review_packet_path)
    gate: dict[str, Any] = {
        "path": _rel(path),
        "model_review_packet_path": _rel(packet_path),
        "status": "DATA_MISSING",
        "reason": "calibration_design_report_missing",
        "fresh_for_plan": False,
        "report_mtime_utc": None,
        "report_age_seconds": None,
        "max_report_age_seconds": max_age_seconds,
        "deployment_control": {},
        "runtime_transform_spec": {},
        "comparison_to_baseline": {},
    }
    if not path.exists():
        return gate
    stat = path.stat()
    report_mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
    report_age_seconds = max(
        0.0,
        (datetime.now(timezone.utc) - report_mtime).total_seconds(),
    )
    gate.update(
        {
            "report_mtime_utc": report_mtime.isoformat(timespec="seconds"),
            "report_age_seconds": round(report_age_seconds, 3),
            "fresh_for_plan": (
                max_age_seconds is None or report_age_seconds <= max_age_seconds
            ),
        }
    )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        gate["reason"] = f"calibration_design_unreadable:{type(exc).__name__}"
        return gate

    failures: list[str] = []
    if gate["fresh_for_plan"] is not True:
        failures.append("calibration_design_report_stale")
    if not isinstance(report, dict):
        failures.append("report_root_not_object")
        report = {}
    if report.get("schema_version") != "calibration_layer_design_v1":
        failures.append("schema_version_mismatch")
    if report.get("status") != "READY_FOR_OPERATOR_DESIGN_REVIEW":
        failures.append("calibration_design_not_ready")
    if report.get("failures"):
        failures.append("calibration_design_contains_failures")

    source_evidence = report.get("source_evidence")
    deployment_control = report.get("deployment_control")
    transform = report.get("runtime_transform_spec")
    comparison = report.get("comparison_to_baseline")
    source_evidence = source_evidence if isinstance(source_evidence, dict) else {}
    deployment_control = (
        deployment_control if isinstance(deployment_control, dict) else {}
    )
    transform = transform if isinstance(transform, dict) else {}
    comparison = comparison if isinstance(comparison, dict) else {}

    observed_packet = source_evidence.get("model_review_packet")
    if (
        observed_packet
        and _normalise_path(Path(str(observed_packet))) != _normalise_path(packet_path)
    ):
        failures.append("model_review_packet_scope_mismatch")
    if not observed_packet:
        failures.append("model_review_packet_scope_missing")
    if transform.get("candidate_arm") != "power_calibrated_baseline":
        failures.append("calibration_candidate_mismatch")
    if transform.get("algorithm") != "power_normalize_per_race":
        failures.append("calibration_algorithm_mismatch")
    if transform.get("rank_preserving_when_alpha_positive") is not True:
        failures.append("calibration_not_rank_preserving")
    if transform.get("uses_labels_at_runtime") is not False:
        failures.append("calibration_uses_labels_at_runtime")
    if transform.get("uses_odds_at_runtime") is not False:
        failures.append("calibration_uses_odds_at_runtime")
    if transform.get("requires_runner_complete_race_group") is not True:
        failures.append("calibration_missing_complete_group_requirement")
    if comparison.get("log_loss_improved") is not True:
        failures.append("calibration_log_loss_not_improved")
    if comparison.get("brier_improved") is not True:
        failures.append("calibration_brier_not_improved")
    for key in (
        "top1_preserved",
        "top2_preserved",
        "top3_preserved",
        "mean_winner_rank_preserved",
    ):
        if comparison.get(key) is not True:
            failures.append(f"calibration_{key}_not_true")
    if deployment_control.get("promotion_allowed") is not False:
        failures.append("calibration_promotion_not_blocked")
    if deployment_control.get("registry_mutation_allowed") is not False:
        failures.append("calibration_registry_mutation_not_blocked")
    if deployment_control.get("model_artifact_written") is not False:
        failures.append("calibration_model_artifact_written")
    if deployment_control.get("production_config_write_allowed") is not False:
        failures.append("calibration_production_config_write_allowed")
    if deployment_control.get("betting_allowed") is not False:
        failures.append("calibration_betting_allowed")
    if deployment_control.get("required_gate") != "APPROVE_MODEL_PROMOTION":
        failures.append("calibration_required_gate_missing")

    gate.update(
        {
            "deployment_control": deployment_control,
            "runtime_transform_spec": transform,
            "comparison_to_baseline": comparison,
            "source_evidence": source_evidence,
        }
    )
    if failures:
        gate["status"] = "NOT_READY"
        gate["reason"] = ",".join(failures)
        return gate
    gate["status"] = "READY"
    gate["reason"] = None
    return gate


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("json_root_not_object")
    return data


def _refresh_report_gate(report_path: Path) -> dict[str, Any]:
    path = _rooted_path(report_path)
    gate: dict[str, Any] = {
        "schema_version": "prejump_refresh_report_gate_v1",
        "path": _rel(path),
        "status": "DATA_MISSING",
        "reason": "refresh_report_missing",
        "total_races_found": 0,
        "selected_count": 0,
        "bucket_counts": {},
        "window": {},
        "next_preferred_window": None,
        "recommended_rerun_after_local": None,
    }
    if not path.exists():
        return gate
    try:
        report = _read_json_object(path)
    except Exception as exc:
        gate["status"] = "NOT_READY"
        gate["reason"] = f"refresh_report_unreadable:{type(exc).__name__}"
        return gate

    next_window = report.get("next_preferred_window")
    next_window = next_window if isinstance(next_window, dict) else None
    selected_count = int(report.get("selected_count") or 0)
    total_races_found = int(report.get("total_races_found") or 0)
    metadata_coverage = report.get("sidecar_metadata_coverage")
    metadata_coverage = metadata_coverage if isinstance(metadata_coverage, dict) else {}
    if report.get("status") != "SUCCESS":
        status = "NOT_READY"
        reason = (
            report.get("reason")
            or metadata_coverage.get("reason")
            or "refresh_report_status_not_success"
        )
    elif selected_count > 0:
        status = "SELECTED_RACES_READY"
        reason = None
    elif next_window and next_window.get("status") == "WAITING_FOR_FUTURE_WINDOW":
        status = "WAITING_FOR_FUTURE_WINDOW"
        reason = next_window.get("reason")
    elif next_window and next_window.get("status") == "DATA_MISSING":
        status = "DATA_MISSING"
        reason = next_window.get("reason") or "race_jump_times_missing"
    elif total_races_found <= 0:
        status = "NO_RACES_FOUND"
        reason = "no_races_available"
    else:
        status = "NO_SELECTED_RACES"
        reason = "no_races_inside_preferred_window"

    gate.update(
        {
            "status": status,
            "reason": reason,
            "generated_at": report.get("generated_at"),
            "total_races_found": total_races_found,
            "selected_count": selected_count,
            "bucket_counts": report.get("bucket_counts")
            if isinstance(report.get("bucket_counts"), dict)
            else {},
            "window": report.get("window")
            if isinstance(report.get("window"), dict)
            else {},
            "next_preferred_window": next_window,
            "metadata_collection_status": report.get("metadata_collection_status")
            or metadata_coverage.get("status"),
            "sidecar_metadata_coverage": metadata_coverage,
            "recommended_rerun_after_local": (
                next_window.get("recommended_rerun_after_local")
                if next_window
                else None
            ),
        }
    )
    return gate


def _promotion_model_review_packet_gate(
    packet_path: Path | None,
    *,
    min_clean_official_races: int = MIN_PROMOTION_CLEAN_OFFICIAL_RACES,
) -> dict[str, Any]:
    gate: dict[str, Any] = {
        "schema_version": "prejump_promotion_model_review_packet_gate_v1",
        "path": _rel(_rooted_path(packet_path)) if packet_path else None,
        "status": "NOT_REQUESTED",
        "reason": "promotion_model_review_packet_not_requested",
        "freshness_policy": "durable_historical_evidence_no_10_min_expiry",
        "source_evidence_type": "historical_or_report_only_promotion_evidence",
        "model_review_packet_gate": {},
        "clean_official_evaluated_races": 0,
        "clean_official_snapshot_instances": 0,
        "clean_official_runner_rows": 0,
        "min_clean_official_evaluated_races": min_clean_official_races,
        "historical_clean_official_races_can_satisfy_minimum": True,
        "current_day_races_required_for_minimum": False,
        "promotion_action_taken": None,
        "promotion_allowed": None,
        "registry_mutation_allowed": None,
        "challenger_review_gate_status": None,
        "challenger_review_stability_status": None,
    }
    if packet_path is None:
        return gate

    path = _rooted_path(packet_path)
    gate["path"] = _rel(path)
    if not path.exists():
        gate["status"] = "DATA_MISSING"
        gate["reason"] = "promotion_model_review_packet_missing"
        return gate

    try:
        packet = _read_json_object(path)
    except Exception as exc:
        gate["status"] = "DATA_MISSING"
        gate["reason"] = f"promotion_model_review_packet_unreadable:{type(exc).__name__}"
        return gate

    source_evidence = packet.get("source_evidence")
    source_evidence = source_evidence if isinstance(source_evidence, dict) else {}
    review_gate = packet.get("review_gate")
    review_gate = review_gate if isinstance(review_gate, dict) else {}
    promotion_control = packet.get("promotion_control")
    promotion_control = promotion_control if isinstance(promotion_control, dict) else {}
    challenger_review_gate = packet.get("challenger_review_gate")
    challenger_review_gate = (
        challenger_review_gate if isinstance(challenger_review_gate, dict) else {}
    )

    evaluation_report = source_evidence.get("evaluation_report")
    evaluation_dataset = source_evidence.get("evaluation_dataset")
    challenger_review = challenger_review_gate.get("path")
    clean_races = int(review_gate.get("clean_official_evaluated_races") or 0)
    clean_snapshots = int(review_gate.get("clean_official_snapshot_instances") or 0)
    clean_rows = int(review_gate.get("clean_official_runner_rows") or 0)

    failures: list[str] = []
    if not evaluation_report:
        failures.append("evaluation_report_scope_missing")
    elif not _rooted_path(Path(str(evaluation_report))).exists():
        failures.append("evaluation_report_source_missing")
    if not evaluation_dataset:
        failures.append("evaluation_dataset_scope_missing")
    elif not _rooted_path(Path(str(evaluation_dataset))).exists():
        failures.append("evaluation_dataset_source_missing")
    if not challenger_review:
        failures.append("challenger_review_scope_missing")
    elif not _rooted_path(Path(str(challenger_review))).exists():
        failures.append("challenger_review_source_missing")
    if clean_races < min_clean_official_races:
        failures.append("insufficient_clean_official_races")
    if clean_snapshots < min_clean_official_races:
        failures.append("insufficient_clean_snapshot_instances")
    if clean_rows <= 0:
        failures.append("clean_official_runner_rows_zero")

    model_gate = (
        _model_review_packet_gate(
            packet_path=path,
            evaluation_report_path=Path(str(evaluation_report)),
            dataset_path=Path(str(evaluation_dataset)),
            challenger_review_path=Path(str(challenger_review)),
            max_age_seconds=None,
        )
        if evaluation_report and evaluation_dataset and challenger_review
        else {}
    )
    if model_gate and model_gate.get("status") != "READY":
        failures.append("model_review_packet_gate_not_ready")

    gate.update(
        {
            "model_review_packet_gate": model_gate,
            "clean_official_evaluated_races": clean_races,
            "clean_official_snapshot_instances": clean_snapshots,
            "clean_official_runner_rows": clean_rows,
            "promotion_action_taken": promotion_control.get("action_taken"),
            "promotion_allowed": promotion_control.get("promotion_allowed"),
            "registry_mutation_allowed": promotion_control.get(
                "registry_mutation_allowed"
            ),
            "challenger_review_gate_status": challenger_review_gate.get("status"),
            "challenger_review_stability_status": challenger_review_gate.get(
                "stability_status"
            ),
        }
    )
    if failures:
        gate["status"] = "NOT_READY"
        gate["reason"] = ",".join(failures)
        gate["failures"] = failures
        return gate

    gate["status"] = "READY"
    gate["reason"] = None
    gate["failures"] = []
    return gate


def _promotion_calibration_design_gate(
    *,
    report_path: Path | None,
    model_review_packet_path: Path | None,
) -> dict[str, Any]:
    if report_path is None:
        return {
            "schema_version": "prejump_promotion_calibration_design_gate_v1",
            "path": None,
            "model_review_packet_path": (
                _rel(_rooted_path(model_review_packet_path))
                if model_review_packet_path
                else None
            ),
            "status": "NOT_REQUESTED",
            "reason": "promotion_calibration_design_not_requested",
            "freshness_policy": "durable_historical_evidence_no_10_min_expiry",
        }
    if model_review_packet_path is None:
        return {
            "schema_version": "prejump_promotion_calibration_design_gate_v1",
            "path": _rel(_rooted_path(report_path)),
            "model_review_packet_path": None,
            "status": "NOT_READY",
            "reason": "promotion_model_review_packet_missing_for_calibration_design",
            "freshness_policy": "durable_historical_evidence_no_10_min_expiry",
        }
    gate = _calibration_design_gate(
        report_path=report_path,
        model_review_packet_path=model_review_packet_path,
        max_age_seconds=None,
    )
    gate["schema_version"] = "prejump_promotion_calibration_design_gate_v1"
    gate["freshness_policy"] = "durable_historical_evidence_no_10_min_expiry"
    return gate


def _promotion_readiness_gate(
    *,
    approvals: dict[str, bool],
    evaluation_report_gate: dict[str, Any],
    snapshot_challenger_review_gate: dict[str, Any],
    model_review_packet_gate: dict[str, Any],
    calibration_design_gate: dict[str, Any],
    promotion_model_review_packet_gate: dict[str, Any] | None = None,
    promotion_calibration_design_gate: dict[str, Any] | None = None,
    min_clean_official_races: int = MIN_PROMOTION_CLEAN_OFFICIAL_RACES,
) -> dict[str, Any]:
    approval_present = approvals.get("promotion") is True
    clean_races = int(
        evaluation_report_gate.get("clean_official_races_evaluated") or 0
    )
    external_packet_gate = promotion_model_review_packet_gate or {
        "status": "NOT_REQUESTED"
    }
    external_calibration_gate = promotion_calibration_design_gate or {
        "status": "NOT_REQUESTED"
    }
    external_evidence_requested = (
        external_packet_gate.get("status") != "NOT_REQUESTED"
        or external_calibration_gate.get("status") != "NOT_REQUESTED"
    )
    evidence_clean_races = clean_races
    evidence_source = "current_run_report_only_evidence"
    if external_evidence_requested:
        evidence_source = "external_report_only_promotion_evidence"
        evidence_clean_races = int(
            external_packet_gate.get("clean_official_evaluated_races") or 0
        )
    blockers: list[str] = []
    blocker_details: list[dict[str, Any]] = []

    def add_blocker(name: str, **details: Any) -> None:
        blockers.append(name)
        blocker_details.append({"name": name, **details})

    if not approval_present:
        add_blocker("approval_missing", required_gate="APPROVE_MODEL_PROMOTION")
    eval_status = evaluation_report_gate.get("status")
    challenger_status = snapshot_challenger_review_gate.get("status")
    packet_status = model_review_packet_gate.get("status")
    calibration_status = calibration_design_gate.get("status")
    external_packet_status = external_packet_gate.get("status")
    external_calibration_status = external_calibration_gate.get("status")

    if external_evidence_requested:
        if external_packet_status != "READY":
            add_blocker(
                "promotion_model_review_packet_gate_not_ready",
                status=external_packet_status,
                reason=external_packet_gate.get("reason"),
            )
        if external_calibration_status != "READY":
            add_blocker(
                "promotion_calibration_design_gate_not_ready",
                status=external_calibration_status,
                reason=external_calibration_gate.get("reason"),
            )
    else:
        if clean_races < min_clean_official_races:
            add_blocker(
                "clean_official_evaluated_races_below_minimum",
                observed=clean_races,
                required=min_clean_official_races,
            )
        if eval_status not in {"READY", "PARTIAL_READY"}:
            add_blocker(
                "evaluation_report_gate_not_ready",
                status=eval_status,
                reason=evaluation_report_gate.get("reason"),
            )
        if challenger_status == "NOT_REQUESTED":
            add_blocker(
                "snapshot_challenger_review_required",
                status=challenger_status,
                reason=snapshot_challenger_review_gate.get("reason"),
            )
        elif challenger_status != "READY":
            add_blocker(
                "snapshot_challenger_review_gate_not_ready",
                status=challenger_status,
                reason=snapshot_challenger_review_gate.get("reason"),
            )
        if packet_status != "READY":
            add_blocker(
                "model_review_packet_gate_not_ready",
                status=packet_status,
                reason=model_review_packet_gate.get("reason"),
            )
        if calibration_status != "READY":
            add_blocker(
                "calibration_design_gate_not_ready",
                status=calibration_status,
                reason=calibration_design_gate.get("reason"),
            )

    if not approval_present:
        status = "APPROVAL_REQUIRED"
    elif blockers:
        status = "APPROVAL_PRESENT_EVIDENCE_NOT_READY"
    else:
        status = "APPROVAL_PRESENT_EVIDENCE_READY_REPORT_ONLY"

    return {
        "schema_version": "prejump_promotion_readiness_gate_v1",
        "status": status,
        "reason": ",".join(blockers) if blockers else None,
        "approval_present": approval_present,
        "required_gate": "APPROVE_MODEL_PROMOTION",
        "blockers": blockers,
        "blocker_details": blocker_details,
        "clean_official_evaluated_races": clean_races,
        "promotion_evidence_source": evidence_source,
        "promotion_evidence_clean_official_evaluated_races": evidence_clean_races,
        "min_clean_official_evaluated_races": min_clean_official_races,
        "historical_clean_official_races_can_satisfy_minimum": True,
        "current_day_races_required_for_minimum": False,
        "evaluation_report_gate_status": eval_status,
        "snapshot_challenger_review_gate_status": challenger_status,
        "snapshot_challenger_review_stability_status": (
            snapshot_challenger_review_gate.get("stability_status")
        ),
        "model_review_packet_gate_status": packet_status,
        "calibration_design_gate_status": calibration_status,
        "promotion_model_review_packet_gate_status": external_packet_status,
        "promotion_calibration_design_gate_status": external_calibration_status,
        "ready_for_separate_promotion_review": (
            approval_present and not blockers
        ),
        "promotion_action_taken": "none",
        "promotion_allowed_by_loop": False,
        "model_artifact_write_allowed_by_loop": False,
        "registry_mutation_allowed_by_loop": False,
        "betting_allowed_by_loop": False,
    }


def _report_pre_state(path: Path) -> dict[str, Any]:
    rooted = _rooted_path(path)
    exists = rooted.exists()
    stat = rooted.stat() if exists else None
    return {
        "path": _rel(rooted),
        "exists_before": exists,
        "mtime_ns_before": stat.st_mtime_ns if stat else None,
    }


def _report_freshness_record(pre_state: dict[str, Any]) -> dict[str, Any]:
    rooted = _rooted_path(Path(str(pre_state.get("path") or "")))
    exists_after = rooted.exists()
    stat = rooted.stat() if exists_after else None
    mtime_ns_after = stat.st_mtime_ns if stat else None
    mtime_ns_before = pre_state.get("mtime_ns_before")
    fresh = bool(
        exists_after
        and (
            pre_state.get("exists_before") is not True
            or (
                isinstance(mtime_ns_after, int)
                and isinstance(mtime_ns_before, int)
                and mtime_ns_after > mtime_ns_before
            )
        )
    )
    return {
        "path": _rel(rooted),
        "exists_before": bool(pre_state.get("exists_before")),
        "exists_after": exists_after,
        "mtime_ns_before": mtime_ns_before,
        "mtime_ns_after": mtime_ns_after,
        "fresh_for_current_execution": fresh,
    }


def _step_output_report_path(plan: dict[str, Any], step_name: str) -> Path | None:
    if step_name == "dry_run_prejump_capture":
        raw = (plan.get("persist_readiness_gate") or {}).get("path")
    elif step_name == "opt_in_live_odds_capture":
        raw = (plan.get("live_odds_approval_packet") or {}).get(
            "odds_capture_report_path"
        )
    elif step_name == "official_result_ingest_dry_run":
        raw = (plan.get("result_dry_run_report_gate") or {}).get("path")
    elif step_name == "result_label_write_readiness_validation":
        raw = (plan.get("label_write_readiness_validation_gate") or {}).get("path")
    elif step_name == "rolling_evaluation_dataset":
        raw = (plan.get("evaluation_report_gate") or {}).get("path")
    elif step_name == "snapshot_challenger_review":
        raw = (plan.get("snapshot_challenger_review_gate") or {}).get("path")
    elif step_name == "model_review_packet":
        raw = (plan.get("model_review_packet_gate") or {}).get("path")
    elif step_name == "calibration_layer_design":
        raw = (plan.get("calibration_design_gate") or {}).get("path")
    else:
        raw = None
    if not raw:
        return None
    return Path(str(raw))


def _report_freshness_failure_reason(step_name: str) -> str:
    return {
        "dry_run_prejump_capture": "dry_run_capture_report_not_fresh_for_current_execution",
        "opt_in_live_odds_capture": "odds_capture_report_not_fresh_for_current_execution",
        "official_result_ingest_dry_run": "result_dry_run_report_not_fresh_for_current_execution",
        "result_label_write_readiness_validation": "label_write_readiness_validation_report_not_fresh_for_current_execution",
        "rolling_evaluation_dataset": "evaluation_report_not_fresh_for_current_execution",
        "snapshot_challenger_review": "snapshot_challenger_review_not_fresh_for_current_execution",
        "model_review_packet": "model_review_packet_not_fresh_for_current_execution",
        "calibration_layer_design": "calibration_design_report_not_fresh_for_current_execution",
    }.get(step_name, "output_report_not_fresh_for_current_execution")


def _step(
    milestone: int,
    name: str,
    *,
    command: list[str] | None,
    status: str,
    write_scope: str,
    gate: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "milestone": milestone,
        "name": name,
        "status": status,
        "gate": gate,
        "reason": reason,
        "write_scope": write_scope,
        "command": command,
    }


def _approval_gated_step_state(
    packet: dict[str, Any],
    *,
    waiting_status: str,
    approval_required_reason: str,
) -> tuple[str, str | None]:
    packet_status = str(packet.get("status") or "DATA_MISSING")
    hard_stops = [str(item) for item in packet.get("hard_stops") or []]
    if packet_status.startswith("APPROVAL_PRESENT") and not hard_stops:
        return "READY_TO_RUN", None
    if packet_status.startswith("AWAITING_EXPLICIT_APPROVAL") and not hard_stops:
        return "APPROVAL_REQUIRED", approval_required_reason
    reason = (
        ",".join(hard_stops)
        if hard_stops
        else f"approval_packet_status:{packet_status}"
    )
    return waiting_status, reason


def _result_label_waiting_status(packet: dict[str, Any]) -> str:
    hard_stops = {str(item) for item in packet.get("hard_stops") or []}
    if "persisted_prejump_corpus_missing" in hard_stops:
        return "WAITING_FOR_PERSISTED_PREJUMP_SNAPSHOTS"
    if "persisted_prejump_races_not_jumped_yet" in hard_stops:
        return "WAITING_FOR_PERSISTED_RACES_TO_JUMP"
    if hard_stops.intersection(
        {
            "result_dry_run_report_missing",
            "result_dry_run_report_not_clean",
            "result_dry_run_report_not_fresh",
        }
    ):
        return "WAITING_FOR_CLEAN_RESULT_DRY_RUN"
    if "label_write_readiness_validation_not_ready" in hard_stops:
        return "WAITING_FOR_LABEL_WRITE_READINESS_VALIDATION"
    return "WAITING_FOR_READY_LABEL_PACKET"


def _steps_with_updated_approval_states(
    steps: list[dict[str, Any]],
    *,
    persist_packet: dict[str, Any],
    live_odds_packet: dict[str, Any],
    result_label_packet: dict[str, Any],
    label_write_preflight_gate: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    persist_status, persist_reason = _approval_gated_step_state(
        persist_packet,
        waiting_status="WAITING_FOR_READY_PERSIST_PACKET",
        approval_required_reason="snapshot persistence is blocked until approved",
    )
    odds_status, odds_reason = _approval_gated_step_state(
        live_odds_packet,
        waiting_status="WAITING_FOR_READY_ODDS_PACKET",
        approval_required_reason="live odds capture is blocked until approved",
    )
    label_status, label_reason = _approval_gated_step_state(
        result_label_packet,
        waiting_status=_result_label_waiting_status(result_label_packet),
        approval_required_reason="official result label writes are blocked until approved",
    )
    label_hard_stops = {str(item) for item in result_label_packet.get("hard_stops") or []}
    if not label_hard_stops or label_hard_stops == {
        "label_write_readiness_validation_not_ready"
    }:
        readiness_status = "READY_TO_RUN"
        readiness_reason = result_label_packet.get(
            "label_write_readiness_validation_reason"
        )
    else:
        readiness_status = _result_label_waiting_status(result_label_packet)
        readiness_reason = ",".join(sorted(label_hard_stops))
    preflight_gate = label_write_preflight_gate or {}
    preflight_gate_status = str(preflight_gate.get("status") or "DATA_MISSING")
    if preflight_gate_status == "READY":
        preflight_status = "READY"
        preflight_reason = None
    elif not label_hard_stops:
        preflight_status = "READY_FOR_OPERATOR_RUN_AFTER_LOOP_PLAN_WRITE"
        preflight_reason = (
            "write the loop plan artifact first, then build the no-write label preflight packet"
        )
    else:
        preflight_status = _result_label_waiting_status(result_label_packet)
        preflight_reason = ",".join(sorted(label_hard_stops))
    replacements = {
        "approved_persist_ready_subset": (persist_status, persist_reason),
        "opt_in_live_odds_capture": (odds_status, odds_reason),
        "result_label_write_readiness_validation": (
            readiness_status,
            readiness_reason,
        ),
        "label_write_preflight_packet": (preflight_status, preflight_reason),
        "approved_official_label_write": (label_status, label_reason),
    }
    updated_steps: list[dict[str, Any]] = []
    for step in steps:
        if (
            step.get("name") == "opt_in_live_odds_capture"
            and step.get("command") is None
            and step.get("status") == "COVERED_BY_APPROVED_PERSIST_WITH_LIVE_ODDS"
        ):
            updated_steps.append(dict(step))
            continue
        updated = dict(step)
        replacement = replacements.get(str(step.get("name")))
        if replacement:
            updated["status"], updated["reason"] = replacement
        updated_steps.append(updated)
    return updated_steps


def _step_execution_allowed(step: dict[str, Any], plan: dict[str, Any]) -> bool:
    if step.get("status") == "READY_TO_RUN":
        return True
    if (
        step.get("name") == "result_label_write_readiness_validation"
        and step.get("status") == "WAITING_FOR_CLEAN_RESULT_DRY_RUN"
    ):
        return True
    approval_key_by_step = {
        "approved_persist_ready_subset": "live_persist",
        "opt_in_live_odds_capture": "live_odds_capture",
        "approved_official_label_write": "result_label_write",
    }
    approval_key = approval_key_by_step.get(str(step.get("name")))
    approvals = plan.get("approvals") or {}
    return bool(approval_key and approvals.get(approval_key) is True)


def _audit_item(
    milestone: int,
    name: str,
    *,
    complete: bool,
    status: str,
    evidence: list[str],
    remaining: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "milestone": milestone,
        "name": name,
        "complete": complete,
        "status": status,
        "evidence": evidence,
        "remaining": remaining or [],
    }


def _milestone_completion_audit(
    *,
    approvals: dict[str, bool],
    current_corpus: dict[str, Any],
    persist_readiness_gate: dict[str, Any],
    result_dry_run_gate: dict[str, Any],
    evaluation_report_gate: dict[str, Any],
    promotion_readiness_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    persisted_count = int(
        current_corpus.get("ready_persisted_prediction_snapshot_count_for_date")
        or 0
    )
    persisted_corpus_present = persisted_count > 0
    jump_status = current_corpus.get("persisted_snapshot_jump_status") or {}
    waiting_for_known_future_jumps = (
        jump_status.get("wait_for_known_future_jumps_before_result_dry_run")
        is True
    )

    if persisted_corpus_present:
        milestone_5 = _audit_item(
            5,
            "persist_first_clean_current_corpus",
            complete=True,
            status="COMPLETE_READY_PERSISTED_PREJUMP_SNAPSHOTS_PRESENT",
            evidence=[
                f"ready persisted pre-jump snapshots for target date: {persisted_count}",
                "current corpus scanner requires prediction_snapshot_v1, result-free pre-jump state, READY readiness, and non-empty predictions",
            ],
        )
    else:
        milestone_5 = _audit_item(
            5,
            "persist_first_clean_current_corpus",
            complete=False,
            status=(
                "PENDING_APPROVED_FRESH_SAME_RUN_PERSIST"
                if approvals.get("live_persist")
                else "APPROVAL_REQUIRED"
            ),
            evidence=[
                f"current corpus status: {current_corpus.get('status')}",
                f"persist readiness gate: {persist_readiness_gate.get('status')}",
            ],
            remaining=[
                "run fresh refresh, validation, and dry-run capture",
                "obtain explicit APPROVE_LIVE_PERSIST or --approve-live-persist",
                "persist only same-run READY races after immediate pre-write freshness checks",
            ],
        )

    if not persisted_corpus_present:
        result_status = "WAITING_FOR_PERSISTED_PREJUMP_SNAPSHOTS"
        result_remaining = [
            "persist a READY pre-jump corpus first",
            "after races jump, run official-first result ingest dry-run",
            "write labels only after clean dry-run and explicit approval",
        ]
    elif waiting_for_known_future_jumps:
        result_status = "WAITING_FOR_PERSISTED_RACES_TO_JUMP"
        latest_known_future_jump = jump_status.get(
            "latest_known_future_jump_datetime_local"
        )
        result_remaining = [
            (
                "wait until latest known persisted jump time: "
                f"{latest_known_future_jump}"
            ),
            "then run official-first result ingest dry-run",
            "write labels only after clean dry-run and explicit approval",
        ]
    elif result_dry_run_gate.get("clean") is True:
        result_status = (
            "READY_FOR_APPROVED_LABEL_WRITE"
            if approvals.get("result_label_write")
            else "APPROVAL_REQUIRED_FOR_LABEL_WRITE"
        )
        result_remaining = (
            []
            if approvals.get("result_label_write")
            else ["obtain explicit APPROVE_RESULT_LABEL_WRITE or --write-labels-approved"]
        )
    else:
        result_status = "WAITING_FOR_CLEAN_OFFICIAL_RESULT_DRY_RUN"
        result_remaining = [
            "run official-first result ingest dry-run after persisted races jump",
            "resolve participant mismatches before label writes",
        ]

    evaluation_status = str(evaluation_report_gate.get("status") or "DATA_MISSING")
    evaluation_dataset_ready = evaluation_report_gate.get("dataset_ready") is True
    clean_metrics_ready = (
        evaluation_report_gate.get("clean_official_metrics_ready") is True
    )
    evaluation_complete = (
        persisted_corpus_present
        and evaluation_dataset_ready
        and clean_metrics_ready
        and evaluation_status == "READY"
    )
    report_only_evaluation_ready = evaluation_dataset_ready and clean_metrics_ready
    if not persisted_corpus_present:
        if report_only_evaluation_ready:
            evaluation_item_status = (
                f"REPORT_ONLY_EVALUATION_{evaluation_status}_AWAITING_PERSISTED_CURRENT_CORPUS"
            )
            evaluation_remaining = [
                "persist a READY pre-jump corpus first",
                "write official labels only after a clean approved result gate",
                "rerun rolling evaluation against the persisted/labelled current corpus",
                "treat the existing evaluation as historical/report-only until the current corpus exists",
            ]
        else:
            evaluation_item_status = "DATA_MISSING_NO_PERSISTED_PREJUMP_CORPUS"
            evaluation_remaining = [
                "persist a READY pre-jump corpus first",
                "write official labels only after a clean approved result gate",
                "rerun rolling evaluation against the persisted/labelled corpus",
            ]
    elif evaluation_complete:
        evaluation_item_status = "COMPLETE_ROLLING_EVALUATION_READY"
        evaluation_remaining = []
    else:
        evaluation_item_status = f"EVALUATION_GATE_{evaluation_status}"
        evaluation_remaining = [
            "rerun evaluation until the report and JSONL dataset are fresh",
            "require clean official races with top-1, top-3, log loss, Brier, and calibration metrics",
            "resolve any snapshot-corpus readiness warnings before treating the corpus as clean",
        ]

    diagnosis_complete = (
        persisted_corpus_present
        and clean_metrics_ready
        and evaluation_report_gate.get("model_quality_status") == "SUCCESS"
    )
    report_only_model_quality_ready = (
        report_only_evaluation_ready
        and evaluation_report_gate.get("model_quality_status") == "SUCCESS"
    )
    if not persisted_corpus_present:
        if report_only_model_quality_ready:
            diagnosis_status = (
                f"REPORT_ONLY_MODEL_QUALITY_{evaluation_status}_AWAITING_PERSISTED_CURRENT_CORPUS"
            )
            diagnosis_remaining = [
                "persist and label current READY races before relying on quality diagnosis",
                "keep retrain/promote blocked because the current target corpus is not persisted and labelled",
            ]
        else:
            diagnosis_status = "DATA_MISSING_NO_PERSISTED_PREJUMP_CORPUS"
            diagnosis_remaining = [
                "persist and label current READY races before relying on quality diagnosis",
            ]
    elif diagnosis_complete:
        diagnosis_status = "COMPLETE_MODEL_QUALITY_DIAGNOSIS_READY"
        diagnosis_remaining = []
    else:
        diagnosis_status = f"MODEL_QUALITY_GATE_{evaluation_status}"
        diagnosis_remaining = [
            "produce a clean evaluation report with official labels",
            "keep retrain/promote blocked unless the report-only gates prove enough clean evidence",
        ]

    promotion_gate = promotion_readiness_gate or {}
    promotion_status = str(promotion_gate.get("status") or "DATA_MISSING")
    promotion_ready = (
        promotion_status == "APPROVAL_PRESENT_EVIDENCE_READY_REPORT_ONLY"
        and promotion_gate.get("ready_for_separate_promotion_review") is True
    )
    if promotion_ready:
        milestone_10_status = "PROMOTION_EVIDENCE_READY_FOR_SEPARATE_REVIEW_REPORT_ONLY"
        milestone_10_remaining = [
            "run the full daily loop once the protected write gates are explicitly approved",
            "perform any actual model/config promotion only in a separate promotion path with registry/config write gates",
        ]
    elif approvals.get("promotion"):
        milestone_10_status = "PROMOTION_APPROVAL_PRESENT_EVIDENCE_NOT_READY"
        milestone_10_remaining = [
            "attach or generate a stable report-only challenger review and calibration design",
            "require at least 100 clean official evaluated races from historical or current evidence",
            "perform any actual model/config promotion only in a separate promotion path with registry/config write gates",
        ]
    else:
        milestone_10_status = "REPORT_ONLY_NO_PROMOTION"
        milestone_10_remaining = [
            "obtain explicit APPROVE_MODEL_PROMOTION before any promotion review",
            "keep promotion report-only until clean evidence and separate write gates are present",
        ]

    items = [
        _audit_item(
            1,
            "subset_aware_persist_gate",
            complete=True,
            status="IMPLEMENTED_IN_CODE_AND_EXECUTOR_GATES",
            evidence=[
                "capture path rechecks each READY race immediately before write",
                "loop executor requires fresh same-run dry-run report before approved persist",
            ],
        ),
        _audit_item(
            2,
            "venue_filename_contract",
            complete=True,
            status="IMPLEMENTED_IN_VALIDATORS_AND_PARSERS",
            evidence=[
                "hyphenated uppercase/alphanumeric venue tokens are accepted by browser, expert-form, metadata, and app filename parsers",
            ],
        ),
        _audit_item(
            3,
            "runner_set_mismatch_reduction",
            complete=True,
            status="IMPLEMENTED_WITH_CANONICAL_ALIGNMENT_AND_QUARANTINE",
            evidence=[
                "canonical final-runner alignment can drop scratched runners and remap promoted reserves",
                "source CSVs missing active canonical runners are quarantined instead of persisted",
            ],
        ),
        _audit_item(
            4,
            "target_metadata_coverage_hardening",
            complete=True,
            status="IMPLEMENTED_WITH_VERIFIED_METADATA_ONLY",
            evidence=[
                "expanded grade normalization covers non-graded, masters, pathways, special-event, and related class terms",
                "unverified grade/distance still quarantines instead of fabricating metadata",
            ],
        ),
        milestone_5,
        _audit_item(
            6,
            "opt_in_live_odds_capture",
            complete=True,
            status=(
                "IMPLEMENTED_APPROVAL_PRESENT"
                if approvals.get("live_odds_capture")
                else "IMPLEMENTED_APPROVAL_REQUIRED_FOR_LIVE_CAPTURE"
            ),
            evidence=[
                "live odds capture is opt-in only",
                "EV remains not ready/null unless dog-level odds provenance is URL-backed, timestamped, pre-prediction, pre-jump, and runner-aligned",
            ],
            remaining=(
                []
                if approvals.get("live_odds_capture")
                else ["obtain explicit APPROVE_LIVE_ODDS_CAPTURE or --approve-live-odds-capture for live odds collection"]
            ),
        ),
        _audit_item(
            7,
            "official_first_result_ingestion_gate",
            complete=False,
            status=result_status,
            evidence=[
                f"current corpus status: {current_corpus.get('status')}",
                f"result dry-run gate: {result_dry_run_gate.get('status')}",
                "known future persisted snapshots not jumped: "
                f"{jump_status.get('known_future_not_jumped_count')}",
            ],
            remaining=result_remaining,
        ),
        _audit_item(
            8,
            "rolling_evaluation_dataset",
            complete=evaluation_complete,
            status=evaluation_item_status,
            evidence=[
                f"evaluation report gate: {evaluation_report_gate.get('status')}",
                f"dataset rows written: {evaluation_report_gate.get('evaluation_dataset_rows_written')}",
                f"clean official races evaluated: {evaluation_report_gate.get('clean_official_races_evaluated')}",
            ],
            remaining=evaluation_remaining,
        ),
        _audit_item(
            9,
            "model_quality_diagnosis",
            complete=diagnosis_complete,
            status=diagnosis_status,
            evidence=[
                f"model quality status: {evaluation_report_gate.get('model_quality_status')}",
                f"retrain gate: {evaluation_report_gate.get('retrain_gate_status')}",
                f"promotion gate: {evaluation_report_gate.get('promotion_gate_status')}",
            ],
            remaining=diagnosis_remaining,
        ),
        _audit_item(
            10,
            "promotion_controlled_daily_loop",
            complete=False,
            status=milestone_10_status,
            evidence=[
                "loop plans refresh, validate, dry-run, approval-gated persist, result ingest, evaluation, and report-only promotion control",
                "no retrain or promotion is performed by this loop",
                f"promotion readiness gate: {promotion_status}",
                f"promotion evidence clean official races: {promotion_gate.get('promotion_evidence_clean_official_evaluated_races')}",
            ],
            remaining=milestone_10_remaining,
        ),
    ]
    incomplete = [item for item in items if item["complete"] is not True]
    return {
        "schema_version": "prejump_milestone_completion_audit_v1",
        "overall_status": "COMPLETE" if not incomplete else "INCOMPLETE",
        "completed_count": len(items) - len(incomplete),
        "incomplete_count": len(incomplete),
        "items": items,
    }


def _operator_next_action_report(
    *,
    approvals: dict[str, bool],
    current_corpus: dict[str, Any],
    persist_packet: dict[str, Any],
    live_odds_packet: dict[str, Any],
    result_label_packet: dict[str, Any],
    milestone_audit: dict[str, Any],
    label_write_readiness_gate: dict[str, Any] | None = None,
    label_write_preflight_gate: dict[str, Any] | None = None,
    safe_persist_packet_refresh_sequence: list[dict[str, Any]] | None = None,
    official_result_dry_run_command: list[str] | None = None,
    label_write_readiness_validation_command: list[str] | None = None,
    label_write_preflight_packet_command: list[str] | None = None,
    promotion_readiness_gate: dict[str, Any] | None = None,
    refresh_report_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    persisted_count = int(
        current_corpus.get("ready_persisted_prediction_snapshot_count_for_date")
        or 0
    )
    persist_status = str(persist_packet.get("status") or "DATA_MISSING")
    persist_hard_stops = [
        str(item) for item in persist_packet.get("hard_stops") or []
    ]
    persist_fresh_for_plan = (
        persist_packet.get("dry_run_report_fresh_for_plan") is True
    )
    persist_approval_window_status = "NOT_APPLICABLE_PERSISTED_CORPUS_PRESENT"
    label_status = str(result_label_packet.get("status") or "DATA_MISSING")
    label_hard_stops = [
        str(item) for item in result_label_packet.get("hard_stops") or []
    ]
    label_fresh_for_plan = (
        result_label_packet.get("result_dry_run_report_fresh_for_plan") is True
    )
    readiness_gate = label_write_readiness_gate or {}
    readiness_status = str(readiness_gate.get("status") or "DATA_MISSING")
    readiness_ready = readiness_status == "READY"
    preflight_gate = label_write_preflight_gate or {}
    preflight_status = str(preflight_gate.get("status") or "DATA_MISSING")
    jump_status = current_corpus.get("persisted_snapshot_jump_status") or {}
    waiting_for_known_future_jumps = (
        persisted_count > 0
        and jump_status.get("wait_for_known_future_jumps_before_result_dry_run")
        is True
    )
    refresh_gate = refresh_report_gate or {}
    refresh_next_window = refresh_gate.get("next_preferred_window")
    refresh_next_window = (
        refresh_next_window if isinstance(refresh_next_window, dict) else None
    )
    refresh_waiting_for_future_window = (
        refresh_gate.get("status") == "WAITING_FOR_FUTURE_WINDOW"
        and int(refresh_gate.get("selected_count") or 0) <= 0
    )
    if persisted_count <= 0:
        result_label_approval_window_status = "WAITING_FOR_PERSISTED_CORPUS"
    elif waiting_for_known_future_jumps:
        result_label_approval_window_status = "WAITING_FOR_PERSISTED_RACES_TO_JUMP"
    elif label_status == "APPROVAL_PRESENT_READY_TO_WRITE_OFFICIAL_LABELS":
        result_label_approval_window_status = "OPEN_APPROVED"
    elif label_status == "AWAITING_EXPLICIT_APPROVAL_READY_FOR_LABEL_WRITE":
        result_label_approval_window_status = "OPEN_AWAITING_APPROVAL"
    elif "result_dry_run_report_missing" in label_hard_stops:
        result_label_approval_window_status = "DRY_RUN_REQUIRED"
    elif "result_dry_run_report_not_fresh" in label_hard_stops:
        result_label_approval_window_status = "REFRESH_REQUIRED"
    elif "result_dry_run_report_not_clean" in label_hard_stops:
        result_label_approval_window_status = "CLEAN_DRY_RUN_REQUIRED"
    elif "label_write_readiness_validation_not_ready" in label_hard_stops:
        result_label_approval_window_status = "READINESS_VALIDATION_REQUIRED"
    else:
        result_label_approval_window_status = "REVIEW_REQUIRED"

    if persisted_count <= 0:
        if persist_status == "APPROVAL_PRESENT_READY_TO_EXECUTE_READY_SUBSET":
            persist_approval_window_status = "OPEN_APPROVED"
            next_step_status = "READY_FOR_APPROVED_PERSIST_EXECUTION"
            next_step_reason = "APPROVE_LIVE_PERSIST is present and the persist packet has no hard stops"
            required_gate = "APPROVE_LIVE_PERSIST"
            approval_required = False
            command_template = persist_packet.get(
                "approved_same_run_execute_ready_command_template"
            )
        elif persist_status == "AWAITING_EXPLICIT_APPROVAL_READY_SUBSET":
            persist_approval_window_status = "OPEN_AWAITING_APPROVAL"
            next_step_status = "APPROVAL_REQUIRED_FOR_READY_PERSIST_SUBSET"
            next_step_reason = (
                "persist only the READY pre-jump subset if explicit approval arrives before dry-run expiry"
            )
            required_gate = "APPROVE_LIVE_PERSIST"
            approval_required = True
            command_template = persist_packet.get(
                "approved_same_run_execute_ready_command_template"
            )
        else:
            if refresh_waiting_for_future_window:
                persist_approval_window_status = "WAITING_FOR_FUTURE_WINDOW"
                next_step_status = "WAIT_FOR_NEXT_PREFERRED_PREJUMP_WINDOW"
                next_step_reason = (
                    refresh_gate.get("recommended_rerun_after_local")
                    or refresh_gate.get("reason")
                    or "next_race_not_yet_inside_preferred_window"
                )
                required_gate = None
                approval_required = False
            else:
                persist_approval_window_status = "REFRESH_REQUIRED"
                next_step_status = "REFRESH_DRY_RUN_REQUIRED_FOR_PERSIST_PACKET"
                next_step_reason = ",".join(persist_hard_stops) or persist_status
                required_gate = "APPROVE_LIVE_PERSIST"
                approval_required = True
            command_template = None
    elif "label_write_readiness_validation_not_ready" in label_hard_stops:
        next_step_status = "RUN_LABEL_WRITE_READINESS_VALIDATION"
        next_step_reason = (
            readiness_gate.get("reason")
            or "label_write_readiness_validation_required"
        )
        required_gate = None
        approval_required = False
        command_template = label_write_readiness_validation_command
    elif (
        label_status
        in {
            "APPROVAL_PRESENT_READY_TO_WRITE_OFFICIAL_LABELS",
            "AWAITING_EXPLICIT_APPROVAL_READY_FOR_LABEL_WRITE",
        }
        and preflight_status != "READY"
    ):
        next_step_status = "RUN_LABEL_WRITE_PREFLIGHT_PACKET"
        next_step_reason = (
            preflight_gate.get("reason")
            or "label_write_preflight_packet_required"
        )
        required_gate = None
        approval_required = False
        command_template = label_write_preflight_packet_command
    elif label_status == "APPROVAL_PRESENT_READY_TO_WRITE_OFFICIAL_LABELS":
        next_step_status = "READY_FOR_APPROVED_OFFICIAL_LABEL_WRITE"
        next_step_reason = "APPROVE_RESULT_LABEL_WRITE is present and the label packet has no hard stops"
        required_gate = "APPROVE_RESULT_LABEL_WRITE"
        approval_required = False
        command_template = result_label_packet.get(
            "approved_same_run_execute_ready_command_template"
        )
    elif label_status == "AWAITING_EXPLICIT_APPROVAL_READY_FOR_LABEL_WRITE":
        next_step_status = "APPROVAL_REQUIRED_FOR_OFFICIAL_LABEL_WRITE"
        next_step_reason = "official result label writes are blocked until explicit approval"
        required_gate = "APPROVE_RESULT_LABEL_WRITE"
        approval_required = True
        command_template = result_label_packet.get(
            "approved_same_run_execute_ready_command_template"
        )
    elif waiting_for_known_future_jumps:
        next_step_status = "WAIT_FOR_PERSISTED_RACES_TO_JUMP_BEFORE_RESULT_DRY_RUN"
        next_step_reason = (
            jump_status.get("result_dry_run_wait_reason")
            or "known_persisted_races_not_jumped_yet"
        )
        required_gate = None
        approval_required = False
        command_template = None
    elif any(
        stop in set(label_hard_stops)
        for stop in {
            "result_dry_run_report_missing",
            "result_dry_run_report_not_clean",
            "result_dry_run_report_not_fresh",
        }
    ):
        next_step_status = "RUN_OR_REFRESH_OFFICIAL_RESULT_DRY_RUN"
        next_step_reason = ",".join(label_hard_stops) or label_status
        required_gate = None
        approval_required = False
        command_template = official_result_dry_run_command
    else:
        next_step_status = "REVIEW_MILESTONE_AUDIT"
        next_step_reason = milestone_audit.get("overall_status")
        required_gate = None
        approval_required = False
        command_template = None
    safe_refresh_sequence = (
        list(safe_persist_packet_refresh_sequence or [])
        if next_step_status
        in {
            "REFRESH_DRY_RUN_REQUIRED_FOR_PERSIST_PACKET",
            "WAIT_FOR_NEXT_PREFERRED_PREJUMP_WINDOW",
        }
        else []
    )
    persist_seconds_until_expiry = persist_packet.get(
        "dry_run_report_seconds_until_expiry"
    )
    persist_approval_window_urgency = _persist_approval_window_urgency(
        persist_approval_window_status,
        persist_seconds_until_expiry,
    )

    odds_status = str(live_odds_packet.get("status") or "DATA_MISSING")
    if odds_status == "APPROVAL_PRESENT_READY_TO_CAPTURE_LIVE_ODDS":
        live_odds_next_step = "READY_FOR_APPROVED_LIVE_ODDS_CAPTURE"
    elif odds_status == "AWAITING_EXPLICIT_APPROVAL_READY_FOR_LIVE_ODDS":
        live_odds_next_step = "APPROVAL_REQUIRED_FOR_LIVE_ODDS_CAPTURE"
    else:
        live_odds_next_step = "WAITING_FOR_READY_ODDS_PACKET"

    promotion_gate = promotion_readiness_gate or {
        "status": "DATA_MISSING",
        "reason": "promotion_readiness_gate_missing",
        "blockers": ["promotion_readiness_gate_missing"],
        "approval_present": approvals.get("promotion") is True,
        "required_gate": "APPROVE_MODEL_PROMOTION",
        "ready_for_separate_promotion_review": False,
    }
    promotion_status = str(promotion_gate.get("status") or "DATA_MISSING")
    if promotion_status == "APPROVAL_REQUIRED":
        promotion_next_step = "APPROVAL_REQUIRED_FOR_MODEL_PROMOTION"
    elif promotion_status == "APPROVAL_PRESENT_EVIDENCE_NOT_READY":
        promotion_next_step = "PROMOTION_APPROVAL_ACCEPTED_EVIDENCE_NOT_READY"
    elif promotion_status == "APPROVAL_PRESENT_EVIDENCE_READY_REPORT_ONLY":
        promotion_next_step = "PROMOTION_APPROVAL_ACCEPTED_READY_FOR_SEPARATE_REVIEW"
    else:
        promotion_next_step = "PROMOTION_REVIEW_REQUIRED"

    milestone_items = [
        item for item in milestone_audit.get("items") or [] if isinstance(item, dict)
    ]
    incomplete_milestones = [
        {
            "milestone": item.get("milestone"),
            "name": item.get("name"),
            "status": item.get("status"),
            "remaining": list(item.get("remaining") or []),
        }
        for item in milestone_items
        if item.get("complete") is not True
    ]

    return {
        "schema_version": "prejump_operator_next_action_v1",
        "overall_status": milestone_audit.get("overall_status"),
        "full_objective_complete": milestone_audit.get("overall_status")
        == "COMPLETE",
        "completed_milestone_count": milestone_audit.get("completed_count"),
        "incomplete_milestone_count": milestone_audit.get("incomplete_count"),
        "incomplete_milestones": incomplete_milestones,
        "next_step_status": next_step_status,
        "next_step_reason": next_step_reason,
        "required_gate": required_gate,
        "approval_required": approval_required,
        "command_template": command_template,
        "safe_no_approval_persist_packet_refresh_sequence": safe_refresh_sequence,
        "safe_no_approval_persist_packet_refresh_sequence_status": (
            "AVAILABLE" if safe_refresh_sequence else "NOT_REQUIRED"
        ),
        "refresh_report_gate_status": refresh_gate.get("status"),
        "refresh_report_gate_reason": refresh_gate.get("reason"),
        "refresh_total_races_found": refresh_gate.get("total_races_found"),
        "refresh_selected_count": refresh_gate.get("selected_count"),
        "refresh_bucket_counts": refresh_gate.get("bucket_counts") or {},
        "refresh_next_preferred_window": refresh_next_window,
        "refresh_recommended_rerun_after_local": refresh_gate.get(
            "recommended_rerun_after_local"
        ),
        "ready_count": int(persist_packet.get("ready_count") or 0),
        "not_ready_count": int(persist_packet.get("not_ready_count") or 0),
        "current_corpus_status": current_corpus.get("status"),
        "ready_persisted_prediction_snapshot_count_for_date": persisted_count,
        "persisted_snapshot_jump_status": jump_status,
        "result_dry_run_waiting_for_known_future_jumps": waiting_for_known_future_jumps,
        "known_future_not_jumped_snapshot_count_for_date": jump_status.get(
            "known_future_not_jumped_count"
        ),
        "result_dry_run_safe_after_latest_known_jump_utc": jump_status.get(
            "latest_known_future_jump_datetime_utc"
        ),
        "result_dry_run_safe_after_latest_known_jump_local": jump_status.get(
            "latest_known_future_jump_datetime_local"
        ),
        "persist_dry_run_fresh_for_plan": persist_fresh_for_plan,
        "persist_approval_window_status": persist_approval_window_status,
        "persist_approval_window_urgency": persist_approval_window_urgency,
        "persist_approval_command_template_status": persist_packet.get(
            "approval_command_template_status"
        ),
        "persist_same_run_execute_ready_command_template_status": persist_packet.get(
            "same_run_execute_ready_command_template_status"
        ),
        "persist_same_run_execute_ready_rechecks": list(
            persist_packet.get("same_run_execute_ready_rechecks") or []
        ),
        "result_label_packet_status": label_status,
        "result_label_hard_stops": label_hard_stops,
        "result_dry_run_fresh_for_plan": label_fresh_for_plan,
        "result_label_approval_window_status": result_label_approval_window_status,
        "result_label_approval_command_template_status": result_label_packet.get(
            "approval_command_template_status"
        ),
        "result_label_same_run_execute_ready_command_template_status": (
            result_label_packet.get("same_run_execute_ready_command_template_status")
        ),
        "result_label_same_run_execute_ready_rechecks": list(
            result_label_packet.get("same_run_execute_ready_rechecks") or []
        ),
        "label_write_readiness_validation_status": readiness_status,
        "label_write_readiness_validation_reason": readiness_gate.get("reason"),
        "label_write_readiness_validation_fresh_for_plan": (
            readiness_gate.get("fresh_for_plan") is True
        ),
        "label_write_readiness_validation_command_template": (
            label_write_readiness_validation_command
        ),
        "label_write_preflight_packet_status": preflight_status,
        "label_write_preflight_packet_reason": preflight_gate.get("reason"),
        "label_write_preflight_packet_fresh_for_plan": (
            preflight_gate.get("fresh_for_plan") is True
        ),
        "label_write_preflight_packet_command_template": (
            label_write_preflight_packet_command
        ),
        "dry_run_report_expires_at_utc": persist_packet.get(
            "dry_run_report_expires_at_utc"
        ),
        "dry_run_report_expires_at_local": persist_packet.get(
            "dry_run_report_expires_at_local"
        ),
        "dry_run_report_expires_at_local_timezone": persist_packet.get(
            "dry_run_report_expires_at_local_timezone"
        ),
        "dry_run_report_seconds_until_expiry": persist_seconds_until_expiry,
        "rerun_required_after_expiry": persist_packet.get(
            "rerun_required_after_expiry"
        )
        is True,
        "same_run_dry_run_required": True,
        "blocked_approval_gates": [
            gate
            for gate, approved in (
                ("APPROVE_LIVE_PERSIST", approvals.get("live_persist")),
                ("APPROVE_LIVE_ODDS_CAPTURE", approvals.get("live_odds_capture")),
                ("APPROVE_RESULT_LABEL_WRITE", approvals.get("result_label_write")),
                ("APPROVE_MODEL_PROMOTION", approvals.get("promotion")),
            )
            if approved is not True
        ],
        "live_odds_next_step_status": live_odds_next_step,
        "live_odds_required_gate": "APPROVE_LIVE_ODDS_CAPTURE",
        "live_odds_current_ev_readiness_counts": live_odds_packet.get(
            "current_ev_readiness_counts"
        )
        or {},
        "ev_summary_source": live_odds_packet.get("ev_summary_source"),
        "ev_ready_count": _int_count(live_odds_packet.get("ev_ready_count")),
        "ev_not_ready_count": _int_count(live_odds_packet.get("ev_not_ready_count")),
        "priced_ev_runner_count": _int_count(
            live_odds_packet.get("priced_ev_runner_count")
        ),
        "odds_exclusion_count": _int_count(
            live_odds_packet.get("odds_exclusion_count")
        ),
        "authoritative_capture_report_path": live_odds_packet.get(
            "authoritative_capture_report_path"
        ),
        "ev_summary_consistency_check": live_odds_packet.get(
            "ev_summary_consistency_check"
        ),
        "ev_summary_failure_reason": live_odds_packet.get(
            "ev_summary_failure_reason"
        ),
        "promotion_readiness_status": promotion_status,
        "promotion_next_step_status": promotion_next_step,
        "promotion_approval_present": promotion_gate.get("approval_present"),
        "promotion_required_gate": promotion_gate.get("required_gate"),
        "promotion_blockers": list(promotion_gate.get("blockers") or []),
        "promotion_blocker_details": list(
            promotion_gate.get("blocker_details") or []
        ),
        "promotion_evidence_source": promotion_gate.get(
            "promotion_evidence_source"
        ),
        "promotion_clean_official_evaluated_races": promotion_gate.get(
            "clean_official_evaluated_races"
        ),
        "promotion_evidence_clean_official_evaluated_races": (
            promotion_gate.get(
                "promotion_evidence_clean_official_evaluated_races"
            )
        ),
        "promotion_min_clean_official_evaluated_races": promotion_gate.get(
            "min_clean_official_evaluated_races"
        ),
        "promotion_historical_clean_official_races_can_satisfy_minimum": (
            promotion_gate.get(
                "historical_clean_official_races_can_satisfy_minimum"
            )
        ),
        "promotion_ready_for_separate_review": promotion_gate.get(
            "ready_for_separate_promotion_review"
        ),
        "promotion_action_taken": promotion_gate.get("promotion_action_taken")
        or "none",
        "forbidden_without_explicit_approval": [
            "snapshot_persist",
            "live_odds_capture",
            "result_label_write",
            "model_retrain_or_promotion",
            "betting",
        ],
    }


def build_loop_plan(args: argparse.Namespace) -> dict[str, Any]:
    approval_details = approval_provenance(args)
    approvals = {
        name: bool(details["approved"])
        for name, details in approval_details.items()
    }
    py = _repo_python()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    snapshot_dir = Path(args.snapshot_dir)
    db_path = Path(args.db)
    date_text = args.date or melbourne_now().date().isoformat()
    result_race_ids = _race_id_scope(list(args.result_race_id or []))
    current_corpus = _current_corpus_report(snapshot_dir, date_text)
    protected_resource_counters = _protected_resource_counters(
        snapshot_dir=snapshot_dir,
        date_text=date_text,
    )
    combined_live_odds_with_persist = bool(
        approvals["live_persist"] and approvals["live_odds_capture"]
    )
    persisted_corpus_present = (
        current_corpus["ready_persisted_prediction_snapshot_count_for_date"] > 0
    )
    persisted_races_waiting_for_jump = (
        current_corpus.get("result_dry_run_waiting_for_known_future_jumps") is True
    )
    upcoming_dir = (
        Path(args.upcoming_dir)
        if args.upcoming_dir
        else run_dir / "upcoming_races"
    )

    refresh_report = run_dir / "refresh_report.json"
    dry_capture_report = run_dir / "dry_run_capture_report.json"
    persist_report = run_dir / "persist_capture_report.json"
    odds_report = run_dir / "odds_capture_dry_snapshot_report.json"
    result_dry_run_report = run_dir / "result_ingest_dry_run_report.json"
    label_write_readiness_report = run_dir / "label_write_readiness_validation.json"
    label_write_preflight_report = run_dir / "label_write_preflight_packet.json"
    result_label_write_report = run_dir / "result_label_write_report.json"
    eval_report = run_dir / "evaluation_report.json"
    eval_dataset = run_dir / "evaluation_dataset.jsonl"
    run_challenger_review = bool(args.run_challenger_review)
    challenger_review_path = (
        Path(args.challenger_review)
        if args.challenger_review
        else run_dir / "snapshot_challenger_review.json"
        if run_challenger_review
        else None
    )
    generated_challenger_review = run_challenger_review and challenger_review_path is not None
    model_review_packet = run_dir / "model_review_packet.json"
    calibration_design_report = run_dir / "calibration_layer_design.json"
    promotion_model_review_packet_path = (
        Path(args.promotion_model_review_packet)
        if args.promotion_model_review_packet
        else None
    )
    promotion_calibration_design_path = (
        Path(args.promotion_calibration_design)
        if args.promotion_calibration_design
        else None
    )
    report_only_calibration_design_path = (
        Path(args.report_only_calibration_design)
        if args.report_only_calibration_design
        else None
    )
    evaluation_snapshots_manifest = (
        Path(args.evaluation_snapshots_manifest)
        if args.evaluation_snapshots_manifest
        else None
    )
    loop_plan_report = Path(args.output) if args.output else run_dir / "loop_plan.json"
    if not loop_plan_report.is_absolute():
        loop_plan_report = ROOT / loop_plan_report
    refresh_report_gate = _refresh_report_gate(refresh_report)
    persist_readiness_gate = _persist_readiness_gate(dry_capture_report)
    prediction_preview_report = _dry_run_prediction_preview_report(
        dry_capture_report,
        persist_readiness_gate,
    )
    persist_command = [
        py,
        "scripts/capture_prediction_snapshot.py",
        "--db",
        _rel(db_path),
        "--upcoming-dir",
        _rel(upcoming_dir),
        "--snapshot-dir",
        _rel(snapshot_dir),
        "--limit",
        str(args.limit),
        "--persist",
    ]
    report_only_calibration_design_args = (
        [
            "--report-only-calibration-design",
            _rel(report_only_calibration_design_path),
        ]
        if report_only_calibration_design_path is not None
        else []
    )
    persist_command.extend(report_only_calibration_design_args)
    if approvals["live_persist"]:
        persist_command.append("--approve-live-persist")
    if combined_live_odds_with_persist:
        persist_command.extend(
            [
                "--capture-live-odds",
                "--approve-live-odds-capture",
            ]
        )
    persist_command.extend(["--output", _rel(persist_report)])
    persist_same_run_execute_ready_command = _same_run_execute_ready_command(
        py=py,
        run_dir=run_dir,
        upcoming_dir=upcoming_dir,
        snapshot_dir=snapshot_dir,
        db_path=db_path,
        date_text=date_text,
        min_minutes=args.min_minutes,
        max_minutes=args.max_minutes,
        limit=args.limit,
        approval_flag="--approve-live-persist",
        output_path=run_dir / "loop_plan_execute_approved_persist.json",
        result_race_ids=result_race_ids,
        report_only_calibration_design=report_only_calibration_design_path,
    )
    persist_approval_packet = _persist_approval_packet(
        persist_readiness_gate=persist_readiness_gate,
        protected_resource_counters=protected_resource_counters,
        approvals=approvals,
        approval_details=approval_details,
        persist_command=persist_command,
        same_run_execute_ready_command=persist_same_run_execute_ready_command,
    )
    odds_command = [
        py,
        "scripts/capture_prediction_snapshot.py",
        "--db",
        _rel(db_path),
        "--upcoming-dir",
        _rel(upcoming_dir),
        "--snapshot-dir",
        _rel(snapshot_dir),
        "--limit",
        str(args.limit),
        "--capture-live-odds",
        "--output",
        _rel(odds_report),
    ]
    if report_only_calibration_design_args:
        odds_command[-2:-2] = report_only_calibration_design_args
    if approvals["live_odds_capture"]:
        odds_command.insert(-2, "--approve-live-odds-capture")
    odds_same_run_execute_ready_command = _same_run_execute_ready_command(
        py=py,
        run_dir=run_dir,
        upcoming_dir=upcoming_dir,
        snapshot_dir=snapshot_dir,
        db_path=db_path,
        date_text=date_text,
        min_minutes=args.min_minutes,
        max_minutes=args.max_minutes,
        limit=args.limit,
        approval_flag="--approve-live-odds-capture",
        output_path=run_dir / "loop_plan_execute_approved_live_odds.json",
        result_race_ids=result_race_ids,
        report_only_calibration_design=report_only_calibration_design_path,
    )
    live_odds_approval_packet = _live_odds_approval_packet(
        persist_readiness_gate=persist_readiness_gate,
        approvals=approvals,
        approval_details=approval_details,
        odds_command=odds_command,
        odds_report_path=odds_report,
        same_run_execute_ready_command=odds_same_run_execute_ready_command,
        ev_summary=_ev_summary_from_persist_readiness_gate(persist_readiness_gate),
    )
    if combined_live_odds_with_persist:
        live_odds_approval_packet[
            "covered_by_approved_persist_command"
        ] = True
        live_odds_approval_packet[
            "combined_persist_live_odds_command"
        ] = list(persist_command)
    persist_step_status, persist_step_reason = _approval_gated_step_state(
        persist_approval_packet,
        waiting_status="WAITING_FOR_READY_PERSIST_PACKET",
        approval_required_reason="snapshot persistence is blocked until approved",
    )
    odds_step_status, odds_step_reason = _approval_gated_step_state(
        live_odds_approval_packet,
        waiting_status="WAITING_FOR_READY_ODDS_PACKET",
        approval_required_reason="live odds capture is blocked until approved",
    )
    odds_step_command = odds_command
    if combined_live_odds_with_persist:
        odds_step_command = None
        odds_step_status = "COVERED_BY_APPROVED_PERSIST_WITH_LIVE_ODDS"
        odds_step_reason = (
            "approved persist command captures live odds before prediction "
            "and persistence"
        )
    label_write_command = [
        py,
        "scripts/ingest_results_for_date.py",
        "--db",
        _rel(db_path),
        "--date",
        date_text,
        "--upcoming-dir",
        _rel(upcoming_dir),
        "--snapshot-dir",
        _rel(snapshot_dir),
        "--require-ready-snapshot",
        "--approved-dry-run-report",
        _rel(result_dry_run_report),
        "--output",
        _rel(result_label_write_report),
    ]
    label_write_command.extend(_race_id_command_args(result_race_ids))
    if approvals["result_label_write"]:
        label_write_command.append("--write-labels-approved")
    label_write_readiness_command = [
        py,
        "scripts/ingest_results_for_date.py",
        "--db",
        _rel(db_path),
        "--date",
        date_text,
        "--upcoming-dir",
        _rel(upcoming_dir),
        "--snapshot-dir",
        _rel(snapshot_dir),
        "--require-ready-snapshot",
        "--approved-dry-run-report",
        _rel(result_dry_run_report),
        "--validate-label-write-readiness",
        "--output",
        _rel(label_write_readiness_report),
    ]
    label_write_readiness_command.extend(_race_id_command_args(result_race_ids))
    label_write_preflight_command = [
        py,
        "scripts/build_label_write_preflight_packet.py",
        "--label-readiness",
        _rel(label_write_readiness_report),
        "--result-dry-run-report",
        _rel(result_dry_run_report),
        "--loop-plan",
        _rel(loop_plan_report),
        "--db",
        _rel(db_path),
        "--output",
        _rel(label_write_preflight_report),
    ]
    label_write_same_run_execute_ready_command = _same_run_execute_ready_command(
        py=py,
        run_dir=run_dir,
        upcoming_dir=upcoming_dir,
        snapshot_dir=snapshot_dir,
        db_path=db_path,
        date_text=date_text,
        min_minutes=args.min_minutes,
        max_minutes=args.max_minutes,
        limit=args.limit,
        approval_flag="--write-labels-approved",
        output_path=run_dir / "loop_plan_execute_approved_label_write.json",
        result_race_ids=result_race_ids,
    )
    result_dry_run_command = [
        py,
        "scripts/ingest_results_for_date.py",
        "--db",
        _rel(db_path),
        "--date",
        date_text,
        "--upcoming-dir",
        _rel(upcoming_dir),
        "--snapshot-dir",
        _rel(snapshot_dir),
        "--require-ready-snapshot",
        "--dry-run",
        "--output",
        _rel(result_dry_run_report),
    ]
    result_dry_run_command.extend(_race_id_command_args(result_race_ids))
    result_dry_run_gate = _clean_result_dry_run_report(
        report_path=result_dry_run_report,
        date_text=date_text,
        db_path=db_path,
        upcoming_dir=upcoming_dir,
        snapshot_dir=snapshot_dir,
        race_ids=result_race_ids,
    )
    label_write_readiness_gate = _label_write_readiness_validation_gate(
        report_path=label_write_readiness_report,
        expected_scope=result_dry_run_gate["expected_scope"],
        approved_dry_run_report=result_dry_run_report,
    )
    label_write_preflight_gate = _label_write_preflight_packet_gate(
        report_path=label_write_preflight_report,
        expected_scope=result_dry_run_gate["expected_scope"],
        label_readiness_report=label_write_readiness_report,
        result_dry_run_report=result_dry_run_report,
        db_path=db_path,
    )
    result_label_approval_packet = _result_label_approval_packet(
        current_corpus=current_corpus,
        result_dry_run_gate=result_dry_run_gate,
        label_write_readiness_gate=label_write_readiness_gate,
        approvals=approvals,
        approval_details=approval_details,
        label_write_command=label_write_command,
        label_write_readiness_command=label_write_readiness_command,
        label_write_preflight_command=label_write_preflight_command,
        label_write_preflight_gate=label_write_preflight_gate,
        same_run_execute_ready_command=label_write_same_run_execute_ready_command,
    )
    evaluation_report_gate = _evaluation_report_gate(
        report_path=eval_report,
        dataset_path=eval_dataset,
    )
    snapshot_challenger_review_gate = (
        _snapshot_challenger_review_gate(
            report_path=challenger_review_path,
            dataset_path=eval_dataset,
        )
        if challenger_review_path
        else {
            "path": None,
            "dataset_path": _rel(eval_dataset),
            "status": "NOT_REQUESTED",
            "reason": "run_challenger_review_not_requested",
        }
    )
    model_review_packet_gate = _model_review_packet_gate(
        packet_path=model_review_packet,
        evaluation_report_path=eval_report,
        dataset_path=eval_dataset,
        challenger_review_path=challenger_review_path,
    )
    calibration_design_gate = _calibration_design_gate(
        report_path=calibration_design_report,
        model_review_packet_path=model_review_packet,
    )
    promotion_model_review_packet_gate = _promotion_model_review_packet_gate(
        promotion_model_review_packet_path,
    )
    promotion_calibration_design_gate = _promotion_calibration_design_gate(
        report_path=promotion_calibration_design_path,
        model_review_packet_path=promotion_model_review_packet_path,
    )
    promotion_readiness_gate = _promotion_readiness_gate(
        approvals=approvals,
        evaluation_report_gate=evaluation_report_gate,
        snapshot_challenger_review_gate=snapshot_challenger_review_gate,
        model_review_packet_gate=model_review_packet_gate,
        calibration_design_gate=calibration_design_gate,
        promotion_model_review_packet_gate=promotion_model_review_packet_gate,
        promotion_calibration_design_gate=promotion_calibration_design_gate,
    )
    promotion_loop_step_status = (
        "APPROVAL_PRESENT_EVIDENCE_READY_REPORT_ONLY"
        if promotion_readiness_gate.get("status")
        == "APPROVAL_PRESENT_EVIDENCE_READY_REPORT_ONLY"
        else "APPROVAL_PRESENT_EVIDENCE_NOT_READY"
        if approvals["promotion"]
        else "REPORT_ONLY"
    )
    result_dry_run_status = (
        "WAITING_FOR_PERSISTED_PREJUMP_SNAPSHOTS"
        if not persisted_corpus_present
        else "WAITING_FOR_PERSISTED_RACES_TO_JUMP"
        if persisted_races_waiting_for_jump
        else "READY_TO_RUN"
    )
    result_dry_run_reason = (
        "official result dry-run waits until pre-jump snapshots exist for the target date"
        if not persisted_corpus_present
        else "known persisted pre-jump snapshots have future jump times"
        if persisted_races_waiting_for_jump
        else None
    )
    if not persisted_corpus_present:
        label_write_status = "WAITING_FOR_PERSISTED_PREJUMP_SNAPSHOTS"
        label_write_reason = "official label writes require persisted pre-jump snapshots first"
    elif persisted_races_waiting_for_jump:
        label_write_status = "WAITING_FOR_PERSISTED_RACES_TO_JUMP"
        label_write_reason = "official label writes require persisted races to jump first"
    elif result_dry_run_gate["clean"] is not True:
        label_write_status = "WAITING_FOR_CLEAN_RESULT_DRY_RUN"
        label_write_reason = "official label writes require a clean result-ingest dry-run report first"
    elif label_write_readiness_gate["status"] != "READY":
        label_write_status = "WAITING_FOR_LABEL_WRITE_READINESS_VALIDATION"
        label_write_reason = "official label writes require a read-only label-write readiness validation first"
    elif not approvals["result_label_write"]:
        label_write_status = "APPROVAL_REQUIRED"
        label_write_reason = "official result label writes are blocked until approved"
    else:
        label_write_status = "READY_TO_RUN"
        label_write_reason = None

    if not persisted_corpus_present:
        label_readiness_status = "WAITING_FOR_PERSISTED_PREJUMP_SNAPSHOTS"
        label_readiness_reason = "label-write readiness validation requires persisted pre-jump snapshots first"
    elif persisted_races_waiting_for_jump:
        label_readiness_status = "WAITING_FOR_PERSISTED_RACES_TO_JUMP"
        label_readiness_reason = "label-write readiness validation requires persisted races to jump first"
    elif result_dry_run_gate["clean"] is not True:
        label_readiness_status = "WAITING_FOR_CLEAN_RESULT_DRY_RUN"
        label_readiness_reason = "label-write readiness validation requires a clean result-ingest dry-run report first"
    elif label_write_readiness_gate["status"] == "READY":
        label_readiness_status = "READY"
        label_readiness_reason = None
    else:
        label_readiness_status = "READY_TO_RUN"
        label_readiness_reason = label_write_readiness_gate.get("reason")

    if label_readiness_status.startswith("WAITING_FOR_"):
        label_preflight_status = label_readiness_status
        label_preflight_reason = (
            "label-write preflight packet requires ready label-write readiness validation"
        )
    elif label_write_preflight_gate["status"] == "READY":
        label_preflight_status = "READY"
        label_preflight_reason = None
    elif label_write_readiness_gate["status"] == "READY":
        label_preflight_status = "READY_FOR_OPERATOR_RUN_AFTER_LOOP_PLAN_WRITE"
        label_preflight_reason = (
            "write the loop plan artifact first, then build the no-write label preflight packet"
        )
    else:
        label_preflight_status = "WAITING_FOR_LABEL_WRITE_READINESS_VALIDATION"
        label_preflight_reason = label_write_readiness_gate.get("reason")

    model_review_packet_command = [
        py,
        "scripts/build_model_review_packet.py",
        "--evaluation-report",
        _rel(eval_report),
        "--dataset-output",
        _rel(eval_dataset),
        "--output",
        _rel(model_review_packet),
    ]
    if challenger_review_path:
        model_review_packet_command.extend(
            ["--challenger-review", _rel(challenger_review_path)]
        )
    challenger_review_command = (
        [
            py,
            "scripts/review_snapshot_challenger.py",
            "--dataset",
            _rel(eval_dataset),
            "--output",
            _rel(challenger_review_path),
        ]
        if generated_challenger_review
        else None
    )
    challenger_review_status = (
        "READY_TO_RUN"
        if generated_challenger_review
        else "PROVIDED_EXTERNALLY"
        if challenger_review_path
        else "WAITING_FOR_RUN_CHALLENGER_REVIEW_OR_EXTERNAL_REVIEW"
    )
    challenger_review_reason = (
        "execute path requires a fresh same-run evaluation dataset before writing the report-only challenger review"
        if generated_challenger_review
        else "external challenger review path supplied"
        if challenger_review_path
        else "add --run-challenger-review or --challenger-review to feed model review and calibration design gates"
    )
    calibration_design_command = [
        py,
        "scripts/design_calibration_layer.py",
        "--model-review-packet",
        _rel(model_review_packet),
        "--output",
        _rel(calibration_design_report),
    ]
    calibration_design_status = (
        "READY_TO_RUN"
        if challenger_review_path
        else "WAITING_FOR_STABLE_REPORT_ONLY_CHALLENGER_REVIEW"
    )
    calibration_design_reason = (
        "execute path requires a fresh same-run model review packet with a stable challenger gate"
        if challenger_review_path
        else "calibration design waits for --challenger-review so the model review packet can validate a stable report-only challenger"
    )
    evaluation_command = [
        py,
        "scripts/evaluate_prediction_snapshots.py",
        "--db",
        _rel(db_path),
        "--output",
        _rel(eval_report),
        "--dataset-output",
        _rel(eval_dataset),
    ]
    if evaluation_snapshots_manifest:
        evaluation_command.extend(
            ["--snapshots-manifest", _rel(evaluation_snapshots_manifest)]
        )
        evaluation_snapshot_scope = "manifest"
    else:
        evaluation_command.extend(["--snapshots", _rel(snapshot_dir)])
        evaluation_snapshot_scope = "snapshot_dir"

    steps = [
        _step(
            1,
            "subset_aware_persist_gate",
            command=None,
            status="IMPLEMENTED_IN_CAPTURE_SCRIPT",
            write_scope="code_gate_only",
            reason="persist writes recheck each READY race immediately before write",
        ),
        _step(
            2,
            "venue_filename_contract",
            command=None,
            status="IMPLEMENTED_IN_VALIDATORS_AND_PARSERS",
            write_scope="code_gate_only",
            reason=(
                "hyphenated uppercase/alphanumeric venue tokens are accepted by "
                "validators, housekeeping parsers, browser filename metadata, "
                "and expert-form duplicate tracking"
            ),
        ),
        _step(
            3,
            "runner_set_mismatch_reduction",
            command=None,
            status="IMPLEMENTED_IN_REFRESH_AND_CAPTURE_GATES",
            write_scope="code_gate_only",
            reason=(
                "canonical final-runner alignment drops scratched/unpromoted reserves, "
                "remaps promoted reserves, and quarantines source CSVs missing active "
                "canonical runners before capture"
            ),
        ),
        _step(
            4,
            "target_metadata_coverage",
            command=None,
            status="IMPLEMENTED_IN_REFRESH_PATH",
            write_scope="code_gate_only",
            reason=(
                "explicit Heat/Masters/Non Graded/Special Event class words are "
                "accepted only from current-race canonical metadata"
            ),
        ),
        _step(
            5,
            "fresh_refresh_current_window",
            command=[
                py,
                "scripts/refresh_prejump_upcoming.py",
                "--upcoming-dir",
                _rel(upcoming_dir),
                "--min-minutes",
                str(args.min_minutes),
                "--max-minutes",
                str(args.max_minutes),
                "--limit",
                str(args.limit),
                "--output",
                _rel(refresh_report),
            ],
            status="READY_TO_RUN",
            write_scope="local_upcoming_csv_raw_export_quarantine_artifacts",
        ),
        _step(
            5,
            "validate_current_upcoming_contract",
            command=[
                py,
                "scripts/validate_upcoming_races.py",
                "--dir",
                _rel(upcoming_dir),
            ],
            status="READY_TO_RUN",
            write_scope="validation_report_only",
        ),
        _step(
            5,
            "dry_run_prejump_capture",
            command=[
                py,
                "scripts/capture_prediction_snapshot.py",
                "--db",
                _rel(db_path),
                "--upcoming-dir",
                _rel(upcoming_dir),
                "--snapshot-dir",
                _rel(snapshot_dir),
                "--limit",
                str(args.limit),
                *report_only_calibration_design_args,
                "--output",
                _rel(dry_capture_report),
            ],
            status="READY_TO_RUN",
            write_scope="report_only_no_snapshot_write",
        ),
        _step(
            5,
            "approved_persist_ready_subset",
            command=persist_command,
            status=persist_step_status,
            gate="APPROVE_LIVE_PERSIST",
            reason=persist_step_reason,
            write_scope="append_only_snapshot_json_and_manifest",
        ),
        _step(
            6,
            "opt_in_live_odds_capture",
            command=odds_step_command,
            status=odds_step_status,
            gate="APPROVE_LIVE_ODDS_CAPTURE",
            reason=odds_step_reason,
            write_scope="append_only_live_odds_rows",
        ),
        _step(
            7,
            "official_result_ingest_dry_run",
            command=result_dry_run_command,
            status=result_dry_run_status,
            reason=result_dry_run_reason,
            write_scope="fetch_and_report_only_no_label_write",
        ),
        _step(
            7,
            "result_label_write_readiness_validation",
            command=label_write_readiness_command,
            status=label_readiness_status,
            reason=label_readiness_reason,
            write_scope="report_only_no_fetch_no_label_write",
        ),
        _step(
            7,
            "label_write_preflight_packet",
            command=label_write_preflight_command,
            status=label_preflight_status,
            reason=label_preflight_reason,
            write_scope="report_only_no_label_write",
        ),
        _step(
            7,
            "approved_official_label_write",
            command=label_write_command,
            status=label_write_status,
            gate="APPROVE_RESULT_LABEL_WRITE",
            reason=label_write_reason,
            write_scope="db_label_rows_with_pre_write_backup",
        ),
        _step(
            8,
            "rolling_evaluation_dataset",
            command=evaluation_command,
            status="READY_TO_RUN",
            write_scope="evaluation_report_and_dataset_artifacts_only",
        ),
        _step(
            9,
            "model_quality_diagnosis",
            command=None,
            status="INCLUDED_IN_EVALUATION_REPORT",
            write_scope="report_only",
            reason="evaluation report includes feature missingness, box-bias, calibration, and retrain gate diagnostics",
        ),
        _step(
            10,
            "snapshot_challenger_review",
            command=challenger_review_command,
            status=challenger_review_status,
            write_scope="report_only_no_model_artifact_registry_or_promotion",
            reason=challenger_review_reason,
        ),
        _step(
            10,
            "model_review_packet",
            command=model_review_packet_command,
            status="READY_TO_RUN",
            write_scope="report_only_no_training_registry_or_promotion",
            reason="execute path requires a fresh same-run evaluation report before writing the review packet",
        ),
        _step(
            10,
            "calibration_layer_design",
            command=calibration_design_command if challenger_review_path else None,
            status=calibration_design_status,
            write_scope="report_only_no_model_registry_config_or_promotion",
            reason=calibration_design_reason,
        ),
        _step(
            10,
            "promotion_controlled_loop",
            command=None,
            status=promotion_loop_step_status,
            gate="APPROVE_MODEL_PROMOTION",
            reason="no retrain or promotion is performed by this loop",
            write_scope="none",
        ),
    ]
    safe_persist_packet_refresh_sequence = [
        {
            "name": step.get("name"),
            "command": list(step.get("command") or []),
            "write_scope": step.get("write_scope"),
        }
        for step in steps
        if step.get("name")
        in {
            "fresh_refresh_current_window",
            "validate_current_upcoming_contract",
            "dry_run_prejump_capture",
        }
    ]
    milestone_completion_audit = _milestone_completion_audit(
        approvals=approvals,
        current_corpus=current_corpus,
        persist_readiness_gate=persist_readiness_gate,
        result_dry_run_gate=result_dry_run_gate,
        evaluation_report_gate=evaluation_report_gate,
        promotion_readiness_gate=promotion_readiness_gate,
    )
    return {
        "schema_version": "prejump_prediction_loop_plan_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_root": str(ROOT),
        "run_dir": str(run_dir),
        "result_race_ids": result_race_ids,
        "evaluation_snapshot_scope": evaluation_snapshot_scope,
        "evaluation_snapshots_manifest": (
            _rel(_rooted_path(evaluation_snapshots_manifest))
            if evaluation_snapshots_manifest
            else None
        ),
        "approvals": approvals,
        "approval_provenance": approval_details,
        "refresh_report_gate": refresh_report_gate,
        "result_dry_run_report_gate": result_dry_run_gate,
        "label_write_readiness_validation_gate": label_write_readiness_gate,
        "label_write_preflight_packet_gate": label_write_preflight_gate,
        "persist_readiness_gate": persist_readiness_gate,
        "ev_readiness_summary": _ev_summary_from_persist_readiness_gate(
            persist_readiness_gate
        ),
        "persist_approval_packet": persist_approval_packet,
        "live_odds_approval_packet": live_odds_approval_packet,
        "result_label_approval_packet": result_label_approval_packet,
        "prediction_preview_report": prediction_preview_report,
        "latest_prediction_preview_report": prediction_preview_report,
        "latest_prediction_preview_report_phase": "initial_plan",
        "evaluation_report_gate": evaluation_report_gate,
        "snapshot_challenger_review_gate": snapshot_challenger_review_gate,
        "model_review_packet_gate": model_review_packet_gate,
        "calibration_design_gate": calibration_design_gate,
        "promotion_model_review_packet_gate": promotion_model_review_packet_gate,
        "promotion_calibration_design_gate": promotion_calibration_design_gate,
        "promotion_readiness_gate": promotion_readiness_gate,
        "current_corpus": current_corpus,
        "protected_resource_counters": protected_resource_counters,
        "milestone_completion_audit": milestone_completion_audit,
        "operator_next_action": _operator_next_action_report(
            approvals=approvals,
            current_corpus=current_corpus,
            persist_packet=persist_approval_packet,
            live_odds_packet=live_odds_approval_packet,
            result_label_packet=result_label_approval_packet,
            label_write_readiness_gate=label_write_readiness_gate,
            label_write_preflight_gate=label_write_preflight_gate,
            milestone_audit=milestone_completion_audit,
            safe_persist_packet_refresh_sequence=safe_persist_packet_refresh_sequence,
            official_result_dry_run_command=result_dry_run_command,
            label_write_readiness_validation_command=label_write_readiness_command,
            label_write_preflight_packet_command=label_write_preflight_command,
            promotion_readiness_gate=promotion_readiness_gate,
            refresh_report_gate=refresh_report_gate,
        ),
        "gated_actions_default_blocked": {
            "snapshot_persist": not approvals["live_persist"],
            "live_odds_capture": not approvals["live_odds_capture"],
            "result_label_write": not approvals["result_label_write"],
            "model_promotion": not approvals["promotion"],
        },
        "steps": steps,
        "guarantees": {
            "no_betting": True,
            "no_fake_odds_or_ev": True,
            "no_retrain": True,
            "no_model_promotion": True,
            "no_push": True,
        },
    }


def execute_ready_steps(plan: dict[str, Any]) -> list[dict[str, Any]]:
    run_dir = Path(plan["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    completed_successful_steps: set[str] = set()
    fresh_reports: set[str] = set()
    generated_challenger_review_requested = any(
        step.get("name") == "snapshot_challenger_review" and step.get("command")
        for step in plan.get("steps") or []
        if isinstance(step, dict)
    )
    for step in plan["steps"]:
        command = step.get("command")
        if not command or not _step_execution_allowed(step, plan):
            continue
        step_name = str(step.get("name"))
        if step_name == "approved_persist_ready_subset":
            if "dry_run_prejump_capture" not in completed_successful_steps:
                results.append(
                    {
                        "name": step_name,
                        "returncode": None,
                        "status": "SKIPPED",
                        "reason": "dry_run_prejump_capture_not_completed_in_this_execution",
                    }
                )
                break
            if "dry_run_prejump_capture" not in fresh_reports:
                results.append(
                    {
                        "name": step_name,
                        "returncode": None,
                        "status": "SKIPPED",
                        "reason": "dry_run_capture_report_not_fresh_for_current_execution",
                    }
                )
                break
            gate_path = Path(
                str((plan.get("persist_readiness_gate") or {}).get("path") or "")
            )
            gate = _persist_readiness_gate(gate_path)
            if gate.get("clean_for_ready_subset_persist") is not True:
                results.append(
                    {
                        "name": step_name,
                        "returncode": None,
                        "status": "SKIPPED",
                        "reason": "persist_readiness_gate_not_clean",
                        "persist_readiness_gate": gate,
                    }
                )
                break
        if step_name == "opt_in_live_odds_capture":
            if "dry_run_prejump_capture" not in completed_successful_steps:
                results.append(
                    {
                        "name": step_name,
                        "returncode": None,
                        "status": "SKIPPED",
                        "reason": "dry_run_prejump_capture_not_completed_in_this_execution",
                    }
                )
                break
            if "dry_run_prejump_capture" not in fresh_reports:
                results.append(
                    {
                        "name": step_name,
                        "returncode": None,
                        "status": "SKIPPED",
                        "reason": "dry_run_capture_report_not_fresh_for_current_execution",
                    }
                )
                break
            odds_gate_path = Path(
                str((plan.get("persist_readiness_gate") or {}).get("path") or "")
            )
            odds_gate = _persist_readiness_gate(odds_gate_path)
            if odds_gate.get("clean_for_ready_subset_persist") is not True:
                results.append(
                    {
                        "name": step_name,
                        "returncode": None,
                        "status": "SKIPPED",
                        "reason": "odds_capture_readiness_gate_not_clean",
                        "live_odds_readiness_gate": odds_gate,
                    }
                )
                break
        if step_name == "result_label_write_readiness_validation":
            result_gate_plan = plan.get("result_dry_run_report_gate") or {}
            expected_scope = result_gate_plan.get("expected_scope")
            if not isinstance(expected_scope, dict):
                results.append(
                    {
                        "name": step_name,
                        "returncode": None,
                        "status": "SKIPPED",
                        "reason": "result_dry_run_gate_expected_scope_missing",
                    }
                )
                break
            result_gate = _clean_result_dry_run_report_for_scope(
                report_path=Path(str(result_gate_plan.get("path") or "")),
                expected_scope=expected_scope,
            )
            if result_gate.get("clean") is not True:
                results.append(
                    {
                        "name": step_name,
                        "returncode": None,
                        "status": "SKIPPED",
                        "reason": "result_dry_run_gate_not_clean",
                        "result_dry_run_report_gate": result_gate,
                    }
                )
                break
        if step_name == "approved_official_label_write":
            if "official_result_ingest_dry_run" not in completed_successful_steps:
                results.append(
                    {
                        "name": step_name,
                        "returncode": None,
                        "status": "SKIPPED",
                        "reason": "official_result_ingest_dry_run_not_completed_in_this_execution",
                    }
                )
                break
            if "official_result_ingest_dry_run" not in fresh_reports:
                results.append(
                    {
                        "name": step_name,
                        "returncode": None,
                        "status": "SKIPPED",
                        "reason": "result_dry_run_report_not_fresh_for_current_execution",
                    }
                )
                break
            result_gate_plan = plan.get("result_dry_run_report_gate") or {}
            expected_scope = result_gate_plan.get("expected_scope")
            if not isinstance(expected_scope, dict):
                results.append(
                    {
                        "name": step_name,
                        "returncode": None,
                        "status": "SKIPPED",
                        "reason": "result_dry_run_gate_expected_scope_missing",
                    }
                )
                break
            result_gate = _clean_result_dry_run_report_for_scope(
                report_path=Path(str(result_gate_plan.get("path") or "")),
                expected_scope=expected_scope,
            )
            if result_gate.get("clean") is not True:
                results.append(
                    {
                        "name": step_name,
                        "returncode": None,
                        "status": "SKIPPED",
                        "reason": "result_dry_run_gate_not_clean",
                        "result_dry_run_report_gate": result_gate,
                    }
                )
                break
            preflight_gate_plan = plan.get("label_write_preflight_packet_gate") or {}
            readiness_gate_plan = (
                plan.get("label_write_readiness_validation_gate") or {}
            )
            preflight_gate = _label_write_preflight_packet_gate(
                report_path=Path(str(preflight_gate_plan.get("path") or "")),
                expected_scope=expected_scope,
                label_readiness_report=Path(
                    str(readiness_gate_plan.get("path") or "")
                ),
                result_dry_run_report=Path(
                    str(result_gate_plan.get("path") or "")
                ),
                db_path=Path(
                    str(
                        preflight_gate_plan.get("db_path")
                        or expected_scope.get("db_path")
                        or ""
                    )
                ),
            )
            if preflight_gate.get("status") != "READY":
                results.append(
                    {
                        "name": step_name,
                        "returncode": None,
                        "status": "SKIPPED",
                        "reason": "label_write_preflight_packet_not_ready",
                        "result_dry_run_report_gate": result_gate,
                        "label_write_preflight_packet_gate": preflight_gate,
                    }
                )
                break
        if step_name == "snapshot_challenger_review":
            if "rolling_evaluation_dataset" not in completed_successful_steps:
                results.append(
                    {
                        "name": step_name,
                        "returncode": None,
                        "status": "SKIPPED",
                        "reason": "rolling_evaluation_dataset_not_completed_in_this_execution",
                    }
                )
                break
            if "rolling_evaluation_dataset" not in fresh_reports:
                results.append(
                    {
                        "name": step_name,
                        "returncode": None,
                        "status": "SKIPPED",
                        "reason": "evaluation_report_not_fresh_for_current_execution",
                    }
                )
                break
            eval_gate_plan = plan.get("evaluation_report_gate") or {}
            eval_dataset = (
                Path(str(eval_gate_plan.get("dataset_path")))
                if eval_gate_plan.get("dataset_path")
                else None
            )
            eval_gate = _evaluation_report_gate(
                report_path=Path(str(eval_gate_plan.get("path") or "")),
                dataset_path=eval_dataset,
            )
            if eval_gate.get("dataset_ready") is not True or eval_gate.get(
                "clean_official_metrics_ready"
            ) is not True:
                results.append(
                    {
                        "name": step_name,
                        "returncode": None,
                        "status": "SKIPPED",
                        "reason": "evaluation_report_gate_not_ready",
                        "evaluation_report_gate": eval_gate,
                    }
                )
                break
        if step_name == "model_review_packet":
            if "rolling_evaluation_dataset" not in completed_successful_steps:
                results.append(
                    {
                        "name": step_name,
                        "returncode": None,
                        "status": "SKIPPED",
                        "reason": "rolling_evaluation_dataset_not_completed_in_this_execution",
                    }
                )
                break
            if "rolling_evaluation_dataset" not in fresh_reports:
                results.append(
                    {
                        "name": step_name,
                        "returncode": None,
                        "status": "SKIPPED",
                        "reason": "evaluation_report_not_fresh_for_current_execution",
                    }
                )
                break
            eval_gate_plan = plan.get("evaluation_report_gate") or {}
            eval_dataset = (
                Path(str(eval_gate_plan.get("dataset_path")))
                if eval_gate_plan.get("dataset_path")
                else None
            )
            eval_gate = _evaluation_report_gate(
                report_path=Path(str(eval_gate_plan.get("path") or "")),
                dataset_path=eval_dataset,
            )
            if eval_gate.get("dataset_ready") is not True or eval_gate.get(
                "clean_official_metrics_ready"
            ) is not True:
                results.append(
                    {
                        "name": step_name,
                        "returncode": None,
                        "status": "SKIPPED",
                        "reason": "evaluation_report_gate_not_ready",
                        "evaluation_report_gate": eval_gate,
                    }
                )
                break
            if generated_challenger_review_requested:
                if "snapshot_challenger_review" not in completed_successful_steps:
                    results.append(
                        {
                            "name": step_name,
                            "returncode": None,
                            "status": "SKIPPED",
                            "reason": "snapshot_challenger_review_not_completed_in_this_execution",
                        }
                    )
                    break
                if "snapshot_challenger_review" not in fresh_reports:
                    results.append(
                        {
                            "name": step_name,
                            "returncode": None,
                            "status": "SKIPPED",
                            "reason": "snapshot_challenger_review_not_fresh_for_current_execution",
                        }
                    )
                    break
                review_gate_plan = plan.get("snapshot_challenger_review_gate") or {}
                review_gate = _snapshot_challenger_review_gate(
                    report_path=Path(str(review_gate_plan.get("path") or "")),
                    dataset_path=Path(
                        str(review_gate_plan.get("dataset_path") or "")
                    ),
                )
                if review_gate.get("status") != "READY":
                    results.append(
                        {
                            "name": step_name,
                            "returncode": None,
                            "status": "SKIPPED",
                            "reason": "snapshot_challenger_review_gate_not_ready",
                            "snapshot_challenger_review_gate": review_gate,
                        }
                    )
                    break
        if step_name == "calibration_layer_design":
            if "model_review_packet" not in completed_successful_steps:
                results.append(
                    {
                        "name": step_name,
                        "returncode": None,
                        "status": "SKIPPED",
                        "reason": "model_review_packet_not_completed_in_this_execution",
                    }
                )
                break
            if "model_review_packet" not in fresh_reports:
                results.append(
                    {
                        "name": step_name,
                        "returncode": None,
                        "status": "SKIPPED",
                        "reason": "model_review_packet_not_fresh_for_current_execution",
                    }
                )
                break
            packet_gate_plan = plan.get("model_review_packet_gate") or {}
            packet_gate = _model_review_packet_gate(
                packet_path=Path(str(packet_gate_plan.get("path") or "")),
                evaluation_report_path=Path(
                    str(packet_gate_plan.get("evaluation_report_path") or "")
                ),
                dataset_path=Path(str(packet_gate_plan.get("dataset_path") or "")),
                challenger_review_path=Path(
                    str(packet_gate_plan.get("challenger_review_path") or "")
                )
                if packet_gate_plan.get("challenger_review_path")
                else None,
            )
            if packet_gate.get("status") != "READY":
                results.append(
                    {
                        "name": step_name,
                        "returncode": None,
                        "status": "SKIPPED",
                        "reason": "model_review_packet_gate_not_ready",
                        "model_review_packet_gate": packet_gate,
                    }
                )
                break
        report_path = _step_output_report_path(plan, step_name)
        report_pre_state = (
            _report_pre_state(report_path) if report_path is not None else None
        )
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        result = {
            "name": step_name,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
        if step_name == "approved_persist_ready_subset":
            result["persist_readiness_gate"] = gate
        if step_name == "opt_in_live_odds_capture":
            result["live_odds_readiness_gate"] = odds_gate
        if step_name == "approved_official_label_write":
            result["result_dry_run_report_gate"] = result_gate
            result["label_write_preflight_packet_gate"] = preflight_gate
        if step_name == "result_label_write_readiness_validation":
            readiness_gate_plan = (
                plan.get("label_write_readiness_validation_gate") or {}
            )
            result["result_dry_run_report_gate"] = result_gate
            result["label_write_readiness_validation_gate"] = (
                _label_write_readiness_validation_gate(
                    report_path=Path(str(readiness_gate_plan.get("path") or "")),
                    expected_scope=result_gate.get("expected_scope") or {},
                    approved_dry_run_report=Path(
                        str((plan.get("result_dry_run_report_gate") or {}).get("path") or "")
                    ),
                )
            )
        if step_name == "snapshot_challenger_review":
            review_gate_plan = plan.get("snapshot_challenger_review_gate") or {}
            result["evaluation_report_gate"] = eval_gate
            result["snapshot_challenger_review_gate"] = (
                _snapshot_challenger_review_gate(
                    report_path=Path(str(review_gate_plan.get("path") or "")),
                    dataset_path=Path(str(review_gate_plan.get("dataset_path") or "")),
                )
            )
        if step_name == "model_review_packet":
            packet_gate_plan = plan.get("model_review_packet_gate") or {}
            result["evaluation_report_gate"] = eval_gate
            result["model_review_packet_gate"] = _model_review_packet_gate(
                packet_path=Path(str(packet_gate_plan.get("path") or "")),
                evaluation_report_path=Path(
                    str(packet_gate_plan.get("evaluation_report_path") or "")
                ),
                dataset_path=Path(str(packet_gate_plan.get("dataset_path") or "")),
                challenger_review_path=Path(
                    str(packet_gate_plan.get("challenger_review_path") or "")
                )
                if packet_gate_plan.get("challenger_review_path")
                else None,
            )
        if step_name == "calibration_layer_design":
            design_gate_plan = plan.get("calibration_design_gate") or {}
            result["model_review_packet_gate"] = packet_gate
            result["calibration_design_gate"] = _calibration_design_gate(
                report_path=Path(str(design_gate_plan.get("path") or "")),
                model_review_packet_path=Path(
                    str(design_gate_plan.get("model_review_packet_path") or "")
                ),
            )
        if report_pre_state is not None:
            freshness = _report_freshness_record(report_pre_state)
            result["output_report_freshness"] = freshness
            if completed.returncode == 0 and freshness.get(
                "fresh_for_current_execution"
            ) is not True:
                result["status"] = "FAILED_REPORT_FRESHNESS"
                result["reason"] = _report_freshness_failure_reason(step_name)
                results.append(result)
                break
            if completed.returncode == 0:
                fresh_reports.add(step_name)
        results.append(result)
        if completed.returncode == 0:
            completed_successful_steps.add(step_name)
        if completed.returncode != 0:
            break
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="greyhound_racing_data_writable.db")
    parser.add_argument(
        "--upcoming-dir",
        default=None,
        help="Upcoming CSV directory. Defaults to <run-dir>/upcoming_races for isolation.",
    )
    parser.add_argument("--snapshot-dir", default="artifacts/prediction_snapshots")
    parser.add_argument(
        "--run-dir",
        default="artifacts/prejump_prediction_loop/latest",
    )
    parser.add_argument("--date")
    parser.add_argument(
        "--result-race-id",
        action="append",
        default=[],
        help=(
            "Optional official-result race_id filter for dry-run, readiness "
            "validation, and any later approved label write. Can be repeated."
        ),
    )
    parser.add_argument("--min-minutes", type=float, default=20.0)
    parser.add_argument("--max-minutes", type=float, default=160.0)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument(
        "--challenger-review",
        help="Optional stable report-only challenger review to attach to the model review packet.",
    )
    parser.add_argument(
        "--run-challenger-review",
        action="store_true",
        help="Run a same-execution report-only challenger review from the fresh evaluation dataset.",
    )
    parser.add_argument(
        "--promotion-model-review-packet",
        help=(
            "Optional durable 100+ clean-race report-only model review packet "
            "to use as promotion evidence. This never promotes by itself."
        ),
    )
    parser.add_argument(
        "--promotion-calibration-design",
        help=(
            "Optional durable report-only calibration design matching "
            "--promotion-model-review-packet."
        ),
    )
    parser.add_argument(
        "--report-only-calibration-design",
        help=(
            "Optional calibration_layer_design_v1 JSON to pass into "
            "pre-jump snapshot capture as additive report-only calibrated "
            "probability fields. This never promotes by itself."
        ),
    )
    parser.add_argument(
        "--evaluation-snapshots-manifest",
        help=(
            "Optional text manifest of snapshot paths for the report-only "
            "evaluation step. Use this to make clean-corpus exclusions explicit."
        ),
    )
    parser.add_argument("--approve-live-persist", action="store_true")
    parser.add_argument("--approve-live-odds-capture", action="store_true")
    parser.add_argument("--write-labels-approved", action="store_true")
    parser.add_argument("--approve-promotion", action="store_true")
    parser.add_argument("--execute-ready", action="store_true")
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_loop_plan(args)
    if args.execute_ready:
        plan["execution_results"] = execute_ready_steps(plan)
        plan["post_execution_current_corpus"] = _current_corpus_report(
            Path(args.snapshot_dir),
            plan["current_corpus"]["target_date"],
        )
        plan["post_execution_refresh_report_gate"] = _refresh_report_gate(
            Path(str((plan.get("refresh_report_gate") or {}).get("path") or ""))
        )
        plan["post_execution_protected_resource_counters"] = (
            _protected_resource_counters(
                snapshot_dir=Path(args.snapshot_dir),
                date_text=plan["current_corpus"]["target_date"],
            )
        )
        plan["post_execution_protected_resource_delta"] = (
            _protected_resource_delta_report(
                before=plan["protected_resource_counters"],
                after=plan["post_execution_protected_resource_counters"],
                approvals=plan["approvals"],
                execution_results=plan["execution_results"],
            )
        )
        plan["post_execution_persist_readiness_gate"] = _persist_readiness_gate(
            Path(str((plan.get("persist_readiness_gate") or {}).get("path") or ""))
        )
        persist_packet_plan = plan.get("persist_approval_packet") or {}
        plan["post_execution_persist_approval_packet"] = _persist_approval_packet(
            persist_readiness_gate=plan["post_execution_persist_readiness_gate"],
            protected_resource_counters=plan[
                "post_execution_protected_resource_counters"
            ],
            approvals=plan["approvals"],
            approval_details=plan["approval_provenance"],
            persist_command=list(
                persist_packet_plan.get("planned_persist_command") or []
            ),
            same_run_execute_ready_command=list(
                persist_packet_plan.get(
                    "approved_same_run_execute_ready_command_template"
                )
                or []
            ),
        )
        live_odds_packet_plan = plan.get("live_odds_approval_packet") or {}
        authoritative_capture_report = Path(str(plan.get("run_dir") or "")) / (
            "persist_capture_report.json"
        )
        plan["post_execution_ev_readiness_summary"] = (
            _authoritative_capture_report_ev_summary(authoritative_capture_report)
        )
        approved_persist_step_attempted = any(
            isinstance(result, dict)
            and result.get("name") == "approved_persist_ready_subset"
            for result in plan["execution_results"]
        )
        post_execution_live_odds_ev_summary = (
            plan["post_execution_ev_readiness_summary"]
            if (
                approved_persist_step_attempted
                or _rooted_path(authoritative_capture_report).exists()
            )
            else _ev_summary_from_persist_readiness_gate(
                plan["post_execution_persist_readiness_gate"]
            )
        )
        plan["post_execution_live_odds_approval_packet"] = (
            _live_odds_approval_packet(
                persist_readiness_gate=plan["post_execution_persist_readiness_gate"],
                approvals=plan["approvals"],
                approval_details=plan["approval_provenance"],
                odds_command=list(
                    live_odds_packet_plan.get("planned_odds_command") or []
                ),
                odds_report_path=Path(
                    str(live_odds_packet_plan.get("odds_capture_report_path") or "")
                ),
                same_run_execute_ready_command=list(
                    live_odds_packet_plan.get(
                        "approved_same_run_execute_ready_command_template"
                    )
                    or []
                ),
                ev_summary=post_execution_live_odds_ev_summary,
            )
        )
        plan["post_execution_prediction_preview_report"] = (
            _dry_run_prediction_preview_report(
                Path(str((plan.get("persist_readiness_gate") or {}).get("path") or "")),
                plan["post_execution_persist_readiness_gate"],
            )
        )
        plan["latest_prediction_preview_report"] = plan[
            "post_execution_prediction_preview_report"
        ]
        plan["latest_prediction_preview_report_phase"] = "post_execution"
        result_gate_plan = plan.get("result_dry_run_report_gate") or {}
        expected_scope = result_gate_plan.get("expected_scope")
        plan["post_execution_result_dry_run_report_gate"] = (
            _clean_result_dry_run_report_for_scope(
                report_path=Path(str(result_gate_plan.get("path") or "")),
                expected_scope=expected_scope,
            )
            if isinstance(expected_scope, dict)
            else result_gate_plan
        )
        readiness_gate_plan = (
            plan.get("label_write_readiness_validation_gate") or {}
        )
        plan["post_execution_label_write_readiness_validation_gate"] = (
            _label_write_readiness_validation_gate(
                report_path=Path(str(readiness_gate_plan.get("path") or "")),
                expected_scope=expected_scope,
                approved_dry_run_report=Path(
                    str(result_gate_plan.get("path") or "")
                ),
            )
            if isinstance(expected_scope, dict)
            else readiness_gate_plan
        )
        preflight_gate_plan = plan.get("label_write_preflight_packet_gate") or {}
        plan["post_execution_label_write_preflight_packet_gate"] = (
            _label_write_preflight_packet_gate(
                report_path=Path(str(preflight_gate_plan.get("path") or "")),
                expected_scope=expected_scope,
                label_readiness_report=Path(
                    str(readiness_gate_plan.get("path") or "")
                ),
                result_dry_run_report=Path(
                    str(result_gate_plan.get("path") or "")
                ),
                db_path=Path(str(preflight_gate_plan.get("db_path") or args.db)),
            )
            if isinstance(expected_scope, dict)
            else preflight_gate_plan
        )
        label_packet_plan = plan.get("result_label_approval_packet") or {}
        plan["post_execution_result_label_approval_packet"] = (
            _result_label_approval_packet(
                current_corpus=plan["post_execution_current_corpus"],
                result_dry_run_gate=plan["post_execution_result_dry_run_report_gate"],
                label_write_readiness_gate=plan[
                    "post_execution_label_write_readiness_validation_gate"
                ],
                approvals=plan["approvals"],
                approval_details=plan["approval_provenance"],
                label_write_command=list(
                    label_packet_plan.get("planned_label_write_command") or []
                ),
                label_write_readiness_command=list(
                    label_packet_plan.get(
                        "planned_label_write_readiness_validation_command"
                    )
                    or []
                ),
                label_write_preflight_command=list(
                    label_packet_plan.get(
                        "planned_label_write_preflight_packet_command"
                    )
                    or []
                ),
                label_write_preflight_gate=plan[
                    "post_execution_label_write_preflight_packet_gate"
                ],
                same_run_execute_ready_command=list(
                    label_packet_plan.get(
                        "approved_same_run_execute_ready_command_template"
                    )
                    or []
                ),
            )
        )
        plan["post_execution_steps"] = _steps_with_updated_approval_states(
            list(plan.get("steps") or []),
            persist_packet=plan["post_execution_persist_approval_packet"],
            live_odds_packet=plan["post_execution_live_odds_approval_packet"],
            result_label_packet=plan["post_execution_result_label_approval_packet"],
            label_write_preflight_gate=plan[
                "post_execution_label_write_preflight_packet_gate"
            ],
        )
        eval_gate_plan = plan.get("evaluation_report_gate") or {}
        plan["post_execution_evaluation_report_gate"] = _evaluation_report_gate(
            report_path=Path(str(eval_gate_plan.get("path") or "")),
            dataset_path=Path(str(eval_gate_plan.get("dataset_path") or ""))
            if eval_gate_plan.get("dataset_path")
            else None,
        )
        review_gate_plan = plan.get("snapshot_challenger_review_gate") or {}
        plan["post_execution_snapshot_challenger_review_gate"] = (
            _snapshot_challenger_review_gate(
                report_path=Path(str(review_gate_plan.get("path") or "")),
                dataset_path=Path(str(review_gate_plan.get("dataset_path") or "")),
            )
            if review_gate_plan.get("path")
            else review_gate_plan
        )
        packet_gate_plan = plan.get("model_review_packet_gate") or {}
        plan["post_execution_model_review_packet_gate"] = _model_review_packet_gate(
            packet_path=Path(str(packet_gate_plan.get("path") or "")),
            evaluation_report_path=Path(
                str(packet_gate_plan.get("evaluation_report_path") or "")
            ),
            dataset_path=Path(str(packet_gate_plan.get("dataset_path") or "")),
            challenger_review_path=Path(
                str(packet_gate_plan.get("challenger_review_path") or "")
            )
            if packet_gate_plan.get("challenger_review_path")
            else None,
        )
        design_gate_plan = plan.get("calibration_design_gate") or {}
        plan["post_execution_calibration_design_gate"] = _calibration_design_gate(
            report_path=Path(str(design_gate_plan.get("path") or "")),
            model_review_packet_path=Path(
                str(design_gate_plan.get("model_review_packet_path") or "")
            ),
        )
        promotion_packet_gate_plan = (
            plan.get("promotion_model_review_packet_gate") or {}
        )
        promotion_packet_path = promotion_packet_gate_plan.get("path")
        plan["post_execution_promotion_model_review_packet_gate"] = (
            _promotion_model_review_packet_gate(Path(str(promotion_packet_path)))
            if promotion_packet_path
            else promotion_packet_gate_plan
        )
        promotion_design_gate_plan = (
            plan.get("promotion_calibration_design_gate") or {}
        )
        promotion_design_path = promotion_design_gate_plan.get("path")
        promotion_design_packet_path = promotion_design_gate_plan.get(
            "model_review_packet_path"
        )
        plan["post_execution_promotion_calibration_design_gate"] = (
            _promotion_calibration_design_gate(
                report_path=Path(str(promotion_design_path)),
                model_review_packet_path=Path(str(promotion_design_packet_path))
                if promotion_design_packet_path
                else None,
            )
            if promotion_design_path
            else promotion_design_gate_plan
        )
        plan["post_execution_promotion_readiness_gate"] = (
            _promotion_readiness_gate(
                approvals=plan["approvals"],
                evaluation_report_gate=plan[
                    "post_execution_evaluation_report_gate"
                ],
                snapshot_challenger_review_gate=plan[
                    "post_execution_snapshot_challenger_review_gate"
                ],
                model_review_packet_gate=plan[
                    "post_execution_model_review_packet_gate"
                ],
                calibration_design_gate=plan[
                    "post_execution_calibration_design_gate"
                ],
                promotion_model_review_packet_gate=plan[
                    "post_execution_promotion_model_review_packet_gate"
                ],
                promotion_calibration_design_gate=plan[
                    "post_execution_promotion_calibration_design_gate"
                ],
            )
        )
        plan["post_execution_milestone_completion_audit"] = (
            _milestone_completion_audit(
                approvals=plan["approvals"],
                current_corpus=plan["post_execution_current_corpus"],
                persist_readiness_gate=plan["post_execution_persist_readiness_gate"],
                result_dry_run_gate=plan[
                    "post_execution_result_dry_run_report_gate"
                ],
                evaluation_report_gate=plan[
                    "post_execution_evaluation_report_gate"
                ],
                promotion_readiness_gate=plan[
                    "post_execution_promotion_readiness_gate"
                ],
            )
        )
        plan["post_execution_operator_next_action"] = (
            _operator_next_action_report(
                approvals=plan["approvals"],
                current_corpus=plan["post_execution_current_corpus"],
                persist_packet=plan["post_execution_persist_approval_packet"],
                live_odds_packet=plan["post_execution_live_odds_approval_packet"],
                result_label_packet=plan[
                    "post_execution_result_label_approval_packet"
                ],
                label_write_readiness_gate=plan[
                    "post_execution_label_write_readiness_validation_gate"
                ],
                label_write_preflight_gate=plan[
                    "post_execution_label_write_preflight_packet_gate"
                ],
                milestone_audit=plan[
                    "post_execution_milestone_completion_audit"
                ],
                promotion_readiness_gate=plan[
                    "post_execution_promotion_readiness_gate"
                ],
                refresh_report_gate=plan["post_execution_refresh_report_gate"],
                safe_persist_packet_refresh_sequence=[
                    {
                        "name": step.get("name"),
                        "command": list(step.get("command") or []),
                        "write_scope": step.get("write_scope"),
                    }
                    for step in plan.get("post_execution_steps", [])
                    if step.get("name")
                    in {
                        "fresh_refresh_current_window",
                        "validate_current_upcoming_contract",
                        "dry_run_prejump_capture",
                    }
                ],
                official_result_dry_run_command=next(
                    (
                        list(step.get("command") or [])
                        for step in plan.get("post_execution_steps", [])
                        if step.get("name") == "official_result_ingest_dry_run"
                    ),
                    None,
                ),
                label_write_readiness_validation_command=next(
                    (
                        list(step.get("command") or [])
                        for step in plan.get("post_execution_steps", [])
                        if step.get("name")
                        == "result_label_write_readiness_validation"
                    ),
                    None,
                ),
                label_write_preflight_packet_command=next(
                    (
                        list(step.get("command") or [])
                        for step in plan.get("post_execution_steps", [])
                        if step.get("name") == "label_write_preflight_packet"
                    ),
                    None,
                ),
            )
        )
    text = json.dumps(plan, indent=2, sort_keys=True)
    print(text)
    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
