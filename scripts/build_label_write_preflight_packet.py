#!/usr/bin/env python3
"""Build a no-write preflight packet for approved official result label writes.

The packet consolidates label-readiness, official dry-run result evidence,
loop-plan gates, and current DB state. It intentionally does not write labels,
does not run the planned ingest command, and does not mutate model, registry,
config, snapshot, odds, or betting surfaces.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "label_write_preflight_packet_v1"
READY_STATUS = "READY_FOR_EXPLICIT_LABEL_WRITE_APPROVAL"
LABEL_READINESS_SCHEMA = "result_label_write_readiness_validation_v1"
RESULT_REPORT_SCHEMA = "official_result_ingest_report_v1"
LOOP_PLAN_SCHEMA = "prejump_prediction_loop_plan_v1"
REQUIRED_FLAG = "--write-labels-approved"
REQUIRED_ENV = "APPROVE_RESULT_LABEL_WRITE"
REQUIRED_LOOP_PACKET_STATUS = "AWAITING_EXPLICIT_APPROVAL_READY_FOR_LABEL_WRITE"
REQUIRED_OPERATOR_NEXT_STATUS = "APPROVAL_REQUIRED_FOR_OFFICIAL_LABEL_WRITE"
ALLOWED_OPERATOR_NEXT_STATUSES = {
    REQUIRED_OPERATOR_NEXT_STATUS,
    "RUN_LABEL_WRITE_PREFLIGHT_PACKET",
}
WRITE_SCOPE = "official_result_label_rows_with_pre_write_backup"
OFFICIAL_RESULT_SOURCE = "thedogs_official"

WRITES_PERFORMED = {
    "result_label_write": False,
    "snapshot_persist": False,
    "live_odds_capture": False,
    "model_artifact_write": False,
    "registry_mutation": False,
    "production_config_write": False,
    "refresh_signal_write": False,
    "retrain": False,
    "betting": False,
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("json_root_not_object")
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _append_failure(condition: bool, failures: list[str], name: str) -> None:
    if condition:
        failures.append(name)


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _list(value) if str(item)]


def _race_ids_from_scope(scope: Mapping[str, Any]) -> list[str]:
    return sorted({race_id.strip() for race_id in _string_list(scope.get("race_ids"))})


def _normalise_path(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(Path(str(value)).expanduser().resolve())


def _normalise_scope(scope: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "date": scope.get("date"),
        "db_path": _normalise_path(scope.get("db_path")),
        "upcoming_dir": _normalise_path(scope.get("upcoming_dir")),
        "snapshot_dir": _normalise_path(scope.get("snapshot_dir")),
        "race_ids": _race_ids_from_scope(scope),
        "require_ready_snapshot": bool(scope.get("require_ready_snapshot")),
    }


def _scopes_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return _normalise_scope(left) == _normalise_scope(right)


def _command_contains(command: Any, token: str) -> bool:
    return isinstance(command, list) and token in command


def _command_is_write_template(command: Any) -> bool:
    return (
        isinstance(command, list)
        and REQUIRED_FLAG in command
        and "--dry-run" not in command
        and "--validate-label-write-readiness" not in command
        and any(str(part).endswith("scripts/ingest_results_for_date.py") for part in command)
    )


def _same_path(left: Any, right: Path) -> bool:
    if not left:
        return False
    try:
        return Path(str(left)).expanduser().resolve() == right.expanduser().resolve()
    except OSError:
        return False


def _extract_ingested_results(result_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in _list(result_report.get("ingested")):
        if not isinstance(item, Mapping):
            continue
        results.append(
            {
                "race_id": item.get("race_id"),
                "source": item.get("source"),
                "status": item.get("status"),
                "winner_name": item.get("winner_name"),
                "box_order": item.get("box_order"),
                "dry_run": item.get("dry_run"),
            }
        )
    return results


def _validate_label_readiness(
    readiness: Mapping[str, Any],
    *,
    label_readiness_path: Path,
    result_dry_run_path: Path,
    failures: list[str],
) -> dict[str, Any]:
    dry_gate = _mapping(readiness.get("dry_run_report_gate"))
    approval = _mapping(readiness.get("result_label_write_approval"))
    planned_command = readiness.get("planned_command_if_approved")
    expected_scope = _mapping(dry_gate.get("expected_scope"))
    observed_scope = _mapping(dry_gate.get("observed_scope"))

    _append_failure(
        readiness.get("schema_version") != LABEL_READINESS_SCHEMA,
        failures,
        "label_readiness_schema_mismatch",
    )
    _append_failure(
        readiness.get("status") != "READY_FOR_EXPLICIT_APPROVAL",
        failures,
        "label_readiness_not_ready_for_explicit_approval",
    )
    _append_failure(
        readiness.get("write_performed") is not False,
        failures,
        "label_readiness_write_already_performed",
    )
    _append_failure(
        readiness.get("approval_required") is not True,
        failures,
        "label_readiness_approval_not_required",
    )
    _append_failure(
        readiness.get("required_cli_flag") != REQUIRED_FLAG,
        failures,
        "label_readiness_required_cli_flag_mismatch",
    )
    _append_failure(
        readiness.get("required_env_var") != REQUIRED_ENV,
        failures,
        "label_readiness_required_env_var_mismatch",
    )
    _append_failure(
        approval.get("approved") is not False,
        failures,
        "label_write_approval_already_present",
    )
    _append_failure(
        dry_gate.get("approved") is not True,
        failures,
        "label_readiness_dry_run_gate_not_approved",
    )
    if expected_scope or observed_scope:
        _append_failure(
            not _scopes_match(expected_scope, observed_scope),
            failures,
            "label_readiness_scope_mismatch",
        )
    _append_failure(
        not _command_is_write_template(planned_command),
        failures,
        "label_readiness_planned_command_not_explicit_label_write",
    )
    _append_failure(
        not _same_path(dry_gate.get("resolved_report_path") or dry_gate.get("report_path"), result_dry_run_path),
        failures,
        "label_readiness_result_dry_run_report_path_mismatch",
    )
    _append_failure(
        _safe_int(readiness.get("candidate_count_loaded_for_write_scope")) <= 0,
        failures,
        "label_readiness_no_loaded_candidates",
    )
    _append_failure(
        not _race_ids_from_scope(_mapping(readiness.get("scope"))),
        failures,
        "label_readiness_scope_has_no_race_ids",
    )
    _append_failure(
        not label_readiness_path.exists(),
        failures,
        "label_readiness_path_missing",
    )

    return {
        "status": readiness.get("status"),
        "write_performed": readiness.get("write_performed"),
        "approval_required": readiness.get("approval_required"),
        "approval_approved": approval.get("approved"),
        "candidate_count_loaded_for_write_scope": readiness.get(
            "candidate_count_loaded_for_write_scope"
        ),
        "candidate_race_ids_loaded_for_write_scope": _string_list(
            readiness.get("candidate_race_ids_loaded_for_write_scope")
        ),
        "scope": _normalise_scope(_mapping(readiness.get("scope"))),
        "dry_run_report_gate": {
            "approved": dry_gate.get("approved"),
            "status": dry_gate.get("status"),
            "report_path": dry_gate.get("report_path"),
            "resolved_report_path": dry_gate.get("resolved_report_path"),
        },
    }


def _validate_result_report(
    result_report: Mapping[str, Any],
    *,
    failures: list[str],
) -> dict[str, Any]:
    ingested = _extract_ingested_results(result_report)
    ingested_count = _safe_int(result_report.get("ingested_count"))

    _append_failure(
        result_report.get("schema_version") != RESULT_REPORT_SCHEMA,
        failures,
        "result_dry_run_schema_mismatch",
    )
    _append_failure(
        result_report.get("status") != "SUCCESS",
        failures,
        "result_dry_run_status_not_success",
    )
    _append_failure(
        result_report.get("dry_run") is not True,
        failures,
        "result_report_not_dry_run",
    )
    _append_failure(
        result_report.get("clean_for_label_write") is not True,
        failures,
        "result_dry_run_not_clean_for_label_write",
    )
    _append_failure(
        _safe_int(result_report.get("failed_count")) != 0,
        failures,
        "result_dry_run_failed_count_nonzero",
    )
    _append_failure(
        ingested_count <= 0 or len(ingested) <= 0,
        failures,
        "result_dry_run_no_ingested_results",
    )
    _append_failure(
        ingested_count != len(ingested),
        failures,
        "result_dry_run_ingested_count_mismatch",
    )
    _append_failure(
        bool(result_report.get("label_write_blockers")),
        failures,
        "result_dry_run_has_label_write_blockers",
    )
    _append_failure(
        result_report.get("backup_path") is not None,
        failures,
        "result_dry_run_backup_path_present",
    )

    for item in ingested:
        race_id = item.get("race_id") or "unknown"
        _append_failure(
            item.get("dry_run") is not True,
            failures,
            f"result_item_not_dry_run:{race_id}",
        )
        _append_failure(
            item.get("source") != OFFICIAL_RESULT_SOURCE,
            failures,
            f"result_item_not_official:{race_id}",
        )
        _append_failure(
            item.get("status") != "resulted",
            failures,
            f"result_item_not_resulted:{race_id}",
        )
        _append_failure(
            not item.get("winner_name"),
            failures,
            f"result_item_missing_winner:{race_id}",
        )
        _append_failure(
            not isinstance(item.get("box_order"), list) or not item.get("box_order"),
            failures,
            f"result_item_missing_box_order:{race_id}",
        )

    return {
        "status": result_report.get("status"),
        "dry_run": result_report.get("dry_run"),
        "clean_for_label_write": result_report.get("clean_for_label_write"),
        "candidate_count": result_report.get("candidate_count"),
        "ingested_count": ingested_count,
        "failed_count": _safe_int(result_report.get("failed_count")),
        "scope": _normalise_scope(_mapping(result_report.get("scope"))),
        "ingested": ingested,
    }


def _validate_loop_plan(
    loop_plan: Mapping[str, Any],
    *,
    label_readiness_path: Path,
    result_dry_run_path: Path,
    failures: list[str],
) -> dict[str, Any]:
    result_packet = _mapping(loop_plan.get("result_label_approval_packet"))
    readiness_gate = _mapping(loop_plan.get("label_write_readiness_validation_gate"))
    result_gate = _mapping(loop_plan.get("result_dry_run_report_gate"))
    operator_next = _mapping(loop_plan.get("operator_next_action"))
    gated_actions = _mapping(loop_plan.get("gated_actions_default_blocked"))
    guarantees = _mapping(loop_plan.get("guarantees"))
    approved_command = result_packet.get("approved_label_write_command_template")
    same_run_command = result_packet.get("approved_same_run_execute_ready_command_template")

    _append_failure(
        loop_plan.get("schema_version") != LOOP_PLAN_SCHEMA,
        failures,
        "loop_plan_schema_mismatch",
    )
    _append_failure(
        result_packet.get("schema_version") != "result_label_approval_packet_v1",
        failures,
        "loop_result_label_packet_schema_mismatch",
    )
    _append_failure(
        result_packet.get("status") != REQUIRED_LOOP_PACKET_STATUS,
        failures,
        "loop_result_label_packet_not_awaiting_explicit_approval",
    )
    _append_failure(
        result_packet.get("can_write_labels_now") is not False,
        failures,
        "loop_result_label_packet_can_write_now",
    )
    _append_failure(
        result_packet.get("approval_required") is not True,
        failures,
        "loop_result_label_packet_approval_not_required",
    )
    _append_failure(
        result_packet.get("approval_gate") != REQUIRED_ENV,
        failures,
        "loop_result_label_packet_gate_mismatch",
    )
    _append_failure(
        result_packet.get("required_cli_flag") != REQUIRED_FLAG,
        failures,
        "loop_result_label_packet_required_cli_flag_mismatch",
    )
    _append_failure(
        result_packet.get("required_env_var") != REQUIRED_ENV,
        failures,
        "loop_result_label_packet_required_env_var_mismatch",
    )
    _append_failure(
        result_packet.get("write_scope") != WRITE_SCOPE,
        failures,
        "loop_result_label_packet_write_scope_mismatch",
    )
    _append_failure(
        not _command_is_write_template(approved_command),
        failures,
        "loop_approved_label_write_command_not_explicit_label_write",
    )
    _append_failure(
        not (
            isinstance(same_run_command, list)
            and REQUIRED_FLAG in same_run_command
            and "--execute-ready" in same_run_command
        ),
        failures,
        "loop_same_run_label_write_command_missing_fresh_recheck_gate",
    )
    _append_failure(
        readiness_gate.get("status") != "READY",
        failures,
        "loop_label_readiness_gate_not_ready",
    )
    _append_failure(
        readiness_gate.get("write_performed") is not False,
        failures,
        "loop_label_readiness_gate_write_performed",
    )
    _append_failure(
        result_gate.get("status") != "READY",
        failures,
        "loop_result_dry_run_gate_not_ready",
    )
    _append_failure(
        result_gate.get("clean") is not True,
        failures,
        "loop_result_dry_run_gate_not_clean",
    )
    _append_failure(
        operator_next.get("next_step_status") not in ALLOWED_OPERATOR_NEXT_STATUSES,
        failures,
        "loop_operator_next_status_not_waiting_for_label_approval_or_preflight",
    )
    _append_failure(
        REQUIRED_ENV not in _string_list(operator_next.get("blocked_approval_gates")),
        failures,
        "loop_operator_next_missing_label_write_gate",
    )
    _append_failure(
        gated_actions.get("result_label_write") is not True,
        failures,
        "loop_result_label_write_not_default_blocked",
    )
    for key, failure in (
        ("no_betting", "loop_no_betting_guarantee_missing"),
        ("no_model_promotion", "loop_no_model_promotion_guarantee_missing"),
        ("no_retrain", "loop_no_retrain_guarantee_missing"),
    ):
        _append_failure(guarantees.get(key) is not True, failures, failure)

    _append_failure(
        not _same_path(readiness_gate.get("path"), label_readiness_path),
        failures,
        "loop_label_readiness_path_mismatch",
    )
    _append_failure(
        not _same_path(result_gate.get("path"), result_dry_run_path),
        failures,
        "loop_result_dry_run_path_mismatch",
    )
    if result_packet.get("expected_scope") or result_packet.get("observed_scope"):
        _append_failure(
            not _scopes_match(
                _mapping(result_packet.get("expected_scope")),
                _mapping(result_packet.get("observed_scope")),
            ),
            failures,
            "loop_result_label_packet_scope_mismatch",
        )

    return {
        "packet_status": result_packet.get("status"),
        "readiness_gate_status": readiness_gate.get("status"),
        "result_dry_run_gate_status": result_gate.get("status"),
        "operator_next_status": operator_next.get("next_step_status"),
        "approval_gate": result_packet.get("approval_gate"),
        "write_scope": result_packet.get("write_scope"),
        "gated_actions_default_blocked": dict(gated_actions),
        "guarantees": dict(guarantees),
        "same_run_rechecks": _string_list(
            result_packet.get("same_run_execute_ready_rechecks")
        ),
    }


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _read_pre_write_db_state(
    *,
    db_path: Path,
    race_ids: list[str],
    failures: list[str],
) -> dict[str, Any]:
    resolved = db_path.expanduser().resolve()
    state: dict[str, Any] = {
        "db_path": str(resolved),
        "quick_check": None,
        "race_ids": race_ids,
        "dog_race_data_rows_by_race": {},
        "race_metadata_by_race": {},
        "result_free_before_write": False,
    }
    if not resolved.exists():
        failures.append("db_path_missing")
        return state

    try:
        with sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True) as conn:
            quick_check = conn.execute("PRAGMA quick_check").fetchone()
            state["quick_check"] = quick_check[0] if quick_check else None
            _append_failure(
                state["quick_check"] != "ok",
                failures,
                "db_quick_check_failed",
            )

            for table in ("dog_race_data", "race_metadata"):
                _append_failure(
                    not _table_exists(conn, table),
                    failures,
                    f"db_table_missing:{table}",
                )
            if not _table_exists(conn, "dog_race_data") or not _table_exists(
                conn, "race_metadata"
            ):
                return state

            metadata_columns = _columns(conn, "race_metadata")
            required_metadata_columns = {
                "race_id",
                "results_status",
                "winner_name",
                "winner_source",
            }
            missing_columns = sorted(required_metadata_columns - metadata_columns)
            if missing_columns:
                failures.append(
                    "race_metadata_missing_columns:" + ",".join(missing_columns)
                )
                return state

            result_free = True
            for race_id in race_ids:
                count = conn.execute(
                    "SELECT COUNT(*) FROM dog_race_data WHERE race_id = ?",
                    (race_id,),
                ).fetchone()[0]
                state["dog_race_data_rows_by_race"][race_id] = count
                if count != 0:
                    failures.append(f"pre_write_db_has_result_rows:{race_id}")
                    result_free = False

                row = conn.execute(
                    """
                    SELECT results_status, winner_name, winner_source
                    FROM race_metadata
                    WHERE race_id = ?
                    """,
                    (race_id,),
                ).fetchone()
                if row is None:
                    failures.append(f"race_metadata_missing:{race_id}")
                    result_free = False
                    state["race_metadata_by_race"][race_id] = None
                    continue

                results_status, winner_name, winner_source = row
                state["race_metadata_by_race"][race_id] = {
                    "results_status": results_status,
                    "winner_name": winner_name,
                    "winner_source": winner_source,
                }
                if results_status != "pending":
                    failures.append(f"race_metadata_not_pending:{race_id}")
                    result_free = False
                if winner_name not in (None, ""):
                    failures.append(f"race_metadata_winner_present:{race_id}")
                    result_free = False
                if winner_source not in (None, ""):
                    failures.append(f"race_metadata_winner_source_present:{race_id}")
                    result_free = False
            state["result_free_before_write"] = result_free
    except sqlite3.Error as exc:
        failures.append(f"db_read_failed:{type(exc).__name__}")
    return state


def _combined_scope(
    readiness_summary: Mapping[str, Any],
    result_summary: Mapping[str, Any],
    loop_summary: Mapping[str, Any],
) -> dict[str, Any]:
    readiness_scope = _mapping(readiness_summary.get("scope"))
    result_scope = _mapping(result_summary.get("scope"))
    race_ids = sorted(
        {
            *_race_ids_from_scope(readiness_scope),
            *_race_ids_from_scope(result_scope),
            *[
                str(item.get("race_id")).strip()
                for item in _list(result_summary.get("ingested"))
                if isinstance(item, Mapping) and item.get("race_id")
            ],
        }
    )
    return {
        "date": readiness_scope.get("date") or result_scope.get("date"),
        "db_path": readiness_scope.get("db_path") or result_scope.get("db_path"),
        "upcoming_dir": readiness_scope.get("upcoming_dir")
        or result_scope.get("upcoming_dir"),
        "snapshot_dir": readiness_scope.get("snapshot_dir")
        or result_scope.get("snapshot_dir"),
        "race_ids": race_ids,
        "require_ready_snapshot": bool(
            readiness_scope.get("require_ready_snapshot")
            or result_scope.get("require_ready_snapshot")
        ),
        "loop_packet_status": loop_summary.get("packet_status"),
    }


def build_packet(
    *,
    label_readiness_path: Path,
    result_dry_run_report_path: Path,
    loop_plan_path: Path,
    db_path: Path,
) -> dict[str, Any]:
    label_readiness_resolved = label_readiness_path.expanduser().resolve()
    result_dry_run_resolved = result_dry_run_report_path.expanduser().resolve()
    loop_plan_resolved = loop_plan_path.expanduser().resolve()
    db_resolved = db_path.expanduser().resolve()
    failures: list[str] = []
    warnings: list[str] = []

    try:
        readiness = _load_json(label_readiness_resolved)
    except Exception as exc:
        readiness = {}
        failures.append(f"label_readiness_unreadable:{type(exc).__name__}")
    try:
        result_report = _load_json(result_dry_run_resolved)
    except Exception as exc:
        result_report = {}
        failures.append(f"result_dry_run_report_unreadable:{type(exc).__name__}")
    try:
        loop_plan = _load_json(loop_plan_resolved)
    except Exception as exc:
        loop_plan = {}
        failures.append(f"loop_plan_unreadable:{type(exc).__name__}")

    readiness_summary = _validate_label_readiness(
        readiness,
        label_readiness_path=label_readiness_resolved,
        result_dry_run_path=result_dry_run_resolved,
        failures=failures,
    )
    result_summary = _validate_result_report(result_report, failures=failures)
    loop_summary = _validate_loop_plan(
        loop_plan,
        label_readiness_path=label_readiness_resolved,
        result_dry_run_path=result_dry_run_resolved,
        failures=failures,
    )

    readiness_scope = _mapping(readiness.get("scope"))
    result_scope = _mapping(result_report.get("scope"))
    if readiness_scope and result_scope:
        _append_failure(
            not _scopes_match(readiness_scope, result_scope),
            failures,
            "readiness_result_scope_mismatch",
        )

    race_scope = _combined_scope(readiness_summary, result_summary, loop_summary)
    scoped_race_ids = _string_list(race_scope.get("race_ids"))
    _append_failure(
        not scoped_race_ids,
        failures,
        "preflight_scope_has_no_race_ids",
    )
    if race_scope.get("db_path") and Path(str(race_scope["db_path"])) != db_resolved:
        failures.append("preflight_db_path_mismatch")

    pre_write_db_state = _read_pre_write_db_state(
        db_path=db_resolved,
        race_ids=scoped_race_ids,
        failures=failures,
    )

    ready = not failures
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "status": READY_STATUS if ready else "NOT_READY",
        "failures": failures,
        "warnings": warnings,
        "source_evidence": {
            "label_readiness": str(label_readiness_resolved),
            "result_dry_run_report": str(result_dry_run_resolved),
            "prejump_loop_plan": str(loop_plan_resolved),
            "db": str(db_resolved),
        },
        "race_scope": race_scope,
        "label_readiness_summary": readiness_summary,
        "official_result_summary": result_summary.get("ingested"),
        "loop_gate_summary": loop_summary,
        "pre_write_db_state": pre_write_db_state,
        "approval_gate": {
            "required": True,
            "required_cli_flag": REQUIRED_FLAG,
            "required_env_var": REQUIRED_ENV,
            "approved": False,
            "accepted_approval_forms": [
                REQUIRED_FLAG,
                f"{REQUIRED_ENV}=true",
            ],
        },
        "approved_label_write_command_template": readiness.get(
            "planned_command_if_approved"
        ),
        "same_run_loop_label_write_command_template": _mapping(
            loop_plan.get("result_label_approval_packet")
        ).get("approved_same_run_execute_ready_command_template"),
        "write_scope": WRITE_SCOPE,
        "writes_performed": dict(WRITES_PERFORMED),
        "rollback_expectation": {
            "backup_created_by": "scripts/ingest_results_for_date.py backup_db",
            "backup_path_runtime_pattern": (
                "<db_dir>/archive/db_backups/"
                "<utc>_pre_results_ingest_official_first/pre_op.sqlite"
            ),
            "backup_expected_before_any_label_write": True,
            "restore_requires_operator_review": True,
        },
        "forbidden_without_explicit_approval": [
            "result_label_write",
            "model_retrain_or_promotion",
            "betting",
        ],
        "no_write_preflight_only": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-readiness", required=True)
    parser.add_argument("--result-dry-run-report", required=True)
    parser.add_argument("--loop-plan", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    packet = build_packet(
        label_readiness_path=Path(args.label_readiness),
        result_dry_run_report_path=Path(args.result_dry_run_report),
        loop_plan_path=Path(args.loop_plan),
        db_path=Path(args.db),
    )
    text = json.dumps(packet, indent=2, sort_keys=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if packet.get("status") == READY_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
