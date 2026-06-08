import json
import sqlite3
from pathlib import Path

from scripts.build_label_write_preflight_packet import build_packet


RACE_ID = "Race 9 - LADBROKES-Q-STRAIGHT - 2026-06-01"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_db(tmp_path: Path, *, result_rows: int = 0) -> Path:
    db_path = tmp_path / "greyhound.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE dog_race_data (race_id TEXT)")
        conn.execute(
            """
            CREATE TABLE race_metadata (
                race_id TEXT PRIMARY KEY,
                results_status TEXT,
                winner_name TEXT,
                winner_source TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO race_metadata
                (race_id, results_status, winner_name, winner_source)
            VALUES (?, 'pending', NULL, NULL)
            """,
            (RACE_ID,),
        )
        for _ in range(result_rows):
            conn.execute("INSERT INTO dog_race_data (race_id) VALUES (?)", (RACE_ID,))
    return db_path


def _scope(tmp_path: Path, db_path: Path) -> dict:
    return {
        "date": "2026-06-01",
        "db_path": str(db_path.resolve()),
        "race_ids": [RACE_ID],
        "require_ready_snapshot": True,
        "snapshot_dir": str((tmp_path / "prediction_snapshots").resolve()),
        "upcoming_dir": str((tmp_path / "upcoming_races").resolve()),
    }


def _label_readiness(scope: dict, result_report_path: Path) -> dict:
    return {
        "schema_version": "result_label_write_readiness_validation_v1",
        "status": "READY_FOR_EXPLICIT_APPROVAL",
        "scope": scope,
        "candidate_count_loaded_for_write_scope": 1,
        "candidate_race_ids_loaded_for_write_scope": [RACE_ID],
        "skipped_before_write_scope_validation": [],
        "dry_run_report_gate": {
            "approved": True,
            "status": "approved",
            "report_path": str(result_report_path),
            "resolved_report_path": str(result_report_path.resolve()),
            "expected_scope": scope,
            "observed_scope": scope,
            "expected_candidate_race_ids": [RACE_ID],
            "observed_candidate_race_ids": [RACE_ID],
        },
        "result_label_write_approval": {
            "approved": False,
            "status": "not_approved",
        },
        "approval_required": True,
        "required_cli_flag": "--write-labels-approved",
        "required_env_var": "APPROVE_RESULT_LABEL_WRITE",
        "planned_command_if_approved": [
            "python",
            "scripts/ingest_results_for_date.py",
            "--db",
            scope["db_path"],
            "--date",
            scope["date"],
            "--upcoming-dir",
            scope["upcoming_dir"],
            "--approved-dry-run-report",
            str(result_report_path),
            "--race-id",
            RACE_ID,
            "--write-labels-approved",
            "--output",
            "result_label_write_report_if_approved.json",
        ],
        "write_performed": False,
    }


def _result_report(scope: dict, *, clean: bool = True, source: str = "thedogs_official") -> dict:
    return {
        "schema_version": "official_result_ingest_report_v1",
        "status": "SUCCESS",
        "dry_run": True,
        "scope": scope,
        "candidate_count": 1,
        "candidate_race_ids": [RACE_ID],
        "skipped_count": 0,
        "skipped": [],
        "ingested_count": 1,
        "ingested": [
            {
                "race_id": RACE_ID,
                "dry_run": True,
                "source": source,
                "status": "resulted",
                "winner_name": "Rays Attack",
                "box_order": [3, 2, 4, 1, 7, 6, 8, 5],
            }
        ],
        "failed_count": 0,
        "failed": [],
        "label_write_blockers": [] if clean else [{"reason": "not_clean"}],
        "backup_path": None,
        "result_label_write_approval": {"approved": False},
        "dry_run_report_gate": None,
        "clean_for_label_write": clean,
    }


def _loop_plan(scope: dict, label_path: Path, result_path: Path) -> dict:
    approved_command = [
        "python",
        "scripts/ingest_results_for_date.py",
        "--db",
        scope["db_path"],
        "--date",
        scope["date"],
        "--upcoming-dir",
        scope["upcoming_dir"],
        "--approved-dry-run-report",
        str(result_path),
        "--race-id",
        RACE_ID,
        "--write-labels-approved",
        "--output",
        "result_label_write_report.json",
    ]
    same_run_command = [
        "python",
        "scripts/prejump_prediction_loop.py",
        "--execute-ready",
        "--write-labels-approved",
        "--output",
        "loop_plan_execute_approved_label_write.json",
    ]
    return {
        "schema_version": "prejump_prediction_loop_plan_v1",
        "result_race_ids": [RACE_ID],
        "result_label_approval_packet": {
            "schema_version": "result_label_approval_packet_v1",
            "status": "AWAITING_EXPLICIT_APPROVAL_READY_FOR_LABEL_WRITE",
            "can_write_labels_now": False,
            "approval_required": True,
            "approval_gate": "APPROVE_RESULT_LABEL_WRITE",
            "required_cli_flag": "--write-labels-approved",
            "required_env_var": "APPROVE_RESULT_LABEL_WRITE",
            "write_scope": "official_result_label_rows_with_pre_write_backup",
            "expected_scope": scope,
            "observed_scope": scope,
            "approved_label_write_command_template": approved_command,
            "approved_same_run_execute_ready_command_template": same_run_command,
            "same_run_execute_ready_rechecks": [
                "current_persisted_prejump_corpus",
                "official_result_ingest_dry_run",
                "result_label_approval_gate",
                "official_first_scope_match",
            ],
        },
        "label_write_readiness_validation_gate": {
            "path": str(label_path.resolve()),
            "status": "READY",
            "write_performed": False,
        },
        "result_dry_run_report_gate": {
            "path": str(result_path.resolve()),
            "status": "READY",
            "clean": True,
        },
        "operator_next_action": {
            "next_step_status": "APPROVAL_REQUIRED_FOR_OFFICIAL_LABEL_WRITE",
            "blocked_approval_gates": ["APPROVE_RESULT_LABEL_WRITE"],
        },
        "gated_actions_default_blocked": {"result_label_write": True},
        "guarantees": {
            "no_betting": True,
            "no_model_promotion": True,
            "no_retrain": True,
        },
    }


def _write_inputs(tmp_path: Path, *, result_rows: int = 0) -> tuple[Path, Path, Path, Path]:
    db_path = _make_db(tmp_path, result_rows=result_rows)
    scope = _scope(tmp_path, db_path)
    label_path = tmp_path / "label_readiness.json"
    result_path = tmp_path / "result_dry_run.json"
    loop_path = tmp_path / "loop_plan.json"
    _write_json(result_path, _result_report(scope))
    _write_json(label_path, _label_readiness(scope, result_path))
    _write_json(loop_path, _loop_plan(scope, label_path, result_path))
    return db_path, label_path, result_path, loop_path


def test_label_write_preflight_packet_ready_without_writes(tmp_path):
    db_path, label_path, result_path, loop_path = _write_inputs(tmp_path)

    packet = build_packet(
        label_readiness_path=label_path,
        result_dry_run_report_path=result_path,
        loop_plan_path=loop_path,
        db_path=db_path,
    )

    assert packet["status"] == "READY_FOR_EXPLICIT_LABEL_WRITE_APPROVAL"
    assert packet["failures"] == []
    assert packet["approval_gate"]["approved"] is False
    assert packet["writes_performed"]["result_label_write"] is False
    assert packet["writes_performed"]["betting"] is False
    assert packet["pre_write_db_state"]["result_free_before_write"] is True
    assert packet["official_result_summary"][0]["source"] == "thedogs_official"
    assert "--write-labels-approved" in packet["approved_label_write_command_template"]
    assert "--execute-ready" in packet["same_run_loop_label_write_command_template"]


def test_label_write_preflight_packet_accepts_loop_preflight_next_step(tmp_path):
    db_path, label_path, result_path, loop_path = _write_inputs(tmp_path)
    loop_plan = json.loads(loop_path.read_text(encoding="utf-8"))
    loop_plan["operator_next_action"][
        "next_step_status"
    ] = "RUN_LABEL_WRITE_PREFLIGHT_PACKET"
    _write_json(loop_path, loop_plan)

    packet = build_packet(
        label_readiness_path=label_path,
        result_dry_run_report_path=result_path,
        loop_plan_path=loop_path,
        db_path=db_path,
    )

    assert packet["status"] == "READY_FOR_EXPLICIT_LABEL_WRITE_APPROVAL"
    assert packet["failures"] == []
    assert packet["loop_gate_summary"]["operator_next_status"] == (
        "RUN_LABEL_WRITE_PREFLIGHT_PACKET"
    )


def test_label_write_preflight_fails_when_db_already_has_result_rows(tmp_path):
    db_path, label_path, result_path, loop_path = _write_inputs(tmp_path, result_rows=1)

    packet = build_packet(
        label_readiness_path=label_path,
        result_dry_run_report_path=result_path,
        loop_plan_path=loop_path,
        db_path=db_path,
    )

    assert packet["status"] == "NOT_READY"
    assert f"pre_write_db_has_result_rows:{RACE_ID}" in packet["failures"]
    assert packet["writes_performed"]["result_label_write"] is False


def test_label_write_preflight_fails_when_result_dry_run_is_not_clean_official(
    tmp_path,
):
    db_path = _make_db(tmp_path)
    scope = _scope(tmp_path, db_path)
    label_path = tmp_path / "label_readiness.json"
    result_path = tmp_path / "result_dry_run.json"
    loop_path = tmp_path / "loop_plan.json"
    _write_json(result_path, _result_report(scope, clean=False, source="sportsbet"))
    _write_json(label_path, _label_readiness(scope, result_path))
    _write_json(loop_path, _loop_plan(scope, label_path, result_path))

    packet = build_packet(
        label_readiness_path=label_path,
        result_dry_run_report_path=result_path,
        loop_plan_path=loop_path,
        db_path=db_path,
    )

    assert packet["status"] == "NOT_READY"
    assert "result_dry_run_not_clean_for_label_write" in packet["failures"]
    assert f"result_item_not_official:{RACE_ID}" in packet["failures"]


def test_label_write_preflight_fails_when_loop_is_not_waiting_for_approval(tmp_path):
    db_path, label_path, result_path, loop_path = _write_inputs(tmp_path)
    loop_plan = json.loads(loop_path.read_text(encoding="utf-8"))
    loop_plan["result_label_approval_packet"]["status"] = "READY"
    loop_plan["result_label_approval_packet"]["can_write_labels_now"] = True
    _write_json(loop_path, loop_plan)

    packet = build_packet(
        label_readiness_path=label_path,
        result_dry_run_report_path=result_path,
        loop_plan_path=loop_path,
        db_path=db_path,
    )

    assert packet["status"] == "NOT_READY"
    assert (
        "loop_result_label_packet_not_awaiting_explicit_approval"
        in packet["failures"]
    )
    assert "loop_result_label_packet_can_write_now" in packet["failures"]
