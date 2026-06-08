import json
import sqlite3
from pathlib import Path

from scripts.build_label_write_rehearsal_packet import build_packet


RACE_ID = "Race 9 - LADBROKES-Q-STRAIGHT - 2026-06-01"
BOX_ORDER = [3, 2, 4, 1, 7, 6, 8, 5]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_db(
    path: Path,
    *,
    labeled: bool,
    live_contaminated: bool = False,
    bad_position: bool = False,
) -> Path:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE dog_race_data (
                race_id TEXT,
                dog_name TEXT,
                box_number INTEGER,
                finish_position INTEGER,
                placing INTEGER,
                scraped_finish_position TEXT,
                data_source TEXT
            )
            """
        )
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
        if labeled:
            conn.execute(
                """
                INSERT INTO race_metadata
                    (race_id, results_status, winner_name, winner_source)
                VALUES (?, 'resulted', 'Rays Attack', 'thedogs_official')
                """,
                (RACE_ID,),
            )
            for position, box_number in enumerate(BOX_ORDER, start=1):
                finish_position = 99 if bad_position and box_number == 3 else position
                conn.execute(
                    """
                    INSERT INTO dog_race_data
                        (race_id, dog_name, box_number, finish_position, placing,
                         scraped_finish_position, data_source)
                    VALUES (?, ?, ?, ?, ?, ?, 'thedogs_official')
                    """,
                    (
                        RACE_ID,
                        f"Dog {box_number}",
                        box_number,
                        finish_position,
                        finish_position,
                        str(finish_position),
                    ),
                )
        else:
            conn.execute(
                """
                INSERT INTO race_metadata
                    (race_id, results_status, winner_name, winner_source)
                VALUES (?, 'pending', NULL, NULL)
                """,
                (RACE_ID,),
            )
            if live_contaminated:
                conn.execute(
                    """
                    INSERT INTO dog_race_data
                        (race_id, dog_name, box_number, finish_position, placing,
                         scraped_finish_position, data_source)
                    VALUES (?, 'Rays Attack', 3, 1, 1, '1', 'thedogs_official')
                    """,
                    (RACE_ID,),
                )
    return path


def _scope(db_path: Path) -> dict:
    return {
        "date": "2026-06-01",
        "db_path": str(db_path.resolve()),
        "race_ids": [RACE_ID],
        "require_ready_snapshot": True,
        "snapshot_dir": str((db_path.parent / "snapshots").resolve()),
        "upcoming_dir": str((db_path.parent / "upcoming").resolve()),
    }


def _live_preflight(live_db: Path) -> dict:
    return {
        "schema_version": "label_write_preflight_packet_v1",
        "status": "READY_FOR_EXPLICIT_LABEL_WRITE_APPROVAL",
        "approval_gate": {
            "approved": False,
            "required_env_var": "APPROVE_RESULT_LABEL_WRITE",
            "required_cli_flag": "--write-labels-approved",
        },
        "writes_performed": {"result_label_write": False},
        "pre_write_db_state": {
            "db_path": str(live_db.resolve()),
            "race_ids": [RACE_ID],
            "result_free_before_write": True,
        },
    }


def _dry_run_report(copy_db: Path) -> dict:
    return {
        "schema_version": "official_result_ingest_report_v1",
        "status": "SUCCESS",
        "dry_run": True,
        "clean_for_label_write": True,
        "scope": _scope(copy_db),
        "ingested_count": 1,
        "failed_count": 0,
        "backup_path": None,
        "result_label_write_approval": {"approved": False},
        "ingested": [
            {
                "race_id": RACE_ID,
                "source": "thedogs_official",
                "status": "resulted",
                "winner_name": "Rays Attack",
                "box_order": BOX_ORDER,
                "dry_run": True,
            }
        ],
    }


def _readiness_report(copy_db: Path, dry_run_path: Path) -> dict:
    return {
        "schema_version": "result_label_write_readiness_validation_v1",
        "status": "READY_FOR_EXPLICIT_APPROVAL",
        "scope": _scope(copy_db),
        "write_performed": False,
        "result_label_write_approval": {"approved": False},
        "dry_run_report_gate": {
            "approved": True,
            "status": "approved",
            "report_path": str(dry_run_path),
            "resolved_report_path": str(dry_run_path.resolve()),
        },
    }


def _write_report(copy_db: Path, backup_db: Path) -> dict:
    return {
        "schema_version": "official_result_ingest_report_v1",
        "status": "SUCCESS",
        "dry_run": False,
        "clean_for_label_write": False,
        "scope": _scope(copy_db),
        "ingested_count": 1,
        "failed_count": 0,
        "backup_path": str(backup_db.resolve()),
        "result_label_write_approval": {"approved": True},
        "ingested": [
            {
                "race_id": RACE_ID,
                "source": "thedogs_official",
                "status": "resulted",
                "winner_name": "Rays Attack",
                "box_order": BOX_ORDER,
            }
        ],
    }


def _evaluation_report() -> dict:
    return {
        "status": "SUCCESS",
        "runner_rows_scored": 314,
        "model_quality_diagnosis": {
            "clean_official_races_evaluated": 28,
            "clean_official_runner_rows_evaluated": 216,
            "clean_official_snapshot_instances_evaluated": 30,
            "retrain_gate": {"status": "NOT_READY", "action_taken": "none"},
        },
    }


def _model_review_packet() -> dict:
    return {
        "schema_version": "model_review_packet_v1",
        "status": "NOT_READY",
        "failures": ["insufficient_clean_official_races"],
        "review_gate": {
            "clean_official_evaluated_races": 28,
            "clean_official_runner_rows": 216,
            "clean_official_snapshot_instances": 30,
        },
        "promotion_control": {
            "action_taken": "none",
            "promotion_allowed": False,
            "registry_mutation_allowed": False,
        },
    }


def _write_inputs(tmp_path: Path, *, live_contaminated: bool = False, bad_position: bool = False):
    live_db = _make_db(tmp_path / "live.sqlite", labeled=False, live_contaminated=live_contaminated)
    copy_db = _make_db(tmp_path / "copy.sqlite", labeled=True, bad_position=bad_position)
    backup_db = _make_db(tmp_path / "backup.sqlite", labeled=False)
    live_preflight_path = tmp_path / "live_preflight.json"
    dry_path = tmp_path / "copy_dry_run.json"
    readiness_path = tmp_path / "copy_readiness.json"
    write_path = tmp_path / "copy_write.json"
    evaluation_path = tmp_path / "copy_evaluation.json"
    model_path = tmp_path / "copy_model_review.json"
    _write_json(live_preflight_path, _live_preflight(live_db))
    _write_json(dry_path, _dry_run_report(copy_db))
    _write_json(readiness_path, _readiness_report(copy_db, dry_path))
    _write_json(write_path, _write_report(copy_db, backup_db))
    _write_json(evaluation_path, _evaluation_report())
    _write_json(model_path, _model_review_packet())
    return live_db, copy_db, backup_db, live_preflight_path, dry_path, readiness_path, write_path, evaluation_path, model_path


def test_label_write_rehearsal_packet_ready_for_live_approval(tmp_path):
    inputs = _write_inputs(tmp_path)

    packet = build_packet(
        live_db_path=inputs[0],
        copied_db_path=inputs[1],
        backup_db_path=inputs[2],
        live_preflight_path=inputs[3],
        copy_dry_run_report_path=inputs[4],
        copy_readiness_path=inputs[5],
        copy_write_report_path=inputs[6],
        copy_evaluation_report_path=inputs[7],
        copy_model_review_packet_path=inputs[8],
    )

    assert packet["status"] == "READY_FOR_LIVE_LABEL_WRITE_APPROVAL"
    assert packet["failures"] == []
    assert packet["approval_gate"]["approved"] is False
    assert packet["writes_performed"]["live_db_result_label_write"] is False
    assert packet["writes_performed"]["copied_db_result_label_write"] is True
    assert packet["copy_evaluation_summary"]["clean_official_races_evaluated"] == 28
    assert packet["copy_model_review_summary"]["promotion_allowed"] is False


def test_label_write_rehearsal_packet_fails_if_live_db_has_labels(tmp_path):
    inputs = _write_inputs(tmp_path, live_contaminated=True)

    packet = build_packet(
        live_db_path=inputs[0],
        copied_db_path=inputs[1],
        backup_db_path=inputs[2],
        live_preflight_path=inputs[3],
        copy_dry_run_report_path=inputs[4],
        copy_readiness_path=inputs[5],
        copy_write_report_path=inputs[6],
        copy_evaluation_report_path=inputs[7],
        copy_model_review_packet_path=inputs[8],
    )

    assert packet["status"] == "NOT_READY"
    assert f"live_db_has_result_rows:{RACE_ID}" in packet["failures"]


def test_label_write_rehearsal_packet_fails_on_copied_position_mismatch(tmp_path):
    inputs = _write_inputs(tmp_path, bad_position=True)

    packet = build_packet(
        live_db_path=inputs[0],
        copied_db_path=inputs[1],
        backup_db_path=inputs[2],
        live_preflight_path=inputs[3],
        copy_dry_run_report_path=inputs[4],
        copy_readiness_path=inputs[5],
        copy_write_report_path=inputs[6],
        copy_evaluation_report_path=inputs[7],
        copy_model_review_packet_path=inputs[8],
    )

    assert packet["status"] == "NOT_READY"
    assert f"copied_db_finish_position_mismatch:{RACE_ID}:3" in packet["failures"]
