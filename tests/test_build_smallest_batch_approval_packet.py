import json
import sqlite3
from pathlib import Path

import pytest

import scripts.build_missing_runner_insert_policy_packet as output_guard
from scripts.build_smallest_batch_approval_packet import (
    INSERT_GUARD_BLOCKED,
    INSERT_GUARD_CLEAR,
    UPDATE_GUARD_MATCH,
    build_smallest_batch_approval_packet,
    main,
)


WRITES_PERFORMED = {
    "db_write": False,
    "label_write": False,
    "metadata_write": False,
    "official_fetch": False,
    "snapshot_mutation": False,
    "manifest_mutation": False,
    "dataset_regeneration": False,
    "model_training": False,
    "model_persistence": False,
    "registry_mutation": False,
    "promotion": False,
    "tgr_enablement": False,
    "betting_decision": False,
    "ev_action": False,
}


def _db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        create table dog_race_data (
            race_id text,
            dog_name text,
            dog_clean_name text,
            box_number integer,
            finish_position integer,
            placing integer,
            scraped_finish_position text,
            data_source text
        )
        """
    )
    conn.executemany(
        "insert into dog_race_data values (?,?,?,?,?,?,?,?)",
        [
            ("R1", "Alpha", "Alpha", 1, 1, 1, "1", None),
            ("R1", "Beta", "Beta", 2, 2, 2, "2", None),
        ],
    )
    conn.commit()
    conn.close()
    return path


def _update_policy(path: Path, db: Path) -> Path:
    payload = {
        "schema_version": "non_terminal_update_policy_manifest_v1",
        "status": "REPORT_ONLY_NON_TERMINAL_UPDATE_POLICY_MANIFEST",
        "report_only": True,
        "safe_to_write_now": False,
        "writes_performed": dict(WRITES_PERFORMED),
        "source_evidence": {"db": str(db)},
        "proposed_update_rows": [
            {
                "update_id": "R1::beta::update_box_finish",
                "race_id": "R1",
                "db_rowid": 2,
                "dog_name_key": "beta",
                "db_dog_name": "Beta",
                "official_dog_name": "Beta",
                "current_box_number": 2,
                "proposed_box_number": 4,
                "current_finish_position": 2,
                "proposed_finish_position": 4,
                "current_placing": 2,
                "proposed_placing": 4,
                "current_scraped_finish_position": "2",
                "proposed_scraped_finish_position": "4",
                "safe_to_apply_now": False,
            }
        ],
        "deferred_insert_rows": [
            {
                "candidate_id": "R1::gamma",
                "race_id": "R1",
                "official_dog_name": "Gamma",
                "name_key": "gamma",
                "box_number": 2,
                "finish_position": 3,
                "insert_values": {
                    "race_id": "R1",
                    "dog_name": "Gamma",
                    "dog_clean_name": "Gamma",
                    "box_number": 2,
                    "finish_position": 3,
                    "placing": 3,
                    "scraped_finish_position": "3",
                    "data_source": "thedogs_official",
                },
                "blocking_update_ids": ["R1::beta::update_box_finish"],
                "safe_to_apply_now": False,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _post_update_forecast(path: Path, db: Path, update_policy: Path) -> Path:
    payload = {
        "schema_version": "post_update_label_gate_forecast_v1",
        "status": "REPORT_ONLY_POST_UPDATE_LABEL_GATE_FORECAST",
        "report_only": True,
        "safe_to_write_now": False,
        "writes_performed": dict(WRITES_PERFORMED),
        "source_evidence": {
            "db": str(db),
            "update_policy_packet": str(update_policy),
        },
        "summary": {
            "smallest_batch_race_id": "R1",
            "smallest_batch_update_count": 1,
            "smallest_batch_insert_count": 1,
            "smallest_batch_metadata_action_count": 1,
            "smallest_batch_total_repair_operation_count": 3,
        },
        "forecast_rows": [
            {
                "race_id": "R1",
                "post_update_gate": "POST_UPDATE_DUPLICATE_GUARD_CLEAR_LABEL_PREFLIGHT_REQUIRED",
                "batch_candidate_rank": 1,
                "proposed_update_count": 1,
                "deferred_insert_count": 1,
                "metadata_policy_action_count": 1,
                "total_repair_operation_count": 3,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _inputs(tmp_path: Path):
    db = _db(tmp_path / "greyhound.sqlite")
    update_policy = _update_policy(tmp_path / "update_policy.json", db)
    post_update = _post_update_forecast(tmp_path / "post_update.json", db, update_policy)
    return db, update_policy, post_update


def test_smallest_batch_packet_freezes_exact_guards_without_writes(tmp_path: Path):
    _db_path, _update_policy, post_update = _inputs(tmp_path)

    packet = build_smallest_batch_approval_packet(post_update_forecast_packet_path=post_update)

    assert packet["schema_version"] == "smallest_batch_approval_packet_v1"
    assert packet["status"] == "REPORT_ONLY_SMALLEST_BATCH_APPROVAL_PACKET"
    assert packet["safe_to_write_now"] is False
    assert packet["writes_performed"]["db_write"] is False
    assert packet["summary"]["target_race_id"] == "R1"
    assert packet["summary"]["update_operation_count"] == 1
    assert packet["summary"]["deferred_insert_operation_count"] == 1
    assert packet["summary"]["metadata_action_count"] == 1
    assert packet["summary"]["update_guard_match_count"] == 1
    assert packet["summary"]["update_guard_mismatch_count"] == 0
    assert packet["summary"]["current_duplicate_guard_hit_count"] == 1
    assert packet["summary"]["simulated_duplicate_guard_hit_count"] == 0
    assert packet["summary"]["exact_batch_review_ready_after_backup_and_explicit_approval"] is True
    assert packet["approval_gate"]["approved_here"] is False

    update = packet["update_guard_rows"][0]
    assert update["guard_status"] == UPDATE_GUARD_MATCH
    assert update["db_rowid"] == 2
    insert = packet["insert_guard_rows"][0]
    assert insert["current_duplicate_guard_status"] == INSERT_GUARD_BLOCKED
    assert insert["current_duplicate_guard_hit_rowids"] == [2]
    assert insert["simulated_duplicate_guard_status"] == INSERT_GUARD_CLEAR


def test_smallest_batch_cli_writes_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(output_guard, "ROOT", tmp_path)
    _db_path, _update_policy, post_update = _inputs(tmp_path)
    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/smallest_batch"

    exit_code = main(
        [
            "--post-update-forecast-packet",
            str(post_update),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    payload = json.loads((output_dir / "smallest_batch_approval_packet.json").read_text())
    assert payload["summary"]["target_race_id"] == "R1"
    assert (output_dir / "smallest_batch_update_guards.csv").exists()
    assert (output_dir / "smallest_batch_insert_guards.csv").exists()
    assert "No DB rows" in (output_dir / "SUMMARY.md").read_text(encoding="utf-8")


def test_smallest_batch_rejects_absolute_output_outside_repo(tmp_path: Path):
    outside = tmp_path.parent / "artifacts/full_evidence_orchestration_20260525/smallest_batch"

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        output_guard._assert_output_dir_safe(outside, root=tmp_path)


def test_smallest_batch_rejects_in_repo_non_artifact_output(tmp_path: Path):
    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        output_guard._assert_output_dir_safe(tmp_path / "reports/smallest_batch", root=tmp_path)
