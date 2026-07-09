import json
import sqlite3
from pathlib import Path

import pytest

import scripts.build_missing_runner_insert_policy_packet as output_guard
from scripts.build_post_update_label_gate_forecast import (
    GATE_BATCH_CANDIDATE,
    GATE_INSERT_ONLY_UNMODELED_UPDATES,
    build_post_update_forecast_packet,
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
            ("R2", "Existing", "Existing", 1, 1, 1, "1", None),
        ],
    )
    conn.commit()
    conn.close()
    return path


def _manifest_candidate(race_id: str, name: str, box: int, status: str) -> dict:
    return {
        "candidate_id": f"{race_id}::{name.lower()}",
        "race_id": race_id,
        "official_dog_name": name,
        "name_key": name.lower(),
        "box_number": box,
        "finish_position": 3,
        "insert_values": {
            "race_id": race_id,
            "dog_name": name,
            "dog_clean_name": name,
            "box_number": box,
            "finish_position": 3,
            "placing": 3,
            "scraped_finish_position": "3",
            "extraction_timestamp": "<APPLY_TIME_UTC>",
            "data_source": "thedogs_official",
        },
        "apply_manifest_status": status,
    }


def _manifest(path: Path, policy: Path, forecast: Path) -> Path:
    payload = {
        "schema_version": "non_terminal_repair_apply_manifest_forecast_v1",
        "status": "REPORT_ONLY_NON_TERMINAL_REPAIR_APPLY_MANIFEST_FORECAST",
        "report_only": True,
        "safe_to_write_now": False,
        "writes_performed": dict(WRITES_PERFORMED),
        "source_evidence": {
            "insert_policy_packet": str(policy),
            "post_repair_forecast_packet": str(forecast),
        },
        "race_manifest_rows": [
            {
                "race_id": "R1",
                "review_lane": "P1",
                "changed_dog_update_candidate_count": 1,
            },
            {
                "race_id": "R2",
                "review_lane": "P4",
                "changed_dog_update_candidate_count": 2,
            },
        ],
        "candidate_manifest_rows": [
            _manifest_candidate("R1", "Gamma", 2, "DEFERRED_RACE_HAS_OTHER_DUPLICATE_GUARD_HIT"),
            _manifest_candidate(
                "R2",
                "Delta",
                3,
                "CANDIDATE_FOR_OPERATOR_REVIEW_AFTER_BACKUP_AND_EXPLICIT_APPROVAL",
            ),
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _policy(path: Path) -> Path:
    payload = {
        "schema_version": "missing_runner_insert_policy_packet_v1",
        "status": "REPORT_ONLY_MISSING_RUNNER_INSERT_POLICY_PACKET",
        "report_only": True,
        "safe_to_write_now": False,
        "writes_performed": dict(WRITES_PERFORMED),
        "metadata_policy_rows": [
            {"race_id": "R1", "after_patch": {"actual_field_size": 3}},
            {"race_id": "R2", "after_patch": {"actual_field_size": 2}},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _post_repair_forecast(path: Path) -> Path:
    payload = {
        "schema_version": "post_repair_label_gate_forecast_packet_v1",
        "status": "REPORT_ONLY_POST_REPAIR_LABEL_GATE_FORECAST_PACKET",
        "report_only": True,
        "safe_to_write_now": False,
        "writes_performed": dict(WRITES_PERFORMED),
        "forecast_rows": [
            {
                "race_id": "R1",
                "review_lane": "P1",
                "runner_set_complete_after_proposed_repair": True,
            },
            {
                "race_id": "R2",
                "review_lane": "P4",
                "runner_set_complete_after_proposed_repair": True,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _update_policy(path: Path, db: Path, manifest: Path) -> Path:
    payload = {
        "schema_version": "non_terminal_update_policy_manifest_v1",
        "status": "REPORT_ONLY_NON_TERMINAL_UPDATE_POLICY_MANIFEST",
        "report_only": True,
        "safe_to_write_now": False,
        "writes_performed": dict(WRITES_PERFORMED),
        "source_evidence": {
            "db": str(db),
            "manifest_packet": str(manifest),
        },
        "proposed_update_rows": [
            {
                "race_id": "R1",
                "db_rowid": 2,
                "dog_name_key": "beta",
                "current_box_number": 2,
                "proposed_box_number": 4,
                "current_finish_position": 2,
                "proposed_finish_position": 4,
                "proposed_placing": 4,
                "proposed_scraped_finish_position": "4",
            }
        ],
        "deferred_insert_rows": [
            {
                "candidate_id": "R1::gamma",
                "race_id": "R1",
                "official_dog_name": "Gamma",
                "name_key": "gamma",
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
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _inputs(tmp_path: Path):
    db = _db(tmp_path / "greyhound.sqlite")
    policy = _policy(tmp_path / "policy.json")
    post_repair = _post_repair_forecast(tmp_path / "post_repair.json")
    manifest = _manifest(tmp_path / "manifest.json", policy, post_repair)
    update_policy = _update_policy(tmp_path / "update_policy.json", db, manifest)
    return db, policy, post_repair, manifest, update_policy


def test_post_update_forecast_ranks_batch_candidates_and_keeps_insert_only_blocked(tmp_path: Path):
    _db_path, _policy_path, _post_repair, _manifest_path, update_policy = _inputs(tmp_path)

    packet = build_post_update_forecast_packet(update_policy_packet_path=update_policy)

    assert packet["schema_version"] == "post_update_label_gate_forecast_v1"
    assert packet["status"] == "REPORT_ONLY_POST_UPDATE_LABEL_GATE_FORECAST"
    assert packet["safe_to_write_now"] is False
    assert packet["writes_performed"]["db_write"] is False
    assert packet["summary"]["batch_candidate_count"] == 1
    assert packet["summary"]["insert_only_unmodeled_update_count"] == 1
    assert packet["summary"]["smallest_batch_race_id"] == "R1"
    by_race = {row["race_id"]: row for row in packet["forecast_rows"]}
    assert by_race["R1"]["post_update_gate"] == GATE_BATCH_CANDIDATE
    assert by_race["R1"]["simulated_duplicate_guard_hit_count"] == 0
    assert by_race["R1"]["batch_candidate_rank"] == 1
    assert by_race["R2"]["post_update_gate"] == GATE_INSERT_ONLY_UNMODELED_UPDATES
    assert by_race["R2"]["label_preflight_required"] is True


def test_post_update_forecast_cli_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(output_guard, "ROOT", tmp_path)
    _db_path, _policy_path, _post_repair, _manifest_path, update_policy = _inputs(tmp_path)
    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/post_update"
    exit_code = main(
        [
            "--update-policy-packet",
            str(update_policy),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    payload = json.loads((output_dir / "post_update_label_gate_forecast_packet.json").read_text())
    assert payload["summary"]["batch_candidate_count"] == 1
    assert (output_dir / "post_update_label_gate_forecast.csv").exists()
    assert "No DB rows" in (output_dir / "SUMMARY.md").read_text(encoding="utf-8")


def test_post_update_forecast_rejects_absolute_output_outside_repo(tmp_path: Path):
    outside = tmp_path.parent / "artifacts/full_evidence_orchestration_20260525/post_update"

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        output_guard._assert_output_dir_safe(outside, root=tmp_path)


def test_post_update_forecast_rejects_in_repo_non_artifact_output(tmp_path: Path):
    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        output_guard._assert_output_dir_safe(tmp_path / "reports/post_update", root=tmp_path)
