import json
import sqlite3
from pathlib import Path

import pytest

import scripts.build_missing_runner_insert_policy_packet as output_guard
from scripts.build_non_terminal_update_policy_manifest import build_update_policy_packet, main


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
            ("R1", "1. Alpha", "Alpha", 1, 1, 1, "1", None),
            ("R1", "2. Beta", "Beta", 2, 2, 2, "2", None),
        ],
    )
    conn.commit()
    conn.close()
    return path


def _candidate(race_id: str, name: str, box: int, status: str) -> dict:
    return {
        "candidate_id": f"{race_id}::{name.lower()}",
        "race_id": race_id,
        "official_dog_name": name,
        "name_key": name.lower(),
        "box_number": box,
        "finish_position": 4,
        "insert_columns": [
            "race_id",
            "dog_name",
            "dog_clean_name",
            "box_number",
            "finish_position",
            "placing",
            "scraped_finish_position",
            "extraction_timestamp",
            "data_source",
        ],
        "insert_values": {
            "race_id": race_id,
            "dog_name": name,
            "dog_clean_name": name,
            "box_number": box,
            "finish_position": 4,
            "placing": 4,
            "scraped_finish_position": "4",
            "extraction_timestamp": "<APPLY_TIME_UTC>",
            "data_source": "thedogs_official",
        },
        "apply_manifest_status": status,
        "safe_to_apply_now": False,
    }


def _manifest(path: Path) -> Path:
    payload = {
        "schema_version": "non_terminal_repair_apply_manifest_forecast_v1",
        "status": "REPORT_ONLY_NON_TERMINAL_REPAIR_APPLY_MANIFEST_FORECAST",
        "report_only": True,
        "safe_to_write_now": False,
        "writes_performed": dict(WRITES_PERFORMED),
        "candidate_manifest_rows": [
            _candidate("R1", "Gamma", 2, "EXCLUDED_DUPLICATE_GUARD_HIT"),
            _candidate("R1", "Eta", 4, "DEFERRED_RACE_HAS_OTHER_DUPLICATE_GUARD_HIT"),
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _reconciliation(path: Path, db: Path, manifest: Path) -> Path:
    payload = {
        "schema_version": "non_terminal_duplicate_guard_update_reconciliation_packet_v1",
        "status": "REPORT_ONLY_NON_TERMINAL_DUPLICATE_GUARD_UPDATE_RECONCILIATION",
        "report_only": True,
        "safe_to_write_now": False,
        "writes_performed": dict(WRITES_PERFORMED),
        "source_evidence": {
            "db": str(db),
            "manifest_packet": str(manifest),
        },
        "race_diagnostics": [
            {
                "race_id": "R1",
                "review_lane": "P1_TOP1_MISS_PARSED_WINNER_ONLY_READY_REVIEW",
                "items": [
                    {
                        "item_type": "duplicate_guard_box_or_name_conflict_policy_required",
                        "candidate_id": "R1::gamma",
                        "official_dog_name": "Gamma",
                        "name_key": "gamma",
                        "official_box_number": 2,
                        "official_finish_position": 4,
                        "db_dog_name": "Beta",
                        "db_name_key": "beta",
                        "db_box_number": 2,
                        "db_finish_position": 2,
                    },
                    {
                        "item_type": "matched_official_finisher_db_update_policy_required",
                        "official_dog_name": "Beta",
                        "name_key": "beta",
                        "official_box_number": 3,
                        "official_finish_position": 3,
                        "db_dog_name": "Beta",
                        "db_name_key": "beta",
                        "db_box_number": 2,
                        "db_finish_position": 2,
                    },
                    {
                        "item_type": "candidate_deferred_until_same_race_duplicate_conflicts_resolved",
                        "candidate_id": "R1::eta",
                        "official_dog_name": "Eta",
                        "name_key": "eta",
                        "official_box_number": 4,
                        "official_finish_position": 4,
                    },
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_update_policy_manifest_separates_updates_and_deferred_inserts(tmp_path: Path):
    db = _db(tmp_path / "greyhound.sqlite")
    manifest = _manifest(tmp_path / "manifest.json")
    reconciliation = _reconciliation(tmp_path / "reconciliation.json", db, manifest)

    packet = build_update_policy_packet(reconciliation_packet_path=reconciliation)

    assert packet["schema_version"] == "non_terminal_update_policy_manifest_v1"
    assert packet["status"] == "REPORT_ONLY_NON_TERMINAL_UPDATE_POLICY_MANIFEST"
    assert packet["safe_to_write_now"] is False
    assert packet["writes_performed"]["db_write"] is False
    assert packet["summary"]["proposed_update_count"] == 1
    assert packet["summary"]["deferred_insert_count"] == 2
    assert packet["summary"]["updates_missing_db_rowid_count"] == 0
    update = packet["proposed_update_rows"][0]
    assert update["dog_name_key"] == "beta"
    assert update["db_rowid"] is not None
    assert update["current_box_number"] == 2
    assert update["proposed_box_number"] == 3
    deferred_by_id = {row["candidate_id"]: row for row in packet["deferred_insert_rows"]}
    assert deferred_by_id["R1::gamma"]["defer_reason"] == "candidate_box_occupied_until_update_policy_resolved"
    assert deferred_by_id["R1::gamma"]["insert_values"]["dog_name"] == "Gamma"
    assert deferred_by_id["R1::eta"]["defer_reason"] == "same_race_duplicate_guard_update_policy_required"


def test_update_policy_manifest_cli_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(output_guard, "ROOT", tmp_path)
    db = _db(tmp_path / "greyhound.sqlite")
    manifest = _manifest(tmp_path / "manifest.json")
    reconciliation = _reconciliation(tmp_path / "reconciliation.json", db, manifest)
    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/update_policy"

    exit_code = main(
        [
            "--reconciliation-packet",
            str(reconciliation),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    payload = json.loads((output_dir / "non_terminal_update_policy_manifest_packet.json").read_text())
    assert payload["summary"]["proposed_update_count"] == 1
    assert (output_dir / "non_terminal_update_policy_updates.csv").exists()
    assert (output_dir / "non_terminal_update_policy_deferred_inserts.csv").exists()
    assert (output_dir / "non_terminal_update_policy_races.csv").exists()
    assert "No DB rows" in (output_dir / "SUMMARY.md").read_text(encoding="utf-8")


def test_update_policy_manifest_rejects_absolute_output_outside_repo(tmp_path: Path):
    outside = tmp_path.parent / "artifacts/full_evidence_orchestration_20260525/update_policy"

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        output_guard._assert_output_dir_safe(outside, root=tmp_path)


def test_update_policy_manifest_rejects_in_repo_non_artifact_output(tmp_path: Path):
    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        output_guard._assert_output_dir_safe(tmp_path / "reports/update_policy", root=tmp_path)
