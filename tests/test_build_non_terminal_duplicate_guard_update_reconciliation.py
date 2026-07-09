import json
import sqlite3
from pathlib import Path

import pytest

import scripts.build_missing_runner_insert_policy_packet as output_guard
from scripts.build_non_terminal_duplicate_guard_update_reconciliation import (
    build_reconciliation_packet,
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
            ("R1", "1. Alpha", "Alpha", 1, 1, None, None, None),
            ("R1", "2. Beta", "Beta", 2, 2, None, None, None),
            ("R2", "2. Extra Row", "Extra Row", 2, 2, None, None, None),
        ],
    )
    conn.commit()
    conn.close()
    return path


def _lookup(path: Path) -> Path:
    payload = {
        "schema_version": "official_reverify_lookup_dry_run_v1",
        "status": "REPORT_ONLY",
        "writes_performed": {
            "db_write": False,
            "label_write": False,
            "official_fetch": True,
        },
        "results": [
            {
                "legacy_race_id": "R1",
                "positions": [
                    {"dog_name": "Alpha", "box_number": 1, "finish_position": 1},
                    {"dog_name": "Gamma", "box_number": 2, "finish_position": 2},
                    {"dog_name": "Beta", "box_number": 3, "finish_position": 3},
                    {"dog_name": "Eta", "box_number": 4, "finish_position": 4},
                ],
                "terminal_statuses": [],
            },
            {
                "legacy_race_id": "R2",
                "positions": [
                    {"dog_name": "Delta", "box_number": 2, "finish_position": 1},
                ],
                "terminal_statuses": [],
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _candidate(race_id: str, name: str, box: int, status: str) -> dict:
    return {
        "candidate_id": f"{race_id}::{name.lower()}",
        "race_id": race_id,
        "review_lane": "P1_TOP1_MISS_PARSED_WINNER_ONLY_READY_REVIEW",
        "field_scope": "partial_db_name_subset_of_official_finishers",
        "official_dog_name": name,
        "name_key": name.lower(),
        "box_number": box,
        "finish_position": 2,
        "insert_values": {
            "race_id": race_id,
            "dog_name": name,
            "dog_clean_name": name,
            "box_number": box,
            "finish_position": 2,
            "placing": 2,
            "scraped_finish_position": "2",
            "extraction_timestamp": "<APPLY_TIME_UTC>",
            "data_source": "thedogs_official",
        },
        "apply_manifest_status": status,
        "safe_to_apply_now": False,
    }


def _policy(path: Path, db: Path, lookup: Path) -> Path:
    payload = {
        "schema_version": "missing_runner_insert_policy_packet_v1",
        "status": "REPORT_ONLY_MISSING_RUNNER_INSERT_POLICY_PACKET",
        "report_only": True,
        "safe_to_write_now": False,
        "writes_performed": dict(WRITES_PERFORMED),
        "source_evidence": {
            "db": str(db),
            "lookup_packets": [str(lookup)],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _manifest(path: Path, db: Path, policy: Path) -> Path:
    payload = {
        "schema_version": "non_terminal_repair_apply_manifest_forecast_v1",
        "status": "REPORT_ONLY_NON_TERMINAL_REPAIR_APPLY_MANIFEST_FORECAST",
        "report_only": True,
        "safe_to_write_now": False,
        "writes_performed": dict(WRITES_PERFORMED),
        "source_evidence": {
            "db": str(db),
            "insert_policy_packet": str(policy),
        },
        "candidate_manifest_rows": [
            _candidate("R1", "Gamma", 2, "EXCLUDED_DUPLICATE_GUARD_HIT"),
            _candidate("R1", "Eta", 4, "DEFERRED_RACE_HAS_OTHER_DUPLICATE_GUARD_HIT"),
            _candidate("R2", "Delta", 2, "EXCLUDED_DUPLICATE_GUARD_HIT"),
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_duplicate_guard_reconciliation_classifies_official_and_extra_conflicts(tmp_path: Path):
    db = _db(tmp_path / "greyhound.sqlite")
    lookup = _lookup(tmp_path / "lookup.json")
    policy = _policy(tmp_path / "policy.json", db, lookup)
    manifest = _manifest(tmp_path / "manifest.json", db, policy)

    packet = build_reconciliation_packet(manifest_packet_path=manifest)

    assert packet["schema_version"] == "non_terminal_duplicate_guard_update_reconciliation_packet_v1"
    assert packet["status"] == "REPORT_ONLY_NON_TERMINAL_DUPLICATE_GUARD_UPDATE_RECONCILIATION"
    assert packet["safe_to_write_now"] is False
    assert packet["writes_performed"]["db_write"] is False
    assert packet["summary"]["races_considered"] == 2
    assert packet["summary"]["duplicate_guard_candidate_count"] == 2
    assert packet["summary"]["race_deferred_candidate_count"] == 1
    assert packet["summary"]["duplicate_guard_conflict_item_count"] == 2
    assert packet["summary"]["matched_official_update_review_count"] == 1
    assert packet["summary"]["other_official_finisher_box_conflict_count"] == 1
    assert packet["summary"]["extra_db_conflict_count"] == 1
    item_types = packet["summary"]["item_type_counts"]
    assert item_types["candidate_deferred_until_same_race_duplicate_conflicts_resolved"] == 1


def test_duplicate_guard_reconciliation_cli_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(output_guard, "ROOT", tmp_path)
    db = _db(tmp_path / "greyhound.sqlite")
    lookup = _lookup(tmp_path / "lookup.json")
    policy = _policy(tmp_path / "policy.json", db, lookup)
    manifest = _manifest(tmp_path / "manifest.json", db, policy)
    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/duplicate_guard"

    exit_code = main(
        [
            "--manifest-packet",
            str(manifest),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    payload = json.loads(
        (output_dir / "non_terminal_duplicate_guard_update_reconciliation_packet.json").read_text()
    )
    assert payload["summary"]["duplicate_guard_candidate_count"] == 2
    assert (output_dir / "non_terminal_duplicate_guard_update_reconciliation_items.csv").exists()
    assert (output_dir / "non_terminal_duplicate_guard_update_reconciliation_races.csv").exists()
    assert "No DB rows" in (output_dir / "SUMMARY.md").read_text(encoding="utf-8")


def test_duplicate_guard_reconciliation_rejects_absolute_output_outside_repo(tmp_path: Path):
    outside = tmp_path.parent / "artifacts/full_evidence_orchestration_20260525/duplicate_guard"

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        output_guard._assert_output_dir_safe(outside, root=tmp_path)


def test_duplicate_guard_reconciliation_rejects_in_repo_non_artifact_output(tmp_path: Path):
    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        output_guard._assert_output_dir_safe(tmp_path / "reports/duplicate_guard", root=tmp_path)
