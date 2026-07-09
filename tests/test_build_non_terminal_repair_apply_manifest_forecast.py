import json
import sqlite3
from pathlib import Path

import pytest

import scripts.build_missing_runner_insert_policy_packet as output_guard
from scripts.build_non_terminal_repair_apply_manifest_forecast import (
    DUPLICATE_EXCLUDED_STATUS,
    RACE_BLOCKED_STATUS,
    REVIEWABLE_STATUS,
    TERMINAL_EXCLUDED_STATUS,
    build_manifest_forecast_packet,
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
            ("R1", "Alpha", "Alpha", 1, 1, None, None, None),
            ("R2", "Terminal Row", "Terminal Row", 1, None, None, None, None),
            ("R3", "Box Conflict", "Box Conflict", 2, 1, None, None, None),
        ],
    )
    conn.commit()
    conn.close()
    return path


def _candidate(race_id: str, name: str, box: int) -> dict:
    return {
        "candidate_id": f"{race_id}::{name.lower()}",
        "race_id": race_id,
        "review_lane": "P1_TOP1_MISS_PARSED_WINNER_ONLY_READY_REVIEW",
        "field_scope": "partial_db_name_subset_of_official_finishers",
        "source_url": "https://example.test/results",
        "official_dog_name": name,
        "name_key": name.lower(),
        "box_number": box,
        "finish_position": 2,
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
            "finish_position": 2,
            "placing": 2,
            "scraped_finish_position": "2",
            "extraction_timestamp": "<APPLY_TIME_UTC>",
            "data_source": "thedogs_official",
        },
        "status": "BLOCKED_REQUIRES_OPERATOR_APPROVAL_AND_BACKUP",
        "blockers": [
            "explicit_operator_approval_required",
            "db_backup_required_before_apply",
            "exact_candidate_allowlist_required",
            "duplicate_guard_required_before_each_insert",
            "post_apply_gap_review_and_label_preflight_required",
        ],
    }


def _policy(path: Path, db: Path) -> Path:
    payload = {
        "schema_version": "missing_runner_insert_policy_packet_v1",
        "status": "REPORT_ONLY_MISSING_RUNNER_INSERT_POLICY_PACKET",
        "report_only": True,
        "safe_to_write_now": False,
        "writes_performed": dict(WRITES_PERFORMED),
        "source_evidence": {"db": str(db)},
        "candidate_rows": [
            _candidate("R1", "Gamma", 3),
            _candidate("R2", "Omega", 2),
            _candidate("R3", "Zeta", 2),
            _candidate("R3", "Eta", 4),
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _forecast(path: Path) -> Path:
    rows = []
    for race_id in ["R1", "R2", "R3"]:
        rows.append(
            {
                "race_id": race_id,
                "review_lane": "P1_TOP1_MISS_PARSED_WINNER_ONLY_READY_REVIEW",
                "forecast_gate": "POST_REPAIR_RUNNER_SET_COMPLETE_TERMINAL_FREE_RECHECK_REQUIRED",
                "terminal_status_count": 0,
                "runner_set_complete_after_proposed_repair": True,
                "changed_dog_update_candidate_count": 0,
                "metadata_update_candidate_count": 0,
                "remaining_blockers": [
                    "explicit_operator_approval_required",
                    "db_backup_required_before_apply",
                    "exact_candidate_allowlist_required",
                ],
            }
        )
    payload = {
        "schema_version": "post_repair_label_gate_forecast_packet_v1",
        "status": "REPORT_ONLY_POST_REPAIR_LABEL_GATE_FORECAST_PACKET",
        "report_only": True,
        "safe_to_write_now": False,
        "writes_performed": dict(WRITES_PERFORMED),
        "forecast_rows": rows,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _terminal_manual(path: Path) -> Path:
    payload = {
        "schema_version": "terminal_manual_reconciliation_packet_v1",
        "status": "REPORT_ONLY_TERMINAL_MANUAL_RECONCILIATION_PACKET",
        "report_only": True,
        "safe_to_write_now": False,
        "writes_performed": dict(WRITES_PERFORMED),
        "race_diagnostics": [
            {
                "race_id": "R2",
                "manual_policy_lane": "TERMINAL_SCOPE_MANUAL_RECONCILIATION_REQUIRED",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_non_terminal_manifest_classifies_reviewable_terminal_and_duplicate(tmp_path: Path):
    db = _db(tmp_path / "greyhound.sqlite")
    packet = build_manifest_forecast_packet(
        insert_policy_packet_path=_policy(tmp_path / "policy.json", db),
        post_repair_forecast_packet_path=_forecast(tmp_path / "forecast.json"),
        terminal_manual_packet_path=_terminal_manual(tmp_path / "terminal_manual.json"),
    )

    assert packet["schema_version"] == "non_terminal_repair_apply_manifest_forecast_v1"
    assert packet["status"] == "REPORT_ONLY_NON_TERMINAL_REPAIR_APPLY_MANIFEST_FORECAST"
    assert packet["safe_to_write_now"] is False
    assert packet["writes_performed"]["db_write"] is False
    assert packet["summary"]["candidate_count"] == 4
    assert packet["summary"]["reviewable_after_backup_and_explicit_approval_candidate_count"] == 1
    assert packet["summary"]["excluded_terminal_candidate_count"] == 1
    assert packet["summary"]["excluded_duplicate_guard_hit_candidate_count"] == 1
    assert packet["summary"]["deferred_race_blocked_candidate_count"] == 1
    by_id = {row["candidate_id"]: row for row in packet["candidate_manifest_rows"]}
    assert by_id["R1::gamma"]["apply_manifest_status"] == REVIEWABLE_STATUS
    assert by_id["R2::omega"]["apply_manifest_status"] == TERMINAL_EXCLUDED_STATUS
    assert by_id["R3::zeta"]["apply_manifest_status"] == DUPLICATE_EXCLUDED_STATUS
    assert by_id["R3::eta"]["apply_manifest_status"] == RACE_BLOCKED_STATUS
    assert by_id["R1::gamma"]["safe_to_apply_now"] is False


def test_non_terminal_manifest_cli_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(output_guard, "ROOT", tmp_path)
    db = _db(tmp_path / "greyhound.sqlite")
    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/non_terminal_manifest"
    exit_code = main(
        [
            "--insert-policy-packet",
            str(_policy(tmp_path / "policy.json", db)),
            "--post-repair-forecast-packet",
            str(_forecast(tmp_path / "forecast.json")),
            "--terminal-manual-packet",
            str(_terminal_manual(tmp_path / "terminal_manual.json")),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    payload = json.loads(
        (output_dir / "non_terminal_repair_apply_manifest_forecast_packet.json").read_text()
    )
    assert payload["summary"]["candidate_count"] == 4
    assert (output_dir / "non_terminal_repair_apply_manifest_candidates.csv").exists()
    assert (output_dir / "non_terminal_repair_apply_manifest_races.csv").exists()
    assert "No DB rows" in (output_dir / "SUMMARY.md").read_text(encoding="utf-8")


def test_non_terminal_manifest_rejects_absolute_output_outside_repo(tmp_path: Path):
    outside = tmp_path.parent / "artifacts/full_evidence_orchestration_20260525/non_terminal_manifest"

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        output_guard._assert_output_dir_safe(outside, root=tmp_path)


def test_non_terminal_manifest_rejects_in_repo_non_artifact_output(tmp_path: Path):
    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        output_guard._assert_output_dir_safe(tmp_path / "reports/non_terminal_manifest", root=tmp_path)
