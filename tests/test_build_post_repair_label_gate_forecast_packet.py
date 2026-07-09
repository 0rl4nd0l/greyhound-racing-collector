import json
from pathlib import Path

import pytest

import scripts.build_post_repair_label_gate_forecast_packet as forecast
from scripts.build_post_repair_label_gate_forecast_packet import build_forecast_packet, main


def _triage(path: Path) -> Path:
    payload = {
        "schema_version": "rolling_failure_repair_triage_packet_v1",
        "status": "REPORT_ONLY_ROLLING_FAILURE_REPAIR_TRIAGE_PACKET",
        "report_only": True,
        "writes_performed": {
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
        },
        "triage_rows": [
            {
                "race_id": "R1",
                "review_lane": "P1_TOP1_MISS_PARSED_WINNER_ONLY_READY_REVIEW",
                "terminal_status_count": 0,
                "field_scope": "partial_db_name_subset_of_official_finishers",
                "official_runner_count": 3,
                "db_runner_count": 2,
                "missing_runner_insert_candidate_count": 1,
                "changed_dog_update_candidate_count": 2,
                "metadata_update_candidate_count": 1,
            },
            {
                "race_id": "R2",
                "review_lane": "P2_TOP1_MISS_PARSED_OFFICIAL_REVIEW",
                "terminal_status_count": 1,
                "field_scope": "partial_db_name_subset_after_nonstarter_terminal_exclusions",
                "official_runner_count": 2,
                "db_runner_count": 1,
                "missing_runner_insert_candidate_count": 1,
                "changed_dog_update_candidate_count": 1,
                "metadata_update_candidate_count": 1,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _policy(path: Path) -> Path:
    payload = {
        "schema_version": "missing_runner_insert_policy_packet_v1",
        "status": "REPORT_ONLY_MISSING_RUNNER_INSERT_POLICY_PACKET",
        "report_only": True,
        "writes_performed": {
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
        },
        "candidate_rows": [
            {"race_id": "R1", "official_dog_name": "Gamma"},
            {"race_id": "R2", "official_dog_name": "Echo"},
        ],
        "metadata_policy_rows": [
            {"race_id": "R1", "deferred_policy_candidates": []},
            {"race_id": "R2", "deferred_policy_candidates": [{"field": "field_size"}]},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_post_repair_forecast_keeps_direct_label_preflight_blocked(tmp_path: Path):
    packet = build_forecast_packet(
        repair_triage_packet_path=_triage(tmp_path / "triage.json"),
        insert_policy_packet_path=_policy(tmp_path / "policy.json"),
    )

    assert packet["schema_version"] == "post_repair_label_gate_forecast_packet_v1"
    assert packet["status"] == "REPORT_ONLY_POST_REPAIR_LABEL_GATE_FORECAST_PACKET"
    assert packet["safe_to_write_now"] is False
    assert packet["direct_label_preflight_ready_forecast"] is False
    assert packet["summary"]["runner_set_complete_after_proposed_repair_count"] == 2
    assert packet["summary"]["terminal_policy_required_count"] == 1
    assert packet["summary"]["direct_label_preflight_ready_forecast_count"] == 0
    r1 = next(row for row in packet["forecast_rows"] if row["race_id"] == "R1")
    assert r1["forecast_gate"] == "POST_REPAIR_RUNNER_SET_COMPLETE_TERMINAL_FREE_RECHECK_REQUIRED"
    assert r1["runner_set_complete_after_proposed_repair"] is True
    assert r1["direct_label_preflight_ready_forecast"] is False
    assert "direct_label_preflight_still_blocks_existing_result_rows" in r1["remaining_blockers"]
    r2 = next(row for row in packet["forecast_rows"] if row["race_id"] == "R2")
    assert r2["forecast_gate"] == "POST_REPAIR_RUNNER_SET_COMPLETE_TERMINAL_POLICY_REQUIRED"
    assert "terminal_status_policy_required" in r2["remaining_blockers"]
    assert "field_size_metadata_policy_required" in r2["remaining_blockers"]


def test_post_repair_forecast_cli_writes_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(forecast, "ROOT", tmp_path)
    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/forecast"
    exit_code = main(
        [
            "--repair-triage-packet",
            str(_triage(tmp_path / "triage.json")),
            "--insert-policy-packet",
            str(_policy(tmp_path / "policy.json")),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    payload = json.loads((output_dir / "post_repair_label_gate_forecast_packet.json").read_text())
    assert payload["summary"]["direct_label_preflight_ready_forecast_count"] == 0
    assert (output_dir / "post_repair_label_gate_forecast.csv").exists()
    assert "No DB rows" in (output_dir / "SUMMARY.md").read_text(encoding="utf-8")


def test_post_repair_forecast_rejects_absolute_output_outside_repo(tmp_path: Path):
    outside = tmp_path.parent / "artifacts/full_evidence_orchestration_20260525/forecast"

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        forecast._assert_output_dir_safe(outside, root=tmp_path)


def test_post_repair_forecast_rejects_in_repo_non_artifact_output(tmp_path: Path):
    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        forecast._assert_output_dir_safe(tmp_path / "reports/forecast", root=tmp_path)
