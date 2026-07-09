import json
from pathlib import Path

import pytest

from scripts.build_no_box_feature_policy_variant_packet import (
    build_feature_policy_variant,
    write_outputs,
)


def _packet() -> dict:
    return {
        "schema_version": "no_box_actual_win_dog_form_feature_join_v1",
        "status": "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_READY",
        "report_only": True,
        "writes_performed": {
            "db_write": False,
            "label_write": False,
            "model_training": False,
            "registry_mutation": False,
            "promotion": False,
        },
        "summary": {
            "joined_rows": 2,
            "feature_column_count": 3,
            "label_proxy_audit": {"status": "PASS"},
        },
        "feature_policy": {"policy_key": "dog_form_only_no_box_no_race_number_no_calendar"},
    }


def _row(race_id: str, dog: str, actual_win: int) -> dict:
    return {
        "schema_version": "no_box_actual_win_dog_form_feature_rows_v1",
        "race_id": race_id,
        "legacy_race_id": race_id,
        "identity_key": f"2025-01-01|TEST|{race_id}",
        "race_date": "2025-01-01",
        "venue": "TEST",
        "race_number": 1,
        "dog_name_key": dog.lower(),
        "dog_name": dog,
        "actual_win": actual_win,
        "candidate_kind": "partial_field",
        "field_scope": "partial_db_name_subset_of_official_finishers",
        "field_complete_for_ranking": False,
        "race_grouped_actual_win_ranking_allowed": False,
        "target_source": "official_winner_name_metadata_confirmed",
        "label_scope": "actual_win_only",
        "box_features_allowed": False,
        "finish_order_labels_allowed": False,
        "top3_labels_allowed": False,
        "official_safe_label_candidate": False,
        "label_write_approved": False,
        "feature_join_status": "MATCHED",
        "feature_win_rate_same_distance": 0.0,
        "feature_place_rate_same_distance": None,
        "feature_avg_time_same_distance": 20.1,
    }


def _rows() -> list[dict]:
    return [_row("R1", "Winner", 1), _row("R1", "Other", 0)]


def test_feature_policy_variant_drops_triage_quarantine_columns_and_preserves_no_write_contract(
    tmp_path: Path,
    monkeypatch,
):
    import scripts.build_no_box_feature_policy_variant_packet as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    packet, rows = build_feature_policy_variant(
        source_packet=_packet(),
        source_rows=_rows(),
        triage_report={
            "summary": {
                "quarantine_candidate_features": [
                    "feature_win_rate_same_distance",
                    "feature_place_rate_same_distance",
                ]
            }
        },
        expected_races=1,
    )

    assert packet["status"] == "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_READY"
    assert packet["report_only"] is True
    assert packet["writes_performed"]["db_write"] is False
    assert packet["writes_performed"]["label_write"] is False
    assert packet["summary"]["source_feature_column_count"] == 3
    assert packet["summary"]["feature_column_count"] == 1
    assert packet["summary"]["dropped_feature_columns_present"] == [
        "feature_place_rate_same_distance",
        "feature_win_rate_same_distance",
    ]
    assert packet["summary"]["variant_validation"]["status"] == "PASS"
    assert packet["feature_policy"]["policy_variant_key"] == "quarantine_same_distance_rates"
    assert "feature_win_rate_same_distance" not in rows[0]
    assert "feature_place_rate_same_distance" not in rows[0]
    assert rows[0]["feature_avg_time_same_distance"] == 20.1

    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/policy_variant"
    write_outputs(output_dir, packet, rows)
    written = json.loads((output_dir / "no_box_actual_win_feature_join_packet.json").read_text())
    assert written["summary"]["feature_column_count"] == 1
    assert (output_dir / "no_box_actual_win_feature_rows.jsonl").exists()
    assert (output_dir / "no_box_actual_win_feature_rows.csv").exists()
    assert (output_dir / "SUMMARY.md").exists()

    cwd = tmp_path / "caller_cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    relative_output_dir = Path(
        "artifacts/full_evidence_orchestration_20260525/relative_policy_variant"
    )
    write_outputs(relative_output_dir, packet, rows)
    assert (
        tmp_path / relative_output_dir / "no_box_actual_win_feature_join_packet.json"
    ).exists()
    assert not (
        cwd / relative_output_dir / "no_box_actual_win_feature_join_packet.json"
    ).exists()


def test_feature_policy_variant_fails_closed_outside_artifacts(tmp_path: Path, monkeypatch):
    import scripts.build_no_box_feature_policy_variant_packet as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    packet, rows = build_feature_policy_variant(
        source_packet=_packet(),
        source_rows=_rows(),
        drop_features=["feature_win_rate_same_distance"],
        expected_races=1,
    )

    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        write_outputs(tmp_path / "outside", packet, rows)
    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        write_outputs(
            tmp_path.parent
            / "outside"
            / "artifacts/full_evidence_orchestration_20260525/policy_variant",
            packet,
            rows,
        )


def test_feature_policy_variant_fails_contract_on_box_feature():
    rows = _rows()
    rows[0]["box_number"] = 1
    packet, _ = build_feature_policy_variant(
        source_packet=_packet(),
        source_rows=rows,
        drop_features=["feature_win_rate_same_distance"],
        expected_races=1,
    )

    assert packet["status"] == "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_FAILED"
    assert packet["summary"]["variant_validation"]["status"] == "FAIL"
    assert any(
        "forbidden_fields_present:box_number" in item
        for item in packet["summary"]["variant_validation"]["failures"]
    )
