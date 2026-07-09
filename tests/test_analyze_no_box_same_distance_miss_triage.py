import json
from pathlib import Path

import pytest

from scripts.analyze_no_box_same_distance_miss_triage import (
    analyze_same_distance_misses,
    classify_same_distance_miss,
    write_outputs,
)


def _report(status: str = "REPORT_ONLY_PAIRWISE_ROLLING_WINDOWS_EVALUATED") -> dict:
    return {
        "schema_version": "fixture",
        "status": status,
        "report_only": True,
        "writes_performed": {
            "db_write": False,
            "label_write": False,
            "model_training": False,
            "registry_mutation": False,
            "promotion": False,
        },
        "sample_size_status": "UNDERPOWERED_BELOW_50_ACTUAL_WIN_RACES",
    }


def _coverage_report() -> dict:
    return {
        "schema_version": "no_box_dog_form_feature_coverage_audit_v1",
        "status": "REPORT_ONLY_DOG_FORM_FEATURE_COVERAGE_AUDIT_COMPLETE",
        "report_only": True,
        "writes_performed": {
            "db_write": False,
            "label_write": False,
            "model_training": False,
            "registry_mutation": False,
            "promotion": False,
        },
        "summary": {
            "sample_size_status": "UNDERPOWERED_BELOW_50_ACTUAL_WIN_RACES",
            "complete_field_status": "UNDERPOWERED_BELOW_100_COMPLETE_FIELD_RACES",
        },
    }


def _column_coverage() -> list[dict]:
    return [
        {
            "feature": "feature_avg_time_same_distance",
            "family": "same_distance",
            "row_coverage": "0.45",
            "winner_coverage": "0.40",
            "prediction_row_coverage": "0.50",
            "zero_rate_of_present": "0.0",
            "distinct_non_null_values": "5",
            "flat_or_all_null": "False",
        },
        {
            "feature": "feature_win_rate_same_distance",
            "family": "same_distance",
            "row_coverage": "0.01",
            "winner_coverage": "0.0",
            "prediction_row_coverage": "0.0",
            "zero_rate_of_present": "1.0",
            "distinct_non_null_values": "1",
            "flat_or_all_null": "True",
        },
    ]


def _prediction(race: str, dog: str, actual_win: int, rank: int, avg_time, win_rate) -> dict:
    return {
        "race_id": race,
        "race_date": "2025-01-01",
        "venue": "TEST",
        "race_number": 1,
        "dog_name_key": dog.lower(),
        "dog_name": dog,
        "actual_win": actual_win,
        "predicted_rank": rank,
        "box_features_allowed": False,
        "finish_order_labels_allowed": False,
        "top3_labels_allowed": False,
        "label_write_approved": False,
        "feature_avg_time_same_distance": avg_time,
        "feature_win_rate_same_distance": win_rate,
        "feature_starts_same_distance": 0,
    }


def _predictions() -> list[dict]:
    return [
        _prediction("R1", "Winner One", 1, 2, None, None),
        _prediction("R1", "Top One", 0, 1, 20.1, 0.1),
        _prediction("R2", "Winner Two", 1, 1, 19.9, None),
        _prediction("R2", "Other Two", 0, 2, None, None),
    ]


def test_classify_same_distance_miss_buckets_sparse_and_richer_cases():
    assert (
        classify_same_distance_miss(
            winner_present_count=1,
            top_pick_present_count=3,
            feature_count=6,
            top1_hit=False,
        )
        == "winner_sparse_top_pick_richer"
    )
    assert (
        classify_same_distance_miss(
            winner_present_count=1,
            top_pick_present_count=1,
            feature_count=6,
            top1_hit=False,
        )
        == "both_sparse_same_distance"
    )
    assert (
        classify_same_distance_miss(
            winner_present_count=4,
            top_pick_present_count=2,
            feature_count=6,
            top1_hit=False,
        )
        == "winner_richer_but_ranked_lower"
    )
    assert (
        classify_same_distance_miss(
            winner_present_count=0,
            top_pick_present_count=0,
            feature_count=6,
            top1_hit=True,
        )
        == "top1_hit"
    )


def test_same_distance_triage_reports_miss_classes_and_feature_actions(
    tmp_path: Path, monkeypatch
):
    import scripts.analyze_no_box_same_distance_miss_triage as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    report, race_rows, feature_rows = analyze_same_distance_misses(
        rolling_report=_report(),
        coverage_report=_coverage_report(),
        column_coverage_rows=_column_coverage(),
        predictions=_predictions(),
        expected_eval_races=2,
    )

    assert report["status"] == "REPORT_ONLY_SAME_DISTANCE_MISS_TRIAGE_COMPLETE"
    assert report["writes_performed"]["db_write"] is False
    assert report["safe_to_write_now"] is False
    assert report["summary"]["evaluated_races"] == 2
    assert report["summary"]["top1_misses"] == 1
    assert report["summary"]["miss_class_counts"] == {
        "winner_sparse_top_pick_richer": 1
    }
    assert "feature_win_rate_same_distance" in report["summary"]["quarantine_candidate_features"]
    assert "feature_avg_time_same_distance" in report["summary"]["repair_candidate_features"]
    assert any(row["miss_class"] == "winner_sparse_top_pick_richer" for row in race_rows)
    assert any(
        row["triage_action"] == "quarantine_or_drop_until_real_winner_coverage"
        for row in feature_rows
    )

    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/same_distance"
    write_outputs(output_dir, report, race_rows, feature_rows)
    written = json.loads((output_dir / "same_distance_miss_triage_report.json").read_text())
    assert written["schema_version"] == "no_box_same_distance_miss_triage_v1"
    assert (output_dir / "same_distance_top1_miss_triage.csv").exists()
    assert (output_dir / "same_distance_feature_triage.csv").exists()
    assert (output_dir / "SUMMARY.md").exists()

    cwd = tmp_path / "caller_cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    relative_output_dir = Path(
        "artifacts/full_evidence_orchestration_20260525/relative_same_distance"
    )
    write_outputs(relative_output_dir, report, race_rows, feature_rows)
    assert (tmp_path / relative_output_dir / "same_distance_miss_triage_report.json").exists()
    assert not (cwd / relative_output_dir / "same_distance_miss_triage_report.json").exists()


def test_same_distance_triage_fails_closed_outside_artifacts(tmp_path: Path, monkeypatch):
    import scripts.analyze_no_box_same_distance_miss_triage as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    report, race_rows, feature_rows = analyze_same_distance_misses(
        rolling_report=_report(),
        coverage_report=_coverage_report(),
        column_coverage_rows=_column_coverage(),
        predictions=_predictions(),
    )

    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        write_outputs(tmp_path / "outside", report, race_rows, feature_rows)
    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        write_outputs(
            tmp_path.parent
            / "outside"
            / "artifacts/full_evidence_orchestration_20260525/same_distance",
            report,
            race_rows,
            feature_rows,
        )


def test_same_distance_triage_fails_contract_on_box_field():
    predictions = _predictions()
    predictions[0]["box_number"] = 1

    report, _, _ = analyze_same_distance_misses(
        rolling_report=_report(),
        coverage_report=_coverage_report(),
        column_coverage_rows=_column_coverage(),
        predictions=predictions,
    )

    assert report["status"] == "REPORT_ONLY_SAME_DISTANCE_MISS_TRIAGE_FAILED_CONTRACT"
    assert report["validation"]["status"] == "FAIL"
    assert any(
        "forbidden_fields_present:box_number" in item
        for item in report["validation"]["failures"]
    )
