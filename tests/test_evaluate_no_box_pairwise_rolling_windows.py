import json
from pathlib import Path

import pytest

from scripts.evaluate_no_box_pairwise_rolling_windows import (
    evaluate_rolling_windows,
    write_outputs,
)


def _packet(status: str = "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_READY") -> dict:
    return {
        "schema_version": "no_box_actual_win_dog_form_feature_join_v1",
        "status": status,
        "report_only": True,
        "writes_performed": {
            "label_write": False,
            "model_training": False,
            "promotion": False,
        },
        "summary": {
            "history_db_fill_policy": "no_outcome_proxy_fields",
            "label_proxy_audit": {
                "status": "POTENTIAL_LABEL_PROXY"
                if status == "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_LEAKAGE_RISK"
                else "PASS",
            },
        },
    }


def _row(race: int, dog: str, actual_win: int, speed: float) -> dict:
    return {
        "race_id": f"R{race:02d}",
        "legacy_race_id": f"R{race:02d}",
        "identity_key": f"2025-01-{race:02d}|TEST|R{race:02d}",
        "race_date": f"2025-01-{race:02d}",
        "venue": "TEST",
        "race_number": race,
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
        "feature_prior_start_count": speed,
    }


def _rows(race_count: int = 16) -> list[dict]:
    rows = []
    for race in range(1, race_count + 1):
        rows.extend(
            [
                _row(race, f"Fast {race}", 1, 10.0),
                _row(race, f"Mid {race}", 0, 5.0),
                _row(race, f"Slow {race}", 0, 1.0),
            ]
        )
    return rows


def test_rolling_windows_scores_multiple_windows_and_reserves_final_races(
    tmp_path: Path,
    monkeypatch,
):
    import scripts.evaluate_no_box_pairwise_ranking_smoke as ranking_module

    monkeypatch.setattr(ranking_module, "ROOT", tmp_path)
    report, predictions = evaluate_rolling_windows(
        feature_join_packet=_packet(),
        rows=_rows(),
        expected_races=16,
        train_races=5,
        eval_races=3,
        step_races=4,
        reserve_final_races=2,
        min_windows=2,
        epochs=8,
    )

    assert report["status"] == "REPORT_ONLY_PAIRWISE_ROLLING_WINDOWS_EVALUATED"
    assert report["writes_performed"]["model_training"] is False
    assert report["writes_performed"]["model_persistence"] is False
    assert report["validation"]["status"] == "PASS"
    assert report["rolling_window_policy"]["reserved_final_races"] == 2
    assert report["rolling_window_policy"]["reserved_races_predicted"] is False
    assert [row["race_id"] for row in report["rolling_window_policy"]["reserved_race_refs"]] == [
        "R15",
        "R16",
    ]
    assert report["window_metric_summary"]["window_count"] == 2
    assert report["aggregate_metrics"]["race_count"] == 6
    assert report["aggregate_metrics"]["top1_accuracy"] == 1.0
    assert report["aggregate_metrics"]["top3_hit_rate"] == 1.0
    assert len(predictions) == 18
    assert {row["window_id"] for row in predictions} == {"window_01", "window_02"}

    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/rolling"
    write_outputs(output_dir, report, predictions)
    assert (output_dir / "no_box_pairwise_rolling_windows_report.json").exists()
    written = json.loads((output_dir / "no_box_pairwise_rolling_windows_report.json").read_text())
    assert written["schema_version"] == "no_box_pairwise_rolling_windows_v1"
    assert (output_dir / "no_box_pairwise_rolling_window_predictions.jsonl").exists()
    assert (output_dir / "no_box_pairwise_rolling_window_predictions.csv").exists()
    assert (output_dir / "SUMMARY.md").exists()

    cwd = tmp_path / "caller_cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    relative_output_dir = Path(
        "artifacts/full_evidence_orchestration_20260525/relative_rolling"
    )
    write_outputs(relative_output_dir, report, predictions)
    assert (
        tmp_path / relative_output_dir / "no_box_pairwise_rolling_windows_report.json"
    ).exists()
    assert not (
        cwd / relative_output_dir / "no_box_pairwise_rolling_windows_report.json"
    ).exists()


def test_rolling_windows_output_guard_fails_closed(tmp_path: Path, monkeypatch):
    import scripts.evaluate_no_box_pairwise_ranking_smoke as ranking_module

    monkeypatch.setattr(ranking_module, "ROOT", tmp_path)
    report, predictions = evaluate_rolling_windows(
        feature_join_packet=_packet(),
        rows=_rows(),
        expected_races=16,
        train_races=5,
        eval_races=3,
        step_races=4,
        reserve_final_races=2,
        min_windows=2,
        epochs=8,
    )

    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        write_outputs(tmp_path / "outside", report, predictions)
    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        write_outputs(
            tmp_path.parent
            / "outside"
            / "artifacts/full_evidence_orchestration_20260525/rolling",
            report,
            predictions,
        )


def test_rolling_windows_rejects_leakage_risk_source_packet():
    report, predictions = evaluate_rolling_windows(
        feature_join_packet=_packet("REPORT_ONLY_DOG_FORM_FEATURE_JOIN_LEAKAGE_RISK"),
        rows=_rows(),
        expected_races=16,
        train_races=5,
        eval_races=3,
        step_races=4,
        reserve_final_races=2,
    )

    assert report["status"] == "REPORT_ONLY_PAIRWISE_ROLLING_WINDOWS_REJECTED_LEAKAGE_RISK"
    assert report["source_packet"]["rejection_reason"] == "feature_join_packet_status_leakage_risk"
    assert report["ranking_model"]["report_local_pairwise_fit_performed"] is False
    assert predictions == []


def test_rolling_windows_fails_closed_on_box_feature():
    rows = _rows()
    rows[0]["feature_box_number"] = 1

    report, predictions = evaluate_rolling_windows(
        feature_join_packet=_packet(),
        rows=rows,
        expected_races=16,
        train_races=5,
        eval_races=3,
        step_races=4,
        reserve_final_races=2,
    )

    assert report["status"] == "REPORT_ONLY_PAIRWISE_ROLLING_WINDOWS_FAILED_CONTRACT"
    assert report["validation"]["status"] == "FAIL"
    assert any(
        "forbidden_numeric_features_present:feature_box_number:box_feature" in item
        for item in report["validation"]["failures"]
    )
    assert predictions == []
