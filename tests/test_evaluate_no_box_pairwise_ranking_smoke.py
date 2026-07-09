import json
from pathlib import Path

import pytest

from scripts.evaluate_no_box_pairwise_ranking_smoke import (
    evaluate_pairwise_ranking,
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


def _rows() -> list[dict]:
    rows = []
    for race in range(1, 13):
        rows.extend(
            [
                _row(race, f"Fast {race}", 1, 10.0),
                _row(race, f"Mid {race}", 0, 5.0),
                _row(race, f"Slow {race}", 0, 1.0),
            ]
        )
    return rows


def test_pairwise_ranking_scores_top1_top3_without_persisted_model(
    tmp_path: Path,
    monkeypatch,
):
    import scripts.evaluate_no_box_pairwise_ranking_smoke as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    report, predictions = evaluate_pairwise_ranking(
        feature_join_packet=_packet(),
        rows=_rows(),
        expected_races=12,
        min_train_races=3,
        min_eval_races=5,
        epochs=8,
    )

    assert report["status"] == "REPORT_ONLY_PAIRWISE_RANKING_EVALUATED"
    assert report["writes_performed"]["model_training"] is False
    assert report["writes_performed"]["model_persistence"] is False
    assert report["ranking_model"]["report_local_pairwise_fit_performed"] is True
    assert report["validation"]["status"] == "PASS"
    assert report["validation"]["usable_feature_columns"] == ["feature_prior_start_count"]
    assert report["metrics"]["race_count"] == 9
    assert report["metrics"]["top1_accuracy"] == 1.0
    assert report["metrics"]["top3_hit_rate"] == 1.0
    assert len(predictions) == 27
    assert all("probability" not in row for row in predictions)
    assert all("box_number" not in row for row in predictions)

    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/pairwise"
    write_outputs(output_dir, report, predictions)
    assert (output_dir / "no_box_pairwise_ranking_report.json").exists()
    written = json.loads((output_dir / "no_box_pairwise_ranking_report.json").read_text())
    assert written["schema_version"] == "no_box_pairwise_ranking_smoke_v1"
    assert (output_dir / "no_box_pairwise_ranking_predictions.jsonl").exists()
    assert (output_dir / "no_box_pairwise_ranking_predictions.csv").exists()
    assert (output_dir / "SUMMARY.md").exists()

    cwd = tmp_path / "caller_cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    relative_output_dir = Path(
        "artifacts/full_evidence_orchestration_20260525/relative_pairwise"
    )
    write_outputs(relative_output_dir, report, predictions)
    assert (tmp_path / relative_output_dir / "no_box_pairwise_ranking_report.json").exists()
    assert not (cwd / relative_output_dir / "no_box_pairwise_ranking_report.json").exists()


def test_pairwise_ranking_output_guard_fails_closed(tmp_path: Path, monkeypatch):
    import scripts.evaluate_no_box_pairwise_ranking_smoke as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    report, predictions = evaluate_pairwise_ranking(
        feature_join_packet=_packet(),
        rows=_rows(),
        expected_races=12,
        min_train_races=3,
        min_eval_races=5,
        epochs=8,
    )

    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        write_outputs(tmp_path / "outside", report, predictions)
    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        write_outputs(
            tmp_path.parent
            / "outside"
            / "artifacts/full_evidence_orchestration_20260525/pairwise",
            report,
            predictions,
        )


def test_pairwise_ranking_rejects_leakage_risk_source_packet():
    report, predictions = evaluate_pairwise_ranking(
        feature_join_packet=_packet("REPORT_ONLY_DOG_FORM_FEATURE_JOIN_LEAKAGE_RISK"),
        rows=_rows(),
        expected_races=12,
        min_train_races=3,
        min_eval_races=5,
    )

    assert report["status"] == "REPORT_ONLY_PAIRWISE_RANKING_REJECTED_LEAKAGE_RISK"
    assert report["source_packet"]["rejection_reason"] == "feature_join_packet_status_leakage_risk"
    assert report["ranking_model"]["report_local_pairwise_fit_performed"] is False
    assert predictions == []


def test_pairwise_ranking_fails_closed_on_non_null_box_feature():
    rows = _rows()
    rows[0]["feature_box_number"] = 1

    report, predictions = evaluate_pairwise_ranking(
        feature_join_packet=_packet(),
        rows=rows,
        expected_races=12,
        min_train_races=3,
        min_eval_races=5,
    )

    assert report["status"] == "REPORT_ONLY_PAIRWISE_RANKING_FAILED_CONTRACT"
    assert report["validation"]["status"] == "FAIL"
    assert any(
        "forbidden_numeric_features_present:feature_box_number:box_feature" in item
        for item in report["validation"]["failures"]
    )
    assert predictions == []


def test_pairwise_ranking_excludes_finish_proxy_features_without_using_them():
    rows = _rows()
    rows[0]["feature_recent_finish_mean_3"] = 1.0

    report, predictions = evaluate_pairwise_ranking(
        feature_join_packet=_packet(),
        rows=rows,
        expected_races=12,
        min_train_races=3,
        min_eval_races=5,
    )

    assert report["status"] == "REPORT_ONLY_PAIRWISE_RANKING_EVALUATED"
    assert report["validation"]["status"] == "PASS"
    assert "feature_recent_finish_mean_3" not in report["validation"]["usable_feature_columns"]
    assert report["validation"]["excluded_numeric_feature_reasons"] == {
        "feature_recent_finish_mean_3": "finish_or_result_proxy_feature",
    }
    assert any(
        "excluded_finish_or_result_proxy_features:feature_recent_finish_mean_3" in item
        for item in report["validation"]["warnings"]
    )
    assert predictions
