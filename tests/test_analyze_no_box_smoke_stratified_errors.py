from pathlib import Path

import pytest

from scripts.analyze_no_box_smoke_stratified_errors import (
    build_stratified_error_analysis,
    write_outputs,
)


def _row(
    *,
    baseline: str,
    race_id: str,
    venue: str,
    dog: str,
    probability: float,
    rank: int,
    actual_win: int,
) -> dict:
    return {
        "baseline": baseline,
        "race_id": race_id,
        "venue": venue,
        "dog_name_key": dog,
        "dog_name": dog.title(),
        "probability": probability,
        "predicted_rank": rank,
        "actual_win": actual_win,
        "candidate_kind": "partial_field",
        "field_scope": "partial_db_name_subset_of_official_finishers",
        "field_complete_for_ranking": False,
        "feature_join_status": "MATCHED",
        "box_features_allowed": False,
        "finish_order_labels_allowed": False,
        "top3_labels_allowed": False,
        "official_safe_label_candidate": False,
        "label_write_approved": False,
    }


def test_stratified_error_analysis_groups_by_venue_and_winner_rank():
    rows = [
        _row(baseline="b1", race_id="R1", venue="A", dog="winner", probability=0.7, rank=1, actual_win=1),
        _row(baseline="b1", race_id="R1", venue="A", dog="other", probability=0.3, rank=2, actual_win=0),
        _row(baseline="b1", race_id="R2", venue="B", dog="winner", probability=0.2, rank=3, actual_win=1),
        _row(baseline="b1", race_id="R2", venue="B", dog="other", probability=0.8, rank=1, actual_win=0),
    ]

    analysis = build_stratified_error_analysis(rows)

    assert analysis["status"] == "REPORT_ONLY_STRATIFIED_ERROR_ANALYSIS"
    assert analysis["summary"]["race_records"] == 2
    assert analysis["baselines"]["b1"]["overall"]["top1_accuracy"] == 0.5
    assert analysis["baselines"]["b1"]["dimensions"]["venue"]["A"]["top1_accuracy"] == 1.0
    assert analysis["baselines"]["b1"]["dimensions"]["winner_rank_bucket"]["rank_3"]["top1_miss_count"] == 1
    assert "distance" in analysis["unavailable_dimensions"]
    assert "box" in analysis["unavailable_dimensions"]
    assert "source_bucket" in analysis["unavailable_dimensions"]


def test_stratified_error_analysis_fails_closed_on_box_field():
    rows = [
        _row(baseline="b1", race_id="R1", venue="A", dog="winner", probability=0.7, rank=1, actual_win=1),
        _row(baseline="b1", race_id="R1", venue="A", dog="other", probability=0.3, rank=2, actual_win=0),
    ]
    rows[0]["box_number"] = 1

    analysis = build_stratified_error_analysis(rows)

    assert analysis["status"] == "REPORT_ONLY_STRATIFIED_ERROR_ANALYSIS_WITH_FAILURES"
    assert any("forbidden_row_fields" in failure for failure in analysis["failures"])


def test_output_dir_must_stay_under_evidence_artifacts(tmp_path, monkeypatch):
    import scripts.analyze_no_box_smoke_stratified_errors as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    analysis = build_stratified_error_analysis([])

    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        write_outputs(tmp_path / "reports" / "not_allowed", analysis)

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        write_outputs(tmp_path.parent / "outside" / "report", analysis)

    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/no_box_smoke_report"
    write_outputs(output_dir, analysis)

    assert (output_dir / "no_box_smoke_stratified_error_analysis.json").exists()
    assert (output_dir / "no_box_smoke_stratified_error_analysis.csv").exists()
    assert (output_dir / "SUMMARY.md").exists()
