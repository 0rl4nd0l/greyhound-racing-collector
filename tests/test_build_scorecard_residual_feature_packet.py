import csv
import json
from pathlib import Path

import pytest

from scripts import build_scorecard_residual_feature_packet as packet


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_scorecard_residual_feature_packet_compares_roles(tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "ROOT", tmp_path)
    feature_dir = tmp_path / "source_run"
    feature_dir.mkdir()
    feature_rows = [
        {
            "race_id": "Race 1 - WPK - 2026-06-10",
            "box_number": 1,
            "dog_name": "Alpha",
            "weather": "Fine",
            "track_condition": "Good",
            "race_time_minutes_since_midnight": 900,
            "prior_start_count": 12,
            "career_win_rate": 0.25,
            "expert_form_metadata_from_sidecar": True,
            "expert_form_career_starts": 12,
            "starts_same_distance": 4,
            "best_time_same_distance": 29.7,
            "field_size": 8,
            "target_distance_safe": 520,
            "target_grade_safe": "Grade 5",
            "box_number": 1,
            "box_band_inside": 1,
            "box_band_middle": 0,
            "box_band_outside": 0,
        },
        {
            "race_id": "Race 1 - WPK - 2026-06-10",
            "box_number": 2,
            "dog_name": "Bravo",
            "weather": None,
            "track_condition": None,
            "race_time_minutes_since_midnight": 900,
            "prior_start_count": 0,
            "career_win_rate": 0.0,
            "expert_form_metadata_from_sidecar": False,
            "expert_form_career_starts": 0,
            "starts_same_distance": 0,
            "best_time_same_distance": None,
            "field_size": 8,
            "target_distance_safe": 520,
            "target_grade_safe": "Grade 5",
            "box_number": 2,
            "box_band_inside": 1,
            "box_band_middle": 0,
            "box_band_outside": 0,
        },
    ]
    (feature_dir / "shadow_feature_rows.json").write_text(
        json.dumps(feature_rows),
        encoding="utf-8",
    )
    prediction_path = feature_dir / "shadow_predictions.jsonl"
    prediction_path.write_text("", encoding="utf-8")
    scorecard_csv = tmp_path / "scorecard.csv"
    _write_csv(
        scorecard_csv,
        [
            {
                "race_id": "Race 1 - WPK - 2026-06-10",
                "race_date": "2026-06-10",
                "venue": "WPK",
                "race_number": "1",
                "runner_count": "8",
                "winner_box": "1",
                "winner_dog_name": "Alpha",
                "model_winner_rank": "4",
                "model_top1_correct": "False",
                "model_top3_correct": "False",
                "model_winner_probability": "0.10",
                "model_logloss": "2.302585092994046",
                "market_winner_rank": "1",
                "market_top1_correct": "True",
                "market_top3_correct": "True",
                "market_winner_probability": "0.50",
                "market_logloss": "0.6931471805599453",
                "model_top_box": "2",
                "market_top_box": "1",
                "winner_prediction_source_path": str(prediction_path),
                "winner_prediction_raw_probability": "0.10",
            },
            {
                "race_id": "Race 2 - WPK - 2026-06-10",
                "race_date": "2026-06-10",
                "venue": "WPK",
                "race_number": "2",
                "runner_count": "8",
                "winner_box": "2",
                "winner_dog_name": "Bravo",
                "model_winner_rank": "1",
                "model_top1_correct": "True",
                "model_top3_correct": "True",
                "model_winner_probability": "0.40",
                "model_logloss": "0.916290731874155",
                "market_winner_rank": "3",
                "market_top1_correct": "False",
                "market_top3_correct": "True",
                "market_winner_probability": "0.20",
                "market_logloss": "1.6094379124341003",
                "model_top_box": "2",
                "market_top_box": "1",
                "winner_prediction_source_path": str(prediction_path),
                "winner_prediction_raw_probability": "0.40",
            },
        ],
    )

    report = packet.build_packet(
        scorecard_csv=scorecard_csv,
        output_dir=tmp_path
        / "artifacts/full_evidence_orchestration_20260525/scorecard_residual_feature_test_report_only",
    )

    assert report["final_status"] == "SCORECARD_RESIDUAL_FEATURE_AUDIT_READY"
    assert report["residual_race_count"] == 1
    assert report["all_roles_joined_race_count"] == 1
    assert report["missing_join_counts"] == {}
    assert report["no_write_guarantees"]["db_write"] is False
    comparisons = {row["feature_family"]: row for row in report["family_comparisons"]}
    assert comparisons["career_stats"]["model_minus_winner_any_nondefault_rate"] == -1.0
    assert (tmp_path / report["role_summary_csv"]).exists()
    assert (tmp_path / report["family_comparison_csv"]).exists()
    assert (tmp_path / report["race_detail_csv"]).exists()


def test_scorecard_residual_feature_output_dir_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="output_dir_must_be_scorecard_residual_feature"):
        packet.assert_output_dir_safe(
            tmp_path / "artifacts/full_evidence_orchestration_20260525/wrong_report_only"
        )


def test_scorecard_residual_feature_output_dir_guard_rejects_symlink_escape(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(packet, "ROOT", tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.mkdir()
    link = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/"
        "scorecard_residual_feature_symlink_report_only"
    )
    link.parent.mkdir(parents=True)
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        packet.assert_output_dir_safe(link)
