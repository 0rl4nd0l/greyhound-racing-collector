import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import build_rolling_model_comparison_packet as comparison


def _odds(decimal):
    return {
        "t30": {
            "capture_timestamp": "2026-06-10T12:30:00+10:00",
            "odds_decimal": decimal,
            "source_url": "https://www.sportsbet.com.au/greyhound-racing/example",
        }
    }


def _row(
    *,
    race_id,
    dog,
    box,
    winner=False,
    primary=0.1,
    stage2=0.1,
    stage2_uncalibrated=None,
    odds=5.0,
    unified=True,
):
    if stage2_uncalibrated is None:
        stage2_uncalibrated = stage2
    return {
        "race_id": race_id,
        "dog_name": dog,
        "box_number": box,
        "is_winner": winner,
        "finish_position": 1 if winner else 2,
        "official_result_available": True,
        "primary_prediction_available": True,
        "stage2_prediction_available": stage2 is not None,
        "strict_prejump_odds_available": odds is not None,
        "label_evaluation_eligible": True,
        "stage2_evaluation_eligible": stage2 is not None,
        "odds_evaluation_eligible": odds is not None,
        "unified_evidence_eligible": unified,
        "primary_shadow_probability": primary,
        "stage2_shadow_probability": stage2,
        "stage2_shadow_uncalibrated_probability": stage2_uncalibrated,
        "odds_by_capture_bucket": _odds(odds) if odds is not None else {},
    }


def _write_dataset(
    tmp_path: Path,
    name: str,
    rows: list[dict],
    *,
    report_extra: dict | None = None,
) -> Path:
    dataset_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525" / name
    dataset_dir.mkdir(parents=True)
    dataset_jsonl = dataset_dir / "unified_evidence_dataset.jsonl"
    dataset_jsonl.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    report = {
        "final_status": "UNIFIED_EVIDENCE_DATASET_BUILT",
        "dataset_jsonl": str(dataset_jsonl.relative_to(tmp_path)),
        "row_count": len(rows),
        "race_count": len({row["race_id"] for row in rows}),
        "unified_evidence_eligible_rows": sum(
            1 for row in rows if row.get("unified_evidence_eligible")
        ),
        "no_write_guarantees": {"db_write": False, "label_write": False},
    }
    if report_extra:
        report.update(report_extra)
    report_path = dataset_dir / "unified_evidence_dataset_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


def test_rolling_comparison_evaluates_stage2_market_and_blends(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "ROOT", tmp_path)
    rows = [
        _row(race_id="Race 1 - BEN - 2026-06-10", dog="A", box=1, winner=True, primary=0.2, stage2=0.7, stage2_uncalibrated=0.8, odds=2.0),
        _row(race_id="Race 1 - BEN - 2026-06-10", dog="B", box=2, primary=0.6, stage2=0.2, stage2_uncalibrated=0.1, odds=6.0),
        _row(race_id="Race 1 - BEN - 2026-06-10", dog="C", box=3, primary=0.2, stage2=0.1, stage2_uncalibrated=0.1, odds=8.0),
        _row(race_id="Race 2 - BEN - 2026-06-10", dog="D", box=1, primary=0.5, stage2=0.2, stage2_uncalibrated=0.1, odds=7.0),
        _row(race_id="Race 2 - BEN - 2026-06-10", dog="E", box=2, winner=True, primary=0.3, stage2=0.6, stage2_uncalibrated=0.8, odds=2.2),
        _row(race_id="Race 2 - BEN - 2026-06-10", dog="F", box=3, primary=0.2, stage2=0.2, stage2_uncalibrated=0.1, odds=9.0),
    ]
    report_path = _write_dataset(
        tmp_path,
        "unified_evidence_dataset_test",
        rows,
        report_extra={
            "rejected_live_odds_candidate_count": 3,
            "rows_with_rejected_live_odds_candidates": 2,
            "rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 1,
                "odds_source_url_missing": 2,
            },
            "artifact_odds_rows_seen": 23,
            "artifact_odds_rows_accepted": 6,
            "artifact_odds_rows_rejected": 17,
            "artifact_odds_rejection_reason_counts": {
                "odds_match_status_not_valid_pre_jump_dog_odds": 17,
            },
            "exclusion_reason_counts": {"official_result_missing": 8},
            "odds_exclusion_reason_counts": {"strict_prejump_odds_missing": 2},
            "official_result_evidence_db_audit": {
                "race_ids_requested": 3,
                "requested_race_ids": [
                    "Race 1 - BEN - 2026-06-10",
                    "Race 2 - BEN - 2026-06-10",
                    "Race 3 - TAREE - 2026-06-13",
                ],
                "missing_race_ids": ["Race 3 - TAREE - 2026-06-13"],
                "race_ids_with_rows": [
                    "Race 1 - BEN - 2026-06-10",
                    "Race 2 - BEN - 2026-06-10",
                ],
            },
            "official_result_runner_paths": [
                "artifacts/full_evidence_orchestration_20260525/autonomous_official_result_capture_test/official_result_runners.jsonl"
            ],
        },
    )
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525"
        / "rolling_model_comparison_test"
    )

    report = comparison.build_comparison(
        unified_evidence_report_paths=[report_path],
        output_dir=output_dir,
        min_races_for_review=2,
        generated_at=datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc),
    )

    assert report["final_status"] == "ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW"
    assert report["sample_race_count"] == 2
    assert report["sample_floor_met"] is True
    assert report["races_needed_for_review"] == 0
    assert report["source_rejected_live_odds_candidate_count"] == 3
    assert report["source_rows_with_rejected_live_odds_candidates"] == 2
    assert report["source_rejected_live_odds_candidate_reason_counts"] == {
        "odds_decimal_invalid": 1,
        "odds_source_url_missing": 2,
    }
    assert report["source_artifact_odds_rows_seen"] == 23
    assert report["source_artifact_odds_rows_accepted"] == 6
    assert report["source_artifact_odds_rows_rejected"] == 17
    assert report["source_artifact_odds_rejection_reason_counts"] == {
        "odds_match_status_not_valid_pre_jump_dog_odds": 17,
    }
    assert report["source_exclusion_reason_counts"] == {
        "official_result_missing": 8
    }
    assert report["source_odds_exclusion_reason_counts"] == {
        "strict_prejump_odds_missing": 2
    }
    assert report["source_official_result_evidence_db_missing_race_ids"] == [
        "Race 3 - TAREE - 2026-06-13"
    ]
    assert report["source_official_result_evidence_db_requested_race_ids"] == [
        "Race 1 - BEN - 2026-06-10",
        "Race 2 - BEN - 2026-06-10",
        "Race 3 - TAREE - 2026-06-13",
    ]
    assert report["source_official_result_evidence_db_requested_race_count"] == 3
    assert report["source_official_result_evidence_db_races_with_rows"] == [
        "Race 1 - BEN - 2026-06-10",
        "Race 2 - BEN - 2026-06-10",
    ]
    assert report["source_official_result_runner_paths"] == [
        "artifacts/full_evidence_orchestration_20260525/autonomous_official_result_capture_test/official_result_runners.jsonl"
    ]
    assert report["official_result_coverage"] == {
        "source": "unified_evidence_reports",
        "requested_race_count": 3,
        "requested_race_count_source": "deduped_requested_or_inferred_race_ids",
        "requested_race_ids": [
            "Race 1 - BEN - 2026-06-10",
            "Race 2 - BEN - 2026-06-10",
            "Race 3 - TAREE - 2026-06-13",
        ],
        "legacy_requested_race_count_without_ids": 0,
        "races_with_rows_count": 2,
        "missing_race_count": 1,
        "missing_race_ids": ["Race 3 - TAREE - 2026-06-13"],
        "races_with_rows": [
            "Race 1 - BEN - 2026-06-10",
            "Race 2 - BEN - 2026-06-10",
        ],
        "runner_path_count": 1,
        "runner_paths_source_field": "source_official_result_runner_paths",
        "missing_exclusion_count": 8,
    }
    assert report["source_reports"][0]["rejected_live_odds_candidate_count"] == 3
    assert report["source_reports"][0][
        "rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 1,
        "odds_source_url_missing": 2,
    }
    summary = (output_dir / "SUMMARY.md").read_text(encoding="utf-8")
    assert "- Source artifact odds rows seen: `23`" in summary
    assert "- Source artifact odds rows accepted: `6`" in summary
    assert "- Source artifact odds rows rejected: `17`" in summary
    assert (
        "- Source artifact odds rejection reasons: `{'odds_match_status_not_valid_pre_jump_dog_odds': 17}`"
        in summary
    )
    assert report["source_reports"][0]["exclusion_reason_counts"] == {
        "official_result_missing": 8
    }
    assert report["source_reports"][0][
        "official_result_evidence_db_missing_race_ids"
    ] == ["Race 3 - TAREE - 2026-06-13"]
    summary_text = (output_dir / "SUMMARY.md").read_text(encoding="utf-8")
    assert "- Source rejected live odds candidates: `3`" in summary_text
    assert "- Source rows with rejected live odds candidates: `2`" in summary_text
    assert "- Source exclusion reasons: `{'official_result_missing': 8}`" in summary_text
    assert "- Official-result coverage requested races: `3`" in summary_text
    assert "- Official-result coverage races with rows: `2`" in summary_text
    assert "- Official-result coverage missing races: `1`" in summary_text
    assert "- Official-result missing exclusion count: `8`" in summary_text
    assert "- Official-result runner path count: `1`" in summary_text
    assert (
        "- Official-result runner paths source field: "
        "`source_official_result_runner_paths`"
    ) in summary_text
    assert "Source official-result runner paths:" not in summary_text
    assert "autonomous_official_result_capture_test/official_result_runners.jsonl" not in summary_text
    assert "Race 3 - TAREE - 2026-06-13" in summary_text
    assert (
        "- Source rejected live odds candidate reasons: "
        "`{'odds_decimal_invalid': 1, 'odds_source_url_missing': 2}`"
    ) in summary_text
    assert report["candidate_count"] == 22
    assert report["baseline_metrics"]["top1"] == 0.0
    assert report["candidate_metrics_by_key"]["stage2_shadow"]["top1"] == 1.0
    assert report["candidate_metrics_by_key"]["stage2_shadow_uncalibrated"]["top1"] == 1.0
    assert report["candidate_metrics_by_key"]["market_only_implied"]["top1"] == 1.0
    assert report["candidate_metrics_by_key"]["stage2_shadow"][
        "calibration_slope_intercept"
    ] == {
        "intercept": None,
        "minimum_required": {
            "negative_labels": 5,
            "positive_labels": 5,
            "sample_size": 30,
        },
        "negative_labels": 4,
        "positive_labels": 2,
        "sample_size": 6,
        "slope": None,
        "status": "insufficient_sample",
    }
    assert "stage2_market_blend_50" in report["candidate_metrics_by_key"]
    assert "stage2_uncalibrated_market_blend_50" in report["candidate_metrics_by_key"]
    assert "stage2_shadow_uncalibrated_power_gamma_1_2" in report[
        "candidate_metrics_by_key"
    ]
    assert report["best_non_baseline_candidate_key"] != "primary_shadow"
    assert report["best_non_market_candidate_key"] != "market_only_implied"
    assert report["market_residual_diagnostics"]["schema_version"] == (
        "rolling_model_market_residual_diagnostics_v1"
    )
    assert report["market_residual_diagnostics"]["candidate_key"] == report[
        "best_non_market_candidate_key"
    ]
    assert report["market_residual_case_count"] == 2
    assert report["market_residual_cases_csv"].endswith("market_residual_cases.csv")
    assert report["market_residual_runner_matrix_row_count"] == 6
    assert report["market_residual_runner_matrix_csv"].endswith(
        "market_residual_runner_matrix.csv"
    )
    assert len(report["residual_hypothesis_backtests"]) == 1
    assert report["residual_hypothesis_backtests"][0]["promotion_eligible"] is False
    assert report["edge_diagnostics"]["schema_version"] == (
        "rolling_model_edge_diagnostics_v1"
    )
    assert "selected_market_agreement" in report["edge_diagnostics"]["dimensions"]
    assert report["no_write_guarantees"]["production_promotion"] is False
    assert report["no_write_guarantees"]["betting_or_ev_action"] is False
    assert (output_dir / "rolling_model_comparison_report.json").exists()
    assert (output_dir / "candidate_metrics.csv").exists()
    residual_cases_path = output_dir / "market_residual_cases.csv"
    assert residual_cases_path.exists()
    with residual_cases_path.open(encoding="utf-8", newline="") as handle:
        residual_cases = list(csv.DictReader(handle))
    assert len(residual_cases) == 2
    assert residual_cases[0]["candidate_key"] == report["best_non_market_candidate_key"]
    assert residual_cases[0]["market_candidate_key"] == "market_only_implied"
    assert "candidate_minus_market_logloss" in residual_cases[0]
    residual_matrix_path = output_dir / "market_residual_runner_matrix.csv"
    assert residual_matrix_path.exists()
    with residual_matrix_path.open(encoding="utf-8", newline="") as handle:
        residual_matrix = list(csv.DictReader(handle))
    assert len(residual_matrix) == 6
    assert residual_matrix[0]["candidate_key"] == report["best_non_market_candidate_key"]
    assert residual_matrix[0]["market_candidate_key"] == "market_only_implied"
    assert residual_matrix[0]["odds_source_url"].startswith("https://www.sportsbet.com.au/")
    assert "candidate_minus_market_probability" in residual_matrix[0]
    assert "stage2_shadow_uncalibrated_probability_norm" in residual_matrix[0]
    assert (output_dir / "SUMMARY.md").exists()


def test_unified_scope_excludes_partial_odds_races(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "ROOT", tmp_path)
    rows = [
        _row(race_id="Race 1 - BEN - 2026-06-10", dog="A", box=1, winner=True, primary=0.6, stage2=0.6, odds=2.0),
        _row(race_id="Race 1 - BEN - 2026-06-10", dog="B", box=2, primary=0.4, stage2=0.4, odds=None, unified=False),
    ]
    report_path = _write_dataset(tmp_path, "unified_evidence_dataset_partial", rows)

    report = comparison.build_comparison(
        unified_evidence_report_paths=[report_path],
        output_dir=(
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525"
            / "rolling_model_comparison_partial"
        ),
        generated_at=datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc),
    )

    assert report["sample_race_count"] == 0
    assert report["sample_floor_met"] is False
    assert report["races_needed_for_review"] == comparison.MIN_RACES_FOR_REVIEW
    assert report["skipped_race_counts"] == {
        "race_not_fully_unified_evidence_eligible": 1
    }
    assert report["final_status"] == "ROLLING_MODEL_COMPARISON_COLLECTING"


def test_dedupe_keeps_latest_report_for_same_race(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "ROOT", tmp_path)
    older = _write_dataset(
        tmp_path,
        "unified_evidence_dataset_older",
        [
            _row(race_id="Race 1 - BEN - 2026-06-10", dog="A", box=1, winner=True, primary=0.9, stage2=0.1, odds=2.0),
            _row(race_id="Race 1 - BEN - 2026-06-10", dog="B", box=2, primary=0.1, stage2=0.9, odds=6.0),
        ],
    )
    newer = _write_dataset(
        tmp_path,
        "unified_evidence_dataset_newer",
        [
            _row(race_id="Race 1 - BEN - 2026-06-10", dog="A", box=1, winner=True, primary=0.1, stage2=0.9, odds=2.0),
            _row(race_id="Race 1 - BEN - 2026-06-10", dog="B", box=2, primary=0.9, stage2=0.1, odds=6.0),
        ],
        report_extra={
            "rejected_live_odds_candidate_count": 4,
            "rows_with_rejected_live_odds_candidates": 3,
            "rejected_live_odds_candidate_reason_counts": {
                "odds_source_url_missing": 4,
            },
            "artifact_odds_rows_seen": 8,
            "artifact_odds_rows_accepted": 3,
            "artifact_odds_rows_rejected": 5,
            "artifact_odds_audits": [
                {
                    "rejection_reason_counts": {
                        "odds_match_status_not_valid_pre_jump_dog_odds": 5,
                    }
                }
            ],
        },
    )

    report = comparison.build_comparison(
        unified_evidence_report_paths=[older, newer],
        output_dir=(
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525"
            / "rolling_model_comparison_dedupe"
        ),
        min_races_for_review=1,
        generated_at=datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc),
    )

    assert report["sample_race_count"] == 1
    assert report["baseline_metrics"]["top1"] == 0.0
    assert report["candidate_metrics_by_key"]["stage2_shadow"]["top1"] == 1.0
    assert report["source_reports"][0]["source_index"] == 0
    assert report["source_reports"][1]["source_index"] == 1
    assert report["source_rejected_live_odds_candidate_count"] == 4
    assert report["source_rejected_live_odds_candidate_reason_counts"] == {
        "odds_source_url_missing": 4,
    }
    assert report["source_artifact_odds_rows_seen"] == 8
    assert report["source_artifact_odds_rows_accepted"] == 3
    assert report["source_artifact_odds_rows_rejected"] == 5
    assert report["source_artifact_odds_rejection_reason_counts"] == {
        "odds_match_status_not_valid_pre_jump_dog_odds": 5,
    }


def test_official_result_coverage_requested_races_are_deduped_across_reports(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(comparison, "ROOT", tmp_path)
    rows = [
        _row(
            race_id="Race 1 - BEN - 2026-06-10",
            dog="A",
            box=1,
            winner=True,
            primary=0.7,
            stage2=0.8,
            odds=2.0,
        ),
        _row(
            race_id="Race 1 - BEN - 2026-06-10",
            dog="B",
            box=2,
            primary=0.3,
            stage2=0.2,
            odds=4.0,
        ),
    ]
    first = _write_dataset(
        tmp_path,
        "unified_evidence_dataset_requested_first",
        rows,
        report_extra={
            "official_result_evidence_db_audit": {
                "race_ids_requested": 2,
                "requested_race_ids": [
                    "Race 1 - BEN - 2026-06-10",
                    "Race 2 - BEN - 2026-06-10",
                ],
                "race_ids_with_rows": ["Race 1 - BEN - 2026-06-10"],
                "missing_race_ids": ["Race 2 - BEN - 2026-06-10"],
            }
        },
    )
    second = _write_dataset(
        tmp_path,
        "unified_evidence_dataset_requested_second",
        rows,
        report_extra={
            "official_result_evidence_db_audit": {
                "race_ids_requested": 2,
                "requested_race_ids": [
                    "Race 1 - BEN - 2026-06-10",
                    "Race 2 - BEN - 2026-06-10",
                ],
                "race_ids_with_rows": ["Race 1 - BEN - 2026-06-10"],
                "missing_race_ids": ["Race 2 - BEN - 2026-06-10"],
            }
        },
    )

    report = comparison.build_comparison(
        unified_evidence_report_paths=[first, second],
        output_dir=(
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525"
            / "rolling_model_comparison_requested_dedupe"
        ),
        min_races_for_review=1,
        generated_at=datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc),
    )

    assert report["source_official_result_evidence_db_requested_race_count"] == 2
    assert report["source_official_result_evidence_db_requested_race_ids"] == [
        "Race 1 - BEN - 2026-06-10",
        "Race 2 - BEN - 2026-06-10",
    ]
    assert report["official_result_coverage"]["requested_race_count"] == 2
    assert report["official_result_coverage"]["requested_race_count_source"] == (
        "deduped_requested_or_inferred_race_ids"
    )


def test_market_residual_diagnostics_compare_best_non_market_when_market_wins(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(comparison, "ROOT", tmp_path)
    rows = [
        _row(race_id="Race 1 - BEN - 2026-06-10", dog="A", box=1, winner=True, primary=0.05, stage2=0.05, stage2_uncalibrated=0.05, odds=2.0),
        _row(race_id="Race 1 - BEN - 2026-06-10", dog="B", box=2, primary=0.9, stage2=0.9, stage2_uncalibrated=0.9, odds=2.1),
        _row(race_id="Race 1 - BEN - 2026-06-10", dog="C", box=3, primary=0.05, stage2=0.05, stage2_uncalibrated=0.05, odds=12.0),
        _row(race_id="Race 2 - BEN - 2026-06-10", dog="D", box=1, winner=True, primary=0.05, stage2=0.05, stage2_uncalibrated=0.05, odds=2.0),
        _row(race_id="Race 2 - BEN - 2026-06-10", dog="E", box=2, primary=0.9, stage2=0.9, stage2_uncalibrated=0.9, odds=2.1),
        _row(race_id="Race 2 - BEN - 2026-06-10", dog="F", box=3, primary=0.05, stage2=0.05, stage2_uncalibrated=0.05, odds=12.0),
    ]
    report_path = _write_dataset(tmp_path, "unified_evidence_dataset_market_residual", rows)

    report = comparison.build_comparison(
        unified_evidence_report_paths=[report_path],
        output_dir=(
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525"
            / "rolling_model_comparison_market_residual"
        ),
        min_races_for_review=2,
        generated_at=datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc),
    )

    assert report["best_candidate_key"] == "market_only_implied"
    assert report["best_non_market_candidate_key"] != "market_only_implied"
    assert report["edge_diagnostics"]["selected_candidate_key"] == report[
        "best_non_market_candidate_key"
    ]
    residual = report["market_residual_diagnostics"]
    assert residual["status"] == "EVALUATED"
    assert residual["candidate_key"] == report["best_non_market_candidate_key"]
    assert residual["market_candidate_key"] == "market_only_implied"
    assert residual["candidate_minus_market"]["top1"] < 0
    assert residual["outcome_counts"]["market_top1_candidate_miss"] == 2
    assert residual["strongest_market_logloss_edges"]


def test_residual_hypothesis_backtest_is_report_only_and_underpowered(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(comparison, "ROOT", tmp_path)
    rows = [
        _row(race_id="Race 1 - BEN - 2026-06-10", dog="A", box=1, winner=True, primary=0.1, stage2=0.8, stage2_uncalibrated=0.8, odds=6.0),
        _row(race_id="Race 1 - BEN - 2026-06-10", dog="B", box=2, primary=0.8, stage2=0.1, stage2_uncalibrated=0.1, odds=5.0),
        _row(race_id="Race 1 - BEN - 2026-06-10", dog="C", box=3, primary=0.1, stage2=0.1, stage2_uncalibrated=0.1, odds=10.0),
        _row(race_id="Race 2 - BEN - 2026-06-10", dog="D", box=1, winner=True, primary=0.1, stage2=0.8, stage2_uncalibrated=0.8, odds=6.0),
        _row(race_id="Race 2 - BEN - 2026-06-10", dog="E", box=2, primary=0.8, stage2=0.1, stage2_uncalibrated=0.1, odds=5.0),
        _row(race_id="Race 2 - BEN - 2026-06-10", dog="F", box=3, primary=0.1, stage2=0.1, stage2_uncalibrated=0.1, odds=10.0),
    ]
    report_path = _write_dataset(tmp_path, "unified_evidence_dataset_hypothesis", rows)

    report = comparison.build_comparison(
        unified_evidence_report_paths=[report_path],
        output_dir=(
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525"
            / "rolling_model_comparison_hypothesis"
        ),
        min_races_for_review=2,
        generated_at=datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc),
    )

    hypothesis = report["residual_hypothesis_backtests"][0]
    assert hypothesis["status"] == "EXPLORATORY_EVALUATED"
    assert hypothesis["promotion_eligible"] is False
    assert hypothesis["triggered_race_count"] == 2
    assert "post_hoc_residual_hypothesis_not_promotion_eligible" in hypothesis[
        "blockers"
    ]
    assert "triggered_race_count_below_directional_floor" in hypothesis["blockers"]
    assert hypothesis["candidate_minus_market"]["top1"] == 1.0
    assert report["candidate_count"] == 22
    assert hypothesis["candidate_key"] not in report["candidate_metrics_by_key"]


def test_edge_diagnostics_surface_market_disagreement_lift(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "ROOT", tmp_path)
    rows = [
        _row(race_id="Race 1 - BEN - 2026-06-10", dog="A", box=1, winner=True, primary=0.2, stage2=0.8, odds=7.0),
        _row(race_id="Race 1 - BEN - 2026-06-10", dog="B", box=2, primary=0.7, stage2=0.1, odds=1.8),
        _row(race_id="Race 1 - BEN - 2026-06-10", dog="C", box=3, primary=0.1, stage2=0.1, odds=10.0),
        _row(race_id="Race 2 - BEN - 2026-06-10", dog="D", box=1, winner=True, primary=0.2, stage2=0.8, odds=6.0),
        _row(race_id="Race 2 - BEN - 2026-06-10", dog="E", box=2, primary=0.7, stage2=0.1, odds=1.9),
        _row(race_id="Race 2 - BEN - 2026-06-10", dog="F", box=3, primary=0.1, stage2=0.1, odds=12.0),
    ]
    report_path = _write_dataset(tmp_path, "unified_evidence_dataset_edge", rows)

    report = comparison.build_comparison(
        unified_evidence_report_paths=[report_path],
        output_dir=(
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525"
            / "rolling_model_comparison_edge"
        ),
        min_races_for_review=2,
        generated_at=datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc),
    )

    diagnostics = report["edge_diagnostics"]
    selected_key = diagnostics["selected_candidate_key"]
    assert selected_key.startswith("stage2_shadow")

    disagreement_slices = diagnostics["dimensions"]["selected_market_agreement"]
    assert disagreement_slices[0]["slice_key"] == (
        "selected_top_differs_from_market_favourite"
    )
    assert disagreement_slices[0]["race_count"] == 2
    assert disagreement_slices[0]["candidate_metrics_by_key"][selected_key]["top1"] == 1.0
    assert disagreement_slices[0]["candidate_metrics_by_key"]["stage2_shadow"]["top1"] == 1.0
    assert disagreement_slices[0]["candidate_metrics_by_key"]["market_only_implied"]["top1"] == 0.0
    assert disagreement_slices[0]["selected_minus_market"]["top1"] == 1.0

    rank_slices = diagnostics["dimensions"]["selected_top_market_rank_band"]
    assert rank_slices[0]["slice_key"] == "selected_top_market_rank_2_3"
    assert rank_slices[0]["race_count"] == 2


def test_rolling_comparison_computes_calibration_slope_intercept_at_sample_floor(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(comparison, "ROOT", tmp_path)
    rows = []
    race_probabilities = {
        1: [0.30, 0.25, 0.20, 0.10, 0.08, 0.07],
        2: [0.25, 0.22, 0.20, 0.15, 0.10, 0.08],
        3: [0.30, 0.25, 0.15, 0.14, 0.10, 0.06],
        4: [0.35, 0.20, 0.18, 0.12, 0.10, 0.05],
        5: [0.40, 0.18, 0.15, 0.12, 0.10, 0.05],
    }
    winners = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    for race_index, probabilities in race_probabilities.items():
        for box, probability in enumerate(probabilities, start=1):
            winner = box == winners[race_index]
            rows.append(
                _row(
                    race_id=f"Race {race_index} - BEN - 2026-06-10",
                    dog=f"Dog {race_index}-{box}",
                    box=box,
                    winner=winner,
                    primary=probability,
                    stage2=probability,
                    odds=2.0 if winner else 8.0 + box,
                )
            )
    report_path = _write_dataset(tmp_path, "unified_evidence_dataset_calibration", rows)

    report = comparison.build_comparison(
        unified_evidence_report_paths=[report_path],
        output_dir=(
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525"
            / "rolling_model_comparison_calibration"
        ),
        min_races_for_review=5,
        generated_at=datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc),
    )

    calibration = report["candidate_metrics_by_key"]["stage2_shadow"][
        "calibration_slope_intercept"
    ]
    assert calibration["status"] == "computed"
    assert calibration["sample_size"] == 30
    assert calibration["positive_labels"] == 5
    assert calibration["negative_labels"] == 25
    assert isinstance(calibration["slope"], float)
    assert isinstance(calibration["intercept"], float)
