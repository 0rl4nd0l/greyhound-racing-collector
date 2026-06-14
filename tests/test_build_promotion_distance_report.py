import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import build_promotion_distance_report as distance


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_promotion_distance_report_quantifies_blocked_accuracy_path(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(distance, "ROOT", tmp_path)
    rolling = _write_json(
        tmp_path / "rolling.json",
        {
            "sample_race_count": 140,
            "sample_runner_rows": 953,
            "best_candidate_key": "market_only_implied",
            "best_non_market_candidate_key": "stage2_uncalibrated_market_blend_75",
            "best_non_market_minus_market": {
                "top1": -0.007,
                "top3": -0.02,
                "mean_winner_rank": 0.0,
                "brier": 0.001,
                "logloss": 0.006,
            },
            "source_rejected_live_odds_candidate_count": 5,
            "source_rows_with_rejected_live_odds_candidates": 4,
            "source_rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 2,
                "odds_source_url_missing": 3,
            },
            "source_exclusion_reason_counts": {"official_result_missing": 32},
            "source_odds_exclusion_reason_counts": {"strict_prejump_odds_missing": 4},
            "source_official_result_evidence_db_missing_race_ids": [
                "Race 3 - TAREE - 2026-06-13"
            ],
            "source_official_result_evidence_db_requested_race_ids": [
                "Race 3 - TAREE - 2026-06-13",
                "Race 5 - TAREE - 2026-06-13",
                "Race 6 - TAREE - 2026-06-13",
            ],
            "source_official_result_evidence_db_requested_race_count": 7,
            "source_official_result_evidence_db_legacy_requested_race_count_without_ids": 4,
            "source_official_result_evidence_db_races_with_rows": [
                "Race 5 - TAREE - 2026-06-13",
                "Race 6 - TAREE - 2026-06-13",
            ],
            "source_official_result_runner_paths": [
                "artifacts/full_evidence_orchestration_20260525/autonomous_official_result_capture_test/official_result_runners.jsonl"
            ],
        },
    )
    gated = _write_json(
        tmp_path / "gated.json",
        {
            "predeclared_residual_candidate": {
                "candidate_key": "market_favourite_gt_4_0__raw_stage2_market_blend_75",
                "status": "PREDECLARED_RESIDUAL_CANDIDATE_COLLECTING",
                "triggered_race_count": 3,
                "minimum_triggered_races_for_directional_read": 10,
                "directional_read_ready": False,
                "candidate_minus_market": {
                    "top1": 0.0,
                    "top3": 0.0,
                    "mean_winner_rank": -0.007,
                    "brier": -0.001,
                    "logloss": -0.0015,
                },
                "blockers": [
                    "predeclared_residual_candidate_report_only",
                    "triggered_race_count_below_directional_floor",
                    "top1_not_above_market",
                ],
            },
        },
    )
    gate = _write_json(
        tmp_path / "gate.json",
        {
            "status": "BLOCKED",
            "blockers": ["no_candidate_passed_rank_first_accuracy_gate"],
        },
    )
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525"
        / "promotion_distance_report_test"
    )

    report = distance.build_report(
        rolling_report_path=rolling,
        pre_race_gated_report_path=gated,
        high_accuracy_gate_path=gate,
        output_dir=output_dir,
        generated_at=datetime(2026, 6, 12, 8, 45, tzinfo=timezone.utc),
    )

    assert report["final_status"] == "PROMOTION_DISTANCE_BLOCKED"
    assert report["rolling_sample"]["sample_race_count"] == 140
    assert report["rolling_sample"]["races_needed_for_review_floor"] == 0
    assert report["rolling_sample"]["source_rejected_live_odds_candidate_count"] == 5
    assert (
        report["rolling_sample"]["source_rows_with_rejected_live_odds_candidates"] == 4
    )
    assert report["rolling_sample"][
        "source_rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 2,
        "odds_source_url_missing": 3,
    }
    assert report["rolling_sample"]["source_exclusion_reason_counts"] == {
        "official_result_missing": 32
    }
    assert report["rolling_sample"]["source_odds_exclusion_reason_counts"] == {
        "strict_prejump_odds_missing": 4
    }
    assert report["rolling_sample"][
        "source_official_result_evidence_db_missing_race_ids"
    ] == ["Race 3 - TAREE - 2026-06-13"]
    assert report["rolling_sample"][
        "source_official_result_evidence_db_requested_race_count"
    ] == 7
    assert report["rolling_sample"][
        "source_official_result_evidence_db_races_with_rows"
    ] == [
        "Race 5 - TAREE - 2026-06-13",
        "Race 6 - TAREE - 2026-06-13",
    ]
    assert report["rolling_sample"]["source_official_result_runner_paths"] == [
        "artifacts/full_evidence_orchestration_20260525/autonomous_official_result_capture_test/official_result_runners.jsonl"
    ]
    assert report["official_result_coverage"] == {
        "source": "rolling_model_comparison",
        "requested_race_count": 7,
        "requested_race_count_source": "rolling_model_comparison_source_count",
        "requested_race_ids": [
            "Race 3 - TAREE - 2026-06-13",
            "Race 5 - TAREE - 2026-06-13",
            "Race 6 - TAREE - 2026-06-13",
        ],
        "legacy_requested_race_count_without_ids": 4,
        "races_with_rows_count": 2,
        "missing_race_count": 1,
        "missing_race_ids": ["Race 3 - TAREE - 2026-06-13"],
        "races_with_rows": [
            "Race 5 - TAREE - 2026-06-13",
            "Race 6 - TAREE - 2026-06-13",
        ],
        "runner_path_count": 1,
        "runner_paths_source_field": (
            "rolling_sample.source_official_result_runner_paths"
        ),
        "missing_exclusion_count": 32,
    }
    assert report["market_benchmark"]["best_candidate_key"] == "market_only_implied"
    assert report["market_benchmark"]["best_non_market_top1_margin_gap"] == 0.027
    residual = report["predeclared_residual_candidate"]
    assert residual["triggered_race_count"] == 3
    assert residual["triggered_races_needed_for_directional_read"] == 7
    assert residual["directional_read_ready"] is False
    assert report["promotion_ready"] is False
    assert "no_candidate_passed_rank_first_accuracy_gate" in report["blockers"]
    assert "predeclared_residual_trigger_count_below_directional_floor" in report[
        "blockers"
    ]
    assert report["no_write_guarantees"]["production_promotion"] is False
    assert report["no_write_guarantees"]["db_write"] is False
    assert report["no_write_guarantees"]["betting_action"] is False
    summary = (output_dir / "SUMMARY.md").read_text(encoding="utf-8")
    assert "Rolling source rejected live odds candidates: `5`" in summary
    assert "Rolling source rows with rejected live odds candidates: `4`" in summary
    assert (
        "Rolling source rejected live odds candidate reasons: `{'odds_decimal_invalid': 2, 'odds_source_url_missing': 3}`"
        in summary
    )
    assert "Rolling source exclusion reasons: `{'official_result_missing': 32}`" in summary
    assert "Official-result coverage requested races: `7`" in summary
    assert (
        "Official-result coverage requested race count source: "
        "`rolling_model_comparison_source_count`"
    ) in summary
    assert "Official-result legacy requested race count without IDs: `4`" in summary
    assert "Official-result coverage races with rows: `2`" in summary
    assert "Official-result coverage missing races: `1`" in summary
    assert "Official-result missing exclusion count: `32`" in summary
    assert "Official-result runner path count: `1`" in summary
    assert (
        "Official-result runner paths source field: "
        "`rolling_sample.source_official_result_runner_paths`"
    ) in summary
    assert "Rolling source official-result runner paths:" not in summary
    assert "autonomous_official_result_capture_test/official_result_runners.jsonl" not in summary
    assert "Race 3 - TAREE - 2026-06-13" in summary
    assert (output_dir / "promotion_distance_report.json").exists()
    assert (output_dir / "SUMMARY.md").exists()
    assert (output_dir / "output_manifest.json").exists()
