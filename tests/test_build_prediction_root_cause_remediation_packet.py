import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import build_prediction_root_cause_remediation_packet as packet


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_prediction_root_cause_remediation_packet_classifies_fail_closed_axes(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(packet, "ROOT", tmp_path)
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"

    aggregate = _write_json(
        evidence_root / "aggregate/forward_shadow_result_aggregate_report.json",
        {
            "final_status": "PARTIAL_AGGREGATE_PENDING_MORE_RESULTS",
            "unsafe_result_matches": {
                "unsafe_match_count": 2,
                "unsafe_result_matches": [
                    {
                        "race_id": "Race 1 - TEST - 2026-06-16",
                        "name_mismatches": [{"box": 7}],
                        "missing_predicted_boxes": [],
                        "disallowed_extra_official_boxes": [{"box": 9}],
                        "allowed_extra_scratched_official_boxes": [],
                        "prejump_runner_alignment": {
                            "remapped_participants": [{"dog_name": "Reserve"}],
                            "dropped_participants": [],
                        },
                        "source_join_artifact": "join-a",
                    },
                    {
                        "race_id": "Race 2 - TEST - 2026-06-16",
                        "name_mismatches": [],
                        "missing_predicted_boxes": [],
                        "disallowed_extra_official_boxes": [],
                        "allowed_extra_scratched_official_boxes": [{"box": 9}],
                        "prejump_runner_alignment": {
                            "remapped_participants": [{"dog_name": "Reserve"}],
                            "dropped_participants": [{"dog_name": "Scratched"}],
                        },
                        "source_join_artifact": "join-b",
                    },
                ],
            },
        },
    )
    promotion = _write_json(
        evidence_root / "promotion/promotion_distance_report.json",
        {
            "final_status": "PROMOTION_DISTANCE_BLOCKED",
            "rolling_sample": {
                "sample_race_count": 12,
                "minimum_races_for_review": 100,
                "races_needed_for_review_floor": 88,
            },
        },
    )
    high_accuracy = _write_json(
        evidence_root / "high/high_accuracy_refinement_packet.json",
        {
            "final_status": "BLOCKED_KEEP_BASELINE",
            "odds_research_gate_summary": {
                "status": "ODDS_RESEARCH_BLOCKED_PROVENANCE",
                "complete_valid_prejump_odds_races": 1,
            },
        },
    )
    rolling = _write_json(
        evidence_root / "rolling/rolling_model_comparison_report.json",
        {
            "final_status": "ROLLING_MODEL_COMPARISON_COLLECTING",
            "market_candidate_key": "market_only_implied",
            "best_candidate_key": "stage2_market_blend_50",
            "candidate_metrics_by_key": {
                "market_only_implied": {
                    "candidate_key": "market_only_implied",
                    "family": "market_only",
                    "race_count": 12,
                    "top1": 0.40,
                    "top3": 0.75,
                    "mean_winner_rank": 2.4,
                    "logloss": 1.50,
                    "brier": 0.72,
                    "calibration_slope_intercept": {"slope": 0.9},
                },
                "stage2_market_blend_50": {
                    "candidate_key": "stage2_market_blend_50",
                    "family": "odds_augmented_blend",
                    "race_count": 12,
                    "top1": 0.45,
                    "top3": 0.70,
                    "mean_winner_rank": 2.3,
                    "logloss": 1.60,
                    "brier": 0.75,
                    "calibration_slope_intercept": {"slope": 1.3},
                },
            },
        },
    )
    feature_gate = _write_json(
        evidence_root / "feature/feature_activation_gate_report.json",
        {
            "final_status": "FEATURE_ACTIVATION_BLOCKED_KEEP_QUARANTINED",
            "features": [
                {
                    "feature": "same_distance_same_grade_best_time",
                    "fail_reasons": [
                        "all_missing_in_train",
                        "same_distance_same_grade_best_time_unsafe_history_source:missing",
                    ],
                    "parity": {
                        "train_present_rows": 0,
                        "train_rows": 20,
                        "train_present_pct": 0.0,
                        "holdout_present_rows": 3,
                        "holdout_rows": 10,
                        "holdout_present_pct": 0.3,
                        "parity_status": "ALL_MISSING_IN_TRAIN_PRESENT_IN_HOLDOUT",
                    },
                },
                {
                    "feature": "recent_avg_time_5",
                    "fail_reasons": ["all_missing_in_train"],
                    "parity": {
                        "train_present_rows": 0,
                        "train_rows": 20,
                        "holdout_present_rows": 0,
                        "holdout_rows": 10,
                    },
                },
            ],
        },
    )
    ablation = _write_csv(
        evidence_root / "ablation/ablation_metrics.csv",
        [
            {
                "scope": "market_comparable_holdout",
                "candidate_key": "market_only_implied",
                "feature_set": "",
                "race_count": 12,
                "runner_rows": 84,
                "top1_minus_market": "",
                "top3_minus_market": "",
                "mean_winner_rank_minus_market": "",
                "race_winner_logloss_minus_market": "",
                "brier_minus_market": "",
            },
            {
                "scope": "market_comparable_holdout",
                "candidate_key": "non_box_no_shadow",
                "feature_set": "non_box_no_shadow",
                "race_count": 12,
                "runner_rows": 84,
                "top1_minus_market": -0.2,
                "top3_minus_market": -0.3,
                "mean_winner_rank_minus_market": 1.2,
                "race_winner_logloss_minus_market": 0.4,
                "brier_minus_market": 0.1,
            },
        ],
    )
    residual = _write_csv(
        evidence_root / "residual/split_summary.csv",
        [
            {
                "dimension": "market_favourite_odds_band",
                "dimension_value": "market_favourite_odds_4_8",
                "pre_race_usable": "True",
                "race_count": 2,
                "rank_first_net_edge_count": 1,
                "mean_candidate_minus_market_logloss": -0.2,
                "mean_winner_rank_delta": -0.5,
            }
        ],
    )

    output_dir = evidence_root / "prediction_root_cause_remediation_test"
    report = packet.build_packet(
        aggregate_report_path=aggregate,
        promotion_report_path=promotion,
        high_accuracy_report_path=high_accuracy,
        rolling_report_path=rolling,
        feature_gate_report_path=feature_gate,
        ablation_metrics_path=ablation,
        residual_split_summary_path=residual,
        output_dir=output_dir,
        min_review_races=100,
        min_residual_trigger_races=10,
        generated_at=datetime(2026, 6, 16, 13, 0, tzinfo=timezone.utc),
    )

    assert report["schema_version"] == "prediction_root_cause_remediation_packet_v1"
    assert report["final_status"] == "ROOT_CAUSE_REMEDIATION_PACKET_BUILT_REPORT_ONLY"
    assert report["next_decision"] == "IDENTITY_LABEL_CLEANUP_NEXT"
    assert report["promotion_ready"] is False
    assert report["no_write_guarantees"]["db_write"] is False
    assert report["no_write_guarantees"]["model_training"] is False
    assert "unsafe_identity_matches_require_cleanup" in report["blockers"]
    assert "objective_probability_tradeoff_not_safe" in report["blockers"]
    assert report["feature_decision_counts"] == {
        "DATA_MISSING": 1,
        "KEEP_QUARANTINED": 1,
    }
    assert report["feature_family_status_counts"]["NO_FAMILY_LIFT_VS_MARKET"] == 1
    assert report["residual_predeclare_status_counts"]["KEEP_COLLECTING_ONLY"] == 1

    identity_rows = list(csv.DictReader((output_dir / "identity_noise_ledger.csv").open()))
    assert identity_rows[0]["metric_decision"] == "EXCLUDE_UNTIL_REJOIN"
    assert identity_rows[1]["metric_decision"] == "NEEDS_CANONICAL_REMAP_RULE"

    objective_rows = list(csv.DictReader((output_dir / "objective_metric_split.csv").open()))
    blend = [row for row in objective_rows if row["candidate_key"] == "stage2_market_blend_50"][0]
    assert "TOP1_ONLY_TRADEOFF" in blend["failure_classification"]
    assert "PROBABILITY_CALIBRATION_BAD" in blend["failure_classification"]

    assert (output_dir / "remediation_report.json").exists()
    assert (output_dir / "SUMMARY.md").exists()
    assert (output_dir / "output_manifest.json").exists()
