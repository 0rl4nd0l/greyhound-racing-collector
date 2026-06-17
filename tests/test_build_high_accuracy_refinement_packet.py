import json
from datetime import datetime, timezone

from scripts import build_high_accuracy_refinement_packet as packet


def _metrics(
    *,
    races: int = 120,
    top1: float = 0.20,
    top3: float = 0.55,
    mean_winner_rank: float = 3.8,
    brier: float = 0.12,
    logloss: float = 1.9,
    slope: float = 0.6,
    intercept: float = -0.4,
    box1: float = 0.25,
    candidate_key: str | None = None,
    family: str | None = None,
) -> dict:
    metrics = {
        "safe_joined_race_count": races,
        "top1": top1,
        "top3": top3,
        "mean_winner_rank": mean_winner_rank,
        "brier": brier,
        "logloss": logloss,
        "probability_sum_max_error_joined_races": 0.0,
        "box1_top_pick_share": box1,
        "calibration_slope_intercept": {
            "status": "fit",
            "slope": slope,
            "intercept": intercept,
        },
    }
    if candidate_key is not None:
        metrics["candidate_key"] = candidate_key
    if family is not None:
        metrics["family"] = family
    return metrics


def test_stage2_forward_metrics_can_open_pr_only_gate_without_direct_switch():
    stage2_forward_metrics = {
        "status": packet.STAGE2_FORWARD_SHADOW_READY_FOR_REVIEW,
        "baseline_forward_shadow_metrics": _metrics(),
        "stage2_challenger_forward_shadow_metrics": _metrics(
            top1=0.24,
            top3=0.58,
            mean_winner_rank=3.5,
            brier=0.11,
            logloss=1.8,
            slope=0.8,
            intercept=-0.2,
            box1=0.22,
        ),
    }

    result = packet.build_packet(
        stage2_forward_metrics=stage2_forward_metrics,
        thresholds=packet.AccuracyGateThresholds(
            min_safe_joined_races=100,
            min_top1_delta=0.02,
        ),
        generated_at=datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc),
        protected_before={"model_registry/best_metadata.json": "abc"},
        protected_after={"model_registry/best_metadata.json": "abc"},
    )

    assert result["final_status"] == packet.FINAL_READY_FOR_PR
    assert result["stages"]["non_tgr_model_challenger"]["gate"]["status"] == "PASS"
    assert result["promotion_pr_gate"]["status"] == "READY_FOR_PR_DRAFT"
    assert result["promotion_pr_gate"]["selected_stage"] == "stage_2_non_tgr_model_challenger"
    boundary = result["promotion_pr_gate"]["pull_request_boundary"]
    assert boundary["promotion_pr_allowed"] is True
    assert boundary["direct_local_switch_allowed"] is False
    assert boundary["local_registry_mutation_allowed"] is False
    assert boundary["production_pointer_update_allowed"] is False
    assert result["no_write_guarantees"]["odds_used_for_shadow_scoring"] is False
    assert result["ev_diagnostics_summary"]["ev_metrics_used_for_promotion"] is False


def test_stage2_prediction_artifact_is_first_class_without_joined_metrics(tmp_path):
    stage2_predictions_path = tmp_path / "stage2_shadow_predictions.jsonl"
    stage2_predictions_path.write_text(
        json.dumps({"race_id": "Race 1 - TEST - 2026-06-13"}) + "\n"
        + json.dumps({"race_id": "Race 1 - TEST - 2026-06-13"}) + "\n",
        encoding="utf-8",
    )

    result = packet.build_packet(
        stage2_predictions_path=stage2_predictions_path,
        generated_at=datetime(2026, 6, 13, 10, 0, tzinfo=timezone.utc),
        protected_before={},
        protected_after={},
    )

    stage2 = result["stages"]["non_tgr_model_challenger"]
    assert stage2["status"] == packet.STAGE2_PREDICTIONS_COLLECTED_METRICS_MISSING
    assert stage2["stage2_prediction_rows"] == 2
    assert stage2["stage2_predictions_path"].endswith("stage2_shadow_predictions.jsonl")
    assert stage2["gate"]["status"] == "BLOCKED"
    assert stage2["gate"]["blockers"] == ["stage2_forward_joined_metrics_missing"]
    assert result["promotion_pr_gate"]["status"] == "BLOCKED"
    summary = packet.build_summary(result)
    assert (
        "Stage 2 status: `STAGE2_PREDICTIONS_COLLECTED_METRICS_MISSING`"
        in summary
    )
    assert "Stage 2 prediction rows: `2`" in summary
    assert "stage2_forward_joined_metrics_missing" in summary


def test_stage2_forward_metrics_are_derived_from_pure_rolling_stage2():
    rolling_report = {
        "final_status": packet.ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW,
        "sample_scope": "unified",
        "sample_race_count": 120,
        "sample_floor_met": True,
        "source_unified_evidence_reports": [
            "artifacts/full_evidence_orchestration_20260525/unified/report.json"
        ],
        "candidate_metrics_by_key": {
            "primary_shadow": _metrics(
                candidate_key="primary_shadow",
                family="baseline",
            )
            | {"status": "EVALUATED", "race_count": 120},
            "stage2_shadow": _metrics(
                top1=0.24,
                top3=0.58,
                mean_winner_rank=3.5,
                brier=0.11,
                logloss=1.8,
                slope=0.8,
                intercept=-0.2,
                box1=0.22,
                candidate_key="stage2_shadow",
                family="stage2",
            )
            | {"status": "EVALUATED", "race_count": 120},
            "stage2_market_blend_50": _metrics(
                candidate_key="stage2_market_blend_50",
                family="odds_augmented_blend",
            )
            | {"status": "EVALUATED", "race_count": 120},
        },
    }

    result = packet.build_packet(
        odds_augmented_report=rolling_report,
        thresholds=packet.AccuracyGateThresholds(
            min_safe_joined_races=100,
            min_top1_delta=0.02,
        ),
        generated_at=datetime(2026, 6, 13, 10, 0, tzinfo=timezone.utc),
        protected_before={},
        protected_after={},
    )

    stage2 = result["stages"]["non_tgr_model_challenger"]
    assert stage2["source_status"] == packet.STAGE2_FORWARD_SHADOW_READY_FOR_REVIEW
    assert stage2["gate"]["status"] == "PASS"
    assert stage2["gate"]["candidate_metrics"]["candidate_key"] == "stage2_shadow"
    assert "stage2_forward_joined_metrics_missing" not in stage2["gate"]["blockers"]
    assert result["source_artifacts"]["stage2_forward_metrics"] is None
    assert (
        result["source_artifacts"]["stage2_forward_metrics_source"]
        == packet.STAGE2_FORWARD_METRICS_FROM_ROLLING
    )
    assert result["promotion_pr_gate"]["status"] == "READY_FOR_PR_DRAFT"
    assert result["no_write_guarantees"]["odds_used_for_shadow_scoring"] is False


def test_odds_ev_improvement_cannot_override_rank_or_box_bias_regression():
    odds_gate_report = {
        "status": packet.ODDS_RESEARCH_READY_REPORT_ONLY,
        "complete_valid_prejump_odds_races": 120,
        "odds_used_for_shadow_scoring": False,
    }
    odds_augmented_report = {
        "final_status": packet.ODDS_AUGMENTED_MODEL_READY_FOR_PR_REVIEW,
        "baseline_metrics": _metrics(top1=0.25, top3=0.60, box1=0.20),
        "candidate_metrics": _metrics(
            top1=0.22,
            top3=0.58,
            mean_winner_rank=4.1,
            brier=0.13,
            logloss=2.0,
            slope=0.2,
            intercept=-1.2,
            box1=0.42,
        ),
        "ev_improved": True,
    }
    ev_diagnostics = {
        "status": "EV_DIAGNOSTICS_REPORT_ONLY",
        "ev_rows": 840,
        "mean_ev": 0.08,
    }

    result = packet.build_packet(
        odds_gate_report=odds_gate_report,
        odds_augmented_report=odds_augmented_report,
        ev_diagnostics=ev_diagnostics,
        thresholds=packet.AccuracyGateThresholds(
            min_safe_joined_races=100,
            min_top1_delta=0.02,
        ),
        generated_at=datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc),
        protected_before={},
        protected_after={},
    )

    odds_stage = result["stages"]["odds_augmented_model"]
    blockers = odds_stage["gate"]["blockers"]
    assert odds_stage["status"] == packet.ODDS_AUGMENTED_MODEL_BLOCKED
    assert "rank_accuracy_top1_delta_below_min" in blockers
    assert "metric_regressed:top3" in blockers
    assert "metric_regressed:mean_winner_rank" in blockers
    assert "metric_regressed:calibration_slope_intercept" in blockers
    assert "candidate_box1_top_pick_share_above_max" in blockers
    assert "ev_improvement_ignored_because_accuracy_guardrails_failed" in blockers
    assert result["promotion_pr_gate"]["status"] == "BLOCKED"
    assert result["ev_diagnostics_summary"]["ev_metrics_used_for_promotion"] is False
    assert result["ev_diagnostics_summary"]["ev_can_override_accuracy_gate"] is False


def test_ready_rolling_comparison_can_satisfy_cumulative_odds_gate():
    odds_gate_report = {
        "status": packet.ODDS_RESEARCH_BLOCKED_PROVENANCE,
        "complete_valid_prejump_odds_races": 1,
        "odds_used_for_shadow_scoring": False,
    }
    rolling_report = {
        "final_status": packet.ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW,
        "sample_scope": "unified",
        "sample_race_count": 120,
        "minimum_races_for_review": 100,
        "sample_floor_met": True,
        "races_needed_for_review": 0,
        "candidate_count": 22,
        "best_candidate_key": "stage2_uncalibrated_market_blend_75",
        "best_non_baseline_candidate_key": "stage2_uncalibrated_market_blend_75",
        "rank_first_sort": [
            "stage2_uncalibrated_market_blend_75",
            "market_only_implied",
        ],
        "baseline_metrics": _metrics(
            races=120,
            top1=0.20,
            top3=0.55,
            mean_winner_rank=3.7,
            brier=0.16,
            logloss=1.9,
            slope=0.6,
            intercept=-0.4,
            box1=0.30,
        ),
        "candidate_metrics": _metrics(
            races=120,
            candidate_key="stage2_uncalibrated_market_blend_75",
            top1=0.25,
            top3=0.60,
            mean_winner_rank=3.2,
            brier=0.12,
            logloss=1.6,
            slope=0.9,
            intercept=-0.1,
            box1=0.25,
        ),
        "candidate_metrics_by_key": {
            "stage2_uncalibrated_market_blend_75": _metrics(
                races=120,
                candidate_key="stage2_uncalibrated_market_blend_75",
                top1=0.25,
                top3=0.60,
                mean_winner_rank=3.2,
                brier=0.12,
                logloss=1.6,
                slope=0.9,
                intercept=-0.1,
                box1=0.25,
            ),
            "market_only_implied": _metrics(
                races=120,
                candidate_key="market_only_implied",
                top1=0.24,
                top3=0.59,
            ),
        },
        "source_artifact_odds_rows_seen": 41041,
        "source_artifact_odds_rows_accepted": 9820,
        "source_artifact_odds_rows_rejected": 31221,
        "source_artifact_odds_rejection_reason_counts": {
            "odds_match_status_not_valid_pre_jump_dog_odds": 31221,
        },
    }

    result = packet.build_packet(
        odds_gate_report=odds_gate_report,
        odds_augmented_report=rolling_report,
        thresholds=packet.AccuracyGateThresholds(
            min_safe_joined_races=100,
            min_top1_delta=0.02,
        ),
        generated_at=datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc),
        protected_before={},
        protected_after={},
    )

    odds_stage = result["stages"]["odds_augmented_model"]
    assert odds_stage["status"] == packet.ODDS_AUGMENTED_MODEL_READY_FOR_PR_REVIEW
    assert odds_stage["odds_evidence_source"] == "cumulative_rolling_model_comparison"
    assert odds_stage["cumulative_odds_evidence"]["ready"] is True
    assert odds_stage["rolling_model_comparison"]["sample_floor_met"] is True
    assert odds_stage["rolling_model_comparison"]["races_needed_for_review"] == 0
    assert odds_stage["rolling_model_comparison"]["candidate_count"] == 22
    assert odds_stage["rolling_model_comparison"]["rank_first_sort"] == [
        "stage2_uncalibrated_market_blend_75",
        "market_only_implied",
    ]
    assert odds_stage["rolling_model_comparison"]["candidate_metrics_by_key"][
        "stage2_uncalibrated_market_blend_75"
    ]["top1"] == 0.25
    assert odds_stage["rolling_model_comparison"]["source_artifact_odds_rows_seen"] == 41041
    assert (
        odds_stage["rolling_model_comparison"]["source_artifact_odds_rows_accepted"]
        == 9820
    )
    assert (
        odds_stage["rolling_model_comparison"]["source_artifact_odds_rows_rejected"]
        == 31221
    )
    assert odds_stage["rolling_model_comparison"][
        "source_artifact_odds_rejection_reason_counts"
    ] == {
        "odds_match_status_not_valid_pre_jump_dog_odds": 31221,
    }
    assert odds_stage["gate"]["status"] == "PASS"
    summary = packet.build_summary(result)
    assert "- Odds evidence source: `cumulative_rolling_model_comparison`" in summary
    assert "- Cumulative odds evidence ready: `True`" in summary
    assert "- Cumulative odds evidence sample races: `120`" in summary
    assert "- Cumulative odds evidence races needed: `0`" in summary
    assert "- Cumulative source artifact odds rows seen: `41041`" in summary
    assert "- Cumulative source artifact odds rows accepted: `9820`" in summary
    assert "- Cumulative source artifact odds rows rejected: `31221`" in summary
    assert (
        "- Cumulative source artifact odds rejection reasons: `{'odds_match_status_not_valid_pre_jump_dog_odds': 31221}`"
        in summary
    )
    assert not any(
        str(blocker).startswith("odds_research_gate_not_ready")
        for blocker in odds_stage["gate"]["blockers"]
    )
    assert result["promotion_pr_gate"]["status"] == "READY_FOR_PR_DRAFT"


def test_market_only_rolling_winner_cannot_open_model_promotion_pr_gate():
    odds_gate_report = {
        "status": packet.ODDS_RESEARCH_BLOCKED_PROVENANCE,
        "complete_valid_prejump_odds_races": 3,
        "odds_used_for_shadow_scoring": False,
    }
    rolling_report = {
        "final_status": packet.ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW,
        "sample_scope": "unified",
        "sample_race_count": 108,
        "minimum_races_for_review": 100,
        "sample_floor_met": True,
        "races_needed_for_review": 0,
        "candidate_count": 22,
        "best_candidate_key": "market_only_implied",
        "best_non_baseline_candidate_key": "market_only_implied",
        "rank_first_sort": [
            "market_only_implied",
            "stage2_uncalibrated_market_blend_75",
        ],
        "baseline_metrics": _metrics(
            races=108,
            top1=0.20,
            top3=0.55,
            mean_winner_rank=3.7,
            brier=0.16,
            logloss=1.9,
            slope=0.6,
            intercept=-0.4,
            box1=0.30,
        ),
        "candidate_metrics": _metrics(
            races=108,
            candidate_key="market_only_implied",
            top1=0.4537037037037037,
            top3=0.8055555555555556,
            mean_winner_rank=2.3055555555555554,
            brier=0.7091622075405496,
            logloss=1.5235542754581377,
            slope=0.9,
            intercept=-0.1,
            box1=0.25,
        ),
        "candidate_metrics_by_key": {
            "market_only_implied": _metrics(
                races=108,
                candidate_key="market_only_implied",
                top1=0.4537037037037037,
                top3=0.8055555555555556,
            ),
            "stage2_uncalibrated_market_blend_75": _metrics(
                races=108,
                candidate_key="stage2_uncalibrated_market_blend_75",
                top1=0.4444444444444444,
                top3=0.8055555555555556,
            ),
        },
    }

    result = packet.build_packet(
        odds_gate_report=odds_gate_report,
        odds_augmented_report=rolling_report,
        thresholds=packet.AccuracyGateThresholds(
            min_safe_joined_races=100,
            min_top1_delta=0.02,
        ),
        generated_at=datetime(2026, 6, 12, 4, 50, tzinfo=timezone.utc),
        protected_before={},
        protected_after={},
    )

    odds_stage = result["stages"]["odds_augmented_model"]
    blockers = odds_stage["gate"]["blockers"]
    assert odds_stage["status"] == packet.ODDS_AUGMENTED_MODEL_BLOCKED
    assert odds_stage["candidate_key"] == "market_only_implied"
    assert "market_only_candidate_not_promotable" in blockers
    assert "rolling_best_candidate_is_market_only" in blockers
    assert "rolling_best_non_baseline_candidate_is_market_only" in blockers
    assert "rolling_rank_first_candidate_is_market_only" in blockers
    assert result["promotion_pr_gate"]["status"] == "BLOCKED"
    assert result["final_status"] == packet.FINAL_BLOCKED


def test_collecting_rolling_comparison_does_not_bypass_latest_odds_gate():
    odds_gate_report = {
        "status": packet.ODDS_RESEARCH_BLOCKED_PROVENANCE,
        "complete_valid_prejump_odds_races": 1,
        "odds_used_for_shadow_scoring": False,
    }
    rolling_report = {
        "final_status": packet.ROLLING_MODEL_COMPARISON_COLLECTING,
        "sample_scope": "unified",
        "sample_race_count": 49,
        "minimum_races_for_review": 100,
        "sample_floor_met": False,
        "races_needed_for_review": 51,
        "candidate_count": 22,
        "best_candidate_key": "market_only_implied",
        "best_non_baseline_candidate_key": "market_only_implied",
        "rank_first_sort": [
            "market_only_implied",
            "stage2_uncalibrated_market_blend_75",
        ],
        "baseline_metrics": _metrics(races=49),
        "candidate_metrics": _metrics(
            races=49,
            candidate_key="market_only_implied",
            top1=0.25,
            top3=0.60,
            mean_winner_rank=3.2,
            brier=0.10,
            logloss=1.6,
            slope=0.9,
            intercept=-0.1,
            box1=0.25,
        ),
        "candidate_metrics_by_key": {
            "market_only_implied": _metrics(
                races=49,
                candidate_key="market_only_implied",
                top1=0.25,
                top3=0.60,
            ),
            "stage2_uncalibrated_market_blend_75": _metrics(
                races=49,
                candidate_key="stage2_uncalibrated_market_blend_75",
                top1=0.23,
                top3=0.58,
            ),
        },
    }

    result = packet.build_packet(
        odds_gate_report=odds_gate_report,
        odds_augmented_report=rolling_report,
        thresholds=packet.AccuracyGateThresholds(
            min_safe_joined_races=100,
            min_top1_delta=0.02,
        ),
        generated_at=datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc),
        protected_before={},
        protected_after={},
    )

    odds_stage = result["stages"]["odds_augmented_model"]
    blockers = odds_stage["gate"]["blockers"]
    assert odds_stage["status"] == packet.ODDS_AUGMENTED_MODEL_BLOCKED
    assert odds_stage["cumulative_odds_evidence"]["ready"] is False
    assert odds_stage["cumulative_odds_evidence"]["sample_floor_met"] is False
    assert odds_stage["cumulative_odds_evidence"]["races_needed_for_review"] == 51
    assert odds_stage["rolling_model_comparison"]["status"] == (
        packet.ROLLING_MODEL_COMPARISON_COLLECTING
    )
    assert odds_stage["rolling_model_comparison"]["sample_floor_met"] is False
    assert odds_stage["rolling_model_comparison"]["candidate_count"] == 22
    assert odds_stage["rolling_model_comparison"]["rank_first_sort"][0] == (
        "market_only_implied"
    )
    assert "candidate_race_sample_below_min" in blockers
    assert "odds_research_gate_not_ready:ODDS_RESEARCH_BLOCKED_PROVENANCE" in blockers
    assert "cumulative_odds_evidence_status_not_ready:ROLLING_MODEL_COMPARISON_COLLECTING" in blockers
    assert "cumulative_odds_evidence_races_below_min" in blockers
    assert result["promotion_pr_gate"]["status"] == "BLOCKED"


def test_unified_evidence_summary_records_current_collection_gate():
    result = packet.build_packet(
        unified_evidence_report={
            "final_status": "UNIFIED_EVIDENCE_DATASET_BUILT",
            "row_count": 23,
            "race_count": 3,
            "rows_with_stage2_predictions": 23,
            "rows_with_strict_prejump_odds": 23,
            "rows_with_artifact_shadow_odds": 6,
            "rows_with_artifact_shadow_odds_candidates": 8,
            "artifact_shadow_odds_candidate_count": 8,
            "artifact_shadow_odds_selected_bucket_count": 6,
            "artifact_odds_rows_seen": 23,
            "artifact_odds_rows_accepted": 6,
            "artifact_odds_rows_rejected": 17,
            "artifact_odds_audits": [
                {
                    "rejection_reason_counts": {
                        "odds_match_status_not_valid_pre_jump_dog_odds": 17
                    }
                }
            ],
            "exclusion_reason_counts": {"official_result_missing": 12},
            "odds_exclusion_reason_counts": {"strict_prejump_odds_missing": 3},
            "official_result_evidence_db_audit": {
                "race_ids_requested": 3,
                "missing_race_ids": ["Race 3 - TAREE - 2026-06-13"],
                "race_ids_with_rows": [
                    "Race 5 - TAREE - 2026-06-13",
                    "Race 6 - TAREE - 2026-06-13",
                ],
            },
            "official_result_runner_paths": [
                "artifacts/full_evidence_orchestration_20260525/autonomous_official_result_capture_test/official_result_runners.jsonl"
            ],
            "rejected_live_odds_candidate_count": 5,
            "rows_with_rejected_live_odds_candidates": 4,
            "rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 2,
                "odds_source_url_missing": 3,
            },
            "rows_with_official_results": 11,
            "stage2_evaluation_eligible_rows": 11,
            "odds_evaluation_eligible_rows": 11,
            "label_evaluation_eligible_rows": 11,
            "unified_evidence_eligible_rows": 11,
            "no_write_guarantees": {
                "db_write": False,
                "label_write": False,
                "production_promotion": False,
            },
        },
        generated_at=datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc),
        protected_before={},
        protected_after={},
    )

    summary = result["unified_evidence_summary"]
    assert summary["status"] == "UNIFIED_EVIDENCE_COLLECTING"
    assert summary["unified_evidence_eligible_rows"] == 11
    assert summary["rows_with_strict_prejump_odds"] == 23
    assert summary["rows_with_artifact_shadow_odds"] == 6
    assert summary["rows_with_artifact_shadow_odds_candidates"] == 8
    assert summary["artifact_shadow_odds_candidate_count"] == 8
    assert summary["artifact_shadow_odds_selected_bucket_count"] == 6
    assert summary["artifact_odds_rows_seen"] == 23
    assert summary["artifact_odds_rows_accepted"] == 6
    assert summary["artifact_odds_rows_rejected"] == 17
    assert summary["artifact_odds_rejection_reason_counts"] == {
        "odds_match_status_not_valid_pre_jump_dog_odds": 17
    }
    assert summary["exclusion_reason_counts"] == {"official_result_missing": 12}
    assert summary["odds_exclusion_reason_counts"] == {
        "strict_prejump_odds_missing": 3
    }
    assert summary["official_result_evidence_db_requested_race_count"] == 3
    assert summary["official_result_evidence_db_missing_race_ids"] == [
        "Race 3 - TAREE - 2026-06-13"
    ]
    assert summary["official_result_evidence_db_races_with_rows"] == [
        "Race 5 - TAREE - 2026-06-13",
        "Race 6 - TAREE - 2026-06-13",
    ]
    assert summary["official_result_runner_paths"] == [
        "artifacts/full_evidence_orchestration_20260525/autonomous_official_result_capture_test/official_result_runners.jsonl"
    ]
    assert summary["official_result_coverage"] == {
        "source": "unified_evidence_report",
        "requested_race_count": 3,
        "races_with_rows_count": 2,
        "missing_race_count": 1,
        "missing_race_ids": ["Race 3 - TAREE - 2026-06-13"],
        "races_with_rows": [
            "Race 5 - TAREE - 2026-06-13",
            "Race 6 - TAREE - 2026-06-13",
        ],
        "runner_path_count": 1,
        "runner_paths_source_field": "official_result_runner_paths",
        "missing_exclusion_count": 12,
    }
    assert summary["rejected_live_odds_candidate_count"] == 5
    assert summary["rows_with_rejected_live_odds_candidates"] == 4
    assert summary["rejected_live_odds_candidate_reason_counts"] == {
        "odds_decimal_invalid": 2,
        "odds_source_url_missing": 3,
    }
    assert "unified_evidence_eligible_rows_below_review_floor" in summary["blockers"]
    assert result["promotion_pr_gate"]["status"] == "BLOCKED"


def test_unified_evidence_summary_preserves_direct_official_result_coverage():
    direct_coverage = {
        "source": "unified_evidence_dataset",
        "requested_race_count": 12,
        "races_with_rows_count": 0,
        "missing_race_count": 12,
        "missing_race_ids": [
            "Race 1 - MOUNT - 2026-06-14",
            "Race 2 - HEA - 2026-06-14",
        ],
        "races_with_rows": [],
        "runner_path_count": 1,
        "runner_paths_source_field": "official_result_runner_paths",
        "missing_exclusion_count": 85,
        "requested_race_count_source": "official_result_evidence_db_audit_requested_race_ids",
        "requested_race_ids": [
            "Race 1 - MOUNT - 2026-06-14",
            "Race 2 - HEA - 2026-06-14",
        ],
    }

    result = packet.build_packet(
        unified_evidence_report={
            "final_status": "UNIFIED_EVIDENCE_DATASET_BUILT",
            "row_count": 85,
            "race_count": 12,
            "rows_with_stage2_predictions": 85,
            "rows_with_official_results": 0,
            "rows_with_strict_prejump_odds": 21,
            "unified_evidence_eligible_rows": 0,
            "official_result_coverage": direct_coverage,
            "official_result_evidence_db_audit": {
                "race_ids_requested": 12,
                "missing_race_ids": ["legacy-fallback-should-not-win"],
                "race_ids_with_rows": ["legacy-fallback-should-not-win"],
            },
            "official_result_runner_paths": ["legacy-fallback-should-not-win.jsonl"],
            "exclusion_reason_counts": {"official_result_missing": 85},
            "no_write_guarantees": {
                "db_write": False,
                "label_write": False,
                "production_promotion": False,
            },
        },
        generated_at=datetime(2026, 6, 14, 1, 0, tzinfo=timezone.utc),
        protected_before={},
        protected_after={},
    )

    summary = result["unified_evidence_summary"]
    assert summary["status"] == "UNIFIED_EVIDENCE_COLLECTING"
    assert summary["official_result_coverage"] == direct_coverage
    summary_text = packet.build_summary(result)
    assert (
        "- Unified official-result coverage requested race count source: "
        "`official_result_evidence_db_audit_requested_race_ids`"
    ) in summary_text
    assert result["promotion_pr_gate"]["status"] == "BLOCKED"


def test_backlog_unified_evidence_summary_surfaces_aggregate_without_opening_gate():
    result = packet.build_packet(
        unified_evidence_report={
            "final_status": "UNIFIED_EVIDENCE_DATASET_BUILT",
            "row_count": 104,
            "race_count": 14,
            "unified_evidence_eligible_rows": 98,
            "no_write_guarantees": {"db_write": False, "label_write": False},
        },
        backlog_unified_evidence_status={
            "status": "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT",
            "aggregation_scope": "per_dataset_totals_not_cross_dataset_deduped",
            "attempted_dataset_count": 20,
            "dataset_count": 20,
            "failed_dataset_count": 0,
            "row_count": 1905,
            "race_count": 268,
            "rows_with_official_results": 434,
            "rows_with_strict_prejump_odds": 1599,
            "rows_with_artifact_shadow_odds": 79,
            "artifact_odds_rows_seen": 250,
            "artifact_odds_rows_accepted": 79,
            "artifact_odds_rows_rejected": 171,
            "artifact_odds_rejection_reason_counts": {
                "odds_match_status_not_valid_pre_jump_dog_odds": 171
            },
            "exclusion_reason_counts": {"official_result_missing": 80},
            "odds_exclusion_reason_counts": {"strict_prejump_odds_missing": 28},
            "official_result_evidence_db_missing_race_ids": [
                "Race 8 - TAREE - 2026-06-13"
            ],
            "official_result_evidence_db_requested_race_count": 14,
            "official_result_evidence_db_races_with_rows": [
                "Race 4 - TAREE - 2026-06-13",
                "Race 7 - TAREE - 2026-06-13",
            ],
            "official_result_coverage": {
                "source": "backlog_unified_evidence_dataset_reports",
                "requested_race_count": 14,
                "requested_race_count_source": (
                    "deduped_backlog_unified_evidence_official_result_coverage_requested_race_ids"
                ),
                "requested_race_ids": [
                    "Race 4 - TAREE - 2026-06-13",
                    "Race 7 - TAREE - 2026-06-13",
                ],
                "legacy_requested_race_count_without_ids": 0,
                "races_with_rows_count": 2,
                "missing_race_count": 1,
                "missing_race_ids": ["Race 8 - TAREE - 2026-06-13"],
                "races_with_rows": [
                    "Race 4 - TAREE - 2026-06-13",
                    "Race 7 - TAREE - 2026-06-13",
                ],
                "runner_path_count": 1,
                "runner_paths_source_field": "official_result_runner_paths",
                "missing_exclusion_count": 80,
            },
            "rejected_live_odds_candidate_count": 9,
            "rows_with_rejected_live_odds_candidates": 6,
            "rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 4,
                "odds_source_url_missing": 5,
            },
            "sample_blocking_gap_count": 4,
            "gap_action_counts": {"inspect_quarantined_official_result_runner_set": 4},
            "evidence_missing_reason_counts": {
                "official_result_quarantined_unsafe_match": 4
            },
            "top_gap_race_ids": [
                "Race 7 - TAREE - 2026-06-13",
                "Race 4 - TAREE - 2026-06-13",
            ],
            "top_gap_races": [
                {
                    "race_id": "Race 7 - TAREE - 2026-06-13",
                    "race_date": "2026-06-13",
                    "venue": "TAREE",
                    "recommended_action": "inspect_quarantined_official_result_runner_set",
                    "evidence_missing_reason": (
                        "official_result_quarantined_unsafe_match"
                    ),
                    "official_result_quarantine_reason": (
                        "result_boxes_not_in_participants"
                    ),
                    "official_result_quarantine_participant_source": (
                        "shadow_run_predictions"
                    ),
                    "official_result_quarantine_participant_boxes": [
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                    ],
                    "official_result_quarantine_result_boxes_not_in_participants": [
                        9
                    ],
                    "official_result_quarantine_attempted_source_box_sets": [
                        {
                            "source": "thedogs_official",
                            "result_boxes": [2, 8, 4, 7, 3, 9, 6, 5],
                            "dog_names_by_box": {},
                            "terminal_status_boxes": [1, 10],
                        }
                    ],
                    "official_result_quarantine_reserve_substitution_diagnostic": {
                        "classification": (
                            "possible_reserve_substitution_manual_review_required"
                        ),
                        "acceptance_status": "not_accepted_report_only",
                        "candidate_reserve_boxes": [9],
                        "scratched_participant_boxes": [1],
                    },
                    "debug_blob": "not propagated",
                }
            ],
            "top_official_result_missing_race_ids": [
                "Race 7 - TAREE - 2026-06-13"
            ],
            "top_official_result_missing_races": [
                {
                    "race_id": "Race 7 - TAREE - 2026-06-13",
                    "recommended_action": "inspect_quarantined_official_result_runner_set",
                    "evidence_missing_reason": (
                        "official_result_quarantined_unsafe_match"
                    ),
                }
            ],
            "unified_evidence_eligible_rows": 415,
            "no_write_guarantees": {"db_write": False, "label_write": False},
        },
        generated_at=datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc),
        protected_before={},
        protected_after={},
    )

    current = result["unified_evidence_summary"]
    backlog = result["backlog_unified_evidence_summary"]
    assert current["status"] == "UNIFIED_EVIDENCE_COLLECTING"
    assert current["unified_evidence_eligible_rows"] == 98
    assert backlog["status"] == "BACKLOG_UNIFIED_EVIDENCE_READY_FOR_REVIEW"
    assert backlog["unified_evidence_eligible_rows"] == 415
    assert backlog["rows_with_artifact_shadow_odds"] == 79
    assert backlog["artifact_odds_rows_seen"] == 250
    assert backlog["artifact_odds_rows_accepted"] == 79
    assert backlog["artifact_odds_rows_rejected"] == 171
    assert backlog["artifact_odds_rejection_reason_counts"] == {
        "odds_match_status_not_valid_pre_jump_dog_odds": 171
    }
    assert backlog["exclusion_reason_counts"] == {"official_result_missing": 80}
    assert backlog["odds_exclusion_reason_counts"] == {
        "strict_prejump_odds_missing": 28
    }
    assert backlog["official_result_evidence_db_missing_race_ids"] == [
        "Race 8 - TAREE - 2026-06-13"
    ]
    assert backlog["official_result_evidence_db_requested_race_count"] == 14
    assert backlog["official_result_evidence_db_races_with_rows"] == [
        "Race 4 - TAREE - 2026-06-13",
        "Race 7 - TAREE - 2026-06-13",
    ]
    assert backlog["official_result_coverage"] == {
        "source": "backlog_unified_evidence_dataset_reports",
        "requested_race_count": 14,
        "requested_race_count_source": (
            "deduped_backlog_unified_evidence_official_result_coverage_requested_race_ids"
        ),
        "requested_race_ids": [
            "Race 4 - TAREE - 2026-06-13",
            "Race 7 - TAREE - 2026-06-13",
        ],
        "legacy_requested_race_count_without_ids": 0,
        "races_with_rows_count": 2,
        "missing_race_count": 1,
        "missing_race_ids": ["Race 8 - TAREE - 2026-06-13"],
        "races_with_rows": [
            "Race 4 - TAREE - 2026-06-13",
            "Race 7 - TAREE - 2026-06-13",
        ],
        "runner_path_count": 1,
        "runner_paths_source_field": "official_result_runner_paths",
        "missing_exclusion_count": 80,
    }
    assert backlog["rejected_live_odds_candidate_count"] == 9
    assert backlog["rows_with_rejected_live_odds_candidates"] == 6
    assert backlog["rejected_live_odds_candidate_reason_counts"] == {
        "odds_decimal_invalid": 4,
        "odds_source_url_missing": 5,
    }
    assert backlog["sample_blocking_gap_count"] == 4
    assert backlog["gap_action_counts"] == {
        "inspect_quarantined_official_result_runner_set": 4
    }
    assert backlog["gap_evidence_missing_reason_counts"] == {
        "official_result_quarantined_unsafe_match": 4
    }
    assert backlog["top_gap_race_ids"] == [
        "Race 7 - TAREE - 2026-06-13",
        "Race 4 - TAREE - 2026-06-13",
    ]
    assert backlog["top_gap_races"] == [
        {
            "race_id": "Race 7 - TAREE - 2026-06-13",
            "race_date": "2026-06-13",
            "venue": "TAREE",
            "recommended_action": "inspect_quarantined_official_result_runner_set",
            "evidence_missing_reason": "official_result_quarantined_unsafe_match",
            "official_result_quarantine_reason": "result_boxes_not_in_participants",
            "official_result_quarantine_participant_source": (
                "shadow_run_predictions"
            ),
            "official_result_quarantine_participant_boxes": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
            ],
            "official_result_quarantine_result_boxes_not_in_participants": [9],
            "official_result_quarantine_attempted_source_box_sets": [
                {
                    "source": "thedogs_official",
                    "result_boxes": [2, 8, 4, 7, 3, 9, 6, 5],
                    "dog_names_by_box": {},
                    "terminal_status_boxes": [1, 10],
                }
            ],
            "official_result_quarantine_reserve_substitution_diagnostic": {
                "classification": (
                    "possible_reserve_substitution_manual_review_required"
                ),
                "acceptance_status": "not_accepted_report_only",
                "candidate_reserve_boxes": [9],
                "scratched_participant_boxes": [1],
            },
        }
    ]
    assert backlog["top_official_result_missing_race_ids"] == [
        "Race 7 - TAREE - 2026-06-13"
    ]
    assert backlog["top_official_result_missing_races"] == [
        {
            "race_id": "Race 7 - TAREE - 2026-06-13",
            "recommended_action": "inspect_quarantined_official_result_runner_set",
            "evidence_missing_reason": "official_result_quarantined_unsafe_match",
        }
    ]
    assert backlog["aggregation_scope"] == "per_dataset_totals_not_cross_dataset_deduped"
    assert backlog["blockers"] == []
    summary_text = packet.build_summary(result)
    assert "- Backlog official-result coverage requested races: `14`" in summary_text
    assert (
        "- Backlog official-result requested race count source: "
        "`deduped_backlog_unified_evidence_official_result_coverage_requested_race_ids`"
    ) in summary_text
    assert (
        "- Backlog official-result coverage missing races: `1`"
        in summary_text
    )
    assert result["promotion_pr_gate"]["status"] == "BLOCKED"
    assert "no_candidate_passed_rank_first_accuracy_gate" in result["promotion_pr_gate"]["blockers"]


def test_rejoin_unified_evidence_summary_is_accepted_as_aggregate_evidence():
    result = packet.build_packet(
        backlog_unified_evidence_status={
            "status": "REJOIN_UNIFIED_EVIDENCE_DATASETS_BUILT",
            "attempted_dataset_count": 3,
            "dataset_count": 3,
            "failed_dataset_count": 0,
            "row_count": 250,
            "race_count": 34,
            "rows_with_official_results": 232,
            "rows_with_strict_prejump_odds": 250,
            "rows_with_artifact_shadow_odds": 79,
            "artifact_odds_rows_seen": 250,
            "artifact_odds_rows_accepted": 79,
            "artifact_odds_rows_rejected": 171,
            "artifact_odds_rejection_reason_counts": {
                "odds_match_status_not_valid_pre_jump_dog_odds": 171
            },
            "rejected_live_odds_candidate_count": 11,
            "rows_with_rejected_live_odds_candidates": 7,
            "rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 6,
                "unsupported_sportsbet_box_source:missing": 5,
            },
            "unified_evidence_eligible_rows": 232,
            "no_write_guarantees": {"db_write": False, "label_write": False},
        },
        generated_at=datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc),
        protected_before={},
        protected_after={},
    )

    summary = result["backlog_unified_evidence_summary"]
    assert summary["status"] == "BACKLOG_UNIFIED_EVIDENCE_READY_FOR_REVIEW"
    assert summary["source_status"] == "REJOIN_UNIFIED_EVIDENCE_DATASETS_BUILT"
    assert summary["unified_evidence_eligible_rows"] == 232
    assert summary["artifact_odds_rows_accepted"] == 79
    assert summary["artifact_odds_rejection_reason_counts"] == {
        "odds_match_status_not_valid_pre_jump_dog_odds": 171
    }
    assert summary["rejected_live_odds_candidate_count"] == 11
    assert summary["rows_with_rejected_live_odds_candidates"] == 7
    assert summary["rejected_live_odds_candidate_reason_counts"] == {
        "odds_decimal_invalid": 6,
        "unsupported_sportsbet_box_source:missing": 5,
    }
    assert summary["blockers"] == []
    assert result["promotion_pr_gate"]["status"] == "BLOCKED"


def test_unified_evidence_summary_fails_closed_on_write_guard_regression():
    result = packet.build_packet(
        unified_evidence_report={
            "final_status": "UNIFIED_EVIDENCE_DATASET_BUILT",
            "unified_evidence_eligible_rows": 120,
            "no_write_guarantees": {"db_write": True},
        },
        generated_at=datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc),
        protected_before={},
        protected_after={},
    )

    summary = result["unified_evidence_summary"]
    assert summary["status"] == "UNIFIED_EVIDENCE_COLLECTING"
    assert "unified_evidence_write_guard_not_blocked:db_write" in summary["blockers"]


def test_promotion_distance_summary_records_market_relative_blockers():
    result = packet.build_packet(
        promotion_distance_report={
            "final_status": "PROMOTION_DISTANCE_BLOCKED",
            "promotion_ready": False,
            "blockers": [
                "no_candidate_passed_rank_first_accuracy_gate",
                "best_non_market_top1_margin_below_target",
                "predeclared_residual_trigger_count_below_directional_floor",
            ],
            "rolling_sample": {
                "sample_race_count": 124,
                "sample_runner_rows": 856,
                "minimum_races_for_review": 100,
                "source_rejected_live_odds_candidate_count": 5,
                "source_rows_with_rejected_live_odds_candidates": 4,
                "source_rejected_live_odds_candidate_reason_counts": {
                    "odds_decimal_invalid": 2,
                    "odds_source_url_missing": 3,
                },
                "source_exclusion_reason_counts": {"official_result_missing": 32},
                "source_odds_exclusion_reason_counts": {
                    "strict_prejump_odds_missing": 4
                },
                "source_official_result_evidence_db_missing_race_ids": [
                    "Race 3 - TAREE - 2026-06-13"
                ],
                "source_official_result_evidence_db_requested_race_count": 7,
                "source_official_result_evidence_db_races_with_rows": [
                    "Race 5 - TAREE - 2026-06-13",
                    "Race 6 - TAREE - 2026-06-13",
                ],
                "source_official_result_runner_paths": [
                    "artifacts/full_evidence_orchestration_20260525/autonomous_official_result_capture_test/official_result_runners.jsonl"
                ],
            },
            "official_result_coverage": {
                "source": "rolling_model_comparison",
                "requested_race_count": 7,
                "requested_race_count_source": "deduped_requested_or_inferred_race_ids",
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
            },
            "market_benchmark": {
                "best_candidate_key": "market_only_implied",
                "best_non_market_candidate_key": "stage2_uncalibrated_market_blend_50",
                "target_top1_margin_vs_market": 0.02,
                "best_non_market_top1_margin_gap": 0.02,
                "best_non_market_minus_market": {
                    "top1": 0.0,
                    "top3": -0.04,
                },
            },
            "predeclared_residual_candidate": {
                "candidate_key": "market_favourite_gt_4_0__raw_stage2_market_blend_75",
                "status": "PREDECLARED_RESIDUAL_CANDIDATE_COLLECTING",
                "triggered_race_count": 2,
                "minimum_triggered_races_for_directional_read": 10,
                "triggered_races_needed_for_directional_read": 8,
                "directional_read_ready": False,
                "candidate_minus_market": {"top1": 0.0},
            },
            "no_write_guarantees": {"db_write": False, "production_promotion": False},
        },
        generated_at=datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc),
        protected_before={},
        protected_after={},
    )

    summary = result["promotion_distance_summary"]
    assert summary["status"] == "PROMOTION_DISTANCE_BLOCKED"
    assert summary["promotion_ready"] is False
    assert summary["sample_race_count"] == 124
    assert summary["source_rejected_live_odds_candidate_count"] == 5
    assert summary["source_rows_with_rejected_live_odds_candidates"] == 4
    assert summary["source_rejected_live_odds_candidate_reason_counts"] == {
        "odds_decimal_invalid": 2,
        "odds_source_url_missing": 3,
    }
    assert summary["source_exclusion_reason_counts"] == {
        "official_result_missing": 32
    }
    assert summary["source_odds_exclusion_reason_counts"] == {
        "strict_prejump_odds_missing": 4
    }
    assert summary["source_official_result_evidence_db_missing_race_ids"] == [
        "Race 3 - TAREE - 2026-06-13"
    ]
    assert summary["source_official_result_evidence_db_requested_race_count"] == 7
    assert summary["source_official_result_evidence_db_races_with_rows"] == [
        "Race 5 - TAREE - 2026-06-13",
        "Race 6 - TAREE - 2026-06-13",
    ]
    assert summary["source_official_result_runner_paths"] == [
        "artifacts/full_evidence_orchestration_20260525/autonomous_official_result_capture_test/official_result_runners.jsonl"
    ]
    assert summary["official_result_coverage"] == {
        "source": "rolling_model_comparison",
        "requested_race_count": 7,
        "requested_race_count_source": "deduped_requested_or_inferred_race_ids",
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
    assert summary["best_candidate_key"] == "market_only_implied"
    assert summary["best_non_market_candidate_key"] == (
        "stage2_uncalibrated_market_blend_50"
    )
    assert summary["best_non_market_top1_margin_gap"] == 0.02
    assert summary["predeclared_residual_triggered_race_count"] == 2
    assert summary["predeclared_residual_triggered_races_needed_for_directional_read"] == 8
    assert "best_non_market_top1_margin_below_target" in summary["blockers"]
    assert result["promotion_pr_gate"]["status"] == "BLOCKED"


def test_reserve_substitution_preflight_summary_is_report_only():
    result = packet.build_packet(
        reserve_substitution_preflight={
            "final_status": "RESERVE_SUBSTITUTION_PREFLIGHT_READY_FOR_POLICY_REVIEW",
            "candidate_count": 2,
            "blocked_candidate_count": 0,
            "ready_for_policy_review_count": 2,
            "blocker_counts": {},
            "candidates": [
                {
                    "race_id": "Race 7 - TAREE - 2026-06-13",
                    "preflight_status": "READY_FOR_MANUAL_POLICY_REVIEW",
                    "policy_review_status": "READY_FOR_MANUAL_POLICY_REVIEW",
                    "acceptance_status": "not_accepted_report_only",
                    "acceptance_effect": "none_report_only",
                    "candidate_reserve_boxes": [9],
                    "scratched_participant_boxes": [1],
                    "readiness_blockers": [],
                    "dataset_join_blockers": [
                        "official_result_remains_quarantined",
                        "manual_policy_review_required_before_join",
                    ],
                    "source_url": "https://www.thedogs.com.au/example",
                },
                {
                    "race_id": "Race 8 - TAREE - 2026-06-13",
                    "preflight_status": "READY_FOR_MANUAL_POLICY_REVIEW",
                    "policy_review_status": "READY_FOR_MANUAL_POLICY_REVIEW",
                    "acceptance_status": "not_accepted_report_only",
                    "acceptance_effect": "none_report_only",
                    "candidate_reserve_boxes": [9, 10],
                    "scratched_participant_boxes": [5, 6],
                    "readiness_blockers": [],
                    "dataset_join_blockers": [
                        "official_result_remains_quarantined",
                        "manual_policy_review_required_before_join",
                    ],
                    "source_url": "https://www.thedogs.com.au/example-8",
                },
            ],
            "no_write_guarantees": {
                "db_write": False,
                "label_write": False,
                "canonical_result_label_write": False,
                "official_result_acceptance": False,
                "quarantine_bypass": False,
                "snapshot_mutation": False,
                "manifest_mutation": False,
                "model_training": False,
                "registry_mutation": False,
                "production_promotion": False,
                "betting_action": False,
                "ev_action": False,
                "tgr_enabled": False,
            },
        },
        reserve_substitution_manual_review={
            "final_status": "RESERVE_SUBSTITUTION_MANUAL_REVIEW_PACKET_READY",
            "candidate_count": 2,
            "ready_candidate_count": 2,
            "blocked_candidate_count": 0,
            "ready_race_ids": [
                "Race 7 - TAREE - 2026-06-13",
                "Race 8 - TAREE - 2026-06-13",
            ],
            "blocked_race_ids": [],
            "automatic_acceptance_allowed": False,
            "dataset_join_allowed": False,
            "official_result_acceptance_allowed": False,
            "db_write": False,
            "candidates": [
                {
                    "race_id": "Race 7 - TAREE - 2026-06-13",
                    "packet_blockers": [],
                    "mapping_hypothesis": {
                        "mapping_status": "report_only_policy_hypothesis",
                        "mapping_acceptance_status": "not_accepted",
                        "mapping_blockers": [],
                        "pairs": [
                            {
                                "scratched_participant_box": 1,
                                "reserve_box": 9,
                                "reserve_dog_name": "Reserve Runner",
                                "mapping_acceptance_status": "not_accepted",
                            }
                        ],
                    },
                },
                {
                    "race_id": "Race 8 - TAREE - 2026-06-13",
                    "packet_blockers": [],
                    "mapping_hypothesis": {
                        "mapping_status": "report_only_policy_hypothesis",
                        "mapping_acceptance_status": "not_accepted",
                        "mapping_blockers": [],
                        "pairs": [
                            {
                                "scratched_participant_box": 5,
                                "reserve_box": 9,
                                "reserve_dog_name": "Reserve A",
                                "mapping_acceptance_status": "not_accepted",
                            },
                            {
                                "scratched_participant_box": 6,
                                "reserve_box": 10,
                                "reserve_dog_name": "Reserve B",
                                "mapping_acceptance_status": "not_accepted",
                            },
                        ],
                    },
                },
            ],
            "no_write_guarantees": {
                "db_write": False,
                "label_write": False,
                "canonical_result_label_write": False,
                "official_result_acceptance": False,
                "quarantine_bypass": False,
                "snapshot_mutation": False,
                "manifest_mutation": False,
                "model_training": False,
                "registry_mutation": False,
                "production_promotion": False,
                "betting_action": False,
                "ev_action": False,
                "tgr_enabled": False,
            },
        },
        reserve_substitution_policy_impact={
            "final_status": "RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY",
            "approval_required": True,
            "automatic_acceptance_allowed": False,
            "dataset_join_allowed": False,
            "official_result_acceptance_allowed": False,
            "db_write": False,
            "candidate_count": 2,
            "ready_candidate_count": 2,
            "ready_race_ids": [
                "Race 7 - TAREE - 2026-06-13",
                "Race 8 - TAREE - 2026-06-13",
            ],
            "mapping_pair_count": 3,
            "potential_official_result_runner_rows_blocked_by_policy": 16,
            "matched_backlog_top_gap_race_count": 2,
            "matched_backlog_top_gap_race_ids": [
                "Race 7 - TAREE - 2026-06-13",
                "Race 8 - TAREE - 2026-06-13",
            ],
            "backlog_sample_blocking_gap_count": 2,
            "backlog_gap_action_counts": {
                "inspect_quarantined_official_result_runner_set": 2,
            },
            "backlog_evidence_missing_reason_counts": {
                "official_result_quarantined_unsafe_match": 2,
            },
            "current_effect": "none_report_only_all_results_remain_quarantined",
            "preview_effect_if_policy_approved_later": (
                "listed_races_could_be_reconsidered_for_explicit_reserve_"
                "substitution_join_rule_after_policy_approval_only"
            ),
            "no_write_guarantees": {
                "db_write": False,
                "label_write": False,
                "canonical_result_label_write": False,
                "official_result_acceptance": False,
                "quarantine_bypass": False,
                "snapshot_mutation": False,
                "manifest_mutation": False,
                "model_training": False,
                "registry_mutation": False,
                "production_promotion": False,
                "betting_action": False,
                "ev_action": False,
                "tgr_enabled": False,
            },
        },
        generated_at=datetime(2026, 6, 14, 6, 0, tzinfo=timezone.utc),
        protected_before={},
        protected_after={},
    )

    summary = result["reserve_substitution_preflight_summary"]
    assert summary["status"] == (
        "RESERVE_SUBSTITUTION_PREFLIGHT_READY_FOR_POLICY_REVIEW"
    )
    assert summary["candidate_count"] == 2
    assert summary["ready_for_policy_review_count"] == 2
    assert summary["blocked_candidate_count"] == 0
    assert summary["readiness_blocker_counts"] == {}
    assert summary["dataset_join_blocker_counts"] == {
        "manual_policy_review_required_before_join": 2,
        "official_result_remains_quarantined": 2,
    }
    assert summary["ready_race_ids"] == [
        "Race 7 - TAREE - 2026-06-13",
        "Race 8 - TAREE - 2026-06-13",
    ]
    assert summary["blockers"] == []
    manual_review = result["reserve_substitution_manual_review_summary"]
    assert manual_review["status"] == "RESERVE_SUBSTITUTION_MANUAL_REVIEW_PACKET_READY"
    assert manual_review["ready_candidate_count"] == 2
    assert manual_review["mapping_pair_count"] == 3
    assert manual_review["dataset_join_allowed"] is False
    assert manual_review["official_result_acceptance_allowed"] is False
    assert manual_review["db_write"] is False
    assert manual_review["blockers"] == []
    policy_impact = result["reserve_substitution_policy_impact_summary"]
    assert policy_impact["status"] == (
        "RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY"
    )
    assert policy_impact["ready_candidate_count"] == 2
    assert policy_impact["mapping_pair_count"] == 3
    assert (
        policy_impact["potential_official_result_runner_rows_blocked_by_policy"]
        == 16
    )
    assert policy_impact["matched_backlog_top_gap_race_count"] == 2
    assert policy_impact["dataset_join_allowed"] is False
    assert policy_impact["official_result_acceptance_allowed"] is False
    assert policy_impact["db_write"] is False
    assert policy_impact["blockers"] == []
    assert result["promotion_pr_gate"]["status"] == "BLOCKED"
    assert "no_candidate_passed_rank_first_accuracy_gate" in result[
        "promotion_pr_gate"
    ]["blockers"]
    summary_text = packet.build_summary(result)
    assert (
        "- Reserve substitution preflight: "
        "`RESERVE_SUBSTITUTION_PREFLIGHT_READY_FOR_POLICY_REVIEW`"
    ) in summary_text
    assert "- Reserve substitution ready for policy review: `2`" in summary_text
    assert (
        "- Reserve substitution dataset join blockers: "
        "`{'manual_policy_review_required_before_join': 2, "
        "'official_result_remains_quarantined': 2}`"
    ) in summary_text
    assert (
        "- Reserve substitution manual review: "
        "`RESERVE_SUBSTITUTION_MANUAL_REVIEW_PACKET_READY`"
    ) in summary_text
    assert "- Reserve substitution manual review mapping pairs: `3`" in summary_text
    assert (
        "- Reserve substitution manual review dataset join allowed: `False`"
        in summary_text
    )
    assert (
        "- Reserve substitution policy impact: "
        "`RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY`"
    ) in summary_text
    assert (
        "- Reserve substitution policy impact potential runner rows blocked: `16`"
        in summary_text
    )
    assert (
        "- Reserve substitution policy impact dataset join allowed: `False`"
        in summary_text
    )


def test_run_packet_writes_report_only_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "ROOT", tmp_path)
    monkeypatch.setattr(packet, "DEFAULT_PROTECTED_PATHS", ())
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525"
        / "high_accuracy_refinement_packet_20260610T010000+0000"
    )
    stage2_forward_metrics_path = tmp_path / "stage2_forward_joined_metrics.json"
    stage2_forward_metrics_path.write_text(
        json.dumps(
            {
                "status": packet.STAGE2_FORWARD_SHADOW_COLLECTING,
                "baseline_forward_shadow_metrics": _metrics(),
                "stage2_challenger_forward_shadow_metrics": _metrics(races=20),
            }
        ),
        encoding="utf-8",
    )
    unified_evidence_report_path = tmp_path / "unified_evidence_dataset_report.json"
    unified_evidence_report_path.write_text(
        json.dumps(
            {
                "final_status": "UNIFIED_EVIDENCE_DATASET_BUILT",
                "unified_evidence_eligible_rows": 11,
                "rows_with_official_results": 11,
                "rows_with_strict_prejump_odds": 23,
                "rows_with_artifact_shadow_odds": 6,
                "artifact_odds_rows_seen": 23,
                "artifact_odds_rows_accepted": 6,
                "artifact_odds_rows_rejected": 17,
                "artifact_odds_rejection_reason_counts": {
                    "odds_match_status_not_valid_pre_jump_dog_odds": 17
                },
                "exclusion_reason_counts": {"official_result_missing": 12},
                "odds_exclusion_reason_counts": {"strict_prejump_odds_missing": 3},
                "official_result_evidence_db_audit": {
                    "race_ids_requested": 3,
                    "missing_race_ids": ["Race 3 - TAREE - 2026-06-13"],
                    "race_ids_with_rows": [
                        "Race 5 - TAREE - 2026-06-13",
                        "Race 6 - TAREE - 2026-06-13",
                    ],
                },
                "official_result_runner_paths": [
                    "artifacts/full_evidence_orchestration_20260525/autonomous_official_result_capture_test/official_result_runners.jsonl"
                ],
                "rejected_live_odds_candidate_count": 5,
                "rows_with_rejected_live_odds_candidates": 4,
                "rejected_live_odds_candidate_reason_counts": {
                    "odds_decimal_invalid": 2,
                    "odds_source_url_missing": 3,
                },
                "rows_with_stage2_predictions": 23,
            }
        ),
        encoding="utf-8",
    )
    backlog_unified_evidence_status_path = tmp_path / "backlog_unified_evidence_datasets_status.json"
    backlog_unified_evidence_status_path.write_text(
        json.dumps(
            {
                "status": "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT",
                "aggregation_scope": "per_dataset_totals_not_cross_dataset_deduped",
                "dataset_count": 3,
                "failed_dataset_count": 0,
                "unified_evidence_eligible_rows": 120,
                "rows_with_artifact_shadow_odds": 79,
                "artifact_odds_rows_seen": 250,
                "artifact_odds_rows_accepted": 79,
                "artifact_odds_rows_rejected": 171,
                "artifact_odds_rejection_reason_counts": {
                    "odds_match_status_not_valid_pre_jump_dog_odds": 171
                },
                "exclusion_reason_counts": {"official_result_missing": 80},
                "odds_exclusion_reason_counts": {"strict_prejump_odds_missing": 28},
                "official_result_evidence_db_missing_race_ids": [
                    "Race 8 - TAREE - 2026-06-13"
                ],
                "rejected_live_odds_candidate_count": 9,
                "rows_with_rejected_live_odds_candidates": 6,
                "rejected_live_odds_candidate_reason_counts": {
                    "odds_decimal_invalid": 4,
                    "odds_source_url_missing": 5,
                },
                "race_coverage": {
                    "gap_action_plan": {
                        "sample_blocking_gap_count": 2,
                        "action_counts": {
                            "inspect_quarantined_official_result_runner_set": 2
                        },
                        "evidence_missing_reason_counts": {
                            "official_result_quarantined_unsafe_match": 2
                        },
                        "top_gap_races": [
                            {
                                "race_id": "Race 4 - TAREE - 2026-06-13",
                                "race_date": "2026-06-13",
                                "venue": "TAREE",
                                "recommended_action": (
                                    "inspect_quarantined_official_result_runner_set"
                                ),
                                "evidence_missing_reason": (
                                    "official_result_quarantined_unsafe_match"
                                ),
                                "official_result_quarantine_errors": [
                                    "result_boxes_not_in_participants:9"
                                ],
                                "debug_blob": "not propagated",
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    promotion_distance_report_path = tmp_path / "promotion_distance_report.json"
    promotion_distance_report_path.write_text(
        json.dumps(
            {
                "final_status": "PROMOTION_DISTANCE_BLOCKED",
                "promotion_ready": False,
                "blockers": ["best_non_market_top1_margin_below_target"],
                "rolling_sample": {
                    "sample_race_count": 124,
                    "source_exclusion_reason_counts": {
                        "official_result_missing": 32
                    },
                    "source_odds_exclusion_reason_counts": {
                        "strict_prejump_odds_missing": 4
                    },
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
                "official_result_coverage": {
                    "source": "rolling_model_comparison",
                    "requested_race_count": 7,
                    "requested_race_count_source": "deduped_requested_or_inferred_race_ids",
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
                },
                "market_benchmark": {
                    "best_candidate_key": "market_only_implied",
                    "best_non_market_candidate_key": "stage2_uncalibrated_market_blend_50",
                    "best_non_market_top1_margin_gap": 0.02,
                },
            }
        ),
        encoding="utf-8",
    )
    reserve_substitution_preflight_path = (
        tmp_path / "official_result_reserve_substitution_preflight.json"
    )
    reserve_substitution_preflight_path.write_text(
        json.dumps(
            {
                "final_status": (
                    "RESERVE_SUBSTITUTION_PREFLIGHT_READY_FOR_POLICY_REVIEW"
                ),
                "candidate_count": 1,
                "blocked_candidate_count": 0,
                "ready_for_policy_review_count": 1,
                "blocker_counts": {},
                "candidates": [
                    {
                        "race_id": "Race 4 - TAREE - 2026-06-13",
                        "preflight_status": "READY_FOR_MANUAL_POLICY_REVIEW",
                        "policy_review_status": "READY_FOR_MANUAL_POLICY_REVIEW",
                        "acceptance_status": "not_accepted_report_only",
                        "acceptance_effect": "none_report_only",
                        "candidate_reserve_boxes": [9],
                        "scratched_participant_boxes": [2],
                        "readiness_blockers": [],
                        "dataset_join_blockers": [
                            "official_result_remains_quarantined",
                            "manual_policy_review_required_before_join",
                        ],
                        "source_url": "https://www.thedogs.com.au/example",
                    }
                ],
                "no_write_guarantees": {
                    "db_write": False,
                    "label_write": False,
                    "official_result_acceptance": False,
                    "quarantine_bypass": False,
                    "production_promotion": False,
                },
            }
        ),
        encoding="utf-8",
    )
    reserve_substitution_manual_review_path = (
        tmp_path / "reserve_substitution_manual_review_packet.json"
    )
    reserve_substitution_manual_review_path.write_text(
        json.dumps(
            {
                "final_status": "RESERVE_SUBSTITUTION_MANUAL_REVIEW_PACKET_READY",
                "candidate_count": 1,
                "ready_candidate_count": 1,
                "blocked_candidate_count": 0,
                "ready_race_ids": ["Race 4 - TAREE - 2026-06-13"],
                "blocked_race_ids": [],
                "automatic_acceptance_allowed": False,
                "dataset_join_allowed": False,
                "official_result_acceptance_allowed": False,
                "db_write": False,
                "candidates": [
                    {
                        "race_id": "Race 4 - TAREE - 2026-06-13",
                        "packet_blockers": [],
                        "mapping_hypothesis": {
                            "mapping_status": "report_only_policy_hypothesis",
                            "mapping_acceptance_status": "not_accepted",
                            "mapping_blockers": [],
                            "pairs": [
                                {
                                    "scratched_participant_box": 2,
                                    "reserve_box": 9,
                                    "reserve_dog_name": "Reserve Runner",
                                    "mapping_acceptance_status": "not_accepted",
                                }
                            ],
                        },
                    }
                ],
                "no_write_guarantees": {
                    "db_write": False,
                    "label_write": False,
                    "official_result_acceptance": False,
                    "quarantine_bypass": False,
                    "production_promotion": False,
                },
            }
        ),
        encoding="utf-8",
    )
    reserve_substitution_policy_impact_path = (
        tmp_path / "reserve_substitution_policy_impact_preview.json"
    )
    reserve_substitution_policy_impact_path.write_text(
        json.dumps(
            {
                "final_status": "RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY",
                "approval_required": True,
                "automatic_acceptance_allowed": False,
                "dataset_join_allowed": False,
                "official_result_acceptance_allowed": False,
                "db_write": False,
                "candidate_count": 1,
                "ready_candidate_count": 1,
                "ready_race_ids": ["Race 4 - TAREE - 2026-06-13"],
                "mapping_pair_count": 1,
                "potential_official_result_runner_rows_blocked_by_policy": 8,
                "matched_backlog_top_gap_race_count": 1,
                "matched_backlog_top_gap_race_ids": [
                    "Race 4 - TAREE - 2026-06-13"
                ],
                "backlog_sample_blocking_gap_count": 2,
                "backlog_gap_action_counts": {
                    "inspect_quarantined_official_result_runner_set": 2,
                },
                "backlog_evidence_missing_reason_counts": {
                    "official_result_quarantined_unsafe_match": 2,
                },
                "current_effect": (
                    "none_report_only_all_results_remain_quarantined"
                ),
                "no_write_guarantees": {
                    "db_write": False,
                    "label_write": False,
                    "official_result_acceptance": False,
                    "quarantine_bypass": False,
                    "production_promotion": False,
                },
            }
        ),
        encoding="utf-8",
    )
    timing_aligned_rerun_plan_path = (
        tmp_path / "timing_aligned_prediction_rerun_plan.json"
    )
    timing_aligned_rerun_plan_path.write_text(
        json.dumps(
            {
                "status": "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_PLAN_BLOCKED",
                "hard_stops": [
                    "timing_aligned_rerun_window_already_closed_after_jump"
                ],
            }
        ),
        encoding="utf-8",
    )
    timing_aligned_rerun_execution_status_path = (
        tmp_path / "timing_aligned_prediction_rerun_execution_status.json"
    )
    timing_aligned_rerun_execution_status_path.write_text(
        json.dumps(
            {
                "status": (
                    "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_SKIPPED_PLAN_NOT_READY"
                ),
                "hard_stops": [
                    "timing_aligned_rerun_window_already_closed_after_jump"
                ],
            }
        ),
        encoding="utf-8",
    )

    result = packet.run_refinement_packet(
        stage2_forward_metrics_path=stage2_forward_metrics_path,
        unified_evidence_report_path=unified_evidence_report_path,
        backlog_unified_evidence_status_path=backlog_unified_evidence_status_path,
        promotion_distance_report_path=promotion_distance_report_path,
        reserve_substitution_preflight_path=reserve_substitution_preflight_path,
        timing_aligned_rerun_plan_path=timing_aligned_rerun_plan_path,
        timing_aligned_rerun_execution_status_path=(
            timing_aligned_rerun_execution_status_path
        ),
        output_dir=output_dir,
        thresholds=packet.AccuracyGateThresholds(min_safe_joined_races=100),
    )

    assert result["final_status"] == packet.FINAL_BLOCKED
    assert result["promotion_pr_gate_status"] == "BLOCKED"
    assert (output_dir / "high_accuracy_refinement_packet.json").exists()
    assert (output_dir / "promotion_pr_gate.json").exists()
    assert (output_dir / "stage2_forward_joined_metrics.json").exists()
    assert (output_dir / "SUMMARY.md").exists()
    written = json.loads((output_dir / "high_accuracy_refinement_packet.json").read_text())
    assert written["no_write_guarantees"]["production_promotion"] is False
    assert written["protected_paths_unchanged"] is True
    assert written["source_artifacts"]["unified_evidence_report"].endswith(
        "unified_evidence_dataset_report.json"
    )
    assert written["source_artifacts"]["backlog_unified_evidence_status"].endswith(
        "backlog_unified_evidence_datasets_status.json"
    )
    assert written["source_artifacts"]["promotion_distance_report"].endswith(
        "promotion_distance_report.json"
    )
    assert written["source_artifacts"]["reserve_substitution_preflight"].endswith(
        "official_result_reserve_substitution_preflight.json"
    )
    assert written["source_artifacts"][
        "reserve_substitution_manual_review"
    ].endswith("reserve_substitution_manual_review_packet.json")
    assert written["source_artifacts"][
        "reserve_substitution_policy_impact_preview"
    ].endswith("reserve_substitution_policy_impact_preview.json")
    assert written["source_artifacts"]["timing_aligned_rerun_plan"].endswith(
        "timing_aligned_prediction_rerun_plan.json"
    )
    assert written["source_artifacts"][
        "timing_aligned_rerun_execution_status"
    ].endswith("timing_aligned_prediction_rerun_execution_status.json")
    assert written["unified_evidence_summary"]["unified_evidence_eligible_rows"] == 11
    assert written["unified_evidence_summary"]["artifact_odds_rows_accepted"] == 6
    assert written["unified_evidence_summary"][
        "artifact_odds_rejection_reason_counts"
    ] == {"odds_match_status_not_valid_pre_jump_dog_odds": 17}
    assert written["unified_evidence_summary"]["exclusion_reason_counts"] == {
        "official_result_missing": 12
    }
    assert written["unified_evidence_summary"]["odds_exclusion_reason_counts"] == {
        "strict_prejump_odds_missing": 3
    }
    assert written["unified_evidence_summary"][
        "official_result_evidence_db_missing_race_ids"
    ] == ["Race 3 - TAREE - 2026-06-13"]
    assert written["unified_evidence_summary"]["official_result_runner_paths"] == [
        "artifacts/full_evidence_orchestration_20260525/autonomous_official_result_capture_test/official_result_runners.jsonl"
    ]
    assert written["unified_evidence_summary"]["official_result_coverage"] == {
        "source": "unified_evidence_report",
        "requested_race_count": 3,
        "races_with_rows_count": 2,
        "missing_race_count": 1,
        "missing_race_ids": ["Race 3 - TAREE - 2026-06-13"],
        "races_with_rows": [
            "Race 5 - TAREE - 2026-06-13",
            "Race 6 - TAREE - 2026-06-13",
        ],
        "runner_path_count": 1,
        "runner_paths_source_field": "official_result_runner_paths",
        "missing_exclusion_count": 12,
    }
    assert written["unified_evidence_summary"][
        "rejected_live_odds_candidate_count"
    ] == 5
    assert written["unified_evidence_summary"][
        "rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 2,
        "odds_source_url_missing": 3,
    }
    assert written["backlog_unified_evidence_summary"][
        "unified_evidence_eligible_rows"
    ] == 120
    assert written["backlog_unified_evidence_summary"][
        "artifact_odds_rows_accepted"
    ] == 79
    assert written["backlog_unified_evidence_summary"]["exclusion_reason_counts"] == {
        "official_result_missing": 80
    }
    assert written["backlog_unified_evidence_summary"][
        "odds_exclusion_reason_counts"
    ] == {"strict_prejump_odds_missing": 28}
    assert written["backlog_unified_evidence_summary"][
        "official_result_evidence_db_missing_race_ids"
    ] == ["Race 8 - TAREE - 2026-06-13"]
    assert written["backlog_unified_evidence_summary"][
        "rejected_live_odds_candidate_count"
    ] == 9
    assert written["backlog_unified_evidence_summary"][
        "rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 4,
        "odds_source_url_missing": 5,
    }
    assert written["backlog_unified_evidence_summary"][
        "sample_blocking_gap_count"
    ] == 2
    assert written["backlog_unified_evidence_summary"][
        "gap_action_counts"
    ] == {"inspect_quarantined_official_result_runner_set": 2}
    assert written["backlog_unified_evidence_summary"][
        "gap_evidence_missing_reason_counts"
    ] == {"official_result_quarantined_unsafe_match": 2}
    assert written["backlog_unified_evidence_summary"][
        "top_gap_race_ids"
    ] == ["Race 4 - TAREE - 2026-06-13"]
    assert written["backlog_unified_evidence_summary"][
        "top_gap_races"
    ] == [
        {
            "race_id": "Race 4 - TAREE - 2026-06-13",
            "race_date": "2026-06-13",
            "venue": "TAREE",
            "recommended_action": "inspect_quarantined_official_result_runner_set",
            "evidence_missing_reason": "official_result_quarantined_unsafe_match",
            "official_result_quarantine_errors": [
                "result_boxes_not_in_participants:9"
            ],
        }
    ]
    summary_text = (output_dir / "SUMMARY.md").read_text(encoding="utf-8")
    assert "- Unified official-result runner path count: `1`" in summary_text
    assert (
        "- Unified official-result runner path source field: "
        "`official_result_runner_paths`"
    ) in summary_text
    assert (
        "- Unified official-result coverage requested races: `3`"
        in summary_text
    )
    assert (
        "- Unified official-result coverage races with rows: `2`"
        in summary_text
    )
    assert (
        "- Unified official-result coverage missing races: `1`"
        in summary_text
    )
    assert (
        "- Unified official-result missing exclusion count: `12`"
        in summary_text
    )
    assert "- Unified official-result runner paths:" not in summary_text
    assert "- Unified rejected live odds candidates: `5`" in summary_text
    assert "- Unified rows with rejected live odds candidates: `4`" in summary_text
    assert "- Backlog rejected live odds candidates: `9`" in summary_text
    assert "- Backlog rows with rejected live odds candidates: `6`" in summary_text
    assert "- Backlog sample-blocking gap races: `2`" in summary_text
    assert (
        "- Backlog gap actions: `{'inspect_quarantined_official_result_runner_set': 2}`"
        in summary_text
    )
    assert (
        "- Backlog top gap race IDs: `['Race 4 - TAREE - 2026-06-13']`"
        in summary_text
    )
    assert (
        "- Timing-aligned rerun plan: "
        "`timing_aligned_prediction_rerun_plan.json`"
    ) in summary_text
    assert (
        "- Timing-aligned rerun execution status: "
        "`timing_aligned_prediction_rerun_execution_status.json`"
    ) in summary_text
    assert written["promotion_distance_summary"]["status"] == (
        "PROMOTION_DISTANCE_BLOCKED"
    )
    assert written["promotion_distance_summary"]["best_candidate_key"] == (
        "market_only_implied"
    )
    assert written["promotion_distance_summary"]["source_exclusion_reason_counts"] == {
        "official_result_missing": 32
    }
    assert written["promotion_distance_summary"][
        "source_official_result_evidence_db_missing_race_ids"
    ] == ["Race 3 - TAREE - 2026-06-13"]
    assert written["promotion_distance_summary"][
        "source_official_result_runner_paths"
    ] == [
        "artifacts/full_evidence_orchestration_20260525/autonomous_official_result_capture_test/official_result_runners.jsonl"
    ]
    assert written["promotion_distance_summary"]["official_result_coverage"] == {
        "source": "rolling_model_comparison",
        "requested_race_count": 7,
        "requested_race_count_source": "deduped_requested_or_inferred_race_ids",
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
    assert (
        "- Promotion distance source exclusion reasons: `{'official_result_missing': 32}`"
        in summary_text
    )
    assert "- Promotion distance official-result missing race IDs: `['Race 3 - TAREE - 2026-06-13']`" in summary_text
    assert "- Promotion distance official-result runner path count: `1`" in summary_text
    assert (
        "- Promotion distance official-result runner path source field: "
        "`rolling_sample.source_official_result_runner_paths`"
    ) in summary_text
    assert (
        "- Promotion distance official-result coverage requested races: `7`"
        in summary_text
    )
    assert (
        "- Promotion distance official-result requested race count source: "
        "`deduped_requested_or_inferred_race_ids`"
    ) in summary_text
    assert (
        "- Promotion distance official-result legacy requested race count without IDs: `4`"
        in summary_text
    )
    assert (
        "- Promotion distance official-result coverage races with rows: `2`"
        in summary_text
    )
    assert (
        "- Promotion distance official-result coverage missing races: `1`"
        in summary_text
    )
    assert (
        "- Promotion distance official-result missing exclusion count: `32`"
        in summary_text
    )
    assert written["reserve_substitution_preflight_summary"]["ready_race_ids"] == [
        "Race 4 - TAREE - 2026-06-13"
    ]
    assert written["reserve_substitution_preflight_summary"][
        "dataset_join_blocker_counts"
    ] == {
        "manual_policy_review_required_before_join": 1,
        "official_result_remains_quarantined": 1,
    }
    assert written["reserve_substitution_manual_review_summary"]["status"] == (
        "RESERVE_SUBSTITUTION_MANUAL_REVIEW_PACKET_READY"
    )
    assert written["reserve_substitution_manual_review_summary"][
        "mapping_pair_count"
    ] == 1
    assert written["reserve_substitution_manual_review_summary"][
        "dataset_join_allowed"
    ] is False
    assert written["reserve_substitution_policy_impact_summary"]["status"] == (
        "RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY"
    )
    assert written["reserve_substitution_policy_impact_summary"][
        "potential_official_result_runner_rows_blocked_by_policy"
    ] == 8
    assert written["reserve_substitution_policy_impact_summary"][
        "matched_backlog_top_gap_race_count"
    ] == 1
    assert written["reserve_substitution_policy_impact_summary"][
        "dataset_join_allowed"
    ] is False
    assert (
        "- Reserve substitution preflight: "
        "`RESERVE_SUBSTITUTION_PREFLIGHT_READY_FOR_POLICY_REVIEW`"
    ) in summary_text
    assert (
        "- Reserve substitution manual review: "
        "`RESERVE_SUBSTITUTION_MANUAL_REVIEW_PACKET_READY`"
    ) in summary_text
    assert (
        "- Reserve substitution policy impact: "
        "`RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY`"
    ) in summary_text
    assert (
        "- Reserve substitution policy impact potential runner rows blocked: `8`"
        in summary_text
    )


def test_run_packet_recovers_rejoin_artifact_rejection_reasons_from_report_dirs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(packet, "ROOT", tmp_path)
    monkeypatch.setattr(packet, "DEFAULT_PROTECTED_PATHS", ())
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525"
        / "high_accuracy_refinement_packet_20260610T010000+0000"
    )
    child_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525"
        / "unified_evidence_dataset_rejoin_001"
    )
    child_dir.mkdir(parents=True)
    (child_dir / "unified_evidence_dataset_report.json").write_text(
        json.dumps(
            {
                "artifact_odds_audits": [
                    {
                        "rejection_reason_counts": {
                            "odds_match_status_not_valid_pre_jump_dog_odds": 76
                        }
                    }
                ],
                "rejected_live_odds_candidate_count": 7,
                "rows_with_rejected_live_odds_candidates": 5,
                "rejected_live_odds_candidate_reason_counts": {
                    "odds_decimal_invalid": 3,
                    "odds_source_url_missing": 4,
                },
            }
        ),
        encoding="utf-8",
    )
    backlog_unified_evidence_status_path = tmp_path / "rejoin_unified_evidence_status.json"
    backlog_unified_evidence_status_path.write_text(
        json.dumps(
            {
                "status": "REJOIN_UNIFIED_EVIDENCE_DATASETS_BUILT",
                "dataset_count": 1,
                "failed_dataset_count": 0,
                "row_count": 108,
                "race_count": 14,
                "rows_with_strict_prejump_odds": 108,
                "rows_with_artifact_shadow_odds": 32,
                "artifact_odds_rows_seen": 108,
                "artifact_odds_rows_accepted": 32,
                "artifact_odds_rows_rejected": 76,
                "unified_evidence_eligible_rows": 98,
                "reports": [
                    {
                        "output_dir": (
                            "artifacts/full_evidence_orchestration_20260525/"
                            "unified_evidence_dataset_rejoin_001"
                        )
                    }
                ],
                "no_write_guarantees": {"db_write": False, "label_write": False},
            }
        ),
        encoding="utf-8",
    )

    packet.run_refinement_packet(
        backlog_unified_evidence_status_path=backlog_unified_evidence_status_path,
        output_dir=output_dir,
    )

    written = json.loads((output_dir / "high_accuracy_refinement_packet.json").read_text())
    summary = written["backlog_unified_evidence_summary"]
    assert summary["status"] == "BACKLOG_UNIFIED_EVIDENCE_COLLECTING"
    assert summary["artifact_odds_rows_accepted"] == 32
    assert summary["artifact_odds_rejection_reason_counts"] == {
        "odds_match_status_not_valid_pre_jump_dog_odds": 76
    }
    assert summary["rejected_live_odds_candidate_count"] == 7
    assert summary["rows_with_rejected_live_odds_candidates"] == 5
    assert summary["rejected_live_odds_candidate_reason_counts"] == {
        "odds_decimal_invalid": 3,
        "odds_source_url_missing": 4,
    }
