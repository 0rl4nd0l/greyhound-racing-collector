import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import shadow_feature_activation_gate as gate


def _parity_feature(
    feature: str,
    *,
    train_present_rows: int = 80,
    train_rows: int = 100,
    holdout_present_rows: int = 30,
    holdout_rows: int = 40,
) -> dict:
    return {
        "feature": feature,
        "train_rows": train_rows,
        "train_present_rows": train_present_rows,
        "train_present_pct": train_present_rows / train_rows,
        "train_unique_present_values": min(train_present_rows, 10),
        "holdout_rows": holdout_rows,
        "holdout_present_rows": holdout_present_rows,
        "holdout_present_pct": holdout_present_rows / holdout_rows,
        "holdout_unique_present_values": min(holdout_present_rows, 10),
        "all_missing_in_train": train_present_rows == 0,
        "all_missing_in_holdout": holdout_present_rows == 0,
        "present_in_holdout": holdout_present_rows > 0,
        "parity_status": "OK",
    }


def _matrix_audit(status: str = "PASS") -> dict:
    return {
        "status": status,
        "matrix_gate": {"status": status},
        "temporal_evaluation": {
            "status": status,
            "race_id_overlap": [],
            "all_dogs_in_race_kept_together": True,
        },
        "label_audit": {"status": status},
        "schema_contract": {
            "status": status,
            "post_outcome_columns_present_as_features": [],
            "tgr_columns": [],
        },
    }


def _provenance_audit(**overrides) -> dict:
    payload = {
        "protected_paths_unchanged": True,
        "rejected_source_rows": 0,
        "target_distance_sources": {"canonical_pre_race_page": 125},
        "target_grade_sources": {"canonical_pre_race_page": 125},
        "by_feature": {
            "target_distance_safe": {"present_rows": 125},
            "target_grade_safe": {"present_rows": 125},
        },
        "target_metadata_readiness": {
            "status": "TARGET_METADATA_READY_FOR_CURRENT_OR_FUTURE_PREJUMP_INPUTS",
            "target_metadata_capture_status": "READY",
            "current_or_future_input_count": 16,
            "ready_current_or_future_input_count": 16,
            "blocker_counts": {},
            "historical_repair_policy": "NO_REPAIR_WITHOUT_PROVENANCE_SAFE_PRE_RACE_SOURCE",
        },
        "same_distance_same_grade_history_provenance": {
            "status": "PASS",
            "by_feature": {
                "same_distance_same_grade_best_time": {
                    "source": "prior_dog_history",
                    "history_cutoff": "strictly_before_target_race",
                    "prior_history_rows_used": 80,
                    "target_race_rows_used": 0,
                    "post_outcome_rows_used": 0,
                    "post_outcome_fields_used": [],
                },
                "same_distance_same_grade_avg_time": {
                    "source": "prior_dog_history",
                    "history_cutoff": "strictly_before_target_race",
                    "prior_history_rows_used": 80,
                    "target_race_rows_used": 0,
                    "post_outcome_rows_used": 0,
                    "post_outcome_fields_used": [],
                },
            },
        },
    }
    payload.update(overrides)
    return payload


def _metrics(**overrides) -> dict:
    payload = {
        "safe_joined_race_count": 120,
        "safe_joined_runner_count": 960,
        "top1": 0.3,
        "top3": 0.65,
        "mean_winner_rank": 3.5,
        "brier": 0.11,
        "logloss": 2.0,
        "probability_sum_max_error_joined_races": 0.0,
        "calibration": {
            "status": "computed",
            "slope": 0.8,
            "intercept": -0.4,
            "sample_size": 240,
        },
        "box_1_share": 0.25,
        "safe_joined_top_pick_box_distribution": {
            "1": 8,
            "2": 6,
            "3": 4,
            "4": 4,
            "5": 3,
            "6": 2,
            "7": 2,
            "8": 3,
        },
    }
    payload.update(overrides)
    return payload


def test_gate_keeps_current_all_missing_train_feature_quarantined():
    feature = "same_distance_same_grade_best_time"
    parity = {
        "by_feature": {
            feature: _parity_feature(
                feature,
                train_present_rows=0,
                train_rows=751,
                holdout_present_rows=10,
                holdout_rows=192,
            )
        }
    }

    report = gate.evaluate_feature_activation(
        feature=feature,
        parity_report=parity,
        inactive_policy_report={"inactive_features_due_to_train_all_missing": [feature]},
        matrix_audit=_matrix_audit(),
        provenance_audit=_provenance_audit(),
        baseline_metrics=_metrics(),
        candidate_metrics=_metrics(top1=0.31, top3=0.66, mean_winner_rank=3.4, brier=0.10, logloss=1.9),
        thresholds=gate.ActivationThresholds(),
    )

    assert report["decision"] == "KEEP_QUARANTINED"
    assert "all_missing_in_train" in report["fail_reasons"]
    assert "currently_inactive_due_to_train_all_missing_policy" in report["fail_reasons"]
    assert "train_present_rows_below_min" in report["fail_reasons"]


def test_gate_allows_activation_report_only_when_all_evidence_passes():
    feature = "same_distance_same_grade_avg_time"
    parity = {"by_feature": {feature: _parity_feature(feature)}}

    report = gate.evaluate_feature_activation(
        feature=feature,
        parity_report=parity,
        inactive_policy_report={"inactive_features_due_to_train_all_missing": []},
        matrix_audit=_matrix_audit(),
        provenance_audit=_provenance_audit(),
        baseline_metrics=_metrics(),
        candidate_metrics=_metrics(top1=0.35, top3=0.70, mean_winner_rank=3.2, brier=0.10, logloss=1.9),
        thresholds=gate.ActivationThresholds(),
    )

    assert report["decision"] == "ACTIVATE_ALLOWED_REPORT_ONLY"
    assert report["fail_reasons"] == []


def test_gate_accepts_forward_challenger_metric_field_names():
    feature = "same_distance_same_grade_avg_time"
    parity = {"by_feature": {feature: _parity_feature(feature)}}
    baseline_metrics = _metrics(
        calibration={},
        box_1_share=None,
        safe_joined_top_pick_box_distribution={},
        calibration_slope_intercept={
            "status": "computed",
            "slope": 0.8,
            "intercept": -0.4,
            "sample_size": 240,
        },
        box1_top_pick_share=0.25,
        metric_cohort_id="eval-cohort-a",
        safe_joined_race_ids_hash="hash-a",
    )
    candidate_metrics = _metrics(
        top1=0.35,
        top3=0.70,
        mean_winner_rank=3.2,
        brier=0.10,
        logloss=1.9,
        calibration={},
        box_1_share=None,
        safe_joined_top_pick_box_distribution={},
        calibration_slope_intercept={
            "status": "computed",
            "slope": 0.85,
            "intercept": -0.3,
            "sample_size": 240,
        },
        box1_top_pick_share=0.24,
        metric_cohort_id="eval-cohort-a",
        safe_joined_race_ids_hash="hash-a",
    )

    report = gate.evaluate_feature_activation(
        feature=feature,
        parity_report=parity,
        inactive_policy_report={"inactive_features_due_to_train_all_missing": []},
        matrix_audit=_matrix_audit(),
        provenance_audit=_provenance_audit(),
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        thresholds=gate.ActivationThresholds(),
    )

    assert report["decision"] == "ACTIVATE_ALLOWED_REPORT_ONLY"
    assert report["fail_reasons"] == []


def test_gate_blocks_candidate_metric_source_status_and_blockers():
    feature = "same_distance_same_grade_avg_time"
    parity = {"by_feature": {feature: _parity_feature(feature)}}
    candidate_metrics = _metrics(
        top1=0.35,
        top3=0.70,
        mean_winner_rank=3.2,
        brier=0.10,
        logloss=1.9,
        source_final_status="CHALLENGER_CALIBRATION_BLOCKED_KEEP_BASELINE",
        source_activation_blockers=[
            "safe_joined_race_count_below_min_total",
            "eval_race_count_below_min",
        ],
    )

    report = gate.evaluate_feature_activation(
        feature=feature,
        parity_report=parity,
        inactive_policy_report={"inactive_features_due_to_train_all_missing": []},
        matrix_audit=_matrix_audit(),
        provenance_audit=_provenance_audit(),
        baseline_metrics=_metrics(),
        candidate_metrics=candidate_metrics,
        thresholds=gate.ActivationThresholds(),
    )

    assert report["decision"] == "KEEP_QUARANTINED"
    assert (
        "candidate_metric_source_not_ready:CHALLENGER_CALIBRATION_BLOCKED_KEEP_BASELINE"
        in report["fail_reasons"]
    )
    assert (
        "candidate_metric_source_blocked:safe_joined_race_count_below_min_total"
        in report["fail_reasons"]
    )
    assert "candidate_metric_source_blocked:eval_race_count_below_min" in report["fail_reasons"]
    assert gate.fail_reason_category(
        "candidate_metric_source_not_ready:CHALLENGER_CALIBRATION_BLOCKED_KEEP_BASELINE"
    ) == "shadow_metric_sample"
    assert gate.fail_reason_category(
        "candidate_metric_source_blocked:safe_joined_race_count_below_min_total"
    ) == "shadow_metric_sample"


def test_default_gate_blocks_activation_until_target_sized_forward_sample():
    feature = "same_distance_same_grade_avg_time"
    parity = {"by_feature": {feature: _parity_feature(feature)}}

    report = gate.evaluate_feature_activation(
        feature=feature,
        parity_report=parity,
        inactive_policy_report={"inactive_features_due_to_train_all_missing": []},
        matrix_audit=_matrix_audit(),
        provenance_audit=_provenance_audit(),
        baseline_metrics=_metrics(safe_joined_race_count=84),
        candidate_metrics=_metrics(
            safe_joined_race_count=84,
            top1=0.35,
            top3=0.70,
            mean_winner_rank=3.2,
            brier=0.10,
            logloss=1.9,
        ),
        thresholds=gate.ActivationThresholds(),
    )

    assert gate.ActivationThresholds().min_shadow_joined_races == 100
    assert report["decision"] == "KEEP_QUARANTINED"
    assert "baseline_joined_race_sample_below_min" in report["fail_reasons"]
    assert "candidate_joined_race_sample_below_min" in report["fail_reasons"]


def test_gate_blocks_metric_regression_and_small_sample():
    feature = "same_distance_same_grade_avg_time"
    parity = {"by_feature": {feature: _parity_feature(feature)}}

    report = gate.evaluate_feature_activation(
        feature=feature,
        parity_report=parity,
        inactive_policy_report={"inactive_features_due_to_train_all_missing": []},
        matrix_audit=_matrix_audit(),
        provenance_audit=_provenance_audit(),
        baseline_metrics=_metrics(),
        candidate_metrics=_metrics(
            safe_joined_race_count=3,
            top1=0.1,
            top3=0.4,
            mean_winner_rank=5.0,
        ),
        thresholds=gate.ActivationThresholds(),
    )

    assert report["decision"] == "KEEP_QUARANTINED"
    assert "candidate_joined_race_sample_below_min" in report["fail_reasons"]
    assert "metric_regressed:top1" in report["fail_reasons"]
    assert "metric_regressed:mean_winner_rank" in report["fail_reasons"]


def test_gate_blocks_when_prejump_target_metadata_readiness_is_blocked():
    feature = "same_distance_same_grade_avg_time"
    parity = {"by_feature": {feature: _parity_feature(feature)}}

    report = gate.evaluate_feature_activation(
        feature=feature,
        parity_report=parity,
        inactive_policy_report={"inactive_features_due_to_train_all_missing": []},
        matrix_audit=_matrix_audit(),
        provenance_audit=_provenance_audit(
            target_metadata_readiness={
                "status": "TARGET_METADATA_BLOCKED_BY_INCOMPLETE_OR_UNSAFE_SIDECARS",
                "target_metadata_capture_status": "BLOCKED",
                "current_or_future_input_count": 4,
                "ready_current_or_future_input_count": 2,
                "blocker_counts": {
                    "target_grade_missing": 1,
                    "canonical_runner_alignment_not_aligned": 1,
                },
                "historical_repair_policy": "NO_REPAIR_WITHOUT_PROVENANCE_SAFE_PRE_RACE_SOURCE",
            }
        ),
        baseline_metrics=_metrics(),
        candidate_metrics=_metrics(
            top1=0.35,
            top3=0.70,
            mean_winner_rank=3.2,
            brier=0.10,
            logloss=1.9,
        ),
        thresholds=gate.ActivationThresholds(),
    )

    assert report["decision"] == "KEEP_QUARANTINED"
    assert (
        "target_metadata_readiness_not_ready:"
        "TARGET_METADATA_BLOCKED_BY_INCOMPLETE_OR_UNSAFE_SIDECARS"
        in report["fail_reasons"]
    )
    assert "target_metadata_ready_input_count_mismatch" in report["fail_reasons"]
    assert "target_metadata_blocked:target_grade_missing" in report["fail_reasons"]
    assert (
        "target_metadata_blocked:canonical_runner_alignment_not_aligned"
        in report["fail_reasons"]
    )
    assert gate.fail_reason_category("target_metadata_blocked:target_grade_missing") == (
        "target_metadata_provenance"
    )


def test_gate_accepts_daily_target_metadata_readiness_verified_count_shape():
    feature = "same_distance_same_grade_avg_time"
    parity = {"by_feature": {feature: _parity_feature(feature)}}

    report = gate.evaluate_feature_activation(
        feature=feature,
        parity_report=parity,
        inactive_policy_report={"inactive_features_due_to_train_all_missing": []},
        matrix_audit=_matrix_audit(),
        provenance_audit=_provenance_audit(
            target_metadata_readiness={
                "schema_version": "daily_shadow_target_metadata_readiness_v1",
                "status": "TARGET_METADATA_READY_FOR_CURRENT_OR_FUTURE_PREJUMP_INPUTS",
                "target_metadata_capture_status": "READY",
                "current_or_future_input_count": 4,
                "verified_eligible_count": 4,
                "blocker_counts": {},
                "historical_repair_policy": "NO_REPAIR_WITHOUT_PROVENANCE_SAFE_PRE_RACE_SOURCE",
            }
        ),
        baseline_metrics=_metrics(),
        candidate_metrics=_metrics(
            top1=0.35,
            top3=0.70,
            mean_winner_rank=3.2,
            brier=0.10,
            logloss=1.9,
        ),
        thresholds=gate.ActivationThresholds(),
    )

    assert report["decision"] == "ACTIVATE_ALLOWED_REPORT_ONLY"
    assert "target_metadata_ready_input_count_mismatch" not in report["fail_reasons"]
    assert not [
        reason
        for reason in report["fail_reasons"]
        if reason.startswith("target_metadata_readiness_not_ready:")
    ]


def test_gate_blocks_daily_target_metadata_readiness_verified_count_mismatch():
    feature = "same_distance_same_grade_avg_time"
    parity = {"by_feature": {feature: _parity_feature(feature)}}

    report = gate.evaluate_feature_activation(
        feature=feature,
        parity_report=parity,
        inactive_policy_report={"inactive_features_due_to_train_all_missing": []},
        matrix_audit=_matrix_audit(),
        provenance_audit=_provenance_audit(
            target_metadata_readiness={
                "schema_version": "daily_shadow_target_metadata_readiness_v1",
                "status": "TARGET_METADATA_BLOCKED_BY_INCOMPLETE_OR_UNSAFE_SIDECARS",
                "target_metadata_capture_status": "BLOCKED",
                "current_or_future_input_count": 4,
                "verified_eligible_count": 3,
                "blocker_counts": {},
                "historical_repair_policy": "NO_REPAIR_WITHOUT_PROVENANCE_SAFE_PRE_RACE_SOURCE",
            }
        ),
        baseline_metrics=_metrics(),
        candidate_metrics=_metrics(
            top1=0.35,
            top3=0.70,
            mean_winner_rank=3.2,
            brier=0.10,
            logloss=1.9,
        ),
        thresholds=gate.ActivationThresholds(),
    )

    assert report["decision"] == "KEEP_QUARANTINED"
    assert "target_metadata_ready_input_count_mismatch" in report["fail_reasons"]
    assert (
        "target_metadata_readiness_not_ready:"
        "TARGET_METADATA_BLOCKED_BY_INCOMPLETE_OR_UNSAFE_SIDECARS"
        in report["fail_reasons"]
    )


def test_gate_blocks_when_target_metadata_readiness_is_missing():
    feature = "same_distance_same_grade_avg_time"
    parity = {"by_feature": {feature: _parity_feature(feature)}}
    provenance = _provenance_audit()
    provenance.pop("target_metadata_readiness")

    report = gate.evaluate_feature_activation(
        feature=feature,
        parity_report=parity,
        inactive_policy_report={"inactive_features_due_to_train_all_missing": []},
        matrix_audit=_matrix_audit(),
        provenance_audit=provenance,
        baseline_metrics=_metrics(),
        candidate_metrics=_metrics(
            top1=0.35,
            top3=0.70,
            mean_winner_rank=3.2,
            brier=0.10,
            logloss=1.9,
        ),
        thresholds=gate.ActivationThresholds(),
    )

    assert report["decision"] == "KEEP_QUARANTINED"
    assert "target_metadata_readiness_missing" in report["fail_reasons"]
    assert gate.fail_reason_category("target_metadata_readiness_missing") == (
        "target_metadata_provenance"
    )


def test_gate_blocks_non_comparable_metric_samples():
    feature = "same_distance_same_grade_avg_time"
    parity = {"by_feature": {feature: _parity_feature(feature)}}

    report = gate.evaluate_feature_activation(
        feature=feature,
        parity_report=parity,
        inactive_policy_report={"inactive_features_due_to_train_all_missing": []},
        matrix_audit=_matrix_audit(),
        provenance_audit=_provenance_audit(),
        baseline_metrics=_metrics(
            safe_joined_race_count=120,
            safe_joined_runner_count=960,
            safe_joined_race_ids_hash="cohort-a",
        ),
        candidate_metrics=_metrics(
            safe_joined_race_count=121,
            safe_joined_runner_count=951,
            safe_joined_race_ids_hash="cohort-b",
            top1=0.35,
            top3=0.70,
            mean_winner_rank=3.2,
            brier=0.10,
            logloss=1.9,
        ),
        thresholds=gate.ActivationThresholds(),
    )

    assert report["decision"] == "KEEP_QUARANTINED"
    assert "metric_sample_mismatch:safe_joined_race_count" in report["fail_reasons"]
    assert "metric_sample_mismatch:safe_joined_runner_count" in report["fail_reasons"]
    assert "metric_cohort_mismatch:safe_joined_race_ids_hash" in report["fail_reasons"]


def test_gate_blocks_calibration_and_box_bias_regression():
    feature = "same_distance_same_grade_avg_time"
    parity = {"by_feature": {feature: _parity_feature(feature)}}

    report = gate.evaluate_feature_activation(
        feature=feature,
        parity_report=parity,
        inactive_policy_report={"inactive_features_due_to_train_all_missing": []},
        matrix_audit=_matrix_audit(),
        provenance_audit=_provenance_audit(),
        baseline_metrics=_metrics(
            calibration={
                "status": "computed",
                "slope": 0.8,
                "intercept": -0.2,
                "sample_size": 240,
            },
            box_1_share=0.25,
        ),
        candidate_metrics=_metrics(
            top1=0.35,
            top3=0.70,
            mean_winner_rank=3.2,
            brier=0.10,
            logloss=1.9,
            calibration={
                "status": "computed",
                "slope": 0.5,
                "intercept": -0.9,
                "sample_size": 240,
            },
            box_1_share=0.40,
        ),
        thresholds=gate.ActivationThresholds(),
    )

    assert report["decision"] == "KEEP_QUARANTINED"
    assert "metric_regressed:calibration_slope" in report["fail_reasons"]
    assert "metric_regressed:calibration_intercept" in report["fail_reasons"]
    assert "candidate_box1_top_pick_share_above_max" in report["fail_reasons"]
    assert "metric_regressed:box1_top_pick_share" in report["fail_reasons"]


def test_gate_blocks_missing_or_underpowered_calibration_and_box_metrics():
    feature = "same_distance_same_grade_avg_time"
    parity = {"by_feature": {feature: _parity_feature(feature)}}

    report = gate.evaluate_feature_activation(
        feature=feature,
        parity_report=parity,
        inactive_policy_report={"inactive_features_due_to_train_all_missing": []},
        matrix_audit=_matrix_audit(),
        provenance_audit=_provenance_audit(),
        baseline_metrics=_metrics(
            calibration={},
            safe_joined_runner_count=None,
            box_1_share=None,
            safe_joined_top_pick_box_distribution={},
        ),
        candidate_metrics=_metrics(
            top1=0.35,
            top3=0.70,
            mean_winner_rank=3.2,
            brier=0.10,
            logloss=1.9,
            calibration={"slope": 0.8, "intercept": -0.4, "sample_size": 20},
            box_1_share=None,
            safe_joined_top_pick_box_distribution={},
        ),
        thresholds=gate.ActivationThresholds(),
    )

    assert report["decision"] == "KEEP_QUARANTINED"
    assert "metric_missing:calibration_slope_baseline" in report["fail_reasons"]
    assert "metric_missing:calibration_intercept_baseline" in report["fail_reasons"]
    assert "metric_missing:calibration_sample_size_baseline" in report["fail_reasons"]
    assert "candidate_calibration_sample_below_min" in report["fail_reasons"]
    assert "metric_missing:box1_top_pick_share_baseline" in report["fail_reasons"]
    assert "metric_missing:box1_top_pick_share_candidate" in report["fail_reasons"]


def test_gate_blocks_underpowered_baseline_calibration_sample():
    feature = "same_distance_same_grade_avg_time"
    parity = {"by_feature": {feature: _parity_feature(feature)}}

    report = gate.evaluate_feature_activation(
        feature=feature,
        parity_report=parity,
        inactive_policy_report={"inactive_features_due_to_train_all_missing": []},
        matrix_audit=_matrix_audit(),
        provenance_audit=_provenance_audit(),
        baseline_metrics=_metrics(
            calibration={
                "status": "computed",
                "slope": 0.8,
                "intercept": -0.4,
                "sample_size": 20,
            },
        ),
        candidate_metrics=_metrics(
            top1=0.35,
            top3=0.70,
            mean_winner_rank=3.2,
            brier=0.10,
            logloss=1.9,
        ),
        thresholds=gate.ActivationThresholds(),
    )

    assert report["decision"] == "KEEP_QUARANTINED"
    assert "baseline_calibration_sample_below_min" in report["fail_reasons"]


def test_gate_blocks_unstable_train_holdout_population_ratio():
    feature = "same_distance_same_grade_avg_time"
    parity = {
        "by_feature": {
            feature: _parity_feature(
                feature,
                train_present_rows=80,
                train_rows=100,
                holdout_present_rows=4,
                holdout_rows=100,
            )
        }
    }

    report = gate.evaluate_feature_activation(
        feature=feature,
        parity_report=parity,
        inactive_policy_report={"inactive_features_due_to_train_all_missing": []},
        matrix_audit=_matrix_audit(),
        provenance_audit=_provenance_audit(),
        baseline_metrics=_metrics(),
        candidate_metrics=_metrics(top1=0.35, top3=0.70, mean_winner_rank=3.2, brier=0.10, logloss=1.9),
        thresholds=gate.ActivationThresholds(min_holdout_present_rows=1),
    )

    assert report["decision"] == "KEEP_QUARANTINED"
    assert "train_holdout_present_pct_ratio_unstable" in report["fail_reasons"]


def test_activation_report_summarizes_blocker_categories():
    feature = "same_distance_same_grade_best_time"
    report = gate.build_activation_report(
        candidate_features=[feature],
        parity_report={
            "by_feature": {
                feature: _parity_feature(
                    feature,
                    train_present_rows=0,
                    train_rows=100,
                    holdout_present_rows=4,
                    holdout_rows=100,
                )
            }
        },
        inactive_policy_report={"inactive_features_due_to_train_all_missing": [feature]},
        matrix_audit=_matrix_audit(),
        provenance_audit=_provenance_audit(
            target_distance_sources={"embedded_form_history": 1},
            same_distance_same_grade_history_provenance={
                "status": "NOT_VERIFIED",
                "by_feature": {},
            },
        ),
        baseline_metrics=_metrics(safe_joined_race_count=2),
        candidate_metrics=_metrics(safe_joined_race_count=2, probability_sum_max_error_joined_races=0.01),
        thresholds=gate.ActivationThresholds(min_holdout_present_rows=1),
        generated_at=datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc),
    )

    summary = report["fail_reason_summary"]
    assert summary["category_counts"]["feature_population_parity"] >= 1
    assert summary["category_counts"]["quarantine_policy"] == 1
    assert summary["category_counts"]["target_metadata_provenance"] >= 1
    assert summary["category_counts"]["same_distance_history_provenance"] >= 1
    assert summary["category_counts"]["shadow_metric_sample"] >= 1
    assert summary["category_counts"]["probability_safety"] == 1
    assert summary["features_by_category"]["quarantine_policy"] == [feature]
    assert summary["reason_counts"]["currently_inactive_due_to_train_all_missing_policy"] == 1


def test_gate_blocks_same_distance_activation_without_prior_history_provenance():
    feature = "same_distance_same_grade_avg_time"
    parity = {"by_feature": {feature: _parity_feature(feature)}}

    report = gate.evaluate_feature_activation(
        feature=feature,
        parity_report=parity,
        inactive_policy_report={"inactive_features_due_to_train_all_missing": []},
        matrix_audit=_matrix_audit(),
        provenance_audit=_provenance_audit(
            same_distance_same_grade_history_provenance={
                "status": "NOT_VERIFIED",
                "by_feature": {},
            }
        ),
        baseline_metrics=_metrics(),
        candidate_metrics=_metrics(top1=0.35, top3=0.70, mean_winner_rank=3.2, brier=0.10, logloss=1.9),
        thresholds=gate.ActivationThresholds(),
    )

    assert report["decision"] == "KEEP_QUARANTINED"
    assert "same_distance_same_grade_history_provenance_not_pass" in report["fail_reasons"]
    assert f"{feature}_history_provenance_missing" in report["fail_reasons"]


def test_gate_blocks_same_distance_activation_when_target_race_history_used():
    feature = "same_distance_same_grade_best_time"
    parity = {"by_feature": {feature: _parity_feature(feature)}}
    provenance = _provenance_audit()
    provenance["same_distance_same_grade_history_provenance"]["by_feature"][feature] = {
        "source": "prior_dog_history",
        "history_cutoff": "target_race_or_before",
        "prior_history_rows_used": 80,
        "target_race_rows_used": 1,
        "post_outcome_rows_used": 0,
        "post_outcome_fields_used": [],
    }

    report = gate.evaluate_feature_activation(
        feature=feature,
        parity_report=parity,
        inactive_policy_report={"inactive_features_due_to_train_all_missing": []},
        matrix_audit=_matrix_audit(),
        provenance_audit=provenance,
        baseline_metrics=_metrics(),
        candidate_metrics=_metrics(top1=0.35, top3=0.70, mean_winner_rank=3.2, brier=0.10, logloss=1.9),
        thresholds=gate.ActivationThresholds(),
    )

    assert report["decision"] == "KEEP_QUARANTINED"
    assert f"{feature}_history_cutoff_not_strictly_before_target_race" in report["fail_reasons"]
    assert f"{feature}_target_race_rows_used" in report["fail_reasons"]


def test_gate_writes_report_only_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/"
        "shadow_feature_activation_gate_20260608T120000+1000"
    )
    parity_path = tmp_path / "parity.json"
    inactive_path = tmp_path / "inactive.json"
    matrix_path = tmp_path / "matrix.json"
    provenance_path = tmp_path / "provenance.json"
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    feature = "same_distance_same_grade_best_time"
    parity_path.write_text(
        json.dumps({"by_feature": {feature: _parity_feature(feature, train_present_rows=0)}}),
        encoding="utf-8",
    )
    inactive_path.write_text(
        json.dumps({"inactive_features_due_to_train_all_missing": [feature]}),
        encoding="utf-8",
    )
    matrix_path.write_text(json.dumps(_matrix_audit()), encoding="utf-8")
    provenance_path.write_text(json.dumps(_provenance_audit()), encoding="utf-8")
    baseline_path.write_text(json.dumps(_metrics()), encoding="utf-8")
    candidate_path.write_text(json.dumps(_metrics()), encoding="utf-8")

    result = gate.run_activation_gate(
        parity_report_path=parity_path,
        inactive_policy_report_path=inactive_path,
        matrix_audit_path=matrix_path,
        provenance_audit_path=provenance_path,
        baseline_metrics_path=baseline_path,
        candidate_metrics_path=candidate_path,
        output_dir=output_dir,
        candidate_features=[feature],
        generated_at=datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc),
    )

    assert result["final_status"] == "FEATURE_ACTIVATION_BLOCKED_KEEP_QUARANTINED"
    report = json.loads((output_dir / "feature_activation_gate_report.json").read_text())
    assert report["canonical_schema_mutation"] is False
    assert report["db_write"] is False
    assert report["kept_quarantined_features"] == [feature]
    assert report["fail_reason_summary"]["category_counts"]["feature_population_parity"] >= 1
    summary = (output_dir / "SUMMARY.md").read_text(encoding="utf-8")
    assert "Blocker categories" in summary
    assert (output_dir / "final_status.txt").read_text(encoding="utf-8").strip() == (
        "FEATURE_ACTIVATION_BLOCKED_KEEP_QUARANTINED"
    )


def test_gate_accepts_configured_external_evidence_root(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", tmp_path / "repo")
    evidence_root = tmp_path / "retained_evidence"
    output_dir = evidence_root / "shadow_feature_activation_gate_20260608T120000+1000"
    assert gate.assert_output_dir_safe(output_dir, evidence_root=evidence_root) == output_dir.resolve()
