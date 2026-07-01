#!/usr/bin/env python3
"""Report-only activation gate for shadow features.

This tool decides whether quarantined shadow features have enough evidence to
be considered active in a future run. It never mutates the canonical feature
contract, model registry, production model artifacts, DB labels, snapshots, EV,
or betting outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from utils.report_output_dir_guard import assert_prefixed_report_output_dir  # noqa: E402

DEFAULT_OUTPUT_PARENT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/shadow_feature_activation_gate_"
OUTPUT_ARTIFACT_PREFIX = "shadow_feature_activation_gate_"
DEFAULT_CANDIDATE_FEATURES = (
    "same_distance_same_grade_best_time",
    "same_distance_same_grade_avg_time",
)
SAFE_TARGET_SOURCES = {"canonical_pre_race_page", "sidecar_target_metadata", "explicit_csv_sidecar"}
SAFE_SAME_DISTANCE_HISTORY_SOURCES = {"prior_dog_history"}
FINAL_ALLOWED = "FEATURE_ACTIVATION_ALLOWED_REPORT_ONLY"
FINAL_BLOCKED = "FEATURE_ACTIVATION_BLOCKED_KEEP_QUARANTINED"
TARGET_METADATA_READY_STATUS = "TARGET_METADATA_READY_FOR_CURRENT_OR_FUTURE_PREJUMP_INPUTS"
READY_SOURCE_STATUSES = {
    "CHALLENGER_CALIBRATION_REPORT_ONLY_READY_FOR_REVIEW",
    FINAL_ALLOWED,
}


@dataclass(frozen=True)
class ActivationThresholds:
    min_train_present_rows: int = 30
    min_train_present_pct: float = 0.05
    min_holdout_present_rows: int = 10
    min_holdout_present_pct: float = 0.05
    min_train_unique_present_values: int = 5
    min_holdout_unique_present_values: int = 5
    max_train_holdout_present_pct_ratio: float = 3.0
    min_shadow_joined_races: int = 100
    max_probability_sum_error: float = 1e-6
    metric_tolerance: float = 0.0
    min_calibration_sample_size: int = 50
    calibration_slope_tolerance: float = 0.0
    calibration_intercept_tolerance: float = 0.0
    max_box1_top_pick_share: float = 0.35
    box1_share_tolerance: float = 0.0


def relpath(path: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), ROOT.resolve())
    except ValueError:
        return str(path)


def now_id(now: datetime | None = None) -> str:
    return (now or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def assert_output_dir_safe(
    output_dir: Path,
    *,
    evidence_root: Path | None = None,
) -> Path:
    return assert_prefixed_report_output_dir(
        output_dir,
        repo_root=ROOT,
        repo_prefix=OUTPUT_PREFIX,
        artifact_prefix=OUTPUT_ARTIFACT_PREFIX,
        prefix_error="output_dir_must_be_shadow_feature_activation_gate_artifact",
        evidence_root=evidence_root,
    )


def ratio_too_unstable(train_pct: float, holdout_pct: float, max_ratio: float) -> bool:
    if train_pct <= 0 or holdout_pct <= 0:
        return train_pct != holdout_pct
    ratio = max(train_pct, holdout_pct) / min(train_pct, holdout_pct)
    return ratio > max_ratio


def matrix_safety_reasons(matrix_audit: Mapping[str, Any] | None) -> list[str]:
    if not matrix_audit:
        return ["matrix_audit_missing"]
    reasons = []
    if matrix_audit.get("status") != "PASS":
        reasons.append("matrix_audit_not_pass")
    matrix_gate = matrix_audit.get("matrix_gate") or {}
    if matrix_gate.get("status") != "PASS":
        reasons.append("matrix_gate_not_pass")
    temporal = matrix_audit.get("temporal_evaluation") or {}
    if temporal.get("status") != "PASS":
        reasons.append("temporal_evaluation_not_pass")
    if temporal.get("race_id_overlap"):
        reasons.append("train_holdout_race_overlap_present")
    label_audit = matrix_audit.get("label_audit") or {}
    if label_audit.get("status") != "PASS":
        reasons.append("label_audit_not_pass")
    schema_contract = matrix_audit.get("schema_contract") or {}
    if schema_contract.get("status") != "PASS":
        reasons.append("schema_contract_not_pass")
    if schema_contract.get("post_outcome_columns_present_as_features"):
        reasons.append("post_outcome_columns_present_as_features")
    if schema_contract.get("tgr_columns"):
        reasons.append("tgr_columns_present")
    return reasons


def same_distance_history_provenance_reasons(
    provenance_audit: Mapping[str, Any],
    feature: str,
) -> list[str]:
    if feature not in DEFAULT_CANDIDATE_FEATURES:
        return []
    history = provenance_audit.get("same_distance_same_grade_history_provenance")
    if not isinstance(history, Mapping):
        return ["same_distance_same_grade_history_provenance_missing"]
    reasons: list[str] = []
    if history.get("status") != "PASS":
        reasons.append("same_distance_same_grade_history_provenance_not_pass")
    by_feature = history.get("by_feature") if isinstance(history.get("by_feature"), Mapping) else {}
    feature_history = by_feature.get(feature) if isinstance(by_feature.get(feature), Mapping) else {}
    if not feature_history:
        reasons.append(f"{feature}_history_provenance_missing")
        return reasons
    source = str(feature_history.get("source") or "").strip()
    if source not in SAFE_SAME_DISTANCE_HISTORY_SOURCES:
        reasons.append(f"{feature}_unsafe_history_source:{source or 'missing'}")
    if feature_history.get("history_cutoff") != "strictly_before_target_race":
        reasons.append(f"{feature}_history_cutoff_not_strictly_before_target_race")
    if int(feature_history.get("target_race_rows_used") or 0) != 0:
        reasons.append(f"{feature}_target_race_rows_used")
    if int(feature_history.get("post_outcome_rows_used") or 0) != 0:
        reasons.append(f"{feature}_post_outcome_rows_used")
    if feature_history.get("post_outcome_fields_used"):
        reasons.append(f"{feature}_post_outcome_fields_used")
    if int(feature_history.get("prior_history_rows_used") or 0) <= 0:
        reasons.append(f"{feature}_prior_history_rows_used_missing")
    return reasons


def provenance_safety_reasons(
    provenance_audit: Mapping[str, Any] | None,
    *,
    feature: str,
) -> list[str]:
    if not provenance_audit:
        return ["provenance_audit_missing"]
    reasons = []
    if provenance_audit.get("protected_paths_unchanged") is False:
        reasons.append("protected_paths_changed")
    if int(provenance_audit.get("rejected_source_rows") or 0) > 0:
        reasons.append("provenance_rejected_source_rows_present")
    distance_sources = set(provenance_audit.get("target_distance_sources") or {})
    grade_sources = set(provenance_audit.get("target_grade_sources") or {})
    unsafe_distance = sorted(distance_sources - SAFE_TARGET_SOURCES)
    unsafe_grade = sorted(grade_sources - SAFE_TARGET_SOURCES)
    if unsafe_distance:
        reasons.append("unsafe_target_distance_sources:" + ",".join(unsafe_distance))
    if unsafe_grade:
        reasons.append("unsafe_target_grade_sources:" + ",".join(unsafe_grade))
    by_feature = provenance_audit.get("by_feature") or {}
    for target_feature in ("target_distance_safe", "target_grade_safe"):
        counts = by_feature.get(target_feature) or {}
        if int(counts.get("present_rows") or 0) <= 0:
            reasons.append(f"{target_feature}_not_populated_in_live_audit")
    reasons.extend(target_metadata_readiness_reasons(provenance_audit))
    reasons.extend(same_distance_history_provenance_reasons(provenance_audit, feature))
    return reasons


def target_metadata_readiness_reasons(provenance_audit: Mapping[str, Any]) -> list[str]:
    readiness = provenance_audit.get("target_metadata_readiness")
    if not isinstance(readiness, Mapping):
        return ["target_metadata_readiness_missing"]
    reasons: list[str] = []
    status = str(readiness.get("status") or "").strip()
    capture_status = str(readiness.get("target_metadata_capture_status") or "").strip()
    if status != TARGET_METADATA_READY_STATUS or capture_status != "READY":
        reasons.append(f"target_metadata_readiness_not_ready:{status or capture_status or 'missing'}")
    input_count = readiness.get("current_or_future_input_count")
    ready_count = readiness.get("ready_current_or_future_input_count")
    if ready_count in (None, ""):
        ready_count = readiness.get("verified_eligible_count")
    if input_count not in (None, "") and ready_count not in (None, ""):
        try:
            if int(input_count) != int(ready_count):
                reasons.append("target_metadata_ready_input_count_mismatch")
        except (TypeError, ValueError):
            reasons.append("target_metadata_ready_input_count_invalid")
    blocker_counts = readiness.get("blocker_counts")
    if isinstance(blocker_counts, Mapping):
        for blocker, count in sorted(blocker_counts.items()):
            try:
                blocked_count = int(count)
            except (TypeError, ValueError):
                blocked_count = 1 if count else 0
            if blocked_count > 0:
                reasons.append(f"target_metadata_blocked:{blocker}")
    policy = str(readiness.get("historical_repair_policy") or "").strip()
    if policy and policy != "NO_REPAIR_WITHOUT_PROVENANCE_SAFE_PRE_RACE_SOURCE":
        reasons.append(f"unsafe_historical_repair_policy:{policy}")
    return reasons


def metric_safety_reasons(
    baseline_metrics: Mapping[str, Any] | None,
    candidate_metrics: Mapping[str, Any] | None,
    thresholds: ActivationThresholds,
) -> list[str]:
    if not baseline_metrics or not candidate_metrics:
        return ["missing_shadow_metric_comparison"]
    reasons = []
    baseline_races = int(baseline_metrics.get("safe_joined_race_count") or 0)
    candidate_races = int(candidate_metrics.get("safe_joined_race_count") or 0)
    if baseline_races < thresholds.min_shadow_joined_races:
        reasons.append("baseline_joined_race_sample_below_min")
    if candidate_races < thresholds.min_shadow_joined_races:
        reasons.append("candidate_joined_race_sample_below_min")
    reasons.extend(metric_sample_comparability_reasons(baseline_metrics, candidate_metrics))
    reasons.extend(metric_source_status_reasons(candidate_metrics))
    max_sum_error = candidate_metrics.get("probability_sum_max_error_joined_races")
    if max_sum_error is None or float(max_sum_error) > thresholds.max_probability_sum_error:
        reasons.append("candidate_probability_sum_error_failed")

    comparisons = (
        ("top1", "higher_or_equal"),
        ("top3", "higher_or_equal"),
        ("mean_winner_rank", "lower_or_equal"),
        ("brier", "lower_or_equal"),
        ("logloss", "lower_or_equal"),
    )
    for key, direction in comparisons:
        baseline_value = baseline_metrics.get(key)
        candidate_value = candidate_metrics.get(key)
        if baseline_value is None or candidate_value is None:
            reasons.append(f"metric_missing:{key}")
            continue
        baseline_float = float(baseline_value)
        candidate_float = float(candidate_value)
        tolerance = thresholds.metric_tolerance
        if direction == "higher_or_equal" and candidate_float + tolerance < baseline_float:
            reasons.append(f"metric_regressed:{key}")
        if direction == "lower_or_equal" and candidate_float > baseline_float + tolerance:
            reasons.append(f"metric_regressed:{key}")
    reasons.extend(calibration_safety_reasons(baseline_metrics, candidate_metrics, thresholds))
    reasons.extend(box_bias_safety_reasons(baseline_metrics, candidate_metrics, thresholds))
    return reasons


def metric_source_status_reasons(candidate_metrics: Mapping[str, Any] | None) -> list[str]:
    """Honor readiness and blockers from the report that generated candidate metrics."""

    reasons: list[str] = []
    source_status = str((candidate_metrics or {}).get("source_final_status") or "").strip()
    if source_status and source_status not in READY_SOURCE_STATUSES:
        reasons.append(f"candidate_metric_source_not_ready:{source_status}")

    blockers = (candidate_metrics or {}).get("source_activation_blockers") or []
    if isinstance(blockers, Sequence) and not isinstance(blockers, (str, bytes)):
        for blocker in blockers:
            if blocker:
                reasons.append(f"candidate_metric_source_blocked:{blocker}")
    elif blockers:
        reasons.append(f"candidate_metric_source_blocked:{blockers}")
    return reasons


def number_from_paths(payload: Mapping[str, Any] | None, paths: Sequence[Sequence[str]]) -> float | None:
    for path in paths:
        current: Any = payload
        for key in path:
            if not isinstance(current, Mapping):
                current = None
                break
            current = current.get(key)
        if current in (None, ""):
            continue
        try:
            return float(current)
        except (TypeError, ValueError):
            continue
    return None


def metric_sample_comparability_reasons(
    baseline_metrics: Mapping[str, Any] | None,
    candidate_metrics: Mapping[str, Any] | None,
) -> list[str]:
    """Require baseline and candidate metrics to describe the same joined sample."""

    reasons: list[str] = []
    count_fields = (
        ("safe_joined_race_count", (("safe_joined_race_count",),)),
        ("safe_joined_runner_count", (("safe_joined_runner_count",),)),
    )
    for field, paths in count_fields:
        baseline_value = number_from_paths(baseline_metrics, paths)
        candidate_value = number_from_paths(candidate_metrics, paths)
        if baseline_value is None or candidate_value is None:
            continue
        if int(baseline_value) != int(candidate_value):
            reasons.append(f"metric_sample_mismatch:{field}")

    for field in (
        "evaluation_cohort_id",
        "metric_cohort_id",
        "safe_joined_race_ids_hash",
        "joined_race_ids_hash",
    ):
        baseline_value = (baseline_metrics or {}).get(field)
        candidate_value = (candidate_metrics or {}).get(field)
        if baseline_value in (None, "") or candidate_value in (None, ""):
            continue
        if str(baseline_value) != str(candidate_value):
            reasons.append(f"metric_cohort_mismatch:{field}")
    return reasons


def calibration_fields(metrics: Mapping[str, Any] | None) -> dict[str, float | None]:
    return {
        "slope": number_from_paths(
            metrics,
            (
                ("calibration", "slope"),
                ("calibration", "slope_intercept", "slope"),
                ("slope_intercept", "slope"),
                ("calibration_slope_intercept", "slope"),
                ("calibration_slope",),
            ),
        ),
        "intercept": number_from_paths(
            metrics,
            (
                ("calibration", "intercept"),
                ("calibration", "slope_intercept", "intercept"),
                ("slope_intercept", "intercept"),
                ("calibration_slope_intercept", "intercept"),
                ("calibration_intercept",),
            ),
        ),
        "sample_size": number_from_paths(
            metrics,
            (
                ("calibration", "sample_size"),
                ("calibration", "slope_intercept", "sample_size"),
                ("slope_intercept", "sample_size"),
                ("calibration_slope_intercept", "sample_size"),
                ("calibration_sample_size",),
                ("safe_joined_runner_count",),
            ),
        ),
    }


def calibration_safety_reasons(
    baseline_metrics: Mapping[str, Any] | None,
    candidate_metrics: Mapping[str, Any] | None,
    thresholds: ActivationThresholds,
) -> list[str]:
    reasons: list[str] = []
    baseline = calibration_fields(baseline_metrics)
    candidate = calibration_fields(candidate_metrics)
    for key in ("slope", "intercept"):
        if baseline[key] is None:
            reasons.append(f"metric_missing:calibration_{key}_baseline")
        if candidate[key] is None:
            reasons.append(f"metric_missing:calibration_{key}_candidate")
    baseline_sample_size = baseline["sample_size"]
    if baseline_sample_size is None:
        reasons.append("metric_missing:calibration_sample_size_baseline")
    elif baseline_sample_size < thresholds.min_calibration_sample_size:
        reasons.append("baseline_calibration_sample_below_min")
    candidate_sample_size = candidate["sample_size"]
    if candidate_sample_size is None:
        reasons.append("metric_missing:calibration_sample_size_candidate")
    elif candidate_sample_size < thresholds.min_calibration_sample_size:
        reasons.append("candidate_calibration_sample_below_min")
    if baseline["slope"] is not None and candidate["slope"] is not None:
        baseline_distance = abs(1.0 - baseline["slope"])
        candidate_distance = abs(1.0 - candidate["slope"])
        if candidate_distance > baseline_distance + thresholds.calibration_slope_tolerance:
            reasons.append("metric_regressed:calibration_slope")
    if baseline["intercept"] is not None and candidate["intercept"] is not None:
        baseline_distance = abs(baseline["intercept"])
        candidate_distance = abs(candidate["intercept"])
        if candidate_distance > baseline_distance + thresholds.calibration_intercept_tolerance:
            reasons.append("metric_regressed:calibration_intercept")
    return reasons


def box1_top_pick_share(metrics: Mapping[str, Any] | None) -> float | None:
    direct = number_from_paths(
        metrics,
        (
            ("box_1_share",),
            ("box1_top_pick_share",),
            ("safe_joined_box_1_top_pick_share",),
            ("box_bias", "safe_joined_box_1_top_pick_share"),
        ),
    )
    if direct is not None:
        return direct
    distribution = None
    if metrics:
        distribution = (
            metrics.get("safe_joined_top_pick_box_distribution")
            or metrics.get("top_pick_box_distribution")
        )
        box_bias = metrics.get("box_bias")
        if distribution is None and isinstance(box_bias, Mapping):
            distribution = box_bias.get("safe_joined_top_pick_box_distribution")
    if not isinstance(distribution, Mapping):
        return None
    total = 0.0
    box1 = 0.0
    for box, count in distribution.items():
        try:
            value = float(count)
        except (TypeError, ValueError):
            continue
        total += value
        if str(box) in {"1", "1.0"}:
            box1 += value
    return box1 / total if total > 0 else None


def box_bias_safety_reasons(
    baseline_metrics: Mapping[str, Any] | None,
    candidate_metrics: Mapping[str, Any] | None,
    thresholds: ActivationThresholds,
) -> list[str]:
    baseline_share = box1_top_pick_share(baseline_metrics)
    candidate_share = box1_top_pick_share(candidate_metrics)
    reasons: list[str] = []
    if baseline_share is None:
        reasons.append("metric_missing:box1_top_pick_share_baseline")
    if candidate_share is None:
        reasons.append("metric_missing:box1_top_pick_share_candidate")
        return reasons
    if candidate_share > thresholds.max_box1_top_pick_share:
        reasons.append("candidate_box1_top_pick_share_above_max")
    if baseline_share is not None and candidate_share > baseline_share + thresholds.box1_share_tolerance:
        reasons.append("metric_regressed:box1_top_pick_share")
    return reasons


def fail_reason_category(reason: str) -> str:
    if reason in {
        "feature_missing_from_parity_report",
        "all_missing_in_train",
        "all_missing_in_holdout",
        "train_present_rows_below_min",
        "train_present_pct_below_min",
        "holdout_present_rows_below_min",
        "holdout_present_pct_below_min",
        "train_unique_present_values_below_min",
        "holdout_unique_present_values_below_min",
        "train_holdout_present_pct_ratio_unstable",
    }:
        return "feature_population_parity"
    if reason == "currently_inactive_due_to_train_all_missing_policy":
        return "quarantine_policy"
    if reason.startswith("matrix_") or reason in {
        "temporal_evaluation_not_pass",
        "train_holdout_race_overlap_present",
        "label_audit_not_pass",
        "schema_contract_not_pass",
        "post_outcome_columns_present_as_features",
        "tgr_columns_present",
    }:
        return "matrix_or_schema_safety"
    if reason.startswith("same_distance_same_grade_") or "_history_" in reason:
        return "same_distance_history_provenance"
    if (
        reason.startswith("unsafe_target_")
        or reason.startswith("target_distance_safe_")
        or reason.startswith("target_grade_safe_")
        or reason.startswith("target_metadata_")
        or reason.startswith("unsafe_historical_repair_policy:")
        or reason in {
            "provenance_audit_missing",
            "protected_paths_changed",
            "provenance_rejected_source_rows_present",
        }
    ):
        return "target_metadata_provenance"
    if reason in {
        "missing_shadow_metric_comparison",
        "baseline_joined_race_sample_below_min",
        "candidate_joined_race_sample_below_min",
    } or reason.startswith("metric_sample_mismatch:") or reason.startswith("metric_cohort_mismatch:"):
        return "shadow_metric_sample"
    if reason.startswith("candidate_metric_source_not_ready:") or reason.startswith(
        "candidate_metric_source_blocked:"
    ):
        return "shadow_metric_sample"
    if reason == "candidate_probability_sum_error_failed":
        return "probability_safety"
    if reason.startswith("metric_missing:") or reason.startswith("metric_regressed:"):
        return "shadow_metric_regression"
    if reason in {
        "baseline_calibration_sample_below_min",
        "candidate_calibration_sample_below_min",
        "candidate_box1_top_pick_share_above_max",
    }:
        return "shadow_metric_regression"
    return "other"


def summarize_fail_reasons(feature_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    features_by_category: dict[str, list[str]] = {}
    for row in feature_rows:
        feature = str(row.get("feature") or "")
        for reason in row.get("fail_reasons") or []:
            reason_text = str(reason)
            category = fail_reason_category(reason_text)
            reason_counts[reason_text] = reason_counts.get(reason_text, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1
            features = features_by_category.setdefault(category, [])
            if feature and feature not in features:
                features.append(feature)
    return {
        "reason_counts": dict(sorted(reason_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "features_by_category": {
            category: sorted(features)
            for category, features in sorted(features_by_category.items())
        },
    }


def evaluate_feature_activation(
    *,
    feature: str,
    parity_report: Mapping[str, Any],
    inactive_policy_report: Mapping[str, Any] | None,
    matrix_audit: Mapping[str, Any] | None,
    provenance_audit: Mapping[str, Any] | None,
    baseline_metrics: Mapping[str, Any] | None,
    candidate_metrics: Mapping[str, Any] | None,
    thresholds: ActivationThresholds,
) -> dict[str, Any]:
    reasons: list[str] = []
    feature_report = (parity_report.get("by_feature") or {}).get(feature)
    if not feature_report:
        reasons.append("feature_missing_from_parity_report")
        feature_report = {"feature": feature}
    else:
        if feature_report.get("all_missing_in_train"):
            reasons.append("all_missing_in_train")
        if feature_report.get("all_missing_in_holdout"):
            reasons.append("all_missing_in_holdout")
        if int(feature_report.get("train_present_rows") or 0) < thresholds.min_train_present_rows:
            reasons.append("train_present_rows_below_min")
        if float(feature_report.get("train_present_pct") or 0.0) < thresholds.min_train_present_pct:
            reasons.append("train_present_pct_below_min")
        if int(feature_report.get("holdout_present_rows") or 0) < thresholds.min_holdout_present_rows:
            reasons.append("holdout_present_rows_below_min")
        if float(feature_report.get("holdout_present_pct") or 0.0) < thresholds.min_holdout_present_pct:
            reasons.append("holdout_present_pct_below_min")
        if int(feature_report.get("train_unique_present_values") or 0) < thresholds.min_train_unique_present_values:
            reasons.append("train_unique_present_values_below_min")
        if int(feature_report.get("holdout_unique_present_values") or 0) < thresholds.min_holdout_unique_present_values:
            reasons.append("holdout_unique_present_values_below_min")
        if ratio_too_unstable(
            float(feature_report.get("train_present_pct") or 0.0),
            float(feature_report.get("holdout_present_pct") or 0.0),
            thresholds.max_train_holdout_present_pct_ratio,
        ):
            reasons.append("train_holdout_present_pct_ratio_unstable")

    inactive_features = set(
        (inactive_policy_report or {}).get("inactive_features_due_to_train_all_missing") or []
    )
    if feature in inactive_features:
        reasons.append("currently_inactive_due_to_train_all_missing_policy")
    reasons.extend(matrix_safety_reasons(matrix_audit))
    reasons.extend(provenance_safety_reasons(provenance_audit, feature=feature))
    reasons.extend(metric_safety_reasons(baseline_metrics, candidate_metrics, thresholds))

    deduped_reasons = list(dict.fromkeys(reasons))
    decision = "ACTIVATE_ALLOWED_REPORT_ONLY" if not deduped_reasons else "KEEP_QUARANTINED"
    return {
        "feature": feature,
        "decision": decision,
        "fail_reasons": deduped_reasons,
        "parity": feature_report,
    }


def build_activation_report(
    *,
    candidate_features: Sequence[str],
    parity_report: Mapping[str, Any],
    inactive_policy_report: Mapping[str, Any] | None,
    matrix_audit: Mapping[str, Any] | None,
    provenance_audit: Mapping[str, Any] | None,
    baseline_metrics: Mapping[str, Any] | None,
    candidate_metrics: Mapping[str, Any] | None,
    thresholds: ActivationThresholds,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    features = [
        evaluate_feature_activation(
            feature=feature,
            parity_report=parity_report,
            inactive_policy_report=inactive_policy_report,
            matrix_audit=matrix_audit,
            provenance_audit=provenance_audit,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            thresholds=thresholds,
        )
        for feature in candidate_features
    ]
    allowed = [row["feature"] for row in features if row["decision"] == "ACTIVATE_ALLOWED_REPORT_ONLY"]
    quarantined = [row["feature"] for row in features if row["decision"] == "KEEP_QUARANTINED"]
    final_status = FINAL_ALLOWED if allowed and not quarantined else FINAL_BLOCKED
    fail_reason_summary = summarize_fail_reasons(features)
    return {
        "schema_version": "shadow_feature_activation_gate_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": final_status,
        "canonical_schema_mutation": False,
        "model_registry_mutation": False,
        "production_model_mutation": False,
        "production_prediction_write": False,
        "db_write": False,
        "label_write": False,
        "tgr_enabled": False,
        "betting_or_ev_output": False,
        "thresholds": asdict(thresholds),
        "candidate_features": list(candidate_features),
        "activation_allowed_features": allowed,
        "kept_quarantined_features": quarantined,
        "fail_reason_summary": fail_reason_summary,
        "features": features,
    }


def build_summary(report: Mapping[str, Any]) -> str:
    lines = [
        "# Shadow Feature Activation Gate",
        "",
        f"- Final status: `{report.get('final_status')}`",
        f"- Candidate features: `{report.get('candidate_features')}`",
        f"- Activation allowed: `{report.get('activation_allowed_features')}`",
        f"- Kept quarantined: `{report.get('kept_quarantined_features')}`",
        f"- Blocker categories: `{(report.get('fail_reason_summary') or {}).get('category_counts')}`",
        "",
        "## Feature Decisions",
    ]
    for feature in report.get("features") or []:
        lines.append(
            f"- `{feature.get('feature')}`: `{feature.get('decision')}` "
            f"reasons=`{feature.get('fail_reasons')}`"
        )
    lines.extend(
        [
            "",
            "## No-Write Guarantees",
            "- Report-only gate; no canonical schema, registry, production model, DB, label, snapshot, EV, betting, TGR, or production prediction mutation.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_activation_gate(
    *,
    parity_report_path: Path,
    inactive_policy_report_path: Path | None = None,
    matrix_audit_path: Path | None = None,
    provenance_audit_path: Path | None = None,
    baseline_metrics_path: Path | None = None,
    candidate_metrics_path: Path | None = None,
    output_parent: Path = DEFAULT_OUTPUT_PARENT,
    output_dir: Path | None = None,
    evidence_root: Path | None = None,
    candidate_features: Sequence[str] = DEFAULT_CANDIDATE_FEATURES,
    thresholds: ActivationThresholds = ActivationThresholds(),
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    output_dir = output_dir or output_parent / f"shadow_feature_activation_gate_{now_id(generated_at)}"
    output_dir = assert_output_dir_safe(output_dir, evidence_root=evidence_root)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = build_activation_report(
        candidate_features=candidate_features,
        parity_report=load_json(parity_report_path) or {},
        inactive_policy_report=load_json(inactive_policy_report_path),
        matrix_audit=load_json(matrix_audit_path),
        provenance_audit=load_json(provenance_audit_path),
        baseline_metrics=load_json(baseline_metrics_path),
        candidate_metrics=load_json(candidate_metrics_path),
        thresholds=thresholds,
        generated_at=generated_at,
    )
    report["source_paths"] = {
        "parity_report": relpath(parity_report_path),
        "inactive_policy_report": relpath(inactive_policy_report_path) if inactive_policy_report_path else None,
        "matrix_audit": relpath(matrix_audit_path) if matrix_audit_path else None,
        "provenance_audit": relpath(provenance_audit_path) if provenance_audit_path else None,
        "baseline_metrics": relpath(baseline_metrics_path) if baseline_metrics_path else None,
        "candidate_metrics": relpath(candidate_metrics_path) if candidate_metrics_path else None,
    }
    write_json(output_dir / "feature_activation_gate_report.json", report)
    write_text(output_dir / "SUMMARY.md", build_summary(report))
    write_text(output_dir / "final_status.txt", str(report["final_status"]) + "\n")
    return {
        "output_dir": relpath(output_dir),
        "final_status": report["final_status"],
        "activation_allowed_features": report["activation_allowed_features"],
        "kept_quarantined_features": report["kept_quarantined_features"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity-report", required=True, type=Path)
    parser.add_argument("--inactive-policy-report", type=Path)
    parser.add_argument("--matrix-audit", type=Path)
    parser.add_argument("--provenance-audit", type=Path)
    parser.add_argument("--baseline-metrics", type=Path)
    parser.add_argument("--candidate-metrics", type=Path)
    parser.add_argument("--output-parent", default=DEFAULT_OUTPUT_PARENT, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evidence-root", default=DEFAULT_OUTPUT_PARENT, type=Path)
    parser.add_argument("--feature", action="append", dest="features")
    parser.add_argument("--min-train-present-rows", type=int, default=ActivationThresholds.min_train_present_rows)
    parser.add_argument("--min-train-present-pct", type=float, default=ActivationThresholds.min_train_present_pct)
    parser.add_argument("--min-holdout-present-rows", type=int, default=ActivationThresholds.min_holdout_present_rows)
    parser.add_argument("--min-holdout-present-pct", type=float, default=ActivationThresholds.min_holdout_present_pct)
    parser.add_argument("--min-shadow-joined-races", type=int, default=ActivationThresholds.min_shadow_joined_races)
    parser.add_argument("--min-calibration-sample-size", type=int, default=ActivationThresholds.min_calibration_sample_size)
    parser.add_argument("--calibration-slope-tolerance", type=float, default=ActivationThresholds.calibration_slope_tolerance)
    parser.add_argument("--calibration-intercept-tolerance", type=float, default=ActivationThresholds.calibration_intercept_tolerance)
    parser.add_argument("--max-box1-top-pick-share", type=float, default=ActivationThresholds.max_box1_top_pick_share)
    parser.add_argument("--box1-share-tolerance", type=float, default=ActivationThresholds.box1_share_tolerance)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    thresholds = ActivationThresholds(
        min_train_present_rows=args.min_train_present_rows,
        min_train_present_pct=args.min_train_present_pct,
        min_holdout_present_rows=args.min_holdout_present_rows,
        min_holdout_present_pct=args.min_holdout_present_pct,
        min_shadow_joined_races=args.min_shadow_joined_races,
        min_calibration_sample_size=args.min_calibration_sample_size,
        calibration_slope_tolerance=args.calibration_slope_tolerance,
        calibration_intercept_tolerance=args.calibration_intercept_tolerance,
        max_box1_top_pick_share=args.max_box1_top_pick_share,
        box1_share_tolerance=args.box1_share_tolerance,
    )
    result = run_activation_gate(
        parity_report_path=args.parity_report,
        inactive_policy_report_path=args.inactive_policy_report,
        matrix_audit_path=args.matrix_audit,
        provenance_audit_path=args.provenance_audit,
        baseline_metrics_path=args.baseline_metrics,
        candidate_metrics_path=args.candidate_metrics,
        output_parent=args.output_parent,
        output_dir=args.output_dir,
        evidence_root=args.evidence_root,
        candidate_features=args.features or DEFAULT_CANDIDATE_FEATURES,
        thresholds=thresholds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
