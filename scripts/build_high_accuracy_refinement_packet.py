#!/usr/bin/env python3
"""Build the report-only high-accuracy refinement packet.

The packet compares baseline forward-shadow evidence, the Stage 2 non-TGR
challenger, and optional odds-augmented research reports. It is deliberately a
PR gate artifact only: it never mutates the model registry, DB labels,
prediction snapshots, production pointers, EV actions, betting output, or TGR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from scripts import build_promotion_gate_contract_audit_packet as gate_contract_audit  # noqa: E402


OUTPUT_PREFIX = (
    "artifacts/full_evidence_orchestration_20260525/"
    "high_accuracy_refinement_packet_"
)
DEFAULT_OUTPUT_PARENT = ROOT / "artifacts/full_evidence_orchestration_20260525"
DEFAULT_PROTECTED_PATHS = (
    ROOT / "greyhound_racing_data.db",
    ROOT / "greyhound_racing_data_writable.db",
    ROOT / "model_registry/best_metadata.json",
    ROOT / "model_registry/current_production.json",
    ROOT / "docs/model_contracts/v4_feature_contract.json",
    ROOT / "artifacts/prediction_snapshots/manifest.jsonl",
)

SCHEMA_VERSION = "high_accuracy_refinement_packet_v2"
FINAL_READY_FOR_PR = "READY_FOR_PROMOTION_PR_DRAFT"
FINAL_NEEDS_MODEL_CHALLENGER = "NEEDS_NON_TGR_MODEL_CHALLENGER_TRAINING"
FINAL_BLOCKED = "BLOCKED_KEEP_BASELINE"

CALIBRATION_READY = "CHALLENGER_CALIBRATION_REPORT_ONLY_READY_FOR_REVIEW"
MODEL_CANDIDATE_KEY = "shadow_calibrated_rf_power_gamma_2_4"
MODEL_BASELINE_KEY = "champion_baseline"

STAGE2_FORWARD_SHADOW_COLLECTING = "STAGE2_FORWARD_SHADOW_COLLECTING"
STAGE2_FORWARD_SHADOW_READY_FOR_REVIEW = "STAGE2_FORWARD_SHADOW_READY_FOR_REVIEW"
STAGE2_FORWARD_METRICS_FROM_ROLLING = "STAGE2_FORWARD_METRICS_FROM_ROLLING"
STAGE2_PREDICTIONS_COLLECTED_METRICS_MISSING = (
    "STAGE2_PREDICTIONS_COLLECTED_METRICS_MISSING"
)
ODDS_RESEARCH_BLOCKED_PROVENANCE = "ODDS_RESEARCH_BLOCKED_PROVENANCE"
ODDS_RESEARCH_READY_REPORT_ONLY = "ODDS_RESEARCH_READY_REPORT_ONLY"
ODDS_AUGMENTED_MODEL_BLOCKED = "ODDS_AUGMENTED_MODEL_BLOCKED"
ODDS_AUGMENTED_MODEL_READY_FOR_PR_REVIEW = "ODDS_AUGMENTED_MODEL_READY_FOR_PR_REVIEW"
ROLLING_MODEL_COMPARISON_COLLECTING = "ROLLING_MODEL_COMPARISON_COLLECTING"
ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW = "ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW"
MARKET_ONLY_CANDIDATE_KEYS = {"market_only_implied"}
MIN_COMPLETE_VALID_PREJUMP_ODDS_RACES = 100
MIN_UNIFIED_EVIDENCE_ELIGIBLE_ROWS = 100
ODDS_AUGMENTED_GATE_CONTRACT_POLICY = "dual_baseline_market_rank_primary_safety"
ACCEPTED_UNIFIED_EVIDENCE_AGGREGATE_STATUSES = {
    "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT",
    "REJOIN_UNIFIED_EVIDENCE_DATASETS_BUILT",
}


@dataclass(frozen=True)
class AccuracyGateThresholds:
    min_safe_joined_races: int = 100
    min_top1_delta: float = 0.02
    min_top3_delta: float = 0.0
    max_mean_winner_rank_delta: float = 0.0
    max_brier_delta: float = 0.0
    max_logloss_delta: float = 0.0
    max_calibration_distance_delta: float = 0.0
    max_box1_top_pick_share: float = 0.35
    max_box1_share_delta: float = 0.0
    max_probability_sum_error: float = 1e-6


def now_id(now: datetime | None = None) -> str:
    return (now or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return os.path.relpath(path.resolve(), ROOT.resolve())
    except ValueError:
        return str(path)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("json_root_not_object")
    return payload


def count_jsonl_rows(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def resolve_repo_path(path_value: Any) -> Path | None:
    if not str(path_value or "").strip():
        return None
    path = Path(str(path_value))
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes(paths: Sequence[Path] = DEFAULT_PROTECTED_PATHS) -> dict[str, str | None]:
    return {relpath(path) or str(path): sha256_file(path) for path in paths}


def assert_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative = logical.absolute().relative_to(ROOT.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_high_accuracy_refinement_packet:{relative}")
    return logical.absolute()


def unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = Path(f"{base}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"output_dir_collision_exhausted:{base}")


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def finite_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def metric_int(metrics: Mapping[str, Any], key: str) -> int:
    try:
        return int(metrics.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def metric(metrics: Mapping[str, Any], key: str) -> float | None:
    if key == "logloss":
        return finite_float(metrics.get("logloss", metrics.get("log_loss")))
    if key == "box1_top_pick_share":
        direct = finite_float(metrics.get("box1_top_pick_share"))
        if direct is not None:
            return direct
        return finite_float(mapping(metrics.get("box_bias")).get("box1_top_pick_share"))
    return finite_float(metrics.get(key))


def string_list(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return []


def int_count_mapping(value: Any) -> dict[str, int]:
    return {
        str(reason): finite_int(count)
        for reason, count in sorted(mapping(value).items())
    }


def unified_official_result_coverage_summary(
    unified_report: Mapping[str, Any],
    official_audit: Mapping[str, Any],
) -> dict[str, Any]:
    direct = mapping(unified_report.get("official_result_coverage"))
    if direct:
        coverage = {
            "source": direct.get("source"),
            "requested_race_count": finite_int(direct.get("requested_race_count")),
            "races_with_rows_count": finite_int(direct.get("races_with_rows_count")),
            "missing_race_count": finite_int(direct.get("missing_race_count")),
            "missing_race_ids": string_list(direct.get("missing_race_ids")),
            "races_with_rows": string_list(direct.get("races_with_rows")),
            "runner_path_count": finite_int(direct.get("runner_path_count")),
            "runner_paths_source_field": direct.get("runner_paths_source_field"),
            "missing_exclusion_count": finite_int(
                direct.get("missing_exclusion_count")
            ),
        }
        if "requested_race_count_source" in direct:
            coverage["requested_race_count_source"] = direct.get(
                "requested_race_count_source"
            )
        if "requested_race_ids" in direct:
            coverage["requested_race_ids"] = string_list(direct.get("requested_race_ids"))
        if "legacy_requested_race_count_without_ids" in direct:
            coverage["legacy_requested_race_count_without_ids"] = finite_int(
                direct.get("legacy_requested_race_count_without_ids")
            )
        return coverage
    missing_race_ids = string_list(official_audit.get("missing_race_ids"))
    races_with_rows = string_list(official_audit.get("race_ids_with_rows"))
    runner_paths = string_list(unified_report.get("official_result_runner_paths"))
    return {
        "source": "unified_evidence_report",
        "requested_race_count": finite_int(official_audit.get("race_ids_requested")),
        "races_with_rows_count": len(races_with_rows),
        "missing_race_count": len(missing_race_ids),
        "missing_race_ids": missing_race_ids,
        "races_with_rows": races_with_rows,
        "runner_path_count": len(runner_paths),
        "runner_paths_source_field": "official_result_runner_paths",
        "missing_exclusion_count": int_count_mapping(
            unified_report.get("exclusion_reason_counts")
        ).get("official_result_missing", 0),
    }


def compact_unified_gap_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    compact_rows: list[dict[str, Any]] = []
    allowed_fields = (
        "race_id",
        "race_date",
        "venue",
        "recommended_action",
        "action",
        "evidence_missing_reason",
        "has_unified_evidence_instance",
        "has_complete_official_result_instance",
        "has_complete_strict_prejump_odds_instance",
        "official_result_quarantine_reason",
        "official_result_quarantine_errors",
        "official_result_quarantine_source_urls",
        "official_result_quarantine_participant_source",
        "official_result_quarantine_participant_count",
        "official_result_quarantine_participant_boxes",
        "official_result_quarantine_result_boxes_not_in_participants",
        "official_result_quarantine_result_boxes_in_participants",
        "official_result_quarantine_participants",
        "official_result_quarantine_attempted_source_box_sets",
        "official_result_quarantine_reserve_substitution_diagnostic",
    )
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        compact = {
            key: row.get(key)
            for key in allowed_fields
            if row.get(key) not in (None, "", [], {})
        }
        if compact.get("race_id"):
            compact_rows.append(compact)
    return compact_rows


def backlog_unified_gap_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    race_coverage = mapping(report.get("race_coverage"))
    gap_action_plan = mapping(report.get("gap_action_plan")) or mapping(
        race_coverage.get("gap_action_plan")
    )
    top_gap_races = compact_unified_gap_rows(
        report.get("top_gap_races") or gap_action_plan.get("top_gap_races")
    )
    top_official_result_missing_races = compact_unified_gap_rows(
        report.get("top_official_result_missing_races")
        or gap_action_plan.get("top_official_result_missing_races")
    )
    top_gap_race_ids = string_list(report.get("top_gap_race_ids")) or [
        str(row.get("race_id"))
        for row in top_gap_races
        if row.get("race_id")
    ]
    top_official_result_missing_race_ids = string_list(
        report.get("top_official_result_missing_race_ids")
    ) or [
        str(row.get("race_id"))
        for row in top_official_result_missing_races
        if row.get("race_id")
    ]
    sample_blocking_gap_count = (
        report.get("sample_blocking_gap_count")
        if "sample_blocking_gap_count" in report
        else gap_action_plan.get("sample_blocking_gap_count")
    )
    return {
        "sample_blocking_gap_count": finite_int(sample_blocking_gap_count),
        "gap_action_counts": int_count_mapping(
            report.get("gap_action_counts") or gap_action_plan.get("action_counts")
        ),
        "gap_evidence_missing_reason_counts": int_count_mapping(
            report.get("evidence_missing_reason_counts")
            or gap_action_plan.get("evidence_missing_reason_counts")
        ),
        "top_gap_race_ids": top_gap_race_ids,
        "top_gap_races": top_gap_races,
        "top_official_result_missing_race_ids": top_official_result_missing_race_ids,
        "top_official_result_missing_races": top_official_result_missing_races,
    }


def artifact_odds_rejection_reason_counts(report: Mapping[str, Any]) -> dict[str, int]:
    direct = mapping(report.get("artifact_odds_rejection_reason_counts"))
    if direct:
        return int_count_mapping(direct)
    counts: Counter[str] = Counter()
    for audit in report.get("artifact_odds_audits") or []:
        if not isinstance(audit, Mapping):
            continue
        for reason, count in mapping(audit.get("rejection_reason_counts")).items():
            counts[str(reason)] += finite_int(count)
    return dict(sorted(counts.items()))


def rejected_live_odds_candidate_reason_counts(
    report: Mapping[str, Any],
) -> dict[str, int]:
    direct = mapping(report.get("rejected_live_odds_candidate_reason_counts"))
    return {
        str(reason): finite_int(count)
        for reason, count in sorted(direct.items())
    }


def enriched_unified_evidence_aggregate_status(
    status: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    report = mapping(status)
    if not report:
        return None
    if (
        mapping(report.get("artifact_odds_rejection_reason_counts"))
        and mapping(report.get("rejected_live_odds_candidate_reason_counts"))
        and "rejected_live_odds_candidate_count" in report
        and "rows_with_rejected_live_odds_candidates" in report
    ):
        return dict(report)
    artifact_reason_counts: Counter[str] = Counter()
    rejected_candidate_reason_counts: Counter[str] = Counter()
    rejected_candidate_count = 0
    rows_with_rejected_candidates = 0
    for item in report.get("reports") or []:
        if not isinstance(item, Mapping):
            continue
        output_dir = resolve_repo_path(item.get("output_dir"))
        if output_dir is None:
            continue
        report_path = output_dir / "unified_evidence_dataset_report.json"
        if not report_path.exists():
            continue
        child_report = load_json(report_path)
        if child_report:
            artifact_reason_counts.update(
                artifact_odds_rejection_reason_counts(child_report)
            )
            rejected_candidate_reason_counts.update(
                rejected_live_odds_candidate_reason_counts(child_report)
            )
            rejected_candidate_count += finite_int(
                child_report.get("rejected_live_odds_candidate_count")
            )
            rows_with_rejected_candidates += finite_int(
                child_report.get("rows_with_rejected_live_odds_candidates")
            )
    if (
        not artifact_reason_counts
        and not rejected_candidate_reason_counts
        and rejected_candidate_count <= 0
        and rows_with_rejected_candidates <= 0
    ):
        return dict(report)
    enriched = dict(report)
    if not mapping(enriched.get("artifact_odds_rejection_reason_counts")):
        enriched["artifact_odds_rejection_reason_counts"] = dict(
            sorted(artifact_reason_counts.items())
        )
    if not mapping(enriched.get("rejected_live_odds_candidate_reason_counts")):
        enriched["rejected_live_odds_candidate_reason_counts"] = dict(
            sorted(rejected_candidate_reason_counts.items())
        )
    if "rejected_live_odds_candidate_count" not in enriched:
        enriched["rejected_live_odds_candidate_count"] = rejected_candidate_count
    if "rows_with_rejected_live_odds_candidates" not in enriched:
        enriched[
            "rows_with_rejected_live_odds_candidates"
        ] = rows_with_rejected_candidates
    return enriched


def is_market_only_candidate(candidate_key: Any) -> bool:
    return str(candidate_key or "") in MARKET_ONLY_CANDIDATE_KEYS


def gate_contract_thresholds(
    thresholds: AccuracyGateThresholds,
) -> gate_contract_audit.GateAuditThresholds:
    return gate_contract_audit.GateAuditThresholds(
        min_safe_joined_races=thresholds.min_safe_joined_races,
        current_min_top1_delta=thresholds.min_top1_delta,
        min_market_top1_delta=0.0,
        min_top3_delta=thresholds.min_top3_delta,
        max_mean_winner_rank_delta=thresholds.max_mean_winner_rank_delta,
        max_brier_delta=thresholds.max_brier_delta,
        max_logloss_delta=thresholds.max_logloss_delta,
        max_calibration_distance_delta=thresholds.max_calibration_distance_delta,
        max_box1_top_pick_share=thresholds.max_box1_top_pick_share,
        max_box1_share_delta=thresholds.max_box1_share_delta,
        max_probability_sum_error=thresholds.max_probability_sum_error,
    )


def rolling_gate_contract_selection(
    rolling_report: Mapping[str, Any],
    thresholds: AccuracyGateThresholds,
) -> dict[str, Any] | None:
    report = mapping(rolling_report)
    if not mapping(report.get("candidate_metrics_by_key")):
        return None
    audit = gate_contract_audit.build_report(
        rolling_report=report,
        thresholds=gate_contract_thresholds(thresholds),
    )
    if audit.get("final_status") != "REPORT_ONLY_GATE_CHANGE_CANDIDATE":
        return None
    policy = next(
        (
            row
            for row in audit.get("policy_summaries") or []
            if mapping(row).get("policy_key") == ODDS_AUGMENTED_GATE_CONTRACT_POLICY
        ),
        None,
    )
    policy = mapping(policy)
    if policy.get("status") != "PASS" or not policy.get("selected_candidate"):
        return None
    candidate_key = str(policy["selected_candidate"])
    by_key = mapping(report.get("candidate_metrics_by_key"))
    candidate = mapping(by_key.get(candidate_key))
    if not candidate:
        return None
    market = mapping(report.get("market_metrics")) or mapping(
        by_key.get("market_only_implied")
    )
    selected_row = next(
        (
            row
            for row in audit.get("candidate_gate_matrix") or []
            if mapping(row).get("candidate_key") == candidate_key
        ),
        {},
    )
    selected_row = mapping(selected_row)
    blocker_text = str(
        selected_row.get(f"{ODDS_AUGMENTED_GATE_CONTRACT_POLICY}_blockers") or ""
    )
    blockers = [item for item in blocker_text.split(";") if item]
    if blockers:
        return None
    gate = {
        "schema_version": "high_accuracy_candidate_gate_v2",
        "stage": "odds_augmented_model_research",
        "status": "PASS",
        "blockers": [],
        "candidate_race_count": race_count(candidate),
        "baseline_key": "market_only_implied",
        "baseline_metrics": dict(market),
        "candidate_metrics": dict(candidate),
        "candidate_minus_baseline": {
            "top1": selected_row.get("market_top1_delta"),
            "top3": selected_row.get("market_top3_delta"),
            "mean_winner_rank": selected_row.get("market_mean_winner_rank_delta"),
            "brier": selected_row.get("market_brier_delta"),
            "logloss": selected_row.get("market_logloss_delta"),
            "box1_top_pick_share": selected_row.get(
                "market_box1_top_pick_share_delta"
            ),
            "calibration_distance": selected_row.get(
                "market_calibration_distance_delta"
            ),
        },
        "thresholds": asdict(thresholds),
        "gate_contract_policy": ODDS_AUGMENTED_GATE_CONTRACT_POLICY,
        "gate_contract_final_status": audit.get("final_status"),
        "gate_contract_policy_summary": dict(policy),
        "ev_metrics_used_for_promotion": False,
    }
    return {
        "candidate_key": candidate_key,
        "candidate_metrics": dict(candidate),
        "gate": gate,
        "audit_summary": {
            "final_status": audit.get("final_status"),
            "selected_policy": ODDS_AUGMENTED_GATE_CONTRACT_POLICY,
            "selected_candidate": candidate_key,
            "policy_summaries": audit.get("policy_summaries"),
        },
    }


def compact_candidate_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    source = mapping(metrics)
    calibration = mapping(source.get("calibration_slope_intercept"))
    return {
        "candidate_key": source.get("candidate_key"),
        "family": source.get("family"),
        "race_count": race_count(source),
        "top1": metric(source, "top1"),
        "top3": metric(source, "top3"),
        "mean_winner_rank": metric(source, "mean_winner_rank"),
        "brier": metric(source, "brier"),
        "logloss": metric(source, "logloss"),
        "box1_top_pick_share": metric(source, "box1_top_pick_share"),
        "calibration_status": calibration.get("status"),
        "calibration_slope": finite_float(calibration.get("slope")),
        "calibration_intercept": finite_float(calibration.get("intercept")),
    }


def rolling_model_comparison_summary(report: Mapping[str, Any]) -> dict[str, Any] | None:
    rolling_report = mapping(report)
    if not rolling_report:
        return None
    sample_races = metric_int(rolling_report, "sample_race_count")
    minimum_races = metric_int(
        rolling_report,
        "minimum_races_for_review",
    ) or MIN_COMPLETE_VALID_PREJUMP_ODDS_RACES
    sample_floor_value = rolling_report.get("sample_floor_met")
    sample_floor_met = (
        sample_floor_value
        if isinstance(sample_floor_value, bool)
        else sample_races >= minimum_races
    )
    rank_first_sort = string_list(rolling_report.get("rank_first_sort"))
    by_key = mapping(rolling_report.get("candidate_metrics_by_key"))
    summary_keys = list(
        dict.fromkeys(
            [
                *rank_first_sort[:5],
                str(rolling_report.get("best_candidate_key") or ""),
                str(rolling_report.get("best_non_baseline_candidate_key") or ""),
                "market_only_implied",
                "stage2_shadow_uncalibrated",
                "stage2_uncalibrated_market_blend_75",
            ]
        )
    )
    candidate_summaries = {
        key: compact_candidate_metrics(candidate)
        for key in summary_keys
        for candidate in [by_key.get(key)]
        if key and isinstance(candidate, Mapping)
    }
    return {
        "status": rolling_report.get("final_status"),
        "sample_scope": rolling_report.get("sample_scope"),
        "sample_race_count": sample_races,
        "minimum_races_for_review": minimum_races,
        "sample_floor_met": sample_floor_met,
        "races_needed_for_review": max(0, minimum_races - sample_races),
        "source_artifact_odds_rows_seen": metric_int(
            rolling_report, "source_artifact_odds_rows_seen"
        ),
        "source_artifact_odds_rows_accepted": metric_int(
            rolling_report, "source_artifact_odds_rows_accepted"
        ),
        "source_artifact_odds_rows_rejected": metric_int(
            rolling_report, "source_artifact_odds_rows_rejected"
        ),
        "source_artifact_odds_rejection_reason_counts": int_count_mapping(
            rolling_report.get("source_artifact_odds_rejection_reason_counts")
        ),
        "candidate_count": metric_int(rolling_report, "candidate_count"),
        "best_candidate_key": rolling_report.get("best_candidate_key"),
        "best_non_baseline_candidate_key": rolling_report.get(
            "best_non_baseline_candidate_key"
        ),
        "rank_first_sort": rank_first_sort[:10],
        "blockers": string_list(rolling_report.get("blockers")),
        "candidate_minus_baseline": dict(
            mapping(rolling_report.get("candidate_minus_baseline"))
        ),
        "candidate_metrics_by_key": candidate_summaries,
    }


def stage2_forward_metrics_from_rolling_report(
    report: Mapping[str, Any],
) -> dict[str, Any] | None:
    rolling_report = mapping(report)
    if not rolling_report:
        return None
    by_key = mapping(rolling_report.get("candidate_metrics_by_key"))
    baseline = mapping(
        by_key.get("primary_shadow") or rolling_report.get("baseline_metrics")
    )
    candidate = mapping(by_key.get("stage2_shadow"))
    if not baseline or not candidate:
        return None
    if str(baseline.get("status") or "") != "EVALUATED":
        return None
    if str(candidate.get("status") or "") != "EVALUATED":
        return None
    source_status = str(rolling_report.get("final_status") or "")
    sample_floor_met = rolling_report.get("sample_floor_met")
    ready = (
        source_status == ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW
        and sample_floor_met is True
    )
    return {
        "schema_version": "stage2_forward_joined_metrics_from_rolling_v1",
        "status": (
            STAGE2_FORWARD_SHADOW_READY_FOR_REVIEW
            if ready
            else STAGE2_FORWARD_SHADOW_COLLECTING
        ),
        "source_status": source_status,
        "source": STAGE2_FORWARD_METRICS_FROM_ROLLING,
        "source_sample_scope": rolling_report.get("sample_scope"),
        "source_sample_race_count": rolling_report.get("sample_race_count"),
        "source_sample_floor_met": sample_floor_met,
        "source_unified_evidence_reports": string_list(
            rolling_report.get("source_unified_evidence_reports")
        ),
        "baseline_forward_shadow_metrics": dict(baseline),
        "stage2_challenger_forward_shadow_metrics": dict(candidate),
        "stage2_candidate_key": "stage2_shadow",
        "tgr_enabled": False,
        "odds_or_ev_used": False,
        "derived_from_odds_augmented_report": True,
    }


def race_count(metrics: Mapping[str, Any]) -> int:
    return finite_int(
        metrics.get("safe_joined_race_count")
        or metrics.get("race_count")
        or metrics.get("holdout_races")
        or metrics.get("source_safe_exact_joined_race_count")
    )


def probability_sum_error(metrics: Mapping[str, Any]) -> float | None:
    direct = finite_float(metrics.get("probability_sum_max_error_joined_races"))
    if direct is not None:
        return direct
    report = metrics.get("probability_sum_error")
    if isinstance(report, Mapping):
        return finite_float(report.get("max_abs_error") or report.get("max_error"))
    return None


def calibration_distance(metrics: Mapping[str, Any]) -> float | None:
    calibration = mapping(metrics.get("calibration_slope_intercept") or metrics.get("slope_intercept"))
    slope = finite_float(calibration.get("slope"))
    intercept = finite_float(calibration.get("intercept"))
    if slope is None or intercept is None:
        return None
    return abs(slope - 1.0) + abs(intercept)


def metric_deltas(
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
) -> dict[str, float | None]:
    deltas: dict[str, float | None] = {
        "top1": None,
        "top3": None,
        "mean_winner_rank": None,
        "brier": None,
        "logloss": None,
        "box1_top_pick_share": None,
        "calibration_distance": None,
    }
    for key in ("top1", "top3", "mean_winner_rank", "brier", "logloss", "box1_top_pick_share"):
        baseline = metric(baseline_metrics, key)
        candidate = metric(candidate_metrics, key)
        if baseline is not None and candidate is not None:
            deltas[key] = candidate - baseline
    baseline_calibration = calibration_distance(baseline_metrics)
    candidate_calibration = calibration_distance(candidate_metrics)
    if baseline_calibration is not None and candidate_calibration is not None:
        deltas["calibration_distance"] = candidate_calibration - baseline_calibration
    return deltas


def gate_candidate(
    *,
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    thresholds: AccuracyGateThresholds,
    source_status: str | None,
    accepted_source_statuses: set[str],
    stage: str,
) -> dict[str, Any]:
    blockers: list[str] = []
    candidate_races = race_count(candidate_metrics)
    deltas = metric_deltas(baseline_metrics, candidate_metrics)
    candidate_box1 = metric(candidate_metrics, "box1_top_pick_share")
    baseline_box1 = metric(baseline_metrics, "box1_top_pick_share")
    candidate_sum_error = probability_sum_error(candidate_metrics)

    if source_status and source_status not in accepted_source_statuses:
        blockers.append(f"source_status_not_ready:{source_status}")
    if candidate_races < thresholds.min_safe_joined_races:
        blockers.append("candidate_race_sample_below_min")
    if deltas["top1"] is None:
        blockers.append("metric_missing:top1")
    elif deltas["top1"] < thresholds.min_top1_delta:
        blockers.append("rank_accuracy_top1_delta_below_min")
    if deltas["top3"] is None:
        blockers.append("metric_missing:top3")
    elif deltas["top3"] < thresholds.min_top3_delta:
        blockers.append("metric_regressed:top3")
    if deltas["mean_winner_rank"] is None:
        blockers.append("metric_missing:mean_winner_rank")
    elif deltas["mean_winner_rank"] > thresholds.max_mean_winner_rank_delta:
        blockers.append("metric_regressed:mean_winner_rank")
    if deltas["brier"] is None:
        blockers.append("metric_missing:brier")
    elif deltas["brier"] > thresholds.max_brier_delta:
        blockers.append("metric_regressed:brier")
    if deltas["logloss"] is None:
        blockers.append("metric_missing:logloss")
    elif deltas["logloss"] > thresholds.max_logloss_delta:
        blockers.append("metric_regressed:logloss")
    if deltas["calibration_distance"] is None:
        blockers.append("metric_missing:calibration_slope_intercept")
    elif deltas["calibration_distance"] > thresholds.max_calibration_distance_delta:
        blockers.append("metric_regressed:calibration_slope_intercept")
    if candidate_box1 is None:
        blockers.append("metric_missing:box1_top_pick_share_candidate")
    else:
        if candidate_box1 > thresholds.max_box1_top_pick_share:
            blockers.append("candidate_box1_top_pick_share_above_max")
        if baseline_box1 is not None and candidate_box1 - baseline_box1 > thresholds.max_box1_share_delta:
            blockers.append("metric_regressed:box1_top_pick_share")
    if candidate_sum_error is not None and candidate_sum_error > thresholds.max_probability_sum_error:
        blockers.append("candidate_probability_sum_error_failed")

    return {
        "schema_version": "high_accuracy_candidate_gate_v1",
        "stage": stage,
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": list(dict.fromkeys(blockers)),
        "candidate_race_count": candidate_races,
        "baseline_metrics": dict(baseline_metrics),
        "candidate_metrics": dict(candidate_metrics),
        "candidate_minus_baseline": deltas,
        "thresholds": asdict(thresholds),
        "ev_metrics_used_for_promotion": False,
    }


def no_write_guarantees() -> dict[str, bool]:
    return {
        "production_promotion": False,
        "registry_mutation": False,
        "production_pointer_update": False,
        "active_model_replacement": False,
        "model_artifact_overwrite": False,
        "production_prediction_write": False,
        "db_write": False,
        "label_write": False,
        "snapshot_rewrite": False,
        "canonical_schema_mutation": False,
        "tgr_enabled": False,
        "odds_used_for_shadow_scoring": False,
        "ev_or_betting_action": False,
        "direct_switch": False,
    }


def source_control_reasons(report: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if report.get("production_activation_allowed") not in (None, False):
        reasons.append("source_report_production_activation_not_blocked")
    no_write = mapping(report.get("no_write_guarantees"))
    for key in (
        "production_promotion",
        "registry_mutation",
        "production_pointer_update",
        "active_model_replacement",
        "db_write",
        "label_write",
        "tgr_enabled",
        "betting_or_ev_output",
        "ev_or_betting_action",
    ):
        if key in no_write and no_write.get(key) is not False:
            reasons.append(f"source_report_write_guard_not_blocked:{key}")
    return reasons


def calibration_stage(
    calibration_report: Mapping[str, Any],
    thresholds: AccuracyGateThresholds,
) -> dict[str, Any] | None:
    if not calibration_report:
        return None
    baseline = mapping(calibration_report.get("baseline_eval_metrics"))
    candidate = mapping(calibration_report.get("candidate_eval_metrics"))
    status = str(calibration_report.get("final_status") or "")
    gate = gate_candidate(
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        thresholds=thresholds,
        source_status=status,
        accepted_source_statuses={CALIBRATION_READY, "PASS", "RUN", "READY"},
        stage="stage_1_calibration_only",
    )
    source_reasons = source_control_reasons(calibration_report)
    if source_reasons:
        gate["blockers"] = list(dict.fromkeys(gate["blockers"] + source_reasons))
        gate["status"] = "BLOCKED"
    if gate["status"] == "PASS":
        stage_status = "STAGE1_CALIBRATION_PASSES_PR_ONLY_GATE"
    elif status == CALIBRATION_READY:
        stage_status = "STAGE1_CALIBRATION_READY_BUT_RANK_GATE_FAILED_RUN_MODEL_CHALLENGERS"
    else:
        stage_status = "STAGE1_CALIBRATION_BLOCKED_KEEP_BASELINE"
    return {
        "schema_version": "high_accuracy_stage_v1",
        "stage": "stage_1_calibration_only",
        "status": stage_status,
        "source_final_status": status,
        "safe_exact_joined_race_count": calibration_report.get("safe_exact_joined_race_count"),
        "activation_blockers_from_source": list(calibration_report.get("activation_blockers") or []),
        "source_control_reasons": source_reasons,
        "gate": gate,
    }


def stage2_model_stage(
    stage2_forward_metrics: Mapping[str, Any] | None,
    shadow_replay_metrics: Mapping[str, Any] | None,
    thresholds: AccuracyGateThresholds,
    stage2_predictions_path: Path | None = None,
    stage2_prediction_rows: int = 0,
) -> dict[str, Any] | None:
    if stage2_forward_metrics:
        baseline = mapping(stage2_forward_metrics.get("baseline_forward_shadow_metrics"))
        candidate = mapping(stage2_forward_metrics.get("stage2_challenger_forward_shadow_metrics"))
        source_status = str(stage2_forward_metrics.get("status") or "")
    elif shadow_replay_metrics:
        baseline = mapping(shadow_replay_metrics.get(MODEL_BASELINE_KEY))
        candidate = mapping(shadow_replay_metrics.get(MODEL_CANDIDATE_KEY))
        source_status = "PASS"
    else:
        if stage2_prediction_rows > 0:
            return {
                "schema_version": "high_accuracy_stage_v1",
                "stage": "stage_2_non_tgr_model_challenger",
                "status": STAGE2_PREDICTIONS_COLLECTED_METRICS_MISSING,
                "source_status": "PREDICTIONS_COLLECTED_JOINED_METRICS_MISSING",
                "candidate_key": MODEL_CANDIDATE_KEY,
                "baseline_key": MODEL_BASELINE_KEY,
                "tgr_enabled": False,
                "odds_or_ev_used": False,
                "stage2_predictions_path": relpath(stage2_predictions_path),
                "stage2_prediction_rows": stage2_prediction_rows,
                "gate": {
                    "schema_version": "accuracy_gate_v1",
                    "stage": "stage_2_non_tgr_model_challenger",
                    "status": "BLOCKED",
                    "blockers": ["stage2_forward_joined_metrics_missing"],
                },
            }
        return None
    gate = gate_candidate(
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        thresholds=thresholds,
        source_status=source_status,
        accepted_source_statuses={STAGE2_FORWARD_SHADOW_READY_FOR_REVIEW, "PASS", "READY"},
        stage="stage_2_non_tgr_model_challenger",
    )
    status = (
        "STAGE2_NON_TGR_MODEL_CHALLENGER_PASSES_PR_ONLY_GATE"
        if gate["status"] == "PASS"
        else "STAGE2_NON_TGR_MODEL_CHALLENGER_BLOCKED_KEEP_SHADOW"
    )
    return {
        "schema_version": "high_accuracy_stage_v1",
        "stage": "stage_2_non_tgr_model_challenger",
        "status": status,
        "source_status": source_status,
        "candidate_key": MODEL_CANDIDATE_KEY,
        "baseline_key": MODEL_BASELINE_KEY,
        "tgr_enabled": False,
        "odds_or_ev_used": False,
        "stage2_predictions_path": relpath(stage2_predictions_path),
        "stage2_prediction_rows": stage2_prediction_rows,
        "gate": gate,
    }


def odds_augmented_stage(
    odds_gate_report: Mapping[str, Any] | None,
    odds_augmented_report: Mapping[str, Any] | None,
    thresholds: AccuracyGateThresholds,
) -> dict[str, Any] | None:
    if not odds_gate_report and not odds_augmented_report:
        return None
    odds_gate_status = str(mapping(odds_gate_report).get("status") or "")
    source_status = str(mapping(odds_augmented_report).get("final_status") or "")
    baseline = mapping(odds_augmented_report).get("baseline_metrics") or mapping(
        odds_augmented_report
    ).get("stage2_no_odds_metrics")
    candidate = mapping(odds_augmented_report).get("candidate_metrics") or mapping(
        odds_augmented_report
    ).get("best_rank_accuracy_candidate_metrics")
    if not isinstance(baseline, Mapping):
        baseline = {}
    if not isinstance(candidate, Mapping):
        candidate = {}
    candidate_key = candidate.get("candidate_key") or mapping(odds_augmented_report).get(
        "best_candidate_key"
    )
    if candidate_key and not candidate.get("candidate_key"):
        candidate = {**dict(candidate), "candidate_key": candidate_key}
    gate = gate_candidate(
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        thresholds=thresholds,
        source_status=source_status,
        accepted_source_statuses={
            ODDS_AUGMENTED_MODEL_READY_FOR_PR_REVIEW,
            ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW,
            "PASS",
            "READY",
        },
        stage="odds_augmented_model_research",
    )
    rank_first_gate = dict(gate)
    gate_contract_selection = rolling_gate_contract_selection(
        mapping(odds_augmented_report),
        thresholds,
    )
    gate_contract_selection_used = False
    if gate["status"] != "PASS" and gate_contract_selection:
        candidate = mapping(gate_contract_selection.get("candidate_metrics"))
        candidate_key = gate_contract_selection.get("candidate_key") or candidate_key
        gate = dict(mapping(gate_contract_selection.get("gate")))
        gate["rank_first_gate_status"] = rank_first_gate.get("status")
        gate["rank_first_gate_blockers"] = rank_first_gate.get("blockers")
        gate_contract_selection_used = True
    blockers = list(gate.get("blockers") or [])
    rolling_report = mapping(odds_augmented_report)
    rolling_summary = rolling_model_comparison_summary(rolling_report)
    rank_first_sort = string_list(rolling_report.get("rank_first_sort"))
    best_candidate_key = rolling_report.get("best_candidate_key")
    best_non_baseline_candidate_key = rolling_report.get("best_non_baseline_candidate_key")
    if is_market_only_candidate(candidate_key):
        blockers.append("market_only_candidate_not_promotable")
    if is_market_only_candidate(best_candidate_key):
        blockers.append("rolling_best_candidate_is_market_only")
    if is_market_only_candidate(best_non_baseline_candidate_key):
        blockers.append("rolling_best_non_baseline_candidate_is_market_only")
    if rank_first_sort and is_market_only_candidate(rank_first_sort[0]):
        blockers.append("rolling_rank_first_candidate_is_market_only")
    cumulative_sample_races = metric_int(rolling_report, "sample_race_count")
    cumulative_sample_scope = str(rolling_report.get("sample_scope") or "")
    cumulative_status = str(rolling_report.get("final_status") or "")
    cumulative_evidence_ready = (
        cumulative_status == ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW
        and cumulative_sample_scope == "unified"
        and cumulative_sample_races >= MIN_COMPLETE_VALID_PREJUMP_ODDS_RACES
    )
    odds_evidence_source = "latest_odds_research_gate"
    if odds_gate_status != ODDS_RESEARCH_READY_REPORT_ONLY:
        if cumulative_evidence_ready:
            odds_evidence_source = "cumulative_rolling_model_comparison"
        else:
            blockers.append(f"odds_research_gate_not_ready:{odds_gate_status or 'missing'}")
            if rolling_report:
                if cumulative_status != ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW:
                    blockers.append(
                        "cumulative_odds_evidence_status_not_ready:"
                        f"{cumulative_status or 'missing'}"
                    )
                if cumulative_sample_scope != "unified":
                    blockers.append(
                        "cumulative_odds_evidence_scope_not_unified:"
                        f"{cumulative_sample_scope or 'missing'}"
                    )
                if cumulative_sample_races < MIN_COMPLETE_VALID_PREJUMP_ODDS_RACES:
                    blockers.append("cumulative_odds_evidence_races_below_min")
    else:
        if metric_int(mapping(odds_gate_report), "complete_valid_prejump_odds_races") < MIN_COMPLETE_VALID_PREJUMP_ODDS_RACES:
            blockers.append("odds_research_gate_complete_valid_races_below_min")
        if metric_int(mapping(odds_gate_report), "source_url_rows_missing") > 0:
            blockers.append("odds_research_gate_source_url_coverage_not_100_pct")
        source_url_coverage = metric(mapping(odds_gate_report), "source_url_coverage_pct")
        if source_url_coverage is not None and source_url_coverage < 100.0:
            blockers.append("odds_research_gate_source_url_coverage_not_100_pct")
    if mapping(odds_augmented_report).get("ev_improved") is True and gate["status"] != "PASS":
        blockers.append("ev_improvement_ignored_because_accuracy_guardrails_failed")
    gate["blockers"] = list(dict.fromkeys(blockers))
    gate["status"] = "PASS" if not gate["blockers"] else "BLOCKED"
    return {
        "schema_version": "high_accuracy_stage_v1",
        "stage": "odds_augmented_model_research",
        "status": (
            ODDS_AUGMENTED_MODEL_READY_FOR_PR_REVIEW
            if gate["status"] == "PASS"
            else ODDS_AUGMENTED_MODEL_BLOCKED
        ),
        "candidate_key": candidate_key or None,
        "odds_research_gate_status": odds_gate_status or None,
        "source_final_status": source_status or None,
        "odds_evidence_source": odds_evidence_source,
        "cumulative_odds_evidence": {
            "status": cumulative_status or None,
            "sample_scope": cumulative_sample_scope or None,
            "sample_race_count": cumulative_sample_races,
            "minimum_complete_valid_prejump_odds_races": MIN_COMPLETE_VALID_PREJUMP_ODDS_RACES,
            "sample_floor_met": (
                rolling_summary.get("sample_floor_met") if rolling_summary else None
            ),
            "races_needed_for_review": (
                rolling_summary.get("races_needed_for_review")
                if rolling_summary
                else max(
                    0,
                    MIN_COMPLETE_VALID_PREJUMP_ODDS_RACES - cumulative_sample_races,
                )
            ),
            "ready": cumulative_evidence_ready,
        },
        "rolling_model_comparison": rolling_summary,
        "gate_contract_selection": (
            gate_contract_selection.get("audit_summary")
            if gate_contract_selection_used
            else None
        ),
        "ev_metrics_used_for_promotion": False,
        "ev_can_override_accuracy_gate": False,
        "gate": gate,
    }


def unified_evidence_summary(
    unified_evidence_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    report = mapping(unified_evidence_report)
    if not report:
        return {
            "status": "NOT_RUN",
            "unified_evidence_eligible_rows": 0,
            "rows_with_official_results": 0,
            "rows_with_strict_prejump_odds": 0,
            "rows_with_artifact_shadow_odds": 0,
            "rows_with_artifact_shadow_odds_candidates": 0,
            "artifact_shadow_odds_candidate_count": 0,
            "artifact_shadow_odds_selected_bucket_count": 0,
            "artifact_odds_rows_seen": 0,
            "artifact_odds_rows_accepted": 0,
            "artifact_odds_rows_rejected": 0,
            "artifact_odds_rejection_reason_counts": {},
            "exclusion_reason_counts": {},
            "odds_exclusion_reason_counts": {},
            "official_result_evidence_db_missing_race_ids": [],
            "official_result_evidence_db_requested_race_count": 0,
            "official_result_evidence_db_races_with_rows": [],
            "official_result_runner_paths": [],
            "official_result_coverage": unified_official_result_coverage_summary({}, {}),
            "rejected_live_odds_candidate_count": 0,
            "rows_with_rejected_live_odds_candidates": 0,
            "rejected_live_odds_candidate_reason_counts": {},
            "rows_with_stage2_predictions": 0,
            "blockers": ["unified_evidence_report_missing"],
        }
    eligible_rows = finite_int(report.get("unified_evidence_eligible_rows"))
    blockers: list[str] = []
    if report.get("final_status") != "UNIFIED_EVIDENCE_DATASET_BUILT":
        blockers.append(f"unified_evidence_status_not_built:{report.get('final_status')}")
    if eligible_rows < MIN_UNIFIED_EVIDENCE_ELIGIBLE_ROWS:
        blockers.append("unified_evidence_eligible_rows_below_review_floor")
    no_write = mapping(report.get("no_write_guarantees"))
    for key in (
        "production_promotion",
        "registry_mutation",
        "production_pointer_update",
        "active_model_replacement",
        "db_write",
        "label_write",
        "odds_write",
        "betting_or_ev_action",
        "snapshot_rewrite",
        "manifest_rewrite",
        "tgr_enabled",
    ):
        if key in no_write and no_write.get(key) is not False:
            blockers.append(f"unified_evidence_write_guard_not_blocked:{key}")
    official_audit = mapping(report.get("official_result_evidence_db_audit"))
    official_result_coverage = unified_official_result_coverage_summary(
        report,
        official_audit,
    )
    return {
        "status": "UNIFIED_EVIDENCE_READY_FOR_REVIEW" if not blockers else "UNIFIED_EVIDENCE_COLLECTING",
        "final_status": report.get("final_status"),
        "row_count": finite_int(report.get("row_count")),
        "race_count": finite_int(report.get("race_count")),
        "unified_evidence_eligible_rows": eligible_rows,
        "stage2_evaluation_eligible_rows": finite_int(
            report.get("stage2_evaluation_eligible_rows")
        ),
        "odds_evaluation_eligible_rows": finite_int(
            report.get("odds_evaluation_eligible_rows")
        ),
        "label_evaluation_eligible_rows": finite_int(
            report.get("label_evaluation_eligible_rows")
        ),
        "rows_with_official_results": finite_int(report.get("rows_with_official_results")),
        "rows_with_strict_prejump_odds": finite_int(
            report.get("rows_with_strict_prejump_odds")
        ),
        "rows_with_artifact_shadow_odds": finite_int(
            report.get("rows_with_artifact_shadow_odds")
        ),
        "rows_with_artifact_shadow_odds_candidates": finite_int(
            report.get("rows_with_artifact_shadow_odds_candidates")
        ),
        "artifact_shadow_odds_candidate_count": finite_int(
            report.get("artifact_shadow_odds_candidate_count")
        ),
        "artifact_shadow_odds_selected_bucket_count": finite_int(
            report.get("artifact_shadow_odds_selected_bucket_count")
        ),
        "artifact_odds_rows_seen": finite_int(report.get("artifact_odds_rows_seen")),
        "artifact_odds_rows_accepted": finite_int(
            report.get("artifact_odds_rows_accepted")
        ),
        "artifact_odds_rows_rejected": finite_int(
            report.get("artifact_odds_rows_rejected")
        ),
        "artifact_odds_rejection_reason_counts": (
            artifact_odds_rejection_reason_counts(report)
        ),
        "exclusion_reason_counts": int_count_mapping(
            report.get("exclusion_reason_counts")
        ),
        "odds_exclusion_reason_counts": int_count_mapping(
            report.get("odds_exclusion_reason_counts")
        ),
        "official_result_evidence_db_missing_race_ids": string_list(
            official_audit.get("missing_race_ids")
        ),
        "official_result_evidence_db_requested_race_count": finite_int(
            official_audit.get("race_ids_requested")
        ),
        "official_result_evidence_db_races_with_rows": string_list(
            official_audit.get("race_ids_with_rows")
        ),
        "official_result_runner_paths": string_list(
            report.get("official_result_runner_paths")
        ),
        "official_result_coverage": official_result_coverage,
        "rejected_live_odds_candidate_count": finite_int(
            report.get("rejected_live_odds_candidate_count")
        ),
        "rows_with_rejected_live_odds_candidates": finite_int(
            report.get("rows_with_rejected_live_odds_candidates")
        ),
        "rejected_live_odds_candidate_reason_counts": (
            rejected_live_odds_candidate_reason_counts(report)
        ),
        "rows_with_stage2_predictions": finite_int(
            report.get("rows_with_stage2_predictions")
        ),
        "minimum_eligible_rows_for_review": MIN_UNIFIED_EVIDENCE_ELIGIBLE_ROWS,
        "blockers": blockers,
        "no_write_guarantees": mapping(report.get("no_write_guarantees")),
    }


def backlog_unified_evidence_summary(
    backlog_unified_evidence_status: Mapping[str, Any] | None,
) -> dict[str, Any]:
    report = mapping(backlog_unified_evidence_status)
    if not report:
        return {
            "status": "NOT_RUN",
            "aggregation_scope": None,
            "dataset_count": 0,
            "failed_dataset_count": 0,
            "unified_evidence_eligible_rows": 0,
            "rows_with_artifact_shadow_odds": 0,
            "rows_with_artifact_shadow_odds_candidates": 0,
            "artifact_shadow_odds_candidate_count": 0,
            "artifact_shadow_odds_selected_bucket_count": 0,
            "artifact_odds_rows_seen": 0,
            "artifact_odds_rows_accepted": 0,
            "artifact_odds_rows_rejected": 0,
            "artifact_odds_rejection_reason_counts": {},
            "exclusion_reason_counts": {},
            "odds_exclusion_reason_counts": {},
            "official_result_evidence_db_missing_race_ids": [],
            "official_result_evidence_db_requested_race_count": 0,
            "official_result_evidence_db_races_with_rows": [],
            "official_result_coverage": {
                "source": "backlog_unified_evidence_status_missing",
                "requested_race_count": 0,
                "requested_race_count_source": None,
                "requested_race_ids": [],
                "legacy_requested_race_count_without_ids": 0,
                "races_with_rows_count": 0,
                "missing_race_count": 0,
                "missing_race_ids": [],
                "races_with_rows": [],
                "runner_path_count": 0,
                "runner_paths_source_field": None,
                "missing_exclusion_count": 0,
            },
            "rejected_live_odds_candidate_count": 0,
            "rows_with_rejected_live_odds_candidates": 0,
            "rejected_live_odds_candidate_reason_counts": {},
            "sample_blocking_gap_count": 0,
            "gap_action_counts": {},
            "gap_evidence_missing_reason_counts": {},
            "top_gap_race_ids": [],
            "top_gap_races": [],
            "top_official_result_missing_race_ids": [],
            "top_official_result_missing_races": [],
            "minimum_eligible_rows_for_review": MIN_UNIFIED_EVIDENCE_ELIGIBLE_ROWS,
            "blockers": ["backlog_unified_evidence_status_missing"],
        }

    eligible_rows = finite_int(report.get("unified_evidence_eligible_rows"))
    failed_dataset_count = finite_int(report.get("failed_dataset_count"))
    blockers: list[str] = []
    if report.get("status") not in ACCEPTED_UNIFIED_EVIDENCE_AGGREGATE_STATUSES:
        blockers.append(f"backlog_unified_evidence_status_not_built:{report.get('status')}")
    if failed_dataset_count > 0:
        blockers.append("backlog_unified_evidence_failed_datasets_present")
    if eligible_rows < MIN_UNIFIED_EVIDENCE_ELIGIBLE_ROWS:
        blockers.append("backlog_unified_evidence_eligible_rows_below_review_floor")
    no_write = mapping(report.get("no_write_guarantees"))
    for key in (
        "production_promotion",
        "registry_mutation",
        "production_pointer_update",
        "active_model_replacement",
        "db_write",
        "label_write",
        "odds_write",
        "betting_or_ev_action",
        "snapshot_rewrite",
        "manifest_rewrite",
        "tgr_enabled",
    ):
        if key in no_write and no_write.get(key) is not False:
            blockers.append(f"backlog_unified_evidence_write_guard_not_blocked:{key}")
    official_result_coverage = unified_official_result_coverage_summary(
        report,
        {
            "race_ids_requested": report.get(
                "official_result_evidence_db_requested_race_count"
            ),
            "missing_race_ids": report.get(
                "official_result_evidence_db_missing_race_ids"
            ),
            "race_ids_with_rows": report.get(
                "official_result_evidence_db_races_with_rows"
            ),
        },
    )

    return {
        "status": (
            "BACKLOG_UNIFIED_EVIDENCE_READY_FOR_REVIEW"
            if not blockers
            else "BACKLOG_UNIFIED_EVIDENCE_COLLECTING"
        ),
        "source_status": report.get("status"),
        "aggregation_scope": report.get("aggregation_scope"),
        "attempted_dataset_count": finite_int(report.get("attempted_dataset_count")),
        "dataset_count": finite_int(report.get("dataset_count")),
        "failed_dataset_count": failed_dataset_count,
        "row_count": finite_int(report.get("row_count")),
        "race_count": finite_int(report.get("race_count")),
        "rows_with_official_results": finite_int(report.get("rows_with_official_results")),
        "rows_with_strict_prejump_odds": finite_int(
            report.get("rows_with_strict_prejump_odds")
        ),
        "rows_with_artifact_shadow_odds": finite_int(
            report.get("rows_with_artifact_shadow_odds")
        ),
        "rows_with_artifact_shadow_odds_candidates": finite_int(
            report.get("rows_with_artifact_shadow_odds_candidates")
        ),
        "artifact_shadow_odds_candidate_count": finite_int(
            report.get("artifact_shadow_odds_candidate_count")
        ),
        "artifact_shadow_odds_selected_bucket_count": finite_int(
            report.get("artifact_shadow_odds_selected_bucket_count")
        ),
        "artifact_odds_rows_seen": finite_int(report.get("artifact_odds_rows_seen")),
        "artifact_odds_rows_accepted": finite_int(
            report.get("artifact_odds_rows_accepted")
        ),
        "artifact_odds_rows_rejected": finite_int(
            report.get("artifact_odds_rows_rejected")
        ),
        "artifact_odds_rejection_reason_counts": (
            artifact_odds_rejection_reason_counts(report)
        ),
        "exclusion_reason_counts": int_count_mapping(
            report.get("exclusion_reason_counts")
        ),
        "odds_exclusion_reason_counts": int_count_mapping(
            report.get("odds_exclusion_reason_counts")
        ),
        "official_result_evidence_db_missing_race_ids": string_list(
            report.get("official_result_evidence_db_missing_race_ids")
        ),
        "official_result_evidence_db_requested_race_count": finite_int(
            report.get("official_result_evidence_db_requested_race_count")
        ),
        "official_result_evidence_db_races_with_rows": string_list(
            report.get("official_result_evidence_db_races_with_rows")
        ),
        "official_result_coverage": official_result_coverage,
        "rejected_live_odds_candidate_count": finite_int(
            report.get("rejected_live_odds_candidate_count")
        ),
        "rows_with_rejected_live_odds_candidates": finite_int(
            report.get("rows_with_rejected_live_odds_candidates")
        ),
        "rejected_live_odds_candidate_reason_counts": (
            rejected_live_odds_candidate_reason_counts(report)
        ),
        **backlog_unified_gap_summary(report),
        "unified_evidence_eligible_rows": eligible_rows,
        "minimum_eligible_rows_for_review": MIN_UNIFIED_EVIDENCE_ELIGIBLE_ROWS,
        "blockers": blockers,
        "no_write_guarantees": no_write,
    }


def promotion_distance_official_result_coverage_summary(
    promotion_distance_report: Mapping[str, Any],
    rolling_sample: Mapping[str, Any],
) -> dict[str, Any]:
    direct = mapping(promotion_distance_report.get("official_result_coverage"))
    if direct:
        coverage = {
            "source": direct.get("source"),
            "requested_race_count": finite_int(direct.get("requested_race_count")),
            "races_with_rows_count": finite_int(direct.get("races_with_rows_count")),
            "missing_race_count": finite_int(direct.get("missing_race_count")),
            "missing_race_ids": string_list(direct.get("missing_race_ids")),
            "races_with_rows": string_list(direct.get("races_with_rows")),
            "runner_path_count": finite_int(direct.get("runner_path_count")),
            "runner_paths_source_field": direct.get("runner_paths_source_field"),
            "missing_exclusion_count": finite_int(
                direct.get("missing_exclusion_count")
            ),
        }
        if "requested_race_count_source" in direct:
            coverage["requested_race_count_source"] = direct.get(
                "requested_race_count_source"
            )
        if "requested_race_ids" in direct:
            coverage["requested_race_ids"] = string_list(direct.get("requested_race_ids"))
        if "legacy_requested_race_count_without_ids" in direct:
            coverage["legacy_requested_race_count_without_ids"] = finite_int(
                direct.get("legacy_requested_race_count_without_ids")
            )
        return coverage
    missing_race_ids = string_list(
        rolling_sample.get("source_official_result_evidence_db_missing_race_ids")
    )
    races_with_rows = string_list(
        rolling_sample.get("source_official_result_evidence_db_races_with_rows")
    )
    runner_paths = string_list(rolling_sample.get("source_official_result_runner_paths"))
    return {
        "source": "rolling_sample",
        "requested_race_count": finite_int(
            rolling_sample.get("source_official_result_evidence_db_requested_race_count")
        ),
        "requested_race_count_source": (
            "rolling_sample_source_requested_race_count"
        ),
        "requested_race_ids": string_list(
            rolling_sample.get("source_official_result_evidence_db_requested_race_ids")
        ),
        "legacy_requested_race_count_without_ids": finite_int(
            rolling_sample.get(
                "source_official_result_evidence_db_legacy_requested_race_count_without_ids"
            )
        ),
        "races_with_rows_count": len(races_with_rows),
        "missing_race_count": len(missing_race_ids),
        "missing_race_ids": missing_race_ids,
        "races_with_rows": races_with_rows,
        "runner_path_count": len(runner_paths),
        "runner_paths_source_field": "source_official_result_runner_paths",
        "missing_exclusion_count": int_count_mapping(
            rolling_sample.get("source_exclusion_reason_counts")
        ).get("official_result_missing", 0),
    }


def promotion_distance_summary(
    promotion_distance_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    report = mapping(promotion_distance_report)
    if not report:
        return {
            "status": "NOT_RUN",
            "promotion_ready": False,
            "blockers": ["promotion_distance_report_missing"],
        }
    market_benchmark = mapping(report.get("market_benchmark"))
    predeclared_residual = mapping(report.get("predeclared_residual_candidate"))
    rolling_sample = mapping(report.get("rolling_sample"))
    official_result_coverage = promotion_distance_official_result_coverage_summary(
        report,
        rolling_sample,
    )
    return {
        "status": report.get("final_status") or report.get("status"),
        "promotion_ready": bool(report.get("promotion_ready")),
        "blockers": string_list(report.get("blockers")),
        "sample_race_count": finite_int(rolling_sample.get("sample_race_count")),
        "sample_runner_rows": finite_int(rolling_sample.get("sample_runner_rows")),
        "minimum_races_for_review": finite_int(
            rolling_sample.get("minimum_races_for_review")
        ),
        "source_rejected_live_odds_candidate_count": finite_int(
            rolling_sample.get("source_rejected_live_odds_candidate_count")
        ),
        "source_rows_with_rejected_live_odds_candidates": finite_int(
            rolling_sample.get("source_rows_with_rejected_live_odds_candidates")
        ),
        "source_rejected_live_odds_candidate_reason_counts": {
            str(reason): finite_int(count)
            for reason, count in sorted(
                mapping(
                    rolling_sample.get(
                        "source_rejected_live_odds_candidate_reason_counts"
                    )
                ).items()
            )
        },
        "source_exclusion_reason_counts": int_count_mapping(
            rolling_sample.get("source_exclusion_reason_counts")
        ),
        "source_odds_exclusion_reason_counts": int_count_mapping(
            rolling_sample.get("source_odds_exclusion_reason_counts")
        ),
        "source_official_result_evidence_db_missing_race_ids": string_list(
            rolling_sample.get("source_official_result_evidence_db_missing_race_ids")
        ),
        "source_official_result_evidence_db_requested_race_ids": string_list(
            rolling_sample.get("source_official_result_evidence_db_requested_race_ids")
        ),
        "source_official_result_evidence_db_requested_race_count": finite_int(
            rolling_sample.get("source_official_result_evidence_db_requested_race_count")
        ),
        "source_official_result_evidence_db_legacy_requested_race_count_without_ids": finite_int(
            rolling_sample.get(
                "source_official_result_evidence_db_legacy_requested_race_count_without_ids"
            )
        ),
        "source_official_result_evidence_db_races_with_rows": string_list(
            rolling_sample.get("source_official_result_evidence_db_races_with_rows")
        ),
        "source_official_result_runner_paths": string_list(
            rolling_sample.get("source_official_result_runner_paths")
        ),
        "official_result_coverage": official_result_coverage,
        "best_candidate_key": market_benchmark.get("best_candidate_key"),
        "best_non_market_candidate_key": market_benchmark.get(
            "best_non_market_candidate_key"
        ),
        "target_top1_margin_vs_market": finite_float(
            market_benchmark.get("target_top1_margin_vs_market")
        ),
        "best_non_market_top1_margin_gap": finite_float(
            market_benchmark.get("best_non_market_top1_margin_gap")
        ),
        "best_non_market_minus_market": dict(
            mapping(market_benchmark.get("best_non_market_minus_market"))
        ),
        "predeclared_residual_candidate_key": predeclared_residual.get(
            "candidate_key"
        ),
        "predeclared_residual_status": predeclared_residual.get("status"),
        "predeclared_residual_triggered_race_count": finite_int(
            predeclared_residual.get("triggered_race_count")
        ),
        "predeclared_residual_minimum_triggered_races_for_directional_read": finite_int(
            predeclared_residual.get(
                "minimum_triggered_races_for_directional_read"
            )
        ),
        "predeclared_residual_triggered_races_needed_for_directional_read": finite_int(
            predeclared_residual.get("triggered_races_needed_for_directional_read")
        ),
        "predeclared_residual_directional_read_ready": bool(
            predeclared_residual.get("directional_read_ready")
        ),
        "predeclared_residual_candidate_minus_market": dict(
            mapping(predeclared_residual.get("candidate_minus_market"))
        ),
        "no_write_guarantees": mapping(report.get("no_write_guarantees")),
    }


def reserve_substitution_preflight_summary(
    reserve_substitution_preflight: Mapping[str, Any] | None,
) -> dict[str, Any]:
    report = mapping(reserve_substitution_preflight)
    if not report:
        return {
            "status": "NOT_RUN",
            "candidate_count": 0,
            "blocked_candidate_count": 0,
            "ready_for_policy_review_count": 0,
            "blocker_counts": {},
            "readiness_blocker_counts": {},
            "dataset_join_blocker_counts": {},
            "ready_race_ids": [],
            "blocked_race_ids": [],
            "candidates": [],
            "blockers": ["reserve_substitution_preflight_missing"],
        }

    readiness_counts: Counter[str] = Counter()
    dataset_join_counts: Counter[str] = Counter()
    ready_race_ids: list[str] = []
    blocked_race_ids: list[str] = []
    candidates: list[dict[str, Any]] = []
    for candidate in report.get("candidates") or []:
        if not isinstance(candidate, Mapping):
            continue
        race_id = str(candidate.get("race_id") or "")
        readiness_blockers = string_list(candidate.get("readiness_blockers"))
        dataset_join_blockers = string_list(candidate.get("dataset_join_blockers"))
        for blocker in readiness_blockers:
            readiness_counts[blocker] += 1
        for blocker in dataset_join_blockers:
            dataset_join_counts[blocker] += 1
        if candidate.get("preflight_status") == "READY_FOR_MANUAL_POLICY_REVIEW":
            if race_id:
                ready_race_ids.append(race_id)
        elif race_id:
            blocked_race_ids.append(race_id)
        candidates.append(
            {
                "race_id": candidate.get("race_id"),
                "preflight_status": candidate.get("preflight_status"),
                "policy_review_status": candidate.get("policy_review_status"),
                "acceptance_status": candidate.get("acceptance_status"),
                "acceptance_effect": candidate.get("acceptance_effect"),
                "candidate_reserve_boxes": candidate.get("candidate_reserve_boxes"),
                "scratched_participant_boxes": candidate.get(
                    "scratched_participant_boxes"
                ),
                "readiness_blockers": readiness_blockers,
                "dataset_join_blockers": dataset_join_blockers,
                "source_url": candidate.get("source_url"),
            }
        )

    no_write = mapping(report.get("no_write_guarantees"))
    blockers: list[str] = []
    for key in (
        "db_write",
        "label_write",
        "canonical_result_label_write",
        "official_result_acceptance",
        "quarantine_bypass",
        "snapshot_mutation",
        "manifest_mutation",
        "model_training",
        "registry_mutation",
        "production_promotion",
        "betting_action",
        "ev_action",
        "tgr_enabled",
    ):
        if key in no_write and no_write.get(key) is not False:
            blockers.append(f"reserve_substitution_preflight_write_guard_not_blocked:{key}")

    return {
        "status": report.get("final_status") or report.get("status"),
        "candidate_count": finite_int(report.get("candidate_count")),
        "blocked_candidate_count": finite_int(report.get("blocked_candidate_count")),
        "ready_for_policy_review_count": finite_int(
            report.get("ready_for_policy_review_count")
        ),
        "blocker_counts": int_count_mapping(report.get("blocker_counts")),
        "readiness_blocker_counts": dict(sorted(readiness_counts.items())),
        "dataset_join_blocker_counts": dict(sorted(dataset_join_counts.items())),
        "ready_race_ids": ready_race_ids,
        "blocked_race_ids": blocked_race_ids,
        "candidates": candidates,
        "no_write_guarantees": no_write,
        "blockers": blockers,
    }


def reserve_substitution_manual_review_summary(
    reserve_substitution_manual_review: Mapping[str, Any] | None,
) -> dict[str, Any]:
    report = mapping(reserve_substitution_manual_review)
    if not report:
        return {
            "status": "NOT_RUN",
            "candidate_count": 0,
            "ready_candidate_count": 0,
            "blocked_candidate_count": 0,
            "ready_race_ids": [],
            "blocked_race_ids": [],
            "mapping_pair_count": 0,
            "mapping_pairs_by_race": {},
            "automatic_acceptance_allowed": None,
            "dataset_join_allowed": None,
            "official_result_acceptance_allowed": None,
            "db_write": None,
            "blockers": ["reserve_substitution_manual_review_packet_missing"],
        }

    mapping_pairs_by_race: dict[str, list[Mapping[str, Any]]] = {}
    mapping_pair_count = 0
    candidate_blocker_counts: Counter[str] = Counter()
    for candidate in report.get("candidates") or []:
        if not isinstance(candidate, Mapping):
            continue
        race_id = str(candidate.get("race_id") or "")
        for blocker in string_list(candidate.get("packet_blockers")):
            candidate_blocker_counts[blocker] += 1
        hypothesis = mapping(candidate.get("mapping_hypothesis"))
        pairs = [
            pair
            for pair in (hypothesis.get("pairs") or [])
            if isinstance(pair, Mapping)
        ]
        if race_id:
            mapping_pairs_by_race[race_id] = pairs
        mapping_pair_count += len(pairs)

    blockers: list[str] = []
    for key in (
        "automatic_acceptance_allowed",
        "dataset_join_allowed",
        "official_result_acceptance_allowed",
        "db_write",
    ):
        if key in report and report.get(key) is not False:
            blockers.append(f"reserve_substitution_manual_review_guard_not_blocked:{key}")
    no_write = mapping(report.get("no_write_guarantees"))
    for key in (
        "db_write",
        "label_write",
        "canonical_result_label_write",
        "official_result_acceptance",
        "quarantine_bypass",
        "snapshot_mutation",
        "manifest_mutation",
        "model_training",
        "registry_mutation",
        "production_promotion",
        "betting_action",
        "ev_action",
        "tgr_enabled",
    ):
        if key in no_write and no_write.get(key) is not False:
            blockers.append(f"reserve_substitution_manual_review_write_guard_not_blocked:{key}")

    return {
        "status": report.get("final_status") or report.get("status"),
        "candidate_count": finite_int(report.get("candidate_count")),
        "ready_candidate_count": finite_int(report.get("ready_candidate_count")),
        "blocked_candidate_count": finite_int(report.get("blocked_candidate_count")),
        "ready_race_ids": string_list(report.get("ready_race_ids")),
        "blocked_race_ids": string_list(report.get("blocked_race_ids")),
        "mapping_pair_count": mapping_pair_count,
        "mapping_pairs_by_race": mapping_pairs_by_race,
        "candidate_blocker_counts": dict(sorted(candidate_blocker_counts.items())),
        "automatic_acceptance_allowed": report.get("automatic_acceptance_allowed"),
        "dataset_join_allowed": report.get("dataset_join_allowed"),
        "official_result_acceptance_allowed": report.get(
            "official_result_acceptance_allowed"
        ),
        "db_write": report.get("db_write"),
        "no_write_guarantees": no_write,
        "blockers": blockers,
    }


def reserve_substitution_policy_impact_summary(
    reserve_substitution_policy_impact: Mapping[str, Any] | None,
) -> dict[str, Any]:
    report = mapping(reserve_substitution_policy_impact)
    if not report:
        return {
            "status": "NOT_RUN",
            "candidate_count": 0,
            "ready_candidate_count": 0,
            "ready_race_ids": [],
            "mapping_pair_count": 0,
            "potential_official_result_runner_rows_blocked_by_policy": 0,
            "matched_backlog_top_gap_race_count": 0,
            "matched_backlog_top_gap_race_ids": [],
            "backlog_sample_blocking_gap_count": 0,
            "dataset_join_allowed": None,
            "official_result_acceptance_allowed": None,
            "db_write": None,
            "blockers": ["reserve_substitution_policy_impact_preview_missing"],
        }

    blockers: list[str] = []
    for key in (
        "approval_required",
        "automatic_acceptance_allowed",
        "dataset_join_allowed",
        "official_result_acceptance_allowed",
        "db_write",
    ):
        expected = True if key == "approval_required" else False
        if key in report and report.get(key) is not expected:
            blockers.append(f"reserve_substitution_policy_impact_guard_unexpected:{key}")
    no_write = mapping(report.get("no_write_guarantees"))
    for key in (
        "db_write",
        "label_write",
        "canonical_result_label_write",
        "official_result_acceptance",
        "quarantine_bypass",
        "snapshot_mutation",
        "manifest_mutation",
        "model_training",
        "registry_mutation",
        "production_promotion",
        "betting_action",
        "ev_action",
        "tgr_enabled",
    ):
        if key in no_write and no_write.get(key) is not False:
            blockers.append(f"reserve_substitution_policy_impact_write_guard_not_blocked:{key}")

    return {
        "status": report.get("final_status") or report.get("status"),
        "candidate_count": finite_int(report.get("candidate_count")),
        "ready_candidate_count": finite_int(report.get("ready_candidate_count")),
        "ready_race_ids": string_list(report.get("ready_race_ids")),
        "mapping_pair_count": finite_int(report.get("mapping_pair_count")),
        "potential_official_result_runner_rows_blocked_by_policy": finite_int(
            report.get("potential_official_result_runner_rows_blocked_by_policy")
        ),
        "matched_backlog_top_gap_race_count": finite_int(
            report.get("matched_backlog_top_gap_race_count")
        ),
        "matched_backlog_top_gap_race_ids": string_list(
            report.get("matched_backlog_top_gap_race_ids")
        ),
        "backlog_sample_blocking_gap_count": finite_int(
            report.get("backlog_sample_blocking_gap_count")
        ),
        "backlog_gap_action_counts": int_count_mapping(
            report.get("backlog_gap_action_counts")
        ),
        "backlog_evidence_missing_reason_counts": int_count_mapping(
            report.get("backlog_evidence_missing_reason_counts")
        ),
        "dataset_join_allowed": report.get("dataset_join_allowed"),
        "official_result_acceptance_allowed": report.get(
            "official_result_acceptance_allowed"
        ),
        "db_write": report.get("db_write"),
        "current_effect": report.get("current_effect"),
        "preview_effect_if_policy_approved_later": report.get(
            "preview_effect_if_policy_approved_later"
        ),
        "no_write_guarantees": no_write,
        "blockers": blockers,
    }


def promotion_pr_gate(
    *,
    stages: Sequence[Mapping[str, Any]],
    protected_paths_unchanged: bool,
    promotion_distance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    passing = [stage for stage in stages if mapping(stage.get("gate")).get("status") == "PASS"]
    blockers: list[str] = []
    if not passing:
        blockers.append("no_candidate_passed_rank_first_accuracy_gate")
    if not protected_paths_unchanged:
        blockers.append("protected_paths_changed")
    selected = passing[-1] if passing else None
    distance = mapping(promotion_distance)
    if (
        selected
        and selected.get("stage") == "odds_augmented_model_research"
        and distance
        and distance.get("status") != "NOT_RUN"
        and distance.get("promotion_ready") is not True
    ):
        blockers.append(f"promotion_distance_not_ready:{distance.get('status')}")
        blockers.extend(
            f"promotion_distance_blocker:{blocker}"
            for blocker in string_list(distance.get("blockers"))
        )
    ready = bool(selected and not blockers)
    return {
        "schema_version": "promotion_pr_only_gate_v1",
        "status": "READY_FOR_PR_DRAFT" if ready else "BLOCKED",
        "selected_stage": selected.get("stage") if selected else None,
        "selected_candidate": selected.get("candidate_key") if selected else None,
        "blockers": blockers,
        "pull_request_boundary": {
            "promotion_pr_allowed": ready,
            "direct_local_switch_allowed": False,
            "local_registry_mutation_allowed": False,
            "production_pointer_update_allowed": False,
            "requires_human_pr_review": True,
        },
    }


def build_packet(
    *,
    calibration_report: Mapping[str, Any] | None = None,
    stage2_forward_metrics: Mapping[str, Any] | None = None,
    shadow_replay_metrics: Mapping[str, Any] | None = None,
    odds_gate_report: Mapping[str, Any] | None = None,
    odds_augmented_report: Mapping[str, Any] | None = None,
    ev_diagnostics: Mapping[str, Any] | None = None,
    unified_evidence_report: Mapping[str, Any] | None = None,
    backlog_unified_evidence_status: Mapping[str, Any] | None = None,
    promotion_distance_report: Mapping[str, Any] | None = None,
    reserve_substitution_preflight: Mapping[str, Any] | None = None,
    reserve_substitution_manual_review: Mapping[str, Any] | None = None,
    reserve_substitution_policy_impact: Mapping[str, Any] | None = None,
    thresholds: AccuracyGateThresholds = AccuracyGateThresholds(),
    calibration_report_path: Path | None = None,
    stage2_forward_metrics_path: Path | None = None,
    stage2_predictions_path: Path | None = None,
    shadow_replay_metrics_path: Path | None = None,
    odds_gate_report_path: Path | None = None,
    odds_augmented_report_path: Path | None = None,
    ev_diagnostics_path: Path | None = None,
    unified_evidence_report_path: Path | None = None,
    backlog_unified_evidence_status_path: Path | None = None,
    promotion_distance_report_path: Path | None = None,
    reserve_substitution_preflight_path: Path | None = None,
    reserve_substitution_manual_review_path: Path | None = None,
    reserve_substitution_policy_impact_path: Path | None = None,
    timing_aligned_rerun_plan_path: Path | None = None,
    timing_aligned_rerun_execution_status_path: Path | None = None,
    output_dir: Path | None = None,
    generated_at: datetime | None = None,
    protected_before: Mapping[str, str | None] | None = None,
    protected_after: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    protected_before = dict(protected_before or protected_hashes())
    protected_after = dict(protected_after or protected_hashes())
    stage2_prediction_rows = count_jsonl_rows(stage2_predictions_path)
    stage2_forward_metrics = stage2_forward_metrics or stage2_forward_metrics_from_rolling_report(
        odds_augmented_report or {}
    )
    stages = {
        "calibration_only": calibration_stage(calibration_report or {}, thresholds),
        "non_tgr_model_challenger": stage2_model_stage(
            stage2_forward_metrics,
            shadow_replay_metrics,
            thresholds,
            stage2_predictions_path=stage2_predictions_path,
            stage2_prediction_rows=stage2_prediction_rows,
        ),
        "odds_augmented_model": odds_augmented_stage(
            odds_gate_report,
            odds_augmented_report,
            thresholds,
        ),
    }
    stage_list = [stage for stage in stages.values() if stage]
    promotion_distance = promotion_distance_summary(promotion_distance_report)
    pr_gate = promotion_pr_gate(
        stages=stage_list,
        protected_paths_unchanged=protected_before == protected_after,
        promotion_distance=promotion_distance,
    )
    if pr_gate["status"] == "READY_FOR_PR_DRAFT":
        final_status = FINAL_READY_FOR_PR
    elif stages["calibration_only"] and stages["calibration_only"].get("source_final_status") == CALIBRATION_READY:
        final_status = FINAL_NEEDS_MODEL_CHALLENGER
    else:
        final_status = FINAL_BLOCKED
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "final_status": final_status,
        "optimization_target": {
            "primary": "rank_accuracy_top1",
            "secondary": ["top3", "mean_winner_rank"],
            "guardrails": [
                "brier",
                "logloss",
                "calibration_slope_intercept",
                "box1_top_pick_share",
                "probability_sum_error",
                "exact_runner_box_identity",
                "no_tgr",
                "no_ev_override",
            ],
        },
        "thresholds": asdict(thresholds),
        "new_statuses": {
            "stage2_collecting": STAGE2_FORWARD_SHADOW_COLLECTING,
            "stage2_ready": STAGE2_FORWARD_SHADOW_READY_FOR_REVIEW,
            "stage2_predictions_collected_metrics_missing": (
                STAGE2_PREDICTIONS_COLLECTED_METRICS_MISSING
            ),
            "stage2_forward_metrics_from_rolling": STAGE2_FORWARD_METRICS_FROM_ROLLING,
            "odds_blocked": ODDS_RESEARCH_BLOCKED_PROVENANCE,
            "odds_ready_report_only": ODDS_RESEARCH_READY_REPORT_ONLY,
            "odds_augmented_blocked": ODDS_AUGMENTED_MODEL_BLOCKED,
            "odds_augmented_ready_for_pr_review": ODDS_AUGMENTED_MODEL_READY_FOR_PR_REVIEW,
        },
        "stages": stages,
        "promotion_pr_gate": pr_gate,
        "odds_research_gate_summary": {
            "status": mapping(odds_gate_report).get("status"),
            "complete_valid_prejump_odds_races": mapping(odds_gate_report).get(
                "complete_valid_prejump_odds_races"
            ),
            "odds_used_for_shadow_scoring": False,
        },
        "ev_diagnostics_summary": {
            "status": mapping(ev_diagnostics).get("status"),
            "ev_rows": mapping(ev_diagnostics).get("ev_rows"),
            "ev_metrics_used_for_promotion": False,
            "ev_can_override_accuracy_gate": False,
        },
        "unified_evidence_summary": unified_evidence_summary(unified_evidence_report),
        "backlog_unified_evidence_summary": backlog_unified_evidence_summary(
            backlog_unified_evidence_status
        ),
        "promotion_distance_summary": promotion_distance,
        "reserve_substitution_preflight_summary": reserve_substitution_preflight_summary(
            reserve_substitution_preflight
        ),
        "reserve_substitution_manual_review_summary": (
            reserve_substitution_manual_review_summary(
                reserve_substitution_manual_review
            )
        ),
        "reserve_substitution_policy_impact_summary": (
            reserve_substitution_policy_impact_summary(
                reserve_substitution_policy_impact
            )
        ),
        "source_artifacts": {
            "calibration_report": relpath(calibration_report_path),
            "stage2_forward_metrics": relpath(stage2_forward_metrics_path),
            "stage2_forward_metrics_source": (
                mapping(stage2_forward_metrics).get("source")
                if stage2_forward_metrics
                else None
            ),
            "stage2_predictions": relpath(stage2_predictions_path),
            "shadow_replay_metrics": relpath(shadow_replay_metrics_path),
            "odds_research_gate_report": relpath(odds_gate_report_path),
            "odds_augmented_challenger_report": relpath(odds_augmented_report_path),
            "report_only_ev_diagnostics": relpath(ev_diagnostics_path),
            "unified_evidence_report": relpath(unified_evidence_report_path),
            "backlog_unified_evidence_status": relpath(
                backlog_unified_evidence_status_path
            ),
            "promotion_distance_report": relpath(promotion_distance_report_path),
            "reserve_substitution_preflight": relpath(
                reserve_substitution_preflight_path
            ),
            "reserve_substitution_manual_review": relpath(
                reserve_substitution_manual_review_path
            ),
            "reserve_substitution_policy_impact_preview": relpath(
                reserve_substitution_policy_impact_path
            ),
            "timing_aligned_rerun_plan": relpath(timing_aligned_rerun_plan_path),
            "timing_aligned_rerun_execution_status": relpath(
                timing_aligned_rerun_execution_status_path
            ),
        },
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "protected_paths_unchanged": protected_before == protected_after,
        "no_write_guarantees": no_write_guarantees(),
        "output_dir": relpath(output_dir),
    }


def build_summary(packet: Mapping[str, Any]) -> str:
    pr_gate = mapping(packet.get("promotion_pr_gate"))
    stages = mapping(packet.get("stages"))
    stage2_stage = mapping(stages.get("non_tgr_model_challenger"))
    odds_stage = mapping(stages.get("odds_augmented_model"))
    stage2_gate = mapping(stage2_stage.get("gate"))
    cumulative_odds_evidence = mapping(odds_stage.get("cumulative_odds_evidence"))
    gate_contract_selection = mapping(odds_stage.get("gate_contract_selection"))
    unified_summary = mapping(packet.get("unified_evidence_summary"))
    backlog_summary = mapping(packet.get("backlog_unified_evidence_summary"))
    promotion_distance = mapping(packet.get("promotion_distance_summary"))
    unified_official_result = mapping(
        unified_summary.get("official_result_coverage")
    )
    promotion_official_result = mapping(
        promotion_distance.get("official_result_coverage")
    )
    reserve_preflight = mapping(packet.get("reserve_substitution_preflight_summary"))
    reserve_manual_review = mapping(
        packet.get("reserve_substitution_manual_review_summary")
    )
    reserve_policy_impact = mapping(
        packet.get("reserve_substitution_policy_impact_summary")
    )
    source_artifacts = mapping(packet.get("source_artifacts"))
    lines = [
        "# High Accuracy Refinement Packet",
        "",
        f"- Final status: `{packet.get('final_status')}`",
        f"- Promotion PR gate: `{pr_gate.get('status')}`",
        f"- Stage 2 status: `{stage2_stage.get('status') or 'NOT_RUN'}`",
        f"- Stage 2 prediction rows: `{stage2_stage.get('stage2_prediction_rows')}`",
        f"- Stage 2 predictions: `{stage2_stage.get('stage2_predictions_path')}`",
        f"- Stage 2 blockers: `{stage2_gate.get('blockers')}`",
        f"- Odds research gate: `{mapping(packet.get('odds_research_gate_summary')).get('status') or 'NOT_RUN'}`",
        f"- Odds augmented status: `{odds_stage.get('status') or 'NOT_RUN'}`",
        f"- Odds evidence source: `{odds_stage.get('odds_evidence_source') or 'NOT_RUN'}`",
        f"- Odds gate contract policy: `{gate_contract_selection.get('selected_policy')}`",
        f"- Odds gate contract selected candidate: `{gate_contract_selection.get('selected_candidate')}`",
        f"- Odds gate contract status: `{gate_contract_selection.get('final_status')}`",
        f"- Cumulative odds evidence ready: `{cumulative_odds_evidence.get('ready')}`",
        f"- Cumulative odds evidence status: `{cumulative_odds_evidence.get('status')}`",
        f"- Cumulative odds evidence sample scope: `{cumulative_odds_evidence.get('sample_scope')}`",
        f"- Cumulative odds evidence sample races: `{cumulative_odds_evidence.get('sample_race_count')}`",
        f"- Cumulative odds evidence races needed: `{cumulative_odds_evidence.get('races_needed_for_review')}`",
        f"- Cumulative source artifact odds rows seen: `{mapping(odds_stage.get('rolling_model_comparison')).get('source_artifact_odds_rows_seen')}`",
        f"- Cumulative source artifact odds rows accepted: `{mapping(odds_stage.get('rolling_model_comparison')).get('source_artifact_odds_rows_accepted')}`",
        f"- Cumulative source artifact odds rows rejected: `{mapping(odds_stage.get('rolling_model_comparison')).get('source_artifact_odds_rows_rejected')}`",
        f"- Cumulative source artifact odds rejection reasons: `{mapping(odds_stage.get('rolling_model_comparison')).get('source_artifact_odds_rejection_reason_counts')}`",
        f"- Unified evidence: `{unified_summary.get('status') or 'NOT_RUN'}`",
        f"- Unified eligible rows: `{unified_summary.get('unified_evidence_eligible_rows')}`",
        f"- Unified artifact odds accepted rows: `{unified_summary.get('artifact_odds_rows_accepted')}`",
        f"- Unified artifact odds rejection reasons: `{unified_summary.get('artifact_odds_rejection_reason_counts')}`",
        f"- Unified exclusion reasons: `{unified_summary.get('exclusion_reason_counts')}`",
        f"- Unified odds exclusion reasons: `{unified_summary.get('odds_exclusion_reason_counts')}`",
        f"- Unified official-result missing race IDs: `{unified_summary.get('official_result_evidence_db_missing_race_ids')}`",
        f"- Unified official-result runner path count: `{unified_official_result.get('runner_path_count')}`",
        f"- Unified official-result runner path source field: `{unified_official_result.get('runner_paths_source_field')}`",
        f"- Unified official-result coverage requested races: `{unified_official_result.get('requested_race_count')}`",
        f"- Unified official-result coverage requested race count source: `{unified_official_result.get('requested_race_count_source')}`",
        f"- Unified official-result coverage races with rows: `{unified_official_result.get('races_with_rows_count')}`",
        f"- Unified official-result coverage missing races: `{unified_official_result.get('missing_race_count')}`",
        f"- Unified official-result missing exclusion count: `{unified_official_result.get('missing_exclusion_count')}`",
        f"- Unified rejected live odds candidates: `{unified_summary.get('rejected_live_odds_candidate_count')}`",
        f"- Unified rows with rejected live odds candidates: `{unified_summary.get('rows_with_rejected_live_odds_candidates')}`",
        f"- Unified rejected live odds candidate reasons: `{unified_summary.get('rejected_live_odds_candidate_reason_counts')}`",
        f"- Backlog unified evidence: `{backlog_summary.get('status') or 'NOT_RUN'}`",
        f"- Backlog unified eligible rows: `{backlog_summary.get('unified_evidence_eligible_rows')}`",
        f"- Backlog artifact odds accepted rows: `{backlog_summary.get('artifact_odds_rows_accepted')}`",
        f"- Backlog artifact odds rejection reasons: `{backlog_summary.get('artifact_odds_rejection_reason_counts')}`",
        f"- Backlog exclusion reasons: `{backlog_summary.get('exclusion_reason_counts')}`",
        f"- Backlog odds exclusion reasons: `{backlog_summary.get('odds_exclusion_reason_counts')}`",
        f"- Backlog official-result coverage requested races: `{mapping(backlog_summary.get('official_result_coverage')).get('requested_race_count')}`",
        f"- Backlog official-result requested race count source: `{mapping(backlog_summary.get('official_result_coverage')).get('requested_race_count_source')}`",
        f"- Backlog official-result legacy requested race count without IDs: `{mapping(backlog_summary.get('official_result_coverage')).get('legacy_requested_race_count_without_ids')}`",
        f"- Backlog official-result coverage races with rows: `{mapping(backlog_summary.get('official_result_coverage')).get('races_with_rows_count')}`",
        f"- Backlog official-result coverage missing races: `{mapping(backlog_summary.get('official_result_coverage')).get('missing_race_count')}`",
        f"- Backlog official-result missing exclusion count: `{mapping(backlog_summary.get('official_result_coverage')).get('missing_exclusion_count')}`",
        f"- Backlog official-result missing race IDs: `{backlog_summary.get('official_result_evidence_db_missing_race_ids')}`",
        f"- Backlog rejected live odds candidates: `{backlog_summary.get('rejected_live_odds_candidate_count')}`",
        f"- Backlog rows with rejected live odds candidates: `{backlog_summary.get('rows_with_rejected_live_odds_candidates')}`",
        f"- Backlog rejected live odds candidate reasons: `{backlog_summary.get('rejected_live_odds_candidate_reason_counts')}`",
        f"- Backlog sample-blocking gap races: `{backlog_summary.get('sample_blocking_gap_count')}`",
        f"- Backlog gap actions: `{backlog_summary.get('gap_action_counts')}`",
        f"- Backlog evidence-missing reasons: `{backlog_summary.get('gap_evidence_missing_reason_counts')}`",
        f"- Backlog top gap race IDs: `{backlog_summary.get('top_gap_race_ids')}`",
        f"- Backlog top gap races: `{backlog_summary.get('top_gap_races')}`",
        f"- Backlog top official-result-missing race IDs: `{backlog_summary.get('top_official_result_missing_race_ids')}`",
        f"- Backlog top official-result-missing races: `{backlog_summary.get('top_official_result_missing_races')}`",
        f"- Backlog aggregation scope: `{backlog_summary.get('aggregation_scope')}`",
        f"- Promotion distance: `{promotion_distance.get('status') or 'NOT_RUN'}`",
        f"- Promotion distance blockers: `{promotion_distance.get('blockers')}`",
        f"- Promotion distance source exclusion reasons: `{promotion_distance.get('source_exclusion_reason_counts')}`",
        f"- Promotion distance official-result missing race IDs: `{promotion_distance.get('source_official_result_evidence_db_missing_race_ids')}`",
        f"- Promotion distance official-result runner path count: `{promotion_official_result.get('runner_path_count')}`",
        f"- Promotion distance official-result runner path source field: `{promotion_official_result.get('runner_paths_source_field')}`",
        f"- Promotion distance official-result coverage requested races: `{promotion_official_result.get('requested_race_count')}`",
        f"- Promotion distance official-result requested race count source: `{promotion_official_result.get('requested_race_count_source')}`",
        f"- Promotion distance official-result legacy requested race count without IDs: `{promotion_official_result.get('legacy_requested_race_count_without_ids')}`",
        f"- Promotion distance official-result coverage races with rows: `{promotion_official_result.get('races_with_rows_count')}`",
        f"- Promotion distance official-result coverage missing races: `{promotion_official_result.get('missing_race_count')}`",
        f"- Promotion distance official-result missing exclusion count: `{promotion_official_result.get('missing_exclusion_count')}`",
        f"- Best candidate: `{promotion_distance.get('best_candidate_key')}`",
        f"- Best non-market candidate: `{promotion_distance.get('best_non_market_candidate_key')}`",
        f"- Best non-market Top1 margin gap: `{promotion_distance.get('best_non_market_top1_margin_gap')}`",
        f"- Predeclared residual triggered races: `{promotion_distance.get('predeclared_residual_triggered_race_count')}`",
        f"- Reserve substitution preflight: `{reserve_preflight.get('status') or 'NOT_RUN'}`",
        f"- Reserve substitution ready for policy review: `{reserve_preflight.get('ready_for_policy_review_count')}`",
        f"- Reserve substitution blocked candidates: `{reserve_preflight.get('blocked_candidate_count')}`",
        f"- Reserve substitution readiness blockers: `{reserve_preflight.get('readiness_blocker_counts')}`",
        f"- Reserve substitution dataset join blockers: `{reserve_preflight.get('dataset_join_blocker_counts')}`",
        f"- Reserve substitution ready race IDs: `{reserve_preflight.get('ready_race_ids')}`",
        f"- Reserve substitution manual review: `{reserve_manual_review.get('status') or 'NOT_RUN'}`",
        f"- Reserve substitution manual review ready candidates: `{reserve_manual_review.get('ready_candidate_count')}`",
        f"- Reserve substitution manual review mapping pairs: `{reserve_manual_review.get('mapping_pair_count')}`",
        f"- Reserve substitution manual review dataset join allowed: `{reserve_manual_review.get('dataset_join_allowed')}`",
        f"- Reserve substitution manual review official-result acceptance allowed: `{reserve_manual_review.get('official_result_acceptance_allowed')}`",
        f"- Reserve substitution manual review DB write: `{reserve_manual_review.get('db_write')}`",
        f"- Reserve substitution manual review blockers: `{reserve_manual_review.get('blockers')}`",
        f"- Reserve substitution policy impact: `{reserve_policy_impact.get('status') or 'NOT_RUN'}`",
        f"- Reserve substitution policy impact ready candidates: `{reserve_policy_impact.get('ready_candidate_count')}`",
        f"- Reserve substitution policy impact mapping pairs: `{reserve_policy_impact.get('mapping_pair_count')}`",
        (
            "- Reserve substitution policy impact potential runner rows blocked: "
            f"`{reserve_policy_impact.get('potential_official_result_runner_rows_blocked_by_policy')}`"
        ),
        f"- Reserve substitution policy impact matched backlog top-gap races: `{reserve_policy_impact.get('matched_backlog_top_gap_race_count')}`",
        f"- Reserve substitution policy impact dataset join allowed: `{reserve_policy_impact.get('dataset_join_allowed')}`",
        f"- Reserve substitution policy impact official-result acceptance allowed: `{reserve_policy_impact.get('official_result_acceptance_allowed')}`",
        f"- Reserve substitution policy impact DB write: `{reserve_policy_impact.get('db_write')}`",
        f"- Reserve substitution policy impact blockers: `{reserve_policy_impact.get('blockers')}`",
        f"- Timing-aligned rerun plan: `{source_artifacts.get('timing_aligned_rerun_plan')}`",
        f"- Timing-aligned rerun execution status: `{source_artifacts.get('timing_aligned_rerun_execution_status')}`",
        f"- Protected paths unchanged: `{packet.get('protected_paths_unchanged')}`",
        "",
        "No registry, DB, label, snapshot, EV action, betting action, TGR, or production pointer write was performed.",
        "",
    ]
    return "\n".join(lines)


def reserve_substitution_manual_review_sibling_path(
    reserve_substitution_preflight_path: Path | None,
) -> Path | None:
    if reserve_substitution_preflight_path is None:
        return None
    candidate = (
        reserve_substitution_preflight_path.parent
        / "reserve_substitution_manual_review_packet.json"
    )
    return candidate if candidate.exists() else None


def reserve_substitution_policy_impact_sibling_path(
    reserve_substitution_preflight_path: Path | None,
) -> Path | None:
    if reserve_substitution_preflight_path is None:
        return None
    candidate = (
        reserve_substitution_preflight_path.parent
        / "reserve_substitution_policy_impact_preview.json"
    )
    return candidate if candidate.exists() else None


def write_packet(
    output_dir: Path,
    packet: Mapping[str, Any],
    *,
    stage2_forward_metrics: Mapping[str, Any] | None = None,
) -> None:
    write_json(output_dir / "high_accuracy_refinement_packet.json", packet)
    write_json(output_dir / "promotion_pr_gate.json", packet["promotion_pr_gate"])
    if stage2_forward_metrics is not None:
        write_json(output_dir / "stage2_forward_joined_metrics.json", stage2_forward_metrics)
    write_text(output_dir / "SUMMARY.md", build_summary(packet))
    write_text(output_dir / "final_status.txt", str(packet["final_status"]) + "\n")


def run_refinement_packet(
    *,
    calibration_report_path: Path | None = None,
    stage2_forward_metrics_path: Path | None = None,
    stage2_predictions_path: Path | None = None,
    shadow_replay_metrics_path: Path | None = None,
    odds_gate_report_path: Path | None = None,
    odds_augmented_report_path: Path | None = None,
    ev_diagnostics_path: Path | None = None,
    unified_evidence_report_path: Path | None = None,
    backlog_unified_evidence_status_path: Path | None = None,
    promotion_distance_report_path: Path | None = None,
    reserve_substitution_preflight_path: Path | None = None,
    timing_aligned_rerun_plan_path: Path | None = None,
    timing_aligned_rerun_execution_status_path: Path | None = None,
    output_dir: Path | None = None,
    thresholds: AccuracyGateThresholds = AccuracyGateThresholds(),
) -> dict[str, Any]:
    if not any(
        (
            calibration_report_path,
            stage2_forward_metrics_path,
            stage2_predictions_path,
            shadow_replay_metrics_path,
            odds_gate_report_path,
            odds_augmented_report_path,
            unified_evidence_report_path,
            backlog_unified_evidence_status_path,
            promotion_distance_report_path,
            reserve_substitution_preflight_path,
            timing_aligned_rerun_plan_path,
            timing_aligned_rerun_execution_status_path,
        )
    ):
        raise ValueError("at_least_one_source_report_required")
    generated_at = datetime.now().astimezone()
    output_dir = output_dir or DEFAULT_OUTPUT_PARENT / f"high_accuracy_refinement_packet_{now_id(generated_at)}"
    output_dir = unique_dir(assert_output_dir_safe(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    protected_before = protected_hashes()
    reserve_substitution_manual_review_path = (
        reserve_substitution_manual_review_sibling_path(
            reserve_substitution_preflight_path
        )
    )
    reserve_substitution_policy_impact_path = (
        reserve_substitution_policy_impact_sibling_path(
            reserve_substitution_preflight_path
        )
    )
    stage2_forward_metrics = load_json(stage2_forward_metrics_path)
    backlog_unified_evidence_status = enriched_unified_evidence_aggregate_status(
        load_json(backlog_unified_evidence_status_path)
    )
    packet = build_packet(
        calibration_report=load_json(calibration_report_path),
        stage2_forward_metrics=stage2_forward_metrics,
        shadow_replay_metrics=load_json(shadow_replay_metrics_path),
        odds_gate_report=load_json(odds_gate_report_path),
        odds_augmented_report=load_json(odds_augmented_report_path),
        ev_diagnostics=load_json(ev_diagnostics_path),
        unified_evidence_report=load_json(unified_evidence_report_path),
        backlog_unified_evidence_status=backlog_unified_evidence_status,
        promotion_distance_report=load_json(promotion_distance_report_path),
        reserve_substitution_preflight=load_json(reserve_substitution_preflight_path),
        reserve_substitution_manual_review=load_json(
            reserve_substitution_manual_review_path
        ),
        reserve_substitution_policy_impact=load_json(
            reserve_substitution_policy_impact_path
        ),
        thresholds=thresholds,
        calibration_report_path=calibration_report_path,
        stage2_forward_metrics_path=stage2_forward_metrics_path,
        stage2_predictions_path=stage2_predictions_path,
        shadow_replay_metrics_path=shadow_replay_metrics_path,
        odds_gate_report_path=odds_gate_report_path,
        odds_augmented_report_path=odds_augmented_report_path,
        ev_diagnostics_path=ev_diagnostics_path,
        unified_evidence_report_path=unified_evidence_report_path,
        backlog_unified_evidence_status_path=backlog_unified_evidence_status_path,
        promotion_distance_report_path=promotion_distance_report_path,
        reserve_substitution_preflight_path=reserve_substitution_preflight_path,
        reserve_substitution_manual_review_path=(
            reserve_substitution_manual_review_path
        ),
        reserve_substitution_policy_impact_path=(
            reserve_substitution_policy_impact_path
        ),
        timing_aligned_rerun_plan_path=timing_aligned_rerun_plan_path,
        timing_aligned_rerun_execution_status_path=timing_aligned_rerun_execution_status_path,
        output_dir=output_dir,
        generated_at=generated_at,
        protected_before=protected_before,
    )
    write_packet(output_dir, packet, stage2_forward_metrics=stage2_forward_metrics)
    return {
        "output_dir": relpath(output_dir),
        "final_status": packet["final_status"],
        "promotion_pr_gate_status": packet["promotion_pr_gate"]["status"],
        "selected_stage": packet["promotion_pr_gate"]["selected_stage"],
        "protected_paths_unchanged": packet["protected_paths_unchanged"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-report", type=Path)
    parser.add_argument("--stage2-forward-metrics", type=Path)
    parser.add_argument("--stage2-predictions", type=Path)
    parser.add_argument("--shadow-replay-metrics", type=Path)
    parser.add_argument("--odds-gate-report", type=Path)
    parser.add_argument("--odds-augmented-report", type=Path)
    parser.add_argument("--ev-diagnostics", type=Path)
    parser.add_argument("--unified-evidence-report", type=Path)
    parser.add_argument("--backlog-unified-evidence-status", type=Path)
    parser.add_argument("--promotion-distance-report", type=Path)
    parser.add_argument("--reserve-substitution-preflight", type=Path)
    parser.add_argument("--timing-aligned-rerun-plan", type=Path)
    parser.add_argument("--timing-aligned-rerun-execution-status", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-safe-joined-races", type=int, default=100)
    parser.add_argument("--min-top1-delta", type=float, default=0.02)
    parser.add_argument("--min-top3-delta", type=float, default=0.0)
    parser.add_argument("--max-mean-winner-rank-delta", type=float, default=0.0)
    parser.add_argument("--max-brier-delta", type=float, default=0.0)
    parser.add_argument("--max-logloss-delta", type=float, default=0.0)
    parser.add_argument("--max-calibration-distance-delta", type=float, default=0.0)
    parser.add_argument("--max-box1-top-pick-share", type=float, default=0.35)
    parser.add_argument("--max-box1-share-delta", type=float, default=0.0)
    parser.add_argument("--max-probability-sum-error", type=float, default=1e-6)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    thresholds = AccuracyGateThresholds(
        min_safe_joined_races=args.min_safe_joined_races,
        min_top1_delta=args.min_top1_delta,
        min_top3_delta=args.min_top3_delta,
        max_mean_winner_rank_delta=args.max_mean_winner_rank_delta,
        max_brier_delta=args.max_brier_delta,
        max_logloss_delta=args.max_logloss_delta,
        max_calibration_distance_delta=args.max_calibration_distance_delta,
        max_box1_top_pick_share=args.max_box1_top_pick_share,
        max_box1_share_delta=args.max_box1_share_delta,
        max_probability_sum_error=args.max_probability_sum_error,
    )
    result = run_refinement_packet(
        calibration_report_path=args.calibration_report,
        stage2_forward_metrics_path=args.stage2_forward_metrics,
        stage2_predictions_path=args.stage2_predictions,
        shadow_replay_metrics_path=args.shadow_replay_metrics,
        odds_gate_report_path=args.odds_gate_report,
        odds_augmented_report_path=args.odds_augmented_report,
        ev_diagnostics_path=args.ev_diagnostics,
        unified_evidence_report_path=args.unified_evidence_report,
        backlog_unified_evidence_status_path=args.backlog_unified_evidence_status,
        promotion_distance_report_path=args.promotion_distance_report,
        reserve_substitution_preflight_path=args.reserve_substitution_preflight,
        timing_aligned_rerun_plan_path=args.timing_aligned_rerun_plan,
        timing_aligned_rerun_execution_status_path=args.timing_aligned_rerun_execution_status,
        output_dir=args.output_dir,
        thresholds=thresholds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
