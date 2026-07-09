#!/usr/bin/env python3
"""Build a fail-closed report-only prediction root-cause remediation packet.

This script consolidates existing reports only. It does not train models,
promote candidates, write labels/odds, rewrite snapshots, mutate registries,
emit EV, or touch betting paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = (
    "artifacts/full_evidence_orchestration_20260525/"
    "prediction_root_cause_remediation_"
)

REPORT_JSON = "remediation_report.json"
SUMMARY_MD = "SUMMARY.md"
FINAL_STATUS = "ROOT_CAUSE_REMEDIATION_PACKET_BUILT_REPORT_ONLY"
FINAL_DATA_MISSING = "ROOT_CAUSE_REMEDIATION_PACKET_DATA_MISSING"
MIN_REVIEW_RACES = 100
MIN_RESIDUAL_TRIGGER_RACES = 10

NO_WRITE_GUARANTEES = {
    "db_write": False,
    "label_write": False,
    "odds_write": False,
    "snapshot_rewrite": False,
    "manifest_rewrite": False,
    "model_training": False,
    "model_artifact_write": False,
    "registry_mutation": False,
    "promotion": False,
    "betting_or_ev_action": False,
}

IDENTITY_FIELDS = [
    "race_id",
    "classification",
    "metric_decision",
    "name_mismatch_count",
    "missing_predicted_box_count",
    "extra_official_box_count",
    "allowed_extra_scratched_official_box_count",
    "remapped_participant_count",
    "dropped_participant_count",
    "source_join_artifact",
]

FEATURE_FIELDS = [
    "feature",
    "decision",
    "decision_reason",
    "train_present_rows",
    "train_rows",
    "train_present_pct",
    "holdout_present_rows",
    "holdout_rows",
    "holdout_present_pct",
    "parity_status",
    "fail_reasons",
]

OBJECTIVE_FIELDS = [
    "candidate_key",
    "family",
    "race_count",
    "top1",
    "top3",
    "mean_winner_rank",
    "logloss",
    "brier",
    "calibration_slope",
    "top1_minus_market",
    "top3_minus_market",
    "mean_winner_rank_minus_market",
    "logloss_minus_market",
    "brier_minus_market",
    "failure_classification",
]

FEATURE_POWER_FIELDS = [
    "family",
    "candidate_key",
    "scope",
    "race_count",
    "runner_rows",
    "top1_minus_market",
    "top3_minus_market",
    "mean_winner_rank_minus_market",
    "race_winner_logloss_minus_market",
    "brier_minus_market",
    "status",
]

RESIDUAL_FIELDS = [
    "dimension",
    "dimension_value",
    "race_count",
    "rank_first_net_edge_count",
    "mean_candidate_minus_market_logloss",
    "mean_winner_rank_delta",
    "decision",
]


def now_id(now: datetime | None = None) -> str:
    return (now or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return os.path.relpath(path.resolve(), ROOT.resolve())
    except ValueError:
        return str(path)


def repo_path(value: str | None, *, base: Path | None = None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if base is not None and (base / path).exists():
        return (base / path).resolve()
    return (ROOT / path).resolve()


def assert_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative = logical.absolute().relative_to(ROOT.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_prediction_root_cause_remediation:{relative}")
    return logical.absolute()


def unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = Path(f"{base}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"output_dir_collision_exhausted:{base}")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_not_object:{path}")
    return payload


def load_csv(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_manifest(output_dir: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        files[relpath(path) or str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "schema_version": "prediction_root_cause_remediation_output_manifest_v1",
        "output_dir": relpath(output_dir),
        "files": files,
    }


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return 0


def csv_float(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in {None, ""}:
        return None
    return to_float(value)


def metric_delta(candidate: Mapping[str, Any], market: Mapping[str, Any], key: str) -> float | None:
    left = to_float(candidate.get(key))
    right = to_float(market.get(key))
    if left is None or right is None:
        return None
    return left - right


def classify_unsafe_match(match: Mapping[str, Any]) -> tuple[str, str]:
    missing = list_value(match.get("missing_predicted_boxes"))
    extra = list_value(match.get("disallowed_extra_official_boxes"))
    name_mismatch = list_value(match.get("name_mismatches"))
    allowed_scratched_extra = list_value(match.get("allowed_extra_scratched_official_boxes"))
    alignment = mapping(match.get("prejump_runner_alignment"))
    remapped = list_value(alignment.get("remapped_participants"))
    dropped = list_value(alignment.get("dropped_participants"))

    labels: list[str] = []
    if remapped:
        labels.append("reserve_remap")
    if dropped or allowed_scratched_extra:
        labels.append("scratch_or_reserve_substitution")
    if missing:
        labels.append("missing_predicted_box")
    if extra:
        labels.append("extra_official_box")
    if name_mismatch:
        labels.append("name_mismatch")
    if not labels:
        labels.append("unknown_identity_mismatch")

    hard_mismatch = bool(missing or extra or name_mismatch)
    if hard_mismatch:
        return "|".join(labels), "EXCLUDE_UNTIL_REJOIN"
    if remapped or dropped or allowed_scratched_extra:
        return "|".join(labels), "NEEDS_CANONICAL_REMAP_RULE"
    return "|".join(labels), "EXCLUDE_UNTIL_REJOIN"


def build_identity_noise_ledger(aggregate_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    unsafe_payload = mapping(aggregate_report.get("unsafe_result_matches"))
    matches = list_value(unsafe_payload.get("unsafe_result_matches"))
    rows: list[dict[str, Any]] = []
    for match in matches:
        match_map = mapping(match)
        classification, decision = classify_unsafe_match(match_map)
        alignment = mapping(match_map.get("prejump_runner_alignment"))
        rows.append(
            {
                "race_id": match_map.get("race_id"),
                "classification": classification,
                "metric_decision": decision,
                "name_mismatch_count": len(list_value(match_map.get("name_mismatches"))),
                "missing_predicted_box_count": len(list_value(match_map.get("missing_predicted_boxes"))),
                "extra_official_box_count": len(list_value(match_map.get("disallowed_extra_official_boxes"))),
                "allowed_extra_scratched_official_box_count": len(
                    list_value(match_map.get("allowed_extra_scratched_official_boxes"))
                ),
                "remapped_participant_count": len(list_value(alignment.get("remapped_participants"))),
                "dropped_participant_count": len(list_value(alignment.get("dropped_participants"))),
                "source_join_artifact": match_map.get("source_join_artifact"),
            }
        )
    return rows


def feature_decision(feature: Mapping[str, Any]) -> tuple[str, str]:
    reasons = [str(item) for item in list_value(feature.get("fail_reasons"))]
    parity = mapping(feature.get("parity"))
    train_rows = to_int(parity.get("train_present_rows"))
    provenance_bad = any(
        "unsafe_history_source" in reason
        or "cutoff_not_strictly_before_target_race" in reason
        or "history_provenance_not_pass" in reason
        for reason in reasons
    )
    if not reasons:
        return "READY_FOR_SHADOW_ONLY_ACTIVATION_REVIEW", "feature_gate_has_no_fail_reasons"
    if provenance_bad:
        return "KEEP_QUARANTINED", "unsafe_or_unproven_history_provenance"
    if train_rows <= 0 or "all_missing_in_train" in reasons:
        return "DATA_MISSING", "train_slice_missing_feature_values"
    return "KEEP_QUARANTINED", "feature_activation_gate_failures_remain"


def build_feature_provenance_ledger(feature_gate_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in list_value(feature_gate_report.get("features")):
        feature = mapping(item)
        parity = mapping(feature.get("parity"))
        decision, reason = feature_decision(feature)
        rows.append(
            {
                "feature": feature.get("feature"),
                "decision": decision,
                "decision_reason": reason,
                "train_present_rows": parity.get("train_present_rows"),
                "train_rows": parity.get("train_rows"),
                "train_present_pct": parity.get("train_present_pct"),
                "holdout_present_rows": parity.get("holdout_present_rows"),
                "holdout_rows": parity.get("holdout_rows"),
                "holdout_present_pct": parity.get("holdout_present_pct"),
                "parity_status": parity.get("parity_status"),
                "fail_reasons": "|".join(str(reason) for reason in list_value(feature.get("fail_reasons"))),
            }
        )
    return rows


def objective_failure_classification(
    *,
    candidate: Mapping[str, Any],
    market: Mapping[str, Any],
    min_review_races: int,
) -> str:
    classes: list[str] = []
    top1_delta = metric_delta(candidate, market, "top1")
    top3_delta = metric_delta(candidate, market, "top3")
    logloss_delta = metric_delta(candidate, market, "logloss")
    brier_delta = metric_delta(candidate, market, "brier")
    mean_rank_delta = metric_delta(candidate, market, "mean_winner_rank")
    if to_int(candidate.get("race_count")) < min_review_races:
        classes.append("SAMPLE_UNDERPOWERED")
    if top1_delta is not None and top1_delta > 0 and (
        (top3_delta is not None and top3_delta < 0)
        or (logloss_delta is not None and logloss_delta > 0)
        or (brier_delta is not None and brier_delta > 0)
    ):
        classes.append("TOP1_ONLY_TRADEOFF")
    if (top1_delta is not None and top1_delta <= 0) or (
        mean_rank_delta is not None and mean_rank_delta > 0
    ):
        classes.append("RANK_SIGNAL_WEAK")
    if (logloss_delta is not None and logloss_delta > 0) or (
        brier_delta is not None and brier_delta > 0
    ):
        classes.append("PROBABILITY_CALIBRATION_BAD")
    return "|".join(classes or ["NO_FAILURE_AGAINST_MARKET_ON_AVAILABLE_METRICS"])


def candidate_metrics_by_key(rolling_report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    metrics = rolling_report.get("candidate_metrics_by_key")
    if isinstance(metrics, Mapping):
        return {str(key): mapping(value) for key, value in metrics.items()}
    rows: dict[str, Mapping[str, Any]] = {}
    for key in [
        "market_metrics",
        "baseline_metrics",
        "best_rank_accuracy_candidate_metrics",
        "best_non_market_candidate_metrics",
    ]:
        payload = mapping(rolling_report.get(key))
        candidate_key = str(payload.get("candidate_key") or key)
        rows[candidate_key] = payload
    return rows


def build_objective_metric_split(
    rolling_report: Mapping[str, Any],
    *,
    min_review_races: int,
) -> list[dict[str, Any]]:
    market_key = str(rolling_report.get("market_candidate_key") or "market_only_implied")
    metrics_by_key = candidate_metrics_by_key(rolling_report)
    market = metrics_by_key.get(market_key) or mapping(rolling_report.get("market_metrics"))
    rows: list[dict[str, Any]] = []
    for key, candidate in sorted(metrics_by_key.items()):
        slope = mapping(candidate.get("calibration_slope_intercept")).get("slope")
        rows.append(
            {
                "candidate_key": key,
                "family": candidate.get("family"),
                "race_count": candidate.get("race_count"),
                "top1": candidate.get("top1"),
                "top3": candidate.get("top3"),
                "mean_winner_rank": candidate.get("mean_winner_rank"),
                "logloss": candidate.get("logloss"),
                "brier": candidate.get("brier"),
                "calibration_slope": slope,
                "top1_minus_market": metric_delta(candidate, market, "top1"),
                "top3_minus_market": metric_delta(candidate, market, "top3"),
                "mean_winner_rank_minus_market": metric_delta(candidate, market, "mean_winner_rank"),
                "logloss_minus_market": metric_delta(candidate, market, "logloss"),
                "brier_minus_market": metric_delta(candidate, market, "brier"),
                "failure_classification": (
                    "MARKET_BASELINE"
                    if key == market_key
                    else objective_failure_classification(
                        candidate=candidate,
                        market=market,
                        min_review_races=min_review_races,
                    )
                ),
            }
        )
    return rows


def feature_family_from_row(row: Mapping[str, Any]) -> str:
    feature_set = str(row.get("feature_set") or row.get("candidate_key") or "")
    if "market" in feature_set:
        return "market"
    if "shadow" in feature_set:
        return "shadow_derived"
    if "box" in feature_set:
        return "box"
    if "same_distance" in feature_set:
        return "same_distance"
    if "same_grade" in feature_set:
        return "same_grade"
    if "history" in feature_set or "non_box" in feature_set:
        return "recent_form_history"
    return feature_set or "unknown"


def family_power_status(row: Mapping[str, Any]) -> str:
    top1 = csv_float(row, "top1_minus_market")
    top3 = csv_float(row, "top3_minus_market")
    logloss = csv_float(row, "race_winner_logloss_minus_market")
    mean_rank = csv_float(row, "mean_winner_rank_minus_market")
    if top1 is None and top3 is None and logloss is None and mean_rank is None:
        return "MARKET_BASELINE_OR_NOT_COMPARABLE"
    if top1 is not None and top1 > 0 and (top3 or 0) >= 0 and (logloss or 0) <= 0 and (mean_rank or 0) <= 0:
        return "FAMILY_LIFT_CANDIDATE"
    return "NO_FAMILY_LIFT_VS_MARKET"


def build_feature_family_power_matrix(ablation_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in ablation_rows:
        rows.append(
            {
                "family": feature_family_from_row(row),
                "candidate_key": row.get("candidate_key"),
                "scope": row.get("scope"),
                "race_count": row.get("race_count"),
                "runner_rows": row.get("runner_rows"),
                "top1_minus_market": row.get("top1_minus_market"),
                "top3_minus_market": row.get("top3_minus_market"),
                "mean_winner_rank_minus_market": row.get("mean_winner_rank_minus_market"),
                "race_winner_logloss_minus_market": row.get("race_winner_logloss_minus_market"),
                "brier_minus_market": row.get("brier_minus_market"),
                "status": family_power_status(row),
            }
        )
    return rows


def build_residual_predeclare_candidates(
    split_rows: Sequence[Mapping[str, Any]],
    *,
    min_residual_trigger_races: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in split_rows:
        race_count = to_int(row.get("race_count"))
        net_edge = to_int(row.get("rank_first_net_edge_count"))
        pre_race_usable = str(row.get("pre_race_usable")).lower() == "true"
        if not pre_race_usable or net_edge <= 0:
            continue
        is_predeclared_trigger_shape = (
            row.get("dimension") == "residual_trigger_regime"
            and str(row.get("dimension_value") or "").startswith("triggered:")
        )
        decision = (
            "PREDECLARE_CANDIDATE"
            if is_predeclared_trigger_shape
            and race_count >= min_residual_trigger_races
            else "KEEP_COLLECTING_ONLY"
        )
        rows.append(
            {
                "dimension": row.get("dimension"),
                "dimension_value": row.get("dimension_value"),
                "race_count": race_count,
                "rank_first_net_edge_count": net_edge,
                "mean_candidate_minus_market_logloss": row.get("mean_candidate_minus_market_logloss"),
                "mean_winner_rank_delta": row.get("mean_winner_rank_delta"),
                "decision": decision,
            }
        )
    return rows


def summarize_identity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "unsafe_identity_race_count": len(rows),
        "classification_counts": dict(Counter(str(row.get("classification")) for row in rows)),
        "metric_decision_counts": dict(Counter(str(row.get("metric_decision")) for row in rows)),
    }


def summarize_markdown(report: Mapping[str, Any]) -> str:
    identity = mapping(report.get("identity_noise_summary"))
    objective = mapping(report.get("objective_summary"))
    return "\n".join(
        [
            "# Prediction Root-Cause Remediation Packet",
            "",
            f"Final status: `{report.get('final_status')}`",
            f"Next decision: `{report.get('next_decision')}`",
            "",
            f"- Unsafe identity races: `{identity.get('unsafe_identity_race_count')}`",
            f"- Metric identity decisions: `{identity.get('metric_decision_counts')}`",
            f"- Feature decisions: `{report.get('feature_decision_counts')}`",
            f"- Objective failure classes: `{objective.get('failure_classification_counts')}`",
            f"- Feature-family statuses: `{report.get('feature_family_status_counts')}`",
            f"- Residual predeclare statuses: `{report.get('residual_predeclare_status_counts')}`",
            "",
            "Artifacts:",
            f"- `{report.get('identity_noise_ledger_csv')}`",
            f"- `{report.get('feature_provenance_parity_ledger_csv')}`",
            f"- `{report.get('objective_metric_split_csv')}`",
            f"- `{report.get('feature_family_power_matrix_csv')}`",
            f"- `{report.get('residual_regime_predeclare_candidates_csv')}`",
            "",
            "Report-only: no DB, label, odds, snapshot, model, registry, promotion, EV, or betting mutation.",
            "",
        ]
    )


def build_packet(
    *,
    aggregate_report_path: Path,
    promotion_report_path: Path,
    high_accuracy_report_path: Path,
    rolling_report_path: Path,
    feature_gate_report_path: Path,
    ablation_metrics_path: Path,
    residual_split_summary_path: Path,
    output_dir: Path,
    min_review_races: int = MIN_REVIEW_RACES,
    min_residual_trigger_races: int = MIN_RESIDUAL_TRIGGER_RACES,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    output_dir = unique_dir(assert_output_dir_safe(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)

    aggregate_report = load_json(aggregate_report_path)
    promotion_report = load_json(promotion_report_path)
    high_accuracy_report = load_json(high_accuracy_report_path)
    rolling_report = load_json(rolling_report_path)
    feature_gate_report = load_json(feature_gate_report_path)
    ablation_rows = load_csv(ablation_metrics_path)
    residual_split_rows = load_csv(residual_split_summary_path)

    identity_rows = build_identity_noise_ledger(aggregate_report)
    feature_rows = build_feature_provenance_ledger(feature_gate_report)
    objective_rows = build_objective_metric_split(
        rolling_report,
        min_review_races=min_review_races,
    )
    family_rows = build_feature_family_power_matrix(ablation_rows)
    residual_rows = build_residual_predeclare_candidates(
        residual_split_rows,
        min_residual_trigger_races=min_residual_trigger_races,
    )

    source_artifacts = {
        "aggregate_report": relpath(aggregate_report_path),
        "promotion_report": relpath(promotion_report_path),
        "high_accuracy_report": relpath(high_accuracy_report_path),
        "rolling_report": relpath(rolling_report_path),
        "feature_gate_report": relpath(feature_gate_report_path),
        "ablation_metrics": relpath(ablation_metrics_path),
        "residual_split_summary": relpath(residual_split_summary_path),
    }

    feature_decisions = Counter(str(row.get("decision")) for row in feature_rows)
    objective_classes = Counter()
    for row in objective_rows:
        for item in str(row.get("failure_classification") or "").split("|"):
            if item:
                objective_classes[item] += 1
    family_statuses = Counter(str(row.get("status")) for row in family_rows)
    residual_statuses = Counter(str(row.get("decision")) for row in residual_rows)

    blockers: list[str] = []
    if identity_rows:
        blockers.append("unsafe_identity_matches_require_cleanup")
    rolling_sample = mapping(promotion_report.get("rolling_sample"))
    if to_int(rolling_sample.get("sample_race_count")) < min_review_races:
        blockers.append("review_sample_below_floor")
    if any(row.get("decision") != "READY_FOR_SHADOW_ONLY_ACTIVATION_REVIEW" for row in feature_rows):
        blockers.append("feature_provenance_or_parity_not_ready")
    if objective_classes.get("TOP1_ONLY_TRADEOFF") or objective_classes.get("PROBABILITY_CALIBRATION_BAD"):
        blockers.append("objective_probability_tradeoff_not_safe")
    if not residual_rows or not residual_statuses.get("PREDECLARE_CANDIDATE"):
        blockers.append("residual_regimes_underpowered")

    next_decision = "IDENTITY_LABEL_CLEANUP_NEXT" if identity_rows else "FEATURE_PROVENANCE_PARITY_NEXT"
    if not blockers:
        next_decision = "READY_FOR_NEXT_REVIEW_PACKET"

    report = {
        "schema_version": "prediction_root_cause_remediation_packet_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": FINAL_STATUS if not blockers else FINAL_STATUS,
        "output_dir": relpath(output_dir),
        "next_decision": next_decision,
        "blockers": blockers,
        "source_artifacts": source_artifacts,
        "identity_noise_ledger_csv": relpath(output_dir / "identity_noise_ledger.csv"),
        "feature_provenance_parity_ledger_csv": relpath(output_dir / "feature_provenance_parity_ledger.csv"),
        "objective_metric_split_csv": relpath(output_dir / "objective_metric_split.csv"),
        "feature_family_power_matrix_csv": relpath(output_dir / "feature_family_power_matrix.csv"),
        "residual_regime_predeclare_candidates_csv": relpath(output_dir / "residual_regime_predeclare_candidates.csv"),
        "identity_noise_summary": summarize_identity(identity_rows),
        "promotion_status": promotion_report.get("final_status"),
        "high_accuracy_status": high_accuracy_report.get("final_status"),
        "odds_gate_summary": high_accuracy_report.get("odds_research_gate_summary"),
        "rolling_status": rolling_report.get("final_status"),
        "rolling_sample": promotion_report.get("rolling_sample"),
        "feature_gate_status": feature_gate_report.get("final_status"),
        "feature_decision_counts": dict(sorted(feature_decisions.items())),
        "objective_summary": {
            "failure_classification_counts": dict(sorted(objective_classes.items())),
            "market_candidate_key": rolling_report.get("market_candidate_key")
            or "market_only_implied",
            "best_candidate_key": rolling_report.get("best_candidate_key"),
            "best_rank_accuracy_candidate_key": mapping(
                rolling_report.get("best_rank_accuracy_candidate_metrics")
            ).get("candidate_key"),
        },
        "feature_family_status_counts": dict(sorted(family_statuses.items())),
        "residual_predeclare_status_counts": dict(sorted(residual_statuses.items())),
        "promotion_ready": False,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }

    write_csv(output_dir / "identity_noise_ledger.csv", identity_rows, IDENTITY_FIELDS)
    write_csv(output_dir / "feature_provenance_parity_ledger.csv", feature_rows, FEATURE_FIELDS)
    write_csv(output_dir / "objective_metric_split.csv", objective_rows, OBJECTIVE_FIELDS)
    write_csv(output_dir / "feature_family_power_matrix.csv", family_rows, FEATURE_POWER_FIELDS)
    write_csv(output_dir / "residual_regime_predeclare_candidates.csv", residual_rows, RESIDUAL_FIELDS)
    write_json(output_dir / REPORT_JSON, report)
    (output_dir / SUMMARY_MD).write_text(summarize_markdown(report), encoding="utf-8")
    (output_dir / "final_status.txt").write_text(str(report["final_status"]) + "\n", encoding="utf-8")
    (output_dir / "source_artifact_manifest.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in source_artifacts.items()) + "\n",
        encoding="utf-8",
    )
    write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-report", type=Path, required=True)
    parser.add_argument("--promotion-report", type=Path, required=True)
    parser.add_argument("--high-accuracy-report", type=Path, required=True)
    parser.add_argument("--rolling-report", type=Path, required=True)
    parser.add_argument("--feature-gate-report", type=Path, required=True)
    parser.add_argument("--ablation-metrics", type=Path, required=True)
    parser.add_argument("--residual-split-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-review-races", type=int, default=MIN_REVIEW_RACES)
    parser.add_argument(
        "--min-residual-trigger-races",
        type=int,
        default=MIN_RESIDUAL_TRIGGER_RACES,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    generated_at = datetime.now().astimezone()
    output_dir = (
        args.output_dir
        or DEFAULT_EVIDENCE_ROOT
        / f"prediction_root_cause_remediation_{now_id(generated_at)}_report_only"
    )
    report = build_packet(
        aggregate_report_path=args.aggregate_report,
        promotion_report_path=args.promotion_report,
        high_accuracy_report_path=args.high_accuracy_report,
        rolling_report_path=args.rolling_report,
        feature_gate_report_path=args.feature_gate_report,
        ablation_metrics_path=args.ablation_metrics,
        residual_split_summary_path=args.residual_split_summary,
        output_dir=output_dir,
        min_review_races=args.min_review_races,
        min_residual_trigger_races=args.min_residual_trigger_races,
        generated_at=generated_at,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
