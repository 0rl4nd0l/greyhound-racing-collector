#!/usr/bin/env python3
"""Report-only calibration layer design from a stable challenger review.

This script validates an existing model review packet and challenger review,
then describes the proposed power-normalization calibration layer without
writing a model artifact, registry entry, production config, or betting surface.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accuracy_program.evaluation import score_predictions
from accuracy_program.calibration import power_normalize_by_race
from scripts.review_snapshot_challenger import (
    _clean_rows,
    _load_jsonl,
)


SCHEMA_VERSION = "calibration_layer_design_v1"
CANDIDATE_ARM = "power_calibrated_baseline"
OUTPUT_KEY = "calibrated_win_prob_report_only"


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("json_root_not_object")
    return data


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return None
        return parsed
    except Exception:
        return None


def _same_path(left: Any, right: Path | None) -> bool:
    if left is None or right is None:
        return False
    return Path(str(left)).resolve() == right.resolve()


def _comparison(calibrated: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    def lower(metric: str) -> bool | None:
        left = _safe_float(calibrated.get(metric))
        right = _safe_float(baseline.get(metric))
        return None if left is None or right is None else left < right

    def same(metric: str) -> bool | None:
        left = _safe_float(calibrated.get(metric))
        right = _safe_float(baseline.get(metric))
        return None if left is None or right is None else left == right

    return {
        "log_loss_improved": lower("log_loss"),
        "brier_improved": lower("brier"),
        "top1_preserved": same("top1"),
        "top2_preserved": same("top2"),
        "top3_preserved": same("top3"),
        "mean_winner_rank_preserved": same("mean_winner_rank"),
    }


def _resolve_inputs(
    packet: Mapping[str, Any],
    *,
    challenger_review_path: Path | None,
    dataset_path: Path | None,
) -> tuple[Path | None, Path | None]:
    gate = packet.get("challenger_review_gate")
    evidence = packet.get("source_evidence")
    gate = gate if isinstance(gate, Mapping) else {}
    evidence = evidence if isinstance(evidence, Mapping) else {}
    if challenger_review_path is None and gate.get("path"):
        challenger_review_path = Path(str(gate["path"]))
    if dataset_path is None and evidence.get("evaluation_dataset"):
        dataset_path = Path(str(evidence["evaluation_dataset"]))
    return challenger_review_path, dataset_path


def build_design(
    *,
    model_review_packet_path: Path,
    challenger_review_path: Path | None = None,
    dataset_path: Path | None = None,
) -> dict[str, Any]:
    packet_path = model_review_packet_path.resolve()
    failures: list[str] = []
    warnings: list[str] = []

    try:
        packet = _json(packet_path)
    except Exception as exc:
        packet = {}
        failures.append(f"model_review_packet_unreadable:{type(exc).__name__}")

    challenger_review_path, dataset_path = _resolve_inputs(
        packet,
        challenger_review_path=challenger_review_path,
        dataset_path=dataset_path,
    )

    gate = packet.get("challenger_review_gate")
    promotion_control = packet.get("promotion_control")
    gate = gate if isinstance(gate, Mapping) else {}
    promotion_control = (
        promotion_control if isinstance(promotion_control, Mapping) else {}
    )

    if packet.get("schema_version") != "model_review_packet_v1":
        failures.append("model_review_packet_schema_mismatch")
    if packet.get("status") != "READY_FOR_CHALLENGER_REVIEW":
        failures.append("model_review_packet_not_ready")
    if packet.get("failures"):
        failures.append("model_review_packet_contains_failures")
    if promotion_control.get("promotion_allowed") is not False:
        failures.append("model_review_packet_promotion_not_blocked")
    if promotion_control.get("registry_mutation_allowed") is not False:
        failures.append("model_review_packet_registry_mutation_not_blocked")

    if gate.get("status") != "READY":
        failures.append("challenger_review_gate_not_ready")
    if gate.get("candidate_arm") != CANDIDATE_ARM:
        failures.append("challenger_review_candidate_mismatch")
    if gate.get("stability_status") != "STABLE_REPORT_ONLY":
        failures.append("challenger_review_not_stable_report_only")
    if gate.get("promotion_allowed") is not False:
        failures.append("challenger_review_promotion_not_blocked")
    if gate.get("registry_mutation_allowed") is not False:
        failures.append("challenger_review_registry_mutation_not_blocked")
    if gate.get("model_artifact_written") is not False:
        failures.append("challenger_review_model_artifact_written")
    if gate.get("all_log_loss_improved") is not True:
        failures.append("challenger_review_log_loss_not_improved")
    if gate.get("all_brier_improved") is not True:
        failures.append("challenger_review_brier_not_improved")
    if gate.get("all_ranking_preserved") is not True:
        failures.append("challenger_review_ranking_not_preserved")

    alpha = _safe_float(gate.get("selected_alpha"))
    if alpha is None or alpha <= 0:
        failures.append("selected_alpha_missing_or_invalid")

    review: dict[str, Any] = {}
    if challenger_review_path is None:
        failures.append("challenger_review_path_missing")
    elif not challenger_review_path.exists():
        failures.append("challenger_review_missing")
    else:
        try:
            review = _json(challenger_review_path.resolve())
        except Exception as exc:
            failures.append(f"challenger_review_unreadable:{type(exc).__name__}")
    if review:
        if review.get("schema_version") != "snapshot_challenger_review_v1":
            failures.append("challenger_review_schema_mismatch")
        if review.get("status") != "SUCCESS":
            failures.append("challenger_review_status_not_success")
        if review.get("failures"):
            failures.append("challenger_review_contains_failures")
        if review.get("warnings"):
            warnings.append("challenger_review_contains_warnings")
        source = review.get("source_evidence")
        source = source if isinstance(source, Mapping) else {}
        if not _same_path(source.get("evaluation_dataset"), dataset_path):
            failures.append("challenger_review_dataset_scope_mismatch")

    baseline_metrics: dict[str, Any] = {}
    calibrated_metrics: dict[str, Any] = {}
    comparison: dict[str, Any] = {}
    clean_rows: list[dict[str, Any]] = []
    if dataset_path is None:
        failures.append("evaluation_dataset_path_missing")
    elif not dataset_path.exists():
        failures.append("evaluation_dataset_missing")
    elif alpha is not None and alpha > 0:
        rows = _load_jsonl(dataset_path)
        clean_rows = _clean_rows(rows)
        if not clean_rows:
            failures.append("clean_official_rows_zero")
        else:
            calibrated_rows = power_normalize_by_race(
                clean_rows,
                alpha=alpha,
                input_key="win_prob_norm",
                output_key=OUTPUT_KEY,
            )
            baseline_metrics = score_predictions(clean_rows)
            calibrated_metrics = score_predictions(
                calibrated_rows,
                probability_key=OUTPUT_KEY,
            )
            comparison = _comparison(calibrated_metrics, baseline_metrics)
            if comparison.get("log_loss_improved") is not True:
                failures.append("design_log_loss_not_improved")
            if comparison.get("brier_improved") is not True:
                failures.append("design_brier_not_improved")
            preserved = all(
                comparison.get(key) is True
                for key in (
                    "top1_preserved",
                    "top2_preserved",
                    "top3_preserved",
                    "mean_winner_rank_preserved",
                )
            )
            if not preserved:
                failures.append("design_ranking_not_preserved")

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "status": (
            "READY_FOR_OPERATOR_DESIGN_REVIEW" if not failures else "NOT_READY"
        ),
        "failures": failures,
        "warnings": warnings,
        "source_evidence": {
            "model_review_packet": str(packet_path),
            "challenger_review": str(challenger_review_path.resolve())
            if challenger_review_path
            else None,
            "evaluation_dataset": str(dataset_path.resolve()) if dataset_path else None,
            "clean_official_rows": len(clean_rows),
            "clean_official_races": len({row.get("race_id") for row in clean_rows}),
        },
        "runtime_transform_spec": {
            "candidate_arm": CANDIDATE_ARM,
            "algorithm": "power_normalize_per_race",
            "alpha": alpha,
            "input_probability_key": "win_prob_norm",
            "output_probability_key": OUTPUT_KEY,
            "formula": "p_cal_i = p_i ** alpha / sum_j(p_j ** alpha)",
            "rank_preserving_when_alpha_positive": alpha is not None and alpha > 0,
            "uses_labels_at_runtime": False,
            "uses_odds_at_runtime": False,
            "requires_runner_complete_race_group": True,
        },
        "baseline_model_metrics": baseline_metrics,
        "calibrated_report_only_metrics": calibrated_metrics,
        "comparison_to_baseline": comparison,
        "deployment_control": {
            "action_taken": "none",
            "model_artifact_written": False,
            "registry_mutation_allowed": False,
            "production_config_write_allowed": False,
            "promotion_allowed": False,
            "required_gate": "APPROVE_MODEL_PROMOTION",
            "betting_allowed": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-review-packet", required=True)
    parser.add_argument("--challenger-review")
    parser.add_argument("--dataset")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_design(
        model_review_packet_path=Path(args.model_review_packet),
        challenger_review_path=Path(args.challenger_review)
        if args.challenger_review
        else None,
        dataset_path=Path(args.dataset) if args.dataset else None,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report.get("status") == "READY_FOR_OPERATOR_DESIGN_REVIEW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
