#!/usr/bin/env python3
"""Build a report-only model review packet from a snapshot evaluation report.

This script never trains, registers, promotes, or mutates a model. It exists to
turn a clean evaluation artifact into an auditable next-action packet for a
separate challenger review.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "model_review_packet_v1"
DEFAULT_MIN_CLEAN_RACES = 100


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _resolve_path(value: str | Path | None, *, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("report_root_not_object")
    return data


def _jsonl_row_count(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _model_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    clean_eval = report.get("clean_official_evaluation")
    if not isinstance(clean_eval, Mapping):
        return {}
    by_arm = clean_eval.get("metrics_by_arm")
    if not isinstance(by_arm, Mapping):
        return {}
    model_only = by_arm.get("model_only")
    return dict(model_only) if isinstance(model_only, Mapping) else {}


def _same_resolved_path(left: str | Path | None, right: str | Path | None) -> bool:
    if left is None or right is None:
        return False
    return Path(left).resolve() == Path(right).resolve()


def _challenger_review_gate(
    challenger_review_path: Path | None,
    *,
    dataset_path: Path | None,
) -> dict[str, Any]:
    gate: dict[str, Any] = {
        "provided": challenger_review_path is not None,
        "path": str(challenger_review_path.resolve()) if challenger_review_path else None,
        "status": "NOT_PROVIDED",
        "failures": [],
        "warnings": [],
        "candidate_arm": None,
        "stability_status": None,
        "promotion_allowed": False,
        "registry_mutation_allowed": False,
        "model_artifact_written": False,
    }
    if challenger_review_path is None:
        return gate
    failures: list[str] = []
    warnings: list[str] = []
    if not challenger_review_path.exists():
        failures.append("challenger_review_missing")
        gate["status"] = "NOT_READY"
        gate["failures"] = failures
        return gate

    try:
        review = _json(challenger_review_path)
    except Exception as exc:
        failures.append(f"challenger_review_unreadable:{type(exc).__name__}")
        gate["status"] = "NOT_READY"
        gate["failures"] = failures
        return gate

    source_evidence = review.get("source_evidence")
    stability = review.get("stability_review")
    promotion_control = review.get("promotion_control")
    challenger_training = review.get("challenger_training")
    source_evidence = source_evidence if isinstance(source_evidence, Mapping) else {}
    stability = stability if isinstance(stability, Mapping) else {}
    promotion_control = (
        promotion_control if isinstance(promotion_control, Mapping) else {}
    )
    challenger_training = (
        challenger_training if isinstance(challenger_training, Mapping) else {}
    )
    power_calibration = challenger_training.get("power_calibration")
    power_calibration = (
        power_calibration if isinstance(power_calibration, Mapping) else {}
    )

    if review.get("schema_version") != "snapshot_challenger_review_v1":
        failures.append("challenger_review_schema_mismatch")
    if review.get("status") != "SUCCESS":
        failures.append("challenger_review_status_not_success")
    if dataset_path is None:
        failures.append("evaluation_dataset_path_missing_for_challenger_review")
    elif not _same_resolved_path(source_evidence.get("evaluation_dataset"), dataset_path):
        failures.append("challenger_review_dataset_scope_mismatch")
    if stability.get("status") != "STABLE_REPORT_ONLY":
        failures.append("challenger_stability_not_stable_report_only")
    if stability.get("candidate_arm") != "power_calibrated_baseline":
        failures.append("challenger_stability_candidate_mismatch")
    if int(stability.get("failed_split_count") or 0) != 0:
        failures.append("challenger_stability_failed_splits")
    if int(stability.get("split_count") or 0) < int(
        stability.get("minimum_split_count") or 2
    ):
        failures.append("challenger_stability_insufficient_splits")
    if stability.get("all_log_loss_improved") is not True:
        failures.append("challenger_stability_log_loss_not_consistently_improved")
    if stability.get("all_brier_improved") is not True:
        failures.append("challenger_stability_brier_not_consistently_improved")
    if stability.get("all_ranking_preserved") is not True:
        failures.append("challenger_stability_ranking_not_preserved")
    if promotion_control.get("promotion_allowed") is not False:
        failures.append("challenger_review_promotion_not_blocked")
    if promotion_control.get("registry_mutation_allowed") is not False:
        failures.append("challenger_review_registry_mutation_allowed")
    if promotion_control.get("model_artifact_written") is not False:
        failures.append("challenger_review_model_artifact_written")
    if power_calibration.get("model_artifact_written") is not False:
        failures.append("power_calibration_model_artifact_written")
    if power_calibration.get("registry_mutation_allowed") is not False:
        failures.append("power_calibration_registry_mutation_allowed")
    if stability.get("promotion_allowed") is not False:
        failures.append("challenger_stability_promotion_not_blocked")
    if review.get("failures"):
        failures.append("challenger_review_contains_failures")
    if review.get("warnings"):
        warnings.append("challenger_review_contains_warnings")

    gate.update(
        {
            "status": "READY" if not failures else "NOT_READY",
            "failures": failures,
            "warnings": warnings,
            "candidate_arm": stability.get("candidate_arm"),
            "stability_status": stability.get("status"),
            "split_count": stability.get("split_count"),
            "failed_split_count": stability.get("failed_split_count"),
            "all_log_loss_improved": stability.get("all_log_loss_improved"),
            "all_brier_improved": stability.get("all_brier_improved"),
            "all_ranking_preserved": stability.get("all_ranking_preserved"),
            "selected_alpha": power_calibration.get("selected_alpha"),
            "promotion_allowed": promotion_control.get("promotion_allowed") is True,
            "registry_mutation_allowed": (
                promotion_control.get("registry_mutation_allowed") is True
            ),
            "model_artifact_written": (
                promotion_control.get("model_artifact_written") is True
            ),
        }
    )
    return gate


def build_packet(
    *,
    evaluation_report_path: Path,
    dataset_path: Path | None = None,
    challenger_review_path: Path | None = None,
    min_clean_races: int = DEFAULT_MIN_CLEAN_RACES,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    report_path = evaluation_report_path.resolve()
    report = _json(report_path)
    root = (repo_root or Path.cwd()).resolve()
    if dataset_path is None:
        dataset_path = _resolve_path(
            report.get("evaluation_dataset_output"),
            base_dir=root,
        )
    else:
        dataset_path = dataset_path.resolve()

    clean_eval = report.get("clean_official_evaluation")
    model_quality = report.get("model_quality_diagnosis")
    if not isinstance(clean_eval, Mapping):
        clean_eval = {}
    if not isinstance(model_quality, Mapping):
        model_quality = {}

    retrain_gate = model_quality.get("retrain_gate")
    promotion_gate = model_quality.get("promotion_gate")
    retrain_gate = dict(retrain_gate) if isinstance(retrain_gate, Mapping) else {}
    promotion_gate = dict(promotion_gate) if isinstance(promotion_gate, Mapping) else {}

    clean_races = _safe_int(clean_eval.get("races_evaluated"))
    clean_snapshots = _safe_int(clean_eval.get("snapshot_instances_evaluated"))
    clean_rows = _safe_int(clean_eval.get("runner_rows_evaluated"))
    dataset_rows_written = _safe_int(report.get("evaluation_dataset_rows_written"))
    dataset_rows_observed = _jsonl_row_count(dataset_path)

    failures: list[str] = []
    warnings: list[str] = []
    if report.get("status") != "SUCCESS":
        failures.append("evaluation_report_status_not_success")
    if clean_races < min_clean_races:
        failures.append("insufficient_clean_official_races")
    if clean_snapshots < min_clean_races:
        failures.append("insufficient_clean_snapshot_instances")
    if clean_rows <= 0:
        failures.append("clean_runner_rows_zero")
    if dataset_path is None:
        failures.append("evaluation_dataset_path_missing")
    elif not dataset_path.exists():
        failures.append("evaluation_dataset_missing")
    elif dataset_rows_observed != dataset_rows_written:
        failures.append("evaluation_dataset_row_count_mismatch")
    if model_quality.get("status") != "SUCCESS":
        failures.append("model_quality_diagnosis_not_success")
    if retrain_gate.get("status") != "READY_FOR_REVIEW":
        failures.append("retrain_gate_not_ready_for_review")
    if retrain_gate.get("action_taken") not in (None, "none"):
        failures.append("retrain_action_already_taken")
    if promotion_gate.get("action_taken") not in (None, "none"):
        failures.append("promotion_action_already_taken")
    if promotion_gate.get("status") != "REPORT_ONLY":
        warnings.append("promotion_gate_not_report_only")

    challenger_review_gate = _challenger_review_gate(
        challenger_review_path,
        dataset_path=dataset_path,
    )
    if challenger_review_gate["provided"] and challenger_review_gate["status"] != "READY":
        failures.append("challenger_review_gate_not_ready")

    ready = not failures
    metrics = _model_metrics(report)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "status": "READY_FOR_CHALLENGER_REVIEW" if ready else "NOT_READY",
        "failures": failures,
        "warnings": warnings,
        "source_evidence": {
            "evaluation_report": str(report_path),
            "evaluation_dataset": str(dataset_path) if dataset_path else None,
            "evaluation_dataset_rows_written": dataset_rows_written,
            "evaluation_dataset_rows_observed": dataset_rows_observed,
        },
        "review_gate": {
            "minimum_clean_evaluated_races": min_clean_races,
            "clean_official_evaluated_races": clean_races,
            "clean_official_snapshot_instances": clean_snapshots,
            "clean_official_runner_rows": clean_rows,
            "retrain_gate": retrain_gate,
            "promotion_gate": promotion_gate,
        },
        "baseline_model_metrics": {
            "top1": metrics.get("top1"),
            "top2": metrics.get("top2"),
            "top3": metrics.get("top3"),
            "log_loss": metrics.get("log_loss"),
            "brier": metrics.get("brier"),
            "mean_winner_rank": metrics.get("mean_winner_rank"),
            "races_evaluated": metrics.get("races_evaluated"),
            "dog_predictions_evaluated": metrics.get("dog_predictions_evaluated"),
        },
        "promotion_control": {
            "action_taken": "none",
            "registry_mutation_allowed": False,
            "promotion_allowed": False,
            "reason": (
                "challenger training/evaluation must run separately and beat this "
                "baseline on clean held-out evidence before any promotion approval"
            ),
        },
        "challenger_review_gate": challenger_review_gate,
        "next_review_steps": [
            {
                "name": "train_challenger_in_isolated_staging",
                "status": "REQUIRES_SEPARATE_OPERATOR_REVIEW",
                "must_not_use": "live production DB or current prediction snapshots as labels",
            },
            {
                "name": "evaluate_challenger_against_clean_holdout",
                "status": "REQUIRED_BEFORE_PROMOTION",
                "baseline_report": str(report_path),
            },
            {
                "name": "calibration_layer_design",
                "status": (
                    "READY_FOR_SEPARATE_DESIGN_REVIEW"
                    if challenger_review_gate["status"] == "READY"
                    else "WAITING_FOR_STABLE_REPORT_ONLY_CHALLENGER_REVIEW"
                ),
                "source": challenger_review_gate["path"],
                "candidate_arm": challenger_review_gate["candidate_arm"],
                "must_not_write": "model artifact, model registry, production config, or betting surface",
            },
            {
                "name": "promotion",
                "status": "BLOCKED",
                "required_gate": "APPROVE_MODEL_PROMOTION",
                "additional_requirement": "challenger metrics beat baseline",
            },
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-report", required=True)
    parser.add_argument("--dataset-output")
    parser.add_argument("--challenger-review")
    parser.add_argument("--output")
    parser.add_argument("--min-clean-races", type=int, default=DEFAULT_MIN_CLEAN_RACES)
    parser.add_argument("--repo-root", default=str(Path.cwd()))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    packet = build_packet(
        evaluation_report_path=Path(args.evaluation_report),
        dataset_path=Path(args.dataset_output) if args.dataset_output else None,
        challenger_review_path=(
            Path(args.challenger_review) if args.challenger_review else None
        ),
        min_clean_races=args.min_clean_races,
        repo_root=Path(args.repo_root),
    )
    text = json.dumps(packet, indent=2, sort_keys=True)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if packet.get("status") == "READY_FOR_CHALLENGER_REVIEW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
