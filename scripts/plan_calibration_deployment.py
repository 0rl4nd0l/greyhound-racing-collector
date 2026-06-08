#!/usr/bin/env python3
"""Build a no-write calibration deployment plan from approved evidence.

This script validates the report-only model review packet, calibration design,
and pre-jump loop plan. It intentionally does not emit a runnable promotion
command and does not mutate model artifacts, registry state, production config,
labels, refresh signals, or betting surfaces.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "calibration_deployment_plan_v1"
READY_STATUS = "READY_FOR_SEPARATE_PROMOTION_IMPLEMENTATION_REVIEW"
MODEL_PACKET_SCHEMA = "model_review_packet_v1"
CALIBRATION_DESIGN_SCHEMA = "calibration_layer_design_v1"
REQUIRED_GATE = "APPROVE_MODEL_PROMOTION"
MIN_CLEAN_OFFICIAL_RACES = 100
CANDIDATE_ARM = "power_calibrated_baseline"
OUTPUT_KEY = "calibrated_win_prob_report_only"

WRITE_BLOCKS = {
    "label_write": False,
    "model_artifact_write": False,
    "registry_mutation": False,
    "production_config_write": False,
    "refresh_signal_write": False,
    "betting": False,
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("json_root_not_object")
    return payload


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _action_none(value: Any) -> bool:
    return value in (None, "none")


def _is_false(mapping: Mapping[str, Any], key: str) -> bool:
    return mapping.get(key) is False


def _is_true(mapping: Mapping[str, Any], key: str) -> bool:
    return mapping.get(key) is True


def _has_required_gate(items: list[Any], gate: str) -> bool:
    for item in items:
        if isinstance(item, Mapping) and item.get("required_gate") == gate:
            return True
    return False


def _same_path(left: str | Path, right: Path) -> bool:
    left_path = Path(str(left))
    return left_path.resolve() == right.resolve() or str(left) == str(right)


def _command_has_design(command: Any, design_path: Path) -> bool:
    if not isinstance(command, list):
        return False
    for index, token in enumerate(command):
        if token != "--report-only-calibration-design":
            continue
        if index + 1 >= len(command):
            return False
        return _same_path(command[index + 1], design_path)
    return False


def _step_command(loop_plan: Mapping[str, Any], name: str) -> Any:
    for step in _list(loop_plan.get("steps")):
        if isinstance(step, Mapping) and step.get("name") == name:
            return step.get("command")
    return None


def _command_matches_any(commands: list[Any], design_path: Path) -> bool:
    return any(_command_has_design(command, design_path) for command in commands)


def _append_failure(condition: bool, failures: list[str], name: str) -> None:
    if condition:
        failures.append(name)


def _validate_model_packet(
    packet: Mapping[str, Any],
    *,
    selected_alpha: float | None,
    failures: list[str],
) -> dict[str, Any]:
    review_gate = _mapping(packet.get("review_gate"))
    promotion_control = _mapping(packet.get("promotion_control"))
    challenger_gate = _mapping(packet.get("challenger_review_gate"))
    clean_races = _safe_int(review_gate.get("clean_official_evaluated_races"))
    minimum_races = _safe_int(
        review_gate.get("minimum_clean_evaluated_races")
        or MIN_CLEAN_OFFICIAL_RACES
    )

    _append_failure(
        packet.get("schema_version") != MODEL_PACKET_SCHEMA,
        failures,
        "model_review_packet_schema_mismatch",
    )
    _append_failure(
        packet.get("status") != "READY_FOR_CHALLENGER_REVIEW",
        failures,
        "model_review_packet_not_ready",
    )
    _append_failure(bool(packet.get("failures")), failures, "model_review_packet_contains_failures")
    _append_failure(
        clean_races < max(minimum_races, MIN_CLEAN_OFFICIAL_RACES),
        failures,
        "model_review_packet_insufficient_clean_official_races",
    )
    _append_failure(
        not _action_none(promotion_control.get("action_taken")),
        failures,
        "model_review_packet_promotion_action_already_taken",
    )
    _append_failure(
        not _is_false(promotion_control, "promotion_allowed"),
        failures,
        "model_review_packet_promotion_not_blocked",
    )
    _append_failure(
        not _is_false(promotion_control, "registry_mutation_allowed"),
        failures,
        "model_review_packet_registry_mutation_not_blocked",
    )

    review_steps = _list(packet.get("steps")) + _list(packet.get("next_review_steps"))
    _append_failure(
        not _has_required_gate(review_steps, REQUIRED_GATE),
        failures,
        "model_review_packet_required_gate_missing",
    )

    _append_failure(
        challenger_gate.get("status") != "READY",
        failures,
        "challenger_review_gate_not_ready",
    )
    _append_failure(
        challenger_gate.get("candidate_arm") != CANDIDATE_ARM,
        failures,
        "challenger_review_candidate_mismatch",
    )
    _append_failure(
        challenger_gate.get("stability_status") != "STABLE_REPORT_ONLY",
        failures,
        "challenger_review_not_stable_report_only",
    )
    _append_failure(
        challenger_gate.get("all_log_loss_improved") is not True,
        failures,
        "challenger_review_log_loss_not_improved",
    )
    _append_failure(
        challenger_gate.get("all_brier_improved") is not True,
        failures,
        "challenger_review_brier_not_improved",
    )
    _append_failure(
        challenger_gate.get("all_ranking_preserved") is not True,
        failures,
        "challenger_review_ranking_not_preserved",
    )
    _append_failure(
        not _is_false(challenger_gate, "promotion_allowed"),
        failures,
        "challenger_review_promotion_not_blocked",
    )
    _append_failure(
        not _is_false(challenger_gate, "registry_mutation_allowed"),
        failures,
        "challenger_review_registry_mutation_not_blocked",
    )
    _append_failure(
        not _is_false(challenger_gate, "model_artifact_written"),
        failures,
        "challenger_review_model_artifact_written",
    )

    packet_alpha = _safe_float(challenger_gate.get("selected_alpha"))
    _append_failure(
        packet_alpha is None or packet_alpha <= 0,
        failures,
        "challenger_review_selected_alpha_missing_or_invalid",
    )
    if packet_alpha is not None and selected_alpha is not None:
        _append_failure(
            abs(packet_alpha - selected_alpha) > 1e-12,
            failures,
            "model_review_packet_alpha_mismatch",
        )

    return {
        "clean_official_evaluated_races": clean_races,
        "minimum_clean_evaluated_races": minimum_races,
        "candidate_arm": challenger_gate.get("candidate_arm"),
        "stability_status": challenger_gate.get("stability_status"),
        "selected_alpha": packet_alpha,
    }


def _validate_calibration_design(
    design: Mapping[str, Any],
    *,
    failures: list[str],
) -> dict[str, Any]:
    transform = _mapping(design.get("runtime_transform_spec"))
    comparison = _mapping(design.get("comparison_to_baseline"))
    deployment = _mapping(design.get("deployment_control"))
    source = _mapping(design.get("source_evidence"))
    alpha = _safe_float(transform.get("alpha"))

    _append_failure(
        design.get("schema_version") != CALIBRATION_DESIGN_SCHEMA,
        failures,
        "calibration_schema_mismatch",
    )
    _append_failure(
        design.get("status") != "READY_FOR_OPERATOR_DESIGN_REVIEW",
        failures,
        "calibration_design_not_ready",
    )
    _append_failure(bool(design.get("failures")), failures, "calibration_design_contains_failures")
    _append_failure(
        transform.get("candidate_arm") != CANDIDATE_ARM,
        failures,
        "calibration_candidate_mismatch",
    )
    _append_failure(
        transform.get("algorithm") != "power_normalize_per_race",
        failures,
        "calibration_algorithm_mismatch",
    )
    _append_failure(alpha is None or alpha <= 0, failures, "calibration_alpha_invalid")
    _append_failure(
        transform.get("input_probability_key") != "win_prob_norm",
        failures,
        "calibration_input_probability_key_mismatch",
    )
    _append_failure(
        transform.get("output_probability_key") != OUTPUT_KEY,
        failures,
        "calibration_output_probability_key_mismatch",
    )
    _append_failure(
        not _is_true(transform, "rank_preserving_when_alpha_positive"),
        failures,
        "calibration_not_rank_preserving",
    )
    _append_failure(
        not _is_false(transform, "uses_labels_at_runtime"),
        failures,
        "calibration_uses_labels_at_runtime",
    )
    _append_failure(
        not _is_false(transform, "uses_odds_at_runtime"),
        failures,
        "calibration_uses_odds_at_runtime",
    )
    _append_failure(
        not _is_true(transform, "requires_runner_complete_race_group"),
        failures,
        "calibration_does_not_require_complete_race_group",
    )

    for key, failure in (
        ("log_loss_improved", "calibration_log_loss_not_improved"),
        ("brier_improved", "calibration_brier_not_improved"),
        ("top1_preserved", "calibration_top1_not_preserved"),
        ("top2_preserved", "calibration_top2_not_preserved"),
        ("top3_preserved", "calibration_top3_not_preserved"),
        (
            "mean_winner_rank_preserved",
            "calibration_mean_winner_rank_not_preserved",
        ),
    ):
        _append_failure(comparison.get(key) is not True, failures, failure)

    _append_failure(
        not _action_none(deployment.get("action_taken")),
        failures,
        "calibration_action_already_taken",
    )
    _append_failure(
        not _is_false(deployment, "model_artifact_written"),
        failures,
        "calibration_model_artifact_written",
    )
    _append_failure(
        not _is_false(deployment, "registry_mutation_allowed"),
        failures,
        "calibration_registry_mutation_not_blocked",
    )
    _append_failure(
        not _is_false(deployment, "production_config_write_allowed"),
        failures,
        "calibration_production_config_write_not_blocked",
    )
    _append_failure(
        not _is_false(deployment, "promotion_allowed"),
        failures,
        "calibration_promotion_not_blocked",
    )
    _append_failure(
        deployment.get("required_gate") != REQUIRED_GATE,
        failures,
        "calibration_required_gate_mismatch",
    )
    _append_failure(
        not _is_false(deployment, "betting_allowed"),
        failures,
        "calibration_betting_not_blocked",
    )

    clean_races = _safe_int(source.get("clean_official_races"))
    if "clean_official_races" in source:
        _append_failure(
            clean_races < MIN_CLEAN_OFFICIAL_RACES,
            failures,
            "calibration_source_insufficient_clean_official_races",
        )

    return {
        "candidate_arm": transform.get("candidate_arm"),
        "algorithm": transform.get("algorithm"),
        "alpha": alpha,
        "input_probability_key": transform.get("input_probability_key"),
        "output_probability_key": transform.get("output_probability_key"),
        "clean_official_races": clean_races if "clean_official_races" in source else None,
        "clean_official_rows": source.get("clean_official_rows"),
    }


def _validate_loop_plan(
    loop_plan: Mapping[str, Any],
    *,
    design_path: Path,
    failures: list[str],
) -> dict[str, Any]:
    promotion_gate = _mapping(loop_plan.get("promotion_readiness_gate"))
    guarantees = _mapping(loop_plan.get("guarantees"))
    persist_packet = _mapping(loop_plan.get("persist_approval_packet"))
    live_odds_packet = _mapping(loop_plan.get("live_odds_approval_packet"))

    _append_failure(
        promotion_gate.get("status") != "APPROVAL_PRESENT_EVIDENCE_READY_REPORT_ONLY",
        failures,
        "loop_promotion_readiness_status_not_ready_report_only",
    )
    _append_failure(
        promotion_gate.get("ready_for_separate_promotion_review") is not True,
        failures,
        "loop_not_ready_for_separate_promotion_review",
    )
    _append_failure(
        not _action_none(promotion_gate.get("promotion_action_taken")),
        failures,
        "loop_promotion_action_already_taken",
    )
    _append_failure(
        not _is_false(promotion_gate, "promotion_allowed_by_loop"),
        failures,
        "loop_promotion_not_blocked",
    )
    _append_failure(
        not _is_false(promotion_gate, "registry_mutation_allowed_by_loop"),
        failures,
        "loop_registry_mutation_not_blocked",
    )
    _append_failure(
        not _is_false(promotion_gate, "model_artifact_write_allowed_by_loop"),
        failures,
        "loop_model_artifact_write_not_blocked",
    )
    _append_failure(
        not _is_false(promotion_gate, "betting_allowed_by_loop"),
        failures,
        "loop_betting_not_blocked",
    )
    _append_failure(
        promotion_gate.get("required_gate") != REQUIRED_GATE,
        failures,
        "loop_required_gate_mismatch",
    )
    _append_failure(
        _safe_int(promotion_gate.get("promotion_evidence_clean_official_evaluated_races"))
        < MIN_CLEAN_OFFICIAL_RACES,
        failures,
        "loop_insufficient_promotion_evidence_clean_official_races",
    )

    for key, failure in (
        ("no_model_promotion", "loop_no_model_promotion_guarantee_missing"),
        ("no_retrain", "loop_no_retrain_guarantee_missing"),
        ("no_betting", "loop_no_betting_guarantee_missing"),
    ):
        _append_failure(guarantees.get(key) is not True, failures, failure)

    dry_run_command = _step_command(loop_plan, "dry_run_prejump_capture")
    approved_persist_command = _step_command(loop_plan, "approved_persist_ready_subset")
    direct_persist_commands = [
        approved_persist_command,
        persist_packet.get("planned_persist_command"),
        live_odds_packet.get("combined_persist_live_odds_command"),
    ]
    direct_live_odds_commands = [
        live_odds_packet.get("planned_odds_command"),
        live_odds_packet.get("combined_persist_live_odds_command"),
    ]
    persist_same_run_command = persist_packet.get(
        "approved_same_run_execute_ready_command_template"
    )
    live_odds_same_run_command = live_odds_packet.get(
        "approved_same_run_execute_ready_command_template"
    )

    pass_through = {
        "dry_run_capture": _command_has_design(dry_run_command, design_path),
        "approved_persist_capture": _command_matches_any(
            direct_persist_commands,
            design_path,
        ),
        "live_odds_capture": _command_matches_any(
            direct_live_odds_commands,
            design_path,
        ),
        "persist_same_run": _command_has_design(persist_same_run_command, design_path),
        "live_odds_same_run": _command_has_design(
            live_odds_same_run_command,
            design_path,
        ),
    }
    for key, value in pass_through.items():
        _append_failure(
            value is not True,
            failures,
            f"loop_missing_report_only_calibration_pass_through:{key}",
        )
    return pass_through


def build_plan(
    *,
    model_review_packet_path: Path,
    calibration_design_path: Path,
    loop_plan_path: Path,
) -> dict[str, Any]:
    packet_path = model_review_packet_path.resolve()
    design_path = calibration_design_path.resolve()
    plan_path = loop_plan_path.resolve()
    failures: list[str] = []
    warnings: list[str] = []

    try:
        packet = _load_json(packet_path)
    except Exception as exc:
        packet = {}
        failures.append(f"model_review_packet_unreadable:{type(exc).__name__}")
    try:
        design = _load_json(design_path)
    except Exception as exc:
        design = {}
        failures.append(f"calibration_design_unreadable:{type(exc).__name__}")
    try:
        loop_plan = _load_json(plan_path)
    except Exception as exc:
        loop_plan = {}
        failures.append(f"loop_plan_unreadable:{type(exc).__name__}")

    calibration_summary = _validate_calibration_design(
        design,
        failures=failures,
    )
    model_summary = _validate_model_packet(
        packet,
        selected_alpha=calibration_summary.get("alpha"),
        failures=failures,
    )
    loop_pass_through = _validate_loop_plan(
        loop_plan,
        design_path=design_path,
        failures=failures,
    )

    ready = not failures
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "status": READY_STATUS if ready else "NOT_READY",
        "failures": failures,
        "warnings": warnings,
        "source_evidence": {
            "model_review_packet": str(packet_path),
            "calibration_design": str(design_path),
            "prejump_loop_plan": str(plan_path),
            "model_review_clean_official_evaluated_races": model_summary.get(
                "clean_official_evaluated_races"
            ),
            "minimum_clean_official_races": MIN_CLEAN_OFFICIAL_RACES,
            "calibration_design_clean_official_races": calibration_summary.get(
                "clean_official_races"
            ),
        },
        "selected_runtime_transform": calibration_summary,
        "model_review_gate": model_summary,
        "loop_pass_through": loop_pass_through,
        "deployment_controls": {
            "required_gate": REQUIRED_GATE,
            "next_required_gate": "APPROVE_MODEL_PROMOTION_FOR_SEPARATE_EXECUTOR",
            "actual_promotion_command_ready": False,
            "promotion_allowed": False,
            "model_artifact_write_allowed": False,
            "registry_mutation_allowed": False,
            "production_config_write_allowed": False,
            "betting_allowed": False,
            "reason": (
                "This plan validates report-only promotion evidence. A separate "
                "implementation executor must be designed and reviewed before "
                "any model/config/registry write can exist."
            ),
        },
        "actual_promotion_command": None,
        "writes_performed": dict(WRITE_BLOCKS),
        "remaining_implementation_work": [
            "design a separate gated runtime calibration executor",
            "define exact model/config/registry write targets and rollback path",
            "run a fresh no-write validation immediately before any write-capable execution",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-review-packet", required=True)
    parser.add_argument("--calibration-design", required=True)
    parser.add_argument("--loop-plan", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_plan(
        model_review_packet_path=Path(args.model_review_packet),
        calibration_design_path=Path(args.calibration_design),
        loop_plan_path=Path(args.loop_plan),
    )
    text = json.dumps(plan, indent=2, sort_keys=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if plan.get("status") == READY_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
