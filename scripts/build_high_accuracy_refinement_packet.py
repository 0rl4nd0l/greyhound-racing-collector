#!/usr/bin/env python3
"""Build the report-only high-accuracy refinement and PR gate packet.

The packet is intentionally evidence-only. It reads challenger reports, applies
rank-first safety gates, and writes a draft PR packet only when the evidence is
strong enough. It never mutates model artifacts, registry pointers, production
config, DB labels, snapshots, odds, EV, or betting outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

SCHEMA_VERSION = "high_accuracy_refinement_packet_v1"
FINAL_READY_FOR_PR = "READY_FOR_PROMOTION_PR_DRAFT"
FINAL_NEEDS_MODEL_CHALLENGER = "NEEDS_NON_TGR_MODEL_CHALLENGER_TRAINING"
FINAL_BLOCKED = "BLOCKED_KEEP_BASELINE"
CALIBRATION_READY = "CHALLENGER_CALIBRATION_REPORT_ONLY_READY_FOR_REVIEW"
MODEL_CANDIDATE_KEY = "shadow_calibrated_rf_power_gamma_2_4"
MODEL_BASELINE_KEY = "champion_baseline"


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


def sha256_file(path: Path) -> str | None:
    if not path.exists():
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


def race_count(metrics: Mapping[str, Any]) -> int:
    return finite_int(
        metrics.get("safe_joined_race_count")
        or metrics.get("race_count")
        or metrics.get("holdout_races")
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
    calibration = mapping(metrics.get("calibration_slope_intercept"))
    slope = finite_float(calibration.get("slope"))
    intercept = finite_float(calibration.get("intercept"))
    if slope is None or intercept is None:
        return None
    return abs(slope - 1.0) + abs(intercept)


def metric_deltas(
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
) -> dict[str, float | None]:
    baseline_calibration = calibration_distance(baseline_metrics)
    candidate_calibration = calibration_distance(candidate_metrics)
    deltas = {
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
    if baseline_calibration is not None and candidate_calibration is not None:
        deltas["calibration_distance"] = candidate_calibration - baseline_calibration
    return deltas


def gate_candidate(
    *,
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    thresholds: AccuracyGateThresholds,
    source_status: str | None,
    stage: str,
) -> dict[str, Any]:
    blockers: list[str] = []
    candidate_races = race_count(candidate_metrics)
    deltas = metric_deltas(baseline_metrics, candidate_metrics)
    candidate_box1 = metric(candidate_metrics, "box1_top_pick_share")
    baseline_box1 = metric(baseline_metrics, "box1_top_pick_share")
    candidate_sum_error = probability_sum_error(candidate_metrics)

    if source_status and source_status not in {CALIBRATION_READY, "PASS", "RUN", "READY"}:
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
        "primary_metric": "top1",
        "secondary_rank_metrics": ["top3", "mean_winner_rank"],
        "safety_metrics": ["brier", "logloss", "calibration_slope_intercept", "box1_top_pick_share"],
    }


def source_control_reasons(report: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if report.get("production_activation_allowed") is not False:
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
    ):
        if key in no_write and no_write.get(key) is not False:
            reasons.append(f"source_report_write_guard_not_blocked:{key}")
    return reasons


def calibration_stage(
    calibration_report: Mapping[str, Any],
    thresholds: AccuracyGateThresholds,
) -> dict[str, Any]:
    baseline = mapping(calibration_report.get("baseline_eval_metrics"))
    candidate = mapping(calibration_report.get("candidate_eval_metrics"))
    status = str(calibration_report.get("final_status") or "")
    gate = gate_candidate(
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        thresholds=thresholds,
        source_status=status,
        stage="stage_1_calibration_only",
    )
    source_reasons = source_control_reasons(calibration_report)
    if source_reasons:
        gate = dict(gate)
        gate["blockers"] = list(dict.fromkeys(list(gate.get("blockers") or []) + source_reasons))
        gate["status"] = "BLOCKED"
    rank_preserving = True
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
        "candidate_alpha": calibration_report.get("candidate_alpha"),
        "calibration_family": calibration_report.get("calibration_family"),
        "rank_preserving_by_construction": rank_preserving,
        "safe_exact_joined_race_count": calibration_report.get("safe_exact_joined_race_count"),
        "train_race_count": calibration_report.get("train_race_count"),
        "eval_race_count": calibration_report.get("eval_race_count"),
        "rejected_joined_races_excluded_count": len(calibration_report.get("rejected_joined_races") or []),
        "duplicate_joined_race_ids_seen_count": len(calibration_report.get("duplicate_joined_race_ids_seen") or []),
        "activation_blockers_from_source": list(calibration_report.get("activation_blockers") or []),
        "source_control_reasons": source_reasons,
        "gate": gate,
    }


def model_stage(
    shadow_replay_metrics: Mapping[str, Any] | None,
    thresholds: AccuracyGateThresholds,
) -> dict[str, Any] | None:
    if not shadow_replay_metrics:
        return None
    baseline = mapping(shadow_replay_metrics.get(MODEL_BASELINE_KEY))
    candidate = mapping(shadow_replay_metrics.get(MODEL_CANDIDATE_KEY))
    gate = gate_candidate(
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        thresholds=thresholds,
        source_status="PASS",
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
        "candidate_key": MODEL_CANDIDATE_KEY,
        "baseline_key": MODEL_BASELINE_KEY,
        "tgr_enabled": False,
        "odds_or_ev_used": False,
        "gate": gate,
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
        "odds_used_for_scoring": False,
        "ev_or_betting_output": False,
        "direct_switch": False,
    }


def promotion_pr_gate(
    *,
    calibration: Mapping[str, Any],
    model: Mapping[str, Any] | None,
    protected_paths_unchanged: bool,
) -> dict[str, Any]:
    candidates = [calibration]
    if model:
        candidates.append(model)
    passing = [
        item
        for item in candidates
        if mapping(item.get("gate")).get("status") == "PASS"
    ]
    blockers: list[str] = []
    if not passing:
        blockers.append("no_candidate_passed_rank_first_accuracy_gate")
    if not protected_paths_unchanged:
        blockers.append("protected_paths_changed")
    selected = passing[-1] if passing else None
    ready = bool(selected and not blockers)
    return {
        "schema_version": "promotion_pr_only_gate_v1",
        "status": "READY_FOR_PR_DRAFT" if ready else "BLOCKED",
        "selected_stage": selected.get("stage") if selected else None,
        "selected_candidate": selected.get("candidate_key") or selected.get("calibration_family") if selected else None,
        "blockers": blockers,
        "pull_request_boundary": {
            "promotion_pr_allowed": ready,
            "direct_local_switch_allowed": False,
            "local_registry_mutation_allowed": False,
            "production_pointer_update_allowed": False,
            "requires_human_pr_review": True,
        },
    }


def next_commands(
    *,
    calibration_report_path: Path | None,
    output_dir: Path | None,
) -> dict[str, Any]:
    output_arg = relpath(output_dir) if output_dir else (
        "artifacts/full_evidence_orchestration_20260525/"
        "high_accuracy_refinement_packet_<timestamp>"
    )
    calibration_arg = relpath(calibration_report_path) if calibration_report_path else (
        "artifacts/full_evidence_orchestration_20260525/"
        "forward_shadow_challenger_calibration_<timestamp>/challenger_calibration_report.json"
    )
    return {
        "stage_1_calibration": [
            "python",
            "scripts/run_forward_shadow_challenger_calibration.py",
            "--evidence-root",
            "artifacts/full_evidence_orchestration_20260525",
        ],
        "stage_2_non_tgr_model_challenger": [
            "python",
            "scripts/run_shadow_non_tgr_rf_evaluation.py",
            "run",
            "--output-dir",
            "artifacts/full_evidence_orchestration_20260525/shadow_evaluation_<timestamp>",
            "--all-missing-train-policy",
            "quarantine_feature",
        ],
        "build_refinement_packet": [
            "python",
            "scripts/build_high_accuracy_refinement_packet.py",
            "--calibration-report",
            calibration_arg,
            "--output-dir",
            output_arg,
        ],
    }


def build_packet(
    *,
    calibration_report: Mapping[str, Any],
    shadow_replay_metrics: Mapping[str, Any] | None = None,
    runtime_state: Mapping[str, Any] | None = None,
    thresholds: AccuracyGateThresholds = AccuracyGateThresholds(),
    calibration_report_path: Path | None = None,
    shadow_replay_metrics_path: Path | None = None,
    runtime_state_path: Path | None = None,
    output_dir: Path | None = None,
    generated_at: datetime | None = None,
    protected_before: Mapping[str, str | None] | None = None,
    protected_after: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    protected_before = dict(protected_before or protected_hashes())
    protected_after = dict(protected_after or protected_hashes())
    calibration = calibration_stage(calibration_report, thresholds)
    model = model_stage(shadow_replay_metrics, thresholds)
    pr_gate = promotion_pr_gate(
        calibration=calibration,
        model=model,
        protected_paths_unchanged=protected_before == protected_after,
    )
    if pr_gate["status"] == "READY_FOR_PR_DRAFT":
        final_status = FINAL_READY_FOR_PR
    elif calibration["source_final_status"] == CALIBRATION_READY:
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
                "exact_join_identity",
                "no_tgr",
                "no_odds_or_ev_scoring",
            ],
        },
        "thresholds": asdict(thresholds),
        "stages": {
            "calibration_only": calibration,
            "non_tgr_model_challenger": model,
        },
        "promotion_pr_gate": pr_gate,
        "runtime_state_summary": {
            "generated_at": mapping(runtime_state).get("generated_at"),
            "safe_joined_races": mapping(runtime_state).get("safe_joined_races"),
            "runtime_action": mapping(runtime_state).get("runtime_action"),
            "daemon_last_verdict": mapping(mapping(runtime_state).get("daemon")).get("last_verdict"),
            "daily_shadow_run_final_status": mapping(mapping(runtime_state).get("daily_shadow_run")).get("final_status"),
        },
        "source_artifacts": {
            "calibration_report": relpath(calibration_report_path),
            "shadow_replay_metrics": relpath(shadow_replay_metrics_path),
            "runtime_state": relpath(runtime_state_path),
        },
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "protected_paths_unchanged": protected_before == protected_after,
        "no_write_guarantees": no_write_guarantees(),
        "next_commands": next_commands(
            calibration_report_path=calibration_report_path,
            output_dir=output_dir,
        ),
    }


def metric_line(label: str, gate: Mapping[str, Any]) -> str:
    candidate = mapping(gate.get("candidate_metrics"))
    baseline = mapping(gate.get("baseline_metrics"))
    deltas = mapping(gate.get("candidate_minus_baseline"))
    values = []
    for key in ("top1", "top3", "mean_winner_rank", "brier", "logloss", "box1_top_pick_share"):
        values.append(
            f"{key}: baseline `{metric(baseline, key)}`, candidate `{metric(candidate, key)}`, delta `{deltas.get(key)}`"
        )
    return f"- {label}: " + "; ".join(values)


def build_pr_body(packet: Mapping[str, Any]) -> str:
    pr_gate = mapping(packet.get("promotion_pr_gate"))
    stages = mapping(packet.get("stages"))
    calibration = mapping(stages.get("calibration_only"))
    model = mapping(stages.get("non_tgr_model_challenger"))
    selected = pr_gate.get("selected_stage")
    lines = [
        "# Greyhound High-Accuracy Promotion Evidence",
        "",
        f"Gate status: `{pr_gate.get('status')}`",
        f"Final status: `{packet.get('final_status')}`",
        f"Selected stage: `{selected}`",
        "",
        "This is a PR-only promotion packet. It does not authorize a direct local switch.",
        "",
        "## Evidence",
        "",
        metric_line("Calibration-only", mapping(calibration.get("gate"))),
    ]
    if model:
        lines.append(metric_line("Non-TGR model challenger", mapping(model.get("gate"))))
    lines.extend(
        [
            "",
            "## Hard Boundaries",
            "",
            "- Direct local switch allowed: `False`",
            "- Local registry mutation allowed by packet builder: `False`",
            "- Production pointer update allowed by packet builder: `False`",
            "- TGR enabled: `False`",
            "- Odds/EV used for scoring: `False`",
            "- Unsafe joined races counted: `False`",
            "",
            "## Blockers",
            "",
        ]
    )
    blockers = list(pr_gate.get("blockers") or [])
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- None from this packet; human PR review is still required.")
    lines.append("")
    return "\n".join(lines)


def build_summary(packet: Mapping[str, Any]) -> str:
    pr_gate = mapping(packet.get("promotion_pr_gate"))
    calibration = mapping(mapping(packet.get("stages")).get("calibration_only"))
    model = mapping(mapping(packet.get("stages")).get("non_tgr_model_challenger"))
    lines = [
        "# High Accuracy Refinement Packet",
        "",
        f"- Final status: `{packet.get('final_status')}`",
        f"- Promotion PR gate: `{pr_gate.get('status')}`",
        f"- Calibration stage: `{calibration.get('status')}`",
        f"- Model stage: `{model.get('status') if model else 'NOT_RUN'}`",
        f"- Protected paths unchanged: `{packet.get('protected_paths_unchanged')}`",
        "",
        "No registry, DB, label, snapshot, EV, betting, or production pointer write was performed.",
        "",
    ]
    return "\n".join(lines)


def write_packet(output_dir: Path, packet: Mapping[str, Any]) -> None:
    write_json(output_dir / "high_accuracy_refinement_packet.json", packet)
    write_json(output_dir / "promotion_pr_gate.json", packet["promotion_pr_gate"])
    write_text(output_dir / "promotion_pr_body.md", build_pr_body(packet))
    write_text(output_dir / "SUMMARY.md", build_summary(packet))
    write_text(output_dir / "final_status.txt", str(packet["final_status"]) + "\n")
    write_text(
        output_dir / "next_commands.md",
        "\n".join(
            [
                "# Next Commands",
                "",
                "```bash",
                " ".join(packet["next_commands"]["stage_1_calibration"]),
                " ".join(packet["next_commands"]["stage_2_non_tgr_model_challenger"]),
                " ".join(packet["next_commands"]["build_refinement_packet"]),
                "```",
                "",
            ]
        ),
    )


def run_refinement_packet(
    *,
    calibration_report_path: Path,
    shadow_replay_metrics_path: Path | None = None,
    runtime_state_path: Path | None = None,
    output_dir: Path | None = None,
    thresholds: AccuracyGateThresholds = AccuracyGateThresholds(),
) -> dict[str, Any]:
    generated_at = datetime.now().astimezone()
    output_dir = output_dir or DEFAULT_OUTPUT_PARENT / f"high_accuracy_refinement_packet_{now_id(generated_at)}"
    output_dir = unique_dir(assert_output_dir_safe(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    protected_before = protected_hashes()
    packet = build_packet(
        calibration_report=load_json(calibration_report_path) or {},
        shadow_replay_metrics=load_json(shadow_replay_metrics_path),
        runtime_state=load_json(runtime_state_path),
        thresholds=thresholds,
        calibration_report_path=calibration_report_path,
        shadow_replay_metrics_path=shadow_replay_metrics_path,
        runtime_state_path=runtime_state_path,
        output_dir=output_dir,
        generated_at=generated_at,
        protected_before=protected_before,
    )
    write_packet(output_dir, packet)
    return {
        "output_dir": relpath(output_dir),
        "final_status": packet["final_status"],
        "promotion_pr_gate_status": packet["promotion_pr_gate"]["status"],
        "selected_stage": packet["promotion_pr_gate"]["selected_stage"],
        "protected_paths_unchanged": packet["protected_paths_unchanged"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--shadow-replay-metrics", type=Path)
    parser.add_argument("--runtime-state", type=Path)
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
        shadow_replay_metrics_path=args.shadow_replay_metrics,
        runtime_state_path=args.runtime_state,
        output_dir=args.output_dir,
        thresholds=thresholds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
