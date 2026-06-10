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

SCHEMA_VERSION = "high_accuracy_refinement_packet_v2"
FINAL_READY_FOR_PR = "READY_FOR_PROMOTION_PR_DRAFT"
FINAL_NEEDS_MODEL_CHALLENGER = "NEEDS_NON_TGR_MODEL_CHALLENGER_TRAINING"
FINAL_BLOCKED = "BLOCKED_KEEP_BASELINE"

CALIBRATION_READY = "CHALLENGER_CALIBRATION_REPORT_ONLY_READY_FOR_REVIEW"
MODEL_CANDIDATE_KEY = "shadow_calibrated_rf_power_gamma_2_4"
MODEL_BASELINE_KEY = "champion_baseline"

STAGE2_FORWARD_SHADOW_COLLECTING = "STAGE2_FORWARD_SHADOW_COLLECTING"
STAGE2_FORWARD_SHADOW_READY_FOR_REVIEW = "STAGE2_FORWARD_SHADOW_READY_FOR_REVIEW"
ODDS_RESEARCH_BLOCKED_PROVENANCE = "ODDS_RESEARCH_BLOCKED_PROVENANCE"
ODDS_RESEARCH_READY_REPORT_ONLY = "ODDS_RESEARCH_READY_REPORT_ONLY"
ODDS_AUGMENTED_MODEL_BLOCKED = "ODDS_AUGMENTED_MODEL_BLOCKED"
ODDS_AUGMENTED_MODEL_READY_FOR_PR_REVIEW = "ODDS_AUGMENTED_MODEL_READY_FOR_PR_REVIEW"
MIN_COMPLETE_VALID_PREJUMP_ODDS_RACES = 100


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
    gate = gate_candidate(
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        thresholds=thresholds,
        source_status=source_status,
        accepted_source_statuses={ODDS_AUGMENTED_MODEL_READY_FOR_PR_REVIEW, "PASS", "READY"},
        stage="odds_augmented_model_research",
    )
    blockers = list(gate.get("blockers") or [])
    if odds_gate_status != ODDS_RESEARCH_READY_REPORT_ONLY:
        blockers.append(f"odds_research_gate_not_ready:{odds_gate_status or 'missing'}")
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
        "odds_research_gate_status": odds_gate_status or None,
        "source_final_status": source_status or None,
        "ev_metrics_used_for_promotion": False,
        "ev_can_override_accuracy_gate": False,
        "gate": gate,
    }


def promotion_pr_gate(
    *,
    stages: Sequence[Mapping[str, Any]],
    protected_paths_unchanged: bool,
) -> dict[str, Any]:
    passing = [stage for stage in stages if mapping(stage.get("gate")).get("status") == "PASS"]
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
    thresholds: AccuracyGateThresholds = AccuracyGateThresholds(),
    calibration_report_path: Path | None = None,
    stage2_forward_metrics_path: Path | None = None,
    shadow_replay_metrics_path: Path | None = None,
    odds_gate_report_path: Path | None = None,
    odds_augmented_report_path: Path | None = None,
    ev_diagnostics_path: Path | None = None,
    output_dir: Path | None = None,
    generated_at: datetime | None = None,
    protected_before: Mapping[str, str | None] | None = None,
    protected_after: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    protected_before = dict(protected_before or protected_hashes())
    protected_after = dict(protected_after or protected_hashes())
    stages = {
        "calibration_only": calibration_stage(calibration_report or {}, thresholds),
        "non_tgr_model_challenger": stage2_model_stage(
            stage2_forward_metrics,
            shadow_replay_metrics,
            thresholds,
        ),
        "odds_augmented_model": odds_augmented_stage(
            odds_gate_report,
            odds_augmented_report,
            thresholds,
        ),
    }
    stage_list = [stage for stage in stages.values() if stage]
    pr_gate = promotion_pr_gate(
        stages=stage_list,
        protected_paths_unchanged=protected_before == protected_after,
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
        "source_artifacts": {
            "calibration_report": relpath(calibration_report_path),
            "stage2_forward_metrics": relpath(stage2_forward_metrics_path),
            "shadow_replay_metrics": relpath(shadow_replay_metrics_path),
            "odds_research_gate_report": relpath(odds_gate_report_path),
            "odds_augmented_challenger_report": relpath(odds_augmented_report_path),
            "report_only_ev_diagnostics": relpath(ev_diagnostics_path),
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
    lines = [
        "# High Accuracy Refinement Packet",
        "",
        f"- Final status: `{packet.get('final_status')}`",
        f"- Promotion PR gate: `{pr_gate.get('status')}`",
        f"- Stage 2 status: `{mapping(stages.get('non_tgr_model_challenger')).get('status') or 'NOT_RUN'}`",
        f"- Odds research gate: `{mapping(packet.get('odds_research_gate_summary')).get('status') or 'NOT_RUN'}`",
        f"- Odds augmented status: `{mapping(stages.get('odds_augmented_model')).get('status') or 'NOT_RUN'}`",
        f"- Protected paths unchanged: `{packet.get('protected_paths_unchanged')}`",
        "",
        "No registry, DB, label, snapshot, EV action, betting action, TGR, or production pointer write was performed.",
        "",
    ]
    return "\n".join(lines)


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
    shadow_replay_metrics_path: Path | None = None,
    odds_gate_report_path: Path | None = None,
    odds_augmented_report_path: Path | None = None,
    ev_diagnostics_path: Path | None = None,
    output_dir: Path | None = None,
    thresholds: AccuracyGateThresholds = AccuracyGateThresholds(),
) -> dict[str, Any]:
    if not any(
        (
            calibration_report_path,
            stage2_forward_metrics_path,
            shadow_replay_metrics_path,
            odds_gate_report_path,
            odds_augmented_report_path,
        )
    ):
        raise ValueError("at_least_one_source_report_required")
    generated_at = datetime.now().astimezone()
    output_dir = output_dir or DEFAULT_OUTPUT_PARENT / f"high_accuracy_refinement_packet_{now_id(generated_at)}"
    output_dir = unique_dir(assert_output_dir_safe(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    protected_before = protected_hashes()
    stage2_forward_metrics = load_json(stage2_forward_metrics_path)
    packet = build_packet(
        calibration_report=load_json(calibration_report_path),
        stage2_forward_metrics=stage2_forward_metrics,
        shadow_replay_metrics=load_json(shadow_replay_metrics_path),
        odds_gate_report=load_json(odds_gate_report_path),
        odds_augmented_report=load_json(odds_augmented_report_path),
        ev_diagnostics=load_json(ev_diagnostics_path),
        thresholds=thresholds,
        calibration_report_path=calibration_report_path,
        stage2_forward_metrics_path=stage2_forward_metrics_path,
        shadow_replay_metrics_path=shadow_replay_metrics_path,
        odds_gate_report_path=odds_gate_report_path,
        odds_augmented_report_path=odds_augmented_report_path,
        ev_diagnostics_path=ev_diagnostics_path,
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
    parser.add_argument("--shadow-replay-metrics", type=Path)
    parser.add_argument("--odds-gate-report", type=Path)
    parser.add_argument("--odds-augmented-report", type=Path)
    parser.add_argument("--ev-diagnostics", type=Path)
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
        shadow_replay_metrics_path=args.shadow_replay_metrics,
        odds_gate_report_path=args.odds_gate_report,
        odds_augmented_report_path=args.odds_augmented_report,
        ev_diagnostics_path=args.ev_diagnostics,
        output_dir=args.output_dir,
        thresholds=thresholds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
