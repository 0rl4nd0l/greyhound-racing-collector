#!/usr/bin/env python3
"""Read-only forward-shadow runtime state packet.

This script answers operational questions such as "are we waiting?", "how many
safe joins do we have?", and "when is the next useful refresh?" from existing
daemon/runtime artifacts. It does not refresh races, score predictions, join
results, train, promote, mutate registries, write DB rows, write labels, enable
TGR, or emit betting/EV outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from accuracy_program.odds_coverage import (
    summarize_read_only_odds_coverage_report,
)

DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
DEFAULT_DAEMON_STATE = (
    DEFAULT_EVIDENCE_ROOT / "shadow_autopilot_daemon_runtime/state.json"
)
OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/forward_shadow_runtime_state_"
DEFAULT_TARGET_JOINED_RACES = 100
NO_WRITE_GUARANTEES = {
    "race_refresh": False,
    "shadow_prediction_write": False,
    "result_join": False,
    "training": False,
    "production_promotion": False,
    "registry_mutation": False,
    "production_pointer_update": False,
    "active_model_replacement": False,
    "db_write": False,
    "label_write": False,
    "tgr_enabled": False,
    "betting_or_ev_output": False,
    "production_prediction_write": False,
}


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
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def rooted_path(value: object) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def artifact_final_status(directory: Path | None) -> str | None:
    if directory is None:
        return None
    path = directory / "final_status.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def systemctl_status(unit: str) -> str | None:
    try:
        result = subprocess.run(
            ["systemctl", "is-active", unit],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return None
    return (result.stdout.strip() or result.stderr.strip() or None)


def systemd_timer_snapshot() -> dict[str, Any]:
    service_status = systemctl_status("shadow-autopilot.service")
    timer_status = systemctl_status("shadow-autopilot.timer")
    snapshot: dict[str, Any] = {
        "service_status": service_status,
        "timer_status": timer_status,
    }
    try:
        result = subprocess.run(
            ["systemctl", "list-timers", "shadow-autopilot.timer", "--no-pager", "--all"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except Exception:
        snapshot["list_timers_status"] = "UNAVAILABLE"
        return snapshot
    snapshot.update(
        {
            "list_timers_status": "PASS" if result.returncode == 0 else "FAIL",
            "list_timers_stdout": result.stdout,
            "list_timers_stderr": result.stderr,
        }
    )
    return snapshot


def daily_child_summary(daily_dir: Path | None) -> dict[str, Any]:
    manifest = load_json(daily_dir / "shadow_manifest.json") if daily_dir else None
    classification_report = (
        load_json(daily_dir / "malformed_or_stale_inputs.json") if daily_dir else None
    )
    metadata = load_json(daily_dir / "prejump_metadata_report.json") if daily_dir else None
    provenance = (
        load_json(daily_dir / "same_distance_same_grade_history_provenance.json")
        if daily_dir
        else None
    )
    classification = (
        (manifest or {}).get("input_classification")
        or (manifest or {}).get("input_summary")
        or classification_report
        or {}
    )
    metadata_readiness = (metadata or {}).get("target_metadata_readiness") or {}
    return {
        "daily_shadow_run_dir": relpath(daily_dir),
        "final_status": artifact_final_status(daily_dir),
        "scanned_csv_count": classification.get("scanned_csv_count"),
        "eligible_count": classification.get("eligible_count"),
        "stale_count": classification.get("stale_count"),
        "malformed_count": classification.get("malformed_count"),
        "target_metadata_readiness": {
            "status": metadata_readiness.get("status")
            or metadata_readiness.get("target_metadata_capture_status"),
            "capture_status": metadata_readiness.get("target_metadata_capture_status"),
            "blocker_counts": metadata_readiness.get("blocker_counts") or {},
        },
        "same_distance_history_provenance": {
            "status": (provenance or {}).get("status"),
            "live_input_status": (provenance or {}).get("live_input_status"),
            "target_race_rows_allowed": (provenance or {}).get("target_race_rows_allowed"),
            "post_outcome_rows_allowed": (provenance or {}).get("post_outcome_rows_allowed"),
        },
    }


def latest_daily_from_output(daemon_output_dir: Path | None) -> Path | None:
    provenance = load_json(
        daemon_output_dir / "prediction_provenance_report.json"
        if daemon_output_dir
        else None
    )
    if provenance:
        daily = rooted_path(provenance.get("daily_shadow_run_dir"))
        if daily and daily.exists():
            return daily
    manifest = load_json(daemon_output_dir / "run_manifest.json" if daemon_output_dir else None)
    if manifest:
        daily = rooted_path(
            ((manifest.get("source_artifacts") or {}).get("daily_shadow_run_dir"))
        )
        if daily and daily.exists():
            return daily
    return None


def feature_activation_gate_summary(daemon_output_dir: Path | None) -> dict[str, Any]:
    dashboard = load_json(daemon_output_dir / "shadow_dashboard.json" if daemon_output_dir else None) or {}
    gate = dashboard.get("feature_activation_gate")
    if not isinstance(gate, Mapping):
        gate = {}
    output_dir = rooted_path(gate.get("output_dir"))
    gate_report = load_json(output_dir / "feature_activation_gate_report.json" if output_dir else None) or {}
    return {
        "status": gate.get("status") or gate_report.get("final_status"),
        "output_dir": relpath(output_dir),
        "status_path": gate.get("status_path"),
        "provenance_audit": gate.get("provenance_audit"),
        "activation_allowed_features": gate.get("activation_allowed_features")
        or gate_report.get("activation_allowed_features")
        or [],
        "kept_quarantined_features": gate.get("kept_quarantined_features")
        or gate_report.get("kept_quarantined_features")
        or [],
        "fail_reason_summary": gate_report.get("fail_reason_summary") or {},
    }


def latest_odds_coverage_report(
    evidence_root: Path,
    daemon_output_dir: Path | None,
) -> Path | None:
    if daemon_output_dir:
        current = daemon_output_dir / "odds_coverage_report.json"
        if current.exists():
            return current
    candidates = [
        item / "odds_coverage_report.json"
        for item in evidence_root.glob("shadow_autopilot_daemonization_v1_*")
        if item.is_dir() and (item / "odds_coverage_report.json").exists()
    ]
    return sorted(candidates)[-1] if candidates else None


def decide_runtime_action(
    *,
    daemon_state: Mapping[str, Any],
    timer: Mapping[str, Any] | None,
    target_joined_races: int,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    service_status = (timer or {}).get("service_status")
    if service_status in {"active", "activating"}:
        return "DAEMON_RUNNING_WAIT_FOR_CYCLE", ["shadow_autopilot_service_active"]

    verdict = daemon_state.get("last_verdict")
    if verdict not in {"DAEMON_READY", "AUTOPILOT_READY", None}:
        reasons.append("daemon_not_ready")
        return "CHECK_DAEMON_FAILURE", reasons

    refresh_status = daemon_state.get("last_next_prejump_refresh_status")
    if refresh_status == "WAITING_FOR_FUTURE_WINDOW":
        reasons.append("next_race_outside_preferred_window")
        return "WAIT_UNTIL_RECOMMENDED_REFRESH", reasons

    safe_joined = int(daemon_state.get("last_safe_joined_races") or 0)
    if safe_joined < target_joined_races:
        reasons.append("safe_joined_race_count_below_target")
        return "CONTINUE_FORWARD_SHADOW_COLLECTION", reasons
    return "READY_FOR_FORWARD_SHADOW_REVIEW_REPORT_ONLY", []


def build_runtime_state(
    *,
    evidence_root: Path = DEFAULT_EVIDENCE_ROOT,
    daemon_state_path: Path = DEFAULT_DAEMON_STATE,
    timer: Mapping[str, Any] | None = None,
    target_joined_races: int = DEFAULT_TARGET_JOINED_RACES,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    daemon_state = load_json(daemon_state_path) or {}
    daemon_output_dir = rooted_path(daemon_state.get("last_output_dir"))
    daily_dir = latest_daily_from_output(daemon_output_dir)
    odds_coverage_path = latest_odds_coverage_report(evidence_root, daemon_output_dir)
    odds_coverage = summarize_read_only_odds_coverage_report(
        load_json(odds_coverage_path)
    )
    action, reasons = decide_runtime_action(
        daemon_state=daemon_state,
        timer=timer,
        target_joined_races=target_joined_races,
    )
    safe_joined = int(daemon_state.get("last_safe_joined_races") or 0)
    return {
        "schema_version": "forward_shadow_runtime_state_v1",
        "generated_at": generated_at.isoformat(),
        "runtime_action": action,
        "runtime_action_reasons": reasons,
        "target_joined_races": target_joined_races,
        "safe_joined_races": safe_joined,
        "safe_joined_races_remaining": max(target_joined_races - safe_joined, 0),
        "daemon": {
            "state_path": relpath(daemon_state_path),
            "last_output_dir": relpath(daemon_output_dir),
            "last_run_id": daemon_state.get("last_run_id"),
            "last_verdict": daemon_state.get("last_verdict"),
            "last_cycle_activity_status": daemon_state.get("last_cycle_activity_status"),
            "last_safe_joined_delta": daemon_state.get("last_safe_joined_delta"),
            "last_next_prejump_refresh_status": daemon_state.get(
                "last_next_prejump_refresh_status"
            ),
            "last_recommended_rerun_after_local": daemon_state.get(
                "last_recommended_rerun_after_local"
            ),
            "last_next_prejump_race": daemon_state.get("last_next_prejump_race"),
            "last_autonomous_live_odds_next_window_opens_at": daemon_state.get(
                "last_autonomous_live_odds_next_window_opens_at"
            ),
            "last_autonomous_live_odds_recommended_rerun_after_local": daemon_state.get(
                "last_autonomous_live_odds_recommended_rerun_after_local"
            ),
            "last_autonomous_live_odds_next_race_id": daemon_state.get(
                "last_autonomous_live_odds_next_race_id"
            ),
            "last_odds_capture_next_meaningful_action": daemon_state.get(
                "last_odds_capture_next_meaningful_action"
            ),
            "last_odds_capture_next_meaningful_action_at": daemon_state.get(
                "last_odds_capture_next_meaningful_action_at"
            ),
            "last_autonomous_live_odds_next_prejump_window": daemon_state.get(
                "last_autonomous_live_odds_next_prejump_window"
            ),
            "last_autonomous_official_result_capture_status": daemon_state.get(
                "last_autonomous_official_result_capture_status"
            ),
            "last_autonomous_official_result_capture_attempted": daemon_state.get(
                "last_autonomous_official_result_capture_attempted"
            ),
            "last_autonomous_official_result_race_rows": daemon_state.get(
                "last_autonomous_official_result_race_rows"
            ),
            "last_autonomous_official_result_runner_rows": daemon_state.get(
                "last_autonomous_official_result_runner_rows"
            ),
            "last_autonomous_official_result_quarantine_rows": daemon_state.get(
                "last_autonomous_official_result_quarantine_rows"
            ),
            "last_autonomous_official_result_quarantined_race_ids": (
                daemon_state.get("last_autonomous_official_result_quarantined_race_ids")
                or []
            ),
            "last_autonomous_official_result_quarantine_reason_counts": (
                daemon_state.get(
                    "last_autonomous_official_result_quarantine_reason_counts"
                )
                or {}
            ),
            "last_autonomous_official_result_quarantine_error_counts": (
                daemon_state.get(
                    "last_autonomous_official_result_quarantine_error_counts"
                )
                or {}
            ),
            "last_autonomous_official_result_quarantine_result_boxes_not_in_participants_counts": (
                daemon_state.get(
                    "last_autonomous_official_result_quarantine_result_boxes_not_in_participants_counts"
                )
                or {}
            ),
            "last_autonomous_official_result_quarantine_runner_set_mismatch_samples": (
                daemon_state.get(
                    "last_autonomous_official_result_quarantine_runner_set_mismatch_samples"
                )
                or []
            ),
            "last_autonomous_official_result_skipped_reason_counts": (
                daemon_state.get("last_autonomous_official_result_skipped_reason_counts")
                or {}
            ),
            "last_autonomous_official_result_awaiting_jump_race_count": daemon_state.get(
                "last_autonomous_official_result_awaiting_jump_race_count"
            ),
            "last_autonomous_official_result_awaiting_jump_race_ids": (
                daemon_state.get("last_autonomous_official_result_awaiting_jump_race_ids")
                or []
            ),
            "last_autonomous_official_result_awaiting_jump_next_recheck_after_local": (
                daemon_state.get(
                    "last_autonomous_official_result_awaiting_jump_next_recheck_after_local"
                )
            ),
            "last_autonomous_official_result_evidence_db_ingest_status": (
                daemon_state.get(
                    "last_autonomous_official_result_evidence_db_ingest_status"
                )
            ),
            "last_autonomous_official_result_evidence_db_execute": daemon_state.get(
                "last_autonomous_official_result_evidence_db_execute"
            ),
            "last_autonomous_official_result_evidence_db_write_performed": daemon_state.get(
                "last_autonomous_official_result_evidence_db_write_performed"
            ),
            "last_autonomous_official_result_evidence_valid_race_rows": daemon_state.get(
                "last_autonomous_official_result_evidence_valid_race_rows"
            ),
            "last_autonomous_official_result_evidence_valid_runner_rows": daemon_state.get(
                "last_autonomous_official_result_evidence_valid_runner_rows"
            ),
            "last_autonomous_official_result_evidence_blocked_race_rows": daemon_state.get(
                "last_autonomous_official_result_evidence_blocked_race_rows"
            ),
            "last_autonomous_official_result_evidence_blocked_runner_rows": daemon_state.get(
                "last_autonomous_official_result_evidence_blocked_runner_rows"
            ),
            "last_autonomous_official_result_evidence_inserted_race_rows": daemon_state.get(
                "last_autonomous_official_result_evidence_inserted_race_rows"
            ),
            "last_autonomous_official_result_evidence_inserted_runner_rows": daemon_state.get(
                "last_autonomous_official_result_evidence_inserted_runner_rows"
            ),
            "last_autonomous_official_result_evidence_blocker_reason_counts": (
                daemon_state.get(
                    "last_autonomous_official_result_evidence_blocker_reason_counts"
                )
                or {}
            ),
            "last_autonomous_official_result_evidence_inserted_rows": daemon_state.get(
                "last_autonomous_official_result_evidence_inserted_rows"
            ),
            "last_live_odds_backlog_unresolved_race_count": daemon_state.get(
                "last_live_odds_backlog_unresolved_race_count"
            ),
            "last_live_odds_backlog_unresolved_reason_counts": daemon_state.get(
                "last_live_odds_backlog_unresolved_reason_counts"
            )
            or {},
            "last_live_odds_backlog_unresolved_recovery_action_counts": (
                daemon_state.get(
                    "last_live_odds_backlog_unresolved_recovery_action_counts"
                )
                or {}
            ),
            "last_live_odds_backlog_unresolved_alias_status_counts": daemon_state.get(
                "last_live_odds_backlog_unresolved_alias_status_counts"
            )
            or {},
            "last_live_odds_backlog_retryable_exact_shadow_match_race_count": (
                daemon_state.get(
                    "last_live_odds_backlog_retryable_exact_shadow_match_race_count"
                )
            ),
            "last_live_odds_backlog_no_exact_shadow_match_race_count": daemon_state.get(
                "last_live_odds_backlog_no_exact_shadow_match_race_count"
            ),
            "last_live_odds_backlog_retryable_exact_shadow_match_race_ids": (
                daemon_state.get(
                    "last_live_odds_backlog_retryable_exact_shadow_match_race_ids"
                )
                or []
            ),
            "last_live_odds_backlog_no_exact_shadow_match_race_ids": daemon_state.get(
                "last_live_odds_backlog_no_exact_shadow_match_race_ids"
            )
            or [],
            "last_live_odds_backlog_recovery_queue_path": daemon_state.get(
                "last_live_odds_backlog_recovery_queue_path"
            ),
            "last_live_odds_backlog_recovery_queue_diagnostic_only": daemon_state.get(
                "last_live_odds_backlog_recovery_queue_diagnostic_only"
            ),
            "last_live_odds_backlog_recovery_queue_join_acceptance_changed": (
                daemon_state.get(
                    "last_live_odds_backlog_recovery_queue_join_acceptance_changed"
                )
            ),
            "last_live_odds_backlog_recovery_queue_db_write_performed": (
                daemon_state.get("last_live_odds_backlog_recovery_queue_db_write_performed")
            ),
            "last_live_odds_backlog_awaiting_official_result_evidence_race_count": (
                daemon_state.get(
                    "last_live_odds_backlog_awaiting_official_result_evidence_race_count"
                )
            ),
            "last_live_odds_backlog_awaiting_official_result_evidence_race_ids": (
                daemon_state.get(
                    "last_live_odds_backlog_awaiting_official_result_evidence_race_ids"
                )
                or []
            ),
            "last_live_odds_backlog_awaiting_official_result_evidence_authorized_action": (
                daemon_state.get(
                    "last_live_odds_backlog_awaiting_official_result_evidence_authorized_action"
                )
            ),
            "last_live_odds_backlog_awaiting_official_result_recheck_ready_race_count": (
                daemon_state.get(
                    "last_live_odds_backlog_awaiting_official_result_recheck_ready_race_count"
                )
            ),
            "last_live_odds_backlog_runner_set_validation_path": daemon_state.get(
                "last_live_odds_backlog_runner_set_validation_path"
            ),
            "last_live_odds_backlog_runner_set_validation_retryable_race_count": (
                daemon_state.get(
                    "last_live_odds_backlog_runner_set_validation_retryable_race_count"
                )
            ),
            "last_live_odds_backlog_runner_set_validation_exact_match_race_count": (
                daemon_state.get(
                    "last_live_odds_backlog_runner_set_validation_exact_match_race_count"
                )
            ),
            "last_live_odds_backlog_runner_set_validation_blocked_race_count": (
                daemon_state.get(
                    "last_live_odds_backlog_runner_set_validation_blocked_race_count"
                )
            ),
            "last_live_odds_backlog_runner_set_validation_diagnostic_only": (
                daemon_state.get(
                    "last_live_odds_backlog_runner_set_validation_diagnostic_only"
                )
            ),
            "last_live_odds_backlog_runner_set_validation_join_authorized": (
                daemon_state.get("last_live_odds_backlog_runner_set_validation_join_authorized")
            ),
            "last_live_odds_backlog_runner_set_validation_db_write_performed": (
                daemon_state.get(
                    "last_live_odds_backlog_runner_set_validation_db_write_performed"
                )
            ),
            "last_live_odds_backlog_join_eligibility_packet_path": daemon_state.get(
                "last_live_odds_backlog_join_eligibility_packet_path"
            ),
            "last_live_odds_backlog_join_eligibility_evaluated_race_count": (
                daemon_state.get(
                    "last_live_odds_backlog_join_eligibility_evaluated_race_count"
                )
            ),
            "last_live_odds_backlog_join_eligibility_eligible_report_only_race_count": (
                daemon_state.get(
                    "last_live_odds_backlog_join_eligibility_eligible_report_only_race_count"
                )
            ),
            "last_live_odds_backlog_join_eligibility_blocked_race_count": (
                daemon_state.get(
                    "last_live_odds_backlog_join_eligibility_blocked_race_count"
                )
            ),
            "last_live_odds_backlog_join_eligibility_diagnostic_only": daemon_state.get(
                "last_live_odds_backlog_join_eligibility_diagnostic_only"
            ),
            "last_live_odds_backlog_join_eligibility_join_authorized": daemon_state.get(
                "last_live_odds_backlog_join_eligibility_join_authorized"
            ),
            "last_live_odds_backlog_join_eligibility_db_write_performed": (
                daemon_state.get(
                    "last_live_odds_backlog_join_eligibility_db_write_performed"
                )
            ),
            "last_live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count": (
                daemon_state.get(
                    "last_live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count"
                )
            ),
            "last_rolling_model_comparison_status": daemon_state.get(
                "last_rolling_model_comparison_status"
            ),
            "last_rolling_model_comparison_sample_races": daemon_state.get(
                "last_rolling_model_comparison_sample_races"
            ),
            "last_rolling_model_comparison_best_candidate": daemon_state.get(
                "last_rolling_model_comparison_best_candidate"
            ),
            "last_rolling_model_comparison_source_rejected_live_odds_candidate_count": (
                daemon_state.get(
                    "last_rolling_model_comparison_source_rejected_live_odds_candidate_count"
                )
            ),
            "last_rolling_model_comparison_source_rows_with_rejected_live_odds_candidates": (
                daemon_state.get(
                    "last_rolling_model_comparison_source_rows_with_rejected_live_odds_candidates"
                )
            ),
            "last_rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts": (
                daemon_state.get(
                    "last_rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts"
                )
                or {}
            ),
            "last_high_accuracy_refinement_status": daemon_state.get(
                "last_high_accuracy_refinement_status"
            ),
            "last_high_accuracy_promotion_pr_gate_status": daemon_state.get(
                "last_high_accuracy_promotion_pr_gate_status"
            ),
            "last_high_accuracy_unified_evidence_eligible_rows": daemon_state.get(
                "last_high_accuracy_unified_evidence_eligible_rows"
            ),
            "last_promotion_distance_status": daemon_state.get(
                "last_promotion_distance_status"
            ),
            "last_promotion_distance_promotion_ready": daemon_state.get(
                "last_promotion_distance_promotion_ready"
            ),
            "last_promotion_distance_sample_race_count": daemon_state.get(
                "last_promotion_distance_sample_race_count"
            ),
            "last_promotion_distance_sample_runner_rows": daemon_state.get(
                "last_promotion_distance_sample_runner_rows"
            ),
            "last_promotion_distance_best_candidate_key": daemon_state.get(
                "last_promotion_distance_best_candidate_key"
            ),
            "last_promotion_distance_best_non_market_candidate_key": daemon_state.get(
                "last_promotion_distance_best_non_market_candidate_key"
            ),
            "last_promotion_distance_blockers": daemon_state.get(
                "last_promotion_distance_blockers"
            )
            or [],
            "last_promotion_distance_source_rejected_live_odds_candidate_count": (
                daemon_state.get(
                    "last_promotion_distance_source_rejected_live_odds_candidate_count"
                )
            ),
            "last_promotion_distance_source_rows_with_rejected_live_odds_candidates": (
                daemon_state.get(
                    "last_promotion_distance_source_rows_with_rejected_live_odds_candidates"
                )
            ),
            "last_promotion_distance_source_rejected_live_odds_candidate_reason_counts": (
                daemon_state.get(
                    "last_promotion_distance_source_rejected_live_odds_candidate_reason_counts"
                )
                or {}
            ),
            "last_promotion_distance_source_exclusion_reason_counts": (
                daemon_state.get("last_promotion_distance_source_exclusion_reason_counts")
                or {}
            ),
            "last_promotion_distance_source_odds_exclusion_reason_counts": (
                daemon_state.get(
                    "last_promotion_distance_source_odds_exclusion_reason_counts"
                )
                or {}
            ),
            "last_promotion_distance_source_official_result_evidence_db_missing_race_ids": (
                daemon_state.get(
                    "last_promotion_distance_source_official_result_evidence_db_missing_race_ids"
                )
                or []
            ),
            "last_promotion_distance_source_official_result_evidence_db_requested_race_count": (
                daemon_state.get(
                    "last_promotion_distance_source_official_result_evidence_db_requested_race_count"
                )
            ),
            "last_promotion_distance_source_official_result_evidence_db_races_with_rows": (
                daemon_state.get(
                    "last_promotion_distance_source_official_result_evidence_db_races_with_rows"
                )
                or []
            ),
            "last_promotion_distance_source_official_result_runner_paths": (
                daemon_state.get(
                    "last_promotion_distance_source_official_result_runner_paths"
                )
                or []
            ),
            "last_promotion_distance_official_result_coverage_requested_race_count": (
                daemon_state.get(
                    "last_promotion_distance_official_result_coverage_requested_race_count"
                )
            ),
            "last_promotion_distance_official_result_coverage_requested_race_count_source": (
                daemon_state.get(
                    "last_promotion_distance_official_result_coverage_requested_race_count_source"
                )
            ),
            "last_promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids": (
                daemon_state.get(
                    "last_promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids"
                )
            ),
            "last_promotion_distance_official_result_coverage_races_with_rows_count": (
                daemon_state.get(
                    "last_promotion_distance_official_result_coverage_races_with_rows_count"
                )
            ),
            "last_promotion_distance_official_result_coverage_missing_race_count": (
                daemon_state.get(
                    "last_promotion_distance_official_result_coverage_missing_race_count"
                )
            ),
            "last_promotion_distance_official_result_coverage_missing_exclusion_count": (
                daemon_state.get(
                    "last_promotion_distance_official_result_coverage_missing_exclusion_count"
                )
            ),
            "last_promotion_distance_official_result_runner_path_count": (
                daemon_state.get(
                    "last_promotion_distance_official_result_runner_path_count"
                )
            ),
            "last_promotion_distance_official_result_runner_paths_source_field": (
                daemon_state.get(
                    "last_promotion_distance_official_result_runner_paths_source_field"
                )
            ),
            "last_feature_activation_gate_status": daemon_state.get(
                "last_feature_activation_gate_status"
            ),
            "last_shadow_odds_snapshot_status": daemon_state.get(
                "last_shadow_odds_snapshot_status"
            ),
            "last_prejump_metadata_status": daemon_state.get("last_prejump_metadata_status"),
            "last_best_aggregate_unified_evidence_status": daemon_state.get(
                "last_best_aggregate_unified_evidence_status"
            ),
            "last_best_aggregate_unified_evidence_eligible_rows": daemon_state.get(
                "last_best_aggregate_unified_evidence_eligible_rows"
            ),
            "last_best_aggregate_unified_rejected_live_odds_candidate_count": (
                daemon_state.get(
                    "last_best_aggregate_unified_rejected_live_odds_candidate_count"
                )
            ),
            "last_best_aggregate_unified_rows_with_rejected_live_odds_candidates": (
                daemon_state.get(
                    "last_best_aggregate_unified_rows_with_rejected_live_odds_candidates"
                )
            ),
            "last_best_aggregate_unified_rejected_live_odds_candidate_reason_counts": (
                daemon_state.get(
                    "last_best_aggregate_unified_rejected_live_odds_candidate_reason_counts"
                )
                or {}
            ),
            "last_rejoin_unified_evidence_status": daemon_state.get(
                "last_rejoin_unified_evidence_status"
            ),
            "last_rejoin_unified_rejected_live_odds_candidate_count": daemon_state.get(
                "last_rejoin_unified_rejected_live_odds_candidate_count"
            ),
            "last_rejoin_unified_rows_with_rejected_live_odds_candidates": daemon_state.get(
                "last_rejoin_unified_rows_with_rejected_live_odds_candidates"
            ),
            "last_rejoin_unified_rejected_live_odds_candidate_reason_counts": (
                daemon_state.get(
                    "last_rejoin_unified_rejected_live_odds_candidate_reason_counts"
                )
                or {}
            ),
            "last_rejoin_rolling_model_comparison_status": daemon_state.get(
                "last_rejoin_rolling_model_comparison_status"
            ),
            "last_rejoin_rolling_model_comparison_sample_races": daemon_state.get(
                "last_rejoin_rolling_model_comparison_sample_races"
            ),
            "last_rejoin_high_accuracy_refinement_status": daemon_state.get(
                "last_rejoin_high_accuracy_refinement_status"
            ),
            "last_rejoin_pre_race_gated_challenger_status": daemon_state.get(
                "last_rejoin_pre_race_gated_challenger_status"
            ),
            "last_rejoin_pre_race_gated_challenger_promotion_ready": daemon_state.get(
                "last_rejoin_pre_race_gated_challenger_promotion_ready"
            ),
            "last_rejoin_rank_first_hypothesis_review_status": daemon_state.get(
                "last_rejoin_rank_first_hypothesis_review_status"
            ),
            "last_rejoin_rank_first_hypothesis_candidate_count": daemon_state.get(
                "last_rejoin_rank_first_hypothesis_candidate_count"
            ),
            "last_rejoin_rank_first_hypothesis_evaluated_candidate_count": (
                daemon_state.get(
                    "last_rejoin_rank_first_hypothesis_evaluated_candidate_count"
                )
            ),
            "last_rejoin_rank_first_hypothesis_best_candidate_key": daemon_state.get(
                "last_rejoin_rank_first_hypothesis_best_candidate_key"
            ),
            "last_rejoin_rank_first_hypothesis_directional_read_ready": daemon_state.get(
                "last_rejoin_rank_first_hypothesis_directional_read_ready"
            ),
            "last_rejoin_time_split_gated_challenger_status": daemon_state.get(
                "last_rejoin_time_split_gated_challenger_status"
            ),
            "last_rejoin_time_split_gated_challenger_promotion_ready": daemon_state.get(
                "last_rejoin_time_split_gated_challenger_promotion_ready"
            ),
            "last_rejoin_market_residual_challenger_status": daemon_state.get(
                "last_rejoin_market_residual_challenger_status"
            ),
            "last_rejoin_market_residual_challenger_promotion_ready": daemon_state.get(
                "last_rejoin_market_residual_challenger_promotion_ready"
            ),
            "last_rejoin_market_residual_regime_audit_status": daemon_state.get(
                "last_rejoin_market_residual_regime_audit_status"
            ),
            "last_rejoin_market_residual_regime_audit_promotion_ready": (
                daemon_state.get(
                    "last_rejoin_market_residual_regime_audit_promotion_ready"
                )
            ),
            "last_rejoin_market_residual_rank_first_hypothesis_status": (
                daemon_state.get(
                    "last_rejoin_market_residual_rank_first_hypothesis_status"
                )
            ),
            "last_rejoin_rank_first_hypothesis_watchlist_status": daemon_state.get(
                "last_rejoin_rank_first_hypothesis_watchlist_status"
            ),
            "last_rejoin_rank_first_hypothesis_watchlist_candidate_count": (
                daemon_state.get(
                    "last_rejoin_rank_first_hypothesis_watchlist_candidate_count"
                )
            ),
            "last_rejoin_rank_first_hypothesis_watchlist_directional_ready_candidate_count": (
                daemon_state.get(
                    "last_rejoin_rank_first_hypothesis_watchlist_directional_ready_candidate_count"
                )
            ),
            "last_rejoin_rank_first_hypothesis_watchlist_best_candidate": (
                daemon_state.get(
                    "last_rejoin_rank_first_hypothesis_watchlist_best_candidate"
                )
            ),
            "last_rejoin_rank_first_hypothesis_watchlist_best_status": daemon_state.get(
                "last_rejoin_rank_first_hypothesis_watchlist_best_status"
            ),
            "last_rejoin_rank_first_hypothesis_watchlist_best_distinct_samples": (
                daemon_state.get(
                    "last_rejoin_rank_first_hypothesis_watchlist_best_distinct_samples"
                )
            ),
            "last_rejoin_promotion_distance_status": daemon_state.get(
                "last_rejoin_promotion_distance_status"
            ),
            "last_rejoin_promotion_distance_promotion_ready": daemon_state.get(
                "last_rejoin_promotion_distance_promotion_ready"
            ),
            "last_rejoin_promotion_distance_blockers": daemon_state.get(
                "last_rejoin_promotion_distance_blockers"
            )
            or [],
            "last_rejoin_promotion_distance_source_exclusion_reason_counts": (
                daemon_state.get(
                    "last_rejoin_promotion_distance_source_exclusion_reason_counts"
                )
                or {}
            ),
            "last_rejoin_promotion_distance_source_odds_exclusion_reason_counts": (
                daemon_state.get(
                    "last_rejoin_promotion_distance_source_odds_exclusion_reason_counts"
                )
                or {}
            ),
            "last_rejoin_promotion_distance_source_official_result_evidence_db_missing_race_ids": (
                daemon_state.get(
                    "last_rejoin_promotion_distance_source_official_result_evidence_db_missing_race_ids"
                )
                or []
            ),
            "last_rejoin_promotion_distance_source_official_result_evidence_db_requested_race_count": (
                daemon_state.get(
                    "last_rejoin_promotion_distance_source_official_result_evidence_db_requested_race_count"
                )
            ),
            "last_rejoin_promotion_distance_source_official_result_evidence_db_races_with_rows": (
                daemon_state.get(
                    "last_rejoin_promotion_distance_source_official_result_evidence_db_races_with_rows"
                )
                or []
            ),
            "last_rejoin_promotion_distance_source_official_result_runner_paths": (
                daemon_state.get(
                    "last_rejoin_promotion_distance_source_official_result_runner_paths"
                )
                or []
            ),
            "last_rejoin_promotion_distance_official_result_coverage_requested_race_count": (
                daemon_state.get(
                    "last_rejoin_promotion_distance_official_result_coverage_requested_race_count"
                )
            ),
            "last_rejoin_promotion_distance_official_result_coverage_requested_race_count_source": (
                daemon_state.get(
                    "last_rejoin_promotion_distance_official_result_coverage_requested_race_count_source"
                )
            ),
            "last_rejoin_promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids": (
                daemon_state.get(
                    "last_rejoin_promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids"
                )
            ),
            "last_rejoin_promotion_distance_official_result_coverage_races_with_rows_count": (
                daemon_state.get(
                    "last_rejoin_promotion_distance_official_result_coverage_races_with_rows_count"
                )
            ),
            "last_rejoin_promotion_distance_official_result_coverage_missing_race_count": (
                daemon_state.get(
                    "last_rejoin_promotion_distance_official_result_coverage_missing_race_count"
                )
            ),
            "last_rejoin_promotion_distance_official_result_coverage_missing_exclusion_count": (
                daemon_state.get(
                    "last_rejoin_promotion_distance_official_result_coverage_missing_exclusion_count"
                )
            ),
            "last_rejoin_promotion_distance_official_result_runner_path_count": (
                daemon_state.get(
                    "last_rejoin_promotion_distance_official_result_runner_path_count"
                )
            ),
            "last_rejoin_promotion_distance_official_result_runner_paths_source_field": (
                daemon_state.get(
                    "last_rejoin_promotion_distance_official_result_runner_paths_source_field"
                )
            ),
            "updated_at": daemon_state.get("updated_at"),
        },
        "daemon_output": {
            "final_status": artifact_final_status(daemon_output_dir),
            "shadow_status_path": relpath(daemon_output_dir / "SHADOW_STATUS.md")
            if daemon_output_dir
            else None,
            "prediction_provenance_report": relpath(
                daemon_output_dir / "prediction_provenance_report.json"
            )
            if daemon_output_dir
            else None,
            "odds_coverage_report": relpath(odds_coverage_path),
        },
        "daily_shadow_run": daily_child_summary(daily_dir),
        "feature_activation_gate": feature_activation_gate_summary(daemon_output_dir),
        "shadow_odds_coverage": odds_coverage,
        "timer": dict(timer or {}),
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def build_summary(report: Mapping[str, Any]) -> str:
    daemon = report.get("daemon") or {}
    daily = report.get("daily_shadow_run") or {}
    next_race = daemon.get("last_next_prejump_race") or {}
    odds_next_window = daemon.get("last_autonomous_live_odds_next_prejump_window") or {}
    if not isinstance(odds_next_window, Mapping):
        odds_next_window = {}
    odds_next_race = odds_next_window.get("next_race") or {}
    if not isinstance(odds_next_race, Mapping):
        odds_next_race = {}
    odds_next_race_id = (
        daemon.get("last_autonomous_live_odds_next_race_id")
        or odds_next_race.get("race_id")
    )
    return "\n".join(
        [
            "# Forward Shadow Runtime State",
            "",
            f"- Runtime action: `{report.get('runtime_action')}`",
            f"- Reasons: `{report.get('runtime_action_reasons')}`",
            f"- Safe joined races: `{report.get('safe_joined_races')}` / `{report.get('target_joined_races')}`",
            f"- Remaining to target: `{report.get('safe_joined_races_remaining')}`",
            f"- Daemon verdict: `{daemon.get('last_verdict')}`",
            f"- Cycle activity: `{daemon.get('last_cycle_activity_status')}`",
            f"- Next pre-jump status: `{daemon.get('last_next_prejump_refresh_status')}`",
            f"- Recommended rerun after: `{daemon.get('last_recommended_rerun_after_local')}`",
            f"- Next race: `{next_race.get('race_id')}` at `{next_race.get('jump_datetime')}`",
            f"- Autonomous odds next window opens: `{daemon.get('last_autonomous_live_odds_next_window_opens_at')}`",
            f"- Autonomous odds rerun after: `{daemon.get('last_autonomous_live_odds_recommended_rerun_after_local')}`",
            f"- Autonomous odds next race: `{odds_next_race_id}`",
            f"- Autonomous odds next action: `{daemon.get('last_odds_capture_next_meaningful_action')}` at `{daemon.get('last_odds_capture_next_meaningful_action_at')}`",
            f"- Autonomous official result capture: `{daemon.get('last_autonomous_official_result_capture_status')}`",
            f"- Autonomous official result capture attempted: `{daemon.get('last_autonomous_official_result_capture_attempted')}`",
            f"- Autonomous official result race rows: `{daemon.get('last_autonomous_official_result_race_rows')}`",
            f"- Autonomous official result runner rows: `{daemon.get('last_autonomous_official_result_runner_rows')}`",
            f"- Autonomous official result quarantine rows: `{daemon.get('last_autonomous_official_result_quarantine_rows')}`",
            f"- Autonomous official result quarantined race IDs: `{daemon.get('last_autonomous_official_result_quarantined_race_ids')}`",
            f"- Autonomous official result quarantine reasons: `{daemon.get('last_autonomous_official_result_quarantine_reason_counts')}`",
            f"- Autonomous official result quarantine errors: `{daemon.get('last_autonomous_official_result_quarantine_error_counts')}`",
            f"- Autonomous official result quarantine result boxes not in participants: `{daemon.get('last_autonomous_official_result_quarantine_result_boxes_not_in_participants_counts')}`",
            f"- Autonomous official result quarantine runner-set mismatch samples: `{daemon.get('last_autonomous_official_result_quarantine_runner_set_mismatch_samples')}`",
            f"- Autonomous official result skipped reasons: `{daemon.get('last_autonomous_official_result_skipped_reason_counts')}`",
            f"- Autonomous official result awaiting-jump races: `{daemon.get('last_autonomous_official_result_awaiting_jump_race_count')}`",
            f"- Autonomous official result awaiting-jump race IDs: `{daemon.get('last_autonomous_official_result_awaiting_jump_race_ids')}`",
            f"- Autonomous official result next recheck: `{daemon.get('last_autonomous_official_result_awaiting_jump_next_recheck_after_local')}`",
            f"- Autonomous official result evidence DB ingest: `{daemon.get('last_autonomous_official_result_evidence_db_ingest_status')}`",
            f"- Autonomous official result evidence DB execute: `{daemon.get('last_autonomous_official_result_evidence_db_execute')}`",
            f"- Autonomous official result evidence DB write performed: `{daemon.get('last_autonomous_official_result_evidence_db_write_performed')}`",
            f"- Autonomous official result evidence valid race rows: `{daemon.get('last_autonomous_official_result_evidence_valid_race_rows')}`",
            f"- Autonomous official result evidence valid runner rows: `{daemon.get('last_autonomous_official_result_evidence_valid_runner_rows')}`",
            f"- Autonomous official result evidence blocked race rows: `{daemon.get('last_autonomous_official_result_evidence_blocked_race_rows')}`",
            f"- Autonomous official result evidence blocked runner rows: `{daemon.get('last_autonomous_official_result_evidence_blocked_runner_rows')}`",
            f"- Autonomous official result evidence inserted race rows: `{daemon.get('last_autonomous_official_result_evidence_inserted_race_rows')}`",
            f"- Autonomous official result evidence inserted runner rows: `{daemon.get('last_autonomous_official_result_evidence_inserted_runner_rows')}`",
            f"- Autonomous official result evidence blocker reasons: `{daemon.get('last_autonomous_official_result_evidence_blocker_reason_counts')}`",
            f"- Autonomous official result evidence inserted rows: `{daemon.get('last_autonomous_official_result_evidence_inserted_rows')}`",
            f"- Live odds backlog unresolved races: `{daemon.get('last_live_odds_backlog_unresolved_race_count')}`",
            f"- Live odds backlog unresolved reasons: `{daemon.get('last_live_odds_backlog_unresolved_reason_counts')}`",
            f"- Live odds backlog recovery actions: `{daemon.get('last_live_odds_backlog_unresolved_recovery_action_counts')}`",
            f"- Live odds backlog alias statuses: `{daemon.get('last_live_odds_backlog_unresolved_alias_status_counts')}`",
            f"- Live odds backlog retryable exact-shadow matches: `{daemon.get('last_live_odds_backlog_retryable_exact_shadow_match_race_count')}`",
            f"- Live odds backlog no exact shadow match: `{daemon.get('last_live_odds_backlog_no_exact_shadow_match_race_count')}`",
            f"- Live odds backlog retryable exact-shadow race IDs: `{daemon.get('last_live_odds_backlog_retryable_exact_shadow_match_race_ids')}`",
            f"- Live odds backlog no-exact-shadow race IDs: `{daemon.get('last_live_odds_backlog_no_exact_shadow_match_race_ids')}`",
            f"- Live odds backlog recovery queue: `{daemon.get('last_live_odds_backlog_recovery_queue_path')}`",
            f"- Live odds backlog recovery queue diagnostic only: `{daemon.get('last_live_odds_backlog_recovery_queue_diagnostic_only')}`",
            f"- Live odds backlog recovery queue changed join acceptance: `{daemon.get('last_live_odds_backlog_recovery_queue_join_acceptance_changed')}`",
            f"- Live odds backlog recovery queue DB write performed: `{daemon.get('last_live_odds_backlog_recovery_queue_db_write_performed')}`",
            f"- Live odds backlog awaiting official-result evidence races: `{daemon.get('last_live_odds_backlog_awaiting_official_result_evidence_race_count')}`",
            f"- Live odds backlog awaiting official-result evidence race IDs: `{daemon.get('last_live_odds_backlog_awaiting_official_result_evidence_race_ids')}`",
            f"- Live odds backlog awaiting official-result authorized action: `{daemon.get('last_live_odds_backlog_awaiting_official_result_evidence_authorized_action')}`",
            f"- Live odds backlog awaiting official-result recheck-ready races: `{daemon.get('last_live_odds_backlog_awaiting_official_result_recheck_ready_race_count')}`",
            f"- Live odds backlog runner-set validation: `{daemon.get('last_live_odds_backlog_runner_set_validation_path')}`",
            f"- Live odds backlog runner-set retryable races: `{daemon.get('last_live_odds_backlog_runner_set_validation_retryable_race_count')}`",
            f"- Live odds backlog runner-set exact matches: `{daemon.get('last_live_odds_backlog_runner_set_validation_exact_match_race_count')}`",
            f"- Live odds backlog runner-set blocked races: `{daemon.get('last_live_odds_backlog_runner_set_validation_blocked_race_count')}`",
            f"- Live odds backlog runner-set validation diagnostic only: `{daemon.get('last_live_odds_backlog_runner_set_validation_diagnostic_only')}`",
            f"- Live odds backlog runner-set join authorized: `{daemon.get('last_live_odds_backlog_runner_set_validation_join_authorized')}`",
            f"- Live odds backlog runner-set DB write performed: `{daemon.get('last_live_odds_backlog_runner_set_validation_db_write_performed')}`",
            f"- Live odds backlog join eligibility packet: `{daemon.get('last_live_odds_backlog_join_eligibility_packet_path')}`",
            f"- Live odds backlog join eligibility evaluated races: `{daemon.get('last_live_odds_backlog_join_eligibility_evaluated_race_count')}`",
            f"- Live odds backlog join eligibility report-only races: `{daemon.get('last_live_odds_backlog_join_eligibility_eligible_report_only_race_count')}`",
            f"- Live odds backlog join eligibility blocked races: `{daemon.get('last_live_odds_backlog_join_eligibility_blocked_race_count')}`",
            f"- Live odds backlog join eligibility diagnostic only: `{daemon.get('last_live_odds_backlog_join_eligibility_diagnostic_only')}`",
            f"- Live odds backlog join eligibility join authorized: `{daemon.get('last_live_odds_backlog_join_eligibility_join_authorized')}`",
            f"- Live odds backlog join eligibility DB write performed: `{daemon.get('last_live_odds_backlog_join_eligibility_db_write_performed')}`",
            f"- Rolling model comparison: `{daemon.get('last_rolling_model_comparison_status')}`",
            f"- Rolling model comparison sample races: `{daemon.get('last_rolling_model_comparison_sample_races')}`",
            f"- Rolling model comparison best candidate: `{daemon.get('last_rolling_model_comparison_best_candidate')}`",
            f"- Rolling model comparison source rejected live odds candidates: `{daemon.get('last_rolling_model_comparison_source_rejected_live_odds_candidate_count')}`",
            f"- Rolling model comparison source rows with rejected live odds candidates: `{daemon.get('last_rolling_model_comparison_source_rows_with_rejected_live_odds_candidates')}`",
            f"- Rolling model comparison source rejected live odds candidate reasons: `{daemon.get('last_rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts')}`",
            f"- High-accuracy refinement: `{daemon.get('last_high_accuracy_refinement_status')}`",
            f"- High-accuracy PR gate: `{daemon.get('last_high_accuracy_promotion_pr_gate_status')}`",
            f"- High-accuracy unified eligible rows: `{daemon.get('last_high_accuracy_unified_evidence_eligible_rows')}`",
            f"- Promotion distance: `{daemon.get('last_promotion_distance_status')}`",
            f"- Promotion distance promotion ready: `{daemon.get('last_promotion_distance_promotion_ready')}`",
            f"- Promotion distance sample races: `{daemon.get('last_promotion_distance_sample_race_count')}`",
            f"- Promotion distance sample runner rows: `{daemon.get('last_promotion_distance_sample_runner_rows')}`",
            f"- Promotion distance best candidate: `{daemon.get('last_promotion_distance_best_candidate_key')}`",
            f"- Promotion distance best non-market candidate: `{daemon.get('last_promotion_distance_best_non_market_candidate_key')}`",
            f"- Promotion distance blockers: `{daemon.get('last_promotion_distance_blockers') or []}`",
            f"- Promotion distance source rejected live odds candidates: `{daemon.get('last_promotion_distance_source_rejected_live_odds_candidate_count')}`",
            f"- Promotion distance source rows with rejected live odds candidates: `{daemon.get('last_promotion_distance_source_rows_with_rejected_live_odds_candidates')}`",
            f"- Promotion distance source rejected live odds candidate reasons: `{daemon.get('last_promotion_distance_source_rejected_live_odds_candidate_reason_counts')}`",
            f"- Promotion distance source exclusion reasons: `{daemon.get('last_promotion_distance_source_exclusion_reason_counts')}`",
            f"- Promotion distance source odds exclusion reasons: `{daemon.get('last_promotion_distance_source_odds_exclusion_reason_counts')}`",
            f"- Promotion distance source official-result missing race IDs: `{daemon.get('last_promotion_distance_source_official_result_evidence_db_missing_race_ids')}`",
            f"- Promotion distance source official-result requested race count: `{daemon.get('last_promotion_distance_source_official_result_evidence_db_requested_race_count')}`",
            f"- Promotion distance source official-result races with rows: `{daemon.get('last_promotion_distance_source_official_result_evidence_db_races_with_rows')}`",
            f"- Promotion distance official-result coverage requested races: `{daemon.get('last_promotion_distance_official_result_coverage_requested_race_count')}`",
            f"- Promotion distance official-result requested race count source: `{daemon.get('last_promotion_distance_official_result_coverage_requested_race_count_source')}`",
            f"- Promotion distance official-result legacy requested race count without IDs: `{daemon.get('last_promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids')}`",
            f"- Promotion distance official-result coverage races with rows: `{daemon.get('last_promotion_distance_official_result_coverage_races_with_rows_count')}`",
            f"- Promotion distance official-result coverage missing races: `{daemon.get('last_promotion_distance_official_result_coverage_missing_race_count')}`",
            f"- Promotion distance official-result missing exclusions: `{daemon.get('last_promotion_distance_official_result_coverage_missing_exclusion_count')}`",
            f"- Promotion distance official-result runner path count: `{daemon.get('last_promotion_distance_official_result_runner_path_count')}`",
            f"- Promotion distance official-result runner paths source: `{daemon.get('last_promotion_distance_official_result_runner_paths_source_field')}`",
            f"- Best aggregate unified evidence: `{daemon.get('last_best_aggregate_unified_evidence_status')}`",
            f"- Best aggregate unified eligible rows: `{daemon.get('last_best_aggregate_unified_evidence_eligible_rows')}`",
            f"- Best aggregate unified rejected live odds candidates: `{daemon.get('last_best_aggregate_unified_rejected_live_odds_candidate_count')}`",
            f"- Best aggregate unified rows with rejected live odds candidates: `{daemon.get('last_best_aggregate_unified_rows_with_rejected_live_odds_candidates')}`",
            f"- Best aggregate unified rejected live odds candidate reasons: `{daemon.get('last_best_aggregate_unified_rejected_live_odds_candidate_reason_counts')}`",
            f"- Rejoin unified evidence: `{daemon.get('last_rejoin_unified_evidence_status')}`",
            f"- Rejoin rejected live odds candidates: `{daemon.get('last_rejoin_unified_rejected_live_odds_candidate_count')}`",
            f"- Rejoin rows with rejected live odds candidates: `{daemon.get('last_rejoin_unified_rows_with_rejected_live_odds_candidates')}`",
            f"- Rejoin rejected live odds candidate reasons: `{daemon.get('last_rejoin_unified_rejected_live_odds_candidate_reason_counts')}`",
            f"- Rejoin rolling comparison: `{daemon.get('last_rejoin_rolling_model_comparison_status')}`",
            f"- Rejoin rolling comparison sample races: `{daemon.get('last_rejoin_rolling_model_comparison_sample_races')}`",
            f"- Rejoin high-accuracy packet: `{daemon.get('last_rejoin_high_accuracy_refinement_status')}`",
            f"- Rejoin pre-race gated challenger: `{daemon.get('last_rejoin_pre_race_gated_challenger_status')}`",
            f"- Rejoin pre-race gated challenger promotion ready: `{daemon.get('last_rejoin_pre_race_gated_challenger_promotion_ready')}`",
            f"- Rejoin rank-first hypothesis review: `{daemon.get('last_rejoin_rank_first_hypothesis_review_status')}`",
            f"- Rejoin rank-first hypothesis candidates: `{daemon.get('last_rejoin_rank_first_hypothesis_candidate_count')}`",
            f"- Rejoin rank-first hypothesis evaluated candidates: `{daemon.get('last_rejoin_rank_first_hypothesis_evaluated_candidate_count')}`",
            f"- Rejoin rank-first hypothesis best candidate: `{daemon.get('last_rejoin_rank_first_hypothesis_best_candidate_key')}`",
            f"- Rejoin rank-first hypothesis directional ready: `{daemon.get('last_rejoin_rank_first_hypothesis_directional_read_ready')}`",
            f"- Rejoin time-split gated challenger: `{daemon.get('last_rejoin_time_split_gated_challenger_status')}`",
            f"- Rejoin time-split gated challenger promotion ready: `{daemon.get('last_rejoin_time_split_gated_challenger_promotion_ready')}`",
            f"- Rejoin market residual challenger: `{daemon.get('last_rejoin_market_residual_challenger_status')}`",
            f"- Rejoin market residual challenger promotion ready: `{daemon.get('last_rejoin_market_residual_challenger_promotion_ready')}`",
            f"- Rejoin market residual regime audit: `{daemon.get('last_rejoin_market_residual_regime_audit_status')}`",
            f"- Rejoin market residual regime audit promotion ready: `{daemon.get('last_rejoin_market_residual_regime_audit_promotion_ready')}`",
            f"- Rejoin market residual rank-first hypothesis: `{daemon.get('last_rejoin_market_residual_rank_first_hypothesis_status')}`",
            f"- Rejoin rank-first watchlist: `{daemon.get('last_rejoin_rank_first_hypothesis_watchlist_status')}`",
            f"- Rejoin rank-first watchlist candidates: `{daemon.get('last_rejoin_rank_first_hypothesis_watchlist_candidate_count')}`",
            f"- Rejoin rank-first watchlist directional-ready candidates: `{daemon.get('last_rejoin_rank_first_hypothesis_watchlist_directional_ready_candidate_count')}`",
            f"- Rejoin rank-first watchlist best candidate: `{daemon.get('last_rejoin_rank_first_hypothesis_watchlist_best_candidate')}`",
            f"- Rejoin rank-first watchlist best status: `{daemon.get('last_rejoin_rank_first_hypothesis_watchlist_best_status')}`",
            f"- Rejoin rank-first watchlist best distinct samples: `{daemon.get('last_rejoin_rank_first_hypothesis_watchlist_best_distinct_samples')}`",
            f"- Rejoin promotion distance: `{daemon.get('last_rejoin_promotion_distance_status')}`",
            f"- Rejoin promotion distance promotion ready: `{daemon.get('last_rejoin_promotion_distance_promotion_ready')}`",
            f"- Rejoin promotion distance blockers: `{daemon.get('last_rejoin_promotion_distance_blockers') or []}`",
            f"- Rejoin promotion distance source exclusion reasons: `{daemon.get('last_rejoin_promotion_distance_source_exclusion_reason_counts')}`",
            f"- Rejoin promotion distance source official-result missing race IDs: `{daemon.get('last_rejoin_promotion_distance_source_official_result_evidence_db_missing_race_ids')}`",
            f"- Rejoin promotion distance official-result coverage requested races: `{daemon.get('last_rejoin_promotion_distance_official_result_coverage_requested_race_count')}`",
            f"- Rejoin promotion distance official-result requested race count source: `{daemon.get('last_rejoin_promotion_distance_official_result_coverage_requested_race_count_source')}`",
            f"- Rejoin promotion distance official-result legacy requested race count without IDs: `{daemon.get('last_rejoin_promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids')}`",
            f"- Rejoin promotion distance official-result coverage races with rows: `{daemon.get('last_rejoin_promotion_distance_official_result_coverage_races_with_rows_count')}`",
            f"- Rejoin promotion distance official-result coverage missing races: `{daemon.get('last_rejoin_promotion_distance_official_result_coverage_missing_race_count')}`",
            f"- Rejoin promotion distance official-result missing exclusions: `{daemon.get('last_rejoin_promotion_distance_official_result_coverage_missing_exclusion_count')}`",
            f"- Rejoin promotion distance official-result runner path count: `{daemon.get('last_rejoin_promotion_distance_official_result_runner_path_count')}`",
            f"- Rejoin promotion distance official-result runner paths source: `{daemon.get('last_rejoin_promotion_distance_official_result_runner_paths_source_field')}`",
            f"- Daily run: `{daily.get('daily_shadow_run_dir')}`",
            f"- Daily status: `{daily.get('final_status')}`",
            f"- Same-distance provenance: `{(daily.get('same_distance_history_provenance') or {}).get('status')}`",
            f"- Feature activation gate: `{(report.get('feature_activation_gate') or {}).get('status')}`",
            f"- Feature activation blockers: `{((report.get('feature_activation_gate') or {}).get('fail_reason_summary') or {}).get('category_counts')}`",
            f"- Odds coverage readiness: `{(report.get('shadow_odds_coverage') or {}).get('readiness_status')}`",
            f"- Odds coverage next action: `{(report.get('shadow_odds_coverage') or {}).get('next_action')}`",
            "",
            "No refresh, scoring, result join, DB write, label write, registry mutation, EV/betting output, TGR enablement, or production prediction write was performed by this report.",
        ]
    ) + "\n"


def assert_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative = logical.absolute().relative_to(ROOT.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_forward_shadow_runtime_state_artifact:{relative}")
    return logical.absolute()


def run_runtime_state(
    *,
    evidence_root: Path = DEFAULT_EVIDENCE_ROOT,
    daemon_state_path: Path = DEFAULT_DAEMON_STATE,
    output_dir: Path | None = None,
    include_systemd: bool = False,
    target_joined_races: int = DEFAULT_TARGET_JOINED_RACES,
) -> dict[str, Any]:
    generated_at = datetime.now().astimezone()
    output_dir = output_dir or evidence_root / (
        f"forward_shadow_runtime_state_{generated_at.strftime('%Y%m%dT%H%M%S%z')}"
    )
    output_dir = assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    timer = systemd_timer_snapshot() if include_systemd else None
    report = build_runtime_state(
        evidence_root=evidence_root,
        daemon_state_path=daemon_state_path,
        timer=timer,
        target_joined_races=target_joined_races,
        generated_at=generated_at,
    )
    write_json(output_dir / "forward_shadow_runtime_state.json", report)
    write_text(output_dir / "SUMMARY.md", build_summary(report))
    write_text(output_dir / "final_status.txt", str(report["runtime_action"]) + "\n")
    return {
        "output_dir": relpath(output_dir),
        "runtime_action": report["runtime_action"],
        "safe_joined_races": report["safe_joined_races"],
        "safe_joined_races_remaining": report["safe_joined_races_remaining"],
        "next_prejump_refresh_status": report["daemon"].get(
            "last_next_prejump_refresh_status"
        ),
        "recommended_rerun_after_local": report["daemon"].get(
            "last_recommended_rerun_after_local"
        ),
        "odds_coverage_readiness": report["shadow_odds_coverage"].get(
            "readiness_status"
        ),
        "odds_coverage_next_action": report["shadow_odds_coverage"].get("next_action"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--daemon-state", type=Path, default=DEFAULT_DAEMON_STATE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--include-systemd", action="store_true")
    parser.add_argument("--target-joined-races", type=int, default=DEFAULT_TARGET_JOINED_RACES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_runtime_state(
        evidence_root=args.evidence_root,
        daemon_state_path=args.daemon_state,
        output_dir=args.output_dir,
        include_systemd=args.include_systemd,
        target_joined_races=args.target_joined_races,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
