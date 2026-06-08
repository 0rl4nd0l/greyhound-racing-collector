import json
from pathlib import Path

from scripts.plan_calibration_deployment import build_plan


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _model_packet(calibration_design: Path) -> dict:
    return {
        "schema_version": "model_review_packet_v1",
        "status": "READY_FOR_CHALLENGER_REVIEW",
        "failures": [],
        "review_gate": {
            "minimum_clean_evaluated_races": 100,
            "clean_official_evaluated_races": 105,
        },
        "promotion_control": {
            "action_taken": "none",
            "promotion_allowed": False,
            "registry_mutation_allowed": False,
        },
        "steps": [
            {
                "name": "promotion",
                "required_gate": "APPROVE_MODEL_PROMOTION",
            }
        ],
        "challenger_review_gate": {
            "status": "READY",
            "candidate_arm": "power_calibrated_baseline",
            "stability_status": "STABLE_REPORT_ONLY",
            "all_log_loss_improved": True,
            "all_brier_improved": True,
            "all_ranking_preserved": True,
            "selected_alpha": 0.5,
            "promotion_allowed": False,
            "registry_mutation_allowed": False,
            "model_artifact_written": False,
        },
        "source_evidence": {
            "calibration_design": str(calibration_design),
        },
    }


def _calibration_design() -> dict:
    return {
        "schema_version": "calibration_layer_design_v1",
        "status": "READY_FOR_OPERATOR_DESIGN_REVIEW",
        "failures": [],
        "runtime_transform_spec": {
            "candidate_arm": "power_calibrated_baseline",
            "algorithm": "power_normalize_per_race",
            "alpha": 0.5,
            "input_probability_key": "win_prob_norm",
            "output_probability_key": "calibrated_win_prob_report_only",
            "rank_preserving_when_alpha_positive": True,
            "uses_labels_at_runtime": False,
            "uses_odds_at_runtime": False,
            "requires_runner_complete_race_group": True,
        },
        "comparison_to_baseline": {
            "log_loss_improved": True,
            "brier_improved": True,
            "top1_preserved": True,
            "top2_preserved": True,
            "top3_preserved": True,
            "mean_winner_rank_preserved": True,
        },
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


def _loop_plan(calibration_design: Path) -> dict:
    design = str(calibration_design)
    dry_run_command = [
        "python",
        "scripts/capture_prediction_snapshot.py",
        "--report-only-calibration-design",
        design,
        "--output",
        "dry_run_capture_report.json",
    ]
    approved_persist_command = [
        "python",
        "scripts/capture_prediction_snapshot.py",
        "--persist",
        "--report-only-calibration-design",
        design,
        "--approve-live-persist",
        "--output",
        "persist_capture_report.json",
    ]
    planned_odds_command = [
        "python",
        "scripts/capture_prediction_snapshot.py",
        "--capture-live-odds",
        "--report-only-calibration-design",
        design,
        "--approve-live-odds-capture",
        "--output",
        "odds_capture_dry_snapshot_report.json",
    ]
    return {
        "promotion_readiness_gate": {
            "status": "APPROVAL_PRESENT_EVIDENCE_READY_REPORT_ONLY",
            "ready_for_separate_promotion_review": True,
            "promotion_action_taken": "none",
            "promotion_allowed_by_loop": False,
            "registry_mutation_allowed_by_loop": False,
            "model_artifact_write_allowed_by_loop": False,
            "betting_allowed_by_loop": False,
            "promotion_evidence_clean_official_evaluated_races": 105,
            "required_gate": "APPROVE_MODEL_PROMOTION",
        },
        "guarantees": {
            "no_model_promotion": True,
            "no_retrain": True,
            "no_betting": True,
        },
        "steps": [
            {
                "name": "dry_run_prejump_capture",
                "command": dry_run_command,
            },
            {
                "name": "approved_persist_ready_subset",
                "command": approved_persist_command,
            }
        ],
        "persist_approval_packet": {
            "planned_persist_command": approved_persist_command,
            "approved_same_run_execute_ready_command_template": [
                "python",
                "scripts/prejump_prediction_loop.py",
                "--report-only-calibration-design",
                design,
                "--execute-ready",
                "--approve-live-persist",
            ]
        },
        "live_odds_approval_packet": {
            "planned_odds_command": planned_odds_command,
            "combined_persist_live_odds_command": approved_persist_command,
            "approved_same_run_execute_ready_command_template": [
                "python",
                "scripts/prejump_prediction_loop.py",
                "--report-only-calibration-design",
                design,
                "--execute-ready",
                "--approve-live-odds-capture",
            ]
        },
    }


def test_deployment_plan_is_ready_without_write_commands(tmp_path):
    calibration_design_path = tmp_path / "calibration_design.json"
    model_packet_path = tmp_path / "model_packet.json"
    loop_plan_path = tmp_path / "loop_plan.json"
    _write_json(calibration_design_path, _calibration_design())
    _write_json(model_packet_path, _model_packet(calibration_design_path))
    _write_json(loop_plan_path, _loop_plan(calibration_design_path))

    plan = build_plan(
        model_review_packet_path=model_packet_path,
        calibration_design_path=calibration_design_path,
        loop_plan_path=loop_plan_path,
    )

    assert plan["status"] == "READY_FOR_SEPARATE_PROMOTION_IMPLEMENTATION_REVIEW"
    assert plan["actual_promotion_command"] is None
    assert plan["writes_performed"] == {
        "label_write": False,
        "model_artifact_write": False,
        "registry_mutation": False,
        "production_config_write": False,
        "refresh_signal_write": False,
        "betting": False,
    }
    assert plan["deployment_controls"]["required_gate"] == "APPROVE_MODEL_PROMOTION"
    assert plan["loop_pass_through"]["dry_run_capture"] is True
    assert plan["loop_pass_through"]["approved_persist_capture"] is True
    assert plan["loop_pass_through"]["live_odds_capture"] is True
    assert plan["loop_pass_through"]["persist_same_run"] is True
    assert plan["loop_pass_through"]["live_odds_same_run"] is True


def test_deployment_plan_fails_closed_when_design_allows_promotion(tmp_path):
    calibration_design_path = tmp_path / "calibration_design.json"
    model_packet_path = tmp_path / "model_packet.json"
    loop_plan_path = tmp_path / "loop_plan.json"
    design = _calibration_design()
    design["deployment_control"]["promotion_allowed"] = True
    _write_json(calibration_design_path, design)
    _write_json(model_packet_path, _model_packet(calibration_design_path))
    _write_json(loop_plan_path, _loop_plan(calibration_design_path))

    plan = build_plan(
        model_review_packet_path=model_packet_path,
        calibration_design_path=calibration_design_path,
        loop_plan_path=loop_plan_path,
    )

    assert plan["status"] == "NOT_READY"
    assert "calibration_promotion_not_blocked" in plan["failures"]
    assert plan["actual_promotion_command"] is None


def test_deployment_plan_fails_closed_when_loop_lacks_pass_through(tmp_path):
    calibration_design_path = tmp_path / "calibration_design.json"
    model_packet_path = tmp_path / "model_packet.json"
    loop_plan_path = tmp_path / "loop_plan.json"
    loop_plan = _loop_plan(calibration_design_path)
    loop_plan["steps"][0]["command"].remove("--report-only-calibration-design")
    loop_plan["steps"][0]["command"].remove(str(calibration_design_path))
    _write_json(calibration_design_path, _calibration_design())
    _write_json(model_packet_path, _model_packet(calibration_design_path))
    _write_json(loop_plan_path, loop_plan)

    plan = build_plan(
        model_review_packet_path=model_packet_path,
        calibration_design_path=calibration_design_path,
        loop_plan_path=loop_plan_path,
    )

    assert plan["status"] == "NOT_READY"
    assert (
        "loop_missing_report_only_calibration_pass_through:dry_run_capture"
        in plan["failures"]
    )
    assert plan["writes_performed"]["model_artifact_write"] is False
