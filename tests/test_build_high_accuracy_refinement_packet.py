import json
from datetime import datetime, timezone

from scripts import build_high_accuracy_refinement_packet as packet


def _metrics(
    *,
    races: int = 120,
    top1: float = 0.20,
    top3: float = 0.55,
    mean_winner_rank: float = 3.8,
    brier: float = 0.12,
    logloss: float = 1.9,
    slope: float = 0.6,
    intercept: float = -0.4,
    box1: float = 0.25,
) -> dict:
    return {
        "safe_joined_race_count": races,
        "top1": top1,
        "top3": top3,
        "mean_winner_rank": mean_winner_rank,
        "brier": brier,
        "logloss": logloss,
        "probability_sum_max_error_joined_races": 0.0,
        "box1_top_pick_share": box1,
        "calibration_slope_intercept": {
            "status": "fit",
            "slope": slope,
            "intercept": intercept,
        },
    }


def test_stage2_forward_metrics_can_open_pr_only_gate_without_direct_switch():
    stage2_forward_metrics = {
        "status": packet.STAGE2_FORWARD_SHADOW_READY_FOR_REVIEW,
        "baseline_forward_shadow_metrics": _metrics(),
        "stage2_challenger_forward_shadow_metrics": _metrics(
            top1=0.24,
            top3=0.58,
            mean_winner_rank=3.5,
            brier=0.11,
            logloss=1.8,
            slope=0.8,
            intercept=-0.2,
            box1=0.22,
        ),
    }

    result = packet.build_packet(
        stage2_forward_metrics=stage2_forward_metrics,
        thresholds=packet.AccuracyGateThresholds(
            min_safe_joined_races=100,
            min_top1_delta=0.02,
        ),
        generated_at=datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc),
        protected_before={"model_registry/best_metadata.json": "abc"},
        protected_after={"model_registry/best_metadata.json": "abc"},
    )

    assert result["final_status"] == packet.FINAL_READY_FOR_PR
    assert result["stages"]["non_tgr_model_challenger"]["gate"]["status"] == "PASS"
    assert result["promotion_pr_gate"]["status"] == "READY_FOR_PR_DRAFT"
    assert result["promotion_pr_gate"]["selected_stage"] == "stage_2_non_tgr_model_challenger"
    boundary = result["promotion_pr_gate"]["pull_request_boundary"]
    assert boundary["promotion_pr_allowed"] is True
    assert boundary["direct_local_switch_allowed"] is False
    assert boundary["local_registry_mutation_allowed"] is False
    assert boundary["production_pointer_update_allowed"] is False
    assert result["no_write_guarantees"]["odds_used_for_shadow_scoring"] is False
    assert result["ev_diagnostics_summary"]["ev_metrics_used_for_promotion"] is False


def test_odds_ev_improvement_cannot_override_rank_or_box_bias_regression():
    odds_gate_report = {
        "status": packet.ODDS_RESEARCH_READY_REPORT_ONLY,
        "complete_valid_prejump_odds_races": 120,
        "odds_used_for_shadow_scoring": False,
    }
    odds_augmented_report = {
        "final_status": packet.ODDS_AUGMENTED_MODEL_READY_FOR_PR_REVIEW,
        "baseline_metrics": _metrics(top1=0.25, top3=0.60, box1=0.20),
        "candidate_metrics": _metrics(
            top1=0.22,
            top3=0.58,
            mean_winner_rank=4.1,
            brier=0.13,
            logloss=2.0,
            slope=0.2,
            intercept=-1.2,
            box1=0.42,
        ),
        "ev_improved": True,
    }
    ev_diagnostics = {
        "status": "EV_DIAGNOSTICS_REPORT_ONLY",
        "ev_rows": 840,
        "mean_ev": 0.08,
    }

    result = packet.build_packet(
        odds_gate_report=odds_gate_report,
        odds_augmented_report=odds_augmented_report,
        ev_diagnostics=ev_diagnostics,
        thresholds=packet.AccuracyGateThresholds(
            min_safe_joined_races=100,
            min_top1_delta=0.02,
        ),
        generated_at=datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc),
        protected_before={},
        protected_after={},
    )

    odds_stage = result["stages"]["odds_augmented_model"]
    blockers = odds_stage["gate"]["blockers"]
    assert odds_stage["status"] == packet.ODDS_AUGMENTED_MODEL_BLOCKED
    assert "rank_accuracy_top1_delta_below_min" in blockers
    assert "metric_regressed:top3" in blockers
    assert "metric_regressed:mean_winner_rank" in blockers
    assert "metric_regressed:calibration_slope_intercept" in blockers
    assert "candidate_box1_top_pick_share_above_max" in blockers
    assert "ev_improvement_ignored_because_accuracy_guardrails_failed" in blockers
    assert result["promotion_pr_gate"]["status"] == "BLOCKED"
    assert result["ev_diagnostics_summary"]["ev_metrics_used_for_promotion"] is False
    assert result["ev_diagnostics_summary"]["ev_can_override_accuracy_gate"] is False


def test_run_packet_writes_report_only_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "ROOT", tmp_path)
    monkeypatch.setattr(packet, "DEFAULT_PROTECTED_PATHS", ())
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525"
        / "high_accuracy_refinement_packet_20260610T010000+0000"
    )
    stage2_forward_metrics_path = tmp_path / "stage2_forward_joined_metrics.json"
    stage2_forward_metrics_path.write_text(
        json.dumps(
            {
                "status": packet.STAGE2_FORWARD_SHADOW_COLLECTING,
                "baseline_forward_shadow_metrics": _metrics(),
                "stage2_challenger_forward_shadow_metrics": _metrics(races=20),
            }
        ),
        encoding="utf-8",
    )

    result = packet.run_refinement_packet(
        stage2_forward_metrics_path=stage2_forward_metrics_path,
        output_dir=output_dir,
        thresholds=packet.AccuracyGateThresholds(min_safe_joined_races=100),
    )

    assert result["final_status"] == packet.FINAL_BLOCKED
    assert result["promotion_pr_gate_status"] == "BLOCKED"
    assert (output_dir / "high_accuracy_refinement_packet.json").exists()
    assert (output_dir / "promotion_pr_gate.json").exists()
    assert (output_dir / "stage2_forward_joined_metrics.json").exists()
    assert (output_dir / "SUMMARY.md").exists()
    written = json.loads((output_dir / "high_accuracy_refinement_packet.json").read_text())
    assert written["no_write_guarantees"]["production_promotion"] is False
    assert written["protected_paths_unchanged"] is True
