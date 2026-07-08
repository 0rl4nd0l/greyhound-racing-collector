import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import build_high_accuracy_refinement_packet as packet


def _metrics(
    *,
    races: int = 120,
    top1: float = 0.20,
    top3: float = 0.55,
    mean_winner_rank: float = 3.8,
    brier: float = 0.12,
    logloss: float = 1.9,
    slope: float = 0.5,
    intercept: float = -0.8,
    box1: float = 0.25,
) -> dict:
    return {
        "safe_joined_race_count": races,
        "safe_joined_runner_count": races * 7,
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


def _calibration_report(
    *,
    baseline: dict | None = None,
    candidate: dict | None = None,
    status: str = packet.CALIBRATION_READY,
) -> dict:
    return {
        "schema_version": "forward_shadow_challenger_calibration_v1",
        "final_status": status,
        "candidate_alpha": 0.75,
        "calibration_family": "per_race_power_normalization",
        "safe_exact_joined_race_count": 120,
        "train_race_count": 96,
        "eval_race_count": 24,
        "baseline_eval_metrics": baseline or _metrics(),
        "candidate_eval_metrics": candidate
        or _metrics(brier=0.11, logloss=1.8, slope=0.7, intercept=-0.5),
        "rejected_joined_races": [{"race_id": "Unsafe Race", "reasons": ["non_exact_identity_match_status"]}],
        "duplicate_joined_race_ids_seen": ["Duplicate Race"],
        "activation_blockers": [],
        "production_activation_allowed": False,
        "no_write_guarantees": {
            "registry_mutation": False,
            "production_promotion": False,
        },
    }


def test_calibration_only_routes_to_model_training_when_rank_does_not_improve():
    report = _calibration_report()

    result = packet.build_packet(
        calibration_report=report,
        thresholds=packet.AccuracyGateThresholds(min_safe_joined_races=100, min_top1_delta=0.02),
        generated_at=datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc),
        protected_before={},
        protected_after={},
    )

    calibration = result["stages"]["calibration_only"]
    assert result["final_status"] == packet.FINAL_NEEDS_MODEL_CHALLENGER
    assert calibration["rank_preserving_by_construction"] is True
    assert calibration["rejected_joined_races_excluded_count"] == 1
    assert calibration["duplicate_joined_race_ids_seen_count"] == 1
    assert calibration["gate"]["status"] == "BLOCKED"
    assert "rank_accuracy_top1_delta_below_min" in calibration["gate"]["blockers"]
    assert result["promotion_pr_gate"]["status"] == "BLOCKED"
    assert result["promotion_pr_gate"]["pull_request_boundary"] == {
        "promotion_pr_allowed": False,
        "direct_local_switch_allowed": False,
        "local_registry_mutation_allowed": False,
        "production_pointer_update_allowed": False,
        "requires_human_pr_review": True,
    }


def test_non_tgr_model_candidate_can_open_pr_only_gate_without_direct_switch():
    calibration = _calibration_report()
    shadow_replay_metrics = {
        "champion_baseline": {
            **_metrics(races=130, top1=0.20, top3=0.55, mean_winner_rank=3.8, brier=0.12, logloss=1.9, box1=0.30),
            "race_count": 130,
            "log_loss": 1.9,
            "box_bias": {"box1_top_pick_share": 0.30},
        },
        "shadow_calibrated_rf_power_gamma_2_4": {
            **_metrics(races=130, top1=0.24, top3=0.58, mean_winner_rank=3.5, brier=0.11, logloss=1.8, slope=0.7, intercept=-0.4, box1=0.26),
            "race_count": 130,
            "log_loss": 1.8,
            "box_bias": {"box1_top_pick_share": 0.26},
        },
    }

    result = packet.build_packet(
        calibration_report=calibration,
        shadow_replay_metrics=shadow_replay_metrics,
        thresholds=packet.AccuracyGateThresholds(min_safe_joined_races=100, min_top1_delta=0.02),
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
    assert result["no_write_guarantees"]["registry_mutation"] is False
    assert result["no_write_guarantees"]["direct_switch"] is False


def test_candidate_blocks_on_box_bias_and_protected_path_change():
    baseline = _metrics(box1=0.20)
    candidate = _metrics(top1=0.25, top3=0.58, mean_winner_rank=3.4, brier=0.11, logloss=1.8, slope=0.7, intercept=-0.4, box1=0.40)
    report = _calibration_report(baseline=baseline, candidate=candidate)

    result = packet.build_packet(
        calibration_report=report,
        thresholds=packet.AccuracyGateThresholds(min_safe_joined_races=100, min_top1_delta=0.02),
        generated_at=datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc),
        protected_before={"model_registry/best_metadata.json": "abc"},
        protected_after={"model_registry/best_metadata.json": "changed"},
    )

    blockers = result["stages"]["calibration_only"]["gate"]["blockers"]
    assert "candidate_box1_top_pick_share_above_max" in blockers
    assert "metric_regressed:box1_top_pick_share" in blockers
    assert "protected_paths_changed" in result["promotion_pr_gate"]["blockers"]
    assert result["promotion_pr_gate"]["pull_request_boundary"]["promotion_pr_allowed"] is False


def test_source_report_must_keep_production_activation_blocked():
    report = _calibration_report(
        candidate=_metrics(
            top1=0.24,
            top3=0.58,
            mean_winner_rank=3.4,
            brier=0.11,
            logloss=1.8,
            slope=0.7,
            intercept=-0.4,
            box1=0.22,
        )
    )
    report["production_activation_allowed"] = True
    report["no_write_guarantees"]["registry_mutation"] = True

    result = packet.build_packet(
        calibration_report=report,
        thresholds=packet.AccuracyGateThresholds(min_safe_joined_races=100, min_top1_delta=0.02),
        generated_at=datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc),
        protected_before={},
        protected_after={},
    )

    reasons = result["stages"]["calibration_only"]["source_control_reasons"]
    blockers = result["stages"]["calibration_only"]["gate"]["blockers"]
    assert "source_report_production_activation_not_blocked" in reasons
    assert "source_report_write_guard_not_blocked:registry_mutation" in reasons
    assert "source_report_production_activation_not_blocked" in blockers
    assert result["promotion_pr_gate"]["status"] == "BLOCKED"


def test_output_dir_guard_rejects_unsafe_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        packet.assert_output_dir_safe(tmp_path.parent / "outside" / "high_accuracy_refinement_packet_test")

    with pytest.raises(ValueError, match="output_dir_must_be_high_accuracy_refinement_packet"):
        packet.assert_output_dir_safe(
            tmp_path / "artifacts/full_evidence_orchestration_20260525/other_packet_test"
        )


def test_run_packet_writes_report_only_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "ROOT", tmp_path)
    monkeypatch.setattr(packet, "DEFAULT_PROTECTED_PATHS", ())
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525"
        / "high_accuracy_refinement_packet_20260610T010000+0000"
    )
    calibration_report_path = tmp_path / "calibration_report.json"
    calibration_report_path.write_text(json.dumps(_calibration_report()), encoding="utf-8")

    result = packet.run_refinement_packet(
        calibration_report_path=calibration_report_path,
        output_dir=output_dir,
        thresholds=packet.AccuracyGateThresholds(min_safe_joined_races=100, min_top1_delta=0.02),
    )

    assert result["final_status"] == packet.FINAL_NEEDS_MODEL_CHALLENGER
    assert result["promotion_pr_gate_status"] == "BLOCKED"
    assert (output_dir / "high_accuracy_refinement_packet.json").exists()
    assert (output_dir / "promotion_pr_gate.json").exists()
    assert (output_dir / "promotion_pr_body.md").exists()
    assert (output_dir / "SUMMARY.md").exists()
    written = json.loads((output_dir / "high_accuracy_refinement_packet.json").read_text())
    assert written["no_write_guarantees"]["production_promotion"] is False
    assert written["protected_paths_unchanged"] is True
