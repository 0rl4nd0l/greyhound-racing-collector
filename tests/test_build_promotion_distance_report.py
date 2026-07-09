import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import build_promotion_distance_report as distance


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _candidate(
    *,
    candidate_key: str,
    top1: float,
    top3: float,
    mean_winner_rank: float,
    brier: float,
    logloss: float,
    slope: float,
    intercept: float,
    box1: float,
    family: str = "odds_augmented_blend",
) -> dict:
    return {
        "candidate_key": candidate_key,
        "family": family,
        "status": "EVALUATED",
        "race_count": 120,
        "top1": top1,
        "top3": top3,
        "mean_winner_rank": mean_winner_rank,
        "brier": brier,
        "logloss": logloss,
        "box1_top_pick_share": box1,
        "probability_sum_max_error_joined_races": 0.0,
        "calibration_slope_intercept": {
            "status": "computed",
            "slope": slope,
            "intercept": intercept,
        },
    }


def test_promotion_distance_report_quantifies_blocked_accuracy_path(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(distance, "ROOT", tmp_path)
    rolling = _write_json(
        tmp_path / "rolling.json",
        {
            "sample_race_count": 140,
            "sample_runner_rows": 953,
            "best_candidate_key": "market_only_implied",
            "best_non_market_candidate_key": "stage2_uncalibrated_market_blend_75",
            "best_non_market_minus_market": {
                "top1": -0.007,
                "top3": -0.02,
                "mean_winner_rank": 0.0,
                "brier": 0.001,
                "logloss": 0.006,
            },
            "source_rejected_live_odds_candidate_count": 5,
            "source_rows_with_rejected_live_odds_candidates": 4,
            "source_rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 2,
                "odds_source_url_missing": 3,
            },
            "source_exclusion_reason_counts": {"official_result_missing": 32},
            "source_odds_exclusion_reason_counts": {"strict_prejump_odds_missing": 4},
            "source_official_result_evidence_db_missing_race_ids": [
                "Race 3 - TAREE - 2026-06-13"
            ],
            "source_official_result_evidence_db_requested_race_ids": [
                "Race 3 - TAREE - 2026-06-13",
                "Race 5 - TAREE - 2026-06-13",
                "Race 6 - TAREE - 2026-06-13",
            ],
            "source_official_result_evidence_db_requested_race_count": 7,
            "source_official_result_evidence_db_legacy_requested_race_count_without_ids": 4,
            "source_official_result_evidence_db_races_with_rows": [
                "Race 5 - TAREE - 2026-06-13",
                "Race 6 - TAREE - 2026-06-13",
            ],
            "source_official_result_runner_paths": [
                "artifacts/full_evidence_orchestration_20260525/autonomous_official_result_capture_test/official_result_runners.jsonl"
            ],
        },
    )
    gated = _write_json(
        tmp_path / "gated.json",
        {
            "predeclared_residual_candidate": {
                "candidate_key": "market_favourite_gt_4_0__raw_stage2_market_blend_75",
                "status": "PREDECLARED_RESIDUAL_CANDIDATE_COLLECTING",
                "triggered_race_count": 3,
                "minimum_triggered_races_for_directional_read": 10,
                "directional_read_ready": False,
                "candidate_minus_market": {
                    "top1": 0.0,
                    "top3": 0.0,
                    "mean_winner_rank": -0.007,
                    "brier": -0.001,
                    "logloss": -0.0015,
                },
                "blockers": [
                    "predeclared_residual_candidate_report_only",
                    "triggered_race_count_below_directional_floor",
                    "top1_not_above_market",
                ],
            },
        },
    )
    gate = _write_json(
        tmp_path / "gate.json",
        {
            "status": "BLOCKED",
            "blockers": ["no_candidate_passed_rank_first_accuracy_gate"],
        },
    )
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525"
        / "promotion_distance_report_test"
    )

    report = distance.build_report(
        rolling_report_path=rolling,
        pre_race_gated_report_path=gated,
        high_accuracy_gate_path=gate,
        output_dir=output_dir,
        generated_at=datetime(2026, 6, 12, 8, 45, tzinfo=timezone.utc),
    )

    assert report["final_status"] == "PROMOTION_DISTANCE_BLOCKED"
    assert report["rolling_sample"]["sample_race_count"] == 140
    assert report["rolling_sample"]["races_needed_for_review_floor"] == 0
    assert report["rolling_sample"]["source_rejected_live_odds_candidate_count"] == 5
    assert (
        report["rolling_sample"]["source_rows_with_rejected_live_odds_candidates"] == 4
    )
    assert report["rolling_sample"][
        "source_rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 2,
        "odds_source_url_missing": 3,
    }
    assert report["rolling_sample"]["source_exclusion_reason_counts"] == {
        "official_result_missing": 32
    }
    assert report["rolling_sample"]["source_odds_exclusion_reason_counts"] == {
        "strict_prejump_odds_missing": 4
    }
    assert report["rolling_sample"][
        "source_official_result_evidence_db_missing_race_ids"
    ] == ["Race 3 - TAREE - 2026-06-13"]
    assert report["rolling_sample"][
        "source_official_result_evidence_db_requested_race_count"
    ] == 7
    assert report["rolling_sample"][
        "source_official_result_evidence_db_races_with_rows"
    ] == [
        "Race 5 - TAREE - 2026-06-13",
        "Race 6 - TAREE - 2026-06-13",
    ]
    assert report["rolling_sample"]["source_official_result_runner_paths"] == [
        "artifacts/full_evidence_orchestration_20260525/autonomous_official_result_capture_test/official_result_runners.jsonl"
    ]
    assert report["official_result_coverage"] == {
        "source": "rolling_model_comparison",
        "requested_race_count": 7,
        "requested_race_count_source": "rolling_model_comparison_source_count",
        "requested_race_ids": [
            "Race 3 - TAREE - 2026-06-13",
            "Race 5 - TAREE - 2026-06-13",
            "Race 6 - TAREE - 2026-06-13",
        ],
        "legacy_requested_race_count_without_ids": 4,
        "races_with_rows_count": 2,
        "missing_race_count": 1,
        "missing_race_ids": ["Race 3 - TAREE - 2026-06-13"],
        "races_with_rows": [
            "Race 5 - TAREE - 2026-06-13",
            "Race 6 - TAREE - 2026-06-13",
        ],
        "runner_path_count": 1,
        "runner_paths_source_field": (
            "rolling_sample.source_official_result_runner_paths"
        ),
        "missing_exclusion_count": 32,
    }
    assert report["market_benchmark"]["best_candidate_key"] == "market_only_implied"
    assert report["market_benchmark"]["best_non_market_top1_margin_gap"] == 0.027
    residual = report["predeclared_residual_candidate"]
    assert residual["triggered_race_count"] == 3
    assert residual["triggered_races_needed_for_directional_read"] == 7
    assert residual["directional_read_ready"] is False
    assert report["promotion_ready"] is False
    assert "no_candidate_passed_rank_first_accuracy_gate" in report["blockers"]
    assert "predeclared_residual_trigger_count_below_directional_floor" in report[
        "blockers"
    ]
    assert report["no_write_guarantees"]["production_promotion"] is False
    assert report["no_write_guarantees"]["db_write"] is False
    assert report["no_write_guarantees"]["betting_action"] is False
    summary = (output_dir / "SUMMARY.md").read_text(encoding="utf-8")
    assert "Rolling source rejected live odds candidates: `5`" in summary
    assert "Rolling source rows with rejected live odds candidates: `4`" in summary
    assert (
        "Rolling source rejected live odds candidate reasons: `{'odds_decimal_invalid': 2, 'odds_source_url_missing': 3}`"
        in summary
    )
    assert "Rolling source exclusion reasons: `{'official_result_missing': 32}`" in summary
    assert "Official-result coverage requested races: `7`" in summary
    assert (
        "Official-result coverage requested race count source: "
        "`rolling_model_comparison_source_count`"
    ) in summary
    assert "Official-result legacy requested race count without IDs: `4`" in summary
    assert "Official-result coverage races with rows: `2`" in summary
    assert "Official-result coverage missing races: `1`" in summary
    assert "Official-result missing exclusion count: `32`" in summary
    assert "Official-result runner path count: `1`" in summary
    assert (
        "Official-result runner paths source field: "
        "`rolling_sample.source_official_result_runner_paths`"
    ) in summary
    assert "Rolling source official-result runner paths:" not in summary
    assert "autonomous_official_result_capture_test/official_result_runners.jsonl" not in summary
    assert "Race 3 - TAREE - 2026-06-13" in summary
    assert (output_dir / "promotion_distance_report.json").exists()
    assert (output_dir / "SUMMARY.md").exists()
    assert (output_dir / "output_manifest.json").exists()


def test_gate_contract_diagnostics_classify_source_not_ready_separately(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(distance, "ROOT", tmp_path)
    primary = _candidate(
        candidate_key="primary_shadow",
        family="baseline",
        top1=0.30,
        top3=0.60,
        mean_winner_rank=3.0,
        brier=0.80,
        logloss=1.70,
        slope=1.0,
        intercept=0.0,
        box1=0.16,
    )
    market = _candidate(
        candidate_key="market_only_implied",
        family="market_only",
        top1=0.44,
        top3=0.80,
        mean_winner_rank=2.35,
        brier=0.70,
        logloss=1.50,
        slope=0.90,
        intercept=-0.10,
        box1=0.20,
    )
    candidate = _candidate(
        candidate_key="stage2_market_blend_85",
        top1=0.448,
        top3=0.816,
        mean_winner_rank=2.27,
        brier=0.69,
        logloss=1.48,
        slope=0.95,
        intercept=0.05,
        box1=0.21,
    )
    rolling = _write_json(
        tmp_path / "rolling_collecting.json",
        {
            "final_status": "ROLLING_MODEL_COMPARISON_COLLECTING",
            "sample_scope": "unified",
            "sample_floor_met": False,
            "sample_race_count": 33,
            "sample_runner_rows": 222,
            "minimum_races_for_review": 100,
            "best_candidate_key": "stage2_market_blend_85",
            "best_non_market_candidate_key": "stage2_market_blend_85",
            "best_non_market_minus_market": {"top1": 0.008},
            "rank_first_sort": ["stage2_market_blend_85"],
            "baseline_metrics": primary,
            "market_metrics": market,
            "candidate_metrics_by_key": {
                "primary_shadow": primary,
                "market_only_implied": market,
                "stage2_market_blend_85": candidate,
            },
        },
    )
    gated = _write_json(
        tmp_path / "gated.json",
        {"predeclared_residual_candidate": {"triggered_race_count": 0}},
    )
    gate = _write_json(
        tmp_path / "gate.json",
        {
            "status": "BLOCKED",
            "blockers": ["no_candidate_passed_rank_first_accuracy_gate"],
            "pull_request_boundary": {"promotion_pr_allowed": False},
        },
    )

    report = distance.build_report(
        rolling_report_path=rolling,
        pre_race_gated_report_path=gated,
        high_accuracy_gate_path=gate,
        output_dir=(
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525"
            / "promotion_distance_report_source_not_ready_test"
        ),
        generated_at=datetime(2026, 6, 23, 9, 0, tzinfo=timezone.utc),
    )

    contract = report["gate_contract_candidate"]
    assert contract["audit_final_status"] == "DATA_MISSING"
    assert contract["audit_classification"] == "SOURCE_NOT_READY"
    assert contract["policy_evaluation_status"] == "NOT_EVALUABLE"
    assert contract["data_missing_reasons"] == []
    assert "rolling_report_status_not_ready:ROLLING_MODEL_COMPARISON_COLLECTING" in (
        contract["source_not_ready_reasons"]
    )
    assert "sample_floor_not_met" in contract["source_not_ready_reasons"]
    assert "rolling_sample_below_review_floor" in contract["source_not_ready_reasons"]
    assert "gate_contract_audit_not_ready:SOURCE_NOT_READY" in contract["blockers"]
    assert (
        "gate_contract_policy_not_evaluable:"
        "dual_baseline_market_rank_primary_safety:SOURCE_NOT_READY"
    ) in contract["blockers"]
    assert not any(
        str(blocker).startswith("gate_contract_policy_failed")
        for blocker in contract["blockers"]
    )
    summary = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525"
        / "promotion_distance_report_source_not_ready_test"
        / "SUMMARY.md"
    ).read_text(encoding="utf-8")
    assert "Gate-contract audit classification: `SOURCE_NOT_READY`" in summary


def test_gate_contract_diagnostics_classify_ready_source_policy_failure(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(distance, "ROOT", tmp_path)
    primary = _candidate(
        candidate_key="primary_shadow",
        family="baseline",
        top1=0.30,
        top3=0.60,
        mean_winner_rank=3.0,
        brier=0.80,
        logloss=1.70,
        slope=1.0,
        intercept=0.0,
        box1=0.16,
    )
    market = _candidate(
        candidate_key="market_only_implied",
        family="market_only",
        top1=0.44,
        top3=0.80,
        mean_winner_rank=2.35,
        brier=0.70,
        logloss=1.50,
        slope=0.90,
        intercept=-0.10,
        box1=0.20,
    )
    failing = _candidate(
        candidate_key="stage2_market_blend_20",
        top1=0.42,
        top3=0.78,
        mean_winner_rank=2.45,
        brier=0.74,
        logloss=1.58,
        slope=0.20,
        intercept=-1.20,
        box1=0.24,
    )
    rolling = _write_json(
        tmp_path / "rolling_ready_policy_failed.json",
        {
            "final_status": "ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW",
            "sample_scope": "unified",
            "sample_floor_met": True,
            "sample_race_count": 120,
            "sample_runner_rows": 840,
            "minimum_races_for_review": 100,
            "best_candidate_key": "stage2_market_blend_20",
            "best_non_market_candidate_key": "stage2_market_blend_20",
            "best_non_market_minus_market": {"top1": -0.02},
            "rank_first_sort": ["stage2_market_blend_20"],
            "baseline_metrics": primary,
            "market_metrics": market,
            "candidate_metrics_by_key": {
                "primary_shadow": primary,
                "market_only_implied": market,
                "stage2_market_blend_20": failing,
            },
        },
    )
    gated = _write_json(
        tmp_path / "gated.json",
        {"predeclared_residual_candidate": {"triggered_race_count": 0}},
    )
    gate = _write_json(
        tmp_path / "gate_ready.json",
        {
            "status": "READY_FOR_PR_DRAFT",
            "selected_stage": "odds_augmented_model_research",
            "selected_candidate": "stage2_market_blend_20",
            "blockers": [],
            "pull_request_boundary": {
                "promotion_pr_allowed": True,
                "direct_local_switch_allowed": False,
                "local_registry_mutation_allowed": False,
                "production_pointer_update_allowed": False,
                "requires_human_pr_review": True,
            },
        },
    )

    report = distance.build_report(
        rolling_report_path=rolling,
        pre_race_gated_report_path=gated,
        high_accuracy_gate_path=gate,
        output_dir=(
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525"
            / "promotion_distance_report_policy_failed_test"
        ),
        generated_at=datetime(2026, 6, 23, 9, 5, tzinfo=timezone.utc),
    )

    contract = report["gate_contract_candidate"]
    assert contract["audit_classification"] == "POLICY_FAILED"
    assert contract["policy_evaluation_status"] == "FAILED"
    assert contract["data_missing_reasons"] == []
    assert contract["source_not_ready_reasons"] == []
    assert "gate_contract_policy_failed:dual_baseline_market_rank_primary_safety" in (
        contract["blockers"]
    )
    assert "metric_delta_below_min:top1" in contract["policy_failure_reasons"]
    assert (
        contract["candidate_policy_blocker_counts"]["metric_delta_below_min:top1"]
        >= 1
    )
    assert "gate_contract_audit_not_ready:DATA_MISSING" not in contract["blockers"]
    assert report["promotion_ready"] is False


def test_gate_contract_candidate_can_make_promotion_distance_review_ready(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(distance, "ROOT", tmp_path)
    primary = _candidate(
        candidate_key="primary_shadow",
        family="baseline",
        top1=0.30,
        top3=0.60,
        mean_winner_rank=3.0,
        brier=0.80,
        logloss=1.70,
        slope=1.0,
        intercept=0.0,
        box1=0.16,
    )
    market = _candidate(
        candidate_key="market_only_implied",
        family="market_only",
        top1=0.44,
        top3=0.80,
        mean_winner_rank=2.35,
        brier=0.70,
        logloss=1.50,
        slope=0.90,
        intercept=-0.10,
        box1=0.20,
    )
    rank_first = _candidate(
        candidate_key="stage2_market_blend_40",
        top1=0.49,
        top3=0.77,
        mean_winner_rank=2.30,
        brier=0.72,
        logloss=1.55,
        slope=1.50,
        intercept=0.70,
        box1=0.19,
    )
    selected = _candidate(
        candidate_key="stage2_market_blend_85",
        top1=0.448,
        top3=0.816,
        mean_winner_rank=2.27,
        brier=0.69,
        logloss=1.48,
        slope=0.95,
        intercept=0.05,
        box1=0.21,
    )
    strict = _candidate(
        candidate_key="stage2_market_blend_90",
        top1=0.44,
        top3=0.82,
        mean_winner_rank=2.30,
        brier=0.69,
        logloss=1.49,
        slope=0.98,
        intercept=0.02,
        box1=0.20,
    )
    rolling = _write_json(
        tmp_path / "rolling.json",
        {
            "final_status": "ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW",
            "sample_scope": "unified",
            "sample_floor_met": True,
            "sample_race_count": 120,
            "sample_runner_rows": 840,
            "minimum_races_for_review": 100,
            "best_candidate_key": "stage2_market_blend_40",
            "best_non_market_candidate_key": "stage2_market_blend_40",
            "best_non_market_minus_market": {
                "top1": 0.05,
                "top3": -0.03,
                "mean_winner_rank": -0.05,
                "brier": 0.02,
                "logloss": 0.03,
            },
            "rank_first_sort": [
                "stage2_market_blend_40",
                "stage2_market_blend_85",
                "stage2_market_blend_90",
            ],
            "baseline_metrics": primary,
            "market_metrics": market,
            "candidate_metrics_by_key": {
                "primary_shadow": primary,
                "market_only_implied": market,
                "stage2_market_blend_40": rank_first,
                "stage2_market_blend_85": selected,
                "stage2_market_blend_90": strict,
            },
        },
    )
    gated = _write_json(
        tmp_path / "gated.json",
        {
            "predeclared_residual_candidate": {
                "candidate_key": "market_favourite_gt_4_0__raw_stage2_market_blend_75",
                "status": "PREDECLARED_RESIDUAL_CANDIDATE_COLLECTING",
                "triggered_race_count": 1,
                "minimum_triggered_races_for_directional_read": 10,
                "directional_read_ready": False,
                "candidate_minus_market": {
                    "top1": 0.0,
                    "top3": 0.0,
                    "mean_winner_rank": 0.0,
                    "brier": -0.0001,
                    "logloss": -0.0001,
                },
            },
        },
    )
    gate = _write_json(
        tmp_path / "gate.json",
        {
            "status": "READY_FOR_PR_DRAFT",
            "selected_stage": "odds_augmented_model_research",
            "selected_candidate": "stage2_market_blend_85",
            "blockers": [],
            "pull_request_boundary": {
                "promotion_pr_allowed": True,
                "direct_local_switch_allowed": False,
                "local_registry_mutation_allowed": False,
                "production_pointer_update_allowed": False,
                "requires_human_pr_review": True,
            },
        },
    )
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525"
        / "promotion_distance_report_gate_contract_test"
    )

    report = distance.build_report(
        rolling_report_path=rolling,
        pre_race_gated_report_path=gated,
        high_accuracy_gate_path=gate,
        output_dir=output_dir,
        generated_at=datetime(2026, 6, 18, 5, 30, tzinfo=timezone.utc),
    )

    assert report["final_status"] == "PROMOTION_DISTANCE_REVIEW_READY"
    assert report["promotion_ready"] is True
    assert report["blockers"] == []
    contract = report["gate_contract_candidate"]
    assert contract["status"] == "PASS"
    assert contract["selected_candidate"] == "stage2_market_blend_85"
    assert contract["candidate_minus_market"]["top1"] == 0.008000000000000007
    assert contract["candidate_minus_market"]["top3"] == 0.015999999999999903
    assert contract["top1_margin_gap_to_target"] == 0.011999999999999993
    assert report["market_benchmark"]["selected_gate_contract_candidate_key"] == (
        "stage2_market_blend_85"
    )
    residual = report["predeclared_residual_candidate"]
    assert residual["triggered_race_count"] == 1
    assert residual["promotion_blocking"] is False
    summary = (output_dir / "SUMMARY.md").read_text(encoding="utf-8")
    assert "Gate-contract candidate: `stage2_market_blend_85`" in summary
    assert "Residual promotion blocking: `False`" in summary

    blocked_gate = _write_json(
        tmp_path / "gate_blocked_without_reason.json",
        {
            "status": "BLOCKED",
            "selected_stage": "odds_augmented_model_research",
            "selected_candidate": "stage2_market_blend_85",
            "blockers": [],
        },
    )
    blocked_report = distance.build_report(
        rolling_report_path=rolling,
        pre_race_gated_report_path=gated,
        high_accuracy_gate_path=blocked_gate,
        output_dir=(
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525"
            / "promotion_distance_report_gate_status_blocked_test"
        ),
        generated_at=datetime(2026, 6, 18, 5, 31, tzinfo=timezone.utc),
    )

    assert blocked_report["final_status"] == "PROMOTION_DISTANCE_BLOCKED"
    assert blocked_report["promotion_ready"] is False
    assert "high_accuracy_gate_not_ready:BLOCKED" in blocked_report["blockers"]

    inconsistent_ready_gate = _write_json(
        tmp_path / "gate_ready_with_stale_blockers.json",
        {
            "status": "READY_FOR_PR_DRAFT",
            "selected_stage": "stage_2_non_tgr_model_challenger",
            "selected_candidate": "stage2_market_blend_85",
            "blockers": ["stale_gate_blocker"],
            "pull_request_boundary": {
                "promotion_pr_allowed": True,
                "direct_local_switch_allowed": True,
                "local_registry_mutation_allowed": False,
                "production_pointer_update_allowed": False,
                "requires_human_pr_review": False,
            },
        },
    )
    inconsistent_report = distance.build_report(
        rolling_report_path=rolling,
        pre_race_gated_report_path=gated,
        high_accuracy_gate_path=inconsistent_ready_gate,
        output_dir=(
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525"
            / "promotion_distance_report_inconsistent_gate_test"
        ),
        generated_at=datetime(2026, 6, 18, 5, 32, tzinfo=timezone.utc),
    )

    assert inconsistent_report["final_status"] == "PROMOTION_DISTANCE_BLOCKED"
    assert inconsistent_report["promotion_ready"] is False
    assert "stale_gate_blocker" in inconsistent_report["blockers"]
    assert (
        "high_accuracy_selected_stage_not_supported:"
        "stage_2_non_tgr_model_challenger"
    ) in inconsistent_report["blockers"]
    assert (
        "high_accuracy_pull_request_boundary_invalid:"
        "direct_local_switch_allowed=True"
    ) in inconsistent_report["blockers"]
    assert (
        "high_accuracy_pull_request_boundary_invalid:"
        "requires_human_pr_review=False"
    ) in inconsistent_report["blockers"]
