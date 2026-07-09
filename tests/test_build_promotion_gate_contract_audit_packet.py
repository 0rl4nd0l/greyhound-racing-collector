import json
import shutil
from datetime import datetime, timezone

from scripts import build_promotion_gate_contract_audit_packet as packet


def _metrics(
    *,
    candidate_key: str,
    family: str = "odds_augmented_blend",
    races: int = 120,
    top1: float = 0.30,
    top3: float = 0.60,
    mean_winner_rank: float = 3.0,
    brier: float = 0.80,
    logloss: float = 1.70,
    slope: float = 1.0,
    intercept: float = 0.0,
    box1: float = 0.16,
    status: str = "EVALUATED",
    evaluated_race_ids_hash: str = "trusted_race_hash",
) -> dict:
    return {
        "candidate_key": candidate_key,
        "family": family,
        "status": status,
        "race_count": races,
        "evaluated_race_ids_hash": evaluated_race_ids_hash,
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


def _rolling_report(candidates: dict[str, dict], rank_first_sort: list[str]) -> dict:
    by_key = {
        "primary_shadow": _metrics(
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
        ),
        "market_only_implied": _metrics(
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
        ),
        **candidates,
    }
    return {
        "schema_version": "rolling_model_comparison_report_v1",
        "final_status": packet.ROLLING_READY_STATUS,
        "sample_scope": "unified",
        "sample_floor_met": True,
        "sample_race_count": 120,
        "sample_runner_rows": 840,
        "candidate_count": len(by_key),
        "best_candidate_key": rank_first_sort[0] if rank_first_sort else None,
        "best_non_market_candidate_key": rank_first_sort[0] if rank_first_sort else None,
        "rank_first_sort": rank_first_sort,
        "baseline_metrics": by_key["primary_shadow"],
        "market_metrics": by_key["market_only_implied"],
        "candidate_metrics_by_key": by_key,
    }


def _policy(report: dict, policy_key: str) -> dict:
    return {
        row["policy_key"]: row
        for row in report["policy_summaries"]
    }[policy_key]


def test_relaxed_market_pass_requires_gate_policy_review():
    rolling_report = _rolling_report(
        {
            "stage2_market_blend_40": _metrics(
                candidate_key="stage2_market_blend_40",
                top1=0.49,
                top3=0.77,
                mean_winner_rank=2.30,
                brier=0.72,
                logloss=1.55,
                slope=1.50,
                intercept=0.70,
                box1=0.19,
            ),
            "stage2_market_blend_55": _metrics(
                candidate_key="stage2_market_blend_55",
                top1=0.46,
                top3=0.81,
                mean_winner_rank=2.25,
                brier=0.69,
                logloss=1.49,
                slope=1.40,
                intercept=0.60,
                box1=0.22,
            ),
        },
        ["stage2_market_blend_40", "stage2_market_blend_55"],
    )

    report = packet.build_report(
        rolling_report=rolling_report,
        high_accuracy_report={
            "final_status": "BLOCKED_KEEP_BASELINE",
            "promotion_pr_gate": {
                "status": "BLOCKED",
                "blockers": ["no_candidate_passed_rank_first_accuracy_gate"],
            },
        },
        generated_at=datetime(2026, 6, 18, 5, 0, tzinfo=timezone.utc),
    )

    assert report["final_status"] == "GATE_POLICY_REVIEW_REQUIRED"
    assert _policy(
        report, "current_primary_relative"
    )["status"] == "BLOCKED"
    cap_only = _policy(report, "market_relative_rank_safe_box_cap_only")
    assert cap_only["status"] == "PASS"
    assert cap_only["selected_candidate"] == "stage2_market_blend_55"
    assert _policy(
        report, "market_relative_rank_safe_box_not_above_market"
    )["status"] == "BLOCKED"
    assert _policy(
        report, "market_relative_strict_calibration_box_not_above_market"
    )["status"] == "BLOCKED"
    assert _policy(
        report, "dual_baseline_market_rank_primary_safety"
    )["status"] == "BLOCKED"
    row = {
        item["candidate_key"]: item
        for item in report["candidate_gate_matrix"]
    }["stage2_market_blend_55"]
    assert row["market_relative_rank_safe_box_cap_only_status"] == "PASS"
    assert "metric_regressed:box1_top_pick_share" in row[
        "market_relative_rank_safe_box_not_above_market_blockers"
    ]


def test_strict_market_and_dual_pass_emit_report_only_gate_change_candidate():
    rolling_report = _rolling_report(
        {
            "stage2_market_blend_90": _metrics(
                candidate_key="stage2_market_blend_90",
                top1=0.44,
                top3=0.82,
                mean_winner_rank=2.30,
                brier=0.69,
                logloss=1.49,
                slope=0.98,
                intercept=0.02,
                box1=0.20,
            ),
        },
        ["stage2_market_blend_90"],
    )

    report = packet.build_report(
        rolling_report=rolling_report,
        generated_at=datetime(2026, 6, 18, 5, 0, tzinfo=timezone.utc),
    )

    assert report["final_status"] == "REPORT_ONLY_GATE_CHANGE_CANDIDATE"
    strict = _policy(
        report, "market_relative_strict_calibration_box_not_above_market"
    )
    dual = _policy(report, "dual_baseline_market_rank_primary_safety")
    assert strict["status"] == "PASS"
    assert strict["selected_candidate"] == "stage2_market_blend_90"
    assert dual["status"] == "PASS"
    assert report["no_write_guarantees"]["production_promotion"] is False
    assert "separate reviewed implementation" in report["recommended_next_action"]


def test_denominator_mismatch_candidate_is_not_gate_eligible():
    sparse = _metrics(
        candidate_key="stage2_sparse_candidate",
        races=60,
        evaluated_race_ids_hash="sparse_race_hash",
        top1=0.50,
        top3=0.82,
        mean_winner_rank=2.20,
        brier=0.68,
        logloss=1.48,
        slope=0.98,
        intercept=0.02,
        box1=0.20,
    )
    rolling_report = _rolling_report(
        {"stage2_sparse_candidate": sparse},
        ["stage2_sparse_candidate"],
    )

    report = packet.build_report(
        rolling_report=rolling_report,
        generated_at=datetime(2026, 6, 18, 5, 0, tzinfo=timezone.utc),
    )

    row = {
        item["candidate_key"]: item
        for item in report["candidate_gate_matrix"]
    }["stage2_sparse_candidate"]
    assert row["market_relative_rank_safe_box_cap_only_status"] == "BLOCKED"
    assert "candidate_denominator_mismatch_primary_shadow" in row[
        "market_relative_rank_safe_box_cap_only_blockers"
    ]
    assert _policy(
        report, "market_relative_rank_safe_box_cap_only"
    )["status"] == "BLOCKED"
    assert report["final_status"] == "KEEP_BASELINE_GATE_VALID"


def test_missing_market_baseline_is_data_missing():
    rolling_report = _rolling_report({}, [])
    rolling_report["candidate_metrics_by_key"].pop("market_only_implied")
    rolling_report.pop("market_metrics")

    report = packet.build_report(
        rolling_report=rolling_report,
        generated_at=datetime(2026, 6, 18, 5, 0, tzinfo=timezone.utc),
    )

    assert report["final_status"] == "DATA_MISSING"
    assert report["blockers"] == ["market_only_baseline_missing"]


def test_primary_shadow_baseline_is_not_promotable_candidate():
    rolling_report = _rolling_report({}, ["primary_shadow"])
    rolling_report["candidate_metrics_by_key"]["primary_shadow"].update(
        {
            "top1": 0.50,
            "top3": 0.90,
            "mean_winner_rank": 2.0,
            "brier": 0.60,
            "logloss": 1.20,
            "box1_top_pick_share": 0.18,
            "calibration_slope_intercept": {
                "status": "computed",
                "slope": 1.0,
                "intercept": 0.0,
            },
        }
    )

    report = packet.build_report(
        rolling_report=rolling_report,
        generated_at=datetime(2026, 6, 18, 5, 0, tzinfo=timezone.utc),
    )

    assert report["final_status"] == "KEEP_BASELINE_GATE_VALID"
    by_key = {
        row["candidate_key"]: row
        for row in report["candidate_gate_matrix"]
    }
    assert "baseline_candidate_not_promotable" in by_key["primary_shadow"][
        "market_relative_rank_safe_box_cap_only_blockers"
    ]
    assert _policy(
        report, "market_relative_rank_safe_box_cap_only"
    )["status"] == "BLOCKED"


def test_run_packet_writes_only_report_artifacts(tmp_path):
    rolling_report = _rolling_report(
        {
            "stage2_market_blend_90": _metrics(
                candidate_key="stage2_market_blend_90",
                top1=0.44,
                top3=0.82,
                mean_winner_rank=2.30,
                brier=0.69,
                logloss=1.49,
                slope=0.98,
                intercept=0.02,
                box1=0.20,
            ),
        },
        ["stage2_market_blend_90"],
    )
    rolling_report_path = tmp_path / "rolling_model_comparison_report.json"
    rolling_report_path.write_text(json.dumps(rolling_report), encoding="utf-8")
    output_dir = (
        packet.DEFAULT_EVIDENCE_ROOT
        / f"promotion_gate_contract_audit_pytest_{tmp_path.name}_report_only"
    )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    try:
        result = packet.run_packet(
            rolling_report_path=rolling_report_path,
            output_dir=output_dir,
        )

        assert result["final_status"] == "REPORT_ONLY_GATE_CHANGE_CANDIDATE"
        assert (output_dir / packet.CANDIDATE_MATRIX_CSV).exists()
        assert (output_dir / packet.POLICY_SUMMARY_CSV).exists()
        assert (output_dir / packet.REPORT_FILE).exists()
        assert (output_dir / packet.SUMMARY_FILE).exists()
        assert (output_dir / packet.FINAL_STATUS_FILE).exists()
        assert (output_dir / packet.OUTPUT_MANIFEST_FILE).exists()
        written = json.loads((output_dir / packet.REPORT_FILE).read_text())
        assert written["no_write_guarantees"]["db_write"] is False
        assert written["no_write_guarantees"]["label_write"] is False
        assert written["no_write_guarantees"]["ev_or_betting_action"] is False
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
