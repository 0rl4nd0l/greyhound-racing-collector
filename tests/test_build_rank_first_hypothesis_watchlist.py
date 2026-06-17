import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import build_rank_first_hypothesis_watchlist as watchlist


def _write_packet(
    *,
    evidence_root: Path,
    run_id: str,
    source_reports: list[str],
    race_ids: list[str],
    top1_delta: float,
    trigger_count: int,
) -> Path:
    rolling_dir = evidence_root / f"rolling_model_comparison_{run_id}_daemon_rejoin"
    rolling_dir.mkdir(parents=True)
    runner_matrix = rolling_dir / "market_residual_runner_matrix.csv"
    runner_matrix.write_text(
        "race_id,dog_name\n"
        + "\n".join(f"{race_id},Dog {index}" for index, race_id in enumerate(race_ids))
        + "\n",
        encoding="utf-8",
    )
    (rolling_dir / "rolling_model_comparison_report.json").write_text(
        json.dumps(
            {
                "schema_version": "rolling_model_comparison_report_v1",
                "sample_race_count": 124,
                "sample_runner_rows": 856,
                "source_unified_evidence_reports": source_reports,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    packet_dir = (
        evidence_root
        / f"pre_race_gated_challenger_{run_id}_daemon_rejoin_rank_first_hypothesis_review"
    )
    packet_dir.mkdir(parents=True)
    (packet_dir / "pre_race_gated_challenger_report.json").write_text(
        json.dumps(
            {
                "schema_version": "pre_race_gated_challenger_report_v1",
                "generated_at": f"2026-06-13T{run_id[-6:-4]}:00:00+10:00",
                "runner_matrix_csv": str(runner_matrix),
                "rank_first_hypothesis_gate_review": {
                    "status": "RANK_FIRST_HYPOTHESIS_REVIEW_READY"
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with (packet_dir / "rank_first_hypothesis_candidate_metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_key",
                "hypothesis_dimension",
                "hypothesis_dimension_value",
                "hypothesis_source_race_count",
                "gate_key",
                "score_mode",
                "gate_triggered_race_count",
                "status",
                "race_count",
                "top1",
                "top3",
                "mean_winner_rank",
                "brier",
                "logloss",
                "box1_top_pick_share",
                "candidate_minus_market",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "candidate_key": (
                    "rank_first_hypothesis_runner_count_eq_4__raw_stage2_uncalibrated"
                ),
                "hypothesis_dimension": "runner_count",
                "hypothesis_dimension_value": "4",
                "hypothesis_source_race_count": "13",
                "gate_key": "rank_first_hypothesis_runner_count_eq_4",
                "score_mode": "raw_stage2_uncalibrated",
                "gate_triggered_race_count": str(trigger_count),
                "status": "EVALUATED",
                "race_count": "124",
                "top1": "0.4032258064516129",
                "top3": "0.7741935483870968",
                "mean_winner_rank": "2.556451612903226",
                "brier": "0.7546792288161775",
                "logloss": "1.6412848582006183",
                "box1_top_pick_share": "0.2661290322580645",
                "candidate_minus_market": {
                    "top1": top1_delta,
                    "top3": 0.008064516129032251,
                    "mean_winner_rank": -0.008064516129032029,
                    "brier": -0.0072532184848540515,
                    "logloss": -0.01229868660097111,
                },
            }
        )
    return packet_dir


def test_rank_first_hypothesis_watchlist_requires_distinct_future_samples(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(watchlist, "ROOT", tmp_path)
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    _write_packet(
        evidence_root=evidence_root,
        run_id="20260613T060211+1000",
        source_reports=["a.json", "b.json"],
        race_ids=["race_a", "race_b"],
        top1_delta=0.008,
        trigger_count=13,
    )
    _write_packet(
        evidence_root=evidence_root,
        run_id="20260613T061711+1000",
        source_reports=["a.json", "b.json"],
        race_ids=["race_a", "race_b"],
        top1_delta=0.008,
        trigger_count=13,
    )
    output_dir = (
        evidence_root
        / "rank_first_hypothesis_watchlist_same_sample"
    )

    report = watchlist.build_packet(
        evidence_root=evidence_root,
        output_dir=output_dir,
        generated_at=datetime(2026, 6, 13, 0, 0, tzinfo=timezone.utc),
    )

    assert report["final_status"] == "RANK_FIRST_HYPOTHESIS_WATCHLIST_READY"
    assert report["packet_count"] == 2
    assert report["evaluation_count"] == 2
    assert report["candidate_count"] == 1
    best = report["best_candidate"]
    assert best["distinct_sample_signature_count"] == 1
    assert best["status"] == "RANK_FIRST_HYPOTHESIS_WAITING_FOR_FRESH_SAMPLE"
    assert "needs_distinct_future_sample" in best["blockers"]
    assert "triggered_race_count_below_directional_floor" not in best["blockers"]
    assert report["directional_ready_candidate_count"] == 0
    assert "no_directional_ready_rank_first_hypotheses" in report["blockers"]
    assert (output_dir / "rank_first_hypothesis_watchlist_report.json").exists()
    assert (output_dir / "rank_first_hypothesis_watchlist.csv").exists()
    assert (output_dir / "rank_first_hypothesis_evaluations.csv").exists()


def test_rank_first_hypothesis_watchlist_marks_directional_ready_on_fresh_sample(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(watchlist, "ROOT", tmp_path)
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    _write_packet(
        evidence_root=evidence_root,
        run_id="20260613T060211+1000",
        source_reports=["a.json", "b.json"],
        race_ids=["race_a", "race_b"],
        top1_delta=0.008,
        trigger_count=13,
    )
    _write_packet(
        evidence_root=evidence_root,
        run_id="20260613T061711+1000",
        source_reports=["a.json", "b.json", "c.json"],
        race_ids=["race_a", "race_b", "race_c"],
        top1_delta=0.008,
        trigger_count=13,
    )
    output_dir = (
        evidence_root
        / "rank_first_hypothesis_watchlist_fresh_sample"
    )

    report = watchlist.build_packet(
        evidence_root=evidence_root,
        output_dir=output_dir,
        generated_at=datetime(2026, 6, 13, 0, 0, tzinfo=timezone.utc),
    )

    best = report["best_candidate"]
    assert best["distinct_sample_signature_count"] == 2
    assert best["status"] == "RANK_FIRST_HYPOTHESIS_DIRECTIONAL_READY"
    assert best["blockers"] == []
    assert report["directional_ready_candidate_count"] == 1
    assert report["blockers"] == []
