import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import build_pre_race_gated_challenger_packet as gated


def _runner_row(
    *,
    race_index: int,
    box: int,
    winner_box: int,
    runner_count: int,
    market_probability: float,
    candidate_probability: float,
    stage2_probability: float,
    odds_decimal: float,
) -> dict[str, object]:
    market_rank_by_box = {1: 1, 2: 2, 3: 3}
    candidate_rank_by_box = {2: 1, 1: 2, 3: 3}
    return {
        "candidate_key": "stage2_uncalibrated_market_blend_75",
        "market_candidate_key": "market_only_implied",
        "race_id": f"Race {race_index} - TEST - 2026-06-10",
        "source_report": "artifacts/source/report.json",
        "venue": "TEST",
        "race_number": race_index,
        "race_date": "2026-06-10",
        "dog_name": f"Dog {race_index}-{box}",
        "box_number": box,
        "is_winner": box == winner_box,
        "finish_position": 1 if box == winner_box else 2,
        "odds_decimal": odds_decimal,
        "odds_source_url": "https://www.sportsbet.com.au/greyhound-racing/example",
        "odds_capture_timestamp": "2026-06-10T12:30:00+10:00",
        "odds_capture_mode": "autonomous_prejump_t10m",
        "odds_level": "dog",
        "market_favourite_odds_decimal": 1.9,
        "market_favourite_odds_band": "market_favourite_odds_lte_2",
        "market_probability": market_probability,
        "candidate_probability": candidate_probability,
        "candidate_minus_market_probability": candidate_probability - market_probability,
        "primary_shadow_probability_norm": stage2_probability,
        "stage2_shadow_probability_norm": stage2_probability,
        "stage2_shadow_uncalibrated_probability_norm": stage2_probability,
        "market_rank": market_rank_by_box[box],
        "candidate_rank": candidate_rank_by_box[box],
        "primary_shadow_rank": candidate_rank_by_box[box],
        "stage2_shadow_rank": candidate_rank_by_box[box],
        "stage2_shadow_uncalibrated_rank": candidate_rank_by_box[box],
        "market_top_pick": box == 1,
        "candidate_top_pick": box == 2,
        "market_favourite": box == 1,
        "candidate_agrees_with_market_top": False,
        "runner_count": runner_count,
    }


def _rows(race_count: int = 6) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for race_index in range(1, race_count + 1):
        runner_count = 5 if race_index % 2 else 6
        winner_box = 2 if runner_count == 5 else 1
        rows.extend(
            [
                _runner_row(
                    race_index=race_index,
                    box=1,
                    winner_box=winner_box,
                    runner_count=runner_count,
                    market_probability=0.60,
                    candidate_probability=0.25,
                    stage2_probability=0.25,
                    odds_decimal=1.9,
                ),
                _runner_row(
                    race_index=race_index,
                    box=2,
                    winner_box=winner_box,
                    runner_count=runner_count,
                    market_probability=0.30,
                    candidate_probability=0.65,
                    stage2_probability=0.65,
                    odds_decimal=4.2,
                ),
                _runner_row(
                    race_index=race_index,
                    box=3,
                    winner_box=winner_box,
                    runner_count=runner_count,
                    market_probability=0.10,
                    candidate_probability=0.10,
                    stage2_probability=0.10,
                    odds_decimal=9.0,
                ),
            ]
        )
    return rows


def _write_matrix(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "market_residual_runner_matrix.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_pre_race_gated_challenger_writes_report_only_packet(tmp_path, monkeypatch):
    monkeypatch.setattr(gated, "ROOT", tmp_path)
    matrix_path = _write_matrix(tmp_path, _rows())
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525"
        / "pre_race_gated_challenger_test"
    )

    report = gated.build_packet(
        runner_matrix_csv=matrix_path,
        output_dir=output_dir,
        fold_count=3,
        min_train_races=2,
        min_races_for_review=3,
        generated_at=datetime(2026, 6, 12, 1, 0, tzinfo=timezone.utc),
    )

    assert report["schema_version"] == "pre_race_gated_challenger_report_v1"
    assert report["final_status"] == "PRE_RACE_GATED_CHALLENGER_REVIEW_READY"
    assert report["accepted_race_count"] == 6
    assert report["matrix_row_count"] == 18
    assert report["candidate_grid_count"] == len(gated.candidate_specs())
    assert report["evaluated_fold_count"] == 3
    residual = report["predeclared_residual_candidate"]
    assert residual["candidate_key"] == (
        "market_favourite_gt_4_0__raw_stage2_market_blend_75"
    )
    assert residual["minimum_triggered_races_for_directional_read"] == 10
    assert residual["promotion_eligible"] is False
    assert "predeclared_residual_candidate_report_only" in residual["blockers"]
    assert report["promotion_gate"]["promotion_ready"] is False
    assert "report_only_pre_race_gated_challenger_not_promotion_eligible" in report[
        "promotion_gate"
    ]["blockers"]
    assert "requires_fresh_future_out_of_sample_packet" in report["promotion_gate"][
        "blockers"
    ]
    assert report["no_write_guarantees"]["production_promotion"] is False
    assert report["no_write_guarantees"]["db_write"] is False
    assert report["no_write_guarantees"]["betting_or_ev_action"] is False

    assert (output_dir / "pre_race_gated_challenger_report.json").exists()
    assert (output_dir / "candidate_metrics.csv").exists()
    assert (output_dir / "cross_validated_fold_summary.csv").exists()
    assert (output_dir / "cross_validated_race_predictions.csv").exists()
    assert (output_dir / "SUMMARY.md").exists()
    assert (output_dir / "output_manifest.json").exists()

    with (output_dir / "candidate_metrics.csv").open(encoding="utf-8", newline="") as handle:
        candidate_rows = list(csv.DictReader(handle))
    assert len(candidate_rows) == report["candidate_grid_count"]
    assert {"gate_key", "score_mode", "gate_triggered_race_count"}.issubset(
        candidate_rows[0]
    )

    with (output_dir / "cross_validated_fold_summary.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        fold_rows = list(csv.DictReader(handle))
    assert len(fold_rows) == 3
    assert {row["status"] for row in fold_rows} == {"EVALUATED"}

    with (output_dir / "cross_validated_race_predictions.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        prediction_rows = list(csv.DictReader(handle))
    assert len(prediction_rows) == 6
    assert "gate_triggered" in prediction_rows[0]
    assert "challenger_minus_market_logloss" in prediction_rows[0]

    manifest = json.loads((output_dir / "output_manifest.json").read_text())
    manifest_files = "\n".join(manifest["files"])
    assert "candidate_metrics.csv" in manifest_files
    assert "cross_validated_fold_summary.csv" in manifest_files
    assert "cross_validated_race_predictions.csv" in manifest_files


def test_rank_first_hypothesis_review_stays_report_only(tmp_path, monkeypatch):
    monkeypatch.setattr(gated, "ROOT", tmp_path)
    matrix_path = _write_matrix(tmp_path, _rows(race_count=12))
    hypotheses_path = tmp_path / "next_hypotheses.json"
    hypotheses_path.write_text(
        json.dumps(
            {
                "schema_version": "market_residual_next_hypotheses_v1",
                "pre_race_rank_first_help_regimes": [
                    {
                        "dimension": "venue",
                        "dimension_value": "TEST",
                        "pre_race_usable": True,
                        "race_count": 12,
                        "stage2_uncalibrated_mean_top1_delta": 0.25,
                    },
                    {
                        "dimension": "winner_box_number",
                        "dimension_value": "2",
                        "pre_race_usable": False,
                        "race_count": 6,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525"
        / "pre_race_gated_challenger_rank_first"
    )

    report = gated.build_packet(
        runner_matrix_csv=matrix_path,
        output_dir=output_dir,
        fold_count=3,
        min_train_races=2,
        min_races_for_review=3,
        rank_first_hypotheses_json=hypotheses_path,
        generated_at=datetime(2026, 6, 12, 1, 0, tzinfo=timezone.utc),
    )

    review = report["rank_first_hypothesis_gate_review"]
    assert review["status"] == "RANK_FIRST_HYPOTHESIS_REVIEW_READY"
    assert review["promotion_eligible"] is False
    assert review["candidate_count"] == 1
    assert review["evaluated_candidate_count"] == 1
    assert review["best_candidate_key"] == (
        "rank_first_hypothesis_venue_eq_test__raw_stage2_uncalibrated"
    )
    assert review["best_candidate"]["gate_triggered_race_count"] == 12
    assert "rank_first_hypothesis_review_report_only" in review["blockers"]
    assert "requires_fresh_future_out_of_sample_packet" in review["blockers"]
    assert review["source"]["unsupported_hypotheses"][0]["dimension"] == "winner_box_number"
    assert report["promotion_gate"]["promotion_ready"] is False
    assert report["no_write_guarantees"]["production_promotion"] is False
    assert report["no_write_guarantees"]["betting_or_ev_action"] is False

    with (output_dir / "rank_first_hypothesis_candidate_metrics.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        metric_rows = list(csv.DictReader(handle))
    assert len(metric_rows) == 1
    assert metric_rows[0]["hypothesis_dimension"] == "venue"
    assert metric_rows[0]["hypothesis_dimension_value"] == "TEST"
    assert metric_rows[0]["gate_triggered_race_count"] == "12"

    summary = (output_dir / "SUMMARY.md").read_text(encoding="utf-8")
    assert "Rank-first hypothesis review" in summary
    assert "RANK_FIRST_HYPOTHESIS_REVIEW_READY" in summary


def test_pre_race_gated_challenger_collects_when_train_folds_underpowered(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(gated, "ROOT", tmp_path)
    matrix_path = _write_matrix(tmp_path, _rows(race_count=4))
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525"
        / "pre_race_gated_challenger_underpowered"
    )

    report = gated.build_packet(
        runner_matrix_csv=matrix_path,
        output_dir=output_dir,
        fold_count=2,
        min_train_races=10,
        min_races_for_review=3,
        generated_at=datetime(2026, 6, 12, 1, 0, tzinfo=timezone.utc),
    )

    assert report["final_status"] == "PRE_RACE_GATED_CHALLENGER_COLLECTING"
    assert report["accepted_race_count"] == 4
    assert report["challenger_metrics"]["status"] == "NO_EVALUABLE_RACES"
    assert report["evaluated_fold_count"] == 0
    assert report["promotion_gate"]["promotion_ready"] is False
    assert "cross_validated_challenger_not_evaluated" in report["blockers"]

    with (output_dir / "cross_validated_fold_summary.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        fold_rows = list(csv.DictReader(handle))
    assert len(fold_rows) == 2
    assert {row["status"] for row in fold_rows} == {
        "SKIPPED_TRAIN_RACE_COUNT_BELOW_MINIMUM"
    }

    with (output_dir / "cross_validated_race_predictions.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        prediction_rows = list(csv.DictReader(handle))
    assert prediction_rows == []
