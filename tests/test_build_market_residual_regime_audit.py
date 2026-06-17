import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import build_market_residual_regime_audit as audit


def _matrix_row(
    *,
    race_index: int,
    box: int,
    winner_box: int,
    market_probability: float,
    candidate_probability: float,
    stage2_probability: float,
    odds_decimal: float,
) -> dict[str, object]:
    market_rank = {1: 1, 2: 2, 3: 3}[box]
    candidate_rank = {2: 1, 1: 2, 3: 3}[box]
    return {
        "candidate_key": "stage2_uncalibrated_market_blend_75",
        "market_candidate_key": "market_only_implied",
        "race_id": f"Race {race_index} - TEST - 2026-06-10",
        "source_report": "artifacts/source/report.json",
        "venue": "TEST" if race_index <= 4 else "ALT",
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
        "market_favourite_odds_decimal": 1.8,
        "market_favourite_odds_band": "market_favourite_odds_lte_2",
        "market_probability": market_probability,
        "candidate_probability": candidate_probability,
        "candidate_minus_market_probability": candidate_probability - market_probability,
        "primary_shadow_probability_norm": stage2_probability,
        "stage2_shadow_probability_norm": stage2_probability,
        "stage2_shadow_uncalibrated_probability_norm": stage2_probability,
        "market_rank": market_rank,
        "candidate_rank": candidate_rank,
        "primary_shadow_rank": candidate_rank,
        "stage2_shadow_rank": candidate_rank,
        "stage2_shadow_uncalibrated_rank": candidate_rank,
        "market_top_pick": box == 1,
        "candidate_top_pick": box == 2,
        "market_favourite": box == 1,
        "candidate_agrees_with_market_top": False,
        "runner_count": 3,
    }


def _rows(race_count: int = 6) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for race_index in range(1, race_count + 1):
        winner_box = 2 if race_index % 2 else 1
        rows.extend(
            [
                _matrix_row(
                    race_index=race_index,
                    box=1,
                    winner_box=winner_box,
                    market_probability=0.60,
                    candidate_probability=0.25,
                    stage2_probability=0.25,
                    odds_decimal=1.8 if winner_box == 1 else 1.8,
                ),
                _matrix_row(
                    race_index=race_index,
                    box=2,
                    winner_box=winner_box,
                    market_probability=0.30,
                    candidate_probability=0.65,
                    stage2_probability=0.65,
                    odds_decimal=4.4,
                ),
                _matrix_row(
                    race_index=race_index,
                    box=3,
                    winner_box=winner_box,
                    market_probability=0.10,
                    candidate_probability=0.10,
                    stage2_probability=0.10,
                    odds_decimal=9.0,
                ),
            ]
        )
    return rows


def _prediction_rows(race_count: int = 6) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for race_index in range(1, race_count + 1):
        candidate_wins = race_index % 2 == 1
        rows.append(
            {
                "fold": race_index % 3,
                "selected_candidate_key": "market_anchor_stage2_uncalibrated_blend50_fav_gt_1_5",
                "race_id": f"Race {race_index} - TEST - 2026-06-10",
                "race_date": "2026-06-10",
                "venue": "TEST" if race_index <= 4 else "ALT",
                "race_number": race_index,
                "winner_dog_name": f"Dog {race_index}-{2 if candidate_wins else 1}",
                "winner_box_number": 2 if candidate_wins else 1,
                "winner_odds_decimal": 4.4 if candidate_wins else 1.8,
                "challenger_winner_rank": 1 if candidate_wins else 2,
                "market_winner_rank": 2 if candidate_wins else 1,
                "challenger_winner_probability": 0.65 if candidate_wins else 0.25,
                "market_winner_probability": 0.30 if candidate_wins else 0.60,
                "challenger_logloss": 0.43 if candidate_wins else 1.38,
                "market_logloss": 1.20 if candidate_wins else 0.51,
                "challenger_minus_market_logloss": -0.77 if candidate_wins else 0.87,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_market_residual_regime_audit_writes_diagnostic_packet(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    matrix_path = _write_csv(tmp_path / "market_residual_runner_matrix.csv", _rows())
    predictions_path = _write_csv(
        tmp_path / "cross_validated_race_predictions.csv",
        _prediction_rows(),
    )
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525"
        / "market_residual_regime_audit_test"
    )

    report = audit.build_audit(
        runner_matrix_csv=matrix_path,
        race_predictions_csv=predictions_path,
        output_dir=output_dir,
        min_races_for_review=4,
        min_regime_races=2,
        generated_at=datetime(2026, 6, 12, 1, 0, tzinfo=timezone.utc),
    )

    assert report["schema_version"] == "market_residual_regime_audit_report_v1"
    assert report["final_status"] == "MARKET_RESIDUAL_REGIME_AUDIT_READY"
    assert report["accepted_race_count"] == 6
    assert report["matrix_row_count"] == 18
    assert report["prediction_row_count"] == 6
    assert report["regime_summary_count"] > 0
    assert report["promotion_ready"] is False
    assert report["no_write_guarantees"]["production_promotion"] is False
    assert report["no_write_guarantees"]["db_write"] is False
    assert report["no_write_guarantees"]["betting_or_ev_action"] is False
    assert (output_dir / "market_residual_regime_audit_report.json").exists()
    assert (output_dir / "race_regime_ledger.csv").exists()
    assert (output_dir / "regime_summary.csv").exists()
    assert (output_dir / "next_hypotheses.json").exists()
    assert (output_dir / "SUMMARY.md").exists()
    assert (output_dir / "output_manifest.json").exists()

    with (output_dir / "race_regime_ledger.csv").open(encoding="utf-8", newline="") as handle:
        ledger = list(csv.DictReader(handle))
    assert len(ledger) == 6
    assert "stage2_uncalibrated_minus_market_logloss" in ledger[0]

    with (output_dir / "regime_summary.csv").open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    runner_count_rows = [row for row in summary_rows if row["dimension"] == "runner_count"]
    winner_odds_rows = [row for row in summary_rows if row["dimension"] == "winner_odds_band"]
    assert runner_count_rows
    assert runner_count_rows[0]["pre_race_usable"] == "True"
    assert winner_odds_rows
    assert {row["pre_race_usable"] for row in winner_odds_rows} == {"False"}

    hypotheses = json.loads((output_dir / "next_hypotheses.json").read_text())
    assert hypotheses["promotion_ready"] is False
    assert hypotheses["rank_first_readiness"]["status"] in {
        "PRE_RACE_RANK_FIRST_EDGE_CANDIDATE_FOUND",
        "NO_PRE_RACE_RANK_FIRST_EDGE_FOUND",
    }
    assert report["rank_first_hypothesis_status"] == (
        hypotheses["rank_first_readiness"]["status"]
    )
    assert "pre_race_rank_first_help_regimes" in hypotheses
    assert "pre_race_logloss_only_help_regimes" in hypotheses
    if hypotheses["rank_first_readiness"]["status"] == "NO_PRE_RACE_RANK_FIRST_EDGE_FOUND":
        assert hypotheses["recommended_next_tests"][0]["status"] == (
            "WAIT_FOR_RANK_FIRST_EDGE_OR_NEW_HYPOTHESIS"
        )
    else:
        assert hypotheses["recommended_next_tests"][0]["status"] == (
            "REQUIRES_NEW_REPORT_ONLY_PACKET"
        )

    summary_text = (output_dir / "SUMMARY.md").read_text()
    assert "Rank-first hypothesis status" in summary_text

    manifest = json.loads((output_dir / "output_manifest.json").read_text())
    manifest_files = "\n".join(manifest["files"])
    assert "race_regime_ledger.csv" in manifest_files
    assert "regime_summary.csv" in manifest_files
    assert "next_hypotheses.json" in manifest_files


def test_derive_hypotheses_surfaces_rank_first_edge_candidate():
    hypotheses = audit.derive_hypotheses(
        [
            {
                "dimension": "market_favourite_odds_group",
                "dimension_value": "market_favourite_odds_4_8",
                "pre_race_usable": True,
                "race_count": 12,
                "current_candidate_mean_top1_delta": 0.08333333333333333,
                "current_candidate_mean_logloss_delta": -0.04,
                "stage2_uncalibrated_mean_top1_delta": 0.0,
                "stage2_uncalibrated_mean_logloss_delta": -0.01,
                "challenger_mean_top1_delta": 0.0,
                "challenger_mean_logloss_delta": 0.02,
            }
        ],
        min_regime_races=5,
    )

    assert hypotheses["rank_first_readiness"]["status"] == (
        "PRE_RACE_RANK_FIRST_EDGE_CANDIDATE_FOUND"
    )
    assert hypotheses["rank_first_readiness"][
        "pre_race_rank_first_help_regime_count"
    ] == 1
    assert hypotheses["rank_first_readiness"]["blockers"] == []
    assert hypotheses["pre_race_rank_first_help_regimes"][0]["dimension"] == (
        "market_favourite_odds_group"
    )
    assert hypotheses["recommended_next_tests"][0]["status"] == (
        "REQUIRES_NEW_REPORT_ONLY_PACKET"
    )


def test_market_residual_regime_audit_collects_when_predictions_missing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    matrix_path = _write_csv(tmp_path / "market_residual_runner_matrix.csv", _rows(2))
    predictions_path = _write_csv(
        tmp_path / "cross_validated_race_predictions.csv",
        _prediction_rows(1),
    )
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525"
        / "market_residual_regime_audit_collecting"
    )

    report = audit.build_audit(
        runner_matrix_csv=matrix_path,
        race_predictions_csv=predictions_path,
        output_dir=output_dir,
        min_races_for_review=3,
        min_regime_races=1,
        generated_at=datetime(2026, 6, 12, 1, 0, tzinfo=timezone.utc),
    )

    assert report["final_status"] == "MARKET_RESIDUAL_REGIME_AUDIT_COLLECTING"
    assert report["accepted_race_count"] == 1
    assert report["collection"]["skipped_counts"] == {
        "race_missing_cv_prediction_row": 1
    }
    assert "accepted_race_count_below_review_floor" in report["blockers"]
    assert report["promotion_ready"] is False

    with (output_dir / "race_regime_ledger.csv").open(encoding="utf-8", newline="") as handle:
        ledger = list(csv.DictReader(handle))
    assert len(ledger) == 1
