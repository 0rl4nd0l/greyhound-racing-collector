import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import build_time_split_gated_challenger_packet as time_split


def _runner_row(
    *,
    race_index: int,
    race_date: str,
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
        "race_id": f"Race {race_index} - TEST - {race_date}",
        "source_report": "artifacts/source/report.json",
        "venue": "TEST",
        "race_number": race_index,
        "race_date": race_date,
        "dog_name": f"Dog {race_index}-{box}",
        "box_number": box,
        "is_winner": box == winner_box,
        "finish_position": 1 if box == winner_box else 2,
        "odds_decimal": odds_decimal,
        "odds_source_url": "https://www.sportsbet.com.au/greyhound-racing/example",
        "odds_capture_timestamp": f"{race_date}T12:30:00+10:00",
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


def _rows(date_count: int = 3, races_per_date: int = 2) -> list[dict[str, object]]:
    dates = ["2026-06-10", "2026-06-11", "2026-06-12"][:date_count]
    rows: list[dict[str, object]] = []
    race_index = 1
    for race_date in dates:
        for local_index in range(races_per_date):
            runner_count = 5 if local_index == 0 else 6
            winner_box = 2 if runner_count == 5 else 1
            rows.extend(
                [
                    _runner_row(
                        race_index=race_index,
                        race_date=race_date,
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
                        race_date=race_date,
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
                        race_date=race_date,
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
            race_index += 1
    return rows


def _write_matrix(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "market_residual_runner_matrix.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_time_split_gated_challenger_writes_chronological_packet(tmp_path, monkeypatch):
    monkeypatch.setattr(time_split, "ROOT", tmp_path)
    matrix_path = _write_matrix(tmp_path, _rows())
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525"
        / "time_split_gated_challenger_test"
    )

    report = time_split.build_packet(
        runner_matrix_csv=matrix_path,
        output_dir=output_dir,
        min_train_races=2,
        min_test_races=2,
        min_train_gate_triggers=1,
        min_races_for_review=4,
        generated_at=datetime(2026, 6, 12, 1, 0, tzinfo=timezone.utc),
    )

    assert report["schema_version"] == "time_split_gated_challenger_report_v1"
    assert report["final_status"] == "TIME_SPLIT_GATED_CHALLENGER_REVIEW_READY"
    assert report["accepted_race_count"] == 6
    assert report["race_dates"] == ["2026-06-10", "2026-06-11", "2026-06-12"]
    assert report["evaluated_split_count"] == 2
    assert report["time_split_metrics"]["status"] == "EVALUATED"
    assert report["time_split_metrics"]["race_count"] == 4
    assert report["promotion_gate"]["promotion_ready"] is False
    assert "report_only_time_split_gated_challenger_not_promotion_eligible" in report[
        "promotion_gate"
    ]["blockers"]
    assert report["no_write_guarantees"]["production_promotion"] is False
    assert report["no_write_guarantees"]["db_write"] is False
    assert report["no_write_guarantees"]["betting_or_ev_action"] is False

    assert (output_dir / "time_split_gated_challenger_report.json").exists()
    assert (output_dir / "time_split_summary.csv").exists()
    assert (output_dir / "time_split_race_predictions.csv").exists()
    assert (output_dir / "SUMMARY.md").exists()
    assert (output_dir / "output_manifest.json").exists()

    with (output_dir / "time_split_summary.csv").open(encoding="utf-8", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    assert len(split_rows) == 2
    assert {row["status"] for row in split_rows} == {"EVALUATED"}
    assert split_rows[0]["train_dates"] == "2026-06-10"
    assert split_rows[0]["test_date"] == "2026-06-11"

    with (output_dir / "time_split_race_predictions.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        prediction_rows = list(csv.DictReader(handle))
    assert len(prediction_rows) == 4
    assert "challenger_minus_market_logloss" in prediction_rows[0]
    assert "test_date" in prediction_rows[0]

    manifest = json.loads((output_dir / "output_manifest.json").read_text())
    manifest_files = "\n".join(manifest["files"])
    assert "time_split_summary.csv" in manifest_files
    assert "time_split_race_predictions.csv" in manifest_files


def test_time_split_gated_challenger_collects_when_train_window_underpowered(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(time_split, "ROOT", tmp_path)
    matrix_path = _write_matrix(tmp_path, _rows(date_count=2, races_per_date=1))
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525"
        / "time_split_gated_challenger_collecting"
    )

    report = time_split.build_packet(
        runner_matrix_csv=matrix_path,
        output_dir=output_dir,
        min_train_races=10,
        min_test_races=1,
        min_train_gate_triggers=1,
        min_races_for_review=1,
        generated_at=datetime(2026, 6, 12, 1, 0, tzinfo=timezone.utc),
    )

    assert report["final_status"] == "TIME_SPLIT_GATED_CHALLENGER_COLLECTING"
    assert report["time_split_metrics"]["status"] == "NO_EVALUABLE_RACES"
    assert report["evaluated_split_count"] == 0
    assert "time_split_challenger_not_evaluated" in report["blockers"]
    assert report["promotion_gate"]["promotion_ready"] is False

    with (output_dir / "time_split_summary.csv").open(encoding="utf-8", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    assert len(split_rows) == 1
    assert split_rows[0]["status"] == "SKIPPED_TRAIN_RACE_COUNT_BELOW_MINIMUM"

    with (output_dir / "time_split_race_predictions.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        prediction_rows = list(csv.DictReader(handle))
    assert prediction_rows == []
