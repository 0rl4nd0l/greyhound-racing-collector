import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import build_market_residual_challenger_packet as challenger


def _runner_row(
    *,
    race_index: int,
    box: int,
    winner_box: int,
    market_probability: float,
    candidate_probability: float,
    stage2_probability: float,
    odds_decimal: float,
) -> dict[str, object]:
    return {
        "race_id": f"Race {race_index} - TEST - 2026-06-10",
        "race_date": "2026-06-10",
        "venue": "TEST",
        "race_number": race_index,
        "dog_name": f"Dog {race_index}-{box}",
        "box_number": box,
        "is_winner": box == winner_box,
        "odds_decimal": odds_decimal,
        "market_probability": market_probability,
        "candidate_probability": candidate_probability,
        "stage2_shadow_uncalibrated_probability_norm": stage2_probability,
    }


def _write_runner_matrix(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    matrix_path = tmp_path / "market_residual_runner_matrix.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return matrix_path


def _synthetic_rows(race_count: int = 6) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for race_index in range(1, race_count + 1):
        winner_box = 2 if race_index % 2 else 1
        rows.extend(
            [
                _runner_row(
                    race_index=race_index,
                    box=1,
                    winner_box=winner_box,
                    market_probability=0.50,
                    candidate_probability=0.20,
                    stage2_probability=0.25 if winner_box != 1 else 0.70,
                    odds_decimal=2.6,
                ),
                _runner_row(
                    race_index=race_index,
                    box=2,
                    winner_box=winner_box,
                    market_probability=0.30,
                    candidate_probability=0.60,
                    stage2_probability=0.65 if winner_box != 1 else 0.20,
                    odds_decimal=4.2,
                ),
                _runner_row(
                    race_index=race_index,
                    box=3,
                    winner_box=winner_box,
                    market_probability=0.20,
                    candidate_probability=0.20,
                    stage2_probability=0.10,
                    odds_decimal=8.0,
                ),
            ]
        )
    return rows


def test_market_residual_challenger_writes_report_only_packet(tmp_path, monkeypatch):
    monkeypatch.setattr(challenger, "ROOT", tmp_path)
    matrix_path = _write_runner_matrix(tmp_path, _synthetic_rows())
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525"
        / "market_residual_challenger_test"
    )

    report = challenger.build_packet(
        runner_matrix_csv=matrix_path,
        output_dir=output_dir,
        fold_count=3,
        min_train_races=2,
        min_races_for_review=3,
        generated_at=datetime(2026, 6, 12, 1, 0, tzinfo=timezone.utc),
    )

    assert report["schema_version"] == "market_residual_challenger_report_v1"
    assert report["final_status"] == "MARKET_RESIDUAL_CHALLENGER_REVIEW_READY"
    assert report["matrix_row_count"] == 18
    assert report["accepted_race_count"] == 6
    assert report["evaluated_fold_count"] == 3
    assert report["challenger_metrics"]["status"] == "EVALUATED"
    assert report["promotion_gate"]["promotion_ready"] is False
    assert "report_only_residual_challenger_not_promotion_eligible" in report[
        "promotion_gate"
    ]["blockers"]
    assert report["no_write_guarantees"]["production_promotion"] is False
    assert report["no_write_guarantees"]["db_write"] is False
    assert report["no_write_guarantees"]["betting_or_ev_action"] is False
    assert (output_dir / "market_residual_challenger_report.json").exists()
    assert (output_dir / "cross_validated_fold_summary.csv").exists()
    assert (output_dir / "cross_validated_race_predictions.csv").exists()
    assert (output_dir / "SUMMARY.md").exists()
    assert (output_dir / "output_manifest.json").exists()

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
    assert "challenger_minus_market_logloss" in prediction_rows[0]

    manifest = json.loads((output_dir / "output_manifest.json").read_text())
    manifest_files = "\n".join(manifest["files"])
    assert "cross_validated_fold_summary.csv" in manifest_files
    assert "cross_validated_race_predictions.csv" in manifest_files
    assert "market_residual_challenger_report.json" in manifest_files


def test_market_residual_challenger_collects_when_train_folds_underpowered(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(challenger, "ROOT", tmp_path)
    matrix_path = _write_runner_matrix(tmp_path, _synthetic_rows(race_count=4))
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525"
        / "market_residual_challenger_underpowered"
    )

    report = challenger.build_packet(
        runner_matrix_csv=matrix_path,
        output_dir=output_dir,
        fold_count=2,
        min_train_races=10,
        min_races_for_review=3,
        generated_at=datetime(2026, 6, 12, 1, 0, tzinfo=timezone.utc),
    )

    assert report["final_status"] == "MARKET_RESIDUAL_CHALLENGER_COLLECTING"
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
