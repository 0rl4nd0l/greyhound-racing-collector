import json
from pathlib import Path

from scripts.review_snapshot_challenger import build_review


def _row(
    *,
    race_id: str,
    race_date: str,
    box_number: int,
    actual_win: int,
    probability: float,
) -> dict:
    return {
        "race_id": race_id,
        "race_date": race_date,
        "dog_name": f"Dog {race_id} {box_number}",
        "box_number": box_number,
        "win_prob_norm": probability,
        "actual_win": actual_win,
        "finish_position": 1 if actual_win else box_number,
        "label_quality": "official_or_complete_result",
        "result_detail_quality": "finish_position",
    }


def _write_dataset(path: Path, *, dates=("2026-01-01", "2026-01-02")) -> None:
    rows = []
    for date_index, race_date in enumerate(dates):
        for race_number in range(1, 13):
            race_id = f"Race {race_number} - TEST - {race_date}"
            winner_box = 1 if (race_number + date_index) % 2 == 0 else 2
            raw = {1: 0.45, 2: 0.3, 3: 0.15, 4: 0.1}
            if winner_box == 2:
                raw = {1: 0.25, 2: 0.4, 3: 0.2, 4: 0.15}
            for box_number in range(1, 5):
                rows.append(
                    _row(
                        race_id=race_id,
                        race_date=race_date,
                        box_number=box_number,
                        actual_win=int(box_number == winner_box),
                        probability=raw[box_number],
                    )
                )
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_snapshot_challenger_review_is_report_only_and_temporal(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset)

    report = build_review(dataset_path=dataset)

    assert report["status"] == "SUCCESS"
    assert report["failures"] == []
    assert report["feature_safety"]["forbidden_feature_columns"] == []
    assert report["temporal_holdout"]["check"]["ok"] is True
    assert report["temporal_holdout"]["train_races"] == 12
    assert report["temporal_holdout"]["test_races"] == 12
    assert report["arms"]["baseline_model"]["races_evaluated"] == 12
    assert report["arms"]["power_calibrated_baseline"]["races_evaluated"] == 12
    assert report["arms"]["logistic_numeric_challenger"]["races_evaluated"] == 12
    assert (
        report["challenger_training"]["power_calibration"]["model_artifact_written"]
        is False
    )
    assert (
        report["challenger_training"]["power_calibration"]["registry_mutation_allowed"]
        is False
    )
    assert "power_calibrated_baseline" in report["comparison_to_baseline"]["by_arm"]
    assert report["stability_review"]["candidate_arm"] == "power_calibrated_baseline"
    assert report["stability_review"]["promotion_allowed"] is False
    assert len(report["stability_review"]["splits"]) == 1
    assert report["stability_review"]["splits"][0]["ranking_preserved"] is True
    assert report["challenger_training"]["model_artifact_written"] is False
    assert report["promotion_control"]["registry_mutation_allowed"] is False
    assert report["promotion_control"]["promotion_allowed"] is False


def test_snapshot_challenger_review_fails_closed_without_holdout_date(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset, dates=("2026-01-01",))

    report = build_review(dataset_path=dataset)

    assert report["status"] == "NOT_READY"
    assert "temporal_holdout:missing_temporal_dates" in report["failures"]
    assert report["arms"] == {}
    assert report["stability_review"]["status"] == "NOT_STABLE"
    assert report["promotion_control"]["promotion_allowed"] is False


def test_snapshot_challenger_review_reports_expanding_stability(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset, dates=("2026-01-01", "2026-01-02", "2026-01-03"))

    report = build_review(dataset_path=dataset)

    stability = report["stability_review"]
    assert stability["split_count"] == 2
    assert stability["failed_split_count"] == 0
    assert len(stability["splits"]) == 2
    assert stability["all_ranking_preserved"] is True
    assert stability["promotion_allowed"] is False
    assert {split["holdout_date"] for split in stability["splits"]} == {
        "2026-01-02",
        "2026-01-03",
    }
    assert all(
        split["power_calibration"]["model_artifact_written"] is False
        for split in stability["splits"]
    )
