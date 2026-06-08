import pytest

from accuracy_program.calibration import (
    power_normalize_by_race,
    power_normalize_prediction_group,
)


def test_power_normalize_by_race_is_additive_rank_preserving_and_label_free():
    rows = [
        {
            "race_id": "Race 1",
            "dog_name": "Alpha",
            "box_number": 1,
            "win_prob_norm": 0.64,
            "actual_win": 0,
            "odds": 12.0,
        },
        {
            "race_id": "Race 1",
            "dog_name": "Bravo",
            "box_number": 2,
            "win_prob_norm": 0.25,
            "actual_win": 1,
            "odds": 2.0,
        },
        {
            "race_id": "Race 1",
            "dog_name": "Charlie",
            "box_number": 3,
            "win_prob_norm": 0.11,
            "actual_win": 0,
            "odds": 8.0,
        },
        {
            "race_id": "Race 2",
            "dog_name": "Delta",
            "box_number": 1,
            "win_prob_norm": 0.4,
        },
        {
            "race_id": "Race 2",
            "dog_name": "Echo",
            "box_number": 2,
            "win_prob_norm": 0.35,
        },
        {
            "race_id": "Race 2",
            "dog_name": "Foxtrot",
            "box_number": 3,
            "win_prob_norm": 0.25,
        },
    ]

    calibrated = power_normalize_by_race(
        rows,
        alpha=0.5,
        output_key="calibrated_win_prob",
    )

    assert rows[0].get("calibrated_win_prob") is None
    assert calibrated != rows
    for race_id in {"Race 1", "Race 2"}:
        race_rows = [row for row in calibrated if row["race_id"] == race_id]
        assert sum(row["calibrated_win_prob"] for row in race_rows) == pytest.approx(
            1.0
        )
        original_order = [
            row["dog_name"]
            for row in sorted(
                [row for row in rows if row["race_id"] == race_id],
                key=lambda row: row["win_prob_norm"],
                reverse=True,
            )
        ]
        calibrated_order = [
            row["dog_name"]
            for row in sorted(
                race_rows,
                key=lambda row: row["calibrated_win_prob"],
                reverse=True,
            )
        ]
        assert calibrated_order == original_order


@pytest.mark.parametrize("alpha", [0, -1, float("nan")])
def test_power_normalize_by_race_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha_must_be_positive_finite"):
        power_normalize_by_race(
            [{"race_id": "Race 1", "win_prob_norm": 1.0}],
            alpha=alpha,
            output_key="calibrated_win_prob",
        )


def test_power_normalize_by_race_rejects_missing_race_groups():
    with pytest.raises(ValueError, match="race_id_missing"):
        power_normalize_by_race(
            [{"win_prob_norm": 1.0}],
            alpha=0.5,
            output_key="calibrated_win_prob",
        )


def test_power_normalize_by_race_rejects_empty_prediction_groups():
    with pytest.raises(ValueError, match="rows_missing"):
        power_normalize_by_race(
            [],
            alpha=0.5,
            output_key="calibrated_win_prob",
        )


def test_power_normalize_prediction_group_adds_report_only_fields():
    predictions = [
        {
            "dog_name": "Alpha",
            "box_number": 1,
            "predicted_rank": 1,
            "win_prob_norm": 0.64,
            "actual_win": 0,
            "odds": 12.0,
        },
        {
            "dog_name": "Bravo",
            "box_number": 2,
            "predicted_rank": 2,
            "win_prob_norm": 0.25,
            "actual_win": 1,
            "odds": 2.0,
        },
        {
            "dog_name": "Charlie",
            "box_number": 3,
            "predicted_rank": 3,
            "win_prob_norm": 0.11,
            "actual_win": 0,
            "odds": 8.0,
        },
    ]
    without_labels_or_odds = [
        {
            "dog_name": row["dog_name"],
            "box_number": row["box_number"],
            "predicted_rank": row["predicted_rank"],
            "win_prob_norm": row["win_prob_norm"],
        }
        for row in predictions
    ]

    calibrated = power_normalize_prediction_group(predictions, alpha=0.5)
    calibrated_without_labels_or_odds = power_normalize_prediction_group(
        without_labels_or_odds,
        alpha=0.5,
    )

    assert predictions[0].get("calibrated_win_prob_report_only") is None
    assert [row["predicted_rank"] for row in calibrated] == [1, 2, 3]
    assert [row["win_prob_norm"] for row in calibrated] == [0.64, 0.25, 0.11]
    assert sum(row["calibrated_win_prob_report_only"] for row in calibrated) == (
        pytest.approx(1.0)
    )
    assert [row["calibrated_predicted_rank_report_only"] for row in calibrated] == [
        1,
        2,
        3,
    ]
    assert [
        row["calibrated_win_prob_report_only"] for row in calibrated
    ] == pytest.approx(
        [
            row["calibrated_win_prob_report_only"]
            for row in calibrated_without_labels_or_odds
        ]
    )
