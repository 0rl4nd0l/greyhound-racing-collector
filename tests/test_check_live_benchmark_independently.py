"""Failure cases for the independently implemented audit arithmetic."""

import copy
import math

import pytest

from scripts.check_live_benchmark_independently import aggregate, compute_race


def field():
    return [
        {
            "runner_id": "dog-a",
            "box": 1,
            "model_probability": 0.75,
            "decimal_odds": 2.0,
            "winner": 1,
        },
        {
            "runner_id": "dog-b",
            "box": 2,
            "model_probability": 0.25,
            "decimal_odds": 2.0,
            "winner": 0,
        },
    ]


def test_exact_two_runner_score_and_tied_market_top():
    result = compute_race(field())
    assert result["model"]["log_loss"] == pytest.approx(-math.log(0.75))
    assert result["market"]["log_loss"] == pytest.approx(math.log(2))
    assert result["model"]["brier"] == 0.125
    assert result["market"]["brier"] == 0.5
    assert result["market"]["top_accuracy"] == 0.5
    assert result["difference"]["brier"] == -0.375


def test_zero_winner_probability_is_infinite_loss_without_floor():
    rows = field()
    rows[0]["model_probability"], rows[1]["model_probability"] = 0.0, 1.0
    assert math.isinf(compute_race(rows)["model"]["log_loss"])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows[1].update(runner_id="dog-a"),
        lambda rows: rows[1].update(box=1),
        lambda rows: rows[1].update(model_probability=0.3),
        lambda rows: rows[0].update(model_probability=float("nan")),
        lambda rows: rows[0].update(decimal_odds=1),
        lambda rows: rows[0].update(decimal_odds=float("inf")),
        lambda rows: rows[1].update(winner=1),
        lambda rows: rows[0].update(winner=0),
        lambda rows: rows[0].update(winner=0.5),
    ],
)
def test_corrupt_fields_fail_closed(mutation):
    rows = field()
    mutation(rows)
    with pytest.raises(ValueError):
        compute_race(rows)


def test_equal_race_weighting_dates_and_calibration():
    first = {"race_id": "r1", "date": "2026-10-01", "model_version": "v1", "runners": field()}
    second = copy.deepcopy(first)
    second.update(race_id="r2", date="2026-10-02")
    second["runners"][0]["winner"], second["runners"][1]["winner"] = 0, 1
    result = aggregate([first, second])
    assert result["races"] == 2
    assert result["model"]["top_accuracy"] == 0.5
    assert result["market"]["top_accuracy"] == 0.5
    assert result["model"]["log_loss"] == pytest.approx(-math.log(0.75 * 0.25) / 2)
    assert result["calibration"]["model"]["runner_weighted_ece"] == 0.25
    assert result["calibration"]["market"]["runner_weighted_ece"] == 0
    assert result["leave_one_date_out"]["2026-10-02"]["model"]["top_accuracy"] == 1
    with pytest.raises(ValueError, match="repeated primary"):
        aggregate([first, first])


def test_empty_population_has_null_scores():
    result = aggregate([])
    assert result["races"] == result["runners"] == 0
    assert result["model"]["log_loss"] is None
    assert result["calibration"]["market"]["runner_weighted_ece"] is None


def admitted_record():
    return {
        "race_id": "r1",
        "date": "2026-10-01",
        "model_version": "v1",
        "prediction_id": "p1",
        "allocation_status": "AUTHORISED_NONRESERVED",
        "forecast_type": "ORIGINAL_SEALED_LIVE",
        "field_status": "EXACT_UNCHANGED",
        "verified": True,
        "quote_at": "2026-10-01T00:00:00Z",
        "cutoff_at": "2026-10-01T00:01:00Z",
        "predicted_at": "2026-10-01T00:02:00Z",
        "sealed_at": "2026-10-01T00:02:01Z",
        "jump_at": "2026-10-01T00:05:00Z",
        "result_at": "2026-10-01T00:10:00Z",
        "runners": [{**r, "probability": r["model_probability"]} for r in field()],
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("allocation_status", "RESERVED"),
        ("forecast_type", "OFFLINE_REPLAY"),
        ("field_status", "SCRATCHED"),
        ("verified", False),
        ("race_id", ""),
        ("quote_at", "2026-10-01T00:03:00Z"),
        ("sealed_at", "2026-10-01T00:05:00Z"),
        ("result_at", "2026-10-01T00:04:00Z"),
        ("quote_at", "2026-10-01T00:00:00"),
    ],
)
def test_admission_and_chronology_fail_closed(key, value):
    from scripts.check_live_benchmark_independently import validate_record

    record = admitted_record()
    record[key] = value
    with pytest.raises(ValueError):
        validate_record(record)


def test_empty_scorecard_accounting_is_checked():
    from scripts.check_live_benchmark_independently import compare_scorecard

    claimed = {
        "race_model_records": 0,
        "unique_races": 0,
        "model_versions": [],
        "status": "NO_ELIGIBLE_DECISION_TIME_COMPARISONS",
        "by_version": {},
        "per_race": [],
    }
    assert compare_scorecard([], claimed)["status"] == "PASS"
    claimed["unique_races"] = 1
    with pytest.raises(ValueError, match="accounting mismatch"):
        compare_scorecard([], claimed)


def test_partial_result_field_requires_explicit_diagnostic_mode():
    from scripts.check_live_benchmark_independently import validate_record

    record = admitted_record()
    record["field_status"] = "RESULT_FIELD_PARTIAL"
    with pytest.raises(ValueError, match="field_status"):
        validate_record(record)
    assert validate_record(record, diagnostic=True)["field_status"] == "RESULT_FIELD_PARTIAL"
    record["field_status"] = "CHANGED_FIELD"
    with pytest.raises(ValueError, match="field_status"):
        validate_record(record, diagnostic=True)
