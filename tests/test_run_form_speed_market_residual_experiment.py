import importlib.util
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).parents[1] / "scripts/run_form_speed_market_residual_experiment.py"
SPEC = importlib.util.spec_from_file_location("form_speed_residual", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_race_probabilities_preserve_market_at_zero_offset():
    rows = [
        {"race_id": "r1", "market_implied_probability": 0.7},
        {"race_id": "r1", "market_implied_probability": 0.3},
        {"race_id": "r2", "market_implied_probability": 0.2},
        {"race_id": "r2", "market_implied_probability": 0.8},
    ]
    got = MODULE.race_probabilities(rows, np.zeros(4))
    assert np.allclose(got, [0.7, 0.3, 0.2, 0.8])


def test_design_preserves_missingness_with_indicator():
    template = {feature: 1.0 for feature in MODULE.BASE_FEATURES}
    a = dict(template); b = dict(template); b[MODULE.BASE_FEATURES[0]] = None
    train, test = MODULE.design([a, b], [b])
    assert train.shape == (2, 2 * len(MODULE.BASE_FEATURES))
    assert test[0, 1] == 1.0


def test_fold_tests_are_chronological_disjoint_and_end_before_forward_boundary():
    assert all(f["train_end"] < f["test_start"] for f in MODULE.FOLDS)
    dates = [(f["test_start"], f["test_end"]) for f in MODULE.FOLDS]
    assert dates[0][1] < dates[1][0] and dates[1][1] < dates[2][0]
    assert dates[-1][1] < "2026-08-18"


def canonical_race():
    common = {
        "race_id": "r1",
        "race_date": "2026-07-01",
        "field_size": 2,
        "market_implied_probability": 0.5,
        "odds_capture_timestamp": "2026-07-01T11:55:00+10:00",
        "jump_at": "2026-07-01T12:00:00+10:00",
    }
    return [
        {**common, "box_number": 1, "label_is_winner": 1},
        {**common, "box_number": 2, "label_is_winner": 0},
    ]


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda rows: rows[0].update(label_is_winner=0), "winner_count"),
        (lambda rows: rows[1].update(box_number=1), "duplicate_or_invalid_box"),
        (lambda rows: rows[0].update(field_size=3), "incomplete_field"),
        (lambda rows: rows[0].update(market_implied_probability=0.4), "market_probabilities_not_normalized"),
        (lambda rows: rows[0].update(odds_capture_timestamp=rows[0]["jump_at"]), "capture_not_before_jump"),
        (lambda rows: rows[0].update(race_date="2026-08-18"), "forward_target_date"),
    ],
)
def test_canonical_population_validation_fails_closed(mutate, reason):
    rows = canonical_race()
    mutate(rows)
    with pytest.raises(SystemExit, match=reason):
        MODULE.validate_canonical_population({"r1": rows})


def test_canonical_population_validation_accepts_complete_normalized_race():
    MODULE.validate_canonical_population({"r1": canonical_race()})


def test_betfair_overlap_retains_only_complete_races():
    predictions = [
        {"race_id": "complete", "box_number": 1},
        {"race_id": "complete", "box_number": 2},
        {"race_id": "partial", "box_number": 1},
        {"race_id": "partial", "box_number": 2},
        {"race_id": "absent", "box_number": 1},
        {"race_id": "absent", "box_number": 2},
    ]
    betfair = [
        {"race_id": "complete", "box_number": 1, "betfair_scheduled_off_back_price": 2.0},
        {"race_id": "complete", "box_number": 2, "betfair_scheduled_off_back_price": 3.0},
        {"race_id": "partial", "box_number": 1, "betfair_scheduled_off_back_price": 4.0},
    ]
    retained, _, partial_races = MODULE.complete_betfair_overlap(predictions, betfair)
    assert retained == [0, 1]
    assert partial_races == 1


def test_betfair_overlap_rejects_extra_runner_or_invalid_price():
    predictions = [
        {"race_id": "extra", "box_number": 1},
        {"race_id": "extra", "box_number": 2},
        {"race_id": "invalid", "box_number": 1},
        {"race_id": "invalid", "box_number": 2},
    ]
    betfair = [
        {"race_id": "extra", "box_number": 1, "betfair_scheduled_off_back_price": 2.0},
        {"race_id": "extra", "box_number": 2, "betfair_scheduled_off_back_price": 3.0},
        {"race_id": "extra", "box_number": 3, "betfair_scheduled_off_back_price": 4.0},
        {"race_id": "invalid", "box_number": 1, "betfair_scheduled_off_back_price": 2.0},
        {"race_id": "invalid", "box_number": 2, "betfair_scheduled_off_back_price": None},
    ]
    retained, _, partial_races = MODULE.complete_betfair_overlap(predictions, betfair)
    assert retained == []
    assert partial_races == 2
