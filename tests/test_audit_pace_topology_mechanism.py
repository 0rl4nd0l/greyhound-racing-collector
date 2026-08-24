from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_pace_topology_mechanism.py"
SPEC = importlib.util.spec_from_file_location("audit_pace_topology_mechanism", SCRIPT)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def row(box: int, pace: float, probability: float) -> dict:
    return {
        "race_id": "race-1",
        "race_date": "2026-06-10",
        "box_number": box,
        "field_size": 3,
        "jump_at": "2026-06-10T10:00:00Z",
        "odds_capture_timestamp": "2026-06-10T09:59:00Z",
        "native_thedogs_dog_id": f"dog-{box}",
        "market_implied_probability": probability,
        "canonical_sportsbet_win_odds": 1.0 / probability,
        "rating_cutoff_exclusive": "2026-06-10T10:00:00Z",
        "pace_rank_fraction": 0.0,
        "pace_gap_to_best": pace,
        "pace_uncertainty": 0.2,
        "pace_effective_starts": 4.0,
        "pace_log_effective_starts": 1.609,
        "pace_missing": 0,
    }


def betfair_row(box: int, **overrides: object) -> dict:
    item = {
        "race_id": "race-1",
        "box_number": box,
        "betfair_scheduled_off_back_price": 2.0 + box,
        "betfair_scheduled_off_back_price_status": "PRESENT",
        "scheduled_clock_precedes_provider_actual_off_clock": True,
    }
    item.update(overrides)
    return item


def test_projection_is_outcome_blind() -> None:
    source = row(1, 0.0, 0.4) | {"label_is_winner": 1, "label_finish_position": 1}
    projected = audit.project_pace_rows([source])
    assert set(projected[0]) == set(audit.PACE_FIELDS)
    assert "label_is_winner" not in projected[0]
    assert "label_finish_position" not in projected[0]


def test_fixed_topology_mechanisms_use_box_neighbours() -> None:
    rows = [row(1, 0.0, 0.30), row(2, -1.0, 0.50), row(3, -0.8, 0.20)]
    thresholds = {
        "large_leader_gap_q75": 0.5,
        "comparable_adjacent_difference_q25": 0.1,
        "high_adjacent_pressure_q75": 0.5,
        "large_inside_outside_imbalance_q75": 0.5,
    }
    folds = [{"id": 1, "start": "2026-06-10", "end": "2026-06-10"}]
    result = audit.assign_topology(rows, thresholds, folds)
    by_box = {item["box_number"]: item for item in result}

    assert by_box[1]["mechanisms"]["LONE_LEADER_POSITIVE"] is True
    assert by_box[1]["mechanisms"]["CLEAR_PATH_NONFAV_POSITIVE"] is True
    assert by_box[2]["mechanisms"]["ADJACENT_PRESSURE_ADVERSE"] is True
    assert by_box[2]["mechanisms"]["PRESSURED_FAVOURITE_ADVERSE"] is True
    assert by_box[3]["mechanisms"]["ADJACENT_PRESSURE_ADVERSE"] is False


def test_incomplete_pace_field_never_defines_topology() -> None:
    rows = [row(1, 0.0, 0.50), row(2, -0.5, 0.30), row(3, -1.0, 0.20)]
    rows[2]["pace_missing"] = 1
    rows[2]["pace_gap_to_best"] = None
    thresholds = {
        "large_leader_gap_q75": 0.5,
        "comparable_adjacent_difference_q25": 0.1,
        "high_adjacent_pressure_q75": 0.5,
        "large_inside_outside_imbalance_q75": 0.5,
    }
    folds = [{"id": 1, "start": "2026-06-10", "end": "2026-06-10"}]
    result = audit.assign_topology(rows, thresholds, folds)
    assert all(item["topology_complete"] is False for item in result)
    assert all(not any(item["mechanisms"].values()) for item in result)


def test_chronological_folds_cover_each_race_once() -> None:
    rows = []
    for day in range(10, 19):
        item = row(1, 0.0, 1.0)
        item["race_date"] = f"2026-06-{day:02d}"
        item["race_id"] = f"race-{day}"
        rows.append(item)
    folds = audit.chronological_folds(rows)
    assert len(folds) == 3
    assert folds[0]["start"] == "2026-06-10"
    assert folds[-1]["end"] == "2026-06-18"
    assert sum(item["races"] for item in folds) == 9
    assert folds[0]["end"] < folds[1]["start"] < folds[2]["start"]


def test_strict_betfair_runner_eligibility_requires_exact_provenance_and_valid_price() -> None:
    assert audit.strict_betfair_runner_is_eligible(betfair_row(1)) is True

    invalid_overrides = (
        {"betfair_scheduled_off_back_price_status": "MISSING_BLANK"},
        {"betfair_scheduled_off_back_price_status": None},
        {"scheduled_clock_precedes_provider_actual_off_clock": False},
        {"scheduled_clock_precedes_provider_actual_off_clock": 1},
        {"scheduled_clock_precedes_provider_actual_off_clock": None},
        {"betfair_scheduled_off_back_price": None},
        {"betfair_scheduled_off_back_price": "3.0"},
        {"betfair_scheduled_off_back_price": 1.0},
        {"betfair_scheduled_off_back_price": math.inf},
    )
    for overrides in invalid_overrides:
        assert audit.strict_betfair_runner_is_eligible(betfair_row(1, **overrides)) is False
    assert audit.strict_betfair_runner_is_eligible(None) is False


def test_strict_betfair_probabilities_excludes_entire_race_if_one_runner_is_ineligible() -> None:
    sportsbet_rows = [row(1, 0.0, 0.6), row(2, -0.5, 0.4)]
    source = [
        betfair_row(1),
        betfair_row(2, scheduled_clock_precedes_provider_actual_off_clock=False),
    ]
    with patch.object(audit, "load_jsonl", return_value=source):
        assert audit.strict_betfair_probabilities(sportsbet_rows) == {}
