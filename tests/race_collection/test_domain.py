from datetime import date, datetime, timezone

import pytest

from race_collection.domain import (
    DogId,
    DogRun,
    DomainValidationError,
    IllegalLifecycleTransition,
    RaceId,
    RaceLifecycle,
    RaceState,
    RacingDay,
    RacingDayId,
)


def ident(prefix: str, digit: str = "1") -> str:
    return f"{prefix}_{digit * 32}"


def test_every_declared_legal_lifecycle_transition_is_accepted():
    legal = {
        RaceState.DISCOVERED: {RaceState.CARD_COLLECTED},
        RaceState.CARD_COLLECTED: {RaceState.COLLECTING_ODDS},
        RaceState.COLLECTING_ODDS: {RaceState.EVIDENCE_SEALED},
        RaceState.EVIDENCE_SEALED: {RaceState.AWAITING_DAY_CLOSE},
        RaceState.AWAITING_DAY_CLOSE: {RaceState.PREDICTION_PENDING},
        RaceState.PREDICTION_PENDING: {
            RaceState.PREDICTION_COMMITTED,
            RaceState.PREDICTION_QUARANTINED,
        },
        RaceState.PREDICTION_COMMITTED: {RaceState.RESULT_PENDING},
        RaceState.RESULT_PENDING: {RaceState.RESULT_COLLECTED, RaceState.RESULT_QUARANTINED},
        RaceState.RESULT_COLLECTED: {
            RaceState.TRAINING_EXAMPLE_READY,
            RaceState.EVALUATION_INELIGIBLE,
        },
    }
    for current, targets in legal.items():
        for target in targets:
            RaceLifecycle.validate(current, target)


def test_every_other_lifecycle_transition_is_rejected():
    for current in RaceState:
        for target in RaceState:
            if target not in RaceLifecycle.legal_targets(current):
                with pytest.raises(IllegalLifecycleTransition):
                    RaceLifecycle.validate(current, target)


def test_result_cannot_precede_committed_prediction():
    with pytest.raises(IllegalLifecycleTransition):
        RaceLifecycle.validate(RaceState.PREDICTION_PENDING, RaceState.RESULT_PENDING)
    assert RaceLifecycle.legal_targets(RaceState.PREDICTION_QUARANTINED) == frozenset()


def test_racing_day_keeps_official_date_timezone_and_aware_instant():
    day = RacingDay(
        RacingDayId(ident("day")),
        date(2026, 7, 22),
        "Australia/Melbourne",
        datetime(2026, 7, 21, 22, tzinfo=timezone.utc),
    )
    assert day.local_date.isoformat() == "2026-07-22"
    assert day.timezone == "Australia/Melbourne"


def test_ids_and_records_are_immutable_and_validated():
    run = DogRun(
        DogId(ident("dog")),
        date(2026, 7, 22),
        False,
        datetime.now(timezone.utc),
    )
    with pytest.raises((AttributeError, TypeError)):
        run.authoritative = True
    with pytest.raises(DomainValidationError):
        RaceId("legacy-filename-race-key")
    with pytest.raises(DomainValidationError):
        RacingDay(
            RacingDayId(ident("day")),
            date(2026, 7, 22),
            "Australia/Melbourne",
            datetime(2026, 7, 22),
        )
