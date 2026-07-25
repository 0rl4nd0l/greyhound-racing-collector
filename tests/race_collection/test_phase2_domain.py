import json
from datetime import datetime, timedelta, timezone

import pytest

from race_collection.domain import (
    DogId,
    DomainValidationError,
    IdentityTier,
    ProgrammeRaceCandidate,
    next_odds_attempt_at,
    odds_capture_interval,
)
from race_collection.identity import DogIdentityDecision, resolve_dog_identity
from race_collection.inventory import JsonProgrammeAdapter

NOW = datetime(2026, 7, 22, 2, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("until_jump", "expected"),
    [
        (timedelta(hours=4), timedelta(minutes=30)),
        (timedelta(hours=2), timedelta(minutes=10)),
        (timedelta(minutes=30), timedelta(minutes=5)),
        (timedelta(minutes=5), timedelta(minutes=1)),
        (timedelta(0), None),
    ],
)
def test_adaptive_odds_cadence(until_jump, expected):
    assert odds_capture_interval(until_jump) == expected


def test_failed_odds_attempts_use_bounded_backoff_without_skipping_base_capture():
    assert next_odds_attempt_at(now=NOW, scheduled_jump=NOW + timedelta(minutes=5)) == NOW
    assert next_odds_attempt_at(
        now=NOW,
        scheduled_jump=NOW + timedelta(minutes=5),
        last_attempt_at=NOW,
        consecutive_failures=3,
    ) == NOW + timedelta(minutes=2)
    assert (
        next_odds_attempt_at(
            now=NOW,
            scheduled_jump=NOW,
            consecutive_failures=0,
        )
        is None
    )


def test_failed_odds_attempt_backoff_crossing_jump_is_not_scheduled():
    assert (
        next_odds_attempt_at(
            now=NOW,
            scheduled_jump=NOW + timedelta(minutes=1),
            last_attempt_at=NOW,
            consecutive_failures=3,
        )
        is None
    )


def test_failed_odds_attempt_backoff_reaching_jump_is_not_scheduled():
    assert (
        next_odds_attempt_at(
            now=NOW,
            scheduled_jump=NOW + timedelta(minutes=2),
            last_attempt_at=NOW,
            consecutive_failures=3,
        )
        is None
    )


def test_normal_odds_cadence_reaching_jump_is_not_scheduled():
    scheduled_jump = NOW + timedelta(minutes=5)

    assert (
        next_odds_attempt_at(
            now=NOW,
            scheduled_jump=scheduled_jump,
            last_attempt_at=scheduled_jump - timedelta(minutes=1),
        )
        is None
    )


def test_programme_adapter_assigns_stable_internal_race_ids():
    payload = json.dumps(
        {
            "races": [
                {
                    "source_race_id": "meet-7-r3",
                    "venue": "Ballarat",
                    "race_number": 3,
                    "scheduled_jump": NOW.isoformat(),
                }
            ]
        }
    ).encode()
    adapter = JsonProgrammeAdapter("official-programme")
    assert adapter.parse(payload) == adapter.parse(payload)
    assert adapter.parse(payload)[0].source_race_id == "meet-7-r3"


def test_programme_rejects_duplicate_expected_slot():
    item = {
        "source_race_id": "one",
        "venue": "Ballarat",
        "race_number": 3,
        "scheduled_jump": NOW.isoformat(),
    }
    with pytest.raises(ValueError, match="duplicate"):
        JsonProgrammeAdapter("official").parse(
            json.dumps({"races": [item, {**item, "source_race_id": "two"}]}).encode()
        )


def test_tiered_dog_resolution_never_merges_ambiguous_names():
    authoritative = resolve_dog_identity(
        source="registry",
        registration_authority="registry",
        registration_id="REG-42",
        name="Swift One",
    )
    provisional = resolve_dog_identity(source="form-guide", registration_id=None, name="Swift One")
    ambiguous = resolve_dog_identity(
        source="form-guide",
        registration_id=None,
        name="Swift One",
        name_candidates=(authoritative.dog_id, provisional.dog_id),
    )
    assert authoritative.tier is IdentityTier.AUTHORITATIVE
    assert provisional.tier is IdentityTier.HIGH_CONFIDENCE_PROVISIONAL
    assert ambiguous.tier is IdentityTier.AMBIGUOUS
    assert ambiguous.dog_id is None


def test_registration_authority_is_separate_from_observation_source():
    first = resolve_dog_identity(
        source="card-a",
        registration_authority="national-registry",
        registration_id="REG-42",
        name="Swift One",
    )
    second = resolve_dog_identity(
        source="card-b",
        registration_authority="national-registry",
        registration_id="REG-42",
        name="Swift One",
    )
    other_authority = resolve_dog_identity(
        source="card-a",
        registration_authority="state-registry",
        registration_id="REG-42",
        name="Swift One",
    )
    provisional_a = resolve_dog_identity(source="card-a", registration_id=None, name="Swift One")
    provisional_b = resolve_dog_identity(source="card-b", registration_id=None, name="Swift One")
    assert first.dog_id == second.dog_id
    assert first.dog_id != other_authority.dog_id
    assert provisional_a.dog_id != provisional_b.dog_id


@pytest.mark.parametrize("authority", [None, "", "   "])
def test_authoritative_registration_requires_explicit_nonblank_authority(authority):
    with pytest.raises(DomainValidationError, match="registration_authority is required"):
        resolve_dog_identity(
            source="card-a",
            registration_authority=authority,
            registration_id="REG-42",
            name="Swift One",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source", 3, "source identity must be text"),
        ("source_race_id", None, "source identity must be text"),
        ("venue", [], "venue must be text"),
        ("race_number", "3", "race_number must be an integer"),
        ("race_number", True, "race_number must be an integer"),
        ("scheduled_jump", "soon", "scheduled_jump must be a datetime"),
    ],
)
def test_programme_candidate_wrong_types_fail_with_domain_validation(field, value, message):
    values = {
        "source": "official",
        "source_race_id": "R-1",
        "venue": "Ballarat",
        "race_number": 1,
        "scheduled_jump": NOW,
    }
    values[field] = value
    with pytest.raises(DomainValidationError, match=message):
        ProgrammeRaceCandidate(**values)


@pytest.mark.parametrize(
    ("tier", "dog_id", "message"),
    [
        (IdentityTier.AMBIGUOUS, DogId("dog_" + "1" * 32), "AMBIGUOUS requires"),
        (IdentityTier.AUTHORITATIVE, None, "AUTHORITATIVE requires"),
        (
            IdentityTier.HIGH_CONFIDENCE_PROVISIONAL,
            None,
            "HIGH_CONFIDENCE_PROVISIONAL requires",
        ),
        ("authoritative", DogId("dog_" + "1" * 32), "tier must be an IdentityTier"),
        (IdentityTier.AUTHORITATIVE, "dog_" + "1" * 32, "dog_id must be a DogId"),
    ],
)
def test_identity_decision_rejects_invalid_tier_and_dog_id_combinations(tier, dog_id, message):
    with pytest.raises(DomainValidationError, match=message):
        DogIdentityDecision(tier=tier, dog_id=dog_id, reason="invalid")
