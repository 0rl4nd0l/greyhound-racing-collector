"""Immutable domain values and forward-only lifecycle rules.

This module deliberately has no dependency on scrapers, Flask, model code, or
SQLite. Persisted enum values are part of the operations schema contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from math import isfinite
from typing import Any, ClassVar, Generic, TypeVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ADAPTIVE_ODDS_TIMING_POLICY = "adaptive-odds-timing-v1"
ADAPTIVE_ODDS_MAX_LATE = timedelta(seconds=5)


class DomainValidationError(ValueError):
    """A value violates a collection-domain invariant."""


def _require_exact_json(value: Any) -> None:
    """Reject values that JSON encoding would coerce instead of round-tripping."""

    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if isfinite(value):
            return
        raise DomainValidationError("field evidence value must be exact JSON")
    if type(value) is list:
        for item in value:
            _require_exact_json(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise DomainValidationError("field evidence value must be exact JSON")
            _require_exact_json(item)
        return
    raise DomainValidationError("field evidence value must be exact JSON")


class IllegalLifecycleTransition(DomainValidationError):
    """A race lifecycle transition is not forward-only or barrier-safe."""


@dataclass(frozen=True, slots=True)
class _Id:
    value: str
    prefix: ClassVar[str]

    def __post_init__(self) -> None:
        if not re.fullmatch(rf"{self.prefix}_[0-9a-f]{{32}}", self.value):
            raise DomainValidationError(f"invalid {type(self).__name__}: {self.value!r}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class RacingDayId(_Id):
    prefix: ClassVar[str] = "day"


@dataclass(frozen=True, slots=True)
class RaceId(_Id):
    prefix: ClassVar[str] = "race"


@dataclass(frozen=True, slots=True)
class DogId(_Id):
    prefix: ClassVar[str] = "dog"


class IdentityTier(str, Enum):
    AUTHORITATIVE = "authoritative"
    HIGH_CONFIDENCE_PROVISIONAL = "high_confidence_provisional"
    AMBIGUOUS = "ambiguous"


class EvidenceAuthority(str, Enum):
    OFFICIAL_PROGRAMME = "official_programme"
    OFFICIAL_CARD = "official_card"
    OFFICIAL_JUMP = "official_jump"
    SOURCE_CARD = "source_card"
    EMBEDDED_FORM = "embedded_form"
    MARKET = "market"


class EvidenceField(str, Enum):
    """Closed evidence-field registry with intrinsic conflict criticality."""

    def __new__(cls, value: str, critical: bool) -> EvidenceField:
        member = str.__new__(cls, value)
        member._value_ = value
        member._critical = critical
        return member

    IDENTITY = ("identity", True)
    RACE_IDENTITY = ("race_identity", True)
    RACE_NUMBER = ("race_number", True)
    RUNNER_IDENTITY = ("runner_identity", True)
    RUNNER_SET = ("runner_set", True)
    RUNNER_FEATURES = ("runner_features", True)
    BOX = ("box", True)
    VENUE = ("venue", True)
    DISTANCE = ("distance", False)
    GRADE = ("grade", False)
    FIELD_SIZE = ("field_size", True)
    SCHEDULED_JUMP = ("scheduled_jump", True)
    ACTUAL_JUMP = ("actual_jump", True)
    JUMP_TIME = ("jump_time", True)
    RESULT_ORDER = ("result_order", True)

    @property
    def critical(self) -> bool:
        return self._critical


class OddsAttemptStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class FreezeAuthority(str, Enum):
    ACTUAL_JUMP = "actual_jump"
    SCHEDULED_MINUS_BUFFER = "scheduled_minus_buffer"


@dataclass(frozen=True, slots=True)
class OperationId(_Id):
    prefix: ClassVar[str] = "op"


@dataclass(frozen=True, slots=True)
class ArtifactChecksum:
    value: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.value):
            raise DomainValidationError(f"invalid SHA-256 checksum: {self.value!r}")

    @property
    def hex_digest(self) -> str:
        return self.value.removeprefix("sha256:")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class FieldEvidence:
    """One append-only, artifact-bound observation of a race field."""

    operation_id: OperationId
    race_id: RaceId
    field: EvidenceField
    authority: EvidenceAuthority
    value: Any
    source: str
    artifact_checksum: ArtifactChecksum
    observed_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, OperationId):
            raise DomainValidationError("operation_id must be an OperationId")
        if not isinstance(self.race_id, RaceId):
            raise DomainValidationError("race_id must be a RaceId")
        if not isinstance(self.field, EvidenceField):
            raise DomainValidationError("field must be an EvidenceField")
        if not isinstance(self.authority, EvidenceAuthority):
            raise DomainValidationError("authority must be an EvidenceAuthority")
        if not isinstance(self.source, str) or not self.source.strip():
            raise DomainValidationError("field evidence source must be nonblank text")
        if not isinstance(self.artifact_checksum, ArtifactChecksum):
            raise DomainValidationError("artifact_checksum must be an ArtifactChecksum")
        if not isinstance(self.observed_at, datetime):
            raise DomainValidationError("observed_at must be a datetime")
        require_aware(self.observed_at, "observed_at")
        temporal_fields = {
            EvidenceField.SCHEDULED_JUMP,
            EvidenceField.ACTUAL_JUMP,
            EvidenceField.JUMP_TIME,
        }
        if self.field in temporal_fields:
            if not isinstance(self.value, datetime):
                raise DomainValidationError(
                    f"{self.field.value} field evidence value must be a datetime"
                )
            require_aware(self.value, f"{self.field.value} field evidence value")
        elif isinstance(self.value, datetime):
            raise DomainValidationError(
                f"{self.field.value} field evidence value must not be a datetime"
            )
        else:
            try:
                _require_exact_json(self.value)
            except RecursionError as error:
                raise DomainValidationError("field evidence value must be exact JSON") from error


def require_aware(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise DomainValidationError(f"{field} must be timezone-aware")


class RaceState(str, Enum):
    DISCOVERED = "discovered"
    CARD_COLLECTED = "card_collected"
    COLLECTING_ODDS = "collecting_odds"
    EVIDENCE_SEALED = "evidence_sealed"
    AWAITING_DAY_CLOSE = "awaiting_day_close"
    PREDICTION_PENDING = "prediction_pending"
    PREDICTION_COMMITTED = "prediction_committed"
    PREDICTION_QUARANTINED = "prediction_quarantined"
    RESULT_PENDING = "result_pending"
    RESULT_COLLECTED = "result_collected"
    RESULT_QUARANTINED = "result_quarantined"
    TRAINING_EXAMPLE_READY = "training_example_ready"
    EVALUATION_INELIGIBLE = "evaluation_ineligible"


class RaceLifecycle:
    """The complete legal transition graph for one race."""

    _LEGAL: ClassVar[dict[RaceState, frozenset[RaceState]]] = {
        RaceState.DISCOVERED: frozenset({RaceState.CARD_COLLECTED}),
        RaceState.CARD_COLLECTED: frozenset({RaceState.COLLECTING_ODDS}),
        RaceState.COLLECTING_ODDS: frozenset({RaceState.EVIDENCE_SEALED}),
        RaceState.EVIDENCE_SEALED: frozenset({RaceState.AWAITING_DAY_CLOSE}),
        RaceState.AWAITING_DAY_CLOSE: frozenset({RaceState.PREDICTION_PENDING}),
        RaceState.PREDICTION_PENDING: frozenset(
            {RaceState.PREDICTION_COMMITTED, RaceState.PREDICTION_QUARANTINED}
        ),
        RaceState.PREDICTION_COMMITTED: frozenset({RaceState.RESULT_PENDING}),
        RaceState.PREDICTION_QUARANTINED: frozenset(),
        RaceState.RESULT_PENDING: frozenset(
            {RaceState.RESULT_COLLECTED, RaceState.RESULT_QUARANTINED}
        ),
        RaceState.RESULT_COLLECTED: frozenset(
            {RaceState.TRAINING_EXAMPLE_READY, RaceState.EVALUATION_INELIGIBLE}
        ),
        RaceState.RESULT_QUARANTINED: frozenset(),
        RaceState.TRAINING_EXAMPLE_READY: frozenset(),
        RaceState.EVALUATION_INELIGIBLE: frozenset(),
    }

    @classmethod
    def can_transition(cls, current: RaceState, target: RaceState) -> bool:
        return target in cls._LEGAL[current]

    @classmethod
    def validate(cls, current: RaceState, target: RaceState) -> None:
        if not cls.can_transition(current, target):
            raise IllegalLifecycleTransition(f"cannot transition {current.value} -> {target.value}")

    @classmethod
    def legal_targets(cls, current: RaceState) -> frozenset[RaceState]:
        return cls._LEGAL[current]


@dataclass(frozen=True, slots=True)
class RacingDay:
    id: RacingDayId
    local_date: date
    timezone: str
    opened_at: datetime

    def __post_init__(self) -> None:
        require_aware(self.opened_at, "opened_at")
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as error:
            raise DomainValidationError(f"unknown racing-day timezone: {self.timezone}") from error


@dataclass(frozen=True, slots=True)
class DogRun:
    dog_id: DogId
    local_racing_date: date
    authoritative: bool
    created_at: datetime

    def __post_init__(self) -> None:
        require_aware(self.created_at, "created_at")


@dataclass(frozen=True, slots=True)
class RunObservation:
    operation_id: OperationId
    dog_id: DogId
    local_racing_date: date
    source: str
    artifact_checksum: ArtifactChecksum
    observed_at: datetime
    starts: int | None = None
    wins: int | None = None

    def __post_init__(self) -> None:
        require_aware(self.observed_at, "observed_at")
        if not self.source.strip():
            raise DomainValidationError("observation source must not be empty")
        if self.starts is not None and self.starts < 0:
            raise DomainValidationError("starts must be non-negative")
        if self.wins is not None and self.wins < 0:
            raise DomainValidationError("wins must be non-negative")
        if self.starts is not None and self.wins is not None and self.wins > self.starts:
            raise DomainValidationError("wins cannot exceed starts")


@dataclass(frozen=True, slots=True)
class EvidenceArtifact:
    checksum: ArtifactChecksum
    media_type: str
    byte_size: int
    created_at: datetime

    def __post_init__(self) -> None:
        require_aware(self.created_at, "created_at")
        if self.byte_size < 0:
            raise DomainValidationError("artifact byte_size must be non-negative")
        if not self.media_type.strip():
            raise DomainValidationError("artifact media_type must not be empty")


ArtifactRecord = EvidenceArtifact


@dataclass(frozen=True, slots=True)
class ExpectedRace:
    race_id: RaceId
    source: str
    source_race_id: str
    venue: str
    race_number: int
    scheduled_jump: datetime

    def __post_init__(self) -> None:
        require_aware(self.scheduled_jump, "scheduled_jump")
        if not all(value.strip() for value in (self.source, self.source_race_id, self.venue)):
            raise DomainValidationError("expected-race source identity must not be empty")
        if self.race_number < 1:
            raise DomainValidationError("race_number must be positive")


@dataclass(frozen=True, slots=True)
class ProgrammeRaceCandidate:
    source: str
    source_race_id: str
    venue: str
    race_number: int
    scheduled_jump: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not isinstance(self.source_race_id, str):
            raise DomainValidationError("programme candidate source identity must be text")
        if not isinstance(self.venue, str):
            raise DomainValidationError("programme candidate venue must be text")
        if not isinstance(self.race_number, int) or isinstance(self.race_number, bool):
            raise DomainValidationError("programme candidate race_number must be an integer")
        if not isinstance(self.scheduled_jump, datetime):
            raise DomainValidationError("programme candidate scheduled_jump must be a datetime")
        require_aware(self.scheduled_jump, "scheduled_jump")
        if not all(value.strip() for value in (self.source, self.source_race_id, self.venue)):
            raise DomainValidationError("programme candidate source identity must not be empty")
        if self.race_number < 1:
            raise DomainValidationError("race_number must be positive")


@dataclass(frozen=True, slots=True)
class OddsObservation:
    operation_id: OperationId
    race_id: RaceId
    source: str
    attempted_at: datetime
    status: OddsAttemptStatus
    artifact_checksum: ArtifactChecksum | None = None
    runner_mapping_checksum: ArtifactChecksum | None = None
    error: str | None = None
    scheduled_due_at: datetime | None = None
    timing_policy: str = ADAPTIVE_ODDS_TIMING_POLICY

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, OperationId):
            raise DomainValidationError("operation_id must be an OperationId")
        if not isinstance(self.race_id, RaceId):
            raise DomainValidationError("race_id must be a RaceId")
        if not isinstance(self.source, str):
            raise DomainValidationError("odds source must be text")
        if not isinstance(self.attempted_at, datetime):
            raise DomainValidationError("attempted_at must be a datetime")
        if not isinstance(self.status, OddsAttemptStatus):
            raise DomainValidationError("status must be an OddsAttemptStatus")
        if self.artifact_checksum is not None and not isinstance(
            self.artifact_checksum, ArtifactChecksum
        ):
            raise DomainValidationError("artifact_checksum must be an ArtifactChecksum or None")
        if self.runner_mapping_checksum is not None and not isinstance(
            self.runner_mapping_checksum, ArtifactChecksum
        ):
            raise DomainValidationError(
                "runner_mapping_checksum must be an ArtifactChecksum or None"
            )
        if self.error is not None and not isinstance(self.error, str):
            raise DomainValidationError("error must be text or None")
        if self.scheduled_due_at is None:
            object.__setattr__(self, "scheduled_due_at", self.attempted_at)
        if not isinstance(self.scheduled_due_at, datetime):
            raise DomainValidationError("scheduled_due_at must be a datetime")
        if self.timing_policy != ADAPTIVE_ODDS_TIMING_POLICY:
            raise DomainValidationError("odds timing policy is unsupported")
        require_aware(self.attempted_at, "attempted_at")
        require_aware(self.scheduled_due_at, "scheduled_due_at")
        if not self.source.strip():
            raise DomainValidationError("odds source must not be empty")
        if self.status is OddsAttemptStatus.SUCCEEDED:
            if self.artifact_checksum is None or self.runner_mapping_checksum is None:
                raise DomainValidationError("successful odds capture requires payload and mapping")
            if self.error is not None:
                raise DomainValidationError("successful odds capture cannot contain an error")
        elif (
            not self.error.strip()
            or self.artifact_checksum is not None
            or self.runner_mapping_checksum is not None
        ):
            raise DomainValidationError(
                "failed odds capture requires no payload or mapping and a nonblank error"
            )


@dataclass(frozen=True, slots=True)
class OddsAttemptRecord:
    source: str
    attempted_at: datetime
    status: OddsAttemptStatus
    artifact_checksum: ArtifactChecksum | None
    runner_mapping_checksum: ArtifactChecksum | None
    error: str | None
    scheduled_due_at: datetime
    timing_policy: str

    def __post_init__(self) -> None:
        require_aware(self.attempted_at, "attempted_at")
        require_aware(self.scheduled_due_at, "scheduled_due_at")
        if not isinstance(self.status, OddsAttemptStatus):
            raise DomainValidationError("status must be an OddsAttemptStatus")
        if self.timing_policy != ADAPTIVE_ODDS_TIMING_POLICY:
            raise DomainValidationError("odds timing policy is unsupported")


@dataclass(frozen=True, slots=True)
class CollectionRaceRecord:
    race_id: RaceId
    state: RaceState
    quarantined: bool


@dataclass(frozen=True, slots=True)
class ExpectedProgrammeArtifact:
    source: str
    checksum: ArtifactChecksum
    scheduled_jump: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise DomainValidationError("expected programme source must be nonblank text")
        if not isinstance(self.checksum, ArtifactChecksum):
            raise DomainValidationError("expected programme checksum must be an ArtifactChecksum")
        if not isinstance(self.scheduled_jump, datetime):
            raise DomainValidationError("expected programme scheduled_jump must be a datetime")
        require_aware(self.scheduled_jump, "expected programme scheduled_jump")


@dataclass(frozen=True, slots=True)
class FreezeDecision:
    cutoff: datetime
    authority: FreezeAuthority
    observation_checksum: ArtifactChecksum

    def __post_init__(self) -> None:
        require_aware(self.cutoff, "cutoff")


def odds_capture_interval(time_to_jump: timedelta) -> timedelta | None:
    """Return the initial adaptive cadence, or None once jump is due."""
    seconds = time_to_jump.total_seconds()
    if seconds <= 0:
        return None
    if seconds > 3 * 3600:
        return timedelta(minutes=30)
    if seconds > 3600:
        return timedelta(minutes=10)
    if seconds > 10 * 60:
        return timedelta(minutes=5)
    return timedelta(minutes=1)


def next_odds_attempt_at(
    *,
    now: datetime,
    scheduled_jump: datetime,
    last_attempt_at: datetime | None = None,
    consecutive_failures: int = 0,
) -> datetime | None:
    """Schedule an adaptive capture with bounded retry backoff."""
    require_aware(now, "now")
    require_aware(scheduled_jump, "scheduled_jump")
    if last_attempt_at is not None:
        require_aware(last_attempt_at, "last_attempt_at")
    if consecutive_failures < 0:
        raise DomainValidationError("consecutive_failures must be non-negative")
    cadence = odds_capture_interval(scheduled_jump - now)
    if cadence is None:
        return None
    if last_attempt_at is None:
        return now
    if consecutive_failures:
        retry = timedelta(seconds=min(30 * (2 ** min(consecutive_failures - 1, 4)), 300))
        candidate = last_attempt_at + retry
    else:
        candidate = last_attempt_at + cadence
    return candidate if candidate < scheduled_jump else None


@dataclass(frozen=True, slots=True)
class Quarantine:
    operation_id: OperationId
    race_id: RaceId
    stage: str
    reason: str
    created_at: datetime

    def __post_init__(self) -> None:
        require_aware(self.created_at, "created_at")
        if not self.stage.strip() or not self.reason.strip():
            raise DomainValidationError("quarantine stage and reason must not be empty")


@dataclass(frozen=True, slots=True)
class CollectionQuarantine:
    """The durable terminal reason that prevents further race collection."""

    stage: str
    code: str
    details: str

    def __post_init__(self) -> None:
        values = (self.stage, self.code, self.details)
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise DomainValidationError(
                "collection quarantine stage, code, and details must be nonblank text"
            )


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Supersession(Generic[T]):
    operation_id: OperationId
    prior_id: T
    replacement_id: T
    reason: str
    created_at: datetime

    def __post_init__(self) -> None:
        require_aware(self.created_at, "created_at")
        if self.prior_id == self.replacement_id:
            raise DomainValidationError("a record cannot supersede itself")
        if not self.reason.strip():
            raise DomainValidationError("supersession reason must not be empty")
