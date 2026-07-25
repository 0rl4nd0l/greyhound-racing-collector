"""Transactional Phase 2 collection commands and authoritative reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .domain import (
    ArtifactChecksum,
    FieldEvidence,
    OddsObservation,
    OperationId,
    ProgrammeRaceCandidate,
    RaceId,
    RaceState,
    RacingDay,
    RunObservation,
)
from .identity import DogIdentityDecision, normalize_dog_name
from .operations import OperationsStore


@dataclass(frozen=True, slots=True)
class Reconciliation:
    racing_day_id: str
    expected: int
    cards_collected: int
    collecting_odds: int
    sealed: int
    quarantined: int
    unresolved_race_ids: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return self.expected > 0 and not self.unresolved_race_ids


class CollectionRepository:
    """Narrow Phase 2 extension over the Phase 1 transactional store."""

    def __init__(self, store: OperationsStore):
        self.store = store

    def record_expected_race(
        self,
        operation_id: OperationId,
        day: RacingDay,
        race: ProgrammeRaceCandidate,
        programme_checksum: ArtifactChecksum,
        at: datetime,
        reconcile_to: RaceId | None = None,
    ) -> RaceId:
        return self.store.record_expected_race(
            operation_id, day, race, programme_checksum, at, reconcile_to
        )

    def record_identity_decision(
        self,
        operation_id: OperationId,
        *,
        source: str,
        source_alias: str,
        name: str,
        decision: DogIdentityDecision,
        at: datetime,
    ) -> bool:
        return self.store.record_identity_decision(
            operation_id,
            source=source,
            source_alias=source_alias,
            normalized_name=normalize_dog_name(name),
            tier=decision.tier,
            dog_id=decision.dog_id,
            reason=decision.reason,
            at=at,
        )

    def ingest_run(self, observation: RunObservation, *, authoritative: bool) -> bool:
        return self.store.ingest_run(observation, authoritative=authoritative)

    def record_odds_attempt(self, observation: OddsObservation) -> bool:
        return self.store.record_odds_attempt(observation)

    def record_field_evidence(self, evidence: FieldEvidence) -> bool:
        return self.store.record_field_evidence(evidence)

    def field_evidence(self, race_id: RaceId) -> tuple[FieldEvidence, ...]:
        return self.store.field_evidence(race_id)

    def quarantine_collection(
        self,
        operation_id: OperationId,
        race_id: RaceId,
        *,
        stage: str,
        code: str,
        details: str,
        at: datetime,
    ) -> bool:
        return self.store.record_collection_quarantine(
            operation_id, race_id, stage=stage, code=code, details=details, at=at
        )

    def reconcile(self, day: RacingDay) -> Reconciliation:
        rows = self.store.collection_rows(day)
        unresolved = tuple(
            str(row.race_id)
            for row in rows
            if not row.quarantined
            and row.state not in {RaceState.AWAITING_DAY_CLOSE, RaceState.PREDICTION_QUARANTINED}
        )
        return Reconciliation(
            str(day.id),
            len(rows),
            sum(row.state is not RaceState.DISCOVERED for row in rows),
            sum(
                row.state
                in {
                    RaceState.COLLECTING_ODDS,
                    RaceState.EVIDENCE_SEALED,
                    RaceState.AWAITING_DAY_CLOSE,
                }
                for row in rows
            ),
            sum(
                row.state in {RaceState.EVIDENCE_SEALED, RaceState.AWAITING_DAY_CLOSE}
                for row in rows
            ),
            sum(row.quarantined for row in rows),
            unresolved,
        )
