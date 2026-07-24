"""Field-authoritative construction of immutable Sealed Race Evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping

from .artifacts import ArtifactStore, ArtifactStoreError
from .domain import (
    ArtifactChecksum,
    DomainValidationError,
    EvidenceAuthority,
    EvidenceField,
    FreezeAuthority,
    OddsAttemptStatus,
    OperationId,
    RaceId,
    require_aware,
)
from .operations import (
    ConflictingOperation,
    OperationsStore,
    OperationsStoreError,
    iso_timestamp,
)


class SealingQuarantine(OperationsStoreError):
    """Evidence cannot be sealed without silently choosing a critical fact."""


AUTHORITY_RANK = {
    EvidenceAuthority.OFFICIAL_PROGRAMME: 60,
    EvidenceAuthority.OFFICIAL_JUMP: 60,
    EvidenceAuthority.OFFICIAL_CARD: 50,
    EvidenceAuthority.SOURCE_CARD: 40,
    EvidenceAuthority.MARKET: 30,
    EvidenceAuthority.EMBEDDED_FORM: 10,
}


def _json_value(value: Any) -> Any:
    return iso_timestamp(value) if isinstance(value, datetime) else value


def _exact_json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return iso_timestamp(value)
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        encoded = json.dumps(value, allow_nan=False)
        return json.loads(encoded)
    if type(value) is list:
        return [_exact_json_value(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _exact_json_value(item) for key, item in value.items()}
    raise DomainValidationError("field observation value must be exact JSON")


def _seal_request_digest(
    *,
    race_id: RaceId,
    requested_source_artifacts: Mapping[str, ArtifactChecksum],
    requested_observations: tuple[FieldObservation, ...],
    effective_source_artifacts: Mapping[str, ArtifactChecksum],
    effective_observations: tuple[FieldObservation, ...],
    effective_evidence_error: str | None,
    scheduled_jump: datetime,
    buffer: timedelta,
    schema_version: str,
    normalization_version: str,
    sealed_at: datetime,
) -> ArtifactChecksum:
    if not isinstance(race_id, RaceId):
        raise DomainValidationError("race_id must be a RaceId")

    def canonical_sources(
        source_artifacts: Mapping[str, ArtifactChecksum],
    ) -> dict[str, str]:
        if not isinstance(source_artifacts, Mapping):
            raise DomainValidationError("source_artifacts must be a mapping")
        sources: dict[str, str] = {}
        for source, checksum in source_artifacts.items():
            if type(source) is not str or not source.strip():
                raise DomainValidationError("source artifact name must be nonblank text")
            if not isinstance(checksum, ArtifactChecksum):
                raise DomainValidationError("source artifact checksum must be an ArtifactChecksum")
            sources[source] = str(checksum)
        return {key: sources[key] for key in sorted(sources)}

    def canonical_observations(
        observations: tuple[FieldObservation, ...],
    ) -> list[dict[str, Any]]:
        values = [
            {
                "field": item.field.value,
                "value": _exact_json_value(item.value),
                "authority": item.authority.value,
                "critical": item.critical,
                "source": item.source,
                "artifact_checksum": str(item.artifact_checksum),
            }
            for item in observations
        ]
        values.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        return values

    if effective_evidence_error is not None and type(effective_evidence_error) is not str:
        raise DomainValidationError("effective evidence error must be text or None")
    require_aware(scheduled_jump, "scheduled_jump")
    require_aware(sealed_at, "sealed_at")
    if type(buffer) is not timedelta:
        raise DomainValidationError("feature-freeze buffer must be a timedelta")
    if type(schema_version) is not str or not schema_version.strip():
        raise DomainValidationError("schema_version must be nonblank text")
    if type(normalization_version) is not str or not normalization_version.strip():
        raise DomainValidationError("normalization_version must be nonblank text")
    payload = {
        "race_id": str(race_id),
        "requested": {
            "sources": canonical_sources(requested_source_artifacts),
            "observations": canonical_observations(requested_observations),
        },
        "effective": {
            "sources": canonical_sources(effective_source_artifacts),
            "observations": canonical_observations(effective_observations),
            "error": effective_evidence_error,
        },
        "scheduled_jump": iso_timestamp(scheduled_jump),
        "buffer": {
            "days": buffer.days,
            "seconds": buffer.seconds,
            "microseconds": buffer.microseconds,
        },
        "schema_version": schema_version,
        "normalization_version": normalization_version,
        "sealed_at": iso_timestamp(sealed_at),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return ArtifactChecksum(f"sha256:{hashlib.sha256(encoded).hexdigest()}")


@dataclass(frozen=True, slots=True)
class FieldObservation:
    field: EvidenceField
    value: Any
    authority: EvidenceAuthority
    critical: bool
    source: str
    artifact_checksum: ArtifactChecksum

    def __post_init__(self) -> None:
        if not isinstance(self.field, EvidenceField):
            raise DomainValidationError("field observation field must be an EvidenceField")
        if not isinstance(self.source, str) or not self.source.strip():
            raise DomainValidationError("field observation source must be nonblank text")
        if not isinstance(self.authority, EvidenceAuthority):
            raise DomainValidationError("field observation authority must be an EvidenceAuthority")
        if not isinstance(self.critical, bool):
            raise DomainValidationError("field observation critical must be a bool")
        if self.critical != self.field.critical:
            raise DomainValidationError(
                "field observation criticality must match intrinsic field criticality"
            )
        if not isinstance(self.artifact_checksum, ArtifactChecksum):
            raise DomainValidationError("field observation checksum must be an ArtifactChecksum")
        if not isinstance(self.value, datetime):
            try:
                json.dumps(self.value, sort_keys=True, separators=(",", ":"))
            except (TypeError, ValueError) as error:
                raise DomainValidationError(
                    "field observation value must be JSON serializable"
                ) from error


def _observation_sort_key(item: FieldObservation) -> tuple[str, int, str, str, str, bool, str]:
    if isinstance(item.value, datetime):
        value = item.value.isoformat()
    else:
        try:
            value = json.dumps(item.value, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            value = repr(item.value)
    return (
        item.field.value,
        -AUTHORITY_RANK[item.authority],
        value,
        item.authority.value,
        item.source,
        item.field.critical,
        str(item.artifact_checksum),
    )


@dataclass(frozen=True, slots=True)
class SealResult:
    raw_manifest_checksum: ArtifactChecksum
    normalized_checksum: ArtifactChecksum
    odds_checksum: ArtifactChecksum
    frozen_at: datetime
    freeze_authority: FreezeAuthority


def normalize_fields(observations: Iterable[FieldObservation]) -> dict[str, Any]:
    grouped: dict[str, list[FieldObservation]] = {}
    for observation in observations:
        grouped.setdefault(observation.field.value, []).append(observation)
    normalized: dict[str, Any] = {}
    for field, candidates in sorted(grouped.items()):
        candidates.sort(key=_observation_sort_key)
        best_rank = max(AUTHORITY_RANK[item.authority] for item in candidates)
        best = [item for item in candidates if AUTHORITY_RANK[item.authority] == best_rank]
        serialized = {
            json.dumps(_json_value(item.value), sort_keys=True, separators=(",", ":"))
            for item in best
        }
        if any(item.field.critical for item in best) and len(serialized) != 1:
            raise SealingQuarantine(f"critical conflict for {field}")
        normalized[field] = _json_value(best[0].value)
    return normalized


class EvidenceSealer:
    def __init__(self, operations: OperationsStore, artifacts: ArtifactStore):
        self.operations = operations
        self.artifacts = artifacts

    def seal(
        self,
        *,
        operation_id: OperationId,
        race_id: RaceId,
        source_artifacts: Mapping[str, ArtifactChecksum],
        observations: Iterable[FieldObservation],
        scheduled_jump: datetime,
        buffer: timedelta,
        schema_version: str,
        normalization_version: str,
        sealed_at: datetime,
    ) -> SealResult:
        committed = self.operations.committed_seal(operation_id, race_id)
        quarantined = self.operations.quarantined_seal(operation_id, race_id)
        if committed is None and quarantined is None:
            quarantine = self.operations.collection_quarantine(race_id)
            if quarantine is not None:
                raise SealingQuarantine(quarantine.details)

        effective_observations: tuple[FieldObservation, ...] = ()
        effective_sources: dict[str, ArtifactChecksum] = {}
        effective_evidence_error: str | None = None
        evidence_error: Exception | None = None
        try:
            persisted = self.operations.field_evidence(race_id)
            if not persisted:
                effective_evidence_error = "missing: race has no transactional field evidence"
                evidence_error = SealingQuarantine("race has no transactional field evidence")
            else:
                effective_observations = tuple(
                    sorted(
                        (
                            FieldObservation(
                                item.field,
                                item.value,
                                item.authority,
                                item.field.critical,
                                item.source,
                                item.artifact_checksum,
                            )
                            for item in persisted
                        ),
                        key=_observation_sort_key,
                    )
                )
                for item in effective_observations:
                    _exact_json_value(item.value)
                    prior = effective_sources.setdefault(item.source, item.artifact_checksum)
                    if prior != item.artifact_checksum:
                        raise SealingQuarantine(
                            f"source {item.source!r} has conflicting transactional checksums"
                        )
        except (
            DomainValidationError,
            OperationsStoreError,
            SealingQuarantine,
            TypeError,
            ValueError,
        ) as error:
            evidence_error = error
            effective_evidence_error = f"{type(error).__name__}: {error}"

        caller_observations = tuple(observations)
        request_intent_digest = _seal_request_digest(
            race_id=race_id,
            requested_source_artifacts=source_artifacts,
            requested_observations=caller_observations,
            effective_source_artifacts=effective_sources,
            effective_observations=effective_observations,
            effective_evidence_error=effective_evidence_error,
            scheduled_jump=scheduled_jump,
            buffer=buffer,
            schema_version=schema_version,
            normalization_version=normalization_version,
            sealed_at=sealed_at,
        )
        if committed is not None:
            if request_intent_digest != committed.request_intent_digest:
                raise ConflictingOperation(
                    f"operation {operation_id} was replayed with different intent"
                )
            return SealResult(
                raw_manifest_checksum=committed.raw_manifest_checksum,
                normalized_checksum=committed.normalized_checksum,
                odds_checksum=committed.odds_checksum,
                frozen_at=committed.frozen_at,
                freeze_authority=committed.freeze_authority,
            )
        if quarantined is not None:
            if request_intent_digest != quarantined.request_intent_digest:
                raise ConflictingOperation(
                    f"operation {operation_id} was replayed with different intent"
                )
            raise SealingQuarantine(quarantined.details)

        if evidence_error is not None:
            code = (
                "missing_field_evidence"
                if str(evidence_error) == "race has no transactional field evidence"
                else "invalid_field_provenance"
            )
            self.operations.record_collection_quarantine(
                operation_id,
                race_id,
                stage="sealing",
                code=code,
                details=str(evidence_error),
                at=sealed_at,
                operation_kind="quarantine_sealing",
                request_intent_digest=request_intent_digest,
            )
            raise SealingQuarantine(str(evidence_error)) from evidence_error

        try:
            caller_observations = tuple(sorted(caller_observations, key=_observation_sort_key))
            observations = effective_observations
            if (
                effective_observations
                and caller_observations
                and caller_observations != effective_observations
            ):
                raise SealingQuarantine(
                    "caller field observations do not exactly match transactional evidence"
                )
            if (
                effective_sources
                and source_artifacts
                and dict(source_artifacts) != effective_sources
            ):
                raise SealingQuarantine(
                    "caller source artifacts do not exactly match transactional evidence"
                )
            source_artifacts = effective_sources
            require_aware(scheduled_jump, "scheduled_jump")
            if buffer < timedelta(0):
                raise DomainValidationError("feature-freeze buffer must be non-negative")
            programme = self.operations.expected_programme_artifact(race_id)
            if scheduled_jump != programme.scheduled_jump:
                raise SealingQuarantine(
                    "caller scheduled_jump does not match persisted expected programme"
                )
            self.artifacts.verify(programme.checksum)
            for checksum in source_artifacts.values():
                self.artifacts.verify(checksum)
            for item in observations:
                json.dumps(_json_value(item.value), sort_keys=True, separators=(",", ":"))
                if source_artifacts.get(item.source) != item.artifact_checksum:
                    raise SealingQuarantine(
                        f"field evidence for {item.field} is not tied to source artifacts"
                    )
                self.artifacts.verify(item.artifact_checksum)
            official_jumps = [
                item
                for item in observations
                if item.field is EvidenceField.ACTUAL_JUMP
                and item.authority is EvidenceAuthority.OFFICIAL_JUMP
            ]
            for item in official_jumps:
                if not isinstance(item.value, datetime):
                    raise SealingQuarantine("official jump value must be a datetime")
                require_aware(item.value, "official jump value")
            jump_values = {item.value for item in official_jumps}
            if len(jump_values) > 1:
                raise SealingQuarantine("critical conflict for actual_jump")
            if official_jumps:
                cutoff = official_jumps[0].value
                authority = FreezeAuthority.ACTUAL_JUMP
            else:
                cutoff = programme.scheduled_jump - buffer
                authority = FreezeAuthority.SCHEDULED_MINUS_BUFFER
        except (
            ArtifactStoreError,
            DomainValidationError,
            OperationsStoreError,
            SealingQuarantine,
            TypeError,
            ValueError,
        ) as error:
            if isinstance(error, SealingQuarantine) and str(error) == (
                "race has no transactional field evidence"
            ):
                raise
            self.operations.record_collection_quarantine(
                operation_id,
                race_id,
                stage="sealing",
                code="invalid_field_provenance",
                details=str(error),
                at=sealed_at,
                operation_kind="quarantine_sealing",
                request_intent_digest=request_intent_digest,
            )
            raise SealingQuarantine(str(error)) from error
        try:
            normalized = normalize_fields(observations)
        except SealingQuarantine as error:
            self.operations.record_collection_quarantine(
                operation_id,
                race_id,
                stage="sealing",
                code="critical_conflict",
                details=str(error),
                at=sealed_at,
                operation_kind="quarantine_sealing",
                request_intent_digest=request_intent_digest,
            )
            raise
        attempts = self.operations.odds_attempts(race_id)
        valid = [
            row
            for row in attempts
            if row.status is OddsAttemptStatus.SUCCEEDED and row.attempted_at < cutoff
        ]
        if not valid:
            error = SealingQuarantine("no valid odds observation before feature-freeze cutoff")
            self.operations.record_collection_quarantine(
                operation_id,
                race_id,
                stage="sealing",
                code="no_pre_freeze_odds",
                details=str(error),
                at=sealed_at,
                operation_kind="quarantine_sealing",
                request_intent_digest=request_intent_digest,
            )
            raise error
        chosen = valid[-1]
        try:
            for attempt in attempts:
                if attempt.artifact_checksum is not None:
                    self.artifacts.verify(attempt.artifact_checksum)
                if attempt.runner_mapping_checksum is not None:
                    self.artifacts.verify(attempt.runner_mapping_checksum)
        except ArtifactStoreError as error:
            self.operations.record_collection_quarantine(
                operation_id,
                race_id,
                stage="sealing",
                code="invalid_odds_provenance",
                details=str(error),
                at=sealed_at,
                operation_kind="quarantine_sealing",
                request_intent_digest=request_intent_digest,
            )
            raise SealingQuarantine(str(error)) from error
        raw_odds_attempts = [
            {
                "source": attempt.source,
                "attempted_at": iso_timestamp(attempt.attempted_at),
                "status": attempt.status.value,
                "artifact_checksum": (
                    str(attempt.artifact_checksum)
                    if attempt.artifact_checksum is not None
                    else None
                ),
                "runner_mapping_checksum": (
                    str(attempt.runner_mapping_checksum)
                    if attempt.runner_mapping_checksum is not None
                    else None
                ),
                "error": attempt.error,
            }
            for attempt in attempts
        ]
        raw_manifest = {
            "schema": "sealed-race-raw-manifest-v1",
            "race_id": str(race_id),
            "programme_artifact": {
                "source": programme.source,
                "artifact_checksum": str(programme.checksum),
            },
            "sources": {
                source: str(checksum) for source, checksum in sorted(source_artifacts.items())
            },
            "odds_attempts": raw_odds_attempts,
            "field_observations": [
                {
                    "field": item.field.value,
                    "value": _json_value(item.value),
                    "authority": item.authority.value,
                    "critical": item.field.critical,
                    "source": item.source,
                    "artifact_checksum": str(item.artifact_checksum),
                }
                for item in observations
            ],
        }
        normalized_package = {
            "schema_version": schema_version,
            "normalization_version": normalization_version,
            "race_id": str(race_id),
            "fields": normalized,
            "field_provenance": [
                {
                    "field": item.field.value,
                    "authority": item.authority.value,
                    "critical": item.field.critical,
                    "value": _json_value(item.value),
                    "source": item.source,
                    "artifact_checksum": str(item.artifact_checksum),
                }
                for item in observations
            ],
            "freeze": {
                "at": iso_timestamp(cutoff),
                "authority": authority.value,
                "odds_checksum": str(chosen.artifact_checksum),
            },
        }
        raw_artifact = self.artifacts.put(
            json.dumps(raw_manifest, sort_keys=True, separators=(",", ":")).encode(),
            media_type="application/json",
        )
        normalized_artifact = self.artifacts.put(
            json.dumps(normalized_package, sort_keys=True, separators=(",", ":")).encode(),
            media_type="application/json",
        )
        self.operations.seal_evidence(
            operation_id,
            race_id=race_id,
            raw_checksum=raw_artifact.checksum,
            normalized_checksum=normalized_artifact.checksum,
            schema_version=schema_version,
            normalization_version=normalization_version,
            frozen_at=cutoff,
            freeze_authority=authority,
            odds_checksum=chosen.artifact_checksum,
            sealed_at=sealed_at,
            request_intent_digest=request_intent_digest,
        )
        committed = self.operations.committed_seal(operation_id, race_id)
        if committed is None:  # pragma: no cover - the transaction above committed this row
            raise OperationsStoreError("committed seal could not be read back")
        if committed.request_intent_digest != request_intent_digest:  # pragma: no cover
            raise ConflictingOperation(
                f"operation {operation_id} was replayed with different intent"
            )
        return SealResult(
            raw_manifest_checksum=committed.raw_manifest_checksum,
            normalized_checksum=committed.normalized_checksum,
            odds_checksum=committed.odds_checksum,
            frozen_at=committed.frozen_at,
            freeze_authority=committed.freeze_authority,
        )
