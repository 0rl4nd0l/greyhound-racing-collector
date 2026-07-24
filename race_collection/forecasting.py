"""Phase-3 deferred prediction and result authority.

Prediction computation is injected.  This module owns only durable intent,
barriers, provenance, retries, and joins; Phase 4 owns loading and features.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Mapping, Protocol

from .domain import (
    ArtifactChecksum,
    OperationId,
    RaceId,
    RaceLifecycle,
    RaceState,
    RacingDayId,
    require_aware,
)
from .operations import (
    BarrierNotSatisfied,
    ConflictingOperation,
    OperationsStoreError,
    SQLiteOperationsStore,
    _payload_hash,
    iso_timestamp,
)


def _text(value: str, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be nonblank text")
    return value


def _json(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("value must be exact JSON") from error
    if json.loads(encoded) != value:
        raise ValueError("value must round-trip as exact JSON")
    return encoded


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json(dict(value)).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class LegacyBundle:
    bundle_id: str
    model_id: str
    artifact_checksum: ArtifactChecksum
    artifact_size: int
    metadata_checksum: ArtifactChecksum
    scaler_checksum: ArtifactChecksum | None
    envelope_kind: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        _text(self.bundle_id, "bundle_id")
        _text(self.model_id, "model_id")
        for value, name in (
            (self.artifact_checksum, "artifact_checksum"),
            (self.metadata_checksum, "metadata_checksum"),
        ):
            if not isinstance(value, ArtifactChecksum):
                raise ValueError(f"{name} must be an ArtifactChecksum")
        if self.scaler_checksum is not None and not isinstance(
            self.scaler_checksum, ArtifactChecksum
        ):
            raise ValueError("scaler_checksum must be an ArtifactChecksum or None")
        if type(self.artifact_size) is not int or self.artifact_size <= 0:
            raise ValueError("artifact_size must be positive")
        if self.envelope_kind not in {"raw_registry_model", "v4_full_envelope"}:
            raise ValueError("invalid envelope_kind")
        _json(dict(self.provenance))


@dataclass(frozen=True, slots=True)
class ModelRelease:
    release_id: str
    bundle_id: str
    policy_id: str
    descriptor: Mapping[str, Any]

    def __post_init__(self) -> None:
        for value, name in (
            (self.release_id, "release_id"),
            (self.bundle_id, "bundle_id"),
            (self.policy_id, "policy_id"),
        ):
            _text(value, name)
        _json(dict(self.descriptor))


@dataclass(frozen=True, slots=True)
class PredictionRequest:
    race_id: RaceId
    racing_day_id: RacingDayId
    seal_id: int
    evidence_checksum: ArtifactChecksum
    bundle: LegacyBundle
    release: ModelRelease
    policy_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.race_id, RaceId):
            raise ValueError("race_id must be a RaceId")
        if not isinstance(self.racing_day_id, RacingDayId):
            raise ValueError("racing_day_id must be a RacingDayId")
        if type(self.seal_id) is not int or self.seal_id <= 0:
            raise ValueError("seal_id must be positive")
        if not isinstance(self.evidence_checksum, ArtifactChecksum):
            raise ValueError("evidence_checksum must be an ArtifactChecksum")
        if not isinstance(self.bundle, LegacyBundle) or not isinstance(self.release, ModelRelease):
            raise ValueError("bundle and release must be validated domain values")
        _text(self.policy_id, "policy_id")
        if self.release.bundle_id != self.bundle.bundle_id:
            raise ValueError("release bundle must match prediction bundle")
        if self.release.policy_id != self.policy_id:
            raise ValueError("release policy must match prediction policy")


@dataclass(frozen=True, slots=True)
class PredictionOutcome:
    status: Literal["committed", "quarantined"]
    prediction_id: str
    artifact_checksum: ArtifactChecksum | None = None
    code: str | None = None
    details: str | None = None
    replayed: bool = False


class DeferredPredictor(Protocol):
    def predict(self, request: PredictionRequest) -> ArtifactChecksum: ...


class ForecastingAuthority:
    """Narrow transactional facade for all Phase-3 workflow mutations."""

    def __init__(self, store: SQLiteOperationsStore):
        self.store = store

    def register_bundle(
        self, operation_id: OperationId, bundle: LegacyBundle, at: datetime
    ) -> bool:
        require_aware(at, "at")
        payload = {
            "bundle": bundle.bundle_id,
            "model": bundle.model_id,
            "artifact": str(bundle.artifact_checksum),
            "size": bundle.artifact_size,
            "metadata": str(bundle.metadata_checksum),
            "scaler": str(bundle.scaler_checksum) if bundle.scaler_checksum else None,
            "envelope": bundle.envelope_kind,
            "provenance": dict(bundle.provenance),
            "at": iso_timestamp(at),
        }
        with self.store._operation(operation_id, "register_legacy_bundle", payload) as (
            db,
            replay,
        ):
            if replay:
                return False
            db.execute(
                "INSERT INTO model_bundles VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    bundle.bundle_id,
                    "legacy-origin",
                    bundle.model_id,
                    str(bundle.artifact_checksum),
                    bundle.artifact_size,
                    str(bundle.metadata_checksum),
                    str(bundle.scaler_checksum) if bundle.scaler_checksum else None,
                    bundle.envelope_kind,
                    _json(dict(bundle.provenance)),
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
        return True

    def register_release(
        self, operation_id: OperationId, release: ModelRelease, at: datetime
    ) -> bool:
        require_aware(at, "at")
        payload = {
            "release": release.release_id,
            "bundle": release.bundle_id,
            "policy": release.policy_id,
            "descriptor": dict(release.descriptor),
            "at": iso_timestamp(at),
        }
        with self.store._operation(operation_id, "register_model_release", payload) as (
            db,
            replay,
        ):
            if replay:
                return False
            db.execute(
                "INSERT INTO model_releases VALUES(?,?,?,?,?,?)",
                (
                    release.release_id,
                    release.bundle_id,
                    release.policy_id,
                    _json(dict(release.descriptor)),
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
        return True

    def pin_day(
        self,
        operation_id: OperationId,
        day_id: RacingDayId,
        release: ModelRelease,
        at: datetime,
    ) -> bool:
        require_aware(at, "at")
        payload = {
            "day": str(day_id),
            "bundle": release.bundle_id,
            "release": release.release_id,
            "policy": release.policy_id,
            "at": iso_timestamp(at),
        }
        with self.store._operation(operation_id, "pin_racing_day", payload) as (
            db,
            replay,
        ):
            if replay:
                return False
            row = db.execute(
                "SELECT closed_at FROM racing_days WHERE racing_day_id=?",
                (str(day_id),),
            ).fetchone()
            if row is None:
                raise OperationsStoreError(f"unknown racing day {day_id}")
            if row["closed_at"] is not None:
                raise BarrierNotSatisfied("model pin must be committed before racing-day closure")
            db.execute(
                "INSERT INTO racing_day_pins VALUES(?,?,?,?,?,?)",
                (
                    str(day_id),
                    release.bundle_id,
                    release.release_id,
                    release.policy_id,
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
        return True

    def _prediction_context(
        self, db: Any, race_id: RaceId, expected_state: RaceState | None
    ) -> Any:
        row = db.execute(
            "SELECT r.state,r.racing_day_id,r.updated_at,d.closed_at,p.bundle_id,p.release_id,p.policy_id,b.model_id,b.artifact_checksum,b.artifact_size,b.metadata_checksum,b.scaler_checksum,b.envelope_kind,b.provenance_json,s.seal_id,s.normalized_checksum,m.descriptor_json FROM races r JOIN racing_days d USING(racing_day_id) JOIN racing_day_pins p USING(racing_day_id) JOIN model_bundles b USING(bundle_id) JOIN model_releases m ON m.release_id=p.release_id AND m.bundle_id=p.bundle_id AND m.policy_id=p.policy_id JOIN sealed_evidence s ON s.race_id=r.race_id WHERE r.race_id=?",
            (str(race_id),),
        ).fetchone()
        if row is None:
            raise BarrierNotSatisfied("race requires sealed evidence and an immutable day pin")
        if row["closed_at"] is None or (
            expected_state is not None and row["state"] != expected_state.value
        ):
            raise BarrierNotSatisfied(
                "deferred prediction requires closed day and "
                f"{expected_state.value if expected_state else 'authoritative context'}"
            )
        return row

    @staticmethod
    def _snapshot(row: Any, race_id: RaceId, prediction_id: str) -> dict[str, Any]:
        return {
            "race": str(race_id),
            "prediction": prediction_id,
            "day": row["racing_day_id"],
            "seal": row["seal_id"],
            "evidence": row["normalized_checksum"],
            "bundle": row["bundle_id"],
            "release": row["release_id"],
            "policy": row["policy_id"],
            "descriptor": json.loads(row["descriptor_json"]),
            "closed_at": row["closed_at"],
            "race_updated_at": row["updated_at"],
        }

    def _replayed_prediction(self, db, operation_id, race_id, prediction_id, at):
        operation_row = db.execute(
            "SELECT kind,payload_sha256 FROM operations WHERE operation_id=?",
            (str(operation_id),),
        ).fetchone()
        if operation_row is None:
            return None
        row = db.execute(
            "SELECT p.prediction_id,p.race_id,p.racing_day_id,p.bundle_id,p.release_id,p.policy_id,p.seal_id,p.evidence_checksum,p.artifact_checksum,p.computed_at terminal_at,p.request_intent_sha256,p.authority_snapshot_json,NULL code,NULL details,'committed' status FROM deferred_predictions p WHERE p.operation_id=? UNION ALL SELECT q.prediction_id,q.race_id,q.racing_day_id,q.bundle_id,q.release_id,q.policy_id,q.seal_id,q.evidence_checksum,NULL,q.quarantined_at,q.request_intent_sha256,q.authority_snapshot_json,q.code,q.details,'quarantined' FROM prediction_quarantines q WHERE q.operation_id=?",
            (str(operation_id), str(operation_id)),
        ).fetchone()
        if (
            operation_row["kind"] != "execute_deferred_prediction"
            or row is None
            or row["race_id"] != str(race_id)
            or row["prediction_id"] != prediction_id
            or row["terminal_at"] != iso_timestamp(at)
        ):
            raise ConflictingOperation(
                f"operation {operation_id} was replayed with different intent"
            )
        snapshot = json.loads(row["authority_snapshot_json"])
        persisted = {
            "race": row["race_id"],
            "prediction": row["prediction_id"],
            "day": row["racing_day_id"],
            "seal": row["seal_id"],
            "evidence": row["evidence_checksum"],
            "bundle": row["bundle_id"],
            "release": row["release_id"],
            "policy": row["policy_id"],
        }
        if any(snapshot.get(name) != value for name, value in persisted.items()):
            raise OperationsStoreError("terminal row disagrees with authority snapshot")
        payload = {
            **snapshot,
            "status": row["status"],
            "artifact": row["artifact_checksum"],
            "code": row["code"],
            "details": row["details"],
            "at": row["terminal_at"],
        }
        if (
            _payload_hash(payload) != operation_row["payload_sha256"]
            or _digest(payload) != row["request_intent_sha256"]
        ):
            raise OperationsStoreError(
                f"prediction operation {operation_id} has inconsistent durable intent"
            )
        artifact = ArtifactChecksum(row["artifact_checksum"]) if row["artifact_checksum"] else None
        return PredictionOutcome(
            row["status"], prediction_id, artifact, row["code"], row["details"], True
        )

    def _replayed_manual_quarantine(self, db, operation_id, race_id, code, details, at):
        operation_row = db.execute(
            "SELECT kind,payload_sha256 FROM operations WHERE operation_id=?",
            (str(operation_id),),
        ).fetchone()
        if operation_row is None:
            return None
        row = db.execute(
            "SELECT q.* FROM prediction_quarantines q WHERE q.operation_id=?",
            (str(operation_id),),
        ).fetchone()
        prediction_id = f"quarantine-{operation_id}"
        if (
            operation_row["kind"] != "quarantine_deferred_prediction"
            or row is None
            or row["race_id"] != str(race_id)
            or row["prediction_id"] != prediction_id
            or row["code"] != code
            or row["details"] != details
            or row["quarantined_at"] != iso_timestamp(at)
        ):
            raise ConflictingOperation(
                f"operation {operation_id} was replayed with different intent"
            )
        snapshot = json.loads(row["authority_snapshot_json"])
        persisted = {
            "race": row["race_id"],
            "prediction": row["prediction_id"],
            "day": row["racing_day_id"],
            "seal": row["seal_id"],
            "evidence": row["evidence_checksum"],
            "bundle": row["bundle_id"],
            "release": row["release_id"],
            "policy": row["policy_id"],
        }
        if any(snapshot.get(name) != value for name, value in persisted.items()):
            raise OperationsStoreError("quarantine row disagrees with authority snapshot")
        payload = {
            **snapshot,
            "code": row["code"],
            "details": row["details"],
            "at": row["quarantined_at"],
        }
        if (
            _payload_hash(payload) != operation_row["payload_sha256"]
            or _digest(payload) != row["request_intent_sha256"]
        ):
            raise OperationsStoreError(
                f"prediction quarantine {operation_id} has inconsistent durable intent"
            )
        return False

    def _replayed_begin(
        self,
        db: Any,
        operation_id: OperationId,
        race_id: RaceId,
        at: datetime,
        expected_payload: Mapping[str, Any] | None = None,
    ) -> bool | None:
        operation_row = db.execute(
            "SELECT kind,payload_sha256 FROM operations WHERE operation_id=?",
            (str(operation_id),),
        ).fetchone()
        begun = db.execute(
            "SELECT * FROM prediction_begins WHERE operation_id=?",
            (str(operation_id),),
        ).fetchone()
        if operation_row is None:
            return None
        if (
            operation_row["kind"] != "begin_deferred_prediction"
            or begun is None
            or begun["race_id"] != str(race_id)
            or begun["begun_at"] != iso_timestamp(at)
        ):
            raise ConflictingOperation(
                f"operation {operation_id} was replayed with different intent"
            )
        payload = {
            **json.loads(begun["authority_snapshot_json"]),
            "at": begun["begun_at"],
        }
        if expected_payload is not None and _payload_hash(expected_payload) != _payload_hash(
            payload
        ):
            raise ConflictingOperation(
                f"operation {operation_id} was replayed with different intent"
            )
        if (
            _payload_hash(payload) != operation_row["payload_sha256"]
            or _digest(payload) != begun["request_intent_sha256"]
        ):
            raise OperationsStoreError(
                f"prediction begin {operation_id} has inconsistent durable intent"
            )
        return False

    def begin_prediction(self, operation_id: OperationId, race_id: RaceId, at: datetime) -> bool:
        require_aware(at, "at")
        with self.store._connect() as db:
            replayed = self._replayed_begin(db, operation_id, race_id, at)
            if replayed is not None:
                return replayed
            row = self._prediction_context(db, race_id, None)
            snapshot = self._snapshot(row, race_id, f"begin-{race_id}")
        payload = {**snapshot, "at": iso_timestamp(at)}
        with self.store._operation(operation_id, "begin_deferred_prediction", payload) as (
            db,
            replay,
        ):
            if replay:
                concurrent = self._replayed_begin(
                    db, operation_id, race_id, at, expected_payload=payload
                )
                if concurrent is None:
                    raise OperationsStoreError("concurrent prediction begin replay disappeared")
                return concurrent
            current = self._prediction_context(db, race_id, RaceState.AWAITING_DAY_CLOSE)
            if self._snapshot(current, race_id, f"begin-{race_id}") != snapshot:
                raise BarrierNotSatisfied("prediction authority snapshot changed before begin")
            db.execute(
                "INSERT INTO prediction_begins VALUES(?,?,?,?,?)",
                (
                    str(race_id),
                    _json(snapshot),
                    iso_timestamp(at),
                    _digest(payload),
                    str(operation_id),
                ),
            )
            self._transition(
                db,
                operation_id,
                race_id,
                RaceState.PREDICTION_PENDING,
                at,
                RaceState.AWAITING_DAY_CLOSE,
            )
        return True

    def predict(
        self,
        operation_id: OperationId,
        race_id: RaceId,
        prediction_id: str,
        predictor: DeferredPredictor,
        at: datetime,
    ) -> PredictionOutcome:
        require_aware(at, "at")
        _text(prediction_id, "prediction_id")
        with self.store._connect() as db:
            replayed = self._replayed_prediction(db, operation_id, race_id, prediction_id, at)
            if replayed is not None:
                return replayed
            row = self._prediction_context(db, race_id, RaceState.PREDICTION_PENDING)
            snapshot = self._snapshot(row, race_id, prediction_id)
            bundle = LegacyBundle(
                row["bundle_id"],
                row["model_id"],
                ArtifactChecksum(row["artifact_checksum"]),
                row["artifact_size"],
                ArtifactChecksum(row["metadata_checksum"]),
                ArtifactChecksum(row["scaler_checksum"]) if row["scaler_checksum"] else None,
                row["envelope_kind"],
                json.loads(row["provenance_json"]),
            )
            release = ModelRelease(
                row["release_id"],
                row["bundle_id"],
                row["policy_id"],
                json.loads(row["descriptor_json"]),
            )
            request = PredictionRequest(
                race_id,
                RacingDayId(row["racing_day_id"]),
                row["seal_id"],
                ArtifactChecksum(row["normalized_checksum"]),
                bundle,
                release,
                row["policy_id"],
            )
        try:
            artifact = predictor.predict(request)
            if not isinstance(artifact, ArtifactChecksum):
                raise OperationsStoreError("predictor must return an ArtifactChecksum")
            authenticate = getattr(predictor, "authenticate", None)
            if callable(authenticate):
                authenticate(artifact, at)
            status, code, details = "committed", None, None
        except Exception as error:
            artifact = None
            status = "quarantined"
            code = type(error).__name__
            details = str(error) or repr(error)
        payload = {
            **snapshot,
            "status": status,
            "artifact": str(artifact) if artifact else None,
            "code": code,
            "details": details,
            "at": iso_timestamp(at),
        }
        with self.store._operation(operation_id, "execute_deferred_prediction", payload) as (
            db,
            replay,
        ):
            if replay:
                durable = db.execute(
                    "SELECT payload_sha256 FROM operations WHERE operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                if durable is None or durable["payload_sha256"] != _payload_hash(payload):
                    raise ConflictingOperation(
                        f"operation {operation_id} computed a different terminal outcome"
                    )
                concurrent = self._replayed_prediction(db, operation_id, race_id, prediction_id, at)
                if concurrent is None:
                    raise OperationsStoreError("concurrent prediction replay disappeared")
                return concurrent
            current = self._prediction_context(db, race_id, RaceState.PREDICTION_PENDING)
            if self._snapshot(current, race_id, prediction_id) != snapshot:
                raise BarrierNotSatisfied("prediction authority snapshot changed before commit")
            intent = _digest(payload)
            if status == "committed":
                assert artifact is not None
                db.execute(
                    "INSERT INTO deferred_predictions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        prediction_id,
                        str(race_id),
                        current["racing_day_id"],
                        current["bundle_id"],
                        current["release_id"],
                        current["policy_id"],
                        current["seal_id"],
                        current["normalized_checksum"],
                        str(artifact),
                        iso_timestamp(at),
                        intent,
                        _json(snapshot),
                        str(operation_id),
                    ),
                )
                self._transition(
                    db,
                    operation_id,
                    race_id,
                    RaceState.PREDICTION_COMMITTED,
                    at,
                    RaceState.PREDICTION_PENDING,
                )
            else:
                db.execute(
                    "INSERT INTO prediction_quarantines VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        str(race_id),
                        prediction_id,
                        current["racing_day_id"],
                        current["bundle_id"],
                        current["release_id"],
                        current["policy_id"],
                        current["seal_id"],
                        current["normalized_checksum"],
                        code,
                        details,
                        iso_timestamp(at),
                        intent,
                        _json(snapshot),
                        str(operation_id),
                    ),
                )
                self._transition(
                    db,
                    operation_id,
                    race_id,
                    RaceState.PREDICTION_QUARANTINED,
                    at,
                    RaceState.PREDICTION_PENDING,
                )
        return PredictionOutcome(status, prediction_id, artifact, code, details)

    def quarantine_prediction(
        self,
        operation_id: OperationId,
        race_id: RaceId,
        code: str,
        details: str,
        at: datetime,
    ) -> bool:
        require_aware(at, "at")
        _text(code, "code")
        _text(details, "details")
        with self.store._connect() as db:
            replayed = self._replayed_manual_quarantine(
                db, operation_id, race_id, code, details, at
            )
            if replayed is not None:
                return replayed
        base_payload = {
            "race": str(race_id),
            "code": code,
            "details": details,
            "at": iso_timestamp(at),
        }
        with self.store._connect() as db:
            snapshot_row = self._prediction_context(db, race_id, RaceState.PREDICTION_PENDING)
            prediction_id = f"quarantine-{operation_id}"
            payload = {
                **self._snapshot(snapshot_row, race_id, prediction_id),
                **base_payload,
            }
        with self.store._operation(operation_id, "quarantine_deferred_prediction", payload) as (
            db,
            replay,
        ):
            if replay:
                return False
            row = self._prediction_context(db, race_id, RaceState.PREDICTION_PENDING)
            if self._snapshot(row, race_id, prediction_id) != self._snapshot(
                snapshot_row, race_id, prediction_id
            ):
                raise BarrierNotSatisfied("prediction authority snapshot changed before quarantine")
            intent = _digest(payload)
            db.execute(
                "INSERT INTO prediction_quarantines VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(race_id),
                    prediction_id,
                    row["racing_day_id"],
                    row["bundle_id"],
                    row["release_id"],
                    row["policy_id"],
                    row["seal_id"],
                    row["normalized_checksum"],
                    code,
                    details,
                    iso_timestamp(at),
                    intent,
                    _json(self._snapshot(snapshot_row, race_id, prediction_id)),
                    str(operation_id),
                ),
            )
            self._transition(
                db,
                operation_id,
                race_id,
                RaceState.PREDICTION_QUARANTINED,
                at,
                RaceState.PREDICTION_PENDING,
            )
        return True

    def open_results(self, operation_id: OperationId, race_id: RaceId, at: datetime) -> bool:
        require_aware(at, "at")
        payload = {"race": str(race_id), "at": iso_timestamp(at)}
        with self.store._operation(operation_id, "open_result_collection", payload) as (
            db,
            replay,
        ):
            if replay:
                return False
            row = db.execute(
                "SELECT racing_day_id,state FROM races WHERE race_id=?", (str(race_id),)
            ).fetchone()
            if row is None or row["state"] != RaceState.PREDICTION_COMMITTED.value:
                raise BarrierNotSatisfied("results require a committed prediction")
            incomplete = db.execute(
                "SELECT COUNT(*) FROM expected_races e JOIN races r USING(race_id) WHERE r.racing_day_id=? AND NOT EXISTS(SELECT 1 FROM collection_quarantines q WHERE q.race_id=r.race_id) AND r.state NOT IN ('prediction_committed','prediction_quarantined','result_pending','result_collected','result_quarantined','training_example_ready','evaluation_ineligible')",
                (row["racing_day_id"],),
            ).fetchone()[0]
            if incomplete:
                raise BarrierNotSatisfied("prediction day barrier is incomplete")
            self._transition(
                db,
                operation_id,
                race_id,
                RaceState.RESULT_PENDING,
                at,
                RaceState.PREDICTION_COMMITTED,
            )
        return True

    def record_result_attempt(
        self,
        operation_id: OperationId,
        race_id: RaceId,
        attempt_id: str,
        *,
        at: datetime,
        max_attempts: int,
        deadline: datetime,
        artifact_checksum: ArtifactChecksum | None = None,
        outcome: Any = None,
        error: str | None = None,
    ) -> str:
        require_aware(at, "at")
        require_aware(deadline, "deadline")
        _text(attempt_id, "attempt_id")
        if type(max_attempts) is not int or max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if artifact_checksum is not None and not isinstance(artifact_checksum, ArtifactChecksum):
            raise ValueError("artifact_checksum must be an ArtifactChecksum or None")
        payload = {
            "race": str(race_id),
            "attempt": attempt_id,
            "at": iso_timestamp(at),
            "max": max_attempts,
            "deadline": iso_timestamp(deadline),
            "artifact": str(artifact_checksum) if artifact_checksum else None,
            "outcome": outcome,
            "error": error,
        }
        with self.store._operation(operation_id, "record_result_attempt", payload) as (
            db,
            replay,
        ):
            if replay:
                prior = db.execute(
                    "SELECT status FROM result_attempts WHERE operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                return prior["status"]
            state = db.execute(
                "SELECT state FROM races WHERE race_id=?", (str(race_id),)
            ).fetchone()
            if state is None or state["state"] != RaceState.RESULT_PENDING.value:
                raise BarrierNotSatisfied("result attempts require result_pending")
            prior_policy = db.execute(
                "SELECT max_attempts,deadline,attempted_at FROM result_attempts "
                "WHERE race_id=? ORDER BY attempt_number DESC LIMIT 1",
                (str(race_id),),
            ).fetchone()
            if prior_policy is not None:
                stored_deadline = datetime.fromisoformat(prior_policy["deadline"])
                if prior_policy["max_attempts"] != max_attempts or stored_deadline != deadline:
                    raise ConflictingOperation("result retry policy is immutable")
                prior_at = datetime.fromisoformat(prior_policy["attempted_at"])
                if (at - prior_at).total_seconds() < 1:
                    raise BarrierNotSatisfied(
                        "result retries require at least one second of backoff"
                    )
            number = (
                db.execute(
                    "SELECT COUNT(*) FROM result_attempts WHERE race_id=?",
                    (str(race_id),),
                ).fetchone()[0]
                + 1
            )
            collected = artifact_checksum is not None and outcome is not None and error is None
            if collected:
                if not isinstance(outcome, Mapping):
                    raise ValueError("collected result must be an exact result mapping")
                order = outcome.get("order")
                if (
                    not isinstance(order, list)
                    or not order
                    or any(type(box) is not int or box <= 0 for box in order)
                    or len(set(order)) != len(order)
                ):
                    raise ValueError("collected result order must be nonempty and unambiguous")
            terminal = not collected and (number >= max_attempts or at >= deadline)
            status = "collected" if collected else "quarantined" if terminal else "failed"
            if not collected and not error:
                raise ValueError("failed result attempt requires error")
            db.execute(
                "INSERT INTO result_attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    attempt_id,
                    str(race_id),
                    number,
                    max_attempts,
                    iso_timestamp(deadline),
                    "result-retry-v1",
                    1,
                    status,
                    str(artifact_checksum) if collected else None,
                    _json(outcome) if collected else None,
                    None if collected else error,
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
            if status == "collected":
                self._transition(
                    db,
                    operation_id,
                    race_id,
                    RaceState.RESULT_COLLECTED,
                    at,
                    RaceState.RESULT_PENDING,
                )
            elif status == "quarantined":
                self._transition(
                    db,
                    operation_id,
                    race_id,
                    RaceState.RESULT_QUARANTINED,
                    at,
                    RaceState.RESULT_PENDING,
                )
        return status

    def join_training_example(
        self,
        operation_id: OperationId,
        race_id: RaceId,
        example_id: str,
        artifact: ArtifactChecksum,
        *,
        eligible: bool,
        reason: str | None,
        at: datetime,
    ) -> bool:
        require_aware(at, "at")
        _text(example_id, "example_id")
        if not isinstance(artifact, ArtifactChecksum):
            raise ValueError("artifact must be an ArtifactChecksum")
        if type(eligible) is not bool:
            raise ValueError("eligible must be bool")
        payload = {
            "race": str(race_id),
            "example": example_id,
            "artifact": str(artifact),
            "eligible": eligible,
            "reason": reason,
            "at": iso_timestamp(at),
        }
        with self.store._operation(operation_id, "join_training_example", payload) as (
            db,
            replay,
        ):
            if replay:
                return False
            row = db.execute(
                "SELECT p.prediction_id,a.attempt_id,r.state FROM deferred_predictions p JOIN races r USING(race_id) JOIN result_attempts a ON a.race_id=p.race_id AND a.status='collected' WHERE p.race_id=?",
                (str(race_id),),
            ).fetchone()
            if row is None or row["state"] != RaceState.RESULT_COLLECTED.value:
                raise BarrierNotSatisfied(
                    "training join requires committed prediction and collected result"
                )
            status = "eligible" if eligible else "evaluation_ineligible"
            if not eligible and not reason:
                raise ValueError("ineligible join requires reason")
            db.execute(
                "INSERT INTO training_examples VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    example_id,
                    str(race_id),
                    row["prediction_id"],
                    row["attempt_id"],
                    str(artifact),
                    status,
                    reason,
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
            self._transition(
                db,
                operation_id,
                race_id,
                RaceState.TRAINING_EXAMPLE_READY if eligible else RaceState.EVALUATION_INELIGIBLE,
                at,
                RaceState.RESULT_COLLECTED,
            )
        return True

    def record_on_demand(
        self,
        operation_id: OperationId,
        forecast_id: str,
        race_id: RaceId,
        artifact: ArtifactChecksum,
        evidence: ArtifactChecksum,
        at: datetime,
    ) -> bool:
        require_aware(at, "at")
        _text(forecast_id, "forecast_id")
        if not isinstance(artifact, ArtifactChecksum) or not isinstance(evidence, ArtifactChecksum):
            raise ValueError("on-demand checksums must be ArtifactChecksum values")
        payload = {
            "forecast": forecast_id,
            "race": str(race_id),
            "artifact": str(artifact),
            "evidence": str(evidence),
            "at": iso_timestamp(at),
        }
        with self.store._operation(operation_id, "record_on_demand_forecast", payload) as (
            db,
            replay,
        ):
            if replay:
                return False
            db.execute(
                "INSERT INTO on_demand_forecasts VALUES(?,?,?,?,?,?)",
                (
                    forecast_id,
                    str(race_id),
                    str(artifact),
                    str(evidence),
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
        return True

    @staticmethod
    def _transition(
        db: Any,
        operation_id: OperationId,
        race_id: RaceId,
        target: RaceState,
        at: datetime,
        expected: RaceState,
    ) -> None:
        RaceLifecycle.validate(expected, target)
        changed = db.execute(
            "UPDATE races SET state=?,updated_at=? WHERE race_id=? AND state=?",
            (target.value, iso_timestamp(at), str(race_id), expected.value),
        ).rowcount
        if changed != 1:
            raise OperationsStoreError(f"race {race_id} is not {expected.value}")
        db.execute(
            "INSERT INTO lifecycle_events(race_id,prior_state,target_state,occurred_at,operation_id) VALUES(?,?,?,?,?)",
            (
                str(race_id),
                expected.value,
                target.value,
                iso_timestamp(at),
                str(operation_id),
            ),
        )
