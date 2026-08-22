"""Narrow transactional authority for Race Collection Service operations."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

from .domain import (
    ArtifactChecksum,
    CollectionQuarantine,
    CollectionRaceRecord,
    DogId,
    DogRun,
    DomainValidationError,
    EvidenceAuthority,
    EvidenceField,
    ExpectedProgrammeArtifact,
    FieldEvidence,
    FreezeAuthority,
    IdentityTier,
    OddsAttemptRecord,
    OddsAttemptStatus,
    OddsObservation,
    OperationId,
    ProgrammeRaceCandidate,
    Quarantine,
    RaceId,
    RaceLifecycle,
    RaceState,
    RacingDay,
    RunObservation,
    Supersession,
)


class OperationsStoreError(RuntimeError):
    """Base error raised at the operations-store boundary."""


class ConflictingOperation(OperationsStoreError):
    """An operation ID was replayed with different intent."""


class BarrierNotSatisfied(OperationsStoreError):
    """A racing-day or prediction barrier has not been reached."""


class _ClosingConnection(sqlite3.Connection):
    """Commit/rollback like sqlite3, then release the owned database handle."""

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class RaceAliasOwnerCollision(OperationsStoreError):
    """Explicit reconciliation contradicted an alias's durable owner."""

    def __init__(self, source: str, alias: str, owner: RaceId, requested: RaceId):
        self.source = source
        self.alias = alias
        self.owner = owner
        self.requested = requested
        super().__init__(
            f"race alias ({source!r}, {alias!r}) remains owned by {owner}; "
            f"cannot reconcile to {requested}"
        )


class ExpectedInventoryConflict(OperationsStoreError):
    """A later programme observation conflicts with immutable expected inventory."""

    def __init__(self, race_id: RaceId, details: str):
        self.race_id = race_id
        self.details = details
        super().__init__(f"immutable expected inventory conflict for {race_id}: {details}")


class CollectionQuarantineBlocksSeal(OperationsStoreError):
    """Terminal collection quarantine prevents a fresh evidence seal."""


@dataclass(frozen=True, slots=True)
class CommittedSeal:
    raw_manifest_checksum: ArtifactChecksum
    normalized_checksum: ArtifactChecksum
    odds_checksum: ArtifactChecksum
    frozen_at: datetime
    freeze_authority: FreezeAuthority
    request_intent_digest: ArtifactChecksum


@dataclass(frozen=True, slots=True)
class QuarantinedSeal:
    stage: str
    code: str
    details: str
    request_intent_digest: ArtifactChecksum


class OperationsStore(Protocol):
    def migrate(self) -> None: ...

    def create_racing_day(self, operation_id: OperationId, day: RacingDay) -> bool: ...

    def register_forward_baseline_cohort(
        self,
        operation_id: OperationId,
        *,
        cohort_id: str,
        artifact_checksum: ArtifactChecksum,
        frozen_at: datetime,
        members: Sequence[Mapping[str, Any]],
        registered_at: datetime,
    ) -> bool: ...

    def forward_baseline_cohort(self, cohort_id: str) -> Mapping[str, Any] | None: ...

    def forward_baseline_cohort_lifecycle(
        self, cohort_id: str
    ) -> Mapping[str, Any] | None: ...

    def discover_race(
        self, operation_id: OperationId, race_id: RaceId, day: RacingDay, at: datetime
    ) -> bool: ...

    def advance_race(
        self, operation_id: OperationId, race_id: RaceId, target: RaceState, at: datetime
    ) -> bool: ...

    def record_expected_race(
        self,
        operation_id: OperationId,
        day: RacingDay,
        race: ProgrammeRaceCandidate,
        programme_checksum: ArtifactChecksum,
        at: datetime,
        reconcile_to: RaceId | None = None,
    ) -> RaceId: ...

    def record_identity_decision(
        self,
        operation_id: OperationId,
        *,
        source: str,
        source_alias: str,
        normalized_name: str,
        tier: IdentityTier,
        dog_id: DogId | None,
        reason: str,
        at: datetime,
    ) -> bool: ...

    def ingest_run(self, observation: RunObservation, *, authoritative: bool) -> bool: ...

    def record_odds_attempt(self, observation: OddsObservation) -> bool: ...

    def record_field_evidence(self, evidence: FieldEvidence) -> bool: ...

    def field_evidence(self, race_id: RaceId) -> tuple[FieldEvidence, ...]: ...

    def record_collection_quarantine(
        self,
        operation_id: OperationId,
        race_id: RaceId,
        *,
        stage: str,
        code: str,
        details: str,
        at: datetime,
        operation_kind: str = "quarantine_collection",
        request_intent_digest: ArtifactChecksum | None = None,
    ) -> bool: ...

    def odds_attempts(self, race_id: RaceId) -> tuple[OddsAttemptRecord, ...]: ...

    def seal_evidence(
        self,
        operation_id: OperationId,
        *,
        race_id: RaceId,
        raw_checksum: ArtifactChecksum,
        normalized_checksum: ArtifactChecksum,
        schema_version: str,
        normalization_version: str,
        frozen_at: datetime,
        freeze_authority: FreezeAuthority,
        odds_checksum: ArtifactChecksum,
        sealed_at: datetime,
        request_intent_digest: ArtifactChecksum,
    ) -> bool: ...

    def collection_rows(self, day: RacingDay) -> tuple[CollectionRaceRecord, ...]: ...

    def expected_programme_artifact(self, race_id: RaceId) -> ExpectedProgrammeArtifact: ...

    def is_collection_quarantined(self, race_id: RaceId) -> bool: ...

    def collection_quarantine(self, race_id: RaceId) -> CollectionQuarantine | None: ...

    def committed_seal(
        self,
        operation_id: OperationId,
        race_id: RaceId,
    ) -> CommittedSeal | None: ...

    def quarantined_seal(
        self, operation_id: OperationId, race_id: RaceId
    ) -> QuarantinedSeal | None: ...


def iso_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("stored timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("sha256_text requires text")
    return hashlib.sha256(value.encode()).hexdigest()


class SQLiteOperationsStore:
    """SQLite implementation; each public mutation is one immediate transaction."""

    def __init__(self, path: str | Path, *, timeout: float = 5.0):
        self.path = Path(path)
        self.timeout = timeout

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.timeout,
            isolation_level=None,
            factory=_ClosingConnection,
        )
        connection.row_factory = sqlite3.Row
        connection.create_function("sha256_text", 1, _sha256_text, deterministic=True)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _migration_scripts(self) -> Sequence[tuple[int, str, bytes]]:
        migration_root = files("race_collection.migrations")
        names = (
            (1, "0001_operations.sql"),
            (2, "0002_collection_and_sealing.sql"),
            (3, "0003_identity_aliases.sql"),
            (4, "0004_internal_identity_and_provenance.sql"),
            (5, "0005_exact_checksum_contracts.sql"),
            (6, "0006_authoritative_field_evidence.sql"),
            (7, "0007_append_only_authority.sql"),
            (8, "0008_seal_request_intent.sql"),
            (9, "0009_quarantined_seal_request_intent.sql"),
            (10, "0010_deferred_forecasting.sql"),
            (11, "0011_canonical_model_bundle.sql"),
            (12, "0012_phase5_training_corpus.sql"),
            (13, "0013_phase6_evaluation_promotion.sql"),
            (14, "0014_phase6_root_acceptance.sql"),
            (15, "0015_phase6_review_closure.sql"),
            (16, "0016_phase6_temporal_service_authority.sql"),
            (17, "0017_phase6_round2_acceptance.sql"),
            (18, "0018_phase7_operational_authority.sql"),
            (19, "0019_phase7_acceptance_hardening.sql"),
            (20, "0020_phase7_root_audit.sql"),
            (21, "0021_phase7_command_receipts.sql"),
            (22, "0022_phase7_backup_reference_contract.sql"),
            (23, "0023_phase7_barrier_rejections.sql"),
            (24, "0024_phase7_probation_generation.sql"),
            (25, "0025_phase7_admin_pause_enforcement.sql"),
            (26, "0026_phase7_schedule_cutover.sql"),
            (27, "0027_phase7_atomic_result_rejection.sql"),
            (28, "0028_phase7_probation_seal_authority.sql"),
            (29, "0029_phase7_bounded_timing_authority.sql"),
            (30, "0030_forward_baseline_cohort_authority.sql"),
        )
        return tuple(
            (version, name, migration_root.joinpath(name).read_bytes()) for version, name in names
        )

    @staticmethod
    def _execute_migration(connection: sqlite3.Connection, script: str) -> None:
        """Execute statements without sqlite3.executescript's implicit COMMIT."""
        statement = ""
        for line in script.splitlines(keepends=True):
            statement += line
            if sqlite3.complete_statement(statement):
                connection.execute(statement)
                statement = ""
        if statement.strip():
            raise OperationsStoreError("migration ends with an incomplete SQL statement")

    def migrate(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS schema_migrations ("
                    "version INTEGER PRIMARY KEY, checksum TEXT NOT NULL, applied_at TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS operations ("
                    "operation_id TEXT PRIMARY KEY, kind TEXT NOT NULL, "
                    "payload_sha256 TEXT NOT NULL, "
                    "created_at TEXT NOT NULL)"
                )
                for version, name, migration in self._migration_scripts():
                    checksum = hashlib.sha256(migration).hexdigest()
                    existing = connection.execute(
                        "SELECT checksum FROM schema_migrations WHERE version = ?", (version,)
                    ).fetchone()
                    if existing is None:
                        self._execute_migration(connection, migration.decode())
                        connection.execute(
                            "INSERT INTO schema_migrations(version, checksum, applied_at) "
                            "VALUES(?, ?, ?)",
                            (version, checksum, iso_timestamp(datetime.now(timezone.utc))),
                        )
                    elif existing["checksum"] != checksum:
                        raise OperationsStoreError(
                            f"migration {version} checksum differs from applied migration"
                        )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    @contextmanager
    def _operation(
        self,
        operation_id: OperationId,
        kind: str,
        payload: Mapping[str, Any] | Callable[[sqlite3.Connection], Mapping[str, Any]],
        *,
        connection: sqlite3.Connection | None = None,
    ) -> Iterator[tuple[sqlite3.Connection, bool]]:
        owns_connection = connection is None
        connection = self._connect() if connection is None else connection
        savepoint = f"operation_{uuid.uuid4().hex}"
        if owns_connection:
            connection.execute("BEGIN IMMEDIATE")
        else:
            if not connection.in_transaction:
                raise OperationsStoreError(
                    "caller-supplied operation connection must have an active transaction"
                )
            connection.execute(f"SAVEPOINT {savepoint}")
        try:
            resolved_payload = payload(connection) if callable(payload) else payload
            digest = _payload_hash(resolved_payload)
            existing = connection.execute(
                "SELECT kind, payload_sha256 FROM operations WHERE operation_id = ?",
                (str(operation_id),),
            ).fetchone()
            if existing is not None:
                if existing["kind"] != kind or existing["payload_sha256"] != digest:
                    raise ConflictingOperation(f"operation {operation_id} has different intent")
                yield connection, True
                if owns_connection:
                    connection.rollback()
                else:
                    connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                return
            connection.execute(
                "INSERT INTO operations(operation_id, kind, payload_sha256, created_at) "
                "VALUES (?, ?, ?, ?)",
                (str(operation_id), kind, digest, iso_timestamp(datetime.now(timezone.utc))),
            )
            yield connection, False
            if owns_connection:
                connection.commit()
            else:
                connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        except Exception:
            if owns_connection:
                connection.rollback()
            else:
                connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise
        finally:
            if owns_connection:
                connection.close()

    def create_racing_day(self, operation_id: OperationId, day: RacingDay) -> bool:
        payload = {
            "id": str(day.id),
            "local_date": day.local_date.isoformat(),
            "timezone": day.timezone,
            "opened_at": iso_timestamp(day.opened_at),
        }
        with self._operation(operation_id, "create_racing_day", payload) as (db, replay):
            if replay:
                return False
            db.execute(
                "INSERT INTO racing_days(racing_day_id, local_date, timezone, opened_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    str(day.id),
                    day.local_date.isoformat(),
                    day.timezone,
                    iso_timestamp(day.opened_at),
                ),
            )
        return True

    def register_forward_baseline_cohort(
        self,
        operation_id: OperationId,
        *,
        cohort_id: str,
        artifact_checksum: ArtifactChecksum,
        frozen_at: datetime,
        members: Sequence[Mapping[str, Any]],
        registered_at: datetime,
    ) -> bool:
        """Bind one immutable cohort artifact and its exact population transactionally."""
        frozen_text = iso_timestamp(frozen_at)
        registered_text = iso_timestamp(registered_at)
        if type(cohort_id) is not str or not cohort_id.strip() or len(members) != 20:
            raise ValueError("forward baseline cohort binding is invalid")
        canonical_members = tuple(
            json.dumps(member, sort_keys=True, separators=(",", ":"), allow_nan=False)
            for member in members
        )
        payload = {
            "cohort_id": cohort_id,
            "artifact_checksum": str(artifact_checksum),
            "frozen_at": frozen_text,
            "members": [json.loads(member) for member in canonical_members],
            "registered_at": registered_text,
        }
        with self._operation(
            operation_id, "register_forward_baseline_cohort", payload
        ) as (db, replay):
            if replay:
                return False
            db.execute(
                "INSERT INTO forward_baseline_cohorts VALUES(?,?,?,?,?,?)",
                (
                    cohort_id,
                    str(artifact_checksum),
                    frozen_text,
                    20,
                    registered_text,
                    str(operation_id),
                ),
            )
            for ordinal, (member, canonical_member) in enumerate(
                zip(members, canonical_members, strict=True)
            ):
                db.execute(
                    "INSERT INTO forward_baseline_cohort_members VALUES(?,?,?,?,?)",
                    (
                        cohort_id,
                        ordinal,
                        member["race_id"],
                        member["source_native_race_id"],
                        canonical_member,
                    ),
                )
        return True

    def forward_baseline_cohort(self, cohort_id: str) -> Mapping[str, Any] | None:
        """Read the authoritative cohort checksum and population projection."""
        with self._connect() as db:
            cohort = db.execute(
                "SELECT * FROM forward_baseline_cohorts WHERE cohort_id=?", (cohort_id,)
            ).fetchone()
            if cohort is None:
                return None
            members = db.execute(
                "SELECT member_json FROM forward_baseline_cohort_members "
                "WHERE cohort_id=? ORDER BY member_ordinal",
                (cohort_id,),
            ).fetchall()
        if len(members) != 20:
            raise OperationsStoreError("forward baseline cohort population is incomplete")
        return {
            "cohort_id": cohort["cohort_id"],
            "artifact_checksum": cohort["artifact_checksum"],
            "frozen_at": cohort["frozen_at"],
            "race_count": cohort["race_count"],
            "members": [json.loads(row["member_json"]) for row in members],
        }

    def forward_baseline_cohort_lifecycle(
        self, cohort_id: str
    ) -> Mapping[str, Any] | None:
        """Read cohort authority and its lifecycle projection from one DB snapshot."""
        with self._connect() as db:
            db.execute("BEGIN")
            cohort = db.execute(
                "SELECT * FROM forward_baseline_cohorts WHERE cohort_id=?", (cohort_id,)
            ).fetchone()
            if cohort is None:
                return None
            member_rows = db.execute(
                "SELECT member_json FROM forward_baseline_cohort_members "
                "WHERE cohort_id=? ORDER BY member_ordinal",
                (cohort_id,),
            ).fetchall()
            if len(member_rows) != 20:
                raise OperationsStoreError(
                    "forward baseline cohort population is incomplete"
                )
            members = [json.loads(row["member_json"]) for row in member_rows]
            races: dict[str, Mapping[str, Any]] = {}
            for member in members:
                race_id = member["race_id"]
                lifecycle = db.execute(
                    "SELECT r.state,d.local_date,e.source_race_id,e.venue,e.race_number,"
                    "e.scheduled_jump,s.frozen_at,p.prediction_id,p.artifact_checksum,p.computed_at,"
                    "q.prediction_id quarantine_prediction_id,q.code prediction_code,"
                    "q.details prediction_details,q.quarantined_at "
                    "FROM races r JOIN racing_days d USING(racing_day_id) "
                    "JOIN expected_races e USING(race_id) "
                    "LEFT JOIN sealed_evidence s USING(race_id) "
                    "LEFT JOIN deferred_predictions p USING(race_id) "
                    "LEFT JOIN prediction_quarantines q USING(race_id) "
                    "WHERE r.race_id=?",
                    (race_id,),
                ).fetchone()
                collection_rejections = db.execute(
                    "SELECT stage,code,details,created_at FROM collection_quarantines "
                    "WHERE race_id=? ORDER BY quarantine_id",
                    (race_id,),
                ).fetchall()
                races[race_id] = {
                    "lifecycle": dict(lifecycle) if lifecycle is not None else None,
                    "collection_rejections": [dict(row) for row in collection_rejections],
                }
            db.commit()
        return {
            "cohort": {
                "cohort_id": cohort["cohort_id"],
                "artifact_checksum": cohort["artifact_checksum"],
                "frozen_at": cohort["frozen_at"],
                "race_count": cohort["race_count"],
                "members": members,
            },
            "races": races,
        }

    def discover_race(
        self, operation_id: OperationId, race_id: RaceId, day: RacingDay, at: datetime
    ) -> bool:
        payload = {"race_id": str(race_id), "racing_day_id": str(day.id), "at": iso_timestamp(at)}
        with self._operation(operation_id, "discover_race", payload) as (db, replay):
            if replay:
                return False
            db.execute(
                "INSERT INTO races(race_id, racing_day_id, state, discovered_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    str(race_id),
                    str(day.id),
                    RaceState.DISCOVERED.value,
                    iso_timestamp(at),
                    iso_timestamp(at),
                ),
            )
            db.execute(
                "INSERT INTO lifecycle_events(race_id, prior_state, target_state, "
                "occurred_at, operation_id) "
                "VALUES (?, NULL, ?, ?, ?)",
                (str(race_id), RaceState.DISCOVERED.value, iso_timestamp(at), str(operation_id)),
            )
        return True

    def add_race_alias(
        self,
        operation_id: OperationId,
        race_id: RaceId,
        source: str,
        alias: str,
        at: datetime,
    ) -> bool:
        payload = {
            "race_id": str(race_id),
            "source": source,
            "alias": alias,
            "at": iso_timestamp(at),
        }
        with self._operation(operation_id, "add_race_alias", payload) as (db, replay):
            if replay:
                return False
            db.execute(
                "INSERT INTO race_aliases(race_id, source, alias, created_at) VALUES (?, ?, ?, ?)",
                (str(race_id), source, alias, iso_timestamp(at)),
            )
        return True

    def close_racing_day(self, operation_id: OperationId, day: RacingDay, at: datetime) -> bool:
        payload = {"racing_day_id": str(day.id), "at": iso_timestamp(at)}
        with self._operation(operation_id, "close_racing_day", payload) as (db, replay):
            if replay:
                return False
            states = db.execute(
                "SELECT r.state, EXISTS(SELECT 1 FROM collection_quarantines q "
                "WHERE q.race_id=r.race_id) collection_quarantined "
                "FROM expected_races e JOIN races r ON r.race_id=e.race_id "
                "WHERE r.racing_day_id = ?",
                (str(day.id),),
            ).fetchall()
            if not states or any(
                row["state"] != RaceState.AWAITING_DAY_CLOSE.value
                and not row["collection_quarantined"]
                for row in states
            ):
                raise BarrierNotSatisfied(
                    "all expected races must await day close or be collection-quarantined"
                )
            db.execute(
                "UPDATE racing_days SET closed_at = ? "
                "WHERE racing_day_id = ? AND closed_at IS NULL",
                (iso_timestamp(at), str(day.id)),
            )
        return True

    def advance_race(
        self, operation_id: OperationId, race_id: RaceId, target: RaceState, at: datetime
    ) -> bool:
        payload = {"race_id": str(race_id), "target": target.value, "at": iso_timestamp(at)}
        with self._operation(operation_id, "advance_race", payload) as (db, replay):
            if replay:
                return False
            row = db.execute(
                "SELECT state, racing_day_id FROM races WHERE race_id = ?", (str(race_id),)
            ).fetchone()
            if row is None:
                raise OperationsStoreError(f"unknown race {race_id}")
            current = RaceState(row["state"])
            RaceLifecycle.validate(current, target)
            if target == RaceState.PREDICTION_PENDING:
                eligible = db.execute(
                    "SELECT EXISTS(SELECT 1 FROM expected_races e WHERE e.race_id=?) "
                    "AND NOT EXISTS(SELECT 1 FROM collection_quarantines q WHERE q.race_id=?)",
                    (str(race_id), str(race_id)),
                ).fetchone()[0]
                if not eligible:
                    raise BarrierNotSatisfied(
                        "only unquarantined expected races may enter prediction"
                    )
                closed = db.execute(
                    "SELECT closed_at FROM racing_days WHERE racing_day_id = ?",
                    (row["racing_day_id"],),
                ).fetchone()
                if closed is None or closed["closed_at"] is None:
                    raise BarrierNotSatisfied("racing day must close before prediction")
            if target == RaceState.RESULT_PENDING:
                incomplete = db.execute(
                    "SELECT COUNT(*) AS count FROM expected_races e "
                    "JOIN races r ON r.race_id=e.race_id WHERE r.racing_day_id = ? "
                    "AND NOT EXISTS(SELECT 1 FROM collection_quarantines q "
                    "WHERE q.race_id=r.race_id) "
                    "AND r.state NOT IN (?, ?, ?, ?, ?, ?, ?)",
                    (
                        row["racing_day_id"],
                        RaceState.PREDICTION_COMMITTED.value,
                        RaceState.PREDICTION_QUARANTINED.value,
                        RaceState.RESULT_PENDING.value,
                        RaceState.RESULT_COLLECTED.value,
                        RaceState.RESULT_QUARANTINED.value,
                        RaceState.TRAINING_EXAMPLE_READY.value,
                        RaceState.EVALUATION_INELIGIBLE.value,
                    ),
                ).fetchone()["count"]
                if incomplete:
                    raise BarrierNotSatisfied(
                        "prediction batch must commit or quarantine before result collection"
                    )
            db.execute(
                "UPDATE races SET state = ?, updated_at = ? WHERE race_id = ?",
                (target.value, iso_timestamp(at), str(race_id)),
            )
            db.execute(
                "INSERT INTO lifecycle_events(race_id, prior_state, target_state, "
                "occurred_at, operation_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(race_id), current.value, target.value, iso_timestamp(at), str(operation_id)),
            )
        return True

    def add_dog(self, operation_id: OperationId, dog_id: DogId, at: datetime) -> bool:
        payload = {"dog_id": str(dog_id), "at": iso_timestamp(at)}
        with self._operation(operation_id, "add_dog", payload) as (db, replay):
            if replay:
                return False
            db.execute(
                "INSERT INTO dogs(dog_id, created_at) VALUES (?, ?)",
                (str(dog_id), iso_timestamp(at)),
            )
        return True

    def add_dog_run(self, operation_id: OperationId, run: DogRun) -> bool:
        payload = {
            "dog_id": str(run.dog_id),
            "local_racing_date": run.local_racing_date.isoformat(),
            "authoritative": run.authoritative,
            "created_at": iso_timestamp(run.created_at),
        }
        with self._operation(operation_id, "add_dog_run", payload) as (db, replay):
            if replay:
                return False
            db.execute(
                "INSERT INTO dog_runs(dog_id, local_racing_date, authoritative, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    str(run.dog_id),
                    run.local_racing_date.isoformat(),
                    int(run.authoritative),
                    iso_timestamp(run.created_at),
                ),
            )
        return True

    def add_run_observation(self, observation: RunObservation) -> bool:
        payload = {
            "dog_id": str(observation.dog_id),
            "local_racing_date": observation.local_racing_date.isoformat(),
            "source": observation.source,
            "artifact_checksum": str(observation.artifact_checksum),
            "observed_at": iso_timestamp(observation.observed_at),
            "starts": observation.starts,
            "wins": observation.wins,
        }
        with self._operation(observation.operation_id, "add_run_observation", payload) as (
            db,
            replay,
        ):
            if replay:
                return False
            run = db.execute(
                "SELECT dog_run_id FROM dog_runs WHERE dog_id = ? AND local_racing_date = ?",
                (str(observation.dog_id), observation.local_racing_date.isoformat()),
            ).fetchone()
            if run is None:
                raise OperationsStoreError("run observation requires an existing Dog Run")
            db.execute(
                "INSERT INTO run_observations(dog_run_id, source, artifact_checksum, observed_at, "
                "starts, wins, operation_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run["dog_run_id"],
                    observation.source,
                    str(observation.artifact_checksum),
                    iso_timestamp(observation.observed_at),
                    observation.starts,
                    observation.wins,
                    str(observation.operation_id),
                ),
            )
        return True

    def record_quarantine(self, quarantine: Quarantine) -> bool:
        payload = {
            "race_id": str(quarantine.race_id),
            "stage": quarantine.stage,
            "reason": quarantine.reason,
            "created_at": iso_timestamp(quarantine.created_at),
        }
        with self._operation(quarantine.operation_id, "record_quarantine", payload) as (
            db,
            replay,
        ):
            if replay:
                return False
            row = db.execute(
                "SELECT state FROM races WHERE race_id = ?", (str(quarantine.race_id),)
            ).fetchone()
            if row is None:
                raise OperationsStoreError(f"unknown race {quarantine.race_id}")
            current = RaceState(row["state"])
            targets = {
                "prediction": RaceState.PREDICTION_QUARANTINED,
                "result": RaceState.RESULT_QUARANTINED,
            }
            try:
                target = targets[quarantine.stage]
            except KeyError as error:
                raise OperationsStoreError(
                    "Phase 1 quarantine stage must be 'prediction' or 'result'"
                ) from error
            RaceLifecycle.validate(current, target)
            db.execute(
                "INSERT INTO quarantines(race_id, stage, reason, created_at, operation_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    str(quarantine.race_id),
                    quarantine.stage,
                    quarantine.reason,
                    iso_timestamp(quarantine.created_at),
                    str(quarantine.operation_id),
                ),
            )
            db.execute(
                "UPDATE races SET state = ?, updated_at = ? WHERE race_id = ?",
                (target.value, iso_timestamp(quarantine.created_at), str(quarantine.race_id)),
            )
            db.execute(
                "INSERT INTO lifecycle_events(race_id, prior_state, target_state, occurred_at, "
                "operation_id) VALUES (?, ?, ?, ?, ?)",
                (
                    str(quarantine.race_id),
                    current.value,
                    target.value,
                    iso_timestamp(quarantine.created_at),
                    str(quarantine.operation_id),
                ),
            )
        return True

    def record_supersession(self, entity_type: str, item: Supersession[Any]) -> bool:
        payload = {
            "entity_type": entity_type,
            "prior_id": str(item.prior_id),
            "replacement_id": str(item.replacement_id),
            "reason": item.reason,
            "created_at": iso_timestamp(item.created_at),
        }
        with self._operation(item.operation_id, "record_supersession", payload) as (db, replay):
            if replay:
                return False
            db.execute(
                "INSERT INTO supersessions(entity_type, prior_id, replacement_id, reason, "
                "created_at, operation_id) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    entity_type,
                    str(item.prior_id),
                    str(item.replacement_id),
                    item.reason,
                    iso_timestamp(item.created_at),
                    str(item.operation_id),
                ),
            )
        return True

    def record_expected_race(
        self,
        operation_id: OperationId,
        day: RacingDay,
        race: ProgrammeRaceCandidate,
        programme_checksum: ArtifactChecksum,
        at: datetime,
        reconcile_to: RaceId | None = None,
    ) -> RaceId:
        payload = {
            "day": str(day.id),
            "source": race.source,
            "source_race_id": race.source_race_id,
            "venue": race.venue,
            "race_number": race.race_number,
            "scheduled_jump": iso_timestamp(race.scheduled_jump),
            "programme_checksum": str(programme_checksum),
            "at": iso_timestamp(at),
            "reconcile_to": str(reconcile_to) if reconcile_to else None,
        }
        collision_owner: RaceId | None = None
        collision_target: RaceId | None = None
        inventory_conflict: tuple[RaceId, str] | None = None
        with self._operation(operation_id, "record_expected_race", payload) as (db, replay):
            if replay:
                row = db.execute(
                    "SELECT race_id, collision FROM programme_race_observations "
                    "WHERE operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                target = row["race_id"]
                if row["collision"]:
                    owner = db.execute(
                        "SELECT race_id FROM race_aliases WHERE source=? AND alias=?",
                        (race.source, race.source_race_id),
                    ).fetchone()
                    collision_owner = RaceId(owner["race_id"])
                    collision_target = RaceId(target)
                quarantine = db.execute(
                    "SELECT code, details FROM collection_quarantines WHERE operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                if quarantine is not None and quarantine["code"] == "expected_inventory_conflict":
                    inventory_conflict = (RaceId(target), quarantine["details"])
            else:
                owner = db.execute(
                    "SELECT race_id FROM race_aliases WHERE source=? AND alias=?",
                    (race.source, race.source_race_id),
                ).fetchone()
                if reconcile_to is not None:
                    target = str(reconcile_to)
                    exists = db.execute(
                        "SELECT racing_day_id FROM races WHERE race_id=?", (target,)
                    ).fetchone()
                    if exists is None or exists["racing_day_id"] != str(day.id):
                        raise OperationsStoreError(
                            "explicit reconciliation target is not in racing day"
                        )
                elif owner is not None:
                    target = owner["race_id"]
                else:
                    target = f"race_{uuid.uuid4().hex}"
                    db.execute(
                        "INSERT INTO races(race_id, racing_day_id, state, discovered_at, "
                        "updated_at) VALUES (?, ?, ?, ?, ?)",
                        (
                            target,
                            str(day.id),
                            RaceState.DISCOVERED.value,
                            iso_timestamp(at),
                            iso_timestamp(at),
                        ),
                    )
                    db.execute(
                        "INSERT INTO lifecycle_events(race_id, prior_state, target_state, "
                        "occurred_at, operation_id) VALUES (?, NULL, ?, ?, ?)",
                        (target, RaceState.DISCOVERED.value, iso_timestamp(at), str(operation_id)),
                    )
                target_day = db.execute(
                    "SELECT racing_day_id FROM races WHERE race_id=?", (target,)
                ).fetchone()
                if target_day is None or target_day["racing_day_id"] != str(day.id):
                    raise OperationsStoreError("race alias target is not in supplied racing day")
                collision = owner is not None and owner["race_id"] != target
                expected = db.execute(
                    "SELECT venue, race_number, scheduled_jump, programme_checksum "
                    "FROM expected_races WHERE race_id=?",
                    (target,),
                ).fetchone()
                if expected is None and not collision:
                    db.execute(
                        "INSERT INTO expected_races(race_id, source, source_race_id, venue, "
                        "race_number, scheduled_jump, programme_checksum, operation_id) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            target,
                            race.source,
                            race.source_race_id,
                            race.venue,
                            race.race_number,
                            iso_timestamp(race.scheduled_jump),
                            str(programme_checksum),
                            str(operation_id),
                        ),
                    )
                conflict_details: str | None = None
                if expected is not None and not collision:
                    persisted = {
                        "venue": expected["venue"],
                        "race_number": expected["race_number"],
                        "scheduled_jump": expected["scheduled_jump"],
                        "programme_checksum": expected["programme_checksum"],
                    }
                    incoming = {
                        "venue": race.venue,
                        "race_number": race.race_number,
                        "scheduled_jump": iso_timestamp(race.scheduled_jump),
                        "programme_checksum": str(programme_checksum),
                    }
                    differences = [
                        f"{name}: persisted={persisted[name]!r}, incoming={incoming[name]!r}"
                        for name in persisted
                        if persisted[name] != incoming[name]
                    ]
                    if differences:
                        conflict_details = "; ".join(differences)
                if owner is None:
                    db.execute(
                        "INSERT INTO race_aliases(race_id, source, alias, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (target, race.source, race.source_race_id, iso_timestamp(at)),
                    )
                db.execute(
                    "INSERT INTO programme_race_observations(race_id, source, source_race_id, "
                    "venue, race_number, scheduled_jump, programme_checksum, observed_at, "
                    "collision, operation_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        target,
                        race.source,
                        race.source_race_id,
                        race.venue,
                        race.race_number,
                        iso_timestamp(race.scheduled_jump),
                        str(programme_checksum),
                        iso_timestamp(at),
                        int(collision),
                        str(operation_id),
                    ),
                )
                if collision:
                    db.execute(
                        "INSERT INTO collection_quarantines(race_id, stage, code, details, "
                        "created_at, operation_id) VALUES (?, 'identity', "
                        "'race_alias_owner_collision', ?, ?, ?)",
                        (
                            target,
                            f"alias remains owned by {owner['race_id']}",
                            iso_timestamp(at),
                            str(operation_id),
                        ),
                    )
                    collision_owner = RaceId(owner["race_id"])
                    collision_target = RaceId(target)
                elif conflict_details is not None:
                    db.execute(
                        "INSERT INTO collection_quarantines(race_id, stage, code, details, "
                        "created_at, operation_id) VALUES (?, 'collection', "
                        "'expected_inventory_conflict', ?, ?, ?)",
                        (target, conflict_details, iso_timestamp(at), str(operation_id)),
                    )
                    inventory_conflict = (RaceId(target), conflict_details)
        if collision_owner is not None and collision_target is not None:
            raise RaceAliasOwnerCollision(
                race.source, race.source_race_id, collision_owner, collision_target
            )
        if inventory_conflict is not None:
            raise ExpectedInventoryConflict(*inventory_conflict)
        return RaceId(target)

    def record_identity_decision(
        self,
        operation_id: OperationId,
        *,
        source: str,
        source_alias: str,
        normalized_name: str,
        tier: IdentityTier,
        dog_id: DogId | None,
        reason: str,
        at: datetime,
    ) -> bool:
        for name, value in (
            ("source", source),
            ("source_alias", source_alias),
            ("normalized_name", normalized_name),
            ("reason", reason),
        ):
            if not isinstance(value, str) or not value.strip():
                raise OperationsStoreError(f"{name} must be nonblank text")
        if not isinstance(tier, IdentityTier):
            raise OperationsStoreError("tier must be an IdentityTier")
        if dog_id is not None and not isinstance(dog_id, DogId):
            raise OperationsStoreError("dog_id must be a DogId or None")
        if tier is IdentityTier.AMBIGUOUS:
            if dog_id is not None:
                raise OperationsStoreError("AMBIGUOUS requires dog_id is None")
        elif dog_id is None:
            raise OperationsStoreError(f"{tier.name} requires a real DogId")
        payload = {
            "source": source,
            "alias": source_alias,
            "normalized_name": normalized_name,
            "tier": tier.value,
            "dog_id": str(dog_id) if dog_id else None,
            "reason": reason,
            "at": iso_timestamp(at),
        }
        with self._operation(operation_id, "record_identity_decision", payload) as (db, replay):
            if replay:
                return False
            alias_conflict = False
            if dog_id is not None:
                db.execute(
                    "INSERT OR IGNORE INTO dogs(dog_id, created_at) VALUES (?, ?)",
                    (str(dog_id), iso_timestamp(at)),
                )
                owner = db.execute(
                    "SELECT dog_id FROM dog_aliases WHERE source=? AND alias=?",
                    (source, source_alias),
                ).fetchone()
                alias_conflict = owner is not None and owner["dog_id"] != str(dog_id)
                if owner is None:
                    db.execute(
                        "INSERT INTO dog_aliases(dog_id, source, alias, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (str(dog_id), source, source_alias, iso_timestamp(at)),
                    )
            db.execute(
                "INSERT INTO dog_identity_decisions(source, source_alias, normalized_name, tier, "
                "dog_id, reason, decided_at, operation_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    source,
                    source_alias,
                    normalized_name,
                    tier.value,
                    payload["dog_id"],
                    reason,
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
            if alias_conflict:
                db.execute(
                    "INSERT INTO identity_quarantines(source, source_alias, normalized_name, "
                    "reason, created_at, operation_id) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        source,
                        source_alias,
                        normalized_name,
                        "source alias already belongs to a different DogId",
                        iso_timestamp(at),
                        str(operation_id),
                    ),
                )
            elif tier is IdentityTier.AMBIGUOUS:
                db.execute(
                    "INSERT INTO identity_quarantines(source, source_alias, normalized_name, "
                    "reason, created_at, operation_id) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        source,
                        source_alias,
                        normalized_name,
                        reason,
                        iso_timestamp(at),
                        str(operation_id),
                    ),
                )
            elif tier is IdentityTier.AUTHORITATIVE and dog_id is not None:
                candidates = db.execute(
                    "SELECT DISTINCT dog_id FROM dog_identity_decisions WHERE normalized_name=? "
                    "AND source=? AND tier='high_confidence_provisional' AND dog_id<>?",
                    (normalized_name, source, str(dog_id)),
                ).fetchall()
                if len(candidates) == 1:
                    provisional = candidates[0]["dog_id"]
                    existing_alias = db.execute(
                        "SELECT canonical_dog_id FROM dog_identity_aliases "
                        "WHERE provisional_dog_id=?",
                        (provisional,),
                    ).fetchone()
                    if existing_alias is None:
                        self._upgrade_dog_identity(db, provisional, str(dog_id), at, operation_id)
                    elif existing_alias["canonical_dog_id"] != str(dog_id):
                        db.execute(
                            "INSERT INTO identity_quarantines(source, source_alias, "
                            "normalized_name, reason, created_at, operation_id) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                source,
                                source_alias,
                                normalized_name,
                                "provisional identity already aliases a different canonical dog",
                                iso_timestamp(at),
                                str(operation_id),
                            ),
                        )
                elif len(candidates) > 1:
                    db.execute(
                        "INSERT INTO identity_quarantines(source, source_alias, normalized_name, "
                        "reason, created_at, operation_id) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            source,
                            source_alias,
                            normalized_name,
                            "multiple provisional identities match authoritative evidence",
                            iso_timestamp(at),
                            str(operation_id),
                        ),
                    )
        return True

    @staticmethod
    def _upgrade_dog_identity(
        db: sqlite3.Connection,
        provisional: str,
        canonical: str,
        at: datetime,
        operation_id: OperationId,
    ) -> None:
        existing_runs = db.execute(
            "SELECT dog_run_id, local_racing_date FROM dog_runs WHERE dog_id=?", (provisional,)
        ).fetchall()
        for provisional_run in existing_runs:
            canonical_run = db.execute(
                "SELECT dog_run_id FROM dog_runs WHERE dog_id=? AND local_racing_date=?",
                (canonical, provisional_run["local_racing_date"]),
            ).fetchone()
            if canonical_run is None:
                db.execute(
                    "UPDATE dog_runs SET dog_id=? WHERE dog_run_id=?",
                    (canonical, provisional_run["dog_run_id"]),
                )
            else:
                db.execute(
                    "UPDATE run_observations SET dog_run_id=? WHERE dog_run_id=?",
                    (canonical_run["dog_run_id"], provisional_run["dog_run_id"]),
                )
                db.execute(
                    "UPDATE dog_runs SET authoritative=max(authoritative, "
                    "(SELECT authoritative FROM dog_runs WHERE dog_run_id=?)) "
                    "WHERE dog_run_id=?",
                    (provisional_run["dog_run_id"], canonical_run["dog_run_id"]),
                )
                db.execute(
                    "DELETE FROM dog_runs WHERE dog_run_id=?", (provisional_run["dog_run_id"],)
                )
        db.execute(
            "INSERT INTO dog_identity_aliases(provisional_dog_id, canonical_dog_id, upgraded_at, "
            "operation_id) VALUES (?, ?, ?, ?)",
            (provisional, canonical, iso_timestamp(at), str(operation_id)),
        )

    def ingest_run(self, observation: RunObservation, *, authoritative: bool) -> bool:
        payload = {
            "dog_id": str(observation.dog_id),
            "date": observation.local_racing_date.isoformat(),
            "source": observation.source,
            "checksum": str(observation.artifact_checksum),
            "observed_at": iso_timestamp(observation.observed_at),
            "starts": observation.starts,
            "wins": observation.wins,
            "authoritative": authoritative,
        }
        with self._operation(observation.operation_id, "ingest_run", payload) as (db, replay):
            if replay:
                return False
            alias = db.execute(
                "SELECT canonical_dog_id FROM dog_identity_aliases " "WHERE provisional_dog_id=?",
                (str(observation.dog_id),),
            ).fetchone()
            dog_id = alias["canonical_dog_id"] if alias else str(observation.dog_id)
            db.execute(
                "INSERT OR IGNORE INTO dogs(dog_id, created_at) VALUES (?, ?)",
                (dog_id, iso_timestamp(observation.observed_at)),
            )
            db.execute(
                "INSERT OR IGNORE INTO dog_runs(dog_id, local_racing_date, authoritative, "
                "created_at) VALUES (?, ?, ?, ?)",
                (
                    dog_id,
                    observation.local_racing_date.isoformat(),
                    int(authoritative),
                    iso_timestamp(observation.observed_at),
                ),
            )
            if authoritative:
                db.execute(
                    "UPDATE dog_runs SET authoritative=1 WHERE dog_id=? AND " "local_racing_date=?",
                    (dog_id, observation.local_racing_date.isoformat()),
                )
            run_id = db.execute(
                "SELECT dog_run_id FROM dog_runs WHERE dog_id=? AND " "local_racing_date=?",
                (dog_id, observation.local_racing_date.isoformat()),
            ).fetchone()[0]
            db.execute(
                "INSERT INTO run_observations(dog_run_id, source, artifact_checksum, "
                "observed_at, starts, wins, operation_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    observation.source,
                    str(observation.artifact_checksum),
                    iso_timestamp(observation.observed_at),
                    observation.starts,
                    observation.wins,
                    str(observation.operation_id),
                ),
            )
        return True

    def record_odds_attempt(self, observation: OddsObservation) -> bool:
        payload = {
            "race": str(observation.race_id),
            "source": observation.source,
            "scheduled_due_at": iso_timestamp(observation.scheduled_due_at),
            "attempted_at": iso_timestamp(observation.attempted_at),
            "timing_policy": observation.timing_policy,
            "status": observation.status.value,
            "artifact": (
                str(observation.artifact_checksum) if observation.artifact_checksum else None
            ),
            "mapping": (
                str(observation.runner_mapping_checksum)
                if observation.runner_mapping_checksum
                else None
            ),
            "error": observation.error,
        }
        with self._operation(observation.operation_id, "record_odds_attempt", payload) as (
            db,
            replay,
        ):
            if replay:
                return False
            state = db.execute(
                "SELECT state FROM races WHERE race_id=?", (str(observation.race_id),)
            ).fetchone()
            terminal_quarantine = db.execute(
                "SELECT 1 FROM collection_quarantines WHERE race_id=? LIMIT 1",
                (str(observation.race_id),),
            ).fetchone()
            if state is None:
                raise OperationsStoreError(f"unknown race {observation.race_id}")
            if state["state"] not in {
                RaceState.DISCOVERED.value,
                RaceState.CARD_COLLECTED.value,
                RaceState.COLLECTING_ODDS.value,
            }:
                raise OperationsStoreError("odds collection has terminated for race")
            if terminal_quarantine is not None:
                raise OperationsStoreError("odds collection has terminal quarantine for race")
            db.execute(
                "INSERT INTO odds_attempts(race_id, source, scheduled_due_at, attempted_at, "
                "timing_policy, status, artifact_checksum, runner_mapping_checksum, error, "
                "operation_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(observation.race_id),
                    observation.source,
                    payload["scheduled_due_at"],
                    iso_timestamp(observation.attempted_at),
                    observation.timing_policy,
                    observation.status.value,
                    payload["artifact"],
                    payload["mapping"],
                    observation.error,
                    str(observation.operation_id),
                ),
            )
        return True

    def record_field_evidence(self, evidence: FieldEvidence) -> bool:
        if not isinstance(evidence, FieldEvidence):
            raise OperationsStoreError("evidence must be FieldEvidence")
        value = (
            evidence.value.isoformat(timespec="microseconds")
            if isinstance(evidence.value, datetime)
            else evidence.value
        )
        value_json = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        payload = {
            "race": str(evidence.race_id),
            "field": evidence.field.value,
            "authority": evidence.authority.value,
            "value_json": value_json,
            "source": evidence.source,
            "artifact": str(evidence.artifact_checksum),
            "observed_at": iso_timestamp(evidence.observed_at),
        }
        with self._operation(evidence.operation_id, "record_field_evidence", payload) as (
            db,
            replay,
        ):
            if replay:
                return False
            state = db.execute(
                "SELECT state FROM races WHERE race_id=?", (str(evidence.race_id),)
            ).fetchone()
            quarantined = db.execute(
                "SELECT 1 FROM collection_quarantines WHERE race_id=? LIMIT 1",
                (str(evidence.race_id),),
            ).fetchone()
            if state is None:
                raise OperationsStoreError(f"unknown race {evidence.race_id}")
            if state["state"] not in {
                RaceState.DISCOVERED.value,
                RaceState.CARD_COLLECTED.value,
                RaceState.COLLECTING_ODDS.value,
            }:
                raise OperationsStoreError("field evidence collection has terminated for race")
            if quarantined is not None:
                raise OperationsStoreError(
                    "field evidence collection has terminal quarantine for race"
                )
            db.execute(
                "INSERT INTO field_evidence(race_id, field_name, authority, value_json, "
                "artifact_checksum, observed_at, critical, operation_id, source) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    str(evidence.race_id),
                    evidence.field.value,
                    evidence.authority.value,
                    value_json,
                    str(evidence.artifact_checksum),
                    iso_timestamp(evidence.observed_at),
                    int(evidence.field.critical),
                    str(evidence.operation_id),
                    evidence.source,
                ),
            )
        return True

    def field_evidence(self, race_id: RaceId) -> tuple[FieldEvidence, ...]:
        if not isinstance(race_id, RaceId):
            raise OperationsStoreError("race_id must be a RaceId")
        with self._connect() as db:
            rows = db.execute(
                "SELECT operation_id, field_name, authority, value_json, source, "
                "artifact_checksum, observed_at, critical FROM field_evidence "
                "WHERE race_id=? ORDER BY observed_at, evidence_id",
                (str(race_id),),
            ).fetchall()
        try:
            result = []
            for row in rows:
                field = EvidenceField(row["field_name"])
                if bool(row["critical"]) != field.critical:
                    raise DomainValidationError(
                        "persisted field criticality disagrees with registry"
                    )
                value = json.loads(row["value_json"])
                if field in {
                    EvidenceField.SCHEDULED_JUMP,
                    EvidenceField.ACTUAL_JUMP,
                    EvidenceField.JUMP_TIME,
                }:
                    value = datetime.fromisoformat(value)
                result.append(
                    FieldEvidence(
                        OperationId(row["operation_id"]),
                        race_id,
                        field,
                        EvidenceAuthority(row["authority"]),
                        value,
                        row["source"],
                        ArtifactChecksum(row["artifact_checksum"]),
                        datetime.fromisoformat(row["observed_at"]),
                    )
                )
            return tuple(result)
        except (DomainValidationError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise OperationsStoreError(
                f"invalid field evidence for race {race_id}: {error}"
            ) from error

    def record_collection_quarantine(
        self,
        operation_id: OperationId,
        race_id: RaceId,
        *,
        stage: str,
        code: str,
        details: str,
        at: datetime,
        operation_kind: str = "quarantine_collection",
        request_intent_digest: ArtifactChecksum | None = None,
    ) -> bool:
        if operation_kind == "quarantine_sealing":
            if not isinstance(request_intent_digest, ArtifactChecksum):
                raise OperationsStoreError(
                    "quarantine_sealing requires an ArtifactChecksum request intent"
                )
            if stage != "sealing":
                raise OperationsStoreError("quarantine_sealing requires the sealing stage")
        elif request_intent_digest is not None:
            raise OperationsStoreError(
                "non-sealing collection quarantine cannot have a sealing request intent"
            )
        payload = {
            "race": str(race_id),
            "stage": stage,
            "code": code,
            "details": details,
            "at": iso_timestamp(at),
        }
        if request_intent_digest is not None:
            payload["request_intent"] = str(request_intent_digest)
        with self._operation(operation_id, operation_kind, payload) as (db, replay):
            if replay:
                return False
            db.execute(
                "INSERT INTO collection_quarantines(race_id, stage, code, details, "
                "created_at, operation_id, request_intent_digest) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(race_id),
                    stage,
                    code,
                    details,
                    iso_timestamp(at),
                    str(operation_id),
                    str(request_intent_digest) if request_intent_digest is not None else None,
                ),
            )
        return True

    def odds_attempts(self, race_id: RaceId) -> tuple[OddsAttemptRecord, ...]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT source, scheduled_due_at, attempted_at, timing_policy, status, "
                "artifact_checksum, runner_mapping_checksum, error FROM odds_attempts "
                "WHERE race_id=? ORDER BY scheduled_due_at, attempted_at, attempt_id",
                (str(race_id),),
            ).fetchall()
        return tuple(
            OddsAttemptRecord(
                source=row["source"],
                scheduled_due_at=datetime.fromisoformat(row["scheduled_due_at"]),
                attempted_at=datetime.fromisoformat(row["attempted_at"]),
                timing_policy=row["timing_policy"],
                status=OddsAttemptStatus(row["status"]),
                artifact_checksum=(
                    ArtifactChecksum(row["artifact_checksum"])
                    if row["artifact_checksum"] is not None
                    else None
                ),
                runner_mapping_checksum=(
                    ArtifactChecksum(row["runner_mapping_checksum"])
                    if row["runner_mapping_checksum"] is not None
                    else None
                ),
                error=row["error"],
            )
            for row in rows
        )

    def seal_evidence(
        self,
        operation_id: OperationId,
        *,
        race_id: RaceId,
        raw_checksum: ArtifactChecksum,
        normalized_checksum: ArtifactChecksum,
        schema_version: str,
        normalization_version: str,
        frozen_at: datetime,
        freeze_authority: FreezeAuthority,
        odds_checksum: ArtifactChecksum,
        sealed_at: datetime,
        request_intent_digest: ArtifactChecksum,
    ) -> bool:
        if not isinstance(freeze_authority, FreezeAuthority):
            raise OperationsStoreError("freeze_authority must be a FreezeAuthority")
        for name, checksum in (
            ("raw_checksum", raw_checksum),
            ("normalized_checksum", normalized_checksum),
            ("odds_checksum", odds_checksum),
            ("request_intent_digest", request_intent_digest),
        ):
            if not isinstance(checksum, ArtifactChecksum):
                raise OperationsStoreError(f"{name} must be an ArtifactChecksum")
        payload = {
            "race": str(race_id),
            "raw": str(raw_checksum),
            "normalized": str(normalized_checksum),
            "schema": schema_version,
            "normalization": normalization_version,
            "frozen_at": iso_timestamp(frozen_at),
            "authority": freeze_authority.value,
            "odds": str(odds_checksum),
            "sealed_at": iso_timestamp(sealed_at),
            "request_intent": str(request_intent_digest),
        }
        with self._operation(operation_id, "seal_evidence", payload) as (db, replay):
            if replay:
                return False
            quarantined = db.execute(
                "SELECT 1 FROM collection_quarantines WHERE race_id=? LIMIT 1",
                (str(race_id),),
            ).fetchone()
            if quarantined is not None:
                raise CollectionQuarantineBlocksSeal(
                    f"terminal collection quarantine blocks sealing race {race_id}"
                )
            has_field_evidence = db.execute(
                "SELECT 1 FROM field_evidence WHERE race_id=? LIMIT 1", (str(race_id),)
            ).fetchone()
            if has_field_evidence is None:
                raise OperationsStoreError("race has no transactional field evidence")
            state = db.execute(
                "SELECT state FROM races WHERE race_id=?", (str(race_id),)
            ).fetchone()
            if state is None or state["state"] != RaceState.COLLECTING_ODDS.value:
                raise OperationsStoreError("race must be collecting odds before sealing")
            db.execute(
                "INSERT INTO sealed_evidence(race_id, raw_manifest_checksum, "
                "normalized_checksum, schema_version, normalization_version, frozen_at, "
                "freeze_authority, odds_checksum, sealed_at, operation_id, "
                "request_intent_digest) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(race_id),
                    str(raw_checksum),
                    str(normalized_checksum),
                    schema_version,
                    normalization_version,
                    iso_timestamp(frozen_at),
                    freeze_authority.value,
                    str(odds_checksum),
                    iso_timestamp(sealed_at),
                    str(operation_id),
                    str(request_intent_digest),
                ),
            )
            db.execute(
                "UPDATE races SET state=?, updated_at=? WHERE race_id=?",
                (RaceState.EVIDENCE_SEALED.value, iso_timestamp(sealed_at), str(race_id)),
            )
            db.execute(
                "INSERT INTO lifecycle_events(race_id, prior_state, target_state, "
                "occurred_at, operation_id) VALUES (?, ?, ?, ?, ?)",
                (
                    str(race_id),
                    RaceState.COLLECTING_ODDS.value,
                    RaceState.EVIDENCE_SEALED.value,
                    iso_timestamp(sealed_at),
                    str(operation_id),
                ),
            )
        return True

    def is_collection_quarantined(self, race_id: RaceId) -> bool:
        return self.collection_quarantine(race_id) is not None

    def collection_quarantine(self, race_id: RaceId) -> CollectionQuarantine | None:
        if not isinstance(race_id, RaceId):
            raise OperationsStoreError("race_id must be a RaceId")
        with self._connect() as db:
            row = db.execute(
                "SELECT stage, code, details FROM collection_quarantines "
                "WHERE race_id=? ORDER BY quarantine_id LIMIT 1",
                (str(race_id),),
            ).fetchone()
        if row is None:
            return None
        try:
            return CollectionQuarantine(
                stage=row["stage"], code=row["code"], details=row["details"]
            )
        except (DomainValidationError, TypeError, ValueError) as error:
            raise OperationsStoreError(
                f"invalid collection quarantine for race {race_id}: {error}"
            ) from error

    def committed_seal(
        self,
        operation_id: OperationId,
        race_id: RaceId,
    ) -> CommittedSeal | None:
        if not isinstance(operation_id, OperationId) or not isinstance(race_id, RaceId):
            raise OperationsStoreError("seal replay lookup requires typed operation and race IDs")
        with self._connect() as db:
            operation = db.execute(
                "SELECT kind, payload_sha256 FROM operations WHERE operation_id=?",
                (str(operation_id),),
            ).fetchone()
            row = db.execute(
                "SELECT race_id, raw_manifest_checksum, normalized_checksum, schema_version, "
                "normalization_version, frozen_at, freeze_authority, odds_checksum, sealed_at, "
                "request_intent_digest "
                "FROM sealed_evidence WHERE operation_id=?",
                (str(operation_id),),
            ).fetchone()
        if row is None:
            return None
        if operation is None:
            raise OperationsStoreError(
                f"committed seal operation {operation_id} has no owning operation"
            )
        if operation["kind"] != "seal_evidence":
            raise ConflictingOperation(
                f"operation {operation_id} was replayed with different intent"
            )
        try:
            stored_race_id = RaceId(row["race_id"])
            raw_checksum = ArtifactChecksum(row["raw_manifest_checksum"])
            normalized_checksum = ArtifactChecksum(row["normalized_checksum"])
            odds_checksum = ArtifactChecksum(row["odds_checksum"])
            if not isinstance(row["schema_version"], str) or not row["schema_version"].strip():
                raise ValueError("schema_version must be nonblank text")
            if (
                not isinstance(row["normalization_version"], str)
                or not row["normalization_version"].strip()
            ):
                raise ValueError("normalization_version must be nonblank text")
            frozen_at = datetime.fromisoformat(row["frozen_at"])
            sealed_at = datetime.fromisoformat(row["sealed_at"])
            iso_timestamp(frozen_at)
            iso_timestamp(sealed_at)
            freeze_authority = FreezeAuthority(row["freeze_authority"])
            stored_digest = ArtifactChecksum(row["request_intent_digest"])
            committed = CommittedSeal(
                raw_manifest_checksum=raw_checksum,
                normalized_checksum=normalized_checksum,
                odds_checksum=odds_checksum,
                frozen_at=frozen_at,
                freeze_authority=freeze_authority,
                request_intent_digest=stored_digest,
            )
        except (DomainValidationError, TypeError, ValueError) as error:
            raise OperationsStoreError(
                f"invalid committed seal for operation {operation_id} and race {race_id}: {error}"
            ) from error
        if stored_race_id != race_id:
            raise ConflictingOperation(
                f"operation {operation_id} was replayed with different intent"
            )
        payload = {
            "race": str(stored_race_id),
            "raw": str(raw_checksum),
            "normalized": str(normalized_checksum),
            "schema": row["schema_version"],
            "normalization": row["normalization_version"],
            "frozen_at": row["frozen_at"],
            "authority": freeze_authority.value,
            "odds": str(odds_checksum),
            "sealed_at": row["sealed_at"],
            "request_intent": str(stored_digest),
        }
        if _payload_hash(payload) != operation["payload_sha256"]:
            raise OperationsStoreError(
                f"committed seal operation {operation_id} has inconsistent request intent"
            )
        return committed

    def quarantined_seal(
        self,
        operation_id: OperationId,
        race_id: RaceId,
    ) -> QuarantinedSeal | None:
        if not isinstance(operation_id, OperationId) or not isinstance(race_id, RaceId):
            raise OperationsStoreError("seal replay lookup requires typed operation and race IDs")
        with self._connect() as db:
            operation = db.execute(
                "SELECT kind, payload_sha256 FROM operations WHERE operation_id=?",
                (str(operation_id),),
            ).fetchone()
            row = db.execute(
                "SELECT race_id, stage, code, details, created_at, request_intent_digest "
                "FROM collection_quarantines WHERE operation_id=?",
                (str(operation_id),),
            ).fetchone()
            committed_row = db.execute(
                "SELECT 1 FROM sealed_evidence WHERE operation_id=?",
                (str(operation_id),),
            ).fetchone()
        if operation is None:
            if row is not None:
                raise OperationsStoreError(
                    f"quarantined sealing operation {operation_id} has no owning operation"
                )
            return None
        if operation["kind"] == "seal_evidence" and committed_row is not None:
            return None
        if operation["kind"] != "quarantine_sealing":
            raise ConflictingOperation(
                f"operation {operation_id} was replayed with different intent"
            )
        if row is None:
            raise OperationsStoreError(
                f"quarantined sealing operation {operation_id} has no quarantine row"
            )
        if row["race_id"] != str(race_id):
            raise ConflictingOperation(
                f"operation {operation_id} was replayed with different intent"
            )
        try:
            digest = ArtifactChecksum(row["request_intent_digest"])
            quarantined = QuarantinedSeal(
                stage=row["stage"],
                code=row["code"],
                details=row["details"],
                request_intent_digest=digest,
            )
        except (DomainValidationError, TypeError, ValueError) as error:
            raise OperationsStoreError(
                f"invalid quarantined seal for operation {operation_id} and race {race_id}: {error}"
            ) from error
        if quarantined.stage != "sealing":
            raise OperationsStoreError(
                f"quarantined sealing operation {operation_id} has inconsistent stage"
            )
        payload = {
            "race": row["race_id"],
            "stage": row["stage"],
            "code": row["code"],
            "details": row["details"],
            "at": row["created_at"],
            "request_intent": str(digest),
        }
        if _payload_hash(payload) != operation["payload_sha256"]:
            raise OperationsStoreError(
                f"quarantined sealing operation {operation_id} has inconsistent request intent"
            )
        return quarantined

    def collection_rows(self, day: RacingDay) -> tuple[CollectionRaceRecord, ...]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT r.race_id, r.state, EXISTS(SELECT 1 FROM collection_quarantines q "
                "WHERE q.race_id=r.race_id) quarantined FROM expected_races e "
                "JOIN races r ON r.race_id=e.race_id WHERE r.racing_day_id=? ORDER BY r.race_id",
                (str(day.id),),
            ).fetchall()
        return tuple(
            CollectionRaceRecord(
                race_id=RaceId(row["race_id"]),
                state=RaceState(row["state"]),
                quarantined=bool(row["quarantined"]),
            )
            for row in rows
        )

    def expected_programme_artifact(self, race_id: RaceId) -> ExpectedProgrammeArtifact:
        if not isinstance(race_id, RaceId):
            raise OperationsStoreError("race_id must be a RaceId")
        with self._connect() as db:
            row = db.execute(
                "SELECT e.source, e.programme_checksum, e.scheduled_jump FROM expected_races e "
                "JOIN races r ON r.race_id=e.race_id WHERE e.race_id=?",
                (str(race_id),),
            ).fetchone()
        if row is None:
            raise OperationsStoreError(f"race {race_id} has no expected programme artifact")
        try:
            return ExpectedProgrammeArtifact(
                source=row["source"],
                checksum=ArtifactChecksum(row["programme_checksum"]),
                scheduled_jump=datetime.fromisoformat(row["scheduled_jump"]),
            )
        except (DomainValidationError, TypeError, ValueError) as error:
            raise OperationsStoreError(
                f"invalid expected programme artifact for race {race_id}: {error}"
            ) from error

    def race_state(self, race_id: RaceId) -> RaceState:
        with self._connect() as db:
            row = db.execute(
                "SELECT state FROM races WHERE race_id = ?", (str(race_id),)
            ).fetchone()
        if row is None:
            raise OperationsStoreError(f"unknown race {race_id}")
        return RaceState(row["state"])

    def count(self, table: str) -> int:
        allowed = {
            "operations",
            "racing_days",
            "races",
            "race_aliases",
            "dog_runs",
            "run_observations",
            "quarantines",
            "supersessions",
            "lifecycle_events",
            "dog_identity_aliases",
            "identity_quarantines",
            "collection_quarantines",
            "sealed_evidence",
            "field_evidence",
        }
        if table not in allowed:
            raise ValueError("table is not part of the public diagnostic surface")
        with self._connect() as db:
            return int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
