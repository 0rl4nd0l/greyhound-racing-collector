import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from race_collection.artifacts import LocalArtifactStore
from race_collection.collection import CollectionRepository
from race_collection.domain import (
    ArtifactChecksum,
    CollectionRaceRecord,
    DogId,
    DomainValidationError,
    EvidenceAuthority,
    EvidenceField,
    ExpectedRace,
    FieldEvidence,
    FreezeAuthority,
    IdentityTier,
    OddsAttemptRecord,
    OddsAttemptStatus,
    OddsObservation,
    OperationId,
    ProgrammeRaceCandidate,
    RaceId,
    RaceState,
    RacingDay,
    RacingDayId,
    RunObservation,
)
from race_collection.features import derive_features
from race_collection.identity import DogIdentityDecision, resolve_dog_identity
from race_collection.operations import (
    CollectionQuarantineBlocksSeal,
    ConflictingOperation,
    ExpectedInventoryConflict,
    OperationsStoreError,
    RaceAliasOwnerCollision,
    SQLiteOperationsStore,
)
from race_collection.sealing import (
    EvidenceSealer,
    FieldObservation,
    SealingQuarantine,
    normalize_fields,
)

NOW = datetime(2026, 7, 22, 2, tzinfo=timezone.utc)


def ident(prefix, number):
    return f"{prefix}_{number:032x}"


def op(number):
    return OperationId(ident("op", number))


def database_counts(store):
    with store._connect() as db:
        tables = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return {
            table["name"]: db.execute(f'SELECT COUNT(*) FROM "{table["name"]}"').fetchone()[0]
            for table in tables
        }


def artifact_bytes(artifacts):
    return {
        path.relative_to(artifacts.root): path.read_bytes()
        for path in artifacts.root.rglob("*")
        if path.is_file()
    }


def empty_caller_effective_digest(
    race_id,
    sources,
    observations,
    scheduled_jump,
    buffer,
    schema_version,
    normalization_version,
    sealed_at,
):
    observation_values = [
        {
            "field": item.field.value,
            "value": item.value,
            "authority": item.authority.value,
            "critical": item.critical,
            "source": item.source,
            "artifact_checksum": str(item.artifact_checksum),
        }
        for item in observations
    ]
    observation_values.sort(
        key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
    )
    payload = {
        "race_id": str(race_id),
        "requested": {"sources": {}, "observations": []},
        "effective": {
            "sources": {key: str(sources[key]) for key in sorted(sources)},
            "observations": observation_values,
            "error": None,
        },
        "scheduled_jump": scheduled_jump.isoformat(timespec="microseconds"),
        "buffer": {
            "days": buffer.days,
            "seconds": buffer.seconds,
            "microseconds": buffer.microseconds,
        },
        "schema_version": schema_version,
        "normalization_version": normalization_version,
        "sealed_at": sealed_at.isoformat(timespec="microseconds"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return ArtifactChecksum(f"sha256:{hashlib.sha256(encoded).hexdigest()}")


def persist_fields(store, race_id, observations, first_operation):
    for offset, item in enumerate(observations):
        store.record_field_evidence(
            FieldEvidence(
                op(first_operation + offset),
                race_id,
                item.field,
                item.authority,
                item.value,
                item.source,
                item.artifact_checksum,
                NOW + timedelta(microseconds=offset),
            )
        )


def committed_seal_fixture(store, repository, race, artifacts, operation_number):
    store.advance_race(op(operation_number), race.race_id, RaceState.CARD_COLLECTED, NOW)
    store.advance_race(op(operation_number + 1), race.race_id, RaceState.COLLECTING_ODDS, NOW)
    odds = artifacts.put(b"authority-odds", media_type="application/json")
    mapping = artifacts.put(b"authority-mapping", media_type="application/json")
    source = artifacts.put(b"authority-source", media_type="application/json")
    repository.record_odds_attempt(
        OddsObservation(
            op(operation_number + 2),
            race.race_id,
            "market",
            NOW,
            OddsAttemptStatus.SUCCEEDED,
            odds.checksum,
            mapping.checksum,
        )
    )
    observations = [
        FieldObservation(
            EvidenceField.RUNNER_SET,
            ["A", "B"],
            EvidenceAuthority.OFFICIAL_CARD,
            True,
            "card",
            source.checksum,
        ),
        FieldObservation(
            EvidenceField.RUNNER_IDENTITY,
            {"A": "authoritative", "B": "authoritative"},
            EvidenceAuthority.OFFICIAL_CARD,
            True,
            "card",
            source.checksum,
        ),
        FieldObservation(
            EvidenceField.RUNNER_FEATURES,
            {
                "A": {"speed": 8, "form": 3},
                "B": {"speed": 2, "form": 6},
            },
            EvidenceAuthority.OFFICIAL_CARD,
            True,
            "card",
            source.checksum,
        ),
    ]
    persist_fields(store, race.race_id, observations, operation_number + 3)
    arguments = {
        "operation_id": op(operation_number + 6),
        "race_id": race.race_id,
        "source_artifacts": {"card": source.checksum},
        "observations": observations,
        "scheduled_jump": race.scheduled_jump,
        "buffer": timedelta(minutes=2),
        "schema_version": "race-evidence-v1",
        "normalization_version": "normalizer-v1",
        "sealed_at": NOW + timedelta(minutes=1),
    }
    EvidenceSealer(store, artifacts).seal(**arguments)
    return arguments


@pytest.fixture
def setup(tmp_path):
    store = SQLiteOperationsStore(tmp_path / "operations.sqlite3")
    store.migrate()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    day = RacingDay(RacingDayId(ident("day", 1)), date(2026, 7, 22), "Australia/Melbourne", NOW)
    store.create_racing_day(op(1), day)
    repository = CollectionRepository(store)
    candidate = ProgrammeRaceCandidate("official", "R-1", "Ballarat", 1, NOW + timedelta(hours=2))
    programme = artifacts.put(b"official programme", media_type="application/json")
    race_id = repository.record_expected_race(op(2), day, candidate, programme.checksum, NOW)
    race = ExpectedRace(race_id, "official", "R-1", "Ballarat", 1, NOW + timedelta(hours=2))
    return store, repository, day, race, artifacts


def test_migrations_are_forward_only_and_repeatable(setup):
    store, *_ = setup
    store.migrate()
    with sqlite3.connect(store.path) as db:
        versions = db.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
        assert versions == [
            (1,),
            (2,),
            (3,),
            (4,),
            (5,),
            (6,),
            (7,),
            (8,),
            (9,),
            (10,),
            (11,),
            (12,),
            (13,),
            (14,),
            (15,),
            (16,),
            (17,),
            (18,),
            (19,),
            (20,),
            (21,),
            (22,),
            (23,),
            (24,),
            (25,),
            (26,),
                (27,),
                (28,),
                (29,),
                (30,),
            ]
        recorded = db.execute("SELECT checksum FROM schema_migrations WHERE version=10").fetchone()[
            0
        ]
    migration = Path("race_collection/migrations/0010_deferred_forecasting.sql")
    assert recorded == hashlib.sha256(migration.read_bytes()).hexdigest()


def test_migration_five_rejects_legacy_malformed_checksum(tmp_path):
    store = SQLiteOperationsStore(tmp_path / "legacy.sqlite3")
    scripts = store._migration_scripts()
    store._migration_scripts = lambda: scripts[:4]
    store.migrate()
    day = RacingDay(RacingDayId(ident("day", 301)), date(2026, 7, 22), "Australia/Melbourne", NOW)
    store.create_racing_day(op(301), day)
    store.record_expected_race(
        op(302),
        day,
        ProgrammeRaceCandidate("official", "legacy", "Ballarat", 1, NOW),
        ArtifactChecksum("sha256:" + "a" * 64),
        NOW,
    )
    with store._connect() as db:
        db.execute("UPDATE expected_races SET programme_checksum='sha256:a'")
    del store._migration_scripts
    with pytest.raises(sqlite3.IntegrityError, match="invalid expected_races.programme_checksum"):
        store.migrate()
    with store._connect() as db:
        assert db.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 4
        assert (
            db.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' "
                "AND name='expected_races_programme_checksum_update'"
            ).fetchone()
            is None
        )


def test_field_evidence_production_round_trip_order_replay_and_append_only(setup):
    store, repository, _, race, artifacts = setup
    card = artifacts.put(b"field-card", media_type="application/json")
    later = FieldEvidence(
        op(2100),
        race.race_id,
        EvidenceField.GRADE,
        EvidenceAuthority.OFFICIAL_CARD,
        "M",
        "card",
        card.checksum,
        NOW + timedelta(seconds=2),
    )
    earlier = FieldEvidence(
        op(2101),
        race.race_id,
        EvidenceField.VENUE,
        EvidenceAuthority.OFFICIAL_CARD,
        "Ballarat",
        "card",
        card.checksum,
        NOW + timedelta(seconds=1),
    )
    assert repository.record_field_evidence(later)
    assert repository.record_field_evidence(earlier)
    assert not repository.record_field_evidence(later)
    assert repository.field_evidence(race.race_id) == (earlier, later)
    with pytest.raises(ConflictingOperation, match="different intent"):
        repository.record_field_evidence(
            FieldEvidence(
                op(2100),
                race.race_id,
                EvidenceField.GRADE,
                EvidenceAuthority.OFFICIAL_CARD,
                "A",
                "card",
                card.checksum,
                later.observed_at,
            )
        )
    with store._connect() as db:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute(
                "UPDATE field_evidence SET value_json=? WHERE operation_id=?",
                ('"changed"', str(op(2100))),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM field_evidence WHERE operation_id=?", (str(op(2100)),))


@pytest.mark.parametrize(
    ("table", "update"),
    [
        ("expected_races", "UPDATE expected_races SET venue=venue WHERE race_id=?"),
        (
            "programme_race_observations",
            "UPDATE programme_race_observations SET venue=venue WHERE race_id=?",
        ),
        ("odds_attempts", "UPDATE odds_attempts SET source=source WHERE race_id=?"),
        (
            "sealed_evidence",
            "UPDATE sealed_evidence SET schema_version=schema_version WHERE race_id=?",
        ),
        (
            "collection_quarantines",
            "UPDATE collection_quarantines SET details=details WHERE race_id=?",
        ),
    ],
)
def test_authoritative_phase2_rows_reject_direct_update_and_delete(setup, table, update):
    store, repository, _, race, _ = setup
    if table == "odds_attempts":
        repository.record_odds_attempt(
            OddsObservation(
                op(2200),
                race.race_id,
                "market",
                NOW,
                OddsAttemptStatus.FAILED,
                error="timeout",
            )
        )
    elif table == "sealed_evidence":
        store.advance_race(op(2201), race.race_id, RaceState.CARD_COLLECTED, NOW)
        store.advance_race(op(2202), race.race_id, RaceState.COLLECTING_ODDS, NOW)
        store.record_field_evidence(
            FieldEvidence(
                op(2203),
                race.race_id,
                EvidenceField.VENUE,
                EvidenceAuthority.OFFICIAL_CARD,
                "Ballarat",
                "card",
                ArtifactChecksum("sha256:" + "a" * 64),
                NOW,
            )
        )
        store.seal_evidence(
            op(2204),
            race_id=race.race_id,
            raw_checksum=ArtifactChecksum("sha256:" + "b" * 64),
            normalized_checksum=ArtifactChecksum("sha256:" + "c" * 64),
            schema_version="v1",
            normalization_version="v1",
            frozen_at=NOW,
            freeze_authority=FreezeAuthority.SCHEDULED_MINUS_BUFFER,
            odds_checksum=ArtifactChecksum("sha256:" + "d" * 64),
            sealed_at=NOW,
            request_intent_digest=ArtifactChecksum("sha256:" + "e" * 64),
        )
    elif table == "collection_quarantines":
        repository.quarantine_collection(
            op(2205),
            race.race_id,
            stage="collection",
            code="hard_cutoff",
            details="collection terminal",
            at=NOW,
        )

    with store._connect() as db:
        assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] > 0
        with pytest.raises(sqlite3.IntegrityError, match=f"{table} is append-only"):
            db.execute(update, (str(race.race_id),))
        with pytest.raises(sqlite3.IntegrityError, match=f"{table} is append-only"):
            db.execute(f"DELETE FROM {table} WHERE race_id=?", (str(race.race_id),))


def test_field_evidence_typed_boundary_rejects_invalid_values_and_timestamps(setup):
    _, _, _, race, _ = setup
    checksum = ArtifactChecksum("sha256:" + "a" * 64)
    with pytest.raises(DomainValidationError, match="timezone-aware"):
        FieldEvidence(
            op(2110),
            race.race_id,
            EvidenceField.VENUE,
            EvidenceAuthority.OFFICIAL_CARD,
            "Ballarat",
            "card",
            checksum,
            datetime(2026, 7, 22),
        )
    with pytest.raises(DomainValidationError, match="exact JSON"):
        FieldEvidence(
            op(2111),
            race.race_id,
            EvidenceField.GRADE,
            EvidenceAuthority.OFFICIAL_CARD,
            float("nan"),
            "card",
            checksum,
            NOW,
        )
    with pytest.raises(DomainValidationError, match="exact JSON"):
        FieldEvidence(
            op(2115),
            race.race_id,
            EvidenceField.GRADE,
            EvidenceAuthority.OFFICIAL_CARD,
            ("M", "Mixed"),
            "card",
            checksum,
            NOW,
        )
    with pytest.raises(DomainValidationError, match="exact JSON"):
        FieldEvidence(
            op(2116),
            race.race_id,
            EvidenceField.GRADE,
            EvidenceAuthority.OFFICIAL_CARD,
            {1: "M"},
            "card",
            checksum,
            NOW,
        )
    with pytest.raises(DomainValidationError, match="must be a datetime"):
        FieldEvidence(
            op(2112),
            race.race_id,
            EvidenceField.ACTUAL_JUMP,
            EvidenceAuthority.OFFICIAL_JUMP,
            "2026-07-22T01:02:03+00:00",
            "jump",
            checksum,
            NOW,
        )
    with pytest.raises(DomainValidationError, match="timezone-aware"):
        FieldEvidence(
            op(2113),
            race.race_id,
            EvidenceField.ACTUAL_JUMP,
            EvidenceAuthority.OFFICIAL_JUMP,
            datetime(2026, 7, 22, 1, 2, 3),
            "jump",
            checksum,
            NOW,
        )
    with pytest.raises(DomainValidationError, match="must not be a datetime"):
        FieldEvidence(
            op(2114),
            race.race_id,
            EvidenceField.VENUE,
            EvidenceAuthority.OFFICIAL_CARD,
            NOW,
            "card",
            checksum,
            NOW,
        )


@pytest.mark.parametrize(
    "field",
    [EvidenceField.SCHEDULED_JUMP, EvidenceField.ACTUAL_JUMP, EvidenceField.JUMP_TIME],
)
def test_temporal_field_evidence_round_trips_through_repository(setup, field):
    _, repository, _, race, artifacts = setup
    proof = artifacts.put(field.value.encode(), media_type="application/json")
    value = datetime(2026, 7, 22, 11, 2, 3, 456789, tzinfo=timezone(timedelta(hours=10)))
    evidence = FieldEvidence(
        op(2140 + list(EvidenceField).index(field)),
        race.race_id,
        field,
        EvidenceAuthority.OFFICIAL_JUMP,
        value,
        "jump",
        proof.checksum,
        NOW,
    )
    assert repository.record_field_evidence(evidence)
    assert repository.field_evidence(race.race_id) == (evidence,)


def test_nested_exact_json_field_evidence_round_trips_through_repository(setup):
    _, repository, _, race, artifacts = setup
    proof = artifacts.put(b"nested-json", media_type="application/json")
    value = {
        "active": True,
        "details": [None, 3, 2.5, "M", {"labels": ["fast", "wide"]}],
    }
    evidence = FieldEvidence(
        op(2117),
        race.race_id,
        EvidenceField.GRADE,
        EvidenceAuthority.OFFICIAL_CARD,
        value,
        "card",
        proof.checksum,
        NOW,
    )

    assert repository.record_field_evidence(evidence)
    assert repository.field_evidence(race.race_id) == (evidence,)


def test_sealing_uses_transactional_field_evidence_without_caller_observations(setup):
    store, repository, _, race, artifacts = setup
    store.advance_race(op(2120), race.race_id, RaceState.CARD_COLLECTED, NOW)
    store.advance_race(op(2121), race.race_id, RaceState.COLLECTING_ODDS, NOW)
    odds = artifacts.put(b"transactional-odds", media_type="application/json")
    mapping = artifacts.put(b"transactional-map", media_type="application/json")
    card = artifacts.put(b"transactional-card", media_type="application/json")
    repository.record_odds_attempt(
        OddsObservation(
            op(2122),
            race.race_id,
            "market",
            NOW,
            OddsAttemptStatus.SUCCEEDED,
            odds.checksum,
            mapping.checksum,
        )
    )
    evidence = FieldEvidence(
        op(2123),
        race.race_id,
        EvidenceField.RUNNER_SET,
        EvidenceAuthority.OFFICIAL_CARD,
        ["A", "B"],
        "card",
        card.checksum,
        NOW,
    )
    store.record_field_evidence(evidence)
    result = EvidenceSealer(store, artifacts).seal(
        operation_id=op(2124),
        race_id=race.race_id,
        source_artifacts={},
        observations=[],
        scheduled_jump=race.scheduled_jump,
        buffer=timedelta(minutes=2),
        schema_version="v1",
        normalization_version="v1",
        sealed_at=NOW,
    )
    package = json.loads(artifacts.read(result.normalized_checksum))
    assert package["fields"] == {"runner_set": ["A", "B"]}
    assert package["field_provenance"][0]["artifact_checksum"] == str(card.checksum)


def test_sealing_rejects_caller_mismatch_without_invalid_seal(setup):
    store, repository, _, race, artifacts = setup
    store.advance_race(op(2130), race.race_id, RaceState.CARD_COLLECTED, NOW)
    store.advance_race(op(2131), race.race_id, RaceState.COLLECTING_ODDS, NOW)
    odds = artifacts.put(b"mismatch-odds", media_type="application/json")
    mapping = artifacts.put(b"mismatch-map", media_type="application/json")
    card = artifacts.put(b"mismatch-card", media_type="application/json")
    repository.record_odds_attempt(
        OddsObservation(
            op(2132),
            race.race_id,
            "market",
            NOW,
            OddsAttemptStatus.SUCCEEDED,
            odds.checksum,
            mapping.checksum,
        )
    )
    store.record_field_evidence(
        FieldEvidence(
            op(2133),
            race.race_id,
            EvidenceField.GRADE,
            EvidenceAuthority.OFFICIAL_CARD,
            "M",
            "card",
            card.checksum,
            NOW,
        )
    )
    with pytest.raises(SealingQuarantine, match="do not exactly match"):
        EvidenceSealer(store, artifacts).seal(
            operation_id=op(2134),
            race_id=race.race_id,
            source_artifacts={"card": card.checksum},
            observations=[
                FieldObservation(
                    EvidenceField.GRADE,
                    "A",
                    EvidenceAuthority.OFFICIAL_CARD,
                    False,
                    "card",
                    card.checksum,
                )
            ],
            scheduled_jump=race.scheduled_jump,
            buffer=timedelta(minutes=2),
            schema_version="v1",
            normalization_version="v1",
            sealed_at=NOW,
        )
    assert store.count("sealed_evidence") == 0
    assert store.count("collection_quarantines") == 1


def test_exact_checksum_triggers_cover_insert_update_and_nullable_odds(setup):
    store, repository, _, race, _ = setup
    repository.ingest_run(
        RunObservation(
            op(310),
            DogId(ident("dog", 310)),
            date(2026, 7, 20),
            "card",
            ArtifactChecksum("sha256:" + "a" * 64),
            NOW,
        ),
        authoritative=False,
    )
    repository.record_odds_attempt(
        OddsObservation(op(311), race.race_id, "market", NOW, OddsAttemptStatus.FAILED, error="x")
    )
    with store._connect() as db:
        db.execute("PRAGMA foreign_keys=OFF")
        for table in (
            "expected_races",
            "programme_race_observations",
            "odds_attempts",
            "sealed_evidence",
        ):
            db.execute(f"DROP TRIGGER {table}_append_only_update")
        db.execute(
            "INSERT INTO field_evidence(race_id, field_name, authority, value_json, "
            "artifact_checksum, observed_at, critical, operation_id) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                str(race.race_id),
                "venue",
                "official_card",
                '"Ballarat"',
                "sha256:" + "b" * 64,
                NOW.isoformat(),
                0,
                str(op(312)),
            ),
        )
        db.execute(
            "INSERT INTO sealed_evidence(race_id, raw_manifest_checksum, normalized_checksum, "
            "schema_version, normalization_version, frozen_at, freeze_authority, odds_checksum, "
            "sealed_at, operation_id, request_intent_digest) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                str(race.race_id),
                "sha256:" + "c" * 64,
                "sha256:" + "d" * 64,
                "v1",
                "v1",
                NOW.isoformat(),
                "scheduled_minus_buffer",
                "sha256:" + "e" * 64,
                NOW.isoformat(),
                str(op(313)),
                "sha256:" + "f" * 64,
            ),
        )
        columns = (
            ("run_observations", "artifact_checksum"),
            ("expected_races", "programme_checksum"),
            ("odds_attempts", "artifact_checksum"),
            ("odds_attempts", "runner_mapping_checksum"),
            ("field_evidence", "artifact_checksum"),
            ("sealed_evidence", "raw_manifest_checksum"),
            ("sealed_evidence", "normalized_checksum"),
            ("sealed_evidence", "odds_checksum"),
            ("programme_race_observations", "programme_checksum"),
        )
        malformed = ("sha256:a", "sha256:" + "A" * 64, "sha256:" + "a" * 64 + "0")
        for table, column in columns:
            for value in malformed:
                expected_update_error = "append-only" if table == "field_evidence" else "invalid"
                with pytest.raises(sqlite3.IntegrityError, match=expected_update_error):
                    db.execute(f"UPDATE {table} SET {column}=?", (value,))
            with pytest.raises(sqlite3.IntegrityError, match="invalid"):
                db.execute(
                    f"INSERT INTO {table} SELECT "
                    + ",".join(
                        "?" if row[1] == column else row[1]
                        for row in db.execute(f"PRAGMA table_info({table})")
                    )
                    + f" FROM {table} LIMIT 1",
                    ("sha256:g" + "0" * 63,),
                )
        db.execute("UPDATE odds_attempts SET artifact_checksum=NULL, runner_mapping_checksum=NULL")


def test_inventory_alias_and_operation_are_idempotent(setup):
    store, repository, day, race, _ = setup
    checksum = store.expected_programme_artifact(race.race_id).checksum
    candidate = ProgrammeRaceCandidate(
        race.source,
        race.source_race_id,
        race.venue,
        race.race_number,
        race.scheduled_jump,
    )
    assert repository.record_expected_race(op(2), day, candidate, checksum, NOW) == race.race_id
    assert store.count("races") == 1


@pytest.mark.parametrize("path", ["alias_owned", "explicit_reconcile"])
def test_immutable_expected_inventory_conflict_is_durable_and_replays_once(setup, path):
    store, repository, day, race, artifacts = setup
    candidate = ProgrammeRaceCandidate(
        race.source if path == "alias_owned" else "second-source",
        race.source_race_id if path == "alias_owned" else "second-id",
        "Geelong",
        8,
        race.scheduled_jump + timedelta(minutes=1),
    )
    checksum = artifacts.put(b"conflicting programme", media_type="application/json").checksum
    kwargs = {} if path == "alias_owned" else {"reconcile_to": race.race_id}

    for _ in range(2):
        with pytest.raises(ExpectedInventoryConflict, match="immutable expected inventory"):
            repository.record_expected_race(op(210), day, candidate, checksum, NOW, **kwargs)

    persisted = store.expected_programme_artifact(race.race_id)
    assert persisted.checksum != checksum
    with store._connect() as db:
        observation = db.execute(
            "SELECT COUNT(*) FROM programme_race_observations WHERE operation_id=?",
            (str(op(210)),),
        ).fetchone()[0]
        quarantine = db.execute(
            "SELECT code, details FROM collection_quarantines WHERE operation_id=?",
            (str(op(210)),),
        ).fetchone()
    assert observation == 1
    assert quarantine["code"] == "expected_inventory_conflict"
    assert "venue" in quarantine["details"] and "programme_checksum" in quarantine["details"]


def test_inventory_attaches_to_preexisting_race(setup):
    store, repository, day, _, artifacts = setup
    preexisting = RaceId(ident("race", 200))
    store.discover_race(op(200), preexisting, day, NOW)
    programme = artifacts.put(b"second programme", media_type="application/json")
    candidate = ProgrammeRaceCandidate("official", "R-200", "Ballarat", 2, NOW + timedelta(hours=3))

    attached = repository.record_expected_race(
        op(201), day, candidate, programme.checksum, NOW, reconcile_to=preexisting
    )

    assert attached == preexisting
    persisted = store.expected_programme_artifact(preexisting)
    assert persisted.checksum == programme.checksum
    assert persisted.scheduled_jump == candidate.scheduled_jump
    with store._connect() as db:
        assert db.execute(
            "SELECT race_id FROM race_aliases WHERE source=? AND alias=?",
            (candidate.source, candidate.source_race_id),
        ).fetchone()["race_id"] == str(preexisting)


def test_alias_owned_inventory_target_must_belong_to_supplied_day(setup):
    store, repository, _, race, artifacts = setup
    other_day = RacingDay(
        RacingDayId(ident("day", 202)), date(2026, 7, 23), "Australia/Melbourne", NOW
    )
    store.create_racing_day(op(202), other_day)
    before = database_counts(store)

    with pytest.raises(OperationsStoreError, match="not in supplied racing day"):
        repository.record_expected_race(
            op(203),
            other_day,
            ProgrammeRaceCandidate(
                race.source,
                race.source_race_id,
                race.venue,
                race.race_number,
                race.scheduled_jump,
            ),
            artifacts.put(b"other day programme", media_type="application/json").checksum,
            NOW,
        )

    assert database_counts(store) == before


def test_race_ids_are_internal_and_cross_source_convergence_is_explicit(setup):
    store, repository, day, first, _ = setup
    checksum = ArtifactChecksum("sha256:" + "2" * 64)
    same_text_other_source = ProgrammeRaceCandidate(
        "second-source",
        first.source_race_id,
        first.venue,
        first.race_number,
        first.scheduled_jump,
    )
    separate = repository.record_expected_race(op(3), day, same_text_other_source, checksum, NOW)
    assert separate != first.race_id

    first_checksum = store.expected_programme_artifact(first.race_id).checksum
    attached = ProgrammeRaceCandidate(
        "third-source",
        "third-local-id",
        first.venue,
        first.race_number,
        first.scheduled_jump,
    )
    assert (
        repository.record_expected_race(
            op(4), day, attached, first_checksum, NOW, reconcile_to=first.race_id
        )
        == first.race_id
    )
    with store._connect() as db:
        aliases = db.execute(
            "SELECT source, alias FROM race_aliases WHERE race_id=? ORDER BY source",
            (str(first.race_id),),
        ).fetchall()
    assert [tuple(row) for row in aliases] == [
        ("official", "R-1"),
        ("third-source", "third-local-id"),
    ]


def test_race_alias_owner_collision_preserves_both_races_and_quarantines(setup):
    store, repository, day, owner, _ = setup
    checksum = ArtifactChecksum("sha256:" + "3" * 64)
    other = repository.record_expected_race(
        op(5),
        day,
        ProgrammeRaceCandidate("other", "O-1", "Geelong", 2, owner.scheduled_jump),
        checksum,
        NOW,
    )
    collision_candidate = ProgrammeRaceCandidate(
        "official", "R-1", "Contradiction", 8, owner.scheduled_jump
    )
    for _ in range(2):
        with pytest.raises(RaceAliasOwnerCollision) as raised:
            repository.record_expected_race(
                op(6), day, collision_candidate, checksum, NOW, reconcile_to=other
            )
        assert raised.value.owner == owner.race_id
        assert raised.value.requested == other
    with store._connect() as db:
        alias_owner = db.execute(
            "SELECT race_id FROM race_aliases WHERE source='official' AND alias='R-1'"
        ).fetchone()["race_id"]
        evidence = db.execute(
            "SELECT race_id, collision FROM programme_race_observations WHERE operation_id=?",
            (str(op(6)),),
        ).fetchone()
    assert alias_owner == str(owner.race_id)
    assert tuple(evidence) == (str(other), 1)
    assert store.count("collection_quarantines") == 1
    with store._connect() as db:
        assert (
            db.execute(
                "SELECT COUNT(*) FROM programme_race_observations WHERE operation_id=?",
                (str(op(6)),),
            ).fetchone()[0]
            == 1
        )


def test_provisional_run_upgrades_without_duplicate_participation(setup):
    store, repository, *_ = setup
    decision = resolve_dog_identity(source="form", registration_id=None, name="Swift One")
    repository.record_identity_decision(
        op(10),
        source="form",
        source_alias="form:swift-one",
        name="Swift One",
        decision=decision,
        at=NOW,
    )
    base = dict(
        dog_id=decision.dog_id,
        local_racing_date=date(2026, 7, 20),
        source="form",
        artifact_checksum=ArtifactChecksum("sha256:" + "2" * 64),
        observed_at=NOW,
    )
    repository.ingest_run(RunObservation(operation_id=op(11), **base), authoritative=False)
    repository.ingest_run(
        RunObservation(operation_id=op(12), **{**base, "source": "official"}),
        authoritative=True,
    )
    with store._connect() as db:
        run = db.execute("SELECT COUNT(*), authoritative FROM dog_runs").fetchone()
        assert tuple(run) == (1, 1)
        assert db.execute("SELECT COUNT(*) FROM run_observations").fetchone()[0] == 2


def test_authoritative_identity_upgrades_different_provisional_dog_id(setup):
    store, repository, *_ = setup
    provisional = resolve_dog_identity(source="form", registration_id=None, name="Swift One")
    repository.record_identity_decision(
        op(50),
        source="form",
        source_alias="form:swift-one",
        name="Swift One",
        decision=provisional,
        at=NOW,
    )
    repository.ingest_run(
        RunObservation(
            op(51),
            provisional.dog_id,
            date(2026, 7, 20),
            "form",
            ArtifactChecksum("sha256:" + "5" * 64),
            NOW,
        ),
        authoritative=False,
    )
    canonical = resolve_dog_identity(
        source="registry",
        registration_authority="registry",
        registration_id="REG-42",
        name="Swift One",
    )
    assert canonical.dog_id != provisional.dog_id
    repository.record_identity_decision(
        op(52),
        source="form",
        source_alias="form:REG-42",
        name="Swift One",
        decision=canonical,
        at=NOW,
    )
    repository.ingest_run(
        RunObservation(
            op(53),
            canonical.dog_id,
            date(2026, 7, 20),
            "official",
            ArtifactChecksum("sha256:" + "6" * 64),
            NOW,
        ),
        authoritative=True,
    )
    assert store.count("dog_runs") == 1
    assert store.count("run_observations") == 2
    assert store.count("dog_identity_aliases") == 1
    assert repository.record_identity_decision(
        op(54),
        source="registry-copy",
        source_alias="registry-copy:REG-42",
        name="Swift One",
        decision=canonical,
        at=NOW,
    )
    assert store.count("dog_identity_aliases") == 1
    assert store.count("identity_quarantines") == 0
    conflicting = resolve_dog_identity(
        source="registry",
        registration_authority="registry",
        registration_id="REG-43",
        name="Swift One",
    )
    assert repository.record_identity_decision(
        op(55),
        source="form",
        source_alias="form:REG-43",
        name="Swift One",
        decision=conflicting,
        at=NOW,
    )
    assert store.count("dog_identity_aliases") == 1
    assert store.count("identity_quarantines") == 1
    with store._connect() as db:
        alias = db.execute(
            "SELECT canonical_dog_id FROM dog_identity_aliases WHERE provisional_dog_id=?",
            (str(provisional.dog_id),),
        ).fetchone()
    assert alias["canonical_dog_id"] == str(canonical.dog_id)


def test_authoritative_identity_upgrade_isolated_from_equal_name_other_source(setup):
    store, repository, *_ = setup
    unrelated = resolve_dog_identity(source="unrelated", registration_id=None, name="Swift One")
    repository.record_identity_decision(
        op(56),
        source="unrelated",
        source_alias="unrelated:swift-one",
        name="Swift One",
        decision=unrelated,
        at=NOW,
    )
    canonical = resolve_dog_identity(
        source="registry",
        registration_authority="registry",
        registration_id="REG-99",
        name="Swift One",
    )

    repository.record_identity_decision(
        op(57),
        source="registry",
        source_alias="registry:REG-99",
        name="Swift One",
        decision=canonical,
        at=NOW,
    )

    assert store.count("dog_identity_aliases") == 0
    assert store.count("identity_quarantines") == 0


def test_ambiguous_identity_decision_is_durably_quarantined(setup):
    store, repository, *_ = setup
    decision = DogIdentityDecision(
        tier=IdentityTier.AMBIGUOUS,
        dog_id=None,
        reason="multiple identity candidates",
    )
    repository.record_identity_decision(
        op(54),
        source="form",
        source_alias="form:swift-one",
        name="Swift One",
        decision=decision,
        at=NOW,
    )
    assert store.count("identity_quarantines") == 1


def test_conflicting_source_alias_preserves_owner_and_quarantines_without_upgrade(
    setup,
):
    store, repository, *_ = setup
    owner = resolve_dog_identity(
        source="registry",
        registration_authority="registry",
        registration_id="REG-42",
        name="Swift One",
    )
    contradictory = resolve_dog_identity(
        source="registry",
        registration_authority="registry",
        registration_id="REG-43",
        name="Swift One",
    )
    provisional = resolve_dog_identity(source="form", registration_id=None, name="Swift One")
    repository.record_identity_decision(
        op(80),
        source="registry",
        source_alias="registry:shared",
        name="Other Name",
        decision=owner,
        at=NOW,
    )
    repository.record_identity_decision(
        op(81),
        source="form",
        source_alias="form:swift-one",
        name="Swift One",
        decision=provisional,
        at=NOW,
    )

    assert repository.record_identity_decision(
        op(82),
        source="registry",
        source_alias="registry:shared",
        name="Swift One",
        decision=contradictory,
        at=NOW,
    )

    with store._connect() as db:
        alias_owner = db.execute(
            "SELECT dog_id FROM dog_aliases WHERE source=? AND alias=?",
            ("registry", "registry:shared"),
        ).fetchone()["dog_id"]
        decision = db.execute(
            "SELECT dog_id FROM dog_identity_decisions WHERE operation_id=?",
            (str(op(82)),),
        ).fetchone()["dog_id"]
        quarantine = db.execute(
            "SELECT reason FROM identity_quarantines WHERE operation_id=?",
            (str(op(82)),),
        ).fetchone()["reason"]
    assert alias_owner == str(owner.dog_id)
    assert decision == str(contradictory.dog_id)
    assert quarantine == "source alias already belongs to a different DogId"
    assert store.count("dog_identity_aliases") == 0


def test_invalid_identity_tier_fails_closed_without_database_changes(setup):
    store, *_ = setup
    before = database_counts(store)

    with pytest.raises(OperationsStoreError, match="tier must be an IdentityTier"):
        store.record_identity_decision(
            op(56),
            source="form",
            source_alias="form:swift-one",
            normalized_name="swift one",
            tier="privileged",  # type: ignore[arg-type]
            dog_id=None,
            reason="invalid tier probe",
            at=NOW,
        )

    assert database_counts(store) == before


@pytest.mark.parametrize(
    ("tier", "dog_id", "message"),
    [
        (IdentityTier.AMBIGUOUS, DogId(ident("dog", 90)), "AMBIGUOUS requires"),
        (IdentityTier.AUTHORITATIVE, None, "AUTHORITATIVE requires"),
        (
            IdentityTier.HIGH_CONFIDENCE_PROVISIONAL,
            None,
            "HIGH_CONFIDENCE_PROVISIONAL requires",
        ),
        (IdentityTier.AUTHORITATIVE, ident("dog", 91), "dog_id must be a DogId"),
    ],
)
def test_store_rejects_invalid_identity_contract_without_database_changes(
    setup, tier, dog_id, message
):
    store, *_ = setup
    before = database_counts(store)
    with pytest.raises(OperationsStoreError, match=message):
        store.record_identity_decision(
            op(92),
            source="form",
            source_alias="form:swift-one",
            normalized_name="swift one",
            tier=tier,
            dog_id=dog_id,
            reason="invalid contract",
            at=NOW,
        )
    assert database_counts(store) == before


@pytest.mark.parametrize(
    ("source", "source_alias"),
    [("form", "swift-one"), ("registry", "upstream:region:REG-42")],
)
def test_repository_preserves_explicit_identity_source_and_alias(setup, source, source_alias):
    store, repository, *_ = setup
    decision = resolve_dog_identity(source=source, registration_id=None, name="Swift One")

    repository.record_identity_decision(
        op(94),
        source=source,
        source_alias=source_alias,
        name="Swift One",
        decision=decision,
        at=NOW,
    )

    with store._connect() as db:
        row = db.execute(
            "SELECT source, source_alias FROM dog_identity_decisions WHERE operation_id=?",
            (str(op(94)),),
        ).fetchone()
    assert tuple(row) == (source, source_alias)


def test_equal_normalized_identity_names_are_isolated_by_explicit_source(setup):
    store, repository, *_ = setup
    for number, source in ((95, "form"), (96, "unrelated")):
        decision = resolve_dog_identity(source=source, registration_id=None, name="Swift One")
        repository.record_identity_decision(
            op(number),
            source=source,
            source_alias="shared-alias",
            name="Swift One",
            decision=decision,
            at=NOW,
        )

    with store._connect() as db:
        rows = db.execute(
            "SELECT source, source_alias, normalized_name FROM dog_identity_decisions "
            "WHERE operation_id IN (?, ?) ORDER BY source",
            (str(op(95)), str(op(96))),
        ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("form", "shared-alias", "swift one"),
        ("unrelated", "shared-alias", "swift one"),
    ]
    assert store.count("identity_quarantines") == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", "   "),
        ("source_alias", ""),
        ("source", 42),
        ("source_alias", None),
        ("normalized_name", []),
        ("reason", object()),
    ],
)
def test_store_rejects_malformed_identity_text_without_database_changes(setup, field, value):
    store, *_ = setup
    arguments = {
        "source": "form",
        "source_alias": "form:swift-one",
        "normalized_name": "swift one",
        "tier": IdentityTier.AMBIGUOUS,
        "dog_id": None,
        "reason": "multiple candidates",
        "at": NOW,
    }
    arguments[field] = value
    before = database_counts(store)

    with pytest.raises(OperationsStoreError, match=f"{field} must be nonblank text"):
        store.record_identity_decision(op(97), **arguments)

    assert database_counts(store) == before


def test_repository_rejects_name_that_normalizes_empty_without_database_changes(setup):
    store, repository, *_ = setup
    decision = DogIdentityDecision(IdentityTier.AMBIGUOUS, None, "multiple candidates")
    before = database_counts(store)

    with pytest.raises(OperationsStoreError, match="normalized_name must be nonblank text"):
        repository.record_identity_decision(
            op(98),
            source="form",
            source_alias="delimiter-free-alias",
            name="   ",
            decision=decision,
            at=NOW,
        )

    assert database_counts(store) == before


def test_public_phase2_reads_reconstruct_typed_immutable_models(setup):
    store, repository, day, race, _ = setup
    observation = OddsObservation(
        op(93),
        race.race_id,
        "market",
        NOW,
        OddsAttemptStatus.FAILED,
        error="timeout",
    )
    repository.record_odds_attempt(observation)

    attempt = store.odds_attempts(race.race_id)[0]
    row = store.collection_rows(day)[0]
    assert isinstance(attempt, OddsAttemptRecord)
    assert attempt.attempted_at == NOW
    assert attempt.scheduled_due_at == NOW
    assert attempt.timing_policy == "adaptive-odds-timing-v1"
    assert attempt.status is OddsAttemptStatus.FAILED
    assert isinstance(row, CollectionRaceRecord)
    assert row.race_id == race.race_id
    assert row.state is RaceState.DISCOVERED


def test_expected_programme_artifact_is_typed(setup):
    store, _, _, race, artifacts = setup
    programme = store.expected_programme_artifact(race.race_id)
    assert programme.source == "official"
    assert artifacts.read(programme.checksum) == b"official programme"
    assert programme.scheduled_jump == race.scheduled_jump


@pytest.mark.parametrize("column", ["source", "programme_checksum", "scheduled_jump"])
def test_expected_programme_artifact_corruption_fails_closed(setup, column):
    store, _, _, race, _ = setup
    invalid = {
        "source": "",
        "programme_checksum": "not-a-checksum",
        "scheduled_jump": "not-a-timestamp",
    }[column]
    with store._connect() as db:
        db.execute("PRAGMA ignore_check_constraints = ON")
        db.execute("DROP TRIGGER expected_races_append_only_update")
        if column == "programme_checksum":
            db.execute("DROP TRIGGER expected_races_programme_checksum_update")
        db.execute(f"UPDATE expected_races SET {column}=?", (invalid,))
    with pytest.raises(OperationsStoreError, match="invalid expected programme artifact"):
        store.expected_programme_artifact(race.race_id)


@pytest.mark.parametrize(
    ("table", "column", "invalid"),
    [
        ("odds_attempts", "status", "unknown"),
        ("odds_attempts", "attempted_at", "not-a-timestamp"),
        ("odds_attempts", "scheduled_due_at", "not-a-timestamp"),
        ("odds_attempts", "timing_policy", "adaptive-odds-timing-v0"),
        ("races", "state", "unknown"),
        ("races", "race_id", "not-a-race-id"),
    ],
)
def test_public_phase2_reads_fail_closed_on_invalid_persisted_values(setup, table, column, invalid):
    store, repository, day, race, _ = setup
    repository.record_odds_attempt(
        OddsObservation(
            op(94),
            race.race_id,
            "market",
            NOW,
            OddsAttemptStatus.FAILED,
            error="timeout",
        )
    )
    with store._connect() as db:
        db.execute("PRAGMA foreign_keys = OFF")
        db.execute("PRAGMA ignore_check_constraints = ON")
        if table == "odds_attempts":
            db.execute("DROP TRIGGER odds_attempts_append_only_update")
        elif table == "races" and column == "race_id":
            db.execute("DROP TRIGGER expected_races_append_only_update")
        db.execute(f"UPDATE {table} SET {column}=?", (invalid,))
        if table == "races" and column == "race_id":
            db.execute("UPDATE expected_races SET race_id=?", (invalid,))
    with pytest.raises((ValueError, TypeError)):
        if table == "odds_attempts":
            store.odds_attempts(race.race_id)
        else:
            store.collection_rows(day)


def test_failed_attempt_is_preserved_and_actual_jump_selects_latest_valid(setup):
    store, repository, _, race, artifacts = setup
    store.advance_race(op(20), race.race_id, RaceState.CARD_COLLECTED, NOW)
    store.advance_race(op(21), race.race_id, RaceState.COLLECTING_ODDS, NOW)
    odds1 = artifacts.put(b"odds-1", media_type="application/json")
    mapping = artifacts.put(b"mapping", media_type="application/json")
    repository.record_odds_attempt(
        OddsObservation(
            op(22),
            race.race_id,
            "market",
            NOW,
            OddsAttemptStatus.SUCCEEDED,
            odds1.checksum,
            mapping.checksum,
        )
    )
    repository.record_odds_attempt(
        OddsObservation(
            op(23),
            race.race_id,
            "market",
            NOW + timedelta(minutes=2),
            OddsAttemptStatus.FAILED,
            error="timeout",
        )
    )
    source = artifacts.put(b"card", media_type="text/csv")
    observations = [
        FieldObservation(
            EvidenceField.RUNNER_SET,
            ["A", "B"],
            EvidenceAuthority.OFFICIAL_CARD,
            True,
            "card",
            source.checksum,
        ),
        FieldObservation(
            EvidenceField.ACTUAL_JUMP,
            NOW + timedelta(minutes=5),
            EvidenceAuthority.OFFICIAL_JUMP,
            True,
            "card",
            source.checksum,
        ),
    ]
    persist_fields(store, race.race_id, observations, 2000)
    result = EvidenceSealer(store, artifacts).seal(
        operation_id=op(24),
        race_id=race.race_id,
        source_artifacts={"card": source.checksum},
        observations=observations,
        scheduled_jump=race.scheduled_jump,
        buffer=timedelta(minutes=2),
        schema_version="race-evidence-v1",
        normalization_version="normalizer-v1",
        sealed_at=NOW + timedelta(minutes=6),
    )
    assert result.odds_checksum == odds1.checksum
    assert result.freeze_authority is FreezeAuthority.ACTUAL_JUMP
    normalized = json.loads(artifacts.read(result.normalized_checksum))
    jump_provenance = next(
        item for item in normalized["field_provenance"] if item["field"] == "actual_jump"
    )
    assert jump_provenance["source"] == "card"
    assert jump_provenance["artifact_checksum"] == str(source.checksum)
    raw_manifest = json.loads(artifacts.read(result.raw_manifest_checksum))
    programme = store.expected_programme_artifact(race.race_id)
    assert raw_manifest["programme_artifact"] == {
        "source": programme.source,
        "artifact_checksum": str(programme.checksum),
    }
    assert raw_manifest["sources"] == {"card": str(source.checksum)}
    assert raw_manifest["odds_attempts"] == [
        {
            "source": "market",
            "attempted_at": NOW.isoformat(timespec="microseconds"),
            "status": "succeeded",
            "artifact_checksum": str(odds1.checksum),
            "runner_mapping_checksum": str(mapping.checksum),
            "error": None,
        },
        {
            "source": "market",
            "attempted_at": (NOW + timedelta(minutes=2)).isoformat(timespec="microseconds"),
            "status": "failed",
            "artifact_checksum": None,
            "runner_mapping_checksum": None,
            "error": "timeout",
        },
    ]
    assert raw_manifest["field_observations"] == normalized["field_provenance"]
    repository.quarantine_collection(
        op(127),
        race.race_id,
        stage="collection",
        code="later_audit_quarantine",
        details="preserve committed seal replay",
        at=NOW + timedelta(minutes=7),
    )
    assert (
        EvidenceSealer(store, artifacts).seal(
            operation_id=op(24),
            race_id=race.race_id,
            source_artifacts={"card": source.checksum},
            observations=observations,
            scheduled_jump=race.scheduled_jump,
            buffer=timedelta(minutes=2),
            schema_version="race-evidence-v1",
            normalization_version="normalizer-v1",
            sealed_at=NOW + timedelta(minutes=6),
        )
        == result
    )
    raw_bytes = artifacts.read(result.raw_manifest_checksum)
    normalized_bytes = artifacts.read(result.normalized_checksum)
    before = database_counts(store)
    with pytest.raises(OperationsStoreError, match="odds collection has terminated"):
        repository.record_odds_attempt(
            OddsObservation(
                op(28),
                race.race_id,
                "market",
                NOW + timedelta(minutes=7),
                OddsAttemptStatus.FAILED,
                error="post-seal probe",
            )
        )
    assert database_counts(store) == before
    assert (
        repository.record_odds_attempt(
            OddsObservation(
                op(22),
                race.race_id,
                "market",
                NOW,
                OddsAttemptStatus.SUCCEEDED,
                odds1.checksum,
                mapping.checksum,
            )
        )
        is False
    )
    assert artifacts.read(result.raw_manifest_checksum) == raw_bytes
    assert artifacts.read(result.normalized_checksum) == normalized_bytes


@pytest.mark.parametrize("damage", ["delete", "corrupt"])
def test_exact_committed_seal_replay_is_zero_write_without_retained_inputs(setup, damage):
    store, repository, _, race, artifacts = setup
    store.advance_race(op(330), race.race_id, RaceState.CARD_COLLECTED, NOW)
    store.advance_race(op(331), race.race_id, RaceState.COLLECTING_ODDS, NOW)
    odds = artifacts.put(b"replay-odds", media_type="application/json")
    mapping = artifacts.put(b"replay-mapping", media_type="application/json")
    source = artifacts.put(b"replay-source", media_type="application/json")
    repository.record_odds_attempt(
        OddsObservation(
            op(332),
            race.race_id,
            "market",
            NOW,
            OddsAttemptStatus.SUCCEEDED,
            odds.checksum,
            mapping.checksum,
        )
    )
    observations = [
        FieldObservation(
            EvidenceField.RUNNER_SET,
            ["A", "B"],
            EvidenceAuthority.OFFICIAL_CARD,
            True,
            "card",
            source.checksum,
        )
    ]
    persist_fields(store, race.race_id, observations, 2010)
    result = EvidenceSealer(store, artifacts).seal(
        operation_id=op(333),
        race_id=race.race_id,
        source_artifacts={"card": source.checksum},
        observations=observations,
        scheduled_jump=race.scheduled_jump,
        buffer=timedelta(minutes=2),
        schema_version="race-evidence-v1",
        normalization_version="normalizer-v1",
        sealed_at=NOW + timedelta(minutes=1),
    )
    programme = store.expected_programme_artifact(race.race_id)
    retained_inputs = (
        programme.checksum,
        odds.checksum,
        mapping.checksum,
        source.checksum,
    )
    for checksum in retained_inputs:
        path = artifacts.path_for(checksum)
        if damage == "delete":
            path.unlink()
        else:
            path.write_bytes(b"corrupt retained input")
    repository.quarantine_collection(
        op(334),
        race.race_id,
        stage="collection",
        code="post_commit_audit",
        details="replay must use the committed seal",
        at=NOW + timedelta(minutes=2),
    )

    before_database = database_counts(store)
    before_artifacts = artifact_bytes(artifacts)
    with closing(sqlite3.connect(store.path)) as observer:
        before_data_version = observer.execute("PRAGMA data_version").fetchone()[0]
        replay = EvidenceSealer(store, artifacts).seal(
            operation_id=op(333),
            race_id=race.race_id,
            source_artifacts={"card": source.checksum},
            observations=observations,
            scheduled_jump=race.scheduled_jump,
            buffer=timedelta(minutes=2),
            schema_version="race-evidence-v1",
            normalization_version="normalizer-v1",
            sealed_at=NOW + timedelta(minutes=1),
        )
        after_data_version = observer.execute("PRAGMA data_version").fetchone()[0]

    assert replay == result
    assert replay.frozen_at == result.frozen_at
    assert replay.freeze_authority is result.freeze_authority
    assert after_data_version == before_data_version
    assert database_counts(store) == before_database
    assert artifact_bytes(artifacts) == before_artifacts


def test_real_phase2_sealer_envelope_derives_runner_features(setup):
    store, repository, _, race, artifacts = setup
    arguments = committed_seal_fixture(store, repository, race, artifacts, 123456789)
    committed = store.committed_seal(arguments["operation_id"], race.race_id)
    assert committed is not None
    schema = json.dumps(
        {
            "contract_version": "sealed-race-features-v1",
            "evidence_schema_version": "race-evidence-v1",
            "normalization_version": "normalizer-v1",
            "fields": [
                {
                    "name": "speed",
                    "source_field": "runner_features",
                    "semantics": "identity-critical",
                },
                {
                    "name": "form",
                    "source_field": "runner_features",
                    "semantics": "forecast-required",
                },
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    policy = b'{"imputation":{}}'
    result = derive_features(
        artifacts.read(committed.normalized_checksum),
        expected_evidence_checksum=committed.normalized_checksum,
        schema_bytes=schema,
        expected_schema_checksum=LocalArtifactStore.checksum(schema),
        missingness_policy_bytes=policy,
        expected_missingness_checksum=LocalArtifactStore.checksum(policy),
    )
    assert result.matrix.runner_ids == ("A", "B")


@pytest.mark.parametrize(
    ("corruption", "error_type", "message"),
    [
        ("missing_owner", OperationsStoreError, "has no owning operation"),
        ("wrong_kind", ConflictingOperation, "different intent"),
        ("payload_hash", OperationsStoreError, "inconsistent request intent"),
    ],
)
def test_corrupt_committed_seal_operation_authority_fails_before_inputs_or_writes(
    setup, monkeypatch, corruption, error_type, message
):
    store, repository, _, race, artifacts = setup
    arguments = committed_seal_fixture(store, repository, race, artifacts, 4000)
    operation_id = str(arguments["operation_id"])
    with sqlite3.connect(store.path) as db:
        if corruption == "missing_owner":
            db.execute("PRAGMA foreign_keys=OFF")
            db.execute("DELETE FROM operations WHERE operation_id=?", (operation_id,))
        elif corruption == "wrong_kind":
            db.execute(
                "UPDATE operations SET kind='advance_race' WHERE operation_id=?",
                (operation_id,),
            )
        else:
            db.execute(
                "UPDATE operations SET payload_sha256=? WHERE operation_id=?",
                ("0" * 64, operation_id),
            )

    class UnconsumableObservations:
        def __iter__(self):
            raise AssertionError("corrupt operation authority consumed observations")

    arguments["observations"] = UnconsumableObservations()
    monkeypatch.setattr(
        type(artifacts),
        "verify",
        lambda _artifacts, *_: pytest.fail("artifact verify"),
    )
    monkeypatch.setattr(
        type(artifacts),
        "put",
        lambda _artifacts, *_args, **_kwargs: pytest.fail("artifact write"),
    )
    before_database = database_counts(store)
    before_artifacts = artifact_bytes(artifacts)
    with closing(sqlite3.connect(store.path)) as observer:
        before_data_version = observer.execute("PRAGMA data_version").fetchone()[0]
        with pytest.raises(error_type, match=message):
            EvidenceSealer(store, artifacts).seal(**arguments)
        after_data_version = observer.execute("PRAGMA data_version").fetchone()[0]

    assert after_data_version == before_data_version
    assert database_counts(store) == before_database
    assert artifact_bytes(artifacts) == before_artifacts


@pytest.mark.parametrize(
    "dimension",
    [
        "race",
        "source_mapping",
        "source_checksum",
        "observation_structure",
        "observation_value",
        "scheduled_jump",
        "buffer",
        "schema_version",
        "normalization_version",
        "sealed_at",
    ],
)
def test_committed_seal_changed_request_intent_conflicts_without_writes(setup, dimension):
    store, repository, _, race, artifacts = setup
    store.advance_race(op(335), race.race_id, RaceState.CARD_COLLECTED, NOW)
    store.advance_race(op(336), race.race_id, RaceState.COLLECTING_ODDS, NOW)
    odds = artifacts.put(b"intent-odds", media_type="application/json")
    mapping = artifacts.put(b"intent-mapping", media_type="application/json")
    source = artifacts.put(b"intent-source", media_type="application/json")
    repository.record_odds_attempt(
        OddsObservation(
            op(337),
            race.race_id,
            "market",
            NOW,
            OddsAttemptStatus.SUCCEEDED,
            odds.checksum,
            mapping.checksum,
        )
    )
    observations = [
        FieldObservation(
            EvidenceField.RUNNER_SET,
            ["A", "B"],
            EvidenceAuthority.OFFICIAL_CARD,
            True,
            "card",
            source.checksum,
        )
    ]
    persist_fields(store, race.race_id, observations, 2020)
    arguments = {
        "operation_id": op(338),
        "race_id": race.race_id,
        "source_artifacts": {"card": source.checksum},
        "observations": observations,
        "scheduled_jump": race.scheduled_jump,
        "buffer": timedelta(minutes=2),
        "schema_version": "race-evidence-v1",
        "normalization_version": "normalizer-v1",
        "sealed_at": NOW + timedelta(minutes=1),
    }
    EvidenceSealer(store, artifacts).seal(**arguments)
    if dimension == "race":
        arguments["race_id"] = RaceId(ident("race", 9999))

        def observations_if_consumed():
            raise AssertionError("observations must not be consumed for a wrong-race replay")
            yield

        arguments["observations"] = observations_if_consumed()
    elif dimension == "source_mapping":
        arguments["source_artifacts"] = {"renamed-card": source.checksum}
    elif dimension == "source_checksum":
        arguments["source_artifacts"] = {"card": ArtifactChecksum("sha256:" + "9" * 64)}
    elif dimension == "observation_structure":
        arguments["observations"] = observations + [
            FieldObservation(
                EvidenceField.VENUE,
                "Ballarat",
                EvidenceAuthority.OFFICIAL_CARD,
                True,
                "card",
                source.checksum,
            )
        ]
    elif dimension == "observation_value":
        arguments["observations"] = [
            FieldObservation(
                EvidenceField.RUNNER_SET,
                ["A", "C"],
                EvidenceAuthority.OFFICIAL_CARD,
                True,
                "card",
                source.checksum,
            )
        ]
    elif dimension == "scheduled_jump":
        arguments["scheduled_jump"] = race.scheduled_jump + timedelta(microseconds=1)
    elif dimension == "buffer":
        arguments["buffer"] = timedelta(minutes=3)
    elif dimension == "schema_version":
        arguments["schema_version"] = "race-evidence-v2"
    elif dimension == "normalization_version":
        arguments["normalization_version"] = "normalizer-v2"
    else:
        arguments["sealed_at"] = NOW + timedelta(minutes=2)
    before_database = database_counts(store)
    before_artifacts = artifact_bytes(artifacts)
    with closing(sqlite3.connect(store.path)) as observer:
        before_data_version = observer.execute("PRAGMA data_version").fetchone()[0]
        with pytest.raises(ConflictingOperation):
            EvidenceSealer(store, artifacts).seal(**arguments)
        after_data_version = observer.execute("PRAGMA data_version").fetchone()[0]
    assert after_data_version == before_data_version
    assert database_counts(store) == before_database
    assert artifact_bytes(artifacts) == before_artifacts


def test_empty_caller_committed_intent_binds_effective_evidence_and_replays_without_io(
    setup, monkeypatch
):
    store, repository, _, race, artifacts = setup
    store.advance_race(op(4100), race.race_id, RaceState.CARD_COLLECTED, NOW)
    store.advance_race(op(4101), race.race_id, RaceState.COLLECTING_ODDS, NOW)
    odds = artifacts.put(b"effective-odds", media_type="application/json")
    mapping = artifacts.put(b"effective-mapping", media_type="application/json")
    card = artifacts.put(b"effective-card", media_type="application/json")
    repository.record_odds_attempt(
        OddsObservation(
            op(4102),
            race.race_id,
            "market",
            NOW,
            OddsAttemptStatus.SUCCEEDED,
            odds.checksum,
            mapping.checksum,
        )
    )
    effective_observations = (
        FieldObservation(
            EvidenceField.RUNNER_SET,
            ["A", "B"],
            EvidenceAuthority.OFFICIAL_CARD,
            True,
            "card",
            card.checksum,
        ),
    )
    persist_fields(store, race.race_id, effective_observations, 4103)
    arguments = {
        "operation_id": op(4104),
        "race_id": race.race_id,
        "source_artifacts": {},
        "observations": [],
        "scheduled_jump": race.scheduled_jump,
        "buffer": timedelta(minutes=2),
        "schema_version": "v1",
        "normalization_version": "n1",
        "sealed_at": NOW + timedelta(minutes=1),
    }
    result = EvidenceSealer(store, artifacts).seal(**arguments)
    committed = store.committed_seal(op(4104), race.race_id)
    assert committed is not None
    expected = empty_caller_effective_digest(
        race.race_id,
        {"card": card.checksum},
        effective_observations,
        race.scheduled_jump,
        timedelta(minutes=2),
        "v1",
        "n1",
        NOW + timedelta(minutes=1),
    )
    assert committed.request_intent_digest == expected
    changed_checksum = ArtifactChecksum("sha256:" + "a" * 64)
    assert expected != empty_caller_effective_digest(
        race.race_id,
        {"card": changed_checksum},
        (
            FieldObservation(
                EvidenceField.RUNNER_SET,
                ["A", "B"],
                EvidenceAuthority.OFFICIAL_CARD,
                True,
                "card",
                changed_checksum,
            ),
        ),
        race.scheduled_jump,
        timedelta(minutes=2),
        "v1",
        "n1",
        NOW + timedelta(minutes=1),
    )

    before_database = database_counts(store)
    before_artifacts = artifact_bytes(artifacts)
    monkeypatch.setattr(type(artifacts), "verify", lambda *_: pytest.fail("artifact verify"))
    monkeypatch.setattr(
        type(artifacts), "put", lambda *_args, **_kwargs: pytest.fail("artifact put")
    )
    with closing(sqlite3.connect(store.path)) as observer:
        before_data_version = observer.execute("PRAGMA data_version").fetchone()[0]
        assert EvidenceSealer(store, artifacts).seal(**arguments) == result
        after_data_version = observer.execute("PRAGMA data_version").fetchone()[0]
    assert after_data_version == before_data_version
    assert database_counts(store) == before_database
    assert artifact_bytes(artifacts) == before_artifacts

    with sqlite3.connect(store.path) as corruptor:
        assert corruptor.execute(
            "SELECT 1 FROM sqlite_master WHERE type='trigger' "
            "AND name='field_evidence_append_only_update'"
        ).fetchone()
        corruptor.execute("DROP TRIGGER field_evidence_append_only_update")
        corruptor.execute(
            "UPDATE field_evidence SET artifact_checksum=? WHERE race_id=?",
            (str(changed_checksum), str(race.race_id)),
        )

    before_database = database_counts(store)
    before_artifacts = artifact_bytes(artifacts)
    with closing(sqlite3.connect(store.path)) as observer:
        before_data_version = observer.execute("PRAGMA data_version").fetchone()[0]
        with pytest.raises(ConflictingOperation):
            EvidenceSealer(store, artifacts).seal(**arguments)
        after_data_version = observer.execute("PRAGMA data_version").fetchone()[0]
    assert after_data_version == before_data_version
    assert database_counts(store) == before_database
    assert artifact_bytes(artifacts) == before_artifacts


@pytest.mark.parametrize("stored_digest", [None, "not-a-request-intent-digest"])
def test_committed_seal_without_valid_request_intent_authority_fails_closed(setup, stored_digest):
    store, repository, _, race, artifacts = setup
    store.advance_race(op(339), race.race_id, RaceState.CARD_COLLECTED, NOW)
    store.advance_race(op(340), race.race_id, RaceState.COLLECTING_ODDS, NOW)
    odds = artifacts.put(b"authority-odds", media_type="application/json")
    mapping = artifacts.put(b"authority-mapping", media_type="application/json")
    source = artifacts.put(b"authority-source", media_type="application/json")
    repository.record_odds_attempt(
        OddsObservation(
            op(341),
            race.race_id,
            "market",
            NOW,
            OddsAttemptStatus.SUCCEEDED,
            odds.checksum,
            mapping.checksum,
        )
    )
    observations = [
        FieldObservation(
            EvidenceField.RUNNER_SET,
            ["A", "B"],
            EvidenceAuthority.OFFICIAL_CARD,
            True,
            "card",
            source.checksum,
        )
    ]
    persist_fields(store, race.race_id, observations, 2030)
    arguments = {
        "operation_id": op(342),
        "race_id": race.race_id,
        "source_artifacts": {"card": source.checksum},
        "observations": observations,
        "scheduled_jump": race.scheduled_jump,
        "buffer": timedelta(minutes=2),
        "schema_version": "race-evidence-v1",
        "normalization_version": "normalizer-v1",
        "sealed_at": NOW + timedelta(minutes=1),
    }
    EvidenceSealer(store, artifacts).seal(**arguments)
    with sqlite3.connect(store.path) as db:
        db.execute("DROP TRIGGER sealed_evidence_append_only_update")
        if stored_digest is not None:
            db.execute("PRAGMA ignore_check_constraints=ON")
        db.execute(
            "UPDATE sealed_evidence SET request_intent_digest=? WHERE operation_id=?",
            (stored_digest, str(op(342))),
        )

    class UnconsumableObservations:
        def __iter__(self):
            raise AssertionError("invalid committed authority consumed observations")

    arguments["observations"] = UnconsumableObservations()
    before_database = database_counts(store)
    before_artifacts = artifact_bytes(artifacts)
    with pytest.raises(OperationsStoreError, match="invalid committed seal"):
        EvidenceSealer(store, artifacts).seal(**arguments)
    assert database_counts(store) == before_database
    assert artifact_bytes(artifacts) == before_artifacts


@pytest.mark.parametrize("failure", ["missing", "corrupt", "checksum_mismatch"])
def test_persisted_programme_artifact_failure_quarantines_once_on_replay(setup, failure):
    store, repository, _, race, artifacts = setup
    store.advance_race(op(95), race.race_id, RaceState.CARD_COLLECTED, NOW)
    store.advance_race(op(96), race.race_id, RaceState.COLLECTING_ODDS, NOW)
    odds = artifacts.put(b"odds", media_type="application/json")
    mapping = artifacts.put(b"mapping", media_type="application/json")
    repository.record_odds_attempt(
        OddsObservation(
            op(97),
            race.race_id,
            "market",
            NOW,
            OddsAttemptStatus.SUCCEEDED,
            odds.checksum,
            mapping.checksum,
        )
    )
    repository.record_field_evidence(
        FieldEvidence(
            op(2097),
            race.race_id,
            EvidenceField.VENUE,
            EvidenceAuthority.OFFICIAL_CARD,
            "Ballarat",
            "odds",
            odds.checksum,
            NOW,
        )
    )
    programme = store.expected_programme_artifact(race.race_id)
    if failure == "missing":
        artifacts.path_for(programme.checksum).unlink()
    elif failure == "corrupt":
        artifacts.path_for(programme.checksum).write_bytes(b"corrupt")
    else:
        other = artifacts.put(b"other programme", media_type="application/json")
        with store._connect() as db:
            db.execute("DROP TRIGGER expected_races_append_only_update")
            db.execute(
                "UPDATE expected_races SET programme_checksum=? WHERE race_id=?",
                (str(other.checksum), str(race.race_id)),
            )
        artifacts.path_for(other.checksum).write_bytes(b"wrong bytes")
    sealer = EvidenceSealer(store, artifacts)
    for _ in range(2):
        with pytest.raises(SealingQuarantine):
            sealer.seal(
                operation_id=op(98),
                race_id=race.race_id,
                source_artifacts={},
                observations=[],
                scheduled_jump=race.scheduled_jump,
                buffer=timedelta(minutes=2),
                schema_version="v1",
                normalization_version="v1",
                sealed_at=NOW,
            )
    assert store.count("collection_quarantines") == 1
    assert store.count("sealed_evidence") == 0


def test_missing_expected_programme_row_quarantines_once_on_replay(tmp_path):
    store = SQLiteOperationsStore(tmp_path / "operations.sqlite3")
    store.migrate()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    day = RacingDay(RacingDayId(ident("day", 2)), date(2026, 7, 22), "Australia/Melbourne", NOW)
    race_id = RaceId(ident("race", 2))
    store.create_racing_day(op(100), day)
    store.discover_race(op(101), race_id, day, NOW)
    store.advance_race(op(102), race_id, RaceState.CARD_COLLECTED, NOW)
    store.advance_race(op(103), race_id, RaceState.COLLECTING_ODDS, NOW)
    evidence_artifact = artifacts.put(b"orphan field evidence", media_type="application/json")
    store.record_field_evidence(
        FieldEvidence(
            op(2098),
            race_id,
            EvidenceField.VENUE,
            EvidenceAuthority.OFFICIAL_CARD,
            "Ballarat",
            "card",
            evidence_artifact.checksum,
            NOW,
        )
    )
    sealer = EvidenceSealer(store, artifacts)
    for _ in range(2):
        with pytest.raises(SealingQuarantine, match="no expected programme artifact"):
            sealer.seal(
                operation_id=op(104),
                race_id=race_id,
                source_artifacts={},
                observations=[],
                scheduled_jump=NOW + timedelta(hours=1),
                buffer=timedelta(minutes=2),
                schema_version="v1",
                normalization_version="v1",
                sealed_at=NOW,
            )
    assert store.count("collection_quarantines") == 1


def test_quarantined_seal_replay_raises_persisted_cause_without_writes(setup, monkeypatch):
    store, _, _, race, artifacts = setup
    arguments = {
        "operation_id": op(320),
        "race_id": race.race_id,
        "source_artifacts": {},
        "observations": [],
        "scheduled_jump": race.scheduled_jump,
        "buffer": timedelta(minutes=2),
        "schema_version": "v1",
        "normalization_version": "v1",
        "sealed_at": NOW,
    }
    with pytest.raises(SealingQuarantine, match="no transactional field evidence"):
        EvidenceSealer(store, artifacts).seal(**arguments)
    persisted = store.collection_quarantine(race.race_id)
    assert persisted is not None
    assert persisted.code == "missing_field_evidence"
    before_counts = database_counts(store)
    before_artifacts = tuple(
        sorted(path.relative_to(artifacts.root) for path in artifacts.root.rglob("*"))
    )
    observations_consumed = False

    def replay_observations():
        nonlocal observations_consumed
        observations_consumed = True
        yield from ()

    arguments["observations"] = replay_observations()
    monkeypatch.setattr(
        type(artifacts),
        "verify",
        lambda _artifacts, *_: pytest.fail("artifact read"),
    )
    monkeypatch.setattr(
        type(artifacts),
        "put",
        lambda _artifacts, *_args, **_kwargs: pytest.fail("artifact write"),
    )

    with pytest.raises(SealingQuarantine) as raised:
        EvidenceSealer(store, artifacts).seal(**arguments)

    assert observations_consumed
    assert str(raised.value) == persisted.details
    assert database_counts(store) == before_counts
    after_artifacts = tuple(
        sorted(path.relative_to(artifacts.root) for path in artifacts.root.rglob("*"))
    )
    assert after_artifacts == before_artifacts


def test_valid_effective_evidence_quarantine_intent_replays_exactly_without_io(setup, monkeypatch):
    store, _, _, race, artifacts = setup
    card = artifacts.put(b"quarantine-effective-card", media_type="application/json")
    effective_observations = (
        FieldObservation(
            EvidenceField.VENUE,
            "Ballarat",
            EvidenceAuthority.OFFICIAL_CARD,
            True,
            "card",
            card.checksum,
        ),
    )
    persist_fields(store, race.race_id, effective_observations, 4200)
    arguments = {
        "operation_id": op(4201),
        "race_id": race.race_id,
        "source_artifacts": {},
        "observations": [],
        "scheduled_jump": race.scheduled_jump,
        "buffer": timedelta(minutes=2),
        "schema_version": "v1",
        "normalization_version": "n1",
        "sealed_at": NOW,
    }
    cause = "no valid odds observation before feature-freeze cutoff"
    with pytest.raises(SealingQuarantine, match=cause):
        EvidenceSealer(store, artifacts).seal(**arguments)
    quarantined = store.quarantined_seal(op(4201), race.race_id)
    assert quarantined is not None
    assert quarantined.request_intent_digest == empty_caller_effective_digest(
        race.race_id,
        {"card": card.checksum},
        effective_observations,
        race.scheduled_jump,
        timedelta(minutes=2),
        "v1",
        "n1",
        NOW,
    )
    before_database = database_counts(store)
    before_artifacts = artifact_bytes(artifacts)
    monkeypatch.setattr(type(artifacts), "verify", lambda *_: pytest.fail("artifact verify"))
    monkeypatch.setattr(
        type(artifacts), "put", lambda *_args, **_kwargs: pytest.fail("artifact put")
    )
    with closing(sqlite3.connect(store.path)) as observer:
        before_data_version = observer.execute("PRAGMA data_version").fetchone()[0]
        with pytest.raises(SealingQuarantine) as replayed:
            EvidenceSealer(store, artifacts).seal(**arguments)
        after_data_version = observer.execute("PRAGMA data_version").fetchone()[0]
    assert after_data_version == before_data_version
    assert str(replayed.value) == cause
    assert database_counts(store) == before_database
    assert artifact_bytes(artifacts) == before_artifacts


def test_orphaned_sealing_quarantine_fails_before_inputs_or_writes(setup, monkeypatch):
    store, _, _, race, artifacts = setup
    arguments = {
        "operation_id": op(4202),
        "race_id": race.race_id,
        "source_artifacts": {},
        "observations": [],
        "scheduled_jump": race.scheduled_jump,
        "buffer": timedelta(minutes=2),
        "schema_version": "v1",
        "normalization_version": "n1",
        "sealed_at": NOW,
    }
    with pytest.raises(SealingQuarantine):
        EvidenceSealer(store, artifacts).seal(**arguments)
    with sqlite3.connect(store.path) as corruptor:
        corruptor.execute("PRAGMA foreign_keys=OFF")
        corruptor.execute("DELETE FROM operations WHERE operation_id=?", (str(op(4202)),))
    with store._connect() as db:
        assert db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='trigger' "
            "AND name='collection_quarantines_append_only_delete'"
        ).fetchone()

    class UnconsumableObservations:
        def __iter__(self):
            raise AssertionError("orphaned authority consumed observations")

    arguments["observations"] = UnconsumableObservations()
    monkeypatch.setattr(type(artifacts), "verify", lambda *_: pytest.fail("artifact verify"))
    monkeypatch.setattr(
        type(artifacts), "put", lambda *_args, **_kwargs: pytest.fail("artifact put")
    )
    before_database = database_counts(store)
    before_artifacts = artifact_bytes(artifacts)
    with closing(sqlite3.connect(store.path)) as observer:
        before_data_version = observer.execute("PRAGMA data_version").fetchone()[0]
        with pytest.raises(OperationsStoreError, match="has no owning operation"):
            EvidenceSealer(store, artifacts).seal(**arguments)
        after_data_version = observer.execute("PRAGMA data_version").fetchone()[0]
    assert after_data_version == before_data_version
    assert database_counts(store) == before_database
    assert artifact_bytes(artifacts) == before_artifacts


@pytest.mark.parametrize(
    ("damage", "column", "value"),
    [
        ("intrinsic criticality disagreement", "critical", 0),
        ("malformed value JSON", "value_json", "not-json"),
        ("unknown field name", "field_name", "venuue"),
    ],
)
def test_corrupt_field_evidence_quarantine_replays_exactly_without_io(
    setup, monkeypatch, damage, column, value
):
    store, _, _, race, artifacts = setup
    card = artifacts.put(b"corrupt-field-card", media_type="application/json")
    observation = FieldObservation(
        EvidenceField.VENUE,
        "Ballarat",
        EvidenceAuthority.OFFICIAL_CARD,
        True,
        "card",
        card.checksum,
    )
    persist_fields(store, race.race_id, (observation,), 4300)
    with sqlite3.connect(store.path) as corruptor:
        assert corruptor.execute(
            "SELECT 1 FROM sqlite_master WHERE type='trigger' "
            "AND name='field_evidence_append_only_update'"
        ).fetchone()
        corruptor.execute("DROP TRIGGER field_evidence_append_only_update")
        corruptor.execute(
            f'UPDATE field_evidence SET "{column}"=? WHERE race_id=?',
            (value, str(race.race_id)),
        )
    arguments = {
        "operation_id": op(4301),
        "race_id": race.race_id,
        "source_artifacts": {},
        "observations": [],
        "scheduled_jump": race.scheduled_jump,
        "buffer": timedelta(minutes=2),
        "schema_version": "v1",
        "normalization_version": "n1",
        "sealed_at": NOW,
    }
    with pytest.raises(SealingQuarantine) as first:
        EvidenceSealer(store, artifacts).seal(**arguments)
    persisted = store.quarantined_seal(op(4301), race.race_id)
    assert persisted is not None, damage
    assert persisted.details == str(first.value)
    before_database = database_counts(store)
    before_artifacts = artifact_bytes(artifacts)
    monkeypatch.setattr(type(artifacts), "verify", lambda *_: pytest.fail("artifact verify"))
    monkeypatch.setattr(
        type(artifacts), "put", lambda *_args, **_kwargs: pytest.fail("artifact put")
    )
    with closing(sqlite3.connect(store.path)) as observer:
        before_data_version = observer.execute("PRAGMA data_version").fetchone()[0]
        with pytest.raises(SealingQuarantine) as replayed:
            EvidenceSealer(store, artifacts).seal(**arguments)
        after_data_version = observer.execute("PRAGMA data_version").fetchone()[0]
    assert after_data_version == before_data_version
    assert str(replayed.value) == str(first.value)
    assert database_counts(store) == before_database
    assert artifact_bytes(artifacts) == before_artifacts


@pytest.mark.parametrize(
    "dimension",
    [
        "race",
        "source_mapping",
        "source_checksum",
        "observations",
        "scheduled_jump",
        "buffer",
        "schema_version",
        "normalization_version",
        "sealed_at",
    ],
)
def test_quarantined_seal_changed_request_intent_conflicts_without_io(
    setup, monkeypatch, dimension
):
    store, _, _, race, artifacts = setup
    source_checksum = ArtifactChecksum("sha256:" + "1" * 64)
    arguments = {
        "operation_id": op(350),
        "race_id": race.race_id,
        "source_artifacts": {"card": source_checksum},
        "observations": [],
        "scheduled_jump": race.scheduled_jump,
        "buffer": timedelta(minutes=2),
        "schema_version": "v1",
        "normalization_version": "n1",
        "sealed_at": NOW,
    }
    with pytest.raises(SealingQuarantine, match="no transactional field evidence"):
        EvidenceSealer(store, artifacts).seal(**arguments)
    if dimension == "race":
        arguments["race_id"] = RaceId(ident("race", 9998))

        def observations_if_consumed():
            raise AssertionError("wrong-race replay consumed observations")
            yield

        arguments["observations"] = observations_if_consumed()
    elif dimension == "source_mapping":
        arguments["source_artifacts"] = {"renamed-card": source_checksum}
    elif dimension == "source_checksum":
        arguments["source_artifacts"] = {"card": ArtifactChecksum("sha256:" + "2" * 64)}
    elif dimension == "observations":
        arguments["observations"] = [
            FieldObservation(
                EvidenceField.VENUE,
                "Ballarat",
                EvidenceAuthority.OFFICIAL_CARD,
                True,
                "card",
                ArtifactChecksum("sha256:" + "3" * 64),
            )
        ]
    elif dimension == "scheduled_jump":
        arguments["scheduled_jump"] += timedelta(microseconds=1)
    elif dimension == "buffer":
        arguments["buffer"] = timedelta(minutes=3)
    elif dimension == "schema_version":
        arguments["schema_version"] = "v2"
    elif dimension == "normalization_version":
        arguments["normalization_version"] = "n2"
    else:
        arguments["sealed_at"] += timedelta(microseconds=1)
    before_counts = database_counts(store)
    before_artifacts = artifact_bytes(artifacts)
    monkeypatch.setattr(
        type(artifacts),
        "verify",
        lambda _artifacts, *_: pytest.fail("artifact read"),
    )
    monkeypatch.setattr(
        type(artifacts),
        "put",
        lambda _artifacts, *_args, **_kwargs: pytest.fail("artifact write"),
    )
    with pytest.raises(ConflictingOperation):
        EvidenceSealer(store, artifacts).seal(**arguments)
    assert database_counts(store) == before_counts
    assert artifact_bytes(artifacts) == before_artifacts


def test_wrong_operation_kind_rejects_before_observation_iteration(setup):
    store, _, _, race, artifacts = setup

    class UnconsumableObservations:
        def __iter__(self):
            raise AssertionError("wrong-kind replay consumed observations")

    with pytest.raises(ConflictingOperation):
        EvidenceSealer(store, artifacts).seal(
            operation_id=op(1),
            race_id=race.race_id,
            source_artifacts={},
            observations=UnconsumableObservations(),
            scheduled_jump=race.scheduled_jump,
            buffer=timedelta(minutes=2),
            schema_version="v1",
            normalization_version="n1",
            sealed_at=NOW,
        )


@pytest.mark.parametrize("damage", ["null", "malformed", "missing_row", "inconsistent"])
def test_invalid_legacy_quarantined_seal_intent_fails_closed(setup, damage):
    store, _, _, race, artifacts = setup
    arguments = {
        "operation_id": op(351),
        "race_id": race.race_id,
        "source_artifacts": {},
        "observations": [],
        "scheduled_jump": race.scheduled_jump,
        "buffer": timedelta(minutes=2),
        "schema_version": "v1",
        "normalization_version": "n1",
        "sealed_at": NOW,
    }
    with pytest.raises(SealingQuarantine):
        EvidenceSealer(store, artifacts).seal(**arguments)
    with sqlite3.connect(store.path) as db:
        if damage == "missing_row":
            db.execute("DROP TRIGGER collection_quarantines_append_only_delete")
            db.execute(
                "DELETE FROM collection_quarantines WHERE operation_id=?",
                (str(op(351)),),
            )
        else:
            db.execute("DROP TRIGGER collection_quarantines_append_only_update")
            if damage in {"null", "malformed"}:
                db.execute("PRAGMA ignore_check_constraints=ON")
            value = {
                "null": None,
                "malformed": "not-a-digest",
                "inconsistent": "sha256:" + "f" * 64,
            }[damage]
            db.execute(
                "UPDATE collection_quarantines SET request_intent_digest=? " "WHERE operation_id=?",
                (value, str(op(351))),
            )

    class UnconsumableObservations:
        def __iter__(self):
            raise AssertionError("invalid durable authority consumed observations")

    arguments["observations"] = UnconsumableObservations()
    with pytest.raises(OperationsStoreError):
        EvidenceSealer(store, artifacts).seal(**arguments)


@pytest.mark.parametrize(
    "values",
    [
        {"status": "failed"},
        {
            "status": OddsAttemptStatus.FAILED,
            "runner_mapping_checksum": ArtifactChecksum("sha256:" + "2" * 64),
        },
    ],
)
def test_odds_observation_contract_fails_before_sql(values):
    arguments = {
        "operation_id": op(110),
        "race_id": RaceId(ident("race", 1)),
        "source": "market",
        "attempted_at": NOW,
        "status": OddsAttemptStatus.FAILED,
        "error": "timeout",
    }
    arguments.update(values)
    with pytest.raises(DomainValidationError):
        OddsObservation(**arguments)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("field", "runner_set"),
        ("field", "unknown_field"),
        ("field", "runer_set"),
        ("source", 3),
        ("authority", "official_card"),
        ("critical", 1),
        ("artifact_checksum", "sha256:" + "1" * 64),
        ("value", {1, 2}),
    ],
)
def test_field_observation_contract_rejects_malformed_values(field, value):
    arguments = {
        "field": EvidenceField.RUNNER_SET,
        "value": ["A"],
        "authority": EvidenceAuthority.OFFICIAL_CARD,
        "critical": True,
        "source": "card",
        "artifact_checksum": ArtifactChecksum("sha256:" + "1" * 64),
    }
    arguments[field] = value
    with pytest.raises(DomainValidationError):
        FieldObservation(**arguments)


def test_invalid_freeze_authority_fails_closed_without_database_changes(setup):
    store, _, _, race, _ = setup
    store.advance_race(op(25), race.race_id, RaceState.CARD_COLLECTED, NOW)
    store.advance_race(op(26), race.race_id, RaceState.COLLECTING_ODDS, NOW)
    before = database_counts(store)

    with pytest.raises(OperationsStoreError, match="freeze_authority must be a FreezeAuthority"):
        store.seal_evidence(
            op(27),
            race_id=race.race_id,
            raw_checksum=ArtifactChecksum("sha256:" + "2" * 64),
            normalized_checksum=ArtifactChecksum("sha256:" + "3" * 64),
            schema_version="race-evidence-v1",
            normalization_version="normalizer-v1",
            frozen_at=NOW,
            freeze_authority="operator_override",  # type: ignore[arg-type]
            odds_checksum=ArtifactChecksum("sha256:" + "4" * 64),
            sealed_at=NOW,
            request_intent_digest=ArtifactChecksum("sha256:" + "5" * 64),
        )

    assert database_counts(store) == before


@pytest.mark.parametrize("field", ["raw_checksum", "normalized_checksum", "odds_checksum"])
def test_seal_evidence_rejects_malformed_string_checksums(setup, field):
    store, _, _, race, _ = setup
    store.advance_race(op(110), race.race_id, RaceState.CARD_COLLECTED, NOW)
    store.advance_race(op(111), race.race_id, RaceState.COLLECTING_ODDS, NOW)
    before = database_counts(store)
    values = {
        "raw_checksum": ArtifactChecksum("sha256:" + "2" * 64),
        "normalized_checksum": ArtifactChecksum("sha256:" + "3" * 64),
        "odds_checksum": ArtifactChecksum("sha256:" + "4" * 64),
    }
    values[field] = "sha256:" + "5" * 64

    with pytest.raises(OperationsStoreError, match=f"{field} must be an ArtifactChecksum"):
        store.seal_evidence(
            op(112),
            race_id=race.race_id,
            **values,
            schema_version="v1",
            normalization_version="v1",
            frozen_at=NOW,
            freeze_authority=FreezeAuthority.SCHEDULED_MINUS_BUFFER,
            sealed_at=NOW,
            request_intent_digest=ArtifactChecksum("sha256:" + "6" * 64),
        )

    assert database_counts(store) == before


def test_caller_programme_jump_mismatch_quarantines_and_read_exposes_persisted_jump(
    setup,
):
    store, repository, _, race, artifacts = setup
    store.advance_race(op(113), race.race_id, RaceState.CARD_COLLECTED, NOW)
    store.advance_race(op(114), race.race_id, RaceState.COLLECTING_ODDS, NOW)
    odds = artifacts.put(b"freeze odds", media_type="application/json")
    mapping = artifacts.put(b"freeze mapping", media_type="application/json")
    repository.record_odds_attempt(
        OddsObservation(
            op(115),
            race.race_id,
            "market",
            NOW,
            OddsAttemptStatus.SUCCEEDED,
            odds.checksum,
            mapping.checksum,
        )
    )
    repository.record_field_evidence(
        FieldEvidence(
            op(2099),
            race.race_id,
            EvidenceField.VENUE,
            EvidenceAuthority.OFFICIAL_CARD,
            "Ballarat",
            "odds",
            odds.checksum,
            NOW,
        )
    )
    sealer = EvidenceSealer(store, artifacts)
    with pytest.raises(SealingQuarantine, match="does not match persisted"):
        sealer.seal(
            operation_id=op(116),
            race_id=race.race_id,
            source_artifacts={},
            observations=[],
            scheduled_jump=race.scheduled_jump + timedelta(minutes=1),
            buffer=timedelta(minutes=2),
            schema_version="v1",
            normalization_version="v1",
            sealed_at=NOW,
        )
    assert store.count("collection_quarantines") == 1
    persisted = store.expected_programme_artifact(race.race_id)
    assert persisted.scheduled_jump == race.scheduled_jump


def test_terminal_collection_quarantine_rejects_new_odds_but_preserves_replay(setup):
    store, repository, _, race, _ = setup
    failed = OddsObservation(
        op(118), race.race_id, "market", NOW, OddsAttemptStatus.FAILED, error="timeout"
    )
    assert repository.record_odds_attempt(failed)
    repository.quarantine_collection(
        op(119),
        race.race_id,
        stage="collection",
        code="hard_cutoff",
        details="collection terminal",
        at=NOW,
    )
    before = database_counts(store)

    with pytest.raises(OperationsStoreError, match="terminal quarantine"):
        repository.record_odds_attempt(
            OddsObservation(
                op(120),
                race.race_id,
                "market",
                NOW + timedelta(minutes=1),
                OddsAttemptStatus.FAILED,
                error="late retry",
            )
        )

    assert database_counts(store) == before
    assert repository.record_odds_attempt(failed) is False


def test_direct_store_terminal_quarantine_blocks_fresh_seal_transactionally(setup):
    store, repository, _, race, _ = setup
    store.advance_race(op(121), race.race_id, RaceState.CARD_COLLECTED, NOW)
    store.advance_race(op(122), race.race_id, RaceState.COLLECTING_ODDS, NOW)
    repository.quarantine_collection(
        op(123),
        race.race_id,
        stage="collection",
        code="hard_cutoff",
        details="terminal",
        at=NOW,
    )
    before = database_counts(store)
    with pytest.raises(CollectionQuarantineBlocksSeal, match="blocks sealing"):
        store.seal_evidence(
            op(124),
            race_id=race.race_id,
            raw_checksum=ArtifactChecksum("sha256:" + "1" * 64),
            normalized_checksum=ArtifactChecksum("sha256:" + "2" * 64),
            schema_version="v1",
            normalization_version="v1",
            frozen_at=NOW,
            freeze_authority=FreezeAuthority.SCHEDULED_MINUS_BUFFER,
            odds_checksum=ArtifactChecksum("sha256:" + "3" * 64),
            sealed_at=NOW,
            request_intent_digest=ArtifactChecksum("sha256:" + "4" * 64),
        )
    assert database_counts(store) == before
    assert store.race_state(race.race_id) is RaceState.COLLECTING_ODDS


def test_evidence_sealer_quarantine_guard_writes_no_artifacts(setup):
    store, repository, _, race, artifacts = setup
    repository.quarantine_collection(
        op(125),
        race.race_id,
        stage="collection",
        code="hard_cutoff",
        details="terminal",
        at=NOW,
    )
    before = sorted(
        path.relative_to(artifacts.root) for path in artifacts.root.rglob("*") if path.is_file()
    )
    with pytest.raises(SealingQuarantine, match="terminal"):
        EvidenceSealer(store, artifacts).seal(
            operation_id=op(126),
            race_id=race.race_id,
            source_artifacts={},
            observations=[],
            scheduled_jump=race.scheduled_jump,
            buffer=timedelta(minutes=2),
            schema_version="v1",
            normalization_version="v1",
            sealed_at=NOW,
        )
    after = sorted(
        path.relative_to(artifacts.root) for path in artifacts.root.rglob("*") if path.is_file()
    )
    assert after == before
    assert store.count("sealed_evidence") == 0


def test_intrinsic_critical_fields_cannot_be_declassified(setup):
    _, _, _, _, artifacts = setup
    card = artifacts.put(b"critical card", media_type="application/json")
    for field, critical in (
        (EvidenceField.RUNNER_SET, False),
        (EvidenceField.DISTANCE, True),
    ):
        with pytest.raises(DomainValidationError, match="must match intrinsic field criticality"):
            FieldObservation(
                field,
                ["A"] if field is EvidenceField.RUNNER_SET else 450,
                EvidenceAuthority.OFFICIAL_CARD,
                critical,
                "card",
                card.checksum,
            )

    with pytest.raises(SealingQuarantine, match="critical conflict for runner_set"):
        normalize_fields(
            [
                FieldObservation(
                    EvidenceField.RUNNER_SET,
                    ["A"],
                    EvidenceAuthority.OFFICIAL_CARD,
                    True,
                    "card",
                    card.checksum,
                ),
                FieldObservation(
                    EvidenceField.RUNNER_SET,
                    ["B"],
                    EvidenceAuthority.OFFICIAL_CARD,
                    True,
                    "card",
                    card.checksum,
                ),
            ]
        )


@pytest.mark.parametrize("corrupt", ["odds", "mapping"])
def test_corrupt_unselected_retained_odds_artifact_quarantines_once_on_replay(setup, corrupt):
    store, repository, _, race, artifacts = setup
    store.advance_race(op(60), race.race_id, RaceState.CARD_COLLECTED, NOW)
    store.advance_race(op(61), race.race_id, RaceState.COLLECTING_ODDS, NOW)
    old_odds = artifacts.put(b"old-odds", media_type="application/json")
    old_mapping = artifacts.put(b"old-mapping", media_type="application/json")
    new_odds = artifacts.put(b"new-odds", media_type="application/json")
    new_mapping = artifacts.put(b"new-mapping", media_type="application/json")
    repository.record_odds_attempt(
        OddsObservation(
            op(62),
            race.race_id,
            "market",
            NOW,
            OddsAttemptStatus.SUCCEEDED,
            old_odds.checksum,
            old_mapping.checksum,
        )
    )
    repository.record_odds_attempt(
        OddsObservation(
            op(63),
            race.race_id,
            "market",
            NOW + timedelta(minutes=1),
            OddsAttemptStatus.SUCCEEDED,
            new_odds.checksum,
            new_mapping.checksum,
        )
    )
    repository.record_field_evidence(
        FieldEvidence(
            op(2065),
            race.race_id,
            EvidenceField.VENUE,
            EvidenceAuthority.OFFICIAL_CARD,
            "Ballarat",
            "odds",
            new_odds.checksum,
            NOW,
        )
    )
    damaged = old_odds if corrupt == "odds" else old_mapping
    artifacts.path_for(damaged.checksum).write_bytes(b"corrupt")
    sealer = EvidenceSealer(store, artifacts)
    for _ in range(2):
        with pytest.raises(SealingQuarantine, match="computed as"):
            sealer.seal(
                operation_id=op(64),
                race_id=race.race_id,
                source_artifacts={},
                observations=[],
                scheduled_jump=race.scheduled_jump,
                buffer=timedelta(minutes=2),
                schema_version="v1",
                normalization_version="v1",
                sealed_at=NOW + timedelta(minutes=6),
            )
    assert store.count("sealed_evidence") == 0
    assert store.count("collection_quarantines") == 1


def test_no_pre_freeze_odds_is_durably_quarantined(setup):
    store, repository, _, race, artifacts = setup
    store.advance_race(op(70), race.race_id, RaceState.CARD_COLLECTED, NOW)
    store.advance_race(op(71), race.race_id, RaceState.COLLECTING_ODDS, NOW)
    odds = artifacts.put(b"late-odds", media_type="application/json")
    mapping = artifacts.put(b"late-mapping", media_type="application/json")
    repository.record_odds_attempt(
        OddsObservation(
            op(72),
            race.race_id,
            "market",
            NOW + timedelta(minutes=10),
            OddsAttemptStatus.SUCCEEDED,
            odds.checksum,
            mapping.checksum,
        )
    )
    observations = [
        FieldObservation(
            EvidenceField.ACTUAL_JUMP,
            NOW + timedelta(minutes=5),
            EvidenceAuthority.OFFICIAL_JUMP,
            True,
            "jump",
            odds.checksum,
        )
    ]
    persist_fields(store, race.race_id, observations, 2074)
    with pytest.raises(SealingQuarantine, match="no valid odds"):
        EvidenceSealer(store, artifacts).seal(
            operation_id=op(73),
            race_id=race.race_id,
            scheduled_jump=race.scheduled_jump,
            observations=observations,
            source_artifacts={"jump": odds.checksum},
            buffer=timedelta(minutes=2),
            schema_version="v1",
            normalization_version="v1",
            sealed_at=NOW + timedelta(minutes=6),
        )
    assert store.count("collection_quarantines") == 1


def test_lower_authority_jump_cannot_control_freeze(setup):
    store, repository, _, race, artifacts = setup
    store.advance_race(op(74), race.race_id, RaceState.CARD_COLLECTED, NOW)
    store.advance_race(op(75), race.race_id, RaceState.COLLECTING_ODDS, NOW)
    odds = artifacts.put(b"odds", media_type="application/json")
    mapping = artifacts.put(b"mapping", media_type="application/json")
    card = artifacts.put(b"card", media_type="application/json")
    repository.record_odds_attempt(
        OddsObservation(
            op(76),
            race.race_id,
            "market",
            NOW,
            OddsAttemptStatus.SUCCEEDED,
            odds.checksum,
            mapping.checksum,
        )
    )
    observations = [
        FieldObservation(
            EvidenceField.ACTUAL_JUMP,
            NOW + timedelta(minutes=1),
            EvidenceAuthority.OFFICIAL_CARD,
            True,
            "card",
            card.checksum,
        )
    ]
    persist_fields(store, race.race_id, observations, 2020)
    result = EvidenceSealer(store, artifacts).seal(
        operation_id=op(77),
        race_id=race.race_id,
        source_artifacts={"card": card.checksum},
        observations=observations,
        scheduled_jump=race.scheduled_jump,
        buffer=timedelta(minutes=2),
        schema_version="v1",
        normalization_version="v1",
        sealed_at=NOW,
    )
    assert result.freeze_authority is FreezeAuthority.SCHEDULED_MINUS_BUFFER
    assert result.frozen_at == race.scheduled_jump - timedelta(minutes=2)


@pytest.mark.parametrize("proof", ["missing", "mismatched_caller"])
def test_unproven_official_jump_fails_closed(setup, proof):
    store, repository, _, race, artifacts = setup
    store.advance_race(op(78), race.race_id, RaceState.CARD_COLLECTED, NOW)
    store.advance_race(op(79), race.race_id, RaceState.COLLECTING_ODDS, NOW)
    artifact = artifacts.put(b"jump", media_type="application/json")
    if proof == "mismatched_caller":
        repository.record_field_evidence(
            FieldEvidence(
                op(2102),
                race.race_id,
                EvidenceField.ACTUAL_JUMP,
                EvidenceAuthority.OFFICIAL_JUMP,
                NOW,
                "jump",
                artifact.checksum,
                NOW,
            )
        )
    expected = (
        "no transactional field evidence"
        if proof == "missing"
        else "caller field observations do not exactly match transactional evidence"
    )
    sealer = EvidenceSealer(store, artifacts)
    for _ in range(2):
        with pytest.raises(SealingQuarantine, match=expected):
            sealer.seal(
                operation_id=op(80),
                race_id=race.race_id,
                source_artifacts={"jump": artifact.checksum},
                observations=[
                    FieldObservation(
                        EvidenceField.ACTUAL_JUMP,
                        NOW + timedelta(seconds=1),
                        EvidenceAuthority.OFFICIAL_JUMP,
                        True,
                        "jump",
                        artifact.checksum,
                    )
                ],
                scheduled_jump=race.scheduled_jump,
                buffer=timedelta(minutes=2),
                schema_version="v1",
                normalization_version="v1",
                sealed_at=NOW,
            )
    assert store.count("sealed_evidence") == 0
    assert store.count("collection_quarantines") == 1


def test_corrupt_persisted_non_jump_field_artifact_quarantines_once(setup):
    store, repository, _, race, artifacts = setup
    store.advance_race(op(87), race.race_id, RaceState.CARD_COLLECTED, NOW)
    store.advance_race(op(88), race.race_id, RaceState.COLLECTING_ODDS, NOW)
    card = artifacts.put(b"card-evidence", media_type="application/json")
    repository.record_field_evidence(
        FieldEvidence(
            op(2125),
            race.race_id,
            EvidenceField.RUNNER_SET,
            EvidenceAuthority.OFFICIAL_CARD,
            ["A", "B"],
            "card",
            card.checksum,
            NOW,
        )
    )
    artifacts.path_for(card.checksum).write_bytes(b"corrupt")
    sealer = EvidenceSealer(store, artifacts)
    for _ in range(2):
        with pytest.raises(SealingQuarantine):
            sealer.seal(
                operation_id=op(89),
                race_id=race.race_id,
                source_artifacts={},
                observations=[],
                scheduled_jump=race.scheduled_jump,
                buffer=timedelta(minutes=2),
                schema_version="v1",
                normalization_version="v1",
                sealed_at=NOW,
            )
    assert store.count("collection_quarantines") == 1
    assert store.count("sealed_evidence") == 0


def test_corrupt_official_jump_proof_is_quarantined_and_not_sealed(setup):
    store, _, _, race, artifacts = setup
    store.advance_race(op(81), race.race_id, RaceState.CARD_COLLECTED, NOW)
    store.advance_race(op(82), race.race_id, RaceState.COLLECTING_ODDS, NOW)
    proof = artifacts.put(b"jump-proof", media_type="application/json")
    observations = [
        FieldObservation(
            EvidenceField.ACTUAL_JUMP,
            NOW + timedelta(minutes=5),
            EvidenceAuthority.OFFICIAL_JUMP,
            True,
            "jump",
            proof.checksum,
        )
    ]
    persist_fields(store, race.race_id, observations, 2083)
    artifacts.path_for(proof.checksum).write_bytes(b"corrupt")
    with pytest.raises(SealingQuarantine, match="computed as"):
        EvidenceSealer(store, artifacts).seal(
            operation_id=op(83),
            race_id=race.race_id,
            source_artifacts={"jump": proof.checksum},
            observations=observations,
            scheduled_jump=race.scheduled_jump,
            buffer=timedelta(minutes=2),
            schema_version="v1",
            normalization_version="v1",
            sealed_at=NOW,
        )
    assert store.count("collection_quarantines") == 1
    assert store.count("sealed_evidence") == 0


def test_equal_official_jump_conflict_is_quarantined_and_not_sealed(setup):
    store, _, _, race, artifacts = setup
    store.advance_race(op(84), race.race_id, RaceState.CARD_COLLECTED, NOW)
    store.advance_race(op(85), race.race_id, RaceState.COLLECTING_ODDS, NOW)
    first = artifacts.put(b"jump-one", media_type="application/json")
    second = artifacts.put(b"jump-two", media_type="application/json")
    observations = [
        FieldObservation(
            EvidenceField.ACTUAL_JUMP,
            NOW + timedelta(minutes=4),
            EvidenceAuthority.OFFICIAL_JUMP,
            True,
            "jump-a",
            first.checksum,
        ),
        FieldObservation(
            EvidenceField.ACTUAL_JUMP,
            NOW + timedelta(minutes=5),
            EvidenceAuthority.OFFICIAL_JUMP,
            True,
            "jump-b",
            second.checksum,
        ),
    ]
    persist_fields(store, race.race_id, observations, 2086)
    with pytest.raises(SealingQuarantine, match="critical conflict for actual_jump"):
        EvidenceSealer(store, artifacts).seal(
            operation_id=op(86),
            race_id=race.race_id,
            source_artifacts={"jump-a": first.checksum, "jump-b": second.checksum},
            observations=observations,
            scheduled_jump=race.scheduled_jump,
            buffer=timedelta(minutes=2),
            schema_version="v1",
            normalization_version="v1",
            sealed_at=NOW,
        )
    assert store.count("collection_quarantines") == 1
    assert store.count("sealed_evidence") == 0


def test_phase2_modules_use_only_public_operations_store_boundary():
    for path in ("race_collection/collection.py", "race_collection/sealing.py"):
        source = Path(path).read_text()
        assert "._connect" not in source
        assert "._operation" not in source
        assert ".execute(" not in source
        assert "SQLiteOperationsStore" not in source


def test_critical_equal_authority_conflict_blocks_sealing(setup):
    store, repository, day, race, artifacts = setup
    store.advance_race(op(30), race.race_id, RaceState.CARD_COLLECTED, NOW)
    store.advance_race(op(31), race.race_id, RaceState.COLLECTING_ODDS, NOW)
    odds = artifacts.put(b"odds", media_type="application/json")
    mapping = artifacts.put(b"mapping", media_type="application/json")
    card = artifacts.put(b"card", media_type="application/json")
    repository.record_odds_attempt(
        OddsObservation(
            op(32),
            race.race_id,
            "market",
            NOW,
            OddsAttemptStatus.SUCCEEDED,
            odds.checksum,
            mapping.checksum,
        )
    )
    observations = [
        FieldObservation(
            EvidenceField.RUNNER_SET,
            ["A"],
            EvidenceAuthority.OFFICIAL_CARD,
            True,
            "card",
            card.checksum,
        ),
        FieldObservation(
            EvidenceField.RUNNER_SET,
            ["B"],
            EvidenceAuthority.OFFICIAL_CARD,
            True,
            "card",
            card.checksum,
        ),
    ]
    persist_fields(store, race.race_id, observations, 2034)
    with pytest.raises(SealingQuarantine, match="critical conflict"):
        EvidenceSealer(store, artifacts).seal(
            operation_id=op(33),
            race_id=race.race_id,
            source_artifacts={"card": card.checksum},
            observations=observations,
            scheduled_jump=race.scheduled_jump,
            buffer=timedelta(minutes=2),
            schema_version="v1",
            normalization_version="v1",
            sealed_at=NOW,
        )
    assert store.collection_rows(day) == (
        CollectionRaceRecord(race.race_id, RaceState.COLLECTING_ODDS, True),
    )


def test_reconciliation_uses_inventory_and_explicit_quarantine(setup):
    _, repository, day, race, _ = setup
    before = repository.reconcile(day)
    assert before.expected == 1 and not before.complete
    repository.quarantine_collection(
        op(40),
        race.race_id,
        stage="collection",
        code="hard_cutoff",
        details="card unavailable",
        at=NOW,
    )
    after = repository.reconcile(day)
    assert after.complete and after.quarantined == 1 and after.unresolved_race_ids == ()


def test_reconciliation_ignores_directly_discovered_non_inventory_race(setup):
    store, repository, day, expected, _ = setup
    stray = RaceId(ident("race", 120))
    store.discover_race(op(120), stray, day, NOW)
    report = repository.reconcile(day)
    assert report.expected == 1
    assert report.unresolved_race_ids == (str(expected.race_id),)
    assert str(stray) not in report.unresolved_race_ids
