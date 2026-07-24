import hashlib
import json
import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest

from race_collection.domain import (
    ArtifactChecksum,
    DogId,
    DogRun,
    OperationId,
    ProgrammeRaceCandidate,
    RaceId,
    RaceState,
    RacingDay,
    RacingDayId,
    Quarantine,
    RunObservation,
    Supersession,
)
from race_collection.operations import (
    BarrierNotSatisfied,
    ConflictingOperation,
    SQLiteOperationsStore,
    iso_timestamp,
)


NOW = datetime(2026, 7, 22, 1, 2, 3, tzinfo=timezone.utc)


def ident(prefix: str, number: int) -> str:
    return f"{prefix}_{number:032x}"


def op(number: int) -> OperationId:
    return OperationId(ident("op", number))


@pytest.fixture
def store(tmp_path):
    result = SQLiteOperationsStore(tmp_path / "operations.sqlite3")
    result.migrate()
    return result


def make_day(number: int = 1) -> RacingDay:
    return RacingDay(
        RacingDayId(ident("day", number)),
        date(2026, 7, 22),
        "Australia/Melbourne",
        NOW,
    )


def seed_race(store, *, day_number=1, race_number=1, first_op=1):
    day = make_day(day_number)
    store.create_racing_day(op(first_op), day)
    candidate = ProgrammeRaceCandidate(
        "official",
        f"day-{day_number}-race-{race_number}",
        "Ballarat",
        race_number,
        NOW,
    )
    race = store.record_expected_race(
        op(first_op + 1),
        day,
        candidate,
        ArtifactChecksum(f"sha256:{race_number:064x}"),
        NOW,
    )
    return day, race


def advance_to(store, race, target, first_op=100):
    path = [
        RaceState.CARD_COLLECTED,
        RaceState.COLLECTING_ODDS,
        RaceState.EVIDENCE_SEALED,
        RaceState.AWAITING_DAY_CLOSE,
    ]
    for offset, state in enumerate(path):
        store.advance_race(op(first_op + offset), race, state, NOW)
        if state == target:
            return


def add_expected_race(store, day, race_number, operation_number):
    return store.record_expected_race(
        op(operation_number),
        day,
        ProgrammeRaceCandidate(
            "official",
            f"day-{day.id}-race-{race_number}",
            "Ballarat",
            race_number,
            NOW,
        ),
        ArtifactChecksum(f"sha256:{race_number:064x}"),
        NOW,
    )


def test_empty_and_repeated_migrations_enable_wal_and_foreign_keys(tmp_path):
    store = SQLiteOperationsStore(tmp_path / "operations.sqlite3")
    store.migrate()
    store.migrate()
    with sqlite3.connect(store.path) as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert db.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 28
    with store._connect() as db:
        assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_store_connection_context_commits_rolls_back_and_closes(tmp_path):
    store = SQLiteOperationsStore(tmp_path / "connection-lifecycle.sqlite3")
    store.migrate()

    committed = store._connect()
    with committed as db:
        db.execute("BEGIN")
        db.execute(
            "INSERT INTO operations VALUES(?,?,?,?)",
            (str(op(1800)), "context_commit", "1" * 64, NOW.isoformat()),
        )
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        committed.execute("SELECT 1")
    with store._connect() as db:
        assert (
            db.execute(
                "SELECT kind FROM operations WHERE operation_id=?",
                (str(op(1800)),),
            ).fetchone()[0]
            == "context_commit"
        )

    rolled_back = store._connect()
    with pytest.raises(RuntimeError, match="rollback probe"):
        with rolled_back as db:
            db.execute("BEGIN")
            db.execute(
                "INSERT INTO operations VALUES(?,?,?,?)",
                (str(op(1801)), "context_rollback", "2" * 64, NOW.isoformat()),
            )
            raise RuntimeError("rollback probe")
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        rolled_back.execute("SELECT 1")
    with store._connect() as db:
        assert (
            db.execute(
                "SELECT 1 FROM operations WHERE operation_id=?",
                (str(op(1801)),),
            ).fetchone()
            is None
        )


def test_repeated_store_contexts_release_every_connection_and_manual_close_is_idempotent(
    tmp_path,
):
    store = SQLiteOperationsStore(tmp_path / "repeated-connections.sqlite3")
    store.migrate()
    released = []
    for _ in range(100):
        connection = store._connect()
        with connection as db:
            assert db.execute("SELECT 1").fetchone()[0] == 1
        released.append(connection)
    for connection in released:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")

    manual = store._connect()
    manual.close()
    manual.close()


def test_populated_schema_17_migrates_forward_to_latest_without_data_loss(tmp_path):
    store = SQLiteOperationsStore(tmp_path / "operations.sqlite3")
    all_scripts = SQLiteOperationsStore._migration_scripts(store)
    store._migration_scripts = lambda: tuple(script for script in all_scripts if script[0] <= 17)
    store.migrate()
    day = make_day(17)
    store.create_racing_day(op(1700), day)
    store.record_expected_race(
        op(1701),
        day,
        ProgrammeRaceCandidate("official", "legacy-race", "Ballarat", 1, NOW),
        ArtifactChecksum("sha256:" + "1" * 64),
        NOW,
    )

    del store._migration_scripts
    store.migrate()
    store.migrate()

    with store._connect() as db:
        assert db.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 28
        assert (
            db.execute(
                "SELECT source_race_id FROM expected_races WHERE source_race_id='legacy-race'"
            ).fetchone()[0]
            == "legacy-race"
        )
        trigger_names = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()
        }
        assert "phase7_probation_acceptance_exact_chain" in trigger_names
        assert "phase7_probation_seal_exact" in trigger_names


def test_populated_schema_27_partial_day_migration_preserves_prefix_and_defines_suffix(tmp_path):
    store = SQLiteOperationsStore(tmp_path / "partial-v27.sqlite3")
    all_scripts = SQLiteOperationsStore._migration_scripts(store)
    store._migration_scripts = lambda: tuple(script for script in all_scripts if script[0] <= 27)
    store.migrate()
    day = make_day(27)
    store.create_racing_day(op(2700), day)
    acquired = iso_timestamp(NOW)
    expires = iso_timestamp(NOW + timedelta(hours=1))
    with store._operation(op(2701), "phase7_acquire_scheduler_lease", {}) as (db, _):
        db.execute(
            "INSERT INTO phase7_scheduler_lease VALUES(1,?,?,?,?,?,?)",
            ("migration-owner", "migration-token", acquired, expires, 1, str(op(2701))),
        )
        db.execute(
            "INSERT INTO phase7_scheduler_history VALUES(?,?,?,?,?,?)",
            (1, "migration-owner", "migration-token", acquired, expires, str(op(2701))),
        )
    result_json = '{"source":"official"}'
    result_checksum = "sha256:" + "2" * 64
    with store._operation(op(2702), "phase7_command_discover_programme", {}) as (db, _):
        db.execute(
            "INSERT INTO phase7_application_command_receipts VALUES(?,?,?,?,?,?)",
            (
                str(op(2702)),
                str(day.id),
                "discover_programme",
                result_json,
                result_checksum,
                acquired,
            ),
        )
    with store._operation(op(2703), "phase7_advance_phase", {}) as (db, _):
        db.execute(
            "INSERT INTO phase7_scheduler_progress VALUES(?,?,?,?,?,?,?,?,?)",
            (
                str(day.id),
                1,
                "discover_programme",
                1,
                str(op(2702)),
                result_json,
                result_checksum,
                acquired,
                str(op(2703)),
            ),
        )

    del store._migration_scripts
    store.migrate()
    store.migrate()

    with store._connect() as db:
        plan = db.execute(
            "SELECT * FROM phase7_day_command_plan WHERE racing_day_id=? ORDER BY phase_ordinal",
            (str(day.id),),
        ).fetchall()
        assert len(plan) == 9
        assert plan[0]["command_operation_id"] == str(op(2702))
        assert len({row["command_operation_id"] for row in plan}) == 9
        assert [row["phase_ordinal"] for row in plan] == list(range(1, 10))
        assert all(row["planned_at"] == acquired for row in plan)
        provenance = db.execute(
            "SELECT kind,payload_sha256 FROM operations WHERE operation_id=?",
            (plan[0]["operation_id"],),
        ).fetchone()
        payload = {
            "completed_prefix": [
                {
                    "command_operation_id": str(op(2702)),
                    "completed_at": acquired,
                    "lease_generation": 1,
                    "migration_plan_anchor_at": acquired,
                    "phase_name": "discover_programme",
                    "phase_ordinal": 1,
                }
            ],
            "migrated_suffix": [
                {
                    "command_operation_id": row["command_operation_id"],
                    "lease_generation": 1,
                    "phase_name": row["phase_name"],
                    "phase_ordinal": row["phase_ordinal"],
                    "planned_at": acquired,
                }
                for row in plan[1:]
            ],
            "provenance_version": "phase7-v27-day-plan-migration-v1",
            "racing_day_id": str(day.id),
        }
        expected_digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert provenance["kind"] == "phase7_migrate_v27_day_command_plan"
        assert provenance["payload_sha256"] == expected_digest


def test_schema_28_refuses_inexact_v27_progress_and_rolls_back(tmp_path):
    store = SQLiteOperationsStore(tmp_path / "inexact-v27.sqlite3")
    all_scripts = SQLiteOperationsStore._migration_scripts(store)
    store._migration_scripts = lambda: tuple(script for script in all_scripts if script[0] <= 27)
    store.migrate()
    day = make_day(28)
    store.create_racing_day(op(2800), day)
    acquired = iso_timestamp(NOW)
    expires = iso_timestamp(NOW + timedelta(hours=1))
    with store._operation(op(2801), "phase7_acquire_scheduler_lease", {}) as (db, _):
        db.execute(
            "INSERT INTO phase7_scheduler_lease VALUES(1,?,?,?,?,?,?)",
            ("migration-owner", "migration-token", acquired, expires, 1, str(op(2801))),
        )
        db.execute(
            "INSERT INTO phase7_scheduler_history VALUES(?,?,?,?,?,?)",
            (1, "migration-owner", "migration-token", acquired, expires, str(op(2801))),
        )
    result_json = '{"source":"official"}'
    result_checksum = "sha256:" + "2" * 64
    with store._operation(op(2802), "phase7_command_discover_programme", {}) as (db, _):
        db.execute(
            "INSERT INTO phase7_application_command_receipts VALUES(?,?,?,?,?,?)",
            (
                str(op(2802)),
                str(day.id),
                "discover_programme",
                result_json,
                result_checksum,
                acquired,
            ),
        )
    with store._operation(op(2803), "phase7_advance_phase", {}) as (db, _):
        db.execute(
            "INSERT INTO phase7_scheduler_progress VALUES(?,?,?,?,?,?,?,?,?)",
            (
                str(day.id),
                1,
                "discover_programme",
                1,
                str(op(2802)),
                result_json,
                result_checksum,
                acquired,
                str(op(2803)),
            ),
        )
    with store._connect() as db:
        db.execute(
            "UPDATE phase7_scheduler_history SET expires_at=? WHERE generation=1",
            (acquired,),
        )

    del store._migration_scripts
    with pytest.raises(sqlite3.IntegrityError):
        store.migrate()
    with store._connect() as db:
        assert db.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == 27
        assert (
            db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='phase7_day_command_plan'"
            ).fetchone()
            is None
        )


def test_migration_failure_rolls_back_every_statement(tmp_path):
    class BrokenMigrationStore(SQLiteOperationsStore):
        def _migration_scripts(self):
            return (
                (
                    1,
                    "broken.sql",
                    b"CREATE TABLE partial(value TEXT);\n"
                    b"INSERT INTO missing_table VALUES (1);\n",
                ),
            )

    path = tmp_path / "operations.sqlite3"
    with pytest.raises(sqlite3.OperationalError, match="missing_table"):
        BrokenMigrationStore(path).migrate()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []


def test_operation_replay_is_idempotent(store):
    day = make_day()
    assert store.create_racing_day(op(1), day) is True
    assert store.create_racing_day(op(1), day) is False
    assert store.count("racing_days") == 1
    assert store.count("operations") == 1


def test_failed_transaction_rolls_back_operation_and_domain_rows(store):
    day, race_one = seed_race(store)
    race_two = RaceId(ident("race", 2))
    store.discover_race(op(3), race_two, day, NOW)
    store.add_race_alias(op(4), race_one, "fasttrack", "source-race-7", NOW)
    aliases_before_conflict = store.count("race_aliases")
    operations_before_conflict = store.count("operations")
    with pytest.raises(sqlite3.IntegrityError):
        store.add_race_alias(op(5), race_two, "fasttrack", "source-race-7", NOW)
    assert store.count("race_aliases") == aliases_before_conflict
    assert store.count("operations") == operations_before_conflict


def test_race_source_alias_is_unique_but_other_sources_may_share_value(store):
    day, race_one = seed_race(store)
    race_two = RaceId(ident("race", 2))
    store.discover_race(op(3), race_two, day, NOW)
    store.add_race_alias(op(4), race_one, "fasttrack", "123", NOW)
    store.add_race_alias(op(5), race_two, "sportsbet", "123", NOW)
    with pytest.raises(sqlite3.IntegrityError):
        store.add_race_alias(op(6), race_two, "fasttrack", "123", NOW)


def test_dog_run_is_unique_by_dog_and_local_racing_date(store):
    dog = DogId(ident("dog", 1))
    store.add_dog(op(1), dog, NOW)
    run = DogRun(dog, date(2026, 7, 22), False, NOW)
    store.add_dog_run(op(2), run)
    with pytest.raises(sqlite3.IntegrityError):
        store.add_dog_run(op(3), DogRun(dog, date(2026, 7, 22), True, NOW))
    assert store.count("dog_runs") == 1


def test_multiple_observations_preserve_sources_without_duplicating_run(store):
    dog = DogId(ident("dog", 1))
    store.add_dog(op(1), dog, NOW)
    store.add_dog_run(op(2), DogRun(dog, date(2026, 7, 22), False, NOW))
    for number, source in ((3, "embedded_form"), (4, "official_result")):
        store.add_run_observation(
            RunObservation(
                op(number),
                dog,
                date(2026, 7, 22),
                source,
                ArtifactChecksum(f"sha256:{number:064x}"),
                NOW,
                starts=1,
                wins=1,
            )
        )
    assert store.count("dog_runs") == 1
    assert store.count("run_observations") == 2


def test_day_close_and_prediction_batch_barriers_are_enforced(store):
    day, race_one = seed_race(store)
    race_two = add_expected_race(store, day, 2, 3)
    advance_to(store, race_one, RaceState.AWAITING_DAY_CLOSE, 10)
    with pytest.raises(BarrierNotSatisfied):
        store.close_racing_day(op(20), day, NOW)
    advance_to(store, race_two, RaceState.AWAITING_DAY_CLOSE, 30)
    store.close_racing_day(op(40), day, NOW)
    store.advance_race(op(41), race_one, RaceState.PREDICTION_PENDING, NOW)
    store.advance_race(op(42), race_one, RaceState.PREDICTION_COMMITTED, NOW)
    with pytest.raises(BarrierNotSatisfied):
        store.advance_race(op(43), race_one, RaceState.RESULT_PENDING, NOW)
    store.advance_race(op(44), race_two, RaceState.PREDICTION_PENDING, NOW)
    store.advance_race(op(45), race_two, RaceState.PREDICTION_QUARANTINED, NOW)
    store.advance_race(op(46), race_one, RaceState.RESULT_PENDING, NOW)
    assert store.race_state(race_one) == RaceState.RESULT_PENDING


def test_collection_quarantine_is_terminal_for_day_and_later_barriers(store):
    day, eligible = seed_race(store)
    terminal = add_expected_race(store, day, 2, 3)
    advance_to(store, eligible, RaceState.AWAITING_DAY_CLOSE, 10)
    assert store.record_collection_quarantine(
        op(20),
        terminal,
        stage="collection",
        code="hard_cutoff",
        details="card unavailable",
        at=NOW,
    )
    assert not store.record_collection_quarantine(
        op(20),
        terminal,
        stage="collection",
        code="hard_cutoff",
        details="card unavailable",
        at=NOW,
    )
    store.close_racing_day(op(21), day, NOW)
    store.advance_race(op(22), eligible, RaceState.PREDICTION_PENDING, NOW)
    store.advance_race(op(23), eligible, RaceState.PREDICTION_COMMITTED, NOW)
    store.advance_race(op(24), eligible, RaceState.RESULT_PENDING, NOW)
    assert store.race_state(terminal) == RaceState.DISCOVERED
    assert store.count("collection_quarantines") == 1


def test_all_collection_quarantined_day_closes_idempotently(store):
    day, race_one = seed_race(store)
    race_two = add_expected_race(store, day, 2, 3)
    for number, race in ((10, race_one), (11, race_two)):
        store.record_collection_quarantine(
            op(number),
            race,
            stage="collection",
            code="hard_cutoff",
            details="collection incomplete",
            at=NOW,
        )
    assert store.close_racing_day(op(12), day, NOW)
    assert not store.close_racing_day(op(12), day, NOW)


def test_unquarantined_incomplete_race_still_blocks_mixed_day_close(store):
    day, awaiting = seed_race(store)
    terminal = add_expected_race(store, day, 2, 3)
    incomplete = add_expected_race(store, day, 3, 4)
    advance_to(store, awaiting, RaceState.AWAITING_DAY_CLOSE, 10)
    store.record_collection_quarantine(
        op(20),
        terminal,
        stage="collection",
        code="hard_cutoff",
        details="collection incomplete",
        at=NOW,
    )
    with pytest.raises(BarrierNotSatisfied):
        store.close_racing_day(op(21), day, NOW)
    assert store.race_state(incomplete) == RaceState.DISCOVERED


def test_zero_expected_inventory_fails_closed(store):
    day = make_day()
    store.create_racing_day(op(1), day)
    with pytest.raises(BarrierNotSatisfied, match="all expected races"):
        store.close_racing_day(op(2), day, NOW)


def test_stray_non_inventory_race_does_not_block_or_enter_prediction(store):
    day, expected = seed_race(store)
    stray = RaceId(ident("race", 99))
    store.discover_race(op(3), stray, day, NOW)
    advance_to(store, expected, RaceState.AWAITING_DAY_CLOSE, 10)
    advance_to(store, stray, RaceState.AWAITING_DAY_CLOSE, 20)
    store.close_racing_day(op(30), day, NOW)
    with pytest.raises(BarrierNotSatisfied, match="only unquarantined expected races"):
        store.advance_race(op(31), stray, RaceState.PREDICTION_PENDING, NOW)


def test_rejected_result_before_prediction_rolls_back(store):
    _day, race = seed_race(store)
    with pytest.raises(Exception):
        store.advance_race(op(3), race, RaceState.RESULT_PENDING, NOW)
    assert store.race_state(race) == RaceState.DISCOVERED
    assert store.count("operations") == 2


def test_ephemeral_lifecycle_reaches_training_example_readiness(store):
    day, race = seed_race(store)
    advance_to(store, race, RaceState.AWAITING_DAY_CLOSE, 10)
    store.close_racing_day(op(20), day, NOW)
    for number, state in enumerate(
        (
            RaceState.PREDICTION_PENDING,
            RaceState.PREDICTION_COMMITTED,
            RaceState.RESULT_PENDING,
            RaceState.RESULT_COLLECTED,
            RaceState.TRAINING_EXAMPLE_READY,
        ),
        start=21,
    ):
        store.advance_race(op(number), race, state, NOW)
    assert store.race_state(race) == RaceState.TRAINING_EXAMPLE_READY


def test_quarantine_reason_and_state_commit_atomically(store):
    day, race = seed_race(store)
    advance_to(store, race, RaceState.AWAITING_DAY_CLOSE, 10)
    store.close_racing_day(op(20), day, NOW)
    store.advance_race(op(21), race, RaceState.PREDICTION_PENDING, NOW)
    store.record_quarantine(Quarantine(op(22), race, "prediction", "missing seal", NOW))
    assert store.race_state(race) == RaceState.PREDICTION_QUARANTINED
    assert store.count("quarantines") == 1


def test_supersession_is_append_only_and_does_not_mutate_prior_record(store):
    _day, prior = seed_race(store)
    replacement = RaceId(ident("race", 2))
    item = Supersession(op(3), prior, replacement, "corrected identity", NOW)
    store.record_supersession("race_identity_decision", item)
    assert store.count("supersessions") == 1
    assert store.race_state(prior) == RaceState.DISCOVERED


def test_wal_reader_sees_committed_snapshot_while_writer_is_open(store):
    day = make_day()
    store.create_racing_day(op(1), day)
    writer = store._connect()
    writer.execute("BEGIN IMMEDIATE")
    writer.execute(
        "INSERT INTO operations(operation_id, kind, payload_sha256, created_at) "
        "VALUES (?, ?, ?, ?)",
        (ident("op", 2), "uncommitted_probe", "0" * 64, NOW.isoformat()),
    )
    try:
        assert store.count("operations") == 1
    finally:
        writer.rollback()
        writer.close()


def test_supplied_operation_connection_never_completes_callers_transaction(store):
    replay_id = op(910)
    with store._operation(replay_id, "replay_probe", {"value": 1}):
        pass

    writer = store._connect()
    writer.execute("BEGIN IMMEDIATE")
    writer.execute(
        "INSERT INTO operations VALUES(?,?,?,?)",
        (ident("op", 911), "unrelated", "1" * 64, NOW.isoformat()),
    )
    try:
        with store._operation(op(912), "nested_success", {"value": 2}, connection=writer) as (
            _,
            replay,
        ):
            assert not replay
        assert writer.in_transaction
        assert store.count("operations") == 1
        assert (
            writer.execute(
                "SELECT count(*) FROM operations WHERE operation_id IN (?,?)",
                (ident("op", 911), ident("op", 912)),
            ).fetchone()[0]
            == 2
        )

        with store._operation(replay_id, "replay_probe", {"value": 1}, connection=writer) as (
            _,
            replay,
        ):
            assert replay
        assert writer.in_transaction
        assert writer.execute(
            "SELECT 1 FROM operations WHERE operation_id=?", (ident("op", 911),)
        ).fetchone()

        with pytest.raises(ConflictingOperation):
            with store._operation(replay_id, "replay_probe", {"value": 99}, connection=writer):
                pass
        assert writer.in_transaction
        assert writer.execute(
            "SELECT 1 FROM operations WHERE operation_id=?", (ident("op", 911),)
        ).fetchone()

        with pytest.raises(RuntimeError, match="nested failure"):
            with store._operation(op(913), "nested_failure", {}, connection=writer):
                raise RuntimeError("nested failure")
        assert writer.in_transaction
        assert writer.execute(
            "SELECT 1 FROM operations WHERE operation_id=?", (ident("op", 911),)
        ).fetchone()
        assert (
            writer.execute(
                "SELECT 1 FROM operations WHERE operation_id=?", (ident("op", 913),)
            ).fetchone()
            is None
        )
    finally:
        writer.rollback()
        writer.close()
