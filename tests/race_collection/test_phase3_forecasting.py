import json
import sqlite3
import threading
from datetime import date, datetime, timedelta, timezone

import pytest

from race_collection.domain import (
    ArtifactChecksum,
    EvidenceAuthority,
    EvidenceField,
    FieldEvidence,
    FreezeAuthority,
    OperationId,
    ProgrammeRaceCandidate,
    RaceId,
    RaceState,
    RacingDay,
    RacingDayId,
)
from race_collection.forecasting import ForecastingAuthority, LegacyBundle, ModelRelease
from race_collection.operations import (
    BarrierNotSatisfied,
    ConflictingOperation,
    OperationsStoreError,
    SQLiteOperationsStore,
)

NOW = datetime(2026, 7, 22, 3, tzinfo=timezone.utc)


def identity(prefix, number):
    return f"{prefix}_{number:032x}"


def operation(number):
    return OperationId(identity("op", number))


def checksum(character):
    return ArtifactChecksum("sha256:" + character * 64)


class Predictor:
    def __init__(self, result=checksum("d"), error=None):
        self.result = result
        self.error = error
        self.requests = []

    def predict(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def setup(tmp_path):
    store = SQLiteOperationsStore(tmp_path / "operations.sqlite3")
    store.migrate()
    day = RacingDayId(identity("day", 1))
    day_value = RacingDay(day, date(2026, 7, 22), "Australia/Melbourne", NOW)
    store.create_racing_day(operation(100), day_value)
    races = []
    for number in (1, 2):
        race = store.record_expected_race(
            operation(100 + number),
            day_value,
            ProgrammeRaceCandidate("official", f"R-{number}", "Ballarat", number, NOW),
            checksum("a"),
            NOW,
        )
        races.append(race)
        store.advance_race(operation(110 + number), race, RaceState.CARD_COLLECTED, NOW)
        store.advance_race(operation(120 + number), race, RaceState.COLLECTING_ODDS, NOW)
        store.record_field_evidence(
            FieldEvidence(
                operation(130 + number),
                race,
                EvidenceField.VENUE,
                EvidenceAuthority.OFFICIAL_CARD,
                "Ballarat",
                "official",
                checksum("a"),
                NOW,
            )
        )
        store.seal_evidence(
            operation(140 + number),
            race_id=race,
            raw_checksum=checksum("a"),
            normalized_checksum=checksum("b"),
            schema_version="v1",
            normalization_version="v1",
            frozen_at=NOW,
            freeze_authority=FreezeAuthority.ACTUAL_JUMP,
            odds_checksum=checksum("c"),
            sealed_at=NOW,
            request_intent_digest=checksum("f"),
        )
        store.advance_race(operation(150 + number), race, RaceState.AWAITING_DAY_CLOSE, NOW)
    authority = ForecastingAuthority(store)
    bundle = LegacyBundle(
        "bundle-legacy-20260329",
        "V4_ExtraTrees_ExtraTreesClassifier_Calibrated_20260329_212033",
        checksum("6"),
        87072784,
        checksum("2"),
        None,
        "raw_registry_model",
        {"selection": "registry_is_best", "scaler_present": False},
    )
    release = ModelRelease("release-legacy-20260329", bundle.bundle_id, "policy-v1", {"phase": 3})
    authority.register_bundle(operation(1), bundle, NOW)
    authority.register_release(operation(2), release, NOW)
    authority.pin_day(operation(3), day, release, NOW)
    return store, authority, day, races, bundle, release


def close(store, day):
    store.close_racing_day(
        operation(900),
        RacingDay(day, date(2026, 7, 22), "Australia/Melbourne", NOW),
        NOW,
    )


def begin(authority, races, first=910):
    for offset, race in enumerate(races):
        authority.begin_prediction(operation(first + offset), race, NOW)


def test_migration_empty_repeat_and_populated_pre_phase3(tmp_path):
    store = SQLiteOperationsStore(tmp_path / "db.sqlite3")
    store._migration_scripts = lambda: SQLiteOperationsStore._migration_scripts(store)[:9]
    store.migrate()
    with store._connect() as db:
        db.execute(
            "INSERT INTO operations VALUES('legacy','fixture',?,?)",
            ("0" * 64, NOW.isoformat()),
        )
    del store._migration_scripts
    store.migrate()
    store.migrate()
    with store._connect() as db:
        assert db.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 28
        assert (
            db.execute("SELECT kind FROM operations WHERE operation_id='legacy'").fetchone()[0]
            == "fixture"
        )


def test_day_pin_is_immutable_replay_safe_and_conflicts_rejected(setup):
    store, authority, day, _, _, release = setup
    assert authority.pin_day(operation(3), day, release, NOW) is False
    with pytest.raises(ConflictingOperation):
        authority.pin_day(
            operation(3),
            day,
            ModelRelease("other", release.bundle_id, "policy-v1", {}),
            NOW,
        )
    close(store, day)
    with pytest.raises(BarrierNotSatisfied):
        authority.pin_day(operation(4), day, release, NOW)


def test_prediction_rejected_before_close_and_binds_exact_seal_and_pin(setup):
    store, authority, day, races, bundle, release = setup
    predictor = Predictor()
    with pytest.raises(BarrierNotSatisfied):
        authority.predict(operation(10), races[0], "prediction-1", predictor, NOW)
    close(store, day)
    with pytest.raises(BarrierNotSatisfied):
        authority.predict(operation(11), races[0], "prediction-1", predictor, NOW)
    authority.begin_prediction(operation(12), races[0], NOW)
    assert (
        authority.predict(operation(13), races[0], "prediction-1", predictor, NOW).status
        == "committed"
    )
    request = predictor.requests[0]
    assert request.evidence_checksum == checksum("b")
    assert request.bundle == bundle
    assert request.release.release_id == release.release_id
    assert request.release.descriptor == {"phase": 3}
    assert request.policy_id == "policy-v1"
    with store._connect() as db:
        events = db.execute(
            "SELECT target_state FROM lifecycle_events WHERE race_id=? ORDER BY event_id",
            (str(races[0]),),
        ).fetchall()
    assert [row[0] for row in events] == [
        "discovered",
        "card_collected",
        "collecting_odds",
        "evidence_sealed",
        "awaiting_day_close",
        "prediction_pending",
        "prediction_committed",
    ]


def test_prediction_replay_conflict_and_failure_isolation(setup):
    store, authority, day, races, *_ = setup
    close(store, day)
    begin(authority, races, 960)
    predictor = Predictor()
    first = authority.predict(operation(970), races[0], "prediction-1", predictor, NOW)
    replay = authority.predict(operation(970), races[0], "prediction-1", Predictor(), NOW)
    assert first.status == "committed" and replay.replayed
    with pytest.raises(ConflictingOperation):
        authority.predict(
            operation(970),
            races[0],
            "prediction-1",
            Predictor(),
            NOW + timedelta(seconds=1),
        )
    with pytest.raises(OperationsStoreError):
        authority.predict(operation(970), races[1], "prediction-2", Predictor(), NOW)
    failed = authority.predict(
        operation(971),
        races[1],
        "prediction-2",
        Predictor(error=RuntimeError("boom")),
        NOW,
    )
    assert failed.status == "quarantined"
    with store._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM deferred_predictions").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM prediction_quarantines").fetchone()[0] == 1


@pytest.mark.parametrize("terminal", ["committed", "quarantined"])
@pytest.mark.parametrize("tamper", ["operation", "intent", "snapshot_token"])
def test_prediction_replay_rejects_tampered_durable_intent(setup, terminal, tamper):
    store, authority, day, races, *_ = setup
    close(store, day)
    authority.begin_prediction(operation(972), races[0], NOW)
    predictor = Predictor() if terminal == "committed" else Predictor(error=RuntimeError("boom"))
    authority.predict(operation(973), races[0], "prediction-1", predictor, NOW)
    table = "deferred_predictions" if terminal == "committed" else "prediction_quarantines"
    with store._connect() as db:
        if tamper == "operation":
            db.execute(
                "UPDATE operations SET payload_sha256=? WHERE operation_id=?",
                ("0" * 64, str(operation(973))),
            )
        elif tamper == "intent":
            db.execute(f"DROP TRIGGER {table}_append_only_update")
            db.execute(
                f"UPDATE {table} SET request_intent_sha256=? WHERE operation_id=?",
                ("0" * 64, str(operation(973))),
            )
        else:
            db.execute(f"DROP TRIGGER {table}_append_only_update")
            db.execute(
                f"UPDATE {table} SET authority_snapshot_json="
                "json_set(authority_snapshot_json,'$.race_updated_at',?) "
                "WHERE operation_id=?",
                ((NOW + timedelta(seconds=1)).isoformat(), str(operation(973))),
            )
    with pytest.raises(OperationsStoreError, match="inconsistent durable intent"):
        authority.predict(operation(973), races[0], "prediction-1", Predictor(), NOW)


def test_concurrent_same_operation_with_different_computed_outcome_fails_closed(setup):
    store, authority, day, races, *_ = setup
    close(store, day)
    authority.begin_prediction(operation(974), races[0], NOW)

    class LosingPredictor:
        def predict(self, request):
            winner = authority.predict(
                operation(975), request.race_id, "prediction-1", Predictor(checksum("d")), NOW
            )
            assert winner.status == "committed"
            return checksum("e")

    with pytest.raises(ConflictingOperation, match="different (intent|terminal outcome)"):
        authority.predict(operation(975), races[0], "prediction-1", LosingPredictor(), NOW)
    with store._connect() as db:
        assert db.execute("SELECT artifact_checksum FROM deferred_predictions").fetchone()[
            0
        ] == str(checksum("d"))


def test_prediction_snapshot_drift_fails_closed_without_operation(setup):
    store, authority, day, races, *_ = setup
    close(store, day)
    authority.begin_prediction(operation(980), races[0], NOW)

    class DriftingPredictor:
        def predict(self, request):
            with store._connect() as db:
                db.execute(
                    "UPDATE races SET updated_at=? WHERE race_id=?",
                    (
                        (NOW + timedelta(seconds=1)).isoformat(),
                        str(request.race_id),
                    ),
                )
            return checksum("d")

    before = store.count("operations")
    with pytest.raises(BarrierNotSatisfied, match="snapshot changed"):
        authority.predict(operation(981), races[0], "prediction-drift", DriftingPredictor(), NOW)
    assert store.count("operations") == before
    with store._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM deferred_predictions").fetchone()[0] == 0


def test_begin_and_manual_quarantine_replay_authenticate_full_context(setup):
    store, authority, day, races, *_ = setup
    close(store, day)
    begin_at = NOW + timedelta(seconds=1)
    assert authority.begin_prediction(operation(982), races[0], begin_at)
    assert not authority.begin_prediction(operation(982), races[0], begin_at)
    with pytest.raises(ConflictingOperation):
        authority.begin_prediction(operation(982), races[0], begin_at + timedelta(seconds=1))
    with store._connect() as db:
        db.execute("DROP TRIGGER prediction_begins_append_only_update")
        db.execute(
            "UPDATE prediction_begins SET authority_snapshot_json="
            "json_set(authority_snapshot_json,'$.race_updated_at',?) WHERE operation_id=?",
            ((begin_at + timedelta(seconds=1)).isoformat(), str(operation(982))),
        )
    with pytest.raises(OperationsStoreError, match="inconsistent durable intent"):
        authority.begin_prediction(operation(982), races[0], begin_at)
    assert authority.quarantine_prediction(
        operation(983), races[0], "manual", "operator decision", begin_at
    )
    assert not authority.quarantine_prediction(
        operation(983), races[0], "manual", "operator decision", begin_at
    )
    with pytest.raises(ConflictingOperation):
        authority.quarantine_prediction(operation(983), races[0], "manual", "changed", begin_at)
    with store._connect() as db:
        db.execute("DROP TRIGGER prediction_quarantines_append_only_update")
        db.execute(
            "UPDATE prediction_quarantines SET evidence_checksum=? WHERE operation_id=?",
            (str(checksum("9")), str(operation(983))),
        )
    with pytest.raises(OperationsStoreError, match="authority snapshot"):
        authority.quarantine_prediction(
            operation(983), races[0], "manual", "operator decision", begin_at
        )


def test_concurrent_identical_begin_replays_durable_snapshot(setup, monkeypatch):
    store, authority, day, races, *_ = setup
    close(store, day)
    begin_at = NOW + timedelta(seconds=1)
    barrier = threading.Barrier(2)
    original = authority._prediction_context

    def synchronized_context(db, race_id, expected_state):
        row = original(db, race_id, expected_state)
        if expected_state is None:
            barrier.wait(timeout=5)
        return row

    monkeypatch.setattr(authority, "_prediction_context", synchronized_context)
    outcomes = []

    def call_begin():
        try:
            outcomes.append(
                ("return", authority.begin_prediction(operation(984), races[0], begin_at))
            )
        except Exception as error:  # pragma: no cover - asserted through outcomes
            outcomes.append(("error", type(error).__name__, str(error)))

    threads = [threading.Thread(target=call_begin) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert sorted(outcomes) == [("return", False), ("return", True)]


def test_storage_rejects_forged_prediction_begin_snapshot(setup):
    store, _, day, races, *_ = setup
    close(store, day)
    with store._connect() as db:
        row = db.execute(
            "SELECT r.racing_day_id,r.updated_at,d.closed_at,p.bundle_id,p.release_id,"
            "p.policy_id,s.seal_id,s.normalized_checksum,m.descriptor_json "
            "FROM races r JOIN racing_days d USING(racing_day_id) "
            "JOIN racing_day_pins p USING(racing_day_id) "
            "JOIN model_releases m USING(release_id) JOIN sealed_evidence s USING(race_id) "
            "WHERE r.race_id=?",
            (str(races[0]),),
        ).fetchone()
        op = operation(985)
        db.execute(
            "INSERT INTO operations VALUES(?,?,?,?)",
            (str(op), "begin_deferred_prediction", "0" * 64, NOW.isoformat()),
        )
        snapshot = {
            "race": str(races[0]),
            "prediction": f"begin-{races[0]}",
            "day": row["racing_day_id"],
            "seal": row["seal_id"],
            "evidence": row["normalized_checksum"],
            "bundle": row["bundle_id"],
            "release": row["release_id"],
            "policy": row["policy_id"],
            "descriptor": json.loads(row["descriptor_json"]),
            "closed_at": row["closed_at"],
            "race_updated_at": (NOW + timedelta(seconds=1)).isoformat(),
        }
        with pytest.raises(sqlite3.IntegrityError, match="begin snapshot disagrees"):
            db.execute(
                "INSERT INTO prediction_begins VALUES(?,?,?,?,?)",
                (str(races[0]), json.dumps(snapshot), NOW.isoformat(), "0" * 64, str(op)),
            )


def test_prediction_begin_replay_rejects_tampered_operation_intent(setup):
    store, authority, day, races, *_ = setup
    close(store, day)
    begin_at = NOW + timedelta(seconds=1)
    operation_id = operation(986)
    authority.begin_prediction(operation_id, races[0], begin_at)
    with store._connect() as db:
        db.execute(
            "UPDATE operations SET payload_sha256=? WHERE operation_id=?",
            ("0" * 64, str(operation_id)),
        )
    with pytest.raises(OperationsStoreError, match="inconsistent durable intent"):
        authority.begin_prediction(operation_id, races[0], begin_at)


def test_result_retry_policy_and_public_value_types_are_strict(setup):
    store, authority, day, races, *_ = setup
    close(store, day)
    begin(authority, races, 984)
    authority.predict(operation(986), races[0], "prediction-1", Predictor(), NOW)
    authority.quarantine_prediction(operation(987), races[1], "manual", "done", NOW)
    authority.open_results(operation(988), races[0], NOW)
    with store._connect() as db:
        db.execute(
            "INSERT INTO operations VALUES(?,?,?,?)",
            (str(operation(999)), "forged_result", "0" * 64, NOW.isoformat()),
        )
        with pytest.raises(sqlite3.IntegrityError, match="result shape"):
            db.execute(
                "INSERT INTO result_attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "forged-attempt",
                    str(races[0]),
                    1,
                    3,
                    (NOW + timedelta(hours=1)).isoformat(),
                    "result-retry-v1",
                    1,
                    "collected",
                    str(checksum("9")),
                    '"scalar"',
                    None,
                    NOW.isoformat(),
                    str(operation(999)),
                ),
            )
    authority.record_result_attempt(
        operation(989),
        races[0],
        "attempt-1",
        at=NOW,
        max_attempts=3,
        deadline=NOW + timedelta(hours=1),
        error="not ready",
    )
    with pytest.raises(ConflictingOperation, match="retry policy"):
        authority.record_result_attempt(
            operation(990),
            races[0],
            "attempt-2",
            at=NOW + timedelta(minutes=1),
            max_attempts=4,
            deadline=NOW + timedelta(hours=2),
            error="not ready",
        )
    with pytest.raises(ValueError, match="ArtifactChecksum"):
        authority.record_on_demand(
            operation(991),
            "forecast-bad",
            races[0],
            "sha256:" + "a" * 64,
            checksum("b"),
            NOW,
        )


def test_all_phase3_authority_tables_are_append_only(setup):
    store, authority, day, races, *_ = setup
    close(store, day)
    begin(authority, races, 1000)
    authority.predict(operation(1002), races[0], "prediction-1", Predictor(), NOW)
    authority.quarantine_prediction(operation(1003), races[1], "manual", "done", NOW)
    authority.open_results(operation(1004), races[0], NOW)
    authority.record_result_attempt(
        operation(1005),
        races[0],
        "attempt-1",
        at=NOW,
        max_attempts=1,
        deadline=NOW,
        artifact_checksum=checksum("9"),
        outcome={"order": [1, 2]},
    )
    authority.join_training_example(
        operation(1006),
        races[0],
        "example-1",
        checksum("e"),
        eligible=True,
        reason=None,
        at=NOW,
    )
    authority.record_on_demand(
        operation(1007), "forecast-1", races[1], checksum("d"), checksum("b"), NOW
    )
    tables = (
        "model_bundles",
        "model_releases",
        "racing_day_pins",
        "prediction_begins",
        "deferred_predictions",
        "prediction_quarantines",
        "result_attempts",
        "training_examples",
        "on_demand_forecasts",
    )
    with store._connect() as db:
        for table in tables:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                db.execute(f"UPDATE {table} SET operation_id=operation_id")
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                db.execute(f"DELETE FROM {table}")


@pytest.mark.parametrize(
    ("attempt_number", "max_attempts", "at_offset", "status"),
    [
        (3, 3, 60, "failed"),
        (2, 4, 60, "failed"),
        (2, 3, 0.5, "failed"),
        (2, 3, 3600, "failed"),
        (2, 3, 60, "quarantined"),
    ],
)
def test_storage_rejects_forged_retry_policy(
    setup, attempt_number, max_attempts, at_offset, status
):
    store, authority, day, races, *_ = setup
    close(store, day)
    begin(authority, races, 1050)
    authority.predict(operation(1052), races[0], "prediction-1", Predictor(), NOW)
    authority.quarantine_prediction(operation(1053), races[1], "manual", "done", NOW)
    authority.open_results(operation(1054), races[0], NOW)
    deadline = NOW + timedelta(hours=1)
    authority.record_result_attempt(
        operation(1055),
        races[0],
        "attempt-1",
        at=NOW,
        max_attempts=3,
        deadline=deadline,
        error="not ready",
    )
    with store._connect() as db:
        db.execute(
            "INSERT INTO operations VALUES(?,?,?,?)",
            (
                str(operation(1056)),
                "retry_forgery",
                "0" * 64,
                NOW.isoformat(),
            ),
        )
        with pytest.raises(sqlite3.IntegrityError, match="retry policy"):
            db.execute(
                "INSERT INTO result_attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "forged-retry",
                    str(races[0]),
                    attempt_number,
                    max_attempts,
                    deadline.isoformat(),
                    "result-retry-v1",
                    1,
                    status,
                    None,
                    None,
                    "not ready",
                    (NOW + timedelta(seconds=at_offset)).isoformat(),
                    str(operation(1056)),
                ),
            )


def test_storage_rejects_checksum_and_composite_relation_forgeries(setup):
    store, authority, day, races, *_ = setup
    close(store, day)
    begin(authority, races, 1100)
    with store._connect() as db:
        row = db.execute(
            "SELECT r.racing_day_id,p.bundle_id,p.release_id,p.policy_id,s.seal_id,"
            "s.normalized_checksum,r.updated_at,d.closed_at,m.descriptor_json "
            "FROM races r JOIN racing_days d USING(racing_day_id) "
            "JOIN racing_day_pins p USING(racing_day_id) "
            "JOIN model_releases m USING(release_id) JOIN sealed_evidence s USING(race_id) "
            "WHERE r.race_id=?",
            (str(races[0]),),
        ).fetchone()
        for number in range(1110, 1118):
            db.execute(
                "INSERT INTO operations VALUES(?,?,?,?)",
                (
                    str(operation(number)),
                    "direct_forgery",
                    "0" * 64,
                    NOW.isoformat(),
                ),
            )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO model_bundles VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "forged-bundle",
                    "legacy-origin",
                    "model",
                    "sha256:" + "A" * 64,
                    1,
                    str(checksum("a")),
                    None,
                    "raw_registry_model",
                    "{}",
                    NOW.isoformat(),
                    str(operation(1113)),
                ),
            )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO model_releases VALUES(?,?,?,?,?,?)",
                (
                    "forged-release",
                    row["bundle_id"],
                    "policy",
                    "[]",
                    NOW.isoformat(),
                    str(operation(1114)),
                ),
            )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO model_releases VALUES(?,?,?,?,?,?)",
                (
                    "forged-release-link",
                    "missing-bundle",
                    "policy",
                    "{}",
                    NOW.isoformat(),
                    str(operation(1117)),
                ),
            )
        db.execute(
            "INSERT INTO model_bundles VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                "bundle-second",
                "legacy-origin",
                "model-second",
                str(checksum("8")),
                1,
                str(checksum("7")),
                None,
                "raw_registry_model",
                "{}",
                NOW.isoformat(),
                str(operation(1113)),
            ),
        )
        db.execute(
            "INSERT INTO model_releases VALUES(?,?,?,?,?,?)",
            (
                "release-second",
                "bundle-second",
                "policy-second",
                "{}",
                NOW.isoformat(),
                str(operation(1114)),
            ),
        )
        second_day = identity("day", 2)
        db.execute(
            "INSERT INTO racing_days VALUES(?,?,?,?,NULL)",
            (second_day, "2026-07-23", "Australia/Melbourne", NOW.isoformat()),
        )
        with pytest.raises(sqlite3.IntegrityError, match="day pin release disagrees"):
            db.execute(
                "INSERT INTO racing_day_pins VALUES(?,?,?,?,?,?)",
                (
                    second_day,
                    row["bundle_id"],
                    "release-second",
                    "policy-second",
                    NOW.isoformat(),
                    str(operation(1115)),
                ),
            )
        snapshot = json.dumps(
            {
                "race": str(races[0]),
                "prediction": "forged",
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
        )
        prediction = (
            "forged",
            str(races[0]),
            row["racing_day_id"],
            row["bundle_id"],
            row["release_id"],
            "wrong-policy",
            row["seal_id"],
            row["normalized_checksum"],
            str(checksum("d")),
            NOW.isoformat(),
            "0" * 64,
            snapshot,
            str(operation(1110)),
        )
        with pytest.raises(sqlite3.IntegrityError, match="(relations|snapshot) disagrees"):
            db.execute(
                "INSERT INTO deferred_predictions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", prediction
            )
        bad_checksum = list(prediction)
        bad_checksum[5] = row["policy_id"]
        bad_checksum[7] = "sha256:" + "a" * 63 + "!"
        bad_checksum[-1] = str(operation(1111))
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO deferred_predictions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", bad_checksum
            )
        forged_snapshot = list(prediction)
        forged_snapshot[5] = row["policy_id"]
        snapshot_value = json.loads(snapshot)
        snapshot_value["race_updated_at"] = (NOW + timedelta(seconds=1)).isoformat()
        forged_snapshot[11] = json.dumps(snapshot_value)
        forged_snapshot[-1] = str(operation(1116))
        with pytest.raises(sqlite3.IntegrityError, match="snapshot disagrees"):
            db.execute(
                "INSERT INTO deferred_predictions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                forged_snapshot,
            )
        quarantine = (
            str(races[0]),
            "forged-q",
            row["racing_day_id"],
            row["bundle_id"],
            row["release_id"],
            "wrong-policy",
            row["seal_id"],
            row["normalized_checksum"],
            "code",
            "details",
            NOW.isoformat(),
            "0" * 64,
            snapshot,
            str(operation(1112)),
        )
        with pytest.raises(sqlite3.IntegrityError, match="(relations|snapshot) disagrees"):
            db.execute(
                "INSERT INTO prediction_quarantines VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", quarantine
            )


def test_prediction_failure_rolls_back_and_can_be_durably_quarantined(setup):
    store, authority, day, races, *_ = setup
    close(store, day)
    authority.begin_prediction(operation(19), races[0], NOW)
    before = store.count("operations")
    outcome = authority.predict(
        operation(20),
        races[0],
        "prediction-1",
        Predictor(error=RuntimeError("load failed")),
        NOW,
    )
    assert outcome.status == "quarantined"
    assert store.count("operations") == before + 1
    with store._connect() as db:
        assert (
            db.execute("SELECT state FROM races WHERE race_id=?", (str(races[0]),)).fetchone()[0]
            == "prediction_quarantined"
        )


def test_storage_rejects_result_training_and_terminal_relation_forgeries(setup):
    store, authority, day, races, *_ = setup
    close(store, day)
    begin(authority, races, 1170)
    authority.predict(operation(1172), races[0], "prediction-1", Predictor(), NOW)
    authority.predict(operation(1173), races[1], "prediction-2", Predictor(), NOW)
    authority.open_results(operation(1174), races[0], NOW)
    authority.open_results(operation(1175), races[1], NOW)
    for offset, race in enumerate(races):
        authority.record_result_attempt(
            operation(1176 + offset),
            race,
            f"attempt-{offset + 1}",
            at=NOW,
            max_attempts=1,
            deadline=NOW,
            artifact_checksum=checksum("9"),
            outcome={"order": [1, 2]},
        )
    with store._connect() as db:
        for number in (1178, 1179, 1180):
            db.execute(
                "INSERT INTO operations VALUES(?,?,?,?)",
                (
                    str(operation(number)),
                    "relation_forgery",
                    "0" * 64,
                    NOW.isoformat(),
                ),
            )
        with pytest.raises(sqlite3.IntegrityError, match="result (race|retry policy)"):
            db.execute(
                "INSERT INTO result_attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "late-attempt",
                    str(races[0]),
                    2,
                    1,
                    NOW.isoformat(),
                    "result-retry-v1",
                    1,
                    "failed",
                    None,
                    None,
                    "late",
                    NOW.isoformat(),
                    str(operation(1178)),
                ),
            )
        with pytest.raises(sqlite3.IntegrityError, match="training example relations"):
            db.execute(
                "INSERT INTO training_examples VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    "forged-example",
                    str(races[0]),
                    "prediction-1",
                    "attempt-2",
                    str(checksum("e")),
                    "eligible",
                    None,
                    NOW.isoformat(),
                    str(operation(1179)),
                ),
            )
        committed = db.execute(
            "SELECT * FROM deferred_predictions WHERE race_id=?", (str(races[0]),)
        ).fetchone()
        with pytest.raises(sqlite3.IntegrityError, match="(relations|snapshot) disagrees"):
            db.execute(
                "INSERT INTO prediction_quarantines VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(races[0]),
                    "forged-terminal",
                    committed["racing_day_id"],
                    committed["bundle_id"],
                    committed["release_id"],
                    committed["policy_id"],
                    committed["seal_id"],
                    committed["evidence_checksum"],
                    "code",
                    "details",
                    NOW.isoformat(),
                    "0" * 64,
                    committed["authority_snapshot_json"],
                    str(operation(1180)),
                ),
            )


def test_partial_batch_blocks_results_mixed_terminal_batch_opens_only_committed(setup):
    store, authority, day, races, *_ = setup
    close(store, day)
    begin(authority, races, 920)
    authority.predict(operation(30), races[0], "prediction-1", Predictor(), NOW)
    with pytest.raises(BarrierNotSatisfied):
        authority.open_results(operation(31), races[0], NOW)
    authority.quarantine_prediction(
        operation(32), races[1], "feature_failed", "missing feature", NOW
    )
    assert authority.open_results(operation(33), races[0], NOW)
    with pytest.raises(BarrierNotSatisfied):
        authority.open_results(operation(34), races[1], NOW)


def test_bounded_result_retries_quarantine_and_prediction_quarantine_has_no_join(setup):
    store, authority, day, races, *_ = setup
    close(store, day)
    begin(authority, races, 930)
    authority.predict(operation(40), races[0], "prediction-1", Predictor(), NOW)
    authority.quarantine_prediction(operation(41), races[1], "failed", "failed", NOW)
    authority.open_results(operation(42), races[0], NOW)
    assert (
        authority.record_result_attempt(
            operation(43),
            races[0],
            "attempt-1",
            at=NOW,
            max_attempts=2,
            deadline=NOW + timedelta(hours=1),
            error="not ready",
        )
        == "failed"
    )
    assert (
        authority.record_result_attempt(
            operation(44),
            races[0],
            "attempt-2",
            at=NOW + timedelta(minutes=1),
            max_attempts=2,
            deadline=NOW + timedelta(hours=1),
            error="not ready",
        )
        == "quarantined"
    )
    with pytest.raises(BarrierNotSatisfied):
        authority.join_training_example(
            operation(45),
            races[1],
            "example-2",
            checksum("e"),
            eligible=True,
            reason=None,
            at=NOW,
        )


def test_collected_result_join_provenance_and_ambiguous_outcome_is_forward_only(setup):
    store, authority, day, races, *_ = setup
    close(store, day)
    begin(authority, races, 940)
    authority.predict(operation(50), races[0], "prediction-1", Predictor(), NOW)
    authority.quarantine_prediction(operation(51), races[1], "failed", "failed", NOW)
    authority.open_results(operation(52), races[0], NOW)
    assert (
        authority.record_result_attempt(
            operation(53),
            races[0],
            "attempt-1",
            at=NOW,
            max_attempts=2,
            deadline=NOW,
            artifact_checksum=checksum("9"),
            outcome={"order": [1, 2]},
        )
        == "collected"
    )
    authority.join_training_example(
        operation(54),
        races[0],
        "example-1",
        checksum("e"),
        eligible=False,
        reason="post-seal scratch",
        at=NOW,
    )
    with store._connect() as db:
        row = db.execute(
            "SELECT prediction_id,result_attempt_id,eligibility FROM training_examples"
        ).fetchone()
        assert tuple(row) == ("prediction-1", "attempt-1", "evaluation_ineligible")
        assert (
            db.execute("SELECT state FROM races WHERE race_id=?", (str(races[0]),)).fetchone()[0]
            == "evaluation_ineligible"
        )


def test_dead_heat_cannot_be_collected_or_joined(setup):
    store, authority, day, races, *_ = setup
    close(store, day)
    begin(authority, races, 1150)
    authority.predict(operation(1152), races[0], "prediction-1", Predictor(), NOW)
    authority.quarantine_prediction(operation(1153), races[1], "manual", "done", NOW)
    authority.open_results(operation(1154), races[0], NOW)
    with pytest.raises(ValueError, match="unambiguous"):
        authority.record_result_attempt(
            operation(1155),
            races[0],
            "dead-heat",
            at=NOW,
            max_attempts=1,
            deadline=NOW,
            artifact_checksum=checksum("9"),
            outcome={"order": [1, 1]},
        )
    with pytest.raises(BarrierNotSatisfied):
        authority.join_training_example(
            operation(1156),
            races[0],
            "example-dead-heat",
            checksum("e"),
            eligible=True,
            reason=None,
            at=NOW,
        )


def test_on_demand_is_separate_and_does_not_open_barrier(setup):
    store, authority, day, races, *_ = setup
    authority.record_on_demand(
        operation(60), "forecast-1", races[0], checksum("d"), checksum("b"), NOW
    )
    close(store, day)
    with pytest.raises(BarrierNotSatisfied):
        authority.open_results(operation(61), races[0], NOW)
    with store._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM deferred_predictions").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM on_demand_forecasts").fetchone()[0] == 1


def test_corrupt_foreign_key_and_transaction_rollback(setup):
    store, authority, day, races, *_ = setup
    close(store, day)
    authority.begin_prediction(operation(950), races[0], NOW)
    with store._connect() as db:
        db.execute("DROP TRIGGER sealed_evidence_append_only_update")
        db.execute("DROP TRIGGER sealed_evidence_checksums_update")
        db.execute(
            "UPDATE sealed_evidence SET normalized_checksum='corrupt' WHERE race_id=?",
            (str(races[0]),),
        )
    before = store.count("operations")
    with pytest.raises(ValueError):
        authority.predict(operation(70), races[0], "prediction-1", Predictor(), NOW)
    assert store.count("operations") == before
    with store._connect() as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO deferred_predictions VALUES('forged',?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(RaceId(identity("race", 99))),
                    str(day),
                    "bundle-legacy-20260329",
                    "release-legacy-20260329",
                    "policy-v1",
                    1,
                    str(checksum("a")),
                    str(checksum("b")),
                    NOW.isoformat(),
                    "x",
                    "{}",
                    identity("op", 99),
                ),
            )
