"""Offline single-cycle contract for the R3-owned research journal."""

from src.operator_ui.journal import JournalCoordinator
from datetime import datetime, timedelta, timezone
from dataclasses import replace

import pytest


def test_disabled_tick_does_not_construct_stores_or_read_sources(tmp_path):
    coordinator = JournalCoordinator()
    assert coordinator.tick() == {"state": "DISABLED"}
    assert list(tmp_path.iterdir()) == []


def test_activation_is_future_only_and_cannot_be_replaced_on_restart(tmp_path):
    from src.operator_ui.journal import JournalActivation

    now = datetime(2026, 9, 15, tzinfo=timezone.utc)
    activation = JournalActivation(
        activation_id="12345678-1234-4123-8123-123456789abc",
        not_before=now + timedelta(minutes=1),
        admit_until=now + timedelta(hours=1),
        maximum_jobs=1,
        source_commit="a" * 40,
        protocol_sha256="b" * 64,
        model_sha256="c" * 64,
        config_sha256="d" * 64,
        excluded_race_ids=(),
    )
    assert JournalActivation.from_fields(activation.fields()) == activation
    coordinator = JournalCoordinator(activation=activation, root=tmp_path, clock=lambda: now)
    assert coordinator.tick()["state"] == "WAITING_START"
    original = (tmp_path / "activation.json").read_bytes()
    assert coordinator.tick()["state"] == "WAITING_START"
    assert (tmp_path / "activation.json").read_bytes() == original
    from dataclasses import replace

    replacement = JournalCoordinator(
        activation=replace(activation, maximum_jobs=2), root=tmp_path, clock=lambda: now
    )
    with pytest.raises(ValueError, match="activation differs"):
        replacement.tick()
    late = JournalCoordinator(
        activation=activation, root=tmp_path / "late", clock=lambda: now + timedelta(minutes=2)
    )
    with pytest.raises(ValueError, match="future cutoff"):
        late.tick()


@pytest.mark.parametrize("outcome", ["failed", "claimed", "stale_index"])
def test_single_cycle_admits_once_and_restart_does_not_retry_failed_race(tmp_path, outcome):
    from src.operator_ui.journal import JournalActivation
    from src.operator_ui.job_store import JobInput, JobStore, Phase
    from src.operator_ui.r3_api import R3Services, ResolvedSubmission, R3Rejected
    from src.operator_ui.security import AuditStore
    from tests.operator_ui.test_r3_api import provenance

    now = datetime(2026, 9, 15, tzinfo=timezone.utc)
    clock = [now]
    activation = JournalActivation(
        "12345678-1234-4123-8123-123456789abc",
        now + timedelta(seconds=1),
        now + timedelta(hours=1),
        2,
        "a" * 40,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        (),
    )
    store, audit = JobStore(tmp_path / "jobs.db"), AuditStore(tmp_path / "audit.db")
    runners = ({"box": 1, "name": "ALPHA", "identity": "ALPHA"},)
    job_input = JobInput(
        "future-race",
        (now + timedelta(minutes=10)).isoformat(),
        "e" * 64,
        "latest-research",
        "market_form_residual_v1",
        "c" * 64,
        "f" * 64,
        "f" * 64,
        "manual-default",
        "d" * 64,
        "receipt",
        runners,
        provenance(),
    )
    # Deterministic current-evidence fixture; no model/scorer or network call.
    receipt = [False]

    def resolve(selected, observed):
        if not receipt[0]:
            raise R3Rejected("PENDING_RECEIPT")
        return ResolvedSubmission(job_input, runners)

    def launch(job_id, confirm):
        _, attempt = store.claim_attempt(job_id, now=clock[0], confirm_audit=confirm)
        if outcome == "claimed":
            return
        store.transition(
            job_id,
            Phase.FAILED,
            now=clock[0],
            status="FAILED",
            reason="PROCESS_LAUNCH_FAILED",
            facts={"attempt_id": attempt, "error": "OSError"},
            confirm_audit=confirm,
        )

    services = R3Services(
        store, resolve, launch, lambda job, confirm: job, lambda job: None, clock=lambda: clock[0]
    )

    def races():
        if outcome == "stale_index":
            raise R3Rejected("CURRENT_INDEX_STALE")
        return ({"race_id": "future-race", "jump_datetime": job_input.jump_timestamp},)

    args = dict(
        activation=activation,
        root=tmp_path / "journal",
        services=services,
        audit=audit,
        races=races,
        clock=lambda: clock[0],
    )
    coordinator = JournalCoordinator(**args)
    coordinator.tick()
    clock[0] += timedelta(seconds=2)
    pending = coordinator.tick()
    if outcome == "stale_index":
        assert pending["state"] == "SOURCE_PENDING"
        assert pending["reason"] == "CURRENT_INDEX_STALE"
        assert pending["jobs"] == []
        return
    assert pending["admissions"] == {"future-race": "PENDING_RECEIPT"}
    assert pending["jobs"] == []
    receipt[0] = True
    admitted = coordinator.tick()
    assert len(admitted["jobs"]) == 1
    job = store.get(admitted["jobs"][0])
    if outcome == "claimed":
        assert job.attempt_claimed and job.phase is Phase.CLAIMED
        assert JournalCoordinator(**args).tick()["state"] == "PREDICTION_PENDING"
        assert len(store.events(job.job_id)) == 4
        return
    assert job.attempt_claimed and job.phase is Phase.FAILED
    assert JournalCoordinator(**args).tick()["state"] == "STOPPED_AFTER_FAILURE"
    assert len(store.events(job.job_id)) == 5
    assert audit.verify_chain() and store.verify()


@pytest.mark.parametrize(
    "result_case",
    ["complete", "native_id", "early_timestamp", "name", "dead_heat", "duplicate", "busy_source"],
)
def test_verified_fixture_prediction_closes_once_from_collector_rows(tmp_path, result_case):
    import hashlib
    import json
    import sqlite3
    from src.operator_ui.journal import JournalActivation
    from src.operator_ui.journal_results import OfficialResultSource
    from src.operator_ui.job_store import JobInput, JobStore, Phase
    from src.operator_ui.r3_api import (
        R3Services,
        ResolvedSubmission,
        build_verified_bundle_reader,
        finalize_producer_bundle,
    )
    from src.operator_ui.security import AuditStore
    from src.predictor.on_demand import canonical_bytes
    from scripts.predict_race_now import run_prediction
    from tests.test_predict_race_now import args, dependencies, scheduled_exact_receipt, NOW
    from tests.operator_ui.test_r3_api import provenance

    # Fixture producer output, not live inference: real sealing and verification,
    # deterministic feature/scoring dependencies from the native predictor tests.
    prototype = tmp_path / "prototype"
    prototype.mkdir()
    protocol, *_ = scheduled_exact_receipt(prototype)
    sample = run_prediction(
        args(prototype, odds_source="receipt", collector_request_root=protocol.root), dependencies()
    )
    assert sample["status"] == "PREDICTION_READY"
    value = sample["race"]
    rows = sample["prediction"]["predictions"]
    runners = tuple(
        {
            "box": row["box_number"],
            "name": row["dog_name"],
            "identity": row["identity"],
            **(
                {"source_native_runner_id": row["source_native_runner_id"]}
                if row.get("source_native_runner_id")
                else {}
            ),
        }
        for row in sorted(rows, key=lambda r: r["box_number"])
    )
    inp = JobInput(
        value["race_id"],
        value["jump_timestamp"],
        sample["evidence"]["runner_set_sha256"],
        "latest-research",
        sample["model"]["resolved"],
        sample["model"]["artifact_sha256"],
        sample["model"]["artifact_manifest_sha256"],
        sample["model"]["schema_sha256"],
        "manual-default",
        sample["config"]["sha256"],
        "receipt",
        runners,
        provenance(),
    )
    clock = [NOW - timedelta(seconds=1)]
    activation = JournalActivation(
        "12345678-1234-4123-8123-123456789abc",
        NOW,
        NOW + timedelta(hours=2),
        1,
        "a" * 40,
        "b" * 64,
        inp.model_sha256,
        inp.config_sha256,
        (),
    )
    authority = object()
    store = JobStore(tmp_path / "jobs.db", verifier_authority=authority)
    audit = AuditStore(tmp_path / "audit.db")
    producer = tmp_path / "producer"
    producer.mkdir()
    scheduled_exact_receipt(producer)

    def launch(job_id, confirm):
        _, attempt = store.claim_attempt(job_id, now=clock[0], confirm_audit=confirm)
        store.transition(
            job_id,
            Phase.ATTEMPT_STARTED,
            now=clock[0],
            status="RUNNING",
            reason="predictor_started",
            facts={"attempt_id": attempt, "pid": 123},
            confirm_audit=confirm,
        )
        result = run_prediction(
            args(
                producer,
                odds_source="receipt",
                job_id=job_id,
                operational_index_provenance=json.dumps(provenance().fields()),
            ),
            dependencies(),
        )
        assert result["status"] == "PREDICTION_READY"
        raw = canonical_bytes(result)
        empty = hashlib.sha256(b"").hexdigest()
        facts = {
            "attempt_id": attempt,
            "pid": 123,
            "exit_code": 0,
            "stdout_complete": True,
            "stdout_length": len(raw),
            "stdout_sha256": hashlib.sha256(raw).hexdigest(),
            "stdout_prefix_length": len(raw),
            "stdout_prefix_sha256": hashlib.sha256(raw).hexdigest(),
            "stderr_complete": True,
            "stderr_length": 0,
            "stderr_sha256": empty,
            "stderr_prefix_length": 0,
            "stderr_prefix_sha256": empty,
            "predictor_status": "PREDICTION_READY",
            "prediction_id": result["prediction_id"],
            "producer_job_id": job_id,
            "protocol_chain": result["evidence"]["protocol_chain"],
            "authenticated_cutoff": result["evidence"]["authenticated_cutoff"],
        }
        store.transition(
            job_id,
            Phase.RESPONSE_RECORDED,
            now=clock[0],
            status="RECORDED",
            reason="bounded_process_response",
            facts=facts,
            confirm_audit=confirm,
        )
        store.transition(
            job_id,
            Phase.PRODUCER_COMPLETED,
            now=clock[0],
            status="PRODUCER_COMPLETED",
            reason="PRODUCER_PREDICTION_READY",
            facts=facts,
            confirm_audit=confirm,
        )

    bundles = producer / "bundles"
    services = R3Services(
        store,
        lambda selected, now: ResolvedSubmission(inp, runners),
        launch,
        lambda job, confirm: finalize_producer_bundle(
            bundles, store, job, capability=authority, now=clock[0], confirm_audit=confirm
        ),
        build_verified_bundle_reader(bundles, store),
        clock=lambda: clock[0],
    )
    results_db = tmp_path / "official.db"
    with sqlite3.connect(results_db) as db:
        db.execute(
            "CREATE TABLE autonomous_official_result_evidence_races(race_id TEXT, row_json TEXT)"
        )
        db.execute(
            "CREATE TABLE autonomous_official_result_evidence_runners(race_id TEXT, row_json TEXT)"
        )
    coordinator_args = dict(
        activation=activation,
        root=tmp_path / "journal",
        services=services,
        audit=audit,
        races=lambda: ({"race_id": inp.race_id, "jump_datetime": inp.jump_timestamp},),
        clock=lambda: clock[0],
        results=OfficialResultSource(results_db),
    )
    coordinator = JournalCoordinator(**coordinator_args)
    coordinator.tick()
    clock[0] = NOW
    report = coordinator.tick()
    job_id = report["jobs"][0]
    original = {p: p.read_bytes() for p in bundles.rglob("*") if p.is_file()}
    assert coordinator.tick()["closures"][job_id] == "RESULT_PENDING"
    clock[0] = NOW + timedelta(hours=1, minutes=5)
    assert coordinator.tick()["closures"][job_id] == "RESULT_PENDING"
    common = {
        "race_id": inp.race_id,
        "race_date": value["race_date"],
        "venue": value["venue"],
        "race_number": value["race_number"],
        "source": "thedogs_official",
        "source_url": value["url"] + "?trial=false",
        "captured_at": clock[0].isoformat(),
    }
    race_row = {
        **common,
        "status": "resulted",
        "start_datetime": inp.jump_timestamp,
        "winner_name": "Alpha",
        "winner_box": 1,
        "box_order": [1, 2],
        "position_count": 2,
        "participant_count": 2,
    }
    result_rows = [
        {
            **common,
            "box_number": r["box"],
            "dog_name": r["name"],
            "finish_position": r["box"],
            "is_winner": r["box"] == 1,
        }
        for r in runners
    ]
    if result_case == "native_id":
        result_rows[0]["source_native_runner_id"] = "different-native-id"
    if result_case == "early_timestamp":
        race_row["captured_at"] = NOW.isoformat()
    if result_case == "name":
        result_rows[0]["dog_name"] = "OTHER DOG"
    if result_case == "dead_heat":
        result_rows[1]["finish_position"] = 1
    if result_case == "duplicate":
        result_rows.append(dict(result_rows[0]))
    with sqlite3.connect(results_db) as db:
        db.execute(
            "INSERT INTO autonomous_official_result_evidence_races VALUES (?,?)",
            (inp.race_id, json.dumps(race_row)),
        )
        db.executemany(
            "INSERT INTO autonomous_official_result_evidence_runners VALUES (?,?)",
            [(inp.race_id, json.dumps(row)) for row in result_rows],
        )
    if result_case == "busy_source":
        results_db.with_name(results_db.name + "-wal").write_bytes(b"retained-live-sidecar")
    if result_case != "complete":
        closure_state = coordinator.tick()["closures"][job_id]
        assert (
            closure_state == "RESULT_PENDING"
            if result_case == "busy_source"
            else closure_state.startswith("RESULT_REJECTED:")
        )
        assert not (tmp_path / "journal" / f"{job_id}.closure.json").exists()
        assert {p: p.read_bytes() for p in bundles.rglob("*") if p.is_file()} == original
        assert len(store.recorded_job_ids()) == 1 and store.get(job_id).attempt_claimed
        return
    assert coordinator.tick()["closures"][job_id] == "CLOSED"
    closure = (tmp_path / "journal" / f"{job_id}.closure.json").read_bytes()
    assert JournalCoordinator(**coordinator_args).tick()["closures"][job_id] == "CLOSED"
    assert (tmp_path / "journal" / f"{job_id}.closure.json").read_bytes() == closure
    assert {p: p.read_bytes() for p in bundles.rglob("*") if p.is_file()} == original
    assert audit.verify_chain() and store.verify()
    # A rehashed local closure is not independent official-source authority.
    altered = json.loads(closure)
    altered["official_result"]["race_rows"][0]["winner_name"] = "Beta"
    from src.operator_ui.job_store import canonical

    altered["official_result_sha256"] = hashlib.sha256(
        canonical(altered["official_result"])
    ).hexdigest()
    (tmp_path / "journal" / f"{job_id}.closure.json").write_bytes(canonical(altered))
    with pytest.raises(ValueError, match="closure official evidence differs"):
        JournalCoordinator(**coordinator_args).tick()
