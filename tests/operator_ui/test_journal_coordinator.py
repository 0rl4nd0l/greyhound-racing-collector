"""Offline single-cycle contract for the R3-owned research journal."""

from src.operator_ui.journal import JournalCoordinator
from datetime import datetime, timedelta, timezone
from dataclasses import replace

import pytest


def collector_readiness(tmp_path, job_input, *, changed_native=False):
    """Real, outcome-free filesystem evidence for the existing collector lane."""
    import hashlib
    import json
    from src.operator_ui.journal_readiness import ResultAcquisitionReadiness
    from tests.test_autonomous_official_result_capture import (
        _write_shadow_run,
        _write_shadow_source_csv,
    )

    evidence = tmp_path / "collector-evidence"
    evidence.mkdir()
    source = evidence / "source.csv"
    _write_shadow_source_csv(source)
    _write_shadow_run(
        evidence,
        source_csv=source,
        race_id=job_input.race_id,
        race_time_minutes=610,
        dirname="daily_race_ingest_shadow_collector-run_daemon_autopilot",
    )
    collector = tmp_path / "collector"
    collector.mkdir()
    unit = tmp_path / "full.service"
    unit.write_text(
        f"[Service]\nWorkingDirectory={collector}\n"
        f"ExecStart=/python {collector}/scripts/shadow_autopilot_daemon.py run-once "
        f"--evidence-root {evidence} --enable-autonomous-result-capture\n"
    )
    authority = {
        "working_directory": str(collector),
        "units": {
            "full_service": {
                "path": str(unit),
                "sha256": hashlib.sha256(unit.read_bytes()).hexdigest(),
            }
        },
    }
    race = {
        "race_id": job_input.race_id,
        "jump_datetime": job_input.jump_timestamp,
        "race_url": "https://www.thedogs.com.au/racing/wentworth-park/2026-09-15/1/test-race",
        "runner_set_sha256": job_input.runner_set_sha256,
        "runners": [
            {
                "box_number": r["box"],
                "dog_name": r["name"],
                "source_native_runner_id": r["source_native_runner_id"],
            }
            for r in job_input.ordered_runners
        ],
    }
    if changed_native:
        race["runners"][0]["source_native_runner_id"] = "999"
    return ResultAcquisitionReadiness(evidence, authority=authority, races=lambda: (race,))


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


@pytest.mark.parametrize(
    "outcome",
    [
        "failed",
        "claimed",
        "stale_index",
        "expired_unclaimed",
        "result_uncovered",
        "changed_native_runner",
        "outcome_contaminated",
        "other_race_contaminated",
        "missing_csv",
        "wrong_csv",
        "fifo_csv",
        "oversized_csv",
        "unit_drift",
        "recovery_unready",
    ],
)
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
    runners = tuple(
        {"box": box, "name": name, "identity": name, "source_native_runner_id": str(100 + box)}
        for box, name in enumerate(("ALPHA", "BRAVO", "CHARLIE", "DELTA"), 1)
    )
    job_input = JobInput(
        "Race 1 - WPK - 2026-09-15",
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
        if outcome in {"expired_unclaimed", "recovery_unready"}:
            return
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
        return ({"race_id": job_input.race_id, "jump_datetime": job_input.jump_timestamp},)

    args = dict(
        activation=activation,
        root=tmp_path / "journal",
        services=services,
        audit=audit,
        races=races,
        result_readiness=(
            None
            if outcome == "result_uncovered"
            else collector_readiness(
                tmp_path, job_input, changed_native=outcome == "changed_native_runner"
            )
        ),
        clock=lambda: clock[0],
    )
    if outcome in {"outcome_contaminated", "other_race_contaminated"}:
        import json

        feature_path = (
            tmp_path
            / "collector-evidence/daily_race_ingest_shadow_collector-run_daemon_autopilot/shadow_feature_rows.json"
        )
        payload = json.loads(feature_path.read_bytes())
        if outcome == "outcome_contaminated":
            payload[0]["finish_position"] = 1
        else:
            payload.append({"race_id": "another-race", "finish_position": 1})
        feature_path.write_text(json.dumps(payload))
    source_csv = tmp_path / "collector-evidence/source.csv"
    if outcome in {"missing_csv", "fifo_csv"}:
        source_csv.rename(source_csv.with_suffix(".retained"))
        if outcome == "fifo_csv":
            import os

            os.mkfifo(source_csv)
    if outcome == "wrong_csv":
        source_csv.write_text("Dog Name,Box\n1. OTHER,1\n2. Bravo,2\n3. Charlie,3\n4. Delta,4\n")
    if outcome == "oversized_csv":
        source_csv.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    if outcome == "unit_drift":
        (tmp_path / "full.service").write_text("[Service]\nExecStart=/other\n")
    coordinator = JournalCoordinator(**args)
    coordinator.tick()
    clock[0] += timedelta(seconds=2)
    pending = coordinator.tick()
    if outcome == "stale_index":
        assert pending["state"] == "SOURCE_PENDING"
        assert pending["reason"] == "CURRENT_INDEX_STALE"
        assert pending["jobs"] == []
        return
    assert pending["admissions"] == {job_input.race_id: "PENDING_RECEIPT"}
    assert pending["jobs"] == []
    receipt[0] = True
    admitted = coordinator.tick()
    if outcome in {
        "result_uncovered",
        "changed_native_runner",
        "outcome_contaminated",
        "other_race_contaminated",
        "missing_csv",
        "wrong_csv",
        "fifo_csv",
        "oversized_csv",
        "unit_drift",
    }:
        assert admitted["admissions"] == {job_input.race_id: "RESULT_ACQUISITION_NOT_READY"}
        assert admitted["jobs"] == []
        assert store.recorded_jobs() == ()
        assert audit.verify_chain() and store.verify()
        return
    assert len(admitted["jobs"]) == 1
    job = store.get(admitted["jobs"][0])
    if outcome == "recovery_unready":
        source_csv.rename(source_csv.with_suffix(".retained"))
        original_events = store.events(job.job_id)
        recovered = JournalCoordinator(**args).tick()
        assert recovered["state"] == "STOPPED_UNCLAIMED_ADMISSION"
        assert recovered["recovery"][job.job_id] == "RESULT_ACQUISITION_NOT_READY"
        assert not store.get(job.job_id).attempt_claimed
        assert store.events(job.job_id) == original_events
        return
    if outcome == "expired_unclaimed":
        clock[0] += timedelta(hours=2)
        recovered = JournalCoordinator(**args).tick()
        assert recovered["state"] == "STOPPED_UNCLAIMED_ADMISSION"
        assert recovered["recovery"][job.job_id] == "OUTSIDE_FUTURE_ADMISSION_WINDOW"
        assert not store.get(job.job_id).attempt_claimed
        assert len(store.events(job.job_id)) == 3
        return
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
    [
        "complete",
        "native_id",
        "early_timestamp",
        "name",
        "dead_heat",
        "duplicate",
        "busy_source",
        "null_timestamp",
        "expired_queue",
    ],
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
        2 if result_case == "expired_queue" else 1,
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
    # Model a previously admitted job. Missing new-admission readiness must
    # never suppress closure of this existing, immutable prediction.
    from src.operator_ui.r3_api import submit_prediction, confirm_prediction_operation

    job, _ = submit_prediction(
        services,
        {
            "race_id": inp.race_id,
            "model_id": "latest-research",
            "config_id": "manual-default",
            "odds_source_id": "receipt",
            "idempotency_key": "22345678-1234-4123-8123-123456789abc",
        },
        identity="r3-journal:" + activation.activation_id,
        confirm_audit=lambda intent: confirm_prediction_operation(
            services,
            audit,
            intent,
            session_identifier=activation.activation_id,
            client_identity="server-owned-r3-journal",
        ),
    )
    job_id = job.job_id
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
    if result_case == "null_timestamp":
        race_row["start_datetime"] = None
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
    if result_case not in {"complete", "expired_queue"}:
        closure_state = coordinator.tick()["closures"][job_id]
        assert (
            closure_state == "RESULT_PENDING"
            if result_case == "busy_source"
            else closure_state.startswith("RESULT_REJECTED:")
        )
        assert not (tmp_path / "journal" / f"{job_id}.closure.json").exists()
        assert {p: p.read_bytes() for p in bundles.rglob("*") if p.is_file()} == original
        assert len(store.recorded_jobs()) == 1 and store.get(job_id).attempt_claimed
        return
    if result_case == "expired_queue":
        from src.operator_ui.r3_api import confirm_prediction_operation

        saved_clock = clock[0]
        clock[0] = NOW
        queued = store.create(
            actor_identity="r3-journal:" + activation.activation_id,
            actor_level=2,
            operation="manual_prediction",
            idempotency_key="32345678-1234-4123-8123-123456789abc",
            job_input=replace(inp, race_id="expired-queued-race"),
            now=NOW,
            confirm_audit=lambda intent: confirm_prediction_operation(
                services,
                audit,
                intent,
                session_identifier=activation.activation_id,
                client_identity="server-owned-r3-journal",
            ),
        )
        clock[0] = saved_clock
        recovered = JournalCoordinator(**coordinator_args).tick()
        assert recovered["state"] == "STOPPED_UNCLAIMED_ADMISSION"
        assert not store.get(queued.job_id).attempt_claimed
        assert recovered["closures"][job_id] == "CLOSED"
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
