"""Offline single-cycle contract for the R3-owned research journal."""

from src.operator_ui.journal import JournalCoordinator
from datetime import datetime, timedelta, timezone
from dataclasses import replace

import pytest


def collector_readiness(
    tmp_path,
    job_input,
    *,
    changed_native=False,
    source_race_id=None,
    index_race_id=None,
    race_url=None,
):
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
    source_race_id = source_race_id or job_input.race_id
    index_race_id = index_race_id or job_input.race_id
    race_url = race_url or (
        "https://www.thedogs.com.au/racing/wentworth-park/2026-09-15/1/test-race"
    )
    _write_shadow_run(
        evidence,
        source_csv=source,
        race_id=source_race_id,
        race_time_minutes=610,
        dirname="daily_race_ingest_shadow_collector-run_daemon_autopilot",
    )
    feature_path = evidence / "daily_race_ingest_shadow_collector-run_daemon_autopilot" / "shadow_feature_rows.json"
    feature_rows = json.loads(feature_path.read_bytes())
    for row in feature_rows:
        row["target_metadata_source_url"] = race_url
    feature_path.write_text(json.dumps(feature_rows))
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
        "race_id": index_race_id,
        "jump_datetime": job_input.jump_timestamp,
        "race_url": race_url,
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


@pytest.mark.parametrize(
    ("requested", "source", "url"),
    [
        (
            "Race 1 - SHEP - 2026-09-15",
            "Race 1 - SHEPPARTON - 2026-09-15",
            "https://www.thedogs.com.au/racing/shepparton/2026-09-15/1/test-race",
        ),
        (
            "Race 2 - QOT - 2026-09-15",
            "Race 2 - LADBROKES-Q-STRAIGHT - 2026-09-15",
            "https://www.thedogs.com.au/racing/ladbrokes-q-straight/2026-09-15/2/test-race",
        ),
    ],
)
def test_readiness_accepts_declared_source_race_aliases(tmp_path, requested, source, url):
    from types import SimpleNamespace

    from tests.operator_ui.test_r3_api import provenance

    runners = tuple(
        {
            "box": box,
            "name": name,
            "identity": name,
            "source_native_runner_id": str(100 + box),
        }
        for box, name in enumerate(("ALPHA", "BRAVO", "CHARLIE", "DELTA"), 1)
    )
    job_input = SimpleNamespace(
        race_id=requested,
        jump_timestamp="2026-09-15T10:10:00+10:00",
        runner_set_sha256="f" * 64,
        ordered_runners=runners,
        operational_index_provenance=provenance(),
    )
    readiness = collector_readiness(
        tmp_path,
        job_input,
        source_race_id=source,
        index_race_id=requested,
        race_url=url,
    )

    readiness.require(job_input, now=datetime(2026, 9, 14, 23, 0, tzinfo=timezone.utc))


def test_readiness_rejects_alias_with_cross_venue_source_url(tmp_path):
    from types import SimpleNamespace

    from src.operator_ui.r3_api import R3Rejected
    from tests.operator_ui.test_r3_api import provenance

    runners = tuple(
        {
            "box": box,
            "name": name,
            "identity": name,
            "source_native_runner_id": str(100 + box),
        }
        for box, name in enumerate(("ALPHA", "BRAVO", "CHARLIE", "DELTA"), 1)
    )
    job_input = SimpleNamespace(
        race_id="Race 1 - SHEP - 2026-09-15",
        jump_timestamp="2026-09-15T10:10:00+10:00",
        runner_set_sha256="f" * 64,
        ordered_runners=runners,
        operational_index_provenance=provenance(),
    )
    readiness = collector_readiness(
        tmp_path,
        job_input,
        source_race_id="Race 1 - SHEPPARTON - 2026-09-15",
        race_url="https://www.thedogs.com.au/racing/q-straight/2026-09-15/1/test-race",
    )

    with pytest.raises(R3Rejected, match="RESULT_ACQUISITION_NOT_READY"):
        readiness.require(job_input, now=datetime(2026, 9, 14, 23, 0, tzinfo=timezone.utc))


def test_recurrence_stops_after_terminal_coordinator_state(monkeypatch):
    from src.operator_ui import journal as journal_module

    class FakeEvent:
        def __init__(self):
            self.set_called = False
            self.wait_calls = 0

        def wait(self, _seconds):
            self.wait_calls += 1
            return self.wait_calls > 1

        def set(self):
            self.set_called = True

    class FakeThread:
        def __init__(self, *, target, name, daemon):
            assert name == "operator-ui-r3-journal"
            assert daemon is True
            self.target = target

        def start(self):
            self.target()

    class FakeCoordinator:
        activation = object()

        def __init__(self):
            self.calls = 0

        def tick(self):
            self.calls += 1
            return {"state": "PREDICTION_PENDING" if self.calls == 1 else "STOPPED_AFTER_CLOSURE"}

    event = FakeEvent()
    monkeypatch.setattr(journal_module.threading, "Event", lambda: event)
    monkeypatch.setattr(journal_module.threading, "Thread", FakeThread)
    coordinator = FakeCoordinator()

    returned = journal_module.start_journal_coordinator(coordinator, type("L", (), {"info": lambda *a: None})())

    assert returned is event
    assert coordinator.calls == 2
    assert event.set_called


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


def test_expired_empty_activation_stops_without_observing_sources(tmp_path):
    import logging
    from src.operator_ui.journal import JournalActivation, start_journal_coordinator
    from src.operator_ui.job_store import JobStore
    from types import SimpleNamespace

    now = datetime(2026, 9, 15, tzinfo=timezone.utc)
    activation = JournalActivation("12345678-1234-4123-8123-123456789abc",
        now + timedelta(minutes=1), now + timedelta(minutes=2), 1,
        "a" * 40, "b" * 64, "c" * 64, "d" * 64, ())
    clock = [now]
    def forbidden():
        pytest.fail("expired activation observed new-admission source")
    coordinator = JournalCoordinator(activation=activation, root=tmp_path / "journal",
        services=SimpleNamespace(job_store=JobStore(tmp_path / "jobs.db")),
        races=forbidden, clock=lambda: clock[0])
    coordinator.tick()
    clock[0] = now + timedelta(minutes=3)
    stop = start_journal_coordinator(coordinator, logging.getLogger(__name__))
    was_stopped = stop.is_set()
    stop.set()  # Always clean up a faulty background thread in the red proof.
    assert was_stopped
    # A clock adjustment on restart must not reopen a durably expired activation.
    clock[0] = now + timedelta(minutes=1)
    assert coordinator.tick()["state"] == "STOPPED_ADMISSION_EXPIRED"


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
        # New-admission readiness is not a recovery prerequisite. The existing
        # durable job remains pending when its dispatcher makes no progress.
        assert recovered["state"] == "PREDICTION_PENDING"
        assert recovered["recovery"] == {}
        assert store.get(job.job_id).phase is Phase.WAITING_FOR_CLAIM
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
        "failed_with_pending",
        "deadline_expired",
        "deadline_tampered",
        "deadline_during_tick",
        "interrupted_closure",
        "verification_lost",
    ],
)
def test_verified_fixture_prediction_closes_once_from_collector_rows(tmp_path, result_case, request, monkeypatch):
    import hashlib
    import json
    import shutil
    import sqlite3
    import tempfile
    from pathlib import Path
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

    config_path = Path(__file__).resolve().parents[2] / "configs/prediction/manual-default.json"

    if result_case == "complete":
        readonly_cwd = tmp_path / "readonly-cwd"
        readonly_cwd.mkdir(mode=0o500)
        request.addfinalizer(lambda: readonly_cwd.chmod(0o700))
        monkeypatch.chdir(readonly_cwd)

    # Isolate fixture-owned writes during each retained snapshot, independently
    # of a writable checkout/cwd. Foreign ancestor churn remains a hypothesis
    # for the historical failure, not something this relocation proves.
    import os
    fixture_parent = Path(os.environ.get(
        "PYTEST_R3_RECEIPT_ROOT", Path.home() / ".cache/greyhound-r3-receipt-tests"
    )).resolve()
    fixture_parent.mkdir(parents=True, exist_ok=True)
    evidence_root = Path(
        tempfile.mkdtemp(prefix="receipt-", dir=fixture_parent)
    )
    request.addfinalizer(lambda: shutil.rmtree(evidence_root))

    # Fixture producer output, not live inference: real sealing and verification,
    # deterministic feature/scoring dependencies from the native predictor tests.
    # Keep the immutable receipt evidence outside the prediction output root.
    # Directory identity is checked within each snapshot call, not between
    # separate calls. Later bundle creation alone does not invalidate a receipt.
    prototype = tmp_path / "prototype"
    prototype.mkdir()
    prototype_receipts = evidence_root / "prototype-receipts"
    prototype_receipts.mkdir()
    protocol, *_ = scheduled_exact_receipt(prototype_receipts)
    sample = run_prediction(
        args(prototype, config=str(config_path), odds_source="receipt", collector_request_root=protocol.root), dependencies()
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
        NOW + (timedelta(hours=1) if result_case == "complete" else timedelta(hours=2)),
        2 if result_case in {"expired_queue", "failed_with_pending"} else 1,
        "a" * 40,
        "b" * 64,
        inp.model_sha256,
        inp.config_sha256,
        (),
        result_observation_grace_seconds=600 if result_case.startswith("deadline_") else None,
    )
    authority = object()
    store = JobStore(tmp_path / "jobs.db", verifier_authority=authority)
    audit = AuditStore(tmp_path / "audit.db")
    producer = tmp_path / "producer"
    producer.mkdir()
    producer_receipts = evidence_root / "producer-receipts"
    producer_receipts.mkdir()
    producer_protocol, *_ = scheduled_exact_receipt(producer_receipts)

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
                config=str(config_path),
                odds_source="receipt",
                collector_request_root=producer_protocol.root,
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
    first_recovery = coordinator.tick()
    if job_id not in first_recovery["closures"]:
        first_recovery = coordinator.tick()
    if result_case == "verification_lost":
        manifest = next(path for path in original if path.name == "bundle_manifest.json")
        manifest.write_bytes(b"damaged fixture bundle")
        assert coordinator.tick()["state"] == "STOPPED_AFTER_FAILURE"
        manifest.write_bytes(original[manifest])
        # Restoring availability cannot reopen a terminally failed observation.
        restarted = JournalCoordinator(**coordinator_args).tick()
        assert restarted["state"] == "STOPPED_AFTER_FAILURE"
        assert restarted["closures"][job_id] == "PREDICTION_VERIFICATION_FAILED"
        assert store.get(job_id).attempt_claimed
        assert len(store.recorded_jobs()) == 1
        return
    if result_case.startswith("deadline_"):
        assert first_recovery["closures"][job_id] == "RESULT_PENDING"
        if result_case == "deadline_during_tick":
            class NoLateRead:
                def read(self, *args, **kwargs):
                    pytest.fail("observation deadline passed before source access")
            times = iter((NOW + timedelta(hours=1, minutes=5),
                          NOW + timedelta(hours=1, minutes=15)))
            coordinator.clock = lambda: next(times)
            coordinator.results = NoLateRead()
            report = coordinator.tick()
            assert report["closures"][job_id] == "RESULT_OBSERVATION_EXPIRED"
            return
        for changed_cap in (None, 172800):
            changed_args = {**coordinator_args, "activation": replace(
                activation, result_observation_grace_seconds=changed_cap)}
            with pytest.raises(ValueError, match="persisted activation differs"):
                JournalCoordinator(**changed_args).tick()
        clock[0] = NOW + timedelta(hours=1, minutes=5)
        assert JournalCoordinator(**coordinator_args).tick()["state"] == "RESULT_PENDING"
        observation_path = tmp_path / "journal" / f"{job_id}.observation.json"
        observation = json.loads(observation_path.read_bytes())
        assert datetime.fromisoformat(observation["deadline"].replace("Z", "+00:00")) == NOW + timedelta(hours=1, minutes=10)
        if result_case == "deadline_tampered":
            from src.operator_ui.job_store import canonical
            observation["deadline"] = (NOW + timedelta(hours=3)).isoformat()
            observation_path.write_bytes(canonical(observation))
            with pytest.raises(ValueError, match="deadline differs from activation"):
                JournalCoordinator(**coordinator_args).tick()
            return
        class NoExpiredResultAccess:
            def read(self, *args, **kwargs):
                pytest.fail("expired observation opened official result source")
        coordinator.results = NoExpiredResultAccess()
        clock[0] = NOW + timedelta(hours=1, minutes=15)
        expired = coordinator.tick()
        assert expired["state"] == "STOPPED_AFTER_CLOSURE"
        assert expired["closures"][job_id] == "RESULT_OBSERVATION_EXPIRED"
        terminal = (tmp_path / "journal" / f"{job_id}.terminal.json").read_bytes()
        restarted_args = dict(coordinator_args)
        restarted_args["results"] = NoExpiredResultAccess()
        clock[0] = NOW + timedelta(hours=1, minutes=6)
        restarted = JournalCoordinator(**restarted_args).tick()
        assert restarted["state"] == "STOPPED_AFTER_CLOSURE"
        assert restarted["closures"][job_id] == "RESULT_OBSERVATION_EXPIRED"
        assert (tmp_path / "journal" / f"{job_id}.terminal.json").read_bytes() == terminal
        return
    assert first_recovery["closures"].get(job_id) == "RESULT_PENDING", first_recovery
    if result_case in {"expired_queue", "failed_with_pending"}:
        confirm = lambda intent: confirm_prediction_operation(
            services, audit, intent, session_identifier=activation.activation_id,
            client_identity="server-owned-r3-journal")
        queued = store.create(
            actor_identity="r3-journal:" + activation.activation_id, actor_level=2,
            operation="manual_prediction",
            idempotency_key="32345678-1234-4123-8123-123456789abc",
            job_input=replace(inp, race_id="other-owned-race"), now=NOW,
            confirm_audit=confirm)
        if result_case == "failed_with_pending":
            for phase, status, reason in (
                (Phase.VALIDATED, "VALID", "validated"),
                (Phase.WAITING_FOR_CLAIM, "WAITING", "ready"),
            ):
                store.transition(queued.job_id, phase, now=NOW, status=status,
                    reason=reason, facts={}, confirm_audit=confirm)
            _, failed_attempt = store.claim_attempt(queued.job_id, now=NOW, confirm_audit=confirm)
            store.transition(queued.job_id, Phase.FAILED, now=NOW, status="FAILED",
                reason="PROCESS_LAUNCH_FAILED", facts={"attempt_id": failed_attempt, "error": "OSError"}, confirm_audit=confirm)
    clock[0] = NOW + timedelta(hours=1, minutes=5)
    expired_admission = coordinator.tick()
    assert expired_admission["closures"][job_id] == "RESULT_PENDING"
    if result_case in {"expired_queue", "failed_with_pending"}:
        assert expired_admission["state"] == "RESULT_PENDING"
        assert JournalCoordinator(**coordinator_args).tick()["state"] == "RESULT_PENDING"
    if result_case == "complete":
        assert expired_admission["state"] == "RESULT_PENDING"
        restarted_pending = JournalCoordinator(**coordinator_args).tick()
        assert restarted_pending["state"] == "RESULT_PENDING"
        assert restarted_pending["closures"][job_id] == "RESULT_PENDING"
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
    def publish_offline_result():
        with sqlite3.connect(results_db) as db:
            db.execute(
                "INSERT INTO autonomous_official_result_evidence_races VALUES (?,?)",
                (inp.race_id, json.dumps(race_row)),
            )
            db.executemany(
                "INSERT INTO autonomous_official_result_evidence_runners VALUES (?,?)",
                [(inp.race_id, json.dumps(row)) for row in result_rows],
            )

    if result_case == "complete":
        # Real recurrence and real coordinator, with only scheduling and result
        # arrival simulated. Initial restart sees an expired admission + pending
        # result; arrival on the next 60-second poll must close and stop.
        import logging
        from src.operator_ui import journal as journal_module
        class ScheduledEvent:
            stopped = False
            polls = 0
            def set(self):
                self.stopped = True
            def wait(self, seconds):
                assert seconds == 60
                self.polls += 1
                assert self.polls == 1
                publish_offline_result()
                return False
        class ScheduledThread:
            def __init__(self, *, target, **kwargs):
                self.target = target
            def start(self):
                self.target()
        event = ScheduledEvent()
        with monkeypatch.context() as scheduler:
            scheduler.setattr(journal_module.threading, "Event", lambda: event)
            scheduler.setattr(journal_module.threading, "Thread", ScheduledThread)
            journal_module.start_journal_coordinator(
                JournalCoordinator(**coordinator_args), logging.getLogger(__name__))
        assert event.stopped and event.polls == 1
    else:
        publish_offline_result()
    if result_case == "busy_source":
        results_db.with_name(results_db.name + "-wal").write_bytes(b"retained-live-sidecar")
    if result_case not in {"complete", "expired_queue", "failed_with_pending", "interrupted_closure"}:
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
    if result_case in {"expired_queue", "failed_with_pending"}:
        recovered = JournalCoordinator(**coordinator_args).tick()
        assert recovered["state"] == (
            "STOPPED_UNCLAIMED_ADMISSION" if result_case == "expired_queue"
            else "STOPPED_AFTER_FAILURE")
        assert store.get(queued.job_id).attempt_claimed is (result_case == "failed_with_pending")
        assert recovered["closures"][job_id] == "CLOSED"
        return
    if result_case == "interrupted_closure":
        original_link = os.link
        def interrupt_terminal(source, target, *args, **kwargs):
            if str(target).endswith(".terminal.json"):
                raise OSError("fixture crash before terminal publication")
            return original_link(source, target, *args, **kwargs)
        with monkeypatch.context() as faults:
            faults.setattr(os, "link", interrupt_terminal)
            with pytest.raises(OSError, match="fixture crash"):
                coordinator.tick()
    else:
        assert coordinator.tick()["closures"][job_id] == "CLOSED"
    closure = (tmp_path / "journal" / f"{job_id}.closure.json").read_bytes()
    # Once closure is durable, neither source availability nor restart grants
    # another official-result read. An attempted read would fail this test.
    class NoFurtherResultAccess:
        def read(self, *args, **kwargs):
            pytest.fail("terminal closure reopened official-result access")

    coordinator_args["results"] = NoFurtherResultAccess()
    restarted = JournalCoordinator(**coordinator_args).tick()
    if result_case == "interrupted_closure":
        assert restarted["state"] == "STOPPED_AFTER_FAILURE"
        assert restarted["closures"][job_id] == "CLOSURE_TERMINAL_PROOF_MISSING"
        assert (tmp_path / "journal" / f"{job_id}.closure.json").read_bytes() == closure
        return
    assert restarted["state"] == "STOPPED_AFTER_CLOSURE"
    assert restarted["closures"][job_id] == "CLOSED"
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
