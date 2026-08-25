from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from flask import Flask
from werkzeug.security import generate_password_hash

from src.operator_ui.job_store import (
    JobInput,
    JobStore,
    JobStoreError,
    OperationalIndexProvenance,
    Phase,
    resolve_audit_confirmation,
)
from src.operator_ui.r3_api import (
    R3Rejected,
    R3Services,
    ResolvedSubmission,
    finalize_producer_bundle,
    install_r3_api,
)
from src.operator_ui.security import install_connected_mode
from src.predictor.on_demand import VerifiedPredictionBundle, VerifiedPredictionBundleIndex

NOW = datetime(2026, 8, 1, tzinfo=timezone.utc)
H = hashlib.sha256(b"r3").hexdigest()
RACE = "race-20260801-richmond-r05"


def provenance():
    return OperationalIndexProvenance(
        "operator_ui_operational_index_admission_v1", "collector_current_race_index_v2",
        "collector-run", H, H, H, H, H,
    )


def application(tmp_path, *, level=2, resolver=None, result=lambda _job: None, rate=20, launch=None, finalize=lambda job, _confirm: job, verifier_authority=None):
    app = Flask(__name__)
    app.config.update(
        TESTING=True, OPERATOR_UI_CONNECTED_MODE=True,
        OPERATOR_UI_SECRET_KEY="r3-test-secret-" + "x" * 40,
        OPERATOR_UI_USERNAME="operator", OPERATOR_UI_PASSWORD_HASH=generate_password_hash("safe-password"),
        OPERATOR_UI_LEVEL=level, OPERATOR_UI_AUDIT_DB_PATH=str(tmp_path / "audit.db"),
        OPERATOR_UI_JOB_DB_PATH=str(tmp_path / "jobs.db"), DATABASE_PATH=str(tmp_path / "canonical.db"),
        OPERATOR_UI_DEPLOYED_COMMIT="c" * 40, OPERATOR_UI_DEPLOYED_TREE="d" * 40,
        OPERATOR_UI_DEPLOYED_VERSION="r3-test", OPERATOR_UI_CLOCK=lambda: NOW,
    )
    install_connected_mode(app)
    store = JobStore(tmp_path / "jobs.db", separate_from=(tmp_path / "audit.db", tmp_path / "canonical.db"), verifier_authority=verifier_authority)
    launched = []
    def resolve(selected, now):
        if resolver is not None:
            return resolver(selected, now)
        assert set(selected) == {"race_id", "model_id", "config_id", "odds_source_id", "idempotency_key"}
        if selected["race_id"] != RACE or selected["model_id"] != "latest-research" or selected["config_id"] != "manual-default" or selected["odds_source_id"] != "auto":
            raise R3Rejected("SELECTION_NOT_ALLOWLISTED")
        runners = ({"box": 1, "name": "ALPHA", "identity": "ALPHA"},)
        return ResolvedSubmission(JobInput(RACE, "2026-08-01T01:00:00Z", H, "latest-research", "model-v1", H, H, H, "manual-default", H, "auto", runners, provenance()), runners)
    dispatcher = launch or (lambda job_id, _confirm: launched.append(job_id))
    install_r3_api(app, R3Services(store, resolve, dispatcher, finalize, result, clock=lambda: NOW, rate_limit=rate))
    return app, store, launched


def login(client):
    token = client.get("/operator-ui/login", base_url="https://localhost").get_json()["csrf_token"]
    response = client.post("/operator-ui/login", base_url="https://localhost", data={"username": "operator", "password": "safe-password", "csrf_token": token})
    return response.get_json()["csrf_token"]


def body(**changes):
    value = {"race_id": RACE, "model_id": "latest-research", "config_id": "manual-default", "odds_source_id": "auto", "idempotency_key": "12345678-1234-4123-8123-123456789abc"}
    value.update(changes)
    return value


def producer_complete(store, job_id):
    confirm=lambda intent:resolve_audit_confirmation(intent,H)
    job,attempt=store.claim_attempt(job_id,now=NOW,confirm_audit=confirm)
    job=store.transition(job_id,Phase.ATTEMPT_STARTED,now=NOW,status="RUNNING",reason="predictor_started",facts={"attempt_id":attempt,"pid":123},confirm_audit=confirm)
    empty=hashlib.sha256(b"").hexdigest(); facts={"attempt_id":attempt,"pid":123,"exit_code":0,"stdout_complete":True,"stdout_length":0,"stdout_sha256":empty,"stdout_prefix_length":0,"stdout_prefix_sha256":empty,"stderr_complete":True,"stderr_length":0,"stderr_sha256":empty,"stderr_prefix_length":0,"stderr_prefix_sha256":empty,"predictor_status":"PREDICTION_READY","prediction_id":"12345678-1234-4123-8123-123456789abc","producer_job_id":job_id}
    job=store.transition(job_id,Phase.RESPONSE_RECORDED,now=NOW,status="RECORDED",reason="bounded_process_response",facts=facts,confirm_audit=confirm)
    return store.transition(job_id,Phase.PRODUCER_COMPLETED,now=NOW,status="PRODUCER_COMPLETED",reason="PRODUCER_PREDICTION_READY",facts=facts,confirm_audit=confirm)


def test_level2_csrf_exact_schema_idempotency_poll_and_actor_isolation(tmp_path):
    app, store, launched = application(tmp_path)
    client = app.test_client(); token = login(client)
    assert client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=body()).status_code == 400
    assert client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=body(extra="forbidden"), headers={"X-CSRF-Token": token}).get_json()["classification"] == "INVALID_REQUEST_SCHEMA"
    assert client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=body(operational_index_provenance=provenance().fields()), headers={"X-CSRF-Token": token}).get_json()["classification"] == "INVALID_REQUEST_SCHEMA"
    first = client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=body(), headers={"X-CSRF-Token": token})
    assert first.status_code == 202 and first.get_json()["phase"] == "WAITING_FOR_CLAIM" and len(launched) == 1
    duplicate = client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=body(), headers={"X-CSRF-Token": token})
    assert duplicate.status_code == 200 and duplicate.get_json()["job_id"] == first.get_json()["job_id"] and len(launched) == 2
    job_id = first.get_json()["job_id"]
    assert client.get(f"/operator-ui/api/v1/prediction-jobs/{job_id}", base_url="https://localhost").get_json()["timeline"][-1]["phase"] == "WAITING_FOR_CLAIM"
    with client.session_transaction() as session:
        session["operator_actor"] = "different-actor"
    assert client.get(f"/operator-ui/api/v1/prediction-jobs/{job_id}", base_url="https://localhost").status_code in {401, 404}
    assert store.verify()


def test_different_key_cannot_retry_race_with_durable_job(tmp_path):
    app, store, launched = application(tmp_path)
    client = app.test_client(); token = login(client)
    first = client.post("/operator-ui/api/v1/prediction-jobs",base_url="https://localhost",json=body(),headers={"X-CSRF-Token":token})
    retry = client.post("/operator-ui/api/v1/prediction-jobs",base_url="https://localhost",json=body(idempotency_key="aaaaaaaa-1234-4123-8123-123456789abc"),headers={"X-CSRF-Token":token})
    assert first.status_code == 202
    assert retry.status_code == 409
    assert retry.get_json()["classification"] == "RACE_ALREADY_RECORDED"
    assert len(launched) == 1
    assert store.verify()


def test_idempotent_retransmission_recovers_persisted_job_after_index_turnover(tmp_path):
    resolutions = []

    def resolve(_selected, _now):
        run_id = f"collector-run-{len(resolutions) + 1}"
        resolutions.append(run_id)
        runners = ({"box": 1, "name": "ALPHA", "identity": "ALPHA"},)
        admitted = OperationalIndexProvenance(
            "operator_ui_operational_index_admission_v1",
            "collector_current_race_index_v2",
            run_id,
            hashlib.sha256(run_id.encode()).hexdigest(),
            H,
            H,
            H,
            H,
        )
        return ResolvedSubmission(
            JobInput(
                RACE, "2026-08-01T01:00:00Z", H, "latest-research", "model-v1",
                H, H, H, "manual-default", H, "auto", runners, admitted,
            ),
            runners,
        )

    app, store, launched = application(tmp_path, resolver=resolve)
    client = app.test_client(); token = login(client)
    first = client.post(
        "/operator-ui/api/v1/prediction-jobs", base_url="https://localhost",
        json=body(), headers={"X-CSRF-Token": token},
    )
    assert first.status_code == 202
    admitted = store.get(first.get_json()["job_id"]).input.operational_index_provenance

    duplicate = client.post(
        "/operator-ui/api/v1/prediction-jobs", base_url="https://localhost",
        json=body(), headers={"X-CSRF-Token": token},
    )

    assert duplicate.status_code == 200
    assert duplicate.get_json()["job_id"] == first.get_json()["job_id"]
    assert store.get(first.get_json()["job_id"]).input.operational_index_provenance == admitted
    assert resolutions == ["collector-run-1"]
    assert launched == [first.get_json()["job_id"], first.get_json()["job_id"]]


def test_level1_cannot_submit_or_read_and_resolution_blockers_disclose_no_job(tmp_path):
    app, _, launched = application(tmp_path, level=1)
    client = app.test_client(); token = login(client)
    assert client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=body(), headers={"X-CSRF-Token": token}).status_code == 403
    assert client.get("/operator-ui/api/v1/prediction-jobs/job_" + "0" * 32, base_url="https://localhost").status_code == 403
    assert launched == []


def test_server_rejects_resolver_output_missing_durable_operational_index_provenance(tmp_path):
    runners=({"box":1,"name":"ALPHA","identity":"ALPHA"},)
    legacy=JobInput(RACE,"2026-08-01T01:00:00Z",H,"latest-research","model-v1",H,H,H,"manual-default",H,"auto",runners)
    app,store,launched=application(tmp_path,resolver=lambda _selected,_now:ResolvedSubmission(legacy,runners))
    client=app.test_client(); token=login(client)
    response=client.post("/operator-ui/api/v1/prediction-jobs",base_url="https://localhost",json=body(),headers={"X-CSRF-Token":token})
    assert response.status_code==409 and response.get_json()["classification"]=="RACE_EVIDENCE_INVALID"
    assert launched==[] and store.verify()


def test_capability_is_authenticated_exact_level2_and_server_owned(tmp_path):
    app, _, _ = application(tmp_path)
    client = app.test_client()
    assert client.get("/operator-ui/api/v1/r3-capability", base_url="https://localhost").status_code == 401
    login(client)
    response = client.get("/operator-ui/api/v1/r3-capability", base_url="https://localhost")
    assert response.status_code == 200
    assert response.get_json() == {
        "schema": "operator_ui_r3_capability_v1", "authorized": True,
        "runtime_configured": True, "level": 2,
    }


def test_server_reresolution_failure_and_idempotency_conflict_are_stable(tmp_path):
    def stale(_selected, _now): raise R3Rejected("CURRENT_INDEX_STALE")
    app, _, launched = application(tmp_path / "stale", resolver=stale)
    client = app.test_client(); token = login(client)
    response = client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=body(), headers={"X-CSRF-Token": token})
    assert response.status_code == 409 and response.get_json()["classification"] == "CURRENT_INDEX_STALE" and launched == []
    app, _, launched = application(tmp_path / "conflict")
    client = app.test_client(); token = login(client)
    assert client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=body(), headers={"X-CSRF-Token": token}).status_code == 202
    response = client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=body(config_id="another"), headers={"X-CSRF-Token": token})
    assert response.get_json()["classification"] == "SELECTION_NOT_ALLOWLISTED" and len(launched) == 1


def test_dispatch_failure_is_terminal_and_same_key_recovers_same_job(tmp_path):
    def fail(_job_id, _confirm):
        raise OSError("dispatcher unavailable")
    app, store, _ = application(tmp_path, launch=fail)
    client = app.test_client(); token = login(client)
    first = client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=body(), headers={"X-CSRF-Token": token})
    assert first.status_code == 202
    assert first.get_json()["phase"] == "FAILED"
    assert first.get_json()["timeline"][-1]["reason"] == "DISPATCH_FAILED"
    recovered = client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=body(), headers={"X-CSRF-Token": token})
    assert recovered.status_code == 200
    assert recovered.get_json()["job_id"] == first.get_json()["job_id"]
    assert store.verify()


def test_restart_observation_redispatches_an_unclaimed_waiting_job(tmp_path):
    app, store, launched = application(tmp_path)
    client = app.test_client(); token = login(client)
    first = client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=body(), headers={"X-CSRF-Token": token})
    job_id = first.get_json()["job_id"]
    app, restarted_store, launched = application(tmp_path)
    client = app.test_client(); token = login(client)
    again = client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=body(), headers={"X-CSRF-Token": token})
    assert again.status_code == 200
    assert again.get_json()["job_id"] == job_id
    assert launched == [job_id]
    assert restarted_store.get(job_id).phase.value == "WAITING_FOR_CLAIM"
    assert restarted_store.verify()


def test_restart_get_alone_dispatches_and_claims_waiting_job_once(tmp_path):
    app, store, _ = application(tmp_path)
    client = app.test_client(); token = login(client)
    created = client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=body(), headers={"X-CSRF-Token": token})
    job_id = created.get_json()["job_id"]
    holder = {}; calls = []
    def claim(job, confirm):
        calls.append(job)
        holder["store"].claim_attempt(job, now=NOW, confirm_audit=confirm)
    app, restarted, _ = application(tmp_path, launch=claim)
    holder["store"] = restarted
    client = app.test_client(); login(client)
    first = client.get(f"/operator-ui/api/v1/prediction-jobs/{job_id}", base_url="https://localhost")
    second = client.get(f"/operator-ui/api/v1/prediction-jobs/{job_id}", base_url="https://localhost")
    assert first.status_code == second.status_code == 200
    assert calls == [job_id]
    assert restarted.get(job_id).phase.value == "CLAIMED"
    assert restarted.verify()


def test_restart_get_resumes_verification_of_durable_producer_completion(tmp_path):
    app, store, _ = application(tmp_path)
    client = app.test_client(); token = login(client)
    created = client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=body(), headers={"X-CSRF-Token": token})
    job_id = created.get_json()["job_id"]
    producer_complete(store,job_id)
    finalized=[]
    app, restarted, _ = application(tmp_path, finalize=lambda job, _confirm: (finalized.append(job.job_id) or job))
    client=app.test_client(); login(client)
    response=client.get(f"/operator-ui/api/v1/prediction-jobs/{job_id}",base_url="https://localhost")
    assert response.status_code==200 and finalized==[job_id]
    assert restarted.get(job_id).phase is Phase.PRODUCER_COMPLETED


def test_missing_sealed_bundle_index_is_a_durable_exact_verifier_failure(tmp_path):
    authority=object(); store=JobStore(tmp_path/"jobs.db",verifier_authority=authority)
    runners=({"box":1,"name":"ALPHA","identity":"ALPHA"},)
    inp=JobInput(RACE,"2026-08-01T01:00:00Z",H,"latest-research","model-v1",H,H,H,"manual-default",H,"receipt",runners,provenance())
    confirm=lambda intent:resolve_audit_confirmation(intent,H)
    job=store.create(actor_identity="operator",actor_level=2,operation="manual_prediction",idempotency_key="12345678-1234-4123-8123-123456789abc",job_input=inp,now=NOW,confirm_audit=confirm)
    job=store.transition(job.job_id,Phase.VALIDATED,now=NOW,status="VALID",reason="validated",confirm_audit=confirm)
    job=store.transition(job.job_id,Phase.WAITING_FOR_CLAIM,now=NOW,status="WAITING",reason="ready",confirm_audit=confirm)
    job=producer_complete(store,job.job_id)
    final=finalize_producer_bundle(tmp_path/"missing-bundles",store,job,capability=authority,now=NOW,confirm_audit=confirm)
    assert final.phase is Phase.FAILED
    assert store.events(job.job_id)[-1]["facts"]["blocker"]=={"code":"BUNDLE_INDEX_VERIFICATION_FAILED","stage":"BUNDLE_VERIFICATION"}
    assert store.verify()


def test_terminal_api_discloses_exact_sealed_bundle_verifier_failure(tmp_path):
    authority=object(); holder={}
    def finalize(job,confirm):
        return finalize_producer_bundle(tmp_path/"missing-bundles",holder["store"],job,capability=authority,now=NOW,confirm_audit=confirm)
    app,store,_=application(tmp_path,finalize=finalize,verifier_authority=authority);holder["store"]=store
    client=app.test_client();token=login(client)
    created=client.post("/operator-ui/api/v1/prediction-jobs",base_url="https://localhost",json=body(),headers={"X-CSRF-Token":token})
    job_id=created.get_json()["job_id"];producer_complete(store,job_id)

    payload=client.get(f"/operator-ui/api/v1/prediction-jobs/{job_id}",base_url="https://localhost").get_json()

    assert payload["phase"]=="FAILED"
    assert payload["blocker"]=="BUNDLE_INDEX_VERIFICATION_FAILED"


def test_mismatched_sealed_blocker_is_a_durable_exact_verifier_failure(tmp_path,monkeypatch):
    from src.operator_ui import r3_api as r3_api_module
    authority=object(); store=JobStore(tmp_path/"jobs.db",verifier_authority=authority)
    runners=({"box":1,"name":"ALPHA","identity":"ALPHA"},); confirm=lambda intent:resolve_audit_confirmation(intent,H)
    inp=JobInput(RACE,"2026-08-01T01:00:00Z",H,"latest-research","model-v1",H,H,H,"manual-default",H,"receipt",runners,provenance())
    job=store.create(actor_identity="operator",actor_level=2,operation="manual_prediction",idempotency_key="12345678-1234-4123-8123-123456789abc",job_input=inp,now=NOW,confirm_audit=confirm)
    job=store.transition(job.job_id,Phase.VALIDATED,now=NOW,status="VALID",reason="validated",confirm_audit=confirm)
    job=store.transition(job.job_id,Phase.WAITING_FOR_CLAIM,now=NOW,status="WAITING",reason="ready",confirm_audit=confirm)
    job,attempt=store.claim_attempt(job.job_id,now=NOW,confirm_audit=confirm)
    job=store.transition(job.job_id,Phase.ATTEMPT_STARTED,now=NOW,status="RUNNING",reason="predictor_started",facts={"attempt_id":attempt,"pid":123},confirm_audit=confirm)
    empty=hashlib.sha256(b"").hexdigest(); facts={"attempt_id":attempt,"pid":123,"exit_code":2,"stdout_complete":True,"stdout_length":0,"stdout_sha256":empty,"stdout_prefix_length":0,"stdout_prefix_sha256":empty,"stderr_complete":True,"stderr_length":0,"stderr_sha256":empty,"stderr_prefix_length":0,"stderr_prefix_sha256":empty,"predictor_status":"PREDICTION_BLOCKED","prediction_id":"12345678-1234-4123-8123-123456789abc","producer_job_id":job.job_id,"producer_blocker":{"code":"POST_JUMP","stage":"VALIDATION"}}
    job=store.transition(job.job_id,Phase.RESPONSE_RECORDED,now=NOW,status="RECORDED",reason="bounded_process_response",facts=facts,confirm_audit=confirm)
    job=store.transition(job.job_id,Phase.PRODUCER_COMPLETED,now=NOW,status="PRODUCER_COMPLETED",reason="PRODUCER_PREDICTION_BLOCKED:POST_JUMP",facts=facts,confirm_audit=confirm)
    entry={"directory":"prediction_safe","prediction_id":facts["prediction_id"],"job_id":job.job_id,"status":"PREDICTION_BLOCKED","manifest_sha256":H,"logical_bundle_sha256":H}
    request={"schema_version":"on_demand_prediction_request_v2","job_id":job.job_id,"race_id":inp.race_id,"jump_timestamp":inp.jump_timestamp,"runner_set_sha256":inp.runner_set_sha256,"odds_source":inp.odds_source,"config_sha256":inp.config_sha256,"model":{"requested":inp.model_selector,"resolved":inp.resolved_model_identity,"model_sha256":inp.model_sha256,"manifest_sha256":inp.model_manifest_sha256,"schema_sha256":inp.model_schema_sha256},"runners":[{"box_number":1,"display_name":"ALPHA","identity":"ALPHA","source_native_runner_id":None}],"operational_index_provenance":inp.operational_index_provenance.fields()}
    bundle=VerifiedPredictionBundle("prediction_safe",entry,{"blocker":{"code":"NO_MATCH"},"blocker_stage":"VALIDATION"},{},request)
    monkeypatch.setattr(r3_api_module,"verify_prediction_bundle_index",lambda *_args,**_kwargs:VerifiedPredictionBundleIndex("on_demand_prediction_bundle_index_v1",NOW.isoformat(),(entry,),b"{}",H))
    monkeypatch.setattr(r3_api_module,"verify_indexed_prediction_bundle",lambda *_args,**_kwargs:bundle)
    final=finalize_producer_bundle(tmp_path,store,job,capability=authority,now=NOW,confirm_audit=confirm)
    assert final.phase is Phase.FAILED
    assert store.events(job.job_id)[-1]["facts"]["blocker"]=={"code":"BUNDLE_BLOCKER_IDENTITY_MISMATCH","stage":"BUNDLE_VERIFICATION"}
    assert store.verify()


def test_blocked_bundle_provenance_divergence_is_a_durable_verifier_failure(tmp_path,monkeypatch):
    from src.operator_ui import r3_api as r3_api_module
    authority=object();store=JobStore(tmp_path/"jobs.db",verifier_authority=authority)
    runners=({"box":1,"name":"ALPHA","identity":"ALPHA"},);confirm=lambda intent:resolve_audit_confirmation(intent,H)
    inp=JobInput(RACE,"2026-08-01T01:00:00Z",H,"latest-research","model-v1",H,H,H,"manual-default",H,"receipt",runners,provenance())
    job=store.create(actor_identity="operator",actor_level=2,operation="manual_prediction",idempotency_key="12345678-1234-4123-8123-123456789abc",job_input=inp,now=NOW,confirm_audit=confirm)
    job=store.transition(job.job_id,Phase.VALIDATED,now=NOW,status="VALID",reason="validated",confirm_audit=confirm)
    job=store.transition(job.job_id,Phase.WAITING_FOR_CLAIM,now=NOW,status="WAITING",reason="ready",confirm_audit=confirm)
    job,attempt=store.claim_attempt(job.job_id,now=NOW,confirm_audit=confirm)
    job=store.transition(job.job_id,Phase.ATTEMPT_STARTED,now=NOW,status="RUNNING",reason="predictor_started",facts={"attempt_id":attempt,"pid":123},confirm_audit=confirm)
    empty=hashlib.sha256(b"").hexdigest();facts={"attempt_id":attempt,"pid":123,"exit_code":2,"stdout_complete":True,"stdout_length":0,"stdout_sha256":empty,"stdout_prefix_length":0,"stdout_prefix_sha256":empty,"stderr_complete":True,"stderr_length":0,"stderr_sha256":empty,"stderr_prefix_length":0,"stderr_prefix_sha256":empty,"predictor_status":"PREDICTION_BLOCKED","prediction_id":"12345678-1234-4123-8123-123456789abc","producer_job_id":job.job_id,"producer_blocker":{"code":"POST_JUMP","stage":"VALIDATION"}}
    job=store.transition(job.job_id,Phase.RESPONSE_RECORDED,now=NOW,status="RECORDED",reason="bounded_process_response",facts=facts,confirm_audit=confirm)
    job=store.transition(job.job_id,Phase.PRODUCER_COMPLETED,now=NOW,status="PRODUCER_COMPLETED",reason="PRODUCER_PREDICTION_BLOCKED:POST_JUMP",facts=facts,confirm_audit=confirm)
    entry={"directory":"prediction_safe","prediction_id":facts["prediction_id"],"job_id":job.job_id,"status":"PREDICTION_BLOCKED","manifest_sha256":H,"logical_bundle_sha256":H}
    admitted=provenance().fields();divergent={**admitted,"run_id":"different-run"}
    request={"schema_version":"on_demand_prediction_request_v2","job_id":job.job_id,"race_id":inp.race_id,"jump_timestamp":inp.jump_timestamp,"runner_set_sha256":inp.runner_set_sha256,"odds_source":inp.odds_source,"config_sha256":inp.config_sha256,"model":{"requested":inp.model_selector,"resolved":inp.resolved_model_identity,"model_sha256":inp.model_sha256,"manifest_sha256":inp.model_manifest_sha256,"schema_sha256":inp.model_schema_sha256},"runners":[{"box_number":1,"display_name":"ALPHA","identity":"ALPHA","source_native_runner_id":None}],"operational_index_provenance":divergent}
    bundle=VerifiedPredictionBundle("prediction_safe",entry,{"blocker":{"code":"POST_JUMP"},"blocker_stage":"VALIDATION"},{},request)
    monkeypatch.setattr(r3_api_module,"verify_prediction_bundle_index",lambda *_args,**_kwargs:VerifiedPredictionBundleIndex("on_demand_prediction_bundle_index_v1",NOW.isoformat(),(entry,),b"{}",H))
    monkeypatch.setattr(r3_api_module,"verify_indexed_prediction_bundle",lambda *_args,**_kwargs:bundle)

    final=finalize_producer_bundle(tmp_path,store,job,capability=authority,now=NOW,confirm_audit=confirm)

    assert final.phase is Phase.FAILED
    assert store.events(job.job_id)[-1]["facts"]["blocker"]=={"code":"BUNDLE_JOB_IDENTITY_MISMATCH","stage":"BUNDLE_VERIFICATION"}
    assert store.verify()


def test_same_process_get_renotifies_when_dispatch_has_no_durable_outcome(tmp_path):
    launches=[]
    def launch(job_id, _confirm):
        launches.append(job_id)
        # The asynchronous worker failed before claim and its FAILED closure
        # also failed: the only durable truth remains WAITING_FOR_CLAIM.
    app,store,_=application(tmp_path,launch=launch)
    client=app.test_client(); token=login(client)
    response=client.post("/operator-ui/api/v1/prediction-jobs",base_url="https://localhost",json=body(),headers={"X-CSRF-Token":token})
    assert response.status_code==202
    job_id=response.get_json()["job_id"]
    assert store.get(job_id).phase is Phase.WAITING_FOR_CLAIM
    assert client.get(f"/operator-ui/api/v1/prediction-jobs/{job_id}",base_url="https://localhost").status_code==200
    assert launches==[job_id,job_id]


def test_exact_ordered_runners_are_persisted_with_job_input(tmp_path):
    app, store, _ = application(tmp_path)
    client = app.test_client(); token = login(client)
    response = client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=body(), headers={"X-CSRF-Token": token})
    job = store.get(response.get_json()["job_id"])
    assert job.input.ordered_runners == ({"box": 1, "identity": "ALPHA", "name": "ALPHA"},)
