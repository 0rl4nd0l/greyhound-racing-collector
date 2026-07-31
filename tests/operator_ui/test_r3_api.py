from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from flask import Flask
from werkzeug.security import generate_password_hash

from src.operator_ui.job_store import JobInput, JobStore
from src.operator_ui.r3_api import R3Rejected, R3Services, ResolvedSubmission, install_r3_api
from src.operator_ui.security import install_connected_mode

NOW = datetime(2026, 8, 1, tzinfo=timezone.utc)
H = hashlib.sha256(b"r3").hexdigest()
RACE = "race-20260801-richmond-r05"


def application(tmp_path, *, level=2, resolver=None, result=lambda _job: None, rate=20, launch=None):
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
    store = JobStore(tmp_path / "jobs.db", separate_from=(tmp_path / "audit.db", tmp_path / "canonical.db"))
    launched = []
    def resolve(selected, now):
        if resolver is not None:
            return resolver(selected, now)
        assert set(selected) == {"race_id", "model_id", "config_id", "odds_source_id", "idempotency_key"}
        if selected["race_id"] != RACE or selected["model_id"] != "latest-research" or selected["config_id"] != "manual-default" or selected["odds_source_id"] != "auto":
            raise R3Rejected("SELECTION_NOT_ALLOWLISTED")
        runners = ({"box": 1, "name": "ALPHA", "identity": "ALPHA"},)
        return ResolvedSubmission(JobInput(RACE, "2026-08-01T01:00:00Z", H, "latest-research", "model-v1", H, H, H, "manual-default", H, "auto", runners), runners)
    dispatcher = launch or (lambda job_id, _confirm: launched.append(job_id))
    install_r3_api(app, R3Services(store, resolve, dispatcher, result, clock=lambda: NOW, rate_limit=rate))
    return app, store, launched


def login(client):
    token = client.get("/operator-ui/login", base_url="https://localhost").get_json()["csrf_token"]
    response = client.post("/operator-ui/login", base_url="https://localhost", data={"username": "operator", "password": "safe-password", "csrf_token": token})
    return response.get_json()["csrf_token"]


def body(**changes):
    value = {"race_id": RACE, "model_id": "latest-research", "config_id": "manual-default", "odds_source_id": "auto", "idempotency_key": "12345678-1234-4123-8123-123456789abc"}
    value.update(changes)
    return value


def test_level2_csrf_exact_schema_idempotency_poll_and_actor_isolation(tmp_path):
    app, store, launched = application(tmp_path)
    client = app.test_client(); token = login(client)
    assert client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=body()).status_code == 400
    assert client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=body(extra="forbidden"), headers={"X-CSRF-Token": token}).get_json()["classification"] == "INVALID_REQUEST_SCHEMA"
    first = client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=body(), headers={"X-CSRF-Token": token})
    assert first.status_code == 202 and first.get_json()["phase"] == "WAITING_FOR_CLAIM" and len(launched) == 1
    duplicate = client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=body(), headers={"X-CSRF-Token": token})
    assert duplicate.status_code == 200 and duplicate.get_json()["job_id"] == first.get_json()["job_id"] and len(launched) == 1
    job_id = first.get_json()["job_id"]
    assert client.get(f"/operator-ui/api/v1/prediction-jobs/{job_id}", base_url="https://localhost").get_json()["timeline"][-1]["phase"] == "WAITING_FOR_CLAIM"
    with client.session_transaction() as session:
        session["operator_actor"] = "different-actor"
    assert client.get(f"/operator-ui/api/v1/prediction-jobs/{job_id}", base_url="https://localhost").status_code in {401, 404}
    assert store.verify()


def test_level1_cannot_submit_or_read_and_resolution_blockers_disclose_no_job(tmp_path):
    app, _, launched = application(tmp_path, level=1)
    client = app.test_client(); token = login(client)
    assert client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=body(), headers={"X-CSRF-Token": token}).status_code == 403
    assert client.get("/operator-ui/api/v1/prediction-jobs/job_" + "0" * 32, base_url="https://localhost").status_code == 403
    assert launched == []


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


def test_exact_ordered_runners_are_persisted_with_job_input(tmp_path):
    app, store, _ = application(tmp_path)
    client = app.test_client(); token = login(client)
    response = client.post("/operator-ui/api/v1/prediction-jobs", base_url="https://localhost", json=body(), headers={"X-CSRF-Token": token})
    job = store.get(response.get_json()["job_id"])
    assert job.input.ordered_runners == ({"box": 1, "identity": "ALPHA", "name": "ALPHA"},)
