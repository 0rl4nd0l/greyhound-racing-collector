from __future__ import annotations

import hashlib
import json
import sqlite3
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from flask import Flask
from werkzeug.security import generate_password_hash

from src.operator_ui.api import (
    API_PREFIX,
    API_SCHEMA,
    APIObservation,
    install_level_1_api,
    register_level_1_provider,
)
from src.operator_ui.foundation import _new_envelope
from src.operator_ui.security import AuditUnavailable, install_connected_mode

NOW = datetime(2026, 7, 31, 1, 2, 3, tzinfo=timezone.utc)
HASH = hashlib.sha256(b"verified evidence").hexdigest()
ROUTES = {
    "overview": f"{API_PREFIX}/overview",
    "upcoming_races": f"{API_PREFIX}/races/upcoming",
    "race_detail": f"{API_PREFIX}/races/race-1",
    "recent_predictions": f"{API_PREFIX}/predictions/recent",
    "prediction_detail": f"{API_PREFIX}/predictions/prediction-1",
    "collector": f"{API_PREFIX}/collector",
    "corpus": f"{API_PREFIX}/corpus",
    "models": f"{API_PREFIX}/models",
    "system": f"{API_PREFIX}/system",
    "audit": f"{API_PREFIX}/audit",
}
DEFAULT_POLICIES = {
    "overview": "P-OPS-5",
    "upcoming_races": "P-UPCOMING-300-PREJUMP",
    "race_detail": "P-UPCOMING-300-PREJUMP",
    "recent_predictions": "P-BUNDLE-LIST-60",
    "prediction_detail": "P-IMMUTABLE-HISTORICAL",
    "collector": "P-COLLECTOR-AGGREGATE",
    "corpus": "P-REPORT-24H",
    "models": "P-CATALOG-60",
    "system": "P-DEPLOY-60",
    "audit": "P-OPS-5",
}


def app_for(tmp_path, *, clock=lambda: NOW):
    app = Flask(__name__)
    app.config.update(
        TESTING=True,
        OPERATOR_UI_CONNECTED_MODE=True,
        OPERATOR_UI_SECRET_KEY="stable-connected-secret-" + "x" * 32,
        OPERATOR_UI_USERNAME="viewer",
        OPERATOR_UI_PASSWORD_HASH=generate_password_hash("correct horse"),
        OPERATOR_UI_LEVEL=1,
        OPERATOR_UI_AUDIT_DB_PATH=str(tmp_path / "audit.sqlite3"),
        OPERATOR_UI_JOB_DB_PATH=str(tmp_path / "jobs.sqlite3"),
        DATABASE_PATH=str(tmp_path / "canonical.sqlite3"),
        OPERATOR_UI_DEPLOYED_COMMIT="c" * 40,
        OPERATOR_UI_DEPLOYED_TREE="d" * 40,
        OPERATOR_UI_DEPLOYED_VERSION="test-v1",
        OPERATOR_UI_CLOCK=clock,
    )
    install_connected_mode(app)
    assert install_level_1_api(app)
    return app


def login(client):
    token = client.get("/operator-ui/login").get_json()["csrf_token"]
    assert client.post(
        "/operator-ui/login",
        data={"username": "viewer", "password": "correct horse", "csrf_token": token},
    ).status_code == 200


def evidence(policy="P-OPS-5", status="AVAILABLE/FRESH", age=3.0, **changes):
    missing = status == "UNAVAILABLE/DATA_MISSING"
    values = dict(
        source_kind="fixture_json",
        source_identity="fixture.operator_ui",
        content_sha256=None if missing else HASH,
        source_locator="configured_fixture",
        source_at=None if missing else "2026-07-31T01:02:00Z",
        generated_at=None,
        observed_at=None,
        server_observed_at="2026-07-31T01:02:03.000000Z",
        age_seconds=None if missing else age,
        freshness_policy=policy,
        availability="missing" if missing else "present",
        schema_integrity="unknown" if missing else "valid",
        reference_hashes=() if missing else (("manifest", HASH),),
        evidence_identity=None if missing else (("fixture_id", "fixture-1"),),
        status=status,
        supported_claim="Exact fixture identity observed for this test only.",
    )
    values.update(changes)
    return _new_envelope(**values)


def overview(_request_now=NOW, **changes):
    return APIObservation(
        evidence=evidence(**changes),
        data={"sections": [{
            "resource": "system",
            "classification": "AVAILABLE/FRESH",
            "evidence_identity": {"component": "operator-ui"},
        }]},
    )


def race():
    return dict(
        race_id="race-1", source_race_id="source-race-1",
        source_url="https://www.thedogs.com.au/racing/sandown/2026-07-31/1",
        racing_date="2026-07-31", venue="Sandown", meeting_slug="sandown",
        race_number=1, jump_utc="2026-07-31T01:30:00Z",
        source_zone="Australia/Melbourne", distance_metres=None, grade=None,
        runners=[dict(runner_id="runner-1", source_runner_id="source-runner-1",
                      box=1, name="Fast Dog", scratch_state="SCHEDULED")],
        runner_set_sha256=HASH,
    )


def provider_error(response):
    assert response.status_code == 503
    assert response.get_json()["classification"] == "NON_OPERATIONAL/PROVIDER_ERROR"


def audited_provider_error(tmp_path, response):
    provider_error(response)
    assert "data" not in response.get_json()
    with sqlite3.connect(tmp_path / "audit.sqlite3") as connection:
        classification = connection.execute(
            "SELECT response_classification FROM audit_events "
            "ORDER BY sequence DESC LIMIT 1"
        ).fetchone()[0]
    assert classification == "NON_OPERATIONAL/PROVIDER_ERROR"


def test_not_installed_without_security():
    app = Flask(__name__)
    assert install_level_1_api(app) is False


@pytest.mark.parametrize(("resource", "route"), ROUTES.items())
def test_default_is_authenticated_audited_unavailable_without_data(tmp_path, resource, route):
    app = app_for(tmp_path)
    client = app.test_client()
    assert client.get(route).status_code == 401
    login(client)
    response = client.get(route)
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["schema"] == API_SCHEMA
    assert payload["resource"] == resource
    assert payload["classification"] == "UNAVAILABLE/DATA_MISSING"
    assert payload["reason"] == "ADAPTER_NOT_REGISTERED"
    assert "data" not in payload
    assert payload["stale"] is False
    assert payload["server_observed_at"] == "2026-07-31T01:02:03.000000Z"
    assert payload["evidence"] == {
        "source_kind": "server_adapter_registry",
        "source_identity": f"operator_ui.adapter.{resource}.unregistered",
        "content_sha256": None,
        "source_locator": f"operator_ui.adapter_registry.{resource}",
        "source_at": None,
        "generated_at": None,
        "observed_at": None,
        "server_observed_at": "2026-07-31T01:02:03.000000Z",
        "age_seconds": None,
        "freshness_policy": DEFAULT_POLICIES[resource],
        "availability": "missing",
        "schema_integrity": "unknown",
        "reference_hashes": {},
        "evidence_identity": None,
        "status": "UNAVAILABLE/DATA_MISSING",
        "supported_claim": (
            f"No server-owned {resource} adapter was registered for this request."
        ),
    }
    digest = hashlib.sha256(response.data).hexdigest()
    with sqlite3.connect(tmp_path / "audit.sqlite3") as connection:
        row = connection.execute(
            "SELECT route,http_method,authorization_policy,"
            "response_classification,reference_hashes FROM audit_events"
        ).fetchone()
    assert row[:4] == (route, "GET", f"LEVEL_1_API_V1_{resource.upper()}",
                       "UNAVAILABLE/DATA_MISSING")
    assert digest in json.loads(row[4])


def test_late_registry_is_narrow_and_nonreplaceable(tmp_path):
    app = app_for(tmp_path)
    register_level_1_provider(app, "overview", overview)
    with pytest.raises(ValueError):
        register_level_1_provider(app, "overview", overview)
    with pytest.raises(ValueError):
        register_level_1_provider(app, "unknown", overview)
    with pytest.raises(TypeError):
        register_level_1_provider(app, "system", object())  # type: ignore[arg-type]
    client = app.test_client()
    login(client)
    assert client.get(ROUTES["overview"]).get_json()["data"]["sections"]


@pytest.mark.parametrize("suffix", ["?path=", "?status=x", "?provider=x", "?locator=x"])
def test_query_rejected_before_provider(tmp_path, suffix):
    calls = []
    app = app_for(tmp_path)
    register_level_1_provider(app, "overview", lambda _now: calls.append(1) or overview())
    client = app.test_client()
    login(client)
    provider_error(client.get(ROUTES["overview"] + suffix))
    assert calls == []


@pytest.mark.parametrize("kwargs", [
    {"data": b"x", "content_type": "application/octet-stream"},
    {"data": {"status": "x"}},
    {"json": {"path": "/tmp/x"}},
])
def test_get_body_rejected_before_provider(tmp_path, kwargs):
    calls = []
    app = app_for(tmp_path)
    register_level_1_provider(app, "overview", lambda _now: calls.append(1) or overview())
    client = app.test_client()
    login(client)
    provider_error(client.open(ROUTES["overview"], method="GET", **kwargs))
    assert calls == []


@pytest.mark.parametrize("identifier", [
    "slash%2Fescape", "..", ".hidden", "-leading", "space%20id",
    "x" * 129, "%00", "semi%3Bid",
])
@pytest.mark.parametrize("kind", ["races", "predictions"])
def test_identifiers_are_bounded_lexical(tmp_path, identifier, kind):
    app = app_for(tmp_path)
    client = app.test_client()
    login(client)
    assert client.get(f"{API_PREFIX}/{kind}/{identifier}").status_code in {404, 503}


@pytest.mark.parametrize("provider", [
    lambda _now: {}, lambda _now: APIObservation(evidence={}, data={}),
    lambda _now: APIObservation(evidence=object(), data={}),
])
def test_arbitrary_provider_output_fails_closed(tmp_path, provider):
    app = app_for(tmp_path)
    register_level_1_provider(app, "overview", provider)
    client = app.test_client()
    login(client)
    provider_error(client.get(ROUTES["overview"]))


@pytest.mark.parametrize("changes", [
    {"status": "FINE"}, {"age": -1}, {"age": 4},
    {"server_observed_at": "2026-07-31T01:02:04Z"},
    {"policy": "P-REPORT-24H"}, {"status": "STALE"},
])
def test_envelope_policy_time_age_and_status_invariants(tmp_path, changes):
    app = app_for(tmp_path)
    register_level_1_provider(app, "overview", lambda now: overview(now, **changes))
    client = app.test_client()
    login(client)
    provider_error(client.get(ROUTES["overview"]))


def test_fixed_stale_is_derived_and_discloses_no_data(tmp_path):
    item = APIObservation(
        evidence=evidence("P-DEPLOY-60", status="STALE", age=63), data={}
    )
    object.__setattr__(item.evidence, "source_at", "2026-07-31T01:01:00Z")
    app = app_for(tmp_path)
    register_level_1_provider(app, "system", lambda _now: item)
    client = app.test_client()
    login(client)
    payload = client.get(ROUTES["system"]).get_json()
    assert payload["classification"] == "STALE" and payload["stale"] is True
    assert "data" not in payload


def test_available_race_exact_binding_and_truthful_empty(tmp_path):
    app = app_for(tmp_path)
    register_level_1_provider(
        app, "upcoming_races",
        lambda _now: APIObservation(
            evidence("P-UPCOMING-300-PREJUMP"), {"races": [race()]}
        )
    )
    client = app.test_client()
    login(client)
    assert client.get(ROUTES["upcoming_races"]).get_json()["data"]["races"][0][
        "runner_set_sha256"] == HASH
    app2 = app_for(tmp_path / "empty")
    register_level_1_provider(
        app2, "upcoming_races",
        lambda _now: APIObservation(
            evidence("P-UPCOMING-300-PREJUMP"), {"races": []}
        )
    )
    client2 = app2.test_client()
    login(client2)
    assert client2.get(ROUTES["upcoming_races"]).get_json()["data"] == {"races": []}


def test_sealed_race_schema_preserves_exact_primary_identity_active_and_nullable_runner(tmp_path):
    item = race()
    item.update(
        route_id="r1.UmFjZSAxIC0gV0FSUk5BTUJPT0wgLSAyMDI2LTA3LTMw",
        race_id="Race 1 - WARRNAMBOOL - 2026-07-30",
        source_race_id="Race 1 - WARRNAMBOOL - 2026-07-30",
        meeting_slug=None,
    )
    item["runners"][0].update(source_runner_id=None, scratch_state="ACTIVE")
    app = app_for(tmp_path)
    register_level_1_provider(
        app, "upcoming_races",
        lambda _now: APIObservation(evidence("P-UPCOMING-300-PREJUMP"), {"races": [item]}),
    )
    client = app.test_client(); login(client)
    disclosed = client.get(ROUTES["upcoming_races"]).get_json()["data"]["races"][0]
    assert disclosed["race_id"] == item["race_id"]
    assert disclosed["route_id"] == item["route_id"]
    assert disclosed["runners"][0]["scratch_state"] == "ACTIVE"
    assert disclosed["runners"][0]["source_runner_id"] is None


@pytest.mark.parametrize("mutation", [
    lambda x: x.update(extra="x"),
    lambda x: x.update(source_url="https://example.com/race"),
    lambda x: x.update(jump_utc="2026-07-31T01:02:03Z"),
    lambda x: x.update(runner_set_sha256="bad"),
    lambda x: x["runners"][0].update(scratch_state="MAYBE"),
])
def test_race_schema_and_prejump_fail_closed(tmp_path, mutation):
    item = race()
    mutation(item)
    app = app_for(tmp_path)
    register_level_1_provider(
        app, "upcoming_races",
        lambda _now: APIObservation(
            evidence("P-UPCOMING-300-PREJUMP"), {"races": [item]}
        )
    )
    client = app.test_client()
    login(client)
    provider_error(client.get(ROUTES["upcoming_races"]))


def test_duplicate_source_runner_identity_fails_closed_and_is_audited(tmp_path):
    item = race()
    item["runners"].append(
        dict(
            runner_id="runner-2",
            source_runner_id="source-runner-1",
            box=2,
            name="Other Dog",
            scratch_state="SCHEDULED",
        )
    )
    app = app_for(tmp_path)
    register_level_1_provider(
        app,
        "upcoming_races",
        lambda _now: APIObservation(
            evidence("P-UPCOMING-300-PREJUMP"), {"races": [item]}
        ),
    )
    client = app.test_client()
    login(client)
    audited_provider_error(tmp_path, client.get(ROUTES["upcoming_races"]))


@pytest.mark.parametrize("duplicate_field", ["source_race_id", "source_url"])
def test_duplicate_upcoming_source_identity_fails_closed_and_is_audited(
    tmp_path, duplicate_field
):
    first = race()
    second = race()
    second.update(race_id="race-2", race_number=2)
    second["runners"][0].update(
        runner_id="runner-2", source_runner_id="source-runner-2", box=2
    )
    if duplicate_field != "source_race_id":
        second["source_race_id"] = "source-race-2"
    if duplicate_field != "source_url":
        second["source_url"] = (
            "https://www.thedogs.com.au/racing/sandown/2026-07-31/2"
        )
    app = app_for(tmp_path)
    register_level_1_provider(
        app,
        "upcoming_races",
        lambda _now: APIObservation(
            evidence("P-UPCOMING-300-PREJUMP"), {"races": [first, second]}
        ),
    )
    client = app.test_client()
    login(client)
    audited_provider_error(tmp_path, client.get(ROUTES["upcoming_races"]))


def test_unavailable_provider_has_no_data(tmp_path):
    app = app_for(tmp_path)
    register_level_1_provider(
        app, "overview",
        lambda _now: APIObservation(
            evidence(status="UNAVAILABLE/DATA_MISSING"), {}
        )
    )
    client = app.test_client()
    login(client)
    assert "data" not in client.get(ROUTES["overview"]).get_json()


def test_audit_failure_withholds_exact_bytes(tmp_path, monkeypatch):
    app = app_for(tmp_path)
    register_level_1_provider(app, "overview", overview)
    monkeypatch.setattr(
        app.extensions["operator_ui_audit"], "append_and_confirm",
        lambda _event: (_ for _ in ()).throw(AuditUnavailable("forced")),
    )
    client = app.test_client()
    login(client)
    response = client.get(ROUTES["overview"])
    assert response.status_code == 503
    assert b"fixture.operator_ui" not in response.data


def test_reads_only_mutate_audit_and_routes_are_get_only(tmp_path):
    canonical, jobs = tmp_path / "canonical.sqlite3", tmp_path / "jobs.sqlite3"
    canonical.write_bytes(b"canonical")
    jobs.write_bytes(b"jobs")
    before = canonical.read_bytes(), jobs.read_bytes()
    app = app_for(tmp_path)
    client = app.test_client()
    login(client)
    for route in ROUTES.values():
        assert client.get(route).status_code == 200
        assert client.post(route).status_code == 405
    assert (canonical.read_bytes(), jobs.read_bytes()) == before


def prediction():
    return {
        "prediction_id": "prediction-1",
        "job_id": "job-1",
        "race_id": "race-1",
        "model_id": "model-1",
        "model_sha256": HASH,
        "config_id": "config-1",
        "config_sha256": HASH,
        "lifecycle_status": "PREDICTION_READY",
        "probabilities": [
            {"runner_id": "runner-1", "probability": 0.4},
            {"runner_id": "runner-2", "probability": 0.6},
        ],
        "bundle_sha256": HASH,
        "evidence_identities": {"race_id": "race-1", "bundle_id": "bundle-1"},
    }


def test_sealed_prediction_nullable_identities_do_not_weaken_legacy_schema(tmp_path):
    sealed = prediction()
    sealed.pop("lifecycle_status")
    sealed.update(
        job_id=None, model_sha256=None, terminal_status="PREDICTION_BLOCKED",
        blocker_stage="VALIDATION", blocker_code="POST_JUMP",
        probabilities=None,
        evidence_names=["bundle_manifest.json", "result.json"],
    )
    app = app_for(tmp_path)
    register_level_1_provider(
        app, "recent_predictions",
        lambda _now: APIObservation(evidence("P-BUNDLE-LIST-60"), {"predictions": [sealed]}),
    )
    client = app.test_client(); login(client)
    disclosed = client.get(ROUTES["recent_predictions"]).get_json()["data"]["predictions"][0]
    assert disclosed["job_id"] is None and disclosed["model_sha256"] is None
    assert disclosed["blocker_stage"] == "VALIDATION" and disclosed["probabilities"] is None

    legacy = prediction(); legacy["job_id"] = None
    app2 = app_for(tmp_path / "legacy")
    register_level_1_provider(
        app2, "recent_predictions",
        lambda _now: APIObservation(evidence("P-BUNDLE-LIST-60"), {"predictions": [legacy]}),
    )
    client2 = app2.test_client(); login(client2)
    provider_error(client2.get(ROUTES["recent_predictions"]))


def collector_lane():
    return {
        "lane": "FULL_DAEMON",
        "status": "ACTIVE",
        "run_id": "run-1",
        "phase": "DAEMON_RUNNING",
        "cycle_state": "ACTIVE",
        "deadline_utc": "2026-07-31T01:03:00Z",
        "state_age_seconds": 3,
        "component_identity": {"service": "shadow-autopilot.service"},
        "reference_hashes": {"state": HASH},
        "operational_context": {
            "final_status": None,
            "final_verdict": "DAEMON_RUNNING",
            "status": "RUNNING",
            "next_meaningful_action": None,
            "next_meaningful_action_at": None,
            "lock_owner": None,
            "recent_capture": {
                "inserted_live_odds_rows": None,
                "ready_count": None,
                "status_counts": None,
                "blocked_attempt_count": None,
            },
        },
    }


def collector_lanes():
    full = collector_lane()
    odds = deepcopy(full)
    odds.update(lane="ODDS_ONLY", run_id="run-2")
    return [full, odds]


def corpus_report():
    return {
        "report_id": "report-1",
        "population_id": "population-1",
        "population_count": 12,
        "funnel_counts": {"source_safe": 12, "evaluated": 8},
        "exclusions": [{"reason": "missing_odds", "count": 4}],
        "chain_hashes": {"population": HASH, "report": HASH},
        "generated_at": "2026-07-31T01:02:00Z",
        "status": "ADMISSIBLE",
        "admission_gap": "none; approved admission contract passed",
    }


def model():
    return {
        "model_id": "model-1",
        "model_sha256": HASH,
        "config_id": "config-1",
        "config_sha256": HASH,
        "manifest_sha256": HASH,
        "role": "BASELINE",
        "evaluation_status": "REPORTED",
        "evaluation_claim": "Exact held-out slice metrics only.",
        "slice_id": "slice-1",
        "evaluation_hashes": {"report": HASH},
    }


def component():
    return {
        "component": "operator-ui",
        "status": "HEALTHY",
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "deployed_commit": "a" * 40,
        "deployed_tree": "b" * 40,
        "version": "test-v1",
        "observed_at": "2026-07-31T01:02:00Z",
        "age_seconds": 3,
        "reference_hashes": {"manifest": HASH},
        "service_status": {
            "full": {
                "active_state": "inactive",
                "sub_state": "dead",
                "exec_main_pid": 0,
            },
            "odds": {
                "active_state": "active",
                "sub_state": "waiting",
                "exec_main_pid": 0,
            },
        },
    }


@pytest.mark.parametrize(
    ("status", "lane_status"),
    [
        ("STALE", "STALE"),
        ("UNAVAILABLE/DATA_MISSING", "DATA_MISSING"),
        ("DIVERGENT", "DIVERGENT"),
    ],
)
def test_collector_non_healthy_discloses_bounded_lane_states(
    tmp_path, status, lane_status
):
    lane = collector_lane()
    lane["status"] = lane_status
    lane["deadline_utc"] = None
    lane["state_age_seconds"] = None
    if lane_status == "DATA_MISSING":
        lane["reference_hashes"] = {}
    envelope = evidence("P-COLLECTOR-AGGREGATE", status=status)
    if status == "UNAVAILABLE/DATA_MISSING":
        envelope = evidence(
            "P-COLLECTOR-AGGREGATE",
            status=status,
            content_sha256=HASH,
            source_at="2026-07-31T01:02:00Z",
            age_seconds=3,
            availability="present",
            schema_integrity="valid",
            reference_hashes=(("state", HASH),),
            evidence_identity=(("lane", "aggregate"),),
        )
    other_lane = deepcopy(collector_lane())
    other_lane.update(lane="ODDS_ONLY", run_id="run-2")
    app = app_for(tmp_path)
    register_level_1_provider(
        app,
        "collector",
        lambda _now: APIObservation(envelope, {"lanes": [lane, other_lane]}),
    )
    client = app.test_client()
    login(client)
    payload = client.get(ROUTES["collector"]).get_json()
    assert payload["classification"] == status
    assert [(item["lane"], item["status"]) for item in payload["data"]["lanes"]] == [
        ("FULL_DAEMON", lane_status),
        ("ODDS_ONLY", "ACTIVE"),
    ]


@pytest.mark.parametrize(
    ("classification", "component_status"),
    [
        ("UNAVAILABLE/DATA_MISSING", "DEGRADED"),
        ("STALE", "STALE"),
        ("DIVERGENT", "DIVERGENT"),
    ],
)
def test_system_non_healthy_discloses_bounded_component_context(
    tmp_path, classification, component_status
):
    item = component()
    item["status"] = component_status
    envelope_changes = {}
    if component_status == "STALE":
        item.update(observed_at="2026-07-31T01:01:02Z", age_seconds=61)
        envelope_changes.update(source_at="2026-07-31T01:01:02Z", age_seconds=61)
    elif component_status == "DIVERGENT":
        item["deployed_tree"] = "d" * 40
    elif component_status == "DEGRADED":
        envelope_changes.update(
            content_sha256=HASH,
            source_at="2026-07-31T01:02:00Z",
            age_seconds=3,
            availability="present",
            schema_integrity="valid",
            reference_hashes=(("manifest", HASH),),
            evidence_identity=(("component", "operator-ui"),),
        )
    envelope = evidence("P-DEPLOY-60", status=classification, **envelope_changes)
    app = app_for(tmp_path)
    register_level_1_provider(
        app,
        "system",
        lambda _now: APIObservation(envelope, {"components": [item]}),
    )
    client = app.test_client()
    login(client)
    payload = client.get(ROUTES["system"]).get_json()
    assert payload["classification"] == classification
    assert payload["data"]["components"][0]["status"] == component_status
    assert payload["data"]["components"][0]["service_status"]["odds"] == {
        "active_state": "active",
        "exec_main_pid": 0,
        "sub_state": "waiting",
    }


def test_live_system_unavailable_projection_reaches_registered_level_1_api(tmp_path):
    from tests.operator_ui.test_live_adapters import NOW as LIVE_NOW, actual_payloads, make_live

    values = actual_payloads()
    values["deployment_manifest"]["deployed_tree"] = None
    live = make_live(tmp_path / "evidence", values)
    app = app_for(tmp_path, clock=lambda: LIVE_NOW)
    register_level_1_provider(app, "system", live.system)
    client = app.test_client()
    login(client)

    payload = client.get(ROUTES["system"]).get_json()

    assert payload["classification"] == "UNAVAILABLE/DATA_MISSING"
    assert payload["data"]["components"][0]["status"] == "DEGRADED"
    assert payload["data"]["components"][0]["deployed_tree"] is None
    assert payload["data"]["components"][0]["reference_hashes"] is None


@pytest.mark.parametrize("lane", ["full", "odds"])
@pytest.mark.parametrize("field", ["active_state", "sub_state", "exec_main_pid"])
@pytest.mark.parametrize(
    ("age", "classification", "component_status"),
    [
        (timedelta(seconds=60), "UNAVAILABLE/DATA_MISSING", "DEGRADED"),
        (timedelta(seconds=60, microseconds=1), "STALE", "STALE"),
    ],
)
def test_registered_level_1_accepts_nullable_service_at_deployment_boundary(
    tmp_path, lane, field, age, classification, component_status
):
    from tests.operator_ui.test_live_adapters import NOW as LIVE_NOW, actual_payloads, make_live

    observed = LIVE_NOW - age
    live = make_live(
        tmp_path / "evidence",
        actual_payloads(at=observed),
        units_observed_at=observed,
    )
    live._units = replace(live._units, **{f"{lane}_{field}": None})
    app = app_for(tmp_path, clock=lambda: LIVE_NOW)
    register_level_1_provider(app, "system", live.system)
    client = app.test_client()
    login(client)

    payload = client.get(ROUTES["system"]).get_json()

    assert payload["classification"] == classification
    component = payload["data"]["components"][0]
    assert component["status"] == component_status
    assert component["service_status"][lane][field] is None


@pytest.mark.parametrize("lane", ["full", "odds"])
@pytest.mark.parametrize("field", ["active_state", "sub_state", "exec_main_pid"])
def test_registered_level_1_preserves_divergence_over_stale_incomplete_service(
    tmp_path, lane, field
):
    from tests.operator_ui.test_live_adapters import (
        NOW as LIVE_NOW,
        _set_deployment_divergence,
        actual_payloads,
        make_live,
    )

    observed = LIVE_NOW - timedelta(seconds=60, microseconds=1)
    root = tmp_path / "evidence"
    live = make_live(root, actual_payloads(at=observed), units_observed_at=observed)
    live._units = replace(live._units, **{f"{lane}_{field}": None})
    _set_deployment_divergence(live, root, "source_commit")
    app = app_for(tmp_path, clock=lambda: LIVE_NOW)
    register_level_1_provider(app, "system", live.system)
    client = app.test_client()
    login(client)

    payload = client.get(ROUTES["system"]).get_json()

    assert payload["classification"] == "DIVERGENT"
    component = payload["data"]["components"][0]
    assert component["status"] == "DIVERGENT"
    assert component["service_status"][lane][field] is None


def test_level_1_preserves_zero_and_nullable_unknown_lane_freshness(tmp_path):
    missing = collector_lane()
    missing.update(
        status="DATA_MISSING", deadline_utc=None, state_age_seconds=None,
        reference_hashes={},
    )
    zero = collector_lane()
    zero.update(lane="ODDS_ONLY", run_id="run-2", state_age_seconds=0)
    envelope = evidence(
        "P-COLLECTOR-AGGREGATE", status="UNAVAILABLE/DATA_MISSING",
        content_sha256=HASH, source_at="2026-07-31T01:02:00Z", age_seconds=3,
        availability="present", schema_integrity="valid",
        reference_hashes=(("state", HASH),),
        evidence_identity=(("lane", "aggregate"),),
    )
    app = app_for(tmp_path)
    register_level_1_provider(
        app, "collector", lambda _now: APIObservation(envelope, {"lanes": [missing, zero]})
    )
    client = app.test_client()
    login(client)

    lanes = client.get(ROUTES["collector"]).get_json()["data"]["lanes"]

    assert lanes[0]["deadline_utc"] is None
    assert lanes[0]["state_age_seconds"] is None
    assert lanes[1]["state_age_seconds"] == 0


def test_live_system_malformed_projection_is_suppressed_by_registered_api(tmp_path):
    from tests.operator_ui.test_live_adapters import NOW as LIVE_NOW, actual_payloads, make_live

    values = actual_payloads()
    values["deployment_manifest"]["deployed_tree"] = "malformed"
    live = make_live(tmp_path / "evidence", values)
    app = app_for(tmp_path, clock=lambda: LIVE_NOW)
    register_level_1_provider(app, "system", live.system)
    client = app.test_client()
    login(client)

    payload = client.get(ROUTES["system"]).get_json()

    assert payload["classification"] == "INVALID/INTEGRITY_FAILED"
    assert payload["data"] == {}


def test_invalid_collector_payload_remains_suppressed(tmp_path):
    app = app_for(tmp_path)
    register_level_1_provider(
        app,
        "collector",
        lambda _now: APIObservation(
            evidence(
                "P-COLLECTOR-AGGREGATE",
                status="INVALID/INTEGRITY_FAILED",
                source_at=None,
                age_seconds=None,
                schema_integrity="failed",
                evidence_identity=None,
            ),
            {"lanes": [collector_lane()]},
        ),
    )
    client = app.test_client()
    login(client)
    audited_provider_error(tmp_path, client.get(ROUTES["collector"]))


@pytest.mark.parametrize(
    "lanes",
    [
        [],
        [collector_lane()],
        [dict(collector_lane(), lane="ODDS_ONLY")],
        [collector_lane(), dict(collector_lane(), run_id="run-2")],
        [
            dict(collector_lane(), lane="ODDS_ONLY"),
            dict(collector_lane(), lane="ODDS_ONLY", run_id="run-2"),
        ],
        [collector_lane(), dict(collector_lane(), lane="OTHER", run_id="run-2")],
    ],
    ids=[
        "missing-both",
        "single-full",
        "single-odds",
        "duplicate-full",
        "duplicate-odds",
        "unknown-lane",
    ],
)
def test_collector_requires_exact_two_lane_aggregate(tmp_path, lanes):
    app = app_for(tmp_path)
    register_level_1_provider(
        app,
        "collector",
        lambda _now: APIObservation(
            evidence("P-COLLECTOR-AGGREGATE"), {"lanes": lanes}
        ),
    )
    client = app.test_client()
    login(client)

    audited_provider_error(tmp_path, client.get(ROUTES["collector"]))


@pytest.mark.parametrize("lane_status", ["ACTIVE"])
def test_collector_current_lane_rejects_passed_deadline(tmp_path, lane_status):
    lanes = collector_lanes()
    lanes[0].update(
        status=lane_status,
        deadline_utc=(NOW - timedelta(microseconds=1)).isoformat().replace(
            "+00:00", "Z"
        ),
    )
    app = app_for(tmp_path)
    register_level_1_provider(
        app,
        "collector",
        lambda _now: APIObservation(
            evidence("P-COLLECTOR-AGGREGATE"), {"lanes": lanes}
        ),
    )
    client = app.test_client()
    login(client)

    audited_provider_error(tmp_path, client.get(ROUTES["collector"]))


@pytest.mark.parametrize(
    "lane_status",
    [
        "RECEIPT_READY",
        "REQUEST_EXPIRED",
        "RACE_NOT_FOUND",
        "CAPTURE_WINDOW_CLOSED",
        "IDENTITY_MISMATCH",
        "CAPTURE_FAILED",
    ],
)
@pytest.mark.parametrize("lane_kind", ["FULL_DAEMON", "ODDS_ONLY"])
def test_collector_terminal_lane_accepts_deadline_equality(
    tmp_path, lane_status, lane_kind
):
    lanes = collector_lanes()
    lane = next(lane for lane in lanes if lane["lane"] == lane_kind)
    equality_deadline = NOW.isoformat().replace("+00:00", "Z")
    lane.update(
        status=lane_status,
        deadline_utc=equality_deadline,
    )
    app = app_for(tmp_path)
    register_level_1_provider(
        app,
        "collector",
        lambda _now: APIObservation(
            evidence("P-COLLECTOR-AGGREGATE"), {"lanes": lanes}
        ),
    )
    client = app.test_client()
    login(client)

    response = client.get(ROUTES["collector"])

    assert response.status_code == 200
    response_lanes = response.get_json()["data"]["lanes"]
    assert len(response_lanes) == 2
    assert [lane["lane"] for lane in response_lanes].count("FULL_DAEMON") == 1
    assert [lane["lane"] for lane in response_lanes].count("ODDS_ONLY") == 1
    run_ids = [lane["run_id"] for lane in response_lanes]
    assert all(run_ids)
    assert run_ids[0] != run_ids[1]
    selected_lane = next(lane for lane in response_lanes if lane["lane"] == lane_kind)
    assert selected_lane["status"] == lane_status
    assert selected_lane["deadline_utc"] == equality_deadline


@pytest.mark.parametrize(
    "lane_status",
    [
        "RECEIPT_READY",
        "REQUEST_EXPIRED",
        "RACE_NOT_FOUND",
        "CAPTURE_WINDOW_CLOSED",
        "IDENTITY_MISMATCH",
        "CAPTURE_FAILED",
    ],
)
@pytest.mark.parametrize("lane_kind", ["FULL_DAEMON", "ODDS_ONLY"])
def test_collector_terminal_lane_rejects_one_microsecond_past_deadline(
    tmp_path, lane_status, lane_kind
):
    lanes = collector_lanes()
    lane = next(lane for lane in lanes if lane["lane"] == lane_kind)
    lane.update(
        status=lane_status,
        deadline_utc=(NOW - timedelta(microseconds=1)).isoformat().replace(
            "+00:00", "Z"
        ),
    )
    app = app_for(tmp_path)
    register_level_1_provider(
        app,
        "collector",
        lambda _now: APIObservation(
            evidence("P-COLLECTOR-AGGREGATE"), {"lanes": lanes}
        ),
    )
    client = app.test_client()
    login(client)

    audited_provider_error(tmp_path, client.get(ROUTES["collector"]))


def audit_event():
    return {
        "event_id": "event-1",
        "event_time_utc": "2026-07-31T01:02:00Z",
        "classification": "AVAILABLE/FRESH",
        "event_hash": HASH,
        "previous_event_hash": None,
        "segment_id": "segment-1",
        "reference_hashes": {"response": HASH},
    }


HAPPY_CASES = [
    ("upcoming_races", "P-UPCOMING-300-PREJUMP", {"races": [race()]}),
    ("race_detail", "P-UPCOMING-300-PREJUMP", {"race": race()}),
    ("recent_predictions", "P-BUNDLE-LIST-60", {"predictions": [prediction()]}),
    (
        "prediction_detail",
        "P-IMMUTABLE-HISTORICAL",
        {"prediction": prediction()},
    ),
    ("collector", "P-COLLECTOR-AGGREGATE", {"lanes": collector_lanes()}),
    ("corpus", "P-REPORT-24H", {"reports": [corpus_report()]}),
    ("models", "P-CATALOG-60", {"models": [model()]}),
    ("system", "P-DEPLOY-60", {"components": [component()]}),
    ("audit", "P-OPS-5", {"events": [audit_event()]}),
]


@pytest.mark.parametrize(("resource", "policy", "data"), HAPPY_CASES)
def test_all_resource_schemas_accept_stable_happy_paths(
    tmp_path, resource, policy, data
):
    app = app_for(tmp_path)
    if resource in {"race_detail", "prediction_detail"}:
        provider = lambda _route_id, _now: APIObservation(evidence(policy), data)
    else:
        provider = lambda _now: APIObservation(evidence(policy), data)
    register_level_1_provider(app, resource, provider)
    client = app.test_client()
    login(client)
    payload = client.get(ROUTES[resource]).get_json()
    assert payload["classification"] == "AVAILABLE/FRESH"
    assert payload["data"]


def test_advancing_clock_identity_is_passed_once_to_provider(tmp_path):
    calls = []
    current = NOW

    def advancing_clock():
        nonlocal current
        result = current
        current = current.replace(microsecond=current.microsecond + 1)
        return result

    app = app_for(tmp_path, clock=advancing_clock)

    def provider(request_now):
        calls.append(request_now)
        observed = request_now.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        )
        return overview(
            request_now,
            server_observed_at=observed,
            source_at="2026-07-31T01:02:00Z",
            age=(request_now - NOW.replace(second=0)).total_seconds(),
        )

    register_level_1_provider(app, "overview", provider)
    client = app.test_client()
    login(client)
    payload = client.get(ROUTES["overview"]).get_json()
    assert len(calls) == 1
    assert payload["server_observed_at"] == calls[0].isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def test_detail_provider_receives_route_and_request_identity(tmp_path):
    observed = []
    app = app_for(tmp_path)

    def provider(route_id, request_now):
        observed.append((route_id, request_now))
        return APIObservation(evidence("P-UPCOMING-300-PREJUMP"), {"race": race()})

    register_level_1_provider(app, "race_detail", provider)
    client = app.test_client()
    login(client)
    assert client.get(ROUTES["race_detail"]).status_code == 200
    assert observed == [("race-1", NOW)]


def test_available_response_byte_hash_is_durably_audited(tmp_path):
    app = app_for(tmp_path)
    register_level_1_provider(app, "overview", overview)
    client = app.test_client()
    login(client)
    response = client.get(ROUTES["overview"])
    digest = hashlib.sha256(response.data).hexdigest()
    with sqlite3.connect(tmp_path / "audit.sqlite3") as connection:
        references = json.loads(
            connection.execute(
                "SELECT reference_hashes FROM audit_events "
                "WHERE route = ? ORDER BY sequence DESC LIMIT 1",
                (ROUTES["overview"],),
            ).fetchone()[0]
        )
    assert digest in references


@pytest.mark.parametrize(
    ("label", "item", "expected"),
    [
        (
            "missing",
            APIObservation(
                evidence(status="UNAVAILABLE/DATA_MISSING"),
                {},
            ),
            "UNAVAILABLE/DATA_MISSING",
        ),
        (
            "unreadable",
            APIObservation(
                evidence(
                    status="UNAVAILABLE/DATA_MISSING",
                    availability="unreadable",
                ),
                {},
            ),
            "UNAVAILABLE/DATA_MISSING",
        ),
        (
            "invalid",
            APIObservation(
                evidence(
                    status="INVALID/INTEGRITY_FAILED",
                    source_at=None,
                    age_seconds=None,
                    availability="present",
                    schema_integrity="failed",
                    evidence_identity=None,
                ),
                {},
            ),
            "INVALID/INTEGRITY_FAILED",
        ),
        (
            "stale",
            APIObservation(
                evidence(
                    status="STALE",
                    source_at="2026-07-31T01:01:57Z",
                    age_seconds=6,
                ),
                {},
            ),
            "STALE",
        ),
        (
            "divergent",
            APIObservation(
                evidence(
                    status="DIVERGENT",
                    source_at=None,
                    age_seconds=None,
                ),
                {},
            ),
            "DIVERGENT",
        ),
    ],
)
def test_no_data_evidence_classifications_remain_distinct(
    tmp_path, label, item, expected
):
    app = app_for(tmp_path)
    register_level_1_provider(app, "overview", lambda _now: item)
    client = app.test_client()
    login(client)
    payload = client.get(ROUTES["overview"]).get_json()
    assert payload["classification"] == expected, label
    assert "data" not in payload


@pytest.mark.parametrize(
    "changes",
    [
        {"source_at": "2026-07-31T01:02:04Z", "age_seconds": 0},
        {"source_at": "2026-07-31T01:02:00Z", "age_seconds": 2},
        {"generated_at": "2026-07-31T01:02:00Z"},
    ],
)
def test_future_inconsistent_or_multiple_evidence_times_fail_closed(
    tmp_path, changes
):
    app = app_for(tmp_path)
    register_level_1_provider(
        app,
        "overview",
        lambda _now: APIObservation(evidence(**changes), {"sections": []}),
    )
    client = app.test_client()
    login(client)
    provider_error(client.get(ROUTES["overview"]))


@pytest.mark.parametrize("field", ["source_commit", "source_tree", "deployed_commit", "deployed_tree"])
@pytest.mark.parametrize("bad_oid", ["a" * 39, "a" * 41, "A" * 40, "g" * 40, "a" * 63])
def test_system_rejects_malformed_git_object_identities(tmp_path, field, bad_oid):
    item = component()
    item[field] = bad_oid
    app = app_for(tmp_path)
    register_level_1_provider(
        app,
        "system",
        lambda _now: APIObservation(
            evidence("P-DEPLOY-60"), {"components": [item]}
        ),
    )
    client = app.test_client()
    login(client)
    provider_error(client.get(ROUTES["system"]))


def test_system_accepts_sha1_and_sha256_git_object_identities(tmp_path):
    item = component()
    item["source_commit"] = "a" * 64
    item["source_tree"] = "b" * 64
    item["deployed_commit"] = "a" * 64
    item["deployed_tree"] = "b" * 64
    app = app_for(tmp_path)
    register_level_1_provider(
        app,
        "system",
        lambda _now: APIObservation(
            evidence("P-DEPLOY-60"), {"components": [item]}
        ),
    )
    client = app.test_client()
    login(client)
    payload = client.get(ROUTES["system"]).get_json()
    assert len(payload["data"]["components"][0]["deployed_commit"]) == 64
    assert len(payload["data"]["components"][0]["source_commit"]) == 64


@pytest.mark.parametrize(
    ("status", "observed_at", "age_seconds"),
    [
        ("HEALTHY", "2026-07-31T01:01:03Z", 60),
        ("DEGRADED", "2026-07-31T01:01:03Z", 60),
        ("STALE", "2026-07-31T01:01:02.999999Z", 60.000001),
    ],
)
def test_system_component_freshness_boundary_is_exact(
    tmp_path, status, observed_at, age_seconds
):
    item = component()
    item.update(status=status, observed_at=observed_at, age_seconds=age_seconds)
    app = app_for(tmp_path)
    register_level_1_provider(
        app,
        "system",
        lambda _now: APIObservation(
            evidence("P-DEPLOY-60"), {"components": [item]}
        ),
    )
    client = app.test_client()
    login(client)
    assert client.get(ROUTES["system"]).status_code == 200


@pytest.mark.parametrize(
    ("observed_at", "age_seconds"),
    [
        ("2026-07-31T01:02:03.000001Z", 0),
        ("2026-07-31T01:02:00Z", 2),
    ],
)
def test_system_component_future_or_inconsistent_age_fails_closed(
    tmp_path, observed_at, age_seconds
):
    item = component()
    item.update(observed_at=observed_at, age_seconds=age_seconds)
    app = app_for(tmp_path)
    register_level_1_provider(
        app,
        "system",
        lambda _now: APIObservation(
            evidence("P-DEPLOY-60"), {"components": [item]}
        ),
    )
    client = app.test_client()
    login(client)
    audited_provider_error(tmp_path, client.get(ROUTES["system"]))


@pytest.mark.parametrize(
    ("status", "observed_at", "age_seconds"),
    [
        ("HEALTHY", "2026-07-31T01:01:02Z", 61),
        ("DEGRADED", "2026-07-31T01:01:02Z", 61),
        ("STALE", "2026-07-31T01:01:03Z", 60),
    ],
)
def test_system_component_status_must_match_freshness(
    tmp_path, status, observed_at, age_seconds
):
    item = component()
    item.update(status=status, observed_at=observed_at, age_seconds=age_seconds)
    app = app_for(tmp_path)
    register_level_1_provider(
        app,
        "system",
        lambda _now: APIObservation(
            evidence("P-DEPLOY-60"), {"components": [item]}
        ),
    )
    client = app.test_client()
    login(client)
    audited_provider_error(tmp_path, client.get(ROUTES["system"]))


@pytest.mark.parametrize("status", ["HEALTHY", "DEGRADED", "STALE"])
def test_system_identity_mismatch_requires_divergent_status(tmp_path, status):
    item = component()
    item.update(status=status, deployed_commit="c" * 40)
    if status == "STALE":
        item.update(observed_at="2026-07-31T01:01:02Z", age_seconds=61)
    app = app_for(tmp_path)
    register_level_1_provider(
        app,
        "system",
        lambda _now: APIObservation(
            evidence("P-DEPLOY-60"), {"components": [item]}
        ),
    )
    client = app.test_client()
    login(client)
    audited_provider_error(tmp_path, client.get(ROUTES["system"]))


def test_system_identity_mismatch_is_accepted_as_divergent(tmp_path):
    item = component()
    item.update(status="DIVERGENT", deployed_tree="d" * 40)
    app = app_for(tmp_path)
    register_level_1_provider(
        app,
        "system",
        lambda _now: APIObservation(
            evidence("P-DEPLOY-60"), {"components": [item]}
        ),
    )
    client = app.test_client()
    login(client)
    payload = client.get(ROUTES["system"]).get_json()
    assert payload["data"]["components"][0]["status"] == "DIVERGENT"


@pytest.mark.parametrize(
    ("resource", "policy", "key", "item"),
    [
        ("upcoming_races", "P-UPCOMING-300-PREJUMP", "races", race()),
        ("recent_predictions", "P-BUNDLE-LIST-60", "predictions", prediction()),
        ("collector", "P-COLLECTOR-AGGREGATE", "lanes", collector_lane()),
        ("corpus", "P-REPORT-24H", "reports", corpus_report()),
        ("models", "P-CATALOG-60", "models", model()),
        ("system", "P-DEPLOY-60", "components", component()),
        ("audit", "P-OPS-5", "events", audit_event()),
    ],
)
def test_unknown_fields_fail_closed_for_every_collection_schema(
    tmp_path, resource, policy, key, item
):
    item = deepcopy(item)
    item["unknown"] = "forbidden"
    app = app_for(tmp_path)
    register_level_1_provider(
        app,
        resource,
        lambda _now: APIObservation(evidence(policy), {key: [item]}),
    )
    client = app.test_client()
    login(client)
    provider_error(client.get(ROUTES[resource]))


@pytest.mark.parametrize(
    ("resource", "policy", "key", "item", "field"),
    [
        (
            "upcoming_races",
            "P-UPCOMING-300-PREJUMP",
            "races",
            race(),
            "runners",
        ),
        (
            "recent_predictions",
            "P-BUNDLE-LIST-60",
            "predictions",
            prediction(),
            "probabilities",
        ),
    ],
)
def test_duplicate_runner_or_probability_identity_fails_closed(
    tmp_path, resource, policy, key, item, field
):
    item = deepcopy(item)
    item[field].append(deepcopy(item[field][0]))
    if field == "probabilities":
        item[field][0]["probability"] = 0.2
        item[field][1]["probability"] = 0.8
    app = app_for(tmp_path)
    register_level_1_provider(
        app,
        resource,
        lambda _now: APIObservation(evidence(policy), {key: [item]}),
    )
    client = app.test_client()
    login(client)
    provider_error(client.get(ROUTES[resource]))


@pytest.mark.parametrize(
    ("resource", "policy", "key", "item"),
    [
        (
            "overview",
            "P-OPS-5",
            "sections",
            {
                "resource": "system",
                "classification": "AVAILABLE/FRESH",
                "evidence_identity": {"component": "operator-ui"},
            },
        ),
        ("upcoming_races", "P-UPCOMING-300-PREJUMP", "races", race()),
        (
            "recent_predictions",
            "P-BUNDLE-LIST-60",
            "predictions",
            prediction(),
        ),
        ("corpus", "P-REPORT-24H", "reports", corpus_report()),
        ("models", "P-CATALOG-60", "models", model()),
        ("system", "P-DEPLOY-60", "components", component()),
        ("audit", "P-OPS-5", "events", audit_event()),
    ],
)
def test_duplicate_collection_primary_identity_fails_closed(
    tmp_path, resource, policy, key, item
):
    app = app_for(tmp_path)
    register_level_1_provider(
        app,
        resource,
        lambda _now: APIObservation(
            evidence(policy), {key: [deepcopy(item), deepcopy(item)]}
        ),
    )
    client = app.test_client()
    login(client)
    response = client.get(ROUTES[resource])
    provider_error(response)
    assert "data" not in response.get_json()
    with sqlite3.connect(tmp_path / "audit.sqlite3") as connection:
        classification = connection.execute(
            "SELECT response_classification FROM audit_events "
            "WHERE route = ? ORDER BY sequence DESC LIMIT 1",
            (ROUTES[resource],),
        ).fetchone()[0]
    assert classification == "NON_OPERATIONAL/PROVIDER_ERROR"


@pytest.mark.parametrize(
    ("resource", "policy", "key", "item", "field"),
    [
        (
            "upcoming_races",
            "P-UPCOMING-300-PREJUMP",
            "races",
            race(),
            "scratch_state",
        ),
        (
            "recent_predictions",
            "P-BUNDLE-LIST-60",
            "predictions",
            prediction(),
            "lifecycle_status",
        ),
        (
            "collector",
            "P-COLLECTOR-AGGREGATE",
            "lanes",
            collector_lane(),
            "status",
        ),
        ("corpus", "P-REPORT-24H", "reports", corpus_report(), "status"),
        ("models", "P-CATALOG-60", "models", model(), "evaluation_status"),
        ("system", "P-DEPLOY-60", "components", component(), "status"),
        ("audit", "P-OPS-5", "events", audit_event(), "classification"),
    ],
)
def test_wrong_per_resource_status_fails_closed(
    tmp_path, resource, policy, key, item, field
):
    item = deepcopy(item)
    if field == "scratch_state":
        item["runners"][0][field] = "UNKNOWN"
    else:
        item[field] = "UNKNOWN"
    app = app_for(tmp_path)
    register_level_1_provider(
        app,
        resource,
        lambda _now: APIObservation(evidence(policy), {key: [item]}),
    )
    client = app.test_client()
    login(client)
    provider_error(client.get(ROUTES[resource]))


@pytest.mark.parametrize(
    ("resource", "route", "data"),
    [
        ("race_detail", ROUTES["race_detail"], {"race": {**race(), "race_id": "other"}}),
        (
            "prediction_detail",
            ROUTES["prediction_detail"],
            {"prediction": {**prediction(), "prediction_id": "other"}},
        ),
    ],
)
def test_detail_route_identity_mismatch_fails_closed(tmp_path, resource, route, data):
    policy = (
        "P-UPCOMING-300-PREJUMP"
        if resource == "race_detail"
        else "P-IMMUTABLE-HISTORICAL"
    )
    app = app_for(tmp_path)
    register_level_1_provider(
        app,
        resource,
        lambda _route_id, _now: APIObservation(evidence(policy), data),
    )
    client = app.test_client()
    login(client)
    provider_error(client.get(route))


@pytest.mark.parametrize(
    ("resource", "policy", "key", "item", "field"),
    [
        ("upcoming_races", "P-UPCOMING-300-PREJUMP", "races", race(), "runner_set_sha256"),
        ("recent_predictions", "P-BUNDLE-LIST-60", "predictions", prediction(), "bundle_sha256"),
        ("collector", "P-COLLECTOR-AGGREGATE", "lanes", collector_lane(), "reference_hashes"),
        ("corpus", "P-REPORT-24H", "reports", corpus_report(), "chain_hashes"),
        ("models", "P-CATALOG-60", "models", model(), "manifest_sha256"),
        ("audit", "P-OPS-5", "events", audit_event(), "event_hash"),
    ],
)
def test_malformed_sha256_fields_remain_strict(
    tmp_path, resource, policy, key, item, field
):
    item = deepcopy(item)
    if isinstance(item[field], dict):
        item[field][next(iter(item[field]))] = "a" * 40
    else:
        item[field] = "a" * 40
    app = app_for(tmp_path)
    register_level_1_provider(
        app,
        resource,
        lambda _now: APIObservation(evidence(policy), {key: [item]}),
    )
    client = app.test_client()
    login(client)
    provider_error(client.get(ROUTES[resource]))
