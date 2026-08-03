from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
import uuid
from flask import Flask, Response
from werkzeug.security import generate_password_hash

from src.operator_ui.security import (
    AuditEvent,
    AuditStore,
    AuditUnavailable,
    ConnectedModeConfigurationError,
    PreparedDisclosure,
    install_connected_mode,
    load_connected_environment,
    OperationAuditEvent,
)
from src.operator_ui.foundation import EvidenceStatus


NOW = datetime(2026, 7, 31, 1, 0, tzinfo=timezone.utc)
HASH = hashlib.sha256(b"safe metadata only").hexdigest()

def configured_app(tmp_path, *, level=1, clock=None):
    app = Flask(__name__)
    app.config.update(
        TESTING=True,
        OPERATOR_UI_CONNECTED_MODE=True,
        OPERATOR_UI_SECRET_KEY="stable-connected-secret-" + "x" * 32,
        OPERATOR_UI_USERNAME="viewer",
        OPERATOR_UI_PASSWORD_HASH=generate_password_hash("correct horse"),
        OPERATOR_UI_LEVEL=level,
        OPERATOR_UI_AUDIT_DB_PATH=str(tmp_path / "ui-audit.sqlite3"),
        OPERATOR_UI_JOB_DB_PATH=str(tmp_path / "jobs.sqlite3"),
        DATABASE_PATH=str(tmp_path / "canonical.sqlite3"),
        OPERATOR_UI_DEPLOYED_COMMIT="c" * 40,
        OPERATOR_UI_DEPLOYED_TREE="d" * 40,
        OPERATOR_UI_DEPLOYED_VERSION="test-v1",
        OPERATOR_UI_CLOCK=clock or (lambda: NOW),
        OPERATOR_UI_INACTIVITY_SECONDS=60,
        OPERATOR_UI_ABSOLUTE_SECONDS=300,
    )
    install_connected_mode(app)
    return app


def login(client, password="correct horse"):
    token = client.get("/operator-ui/login").get_json()["csrf_token"]
    return client.post(
        "/operator-ui/login",
        data={"username": "viewer", "password": password, "csrf_token": token},
    )


def event(number: int) -> AuditEvent:
    return AuditEvent(
        event_id=str(uuid.uuid5(uuid.NAMESPACE_OID,f"event-{number}")),
        event_time_utc="2026-07-31T01:00:00.000000Z",
        actor_identity="viewer",
        actor_level=1,
        session_identifier=str(uuid.uuid5(uuid.NAMESPACE_OID,"session-safe")),
        request_identifier=str(uuid.uuid5(uuid.NAMESPACE_OID,f"request-{number}")),
        route="/operator-ui/connected/sentinel",
        http_method="GET",
        authorization_decision="allowed",
        authorization_policy="LEVEL_1_CONNECTED_SENTINEL",
        evidence_source_identifiers=("fixture.identity",),
        content_hashes=(HASH,),
        reference_hashes=(HASH,),
        deployed_commit="c" * 40,
        deployed_tree="d" * 40,
        deployed_version="test-v1",
        response_classification=EvidenceStatus.AVAILABLE_FRESH.value,
    )


def operation_event(number: int) -> OperationAuditEvent:
    return OperationAuditEvent(
        event_id=str(uuid.uuid5(uuid.NAMESPACE_OID,f"operation-{number}")), event_time_utc="2026-07-31T01:00:00.000000Z",
        actor_identity="operator", actor_level=2, session_identifier=str(uuid.uuid5(uuid.NAMESPACE_OID,"operation-session")),
        request_identifier=str(uuid.uuid5(uuid.NAMESPACE_OID,f"operation-request-{number}")), client_identity="privacy-client-v1:local",
        operation="manual_prediction_claim", idempotency_key_sha256=HASH,
        job_id="job_safe", race_id="race-safe", runner_set_sha256=HASH,
        model_identity="model-safe", model_sha256=HASH, config_id="config-safe",
        config_sha256=HASH, input_identity_sha256=HASH, prior_state="WAITING_FOR_CLAIM",
        new_state="CLAIMED", status="CONFIRMED", reason="unique_attempt_claim",
        reference_hashes=(HASH,),
    )


def test_operation_audit_is_separate_insert_only_confirmed_and_verified(tmp_path):
    store = AuditStore(tmp_path / "audit.sqlite3", (tmp_path / "canonical.sqlite3", tmp_path / "jobs.sqlite3"))
    assert store.append_operation_and_confirm(operation_event(1))
    assert store.verify_chain()
    with sqlite3.connect(store.path) as db:
        with pytest.raises(sqlite3.IntegrityError): db.execute("UPDATE operation_audit_events SET reason='changed'")
        with pytest.raises(sqlite3.IntegrityError): db.execute("DELETE FROM operation_audit_events")

def test_operation_audit_preserves_collector_stable_race_identity(tmp_path):
    original = operation_event(1)
    values = original.__dict__.copy()
    values["race_id"] = "Race 5 - RICH - 2026-08-01"
    store = AuditStore(tmp_path / "audit.sqlite3")
    assert store.append_operation_and_confirm(OperationAuditEvent(**values))
    assert store.verify_chain()

@pytest.mark.parametrize("field",["event_id","session_identifier","request_identifier"])
@pytest.mark.parametrize("spelling",["upper","braced"])
def test_audit_uuid_fields_require_exact_canonical_serialization(tmp_path,field,spelling):
    original=event(1); values=original.__dict__.copy(); value=values[field]
    values[field]=value.upper() if spelling=="upper" else "{"+value+"}"
    with pytest.raises(AuditUnavailable): AuditStore(tmp_path/"audit.sqlite3").append_and_confirm(AuditEvent(**values))

def test_undeclared_audit_tuple_cannot_be_registered_or_reconstructed(tmp_path):
    import src.operator_ui.security as security
    assert not hasattr(security,"register_test_audit_contract")
    values=event(1).__dict__.copy(); values.update(route="/operator-ui/connected/undeclared",authorization_policy="LEVEL_1_UNDECLARED")
    with pytest.raises(AuditUnavailable): AuditStore(tmp_path/"audit.sqlite3").append_and_confirm(AuditEvent(**values))

def test_access_contract_authority_cannot_be_expanded_at_runtime(tmp_path):
    import src.operator_ui.security as security
    invented=("/operator-ui/connected/invented","GET","LEVEL_1_INVENTED")
    with pytest.raises(AttributeError): security._ACCESS_CONTRACTS.add(invented)
    assert invented not in security._ACCESS_CONTRACTS
    values=event(1).__dict__.copy(); values.update(route=invented[0],http_method=invented[1],authorization_policy=invented[2])
    with pytest.raises(AuditUnavailable): AuditStore(tmp_path/"audit.sqlite3").append_and_confirm(AuditEvent(**values))


def test_operation_audit_tamper_and_append_failure_fail_closed(tmp_path, monkeypatch):
    store = AuditStore(tmp_path / "audit.sqlite3")
    store.append_operation_and_confirm(operation_event(1))
    with sqlite3.connect(store.path) as db:
        db.execute("DROP TRIGGER operation_audit_no_update")
        db.execute("UPDATE operation_audit_events SET reason='tampered'")
    assert store.verify_chain() is False
    with pytest.raises(AuditUnavailable): store.append_operation_and_confirm(operation_event(2))

def test_rehashed_semantically_invalid_operation_row_fails_closed(tmp_path):
    store = AuditStore(tmp_path / "audit.sqlite3")
    store.append_operation_and_confirm(operation_event(1))
    with sqlite3.connect(store.path) as db:
        db.execute("DROP TRIGGER operation_audit_no_update")
        row=db.execute("SELECT * FROM operation_audit_events").fetchone()
        db.execute("UPDATE operation_audit_events SET status='bad status'")
        fields=operation_event(1).fields(); fields["status"]="bad status"
        import json
        fields["reference_hashes"]=list(operation_event(1).reference_hashes)
        bad=hashlib.sha256(json.dumps({**fields,"previous_event_hash":row[-2]},sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        db.execute("UPDATE operation_audit_events SET event_hash=?",(bad,))
    assert store.verify_chain() is False
    with pytest.raises(AuditUnavailable): store.append_operation_and_confirm(operation_event(2))

def test_undeclared_route_and_policy_reviewer_probe_is_rejected(tmp_path):
    store = AuditStore(tmp_path / "audit.sqlite3")
    bad = event(1)
    bad = AuditEvent(**{
        **{name: getattr(bad, name) for name in bad.__dataclass_fields__},
        "route": "/operator-ui/api/v1/undeclared/foo/bar",
        "http_method": "POST",
        "authorization_policy": "LEVEL_1_INVENTED_POLICY",
    })
    with pytest.raises(AuditUnavailable):
        store.append_and_confirm(bad)

_AUDIT_CORRUPTIONS=("sequence","event_id","time","actor","level","session","request","route","method","decision","policy","sources","content","references","commit","tree","version","classification")

@pytest.mark.parametrize("family",_AUDIT_CORRUPTIONS)
def test_corrupt_and_rehash_access_audit_field_family_fails_reopen_and_next_append(tmp_path,family):
    store=AuditStore(tmp_path/"audit.sqlite3"); original=event(1); store.append_and_confirm(original); fields=original.fields()
    changes={
      "event_id":("event_id","not-a-uuid"),"time":("event_time_utc","2026-07-31T01:00:00Z"),"actor":("actor_identity",""),
      "level":("actor_level",9),"session":("session_identifier",""),"request":("request_identifier",""),
      "route":("route","/operator-ui/api/v1/undeclared/foo/bar"),"method":("http_method","POST"),
      "decision":("authorization_decision","maybe"),"policy":("authorization_policy","LEVEL_1_INVENTED_POLICY"),
      "commit":("deployed_commit","g"*40),"tree":("deployed_tree","0"*39),"version":("deployed_version",""),
      "classification":("response_classification","NON_OPERATIONAL/AUTHORIZATION_DENIED"),
    }
    with sqlite3.connect(store.path) as db:
        db.execute("DROP TRIGGER audit_events_no_update")
        if family=="sequence": db.execute("UPDATE audit_events SET sequence=2")
        elif family in {"sources","content","references"}:
            column={"sources":"evidence_source_identifiers","content":"content_hashes","references":"reference_hashes"}[family]
            value=["z","a"] if family=="sources" else ["bad"]
            fields[column]=value; db.execute(f"UPDATE audit_events SET {column}=?",(json.dumps(value,separators=(",",":")),))
        else:
            column,value=changes[family]; fields[column]=value; db.execute(f"UPDATE audit_events SET {column}=?",(value,))
        digest=hashlib.sha256(json.dumps({**fields,"previous_event_hash":"0"*64},sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        db.execute("UPDATE audit_events SET event_hash=?",(digest,))
    assert not store.verify_chain()
    with pytest.raises(AuditUnavailable): store.append_and_confirm(event(2))

_OPERATION_CORRUPTIONS=("sequence","event_id","time","actor","level","session","request","client","operation","idempotency","job","race","runner","model_id","model_hash","config_id","config_hash","input","prior","new","status","reason","references")

@pytest.mark.parametrize("family",_OPERATION_CORRUPTIONS)
def test_corrupt_and_rehash_operation_audit_field_family_fails_closed(tmp_path,family):
    store=AuditStore(tmp_path/"audit.sqlite3"); original=operation_event(1); store.append_operation_and_confirm(original); fields=original.fields()
    columns={"event_id":"event_id","time":"event_time_utc","actor":"actor_identity","level":"actor_level","session":"session_identifier","request":"request_identifier","client":"client_identity","operation":"operation","idempotency":"idempotency_key_sha256","job":"job_id","race":"race_id","runner":"runner_set_sha256","model_id":"model_identity","model_hash":"model_sha256","config_id":"config_id","config_hash":"config_sha256","input":"input_identity_sha256","prior":"prior_state","new":"new_state","status":"status","reason":"reason"}
    hash_families={"idempotency","runner","model_hash","config_hash","input"}
    with sqlite3.connect(store.path) as db:
        db.execute("DROP TRIGGER operation_audit_no_update")
        db.execute("PRAGMA ignore_check_constraints=ON")
        if family=="sequence": db.execute("UPDATE operation_audit_events SET sequence=2")
        elif family=="references": fields["reference_hashes"]=["bad"]; db.execute("UPDATE operation_audit_events SET reference_hashes='[\"bad\"]'")
        else:
            column=columns[family]
            if family=="event_id": value="not-a-uuid"
            elif family=="time": value="2026-07-31T01:00:00Z"
            elif family=="level": value=1
            elif family in hash_families: value="bad"
            elif family=="job": value="not-job"
            elif family=="race": value="not-race"
            else: value=""
            fields[column]=value; db.execute(f"UPDATE operation_audit_events SET {column}=?",(value,))
        digest=hashlib.sha256(json.dumps({**fields,"previous_event_hash":"0"*64},sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        db.execute("UPDATE operation_audit_events SET event_hash=?",(digest,))
    assert not store.verify_chain()
    with pytest.raises(AuditUnavailable): store.append_operation_and_confirm(operation_event(2))


def test_operation_audit_reopen_does_not_repair_removed_trigger(tmp_path):
    store = AuditStore(tmp_path / "audit.sqlite3")
    store.append_operation_and_confirm(operation_event(1))
    with sqlite3.connect(store.path) as db:
        db.execute("DROP TRIGGER operation_audit_no_update")
    with pytest.raises(AuditUnavailable): AuditStore(store.path)

def test_audit_same_named_noop_trigger_fails_before_use(tmp_path):
    store=AuditStore(tmp_path/"audit.sqlite3")
    with sqlite3.connect(store.path) as db:
        db.execute("DROP TRIGGER audit_events_no_update")
        db.execute("CREATE TRIGGER audit_events_no_update BEFORE UPDATE ON audit_events BEGIN SELECT 1; END")
    assert store.verify_chain() is False


def test_connected_mode_is_default_off_and_registers_no_routes():
    app = Flask(__name__)
    assert install_connected_mode(app) is None
    assert all(
        not rule.rule.startswith("/operator-ui/connected")
        for rule in app.url_map.iter_rules()
    )


def test_environment_loader_is_fixed_and_default_off(monkeypatch):
    app = Flask(__name__)
    monkeypatch.setenv("OPERATOR_UI_AUDIT_DB_PATH", "/server/configured/audit.db")
    monkeypatch.setenv("CALLER_SUPPLIED_PATH", "/untrusted/request/path")
    monkeypatch.setenv("OPERATOR_UI_DEPLOYED_PROFILE", "repository-v1")
    load_connected_environment(app)
    assert app.config["OPERATOR_UI_CONNECTED_MODE"] is False
    assert app.config["OPERATOR_UI_AUDIT_DB_PATH"] == "/server/configured/audit.db"
    assert app.config["OPERATOR_UI_DEPLOYED_PROFILE"] == "repository-v1"
    assert "CALLER_SUPPLIED_PATH" not in app.config


@pytest.mark.parametrize(
    "change",
    [
        {"OPERATOR_UI_SECRET_KEY": ""},
        {"OPERATOR_UI_SECRET_KEY": "weak"},
        {"OPERATOR_UI_PASSWORD_HASH": "plaintext"},
        {"OPERATOR_UI_AUDIT_DB_PATH": None},
    ],
)
def test_connected_mode_fails_closed_for_missing_or_weak_config(tmp_path, change):
    app = Flask(__name__)
    values = {
        "OPERATOR_UI_CONNECTED_MODE": True,
        "OPERATOR_UI_SECRET_KEY": "stable-connected-secret-" + "x" * 32,
        "OPERATOR_UI_USERNAME": "viewer",
        "OPERATOR_UI_PASSWORD_HASH": generate_password_hash("correct horse"),
        "OPERATOR_UI_AUDIT_DB_PATH": str(tmp_path / "audit.sqlite3"),
        "OPERATOR_UI_DEPLOYED_COMMIT": "c" * 40,
        "OPERATOR_UI_DEPLOYED_TREE": "d" * 40,
        "OPERATOR_UI_DEPLOYED_VERSION": "v1",
    }
    app.config.update({**values, **change})
    with pytest.raises(ConnectedModeConfigurationError):
        install_connected_mode(app)


@pytest.mark.parametrize("other_key", ["DATABASE_PATH", "OPERATOR_UI_JOB_DB_PATH"])
def test_audit_store_must_be_separate_from_other_databases(tmp_path, other_key):
    app = Flask(__name__)
    audit = tmp_path / "same.sqlite3"
    app.config.update(
        OPERATOR_UI_CONNECTED_MODE=True,
        OPERATOR_UI_SECRET_KEY="stable-connected-secret-" + "x" * 32,
        OPERATOR_UI_USERNAME="viewer",
        OPERATOR_UI_PASSWORD_HASH=generate_password_hash("correct horse"),
        OPERATOR_UI_AUDIT_DB_PATH=str(audit),
        OPERATOR_UI_DEPLOYED_COMMIT="c" * 40,
        OPERATOR_UI_DEPLOYED_TREE="d" * 40,
        OPERATOR_UI_DEPLOYED_VERSION="v1",
    )
    app.config[other_key] = str(audit)
    with pytest.raises(ConnectedModeConfigurationError):
        install_connected_mode(app)


@pytest.mark.parametrize("other_key", ["DATABASE_PATH", "OPERATOR_UI_JOB_DB_PATH"])
def test_existing_hard_link_alias_is_rejected_before_other_database_changes(
    tmp_path, other_key
):
    canonical = tmp_path / "other.sqlite3"
    original = b"canonical-bytes-must-remain-exact"
    canonical.write_bytes(original)
    audit = tmp_path / "audit.sqlite3"
    os.link(canonical, audit)
    app = Flask(__name__)
    app.config.update(
        OPERATOR_UI_CONNECTED_MODE=True,
        OPERATOR_UI_SECRET_KEY="stable-connected-secret-" + "x" * 32,
        OPERATOR_UI_USERNAME="viewer",
        OPERATOR_UI_PASSWORD_HASH=generate_password_hash("correct horse"),
        OPERATOR_UI_AUDIT_DB_PATH=str(audit),
        OPERATOR_UI_DEPLOYED_COMMIT="c" * 40,
        OPERATOR_UI_DEPLOYED_TREE="d" * 40,
        OPERATOR_UI_DEPLOYED_VERSION="v1",
    )
    app.config[other_key] = str(canonical)
    with pytest.raises(ConnectedModeConfigurationError):
        install_connected_mode(app)
    assert canonical.read_bytes() == original
    assert audit.read_bytes() == original


@pytest.mark.parametrize("replacement_kind", ["ordinary", "symlink"])
@pytest.mark.parametrize("operation", ["append", "verify"])
def test_initialized_store_rejects_path_replacement_without_writing_it(
    tmp_path, replacement_kind, operation
):
    audit = tmp_path / "audit.sqlite3"
    canonical = tmp_path / "canonical.sqlite3"
    store = AuditStore(audit, (canonical,))
    replacement = tmp_path / "replacement"
    replacement_bytes = b"replacement-must-not-be-opened-as-sqlite"
    replacement.write_bytes(replacement_bytes)
    displaced = tmp_path / "original-audit.sqlite3"
    audit.rename(displaced)
    if replacement_kind == "ordinary":
        replacement.rename(audit)
        observed_path = audit
    else:
        audit.symlink_to(replacement)
        observed_path = replacement

    if operation == "append":
        with pytest.raises(AuditUnavailable):
            store.append_and_confirm(event(99))
    else:
        assert store.verify_chain() is False
    assert observed_path.read_bytes() == replacement_bytes


def test_distinct_store_paths_remain_usable(tmp_path):
    store = AuditStore(
        tmp_path / "audit.sqlite3",
        (tmp_path / "canonical.sqlite3", tmp_path / "jobs.sqlite3"),
    )
    store.append_and_confirm(event(1))
    assert store.verify_chain()


@pytest.mark.parametrize("other_name", ["canonical.sqlite3", "jobs.sqlite3"])
def test_initialized_store_rejects_later_hard_link_alias(tmp_path, other_name):
    audit = tmp_path / "audit.sqlite3"
    other = tmp_path / other_name
    store = AuditStore(audit, (other,))
    store.append_and_confirm(event(1))
    original = audit.read_bytes()
    with sqlite3.connect(audit) as connection:
        original_row_count = connection.execute(
            "SELECT COUNT(*) FROM audit_events"
        ).fetchone()[0]

    os.link(audit, other)

    with pytest.raises(AuditUnavailable):
        store.append_and_confirm(event(2))
    assert store.verify_chain() is False
    assert audit.read_bytes() == original
    assert other.read_bytes() == original
    with sqlite3.connect(audit) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
            == original_row_count
        )


@pytest.mark.parametrize("replacement_kind", ["ordinary", "symlink"])
def test_append_revalidates_path_after_commit_without_writing_replacement(
    tmp_path, monkeypatch, replacement_kind
):
    audit = tmp_path / "audit.sqlite3"
    store = AuditStore(audit)
    replacement = tmp_path / "replacement"
    replacement_bytes = b"replacement-must-remain-exact"
    replacement.write_bytes(replacement_bytes)
    displaced = tmp_path / "original-audit.sqlite3"
    original_connect = store._connect

    def connect_then_replace():
        connection = original_connect()
        audit.rename(displaced)
        if replacement_kind == "ordinary":
            replacement.rename(audit)
        else:
            audit.symlink_to(replacement)
        return connection

    monkeypatch.setattr(store, "_connect", connect_then_replace)
    with pytest.raises(AuditUnavailable):
        store.append_and_confirm(event(2))

    observed_replacement = audit if replacement_kind == "ordinary" else replacement
    assert observed_replacement.read_bytes() == replacement_bytes


@pytest.mark.parametrize("other_name", ["canonical.sqlite3", "jobs.sqlite3"])
def test_append_revalidates_later_hard_link_alias_after_commit(
    tmp_path, monkeypatch, other_name
):
    audit = tmp_path / "audit.sqlite3"
    other = tmp_path / other_name
    store = AuditStore(audit, (other,))
    original = audit.read_bytes()
    original_connect = store._connect

    def connect_then_alias():
        connection = original_connect()
        os.link(audit, other)
        return connection

    monkeypatch.setattr(store, "_connect", connect_then_alias)
    with pytest.raises(AuditUnavailable):
        store.append_and_confirm(event(2))

    assert audit.read_bytes() == original
    assert other.read_bytes() == original
    with sqlite3.connect(audit) as connection:
        assert connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0] == 0


def test_login_wrong_credentials_csrf_rotation_logout_and_cookie_flags(tmp_path):
    app = configured_app(tmp_path)
    client = app.test_client()
    unauthenticated = client.get("/operator-ui/connected/sentinel")
    assert unauthenticated.status_code == 401
    assert b"connected-mode-boundary" not in unauthenticated.data

    initial = client.get("/operator-ui/login")
    initial_cookie = initial.headers["Set-Cookie"]
    assert client.post(
        "/operator-ui/login", data={"username": "viewer", "password": "correct horse"}
    ).status_code == 400
    denied = login(client, "wrong")
    assert denied.status_code == 401
    assert b"correct horse" not in denied.data

    accepted = login(client)
    assert accepted.status_code == 200
    rotated_cookie = accepted.headers["Set-Cookie"]
    assert initial_cookie != rotated_cookie
    assert "Secure" in rotated_cookie
    assert "HttpOnly" in rotated_cookie
    assert "SameSite=Strict" in rotated_cookie

    allowed = client.get("/operator-ui/connected/sentinel")
    assert allowed.status_code == 200
    assert allowed.get_json()["sentinel"] == "connected-mode-boundary"
    csrf = accepted.get_json()["csrf_token"]
    logout = client.post("/operator-ui/logout", headers={"X-CSRF-Token": csrf})
    assert logout.status_code == 200
    assert "Expires=Thu, 01 Jan 1970" in logout.headers["Set-Cookie"]
    assert client.get("/operator-ui/connected/sentinel").status_code == 401


def test_relogin_and_logout_revoke_replayed_authenticated_cookies(tmp_path):
    app = configured_app(tmp_path)
    client = app.test_client()
    assert login(client).status_code == 200
    cookie_name = app.config.get("SESSION_COOKIE_NAME", "session")
    cookie_a = client.get_cookie(cookie_name).value

    relogin = login(client)
    assert relogin.status_code == 200
    cookie_b = client.get_cookie(cookie_name).value
    assert cookie_a != cookie_b

    replay_a = app.test_client()
    replay_a.set_cookie(cookie_name, cookie_a)
    assert replay_a.get("/operator-ui/connected/sentinel").status_code == 401
    assert client.get("/operator-ui/connected/sentinel").status_code == 200

    csrf = relogin.get_json()["csrf_token"]
    assert client.post("/operator-ui/logout", headers={"X-CSRF-Token": csrf}).status_code == 200
    replay_b = app.test_client()
    replay_b.set_cookie(cookie_name, cookie_b)
    assert replay_b.get("/operator-ui/connected/sentinel").status_code == 401


def test_overlapping_authenticated_requests_accept_stale_rotated_cookie(tmp_path):
    request_clock = threading.local()

    def clock():
        return NOW + timedelta(seconds=getattr(request_clock, "advance", 0))

    app = configured_app(tmp_path, clock=clock)
    decorator = app.extensions["operator_ui_operational_get"]
    first_authenticated = threading.Event()
    release_first = threading.Event()

    @app.get("/operator-ui/api/v1/races/overlap")
    @decorator(policy="LEVEL_1_API_V1_RACE_DETAIL")
    def overlap():
        if request_clock.advance == 1:
            first_authenticated.set()
            assert release_first.wait(timeout=5)
        return PreparedDisclosure(
            body=b'{"classification":"AVAILABLE/FRESH"}',
            classification=EvidenceStatus.AVAILABLE_FRESH,
            evidence_source_identifiers=("fixture.overlap",),
            content_hashes=(HASH,),
        )

    login_client = app.test_client()
    assert login(login_client).status_code == 200
    cookie_name = app.config.get("SESSION_COOKIE_NAME", "session")
    original_cookie = login_client.get_cookie(cookie_name).value

    def authenticated_get(advance):
        request_clock.advance = advance
        client = app.test_client()
        client.set_cookie(cookie_name, original_cookie)
        response = client.get("/operator-ui/api/v1/races/overlap")
        return response, client.get_cookie(cookie_name).value

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(authenticated_get, 1)
        assert first_authenticated.wait(timeout=5)
        second_response, second_cookie = authenticated_get(2)
        release_first.set()
        first_response, first_cookie = first.result(timeout=5)

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_cookie != original_cookie
    assert second_cookie != original_cookie

    request_clock.advance = 3
    resulting_client = app.test_client()
    resulting_client.set_cookie(cookie_name, first_cookie)
    assert resulting_client.get("/operator-ui/connected/sentinel").status_code == 200
    assert len(app.extensions["operator_ui_active_sessions"]) == 1


@pytest.mark.parametrize(("advance", "status"), [(61, 401), (301, 401), (60, 200)])
def test_inactivity_and_absolute_expiry_are_finite(tmp_path, advance, status):
    current = [NOW]
    app = configured_app(tmp_path, clock=lambda: current[0])
    client = app.test_client()
    assert login(client).status_code == 200
    current[0] = NOW + timedelta(seconds=advance)
    assert client.get("/operator-ui/connected/sentinel").status_code == status
    if status == 401:
        assert len(app.extensions["operator_ui_active_sessions"]) == 0


def test_active_session_registry_is_bounded_and_evicts_oldest(tmp_path):
    app = configured_app(tmp_path)
    registry = app.extensions["operator_ui_active_sessions"]
    for number in range(registry.maximum + 1):
        registry.register(
            f"session-{number}",
            "viewer",
            1,
            number,
            inactivity=10_000,
            absolute=10_000,
        )
    assert len(registry) == registry.maximum
    assert "session-0" not in registry._sessions


def test_insufficient_level_is_audited_and_provider_is_not_called(tmp_path):
    app = configured_app(tmp_path)
    client = app.test_client()
    assert login(client).status_code == 200
    with client.session_transaction() as connected_session:
        session_identifier = connected_session["operator_session_id"]
        issued_at = connected_session["operator_issued_at"]
        connected_session["operator_level"] = 0
    app.extensions["operator_ui_active_sessions"].register(
        session_identifier,
        "viewer",
        0,
        issued_at,
        inactivity=60,
        absolute=300,
    )
    result = client.get("/operator-ui/connected/sentinel")
    assert result.status_code == 403
    with sqlite3.connect(tmp_path / "ui-audit.sqlite3") as connection:
        row = connection.execute(
            "SELECT authorization_decision, response_classification FROM audit_events"
        ).fetchone()
    assert row == ("denied", "NON_OPERATIONAL/AUTHORIZATION_DENIED")
    assert b"connected-mode-boundary" not in result.data


@pytest.mark.parametrize("classification", list(EvidenceStatus))
def test_prepared_result_audits_exact_dynamic_classification_and_metadata(
    tmp_path, classification
):
    app = configured_app(tmp_path)
    decorator = app.extensions["operator_ui_operational_get"]
    source = f"source.{classification.name.lower()}"
    content_hash = hashlib.sha256(classification.value.encode()).hexdigest()
    reference_hash = hashlib.sha256(source.encode()).hexdigest()
    protected_payload_marker = "protected-payload-marker-must-not-enter-audit"

    @app.get("/operator-ui/api/v1/races/probe")
    @decorator(policy="LEVEL_1_API_V1_RACE_DETAIL")
    def probe():
        return PreparedDisclosure(
            body=json.dumps(
                {
                    "classification": classification.value,
                    "protected_payload": protected_payload_marker,
                }
            ).encode(),
            classification=classification,
            evidence_source_identifiers=(source,),
            content_hashes=(content_hash,),
            reference_hashes=(reference_hash,),
        )

    client = app.test_client()
    assert login(client).status_code == 200
    response = client.get("/operator-ui/api/v1/races/probe", headers={"X-Request-ID": "req-1"})
    assert response.status_code == 200
    assert response.get_json()["classification"] == classification.value
    with sqlite3.connect(tmp_path / "ui-audit.sqlite3") as connection:
        row = connection.execute(
            """
            SELECT response_classification, evidence_source_identifiers,
                   content_hashes, reference_hashes
            FROM audit_events
            """
        ).fetchone()
    assert row == (
        classification.value,
        json.dumps([source], separators=(",", ":")),
        json.dumps([content_hash], separators=(",", ":")),
        json.dumps([reference_hash], separators=(",", ":")),
    )
    raw = (tmp_path / "ui-audit.sqlite3").read_bytes()
    assert protected_payload_marker.encode() not in raw
    assert b"correct horse" not in raw


def test_audit_failure_withholds_already_buffered_provider_bytes(tmp_path, monkeypatch):
    app = configured_app(tmp_path)
    called = []
    decorator = app.extensions["operator_ui_operational_get"]

    @app.get("/operator-ui/api/v1/races/protected")
    @decorator(policy="LEVEL_1_API_V1_RACE_DETAIL")
    def protected():
        called.append(True)
        return PreparedDisclosure(
            body=b'{"classification":"AVAILABLE/FRESH","secret_evidence":"must-not-disclose"}',
            classification=EvidenceStatus.AVAILABLE_FRESH,
            evidence_source_identifiers=("safe.identity",),
            content_hashes=(HASH,),
        )

    def unavailable(_event):
        raise AuditUnavailable("forced")

    monkeypatch.setattr(
        app.extensions["operator_ui_audit"], "append_and_confirm", unavailable
    )
    client = app.test_client()
    assert login(client).status_code == 200
    response = client.get("/operator-ui/api/v1/races/protected")
    assert response.status_code == 503
    assert response.get_json()["classification"] == "NON_OPERATIONAL/AUDIT_UNAVAILABLE"
    assert b"must-not-disclose" not in response.data
    assert called == [True]


def test_same_inode_earlier_corruption_blocks_next_operational_disclosure(tmp_path):
    app = configured_app(tmp_path)
    store = app.extensions["operator_ui_audit"]
    store.append_and_confirm(event(1))
    original_identity = (store.path.stat().st_dev, store.path.stat().st_ino)
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TRIGGER audit_events_no_update")
        connection.execute(
            "UPDATE audit_events SET route = ? WHERE sequence = 1",
            ("/corrupted-after-start",),
        )
        connection.execute(
            """
            CREATE TRIGGER audit_events_no_update
            BEFORE UPDATE ON audit_events BEGIN
                SELECT RAISE(ABORT, 'audit events are immutable');
            END
            """
        )
    assert (store.path.stat().st_dev, store.path.stat().st_ino) == original_identity
    assert store.verify_chain() is False

    protected_marker = b"same-inode-corruption-must-not-disclose"
    decorator = app.extensions["operator_ui_operational_get"]

    @app.get("/operator-ui/api/v1/races/corrupt-chain")
    @decorator(policy="LEVEL_1_API_V1_RACE_DETAIL")
    def protected():
        return PreparedDisclosure(
            body=(
                b'{"classification":"AVAILABLE/FRESH","protected":"'
                + protected_marker
                + b'"}'
            ),
            classification=EvidenceStatus.AVAILABLE_FRESH,
            evidence_source_identifiers=("safe.identity",),
            content_hashes=(HASH,),
        )

    client = app.test_client()
    assert login(client).status_code == 200
    response = client.get("/operator-ui/api/v1/races/corrupt-chain")

    assert response.status_code == 503
    assert response.get_json() == {
        "classification": "NON_OPERATIONAL/AUDIT_UNAVAILABLE",
        "error": "operational disclosure unavailable",
    }
    assert protected_marker not in response.data
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM audit_events"
        ).fetchone()[0] == 1


@pytest.mark.parametrize(
    ("mutation", "other_name"),
    [
        ("ordinary", None),
        ("symlink", None),
        ("hard_link", "canonical.sqlite3"),
        ("hard_link", "jobs.sqlite3"),
    ],
)
def test_post_connect_identity_change_withholds_operational_response(
    tmp_path, monkeypatch, mutation, other_name
):
    app = configured_app(tmp_path)
    store = app.extensions["operator_ui_audit"]
    audit = store.path
    replacement = tmp_path / "replacement"
    replacement_bytes = b"replacement-must-remain-exact"
    replacement.write_bytes(replacement_bytes)
    displaced = tmp_path / "original-audit.sqlite3"
    original_audit_bytes = audit.read_bytes()
    original_connect = store._connect
    mutated = False

    def connect_then_mutate():
        nonlocal mutated
        connection = original_connect()
        if not mutated:
            mutated = True
            if mutation == "hard_link":
                os.link(audit, tmp_path / other_name)
            else:
                audit.rename(displaced)
                if mutation == "ordinary":
                    replacement.rename(audit)
                else:
                    audit.symlink_to(replacement)
        return connection

    monkeypatch.setattr(store, "_connect", connect_then_mutate)
    protected_marker = b"prepared-protected-marker-must-not-disclose"
    decorator = app.extensions["operator_ui_operational_get"]

    @app.get("/operator-ui/api/v1/races/post-connect-race")
    @decorator(policy="LEVEL_1_API_V1_RACE_DETAIL")
    def protected():
        return PreparedDisclosure(
            body=(
                b'{"classification":"AVAILABLE/FRESH","protected":"'
                + protected_marker
                + b'"}'
            ),
            classification=EvidenceStatus.AVAILABLE_FRESH,
            evidence_source_identifiers=("safe.identity",),
            content_hashes=(HASH,),
        )

    client = app.test_client()
    assert login(client).status_code == 200
    response = client.get("/operator-ui/api/v1/races/post-connect-race")

    assert response.status_code == 503
    assert response.get_json()["classification"] == "NON_OPERATIONAL/AUDIT_UNAVAILABLE"
    assert protected_marker not in response.data
    if mutation != "hard_link":
        observed_replacement = audit if mutation == "ordinary" else replacement
        assert observed_replacement.read_bytes() == replacement_bytes
    else:
        assert audit.read_bytes() == original_audit_bytes
        assert (tmp_path / other_name).read_bytes() == original_audit_bytes


@pytest.mark.parametrize(
    "provider",
    [
        lambda: (_ for _ in ()).throw(RuntimeError("provider failed")),
        lambda: Response(iter((b"deferred-protected-byte",))),
        lambda: PreparedDisclosure(
            body=b'{"classification":"AVAILABLE/FRESH"}',
            classification=EvidenceStatus.AVAILABLE_FRESH,
            evidence_source_identifiers=("source",),
            content_hashes=("invalid",),
        ),
        lambda: PreparedDisclosure(
            body=b'{"classification":"STALE","secret":"withheld"}',
            classification=EvidenceStatus.AVAILABLE_FRESH,
            evidence_source_identifiers=("source",),
            content_hashes=(HASH,),
        ),
    ],
    ids=("exception", "streamed-response", "invalid-metadata", "mismatch"),
)
def test_provider_failures_are_evidence_free_and_never_audit_available(
    tmp_path, provider
):
    app = configured_app(tmp_path)
    decorator = app.extensions["operator_ui_operational_get"]
    app.add_url_rule(
        "/operator-ui/api/v1/races/failing",
        view_func=decorator(policy="LEVEL_1_API_V1_RACE_DETAIL")(provider),
    )
    client = app.test_client()
    assert login(client).status_code == 200
    response = client.get("/operator-ui/api/v1/races/failing")
    assert response.status_code == 503
    assert response.get_json()["classification"] == "NON_OPERATIONAL/PROVIDER_ERROR"
    assert b"withheld" not in response.data
    with sqlite3.connect(tmp_path / "ui-audit.sqlite3") as connection:
        row = connection.execute(
            """
            SELECT response_classification, evidence_source_identifiers,
                   content_hashes, reference_hashes
            FROM audit_events
            """
        ).fetchone()
    assert row == ("NON_OPERATIONAL/PROVIDER_ERROR", "[]", "[]", "[]")


@pytest.mark.parametrize(
    ("classification", "sources", "content_hashes"),
    [
        (EvidenceStatus.AVAILABLE_FRESH, (), (HASH,)),
        (EvidenceStatus.UNAVAILABLE_DATA_MISSING, (), ()),
        (EvidenceStatus.AVAILABLE_FRESH, ("source",), ()),
        (EvidenceStatus.STALE, ("source",), ()),
        (EvidenceStatus.INVALID_INTEGRITY_FAILED, ("source",), ()),
        (EvidenceStatus.DIVERGENT, ("source",), ()),
    ],
)
def test_empty_required_evidence_metadata_fails_without_disclosure_or_false_audit(
    tmp_path, classification, sources, content_hashes
):
    app = configured_app(tmp_path)
    decorator = app.extensions["operator_ui_operational_get"]
    protected_marker = "empty-metadata-protected-marker"

    @app.get("/operator-ui/api/v1/races/empty-metadata")
    @decorator(policy="LEVEL_1_API_V1_RACE_DETAIL")
    def empty_metadata():
        return PreparedDisclosure(
            body=json.dumps(
                {
                    "classification": classification.value,
                    "protected": protected_marker,
                }
            ).encode(),
            classification=classification,
            evidence_source_identifiers=sources,
            content_hashes=content_hashes,
        )

    client = app.test_client()
    assert login(client).status_code == 200
    response = client.get("/operator-ui/api/v1/races/empty-metadata")
    assert response.status_code == 503
    assert response.get_json()["classification"] == "NON_OPERATIONAL/PROVIDER_ERROR"
    assert protected_marker.encode() not in response.data
    with sqlite3.connect(tmp_path / "ui-audit.sqlite3") as connection:
        classifications = [
            row[0]
            for row in connection.execute(
                "SELECT response_classification FROM audit_events"
            )
        ]
    assert classifications == ["NON_OPERATIONAL/PROVIDER_ERROR"]
    assert EvidenceStatus.AVAILABLE_FRESH.value not in classifications
    assert protected_marker.encode() not in (tmp_path / "ui-audit.sqlite3").read_bytes()


def test_hash_chain_concurrency_and_database_immutability(tmp_path):
    store = AuditStore(tmp_path / "audit.sqlite3")
    with ThreadPoolExecutor(max_workers=8) as executor:
        hashes = list(
            executor.map(
                lambda number: store.append_and_confirm(event(number)), range(24)
            )
        )
    assert len(set(hashes)) == 24
    assert store.verify_chain()
    with sqlite3.connect(store.path) as connection:
        sequences = [
            row[0]
            for row in connection.execute(
                "SELECT sequence FROM audit_events ORDER BY sequence"
            )
        ]
        assert sequences == list(range(1, 25))
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("UPDATE audit_events SET route = '/changed'")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM audit_events")
