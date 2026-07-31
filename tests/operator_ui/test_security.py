from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
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
        event_id=f"event-{number}",
        event_time_utc="2026-07-31T01:00:00.000000Z",
        actor_identity="viewer",
        actor_level=1,
        session_identifier="session-safe",
        request_identifier=f"request-{number}",
        route="/operator-ui/connected/sentinel",
        http_method="GET",
        authorization_decision="allowed",
        authorization_policy="LEVEL_1_TEST",
        evidence_source_identifiers=("fixture.identity",),
        content_hashes=(HASH,),
        reference_hashes=(HASH,),
        deployed_commit="c" * 40,
        deployed_tree="d" * 40,
        deployed_version="test-v1",
        response_classification="OPERATIONAL/AVAILABLE",
    )


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
    load_connected_environment(app)
    assert app.config["OPERATOR_UI_CONNECTED_MODE"] is False
    assert app.config["OPERATOR_UI_AUDIT_DB_PATH"] == "/server/configured/audit.db"
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

    @app.get("/operator-ui/connected/overlap")
    @decorator(policy="LEVEL_1_OVERLAP")
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
        response = client.get("/operator-ui/connected/overlap")
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

    @app.get("/operator-ui/connected/probe")
    @decorator(policy="LEVEL_1_PROBE")
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
    response = client.get("/operator-ui/connected/probe", headers={"X-Request-ID": "req-1"})
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

    @app.get("/operator-ui/connected/protected")
    @decorator(policy="LEVEL_1_PROTECTED")
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
    response = client.get("/operator-ui/connected/protected")
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

    @app.get("/operator-ui/connected/corrupt-chain")
    @decorator(policy="LEVEL_1_CORRUPT_CHAIN")
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
    response = client.get("/operator-ui/connected/corrupt-chain")

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

    @app.get("/operator-ui/connected/post-connect-race")
    @decorator(policy="LEVEL_1_POST_CONNECT_RACE")
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
    response = client.get("/operator-ui/connected/post-connect-race")

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
        "/operator-ui/connected/failing",
        view_func=decorator(policy="LEVEL_1_FAILING")(provider),
    )
    client = app.test_client()
    assert login(client).status_code == 200
    response = client.get("/operator-ui/connected/failing")
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

    @app.get("/operator-ui/connected/empty-metadata")
    @decorator(policy="LEVEL_1_EMPTY_METADATA")
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
    response = client.get("/operator-ui/connected/empty-metadata")
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
