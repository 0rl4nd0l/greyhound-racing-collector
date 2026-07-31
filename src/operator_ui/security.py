"""Fail-closed connected-mode security and access auditing for Operator UI."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import stat
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping

from flask import Flask, Response, jsonify, request, session
from werkzeug.security import check_password_hash

from .foundation import EvidenceStatus


AUDIT_SCHEMA = "operator_ui_access_audit_v1"
ZERO_HASH = "0" * 64
NON_OPERATIONAL_ERROR = {
    "classification": "NON_OPERATIONAL/AUDIT_UNAVAILABLE",
    "error": "operational disclosure unavailable",
}
PROVIDER_ERROR = {
    "classification": "NON_OPERATIONAL/PROVIDER_ERROR",
    "error": "operational provider unavailable",
}
_REQUIRED_HASH = frozenset("0123456789abcdef")
_EVIDENCE_CLASSIFICATIONS = frozenset(status.value for status in EvidenceStatus)
_CONTENT_HASH_REQUIRED = frozenset(
    {
        EvidenceStatus.AVAILABLE_FRESH,
        EvidenceStatus.STALE,
        EvidenceStatus.INVALID_INTEGRITY_FAILED,
        EvidenceStatus.DIVERGENT,
    }
)
_ACTIVE_SESSION_LIMIT = 256
_ACTIVE_SESSION_LIMIT_HARD_MAX = 4096


class ConnectedModeConfigurationError(RuntimeError):
    """Connected mode was requested without a stable, safe configuration."""


class AuditUnavailable(RuntimeError):
    """An audit event could not be durably appended and confirmed."""


@dataclass(frozen=True)
class PreparedDisclosure:
    """A finite response whose exact evidence metadata is ready for audit."""

    body: bytes
    classification: EvidenceStatus
    evidence_source_identifiers: tuple[str, ...]
    content_hashes: tuple[str, ...]
    reference_hashes: tuple[str, ...] = ()
    status_code: int = 200
    content_type: str = "application/json"


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and set(value) <= _REQUIRED_HASH


def _valid_identifier(value: str) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value.encode("utf-8")) <= 512
        and all(character >= " " and character != "\x7f" for character in value)
    )


def _validate_prepared_disclosure(value: Any) -> None:
    if not isinstance(value, PreparedDisclosure):
        raise ValueError("provider must return a prepared disclosure")
    if not isinstance(value.body, bytes):
        raise ValueError("prepared response body must be buffered bytes")
    if (
        not isinstance(value.classification, EvidenceStatus)
        or value.classification.value not in _EVIDENCE_CLASSIFICATIONS
    ):
        raise ValueError("invalid evidence classification")
    if not isinstance(value.status_code, int) or not 200 <= value.status_code < 300:
        raise ValueError("invalid prepared response status")
    if not _valid_identifier(value.content_type):
        raise ValueError("invalid prepared response content type")
    identities = value.evidence_source_identifiers
    if (
        not isinstance(identities, tuple)
        or not identities
        or any(not _valid_identifier(identity) for identity in identities)
    ):
        raise ValueError("invalid evidence source identifiers")
    for hashes in (value.content_hashes, value.reference_hashes):
        if not isinstance(hashes, tuple) or any(
            not isinstance(item, str) or not _valid_sha256(item) for item in hashes
        ):
            raise ValueError("invalid evidence hashes")
    if value.classification in _CONTENT_HASH_REQUIRED and not value.content_hashes:
        raise ValueError("content hash required for evidence classification")
    try:
        decoded = json.loads(value.body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("prepared response must be finite JSON") from exc
    if (
        not isinstance(decoded, dict)
        or decoded.get("classification") != value.classification.value
    ):
        raise ValueError("response and audit classifications differ")


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    event_time_utc: str
    actor_identity: str
    actor_level: int
    session_identifier: str
    request_identifier: str
    route: str
    http_method: str
    authorization_decision: str
    authorization_policy: str
    evidence_source_identifiers: tuple[str, ...]
    content_hashes: tuple[str, ...]
    reference_hashes: tuple[str, ...]
    deployed_commit: str
    deployed_tree: str
    deployed_version: str
    response_classification: str

    def fields(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "schema": AUDIT_SCHEMA,
            "event_time_utc": self.event_time_utc,
            "actor_identity": self.actor_identity,
            "actor_level": self.actor_level,
            "session_identifier": self.session_identifier,
            "request_identifier": self.request_identifier,
            "route": self.route,
            "http_method": self.http_method,
            "authorization_decision": self.authorization_decision,
            "authorization_policy": self.authorization_policy,
            "evidence_source_identifiers": list(self.evidence_source_identifiers),
            "content_hashes": list(self.content_hashes),
            "reference_hashes": list(self.reference_hashes),
            "deployed_commit": self.deployed_commit,
            "deployed_tree": self.deployed_tree,
            "deployed_version": self.deployed_version,
            "response_classification": self.response_classification,
        }


@dataclass(frozen=True)
class _ActiveSession:
    actor: str
    level: int
    issued_at: int
    last_active: int


class _ActiveSessionRegistry:
    """Bounded process-local authority for connected authenticated sessions."""

    def __init__(self, maximum: int = _ACTIVE_SESSION_LIMIT):
        if (
            not isinstance(maximum, int)
            or not 1 <= maximum <= _ACTIVE_SESSION_LIMIT_HARD_MAX
        ):
            raise ConnectedModeConfigurationError("invalid active session registry limit")
        self.maximum = maximum
        self._sessions: dict[str, _ActiveSession] = {}
        self._lock = threading.Lock()

    def revoke(self, session_identifier: Any) -> None:
        if not isinstance(session_identifier, str):
            return
        with self._lock:
            self._sessions.pop(session_identifier, None)

    def register(
        self,
        session_identifier: str,
        actor: str,
        level: int,
        issued_at: int,
        *,
        inactivity: int,
        absolute: int,
    ) -> None:
        with self._lock:
            expired = [
                identifier
                for identifier, active_session in self._sessions.items()
                if (
                    issued_at < active_session.issued_at
                    or issued_at - active_session.last_active > inactivity
                    or issued_at - active_session.issued_at > absolute
                )
            ]
            for identifier in expired:
                self._sessions.pop(identifier, None)
            if len(self._sessions) >= self.maximum:
                oldest = min(
                    self._sessions,
                    key=lambda identifier: (
                        self._sessions[identifier].last_active,
                        self._sessions[identifier].issued_at,
                        identifier,
                    ),
                )
                self._sessions.pop(oldest)
            self._sessions[session_identifier] = _ActiveSession(
                actor=actor,
                level=level,
                issued_at=issued_at,
                last_active=issued_at,
            )

    def authenticate(
        self,
        session_identifier: Any,
        actor: Any,
        level: Any,
        issued_at: Any,
        last_active: Any,
        *,
        current: int,
        inactivity: int,
        absolute: int,
    ) -> tuple[str, int] | None:
        if (
            not isinstance(session_identifier, str)
            or not isinstance(actor, str)
            or not all(isinstance(value, int) for value in (level, issued_at, last_active))
        ):
            return None
        with self._lock:
            active_session = self._sessions.get(session_identifier)
            if active_session is None or (
                (actor, level, issued_at)
                != (
                    active_session.actor,
                    active_session.level,
                    active_session.issued_at,
                )
                or not issued_at <= last_active <= active_session.last_active
            ):
                self._sessions.pop(session_identifier, None)
                return None
            if (
                current < active_session.issued_at
                or current - active_session.last_active > inactivity
                or current - active_session.issued_at > absolute
            ):
                self._sessions.pop(session_identifier, None)
                return None
            self._sessions[session_identifier] = _ActiveSession(
                actor=active_session.actor,
                level=active_session.level,
                issued_at=active_session.issued_at,
                last_active=max(current, active_session.last_active),
            )
            return active_session.actor, active_session.level

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)


class AuditStore:
    """A separate SQLite insert-only, hash-chained operations store."""

    def __init__(self, path: Path, separate_from: tuple[Path, ...] = ()):
        self.path = Path(path).absolute()
        self._separate_from = tuple(Path(value).absolute() for value in separate_from)
        self._file_identity: tuple[int, int] | None = None
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        if self._file_identity is not None:
            self._validate_pinned_path()
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        if self._file_identity is not None:
            try:
                self._validate_pinned_path()
            except AuditUnavailable:
                connection.close()
                raise
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _existing_identity(self, path: Path) -> tuple[int, int] | None:
        try:
            details = path.stat()
        except FileNotFoundError:
            return None
        return details.st_dev, details.st_ino

    def _validate_separation(self, audit_identity: tuple[int, int] | None) -> None:
        for other_path in self._separate_from:
            other_identity = self._existing_identity(other_path)
            if self.path.resolve(strict=False) == other_path.resolve(strict=False) or (
                audit_identity is not None and audit_identity == other_identity
            ):
                raise AuditUnavailable(
                    "audit store must be separate from canonical and job stores"
                )

    def _validate_pinned_path(self) -> None:
        try:
            details = self.path.lstat()
        except OSError as exc:
            raise AuditUnavailable("audit store pathname is unavailable") from exc
        identity = (details.st_dev, details.st_ino)
        if (
            not stat.S_ISREG(details.st_mode)
            or identity != self._file_identity
            or self.path.is_symlink()
        ):
            raise AuditUnavailable("audit store pathname identity changed")
        self._validate_separation(identity)

    def _initialize(self) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            existing = self.path.lstat()
        except FileNotFoundError:
            existing = None
        if existing is not None and (
            not stat.S_ISREG(existing.st_mode) or self.path.is_symlink()
        ):
            raise AuditUnavailable("audit store must be a regular file")
        self._validate_separation(
            None if existing is None else (existing.st_dev, existing.st_ino)
        )
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    schema TEXT NOT NULL,
                    event_time_utc TEXT NOT NULL,
                    actor_identity TEXT NOT NULL,
                    actor_level INTEGER NOT NULL,
                    session_identifier TEXT NOT NULL,
                    request_identifier TEXT NOT NULL,
                    route TEXT NOT NULL,
                    http_method TEXT NOT NULL,
                    authorization_decision TEXT NOT NULL,
                    authorization_policy TEXT NOT NULL,
                    evidence_source_identifiers TEXT NOT NULL,
                    content_hashes TEXT NOT NULL,
                    reference_hashes TEXT NOT NULL,
                    deployed_commit TEXT NOT NULL,
                    deployed_tree TEXT NOT NULL,
                    deployed_version TEXT NOT NULL,
                    response_classification TEXT NOT NULL,
                    previous_event_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE
                );
                CREATE TRIGGER IF NOT EXISTS audit_events_no_update
                BEFORE UPDATE ON audit_events BEGIN
                    SELECT RAISE(ABORT, 'audit events are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS audit_events_no_delete
                BEFORE DELETE ON audit_events BEGIN
                    SELECT RAISE(ABORT, 'audit events are immutable');
                END;
                """
            )
        finally:
            connection.close()
        os.chmod(self.path, 0o600)
        details = self.path.lstat()
        if not stat.S_ISREG(details.st_mode) or self.path.is_symlink():
            raise AuditUnavailable("audit store must be a regular file")
        self._file_identity = (details.st_dev, details.st_ino)
        self._validate_pinned_path()

    def append_and_confirm(self, event: AuditEvent) -> str:
        fields = event.fields()
        for name in ("content_hashes", "reference_hashes"):
            if any(not _valid_sha256(value) for value in fields[name]):
                raise AuditUnavailable(f"invalid {name}")
        connection = self._connect()
        try:
            self._validate_pinned_path()
            connection.execute("BEGIN IMMEDIATE")
            previous_hash = self._verified_tail(connection)
            event_hash = hashlib.sha256(
                _canonical({**fields, "previous_event_hash": previous_hash})
            ).hexdigest()
            values = {
                **fields,
                "evidence_source_identifiers": json.dumps(
                    fields["evidence_source_identifiers"], separators=(",", ":")
                ),
                "content_hashes": json.dumps(
                    fields["content_hashes"], separators=(",", ":")
                ),
                "reference_hashes": json.dumps(
                    fields["reference_hashes"], separators=(",", ":")
                ),
                "previous_event_hash": previous_hash,
                "event_hash": event_hash,
            }
            connection.execute(
                """
                INSERT INTO audit_events (
                    event_id, schema, event_time_utc, actor_identity, actor_level,
                    session_identifier, request_identifier, route, http_method,
                    authorization_decision, authorization_policy,
                    evidence_source_identifiers, content_hashes, reference_hashes,
                    deployed_commit, deployed_tree, deployed_version,
                    response_classification, previous_event_hash, event_hash
                ) VALUES (
                    :event_id, :schema, :event_time_utc, :actor_identity, :actor_level,
                    :session_identifier, :request_identifier, :route, :http_method,
                    :authorization_decision, :authorization_policy,
                    :evidence_source_identifiers, :content_hashes, :reference_hashes,
                    :deployed_commit, :deployed_tree, :deployed_version,
                    :response_classification, :previous_event_hash, :event_hash
                )
                """,
                values,
            )
            confirmed = connection.execute(
                "SELECT event_hash FROM audit_events WHERE event_id = ?",
                (event.event_id,),
            ).fetchone()
            if confirmed is None or not hmac.compare_digest(
                confirmed["event_hash"], event_hash
            ):
                raise AuditUnavailable("audit confirmation failed")
            connection.commit()
            self._validate_pinned_path()
            return event_hash
        except AuditUnavailable:
            connection.rollback()
            raise
        except (sqlite3.Error, ValueError, TypeError) as exc:
            connection.rollback()
            raise AuditUnavailable("audit append failed") from exc
        finally:
            connection.close()

    def _verified_tail(self, connection: sqlite3.Connection) -> str:
        previous_hash = ZERO_HASH
        rows = connection.execute(
            "SELECT * FROM audit_events ORDER BY sequence"
        ).fetchall()
        for expected_sequence, row in enumerate(rows, start=1):
            if row["sequence"] != expected_sequence:
                raise AuditUnavailable("audit chain sequence invalid")
            fields = {
                key: row[key]
                for key in AuditEvent.__dataclass_fields__
                if key
                not in {
                    "evidence_source_identifiers",
                    "content_hashes",
                    "reference_hashes",
                }
            }
            fields["schema"] = row["schema"]
            for name in (
                "evidence_source_identifiers",
                "content_hashes",
                "reference_hashes",
            ):
                fields[name] = json.loads(row[name])
            calculated = hashlib.sha256(
                _canonical({**fields, "previous_event_hash": previous_hash})
            ).hexdigest()
            if row["previous_event_hash"] != previous_hash or not hmac.compare_digest(
                row["event_hash"], calculated
            ):
                raise AuditUnavailable("audit chain integrity invalid")
            previous_hash = row["event_hash"]
        return previous_hash

    def verify_chain(self) -> bool:
        try:
            connection = self._connect()
        except (AuditUnavailable, OSError, sqlite3.Error):
            return False
        try:
            self._verified_tail(connection)
            return True
        except (
            AuditUnavailable,
            sqlite3.Error,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ):
            return False
        finally:
            connection.close()


def _enabled(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_connected_environment(app: Flask) -> None:
    """Load only the fixed server-owned connected-mode environment keys."""
    keys = (
        "OPERATOR_UI_SECRET_KEY",
        "OPERATOR_UI_USERNAME",
        "OPERATOR_UI_PASSWORD_HASH",
        "OPERATOR_UI_LEVEL",
        "OPERATOR_UI_AUDIT_DB_PATH",
        "OPERATOR_UI_JOB_DB_PATH",
        "OPERATOR_UI_DEPLOYED_COMMIT",
        "OPERATOR_UI_DEPLOYED_TREE",
        "OPERATOR_UI_DEPLOYED_VERSION",
        "OPERATOR_UI_INACTIVITY_SECONDS",
        "OPERATOR_UI_ABSOLUTE_SECONDS",
    )
    app.config["OPERATOR_UI_CONNECTED_MODE"] = _enabled(
        os.environ.get("OPERATOR_UI_CONNECTED_MODE", "0")
    )
    for key in keys:
        if key in os.environ:
            app.config[key] = os.environ[key]


def _configuration(
    app: Flask,
) -> tuple[Path, tuple[Path, ...], Callable[[], datetime]]:
    required = (
        "OPERATOR_UI_SECRET_KEY",
        "OPERATOR_UI_USERNAME",
        "OPERATOR_UI_PASSWORD_HASH",
        "OPERATOR_UI_AUDIT_DB_PATH",
        "OPERATOR_UI_DEPLOYED_COMMIT",
        "OPERATOR_UI_DEPLOYED_TREE",
        "OPERATOR_UI_DEPLOYED_VERSION",
    )
    missing = [name for name in required if not app.config.get(name)]
    secret = str(app.config.get("OPERATOR_UI_SECRET_KEY", ""))
    weak = len(secret.encode("utf-8")) < 32 or secret in {
        "greyhound_racing_secret_key_2025",
        "dev",
        "development",
        "secret",
    }
    if missing or weak:
        raise ConnectedModeConfigurationError(
            "connected mode requires complete stable server configuration"
        )
    password_hash = str(app.config["OPERATOR_UI_PASSWORD_HASH"])
    if not password_hash.startswith(("scrypt:", "pbkdf2:")):
        raise ConnectedModeConfigurationError("password must use a Werkzeug hash")
    if int(app.config.get("OPERATOR_UI_LEVEL", 1)) < 1:
        raise ConnectedModeConfigurationError("connected user requires Level 1")
    for name in ("OPERATOR_UI_DEPLOYED_COMMIT", "OPERATOR_UI_DEPLOYED_TREE"):
        identity = str(app.config[name])
        if len(identity) != 40 or not set(identity) <= _REQUIRED_HASH:
            raise ConnectedModeConfigurationError(
                "connected mode requires exact deployed commit and tree identities"
            )
    audit_path = Path(app.config["OPERATOR_UI_AUDIT_DB_PATH"]).absolute()
    other_paths = tuple(
        Path(app.config[name]).absolute()
        for name in ("DATABASE_PATH", "OPERATOR_UI_JOB_DB_PATH")
        if app.config.get(name)
    )
    if any(
        audit_path.resolve(strict=False) == path.resolve(strict=False)
        for path in other_paths
    ):
        raise ConnectedModeConfigurationError(
            "audit store must be separate from canonical and job stores"
        )
    clock = app.config.get("OPERATOR_UI_CLOCK", lambda: datetime.now(timezone.utc))
    if not callable(clock):
        raise ConnectedModeConfigurationError("connected clock must be callable")
    return audit_path, other_paths, clock


def install_connected_mode(app: Flask) -> AuditStore | None:
    """Install the reusable boundary; do nothing unless explicitly enabled."""
    app.config.setdefault("OPERATOR_UI_CONNECTED_MODE", False)
    if not _enabled(app.config["OPERATOR_UI_CONNECTED_MODE"]):
        return None

    audit_path, other_paths, clock = _configuration(app)
    app.secret_key = app.config["OPERATOR_UI_SECRET_KEY"]
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_SAMESITE="Strict",
    )
    inactivity = int(app.config.get("OPERATOR_UI_INACTIVITY_SECONDS", 900))
    absolute = int(app.config.get("OPERATOR_UI_ABSOLUTE_SECONDS", 28800))
    if inactivity <= 0 or absolute <= 0 or inactivity > absolute:
        raise ConnectedModeConfigurationError("invalid connected session expiry")
    try:
        store = AuditStore(audit_path, other_paths)
    except AuditUnavailable as exc:
        raise ConnectedModeConfigurationError("unsafe audit store configuration") from exc
    if not store.verify_chain():
        raise ConnectedModeConfigurationError("audit chain integrity check failed")
    app.extensions["operator_ui_audit"] = store
    active_sessions = _ActiveSessionRegistry()
    app.extensions["operator_ui_active_sessions"] = active_sessions

    def now_epoch() -> int:
        return int(clock().timestamp())

    def csrf_token() -> str:
        token = session.get("_operator_csrf")
        if not isinstance(token, str):
            token = secrets.token_urlsafe(32)
            session["_operator_csrf"] = token
        return token

    def valid_csrf() -> bool:
        supplied = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
        expected = session.get("_operator_csrf")
        return (
            isinstance(supplied, str)
            and isinstance(expected, str)
            and hmac.compare_digest(supplied, expected)
        )

    def actor() -> tuple[str, int] | None:
        session_identifier = session.get("operator_session_id")
        identity = session.get("operator_actor")
        level = session.get("operator_level")
        issued = session.get("operator_issued_at")
        active = session.get("operator_last_active")
        current = now_epoch()
        authenticated = active_sessions.authenticate(
            session_identifier,
            identity,
            level,
            issued,
            active,
            current=current,
            inactivity=inactivity,
            absolute=absolute,
        )
        if authenticated is None:
            session.clear()
            return None
        session["operator_last_active"] = current
        return authenticated

    def csrf_protect(view: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(view)
        def protected(*args: Any, **kwargs: Any) -> Any:
            if not valid_csrf():
                return jsonify(classification="NON_OPERATIONAL/CSRF_REJECTED"), 400
            return view(*args, **kwargs)

        return protected

    app.extensions["operator_ui_csrf_protect"] = csrf_protect

    @app.get("/operator-ui/login")
    def operator_ui_login_form() -> Response:
        return jsonify(
            classification="NON_OPERATIONAL/AUTHENTICATION_REQUIRED",
            csrf_token=csrf_token(),
        )

    @app.post("/operator-ui/login")
    @csrf_protect
    def operator_ui_login() -> tuple[Response, int] | Response:
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        correct_user = hmac.compare_digest(
            username, str(app.config["OPERATOR_UI_USERNAME"])
        )
        correct_password = check_password_hash(
            str(app.config["OPERATOR_UI_PASSWORD_HASH"]), password
        )
        if not (correct_user and correct_password):
            return jsonify(classification="NON_OPERATIONAL/AUTHENTICATION_DENIED"), 401
        active_sessions.revoke(session.get("operator_session_id"))
        session.clear()
        current = now_epoch()
        identity = str(app.config["OPERATOR_UI_USERNAME"])
        level = int(app.config.get("OPERATOR_UI_LEVEL", 1))
        session_identifier = secrets.token_urlsafe(24)
        active_sessions.register(
            session_identifier,
            identity,
            level,
            current,
            inactivity=inactivity,
            absolute=absolute,
        )
        session.update(
            operator_actor=identity,
            operator_level=level,
            operator_session_id=session_identifier,
            operator_issued_at=current,
            operator_last_active=current,
            _operator_csrf=secrets.token_urlsafe(32),
        )
        return jsonify(
            classification="NON_OPERATIONAL/AUTHENTICATED",
            csrf_token=session["_operator_csrf"],
        )

    @app.post("/operator-ui/logout")
    @csrf_protect
    def operator_ui_logout() -> tuple[Response, int] | Response:
        if actor() is None:
            return (
                jsonify(classification="NON_OPERATIONAL/AUTHENTICATION_REQUIRED"),
                401,
            )
        active_sessions.revoke(session.get("operator_session_id"))
        session.clear()
        response = jsonify(classification="NON_OPERATIONAL/LOGGED_OUT")
        response.delete_cookie(
            app.config.get("SESSION_COOKIE_NAME", "session"),
            secure=True,
            httponly=True,
            samesite="Strict",
        )
        return response

    def operational_get(
        *,
        policy: str,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Prepare, audit and confirm a finite response before releasing its bytes."""

        def decorate(provider: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(provider)
            def protected(*args: Any, **kwargs: Any) -> Any:
                authenticated = actor()
                if authenticated is None:
                    return (
                        jsonify(classification="NON_OPERATIONAL/AUTHENTICATION_REQUIRED"),
                        401,
                    )
                identity, level = authenticated
                def audit_event(
                    classification: str,
                    *,
                    decision: str = "allowed",
                    sources: tuple[str, ...] = (),
                    content: tuple[str, ...] = (),
                    references: tuple[str, ...] = (),
                ) -> AuditEvent:
                    return AuditEvent(
                        event_id=secrets.token_urlsafe(24),
                        event_time_utc=_utc_text(clock()),
                        actor_identity=identity,
                        actor_level=level,
                        session_identifier=str(session["operator_session_id"]),
                        request_identifier=secrets.token_urlsafe(18),
                        route=request.path,
                        http_method=request.method,
                        authorization_decision=decision,
                        authorization_policy=policy,
                        evidence_source_identifiers=sources,
                        content_hashes=content,
                        reference_hashes=references,
                        deployed_commit=str(app.config["OPERATOR_UI_DEPLOYED_COMMIT"]),
                        deployed_tree=str(app.config["OPERATOR_UI_DEPLOYED_TREE"]),
                        deployed_version=str(app.config["OPERATOR_UI_DEPLOYED_VERSION"]),
                        response_classification=classification,
                    )

                if level < 1:
                    try:
                        store.append_and_confirm(
                            audit_event(
                                "NON_OPERATIONAL/AUTHORIZATION_DENIED",
                                decision="denied",
                            )
                        )
                    except (AuditUnavailable, OSError, sqlite3.Error):
                        return jsonify(NON_OPERATIONAL_ERROR), 503
                    return (
                        jsonify(classification="NON_OPERATIONAL/AUTHORIZATION_DENIED"),
                        403,
                    )

                try:
                    prepared = provider(*args, **kwargs)
                    _validate_prepared_disclosure(prepared)
                except Exception:
                    try:
                        store.append_and_confirm(
                            audit_event("NON_OPERATIONAL/PROVIDER_ERROR")
                        )
                    except (AuditUnavailable, OSError, sqlite3.Error):
                        return jsonify(NON_OPERATIONAL_ERROR), 503
                    return jsonify(PROVIDER_ERROR), 503

                try:
                    store.append_and_confirm(
                        audit_event(
                            prepared.classification.value,
                            sources=prepared.evidence_source_identifiers,
                            content=prepared.content_hashes,
                            references=prepared.reference_hashes,
                        )
                    )
                except (AuditUnavailable, OSError, sqlite3.Error):
                    return jsonify(NON_OPERATIONAL_ERROR), 503
                return Response(
                    prepared.body,
                    status=prepared.status_code,
                    content_type=prepared.content_type,
                )

            return protected

        return decorate

    app.extensions["operator_ui_operational_get"] = operational_get
    sentinel_hash = hashlib.sha256(b"connected-mode-sentinel-v1").hexdigest()

    @app.get("/operator-ui/connected/sentinel")
    @operational_get(
        policy="LEVEL_1_CONNECTED_SENTINEL",
    )
    def operator_ui_connected_sentinel() -> PreparedDisclosure:
        body = _canonical(
            {
                "classification": EvidenceStatus.AVAILABLE_FRESH.value,
                "sentinel": "connected-mode-boundary",
            }
        )
        return PreparedDisclosure(
            body=body,
            classification=EvidenceStatus.AVAILABLE_FRESH,
            evidence_source_identifiers=("server.connected_sentinel",),
            content_hashes=(sentinel_hash,),
        )

    return store
