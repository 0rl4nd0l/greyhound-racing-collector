"""Narrow Level-2 submission and reconnect API for one exact prediction job."""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from flask import Flask, jsonify, request, session

from .job_store import (
    IdempotencyConflict,
    Job,
    JobInput,
    JobStore,
    JobStoreError,
    Phase,
    TERMINAL_PHASES,
    resolve_audit_confirmation,
)
from .security import AuditUnavailable, OperationAuditEvent

API_PREFIX = "/operator-ui/api/v1"
SUBMISSION_FIELDS = frozenset(
    {"race_id", "model_id", "config_id", "odds_source_id", "idempotency_key"}
)
PUBLIC_FACTS = frozenset(
    {
        "attempt_id", "prediction_id", "producer_job_id", "producer_blocker",
        "index_sha256", "result_sha256", "manifest_sha256", "logical_bundle_sha256",
        "runner_set_sha256", "model_sha256", "model_manifest_sha256",
        "model_schema_sha256", "config_sha256", "verification_status", "blocker",
        "producer_status", "research_only", "production_persisted", "betting_output",
    }
)


class R3Rejected(RuntimeError):
    def __init__(self, classification: str, status_code: int = 409):
        super().__init__(classification)
        self.classification = classification
        self.status_code = status_code


@dataclass(frozen=True)
class ResolvedSubmission:
    job_input: JobInput
    ordered_runners: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class R3Services:
    job_store: JobStore
    resolve_submission: Callable[[Mapping[str, str], datetime], ResolvedSubmission]
    launch_once: Callable[[str], None]
    read_verified_result: Callable[[Job], Mapping[str, Any] | None]
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    rate_limit: int = 5
    rate_window_seconds: int = 60


class _ActorRateLimit:
    def __init__(self, maximum: int, window: int):
        if type(maximum) is not int or type(window) is not int or not 1 <= maximum <= 100 or not 1 <= window <= 3600:
            raise ValueError("invalid Level-2 rate limit")
        self.maximum, self.window = maximum, window
        self._entries: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def admit(self, actor: str, now: float) -> bool:
        with self._lock:
            recent = [value for value in self._entries.get(actor, ()) if now - value < self.window]
            if len(recent) >= self.maximum:
                self._entries[actor] = recent
                return False
            recent.append(now)
            self._entries[actor] = recent
            return True


def _bounded(value: Any, name: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value.encode()) > maximum or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise R3Rejected(f"INVALID_{name.upper()}", 400)
    return value


def _public_facts(value: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value[name] for name in PUBLIC_FACTS if name in value}


def _verified_result(job: Job, value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or set(value) != {"schema", "verification_status", "probabilities", "evidence"}:
        return None
    if value.get("schema") != "operator_ui_verified_prediction_result_v1" or value.get("verification_status") != "VERIFIED":
        return None
    evidence = value.get("evidence")
    evidence_fields = {
        "job_id", "race_id", "jump_timestamp", "runner_set_sha256", "prediction_id",
        "request_id", "claim_id", "attempt_id", "response_id", "receipt_id", "consume_id",
        "source_hashes", "temporal_cutoff", "bundle_manifest", "model_sha256",
        "config_sha256", "input_sha256", "terminal_classification",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != evidence_fields:
        return None
    expected = (job.job_id, job.input.race_id, job.input.jump_timestamp, job.input.runner_set_sha256,
                job.input.model_sha256, job.input.config_sha256, job.input.identity_sha256, "PREDICTION_READY")
    actual = tuple(evidence[name] for name in ("job_id", "race_id", "jump_timestamp", "runner_set_sha256",
                                               "model_sha256", "config_sha256", "input_sha256", "terminal_classification"))
    if actual != expected or not isinstance(evidence["source_hashes"], Mapping) or not evidence["source_hashes"] or not isinstance(evidence["bundle_manifest"], Mapping) or not evidence["bundle_manifest"]:
        return None
    for digest in (*evidence["source_hashes"].values(),):
        if not isinstance(digest, str) or len(digest) != 64 or set(digest) - set("0123456789abcdef"):
            return None
    probabilities = value.get("probabilities")
    if not isinstance(probabilities, list) or not probabilities:
        return None
    total = 0.0; runner_ids = set()
    for index, row in enumerate(probabilities, 1):
        if not isinstance(row, Mapping) or set(row) != {"rank", "runner_id", "probability"} or row["rank"] != index:
            return None
        probability = row["probability"]
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1 or not isinstance(row["runner_id"], str) or not row["runner_id"] or row["runner_id"] in runner_ids:
            return None
        runner_ids.add(row["runner_id"]); total += probability
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        return None
    return {"schema": value["schema"], "verification_status": "VERIFIED", "probabilities": [dict(row) for row in probabilities], "evidence": dict(evidence)}


def _job_payload(store: JobStore, job: Job, result_reader: Callable[[Job], Mapping[str, Any] | None]) -> dict[str, Any]:
    events = store.events(job.job_id)
    timeline = [
        {"event_id": event["event_id"], "phase": event["phase"], "event_at": event["event_at"],
         "status": event["status"], "reason": event["reason"], "event_hash": event["event_hash"],
         "facts": _public_facts(event["facts"])}
        for event in events
    ]
    payload: dict[str, Any] = {
        "schema": "operator_ui_prediction_job_response_v1", "job_id": job.job_id,
        "phase": job.phase.value, "terminal": job.phase in TERMINAL_PHASES,
        "race_id": job.input.race_id, "jump_timestamp": job.input.jump_timestamp,
        "runner_set_sha256": job.input.runner_set_sha256,
        "model_id": job.input.model_selector, "resolved_model_identity": job.input.resolved_model_identity,
        "config_id": job.input.config_id, "odds_source_id": job.input.odds_source,
        "timeline": timeline, "result": None,
    }
    if job.phase is Phase.PREDICTION_READY:
        result = _verified_result(job, result_reader(job))
        if result is None:
            payload["blocker"] = "VERIFIED_RESULT_UNAVAILABLE"
        else:
            payload["result"] = result
    elif job.phase in TERMINAL_PHASES:
        payload["blocker"] = timeline[-1]["reason"]
    return payload


def install_r3_api(app: Flask, services: R3Services | None = None) -> bool:
    """Install R3 only when explicitly composed with server-owned services."""
    if services is None:
        return False
    authenticate = app.extensions.get("operator_ui_authenticated_actor")
    csrf = app.extensions.get("operator_ui_csrf_protect")
    audit = app.extensions.get("operator_ui_audit")
    if not callable(authenticate) or not callable(csrf) or audit is None:
        raise RuntimeError("R3 requires the installed connected security boundary")
    limiter = _ActorRateLimit(services.rate_limit, services.rate_window_seconds)

    def authority() -> tuple[str, int] | None:
        actor = authenticate()
        if actor is None:
            return None
        return actor

    def response_error(code: str, status: int):
        return jsonify(schema="operator_ui_prediction_error_v1", classification=code), status

    def confirm(intent: Mapping[str, Any]):
        operation = str(intent["operation"])
        proposed = intent["proposed_event"]
        input_value = intent["input"]
        event = OperationAuditEvent(
            event_id=str(uuid.uuid4()), event_time_utc=services.clock().astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
            actor_identity=str(intent["actor_identity"]), actor_level=2,
            session_identifier=str(session["operator_session_id"]), request_identifier=str(uuid.uuid4()),
            client_identity="same-origin-authenticated-session", operation=f"manual_prediction_{operation}",
            idempotency_key_sha256=str(intent["idempotency_key_sha256"]), job_id=str(intent["job_id"]),
            race_id=str(input_value["race_id"]), runner_set_sha256=str(input_value["runner_set_sha256"]),
            model_identity=str(input_value["resolved_model_identity"]), model_sha256=str(input_value["model_sha256"]),
            config_id=str(input_value["config_id"]), config_sha256=str(input_value["config_sha256"]),
            input_identity_sha256=str(intent["input_identity_sha256"]), prior_state=str(intent["prior_state"]),
            new_state=str(proposed["phase"]), status=str(proposed["status"]), reason=str(proposed["reason"]),
            reference_hashes=tuple(sorted({str(input_value["model_manifest_sha256"]), str(input_value["model_schema_sha256"])})),
        )
        audit_hash = audit.append_operation_and_confirm(event)
        return resolve_audit_confirmation(intent, audit_hash)

    @app.post(f"{API_PREFIX}/prediction-jobs", endpoint="operator_ui_r3_submit")
    @csrf
    def submit():
        actor = authority()
        if actor is None:
            return response_error("NON_OPERATIONAL/AUTHENTICATION_REQUIRED", 401)
        identity, level = actor
        if level != 2:
            return response_error("NON_OPERATIONAL/LEVEL_2_REQUIRED", 403)
        if not request.is_json:
            return response_error("INVALID_REQUEST_MEDIA_TYPE", 415)
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or set(body) != SUBMISSION_FIELDS:
            return response_error("INVALID_REQUEST_SCHEMA", 400)
        try:
            selected = {name: _bounded(body[name], name) for name in SUBMISSION_FIELDS}
        except R3Rejected as exc:
            return response_error(exc.classification, exc.status_code)
        if not limiter.admit(identity, time.monotonic()):
            return response_error("RATE_LIMITED", 429)
        now = services.clock()
        try:
            resolved = services.resolve_submission(selected, now)
            if type(resolved) is not ResolvedSubmission or type(resolved.job_input) is not JobInput or not resolved.ordered_runners:
                raise R3Rejected("RACE_EVIDENCE_INVALID")
            job = services.job_store.create(actor_identity=identity, actor_level=2, operation="manual_prediction",
                idempotency_key=selected["idempotency_key"], job_input=resolved.job_input, now=now, confirm_audit=confirm)
            created = job.phase is Phase.SUBMITTED
            if created:
                job = services.job_store.transition(job.job_id, Phase.VALIDATED, now=services.clock(), status="VALID", reason="validated", confirm_audit=confirm)
                job = services.job_store.transition(job.job_id, Phase.WAITING_FOR_CLAIM, now=services.clock(), status="WAITING", reason="ready", confirm_audit=confirm)
                services.launch_once(job.job_id)
                job = services.job_store.get(job.job_id)
            return jsonify(_job_payload(services.job_store, job, services.read_verified_result)), 202 if created else 200
        except IdempotencyConflict:
            return response_error("IDEMPOTENCY_CONFLICT", 409)
        except R3Rejected as exc:
            return response_error(exc.classification, exc.status_code)
        except (AuditUnavailable, JobStoreError):
            return response_error("MUTATION_AUDIT_UNAVAILABLE", 503)

    @app.get(f"{API_PREFIX}/prediction-jobs/<job_id>", endpoint="operator_ui_r3_job")
    def read_job(job_id: str):
        actor = authority()
        if actor is None:
            return response_error("NON_OPERATIONAL/AUTHENTICATION_REQUIRED", 401)
        identity, level = actor
        if level != 2:
            return response_error("NON_OPERATIONAL/LEVEL_2_REQUIRED", 403)
        try:
            job = services.job_store.get(_bounded(job_id, "job_id", 64))
            if job.actor_identity != identity:
                return response_error("JOB_NOT_FOUND", 404)
            return jsonify(_job_payload(services.job_store, job, services.read_verified_result))
        except (R3Rejected, JobStoreError):
            return response_error("JOB_NOT_FOUND", 404)

    app.extensions["operator_ui_r3_services"] = services
    return True


__all__ = ["R3Rejected", "R3Services", "ResolvedSubmission", "install_r3_api"]
