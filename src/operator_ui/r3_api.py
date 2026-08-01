"""Narrow Level-2 submission and reconnect API for one exact prediction job."""
from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import replace
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from flask import Flask, jsonify, request, session

from .job_store import (
    IdempotencyConflict,
    IllegalTransition,
    Job,
    JobInput,
    JobStore,
    JobStoreError,
    Phase,
    TERMINAL_PHASES,
    resolve_audit_confirmation,
)
from .security import AuditUnavailable, OperationAuditEvent
from pathlib import Path
from src.predictor.on_demand import PredictionBlocked, VerifiedPredictionBundle, VerifiedPredictionBundleIndex, verify_indexed_prediction_bundle, verify_prediction_bundle_index

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
        "protocol_chain", "authenticated_cutoff",
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
    launch_once: Callable[[str, Callable[[Mapping[str, Any]], Any]], None]
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


def build_verified_bundle_reader(root: Path, store: JobStore) -> Callable[[Job], VerifiedPredictionBundle | None]:
    """Read exactly one producer-sealed bundle selected by persisted producer facts."""
    fixed_root=Path(root)
    def read(job: Job) -> VerifiedPredictionBundle | None:
        try:
            facts=[event["facts"] for event in store.events(job.job_id) if event["phase"] in {Phase.PRODUCER_COMPLETED.value,Phase.PREDICTION_READY.value}]
            prediction_ids={fact.get("prediction_id") for fact in facts if fact.get("prediction_id") is not None}
            if len(prediction_ids)!=1: return None
            prediction_id=next(iter(prediction_ids))
            view=verify_prediction_bundle_index(fixed_root,return_verified_view=True)
            if not isinstance(view,VerifiedPredictionBundleIndex): return None
            matches=[entry for entry in view.entries if entry.get("job_id")==job.job_id and entry.get("prediction_id")==prediction_id and entry.get("status")=="PREDICTION_READY" and entry.get("blocker_stage") is None]
            if len(matches)!=1: return None
            return verify_indexed_prediction_bundle(fixed_root,matches[0])
        except (OSError,ValueError,TypeError,JobStoreError,PredictionBlocked):
            return None
    return read


def _bounded(value: Any, name: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value.encode()) > maximum or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise R3Rejected(f"INVALID_{name.upper()}", 400)
    return value


def _public_facts(value: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value[name] for name in PUBLIC_FACTS if name in value}


def _verified_result(job: Job, value: Any, events: list[Mapping[str,Any]] | None = None) -> dict[str, Any] | None:
    if not isinstance(value, VerifiedPredictionBundle):
        return None
    result=value.result; entry=value.index_entry; manifest=value.manifest
    request_value=value.request
    if not isinstance(request_value,Mapping):
        return None
    if result.get("status") != "PREDICTION_READY" or entry.get("status") != "PREDICTION_READY":
        return None
    chain=result.get("evidence",{}).get("protocol_chain"); cutoff=result.get("evidence",{}).get("authenticated_cutoff")
    if not isinstance(chain,Mapping) or not isinstance(cutoff,Mapping):return None
    if not isinstance(events,list):return None
    claimed=[event.get("facts",{}).get("attempt_id") for event in events if event.get("phase")==Phase.CLAIMED.value]
    response_events=[event for event in events if event.get("phase")==Phase.RESPONSE_RECORDED.value]
    completed=[event for event in events if event.get("phase")==Phase.PRODUCER_COMPLETED.value]
    if len(claimed)!=1 or len(response_events)!=1 or len(completed)!=1:return None
    attempt=claimed[0]
    for event in (*response_events,*completed):
        facts=event.get("facts",{})
        if facts.get("attempt_id")!=attempt or facts.get("protocol_chain")!=chain or facts.get("authenticated_cutoff")!=cutoff:return None
    if (entry.get("job_id"),result.get("job_id"),manifest.get("job_id")) != (job.job_id,job.job_id,job.job_id):
        return None
    if (entry.get("prediction_id"),manifest.get("prediction_id")) != (result.get("prediction_id"),result.get("prediction_id")):
        return None
    race=result.get("race",{}); model=result.get("model",{}); config=result.get("config",{}); evidence=result.get("evidence",{})
    if (race.get("race_id"),race.get("jump_timestamp"),evidence.get("runner_set_sha256")) != (job.input.race_id,job.input.jump_timestamp,job.input.runner_set_sha256):
        return None
    if (model.get("requested"),model.get("resolved"),model.get("artifact_sha256"),model.get("artifact_manifest_sha256"),model.get("schema_sha256"),config.get("sha256")) != (job.input.model_selector,job.input.resolved_model_identity,job.input.model_sha256,job.input.model_manifest_sha256,job.input.model_schema_sha256,job.input.config_sha256):
        return None
    request_model=request_value.get("model",{})
    if (request_value.get("job_id"),request_value.get("race_id"),request_value.get("jump_timestamp"),request_value.get("runner_set_sha256"),request_value.get("odds_source"),request_value.get("config_sha256")) != (job.job_id,job.input.race_id,job.input.jump_timestamp,job.input.runner_set_sha256,job.input.odds_source,job.input.config_sha256):
        return None
    if (request_model.get("requested"),request_model.get("resolved"),request_model.get("model_sha256"),request_model.get("manifest_sha256"),request_model.get("schema_sha256")) != (job.input.model_selector,job.input.resolved_model_identity,job.input.model_sha256,job.input.model_manifest_sha256,job.input.model_schema_sha256):
        return None
    rows=result.get("prediction",{}).get("predictions") if isinstance(result.get("prediction"),Mapping) else None
    if not isinstance(rows,list):
        return None
    expected={(r["box"],r["name"],r["identity"],r.get("source_native_runner_id")) for r in job.input.fields()["ordered_runners"]}
    actual={(r.get("box_number"),r.get("dog_name"),r.get("identity"),r.get("source_native_runner_id")) for r in rows}
    if actual != expected or len(actual)!=len(rows):
        return None
    request_runners=request_value.get("runners")
    if not isinstance(request_runners,list) or {(r.get("box_number"),r.get("display_name"),r.get("identity"),r.get("source_native_runner_id")) for r in request_runners} != expected or len(request_runners)!=len(expected):
        return None
    probabilities=[{"rank":r["rank"],"runner_id":r["identity"],"box":r["box_number"],"name":r["dog_name"],"probability":r["probability"]} for r in rows]
    return {"schema":"operator_ui_verified_prediction_result_v1","verification_status":"VERIFIED","probabilities":probabilities,"evidence":{"prediction_id":result["prediction_id"],"job_id":job.job_id,"bundle_locator":value.directory,"manifest":dict(manifest),"index_entry":dict(entry)}}


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
        result = _verified_result(job, result_reader(job), events)
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
    app.extensions["operator_ui_r3_services"] = services
    limiter = _ActorRateLimit(services.rate_limit, services.rate_window_seconds)

    def authority() -> tuple[str, int] | None:
        actor = authenticate()
        if actor is None:
            return None
        return actor

    def response_error(code: str, status: int):
        return jsonify(schema="operator_ui_prediction_error_v1", classification=code), status

    @app.get(f"{API_PREFIX}/r3-capability", endpoint="operator_ui_r3_capability")
    def capability():
        actor = authority()
        if actor is None:
            return response_error("NON_OPERATIONAL/AUTHENTICATION_REQUIRED", 401)
        _identity, level = actor
        if level != 2:
            return response_error("NON_OPERATIONAL/LEVEL_2_REQUIRED", 403)
        return jsonify(schema="operator_ui_r3_capability_v1", authorized=True, runtime_configured=True, level=2)

    def confirm(intent: Mapping[str, Any], session_identifier: str):
        operation = str(intent["operation"])
        proposed = intent["proposed_event"]
        input_value = intent["input"]
        event = OperationAuditEvent(
            event_id=str(uuid.uuid4()), event_time_utc=services.clock().astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
            actor_identity=str(intent["actor_identity"]), actor_level=2,
            session_identifier=session_identifier, request_identifier=str(uuid.uuid4()),
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

    def dispatch_waiting(job: Job, confirm_audit) -> Job:
        if job.phase is not Phase.WAITING_FOR_CLAIM or job.attempt_claimed:
            return job
        try:
            services.launch_once(job.job_id, confirm_audit)
        except Exception as exc:
            current = services.job_store.get(job.job_id)
            if current.phase is Phase.WAITING_FOR_CLAIM and not current.attempt_claimed:
                return services.job_store.transition(current.job_id, Phase.FAILED, now=services.clock(), status="FAILED", reason="DISPATCH_FAILED", facts={"error": type(exc).__name__}, confirm_audit=confirm_audit)
            return current
        return services.job_store.get(job.job_id)

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
        session_identifier = str(session["operator_session_id"])
        confirm_audit = lambda intent: confirm(intent, session_identifier)
        try:
            resolved = services.resolve_submission(selected, now)
            if type(resolved) is not ResolvedSubmission or type(resolved.job_input) is not JobInput or not resolved.ordered_runners:
                raise R3Rejected("RACE_EVIDENCE_INVALID")
            ordered_runners = tuple(dict(runner) for runner in resolved.ordered_runners)
            if resolved.job_input.ordered_runners:
                if resolved.job_input.fields()["ordered_runners"] != [dict(runner) for runner in ordered_runners]:
                    raise R3Rejected("RUNNER_SET_BINDING_MISMATCH")
                job_input = resolved.job_input
            else:
                job_input = replace(resolved.job_input, ordered_runners=ordered_runners)
            job = services.job_store.create(actor_identity=identity, actor_level=2, operation="manual_prediction",
                idempotency_key=selected["idempotency_key"], job_input=job_input, now=now, confirm_audit=confirm_audit)
            newly_observed = job.phase is Phase.SUBMITTED
            launch_eligible = False
            if job.phase is Phase.SUBMITTED:
                try:
                    job = services.job_store.transition(job.job_id, Phase.VALIDATED, now=services.clock(), status="VALID", reason="validated", confirm_audit=confirm_audit)
                except IllegalTransition:
                    job = services.job_store.get(job.job_id)
            if job.phase is Phase.VALIDATED:
                try:
                    job = services.job_store.transition(job.job_id, Phase.WAITING_FOR_CLAIM, now=services.clock(), status="WAITING", reason="ready", confirm_audit=confirm_audit)
                    launch_eligible = True
                except IllegalTransition:
                    job = services.job_store.get(job.job_id)
            # WAITING is the durable dispatch queue.  Every observation may
            # notify the fixed dispatcher; claim_attempt remains the sole,
            # atomic owner of the one permitted worker attempt/process.
            if job.phase is Phase.WAITING_FOR_CLAIM and not job.attempt_claimed:
                launch_eligible = True
            if launch_eligible:
                job = dispatch_waiting(job, confirm_audit)
            return jsonify(_job_payload(services.job_store, job, services.read_verified_result)), 202 if newly_observed else 200
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
            session_identifier = str(session["operator_session_id"])
            job = dispatch_waiting(job, lambda intent: confirm(intent, session_identifier))
            return jsonify(_job_payload(services.job_store, job, services.read_verified_result))
        except (R3Rejected, JobStoreError):
            return response_error("JOB_NOT_FOUND", 404)

    app.extensions["operator_ui_r3_services"] = services
    return True


__all__ = ["R3Rejected", "R3Services", "ResolvedSubmission", "install_r3_api"]
