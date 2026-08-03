from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event

import pytest
from src.predictor.on_demand import BLOCKER_STAGE_BY_CODE

from src.operator_ui.job_store import (
    AttemptAlreadyClaimed, IdempotencyConflict, IllegalTransition, JobInput,
    JobStore, JobStoreError, Phase, VerifierAuthorizationError,
    canonical, resolve_audit_confirmation,
)

NOW = datetime(2026, 8, 1, tzinfo=timezone.utc)
H = hashlib.sha256(b"identity").hexdigest()
AUDIT = hashlib.sha256(b"confirmed-operation-audit").hexdigest()
CONFIRM = lambda intent: resolve_audit_confirmation(intent, AUDIT)


def inputs(race="race-5"):
    return JobInput(race, "2026-08-01T01:00:00Z", H, "latest-research", "model-v1", H, H, H, "manual-default", H, "auto", ({"box":1,"name":"ALPHA","identity":"ALPHA"},))


def store(tmp_path):
    return JobStore(tmp_path / "jobs.db", separate_from=(tmp_path / "canonical.db", tmp_path / "audit.db"))


def create(value, key="idempotency-key-1234", actor="operator"):
    return value.create(actor_identity=actor, actor_level=2, operation="manual_prediction", idempotency_key=key, job_input=inputs(), now=NOW, confirm_audit=CONFIRM)

def process_facts(job,attempt,*,status="PREDICTION_READY",blocker=None):
    empty=hashlib.sha256(b"").hexdigest()
    return {"attempt_id":attempt,"pid":123,"exit_code":0,"stdout_complete":True,"stdout_length":0,"stdout_sha256":empty,"stdout_prefix_length":0,"stdout_prefix_sha256":empty,"stderr_complete":True,"stderr_bytes":"","stderr_length":0,"stderr_sha256":empty,"stderr_prefix_length":0,"stderr_prefix_sha256":empty,"predictor_status":status,"prediction_id":"12345678-1234-4123-8123-123456789abc","producer_job_id":job.job_id,**({"producer_blocker":blocker} if blocker else {})}

def producer_completed(value,authority=None):
    job,attempt=value.claim_attempt(claimable(value,create(value)).job_id,now=NOW,confirm_audit=CONFIRM)
    job=value.transition(job.job_id,Phase.ATTEMPT_STARTED,now=NOW,status="RUNNING",reason="predictor_started",facts={"attempt_id":attempt,"pid":123},confirm_audit=CONFIRM)
    facts=process_facts(job,attempt)
    job=value.transition(job.job_id,Phase.RESPONSE_RECORDED,now=NOW,status="RECORDED",reason="bounded_process_response",facts=facts,confirm_audit=CONFIRM)
    return value.transition(job.job_id,Phase.PRODUCER_COMPLETED,now=NOW,status="PRODUCER_COMPLETED",reason="PRODUCER_PREDICTION_READY",facts=facts,confirm_audit=CONFIRM),facts

def verifier_facts(job,evidence,*,blocker=None,verification="VERIFIED",producer="PREDICTION_READY"):
    return {"prediction_id":evidence["prediction_id"],"job_id":job.job_id,"race_id":job.input.race_id,"jump_timestamp":job.input.jump_timestamp,"runner_set_sha256":job.input.runner_set_sha256,"resolved_model_identity":job.input.resolved_model_identity,"model_sha256":job.input.model_sha256,"model_manifest_sha256":job.input.model_manifest_sha256,"model_schema_sha256":job.input.model_schema_sha256,"config_id":job.input.config_id,"config_sha256":job.input.config_sha256,"index_sha256":H,"result_sha256":H,"manifest_sha256":H,"logical_bundle_sha256":H,"bundle_locator":"prediction_safe","producer_status":producer,"research_only":True,"production_persisted":False,"betting_output":False,"verification_status":verification,"blocker":blocker}

def producer_blocked(value,code):
    job,attempt=value.claim_attempt(claimable(value,create(value)).job_id,now=NOW,confirm_audit=CONFIRM)
    job=value.transition(job.job_id,Phase.ATTEMPT_STARTED,now=NOW,status="RUNNING",reason="predictor_started",facts={"attempt_id":attempt,"pid":123},confirm_audit=CONFIRM)
    blocker={"code":code,"stage":BLOCKER_STAGE_BY_CODE[code]}; facts=process_facts(job,attempt,status="PREDICTION_BLOCKED",blocker=blocker)
    job=value.transition(job.job_id,Phase.RESPONSE_RECORDED,now=NOW,status="RECORDED",reason="bounded_process_response",facts=facts,confirm_audit=CONFIRM)
    return value.transition(job.job_id,Phase.PRODUCER_COMPLETED,now=NOW,status="PRODUCER_COMPLETED",reason=f"PRODUCER_PREDICTION_BLOCKED:{code}",facts=facts,confirm_audit=CONFIRM),facts,blocker


def claimable(value, job):
    job = value.transition(job.job_id, Phase.VALIDATED, now=NOW, status="VALID", reason="validated", confirm_audit=CONFIRM)
    return value.transition(job.job_id, Phase.WAITING_FOR_CLAIM, now=NOW, status="WAITING", reason="ready", confirm_audit=CONFIRM)


def test_store_is_separate_and_immutable(tmp_path):
    canonical = tmp_path / "canonical.db"; canonical.write_bytes(b"canonical")
    audit = tmp_path / "audit.db"; audit.write_bytes(b"audit")
    with pytest.raises(JobStoreError): JobStore(canonical, separate_from=(canonical, audit))
    value = store(tmp_path); job = create(value)
    with sqlite3.connect(value.path) as db:
        for sql in ("UPDATE jobs SET operation='x'", "DELETE FROM job_events"):
            with pytest.raises(sqlite3.IntegrityError): db.execute(sql)
    assert canonical.read_bytes() == b"canonical" and audit.read_bytes() == b"audit"
    assert value.get(job.job_id).input.race_id == "race-5"


def test_idempotency_duplicate_conflict_cross_actor_and_raw_key_absent(tmp_path):
    value = store(tmp_path); first = create(value)
    assert create(value).job_id == first.job_id
    with pytest.raises(IdempotencyConflict):
        value.create(actor_identity="operator", actor_level=2, operation="manual_prediction", idempotency_key="idempotency-key-1234", job_input=inputs("race-6"), now=NOW, confirm_audit=CONFIRM)
    assert create(value, actor="other").job_id != first.job_id
    assert b"idempotency-key-1234" not in value.path.read_bytes()


def test_concurrent_duplicate_creation_is_one_job(tmp_path):
    value = store(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(lambda _: create(value).job_id, range(20)))
    assert len(set(ids)) == 1


@pytest.mark.parametrize("read", ["get", "events"])
def test_concurrent_commit_during_integrity_verification_uses_one_read_snapshot(tmp_path, monkeypatch, read):
    value = store(tmp_path); job = create(value)
    with sqlite3.connect(value.path) as db:
        assert db.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    writer = store(tmp_path); verification_reached = Event(); writer_committed = Event()
    original_rows_hash = value._rows_hash

    def pause_after_anchor_read(db):
        verification_reached.set()
        assert writer_committed.wait(5)
        return original_rows_hash(db)

    monkeypatch.setattr(value, "_rows_hash", pause_after_anchor_read)

    def commit_transition():
        assert verification_reached.wait(5)
        writer.transition(job.job_id, Phase.VALIDATED, now=NOW, status="VALID", reason="validated", confirm_audit=CONFIRM)
        writer_committed.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        committed = pool.submit(commit_transition)
        result = value.get(job.job_id) if read == "get" else value.events(job.job_id)
        committed.result(timeout=5)

    assert (result.phase if read == "get" else Phase(result[-1]["phase"])) is Phase.SUBMITTED
    assert writer.get(job.job_id).phase is Phase.VALIDATED


def test_transitions_order_terminal_and_attempt_are_stable(tmp_path):
    value = store(tmp_path); job = create(value)
    with pytest.raises(IllegalTransition): value.transition(job.job_id, Phase.CLAIMED, now=NOW, status="x", reason="x", confirm_audit=CONFIRM)
    job = claimable(value, job)
    with pytest.raises(IllegalTransition): value.transition(job.job_id, Phase.REJECTED, now=NOW - timedelta(seconds=1), status="x", reason="x", confirm_audit=CONFIRM)
    job, attempt = value.claim_attempt(job.job_id, now=NOW, confirm_audit=CONFIRM)
    assert attempt and job.phase is Phase.CLAIMED
    with pytest.raises(AttemptAlreadyClaimed): value.claim_attempt(job.job_id, now=NOW, confirm_audit=CONFIRM)
    job = value.transition(job.job_id, Phase.FAILED, now=NOW, status="FAILED", reason="PROCESS_LAUNCH_FAILED", facts={"attempt_id":attempt,"error":"OSError"}, confirm_audit=CONFIRM)
    with pytest.raises(IllegalTransition): value.transition(job.job_id, Phase.ATTEMPT_STARTED, now=NOW, status="x", reason="x", confirm_audit=CONFIRM)


def test_reopen_returns_same_job_and_claim_and_tamper_fails(tmp_path):
    value = store(tmp_path); job = claimable(value, create(value)); job, attempt = value.claim_attempt(job.job_id, now=NOW, confirm_audit=CONFIRM)
    reopened = store(tmp_path)
    assert reopened.get(job.job_id).attempt_claimed and reopened.get(job.job_id).job_id == job.job_id
    with pytest.raises(AttemptAlreadyClaimed): reopened.claim_attempt(job.job_id, now=NOW, confirm_audit=CONFIRM)
    with sqlite3.connect(value.path) as db:
        db.execute("DROP TRIGGER events_no_update")
        db.execute("UPDATE job_events SET reason='tampered' WHERE sequence=1")
    assert reopened.verify() is False
    with pytest.raises(JobStoreError): reopened.get(job.job_id)
    with pytest.raises(JobStoreError): reopened.events(job.job_id)


@pytest.mark.parametrize("operation", ["create", "transition", "claim"])
def test_audit_failure_leaves_mutation_unapplied(tmp_path, operation):
    value = store(tmp_path)
    failed = lambda intent: (_ for _ in ()).throw(RuntimeError("audit unavailable"))
    if operation == "create":
        with pytest.raises(JobStoreError):
            value.create(actor_identity="operator", actor_level=2, operation="manual_prediction", idempotency_key="idempotency-key-1234", job_input=inputs(), now=NOW, confirm_audit=failed)
        with sqlite3.connect(value.path) as db: assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
        return
    job = create(value)
    if operation == "transition":
        with pytest.raises(JobStoreError): value.transition(job.job_id, Phase.VALIDATED, now=NOW, status="VALID", reason="ok", confirm_audit=failed)
        assert value.get(job.job_id).phase is Phase.SUBMITTED
    else:
        job = claimable(value, job)
        with pytest.raises(JobStoreError): value.claim_attempt(job.job_id, now=NOW, confirm_audit=failed)
        assert not value.get(job.job_id).attempt_claimed


@pytest.mark.parametrize("sql", [
    "DROP TRIGGER attempts_no_update",
    "DROP TRIGGER jobs_no_update",
    "DROP TRIGGER events_no_delete",
    "INSERT INTO job_attempts(attempt_id,schema,job_id,claimed_at,audit_hash) SELECT '00000000-0000-0000-0000-000000000001',schema,job_id,created_at,creation_audit_hash FROM jobs",
])
def test_whole_store_schema_and_extra_row_tamper_fails_closed(tmp_path, sql):
    value=store(tmp_path); job=create(value)
    with sqlite3.connect(value.path) as db: db.execute(sql)
    assert value.verify() is False
    with pytest.raises(JobStoreError): value.get(job.job_id)


def test_reopen_does_not_repair_removed_integrity_trigger(tmp_path):
    value=store(tmp_path); create(value)
    with sqlite3.connect(value.path) as db: db.execute("DROP TRIGGER jobs_no_update")
    with pytest.raises(JobStoreError): store(tmp_path)

def test_same_named_replacement_trigger_fails_closed(tmp_path):
    value=store(tmp_path); job=create(value)
    with sqlite3.connect(value.path) as db:
        db.execute("DROP TRIGGER jobs_no_update")
        db.execute("CREATE TRIGGER jobs_no_update BEFORE UPDATE ON jobs BEGIN SELECT 1; END")
    assert not value.verify()
    with pytest.raises(JobStoreError): value.get(job.job_id)

def test_audit_intent_is_complete_proposed_persisted_truth(tmp_path):
    intents=[]; value=store(tmp_path)
    job=value.create(actor_identity="operator",actor_level=2,operation="manual_prediction",idempotency_key="idempotency-key-1234",job_input=inputs(),now=NOW,confirm_audit=lambda intent:(intents.append(intent) or resolve_audit_confirmation(intent,AUDIT)))
    intent=intents[0]
    assert intent["actor_identity"]==job.actor_identity and intent["input"]==job.input.fields()
    assert intent["proposed_event"]=={"schema":"operator_ui_manual_prediction_event_v2","phase":"SUBMITTED","event_at":job.created_at,"status":"ACCEPTED","reason":"submitted","facts":{}}

def test_only_verifier_capability_can_finalize_producer_completion(tmp_path):
    authority=object(); value=JobStore(tmp_path/"jobs.db",separate_from=(tmp_path/"canonical.db",tmp_path/"audit.db"),verifier_authority=authority); job,attempt=value.claim_attempt(claimable(value,create(value)).job_id,now=NOW,confirm_audit=CONFIRM)
    job=value.transition(job.job_id,Phase.ATTEMPT_STARTED,now=NOW,status="RUNNING",reason="predictor_started",facts={"attempt_id":attempt,"pid":123},confirm_audit=CONFIRM)
    evidence={"attempt_id":attempt,"pid":123,"exit_code":0,"stdout_complete":True,"stdout_length":0,"stdout_sha256":hashlib.sha256(b"").hexdigest(),"stdout_prefix_length":0,"stdout_prefix_sha256":hashlib.sha256(b"").hexdigest(),"stderr_complete":True,"stderr_bytes":"","stderr_length":0,"stderr_sha256":hashlib.sha256(b"").hexdigest(),"stderr_prefix_length":0,"stderr_prefix_sha256":hashlib.sha256(b"").hexdigest(),"predictor_status":"PREDICTION_READY","prediction_id":"12345678-1234-4123-8123-123456789abc","producer_job_id":job.job_id}
    job=value.transition(job.job_id,Phase.RESPONSE_RECORDED,now=NOW,status="RECORDED",reason="bounded_process_response",facts=evidence,confirm_audit=CONFIRM)
    job=value.transition(job.job_id,Phase.PRODUCER_COMPLETED,now=NOW,status="PRODUCER_COMPLETED",reason="PRODUCER_PREDICTION_READY",facts=evidence,confirm_audit=CONFIRM)
    with pytest.raises(IllegalTransition): value.transition(job.job_id,Phase.PREDICTION_READY,now=NOW,status="READY",reason="verified",facts={},confirm_audit=CONFIRM)
    facts={"prediction_id":evidence["prediction_id"],"job_id":job.job_id,"race_id":job.input.race_id,"jump_timestamp":job.input.jump_timestamp,"runner_set_sha256":job.input.runner_set_sha256,"resolved_model_identity":job.input.resolved_model_identity,"model_sha256":job.input.model_sha256,"model_manifest_sha256":job.input.model_manifest_sha256,"model_schema_sha256":job.input.model_schema_sha256,"config_id":job.input.config_id,"config_sha256":job.input.config_sha256,"index_sha256":H,"result_sha256":H,"manifest_sha256":H,"logical_bundle_sha256":H,"bundle_locator":"prediction_20260801T000001000000+0000_123456789abc","producer_status":"PREDICTION_READY","research_only":True,"production_persisted":False,"betting_output":False,"verification_status":"VERIFIED","blocker":None}
    with pytest.raises(VerifierAuthorizationError): value.verifier_transition(job.job_id,Phase.PREDICTION_READY,capability=object(),now=NOW,status="READY",reason="verified",facts=facts,confirm_audit=CONFIRM)
    assert not hasattr(value,"verifier_capability")
    assert value.verifier_transition(job.job_id,Phase.PREDICTION_READY,capability=authority,now=NOW,status="READY",reason="verified",facts=facts,confirm_audit=CONFIRM).phase is Phase.PREDICTION_READY

def test_store_without_composition_authority_cannot_finalize(tmp_path):
    value=store(tmp_path)
    assert not hasattr(value,"verifier_capability")
    with pytest.raises(VerifierAuthorizationError):
        value.verifier_transition("job_"+"0"*32,Phase.PREDICTION_READY,capability=object(),now=NOW,status="READY",reason="verified",facts={},confirm_audit=CONFIRM)

def test_failed_after_start_empty_is_rejected(tmp_path):
    value=store(tmp_path); job,attempt=value.claim_attempt(claimable(value,create(value)).job_id,now=NOW,confirm_audit=CONFIRM)
    job=value.transition(job.job_id,Phase.ATTEMPT_STARTED,now=NOW,status="RUNNING",reason="predictor_started",facts={"attempt_id":attempt,"pid":123},confirm_audit=CONFIRM)
    with pytest.raises(JobStoreError):
        value.transition(job.job_id,Phase.FAILED,now=NOW,status="FAILED",reason="POST_SPAWN_FAILURE",facts={},confirm_audit=CONFIRM)
    assert value.get(job.job_id).phase is Phase.ATTEMPT_STARTED

def test_producer_arbitrary_vocab_is_rejected(tmp_path):
    value=store(tmp_path); job,attempt=value.claim_attempt(claimable(value,create(value)).job_id,now=NOW,confirm_audit=CONFIRM)
    job=value.transition(job.job_id,Phase.ATTEMPT_STARTED,now=NOW,status="RUNNING",reason="predictor_started",facts={"attempt_id":attempt,"pid":123},confirm_audit=CONFIRM)
    empty=hashlib.sha256(b"").hexdigest()
    evidence={"attempt_id":attempt,"pid":123,"exit_code":0,"stdout_complete":True,"stdout_length":0,"stdout_sha256":empty,"stdout_prefix_length":0,"stdout_prefix_sha256":empty,"stderr_complete":True,"stderr_length":0,"stderr_sha256":empty,"stderr_prefix_length":0,"stderr_prefix_sha256":empty,"predictor_status":"TOTALLY_READY","prediction_id":"12345678-1234-4123-8123-123456789abc","producer_job_id":job.job_id}
    with pytest.raises(JobStoreError):
        value.transition(job.job_id,Phase.RESPONSE_RECORDED,now=NOW,status="RECORDED",reason="bounded_process_response",facts=evidence,confirm_audit=CONFIRM)
    assert value.get(job.job_id).phase is Phase.ATTEMPT_STARTED

def test_verifier_failure_requires_exact_nonempty_blocker(tmp_path):
    authority=object(); value=JobStore(tmp_path/"jobs.db",verifier_authority=authority); job,attempt=value.claim_attempt(claimable(value,create(value)).job_id,now=NOW,confirm_audit=CONFIRM)
    empty=hashlib.sha256(b"").hexdigest(); evidence={"attempt_id":attempt,"pid":123,"exit_code":0,"stdout_complete":True,"stdout_length":0,"stdout_sha256":empty,"stdout_prefix_length":0,"stdout_prefix_sha256":empty,"stderr_complete":True,"stderr_length":0,"stderr_sha256":empty,"stderr_prefix_length":0,"stderr_prefix_sha256":empty,"predictor_status":"PREDICTION_READY","prediction_id":"12345678-1234-4123-8123-123456789abc","producer_job_id":job.job_id}
    job=value.transition(job.job_id,Phase.ATTEMPT_STARTED,now=NOW,status="RUNNING",reason="predictor_started",facts={"attempt_id":attempt,"pid":123},confirm_audit=CONFIRM)
    job=value.transition(job.job_id,Phase.RESPONSE_RECORDED,now=NOW,status="RECORDED",reason="bounded_process_response",facts=evidence,confirm_audit=CONFIRM)
    job=value.transition(job.job_id,Phase.PRODUCER_COMPLETED,now=NOW,status="PRODUCER_COMPLETED",reason="PRODUCER_PREDICTION_READY",facts=evidence,confirm_audit=CONFIRM)
    facts={"prediction_id":evidence["prediction_id"],"job_id":job.job_id,"race_id":job.input.race_id,"jump_timestamp":job.input.jump_timestamp,"runner_set_sha256":job.input.runner_set_sha256,"resolved_model_identity":job.input.resolved_model_identity,"model_sha256":job.input.model_sha256,"model_manifest_sha256":job.input.model_manifest_sha256,"model_schema_sha256":job.input.model_schema_sha256,"config_id":job.input.config_id,"config_sha256":job.input.config_sha256,"index_sha256":H,"result_sha256":H,"manifest_sha256":H,"logical_bundle_sha256":H,"bundle_locator":"prediction_safe","producer_status":"PREDICTION_READY","research_only":True,"production_persisted":False,"betting_output":False,"verification_status":"FAILED","blocker":None}
    with pytest.raises(ValueError): value.verifier_transition(job.job_id,Phase.FAILED,capability=authority,now=NOW,status="FAILED",reason="verification_failed",facts=facts,confirm_audit=CONFIRM)
    assert value.get(job.job_id).phase is Phase.PRODUCER_COMPLETED

def test_claimed_failure_rejects_reviewer_impossible_error_shape(tmp_path):
    value=store(tmp_path); job,attempt=value.claim_attempt(claimable(value,create(value)).job_id,now=NOW,confirm_audit=CONFIRM)
    with pytest.raises(JobStoreError):
        value.transition(job.job_id,Phase.FAILED,now=NOW,status="FAILED",reason="terminal",facts={"impossible":[]},confirm_audit=CONFIRM)
    assert value.get(job.job_id).phase is Phase.CLAIMED

def test_confirmation_rejects_absent_prior_row_collection_before_persistence(tmp_path):
    value=store(tmp_path); observed=[]
    def malformed(intent):
        changed=copy.deepcopy(intent)
        del changed["complete_proposal"]["prior_rows"]["job_attempts"]
        observed.append(changed)
        return resolve_audit_confirmation(changed,AUDIT)
    with pytest.raises(JobStoreError):
        value.create(actor_identity="operator",actor_level=2,operation="manual_prediction",idempotency_key="idempotency-key-1234",job_input=inputs(),now=NOW,confirm_audit=malformed)
    with sqlite3.connect(value.path) as db:
        assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0

_RECEIPT_MUTATIONS=("intent_extra","proposal_missing","preimage_extra","rows_missing","row_extra","anchor_type","prior_hash","event_mismatch","count","marker","presence")

def _mutate_receipt(intent,mutation):
    changed=copy.deepcopy(intent); proposal=changed["complete_proposal"]
    if mutation=="intent_extra": changed["legacy"]=False
    elif mutation=="proposal_missing": del proposal["event_hash_derivation"]
    elif mutation=="preimage_extra": proposal["event_preimage"]["legacy"]=False
    elif mutation=="rows_missing": del proposal["prior_rows"]["job_attempts"]
    elif mutation=="row_extra":
        table=next((name for name,rows in proposal["prior_rows"].items() if rows),None)
        if table: proposal["prior_rows"][table][0]["legacy"]=False
        else: proposal["prior_rows"]["jobs"]=[{"legacy":False}]
    elif mutation=="anchor_type": changed["prior_store_anchor"]["mutation_count"]=False
    elif mutation=="prior_hash": changed["prior_event_hash"]="f"*64
    elif mutation=="event_mismatch": changed["proposed_event"]["reason"]="different"
    elif mutation=="count": proposal["next_store_mutation_count"]+=1
    elif mutation=="marker":
        if proposal["prior_rows"]["jobs"]: proposal["prior_rows"]["jobs"][0]["operation"]="x<confirmed-audit-sha256>"
        else: proposal["event_preimage"]["reason"]="<confirmed-audit-sha256>"
    elif mutation=="presence": proposal["job_row"]={} if proposal["job_row"] is None else None
    return changed

@pytest.mark.parametrize("operation",["create","claim","transition","verify"])
@pytest.mark.parametrize("mutation",_RECEIPT_MUTATIONS)
def test_confirmation_exact_nested_shape_matrix_fails_before_target_mutation(tmp_path,operation,mutation):
    authority=object(); value=JobStore(tmp_path/"jobs.db",verifier_authority=authority)
    def bad(intent): return resolve_audit_confirmation(_mutate_receipt(intent,mutation),AUDIT)
    if operation=="create":
        with pytest.raises(JobStoreError): value.create(actor_identity="operator",actor_level=2,operation="manual_prediction",idempotency_key="idempotency-key-1234",job_input=inputs(),now=NOW,confirm_audit=bad)
        with sqlite3.connect(value.path) as db: assert db.execute("SELECT count(*) FROM jobs").fetchone()[0]==0
    elif operation=="claim":
        job=claimable(value,create(value))
        with pytest.raises(JobStoreError): value.claim_attempt(job.job_id,now=NOW,confirm_audit=bad)
        assert not value.get(job.job_id).attempt_claimed
    elif operation=="transition":
        job=create(value)
        with pytest.raises(JobStoreError): value.transition(job.job_id,Phase.VALIDATED,now=NOW,status="VALID",reason="validated",confirm_audit=bad)
        assert value.get(job.job_id).phase is Phase.SUBMITTED
    else:
        job,evidence=producer_completed(value,authority)
        facts={"prediction_id":evidence["prediction_id"],"job_id":job.job_id,"race_id":job.input.race_id,"jump_timestamp":job.input.jump_timestamp,"runner_set_sha256":job.input.runner_set_sha256,"resolved_model_identity":job.input.resolved_model_identity,"model_sha256":job.input.model_sha256,"model_manifest_sha256":job.input.model_manifest_sha256,"model_schema_sha256":job.input.model_schema_sha256,"config_id":job.input.config_id,"config_sha256":job.input.config_sha256,"index_sha256":H,"result_sha256":H,"manifest_sha256":H,"logical_bundle_sha256":H,"bundle_locator":"prediction_safe","producer_status":"PREDICTION_READY","research_only":True,"production_persisted":False,"betting_output":False,"verification_status":"VERIFIED","blocker":None}
        with pytest.raises(JobStoreError): value.verifier_transition(job.job_id,Phase.PREDICTION_READY,capability=authority,now=NOW,status="READY",reason="verified",facts=facts,confirm_audit=bad)
        assert value.get(job.job_id).phase is Phase.PRODUCER_COMPLETED

@pytest.mark.parametrize("code,stage",sorted(BLOCKER_STAGE_BY_CODE.items()))
def test_every_sealed_producer_blocker_preserves_exact_identity_through_verifier_and_reopen(tmp_path,code,stage):
    authority=object(); value=JobStore(tmp_path/"jobs.db",verifier_authority=authority); job,evidence,blocker=producer_blocked(value,code)
    facts=verifier_facts(job,evidence,blocker=blocker,verification="REJECTED",producer="PREDICTION_BLOCKED")
    final=value.verifier_transition(job.job_id,Phase.REJECTED,capability=authority,now=NOW,status="REJECTED",reason="verification_rejected",facts=facts,confirm_audit=CONFIRM)
    assert final.phase is Phase.REJECTED and JobStore(value.path,verifier_authority=authority).verify()

@pytest.mark.parametrize("mutation",["code","drop","stage","extra","producer_status","reason"])
def test_verifier_rejects_relabelled_dropped_or_contradictory_producer_blocker(tmp_path,mutation):
    authority=object(); value=JobStore(tmp_path/"jobs.db",verifier_authority=authority); code="POST_JUMP"; job,evidence,blocker=producer_blocked(value,code)
    facts=verifier_facts(job,evidence,blocker=copy.deepcopy(blocker),verification="REJECTED",producer="PREDICTION_BLOCKED"); reason="verification_rejected"
    if mutation=="code": facts["blocker"]["code"]="NO_MATCH"
    elif mutation=="drop": facts["blocker"]=None
    elif mutation=="stage": facts["blocker"]["stage"]="SCORING"
    elif mutation=="extra": facts["blocker"]["explanation"]="invented"
    elif mutation=="producer_status": facts["producer_status"]="PREDICTION_READY"
    else: reason="verification_failed"
    with pytest.raises((ValueError,JobStoreError)):
        value.verifier_transition(job.job_id,Phase.REJECTED,capability=authority,now=NOW,status="REJECTED",reason=reason,facts=facts,confirm_audit=CONFIRM)
    assert value.get(job.job_id).phase is Phase.PRODUCER_COMPLETED

def _rehash_job_rows(db):
    previous_by_job={}
    for row in db.execute("SELECT * FROM job_events ORDER BY sequence").fetchall():
        names=[item[1] for item in db.execute("PRAGMA table_info(job_events)")]; event=dict(zip(names,row)); previous=previous_by_job.get(event["job_id"],"0"*64)
        fields={"event_id":event["event_id"],"schema":event["schema"],"job_id":event["job_id"],"phase":event["phase"],"event_at":event["event_at"],"status":event["status"],"reason":event["reason"],"facts":json.loads(event["facts_json"]),"audit_hash":event["audit_hash"],"previous_event_hash":previous}
        digest=hashlib.sha256(canonical(fields)).hexdigest(); db.execute("UPDATE job_events SET previous_event_hash=?,event_hash=? WHERE sequence=?",(previous,digest,event["sequence"])); previous_by_job[event["job_id"]]=digest
    payload={name:[dict(zip([item[1] for item in db.execute(f"PRAGMA table_info({name})")],row)) for row in db.execute(f"SELECT * FROM {name} ORDER BY sequence")] for name in ("jobs","job_events","job_attempts")}
    db.execute("UPDATE store_anchor SET store_hash=?",(hashlib.sha256(canonical(payload)).hexdigest(),))

@pytest.mark.parametrize("mutation",["predecessor","target","status","reason","missing","extra","type","coupling"])
def test_rehashed_impossible_durable_event_matrix_fails_reopen_and_next_append(tmp_path,mutation):
    value=store(tmp_path); job,attempt=value.claim_attempt(claimable(value,create(value)).job_id,now=NOW,confirm_audit=CONFIRM)
    with sqlite3.connect(value.path) as db:
        db.row_factory=sqlite3.Row; db.execute("DROP TRIGGER events_no_update")
        sequence=db.execute("SELECT max(sequence) FROM job_events").fetchone()[0]; facts={"attempt_id":attempt}
        if mutation=="predecessor": db.execute("UPDATE job_events SET phase='SUBMITTED' WHERE sequence=?",(sequence-1,))
        elif mutation=="target": db.execute("UPDATE job_events SET phase='FAILED' WHERE sequence=?",(sequence,))
        elif mutation=="status": db.execute("UPDATE job_events SET status='INVENTED' WHERE sequence=?",(sequence,))
        elif mutation=="reason": db.execute("UPDATE job_events SET reason='invented' WHERE sequence=?",(sequence,))
        elif mutation=="missing": facts={}
        elif mutation=="extra": facts["impossible"]=[]
        elif mutation=="type": facts["attempt_id"]=False
        else: facts["attempt_id"]="00000000-0000-4000-8000-000000000000"
        if mutation in {"missing","extra","type","coupling"}: db.execute("UPDATE job_events SET facts_json=? WHERE sequence=?",(json.dumps(facts,sort_keys=True,separators=(",",":")),sequence))
        _rehash_job_rows(db)
    assert not value.verify()
    with pytest.raises(JobStoreError): value.get(job.job_id)
