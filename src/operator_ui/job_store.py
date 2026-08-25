"""Fail-closed durable one-attempt manual-prediction control store."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

JOB_SCHEMA = "operator_ui_manual_prediction_job_v2"
EVENT_SCHEMA = "operator_ui_manual_prediction_event_v2"
ATTEMPT_SCHEMA = "operator_ui_manual_prediction_attempt_v2"
STORE_SCHEMA = "operator_ui_manual_prediction_store_v2"
ZERO_HASH = "0" * 64
AUDIT_HASH_BINDING = "<confirmed-audit-sha256>"

_SCHEMA_SQL = f"""
PRAGMA foreign_keys=ON;
CREATE TABLE store_anchor(singleton INTEGER PRIMARY KEY CHECK(singleton=1),schema TEXT NOT NULL,mutation_count INTEGER NOT NULL,store_hash TEXT NOT NULL);
CREATE TABLE jobs(sequence INTEGER PRIMARY KEY AUTOINCREMENT,job_id TEXT UNIQUE NOT NULL,schema TEXT NOT NULL,actor_identity TEXT NOT NULL,actor_level INTEGER NOT NULL,operation TEXT NOT NULL,idempotency_key_sha256 TEXT NOT NULL,input_identity_sha256 TEXT NOT NULL,input_json TEXT NOT NULL,created_at TEXT NOT NULL,creation_audit_hash TEXT NOT NULL,UNIQUE(actor_identity,operation,idempotency_key_sha256));
CREATE TABLE job_events(sequence INTEGER PRIMARY KEY AUTOINCREMENT,event_id TEXT UNIQUE NOT NULL,schema TEXT NOT NULL,job_id TEXT NOT NULL REFERENCES jobs(job_id),phase TEXT NOT NULL,event_at TEXT NOT NULL,status TEXT NOT NULL,reason TEXT NOT NULL,facts_json TEXT NOT NULL,audit_hash TEXT NOT NULL,previous_event_hash TEXT NOT NULL,event_hash TEXT UNIQUE NOT NULL);
CREATE TABLE job_attempts(sequence INTEGER PRIMARY KEY AUTOINCREMENT,attempt_id TEXT UNIQUE NOT NULL,schema TEXT NOT NULL,job_id TEXT UNIQUE NOT NULL REFERENCES jobs(job_id),claimed_at TEXT NOT NULL,audit_hash TEXT NOT NULL);
CREATE TRIGGER jobs_no_update BEFORE UPDATE ON jobs BEGIN SELECT RAISE(ABORT,'jobs immutable'); END;
CREATE TRIGGER jobs_no_delete BEFORE DELETE ON jobs BEGIN SELECT RAISE(ABORT,'jobs immutable'); END;
CREATE TRIGGER events_no_update BEFORE UPDATE ON job_events BEGIN SELECT RAISE(ABORT,'events immutable'); END;
CREATE TRIGGER events_no_delete BEFORE DELETE ON job_events BEGIN SELECT RAISE(ABORT,'events immutable'); END;
CREATE TRIGGER attempts_no_update BEFORE UPDATE ON job_attempts BEGIN SELECT RAISE(ABORT,'attempts immutable'); END;
CREATE TRIGGER attempts_no_delete BEFORE DELETE ON job_attempts BEGIN SELECT RAISE(ABORT,'attempts immutable'); END;
CREATE TRIGGER anchor_no_delete BEFORE DELETE ON store_anchor BEGIN SELECT RAISE(ABORT,'anchor required'); END;
INSERT INTO store_anchor VALUES(1,'{STORE_SCHEMA}',0,'{ZERO_HASH}');
"""

class JobStoreError(RuntimeError): pass
class IdempotencyConflict(JobStoreError): pass
class IllegalTransition(JobStoreError): pass
class AttemptAlreadyClaimed(JobStoreError): pass
class VerifierAuthorizationError(JobStoreError): pass
class ConfirmationMismatch(JobStoreError): pass

@dataclass(frozen=True,slots=True)
class AuditConfirmationReceipt:
    schema:str
    audit_sha256:str
    resolved_event_hash:str
    next_store_hash:str
    next_mutation_count:int
    binding_sha256:str

CONFIRMATION_SCHEMA="operator_ui_job_mutation_confirmation_v1"

_INTENT_KEYS={"schema","audit_hash_binding","resolution_algorithm","operation","job_id","actor_identity","actor_level","job_operation","idempotency_key_sha256","input_identity_sha256","input","prior_state","prior_store_anchor","prior_event_hash","proposed_event","complete_proposal"}
_PROPOSAL_KEYS={"job_row","attempt_row","event_preimage","prior_rows","event_hash_derivation","next_store_mutation_count"}
_ROW_KEYS={
 "jobs":{"sequence","job_id","schema","actor_identity","actor_level","operation","idempotency_key_sha256","input_identity_sha256","input_json","created_at","creation_audit_hash"},
 "job_events":{"sequence","event_id","schema","job_id","phase","event_at","status","reason","facts_json","audit_hash","previous_event_hash","event_hash"},
 "job_attempts":{"sequence","attempt_id","schema","job_id","claimed_at","audit_hash"},
}

def _exact_mapping(value:Any,keys:set[str],name:str)->Mapping[str,Any]:
    if not isinstance(value,Mapping) or set(value)!=keys: raise ValueError(f"invalid {name} shape")
    return value

def _validate_confirmation_preimage(intent:Mapping[str,Any],proposal:Mapping[str,Any],preimage:Mapping[str,Any],rows:Mapping[str,Any])->None:
    operation=intent["operation"]
    if operation not in {"create","claim","transition","verify"}: raise ValueError("invalid mutation operation")
    _job_id(intent["job_id"]); _identifier(intent["actor_identity"],"actor"); _identifier(intent["job_operation"],"operation")
    if intent["actor_level"]!=2: raise ValueError("invalid mutation actor")
    _hash(intent["idempotency_key_sha256"],"idempotency"); _hash(intent["input_identity_sha256"],"input identity")
    input_fields=JobInput(**_exact_mapping(intent["input"],set(JobInput.__dataclass_fields__),"input"))
    if input_fields.identity_sha256!=intent["input_identity_sha256"]: raise ValueError("input identity mismatch")
    anchor=_exact_mapping(intent["prior_store_anchor"],{"singleton","schema","mutation_count","store_hash"},"prior anchor")
    if anchor["singleton"]!=1 or anchor["schema"]!=STORE_SCHEMA or isinstance(anchor["mutation_count"],bool) or not isinstance(anchor["mutation_count"],int) or anchor["mutation_count"]<0: raise ValueError("prior anchor invalid")
    _hash(anchor["store_hash"],"prior store hash"); _hash(intent["prior_event_hash"],"prior event hash")
    calculated_prior=ZERO_HASH if anchor["mutation_count"]==0 else _sha(canonical(rows))
    if not hmac.compare_digest(anchor["store_hash"],calculated_prior): raise ValueError("prior anchor hash mismatch")
    for table in _ROW_KEYS:
        if [row["sequence"] for row in rows[table]]!=list(range(1,len(rows[table])+1)): raise ValueError("prior row sequence invalid")
    proposed=_exact_mapping(intent["proposed_event"],{"schema","phase","event_at","status","reason","facts"},"proposed event")
    if proposed!={name:preimage[name] for name in ("schema","phase","event_at","status","reason","facts")} or preimage["job_id"]!=intent["job_id"] or preimage["previous_event_hash"]!=intent["prior_event_hash"]: raise ValueError("event proposal mismatch")
    if preimage["schema"]!=EVENT_SCHEMA or preimage["audit_hash"]!=AUDIT_HASH_BINDING: raise ValueError("event preimage invalid")
    _canonical_uuid(preimage["event_id"],"event_id"); utc_text(datetime.fromisoformat(preimage["event_at"].replace("Z","+00:00")))
    prior_events=[row for row in rows["job_events"] if row["job_id"]==intent["job_id"]]
    predecessor=Phase(prior_events[-1]["phase"]) if prior_events else None
    tail=prior_events[-1]["event_hash"] if prior_events else ZERO_HASH
    if tail!=intent["prior_event_hash"] or intent["prior_state"]!=(predecessor.value if predecessor else "NONE"): raise ValueError("prior event identity mismatch")
    _validate_facts(Phase(preimage["phase"]),dict(preimage["facts"]),predecessor,preimage["status"],preimage["reason"])
    if proposal["event_hash_derivation"]!="sha256(canonical_json(event_preimage with audit marker substituted once))" or proposal["next_store_mutation_count"]!=anchor["mutation_count"]+1: raise ValueError("proposal count or derivation invalid")
    expected_presence={"create":(True,False),"claim":(False,True),"transition":(False,False),"verify":(False,False)}[operation]
    if (proposal["job_row"] is not None,proposal["attempt_row"] is not None)!=expected_presence: raise ValueError("operation row presence invalid")
    if proposal["job_row"] is not None:
        row=_exact_mapping(proposal["job_row"],_ROW_KEYS["jobs"]-{"sequence"},"proposed job")
        if row["job_id"]!=intent["job_id"] or row["schema"]!=JOB_SCHEMA or row["creation_audit_hash"]!=AUDIT_HASH_BINDING or row["input_identity_sha256"]!=intent["input_identity_sha256"] or row["input_json"]!=canonical(intent["input"]).decode() or row["actor_identity"]!=intent["actor_identity"] or row["operation"]!=intent["job_operation"]: raise ValueError("proposed job mismatch")
        utc_text(datetime.fromisoformat(row["created_at"].replace("Z","+00:00")))
    if proposal["attempt_row"] is not None:
        row=_exact_mapping(proposal["attempt_row"],_ROW_KEYS["job_attempts"]-{"sequence"},"proposed attempt")
        if row["job_id"]!=intent["job_id"] or row["schema"]!=ATTEMPT_SCHEMA or row["audit_hash"]!=AUDIT_HASH_BINDING or row["claimed_at"]!=preimage["event_at"] or row["attempt_id"]!=preimage["facts"].get("attempt_id"): raise ValueError("proposed attempt mismatch")
        _canonical_uuid(row["attempt_id"],"attempt_id")
    if anchor["mutation_count"]!=len(rows["job_events"]) or proposal["next_store_mutation_count"]!=len(rows["job_events"])+1: raise ValueError("event count mismatch")
    if any(AUDIT_HASH_BINDING in canonical(row).decode() for table in rows.values() for row in table): raise ValueError("marker in prior rows")
    marker_paths=[]
    def locate(value:Any,path:tuple[str,...]=()):
        if isinstance(value,Mapping):
            for key,item in value.items(): locate(item,(*path,str(key)))
        elif isinstance(value,list):
            for index,item in enumerate(value): locate(item,(*path,str(index)))
        elif value==AUDIT_HASH_BINDING: marker_paths.append(path)
    locate(intent)
    allowed={("audit_hash_binding",),("complete_proposal","event_preimage","audit_hash")}
    if operation=="create": allowed.add(("complete_proposal","job_row","creation_audit_hash"))
    if operation=="claim": allowed.add(("complete_proposal","attempt_row","audit_hash"))
    if set(marker_paths)!=allowed: raise ValueError("audit marker placement invalid")

def resolve_audit_confirmation(intent:Mapping[str,Any],audit_sha256:str)->AuditConfirmationReceipt:
    """Independently resolve the exact frozen proposal presented to an auditor."""
    _hash(audit_sha256,"audit_sha256")
    _exact_mapping(intent,_INTENT_KEYS,"mutation intent")
    if intent.get("schema")!="operator_ui_job_mutation_intent_v3" or intent.get("audit_hash_binding")!=AUDIT_HASH_BINDING or intent.get("resolution_algorithm")!="operator_ui_job_mutation_resolution_v1": raise ValueError("invalid mutation intent")
    proposal=_exact_mapping(intent.get("complete_proposal"),_PROPOSAL_KEYS,"mutation proposal")
    preimage=proposal.get("event_preimage")
    rows=proposal.get("prior_rows")
    _exact_mapping(preimage,{"event_id","schema","job_id","phase","event_at","status","reason","facts","audit_hash","previous_event_hash"},"event preimage")
    _exact_mapping(rows,set(_ROW_KEYS),"prior rows")
    for table, expected in _ROW_KEYS.items():
        if not isinstance(rows[table],list): raise ValueError("invalid prior row collection")
        for row in rows[table]: _exact_mapping(row,expected,table+" row")
    _validate_confirmation_preimage(intent,proposal,preimage,rows)
    event=dict(preimage)
    if event.get("audit_hash")!=AUDIT_HASH_BINDING: raise ValueError("invalid audit marker")
    event["audit_hash"]=audit_sha256
    event_hash=_sha(canonical(event))
    job_row=proposal.get("job_row")
    attempt_row=proposal.get("attempt_row")
    resolved={name:[dict(row) for row in rows[name]] for name in ("jobs","job_events","job_attempts")}
    if job_row is not None:
        row=dict(job_row); row["creation_audit_hash"]=audit_sha256; row["sequence"]=len(resolved["jobs"])+1; resolved["jobs"].append(row)
    if attempt_row is not None:
        row=dict(attempt_row); row["audit_hash"]=audit_sha256; row["sequence"]=len(resolved["job_attempts"])+1; resolved["job_attempts"].append(row)
    facts_json=canonical(event.pop("facts")).decode()
    event_row={**event,"facts_json":facts_json,"event_hash":event_hash,"sequence":len(resolved["job_events"])+1}
    resolved["job_events"].append(event_row)
    next_hash=_sha(canonical(resolved)); count=proposal.get("next_store_mutation_count")
    binding=_sha(canonical({"schema":CONFIRMATION_SCHEMA,"audit_sha256":audit_sha256,"resolved_event_hash":event_hash,"next_store_hash":next_hash,"next_mutation_count":count}))
    return AuditConfirmationReceipt(CONFIRMATION_SCHEMA,audit_sha256,event_hash,next_hash,count,binding)

class Phase(str, Enum):
    SUBMITTED="SUBMITTED"; VALIDATED="VALIDATED"; WAITING_FOR_CLAIM="WAITING_FOR_CLAIM"
    CLAIMED="CLAIMED"; ATTEMPT_STARTED="ATTEMPT_STARTED"; RESPONSE_RECORDED="RESPONSE_RECORDED"
    RECEIPT_VERIFIED="RECEIPT_VERIFIED"; CONSUMED="CONSUMED"; SCORING="SCORING"
    PRODUCER_COMPLETED="PRODUCER_COMPLETED"; PREDICTION_READY="PREDICTION_READY"
    REAP_UNCONFIRMED="REAP_UNCONFIRMED"; FAILED="FAILED"; REJECTED="REJECTED"
    EXPIRED="EXPIRED"; TIMED_OUT="TIMED_OUT"; CANCELLED="CANCELLED"

TERMINAL_PHASES=frozenset({Phase.PREDICTION_READY,Phase.FAILED,Phase.REJECTED,Phase.EXPIRED,Phase.TIMED_OUT,Phase.CANCELLED})
_NEXT={
 Phase.SUBMITTED:{Phase.VALIDATED,Phase.REJECTED,Phase.EXPIRED},
 Phase.VALIDATED:{Phase.WAITING_FOR_CLAIM,Phase.REJECTED,Phase.EXPIRED},
 Phase.WAITING_FOR_CLAIM:{Phase.CLAIMED,Phase.FAILED,Phase.REJECTED,Phase.EXPIRED},
 Phase.CLAIMED:{Phase.ATTEMPT_STARTED,Phase.FAILED,Phase.REAP_UNCONFIRMED},
 Phase.ATTEMPT_STARTED:{Phase.RESPONSE_RECORDED,Phase.FAILED,Phase.TIMED_OUT,Phase.CANCELLED,Phase.REAP_UNCONFIRMED},
 Phase.RESPONSE_RECORDED:{Phase.RECEIPT_VERIFIED,Phase.CONSUMED,Phase.PRODUCER_COMPLETED,Phase.FAILED,Phase.REJECTED},
 Phase.RECEIPT_VERIFIED:{Phase.CONSUMED,Phase.FAILED,Phase.REJECTED},
 Phase.CONSUMED:{Phase.SCORING,Phase.PRODUCER_COMPLETED,Phase.FAILED,Phase.REJECTED},
 Phase.SCORING:{Phase.PRODUCER_COMPLETED,Phase.FAILED,Phase.REJECTED},
 Phase.PRODUCER_COMPLETED:{Phase.PREDICTION_READY,Phase.FAILED,Phase.REJECTED},
}

def utc_text(value:datetime)->str:
    if value.tzinfo is None or value.utcoffset() is None: raise ValueError("timestamp must be timezone aware")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00","Z")
def canonical(value:Any)->bytes:
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def _sha(value:bytes)->str:return hashlib.sha256(value).hexdigest()
def _hash(value:str,name:str)->str:
    if not isinstance(value,str) or len(value)!=64 or set(value)-set("0123456789abcdef"): raise ValueError(f"invalid {name}")
    return value
def _identifier(value:str,name:str,maximum:int=512)->str:
    if not isinstance(value,str) or not value or len(value.encode())>maximum or any(ord(c)<32 or ord(c)==127 for c in value): raise ValueError(f"invalid {name}")
    return value
def _job_id(value:str)->str:
    if not isinstance(value,str) or len(value)!=36 or not value.startswith("job_") or set(value[4:])-set("0123456789abcdef"): raise ValueError("invalid job_id")
    return value
def _canonical_uuid(value:Any,name:str)->str:
    if not isinstance(value,str): raise ValueError(f"invalid {name}")
    try: parsed=uuid.UUID(value)
    except (ValueError,AttributeError): raise ValueError(f"invalid {name}") from None
    if str(parsed)!=value: raise ValueError(f"invalid {name}")
    return value
def hash_idempotency_key(key:str)->str:
    if not isinstance(key,str) or not 16<=len(key.encode())<=256: raise ValueError("idempotency key must contain 16..256 UTF-8 bytes")
    return _sha(key.encode())

_PROCESS_FACTS=frozenset({"attempt_id","pid","exit_code","stdout_complete","stdout_reader_error","stdout_length","stdout_sha256","stdout_prefix_length","stdout_prefix_sha256","stdout_bytes","stderr_complete","stderr_reader_error","stderr_length","stderr_sha256","stderr_prefix_length","stderr_prefix_sha256","stderr_bytes","predictor_status","prediction_id","producer_job_id","producer_blocker","protocol_chain","authenticated_cutoff","error"})
_STREAM_REQUIRED=frozenset({"attempt_id","pid","exit_code","stdout_complete","stdout_prefix_length","stdout_prefix_sha256","stderr_complete","stderr_prefix_length","stderr_prefix_sha256"})
_STREAM_OPTIONAL=frozenset({"stdout_reader_error","stdout_bytes","stdout_length","stdout_sha256","stderr_reader_error","stderr_bytes","stderr_length","stderr_sha256"})
_VERIFIER_FACTS=frozenset({"prediction_id","job_id","race_id","jump_timestamp","runner_set_sha256","resolved_model_identity","model_sha256","model_manifest_sha256","model_schema_sha256","config_id","config_sha256","index_sha256","result_sha256","manifest_sha256","logical_bundle_sha256","bundle_locator","producer_status","research_only","production_persisted","betting_output","verification_status","blocker"})
_VERIFIER_FAILURE_FACTS=frozenset({"prediction_id","job_id","race_id","jump_timestamp","runner_set_sha256","resolved_model_identity","model_sha256","model_manifest_sha256","model_schema_sha256","config_id","config_sha256","producer_status","verification_status","blocker"})
_VERIFIER_FAILURE_CODES=frozenset({"BUNDLE_INDEX_VERIFICATION_FAILED","BUNDLE_IDENTITY_MISSING_OR_AMBIGUOUS","BUNDLE_VERIFICATION_FAILED","BUNDLE_JOB_IDENTITY_MISMATCH","BUNDLE_BLOCKER_IDENTITY_MISMATCH","BUNDLE_RESULT_IDENTITY_UNAVAILABLE"})

def _event_contracts():
    empty=frozenset(); claim=frozenset({"attempt_id"}); start=frozenset({"attempt_id","pid"})
    process=_STREAM_REQUIRED; producer=process|{"predictor_status","prediction_id","producer_job_id"}
    contracts={
      (None,Phase.SUBMITTED,"ACCEPTED","submitted"):(empty,empty,"empty"),
      (Phase.SUBMITTED,Phase.VALIDATED,"VALID","validated"):(empty,empty,"empty"),
      (Phase.VALIDATED,Phase.WAITING_FOR_CLAIM,"WAITING","ready"):(empty,empty,"empty"),
      (Phase.WAITING_FOR_CLAIM,Phase.FAILED,"FAILED","DISPATCH_FAILED"):(frozenset({"error"}),empty,"dispatch"),
      (Phase.WAITING_FOR_CLAIM,Phase.CLAIMED,"CLAIMED","unique_attempt_claimed"):(claim,empty,"claim"),
      (Phase.CLAIMED,Phase.ATTEMPT_STARTED,"RUNNING","predictor_started"):(start,empty,"start"),
      (Phase.CLAIMED,Phase.FAILED,"FAILED","PROCESS_LAUNCH_FAILED"):(frozenset({"attempt_id","error"}),empty,"prelaunch"),
      (Phase.ATTEMPT_STARTED,Phase.RESPONSE_RECORDED,"RECORDED","bounded_process_response"):(process,_STREAM_OPTIONAL|{"predictor_status","prediction_id","producer_job_id","producer_blocker","protocol_chain","authenticated_cutoff"},"process"),
      (Phase.RESPONSE_RECORDED,Phase.PRODUCER_COMPLETED,"PRODUCER_COMPLETED","PRODUCER_PREDICTION_READY"):(producer,_STREAM_OPTIONAL|{"protocol_chain","authenticated_cutoff"},"producer_ready"),
      (Phase.PRODUCER_COMPLETED,Phase.PREDICTION_READY,"READY","verified"):(_VERIFIER_FACTS,empty,"verifier_ready"),
      (Phase.PRODUCER_COMPLETED,Phase.FAILED,"FAILED","verification_failed"):(empty,_VERIFIER_FACTS|_VERIFIER_FAILURE_FACTS,"verifier_failed"),
      (Phase.PRODUCER_COMPLETED,Phase.REJECTED,"REJECTED","verification_rejected"):(_VERIFIER_FACTS,empty,"verifier_rejected"),
    }
    for predecessor in (Phase.CLAIMED,Phase.ATTEMPT_STARTED,Phase.RESPONSE_RECORDED):
      for target,reason in ((Phase.FAILED,"POST_SPAWN_FAILURE"),(Phase.FAILED,"POST_SPAWN_OSERROR"),(Phase.REAP_UNCONFIRMED,"PREDICTOR_REAP_UNCONFIRMED")):
        if target not in _NEXT.get(predecessor,set()): continue
        contracts[(predecessor,target,target.value,reason)]=(process,_STREAM_OPTIONAL,"process")
    for target,reason in ((Phase.TIMED_OUT,"PREDICTOR_TIMEOUT"),(Phase.CANCELLED,"PREDICTOR_CANCELLED"),(Phase.REAP_UNCONFIRMED,"PREDICTOR_REAP_UNCONFIRMED")):
      contracts[(Phase.ATTEMPT_STARTED,target,target.value,reason)]=(process,_STREAM_OPTIONAL,"process")
    for reason in ("PROCESS_OUTPUT_INVALID","PROCESS_OUTPUT_OVERSIZED","RECEIPT_READY_IS_NOT_PREDICTION_SUCCESS","PROCESS_EXIT_STATUS_MISMATCH"):
      contracts[(Phase.RESPONSE_RECORDED,Phase.FAILED,"FAILED",reason)]=(process,_STREAM_OPTIONAL|{"predictor_status","prediction_id","producer_job_id","producer_blocker"},"process")
    from src.predictor.on_demand import BLOCKER_STAGE_BY_CODE
    for code in BLOCKER_STAGE_BY_CODE:
      contracts[(Phase.RESPONSE_RECORDED,Phase.PRODUCER_COMPLETED,"PRODUCER_COMPLETED",f"PRODUCER_PREDICTION_BLOCKED:{code}")]=(process|{"predictor_status","prediction_id","producer_job_id","producer_blocker"},_STREAM_OPTIONAL|{"protocol_chain","authenticated_cutoff"},"producer_blocker")
    for code in set(BLOCKER_STAGE_BY_CODE)|{"BUSY"}:
      contracts[(Phase.RESPONSE_RECORDED,Phase.REJECTED,"REJECTED",f"PREDICTOR_BLOCKER:{code}")]=(process|{"predictor_status","producer_blocker"},_STREAM_OPTIONAL|{"prediction_id","producer_job_id"},"producer_blocker")
    return contracts

_EVENT_CONTRACTS=_event_contracts()

def _validate_facts(phase:Phase,facts:Any,prior:Phase|None,status:str="",reason:str="")->dict[str,Any]:
    if not isinstance(facts,dict): raise ValueError("facts must be an exact object")
    contract=_EVENT_CONTRACTS.get((prior,phase,status,reason))
    if contract is None: raise ValueError("undeclared event contract")
    required,optional,_kind=contract
    if not required.issubset(facts) or set(facts)-required-optional: raise ValueError("event facts do not match exact contract")
    verifier=set(_VERIFIER_FACTS); verifier_failure=set(_VERIFIER_FAILURE_FACTS)
    if prior is Phase.PRODUCER_COMPLETED and phase is Phase.FAILED and set(facts) not in {frozenset(verifier),frozenset(verifier_failure)}: raise ValueError("exact verifier failure evidence required")
    if "verification_status" in facts:
        if phase is Phase.FAILED and set(facts)==verifier_failure:
            blocker=facts.get("blocker")
            if facts.get("verification_status")!="FAILED" or facts.get("producer_status") not in {"PREDICTION_READY","PREDICTION_BLOCKED"} or not isinstance(blocker,dict) or set(blocker)!={"code","stage"} or blocker.get("code") not in _VERIFIER_FAILURE_CODES or blocker.get("stage")!="BUNDLE_VERIFICATION": raise ValueError("invalid bundle verification failure facts")
        elif set(facts)!=verifier: raise ValueError("exact verifier evidence required")
        if phase is Phase.PREDICTION_READY and (facts.get("verification_status")!="VERIFIED" or facts.get("producer_status")!="PREDICTION_READY" or facts.get("blocker") is not None): raise ValueError("invalid ready verification facts")
        if phase in {Phase.FAILED,Phase.REJECTED} and set(facts)==verifier:
            blocker=facts.get("blocker")
            expected="FAILED" if phase is Phase.FAILED else "REJECTED"
            if facts.get("verification_status")!=expected or facts.get("producer_status")!="PREDICTION_BLOCKED" or not isinstance(blocker,dict) or set(blocker)!={"code","stage"}: raise ValueError("invalid verifier blocker facts")
    if phase is Phase.CLAIMED and set(facts)!={"attempt_id"}: raise ValueError("claim facts incomplete")
    if phase is Phase.ATTEMPT_STARTED and set(facts)!={"attempt_id","pid"}: raise ValueError("start facts incomplete")
    if phase in {Phase.SUBMITTED,Phase.VALIDATED,Phase.WAITING_FOR_CLAIM} and facts: raise ValueError("phase facts must be empty")
    stream_required=set(_STREAM_REQUIRED)-{"stdout_reader_error","stderr_reader_error","stdout_bytes","stderr_bytes"}
    if phase is Phase.RESPONSE_RECORDED and not stream_required.issubset(facts): raise ValueError("response evidence incomplete")
    if phase is Phase.PRODUCER_COMPLETED and not (stream_required|{"predictor_status","prediction_id","producer_job_id"}).issubset(facts): raise ValueError("producer evidence incomplete")
    if phase in {Phase.TIMED_OUT,Phase.CANCELLED,Phase.REAP_UNCONFIRMED} and not {"attempt_id","pid","exit_code"}.issubset(facts): raise ValueError("lifetime evidence incomplete")
    if phase in {Phase.FAILED,Phase.REJECTED} and prior in {Phase.ATTEMPT_STARTED,Phase.RESPONSE_RECORDED} and not stream_required.issubset(facts): raise ValueError("postspawn terminal evidence incomplete")
    if phase is Phase.FAILED and prior is Phase.CLAIMED and set(facts) not in ({"attempt_id","error"},{"attempt_id","pid","exit_code"}) and not stream_required.issubset(facts): raise ValueError("claimed launch failure evidence invalid")
    if "attempt_id" in facts:
        _canonical_uuid(facts["attempt_id"],"attempt_id")
    if "prediction_id" in facts: _canonical_uuid(facts["prediction_id"],"prediction_id")
    if "predictor_status" in facts and facts["predictor_status"] not in {"PREDICTION_READY","PREDICTION_BLOCKED","RECEIPT_READY","BUSY"}: raise ValueError("invalid predictor status")
    if phase is Phase.PRODUCER_COMPLETED:
        if facts.get("predictor_status")=="PREDICTION_READY":
            if facts.get("producer_blocker") is not None or facts.get("exit_code")!=0 or facts.get("stdout_complete") is not True or facts.get("stderr_complete") is not True: raise ValueError("producer completion facts contradict readiness")
        elif facts.get("predictor_status")=="PREDICTION_BLOCKED":
            blocker=facts.get("producer_blocker")
            from src.predictor.on_demand import BLOCKER_STAGE_BY_CODE
            if not isinstance(blocker,dict) or set(blocker)!={"code","stage"} or BLOCKER_STAGE_BY_CODE.get(blocker["code"])!=blocker["stage"]: raise ValueError("producer blocker identity invalid")
        else: raise ValueError("producer completion status invalid")
    if "pid" in facts and (isinstance(facts["pid"],bool) or not isinstance(facts["pid"],int) or facts["pid"]<=0): raise ValueError("invalid predictor pid")
    if "exit_code" in facts and facts["exit_code"] is not None and (isinstance(facts["exit_code"],bool) or not isinstance(facts["exit_code"],int)): raise ValueError("invalid exit fact")
    for name in ("stdout_length","stderr_length"):
        if name in facts and (isinstance(facts[name],bool) or not isinstance(facts[name],int) or facts[name]<0): raise ValueError("invalid stream length")
    for name in ("stdout_sha256","stderr_sha256"):
        if name in facts: _hash(facts[name],name)
    for name in ("stderr_bytes","stdout_bytes"):
        if name in facts:
            if not isinstance(facts[name],str): raise ValueError("invalid stream bytes")
            bytes.fromhex(facts[name])
    for stream in ("stdout","stderr"):
        complete=facts.get(stream+"_complete")
        if complete is not None and not isinstance(complete,bool): raise ValueError("invalid stream completeness")
        reader_error=facts.get(stream+"_reader_error")
        if reader_error is not None and (not isinstance(reader_error,str) or reader_error not in {"NONE","READ_ERROR","START_ERROR","JOIN_ERROR","CLOSE_ERROR","INCOMPLETE"}): raise ValueError("invalid stream reader error")
        if complete is True and reader_error not in {None,"NONE"}: raise ValueError("complete stream cannot have reader error")
        prefix_name=stream+"_prefix_length"; prefix_hash=stream+"_prefix_sha256"; bytes_name=stream+"_bytes"
        if prefix_name in facts:
            if isinstance(facts[prefix_name],bool) or not isinstance(facts[prefix_name],int) or facts[prefix_name]<0: raise ValueError("invalid stream prefix length")
            _hash(facts[prefix_hash],prefix_hash)
            if bytes_name in facts:
                raw=bytes.fromhex(facts[bytes_name])
                if len(raw)!=facts[prefix_name] or _sha(raw)!=facts[prefix_hash]: raise ValueError("stream prefix evidence mismatch")
        if complete is True and (stream+"_length" not in facts or stream+"_sha256" not in facts): raise ValueError("complete stream evidence missing")
        if complete is False and (stream+"_length" in facts or stream+"_sha256" in facts): raise ValueError("incomplete stream fabricated as complete")
    return facts

@dataclass(frozen=True,slots=True)
class JobInput:
    race_id:str; jump_timestamp:str; runner_set_sha256:str; model_selector:str
    resolved_model_identity:str; model_sha256:str; model_manifest_sha256:str
    model_schema_sha256:str; config_id:str; config_sha256:str; odds_source:str
    ordered_runners:tuple[Mapping[str,Any],...]=()
    def __post_init__(self):
        if isinstance(self.ordered_runners,list): object.__setattr__(self,"ordered_runners",tuple(self.ordered_runners))
    def fields(self)->dict[str,Any]:
        values={n:getattr(self,n) for n in self.__dataclass_fields__}
        for n,v in values.items():
            if n == "ordered_runners": continue
            _hash(v,n) if n.endswith("sha256") else _identifier(v,n)
        if self.odds_source not in {"auto","receipt","capture"}: raise ValueError("invalid odds_source")
        if not isinstance(self.ordered_runners,tuple) or not self.ordered_runners: raise ValueError("ordered runners required")
        normalized=[]; identities=set()
        for runner in self.ordered_runners:
            if not isinstance(runner,Mapping) or set(runner) not in ({"box","name","identity"},{"box","name","identity","source_native_runner_id"}): raise ValueError("invalid ordered runner")
            row=dict(runner)
            if type(row["box"]) is not int or not 1<=row["box"]<=32: raise ValueError("invalid runner box")
            name=_identifier(row["name"],"runner name"); identity=_identifier(row["identity"],"runner identity")
            if identity in identities: raise ValueError("duplicate runner identity")
            normalized_row={"box":row["box"],"name":name,"identity":identity}
            if "source_native_runner_id" in row: normalized_row["source_native_runner_id"]=_identifier(row["source_native_runner_id"],"source native runner id")
            identities.add(identity); normalized.append(normalized_row)
        values["ordered_runners"]=normalized
        parsed=datetime.fromisoformat(self.jump_timestamp.replace("Z","+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None: raise ValueError("jump_timestamp must be timezone aware")
        return values
    @property
    def identity_sha256(self)->str:return _sha(canonical(self.fields()))

@dataclass(frozen=True,slots=True)
class Job:
    job_id:str; actor_identity:str; actor_level:int; operation:str; idempotency_key_sha256:str
    input:JobInput; created_at:str; phase:Phase; phase_at:str; status:str; reason:str
    evidence_bundle_ref:str|None; evidence_bundle_sha256:str|None; attempt_claimed:bool

AuditConfirmation=Callable[[Mapping[str,Any]],AuditConfirmationReceipt]

class JobStore:
    """Separate SQLite control truth; all mutations are audited then append-only."""
    _TRIGGERS={
      "jobs_no_update":"BEFORE UPDATE ON jobs", "jobs_no_delete":"BEFORE DELETE ON jobs",
      "events_no_update":"BEFORE UPDATE ON job_events", "events_no_delete":"BEFORE DELETE ON job_events",
      "attempts_no_update":"BEFORE UPDATE ON job_attempts", "attempts_no_delete":"BEFORE DELETE ON job_attempts",
      "anchor_no_delete":"BEFORE DELETE ON store_anchor",
    }
    def __init__(self,path:Path,*,separate_from:tuple[Path,...]=(),verifier_authority:object|None=None):
        self.path=Path(path).absolute(); self._separate_from=tuple(Path(p).absolute() for p in separate_from); self._identity=None; self._verifier_authority=verifier_authority; self._initialize()
    def _validate_separation(self,identity):
        for other in self._separate_from:
            try: st=other.stat(); oid=(st.st_dev,st.st_ino)
            except FileNotFoundError: oid=None
            if self.path.resolve(strict=False)==other.resolve(strict=False) or identity is not None and identity==oid: raise JobStoreError("job store must be separate from canonical and audit stores")
    def _validate_path(self):
        try: st=self.path.lstat()
        except OSError as exc: raise JobStoreError("job store path unavailable") from exc
        identity=(st.st_dev,st.st_ino)
        if not stat.S_ISREG(st.st_mode) or self.path.is_symlink() or identity!=self._identity: raise JobStoreError("job store path identity changed")
        self._validate_separation(identity)
    def _connect(self):
        if self._identity is not None:self._validate_path()
        db=sqlite3.connect(self.path,timeout=10,isolation_level=None); db.row_factory=sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON"); db.execute("PRAGMA busy_timeout=10000"); return db
    def _initialize(self):
        self.path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        try: st=self.path.lstat()
        except FileNotFoundError: st=None
        if st and (not stat.S_ISREG(st.st_mode) or self.path.is_symlink()): raise JobStoreError("job store must be a regular file")
        self._validate_separation(None if st is None else (st.st_dev,st.st_ino))
        existing=st is not None and st.st_size>0
        db=sqlite3.connect(self.path,isolation_level=None)
        try:
            if not existing: db.executescript(_SCHEMA_SQL)
        finally: db.close()
        os.chmod(self.path,0o600); st=self.path.lstat(); self._identity=(st.st_dev,st.st_ino); self._validate_path()
        if not self.verify(): raise JobStoreError("job store integrity invalid")
    def _rows_hash(self,db):
        payload={name:[dict(r) for r in db.execute(f"SELECT * FROM {name} ORDER BY sequence")] for name in ("jobs","job_events","job_attempts")}
        return _sha(canonical(payload))
    def _seal(self,db):
        row=db.execute("SELECT mutation_count FROM store_anchor WHERE singleton=1").fetchone()
        db.execute("UPDATE store_anchor SET mutation_count=?,store_hash=? WHERE singleton=1",(row[0]+1,self._rows_hash(db)))
    def _schema_valid(self,db):
        expected=sqlite3.connect(":memory:")
        try:
            expected.executescript(_SCHEMA_SQL)
            def master(connection):
                return [tuple(row) for row in connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name")]
            if master(db)!=master(expected): return False
            for table in ("store_anchor","jobs","job_events","job_attempts"):
                for pragma in ("table_info","index_list","foreign_key_list"):
                    if [tuple(r) for r in db.execute(f"PRAGMA {pragma}({table})")] != [tuple(r) for r in expected.execute(f"PRAGMA {pragma}({table})")]: return False
                for row in expected.execute(f"PRAGMA index_list({table})"):
                    name=row[1]
                    if [tuple(r) for r in db.execute(f"PRAGMA index_info('{name}')")] != [tuple(r) for r in expected.execute(f"PRAGMA index_info('{name}')")]: return False
            return True
        finally: expected.close()
    def _verify_db(self,db):
        if not self._schema_valid(db) or db.execute("PRAGMA foreign_key_check").fetchone(): return False
        anchor=db.execute("SELECT * FROM store_anchor").fetchall()
        if len(anchor)!=1 or anchor[0]["singleton"]!=1 or anchor[0]["schema"]!=STORE_SCHEMA or anchor[0]["store_hash"]!=(ZERO_HASH if anchor[0]["mutation_count"]==0 else self._rows_hash(db)): return False
        jobs=db.execute("SELECT * FROM jobs ORDER BY sequence").fetchall(); events=db.execute("SELECT * FROM job_events ORDER BY sequence").fetchall(); attempts=db.execute("SELECT * FROM job_attempts ORDER BY sequence").fetchall()
        for rows in (jobs,events,attempts):
            if [r["sequence"] for r in rows]!=list(range(1,len(rows)+1)): return False
        if anchor[0]["mutation_count"] != len(events): return False
        by_job={j["job_id"]:[] for j in jobs}; attempt_by={a["job_id"]:a for a in attempts}
        if len(by_job)!=len(jobs) or any(e["job_id"] not in by_job for e in events) or any(a["job_id"] not in by_job for a in attempts): return False
        for e in events: by_job[e["job_id"]].append(e)
        for j in jobs:
            try:
                if j["schema"]!=JOB_SCHEMA or j["actor_level"]!=2 or _hash(j["creation_audit_hash"],"audit")!=j["creation_audit_hash"]: return False
                data=json.loads(j["input_json"]); inp=JobInput(**data)
                if canonical(inp.fields()).decode()!=j["input_json"] or inp.identity_sha256!=j["input_identity_sha256"] or j["created_at"]!=utc_text(datetime.fromisoformat(j["created_at"].replace("Z","+00:00"))): return False
                _job_id(j["job_id"]); _identifier(j["actor_identity"],"actor"); _identifier(j["operation"],"operation"); _hash(j["idempotency_key_sha256"],"key")
                rows=by_job[j["job_id"]]; previous=ZERO_HASH; prior=None; claimed=[]
                if not rows or rows[0]["phase"]!=Phase.SUBMITTED.value or rows[0]["event_at"]!=j["created_at"] or rows[0]["audit_hash"]!=j["creation_audit_hash"]: return False
                for e in rows:
                    phase=Phase(e["phase"]); facts=_validate_facts(phase,json.loads(e["facts_json"]),None if prior is None else prior[0],e["status"],e["reason"]); _canonical_uuid(e["event_id"],"event_id"); _hash(e["audit_hash"],"audit")
                    if e["event_at"]!=utc_text(datetime.fromisoformat(e["event_at"].replace("Z","+00:00"))): return False
                    _identifier(e["status"],"status"); _identifier(e["reason"],"reason")
                    if canonical(facts).decode()!=e["facts_json"] or e["schema"]!=EVENT_SCHEMA: return False
                    fields={"event_id":e["event_id"],"schema":e["schema"],"job_id":e["job_id"],"phase":e["phase"],"event_at":e["event_at"],"status":e["status"],"reason":e["reason"],"facts":facts,"audit_hash":e["audit_hash"],"previous_event_hash":previous}
                    if e["previous_event_hash"]!=previous or not hmac.compare_digest(e["event_hash"],_sha(canonical(fields))): return False
                    if prior and (phase not in _NEXT.get(prior[0],set()) or e["event_at"]<prior[1] or prior[0] in TERMINAL_PHASES): return False
                    if phase is Phase.CLAIMED: claimed.append((e,facts))
                    previous=e["event_hash"]; prior=(phase,e["event_at"])
                attempt=attempt_by.get(j["job_id"])
                if (attempt is None)!=(not claimed) or len(claimed)>1: return False
                if attempt:
                    ce,cf=claimed[0]
                    if attempt["schema"]!=ATTEMPT_SCHEMA or attempt["attempt_id"]!=cf.get("attempt_id") or attempt["claimed_at"]!=ce["event_at"] or attempt["audit_hash"]!=ce["audit_hash"]: return False
                    _canonical_uuid(attempt["attempt_id"],"attempt_id")
                    if attempt["claimed_at"]!=utc_text(datetime.fromisoformat(attempt["claimed_at"].replace("Z","+00:00"))): return False
                    for event,facts in ((e,json.loads(e["facts_json"])) for e in rows if Phase(e["phase"]) in {Phase.ATTEMPT_STARTED,Phase.RESPONSE_RECORDED,Phase.PRODUCER_COMPLETED,Phase.REAP_UNCONFIRMED,Phase.TIMED_OUT,Phase.CANCELLED}):
                        if facts.get("attempt_id")!=attempt["attempt_id"]: return False
                    started=[json.loads(e["facts_json"]) for e in rows if e["phase"]==Phase.ATTEMPT_STARTED.value]
                    if len(started)>1 or started and (not isinstance(started[0].get("pid"),int) or started[0]["pid"]<=0): return False
                    if started:
                        pid=started[0]["pid"]
                        for e in rows:
                            facts=json.loads(e["facts_json"])
                            if e["phase"] in {p.value for p in (Phase.RESPONSE_RECORDED,Phase.PRODUCER_COMPLETED,Phase.REAP_UNCONFIRMED,Phase.TIMED_OUT,Phase.CANCELLED,Phase.FAILED,Phase.REJECTED)} and "pid" in facts and facts["pid"]!=pid:return False
            except (ValueError,TypeError,KeyError,json.JSONDecodeError): return False
        return True
    def verify(self):
        try:
            db=self._connect(); db.execute("BEGIN"); return self._verify_db(db)
        except (OSError,sqlite3.Error,JobStoreError): return False
        finally:
            if "db" in locals(): db.close()
    def _require_valid(self,db):
        if not self._verify_db(db): raise JobStoreError("job store integrity invalid")
    def _confirm(self,confirm:AuditConfirmation,intent:Mapping[str,Any]):
        try: value=confirm(intent)
        except Exception as exc: raise JobStoreError("operation audit confirmation failed") from exc
        if not isinstance(value,AuditConfirmationReceipt): raise ConfirmationMismatch("exact audit confirmation receipt required")
        try:
            expected=resolve_audit_confirmation(intent,value.audit_sha256)
        except (TypeError,ValueError) as exc: raise ConfirmationMismatch("invalid audit confirmation receipt") from exc
        for name in AuditConfirmationReceipt.__dataclass_fields__:
            actual=getattr(value,name); wanted=getattr(expected,name)
            if not isinstance(actual,type(wanted)) or (isinstance(actual,str) and not hmac.compare_digest(actual,wanted)) or (not isinstance(actual,str) and actual!=wanted):
                raise ConfirmationMismatch("audit confirmation receipt mismatch")
        return value
    def _mutation_intent(self,db,job,operation,prior,phase,at,status,reason,facts,*,proposed_job=None,proposed_attempt=None):
        anchor=dict(db.execute("SELECT * FROM store_anchor WHERE singleton=1").fetchone())
        tail=db.execute("SELECT event_hash FROM job_events WHERE job_id=? ORDER BY sequence DESC LIMIT 1",(job["job_id"],)).fetchone(); previous=tail[0] if tail else ZERO_HASH
        event_id=str(uuid.uuid4())
        event={"event_id":event_id,"schema":EVENT_SCHEMA,"job_id":job["job_id"],"phase":phase.value,"event_at":at,"status":status,"reason":reason,"facts":dict(facts),"audit_hash":AUDIT_HASH_BINDING,"previous_event_hash":previous}
        rows={name:[dict(r) for r in db.execute(f"SELECT * FROM {name} ORDER BY sequence")] for name in ("jobs","job_events","job_attempts")}
        intent={"schema":"operator_ui_job_mutation_intent_v3","audit_hash_binding":AUDIT_HASH_BINDING,"resolution_algorithm":"operator_ui_job_mutation_resolution_v1","operation":operation,"job_id":job["job_id"],"actor_identity":job["actor_identity"],"actor_level":job["actor_level"],"job_operation":job["operation"],"idempotency_key_sha256":job["idempotency_key_sha256"],"input_identity_sha256":job["input_identity_sha256"],"input":json.loads(job["input_json"]),"prior_state":prior.value if isinstance(prior,Phase) else prior,"prior_store_anchor":anchor,"prior_event_hash":previous,"proposed_event":{"schema":EVENT_SCHEMA,"phase":phase.value,"event_at":at,"status":status,"reason":reason,"facts":dict(facts)},"complete_proposal":{"job_row":proposed_job,"attempt_row":proposed_attempt,"event_preimage":event,"prior_rows":rows,"event_hash_derivation":"sha256(canonical_json(event_preimage with audit marker substituted once))","next_store_mutation_count":anchor["mutation_count"]+1}}
        return intent,event_id,previous
    def _append_event(self,db,job_id,phase,at,status,reason,facts,audit_hash,*,prior,event_id=None,previous=None):
        facts=_validate_facts(phase,dict(facts),prior,status,reason)
        if previous is None:
            prior=db.execute("SELECT event_hash FROM job_events WHERE job_id=? ORDER BY sequence DESC LIMIT 1",(job_id,)).fetchone(); previous=prior[0] if prior else ZERO_HASH
        event_id=event_id or str(uuid.uuid4()); fields={"event_id":event_id,"schema":EVENT_SCHEMA,"job_id":job_id,"phase":phase.value,"event_at":at,"status":status,"reason":reason,"facts":dict(facts),"audit_hash":audit_hash,"previous_event_hash":previous}; event_hash=_sha(canonical(fields))
        db.execute("INSERT INTO job_events(event_id,schema,job_id,phase,event_at,status,reason,facts_json,audit_hash,previous_event_hash,event_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?)",(event_id,EVENT_SCHEMA,job_id,phase.value,at,status,reason,canonical(dict(facts)).decode(),audit_hash,previous,event_hash))
    def create(self,*,actor_identity,actor_level,operation,idempotency_key,job_input,now,confirm_audit:AuditConfirmation):
        actor=_identifier(actor_identity,"actor_identity"); operation=_identifier(operation,"operation")
        if actor_level!=2: raise ValueError("manual prediction requires exact Level 2 authority")
        key_hash=hash_idempotency_key(idempotency_key); input_json=canonical(job_input.fields()).decode(); identity=job_input.identity_sha256; created=utc_text(now)
        db=self._connect()
        try:
            db.execute("BEGIN IMMEDIATE"); self._require_valid(db)
            row=db.execute("SELECT * FROM jobs WHERE actor_identity=? AND operation=? AND idempotency_key_sha256=?",(actor,operation,key_hash)).fetchone()
            if row:
                if not hmac.compare_digest(row["input_identity_sha256"],identity): raise IdempotencyConflict("idempotency key is already bound to different inputs")
                db.commit(); return self.get(row["job_id"])
            job_id="job_"+uuid.uuid4().hex; proposed={"job_id":job_id,"actor_identity":actor,"actor_level":2,"operation":operation,"idempotency_key_sha256":key_hash,"input_identity_sha256":identity,"input_json":input_json}; job_row={"job_id":job_id,"schema":JOB_SCHEMA,"actor_identity":actor,"actor_level":2,"operation":operation,"idempotency_key_sha256":key_hash,"input_identity_sha256":identity,"input_json":input_json,"created_at":created,"creation_audit_hash":AUDIT_HASH_BINDING}; intent,event_id,previous=self._mutation_intent(db,proposed,"create","NONE",Phase.SUBMITTED,created,"ACCEPTED","submitted",{},proposed_job=job_row); receipt=self._confirm(confirm_audit,intent); audit=receipt.audit_sha256
            db.execute("INSERT INTO jobs(job_id,schema,actor_identity,actor_level,operation,idempotency_key_sha256,input_identity_sha256,input_json,created_at,creation_audit_hash) VALUES(?,?,?,?,?,?,?,?,?,?)",(job_id,JOB_SCHEMA,actor,2,operation,key_hash,identity,input_json,created,audit))
            self._append_event(db,job_id,Phase.SUBMITTED,created,"ACCEPTED","submitted",{},audit,prior=None,event_id=event_id,previous=previous); self._seal(db)
            anchor=db.execute("SELECT mutation_count,store_hash FROM store_anchor WHERE singleton=1").fetchone()
            if anchor[0]!=receipt.next_mutation_count or not hmac.compare_digest(anchor[1],receipt.next_store_hash): raise ConfirmationMismatch("persisted store differs from confirmation")
            db.commit(); return self.get(job_id)
        except (IdempotencyConflict,JobStoreError): db.rollback(); raise
        except (sqlite3.Error,ValueError,TypeError) as exc: db.rollback(); raise JobStoreError("job creation failed") from exc
        finally: db.close()
    def _current(self,db,job_id):
        row=db.execute("SELECT phase,event_at FROM job_events WHERE job_id=? ORDER BY sequence DESC LIMIT 1",(job_id,)).fetchone()
        if not row: raise JobStoreError("unknown job")
        return Phase(row["phase"]),row["event_at"]
    def transition(self,job_id,phase,*,now,status,reason,facts=None,confirm_audit:AuditConfirmation):
        if phase in {Phase.CLAIMED,Phase.PREDICTION_READY} or phase in {Phase.FAILED,Phase.REJECTED} and self.get(job_id).phase is Phase.PRODUCER_COMPLETED: raise IllegalTransition("transition is reserved for its owning capability")
        at=utc_text(now); status=_identifier(status,"status"); reason=_identifier(reason,"reason"); db=self._connect()
        try:
            db.execute("BEGIN IMMEDIATE"); self._require_valid(db); current,current_at=self._current(db,job_id)
            if phase is Phase.ATTEMPT_STARTED and not db.execute("SELECT 1 FROM job_attempts WHERE job_id=?",(job_id,)).fetchone(): raise IllegalTransition("attempt must be durably claimed before start")
            if phase not in _NEXT.get(current,set()) or at<current_at: raise IllegalTransition(f"illegal transition {current.value}->{phase.value}")
            event_facts=_validate_facts(phase,dict(facts or {}),current,status,reason)
            if "producer_job_id" in event_facts and event_facts["producer_job_id"]!=job_id: raise ValueError("producer job identity mismatch")
            job=db.execute("SELECT * FROM jobs WHERE job_id=?",(job_id,)).fetchone(); intent,event_id,previous=self._mutation_intent(db,job,"transition",current,phase,at,status,reason,event_facts); receipt=self._confirm(confirm_audit,intent); audit=receipt.audit_sha256
            self._append_event(db,job_id,phase,at,status,reason,event_facts,audit,prior=current,event_id=event_id,previous=previous); self._seal(db); anchor=db.execute("SELECT mutation_count,store_hash FROM store_anchor WHERE singleton=1").fetchone()
            if anchor[0]!=receipt.next_mutation_count or not hmac.compare_digest(anchor[1],receipt.next_store_hash): raise ConfirmationMismatch("persisted store differs from confirmation")
            db.commit(); return self.get(job_id)
        except (IllegalTransition,JobStoreError): db.rollback(); raise
        except (sqlite3.Error,ValueError,TypeError) as exc: db.rollback(); raise JobStoreError("transition failed") from exc
        finally: db.close()
    def verifier_transition(self,job_id,phase,*,capability:object,now,status,reason,facts,confirm_audit:AuditConfirmation):
        if self._verifier_authority is None or capability is not self._verifier_authority: raise VerifierAuthorizationError("verifier authority required")
        if phase not in {Phase.PREDICTION_READY,Phase.FAILED,Phase.REJECTED}: raise IllegalTransition("invalid verifier transition")
        allowed={_VERIFIER_FACTS,_VERIFIER_FAILURE_FACTS}
        if not isinstance(facts,dict) or frozenset(facts) not in allowed: raise ValueError("exact verifier evidence required")
        for name in ("runner_set_sha256","model_sha256","model_manifest_sha256","model_schema_sha256","config_sha256"): _hash(facts[name],name)
        for name in ("index_sha256","result_sha256","manifest_sha256","logical_bundle_sha256"):
            if name in facts: _hash(facts[name],name)
        _canonical_uuid(facts["prediction_id"],"prediction_id"); _identifier(facts["verification_status"],"verification_status")
        if "bundle_locator" in facts:
            locator=facts["bundle_locator"]
            if not isinstance(locator,str) or not locator or Path(locator).is_absolute() or "\\" in locator or any(part in {"",".",".."} for part in locator.split("/")): raise ValueError("unsafe bundle locator")
            for flag in ("research_only","production_persisted","betting_output"):
                if not isinstance(facts[flag],bool): raise ValueError("invalid verifier safety flag")
        if phase is Phase.PREDICTION_READY and (facts["verification_status"]!="VERIFIED" or facts["producer_status"]!="PREDICTION_READY" or facts["research_only"] is not True or facts["production_persisted"] is not False or facts["betting_output"] is not False or facts["blocker"] is not None): raise ValueError("readiness requires verified safe sealed evidence")
        if phase in {Phase.FAILED,Phase.REJECTED} and frozenset(facts)==_VERIFIER_FACTS:
            expected_status="FAILED" if phase is Phase.FAILED else "REJECTED"
            blocker=facts["blocker"]
            if facts["verification_status"]!=expected_status or facts["producer_status"]!="PREDICTION_BLOCKED" or facts["research_only"] is not True or facts["production_persisted"] is not False or facts["betting_output"] is not False: raise ValueError("verifier blocker contradicts target phase")
            if not isinstance(blocker,dict) or set(blocker)!={"code","stage"}: raise ValueError("exact verifier blocker required")
            from src.predictor.on_demand import BLOCKER_STAGE_BY_CODE
            if BLOCKER_STAGE_BY_CODE.get(blocker["code"])!=blocker["stage"]: raise ValueError("invalid verifier blocker vocabulary")
        at=utc_text(now); status=_identifier(status,"status"); reason=_identifier(reason,"reason"); db=self._connect()
        try:
            db.execute("BEGIN IMMEDIATE"); self._require_valid(db); current,current_at=self._current(db,job_id)
            if current is not Phase.PRODUCER_COMPLETED or at<current_at: raise IllegalTransition("job is not awaiting verifier")
            job=db.execute("SELECT * FROM jobs WHERE job_id=?",(job_id,)).fetchone()
            inp=JobInput(**json.loads(job["input_json"]))
            expected=(job_id,inp.race_id,inp.jump_timestamp,inp.runner_set_sha256,inp.resolved_model_identity,inp.model_sha256,inp.model_manifest_sha256,inp.model_schema_sha256,inp.config_id,inp.config_sha256)
            actual=tuple(facts[n] for n in ("job_id","race_id","jump_timestamp","runner_set_sha256","resolved_model_identity","model_sha256","model_manifest_sha256","model_schema_sha256","config_id","config_sha256"))
            if actual!=expected: raise ValueError("verifier evidence identity mismatch")
            producer=json.loads(db.execute("SELECT facts_json FROM job_events WHERE job_id=? AND phase=? ORDER BY sequence DESC LIMIT 1",(job_id,Phase.PRODUCER_COMPLETED.value)).fetchone()[0])
            if producer.get("prediction_id")!=facts["prediction_id"] or producer.get("predictor_status")!=facts["producer_status"]: raise ValueError("verifier differs from producer identity")
            if phase is Phase.PREDICTION_READY:
                if producer.get("predictor_status")!="PREDICTION_READY" or producer.get("producer_blocker") is not None: raise ValueError("verifier readiness contradicts producer")
            elif phase is Phase.REJECTED and (producer.get("predictor_status")!="PREDICTION_BLOCKED" or producer.get("producer_blocker")!=facts["blocker"]): raise ValueError("verifier blocker differs from producer")
            intent,event_id,previous=self._mutation_intent(db,job,"verify",current,phase,at,status,reason,facts)
            receipt=self._confirm(confirm_audit,intent); audit=receipt.audit_sha256; self._append_event(db,job_id,phase,at,status,reason,facts,audit,prior=current,event_id=event_id,previous=previous); self._seal(db); anchor=db.execute("SELECT mutation_count,store_hash FROM store_anchor WHERE singleton=1").fetchone()
            if anchor[0]!=receipt.next_mutation_count or not hmac.compare_digest(anchor[1],receipt.next_store_hash): raise ConfirmationMismatch("persisted store differs from confirmation")
            db.commit(); return self.get(job_id)
        except (IllegalTransition,JobStoreError): db.rollback(); raise
        except (sqlite3.Error,ValueError,TypeError) as exc: db.rollback(); raise JobStoreError("verifier transition failed") from exc
        finally: db.close()
    def claim_attempt(self,job_id,*,now,confirm_audit:AuditConfirmation):
        at=utc_text(now); db=self._connect()
        try:
            db.execute("BEGIN IMMEDIATE"); self._require_valid(db); current,current_at=self._current(db,job_id)
            existing=db.execute("SELECT attempt_id FROM job_attempts WHERE job_id=?",(job_id,)).fetchone()
            if existing: raise AttemptAlreadyClaimed(existing[0])
            if current is not Phase.WAITING_FOR_CLAIM or at<current_at: raise IllegalTransition("job is not claimable")
            attempt=str(uuid.uuid4()); facts={"attempt_id":attempt}; job=db.execute("SELECT * FROM jobs WHERE job_id=?",(job_id,)).fetchone(); attempt_row={"attempt_id":attempt,"schema":ATTEMPT_SCHEMA,"job_id":job_id,"claimed_at":at,"audit_hash":AUDIT_HASH_BINDING}; intent,event_id,previous=self._mutation_intent(db,job,"claim",current,Phase.CLAIMED,at,"CLAIMED","unique_attempt_claimed",facts,proposed_attempt=attempt_row); receipt=self._confirm(confirm_audit,intent); audit=receipt.audit_sha256
            db.execute("INSERT INTO job_attempts(attempt_id,schema,job_id,claimed_at,audit_hash) VALUES(?,?,?,?,?)",(attempt,ATTEMPT_SCHEMA,job_id,at,audit)); self._append_event(db,job_id,Phase.CLAIMED,at,"CLAIMED","unique_attempt_claimed",{"attempt_id":attempt},audit,prior=current,event_id=event_id,previous=previous); self._seal(db); anchor=db.execute("SELECT mutation_count,store_hash FROM store_anchor WHERE singleton=1").fetchone()
            if anchor[0]!=receipt.next_mutation_count or not hmac.compare_digest(anchor[1],receipt.next_store_hash): raise ConfirmationMismatch("persisted store differs from confirmation")
            db.commit(); return self.get(job_id),attempt
        except (AttemptAlreadyClaimed,IllegalTransition,JobStoreError): db.rollback(); raise
        except (sqlite3.Error,ValueError,TypeError) as exc: db.rollback(); raise JobStoreError("attempt claim failed") from exc
        finally: db.close()
    def get(self,job_id):
        db=self._connect()
        try:
            db.execute("BEGIN"); self._require_valid(db); row=db.execute("SELECT * FROM jobs WHERE job_id=?",(job_id,)).fetchone(); event=db.execute("SELECT * FROM job_events WHERE job_id=? ORDER BY sequence DESC LIMIT 1",(job_id,)).fetchone(); attempt=db.execute("SELECT 1 FROM job_attempts WHERE job_id=?",(job_id,)).fetchone()
            if row is None or event is None: raise JobStoreError("unknown job")
            facts=json.loads(event["facts_json"])
            return Job(row["job_id"],row["actor_identity"],row["actor_level"],row["operation"],row["idempotency_key_sha256"],JobInput(**json.loads(row["input_json"])),row["created_at"],Phase(event["phase"]),event["event_at"],event["status"],event["reason"],facts.get("evidence_bundle_ref"),facts.get("evidence_bundle_sha256"),attempt is not None)
        finally: db.close()
    def events(self,job_id):
        """Return the persisted finite timeline for an existing job."""
        _job_id(job_id); db=self._connect()
        try:
            db.execute("BEGIN"); self._require_valid(db)
            if db.execute("SELECT 1 FROM jobs WHERE job_id=?",(job_id,)).fetchone() is None: raise JobStoreError("unknown job")
            rows=db.execute("SELECT event_id,phase,event_at,status,reason,facts_json,event_hash FROM job_events WHERE job_id=? ORDER BY sequence",(job_id,)).fetchall()
            return tuple({"event_id":row["event_id"],"phase":row["phase"],"event_at":row["event_at"],"status":row["status"],"reason":row["reason"],"facts":json.loads(row["facts_json"]),"event_hash":row["event_hash"]} for row in rows)
        finally: db.close()
