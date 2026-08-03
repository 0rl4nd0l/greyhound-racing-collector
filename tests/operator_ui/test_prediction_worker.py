from __future__ import annotations
import hashlib,io,json,sqlite3,subprocess,threading,time
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
import pytest
from race_collection.synchronous_manual_capture import VerifiedCurrentRaceIndex
from scripts.refresh_prejump_upcoming import stable_race_id
from src.predictor.on_demand import canonical_bytes
from src.operator_ui.job_store import JobInput,JobStore,JobStoreError,Phase,resolve_audit_confirmation
from src.operator_ui.prediction_worker import MAX_STDERR_BYTES,MAX_STDOUT_BYTES,ServerChoice,WorkerConfig,WorkerRejected,drain_bounded,fixed_argv,revalidate_current_race,run_once

NOW=datetime(2026,8,1,tzinfo=timezone.utc); H=hashlib.sha256(b"x").hexdigest(); AUDIT=hashlib.sha256(b"audit").hexdigest(); CONFIRM=lambda intent:resolve_audit_confirmation(intent,AUDIT)
RACE={"race_number":5,"venue":"RICH","race_date":"2026-08-01","url":"https://www.thedogs.com.au/racing/richmond/2026-08-01/5"}; RACE_ID=stable_race_id(RACE)

def setup(tmp_path):
    paths=[]
    for name in ("config.json","model.json","manifest.json","schema.json"):
        path=tmp_path/name; path.write_bytes(b"x"); paths.append(path)
    choice=ServerChoice(paths[0],"manual-default",H,"market_form_residual_v1",H,H,H,*paths[1:])
    python=Path("/tmp/ghu010-validation-73f1e5d/bin/python")
    if not python.is_file(): python=Path("/usr/bin/python3")
    cfg=WorkerConfig(python,Path(__file__).parents[2],{"latest-research":choice},tmp_path/"canonical.db",tmp_path/"output",(tmp_path/"evidence-a",tmp_path/"evidence-b"),tmp_path/"requests",tmp_path/"index.json",tmp_path/"evidence",1,45.0,90.0,2)
    value=JobStore(tmp_path/"jobs.db",separate_from=(tmp_path/"canonical.db",tmp_path/"audit.db"))
    inp=JobInput(RACE_ID,"2026-08-01T01:00:00+00:00",H,"latest-research","market_form_residual_v1",H,H,H,"manual-default",H,"auto",({"box":1,"name":"ALPHA","identity":"ALPHA"},))
    job=value.create(actor_identity="op",actor_level=2,operation="manual_prediction",idempotency_key="idempotency-key-1234",job_input=inp,now=NOW,confirm_audit=CONFIRM)
    job=value.transition(job.job_id,Phase.VALIDATED,now=NOW,status="VALID",reason="validated",confirm_audit=CONFIRM)
    job=value.transition(job.job_id,Phase.WAITING_FOR_CLAIM,now=NOW,status="WAITING",reason="ready",confirm_audit=CONFIRM)
    return cfg,value,job

def view(races=None):
    return VerifiedCurrentRaceIndex("collector_current_race_index_v2","run","2026-08-01T00:00:00Z",H,b"{}",tuple(races if races is not None else [{"race_id":RACE_ID,"jump_datetime":"2026-08-01T01:00:00+00:00","runner_set_sha256":H}]),"source.json",H,H,H,H)

def ready(job):
    rows=[{"rank":1,"box_number":1,"dog_name":"ALPHA","identity":"ALPHA","source_native_runner_id":"dog-1","probability":0.6},{"rank":2,"box_number":2,"dog_name":"BETA","identity":"BETA","source_native_runner_id":"dog-2","probability":0.4}]
    inp=job.input
    return canonical_bytes({"schema_version":"on_demand_race_prediction_v2","prediction_id":"12345678-1234-4123-8123-123456789abc","job_id":job.job_id,"generated_at":"2026-08-01T00:00:01+00:00","status":"PREDICTION_READY","blocker_stage":None,"blocker":None,"research_only":True,"production_persisted":False,"betting_output":False,"race":{**RACE,"venue_slug":"richmond","race_id":inp.race_id,"jump_timestamp":inp.jump_timestamp},"model":{"requested":inp.model_selector,"resolved":inp.resolved_model_identity,"alias_resolved":True,"schema_sha256":inp.model_schema_sha256,"artifact_identity":"AVAILABLE","artifact_sha256":inp.model_sha256,"artifact_manifest_identity":"AVAILABLE","artifact_manifest_sha256":inp.model_manifest_sha256},"config":{"sha256":inp.config_sha256},"evidence":{"request":"request.json","config":"config.json","model_schema":"model/config.schema.json","model_artifact":"model/model.json","model_manifest":"model/manifest.json","runner_set_sha256":inp.runner_set_sha256,"prediction_output_sha256":hashlib.sha256(canonical_bytes(rows)).hexdigest(),"protocol_chain":{"request_id":"request-1","request_sha256":H,"claim_sha256":H,"attempt_sha256":H,"response_sha256":H,"receipt_sha256":H,"consume_sha256":H,"authenticated_receipt_sha256":H},"authenticated_cutoff":{"history_seal_sha256":H,"cutoff_timestamp":inp.jump_timestamp,"source_sha256":H,"sealed_sha256":H}},"prediction":{"predictions":rows}})

def legacy_blocked(code="BUSY"):
    return canonical_bytes({"schema_version":"on_demand_race_prediction_v1","status":code,"research_only":True,"production_persisted":False,"blockers":[{"code":code}]})

def sealed_blocked(job,code="POST_JUMP"):
    value=json.loads(ready(job)); value.update(status="PREDICTION_BLOCKED",blocker_stage="VALIDATION",blocker={"code":code},prediction=None); value["evidence"]["prediction_output_sha256"]=None
    return canonical_bytes(value)

class Process:
    def __init__(self,stdout=b"",stderr=b"",code=0,wait_errors=()):
        self.pid=123; self.stdout=io.BytesIO(stdout); self.stderr=io.BytesIO(stderr); self.returncode=code; self.wait_errors=list(wait_errors); self.terminated=False; self.running=bool(wait_errors)
    def wait(self,timeout):
        if self.wait_errors: raise self.wait_errors.pop(0)
        self.running=False
        return self.returncode
    def terminate(self):self.terminated=True
    def poll(self):return None if self.running else self.returncode

def test_exact_argv_and_forbidden_surface(tmp_path):
    cfg,_,job=setup(tmp_path); argv=fixed_argv(job,cfg)
    assert argv[0:6]==(str(cfg.pinned_python),str(cfg.script),"--race-id",RACE_ID,"--model","latest-research")
    assert {"--race","--race-url","--replay-bundle","--list-configs","--current-time","--lock-path"}.isdisjoint(argv)

def test_launch_uses_only_retained_runtime_descriptors(tmp_path):
    cfg,store,job=setup(tmp_path); calls=[]
    def popen(argv,**kwargs):
        calls.append((argv,kwargs))
        assert argv[0].startswith("/proc/self/fd/") and argv[1].startswith("/proc/self/fd/")
        assert kwargs["executable"]==argv[0]
        assert tuple(sorted(kwargs["pass_fds"]))==tuple(sorted((int(argv[0].rsplit("/",1)[1]),int(argv[1].rsplit("/",1)[1]))))
        return Process(ready(job))
    assert run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=popen,reader=lambda **_:view()).phase is Phase.PRODUCER_COMPLETED
    assert len(calls)==1

@pytest.mark.parametrize("races,reason",[([],"RACE_ID_MISSING_OR_AMBIGUOUS"),([{"race_id":RACE_ID,"jump_datetime":"bad","runner_set_sha256":H}],"RACE_JUMP_CHANGED"),([{"race_id":RACE_ID,"jump_datetime":"2026-08-01T01:00:00+00:00","runner_set_sha256":"0"*64}],"RUNNER_SET_CHANGED")])
def test_revalidation_missing_changed(tmp_path,races,reason):
    cfg,_,job=setup(tmp_path)
    with pytest.raises(WorkerRejected,match=reason):revalidate_current_race(job,cfg,now=NOW,reader=lambda **_:view(races))

def test_identity_change_before_claim_prevents_launch(tmp_path):
    cfg,store,job=setup(tmp_path); cfg.choices["latest-research"].model_path.write_bytes(b"changed"); calls=[]
    with pytest.raises(WorkerRejected,match="BYTES_CHANGED"):run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=lambda *a,**k:calls.append(1),reader=lambda **_:view())
    assert not store.get(job.job_id).attempt_claimed and not calls

def test_valid_launch_is_nonterminal_producer_completion_and_restart_never_launches(tmp_path):
    cfg,store,job=setup(tmp_path); calls=[]
    def popen(argv,**kwargs):calls.append((argv,kwargs));return Process(ready(job))
    finished=run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=popen,reader=lambda **_:view())
    assert finished.phase is Phase.PRODUCER_COMPLETED and finished.evidence_bundle_ref is None and len(calls)==1
    assert calls[0][1]["shell"] is False and calls[0][1]["start_new_session"] is False
    with pytest.raises(WorkerRejected):run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=popen,reader=lambda **_:view())
    assert len(calls)==1

@pytest.mark.parametrize("stdout,code,phase,reason",[(legacy_blocked(),2,Phase.FAILED,"PROCESS_OUTPUT_INVALID"),(b"not-json",2,Phase.FAILED,"PROCESS_OUTPUT_INVALID"),(canonical_bytes({"status":"BUSY"})+b"\n",2,Phase.FAILED,"PROCESS_OUTPUT_INVALID"),(legacy_blocked("RECEIPT_READY"),0,Phase.FAILED,"PROCESS_OUTPUT_INVALID")])
def test_authoritative_framing_and_mapping(tmp_path,stdout,code,phase,reason):
    cfg,store,job=setup(tmp_path); result=run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=lambda *a,**k:Process(stdout,code=code),reader=lambda **_:view())
    assert result.phase is phase and result.reason==reason

@pytest.mark.parametrize("stdout,stderr",[(b"x"*(MAX_STDOUT_BYTES+1),b""),(legacy_blocked(),b"x"*(MAX_STDERR_BYTES+1))])
def test_output_caps_classify_without_deadlock(tmp_path,stdout,stderr):
    cfg,store,job=setup(tmp_path); result=run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=lambda *a,**k:Process(stdout,stderr,2),reader=lambda **_:view())
    assert result.reason=="PROCESS_OUTPUT_OVERSIZED"

def test_concurrent_pipe_drain_keeps_exact_lengths_and_hashes():
    proc=Process(b"a"*100000,b"b"*100000)
    out,out_len,out_hash,err,err_len,err_hash=drain_bounded(proc,timeout=2)
    assert out_len==err_len==100000 and out_hash==hashlib.sha256(b"a"*100000).hexdigest() and err_hash==hashlib.sha256(b"b"*100000).hexdigest()

@pytest.mark.parametrize("confirmed,phase",[(True,Phase.TIMED_OUT),(False,Phase.REAP_UNCONFIRMED)])
def test_timeout_reap_truth_is_durable_and_never_retries(tmp_path,confirmed,phase):
    cfg,store,job=setup(tmp_path); timeout=subprocess.TimeoutExpired("predictor",90); proc=Process(wait_errors=(timeout,) if confirmed else (timeout,subprocess.TimeoutExpired("predictor",2))); calls=[]
    result=run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=lambda *a,**k:(calls.append(1) or proc),reader=lambda **_:view())
    assert result.phase is phase and proc.terminated and len(calls)==1
    with pytest.raises(WorkerRejected):run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=lambda *a,**k:calls.append(1),reader=lambda **_:view())
    assert len(calls)==1

def test_blocking_pipe_close_cannot_exceed_shared_cleanup_deadline(tmp_path):
    cfg,store,job=setup(tmp_path); cfg=replace(cfg,cancellation_grace_seconds=.02)
    release=threading.Event()
    class BlockingClose(io.BytesIO):
        def close(self): release.wait()
    proc=Process(); proc.stdout=BlockingClose(ready(job)); proc.stderr=io.BytesIO()
    started=time.monotonic()
    result=run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=lambda *a,**k:proc,reader=lambda **_:view())
    elapsed=time.monotonic()-started; release.set()
    assert elapsed<.5 and result.phase is Phase.FAILED
    with sqlite3.connect(store.path) as db:
        facts=json.loads(db.execute("SELECT facts_json FROM job_events ORDER BY sequence DESC LIMIT 1").fetchone()[0])
    assert facts["stdout_reader_error"]=="CLOSE_ERROR"

def test_pipe_closer_start_failure_is_bounded_and_durably_reported(tmp_path,monkeypatch):
    cfg,store,job=setup(tmp_path); proc=Process(ready(job)); constructed=[]
    real_thread=threading.Thread
    class StartFailingCloser:
        def __init__(self,*args,**kwargs): self.inner=real_thread(*args,**kwargs)
        def start(self): raise RuntimeError("closer start failed")
        def join(self,timeout=None): return self.inner.join(timeout)
        def is_alive(self): return self.inner.is_alive()
    def thread(*args,**kwargs):
        name=kwargs.get("name"); constructed.append(name)
        if name=="predictor-stdout-close": return StartFailingCloser(*args,**kwargs)
        return real_thread(*args,**kwargs)
    monkeypatch.setattr("src.operator_ui.prediction_worker.threading.Thread",thread)
    began=time.monotonic()
    result=run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=lambda *a,**k:proc,reader=lambda **_:view())
    assert time.monotonic()-began<.5 and result.phase is Phase.FAILED
    assert constructed[:2]==["predictor-stdout","predictor-stderr"]
    with sqlite3.connect(store.path) as db:
        facts=json.loads(db.execute("SELECT facts_json FROM job_events ORDER BY sequence DESC LIMIT 1").fetchone()[0])
    assert facts["stdout_reader_error"]=="CLOSE_ERROR"

def test_claim_audit_failure_prevents_claim_and_launch(tmp_path):
    cfg,store,job=setup(tmp_path); calls=[]; fail=lambda intent:(_ for _ in ()).throw(RuntimeError("audit failed"))
    with pytest.raises(JobStoreError):run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=fail,popen=lambda *a,**k:calls.append(1),reader=lambda **_:view())
    assert not store.get(job.job_id).attempt_claimed and not calls

def test_post_spawn_persistence_failure_reaps_and_never_retries(tmp_path,monkeypatch):
    cfg,store,job=setup(tmp_path); proc=Process(ready(job)); original=store.transition
    monkeypatch.setattr(store,"transition",lambda *a,**k:(_ for _ in ()).throw(JobStoreError("injected")))
    with pytest.raises(WorkerRejected,match="POST_SPAWN"):run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=lambda *a,**k:proc,reader=lambda **_:view())
    monkeypatch.setattr(store,"transition",original)
    assert not proc.terminated and store.get(job.job_id).attempt_claimed and store.get(job.job_id).phase is Phase.CLAIMED

def test_change_during_spawn_reaps_and_leaves_unique_nonretryable_attempt(tmp_path):
    cfg,store,job=setup(tmp_path); proc=Process(ready(job))
    def popen(*args,**kwargs):
        cfg.choices["latest-research"].model_path.write_bytes(b"replaced-across-spawn")
        return proc
    with pytest.raises(WorkerRejected,match="POST_SPAWN_IDENTITY_CHANGED"):
        run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=popen,reader=lambda **_:view())
    assert not proc.terminated and store.get(job.job_id).attempt_claimed

@pytest.mark.parametrize("confirmed,phase",[(True,Phase.CANCELLED),(False,Phase.REAP_UNCONFIRMED)])
def test_cancellation_requires_confirmed_reap(tmp_path,confirmed,phase):
    from src.operator_ui.prediction_worker import CancellationRequested
    cfg,store,job=setup(tmp_path); cancelled=CancellationRequested("cancelled")
    proc=Process(wait_errors=(cancelled,) if confirmed else (cancelled,subprocess.TimeoutExpired("predictor",2)))
    result=run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=lambda *a,**k:proc,reader=lambda **_:view())
    assert result.phase is phase and proc.terminated

def test_bundle_text_is_never_persisted_as_response_bytes(tmp_path):
    import json,sqlite3
    cfg,store,job=setup(tmp_path); value=json.loads(legacy_blocked()); value["bundle"]="/private/absolute/producer/path"
    result=run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=lambda *a,**k:Process(canonical_bytes(value),code=2),reader=lambda **_:view())
    assert result.reason=="PROCESS_OUTPUT_INVALID" and result.evidence_bundle_ref is None
    with sqlite3.connect(store.path) as db:
        facts=[json.loads(row[0]) for row in db.execute("SELECT facts_json FROM job_events WHERE phase='RESPONSE_RECORDED'")]
    assert len(facts)==1 and "stdout_bytes" not in facts[0] and facts[0]["stdout_sha256"]==hashlib.sha256(canonical_bytes(value)).hexdigest()

@pytest.mark.parametrize("field,value",[("research_only",False),("production_persisted",True),("betting_output",True)])
def test_sealed_v2_safety_flags_fail_closed(tmp_path,field,value):
    cfg,store,job=setup(tmp_path); payload=json.loads(ready(job)); payload[field]=value
    result=run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=lambda *a,**k:Process(canonical_bytes(payload)),reader=lambda **_:view())
    assert result.phase is Phase.FAILED and result.reason=="PROCESS_OUTPUT_INVALID"

def test_server_owned_cancellation_before_wait_uses_predictor_only_reap(tmp_path):
    cfg,store,job=setup(tmp_path); proc=Process(ready(job)); calls=[]
    result=run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=lambda *a,**k:(calls.append(1) or proc),reader=lambda **_:view(),cancel_requested=lambda:True)
    assert result.phase is Phase.CANCELLED and not proc.terminated and len(calls)==1

def test_second_reader_constructor_failure_keeps_single_owner_and_closes_both_pipes(tmp_path,monkeypatch):
    cfg,store,job=setup(tmp_path); proc=Process(ready(job)); popen_calls=[]; constructed=[]
    real_thread=threading.Thread
    def thread(*args,**kwargs):
        constructed.append(kwargs.get("name"))
        if len(constructed)==2: raise RuntimeError("second reader constructor failed")
        return real_thread(*args,**kwargs)
    monkeypatch.setattr("src.operator_ui.prediction_worker.threading.Thread",thread)
    result=run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=lambda *a,**k:(popen_calls.append(1) or proc),reader=lambda **_:view())
    assert result.phase is Phase.FAILED
    assert len(popen_calls)==1 and constructed[:2]==["predictor-stdout","predictor-stderr"]
    assert proc.stdout.closed and proc.stderr.closed

class FaultPipe(io.BytesIO):
    def __init__(self,data=b"",fault=None): super().__init__(data); self.fault=fault; self.close_attempts=0; self.release=threading.Event()
    def read(self,*args):
        if self.fault=="read": raise OSError("read failed")
        if self.fault=="stall": self.release.wait(5)
        return super().read(*args)
    def close(self):
        self.close_attempts+=1; self.release.set()
        if self.fault=="close": raise OSError("close failed")
        return super().close()

@pytest.mark.parametrize("fault",["read","start","join","close","stall"])
def test_lifetime_reader_error_matrix_is_bounded_truthful_and_closes_every_pipe(tmp_path,monkeypatch,fault):
    cfg,store,job=setup(tmp_path); proc=Process(ready(job)); proc.stdout=FaultPipe(ready(job),fault if fault in {"read","close","stall"} else None); proc.stderr=FaultPipe()
    real=threading.Thread; constructed=[]
    if fault in {"start","join"}:
        class Wrapped:
            def __init__(self,*args,**kwargs): self.inner=real(*args,**kwargs)
            def start(self):
                if fault=="start" and len(constructed)==2: raise RuntimeError("start failed")
                self.inner.start()
            def join(self,timeout=None):
                if fault=="join": raise RuntimeError("join failed")
                return self.inner.join(timeout)
            def is_alive(self): return self.inner.is_alive()
        def factory(*args,**kwargs): constructed.append(kwargs.get("name")); return Wrapped(*args,**kwargs)
        monkeypatch.setattr("src.operator_ui.prediction_worker.threading.Thread",factory)
    began=time.monotonic(); result=run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=lambda *a,**k:proc,reader=lambda **_:view()); elapsed=time.monotonic()-began
    assert elapsed<3.0 and result.phase is Phase.FAILED
    assert proc.stdout.close_attempts>=1 and proc.stderr.close_attempts>=1
    with sqlite3.connect(store.path) as db: facts=json.loads(db.execute("SELECT facts_json FROM job_events ORDER BY sequence DESC LIMIT 1").fetchone()[0])
    expected={"read":"READ_ERROR","start":"START_ERROR","join":"JOIN_ERROR","close":"CLOSE_ERROR","stall":"INCOMPLETE"}[fault]
    assert facts["stdout_reader_error"]==expected or fault=="start" and facts["stderr_reader_error"]==expected
    assert facts["stdout_complete"] is False or facts["stderr_complete"] is False
    assert "stdout_length" not in facts if facts["stdout_complete"] is False else True

@pytest.mark.parametrize("role,method,expected",[("reader","is_alive","INCOMPLETE"),("closer","join","CLOSE_ERROR"),("closer","is_alive","CLOSE_ERROR")])
def test_cleanup_helper_observation_exceptions_are_bounded_and_durable(tmp_path,monkeypatch,role,method,expected):
    cfg,store,job=setup(tmp_path); proc=Process(ready(job)); real=threading.Thread
    class Wrapped:
        def __init__(self,*args,**kwargs): self.inner=real(*args,**kwargs); self.role="closer" if kwargs.get("name","").endswith("-close") else "reader"
        def start(self): self.inner.start()
        def join(self,timeout=None):
            if self.role==role and method=="join": raise RuntimeError(f"{role} join failed")
            return self.inner.join(timeout)
        def is_alive(self):
            if self.role==role and method=="is_alive": raise RuntimeError(f"{role} is_alive failed")
            return self.inner.is_alive()
    monkeypatch.setattr("src.operator_ui.prediction_worker.threading.Thread",Wrapped)
    began=time.monotonic(); result=run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=lambda *a,**k:proc,reader=lambda **_:view())
    assert time.monotonic()-began<3.0 and result.phase is Phase.FAILED
    with sqlite3.connect(store.path) as db: facts=json.loads(db.execute("SELECT facts_json FROM job_events ORDER BY sequence DESC LIMIT 1").fetchone()[0])
    assert facts["stdout_reader_error"]==expected or facts["stderr_reader_error"]==expected
    assert proc.poll()==0

@pytest.mark.parametrize("failed_phase",[Phase.ATTEMPT_STARTED,Phase.RESPONSE_RECORDED,Phase.PRODUCER_COMPLETED])
@pytest.mark.parametrize("fallback_fails",[False,True])
def test_postspawn_persistence_and_fallback_matrix_preserves_single_process_snapshot(tmp_path,monkeypatch,failed_phase,fallback_fails):
    from src.operator_ui.prediction_worker import PredictorDurabilityError
    cfg,store,job=setup(tmp_path); proc=Process(ready(job)); original=store.transition; failed=False
    def transition(job_id,phase,**kwargs):
        nonlocal failed
        if phase is failed_phase and not failed:
            failed=True; raise JobStoreError("primary persistence failure")
        if failed and fallback_fails: raise JobStoreError("fallback persistence failure")
        return original(job_id,phase,**kwargs)
    monkeypatch.setattr(store,"transition",transition)
    if fallback_fails:
        with pytest.raises(PredictorDurabilityError) as caught: run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=lambda *a,**k:proc,reader=lambda **_:view())
        assert caught.value.snapshot["pid"]==123
    else:
        if failed_phase is Phase.ATTEMPT_STARTED:
            with pytest.raises(WorkerRejected): run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=lambda *a,**k:proc,reader=lambda **_:view())
            assert store.get(job.job_id).phase is Phase.FAILED
        else:
            outcome=run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=lambda *a,**k:proc,reader=lambda **_:view())
            assert outcome.phase is Phase.FAILED
    assert proc.stdout.closed and proc.stderr.closed

def test_producer_json_cannot_upgrade_process_or_stream_evidence(tmp_path):
    cfg,store,job=setup(tmp_path); payload=json.loads(ready(job)); payload.update(pid=999,exit_code=99,stdout_complete=True,stdout_sha256="f"*64)
    raw=canonical_bytes(payload); result=run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=lambda *a,**k:Process(raw,code=0),reader=lambda **_:view())
    assert result.phase is Phase.FAILED and result.reason=="PROCESS_OUTPUT_INVALID"
    with sqlite3.connect(store.path) as db: facts=json.loads(db.execute("SELECT facts_json FROM job_events WHERE phase='RESPONSE_RECORDED'").fetchone()[0])
    assert facts["pid"]==123 and facts["exit_code"]==0 and facts["stdout_sha256"]==hashlib.sha256(raw).hexdigest()

def test_sealed_v2_blocker_is_nonfinal_and_persists_exact_identity(tmp_path):
    cfg,store,job=setup(tmp_path); result=run_once(store,job.job_id,cfg,now=lambda:NOW,confirm_audit=CONFIRM,popen=lambda *a,**k:Process(sealed_blocked(job),code=2),reader=lambda **_:view())
    assert result.phase is Phase.PRODUCER_COMPLETED and result.reason=="PRODUCER_PREDICTION_BLOCKED:POST_JUMP"
    with sqlite3.connect(store.path) as db: facts=json.loads(db.execute("SELECT facts_json FROM job_events ORDER BY sequence DESC LIMIT 1").fetchone()[0])
    assert facts["producer_blocker"]=={"code":"POST_JUMP","stage":"VALIDATION"}
