"""One-claim, fixed-argument, bounded Operator UI predictor worker."""
from __future__ import annotations
import hashlib
import io
import json
import os
import stat
import subprocess
import threading
import time
from dataclasses import dataclass,field
from datetime import datetime
from pathlib import Path
from typing import Any,Callable,Mapping

from race_collection.synchronous_manual_capture import CaptureOneRejected,VerifiedCurrentRaceIndex,bounded_current_race_index
from src.predictor.on_demand import PredictionBlocked,canonical_bytes,validate_prediction_result_v2
from .job_store import AuditConfirmation,Job,JobStore,JobStoreError,Phase

MAX_STDOUT_BYTES=1_048_576; MAX_STDERR_BYTES=65_536
class WorkerRejected(RuntimeError): pass
class _DurablyHandled(WorkerRejected): pass
class CancellationRequested(RuntimeError): pass
class PredictorLifetimeUnknown(WorkerRejected): pass
class PredictorDurabilityError(PredictorLifetimeUnknown):
    def __init__(self,message:str,*,primary:BaseException,fallback:BaseException,snapshot:Mapping[str,Any]|None=None):
        super().__init__(message); self.primary=primary; self.fallback=fallback; self.snapshot=dict(snapshot or {})

@dataclass(frozen=True,slots=True)
class ServerChoice:
    config_path:Path; config_id:str; config_sha256:str; resolved_model_identity:str; model_sha256:str; model_manifest_sha256:str; model_schema_sha256:str
    model_path:Path|None=None; model_manifest_path:Path|None=None; model_schema_path:Path|None=None

@dataclass(frozen=True,slots=True)
class WorkerConfig:
    python_executable:Path; repository_root:Path; choices:Mapping[str,ServerChoice]; canonical_db:Path; output_root:Path
    capture_evidence_roots:tuple[Path,...]; collector_request_root:Path; current_index_path:Path; current_index_evidence_root:Path
    current_index_timeout_seconds:float; fetch_timeout_seconds:float; process_timeout_seconds:float; cancellation_grace_seconds:float=15.0
    _identities:Mapping[str,tuple[tuple[int,int],...]]=field(init=False,repr=False,compare=False)
    _runtime:tuple[tuple[Path,tuple[int,int],str],...]=field(init=False,repr=False,compare=False)
    def __post_init__(self):
        if not self.capture_evidence_roots or not self.choices: raise ValueError("worker allowlists must be non-empty")
        for v in (self.current_index_timeout_seconds,self.fetch_timeout_seconds,self.process_timeout_seconds,self.cancellation_grace_seconds):
            if not isinstance(v,(int,float)) or isinstance(v,bool) or v<=0: raise ValueError("worker timeouts must be positive")
        executable=self.python_executable.resolve(strict=True); script=(self.repository_root/"scripts/predict_race_now.py").absolute()
        runtime=[]
        for path in (executable,script):
            identity=_regular_identity(path)
            if identity is None: raise ValueError("fixed predictor executable unavailable")
            runtime.append((path,identity,hashlib.sha256(path.read_bytes()).hexdigest()))
        identities={}
        for selector,choice in self.choices.items():
            paths=(choice.config_path,choice.model_path,choice.model_manifest_path,choice.model_schema_path)
            hashes=(choice.config_sha256,choice.model_sha256,choice.model_manifest_sha256,choice.model_schema_sha256)
            if not selector or any(p is None for p in paths): raise ValueError("invalid model/config allowlist")
            current=[]
            for path,expected in zip(paths,hashes):
                identity=_regular_identity(path)
                if identity is None or hashlib.sha256(path.read_bytes()).hexdigest()!=expected: raise ValueError("invalid model/config allowlist")
                current.append(identity)
            identities[selector]=tuple(current)
        object.__setattr__(self,"_identities",identities)
        object.__setattr__(self,"_runtime",tuple(runtime))
    @property
    def script(self): return self._runtime[1][0]
    @property
    def pinned_python(self): return self._runtime[0][0]

def _regular_identity(path:Path)->tuple[int,int]|None:
    try: st=Path(path).lstat()
    except OSError:return None
    return (st.st_dev,st.st_ino) if stat.S_ISREG(st.st_mode) and not Path(path).is_symlink() else None

def _validate_choice(job:Job,config:WorkerConfig)->ServerChoice:
    choice=config.choices.get(job.input.model_selector)
    if choice is None: raise WorkerRejected("MODEL_CONFIG_NOT_ALLOWLISTED")
    expected=(choice.config_id,choice.config_sha256,choice.resolved_model_identity,choice.model_sha256,choice.model_manifest_sha256,choice.model_schema_sha256)
    actual=(job.input.config_id,job.input.config_sha256,job.input.resolved_model_identity,job.input.model_sha256,job.input.model_manifest_sha256,job.input.model_schema_sha256)
    if actual!=expected: raise WorkerRejected("MODEL_CONFIG_IDENTITY_CHANGED")
    paths=(choice.config_path,choice.model_path,choice.model_manifest_path,choice.model_schema_path); hashes=(choice.config_sha256,choice.model_sha256,choice.model_manifest_sha256,choice.model_schema_sha256)
    for path,expected_hash,expected_identity in zip(paths,hashes,config._identities[job.input.model_selector]):
        if path is None or _regular_identity(path)!=expected_identity:
            raise WorkerRejected("MODEL_CONFIG_FILE_IDENTITY_CHANGED")
        try: digest=hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc: raise WorkerRejected("MODEL_CONFIG_FILE_IDENTITY_CHANGED") from exc
        if digest!=expected_hash: raise WorkerRejected("MODEL_CONFIG_BYTES_CHANGED")
    return choice

def _validate_runtime(config:WorkerConfig)->None:
    for path,identity,digest in config._runtime:
        if _regular_identity(path)!=identity: raise WorkerRejected("PREDICTOR_RUNTIME_IDENTITY_CHANGED")
        try: actual=hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc: raise WorkerRejected("PREDICTOR_RUNTIME_IDENTITY_CHANGED") from exc
        if actual!=digest: raise WorkerRejected("PREDICTOR_RUNTIME_BYTES_CHANGED")

def fixed_argv(job:Job,config:WorkerConfig)->tuple[str,...]:
    choice=_validate_choice(job,config)
    _validate_runtime(config)
    argv=[str(config.pinned_python),str(config.script),"--race-id",job.input.race_id,"--model",job.input.model_selector,"--job-id",job.job_id,"--config",str(choice.config_path),"--odds-source",job.input.odds_source,"--db",str(config.canonical_db),"--output-root",str(config.output_root)]
    for root in config.capture_evidence_roots: argv.extend(("--capture-evidence-root",str(root)))
    argv.extend(("--collector-request-root",str(config.collector_request_root),"--fetch-timeout-seconds",str(config.fetch_timeout_seconds)))
    return tuple(argv)

def _open_runtime_descriptors(config:WorkerConfig)->tuple[int,int]:
    """Open and validate the exact runtime objects retained across exec."""
    if not Path("/proc/self/fd").is_dir(): raise WorkerRejected("PREDICTOR_PROCFS_UNAVAILABLE")
    opened=[]
    try:
        for path,identity,digest in config._runtime:
            fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
            opened.append(fd); st=os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or (st.st_dev,st.st_ino)!=identity: raise WorkerRejected("PREDICTOR_RUNTIME_IDENTITY_CHANGED")
            h=hashlib.sha256()
            while True:
                chunk=os.read(fd,1024*1024)
                if not chunk: break
                h.update(chunk)
            os.lseek(fd,0,os.SEEK_SET)
            if h.hexdigest()!=digest: raise WorkerRejected("PREDICTOR_RUNTIME_BYTES_CHANGED")
        return opened[0],opened[1]
    except BaseException:
        for fd in opened:
            try: os.close(fd)
            except OSError: pass
        raise

def revalidate_current_race(job,config,*,now,reader=bounded_current_race_index):
    try:view=reader(current_time=now,timeout_seconds=config.current_index_timeout_seconds,index_path=config.current_index_path,evidence_root=config.current_index_evidence_root,max_age_seconds=1200,return_verified_view=True)
    except CaptureOneRejected as exc: raise WorkerRejected(exc.code) from exc
    if not isinstance(view,VerifiedCurrentRaceIndex): raise WorkerRejected("CURRENT_INDEX_INVALID")
    matches=[r for r in view.races if r.get("race_id")==job.input.race_id]
    if len(matches)!=1: raise WorkerRejected("RACE_ID_MISSING_OR_AMBIGUOUS")
    if matches[0].get("jump_datetime")!=job.input.jump_timestamp: raise WorkerRejected("RACE_JUMP_CHANGED")
    if matches[0].get("runner_set_sha256")!=job.input.runner_set_sha256: raise WorkerRejected("RUNNER_SET_CHANGED")
    return view

def _drain(pipe:Any,cap:int,sink:dict[str,Any],name:str):
    captured=bytearray(); total=0; digest=hashlib.sha256()
    while True:
        chunk=pipe.read(65536)
        if not chunk:break
        if not isinstance(chunk,bytes): raise TypeError("process pipe must yield bytes")
        total+=len(chunk); digest.update(chunk)
        if len(captured)<cap: captured.extend(chunk[:cap-len(captured)])
    sink[name]=(bytes(captured),total,digest.hexdigest())

class _LifetimeOwner:
    """The sole bounded owner of a successfully spawned predictor and its pipes."""
    def __init__(self,process:Any,cleanup_budget:float):
        self.process=process; self.cleanup_budget=cleanup_budget; self.sink={}; self.errors={}; self.threads=[]; self.closed={"stdout":False,"stderr":False}; self.primary=None; self._cleaned=False; self._reaped=False
        for name,pipe,cap in (("stdout",process.stdout,MAX_STDOUT_BYTES),("stderr",process.stderr,MAX_STDERR_BYTES)):
            def target(pipe=pipe,cap=cap,name=name):
                try:_drain(pipe,cap,self.sink,name)
                except BaseException as exc:self.errors.setdefault(name,"READ_ERROR")
            try: thread=threading.Thread(target=target,daemon=True,name=f"predictor-{name}")
            except BaseException as exc:
                self.errors[name]="START_ERROR"; self.primary=self.primary or exc
                self.threads.append((name,None,pipe))
                continue
            self.threads.append((name,thread,pipe))
            try:thread.start()
            except BaseException as exc:self.errors[name]="START_ERROR"; self.primary=self.primary or exc
    def wait(self,timeout:float,cancel_requested):
        if cancel_requested is None:
            return self.process.wait(timeout=timeout)
        deadline=time.monotonic()+timeout
        while True:
            if cancel_requested(): raise CancellationRequested("server cancellation requested")
            if self.process.poll() is not None:return self.process.poll()
            remaining=deadline-time.monotonic()
            if remaining<=0: raise subprocess.TimeoutExpired("predictor",timeout)
            try:self.process.wait(timeout=min(.05,remaining))
            except subprocess.TimeoutExpired:continue
        return self.process.poll()
    def cleanup(self,*,stop:bool)->bool:
        if self._cleaned:return self._reaped
        deadline=time.monotonic()+self.cleanup_budget
        def reader_alive(name,thread):
            try:return thread.is_alive()
            except BaseException as exc:
                self.errors.setdefault(name,"INCOMPLETE"); self.primary=self.primary or exc
                return True
        def closer_alive(name,closer):
            try:return closer.is_alive()
            except BaseException as exc:
                self.errors.setdefault(name,"CLOSE_ERROR"); self.primary=self.primary or exc
                return True
        if stop and self.process.poll() is None:
            try:self.process.terminate()
            except ProcessLookupError:pass
            except BaseException as exc:self.primary=self.primary or exc
        if self.process.poll() is None:
            try:self.process.wait(timeout=max(0,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:pass
            except BaseException as exc:self.primary=self.primary or exc
        for name,thread,pipe in self.threads:
            if thread is None: continue
            try:thread.join(timeout=max(0,deadline-time.monotonic()))
            except BaseException as exc:self.errors.setdefault(name,"JOIN_ERROR"); self.primary=self.primary or exc
        close_threads=[]
        for name,thread,pipe in self.threads:
            if thread is not None and reader_alive(name,thread):
                self.errors.setdefault(name,"INCOMPLETE")
            if not self.closed[name]:
                def close(target=pipe,stream=name):
                    try:target.close(); self.closed[stream]=True
                    except BaseException as exc:self.errors.setdefault(stream,"CLOSE_ERROR"); self.primary=self.primary or exc
                try:
                    closer=threading.Thread(target=close,daemon=True,name=f"predictor-{name}-close")
                except BaseException as exc:
                    self.errors[name]="CLOSE_ERROR"; self.primary=self.primary or exc
                    continue
                try:closer.start()
                except BaseException as exc:
                    self.errors[name]="CLOSE_ERROR"; self.primary=self.primary or exc
                    continue
                close_threads.append((name,closer))
        for name,closer in close_threads:
            try:closer.join(timeout=max(0,deadline-time.monotonic()))
            except BaseException as exc:self.errors.setdefault(name,"CLOSE_ERROR"); self.primary=self.primary or exc
            if closer_alive(name,closer): self.errors.setdefault(name,"CLOSE_ERROR")
        for name,thread,_pipe in self.threads:
            if thread is not None and reader_alive(name,thread):
                try:thread.join(timeout=max(0,deadline-time.monotonic()))
                except BaseException as exc:self.errors.setdefault(name,"JOIN_ERROR"); self.primary=self.primary or exc
        self._reaped=self.process.poll() is not None; self._cleaned=True
        return self._reaped
    def evidence(self)->dict[str,Any]:
        facts={"exit_code":self.process.poll()}
        for name,cap in (("stdout",MAX_STDOUT_BYTES),("stderr",MAX_STDERR_BYTES)):
            captured,total,digest=self.sink.get(name,(b"",0,None)); error=self.errors.get(name)
            complete=name in self.sink and error is None
            facts.update({name+"_complete":complete,name+"_reader_error":"NONE" if complete else (error or "INCOMPLETE"),name+"_prefix_length":len(captured),name+"_prefix_sha256":hashlib.sha256(captured).hexdigest()})
            if complete:facts.update({name+"_length":total,name+"_sha256":digest})
            if name=="stderr":facts[name+"_bytes"]=captured.hex()
        return facts

def drain_bounded(process:Any,*,timeout:float,cancel_requested:Callable[[],bool]|None=None)->tuple[bytes,int,str,bytes,int,str]:
    sink={}; errors=[]
    def target(pipe,cap,name):
        try:_drain(pipe,cap,sink,name)
        except BaseException as exc:errors.append(exc)
    threads=[threading.Thread(target=target,args=(process.stdout,MAX_STDOUT_BYTES,"stdout"),daemon=True),threading.Thread(target=target,args=(process.stderr,MAX_STDERR_BYTES,"stderr"),daemon=True)]
    for thread in threads:thread.start()
    primary=None
    try:
        if cancel_requested is None:
            process.wait(timeout=timeout)
        else:
            deadline=time.monotonic()+timeout
            while process.poll() is None:
                if cancel_requested(): raise CancellationRequested("server cancellation requested")
                remaining=deadline-time.monotonic()
                if remaining<=0: raise subprocess.TimeoutExpired("predictor",timeout)
                try: process.wait(timeout=min(.05,remaining))
                except subprocess.TimeoutExpired: pass
    except BaseException as exc: primary=exc
    finally:
        for thread in threads: thread.join(timeout=timeout)
        for pipe in (process.stdout,process.stderr):
            try: pipe.close()
            except BaseException as exc:
                if primary is None: primary=exc
    if any(thread.is_alive() for thread in threads): raise subprocess.TimeoutExpired("predictor-pipes",timeout) from primary
    if primary is not None: raise primary
    if errors: raise WorkerRejected("PROCESS_PIPE_INVALID") from errors[0]
    out,out_len,out_hash=sink["stdout"]; err,err_len,err_hash=sink["stderr"]; return out,out_len,out_hash,err,err_len,err_hash

def _evidence(stdout,stdout_len,stdout_hash,stderr,stderr_len,stderr_hash,returncode):
    return {"exit_code":returncode,"stdout_complete":True,"stdout_length":stdout_len,"stdout_sha256":stdout_hash,"stdout_prefix_length":len(stdout),"stdout_prefix_sha256":hashlib.sha256(stdout).hexdigest(),"stderr_complete":True,"stderr_bytes":stderr.hex(),"stderr_length":stderr_len,"stderr_sha256":stderr_hash,"stderr_prefix_length":len(stderr),"stderr_prefix_sha256":hashlib.sha256(stderr).hexdigest()}

def _bounded_result(job:Job,stdout:bytes,stdout_len:int,stdout_hash:str,stderr:bytes,stderr_len:int,stderr_hash:str,returncode:int):
    facts=_evidence(stdout,stdout_len,stdout_hash,stderr,stderr_len,stderr_hash,returncode)
    def semantics(): return {name:facts[name] for name in ("predictor_status","prediction_id","producer_job_id","producer_blocker","protocol_chain","authenticated_cutoff") if name in facts}
    if stdout_len>MAX_STDOUT_BYTES or stderr_len>MAX_STDERR_BYTES:return Phase.FAILED,"PROCESS_OUTPUT_OVERSIZED",{}
    try:
        value=json.loads(stdout,parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        if not isinstance(value,dict) or canonical_bytes(value)!=stdout: raise ValueError
        schema=value.get("schema_version"); status=value.get("status")
        if not isinstance(status,str) or not status or len(status)>128:raise ValueError
        if schema=="on_demand_race_prediction_v2":
            validate_prediction_result_v2(value)
            if value["job_id"]!=job.job_id or value["race"]["race_id"]!=job.input.race_id or value["race"]["jump_timestamp"]!=job.input.jump_timestamp or value["evidence"]["runner_set_sha256"]!=job.input.runner_set_sha256 or value["config"]!={"sha256":job.input.config_sha256}:raise ValueError
            model=value["model"]
            if model.get("resolved")!=job.input.resolved_model_identity or model.get("artifact_sha256")!=job.input.model_sha256 or model.get("artifact_manifest_sha256")!=job.input.model_manifest_sha256 or model.get("schema_sha256")!=job.input.model_schema_sha256:raise ValueError
            prediction_id=value["prediction_id"]
            if not isinstance(prediction_id,str) or not prediction_id or len(prediction_id)>128:raise ValueError
            facts.update({"predictor_status":status,"prediction_id":prediction_id,"producer_job_id":value["job_id"]})
            facts.update({"protocol_chain":value["evidence"]["protocol_chain"],"authenticated_cutoff":value["evidence"]["authenticated_cutoff"]})
            if status=="PREDICTION_BLOCKED":
                code=value["blocker"]["code"]
                facts["producer_blocker"]={"code":code,"stage":value["blocker_stage"]}
        else:raise ValueError
    except (UnicodeDecodeError,json.JSONDecodeError,PredictionBlocked,AttributeError,KeyError,TypeError,ValueError):return Phase.FAILED,"PROCESS_OUTPUT_INVALID",{}
    if returncode==0 and status=="PREDICTION_READY":return Phase.PRODUCER_COMPLETED,"PRODUCER_PREDICTION_READY",semantics()
    if schema=="on_demand_race_prediction_v2" and status=="PREDICTION_BLOCKED":return Phase.PRODUCER_COMPLETED,f"PRODUCER_PREDICTION_BLOCKED:{facts['producer_blocker']['code']}",semantics()
    if status=="RECEIPT_READY":return Phase.FAILED,"RECEIPT_READY_IS_NOT_PREDICTION_SUCCESS",semantics()
    if returncode!=0 and status!="PREDICTION_READY":return Phase.REJECTED,f"PREDICTOR_BLOCKER:{status}",semantics()
    return Phase.FAILED,"PROCESS_EXIT_STATUS_MISMATCH",semantics()

def _stop_and_reap(process,grace):
    process.terminate()
    try:process.wait(timeout=grace)
    except subprocess.TimeoutExpired:return False
    return process.poll() is not None

def run_once(store:JobStore,job_id:str,config:WorkerConfig,*,now:Callable[[],datetime],confirm_audit:AuditConfirmation,popen:Callable[...,Any]=subprocess.Popen,reader=bounded_current_race_index,cancel_requested:Callable[[],bool]|None=None)->Job:
    job=store.get(job_id)
    if job.phase is not Phase.WAITING_FOR_CLAIM or job.attempt_claimed:raise WorkerRejected("JOB_NOT_CLAIMABLE")
    _validate_runtime(config); _validate_choice(job,config); revalidate_current_race(job,config,now=now(),reader=reader); _validate_choice(job,config); _validate_runtime(config)
    job,attempt_id=store.claim_attempt(job_id,now=now(),confirm_audit=confirm_audit)
    argv=fixed_argv(job,config); process=None; owner=None; started=False; runtime_fds=()
    try:
        _validate_runtime(config); _validate_choice(job,config)
        runtime_fds=_open_runtime_descriptors(config)
        retained=(str(config.pinned_python),f"/proc/self/fd/{runtime_fds[1]}",*argv[2:])
        try: process=popen(retained,executable=f"/proc/self/fd/{runtime_fds[0]}",pass_fds=runtime_fds,shell=False,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=False)
        finally:
            for fd in runtime_fds:
                try: os.close(fd)
                except OSError: pass
            runtime_fds=()
        owner=_LifetimeOwner(process,config.cancellation_grace_seconds)
        _validate_runtime(config); _validate_choice(job,config)
        job=store.transition(job_id,Phase.ATTEMPT_STARTED,now=now(),status="RUNNING",reason="predictor_started",facts={"attempt_id":attempt_id,"pid":process.pid},confirm_audit=confirm_audit)
        started=True
        owner.wait(config.process_timeout_seconds,cancel_requested)
        reaped=owner.cleanup(stop=False); facts=owner.evidence()
        if not reaped: raise PredictorLifetimeUnknown("PREDICTOR_REAP_UNCONFIRMED")
        if owner.primary is not None: raise owner.primary
        stdout=owner.sink.get("stdout",(b"",0,""))[0]; stderr=owner.sink.get("stderr",(b"",0,""))[0]
        stdout_len=owner.sink.get("stdout",(b"",0,""))[1]; stdout_hash=owner.sink.get("stdout",(b"",0,""))[2]
        stderr_len=owner.sink.get("stderr",(b"",0,""))[1]; stderr_hash=owner.sink.get("stderr",(b"",0,""))[2]
        phase,reason,parsed=_bounded_result(job,stdout,stdout_len,stdout_hash,stderr,stderr_len,stderr_hash,process.poll()); facts.update(parsed)
        store.transition(job_id,Phase.RESPONSE_RECORDED,now=now(),status="RECORDED",reason="bounded_process_response",facts={**facts,"attempt_id":attempt_id,"pid":process.pid},confirm_audit=confirm_audit)
        return store.transition(job_id,phase,now=now(),status=phase.value,reason=reason,facts={**facts,"attempt_id":attempt_id,"pid":process.pid},confirm_audit=confirm_audit)
    except BaseException as exc:
        if isinstance(exc,(_DurablyHandled,PredictorDurabilityError)): raise
        if process is None:
            if isinstance(exc,OSError):
                try:return store.transition(job_id,Phase.FAILED,now=now(),status="FAILED",reason="PROCESS_LAUNCH_FAILED",facts={"attempt_id":attempt_id,"error":type(exc).__name__},confirm_audit=confirm_audit)
                except BaseException as fallback_exc: raise PredictorDurabilityError("PRELAUNCH_DURABILITY_FAILED",primary=exc,fallback=fallback_exc,snapshot={"attempt_id":attempt_id}) from exc
            raise WorkerRejected("PROCESS_LAUNCH_FAILED") from exc
        owner=owner or _LifetimeOwner(process,config.cancellation_grace_seconds)
        reaped=owner.cleanup(stop=True); evidence={**owner.evidence(),"attempt_id":attempt_id,"pid":process.pid}
        if not reaped: phase=Phase.REAP_UNCONFIRMED; reason="PREDICTOR_REAP_UNCONFIRMED"
        elif isinstance(exc,CancellationRequested): phase=Phase.CANCELLED; reason="PREDICTOR_CANCELLED"
        elif isinstance(exc,subprocess.TimeoutExpired): phase=Phase.TIMED_OUT; reason="PREDICTOR_TIMEOUT"
        else: phase=Phase.FAILED; reason="POST_SPAWN_OSERROR" if isinstance(exc,OSError) else "POST_SPAWN_FAILURE"
        try:result=store.transition(job_id,phase,now=now(),status=phase.value,reason=reason,facts=evidence,confirm_audit=confirm_audit)
        except BaseException as fallback_exc: raise PredictorDurabilityError("POST_SPAWN_LIFETIME_DURABILITY_FAILED",primary=exc,fallback=fallback_exc,snapshot=evidence) from exc
        if not started:
            label="POST_SPAWN_IDENTITY_CHANGED" if isinstance(exc,WorkerRejected) else "POST_SPAWN_PERSISTENCE_FAILED"
            raise _DurablyHandled(label) from exc
        return result
    finally:
        for fd in runtime_fds:
            try: os.close(fd)
            except OSError: pass
