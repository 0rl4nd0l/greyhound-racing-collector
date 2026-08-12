"""Finite repository-owned startup composition for connected Operator UI."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import stat
import sys
import threading
import tomllib
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from flask import Flask

from race_collection.synchronous_manual_capture import CaptureOneRejected, VerifiedCurrentRaceIndex, bounded_current_race_index
from src.predictor.on_demand import PredictionBlocked
from .api import register_level_1_provider
from .job_store import JobInput, JobStore, Phase
from .foundation import JsonSerializationPolicy, JsonSource, OperatorEvidenceReader, RawSourceConfig, SourceConfig, TimestampSyntax
from .live_adapters import InstalledUnits, LiveEvidenceAdapters, PredictionBundleSource, UpcomingRaceSource
from .prediction_worker import ServerChoice, WorkerConfig, run_once
from .r3_api import R3Rejected, R3Services, ResolvedSubmission, build_verified_bundle_reader, install_r3_api

CONFIG_KEY = "OPERATOR_UI_LIVE_EVIDENCE_ADAPTERS"
R3_PROFILE_KEY = "OPERATOR_UI_R3_PROFILE"
_BOUND_KEY = "operator_ui_live_evidence_bound"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_METHODS = (("upcoming_races","upcoming"),("race_detail","race_detail"),("recent_predictions","recent_predictions"),("prediction_detail","prediction_detail"),("collector","collector"),("corpus","corpus"),("models","models"),("system","system"))
_PROFILES = {
    "repository-v1": "configs/operator_ui/repository-v1.toml",
    "fixture-v1": "tests/operator_ui/fixtures/r3_runtime",
}
_PROFILE_KEYS={"schema_version","profile_id","generated_binding","deployment","locators"}
_LOCATOR_KEYS={"prediction_script","prediction_config","model_artifact","model_manifest","model_schema","current_index","collector_protocol","prediction_bundles","audit_store","job_store"}
_BINDING_KEYS={"schema_version","profile_id","generator","deployment","profile_sha256","artifacts","roots"}
_ROOT_KEYS={"source_root","pinned_python","evidence_root","producer_root","canonical_db","operations_root"}
_GENERATOR_KEYS={"generator_id","schema_version","version"}
_DEPLOYMENT_KEYS={"source_commit","source_tree","ui_version","profile_id"}
_PROFILE_DEPLOYMENT_KEYS={"ui_version","profile_id"}
_ARTIFACT_KEYS={"prediction_script","prediction_config","model_artifact","model_manifest","model_schema"}
_MAX_CONTROL_BYTES=256*1024
_MAX_FULL_SOURCE_BYTES=512*1024
_FULL_SOURCE_KEYS={"full_state","full_report"}
_DIGEST_ONLY_RAW_KEYS={"corpus_inventory_csv","corpus_inventory_jsonl"}
_HEX40_RE=re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE=re.compile(r"^[0-9a-f]{64}$")
_R3_ALLOWED_SELECTION = ("latest-research", "manual-default", "receipt")


def _finite_text(value:Any)->bool:
    return isinstance(value,str) and 0<len(value)<=512 and value==value.strip() and all(ord(character)>=32 and ord(character)!=127 for character in value)


def bind_configured_live_evidence(app: Flask) -> bool:
    if not isinstance(app, Flask): raise TypeError("a Flask application is required")
    adapter=app.config.get(CONFIG_KEY)
    if adapter is None:return False
    if type(adapter) is not LiveEvidenceAdapters:raise TypeError("configured live evidence must be an exact LiveEvidenceAdapters")
    if app.extensions.get(_BOUND_KEY):raise RuntimeError("live evidence adapter is already bound")
    registry=app.extensions.get("operator_ui_level_1_api_providers")
    if not isinstance(registry,dict):raise RuntimeError("Operator UI Level-1 API is not installed")
    resources=tuple(resource for resource,_ in _METHODS)
    if any(resource in registry for resource in resources):raise ValueError("partial or replacement live evidence binding is forbidden")
    providers=tuple((resource,getattr(adapter,method)) for resource,method in _METHODS)
    if any(not callable(provider) for _,provider in providers):raise TypeError("live evidence adapter method is not callable")
    for resource,provider in providers:register_level_1_provider(app,resource,provider)
    app.extensions[_BOUND_KEY]=adapter
    return True


def _open_fixed(path:Path,*,directory:bool)->tuple[list[int],list[tuple[int,int,int,int]],list[tuple[int,str,tuple[int,int]]]]:
    """Open and retain every component, enforcing the fixed-path trust policy."""
    parts=path.absolute().parts; descriptor=os.open("/",os.O_RDONLY|os.O_DIRECTORY); opened=[descriptor]
    try:
        named=[]
        for offset,component in enumerate(parts[1:]):
            flags=os.O_RDONLY|os.O_NOFOLLOW
            final=offset==len(parts)-2
            if not final or directory:flags|=os.O_DIRECTORY
            child=os.open(component,flags,dir_fd=descriptor);opened.append(child);descriptor=child
            info=os.fstat(child);named.append((opened[-2],component,(info.st_dev,info.st_ino)))
        identities=[]; owner=os.geteuid()
        for item in opened:
            info=os.fstat(item);mode=stat.S_IMODE(info.st_mode)
            if mode&0o002 and not (stat.S_ISDIR(info.st_mode) and mode&stat.S_ISVTX):raise RuntimeError("fixed R3 runtime unsafe")
            if mode&0o020 and info.st_uid!=owner:raise RuntimeError("fixed R3 runtime unsafe")
            identities.append((info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns))
        final_info=os.fstat(descriptor)
        if directory != stat.S_ISDIR(final_info.st_mode) or (not directory and not stat.S_ISREG(final_info.st_mode)):raise RuntimeError("fixed R3 runtime unsafe")
        return opened,identities,named
    except OSError as exc:
        for item in reversed(opened):os.close(item)
        raise RuntimeError("fixed R3 runtime unavailable") from exc
    except BaseException:
        for item in reversed(opened):os.close(item)
        raise


def _retained_read(path:Path,*,maximum:int=_MAX_CONTROL_BYTES)->bytes:
    """Bounded read for a small fixed control/artifact file only."""
    opened,identities,named=_open_fixed(path,directory=False);descriptor=opened[-1]
    try:
        chunks=[];total=0
        while True:
            chunk=os.read(descriptor,min(65536,maximum+1-total))
            if not chunk:break
            total+=len(chunk)
            if total>maximum:raise RuntimeError("fixed R3 runtime oversized")
            chunks.append(chunk)
        if any((value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns)!=expected for value,expected in zip(map(os.fstat,opened),identities)) or any((value.st_dev,value.st_ino)!=expected for parent,name,expected in named for value in (os.stat(name,dir_fd=parent,follow_symlinks=False),)):raise RuntimeError("fixed R3 runtime changed")
        return b"".join(chunks)
    finally:
        for item in reversed(opened):os.close(item)


def _retained_source_read(path:Path,key:str)->bytes:
    maximum=_MAX_FULL_SOURCE_BYTES if key in _FULL_SOURCE_KEYS else _MAX_CONTROL_BYTES
    return _retained_read(path,maximum=maximum)


def _regular(path: Path) -> None:
    opened,_,_=_open_fixed(path,directory=False)
    for item in reversed(opened):os.close(item)


def _directory(path: Path) -> None:
    opened,_,_=_open_fixed(path,directory=True)
    for item in reversed(opened):os.close(item)


def _sha(path: Path) -> str:
    return hashlib.sha256(_retained_read(path)).hexdigest()


def _object(path:Path,label:str)->dict[str,Any]:
    def exact(pairs):
        value={}
        for key,item in pairs:
            if key in value:raise ValueError("duplicate key")
            value[key]=item
        return value
    try:value=json.loads(_retained_read(path),parse_constant=lambda value:(_ for _ in ()).throw(ValueError(value)),object_pairs_hook=exact)
    except (OSError,UnicodeDecodeError,json.JSONDecodeError,ValueError) as exc:raise RuntimeError(f"{label} invalid") from exc
    if type(value) is not dict:raise RuntimeError(f"{label} invalid")
    return value


def _relative(value:Any,label:str)->Path:
    if not isinstance(value,str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:raise RuntimeError(f"{label} invalid")
    return Path(value)


def _repository_layout()->dict[str,Any]:
    profile_path=(_REPOSITORY_ROOT/_PROFILES["repository-v1"]).absolute()
    profile_raw=_retained_read(profile_path)
    try:profile=tomllib.loads(profile_raw.decode("utf-8"))
    except (OSError,UnicodeDecodeError,tomllib.TOMLDecodeError) as exc:raise RuntimeError("repository-v1 profile invalid") from exc
    if set(profile)!=_PROFILE_KEYS or any(not _finite_text(value) for value in (profile.get("schema_version"),profile.get("profile_id"),profile.get("generated_binding"))) or profile["schema_version"]!="operator_ui_repository_profile_v1" or profile["profile_id"]!="repository-v1" or type(profile["locators"]) is not dict or set(profile["locators"])!=_LOCATOR_KEYS or any(not _finite_text(value) for value in profile["locators"].values()) or type(profile["deployment"]) is not dict or set(profile["deployment"])!=_PROFILE_DEPLOYMENT_KEYS or any(not _finite_text(value) for value in profile["deployment"].values()):raise RuntimeError("repository-v1 profile invalid")
    locators={name:_relative(value,f"repository-v1 locator {name}") for name,value in profile["locators"].items()}
    binding_path=(_REPOSITORY_ROOT/_relative(profile["generated_binding"],"generated binding locator")).absolute()
    try:binding=_object(binding_path,"generated repository-v1 binding")
    except RuntimeError as exc:
        if not binding_path.exists():raise RuntimeError("generated repository-v1 binding unavailable") from exc
        raise
    deployment=binding.get("deployment")
    if set(binding) not in (_BINDING_KEYS, _BINDING_KEYS|{"live_evidence"}) or binding["schema_version"]!="operator_ui_repository_binding_v1" or binding["profile_id"]!="repository-v1" or type(binding["roots"]) is not dict or set(binding["roots"])!=_ROOT_KEYS or type(binding["generator"]) is not dict or set(binding["generator"])!=_GENERATOR_KEYS or binding["generator"]!={"generator_id":"GHU-036-repository-v1-generator","schema_version":"operator_ui_repository_binding_generator_v1","version":"1"} or any(not _finite_text(value) for value in binding["generator"].values()) or type(deployment) is not dict or set(deployment)!=_DEPLOYMENT_KEYS or not _HEX40_RE.fullmatch(deployment.get("source_commit","") or "") or not _HEX40_RE.fullmatch(deployment.get("source_tree","") or "") or any(not _finite_text(deployment.get(key)) for key in ("ui_version","profile_id")) or any(deployment[key]!=profile["deployment"][key] for key in _PROFILE_DEPLOYMENT_KEYS) or not _HEX64_RE.fullmatch(binding.get("profile_sha256","") or "") or binding["profile_sha256"]!=hashlib.sha256(profile_raw).hexdigest() or type(binding["artifacts"]) is not dict or set(binding["artifacts"])!=_ARTIFACT_KEYS or any(not isinstance(value,str) or _HEX64_RE.fullmatch(value) is None for value in binding["artifacts"].values()):raise RuntimeError("generated repository-v1 binding invalid")
    roots={}
    for name,value in binding["roots"].items():
        path=Path(value) if isinstance(value,str) else Path("")
        if not isinstance(value,str) or not path.is_absolute():raise RuntimeError("generated repository-v1 binding invalid")
        try:resolved=path.resolve(strict=True)
        except OSError as exc:raise RuntimeError("generated repository-v1 binding invalid") from exc
        if resolved!=path:raise RuntimeError("generated repository-v1 binding invalid")
        roots[name]=path
    for name in ("source_root","evidence_root","producer_root","operations_root"):_directory(roots[name])
    for name in ("pinned_python","canonical_db"):_regular(roots[name])
    read_roots=(roots["source_root"].resolve(),roots["evidence_root"].resolve(),roots["producer_root"].resolve())
    operations=roots["operations_root"].resolve()
    if any(operations==root or operations.is_relative_to(root) or root.is_relative_to(operations) for root in read_roots) or roots["canonical_db"].resolve().is_relative_to(operations):raise RuntimeError("fixed R3 paths overlap")
    paths={"audit.sqlite3":operations/locators["audit_store"],"jobs.sqlite3":operations/locators["job_store"],"canonical.sqlite3":roots["canonical_db"],"current_index.json":roots["evidence_root"]/locators["current_index"]}
    dirs={"current_evidence":roots["evidence_root"],"prediction_bundles":roots["producer_root"]/locators["prediction_bundles"],"collector_requests":roots["evidence_root"]/locators["collector_protocol"],"capture_evidence_a":roots["evidence_root"]}
    artifacts={"script":roots["source_root"]/locators["prediction_script"],"config":roots["source_root"]/locators["prediction_config"],"model":roots["source_root"]/locators["model_artifact"],"manifest":roots["source_root"]/locators["model_manifest"],"schema":roots["source_root"]/locators["model_schema"]}
    if roots["source_root"]!=_REPOSITORY_ROOT:raise RuntimeError("generated repository-v1 source identity mismatch")
    _regular(paths["current_index.json"])
    for path in dirs.values():_directory(path)
    for path in artifacts.values():_regular(path)
    artifact_binding={"prediction_script":artifacts["script"],"prediction_config":artifacts["config"],"model_artifact":artifacts["model"],"model_manifest":artifacts["manifest"],"model_schema":artifacts["schema"]}
    if any(binding["artifacts"].get(name)!=_sha(path) for name,path in artifact_binding.items()):raise RuntimeError("generated repository-v1 artifact identity mismatch")
    return {"base":operations,"paths":paths,"dirs":dirs,"artifacts":artifacts,"source_root":roots["source_root"],"pinned_python":roots["pinned_python"],"deployment":dict(deployment),"live_evidence":binding.get("live_evidence")}


def _configured_live(layout:Mapping[str,Any])->LiveEvidenceAdapters:
    live=layout.get("live_evidence")
    if not isinstance(live,dict) or set(live)!={"schema_version","observed_at","working_directory","sources","raw_sources","units","service_status"} or live.get("schema_version")!="operator_ui_live_authority_v1":raise RuntimeError("generated live evidence binding invalid")
    def entry(group:str,key:str)->tuple[Path,str,Path|None]:
        item=live[group].get(key)
        digest_only=group=="raw_sources" and key in _DIGEST_ONLY_RAW_KEYS
        allowed=({"path","sha256","bytes","authentication"} if digest_only else {"path","sha256"})|({"allowlisted_root"} if group=="sources" and key=="odds_refresh" else set())
        if not isinstance(item,dict) or set(item)!=allowed or not isinstance(item["path"],str) or not Path(item["path"]).is_absolute() or not _HEX64_RE.fullmatch(item["sha256"]):raise RuntimeError("generated live evidence binding invalid")
        if digest_only and (item["authentication"]!="sha256_size_only_v1" or type(item["bytes"]) is not int or not 0<=item["bytes"]<=64*1024*1024):raise RuntimeError("generated live evidence binding invalid")
        path=Path(item["path"]);_regular(path)
        root=None
        if "allowlisted_root" in item:
            root=Path(item["allowlisted_root"])
            if not root.is_absolute():raise RuntimeError("generated live evidence binding invalid")
            _directory(root)
        return path,item["sha256"],root
    policies={"full_state":"P-COLLECTOR-FULL-DYNAMIC","full_report":"P-COLLECTOR-FULL-DYNAMIC","odds_state":"P-COLLECTOR-ODDS-DYNAMIC","odds_report":"P-COLLECTOR-ODDS-DYNAMIC","odds_refresh":"P-COLLECTOR-ODDS-DYNAMIC","corpus_report":"P-REPORT-24H","corpus_manifest":"P-REPORT-24H","deployment_manifest":"P-DEPLOY-60","model_catalog":"P-CATALOG-60"}
    sources={}
    for key,policy in policies.items():
        path,digest,sealed_root=entry("sources",key)
        try:payload=json.loads(_retained_source_read(path,key))
        except (UnicodeDecodeError,json.JSONDecodeError) as exc:raise RuntimeError("generated live evidence source invalid") from exc
        if type(payload) is not dict:raise RuntimeError("generated live evidence source invalid")
        schema=payload.get("schema_version")
        time_field=None if key in {"full_state","corpus_manifest","model_catalog"} else ("updated_at" if key=="odds_state" else "generated_at")
        evidence_root=layout["dirs"]["current_evidence"]
        allowlisted=sealed_root or (evidence_root if path.is_relative_to(evidence_root) else path.parent)
        serialization=(JsonSerializationPolicy.PRODUCER_COMPACT_CANONICAL_LINE if key=="model_catalog" else JsonSerializationPolicy.PRODUCER_PRETTY_SORTED)
        max_bytes=_MAX_FULL_SOURCE_BYTES if key in _FULL_SOURCE_KEYS else _MAX_CONTROL_BYTES
        sources[key]=SourceConfig(path,allowlisted,"producer_report",str(schema or "shadow_autopilot_refresh_report"),f"operator_ui.{key}",policy,"Exact producer evidence only.",JsonSource("schema_version" if schema else None,schema,tuple(payload),time_field,identity_fields=("schema_version",) if schema else (),max_items=100000,timestamp_syntax=TimestampSyntax.AWARE_ISO8601,serialization_policy=serialization,authority_observed_at=live["observed_at"] if key=="model_catalog" else None),max_bytes=max_bytes,expected_sha256=digest)
    raw_sources={}
    for key in live.get("raw_sources",{}):
        path,digest,_=entry("raw_sources",key);policy="P-CATALOG-60" if key.startswith("model_") else "P-REPORT-24H"
        item=live["raw_sources"][key];digest_only=key in _DIGEST_ONLY_RAW_KEYS
        raw_sources[key]=RawSourceConfig(path,path.parent,"fixed_file",key,f"operator_ui.{key}",policy,"Exact authenticated file identity only." if digest_only else "Exact fixed bytes only.",max_bytes=64*1024*1024 if digest_only else 16_777_216,expected_sha256=digest,expected_bytes=item.get("bytes") if digest_only else None,digest_only=digest_only)
    unit_values={}
    for key in ("full_timer","full_service","odds_timer","odds_service"):
        path,digest,_=entry("units",key);raw=_retained_read(path)
        if hashlib.sha256(raw).hexdigest()!=digest:raise RuntimeError("installed unit identity mismatch")
        unit_values[key]=raw;unit_values[f"{key}_sha256"]=digest
    try:observed=datetime.fromisoformat(live["observed_at"].replace("Z","+00:00"))
    except (AttributeError,ValueError) as exc:raise RuntimeError("installed unit observation time invalid") from exc
    status=live.get("service_status",{})
    for lane in ("full","odds"):
        if not isinstance(status.get(lane),dict) or set(status[lane])!={"unit_name","active_state","sub_state","exec_main_pid"}:raise RuntimeError("installed unit status invalid")
        expected={"full":"shadow-autopilot.service","odds":"shadow-autopilot-odds-capture.service"}[lane]
        if status[lane]["unit_name"]!=expected:raise RuntimeError("installed unit status invalid")
        unit_values.update({f"{lane}_unit_name":status[lane]["unit_name"],f"{lane}_active_state":status[lane]["active_state"],f"{lane}_sub_state":status[lane]["sub_state"],f"{lane}_exec_main_pid":status[lane]["exec_main_pid"]})
    units=InstalledUnits(**unit_values,observed_at=observed,working_directory=live["working_directory"])
    reader=OperatorEvidenceReader(sources,raw_sources=raw_sources)
    return LiveEvidenceAdapters(reader,units=units,upcoming_races=UpcomingRaceSource(layout["paths"]["current_index.json"],layout["dirs"]["current_evidence"]),prediction_bundles=PredictionBundleSource(layout["dirs"]["prediction_bundles"]))


def _runner(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row,Mapping):raise R3Rejected("RACE_EVIDENCE_INVALID")
    box=row.get("box_number",row.get("box")); name=row.get("display_name",row.get("name")); identity=row.get("identity")
    value={"box":box,"name":name,"identity":identity}
    native=row.get("source_native_runner_id")
    if native is not None:value["source_native_runner_id"]=native
    return value


class _FixedDispatcher:
    """One fixed in-process dispatch lane; JobStore.claim_attempt owns execution."""
    def __init__(self, store: JobStore, worker: WorkerConfig, clock, runner=None):
        self._store,self._worker,self._clock,self._runner=store,worker,clock,runner or run_once
        self._queue: queue.Queue[tuple[str,Any]] = queue.Queue(maxsize=256)
        self._pending:set[str]=set(); self._pending_lock=threading.Lock()
        self._thread=threading.Thread(target=self._serve,name="operator-ui-r3-dispatch",daemon=True)
        self._thread.start()
    def __call__(self, job_id: str, confirm) -> None:
        with self._pending_lock:
            if job_id in self._pending:return
            self._pending.add(job_id)
        try:self._queue.put_nowait((job_id,confirm))
        except BaseException:
            with self._pending_lock:self._pending.discard(job_id)
            raise
    def _serve(self) -> None:
        while True:
            job_id,confirm=self._queue.get()
            try:self._runner(self._store,job_id,self._worker,now=self._clock,confirm_audit=confirm)
            except BaseException as exc:
                try:
                    job=self._store.get(job_id)
                    if job.phase is Phase.WAITING_FOR_CLAIM and not job.attempt_claimed:
                        self._store.transition(job_id,Phase.FAILED,now=self._clock(),status="FAILED",reason="DISPATCH_FAILED",facts={"error":type(exc).__name__},confirm_audit=confirm)
                except BaseException as closure_exc:
                    warnings.warn(f"Operator UI dispatch and durable closure failed: {type(exc).__name__}/{type(closure_exc).__name__}",RuntimeWarning,stacklevel=2)
            finally:
                with self._pending_lock:self._pending.discard(job_id)
                self._queue.task_done()


def _build_r3_services(app: Flask, profile: str) -> R3Services:
    if profile=="fixture-v1" and app.config.get("TESTING") is not True:raise ValueError("fixture-v1 requires TESTING")
    layout=_repository_layout() if profile=="repository-v1" else None
    base=(_REPOSITORY_ROOT/_PROFILES[profile]).absolute() if profile=="fixture-v1" else layout["base"]
    if layout is None:
        try: base.relative_to(_REPOSITORY_ROOT)
        except ValueError as exc: raise RuntimeError("fixed R3 root escaped repository") from exc
    _directory(base)
    if layout is not None:
        paths=layout["paths"]; dirs=layout["dirs"]
    else:
        paths={name:base/name for name in ("audit.sqlite3","canonical.sqlite3","jobs.sqlite3","current_index.json")}
        dirs={name:base/name for name in ("current_evidence","prediction_bundles","collector_requests","capture_evidence_a","capture_evidence_b")}
    for name,path in paths.items():
        if name!="jobs.sqlite3":_regular(path)
        else:
            try: job_info=path.lstat()
            except FileNotFoundError: job_info=None
            except OSError as exc: raise RuntimeError("fixed R3 job store unavailable") from exc
            if job_info is not None and (path.is_symlink() or not stat.S_ISREG(job_info.st_mode)):
                raise RuntimeError("fixed R3 job store unsafe")
    for path in dirs.values():_directory(path)
    configured_audit=Path(str(app.config.get("OPERATOR_UI_AUDIT_DB_PATH",""))).absolute()
    configured_canonical=Path(str(app.config.get("DATABASE_PATH",""))).absolute()
    if configured_audit!=paths["audit.sqlite3"] or configured_canonical!=paths["canonical.sqlite3"]:raise RuntimeError("fixed R3 protected store binding mismatch")
    if layout is None:
        protected=(*paths.values(),*dirs.values())
        if len({p.resolve() for p in protected}) != len(protected):raise RuntimeError("fixed R3 paths overlap")

    product_root=layout["source_root"] if layout is not None else _REPOSITORY_ROOT
    config_path=layout["artifacts"]["config"] if layout is not None else product_root/"configs/prediction/manual-default.json"
    model_path=layout["artifacts"]["model"] if layout is not None else product_root/"artifacts/frozen_models/market_form_residual_v1/model.json"
    manifest_path=layout["artifacts"]["manifest"] if layout is not None else product_root/"artifacts/frozen_models/market_form_residual_v1/manifest.json"
    schema_path=layout["artifacts"]["schema"] if layout is not None else product_root/"configs/prediction/schemas/market_form_residual_v1.schema.json"
    for artifact in (config_path,model_path,manifest_path,schema_path):
        try:artifact.resolve(strict=True).relative_to(product_root.resolve(strict=True))
        except (OSError,ValueError) as exc:raise RuntimeError("fixed R3 artifact escaped repository") from exc
        _regular(artifact)
    resolved_model="market_form_residual_v1"; model_sha=_sha(model_path); manifest_sha=_sha(manifest_path); schema_sha=_sha(schema_path)
    config=json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version")!="on_demand_prediction_config_v1" or config.get("model")!=resolved_model:raise RuntimeError("fixed R3 config divergent")
    choice=ServerChoice(config_path,"manual-default",_sha(config_path),resolved_model,model_sha,manifest_sha,schema_sha,model_path,manifest_path,schema_path)
    captures=(dirs["capture_evidence_a"],) if profile=="repository-v1" else (dirs["capture_evidence_a"],dirs["capture_evidence_b"])
    worker=WorkerConfig(layout["pinned_python"] if layout is not None else Path(sys.executable),product_root,{"latest-research":choice},paths["canonical.sqlite3"],dirs["prediction_bundles"],captures,dirs["collector_requests"],paths["current_index.json"],dirs["current_evidence"],1.0,45.0,90.0,2.0)
    store=JobStore(paths["jobs.sqlite3"],separate_from=(paths["audit.sqlite3"],paths["canonical.sqlite3"]))

    def clock() -> datetime:return datetime.now(timezone.utc)
    def resolve(selected: Mapping[str,str], now: datetime) -> ResolvedSubmission:
        model_id, config_id, odds_source = _R3_ALLOWED_SELECTION
        if (selected.get("model_id"),selected.get("config_id"),selected.get("odds_source_id")) != _R3_ALLOWED_SELECTION:raise R3Rejected("SELECTION_NOT_ALLOWLISTED")
        try:view=bounded_current_race_index(current_time=now,timeout_seconds=1.0,index_path=paths["current_index.json"],evidence_root=dirs["current_evidence"],max_age_seconds=300,return_verified_view=True)
        except (CaptureOneRejected,PredictionBlocked) as exc:raise R3Rejected(getattr(exc,"code","RACE_EVIDENCE_INVALID")) from exc
        if not isinstance(view,VerifiedCurrentRaceIndex):raise R3Rejected("RACE_EVIDENCE_INVALID")
        matches=[row for row in view.races if row.get("race_id")==selected.get("race_id")]
        if len(matches)!=1:raise R3Rejected("RACE_ID_MISSING_OR_AMBIGUOUS")
        race=matches[0]; runners=tuple(_runner(row) for row in race.get("runners",()))
        jump=race.get("jump_datetime",race.get("jump_timestamp"))
        job_input=JobInput(str(race["race_id"]),str(jump),str(race["runner_set_sha256"]),model_id,resolved_model,model_sha,manifest_sha,schema_sha,config_id,choice.config_sha256,odds_source,runners)
        job_input.fields()
        return ResolvedSubmission(job_input,runners)
    dispatcher=_FixedDispatcher(store,worker,clock)
    return R3Services(store,resolve,dispatcher,build_verified_bundle_reader(dirs["prediction_bundles"],store),clock=clock)


def bind_configured_r3(app: Flask) -> bool:
    selector=app.config.get(R3_PROFILE_KEY,False)
    if selector in (False,None,"","disabled"):return False
    if selector is True:selector="repository-v1"
    if not isinstance(selector,str) or selector not in _PROFILES:raise ValueError("unknown finite R3 profile")
    return install_r3_api(app,_build_r3_services(app,selector))


def configure_r3_startup(app: Flask) -> bool:
    """Apply fixed protected-store bindings before connected security installs."""
    selector=app.config.get(R3_PROFILE_KEY,False)
    if selector in (False,None,"","disabled"):return False
    if selector is True:selector="repository-v1"
    if not isinstance(selector,str) or selector not in _PROFILES:raise ValueError("unknown finite R3 profile")
    if selector=="fixture-v1" and app.config.get("TESTING") is not True:raise ValueError("fixture-v1 requires TESTING")
    if selector=="repository-v1":
        layout=_repository_layout(); audit=layout["paths"]["audit.sqlite3"]; canonical=layout["paths"]["canonical.sqlite3"]
        deployed=layout["deployment"]
        if (app.config.get("OPERATOR_UI_DEPLOYED_COMMIT"),app.config.get("OPERATOR_UI_DEPLOYED_TREE"),app.config.get("OPERATOR_UI_DEPLOYED_VERSION"),app.config.get("OPERATOR_UI_DEPLOYED_PROFILE")) != (deployed["source_commit"],deployed["source_tree"],deployed["ui_version"],deployed["profile_id"]):raise RuntimeError("generated repository-v1 deployment identity mismatch")
        if int(app.config.get("OPERATOR_UI_LEVEL",1))>=2:app.config[CONFIG_KEY]=_configured_live(layout)
    else:
        base=(_REPOSITORY_ROOT/_PROFILES[selector]).absolute(); audit=base/"audit.sqlite3"; canonical=base/"canonical.sqlite3"
    app.config["OPERATOR_UI_AUDIT_DB_PATH"]=str(audit)
    app.config["DATABASE_PATH"]=str(canonical)
    return True


__all__=["CONFIG_KEY","R3_PROFILE_KEY","bind_configured_live_evidence","bind_configured_r3","configure_r3_startup"]
