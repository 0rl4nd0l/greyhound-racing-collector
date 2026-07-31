"""Finite repository-owned startup composition for connected Operator UI."""
from __future__ import annotations

import hashlib
import json
import queue
import stat
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from flask import Flask

from race_collection.synchronous_manual_capture import CaptureOneRejected, VerifiedCurrentRaceIndex, bounded_current_race_index
from src.predictor.on_demand import PredictionBlocked
from .api import register_level_1_provider
from .job_store import JobInput, JobStore, Phase
from .live_adapters import LiveEvidenceAdapters
from .prediction_worker import ServerChoice, WorkerConfig, run_once
from .r3_api import R3Rejected, R3Services, ResolvedSubmission, build_verified_bundle_reader, install_r3_api

CONFIG_KEY = "OPERATOR_UI_LIVE_EVIDENCE_ADAPTERS"
R3_PROFILE_KEY = "OPERATOR_UI_R3_PROFILE"
_BOUND_KEY = "operator_ui_live_evidence_bound"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_METHODS = (("upcoming_races","upcoming"),("race_detail","race_detail"),("recent_predictions","recent_predictions"),("prediction_detail","prediction_detail"),("collector","collector"),("corpus","corpus"),("models","models"),("system","system"))
_PROFILES = {
    "repository-v1": "var/operator_ui/r3",
    "fixture-v1": "tests/operator_ui/fixtures/r3_runtime",
}


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


def _regular(path: Path) -> None:
    try: info=path.lstat()
    except OSError as exc: raise RuntimeError("fixed R3 runtime unavailable") from exc
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):raise RuntimeError("fixed R3 runtime unsafe")


def _directory(path: Path) -> None:
    try: info=path.lstat()
    except OSError as exc: raise RuntimeError("fixed R3 runtime unavailable") from exc
    if path.is_symlink() or not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode)&0o022:raise RuntimeError("fixed R3 runtime unsafe")


def _sha(path: Path) -> str:
    _regular(path); return hashlib.sha256(path.read_bytes()).hexdigest()


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
        self._thread=threading.Thread(target=self._serve,name="operator-ui-r3-dispatch",daemon=True)
        self._thread.start()
    def __call__(self, job_id: str, confirm) -> None:
        self._queue.put_nowait((job_id,confirm))
    def _serve(self) -> None:
        while True:
            job_id,confirm=self._queue.get()
            try:self._runner(self._store,job_id,self._worker,now=self._clock,confirm_audit=confirm)
            except BaseException as exc:
                try:
                    job=self._store.get(job_id)
                    if job.phase is Phase.WAITING_FOR_CLAIM and not job.attempt_claimed:
                        self._store.transition(job_id,Phase.FAILED,now=self._clock(),status="FAILED",reason="DISPATCH_FAILED",facts={"error":type(exc).__name__},confirm_audit=confirm)
                except BaseException:pass
            finally:self._queue.task_done()


def _build_r3_services(app: Flask, profile: str) -> R3Services:
    if profile=="fixture-v1" and app.config.get("TESTING") is not True:raise ValueError("fixture-v1 requires TESTING")
    base=(_REPOSITORY_ROOT/_PROFILES[profile]).absolute()
    try: base.relative_to(_REPOSITORY_ROOT)
    except ValueError as exc: raise RuntimeError("fixed R3 root escaped repository") from exc
    _directory(base)
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
    if len({p.resolve() for p in (*paths.values(),*dirs.values())}) != len(paths)+len(dirs):raise RuntimeError("fixed R3 paths overlap")

    config_path=_REPOSITORY_ROOT/"configs/prediction/manual-default.json"
    model_path=_REPOSITORY_ROOT/"artifacts/frozen_models/market_form_residual_v1/model.json"
    manifest_path=_REPOSITORY_ROOT/"artifacts/frozen_models/market_form_residual_v1/manifest.json"
    schema_path=_REPOSITORY_ROOT/"configs/prediction/schemas/market_form_residual_v1.schema.json"
    for artifact in (config_path,model_path,manifest_path,schema_path):
        try:artifact.resolve(strict=True).relative_to(_REPOSITORY_ROOT.resolve(strict=True))
        except (OSError,ValueError) as exc:raise RuntimeError("fixed R3 artifact escaped repository") from exc
        _regular(artifact)
    resolved_model="market_form_residual_v1"; model_sha=_sha(model_path); manifest_sha=_sha(manifest_path); schema_sha=_sha(schema_path)
    config=json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version")!="on_demand_prediction_config_v1" or config.get("model")!=resolved_model:raise RuntimeError("fixed R3 config divergent")
    choice=ServerChoice(config_path,"manual-default",_sha(config_path),resolved_model,model_sha,manifest_sha,schema_sha,model_path,manifest_path,schema_path)
    worker=WorkerConfig(Path(sys.executable),_REPOSITORY_ROOT,{"latest-research":choice},paths["canonical.sqlite3"],dirs["prediction_bundles"],(dirs["capture_evidence_a"],dirs["capture_evidence_b"]),dirs["collector_requests"],paths["current_index.json"],dirs["current_evidence"],1.0,45.0,90.0,2.0)
    store=JobStore(paths["jobs.sqlite3"],separate_from=(paths["audit.sqlite3"],paths["canonical.sqlite3"]))

    def clock() -> datetime:return datetime.now(timezone.utc)
    def resolve(selected: Mapping[str,str], now: datetime) -> ResolvedSubmission:
        if (selected.get("model_id"),selected.get("config_id"),selected.get("odds_source_id")) != ("latest-research","manual-default","auto"):raise R3Rejected("SELECTION_NOT_ALLOWLISTED")
        try:view=bounded_current_race_index(current_time=now,timeout_seconds=1.0,index_path=paths["current_index.json"],evidence_root=dirs["current_evidence"],max_age_seconds=300,return_verified_view=True)
        except (CaptureOneRejected,PredictionBlocked) as exc:raise R3Rejected(getattr(exc,"code","RACE_EVIDENCE_INVALID")) from exc
        if not isinstance(view,VerifiedCurrentRaceIndex):raise R3Rejected("RACE_EVIDENCE_INVALID")
        matches=[row for row in view.races if row.get("race_id")==selected.get("race_id")]
        if len(matches)!=1:raise R3Rejected("RACE_ID_MISSING_OR_AMBIGUOUS")
        race=matches[0]; runners=tuple(_runner(row) for row in race.get("runners",()))
        jump=race.get("jump_datetime",race.get("jump_timestamp"))
        job_input=JobInput(str(race["race_id"]),str(jump),str(race["runner_set_sha256"]),"latest-research",resolved_model,model_sha,manifest_sha,schema_sha,"manual-default",choice.config_sha256,"auto",runners)
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
    base=(_REPOSITORY_ROOT/_PROFILES[selector]).absolute()
    app.config["OPERATOR_UI_AUDIT_DB_PATH"]=str(base/"audit.sqlite3")
    app.config["DATABASE_PATH"]=str(base/"canonical.sqlite3")
    return True


__all__=["CONFIG_KEY","R3_PROFILE_KEY","bind_configured_live_evidence","bind_configured_r3","configure_r3_startup"]
