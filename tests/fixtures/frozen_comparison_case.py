"""Invented source evidence through real publication, retention and R3 worker.

Never reads the canonical database, providers, units or live collector state.
Run this module under check_freshness_service.deny_network for a packaged proof.
"""
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sqlite3
import resource
import subprocess
import sys
import threading
import time
from zoneinfo import ZoneInfo

from src.predictor.on_demand import canonical_bytes, sha256_file

ROOT=Path(__file__).resolve().parents[2]


def prepare(root, *, late=False):
    from tests.test_predict_market_form_residual import _write_fixture, RUNNERS
    from scripts.capture_thedogs_market_history import TimedResponse, persist_primary_race_page_evidence
    from scripts.shadow_autopilot_v1 import publish_current_race_index_after_refresh
    from race_collection.synchronous_manual_capture import publish_scheduled_capture_receipts, bounded_current_race_index
    from race_collection.manual_prediction_collector_request import ManualPredictionCollectorProtocol
    from race_collection import scheduled_input_retention as scheduled
    from tests.test_prospective_input_retention import generator_files
    now=datetime.now(ZoneInfo("Australia/Melbourne"))
    # Discovery canonicalizes jumps to minutes. Leave at least 70 seconds for
    # production's T-60 guard while placing comparison strictly after T-120.
    if late and now.second >= 50:
        time.sleep(60-now.second+.1)
        now=datetime.now(ZoneInfo("Australia/Melbourne"))
    observed=now-timedelta(seconds=15)
    jump=(now+timedelta(minutes=2 if late else 9)).replace(second=0,microsecond=0)
    day=jump.date().isoformat()
    rid=f"Race 2 - SAN - {day}"; url=f"https://www.thedogs.com.au/racing/sandown/{day}/2/test"
    evidence=root/"evidence"; source=evidence/"invented-source"
    paths=_write_fixture(source)
    form=source/(rid+".csv"); paths["form_csv"].rename(form)
    metadata=json.loads(paths["sidecar"].read_bytes())
    metadata=json.loads(json.dumps(metadata).replace("2026-07-16",day))
    metadata.update(filename=form.name,accepted_csv_path=str(form),normalization_status="verified",metadata_captured_at=observed.isoformat(),target_distance=515,target_grade="Grade 5")
    metadata["race_info"]["race_time"]=jump.strftime("%H:%M")
    metadata.update(weather="Clear",track_condition="Good",weather_track_metadata_source="open_meteo_forecast_api+sportsbet_pre_race_page",
        weather_track_metadata_source_url={"open_meteo_forecast_api":"https://api.open-meteo.com/v1/forecast","sportsbet_pre_race_page":"https://www.sportsbet.com.au/betting/greyhound-racing/sandown/race-2-123"},
        weather_track_metadata_is_leakage_safe=True)
    metadata["prejump_shadow_metadata"].update(metadata_captured_at=observed.isoformat(),jump_datetime=jump.isoformat(),source_native_race_id="987654321")
    participants=[{"box_number":b,"dog_name":n,"source_native_runner_id":str(99000+b),"scratch_state":"ACTIVE"} for b,n,_,_ in RUNNERS]
    metadata["runner_completeness"]["participants"]=participants
    metadata["runner_completeness_after_canonical_alignment"]={"status":"COMPLETE","runner_count":3,"participants":participants}
    metadata["prejump_shadow_metadata"]["runner_box_name_list"]=participants
    metadata["prejump_shadow_metadata"]["canonical_final_runner_alignment"]["canonical_runner_set_status"]="available"
    raw=source/"raw_exports"/form.name; raw.parent.mkdir(); raw.write_bytes(form.read_bytes())
    metadata.update(raw_export_path=str(raw),raw_content_sha256=sha256_file(raw),raw_content_length=raw.stat().st_size)
    page=TimedResponse(requested_url=url,final_url=url,request_start_utc=observed-timedelta(seconds=1),request_end_utc=observed,status_code=200,headers={},body=b"<html>synthetic pre-race fixture</html>")
    metadata["primary_race_page_evidence"]=persist_primary_race_page_evidence(artifact_root=source,race_discovery_key=rid,response=page,canonical_runner_set={})
    sidecar=Path(str(form)+".metadata.json"); sidecar.write_bytes(canonical_bytes(metadata))
    selected={"date":day,"jump_datetime":jump.isoformat(),"race_id":rid,"race_id_aliases":[rid,f"Race 2 - SANDOWN - {day}"],
        "race_number":2,"source_native_race_id":"987654321","race_time":jump.strftime("%H:%M"),"race_url":url,"venue":"SAN"}
    refresh=source/"refresh_prejump_report.json"
    refresh.write_bytes(canonical_bytes({"status":"SUCCESS","generated_at":observed.isoformat(),"selected_count":1,"selected_races":[selected],
        "sidecar_metadata_coverage":{"schema_version":"prejump_sidecar_metadata_coverage_v1","races":[{"race_url":url,"csv_path":str(form),"sidecar_path":str(sidecar),"source_native_race_id":"987654321","source_native_runner_ids":[p["source_native_runner_id"] for p in participants]}]}}))
    state=evidence/"shadow_autopilot_daemon_runtime/odds_capture_state.json"
    published=publish_current_race_index_after_refresh(state_path=state,evidence_root=evidence,output_dir=source,run_id="synthetic-comparison",source_refresh_report_path=refresh)
    assert published["status"]=="PUBLISHED",published
    index=state.parent/"manual_prediction_current_race_index.json"
    view=bounded_current_race_index(current_time=now,timeout_seconds=5,index_path=index,evidence_root=evidence,max_age_seconds=300,return_verified_view=True)
    protocol=ManualPredictionCollectorProtocol(evidence/"manual_prediction_collector_requests_v1")
    plan_item={"schema_version":"autonomous_live_odds_capture_plan_item_v1","status":"READY_TO_CAPTURE","csv_path":str(form),"sidecar_path":str(sidecar),
        **selected,"race_date":day,"thedogs_source_url":url,"minutes_to_jump":9,"capture_window_minutes":10,"expected_runners":[{"box_number":b,"dog_name":n,"identity":i} for b,n,i,_ in RUNNERS],"blockers":[]}
    attempt=json.loads(paths["capture"].read_bytes())["attempts"][0]
    attempt["validation"]["source_url"]="https://www.sportsbet.com.au/betting/greyhound-racing/sandown/race-2-123"
    attempt.update(race_id=rid,fetch_time=(observed+timedelta(seconds=2)).isoformat(),append_time=(observed+timedelta(seconds=3)).isoformat(),capture_window_minutes=10)
    attempt["validation"]["accepted_place_rows"]=attempt["validation"]["accepted_rows"]
    attempt["append_report"]={"status":"SUCCESS","race_id":rid,"inserted_rows":6,"append_only":True,"capture_timestamp":attempt["append_time"]}
    receipt=publish_scheduled_capture_receipts(protocol=protocol,evidence_root=evidence,collector_run_id="synthetic-comparison",plan_item=plan_item,attempt=attempt,output_dir=source,emitted_at=observed+timedelta(seconds=4))
    assert receipt["status"]=="PUBLISHED",receipt
    db=root/"synthetic_history.db"
    with sqlite3.connect(db) as conn:
        conn.executescript("CREATE TABLE race_metadata(race_id TEXT,race_date TEXT,data_source TEXT,url TEXT); CREATE TABLE dog_race_data(race_id TEXT,dog_name TEXT,finish_position INTEGER,data_source TEXT);")
    static={k:{"path":str(p),"sha256":h} for k,(p,h) in generator_files(root).items()}
    claim=root/"retention-claim";claim.mkdir()
    request={"config":{"static_files":static,"max_bundle_bytes":15_000_000},"config_sha256":"a"*64,
        "context":{"protocol_root":str(protocol.root),"evidence_root":str(evidence),"collector_run_id":"synthetic-comparison","history_source":str(db)},
        "plan_item":plan_item,"attempt":attempt,"receipt_publish":receipt,"cutoff":(jump-timedelta(seconds=60)).isoformat()}
    request_path=claim/"request.json";request_path.write_bytes(canonical_bytes(request))
    retained=scheduled._retain(request_path)
    assert retained["status"]=="RETAINED",retained
    (claim/"terminal.json").write_bytes(canonical_bytes({**retained,"config_sha256":"a"*64,"accepted_at":datetime.now(ZoneInfo("Australia/Melbourne")).isoformat()}))
    registry=ROOT/"artifacts/research_comparison/frozen_20260924/registry.json"
    comparison={"schema_version":"frozen_four_way_comparison_plan_v1","status":"SYNTHETIC_REHEARSAL_ONLY","authority_reference":"invented offline test only",
        "programme_root":str(root/"comparison-programme"),
        "activated_at":(now-timedelta(days=2)).isoformat(),"starts_at":(now-timedelta(days=1)).isoformat(),"ends_at":(now+timedelta(days=1)).isoformat(),
        "race_ids":[rid],"decision_seconds_before_jump":120,"quote_lead_seconds":[120,600],"denied_history_intervals":[["2026-07-15","2026-10-31"]],
        "candidate_registry":{"path":str(registry),"sha256":sha256_file(registry)}}
    config=root/"comparison-plan.json";config.write_bytes(canonical_bytes(comparison))
    return dict(root=root,evidence=evidence,protocol=protocol,index=index,view=view,db=db,retained=retained,retained_root=claim/"bundle",comparison=config,jump=jump,race_id=rid)


def execute(case, *, comparison=True, output_name="predictions"):
    from src.operator_ui.job_store import JobInput, JobStore, OperationalIndexProvenance, Phase, resolve_audit_confirmation
    from src.operator_ui.prediction_worker import WorkerConfig, ServerChoice, run_once
    from src.operator_ui.r3_api import finalize_producer_bundle
    from scripts.predict_race_now import _request_race, _request_expected_runners
    from src.predictor.on_demand import resolve_model,sealed_runner_set_sha256
    clock=lambda:datetime.now(ZoneInfo("Australia/Melbourne"))
    root=case["root"];target=case["view"].races[0];model=resolve_model("latest-research")
    config=ROOT/"configs/prediction/manual-default.json"
    choice=ServerChoice(config,"manual-default",sha256_file(config),model.resolved,model.model_sha256,model.manifest_sha256,model.schema_sha256,model.model_path,model.manifest_path,model.schema_path)
    runners=_request_expected_runners(target)
    prediction_hash=sealed_runner_set_sha256(_request_race(target,race_id=case["race_id"],jump=case["jump"]),runners)
    inp=JobInput(case["race_id"],case["jump"].isoformat(),target["runner_set_sha256"],"latest-research",model.resolved,model.model_sha256,model.manifest_sha256,model.schema_sha256,
        "manual-default",choice.config_sha256,"receipt",tuple({"box":r["box_number"],"name":r["display_name"],"identity":r["identity"],"source_native_runner_id":r["source_native_runner_id"]} for r in runners),
        OperationalIndexProvenance.from_verified_current_race_index(case["view"]),case["retained"]["manifest_sha256"],prediction_runner_set_sha256=prediction_hash)
    output=root/output_name
    worker=WorkerConfig(Path(sys.executable),ROOT,{"latest-research":choice},case["db"],output,(case["evidence"],),case["protocol"].root,case["index"],case["evidence"],5,45,90,2,
        retained_input_bindings={case["race_id"]:{"path":str(case["retained_root"]),"manifest_sha256":case["retained"]["manifest_sha256"]}},
        comparison_plan=case["comparison"] if comparison else None,comparison_plan_sha256=sha256_file(case["comparison"]) if comparison else None)
    authority=object();store=JobStore(root/(output_name+"-jobs.db"),separate_from=(case["db"],),verifier_authority=authority)
    def confirm(intent):
        raw=canonical_bytes(intent); h=hashlib.sha256(raw).hexdigest();path=root/"audit"/(h+".json");path.parent.mkdir(exist_ok=True)
        if not path.exists():path.write_bytes(raw)
        return resolve_audit_confirmation(intent,h)
    job=store.create(actor_identity="synthetic-rehearsal",actor_level=2,operation="operational_prediction",idempotency_key=output_name+"-synthetic-test-key",job_input=inp,now=clock(),confirm_audit=confirm)
    for phase,status,reason in ((Phase.VALIDATED,"VALID","validated"),(Phase.WAITING_FOR_CLAIM,"WAITING","ready")):
        job=store.transition(job.job_id,phase,now=clock(),status=status,reason=reason,confirm_audit=confirm)
    before=time.monotonic()
    usage_before=resource.getrusage(resource.RUSAGE_CHILDREN)
    sampled_peak=[0]; stop=threading.Event(); monitors=[]
    def measured_popen(*args,**kwargs):
        child=subprocess.Popen(*args,**kwargs)
        def monitor():
            while not stop.wait(.01):
                try:
                    for line in Path(f'/proc/{child.pid}/status').read_text().splitlines():
                        if line.startswith('VmHWM:'): sampled_peak[0]=max(sampled_peak[0],int(line.split()[1]))
                except FileNotFoundError: return
        thread=threading.Thread(target=monitor,daemon=True);thread.start();monitors.append(thread)
        return child
    # The real fixed argv, retained descriptors, subprocess and reader execute.
    try:
        job=run_once(store,job.job_id,worker,now=clock,confirm_audit=confirm,popen=measured_popen)
    finally:
        stop.set()
        for thread in monitors: thread.join(timeout=1)
    subprocess_seconds=time.monotonic()-before
    usage_after=resource.getrusage(resource.RUSAGE_CHILDREN)
    job=finalize_producer_bundle(output,store,job,capability=authority,now=clock(),confirm_audit=confirm)
    result={"phase":job.phase.value,"reason":job.reason,"subprocess_seconds":subprocess_seconds,"output_root":str(output),
        "subprocess_cpu_seconds":usage_after.ru_utime+usage_after.ru_stime-usage_before.ru_utime-usage_before.ru_stime,
        "sampled_subprocess_peak_rss_kib":sampled_peak[0]}
    if job.phase is not Phase.PREDICTION_READY:
        result["job"]=str(job)
    (root/(output_name+"-execution.json")).write_bytes(canonical_bytes(result))
    return result


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument('--comparison-first',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from scripts.check_freshness_service import deny_network
    deny_network()
    import socket
    for family in (socket.AF_INET,socket.AF_INET6):
        try:socket.socket(family,socket.SOCK_STREAM)
        except PermissionError:continue
        raise RuntimeError("network not denied")
    case=prepare(a.output)
    case["db"].unlink()  # Genuine retained consumer must not fall back to DB.
    if a.comparison_first:
        compared=execute(case);baseline=execute(case,comparison=False,output_name="baseline")
    else:
        baseline=execute(case,comparison=False,output_name="baseline");compared=execute(case)
    print(json.dumps({"network_denied":"kernel seccomp IPv4/IPv6 inherited by children","baseline":baseline,"comparison":compared}))
