"""Actual daemon waiting-report shape with an independently active peer."""
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from scripts import shadow_autopilot_daemon as daemon
from tests.operator_ui.test_live_adapters import NOW, actual_payloads, make_live


def waiting_values(tmp_path):
    values = actual_payloads(include_models=False)
    started = NOW - timedelta(seconds=20)
    owner = dict(schema_version="shadow_autopilot_daemon_lock_v1", run_id="fixture_odds_capture",
                 pid=2222, hostname="fixture", started_at=started.isoformat(),
                 output_dir=str(tmp_path / "odds"))
    peer = values["odds_report"]
    peer.update(run_id=owner["run_id"], output_dir=owner["output_dir"], generated_at=started.isoformat(),
                status="RUNNING", final_status="ODDS_CAPTURE_ONLY_RUNNING", lock=owner, lock_path=str(tmp_path/"collector.lock"),
                timing={"process_pid":2222,"service_invocation_id":"b"*32},
                odds_capture_refresh_report={},autopilot_output_dir=None)
    values["odds_state"].update(run_id=owner["run_id"],output_dir=owner["output_dir"],
                updated_at=started.isoformat(), status="RUNNING",final_status="ODDS_CAPTURE_ONLY_RUNNING",autopilot_output_dir=None)
    output = tmp_path / "full"
    output.mkdir()
    base = dict(schema_version="shadow_autopilot_daemon_run_v1", run_id="full-next",
                generated_at=(NOW-timedelta(seconds=10)).isoformat(),output_dir=str(output),
                live_freshness_profile="bounded80-v1",timing={"process_pid":3333,"service_invocation_id":"a"*32})
    daemon.write_json(output / "daemon_run_report.json",base)
    details={"reason":"active_lock_present","lock_path":str(tmp_path/'collector.lock'),"existing_lock":owner}
    values["full_report"] = daemon.write_full_daemon_lock_wait_report(output_dir=output,
        lock_path=tmp_path/'collector.lock',lock_details=details,first_lock=details,
        attempt_count=3,waited_seconds=10,retry_seconds=95,poll_seconds=5)
    return values


def observe(tmp_path, values, *, peer_status=("activating","start",2000), **kwargs):
    adapter=make_live(tmp_path / "observer",values,include_models=False,
        full_status=("activating","start",3000),odds_status=peer_status, **kwargs)
    # Wrapper PIDs deliberately differ from the authenticated collector children.
    if hasattr(adapter._units,"odds_service_invocation_id"):
        adapter._units=replace(adapter._units,full_service_invocation_id="a"*32,odds_service_invocation_id="b"*32)
    return adapter.collector(NOW)


def test_actual_wait_report_preserves_prior_full_state_and_active_peer(tmp_path):
    values=waiting_values(tmp_path)
    assert values["full_state"]["last_run_id"] != values["full_report"]["run_id"]
    result=observe(tmp_path,values)
    assert result.evidence.status=="AVAILABLE/FRESH"
    assert result.data["lanes"][0]["status"]=="WAITING_FOR_PEER"
    assert result.data["lanes"][1]["status"]=="ACTIVE"


@pytest.mark.parametrize("change",["foreign_run","foreign_pid","foreign_start","foreign_output", "missing_lock", "invocation", "expired", "retry_exhausted", "ordinary_active", "peer_dead", "legacy_profile", "full_invocation", "missing_invocation", "lock_proof"])
def test_wait_report_rejects_unproven_peer_or_expired_wait(tmp_path,change):
    values=waiting_values(tmp_path);report=values["full_report"];peer=values["odds_report"]
    if change=="foreign_run":report["lock_owner_run_id"]="other_odds_capture"
    elif change=="foreign_pid":report["lock_owner_pid"]=5555
    elif change=="foreign_start":report["lock_owner_started_at"]=(NOW-timedelta(seconds=21)).isoformat()
    elif change=="foreign_output":report["lock_owner_output_dir"]=str(tmp_path/'other')
    elif change=="missing_lock":peer.pop('lock')
    elif change=="invocation":peer['timing']['service_invocation_id']='c'*32
    elif change=="expired":report['generated_at']=(NOW-timedelta(seconds=96)).isoformat()
    elif change=="retry_exhausted":report['lock_retry']['waited_seconds']=95
    elif change=="ordinary_active":report.update(status='RUNNING',final_verdict='DAEMON_RUNNING')
    elif change=="legacy_profile":report.pop("live_freshness_profile")
    elif change=="full_invocation":report["timing"]["service_invocation_id"]="c"*32
    elif change=="missing_invocation":peer["timing"].pop("service_invocation_id")
    elif change=="lock_proof":report["lock_retry"]["last_lock"]["existing_lock"]=dict(report["lock_retry"]["last_lock"]["existing_lock"],pid=5555)
    result=observe(tmp_path,values,peer_status=('inactive','dead',0) if change=='peer_dead' else ('activating','start',2000))
    assert result.evidence.status != 'AVAILABLE/FRESH'


def test_profile_wait_report_writer_respects_native_publication_mutex(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import race_collection.live_phase_checkpoint as publication

    values=waiting_values(tmp_path)
    report_path=tmp_path/'full/daemon_run_report.json'
    original=report_path.read_bytes()
    blocked=threading.Event()
    flock=publication.fcntl.flock
    def observed_flock(fd, operation):
        try:return flock(fd,operation)
        except BlockingIOError:
            if operation & publication.fcntl.LOCK_EX:blocked.set()
            raise
    monkeypatch.setattr(publication.fcntl,'flock',observed_flock)
    details=values['full_report']['lock_retry']['last_lock']
    with ThreadPoolExecutor(max_workers=1) as pool:
        with publication.native_publication_lock(tmp_path,exclusive=False):
            future=pool.submit(daemon.write_full_daemon_lock_wait_report,
                output_dir=tmp_path/'full',lock_path=tmp_path/'collector.lock',
                lock_details=details,first_lock=details,attempt_count=4,
                waited_seconds=15,retry_seconds=95,poll_seconds=5)
            assert blocked.wait(2)
            assert report_path.read_bytes()==original
            assert not future.done()
        assert future.result(timeout=2)['lock_retry']['waited_seconds']==15
    assert json.loads(report_path.read_text())['lock_retry']['waited_seconds']==15


@pytest.mark.parametrize('age,index_owner,expected',[(70,'fixture_odds_capture','WAITING_FOR_PEER'),(271,'fixture_odds_capture','DIVERGENT'),(70,'different_odds_capture','DIVERGENT')])
def test_peer_completion_before_next_lock_poll_requires_bound_fresh_publication(tmp_path,monkeypatch,age,index_owner,expected):
    from race_collection.synchronous_manual_capture import VerifiedCurrentRaceIndex
    from src.operator_ui import live_adapters
    values=waiting_values(tmp_path)
    peer=values['odds_report'];state=values['odds_state']
    peer.pop('lock');peer.pop('lock_path')
    peer.update(status='READY',final_status='ODDS_CAPTURE_ONLY_READY',runtime_action='LIVE_COLLECTION_COMPLETE',
        generated_at=(NOW-timedelta(seconds=2)).isoformat(),autopilot_output_dir='artifacts/autopilot-9',
        odds_capture_refresh_report=values['odds_refresh'])
    state.update(status='READY',final_status='ODDS_CAPTURE_ONLY_READY',runtime_action='LIVE_COLLECTION_COMPLETE',
        updated_at=peer['generated_at'],autopilot_output_dir=peer['autopilot_output_dir'],odds_capture_refresh_status='SUCCESS')
    view=VerifiedCurrentRaceIndex('collector_current_race_index_v2',index_owner,
        (NOW-timedelta(seconds=age)).isoformat(),'1'*64,b'packet',(),'refresh.json','2'*64,'3'*64,'4'*64,'5'*64)
    monkeypatch.setattr(live_adapters,'bounded_current_race_index',lambda **kwargs:view)
    result=observe(tmp_path,values,peer_status=('inactive','dead',0),
        upcoming_races=live_adapters.UpcomingRaceSource(tmp_path/'index.json',tmp_path))
    assert result.data['lanes'][0]['status']==expected


@pytest.mark.parametrize('change,expected', [('none','WAITING_FOR_PEER'),('wrong_child','CAPTURE_WINDOW_CLOSED'),('wrong_invocation','CAPTURE_WINDOW_CLOSED'),('missing_invocation','CAPTURE_WINDOW_CLOSED'),('wrong_lock','CAPTURE_WINDOW_CLOSED')])
def test_reverse_cooperation_binds_active_full_child_not_wrapper(tmp_path,monkeypatch,change,expected):
    from race_collection.synchronous_manual_capture import VerifiedCurrentRaceIndex
    from src.operator_ui import live_adapters
    values=actual_payloads(include_models=False)
    full=values['full_report']
    full.update(run_id='full-next',status='RUNNING',final_verdict='DAEMON_RUNNING',
        lock={'schema_version':'shadow_autopilot_daemon_lock_v1','pid':3333,'run_id':'full-next'},
        timing={'process_pid':3333,'service_invocation_id':'a'*32})
    odds=values['odds_report']
    odds.update(status='SKIPPED_LOCK_HELD',final_status='SKIPPED_LOCK_HELD',
        runtime_action='DEFERRED_LOCK_HELD',live_freshness_profile='bounded80-v1',
        deferred_lock_owner={'run_id':'full-next','pid':3333},autopilot_output_dir=None)
    odds.pop('odds_capture_refresh_report')
    if change=='wrong_child':odds['deferred_lock_owner']['pid']=4444
    elif change=='wrong_invocation':full['timing']['service_invocation_id']='c'*32
    elif change=='missing_invocation':full['timing'].pop('service_invocation_id')
    elif change=='wrong_lock':full['lock']['pid']=4444
    view=VerifiedCurrentRaceIndex('collector_current_race_index_v2','full-next',
        (NOW-timedelta(seconds=70)).isoformat(),'1'*64,b'packet',(),'refresh.json','2'*64,'3'*64,'4'*64,'5'*64)
    monkeypatch.setattr(live_adapters,'bounded_current_race_index',lambda **kwargs:view)
    unit_args=dict(repo_path=Path('/srv/app'),timeout_seconds=600,live_freshness=True,
        live_freshness_profile='bounded80-v1',live_freshness_contract=tmp_path/'contract.json')
    result=observe(tmp_path,values,peer_status=('inactive','dead',0),
        upcoming_races=live_adapters.UpcomingRaceSource(tmp_path/'index.json',tmp_path),
        unit_overrides={'full_service':daemon.service_file_text(**unit_args).encode(),
                        'odds_service':daemon.odds_capture_service_file_text(**unit_args).encode()})
    assert result.data['lanes'][1]['status']==expected
