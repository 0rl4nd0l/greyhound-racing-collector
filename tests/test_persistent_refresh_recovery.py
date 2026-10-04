"""Retained failed refreshes block dispatch until authenticated new publication."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from race_collection import persistent_collector as collector
from race_collection.live_freshness_contract import classify_refresh_outage, create_once
from tests.test_scheduled_refresh_outage import outage
from tests.test_schedule_change_refresh_isolation import changed_report
from tests.test_empty_eligible_refresh import publish
from tests.test_persistent_dispatch_verdict import evidence


def mixed_outage(tmp_path):
    output, plan, rid, root, refresh = outage(tmp_path)
    report = changed_report(tmp_path)
    candidate = dict(report['selected_races'][0], race_url='https://www.thedogs.com.au/racing/gunnedah/2026-07-19/7',
                     race_id='Race 7 - GUNN - 2026-07-19', race_number=7)
    report['selected_races'].append(candidate)
    report['selected_count'] += 1
    report['downloads'].append(dict(race_url=candidate['race_url'],success=False,result=dict(
        success=False,error='Source HTTP status 502',source_http_status=502,source_retry_headers={'date':'retained'})))
    report['sidecar_metadata_coverage']['races'].append(dict(race_url=candidate['race_url'],
        race_id=candidate['race_id'],csv_path=None,sidecar_path=None,weather_track_rejected_reasons=['accepted_csv_missing']))
    report.update(status='ACQUISITION_INCOMPLETE',reason='unisolated_selected_race_acquisition_failure')
    refresh.write_text(json.dumps(report))
    return output, plan, rid, root, refresh, report


def test_mixed_failure_preserves_every_selected_race_and_never_publishes_subset(tmp_path):
    output, plan, rid, root, refresh, report = mixed_outage(tmp_path)
    before = refresh.read_bytes()
    classified = classify_refresh_outage(plan['evidence_root'],rid)
    assert classified and classified['upstream_statuses']==[502]
    assert classified['request_retries_added']==0
    assert len(report['selected_races'])==len(report['downloads'])==3
    assert publish(tmp_path,tmp_path/'runtime/state.json',report,'incomplete')['status']=='REJECTED'
    assert refresh.read_bytes()==before


@pytest.mark.parametrize('defect',['denial','guidance','unknown','missing_status','missing_race','partial_csv',
                                   'hidden_failure','count','snapshot','unproved_local','budget','changed_hash'])
def test_mixed_recovery_never_excuses_unknown_denied_or_incomplete_proof(tmp_path,defect):
    output, plan, rid, root, refresh, report = mixed_outage(tmp_path)
    item=report['downloads'][-1]['result']
    if defect=='denial':item.update(source_http_status=429,error='Source HTTP status 429')
    elif defect=='guidance':item['source_retry_headers']['Retry-After']='60'
    elif defect=='unknown':item['error']='unknown'
    elif defect=='missing_status':item.pop('source_http_status')
    elif defect=='missing_race':report['selected_races'].pop()
    elif defect=='partial_csv':report['sidecar_metadata_coverage']['races'][-1]['csv_path']='partial.csv'
    elif defect=='hidden_failure':item['source_failure_category']='request_cap_exhausted'
    elif defect=='count':report['accepted_csv_count']+=1
    elif defect=='snapshot':report['shared_sportsbet_snapshot']['status']='DENIED'
    elif defect=='unproved_local':report['downloads'][1]['result'].pop('schedule_change_evidence')
    else:
        checkpoint=json.loads((root/'phase-checkpoint.json').read_bytes())
        checkpoint['phases'][0]['budget_exceeded' if defect=='budget' else 'result_sha256']=True if defect=='budget' else '0'*64
        (root/'phase-checkpoint.json').write_text(json.dumps(checkpoint))
    refresh.write_text(json.dumps(report))
    assert classify_refresh_outage(plan['evidence_root'],rid) is None


def deferred_fixture(tmp_path, *, empty_mixed=False):
    output,plan,rid,root,refresh=outage(tmp_path)
    if empty_mixed:
        from tests.test_mixed_empty_refresh_outage import empty_mixed as build_report
        refresh.write_text(json.dumps(build_report(tmp_path)))
    record,runtime=evidence(Path(plan['evidence_root']),action='LIVE_PHASE_FAILED',status='FAILED',code=2)
    path=runtime/'service-terminals'/('a'*32+'.json')
    terminal=json.loads(path.read_bytes());terminal.update(run_id=rid,output_dir=str(root),final_verdict='NEEDS_MORE_AUTOMATION')
    path.write_text(json.dumps(terminal))
    observed='2099-01-01T12:00:00+00:00'
    retained=output/'refresh-deferrals'/(rid+'.json')
    create_once(retained,{**classify_refresh_outage(plan['evidence_root'],rid),'observed_at':observed,'failed_cycle_count':1})
    return output,plan,record,retained


def test_owner_recognizes_only_authenticated_native_deferral_not_success(tmp_path,monkeypatch):
    monkeypatch.setattr(collector,'now',lambda:datetime.fromisoformat('2099-01-01T12:02:00+00:00'))
    output,plan,record,retained=deferred_fixture(tmp_path)
    verdict=collector.classify_native_dispatch(plan['evidence_root'],record,'b'*64,output=output)
    assert verdict['disposition']=='REFRESH_OUTAGE'
    assert verdict['refresh_deferral']==collector.reference(retained)
    retained.unlink()
    with pytest.raises(ValueError,match='terminal_failure'):
        collector.classify_native_dispatch(plan['evidence_root'],record,'b'*64,output=output)


@pytest.mark.parametrize('defect',['hash','naive_time','future_time','cap','interrupted','wrong_cycle'])
def test_owner_rejects_unverified_deferral(tmp_path,defect,monkeypatch):
    monkeypatch.setattr(collector,'now',lambda:datetime.fromisoformat('2099-01-01T12:02:00+00:00'))
    output,plan,record,retained=deferred_fixture(tmp_path)
    value=json.loads(retained.read_bytes())
    if defect=='hash':value['refresh_sha256']='0'*64
    elif defect=='naive_time':value['observed_at']='2099-01-01T12:00:00'
    elif defect=='future_time':value['observed_at']='2100-01-01T12:00:00+00:00'
    elif defect=='cap':value['failed_cycle_count']=3
    elif defect=='interrupted':
        path=Path(plan['evidence_root'])/'shadow_autopilot_daemon_runtime/service-lifecycles'/('a'*32+'.json')
        lifecycle=json.loads(path.read_bytes());lifecycle['interrupted']=True;path.write_text(json.dumps(lifecycle))
    elif defect=='wrong_cycle':value['run_id']='other'
    retained.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        collector.classify_native_dispatch(plan['evidence_root'],record,'b'*64,output=output)


def test_pending_outage_requires_new_verified_index_and_survives_reload(tmp_path,monkeypatch):
    output,plan,record,retained=deferred_fixture(tmp_path)
    current=datetime.fromisoformat('2099-01-01T12:02:00+00:00')
    monkeypatch.setattr(collector,'now',lambda:current)
    from race_collection import synchronous_manual_capture as capture
    observed=['2099-01-01T11:59:59+00:00']
    calls=[]
    def view(**kwargs):
        calls.append(kwargs)
        if observed[0]=='EXPIRED':raise capture.CaptureOneRejected('CURRENT_INDEX_STALE')
        return SimpleNamespace(source_generated_at=observed[0])
    monkeypatch.setattr(capture,'bounded_current_race_index',view)
    refs=[collector.reference(retained)]
    assert collector.refresh_outage_pending(output,plan,refs,'b'*64)
    assert collector.refresh_outage_pending(output,plan,json.loads(json.dumps(refs)),'b'*64)
    observed[0]='EXPIRED'
    assert collector.refresh_outage_pending(output,plan,refs,'b'*64)
    monkeypatch.setattr(capture,'bounded_current_race_index',lambda **kwargs: (_ for _ in ()).throw(capture.CaptureOneRejected('CURRENT_INDEX_INVALID')))
    with pytest.raises(capture.CaptureOneRejected):collector.refresh_outage_pending(output,plan,refs,'b'*64)
    monkeypatch.setattr(capture,'bounded_current_race_index',view)
    observed[0]='2099-01-01T12:01:00+00:00'
    assert not collector.refresh_outage_pending(output,plan,refs,'b'*64)
    assert calls[-1]['max_age_seconds']==270 and calls[-1]['return_verified_view'] is True
    current += timedelta(seconds=300)
    assert collector.refresh_outage_pending(output,plan,refs,'b'*64)  # VerifiedView may still return stale data.
    observed[0]=(current+timedelta(seconds=1)).isoformat()
    assert collector.refresh_outage_pending(output,plan,refs,'b'*64)  # Future timestamps are not fresh.
    retained.write_text('{}')
    with pytest.raises(ValueError):collector.refresh_outage_pending(output,plan,refs,'b'*64)


def test_real_reader_missing_successor_index_stays_pending(tmp_path,monkeypatch):
    output,plan,record,retained=deferred_fixture(tmp_path)
    monkeypatch.setattr(collector,'now',lambda:datetime.fromisoformat('2099-01-01T12:02:00+00:00'))
    # Exercise the real bounded reader's CaptureOneRejected, not a ValueError stub.
    assert collector.refresh_outage_pending(output,plan,[collector.reference(retained)],'b'*64)


def test_supervisor_hold_reaps_child_but_never_observes_or_dispatches(tmp_path,monkeypatch):
    from race_collection.operational_prediction import Supervisor
    scope=SimpleNamespace(end=datetime.now(timezone.utc)+timedelta(hours=1))
    supervisor=Supervisor(tmp_path,{'operational_predictions':True,'frozen_comparison':{'path':'not-read'}},scope)
    observed=[]
    supervisor.observe_comparison_schedule=lambda:observed.append(True)
    supervisor.child=SimpleNamespace(poll=lambda:0,returncode=0)
    supervisor.log=SimpleNamespace(close=lambda:None)
    supervisor.tick(allow_dispatch=False)
    assert supervisor.child is None and not observed


from tests.test_persistent_collector import owner_case, retain_inventory, finish_native
from tests.test_persistent_native import backend


@pytest.mark.parametrize('empty_mixed', [False, True])
def test_daily_owner_holds_health_preserves_cadence_and_restarts_without_reset(owner_case,monkeypatch,empty_mixed):
    c=owner_case
    retained_output,plan,_,retained=deferred_fixture(c.output/'failure-fixture', empty_mixed=empty_mixed)
    c.owner.plan['evidence_root']=plan['evidence_root']
    value=json.loads(retained.read_bytes());value['observed_at']=c.clock[0].isoformat()
    target=c.output/'refresh-deferrals'/retained.name
    create_once(target,value)
    # A durable wrapper failure blocks dispatch even before its terminal is polled.
    assert c.owner.outage_pending()
    c.owner.activate();retain_inventory(c)
    c.owner.tick()
    assert len(c.children)==1  # The installed owner serializes publishers.
    finish_native(c,c.children[0])
    c.owner.tick()
    assert len(c.children)==2
    native_runtime=Path(plan['evidence_root'])/'shadow_autopilot_daemon_runtime'
    invocation=c.children[1].kwargs['env']['INVOCATION_ID']
    for category in ('service-terminals','service-lifecycles'):
        original=json.loads((native_runtime/category/('a'*32+'.json')).read_bytes())
        original['invocation_id']=invocation
        if category=='service-terminals':original['allocation_sha256']=c.prepared['allocation_ref']['sha256']
        create_once(native_runtime/category/(invocation+'.json'),original)
    c.children[1].returncode=2
    from race_collection import synchronous_manual_capture as capture
    source=[(c.clock[0]-timedelta(seconds=5)).isoformat()]
    monkeypatch.setattr(capture,'bounded_current_race_index',lambda **kwargs:SimpleNamespace(source_generated_at=source[0]))
    c.owner.tick()
    health=json.loads((c.output/'persistent-health.json').read_bytes())
    assert health['status']=='HOLD' and health['forecast_admission_ready'] is False
    assert health['reason']=='UPSTREAM_TEMPORARY_UNAVAILABLE'
    assert c.owner.state['completed_lanes']=={'full':1,'odds':0}
    assert c.owner.state['deferred_lanes']=={'full':0,'odds':1}
    assert c.owner.state['dispatches'][1]['native_disposition']=='REFRESH_OUTAGE'
    assert c.owner.state['refresh_failures']==[collector.reference(target)]
    assert len(c.children)==2 and not (c.output/'HALT.json').exists()
    due=dict(c.owner.state['next_due_at'])
    c.owner.drain()
    resumed=collector.DailyOwner(c.cfg,c.prepared)
    resumed.activate()
    assert resumed.state['next_due_at']==due and resumed.outage_pending()
    assert resumed.state['refresh_failures']==[collector.reference(target)]
    c.clock[0]+=timedelta(seconds=30)
    source[0]=c.clock[0].isoformat()
    resumed.tick()
    assert len(c.children)==2
    assert json.loads((c.output/'persistent-health.json').read_bytes())['forecast_admission_ready'] is True
    assert len(list((c.output/'refresh-deferrals').glob('*.json')))==1
