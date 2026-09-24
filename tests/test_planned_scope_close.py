"""Planned admission closure preserves stale native evidence and admits no data."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_freshness_rehearsal as run
from race_collection.live_freshness_contract import digest
from utils.sportsbet_access import SportsbetAccess


def closure_fixture(tmp_path):
    end = datetime(2099, 1, 1, 13, tzinfo=timezone.utc)
    observed = end - timedelta(seconds=5)
    evidence = tmp_path / 'evidence'
    gate = tmp_path / 'gate.json'
    SportsbetAccess(gate).initialize(access_basis={'status': 'permitted', 'reference': 'synthetic'})
    plan = {'operational_predictions': {'enabled': True}, 'ends_at': end.isoformat(),
            'rehearsal_id': 'synthetic', 'evidence_root': str(evidence),
            'lock_path': str(tmp_path / 'collector.lock'), 'sportsbet_access_state': str(gate)}
    scope = SimpleNamespace(value={'synthetic': True}, session=tmp_path / 'session', end=end)
    invocation = 'a' * 32
    report = evidence / 'shadow_autopilot_daemonization_v1_previous_odds_capture/odds_capture_only_daemon_report.json'
    run.atomic_json(report, {'status': 'READY', 'final_status': 'ODDS_CAPTURE_ONLY_READY',
        'runtime_action': 'LIVE_COLLECTION_COMPLETE', 'run_id': 'previous_odds_capture',
        'generated_at': (end-timedelta(seconds=120)).isoformat()})
    marker = scope.session / 'admission-closures' / (invocation+'.json')
    run.atomic_json(marker, {'schema_version': 'live_admission_closed_v1', 'lane': 'odds',
        'rehearsal_id': 'synthetic', 'contract_sha256': digest(scope.value), 'ends_at': end.isoformat(),
        'service_invocation_id': invocation, 'process_pid': 1234,
        'observed_at': (end-timedelta(seconds=60)).isoformat(), 'observed_monotonic': 101.0,
        'required_seconds': 90, 'runtime_action': 'OPERATING_SCOPE_CLOSED'})
    lifecycle = evidence / 'shadow_autopilot_daemon_runtime/service-lifecycles' / (invocation+'.json')
    run.atomic_json(lifecycle, {'invocation_id': invocation, 'wrapper_pid': 1233, 'child_pid': 1234,
        'status': 'COMPLETE', 'children_reaped': True, 'interrupted': False, 'returncode': 2,
        'launch_started_monotonic': 100.0, 'completed_monotonic': 102.0})
    import hashlib
    current = {'observed_at': observed.isoformat(), 'collector_status': 'STALE',
        'index_status': 'AVAILABLE/FRESH', 'authority_status': 'AVAILABLE/FRESH',
        'source_age_seconds': 114.0, 'lock': None,
        'unit_status': {'odds': {'ActiveState': 'failed', 'SubState': 'failed', 'MainPID': '0',
            'ExecMainPID': '1233', 'InvocationID': invocation}},
        'lanes': [{'lane': 'FULL_DAEMON', 'status': 'RECEIPT_READY'},
                  {'lane': 'ODDS_ONLY', 'status': 'STALE', 'phase': 'ODDS_CAPTURE_ONLY_READY',
                   'run_id': 'previous_odds_capture', 'reference_hashes': {'report': hashlib.sha256(report.read_bytes()).hexdigest()}}]}
    return plan, scope, current, marker, lifecycle, report


def test_planned_close_preserves_native_stale_and_has_no_new_data(tmp_path):
    plan, scope, current, *_ = closure_fixture(tmp_path)
    original = json.loads(json.dumps(current))
    result = run.planned_scope_close(plan, scope, current)
    assert result['status'] == 'PLANNED_ADMISSION_CLOSED'
    assert result['new_data_accepted'] is False
    assert current == original and current['collector_status'] == 'STALE'


@pytest.mark.parametrize('change', ['early', 'expired', 'future_marker', 'old_marker', 'wrong_contract',
    'wrong_invocation', 'wrong_child', 'child_running', 'interrupted', 'wrong_exit', 'wrapper_mismatch',
    'lock', 'stale_index', 'authority', 'full_failed', 'odds_failed', 'report_mutation', 'scope_stop', 'source_stop'])
def test_unproven_or_unhealthy_close_rejects(tmp_path, change):
    plan, scope, current, marker, lifecycle, report = closure_fixture(tmp_path)
    m=json.loads(marker.read_bytes()); l=json.loads(lifecycle.read_bytes())
    if change=='early': current['observed_at']=(scope.end-timedelta(seconds=91)).isoformat()
    elif change=='expired': current['observed_at']=(scope.end+timedelta(seconds=1)).isoformat()
    elif change=='future_marker':m['observed_at']=(scope.end-timedelta(seconds=1)).isoformat()
    elif change=='old_marker':m['observed_at']=(scope.end-timedelta(seconds=91)).isoformat()
    elif change=='wrong_contract':m['contract_sha256']='b'*64
    elif change=='wrong_invocation':m['service_invocation_id']='b'*32
    elif change=='wrong_child':l['child_pid']=999
    elif change=='child_running':l['children_reaped']=False
    elif change=='interrupted':l['interrupted']=True
    elif change=='wrong_exit':l['returncode']=1
    elif change=='wrapper_mismatch':current['unit_status']['odds']['ExecMainPID']='999'
    elif change=='lock':run.atomic_json(Path(plan['lock_path']), {'owner':'other'})
    elif change=='stale_index':current['source_age_seconds']=270
    elif change=='authority':current['authority_status']='STALE'
    elif change=='full_failed':current['lanes'][0]['status']='CAPTURE_FAILED'
    elif change=='odds_failed':current['lanes'][1]['phase']='ODDS_CAPTURE_ONLY_FAILED'
    elif change=='report_mutation':report.write_text('{}')
    elif change=='scope_stop':run.atomic_json(scope.session/'STOP.json', {'reason':'denial'})
    elif change=='source_stop':
        gate=SportsbetAccess(plan['sportsbet_access_state']);value=gate.read();value['phase']='STOP';run.atomic_json(gate.path,value)
    run.atomic_json(marker,m);run.atomic_json(lifecycle,l)
    assert run.planned_scope_close(plan, scope, current) is None


def test_actual_exported_wrapper_retains_closed_admission_without_traffic(tmp_path, monkeypatch):
    import os
    import sqlite3
    import subprocess
    import sys
    from zoneinfo import ZoneInfo
    from scripts import prepare_freshness_rehearsal as packaging
    from scripts.check_freshness_service import service_command
    from tests.test_freshness_campaign import make_campaign
    from tests.test_refresh_shared_sportsbet_snapshot import access
    from race_collection.live_freshness_contract import AttemptAllowance, FreshnessContract

    gate = access(tmp_path)
    monkeypatch.setenv('GREYHOUND_SPORTSBET_ACCESS_STATE', str(gate))
    installed=tmp_path/'installed';installed.mkdir()
    for name in (*packaging.UNITS,'greyhound-operator-ui-r3.service'):
        (installed/name).write_text('synthetic original '+name)
    db=tmp_path/'empty-history.sqlite';sqlite3.connect(db).close()
    campaign=make_campaign(tmp_path/'campaign')
    stamp=datetime.now(ZoneInfo('Australia/Melbourne'))
    package=tmp_path/'package'
    packaging.prepare(output=package,start=stamp-timedelta(minutes=59),python=Path(sys.executable),
        db=db,lock=tmp_path/'collector.lock',reconciliation_roots={},installed_dir=installed,
        campaign_root=campaign.root,operational_predictions=True,observation_minutes=60)
    plan=json.loads((package/'plan.json').read_bytes())
    accounting=dict(schema_version='freshness_attempt_reconciliation_v1',complete=True,
                    consumed=[],sources=[{'sha256':'a'*64}])
    keys=('profile','rehearsal_id','starts_at','ends_at','lock_path','evidence_root','db_path',
          'cleanup_seconds','max_capture_attempts','max_logical_requests','source_identity_sha256',
          'runtime_sha256','campaign_root','campaign_authorization_sha256','operational_predictions')
    contract={k:plan[k] for k in keys}
    contract.update(schema_version='freshness_rehearsal_contract_v1',source_date=stamp.date().isoformat(),
                    reconciliation_sha256=digest(accounting))
    (package/'contract.json').write_text(json.dumps(contract))
    scope=FreshnessContract(contract);AttemptAllowance(scope).initialize(accounting)
    campaign.begin(plan['rehearsal_id'],now=stamp,deadline=stamp+timedelta(minutes=20))
    command,cwd,env=service_command(package/'units/shadow-autopilot-odds-capture.service')
    invocation='a'*32;env['INVOCATION_ID']=invocation
    launcher='from scripts.check_freshness_service import deny_network; import os,sys; deny_network(); os.execv(sys.argv[1],sys.argv[1:])'
    result=subprocess.run([sys.executable,'-c',launcher,*command],cwd=cwd,env=env,
                          capture_output=True,text=True,timeout=30)
    (tmp_path/'wrapper.log').write_text(result.stdout+result.stderr)
    assert result.returncode==2,(result.stdout+result.stderr)[-1000:]
    marker=scope.session/'admission-closures'/(invocation+'.json')
    assert marker.exists(),'actual service did not retain its planned admission closure'
    value=json.loads(marker.read_bytes())
    lifecycle=json.loads((Path(plan['evidence_root'])/'shadow_autopilot_daemon_runtime/service-lifecycles'/(invocation+'.json')).read_bytes())
    assert value['contract_sha256']==digest(contract)
    assert value['process_pid']==lifecycle['child_pid']!=lifecycle['wrapper_pid']
    assert lifecycle['children_reaped'] is True and lifecycle['returncode']==2
    assert value['runtime_action']=='OPERATING_SCOPE_CLOSED'
    assert not Path(plan['lock_path']).exists()
    assert not AttemptAllowance(scope).claims()
    assert not (scope.session/'request-count.json').exists()
    assert not (scope.session/'network-count.json').exists()
    assert not (scope.session/'STOP.json').exists()
    assert json.loads(gate.read_bytes())['operations']==[]
    # Feed the actual wrapper/child evidence through the supervisor guard.
    # Native state is fabricated here; its validation has separate fixtures.
    previous=Path(plan['evidence_root'])/'shadow_autopilot_daemonization_v1_previous_odds_capture/odds_capture_only_daemon_report.json'
    run.atomic_json(previous,{'run_id':'previous_odds_capture','status':'READY',
        'final_status':'ODDS_CAPTURE_ONLY_READY','runtime_action':'LIVE_COLLECTION_COMPLETE',
        'generated_at':(stamp-timedelta(seconds=120)).isoformat()})
    import hashlib
    current={'observed_at':datetime.now(ZoneInfo('Australia/Melbourne')).isoformat(),
        'collector_status':'STALE','index_status':'AVAILABLE/FRESH','authority_status':'AVAILABLE/FRESH',
        'source_age_seconds':114,'lock':None,'unit_status':{'odds':{'ActiveState':'failed',
        'SubState':'failed','MainPID':'0','ExecMainPID':str(lifecycle['wrapper_pid']),'InvocationID':invocation}},
        'lanes':[{'lane':'FULL_DAEMON','status':'RECEIPT_READY'}, {'lane':'ODDS_ONLY','status':'STALE',
        'phase':'ODDS_CAPTURE_ONLY_READY','run_id':'previous_odds_capture',
        'reference_hashes':{'report':hashlib.sha256(previous.read_bytes()).hexdigest()}}]}
    assert run.planned_scope_close(plan,scope,current)['status']=='PLANNED_ADMISSION_CLOSED'
    assert current['collector_status']=='STALE'


@pytest.mark.parametrize('duration,operational,minimum,accepted', [
    (300,True,1,True),(3540,True,2,True),(3600,True,1,False),
    (600,False,1,False),(599,True,1,False),(600,True,0,False),
    (600,True,True,False),(600,True,4,False)])
def test_short_observation_minimums_are_explicit_and_bounded(duration,operational,minimum,accepted):
    start=datetime(2099,1,1,tzinfo=timezone.utc)
    plan={'starts_at':start.isoformat(),'ends_at':(start+timedelta(seconds=duration)).isoformat(),
          'operational_predictions':operational,'minimum_completed_full_cycles':minimum,
          'minimum_distinct_captures':minimum}
    if accepted:assert run.observation_minimums(plan)==(minimum,minimum,6)
    else:
        with pytest.raises(ValueError,match='invalid_observation_minimums'):run.observation_minimums(plan)


def test_actual_observer_retains_stale_sample_and_explicit_shutdown_classification(tmp_path,monkeypatch):
    plan,scope,current,*_=closure_fixture(tmp_path)
    observed=datetime.fromisoformat(current['observed_at'])
    plan.update(starts_at=(scope.end-timedelta(hours=1)).isoformat(),
        first_index_deadline_seconds=180,readiness_warmup_seconds=1200,sample_period_seconds=0)
    scope.campaign=None
    current.update(packet_sha256='packet',source_at=(observed-timedelta(seconds=114)).isoformat(),
                   external_service_overhead_seconds={})
    output=tmp_path/'output'
    run.atomic_json(Path(plan['evidence_root'])/'shadow_autopilot_daemon_runtime/live-publication-events/000000.json',
                    {'previous_event_sha256':None,'packet_sha256':'packet'})
    monkeypatch.setattr(run,'now',lambda:observed)
    monkeypatch.setattr(run,'sample',lambda *args:current.copy())
    monkeypatch.setattr(run,'AttemptAllowance',lambda scope:SimpleNamespace(claims=lambda:[]))
    monkeypatch.setattr(run,'window_accounting',lambda *args:{})
    import race_collection.freshness_rehearsal as policy
    monkeypatch.setattr(policy,'TimerAccounting',lambda start:SimpleNamespace(observe=lambda current:None,summary=lambda now:{}))
    class SampleComplete(Exception):pass
    ticks=[]
    def tick():
        ticks.append(1)
        if len(ticks)==2:raise SampleComplete
    with pytest.raises(SampleComplete):
        run.observe(output,plan,None,scope,predictions=SimpleNamespace(tick=tick))
    sample=json.loads((output/'samples/000000.json').read_bytes())
    assert sample['collector_status']=='STALE'
    assert sample['lanes'][1]['status']=='STALE'
    assert sample['planned_shutdown']['status']=='PLANNED_ADMISSION_CLOSED'
    assert sample['planned_shutdown']['new_data_accepted'] is False
    progress=json.loads((output/'progress.json').read_bytes())
    assert progress['completed_cycles']=={'full':0,'odds':0}


@pytest.mark.parametrize("duration,odds,accepted", [(300,3,True),(540,3,True),(600,3,False),(300,2,False),(300,True,False),(300,6,True)])
def test_shutdown_followup_odds_minimum_is_finite(duration,odds,accepted):
    start=datetime(2099,1,1,tzinfo=timezone.utc)
    plan={"starts_at":start.isoformat(),"ends_at":(start+timedelta(seconds=duration)).isoformat(),
          "operational_predictions":True,"minimum_completed_full_cycles":1,
          "minimum_distinct_captures":1,"minimum_completed_odds_cycles":odds}
    if accepted:assert run.observation_minimums(plan)==(1,1,odds)
    else:
        with pytest.raises(ValueError,match="invalid_observation_minimums"):run.observation_minimums(plan)
