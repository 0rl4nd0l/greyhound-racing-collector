"""Exact generated wrapper/daemon processes, real ownership gates, denied network."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys
import time

import pytest

from tests.fixtures.incident_engineering_case import put
from tests.test_incident_comparison import case
from race_collection.live_freshness_contract import AttemptAllowance, FreshnessContract, digest

ROOT = Path(__file__).resolve().parents[1]
STAMP = datetime.fromisoformat('2026-10-01T16:11:00+10:00')


@pytest.fixture
def incident_service(tmp_path, monkeypatch):
    from scripts import prepare_freshness_rehearsal as packaging
    from race_collection import freshness_campaign as campaigns
    from utils.sportsbet_access import SportsbetAccess
    from tests import test_refresh_shared_sportsbet_snapshot as transport
    authority, reference, comparison, comparison_ref = case(tmp_path)
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None): return STAMP.astimezone(tz or timezone.utc)
    monkeypatch.setattr(campaigns, 'datetime', Clock)
    monkeypatch.setattr(transport, 'datetime', Clock)
    http = transport.fixture(tmp_path, 1)
    gate = SportsbetAccess(tmp_path/'access.json', clock=lambda:STAMP.timestamp())
    gate.initialize(access_basis={'status':'permitted','reference':'invented incident startup'})
    gate.authorize_diagnostic(reference=authority['authority_reference']+':slot:001',
        expected_sha256=hashlib.sha256(gate.path.read_bytes()).hexdigest(),
        expires_at=datetime.fromisoformat(authority['slots'][0]['ends_at']).timestamp(),
        max_operations=192,rationale='invented finite startup',incident_authority=reference,incident_slot='001')
    monkeypatch.setenv('GREYHOUND_SPORTSBET_ACCESS_STATE',str(gate.path))
    campaign_root=tmp_path/'campaign'
    put(campaign_root/'authorization.json',{'schema_version':'collector_engineering_campaign_v1',
        'campaign_id':'SYNTHETIC','max_capture_attempts':12,'max_logical_requests':48000,'max_live_seconds':10800})
    put(campaign_root/'ledger.json',{'campaign_id':'SYNTHETIC','attempts':[],
        'logical_requests':0,'launches':{},'source_holds':[]})
    campaign=campaigns.Campaign(campaign_root,incident_authority=reference,incident_slot='001')
    installed=tmp_path/'installed';installed.mkdir()
    for name in (*packaging.UNITS,'greyhound-operator-ui-r3.service'):
        (installed/name).write_text('invented original '+name)
    history=tmp_path/'empty-history.sqlite';sqlite3.connect(history).close()
    package=tmp_path/'package'
    packaging.prepare(output=package,start=datetime.fromisoformat(authority['slots'][0]['starts_at']),
        python=Path(sys.executable),db=history,lock=tmp_path/'collector.lock',reconciliation_roots={},
        installed_dir=installed,campaign_root=campaign_root,operational_predictions=True,
        comparison_plan=Path(comparison_ref['path']),prediction_root=Path(authority['prediction_root']),
        incident_authority=reference,incident_slot='001')
    plan=json.loads((package/'plan.json').read_bytes())
    accounting={'schema_version':'freshness_attempt_reconciliation_v1','complete':True,
        'consumed':[],'sources':[{'sha256':'a'*64}]}
    keys=('profile','rehearsal_id','starts_at','ends_at','lock_path','evidence_root','db_path',
        'cleanup_seconds','max_capture_attempts','max_logical_requests','source_identity_sha256',
        'runtime_sha256','campaign_root','campaign_authorization_sha256','operational_predictions',
        'incident_authority','incident_slot','frozen_comparison','prediction_root')
    contract={key:plan[key] for key in keys}
    contract.update(schema_version='freshness_rehearsal_contract_v1',source_date='2026-10-01',
        reconciliation_sha256=digest(accounting))
    put(package/'contract.json',contract)
    scope=FreshnessContract(contract)
    AttemptAllowance(scope).initialize(accounting)
    campaign.begin(plan['rehearsal_id'],now=STAMP,
        deadline=datetime.fromisoformat(authority['slots'][0]['cleanup_by']))
    return package,plan,contract,reference,gate.path,http


def launch(fixture, tmp_path, lane, *, mode='valid'):
    from scripts.check_freshness_service import service_command
    package,plan,contract,reference,gate,http=fixture
    name='shadow-autopilot-odds-capture.service' if lane=='odds' else 'shadow-autopilot.service'
    unit=package/'units'/name
    command,cwd,env=service_command(unit)
    assert not any(key.startswith('GREYHOUND_INCIDENT_') for key in env)
    env.update(PYTHONPATH=os.pathsep.join((str(ROOT/'tests/fixtures/incident_service_transport'),
        str(ROOT/'tests/fixtures/shared_snapshot_transport'),str(package/'source'))),
        GREYHOUND_SHARED_SNAPSHOT_FIXTURE=str(http),LIVE_ENHANCE_LIMIT='0',
        GREYHOUND_INCIDENT_SERVICE_FIXTURE=str(tmp_path/'startup-witness.jsonl'),
        GREYHOUND_FIXTURE_EPOCH=str(STAMP.timestamp()),GREYHOUND_FIXTURE_MONOTONIC=str(time.monotonic()),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    if mode=='unbound_env':
        env.update(GREYHOUND_INCIDENT_AUTHORITY_SHA256=reference['sha256'],GREYHOUND_INCIDENT_SLOT='001')
        command[command.index('--db')+1]=json.loads((package/'contract.json').read_bytes())['db_path']
    launcher='from scripts.check_freshness_service import deny_network; import os,sys; deny_network(); os.execv(sys.argv[1],sys.argv[1:])'
    def run(cmd):
        return subprocess.run([sys.executable,'-c',launcher,*cmd],cwd=cwd,env=env,
            capture_output=True,text=True,timeout=100)
    if mode=='valid':
        condition=shlex.split(next(line.split('=',1)[1] for line in unit.read_text().splitlines()
                                   if line.startswith('ExecCondition=')))
        checked=run(condition)
        assert checked.returncode==0,checked.stdout+checked.stderr
        assert not any(key.startswith('GREYHOUND_INCIDENT_AUTHORITY') for key in env)
    if mode=='missing': (package/'contract.json').unlink()
    elif mode=='mismatch':
        contract['incident_slot']='002';put(package/'contract.json',contract)
    elif mode=='changed_authority':
        contract['incident_authority']={**reference,'sha256':'f'*64};put(package/'contract.json',contract)
    if mode=='preflight':
        (package/'contract.json').unlink()
        command=[*command,'--verify-live-runtime']
    run_id='incident_boot_odds_capture' if lane=='odds' else 'incident_boot_full'
    result=run([*command,'--run-id',run_id])
    (tmp_path/'wrapper.log').write_text(result.stdout+result.stderr)
    return result,Path(plan['evidence_root']),run_id


@pytest.mark.parametrize('lane',['full','odds'])
def test_exact_generated_wrapper_propagates_validated_incident_owner(incident_service,tmp_path,lane):
    result,evidence,run_id=launch(incident_service,tmp_path,lane)
    assert result.returncode==0,result.stdout+result.stderr
    terminal=json.loads((evidence/f'shadow_autopilot_daemonization_v1_{run_id}/terminal-timing.json').read_bytes())
    assert terminal['runtime_action']=='LIVE_COLLECTION_COMPLETE'
    assert terminal['status']==('READY' if lane=='odds' else 'DAEMON_READY')
    witnesses=[json.loads(line) for line in (tmp_path/'startup-witness.jsonl').read_text().splitlines()]
    wrapper=next(row for row in witnesses if row['script']=='run_freshness_service.py')
    daemon=next(row for row in witnesses if row['script']=='shadow_autopilot_daemon.py')
    assert wrapper['authority'] is None and wrapper['slot'] is None
    assert daemon['authority']==incident_service[3]['sha256'] and daemon['slot']=='001'
    state=json.loads(incident_service[4].read_bytes())
    assert state['phase']=='OPEN' and state['active'] is None and state['operations']
    assert all(row['incident_authority_sha256']==incident_service[3]['sha256']
               and row['incident_slot']=='001' for row in state['operations'])
    lifecycle=json.loads(next((evidence/'shadow_autopilot_daemon_runtime/service-lifecycles').glob('*.json')).read_bytes())
    assert lifecycle['children_reaped'] and lifecycle['returncode']==0
    assert not Path(incident_service[2]['lock_path']).exists()


@pytest.mark.parametrize('mode',['missing','mismatch','changed_authority'])
def test_wrapper_rejects_absent_or_mismatched_incident_authority(incident_service,tmp_path,mode):
    before=incident_service[4].read_bytes()
    result,evidence,_=launch(incident_service,tmp_path,'odds',mode=mode)
    assert result.returncode!=0
    assert incident_service[4].read_bytes()==before
    assert not (tmp_path/'transport.jsonl').exists()
    assert not (evidence/'shadow_autopilot_daemon_runtime/service-lifecycles').exists()


def test_wrapper_cannot_use_another_valid_incident_source_grant(incident_service,tmp_path):
    from utils.sportsbet_access import SportsbetAccess
    authority=json.loads(Path(incident_service[3]['path']).read_bytes())
    authority['incident_id']='another-invented-incident'
    reference=put(tmp_path/'another-authority.json',authority)
    gate=SportsbetAccess(incident_service[4],clock=lambda:STAMP.timestamp())
    gate.authorize_diagnostic(reference=authority['authority_reference']+':slot:001',
        expected_sha256=hashlib.sha256(gate.path.read_bytes()).hexdigest(),
        expires_at=datetime.fromisoformat(authority['slots'][0]['ends_at']).timestamp(),
        max_operations=192,rationale='different invented owner',incident_authority=reference,incident_slot='001')
    before=gate.path.read_bytes()
    result,evidence,_=launch(incident_service,tmp_path,'odds',mode='foreign_lease')
    assert result.returncode!=0 and 'sportsbet_incident_owner_required' in result.stderr
    assert gate.path.read_bytes()==before
    assert not (tmp_path/'transport.jsonl').exists()
    assert not (evidence/'shadow_autopilot_daemon_runtime/service-lifecycles').exists()


def test_nonincident_contract_cannot_borrow_inherited_incident_owner(incident_service,tmp_path):
    from race_collection.freshness_campaign import Campaign
    package,_,value,_,gate,_=incident_service
    contract=dict(value)
    for key in ('incident_authority','incident_slot','frozen_comparison','prediction_root'):
        contract.pop(key)
    campaign=Campaign(contract['campaign_root'])
    contract['campaign_authorization_sha256']=digest(campaign.value)
    contract['max_capture_attempts']=campaign.value['max_capture_attempts']
    contract['db_path']=str(campaign.root/'operational-predictions/capture.sqlite3')
    contract['operational_predictions']={**contract['operational_predictions'],
        'capture_db_path':contract['db_path']}
    FreshnessContract(contract)  # The negative is ownership, not invalid scope.
    put(package/'contract.json',contract)
    before=gate.read_bytes()
    result,evidence,_=launch(incident_service,tmp_path,'odds',mode='unbound_env')
    assert result.returncode!=0 and 'sportsbet_incident_owner_required' in result.stderr
    assert gate.read_bytes()==before
    assert not (tmp_path/'transport.jsonl').exists()
    assert not (evidence/'shadow_autopilot_daemon_runtime/service-lifecycles').exists()


@pytest.mark.parametrize('lane',['full','odds'])
def test_runtime_preflight_remains_available_before_contract_creation(incident_service,tmp_path,lane):
    result,evidence,_=launch(incident_service,tmp_path,lane,mode='preflight')
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout.splitlines()[-1])['dependency_installation'] is False
    assert not (tmp_path/'transport.jsonl').exists()
    assert not (evidence/'shadow_autopilot_daemon_runtime/service-lifecycles').exists()
