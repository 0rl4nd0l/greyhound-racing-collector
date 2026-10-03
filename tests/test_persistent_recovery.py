"""A reviewed correction reuses one grant and preserves failed membership."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from race_collection import persistent_native as native
from race_collection.freshness_campaign import Campaign
from race_collection.live_freshness_contract import FreshnessContract, AttemptAllowance, create_once, digest
from tests.test_persistent_native import backend
from tests.fixtures.persistent_operation_case import put
from utils.sportsbet_access import SportsbetAccess


@pytest.fixture
def recovery(backend, monkeypatch):
    cfg,standing,standing_ref,calls=backend
    cfg['source_commit']='f'*40
    first=native.prepare_day(cfg,standing_ref,'2026-10-03',native.stamp('2026-10-03T12:00:00+10:00'))
    oldcfg=put(Path(cfg['campaign_root']).parent/'old-config.json',cfg)
    now=native.stamp('2026-10-03T12:05:00+10:00')
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):return now.astimezone(tz or timezone.utc)
    monkeypatch.setattr('race_collection.freshness_campaign.datetime',Clock)
    scope=FreshnessContract(first['contract']);campaign=scope.campaign
    campaign.begin(first['plan']['rehearsal_id'],now=native.stamp('2026-10-03T12:03:00+10:00'),deadline=native.stamp(campaign.persistent['cleanup_by']))
    item={'race_id':'failed-race','race_id_aliases':['failed-race'],'capture_window_minutes':10}
    campaign.consume(Path(cfg['campaign_root'])/'consumed.json',item)
    for _ in range(5):campaign.request()
    campaign.request(kind='browser')
    campaign.close(first['plan']['rehearsal_id'],now=now)
    source=SportsbetAccess(cfg['source_state'],clock=lambda:now.timestamp())
    source.initialize(access_basis={'status':'permitted','reference':'SYNTHETIC_NO_PROVIDER'})
    source.authorize_diagnostic(reference=campaign.persistent['authority_reference']+':day:'+campaign.persistent['allocation_id'],expected_sha256=hashlib.sha256(Path(cfg['source_state']).read_bytes()).hexdigest(),expires_at=native.stamp(campaign.persistent['ends_at']).timestamp(),max_operations=campaign.persistent['max_source_operations'],rationale='SYNTHETIC',persistent_allocation=first['allocation_ref'])
    output=Path(first['output'])
    create_once(output/'persistent-owner-state.json',{'dispatches':[],'completed_lanes':{'full':3,'odds':7},'deferred_lanes':{'full':0,'odds':1},'next_due_at':{'full':'2026-10-03T12:15:00+10:00'}})
    halt=put(output/'HALT.json',{'reason':'operational_prediction_failed_preserved_consumption'})
    stop=put(scope.session/'STOP.json',{'reason':'PERSISTENT_OWNER_FAILURE'})
    put(Path(standing['state_root'])/'health.json',{'status':'HOLD','reason':'preserved-failure'})
    failed=Path(first['plan']['prediction_root'])/'races'/hashlib.sha256(b'failed-race').hexdigest()
    terminal=put(failed/'terminal.json',{'status':'REJECTED','reason':'verification_rejected'})
    baseline=native.recovery_baseline(first,cfg)
    cleanup=put(output/'cleanup.json',{'status':'FAILED_DRAINED_LEASE_CLOSED','allocation':first['allocation_ref'],
        'package':first['output'],'plan':native._ref(Path(first['plan_path'])),'halt':halt,
        'closed_launch':baseline['closed_launch'],'source_active':False,
        **{k:True for k in ('all_dispatches_reaped','prediction_lifetimes_complete','collector_lock_absent','consumption_preserved','failed_attempt_preserved')}})
    selection=put(output/'recovery-selection.json',{'schema_version':'persistent_recovery_selection_v1',
        'status':'AUTHORIZED_SAME_ALLOCATION_RECOVERY','authority_reference':'SYNTHETIC_REVIEWED',
        'racing_date':'2026-10-03','source_commit':'e'*40,'prior_configuration':oldcfg,
        'prior_preparation':first['receipt_ref'],'cleanup':cleanup,'prior_stop':stop,'baseline':baseline})
    newcfg={**cfg,'source_commit':'e'*40,'recovery_selection':selection}
    original_prepare=native.prepare
    def corrected_prepare(**kwargs):
        result=original_prepare(**kwargs)
        plan=json.loads(Path(result['plan']).read_bytes())
        identity_path=Path(plan['source_root'])/'SOURCE_IDENTITY.json'
        identity=json.loads(identity_path.read_bytes());identity['commit']='e'*40;put(identity_path,identity)
        plan['commit']='e'*40;plan['source_identity_sha256']=digest(identity);put(Path(result['plan']),plan)
        return {'plan':result['plan'],'plan_sha256':native._ref(Path(result['plan']))['sha256']}
    monkeypatch.setattr(native,'prepare',corrected_prepare)
    return newcfg,standing_ref,first,now,item,terminal,calls


def test_successor_reuses_exact_grant_caps_roots_and_failed_race(recovery):
    cfg,standing,old,now,item,terminal,calls=recovery
    ledger=Path(cfg['campaign_root'])/'ledger.json';source=Path(cfg['source_state'])
    before=(ledger.read_bytes(),source.read_bytes(),Path(terminal['path']).read_bytes())
    new=native.prepare_day(cfg,standing,'2026-10-03',now)
    assert new['allocation_ref']==old['allocation_ref']
    assert new['plan']['frozen_comparison']==old['plan']['frozen_comparison']
    assert new['plan']['prediction_root']==old['plan']['prediction_root']
    assert new['output']!=old['output'] and new['plan']['commit']=='e'*40
    assert native.prepare_day(cfg,standing,'2026-10-03',now)==new
    assert len(calls)==2
    assert before==(ledger.read_bytes(),source.read_bytes(),Path(terminal['path']).read_bytes())
    scope=FreshnessContract(new['contract'])
    assert AttemptAllowance(scope).consumed(item)
    with scope.campaign.ledger() as value:used=scope.campaign.persistent_usage(value,selected=True)
    assert used['python']==5 and used['browser']==1 and used['capture_attempts']==1
    state=json.loads((Path(new['output'])/'persistent-owner-state.json').read_bytes())
    assert state['next_due_at']['full']=='2026-10-03T12:15:00+10:00'
    assert state['completed_lanes']=={'full':3,'odds':7}
    assert (Path(old['output'])/'HALT.json').exists()
    receipt=json.loads(Path(new['receipt_ref']['path']).read_bytes())
    assert native.checked(receipt['prior_root_health'])['status']=='HOLD'


@pytest.mark.parametrize('change',['halt','stop','cleanup','source_grant','count_regression','failed_record','config','open_lease'])
def test_recovery_rejects_changed_history_or_authority(recovery,change):
    cfg,standing,old,now,item,terminal,calls=recovery
    selection=json.loads(Path(cfg['recovery_selection']['path']).read_bytes())
    if change=='halt':put(Path(old['output'])/'HALT.json',{'reason':'changed'})
    if change=='stop':put(Path(selection['prior_stop']['path']),{'reason':'changed'})
    if change=='cleanup':put(Path(selection['cleanup']['path']),{})
    if change=='failed_record':put(Path(terminal['path']),{'status':'READY'})
    if change=='config':cfg['history_database']+='changed'
    if change=='source_grant':
        p=Path(cfg['source_state']);v=json.loads(p.read_bytes());v['diagnostic_authority']['max_operations']+=1;put(p,v)
    if change in ('count_regression','open_lease'):
        p=Path(cfg['campaign_root'])/'ledger.json';v=json.loads(p.read_bytes())
        if change=='count_regression':next(iter(v['persistent_operation_request_usage'].values()))['counts']['python']=4
        else:v['launches'][old['plan']['rehearsal_id']].pop('closed_at')
        put(p,v)
    with pytest.raises(Exception):native.prepare_day(cfg,standing,'2026-10-03',now)
    assert len(calls)==1


def test_failed_successor_preparation_is_never_automatically_retried(recovery,monkeypatch):
    cfg,standing,old,now,item,terminal,calls=recovery
    def broken(**kwargs):raise RuntimeError('synthetic package interruption')
    monkeypatch.setattr(native,'prepare',broken)
    with pytest.raises(RuntimeError,match='interruption'):native.prepare_day(cfg,standing,'2026-10-03',now)
    with pytest.raises(ValueError,match='partial_or_baseline_changed'):native.prepare_day(cfg,standing,'2026-10-03',now)


def test_successor_activation_and_restart_charge_same_allocation_without_new_source_grant(recovery,monkeypatch):
    from race_collection import persistent_collector as collector
    cfg,standing,old,now,item,terminal,calls=recovery
    new=native.prepare_day(cfg,standing,'2026-10-03',now)
    monkeypatch.setattr(collector,'now',lambda:now)
    source_before=Path(cfg['source_state']).read_bytes()
    owner=collector.DailyOwner(cfg,new);owner.activate()
    owner.campaign.request()
    resumed=native.prepare_day(cfg,standing,'2026-10-03',now)
    assert resumed==new and len(calls)==2
    assert Path(cfg['source_state']).read_bytes()==source_before
    with owner.campaign.ledger() as ledger:
        used=owner.campaign.persistent_usage(ledger,selected=True)
        assert used['python']==6 and used['browser']==1 and used['capture_attempts']==1
        assert len(ledger['launches'])==2
        assert all(r['persistent_allocation']==old['allocation_ref'] for r in ledger['launches'].values())
    assert (Path(old['output'])/'HALT.json').exists()
    assert AttemptAllowance(owner.scope).consumed(item)


@pytest.mark.parametrize('status',[None,'PAUSED','DAY_ENDED'])
def test_owner_health_tracks_successor_package_and_completed_drain(tmp_path,status):
    from race_collection.persistent_collector import publish_owner_health
    output=tmp_path/'successor';output.mkdir()
    put(output/'persistent-health.json',{'status':'ACTIVE_COLLECTION','children':['full']})
    ref={'path':str(output.parent/'native-prepared.json'),'sha256':'a'*64}
    daily=SimpleNamespace(output=output,prepared={'receipt_ref':ref})
    cfg={'source_commit':'b'*40,'recovery_selection':{'path':'/selection','sha256':'c'*64}}
    publish_owner_health(tmp_path,cfg,daily,status)
    value=json.loads((tmp_path/'health.json').read_bytes())
    assert value['status']==(status or 'ACTIVE_COLLECTION')
    assert value['output']==str(output) and value['preparation']==ref
    assert value['recovery_selection']==cfg['recovery_selection']
    assert value['children']==([] if status else ['full'])
