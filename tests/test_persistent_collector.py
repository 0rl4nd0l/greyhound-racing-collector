"""Real owner lifecycle and accounting with native child execution replaced."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from race_collection import persistent_collector as collector
from race_collection import persistent_native as native
from race_collection.daily_race_inventory import write_daily_inventory
from race_collection.live_freshness_contract import create_once
from tests.test_persistent_native import backend
from utils.sportsbet_access import SportsbetAccess as RealSportsbetAccess


@pytest.fixture
def owner_case(backend,monkeypatch):
    cfg,standing,standing_ref,_=backend
    clock=[datetime.fromisoformat('2026-10-03T12:03:00+10:00')]
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):return clock[0].astimezone(tz or timezone.utc)
    monkeypatch.setattr('race_collection.freshness_campaign.datetime',Clock)
    monkeypatch.setattr(collector,'now',lambda:clock[0])
    monkeypatch.setattr('race_collection.operational_prediction.now',lambda:clock[0])
    monkeypatch.setattr(collector.time,'monotonic',lambda:clock[0].timestamp())
    monkeypatch.setattr(collector.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+timedelta(seconds=seconds)))
    def source(path=None,**kwargs):
        return RealSportsbetAccess(path or cfg['source_state'],clock=lambda:clock[0].timestamp())
    monkeypatch.setattr('utils.sportsbet_access.SportsbetAccess',source)
    access=source()
    access.initialize(access_basis={'status':'permitted','reference':'SYNTHETIC_NO_PROVIDER'})
    prepared=native.prepare_day(cfg,standing_ref,'2026-10-03',clock[0]-timedelta(minutes=3))
    output=Path(prepared['output'])
    for lane,name in [('full','shadow-autopilot.service'),('odds','shadow-autopilot-odds-capture.service')]:
        (output/'units'/name).write_text('[Service]\nWorkingDirectory='+prepared['plan']['source_root']+'\nExecStart=/synthetic/native-child '+lane+'\n')
    owner=collector.DailyOwner(cfg,prepared)
    children=[]
    class Child:
        def __init__(self,command,kwargs):
            self.command=command;self.kwargs=kwargs;self.returncode=None;self.pid=123456
            state=json.loads(owner.state_path.read_bytes())
            assert state['dispatches'][-1]['returncode'] is None
            assert state['dispatches'][-1]['invocation_id']==kwargs['env']['INVOCATION_ID']
            if state['dispatches'][-1]['lane'] in ('full','odds'):
                assert state['next_due_at'][state['dispatches'][-1]['lane']]
        def poll(self):return self.returncode
        def wait(self,timeout=None):
            if self.returncode is None:self.returncode=0
            return self.returncode
    def spawn(command,**kwargs):
        child=Child(command,kwargs);children.append(child);return child
    monkeypatch.setattr(collector.subprocess,'Popen',spawn)
    cfg['standing_authority']=standing_ref
    return SimpleNamespace(owner=owner,cfg=cfg,standing=standing,prepared=prepared,
        clock=clock,children=children,source=source,output=output)


def race(case,minutes=30,number=1):
    jump=case.clock[0]+timedelta(minutes=minutes)
    return {'url':f'https://www.thedogs.com.au/racing/test/2026-10-03/{number}/example',
        'date':'2026-10-03','venue':'TEST','race_number':number,
        'race_time':jump.strftime('%I:%M %p'),'scheduled_jump_datetime':jump.isoformat()}


def retain_inventory(case,minutes=30,age=0):
    ref=write_daily_inventory(case.output/'inventories'/f'inventory-{len(list(case.output.glob("inventories/*.json")))}.json',
        races=[race(case,minutes)],source_date='2026-10-03',
        observed_at=case.clock[0]-timedelta(seconds=age))
    case.owner.state['inventory']=ref;case.owner.save();return ref


def finish_inventory(case,child,minutes=120):
    offset=child.command.index('--discover')
    _,target,receipt=child.command[offset+1:offset+4]
    ref=write_daily_inventory(target,races=[race(case,minutes)],source_date='2026-10-03',observed_at=case.clock[0])
    create_once(receipt,{'inventory':ref,'completed_at':case.clock[0].isoformat()})
    child.returncode=0
    return ref


def finish_native(case,child,*,action='LIVE_COLLECTION_COMPLETE',status='READY',code=0):
    identity=child.kwargs['env']['INVOCATION_ID']
    runtime=Path(case.owner.plan['evidence_root'])/'shadow_autopilot_daemon_runtime'
    create_once(runtime/'service-lifecycles'/f'{identity}.json',{
        'invocation_id':identity,'children_reaped':True,'status':'COMPLETE','returncode':code})
    create_once(runtime/'service-terminals'/f'{identity}.json',{
        'invocation_id':identity,'allocation_sha256':case.prepared['allocation_ref']['sha256'],
        'status':status,'runtime_action':action,'final_verdict':'DAEMON_READY'})
    child.returncode=code


def test_idle_start_acquires_first_inventory_then_waits_without_native_capture(owner_case):
    c=owner_case;c.owner.activate()
    assert c.owner.tick()=='RUNNING'
    assert len(c.children)==1 and '--discover' in c.children[0].command
    assert c.owner.state['dispatches'][0]['returncode'] is None
    ref=finish_inventory(c,c.children[0])
    assert c.owner.tick()=='RUNNING'
    assert len(c.children)==1
    assert c.owner.state['inventory']==ref
    assert json.loads((c.output/'persistent-health.json').read_bytes())['status']=='WAITING_FOR_RACE'


def test_active_inventory_reuses_pin_and_launches_only_native_selected_lanes(owner_case):
    c=owner_case;c.owner.activate();ref=retain_inventory(c)
    assert c.owner.tick()=='RUNNING'
    assert len(c.children)==2
    for child in c.children:
        assert '--discover' not in child.command
        assert child.command[child.command.index('--discovery-inventory')+1]==ref['path']
        assert child.command[child.command.index('--discovery-inventory-sha256')+1]==ref['sha256']
        assert child.kwargs['env']['GREYHOUND_SPORTSBET_ACCESS_STATE']==c.cfg['source_state']
    c.owner.tick()
    assert len(c.children)==2  # Running lanes cannot acquire a duplicate owner.


@pytest.mark.parametrize('age',[841,901,1801])
def test_stale_inventory_prevents_new_native_capture_until_replacement(owner_case,age):
    c=owner_case;c.owner.activate();retain_inventory(c,age=age)
    assert c.owner.tick()=='RUNNING'
    assert len(c.children)==1 and '--discover' in c.children[0].command
    assert all(d['lane']=='inventory' for d in c.owner.state['dispatches'])


def test_source_stop_cannot_be_renewed_by_owner_activation(owner_case):
    c=owner_case;source=c.source();state=source.read();state['phase']='STOP'
    state['denials']=[{'status':429,'reason':'SYNTHETIC_PRESERVED_DENIAL'}];source.write(state)
    before=Path(c.cfg['source_state']).read_bytes()
    with pytest.raises(Exception,match='requires_open|operating_policy|source'):
        c.owner.activate()
    assert Path(c.cfg['source_state']).read_bytes()==before
    assert not c.children


def test_restart_preserves_unknown_dispatch_and_never_replays_it(owner_case):
    c=owner_case;c.owner.activate();c.owner.tick()
    before=c.owner.state_path.read_bytes()
    with pytest.raises(ValueError,match='interrupted_dispatch_requires_reconciliation'):
        collector.DailyOwner(c.cfg,c.prepared)
    assert c.owner.state_path.read_bytes()==before and len(c.children)==1


def test_graceful_drain_reaps_child_and_retains_same_open_lease(owner_case):
    c=owner_case;c.owner.activate();c.owner.tick();finish_inventory(c,c.children[0])
    ledger_path=Path(c.cfg['campaign_root'])/'ledger.json'
    prior=json.loads(ledger_path.read_bytes())
    c.owner.drain(close=False)
    assert not c.owner.children
    assert json.loads(ledger_path.read_bytes())==prior
    assert 'closed_at' not in prior['launches'][c.owner.plan['rehearsal_id']]
    assert not (c.output/'day-closed.json').exists()
    assert c.owner.state['restartable_pause_at']
    resumed=collector.DailyOwner(c.cfg,c.prepared);resumed.activate()
    assert json.loads(ledger_path.read_bytes())==prior
    assert len(c.source().read()['diagnostic_authorizations'])==1


def test_after_midnight_terminal_inventory_closes_prior_racing_day(owner_case):
    c=owner_case;c.owner.activate()
    c.clock[0]=datetime.fromisoformat('2026-10-04T00:30:00+10:00')
    retain_inventory(c,minutes=-20)
    assert c.owner.tick()=='DAY_ENDED'
    c.owner.drain(close=True)
    closure=json.loads((c.output/'day-closed.json').read_bytes())
    assert closure['source_date']=='2026-10-03'
    ledger=json.loads((Path(c.cfg['campaign_root'])/'ledger.json').read_bytes())
    assert ledger['launches'][c.owner.plan['rehearsal_id']]['closed_at']
    assert not c.children


def test_restart_keeps_consumed_lane_cadence_and_counts_only_verified_completion(owner_case):
    c=owner_case;c.owner.activate();retain_inventory(c);c.owner.tick()
    finish_native(c,c.children[0])
    finish_native(c,c.children[1],action='DEFERRED_FULL_LOCK_HANDOFF',
        status='SKIPPED_FULL_DAEMON_LOCK_HANDOFF',code=2)
    c.owner.drain(close=False)
    assert c.owner.state['completed_lanes']=={'full':1,'odds':0}
    assert c.owner.state['deferred_lanes']=={'full':0,'odds':1}
    due=dict(c.owner.state['next_due_at'])
    resumed=collector.DailyOwner(c.cfg,c.prepared);resumed.activate()
    assert resumed.state['next_due_at']==due
    resumed.tick()
    assert len(c.children)==2
    c.clock[0]=datetime.fromisoformat(due['odds'])
    resumed.tick()
    assert len(c.children)==3
    assert c.children[-1].command[1]=='odds'
    assert resumed.state['completed_lanes']=={'full':1,'odds':0}


def test_zero_exit_without_native_terminal_never_counts_success(owner_case):
    c=owner_case;c.owner.activate();retain_inventory(c);c.owner.tick()
    c.children[0].returncode=0
    finish_native(c,c.children[1])
    with pytest.raises(ValueError,match='terminal_missing'):
        c.owner.poll()
    assert c.owner.state['completed_lanes']=={'full':0,'odds':1}
    assert c.owner.state['dispatches'][0]['native_disposition']=='FAILED_OR_UNVERIFIED'
    assert not c.owner.children


def test_expired_unstarted_day_closes_without_granting_or_consuming_lease(owner_case):
    c=owner_case
    c.clock[0]=c.owner.scope.end+timedelta(minutes=1)
    before=Path(c.cfg['source_state']).read_bytes()
    c.owner.drain(close=True)
    closure=json.loads((c.output/'day-closed.json').read_bytes())
    assert closure['status']=='EXPIRED_UNSTARTED'
    ledger=json.loads((Path(c.cfg['campaign_root'])/'ledger.json').read_bytes())
    assert ledger['launches']=={}
    assert Path(c.cfg['source_state']).read_bytes()==before
    assert not c.children
