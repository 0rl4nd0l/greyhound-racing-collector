"""Public pilot authority and scheduler behavior with labelled synthetic inputs."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from race_collection import development_pilot as pilot
from race_collection import freshness_campaign as campaigns
from race_collection.development_examples import put, race_key, selected_population as select, SELECTION_POLICY
from race_collection.development_source_authority import CAPS, DATES
from race_collection.freshness_campaign import Campaign
from race_collection.live_freshness_contract import digest, create_once


def setup(tmp_path,monkeypatch):
    from tests.test_preprogramme_operational_campaign import setup_campaign
    campaign_root=tmp_path/'campaign';setup_campaign(campaign_root)
    normal=Campaign(campaign_root);before=digest(normal.value)
    state=tmp_path/'state';state.mkdir(mode=0o700)
    authority={'schema_version':'collector_development_pilot_authority_v1','status':'AUTHORIZED_DEVELOPMENT_PILOT',
        'campaign_id':'synthetic','allocation_id':'development-single-snapshot-20261003-v1',
        'authority_reference':'synthetic:explicit-approved-pilot','allocation_sha256':'1'*64,
        'prior_effective_authorization_sha256':before,'state_root':str(state),
        'prediction_root':str(tmp_path/'predictions'),'dates':DATES,**CAPS,
        'result_closure_at':'2026-10-25T12:00:00+11:00'}
    path=campaign_root/'development-pilot-authority.json';put(path,authority);ref=pilot.reference(path)
    clock=[datetime.fromisoformat('2026-10-03T12:40:00+10:00')]
    class FakeDate(datetime):
        @classmethod
        def now(cls,tz=None):return clock[0].astimezone(tz) if tz else clock[0]
    monkeypatch.setattr(campaigns,'datetime',FakeDate)
    monkeypatch.setattr(pilot,'utc_now',lambda:clock[0])
    current=Campaign(campaign_root,development_authority=ref)
    config={'status':'SYNTHETIC_FIXTURE','state_root':str(state),'campaign_root':str(campaign_root),'allocation':{'path':'/synthetic/allocation','sha256':'1'*64},
            'pilot_campaign_authority':ref,'prediction_root':str(tmp_path/'predictions')}
    schedule=tmp_path/'study-schedule.json';put(schedule,{'slots':['2026-10-01T13:00:00+10:00'],'session_minutes':90})
    config['study_schedule']=pilot.reference(schedule)
    health=tmp_path/'study-health.json';put(health,{'status':'NO_SLOT_DUE','at':clock[0].isoformat()});config['synthetic_study_health']=pilot.reference(health)
    health=tmp_path/'result-health.json';put(health,{'status':'CYCLE_COMPLETE','at':clock[0].isoformat(),'counts':{},'oldest_due':None});config['synthetic_result_health']=pilot.reference(health)
    return current,Campaign(campaign_root),before,config,clock


def population(session,allocation_sha,count=7):
    day=session.name
    local=lambda hhmm:datetime.fromisoformat(day+'T'+hhmm).replace(tzinfo=pilot.ZONE).isoformat()
    rows=[{'race_id':f'Race {i+1} - SYN - {day}','race_key':f'{day}|SYN|{i+1}',
           'jump_at':local(f'13:{10+i*5:02d}:00'),'url':f'https://synthetic.invalid/fabricated/{i+1}',
           'source_native_race_id':str(i+1),'runners':[{'box':1,'display_name':'Synthetic A','identity':'SYNTHETICA','source_native_runner_id':'100'}]}
          for i in range(count)]
    # Canonical identity helper is authoritative, including its actual separators.
    for row in rows:row['race_key']=race_key(row['race_id'])
    intended,selected=select(rows,day)
    value={'schema_version':'development_population_freeze_v1','synthetic':False,'local_date':day,
        'allocation_sha256':allocation_sha,'selection_policy':SELECTION_POLICY,
        'frozen_at':local('12:50:00'),'source_observed_at':local('12:49:00'),
        'observed_races':rows,'intended':intended,'selected_race_ids':selected}
    put(session/'population.json',value)
    put(session/'population.json.completion.json',{'status':'POPULATION_FROZEN',
        'population_sha256':pilot.reference(session/'population.json')['sha256'],'completed_at':local('12:50:01')})
    return rows


def item_for(row):
    from race_collection.synchronous_manual_capture import runner_set_sha256
    return {'race_id':row['race_id'],'race_id_aliases':[row['race_id']],'capture_window_minutes':10,'development_active_runners':row['runners'],
        'race_identity':{'race_id':row['race_id'],'race_url':row['url'],'jump_datetime':row['jump_at'],
            'source_native_race_id':row['source_native_race_id'],'runner_set_sha256':'a'*64}}


def test_shared_ledger_pilot_preserves_study_identity_and_old_consumption(tmp_path,monkeypatch):
    campaign,normal,before,config,clock=setup(tmp_path,monkeypatch)
    with normal.ledger() as ledger:
        ledger['attempts']=[{'race_id':str(i),'aliases':[str(i)],'window':10} for i in range(77)]
        ledger['logical_requests']=87489
        ledger['launches']['previous']={'charged_seconds':35000,'closed_at':'synthetic'}
    original=normal.programme_usage(json.loads((normal.root/'ledger.json').read_bytes()))
    assert digest(Campaign(normal.root).value)==before
    assert campaign.available()
    campaign.begin('pilot',now=clock[0],deadline=clock[0]+timedelta(seconds=7200))
    campaign.request()
    session=Path(config['state_root'])/'sessions'/'2026-10-03'
    rows=population(session,'1'*64)
    clock[0]=clock[0].replace(hour=13)
    for i,row in enumerate(rows[:6]):campaign.consume(session/str(i),item_for(row))
    assert not campaign.available()
    with pytest.raises(ValueError,match='not_frozen_member'):campaign.consume(session/'extra',item_for(rows[6]))
    with pytest.raises(ValueError,match='allowance_consumed'):campaign.consume(session/'retry',item_for(rows[0]))
    campaign.close('pilot',now=clock[0])
    ledger=json.loads((normal.root/'ledger.json').read_bytes())
    assert len(ledger['attempts'])==83 and ledger['logical_requests']==87490
    assert normal.programme_usage(ledger)==original
    assert digest(Campaign(normal.root).value)==before


def test_daily_request_and_separate_result_caps_survive_restart(tmp_path,monkeypatch):
    campaign,_,_,_,clock=setup(tmp_path,monkeypatch)
    with campaign.ledger() as value:
        value['development_request_authority_sha256']=campaign.development_authority['sha256']
        value['development_request_usage']={'2026-10-03':{'prediction':5999,'results':719}}
        value['logical_requests']=100000
    campaign.request()
    with pytest.raises(ValueError,match='request_cap'):Campaign(campaign.root,development_authority=campaign.development_authority).request()
    clock[0]=datetime.fromisoformat('2026-10-05T12:00:00+11:00')
    campaign.request(kind='results')
    with pytest.raises(ValueError,match='request_cap'):campaign.request(kind='results')
    clock[0]=datetime.fromisoformat('2026-10-25T12:00:00+11:00')
    with pytest.raises(ValueError,match='window_closed'):campaign.request(kind='results')


def test_foreign_tags_and_member_identity_are_not_allowance(tmp_path,monkeypatch):
    campaign,normal,_,config,clock=setup(tmp_path,monkeypatch)
    rows=population(Path(config['state_root'])/'sessions'/'2026-10-03','1'*64)
    clock[0]=clock[0].replace(hour=13)
    wrong=item_for(rows[0]);wrong['race_identity']['source_native_race_id']='unrelated'
    with pytest.raises(ValueError,match='not_frozen_member'):campaign.consume('/synthetic/claim',wrong)
    with campaign.ledger() as ledger:
        ledger['attempts'].append({'development_authority_sha256':'f'*64,'development_slot':'2026-10-03'})
    with pytest.raises(ValueError,match='invalid_development_consumption'):
        normal.programme_usage(json.loads((normal.root/'ledger.json').read_bytes()))


class SyntheticCollector:
    """No provider or real labels; state transitions still use immutable runtime outputs."""
    def __init__(self,config,clock,fail=None):self.config=config;self.clock=clock;self.calls=[];self.scope=None;self.children_reaped=True;self.fail=fail
    def prepare(self,session,start):self.calls.append('prepare')
    def refresh(self,session,name):self.calls.append(name)
    def freeze(self,session):
        self.calls.append('freeze');population(session,self.config['allocation']['sha256'])
    def consumed(self,session,race_id):return (session/'attempts'/(hashlib.sha256(race_id.encode()).hexdigest()+'.json')).exists()
    def capture(self,session,race):
        self.calls.append(('capture',race['race_id']))
        put(session/'attempts'/(hashlib.sha256(race['race_id'].encode()).hexdigest()+'.json'),{'synthetic':True})
        if self.fail=='capture':raise ValueError('synthetic_interrupted_capture')
        return 'synthetic-claim'
    def predict(self,session,claim):self.calls.append('predict');return 'synthetic-verification'
    def seal(self,session,race,accounting,verification):
        self.calls.append('seal');put(session/'synthetic-seals'/(hashlib.sha256(race['race_id'].encode()).hexdigest()+'.json'),
            {'synthetic':True,'race_id':race['race_id'],'accounting':accounting,'verification':verification})
    def close(self,session):self.calls.append('close')


def test_synthetic_scheduler_freezes_before_prices_seals_and_never_retries(tmp_path,monkeypatch):
    _,_,_,config,clock=setup(tmp_path,monkeypatch);collector=SyntheticCollector(config,clock)
    def sleep(seconds):clock[0]+=timedelta(seconds=seconds)
    result=pilot.run_session(config,clock=lambda:clock[0],sleep=sleep,collector=collector)
    assert result['status']=='COMPLETE' and result['intended']==7 and result['attempts']==6 and result['verified_forecasts']==6
    assert collector.calls.index('freeze')<next(i for i,x in enumerate(collector.calls) if isinstance(x,tuple))
    before=list(collector.calls)
    assert pilot.run_session(config,clock=lambda:clock[0],sleep=sleep,collector=collector)==result
    assert collector.calls==before
    accounting=json.loads((Path(config['state_root'])/'sessions/2026-10-03/opportunities.final.json').read_bytes())
    assert accounting['opportunities'][-1]['disposition']=='EXCLUDED'


def test_consumed_failure_not_replaced_and_late_preparation_skips(tmp_path,monkeypatch):
    _,_,_,config,clock=setup(tmp_path,monkeypatch);collector=SyntheticCollector(config,clock,fail='capture')
    def sleep(seconds):clock[0]+=timedelta(seconds=seconds)
    result=pilot.run_session(config,clock=lambda:clock[0],sleep=sleep,collector=collector)
    assert result['attempts']==6 and result['verified_forecasts']==0
    assert len([x for x in collector.calls if isinstance(x,tuple)])==6
    clock[0]=datetime.fromisoformat('2026-10-04T12:42:00+11:00')
    before=list(collector.calls)
    assert pilot.run_session(config,clock=lambda:clock[0],collector=collector)['status']=='SKIPPED_LATE_PREPARATION'
    assert collector.calls==before


def test_interrupted_session_and_source_hold_never_restart_provider(tmp_path,monkeypatch):
    campaign,_,_,config,clock=setup(tmp_path,monkeypatch)
    session=Path(config['state_root'])/'sessions/2026-10-03';put(session/'started.json',{'status':'STARTED','synthetic':True})
    collector=SyntheticCollector(config,clock)
    assert pilot.run_session(config,clock=lambda:clock[0],collector=collector)['status']=='INTERRUPTED_REQUIRES_RECONCILIATION'
    assert collector.calls==[]
    campaign.hold_source({'status':403,'synthetic':True})
    with pytest.raises(ValueError,match='source_hold'):campaign.request()


def test_not_due_is_read_only_and_study_schedule_takes_priority(tmp_path,monkeypatch):
    _,_,_,config,clock=setup(tmp_path,monkeypatch)
    clock[0]=datetime.fromisoformat('2026-09-30T13:00:00+10:00')
    assert pilot.run_session(config,clock=lambda:clock[0],collector=SyntheticCollector(config,clock))['status']=='NO_SLOT_DUE'
    assert not (Path(config['state_root'])/'sessions').exists()
    with pytest.raises(ValueError,match='priority'):
        pilot.require_study_priority(config,datetime.fromisoformat('2026-10-01T12:40:00+10:00'),datetime.fromisoformat('2026-10-01T14:40:00+10:00'))


def test_real_authority_rejects_clock_or_adapter_override(tmp_path,monkeypatch):
    _,_,_,config,clock=setup(tmp_path,monkeypatch)
    config['status']='AUTHORIZED'
    with pytest.raises(ValueError,match='synthetic_adapter_forbidden'):
        pilot.run_session(config,clock=lambda:clock[0],collector=SyntheticCollector(config,clock))
    assert not (Path(config['state_root'])/'sessions').exists()


def test_study_due_or_recovery_metadata_blocks_acquisition(tmp_path,monkeypatch):
    _,_,_,config,clock=setup(tmp_path,monkeypatch)
    pilot.require_study_health(config,clock[0])
    for status in ('RESTORATION_HELD','CANARY_NOT_VERIFIED','SESSION_FAILED_RESTORED','ADMISSIONS_PAUSED'):
        p=tmp_path/(status+'.json');put(p,{'status':status,'at':clock[0].isoformat()})
        with pytest.raises(ValueError,match='priority'):
            pilot.require_study_health({**config,'synthetic_study_health':pilot.reference(p)},clock[0])
    p=tmp_path/'due.json';put(p,{'status':'CYCLE_COMPLETE','at':clock[0].isoformat(),'oldest_due':clock[0].isoformat()})
    with pytest.raises(ValueError,match='priority'):
        pilot.require_study_health({**config,'synthetic_result_health':pilot.reference(p)},clock[0])


def test_signal_during_capture_aborts_remaining_selected_races(tmp_path,monkeypatch):
    import signal
    _,_,_,config,clock=setup(tmp_path,monkeypatch)
    class Interrupted(SyntheticCollector):
        def capture(self,session,race):
            super().capture(session,race)
            signal.raise_signal(signal.SIGTERM)
    adapter=Interrupted(config,clock)
    def sleep(seconds):clock[0]+=timedelta(seconds=seconds)
    result=pilot.run_session(config,clock=lambda:clock[0],sleep=sleep,collector=adapter)
    assert result['status']=='ABORTED'
    assert len([x for x in adapter.calls if isinstance(x,tuple)])==1
    assert adapter.calls[-1]=='close'


def test_actual_native_planner_receives_melbourne_time_from_utc_clock(tmp_path,monkeypatch):
    from tests.test_live_capture_binding import reserved_alias_plan
    from scripts.autonomous_live_odds_capture import sidecar_path_for
    from race_collection.synchronous_manual_capture import runner_set_sha256
    allowance,claim,plan,_ = reserved_alias_plan(tmp_path)
    scope=allowance.scope
    native=plan['races'][0]
    evidence=Path(scope.value['evidence_root']);evidence.mkdir(parents=True,exist_ok=True)
    original=Path(native['csv_path'])
    target=evidence/'workers/0'/original.name;target.parent.mkdir(parents=True)
    target.write_bytes(original.read_bytes())
    target.with_name(target.name+'.metadata.json').write_bytes(sidecar_path_for(original).read_bytes())
    race_id=json.loads(claim.read_bytes())['item']['race_id']
    runners=[{'box':r['box_number'],'display_name':r['dog_name'],'identity':r['identity'],'source_native_runner_id':str(r['box_number'])}
             for r in native['expected_runners']]
    selected={'race_id':race_id,'url':native['thedogs_source_url'],'jump_at':native['jump_datetime'],
              'source_native_race_id':'synthetic-9','runners':runners}
    row={'race_id':race_id,'race_id_aliases':[race_id,native['race_id']],'race_url':selected['url'],
         'jump_datetime':selected['jump_at'],'source_native_race_id':'synthetic-9','runners':runners,
         'runner_set_sha256':'a'*64}
    put(evidence/'refresh.json',{'sidecar_metadata_coverage':{'races':[{'race_url':selected['url'],'csv_path':str(target)}]}})
    view=SimpleNamespace(races=[row],source_refresh_report_path='refresh.json',packet_sha256='b'*64)
    import race_collection.synchronous_manual_capture as captures
    monkeypatch.setattr(captures,'bounded_current_race_index',lambda **kwargs:view)
    stamp=datetime.fromisoformat('2026-06-10T04:40:05+00:00')
    monkeypatch.setattr(pilot,'utc_now',lambda:stamp)
    collector=pilot.Collector({'state_root':str(tmp_path),'pilot_campaign_authority':{'sha256':'c'*64}})
    collector.plan={'evidence_root':str(evidence),'source_root':str(tmp_path),'python':'/synthetic/python','db_path':'/synthetic/db',
                    'lock_path':'/synthetic/lock'}
    collector.scope=SimpleNamespace(value={'source_date':'2026-06-10'})
    task=collector.task(selected)
    assert task['capture_window_minutes']==10
    command=collector.command('refresh','synthetic')
    assert command[command.index('--current-time')+1].endswith('+10:00')


def test_twenty_four_capture_cap_and_four_date_time_charges_are_separate(tmp_path,monkeypatch):
    campaign,_,_,config,clock=setup(tmp_path,monkeypatch)
    for day in DATES:
        clock[0]=datetime.fromisoformat(day+'T12:40:00').replace(tzinfo=pilot.ZONE)
        campaign.begin(day,now=clock[0],deadline=clock[0]+timedelta(seconds=7200))
        rows=population(Path(config['state_root'])/'sessions'/day,'1'*64)
        clock[0]=clock[0].replace(hour=13)
        for n,row in enumerate(rows[:6]):campaign.consume('/synthetic/'+day+'/'+str(n),item_for(row))
        clock[0]=clock[0].replace(hour=14,minute=40)
        campaign.close(day,now=clock[0])
    usage=campaign.development_usage(json.loads((campaign.root/'ledger.json').read_bytes()))
    assert usage['capture_attempts']==24 and usage['live_seconds']==28800
    clock[0]=clock[0].replace(hour=13)
    assert not campaign.available()
    with pytest.raises(ValueError,match='time_exhausted|slot_consumed'):
        campaign.begin('replenishment',now=clock[0],deadline=clock[0]+timedelta(minutes=1))
