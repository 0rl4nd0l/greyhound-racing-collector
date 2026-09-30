"""Public result-worker seam; all provider responses and race records synthetic."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from race_collection.development_examples import canonical, digest, put, race_key


def test_empty_queue_is_read_only_and_makes_no_provider_call(tmp_path):
    from race_collection.development_pilot_results import inspect_queue
    config = {'schema_version': 'development_pilot_runtime_v1', 'status': 'SYNTHETIC_FIXTURE',
        'authority_reference': 'synthetic fixture only', 'state_root': str(tmp_path/'state'),
        'allocation': {'path': str(tmp_path/'allocation.json'), 'sha256': ''},
        'result_closure_at': '2030-10-25T12:00:00+11:00', 'max_result_operations': 72,
        'max_result_transport_requests': 720, 'max_result_checks_per_race': 3}
    put(tmp_path/'allocation.json', {'allocation_id': 'synthetic_development', 'status': 'SYNTHETIC_FIXTURE'})
    config['allocation']['sha256'] = digest((tmp_path/'allocation.json').read_bytes())
    pin = put(tmp_path/'config.json', config)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    status = inspect_queue(tmp_path/'config.json', pin)
    assert status == {'status': 'NO_WORK', 'ready': 0, 'due': 0, 'completed': 0,
                      'operations_consumed': 0, 'transport_requests_consumed': 0,
                      'transport_requests_unconfirmed': 0, 'rejected': 0, 'synthetic': True}
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert not (tmp_path/'state').exists()

import io
from types import SimpleNamespace
import pytest
from tests.test_development_examples import synthetic


@pytest.fixture
def pilot(synthetic, tmp_path):
    from race_collection.development_examples import seal
    seed, control, race_id, access, pin = synthetic
    root=tmp_path/'runtime'
    root.mkdir(mode=0o700)
    output=root/'sessions'/'synthetic'/'packet'
    seal(access,pin,race_id,output)
    completion=json.loads((output/'completion.json').read_bytes())
    packet=json.loads((output/'pre_result.json').read_bytes())
    jump=datetime.fromisoformat(packet['times']['scheduled_jump_at'])
    access_value=json.loads(access.read_bytes())
    allocation=access_value['allocation']
    member=access_value['members'][race_id]
    closure=(jump+timedelta(days=25)).replace(hour=12,minute=0,second=0,microsecond=0)
    config={'schema_version':'development_pilot_runtime_v1','status':'SYNTHETIC_FIXTURE',
        'authority_reference':'synthetic fixture only','allocation':allocation,'state_root':str(root),
        'result_closure_at':closure.isoformat(),'max_result_operations':72,
        'max_result_transport_requests':720,'max_result_checks_per_race':3}
    ready={'schema_version':'development_pilot_capture_ready_v1','race_id':race_id,
        'race_key':race_key(race_id),'jump_at':jump.isoformat(),'access_path':str(access),
        'access_sha256':pin,'example_dir':str(output),'pre_result_sha256':completion['pre_result_sha256'],
        'completion_sha256':digest((output/'completion.json').read_bytes()),'job_id':member['entry']['job_id'],
        'prediction_entry':member['entry'],'published_at':completion['sealed_at'],'allocation_sha256':allocation['sha256']}
    put(root/'ready'/(digest(race_id.encode())+'.json'),ready)
    cfg=tmp_path/'config.json';cfg_pin=put(cfg,config)
    return SimpleNamespace(config=cfg,pin=cfg_pin,root=root,output=output,jump=jump,closure=closure,packet=packet)


class Session:
    def __init__(self, markup, status=200, headers=None):
        self.markup=markup;self.status=status;self.headers=headers or {};self.calls=[]
    def get(self,url,**kwargs):
        self.calls.append((url,kwargs))
        return SimpleNamespace(raw=SimpleNamespace(read=lambda size,decode_content=False:self.markup.encode()[:size]),status_code=self.status,url=url,
            headers={'content-type':'text/html',**self.headers},close=lambda:None)


def official_html(packet, change=None):
    rows=[]
    for n,runner in enumerate(packet['runners'],1):
        name=runner['display_name'];position=f'{n}th'
        if change=='wrong_name' and n==1:name='Synthetic Wrong'
        if change=='dead_heat' and n==2:position='1st'
        rows.append(f'<tr class="race-runner"><td class="race-runners__finish-position">{position}</td>'
            f'<td class="race-runners__box"><img name="rug_{runner["box_number"]}"></td>'
            f'<td class="race-runners__name">{name}</td></tr>')
    return '<table class="race-runners--result">'+''.join(rows)+'</table>'


def test_exact_official_result_closes_one_sealed_example_without_repeat(pilot):
    from race_collection.development_pilot_results import run_cycle, inspect_queue
    session=Session(official_html(pilot.packet))
    now=pilot.jump+timedelta(minutes=30,seconds=1)
    status=run_cycle(pilot.config,pilot.pin,now=now,session=session)
    assert status['status']=='ASSEMBLED' and status['trainable'] and status['synthetic']
    assert len(session.calls)==1
    assert session.calls[0][0]==pilot.packet['race']['url']+'?trial=false'
    assert session.calls[0][1]['allow_redirects'] is False
    assert inspect_queue(pilot.config,pilot.pin,now=now)['completed']==1
    assert run_cycle(pilot.config,pilot.pin,now=pilot.closure,session=session)['status']=='NO_WORK'
    assert len(session.calls)==1
    example=json.loads((pilot.output/'example.json').read_bytes())
    assert example['result']['disposition']=='OFFICIAL'
    assert all('source_native_runner_id' not in r for r in example['result']['official_evidence']['runner_rows'])


def test_future_nomination_does_not_decode_packet_or_make_request(pilot, monkeypatch):
    from race_collection.development_pilot_results import run_cycle
    original=Path.read_bytes
    def guard(path):
        assert path.name not in {'pre_result.json','completion.json','official-result.json'}
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',guard)
    session=Session('never consumed')
    assert run_cycle(pilot.config,pilot.pin,now=pilot.jump,session=session)['status']=='NO_WORK'
    assert not session.calls


@pytest.mark.parametrize('change',['wrong_name','dead_heat','duplicate_box'])
def test_ambiguous_field_never_becomes_training_label(pilot,change):
    from race_collection.development_pilot_results import run_cycle
    markup=official_html(pilot.packet,change)
    if change=='duplicate_box':markup=markup.replace('</table>',markup.split('<table class="race-runners--result">')[1])
    session=Session(markup)
    result=run_cycle(pilot.config,pilot.pin,now=pilot.closure-timedelta(minutes=29),session=session)
    assert result['status']=='ASSEMBLED' and result['disposition']=='AMBIGUOUS' and not result['trainable']
    assert len(session.calls)==1


def test_three_fixed_checks_preserve_missing_and_do_not_retry(pilot):
    from race_collection.development_pilot_results import run_cycle,inspect_queue
    from zoneinfo import ZoneInfo
    session=Session('<html>No result yet</html>')
    first=pilot.jump+timedelta(minutes=30)
    second=(pilot.jump.astimezone(ZoneInfo('Australia/Melbourne'))+timedelta(days=1)).replace(hour=12,minute=0,second=0,microsecond=0)
    for instant in (first,second):
        assert run_cycle(pilot.config,pilot.pin,now=instant,session=session)['status']=='CHECK_RETAINED'
        assert run_cycle(pilot.config,pilot.pin,now=instant+timedelta(seconds=1),session=session)['status']=='NO_WORK'
    result=run_cycle(pilot.config,pilot.pin,now=pilot.closure-timedelta(minutes=30),session=session)
    assert result['disposition']=='MISSING' and not result['trainable']
    status=inspect_queue(pilot.config,pilot.pin,now=pilot.closure)
    assert status['operations_consumed']==status['transport_requests_consumed']==3
    assert len(session.calls)==3


def test_after_hard_closure_seals_missing_without_transport(pilot):
    from race_collection.development_pilot_results import run_cycle
    session=Session(official_html(pilot.packet))
    result=run_cycle(pilot.config,pilot.pin,now=pilot.closure+timedelta(seconds=1),session=session)
    assert result['disposition']=='MISSING'
    assert session.calls==[]


@pytest.mark.parametrize('status,headers',[(403,{}),(429,{}),(200,{'Retry-After':'20'})])
def test_denial_is_durable_no_unchanged_retry_even_at_next_milestone(pilot,status,headers):
    from race_collection.development_pilot_results import run_cycle
    session=Session('denied',status,headers)
    now=pilot.jump+timedelta(minutes=30)
    assert run_cycle(pilot.config,pilot.pin,now=now,session=session)['disposition']=='MISSING'
    assert (pilot.root/'results'/'source-stop.json').exists()
    assert run_cycle(pilot.config,pilot.pin,now=pilot.closure-timedelta(minutes=30),session=session)['status']=='SOURCE_STOP'
    assert run_cycle(pilot.config,pilot.pin,now=pilot.closure,session=session)['disposition']=='MISSING'
    assert len(session.calls)==1


def test_interrupted_consumed_final_attempt_closes_without_replenishing(pilot):
    from race_collection.development_pilot_results import run_cycle, inspect_queue
    race=pilot.packet['race']['race_id']
    (pilot.root/'results').mkdir(mode=0o700)
    (pilot.root/'results'/digest(race.encode())).mkdir(mode=0o700)
    attempt=pilot.root/'results'/digest(race.encode())/'attempt-2'
    put(attempt/'started.json',{'race_id':race,'stage':2,'at':(pilot.closure-timedelta(minutes=30)).isoformat()})
    session=Session(official_html(pilot.packet))
    result=run_cycle(pilot.config,pilot.pin,now=pilot.closure,session=session)
    assert result['disposition']=='MISSING' and not session.calls
    assert inspect_queue(pilot.config,pilot.pin,now=pilot.closure)['operations_consumed']==1


@pytest.mark.parametrize('corrupt_first',[False,True])
def test_two_independent_ready_races_get_final_checks_before_noon(pilot,tmp_path,monkeypatch,corrupt_first):
    from tests.test_operational_prediction_packaged import test_packaged_capture_retention_frozen_prediction
    from tests.fixtures.development_pipeline import prepare_access
    from race_collection.development_examples import seal
    from race_collection.development_pilot_results import run_cycle, inspect_queue
    seed=tmp_path/'second-capture';seed.mkdir()
    test_packaged_capture_retention_frozen_prediction(seed,monkeypatch,False,'sandown_park')
    race_id, access, _=prepare_access(seed,tmp_path/'second-control')
    value=json.loads(access.read_bytes())
    cfg=json.loads(pilot.config.read_bytes())
    value['allocation']=cfg['allocation']
    access.write_bytes(canonical(value));pin=digest(access.read_bytes())
    output=pilot.root/'sessions'/'synthetic-second'/'packet'
    seal(access,pin,race_id,output)
    completion=json.loads((output/'completion.json').read_bytes())
    member=value['members'][race_id]
    put(pilot.root/'ready'/(digest(race_id.encode())+'.json'),{
        'schema_version':'development_pilot_capture_ready_v1','race_id':race_id,'race_key':race_key(race_id),
        'jump_at':member['jump_at'],'access_path':str(access),'access_sha256':pin,'example_dir':str(output),
        'pre_result_sha256':completion['pre_result_sha256'],
        'completion_sha256':digest((output/'completion.json').read_bytes()),'job_id':member['entry']['job_id'],
        'prediction_entry':member['entry'],'published_at':completion['sealed_at'],
        'allocation_sha256':cfg['allocation']['sha256']})
    now=pilot.closure-timedelta(minutes=30)
    session=Session('<html>No result yet</html>')
    assert inspect_queue(pilot.config,pilot.pin,now=now)['due']==2
    if corrupt_first:
        nomination=next(p for p in (pilot.root/'ready').glob('*.json') if json.loads(p.read_bytes())['race_id']==pilot.packet['race']['race_id'])
        value=json.loads(nomination.read_bytes());value['completion_sha256']='0'*64
        nomination.write_bytes(canonical(value))
    for minute in range(2):
        result=run_cycle(pilot.config,pilot.pin,now=now+timedelta(minutes=minute),session=session)
        if corrupt_first and minute==0:assert result['status']=='NOMINATION_REJECTED'
        else:assert result['disposition']=='MISSING'
    status=inspect_queue(pilot.config,pilot.pin,now=pilot.closure)
    assert status['completed']==2 and status['operations_consumed']==2-int(corrupt_first)
    assert status['rejected']==int(corrupt_first)
    assert len(session.calls)==2-int(corrupt_first)


def test_published_result_recovers_restricted_join_without_new_request(pilot):
    from race_collection.development_pilot_results import run_cycle,inspect_queue
    session=Session(official_html(pilot.packet))
    now=pilot.jump+timedelta(minutes=30)
    result=run_cycle(pilot.config,pilot.pin,now=now,session=session)
    directory=pilot.root/'results'/digest(pilot.packet['race']['race_id'].encode())
    # Model a crash after durable result/example but before queue completion.
    (directory/'complete.json').unlink()
    before=(pilot.output/'example.json').read_bytes()
    assert inspect_queue(pilot.config,pilot.pin,now=now)['due']==1
    assert run_cycle(pilot.config,pilot.pin,now=now,session=session)==result
    assert (pilot.output/'example.json').read_bytes()==before
    assert len(session.calls)==1


def test_wrong_nomination_seal_rejects_before_transport_or_attempt_charge(pilot):
    from race_collection.development_pilot_results import run_cycle
    from race_collection.development_examples import DevelopmentRejected
    path=next((pilot.root/'ready').glob('*.json'))
    value=json.loads(path.read_bytes());value['completion_sha256']='0'*64
    path.write_bytes(canonical(value))
    session=Session(official_html(pilot.packet))
    result=run_cycle(pilot.config,pilot.pin,now=pilot.jump+timedelta(minutes=30),session=session)
    assert result['status']=='NOMINATION_REJECTED' and result['reason']=='SOURCE_HASH_CHANGED'
    assert not session.calls and not list((pilot.root/'results').glob('*/attempt-*/started.json'))


def test_charge_acknowledgement_write_failure_preserves_visible_reservation(pilot,tmp_path,monkeypatch):
    import os
    from race_collection.development_pilot_results import run_cycle,inspect_queue
    from tests.test_refresh_shared_sportsbet_snapshot import access as source_access
    source=source_access(tmp_path/'source')
    campaign_root=tmp_path/'campaign';campaign_root.mkdir()
    put(campaign_root/'ledger.json',{'source_holds':[]})
    config=json.loads(pilot.config.read_bytes())
    config.update(source_state=str(source),campaign_root=str(campaign_root))
    pilot.config.write_bytes(canonical(config));pilot.pin=digest(pilot.config.read_bytes())
    charges=[]
    campaign=SimpleNamespace(request=lambda **kwargs:charges.append(kwargs),hold_source=lambda _:None)
    original=os.open
    def fail_ack(path,*args,**kwargs):
        if Path(path).name=='charge.json':raise OSError('synthetic fsync/ack interruption')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(os,'open',fail_ack)
    session=Session(official_html(pilot.packet))
    now=pilot.jump+timedelta(minutes=30)
    assert run_cycle(pilot.config,pilot.pin,now=now,session=session,campaign=campaign)['status']=='CHECK_RETAINED'
    status=inspect_queue(pilot.config,pilot.pin,now=now)
    assert status['operations_consumed']==status['transport_requests_consumed']==status['transport_requests_unconfirmed']==1
    assert charges==[{'kind':'results'}] and session.calls==[]
    assert run_cycle(pilot.config,pilot.pin,now=now+timedelta(seconds=1),session=session,campaign=campaign)['status']=='NO_WORK'


def test_parent_traversal_nomination_is_quarantined_before_packet_or_result_access(pilot):
    from race_collection.development_pilot_results import run_cycle
    path=next((pilot.root/'ready').glob('*.json'))
    value=json.loads(path.read_bytes())
    value['example_dir']=str(pilot.root/'sessions'/'..'/'outside-pilot-packet')
    path.write_bytes(canonical(value))
    session=Session('must not be read')
    result=run_cycle(pilot.config,pilot.pin,now=pilot.jump+timedelta(minutes=30),session=session)
    assert result['status']=='NOMINATION_REJECTED' and result['reason']=='RESULT_EXAMPLE_OUTSIDE_PILOT'
    assert not session.calls


def test_cli_reports_existing_collector_contention_without_failure_alert(monkeypatch,capsys):
    import sys
    import scripts.development_pilot_results as command
    from race_collection.synchronous_manual_capture import CollectorBusy
    def existing_owner(*args):
        raise CollectorBusy({'pid':123,'run_id':'synthetic-existing-owner'})
    monkeypatch.setattr(command,'run_cycle',existing_owner)
    monkeypatch.setattr(sys,'argv',['results','cycle','--config','unused','--config-sha256','unused'])
    assert command.main()==0
    assert json.loads(capsys.readouterr().out)=={'status':'RESULT_COLLECTOR_BUSY'}


@pytest.mark.parametrize('variable',['SYNTHETIC_DEVELOPMENT_CLOCK','FRESHNESS_FABRICATED_SOURCE','GREYHOUND_SHARED_SNAPSHOT_FIXTURE'])
def test_real_result_configuration_rejects_fixture_environment_before_authority_read(tmp_path,monkeypatch,variable):
    from race_collection.development_pilot_results import inspect_queue
    from race_collection.development_examples import DevelopmentRejected
    path=tmp_path/'real-shaped.json';pin=put(path,{'status':'AUTHORIZED'})
    monkeypatch.setenv(variable,'/synthetic/fixture')
    with pytest.raises(DevelopmentRejected,match='RESULT_FIXTURE_ENVIRONMENT_FORBIDDEN'):
        inspect_queue(path,pin)


def test_study_restoration_hold_blocks_results_despite_healthy_result_queue(tmp_path):
    from race_collection.development_pilot_results import _study_priority
    from race_collection.development_examples import DevelopmentRejected
    from race_collection.development_pilot import reference
    now=datetime.fromisoformat('2026-10-03T13:40:00+10:00')
    study=tmp_path/'study';result=tmp_path/'study-results'
    healthy={'status':'NO_SLOT_DUE','at':now.isoformat()}
    put(study/'health.json',healthy)
    put(study/'canary.json',{'status':'CANARY_STRUCTURALLY_VERIFIED','plan_sha256':'a'*64,
        'verified_predictions':1,'closed_results':1})
    put(result/'health.json',{'status':'CYCLE_COMPLETE','at':now.isoformat(),'counts':{},'oldest_due':None})
    authority=tmp_path/'result-authority.json';put(authority,{'runtime':{'state_root':str(result)}})
    binding=tmp_path/'result-binding.json';ref=reference(authority)
    put(binding,{'authority':ref['path'],'authority_sha256':ref['sha256']})
    schedule=tmp_path/'schedule.json';put(schedule,{'slots':['2026-10-02T13:00:00+10:00'],
        'session_minutes':90,'state_root':str(study),'comparison_plan_sha256':'a'*64,'result_binding':str(binding)})
    cfg={'status':'AUTHORIZED','study_schedule':reference(schedule)}
    _study_priority(cfg,now)
    (study/'health.json').write_bytes(canonical({**healthy,'status':'RESTORATION_HELD'}))
    with pytest.raises(DevelopmentRejected,match='RESULT_STUDY_RECOVERY_OR_RETENTION_PRIORITY'):
        _study_priority(cfg,now)
    (study/'health.json').write_bytes(canonical(healthy))
    (study/'canary.json').unlink()
    with pytest.raises(DevelopmentRejected,match='RESULT_STUDY_RECOVERY_OR_RETENTION_PRIORITY'):
        _study_priority(cfg,now)
