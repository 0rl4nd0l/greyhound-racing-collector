"""Invented control and queue metadata only; no race outcomes or transports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,sqlite3
import pytest
from scripts.run_comparison_schedule import verify_canary
from race_collection.live_freshness_contract import digest


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n')
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture(tmp_path):
    root=tmp_path/'state';old=root/'slots/001';original=old/'synthetic-001'
    parent=root/'continuations/001/001';package=parent/'package';pred=tmp_path/'pred';campaign=tmp_path/'campaign';results=tmp_path/'results';results.mkdir()
    plan_sha='a'*64;user='SYNTHETIC_USER_CONTINUATION'
    original_plan=put(original/'plan.json',{'rehearsal_id':'old','starts_at':'2026-10-01T13:00:00+10:00'})
    authority={'schema_version':'scientific_session_continuation_v1','status':'AUTHORIZED_CONTINUATION','authority_reference':user,
      'issued_at':'2026-10-01T14:20:00+10:00','original_slot':'001','original_admission':put(old/'admission.json',{'slot':'2026-10-01T13:00:00+10:00'}),
      'original_terminal':put(old/'terminal.json',{'status':'FAILED_RESTORED'}),'original_plan':original_plan,
      'original_restored':put(original/'restored.json',{'status':'RESTORED_COLLECTOR_TRIGGERS_HELD','sportsbet_hold':False}),
      'comparison_plan_sha256':plan_sha,'campaign_root':str(campaign),'prediction_root':str(pred),'source_commit':'b'*40,
      'source_lease':{'max_operations':192,'expires_at':datetime.fromisoformat('2026-10-01T15:50:00+10:00').timestamp(),
                      'operation_start':632,'prior_state_sha256':'c'*64,'reference':'SYNTHETIC_PROGRAMME:slot:1'},
      'source_operations_total_limit':192,'prior_charged_seconds':3121,'total_charge_ceiling_seconds':7260,
      'prior_prediction_requests':8467,'prediction_request_total_limit':16000,'continuation_request_cap':7533,
      'starts_at':'2026-10-01T14:35:00+10:00','ends_at':'2026-10-01T15:05:00+10:00','cleanup_seconds':1860,
      'original_failure_preserved':True,'consumed_races_retriable':False,'new_source_allocation':False,'outcome_values_accessed':False,'metrics_computed':False}
    authority['source_lease_sha256']=digest(authority['source_lease'])
    put(old/'source-lease.json',{'slot':'1','operation_start':632,'before_sha256':'c'*64})
    put(tmp_path/'source.json',{'diagnostic_authorizations':[authority['source_lease']]})
    plan={'rehearsal_id':'new','commit':'b'*40,'starts_at':authority['starts_at'],'ends_at':authority['ends_at'],'cleanup_seconds':1860,
      'max_logical_requests':7533,'campaign_root':str(campaign),'prediction_root':str(pred),'frozen_comparison':{'sha256':plan_sha}}
    authority['continuation_plan']=put(package/'plan.json',plan)
    reference=put(parent/'authority.json',authority)
    put(package/'started.json',{'approval_id':user,'plan_sha256':digest(plan)})
    put(package/'measurement.json',{'status':'REHEARSAL_MEASURED_NOT_RELEASED','logical_requests':100})
    put(package/'restored.json',{'status':'RESTORED_COLLECTOR_TRIGGERS_HELD','sportsbet_hold':False})
    put(parent/'terminal.json',{'status':'COMPLETED','returncode':0,'authority_sha256':digest(authority),'plan_sha256':digest(plan),'outcomes_released':False})
    put(campaign/'ledger.json',{'launches':{'old':{'charged_seconds':3121,'closed_at':'2026-10-01T13:52:00+10:00'},'new':{'charged_seconds':1900,'closed_at':'2026-10-01T15:07:00+10:00'}}})
    race='Race 1 - INVENTED - 2026-10-01';key=hashlib.sha256(race.encode()).hexdigest()
    put(pred/'dispatches'/f'{key}.json',{'plan':str(original/'plan.json'),'race_id':race})
    put(pred/'races'/key/'terminal.json',{'status':'PREDICTION_READY','job_id':'invented_job'})
    with sqlite3.connect(results/'queue.sqlite3') as db:
        db.execute('CREATE TABLE jobs(job TEXT,state TEXT)');db.execute('INSERT INTO jobs VALUES(?,?)',('invented_job','CLOSED'))
    cfg={'programme_id':'synthetic','authority_reference':'SYNTHETIC_PROGRAMME','source_state':str(tmp_path/'source.json'),
         'comparison_plan_sha256':plan_sha,'campaign_root':str(campaign),'prediction_root':str(pred),
         'slots':['2026-10-01T13:00:00+10:00'],'first_session_continuation':reference}
    return cfg,{'state_root':str(results)},root,parent


def test_completed_authorised_continuation_accepts_canary_without_rewriting_failure(tmp_path):
    cfg,results,root,parent=fixture(tmp_path);old=root/'slots/001/terminal.json';before=old.read_bytes()
    assert verify_canary(cfg,results,root,datetime(2026,10,1,6,tzinfo=timezone.utc))
    assert old.read_bytes()==before
    canary=json.loads((root/'canary.json').read_bytes())
    assert canary['continuation']['original_failure_preserved'] is True
    assert canary['closed_results']==1


def test_regular_tick_accepts_completed_continuation_without_new_admission(tmp_path,monkeypatch):
    from scripts import run_comparison_schedule as schedule
    cfg,results,root,parent=fixture(tmp_path)
    cfg['state_root']=str(root)
    cfg['result_binding']=str(tmp_path/'binding.json')
    put(Path(cfg['result_binding']),{'synthetic':True})
    monkeypatch.setattr(schedule,'load_config',lambda _: (cfg,{'ends_at':'2099-01-01T00:00:00+00:00'}))
    monkeypatch.setattr('src.predictor.comparison_result_runtime.load_runtime',lambda binding,now: ({},{},results))
    monkeypatch.setattr(schedule,'renew_source',lambda *args,**kwargs: pytest.fail('must not allocate another source lease'))
    monkeypatch.setattr(schedule,'prepare_session',lambda *args,**kwargs: pytest.fail('must not admit another collection'))
    assert schedule.tick(tmp_path/'invented-config.json')=={'status':'NO_SLOT_DUE'}
    assert (root/'canary.json').exists()


def test_status_includes_continuation_opportunities_and_preserves_failed_slot(tmp_path):
    from scripts.comparison_status import programme
    cfg,results,root,parent=fixture(tmp_path)
    cfg.update(state_root=str(root),source_commit='b'*40)
    put(parent/'package/measurement.json',{'sample_count':3,'windows':{'eligible_observed_windows':[{},{}],
        'attempted_windows':[{}]},'index_status':'AVAILABLE/FRESH','source_age_seconds':41})
    value=programme(cfg,{},now=datetime(2026,10,1,6,tzinfo=timezone.utc))
    assert value['sessions']=={'FAILED_RESTORED':1}
    assert value['continuations'][0]['status']=='COMPLETED'
    assert value['observed_opportunities']==2 and value['attempted_captures']==1
    assert value['input_freshness']['source_age_seconds']==41


def test_later_regular_slot_supersedes_continuation_freshness(tmp_path):
    from scripts.comparison_status import programme
    cfg,results,root,parent=fixture(tmp_path)
    cfg.update(state_root=str(root),source_commit='b'*40)
    put(parent/'package/measurement.json',{'sample_count':3,'windows':{'eligible_observed_windows':[{},{}]},
        'last_sample_at':'2026-10-01T15:04:00+10:00','index_status':'AVAILABLE/FRESH','source_age_seconds':41})
    put(root/'slots/002/synthetic-002/progress.json',{'sample_count':2,'windows':{'eligible_observed_windows':[{}]},
        'last_sample_at':'2026-10-02T13:12:00+10:00','index_status':'AVAILABLE/FRESH','source_age_seconds':12})
    value=programme(cfg,{},now=datetime(2026,10,2,4,tzinfo=timezone.utc))
    assert value['input_freshness']['last_sample_at']=='2026-10-02T13:12:00+10:00'
    assert value['input_freshness']['source_age_seconds']==12
    assert value['observed_opportunities']==3 and value['observation_samples']==5


@pytest.mark.parametrize('damage',['none','missing_resolution','omitted_charge','open_prior','changed_ack'])
def test_resolved_prior_continuation_remains_in_cumulative_gate(tmp_path,damage):
    cfg,results,root,parent=fixture(tmp_path)
    authority=json.loads((parent/'authority.json').read_bytes())
    plan=json.loads((parent/'package/plan.json').read_bytes())
    previous=root/'continuations/001/000';previous_plan=put(previous/'earlier/plan.json',{**plan,'rehearsal_id':'earlier'})
    previous_authority=put(previous/'authority.json',{**authority,'continuation_plan':previous_plan})
    terminal=put(previous/'terminal.json',{'status':'RESTORATION_HELD','plan_sha256':previous_plan['sha256'],
        'authority_sha256':previous_authority['sha256']})
    ack=put(previous/'ack.json',{'plan_sha256':previous_plan['sha256'],'collection_resume_allowed':False})
    restored=put(previous/'earlier/restored.json',{'status':'RESTORED_COLLECTOR_TRIGGERS_HELD','sportsbet_hold':False,
        'r3_replacement_acknowledgement':{'sha256':ack['sha256']}})
    put(previous/'earlier/failure.json',{'reason':'installed_r3_changed'})
    resolution=put(previous/'resolution.json',{'status':'RESTORATION_COMPLETED_AFTER_EXPLICIT_UI_REPLACEMENT',
        'original_terminal_sha256':terminal['sha256'],'restored_sha256':restored['sha256'],
        'plan_sha256':previous_plan['sha256'],'ack_sha256':ack['sha256'],'collection_resumed':False,'outcomes_released':False})
    authority['prior_continuations']=[{'authority':previous_authority,'plan':previous_plan,'terminal':terminal,
        'r3_ack':ack,'restored':restored,'resolution':resolution}]
    authority['original_charged_seconds']=3121;authority['prior_charged_seconds']=3421
    if damage=='omitted_charge':authority['prior_charged_seconds']=3121
    cfg['first_session_continuation']=put(parent/'authority.json',authority)
    put(parent/'terminal.json',{'status':'COMPLETED','returncode':0,'authority_sha256':digest(authority),
        'plan_sha256':digest(plan),'outcomes_released':False})
    path=Path(cfg['campaign_root'])/'ledger.json';ledger=json.loads(path.read_bytes())
    ledger['launches']['earlier']={'charged_seconds':300,'closed_at':None if damage=='open_prior' else '2026-10-01T14:25:00+10:00'}
    put(path,ledger)
    if damage=='missing_resolution':Path(resolution['path']).unlink()
    if damage=='changed_ack':Path(ack['path']).write_text('{}')
    assert verify_canary(cfg,results,root,datetime(2026,10,1,6,tzinfo=timezone.utc)) is (damage=='none')


@pytest.mark.parametrize('damage',['missing_authority','authority_changed','unfinished','missing_measurement','missing_restore','open_lease','over_budget','negative_charge','nan_charge','different_source_lease','no_closed_result'])
def test_continuation_never_bypasses_missing_proof(tmp_path,damage):
    cfg,results,root,parent=fixture(tmp_path)
    if damage=='missing_authority':cfg.pop('first_session_continuation')
    elif damage=='authority_changed':(parent/'authority.json').write_text('{}')
    elif damage=='unfinished':(parent/'terminal.json').unlink()
    elif damage=='missing_measurement':(parent/'package/measurement.json').unlink()
    elif damage=='missing_restore':(parent/'package/restored.json').unlink()
    elif damage in {'open_lease','over_budget','negative_charge','nan_charge'}:
        p=Path(cfg['campaign_root'])/'ledger.json';d=json.loads(p.read_bytes())
        d['launches']['new']['closed_at']=None if damage=='open_lease' else d['launches']['new']['closed_at']
        if damage=='over_budget':d['launches']['new']['charged_seconds']=6000
        if damage=='negative_charge':d['launches']['new']['charged_seconds']=-1
        if damage=='nan_charge':d['launches']['new']['charged_seconds']=float('nan')
        put(p,d)
    elif damage=='different_source_lease':
        put(Path(cfg['source_state']),{'diagnostic_authorizations':[]})
    elif damage=='no_closed_result':
        with sqlite3.connect(Path(results['state_root'])/'queue.sqlite3') as db:db.execute("UPDATE jobs SET state='PENDING'")
    assert not verify_canary(cfg,results,root,datetime(2026,10,1,6,tzinfo=timezone.utc))
    assert not (root/'canary.json').exists()
