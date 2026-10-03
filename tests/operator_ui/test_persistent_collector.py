"""Read-only binding, rollover, stale status, verification and access boundaries."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

from src.operator_ui import persistent_collector as pc
from src.operator_ui import retained_forecasts as rf
from tests.operator_ui.test_security import configured_app, login

NOW = datetime(2026, 10, 3, 5, 30, tzinfo=timezone.utc)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def daily(tmp_path, monkeypatch):
    root = tmp_path/'state'
    config = tmp_path/'config.json'
    authority = tmp_path/'authority.json'
    ref = {'path':str(authority), 'sha256':put(authority, {'state_root':str(root), 'engineering_only':True, 'human_outcome_access':False})}
    cfg = {'source_commit':'a'*40, 'python':str(tmp_path/'python'), 'standing_authority':ref}
    binding = {'config':str(config), 'config_sha256':put(config,cfg), 'source':str(tmp_path/'source'), 'source_commit':'a'*40, 'python':cfg['python']}
    plans = {}
    future = types.ModuleType('src.predictor.future_comparison')
    def load_plan(path, sha):
        value = pc.read(path,sha)
        return value, None
    future.load_plan = load_plan
    future.verify_comparison = lambda *a, **k: (_ for _ in ()).throw(ValueError('invalid comparison'))
    persistent = types.ModuleType('race_collection.persistent_comparison')
    persistent.validate_persistent_plan = lambda plan:plans[plan['date']]
    monkeypatch.setitem(sys.modules,'src.predictor.future_comparison',future)
    monkeypatch.setitem(sys.modules,'race_collection.persistent_comparison',persistent)
    def publish(day, title):
        output=root/'days'/day/'native-test'
        allocation={'racing_date':day,'standing_authority':ref,'prediction_root':str(tmp_path/'predictions'/day),'max_capture_attempts':255}
        plan={'date':day,'programme_root':str(root/'days'/day/'admission')}
        plans[day]=allocation
        plan_path=output.parent/'comparison.json';sha=put(plan_path,plan)
        native_path=output/'plan.json';native_sha=put(native_path,{'commit':'a'*40})
        put(output.parent/'native-prepared.json',{'at':NOW.isoformat(),'plan':{'path':str(native_path),'sha256':native_sha},'standing_authority':ref,'output':str(output),'racing_date':day,'comparison':{'path':str(plan_path),'sha256':sha}})
        inventory=output/'inventories/test.json'
        inventory_sha=put(inventory,{'observed_at':(NOW-timedelta(hours=1)).isoformat(),'race_count':2,'races':[
            {'title':title,'venue_name':'Venue','race_number':'1','scheduled_jump_datetime':(NOW+timedelta(hours=1)).isoformat()},
            {'title':'Past race','venue_name':'Venue','race_number':'2','scheduled_jump_datetime':(NOW-timedelta(minutes=1)).isoformat()}]})
        put(output/'persistent-health.json',{'source_commit':'a'*40,'source_date':day,'at':NOW.isoformat(),'status':'WAITING_FOR_RACE','inventory':{'path':str(inventory),'sha256':inventory_sha}})
        put(root/'current-day.json',{'racing_date':day,'output':str(output)})
        return output,sha
    output,sha=publish('2026-10-03','Today race')
    return binding,publish,output,sha,root


def test_discovery_is_separate_from_inputs_and_status(daily):
    binding,_,_,_,_=daily
    value=pc.snapshot(binding,NOW)
    assert value['state']=='WAITING_FOR_RACE'
    assert value['inventory_state']=='OLDER_INVENTORY'
    assert len(value['upcoming'])==1 and value['race_count']==2
    assert value['upcoming'][0]['forecast_readiness']=='NOT_ESTABLISHED_BY_DISCOVERY'
    assert value['forecasts']==[] and value['result_access'] is False
    assert value['scientific_admission']=='CANARY_NOT_VERIFIED'


def test_daily_rollover_follows_new_package(daily):
    binding,publish,_,_,_=daily
    assert pc.snapshot(binding,NOW)['upcoming'][0]['title']=='Today race'
    publish('2026-10-04','Next day race')
    value=pc.snapshot(binding,NOW)
    assert value['racing_date']=='2026-10-04' and value['upcoming'][0]['title']=='Next day race'


def test_hash_change_rejects_inventory(daily):
    binding,_,output,_,_=daily
    put(output/'inventories/test.json',{})
    with pytest.raises(ValueError,match='retained_identity_mismatch'):pc.snapshot(binding,NOW)


def test_stale_status_and_missing_package(daily):
    binding,_,output,_,root=daily
    assert pc.snapshot(binding,NOW+timedelta(minutes=6))['state']=='STATUS_STALE'
    put(root/'current-day.json',{'racing_date':'2026-10-03','output':str(root/'days/2026-10-02/native-test')})
    with pytest.raises(ValueError):pc.snapshot(binding,NOW)


def test_incomplete_or_unverified_comparison_is_never_displayed(daily):
    binding,_,output,sha,_=daily
    admission=output.parent/'admission'/sha/'attempts/test/admission.json'
    put(admission,{'job_id':'j'})
    assert pc.snapshot(binding,NOW)['forecasts']==[]
    put(admission.with_name('completion.json'),{'bundle_entry':{'directory':'bundle','manifest_sha256':'f'*64}})
    value=pc.snapshot(binding,NOW)
    assert value['forecasts']==[] and len(value['forecast_errors'])==1


def test_projection_requires_all_four_identity_matched_normalized_candidates():
    records={name:{'status':'SEALED','model_sha256':name,'predictions':[
        {'box_number':1,'identity':'dog1','dog_name':'Dog 1','probability':.6},
        {'box_number':2,'identity':'dog2','dog_name':'Dog 2','probability':.4}]} for name in pc.MODELS}
    native={'engineering_evidence':True,'future_race_evidence':False,'eligible_common_race':True,'records':records,'completion':{'published_complete_at':NOW.isoformat()}}
    admission={'race':{'race_id':'race','jump_timestamp':NOW.isoformat()},'job_id':'j'}
    value=pc.project_forecast(native,admission,'f'*64,NOW)
    assert set(value['runners'][0]['probabilities'])==set(pc.MODELS)
    native['records']['market']['predictions'][0]['probability']=float('nan')
    with pytest.raises(ValueError):pc.project_forecast(native,admission,'f'*64,NOW)


def test_worker_is_readonly_network_denied_and_timeout_fails_closed(daily,monkeypatch):
    binding,_,_,_,_=daily
    commands=[]
    def timeout(command,**kwargs):
        commands.append((command,kwargs));raise subprocess.TimeoutExpired(command,kwargs['timeout'])
    monkeypatch.setattr(pc.subprocess,'run',timeout)
    assert pc.observe(binding)['state']=='UNAVAILABLE'
    command,kwargs=commands[0]
    assert '--ro-bind' in command and '--unshare-net' in command and kwargs['timeout']==40


def test_authenticated_audited_no_store_api(daily,tmp_path,monkeypatch):
    binding,_,_,_,_=daily
    app=configured_app(tmp_path/'app')
    pc.install_display(app,binding,app.extensions['operator_ui_operational_get'])
    monkeypatch.setattr(pc,'observe',lambda _:pc.snapshot(binding,NOW))
    client=app.test_client();url='/operator-ui/api/v1/predictions/persistent'
    assert client.get(url).status_code==401
    assert login(client).status_code==200
    response=client.get(url)
    assert response.status_code==200 and response.json['race_count']==2
    assert 'no-store' in response.headers['Cache-Control']
    from src.operator_ui.security import AuditUnavailable
    monkeypatch.setattr(app.extensions['operator_ui_audit'],'append_and_confirm',lambda *a,**k:(_ for _ in ()).throw(AuditUnavailable('fixture')))
    assert client.get(url).status_code==503


@pytest.mark.parametrize('minutes',[0,40])
def test_runtime_hold_overrides_fresh_or_stale_daily_activity(daily,minutes):
    binding,_,output,_,root=daily
    health=pc.read(output/'persistent-health.json')
    health['status']='ACTIVE_COLLECTION'
    put(output/'persistent-health.json',health)
    put(root/'health.json',{'at':NOW.isoformat(),'status':'HOLD',
                          'reason':'operational_prediction_failed_preserved_consumption'})
    value=pc.snapshot(binding,NOW+timedelta(minutes=minutes))
    assert value['state']=='HOLD'
    assert value['status_source']=='runtime_hold'
    assert value['status_reason']=='OPERATIONAL_PREDICTION_FAILED_PRESERVED_CONSUMPTION'
    assert value['daily_state']=='ACTIVE_COLLECTION'
    assert value['forecasts']==[]


def test_failed_attempt_projection_never_exposes_probabilities_or_timing_success():
    native={'evidence_class':'AUTHORIZED_ENGINEERING','future_race_evidence':False,
            'records':{name:{'status':'FAILED','failure':'RESIDUAL_SCORER_FAILED','predictions':None} for name in pc.MODELS},
            'completion':{'status':'COMPLETE_BEFORE_CUTOFF','published_complete_at':NOW.isoformat()}}
    admission={'race':{'race_id':'Race 1 - DUBBO - 2026-10-03','jump_timestamp':NOW.isoformat()},'job_id':'job'}
    value=pc.project_failed_attempt(native,admission,'f'*64,NOW)
    assert value['status']=='FAILED'
    assert all(row['failure']=='RESIDUAL_SCORER_FAILED' for row in value['candidates'].values())
    assert 'COMPLETE_BEFORE_CUTOFF' not in json.dumps(value)
    assert 'predictions' not in json.dumps(value) and 'probabilities' not in json.dumps(value)
    native['records']['production']['failure']='secret/path/provider/token'
    assert pc.project_failed_attempt(native,admission,'f'*64,NOW)['candidates']['production']['failure']=='FAILURE_DETAIL_WITHHELD'


def test_failed_service_overrides_fresh_activity(monkeypatch):
    monkeypatch.setattr(pc.subprocess,'run',lambda *a,**k:types.SimpleNamespace(stdout='ActiveState=failed\nSubState=failed\nMainPID=0\nExecMainStatus=78\n'))
    value=pc.service_status({'state':'ACTIVE_COLLECTION'})
    assert value['state']=='FAILED' and value['service']['exit_status']==78


def test_successor_binding_retains_prior_package_and_failed_stop(daily):
    binding,_,output,_,root=daily
    cfg=pc.read(binding['config']);cfg['source_commit']='b'*40
    binding.update(source_commit='b'*40,config_sha256=put(Path(binding['config']),cfg))
    put(root/'health.json',{'at':NOW.isoformat(),'status':'HOLD','reason':'unknown secret'})
    value=pc.snapshot(binding,NOW)
    assert value['state']=='HOLD' and value['status_reason']=='FAILURE_DETAIL_WITHHELD'


def test_recovery_pointer_uses_hash_pinned_nested_preparation(daily):
    binding,_,output,_,root=daily
    receipt=pc.read(output.parent/'native-prepared.json')
    nested=output.parent/'recoveries/recovery-01/native-recovery-test'
    receipt['output']=str(nested)
    plan_path=nested/'plan.json'
    receipt['plan']={'path':str(plan_path),'sha256':put(plan_path,{'commit':'a'*40})}
    receipt_path=nested.parent/'native-prepared.json'
    preparation={'path':str(receipt_path),'sha256':put(receipt_path,receipt)}
    put(root/'current-day.json',{'racing_date':'2026-10-03','output':str(nested),'preparation':preparation})
    value=pc.snapshot(binding,NOW)
    assert value['state']=='PREPARED_NOT_STARTED' and value['inventory_state']=='NOT_YET_DISCOVERED'
    put(root/'health.json',{'at':NOW.isoformat(),'status':'PAUSED','output':str(nested),
                          'preparation':preparation,'source_commit':'a'*40})
    assert pc.snapshot(binding,NOW+timedelta(minutes=40))['state']=='PAUSED'
    put(root/'health.json',{'at':NOW.isoformat(),'status':'ACTIVE_COLLECTION','output':str(nested),
                          'preparation':preparation,'source_commit':'b'*40})
    with pytest.raises(ValueError,match='runtime_identity'):pc.snapshot(binding,NOW)


def test_verified_failure_survives_snapshot_without_success_claim(daily,monkeypatch):
    binding,_,output,sha,root=daily
    admission=output.parent/'admission'/sha/'attempts/test/admission.json'
    put(admission,{'race':{'race_id':'Dubbo R1','jump_timestamp':NOW.isoformat()},'job_id':'job'})
    bundle=root.parent/'predictions/2026-10-03/bundles/bundle'
    manifest_sha=put(bundle/'bundle_manifest.json',{'fixture':True})
    put(admission.with_name('completion.json'),{'bundle_entry':{'directory':'bundle','manifest_sha256':manifest_sha}})
    native={'evidence_class':'AUTHORIZED_ENGINEERING','future_race_evidence':False,
            'records':{name:{'status':'FAILED','failure':'RESIDUAL_SCORER_FAILED','predictions':None} for name in pc.MODELS}}
    monkeypatch.setattr(sys.modules['src.predictor.future_comparison'],'verify_comparison',lambda *a,**k:native)
    value=pc.snapshot(binding,NOW)
    assert len(value['failed_forecasts'])==1 and value['forecast_errors']==[] and value['forecasts']==[]


def test_unavailable_evidence_still_reports_failed_installed_service(daily,monkeypatch):
    binding,*_=daily
    def run(command,**kwargs):
        if command[0]=='bwrap':raise subprocess.TimeoutExpired(command,40)
        return types.SimpleNamespace(stdout='ActiveState=failed\nSubState=failed\nMainPID=0\nExecMainStatus=78\n')
    monkeypatch.setattr(pc.subprocess,'run',run)
    value=pc.observe(binding)
    assert value['state']=='UNAVAILABLE' and value['forecasts']==[]
    assert value['service']['state']=='failed' and value['service']['exit_status']==78
