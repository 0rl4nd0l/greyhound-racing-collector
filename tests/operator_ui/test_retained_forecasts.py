from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json

import pytest

from src.operator_ui import retained_forecasts as rf
from tests.operator_ui.test_security import configured_app, login

NOW = datetime(2026, 9, 29, 8, tzinfo=timezone.utc)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rf.canonical(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def config(tmp_path):
    operational, programme = tmp_path/'operational', tmp_path/'programme'
    operational.mkdir(); programme.mkdir()
    schedule = tmp_path/'schedule.json'
    sha = put(schedule, {'status':'AUTHORIZED_PERSISTENT_SCHEDULE', 'prediction_root':str(programme),
          'state_root':str(tmp_path/'state'), 'session_minutes':90, 'slots':['2026-10-01T13:00:00+10:00']})
    put(tmp_path/'state/health.json', {'at':NOW.isoformat(), 'status':'NO_SLOT_DUE'})
    return {'schema':rf.SCHEMA, 'roots':{'operational':str(operational),'programme':str(programme)},
            'model_sha256':'a'*64,'model_manifest_sha256':'b'*64,'config_sha256':'c'*64,
            'schedule':str(schedule),'schedule_sha256':sha,'independent_audits':[], 'operational_job_ids':['job_expected']}


def inactive(unit):
    return {'ActiveState':'active' if unit.endswith('.timer') else 'inactive',
            'MainPID':'0','UnitFileState':'enabled'}


def test_scheduled_idle_is_not_collecting(config):
    value=rf.programme_status(config,NOW,inactive)
    assert value['state']=='ARMED_SCHEDULED_IDLE' and value['collecting'] is False
    assert value['next_session_end']=='2026-10-01T14:30:00+10:00'


@pytest.mark.parametrize('status,expected', [('SCHEDULE_FAILED','SCHEDULE_FAILED'),('SESSION_RUNNING','SESSION_STARTING_OR_RESTORING')])
def test_programme_failures_and_starting(config,status,expected):
    put(Path(config['schedule']).parent/'state/health.json',{'at':NOW.isoformat(),'status':status})
    assert rf.programme_status(config,NOW,inactive)['state']==expected


def test_stale_status_is_not_active(config):
    assert rf.programme_status(config,NOW+timedelta(hours=1),inactive)['state']=='STATUS_STALE'


def test_changed_schedule_fails_closed(config):
    put(Path(config['schedule']),{})
    assert rf.programme_status(config,NOW,inactive)['state']=='UNAVAILABLE'


def test_empty_missing_and_malformed_inventory(config,monkeypatch):
    monkeypatch.setattr(rf,'unit_state',inactive)
    values=rf.forecasts(config,NOW)['sources']
    assert values[1]['state']=='EMPTY'
    root=Path(config['roots']['operational']); (root/'bundles/prediction_missing').mkdir(parents=True)
    assert rf.forecasts(config,NOW)['sources'][0]['state']=='UNAVAILABLE'
    put(root/'bundles/prediction_bundle_index_v1.json',{'invalid':True})
    assert rf.forecasts(config,NOW)['sources'][0]['state']=='UNAVAILABLE'


def test_read_rejects_tamper_and_symlink(tmp_path):
    original=tmp_path/'original.json'; sha=put(original,{'v':1})
    assert rf.read(original,sha)=={'v':1}
    put(original,{'v':2})
    with pytest.raises(ValueError):rf.read(original,sha)
    link=tmp_path/'link.json';link.symlink_to(original)
    with pytest.raises(ValueError):rf.read(link)


def test_api_requires_authentication_and_audit(config,tmp_path,monkeypatch):
    app=configured_app(tmp_path)
    rf.install_forecast_display(app,config)
    client=app.test_client()
    endpoint='/operator-ui/api/v1/predictions/retained'
    assert client.get(endpoint).status_code==401
    assert login(client).status_code==200
    value=client.get(endpoint)
    assert value.status_code==200 and value.json['schema']==rf.SCHEMA
    assert value.headers['Cache-Control'].find('no-store')>=0
    from src.operator_ui.security import AuditUnavailable
    def fail(*args,**kwargs):raise AuditUnavailable('fixture')
    monkeypatch.setattr(app.extensions['operator_ui_audit'],'append_and_confirm',fail)
    assert client.get(endpoint).status_code==503
    assert 'sources' not in client.get(endpoint).json


def test_login_csrf_and_cookie_security(tmp_path):
    app=configured_app(tmp_path);client=app.test_client()
    assert client.post('/operator-ui/login',data={'username':'viewer','password':'correct horse'}).status_code==400
    response=login(client)
    cookie=response.headers['Set-Cookie']
    assert 'Secure' in cookie and 'HttpOnly' in cookie and 'SameSite=Strict' in cookie


def test_forecast_config_packaged_with_identity(config,tmp_path,monkeypatch):
    from tests.operator_ui.test_deployment_generator import deployment_inputs, git_identity
    from src.operator_ui.deployment import generate_package
    args=deployment_inputs(tmp_path/'deployment')
    git_identity(monkeypatch)
    path=tmp_path/'display.json';sha=put(path,config)
    generate_package(**args,forecast_display=path)
    assert f'OPERATOR_UI_FORECAST_DISPLAY_SHA256={sha}' in (args['output_dir']/'operator-ui-r3.env').read_text()
    assert (args['source_root']/'var/operator_ui/generated/forecast-display.json').read_bytes()==path.read_bytes()


def test_missing_expected_forecast_does_not_become_empty(config):
    source=rf.forecasts(config,NOW)['sources'][0]
    assert source['state']=='UNAVAILABLE' and source['forecasts']==[]


def test_active_collection_requires_process_evidence(config):
    put(Path(config['schedule']).parent/'state/health.json',{'at':NOW.isoformat(),'status':'SESSION_RUNNING'})
    def active(unit):return {'ActiveState':'active','MainPID':'123' if unit.endswith('.service') else '0','UnitFileState':'enabled'}
    assert rf.programme_status(config,NOW,active)['state']=='ACTIVE_COLLECTION'


def test_private_outcomes_cannot_be_projected_as_ready_forecast(config,tmp_path,monkeypatch):
    from src.predictor.on_demand import VerifiedPredictionBundle
    # Even a structurally verified bundle must pass the production-model gate.
    bundle=VerifiedPredictionBundle('x',{}, {'status':'PREDICTION_READY','model':{'resolved':'private_challenger'}},{},{'retained_input_manifest_sha256':'a'*64})
    monkeypatch.setattr(rf,'verify_indexed_prediction_bundle',lambda *args:bundle)
    with pytest.raises(ValueError,match='production_model_identity_mismatch'):
        rf.project_bundle(tmp_path,{}, {},config,NOW,{})
