"""Prospective scheduler gate consumes immutable metadata, never result bodies."""
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import pytest
from race_collection.live_freshness_contract import digest,encoded
from scripts import run_comparison_schedule as schedule


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(encoded(value))
    return {'path':str(path),'sha256':digest(value)}


def test_expired_old_incident_does_not_block_authorized_retained_dimensions(tmp_path,monkeypatch):
    from race_collection.retained_study_readiness import verify_retained_readiness
    cfg,at=fixture(tmp_path,monkeypatch)
    before=(Path(cfg['state_root'])/'slots/001/terminal.json').read_bytes()
    evidence=verify_retained_readiness(cfg,at)
    assert evidence['verified_forecasts']==4 and evidence['historical_closed_results']==3
    assert evidence['same_cohort_end_to_end_closed'] is False and evidence['study_enrolment'] is False
    assert schedule.verify_canary(cfg,None,Path(cfg['state_root']),at)
    assert (Path(cfg['state_root'])/'slots/001/terminal.json').read_bytes()==before


def fixture(tmp_path,monkeypatch):
    from race_collection import retained_study_readiness as module
    at=datetime(2026,10,4,7,tzinfo=timezone.utc)
    root=tmp_path/'state';slot='2026-10-01T13:00:00+10:00'
    old={'state_root':str(root),'programme_id':'invented','campaign_root':str(tmp_path/'campaign'),
         'source_commit':'a'*40,'comparison_plan':'/synthetic/plan','comparison_plan_sha256':'p'*64,
         'slots':[slot,'2026-10-05T13:00:00+11:00'],'source_operations_per_session':192,'max_source_operations':384}
    oldref=put(tmp_path/'old.json',old)
    terminal=put(root/'slots/001/terminal.json',{'status':'FAILED_RESTORED','at':'2026-10-01T04:00:00+00:00'})
    admission=put(root/'slots/001/admission.json',{'slot':slot,'claimed_at':'2026-10-01T02:50:00+00:00'})
    modelnames=('production','market','residual_box','residual_half')
    proof=put(tmp_path/'forecast-audit.json',{'status':'VERIFIED_PREJUMP'})
    refs=[put(tmp_path/name,{'fixture':True}) for name in ('admission.json','completion.json','manifest.json')]
    metadata=put(tmp_path/'metadata.json',{'status':'FULL_90_MINUTES_OBSERVED_REQUIRES_ROOT_FINAL_RECONCILIATION',
        'source_commit':'b'*40,'configuration_sha256':'c'*64,'seconds':5400,'window_start':'2026-10-04T02:00:00+00:00',
        'window_end':'2026-10-04T03:30:00+00:00','verified_races':1,'verified_forecasts':4,'halt':None,
        'native_publisher_overlap_pairs':[],'observer_record_coverage':{'spans_window':True},
        'forecast_audits':[dict(proof,status='VERIFIED_PREJUMP',native_engineering_evidence=True,completion_status='COMPLETE_BEFORE_CUTOFF',
            candidate_statuses={k:'SEALED' for k in modelnames},metadata=refs,job_id='invented-job',race_id='invented-race',
            checked_at='2026-10-04T02:10:00+00:00',jump_at='2026-10-04T02:15:00+00:00',published_complete_at='2026-10-04T02:09:59+00:00')]})
    freeze=put(tmp_path/'freeze.json',{'status':'CONTROL_FREEZE_AND_SERIALIZATION_CLEAR_WITH_DECLARED_AUXILIARY_GAPS',
        'all_sample_violations':[],'pin_matches':{'source':True},'native_pointer_matches':True})
    integrity=put(tmp_path/'integrity.json',{'status':'PASSED_90_MINUTE_COLLECTION_TO_PREDICTION_WITH_EXCLUSIONS',
        'source_commit':'b'*40,'configuration_sha256':'c'*64,'seconds':5400,'window_start':'2026-10-04T02:00:00+00:00',
        'window_end':'2026-10-04T03:30:00+00:00','verified_races':1,'verified_forecasts':4,'models_changed':False,
        'private_closure_complete':False,'metadata_audit':metadata,'independent_freeze_audit':freeze})
    native=tmp_path/'native-closure.json';native.write_bytes(b'opaque sealed bytes, not JSON')
    import hashlib
    opaque={'path':str(native),'sha256':hashlib.sha256(native.read_bytes()).hexdigest()}
    authority=put(tmp_path/'closed-authority.json',{'fixture':'opaque ref only'})
    cohort=put(tmp_path/'cohort.json',{'fixture':'opaque ref only'});binding=put(tmp_path/'bound.json',{'fixture':'opaque ref only'})
    accounting=put(tmp_path/'accounting.json',{'schema_version':'independent_sealed_private_closure_accounting_v1','status':'PASS_METADATA_ONLY',
        'states':{'CLOSED':3,'QUARANTINED':1},'pending':0,'cohort_races':4,'frozen_forecasts':16,'result_request_consumed':4,
        'result_request_allowance':8,'outside_cohort_charged_races':0,'quarantines_terminal_no_automatic_retry':True,'raw_outcomes_released':False,
        'evidence':{'closure':opaque,'authority':authority,'cohort':cohort,'binding':binding},'sealed_at':'2026-10-03T14:55:00+00:00'})
    completion=put(tmp_path/'closure-completion.json',{'status':'PRIVATE_CLOSURE_SEALED_COLLECTION_RESUMED','outcomes_released':False,
        'performance_evaluation':False,'pending':0,'result_projection':{'closure':{'status':'RESULT_CLOSURE_SEALED_NOT_EVALUATED',
        'sealed_at':'2026-10-03T14:55:00+00:00','target_values_decoded':False,'result_authority_sha256':authority['sha256']},
        'queue_health':{'status':'CLOSURE_SEALED','counts':{'CLOSED':3,'QUARANTINED':1},'outcomes_released':False},
        'final_cohort':{'eligible_races':4,'max_requests':8}}})
    authorization=put(tmp_path/'user.json',{'schema_version':'study_recovery_user_authorization_v1',
        'status':'AUTHORIZED_IMPLEMENTATION_AND_COORDINATED_ACTIVATION','predecessor_schedule':oldref,
        'current_release_live_acceptance':integrity,'recorded_at':(at-timedelta(hours=1)).isoformat(),
        'protected_outcomes_decoded':False})
    compatibility=put(tmp_path/'compatibility.json',{'status':'REVIEWED_READINESS_COMPATIBLE_SUCCESSOR','producing_source_commit':'b'*40,
        'target_source_commit':'d'*40,'changed_paths':['scripts/run_comparison_schedule.py'],
        'models_changed':False,'feature_generators_changed':False,'frozen_membership_changed':False})
    cfg={**old,'source_commit':'d'*40,'source_operations_per_session':256,'max_source_operations':448}
    amendment={'schema_version':'prospective_study_amendment_v1','status':'AUTHORIZED_PROSPECTIVE_STUDY_AMENDMENT',
        'operation_mode':'OUTCOME_BLIND_RETAINED_EVIDENCE_OBSERVER','authority_reference':'SYNTHETIC','user_authorization':authorization,'predecessor_config':oldref,
        'target_config_sha256':digest(cfg),'campaign_root':old['campaign_root'],'issued_at':at.isoformat(),
        'effective_at':(at+timedelta(hours=1)).isoformat(),'historical_slots':[{'index':1,'slot':slot,'files':[admission,terminal]}],
        'readiness':{'integrity':integrity,'closure_accounting':accounting,'closure_completion':completion,'compatibility':compatibility}}
    cfg['study_amendment']=put(tmp_path/'amendment.json',amendment)
    monkeypatch.setattr(module,'source_delta',lambda *args:['scripts/run_comparison_schedule.py'])
    return cfg,at+timedelta(hours=2)


@pytest.mark.parametrize('changed',['integrity','forecast_bytes','closure_bytes','compatibility','history','config','user'])
def test_changed_bound_evidence_never_creates_canary(tmp_path,monkeypatch,changed):
    from race_collection.retained_study_readiness import checked
    cfg,at=fixture(tmp_path,monkeypatch);amendment=checked(cfg['study_amendment'])
    if changed=='config':cfg['prediction_root']='/different'
    elif changed=='history':(Path(cfg['state_root'])/'slots/001/terminal.json').write_text('{}')
    elif changed=='forecast_bytes':(tmp_path/'manifest.json').write_text('{}')
    elif changed=='closure_bytes':(tmp_path/'native-closure.json').write_bytes(b'changed sealed bytes')
    elif changed=='user':(tmp_path/'user.json').write_text('{}')
    else:
        path=Path(amendment['readiness'][changed]['path']);path.write_text('{}')
    with pytest.raises(ValueError):schedule.verify_canary(cfg,None,Path(cfg['state_root']),at)
    assert not (Path(cfg['state_root'])/'canary-amendments').exists()


def test_future_effective_date_does_not_create_gate_early(tmp_path,monkeypatch):
    from race_collection.retained_study_readiness import checked
    cfg,at=fixture(tmp_path,monkeypatch);row=checked(cfg['study_amendment'])
    assert not schedule.verify_canary(cfg,None,Path(cfg['state_root']),datetime.fromisoformat(row['issued_at']))
    assert not (Path(cfg['state_root'])/'canary-amendments').exists()


def test_config_identity_migration_preserves_original_and_absent_historical_slot(tmp_path,monkeypatch):
    from race_collection.retained_study_readiness import checked
    cfg,at=fixture(tmp_path,monkeypatch);root=Path(cfg['state_root'])
    amendment=checked(cfg['study_amendment']);old=checked(amendment['predecessor_config'])
    # Existing failed slot plus an absent blocked slot must stay exactly that way.
    old['slots'].insert(1,'2026-10-02T13:00:00+10:00');put(Path(amendment['predecessor_config']['path']),old)
    amendment['predecessor_config']['sha256']=digest(old)
    user=checked(amendment['user_authorization']);user['predecessor_schedule']=amendment['predecessor_config']
    amendment['user_authorization']=put(Path(amendment['user_authorization']['path']),user)
    cfg['slots']=old['slots'];amendment['historical_slots'].append({'index':2,'slot':old['slots'][1],'files':[]})
    amendment['target_config_sha256']=digest({k:v for k,v in cfg.items() if k!='study_amendment'})
    cfg['study_amendment']=put(Path(cfg['study_amendment']['path']),amendment)
    identity=put(root/'config-identity.json',{'sha256':digest(old)});before=Path(identity['path']).read_bytes()
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):return at
    monkeypatch.setattr(schedule,'datetime',Clock)
    monkeypatch.setattr(schedule,'load_config',lambda *a:(cfg,{'ends_at':(at+timedelta(days=100)).isoformat()}))
    monkeypatch.setattr(schedule,'renew_source',lambda *a,**k:pytest.fail('observer cannot own provider'))
    monkeypatch.setattr(schedule,'child',lambda *a,**k:pytest.fail('observer cannot launch collector'))
    assert schedule.tick(tmp_path/'cfg')['status']=='RETAINED_STUDY_OBSERVER_READY'
    assert Path(identity['path']).read_bytes()==before
    assert not (root/'slots/002').exists() and not (root/'canary.json').exists()
    assert len(list((root/'config-amendments').glob('*.json')))==1
    assert schedule.tick(tmp_path/'cfg')['status']=='RETAINED_STUDY_OBSERVER_READY'
    assert len(list((root/'config-amendments').glob('*.json')))==1


def test_changed_forecast_generator_cannot_transfer_readiness(tmp_path,monkeypatch):
    from race_collection import retained_study_readiness as module
    cfg,at=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(module,'source_delta',lambda *a:['utils/prejump_weather.py'])
    with pytest.raises(ValueError,match='source_incompatible'):module.verify_retained_readiness(cfg,at)


@pytest.mark.parametrize('condition',['missing_historical_slot','backdated_issue','old_source_mode','future_effective_slot_rewritten'])
def test_invalid_amendment_cannot_write_config_identity(tmp_path,monkeypatch,condition):
    from race_collection.retained_study_readiness import checked,bind_configuration
    cfg,at=fixture(tmp_path,monkeypatch);row=checked(cfg['study_amendment']);root=Path(cfg['state_root'])
    old=checked(row['predecessor_config']);put(root/'config-identity.json',{'sha256':digest(old)})
    before=(root/'config-identity.json').read_bytes()
    if condition=='missing_historical_slot':row['historical_slots']=[]
    if condition=='backdated_issue':row['issued_at']='2026-10-01T00:00:00+00:00'
    if condition=='old_source_mode':row['operation_mode']='RUN_SECOND_SCIENTIFIC_COLLECTOR'
    if condition=='future_effective_slot_rewritten':cfg['slots'][0]='2026-10-02T13:00:00+10:00'
    cfg['study_amendment']=put(Path(cfg['study_amendment']['path']),row)
    with pytest.raises(ValueError):bind_configuration(cfg,root,at)
    assert (root/'config-identity.json').read_bytes()==before
    assert not (root/'config-amendments').exists()
