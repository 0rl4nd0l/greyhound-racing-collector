"""Fresh dated authority, real historical runtime validation, no outcome fixtures."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pytest
from tests.test_incident_comparison import case
from tests.fixtures.incident_engineering_case import put

@pytest.fixture
def renewal(tmp_path, monkeypatch):
    incident, incident_ref, plan, plan_ref = case(tmp_path)
    root = Path(incident['result_root'])/'001'; root.mkdir(parents=True)
    cfg = dict(schema_version='comparison_result_runtime_v1',state_root=str(root),
        prediction_bundles=str(Path(incident['prediction_root'])/'bundles'),
        job_store=str(Path(incident['prediction_root'])/'jobs.sqlite3'),
        campaign_root=str(tmp_path/'campaign'),lock_path=str(tmp_path/'lock'),
        source_state=str(tmp_path/'source'),storage_mount={'path':str(tmp_path),'uuid':'fixture'},
        max_races=24,max_requests=72,max_attempts_per_race=3,races_per_cycle=8,
        max_storage_bytes=2**32,expires_at=incident['result_deadline'],
        incident_authority=incident_ref,incident_slot='001')
    old= dict(status='AUTHORIZED_ENGINEERING_MACHINE_RESULT_RETENTION',plan_sha256=plan_ref['sha256'],
        owner='old-owner',authority_reference='SYNTHETIC_OLD',source_budget_reference='SYNTHETIC_OLD',
        human_outcome_access=False,issued_at=incident['issued_at'],runtime=cfg,
        incident_authority=incident_ref,incident_slot='001',result_database=str(root/'results.sqlite3'))
    old_ref=put(tmp_path/'old-authority.json',old)
    binding=put(tmp_path/'binding.json',dict(plan=plan_ref['path'],plan_sha256=plan_ref['sha256'],
        authority=old_ref['path'],authority_sha256=old_ref['sha256']))
    key=hashlib.sha256(b'invented-race').hexdigest()
    claim=Path(plan['programme_root'])/plan_ref['sha256']/'attempts'/key
    admission=put(claim/'admission.json',{'job_id':'invented-job','race':{'race_id':'invented-race'}})
    completion=put(claim/'completion.json',{'status':'COMPLETE_BEFORE_CUTOFF'})
    value=dict(schema_version='engineering_retained_result_renewal_v1',status='AUTHORIZED_RENEWED_RETAINED_PROCESSING',
        authority_reference='SYNTHETIC_NEW',owner='root',issued_at='2026-10-02T12:30:00+10:00',
        expires_at='2026-10-04T12:00:00+11:00',original_binding=binding,
        network_requests_allowed=False,max_additional_requests=0,human_outcome_access=False,
        performance_evaluation=False,study_enrolment=False,preserve_attempts=True,
        jobs=[dict(job_id='invented-job',race_id='invented-race',admission=admission,completion=completion)])
    path=tmp_path/'renewal.json'; put(path,value)
    import race_collection.persistent_storage as storage
    monkeypatch.setattr(storage,'check_mount',lambda *args:None)
    return path,value,binding,old_ref,cfg

def load(renewal):
    from src.predictor.engineering_result_renewal import load_renewal
    path,value,*_=renewal
    reference=put(path,value)
    return load_renewal(reference,'SYNTHETIC_NEW',now=datetime.fromisoformat('2026-10-02T13:00:00+10:00'))

def test_fresh_renewal_validates_expired_binding_without_changing_it(renewal):
    path,value,binding,old_ref,cfg=renewal
    before={ref['path']:Path(ref['path']).read_bytes() for ref in (binding,old_ref)}
    renewed,original,plan,old,runtime=load(renewal)
    assert renewed['issued_at']>old['issued_at']
    assert runtime['expires_at']==cfg['expires_at']
    assert plan['status']=='AUTHORIZED_ENGINEERING'
    assert all(Path(path).read_bytes()==data for path,data in before.items())

@pytest.mark.parametrize('field,value',[
    ('expires_at','2026-10-02T13:00:00+10:00'),('expires_at','2026-10-04T12:00:01+11:00'),
    ('issued_at','2026-10-01T23:59:59+10:00'),('issued_at','2026-10-02T14:00:00+10:00'),
    ('network_requests_allowed',True),('max_additional_requests',1),('human_outcome_access',True),
    ('performance_evaluation',True),('study_enrolment',True),('preserve_attempts',False),
    ('authority_reference','SYNTHETIC_OLD'),('jobs',[])])
def test_renewal_rejects_scope_or_time_expansion(renewal,field,value):
    renewal[1][field]=value
    with pytest.raises(ValueError):load(renewal)

@pytest.mark.parametrize('mutation',['different_job','different_race','bad_admission','duplicate','bad_binding'])
def test_membership_is_original_and_hash_bound(renewal,mutation):
    job=renewal[1]['jobs'][0]
    if mutation=='different_job':job['job_id']='unadmitted'
    if mutation=='different_race':job['race_id']='unadmitted'
    if mutation=='bad_admission':job['admission']['sha256']='0'*64
    if mutation=='duplicate':renewal[1]['jobs'].append(dict(job))
    if mutation=='bad_binding':renewal[1]['original_binding']['sha256']='0'*64
    with pytest.raises(ValueError):load(renewal)


def test_changed_authority_cannot_publish_completed_audit(renewal, tmp_path):
    from scripts.audit_renewed_engineering_results import publish_result
    path,value,*_=renewal
    reference=put(path,value)
    value['expires_at']='2026-10-04T12:00:00+11:00'
    value['owner']='changed-after-private-audit'
    put(path,value)
    output=tmp_path/'never-published.json'
    with pytest.raises(ValueError):
        publish_result(reference,'SYNTHETIC_NEW',output,{'counts':{'IDENTITY_VERIFIED_CLOSED':1}})
    assert not output.exists()
