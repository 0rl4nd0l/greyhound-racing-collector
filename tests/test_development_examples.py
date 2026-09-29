"""Public assembly seam: admission precedes reads; sealed inputs precede labels."""
import hashlib
import json

import pytest

from datetime import datetime, timedelta
from pathlib import Path


@pytest.fixture(scope='module')
def synthetic(tmp_path_factory):
    from tests.test_operational_prediction_packaged import test_packaged_capture_retention_frozen_prediction
    from tests.fixtures.development_pipeline import prepare_access
    seed = tmp_path_factory.mktemp('development-exported')
    with pytest.MonkeyPatch.context() as monkeypatch:
        test_packaged_capture_retention_frozen_prediction(seed, monkeypatch, False, 'murray')
    control = seed / 'development-control'
    race_id, access, pin = prepare_access(seed, control)
    return seed, control, race_id, access, pin


def test_complete_assembly_uses_real_retention_and_prediction_seals(synthetic, tmp_path):
    from race_collection.development_examples import seal, join_result, verify_package
    from tests.fixtures.development_pipeline import prepare_result
    seed, control, race_id, access, pin = synthetic
    result = seal(access,pin,race_id,tmp_path)
    packet = json.loads((tmp_path/'pre_result.json').read_bytes())
    assert packet['label']=='SYNTHETIC'
    assert [r['normalized_market_probability'] for r in packet['runners']]==[.25]*4
    assert packet['accounting']==dict(intended=3, qualified=2, attempts=2, successful_forecasts=1)
    assert result['status']=='SEALED_PRE_RESULT'
    before = {str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert seal(access,pin,race_id,tmp_path)==result
    assert {str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}==before
    authority, authority_pin=prepare_result(control,tmp_path,race_id)
    joined=join_result(access,pin,race_id,tmp_path,authority,authority_pin)
    assert joined['synthetic'] and joined['trainable']
    assert join_result(access,pin,race_id,tmp_path,authority,authority_pin)==joined
    assert verify_package(access,pin,race_id,tmp_path)['status']=='REPLAY_VERIFIED'
    assert packet['history']['runners'][0]['card_quality']['quality']['retained_start_count']['observed_denominator']==1


def changed_access(synthetic,tmp_path,change):
    from tests.fixtures.development_pipeline import prepare_access, ref
    from race_collection.development_examples import canonical
    seed, _, _, _, _ = synthetic
    rid,path,pin=prepare_access(seed,tmp_path/'control')
    access=json.loads(path.read_bytes())
    change(access,path.parent)
    path.write_bytes(canonical(access))
    return rid,path,ref(path)['sha256']


@pytest.mark.parametrize('case,reason',[
    ('missing_member','RACE_NOT_ALLOCATED'),('real_empty_registry','RESERVATION_REVIEW_REQUIRED'),
    ('collision','RESERVATION_COLLISION'),('unknown_reservation','RESERVATION_SCHEMA_UNKNOWN'),
    ('accounting','OPPORTUNITY_ACCOUNTING_INCOMPLETE'),('budget','ATTEMPT_BUDGET_EXCEEDED')])
def test_admission_rejects_before_bundle_reads(synthetic,tmp_path,monkeypatch,case,reason):
    from race_collection.development_examples import admit,DevelopmentRejected,canonical
    from tests.fixtures.development_pipeline import ref
    def change(access,control):
        allocation=json.loads((control/'allocation.json').read_bytes())
        if case=='missing_member':access['members']={}
        elif case=='real_empty_registry':
            access['status']='AUTHORIZED_DEVELOPMENT';allocation['status']='AUTHORIZED'
        elif case in ('collision','unknown_reservation'):
            reservation={'status':'AUTHORIZED_EXCLUSIVE_ALLOCATION','starts_at':allocation['starts_at'],'ends_at':allocation['ends_at']}
            (control/'reservation.json').write_bytes(canonical(reservation))
            registry={'schema_version':'development_reservation_registry_v1','sources':[
                {**ref(control/'reservation.json'),'kind':'exclusive_comparison' if case=='collision' else 'unknown'}]}
            (control/'reservations.json').write_bytes(canonical(registry))
            allocation['reservation_registry']=ref(control/'reservations.json')
        elif case=='accounting':
            value=json.loads((control/'opportunities.json').read_bytes());value['complete']=False
            (control/'opportunities.json').write_bytes(canonical(value));access['opportunities']=ref(control/'opportunities.json')
        elif case=='budget':allocation['max_capture_attempts']=1
        (control/'allocation.json').write_bytes(canonical(allocation));access['allocation']=ref(control/'allocation.json')
    rid,path,pin=changed_access(synthetic,tmp_path,change)
    with pytest.raises(DevelopmentRejected,match=reason):admit(path,pin,rid)


@pytest.mark.parametrize('disposition',['MISSING','AMBIGUOUS','VOID'])
def test_durable_nontrainable_result_dispositions(synthetic,tmp_path,disposition):
    from race_collection.development_examples import seal,join_result
    from tests.fixtures.development_pipeline import prepare_result
    rid,path,pin=changed_access(synthetic,tmp_path,lambda *args:None)
    output=tmp_path/'output';seal(path,pin,rid,output)
    authority,authority_pin=prepare_result(path.parent,output,rid,disposition=disposition)
    result=join_result(path,pin,rid,output,authority,authority_pin)
    assert result['disposition']==disposition and result['trainable'] is False


def test_result_authority_checked_before_result_path_read(synthetic,tmp_path):
    from race_collection.development_examples import seal,join_result,DevelopmentRejected,put
    from tests.fixtures.development_pipeline import ref
    rid,path,pin=changed_access(synthetic,tmp_path,lambda *args:None)
    output=tmp_path/'output';seal(path,pin,rid,output)
    authority=path.parent/'denied.json'
    put(authority,{'schema_version':'development_result_authority_v1','status':'PROPOSED','members':{}})
    with pytest.raises(DevelopmentRejected,match='RESULT_ACCESS_NOT_AUTHORIZED'):
        join_result(path,pin,rid,output,authority,ref(authority)['sha256'])


def test_crossing_jump_is_preserved_and_cannot_restart_as_success(synthetic,tmp_path):
    from race_collection.development_examples import seal,DevelopmentRejected
    rid,path,pin=changed_access(synthetic,tmp_path,lambda *args:None)
    access=json.loads(path.read_bytes());jump=datetime.fromisoformat(access['members'][rid]['jump_at'])
    times=iter([jump-timedelta(seconds=2),jump-timedelta(seconds=1),jump])
    output=tmp_path/'output'
    with pytest.raises(DevelopmentRejected,match='COMPLETION_CROSSED_JUMP'):
        seal(path,pin,rid,output,clock=lambda:next(times))
    assert (output/'failure.json').exists()
    with pytest.raises(DevelopmentRejected,match='INCOMPLETE_ATTEMPT_PRESERVED'):
        seal(path,pin,rid,output)


@pytest.mark.parametrize('change,reason', [('native','OFFICIAL_RESULT_NATIVE_ID_MISMATCH'),
    ('url','OFFICIAL_RESULT_URL_MISMATCH'),('labels','OFFICIAL_RESULT_LABEL_MISMATCH')])
def test_official_source_identity_and_labels_cannot_be_rebound(synthetic,tmp_path,change,reason):
    from race_collection.development_examples import seal,join_result,canonical,digest
    from tests.fixtures.development_pipeline import prepare_result,ref
    rid,path,pin=changed_access(synthetic,tmp_path,lambda *args:None)
    output=tmp_path/'output';seal(path,pin,rid,output)
    authority,_=prepare_result(path.parent,output,rid)
    record=path.parent/'official-result.json';result=json.loads(record.read_bytes())
    if change=='native':result['official_evidence']['runner_rows'][0]['source_native_runner_id']='other'
    elif change=='url':result['official_evidence']['race_rows'][0]['source_url']='https://www.thedogs.com.au/racing/sale/2030-01-01/1/wrong'
    else:
        ids=list(result['finishers']);result['finishers'][ids[0]],result['finishers'][ids[1]]=2,1
    result['source_evidence_sha256']=digest(canonical(result['official_evidence']))
    record.write_bytes(canonical(result));value=json.loads(authority.read_bytes())
    value['members'][rid]['result']=ref(record);authority.write_bytes(canonical(value))
    with pytest.raises(ValueError,match=reason):join_result(path,pin,rid,output,authority,ref(authority)['sha256'])


def test_private_comparison_manifest_rejected_before_full_verifier(synthetic,tmp_path,monkeypatch):
    from race_collection.development_examples import seal,DevelopmentRejected,canonical
    from tests.fixtures.development_pipeline import ref
    def change(access,control):
        member=next(iter(access['members'].values()))
        member['bundle_root']=str(control/'private')
        manifest=Path(member['bundle_root'])/member['entry']['directory']/'bundle_manifest.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_bytes(canonical({'files':{'comparison/private.json':{}}}))
        member['entry']['manifest_sha256']=ref(manifest)['sha256']
    rid,path,pin=changed_access(synthetic,tmp_path,change)
    monkeypatch.setattr('src.predictor.on_demand.verify_indexed_prediction_bundle',
                        lambda *a,**k:pytest.fail('private payload verifier must not run'))
    with pytest.raises(DevelopmentRejected,match='FROZEN_COMPARISON_INPUT_FORBIDDEN'):
        seal(path,pin,rid,tmp_path/'output')


def test_default_off_does_not_open_untrusted_inputs(tmp_path):
    from race_collection.development_examples import admit, DevelopmentRejected
    access = tmp_path / 'access.json'
    access.write_text(json.dumps({'schema_version': 'development_access_v1', 'status': 'PROPOSED'}))
    with pytest.raises(DevelopmentRejected, match='DEVELOPMENT_DISABLED'):
        admit(access, hashlib.sha256(access.read_bytes()).hexdigest(), 'unopened')
