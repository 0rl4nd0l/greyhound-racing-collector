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


def test_shared_output_directory_rejected_before_values(synthetic,tmp_path):
    from race_collection.development_examples import seal,DevelopmentRejected
    rid,path,pin=changed_access(synthetic,tmp_path,lambda *args:None)
    output=tmp_path/'output';output.mkdir(mode=0o755)
    with pytest.raises(DevelopmentRejected,match='OUTPUT_NOT_PRIVATE'):seal(path,pin,rid,output)
    assert list(output.iterdir())==[]


def test_population_selection_precedes_WIN_qualification_and_keeps_first_six():
    from race_collection.development_examples import selected_population
    rows=[{'race_id':f'Race {i} - MURR - 2026-10-03','race_key':f'2026-10-03|MURR|{i}',
           'jump_at':f'2026-10-03T13:{i+10:02d}:00+10:00','WIN_qualified':i>1} for i in range(1,9)]
    intended,selected=selected_population(list(reversed(rows)),'2026-10-03')
    assert len(intended)==8
    assert selected==[f'Race {i} - MURR - 2026-10-03' for i in range(1,7)]


def test_synthetic_grant_rejects_real_identity_before_verifier(synthetic,tmp_path,monkeypatch):
    from race_collection.development_examples import seal,DevelopmentRejected,canonical
    from tests.fixtures.development_pipeline import ref
    def change(access,control):
        member=next(iter(access['members'].values()));member['bundle_root']=str(control/'untrusted')
        root=Path(member['bundle_root'])/member['entry']['directory'];(root/'protocol').mkdir(parents=True)
        request=root/'request.json';request.write_bytes(canonical({'race_id':next(iter(access['members'])),
            'runners':[{'display_name':'Real Runner'}]}))
        receipt=root/'protocol/collector_exact_receipt.json';receipt.write_bytes(canonical({'sealed_handoff':{'race':{'url':'real'}}}))
        manifest=root/'bundle_manifest.json';manifest.write_bytes(canonical({'files':{
            'request.json':ref(request),'protocol/collector_exact_receipt.json':ref(receipt)}}))
        member['entry']['manifest_sha256']=ref(manifest)['sha256']
    rid,path,pin=changed_access(synthetic,tmp_path,change)
    monkeypatch.setattr('src.predictor.on_demand.verify_indexed_prediction_bundle',lambda *a,**k:pytest.fail('no real payload reads'))
    with pytest.raises(DevelopmentRejected,match='SYNTHETIC_DATA_REQUIRED'):seal(path,pin,rid,tmp_path/'output')


@pytest.mark.parametrize('field',['probability','availability','index_observation'])
def test_example_replay_rejects_rehashed_packet_source_change(synthetic,tmp_path,field):
    from race_collection.development_examples import seal,verify_package,DevelopmentRejected,canonical,digest
    rid,path,pin=changed_access(synthetic,tmp_path,lambda *args:None)
    output=tmp_path/'output';seal(path,pin,rid,output)
    packet=json.loads((output/'pre_result.json').read_bytes())
    if field=='probability':packet['runners'][0]['model_win_probability']=.9
    elif field=='availability':packet['times']['availability_upper_bound']='1900-01-01T00:00:00+00:00'
    else:packet['times']['index_observed_at']='1900-01-01T00:00:00+00:00'
    (output/'pre_result.json').write_bytes(canonical(packet))
    complete=json.loads((output/'completion.json').read_bytes());complete['pre_result_sha256']=digest(canonical(packet))
    (output/'completion.json').write_bytes(canonical(complete))
    with pytest.raises(DevelopmentRejected,match='EXAMPLE_SOURCE_REPLAY_MISMATCH'):verify_package(path,pin,rid,output)


@pytest.mark.parametrize('late',[False,True])
def test_population_freeze_uses_actual_index_shape_and_preserves_cutoff_failure(tmp_path,monkeypatch,late):
    from types import SimpleNamespace
    from race_collection import development_examples as dev
    from tests.fixtures.development_pipeline import ref
    start=datetime.fromisoformat('2026-10-03T02:50:30+00:00')
    times=iter([start,start+timedelta(seconds=31) if late else start,start])
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls,tz=None):return next(times)
    monkeypatch.setattr(dev,'datetime',FrozenDateTime)
    source=tmp_path/'review.json';source.write_bytes(dev.canonical({'identity_only':True}))
    monkeypatch.setattr(dev,'RESERVATION_PINS',{'fixture':ref(source)['sha256']})
    registry=tmp_path/'registry.json';registry.write_bytes(dev.canonical({'sources':[{**ref(source),'kind':'fixture'}]}))
    allocation=tmp_path/'allocation.json';allocation.write_bytes(dev.canonical({'status':'AUTHORIZED','dates':dev.PILOT_DATES,
        'allocation_id':'test','reservation_registry':ref(registry)}))
    view=SimpleNamespace(source_generated_at='2026-10-03T12:50:00+10:00',packet_sha256='a'*64,
        races=({'race_id':'Race 1 - MURR - 2026-10-03','jump_datetime':'2026-10-03T13:10:00+10:00',
                'runners':[],'source_native_race_id':'999','race_url':'https://example.invalid/fabricated'},))
    monkeypatch.setattr('race_collection.synchronous_manual_capture.bounded_current_race_index',lambda **kw:view)
    output=tmp_path/'population.json'
    if late:
        with pytest.raises(dev.DevelopmentRejected,match='POPULATION_FREEZE_CROSSED_CUTOFF'):
            dev.freeze_population(tmp_path/'index',tmp_path,allocation,ref(allocation)['sha256'],output)
        assert Path(str(output)+'.failure.json').exists()
    else:
        value=dev.freeze_population(tmp_path/'index',tmp_path,allocation,ref(allocation)['sha256'],output)
        assert value['intended'][0]['url']=='https://example.invalid/fabricated'
        assert Path(str(output)+'.completion.json').exists()


def test_default_off_does_not_open_untrusted_inputs(tmp_path):
    from race_collection.development_examples import admit, DevelopmentRejected
    access = tmp_path / 'access.json'
    access.write_text(json.dumps({'schema_version': 'development_access_v1', 'status': 'PROPOSED'}))
    with pytest.raises(DevelopmentRejected, match='DEVELOPMENT_DISABLED'):
        admit(access, hashlib.sha256(access.read_bytes()).hexdigest(), 'unopened')
