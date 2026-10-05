import copy
from datetime import timedelta
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import io
import zipfile
from types import SimpleNamespace

import pytest

from scripts.predict_race_now import _sealed_result,_selected_variant
from src.predictor.controlled_retained_inputs import canonical,digest
from src.predictor.retrospective_native_v2 import (
    QualificationFailure,WORKER,pair_from_replay)
import src.predictor.retrospective_native_v2 as retro
from tests.test_predict_market_form_residual import (
    ARTIFACT_DIR,RACE_ID,RUNNERS,SCORE_TIME,_score_paths,_write_fixture)


def fixture(tmp_path):
    paths=_write_fixture(tmp_path);artifact=_score_paths(paths)
    roster=[{'box_number':box,'display_name':name,'identity':identity,'source_native_runner_id':str(1000+box)}
        for box,name,identity,price in RUNNERS]
    model=SimpleNamespace(model_sha256=artifact['model_sha256'],manifest_sha256=artifact['manifest_sha256'],
        schema_sha256='a'*64,requested='market_form_residual_v1',resolved='market_form_residual_v1',alias=False,
        model_path=ARTIFACT_DIR/'model.json',manifest_path=ARTIFACT_DIR/'manifest.json')
    state={'model':model,'race':{'race_id':RACE_ID,'jump_timestamp':artifact['jump_timestamp']},
        'prediction_id':'p1','job_id':'j1','config_sha':'a'*64,'runner_set_sha256':'b'*64,'runners':roster}
    prediction=_selected_variant(artifact,'full_strength')
    result=_sealed_result(state=state,generated_at=SCORE_TIME+timedelta(seconds=1),blocker=None,prediction=prediction)
    prices={box:price for box,name,identity,price in RUNNERS}
    identity={'form_sha256':artifact['input_hashes']['form_csv_sha256'],
        'sidecar_sha256':artifact['input_hashes']['sidecar_sha256'],
        'capture_sha256':artifact['input_hashes']['capture_artifact_sha256'],
        'production_feature_rows_sha256':artifact['input_hashes']['feature_rows_sha256'],
        'captured_at':artifact['odds_append_timestamp']}
    inputs={**identity,'runners':[{**r,'win_odds':prices[r['box_number']]} for r in roster]}
    by_box={r['box']:r for r in artifact['predictions']}
    production={'completed_at':(SCORE_TIME+timedelta(seconds=2)).isoformat(),
        'predictions':[{'box_number':r['box_number'],'identity':r['identity'],'dog_name':r['display_name'],
            'probability':by_box[r['box_number']]['full_probability']} for r in roster]}
    member={'race_id':RACE_ID,'original_admitted_at':(SCORE_TIME-timedelta(seconds=1)).isoformat(),
        'original_published_complete_at':(SCORE_TIME+timedelta(seconds=3)).isoformat(),
        'jump_at':artifact['jump_timestamp'],'bundle_manifest':{'path':'/sealed/bundle_manifest.json','sha256':'f'*64},
        'retained_input_manifest_sha256':'e'*64}
    replay={key:artifact[key] for key in ('race_id','jump_timestamp','model_sha256','manifest_sha256',
        'effective_state_sha256','variants','predictions','input_hashes','feature_freeze_timestamp',
        'odds_capture_timestamp','odds_append_timestamp')}
    return dict(member=member,result=result,request={'runners':roster},production=production,
        inputs=inputs,replay=replay,provenance={'source_commit':'c'*40},derived_at=SCORE_TIME+timedelta(days=1)),paths


def test_actual_native_serialization_and_same_parent_pair(tmp_path):
    values,_=fixture(tmp_path)
    pair=pair_from_replay(**values)
    assert pair['original_score_timestamp'] is None
    assert pair['derived_at']==values['derived_at'].isoformat()
    assert pair['historical_validation_anchor']['at']==values['production']['completed_at']
    assert 'record_key' not in pair and 'record_checksum_sha256' not in pair
    assert pair['candidates'][0]['common_input_sha256']==pair['candidates'][1]['common_input_sha256']
    assert [c['strength'] for c in pair['candidates']]==[1.0,0.5]
    native_order=sorted(values['request']['runners'],key=lambda r:r['source_native_runner_id'])
    expected={r['box']:r['half_probability'] for r in values['replay']['predictions']}
    assert pair['candidates'][1]['probabilities']==[expected[r['box_number']] for r in native_order]


def test_distinct_fetch_and_append_retain_exact_native_timing(tmp_path):
    values,_=fixture(tmp_path)
    assert values['replay']['odds_capture_timestamp'] != values['inputs']['captured_at']
    assert values['replay']['odds_append_timestamp'] == values['inputs']['captured_at']
    pair=pair_from_replay(**values)
    timing=pair['capture_timing']
    assert timing['fetch_at']==values['replay']['odds_capture_timestamp']
    assert timing['append_at']==values['inputs']['captured_at']
    assert timing['freshness_basis']=='NATIVE_RECEIPT_APPEND_TIME'
    assert timing['fetch_lead_seconds']>timing['append_lead_seconds']


@pytest.mark.parametrize('change',['reversed','append_mismatch','append_after_anchor','stale_append'])
def test_distinct_timing_rejects_changed_order_binding_or_freshness(tmp_path,change):
    values,_=fixture(tmp_path)
    if change=='reversed':
        values['replay']['odds_capture_timestamp']=values['production']['completed_at']
    elif change=='append_mismatch':
        values['replay']['odds_append_timestamp']=values['replay']['odds_capture_timestamp']
    elif change=='append_after_anchor':
        late=(SCORE_TIME+timedelta(seconds=3)).isoformat()
        values['replay']['odds_append_timestamp']=values['inputs']['captured_at']=late
    else:
        stale=(retro.stamp(values['member']['jump_at'])-timedelta(seconds=601)).isoformat()
        values['replay']['odds_capture_timestamp']=stale
        values['replay']['odds_append_timestamp']=values['inputs']['captured_at']=stale
    with pytest.raises(QualificationFailure,match='HISTORICAL_TIMING_INVALID'):
        pair_from_replay(**values)


def test_native_freshness_window_stays_on_append_not_fetch(tmp_path):
    values,_=fixture(tmp_path)
    values['replay']['odds_capture_timestamp']=(retro.stamp(values['member']['jump_at'])
        -timedelta(seconds=601)).isoformat()
    pair=pair_from_replay(**values)
    assert pair['capture_timing']['fetch_lead_seconds']==601
    assert 120<=pair['capture_timing']['append_lead_seconds']<=600


def test_one_ulp_full_probability_mismatch_is_not_tolerated(tmp_path):
    import math
    values,_=fixture(tmp_path)
    row=values['replay']['predictions'][0]
    row['full_probability']=math.nextafter(row['full_probability'],1.0)
    with pytest.raises(QualificationFailure,match='FULL_ARM_EXACT_REPLAY_MISMATCH'):
        pair_from_replay(**values)


@pytest.mark.parametrize('change',['native_id','name','box','rank','comparison','price','source_hash','parent','half_strength'])
def test_identity_model_or_source_change_cannot_admit_pair(tmp_path,change):
    values,_=fixture(tmp_path)
    if change in {'native_id','name','box','rank'}:
        key={'native_id':'source_native_runner_id','name':'dog_name','box':'box_number','rank':'rank'}[change]
        values['result']['prediction']['predictions'][0][key]='changed' if change in {'native_id','name'} else 99
    elif change=='comparison':values['production']['predictions'][0]['probability']=0.9
    elif change=='price':values['inputs']['runners'][0]['win_odds']=99.0
    elif change=='source_hash':values['replay']['input_hashes']['form_csv_sha256']='0'*64
    elif change=='parent':values['replay']['model_sha256']='0'*64
    else:values['replay']['variants']['half_strength']=0.2
    with pytest.raises(QualificationFailure):pair_from_replay(**values)


@pytest.mark.parametrize('change',['after_seal','after_jump','before_capture','late_seal','naive','derived_before_seal'])
def test_historical_anchor_is_never_moved_to_make_replay_pass(tmp_path,change):
    values,_=fixture(tmp_path)
    if change=='after_seal':values['production']['completed_at']=(SCORE_TIME+timedelta(seconds=4)).isoformat()
    elif change=='after_jump':values['production']['completed_at']=(SCORE_TIME+timedelta(hours=1)).isoformat()
    elif change=='before_capture':values['production']['completed_at']=(SCORE_TIME-timedelta(minutes=10)).isoformat()
    elif change=='late_seal':values['member']['original_published_complete_at']=values['member']['jump_at']
    elif change=='naive':values['production']['completed_at']=SCORE_TIME.replace(tzinfo=None).isoformat()
    else:values['derived_at']=SCORE_TIME
    with pytest.raises(QualificationFailure):pair_from_replay(**values)


def worker_request(paths,values):
    distributions=sorted(({'name':d.metadata['Name'],'version':d.version,
        'record_sha256':hashlib.sha256((d.read_text('RECORD') or '').encode()).hexdigest()}
        for d in importlib.metadata.distributions()),key=lambda r:(r['name'],r['version'],r['record_sha256']))
    return {'source_root':str(Path(__file__).resolve().parents[1]),
        'runtime_identity':{'executable':sys.executable,'version':sys.version,'prefix':sys.prefix,'distributions':distributions},
        'environment_lock':{'python':platform.python_version(),'packages':{r['name']:r['version'] for r in distributions}},
        'race_id':RACE_ID,'historical_validation_anchor':values['production']['completed_at'],
        'replay_paths':{**{k:str(v) for k,v in paths.items()},'model':str(ARTIFACT_DIR/'model.json'),
            'manifest':str(ARTIFACT_DIR/'manifest.json')}}


def test_fresh_original_source_worker_replays_without_emitting_original_commitment(tmp_path):
    values,paths=fixture(tmp_path)
    request=worker_request(paths,values)
    p=subprocess.run([sys.executable,'-B',str(WORKER),'--worker'],input=canonical(request),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10)
    assert p.returncode==0,p.stderr.decode()
    replay=json.loads(p.stdout)
    assert replay['worker_status']=='REPLAYED_WITH_HISTORICAL_VALIDATION_ANCHOR'
    assert not {'score_timestamp','record_key','record_checksum_sha256'} & replay.keys()
    values['replay']=replay
    assert pair_from_replay(**values)['verified_original_full_rows_sha256']==values['result']['evidence']['prediction_output_sha256']


def test_worker_rejects_same_python_changed_distribution_metadata(tmp_path):
    values,paths=fixture(tmp_path)
    request=worker_request(paths,values);request['runtime_identity']['distributions'][0]['record_sha256']='0'*64
    p=subprocess.run([sys.executable,'-B',str(WORKER),'--worker'],input=canonical(request),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10)
    assert p.returncode==2
    assert json.loads(p.stdout)=={'worker_status':'REPLAY_REJECTED','exception_type':'ValueError',
        'failure_category':'worker_computational_runtime_changed'}


def test_cli_default_off(capsys):
    from scripts.derive_retrospective_native_v2_pairs import main
    assert main([])==0
    assert 'RETROSPECTIVE_NATIVE_V2_DISABLED' in capsys.readouterr().out


def write(path,value):
    path.write_bytes(canonical(value))
    return {'path':str(path),'sha256':digest(value)}


def execution(tmp_path,monkeypatch):
    from datetime import datetime,timezone
    membership=write(tmp_path/'membership.json',{'members':[{'race_id':f'Race{i}'} for i in range(82)]})
    monkeypatch.setattr(retro,'MEMBERSHIP_SHA256',membership['sha256'])
    scope=write(tmp_path/'scope.json',{'membership_sha256':membership['sha256'],'cohort_members':82})
    monkeypatch.setattr(retro,'SCOPE_SHA256',scope['sha256'])
    root=Path(retro.__file__).resolve().parents[2]
    required={'scripts/derive_retrospective_native_v2_pairs.py','src/predictor/retrospective_native_v2.py',
        'src/predictor/controlled_retained_inputs.py','src/predictor/on_demand.py','src/predictor/retained_inputs.py'}
    now=datetime.now(timezone.utc)
    authority={'schema_version':'retrospective_native_v2_execution_authority_v1',
        'issued_at':(now-timedelta(seconds=1)).isoformat(),'expires_at':(now+timedelta(seconds=60)).isoformat(),
        'scope':scope,'membership':membership,'serialization_contract':retro.SERIALIZATION,
        'source_commit':'a'*40,'implementation_files':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in required},
        'limits':{'members':82,'max_bytes':10*1024*1024,'max_reads':1000,'max_wall_seconds':30,'worker_seconds':1,
            'max_output_bytes':24*1024*1024},
        'package_bindings':write(tmp_path/'packages.json',{'packages':{}}),'output_path':str(tmp_path/'new-output')}
    monkeypatch.setattr(retro.subprocess,'check_output',lambda command,**kw: 'a'*40+'\n' if command[1]=='rev-parse' else b'')
    return authority


def test_complete_denominator_retains_mismatch_and_result_independent_members(tmp_path,monkeypatch):
    authority=execution(tmp_path,monkeypatch);visited=[]
    def derive(member,*args,**kw):
        visited.append(member['race_id'])
        if member['race_id']=='Race7':raise QualificationFailure('FULL_ARM_EXACT_REPLAY_MISMATCH')
        return {'race_id':member['race_id'],'test_fixture':True}
    monkeypatch.setattr(retro,'load_and_derive',derive)
    ref=write(tmp_path/'authority.json',authority)
    assert retro.run(Path(ref['path']),ref['sha256'])==0
    result=json.loads((tmp_path/'new-output/inventory.json').read_bytes())
    assert len(visited)==82 and result['denominator']==len(result['records'])==82
    assert result['categories']=={'EXACT_FULL_ARM_VERIFIED_PAIR_DERIVED':81,'FULL_ARM_EXACT_REPLAY_MISMATCH':1}
    assert result['records'][7]=={'race_id':'Race7','status':'EXCLUDED','reason':'FULL_ARM_EXACT_REPLAY_MISMATCH'}
    assert retro.run(Path(ref['path']),ref['sha256'])==2
    assert len(visited)==82  # Exclusive output/claim prevents a second derivation.


def test_shared_integrity_failure_preserves_claim_and_prior_pairs_no_success_index(tmp_path,monkeypatch):
    authority=execution(tmp_path,monkeypatch)
    def derive(member,*args,**kw):
        if member['race_id']=='Race2':raise ValueError('controlled_input_hash_changed')
        return {'race_id':member['race_id']}
    monkeypatch.setattr(retro,'load_and_derive',derive)
    ref=write(tmp_path/'authority.json',authority)
    assert retro.run(Path(ref['path']),ref['sha256'])==2
    out=tmp_path/'new-output'
    assert (out/'claim.json').exists() and len(list(out.glob('pair-*.private.json')))==2
    assert not (out/'inventory.json').exists()
    assert json.loads((out/'status.json').read_bytes())['status']=='FAILED_PRESERVED_DERIVATION_CLAIM'


def test_expired_authority_cannot_read_membership_or_claim(tmp_path,monkeypatch):
    from datetime import datetime,timezone
    authority=execution(tmp_path,monkeypatch)
    authority['expires_at']=(datetime.now(timezone.utc)-timedelta(minutes=1)).isoformat()
    Path(authority['membership']['path']).unlink()
    ref=write(tmp_path/'authority.json',authority)
    assert retro.run(Path(ref['path']),ref['sha256'])==2
    assert not (tmp_path/'new-output').exists()


def test_budget_exhaustion_does_not_publish_success(tmp_path,monkeypatch):
    authority=execution(tmp_path,monkeypatch);authority['limits']['max_bytes']=1
    ref=write(tmp_path/'authority.json',authority)
    assert retro.run(Path(ref['path']),ref['sha256'])==2
    assert not (tmp_path/'new-output').exists()


def test_read_stability_ignores_access_time_without_ignoring_content_changes(tmp_path,monkeypatch):
    import src.predictor.controlled_retained_inputs as retained
    path=tmp_path/'input';path.write_bytes(b'unchanged')
    actual=retained.os.fstat;calls=0
    def atime_advance(fd):
        nonlocal calls
        calls+=1;st=actual(fd)
        return SimpleNamespace(**{name:getattr(st,name) for name in
            ('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns')},st_atime=st.st_atime+calls)
    monkeypatch.setattr(retained.os,'fstat',atime_advance)
    assert retained.BoundedReader().read(path)==b'unchanged'
    def changed(fd):
        st=atime_advance(fd)
        if calls%2==0:st.st_mtime_ns+=1
        return st
    monkeypatch.setattr(retained.os,'fstat',changed)
    with pytest.raises(ValueError,match='changed_during_read'):retained.BoundedReader().read(path)


def test_loader_metadata_to_original_worker_to_exact_native_rows(tmp_path,monkeypatch):
    """Exercise new integration; existing native verifier/package hashing are reused seams."""
    from tests.test_predict_market_form_residual import _seal_packet
    from src.predictor.controlled_retained_inputs import BoundedReader
    import src.predictor.on_demand as native
    bundle=tmp_path/'bundle';values,paths=fixture(bundle/'source')
    rows=json.loads(paths['feature_rows'].read_bytes());feature_manifest=json.loads(paths['feature_manifest'].read_bytes())
    for key,name in [('feature_rows','shadow_feature_rows.json'),('feature_manifest','shadow_manifest.json'),
            ('implementation_manifest','implementation_file_manifest.json')]:paths[key]=bundle/'features/sealed'/name
    feature_manifest['feature_rows']=str(paths['feature_rows']);_seal_packet(paths,rows,feature_manifest)
    capture=bundle/'source/capture.json';capture.write_bytes(paths['capture'].read_bytes());paths['capture']=capture
    artifact=_score_paths(paths)
    for key in values['replay']:values['replay'][key]=artifact[key]
    values['inputs']['production_feature_rows_sha256']=artifact['input_hashes']['feature_rows_sha256']
    paths.update({'model':bundle/'model/model.json','manifest':bundle/'model/manifest.json'})
    paths['model'].parent.mkdir(parents=True)
    for key in ('model','manifest'):paths[key].write_bytes((ARTIFACT_DIR/f'{key}.json').read_bytes())
    member=values['member'];member.update(job_id='j1',prediction_id='p1',runner_set_sha256='b'*64)
    plan=write(tmp_path/'original-plan.json',{'schema_version':'fabricated_plan'})
    member['original_plan']=plan
    admission={'race':values['result']['race'],'job_id':'j1','prediction_id':'p1','runner_set_sha256':'b'*64,
        'plan_sha256':plan['sha256'],'admitted_at':member['original_admitted_at'],'retained_input_manifest_sha256':None}
    # The worker runs the current fixture source as the authenticated original source.
    worker=worker_request(paths,values);package=tmp_path/'package';package.mkdir()
    generator={'path':str(package/'operational-generator.zip'),'sha256':'d'*64}
    lock=worker['environment_lock'];lockraw=canonical(lock)
    retained={'files':{'generator_source_archive':{'original_path':generator['path'],'sha256':generator['sha256']},
        'environment_lock':{'path':'environment.json','sha256':hashlib.sha256(lockraw).hexdigest()}},
        'history':{'source_sha256':'c'*64}}
    member['retained_input_manifest_sha256']=admission['retained_input_manifest_sha256']=digest(retained)
    member['admission']=write(tmp_path/'admission.json',admission)
    production=values['production'];production.update({k:v for k,v in admission.items() if k!='admitted_at'})
    production.update(candidate='production',status='SEALED',failure=None,model_sha256=artifact['model_sha256'],
        admission_sha256=member['admission']['sha256'])
    receipt={};values['inputs']['odds_receipt_sha256']=digest(receipt)
    production['input_identity']={k:v for k,v in values['inputs'].items() if k!='runners'}
    (bundle/'comparison').mkdir()
    for name,value in [('production.json',production),('inputs.json',values['inputs'])]:write(bundle/'comparison'/name,value)
    (bundle/'comparison/plan.json').write_bytes(Path(plan['path']).read_bytes());write(bundle/'odds_receipt.json',receipt)
    (bundle/'features/sealed_history.db').write_bytes(b'opaque-fabricated-history-not-opened-as-database')
    history={'target_race_id':RACE_ID,'cutoff_timestamp':member['jump_at'],'target_rows_materialized':0,
        'at_or_after_cutoff_rows_materialized':0,'source_sha256':'c'*64,
        'sealed_sha256':hashlib.sha256((bundle/'features/sealed_history.db').read_bytes()).hexdigest()}
    write(bundle/'features/history_seal.json',history)
    with zipfile.ZipFile(bundle/'retained_inputs.zip','w') as z:
        z.writestr('bundle/manifest.json',canonical(retained));z.writestr('bundle/environment.json',lockraw)
    files={str(path.relative_to(bundle)):{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in bundle.rglob('*') if path.is_file() and 'shadow_score_live' not in path.parts
            and path.name!='autonomous_live_odds_capture_report.json'}
    manifest={'files':files};member['bundle_manifest']=write(bundle/'bundle_manifest.json',manifest)
    entry={'directory':bundle.name,'manifest_sha256':member['bundle_manifest']['sha256']}
    completion={**admission,'admission_sha256':member['admission']['sha256'],'status':'COMPLETE_BEFORE_CUTOFF',
        'published_complete_at':member['original_published_complete_at'],'bundle_entry':entry}
    member['completion']=write(tmp_path/'completion.json',completion)
    called=[]
    def verify(root,supplied):
        called.append((root,supplied));return SimpleNamespace(manifest=manifest,result=values['result'],request=values['request'])
    monkeypatch.setattr(native,'verify_indexed_prediction_bundle',verify)
    root=Path(__file__).resolve().parents[1]
    sourcefiles={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
        for prefix in ('src','scripts','utils','config') for path in (root/prefix).rglob('*.py')}
    src={'plan':{'python':sys.executable,'source_root':str(root),'commit':'c'*40},
        'runtime':worker['runtime_identity'],'identity':{'files':sourcefiles},'entry':{'generator_archive':generator,
            'source_identity':{'path':'/fixture/source-identity','sha256':'a'*64},
            'runtime_identity':{'path':'/fixture/runtime','sha256':'b'*64},
            'source_archive':{'path':'/fixture/source.tar','sha256':'c'*64}}}
    monkeypatch.setattr(retro,'verify_package',lambda *args:src)
    pair=retro.load_and_derive(member,{str(package):{}},{},BoundedReader(),
        derived_at=values['derived_at'],worker_seconds=10)
    assert called==[(bundle.parent,entry)]
    assert pair['verified_original_full_rows_sha256']==values['result']['evidence']['prediction_output_sha256']


@pytest.mark.parametrize('category', ['ORIGINAL_SCORER_CHANGED', 'ORIGINAL_PACKAGE_IDENTITY_CHANGED',
    'WORKER_IMPORTED_SOURCE_CHANGED', 'UNCLASSIFIED_FAILURE'])
def test_shared_source_failure_cannot_publish_complete(tmp_path, monkeypatch, category):
    authority = execution(tmp_path, monkeypatch)
    def changed(*args, **kwargs):
        raise QualificationFailure(category)
    monkeypatch.setattr(retro, "load_and_derive", changed)
    ref = write(tmp_path/"authority.json", authority)
    assert retro.run(Path(ref["path"]), ref["sha256"]) == 2
    assert not (tmp_path/"new-output/inventory.json").exists()


def test_final_local_exclusion_after_expiry_cannot_publish_complete(tmp_path, monkeypatch):
    authority = execution(tmp_path, monkeypatch)
    expired = retro.datetime.fromisoformat(authority["expires_at"]) + timedelta(seconds=1)
    class Expired(retro.datetime):
        @classmethod
        def now(cls, tz=None):
            return expired
    def derive(member, *args, **kwargs):
        if member["race_id"] == "Race81":
            monkeypatch.setattr(retro, "datetime", Expired)
            raise QualificationFailure("FULL_ARM_EXACT_REPLAY_MISMATCH")
        return {"race_id": member["race_id"]}
    monkeypatch.setattr(retro, "load_and_derive", derive)
    ref = write(tmp_path/"authority.json", authority)
    assert retro.run(Path(ref["path"]), ref["sha256"]) == 2
    assert not (tmp_path/"new-output/inventory.json").exists()


def test_expiry_during_inventory_write_cannot_publish_success_status(tmp_path, monkeypatch):
    authority = execution(tmp_path, monkeypatch)
    expired = retro.datetime.fromisoformat(authority['expires_at']) + timedelta(seconds=1)
    class Expired(retro.datetime):
        @classmethod
        def now(cls, tz=None):
            return expired
    monkeypatch.setattr(retro, 'load_and_derive', lambda member, *args, **kw: {'race_id':member['race_id']})
    original_put = retro.put
    def put_then_expire(path, value):
        original_put(path, value)
        if path.name == 'inventory.json':
            monkeypatch.setattr(retro, 'datetime', Expired)
    monkeypatch.setattr(retro, 'put', put_then_expire)
    ref = write(tmp_path/'authority.json', authority)
    assert retro.run(Path(ref['path']), ref['sha256']) == 2
    output = tmp_path/'new-output'
    assert json.loads((output/'inventory.json').read_bytes())['status'] == 'DISPOSITIONS_RECORDED'
    assert json.loads((output/'status.json').read_bytes())['status'] == 'FAILED_PRESERVED_DERIVATION_CLAIM'
