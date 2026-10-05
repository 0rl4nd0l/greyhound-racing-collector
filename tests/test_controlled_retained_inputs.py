import copy
import json
import hashlib
import io
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
import zipfile

import pytest

from scripts.predict_market_form_residual import load_frozen_model
import src.predictor.controlled_retained_inputs as recovery
from src.predictor.controlled_retained_inputs import MISSING, reconstruct_committed_record
from tests.test_predict_market_form_residual import ARTIFACT_DIR, _score_paths, _write_fixture


def committed(tmp_path):
    paths = _write_fixture(tmp_path)
    artifact = _score_paths(paths)
    frozen = load_frozen_model(ARTIFACT_DIR / 'model.json', ARTIFACT_DIR / 'manifest.json')
    features = json.loads(paths['feature_rows'].read_bytes())
    return frozen, artifact, features


def test_exact_native_producer_record_can_be_reconstructed(tmp_path):
    frozen, artifact, features = committed(tmp_path)
    record = reconstruct_committed_record(frozen, artifact, features, execute=True)
    assert record['record_key'] == artifact['record_key']
    assert record['record_checksum_sha256'] == artifact['record_checksum_sha256']
    assert record['score_timestamp'] == artifact['score_timestamp']


@pytest.mark.parametrize('key', ['record_key', 'record_checksum_sha256', 'score_timestamp'])
def test_missing_original_commitment_never_gets_invented(tmp_path, key):
    frozen, artifact, features = committed(tmp_path)
    del artifact[key]
    with pytest.raises(ValueError, match=MISSING):
        reconstruct_committed_record(frozen, artifact, features, execute=True)


def test_default_off(tmp_path):
    with pytest.raises(ValueError, match='disabled'):
        reconstruct_committed_record(*committed(tmp_path))


@pytest.mark.parametrize('key', ['record_key', 'record_checksum_sha256'])
def test_wrong_original_commitment_rejected(tmp_path, key):
    frozen, artifact, features = committed(tmp_path)
    artifact[key] = '0' * 64
    with pytest.raises(ValueError, match='commitment_mismatch'):
        reconstruct_committed_record(frozen, artifact, features, execute=True)


def test_changed_feature_cannot_reconstruct_original(tmp_path):
    frozen, artifact, features = committed(tmp_path)
    features[0]['days_since_last_start'] += 1
    with pytest.raises(ValueError, match='commitment_mismatch'):
        reconstruct_committed_record(frozen, artifact, features, execute=True)


def test_changed_runner_identity_rejected(tmp_path):
    frozen, artifact, features = committed(tmp_path)
    features[0]['dog_name'] = 'Another Dog'
    with pytest.raises(ValueError, match='identity_changed'):
        reconstruct_committed_record(frozen, artifact, features, execute=True)


def test_duplicate_feature_rejected(tmp_path):
    frozen, artifact, features = committed(tmp_path)
    features.append(copy.deepcopy(features[0]))
    with pytest.raises(ValueError, match='field_changed'):
        reconstruct_committed_record(frozen, artifact, features, execute=True)


def test_original_score_time_is_not_replaceable_with_publication_time(tmp_path):
    frozen, artifact, features = committed(tmp_path)
    artifact['score_timestamp']='2026-07-16T18:53:00+10:00'
    with pytest.raises(ValueError,match='commitment_mismatch'):
        reconstruct_committed_record(frozen,artifact,features,execute=True)


def test_parent_hash_and_parity_cannot_change(tmp_path):
    frozen, artifact, features = committed(tmp_path)
    changed=copy.deepcopy(artifact);changed['model_sha256']='0'*64
    with pytest.raises(ValueError,match='parent_changed'):
        reconstruct_committed_record(frozen,changed,features,execute=True)
    artifact['scoring_parity']={}
    with pytest.raises(ValueError,match='parity_mismatch'):
        reconstruct_committed_record(frozen,artifact,features,execute=True)


def write(path, value):
    raw=recovery.canonical(value)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(raw)
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}


def inventory_fixture(tmp_path):
    members=[]
    for index in range(82):
        root=tmp_path/str(index)
        output=write(root/'result.json',{'prediction':{'predictions':[]}})
        fixture=write(root/'model/fixture.json',{'schema_version':recovery.SHADOW_RECORD_SCHEMA,
            'inputs':{},'score_timestamp':'fixed-fixture','record_key':'fixture','record_checksum_sha256':'fixture'})
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as archive:
            archive.writestr('bundle/manifest.json','{}')
            archive.writestr('bundle/inputs/model/fixture.json',Path(fixture['path']).read_bytes())
        (root/'retained_inputs.zip').write_bytes(stream.getvalue())
        files={name:{'sha256':hashlib.sha256((root/name).read_bytes()).hexdigest(),
                     'bytes':(root/name).stat().st_size} for name in
               ('result.json','model/fixture.json','retained_inputs.zip')}
        manifest=write(root/'bundle_manifest.json',{'files':files})
        members.append({'race_id':f'Race {index}','bundle_manifest':manifest})
    return write(tmp_path/'membership.json',{'members':members})


def test_whole_82_denominator_model_fixtures_are_not_originals(tmp_path):
    ref=inventory_fixture(tmp_path)
    result=recovery.audit_membership(ref['path'],expected_sha256=ref['sha256'])
    assert result['denominator']==82
    assert len(result['records'])==82
    assert result['failure_categories']=={MISSING:82}
    assert result['controlled_pairs_derived']==0
    assert all(not row['commitment_field_counts'] for row in result['records'])


def test_tampered_last_bundle_prevents_partial_success_inventory(tmp_path):
    ref=inventory_fixture(tmp_path)
    (tmp_path/'81/result.json').write_text('{}')
    with pytest.raises(ValueError,match='hash_changed'):
        recovery.audit_membership(ref['path'],expected_sha256=ref['sha256'])


@pytest.mark.parametrize('name',['../outside.json','/absolute.json'])
def test_member_path_escape_rejected(name):
    with pytest.raises(ValueError,match='path_invalid'):
        recovery.member_path(Path('/bundle'),name)


def test_declared_limits_are_enforced(tmp_path):
    path=tmp_path/'control.json';path.write_text('{}')
    with pytest.raises(ValueError,match='budget'):
        recovery.BoundedReader(max_bytes=1).read(path)
    with pytest.raises(ValueError,match='budget'):
        recovery.BoundedReader(max_files=0).read(path)


def execution_fixture(tmp_path,monkeypatch):
    from scripts.predict_market_form_residual import FEATURE_GENERATOR_FILES
    root=Path(recovery.__file__).resolve().parents[2]
    required=set(FEATURE_GENERATOR_FILES)|{
        'scripts/predict_market_form_residual.py','src/predictor/market_form_residual.py',
        'src/predictor/scoring_parity.py','config/venue_mapping.py','utils/csv_metadata.py',
        'utils/race_identity_equivalence.py'}
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in required}
    for name in required:
        dest=tmp_path/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((root/name).read_bytes())
    identity={'commit':'a'*40,'tree':'b'*40,'files':hashes}
    dist=SimpleNamespace(metadata={'Name':'fake'},version='1',read_text=lambda name:'record')
    monkeypatch.setattr(recovery.importlib.metadata,'distributions',lambda:[dist])
    runtime={'version':sys.version,'prefix':sys.prefix,'executable':sys.executable,
        'distributions':[{'name':'fake','version':'1','record_sha256':hashlib.sha256(b'record').hexdigest()}]}
    plan={'commit':identity['commit'],'tree':identity['tree'],'source_root':str(tmp_path/'source'),
        'source_identity_sha256':recovery.digest(identity),'runtime_sha256':recovery.digest(runtime),
        'python_sha256':hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest()}
    generator=tmp_path/'operational-generator.zip';generator.write_bytes(b'fabricated-source-archive')
    binding={'original_package_plan':write(tmp_path/'plan.json',plan),
        'original_source_identity':write(tmp_path/'source/SOURCE_IDENTITY.json',identity),
        'original_runtime_identity':write(tmp_path/'runtime-identity.json',runtime),
        'original_source_commit':identity['commit'],'replay_source_hashes':dict(hashes)}
    retained={'files':{'generator_source_archive':{'original_path':str(generator),
        'sha256':hashlib.sha256(generator.read_bytes()).hexdigest()}}}
    lock={'python':platform.python_version(),'packages':{'fake':'1'}}
    return binding,retained,lock


def test_original_source_map_and_actual_runtime_are_verified(tmp_path,monkeypatch):
    binding,retained,lock=execution_fixture(tmp_path,monkeypatch)
    plan=recovery.verify_execution_binding(binding,retained,lock,recovery.BoundedReader())
    assert plan['commit']=='a'*40


def test_two_mutable_trees_cannot_replace_original_source_map(tmp_path,monkeypatch):
    binding,retained,lock=execution_fixture(tmp_path,monkeypatch)
    name='src/predictor/market_form_residual.py'
    identity=json.loads(Path(binding['original_source_identity']['path']).read_bytes())
    identity['files'][name]='0'*64
    binding['original_source_identity']=write(Path(binding['original_source_identity']['path']),identity)
    plan=json.loads(Path(binding['original_package_plan']['path']).read_bytes())
    plan['source_identity_sha256']=recovery.digest(identity)
    binding['original_package_plan']=write(Path(binding['original_package_plan']['path']),plan)
    with pytest.raises(ValueError,match='source_map_mismatch'):
        recovery.verify_execution_binding(binding,retained,lock,recovery.BoundedReader())


@pytest.mark.parametrize('change',['version','record','lock'])
def test_same_interpreter_with_changed_distribution_rejected(tmp_path,monkeypatch,change):
    binding,retained,lock=execution_fixture(tmp_path,monkeypatch)
    if change=='lock':
        lock['packages']['fake']='2'
    else:
        dist=SimpleNamespace(metadata={'Name':'fake'},version='2' if change=='version' else '1',
            read_text=lambda name:'changed' if change=='record' else 'record')
        monkeypatch.setattr(recovery.importlib.metadata,'distributions',lambda:[dist])
    with pytest.raises(ValueError,match='runtime_changed'):
        recovery.verify_execution_binding(binding,retained,lock,recovery.BoundedReader())


def test_loader_default_off_does_not_read():
    with pytest.raises(ValueError,match='disabled'):
        recovery.load_committed_original('/absent',{},'absent',{})


def test_cli_default_off_does_not_read(capsys):
    from scripts.audit_controlled_retained_inputs import main
    assert main([])==0
    assert 'DISABLED' in capsys.readouterr().out
