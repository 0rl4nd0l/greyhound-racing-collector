"""Strict retained original-record recovery; no new pair or outcome processing.

A native v2 packet's forecast rows cannot substitute for an original v3 record
commitment. Missing original checksums or timestamps are explicit failures.
"""
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys
import time
import zipfile

from src.predictor.market_form_residual import FEATURES, SHADOW_RECORD_SCHEMA, score_race
from src.predictor.scoring_parity import build_scoring_input, build_core_output, parity_binding
from src.predictor.controlled_adjustment_pair import PARENT_MODEL_SHA256, PARENT_MANIFEST_SHA256

MEMBERSHIP_SHA256 = '782d6af9bcbe1dd84d1c2674de44f93bc6720cd77b3b7e0195bdcccc03dd4342'
MISSING = 'ORIGINAL_NATIVE_RECORD_COMMITMENT_AND_SCORE_TIMESTAMP_MISSING'
FIELDS = ('record_key', 'record_checksum_sha256', 'packet_record_checksum_sha256', 'score_timestamp')


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def commitment_inventory(value):
    """Inspect keys and record schemas; never publish input or forecast values."""
    counts = Counter()
    def visit(item):
        if isinstance(item, dict):
            if item.get('schema_version') == SHADOW_RECORD_SCHEMA and 'inputs' in item:
                counts['original_record'] += 1
            counts.update(key for key in FIELDS if key in item)
            for child in item.values():
                if isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(value)
    return dict(counts)


def reconstruct_committed_record(frozen, artifact, feature_rows, *, execute=False):
    """Rebuild computational content only when the original committed it.

    This pure seam is not an input authentication grant. The loader must bind
    the producer artifact and all native input/source/runtime seals first.
    Original timestamps are copied from the original artifact, never inferred
    from publication time or substituted with today's clock.
    """
    if execute is not True:
        raise ValueError('controlled_reconstruction_disabled')
    if (not isinstance(artifact, dict) or artifact.get('record_schema_version') != SHADOW_RECORD_SCHEMA
            or any(not isinstance(artifact.get(key), str) or not re.fullmatch('[0-9a-f]{64}', artifact[key])
                   for key in ('record_key', 'record_checksum_sha256'))
            or not artifact.get('score_timestamp')):
        raise ValueError(MISSING)
    if (frozen.model_sha256 != PARENT_MODEL_SHA256 or frozen.manifest_sha256 != PARENT_MANIFEST_SHA256
            or any(artifact.get(key) != getattr(frozen, key) for key in
           ('model_sha256', 'manifest_sha256', 'effective_state_sha256'))):
        raise ValueError('controlled_reconstruction_parent_changed')
    ordered = artifact['canonical_runner_order']
    features = {row['box_number']: row for row in feature_rows}
    predictions = {row['box']: row for row in artifact['predictions']}
    if (len(features) != len(feature_rows) or len(predictions) != len(artifact['predictions'])
            or len({row['box'] for row in ordered}) != len(ordered)
            or set(features) != set(predictions) or set(features) != {row['box'] for row in ordered}):
        raise ValueError('controlled_reconstruction_field_changed')
    runners = []
    for row in ordered:
        feature, prediction = features[row['box']], predictions[row['box']]
        if feature['dog_name'] != row['dog'] or prediction['dog'] != row['dog']:
            raise ValueError('controlled_reconstruction_identity_changed')
        runners.append(dict(runner_id=row['runner_id'], box_number=row['box'], dog_name=row['dog'],
            strict_win_odds=prediction['win_odds'], features={key:feature[key] for key in FEATURES},
            feature_source_sha256=artifact['input_hashes']['feature_source_sha256'],
            odds_source_sha256=artifact['input_hashes']['odds_source_sha256'],
            feature_freeze_timestamp=artifact['feature_freeze_timestamp'],
            odds_capture_timestamp=artifact['odds_capture_timestamp']))
    scoring_input = build_scoring_input(race_id=artifact['race_id'],runner_set_sha256=artifact['runner_set_sha256'],
        runners=sorted(runners,key=lambda row:row['runner_id']),cutoff_timestamp=artifact['feature_freeze_timestamp'],
        capture_timestamp=artifact['odds_capture_timestamp'],score_timestamp=artifact['score_timestamp'],
        jump_timestamp=artifact['jump_timestamp'],model_sha256=frozen.model_sha256,
        manifest_sha256=frozen.manifest_sha256,effective_state_sha256=frozen.effective_state_sha256)
    record = score_race(frozen,scoring_input.scorer_runners,scoring_input.provenance)
    if any(record[key] != artifact[key] for key in ('record_key','record_checksum_sha256')):
        raise ValueError('controlled_original_record_commitment_mismatch')
    if parity_binding(scoring_input,build_core_output(scoring_input,record)) != artifact['scoring_parity']:
        raise ValueError('controlled_original_scoring_parity_mismatch')
    if canonical(score_race(frozen,record['inputs']['runners'],record['inputs']['provenance'])) != canonical(record):
        raise ValueError('controlled_original_record_replay_mismatch')
    return record


class BoundedReader:
    """Finite local reads; no source requests or database connections."""
    def __init__(self, *, max_bytes=512*1024*1024, max_files=6000, max_seconds=120):
        self.max_bytes, self.max_files = max_bytes, max_files
        self.max_seconds, self.began = max_seconds, time.monotonic()
        self.bytes = self.files = 0

    def charge(self, size):
        if (self.files >= self.max_files or self.bytes + size > self.max_bytes
                or time.monotonic() - self.began > self.max_seconds):
            raise ValueError('controlled_input_read_budget')
        self.files += 1
        self.bytes += size

    def read(self,path,expected=None,*,limit=8*1024*1024):
        path=Path(path)
        if not path.is_absolute() or path.resolve()!=path:
            raise ValueError('controlled_input_read_bound_or_path')
        with path.open('rb') as stream:
            before=os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
                raise ValueError('controlled_input_read_bound_or_path')
            self.charge(before.st_size)
            raw=stream.read(limit+1)
            after=os.fstat(stream.fileno())
        def identity(value):
            return (value.st_dev,value.st_ino,value.st_mode,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
        if (identity(before)!=identity(after) or identity(after)!=identity(path.stat()) or len(raw)!=before.st_size
                or time.monotonic()-self.began>self.max_seconds):
            raise ValueError('controlled_input_changed_during_read')
        if expected is not None and hashlib.sha256(raw).hexdigest()!=expected:
            raise ValueError('controlled_input_hash_changed')
        return raw


def member_path(root, name):
    relative=Path(name)
    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        raise ValueError('controlled_member_path_invalid')
    return root/relative


def checked_archive(raw):
    archive=zipfile.ZipFile(io.BytesIO(raw))
    infos=archive.infolist()
    if (len(infos)>32 or len({item.filename for item in infos})!=len(infos)
            or sum(item.file_size for item in infos)>64*1024*1024):
        archive.close()
        raise ValueError('controlled_retained_archive_bound')
    for item in infos:
        member_path(Path('/unused'),item.filename)
    return archive


def archive_read(archive, name, reader):
    info=archive.getinfo(name)
    if info.file_size>8*1024*1024:
        raise ValueError('controlled_retained_archive_bound')
    reader.charge(info.file_size)
    raw=archive.read(info)
    if len(raw)!=info.file_size:
        raise ValueError('controlled_input_size_changed')
    return raw


def verify_execution_binding(source_binding, retained_manifest, environment_lock, reader):
    """Authenticate source hashes to the original map and the actual scorer env.

    This narrower computational replay checks Python, the distribution census
    and RECORD hashes, and the exact scorer/feature source files. It does not
    claim to reproduce browser startup or the old live service environment.
    """
    def ref(value):
        return json.loads(reader.read(value['path'],value['sha256']))
    plan=ref(source_binding['original_package_plan'])
    identity=ref(source_binding['original_source_identity'])
    runtime=ref(source_binding['original_runtime_identity'])
    generator=retained_manifest['files']['generator_source_archive']
    package=Path(generator['original_path']).parent
    if (Path(source_binding['original_package_plan']['path'])!=package/'plan.json'
            or Path(source_binding['original_runtime_identity']['path'])!=package/'runtime-identity.json'
            or Path(source_binding['original_source_identity']['path'])!=Path(plan['source_root'])/'SOURCE_IDENTITY.json'
            or plan['commit']!=source_binding['original_source_commit'] or identity['commit']!=plan['commit']
            or identity['tree']!=plan['tree'] or not re.fullmatch('[0-9a-f]{40}',plan['commit'])
            or digest(identity)!=plan['source_identity_sha256'] or digest(runtime)!=plan['runtime_sha256']):
        raise ValueError('controlled_original_execution_binding_changed')
    # Retention authenticates these exact original package bytes, not a copied
    # file agreeing with an independently mutable replay checkout.
    reader.read(Path(generator['original_path']),generator['sha256'])
    reader.read(Path(sys.executable).resolve(),plan['python_sha256'],limit=32*1024*1024)
    distributions=sorted(({
        'name':dist.metadata['Name'],'version':dist.version,
        'record_sha256':hashlib.sha256((dist.read_text('RECORD') or '').encode()).hexdigest()
    } for dist in importlib.metadata.distributions()),
        key=lambda row:(row['name'],row['version'],row['record_sha256']))
    if (runtime['version']!=sys.version or runtime['prefix']!=sys.prefix
            or runtime['executable']!=sys.executable or distributions!=runtime['distributions']
            or environment_lock['python']!=platform.python_version()
            or environment_lock['packages']!={row['name']:row['version'] for row in distributions}):
        raise ValueError('controlled_original_runtime_changed')
    root=Path(__file__).resolve().parents[2]
    from scripts.predict_market_form_residual import FEATURE_GENERATOR_FILES
    required=set(FEATURE_GENERATOR_FILES)|{
        'scripts/predict_market_form_residual.py','src/predictor/market_form_residual.py',
        'src/predictor/scoring_parity.py','config/venue_mapping.py','utils/csv_metadata.py',
        'utils/race_identity_equivalence.py'}
    if set(source_binding['replay_source_hashes'])!=required:
        raise ValueError('controlled_replay_source_pin_set')
    for name in required:
        expected=source_binding['replay_source_hashes'][name]
        if identity['files'].get(name)!=expected:
            raise ValueError('controlled_original_source_map_mismatch')
        reader.read(member_path(root,name),expected)
        reader.read(member_path(Path(plan['source_root']),name),expected)
    return plan


def load_committed_original(bundle, member, artifact_name, source_binding, *, execute=False):
    """Verify native seals and external pins before strict reconstruction.

    Source binding is a root-reviewed execution input: it must match the
    original producing package and the current replay implementation/runtime.
    An original producer artifact must itself be a declared bundle member.
    No loose file or newly invented checksum is accepted as its substitute.
    """
    if execute is not True:
        raise ValueError('controlled_reconstruction_disabled')
    from scripts.predict_market_form_residual import score_from_artifacts, load_frozen_model
    from src.predictor.on_demand import verify_indexed_prediction_bundle
    reader=BoundedReader()
    def ref(value):
        return json.loads(reader.read(value['path'],value['sha256']))
    completion=ref(member['completion']); admission=ref(member['admission'])
    bundle=Path(bundle)
    if (bundle/'bundle_manifest.json' != Path(member['bundle_manifest']['path'])
            or completion['bundle_entry']['directory'] != bundle.name
            or completion['bundle_entry']['manifest_sha256'] != member['bundle_manifest']['sha256']
            or completion['admission_sha256'] != member['admission']['sha256']
            or any(admission[k] != member[k] for k in ('job_id','prediction_id','runner_set_sha256','retained_input_manifest_sha256'))
            or completion['status']!='COMPLETE_BEFORE_CUTOFF'
            or completion['published_complete_at']!=member['original_published_complete_at']):
        raise ValueError('controlled_original_admission_changed')
    verified=verify_indexed_prediction_bundle(bundle.parent,completion['bundle_entry'])
    if (verified.result['status']!='PREDICTION_READY'
            or any(verified.request[key]!=member[key] for key in
                   ('race_id','runner_set_sha256','retained_input_manifest_sha256'))
            or any(verified.result[key]!=member[key] for key in ('job_id','prediction_id'))):
        raise ValueError('controlled_original_bundle_identity_changed')
    files=verified.manifest['files']
    if artifact_name not in files:
        raise ValueError(MISSING)
    def content(name):
        if name not in files:
            raise ValueError('controlled_input_role_missing')
        raw=reader.read(member_path(bundle,name),files[name]['sha256'])
        if len(raw)!=files[name]['bytes']:
            raise ValueError('controlled_input_size_changed')
        return raw
    artifact=json.loads(content(artifact_name))
    if not all(artifact.get(k) for k in ('record_key','record_checksum_sha256','score_timestamp')):
        raise ValueError(MISSING)
    if (source_binding.get('schema_version')!='controlled_original_execution_binding_v1'
            or source_binding.get('bundle_manifest')!=member['bundle_manifest']):
        raise ValueError('controlled_original_execution_binding_missing')
    with checked_archive(content('retained_inputs.zip')) as archive:
        retained_raw=archive_read(archive,'bundle/manifest.json',reader)
        if hashlib.sha256(retained_raw).hexdigest()!=member['retained_input_manifest_sha256']:
            raise ValueError('controlled_retained_manifest_changed')
        retained=json.loads(retained_raw)
        lock_entry=retained['files']['environment_lock']
        lock_raw=archive_read(archive,'bundle/'+lock_entry['path'],reader)
        if hashlib.sha256(lock_raw).hexdigest()!=lock_entry['sha256']:
            raise ValueError('controlled_original_environment_lock_changed')
        plan=verify_execution_binding(source_binding,retained,json.loads(lock_raw),reader)
    replay_paths=source_binding['replay_paths']
    path_keys=('form_csv','sidecar','feature_rows','feature_manifest','implementation_manifest','capture','model','manifest')
    if set(replay_paths)!=set(path_keys) or any(name not in files for name in replay_paths.values()):
        raise ValueError('controlled_replay_path_set')
    for name in replay_paths.values():content(name)
    replay=score_from_artifacts(race_id=member['race_id'],score_timestamp=datetime.fromisoformat(artifact['score_timestamp']),
        **{key+'_path':bundle/name for key,name in replay_paths.items()})
    if canonical(replay)!=canonical(artifact):
        raise ValueError('controlled_original_producer_replay_mismatch')
    frozen=load_frozen_model(bundle/replay_paths['model'],bundle/replay_paths['manifest'])
    record=reconstruct_committed_record(frozen,artifact,json.loads(content(replay_paths['feature_rows'])),execute=True)
    sealed=datetime.fromisoformat(member['original_published_complete_at'])
    jump=datetime.fromisoformat(member['jump_at'])
    if not datetime.fromisoformat(record['score_timestamp'])<=sealed<=jump-timedelta(seconds=120):
        raise ValueError('controlled_original_seal_timing')
    return frozen,record,{'status':'ORIGINAL_COMPUTATIONAL_CONTENT_CHECKSUM_VERIFIED',
        'original_record_sha256':digest(record),'record_checksum_sha256':record['record_checksum_sha256'],
        'record_key':record['record_key'],'original_source_commit':plan['commit'],
        'original_sealed_at':member['original_published_complete_at'],
        'history_snapshot_sha256':files['features/sealed_history.db']['sha256'],
        'retained_manifest_sha256':member['retained_input_manifest_sha256'],
        'original_artifact_file_sha256':files[artifact_name]['sha256'],
        'native_identity_and_input_seals_verified':True,'new_forecast_created':False}


def audit_membership(membership_path, *, expected_sha256=MEMBERSHIP_SHA256):
    """Whole fixed denominator; an inventory is not numerical qualification."""
    reader=BoundedReader()
    membership=json.loads(reader.read(membership_path,expected_sha256,limit=2*1024*1024))
    members=membership['members']
    if len(members)!=82 or len({m['race_id'] for m in members})!=82:
        raise ValueError('controlled_fixed_membership_changed')
    rows=[]
    for member in members:
        counts=Counter();ref=member['bundle_manifest'];bundle=Path(ref['path']).parent
        manifest=json.loads(reader.read(ref['path'],ref['sha256']))
        for name,entry in manifest['files'].items():
            if not name.endswith('.json') or name.startswith('model/'):
                continue
            raw=reader.read(member_path(bundle,name),entry['sha256'])
            if len(raw)!=entry['bytes']:
                raise ValueError('controlled_input_size_changed')
            counts.update(commitment_inventory(json.loads(raw)))
        archive=manifest['files']['retained_inputs.zip']
        archive_raw=reader.read(bundle/'retained_inputs.zip',archive['sha256'])
        if len(archive_raw)!=archive['bytes']:
            raise ValueError('controlled_input_size_changed')
        with checked_archive(archive_raw) as zipped:
            infos=zipped.infolist()
            for info in infos:
                if info.filename.endswith('.json') and '/model' not in info.filename:
                    raw=archive_read(zipped,info.filename,reader)
                    counts.update(commitment_inventory(json.loads(raw)))
        present=bool(counts['original_record'] or (counts['record_key'] and counts['record_checksum_sha256'] and counts['score_timestamp']))
        rows.append({'race_id':member['race_id'],'bundle_manifest':ref,'commitment_field_counts':dict(counts),
            'status':'COMMITMENT_PRESENT_REQUIRES_FULL_QUALIFICATION' if present else MISSING})
    return {'schema_version':'controlled_input_commitment_inventory_v1','membership_sha256':expected_sha256,
        'denominator':82,'records':rows,'failure_categories':dict(Counter(row['status'] for row in rows)),
        'machine_read_bytes':reader.bytes,'machine_read_files':reader.files,'controlled_pairs_derived':0,
        'original_forecasts_modified':False,'official_result_reads':0,'history_database_connections':0,
        'performance_metrics':False,'qualification':'COMMITMENT_INVENTORY_ONLY_NOT_INPUT_QUALIFICATION'}
