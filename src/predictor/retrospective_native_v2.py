"""Authenticate native v2 evidence, replay its parent, derive a retrospective pair.

No original shadow record is manufactured. All 82 members remain represented;
numerical mismatch is an exclusion, never a request to tune the comparison.
"""
from collections import Counter
from datetime import datetime,timedelta,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from src.predictor.controlled_retained_inputs import (
    BoundedReader,MEMBERSHIP_SHA256,archive_read,canonical,checked_archive,digest,member_path)
from src.predictor.controlled_adjustment_pair import PARENT_MODEL_SHA256,PARENT_MANIFEST_SHA256,SCORER_SOURCE_SHA256

SCOPE_SHA256='f4a0464fdc11b80301fd542bd5ae4865c6a752dd5047fba3a0f54705077c5932'
SERIALIZATION='EXACT_NATIVE_V2_CANONICAL_FULL_ROWS_NO_ROUNDING_OR_TOLERANCE'
ROOT=Path(__file__).resolve().parents[2]
WORKER=ROOT/'scripts/derive_retrospective_native_v2_pairs.py'


class QualificationFailure(ValueError):
    pass


def require(condition,category):
    if not condition:raise QualificationFailure(category)


def stamp(value):
    parsed=datetime.fromisoformat(value)
    require(parsed.utcoffset() is not None,'TIMESTAMP_TIMEZONE_MISSING')
    return parsed


def reference(reader,ref):
    return json.loads(reader.read(ref['path'],ref['sha256']))


def pair_from_replay(*,member,result,request,production,inputs,replay,provenance,derived_at):
    """Only the checked loader supplies these evidence objects in production."""
    require(isinstance(derived_at,datetime) and derived_at.utcoffset() is not None,'DERIVATION_TIME_INVALID')
    require(replay['model_sha256']==PARENT_MODEL_SHA256 and replay['manifest_sha256']==PARENT_MANIFEST_SHA256
        and replay['variants']=={'full_strength':1.0,'half_strength':0.5},'PARENT_OR_STRENGTH_CHANGED')
    require(replay['race_id']==member['race_id'] and stamp(replay['jump_timestamp'])==stamp(member['jump_at']),
        'REPLAY_RACE_IDENTITY_CHANGED')
    jump=stamp(member['jump_at']);sealed=stamp(member['original_published_complete_at'])
    anchor=stamp(production['completed_at']);capture=stamp(inputs['captured_at'])
    require(stamp(member['original_admitted_at'])<=anchor<=sealed<jump-timedelta(seconds=120)
        and stamp(replay['feature_freeze_timestamp'])<=anchor
        and stamp(replay['odds_capture_timestamp'])==capture<=anchor
        and 120<=(jump-capture).total_seconds()<=600 and derived_at>=sealed,'HISTORICAL_TIMING_INVALID')
    roster=request['runners'];boxes={r['box_number']:r for r in roster}
    native_ids=[r['source_native_runner_id'] for r in roster]
    require(len(boxes)==len(roster)>=2 and all(isinstance(v,str) and v for v in native_ids)
        and len(set(native_ids))==len(native_ids),'NATIVE_IDENTITY_INCOMPLETE')
    enriched=[{key:value for key,value in row.items() if key!='win_odds'} for row in inputs['runners']]
    require(canonical(enriched)==canonical(roster),'COMMON_NATIVE_ROSTER_CHANGED')
    prices={r['box_number']:r['win_odds'] for r in inputs['runners']}
    rows=replay['predictions'];by_box={r['box']:r for r in rows}
    require(len(by_box)==len(rows)==len(roster) and set(by_box)==set(boxes),'REPLAY_FIELD_INCOMPLETE')
    for box,row in by_box.items():
        require(row['dog']==boxes[box]['display_name'] and row['win_odds']==prices[box],
            'REPLAY_RUNNER_OR_ODDS_CHANGED')
        require(all(type(row[key]) in (float,int) and math.isfinite(row[key]) and 0<row[key]<1
            for key in ('full_probability','half_probability','market_probability')),'REPLAY_PROBABILITY_INVALID')
    ordered=sorted(rows,key=lambda r:(-r['full_probability'],r['box']))
    rebuilt=[{'rank':rank,'box_number':r['box'],'dog_name':r['dog'],
        'identity':boxes[r['box']]['identity'],'source_native_runner_id':boxes[r['box']]['source_native_runner_id'],
        'probability':r['full_probability']} for rank,r in enumerate(ordered,1)]
    require(canonical(rebuilt)==canonical(result['prediction']['predictions'])
        and digest(rebuilt)==result['evidence']['prediction_output_sha256'],'FULL_ARM_EXACT_REPLAY_MISMATCH')
    comparative=[{'box_number':r['box'],'identity':boxes[r['box']]['identity'],'dog_name':r['dog'],
        'probability':r['full_probability']} for r in sorted(rows,key=lambda r:r['box'])]
    require(canonical(comparative)==canonical(production['predictions']),'COMPARISON_FULL_ARM_CHANGED')
    source_hashes=replay['input_hashes']
    for a,b in (('form_sha256','form_csv_sha256'),('sidecar_sha256','sidecar_sha256'),
                ('capture_sha256','capture_artifact_sha256'),('production_feature_rows_sha256','feature_rows_sha256')):
        require(inputs[a]==source_hashes[b],'REPLAY_INPUT_HASH_CHANGED')
    common={'membership_sha256':MEMBERSHIP_SHA256,'bundle_manifest':member['bundle_manifest'],
        'retained_input_manifest_sha256':member['retained_input_manifest_sha256'],
        'input_hashes':source_hashes,'native_roster_sha256':digest(roster),
        'parent_effective_state_sha256':replay['effective_state_sha256'],'provenance':provenance}
    ordered_ids=sorted(native_ids)
    native_to_row={boxes[r['box']]['source_native_runner_id']:r for r in rows}
    output={'schema_version':'retrospective_native_v2_controlled_pair_v1',
        'status':'RETROSPECTIVE_COMPUTATION_NOT_ORIGINAL_PREJUMP_OR_SCIENTIFIC',
        'race_id':member['race_id'],'derived_at':derived_at.isoformat(),'original_score_timestamp':None,
        'historical_validation_anchor':{'at':production['completed_at'],
            'role':'RECORDED_PRODUCTION_COMPLETION_NOT_ORIGINAL_SCORE_TIME'},
        'original_published_complete_at':member['original_published_complete_at'],
        'parent_model_sha256':PARENT_MODEL_SHA256,'parent_manifest_sha256':PARENT_MANIFEST_SHA256,
        'serialization_contract':SERIALIZATION,'verified_original_full_rows_sha256':digest(rebuilt),
        'common_inputs':common,'common_input_sha256':digest(common),'native_runner_ids':ordered_ids,
        'candidates':[{'candidate_id':f'production_parent_alpha_{label}','strength':strength,
            'common_input_sha256':digest(common),'probabilities':[native_to_row[r][key] for r in ordered_ids]}
            for label,strength,key in [('1_0',1.0,'full_probability'),('0_5',0.5,'half_probability')]],
        'training':False,'performance_evaluation':False,'outcomes_present':False,
        'original_forecasts_modified':False,'scientific_membership_created':False}
    return {**output,'derivation_sha256':digest(output)}


def verify_package(package,entry,reader):
    plan=reference(reader,entry['plan']);identity=reference(reader,entry['source_identity'])
    runtime=reference(reader,entry['runtime_identity'])
    require(Path(entry['plan']['path'])==package/'plan.json'
        and Path(entry['runtime_identity']['path'])==package/'runtime-identity.json'
        and Path(entry['source_identity']['path'])==Path(plan['source_root'])/'SOURCE_IDENTITY.json'
        and plan['commit']==identity['commit']==entry['commit'] and plan['tree']==identity['tree']
        and digest(identity)==plan['source_identity_sha256'] and digest(runtime)==plan['runtime_sha256'],
        'ORIGINAL_PACKAGE_IDENTITY_CHANGED')
    require(identity['files'].get('src/predictor/market_form_residual.py')==SCORER_SOURCE_SHA256,
        'ORIGINAL_SCORER_CHANGED')
    for name,expected in identity['files'].items():
        reader.read(member_path(Path(plan['source_root']),name),expected,limit=64*1024*1024)
    for key in ('generator_archive','source_archive'):
        ref=entry[key];reader.read(ref['path'],ref['sha256'],limit=64*1024*1024)
    require(entry['source_archive']['sha256']==plan['source_archive_sha256'],'ORIGINAL_SOURCE_ARCHIVE_CHANGED')
    reader.read(Path(plan['python']).resolve(),plan['python_sha256'],limit=32*1024*1024)
    return {'plan':plan,'identity':identity,'runtime':runtime,'entry':entry}


def load_and_derive(member,packages,cache,reader,*,derived_at,worker_seconds):
    from src.predictor.on_demand import verify_indexed_prediction_bundle
    admission=reference(reader,member['admission']);completion=reference(reader,member['completion'])
    require(completion['admission_sha256']==member['admission']['sha256']
        and completion['status']=='COMPLETE_BEFORE_CUTOFF'
        and completion['published_complete_at']==member['original_published_complete_at']
        and admission['admitted_at']==member['original_admitted_at']
        and admission['race']['race_id']==member['race_id']
        and stamp(admission['race']['jump_timestamp'])==stamp(member['jump_at'])
        and all(admission[k]==completion[k]==member[k] for k in
            ('job_id','prediction_id','runner_set_sha256','retained_input_manifest_sha256')),
        'ADMISSION_OR_COMPLETION_CHANGED')
    bundle=Path(member['bundle_manifest']['path']).parent
    manifest=reference(reader,member['bundle_manifest']);files=manifest['files']
    require(completion['bundle_entry']['directory']==bundle.name
        and completion['bundle_entry']['manifest_sha256']==member['bundle_manifest']['sha256'],
        'BUNDLE_ENTRY_CHANGED')
    # Charge the native verifier's complete outer-file pass before invoking it.
    reader.charge(Path(member['bundle_manifest']['path']).stat().st_size)
    for entry in files.values():reader.charge(entry['bytes'])
    verified=verify_indexed_prediction_bundle(bundle.parent,completion['bundle_entry'])
    require(verified.manifest==manifest and verified.result['status']=='PREDICTION_READY'
        and verified.result['race']==admission['race'],'NATIVE_BUNDLE_INVALID')
    def content(name):
        entry=files[name];raw=reader.read(member_path(bundle,name),entry['sha256'],limit=64*1024*1024)
        require(len(raw)==entry['bytes'],'BUNDLE_MEMBER_SIZE_CHANGED');return raw
    production=json.loads(content('comparison/production.json'));inputs=json.loads(content('comparison/inputs.json'))
    require(production['candidate']=='production' and production['status']=='SEALED' and production['failure'] is None
        and production['model_sha256']==PARENT_MODEL_SHA256
        and all(production[k]==admission[k] for k in ('race','job_id','prediction_id','runner_set_sha256',
            'plan_sha256','retained_input_manifest_sha256'))
        and production['admission_sha256']==member['admission']['sha256'],'PRODUCTION_SEAL_CHANGED')
    require(content('comparison/plan.json')==reader.read(member['original_plan']['path'],member['original_plan']['sha256'])
        and files['comparison/plan.json']['sha256']==admission['plan_sha256'],'ORIGINAL_COMPARISON_PLAN_CHANGED')
    require(digest(json.loads(content('odds_receipt.json')))==inputs['odds_receipt_sha256'],
        'ODDS_RECEIPT_CHANGED')
    require(production['input_identity']=={key:inputs[key] for key in production['input_identity']},
        'COMMON_INPUT_IDENTITY_CHANGED')
    with checked_archive(content('retained_inputs.zip')) as archive:
        raw=archive_read(archive,'bundle/manifest.json',reader)
        require(hashlib.sha256(raw).hexdigest()==member['retained_input_manifest_sha256'],'RETAINED_MANIFEST_CHANGED')
        retained=json.loads(raw)
        generator=retained['files']['generator_source_archive'];package=Path(generator['original_path']).parent
        require(str(package) in packages,'ORIGINAL_PACKAGE_UNBOUND')
        if str(package) not in cache:cache[str(package)]=verify_package(package,packages[str(package)],reader)
        source=cache[str(package)]
        require(source['entry']['generator_archive']=={'path':generator['original_path'],'sha256':generator['sha256']},
            'RETAINED_GENERATOR_PACKAGE_CHANGED')
        lockentry=retained['files']['environment_lock'];lockraw=archive_read(archive,'bundle/'+lockentry['path'],reader)
        require(hashlib.sha256(lockraw).hexdigest()==lockentry['sha256'],'ENVIRONMENT_LOCK_CHANGED')
        environment=json.loads(lockraw)
    history=json.loads(content('features/history_seal.json'))
    require(history['target_race_id']==member['race_id']
        and stamp(history['cutoff_timestamp'])==stamp(member['jump_at'])
        and history['target_rows_materialized']==0 and history['at_or_after_cutoff_rows_materialized']==0
        and history['sealed_sha256']==files['features/sealed_history.db']['sha256']
        and history['source_sha256']==retained['history']['source_sha256'],'HISTORY_SEAL_CHANGED')
    forms=[name for name in files if name.startswith('source/') and name.endswith('.csv')]
    require(len(forms)==1,'FORM_ROLE_AMBIGUOUS')
    paths={'form_csv':forms[0],'sidecar':forms[0]+'.metadata.json',
        'feature_rows':'features/sealed/shadow_feature_rows.json','feature_manifest':'features/sealed/shadow_manifest.json',
        'implementation_manifest':'features/sealed/implementation_file_manifest.json','capture':'source/capture.json',
        'model':'model/model.json','manifest':'model/manifest.json'}
    for name in paths.values():reader.charge(files[name]['bytes'])
    worker_request={'source_root':source['plan']['source_root'],'runtime_identity':source['runtime'],
        'environment_lock':environment,'race_id':member['race_id'],'historical_validation_anchor':production['completed_at'],
        'replay_paths':{key:str(member_path(bundle,name)) for key,name in paths.items()}}
    remaining=reader.max_seconds-(time.monotonic()-reader.began)
    require(remaining>0,'EXECUTION_TIME_BOUND')
    process=subprocess.run([source['plan']['python'],'-B',str(WORKER),'--worker'],input=canonical(worker_request),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=min(worker_seconds,remaining),cwd=source['plan']['source_root'])
    require(not process.stderr and len(process.stdout)<=256*1024,'ORIGINAL_PRODUCER_REPLAY_REJECTED')
    replay=json.loads(process.stdout)
    if replay.get('failure_category') in {'worker_computational_runtime_changed','worker_mixed_source_import',
            'worker_source_path','feature_generator_implementation_hash_mismatch'}:
        raise QualificationFailure('EXECUTION_SOURCE_OR_RUNTIME_CHANGED')
    require(process.returncode==0,'ORIGINAL_PRODUCER_REPLAY_REJECTED')
    require(replay['worker_status']=='REPLAYED_WITH_HISTORICAL_VALIDATION_ANCHOR'
        and all(source['identity']['files'].get(name)==sha for name,sha in replay['loaded_source_files'].items()),
        'WORKER_IMPORTED_SOURCE_CHANGED')
    provenance={'source_commit':source['plan']['commit'],'source_identity':source['entry']['source_identity'],
        'runtime_identity':source['entry']['runtime_identity'],'environment_lock_sha256':lockentry['sha256'],
        'history_snapshot_sha256':history['sealed_sha256'],'source_archive':source['entry']['source_archive'],
        'generator_archive':source['entry']['generator_archive'],'loaded_source_hashes':replay['loaded_source_files']}
    return pair_from_replay(member=member,result=verified.result,request=verified.request,production=production,
        inputs=inputs,replay=replay,provenance=provenance,derived_at=derived_at)


def put(path,value):
    raw=canonical(value)
    require(len(raw)<=1024*1024,'EXECUTION_OUTPUT_FILE_BOUND')
    with path.open('xb') as stream:
        os.fchmod(stream.fileno(),0o600);stream.write(raw);stream.flush();os.fsync(stream.fileno())


def run(authority_path,authority_sha256):
    began=time.monotonic();output=None;claim=False;records=[];members=[]
    try:
        bootstrap=BoundedReader();authority=reference(bootstrap,{'path':str(authority_path),'sha256':authority_sha256})
        now=datetime.now(timezone.utc)
        require(authority['schema_version']=='retrospective_native_v2_execution_authority_v1'
            and stamp(authority['issued_at'])<=now<stamp(authority['expires_at'])
            and authority['scope']['sha256']==SCOPE_SHA256
            and authority['membership']['sha256']==MEMBERSHIP_SHA256
            and authority['serialization_contract']==SERIALIZATION,'EXECUTION_AUTHORITY_INVALID')
        scope=reference(bootstrap,authority['scope'])
        require(scope['membership_sha256']==MEMBERSHIP_SHA256 and scope['cohort_members']==82,'SCOPE_CHANGED')
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        require(commit==authority['source_commit'] and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT),
            'EXECUTION_SOURCE_CHANGED')
        required={'scripts/derive_retrospective_native_v2_pairs.py','src/predictor/retrospective_native_v2.py',
            'src/predictor/controlled_retained_inputs.py','src/predictor/on_demand.py','src/predictor/retained_inputs.py'}
        require(set(authority['implementation_files'])==required,'EXECUTION_IMPLEMENTATION_PINS_CHANGED')
        for name,expected in authority['implementation_files'].items():bootstrap.read(ROOT/name,expected)
        limits=authority['limits']
        require(limits['members']==82 and 0<limits['max_bytes']<=4*1024**3
            and 0<limits['max_reads']<=50000 and 0<limits['max_wall_seconds']<=600
            and 0<limits['worker_seconds']<=30 and limits['max_output_bytes']==24*1024*1024,
            'EXECUTION_LIMITS_INVALID')
        # The monotonic read deadline covers the original wall-clock expiry too;
        # a slow source/package verification cannot continue past the authority.
        remaining=(stamp(authority['expires_at'])-datetime.now(timezone.utc)).total_seconds()
        reader=BoundedReader(max_bytes=limits['max_bytes'],max_files=limits['max_reads'],
            max_seconds=min(limits['max_wall_seconds'],time.monotonic()-began+remaining))
        reader.bytes=bootstrap.bytes;reader.files=bootstrap.files;reader.began=began
        membership=reference(reader,authority['membership']);members=membership['members']
        require(len(members)==82 and len({m['race_id'] for m in members})==82,'MEMBERSHIP_CHANGED')
        packages=reference(reader,authority['package_bindings'])['packages']
        output=Path(authority['output_path'])
        require(output.is_absolute() and output.resolve()==output,'OUTPUT_PATH_INVALID')
        require(datetime.now(timezone.utc)<stamp(authority['expires_at']),'EXECUTION_AUTHORITY_EXPIRED')
        reader.charge(0);output.mkdir(mode=0o700,exist_ok=False)
        put(output/'claim.json',{'authority_sha256':authority_sha256,'membership_sha256':MEMBERSHIP_SHA256,
            'at':datetime.now(timezone.utc).isoformat()});claim=True
        cache={}
        for index,member in enumerate(members):
            require(datetime.now(timezone.utc)<stamp(authority['expires_at']),'EXECUTION_AUTHORITY_EXPIRED')
            reader.charge(0)
            try:
                pair=load_and_derive(member,packages,cache,reader,derived_at=datetime.now(timezone.utc),
                    worker_seconds=min(limits['worker_seconds'],
                        (stamp(authority['expires_at'])-datetime.now(timezone.utc)).total_seconds()))
                require(datetime.now(timezone.utc)<stamp(authority['expires_at']),'EXECUTION_AUTHORITY_EXPIRED')
                reader.charge(0)
                require(len(canonical(pair))<=256*1024,'EXECUTION_PAIR_OUTPUT_BOUND')
                path=output/f'pair-{index:03d}.private.json';put(path,pair)
                records.append({'race_id':member['race_id'],'status':'EXACT_FULL_ARM_VERIFIED_PAIR_DERIVED',
                    'artifact':{'path':str(path),'sha256':digest(pair)}})
            except QualificationFailure as exc:
                if str(exc).startswith('EXECUTION_'):raise
                records.append({'race_id':member['race_id'],'status':'EXCLUDED','reason':str(exc)})
            except (ValueError,KeyError,TypeError,OSError,subprocess.TimeoutExpired) as exc:
                # Budget/path/hash failures are global integrity stops, not ordinary exclusions.
                raise QualificationFailure('RETAINED_INPUT_OR_EXECUTION_FAILURE') from exc
        summary={'schema_version':'retrospective_native_v2_derivation_inventory_v1','status':'COMPLETE',
            'membership_sha256':MEMBERSHIP_SHA256,'denominator':82,'records':records,
            'categories':dict(Counter(r.get('reason',r['status']) for r in records)),
            'guarded_evidence_read_bytes':reader.bytes,'guarded_evidence_read_operations':reader.files,
            'accounting_scope':'application evidence reads and precharged native verifier/producer passes; excludes interpreter imports and distribution census',
            'wall_seconds':time.monotonic()-began,'source_commit':commit,'authority_sha256':authority_sha256,
            'provider_requests':0,'official_result_reads':0,'performance_metrics':False}
        put(output/'inventory.json',summary)
        put(output/'status.json',{'status':'RETROSPECTIVE_NATIVE_V2_DERIVATION_COMPLETE','denominator':82,
            'categories':summary['categories'],'inventory_sha256':digest(summary)})
        print(json.dumps({'status':'RETROSPECTIVE_NATIVE_V2_DERIVATION_COMPLETE','denominator':82,
            'categories':summary['categories']}));return 0
    except Exception as exc:
        status={'status':'FAILED_PRESERVED_DERIVATION_CLAIM' if claim else 'NOT_STARTED',
            'failure_category':str(exc) if isinstance(exc,QualificationFailure) else 'EXECUTION_FAILED',
            'exception_type':type(exc).__name__,'denominator':82,'completed_records':records,
            'unattempted_race_ids':[m['race_id'] for m in members[len(records):]],
            'unattempted_members_not_yet_loaded':82 if not members else 0}
        if claim and output is not None and not (output/'status.json').exists():put(output/'status.json',status)
        print(json.dumps(status));return 2
