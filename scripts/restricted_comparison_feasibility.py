"""Fixed retained-input feasibility sample; structural egress only, no scoring.

Run after boundary review. The supplied sample must have been sealed before any
eligibility processing. Each failure remains in the output. All source reads
are read-only; feature subprocess files live only in a private temporary copy.
"""
from pathlib import Path
from datetime import datetime,date,timedelta,timezone
import argparse,contextlib,hashlib,io,json,os,resource,shutil,sys,tempfile,time

from src.predictor.research_input_firewall import InputBoundary,project_metadata,FORM_PATHS,REQUEST_PATHS,RECEIPT_PATHS,card_dates_before_decode,database_dates_before_decode
from src.predictor.on_demand import canonical_bytes


def digest(raw):return hashlib.sha256(raw).hexdigest()
def load_pinned(path,expected):
    if path.is_symlink() or not path.is_file() or path.stat().st_size>64*1024*1024:raise InputBoundary('INPUT_FILE_UNSAFE')
    raw=path.read_bytes()
    if digest(raw)!=expected:raise InputBoundary('INPUT_HASH_CHANGED')
    return raw


def verify_replay_source(blobs):
    import zipfile
    from scripts.predict_market_form_residual import FEATURE_GENERATOR_FILES
    repository=Path(__file__).resolve().parents[1]
    if blobs['feature_replay_worker']!=(repository/'scripts/retained_feature_worker.py').read_bytes():
        raise InputBoundary('UNAPPROVED_REPLAY_WORKER')
    if blobs['feature_schema']!=(repository/'accuracy_program/repaired_non_tgr_schema.json').read_bytes():
        raise InputBoundary('UNAPPROVED_FEATURE_SCHEMA')
    required={n for n in FEATURE_GENERATOR_FILES if not n.startswith('tests/')}
    required.update({'scripts/__init__.py','scripts/utils.py','utils/__init__.py','config/__init__.py'})
    with zipfile.ZipFile(io.BytesIO(blobs['generator_source_archive'])) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or not required.issubset(names):raise InputBoundary('REPLAY_SOURCE_CLOSURE')
        for name in names:
            if not name.endswith('.py') or Path(name).is_absolute() or '..' in Path(name).parts:raise InputBoundary('REPLAY_SOURCE_PATH')
            if archive.read(name)!=(repository/name).read_bytes():raise InputBoundary('UNAPPROVED_REPLAY_SOURCE')


def assess(row):
    wall=time.monotonic();cpu=time.process_time();phase='MANIFEST';result={k:row[k] for k in ('sample_id','venue','date','field_size')}
    try:
        manifest_path=Path(row['retained_manifest_path']);root=manifest_path.parent
        manifest=json.loads(load_pinned(manifest_path,row['retained_manifest_sha256']))
        bundle=Path(row['prediction_bundle_path']);bm=json.loads(load_pinned(bundle/'bundle_manifest.json',row['prediction_manifest_sha256']))
        def member(relative):
            p=root/relative
            if p.resolve()!=p or not p.is_relative_to(root):raise InputBoundary('INPUT_PATH_UNSAFE')
            return p
        phase='SOURCE_HASHES';files=manifest['files'];blobs={}
        for role,entry in {**files,'history':manifest['history'],'history_seal':manifest['history_seal']}.items():
            blobs[role]=load_pinned(member(entry['path']),entry['sha256'])
        for key in ('request.json','odds_receipt.json'):
            blobs[key]=load_pinned(bundle/key,bm['files'][key]['sha256'])
        phase='RESULT_FREE_METADATA';request=project_metadata(blobs['request.json'],REQUEST_PATHS);receipt=project_metadata(blobs['odds_receipt.json'],RECEIPT_PATHS)
        metadata=project_metadata(blobs['form_metadata'],FORM_PATHS)
        phase='TIMING_ALIGNMENT';jump=datetime.fromisoformat(manifest['jump_at']);target=date.fromisoformat(row['date'])
        complete=json.loads((root/'completion.json').read_bytes());sealed=datetime.fromisoformat(complete['inputs_sealed_at'])
        if complete['manifest_sha256']!=row['retained_manifest_sha256']:raise InputBoundary('RETENTION_COMPLETION_BINDING')
        if manifest['race_id']!=row['race_id'] or request['race_id']!=row['race_id'] or request['retained_input_manifest_sha256']!=row['retained_manifest_sha256']:raise InputBoundary('RACE_IDENTITY_MISMATCH')
        from scripts.build_form_only_v1_packet import capture_timestamp,dog_token
        captured=capture_timestamp(metadata,require_timezone=True);quote=datetime.fromisoformat(receipt['captured_at'])
        if not captured<=datetime.fromisoformat(manifest['capture_started_at'])<=datetime.fromisoformat(manifest['capture_completed_at'])<=sealed<jump-timedelta(seconds=120):raise InputBoundary('SOURCE_OR_RETENTION_LATE')
        if not 120<=(jump-quote).total_seconds()<=600:raise InputBoundary('QUOTE_OUTSIDE_T10_T2')
        runners=request['runners'];win=receipt['markets']['win']
        if len(runners)!=row['field_size'] or len(win)!=len(runners) or sorted((r['box_number'],dog_token(r['dog_name'])) for r in win)!=sorted((r['box_number'],dog_token(r['display_name'])) for r in runners):raise InputBoundary('RUNNER_ALIGNMENT')
        result.update(source_timing_identity='PASS',quote_lead_seconds=(jump-quote).total_seconds(),retention_lead_seconds=(jump-sealed).total_seconds())
        phase='TARGET_FIREWALL'
        card=card_dates_before_decode(blobs['normalized_form'],target,captured.date())
        db=database_dates_before_decode(member(manifest['history']['path']),row['race_id'],target,captured.date())
        old=lambda ds:sum(date(2026,7,15)<=d<=date(2026,10,31) for d in ds)
        result.update(card_history_dates=len(card),database_history_races=len(db),card_blanket_overlap=old(card),database_blanket_overlap=old(db),
            history_date_min=min(card+db).isoformat() if card+db else None,history_date_max=max(card+db).isoformat() if card+db else None,
            blanket_history_policy='REJECTED' if old(card)+old(db) else 'PASS',target_firewall='PASS')
        phase='CANDIDATE_FEATURE_ROUTE';start=time.monotonic()
        from src.predictor.comparison_candidates import card_features
        features=card_features(blobs['normalized_form'],metadata,row['race_id'],runners,captured_at=captured,denied_history_intervals=())
        result.update(candidate_feature_route='PASS',candidate_feature_sha256=digest(canonical_bytes(features)),candidate_feature_seconds=time.monotonic()-start)
        phase='APPROVED_REPLAY_SOURCE';verify_replay_source(blobs)
        phase='PRODUCTION_FEATURE_ROUTE';start=time.monotonic()
        from race_collection.retained_feature_replay import generate_retained_features
        # Same immutable worker/source ZIP/environment. No original-file writes;
        # guard established that all DB/card outcomes are strictly earlier.
        with tempfile.TemporaryDirectory(prefix='restricted-features-') as temporary:
            private=Path(temporary)
            for role,entry in {**files,'history':manifest['history']}.items():
                dest=private/entry['path'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(canonical_bytes(metadata) if role=='form_metadata' else blobs[role])
            replay=generate_retained_features(private,files)
        if digest(replay)!=manifest['feature_values_sha256']:raise InputBoundary('PRODUCTION_FEATURE_REPLAY_MISMATCH')
        result.update(production_feature_route='IDENTICAL_FEATURES_WITH_RESTRICTED_METADATA_PROJECTION',projected_metadata_sha256=digest(canonical_bytes(metadata)),production_feature_sha256=digest(replay),production_feature_seconds=time.monotonic()-start,status='QUALIFIED_INPUT_EXECUTION')
    except Exception as exc:
        result.update(status='NOT_QUALIFIED',failure_phase=phase,reason=str(exc) if isinstance(exc,InputBoundary) else 'EXECUTION_FAILED_'+type(exc).__name__)
    result.update(wall_seconds=time.monotonic()-wall,cpu_seconds=time.process_time()-cpu)
    return result


def run(sample_path,expected,out):
    from scripts.check_freshness_service import deny_network
    deny_network();os.umask(0o077)
    sample=json.loads(load_pinned(sample_path,expected))
    if sample['status']!='SAMPLE_FROZEN_BEFORE_FEATURE_ELIGIBILITY':raise InputBoundary('UNSEALED_SAMPLE')
    rows=[]
    for row in sample['records']:
        # Do not persist or release arbitrary library messages, even on failure.
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):value=assess(row)
        rows.append(value)
    value={'sample_sha256':expected,'records':rows,'denominator':len(rows),'predictions_computed':False,'target_outcomes_accessed':False,'protected_history_processing':'user-authorized, strictly earlier feature inputs only','raw_values_released':False,'network':'kernel denied','peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    with out.open('x') as f:json.dump(value,f,sort_keys=True,indent=2)
    print(json.dumps({'output':str(out),'sha256':digest(out.read_bytes()),'denominator':len(rows),'qualified':sum(r['status']=='QUALIFIED_INPUT_EXECUTION' for r in rows)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sample',type=Path,required=True);p.add_argument('--sample-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:run(a.sample,a.sample_sha256,a.out)
    except Exception:print('{"status":"RESTRICTED_WORKER_FAILED"}');raise SystemExit(2)
