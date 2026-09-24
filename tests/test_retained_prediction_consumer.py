"""Invented races through real receipt, retention, generator and prediction entrypoint."""
from datetime import timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import pytest

from race_collection import scheduled_input_retention as scheduled
from scripts import predict_race_now as predictor
from src.predictor.on_demand import Dependencies, PredictionBlocked, sha256_file, verify_bundle, verify_indexed_prediction_bundle
from tests.test_prospective_input_retention import generator_files

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def retained_case(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('consumer_scheduled_fixture', ROOT/'tests/race_collection/test_scheduled_forward_corpus.py')
    source = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = source
    spec.loader.exec_module(source)
    from scripts.capture_thedogs_market_history import TimedResponse, persist_primary_race_page_evidence
    def add_page(sidecar):
        page = TimedResponse(requested_url=source.RACE_URL, final_url=source.RACE_URL,
            request_start_utc=source.NOW-timedelta(seconds=1), request_end_utc=source.NOW,
            status_code=200, headers={}, body=b'<html>invented pre-race source</html>')
        sidecar['primary_race_page_evidence'] = persist_primary_race_page_evidence(
            artifact_root=Path(sidecar['raw_export_path']).parent.parent,
            race_discovery_key=source.RACE_ID, response=page, canonical_runner_set={})
    case = source._fixture(tmp_path, mutate_sidecar=add_page, venue="WAR",
        sportsbet_url="https://www.sportsbet.com.au/betting/greyhound-racing/warrnambool/race-1-999")
    db = tmp_path/'synthetic.db'
    with sqlite3.connect(db) as conn:
        conn.executescript("CREATE TABLE race_metadata(race_id TEXT,race_date TEXT,data_source TEXT,url TEXT); CREATE TABLE dog_race_data(race_id TEXT,dog_name TEXT,finish_position INTEGER,data_source TEXT); INSERT INTO race_metadata VALUES ('invented-prior','2026-08-01','fixture','https://fixture.test/prior'); INSERT INTO dog_race_data VALUES ('invented-prior','Alpha',1,'fixture'),('invented-prior','Bravo',2,'fixture');")
    static = {key:{'path':str(path),'sha256':digest} for key,(path,digest) in generator_files(tmp_path).items()}
    claim = tmp_path/'retention-claim'
    claim.mkdir()
    request = {'config':{'static_files':static,'max_bundle_bytes':10_000_000}, 'config_sha256':'a'*64,
        'context':{'protocol_root':str(case.protocol.root),'evidence_root':str(case.evidence_root),
                   'collector_run_id':'scheduled-run-1','history_source':str(db)},
        'plan_item':case.plan_item,'attempt':case.attempt,'receipt_publish':case.receipt_publish,
        'cutoff':(source.JUMP-timedelta(minutes=1)).isoformat()}
    path = claim/'request.json'; path.write_text(json.dumps(request))
    monkeypatch.setattr(scheduled, '_now', lambda: source.NOW+timedelta(seconds=5))
    result = scheduled._retain(path)
    (claim/'terminal.json').write_text(json.dumps({**result,'config_sha256':'a'*64,
        'accepted_at':(source.NOW+timedelta(seconds=6)).isoformat()}))
    now = source.NOW+timedelta(seconds=10)
    args = predictor.build_parser().parse_args(['--race-id',case.plan_item['race_id'],'--odds-source','receipt',
        '--db',str(db),'--output-root',str(tmp_path/'predictions'),
        '--capture-evidence-root',str(case.evidence_root),'--collector-request-root',str(case.protocol.root),
        '--retained-input-bundle',str(claim/'bundle'), '--retained-input-manifest-sha256',result['manifest_sha256']])
    scored = []
    def fixture_score(**values):
        rows = json.loads(values['feature_rows_path'].read_bytes())
        scored.append(rows)
        return {'model_sha256':sha256_file(values['model_path']),
            'manifest_sha256':sha256_file(values['manifest_path']),
            'predictions':[{'box_number':row['box_number'],'dog_name':row['dog_name'],
                'full_probability':0.5,'half_probability':0.5,'market_probability':0.5,'win_odds':2.0}
                for row in rows]}
    target = {**case.verified_index.races[0], 'race_time':'12:20'}
    deps = Dependencies(schedule=lambda *a:[target],
        seal_features=predictor.seal_live_features,score_residual=fixture_score,now=lambda:now,monotonic=lambda:0)
    return args,deps,scored,case,source


def test_actual_consumer_uses_retained_history_after_original_removed(retained_case):
    args,deps,scored,case,source = retained_case
    args.db.unlink()
    result = predictor.run_prediction(args,deps)
    assert result['status'] == 'PREDICTION_READY'
    assert len(scored) == 1
    bundle = next(path for path in args.output_root.iterdir() if path.is_dir())
    assert sha256_file(bundle/'features/sealed_history.db') == sha256_file(args.retained_input_bundle/'history.db')
    expected = json.loads((args.retained_input_bundle/'feature_values.json').read_bytes())
    names = expected[0]['features'].keys()
    actual = [{'race_id':row['race_id'],'dog_name':row['dog_name'],'box_number':row['box_number'],
               'features':{name:row[name] for name in names}} for row in scored[0]]
    actual.sort(key=lambda row:(row['race_id'],row['box_number'],row['dog_name']))
    assert actual == expected
    assert len(names) == 16
    assert (bundle/'retained_inputs.zip').is_file()
    assert verify_bundle(bundle)['schema_version'] == 'on_demand_prediction_bundle_manifest_v2'
    entry = json.loads((args.output_root/'prediction_bundle_index_v1.json').read_bytes())['entries'][0]
    assert verify_indexed_prediction_bundle(args.output_root,entry).result == result


@pytest.mark.parametrize('failure',['missing','manifest','history','features','receipt','source','model','race','runner','late'])
def test_retained_consumer_rejects_mismatch_without_live_fallback(retained_case, failure):
    args,deps,scored,case,source = retained_case
    root=args.retained_input_bundle
    manifest=json.loads((root/'manifest.json').read_bytes())
    if failure == 'missing': (root/'history_seal.json').unlink()
    elif failure == 'manifest': args.retained_input_manifest_sha256='0'*64
    elif failure in {'history','features','receipt','source','model'}:
        relative={'history':'history.db','features':'feature_values.json',
                  'receipt':manifest['files']['exact_odds_receipt']['path'],
                  'source':manifest['files']['normalized_form']['path'],
                  'model':manifest['files']['model']['path']}[failure]
        (root/relative).write_bytes(b'changed')
    elif failure == 'race':
        manifest['race_id']='invented different race'
    elif failure == 'runner':
        rows=json.loads((root/'feature_values.json').read_bytes());rows[0]['dog_name']='Different Dog'
        raw=(json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n').encode()
        (root/'feature_values.json').write_bytes(raw)
        manifest['feature_values_sha256']=hashlib.sha256(raw).hexdigest()
    elif failure == 'late':
        deps.now=lambda:source.JUMP-timedelta(seconds=30)
    if failure in {'race','runner'}:
        raw=(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n').encode()
        (root/'manifest.json').write_bytes(raw);args.retained_input_manifest_sha256=hashlib.sha256(raw).hexdigest()
        for path in [root/'completion.json',root.parent/'terminal.json']:
            value=json.loads(path.read_bytes());value['manifest_sha256']=args.retained_input_manifest_sha256;path.write_text(json.dumps(value))
    args.db.unlink()
    with pytest.raises(PredictionBlocked): predictor.run_prediction(args,deps)
    assert scored == []


def _rebind_synthetic_manifest(args, manifest):
    raw=(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n').encode()
    (args.retained_input_bundle/'manifest.json').write_bytes(raw)
    args.retained_input_manifest_sha256=hashlib.sha256(raw).hexdigest()
    for path in [args.retained_input_bundle/'completion.json',args.retained_input_bundle.parent/'terminal.json']:
        value=json.loads(path.read_bytes())
        value['manifest_sha256']=args.retained_input_manifest_sha256
        path.write_text(json.dumps(value))


@pytest.mark.parametrize('role',['environment_lock','generator_source_archive','configuration','normalized_form','exact_odds_receipt','model'])
def test_rebound_synthetic_manifest_cannot_substitute_consumer_identity(retained_case, role):
    import io
    import zipfile
    args,deps,scored,_,_=retained_case
    root=args.retained_input_bundle
    manifest=json.loads((root/'manifest.json').read_bytes())
    entry=manifest['files'][role]
    if role == 'environment_lock':
        value=json.loads((root/entry['path']).read_bytes());value['python']='0.0.0';raw=json.dumps(value).encode()
    elif role == 'generator_source_archive':
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w'): pass
        raw=output.getvalue()
    else: raw=b'changed invented identity'
    (root/entry['path']).write_bytes(raw)
    entry['sha256']=hashlib.sha256(raw).hexdigest()
    _rebind_synthetic_manifest(args,manifest)
    args.db.unlink()
    with pytest.raises(PredictionBlocked,match='RETAINED_INPUT_INVALID'):
        predictor.run_prediction(args,deps)
    assert scored == []


def test_retained_cutoff_applies_before_consumer_even_with_fresh_receipt(retained_case):
    args,deps,scored,_,source=retained_case
    manifest=json.loads((args.retained_input_bundle/'manifest.json').read_bytes())
    manifest['prediction_cutoff']=(source.NOW+timedelta(seconds=60)).isoformat()
    _rebind_synthetic_manifest(args,manifest)
    deps.now=lambda:source.NOW+timedelta(seconds=61)
    with pytest.raises(PredictionBlocked,match='RETAINED_INPUT_INVALID'):
        predictor.run_prediction(args,deps)
    assert scored == []


def test_retained_cutoff_crossed_during_scoring_never_publishes_ready(retained_case):
    args,deps,scored,_,source=retained_case
    manifest=json.loads((args.retained_input_bundle/'manifest.json').read_bytes())
    manifest['prediction_cutoff']=(source.NOW+timedelta(seconds=60)).isoformat()
    _rebind_synthetic_manifest(args,manifest)
    clock={'now':source.NOW+timedelta(seconds=10)}
    deps.now=lambda:clock['now']
    scorer=deps.score_residual
    def slow_fixture(**values):
        result=scorer(**values)
        clock['now']=source.NOW+timedelta(seconds=61)
        return result
    deps.score_residual=slow_fixture
    with pytest.raises(PredictionBlocked,match='RETAINED_INPUT_INVALID'):
        predictor.run_prediction(args,deps)
    assert len(scored) == 1
    bundle=next(path for path in args.output_root.iterdir() if path.is_dir())
    assert json.loads((bundle/'result.json').read_bytes())['status']=='PREDICTION_BLOCKED'


def test_real_worker_consumer_finalizer_and_restart_share_retained_identity(retained_case, tmp_path):
    """Only OS process boundary/scorer are fixtures; production adapters execute."""
    import io
    from dataclasses import replace
    from src.operator_ui.job_store import JobInput, JobStore, OperationalIndexProvenance, Phase, resolve_audit_confirmation
    from src.operator_ui.prediction_worker import WorkerConfig, ServerChoice, WorkerRejected, run_once
    from src.operator_ui.r3_api import finalize_producer_bundle
    from src.predictor.on_demand import canonical_bytes, resolve_model, sealed_runner_set_sha256
    args,deps,scored,case,source=retained_case
    target=deps.schedule()[0]
    model=resolve_model(args.model)
    race=predictor._request_race(target,race_id=args.race_id,jump=source.JUMP)
    runners=predictor._request_expected_runners(target)
    runner_hash=sealed_runner_set_sha256(race,runners)
    index=replace(case.verified_index,races=({**target,'runner_set_sha256':runner_hash},))
    choice=ServerChoice(args.config,'manual-default',sha256_file(args.config),model.resolved,
        model.model_sha256,model.manifest_sha256,model.schema_sha256,model.model_path,model.manifest_path,model.schema_path)
    worker=WorkerConfig(Path(sys.executable),ROOT,{'latest-research':choice},args.db,args.output_root,
        (case.evidence_root,),case.protocol.root,case.evidence_root/'shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json',
        case.evidence_root,1,45,90,2,retained_input_bindings={args.race_id:{'path':str(args.retained_input_bundle),
            'manifest_sha256':args.retained_input_manifest_sha256}})
    inp=JobInput(args.race_id,source.JUMP.isoformat(),runner_hash,'latest-research',model.resolved,
        model.model_sha256,model.manifest_sha256,model.schema_sha256,'manual-default',choice.config_sha256,'receipt',
        tuple({'box':r['box_number'],'name':r['display_name'],'identity':r['identity'],
               'source_native_runner_id':r['source_native_runner_id']} for r in runners),
        OperationalIndexProvenance.from_verified_current_race_index(index),args.retained_input_manifest_sha256)
    authority=object()
    store=JobStore(tmp_path/'jobs.db',separate_from=(args.db,),verifier_authority=authority)
    confirm=lambda intent:resolve_audit_confirmation(intent,'a'*64)
    job=store.create(actor_identity='fixture',actor_level=2,operation='manual_prediction',
        idempotency_key='synthetic-retained-key-123',job_input=inp,now=deps.now(),confirm_audit=confirm)
    for phase,status,reason in ((Phase.VALIDATED,'VALID','validated'),(Phase.WAITING_FOR_CLAIM,'WAITING','ready')):
        job=store.transition(job.job_id,phase,now=deps.now(),status=status,reason=reason,confirm_audit=confirm)
    args.db.unlink()
    class Process:
        pid=99999
        def __init__(self,result):
            self.stdout=io.BytesIO(canonical_bytes(result));self.stderr=io.BytesIO(b'')
        def poll(self): return 0
        def wait(self,timeout): return 0
    def fixture_process(argv,**kwargs):
        parsed=predictor.build_parser().parse_args(list(argv[2:]))
        assert parsed.retained_input_manifest_sha256 == inp.retained_input_manifest_sha256
        return Process(predictor.run_prediction(parsed,deps))
    produced=run_once(store,job.job_id,worker,now=deps.now,confirm_audit=confirm,
        popen=fixture_process,reader=lambda **kwargs:index)
    assert produced.phase is Phase.PRODUCER_COMPLETED
    final=finalize_producer_bundle(args.output_root,store,produced,capability=authority,now=deps.now(),confirm_audit=confirm)
    assert final.phase is Phase.PREDICTION_READY
    assert len(scored)==1
    restarted=JobStore(store.path,separate_from=(args.db,),verifier_authority=authority)
    assert restarted.get(job.job_id).input.retained_input_manifest_sha256 == args.retained_input_manifest_sha256
    with pytest.raises(WorkerRejected,match='JOB_NOT_CLAIMABLE'):
        run_once(restarted,job.job_id,worker,now=deps.now,confirm_audit=confirm)


@pytest.mark.parametrize('mutation',['rejected_completion','seal_before_completed'])
def test_indexed_verifier_rejects_resealed_invalid_retention_completion(retained_case, mutation):
    import io
    import zipfile
    from src.predictor.on_demand import build_prediction_bundle_manifest_v2, canonical_bytes, prediction_bundle_index_entry
    args,deps,_,_,source=retained_case
    result=predictor.run_prediction(args,deps)
    bundle=next(path for path in args.output_root.iterdir() if path.is_dir())
    original_entry=json.loads((args.output_root/'prediction_bundle_index_v1.json').read_bytes())['entries'][0]
    assert verify_indexed_prediction_bundle(args.output_root,original_entry).result['status']=='PREDICTION_READY'
    with zipfile.ZipFile(bundle/'retained_inputs.zip') as archive:
        members={name:archive.read(name) for name in archive.namelist()}
    completion=json.loads(members['bundle/completion.json'])
    if mutation == 'rejected_completion':
        completion['status']='REJECTED'
    else:
        completion['inputs_sealed_at']=source.NOW.isoformat()
    members['bundle/completion.json']=canonical_bytes(completion)
    rewritten=io.BytesIO()
    with zipfile.ZipFile(rewritten,'w') as archive:
        for name,raw in members.items(): archive.writestr(name,raw)
    (bundle/'retained_inputs.zip').write_bytes(rewritten.getvalue())
    manifest=build_prediction_bundle_manifest_v2(bundle,prediction_id=result['prediction_id'],job_id=result['job_id'])
    raw=canonical_bytes(manifest)
    (bundle/'bundle_manifest.json').write_bytes(raw)
    entry=prediction_bundle_index_entry(bundle=bundle,result=result,manifest_raw=raw)
    # The retained manifest and request digest remain unchanged. Recomputing
    # outer producer hashes cannot qualify an invalid completion receipt.
    with pytest.raises(PredictionBlocked,match='RETAINED_INPUT_INVALID'):
        verify_indexed_prediction_bundle(args.output_root,entry)
