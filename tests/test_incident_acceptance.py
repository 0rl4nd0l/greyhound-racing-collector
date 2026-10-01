"""Independent structural gate tests; invented files only, no outcome decoding.

Native plan/authority/hash/queue and bridge code run. The deep forecast and
official-result readers are replaced by structural witnesses at their seams.
"""
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from tests.test_incident_comparison import case
from tests.fixtures.incident_engineering_case import put
from race_collection.live_freshness_contract import digest
from race_collection import incident_acceptance as acceptance


@pytest.fixture
def readiness(tmp_path, monkeypatch):
    clock = SimpleNamespace(value=datetime.fromisoformat('2026-10-01T18:12:00+10:00'))
    class FixtureDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.value.astimezone(tz) if tz else clock.value.replace(tzinfo=None)
    monkeypatch.setattr(acceptance, 'datetime', FixtureDatetime)
    authority, authority_ref, plan, plan_ref = case(tmp_path)
    campaign = tmp_path / 'campaign'
    cfg = {'status':'AUTHORIZED_INCIDENT_SCHEDULE', 'source_commit':'a'*40,
           'campaign_root':str(campaign), 'comparison_plan':plan_ref['path'],
           'comparison_plan_sha256':plan_ref['sha256'],
           'incident_authority':authority_ref, 'incident_slot':'001',
           'slots':[authority['slots'][0]['starts_at']],
           'state_root':str(Path(authority['state_root'])/'windows/001'),
           'programme_id':'invented-window', 'result_binding':str(tmp_path/'binding.json')}
    study = {'source_commit':cfg['source_commit'], 'campaign_root':str(campaign),
             'comparison_plan':authority['study_plan']['path'],
             'comparison_plan_sha256':authority['study_plan']['sha256']}
    reference = put(tmp_path/'schedule.json', cfg)
    claim = Path(cfg['state_root'])/'slots/001'
    package = claim/(cfg['programme_id']+'-001')
    prepared = {'commit':cfg['source_commit'], 'incident_authority':authority_ref,
                'incident_slot':'001', 'starts_at':authority['slots'][0]['starts_at'],
                'ends_at':authority['slots'][0]['ends_at'], 'rehearsal_id':'invented-launch',
                'frozen_comparison':plan_ref}
    package.mkdir(parents=True)
    (package/'source.tar').write_bytes(b'invented immutable source archive')
    prepared['source_archive_sha256']=hashlib.sha256((package/'source.tar').read_bytes()).hexdigest()
    put(package/'plan.json', prepared)
    put(package/'started.json', {'plan_sha256':digest(prepared)})
    put(claim/'terminal.json', {'status':'COMPLETED'})
    put(package/'measurement.json', {'status':'REHEARSAL_MEASURED_NOT_RELEASED',
        'completed_cycles':{'full':3,'odds':6}, 'capture_count':3,
        'maximum_conservative_source_age':100, 'logical_requests':100})
    put(package/'restored.json', {'sportsbet_hold':False,'status':'RESTORED'})
    put(campaign/'ledger.json', {'source_holds':[], 'launches':{'invented-launch':{
        'closed_at':'2026-10-01T17:41:00+10:00','charged_seconds':5400,
        'incident_authority_sha256':authority_ref['sha256'],'incident_slot':'001'}}})
    result_cfg = {'state_root':str(Path(authority['result_root'])/'001'),
                  'job_store':str(Path(authority['prediction_root'])/'jobs.sqlite3'),
                  'prediction_bundles':str(Path(authority['prediction_root'])/'bundles')}
    result_authority = {'result_database':str(Path(result_cfg['state_root'])/'official.sqlite3'),
        'status':'AUTHORIZED_ENGINEERING_MACHINE_RESULT_RETENTION', 'plan_sha256':plan_ref['sha256'],
        'owner':'invented-result-worker','authority_reference':'SYNTHETIC','source_budget_reference':'SYNTHETIC',
        'human_outcome_access':False,'issued_at':authority['issued_at'],
        'incident_authority':authority_ref,'incident_slot':'001'}
    result_ref=put(tmp_path/'result-authority.json',result_authority)
    binding = {'plan':plan_ref['path'],'plan_sha256':plan_ref['sha256'],
               'authority':result_ref['path'],'authority_sha256':result_ref['sha256']}
    put(Path(cfg['result_binding']), binding)
    job_store=Path(result_cfg['job_store']);job_store.parent.mkdir(parents=True)
    job_store.write_bytes(b'invented immutable job-store witness')
    import src.predictor.comparison_result_runtime as runtime
    monkeypatch.setattr(runtime, 'load_runtime', lambda *a,**kw:(plan,result_authority,result_cfg))
    jobs = [SimpleNamespace(job_id=f'job-{number}',input=SimpleNamespace(race_id=f'invented-race-{number}'))
            for number in range(3)]
    import src.operator_ui.job_store as store
    monkeypatch.setattr(store, 'JobStore', lambda *a,**kw:SimpleNamespace(recorded_jobs=lambda:jobs))
    import src.operator_ui.r3_api as api
    bundles={}
    monkeypatch.setattr(api, 'build_verified_bundle_reader', lambda *a:lambda job:bundles[job.job_id])
    import src.predictor.comparison_results as results
    monkeypatch.setattr(results, 'ComparisonResultSource', lambda *a:SimpleNamespace(
        read=lambda *a,**kw:{'state':'RESULT_AVAILABLE','evidence_sha256':'e'*64}))
    monkeypatch.setattr(acceptance, 'verify_comparison', lambda *a,**kw:{
        'engineering_evidence':True, 'future_race_evidence':False})
    admissions = Path(plan['programme_root'])/plan_ref['sha256']/'attempts'
    for number, job in enumerate(jobs):
        race = f'invented-race-{number}'
        directory=Path(result_cfg['prediction_bundles'])/job.job_id
        data_ref=put(directory/'invented-forecast.json',{'all_values':'invented structural witness'})
        manifest={'files':{'invented-forecast.json':{'sha256':data_ref['sha256']}}}
        manifest_ref=put(directory/'bundle_manifest.json',manifest)
        entry={'directory':job.job_id,'manifest_sha256':manifest_ref['sha256']}
        bundles[job.job_id]=SimpleNamespace(directory=job.job_id,index_entry=entry,manifest=manifest)
        admission=put(admissions/hashlib.sha256(race.encode()).hexdigest()/'admission.json',
            {'job_id':job.job_id,'race':{'race_id':race},'plan_sha256':plan_ref['sha256']})
        put(Path(admission['path']).with_name('completion.json'),{'admission_sha256':admission['sha256'],
            'status':'COMPLETE_BEFORE_CUTOFF','models':{name:'SEALED' for name in
            ('production','market','residual_box','residual_half')},'bundle_entry':entry})
    Path(result_cfg['state_root']).mkdir(parents=True)
    with sqlite3.connect(Path(result_cfg['state_root'])/'queue.sqlite3') as queue:
        queue.execute('CREATE TABLE jobs(race TEXT,job TEXT PRIMARY KEY,jump TEXT,state TEXT,due TEXT,attempts INTEGER)')
        queue.execute('CREATE TABLE events(id INTEGER PRIMARY KEY,at TEXT,race TEXT,status TEXT,artifact TEXT)')
        queue.executemany('INSERT INTO jobs VALUES(?,?,?,?,?,?)', [(job.input.race_id,job.job_id,
            '2026-10-01T17:00:00+10:00','CLOSED',None,1) for job in jobs])
        for job in jobs:
            artifact=Path(result_cfg['state_root'])/'attempts'/job.job_id
            put(artifact/'response.json',{'synthetic':True})
            queue.execute('INSERT INTO events(at,race,status,artifact) VALUES(?,?,?,?)',
                ('2026-10-01T17:20:00+10:00',job.input.race_id,'CLOSED',str(artifact)))
    with sqlite3.connect(result_authority['result_database']) as database:
        database.execute('CREATE TABLE invented_private_results(id INTEGER PRIMARY KEY,payload TEXT)')
        database.execute("INSERT INTO invented_private_results(payload) VALUES('invented private bytes')")
    return SimpleNamespace(cfg=cfg, study=study, reference=reference, package=package,
        claim=claim, campaign=campaign, admissions=admissions, result_state=Path(result_cfg['state_root']), jobs=jobs,
        result_cfg=result_cfg,result_authority=result_authority,plan=plan,binding=binding,
        now=clock.value,clock=clock)


def verify(value):
    return acceptance.verified_incident_acceptance(value.study,value.reference,value.now)


def test_three_distinct_closed_structural_witnesses_qualify(readiness):
    assert verify(readiness)['closed_results']==3


def test_pending_official_result_cannot_count_as_closed(readiness):
    with sqlite3.connect(readiness.result_state/'queue.sqlite3') as queue:
        queue.execute("UPDATE jobs SET state='PENDING' WHERE job='job-2'")
    assert verify(readiness) is None


def test_mismatched_job_identity_cannot_count_as_closed(readiness):
    readiness.jobs[2].input.race_id='different-invented-race'
    assert verify(readiness) is None


def test_scientific_evidence_cannot_substitute_for_engineering(readiness,monkeypatch):
    monkeypatch.setattr(acceptance,'verify_comparison',lambda *a,**kw:{
        'engineering_evidence':False,'future_race_evidence':True})
    assert verify(readiness) is None


@pytest.mark.parametrize('change',['terminal_failure','failure_marker','source_changed','unrestored','late_source'])
def test_failure_and_pinning_negatives_stay_held(readiness, change):
    value=readiness
    if change=='terminal_failure': put(value.claim/'terminal.json',{'status':'FAILED_RESTORED'})
    if change=='failure_marker': put(value.package/'failure.json',{'status':'FAILED'})
    if change=='source_changed': value.study['source_commit']='b'*40
    if change=='unrestored': put(value.package/'restored.json',{'sportsbet_hold':True,'status':'RESTORED'})
    if change=='late_source':
        path=value.package/'measurement.json'; record=json.loads(path.read_bytes())
        record['maximum_conservative_source_age']=270;put(path,record)
    assert verify(value) is None


def test_duplicate_admission_cannot_turn_one_closed_race_into_three(readiness):
    paths=sorted(readiness.admissions.glob('*/admission.json'))
    first=json.loads(paths[0].read_bytes())
    for path in paths[1:]: put(path,first)
    assert verify(readiness) is None


def test_different_prepared_comparison_plan_cannot_clear_gate(readiness):
    path=readiness.package/'plan.json'; prepared=json.loads(path.read_bytes())
    prepared['frozen_comparison']={'path':'/tmp/not-the-authorized-plan','sha256':'f'*64}
    put(path,prepared)
    put(readiness.package/'started.json',{'plan_sha256':digest(prepared)})
    assert verify(readiness) is None


def test_failed_incident_gate_cannot_fall_through_to_legacy_one_result_canary(tmp_path,monkeypatch):
    from scripts import run_comparison_schedule as schedule
    monkeypatch.setattr(acceptance,'verified_incident_acceptance',lambda *a,**kw:None)
    root=tmp_path/'schedule'; predictions=tmp_path/'predictions'; results=tmp_path/'results'
    cfg={'incident_acceptance':{'path':'/invented','sha256':'a'*64},
         'programme_id':'invented','prediction_root':str(predictions),
         'comparison_plan_sha256':'b'*64,'slots':['2026-10-01T12:00:00+10:00']}
    first=root/'slots/001'
    put(first/'terminal.json',{'status':'COMPLETED'})
    plan_path=first/'invented-001/plan.json'
    race='invented-race'
    put(predictions/'dispatches/one.json',{'plan':str(plan_path),'race_id':race})
    put(predictions/'races'/hashlib.sha256(race.encode()).hexdigest()/'terminal.json',
        {'status':'PREDICTION_READY','job_id':'one'})
    results.mkdir()
    with sqlite3.connect(results/'queue.sqlite3') as db:
        db.execute('CREATE TABLE jobs(job TEXT PRIMARY KEY,state TEXT)')
        db.execute("INSERT INTO jobs VALUES('one','CLOSED')")
    assert schedule.verify_canary(cfg,{'state_root':str(results)},root,
        datetime.fromisoformat('2026-10-02T12:00:00+10:00')) is False
    assert not (root/'canary.json').exists()


def seal_readiness(value):
    return acceptance.verified_incident_acceptance(value.study,value.reference,value.now,seal=True)


def close_results(value):
    from scripts.seal_comparison_result_closure import seal
    return seal(Path(value.cfg['result_binding']),value.result_state/'closure',
                now=datetime.fromisoformat('2026-10-02T12:00:00+10:00'))


def forbid_deep_decoding(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('post-deadline native forecast/result decoding attempted')
    monkeypatch.setattr(acceptance,'verify_comparison',forbidden)
    monkeypatch.setattr('src.operator_ui.job_store.JobStore',forbidden)
    monkeypatch.setattr('src.operator_ui.r3_api.build_verified_bundle_reader',forbidden)
    monkeypatch.setattr('src.predictor.comparison_results.ComparisonResultSource',forbidden)


def test_expired_authority_without_proof_stays_held_without_decoding(readiness,monkeypatch):
    close_results(readiness)
    forbid_deep_decoding(monkeypatch)
    readiness.now=datetime.fromisoformat('2026-10-02T13:00:00+10:00')
    assert seal_readiness(readiness) is None
    assert not (Path(readiness.cfg['state_root'])/'readiness-proofs').exists()


def test_native_closure_and_predeadline_proof_verify_after_expiry_without_decoding(readiness,monkeypatch):
    first=seal_readiness(readiness)
    assert first is not None
    proof_path=Path(first['structural_proof']['path'])
    original=proof_path.read_bytes()
    assert proof_path.stat().st_mode & 0o222 == 0
    close_results(readiness)
    forbid_deep_decoding(monkeypatch)
    original_connect=sqlite3.connect
    def metadata_only(path,*args,**kwargs):
        assert 'official' not in str(path), 'private result database was opened as SQL'
        return original_connect(path,*args,**kwargs)
    monkeypatch.setattr(sqlite3,'connect',metadata_only)
    readiness.now=datetime.fromisoformat('2026-10-02T13:00:00+10:00')
    final=seal_readiness(readiness)
    assert final['post_deadline_hash_verification'] is True
    assert final['closed_results']==3 and final['outcomes_released'] is False
    assert final['structural_proof']==first['structural_proof']
    assert proof_path.read_bytes()==original


def test_read_only_predeadline_verification_does_not_publish_proof(readiness):
    assert verify(readiness)['closed_results']==3
    assert not (Path(readiness.cfg['state_root'])/'readiness-proofs').exists()


@pytest.mark.parametrize('refresh',[False,True])
def test_result_database_growth_requires_refreshed_proof_before_expiry(readiness,monkeypatch,refresh):
    first=seal_readiness(readiness)
    original=Path(first['structural_proof']['path']).read_bytes()
    with sqlite3.connect(readiness.result_authority['result_database']) as database:
        database.execute("INSERT INTO invented_private_results(payload) VALUES('later invented closure')")
    if refresh:
        readiness.now+=timedelta(minutes=5)
        second=seal_readiness(readiness)
        assert second['structural_proof']!=first['structural_proof']
    close_results(readiness)
    forbid_deep_decoding(monkeypatch)
    readiness.now=datetime.fromisoformat('2026-10-02T13:00:00+10:00')
    final=verify(readiness)
    assert (final is not None) is refresh
    assert Path(first['structural_proof']['path']).read_bytes()==original


def test_result_database_growth_during_native_verification_prevents_sealing(readiness,monkeypatch):
    calls=[]
    def read(*args,**kwargs):
        calls.append(True)
        if len(calls)==1:
            with sqlite3.connect(readiness.result_authority['result_database']) as database:
                database.execute("INSERT INTO invented_private_results(payload) VALUES('concurrent invented append')")
        return {'state':'RESULT_AVAILABLE','evidence_sha256':'e'*64}
    monkeypatch.setattr('src.predictor.comparison_results.ComparisonResultSource',
                        lambda *args:SimpleNamespace(read=read))
    assert seal_readiness(readiness) is None
    assert len(calls)==3
    assert not (Path(readiness.cfg['state_root'])/'readiness-proofs').exists()


def test_stale_tick_clock_uses_postdeadline_structural_path(readiness,monkeypatch):
    sealed=seal_readiness(readiness)
    close_results(readiness)
    readiness.clock.value=datetime.fromisoformat('2026-10-02T13:00:00+10:00')
    forbid_deep_decoding(monkeypatch)
    final=seal_readiness(readiness)
    assert final['post_deadline_hash_verification'] is True
    assert final['structural_proof']==sealed['structural_proof']


@pytest.mark.parametrize('crossing',['runtime','job_store','recorded_jobs','bundle_factory',
    'result_factory','comparison','bundle_read','result_read','initial_database_hash',
    'binding_hashes','final_database_hash'])
def test_deadline_crossing_stops_all_later_deep_reads_and_proof_publication(readiness,monkeypatch,crossing):
    import src.operator_ui.job_store as stores
    import src.operator_ui.r3_api as api
    import src.predictor.comparison_results as results
    import src.predictor.comparison_result_runtime as runtime
    deadline=datetime.fromisoformat('2026-10-02T12:00:00+10:00')
    calls=[]
    def wrap(name, function):
        def call(*args,**kwargs):
            assert readiness.clock.value<deadline, f'late protected call: {name}'
            calls.append(name)
            value=function(*args,**kwargs)
            if name==crossing:
                readiness.clock.value=deadline
            return value
        return call
    old_store=stores.JobStore
    def store(*args,**kwargs):
        value=old_store(*args,**kwargs)
        value.recorded_jobs=wrap('recorded_jobs',value.recorded_jobs)
        return value
    monkeypatch.setattr(stores,'JobStore',wrap('job_store',store))
    old_builder=api.build_verified_bundle_reader
    monkeypatch.setattr(api,'build_verified_bundle_reader',wrap('bundle_factory',
        lambda *args:wrap('bundle_read',old_builder(*args))))
    old_results=results.ComparisonResultSource
    def result_factory(*args):
        value=old_results(*args)
        value.read=wrap('result_read',value.read)
        return value
    monkeypatch.setattr(results,'ComparisonResultSource',wrap('result_factory',result_factory))
    monkeypatch.setattr(runtime,'load_runtime',wrap('runtime',runtime.load_runtime))
    monkeypatch.setattr(acceptance,'verify_comparison',wrap('comparison',acceptance.verify_comparison))
    monkeypatch.setattr(acceptance,'proof_binding_paths',wrap('binding_hashes',acceptance.proof_binding_paths))
    old_database=acceptance.database_reference
    hashes=[]
    def database(*args,**kwargs):
        name='initial_database_hash' if not hashes else 'final_database_hash'
        hashes.append(name)
        return wrap(name,old_database)(*args,**kwargs)
    monkeypatch.setattr(acceptance,'database_reference',database)
    assert seal_readiness(readiness) is None
    assert crossing in calls
    assert not (Path(readiness.cfg['state_root'])/'readiness-proofs').exists()


@pytest.mark.parametrize('suffix',['-wal','-shm','-journal'])
@pytest.mark.parametrize('stage',['before','after_initial_hash','after_final_hash',
                                'postdeadline_live','postdeadline_snapshot'])
def test_result_sidecars_prevent_incomplete_database_proof(readiness,monkeypatch,suffix,stage):
    database=Path(readiness.result_authority['result_database'])
    if stage.startswith('postdeadline'):
        assert seal_readiness(readiness) is not None
        close_results(readiness)
        forbid_deep_decoding(monkeypatch)
        readiness.now=datetime.fromisoformat('2026-10-02T13:00:00+10:00')
        if stage=='postdeadline_snapshot':
            database=readiness.result_state/'closure/official-results.sqlite3'
        Path(str(database)+suffix).write_bytes(b'invented pending SQLite state')
    elif stage=='before':
        Path(str(database)+suffix).write_bytes(b'invented pending SQLite state')
    else:
        old_reference=acceptance.file_reference
        calls=[]
        def reference(path,*args,**kwargs):
            value=old_reference(path,*args,**kwargs)
            if Path(path)==database:
                calls.append(True)
                if len(calls)==(1 if stage=='after_initial_hash' else 2):
                    Path(str(database)+suffix).write_bytes(b'invented concurrent SQLite state')
            return value
        monkeypatch.setattr(acceptance,'file_reference',reference)
    assert seal_readiness(readiness) is None


@pytest.mark.parametrize('lease',['later_pilot','incident_reopened'])
def test_postdeadline_idle_scheduler_accepts_later_lease_but_not_reopened_incident(readiness,monkeypatch,lease):
    from scripts import run_comparison_schedule as schedule
    root=Path(readiness.cfg['state_root'])/'invented-study-schedule'
    readiness.study.update(state_root=str(root),incident_acceptance=readiness.reference,
                           slots=['2099-10-01T13:00:00+10:00'])
    assert seal_readiness(readiness) is not None
    close_results(readiness)
    path=readiness.campaign/'ledger.json'
    ledger=json.loads(path.read_bytes())
    if lease=='later_pilot':
        ledger['launches']['later-authorised-pilot']={'closed_at':None}
    else:
        ledger['launches']['invented-launch']['closed_at']=None
    put(path,ledger)
    readiness.clock.value=datetime.fromisoformat('2026-10-03T13:00:00+10:00')
    forbid_deep_decoding(monkeypatch)
    monkeypatch.setattr(schedule,'datetime',acceptance.datetime)
    monkeypatch.setattr(schedule,'load_config',lambda path:(readiness.study,{'ends_at':'2099-10-02T00:00:00+10:00'}))
    result=schedule.tick(root/'unused-config.json')
    assert result['status']==('NO_SLOT_DUE' if lease=='later_pilot' else 'CANARY_NOT_VERIFIED')


def test_live_proof_sealing_still_requires_shared_lease_quiescence(readiness):
    path=readiness.campaign/'ledger.json'
    ledger=json.loads(path.read_bytes())
    ledger['launches']['another-active-owner']={'closed_at':None}
    put(path,ledger)
    assert seal_readiness(readiness) is None


@pytest.mark.parametrize('damage',['no_closure','closure_bytes','source_archive','forecast_bytes',
                                  'completion','queue_state','proof_bytes','missing_bindings'])
def test_postdeadline_proof_rejects_changed_or_incomplete_bindings(readiness,monkeypatch,damage):
    sealed=seal_readiness(readiness)
    close_results(readiness)
    proof_path=Path(sealed['structural_proof']['path'])
    if damage=='no_closure':
        (readiness.result_state/'closure/closure.json').unlink()
    elif damage=='closure_bytes':
        (readiness.result_state/'closure/official-results.sqlite3').chmod(0o600)
        with (readiness.result_state/'closure/official-results.sqlite3').open('ab') as stream:
            stream.write(b'changed')
    elif damage=='source_archive':
        (readiness.package/'source.tar').write_bytes(b'changed archive')
    elif damage=='forecast_bytes':
        path=Path(readiness.result_cfg['prediction_bundles'])/'job-0/invented-forecast.json'
        put(path,{'changed':True})
    elif damage=='completion':
        path=sorted(readiness.admissions.glob('*/completion.json'))[0]
        value=json.loads(path.read_bytes());value['models']['market']='FAILED';put(path,value)
    elif damage=='queue_state':
        with sqlite3.connect(readiness.result_state/'queue.sqlite3') as queue:
            queue.execute("UPDATE jobs SET state='PENDING' WHERE job='job-0'")
    elif damage=='proof_bytes':
        proof_path.chmod(0o600);proof_path.write_bytes(b'{}')
    else:
        proof=json.loads(proof_path.read_bytes());proof['files']=[]
        proof_path.unlink()
        from race_collection.live_freshness_contract import create_once
        create_once(proof_path.parent/(digest(proof)+'.json'),proof)
    forbid_deep_decoding(monkeypatch)
    readiness.now=datetime.fromisoformat('2026-10-02T13:00:00+10:00')
    assert verify(readiness) is None


@pytest.mark.parametrize('accepted',[False,True])
@pytest.mark.parametrize('paused',[False,True])
def test_scheduler_refreshes_incident_proof_despite_existing_canary(tmp_path,monkeypatch,accepted,paused):
    from scripts import run_comparison_schedule as schedule
    root=tmp_path/'scheduler'
    put(root/'canary.json',{'manual_green_is_not_authority':True})
    if paused:
        (root/'PAUSE_ADMISSIONS').write_text('admissions remain held')
    cfg={'state_root':str(root),'incident_acceptance':{'path':'/invented','sha256':'a'*64},
         'slots':['2099-10-01T13:00:00+10:00']}
    monkeypatch.setattr(schedule,'load_config',lambda path:(cfg,{'ends_at':'2099-10-02T00:00:00+10:00'}))
    calls=[]
    def verify(*args):
        calls.append(args)
        return accepted
    monkeypatch.setattr(schedule,'verify_canary',verify)
    value=schedule.tick(tmp_path/'unused-config.json')
    assert len(calls)==1
    assert value['status']==('ADMISSIONS_PAUSED' if paused else 'NO_SLOT_DUE' if accepted else 'CANARY_NOT_VERIFIED')
    assert not (root/'slots').exists()
