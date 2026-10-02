"""Derive readiness from a completed native engineering window and private joins."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import stat

from race_collection.live_freshness_contract import create_once, digest
from race_collection.incident_engineering import checked
from src.predictor.future_comparison import stamp, verify_comparison


PROOF_SCHEMA = 'incident_readiness_before_result_deadline_v1'


def file_reference(path, expected=None):
    """Hash retained bytes without decoding their potentially protected values."""
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or path.is_symlink():
        raise ValueError('readiness_file_unsafe')
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('readiness_file_unsafe')
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    after = path.stat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
    sha = value.hexdigest()
    if identity(before) != identity(after) or expected is not None and sha != expected:
        raise ValueError('readiness_file_changed')
    return {'path':str(path), 'sha256':sha}


def database_reference(path, expected=None):
    """A main-file hash is complete only with no SQLite sidecar state."""
    path = Path(path)
    def require_no_sidecars():
        if any(Path(str(path)+suffix).exists() or Path(str(path)+suffix).is_symlink()
               for suffix in ('-wal', '-shm', '-journal')):
            raise ValueError('readiness_database_sidecar_present')
    require_no_sidecars()
    reference = file_reference(path, expected)
    require_no_sidecars()
    return reference


def protected_now(now, deadline):
    current = max(now, datetime.now(timezone.utc))
    if current >= deadline:
        raise ValueError('readiness_result_authority_expired')
    return current


def closed_queue_witness(queue, job_id):
    row = queue.execute('SELECT race,job,jump,state,due,attempts FROM jobs WHERE job=?', (job_id,)).fetchone()
    if row is None or row[3] != 'CLOSED':
        raise ValueError('readiness_job_not_closed')
    events = queue.execute('SELECT at,race,status,artifact FROM events WHERE race=? ORDER BY id', (row[0],)).fetchall()
    if not events or events[-1][2] not in {'CLOSED', 'CLOSED_FROM_RETAINED_EVIDENCE',
                                         'CLOSED_FROM_RETAINED_IDENTITY_RECONCILIATION'}:
        raise ValueError('readiness_native_closure_missing')
    return {'row':list(row), 'events':[list(event) for event in events]}


def proof_binding_paths(cfg, plan, prepared, binding, result_cfg, jobs):
    """Rebuild required file bindings using metadata only, even after expiry."""
    package = Path(cfg['state_root'])/'slots/001'/(cfg['programme_id']+'-001')
    paths = {package.parent/'terminal.json', package/'plan.json', package/'started.json',
             package/'measurement.json', package/'restored.json', package/'source.tar',
             Path(result_cfg['job_store']), Path(cfg['result_binding']), Path(binding['authority'])}
    bundles = Path(result_cfg['prediction_bundles'])
    for job in jobs:
        admission = Path(plan['programme_root'])/binding['plan_sha256']/'attempts'/hashlib.sha256(job['race_id'].encode()).hexdigest()/'admission.json'
        admitted = json.loads(admission.read_bytes())
        completion = json.loads(admission.with_name('completion.json').read_bytes())
        if (admitted['job_id'] != job['job_id'] or admitted['race']['race_id'] != job['race_id']
                or admitted['plan_sha256'] != binding['plan_sha256']
                or completion['admission_sha256'] != file_reference(admission)['sha256']
                or completion['status'] != 'COMPLETE_BEFORE_CUTOFF'
                or set(completion['models']) != {'production','market','residual_box','residual_half'}
                or set(completion['models'].values()) != {'SEALED'}):
            raise ValueError('readiness_completed_chain_changed')
        entry = completion['bundle_entry']
        directory = bundles/entry['directory']
        manifest_path = directory/'bundle_manifest.json'
        file_reference(manifest_path, entry['manifest_sha256'])
        manifest = json.loads(manifest_path.read_bytes())
        paths.update((admission, admission.with_name('completion.json'), manifest_path))
        paths.update(directory/name for name in manifest['files'])
        for event in job['queue']['events']:
            if event[3] is not None:
                artifact = Path(event[3])
                if not artifact.is_relative_to(Path(result_cfg['state_root'])):
                    raise ValueError('readiness_result_artifact_unbound')
                paths.update(path for path in artifact.rglob('*') if path.is_file())
    return {str(path) for path in paths}


def seal_structural_proof(study_cfg, reference, now, cfg, plan, package, claim,
                          prepared, launch, binding, result_cfg, result_authority, result_reference,
                          witnesses, evidence):
    """Called only after the native result reader has verified before expiry."""
    from race_collection.incident_comparison import result_deadline
    deadline = result_deadline(plan)
    if now >= deadline:
        raise ValueError('readiness_result_authority_expired')
    files = {}
    def retain(path, expected=None):
        ref = file_reference(path, expected)
        files[ref['path']] = ref
    for path in (claim/'terminal.json', package/'plan.json', package/'started.json',
                 package/'measurement.json', package/'restored.json'):
        retain(path)
    retain(package/'source.tar', prepared['source_archive_sha256'])
    retain(Path(result_cfg['job_store']))
    retain(Path(cfg['result_binding']))
    retain(Path(binding['authority']), binding['authority_sha256'])
    bundles = Path(result_cfg['prediction_bundles'])
    jobs = []
    with sqlite3.connect((Path(result_cfg['state_root'])/'queue.sqlite3').as_uri()+'?mode=ro', uri=True) as queue:
        for job, bundle, admission, result in witnesses:
            retain(admission)
            retain(admission.with_name('completion.json'))
            directory = bundles / bundle.directory
            retain(directory/'bundle_manifest.json', bundle.index_entry['manifest_sha256'])
            for name, entry in bundle.manifest['files'].items():
                retain(directory/name, entry['sha256'])
            witness = closed_queue_witness(queue, job.job_id)
            if witness['row'][0] != job.input.race_id:
                raise ValueError('readiness_closed_identity_changed')
            # Hash the original retained collector/repair artifacts as opaque
            # bytes. None of their protected values enter this structural proof.
            for event in witness['events']:
                if event[3] is None:
                    continue
                artifact = Path(event[3])
                if not artifact.is_relative_to(Path(result_cfg['state_root'])):
                    raise ValueError('readiness_result_artifact_unbound')
                for path in sorted(artifact.rglob('*')):
                    if path.is_file():
                        retain(path)
            jobs.append({'job_id':job.job_id, 'race_id':job.input.race_id,
                         'queue':witness, 'evidence_sha256':result['evidence_sha256']})
    if set(files) != proof_binding_paths(cfg, plan, prepared, binding, result_cfg, jobs):
        raise ValueError('readiness_proof_incomplete')
    protected_now(now, deadline)
    final_database = database_reference(Path(result_authority['result_database']), result_reference['sha256'])
    proof = {'schema_version':PROOF_SCHEMA,
             'result_deadline':deadline.isoformat(), 'study_config_sha256':digest(study_cfg),
             'schedule':reference, 'source_commit':cfg['source_commit'],
             'launch_sha256':digest(launch), 'files':sorted(files.values(), key=lambda row:row['path']),
             'result_database':final_database,
             'jobs':sorted(jobs,key=lambda row:row['job_id']), 'evidence':evidence,
             'outcomes_released':False, 'study_enrolment':False}
    if len(json.dumps(proof).encode()) > 4*1024*1024:
        raise ValueError('readiness_proof_oversized')
    # All retained-file hashing must finish under authority. Never backdate a
    # proof using the clock captured before a slow hash or native verification.
    proof['verified_at'] = protected_now(now, deadline).isoformat()
    path = Path(cfg['state_root'])/'readiness-proofs'/(digest(proof)+'.json')
    if path.exists():
        if file_reference(path)['sha256'] != digest(proof):
            raise ValueError('readiness_proof_changed')
    else:
        protected_now(now, deadline)
        create_once(path, proof)
        path.chmod(0o400)
    return {**evidence, 'structural_proof':file_reference(path)}


def verify_structural_proof(study_cfg, reference, now, cfg, plan, prepared, launch, binding, result_cfg, result_authority):
    """After expiry, verify only metadata and byte hashes plus native closure.

    No forecast replay, result reader or protected JSON decoding is permitted.
    """
    from race_collection.incident_comparison import result_deadline
    deadline = result_deadline(plan)
    root = Path(result_cfg['state_root'])
    closure = json.loads((root/'closure/closure.json').read_bytes())
    if (closure['status'] != 'RESULT_CLOSURE_SEALED_NOT_EVALUATED'
            or closure['plan_sha256'] != binding['plan_sha256']
            or closure['result_authority_sha256'] != binding['authority_sha256']
            or stamp(closure['closure_cutoff']) != deadline
            or not deadline <= stamp(closure['sealed_at']) <= now
            or closure.get('target_values_decoded') is not False
            or closure['result_database'] != str(root/'closure/official-results.sqlite3')):
        return None
    final = database_reference(root/'closure/official-results.sqlite3', closure['result_database_sha256'])
    live = database_reference(Path(result_authority['result_database']), final['sha256'])
    paths = sorted((Path(cfg['state_root'])/'readiness-proofs').glob('*.json'), reverse=True)
    if len(paths) > 512:
        return None
    for path in paths:
        try:
            if path.stat().st_size > 4*1024*1024:
                continue
            ref = file_reference(path, path.stem)
            proof = json.loads(path.read_bytes())
            if (proof['schema_version'] != PROOF_SCHEMA or proof['schedule'] != reference
                    or proof['study_config_sha256'] != digest(study_cfg)
                    or proof['source_commit'] != cfg['source_commit']
                    or proof['launch_sha256'] != digest(launch)
                    or stamp(proof['result_deadline']) != deadline
                    or not stamp(prepared['ends_at']) <= stamp(proof['verified_at']) < deadline
                    or proof['result_database'] != live
                    or proof['outcomes_released'] is not False or proof['study_enrolment'] is not False):
                continue
            identities = {(row['job_id'],row['race_id']) for row in proof['jobs']}
            evidence = proof['evidence']
            if (len(identities) != len(proof['jobs']) or len(identities) < 3
                    or len({row[0] for row in identities}) != len(identities)
                    or len({row[1] for row in identities}) != len(identities)
                    or evidence['status'] != 'NATIVE_ENGINEERING_CHAIN_VERIFIED'
                    or evidence['schedule'] != reference
                    or evidence['authority_sha256'] != cfg['incident_authority']['sha256']
                    or evidence['closed_results'] != len(identities)
                    or type(evidence['verified_predictions']) is not int
                    or not len(identities) <= evidence['verified_predictions'] <= checked(cfg['incident_authority'])['max_capture_attempts_per_window']
                    or evidence['source_commit'] != cfg['source_commit']
                    or evidence['terminal_sha256'] != file_reference(Path(cfg['state_root'])/'slots/001/terminal.json')['sha256']
                    or evidence['outcomes_released'] is not False
                    or evidence['study_enrolment'] is not False
                    or evidence['original_failure_preserved'] is not True):
                continue
            if {file['path'] for file in proof['files']} != proof_binding_paths(
                    cfg, plan, prepared, binding, result_cfg, proof['jobs']):
                continue
            for file in proof['files']:
                file_reference(file['path'], file['sha256'])
            with sqlite3.connect((root/'queue.sqlite3').as_uri()+'?mode=ro', uri=True) as queue:
                if any(closed_queue_witness(queue, row['job_id']) != row['queue'] for row in proof['jobs']):
                    continue
            database_reference(root/'closure/official-results.sqlite3', final['sha256'])
            database_reference(Path(result_authority['result_database']), live['sha256'])
            return {**proof['evidence'], 'structural_proof':ref,
                    'result_closure':file_reference(root/'closure/closure.json'),
                    'post_deadline_hash_verification':True}
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
            continue
    return None


def verified_incident_acceptance(study_cfg, reference, now, *, seal=False):
    """Verify live within authority, or a sealed structural proof after expiry.

    Only the native scheduler passes ``seal=True`` to publish append-only proofs.
    Failed windows and missing proofs/results remain held.
    """
    try:
        cfg = checked(reference)
        if (cfg['status'] != 'AUTHORIZED_INCIDENT_SCHEDULE'
                or cfg['source_commit'] != study_cfg['source_commit']
                or cfg['campaign_root'] != study_cfg['campaign_root']):
            return None
        from race_collection.incident_comparison import result_deadline, validate_incident_plan
        from src.predictor.future_comparison import load_plan
        plan, _ = load_plan(Path(cfg['comparison_plan']), cfg['comparison_plan_sha256'])
        authority = validate_incident_plan(plan)
        if authority['study_plan'] != {'path':study_cfg['comparison_plan'], 'sha256':study_cfg['comparison_plan_sha256']}:
            return None
        slot = next(row for row in authority['slots'] if row['id'] == cfg['incident_slot'])
        if cfg['slots'] != [slot['starts_at']] or now < stamp(slot['ends_at']):
            return None
        claim = Path(cfg['state_root']) / 'slots/001'
        package = claim / (cfg['programme_id']+'-001')
        read = lambda path: json.loads(path.read_bytes())
        terminal = read(claim/'terminal.json')
        prepared = read(package/'plan.json')
        started = read(package/'started.json')
        measured = read(package/'measurement.json')
        restored = read(package/'restored.json')
        if (terminal['status'] != 'COMPLETED' or (package/'failure.json').exists()
                or started['plan_sha256'] != digest(prepared)
                or prepared['commit'] != cfg['source_commit']
                or prepared['incident_authority'] != cfg['incident_authority']
                or prepared['incident_slot'] != cfg['incident_slot']
                or prepared['frozen_comparison'] != {'path':cfg['comparison_plan'], 'sha256':cfg['comparison_plan_sha256']}
                or prepared['starts_at'] != slot['starts_at'] or prepared['ends_at'] != slot['ends_at']
                or measured['status'] != 'REHEARSAL_MEASURED_NOT_RELEASED'
                or measured['completed_cycles']['full'] < 3 or measured['completed_cycles']['odds'] < 6
                or measured['capture_count'] < 3 or measured['maximum_conservative_source_age'] >= 270
                or measured['logical_requests'] > authority.get('max_python_requests_per_window', authority['max_prediction_logical_requests_per_window'])
                or restored['sportsbet_hold'] is not False
                or restored['status'] not in {'RESTORED', 'RESTORED_COLLECTOR_TRIGGERS_HELD'}):
            return None
        ledger = read(Path(cfg['campaign_root'])/'ledger.json')
        launch = ledger['launches'][prepared['rehearsal_id']]
        if (not launch['closed_at'] or launch['incident_authority_sha256'] != cfg['incident_authority']['sha256']
                or launch['incident_slot'] != cfg['incident_slot']
                or launch['charged_seconds'] > 7260):
            return None
        from src.predictor.comparison_result_runtime import load_runtime
        binding = read(Path(cfg['result_binding']))
        deadline = result_deadline(plan)
        now = max(now, datetime.now(timezone.utc))
        expired = now >= deadline
        _, result_authority, result_cfg = load_runtime(binding, now=now, allow_closure=expired)
        if binding['plan_sha256'] != cfg['comparison_plan_sha256']:
            return None
        now = max(now, datetime.now(timezone.utc))
        if now >= deadline:
            return verify_structural_proof(study_cfg, reference, now, cfg, plan, prepared, launch,
                                           binding, result_cfg, result_authority)
        # Live proof sealing requires quiescence. Once the native incident has
        # closed, a later authorised pilot lease cannot invalidate its history;
        # actual admissions still enforce their own shared owner/lease controls.
        if any(not row.get('closed_at') for row in ledger['launches'].values()):
            return None
        from src.operator_ui.job_store import JobStore
        from src.operator_ui.r3_api import build_verified_bundle_reader
        from src.predictor.comparison_results import ComparisonResultSource
        protected_now(now, deadline)
        store = JobStore(Path(result_cfg['job_store']), readonly=True)
        protected_now(now, deadline)
        jobs = {job.job_id:job for job in store.recorded_jobs()}
        bundles = Path(result_cfg['prediction_bundles'])
        protected_now(now, deadline)
        reader = build_verified_bundle_reader(bundles, store)
        protected_now(now, deadline)
        results = ComparisonResultSource(Path(result_authority['result_database']))
        protected_now(now, deadline)
        result_reference = database_reference(Path(result_authority['result_database'])) if seal else None
        verified = closed = 0
        witnesses = []
        identities = set()
        job_ids = set()
        with sqlite3.connect((Path(result_cfg['state_root'])/'queue.sqlite3').as_uri()+'?mode=ro', uri=True) as queue:
            for admission in (Path(plan['programme_root'])/binding['plan_sha256']/'attempts').glob('*/admission.json'):
                admitted = read(admission)
                race_id = admitted['race']['race_id']
                job_id = admitted['job_id']
                if (admission.parent.name != hashlib.sha256(race_id.encode()).hexdigest()
                        or race_id in identities or job_id in job_ids):
                    return None
                identities.add(race_id)
                job_ids.add(job_id)
                protected_now(now, deadline)
                value = verify_comparison(bundles, admission, expected_plan_sha256=binding['plan_sha256'])
                if not value.get('engineering_evidence') or value['future_race_evidence']:
                    continue
                verified += 1
                job = jobs[job_id]
                if job.input.race_id != race_id:
                    return None
                if queue.execute("SELECT 1 FROM jobs WHERE job=? AND state='CLOSED'",(job.job_id,)).fetchone():
                    protected_now(now, deadline)
                    bundle = reader(job)
                    result_now = protected_now(now, deadline)
                    result = results.read(job, bundle, now=result_now)
                    if result['state'] == 'RESULT_AVAILABLE':
                        closed += 1
                        witnesses.append((job, bundle, admission, result))
        if closed < 3:
            return None
        protected_now(now, deadline)
        evidence = {'status':'NATIVE_ENGINEERING_CHAIN_VERIFIED', 'schedule':reference,
                'authority_sha256':cfg['incident_authority']['sha256'], 'source_commit':cfg['source_commit'],
                'terminal_sha256':hashlib.sha256((claim/'terminal.json').read_bytes()).hexdigest(),
                'verified_predictions':verified, 'closed_results':closed, 'outcomes_released':False,
                'study_enrolment':False, 'original_failure_preserved':True}
        if seal:
            return seal_structural_proof(study_cfg, reference, now, cfg, plan, package, claim,
                prepared, launch, binding, result_cfg, result_authority, result_reference, witnesses, evidence)
        return evidence
    except (OSError, ValueError, KeyError, TypeError, StopIteration, sqlite3.Error):
        return None
