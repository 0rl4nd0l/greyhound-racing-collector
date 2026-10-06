"""Append-only study references to original engineering seals; no acquisition/scoring."""
from datetime import date, timedelta
import fcntl
import hashlib
import json
import os
from contextvars import ContextVar
from pathlib import Path

from race_collection.retained_study_readiness import verify_retained_readiness
from race_collection import retained_study_reservations as reservations
from src.predictor.future_comparison import load_plan, stamp

MODELS = {'market', 'production', 'residual_box', 'residual_half'}
BUDGET = ContextVar('observer_budget', default=None)
MODEL_FILES = {'model/model.json', 'model/manifest.json', 'comparison/registry.json'}
REQUIRED_INPUTS = {'source/capture.json', 'retained_inputs.zip', 'features/sealed_history.db',
    'features/history_seal.json', 'features/sealed/implementation_file_manifest.json',
    'protocol/collector_exact_receipt.json', 'odds_receipt.json', 'request.json'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def raw(path, maximum=2*1024*1024):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or not path.is_file():
        raise ValueError('observer_path_unsafe')
    stat = path.stat()
    if stat.st_nlink != 1 or stat.st_size > maximum:
        raise ValueError('observer_file_unsafe')
    budget = BUDGET.get()
    if budget is not None:
        budget[0] -= 1; budget[1] -= stat.st_size
        if min(budget) < 0:
            raise RuntimeError('observer_scan_allowance_exhausted')
    value = path.read_bytes()
    after = path.stat()
    if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('observer_file_changed')
    return value


def opaque_hash(path, expected_bytes):
    path = Path(path)
    if (not path.is_absolute() or path.resolve() != path or not path.is_file()
            or type(expected_bytes) is not int or expected_bytes < 0):
        raise ValueError('NATIVE_DECLARED_FILE_UNSAFE')
    before = path.stat()
    if before.st_nlink != 1 or before.st_size != expected_bytes:
        raise ValueError('NATIVE_DECLARED_FILE_CHANGED')
    budget = BUDGET.get()
    if budget is not None:
        budget[0] -= 1; budget[1] -= expected_bytes
        if min(budget) < 0:
            raise RuntimeError('observer_scan_allowance_exhausted')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('NATIVE_DECLARED_FILE_CHANGED')
    return digest.hexdigest()


def verify_bundle_files(bundle, manifest):
    if not REQUIRED_INPUTS.issubset(manifest['files']):
        raise ValueError('NATIVE_INPUT_CHAIN_INCOMPLETE')
    for name, declared in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError('NATIVE_DECLARED_FILE_UNSAFE')
        if opaque_hash(bundle/relative, declared['bytes']) != declared['sha256']:
            raise ValueError('NATIVE_DECLARED_FILE_CHANGED')


def reference(path):
    return {'path': str(path), 'sha256': hashlib.sha256(raw(path)).hexdigest()}


def checked(ref):
    if set(ref) != {'path', 'sha256'}:
        raise ValueError('observer_reference_invalid')
    value = raw(ref['path'])
    if hashlib.sha256(value).hexdigest() != ref['sha256']:
        raise ValueError('observer_reference_changed')
    return json.loads(value)


def root_path(text):
    path = Path(text)
    if not path.is_absolute() or path.resolve() != path or path == Path('/'):
        raise ValueError('observer_root_unsafe')
    return path


def protocol_for(cfg, now):
    protocol = checked(cfg['retained_study_protocol'])
    original = checked(protocol['original_study_plan'])
    prior = protocol['prior_scientific_capture_attempts']
    cap = protocol['max_new_members']
    if (protocol['schema_version'] != 'retained_study_protocol_v1'
            or protocol['status'] != 'AUTHORIZED_OUTCOME_BLIND_RETAINED_STUDY'
            or not stamp(protocol['issued_at']) < stamp(protocol['effective_at']) <= now
            or stamp(protocol['ends_at']) != stamp(original['ends_at'])
            or type(prior) is not int or prior < 0 or type(cap) is not int or cap < 1
            or prior+cap != protocol['max_total_members']
            or set(protocol['frozen_model_files']) != MODEL_FILES
            or len(set(protocol['prior_member_race_ids'])) != len(protocol['prior_member_race_ids'])
            or len(protocol['prior_member_race_ids']) > prior
            or set(protocol['scan_limits']) != {'max_files', 'max_bytes'}
            or any(type(v) is not int or v < 1 for v in protocol['scan_limits'].values())):
        raise ValueError('observer_protocol_invalid')
    return protocol


class Journal:
    def __init__(self, root):
        self.path = root/'events.jsonl'
        self.events = []
        self.previous = '0'*64
        if self.path.exists():
            for line in raw(self.path, 64*1024*1024).splitlines():
                row = json.loads(line)
                expected = hashlib.sha256(canonical({'sequence': len(self.events), 'previous': self.previous, 'event': row['event']})).hexdigest()
                if row != {'sequence': len(self.events), 'previous': self.previous, 'event': row['event'], 'sha256': expected}:
                    raise ValueError('observer_journal_changed')
                self.events.append(row['event']); self.previous = expected

    def append(self, event):
        row = {'sequence': len(self.events), 'previous': self.previous, 'event': event}
        row['sha256'] = hashlib.sha256(canonical(row)).hexdigest()
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, 'ab') as stream:
            stream.write(canonical(row)+b'\n'); stream.flush(); os.fsync(stream.fileno())
        self.events.append(event); self.previous = row['sha256']


def metadata_candidate(plan_ref, plan, admission_path, protocol):
    ar = reference(admission_path); admission = checked(ar)
    completion_path = admission_path.with_name('completion.json')
    if not completion_path.exists():
        raise ValueError('NATIVE_COMPLETION_PENDING')
    cr = reference(completion_path); completion = checked(cr)
    race = admission['race']; jump = stamp(race['jump_timestamp'])
    if (admission['plan_sha256'] != plan_ref['sha256']
            or admission_path.parent.name != hashlib.sha256(race['race_id'].encode()).hexdigest()
            or completion['admission_sha256'] != ar['sha256']
            or any(completion[k] != admission[k] for k in ('race', 'job_id', 'prediction_id', 'plan_sha256', 'runner_set_sha256', 'retained_input_manifest_sha256', 'evidence_class'))
            or admission['evidence_class'] != 'AUTHORIZED_ENGINEERING'
            or plan['status'] != 'AUTHORIZED_ENGINEERING'
            or not stamp(plan['starts_at']) <= jump < stamp(plan['ends_at'])
            or stamp(admission['decision_at']) != jump-timedelta(seconds=120)
            or not stamp(admission['admitted_at']) < stamp(admission['decision_at'])
            or not stamp(admission['admitted_at']) <= stamp(completion['published_complete_at']) < stamp(admission['decision_at'])
            or completion['status'] != 'COMPLETE_BEFORE_CUTOFF'
            or completion['models'] != {name: 'SEALED' for name in MODELS}):
        raise ValueError('NATIVE_SEAL_METADATA_INVALID')
    directory = completion['bundle_entry']['directory']
    if Path(directory).name != directory:
        raise ValueError('NATIVE_BUNDLE_PATH_INVALID')
    bundles = root_path(plan['prediction_output_roots'][0]); bundle = bundles/directory
    manifest_ref = {'path': str(bundle/'bundle_manifest.json'), 'sha256': completion['bundle_entry']['manifest_sha256']}
    manifest = checked(manifest_ref)
    if manifest['job_id'] != admission['job_id'] or manifest['prediction_id'] != admission['prediction_id']:
        raise ValueError('NATIVE_BUNDLE_IDENTITY_INVALID')
    verify_bundle_files(bundle, manifest)
    request = checked({'path': str(bundle/'request.json'), 'sha256': manifest['files']['request.json']['sha256']})
    if (request['race_id'] != race['race_id'] or stamp(request['jump_timestamp']) != jump
            or any(request[k] != admission[k] for k in ('job_id', 'prediction_id',
                'runner_set_sha256', 'retained_input_manifest_sha256'))):
        raise ValueError('NATIVE_REQUEST_IDENTITY_CHANGED')
    implementation_ref = {'path': str(bundle/'features/sealed/implementation_file_manifest.json'),
        'sha256': manifest['files']['features/sealed/implementation_file_manifest.json']['sha256']}
    implementation = checked(implementation_ref)
    for name, expected in protocol['frozen_model_files'].items():
        if manifest['files'][name]['sha256'] != expected or reference(bundle/name)['sha256'] != expected:
            raise ValueError('FROZEN_MODEL_BINDING_CHANGED')
    forecasts = {}
    for name in sorted(MODELS):
        path = bundle/f'comparison/{name}.json'; ref = reference(path)
        if ref['sha256'] != manifest['files'][f'comparison/{name}.json']['sha256']:
            raise ValueError('NATIVE_FORECAST_FILE_CHANGED')
        forecasts[name] = ref
    return {'race_id': race['race_id'], 'job_id': admission['job_id'], 'prediction_id': admission['prediction_id'],
            'jump_at': race['jump_timestamp'], 'original_evidence_class': admission['evidence_class'],
            'original_plan': plan_ref, 'admission': ar, 'completion': cr, 'bundle_manifest': manifest_ref,
            'original_forecasts': forecasts, 'original_admitted_at': admission['admitted_at'],
            'original_published_complete_at': completion['published_complete_at'],
            'original_implementation': implementation_ref, 'original_source_commit': implementation['git_head'],
            'runner_set_sha256': admission['runner_set_sha256'],
            'retained_input_manifest_sha256': admission['retained_input_manifest_sha256'],
            'qualification': 'ALL_DECLARED_CAPTURE_INPUT_HISTORY_FEATURE_FORECAST_FILES_HASH_VERIFIED_NO_NEW_REPLAY',
            'independent_verifier_evidence': protocol.get('independent_verifier_evidence', []),
            'new_scores_generated': False, 'result_status_used_for_selection': False}


def discover_plans(protocol):
    """Use only original approved plans and authenticated single-producer days."""
    refs = list(protocol['historical_plans'])
    persistent = protocol.get('persistent_source')
    if persistent:
        from race_collection.persistent_authority import load_persistent_allocation, load_standing_authority
        standing = load_standing_authority(persistent['standing_authority'])
        runtime = root_path(persistent['runtime_root'])
        if runtime != root_path(standing['state_root']):
            raise ValueError('observer_persistent_root_changed')
        first = date.fromisoformat(persistent['first_racing_date'])
        last = stamp(protocol['ends_at']).date()
        for day in sorted((runtime/'days').iterdir()):
            if not first <= date.fromisoformat(day.name) <= last:
                continue
            root_path(str(day))
            allocation_path, comparison_path = day/'allocation.json', day/'comparison.json'
            if not allocation_path.exists() or not comparison_path.exists():
                continue  # Interrupted preparation has no comparison membership.
            allocation_ref = reference(allocation_path)
            allocation = load_persistent_allocation(allocation_ref)
            ref = reference(comparison_path); plan = checked(ref)
            if (allocation['standing_authority'] != persistent['standing_authority']
                    or allocation['racing_date'] != day.name
                    or plan['persistent_allocation'] != allocation_ref
                    or plan['programme_root'] != str(day/'admission')
                    or plan['prediction_output_roots'] != [str(Path(allocation['prediction_root'])/'bundles')]
                    or plan['candidate_registry'] != standing['candidate_registry']):
                raise ValueError('observer_daily_plan_binding_changed')
            refs.append(ref)
    unique = {r['path']: r for r in refs}
    if any(unique[r['path']] != r for r in refs):
        raise ValueError('observer_plan_reference_conflict')
    plans = []
    for ref in sorted(unique.values(), key=lambda r: (r['sha256'], r['path'])):
        checked(ref)
        plans.append((ref, load_plan(Path(ref['path']), ref['sha256'])[0]))
    return plans


def append_once(journal, event, now):
    key = hashlib.sha256(canonical(event)).hexdigest()
    if not any(row.get('event_key') == key for row in journal.events):
        journal.append({**event, 'event_key': key, 'observed_at': now.isoformat()})


def observe(cfg, *, now):
    now = stamp(now.isoformat())
    readiness = verify_retained_readiness(cfg, now)
    if readiness is None:
        return {'status': 'NOT_EFFECTIVE', 'mutations': False}
    protocol = protocol_for(cfg, now)
    token = BUDGET.set([protocol['scan_limits']['max_files'], protocol['scan_limits']['max_bytes']])
    try:
        return _observe(cfg, protocol, now)
    finally:
        BUDGET.reset(token)


def _observe(cfg, protocol, now):
    root = root_path(protocol['state_root'])
    reservation = reservations.load(cfg, checked, root_path)
    plans = discover_plans(protocol)
    protected_roots = [root_path(text) for _, plan in plans
                       for text in [plan['programme_root'], *plan['prediction_output_roots']]]
    original = checked(protocol['original_study_plan'])
    if 'programme_root' in original:
        protected_roots.append(root_path(original['programme_root']))
    protected_roots.extend(root_path(text) for text in original.get('prediction_output_roots', []))
    protected_roots.extend(root_path(cfg[key]) for key in ('campaign_root', 'state_root') if key in cfg)
    if protocol.get('persistent_source'):
        protected_roots.append(root_path(protocol['persistent_source']['runtime_root']))
    if reservation:
        protected_roots.append(reservation['root'])
    for protected in protected_roots:
        if root == protected or root.is_relative_to(protected) or protected.is_relative_to(root):
            raise ValueError('observer_root_overlaps_original')
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(root/'observer.lock', os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'a') as lock:
        if os.fstat(lock.fileno()).st_nlink != 1:
            raise ValueError('observer_lock_unsafe')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = Journal(root)
        identity = {'kind': 'IDENTITY', 'protocol': cfg['retained_study_protocol'], 'amendment': cfg['study_amendment']}
        if reservation:
            previous = reservation['predecessor']
            identity = {'kind': 'IDENTITY', 'protocol': previous['retained_study_protocol'],
                        'amendment': previous['study_amendment']}
        if journal.events and journal.events[0] != identity:
            raise ValueError('observer_authority_changed')
        if not journal.events:
            journal.append(identity)
        reservations.bind(reservation, cfg, journal)
        members = {r['race_id']: r for r in journal.events if r['kind'] == 'MEMBER'}
        if len(members) != sum(r['kind'] == 'MEMBER' for r in journal.events):
            raise ValueError('observer_duplicate_journal_members')
        if len(members) > protocol['max_new_members']:
            raise ValueError('observer_membership_cap_invalid')
        # Revalidate every already-selected binding before accepting additional data.
        for member in members.values():
            for ref in [member['admission'], member['completion'], member['bundle_manifest'],
                        *member['original_forecasts'].values()]:
                if reference(ref['path']) != ref:
                    raise ValueError('observer_selected_evidence_changed')
            manifest = checked(member['bundle_manifest'])
            verify_bundle_files(Path(member['bundle_manifest']['path']).parent, manifest)
        reservation_states = reservations.states(reservation, checked, reference, stamp, canonical, journal, now)
        for day, state in reservation_states.items():
            append_once(journal, {'kind': 'DEVELOPMENT_RESERVATION_DATE', 'local_date': day,
                'status': state['status'], 'evidence': state['evidence']}, now)
        for ref in protocol.get('independent_verifier_evidence', []):
            if reference(ref['path']) != ref:
                raise ValueError('observer_verifier_evidence_changed')
        for ref in protocol['opportunity_evidence']:
            if reference(ref['path']) != ref:
                raise ValueError('observer_opportunity_evidence_changed')
            append_once(journal, {'kind': 'ORIGINAL_DENOMINATOR_REFERENCE', 'reference': ref}, now)
        added = pending_reservations = 0
        for plan_ref, plan in plans:
            programme = Path(plan['programme_root'])/plan_ref['sha256']
            for path in sorted((programme/'opportunities').glob('*.json')):
                value = checked(reference(path))
                if value['plan_sha256'] != plan_ref['sha256']:
                    raise ValueError('observer_opportunity_plan_changed')
                append_once(journal, {'kind': 'ORIGINAL_OPPORTUNITY', 'plan': plan_ref,
                                     'reference': reference(path), 'race_id': value['race_id']}, now)
            for claim in sorted((programme/'attempts').iterdir() if (programme/'attempts').exists() else []):
                root_path(str(claim))
                path = claim/'admission.json'
                if not path.exists():
                    dispatch = claim/'dispatch.json'
                    append_once(journal, {'kind': 'UNADMITTED_NATIVE_ATTEMPT', 'plan': plan_ref,
                        'claim_path': str(claim), 'dispatch': reference(dispatch) if dispatch.exists() else None,
                        'reason': 'NO_NATIVE_ADMISSION'}, now)
                    continue
                try:
                    candidate = metadata_candidate(plan_ref, plan, path, protocol)
                    race = candidate['race_id']
                    if stamp(candidate['original_published_complete_at']) > now:
                        raise ValueError('NATIVE_COMPLETION_AFTER_OBSERVATION')
                    if race in members:
                        if members[race]['admission'] != candidate['admission']:
                            raise ValueError('DUPLICATE_RACE_DIFFERENT_ADMISSION')
                        continue
                    if any(candidate['job_id'] == m['job_id'] or candidate['prediction_id'] == m['prediction_id']
                           for m in members.values()):
                        raise ValueError('DUPLICATE_ORIGINAL_JOB_OR_PREDICTION')
                    if race in protocol['prior_member_race_ids']:
                        raise ValueError('ALREADY_IN_ORIGINAL_SCIENTIFIC_MEMBERSHIP')
                    if now >= stamp(protocol['ends_at']):
                        raise ValueError('OBSERVER_ENDPOINT_REACHED')
                    if len(members) >= protocol['max_new_members']:
                        raise ValueError('OBSERVER_MEMBERSHIP_CAP_REACHED')
                    disposition = reservations.disposition(reservation, reservation_states, candidate)
                    if disposition:
                        pending_reservations += disposition['reason'] == 'DEVELOPMENT_SELECTION_PENDING'
                        append_once(journal, {'kind': 'DEVELOPMENT_RESERVATION',
                            'race_id': race, 'admission': candidate['admission'],
                            'reservation': reservation['reference'], **disposition}, now)
                        continue
                    event = {'kind': 'MEMBER', **candidate, 'selection_at': now.isoformat(),
                             'membership_class': 'RETROSPECTIVE_RETAINED_PREJUMP_FORECAST'}
                    journal.append(event); members[race] = event; added += 1
                except RuntimeError as error:
                    if str(error) != 'observer_scan_allowance_exhausted':
                        raise
                    append_once(journal, {'kind': 'PENDING_QUALIFICATION', 'admission_path': str(path),
                                         'plan': plan_ref, 'reason': 'SCAN_ALLOWANCE_EXHAUSTED'}, now)
                    return {'status': 'PENDING_QUALIFICATION_SCAN_ALLOWANCE_EXHAUSTED',
                            'members': len(members), 'new_members': added, 'complete_scan': False}
                except (ValueError, KeyError, OSError, TypeError) as error:
                    reason = str(error)
                    if not reason or not all(c.isupper() or c == '_' for c in reason):
                        reason = 'NATIVE_METADATA_UNAVAILABLE_OR_INVALID'
                    append_once(journal, {'kind': 'EXCLUDED', 'admission_path': str(path),
                                         'plan': plan_ref, 'reason': reason}, now)
        input_state = 'NOT_ESTABLISHED_OBSERVER_DOES_NOT_CONTROL_PRODUCER'
        if protocol.get('persistent_source'):
            health_path = Path(protocol['persistent_source']['runtime_root'])/'health.json'
            if health_path.exists():
                health_ref = reference(health_path); health = checked(health_ref)
                input_state = ('REPORTED_PAUSED' if health.get('status') in {'HOLD', 'PAUSED', 'STOPPED', 'HALT'}
                               else 'REPORTED_STATE_NOT_A_CURRENT_HEALTH_GUARANTEE')
                append_once(journal, {'kind': 'PRODUCER_STATUS', 'reference': health_ref,
                    'new_input_state': input_state, 'reported_at': health.get('at')}, now)
        return {'status': 'OBSERVER_ENDPOINT_REACHED' if now >= stamp(protocol['ends_at']) else 'OBSERVATION_COMPLETE',
                'members': len(members), 'new_members': added, 'remaining_members': protocol['max_new_members']-len(members),
                'prior_scientific_capture_attempts': protocol['prior_scientific_capture_attempts'],
                'pending_development_reservations': pending_reservations,
                'provider_requests': 0, 'result_requests': 0, 'scores_generated': 0,
                'source_health_claim': 'NOT_ESTABLISHED_BY_HISTORICAL_READINESS',
                'new_input_state': input_state, 'complete_scan': True}
