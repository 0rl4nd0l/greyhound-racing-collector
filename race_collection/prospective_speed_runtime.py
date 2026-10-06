"""Immutable experiment execution, isolated from the original prediction owner.

Only a prospective allocation coordinator may enqueue hash-bound jobs. This
module neither discovers races nor grants allocation or official-result access.
The same calculation entrypoint supports explicitly labelled retained-input
reconstruction without manufacturing a historical prospective seal.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import subprocess
import sys
import time

from race_collection import prospective_speed_inputs as inputs
from race_collection import prospective_speed_plan as plan
from race_collection.retained_card_timing_coverage import Reader, instant


def utc_now():
    return datetime.now(timezone.utc)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def reference(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read_output(ref):
    """Finite output allowance: measured 10.7 MB maximum, with 32 MiB headroom.

Source read limits are deliberately unchanged. The retained observation bank is
larger than an individual source file and has its own local storage allowance.
"""
    path = Path(ref['path'])
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError('OUTPUT_PATH_UNSAFE')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > 32 * 1024**2:
            raise ValueError('OUTPUT_FILE_LIMIT')
        raw = stream.read(32 * 1024**2 + 1)
        after = os.fstat(stream.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if (identity(before) != identity(after) or identity(after) != identity(path.stat())
            or len(raw) != before.st_size or hashlib.sha256(raw).hexdigest() != ref['sha256']):
        raise ValueError('OUTPUT_CHANGED')
    return json.loads(raw)


def put_new(path, value):
    """Exclusive, durable evidence; an interrupted write is never overwritten."""
    path = Path(path)
    payload = encoded(value) + b'\n'
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return reference(path)


@contextmanager
def exclusive(root):
    """Experiment lock only: never touches the collector or provider locks."""
    root = Path(root)
    if not root.is_absolute() or root.resolve() != root:
        raise ValueError('EXPERIMENT_ROOT_UNSAFE')
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.stat().st_uid != os.getuid() or stat.S_IMODE(root.stat().st_mode) & 0o077:
        raise ValueError('EXPERIMENT_ROOT_NOT_PRIVATE')
    fd = os.open(root / 'worker.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def calculate(job):
    """Read only the admitted source graph. Target labels are not an argument."""
    started = time.perf_counter()
    reader = Reader()
    seed = reader.json(job['history_inventory'])
    if seed['schema_version'] != 'prospective_speed_history_inventory_v1':
        raise ValueError('HISTORY_INVENTORY_INVALID')
    observations = []
    if len(seed['cards']) > 96:
        raise ValueError('HISTORY_CARD_CAP_EXCEEDED')
    read_count, read_bytes = 0, 0
    for item in seed['cards']:
        card_reader = Reader()
        _, rows, _ = inputs.retained.construct_member(card_reader, item['member'], item['original'])
        read_count += card_reader.reads
        read_bytes += card_reader.bytes
        if read_bytes > 64 * 1024**2 or time.perf_counter() - started > 60:
            raise ValueError('HISTORY_LOCAL_RESOURCE_CAP_EXCEEDED')
        observations.extend(rows)
    history_seconds = time.perf_counter() - started
    calculated = time.perf_counter()
    native_seconds = 0.0
    if job.get('execution_mode') == 'PROSPECTIVE':
        native = job['native']
        native_reader = Reader()
        native_started = time.perf_counter()
        member, original = inputs.member_from_native(native_reader, native['bundle_root'],
            job['member']['admission'], job['original']['completion'],
            expected_plan_sha256=native['plan_sha256'],
            allowed_source_roots=native['allowed_source_roots'],
            verifier_source_reference=native['verifier_source_reference'])
        native_seconds = time.perf_counter() - native_started
        read_count += native_reader.reads
        read_bytes += native_reader.bytes
        for key, value in member.items():
            if job['member'].get(key) != value:
                raise ValueError('NATIVE_MEMBER_CHANGED')
        for key, value in original.items():
            if job['original'].get(key) != value:
                raise ValueError('NATIVE_PUBLICATION_CHANGED')
    result = inputs.forecast(reader, job['member'], job['original'], job['model'],
        forecast_at=job['forecast_at'], prior_observations=observations)
    result['speed_packet']['target']['observation_pool_scope'] = 'ALL_SOURCE_CARDS'
    result['execution_metrics'] = {'history_authentication_seconds': history_seconds,
        'forecast_seconds': time.perf_counter() - calculated,
        'calculation_wall_seconds': time.perf_counter() - started,
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'adapter_source_reads': read_count + reader.reads, 'adapter_source_bytes': read_bytes + reader.bytes,
        'native_verification_seconds': native_seconds,
        'io_count_scope': 'HASH_BOUND_ADAPTER_READS_EXCLUDING_NATIVE_VERIFIER_INTERNAL_IO'}
    return result


def isolated_command(job_reference, output, *, python=sys.executable):
    """The only writable host path is this attempt; networking is unavailable."""
    source = Path(__file__).resolve().parents[1]
    return ['bwrap', '--die-with-parent', '--unshare-net', '--ro-bind', '/', '/',
        '--bind', str(output), str(output), '--tmpfs', '/tmp', '--proc', '/proc',
        '--dev', '/dev', '--chdir', str(source), '--setenv', 'PYTHONPATH', str(source),
        '--setenv', 'PYTHONDONTWRITEBYTECODE', '1', '--setenv', 'OPENBLAS_NUM_THREADS', '1',
        '--setenv', 'OMP_NUM_THREADS', '1', python, '-B', '-m',
        'scripts.run_prospective_speed', 'calculate', '--job', job_reference['path'],
        '--job-sha256', job_reference['sha256'], '--output', str(output)]


def _limits():
    os.nice(15)
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024**2, 768 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))


def consumed_status(attempt, job_reference, replay):
    """Authenticate the stored chain before reporting an already consumed job."""
    if not (attempt / 'claim.json').exists():
        orphan = attempt / 'interrupted-before-claim.json'
        if not orphan.exists():
            put_new(orphan, {'status': 'INTERRUPTED_BEFORE_CLAIM', 'source_job': job_reference,
                'at': utc_now().isoformat()})
        return {'status': 'INTERRUPTED_BEFORE_CLAIM', 'evidence': reference(orphan)}
    try:
        claim = Reader().json(reference(attempt / 'claim.json'))
    except json.JSONDecodeError:
        partial = attempt / 'interrupted-partial-claim.json'
        if not partial.exists():
            put_new(partial, {'status': 'INTERRUPTED_PARTIAL_CLAIM',
                'claim': reference(attempt / 'claim.json'), 'at': utc_now().isoformat()})
        return {'status': 'INTERRUPTED_PARTIAL_CLAIM', 'evidence': reference(partial)}
    if claim['source_job'] != job_reference or claim['mode'] != ('REPLAY' if replay else 'PROSPECTIVE'):
        raise ValueError('CONSUMED_JOB_IDENTITY_CHANGED')
    terminal_path = attempt / 'terminal.json'
    if not terminal_path.exists():
        put_new(terminal_path, {'status': 'INTERRUPTED_ATTEMPT', 'race_id': claim['race_id'],
            'claim': reference(attempt / 'claim.json'), 'at': utc_now().isoformat()})
        return {'status': 'INTERRUPTED_ATTEMPT'}
    terminal = Reader().json(reference(terminal_path))
    if terminal['race_id'] != claim['race_id'] or terminal['claim'] != reference(attempt / 'claim.json'):
        raise ValueError('CONSUMED_CLAIM_CHANGED')
    if terminal.get('payload'):
        payload = read_output(terminal['payload'])
        if payload['race_id'] != claim['race_id']:
            raise ValueError('CONSUMED_PAYLOAD_IDENTITY_CHANGED')
    status = terminal['status']
    if status == 'FORECAST_PAYLOAD_DURABLE':
        completion_path = attempt / 'completion.json'
        if not completion_path.exists():
            return {'status': 'INTERRUPTED_SEAL', 'terminal': reference(terminal_path)}
        completion = Reader().json(reference(completion_path))
        seal = Reader().json(completion['seal'])
        if seal['terminal'] != reference(terminal_path) or seal['payload'] != terminal['payload']:
            raise ValueError('CONSUMED_SEAL_CHAIN_CHANGED')
        status = completion['status']
        if status == 'SEALED_PREJUMP' and (instant(completion['completed_at']) >= instant(payload['jump_at'])
                or instant(seal['completed_at']) >= instant(payload['jump_at'])):
            raise ValueError('CONSUMED_SEAL_NOT_PREJUMP')
    return {'status': 'ATTEMPT_ALREADY_CONSUMED', 'original_status': status,
        'terminal': reference(terminal_path)}


def run_job(job_reference, root, *, replay=False):
    """Exactly one consumed attempt; live clock, durable payload then seal.

Replay permits only source reconstruction and always records REPLAY_NOT_LIVE.
Live execution additionally requires a hash-bound allocation activation, frozen
population and original publication verification from the coordinator.
"""
    root = Path(root)
    job = Reader().json(job_reference)
    race_id = job['member']['race_id']
    key = hashlib.sha256(race_id.encode()).hexdigest()
    with exclusive(root):
        attempt = root / key
        if attempt.exists():
            return consumed_status(attempt, job_reference, replay)
        experiment_plan = population = None
        if not replay:
            reader = Reader()
            experiment_plan = reader.json(job['plan'])
            population = reader.json(job['population'])
            activation = reader.json(job['activation'])
            if (activation.get('status') != 'AUTHORIZED_PROSPECTIVE_DEVELOPMENT'
                    or activation.get('plan_sha256') != job['plan']['sha256']
                    or activation.get('population_sha256') != job['population']['sha256']
                    or activation.get('candidate_commit') != inputs.FROZEN_CANDIDATE_COMMIT
                    or activation.get('additional_source_requests') != 0
                    or activation.get('additional_result_requests') != 0
                    or activation.get('development_precedence_verified') is not True):
                raise ValueError('ALLOCATION_ACTIVATION_NOT_VERIFIED')
            job['forecast_at'] = utc_now().isoformat()
            plan.forecast_admission(experiment_plan, population, race_id,
                cutoff=job['forecast_at'], sealed_at=job['forecast_at'],
                input_available_at=job['original']['original_published_complete_at'])
        job['execution_mode'] = 'REPLAY' if replay else 'PROSPECTIVE'
        attempt.mkdir(mode=0o700)
        claim_ref = put_new(attempt / 'claim.json', {'race_id': race_id, 'source_job': job_reference,
            'claimed_at': utc_now().isoformat(), 'mode': 'REPLAY' if replay else 'PROSPECTIVE'})
        execution_job = put_new(attempt / 'job.json', job)
        started = time.perf_counter()
        try:
            with (attempt / 'worker.log').open('xb') as log:
                child = subprocess.run(isolated_command(execution_job, attempt), stdout=log,
                    stderr=subprocess.STDOUT, timeout=90, check=False, preexec_fn=_limits)
            if child.returncode:
                status = 'SPEED_PROCESSING_FAILED'
                payload = None
            else:
                payload = reference(attempt / 'forecast.json')
                value = read_output(payload)
                if value['race_id'] != race_id or value['forecast_at'] != job['forecast_at']:
                    raise ValueError('WORKER_IDENTITY_CHANGED')
                stamp = utc_now().isoformat()
                if replay:
                    status = 'REPLAY_NOT_LIVE'
                elif instant(stamp) >= instant(job['member']['jump_at']):
                    status = 'LATE_SPEED_FORECAST'
                else:
                    plan.forecast_admission(experiment_plan, population, race_id,
                        cutoff=value['information_cutoff'], sealed_at=stamp,
                        input_available_at=job['original']['original_published_complete_at'])
                    status = 'FORECAST_PAYLOAD_DURABLE'
            terminal = {'race_id': race_id, 'status': status, 'payload': payload,
                'completed_at': utc_now().isoformat(),
                'total_added_wall_seconds': time.perf_counter() - started,
                'provider_requests': 0, 'result_requests': 0}
        except subprocess.TimeoutExpired:
            terminal = {'race_id': race_id, 'status': 'SPEED_PROCESSING_TIMEOUT',
                'completed_at': utc_now().isoformat()}
        except Exception as error:
            terminal = {'race_id': race_id, 'status': 'SPEED_INTEGRITY_FAILURE',
                'error_type': type(error).__name__, 'completed_at': utc_now().isoformat()}
        terminal['claim'] = claim_ref
        terminal['execution_job'] = execution_job
        ref = put_new(attempt / 'terminal.json', terminal)
        # Durability completion, not the pre-write clock, decides pre-jump status.
        if terminal['status'] == 'FORECAST_PAYLOAD_DURABLE':
            completed = utc_now()
            seal_ref = put_new(attempt / 'seal.json', {'terminal': ref, 'payload': payload,
                'completed_at': completed.isoformat()})
            sealed = utc_now()
            final_status = 'SEALED_PREJUMP' if sealed < instant(job['member']['jump_at']) else 'LATE_SPEED_SEAL'
            completion = put_new(attempt / 'completion.json', {'seal': seal_ref,
                'completed_at': sealed.isoformat(), 'status': final_status})
            return {'status': final_status, 'completion': completion}
        return {'status': terminal['status'], 'terminal': ref}
