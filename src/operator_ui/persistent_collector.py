"""Private read-only display of persistent engineering collection and forecasts."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from itertools import islice
from zoneinfo import ZoneInfo
from datetime import date, datetime, timezone

if __package__:
    from .readonly_json import read
else:
    from readonly_json import read

MODELS = ('production', 'market', 'residual_box', 'residual_half')
SCHEMA = 'operator_ui_persistent_collector_v1'
SAFE_FAILURE_CODES = {
    'RESIDUAL_SCORER_FAILED', 'production_not_ready', 'candidate_completed_after_cutoff',
    'operational_prediction_failed_preserved_consumption',
}


def failure_code(value):
    return value.upper() if isinstance(value, str) and value in SAFE_FAILURE_CODES else 'FAILURE_DETAIL_WITHHELD'


def control_status(pointer_path, output, health, now):
    """A retained HOLD/HALT is a stop signal, never replaced by old activity."""
    age = (now-stamp(health['at'])).total_seconds()
    value = {'state': health['status'] if 0 <= age <= 300 else 'STATUS_STALE',
             'status_at': health['at'], 'status_source': 'daily_health',
             'daily_state': health['status'], 'daily_status_at': health['at']}
    try:
        runtime = read(pointer_path.parent/'health.json')
    except FileNotFoundError:
        runtime = None
    try:
        halt = read(output/'HALT.json')
    except FileNotFoundError:
        halt = None
    stopped = runtime if runtime and runtime.get('status') == 'HOLD' else halt
    if stopped is not None:
        stamp(stopped['at'])
        value.update(state='HOLD' if stopped is runtime else 'HALT', status_at=stopped['at'],
                     status_source='runtime_hold' if stopped is runtime else 'daily_halt',
                     status_reason=failure_code(stopped.get('reason')))
    elif runtime and runtime.get('output') == str(output):
        preparation = read(pointer_path).get('preparation')
        if runtime.get('source_commit') != health['source_commit'] or runtime.get('preparation') != preparation:
            raise ValueError('persistent_runtime_identity_changed')
        age = (now-stamp(runtime['at'])).total_seconds()
        value.update(state=runtime['status'] if runtime['status'] in ('PAUSED', 'DAY_ENDED') or 0 <= age <= 300 else 'STATUS_STALE',
                     status_at=runtime['at'], status_source='runtime_health')
    return value


def validate_binding(binding):
    if set(binding)-{'producer_packages'} != {'config', 'config_sha256', 'source', 'source_commit', 'python'}:
        raise ValueError('invalid_persistent_binding')
    for name in ('config', 'source', 'python'):
        path = Path(binding[name])
        if not path.is_absolute() or path.resolve() != path:
            raise ValueError('unsafe_persistent_binding')
    for name, size in (('config_sha256', 64), ('source_commit', 40)):
        if re.fullmatch('[0-9a-f]{'+str(size)+'}', binding[name]) is None:
            raise ValueError('invalid_persistent_identity')
    if 'producer_packages' in binding:
        if __package__:
            from .native_verification import validate_packages
        else:
            from native_verification import validate_packages
        validate_packages(binding['producer_packages'])
    return binding


def stamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('persistent_timestamp_requires_zone')
    return result


def bound_path(value, parent):
    path = Path(value)
    if path.resolve() != path or not path.is_relative_to(parent):
        raise ValueError('persistent_path_outside_binding')
    return path


def daily_bindings(binding):
    """Resolve today's immutable package each time; never retain yesterday's path."""
    cfg = read(binding['config'], binding['config_sha256'])
    if cfg['source_commit'] != binding['source_commit'] or cfg['python'] != binding['python']:
        raise ValueError('persistent_config_identity_changed')
    standing_ref = cfg['standing_authority']
    standing = read(standing_ref['path'], standing_ref['sha256'])
    if standing['engineering_only'] is not True or standing['human_outcome_access'] is not False:
        raise ValueError('persistent_scope_changed')
    state = Path(standing['state_root'])
    pointer_path = state/'current-day.json'
    pointer = read(pointer_path)
    racing_date = date.fromisoformat(pointer['racing_date']).isoformat()
    day = state/'days'/racing_date
    output = bound_path(pointer['output'], day)
    if not output.name.startswith('native-'):
        raise ValueError('persistent_daily_package_mismatch')
    preparation = pointer.get('preparation')
    receipt_path = bound_path(preparation['path'], day) if preparation else day/'native-prepared.json'
    receipt = read(receipt_path, preparation['sha256'] if preparation else None)
    if (receipt['standing_authority'] != standing_ref or receipt['output'] != str(output)
            or receipt['racing_date'] != racing_date):
        raise ValueError('persistent_preparation_changed')
    plan_ref = receipt['comparison']
    bound_path(plan_ref['path'], day)
    from src.predictor.future_comparison import load_plan
    from race_collection.persistent_comparison import validate_persistent_plan
    plan, _ = load_plan(Path(plan_ref['path']), plan_ref['sha256'])
    allocation = validate_persistent_plan(plan)
    if allocation['racing_date'] != racing_date or allocation['standing_authority'] != standing_ref:
        raise ValueError('persistent_allocation_changed')
    return cfg, pointer_path, pointer, output, plan_ref['sha256'], plan, allocation


def project_failed_attempt(native, admission, manifest_sha, checked_at):
    """Only independently verified failed records yield a failure card."""
    records = native['records']
    if (native['evidence_class'] != 'AUTHORIZED_ENGINEERING' or native['future_race_evidence'] is not False
            or set(records) != set(MODELS) or not any(row['status'] == 'FAILED' for row in records.values())):
        raise ValueError('persistent_failure_not_verified')
    if any(row['status'] not in ('FAILED', 'SEALED') for row in records.values()):
        raise ValueError('persistent_failure_status_invalid')
    if any(row['predictions'] is not None for row in records.values() if row['status'] == 'FAILED'):
        raise ValueError('persistent_failure_has_predictions')
    race = admission['race']
    stamp(race['jump_timestamp'])
    if not isinstance(race['race_id'], str) or len(race['race_id']) > 256:
        raise ValueError('persistent_failure_race_identity_invalid')
    return {'race': {'race_id': race['race_id'], 'jump_timestamp': race['jump_timestamp']},
            'job_id': admission['job_id'], 'status': 'FAILED',
            'candidates': {name: {'status': records[name]['status'],
                                'failure': failure_code(records[name].get('failure')) if records[name]['status'] == 'FAILED' else None}
                           for name in MODELS},
            'verified_at': checked_at.isoformat(), 'manifest_sha256': manifest_sha,
            'evidence_class': 'ENGINEERING', 'scientific_admission': 'CANARY_NOT_VERIFIED'}


def project_forecast(native, admission, manifest_sha, checked_at):
    if (native['engineering_evidence'] is not True or native['future_race_evidence'] is not False
            or native['eligible_common_race'] is not True or set(native['records']) != set(MODELS)):
        raise ValueError('persistent_four_forecasts_not_eligible')
    records = native['records']
    if any(records[name]['status'] != 'SEALED' for name in MODELS):
        raise ValueError('persistent_candidate_not_sealed')
    market = records['market']['predictions']
    rows = []
    for index, runner in enumerate(market):
        probabilities = {name: records[name]['predictions'][index]['probability'] for name in MODELS}
        identities = {(records[name]['predictions'][index]['box_number'],
                       records[name]['predictions'][index]['identity']) for name in MODELS}
        if len(identities) != 1 or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 < p < 1 for p in probabilities.values()):
            raise ValueError('persistent_invalid_probability_or_identity')
        rows.append({'box': runner['box_number'], 'name': runner['dog_name'], 'probabilities': probabilities})
    if not rows or any(abs(math.fsum(row['probabilities'][name] for row in rows)-1) > 1e-12 for name in MODELS):
        raise ValueError('persistent_invalid_normalization')
    return {'race': admission['race'], 'job_id': admission['job_id'], 'runners': rows,
            'models': {name: records[name]['model_sha256'] for name in MODELS},
            'published_at': native['completion']['published_complete_at'],
            'verified_at': checked_at.isoformat(), 'manifest_sha256': manifest_sha,
            'evidence_class': 'ENGINEERING', 'scientific_admission': 'CANARY_NOT_VERIFIED'}


def verified_attempts(plan_sha, plan, allocation, *, remaining=255, deadline=None, selected=None):
    value = {'forecasts': [], 'failed_forecasts': [], 'forecast_errors': []}
    attempts = Path(plan['programme_root'])/plan_sha/'attempts'
    admissions = sorted(attempts.glob('*/admission.json'))
    if len(admissions) > min(allocation['max_capture_attempts'], 255):
        raise ValueError('persistent_admission_bound')
    if selected is not None:
        requested=set(selected)
        if not requested.issubset({str(path) for path in admissions}):
            raise ValueError('persistent_selected_admission_outside_plan')
        admissions=[path for path in admissions if str(path) in requested]
    from src.predictor.future_comparison import verify_comparison
    bundles = Path(allocation['prediction_root'])/'bundles'
    limited = False
    processed = 0
    for index, path in enumerate(admissions):
        if index >= remaining or (deadline is not None and time.monotonic() >= deadline):
            limited = True
            break
        processed = index+1
        if not path.with_name('completion.json').exists():
            continue
        try:
            admission = read(path)
            completion = read(path.with_name('completion.json'))
            directory = bound_path(str(bundles/completion['bundle_entry']['directory']), bundles)
            manifest_sha = completion['bundle_entry']['manifest_sha256']
            read(directory/'bundle_manifest.json', manifest_sha)
            fingerprint = hashlib.sha256(path.read_bytes()+path.with_name('completion.json').read_bytes()).hexdigest()
            native = verify_comparison(bundles, path, expected_plan_sha256=plan_sha)
            if fingerprint != hashlib.sha256(path.read_bytes()+path.with_name('completion.json').read_bytes()).hexdigest():
                raise ValueError('persistent_completion_changed_during_verification')
            read(directory/'bundle_manifest.json', manifest_sha)
            checked_at = datetime.now(timezone.utc)
            if any(row['status'] == 'FAILED' for row in native['records'].values()):
                value['failed_forecasts'].append(project_failed_attempt(native, admission, manifest_sha, checked_at))
            else:
                value['forecasts'].append(project_forecast(native, admission, manifest_sha, checked_at))
        except Exception:
            value['forecast_errors'].append({'attempt': path.parent.name, 'reason': 'Sealed comparison could not be verified.'})
    return value, limited, processed


def current_snapshot(binding, now, *, clock=None):
    cfg, pointer_path, pointer, output, plan_sha, plan, allocation = daily_bindings(binding)
    preparation = pointer.get('preparation')
    receipt = read(preparation['path'], preparation['sha256']) if preparation else read(output.parent/'native-prepared.json')
    native_ref = receipt['plan']
    bound_path(native_ref['path'], output)
    native_plan = read(native_ref['path'], native_ref['sha256'])
    try:
        health = read(output/'persistent-health.json')
    except FileNotFoundError:
        health = {'at': receipt['at'], 'status': 'PREPARED_NOT_STARTED',
                  'source_commit': native_plan['commit'], 'source_date': pointer['racing_date']}
    if health['source_commit'] != native_plan['commit'] or health['source_date'] != pointer['racing_date']:
        raise ValueError('persistent_health_identity_changed')
    if 'inventory' in health:
        inventory_path = bound_path(health['inventory']['path'], output/'inventories')
        inventory = read(inventory_path, health['inventory']['sha256'])
    else:
        inventory = {'observed_at': health['at'], 'races': [], 'race_count': 0}
    if not isinstance(inventory['races'], list) or len(inventory['races']) > 1024 or inventory['race_count'] != len(inventory['races']):
        raise ValueError('persistent_inventory_bound')
    if clock is not None:
        now = clock()
    upcoming = []
    for race in inventory['races']:
        jump = race.get('scheduled_jump_datetime')
        if not jump or stamp(jump) <= now:
            continue
        upcoming.append({'title': race['title'], 'venue': race['venue_name'],
                         'race_number': race['race_number'], 'jump_at': jump,
                         'forecast_readiness': 'NOT_ESTABLISHED_BY_DISCOVERY'})
    upcoming.sort(key=lambda race: stamp(race['jump_at']))
    inventory_age = (now-stamp(inventory['observed_at'])).total_seconds()
    value = {'schema': SCHEMA, 'observed_at': now.isoformat(), 'racing_date': pointer['racing_date'],
             **control_status(pointer_path, output, health, now),
             'inventory_at': inventory['observed_at'],
             'inventory_state': 'NOT_YET_DISCOVERED' if 'inventory' not in health else 'RECENT' if 0 <= inventory_age <= 300 else 'OLDER_INVENTORY',
             'race_count': inventory['race_count'], 'upcoming': upcoming,
             'forecasts': [], 'failed_forecasts': [], 'forecast_errors': [], 'scientific_admission': 'CANARY_NOT_VERIFIED',
             'result_access': False, 'engineering_only': True}
    comparisons, _, _ = package_attempts(binding,receipt,output,plan_sha,plan,allocation)
    value.update(comparisons)
    if read(pointer_path) != pointer:
        raise ValueError('persistent_day_changed_during_read')
    value['forecasts'].sort(key=lambda forecast: stamp(forecast['race']['jump_timestamp']), reverse=True)
    return value


def package_attempts(binding,receipt,output,plan_sha,plan,allocation,*,remaining=255,deadline=None):
    if 'producer_packages' not in binding:
        return verified_attempts(plan_sha,plan,allocation,remaining=remaining,deadline=deadline)
    if __package__:
        from .native_verification import replay
    else:
        from native_verification import replay
    return replay(binding,receipt,output,receipt['comparison'],remaining=remaining,deadline=deadline)



def preparation_reference(path):
    value = read(path)
    with Path(path).open('rb') as handle:
        raw = handle.read(4*1024*1024+1)
    if len(raw) > 4*1024*1024:
        raise ValueError('persistent_preparation_bound')
    reference = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
    if read(path, reference['sha256']) != value:
        raise ValueError('persistent_preparation_changed')
    return reference, value


def retained_history(binding, now, current=None):
    """Verified artifact history never establishes collector or input freshness."""
    cfg = read(binding['config'], binding['config_sha256'])
    if cfg['source_commit'] != binding['source_commit'] or cfg['python'] != binding['python']:
        raise ValueError('persistent_config_identity_changed')
    authority = cfg['standing_authority']
    standing = read(authority['path'], authority['sha256'])
    if standing['engineering_only'] is not True or standing['human_outcome_access'] is not False:
        raise ValueError('persistent_scope_changed')
    days = Path(standing['state_root'])/'days'
    cutoff = current['racing_date'] if current else now.astimezone(ZoneInfo('Australia/Melbourne')).date().isoformat()
    value = {'forecasts': [], 'failed_forecasts': [], 'forecast_errors': [], 'history_limited': False}
    if not days.exists():
        return value
    children = list(islice(days.iterdir(),513))
    if len(children) > 512:
        value['history_limited'] = True
        value['forecast_errors'].append({'reason':'Retained native history directory bound exceeded.'})
        return value
    directories = [path for path in children if re.fullmatch(r'\d{4}-\d{2}-\d{2}', path.name) and path.name <= cutoff]
    directories.sort(key=lambda path:path.name, reverse=True)
    value['history_limited'] = len(directories) > 32
    deadline = time.monotonic()+15
    remaining = 255
    seen = set()
    if current:
        pointer = read(Path(standing['state_root'])/'current-day.json')
        ref = pointer.get('preparation')
        receipt = read(ref['path'], ref['sha256']) if ref else read(Path(pointer['output']).parent/'native-prepared.json')
        seen.add((current['racing_date'],receipt['comparison']['sha256']))
    from src.predictor.future_comparison import load_plan
    from race_collection.persistent_comparison import validate_persistent_plan
    for day in directories[:32]:
        try:
            bound_path(str(day), days)
            racing_date = date.fromisoformat(day.name).isoformat()
            receipts = [day/'native-prepared.json', *sorted(islice(day.glob('recoveries/*/native-prepared.json'),32))]
            if len(receipts) > 32:
                value['history_limited'] = True
        except Exception:
            value['forecast_errors'].append({'racing_date':day.name,'reason':'Retained native comparison could not be verified.'})
            continue
        for path in receipts[:32]:
            if not path.exists():
                continue
            if remaining <= 0 or time.monotonic() >= deadline:
                value['history_limited'] = True
                return value
            try:
                bound_path(str(path),day)
                preparation, receipt = preparation_reference(path)
                if receipt['standing_authority'] != authority or receipt['racing_date'] != racing_date:
                    raise ValueError('persistent_history_authority_changed')
                output = bound_path(receipt['output'],day)
                if not output.name.startswith('native-'):
                    raise ValueError('persistent_history_package_changed')
                reference = receipt['comparison']
                identity = (racing_date,reference['sha256'])
                if identity in seen:
                    continue
                bound_path(reference['path'],day)
                plan, _ = load_plan(Path(reference['path']),reference['sha256'])
                allocation = validate_persistent_plan(plan)
                if allocation['racing_date'] != racing_date or allocation['standing_authority'] != authority:
                    raise ValueError('persistent_history_allocation_changed')
                bound_path(plan['programme_root'],day)
                comparisons, limited, used = package_attempts(binding,receipt,output,reference['sha256'],plan,allocation,remaining=remaining,deadline=deadline)
                remaining -= used
                value['history_limited'] |= limited
                read(path,preparation['sha256'])
                read(reference['path'],reference['sha256'])
                seen.add(identity)
                for category in ('forecasts','failed_forecasts'):
                    for forecast in comparisons[category]:
                        forecast.update(retained_history=True,provenance={'racing_date':racing_date,'preparation':preparation,'comparison':reference})
                    value[category].extend(comparisons[category])
                value['forecast_errors'].extend(comparisons['forecast_errors'])
            except Exception:
                value['forecast_errors'].append({'racing_date':day.name,'reason':'Retained native comparison could not be verified.'})
    return value


def merge_forecasts(value, history):
    """Exact duplicates collapse; conflicting identities are withheld together."""
    combined={}; conflicts=set()
    for category in ('forecasts','failed_forecasts'):
        for forecast in [*value[category],*history[category]]:
            job=forecast['job_id']
            identity=(category,forecast['manifest_sha256'],forecast['race'])
            if job in combined and combined[job][0] != identity:
                conflicts.add(job)
            else:
                combined.setdefault(job,(identity,forecast))
    for category in ('forecasts','failed_forecasts'):
        value[category]=[forecast for job,(identity,forecast) in combined.items() if job not in conflicts and identity[0]==category]
        value[category].sort(key=lambda forecast:stamp(forecast['race']['jump_timestamp']),reverse=True)
    value['forecast_errors'].extend(history['forecast_errors'])
    value['forecast_errors'].extend({'reason':'Conflicting retained comparison identities were withheld.'} for _ in conflicts)
    value['history_limited']=history['history_limited']
    return value


def snapshot(binding, now, *, clock=None):
    current_error=None
    try:
        value=current_snapshot(binding,now,clock=clock)
    except Exception as error:
        current_error=error
        value={'schema':SCHEMA,'state':'UNAVAILABLE','observed_at':now.isoformat(),'upcoming':[],
               'forecasts':[],'failed_forecasts':[],'forecast_errors':[],
               'reason':'Current collector evidence could not be verified. Retained forecasts do not establish readiness.',
               'scientific_admission':'CANARY_NOT_VERIFIED','result_access':False,'engineering_only':True}
    try:
        history=retained_history(binding,now,None if current_error else value)
    except Exception:
        if current_error:
            raise current_error
        history={'forecasts':[],'failed_forecasts':[],'forecast_errors':[{'reason':'Retained native history could not be verified.'}],'history_limited':False}
    if current_error and not history['forecasts'] and not history['failed_forecasts']:
        raise current_error
    return merge_forecasts(value,history)

def service_status(value):
    """Observe the installed owner without changing its state."""
    try:
        result = subprocess.run(['systemctl', '--user', 'show', 'greyhound-persistent-collector.service',
                                 '--property=ActiveState,SubState,MainPID,ExecMainStatus'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3, check=True)
        fields = dict(line.split('=', 1) for line in result.stdout.splitlines())
        active, sub = fields['ActiveState'], fields['SubState']
        if active not in ('active', 'inactive', 'failed', 'activating', 'deactivating') or not re.fullmatch('[a-z-]{1,32}', sub):
            raise ValueError('unexpected_service_state')
        value['service'] = {'state': active, 'substate': sub, 'pid': int(fields['MainPID']), 'exit_status': int(fields['ExecMainStatus'])}
        if value['state'] not in ('HOLD', 'HALT', 'UNAVAILABLE') and active != 'active':
            value.update(state='FAILED' if active == 'failed' else 'STOPPED', status_source='installed_service')
    except Exception:
        value['service'] = {'state': 'UNAVAILABLE'}
        if value['state'] not in ('HOLD', 'HALT', 'UNAVAILABLE'):
            value.update(state='UNAVAILABLE', reason='Installed collector state could not be verified.')
    return value


def observe(binding):
    """No production root is writable and no provider connection is possible."""
    validate_binding(binding)
    command = ['bwrap', '--die-with-parent', '--unshare-pid', '--ro-bind', '/', '/',
               '--tmpfs', '/tmp', '--dev', '/dev', '--unshare-net', '--chdir', binding['source'],
               '/usr/bin/env', 'PYTHONDONTWRITEBYTECODE=1', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1',
               binding['python'], '-B', str(Path(__file__).resolve()), '--worker']
    try:
        result = subprocess.run(command, input=json.dumps(binding).encode(), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=40, check=True)
        if len(result.stdout) > 4*1024*1024:
            raise ValueError('persistent_display_response_bound')
        value = json.loads(result.stdout)
        if value['schema'] != SCHEMA:
            raise ValueError('persistent_display_schema_changed')
        return service_status(value)
    except Exception:
        return service_status({'schema': SCHEMA, 'state': 'UNAVAILABLE', 'upcoming': [], 'forecasts': [],
                'reason': 'Persistent collector evidence could not be verified. Refresh display to check again.',
                'scientific_admission': 'CANARY_NOT_VERIFIED', 'result_access': False})


def install_display(app, binding, protected):
    from .foundation import EvidenceStatus
    from .security import PreparedDisclosure
    app.config['OPERATOR_UI_PERSISTENT_DISPLAY'] = True

    @app.get('/operator-ui/api/v1/predictions/persistent')
    @protected(policy='LEVEL_1_API_V1_PREDICTION_DETAIL')
    def persistent_api():
        value = observe(binding)
        status = EvidenceStatus.UNAVAILABLE_DATA_MISSING if value['state'] in ('UNAVAILABLE', 'HOLD', 'HALT', 'FAILED', 'STOPPED', 'PAUSED', 'DAY_ENDED', 'PREPARED_NOT_STARTED') else EvidenceStatus.STALE if value['state'] == 'STATUS_STALE' else EvidenceStatus.AVAILABLE_FRESH
        value['classification'] = status.value
        raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        return PreparedDisclosure(body=raw, classification=status,
                                  evidence_source_identifiers=('persistent.engineering.display',),
                                  content_hashes=(hashlib.sha256(raw).hexdigest(),))

    @app.after_request
    def private_cache(response):
        from flask import request
        if request.path == '/operator-ui/api/v1/predictions/persistent':
            response.headers['Cache-Control'] = 'private, no-store'
        return response


if __name__ == '__main__':
    from contextlib import redirect_stdout, redirect_stderr
    import io
    binding = validate_binding(json.loads(sys.stdin.buffer.read(65537)))
    if Path(sys.executable) != Path(binding['python']) or not os.statvfs(binding['source']).f_flag & os.ST_RDONLY:
        raise ValueError('persistent_worker_not_readonly')
    git = lambda *args: subprocess.check_output(['git', '--no-optional-locks', *args], cwd=binding['source'], text=True, timeout=5).strip()
    if git('rev-parse', 'HEAD') != binding['source_commit'] or git('status', '--porcelain', '--untracked-files=no'):
        raise ValueError('persistent_source_changed')
    sys.path.insert(0, binding['source'])
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        value = snapshot(binding, datetime.now(timezone.utc), clock=lambda: datetime.now(timezone.utc))
    print(json.dumps(value, sort_keys=True, allow_nan=False))
