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
from datetime import date, datetime, timezone

if __package__:
    from .readonly_json import read
else:
    from readonly_json import read

MODELS = ('production', 'market', 'residual_box', 'residual_half')
SCHEMA = 'operator_ui_persistent_collector_v1'


def validate_binding(binding):
    if set(binding) != {'config', 'config_sha256', 'source', 'source_commit', 'python'}:
        raise ValueError('invalid_persistent_binding')
    for name in ('config', 'source', 'python'):
        path = Path(binding[name])
        if not path.is_absolute() or path.resolve() != path:
            raise ValueError('unsafe_persistent_binding')
    for name, size in (('config_sha256', 64), ('source_commit', 40)):
        if re.fullmatch('[0-9a-f]{'+str(size)+'}', binding[name]) is None:
            raise ValueError('invalid_persistent_identity')
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
    if output.parent != day or not output.name.startswith('native-'):
        raise ValueError('persistent_daily_package_mismatch')
    receipt = read(day/'native-prepared.json')
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


def snapshot(binding, now, *, clock=None):
    cfg, pointer_path, pointer, output, plan_sha, plan, allocation = daily_bindings(binding)
    health = read(output/'persistent-health.json')
    if health['source_commit'] != binding['source_commit'] or health['source_date'] != pointer['racing_date']:
        raise ValueError('persistent_health_identity_changed')
    inventory_path = bound_path(health['inventory']['path'], output/'inventories')
    inventory = read(inventory_path, health['inventory']['sha256'])
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
    age = (now-stamp(health['at'])).total_seconds()
    inventory_age = (now-stamp(inventory['observed_at'])).total_seconds()
    value = {'schema': SCHEMA, 'observed_at': now.isoformat(), 'racing_date': pointer['racing_date'],
             'state': health['status'] if 0 <= age <= 300 else 'STATUS_STALE',
             'status_at': health['at'], 'inventory_at': inventory['observed_at'],
             'inventory_state': 'RECENT' if 0 <= inventory_age <= 300 else 'OLDER_INVENTORY',
             'race_count': inventory['race_count'], 'upcoming': upcoming,
             'forecasts': [], 'forecast_errors': [], 'scientific_admission': 'CANARY_NOT_VERIFIED',
             'result_access': False, 'engineering_only': True}
    attempts = Path(plan['programme_root'])/plan_sha/'attempts'
    admissions = sorted(attempts.glob('*/admission.json'))
    if len(admissions) > min(allocation['max_capture_attempts'], 255):
        raise ValueError('persistent_admission_bound')
    from src.predictor.future_comparison import verify_comparison
    bundles = Path(allocation['prediction_root'])/'bundles'
    for path in admissions:
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
            value['forecasts'].append(project_forecast(native, admission, manifest_sha, datetime.now(timezone.utc)))
        except Exception:
            value['forecast_errors'].append({'attempt': path.parent.name, 'reason': 'Sealed comparison could not be verified.'})
    if read(pointer_path) != pointer:
        raise ValueError('persistent_day_changed_during_read')
    value['forecasts'].sort(key=lambda forecast: stamp(forecast['race']['jump_timestamp']), reverse=True)
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
        return value
    except Exception:
        return {'schema': SCHEMA, 'state': 'UNAVAILABLE', 'upcoming': [], 'forecasts': [],
                'reason': 'Persistent collector evidence could not be verified. Refresh display to check again.',
                'scientific_admission': 'CANARY_NOT_VERIFIED', 'result_access': False}


def install_display(app, binding, protected):
    from .foundation import EvidenceStatus
    from .security import PreparedDisclosure
    app.config['OPERATOR_UI_PERSISTENT_DISPLAY'] = True

    @app.get('/operator-ui/api/v1/predictions/persistent')
    @protected(policy='LEVEL_1_API_V1_PREDICTION_DETAIL')
    def persistent_api():
        value = observe(binding)
        status = EvidenceStatus.UNAVAILABLE_DATA_MISSING if value['state'] == 'UNAVAILABLE' else EvidenceStatus.STALE if value['state'] == 'STATUS_STALE' else EvidenceStatus.AVAILABLE_FRESH
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
