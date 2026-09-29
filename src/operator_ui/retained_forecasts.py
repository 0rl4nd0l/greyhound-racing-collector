"""Read-only, audited projection of verified production bundles, never study results."""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import render_template

from src.predictor.on_demand import verify_indexed_prediction_bundle, verify_prediction_bundle_index
from .foundation import EvidenceStatus
from .security import PreparedDisclosure

SCHEMA = 'operator_ui_retained_forecasts_v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def read(path, expected=None, limit=4 * 1024 * 1024):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError('unsafe_retained_path')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError('retained_file_bound')
        raw = os.read(fd, limit + 1)
        after = os.fstat(fd)
        identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if identity(before) != identity(after) or identity(path.stat()) != identity(after) or len(raw) != before.st_size:
            raise ValueError('retained_file_changed')
    finally:
        os.close(fd)
    if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('retained_identity_mismatch')
    return json.loads(raw)


def stamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('timestamp_requires_zone')
    return result


def validate_config(value):
    if set(value) != {'schema', 'roots', 'model_sha256', 'model_manifest_sha256', 'config_sha256', 'schedule', 'schedule_sha256', 'independent_audits', 'operational_job_ids'} or value['schema'] != SCHEMA:
        raise ValueError('invalid_forecast_display_config')
    if set(value['roots']) != {'operational', 'programme'}:
        raise ValueError('invalid_forecast_roots')
    for path in [*value['roots'].values(), value['schedule']]:
        if not isinstance(path, str) or not Path(path).is_absolute() or Path(path).resolve() != Path(path):
            raise ValueError('unsafe_forecast_root')
    for key in ('model_sha256', 'model_manifest_sha256', 'config_sha256', 'schedule_sha256'):
        if len(value[key]) != 64 or any(c not in '0123456789abcdef' for c in value[key]):
            raise ValueError('invalid_forecast_binding')
    if not isinstance(value['independent_audits'], list) or len(value['independent_audits']) > 32:
        raise ValueError('invalid_independent_audit_list')
    if not isinstance(value['operational_job_ids'], list) or not value['operational_job_ids'] or len(value['operational_job_ids']) != len(set(value['operational_job_ids'])):
        raise ValueError('invalid_operational_view_allowlist')
    return value


def unit_state(unit):
    output = subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID', '-p', 'UnitFileState'], text=True, timeout=3)
    return dict(line.split('=', 1) for line in output.splitlines() if '=' in line)


def programme_status(config, now, observe=unit_state):
    """An active timer is waiting, not evidence of collector activity."""
    try:
        schedule = read(config['schedule'], config['schedule_sha256'])
        if schedule['status'] != 'AUTHORIZED_PERSISTENT_SCHEDULE' or schedule['prediction_root'] != config['roots']['programme']:
            raise ValueError('programme_binding_mismatch')
        root = Path(schedule['state_root'])
        health = read(root / 'health.json')
        age = (now - stamp(health['at'])).total_seconds()
        next_slot = next((s for s in schedule['slots'] if stamp(s) > now), None)
        timer = observe('greyhound-comparison-schedule.timer')
        service = observe('greyhound-comparison-schedule.service')
        lanes = [observe(name) for name in ('shadow-autopilot.service', 'shadow-autopilot-odds-capture.service')]
        collecting = any(int(lane.get('MainPID', 0)) > 0 for lane in lanes)
        status = health['status']
        if (root / 'PAUSE_ADMISSIONS').exists():
            state = 'ADMISSIONS_PAUSED'
        elif not 0 <= age <= 900:
            state = 'STATUS_STALE'
        elif status == 'SESSION_RUNNING':
            state = 'ACTIVE_COLLECTION' if collecting and int(service.get('MainPID', 0)) > 0 else 'SESSION_STARTING_OR_RESTORING'
        elif status == 'NO_SLOT_DUE' and collecting:
            state = 'OTHER_COLLECTION_ACTIVE'
        elif status == 'NO_SLOT_DUE' and timer.get('ActiveState') == 'active' and timer.get('UnitFileState') == 'enabled':
            state = 'ARMED_SCHEDULED_IDLE'
        else:
            state = status if status != 'NO_SLOT_DUE' else 'SCHEDULE_NOT_ARMED'
        return {'state': state, 'health_status': status, 'observed_at': health['at'], 'next_session': next_slot, 'first_session': schedule['slots'][0],
                'next_session_end': (stamp(next_slot) + timedelta(minutes=schedule['session_minutes'])).isoformat() if next_slot else None,
                'collecting': collecting, 'schedule_timer': timer.get('ActiveState', 'unknown'),
                'collector_services': [lane.get('ActiveState', 'unknown') for lane in lanes]}
    except Exception:
        return {'state': 'UNAVAILABLE', 'reason': 'Programme status or identity could not be verified.', 'collecting': None}


def _verification_audits(root, references=None):
    paths = list((root / 'audit').glob('*.json')) if references is None else [Path(p) for p in references]
    if any(p.parent != root / 'audit' for p in paths):
        raise ValueError('verification_audit_outside_authorized_root')
    if len(paths) > 8192:
        raise ValueError('audit_inventory_bound')
    result = {}
    for path in paths:
        value = read(path, path.stem)
        if value.get('operation') != 'verify':
            continue
        if value.get('job_operation') != 'operational_prediction':
            continue
        if value.get('proposed_event', {}).get('facts', {}).get('verification_status') == 'VERIFIED':
            result.setdefault(value['job_id'], []).append(value)
    return result


def project_bundle(root, entry, audits, config, now, independent):
    verified = verify_indexed_prediction_bundle(root / 'bundles', entry)
    result, request = verified.result, verified.request
    if result['status'] != 'PREDICTION_READY' or request is None or not request.get('retained_input_manifest_sha256'):
        raise ValueError('retained_verified_forecast_required')
    if (result['model']['resolved'] != 'market_form_residual_v1' or result['model']['artifact_sha256'] != config['model_sha256']
            or result['model']['artifact_manifest_sha256'] != config['model_manifest_sha256'] or result['config']['sha256'] != config['config_sha256']):
        raise ValueError('production_model_identity_mismatch')
    matches = audits.get(result['job_id'], [])
    if len(matches) != 1:
        raise ValueError('independent_verification_missing_or_ambiguous')
    audit = matches[0]
    event, binding = audit['proposed_event'], audit['input']
    facts = event['facts']
    if hashlib.sha256(canonical(binding)).hexdigest() != audit['input_identity_sha256']:
        raise ValueError('verification_input_identity_mismatch')
    expected = {'verification_status': 'VERIFIED', 'prediction_id': result['prediction_id'], 'job_id': result['job_id'],
                'race_id': result['race']['race_id'], 'model_sha256': config['model_sha256'],
                'manifest_sha256': entry['manifest_sha256'], 'result_sha256': verified.manifest['files']['result.json']['sha256']}
    if event['status'] != 'READY' or any(facts.get(k) != v for k, v in expected.items()):
        raise ValueError('verification_fact_mismatch')
    if binding['race_id'] != result['race']['race_id'] or binding['retained_input_manifest_sha256'] != request['retained_input_manifest_sha256']:
        raise ValueError('retained_verification_identity_mismatch')
    bundle = root / 'bundles' / verified.directory
    capture = read(bundle / 'source/capture.json', verified.manifest['files']['source/capture.json']['sha256'])['source_attempt']
    validation = capture['validation']
    rows = validation['accepted_rows']
    modeled = result['prediction']['predictions']
    if validation['status'] != 'PASS' or len(rows) != len(modeled) or len(rows) != validation['active_expected_runner_count']:
        raise ValueError('market_coverage_invalid')
    by_box = {v['box_number']: v for v in rows}
    if len(by_box) != len(rows):
        raise ValueError('duplicate_runner_box')
    total = sum(1 / float(v['odds_decimal']) for v in rows)
    runners = []
    for row in sorted(modeled, key=lambda r: r['box_number']):
        quote = by_box[row['box_number']]
        odds, probability = float(quote['odds_decimal']), row['probability']
        if quote['identity'] != row['identity'] or not math.isfinite(odds) or odds <= 1 or not isinstance(probability, (float, int)) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('runner_probability_or_quote_invalid')
        runners.append({'box': row['box_number'], 'name': row['dog_name'], 'win_odds': odds,
                        'market_probability': (1 / odds) / total, 'model_probability': probability, 'model_rank': row['rank']})
    if not math.isclose(sum(r['model_probability'] for r in runners), 1, abs_tol=1e-6):
        raise ValueError('model_probability_sum_invalid')
    race = result['race']
    terminal = read(root / 'races' / hashlib.sha256(race['race_id'].encode()).hexdigest() / 'terminal.json')
    if terminal['job_id'] != result['job_id'] or terminal['race_id'] != race['race_id'] or terminal['status'] != 'PREDICTION_READY':
        raise ValueError('terminal_identity_mismatch')
    quote_at = terminal['price_observed_at']
    # The captured fetch start is conservative; verify it against the sealed source.
    if stamp(quote_at) != stamp(capture['fetch_time']):
        raise ValueError('quote_time_mismatch')
    jump = stamp(race['jump_timestamp'])
    if not stamp(quote_at) <= stamp(result['generated_at']) <= stamp(event['event_at']) <= stamp(terminal['completed_at']) < jump:
        raise ValueError('prejump_verification_order_invalid')
    return {'prediction_id': result['prediction_id'], 'job_id': result['job_id'],
            'race': {k: race[k] for k in ('race_id', 'venue', 'race_number', 'race_date', 'jump_timestamp')},
            'model': {'identity': result['model']['resolved'], 'sha256': config['model_sha256']},
            'verification': 'VERIFIED', 'temporal_status': 'HISTORICAL' if now >= jump else 'PRE_JUMP',
            'quote_at': quote_at, 'prediction_at': result['generated_at'], 'verification_at': terminal['completed_at'],
            'independent_verification_at': event['event_at'], 'independent_chain_audit_at': independent.get(result['job_id']),
            'manifest_sha256': entry['manifest_sha256'], 'runners': runners}


def forecasts(config, now):
    independent = {}
    verification_paths = {}
    for item in config['independent_audits']:
        audit = read(item['path'], item['sha256'])
        if audit['status'] != 'COMPLETED_CHAINS_VERIFIED':
            raise ValueError('independent_audit_invalid')
        for chain in audit['chains']:
            if chain['findings'] or not all(chain['checks'].values()):
                raise ValueError('independent_audit_findings')
            verification_paths[chain['job_id']] = chain['verification_audit_path']
            old = independent.get(chain['job_id'])
            if old is None or stamp(audit['recorded_at']) < stamp(old):
                independent[chain['job_id']] = audit['recorded_at']
    programme = programme_status(config, now)
    sources = []
    for label, name in config['roots'].items():
        root = Path(name)
        source = {'source': label, 'forecasts': [], 'errors': []}
        try:
            index = root / 'bundles/prediction_bundle_index_v1.json'
            if label == 'programme' and not root.exists() and programme['state'] == 'ARMED_SCHEDULED_IDLE' and programme.get('next_session') and now < stamp(programme['first_session']):
                source.update(state='NOT_STARTED', reason='Programme is armed and waiting for its first session. No forecasts are available yet.')
            elif not index.exists():
                source['state'] = 'EMPTY' if label != 'operational' and root.is_dir() and not list((root / 'bundles').glob('prediction_*')) and not list((root / 'races').glob('*/terminal.json')) else 'UNAVAILABLE'
                source['reason'] = 'No retained forecasts yet.' if source['state'] == 'EMPTY' else 'Retained forecast index is unavailable.'
            else:
                view = verify_prediction_bundle_index(root / 'bundles', return_verified_view=True)
                references = None
                if label == 'operational' and all(job in verification_paths for job in config['operational_job_ids']):
                    references = [verification_paths[job] for job in config['operational_job_ids']]
                audits = _verification_audits(root, references)
                selected = [e for e in view.entries if label != 'operational' or e['job_id'] in config['operational_job_ids']]
                if label == 'operational':
                    for missing in set(config['operational_job_ids']) - {e['job_id'] for e in selected}:
                        source['errors'].append({'prediction_id': missing, 'reason': 'Expected retained forecast is missing.'})
                for entry in sorted(selected, key=lambda e: e['generated_at'], reverse=True):
                    if entry['status'] != 'PREDICTION_READY':
                        source['errors'].append({'prediction_id': entry['prediction_id'], 'reason': 'Prediction failed or was blocked; no probabilities disclosed.'})
                        continue
                    try:
                        source['forecasts'].append(project_bundle(root, entry, audits, config, now, independent))
                    except Exception as exc:
                        source['errors'].append({'prediction_id': entry['prediction_id'], 'reason': 'Bundle unavailable or unverifiable.', 'error_type': type(exc).__name__})
                source['state'] = 'PARTIAL_ERROR' if source['errors'] else 'VERIFIED' if source['forecasts'] else 'EMPTY'
        except Exception:
            source.update(state='UNAVAILABLE', reason='Retained inventory or verification evidence is malformed or unavailable.', forecasts=[])
        sources.append(source)
    return {'schema': SCHEMA, 'observed_at': now.isoformat(), 'sources': sources, 'programme': programme}


def install_forecast_display(app, config):
    validate_config(config)
    protected = app.extensions['operator_ui_operational_get']

    @app.after_request
    def private_forecast_cache(response):
        from flask import request
        if request.path in {'/operator-ui/forecasts', '/operator-ui/api/v1/predictions/retained', '/operator-ui/sign-in', '/operator-ui/login'}:
            response.headers['Cache-Control'] = 'private, no-store'
        return response

    @app.get('/operator-ui/forecasts')
    def forecast_page():
        return render_template('operator_ui_forecasts.jinja')

    @app.get('/operator-ui/api/v1/predictions/retained')
    @protected(policy='LEVEL_1_API_V1_PREDICTION_DETAIL')
    def forecast_api():
        payload = forecasts(config, datetime.now(timezone.utc))
        classification = EvidenceStatus.UNAVAILABLE_DATA_MISSING if any(s['state'] in {'UNAVAILABLE', 'PARTIAL_ERROR'} for s in payload['sources']) else EvidenceStatus.AVAILABLE_FRESH
        payload['classification'] = classification.value
        raw = canonical(payload)
        return PreparedDisclosure(body=raw, classification=classification,
                                  evidence_source_identifiers=('retained.production.forecasts',),
                                  content_hashes=(hashlib.sha256(raw).hexdigest(),))
