"""Read-only native producer consumer for the two reserved development dates.

Activation requires the separately approved ownership amendment and verified
observer exclusion. No collector, provider client, result client or study writer
is invoked here. Configured source controls remain read-only under systemd/bwrap.
"""
import hashlib
import json
from pathlib import Path

from race_collection import prospective_speed_inputs as inputs
from race_collection import prospective_speed_plan as planning
from race_collection import prospective_speed_runtime as runtime
from race_collection.development_examples import freeze_population
from race_collection.retained_card_timing_coverage import Reader, instant


def _checked_local(ref, root):
    path = Path(ref['path'])
    if not path.is_relative_to(root):
        raise ValueError('PRODUCER_REFERENCE_OUTSIDE_ROOT')
    return Reader().json(ref)


def _protected_snapshot(path):
    """Study observer metadata only; verify its append-only hash chain."""
    ref = runtime.reference(path)
    raw = Reader().read(ref)
    previous, ids = '0' * 64, []
    for index, line in enumerate(raw.splitlines()):
        row = json.loads(line)
        expected = hashlib.sha256(runtime.encoded({
            'sequence': index, 'previous': previous, 'event': row['event']})).hexdigest()
        if row != {'sequence': index, 'previous': previous, 'event': row['event'], 'sha256': expected}:
            raise ValueError('PROTECTED_JOURNAL_CHANGED')
        previous = expected
        if row['event'].get('kind') == 'MEMBER':
            ids.append(row['event']['race_id'])
    return {'source': ref, 'race_ids': sorted(set(ids)), 'last_chain_sha256': previous}


def _without_forecast(state, race_id, status, reason, plan, population):
    """Account for upstream failures even if no speed worker could be launched."""
    root = state / 'attempts'
    key = hashlib.sha256(race_id.encode()).hexdigest()
    with runtime.exclusive(root):
        attempt = root / key
        if attempt.exists():
            orphan = attempt / 'interrupted-before-claim.json'
            if orphan.exists():
                return {'status': 'INTERRUPTED_BEFORE_CLAIM', 'evidence': runtime.reference(orphan)}
            claim = attempt / 'claim.json'
            if claim.exists():
                try:
                    job_ref = Reader().json(runtime.reference(claim))['source_job']
                except json.JSONDecodeError:
                    return runtime.consumed_status(attempt, {}, False)
                value = runtime.consumed_status(attempt, job_ref, False)
                return _normalise_consumed(value, attempt)
            if (attempt / 'terminal.json').exists():
                terminal = Reader().json(runtime.reference(attempt / 'terminal.json'))
                return {'status': terminal['status'], 'terminal': runtime.reference(attempt / 'terminal.json')}
        else:
            attempt.mkdir(mode=0o700)
        ref = runtime.put_new(attempt / 'terminal.json', {'schema_version': 'prospective_speed_stage_failure_v1',
            'race_id': race_id, 'status': status, 'reason': reason, 'at': runtime.utc_now().isoformat(),
            'plan_sha256': planning._digest(plan), 'population_sha256': planning._digest(population)})
        return {'status': status, 'terminal': ref}


def _normalise_consumed(result, attempt):
    if result['status'] == 'ATTEMPT_ALREADY_CONSUMED':
        result['status'] = result.pop('original_status')
        if result['status'] in {'SEALED_PREJUMP', 'LATE_SPEED_SEAL'}:
            result['completion'] = runtime.reference(attempt / 'completion.json')
    return result


def _run_existing(job_ref, state):
    result = runtime.run_job(job_ref, state / 'attempts')
    race_id = Reader().json(job_ref)['member']['race_id']
    attempt = state / 'attempts' / hashlib.sha256(race_id.encode()).hexdigest()
    return _normalise_consumed(result, attempt)


def close_elapsed_dates(state, plan, local):
    """A stopped scheduler must leave whole missed dates and races visible."""
    for day in plan['dates']:
        if day > local.date().isoformat() or (day == local.date().isoformat() and local.strftime('%H:%M') <= '14:30'):
            continue
        day_root = state / day
        day_root.mkdir(exist_ok=True, mode=0o700)
        account_path = day_root / 'date-accounting.json'
        if not account_path.exists():
            runtime.put_new(account_path, {'local_date': day, 'status': 'FREEZE_INTERRUPTED',
                'reason': 'NO_COMPLETED_FREEZE_BY_FIXED_DEADLINE', 'population_sha256': None})
        account = Reader().json(runtime.reference(account_path))
        if account['status'] != 'POPULATION_FROZEN':
            continue
        population = Reader().json(account['population'])
        planning._validate_population(plan, population)
        jobs = day_root / 'jobs'
        jobs.mkdir(exist_ok=True, mode=0o700)
        for race_id in population['selected_race_ids']:
            key = hashlib.sha256(race_id.encode()).hexdigest()
            disposition = jobs / (key + '.disposition.json')
            if not disposition.exists():
                result = _without_forecast(state, race_id, 'UNATTEMPTED_AT_HORIZON', 'FIXED_HORIZON_ENDED', plan, population)
                runtime.put_new(disposition, {'race_id': race_id, **result})


def tick(config_reference):
    config = Reader().json(config_reference)
    plan = Reader().json(config['plan'])
    planning._validate_plan(plan)
    activation = Reader().json(config['activation'])
    if (activation.get('status') != 'AUTHORIZED_PROSPECTIVE_DEVELOPMENT'
            or activation.get('plan_sha256') != config['plan']['sha256']
            or activation.get('development_precedence_verified') is not True
            or activation.get('competing_pilot_disabled') is not True
            or activation.get('result_retention_routing_verified') is not True
            or activation.get('additional_source_requests') != 0
            or activation.get('additional_result_requests') != 0):
        raise ValueError('ALLOCATION_AND_OWNERSHIP_AMENDMENT_REQUIRED')
    for ref in activation['verified_control_files']:
        Reader().read(ref)
    state = Path(config['state_root'])
    producer = Path(config['producer_runtime_root'])
    if state.is_relative_to(producer) or producer.is_relative_to(state):
        raise ValueError('STATE_MUST_BE_SEPARATE_FROM_PRODUCER')
    with runtime.exclusive(state):
        now = runtime.utc_now()
        local = now.astimezone(planning.ZONE)
        day = local.date().isoformat()
        close_elapsed_dates(state, plan, local)
        if day not in plan['dates']:
            return {'status': 'NO_DEVELOPMENT_DATE_DUE', 'next_dates': [d for d in plan['dates'] if d > day]}
        day_root = state / day
        day_root.mkdir(exist_ok=True, mode=0o700)
        population_path = day_root / 'population.json'
        date_status = day_root / 'date-accounting.json'
        if not population_path.exists():
            if local.strftime('%H:%M') < '12:50':
                return {'status': 'WAIT_FOR_POPULATION_FREEZE'}
            if date_status.exists():
                return {'status': 'DATE_ALREADY_CONSUMED'}
            if local.strftime('%H:%M') > '12:50':
                runtime.put_new(date_status, {'local_date': day, 'status': 'FREEZE_INTERRUPTED',
                    'reason': 'NO_IMMUTABLE_POPULATION_BY_1251', 'population_sha256': None})
                return {'status': 'MISSED_FREEZE'}
            try:
                snapshot = _protected_snapshot(config['protected_membership_journal'])
                protected_ref = runtime.put_new(day_root / 'protected-membership.json', snapshot)
                allocation = plan['authority']['allocation']
                original = freeze_population(config['current_index'], config['index_evidence_root'],
                    allocation['path'], allocation['sha256'], day_root / 'original-population.json')
                population = planning.select_population(plan, original['observed_races'], local_date=day,
                    frozen_at=original['frozen_at'], source_observed_at=original['source_observed_at'],
                    index_complete=True, protected_membership_reference=protected_ref,
                    protected_race_ids=snapshot['race_ids'])
                population_ref = runtime.put_new(population_path, population)
                if runtime.utc_now().astimezone(planning.ZONE).strftime('%H:%M') != '12:50':
                    raise ValueError('FREEZE_CROSSED_CUTOFF')
                runtime.put_new(date_status, {'local_date': day, 'status': 'POPULATION_FROZEN',
                    'population_sha256': planning._digest(population), 'population': population_ref})
            except Exception as error:
                runtime.put_new(date_status, {'local_date': day, 'status': 'SOURCE_OR_AUTHORITY_UNAVAILABLE',
                    'reason': type(error).__name__, 'population_sha256': None})
                return {'status': 'POPULATION_FREEZE_FAILED'}
        if not date_status.exists():
            runtime.put_new(date_status, {'local_date': day, 'status': 'FREEZE_INTERRUPTED',
                'reason': 'POPULATION_WITHOUT_DURABLE_FREEZE_COMPLETION', 'population_sha256': None})
        account = Reader().json(runtime.reference(date_status))
        if account['status'] != 'POPULATION_FROZEN':
            return {'status': 'DATE_ALREADY_CONSUMED'}
        population_ref = runtime.reference(population_path)
        population = Reader().json(population_ref)
        planning._validate_population(plan, population)
        if planning._digest(population) != account['population_sha256']:
            raise ValueError('POPULATION_CHANGED')
        pointer = Reader().json(runtime.reference(producer / 'current-day.json'))
        preparation = _checked_local(pointer['preparation'], producer)
        source_plan = _checked_local(preparation['plan'], producer)
        comparison = _checked_local(source_plan['frozen_comparison'], producer)
        if pointer['racing_date'] != day or preparation['racing_date'] != day:
            return {'status': 'WAIT_FOR_MATCHING_PRODUCER_DAY'}
        admission_root = Path(comparison['programme_root']) / source_plan['frozen_comparison']['sha256'] / 'attempts'
        jobs_root = day_root / 'jobs'
        jobs_root.mkdir(exist_ok=True, mode=0o700)
        for race_id in population['selected_race_ids']:
            key = hashlib.sha256(race_id.encode()).hexdigest()
            disposition = jobs_root / (key + '.disposition.json')
            if disposition.exists():
                continue
            row = next(r for r in population['observed_races'] if r['race_id'] == race_id)
            admission = admission_root / key / 'admission.json'
            completion = admission.with_name('completion.json')
            if now >= instant(row['jump_at']):
                result = _without_forecast(state, race_id, 'UPSTREAM_FORECAST_UNAVAILABLE', 'NO_CONSUMABLE_SEAL_BEFORE_JUMP', plan, population)
                runtime.put_new(disposition, {'race_id': race_id, **result})
                continue
            if not completion.exists():
                continue
            try:
                existing_job = jobs_root / (key + '.job.json')
                if existing_job.exists():
                    result = _run_existing(runtime.reference(existing_job), state)
                    runtime.put_new(disposition, {'race_id': race_id, **result})
                    continue
                native = {'bundle_root': str(Path(source_plan['prediction_root']) / 'bundles'),
                    'plan_sha256': source_plan['frozen_comparison']['sha256'],
                    'allowed_source_roots': config['allowed_source_roots'],
                    'verifier_source_reference': preparation['plan']}
                member, original = inputs.member_from_native(Reader(), native['bundle_root'],
                    runtime.reference(admission), runtime.reference(completion),
                    expected_plan_sha256=native['plan_sha256'],
                    allowed_source_roots=native['allowed_source_roots'],
                    verifier_source_reference=native['verifier_source_reference'])
                race_activation = dict(activation, population_sha256=population_ref['sha256'])
                activation_ref = runtime.put_new(jobs_root / (key + '.activation.json'), race_activation)
                # Update from prior admitted captures, never from results. Original
                # source timestamps are preserved; the candidate enforces each cutoff.
                history = Reader().json(config['history_inventory'])
                known = {c['member']['race_id'] for c in history['cards']}
                for prior in sorted(state.glob('2026-10-*/jobs/*.job.json')):
                    value = Reader().json(runtime.reference(prior))
                    rid = value['member']['race_id']
                    if rid not in known and instant(value['original']['original_published_complete_at']) < runtime.utc_now():
                        history['cards'].append({'member': value['member'], 'original': value['original']})
                        known.add(rid)
                history_ref = runtime.put_new(jobs_root / (key + '.history.json'), history)
                job_ref = runtime.put_new(jobs_root / (key + '.job.json'), {
                    'member': member, 'original': original, 'model': config['model'], 'native': native,
                    'plan': config['plan'], 'population': population_ref, 'activation': activation_ref,
                    'history_inventory': history_ref})
                result = _run_existing(job_ref, state)
                runtime.put_new(disposition, {'race_id': race_id, **result})
            except Exception as error:
                result = _without_forecast(state, race_id, 'SPEED_PROCESSING_FAILED', type(error).__name__, plan, population)
                runtime.put_new(disposition, {'race_id': race_id, **result})
        return {'status': 'PROSPECTIVE_TICK_COMPLETE', 'date': day,
            'selected': len(population['selected_race_ids']),
            'terminal': len(list(jobs_root.glob('*.disposition.json')))}
