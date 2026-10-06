"""Read-only native producer consumer for the two reserved development dates.

Activation requires the approved development allocation and verified observer
exclusion. No collector, provider client, result client or study writer
is invoked here. Configured source controls remain read-only under systemd/bwrap.
"""
import hashlib
import json
from datetime import timedelta
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


def _producer_view(producer, day):
    pointer = Reader().json(runtime.reference(producer / 'current-day.json'))
    preparation = _checked_local(pointer['preparation'], producer)
    source_plan = _checked_local(preparation['plan'], producer)
    comparison = _checked_local(source_plan['frozen_comparison'], producer)
    if pointer['racing_date'] != day or preparation['racing_date'] != day:
        raise ValueError('PRODUCER_DAY_UNAVAILABLE')
    return preparation, source_plan, comparison


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
        if local <= instant(planning.selection_window(plan, day)['jump_end'])+timedelta(minutes=10):
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


def _freeze_daily_inventory(config, plan, producer, day, day_root, protected_ref, protected_ids):
    """Freeze the full native discovery census, never a price-qualified index."""
    from race_collection.daily_race_inventory import load_daily_inventory
    start = runtime.utc_now()
    preparation, _, _ = _producer_view(producer, day)
    health = Reader().json(runtime.reference(producer/'health.json'))
    if health['preparation'] != runtime.reference(Path(health['preparation']['path'])):
        raise ValueError('PRODUCER_PREPARATION_CHANGED')
    if (health.get('source_date') != day or Reader().json(health['preparation']) != preparation
            or health.get('output') != preparation['output']):
        raise ValueError('INVENTORY_PRODUCER_BINDING_CHANGED')
    inventory_ref = health['inventory']
    if not Path(inventory_ref['path']).is_relative_to(Path(preparation['output'])/'inventories'):
        raise ValueError('INVENTORY_OUTSIDE_NATIVE_OUTPUT')
    Reader().read(inventory_ref)
    inventory = load_daily_inventory(**inventory_ref, source_date=day, now=start, max_age_seconds=1800)
    rows = planning.inventory_rows(inventory, day)
    population = planning.select_population(plan, rows, local_date=day, frozen_at=start.isoformat(),
        source_observed_at=inventory['observed_at'], index_complete=True,
        protected_membership_reference=protected_ref, protected_race_ids=protected_ids)
    original = {'schema_version': 'development_population_freeze_v2', 'synthetic': False,
        'allocation_id': plan['allocation_id'], 'allocation_sha256': plan['authority']['allocation']['sha256'],
        'selection_policy': plan['population']['selection_policy'], 'local_date': day,
        'frozen_at': start.isoformat(), 'source_observed_at': inventory['observed_at'],
        'inventory_reference': inventory_ref, 'producer_preparation': health['preparation'],
        'schedule_amendment_reference': plan['schedule_amendment_reference'],
        'selection_window': planning.selection_window(plan, day),
        'observed_races': rows, 'observed_races_sha256': planning._digest(rows),
        'selected_race_ids': population['first_six_race_ids'],
        'coverage_basis': 'ALL_RETAINED_DAILY_DISCOVERY_ROWS_BEFORE_WIN_QUALIFICATION'}
    original_ref = runtime.put_new(day_root/'original-population.json', original)
    completed = runtime.utc_now()
    window = planning.selection_window(plan, day)
    if not instant(window['freeze_at']) <= completed < instant(window['freeze_at'])+timedelta(minutes=1):
        raise ValueError('FREEZE_CROSSED_CUTOFF')
    runtime.put_new(day_root/'original-population.json.completion.json', {'status': 'POPULATION_FROZEN',
        'population_sha256': original_ref['sha256'], 'completed_at': completed.isoformat()})
    return population


def tick(config_reference):
    config = Reader().json(config_reference)
    plan = Reader().json(config['plan'])
    planning._validate_plan(plan)
    if plan['schema_version'] == 'prospective_sectional_plan_v2':
        if Reader().json(plan['schedule_amendment_reference']) != plan['schedule_amendment']:
            raise ValueError('SCHEDULE_AMENDMENT_CHANGED')
        allocation = Reader().json(plan['authority']['allocation'])
        if (allocation.get('status') != 'AUTHORIZED' or allocation.get('allocation_id') != plan['allocation_id']
                or allocation.get('dates') != plan['dates']):
            raise ValueError('EARLIER_DEVELOPMENT_ALLOCATION_CHANGED')
    activation = Reader().json(config['activation'])
    if (activation.get('status') != 'AUTHORIZED_PROSPECTIVE_DEVELOPMENT'
            or activation.get('plan_sha256') != config['plan']['sha256']
            or activation.get('development_precedence_verified') is not True
            or activation.get('competing_pilot_disabled') is not True
            or activation.get('result_retention_routing_verified') is not True
            or activation.get('additional_source_requests') != 0
            or activation.get('additional_result_requests') != 0):
        raise ValueError('DEVELOPMENT_ALLOCATION_AND_OWNERSHIP_NOT_VERIFIED')
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
        window = planning.selection_window(plan, day)
        freeze_at = instant(window['freeze_at'])
        population_path = day_root / 'population.json'
        date_status = day_root / 'date-accounting.json'
        if not population_path.exists():
            if local < freeze_at:
                return {'status': 'WAIT_FOR_POPULATION_FREEZE'}
            if date_status.exists():
                return {'status': 'DATE_ALREADY_CONSUMED'}
            if local >= freeze_at+timedelta(minutes=1):
                runtime.put_new(date_status, {'local_date': day, 'status': 'FREEZE_INTERRUPTED',
                    'reason': 'NO_IMMUTABLE_POPULATION_BY_DECLARED_FREEZE_END', 'population_sha256': None})
                return {'status': 'MISSED_FREEZE'}
            try:
                snapshot = _protected_snapshot(config['protected_membership_journal'])
                protected_ref = runtime.put_new(day_root / 'protected-membership.json', snapshot)
                allocation = plan['authority']['allocation']
                if plan['schema_version'] == 'prospective_sectional_plan_v2':
                    population = _freeze_daily_inventory(config, plan, producer, day, day_root,
                        protected_ref, snapshot['race_ids'])
                else:
                    if config.get('index_location') == 'CURRENT_NATIVE_PACKAGE':
                        _, current_source, _ = _producer_view(producer, day)
                        evidence_root = Path(current_source['evidence_root'])
                        index_path = evidence_root / 'shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json'
                    else:
                        evidence_root, index_path = config['index_evidence_root'], config['current_index']
                    original = freeze_population(index_path, evidence_root,
                        allocation['path'], allocation['sha256'], day_root / 'original-population.json')
                    population = planning.select_population(plan, original['observed_races'], local_date=day,
                        frozen_at=original['frozen_at'], source_observed_at=original['source_observed_at'],
                        index_complete=True, protected_membership_reference=protected_ref,
                        protected_race_ids=snapshot['race_ids'])
                population_ref = runtime.put_new(population_path, population)
                completed = runtime.utc_now()
                if not freeze_at <= completed < freeze_at+timedelta(minutes=1):
                    raise ValueError('FREEZE_CROSSED_CUTOFF')
                runtime.put_new(date_status, {'local_date': day, 'status': 'POPULATION_FROZEN',
                    'population_sha256': planning._digest(population), 'population': population_ref,
                    'freeze_completed_at': completed.isoformat()})
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
        try:
            preparation, source_plan, comparison = _producer_view(producer, day)
        except (FileNotFoundError, ValueError):
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
