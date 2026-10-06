"""Identity-only routing for the original first-six development reservations."""
from datetime import datetime, timedelta
import hashlib
import json
import re
from pathlib import Path
from zoneinfo import ZoneInfo

from race_collection.development_examples import selected_population

DATES = ['2026-10-10', '2026-10-11']
PILOT_DATES = ['2026-10-03', '2026-10-04', *DATES]
ALLOCATION_ID = 'development-single-snapshot-20261003-v1'
AUTHORITY = 'user:20260930:approved-development-single-snapshot-20261003-v1'
POLICY = 'first_six_1310_1420_melbourne_before_WIN_qualification_v1'
ZONE = ZoneInfo('Australia/Melbourne')
FAILED_FREEZES = {'INDEX_MISSING', 'INDEX_STALE', 'INDEX_INCOMPLETE',
                 'FREEZE_INTERRUPTED', 'SOURCE_OR_AUTHORITY_UNAVAILABLE'}


def require(condition, reason='DEVELOPMENT_RESERVATION_AUTHORITY_INVALID'):
    if not condition:
        raise ValueError(reason)


def load(cfg, checked, root_path):
    ref = cfg.get('development_reservations')
    if ref is None:
        return None
    value = checked(ref)
    if value.get('schema_version') == 'retained_study_development_reservations_v2':
        return _load_rescheduled(value, cfg, checked, root_path)
    allocation = checked(value['allocation'])
    amendment = checked(value['exclusive_amendment'])
    approval = checked(allocation['approval'])
    plan = checked(value['plan'])
    predecessor = checked(value['predecessor_observer_config'])
    allowed = {'source_commit', 'study_amendment', 'development_reservations'}
    require({k: v for k, v in cfg.items() if k not in allowed} ==
            {k: v for k, v in predecessor.items() if k not in allowed},
            'DEVELOPMENT_RESERVATION_SUCCESSOR_CHANGED')
    require('development_reservations' not in predecessor)
    require(value['schema_version'] == 'retained_study_development_reservations_v1'
        and value['status'] == 'AUTHORIZED_ORIGINAL_DEVELOPMENT_RESERVATIONS'
        and value['dates'] == DATES
        and allocation['schema_version'] == 'development_allocation_v1'
        and allocation['status'] == amendment['status'] == 'AUTHORIZED'
        and allocation['allocation_id'] == amendment['development_allocation_id'] == ALLOCATION_ID
        and allocation['authority_reference'] == amendment['authority_reference'] == AUTHORITY
        and allocation['dates'] == amendment['candidate_local_dates'] == PILOT_DATES
        and allocation['selection_policy'] == amendment['selection_policy'] == POLICY
        and allocation['max_attempts_per_date'] == 6
        and amendment['schema_version'] == 'development_reservation_amendment_v1'
        and amendment['prior_allocation_sha256'] ==
            'b708fa973aa972b8cd248b4b4d3269fa7aa16402755ee0fb84da5212db6822d1'
        and value['exclusive_amendment'] in allocation['reservation_amendments']
        and amendment['approval'] == allocation['approval']
        and approval['schema_version'] == 'development_pilot_user_approval_v1'
        and approval['status'] == 'APPROVED' and approval['authority_reference'] == AUTHORITY
        and plan['schema_version'] == 'prospective_sectional_plan_v1'
        and plan['authority']['allocation'] == value['allocation']
        and plan['authority']['exclusive_amendment'] == value['exclusive_amendment']
        and plan['dates'] == DATES and plan['timezone'] == 'Australia/Melbourne'
        and plan['allocation_id'] == ALLOCATION_ID
        and plan['population'] == {'freeze_local_time': '12:50', 'max_index_age_seconds': 300,
            'selection_policy': POLICY, 'maximum_per_date': 6, 'maximum_total': 12,
            'no_replacement_after_failure': True, 'protect_existing_study_members': True})
    value = {**value, 'reference': ref, 'plan_value': plan, 'predecessor': predecessor}
    value['root'] = root_path(value['state_root'])
    value['journal_predecessor'] = predecessor
    return value



def _instant(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(result.utcoffset() is not None, 'DEVELOPMENT_WINDOW_INVALID')
    return result


def _load_rescheduled(value, cfg, checked, root_path):
    previous = checked(value['predecessor_observer_config'])
    require(previous.get('development_reservations') == value['predecessor_reservations'],
            'DEVELOPMENT_RESCHEDULE_PREDECESSOR_CHANGED')
    prior_document = checked(value['predecessor_reservations'])
    require(prior_document['schema_version'] == 'retained_study_development_reservations_v1')
    prior = load(previous, checked, root_path)
    allowed = {'source_commit', 'study_amendment', 'development_reservations'}
    require({k: v for k, v in cfg.items() if k not in allowed} ==
            {k: v for k, v in previous.items() if k not in allowed},
            'DEVELOPMENT_RESERVATION_SUCCESSOR_CHANGED')
    authority = checked(value['reschedule_authorization'])
    allocation = checked(value['allocation'])
    amendment = checked(value['exclusive_amendment'])
    plan = checked(value['plan'])
    windows = plan['selection_windows']
    schedule = checked(plan['schedule_amendment_reference'])
    require(schedule == plan['schedule_amendment']
        and hashlib.sha256(json.dumps(schedule, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() == plan['schedule_amendment_sha256']
        and schedule['schema_version'] == 'prospective_speed_schedule_amendment_v1'
        and schedule['status'] == 'AUTHORIZED_EARLIER_DEVELOPMENT_SCHEDULE'
        and schedule['selection_windows'] == windows
        and schedule['maximum_total'] == 12 and schedule['maximum_per_date'] == 6
        and schedule['beta'] == .1 and schedule['additional_source_requests'] == schedule['additional_result_requests'] == 0
        and schedule['preserve_existing_membership'] is True, 'DEVELOPMENT_SCHEDULE_CHANGED')
    rescheduled_fields = {'schema_version', 'frozen_at', 'authority', 'dates', 'population',
        'selection_windows', 'schedule_amendment_reference', 'schedule_amendment',
        'schedule_amendment_sha256', 'result_requests_stop_at', 'evaluation_at', 'horizon_basis'}
    require({k: v for k, v in plan.items() if k not in rescheduled_fields} ==
            {k: v for k, v in prior['plan_value'].items() if k not in rescheduled_fields},
            'DEVELOPMENT_FROZEN_EXPERIMENT_CHANGED')
    dates = [w['local_date'] for w in windows]
    require(value['status'] == 'AUTHORIZED_RESCHEDULED_DEVELOPMENT_RESERVATIONS'
        and len(windows) == 2 and len(set(dates)) == 2 and dates == sorted(dates)
        and dates == value['dates'] == plan['dates'] == allocation['dates'] == amendment['candidate_local_dates']
        and authority['schema_version'] == 'development_reschedule_authorization_v1'
        and authority['status'] == 'AUTHORIZED' and bool(authority['authority_reference'])
        and authority['predecessor_allocation'] == prior['allocation']
        and authority['predecessor_reservations'] == value['predecessor_reservations']
        and authority['selection_windows'] == allocation['selection_windows'] == amendment['selection_windows'] == windows
        and authority['maximum_per_date'] == 6 and authority['maximum_total'] == 12
        and authority['preserve_existing_study_members'] is True
        and authority['preserve_consumed_allowances'] is True
        and allocation['schema_version'] == 'development_allocation_v2'
        and allocation['status'] == amendment['status'] == 'AUTHORIZED'
        and allocation['approval'] == amendment['approval'] == value['reschedule_authorization']
        and allocation['predecessor_allocation'] == prior['allocation']
        and allocation['max_attempts_per_date'] == 6 and allocation['max_capture_attempts'] == 12
        and amendment['schema_version'] == 'development_reservation_amendment_v2'
        and amendment['predecessor_amendment'] == prior['exclusive_amendment']
        and allocation['allocation_id'] == amendment['development_allocation_id'] == plan['allocation_id'] == ALLOCATION_ID
        and plan['schema_version'] == 'prospective_sectional_plan_v2'
        and plan['timezone'] == 'Australia/Melbourne'
        and plan['authority']['allocation'] == value['allocation']
        and plan['authority']['exclusive_amendment'] == value['exclusive_amendment']
        and plan['authority']['current_user'] == value['reschedule_authorization']
        and plan['population'] == {'max_index_age_seconds': 1800,
            'selection_policy': 'first_six_declared_daily_inventory_window_before_WIN_qualification_v2',
            'maximum_per_date': 6, 'maximum_total': 12,
            'no_replacement_after_failure': True, 'protect_existing_study_members': True,
            'source': 'COMPLETE_RETAINED_DAILY_INVENTORY'})
    for window in windows:
        freeze, start, end = (_instant(window[k]) for k in ('freeze_at', 'jump_start', 'jump_end'))
        require(set(window) == {'local_date', 'freeze_at', 'jump_start', 'jump_end'}
            and all(t.astimezone(ZONE).date().isoformat() == window['local_date'] for t in (freeze, start, end))
            and freeze.second == freeze.microsecond == 0
            and freeze + timedelta(minutes=20) <= start <= end
            and _instant(authority['recorded_at']) <= _instant(plan['frozen_at']) < freeze,
            'DEVELOPMENT_WINDOW_INVALID')
    root = root_path(value['state_root'])
    require(root != prior['root'] and not root.is_relative_to(prior['root'])
        and not prior['root'].is_relative_to(root), 'DEVELOPMENT_RESCHEDULE_STATE_OVERLAP')
    _unconsumed(prior, root_path)
    return {**value, 'reference': cfg['development_reservations'], 'plan_value': plan,
            'predecessor': previous, 'prior_reservation': prior, 'root': root,
            'journal_predecessor': prior['journal_predecessor']}


def _unconsumed(prior, root_path):
    # The withdrawn operation must retain its complete empty state. A populated
    # date, claim, result or evaluation record cannot be abandoned by rescheduling.
    root = prior['root']
    directories = {'result-controller', 'attempts', 'closure'}
    locks = {'worker.lock', 'result-controller/worker.lock'}
    if root.exists():
        for path in root.rglob('*'):
            root_path(str(path))
            name = path.relative_to(root).as_posix()
            require((name in directories and path.is_dir()) or (name in locks and path.is_file()),
                    'DEVELOPMENT_PREDECESSOR_ALREADY_CONSUMED')


def _binding(value, cfg):
    return {'kind': 'DEVELOPMENT_RESERVATION_BINDING', 'reference': value['reference'],
            'predecessor_config': value['predecessor_observer_config'],
            'successor_amendment': cfg['study_amendment'], 'source_commit': cfg.get('source_commit')}


def bind(value, cfg, journal):
    """Preserve the original journal identity and append the checked successor."""
    bindings = [r for r in journal.events if r['kind'] == 'DEVELOPMENT_RESERVATION_BINDING']
    if value is None:
        require(not bindings, 'DEVELOPMENT_RESERVATION_BINDING_REMOVED')
        return
    previous = value['journal_predecessor']
    original = {'kind': 'IDENTITY', 'protocol': previous['retained_study_protocol'],
                'amendment': previous['study_amendment']}
    require(journal.events[0] == original, 'DEVELOPMENT_RESERVATION_PREDECESSOR_CHANGED')
    supersessions = [r for r in journal.events if r['kind'] == 'DEVELOPMENT_RESERVATION_SUPERSESSION']
    if 'prior_reservation' in value:
        require(bindings == [_binding(value['prior_reservation'], value['predecessor'])],
                'DEVELOPMENT_RESCHEDULE_PREDECESSOR_CHANGED')
        require(not any(r['kind'] == 'DEVELOPMENT_RESERVATION_DATE' and
                        Path(r['evidence'][0]['path']).is_relative_to(value['prior_reservation']['root'])
                        for r in journal.events),
                'DEVELOPMENT_PREDECESSOR_ALREADY_CONSUMED')
        event = {'kind': 'DEVELOPMENT_RESERVATION_SUPERSESSION',
            'predecessor_reference': value['predecessor_reservations'],
            'reference': value['reference'], 'authorization': value['reschedule_authorization'],
            'predecessor_config': value['predecessor_observer_config'],
            'successor_amendment': cfg['study_amendment'], 'source_commit': cfg.get('source_commit'),
            'predecessor_disposition': 'SUPERSEDED_UNCONSUMED'}
        require(not supersessions or supersessions == [event], 'DEVELOPMENT_RESERVATION_BINDING_CHANGED')
        if not supersessions:
            journal.append(event)
        return
    require(not supersessions, 'DEVELOPMENT_RESERVATION_SUPERSESSION_REMOVED')
    event = _binding(value, cfg)
    require(not bindings or bindings == [event], 'DEVELOPMENT_RESERVATION_BINDING_CHANGED')
    if not bindings:
        journal.append(event)



def _digest(value, canonical):
    return hashlib.sha256(canonical(value)).hexdigest()


def _snapshot(population, checked, journal, canonical):
    snapshot = checked(population['protected_membership_reference'])
    require(snapshot['source']['path'] == str(journal.path), 'DEVELOPMENT_PROTECTED_JOURNAL_CHANGED')
    # The source journal is append-only, so authenticate its historical prefix.
    prefix = hashlib.sha256()
    previous, protected = '0' * 64, set()
    found = snapshot['source']['sha256'] == prefix.hexdigest()
    for index, event in enumerate(journal.events):
        if found:
            break
        row = {'sequence': index, 'previous': previous, 'event': event}
        row['sha256'] = _digest(row, canonical)
        prefix.update(canonical(row) + b'\n')
        previous = row['sha256']
        if event['kind'] == 'MEMBER':
            protected.add(event['race_id'])
        found = snapshot['source']['sha256'] == prefix.hexdigest()
    require(found and snapshot['last_chain_sha256'] == previous
        and snapshot['race_ids'] == sorted(protected)
        and population['protected_race_ids'] == sorted(protected),
        'DEVELOPMENT_PROTECTED_JOURNAL_CHANGED')
    return protected



def _window(value, day):
    if 'prior_reservation' in value:
        return next(w for w in value['plan_value']['selection_windows'] if w['local_date'] == day)
    return {'local_date': day, 'freeze_at': day+'T12:50:00+11:00',
            'jump_start': day+'T13:10:00+11:00', 'jump_end': day+'T14:20:59.999999+11:00'}



def _inventory_key(race_id):
    match = re.fullmatch(r'Race ([1-9][0-9]*) - ([A-Z0-9_]+(?:-[A-Z0-9_]+)*) - (\d{4}-\d{2}-\d{2})', race_id)
    require(match is not None, 'DEVELOPMENT_INVENTORY_IDENTITY_INVALID')
    return f'{match[3]}|{match[2]}|{int(match[1])}'


def _selected(value, rows, day):
    if 'prior_reservation' not in value:
        return selected_population(rows, day)
    window = _window(value, day)
    require(all(r['race_key'] == _inventory_key(r['race_id']) for r in rows), 'DEVELOPMENT_SELECTION_CHANGED')
    intended = sorted([r for r in rows if r['jump_at'] is not None
        and _instant(window['jump_start']) <= _instant(r['jump_at']) <= _instant(window['jump_end'])],
        key=lambda r: (_instant(r['jump_at']), r['race_id']))
    return intended, [r['race_id'] for r in intended[:6]]


def _daily_inventory_census(value, original, account, checked, rows, day, canonical):
    from race_collection.daily_race_inventory import _validate
    inventory = checked(original['inventory_reference'])
    _validate(inventory, day)
    expected = []
    for index, row in enumerate(inventory['races']):
        race = f"Race {int(row['race_number'])} - {row['venue']} - {day}"
        expected.append({'race_id': race, 'race_key': _inventory_key(race),
            'jump_at': row.get('scheduled_jump_datetime') or None, 'url': row['url'], 'source_row_index': index})
    preparation = checked(original['producer_preparation'])
    require(original['schema_version'] == 'development_population_freeze_v2'
        and original['inventory_reference']['path'].startswith(str(Path(preparation['output'])/'inventories') + '/')
        and preparation['racing_date'] == day
        and inventory['observed_at'] == original['source_observed_at']
        and original['observed_races_sha256'] == _digest(rows, canonical)
        and rows == expected and original['selection_window'] == _window(value, day)
        and original['schedule_amendment_reference'] == value['plan_value']['schedule_amendment_reference']
        and original['coverage_basis'] == 'ALL_RETAINED_DAILY_DISCOVERY_ROWS_BEFORE_WIN_QUALIFICATION',
        'DEVELOPMENT_ORIGINAL_CENSUS_CHANGED')
    window = _window(value, day)
    require(_instant(original['frozen_at']) <= _instant(account['freeze_completed_at'])
        < _instant(window['freeze_at']) + timedelta(minutes=1), 'DEVELOPMENT_FREEZE_COMPLETION_INVALID')
    return [original['inventory_reference'], original['producer_preparation']]


def states(value, checked, reference, stamp, canonical, journal, now):
    if value is None:
        return {}
    # A previously observed terminal disposition can never disappear or change.
    for event in journal.events:
        if event['kind'] == 'DEVELOPMENT_RESERVATION_DATE':
            for ref in event['evidence']:
                checked(ref)
    result = {}
    for day in value['dates']:
        root = value['root'] / day
        path = root / 'date-accounting.json'
        if not path.exists():
            continue
        account_ref = reference(path)
        account = checked(account_ref)
        require(account['local_date'] == day and day <= now.astimezone(ZONE).date().isoformat(),
                'DEVELOPMENT_DATE_DISPOSITION_INVALID')
        evidence = [account_ref]
        if account['status'] != 'POPULATION_FROZEN':
            require(account['status'] in FAILED_FREEZES
                and account.get('population_sha256') is None and 'population' not in account
                and isinstance(account.get('reason'), str) and bool(account['reason']),
                'DEVELOPMENT_DATE_DISPOSITION_INVALID')
            result[day] = {'status': account['status'], 'evidence': evidence}
            continue
        require(account['population']['path'] == str(root/'population.json'),
                'DEVELOPMENT_POPULATION_PATH_CHANGED')
        population = checked(account['population'])
        require(account['population_sha256'] == _digest(population, canonical),
                'DEVELOPMENT_POPULATION_CHANGED')
        frozen = stamp(population['frozen_at']).astimezone(ZONE)
        window = _window(value, day)
        freeze_start = stamp(window['freeze_at'])
        rows = population['observed_races']
        require(population['schema_version'] == 'prospective_sectional_population_v1'
            and population['plan_sha256'] == _digest(value['plan_value'], canonical)
            and population['local_date'] == day and frozen.date().isoformat() == day
            and freeze_start <= frozen < freeze_start + timedelta(minutes=1)
            and stamp(value['plan_value']['frozen_at']) < frozen <= now
            and population['index_complete'] is True
            and 0 <= (frozen - stamp(population['source_observed_at'])).total_seconds() <= value['plan_value']['population']['max_index_age_seconds']
            and population['observed_races_sha256'] == _digest(rows, canonical)
            and len({r['race_id'] for r in rows}) == len(rows), 'DEVELOPMENT_POPULATION_CHANGED')
        intended, first_six = _selected(value, rows, day)
        protected = _snapshot(population, checked, journal, canonical)
        selected = [race for race in first_six if race not in protected]
        intended_ids = {r['race_id'] for r in intended}
        dispositions = [{'race_id': r['race_id'], 'jump_at': r['jump_at'], 'disposition':
            'MISSING_JUMP_TIME' if r['jump_at'] is None else
            'PROTECTED_STUDY_MEMBER' if r['race_id'] in first_six and r['race_id'] in protected else
            'SELECTED' if r['race_id'] in selected else
            'BEYOND_FIRST_SIX' if r['race_id'] in intended_ids else 'OUTSIDE_SELECTION_WINDOW'} for r in rows]
        require(population['first_six_race_ids'] == first_six
            and population['selected_race_ids'] == selected and population['dispositions'] == dispositions,
            'DEVELOPMENT_SELECTION_CHANGED')
        original_ref = reference(root/'original-population.json')
        original = checked(original_ref)
        completion_ref = reference(root/'original-population.json.completion.json')
        completion = checked(completion_ref)
        if 'prior_reservation' in value:
            evidence.extend(_daily_inventory_census(value, original, account, checked, rows, day, canonical))
        else:
            require(original['schema_version'] == 'development_population_freeze_v1'
                and original['intended'] == intended, 'DEVELOPMENT_ORIGINAL_CENSUS_CHANGED')
        require(original['synthetic'] is False and original['allocation_id'] == value['plan_value']['allocation_id']
            and original['allocation_sha256'] == value['allocation']['sha256']
            and original['selection_policy'] == value['plan_value']['population']['selection_policy'] and original['local_date'] == day
            and original['frozen_at'] == population['frozen_at']
            and original['source_observed_at'] == population['source_observed_at']
            and original['observed_races'] == rows
            and original['selected_race_ids'] == first_six
            and completion['status'] == 'POPULATION_FROZEN'
            and completion['population_sha256'] == original_ref['sha256']
            and frozen <= stamp(completion['completed_at']) <= now
            and freeze_start <= stamp(completion['completed_at']) < freeze_start + timedelta(minutes=1)
            and stamp(completion['completed_at']).astimezone(ZONE).date().isoformat() == day
            and not (root/'original-population.json.failure.json').exists(),
            'DEVELOPMENT_ORIGINAL_CENSUS_CHANGED')
        if 'prior_reservation' in value:
            require(stamp(completion['completed_at']) <= stamp(account['freeze_completed_at']) <= now,
                    'DEVELOPMENT_FREEZE_COMPLETION_INVALID')
        evidence.extend([account['population'], population['protected_membership_reference'],
                         original_ref, completion_ref])
        result[day] = {'status': 'POPULATION_FROZEN', 'evidence': evidence,
                       'selected': {r['race_id']: r for r in rows if r['race_id'] in selected}}
    return result


def disposition(value, states, candidate):
    if value is None:
        return None
    local = datetime.fromisoformat(candidate['jump_at'].replace('Z', '+00:00')).astimezone(ZONE)
    day = local.date().isoformat()
    for reserved_day, state in states.items():
        row = state.get('selected', {}).get(candidate['race_id'])
        if row:
            require(datetime.fromisoformat(row['jump_at'].replace('Z', '+00:00')) == local,
                    'DEVELOPMENT_RESERVED_IDENTITY_CHANGED')
            return {'reason': 'ORIGINAL_FIRST_SIX_DEVELOPMENT_RESERVATION', 'local_date': reserved_day,
                    'date_accounting': state['evidence'][0]}
    if day not in value['dates'] or day in states:
        return None
    window = _window(value, day)
    if _instant(window['jump_start']) <= local <= _instant(window['jump_end']):
        return {'reason': 'DEVELOPMENT_SELECTION_PENDING', 'local_date': day}
    return None
