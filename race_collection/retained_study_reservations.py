"""Identity-only routing for the original first-six development reservations."""
from datetime import datetime
import hashlib
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
    return value


def bind(value, cfg, journal):
    """Preserve the original journal identity and append the checked successor."""
    bindings = [r for r in journal.events if r['kind'] == 'DEVELOPMENT_RESERVATION_BINDING']
    if value is None:
        require(not bindings, 'DEVELOPMENT_RESERVATION_BINDING_REMOVED')
        return
    previous = value['predecessor']
    original = {'kind': 'IDENTITY', 'protocol': previous['retained_study_protocol'],
                'amendment': previous['study_amendment']}
    require(journal.events[0] == original, 'DEVELOPMENT_RESERVATION_PREDECESSOR_CHANGED')
    event = {'kind': 'DEVELOPMENT_RESERVATION_BINDING', 'reference': value['reference'],
             'predecessor_config': value['predecessor_observer_config'],
             'successor_amendment': cfg['study_amendment'], 'source_commit': cfg.get('source_commit')}
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


def states(value, checked, reference, stamp, canonical, journal, now):
    if value is None:
        return {}
    # A previously observed terminal disposition can never disappear or change.
    for event in journal.events:
        if event['kind'] == 'DEVELOPMENT_RESERVATION_DATE':
            for ref in event['evidence']:
                checked(ref)
    result = {}
    for day in DATES:
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
        rows = population['observed_races']
        require(population['schema_version'] == 'prospective_sectional_population_v1'
            and population['plan_sha256'] == _digest(value['plan_value'], canonical)
            and population['local_date'] == day and frozen.date().isoformat() == day
            and frozen.strftime('%H:%M') == '12:50'
            and stamp(value['plan_value']['frozen_at']) < frozen <= now
            and population['index_complete'] is True
            and 0 <= (frozen - stamp(population['source_observed_at'])).total_seconds() <= 300
            and population['observed_races_sha256'] == _digest(rows, canonical)
            and len({r['race_id'] for r in rows}) == len(rows), 'DEVELOPMENT_POPULATION_CHANGED')
        intended, first_six = selected_population(rows, day)
        protected = _snapshot(population, checked, journal, canonical)
        selected = [race for race in first_six if race not in protected]
        intended_ids = {r['race_id'] for r in intended}
        dispositions = [{'race_id': r['race_id'], 'jump_at': r['jump_at'], 'disposition':
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
        require(original['schema_version'] == 'development_population_freeze_v1'
            and original['synthetic'] is False and original['allocation_id'] == ALLOCATION_ID
            and original['allocation_sha256'] == value['allocation']['sha256']
            and original['selection_policy'] == POLICY and original['local_date'] == day
            and original['frozen_at'] == population['frozen_at']
            and original['source_observed_at'] == population['source_observed_at']
            and original['observed_races'] == rows and original['intended'] == intended
            and original['selected_race_ids'] == first_six
            and completion['status'] == 'POPULATION_FROZEN'
            and completion['population_sha256'] == original_ref['sha256']
            and frozen <= stamp(completion['completed_at']) <= now
            and stamp(completion['completed_at']).astimezone(ZONE).strftime('%H:%M') == '12:50'
            and stamp(completion['completed_at']).astimezone(ZONE).date().isoformat() == day
            and not (root/'original-population.json.failure.json').exists(),
            'DEVELOPMENT_ORIGINAL_CENSUS_CHANGED')
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
    if day not in DATES or day in states:
        return None
    if '13:10' <= local.strftime('%H:%M') <= '14:20':
        return {'reason': 'DEVELOPMENT_SELECTION_PENDING', 'local_date': day}
    return None
