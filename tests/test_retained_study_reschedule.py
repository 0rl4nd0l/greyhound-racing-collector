"""Explicit pre-consumption reservation rescheduling at the public observer seam."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

import pytest

from tests.test_retained_study_observer import case, put, reserved_case


@pytest.fixture
def rescheduled_case(reserved_case):
    observer, cfg, protocol, now, claim, bundle, old = reserved_case
    observer.observe(cfg, now=datetime(2026, 10, 6, 9, tzinfo=timezone.utc))
    root = bundle.parent.parent
    previous = put(root/'prior-installed-config.json', cfg)
    windows = [
        {'local_date': '2026-10-06', 'freeze_at': '2026-10-06T22:00:00+11:00',
         'jump_start': '2026-10-06T22:20:00+11:00', 'jump_end': '2026-10-06T23:59:00+11:00'},
        {'local_date': '2026-10-07', 'freeze_at': '2026-10-07T10:00:00+11:00',
         'jump_start': '2026-10-07T10:20:00+11:00', 'jump_end': '2026-10-07T12:00:00+11:00'}]
    authority = put(root/'reschedule-user.json', {
        'schema_version': 'development_reschedule_authorization_v1', 'status': 'AUTHORIZED',
        'recorded_at': '2026-10-06T21:00:00+11:00', 'authority_reference': 'user:20261006:earlier-development',
        'predecessor_allocation': old['allocation'], 'predecessor_reservations': cfg['development_reservations'],
        'selection_windows': windows, 'maximum_per_date': 6, 'maximum_total': 12,
        'preserve_existing_study_members': True, 'preserve_consumed_allowances': True})
    allocation = put(root/'earlier-allocation.json', {'schema_version': 'development_allocation_v2',
        'status': 'AUTHORIZED', 'allocation_id': 'development-single-snapshot-20261003-v1',
        'approval': authority, 'predecessor_allocation': old['allocation'],
        'dates': ['2026-10-06', '2026-10-07'], 'selection_windows': windows,
        'max_attempts_per_date': 6, 'max_capture_attempts': 12})
    amendment = put(root/'earlier-exclusive.json', {'schema_version': 'development_reservation_amendment_v2',
        'status': 'AUTHORIZED', 'approval': authority, 'predecessor_amendment': old['exclusive_amendment'],
        'development_allocation_id': 'development-single-snapshot-20261003-v1',
        'candidate_local_dates': ['2026-10-06', '2026-10-07'], 'selection_windows': windows})
    plan = json.loads(Path(old['plan']['path']).read_bytes())
    plan.update(schema_version='prospective_sectional_plan_v2', frozen_at='2026-10-06T21:30:00+11:00',
        allocation_id='development-single-snapshot-20261003-v1', dates=['2026-10-06', '2026-10-07'],
        selection_windows=windows)
    plan['authority'].update(allocation=allocation, exclusive_amendment=amendment, current_user=authority)
    plan['population'] = {'max_index_age_seconds': 1800, 'maximum_per_date': 6, 'maximum_total': 12,
        'selection_policy': 'first_six_declared_daily_inventory_window_before_WIN_qualification_v2',
        'no_replacement_after_failure': True, 'protect_existing_study_members': True,
        'source': 'COMPLETE_RETAINED_DAILY_INVENTORY'}
    schedule = {'schema_version': 'prospective_speed_schedule_amendment_v1',
        'status': 'AUTHORIZED_EARLIER_DEVELOPMENT_SCHEDULE', 'selection_windows': windows,
        'maximum_total': 12, 'maximum_per_date': 6, 'beta': .1,
        'additional_source_requests': 0, 'additional_result_requests': 0,
        'preserve_existing_membership': True}
    plan.update(schedule_amendment=schedule, schedule_amendment_reference=put(root/'schedule-amendment.json', schedule),
        schedule_amendment_sha256=hashlib.sha256(observer.canonical(schedule)).hexdigest())
    value = {'schema_version': 'retained_study_development_reservations_v2',
        'status': 'AUTHORIZED_RESCHEDULED_DEVELOPMENT_RESERVATIONS', 'allocation': allocation,
        'exclusive_amendment': amendment, 'reschedule_authorization': authority,
        'plan': put(root/'earlier-plan.json', plan), 'state_root': str(root/'earlier-speed'),
        'dates': plan['dates'], 'predecessor_observer_config': previous,
        'predecessor_reservations': cfg['development_reservations']}
    cfg = {**cfg, 'development_reservations': put(root/'earlier-reservations.json', value),
           'study_amendment': {'path': '/synthetic/earlier-study-receipt.json', 'sha256': '9'*64}}
    return observer, cfg, protocol, now, claim, bundle, old, value


def test_explicit_unconsumed_supersession_preserves_original_journal_and_releases_old_scope(rescheduled_case):
    observer, cfg, protocol, now, claim, bundle, old, value = rescheduled_case
    journal = Path(protocol['state_root'])/'events.jsonl'
    before = journal.read_bytes()
    result = observer.observe(cfg, now=now)
    assert result['members'] == 1
    assert journal.read_bytes().startswith(before)
    assert 'DEVELOPMENT_RESERVATION_SUPERSESSION' in journal.read_text()
    before = journal.read_bytes()
    assert observer.observe(cfg, now=now)['new_members'] == 0
    assert journal.read_bytes() == before


@pytest.mark.parametrize('consumed', ['2026-10-10/date-accounting.json', '2026-10-10/population.json',
                                     'attempts/race/claim.json', 'result-controller/evaluation.json'])
def test_supersession_rejects_any_consumed_original_scope(rescheduled_case, consumed):
    observer, cfg, protocol, now, claim, bundle, old, value = rescheduled_case
    journal = Path(protocol['state_root'])/'events.jsonl'
    before = journal.read_bytes()
    put(Path(old['state_root'])/consumed, {'retained': True})
    with pytest.raises(ValueError, match='PREDECESSOR_ALREADY_CONSUMED'):
        observer.observe(cfg, now=now)
    assert journal.read_bytes() == before


def move_native(case, jump_at):
    observer, cfg, protocol, now, claim, bundle, old, value = case
    jump = datetime.fromisoformat(jump_at)
    race = 'Race 1 - SYNTHETIC - ' + jump.date().isoformat()
    admission = json.loads((claim/'admission.json').read_bytes())
    completion = json.loads((claim/'completion.json').read_bytes())
    admission['race'].update(race_id=race, race_date=jump.date().isoformat(), jump_timestamp=jump_at)
    admission.update(admitted_at=(jump-timedelta(minutes=9)).isoformat(), decision_at=(jump-timedelta(minutes=2)).isoformat())
    destination = claim.parent/hashlib.sha256(race.encode()).hexdigest()
    claim.rename(destination)
    ar = put(destination/'admission.json', admission)
    request = json.loads((bundle/'request.json').read_bytes())
    request.update(race_id=race, jump_timestamp=jump_at)
    rr = put(bundle/'request.json', request)
    manifest = json.loads((bundle/'bundle_manifest.json').read_bytes())
    manifest['files']['request.json'] = {'sha256': rr['sha256'], 'bytes': (bundle/'request.json').stat().st_size}
    mr = put(bundle/'bundle_manifest.json', manifest)
    completion.update(admission, admission_sha256=ar['sha256'], published_complete_at=(jump-timedelta(minutes=8)).isoformat())
    completion['bundle_entry']['manifest_sha256'] = mr['sha256']
    put(destination/'completion.json', completion)
    return jump-timedelta(minutes=6)


def test_new_evening_window_is_held_while_its_freeze_is_unknown(rescheduled_case):
    observer, cfg, protocol, now, claim, bundle, old, value = rescheduled_case
    now = move_native(rescheduled_case, '2026-10-06T22:45:00+11:00')
    result = observer.observe(cfg, now=now)
    assert result['members'] == 0 and result['pending_development_reservations'] == 1


def test_supersession_cannot_be_removed_after_adoption(rescheduled_case):
    observer, cfg, protocol, now, claim, bundle, old, value = rescheduled_case
    observer.observe(cfg, now=now)
    original_cfg = json.loads(Path(value['predecessor_observer_config']['path']).read_bytes())
    journal = Path(protocol['state_root'])/'events.jsonl'
    before = journal.read_bytes()
    with pytest.raises(ValueError, match='SUPERSESSION_REMOVED'):
        observer.observe(original_cfg, now=now)
    assert journal.read_bytes() == before


def freeze_earlier(case, selected=True):
    observer, cfg, protocol, now, claim, bundle, old, value = case
    now = move_native(case, '2026-10-06T22:45:00+11:00')
    observer.observe(cfg, now=datetime(2026, 10, 6, 10, 59, tzinfo=timezone.utc))
    root = Path(value['state_root'])/'2026-10-06'
    journal = Path(protocol['state_root'])/'events.jsonl'
    entries = [json.loads(line) for line in journal.read_bytes().splitlines()]
    snapshot = put(root/'protected-membership.json', {'source': observer.reference(journal),
        'race_ids': [], 'last_chain_sha256': entries[-1]['sha256']})
    races = [{'race_number': 1, 'venue': 'SYNTHETIC', 'date': '2026-10-06',
        'scheduled_jump_datetime': '2026-10-06T22:45:00+11:00',
        'url': 'https://www.thedogs.com.au/racing/synthetic/2026-10-06/1'}]
    if not selected:
        races += [{'race_number': i+2, 'venue': 'SYNTHETIC', 'date': '2026-10-06',
            'scheduled_jump_datetime': f'2026-10-06T22:{20+i}:00+11:00',
            'url': f'https://www.thedogs.com.au/racing/synthetic/2026-10-06/{i+2}'} for i in range(6)]
    races.append({'race_number': 9, 'venue': 'SYNTHETIC-OTHER', 'date': '2026-10-06',
        'scheduled_jump_datetime': None, 'url': 'https://www.thedogs.com.au/racing/synthetic-other/2026-10-06/9'})
    inventory = {'schema_version': 'daily_race_inventory_v1', 'status': 'COMPLETE',
        'source_date': '2026-10-06', 'observed_at': '2026-10-06T21:35:00+11:00',
        'races': races, 'race_count': len(races), 'discovery_failures': []}
    source_root = bundle.parent.parent/'producer'
    source = put(source_root/'inventories/census.json', inventory)
    preparation = put(source_root/'preparation.json', {'racing_date': '2026-10-06', 'output': str(source_root)})
    rows = [{'race_id': f"Race {r['race_number']} - {r['venue']} - 2026-10-06",
        'race_key': f"2026-10-06|{r['venue']}|{r['race_number']}",
        'jump_at': r['scheduled_jump_datetime'], 'url': r['url'], 'source_row_index': i} for i, r in enumerate(races)]
    selected_ids = ['Race 1 - SYNTHETIC - 2026-10-06'] if selected else [
        f'Race {i+2} - SYNTHETIC - 2026-10-06' for i in range(6)]
    digest = lambda v: hashlib.sha256(observer.canonical(v)).hexdigest()
    plan = json.loads(Path(value['plan']['path']).read_bytes())
    original = {'schema_version': 'development_population_freeze_v2', 'synthetic': False,
        'allocation_id': plan['allocation_id'], 'allocation_sha256': value['allocation']['sha256'],
        'selection_policy': plan['population']['selection_policy'], 'local_date': '2026-10-06',
        'frozen_at': '2026-10-06T22:00:01+11:00', 'source_observed_at': inventory['observed_at'],
        'inventory_reference': source, 'producer_preparation': preparation,
        'schedule_amendment_reference': plan['schedule_amendment_reference'],
        'selection_window': plan['selection_windows'][0], 'observed_races': rows,
        'observed_races_sha256': digest(rows), 'selected_race_ids': selected_ids,
        'coverage_basis': 'ALL_RETAINED_DAILY_DISCOVERY_ROWS_BEFORE_WIN_QUALIFICATION'}
    original_ref = put(root/'original-population.json', original)
    put(root/'original-population.json.completion.json', {'status': 'POPULATION_FROZEN',
        'population_sha256': original_ref['sha256'], 'completed_at': '2026-10-06T22:00:02+11:00'})
    population = {'schema_version': 'prospective_sectional_population_v1', 'plan_sha256': digest(plan),
        'local_date': '2026-10-06', 'frozen_at': original['frozen_at'], 'source_observed_at': inventory['observed_at'],
        'index_complete': True, 'observed_races_sha256': digest(rows), 'observed_races': rows,
        'first_six_race_ids': selected_ids, 'selected_race_ids': selected_ids,
        'protected_membership_reference': snapshot, 'protected_race_ids': [],
        'dispositions': [{'race_id': r['race_id'], 'jump_at': r['jump_at'], 'disposition':
            'MISSING_JUMP_TIME' if r['jump_at'] is None else 'SELECTED' if r['race_id'] in selected_ids
            else 'BEYOND_FIRST_SIX'} for r in rows]}
    pref = put(root/'population.json', population)
    put(root/'date-accounting.json', {'local_date': '2026-10-06', 'status': 'POPULATION_FROZEN',
        'population_sha256': digest(population), 'population': pref, 'freeze_completed_at': '2026-10-06T22:00:03+11:00'})
    return root, now


@pytest.mark.parametrize('selected', [True, False])
def test_declared_windows_use_complete_daily_inventory_before_qualification(rescheduled_case, selected):
    observer, cfg, protocol, now, claim, bundle, old, value = rescheduled_case
    root, now = freeze_earlier(rescheduled_case, selected)
    result = observer.observe(cfg, now=now)
    assert result['members'] == (0 if selected else 1)
    assert result['pending_development_reservations'] == 0
    assert 'MISSING_JUMP_TIME' in (root/'population.json').read_text()


@pytest.mark.parametrize('target', ['inventory_reference', 'producer_preparation'])
def test_changed_retained_inventory_or_producer_binding_holds(rescheduled_case, target):
    observer, cfg, protocol, now, claim, bundle, old, value = rescheduled_case
    root, now = freeze_earlier(rescheduled_case)
    original = json.loads((root/'original-population.json').read_bytes())
    Path(original[target]['path']).write_bytes(b'{}')
    journal = Path(protocol['state_root'])/'events.jsonl'
    before = journal.read_bytes()
    with pytest.raises(ValueError):
        observer.observe(cfg, now=now)
    assert journal.read_bytes() == before



def test_reschedule_preserves_an_already_admitted_member_in_new_window(rescheduled_case):
    observer, cfg, protocol, now, claim, bundle, old, value = rescheduled_case
    now = move_native(rescheduled_case, '2026-10-06T22:45:00+11:00')
    prior_cfg = json.loads(Path(value['predecessor_observer_config']['path']).read_bytes())
    assert observer.observe(prior_cfg, now=now)['members'] == 1
    journal = Path(protocol['state_root'])/'events.jsonl'
    before = journal.read_bytes()
    result = observer.observe(cfg, now=now)
    assert result['members'] == 1 and result['new_members'] == 0
    assert journal.read_bytes().startswith(before)


@pytest.mark.parametrize('defect', ['stale', 'late_completion', 'unknown_rows_removed'])
def test_rescheduled_census_semantics_fail_closed_even_with_rebound_hashes(rescheduled_case, defect):
    observer, cfg, protocol, now, claim, bundle, old, value = rescheduled_case
    root, now = freeze_earlier(rescheduled_case)
    population = json.loads((root/'population.json').read_bytes())
    account = json.loads((root/'date-accounting.json').read_bytes())
    if defect == 'stale':
        population['source_observed_at'] = '2026-10-06T21:30:00+11:00'
    elif defect == 'late_completion':
        account['freeze_completed_at'] = '2026-10-06T22:01:00+11:00'
    else:
        # A self-consistent filtered census still conflicts with full discovery.
        population['observed_races'].pop()
        population['dispositions'].pop()
        population['observed_races_sha256'] = hashlib.sha256(observer.canonical(population['observed_races'])).hexdigest()
        original = json.loads((root/'original-population.json').read_bytes())
        original['observed_races'] = population['observed_races']
        original['observed_races_sha256'] = population['observed_races_sha256']
        original_ref = put(root/'original-population.json', original)
        completion = json.loads((root/'original-population.json.completion.json').read_bytes())
        completion['population_sha256'] = original_ref['sha256']
        put(root/'original-population.json.completion.json', completion)
    account['population'] = put(root/'population.json', population)
    account['population_sha256'] = hashlib.sha256(observer.canonical(population)).hexdigest()
    put(root/'date-accounting.json', account)
    journal = Path(protocol['state_root'])/'events.jsonl'
    before = journal.read_bytes()
    with pytest.raises(ValueError):
        observer.observe(cfg, now=now)
    assert journal.read_bytes() == before


def test_empty_precreated_attempt_and_closure_directories_are_unconsumed(rescheduled_case):
    observer, cfg, protocol, now, claim, bundle, old, value = rescheduled_case
    for name in ('attempts', 'closure', 'result-controller'):
        (Path(old['state_root'])/name).mkdir(parents=True, exist_ok=True)
    assert observer.observe(cfg, now=now)['members'] == 1


@pytest.mark.parametrize('entry', ['attempts/claim.json', 'closure/claim.json', 'attempts'])
def test_precreated_paths_must_be_empty_directories(rescheduled_case, entry):
    observer, cfg, protocol, now, claim, bundle, old, value = rescheduled_case
    put(Path(old['state_root'])/entry, {'consumed': True})
    with pytest.raises(ValueError, match='PREDECESSOR_ALREADY_CONSUMED'):
        observer.observe(cfg, now=now)
