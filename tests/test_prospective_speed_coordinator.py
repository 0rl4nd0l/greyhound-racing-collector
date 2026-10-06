"""Fabricated coordinator lifecycle; no retained real producer or result reads."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

import pytest

from race_collection import prospective_speed_coordinator as coordinator
from tests.test_prospective_speed_plan import plan, races


class Clock:
    def __init__(self, value='2026-10-10T12:50:10+11:00'):
        self.value = datetime.fromisoformat(value)

    def __call__(self):
        return self.value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    return coordinator.runtime.put_new(path, value)


def fixture(tmp_path, monkeypatch, *, active=True, time='2026-10-10T12:50:10+11:00'):
    producer = tmp_path/'producer'; producer.mkdir()
    state = tmp_path/'experiment'
    plan_ref = write(tmp_path/'plan.json', plan())
    activation = {'status': 'AUTHORIZED_PROSPECTIVE_DEVELOPMENT' if active else 'PROPOSED',
        'plan_sha256': plan_ref['sha256'], 'development_precedence_verified': True,
        'competing_pilot_disabled': True, 'result_retention_routing_verified': True,
        'additional_source_requests': 0, 'additional_result_requests': 0,
        'verified_control_files': []}
    activation_ref = write(tmp_path/'activation.json', activation)
    journal = tmp_path/'protected.jsonl'; journal.write_bytes(b'')
    config = {'plan': plan_ref, 'activation': activation_ref, 'state_root': str(state),
        'producer_runtime_root': str(producer), 'protected_membership_journal': str(journal),
        'current_index': str(producer/'current-index.json'),
        'index_evidence_root': str(producer/'index'), 'allowed_source_roots': [str(producer)],
        'history_inventory': write(tmp_path/'history.json', {
            'schema_version': 'prospective_speed_history_inventory_v1', 'cards': []}),
        'model': {'path': str(tmp_path/'model.json'), 'sha256': 'a'*64}}
    ref = write(tmp_path/'config.json', config)
    clock = Clock(time)
    monkeypatch.setattr(coordinator.runtime, 'utc_now', clock)
    frozen = []

    def freeze(index, evidence, allocation, digest, output):
        frozen.append((index, evidence, allocation, digest))
        day = clock.value.date().isoformat()
        original = {'observed_races': races(day), 'frozen_at': clock.value.isoformat(),
            'source_observed_at': day+'T12:48:00+11:00'}
        write(output, original)
        return original

    monkeypatch.setattr(coordinator, 'freeze_population', freeze)
    return config, ref, clock, frozen


def producer_day(config, day='2026-10-10', *, ready=()):
    producer = Path(config['producer_runtime_root'])
    base = producer/day
    comparison_ref = write(base/'comparison-plan.json', {'programme_root': str(base/'admission')})
    source_ref = write(base/'package-plan.json', {'frozen_comparison': comparison_ref,
        'prediction_root': str(base/'predictions')})
    preparation = write(base/'preparation.json', {'racing_date': day, 'plan': source_ref})
    pointer = producer/'current-day.json'
    if pointer.exists():
        pointer.unlink()  # Fixture producer rolls its own pointer, never the coordinator.
    write(pointer, {'racing_date': day, 'preparation': preparation})
    for row in ready:
        key = hashlib.sha256(row['race_id'].encode()).hexdigest()
        attempt = base/'admission'/comparison_ref['sha256']/'attempts'/key
        write(attempt/'admission.json', row)
        write(attempt/'completion.json', {'fixture_complete': True})


def forbid_producer_reads(config, monkeypatch):
    original = coordinator.Reader.read
    producer = Path(config['producer_runtime_root'])

    def guarded(self, reference):
        assert not Path(reference['path']).is_relative_to(producer), 'unexpected producer read'
        return original(self, reference)

    monkeypatch.setattr(coordinator.Reader, 'read', guarded)


def test_missing_activation_prevents_any_producer_read_or_state_claim(tmp_path, monkeypatch):
    config, reference, _, frozen = fixture(tmp_path, monkeypatch, active=False)
    forbid_producer_reads(config, monkeypatch)
    with pytest.raises(ValueError, match='DEVELOPMENT_ALLOCATION_AND_OWNERSHIP_NOT_VERIFIED'):
        coordinator.tick(reference)
    assert frozen == []
    assert not Path(config['state_root']).exists()


def test_date_outside_window_never_opens_producer(tmp_path, monkeypatch):
    config, reference, _, frozen = fixture(tmp_path, monkeypatch, time='2026-10-09T12:50:10+11:00')
    forbid_producer_reads(config, monkeypatch)
    result = coordinator.tick(reference)
    assert result['status'] == 'NO_DEVELOPMENT_DATE_DUE'
    assert result['next_dates'] == ['2026-10-10', '2026-10-11']
    assert frozen == []


def test_before_freeze_waits_without_observing_or_consuming_population(tmp_path, monkeypatch):
    config, reference, _, frozen = fixture(tmp_path, monkeypatch, time='2026-10-10T12:49:59+11:00')
    forbid_producer_reads(config, monkeypatch)
    assert coordinator.tick(reference)['status'] == 'WAIT_FOR_POPULATION_FREEZE'
    assert frozen == []
    assert not (Path(config['state_root'])/'2026-10-10/population.json').exists()


def test_late_start_marks_missed_freeze_once_without_replacement(tmp_path, monkeypatch):
    config, reference, clock, frozen = fixture(tmp_path, monkeypatch, time='2026-10-10T12:51:00+11:00')
    forbid_producer_reads(config, monkeypatch)
    assert coordinator.tick(reference)['status'] == 'MISSED_FREEZE'
    account = Path(config['state_root'])/'2026-10-10/date-accounting.json'
    preserved = account.read_bytes()
    clock.value = datetime.fromisoformat('2026-10-10T13:00:00+11:00')
    assert coordinator.tick(reference)['status'] == 'DATE_ALREADY_CONSUMED'
    assert account.read_bytes() == preserved
    assert frozen == []


def test_frozen_population_is_not_replaced_by_later_index(tmp_path, monkeypatch):
    config, reference, clock, frozen = fixture(tmp_path, monkeypatch)
    producer_day(config)
    first = coordinator.tick(reference)
    population = Path(config['state_root'])/'2026-10-10/population.json'
    preserved = population.read_bytes()
    assert first['selected'] == 6
    clock.value = datetime.fromisoformat('2026-10-10T13:00:00+11:00')

    def later_index(*args, **kwargs):
        raise AssertionError('A later index must not replace frozen membership')

    monkeypatch.setattr(coordinator, 'freeze_population', later_index)
    assert coordinator.tick(reference)['selected'] == 6
    assert len(frozen) == 1
    assert population.read_bytes() == preserved


def test_bad_freeze_is_consumed_without_later_retry(tmp_path, monkeypatch):
    config, reference, _, frozen = fixture(tmp_path, monkeypatch)
    forbid_producer_reads(config, monkeypatch)

    def bad_index(*args, **kwargs):
        frozen.append('failed')
        raise ValueError('fabricated incomplete index')

    monkeypatch.setattr(coordinator, 'freeze_population', bad_index)
    assert coordinator.tick(reference)['status'] == 'POPULATION_FREEZE_FAILED'
    assert coordinator.tick(reference)['status'] == 'DATE_ALREADY_CONSUMED'
    assert frozen == ['failed']


def test_single_source_failure_does_not_stop_other_selected_races(tmp_path, monkeypatch):
    config, reference, clock, _ = fixture(tmp_path, monkeypatch)
    producer_day(config)
    coordinator.tick(reference)
    state = Path(config['state_root'])
    source = Path(config['producer_runtime_root'])/'2026-10-10'
    comparison_ref = coordinator.runtime.reference(source/'comparison-plan.json')
    for row in races()[:6]:
        key = hashlib.sha256(row['race_id'].encode()).hexdigest()
        attempt = source/'admission'/comparison_ref['sha256']/'attempts'/key
        write(attempt/'admission.json', row)
        write(attempt/'completion.json', {'fixture_complete': True})
    invoked, jobs = [], []

    def native(reader, bundle_root, admission, completion, **kwargs):
        row = json.loads(Path(admission['path']).read_bytes())
        invoked.append(row['race_id'])
        if row['race_id'] == races()[0]['race_id']:
            raise ValueError('fabricated conflicting native runner identity')
        return {'race_id': row['race_id'], 'jump_at': row['jump_at']}, {
            'original_published_complete_at': '2026-10-10T13:01:00+11:00'}

    def execute(job_reference, root):
        job = json.loads(Path(job_reference['path']).read_bytes())
        jobs.append((job['member']['race_id'], Path(root)))
        assert not Path(root).is_relative_to(Path(config['producer_runtime_root']))
        return {'status': 'SEALED_PREJUMP'}

    monkeypatch.setattr(coordinator.inputs, 'member_from_native', native)
    monkeypatch.setattr(coordinator.runtime, 'run_job', execute)
    clock.value = datetime.fromisoformat('2026-10-10T13:05:00+11:00')
    result = coordinator.tick(reference)
    assert result['selected'] == result['terminal'] == 6
    assert len(invoked) == 6 and len(jobs) == 5
    dispositions = [json.loads(p.read_bytes()) for p in state.glob('*/jobs/*.disposition.json')]
    assert sum(d['status'] == 'SPEED_PROCESSING_FAILED' for d in dispositions) == 1
    assert sum(d['status'] == 'SEALED_PREJUMP' for d in dispositions) == 5
    preserved = {str(p): p.read_bytes() for p in state.glob('*/jobs/*.disposition.json')}
    coordinator.tick(reference)
    assert len(invoked) == 6 and len(jobs) == 5
    assert preserved == {str(p): p.read_bytes() for p in state.glob('*/jobs/*.disposition.json')}


def test_past_races_are_accounted_without_postjump_forecasting(tmp_path, monkeypatch):
    config, reference, clock, _ = fixture(tmp_path, monkeypatch)
    producer_day(config)
    coordinator.tick(reference)
    clock.value = datetime.fromisoformat('2026-10-10T14:21:00+11:00')

    def forbidden(*args, **kwargs):
        raise AssertionError('Post-jump native input must not be forecast')

    monkeypatch.setattr(coordinator.inputs, 'member_from_native', forbidden)
    monkeypatch.setattr(coordinator.runtime, 'run_job', forbidden)
    result = coordinator.tick(reference)
    assert result['terminal'] == 6
    dispositions = [json.loads(p.read_bytes()) for p in Path(config['state_root']).glob('*/jobs/*.disposition.json')]
    assert all(d['status'] == 'UPSTREAM_FORECAST_UNAVAILABLE' for d in dispositions)


def test_modified_protected_membership_hash_chain_prevents_population(tmp_path, monkeypatch):
    config, reference, _, frozen = fixture(tmp_path, monkeypatch)
    Path(config['protected_membership_journal']).write_text(json.dumps({
        'sequence': 0, 'previous': '0'*64, 'event': {'kind': 'MEMBER', 'race_id': races()[0]['race_id']},
        'sha256': 'f'*64})+'\n')
    forbid_producer_reads(config, monkeypatch)
    assert coordinator.tick(reference)['status'] == 'POPULATION_FREEZE_FAILED'
    assert frozen == []


def test_rollover_closes_unattempted_members_without_reopening_prior_date(tmp_path, monkeypatch):
    config, reference, clock, frozen = fixture(tmp_path, monkeypatch)
    producer_day(config)
    coordinator.tick(reference)
    state = Path(config['state_root'])
    original_population = (state/'2026-10-10/population.json').read_bytes()
    clock.value = datetime.fromisoformat('2026-10-11T12:49:00+11:00')
    assert coordinator.tick(reference)['status'] == 'WAIT_FOR_POPULATION_FREEZE'
    old_dispositions = {str(p): p.read_bytes() for p in (state/'2026-10-10/jobs').glob('*.disposition.json')}
    assert len(old_dispositions) == 6
    assert all(json.loads(value)['status'] == 'UNATTEMPTED_AT_HORIZON' for value in old_dispositions.values())
    terminals = [json.loads(p.read_bytes()) for p in (state/'attempts').glob('*/terminal.json')]
    assert len(terminals) == 6
    assert all(t['status'] == 'UNATTEMPTED_AT_HORIZON' for t in terminals)
    producer_day(config, day='2026-10-11')
    clock.value = datetime.fromisoformat('2026-10-11T12:50:10+11:00')
    assert coordinator.tick(reference)['selected'] == 6
    assert len(frozen) == 2
    assert (state/'2026-10-10/population.json').read_bytes() == original_population
    assert old_dispositions == {str(p): p.read_bytes() for p in (state/'2026-10-10/jobs').glob('*.disposition.json')}


def test_start_after_horizon_accounts_for_both_missed_dates_without_source_reads(tmp_path, monkeypatch):
    config, reference, _, frozen = fixture(tmp_path, monkeypatch, time='2026-10-12T08:00:00+11:00')
    forbid_producer_reads(config, monkeypatch)
    result = coordinator.tick(reference)
    assert result['status'] == 'NO_DEVELOPMENT_DATE_DUE'
    assert result['next_dates'] == []
    accounts = {str(p): p.read_bytes() for p in Path(config['state_root']).glob('*/date-accounting.json')}
    assert len(accounts) == 2
    assert all(json.loads(value)['status'] == 'FREEZE_INTERRUPTED' for value in accounts.values())
    coordinator.tick(reference)
    assert frozen == []
    assert accounts == {str(p): p.read_bytes() for p in Path(config['state_root']).glob('*/date-accounting.json')}


def test_interrupted_freeze_without_account_is_consumed_not_reconstructed(tmp_path, monkeypatch):
    config, reference, _, frozen = fixture(tmp_path, monkeypatch)
    Path(config['state_root']).mkdir(mode=0o700)
    population = Path(config['state_root'])/'2026-10-10/population.json'
    write(population, {'fixture': 'interrupted population bytes'})
    forbid_producer_reads(config, monkeypatch)
    assert coordinator.tick(reference)['status'] == 'DATE_ALREADY_CONSUMED'
    account = json.loads(population.with_name('date-accounting.json').read_bytes())
    assert account['status'] == 'FREEZE_INTERRUPTED'
    assert frozen == []
