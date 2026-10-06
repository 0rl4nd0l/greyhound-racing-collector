"""Original development counters and retained native projections, no real providers."""
from contextlib import contextmanager
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from race_collection import prospective_speed_results as worker
from race_collection import prospective_speed_runtime as runtime
from race_collection.development_source_authority import CAPS, DATES
from tests.test_prospective_speed_closure import fixture as closure_fixture, load, replace


def fixture(tmp_path, monkeypatch):
    f = closure_fixture(tmp_path, monkeypatch)
    def put(name, value):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        return runtime.put_new(path, value)
    allocation_ref = runtime.reference(tmp_path/'allocation.json')
    allocation_ref = replace(allocation_ref, {**load(allocation_ref), 'authority_reference': 'FABRICATED:original'})
    monkeypatch.setattr(worker.planning, 'ALLOCATION_SHA256', allocation_ref['sha256'])
    original_root = tmp_path/'original-development'
    original_root.mkdir(mode=0o700)
    (original_root/'ready').mkdir(mode=0o700)
    (original_root/'results').mkdir(mode=0o700)
    campaign = tmp_path/'campaign'
    campaign.mkdir(mode=0o700)
    profile_ref = put('original-authority.json', {'schema_version': 'collector_development_pilot_authority_v1',
        'status': 'AUTHORIZED_DEVELOPMENT_PILOT', 'allocation_id': 'development-single-snapshot-20261003-v1',
        'dates': DATES, 'authority_reference': 'FABRICATED:original', 'campaign_id': 'fixture',
        'state_root': str(original_root), 'allocation': allocation_ref, 'allocation_sha256': allocation_ref['sha256'],
        'prior_effective_authorization_sha256': 'a'*64, 'result_closure_at': '2026-10-25T12:00:00+11:00', **CAPS})
    cfg = {'schema_version': 'development_pilot_runtime_v1', 'status': 'AUTHORIZED',
        'allocation': allocation_ref, 'authority_reference': 'FABRICATED:original',
        'pilot_campaign_authority': profile_ref, 'state_root': str(original_root),
        'campaign_root': str(campaign), 'source_state': str(tmp_path/'source-state.json'),
        'lock_path': str(tmp_path/'collector.lock'), 'max_result_operations': 72,
        'max_result_transport_requests': 720, 'max_result_checks_per_race': 3,
        'result_closure_at': '2026-10-25T12:00:00+11:00'}
    legacy_ref = put('original-runtime.json', cfg)
    plan_ref = runtime.reference(tmp_path/'plan.json')
    plan = load(plan_ref)
    plan['authority'].update(allocation=allocation_ref, result_runtime=legacy_ref)
    plan_ref = replace(plan_ref, plan)
    pop_ref = runtime.reference(tmp_path/'population.json')
    old = load(pop_ref)
    population = worker.planning.select_population(plan,
        [{'race_id': f['race_id'], 'race_key': '2026-10-10|FIXTURE|1', 'jump_at': '2026-10-10T13:10:00+11:00'}],
        local_date='2026-10-10', frozen_at='2026-10-10T12:50:00+11:00',
        source_observed_at='2026-10-10T12:48:00+11:00', index_complete=True,
        protected_membership_reference=runtime.reference(tmp_path/'protected-members.json'))
    pop_ref = replace(pop_ref, population)
    account_ref = runtime.reference(tmp_path/'2026-10-10'/'date-accounting.json')
    replace(account_ref, {**load(account_ref), 'population': pop_ref,
        'population_sha256': worker.planning._digest(population)})
    activation_ref = runtime.reference(tmp_path/'activation.json')
    activation_ref = replace(activation_ref, {**load(activation_ref),
        'plan_sha256': plan_ref['sha256'], 'population_sha256': pop_ref['sha256']})
    source_ref = runtime.reference(tmp_path/'source-job.json')
    source = load(source_ref)
    source.update(plan=plan_ref, population=pop_ref, activation=activation_ref)
    source_ref = replace(source_ref, source)
    attempt = f['attempt']
    execution_ref = replace(runtime.reference(attempt/'job.json'), {**source, 'execution_mode': 'PROSPECTIVE'})
    claim_ref = runtime.reference(attempt/'claim.json')
    claim_ref = replace(claim_ref, {**load(claim_ref), 'source_job': source_ref})
    terminal_ref = runtime.reference(attempt/'terminal.json')
    terminal_ref = replace(terminal_ref, {**load(terminal_ref), 'claim': claim_ref, 'execution_job': execution_ref})
    seal_ref = runtime.reference(attempt/'seal.json')
    seal_ref = replace(seal_ref, {**load(seal_ref), 'terminal': terminal_ref})
    completion_ref = runtime.reference(attempt/'completion.json')
    completion_ref = replace(completion_ref, {**load(completion_ref), 'seal': seal_ref})
    key = hashlib.sha256(f['race_id'].encode()).hexdigest()
    disposition_ref = runtime.reference(tmp_path/'2026-10-10'/'jobs'/(key+'.disposition.json'))
    replace(disposition_ref, {**load(disposition_ref), 'completion': completion_ref})
    activate = put('bridge-activation.json', {'status': 'AUTHORIZED_PROSPECTIVE_DEVELOPMENT',
        'plan_sha256': plan_ref['sha256'], 'development_precedence_verified': True,
        'competing_pilot_disabled': True, 'legacy_result_worker_disabled': True,
        'development_readiness_substitution_verified': True, 'result_retention_routing_verified': True,
        'additional_source_requests': 0, 'additional_result_requests': 0, 'verified_control_files': [legacy_ref]})
    bridge_ref = put('bridge-config.json', {'schema_version': 'prospective_speed_result_bridge_v1',
        'status': 'AUTHORIZED_SELECTED_NATIVE_DEVELOPMENT_RESULTS', 'plan': plan_ref,
        'activation': activate, 'legacy_runtime': legacy_ref, 'coordinator_state_root': str(tmp_path)})
    # Bind final closure to the same original-development retention namespace.
    closure_config = load(f['config'])
    sources = {day: {'kind': 'DEVELOPMENT_SELECTED_NATIVE', 'root': str(original_root),
        'authority': legacy_ref, 'bridge_config': bridge_ref} for day in worker.planning.DATES}
    closure_activation = load(closure_config['activation'])
    closure_activation.update(plan_sha256=plan_ref['sha256'], retention_by_date=sources)
    closure_config.update(plan=plan_ref, retention_by_date=sources,
        activation=replace(closure_config['activation'], closure_activation))
    f['config'] = replace(f['config'], closure_config)
    f.update(bridge=bridge_ref, original=original_root, legacy=cfg, plan=plan, key=key)
    f['clock']['now'] = datetime.fromisoformat('2026-10-10T14:00:00+11:00')
    return f


@contextmanager
def no_source_owner(*args):
    yield


def fake_transport(monkeypatch, f, *, status=200, second_finish=b'2nd'):
    import requests
    from race_collection import freshness_campaign
    calls, charges, holds = [], [], []
    body = (b'<table class="race-runners--result">'
        b'<tr class="race-runner"><td class="race-runners__finish-position">1st</td>'
        b'<td class="race-runners__box"><sprite-svg name="rug_1"></sprite-svg></td>'
        b'<td class="race-runners__name"><a href="/dogs/501/dog-one" data-dog-id="501">DOG ONE</a></td></tr>'
        b'<tr class="race-runner"><td class="race-runners__finish-position">'+second_finish+b'</td>'
        b'<td class="race-runners__box"><sprite-svg name="rug_2"></sprite-svg></td>'
        b'<td class="race-runners__name"><a href="/dogs/502/dog-two" data-dog-id="502">DOG TWO</a></td></tr></table>')
    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, **kwargs):
            # Original operation and request reservations exist BEFORE transport.
            directory = f['original']/'results'/f['key']/'attempt-0'
            assert (directory/'started.json').exists() and (directory/'request.json').exists()
            calls.append((url, kwargs))
            raw = io.BytesIO(body)
            return SimpleNamespace(status_code=status, url=url,
                raw=SimpleNamespace(read=lambda size, decode_content=False: raw.read(size)),
                headers={'content-type': 'text/html'}, close=lambda: None)
    class Campaign:
        def __init__(self, *args, **kwargs): pass
        def request(self, **kwargs): charges.append(kwargs)
        def hold_source(self, value): holds.append(value)
    class Clock:
        @classmethod
        def now(cls, *args): return f['clock']['now']
    monkeypatch.setattr(requests, 'Session', Session)
    monkeypatch.setattr(freshness_campaign, 'Campaign', Campaign)
    monkeypatch.setattr(worker, 'ownership', no_source_owner)
    monkeypatch.setattr(worker.legacy, '_shared_stop', lambda cfg: None)
    monkeypatch.setattr(worker.legacy, 'datetime', Clock)
    return calls, charges, holds


def test_original_transport_and_counters_then_automatic_selected_closure(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    calls, charges, _ = fake_transport(monkeypatch, f)
    value = worker.run_cycle(f['bridge'])
    assert value['status'] == 'SELECTED_RESULT_CHECK_RETAINED'
    directory = f['original']/'results'/f['key']/'attempt-0'
    refs = {key: runtime.reference(directory/name) for key, name in (
        ('body', 'response.html'), ('request', 'request.json'), ('response', 'response.json'), ('http_envelope', 'http-envelope.json'))}
    context = worker.closure._native_context(worker.evaluation._Checked(), runtime.read_output(runtime.reference(f['attempt']/'forecast.json')))
    worker.validate_retained_response(context, refs, f['clock']['now'])
    assert load(runtime.reference(f['original']/'results'/f['key']/'attempt-0'/'finished.json'))['disposition'] == 'OFFICIAL'
    assert len(calls) == len(charges) == 1 and charges == [{'kind': 'results'}]
    metadata = worker.inspect_queue(f['bridge'])
    assert metadata['operations_consumed'] == metadata['transport_requests_consumed'] == 1
    assert worker.run_cycle(f['bridge'])['status'] == 'NO_SELECTED_RESULT_DUE'
    assert len(calls) == 1
    f['clock']['now'] = datetime.fromisoformat('2026-10-25T12:05:00+11:00')
    packet = runtime.read_output(runtime.reference(f['attempt']/'forecast.json'))
    context = worker.closure._native_context(worker.evaluation._Checked(), packet)
    source = load(f['config'])['retention_by_date']['2026-10-10']
    assert worker.closure._development_closure(source, context,
        datetime.fromisoformat('2026-10-25T12:00:00+11:00'))[0] == 'FULL_ORDER_WIN_ELIGIBLE'
    assert worker.closure.finalize(f['config'])['status'] == 'COMPLETE_SINGLE_PLANNED_EVALUATION'
    report = json.loads((f['root']/'evaluation'/'evaluation.private.json').read_bytes())
    assert report['accounting']['scored_races'] == 1


def test_waiting_for_collector_consumes_no_attempt(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    @contextmanager
    def busy(*args):
        raise ValueError('RESULT_COLLECTOR_BUSY')
        yield
    monkeypatch.setattr(worker, 'ownership', busy)
    assert worker.run_cycle(f['bridge'])['status'] == 'WAITING_FOR_EXISTING_SOURCE_OWNER'
    value = worker.inspect_queue(f['bridge'])
    assert value['selected_due'] == value['transport_due'] == 1
    assert value['operations_consumed'] == value['transport_requests_consumed'] == 0


def test_denial_is_preserved_and_no_further_request_is_made(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    calls, charges, holds = fake_transport(monkeypatch, f, status=403)
    worker.run_cycle(f['bridge'])
    assert len(calls) == len(charges) == len(holds) == 1
    assert (f['original']/'results'/'source-stop.json').exists()
    f['clock']['now'] = datetime.fromisoformat('2026-10-11T12:00:00+11:00')
    assert worker.run_cycle(f['bridge'])['status'] == 'SOURCE_STOP'
    assert len(calls) == 1


def test_previous_operations_reduce_shared_remaining_budget(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    for i in range(72):
        path = f['original']/'results'/f'prior-{i}'/'attempt-0'
        path.mkdir(parents=True)
        runtime.put_new(path/'started.json', {'preserved_previous': True})
    monkeypatch.setattr(worker, 'ownership', lambda *args: pytest.fail('exhausted source access'))
    value = worker.run_cycle(f['bridge'])
    assert value['status'] == 'SELECTED_RESULT_UNRESOLVED'
    assert worker.inspect_queue(f['bridge'])['operations_consumed'] == 72


def test_invalid_native_nomination_rejected_before_transport(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    config, plan, cfg = worker.load_config(f['bridge'])
    worker.nominate(config, plan, cfg)
    path = f['original']/'ready'/(f['key']+'.json')
    ready = load(runtime.reference(path))
    ready['job_id'] = 'not-the-frozen-job'
    with pytest.raises(ValueError, match='NATIVE_READY_BINDING_CHANGED'):
        worker._admitted(config, plan, ready)


def test_partial_original_started_marker_stays_consumed(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    config, plan, cfg = worker.load_config(f['bridge'])
    worker.nominate(config, plan, cfg)
    path = f['original']/'results'/f['key']/'attempt-0'
    path.mkdir(parents=True)
    (path/'started.json').write_bytes(b'{"race_id":')
    monkeypatch.setattr(worker, 'ownership', lambda *args: pytest.fail('consumed attempt retried'))
    assert worker.run_cycle(f['bridge'])['status'] == 'NO_SELECTED_RESULT_DUE'
    assert worker.inspect_queue(f['bridge'])['operations_consumed'] == 1
    assert (path/'started.json').read_bytes() == b'{"race_id":'


def test_original_other_cohort_metadata_does_not_open_its_labels(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    other = 'Race 1 - OTHER - 2026-10-03'
    key = hashlib.sha256(other.encode()).hexdigest()
    runtime.put_new(f['original']/'ready'/(key+'.json'), {
        'schema_version': 'development_pilot_capture_ready_v1', 'race_id': other,
        'race_key': worker.legacy.race_key(other), 'allocation_sha256': f['legacy']['allocation']['sha256'],
        'pre_result_sha256': 'a'*64, 'jump_at': '2026-10-03T13:10:00+10:00'})
    directory = f['original']/'results'/key
    directory.mkdir(mode=0o700)
    (directory/'official-result.json').write_bytes(b'PROTECTED FOR THIS EXECUTION: MUST NOT DECODE')
    calls, _, _ = fake_transport(monkeypatch, f)
    assert worker.prepare_queue(f['bridge'])['ready_total'] == 2
    assert worker.run_cycle(f['bridge'])['status'] == 'SELECTED_RESULT_CHECK_RETAINED'
    assert len(calls) == 1
    assert not (directory/'complete.json').exists()


def test_finite_deadline_closes_missing_without_ownership_or_transport(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    f['clock']['now'] = datetime.fromisoformat('2026-10-25T12:00:00+11:00')
    monkeypatch.setattr(worker, 'ownership', lambda *args: pytest.fail('expired request'))
    assert worker.run_cycle(f['bridge'])['status'] == 'SELECTED_RESULT_UNRESOLVED'
    assert worker.inspect_queue(f['bridge'])['operations_consumed'] == 0
    f['clock']['now'] = datetime.fromisoformat('2026-10-25T12:05:00+11:00')
    assert worker.closure.finalize(f['config'])['status'] == 'COMPLETE_SINGLE_PLANNED_EVALUATION'
    manifest = json.loads((f['root']/'result-manifest.json').read_bytes())
    assert manifest['entries'][0]['label_status'] == 'MISSING_AT_DEADLINE'
    assert manifest['entries'][0]['target'] is None


def test_existing_ready_repeated_preparation_reads_metadata_not_forecast_payload(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    assert worker.prepare_queue(f['bridge'])['nominated'] == 1
    monkeypatch.setattr(runtime, 'read_output', lambda *args: pytest.fail('repeated large forecast decode'))
    result = worker.prepare_queue(f['bridge'])
    assert result['nominated'] == 0 and result['selected_due'] == 1


@pytest.mark.parametrize('finish,target,label', [
    (b'1st', [.5, .5], 'FULL_ORDER_WIN_ELIGIBLE'),
    (b'DNF', [1., 0.], 'KNOWN_NONFINISH_WIN_ELIGIBLE')])
def test_actual_bridge_deadheat_and_nonfinish_reuse_same_response(tmp_path, monkeypatch, finish, target, label):
    f = fixture(tmp_path, monkeypatch)
    calls, charges, _ = fake_transport(monkeypatch, f, second_finish=finish)
    assert worker.run_cycle(f['bridge'])['status'] == 'SELECTED_RESULT_CHECK_RETAINED'
    assert len(calls) == len(charges) == 1
    assert worker.run_cycle(f['bridge'])['status'] == 'NO_SELECTED_RESULT_DUE'
    f['clock']['now'] = datetime.fromisoformat('2026-10-25T12:05:00+11:00')
    assert worker.closure.finalize(f['config'])['status'] == 'COMPLETE_SINGLE_PLANNED_EVALUATION'
    entry = json.loads((f['root']/'result-manifest.json').read_bytes())['entries'][0]
    assert entry['label_status'] == label
    assert load(entry['target'])['outcome'] == target
    assert len(calls) == 1


def test_denial_precedes_optional_envelope_write_failure(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    calls, _, holds = fake_transport(monkeypatch, f, status=403)
    real_put = runtime.put_new
    def full_disk(path, value):
        if Path(path).name == 'http-envelope.json':
            raise OSError('fabricated envelope disk failure')
        return real_put(path, value)
    monkeypatch.setattr(runtime, 'put_new', full_disk)
    assert worker.run_cycle(f['bridge'])['status'] == 'SELECTED_RESULT_CHECK_RETAINED'
    assert len(calls) == len(holds) == 1
    assert (f['original']/'results'/'source-stop.json').exists()


def test_unknown_terminal_stays_quarantined_after_deadline(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    calls, _, _ = fake_transport(monkeypatch, f, second_finish=b'-')
    worker.run_cycle(f['bridge'])
    f['clock']['now'] = datetime.fromisoformat('2026-10-25T12:00:00+11:00')
    assert worker.run_cycle(f['bridge'])['status'] == 'SELECTED_RESULT_UNRESOLVED'
    f['clock']['now'] = datetime.fromisoformat('2026-10-25T12:05:00+11:00')
    worker.closure.finalize(f['config'])
    entry = json.loads((f['root']/'result-manifest.json').read_bytes())['entries'][0]
    assert entry['label_status'] == 'QUARANTINED_RETAINED_RESULT' and entry['target'] is None
    assert len(calls) == 1


def amended_authority(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    previous = f['legacy']
    profile = load(previous['pilot_campaign_authority'])
    windows = [{'local_date': '2026-10-06'}, {'local_date': '2026-10-07'}]
    approval = runtime.put_new(tmp_path/'reschedule.json', {
        'schema_version': 'development_reschedule_authorization_v1', 'status': 'AUTHORIZED',
        'authority_reference': 'FABRICATED:earlier', 'predecessor_allocation': profile['allocation'],
        'selection_windows': windows, 'maximum_total': 12, 'maximum_per_date': 6,
        'preserve_existing_study_members': True, 'preserve_consumed_allowances': True})
    amendment = {'user_receipt': approval}
    amendment_ref = runtime.put_new(tmp_path/'amendment.json', amendment)
    allocation = {'schema_version': 'development_allocation_v2', 'status': 'AUTHORIZED',
        'allocation_id': profile['allocation_id'], 'authority_reference': 'FABRICATED:earlier',
        'predecessor_allocation': profile['allocation'], 'approval': approval,
        'dates': ['2026-10-06', '2026-10-07'], 'selection_windows': windows,
        'max_capture_attempts': 12, 'max_attempts_per_date': 6}
    allocation_ref = runtime.put_new(tmp_path/'earlier-allocation.json', allocation)
    cfg = {**previous, 'schema_version': 'development_pilot_runtime_v2', 'allocation': allocation_ref,
        'authority_reference': 'FABRICATED:earlier', 'result_closure_at': '2026-10-09T12:00:00+11:00',
        'predecessor_runtime': load(f['bridge'])['legacy_runtime'], 'reschedule_authorization': approval}
    plan = {'schema_version': 'prospective_sectional_plan_v2', 'dates': allocation['dates'],
        'selection_windows': windows, 'schedule_amendment_reference': amendment_ref,
        'schedule_amendment': amendment}
    return plan, cfg, profile, allocation


def test_earlier_result_permission_reuses_original_campaign_profile_and_ledger(tmp_path, monkeypatch):
    plan, cfg, profile, allocation = amended_authority(tmp_path, monkeypatch)
    worker._result_authority(plan, cfg, profile, allocation)
    previous = load(cfg['predecessor_runtime'])
    assert cfg['pilot_campaign_authority'] == previous['pilot_campaign_authority']
    assert cfg['state_root'] == previous['state_root']
    assert cfg['max_result_operations'] == 72 and cfg['max_result_transport_requests'] == 720


@pytest.mark.parametrize('change', ['ledger', 'profile', 'budget', 'deadline', 'allocation', 'authority'])
def test_earlier_permission_cannot_reset_budget_or_detach_approval(tmp_path, monkeypatch, change):
    plan, cfg, profile, allocation = amended_authority(tmp_path, monkeypatch)
    if change == 'ledger': cfg['state_root'] += '-fresh'
    if change == 'profile': cfg['pilot_campaign_authority'] = {'path': '/new-profile', 'sha256': 'a'*64}
    if change == 'budget': cfg['max_result_operations'] = 73
    if change == 'deadline': cfg['result_closure_at'] = '2026-10-26T12:00:00+11:00'
    if change == 'allocation': allocation['max_capture_attempts'] = 24
    if change == 'authority': cfg['authority_reference'] = 'FABRICATED:unbound'
    with pytest.raises(ValueError, match='AMENDED_RESULT_AUTHORITY_INVALID'):
        worker._result_authority(plan, cfg, profile, allocation)


def test_mixed_original_and_amended_nominations_keep_prior_counters(tmp_path, monkeypatch):
    plan, cfg, profile, allocation = amended_authority(tmp_path, monkeypatch)
    root = Path(cfg['state_root'])
    old = load(cfg['predecessor_runtime'])
    names = [('Race 1 - DUBO - 2026-10-03', old['allocation']['sha256'], '2026-10-03T13:10:00+10:00'),
             ('Race 1 - LADBROKES-Q1-LAKESIDE - 2026-10-06', cfg['allocation']['sha256'], '2026-10-06T22:21:00+11:00')]
    for race_id, pin, jump in names:
        key = hashlib.sha256(race_id.encode()).hexdigest()
        runtime.put_new(root/'ready'/(key+'.json'), {
            'schema_version': 'development_pilot_capture_ready_v1', 'race_id': race_id,
            'race_key': worker.planning.race_key_for_plan(plan, race_id),
            'allocation_sha256': pin, 'jump_at': jump, 'pre_result_sha256': 'b'*64})
        d = root/'results'/key/'attempt-0'; d.mkdir(parents=True)
        runtime.put_new(d/'started.json', {'race_id': race_id})
        runtime.put_new(d/'request.json', {'race_id': race_id})
    rows, due, completed, operations, transports = worker._inventory(cfg, plan, root,
        datetime.fromisoformat('2026-10-07T06:00:00+11:00'))
    assert len(rows) == 2 and operations == transports == 2
    assert {r['allocation_sha256'] for r in rows} == {old['allocation']['sha256'], cfg['allocation']['sha256']}
    # No target records are needed or opened to preserve earlier accounting.
    assert not list((root/'results').glob('*/official-result.json'))
