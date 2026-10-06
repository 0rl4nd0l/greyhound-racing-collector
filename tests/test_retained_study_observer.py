"""Observer consumes metadata references, never a provider or scoring path."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def case(tmp_path, monkeypatch):
    from race_collection import retained_study_observer as observer
    now = datetime(2026, 10, 4, 6, tzinfo=timezone.utc)
    models = ['market', 'production', 'residual_box', 'residual_half']
    bundle = tmp_path/'bundles'/'prediction1'
    files = {}
    for name in [*[f'comparison/{m}.json' for m in models], 'model/model.json', 'model/manifest.json', 'comparison/registry.json', *observer.REQUIRED_INPUTS]:
        files[name] = {'sha256': put(bundle/name, {'synthetic': name})['sha256'], 'bytes': (bundle/name).stat().st_size}
    pins = {k: files[k]['sha256'] for k in ('model/model.json', 'model/manifest.json', 'comparison/registry.json')}
    manifest = put(bundle/'bundle_manifest.json', {'prediction_id': 'p1', 'job_id': 'j1', 'files': files})
    plan = {'programme_root': str(tmp_path/'original'), 'prediction_output_roots': [str(tmp_path/'bundles')],
            'status': 'AUTHORIZED_ENGINEERING', 'starts_at': '2026-10-01T00:00:00+00:00',
            'ends_at': '2026-10-05T00:00:00+00:00'}
    plan_ref = put(tmp_path/'plan.json', plan)
    race = 'Race 1 - SYNTHETIC - 2026-10-03'
    key = hashlib.sha256(race.encode()).hexdigest()
    claim = tmp_path/'original'/plan_ref['sha256']/'attempts'/key
    admission = {'race': {'race_id': race, 'race_date': '2026-10-03', 'jump_timestamp': '2026-10-03T08:00:00+00:00'},
                 'job_id': 'j1', 'prediction_id': 'p1', 'plan_sha256': plan_ref['sha256'],
                 'runner_set_sha256': 'a'*64, 'retained_input_manifest_sha256': 'b'*64,
                 'admitted_at': '2026-10-03T07:51:00+00:00', 'decision_at': '2026-10-03T07:58:00+00:00',
                 'evidence_class': 'AUTHORIZED_ENGINEERING'}
    request = {k: admission[k] for k in ('job_id', 'prediction_id', 'runner_set_sha256', 'retained_input_manifest_sha256')}
    request.update(race_id=race, jump_timestamp=admission['race']['jump_timestamp'])
    for name, value in [('request.json', request), ('features/sealed/implementation_file_manifest.json', {'git_head': 'd'*40})]:
        files[name] = {**put(bundle/name, value), 'bytes': (bundle/name).stat().st_size}
        files[name].pop('path')
    manifest = put(bundle/'bundle_manifest.json', {'prediction_id': 'p1', 'job_id': 'j1', 'files': files})
    ar = put(claim/'admission.json', admission)
    completion = {**admission, 'admission_sha256': ar['sha256'], 'status': 'COMPLETE_BEFORE_CUTOFF',
                  'models': dict.fromkeys(models, 'SEALED'), 'published_complete_at': '2026-10-03T07:52:00+00:00',
                  'bundle_entry': {'directory': 'prediction1', 'manifest_sha256': manifest['sha256']}}
    put(claim/'completion.json', completion)
    original = put(tmp_path/'study.json', {'ends_at': '2027-01-21T12:00:00+11:00'})
    protocol = {'schema_version': 'retained_study_protocol_v1', 'status': 'AUTHORIZED_OUTCOME_BLIND_RETAINED_STUDY',
                'issued_at': '2026-10-04T04:00:00+00:00', 'effective_at': '2026-10-04T05:00:00+00:00',
                'ends_at': '2027-01-21T12:00:00+11:00', 'original_study_plan': original,
                'state_root': str(tmp_path/'observer'), 'max_total_members': 1000,
                'prior_scientific_capture_attempts': 17, 'max_new_members': 983,
                'prior_member_race_ids': [], 'historical_plans': [plan_ref], 'persistent_source': None,
                'frozen_model_files': pins, 'opportunity_evidence': [],
                'scan_limits': {'max_files': 1000, 'max_bytes': 10000000}}
    cfg = {'retained_study_protocol': put(tmp_path/'protocol.json', protocol),
           'study_amendment': {'path': '/synthetic/amendment.json', 'sha256': 'c'*64}}
    monkeypatch.setattr(observer, 'verify_retained_readiness', lambda cfg, now: {'status': 'VERIFIED'})
    monkeypatch.setattr(observer, 'load_plan', lambda p, h: (json.loads(p.read_bytes()), p.read_bytes()))
    return observer, cfg, protocol, now, claim, bundle


def test_original_prejump_seals_are_referenced_once_without_reenrolment(case):
    observer, cfg, protocol, now, claim, bundle = case
    before = {p: p.read_bytes() for p in [claim/'admission.json', claim/'completion.json', bundle/'bundle_manifest.json']}
    first = observer.observe(cfg, now=now)
    second = observer.observe(cfg, now=now)
    assert first['new_members'] == 1 and second['new_members'] == 0
    assert second['members'] == 1 and second['remaining_members'] == 982
    assert second['prior_scientific_capture_attempts'] == 17
    assert all(p.read_bytes() == raw for p, raw in before.items())
    rows = [json.loads(line) for line in (Path(protocol['state_root'])/'events.jsonl').read_text().splitlines()]
    member = next(r['event'] for r in rows if r['event']['kind'] == 'MEMBER')
    assert member['selection_at'] == now.isoformat()
    assert member['original_evidence_class'] == 'AUTHORIZED_ENGINEERING'
    assert member['membership_class'] == 'RETROSPECTIVE_RETAINED_PREJUMP_FORECAST'
    assert member['new_scores_generated'] is False


def rebind(case, **changes):
    observer, cfg, protocol, now, claim, bundle = case
    protocol.update(changes)
    cfg['retained_study_protocol'] = put(Path(cfg['retained_study_protocol']['path']), protocol)


@pytest.mark.parametrize('defect', ['forecast', 'history', 'capture', 'late', 'missing', 'symlink'])
def test_incomplete_changed_or_late_evidence_is_excluded(case, defect):
    observer, cfg, protocol, now, claim, bundle = case
    if defect in ('forecast', 'history', 'capture'):
        name = {'forecast': 'comparison/market.json', 'history': 'features/sealed_history.db', 'capture': 'source/capture.json'}[defect]
        (bundle/name).write_bytes(b'corrupt')
    elif defect == 'late':
        completion = json.loads((claim/'completion.json').read_bytes())
        completion['published_complete_at'] = completion['decision_at']
        put(claim/'completion.json', completion)
    elif defect == 'missing':
        (claim/'completion.json').unlink()
    else:
        path = bundle/'source/capture.json'; target = bundle/'elsewhere.json'
        path.rename(target); path.symlink_to(target)
    result = observer.observe(cfg, now=now)
    assert result['new_members'] == result['members'] == 0
    assert 'EXCLUDED' in (Path(protocol['state_root'])/'events.jsonl').read_text()


def test_changed_already_selected_input_holds_before_further_admission(case):
    observer, cfg, protocol, now, claim, bundle = case
    observer.observe(cfg, now=now)
    (bundle/'source/capture.json').write_bytes(b'corrupt')
    before = (Path(protocol['state_root'])/'events.jsonl').read_bytes()
    with pytest.raises(ValueError, match='DECLARED_FILE_CHANGED'):
        observer.observe(cfg, now=now)
    assert (Path(protocol['state_root'])/'events.jsonl').read_bytes() == before


@pytest.mark.parametrize('limit', ['end', 'prior'])
def test_original_membership_and_endpoint_never_create_new_member(case, limit):
    observer, cfg, protocol, now, claim, bundle = case
    if limit == 'prior':
        rebind(case, prior_member_race_ids=[json.loads((claim/'admission.json').read_bytes())['race']['race_id']])
    else:
        now = datetime(2027, 1, 22, tzinfo=timezone.utc)
    assert observer.observe(cfg, now=now)['members'] == 0


def test_gate_before_effective_has_no_writes(case, monkeypatch):
    observer, cfg, protocol, now, claim, bundle = case
    monkeypatch.setattr(observer, 'verify_retained_readiness', lambda c, n: None)
    assert observer.observe(cfg, now=now)['status'] == 'NOT_EFFECTIVE'
    assert not Path(protocol['state_root']).exists()


def test_opportunities_and_unadmitted_attempts_are_retained(case):
    observer, cfg, protocol, now, claim, bundle = case
    programme = claim.parent.parent
    put(programme/'opportunities'/'race.json', {'race_id': 'excluded-original', 'plan_sha256': programme.name})
    put(claim.parent/'failed'/'dispatch.json', {'race_id': 'failed-original'})
    observer.observe(cfg, now=now)
    text = (Path(protocol['state_root'])/'events.jsonl').read_text()
    assert 'ORIGINAL_OPPORTUNITY' in text and 'UNADMITTED_NATIVE_ATTEMPT' in text
    before = text
    observer.observe(cfg, now=now)
    assert (Path(protocol['state_root'])/'events.jsonl').read_text() == before


def test_original_root_collision_and_invalid_authority_fail_without_original_write(case):
    observer, cfg, protocol, now, claim, bundle = case
    rebind(case, state_root=str(bundle.parent))
    with pytest.raises(ValueError, match='overlaps_original'):
        observer.observe(cfg, now=now)
    assert not (bundle.parent/'events.jsonl').exists()


def test_budget_exhaustion_is_not_success(case):
    observer, cfg, protocol, now, claim, bundle = case
    rebind(case, scan_limits={'max_files': 1, 'max_bytes': 1})
    with pytest.raises(RuntimeError, match='scan_allowance_exhausted'):
        observer.observe(cfg, now=now)


def test_changed_protocol_cannot_adopt_existing_ledger(case):
    observer, cfg, protocol, now, claim, bundle = case
    observer.observe(cfg, now=now)
    rebind(case, max_new_members=982, max_total_members=999)
    with pytest.raises(ValueError, match='authority_changed'):
        observer.observe(cfg, now=now)


def test_future_day_discovery_authenticates_original_standing_and_allocation(case, tmp_path):
    from tests.fixtures.persistent_operation_case import make_persistent
    observer, cfg, protocol, now, claim, bundle = case
    standing, standing_ref, allocation, unused = make_persistent(tmp_path/'future')
    day = Path(allocation['state_root'])
    allocation_ref = put(day/'allocation.json', allocation)
    plan = json.loads(Path(protocol['historical_plans'][0]['path']).read_bytes())
    plan.update(persistent_allocation=allocation_ref, programme_root=str(day/'admission'),
                prediction_output_roots=[str(Path(allocation['prediction_root'])/'bundles')],
                candidate_registry=standing['candidate_registry'])
    comparison = put(day/'comparison.json', plan)
    rebind(case, persistent_source={'standing_authority': standing_ref,
        'runtime_root': standing['state_root'], 'first_racing_date': '2026-10-03'})
    assert comparison in [r for r, p in observer.discover_plans(protocol)]
    put(Path(standing['state_root'])/'health.json', {'status': 'HOLD', 'at': now.isoformat()})
    assert observer.observe(cfg, now=now)['new_input_state'] == 'REPORTED_PAUSED'
    plan['persistent_allocation']['sha256'] = '0'*64
    put(day/'comparison.json', plan)
    with pytest.raises(ValueError, match='daily_plan_binding_changed'):
        observer.discover_plans(protocol)


def test_cli_wrong_pin_stops_before_observer_write(case, capsys):
    from scripts.run_retained_study_observer import main
    observer, cfg, protocol, now, claim, bundle = case
    assert main(['--config', cfg['retained_study_protocol']['path'], '--config-sha256', '0'*64]) == 78
    assert 'OBSERVER_HOLD' in capsys.readouterr().out
    assert not Path(protocol['state_root']).exists()


def test_candidate_budget_pending_does_not_admit_partial_chain(case):
    observer, cfg, protocol, now, claim, bundle = case
    rebind(case, scan_limits={'max_files': 5, 'max_bytes': 10000000})
    result = observer.observe(cfg, now=now)
    assert result['status'] == 'PENDING_QUALIFICATION_SCAN_ALLOWANCE_EXHAUSTED'
    assert result['members'] == 0 and result['complete_scan'] is False


def test_declared_manifest_path_escape_is_rejected(case):
    observer, cfg, protocol, now, claim, bundle = case
    manifest = json.loads((bundle/'bundle_manifest.json').read_bytes())
    manifest['files']['../outside.json'] = {'bytes': 0, 'sha256': '0'*64}
    ref = put(bundle/'bundle_manifest.json', manifest)
    complete = json.loads((claim/'completion.json').read_bytes())
    complete['bundle_entry']['manifest_sha256'] = ref['sha256']
    put(claim/'completion.json', complete)
    assert observer.observe(cfg, now=now)['members'] == 0


def test_corrupt_journal_never_restarts_or_resets(case):
    observer, cfg, protocol, now, claim, bundle = case
    observer.observe(cfg, now=now)
    journal = Path(protocol['state_root'])/'events.jsonl'
    with journal.open('ab') as stream:
        stream.write(b'{partial')
    before = journal.read_bytes()
    with pytest.raises(ValueError):
        observer.observe(cfg, now=now)
    assert journal.read_bytes() == before


def test_completion_cannot_be_selected_before_its_original_publication(case):
    observer, cfg, protocol, now, claim, bundle = case
    rebind(case, issued_at='2026-10-02T00:00:00+00:00', effective_at='2026-10-02T01:00:00+00:00')
    now = datetime(2026, 10, 3, 7, 51, tzinfo=timezone.utc)
    assert observer.observe(cfg, now=now)['members'] == 0
    assert 'NATIVE_COMPLETION_AFTER_OBSERVATION' in (Path(protocol['state_root'])/'events.jsonl').read_text()


@pytest.fixture
def reserved_case(case):
    """Native seal during the exact future reservation window, with no outcomes."""
    observer, cfg, protocol, now, claim, bundle = case
    root = bundle.parent.parent
    race = 'Race 1 - SYNTHETIC - 2026-10-10'
    plan = json.loads(Path(protocol['historical_plans'][0]['path']).read_bytes())
    plan['ends_at'] = '2026-10-12T00:00:00+11:00'
    plan_ref = put(root/'plan.json', plan)
    protocol['historical_plans'] = [plan_ref]
    cfg['retained_study_protocol'] = put(root/'protocol.json', protocol)
    prior = put(root/'predecessor-config.json', dict(cfg))
    admission = json.loads((claim/'admission.json').read_bytes())
    completion = json.loads((claim/'completion.json').read_bytes())
    admission.update(plan_sha256=plan_ref['sha256'], admitted_at='2026-10-10T02:11:00+00:00',
                     decision_at='2026-10-10T02:18:00+00:00')
    admission['race'] = {'race_id': race, 'race_date': '2026-10-10',
                         'jump_timestamp': '2026-10-10T02:20:00+00:00'}
    claim = root/'original'/plan_ref['sha256']/'attempts'/hashlib.sha256(race.encode()).hexdigest()
    ar = put(claim/'admission.json', admission)
    request = json.loads((bundle/'request.json').read_bytes())
    request.update(race_id=race, jump_timestamp=admission['race']['jump_timestamp'])
    rr = put(bundle/'request.json', request)
    manifest = json.loads((bundle/'bundle_manifest.json').read_bytes())
    manifest['files']['request.json'] = {'sha256': rr['sha256'], 'bytes': (bundle/'request.json').stat().st_size}
    mr = put(bundle/'bundle_manifest.json', manifest)
    completion.update(admission, admission_sha256=ar['sha256'], published_complete_at='2026-10-10T02:12:00+00:00')
    completion['bundle_entry']['manifest_sha256'] = mr['sha256']
    put(claim/'completion.json', completion)
    authority = 'user:20260930:approved-development-single-snapshot-20261003-v1'
    dates = ['2026-10-03', '2026-10-04', '2026-10-10', '2026-10-11']
    policy = 'first_six_1310_1420_melbourne_before_WIN_qualification_v1'
    approval = put(root/'approval.json', {'schema_version': 'development_pilot_user_approval_v1',
        'status': 'APPROVED', 'authority_reference': authority})
    amendment = put(root/'exclusive.json', {'schema_version': 'development_reservation_amendment_v1',
        'status': 'AUTHORIZED', 'approval': approval, 'authority_reference': authority,
        'development_allocation_id': 'development-single-snapshot-20261003-v1',
        'candidate_local_dates': dates, 'selection_policy': policy,
        'prior_allocation_sha256': 'b708fa973aa972b8cd248b4b4d3269fa7aa16402755ee0fb84da5212db6822d1'})
    allocation = put(root/'allocation.json', {'schema_version': 'development_allocation_v1',
        'status': 'AUTHORIZED', 'allocation_id': 'development-single-snapshot-20261003-v1',
        'authority_reference': authority, 'approval': approval, 'dates': dates,
        'selection_policy': policy, 'max_attempts_per_date': 6, 'reservation_amendments': [amendment]})
    speed_plan = put(root/'speed-plan.json', {'schema_version': 'prospective_sectional_plan_v1',
        'frozen_at': '2026-10-06T00:00:00+11:00', 'authority': {'allocation': allocation,
        'exclusive_amendment': amendment}, 'dates': dates[2:], 'timezone': 'Australia/Melbourne',
        'allocation_id': 'development-single-snapshot-20261003-v1',
        'population': {'freeze_local_time': '12:50', 'max_index_age_seconds': 300,
            'selection_policy': policy, 'maximum_per_date': 6, 'maximum_total': 12,
            'no_replacement_after_failure': True, 'protect_existing_study_members': True}})
    reservation = {'schema_version': 'retained_study_development_reservations_v1',
        'status': 'AUTHORIZED_ORIGINAL_DEVELOPMENT_RESERVATIONS', 'allocation': allocation,
        'exclusive_amendment': amendment, 'plan': speed_plan, 'state_root': str(root/'speed'),
        'dates': dates[2:], 'predecessor_observer_config': prior}
    cfg['development_reservations'] = put(root/'reservations.json', reservation)
    now = datetime(2026, 10, 10, 2, 15, tzinfo=timezone.utc)
    return observer, cfg, protocol, now, claim, bundle, reservation


def test_pending_freeze_holds_potential_reserved_candidate(reserved_case):
    observer, cfg, protocol, now, claim, bundle, reservation = reserved_case
    result = observer.observe(cfg, now=now)
    assert result['members'] == 0
    assert result['pending_development_reservations'] == 1
    events = (Path(protocol['state_root'])/'events.jsonl').read_text()
    assert 'DEVELOPMENT_SELECTION_PENDING' in events
    assert 'Race 1 - SYNTHETIC - 2026-10-10' in events


def complete_reserved_freeze(case, *, selected=True):
    observer, cfg, protocol, now, claim, bundle, reservation = case
    observer.observe(cfg, now=now)
    root = Path(reservation['state_root'])/'2026-10-10'
    journal = Path(protocol['state_root'])/'events.jsonl'
    entries = [json.loads(line) for line in journal.read_bytes().splitlines()]
    snapshot = put(root/'protected-membership.json', {'source': observer.reference(journal),
        'race_ids': sorted({r['event']['race_id'] for r in entries if r['event']['kind'] == 'MEMBER'}),
        'last_chain_sha256': entries[-1]['sha256']})
    row = {'race_id': 'Race 1 - SYNTHETIC - 2026-10-10', 'race_key': '2026-10-10|SYNTHETIC|1',
           'jump_at': '2026-10-10T02:20:00+00:00'}
    rows = [row] if selected else [
        {'race_id': f'Race {i+2} - SYNTHETIC - 2026-10-10',
         'race_key': f'2026-10-10|SYNTHETIC|{i+2}', 'jump_at': f'2026-10-10T02:{10+i}:00+00:00'}
        for i in range(6)] + [row]
    first_six = [r['race_id'] for r in rows[:6]]
    frozen = '2026-10-10T01:50:00+00:00'
    observed = '2026-10-10T01:49:00+00:00'
    original = {'schema_version': 'development_population_freeze_v1', 'synthetic': False,
        'allocation_id': 'development-single-snapshot-20261003-v1',
        'allocation_sha256': reservation['allocation']['sha256'],
        'selection_policy': 'first_six_1310_1420_melbourne_before_WIN_qualification_v1',
        'local_date': '2026-10-10', 'frozen_at': frozen, 'source_observed_at': observed,
        'source_index': str(root/'source-index.json'), 'source_packet_sha256': 'e'*64,
        'observed_races': rows, 'intended': rows, 'selected_race_ids': first_six}
    original_ref = put(root/'original-population.json', original)
    put(root/'original-population.json.completion.json', {'status': 'POPULATION_FROZEN',
        'population_sha256': original_ref['sha256'], 'completed_at': '2026-10-10T01:50:01+00:00'})
    plan = json.loads(Path(reservation['plan']['path']).read_bytes())
    digest = lambda v: hashlib.sha256(observer.canonical(v)).hexdigest()
    population = {'schema_version': 'prospective_sectional_population_v1', 'plan_sha256': digest(plan),
        'local_date': '2026-10-10', 'frozen_at': frozen, 'source_observed_at': observed,
        'index_complete': True, 'observed_races_sha256': digest(rows), 'observed_races': rows,
        'first_six_race_ids': first_six, 'selected_race_ids': first_six,
        'protected_membership_reference': snapshot, 'protected_race_ids': [],
        'dispositions': [{'race_id': r['race_id'], 'jump_at': r['jump_at'],
            'disposition': 'SELECTED' if r['race_id'] in first_six else 'BEYOND_FIRST_SIX'} for r in rows]}
    population_ref = put(root/'population.json', population)
    put(root/'date-accounting.json', {'local_date': '2026-10-10', 'status': 'POPULATION_FROZEN',
        'population_sha256': digest(population), 'population': population_ref})
    return root


@pytest.mark.parametrize('selected', [True, False])
def test_frozen_selection_routes_only_exact_first_six(reserved_case, selected):
    observer, cfg, protocol, now, claim, bundle, reservation = reserved_case
    complete_reserved_freeze(reserved_case, selected=selected)
    result = observer.observe(cfg, now=now)
    assert result['members'] == (0 if selected else 1)
    assert result['pending_development_reservations'] == 0
    if selected:
        assert 'ORIGINAL_FIRST_SIX_DEVELOPMENT_RESERVATION' in (Path(protocol['state_root'])/'events.jsonl').read_text()


@pytest.mark.parametrize('status', ['INDEX_MISSING', 'INDEX_STALE', 'INDEX_INCOMPLETE',
                                   'FREEZE_INTERRUPTED', 'SOURCE_OR_AUTHORITY_UNAVAILABLE'])
def test_failed_freeze_preserves_date_disposition_and_resumes_consideration(reserved_case, status):
    observer, cfg, protocol, now, claim, bundle, reservation = reserved_case
    observer.observe(cfg, now=now)
    root = Path(reservation['state_root'])/'2026-10-10'
    failure = put(root/'date-accounting.json', {'local_date': '2026-10-10', 'status': status,
        'reason': 'NO_COMPLETED_FREEZE', 'population_sha256': None})
    # An interrupted freeze may leave a partial artifact, never a selected cohort.
    (root/'population.json').write_bytes(b'{interrupted')
    result = observer.observe(cfg, now=now)
    assert result['members'] == 1 and result['pending_development_reservations'] == 0
    text = (Path(protocol['state_root'])/'events.jsonl').read_text()
    assert 'DEVELOPMENT_SELECTION_PENDING' in text and status in text
    assert observer.reference(root/'date-accounting.json') == failure


def test_prior_observer_member_and_identity_are_preserved_on_successor_adoption(reserved_case):
    observer, cfg, protocol, now, claim, bundle, reservation = reserved_case
    original_cfg = json.loads(Path(reservation['predecessor_observer_config']['path']).read_bytes())
    assert observer.observe(original_cfg, now=now)['members'] == 1
    journal = Path(protocol['state_root'])/'events.jsonl'
    before = journal.read_bytes()
    cfg['study_amendment'] = {'path': '/synthetic/successor-amendment.json', 'sha256': 'f'*64}
    result = observer.observe(cfg, now=now)
    assert result['members'] == 1 and result['new_members'] == 0
    assert journal.read_bytes().startswith(before)
    assert len([r for r in journal.read_text().splitlines() if '"kind":"MEMBER"' in r]) == 1
    assert 'DEVELOPMENT_RESERVATION_BINDING' in journal.read_text()


@pytest.mark.parametrize('authority', ['allocation', 'exclusive_amendment', 'plan', 'predecessor_observer_config'])
def test_changed_reservation_authority_holds_without_new_admission(reserved_case, authority):
    observer, cfg, protocol, now, claim, bundle, reservation = reserved_case
    observer.observe(cfg, now=now)
    journal = Path(protocol['state_root'])/'events.jsonl'
    before = journal.read_bytes()
    Path(reservation[authority]['path']).write_bytes(b'{}')
    with pytest.raises(ValueError):
        observer.observe(cfg, now=now)
    assert journal.read_bytes() == before


@pytest.mark.parametrize('change', ['configuration', 'remove_binding', 'predecessor'])
def test_successor_cannot_widen_scope_or_rebind_prior_journal(reserved_case, change):
    observer, cfg, protocol, now, claim, bundle, reservation = reserved_case
    observer.observe(cfg, now=now)
    journal = Path(protocol['state_root'])/'events.jsonl'
    before = journal.read_bytes()
    if change == 'configuration':
        cfg['slots'] = ['2026-10-10T13:00:00+11:00']
    elif change == 'remove_binding':
        cfg.pop('development_reservations')
    else:
        previous = json.loads(Path(reservation['predecessor_observer_config']['path']).read_bytes())
        previous['study_amendment']['sha256'] = '0'*64
        reservation['predecessor_observer_config'] = put(Path(reservation['predecessor_observer_config']['path']), previous)
        cfg['development_reservations'] = put(Path(cfg['development_reservations']['path']), reservation)
    with pytest.raises(ValueError):
        observer.observe(cfg, now=now)
    assert journal.read_bytes() == before


@pytest.mark.parametrize('target', ['population.json', 'original-population.json',
                                   'original-population.json.completion.json', 'protected-membership.json',
                                   'date-accounting.json'])
def test_changed_frozen_census_or_disposition_fails_closed(reserved_case, target):
    observer, cfg, protocol, now, claim, bundle, reservation = reserved_case
    root = complete_reserved_freeze(reserved_case)
    observer.observe(cfg, now=now)
    journal = Path(protocol['state_root'])/'events.jsonl'
    before = journal.read_bytes()
    (root/target).write_bytes(b'{}')
    with pytest.raises(ValueError):
        observer.observe(cfg, now=now)
    assert journal.read_bytes() == before


def test_selected_key_cannot_escape_reservation_by_a_different_jump_date(reserved_case):
    observer, cfg, protocol, now, claim, bundle, reservation = reserved_case
    complete_reserved_freeze(reserved_case)
    # Both native admission/completion and request agree on the changed jump;
    # the immutable development census still owns this exact canonical key.
    admission = json.loads((claim/'admission.json').read_bytes())
    admission['race']['jump_timestamp'] = '2026-10-11T02:20:00+00:00'
    admission['admitted_at'] = '2026-10-11T02:11:00+00:00'
    admission['decision_at'] = '2026-10-11T02:18:00+00:00'
    ar = put(claim/'admission.json', admission)
    request = json.loads((bundle/'request.json').read_bytes())
    request['jump_timestamp'] = admission['race']['jump_timestamp']
    rr = put(bundle/'request.json', request)
    manifest = json.loads((bundle/'bundle_manifest.json').read_bytes())
    manifest['files']['request.json'] = {'sha256': rr['sha256'], 'bytes': (bundle/'request.json').stat().st_size}
    mr = put(bundle/'bundle_manifest.json', manifest)
    completion = json.loads((claim/'completion.json').read_bytes())
    completion.update(admission, admission_sha256=ar['sha256'], published_complete_at='2026-10-11T02:12:00+00:00')
    completion['bundle_entry']['manifest_sha256'] = mr['sha256']
    put(claim/'completion.json', completion)
    put(Path(reservation['state_root'])/'2026-10-11'/'date-accounting.json', {
        'local_date': '2026-10-11', 'status': 'INDEX_MISSING', 'reason': 'NO_INDEX', 'population_sha256': None})
    result = observer.observe(cfg, now=datetime(2026, 10, 11, 2, 15, tzinfo=timezone.utc))
    assert result['members'] == 0
    assert 'DEVELOPMENT_RESERVED_IDENTITY_CHANGED' in (Path(protocol['state_root'])/'events.jsonl').read_text()


def test_successor_uses_complete_readiness_gate_and_preserves_existing_identity(reserved_case, monkeypatch):
    from race_collection import retained_study_readiness as readiness
    from race_collection.live_freshness_contract import digest
    from tests.test_retained_study_readiness import fixture as readiness_case
    observer, cfg, protocol, now, claim, bundle, reservation = reserved_case
    ready_cfg, ready_at = readiness_case(bundle.parent.parent/'readiness', monkeypatch)
    ready_cfg['retained_study_protocol'] = cfg['retained_study_protocol']
    amendment = readiness.checked(ready_cfg['study_amendment'])
    amendment['target_config_sha256'] = digest({k: v for k, v in ready_cfg.items() if k != 'study_amendment'})
    ready_cfg['study_amendment'] = put(bundle.parent.parent/'original-amendment.json', amendment)
    monkeypatch.setattr(observer, 'verify_retained_readiness', readiness.verify_retained_readiness)
    assert observer.observe(ready_cfg, now=ready_at)['members'] == 0
    journal = Path(protocol['state_root'])/'events.jsonl'
    before = journal.read_bytes()
    reservation['predecessor_observer_config'] = put(Path(reservation['predecessor_observer_config']['path']), ready_cfg)
    successor = {**ready_cfg, 'development_reservations': put(Path(cfg['development_reservations']['path']), reservation)}
    amendment['target_config_sha256'] = digest({k: v for k, v in successor.items() if k != 'study_amendment'})
    successor['study_amendment'] = put(bundle.parent.parent/'successor-amendment.json', amendment)
    result = observer.observe(successor, now=now)
    assert result['members'] == 0 and result['pending_development_reservations'] == 1
    assert journal.read_bytes().startswith(before)
    # A source mismatch still fails in the original readiness gate.
    successor['source_commit'] = 'f'*40
    amendment['target_config_sha256'] = digest({k: v for k, v in successor.items() if k != 'study_amendment'})
    successor['study_amendment'] = put(bundle.parent.parent/'invalid-amendment.json', amendment)
    before = journal.read_bytes()
    with pytest.raises(ValueError, match='source_incompatible'):
        observer.observe(successor, now=now)
    assert journal.read_bytes() == before
