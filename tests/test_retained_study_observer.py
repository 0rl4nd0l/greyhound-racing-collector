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
