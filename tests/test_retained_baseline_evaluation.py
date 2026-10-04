"""Fabricated records only; never load production forecast or result values."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from race_collection import retained_baseline_evaluation as baseline

NOW = datetime(2026, 10, 4, 8, tzinfo=timezone.utc)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(baseline.canonical(value))
    return baseline.reference(path)


def test_frozen_membership_is_outcome_blind_and_retains_original_denominators(tmp_path):
    protocol = put(tmp_path/'protocol.json', {'schema_version': 'retained_study_protocol_v1',
        'status': 'AUTHORIZED_OUTCOME_BLIND_RETAINED_STUDY', 'state_root': str(tmp_path/'observer'),
        'prior_scientific_capture_attempts': 17, 'opportunity_evidence': []})
    journal = tmp_path/'observer'/'events.jsonl'; journal.parent.mkdir()
    events = [{'kind': 'IDENTITY', 'protocol': protocol}, {'kind': 'UNADMITTED_NATIVE_ATTEMPT', 'reason': 'NO_NATIVE_ADMISSION'}]
    for number, day in [(1, '2026-10-03'), (2, '2026-10-04')]:
        ar = put(tmp_path/f'admission{number}.json', {'race': {'race_date': day}})
        events.append({'kind': 'MEMBER', 'race_id': str(number), 'job_id': str(number),
            'admission': ar, 'selection_at': NOW.isoformat()})
    previous = '0'*64
    with journal.open('wb') as f:
        for i, event in enumerate(events):
            row = {'sequence': i, 'previous': previous, 'event': event}
            row['sha256'] = hashlib.sha256(baseline.canonical(row)).hexdigest()
            previous = row['sha256'];f.write(baseline.canonical(row)+b'\n')
    result = baseline.freeze_membership(protocol, journal, through_date='2026-10-03',
        output=tmp_path/'proposal', now=NOW)
    manifest = baseline.checked(result)
    assert [m['race_id'] for m in manifest['members']] == ['1']
    assert manifest['prior_scientific_capture_attempts'] == 17
    assert manifest['journal_event_counts']['UNADMITTED_NATIVE_ATTEMPT'] == 1
    assert manifest['status'] == 'PROPOSED_NOT_EVALUATION_AUTHORITY'
    assert baseline.load_membership(result)['members'] == manifest['members']
    with pytest.raises(FileExistsError):
        baseline.freeze_membership(protocol, journal, through_date='2026-10-03', output=tmp_path/'proposal', now=NOW)


def test_race_weighted_losses_and_explicit_nonfinisher_win_label():
    field = [{'box_number': 1, 'dog_name': 'A', 'source_native_runner_id': 'entry1'},
             {'box_number': 2, 'dog_name': 'B', 'source_native_runner_id': 'entry2'}]
    result = {'race_id': 'race1', 'identity_verified': True,
        'runner_results': [{**field[0], 'finish_position': 1, 'terminal_status': None},
                           {**field[1], 'finish_position': None, 'terminal_status': 'DNF'}]}
    outcome, category = baseline.win_target('race1', field, result)
    assert outcome == [1.0, 0.0] and category == 'KNOWN_NONFINISH_WIN_ELIGIBLE'
    rows = [{'probabilities': {m: [.8, .2] for m in baseline.MODELS}, 'outcome': outcome},
            {'probabilities': {m: [.5, .5] for m in baseline.MODELS}, 'outcome': [0., 1.]}]
    metrics = baseline.summarize(rows)
    assert metrics['market']['log_loss'] == pytest.approx(0.4581453659370775)
    assert metrics['market']['brier'] == pytest.approx(.29)


@pytest.mark.parametrize('defect', ['dash', 'wrong_native_id', 'unknown_terminal', 'no_winner', 'duplicate_box'])
def test_uncertain_terminal_or_field_identity_never_creates_a_label(defect):
    field = [{'box_number': 1, 'dog_name': 'A', 'source_native_runner_id': '1'},
             {'box_number': 2, 'dog_name': 'B', 'source_native_runner_id': '2'}]
    rows = [{**field[0], 'finish_position': 1, 'terminal_status': None},
            {**field[1], 'finish_position': None, 'terminal_status': 'DNF'}]
    if defect == 'dash': rows[1]['terminal_status'] = '-'
    elif defect == 'wrong_native_id': rows[1]['source_native_runner_id'] = 'wrong'
    elif defect == 'unknown_terminal': rows[1]['terminal_status'] = None
    elif defect == 'no_winner': rows[0]['finish_position'] = 2
    else: rows[1]['box_number'] = 1
    with pytest.raises(ValueError, match='baseline_label'):
        baseline.win_target('r', field, {'race_id': 'r', 'identity_verified': True, 'runner_results': rows})


def test_default_off_never_opens_paths(tmp_path):
    assert baseline.run_baseline({'path': str(tmp_path/'missing'), 'sha256': 'x'}, None,
        execute=False, now=NOW) == {'status': 'DEFAULT_OFF', 'provider_requests': 0, 'result_reads': 0}
    assert not list(tmp_path.iterdir())


def test_authority_rejects_closure_permission_before_output_or_reader(tmp_path, monkeypatch):
    authority = put(tmp_path/'closure.json', {'status': 'USER_AUTHORIZED_PENDING_FINAL_COHORT_BINDING',
        'performance_evaluation': False})
    calls = []
    monkeypatch.setattr(baseline, 'read_member', lambda *args: calls.append(args))
    with pytest.raises(ValueError, match='baseline_evaluation_authority'):
        baseline.run_baseline({'path': str(tmp_path/'missing'), 'sha256': 'x'}, authority,
            execute=True, now=NOW)
    assert not calls and not (tmp_path/'output').exists()


@pytest.fixture
def authorized_case(tmp_path):
    protocol = put(tmp_path/'protocol.json', {'schema_version': 'retained_study_protocol_v1',
        'status': 'AUTHORIZED_OUTCOME_BLIND_RETAINED_STUDY', 'state_root': str(tmp_path/'observer'),
        'prior_scientific_capture_attempts': 17, 'opportunity_evidence': []})
    plan = put(tmp_path/'plan.json', {'programme_root': str(tmp_path/'original'),
        'prediction_output_roots': [str(tmp_path/'bundles')]})
    journal = tmp_path/'observer/events.jsonl';journal.parent.mkdir()
    members = []
    for number in range(3):
        ar = put(tmp_path/f'a{number}.json', {'race': {'race_date': '2026-10-03'}})
        members.append({'kind': 'MEMBER', 'race_id': str(number), 'job_id': str(number),
            'admission': ar, 'selection_at': NOW.isoformat(), 'original_plan': plan})
    events = [{'kind': 'IDENTITY', 'protocol': protocol}, *members];previous = '0'*64
    with journal.open('wb') as f:
        for i, event in enumerate(events):
            row = {'sequence': i, 'previous': previous, 'event': event}
            row['sha256'] = hashlib.sha256(baseline.canonical(row)).hexdigest()
            previous = row['sha256'];f.write(baseline.canonical(row)+b'\n')
    membership = baseline.freeze_membership(protocol, journal, through_date='2026-10-03', output=tmp_path/'proposal', now=NOW)
    records = [{'race_id': str(i), 'state': state} for i, state in enumerate(['CLOSED', 'CLOSED_NON_FINISH', 'QUARANTINED'])]
    verification = put(tmp_path/'verified.json', {'schema_version': 'baseline_closure_verification_v1',
        'independent_identity_verification': True, 'membership': membership, 'result_cutoff': NOW.isoformat(),
        'records_sha256': hashlib.sha256(baseline.canonical(records)).hexdigest()})
    closure = put(tmp_path/'closure.json', {'schema_version': 'sealed_baseline_closure_manifest_v1',
        'status': 'SEALED_INDEPENDENTLY_VERIFIED', 'membership': membership, 'verification_receipt': verification,
        'records': records})
    authority = {'schema_version': 'private_retained_baseline_authority_v1',
        'status': 'AUTHORIZED_ONE_SHOT_PRIVATE_BASELINE', 'authority_reference': 'SYNTHETIC_ONLY',
        'evaluation_id': 'synthetic', 'performance_evaluation': True, 'membership': membership,
        'policy': baseline.POLICY, 'implementation_files': baseline.implementation_pins(),
        'issued_at': NOW.isoformat(), 'expires_at': '2026-10-04T09:00:00+00:00',
        'result_cutoff': NOW.isoformat(), 'provider_requests': 0, 'result_requests': 0,
        'training': False, 'promotion': False, 'human_outcome_access': False, 'public_performance_outputs': False,
        'output_root': str(tmp_path/'private_output'), 'closure_manifest': closure,
        'limits': {'max_races': 3, 'max_files': 100, 'max_bytes': 1000000, 'max_wall_seconds': 30}}
    return membership, authority, tmp_path


def test_isolated_claim_common_denominator_and_private_metrics(authorized_case, monkeypatch):
    membership, authority, root = authorized_case
    ref = put(root/'authority.json', authority)
    def reader(member, closure, protocol, cutoff, **kwargs):
        if closure['state'] == 'QUARANTINED': return None, 'QUARANTINED'
        return {'probabilities': {m: [.8, .2] for m in baseline.MODELS}, 'outcome': [1., 0.]}, closure['state']
    monkeypatch.setattr(baseline, 'read_member', reader)
    result = baseline.run_baseline(membership, ref, execute=True, now=NOW)
    assert result['denominator'] == 3 and result['eligible_races'] == 2 and result['excluded_races'] == 1
    assert 'log_loss' not in json.dumps(result) and 'metrics' not in result
    assert (root/'private_output/private_metrics.json').stat().st_mode & 0o777 == 0o600
    assert (root/'private_output').stat().st_mode & 0o777 == 0o700
    assert not (root/'original').exists()
    # Changing output cannot erase an already consumed claim.
    authority['output_root'] = str(root/'other_output'); ref = put(root/'authority2.json', authority)
    with pytest.raises(FileExistsError): baseline.run_baseline(membership, ref, execute=True, now=NOW)
    assert not (root/'other_output').exists()


@pytest.mark.parametrize('defect', ['expired', 'wrong_manifest', 'wrong_code', 'provider', 'public', 'overlap', 'cap'])
def test_invalid_authority_never_constructs_protected_reader(authorized_case, monkeypatch, defect):
    membership, authority, root = authorized_case
    if defect == 'expired': authority['expires_at'] = NOW.isoformat()
    elif defect == 'wrong_manifest': authority['membership'] = {**membership, 'sha256': '0'*64}
    elif defect == 'wrong_code': authority['implementation_files'] = {}
    elif defect == 'provider': authority['result_requests'] = 1
    elif defect == 'public': authority['public_performance_outputs'] = True
    elif defect == 'overlap': authority['output_root'] = str(root/'original'/'evaluation')
    else: authority['limits']['max_races'] = 2
    calls = []
    monkeypatch.setattr(baseline, 'read_member', lambda *args: calls.append(args))
    with pytest.raises(ValueError):
        baseline.run_baseline(membership, put(root/'authority.json', authority), execute=True, now=NOW)
    assert not calls and not (root/'proposal/evaluation_claim.json').exists()


def test_reader_failure_preserves_claim_without_partial_success(authorized_case, monkeypatch):
    membership, authority, root = authorized_case
    def reader(*args, **kwargs): raise ValueError('synthetic private detail must never escape to status')
    monkeypatch.setattr(baseline, 'read_member', reader)
    with pytest.raises(ValueError):
        baseline.run_baseline(membership, put(root/'authority.json', authority), execute=True, now=NOW)
    assert (root/'proposal/evaluation_claim.json').exists()
    status = json.loads((root/'private_output/status.json').read_text())
    assert status['status'] == 'FAILED_PRESERVED_CLAIM'
    assert 'detail' not in json.dumps(status) and not (root/'private_output/private_metrics.json').exists()


@pytest.fixture
def sealed_member(tmp_path):
    """A tiny fabricated native-shaped seal, including a fabricated SQLite result."""
    import sqlite3
    from race_collection.retained_study_observer import REQUIRED_INPUTS, metadata_candidate
    bundle = tmp_path/'bundles'/'synthetic'
    race = {'race_id': 'Race 1 - BAL - 2026-10-03', 'race_date': '2026-10-03',
        'race_number': 1, 'venue': 'BAL', 'jump_timestamp': '2026-10-03T08:00:00+00:00',
        'url': 'https://www.thedogs.com.au/racing/ballarat/2026-10-03/1/synthetic'}
    plan = {'status': 'AUTHORIZED_ENGINEERING', 'programme_root': str(tmp_path/'original'),
        'prediction_output_roots': [str(tmp_path/'bundles')], 'starts_at': '2026-10-03T00:00:00+00:00',
        'ends_at': '2026-10-04T00:00:00+00:00'}
    plan_ref = put(tmp_path/'plan.json', plan)
    a = {'race': race, 'job_id': 'job', 'prediction_id': 'prediction', 'plan_sha256': plan_ref['sha256'],
        'runner_set_sha256': 'a'*64, 'retained_input_manifest_sha256': 'b'*64,
        'admitted_at': '2026-10-03T07:51:00+00:00', 'decision_at': '2026-10-03T07:58:00+00:00',
        'evidence_class': 'AUTHORIZED_ENGINEERING'}
    field = [{'box_number': i, 'display_name': name, 'identity': name.upper(), 'source_native_runner_id': str(i)}
             for i, name in [(1, 'Alpha'), (2, 'Beta')]]
    from src.predictor.on_demand import sealed_runner_set_sha256
    a['runner_set_sha256'] = sealed_runner_set_sha256(race, field)
    ar = put(tmp_path/'original'/hashlib.sha256(race['race_id'].encode()).hexdigest()/'admission.json', a)
    identity = {k: 'c'*64 for k in ('form_sha256', 'sidecar_sha256', 'odds_receipt_sha256', 'capture_sha256', 'production_feature_rows_sha256')}
    identity['captured_at'] = '2026-10-03T07:51:00+00:00'
    contents = {name: {'fabricated': True} for name in REQUIRED_INPUTS}
    contents.update({'model/model.json': {'fabricated': True}, 'model/manifest.json': {'fabricated': True},
        'comparison/registry.json': {'fabricated': True}, 'comparison/inputs.json': {**identity, 'runners': field},
        'comparison/artifacts/residual_box.json': {'fabricated': True},
        'comparison/artifacts/residual_half.json': {'fabricated': True},
        'features/sealed/implementation_file_manifest.json': {'git_head': 'd'*40},
        'request.json': {**{k: a[k] for k in ('job_id', 'prediction_id', 'runner_set_sha256', 'retained_input_manifest_sha256')},
            'race_id': race['race_id'], 'jump_timestamp': race['jump_timestamp'], 'runners': field},
        'result.json': {'race': race, 'prediction_id': a['prediction_id'], 'generated_at': '2026-10-03T07:52:00+00:00'}})
    files = {}
    for name, value in contents.items():
        ref = put(bundle/name, value);files[name] = {'sha256': ref['sha256'], 'bytes': (bundle/name).stat().st_size}
    records = {}
    for model in baseline.MODELS:
        key = 'model/model.json' if model == 'production' else f'comparison/artifacts/{model}.json'
        records[model] = {**a, 'candidate': model, 'status': 'SEALED', 'failure': None,
            'completed_at': '2026-10-03T07:52:00+00:00', 'admission_sha256': ar['sha256'],
            'input_identity': identity, 'model_sha256': files[key]['sha256'] if model != 'market' else None,
            'predictions': [{'box_number': r['box_number'], 'identity': r['identity'], 'dog_name': r['display_name'],
                            'probability': p} for r, p in zip(field, [.8, .2])]}
        name = f'comparison/{model}.json';ref = put(bundle/name, records[model]);files[name] = {'sha256': ref['sha256'], 'bytes': (bundle/name).stat().st_size}
    manifest = {'job_id': a['job_id'], 'prediction_id': a['prediction_id'], 'files': files}
    mr = put(bundle/'bundle_manifest.json', manifest)
    put(Path(ar['path']).with_name('completion.json'), {**a, 'admission_sha256': ar['sha256'],
        'status': 'COMPLETE_BEFORE_CUTOFF', 'models': dict.fromkeys(baseline.MODELS, 'SEALED'),
        'published_complete_at': '2026-10-03T07:52:00+00:00',
        'bundle_entry': {'directory': bundle.name, 'manifest_sha256': mr['sha256']}})
    protocol = {'frozen_model_files': {n: files[n]['sha256'] for n in ('model/model.json', 'model/manifest.json', 'comparison/registry.json')}}
    member = metadata_candidate(plan_ref, plan, Path(ar['path']), protocol)
    db = tmp_path/'synthetic_results.sqlite3'
    common = {'race_id': race['race_id'], 'race_date': race['race_date'], 'race_number': 1,
        'venue': 'BAL', 'source': 'thedogs_official', 'source_url': race['url'], 'captured_at': '2026-10-03T08:05:00+00:00'}
    result_race = {**common, 'status': 'resulted', 'start_datetime': race['jump_timestamp'], 'winner_box': 1, 'winner_name': 'Alpha',
        'position_count': 2, 'participant_count': 2, 'box_order': [1, 2]}
    runners = [{**common, 'box_number': i, 'dog_name': name, 'source_native_runner_id': str(i),
                'finish_position': i, 'is_winner': i == 1} for i, name in [(1, 'Alpha'), (2, 'Beta')]]
    with sqlite3.connect(db) as connection:
        for table, rows in [('autonomous_official_result_evidence_races', [result_race]), ('autonomous_official_result_evidence_runners', runners)]:
            connection.execute(f'CREATE TABLE {table} (race_id TEXT, row_json TEXT)')
            for row in rows:connection.execute(f'INSERT INTO {table} VALUES (?,?)', (race['race_id'], json.dumps(row)))
    closure = {'race_id': race['race_id'], 'state': 'CLOSED', 'evidence': baseline.reference(db), 'bytes': db.stat().st_size}
    return member, protocol, closure, a, contents['comparison/inputs.json'], records, manifest


def test_real_reader_joins_fabricated_native_seals_to_exact_sqlite_identity(sealed_member):
    member, protocol, closure, *_ = sealed_member
    race, category = baseline.read_member(member, closure, protocol, NOW)
    assert category == 'FULL_ORDER_WIN_ELIGIBLE'
    assert race['outcome'] == [1., 0.] and set(race['probabilities']) == set(baseline.MODELS)


@pytest.mark.parametrize('defect', ['field', 'model', 'late', 'inputs', 'missing', 'probability'])
def test_four_original_records_must_share_identity_and_valid_simplex(sealed_member, defect):
    member, _, _, admission, inputs, records, manifest = sealed_member
    if defect == 'field': records['market']['predictions'][0]['identity'] = 'other'
    elif defect == 'model': records['residual_half']['model_sha256'] = 'wrong'
    elif defect == 'late': records['production']['completed_at'] = admission['decision_at']
    elif defect == 'inputs': records['market']['input_identity'] = {}
    elif defect == 'missing': records.pop('residual_half')
    else: records['market']['predictions'][0]['probability'] = float('nan')
    with pytest.raises((ValueError, KeyError)):
        baseline.join_forecasts(member, admission, inputs, records, manifest)


def test_known_quarantine_does_not_read_result_path(sealed_member):
    member, protocol, _, *_ = sealed_member
    row, reason = baseline.read_member(member, {'state': 'QUARANTINED', 'evidence': {'path': '/does/not/exist'}}, protocol, NOW)
    assert row is None and reason == 'QUARANTINED'


def test_real_reader_accepts_only_explicit_verified_nonfinish_receipt(sealed_member, tmp_path):
    member, protocol, _, admission, *_ = sealed_member
    body = tmp_path/'synthetic_body.html';body.write_bytes(b'<html>FABRICATED TEST ONLY</html>')
    captured = '2026-10-03T08:05:00+00:00';url = admission['race']['url']
    source = {'body': baseline.reference(body),
        'request': put(tmp_path/'request.json', {'url': url, 'at': captured}),
        'response': put(tmp_path/'response.json', {'final_url': url, 'observed_at': captured,
            'status': 200, 'host': 'www.thedogs.com.au', 'content_type': 'text/html', 'retry_headers': {},
            'bytes': body.stat().st_size, 'sha256': baseline.reference(body)['sha256']})}
    record = {'schema_version': 'comparison_known_nonfinish_result_v1', 'state': 'RESULT_KNOWN_NON_FINISH',
        'job_id': member['job_id'], 'race_id': member['race_id'], 'source': 'thedogs_official',
        'source_url': url, 'captured_at': captured, 'result_known': True, 'identity_verified': True, 'full_order_eligible': False,
        'source_evidence': source, 'runner_results': [
            {'box_number': 1, 'dog_name': 'Alpha', 'source_native_runner_id': '1', 'finish_position': 1, 'terminal_status': None},
            {'box_number': 2, 'dog_name': 'Beta', 'source_native_runner_id': '2', 'finish_position': None, 'terminal_status': 'DNF'}]}
    record['evidence_sha256'] = hashlib.sha256(baseline.canonical(record)).hexdigest()
    closure = {'state': 'CLOSED_NON_FINISH', 'evidence': put(tmp_path/'known.json', record)}
    race, reason = baseline.read_member(member, closure, protocol, NOW)
    assert race['outcome'] == [1., 0.] and reason == 'KNOWN_NONFINISH_WIN_ELIGIBLE'
    body.write_bytes(b'changed')
    with pytest.raises(ValueError, match='nonfinish_source_changed'):
        baseline.read_member(member, closure, protocol, NOW)


def test_cli_default_off_does_not_open_missing_files_and_does_not_print_metrics(capsys):
    from scripts.evaluate_retained_baseline import main
    assert main(['--membership', '/missing', '--authority', '/missing']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {'status': 'DEFAULT_OFF', 'provider_requests': 0, 'result_reads': 0}


def test_cli_failure_redacts_all_private_exception_text(monkeypatch, capsys):
    from scripts import evaluate_retained_baseline as cli
    def denied(*args, **kwargs): raise ValueError('SYNTHETIC SECRET METRIC .123')
    monkeypatch.setattr(cli, 'run_baseline', denied)
    assert cli.main(['--execute', '--membership', '/missing', '--membership-sha256', 'a',
                     '--authority', '/missing', '--authority-sha256', 'b']) == 2
    value = capsys.readouterr().out
    assert 'SECRET' not in value and '.123' not in value


def test_result_snapshot_tamper_blocks_real_reader(sealed_member):
    member, protocol, closure, *_ = sealed_member
    Path(closure['evidence']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError): baseline.read_member(member, closure, protocol, NOW)


def test_membership_cannot_drop_a_quarantined_or_unknown_member(authorized_case):
    membership, _, root = authorized_case
    manifest = baseline.checked(membership)
    manifest['members'].pop()
    changed = put(root/'tampered_membership.json', manifest)
    with pytest.raises(ValueError, match='baseline_membership_changed'):
        baseline.load_membership(changed)


def test_fabricated_dead_heat_has_fixed_equal_mass_not_winner_selection():
    field = [{'box_number': i, 'dog_name': str(i), 'source_native_runner_id': str(i)} for i in (1, 2, 3)]
    evidence = {'race_id': 'r', 'identity_verified': True, 'runner_results': [
        {**r, 'finish_position': p, 'terminal_status': None} for r, p in zip(field, [1, 1, 3])]}
    outcome, reason = baseline.win_target('r', field, evidence)
    assert outcome == [.5, .5, 0.] and reason == 'FULL_ORDER_WIN_ELIGIBLE'


def test_unknown_closure_verification_blocks_before_any_reader(authorized_case, monkeypatch):
    membership, authority, root = authorized_case
    closure = baseline.checked(authority['closure_manifest'])
    closure['verification_receipt'] = put(root/'bad_verification.json', {'independent_identity_verification': False})
    authority['closure_manifest'] = put(root/'bad_closure.json', closure)
    called = []
    monkeypatch.setattr(baseline, 'read_member', lambda *args: called.append(args))
    with pytest.raises(ValueError, match='baseline_closure_verification_invalid'):
        baseline.run_baseline(membership, put(root/'authority.json', authority), execute=True, now=NOW)
    assert not called and (root/'proposal/evaluation_claim.json').exists()


def test_expiry_after_opaque_verification_stops_before_forecast_or_result_decode(sealed_member, monkeypatch):
    member, protocol, closure, *_ = sealed_member
    checked_paths = []
    original = baseline.checked
    def tracked(ref):
        checked_paths.append(ref['path']);return original(ref)
    monkeypatch.setattr(baseline, 'checked', tracked)
    calls = 0
    def deadline():
        nonlocal calls
        calls += 1
        if calls == 2: raise TimeoutError('expired_after_opaque_hashing')
    with pytest.raises(TimeoutError):
        baseline.read_member(member, closure, protocol, NOW, check_deadline=deadline)
    assert not any('/comparison/' in p or p.endswith('.sqlite3') for p in checked_paths)
