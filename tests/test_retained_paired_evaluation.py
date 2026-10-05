"""Fabricated pairs/labels only; no original forecast or result access."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

import pytest
from race_collection import retained_paired_evaluation as p


def producer_digest(value):
    # Independent exact contract from controlled_retained_inputs.canonical:
    # the producer appends one newline before every digest, unlike baseline JSON.
    data = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
    return hashlib.sha256(data).hexdigest()


def test_native_pair_digest_matches_independent_producer_contract():
    value = {'fixture': [1.0, 0.5]}
    assert p.digest(value) == producer_digest(value)
    assert p.digest(value) != hashlib.sha256(p.observer.canonical(value)).hexdigest()


def test_no_newline_pair_digest_remains_rejected():
    pair, member, inputs, production = fixture_pair()
    unsigned = {k: v for k, v in pair.items() if k != 'derivation_sha256'}
    pair['derivation_sha256'] = hashlib.sha256(p.observer.canonical(unsigned)).hexdigest()
    with pytest.raises(ValueError, match='paired_artifact_invalid'):
        p.align_pair(pair, member, inputs, production)


def fixture_pair():
    member = {'race_id': 'synthetic', 'jump_at': '2026-10-01T20:00:00+10:00',
        'original_published_complete_at': '2026-10-01T19:54:20+10:00',
        'bundle_manifest': {'path': '/fabricated/bundle_manifest.json', 'sha256': 'b'*64},
        'retained_input_manifest_sha256': 'c'*64}
    roster = [{'box_number': 1, 'display_name': 'A', 'identity': 'A', 'source_native_runner_id': 'z'},
        {'box_number': 2, 'display_name': 'B', 'identity': 'B', 'source_native_runner_id': 'a'}]
    inputs = {'runners': [{**r, 'win_odds': 2.5} for r in roster], 'captured_at': '2026-10-01T19:54:00+10:00',
        'form_sha256': '1'*64, 'sidecar_sha256': '2'*64, 'capture_sha256': '3'*64, 'production_feature_rows_sha256': '4'*64}
    production = {'completed_at': '2026-10-01T19:54:15+10:00',
        'predictions': [{'probability': .8}, {'probability': .2}]}
    common = {'membership_sha256': p.MEMBERSHIP_SHA, 'bundle_manifest': member['bundle_manifest'],
        'retained_input_manifest_sha256': 'c'*64, 'input_hashes': {'form_csv_sha256': '1'*64,
        'sidecar_sha256': '2'*64, 'capture_artifact_sha256': '3'*64, 'feature_rows_sha256': '4'*64},
        'native_roster_sha256': producer_digest(roster), 'parent_effective_state_sha256': 'd'*64, 'provenance': {'fixture': True}}
    pair = {'schema_version': 'retrospective_native_v2_controlled_pair_v1',
        'status': 'RETROSPECTIVE_COMPUTATION_NOT_ORIGINAL_PREJUMP_OR_SCIENTIFIC', 'race_id': 'synthetic',
        'derived_at': '2026-10-05T20:00:00+11:00', 'original_score_timestamp': None,
        'historical_validation_anchor': {'at': production['completed_at'], 'role': 'RECORDED_PRODUCTION_COMPLETION_NOT_ORIGINAL_SCORE_TIME'},
        'capture_timing': {'fetch_at': '2026-10-01T19:53:48+10:00', 'append_at': inputs['captured_at'],
        'freshness_basis': 'NATIVE_RECEIPT_APPEND_TIME', 'fetch_lead_seconds': 372., 'append_lead_seconds': 360.},
        'original_published_complete_at': member['original_published_complete_at'],
        'parent_model_sha256': p.MODEL_SHA, 'parent_manifest_sha256': p.MODEL_MANIFEST_SHA,
        'serialization_contract': 'EXACT_NATIVE_V2_CANONICAL_FULL_ROWS_NO_ROUNDING_OR_TOLERANCE',
        'common_inputs': common, 'common_input_sha256': producer_digest(common), 'native_runner_ids': ['a', 'z'],
        'candidates': [{'candidate_id': arm, 'strength': strength, 'common_input_sha256': producer_digest(common), 'probabilities': values}
            for arm, strength, values in [(p.ARMS[0], 1., [.2, .8]), (p.ARMS[1], .5, [.35, .65])]],
        **{k: False for k in ('training', 'performance_evaluation', 'outcomes_present', 'original_forecasts_modified', 'scientific_membership_created')}}
    seal(pair)
    return pair, member, inputs, production


def seal(pair):
    pair['derivation_sha256'] = producer_digest({k: v for k, v in pair.items() if k != 'derivation_sha256'})


def test_native_id_permutation_and_exact_full_arm():
    pair, member, inputs, production = fixture_pair()
    arms = p.align_pair(pair, member, inputs, production)
    assert arms == {p.ARMS[0]: [.8, .2], p.ARMS[1]: [.65, .35]}
    assert pair['candidates'][0]['probabilities'] == [.2, .8]


@pytest.mark.parametrize('defect', ['duplicate_id', 'missing_id', 'wrong_id', 'wrong_member', 'changed_input',
    'wrong_parent', 'changed_common', 'wrong_strength', 'extra_arm', 'full_mismatch', 'rounded_full',
    'nan', 'zero', 'simplex', 'wrong_hash', 'stale_append', 'fetch_after_append'])
def test_pair_integrity_and_freshness_reject(defect):
    pair, member, inputs, production = fixture_pair()
    if defect == 'duplicate_id': pair['native_runner_ids'] = ['a', 'a']
    elif defect == 'missing_id': pair['native_runner_ids'] = ['a']
    elif defect == 'wrong_id': pair['native_runner_ids'] = ['a', 'x']
    elif defect == 'wrong_member': pair['race_id'] = 'other'
    elif defect == 'changed_input': inputs['capture_sha256'] = '0'*64
    elif defect == 'wrong_parent': pair['parent_model_sha256'] = '0'*64
    elif defect == 'changed_common': pair['common_inputs']['parent_effective_state_sha256'] = '0'*64
    elif defect == 'wrong_strength': pair['candidates'][1]['strength'] = .6
    elif defect == 'extra_arm': pair['candidates'].append(deepcopy(pair['candidates'][1]))
    elif defect == 'full_mismatch': pair['candidates'][0]['probabilities'] = [.3, .7]
    elif defect == 'rounded_full': production['predictions'][0]['probability'] = .8000000000000002
    elif defect == 'nan': pair['candidates'][1]['probabilities'][0] = float('nan')
    elif defect == 'zero': pair['candidates'][1]['probabilities'] = [0, 1]
    elif defect == 'simplex': pair['candidates'][1]['probabilities'] = [.2, .2]
    elif defect == 'stale_append': pair['capture_timing']['append_at'] = '2026-10-01T19:40:00+10:00'
    elif defect == 'fetch_after_append': pair['capture_timing']['fetch_at'] = '2026-10-01T19:55:00+10:00'
    if defect != 'nan': seal(pair)
    if defect == 'wrong_hash': pair['derivation_sha256'] = '0'*64
    with pytest.raises(ValueError): p.align_pair(pair, member, inputs, production)


def test_race_weights_direction_brier_sum_and_tied_target():
    field = [{'box_number': i+1, 'dog_name': chr(65+i), 'source_native_runner_id': str(i)} for i in range(3)]
    evidence = {'race_id': 'tie', 'identity_verified': True, 'runner_results': [
        {**r, 'finish_position': place, 'terminal_status': None} for r, place in zip(field, [1, 1, 3])]}
    y, category = p.baseline.win_target('tie', field, evidence)
    assert y == [.5, .5, 0] and category == 'FULL_ORDER_WIN_ELIGIBLE'
    rows = [{'outcome': [1., 0.], 'probabilities': {p.ARMS[0]: [.8, .2], p.ARMS[1]: [.6, .4]}},
        {'outcome': y, 'probabilities': {p.ARMS[0]: [.4, .4, .2], p.ARMS[1]: [.45, .45, .1]}}]
    result = p.summarize_pairs(rows)
    difference = ((-math.log(.6)+math.log(.8))+(-math.log(.45)+math.log(.4)))/2
    assert result['paired_mean_half_minus_full']['log_loss'] == pytest.approx(difference)
    assert result['arm_means'][p.ARMS[0]]['brier_sum'] == pytest.approx((.08+.06)/2)
    assert result['arm_means'][p.ARMS[1]]['brier_sum'] == pytest.approx((.32+.015)/2)


def test_nonfinisher_target_unchanged_and_unknown_rejected():
    field = [{'box_number': i+1, 'dog_name': chr(65+i), 'source_native_runner_id': str(i)} for i in range(2)]
    evidence = {'race_id': 'r', 'identity_verified': True, 'runner_results': [
        {**field[0], 'finish_position': 1, 'terminal_status': None},
        {**field[1], 'finish_position': None, 'terminal_status': 'DNF'}]}
    assert p.baseline.win_target('r', field, evidence) == ([1., 0.], 'KNOWN_NONFINISH_WIN_ELIGIBLE')
    evidence['runner_results'][1]['terminal_status'] = '-'
    with pytest.raises(ValueError): p.baseline.win_target('r', field, evidence)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(p.observer.canonical(value))
    return p.observer.reference(path)


def test_guard_binds_exact_paths_budget_hash_and_restores(tmp_path):
    ref = put(tmp_path/'allowed.json', {'value': 1}); size = Path(ref['path']).stat().st_size
    authority = {'expires_at': (datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat(),
        'limits': {'max_files': 1, 'max_bytes': size, 'max_wall_seconds': 30}}
    scope = {'allowed_files': {ref['path']: {'bytes': size, 'sha256': ref['sha256']}}}
    original = p.observer.raw
    guard = p.ReadGuard(authority, scope)
    with pytest.raises(ValueError, match='paired_read_limit'), guard.installed():
        assert p.observer.checked(ref) == {'value': 1}
        p.observer.checked(ref)
    assert p.observer.raw is original and guard.operations == 2
    guard = p.ReadGuard(authority, scope)
    with pytest.raises(ValueError, match='paired_unbound_input'), guard.installed():
        p.observer.raw(put(tmp_path/'other.json', {})['path'])
    assert p.observer.raw is original


def test_guard_expiry_prevents_read(tmp_path):
    guard = p.ReadGuard({'expires_at': datetime.now(timezone.utc).isoformat(), 'limits': {'max_wall_seconds': 30}}, {'allowed_files': {}})
    with pytest.raises(ValueError, match='paired_deadline'): guard.check()


def test_default_off_does_not_open_any_path(capsys):
    from scripts.evaluate_retained_pairs import main
    assert main(['--authority', '/absent']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'DEFAULT_OFF'

from tests.test_retained_baseline_evaluation import sealed_member, NOW


def test_actual_native_label_reader_composes_with_allowlisted_guard(sealed_member, tmp_path):
    member, protocol, closure, *_ = sealed_member
    allowed = {}
    for path in tmp_path.rglob('*'):
        if path.is_file():
            allowed[str(path)] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    authority = {'expires_at': (datetime.now(timezone.utc)+timedelta(minutes=1)).isoformat(),
        'limits': {'max_files': 1000, 'max_bytes': 10**7, 'max_wall_seconds': 30}}
    guard = p.ReadGuard(authority, {'allowed_files': allowed})
    with guard.installed():
        row, category = p.baseline.read_member(member, closure, protocol, NOW, check_deadline=guard.check)
    assert category == 'FULL_ORDER_WIN_ELIGIBLE' and row['outcome'] == [1., 0.]
    assert guard.operations > 30 and guard.bytes > closure['bytes']*2


def test_quarantine_avoids_label_read_with_guard(sealed_member, tmp_path):
    member, protocol, closure, *_ = sealed_member
    allowed = {str(path): {'bytes': path.stat().st_size} for path in tmp_path.rglob('*')
        if path.is_file() and str(path) != closure['evidence']['path']}
    authority = {'expires_at': (datetime.now(timezone.utc)+timedelta(minutes=1)).isoformat(),
        'limits': {'max_files': 1000, 'max_bytes': 10**7, 'max_wall_seconds': 30}}
    with p.ReadGuard(authority, {'allowed_files': allowed}).installed():
        assert p.baseline.read_member(member, {'state': 'QUARANTINED'}, protocol, NOW) == (None, 'QUARANTINED')


@pytest.fixture
def execution_case(tmp_path, monkeypatch):
    """Real execution/guard/82 pair pipeline, replacing protected label boundary only."""
    members = []; records = []; closures = []
    for i in range(82):
        pair, member, inputs, production = fixture_pair()
        member['race_id'] = pair['race_id'] = f'fabricated-{i}'
        base = tmp_path/'original'/str(i)
        ir = put(base/'comparison/inputs.json', inputs)
        pr = put(base/'comparison/production.json', production)
        br = put(base/'bundle_manifest.json', {'files': {'comparison/inputs.json': {'sha256': ir['sha256']}}})
        member['bundle_manifest'] = br; member['original_forecasts'] = {'production': pr}
        pair['common_inputs']['bundle_manifest'] = br
        pair['common_input_sha256'] = producer_digest(pair['common_inputs'])
        for c in pair['candidates']: c['common_input_sha256'] = pair['common_input_sha256']
        seal(pair)
        members.append(member)
        records.append({'race_id': member['race_id'], 'status': 'EXACT_FULL_ARM_VERIFIED_PAIR_DERIVED',
            'artifact': put(tmp_path/'pairs'/f'{i}.json', pair)})
        closures.append({'race_id': member['race_id'], 'state': 'QUARANTINED' if i >= 74 else ('CLOSED_NON_FINISH' if i == 73 else 'CLOSED')})
    mr = put(tmp_path/'membership.json', {'members': members}); mr['sha256'] = p.MEMBERSHIP_SHA
    protocol = put(tmp_path/'protocol.json', {})
    inventory = put(tmp_path/'inventory.json', {'schema_version': 'retrospective_native_v2_derivation_inventory_v1',
        'status': 'DISPOSITIONS_RECORDED', 'denominator': 82, 'records': records,
        'membership_sha256': p.MEMBERSHIP_SHA, 'source_commit': p.DERIVATION_SOURCE})
    terminal = put(tmp_path/'pair-status.json', {'status': 'RETROSPECTIVE_NATIVE_V2_DERIVATION_COMPLETE',
        'denominator': 82, 'categories': {'EXACT_FULL_ARM_VERIFIED_PAIR_DERIVED': 82}, 'inventory_sha256': inventory['sha256']})
    qualification = put(tmp_path/'qualification.json', {'schema_version': 'native_v2_independent_execution_qualification_v1',
        'status': 'QUALIFIED_RETROSPECTIVE_CONTROLLED_PAIRS', 'source_commit': p.DERIVATION_SOURCE,
        'inventory': inventory, 'terminal': terminal, 'source_clean': True, 'provider_requests': 0,
        'performance_evaluation': False, 'denominator': 82, 'verified_pairs': 82,
        'exact_full_arm_replay_gate_passed': 82, 'exclusions': 0, 'unattempted': 0, 'result_reads': 0})
    closure = put(tmp_path/'closure.json', {'membership': mr, 'records': closures})
    prior = put(tmp_path/'baseline-authority.json', {'membership': mr, 'closure_manifest': closure,
        'result_cutoff': NOW.isoformat(), 'output_root': str(tmp_path/'old-output'),
        'label_provenance_successor': {'label_provenance': {'original_closure_manifest': {}}}})
    status = put(tmp_path/'old-output/status.json', {'membership': mr, 'status': 'PRIVATE_BASELINE_COMPLETE',
        'denominator': 82, 'eligible_races': 74, 'excluded_races': 8, 'categories': p.CATEGORIES})
    pins = p.implementation_pins()
    allowed = {str(path): {'bytes': path.stat().st_size} for path in tmp_path.rglob('*') if path.is_file()}
    source = Path(p.__file__).resolve().parents[1]
    allowed.update({str(source/name): {'bytes': (source/name).stat().st_size, 'sha256': sha} for name, sha in pins.items()})
    limits = {'max_files': 1500, 'max_bytes': 10**7, 'max_wall_seconds': 60, 'max_label_members': 74, 'max_output_bytes': 65536}
    scope = {'schema_version': 'retained_paired_evaluation_inputs_v1', 'policy': p.POLICY, 'membership': mr,
        'expected_categories': p.CATEGORIES, 'baseline_authority': prior, 'baseline_status': status,
        'closure_manifest': closure, 'result_cutoff': NOW.isoformat(), 'pair_inventory': inventory,
        'pair_qualification': qualification, 'protocol': protocol, 'allowed_files': allowed, 'limits': limits,
        'claim_path': str(tmp_path/'scope/paired_evaluation_claim.json'), 'protected_roots': [str(tmp_path/'original')]} 
    sr = put(tmp_path/'scope/scope.json', scope)
    now = datetime.now(timezone.utc)
    authority = {'schema_version': 'private_retained_pair_evaluation_authority_v1',
        'status': 'AUTHORIZED_ONE_SHOT_PRIVATE_PAIRED_EVALUATION', 'policy': p.POLICY,
        'evaluation_id': 'fabricated', 'authority_reference': 'fabricated_user_approval',
        'performance_evaluation': True, 'provider_requests': 0, 'result_requests': 0,
        **{k: False for k in ('training', 'promotion', 'human_outcome_access', 'public_performance_outputs')},
        'issued_at': (now-timedelta(seconds=1)).isoformat(), 'expires_at': (now+timedelta(minutes=5)).isoformat(),
        'input_scope': sr, 'limits': limits, 'protocol': protocol, 'output_root': str(tmp_path/'new-output'),
        'claim_path': scope['claim_path'], 'implementation_files': pins}
    monkeypatch.setattr(p, 'verify_runtime', lambda authority: None)
    monkeypatch.setattr(p.baseline, 'load_membership', lambda ref: {'members': members, 'protocol': protocol})
    monkeypatch.setattr(p.baseline, 'validate_label_provenance', lambda *args: None)
    calls = []
    def labels(member, closure, protocol, cutoff, **kwargs):
        calls.append(member['race_id'])
        if closure['state'] == 'QUARANTINED': return None, 'QUARANTINED'
        category = 'KNOWN_NONFINISH_WIN_ELIGIBLE' if closure['state'] == 'CLOSED_NON_FINISH' else 'FULL_ORDER_WIN_ELIGIBLE'
        return {'probabilities': {'production': [.8, .2]}, 'outcome': [1., 0.]}, category
    monkeypatch.setattr(p.baseline, 'read_member', labels)
    return authority, scope, tmp_path, calls


def test_full82_execution_private_output_and_consumed_claim(execution_case):
    authority, scope, root, calls = execution_case
    ref = put(root/'authority.json', authority)
    result = p.run_paired_evaluation(ref, execute=True)
    assert result['denominator'] == 82 and result['eligible_races'] == 74 and result['excluded_races'] == 8
    assert len(calls) == 82 and 'log_loss' not in json.dumps(result) and 'metrics' not in result
    assert (root/'new-output/private_metrics.json').stat().st_mode & 0o777 == 0o600
    authority['output_root'] = str(root/'another-output')
    with pytest.raises(FileExistsError): p.run_paired_evaluation(put(root/'authority2.json', authority), execute=True)
    assert not (root/'another-output').exists()


@pytest.mark.parametrize('defect', ['pair_tamper', 'unqualified', 'read_limit', 'scope_cap', 'expired', 'provider', 'wrong_code', 'eligibility', 'overlap'])
def test_failed_execution_preserves_claim_and_never_publishes_complete(execution_case, monkeypatch, defect):
    authority, scope, root, calls = execution_case
    if defect == 'pair_tamper': (root/'pairs/81.json').write_text('{}')
    elif defect == 'unqualified': (root/'qualification.json').write_text('{}')
    elif defect == 'read_limit':
        scope['limits']['max_files'] = 2
        authority['input_scope'] = put(root/'scope/scope.json', scope)
    elif defect == 'scope_cap': authority['limits'] = {**authority['limits'], 'max_label_members': 75}
    elif defect == 'expired': authority['expires_at'] = authority['issued_at']
    elif defect == 'provider': authority['result_requests'] = 1
    elif defect == 'wrong_code': authority['implementation_files'] = {}
    elif defect == 'overlap': authority['output_root'] = str(root/'original'/'new-output')
    else: monkeypatch.setattr(p.baseline, 'read_member', lambda *a, **k: (None, 'QUARANTINED'))
    with pytest.raises((ValueError, RuntimeError)):
        p.run_paired_evaluation(put(root/'authority.json', authority), execute=True)
    status = root/'new-output/status.json'
    if status.exists(): assert json.loads(status.read_text())['status'] == 'FAILED_PRESERVED_CLAIM'
    assert not (root/'new-output/private_metrics.json').exists()
    if defect in {'scope_cap', 'expired', 'provider', 'overlap'}: assert not calls and not Path(scope['claim_path']).exists()
