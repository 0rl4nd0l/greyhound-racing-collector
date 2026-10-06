"""Synthetic native bundles; no provider or result access."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from race_collection import prospective_speed_inputs as inputs
from tests.test_retained_card_timing_coverage import card, put
from tests.test_sectional_speed_candidate import observation
from tests.test_speed_candidate_inputs import case


def native_case(tmp_path, monkeypatch, *, profile='101'):
    member, original = case(tmp_path, profile=profile)
    base = Path(member['bundle_manifest']['path']).parent
    runners = [{'box_number': 1, 'identity': 'ALPHA', 'source_native_runner_id': '201', 'win_odds': 2.0},
               {'box_number': 2, 'identity': 'BETA', 'source_native_runner_id': '202', 'win_odds': 4.0}]
    receipt = {'schema_version': 'on_demand_odds_receipt_v1', 'race_id': member['race_id'],
        'captured_at': '2026-10-01T19:54:00+10:00', 'source_url': 'https://example.invalid/retained',
        'source_hashes': {'source_form_sha256': member['accepted_csv']['sha256'],
            'source_sidecar_sha256': member['sidecar']['sha256'], 'source_report_sha256': 'c'*64},
        'markets': {'win': [{'box_number': r['box_number'], 'identity': r['identity'],
                            'odds_decimal': r['win_odds']} for r in runners]}}
    member['odds_receipt'] = put(base/'odds_receipt.json', receipt)
    member['comparison_inputs'] = put(base/'comparison/inputs.json', {
        'form_sha256': member['accepted_csv']['sha256'], 'sidecar_sha256': member['sidecar']['sha256'],
        'odds_receipt_sha256': member['odds_receipt']['sha256'], 'capture_sha256': 'c'*64,
        'captured_at': receipt['captured_at'], 'runners': runners})
    member['request'] = put(base/'request.json', {'race_id': member['race_id'],
        'jump_timestamp': member['jump_at'], 'runner_set_sha256': member['runner_set_sha256'],
        'model': {'resolved': 'different-production-model', 'model_sha256': 'd'*64},
        'runners': [{k: r[k] for k in ('box_number', 'identity', 'source_native_runner_id')} for r in runners]})
    manifest = json.loads(Path(member['bundle_manifest']['path']).read_bytes())
    for role, relative in [('comparison_inputs', 'comparison/inputs.json'),
                           ('odds_receipt', 'odds_receipt.json'), ('request', 'request.json')]:
        manifest['files'][relative] = member[role]
    member['bundle_manifest'] = put(Path(member['bundle_manifest']['path']), manifest)
    original['bundle_manifest'] = member['bundle_manifest']
    model = {'kind': 'linear', 'l2': 1.0, 'prep': {'center': True, 'names': list(inputs.form.FEATURES),
        'median': [0.0]*16, 'mean': [0.0]*32, 'scale': [1.0]*32}, 'beta': [0.2]+[0.0]*31}
    reference = put(tmp_path/'fixture-model.json', {'base16': model})
    # Test-only artifact identity; production requires the exact saved artifact.
    monkeypatch.setattr(inputs, 'BASELINE_ARTIFACT_SHA256', reference['sha256'])
    return member, original, reference


def predict(case, **kwargs):
    member, original, model = case
    return inputs.forecast(inputs.retained.coverage.Reader(), member, original, model,
        forecast_at='2026-10-01T19:56:00+10:00', **kwargs)


def test_frozen_contract_uses_exact_original_sources():
    contract = inputs.frozen_contract()
    assert contract['candidate_commit'] == 'af55ae9322d08d0f255f767a14e385c794205dfe'
    assert contract['adjustment_strength'] == 0.1
    assert contract['baseline_artifact_sha256'] == 'df6bd595e1905bd67a1da9e5becab10f8b5f3db95707090079421201ccf6861e'
    assert contract['production_model_used_as_baseline'] is False


def test_entire_field_without_support_is_exact_baseline_copy(tmp_path, monkeypatch):
    result = predict(native_case(tmp_path, monkeypatch))
    assert result['speed_features']['supported_runner_count'] == 0
    assert [r['market'] for r in result['predictions']] == [2/3, 1/3]
    assert all(r['baseline_plus_speed'] == r['baseline'] for r in result['predictions'])
    assert result['market']['production_identity']['resolved'] == 'different-production-model'
    assert result['information_cutoff'] == '2026-10-01T19:56:00+10:00'
    assert result['provider_requests'] == result['result_requests'] == result['result_payloads_opened'] == 0
    assert len(result['baseline_rows'][0]['features']) == 16


def test_exact_replay_and_prior_copies_do_not_change_predictions(tmp_path, monkeypatch):
    fixture = native_case(tmp_path, monkeypatch)
    first = predict(fixture)
    second = predict(fixture, prior_observations=first['new_observations'])
    assert second['predictions'] == first['predictions']
    assert second['speed_features']['population_exclusions']['DUPLICATE_COPY'] == 2


def test_supported_runner_adjusted_without_discarding_unsupported_runner(tmp_path, monkeypatch):
    fixture = native_case(tmp_path, monkeypatch, profile=None)
    # Alpha lacks a verified profile identity and receives no direct adjustment.
    peers = [observation(identity=f'peer:{i}', value=6+i/10, track='GUNN', distance=340,
        available='2026-10-01T19:50:00+10:00') for i in range(6)]
    result = predict(fixture, prior_observations=peers)
    assert result['speed_features']['supported_runner_count'] == 1
    assert result['speed_features']['runners'][0]['speed_estimate'] == 0
    assert result['predictions'][0]['baseline_plus_speed'] != result['predictions'][0]['baseline']
    assert any(r['baseline_plus_speed'] != r['baseline'] for r in result['predictions'])
    assert sum(r['baseline_plus_speed'] for r in result['predictions']) == pytest.approx(1.0)


def test_future_capture_cannot_supply_benchmark_or_conflict(tmp_path, monkeypatch):
    fixture = native_case(tmp_path, monkeypatch)
    baseline = predict(fixture)
    future = [observation(identity=f'peer:{i}', value=6+i/10, track='GUNN', distance=340,
        available='2026-10-01T19:57:00+10:00') for i in range(6)]
    result = predict(fixture, prior_observations=future)
    assert result['predictions'] == baseline['predictions']
    assert result['speed_features']['population_exclusions']['EVIDENCE_NOT_BEFORE_CUTOFF'] == 6


@pytest.mark.parametrize('at', ['2026-10-01T19:55:01+10:00', '2026-10-01T20:00:00+10:00'])
def test_forecast_must_follow_native_seal_and_precede_jump(tmp_path, monkeypatch, at):
    member, original, model = native_case(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='FORECAST_NOT_AFTER_SEAL_AND_PREJUMP'):
        inputs.forecast(inputs.retained.coverage.Reader(), member, original, model, forecast_at=at)


@pytest.mark.parametrize('role,field,value,reason', [
    ('comparison_inputs', 'form_sha256', 'a'*64, 'MARKET_SOURCE_BINDING'),
    ('request', 'jump_timestamp', '2026-10-01T21:00:00+10:00', 'MARKET_REQUEST_BINDING'),
    ('request', 'runner_set_sha256', 'a'*64, 'MARKET_REQUEST_BINDING'),
])
def test_valid_hash_does_not_make_semantically_wrong_bundle_acceptable(tmp_path, monkeypatch, role, field, value, reason):
    member, original, model = native_case(tmp_path, monkeypatch)
    body = json.loads(Path(member[role]['path']).read_bytes()); body[field] = value
    member[role] = put(Path(member[role]['path']), body)
    manifest = json.loads(Path(member['bundle_manifest']['path']).read_bytes())
    relative = str(Path(member[role]['path']).relative_to(Path(member['bundle_manifest']['path']).parent))
    manifest['files'][relative] = member[role]
    member['bundle_manifest'] = put(Path(member['bundle_manifest']['path']), manifest)
    original['bundle_manifest'] = member['bundle_manifest']
    with pytest.raises(ValueError, match=reason):
        predict((member, original, model))


def test_changed_baseline_pin_rejected_before_reading_model(tmp_path, monkeypatch):
    fixture = native_case(tmp_path, monkeypatch)
    fixture[2]['sha256'] = 'a'*64
    with pytest.raises(ValueError, match='FROZEN_BASELINE_ARTIFACT_CHANGED'):
        predict(fixture)


def test_original_base16_mapping_rounding_sorting_and_target_exclusion():
    rows = []
    finishes = [8, 4, 1, 2, 3, 7]
    for index, finish in enumerate(finishes):
        rows.append({'Dog Name': '1. Alpha' if index == 0 else '', 'DATE': f'2026-09-{30-index}',
            'TRACK': 'GUNN', 'DIST': '340', 'G': '5', 'PLC': str(finish), 'MGN': str(index+1), 'BOX': '1'})
    rows.append({'Dog Name': '', 'DATE': '2026-10-01', 'TRACK': 'GUNN', 'DIST': '340',
                 'G': '5', 'PLC': '1', 'MGN': '0', 'BOX': '1'})
    rows.append(dict(rows[-2]))  # A repeated historical source copy is not another start.
    packet = {'target': {'race_id': 'Race 1 - GUNN - 2026-10-01', 'date': '2026-10-01',
        'cutoff': '2026-10-01T19:56:00+10:00', 'distance_m': 340,
        'source_card': {'available_at': '2026-10-01T19:55:00+10:00',
            'accepted_csv': {'path': '/fixture', 'sha256': 'a'*64},
            'roster': [{'runner_id': '201', 'box_number': 1, 'block_token': 'ALPHA'}]}}}
    feature_rows, audit = inputs.base16_features(card(rows), {'target_distance': '340m', 'target_grade': '5'}, packet)
    assert feature_rows[0]['features'] == {
        'prior_start_count': 6.0, 'days_since_last_start': 1.0,
        'recent_finish_mean_3': 4.33333333, 'recent_finish_best_5': 1.0,
        'recent_win_rate_5': 0.2, 'recent_place_rate_5': 0.6, 'recent_avg_margin_5': 3.0,
        'career_win_rate': 0.16666667, 'career_place_rate': 0.5,
        'career_avg_finish': 4.16666667, 'starts_same_venue': 6.0,
        'win_rate_same_venue': 0.16666667, 'starts_same_distance': 6.0,
        'win_rate_same_distance': 0.16666667, 'same_grade_start_count': 6.0,
        'same_grade_win_rate': 0.16666667}
    assert audit['runners'][0]['rejections'] == {
        'TARGET_OR_POST_TARGET_HISTORY': 1, 'NORMALIZED_DUPLICATE_HISTORY': 1}


def test_missing_history_remains_missing_feature_values():
    packet = {'target': {'race_id': 'Race 1 - GUNN - 2026-10-01', 'date': '2026-10-01',
        'cutoff': '2026-10-01T19:56:00+10:00', 'distance_m': 340,
        'source_card': {'available_at': '2026-10-01T19:55:00+10:00',
            'accepted_csv': {'path': '/fixture', 'sha256': 'a'*64},
            'roster': [{'runner_id': '201', 'box_number': 1, 'block_token': 'ALPHA'}]}}}
    result, _ = inputs.base16_features(card([{'Dog Name': '1. Alpha'}]), {'target_distance': '340'}, packet)
    assert result[0]['features']['prior_start_count'] == 0
    assert result[0]['features']['recent_finish_best_5'] is None
    assert result[0]['features']['recent_avg_margin_5'] is None


def discovery_case(tmp_path, monkeypatch):
    from src.predictor import future_comparison
    member, original, model = native_case(tmp_path, monkeypatch)
    base = Path(member['bundle_manifest']['path']).parent
    metadata = json.loads(Path(member['sidecar']['path']).read_bytes())
    metadata.update({'accepted_csv_path': str(base/'original.csv'),
        'raw_export_path': member['raw_export']['path'], 'raw_content_length': member['raw_export']['bytes']})
    metadata['primary_race_page_evidence'].update({'raw_path': 'page.html', 'receipt_path': 'receipt.json'})
    member['sidecar'] = put(Path(member['sidecar']['path']), metadata)
    receipt = json.loads(Path(member['odds_receipt']['path']).read_bytes())
    receipt['source_hashes']['source_sidecar_sha256'] = member['sidecar']['sha256']
    member['odds_receipt'] = put(Path(member['odds_receipt']['path']), receipt)
    comparison = json.loads(Path(member['comparison_inputs']['path']).read_bytes())
    comparison.update({'sidecar_sha256': member['sidecar']['sha256'],
        'odds_receipt_sha256': member['odds_receipt']['sha256']})
    member['comparison_inputs'] = put(Path(member['comparison_inputs']['path']), comparison)
    manifest = json.loads(Path(member['bundle_manifest']['path']).read_bytes())
    for role in ('sidecar', 'odds_receipt', 'comparison_inputs'):
        manifest['files'][str(Path(member[role]['path']).relative_to(base))] = member[role]
    manifest['prediction_id'] = 'prediction-fixture'
    member['bundle_manifest'] = put(base/'bundle_manifest.json', manifest)
    admission = json.loads(Path(member['admission']['path']).read_bytes())
    admission.update({'plan_sha256': 'f'*64, 'prediction_id': 'prediction-fixture',
        'bundle_directory': base.name, 'runner_set_sha256': member['runner_set_sha256'],
        'retained_input_manifest_sha256': 'e'*64})
    admission['race']['jump_timestamp'] = member['jump_at']
    member['admission'] = put(Path(member['admission']['path']), admission)
    completion = {key: admission[key] for key in ('race', 'runner_set_sha256', 'plan_sha256',
        'prediction_id', 'retained_input_manifest_sha256')}
    completion.update({'admission_sha256': member['admission']['sha256'],
        'status': 'COMPLETE_BEFORE_CUTOFF', 'published_complete_at': original['original_published_complete_at'],
        'bundle_entry': {'directory': base.name, 'manifest_sha256': member['bundle_manifest']['sha256']}})
    completion_ref = put(base/'completion.json', completion)
    calls = []

    def fake_native_verifier(root, admission_path, *, expected_plan_sha256):
        calls.append((root, admission_path, expected_plan_sha256))
        return {'schema_version': 'verified_four_way_comparison_v1',
            'eligible_common_race': True, 'completion': completion}

    monkeypatch.setattr(future_comparison, 'verify_comparison', fake_native_verifier)
    return member, completion_ref, calls


def test_dynamic_discovery_invokes_native_verifier_and_binds_original_receipts(tmp_path, monkeypatch):
    member, completion_ref, calls = discovery_case(tmp_path, monkeypatch)
    discovered, original = inputs.member_from_native(inputs.retained.coverage.Reader(), tmp_path,
        member['admission'], completion_ref, expected_plan_sha256='f'*64, allowed_source_roots=[tmp_path])
    assert len(calls) == 1
    assert calls[0] == (tmp_path, Path(member['admission']['path']), 'f'*64)
    for key in member:
        assert discovered[key] == member[key]
    assert original['original_published_complete_at'] == '2026-10-01T19:55:01+10:00'
    assert original['completion'] == completion_ref


def test_native_discovery_rejects_plan_before_native_verifier(tmp_path, monkeypatch):
    member, completion_ref, calls = discovery_case(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='NATIVE_PLAN_OUTSIDE_ALLOCATION'):
        inputs.member_from_native(inputs.retained.coverage.Reader(), tmp_path,
            member['admission'], completion_ref, expected_plan_sha256='a'*64, allowed_source_roots=[tmp_path])
    assert calls == []


def test_native_discovery_source_paths_remain_inside_authorized_roots(tmp_path, monkeypatch):
    member, completion_ref, _ = discovery_case(tmp_path, monkeypatch)
    narrower = tmp_path/'unrelated'; narrower.mkdir()
    with pytest.raises(ValueError, match='NATIVE_SOURCE_OUTSIDE_ALLOCATION'):
        inputs.member_from_native(inputs.retained.coverage.Reader(), tmp_path,
            member['admission'], completion_ref, expected_plan_sha256='f'*64, allowed_source_roots=[narrower])


def original_verifier_case(tmp_path, monkeypatch, *, proof_change=None):
    from race_collection.live_freshness_contract import digest
    member, completion_ref, _ = discovery_case(tmp_path, monkeypatch)
    native_plan = put(tmp_path/'native-plan.json', {'fixture': 'original native plan'})
    admission = json.loads(Path(member['admission']['path']).read_bytes())
    admission['plan_sha256'] = native_plan['sha256']
    member['admission'] = put(Path(member['admission']['path']), admission)
    completion = json.loads(Path(completion_ref['path']).read_bytes())
    completion.update({'plan_sha256': native_plan['sha256'], 'admission_sha256': member['admission']['sha256']})
    completion_ref = put(Path(completion_ref['path']), completion)
    package_dir = tmp_path/'original-package'; source = package_dir/'source'
    script = put(source/'src/predictor/future_comparison.py', b'# Fabricated verifier identity for subprocess-boundary tests.\n')
    identity = {'commit': 'b'*40, 'files': {'src/predictor/future_comparison.py': script['sha256']}}
    put(source/'SOURCE_IDENTITY.json', identity)
    package = {'source_root': str(source), 'source_identity_sha256': digest(identity),
        'commit': identity['commit'], 'frozen_comparison': native_plan,
        'prediction_root': str(tmp_path/'prediction-root'), 'python': sys.executable,
        'python_sha256': inputs.hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()}
    # Helper binds bundle root via original prediction_root/bundles. Move only
    # the fixture value; no provider or production directory is involved.
    package['prediction_root'] = str(tmp_path.parent)
    root = tmp_path.parent/'bundles'
    reference = put(package_dir/'plan.json', package)
    proof = {'source_identity_sha256': package['source_identity_sha256'],
        'source_commit': identity['commit'], 'admission_sha256': member['admission']['sha256'],
        'completion_sha256': completion_ref['sha256'], 'verified': {
            'schema_version': 'verified_four_way_comparison_v1', 'eligible_common_race': True,
            'completion': completion}}
    proof.update(proof_change or {})
    calls = []

    def child(command, **kwargs):
        calls.append(command)
        assert command[0] == 'bwrap' and '--unshare-net' in command
        assert '--bind' not in command
        assert command[command.index('--chdir')+1] == str(source)
        pythonpath_index = command.index('PYTHONPATH')
        assert command[pythonpath_index+1] == str(source)
        assert kwargs == {'capture_output': True, 'timeout': 60, 'check': False}
        compile(command[command.index('-c')+1], '<native-verifier-wrapper>', 'exec')
        return SimpleNamespace(returncode=0, stdout=json.dumps(proof).encode(), stderr=b'')

    monkeypatch.setattr(inputs.subprocess, 'run', child)
    return member, completion_ref, reference, package, root, source, calls


def test_original_source_verifier_is_package_bound_and_runs_isolated(tmp_path, monkeypatch):
    member, completion, reference, package, root, _, calls = original_verifier_case(tmp_path, monkeypatch)
    verified = inputs._original_native_verification(inputs.retained.SnapshotReader(), reference,
        root, member['admission'], completion, package['frozen_comparison']['sha256'])
    assert verified['eligible_common_race'] is True
    assert len(calls) == 1


def test_byte_compatible_but_unrelated_package_plan_is_rejected(tmp_path, monkeypatch):
    member, completion, reference, _, root, _, calls = original_verifier_case(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='NATIVE_VERIFIER_PACKAGE_BINDING'):
        inputs._original_native_verification(inputs.retained.SnapshotReader(), reference,
            root, member['admission'], completion, 'a'*64)
    assert calls == []


def test_original_verifier_source_change_fails_before_execution(tmp_path, monkeypatch):
    member, completion, reference, package, root, source, calls = original_verifier_case(tmp_path, monkeypatch)
    (source/'src/predictor/future_comparison.py').write_text('altered source')
    with pytest.raises(ValueError, match='source_package_file_changed'):
        inputs._original_native_verification(inputs.retained.SnapshotReader(), reference,
            root, member['admission'], completion, package['frozen_comparison']['sha256'])
    assert calls == []


def test_original_verifier_proof_must_bind_exact_completion_bytes(tmp_path, monkeypatch):
    member, completion, reference, package, root, _, calls = original_verifier_case(tmp_path, monkeypatch,
        proof_change={'completion_sha256': 'a'*64})
    with pytest.raises(ValueError, match='ORIGINAL_NATIVE_VERIFIER_PROOF_CHANGED'):
        inputs._original_native_verification(inputs.retained.SnapshotReader(), reference,
            root, member['admission'], completion, package['frozen_comparison']['sha256'])
    assert len(calls) == 1
