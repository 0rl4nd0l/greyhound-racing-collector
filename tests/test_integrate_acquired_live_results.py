import copy
import json
from pathlib import Path
import pytest
from scripts import integrate_acquired_live_results as s
from tests.test_project_acquired_live_results import URL, RID


def prior(race='race-a', winner=1):
    return {'race_id': race, 'official_runner_identity_observed': False,
            'runners': [{'box_number': 1, 'is_winner': int(winner == 1), 'finish_position': 1 if winner == 1 else 2},
                        {'box_number': 2, 'is_winner': int(winner == 2), 'finish_position': 1 if winner == 2 else 2}]}


def observation(race='race-a', ready=True):
    return {'race_id': race, 'projection_item': {'race_id': race} if ready else None,
            'status': s.READY if ready else 'TERMINAL_SOURCE_HOLD_NO_RETRY',
            'reason': None if ready else 'SOURCE_DENIAL_OR_RETRY_INSTRUCTION',
            'receipt_reference': {'path': 'synthetic/completion.json', 'sha256': race}}


def test_partial_acquisition_supersedes_only_matching_prior_once():
    old = prior()
    new = {**copy.deepcopy(old), 'official_runner_identity_observed': True}
    selected, dispositions, events = s.integrate({'race-a': 1, 'race-b': 2}, [old], [observation()], lambda _: new)
    assert selected == [new]
    assert old['official_runner_identity_observed'] is False
    assert [r['disposition'] for r in dispositions] == ['NEW_STRICT_RESULT_SUPERSEDES_PRIOR_DIAGNOSTIC', 'NO_RESULT_ACQUIRED']
    assert len(events) == 1


def test_transport_failure_retains_weak_evidence_and_failure():
    old = prior()
    selected, dispositions, _ = s.integrate({'race-a': 1}, [old], [observation(ready=False)], lambda _: pytest.fail('must not project'))
    assert selected == [old]
    assert dispositions[0]['disposition'] == 'PRIOR_DIAGNOSTIC_RETAINED_AFTER_UNAVAILABLE_ACQUISITION'
    assert dispositions[0]['reasons'] == ['SOURCE_DENIAL_OR_RETRY_INSTRUCTION']


def test_consumed_interruption_never_disappears_as_unacquired():
    obs = {**observation(ready=False), 'status': 'CONSUMED_WITHOUT_COMPLETION', 'receipt_reference': None}
    _, dispositions, _ = s.integrate({'race-a': 1}, [], [obs], lambda _: pytest.fail('no response'))
    assert dispositions[0]['disposition'] == 'CONSUMED_INTERRUPTED_WITHOUT_COMPLETION'


def test_new_official_scratch_suppresses_old_diagnostic_without_erasing():
    old = prior()
    def reject(_):
        raise ValueError('forecast_runner_scratched:2')
    selected, dispositions, events = s.integrate({'race-a': 1}, [old], [observation()], reject)
    assert selected == []
    assert old['runners'][0]['is_winner'] == 1
    assert dispositions[0]['prior_projection_sha256']
    assert events[0]['event'] == 'OFFICIAL_FIELD_CHANGED_PRIOR_DIAGNOSTIC_SUPPRESSED'


def test_unqualified_response_isolated_from_valid_other_race():
    def project(obs):
        if obs['race_id'] == 'race-a':
            raise ValueError('official_field_incomplete')
        return {**prior('race-b'), 'official_runner_identity_observed': True}
    selected, dispositions, _ = s.integrate({'race-a': 1, 'race-b': 2}, [], [observation(), observation('race-b')], project)
    assert [r['race_id'] for r in selected] == ['race-b']
    assert dispositions[0]['disposition'] == 'RESPONSE_NOT_QUALIFIED'


def test_conflicting_win_targets_quarantined_instead_of_choosing_latest():
    old, new = prior(), {**prior(winner=2), 'official_runner_identity_observed': True}
    selected, dispositions, _ = s.integrate({'race-a': 1}, [old], [observation()], lambda _: new)
    assert selected == []
    assert dispositions[0]['disposition'] == 'OFFICIAL_RESULT_CONTRADICTION_REQUIRES_REVIEW'
    assert any('win_target_conflict' in x for x in dispositions[0]['reasons'])


def test_distinct_completions_not_selected_by_performance():
    selected, dispositions, _ = s.integrate({'race-a': 1}, [prior()], [observation(), observation()], lambda _: pytest.fail('ambiguous acquisition'))
    assert selected == []
    assert dispositions[0]['disposition'] == 'MULTIPLE_DISTINCT_COMPLETIONS_REQUIRE_REVIEW'


def test_outside_population_is_global_failure_before_projection():
    with pytest.raises(ValueError, match='outside_population'):
        s.integrate({'race-a': 1}, [], [observation('reserved')], lambda _: pytest.fail('not authorized'))


def put(root, name, value):
    path = root / name
    path.write_text(json.dumps(value))
    return s.reference(path)


def chain(tmp_path):
    key = '2026-07-17|HEA|7'
    target = {'race_id': RID, 'canonical_race_key': key, 'deduplication_key': key, 'official_source_url': URL}
    approval = put(tmp_path, 'approval.json', {'authorized': 'synthetic fixture only'})
    queue = put(tmp_path, 'queue.json', {'records': [target]})
    wanted = put(tmp_path, 'wanted.json', {'races': [target]})
    admission = put(tmp_path, 'admission.json', {'status': 'AUTHORIZED_JULY_OWNER_DELEGATION', 'approval_reference': approval,
                    'queue_reference': queue, 'race_manifest_reference': wanted})
    intent = put(tmp_path, 'intent.json', {'schema_version': 'july_result_request_intent_v1', 'status': 'CONSUMED_BEFORE_TRANSPORT',
                  'target': target, 'attempt': 1, 'charged_at': '2026-10-11T00:00:00+00:00', 'admission_reference': admission, 'approval_reference': approval,
                  'queue_reference': queue, 'manifest_reference': wanted})
    request = put(tmp_path, 'request.json', {'method': 'GET', 'url': URL, 'redirects': False,
                   'automatic_transport_retries': 0, 'intent_reference': intent, 'started_at': '2026-10-11T00:00:01+00:00'})
    body = tmp_path / 'response.body'
    body.write_bytes(b'synthetic unread outcome placeholder')
    body_ref = s.reference(body)
    response = put(tmp_path, 'response.json', {'response_sha256': body_ref['sha256'], 'retry_headers': {'date': 'Sun, 11 Oct 2026'}, 'body_truncated': False,
                    'request_started_at': '2026-10-11T00:00:01+00:00', 'response_received_at': '2026-10-11T00:00:02+00:00',
                    'acquired_at': '2026-10-11T00:00:03+00:00', 'body_retained_at': '2026-10-11T00:00:04+00:00'})
    result = put(tmp_path, 'manifest.json', {'target': target, 'intent_reference': intent, 'admission_reference': admission,
                  'approval_reference': approval, 'source_evidence': {'request': request, 'response': response, 'body': body_ref}})
    receipt = {'schema_version': 'july_result_acquisition_terminal_v1', 'intent_reference': intent,
               'admission_reference': admission, 'status': s.READY, 'recorded_at': '2026-10-11T00:00:05+00:00', 'result_reference': result,
               'charged_attempts': 1, 'provider_requests': 1, 'synthetic_transport_calls': 0, 'source_hold': False,
               'projection_records': [{'race_id': RID, 'response_path': str(body), 'metadata_path': response['path']}]}
    ref = put(tmp_path, 'completion.json', receipt)
    return ref, receipt, {RID: 'key'}, {RID: URL}, {RID: key}, admission


def test_actual_interface_reference_chain_can_be_verified_without_body_decode(tmp_path):
    ref, _, allowed, urls, keys, admission = chain(tmp_path)
    obs = s.completion_item(ref, allowed, urls, keys, admission)
    assert obs['projection_item']['response_path'].endswith('response.body')
    assert obs['status'] == s.READY


def test_projection_record_substitution_rejected_before_body_decode(tmp_path):
    ref, receipt, allowed, urls, keys, admission = chain(tmp_path)
    receipt['projection_records'][0]['response_path'] = '/missing/reserved.body'
    ref = put(tmp_path, 'changed-completion.json', receipt)
    with pytest.raises(ValueError, match='projection_paths_mismatch'):
        s.completion_item(ref, allowed, urls, keys, admission)


def test_tampered_queue_hash_rejected(tmp_path):
    ref, _, allowed, urls, keys, admission = chain(tmp_path)
    (tmp_path / 'queue.json').write_text('{}')
    with pytest.raises(ValueError, match='artifact_hash_mismatch'):
        s.completion_item(ref, allowed, urls, keys, admission)


def test_synthetic_transport_cannot_be_presented_as_acquisition(tmp_path):
    ref, receipt, allowed, urls, keys, admission = chain(tmp_path)
    receipt['provider_requests'], receipt['synthetic_transport_calls'] = 0, 1
    ref = put(tmp_path, 'synthetic-completion.json', receipt)
    with pytest.raises(ValueError, match='not_one_actual_provider_request'):
        s.completion_item(ref, allowed, urls, keys, admission)


def test_losing_finish_revision_preserved_without_vetoing_same_win_target():
    old, new = prior(), {**prior(), 'official_runner_identity_observed': True}
    new['runners'][1]['finish_position'] = None
    new['runners'][1]['terminal_status'] = 'DISQ'
    selected, dispositions, events = s.integrate({'race-a': 1}, [old], [observation()], lambda _: new)
    assert selected == [new]
    assert dispositions[0]['reasons'] == []
    assert events[0]['nonblocking_official_revisions'][0]['reason'] == 'NONWINNING_FINISH_REVISION'
    assert old['runners'][1]['finish_position'] == 2


def test_missing_one_response_isolated_from_valid_other_response():
    def project(obs):
        if obs['race_id'] == 'race-a':
            raise FileNotFoundError('retained body missing')
        return {**prior('race-b'), 'official_runner_identity_observed': True}
    selected, dispositions, _ = s.integrate({'race-a': 1, 'race-b': 2}, [], [observation(), observation('race-b')], project)
    assert [r['race_id'] for r in selected] == ['race-b']
    assert dispositions[0]['reasons'] == ['retained body missing']


def rewritten_response(tmp_path, change):
    ref, receipt, allowed, urls, keys, admission = chain(tmp_path)
    result = s.load_reference(receipt['result_reference'])
    response = s.load_reference(result['source_evidence']['response'])
    change(response)
    result['source_evidence']['response'] = put(tmp_path, 'response-revised.json', response)
    receipt['projection_records'][0]['metadata_path'] = result['source_evidence']['response']['path']
    receipt['result_reference'] = put(tmp_path, 'manifest-revised.json', result)
    return put(tmp_path, 'completion-revised.json', receipt), allowed, urls, keys, admission


def test_normal_date_retry_headers_allowed_but_actual_retry_instruction_rejected(tmp_path):
    ref, _, allowed, urls, keys, admission = chain(tmp_path)
    assert s.completion_item(ref, allowed, urls, keys, admission)['status'] == s.READY
    ref, allowed, urls, keys, admission = rewritten_response(tmp_path, lambda response: response.update(retry_headers={'retry-after': '60'}))
    with pytest.raises(ValueError, match='retry_instruction'):
        s.completion_item(ref, allowed, urls, keys, admission)


def test_receipt_chronology_mismatch_rejected_before_body_decode(tmp_path):
    ref, allowed, urls, keys, admission = rewritten_response(tmp_path, lambda response: response.update(response_received_at='2026-10-10T23:59:00+00:00'))
    with pytest.raises(ValueError, match='chronology_mismatch'):
        s.completion_item(ref, allowed, urls, keys, admission)


def test_self_consistent_unapproved_admission_is_rejected(tmp_path):
    ref, _, allowed, urls, keys, admission = chain(tmp_path)
    different_approved = {**admission, 'sha256': 'f' * 64}
    with pytest.raises(ValueError, match='not_pinned_approved_baseline'):
        s.completion_item(ref, allowed, urls, keys, different_approved)


def test_explicit_new_winning_dead_heat_suppresses_old_single_winner():
    def reject(_):
        raise ValueError('official_win_target_ambiguous:multiple_winners')
    selected, dispositions, _ = s.integrate({'race-a': 1}, [prior()], [observation()], reject)
    assert selected == []
    assert dispositions[0]['disposition'] == 'OFFICIAL_FIELD_CHANGED_PRIOR_DIAGNOSTIC_SUPPRESSED'


def test_prior_independent_name_conflict_blocks_but_copied_name_does_not():
    old, new = prior(), {**prior(), 'official_runner_identity_observed': True}
    old['runners'][1]['dog_name'] = 'Earlier dog'
    new['runners'][1]['dog_name'] = 'Source dog'
    old['official_runner_identity_observed'] = True
    assert 'independently_observed_runner_identity_conflict:2' in s.conflicts_with_prior(old, new)['blocking']
    old['official_runner_identity_observed'] = False
    assert s.conflicts_with_prior(old, new)['blocking'] == []


def test_nonwinning_terminal_status_only_revision_is_explicit():
    old, new = prior(), prior()
    old['runners'][1].update(finish_position=None, terminal_status='DNF')
    new['runners'][1].update(finish_position=None, terminal_status='DISQ')
    changes = s.conflicts_with_prior(old, new)
    assert changes['blocking'] == []
    assert changes['nonblocking'][0]['prior_terminal_status'] == 'DNF'
    assert changes['nonblocking'][0]['new_terminal_status'] == 'DISQ'
