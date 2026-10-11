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


def reviewed_chain(tmp_path):
    ref, receipt, allowed, urls, keys, admission = chain(tmp_path)
    manifest = s.load_reference(receipt['result_reference'])
    evidence = manifest['source_evidence']
    request = s.load_reference(evidence['request'])
    request['schema_version'] = 'july_result_http_request_v1'
    evidence['request'] = put(tmp_path, 'reviewed-request.json', request)
    response = s.load_reference(evidence['response'])
    response.update(schema_version='july_result_http_response_v1', http_status=200, requested_url=URL, final_url=URL)
    evidence['response'] = put(tmp_path, 'reviewed-response.json', response)
    manifest['schema_version'] = 'july_raw_result_manifest_v1'
    receipt['result_reference'] = put(tmp_path, 'reviewed-manifest.json', manifest)
    receipt.update(status='TERMINAL_SOURCE_HOLD_NO_RETRY', reason='SOURCE_ACCESS_CHALLENGE', source_hold=True)
    receipt.pop('projection_records')
    ref = put(tmp_path, 'held-completion.json', receipt)
    review = {'schema': 'machine_only_july_challenge_classification_review_v1',
              'classification': 'DEMONSTRATED_FALSE_POSITIVE_GENERIC_RECAPTCHA_SCRIPT_REFERENCE',
              'canonical_race_key': keys[RID], 'admission_reference': admission,
              'response_reference': evidence['body'], 'original_completion_reference': ref,
              'source_hold_unchanged': True, 'request_or_retry_authorized': False}
    criteria = {'schema': 'independent_source_identity_criteria_v1', 'status': 'ACCEPTED_BEFORE_METRIC_INSPECTION',
                'identity_rule': s.reviewed_projector.IDENTITY_RULE}
    event = {'race_id': RID, 'completion_reference': ref, 'intent_reference': receipt['intent_reference'],
             'admission_reference': admission, 'response_reference': evidence['body'],
             'identity_rule': s.reviewed_projector.IDENTITY_RULE,
             'adapter_reference': s.reference(Path(s.reviewed_projector.__file__).resolve()),
             'independent_review_reference': put(tmp_path, 'review.json', review),
             'identity_criteria_reference': put(tmp_path, 'criteria.json', criteria)}
    obs = s.completion_item(ref, allowed, urls, keys, admission)
    return obs, event, allowed, urls, keys, admission


def test_reviewed_interpretation_preserves_original_transport_hold_and_consumption(tmp_path):
    obs, event, *args = reviewed_chain(tmp_path)
    before = Path(obs['receipt_reference']['path']).read_bytes()
    assert obs['projection_item'] is None
    derived = s.reviewed_completion_item(obs, event, *args)
    assert derived['projection_item']['response_path'].endswith('response.body')
    assert derived['status'] == 'TERMINAL_SOURCE_HOLD_NO_RETRY'
    assert derived['source_hold'] is True
    assert derived['charged_attempts'] == derived['provider_requests'] == 1
    assert Path(obs['receipt_reference']['path']).read_bytes() == before
    assert obs['projection_item'] is None


@pytest.mark.parametrize('key', ['completion_reference', 'intent_reference', 'admission_reference', 'response_reference', 'adapter_reference'])
def test_reviewed_reclassification_rejects_reference_substitution(tmp_path, key):
    obs, event, *args = reviewed_chain(tmp_path)
    event[key] = {'path': 'other', 'sha256': 'f'*64}
    with pytest.raises(ValueError, match='binding_mismatch'):
        s.reviewed_completion_item(obs, event, *args)


def test_reviewed_reclassification_requires_actual_independent_false_positive(tmp_path):
    obs, event, *args = reviewed_chain(tmp_path)
    review = s.load_reference(event['independent_review_reference'])
    review['classification'] = 'UNRESOLVED'
    event['independent_review_reference'] = put(tmp_path, 'not-reviewed.json', review)
    with pytest.raises(ValueError, match='independent_review_mismatch'):
        s.reviewed_completion_item(obs, event, *args)


def test_reviewed_derived_disposition_keeps_repair_trail(tmp_path):
    obs, event, *args = reviewed_chain(tmp_path)
    derived = s.reviewed_completion_item(obs, event, *args)
    new = {**prior(RID), 'official_runner_identity_observed': True}
    selected, dispositions, events = s.integrate({RID: 1}, [prior(RID)], [derived], lambda _: new)
    assert selected == [new]
    assert dispositions[0]['disposition'] == 'REVIEWED_OFFLINE_RESULT_SUPERSEDES_PRIOR_DIAGNOSTIC'
    assert events[0]['original_transport_hold_preserved'] is True
    assert events[0]['reviewed_reclassifications'] == [event]


@pytest.mark.parametrize('field,value,reason', [
    ('response_received_at', '2026-09-01T00:00:00+00:00', 'chronology_mismatch'),
    ('retry_headers', {'Retry-After': '30'}, 'retry_instruction'),
    ('http_status', 403, 'response_binding_mismatch'),
    ('final_url', URL + '&redirect=true', 'response_binding_mismatch'),
    ('schema_version', 'unrecognized_response', 'source_schema_mismatch'),
])
def test_reviewed_path_rechecks_actual_response_gates(tmp_path, field, value, reason):
    obs, event, allowed, urls, keys, admission = reviewed_chain(tmp_path)
    receipt = s.load_reference(obs['receipt_reference'])
    manifest = s.load_reference(receipt['result_reference'])
    response = s.load_reference(manifest['source_evidence']['response'])
    response[field] = value
    manifest['source_evidence']['response'] = put(tmp_path, 'bad-response.json', response)
    receipt['result_reference'] = put(tmp_path, 'bad-source-manifest.json', manifest)
    ref = put(tmp_path, 'bad-held-completion.json', receipt)
    obs = s.completion_item(ref, allowed, urls, keys, admission)
    event['completion_reference'] = ref
    review = s.load_reference(event['independent_review_reference'])
    review['original_completion_reference'] = ref
    event['independent_review_reference'] = put(tmp_path, 'bad-chain-review.json', review)
    with pytest.raises(ValueError, match=reason):
        s.reviewed_completion_item(obs, event, allowed, urls, keys, admission)


def test_rejected_offline_correction_keeps_actual_source_hold_in_accounting(tmp_path):
    obs, event, allowed, urls, keys, admission = reviewed_chain(tmp_path)
    event['identity_rule'] = 'not_frozen'
    try:
        s.reviewed_completion_item(obs, event, allowed, urls, keys, admission)
    except ValueError as error:
        receipt = s.load_reference(obs['receipt_reference'])
        invalid = s.invalid_completion_observation(obs['receipt_reference'], receipt, s.load_reference(receipt['intent_reference']), error)
    else:
        pytest.fail('invalid correction accepted')
    selected, dispositions, _ = s.integrate({RID: 1}, [prior(RID)], [invalid], lambda _: pytest.fail('must not parse'))
    assert selected == [prior(RID)]
    retained = dispositions[0]['completion_observations'][0]
    assert retained['original_terminal_status'] == 'TERMINAL_SOURCE_HOLD_NO_RETRY'
    assert retained['original_terminal_reason'] == 'SOURCE_ACCESS_CHALLENGE'
    assert retained['source_hold'] is True
    assert retained['charged_attempts'] == retained['provider_requests'] == 1


def admission_population(tmp_path):
    allowed = {f'race-{n}': f'prediction-{n}' for n in range(210)}
    keys = {rid: f'2026-07-17|HEA|{n+1}' for n, rid in enumerate(allowed)}
    urls = {rid: f'https://www.thedogs.com.au/racing/healesville/2026-07-17/{n+1}/results' for n, rid in enumerate(allowed)}
    targets = [{'race_id': rid, 'canonical_race_key': keys[rid], 'deduplication_key': keys[rid], 'official_source_url': urls[rid]} for rid in allowed]
    approval = put(tmp_path, 'multi-approval.json', {'fixture': True})
    request = put(tmp_path, 'multi-request.json', {'fixture': True, 'max_requests': 210})
    population = put(tmp_path, 'multi-population.json', {'records': [{'race_id': rid, 'prediction_id': pid} for rid, pid in allowed.items()]})
    queue = put(tmp_path, 'multi-queue.json', {'records': targets, 'approval_reference': approval, 'request_reference': request, 'population_reference': population})
    wanted = put(tmp_path, 'multi-wanted.json', {'races': targets})
    common = {'schema_version': 'july_result_owner_admission_v1', 'status': 'AUTHORIZED_JULY_OWNER_DELEGATION',
              'state_root': '/synthetic/exact-state', 'source_state': '/synthetic/exact-source-state',
              'maximum_requests': 210, 'maximum_attempts_per_race': 1, 'minimum_request_interval_seconds': 60,
              'automatic_retries': False, 'redirects': False, 'browser_fallback': False,
              'proxy_fallback': False, 'cookies': False, 'borrow_study_budget': False,
              'approval_reference': approval, 'request_reference': request, 'population_reference': population,
              'queue_reference': queue, 'race_manifest_reference': wanted}
    refs = [put(tmp_path, f'multi-admission-{n}.json', {**common, 'runtime_release': str(n)}) for n in range(2)]
    manifest = put(tmp_path, 'multi-approved.json', {'schema': 'approved_july_admission_bindings_v1', 'admissions': refs})
    intents = []
    for n in range(2):
        intents.append(put(tmp_path, f'multi-intent-{n}.json', {
            'schema_version': 'july_result_request_intent_v1', 'status': 'CONSUMED_BEFORE_TRANSPORT',
            'target': targets[n], 'attempt': 1, 'admission_reference': refs[n], 'approval_reference': approval,
            'queue_reference': queue, 'manifest_reference': wanted}))
    return refs, manifest, allowed, urls, keys, intents


def test_mixed_externally_pinned_admissions_verify_each_original_intent(tmp_path):
    refs, manifest, allowed, urls, keys, intents = admission_population(tmp_path)
    approved = s.approved_admission_bindings(None, manifest, allowed, urls, keys)
    for ref, expected in zip(intents, refs):
        pin = s.admission_for_intent(ref, approved)
        assert pin == expected
        assert s.verified_intent(ref, allowed, urls, keys, pin)['admission_reference'] == expected


def test_intent_cannot_self_admit_a_new_authority_or_read_unlisted_path(tmp_path):
    refs, manifest, allowed, urls, keys, intents = admission_population(tmp_path)
    approved = s.approved_admission_bindings(None, manifest, allowed, urls, keys)
    intent = s.load_reference(intents[0])
    intent['admission_reference'] = {'path': '/does/not/exist/unapproved.json', 'sha256': 'f'*64}
    ref = put(tmp_path, 'unapproved-intent.json', intent)
    with pytest.raises(ValueError, match='not_pinned_approved'):
        s.admission_for_intent(ref, approved)


@pytest.mark.parametrize('changed', ['approval_reference', 'queue_reference', 'race_manifest_reference', 'population_reference', 'request_reference'])
def test_mixed_admissions_must_bind_identical_authority_and_scope(tmp_path, changed):
    refs, manifest, allowed, urls, keys, _ = admission_population(tmp_path)
    admission = s.load_reference(refs[1])
    admission[changed] = put(tmp_path, 'changed-binding.json', s.load_reference(admission[changed]))
    refs[1] = put(tmp_path, 'changed-admission.json', admission)
    manifest = put(tmp_path, 'changed-approved.json', {'schema': 'approved_july_admission_bindings_v1', 'admissions': refs})
    with pytest.raises(ValueError, match='scope_or_authority_changed'):
        s.approved_admission_bindings(None, manifest, allowed, urls, keys)


def test_multi_admission_targets_must_match_frozen_210(tmp_path):
    refs, manifest, allowed, urls, keys, _ = admission_population(tmp_path)
    wrong = dict(urls)
    wrong['race-209'] += '?different=true'
    with pytest.raises(ValueError, match='target_population_mismatch'):
        s.approved_admission_bindings(None, manifest, allowed, wrong, keys)


def test_duplicate_or_conflicting_admission_pins_rejected(tmp_path):
    refs, manifest, allowed, urls, keys, _ = admission_population(tmp_path)
    for pins in ([refs[0], refs[0]], [refs[0], {**refs[0], 'sha256': 'f'*64}]):
        manifest = put(tmp_path, 'duplicate-approved.json', {'schema': 'approved_july_admission_bindings_v1', 'admissions': pins})
        with pytest.raises(ValueError, match='duplicate_or_conflicting_pin'):
            s.approved_admission_bindings(None, manifest, allowed, urls, keys)


def test_single_admission_mode_remains_backward_compatible(tmp_path):
    refs, _, allowed, urls, keys, _ = admission_population(tmp_path)
    assert s.approved_admission_bindings(refs[0], None, allowed, urls, keys) == [refs[0]]


def test_external_admission_manifest_hash_is_required(tmp_path):
    _, manifest, allowed, urls, keys, _ = admission_population(tmp_path)
    manifest['sha256'] = 'f'*64
    with pytest.raises(ValueError, match='artifact_hash_mismatch'):
        s.approved_admission_bindings(None, manifest, allowed, urls, keys)


@pytest.mark.parametrize('key,value', [('state_root', '/different'), ('source_state', '/different'),
    ('maximum_requests', 211), ('maximum_attempts_per_race', 2), ('minimum_request_interval_seconds', 30),
    ('automatic_retries', True), ('redirects', True), ('browser_fallback', True), ('proxy_fallback', True)])
def test_multi_admissions_cannot_reset_state_or_widen_request_policy(tmp_path, key, value):
    refs, _, allowed, urls, keys, _ = admission_population(tmp_path)
    changed = s.load_reference(refs[1])
    changed[key] = value
    refs[1] = put(tmp_path, 'changed-policy.json', changed)
    manifest = put(tmp_path, 'changed-policy-pins.json', {'schema': 'approved_july_admission_bindings_v1', 'admissions': refs})
    with pytest.raises(ValueError, match='limits_invalid|limits_changed'):
        s.approved_admission_bindings(None, manifest, allowed, urls, keys)


def frozen_projection_policy_fixture(tmp_path):
    refs, _, allowed, urls, keys, _ = admission_population(tmp_path)
    admission = s.load_reference(refs[0])
    original_allowed, original_urls, original_keys = allowed, urls, keys
    names = {rid: f'Race {n+1} - HEA - 2026-07-17' for n, rid in enumerate(original_allowed)}
    allowed = {names[rid]: pid for rid, pid in original_allowed.items()}
    urls = {names[rid]: url for rid, url in original_urls.items()}
    keys = {names[rid]: key for rid, key in original_keys.items()}
    population = put(tmp_path, 'projection-population.json', {'records': [{'race_id': rid, 'prediction_id': pid} for rid, pid in allowed.items()]})
    policy = {'schema': 'frozen_offline_source_title_projection_policy_v3',
              'status': 'FROZEN_BEFORE_ADDITIONAL_RESULT_DECODING',
              'new_runtime_or_provider_authority': False, 'field_policy_unchanged': True,
              'population_reference': population,
              'independent_review_reference': put(tmp_path, 'policy-review.json', {'status': 'PASS'}),
              'mappings': [{'native_venue': 'HEA', 'official_url_slug': 'healesville',
                            'proposed_exact_title_venues': ['Healesville']}]}
    return policy, allowed, urls, keys


def test_projection_policy_hash_and_exact_frozen_population_checked_before_body(tmp_path):
    policy, allowed, urls, keys = frozen_projection_policy_fixture(tmp_path)
    ref = put(tmp_path, 'policy.json', policy)
    assert s.verified_projection_policy(ref, allowed, urls, keys) == policy
    with pytest.raises(ValueError, match='frozen_population_mismatch'):
        s.verified_projection_policy(ref, {**allowed, 'race-0': 'changed-forecast'}, urls, keys)
    with pytest.raises(ValueError, match='hash_mismatch'):
        s.verified_projection_policy({**ref, 'sha256': 'f'*64}, allowed, urls, keys)


@pytest.mark.parametrize('field,value,reason', [
    ('status', 'PROPOSED', 'not_frozen'),
    ('new_runtime_or_provider_authority', True, 'not_frozen'),
    ('field_policy_unchanged', False, 'not_frozen'),
    ('mappings', [{'native_venue': 'WRONG', 'official_url_slug': 'healesville', 'proposed_exact_title_venues': ['Healesville']}], 'scope_mismatch'),
])
def test_projection_policy_cannot_change_scope_or_admit_unfrozen_rule(tmp_path, field, value, reason):
    policy, allowed, urls, keys = frozen_projection_policy_fixture(tmp_path)
    policy[field] = value
    with pytest.raises(ValueError, match=reason):
        s.verified_projection_policy(put(tmp_path, 'changed-policy.json', policy), allowed, urls, keys)


def test_projection_policy_uses_sealed_native_label_not_allocation_canonical_alias(tmp_path):
    policy, allowed, urls, keys = frozen_projection_policy_fixture(tmp_path)
    names = {rid: rid.replace('HEA', 'SHEPPARTON') for rid in allowed}
    allowed = {names[rid]: pid for rid, pid in allowed.items()}
    urls = {names[rid]: url.replace('healesville', 'shepparton') for rid, url in urls.items()}
    keys = {names[rid]: key.replace('|HEA|', '|SHEP|') for rid, key in keys.items()}
    policy['population_reference'] = put(tmp_path, 'long-name-population.json', {'records': [{'race_id': rid, 'prediction_id': pid} for rid, pid in allowed.items()]})
    policy['mappings'] = [{'native_venue': 'SHEPPARTON', 'official_url_slug': 'shepparton', 'proposed_exact_title_venues': ['Shepparton']}]
    assert s.verified_projection_policy(put(tmp_path, 'native-policy.json', policy), allowed, urls, keys) == policy


def field_policy_fixture(tmp_path):
    title, allowed, _, _ = frozen_projection_policy_fixture(tmp_path)
    originals=tmp_path/'authenticated-originals.json';originals.write_text('[]')
    refs=[put(tmp_path,name+'.json',{'synthetic':name}) for name in ('title','names','navigation')]
    policy={'schema':'frozen_offline_pre_race_field_policy_v6','status':'FROZEN_BEFORE_CORRECTED_METRICS',
            'field_and_scoring_policy_unchanged':True,'no_new_provider_or_runtime_authority':True,
            'population_reference':title['population_reference'], 'authenticated_originals_reference':s.reference(originals),
            'source_title_policy_reference':refs[0],'runner_name_policy_reference':refs[1],
            'structural_identity_policy_reference':refs[2],'independent_review_reference':title['independent_review_reference'],
            'prior_parser_references':[s.reference(module.__file__) for module in (s.projector,s.reviewed_projector,s.policy_projector,s.name_projector,s.navigation_projector)]}
    return policy,allowed,refs,originals


def test_field_policy_pins_all_originals_dependencies_and_population(tmp_path):
    policy,allowed,refs,originals=field_policy_fixture(tmp_path)
    assert s.verified_pre_race_field_policy(put(tmp_path,'field.json',policy),allowed,*refs,originals)==policy
    originals.write_text('["changed"]')
    with pytest.raises(ValueError,match='authenticated_originals_mismatch'):
        s.verified_pre_race_field_policy(put(tmp_path,'field.json',policy),allowed,*refs,originals)


@pytest.mark.parametrize('change,reason',[
 ('status','not_frozen'),('population','population_mismatch'),('dependency','dependency_policy_mismatch'),
 ('parser','prior_parser_mismatch'),('external_hash','hash_mismatch')])
def test_field_policy_rejects_unapproved_scope_or_code(tmp_path,change,reason):
    policy,allowed,refs,originals=field_policy_fixture(tmp_path)
    if change=='status':policy['status']='PROPOSED'
    if change=='population':allowed={**allowed,'outside':'not-a-forecast'}
    if change=='dependency':policy['runner_name_policy_reference']=refs[0]
    if change=='parser':policy['prior_parser_references'][0]['sha256']='0'*64
    ref=put(tmp_path,'field.json',policy)
    if change=='external_hash':ref['sha256']='0'*64
    with pytest.raises(ValueError,match=reason):
        s.verified_pre_race_field_policy(ref,allowed,*refs,originals)


def test_nontrial_policy_binds_prior_readers_originals_and_four_policies(tmp_path):
    policy,allowed,refs,originals=field_policy_fixture(tmp_path)
    field_ref=put(tmp_path,'pre-race-policy.json',policy)
    policy['schema']='frozen_offline_nontrial_navigation_policy_v7'
    policy['pre_race_field_policy_reference']=field_ref
    policy['prior_parser_references'].append(s.reference(s.field_projector.__file__))
    assert s.verified_nontrial_navigation_policy(put(tmp_path,'v7.json',policy),allowed,*refs,field_ref,originals)==policy
    for key,reason in [('pre_race_field_policy_reference','dependency_policy')]:
        changed=dict(policy);changed[key]={'path':'/unapproved','sha256':'0'*64}
        with pytest.raises(ValueError,match=reason):
            s.verified_nontrial_navigation_policy(put(tmp_path,'v7-changed.json',changed),allowed,*refs,field_ref,originals)
    originals.write_text('["changed"]')
    with pytest.raises(ValueError,match='authenticated_originals'):
        s.verified_nontrial_navigation_policy(put(tmp_path,'v7-originals.json',policy),allowed,*refs,field_ref,originals)
    originals.write_text('[]')
    policy['prior_parser_references'][-1]['sha256']='0'*64
    with pytest.raises(ValueError,match='prior_parser_mismatch'):
        s.verified_nontrial_navigation_policy(put(tmp_path,'v7-parser.json',policy),allowed,*refs,field_ref,originals)
