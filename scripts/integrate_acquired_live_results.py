#!/usr/bin/env python3
"""Offline integration of hash-bound July completion receipts and frozen forecasts.

Input census: {"completion_references": [{"path": ".../completion.json", "sha256": "..."}]}.
Only the external operational owner acquires responses. This utility never fetches,
changes consumption records, refits models, or regenerates selector thresholds.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import live_benchmark_alignment as alignment
from scripts import project_acquired_live_results as projector
from scripts import project_reviewed_live_results_v2 as reviewed_projector

READY = 'RETAINED_RAW_RESULT_IDENTITY_UNVERIFIED'
FIELD_CONTRADICTIONS = ('forecast_runner_scratched:', 'official_runner_name_changed_at_box:',
                        'official_runner_outside_forecast_field', 'official_reserve_mapping_rejected',
                        'official_win_target_ambiguous')


def reference(path):
    path = Path(path)
    return {'path': str(path), 'sha256': projector.digest(path.read_bytes())}


def load_reference(ref):
    path = Path(ref['path'])
    raw = path.read_bytes()
    if projector.digest(raw) != ref['sha256']:
        raise ValueError('referenced_artifact_hash_mismatch:' + str(path))
    return json.loads(raw)


def verified_intent(ref, allowed, expected_urls, canonical_keys, approved_admission_reference):
    intent = load_reference(ref)
    if intent.get('schema_version') != 'july_result_request_intent_v1' or intent.get('status') != 'CONSUMED_BEFORE_TRANSPORT':
        raise ValueError('unsupported_intent_schema_or_state')
    if intent['admission_reference'] != approved_admission_reference:
        raise ValueError('intent_admission_not_pinned_approved_baseline')
    target = intent['target']
    rid = target['race_id']
    if rid not in allowed or target['canonical_race_key'] != canonical_keys[rid] or target.get('deduplication_key') != canonical_keys[rid]:
        raise ValueError('intent_outside_exact_population')
    if target['official_source_url'] != expected_urls[rid]:
        raise ValueError('intent_target_url_not_prepared')
    admission = load_reference(intent['admission_reference'])
    if admission.get('status') != 'AUTHORIZED_JULY_OWNER_DELEGATION':
        raise ValueError('intent_admission_not_authorized')
    for intent_key, admission_key in [('approval_reference', 'approval_reference'), ('queue_reference', 'queue_reference'), ('manifest_reference', 'race_manifest_reference')]:
        if intent[intent_key] != admission[admission_key]:
            raise ValueError('intent_authority_reference_mismatch:' + intent_key)
    load_reference(intent['approval_reference'])
    queue = load_reference(intent['queue_reference'])
    wanted = load_reference(intent['manifest_reference'])
    for rows in (queue['records'], wanted['races']):
        matching = [{k: row.get(k) for k in target} for row in rows if row.get('race_id') == rid]
        if matching != [target]:
            raise ValueError('intent_target_not_exact_queue_manifest_member')
    if intent.get('attempt') != 1:
        raise ValueError('intent_attempt_not_one')
    return intent


def completion_item(ref, allowed, expected_urls, canonical_keys, approved_admission_reference):
    """Verify metadata chain and membership before any outcome body is parsed."""
    receipt = load_reference(ref)
    if receipt.get('schema_version') != 'july_result_acquisition_terminal_v1':
        raise ValueError('unsupported_completion_schema')
    intent_ref = receipt['intent_reference']
    intent = verified_intent(intent_ref, allowed, expected_urls, canonical_keys, approved_admission_reference)
    target = intent['target']
    rid = target['race_id']
    if rid not in allowed or target['canonical_race_key'] != canonical_keys[rid]:
        raise ValueError('completion_outside_exact_population')
    if target['official_source_url'] != expected_urls[rid]:
        raise ValueError('completion_target_url_not_prepared')
    if receipt['admission_reference'] != intent['admission_reference']:
        raise ValueError('completion_intent_admission_mismatch')
    item = {'race_id': rid, 'receipt_reference': ref, 'intent_reference': intent_ref,
            'status': receipt['status'], 'reason': receipt.get('reason'),
            'charged_attempts': receipt.get('charged_attempts'), 'provider_requests': receipt.get('provider_requests'),
            'source_hold': receipt.get('source_hold'), 'projection_item': None}
    if receipt['status'] != READY:
        if receipt.get('projection_records'):
            raise ValueError('nonready_completion_exposes_projection')
        return item
    if receipt.get('synthetic_transport_calls', 0) or receipt.get('provider_requests') != 1:
        raise ValueError('ready_completion_not_one_actual_provider_request')
    if receipt.get('source_hold') is not False or receipt.get('charged_attempts') != 1:
        raise ValueError('ready_completion_hold_or_consumption_inconsistent')
    projected, evidence = verified_response_chain(receipt, intent, expected_urls[rid])
    if receipt.get('projection_records') != [projected]:
        raise ValueError('completion_projection_paths_mismatch')
    item['projection_item'] = projected
    item['result_reference'] = receipt['result_reference']
    item['source_evidence'] = evidence
    return item


def verified_response_chain(receipt, intent, expected_url):
    target = intent['target']
    rid = target['race_id']
    intent_ref = receipt['intent_reference']
    manifest = load_reference(receipt['result_reference'])
    if manifest['target'] != target or manifest['intent_reference'] != intent_ref or manifest['admission_reference'] != receipt['admission_reference'] or manifest.get('approval_reference') != intent['approval_reference']:
        raise ValueError('result_manifest_identity_chain_mismatch')
    evidence = manifest['source_evidence']
    request = load_reference(evidence['request'])
    response = load_reference(evidence['response'])
    if request.get('method') != 'GET' or request.get('url') != expected_url or request.get('redirects') is not False:
        raise ValueError('result_request_method_or_url_mismatch')
    if request.get('automatic_transport_retries') != 0 or request.get('intent_reference') != intent_ref:
        raise ValueError('result_request_consumption_chain_mismatch')
    retry_keys = {str(key).lower() for key in response.get('retry_headers', {})}
    if response.get('body_truncated') or retry_keys & {'retry-after', 'ratelimit-reset', 'x-ratelimit-reset'}:
        raise ValueError('result_response_truncated_or_retry_instruction')
    chronology = [intent['charged_at'], request['started_at'], response['response_received_at'],
                  response['acquired_at'], response['body_retained_at'], receipt['recorded_at']]
    times = [projector.timestamp(value) for value in chronology]
    if times != sorted(times) or request['started_at'] != response['request_started_at']:
        raise ValueError('completion_request_response_chronology_mismatch')
    if response['response_sha256'] != evidence['body']['sha256']:
        raise ValueError('result_response_body_hash_reference_mismatch')
    projected = {'race_id': rid, 'response_path': evidence['body']['path'], 'metadata_path': evidence['response']['path']}
    return projected, evidence


def reviewed_completion_item(item, event, allowed, expected_urls, canonical_keys, admission_ref):
    """Append a reviewed offline interpretation; never rewrite the transport hold."""
    ref = item['receipt_reference']
    receipt = load_reference(ref)
    intent = verified_intent(receipt['intent_reference'], allowed, expected_urls, canonical_keys, admission_ref)
    rid = intent['target']['race_id']
    if (event.get('race_id') != rid or event.get('completion_reference') != ref
            or event.get('intent_reference') != receipt['intent_reference']
            or event.get('admission_reference') != admission_ref
            or event.get('identity_rule') != reviewed_projector.IDENTITY_RULE
            or event.get('adapter_reference') != reference(Path(reviewed_projector.__file__).resolve())):
        raise ValueError('reviewed_reclassification_identity_binding_mismatch')
    if (receipt.get('status') != 'TERMINAL_SOURCE_HOLD_NO_RETRY'
            or receipt.get('reason') != 'SOURCE_ACCESS_CHALLENGE'
            or receipt.get('source_hold') is not True
            or receipt.get('charged_attempts') != 1 or receipt.get('provider_requests') != 1
            or receipt.get('synthetic_transport_calls') != 0):
        raise ValueError('reviewed_reclassification_not_exact_actual_challenge_hold')
    projected, evidence = verified_response_chain(receipt, intent, expected_urls[rid])
    manifest = load_reference(receipt['result_reference'])
    request, response = load_reference(evidence['request']), load_reference(evidence['response'])
    if (manifest.get('schema_version') != 'july_raw_result_manifest_v1'
            or request.get('schema_version') != 'july_result_http_request_v1'
            or response.get('schema_version') != 'july_result_http_response_v1'):
        raise ValueError('reviewed_reclassification_source_schema_mismatch')
    if (response.get('http_status') != 200 or response.get('requested_url') != expected_urls[rid]
            or response.get('final_url') != expected_urls[rid]
            or event.get('response_reference') != evidence['body']):
        raise ValueError('reviewed_reclassification_response_binding_mismatch')
    review = load_reference(event['independent_review_reference'])
    if (review.get('schema') != 'machine_only_july_challenge_classification_review_v1'
            or review.get('classification') != 'DEMONSTRATED_FALSE_POSITIVE_GENERIC_RECAPTCHA_SCRIPT_REFERENCE'
            or review.get('canonical_race_key') != canonical_keys[rid]
            or review.get('admission_reference') != admission_ref
            or review.get('response_reference') != evidence['body']
            or review.get('original_completion_reference') != ref
            or review.get('source_hold_unchanged') is not True
            or review.get('request_or_retry_authorized') is not False):
        raise ValueError('reviewed_reclassification_independent_review_mismatch')
    criteria = load_reference(event['identity_criteria_reference'])
    if (criteria.get('schema') != 'independent_source_identity_criteria_v1'
            or criteria.get('status') != 'ACCEPTED_BEFORE_METRIC_INSPECTION'
            or criteria.get('identity_rule') != reviewed_projector.IDENTITY_RULE):
        raise ValueError('reviewed_reclassification_identity_criteria_not_frozen')
    return {**item, 'projection_item': projected, 'source_evidence': evidence,
            'result_reference': receipt['result_reference'], 'reviewed_reclassification': event,
            'derived_interpretation': 'OFFLINE_REVIEWED_CLASSIFIER_FALSE_POSITIVE',
            'runtime_source_hold_preserved': True}


def invalid_completion_observation(ref, receipt, intent, error):
    """Evidence processing failure never changes an actual transport terminal."""
    return {'race_id': intent['target']['race_id'], 'receipt_reference': ref,
            'intent_reference': receipt['intent_reference'],
            'status': 'COMPLETION_EVIDENCE_INVALID', 'reason': str(error),
            'original_terminal_status': receipt.get('status'),
            'original_terminal_reason': receipt.get('reason'),
            'source_hold': receipt.get('source_hold'),
            'charged_attempts': receipt.get('charged_attempts'),
            'provider_requests': receipt.get('provider_requests'),
            'projection_item': None}


def conflicts_with_prior(prior, acquired):
    if not prior:
        return {'blocking': [], 'nonblocking': []}
    previous = {r['box_number']: r for r in prior['runners']}
    current = {r['box_number']: r for r in acquired['runners']}
    conflicts, revisions = [], []
    for box, old in previous.items():
        new = current.get(box)
        if new is None:
            conflicts.append('previous_result_box_now_missing:' + str(box))
        elif prior.get('official_runner_identity_observed') is True and projector.token(old.get('dog_name')) != projector.token(new.get('dog_name')):
            conflicts.append('independently_observed_runner_identity_conflict:' + str(box))
        elif bool(old['is_winner']) != bool(new['is_winner']):
            conflicts.append('official_win_target_conflict:' + str(box))
        elif old.get('finish_position') != new.get('finish_position') or (old.get('terminal_status') or ('FINISHED' if old.get('finish_position') is not None else None)) != (new.get('terminal_status') or ('FINISHED' if new.get('finish_position') is not None else None)):
            revisions.append({'reason': 'NONWINNING_FINISH_REVISION', 'box': box,
                              'prior_finish': old.get('finish_position'), 'new_finish': new.get('finish_position'),
                              'prior_terminal_status': old.get('terminal_status') or ('FINISHED' if old.get('finish_position') is not None else None),
                              'new_terminal_status': new.get('terminal_status')})
    return {'blocking': conflicts, 'nonblocking': revisions}


def integrate(population, prior, observations, acquire):
    """One deterministic disposition per race; prior evidence remains a reference.

    acquire(observation) returns a qualified projection or raises ValueError.
    A new field contradiction suppresses the old weak diagnostic without erasing it.
    """
    previous = {r['race_id']: r for r in prior}
    if len(previous) != len(prior) or not set(previous) <= set(population):
        raise ValueError('prior_projection_duplicate_or_outside_population')
    groups = defaultdict(list)
    for observation in observations:
        if observation['race_id'] not in population:
            raise ValueError('observation_outside_population')
        groups[observation['race_id']].append(observation)
    selected, dispositions, events = [], [], []
    for rid in sorted(population):
        old = previous.get(rid)
        rows = groups[rid]
        current, disposition, reasons = old, 'PRIOR_DIAGNOSTIC_RETAINED' if old else 'NO_RESULT_ACQUIRED', []
        nonblocking = []
        if len(rows) > 1:
            current, disposition = None, 'MULTIPLE_DISTINCT_COMPLETIONS_REQUIRE_REVIEW'
        elif rows:
            observation = rows[0]
            if observation['status'] == 'CONSUMED_WITHOUT_COMPLETION':
                disposition = 'CONSUMED_INTERRUPTED_PRIOR_DIAGNOSTIC_RETAINED' if old else 'CONSUMED_INTERRUPTED_WITHOUT_COMPLETION'
                reasons = ['request_consumed_transport_and_result_state_unresolved']
            elif observation['projection_item'] is None:
                disposition = 'PRIOR_DIAGNOSTIC_RETAINED_AFTER_UNAVAILABLE_ACQUISITION' if old else 'ACQUISITION_UNAVAILABLE'
                reasons = [observation.get('reason') or observation['status']]
            else:
                try:
                    new = acquire(observation)
                except (ValueError, KeyError, OSError) as error:
                    reason = str(error)
                    reasons = [reason]
                    if reason.startswith(FIELD_CONTRADICTIONS):
                        current, disposition = None, 'OFFICIAL_FIELD_CHANGED_PRIOR_DIAGNOSTIC_SUPPRESSED'
                    else:
                        disposition = 'PRIOR_DIAGNOSTIC_RETAINED_AFTER_UNQUALIFIED_RESPONSE' if old else 'RESPONSE_NOT_QUALIFIED'
                else:
                    comparison = conflicts_with_prior(old, new)
                    reasons, nonblocking = comparison['blocking'], comparison['nonblocking']
                    if reasons:
                        current, disposition = None, 'OFFICIAL_RESULT_CONTRADICTION_REQUIRES_REVIEW'
                    else:
                        current, disposition = new, 'NEW_STRICT_RESULT_SUPERSEDES_PRIOR_DIAGNOSTIC' if old else 'NEW_STRICT_RESULT'
                        if observation.get('reviewed_reclassification'):
                            disposition = 'REVIEWED_OFFLINE_RESULT_SUPERSEDES_PRIOR_DIAGNOSTIC' if old else 'REVIEWED_OFFLINE_STRICT_RESULT'
        if current is not None:
            selected.append(current)
        entry = {'race_id': rid, 'disposition': disposition, 'reasons': reasons,
                 'nonblocking_official_revisions': nonblocking,
                 'prior_projection_present': old is not None,
                 'prior_projection_sha256': projector.digest(json.dumps(old, sort_keys=True).encode()) if old else None,
                 'completion_observations': rows, 'selected_projection': 'ACQUIRED' if current is not None and current is not old else 'PRIOR' if current is old and old else None}
        dispositions.append(entry)
        if rows:
            events.append({'race_id': rid, 'event': disposition, 'reasons': reasons,
                           'nonblocking_official_revisions': nonblocking,
                           'win_only_policy': 'Independent review before acquisition: losing ranks or terminal statuses cannot veto a complete source-identified unchanged field with the same unique winner.',
                           'prior_projection_sha256': entry['prior_projection_sha256'],
                           'reviewed_reclassifications': [row['reviewed_reclassification'] for row in rows if row.get('reviewed_reclassification')],
                           'original_transport_hold_preserved': any(row.get('runtime_source_hold_preserved') for row in rows),
                           'completion_references': [row['receipt_reference'] for row in rows]})
    return selected, dispositions, events


def run(benchmark_evidence, prior_projection_path, reconciliation_path, completion_manifest_path, output, approved_admission_reference, reviewed_reclassifications_reference=None):
    base = Path(benchmark_evidence)
    boundary = projector.load(base / 'inventory/access-boundary.json')
    membership = projector.load(base / 'inventory/residual-membership.json')
    forecasts = projector.load(base / 'alignment/forecast-records.json')
    by_key = {r['record_key']: r for r in membership['records']}
    allowed, canonical = {}, {}
    for row in forecasts:
        member = by_key[row['prediction_id']]
        if member['access_disposition'] != 'AUTHORIZED_NONRESERVED_RETAINED_OPERATIONAL' or member['canonical_race_key'] in boundary['canonical_protected_identity_keys']:
            raise ValueError('population_not_authorized')
        allowed[row['race_id']] = row['prediction_id']
        canonical[row['race_id']] = member['canonical_race_key']
    if len(allowed) != 210 or len(forecasts) != 210:
        raise ValueError('population_not_exact_210')
    originals = {r['original']['record_key']: r for r in projector.load(base / 'alignment/authenticated-originals.json')}
    expected_urls = {r['race_id']: r['official_source_url'] for r in projector.load(reconciliation_path)['records']}
    previous = projector.load(prior_projection_path)
    manifest = projector.load(completion_manifest_path)
    references = manifest['completion_references']
    intent_references = manifest.get('intent_references', [])
    unique, duplicates = {}, []
    for ref in references:
        # Equal receipt bytes are the same completion even when indexed twice.
        load_reference(ref)
        if ref['sha256'] in unique:
            duplicates.append(ref)
        else:
            unique[ref['sha256']] = ref
    load_reference(approved_admission_reference)
    # Global scope/authority pass before any result body can be decoded.
    completion_metadata = []
    for ref in unique.values():
        receipt = load_reference(ref)
        intent = verified_intent(receipt['intent_reference'], allowed, expected_urls, canonical, approved_admission_reference)
        completion_metadata.append((ref, receipt, intent))
    unique_intents = {}
    for ref in intent_references:
        intent = verified_intent(ref, allowed, expected_urls, canonical, approved_admission_reference)
        unique_intents.setdefault(ref['sha256'], (ref, intent))
    reviewed_events = {}
    if reviewed_reclassifications_reference:
        reviewed = load_reference(reviewed_reclassifications_reference)
        if reviewed.get('schema') != 'reviewed_offline_live_reclassifications_v1':
            raise ValueError('reviewed_reclassification_manifest_schema_mismatch')
        for event in reviewed['records']:
            key = event['completion_reference']['sha256']
            if event['race_id'] not in allowed or key not in unique or key in reviewed_events:
                raise ValueError('reviewed_reclassification_outside_census_or_duplicate')
            reviewed_events[key] = event
    observations = []
    for ref, receipt, intent in completion_metadata:
        try:
            observation = completion_item(ref, allowed, expected_urls, canonical, approved_admission_reference)
            if ref['sha256'] in reviewed_events:
                observation = reviewed_completion_item(observation, reviewed_events[ref['sha256']], allowed, expected_urls, canonical, approved_admission_reference)
        except (ValueError, KeyError, OSError) as error:
            # The race and authority are already verified; broken retained
            # response metadata for it cannot terminate unrelated records.
            observation = invalid_completion_observation(ref, receipt, intent, error)
        observations.append(observation)
    completed_intents = {row['intent_reference']['sha256'] for row in observations}
    for key, (ref, intent) in unique_intents.items():
        if key not in completed_intents:
            observations.append({'race_id': intent['target']['race_id'], 'receipt_reference': None,
                                 'intent_reference': ref, 'status': 'CONSUMED_WITHOUT_COMPLETION',
                                 'charged_attempts': 1, 'provider_requests': None, 'projection_item': None})
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('output_directory_must_be_new_or_empty')
    output.mkdir(parents=True, exist_ok=True)
    def acquire(observation):
        rid = observation['race_id']
        selected_projector = reviewed_projector if observation.get('reviewed_reclassification') else projector
        projection = selected_projector.project_one(observation['projection_item'], originals[allowed[rid]]['original'], expected_urls[rid], output / 'acquired-native' / allowed[rid])
        if observation.get('reviewed_reclassification'):
            projection['reviewed_reclassification'] = observation['reviewed_reclassification']
            projection['original_transport_hold_preserved'] = True
        joined, error = alignment.join_result(originals[allowed[rid]]['original'], projection)
        if error or joined['field_status'] != 'EXACT_UNCHANGED':
            raise ValueError('acquired_alignment_not_strict:' + str(error))
        return projection
    selected, dispositions, events = integrate(allowed, previous['records'], observations, acquire)
    strict, diagnostic = [], []
    for projection in selected:
        original = originals[allowed[projection['race_id']]]
        joined, error = alignment.join_result(original['original'], projection)
        if error:
            raise ValueError('selected_result_alignment_failed:' + str(error))
        target = strict if joined['field_status'] == 'EXACT_UNCHANGED' else diagnostic
        for variant in ('full', 'half'):
            target.append(alignment.benchmark_row(original, joined, variant))
    integrated_projection = {**previous, 'schema': 'integrated_july_acquired_result_projection_v1',
                             'records': selected, 'result_races': len(selected),
                             'missing_race_ids': sorted(set(allowed) - {r['race_id'] for r in selected})}
    projector.write(output / 'integrated-result-projection.json', integrated_projection)
    projector.write(output / 'strict-records.json', strict)
    projector.write(output / 'diagnostic-records.json', diagnostic)
    projector.write(output / 'integration-dispositions.json', dispositions)
    projector.write(output / 'integration-correction-events.json', events)
    projector.write(output / 'completion-index-audit.json', {'original_index_references': references, 'original_intent_references': intent_references, 'duplicate_index_entries': duplicates,
                    'unique_completions': len(unique), 'consumed_without_completion': sum(row['status'] == 'CONSUMED_WITHOUT_COMPLETION' for row in observations), 'consumption_records_modified': False})
    summary = {'schema': 'july_acquired_live_integration_v1', 'original_population': 210,
               'strict_races': len(strict) // 2, 'diagnostic_races': len(diagnostic) // 2,
               'unique_completion_receipts': len(unique), 'duplicate_index_entries': len(duplicates),
               'consumed_without_completion': sum(row['status'] == 'CONSUMED_WITHOUT_COMPLETION' for row in observations),
               'reviewed_offline_reclassifications': sum(bool(row.get('reviewed_reclassification')) for row in observations),
               'dispositions': dict(Counter(r['disposition'] for r in dispositions)),
               'provider_requests_by_integrator': 0, 'fits': 0, 'selector_changes': 0}
    projector.write(output / 'summary.json', summary)
    prior_corrections = Path(prior_projection_path).with_name('correction-trail.json')
    projector.write(output / 'provenance.json', {'prior_correction_reference': reference(prior_corrections) if prior_corrections.exists() else None,
                    'inputs': [reference(p) for p in [prior_projection_path, reconciliation_path, completion_manifest_path,
                    base / 'inventory/access-boundary.json', base / 'inventory/residual-membership.json', base / 'alignment/forecast-records.json', base / 'alignment/authenticated-originals.json']],
                    'approved_admission_reference': approved_admission_reference,
                    'reviewed_reclassifications_reference': reviewed_reclassifications_reference,
                    'implementation': [reference(Path(__file__)), reference(Path(projector.__file__)), reference(Path(reviewed_projector.__file__)), reference(Path(alignment.__file__))],
                    'prior_evidence_preserved': True})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-evidence', type=Path, required=True)
    parser.add_argument('--prior-projection', type=Path, required=True)
    parser.add_argument('--reconciliation', type=Path, required=True)
    parser.add_argument('--completion-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--approved-admission', type=Path, required=True)
    parser.add_argument('--approved-admission-sha256', required=True)
    parser.add_argument('--reviewed-reclassifications', type=Path)
    parser.add_argument('--reviewed-reclassifications-sha256')
    args = parser.parse_args()
    if bool(args.reviewed_reclassifications) != bool(args.reviewed_reclassifications_sha256):
        parser.error('reviewed reclassifications require both path and SHA256')
    reviewed = {'path': str(args.reviewed_reclassifications), 'sha256': args.reviewed_reclassifications_sha256} if args.reviewed_reclassifications else None
    print(json.dumps(run(args.benchmark_evidence, args.prior_projection, args.reconciliation, args.completion_manifest, args.output, {'path': str(args.approved_admission), 'sha256': args.approved_admission_sha256}, reviewed)))


if __name__ == '__main__':
    main()
