"""Default-off retrospective loss adapter. Original journals and seals are read-only.

Membership preparation reads metadata only. Execution requires a separately
pinned evaluation authority; closure/observer permission never authorizes it.
"""
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

from race_collection.retained_study_observer import canonical, checked, raw, reference, root_path
from src.predictor.future_comparison import stamp

MODELS = ('market', 'production', 'residual_box', 'residual_half')
POLICY = 'RETROSPECTIVE_COMMON_FOUR_WIN_LOGLOSS_BRIER_V1'


def _write(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(canonical(value)+b'\n'); stream.flush(); os.fsync(stream.fileno())


def _events(content, protocol_ref):
    previous = '0'*64; events = []
    for line in content.splitlines():
        row = json.loads(line)
        unsigned = {'sequence': len(events), 'previous': previous, 'event': row['event']}
        previous = hashlib.sha256(canonical(unsigned)).hexdigest()
        if row != {**unsigned, 'sha256': previous}:
            raise ValueError('baseline_journal_changed')
        events.append(row['event'])
    if not events or events[0].get('kind') != 'IDENTITY' or events[0]['protocol'] != protocol_ref:
        raise ValueError('baseline_protocol_changed')
    return events


def _members(events, through_date, selected_at):
    cutoff = date.fromisoformat(through_date); members = []; races = set(); jobs = set()
    for event in events:
        if event['kind'] != 'MEMBER': continue
        if stamp(event['selection_at']) > selected_at:
            raise ValueError('baseline_future_selection')
        admission = checked(event['admission'])
        if date.fromisoformat(admission['race']['race_date']) > cutoff: continue
        if event['race_id'] in races or event['job_id'] in jobs:
            raise ValueError('baseline_duplicate_member')
        races.add(event['race_id']); jobs.add(event['job_id']); members.append(event)
    return members


def freeze_membership(protocol_ref, journal, *, through_date, output, now):
    """Freeze the entire historical membership, never select by result status."""
    protocol = checked(protocol_ref); output = root_path(str(output)); journal = Path(journal)
    if (protocol['schema_version'] != 'retained_study_protocol_v1'
            or protocol['status'] != 'AUTHORIZED_OUTCOME_BLIND_RETAINED_STUDY'
            or journal != Path(protocol['state_root'])/'events.jsonl'):
        raise ValueError('baseline_observer_scope')
    if output.is_relative_to(Path(protocol['state_root'])):
        raise ValueError('baseline_output_overlaps_original')
    content = raw(journal, 64*1024*1024); events = _events(content, protocol_ref)
    members = _members(events, through_date, now)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    # Immutable snapshot, not a live prefix whose hash can change during execution.
    snapshot = output/'observer_snapshot.jsonl'
    descriptor = os.open(snapshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    value = {'schema_version': 'retained_baseline_membership_v1',
        'status': 'PROPOSED_NOT_EVALUATION_AUTHORITY', 'created_at': now.isoformat(),
        'through_date': through_date, 'protocol': protocol_ref,
        'journal_snapshot': reference(snapshot), 'members': members,
        'journal_event_counts': dict(Counter(e['kind'] for e in events)),
        'original_denominator_references': protocol.get('opportunity_evidence', []),
        'prior_scientific_capture_attempts': protocol['prior_scientific_capture_attempts'],
        'policy': POLICY}
    _write(output/'membership.json', value)
    return reference(output/'membership.json')


def load_membership(ref):
    manifest = checked(ref); protocol = checked(manifest['protocol'])
    snap = manifest['journal_snapshot']; content = raw(snap['path'], 64*1024*1024)
    if hashlib.sha256(content).hexdigest() != snap['sha256']:
        raise ValueError('baseline_journal_changed')
    events = _events(content, manifest['protocol'])
    if (manifest['schema_version'] != 'retained_baseline_membership_v1'
            or manifest['policy'] != POLICY
            or manifest['members'] != _members(events, manifest['through_date'], stamp(manifest['created_at']))
            or manifest['journal_event_counts'] != dict(Counter(e['kind'] for e in events))
            or manifest['prior_scientific_capture_attempts'] != protocol['prior_scientific_capture_attempts']
            or manifest['original_denominator_references'] != protocol.get('opportunity_evidence', [])):
        raise ValueError('baseline_membership_changed')
    return manifest


def win_target(race_id, field, evidence):
    """A known non-finisher is not a missing winner; never invent a placing."""
    keys = ('box_number', 'dog_name', 'source_native_runner_id')
    expected = [tuple(r[k] for k in keys) for r in field]
    rows = evidence['runner_results']
    actual = [tuple(r[k] for k in keys) for r in rows]
    if (evidence['race_id'] != race_id or evidence.get('identity_verified') is not True
            or not expected or len(set(r[0] for r in expected)) != len(expected)
            or len(set(r[2] for r in expected)) != len(expected) or any(r[2] is None for r in expected)
            or sorted(actual) != sorted(expected)):
        raise ValueError('baseline_label_identity')
    positions = []; nonfinish = False
    for row in rows:
        finish, terminal = row.get('finish_position'), row.get('terminal_status')
        if type(finish) is int and finish > 0 and terminal is None:
            positions.append(finish)
        elif finish is None and terminal in {'FELL', 'DNF', 'DISQ'}:
            nonfinish = True
        else:
            raise ValueError('baseline_label_terminal_unknown')
    winners = [r['box_number'] for r in rows if r.get('finish_position') == 1]
    if (not winners or (nonfinish and len(winners) != 1)
            or any(p != 1+sum(q < p for q in positions) for p in set(positions))):
        raise ValueError('baseline_label_winner_or_order')
    outcome = [1.0/len(winners) if r['box_number'] in winners else 0.0 for r in field]
    return outcome, 'KNOWN_NONFINISH_WIN_ELIGIBLE' if nonfinish else 'FULL_ORDER_WIN_ELIGIBLE'


def summarize(races):
    """Descriptive race means only. No fitting, bootstrap, subgroup selection."""
    if not races: return None
    metrics = {}
    for model in MODELS:
        losses = []; briers = []
        for race in races:
            p, y = race['probabilities'][model], race['outcome']
            if (len(p) != len(y) or not p or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v < 1 for v in p)
                    or abs(math.fsum(p)-1) > 1e-12 or any(not math.isfinite(v) or v < 0 for v in y)
                    or abs(math.fsum(y)-1) > 1e-12):
                raise ValueError('baseline_probability_invalid')
            losses.append(-math.fsum(a*math.log(b) for a, b in zip(y, p)))
            briers.append(math.fsum((a-b)**2 for a, b in zip(y, p)))
        metrics[model] = {'log_loss': math.fsum(losses)/len(races), 'brier': math.fsum(briers)/len(races)}
    return metrics


def implementation_pins():
    """Pin the adapter and the existing read/validation dependencies, not models."""
    root = Path(__file__).resolve().parents[1]
    names = ('race_collection/retained_baseline_evaluation.py', 'scripts/evaluate_retained_baseline.py',
        'race_collection/retained_study_observer.py', 'src/predictor/future_comparison.py',
        'src/predictor/on_demand.py', 'src/operator_ui/job_store.py', 'utils/csv_metadata.py',
        'src/predictor/comparison_results.py', 'src/operator_ui/journal_results.py')
    return {name: reference(root/name)['sha256'] for name in names}


def authorize(manifest_ref, authority_ref, now):
    if authority_ref is None:
        raise ValueError('baseline_evaluation_authority_missing')
    authority = checked(authority_ref)
    if (authority.get('schema_version') != 'private_retained_baseline_authority_v1'
            or authority.get('status') != 'AUTHORIZED_ONE_SHOT_PRIVATE_BASELINE'
            or authority.get('performance_evaluation') is not True
            or not authority.get('authority_reference') or not authority.get('evaluation_id')):
        raise ValueError('baseline_evaluation_authority_missing')
    if (authority['membership'] != manifest_ref or authority['policy'] != POLICY
            or authority['implementation_files'] != implementation_pins()
            or not stamp(authority['issued_at']) <= now < stamp(authority['expires_at'])
            or stamp(authority['result_cutoff']) > stamp(authority['issued_at'])
            or authority['provider_requests'] != 0 or authority['result_requests'] != 0
            or any(authority[k] is not False for k in ('training', 'promotion', 'human_outcome_access', 'public_performance_outputs'))):
        raise ValueError('baseline_evaluation_authority_invalid')
    manifest = load_membership(manifest_ref)
    limits = authority['limits']
    if (set(limits) != {'max_races', 'max_files', 'max_bytes', 'max_wall_seconds'}
            or any(type(v) is not int or v < 1 for v in limits.values())
            or limits['max_races'] != len(manifest['members'])
            or stamp(manifest['created_at']) > stamp(authority['issued_at'])):
        raise ValueError('baseline_evaluation_limits_invalid')
    output = root_path(authority['output_root'])
    protocol = checked(manifest['protocol'])
    protected = [root_path(protocol['state_root']), Path(__file__).resolve().parents[1]]
    for member in manifest['members']:
        plan = checked(member['original_plan'])
        protected.extend(root_path(p) for p in [plan['programme_root'], *plan['prediction_output_roots']])
    if 'original_study_plan' in protocol:
        original = checked(protocol['original_study_plan'])
        protected.extend(root_path(p) for p in [original['programme_root'], *original['prediction_output_roots']])
    claim_root = root_path(str(Path(manifest_ref['path']).parent))
    if any(candidate == p or candidate.is_relative_to(p) or p.is_relative_to(candidate)
           for candidate in (output, claim_root) for p in protected):
        raise ValueError('baseline_output_overlaps_original')
    if output.exists(): raise ValueError('baseline_output_exists')
    return authority, manifest



def comparison_roster(rows):
    """Validate future_comparison's sole enrichment, retaining the sealed roster."""
    from src.predictor.on_demand import canonical_runner_set
    fields = {'box_number', 'display_name', 'identity', 'source_native_runner_id', 'win_odds'}
    if not isinstance(rows, list):
        raise ValueError('baseline_comparison_runner_schema')
    roster = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError('baseline_comparison_runner_schema')
        odds = row['win_odds']
        if type(odds) not in (int, float) or not math.isfinite(odds) or odds <= 1:
            raise ValueError('baseline_comparison_runner_odds')
        roster.append({key: value for key, value in row.items() if key != 'win_odds'})
    return canonical_runner_set(roster, 'baseline.runners')


def label_provenance_claim(manifest_ref, authority, now):
    """One new label version, anchored to both consumed original attempts."""
    proof = authority['label_provenance_successor']
    keys = {'schema_version', 'original_claim', 'original_authority', 'original_status',
            'corrected_claim', 'corrected_authority', 'corrected_status',
            'corrected_metrics', 'label_provenance'}
    parent = root_path(str(Path(manifest_ref['path']).parent))
    if (set(proof) != keys or proof['schema_version'] != 'baseline_label_provenance_successor_v1'
            or 'corrected_attempt' in authority
            or proof['original_claim']['path'] != str(parent/'evaluation_claim.json')
            or proof['corrected_claim']['path'] != str(parent/'evaluation_claim.corrected-01.json')):
        raise ValueError('baseline_label_successor_scope')
    originals = {}
    same = ('membership', 'policy', 'result_cutoff', 'provider_requests', 'result_requests',
            'performance_evaluation', 'training', 'promotion', 'human_outcome_access',
            'public_performance_outputs')
    for kind in ('original', 'corrected'):
        prior = checked(proof[kind+'_authority']); claim = checked(proof[kind+'_claim'])
        status = checked(proof[kind+'_status'])
        expected_claim_keys = {'claimed_at', 'membership', 'authority', 'output_root'}
        if kind == 'corrected': expected_claim_keys.add('corrected_attempt')
        old_output = root_path(prior['output_root']); output = root_path(authority['output_root'])
        if (set(claim) != expected_claim_keys or claim['authority'] != proof[kind+'_authority']
                or claim['membership'] != manifest_ref or claim['output_root'] != str(old_output)
                or 'label_provenance_successor' in prior
                or any(prior[k] != authority[k] for k in same)
                or prior['schema_version'] != authority['schema_version'] or prior['status'] != authority['status']
                or prior['evaluation_id'] == authority['evaluation_id']
                or not stamp(prior['issued_at']) <= stamp(claim['claimed_at']) < stamp(prior['expires_at'])
                or not stamp(claim['claimed_at']) < stamp(authority['issued_at']) <= now
                or proof[kind+'_status']['path'] != str(old_output/'status.json')
                or status['membership'] != manifest_ref
                or output == old_output or output.is_relative_to(old_output) or old_output.is_relative_to(output)):
            raise ValueError('baseline_label_predecessor')
        originals[kind] = (prior, claim, status)
    original, _, original_status = originals['original']
    corrected, claim, status = originals['corrected']
    correction = corrected.get('corrected_attempt', {})
    if (original_status != {'status': 'FAILED_PRESERVED_CLAIM', 'membership': manifest_ref}
            or 'corrected_attempt' in original
            or claim['corrected_attempt'] != correction
            or correction.get('predecessor_claim') != proof['original_claim']
            or correction.get('predecessor_authority') != proof['original_authority']
            or correction.get('predecessor_status') != proof['original_status']
            or corrected['closure_manifest'] != original['closure_manifest']
            or status['status'] != 'PRIVATE_BASELINE_COMPLETE'
            or status['private_output'] != proof['corrected_metrics']
            or proof['corrected_metrics']['path'] != str(Path(corrected['output_root'])/'private_metrics.json')
            or reference(proof['corrected_metrics']['path']) != proof['corrected_metrics']
            or authority['closure_manifest'] == original['closure_manifest']):
        raise ValueError('baseline_label_consumed_attempts')
    adapter = 'race_collection/retained_baseline_evaluation.py'
    before, after = corrected['implementation_files'], authority['implementation_files']
    if (set(before) != set(after) or before[adapter] == after[adapter]
            or any(before[k] != after[k] for k in before if k != adapter)):
        raise ValueError('baseline_label_implementation')
    validate_label_provenance(manifest_ref, authority, original['closure_manifest'])
    return parent/'evaluation_claim.label-provenance-v1.json', {'label_provenance_successor': proof}


def validate_label_provenance(manifest_ref, authority, original_ref):
    """Authenticate the separate verification; never self-certify repaired labels."""
    provenance = authority['label_provenance_successor']['label_provenance']
    ref_keys = {'original_closure_manifest', 'revalidation_authority', 'revalidation_claim',
                'revalidation_status', 'proposed_manifest', 'proof_bindings',
                'verification_authority', 'verification_claim', 'verification_status',
                'verification_source', 'verification_helper', 'independent_review'}
    if (set(provenance) != ref_keys | {'schema_version'}
            or provenance['schema_version'] != 'baseline_label_provenance_v1'
            or provenance['original_closure_manifest'] != original_ref):
        raise ValueError('baseline_label_provenance_invalid')
    # These are sealed provenance artifacts, not result values. Hash-only for
    # opaque proof/source/helper files; decode just the terminal mapping below.
    for key in ref_keys:
        ref = provenance[key]
        if set(ref) != {'path', 'sha256'} or reference(ref['path']) != ref:
            raise ValueError('baseline_label_provenance_changed')
    original = checked(original_ref); successor = checked(authority['closure_manifest'])
    verification = checked(successor['verification_receipt'])
    proposal = checked(provenance['proposed_manifest'])
    revalidation_status = checked(provenance['revalidation_status'])
    verification_status = checked(provenance['verification_status'])
    records_hash = hashlib.sha256(canonical(successor['records'])).hexdigest()
    if (original['schema_version'] != 'sealed_baseline_closure_manifest_v1'
            or original['status'] != 'SEALED_INDEPENDENTLY_VERIFIED'
            or successor['schema_version'] != 'sealed_baseline_closure_manifest_v1'
            or successor['status'] != 'SEALED_INDEPENDENTLY_VERIFIED'
            or original['membership'] != manifest_ref or successor['membership'] != manifest_ref
            or verification['schema_version'] != 'baseline_closure_verification_v1'
            or verification['independent_identity_verification'] is not True
            or verification['membership'] != manifest_ref
            or verification['result_cutoff'] != authority['result_cutoff']
            or verification['records_sha256'] != records_hash
            or verification.get('label_provenance') != provenance
            or revalidation_status['status'] != 'RETAINED_IDENTITY_REVALIDATION_COMPLETE'
            or verification_status['status'] != 'RETAINED_IDENTITY_VERIFICATION_COMPLETE'
            or verification_status['membership'] != manifest_ref
            or verification_status['result_cutoff'] != authority['result_cutoff']
            or verification_status['records_sha256'] != records_hash
            or proposal['status'] != 'REQUIRES_INDEPENDENT_VERIFICATION'
            or proposal['evaluation_authority'] is not False
            or len(original['records']) != len(successor['records'])):
        raise ValueError('baseline_label_verification_invalid')
    repairs = {r['race_id']: r for r in proposal['records']}
    originals = [r['race_id'] for r in original['records'] if r['state'] == 'CLOSED']
    if (len(repairs) != len(proposal['records']) or set(repairs) != set(originals)
            or proposal['original_denominator'] != len(original['records'])
            or proposal['revalidated_scope'] != len(originals)):
        raise ValueError('baseline_label_repair_denominator')
    for old, new in zip(original['records'], successor['records']):
        if old['race_id'] != new['race_id']:
            raise ValueError('baseline_label_member_order')
        if old['state'] != 'CLOSED':
            if new != old: raise ValueError('baseline_label_unchanged_exclusion')
            continue
        repair = repairs[old['race_id']]
        if repair['status'] == 'IDENTITY_REVALIDATED':
            expected = {**old, 'evidence': proposal['database'], 'bytes': proposal['database_bytes']}
        else:
            expected = {'race_id': old['race_id'], 'state': 'QUARANTINED',
                        'prior_state_preserved': 'CLOSED', 'reason': repair['status']}
        if new != expected:
            raise ValueError('baseline_label_repair_mapping')


def evaluation_claim(manifest_ref, authority, now):
    """Original, pre-metric repair, or one explicit independently repaired label version."""
    parent = root_path(str(Path(manifest_ref['path']).parent))
    original = parent/'evaluation_claim.json'
    if 'label_provenance_successor' in authority:
        return label_provenance_claim(manifest_ref, authority, now)
    if 'corrected_attempt' not in authority:
        return original, {}
    correction = authority['corrected_attempt']
    keys = {'schema_version', 'predecessor_claim', 'predecessor_authority',
            'predecessor_status', 'failure_diagnosis'}
    if (not isinstance(correction, dict) or set(correction) != keys
            or correction['schema_version'] != 'baseline_corrected_attempt_v1'
            or correction['predecessor_claim']['path'] != str(original)):
        raise ValueError('baseline_correction_scope')
    claim = checked(correction['predecessor_claim'])
    prior = checked(correction['predecessor_authority'])
    terminal = checked(correction['predecessor_status'])
    diagnosis = checked(correction['failure_diagnosis'])
    if (set(claim) != {'claimed_at', 'membership', 'authority', 'output_root'}
            or claim['membership'] != manifest_ref
            or claim['authority'] != correction['predecessor_authority']
            or claim['output_root'] != prior['output_root']
            or 'corrected_attempt' in prior
            or correction['predecessor_status']['path'] != str(Path(prior['output_root'])/'status.json')
            or terminal != {'status': 'FAILED_PRESERVED_CLAIM', 'membership': manifest_ref}):
        raise ValueError('baseline_correction_predecessor')
    same = ('schema_version', 'status', 'performance_evaluation', 'membership',
            'policy', 'closure_manifest', 'result_cutoff', 'provider_requests',
            'result_requests', 'training', 'promotion', 'human_outcome_access',
            'public_performance_outputs', 'limits')
    if (any(prior[key] != authority[key] for key in same)
            or prior['evaluation_id'] == authority['evaluation_id']
            or not stamp(prior['issued_at']) <= stamp(claim['claimed_at']) < stamp(prior['expires_at'])
            or not stamp(claim['claimed_at']) < stamp(authority['issued_at']) <= now):
        raise ValueError('baseline_correction_scope')
    adapter = 'race_collection/retained_baseline_evaluation.py'
    old_pins, new_pins = prior['implementation_files'], authority['implementation_files']
    if (set(old_pins) != set(new_pins) or old_pins.get(adapter) == new_pins[adapter]
            or any(old_pins[key] != value for key, value in new_pins.items() if key != adapter)):
        raise ValueError('baseline_correction_implementation')
    old_output, new_output = root_path(prior['output_root']), root_path(authority['output_root'])
    if (old_output == new_output or old_output.is_relative_to(new_output)
            or new_output.is_relative_to(old_output)
            or os.path.lexists(old_output/'private_metrics.json')):
        raise ValueError('baseline_correction_output_or_metric_exists')
    if (diagnosis.get('schema_version') != 'baseline_failed_execution_diagnosis_v1'
            or diagnosis.get('status') != 'AUTHENTICATED_IMPLEMENTATION_FAILURE_BEFORE_RESULT_OR_METRIC_READS'
            or diagnosis.get('failure_code') != 'baseline_request_field_changed'
            or type(diagnosis.get('result_reads')) is not int or diagnosis['result_reads'] != 0
            or diagnosis.get('metric_calculation') is not False
            or diagnosis.get('metric_artifact_exists') is not False
            or diagnosis.get('original_claim') != correction['predecessor_claim']
            or diagnosis.get('original_authority') != correction['predecessor_authority']
            or diagnosis.get('original_terminal_status') != correction['predecessor_status']
            or diagnosis.get('private_output_root') != prior['output_root']
            or any(diagnosis.get(key) != authority[key] for key in ('membership', 'closure_manifest', 'policy', 'result_cutoff'))
            or not stamp(claim['claimed_at']) <= stamp(diagnosis['created_at']) <= stamp(authority['issued_at'])):
        raise ValueError('baseline_correction_diagnosis')
    return parent/'evaluation_claim.corrected-01.json', {'corrected_attempt': correction}


def join_forecasts(member, admission, inputs, records, manifest):
    """Verify the original four stored records; never replay models or features."""
    if set(records) != set(MODELS): raise ValueError('baseline_four_models_required')
    from src.predictor.on_demand import sealed_runner_set_sha256
    field = comparison_roster(inputs['runners'])
    if sealed_runner_set_sha256(admission['race'], field) != admission['runner_set_sha256']:
        raise ValueError('baseline_native_field_hash')
    expected = [(r['box_number'], r['identity'], r['display_name']) for r in field]
    if len(set(r[0] for r in expected)) != len(expected):
        raise ValueError('baseline_field_identity')
    probabilities = {}
    for name in MODELS:
        record = records[name]
        if (record['candidate'] != name or record['status'] != 'SEALED' or record['failure'] is not None
                or record['admission_sha256'] != member['admission']['sha256']
                or any(record[k] != admission[k] for k in ('race', 'runner_set_sha256', 'plan_sha256',
                    'prediction_id', 'retained_input_manifest_sha256'))
                or not stamp(admission['admitted_at']) <= stamp(record['completed_at']) < stamp(admission['decision_at'])
                or [(r['box_number'], r['identity'], r['dog_name']) for r in record['predictions']] != expected):
            raise ValueError('baseline_forecast_identity_or_timing')
        identity = record['input_identity']
        if any(identity[k] != inputs[k] for k in ('form_sha256', 'sidecar_sha256',
                'odds_receipt_sha256', 'capture_sha256', 'production_feature_rows_sha256', 'captured_at')):
            raise ValueError('baseline_input_identity')
        model_path = 'model/model.json' if name == 'production' else f'comparison/artifacts/{name}.json'
        if name != 'market' and record['model_sha256'] != manifest['files'][model_path]['sha256']:
            raise ValueError('baseline_model_identity')
        probabilities[name] = [r['probability'] for r in record['predictions']]
    # Validate numerical simplex now, before the result reader; this makes no label claim.
    summarize([{'probabilities': probabilities, 'outcome': [1.0]+[0.0]*(len(field)-1)}])
    return probabilities, [{'box_number': r['box_number'], 'dog_name': r['display_name'],
        'source_native_runner_id': r.get('source_native_runner_id')} for r in field]


def read_member(member, closure, protocol, cutoff, *, check_deadline=lambda: None):
    """Protected seam: called only after authority/claim, for one fixed member."""
    from race_collection.retained_study_observer import metadata_candidate, opaque_hash
    from src.predictor.comparison_results import ComparisonResultSource
    check_deadline()
    plan = checked(member['original_plan'])
    candidate = metadata_candidate(member['original_plan'], plan, Path(member['admission']['path']), protocol)
    if any(member.get(k) != v for k, v in candidate.items()):
        raise ValueError('baseline_original_member_changed')
    bundle_path = Path(member['bundle_manifest']['path']).parent
    manifest = checked(member['bundle_manifest']); admission = checked(member['admission'])
    def content(name):
        check_deadline()
        return checked({'path': str(bundle_path/name), 'sha256': manifest['files'][name]['sha256']})
    inputs = content('comparison/inputs.json')
    request = content('request.json')
    from src.predictor.on_demand import canonical_runner_set
    if canonical_runner_set(request['runners'], 'baseline.request.runners') != comparison_roster(inputs['runners']):
        raise ValueError('baseline_request_field_changed')
    records = {}
    for model in MODELS:
        check_deadline()
        records[model] = checked(member['original_forecasts'][model])
    probabilities, field = join_forecasts(member, admission, inputs, records, manifest)
    if closure['state'] in {'QUARANTINED', 'PENDING', 'UNRESOLVED'}:
        return None, closure['state']
    result = content('result.json')
    if (result['race'] != admission['race'] or result['prediction_id'] != admission['prediction_id']
            or stamp(result['generated_at']) >= stamp(admission['decision_at'])):
        raise ValueError('baseline_production_binding')
    bundle = SimpleNamespace(result=result, manifest=manifest, directory=bundle_path.name)
    job = SimpleNamespace(job_id=member['job_id'], input=SimpleNamespace(
        race_id=member['race_id'], jump_timestamp=member['jump_at'], ordered_runners=[
            {'box': r['box_number'], 'name': r['dog_name'], 'source_native_runner_id': r['source_native_runner_id']} for r in field]))
    ref = closure['evidence']
    if closure['state'] == 'CLOSED':
        database = Path(ref['path'])
        if any(Path(str(database)+suffix).exists() for suffix in ('-wal', '-shm', '-journal')):
            raise ValueError('baseline_result_snapshot_busy')
        if opaque_hash(database, closure['bytes']) != ref['sha256']:
            raise ValueError('baseline_result_snapshot_changed')
        check_deadline()
        value = ComparisonResultSource(database).read(job, bundle, now=cutoff)
        check_deadline()
        if opaque_hash(database, closure['bytes']) != ref['sha256']:
            raise ValueError('baseline_result_snapshot_changed')
        if value['state'] != 'RESULT_AVAILABLE': return None, 'RESULT_IDENTITY_OR_COMPLETENESS_UNRESOLVED'
        evidence = {'race_id': member['race_id'], 'identity_verified': True,
            'runner_results': value['evidence']['runner_rows']}
    elif closure['state'] == 'CLOSED_NON_FINISH':
        check_deadline()
        evidence = checked(ref)
        unsigned = {k: v for k, v in evidence.items() if k != 'evidence_sha256'}
        if (evidence.get('schema_version') != 'comparison_known_nonfinish_result_v1'
                or evidence.get('state') != 'RESULT_KNOWN_NON_FINISH'
                or evidence.get('job_id') != member['job_id']
                or evidence.get('result_known') is not True
                or evidence.get('source') != 'thedogs_official'
                or evidence.get('full_order_eligible') is not False
                or evidence.get('source_url') not in {result['race']['url'], result['race']['url']+'?trial=false'}
                or not max(stamp(member['jump_at']), stamp(result['generated_at'])) < stamp(evidence['captured_at']) <= cutoff
                or hashlib.sha256(canonical(unsigned)).hexdigest() != evidence['evidence_sha256']):
            raise ValueError('baseline_nonfinish_receipt_invalid')
        source = evidence['source_evidence']
        if set(source) != {'body', 'request', 'response'}:
            raise ValueError('baseline_nonfinish_source_invalid')
        # The separately pinned closure receipt attests its independently verified
        # parser result. Recheck exact retained HTTP evidence, without a new fetch.
        for source_ref in source.values():
            check_deadline()
            if reference(source_ref['path']) != source_ref:
                raise ValueError('baseline_nonfinish_source_changed')
        request, response = checked(source['request']), checked(source['response'])
        if (request['url'] != evidence['source_url'] or response['final_url'] != evidence['source_url']
                or type(response['status']) is not int or response['status'] != 200
                or response['host'] != 'www.thedogs.com.au'
                or not response['content_type'].lower().startswith('text/html')
                or set(response['retry_headers']) - {'date'}
                or response['bytes'] != Path(source['body']['path']).stat().st_size
                or response['sha256'] != source['body']['sha256']
                or stamp(request['at']) != stamp(evidence['captured_at'])
                or stamp(response['observed_at']) != stamp(evidence['captured_at'])):
            raise ValueError('baseline_nonfinish_source_invalid')
    else:
        raise ValueError('baseline_closure_state_invalid')
    check_deadline()
    try:
        target, category = win_target(member['race_id'], field, evidence)
    except (ValueError, KeyError, TypeError):
        return None, 'WIN_LABEL_IDENTITY_OR_TERMINAL_UNRESOLVED'
    return {'probabilities': probabilities, 'outcome': target}, category


def run_baseline(manifest_ref, authority_ref, *, execute=False, now=None):
    if not execute:
        return {'status': 'DEFAULT_OFF', 'provider_requests': 0, 'result_reads': 0}
    import time
    from race_collection.retained_study_observer import BUDGET
    began = time.monotonic()
    now = now or datetime.now(timezone.utc)
    authority, manifest = authorize(manifest_ref, authority_ref, now)
    output = root_path(authority['output_root'])
    def check_deadline():
        elapsed = time.monotonic()-began
        if elapsed > authority['limits']['max_wall_seconds'] or now.timestamp()+elapsed >= stamp(authority['expires_at']).timestamp():
            raise TimeoutError('baseline_deadline')
    check_deadline()
    # Reissue/output changes never reset a claim. The single repair is explicit.
    claim, linkage = evaluation_claim(manifest_ref, authority, now)
    check_deadline()
    _write(claim, {'claimed_at': now.isoformat(), 'membership': manifest_ref,
        'authority': authority_ref, 'output_root': str(output), **linkage})
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    token = BUDGET.set([authority['limits']['max_files'], authority['limits']['max_bytes']])
    try:
        protocol = checked(manifest['protocol'])
        check_deadline()
        closure = checked(authority['closure_manifest'])
        if (closure['schema_version'] != 'sealed_baseline_closure_manifest_v1'
                or closure['membership'] != manifest_ref or closure['status'] != 'SEALED_INDEPENDENTLY_VERIFIED'
                or not closure.get('verification_receipt')
                or {r['race_id'] for r in closure['records']} != {m['race_id'] for m in manifest['members']}
                or len(closure['records']) != len(manifest['members'])):
            raise ValueError('baseline_closure_membership')
        verification = checked(closure['verification_receipt'])
        if (verification.get('schema_version') != 'baseline_closure_verification_v1'
                or verification.get('independent_identity_verification') is not True
                or verification.get('membership') != manifest_ref
                or verification.get('result_cutoff') != authority['result_cutoff']
                or verification.get('records_sha256') != hashlib.sha256(canonical(closure['records'])).hexdigest()):
            raise ValueError('baseline_closure_verification_invalid')
        closures = {r['race_id']: r for r in closure['records']}
        counts = Counter(); races = []
        for member in manifest['members']:
            check_deadline()
            race, category = read_member(member, closures[member['race_id']], protocol,
                stamp(authority['result_cutoff']), check_deadline=check_deadline)
            counts[category] += 1
            if race is not None: races.append(race)
        check_deadline()
        metrics = summarize(races)
        check_deadline()
        _write(output/'private_metrics.json', {'status': 'RETROSPECTIVE_DESCRIPTIVE_ONLY',
            'membership': manifest_ref, 'authority': authority_ref, 'eligible_races': len(races),
            'metrics': metrics, 'promotion': False})
        summary = {'status': 'PRIVATE_BASELINE_COMPLETE', 'denominator': len(manifest['members']),
            'eligible_races': len(races), 'excluded_races': len(manifest['members'])-len(races),
            'categories': dict(counts), 'private_output': reference(output/'private_metrics.json'),
            'membership': manifest_ref, 'provider_requests': 0, 'result_requests': 0}
        _write(output/'status.json', summary)
        return summary
    except Exception:
        _write(output/'status.json', {'status': 'FAILED_PRESERVED_CLAIM', 'membership': manifest_ref})
        raise
    finally:
        BUDGET.reset(token)
