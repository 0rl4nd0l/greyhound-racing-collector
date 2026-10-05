"""One private descriptive comparison of independently qualified retained pairs.

No acquisition, scoring, fitting, original writes, or prospective eligibility.
The existing label reader and target policy remain unchanged.
"""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from race_collection import retained_baseline_evaluation as baseline
from race_collection import retained_study_observer as observer
from src.predictor.future_comparison import stamp

POLICY = 'RETROSPECTIVE_PRODUCTION_ALPHA_1_VS_HALF_PAIRED_DESCRIPTIVE_V1'
MEMBERSHIP_SHA = '782d6af9bcbe1dd84d1c2674de44f93bc6720cd77b3b7e0195bdcccc03dd4342'
MODEL_SHA = '624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d'
MODEL_MANIFEST_SHA = '8537cbc3d843d106a1fe48793ef01197454ef092c0244025fd65685636a42080'
DERIVATION_SOURCE = 'e716db17a99dd996a1f9986df2d7bfc1ea755027'
ARMS = ('production_parent_alpha_1_0', 'production_parent_alpha_0_5')
DENOMINATOR = 82
CATEGORIES = {'FULL_ORDER_WIN_ELIGIBLE': 73, 'KNOWN_NONFINISH_WIN_ELIGIBLE': 1, 'QUARANTINED': 8}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(observer.canonical(value)).hexdigest()


def implementation_pins():
    root = Path(__file__).resolve().parents[1]
    return {**baseline.implementation_pins(), **{name: observer.reference(root/name)['sha256'] for name in (
        'race_collection/retained_paired_evaluation.py', 'scripts/evaluate_retained_pairs.py')}}


def verify_runtime(authority):
    root = Path(__file__).resolve().parents[1]
    require(authority['source']['root'] == str(root)
        and subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == authority['source']['commit']
        and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip(), 'paired_source_not_frozen')
    python = Path(sys.executable).resolve()
    require(authority['python']['path'] == str(python)
        and hashlib.sha256(python.read_bytes()).hexdigest() == authority['python']['sha256'], 'paired_runtime_changed')


def simplex(values):
    require(isinstance(values, list) and values and all(type(v) in (int, float)
        and math.isfinite(v) and 0 < v < 1 for v in values)
        and abs(math.fsum(values)-1) <= 1e-12, 'paired_probability_invalid')


def align_pair(pair, member, inputs, production):
    """Authenticate the derived two arms and map native-ID order to original field order."""
    unsigned = {k: v for k, v in pair.items() if k != 'derivation_sha256'}
    common = pair['common_inputs']
    roster = baseline.comparison_roster(inputs['runners'])
    ids = [r['source_native_runner_id'] for r in roster]
    require(all(isinstance(i, str) and i for i in ids) and len(ids) == len(set(ids)), 'paired_native_ids')
    require(pair['schema_version'] == 'retrospective_native_v2_controlled_pair_v1'
        and pair['status'] == 'RETROSPECTIVE_COMPUTATION_NOT_ORIGINAL_PREJUMP_OR_SCIENTIFIC'
        and pair['race_id'] == member['race_id'] and pair['derivation_sha256'] == digest(unsigned)
        and pair['serialization_contract'] == 'EXACT_NATIVE_V2_CANONICAL_FULL_ROWS_NO_ROUNDING_OR_TOLERANCE'
        and pair['parent_model_sha256'] == MODEL_SHA and pair['parent_manifest_sha256'] == MODEL_MANIFEST_SHA
        and pair['original_score_timestamp'] is None
        and pair['original_published_complete_at'] == member['original_published_complete_at']
        and all(pair[k] is False for k in ('training', 'performance_evaluation', 'outcomes_present',
            'original_forecasts_modified', 'scientific_membership_created')), 'paired_artifact_invalid')
    require(common['membership_sha256'] == MEMBERSHIP_SHA
        and common['bundle_manifest'] == member['bundle_manifest']
        and common['retained_input_manifest_sha256'] == member['retained_input_manifest_sha256']
        and common['native_roster_sha256'] == digest(roster)
        and pair['common_input_sha256'] == digest(common)
        and pair['native_runner_ids'] == sorted(ids), 'paired_common_input_invalid')
    hashes = common['input_hashes']
    for old, new in (('form_sha256', 'form_csv_sha256'), ('sidecar_sha256', 'sidecar_sha256'),
            ('capture_sha256', 'capture_artifact_sha256'), ('production_feature_rows_sha256', 'feature_rows_sha256')):
        require(inputs[old] == hashes[new], 'paired_input_hash_invalid')
    timing = pair['capture_timing']
    jump = stamp(member['jump_at']); append = stamp(timing['append_at']); fetch = stamp(timing['fetch_at'])
    anchor = pair['historical_validation_anchor']
    require(anchor == {'at': production['completed_at'], 'role': 'RECORDED_PRODUCTION_COMPLETION_NOT_ORIGINAL_SCORE_TIME'}
        and fetch <= append == stamp(inputs['captured_at']) <= stamp(anchor['at']) < jump
        and 120 <= (jump-append).total_seconds() <= 600
        and timing['freshness_basis'] == 'NATIVE_RECEIPT_APPEND_TIME'
        and timing['append_lead_seconds'] == (jump-append).total_seconds()
        and timing['fetch_lead_seconds'] == (jump-fetch).total_seconds()
        and stamp(pair['derived_at']) >= stamp(member['original_published_complete_at']), 'paired_historical_timing')
    candidates = pair['candidates']
    require(len(candidates) == 2 and [c['candidate_id'] for c in candidates] == list(ARMS)
        and [c['strength'] for c in candidates] == [1.0, 0.5]
        and all(c['common_input_sha256'] == pair['common_input_sha256'] for c in candidates), 'paired_arms_invalid')
    order = {native: i for i, native in enumerate(pair['native_runner_ids'])}
    arrays = []
    for c in candidates:
        simplex(c['probabilities'])
        require(len(c['probabilities']) == len(ids), 'paired_field_size')
        arrays.append([c['probabilities'][order[native]] for native in ids])
    # Equality is exact canonical serialization, not tolerance or rounded equality.
    require(observer.canonical(arrays[0]) == observer.canonical([r['probability'] for r in production['predictions']]),
        'paired_full_arm_changed')
    return dict(zip(ARMS, arrays))


def summarize_pairs(races):
    require(bool(races), 'paired_no_eligible_races')
    losses = {arm: {'log_loss': [], 'brier_sum': []} for arm in ARMS}
    for race in races:
        y = race['outcome']
        require(isinstance(y, list) and y and all(type(v) in (int, float) and math.isfinite(v)
            and v >= 0 for v in y) and abs(math.fsum(y)-1) <= 1e-12, 'paired_target_invalid')
        for arm in ARMS:
            p = race['probabilities'][arm]; simplex(p)
            require(len(p) == len(y), 'paired_target_size')
            losses[arm]['log_loss'].append(-math.fsum(a*math.log(b) for a, b in zip(y, p)))
            losses[arm]['brier_sum'].append(math.fsum((a-b)**2 for a, b in zip(y, p)))
    count = len(races)
    return {'arm_means': {arm: {k: math.fsum(v)/count for k, v in values.items()} for arm, values in losses.items()},
        'paired_mean_half_minus_full': {key: math.fsum(h-f for f, h in zip(losses[ARMS[0]][key],
            losses[ARMS[1]][key]))/count for key in ('log_loss', 'brier_sum')}}


class ReadGuard:
    """Finite allowlisted artifact IO; SQLite's internal page reads are separate."""
    def __init__(self, authority, scope, *, started=None):
        self.authority = authority
        self.allowed = {Path(p): value for p, value in scope['allowed_files'].items()}
        self.started = time.monotonic() if started is None else started
        self.operations = self.bytes = self.label_members = 0

    def check(self):
        require(datetime.now(timezone.utc) < stamp(self.authority['expires_at'])
            and time.monotonic()-self.started < self.authority['limits']['max_wall_seconds'], 'paired_deadline')

    def charge(self, path, size):
        self.check(); path = Path(path)
        require(path in self.allowed and path.is_absolute() and path.resolve() == path, 'paired_unbound_input')
        require(size == self.allowed[path]['bytes'], 'paired_input_size')
        self.operations += 1; self.bytes += size
        require(self.operations <= self.authority['limits']['max_files']
            and self.bytes <= self.authority['limits']['max_bytes'], 'paired_read_limit')
        return self.allowed[path].get('sha256')

    @contextmanager
    def installed(self):
        # Process-local, single invocation. Always restore even on a failed read.
        old_raw, old_hash, old_baseline_raw = observer.raw, observer.opaque_hash, baseline.raw
        def raw(path, maximum=2*1024*1024):
            expected = self.charge(path, Path(path).stat().st_size)
            value = old_raw(path, maximum)
            require(expected is None or hashlib.sha256(value).hexdigest() == expected, 'paired_input_changed')
            self.check(); return value
        def opaque(path, expected_bytes):
            expected = self.charge(path, expected_bytes)
            value = old_hash(path, expected_bytes)
            require(expected is None or value == expected, 'paired_input_changed')
            self.check(); return value
        observer.raw = baseline.raw = raw; observer.opaque_hash = opaque
        try:
            yield
        finally:
            observer.raw, observer.opaque_hash, baseline.raw = old_raw, old_hash, old_baseline_raw


def validate_controls(authority, scope):
    """Metadata only; no scoring or label decoding."""
    require(authority['implementation_files'] == implementation_pins(), 'paired_implementation_changed')
    require(scope['schema_version'] == 'retained_paired_evaluation_inputs_v1'
        and scope['policy'] == POLICY and scope['membership']['sha256'] == MEMBERSHIP_SHA
        and scope['expected_categories'] == CATEGORIES, 'paired_scope_invalid')
    membership = baseline.load_membership(scope['membership'])
    require(len(membership['members']) == DENOMINATOR, 'paired_denominator')
    prior = observer.checked(scope['baseline_authority'])
    terminal = observer.checked(scope['baseline_status'])
    require(prior['membership'] == scope['membership'] and prior['closure_manifest'] == scope['closure_manifest']
        and prior['result_cutoff'] == scope['result_cutoff']
        and terminal['membership'] == scope['membership'] and terminal['status'] == 'PRIVATE_BASELINE_COMPLETE'
        and terminal['denominator'] == DENOMINATOR and terminal['eligible_races'] == 74
        and terminal['excluded_races'] == 8 and terminal['categories'] == CATEGORIES
        and scope['baseline_status']['path'] == str(Path(prior['output_root'])/'status.json'), 'paired_baseline_binding')
    # Validate sealed label provenance directly, without consuming or reopening an old evaluation claim.
    baseline.validate_label_provenance(scope['membership'], prior,
        prior['label_provenance_successor']['label_provenance']['original_closure_manifest'])
    qualification = observer.checked(scope['pair_qualification'])
    inventory = observer.checked(scope['pair_inventory'])
    pair_terminal = observer.checked(qualification['terminal'])
    require(qualification['schema_version'] == 'native_v2_independent_execution_qualification_v1'
        and qualification['status'] == 'QUALIFIED_RETROSPECTIVE_CONTROLLED_PAIRS'
        and qualification['source_commit'] == DERIVATION_SOURCE
        and qualification['inventory'] == scope['pair_inventory']
        and qualification['denominator'] == qualification['verified_pairs'] == qualification['exact_full_arm_replay_gate_passed'] == DENOMINATOR
        and qualification['source_clean'] is True
        and qualification['provider_requests'] == 0 and qualification['performance_evaluation'] is False
        and qualification['exclusions'] == qualification['unattempted'] == qualification['result_reads'] == 0
        and inventory['schema_version'] == 'retrospective_native_v2_derivation_inventory_v1'
        and inventory['status'] == 'DISPOSITIONS_RECORDED'
        and inventory['denominator'] == DENOMINATOR and inventory['membership_sha256'] == MEMBERSHIP_SHA
        and inventory['source_commit'] == DERIVATION_SOURCE
        and pair_terminal == {'status': 'RETROSPECTIVE_NATIVE_V2_DERIVATION_COMPLETE', 'denominator': DENOMINATOR,
            'categories': {'EXACT_FULL_ARM_VERIFIED_PAIR_DERIVED': DENOMINATOR}, 'inventory_sha256': scope['pair_inventory']['sha256']},
        'paired_derivation_qualification')
    records = inventory['records']
    require(len(records) == DENOMINATOR and len({r['race_id'] for r in records}) == DENOMINATOR
        and {r['race_id'] for r in records} == {m['race_id'] for m in membership['members']}
        and all(r['status'] == 'EXACT_FULL_ARM_VERIFIED_PAIR_DERIVED' for r in records), 'paired_inventory_members')
    closure = observer.checked(scope['closure_manifest'])
    require(closure['membership'] == scope['membership'] and len(closure['records']) == DENOMINATOR
        and len({r['race_id'] for r in closure['records']}) == DENOMINATOR
        and {r['race_id'] for r in closure['records']} == {m['race_id'] for m in membership['members']}, 'paired_closure_members')
    return membership, records, closure


def authorize(authority_ref):
    authority = observer.checked(authority_ref)
    now = datetime.now(timezone.utc)
    require(authority['schema_version'] == 'private_retained_pair_evaluation_authority_v1'
        and authority['status'] == 'AUTHORIZED_ONE_SHOT_PRIVATE_PAIRED_EVALUATION'
        and authority.get('evaluation_id') and authority.get('authority_reference')
        and authority['policy'] == POLICY and authority['performance_evaluation'] is True
        and all(authority[k] == 0 for k in ('provider_requests', 'result_requests'))
        and all(authority[k] is False for k in ('training', 'promotion', 'human_outcome_access', 'public_performance_outputs'))
        and stamp(authority['issued_at']) <= now < stamp(authority['expires_at']), 'paired_authority_invalid')
    verify_runtime(authority)
    require(authority['implementation_files'] == implementation_pins(), 'paired_implementation_changed')
    scope = observer.checked(authority['input_scope'])
    require(authority['limits'] == scope['limits'] and scope['protocol'] == authority['protocol']
        and scope['claim_path'] == authority['claim_path'], 'paired_authority_scope')
    limits = scope['limits']
    require(set(limits) == {'max_files', 'max_bytes', 'max_wall_seconds', 'max_label_members', 'max_output_bytes'}
        and all(type(v) is int and v > 0 for v in limits.values()) and limits['max_label_members'] == 74,
        'paired_limits_invalid')
    protocol = authority['protocol']
    require(hashlib.sha256(observer.raw(protocol['path'])).hexdigest() == protocol['sha256'], 'paired_protocol_changed')
    output = observer.root_path(authority['output_root'])
    claim = observer.root_path(authority['claim_path'])
    require(claim.name == 'paired_evaluation_claim.json' and claim.parent == Path(authority['input_scope']['path']).parent
        and output != claim.parent and not claim.parent.is_relative_to(output)
        and not output.exists(), 'paired_output_invalid')
    protected = [Path(__file__).resolve().parents[1], *map(Path, scope['allowed_files']),
        *[observer.root_path(value) for value in scope['protected_roots']]]
    require(all(not (out == p or out.is_relative_to(p) or p.is_relative_to(out))
        for out in (output, claim) for p in protected), 'paired_output_overlap')
    return authority, scope, output, claim


def run_paired_evaluation(authority_ref=None, *, execute=False):
    if not execute:
        return {'status': 'DEFAULT_OFF', 'provider_requests': 0, 'result_reads': 0}
    started = time.monotonic()
    authority, scope, output, claim = authorize(authority_ref)
    guard = ReadGuard(authority, scope, started=started)
    guard.check()
    baseline._write(claim, {'schema_version': 'retained_pair_evaluation_claim_v1',
        'claimed_at': datetime.now(timezone.utc).isoformat(), 'authority': authority_ref,
        'membership': scope['membership'], 'input_scope': authority['input_scope'], 'output_root': str(output)})
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    try:
        with guard.installed():
            membership, records, closure = validate_controls(authority, scope)
            protocol = observer.checked(membership['protocol'])
            pairs = {r['race_id']: r['artifact'] for r in records}
            closures = {r['race_id']: r for r in closure['records']}
            counts = Counter(); eligible = []
            for member in membership['members']:
                guard.check()
                pair = observer.checked(pairs[member['race_id']])
                manifest = observer.checked(member['bundle_manifest'])
                base = Path(member['bundle_manifest']['path']).parent
                inputs = observer.checked({'path': str(base/'comparison/inputs.json'), 'sha256': manifest['files']['comparison/inputs.json']['sha256']})
                production = observer.checked(member['original_forecasts']['production'])
                arms = align_pair(pair, member, inputs, production)
                if closures[member['race_id']]['state'] == 'CLOSED' or closures[member['race_id']]['state'] == 'CLOSED_NON_FINISH':
                    guard.label_members += 1
                    require(guard.label_members <= authority['limits']['max_label_members'], 'paired_label_limit')
                race, category = baseline.read_member(member, closures[member['race_id']], protocol,
                    stamp(scope['result_cutoff']), check_deadline=guard.check)
                counts[category] += 1
                if race is not None:
                    require(observer.canonical(arms[ARMS[0]]) == observer.canonical(race['probabilities']['production']),
                        'paired_validated_full_arm_changed')
                    eligible.append({'probabilities': arms, 'outcome': race['outcome']})
            require(dict(counts) == CATEGORIES and len(eligible) == 74, 'paired_eligibility_changed')
            metrics = summarize_pairs(eligible)
            guard.check()
            payload = {'status': 'RETROSPECTIVE_DESCRIPTIVE_ONLY', 'policy': POLICY,
                'membership': scope['membership'], 'authority': authority_ref, 'eligible_races': 74,
                'metrics': metrics, 'training': False, 'promotion': False}
            require(len(observer.canonical(payload)) <= authority['limits']['max_output_bytes']-8192, 'paired_output_limit')
            baseline._write(output/'private_metrics.json', payload)
            guard.check()
            # Output hash is not an input read and never enters the protected read allowlist.
            metric_ref = {'path': str(output/'private_metrics.json'), 'sha256': hashlib.sha256(observer.canonical(payload)+b'\n').hexdigest()}
            summary = {'status': 'PRIVATE_PAIRED_EVALUATION_COMPLETE', 'denominator': DENOMINATOR,
                'eligible_races': 74, 'excluded_races': 8, 'categories': dict(counts),
                'membership': scope['membership'], 'pair_inventory': scope['pair_inventory'],
                'private_output': metric_ref, 'read_operations': guard.operations, 'read_bytes': guard.bytes,
                'provider_requests': 0, 'result_requests': 0, 'training': False, 'promotion': False}
            baseline._write(output/'status.json', summary)
            return summary
    except Exception:
        baseline._write(output/'status.json', {'status': 'FAILED_PRESERVED_CLAIM', 'membership': scope['membership']})
        raise
