"""Fabricated only: preserved attempts and a separately versioned label cohort."""
import copy
import hashlib
from datetime import timedelta
from pathlib import Path

import pytest
from race_collection import retained_baseline_evaluation as baseline
from tests.test_retained_baseline_evaluation import NOW, put, authorized_case, corrected_case


@pytest.fixture
def label_case(corrected_case, monkeypatch):
    membership, corrected, root, now = corrected_case
    current_pins = baseline.implementation_pins()
    corrected['implementation_files'] = {**current_pins, 'race_collection/retained_baseline_evaluation.py': '1'*64}
    monkeypatch.setattr(baseline, 'implementation_pins', lambda: corrected['implementation_files'])
    corrected_ref = put(root/'corrected_authority.json', corrected)
    baseline.run_baseline(membership, corrected_ref, execute=True, now=now)
    monkeypatch.setattr(baseline, 'implementation_pins', lambda: current_pins)
    correction = corrected['corrected_attempt']
    predecessor = {
        'original_claim': correction['predecessor_claim'],
        'original_authority': correction['predecessor_authority'],
        'original_status': correction['predecessor_status'],
        'corrected_claim': baseline.reference(root/'proposal/evaluation_claim.corrected-01.json'),
        'corrected_authority': corrected_ref,
        'corrected_status': baseline.reference(root/'corrected_output/status.json'),
        'corrected_metrics': baseline.reference(root/'corrected_output/private_metrics.json')}
    old_closure = baseline.checked(corrected['closure_manifest'])
    records = copy.deepcopy(old_closure['records'])
    records[0] = {'race_id': '0', 'state': 'QUARANTINED', 'prior_state_preserved': 'CLOSED', 'reason': 'IDENTITY_UNRESOLVED'}
    records_hash = hashlib.sha256(baseline.canonical(records)).hexdigest()
    provenance = {'schema_version': 'baseline_label_provenance_v1',
        'original_closure_manifest': corrected['closure_manifest'],
        'proposed_manifest': put(root/'repair-proposal.json', {
            'status': 'REQUIRES_INDEPENDENT_VERIFICATION', 'evaluation_authority': False,
            'original_denominator': 3, 'revalidated_scope': 1,
            'records': [{'race_id': '0', 'job_id': '0', 'status': 'IDENTITY_UNRESOLVED'}]}),
        'revalidation_status': put(root/'repair-status.json', {'status': 'RETAINED_IDENTITY_REVALIDATION_COMPLETE'}),
        'verification_status': put(root/'verification-status.json', {'status': 'RETAINED_IDENTITY_VERIFICATION_COMPLETE',
            'membership': membership, 'result_cutoff': corrected['result_cutoff'], 'records_sha256': records_hash})}
    for key in ('revalidation_authority', 'revalidation_claim', 'proof_bindings',
                'verification_authority', 'verification_claim', 'verification_source',
                'verification_helper', 'independent_review'):
        provenance[key] = put(root/(key+'.json'), {'synthetic': key})
    verified = copy.deepcopy(baseline.checked(old_closure['verification_receipt']))
    verified.update(records_sha256=hashlib.sha256(baseline.canonical(records)).hexdigest(),
                    label_provenance=provenance)
    closure = copy.deepcopy(old_closure)
    closure.update(records=records, verification_receipt=put(root/'verified-labels.json', verified))
    authority = copy.deepcopy(corrected)
    authority.pop('corrected_attempt')
    authority['implementation_files'] = current_pins
    authority.update(evaluation_id='label-provenance-v1', output_root=str(root/'label-output'),
        issued_at=(now+timedelta(seconds=1)).isoformat(),
        closure_manifest=put(root/'label-closure.json', closure),
        label_provenance_successor={'schema_version': 'baseline_label_provenance_successor_v1',
            **predecessor, 'label_provenance': provenance})
    return membership, authority, root, now+timedelta(seconds=2)


def test_separate_claim_preserves_both_attempts_and_metrics(label_case):
    membership, authority, root, now = label_case
    proof = authority['label_provenance_successor']
    before = {k: Path(v['path']).read_bytes() for k, v in proof.items() if k not in {'schema_version', 'label_provenance'}}
    result = baseline.run_baseline(membership, put(root/'label-authority.json', authority), execute=True, now=now)
    assert result['status'] == 'PRIVATE_BASELINE_COMPLETE'
    assert result['denominator'] == 3
    assert (root/'proposal/evaluation_claim.label-provenance-v1.json').exists()
    for k, content in before.items():
        assert Path(proof[k]['path']).read_bytes() == content
    authority['output_root'] = str(root/'retry-output')
    with pytest.raises(FileExistsError):
        baseline.run_baseline(membership, put(root/'retry-authority.json', authority), execute=True, now=now)


def repin(root, proof, key, change):
    value = baseline.checked(proof[key]); change(value)
    proof[key] = put(root/(key+'-changed.json'), value)


@pytest.mark.parametrize('defect', [
    'both_modes', 'claim_path', 'claim_hash', 'status_hash', 'metrics_hash',
    'changed_membership', 'copied_membership', 'changed_policy', 'changed_cutoff',
    'same_code', 'changed_dependency', 'same_evaluation_id', 'expired',
    'changed_corrected_metrics', 'original_claim_symlink', 'changed_output',
    'missing_verification_status', 'failed_verification_status', 'changed_proof',
    'changed_closure', 'changed_original_closure', 'proposal_only',
    'changed_nonfinish', 'changed_quarantine', 'reordered_records', 'drop_member',
    'wrong_repair_denominator', 'unverified_promoted', 'failed_revalidation'])
def test_invalid_label_version_never_claims_or_reads(label_case, monkeypatch, defect):
    membership, authority, root, now = label_case
    proof = authority['label_provenance_successor']; provenance = proof['label_provenance']
    if defect == 'both_modes': authority['corrected_attempt'] = {}
    elif defect == 'claim_path': proof['original_claim']['path'] = str(root/'copy.json')
    elif defect == 'claim_hash': proof['corrected_claim']['sha256'] = 'f'*64
    elif defect == 'status_hash': proof['corrected_status']['sha256'] = 'f'*64
    elif defect == 'metrics_hash': proof['corrected_metrics']['sha256'] = 'f'*64
    elif defect == 'changed_membership': authority['membership'] = {**membership, 'sha256': 'f'*64}
    elif defect == 'copied_membership':
        copy_path = root/'copy/membership.json'; copy_path.parent.mkdir()
        copy_path.write_bytes(Path(membership['path']).read_bytes())
        membership = baseline.reference(copy_path); authority['membership'] = membership
    elif defect == 'changed_policy': authority['policy'] = 'NEW_POLICY'
    elif defect == 'changed_cutoff': authority['result_cutoff'] = '2026-10-04T07:59:00+00:00'
    elif defect in {'same_code', 'changed_dependency'}:
        pins = dict(authority['implementation_files'])
        key = 'race_collection/retained_baseline_evaluation.py' if defect == 'same_code' else 'src/predictor/on_demand.py'
        pins[key] = '1'*64
        authority['implementation_files'] = pins
        monkeypatch.setattr(baseline, 'implementation_pins', lambda: pins)
    elif defect == 'same_evaluation_id': authority['evaluation_id'] = 'corrected'
    elif defect == 'expired': authority['expires_at'] = now.isoformat()
    elif defect == 'changed_corrected_metrics': Path(proof['corrected_metrics']['path']).write_bytes(b'changed')
    elif defect == 'original_claim_symlink':
        path = Path(proof['original_claim']['path']); content = path.read_bytes()
        path.unlink(); other = root/'saved-claim.json';other.write_bytes(content);path.symlink_to(other)
    elif defect == 'changed_output': authority['output_root'] = str(root/'corrected_output/child')
    elif defect == 'missing_verification_status': Path(provenance['verification_status']['path']).unlink()
    elif defect == 'failed_verification_status':
        repin(root, provenance, 'verification_status', lambda v: v.update(status='FAILED'))
    elif defect == 'changed_proof': provenance['proof_bindings']['sha256'] = 'f'*64
    elif defect == 'changed_original_closure': provenance['original_closure_manifest']['sha256'] = 'f'*64
    elif defect == 'proposal_only':
        repin(root, provenance, 'proposed_manifest', lambda v: v.update(status='AUTHORIZED'))
    elif defect == 'wrong_repair_denominator':
        repin(root, provenance, 'proposed_manifest', lambda v: v.update(records=[]))
    elif defect == 'failed_revalidation':
        repin(root, provenance, 'revalidation_status', lambda v: v.update(status='FAILED_PRESERVED_REVALIDATION_CLAIM'))
    else:
        closure = baseline.checked(authority['closure_manifest'])
        if defect == 'changed_closure': closure['status'] = 'PROPOSED'
        elif defect == 'changed_nonfinish': closure['records'][1]['state'] = 'CLOSED'
        elif defect == 'changed_quarantine': closure['records'][2]['state'] = 'CLOSED'
        elif defect == 'reordered_records': closure['records'].reverse()
        elif defect == 'drop_member': closure['records'].pop()
        elif defect == 'unverified_promoted': closure['records'][0]['state'] = 'CLOSED'
        # Re-sign the records/terminal/receipt to exercise mapping, not just a stale hash.
        digest = hashlib.sha256(baseline.canonical(closure['records'])).hexdigest()
        repin(root, provenance, 'verification_status', lambda v: v.update(records_sha256=digest))
        verification = baseline.checked(closure['verification_receipt'])
        verification.update(records_sha256=digest, label_provenance=provenance)
        closure['verification_receipt'] = put(root/'changed-verification.json', verification)
        authority['closure_manifest'] = put(root/'changed-closure.json', closure)
    monkeypatch.setattr(baseline, 'read_member', lambda *a, **k: pytest.fail('protected reader reached'))
    with pytest.raises((ValueError, FileNotFoundError, KeyError)):
        baseline.run_baseline(membership, put(root/'bad-authority.json', authority), execute=True, now=now)
    assert not (root/'proposal/evaluation_claim.label-provenance-v1.json').exists()
    assert not (root/'label-output').exists()


def test_failed_label_execution_consumes_only_new_claim(label_case, monkeypatch):
    membership, authority, root, now = label_case
    def fail(*a, **k): raise ValueError('synthetic private failure')
    monkeypatch.setattr(baseline, 'read_member', fail)
    with pytest.raises(ValueError):
        baseline.run_baseline(membership, put(root/'authority-label.json', authority), execute=True, now=now)
    assert baseline.checked(baseline.reference(root/'label-output/status.json'))['status'] == 'FAILED_PRESERVED_CLAIM'
    assert (root/'proposal/evaluation_claim.json').exists()
    assert (root/'proposal/evaluation_claim.corrected-01.json').exists()
    assert (root/'proposal/evaluation_claim.label-provenance-v1.json').exists()
    assert not (root/'label-output/private_metrics.json').exists()
    authority['output_root'] = str(root/'another')
    with pytest.raises(FileExistsError):
        baseline.run_baseline(membership, put(root/'another-authority.json', authority), execute=True, now=now)


def test_label_default_off_reads_nothing(monkeypatch):
    monkeypatch.setattr(baseline, 'checked', lambda *a: pytest.fail('read'))
    assert baseline.run_baseline(None, None)['status'] == 'DEFAULT_OFF'


def test_label_proof_cannot_extend_deadline(label_case, monkeypatch):
    import time
    membership, authority, root, now = label_case
    clock = [0]
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    original = baseline.label_provenance_claim
    def delayed(*a):
        result = original(*a);clock[0] = authority['limits']['max_wall_seconds']+1
        return result
    monkeypatch.setattr(baseline, 'label_provenance_claim', delayed)
    with pytest.raises(TimeoutError):
        baseline.run_baseline(membership, put(root/'slow-authority.json', authority), execute=True, now=now)
    assert not (root/'proposal/evaluation_claim.label-provenance-v1.json').exists()


def test_verified_label_substitution_is_bound_to_exact_database(label_case):
    membership, authority, root, now = label_case
    provenance = authority['label_provenance_successor']['label_provenance']
    proposal = baseline.checked(provenance['proposed_manifest'])
    db = root/'fabricated-labels.sqlite3'; db.write_bytes(b'fabricated: not opened by this claim test')
    proposal.update(database=baseline.reference(db), database_bytes=db.stat().st_size)
    proposal['records'][0]['status'] = 'IDENTITY_REVALIDATED'
    provenance['proposed_manifest'] = put(root/'repaired-proposal.json', proposal)
    closure = baseline.checked(authority['closure_manifest'])
    old = baseline.checked(provenance['original_closure_manifest'])['records'][0]
    closure['records'][0] = {**old, 'evidence': proposal['database'], 'bytes': proposal['database_bytes']}
    digest = hashlib.sha256(baseline.canonical(closure['records'])).hexdigest()
    repin(root, provenance, 'verification_status', lambda v: v.update(records_sha256=digest))
    receipt = baseline.checked(closure['verification_receipt'])
    receipt.update(records_sha256=digest, label_provenance=provenance)
    closure['verification_receipt'] = put(root/'repaired-verification.json', receipt)
    authority['closure_manifest'] = put(root/'repaired-closure.json', closure)
    claim, _ = baseline.evaluation_claim(membership, authority, now)
    assert claim.name == 'evaluation_claim.label-provenance-v1.json'
    # Re-signed mapping to another database is still forbidden by the repair proof.
    closure['records'][0]['evidence'] = {**proposal['database'], 'sha256': 'f'*64}
    digest = hashlib.sha256(baseline.canonical(closure['records'])).hexdigest()
    repin(root, provenance, 'verification_status', lambda v: v.update(records_sha256=digest))
    receipt.update(records_sha256=digest, label_provenance=provenance)
    closure['verification_receipt'] = put(root/'substituted-verification.json', receipt)
    authority['closure_manifest'] = put(root/'substituted-closure.json', closure)
    with pytest.raises(ValueError, match='baseline_label_repair_mapping'):
        baseline.evaluation_claim(membership, authority, now)


def test_label_version_cannot_chain_from_a_prior_label_version(label_case):
    membership, authority, root, now = label_case
    proof = authority['label_provenance_successor']
    previous = baseline.checked(proof['corrected_authority'])
    previous['label_provenance_successor'] = {'schema_version': 'previous'}
    proof['corrected_authority'] = put(root/'chained-authority.json', previous)
    claim = baseline.checked(proof['corrected_claim']);claim['authority'] = proof['corrected_authority']
    # Same original claim path, fabricated re-signing cannot turn a label attempt into corrected-01.
    proof['corrected_claim'] = put(root/'proposal/evaluation_claim.corrected-01.json', claim)
    with pytest.raises(ValueError, match='baseline_label_predecessor'):
        baseline.evaluation_claim(membership, authority, now)
