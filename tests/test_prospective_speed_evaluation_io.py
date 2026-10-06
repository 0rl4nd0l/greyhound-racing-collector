"""Durable evaluation uses fabricated targets and never opens real results."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

import pytest

from race_collection import prospective_speed_evaluation_io as worker
from race_collection import prospective_speed_plan as plan
from race_collection import prospective_speed_runtime as runtime
from race_collection.sectional_speed_evaluation import adjusted_probabilities


def pin(value):
    return hashlib.sha256(runtime.encoded(value)).hexdigest()


def make_fixture(tmp_path, monkeypatch, *, label_status='FULL_ORDER_WIN_ELIGIBLE',
                 target_change=None, failure=False, missing_date=False):
    def put(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        return runtime.put_new(path, value)

    allocation = put('allocation.json', {'status': 'AUTHORIZED',
        'allocation_id': 'development-single-snapshot-20261003-v1'})
    monkeypatch.setattr(plan, 'ALLOCATION_SHA256', allocation['sha256'])
    refs = {key: {'path': str(tmp_path / (key + '.json')), 'sha256': 'a' * 64}
        for key in ('historical_amendment', 'exclusive_amendment', 'current_user',
                    'result_runtime', 'reservation_registry')}
    refs['allocation'] = allocation
    frozen_plan = plan.build_plan(frozen_at='2026-10-06T19:00:00+11:00',
        candidate_reference={'path': str(tmp_path/'candidate.json'), 'sha256': 'b'*64},
        authority_references=refs, precision=plan.precision_diagnostic([-.01, .01]))
    plan_ref = put('plan.json', frozen_plan)
    race_id = 'Race 1 - FIXTURE - 2026-10-10'
    jump = '2026-10-10T13:10:00+11:00'
    rows = [{'race_id': race_id, 'race_key': '2026-10-10|FIXTURE|1', 'jump_at': jump}]
    protected = put('protected-members.json', {'members': []})
    population = plan.select_population(frozen_plan, rows, local_date='2026-10-10',
        frozen_at='2026-10-10T12:50:00+11:00', source_observed_at='2026-10-10T12:48:00+11:00',
        index_complete=True, protected_membership_reference=protected)
    population_ref = put('population.json', population)
    days = [{'local_date': '2026-10-10', 'status': 'POPULATION_FROZEN',
        'population_sha256': pin(population)}, {'local_date': '2026-10-11',
        'status': 'INDEX_MISSING', 'reason': 'Fabricated absent index', 'population_sha256': None}]
    day_ref = put('date-accounting.json', days[:1] if missing_date else days)
    activation = put('activation.json', {'status': 'AUTHORIZED_PROSPECTIVE_DEVELOPMENT',
        'plan_sha256': plan_ref['sha256'], 'population_sha256': population_ref['sha256'],
        'candidate_commit': runtime.inputs.FROZEN_CANDIDATE_COMMIT,
        'additional_source_requests': 0, 'additional_result_requests': 0,
        'development_precedence_verified': True})
    member = {'race_id': race_id, 'jump_at': jump}
    original = {'original_published_complete_at': '2026-10-10T13:04:00+11:00'}
    model = {'path': str(tmp_path/'frozen-model.json'), 'sha256': runtime.inputs.BASELINE_ARTIFACT_SHA256}
    source_job = {'plan': plan_ref, 'population': population_ref, 'activation': activation,
        'member': member, 'original': original, 'model': model,
        'forecast_at': '2026-10-10T13:05:00+11:00'}
    source_job_ref = put('source-job.json', source_job)
    forecast_root = tmp_path/'forecasts'
    directory = 'forecasts/' + hashlib.sha256(race_id.encode()).hexdigest() + '/'
    claim_ref = put(directory+'claim.json', {'mode': 'PROSPECTIVE', 'race_id': race_id,
        'source_job': source_job_ref, 'claimed_at': '2026-10-10T13:04:59+11:00'})
    execution_ref = put(directory+'job.json', {**source_job, 'execution_mode': 'PROSPECTIVE'})
    baseline, estimates, support = [.6, .4], [.5, 0.], [True, False]
    speed = adjusted_probabilities(baseline, estimates, support, .1)
    ids = ['native:dog1', 'native:dog2']
    payload = {'schema_version': 'prospective_frozen_speed_forecast_v1', 'race_id': race_id,
        'race_date': '2026-10-10', 'jump_at': jump, 'forecast_at': source_job['forecast_at'],
        'information_cutoff': source_job['forecast_at'], 'source_member': member,
        'original_publication': original, 'baseline_artifact': model,
        'contract': runtime.inputs.frozen_contract(),
        'predictions': [{'runner_id': name, 'market': .5, 'baseline': b, 'baseline_plus_speed': s}
            for name, b, s in zip(ids, baseline, speed)],
        'speed_features': {'runners': [{'runner_id': name, 'speed_estimate': value,
            'status': 'SUPPORTED' if ok else 'UNSUPPORTED'} for name, value, ok in zip(ids, estimates, support)]},
        'speed_packet': {'roster': [{'runner_id': name} for name in ids]}}
    payload_ref = put(directory+'forecast.json', payload)
    failure_kind = 'SPEED_PROCESSING_TIMEOUT' if failure is True else failure
    terminal_status = failure_kind if failure_kind not in {False, 'LATE_SPEED_SEAL', 'INTERRUPTED_SEAL'} else 'FORECAST_PAYLOAD_DURABLE'
    terminal_ref = put(directory+'terminal.json', {'race_id': race_id, 'claim': claim_ref,
        'execution_job': execution_ref, 'payload': payload_ref,
        'status': terminal_status,
        'completed_at': '2026-10-10T13:05:01+11:00'})
    seal_ref = put(directory+'seal.json', {'terminal': terminal_ref, 'payload': payload_ref,
        'completed_at': '2026-10-10T13:05:02+11:00'})
    completion_ref = put(directory+'completion.json', {'seal': seal_ref,
        'status': 'LATE_SPEED_SEAL' if failure_kind == 'LATE_SPEED_SEAL' else 'SEALED_PREJUMP',
        'completed_at': jump if failure_kind == 'LATE_SPEED_SEAL' else '2026-10-10T13:05:03+11:00'})
    failure_evidence = None
    if failure_kind == 'INTERRUPTED_SEAL':
        Path(completion_ref['path']).unlink()
    if failure_kind == 'INTERRUPTED_BEFORE_CLAIM':
        for name in ('claim.json', 'job.json', 'forecast.json', 'terminal.json', 'seal.json', 'completion.json'):
            (tmp_path/directory/name).unlink()
        failure_evidence = put(directory+'interrupted-before-claim.json',
            {'status': failure_kind, 'source_job': source_job_ref, 'at': '2026-10-10T13:06:00+11:00'})
    target_ref = proof_ref = closure_ref = None
    if label_status in plan.VERIFIED_LABELS and not failure:
        closure_ref = put('official-closure.private.json', {'race_id': race_id,
            'fabricated_only': True, 'official_winner_id': ids[0]})
        target = {'schema_version': 'prospective_speed_verified_win_target_v1',
            'role': 'SELECTED_DEVELOPMENT_WIN_TARGET', 'race_id': race_id,
            'race_date': '2026-10-10', 'jump_at': jump, 'runner_ids': ids,
            'outcome': [1., 0.], 'label_status': label_status,
            'official_observed_at': '2026-10-10T14:00:00+11:00',
            'closure_evidence': closure_ref, 'plan': plan_ref, 'allocation': allocation}
        target.update(target_change or {})
        target_ref = put('target.private.json', target)
        proof_ref = put('identity-proof.json', {
            'schema_version': 'prospective_speed_result_identity_proof_v1',
            'status': 'VERIFIED_SELECTED_DEVELOPMENT_WIN_TARGET', 'race_id': race_id,
            'runner_ids': ids, 'target': target_ref, 'closure_evidence': closure_ref,
            'forecast_completion': completion_ref, 'allocation': allocation, 'plan': plan_ref})
    entries = [{'race_id': race_id, 'race_date': '2026-10-10',
        'forecast_status': failure_kind if failure else 'SEALED_PREJUMP',
        'forecast_completion': completion_ref if not failure or failure_kind == 'LATE_SPEED_SEAL' else None,
        'forecast_terminal': terminal_ref if failure and failure_kind != 'INTERRUPTED_BEFORE_CLAIM' else None, 'runner_ids': ids,
        'label_status': 'UNREAD_FORECAST_FAILURE' if failure else label_status,
        'target': target_ref, 'target_role': 'SELECTED_DEVELOPMENT_WIN_TARGET' if target_ref else None,
        'identity_proof': proof_ref, 'closure_evidence': closure_ref, 'failure_evidence': failure_evidence}]
    members_hash = pin([{'race_id': race_id, 'race_date': '2026-10-10'}])
    manifest = put('result-manifest.json', {'schema_version': 'prospective_speed_admitted_result_manifest_v1',
        'status': 'ROOT_VERIFIED_SELECTED_DEVELOPMENT_CLOSURE', 'plan': plan_ref,
        'allocation': allocation, 'members_sha256': members_hash,
        'collection_terminal': True, 'closure_terminal': True, 'entries': entries})
    authority = put('evaluation-authority.json', {'schema_version': 'prospective_speed_evaluation_authority_v1',
        'status': 'AUTHORIZED_SELECTED_DEVELOPMENT_EVALUATION', 'authority_reference': 'fabricated:test-only',
        'plan': plan_ref, 'allocation': allocation, 'result_manifest': manifest,
        'members_sha256': members_hash, 'additional_result_requests': 0})
    job = put('evaluation-job.json', {'schema_version': 'prospective_speed_evaluation_job_v1',
        'plan': plan_ref, 'populations': [population_ref], 'date_accounting': day_ref,
        'allocation': allocation, 'result_manifest': manifest, 'evaluation_authority': authority,
        'forecast_root': str(forecast_root)})
    clock = {'now': datetime.fromisoformat('2026-10-25T12:05:00+11:00')}
    monkeypatch.setattr(runtime, 'utc_now', lambda: clock['now'])
    return {'job': job, 'clock': clock, 'output': tmp_path/'evaluation',
        'target': target_ref, 'proof': proof_ref, 'closure': closure_ref,
        'completion': completion_ref, 'manifest': manifest, 'payload': payload_ref}


def test_complete_durable_single_analysis_then_restart_without_target_reads(tmp_path, monkeypatch):
    f = make_fixture(tmp_path, monkeypatch)
    result = worker.run_evaluation(f['job'], f['output'])
    assert result['status'] == 'COMPLETE_SINGLE_PLANNED_EVALUATION'
    report = json.loads((f['output']/'evaluation.private.json').read_bytes())
    assert report['accounting']['selected_races'] == report['accounting']['scored_races'] == 1
    first = (f['output']/'evaluation.private.json').read_bytes()
    Path(f['target']['path']).unlink()
    second = worker.run_evaluation(f['job'], f['output'])
    assert second['status'] == 'EVALUATION_ALREADY_CONSUMED'
    assert second['result_accesses_consumed'] == 0
    assert (f['output']/'evaluation.private.json').read_bytes() == first


def test_actual_clock_gate_precedes_manifest_and_label_reads(tmp_path, monkeypatch):
    f = make_fixture(tmp_path, monkeypatch)
    f['clock']['now'] = datetime.fromisoformat('2026-10-25T12:04:59+11:00')
    Path(f['manifest']['path']).unlink()
    Path(f['target']['path']).unlink()
    result = worker.run_evaluation(f['job'], f['output'])
    assert result == {'status': 'WAIT_FOR_FIXED_EVALUATION_TIME', 'result_accesses_consumed': 0}
    assert not f['output'].exists()


def test_claim_and_access_record_exist_before_any_result_read(tmp_path, monkeypatch):
    f = make_fixture(tmp_path, monkeypatch)
    original = worker.Reader.read
    target_paths = {f[key]['path'] for key in ('target', 'proof', 'closure')}
    seen = []
    def audited(reader, ref):
        if ref['path'] in target_paths:
            assert (f['output']/'evaluation-claim.json').exists()
            assert (f['output']/'result-access-001.json').exists()
            seen.append(ref['path'])
        return original(reader, ref)
    monkeypatch.setattr(worker.Reader, 'read', audited)
    assert worker.run_evaluation(f['job'], f['output'])['status'] == 'COMPLETE_SINGLE_PLANNED_EVALUATION'
    assert set(seen) == target_paths


def test_partial_outcome_read_failure_consumes_look_and_is_not_retried(tmp_path, monkeypatch):
    f = make_fixture(tmp_path, monkeypatch, target_change={'runner_ids': ['native:dog2', 'native:dog1']})
    result = worker.run_evaluation(f['job'], f['output'])
    assert result['status'] == 'FAILED_LOOK_NO_RETRY'
    assert result['result_accesses_consumed'] == 1
    assert not (f['output']/'evaluation.private.json').exists()
    Path(f['target']['path']).unlink()
    repeated = worker.run_evaluation(f['job'], f['output'])
    assert repeated['status'] == 'EVALUATION_ALREADY_CONSUMED'
    assert repeated['original_status'] == 'FAILED_LOOK_NO_RETRY'


def test_interrupted_claim_is_terminal_without_label_retry(tmp_path, monkeypatch):
    f = make_fixture(tmp_path, monkeypatch)
    f['output'].mkdir(mode=0o700)
    runtime.put_new(f['output']/'evaluation-claim.json', {'job': f['job']})
    Path(f['target']['path']).unlink()
    result = worker.run_evaluation(f['job'], f['output'])
    assert result['status'] == 'INTERRUPTED_LOOK_NO_RETRY'
    assert result['result_accesses_consumed'] == 0


@pytest.mark.parametrize('status', ['QUARANTINED_IDENTITY', 'MISSING_AT_DEADLINE'])
def test_missing_or_quarantined_results_are_not_read(tmp_path, monkeypatch, status):
    f = make_fixture(tmp_path, monkeypatch, label_status=status)
    result = worker.run_evaluation(f['job'], f['output'])
    assert result['status'] == 'COMPLETE_SINGLE_PLANNED_EVALUATION'
    assert result['result_accesses_consumed'] == 0
    report = json.loads((f['output']/'evaluation.private.json').read_bytes())
    assert report['accounting']['selected_races'] == 1
    assert report['accounting']['scored_races'] == 0
    assert report['accounting']['label_status_counts'] == {status: 1}


def test_real_failure_terminal_stays_in_population_without_target(tmp_path, monkeypatch):
    f = make_fixture(tmp_path, monkeypatch, failure=True)
    assert worker.run_evaluation(f['job'], f['output'])['result_accesses_consumed'] == 0
    report = json.loads((f['output']/'evaluation.private.json').read_bytes())
    assert report['accounting']['forecast_status_counts'] == {'SPEED_PROCESSING_TIMEOUT': 1}


@pytest.mark.parametrize('failure', ['LATE_SPEED_SEAL', 'INTERRUPTED_SEAL', 'INTERRUPTED_BEFORE_CLAIM'])
def test_late_and_interrupted_seal_paths_remain_in_full_denominator(tmp_path, monkeypatch, failure):
    f = make_fixture(tmp_path, monkeypatch, failure=failure)
    result = worker.run_evaluation(f['job'], f['output'])
    assert result['status'] == 'COMPLETE_SINGLE_PLANNED_EVALUATION'
    assert result['result_accesses_consumed'] == 0
    report = json.loads((f['output']/'evaluation.private.json').read_bytes())
    assert report['accounting']['forecast_status_counts'] == {failure: 1}
    assert report['accounting']['selected_races'] == 1


@pytest.mark.parametrize('changed', [False, True])
def test_coordinator_preforecast_failure_is_bound_to_exact_plan_population(tmp_path, monkeypatch, changed):
    f = make_fixture(tmp_path, monkeypatch, failure=True)
    job = json.loads(Path(f['job']['path']).read_bytes())
    manifest = json.loads(Path(f['manifest']['path']).read_bytes())
    entry = manifest['entries'][0]
    terminal_path = Path(entry['forecast_terminal']['path'])
    (terminal_path.parent/'claim.json').unlink()
    (terminal_path.parent/'completion.json').unlink()
    terminal_path.unlink()
    frozen_plan = json.loads(Path(job['plan']['path']).read_bytes())
    population = json.loads(Path(job['populations'][0]['path']).read_bytes())
    entry['forecast_status'] = 'UPSTREAM_FORECAST_UNAVAILABLE'
    entry['forecast_terminal'] = runtime.put_new(terminal_path, {
        'schema_version': 'prospective_speed_stage_failure_v1', 'race_id': entry['race_id'],
        'status': entry['forecast_status'], 'reason': 'NO_CONSUMABLE_SEAL_BEFORE_JUMP',
        'at': '2026-10-10T13:11:00+11:00', 'plan_sha256': pin(frozen_plan),
        'population_sha256': '0'*64 if changed else pin(population)})
    Path(f['manifest']['path']).unlink()
    job['result_manifest'] = runtime.put_new(Path(f['manifest']['path']), manifest)
    auth_path = Path(job['evaluation_authority']['path'])
    auth = json.loads(auth_path.read_bytes()); auth_path.unlink()
    auth['result_manifest'] = job['result_manifest']
    job['evaluation_authority'] = runtime.put_new(auth_path, auth)
    Path(f['job']['path']).unlink()
    f['job'] = runtime.put_new(Path(f['job']['path']), job)
    if changed:
        with pytest.raises(worker.EvaluationIORejected, match='PRE_FORECAST_FAILURE_BINDING_CHANGED'):
            worker.run_evaluation(f['job'], f['output'])
        assert not (f['output']/'evaluation-claim.json').exists()
    else:
        result = worker.run_evaluation(f['job'], f['output'])
        assert result['status'] == 'COMPLETE_SINGLE_PLANNED_EVALUATION'
        report = json.loads((f['output']/'evaluation.private.json').read_bytes())
        assert report['accounting']['forecast_status_counts'] == {'UPSTREAM_FORECAST_UNAVAILABLE': 1}


def test_missing_entire_date_is_rejected_before_claim_or_targets(tmp_path, monkeypatch):
    f = make_fixture(tmp_path, monkeypatch, missing_date=True)
    with pytest.raises(plan.PlanRejected, match='DATE_ACCOUNTING_INCOMPLETE'):
        worker.run_evaluation(f['job'], f['output'])
    assert not (f['output']/'evaluation-claim.json').exists()


def test_changed_seal_chain_rejects_before_any_target_read(tmp_path, monkeypatch):
    f = make_fixture(tmp_path, monkeypatch)
    path = Path(f['completion']['path'])
    path.write_text('{}')
    with pytest.raises(ValueError, match='INPUT_HASH'):
        worker.run_evaluation(f['job'], f['output'])
    assert not (f['output']/'evaluation-claim.json').exists()


def test_bound_retained_closure_bytes_must_still_match(tmp_path, monkeypatch):
    f = make_fixture(tmp_path, monkeypatch)
    Path(f['closure']['path']).write_text('{}')
    result = worker.run_evaluation(f['job'], f['output'])
    assert result['status'] == 'FAILED_LOOK_NO_RETRY'
    assert result['result_accesses_consumed'] == 1


def test_target_from_after_result_authority_deadline_is_not_scored(tmp_path, monkeypatch):
    f = make_fixture(tmp_path, monkeypatch,
        target_change={'official_observed_at': '2026-10-25T12:00:01+11:00'})
    assert worker.run_evaluation(f['job'], f['output'])['status'] == 'FAILED_LOOK_NO_RETRY'


def test_forecast_payload_uses_measured_output_reader(tmp_path, monkeypatch):
    f = make_fixture(tmp_path, monkeypatch)
    seen = []
    original = runtime.read_output
    def read(ref):
        seen.append(ref)
        return original(ref)
    monkeypatch.setattr(runtime, 'read_output', read)
    worker.run_evaluation(f['job'], f['output'])
    assert seen == [f['payload']]
