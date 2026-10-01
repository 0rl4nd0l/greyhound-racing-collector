"""Offline incident boundaries: native validators, source control and admission."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pytest

from tests.fixtures.incident_engineering_case import make_incident, put
from src.predictor.future_comparison import load_plan
from race_collection.incident_comparison import assert_incident_race_allowed, result_deadline


def case(tmp_path):
    registry = put(tmp_path/'registry.json', {'frozen': True})
    study = dict(schema_version='frozen_four_way_comparison_plan_v2', status='AUTHORIZED',
        authority_reference='SYNTHETIC', candidate_registry=registry,
        activated_at='2026-09-30T12:00:00+10:00', starts_at='2026-10-01T12:00:00+10:00',
        ends_at='2027-01-21T12:00:00+11:00', programme_root=str(tmp_path/'study-members'),
        prediction_output_roots=[str(tmp_path/'study-bundles')], decision_seconds_before_jump=120,
        quote_lead_seconds=[120,600], denied_history_intervals=[],
        machine_history_authority_reference='SYNTHETIC', history_policy='strictly_earlier_machine_features_only',
        fixed_closure_days=14, missing_result_policy='bounded_paired_losses_v1',
        exclusive_population_allocation_reference='SYNTHETIC', reservation_review_sha256='a'*64)
    study_ref = put(tmp_path/'study.json', study)
    authority, ref = make_incident(tmp_path, study_plan=study_ref, candidate_registry=registry)
    slot = authority['slots'][0]
    plan = {**study, 'status':'AUTHORIZED_ENGINEERING', 'incident_authority':ref, 'incident_slot':'001',
        'authority_reference':authority['authority_reference'], 'activated_at':'2026-10-01T16:00:01+10:00',
        'starts_at':slot['starts_at'], 'ends_at':slot['cleanup_by'],
        'programme_root':str(Path(authority['state_root'])/'admission/001'),
        'prediction_output_roots':[str(Path(authority['prediction_root'])/'bundles')],
        'study_enrolment':False, 'performance_evaluation':False}
    plan_ref = put(tmp_path/'plan.json', plan)
    return authority, ref, plan, plan_ref


def test_native_plan_and_result_deadline_preserve_study_population(tmp_path):
    authority, ref, plan, plan_ref = case(tmp_path)
    assert load_plan(Path(plan_ref['path']), plan_ref['sha256'])[0] == plan
    assert result_deadline(plan) == datetime.fromisoformat(authority['result_deadline'])
    assert_incident_race_allowed(plan, 'invented-race')
    study = json.loads(Path(authority['study_plan']['path']).read_bytes())
    claim = Path(study['programme_root'])/authority['study_plan']['sha256']/'attempts'/hashlib.sha256(b'invented-race').hexdigest()
    claim.mkdir(parents=True)
    with pytest.raises(ValueError, match='study_admitted'):
        assert_incident_race_allowed(plan, 'invented-race')
    assert claim.is_dir()


@pytest.mark.parametrize('key,value', [('study_enrolment',True), ('performance_evaluation',True),
    ('quote_lead_seconds',[0,600]), ('machine_history_authority_reference','changed'),
    ('programme_root','/tmp/collision'), ('incident_slot','003')])
def test_native_plan_rejects_unbound_engineering_policy(tmp_path, key, value):
    _, _, plan, _ = case(tmp_path)
    plan[key] = value
    ref = put(tmp_path/'invalid.json', plan)
    with pytest.raises((ValueError, KeyError)):
        load_plan(Path(ref['path']), ref['sha256'])


def test_engineering_never_enters_evaluator(tmp_path):
    from scripts.evaluate_frozen_comparison import evaluate
    _, _, _, ref = case(tmp_path)
    # Missing outcome files are never opened: class rejection precedes access.
    with pytest.raises(ValueError, match='real_evaluation_not_authorized'):
        evaluate(Path(ref['path']), ref['sha256'], tmp_path/'ABSENT_AUTHORITY',
                 'a'*64, tmp_path/'ABSENT_RESULTS', tmp_path/'ABSENT_OUTPUT')


def test_source_grant_requires_exact_owner_and_retains_consumption(tmp_path, monkeypatch):
    from utils.sportsbet_access import SportsbetAccess, SportsbetAccessBlocked
    from race_collection.incident_engineering import incident_source_usage
    authority, ref, _, _ = case(tmp_path)
    now = datetime.fromisoformat('2026-10-01T16:05:00+10:00').timestamp()
    source = SportsbetAccess(tmp_path/'source.json', clock=lambda:now)
    source.initialize(access_basis={'status':'permitted','reference':'SYNTHETIC'})
    source.authorize_diagnostic(reference=authority['authority_reference']+':slot:001',
        expected_sha256=hashlib.sha256(source.path.read_bytes()).hexdigest(),
        expires_at=datetime.fromisoformat(authority['slots'][0]['ends_at']).timestamp(),
        max_operations=192, rationale='SYNTHETIC', incident_authority=ref, incident_slot='001')
    with pytest.raises(SportsbetAccessBlocked, match='owner_required'):
        source.check_admission()
    monkeypatch.setenv('GREYHOUND_INCIDENT_AUTHORITY_SHA256', ref['sha256'])
    monkeypatch.setenv('GREYHOUND_INCIDENT_SLOT', '001')
    source.check_admission()
    assert incident_source_usage(source.read(),0) == 0
    source.retain_denial(403, reason='SYNTHETIC')
    with pytest.raises(SportsbetAccessBlocked):
        source.check_admission()
    assert len(source.read()['denials']) == 1
