"""Outcome-blind population and once-after-closure analysis contract."""
import math
import hashlib
import json

import pytest

from race_collection.prospective_speed_plan import (
    ALLOCATION_SHA256, PlanRejected, build_plan, evaluate_closed_population,
    evaluation_gate, forecast_admission, precision_diagnostic, select_population,
)
from race_collection.sectional_speed_evaluation import adjusted_probabilities


PROTECTED = {'path': '/private/protected-membership.json', 'sha256': 'c' * 64}


def plan():
    references = {key: {'path': '/private/' + key + '.json', 'sha256': 'a' * 64}
        for key in ('allocation', 'historical_amendment', 'exclusive_amendment',
                    'current_user', 'result_runtime', 'reservation_registry')}
    references['allocation']['sha256'] = ALLOCATION_SHA256
    return build_plan(frozen_at='2026-10-06T19:00:00+11:00',
        candidate_reference={'path': '/private/candidate.json', 'sha256': 'b' * 64},
        authority_references=references,
        precision=precision_diagnostic([-.0235, .0229, 0, .0039, -.0157, -.0047]))


def races(day='2026-10-10'):
    return [{'race_id': f'Race {i + 1} - TEST - {day}', 'race_key': f'{day}|TEST|{i + 1}',
        'jump_at': f'{day}T13:{10 + i * 5:02d}:00+11:00'} for i in range(8)]


def population(day='2026-10-10', protected=()):
    return select_population(plan(), races(day), local_date=day,
        frozen_at=day + 'T12:50:02+11:00', source_observed_at=day + 'T12:48:00+11:00',
        index_complete=True, protected_membership_reference=PROTECTED, protected_race_ids=protected)


def records(pop):
    rows = []
    for item in pop['observed_races'][:6]:
        if item['race_id'] not in pop['selected_race_ids']:
            continue
        minute = int(item['jump_at'][14:16]) - 5
        base = [.5, .3, .2]
        values, support = [.5, 0., -.2], [True, False, True]
        rows.append({'race_id': item['race_id'], 'race_date': pop['local_date'],
            'forecast_status': 'SEALED_PREJUMP', 'label_status': 'FULL_ORDER_WIN_ELIGIBLE',
            'cutoff': f"{pop['local_date']}T13:{minute:02d}:00+11:00",
            'sealed_at': f"{pop['local_date']}T13:{minute:02d}:01+11:00",
            'input_available_at': f"{pop['local_date']}T13:{minute:02d}:00+11:00",
            'runner_ids': ['d1', 'd2', 'd3'], 'speed_estimates': values,
            'speed_supported': support, 'result_identity_verified': True,
            'probabilities': {'market': [.4, .3, .3], 'baseline': base,
                'baseline_speed': adjusted_probabilities(base, values, support, .1)},
            'outcome': [1., 0., 0.]})
    return rows


def evaluate(rows, pops=None, **overrides):
    pops = pops or [population()]
    by_date = {p['local_date']: p for p in pops}
    date_accounting = [{'local_date': day, 'status': 'POPULATION_FROZEN',
        'population_sha256': hashlib.sha256(json.dumps(by_date[day],sort_keys=True,separators=(',', ':')).encode()).hexdigest()}
        if day in by_date else {'local_date': day, 'status': 'INDEX_MISSING', 'reason': 'No fixture index',
            'population_sha256': None} for day in ('2026-10-10', '2026-10-11')]
    return evaluate_closed_population(plan(), pops, rows, date_accounting=date_accounting,
        **dict(now='2026-10-25T12:05:00+11:00', collection_terminal=True,
               closure_terminal=True, **overrides))


def test_selects_original_first_six_before_any_qualification():
    pop = population()
    assert pop['first_six_race_ids'] == [r['race_id'] for r in races()[:6]]
    assert len(pop['dispositions']) == 8
    assert [r['disposition'] for r in pop['dispositions']][-2:] == ['BEYOND_FIRST_SIX'] * 2


def test_protected_selected_race_is_not_replaced_by_seventh():
    blocked = races()[0]['race_id']
    pop = population(protected=[blocked])
    assert len(pop['selected_race_ids']) == 5
    assert blocked not in pop['selected_race_ids']
    assert races()[6]['race_id'] not in pop['selected_race_ids']
    assert pop['dispositions'][0]['disposition'] == 'PROTECTED_STUDY_MEMBER'


@pytest.mark.parametrize('time', ['12:49:59', '12:51:00', '13:00:00'])
def test_no_late_or_early_population_freeze(time):
    with pytest.raises(PlanRejected, match='FREEZE_NOT_DUE'):
        select_population(plan(), races(), local_date='2026-10-10',
            frozen_at=f'2026-10-10T{time}+11:00',
            source_observed_at='2026-10-10T12:49:00+11:00', index_complete=True, protected_membership_reference=PROTECTED)


@pytest.mark.parametrize('observed,complete', [
    ('2026-10-10T12:44:59+11:00', True),
    ('2026-10-10T12:50:01+11:00', True),
    ('2026-10-10T12:48:00+11:00', False),
])
def test_census_must_be_complete_fresh_and_already_available(observed, complete):
    with pytest.raises(PlanRejected, match='INDEX_NOT_COMPLETE_AND_FRESH'):
        select_population(plan(), races(), local_date='2026-10-10',
            frozen_at='2026-10-10T12:50:00+11:00', source_observed_at=observed,
            index_complete=complete, protected_membership_reference=PROTECTED)


def test_normalizes_utc_freeze_to_melbourne():
    pop = select_population(plan(), races(), local_date='2026-10-10',
        frozen_at='2026-10-10T01:50:00+00:00',
        source_observed_at='2026-10-10T01:48:00+00:00', index_complete=True, protected_membership_reference=PROTECTED)
    assert len(pop['selected_race_ids']) == 6


def test_duplicate_census_race_fails():
    with pytest.raises(PlanRejected, match='POPULATION_DUPLICATE'):
        select_population(plan(), races() + races()[:1], local_date='2026-10-10',
            frozen_at='2026-10-10T12:50:00+11:00',
            source_observed_at='2026-10-10T12:48:00+11:00', index_complete=True, protected_membership_reference=PROTECTED)


def test_plan_and_population_mutation_are_rejected():
    altered = plan(); altered['beta'] = .25
    with pytest.raises(PlanRejected, match='PLAN_CHANGED'):
        evaluation_gate(altered, now='2026-10-25T13:00:00+11:00',
                        collection_terminal=True, closure_terminal=True)
    pop = population(); pop['selected_race_ids'].append(races()[6]['race_id'])
    with pytest.raises(PlanRejected, match='POPULATION_CHANGED'):
        forecast_admission(plan(), pop, races()[0]['race_id'],
            cutoff='2026-10-10T13:05:00+11:00', sealed_at='2026-10-10T13:05:01+11:00',
            input_available_at='2026-10-10T13:05:00+11:00')


@pytest.mark.parametrize('change,reason', [
    ({'sealed_at': '2026-10-10T13:10:00+11:00'}, 'FORECAST_NOT_PREJUMP'),
    ({'input_available_at': '2026-10-10T13:05:01+11:00'}, 'INPUT_AFTER_CUTOFF'),
    ({'cutoff': '2026-10-10T13:09:00+11:00',
      'sealed_at': '2026-10-10T13:09:01+11:00'}, 'PRICE_SNAPSHOT'),
    ({'prior_attempt': True}, 'ATTEMPT_ALREADY_CONSUMED'),
])
def test_real_seal_time_cutoff_and_once_attempt_gate(change, reason):
    args = dict(cutoff='2026-10-10T13:05:00+11:00',
        sealed_at='2026-10-10T13:05:01+11:00', input_available_at='2026-10-10T13:05:00+11:00')
    args.update(change)
    with pytest.raises(PlanRejected, match=reason):
        forecast_admission(plan(), population(), races()[0]['race_id'], **args)


def test_cannot_inspect_comparative_performance_before_fixed_time():
    # Outcomes are not touched before gate: poison records still yield time reason.
    with pytest.raises(PlanRejected, match='WAIT_FOR_FIXED_EVALUATION_TIME'):
        evaluate_closed_population(plan(), [], None, date_accounting=[], now='2026-10-11T15:00:00+11:00',
            collection_terminal=True, closure_terminal=True)
    assert evaluation_gate(plan(), now='2026-10-26T12:00:00+11:00',
        collection_terminal=True, closure_terminal=True,
        prior_evaluation_claim=True) == 'EVALUATION_ALREADY_CLAIMED'


def test_rejects_missing_member_in_evaluation():
    with pytest.raises(PlanRejected, match='DENOMINATOR_CHANGED'):
        evaluate(records(population())[:-1])


def test_neutral_fallback_missing_results_and_failures_remain_in_accounting():
    rows = records(population())
    rows[0]['speed_supported'] = [False] * 3
    rows[0]['speed_estimates'] = [0.] * 3
    rows[0]['probabilities']['baseline_speed'] = list(rows[0]['probabilities']['baseline'])
    rows[1].update(label_status='MISSING_AT_DEADLINE', outcome=None)
    rows[2] = {key: rows[2][key] for key in ('race_id', 'race_date')}
    rows[2].update(forecast_status='SPEED_WORKER_TIMEOUT', label_status='UNREAD',
                   outcome=None, probabilities=None)
    result = evaluate(rows)
    assert result['accounting']['selected_races'] == 6
    assert result['accounting']['scored_races'] == 4
    assert result['accounting']['scored_races_neutral_fallback'] == 1
    assert len(result['dispositions']) == 6
    assert result['principal_complete_case_comparison']['uncertainty']['interval_95'] is None
    assert result['fit_trials'] == []


def test_scoring_is_paired_and_arithmetically_correct():
    rows = records(population())
    result = evaluate(rows)
    p = rows[0]['probabilities']['baseline_speed']
    main = result['principal_complete_case_comparison']
    assert main['metrics']['baseline_speed']['log_loss'] == pytest.approx(-math.log(p[0]))
    assert main['metrics']['baseline_speed']['brier'] == pytest.approx((p[0]-1)**2+p[1]**2+p[2]**2)
    assert main['speed_minus']['baseline']['log_loss'] == pytest.approx(-math.log(p[0]) + math.log(.5))


def test_exact_frozen_strength_is_checked_at_evaluation():
    rows = records(population())
    rows[0]['probabilities']['baseline_speed'] = adjusted_probabilities(
        rows[0]['probabilities']['baseline'], rows[0]['speed_estimates'], rows[0]['speed_supported'], .25)
    with pytest.raises(PlanRejected, match='FROZEN_ADJUSTMENT_CHANGED'):
        evaluate(rows)


@pytest.mark.parametrize('mutation,reason', [
    ({'label_status': 'QUARANTINED'}, 'UNVERIFIED_TARGET_MUST_BE_ABSENT'),
    ({'result_identity_verified': False}, 'RESULT_IDENTITY_NOT_VERIFIED'),
    ({'outcome': [.6, .4, 0]}, 'OUTCOME_INVALID'),
])
def test_result_identity_and_missingness_cannot_be_weakened(mutation, reason):
    rows = records(population()); rows[0].update(mutation)
    with pytest.raises(PlanRejected, match=reason):
        evaluate(rows)


def test_known_nonfinish_win_result_and_dead_heat_are_supported():
    rows = records(population())
    rows[0].update(label_status='KNOWN_NONFINISH_WIN_ELIGIBLE', outcome=[.5, .5, 0.])
    assert evaluate(rows)['accounting']['scored_races'] == 6


def test_two_date_influence_never_refits_candidate():
    pops = [population(), population('2026-10-11')]
    result = evaluate(records(pops[0]) + records(pops[1]), pops)
    assert result['accounting']['selected_races'] == 12
    assert result['principal_complete_case_comparison']['date_count'] == 2
    assert all(row['race_count'] == 6 for row in result['leave_one_date_out'].values())
    assert result['beta'] == .1 and result['fit_trials'] == []


def test_precision_plan_does_not_promise_tiny_effect_resolution():
    result = precision_diagnostic([-.0235452320, .0228888375, 0, .0039051624, -.0157188123, -.0046933031])
    assert result['approximate_dates_for_95_percent_half_width_equal_effect'] == 180
    assert result['approximate_dates_for_80_percent_power_two_sided_5_percent'] == 367
    assert not result['supports_precise_test_of_exploratory_effect']


def test_entire_missing_collection_date_cannot_disappear():
    with pytest.raises(PlanRejected, match='DATE_ACCOUNTING_INCOMPLETE'):
        evaluate_closed_population(plan(), [], [], date_accounting=[],
            now='2026-10-25T12:05:00+11:00', collection_terminal=True, closure_terminal=True)


def amendment():
    return {'schema_version': 'prospective_speed_schedule_amendment_v1',
        'status': 'AUTHORIZED_EARLIER_DEVELOPMENT_SCHEDULE', 'authority_reference': 'fixture:user-earlier-authority',
        'issued_at': '2026-10-06T21:00:00+11:00',
        'selection_windows': [{'local_date': day, 'freeze_at': day+'T'+freeze+'+11:00',
            'jump_start': day+'T'+begin+'+11:00', 'jump_end': day+'T23:59:59+11:00'}
            for day, freeze, begin in [('2026-10-06', '22:00:00', '22:20:00'),
                                      ('2026-10-07', '07:00:00', '07:20:00')]],
        'result_requests_stop_at': '2026-10-09T12:00:00+11:00',
        'evaluation_at': '2026-10-09T12:05:00+11:00', 'maximum_total': 12, 'maximum_per_date': 6,
        'beta': .1, 'additional_source_requests': 0, 'additional_result_requests': 0,
        'preserve_existing_membership': True}


def earlier_plan(reference=None, document=None, allocation=None):
    original = plan()
    authorities = original['authority']
    if allocation:
        authorities['allocation'] = allocation
    return build_plan(frozen_at='2026-10-06T21:10:00+11:00', candidate_reference=original['candidate'],
        authority_references=authorities, precision=original['precision'],
        schedule_amendment_reference=reference or {'path': '/private/amendment.json', 'sha256': 'c'*64},
        schedule_amendment=document or amendment())


def late_races():
    return [{'race_id': f'Race {i+1} - TEST - 2026-10-06', 'race_key': f'2026-10-06|TEST|{i+1}',
        'jump_at': f'2026-10-06T22:{20+i*5:02d}:00+11:00'} for i in range(8)]


def test_v2_changes_only_declared_schedule_and_uses_new_allocation_reference():
    original, new = plan(), earlier_plan(allocation={'path': '/private/new-allocation.json', 'sha256': 'd'*64})
    for field in ('candidate', 'candidate_commit', 'baseline', 'beta', 'methods', 'history_rule',
                  'primary', 'secondary', 'missing_history', 'dead_heat', 'evaluation_looks'):
        assert new[field] == original[field]
    assert new['dates'] == ['2026-10-06', '2026-10-07']
    assert new['population']['maximum_total'] == 12 and new['population']['maximum_per_date'] == 6
    assert evaluation_gate(new, now='2026-10-09T12:04:59+11:00',
        collection_terminal=True, closure_terminal=True) == 'WAIT_FOR_FIXED_EVALUATION_TIME'


def test_v2_first_six_daily_census_missing_time_and_protected_do_not_replace():
    rows = late_races()
    rows.append({'race_id': 'Race 9 - TEST - 2026-10-06', 'race_key': '2026-10-06|TEST|9', 'jump_at': None})
    result = select_population(earlier_plan(), list(reversed(rows)), local_date='2026-10-06',
        frozen_at='2026-10-06T22:00:30+11:00', source_observed_at='2026-10-06T21:31:00+11:00',
        index_complete=True, protected_membership_reference=PROTECTED, protected_race_ids=[rows[0]['race_id']])
    assert result['first_six_race_ids'] == [row['race_id'] for row in rows[:6]]
    assert result['selected_race_ids'] == [row['race_id'] for row in rows[1:6]]
    assert len(result['dispositions']) == 9
    assert result['dispositions'][0]['disposition'] == 'MISSING_JUMP_TIME'


@pytest.mark.parametrize('mutation', [{'maximum_total': 13}, {'beta': .2}, {'additional_source_requests': 1}])
def test_v2_cannot_change_frozen_scope_or_candidate(mutation):
    value = amendment(); value.update(mutation)
    with pytest.raises(PlanRejected, match='SCHEDULE_AMENDMENT_INVALID'):
        earlier_plan(document=value)


def test_v2_rejects_stale_inventory_and_after_midnight_is_outside_declared_window():
    rows = late_races()
    rows[-1]['jump_at'] = '2026-10-07T00:03:00+11:00'
    kwargs = dict(local_date='2026-10-06', frozen_at='2026-10-06T22:00:00+11:00',
        index_complete=True, protected_membership_reference=PROTECTED)
    with pytest.raises(PlanRejected, match='INDEX_NOT_COMPLETE_AND_FRESH'):
        select_population(earlier_plan(), rows, source_observed_at='2026-10-06T21:29:59+11:00', **kwargs)
    result = select_population(earlier_plan(), rows, source_observed_at='2026-10-06T21:30:00+11:00', **kwargs)
    assert result['dispositions'][-1]['disposition'] == 'OUTSIDE_SELECTION_WINDOW'


def test_v2_evaluation_accounts_for_amended_dates_without_old_dates():
    new = earlier_plan()
    accounts = [{'local_date': day, 'status': 'INDEX_MISSING', 'reason': 'fixture absent',
                 'population_sha256': None} for day in new['dates']]
    result = evaluate_closed_population(new, [], [], date_accounting=accounts,
        now=new['evaluation_at'], collection_terminal=True, closure_terminal=True)
    assert set(result['date_results']) == set(new['dates'])
