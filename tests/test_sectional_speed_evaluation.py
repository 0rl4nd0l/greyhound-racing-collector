"""Fabricated races test public evaluation and probability interfaces only."""
from copy import deepcopy
import math

import pytest

from race_collection.sectional_speed_evaluation import (
    EvaluationRejected, adjusted_probabilities, evaluate_experiment,
    frozen_protocol, normalized_market,
)


def ref(kind):
    return {'path': '/fabricated/' + kind, 'sha256': 'a' * 64}


def race(day, suffix='', *, winner=0, supported=True):
    return {'race_id': f'fictional-{day}-{suffix}', 'race_date': f'2026-10-{day:02d}',
        'cutoff': f'2026-10-{day:02d}T12:00:00+11:00', 'runner_ids': ['entry-a', 'entry-b'],
        'market_odds': [2.0, 2.0], 'stored_market_probabilities': [0.5, 0.5],
        'stored_baseline_probabilities': [0.5, 0.5],
        'reproduced_baseline_probabilities': [0.5, 0.5],
        'speed_estimates': [1.0, 0.0] if supported else [0.0, 0.0],
        'speed_supported': [True, False] if supported else [False, False],
        'label_status': 'FULL_ORDER_WIN_ELIGIBLE', 'outcome': [1.0, 0.0] if winner == 0 else [0.0, 1.0],
        'source_bindings': {k: ref(k) for k in ('forecast', 'reproduction', 'features', 'label')}}


def protocol(rows):
    return frozen_protocol([{'race_id': row['race_id'], 'race_date': row['race_date']} for row in rows],
        membership_reference=ref('membership'), label_authority_reference=ref('authority'),
        feature_policy_reference=ref('features'), baseline_reproduction_reference=ref('reproduction'))


def test_inverse_odds_normalization_matches_worked_example():
    assert normalized_market([2.0, 4.0, 4.0]) == [0.5, 0.25, 0.25]


def test_supported_adjustment_changes_unsupported_probability_only_through_normalization():
    result = adjusted_probabilities([0.5, 0.5], [1.0, 0.0], [True, False], 1.0)
    assert result == pytest.approx([0.7310585786300049, 0.2689414213699951])
    assert math.fsum(result) == pytest.approx(1.0)


def test_all_unsupported_or_zero_coefficient_copies_baseline_exactly():
    baseline = [0.10000000000000002, 0.19999999999999998, 0.7]
    result = adjusted_probabilities(baseline, [0.0] * 3, [False] * 3, 1.0)
    assert [x.hex() for x in result] == [x.hex() for x in baseline]
    assert result is not baseline
    assert adjusted_probabilities(baseline, [1.0, -1.0, 0.0], [True, True, False], 0.0) == baseline


def test_training_grid_validation_gate_and_evaluation_use_one_frozen_coefficient():
    rows = [race(1), race(2), race(3, winner=1)]
    result = evaluate_experiment(rows, protocol(rows))
    assert result['trained_beta'] == result['selected_beta'] == 1.0
    assert result['validation']['accepted_trained_candidate'] is True
    assert [trial['beta'] for trial in result['fit_trials']] == [0.0, 0.1, 0.25, 0.5, 1.0]
    assert all(trial['training_race_ids'] == ['fictional-1-'] for trial in result['fit_trials'])
    metric = result['principal_evaluation']['metrics']
    assert metric['baseline']['log_loss'] == pytest.approx(0.6931471805599453)
    assert metric['baseline']['brier'] == pytest.approx(0.5)
    assert metric['baseline_speed']['log_loss'] == pytest.approx(1.3132616875182228)
    assert metric['baseline_speed']['brier'] == pytest.approx(1.068893290777046)
    assert result['principal_evaluation']['paired_changes']['baseline_speed_minus_baseline']['log_loss'] > 0


def test_progress_sink_preserves_completed_trials_and_stops_on_persistence_failure():
    rows = [race(1), race(2), race(3)]
    events = []

    def sink(event):
        if event['kind'] == 'FIT_TRIAL_STARTED' and event['beta'] == 0.25:
            raise ValueError('FABRICATED_SINK_WRITE_FAILURE')
        events.append(deepcopy(event))

    with pytest.raises(ValueError, match='FABRICATED_SINK_WRITE_FAILURE'):
        evaluate_experiment(rows, protocol(rows), progress_sink=sink)
    completed = [event['trial'] for event in events if event['kind'] == 'FIT_TRIAL_COMPLETED']
    assert [trial['beta'] for trial in completed] == [0.0, 0.1]
    assert completed[0]['training_log_loss'] == pytest.approx(math.log(2))
    assert not any(event['kind'].startswith('VALIDATION') for event in events)


def test_progress_records_bind_selected_coefficient_and_every_final_prediction():
    rows = [race(1), race(2), race(3)]
    events = []
    result = evaluate_experiment(rows, protocol(rows), progress_sink=events.append)
    assert [event['trial'] for event in events if event['kind'] == 'FIT_TRIAL_COMPLETED'] == result['fit_trials']
    validation = next(event for event in events if event['kind'] == 'VALIDATION_COMPLETED')
    assert validation['selected_beta'] == result['selected_beta']
    assert validation['validation'] == result['validation']
    assert [event['record'] for event in events if event['kind'] == 'PREDICTION_COMPLETED'] == result['records']
    assert events[-1]['kind'] == 'REPORTS_COMPLETED'


def test_evaluation_outcome_cannot_change_fit_or_validation_choice():
    rows = [race(1), race(2), race(3)]
    first = evaluate_experiment(rows, protocol(rows))
    rows[-1]['outcome'] = [0.0, 1.0]
    second = evaluate_experiment(rows, protocol(rows))
    assert first['fit_trials'] == second['fit_trials']
    assert first['validation'] == second['validation']
    assert first['selected_beta'] == second['selected_beta']
    assert first['records'][-1]['probabilities'] == second['records'][-1]['probabilities']
    assert first['principal_evaluation']['metrics'] != second['principal_evaluation']['metrics']


def test_bad_validation_reverts_to_zero_without_searching_other_coefficients():
    rows = [race(1), race(2, winner=1), race(3)]
    result = evaluate_experiment(rows, protocol(rows))
    assert result['trained_beta'] == 1.0
    assert result['selected_beta'] == 0.0
    assert result['validation']['accepted_trained_candidate'] is False
    assert all(row['probabilities']['baseline_speed'] == row['probabilities']['baseline'] for row in result['records'])


def test_validation_tie_rejects_trained_nonzero_adjustment():
    rows = [race(1), race(2, supported=False), race(3)]
    result = evaluate_experiment(rows, protocol(rows))
    assert result['trained_beta'] == 1.0
    assert result['selected_beta'] == 0.0


def test_training_tie_chooses_smallest_beta_and_never_refits_on_validation():
    rows = [race(1, supported=False), race(2), race(3)]
    result = evaluate_experiment(rows, protocol(rows))
    assert result['trained_beta'] == result['selected_beta'] == 0.0


def test_incomplete_fields_and_no_speed_races_stay_in_principal_denominator():
    rows = [race(1), race(2), race(3), race(3, 'missing', supported=False)]
    result = evaluate_experiment(rows, protocol(rows))
    assert result['principal_evaluation']['race_count'] == 2
    assert result['supported_only_secondary_evaluation']['race_count'] == 1
    assert result['accounting']['eligible_races_with_exact_baseline_fallback'] == 1
    unsupported = next(row for row in result['records'] if row['race_id'].endswith('missing'))
    assert unsupported['probabilities']['baseline_speed'] == unsupported['probabilities']['baseline']


def test_quarantined_race_is_preserved_without_opening_label_or_contributing_loss():
    rows = [race(1), race(2), race(3), race(3, 'quarantine')]
    rows[-1].update(label_status='QUARANTINED_IDENTITY_UNRESOLVED', outcome=None)
    rows[-1]['source_bindings']['label'] = None
    result = evaluate_experiment(rows, protocol(rows))
    assert result['accounting']['population_races'] == 4
    assert result['accounting']['excluded_races'] == 1
    assert result['principal_evaluation']['race_count'] == 1
    assert result['records'][-1]['losses'] is None


def test_quarantined_label_values_are_rejected_even_if_not_used_for_metrics():
    rows = [race(1), race(2), race(3)]
    rows[-1]['label_status'] = 'QUARANTINED'
    with pytest.raises(EvaluationRejected, match='EXCLUDED_LABEL_MUST_BE_UNREAD'):
        evaluate_experiment(rows, protocol(rows))


def test_one_evaluation_date_has_no_spurious_independence_interval():
    rows = [race(1), race(2), race(3), race(3, 'other')]
    result = evaluate_experiment(rows, protocol(rows))
    uncertainty = result['principal_evaluation']['uncertainty']['baseline']
    assert uncertainty['status'] == 'INSUFFICIENT_DATE_CLUSTERS'
    assert uncertainty['date_count'] == 1
    assert uncertainty['log_loss_interval_95'] is None
    assert uncertainty['brier_interval_95'] is None
    assert result['pooled_descriptive_only']['uncertainty']['baseline']['status'] == 'INSUFFICIENT_DATE_CLUSTERS'


def test_one_evaluation_date_omission_is_not_replaced_by_training_or_validation_races():
    rows = [race(1), race(2), race(3, winner=1)]
    result = evaluate_experiment(rows, protocol(rows))
    omitted = result['evaluation_leave_one_date_out_fixed_beta']['2026-10-03']
    assert omitted['race_count'] == 0
    assert omitted['metrics']['baseline_speed']['log_loss'] is None
    assert set(result['evaluation_leave_one_date_out_fixed_beta']) == {'2026-10-03'}
    assert result['date_reports']['2026-10-01']['role'] == 'training'
    assert result['date_reports']['2026-10-02']['role'] == 'validation'
    assert result['date_reports']['2026-10-03']['role'] == 'evaluation'


def test_known_nonfinisher_win_target_and_dead_heat_probabilities_are_supported():
    rows = [race(1), race(2), race(3)]
    rows[2]['label_status'] = 'KNOWN_NONFINISH_WIN_ELIGIBLE'
    result = evaluate_experiment(rows, protocol(rows))
    assert result['accounting']['eligible_races'] == 3
    rows[2].update(label_status='FULL_ORDER_WIN_ELIGIBLE', outcome=[0.5, 0.5])
    result = evaluate_experiment(rows, protocol(rows))
    assert result['principal_evaluation']['metrics']['baseline']['brier'] == 0.0
    assert result['principal_evaluation']['metrics']['baseline']['log_loss'] == pytest.approx(math.log(2))


@pytest.mark.parametrize('key,value,error', [
    ('reproduced_baseline_probabilities', [0.6, 0.4], 'BASELINE_REPRODUCTION_FAILED'),
    ('stored_market_probabilities', [0.6, 0.4], 'MARKET_REPRODUCTION_FAILED'),
    ('runner_ids', ['entry-a', 'entry-a'], 'RUNNER_IDENTITY_INVALID'),
    ('outcome', [0.9, 0.1], 'OUTCOME_INVALID'),
    ('speed_estimates', [1.0, -1.0], 'SPEED_FEATURE_INVALID'),
    ('speed_estimates', [float('nan'), 0], 'SPEED_FEATURE_INVALID'),
    ('cutoff', '2026-09-30T12:00:00Z', 'CUTOFF_INVALID'),
])
def test_changed_or_inconsistent_evidence_fails_closed(key, value, error):
    rows = [race(1), race(2), race(3)]
    rows[0][key] = value
    with pytest.raises(EvaluationRejected, match=error):
        evaluate_experiment(rows, protocol(rows))


def test_protocol_changes_and_duplicate_or_missing_members_are_rejected():
    rows = [race(1), race(2), race(3)]
    frozen = protocol(rows)
    changed = deepcopy(frozen)
    changed['coefficient_grid'].append(2.0)
    with pytest.raises(EvaluationRejected, match='PROTOCOL_CHANGED'):
        evaluate_experiment(rows, changed)
    with pytest.raises(EvaluationRejected, match='DENOMINATOR_CHANGED'):
        evaluate_experiment(rows[:-1], frozen)
    with pytest.raises(EvaluationRejected, match='DENOMINATOR_CHANGED'):
        evaluate_experiment([rows[0], rows[0], rows[2]], frozen)


def test_changing_returned_split_document_does_not_change_fixed_protocol():
    rows = [race(1), race(2), race(3)]
    changed = protocol(rows)
    changed['splits']['training'].append('2026-10-03')
    with pytest.raises(EvaluationRejected, match='SPLITS_NOT_CHRONOLOGICAL'):
        evaluate_experiment(rows, changed)
    assert protocol(rows)['splits']['training'] == ['2026-10-01']


def test_empty_required_split_returns_complete_blocker_accounting_and_no_fit():
    rows = [race(1), race(2), race(3)]
    rows[2].update(label_status='QUARANTINED', outcome=None)
    rows[2]['source_bindings']['label'] = None
    result = evaluate_experiment(rows, protocol(rows))
    assert result['status'] == 'BLOCKED_EMPTY_CHRONOLOGICAL_SPLIT'
    assert result['fit_trials'] == []
    assert len(result['records']) == 3
    assert result['accounting']['split_counts']['evaluation'] == 0


def test_explicit_development_dates_cluster_uncertainty_is_deterministic_and_fixed_fit():
    splits = {'training': ['2026-06-24'], 'validation': ['2026-07-01'],
              'evaluation': [f'2026-07-{day:02d}' for day in range(3, 9)]}
    rows = []
    for index, day in enumerate([day for days in splits.values() for day in days]):
        row = race(1, str(index), winner=1 if index == 7 else 0)
        row.update(race_date=day, cutoff=day + 'T12:00:00+10:00')
        rows.append(row)
    frozen = frozen_protocol([{'race_id': r['race_id'], 'race_date': r['race_date']} for r in rows],
        membership_reference=ref('membership'), label_authority_reference=ref('authority'),
        feature_policy_reference=ref('features'), baseline_reproduction_reference=ref('reproduction'),
        split_dates=splits)
    first = evaluate_experiment(rows, frozen)
    second = evaluate_experiment(rows, frozen)
    assert first == second
    interval = first['principal_evaluation']['uncertainty']['baseline']
    assert interval['status'] == 'DESCRIPTIVE_CLUSTER_BOOTSTRAP'
    assert interval['date_count'] == 6
    assert interval['draws'] == 2000
    assert interval['log_loss_interval_95'][0] < interval['log_loss_interval_95'][1]
    assert first['accounting']['split_counts'] == {'training': 1, 'validation': 1, 'evaluation': 6}
    assert 'FEWER_THAN_TEN_EVALUATION_DATES_LIMIT_CLUSTER_UNCERTAINTY' in first['caveats']
    omissions = first['evaluation_leave_one_date_out_fixed_beta']
    assert set(omissions) == set(splits['evaluation'])
    assert all(report['race_count'] == 5 for report in omissions.values())
    assert omissions['2026-07-08']['metrics']['baseline_speed']['log_loss'] == pytest.approx(0.3132616875182228)


def test_prespecified_tight_baseline_reproduction_tolerance_is_measured():
    rows = [race(1), race(2), race(3)]
    rows[0]['reproduced_baseline_probabilities'] = [0.5 + 5e-14, 0.5 - 5e-14]
    frozen = protocol(rows)
    with pytest.raises(EvaluationRejected, match='BASELINE_REPRODUCTION_FAILED'):
        evaluate_experiment(rows, frozen)
    frozen['baseline_reproduction_max_absolute_tolerance'] = 1e-12
    result = evaluate_experiment(rows, frozen)
    assert 0 < result['accounting']['maximum_supplied_baseline_vector_absolute_error'] < 1e-12
    assert result['records'][0]['probabilities']['baseline'] == [0.5, 0.5]


@pytest.mark.parametrize('splits', [
    {'training': ['2026-10-01'], 'validation': ['2026-10-01'], 'evaluation': ['2026-10-03']},
    {'training': ['2026-10-01'], 'validation': ['2026-10-04'], 'evaluation': ['2026-10-03']},
    {'training': ['2026-10-01'], 'validation': [], 'evaluation': ['2026-10-03']},
])
def test_explicit_splits_must_be_nonempty_disjoint_and_chronological(splits):
    rows = [race(1), race(2), race(3)]
    with pytest.raises(EvaluationRejected, match='SPLITS_'):
        frozen_protocol([{'race_id': r['race_id'], 'race_date': r['race_date']} for r in rows],
            membership_reference=ref('membership'), label_authority_reference=ref('authority'),
            feature_policy_reference=ref('features'), baseline_reproduction_reference=ref('reproduction'),
            split_dates=splits)


@pytest.mark.parametrize('beta', [-1.0, 0.2, 2.0, True, float('nan')])
def test_unplanned_coefficient_is_rejected(beta):
    with pytest.raises(EvaluationRejected, match='COEFFICIENT_NOT_PRESPECIFIED'):
        adjusted_probabilities([0.5, 0.5], [1.0, 0.0], [True, False], beta)
