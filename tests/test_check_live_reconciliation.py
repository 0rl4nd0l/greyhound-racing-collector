"""Continuation review rejects population drift and mixed counting units."""
import copy

import pytest

from scripts.check_live_reconciliation import (
    check_attempt_units, check_forecast_continuity, check_frozen_selection,
)


def test_forecast_outcome_join_cannot_replace_original_probability():
    original = [{'race_id': 'r1', 'runners': [{'probability': .5}]}]
    changed = copy.deepcopy(original)
    changed[0]['runners'][0]['probability'] = .6
    with pytest.raises(ValueError, match='population changed'):
        check_forecast_continuity(original, changed)
    with pytest.raises(ValueError, match='duplicate'):
        check_forecast_continuity(original * 2, original * 2)


def test_failures_after_success_are_not_unsuccessful_races():
    attempts = [dict(canonical_race_key='a', race_id='a', reserved=False, status='APPENDED'),
                dict(canonical_race_key='a', race_id='a', reserved=False, status='BLOCKED', failure_reason='duplicate'),
                dict(canonical_race_key='b', race_id='b', reserved=False, status='BLOCKED', failure_reason='grade'),
                dict(canonical_race_key='b', race_id='b', reserved=False, status='BLOCKED', failure_reason='grade'),
                dict(canonical_race_key='c', race_id='c', reserved=True, status='APPENDED')]
    coverage = dict(attempts=attempts, summary=dict(nonreserved_unique_races=2,
                     nonreserved_races_with_append=1, nonreserved_without_append=1))
    result = check_attempt_units(coverage, [{'race_id': 'a'}])
    assert result['unsuccessful_unique_races'] == 1
    assert result['nonreserved_attempt_rows'] == 4
    assert result['failure_reason_combinations'] == [{'reasons': ['grade'], 'unique_races': 1}]
    coverage['summary']['nonreserved_without_append'] = 3
    with pytest.raises(ValueError, match='denominator'):
        check_attempt_units(coverage, [{'race_id': 'a'}])


@pytest.mark.parametrize('field', ['comparisons', 'dispositions'])
def test_outcome_availability_cannot_reselect_or_erase_passes(field):
    selection = {'comparisons': ['a'], 'dispositions': ['pass']}
    changed = copy.deepcopy(selection)
    changed[field] = []
    with pytest.raises(ValueError, match='selections or pass'):
        check_frozen_selection({}, {}, selection, changed)
    with pytest.raises(ValueError, match='plan changed'):
        check_frozen_selection({'threshold': .7}, {'threshold': .6}, selection, selection)


def test_circular_result_names_cannot_enter_strict_even_with_complete_boxes():
    from scripts.check_live_benchmark_independently import validate_record

    row = {
        'race_id': 'r1', 'date': '2026-07-17', 'model_version': 'v1', 'prediction_id': 'p1',
        'allocation_status': 'AUTHORISED_NONRESERVED', 'forecast_type': 'ORIGINAL_SEALED_LIVE',
        'field_status': 'EXACT_UNCHANGED', 'verified': True,
        'official_runner_identity_observed': False,
        'quote_at': '2026-07-17T00:00:00Z', 'cutoff_at': '2026-07-17T00:01:00Z',
        'predicted_at': '2026-07-17T00:01:00Z', 'sealed_at': '2026-07-17T00:02:00Z',
        'jump_at': '2026-07-17T00:05:00Z', 'result_at': '2026-07-17T00:10:00Z',
        'runners': [{'runner_id': 'a', 'box': 1, 'probability': .7, 'decimal_odds': 2., 'winner': 1},
                    {'runner_id': 'b', 'box': 2, 'probability': .3, 'decimal_odds': 3., 'winner': 0}],
    }
    with pytest.raises(ValueError, match='identity not observed'):
        validate_record(row)
    row['field_status'] = 'RESULT_RUNNER_IDENTITY_UNVERIFIED'
    assert validate_record(row, diagnostic=True)['runners'][0]['model_probability'] == .7
