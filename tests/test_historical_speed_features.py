"""Fabricated observations exercise the public pure feature interface."""
from copy import deepcopy
import math
import pytest

from race_collection.historical_speed_features import FeatureRejected, build_speed_features


def row(day, value, **updates):
    return {'date': f'2026-09-{day:02d}', 'source_track': 'RAW', 'distance_m': 500,
            'first_sectional': value, **updates}


def packet():
    return {
        'target': {'race_id': 'fictional-race', 'date': '2026-10-06',
                   'source_track': 'RAW', 'distance_m': 500,
                   'cutoff': '2026-10-06T08:00:00Z'},
        'captured_at': '2026-10-06T07:50:00Z',
        'roster': ['dog-a', 'dog-b'],
        'histories': {
            'dog-a': [row(1, '99'), row(15, '5.4'), row(8, '5.0'), row(22, '5.2')],
            'dog-b': [row(8, '5.4'), row(22, '5.8'), row(15, '5.6')],
        },
    }


def test_latest_three_usable_dates_median_mad_and_complete_field_gap():
    result = build_speed_features(packet())
    assert result['full_field_supported'] is True
    assert result['field_median_first_sectional'] == pytest.approx(5.4)
    assert result['supported_runner_count'] == result['runner_count'] == 2
    first, second = result['runners']
    assert first['selected_dates'] == ['2026-09-22', '2026-09-15', '2026-09-08']
    assert first['selected_ages_days'] == [14, 21, 28]
    assert first['median_first_sectional'] == pytest.approx(5.2)
    assert first['mad_first_sectional'] == pytest.approx(0.2)
    assert first['field_median_gap'] == pytest.approx(-0.2)
    assert second['field_median_gap'] == pytest.approx(0.2)
    assert first['usable_prior_dates'] == 4


def test_incomplete_roster_keeps_individual_features_but_no_field_gap():
    supplied = packet()
    del supplied['histories']['dog-b']
    result = build_speed_features(supplied)
    assert result['full_field_supported'] is False
    assert result['supported_runner_count'] == 1
    assert result['field_median_first_sectional'] is None
    assert result['field_blocker'] == 'INCOMPLETE_ROSTER_SUPPORT'
    first, second = result['runners']
    assert first['median_first_sectional'] == pytest.approx(5.2)
    assert first['field_median_gap'] is None
    assert second['status'] == 'NO_RETAINED_HISTORY'
    assert second['median_first_sectional'] is None


def test_missing_invalid_nonprior_and_other_context_rows_are_excluded_before_selection():
    supplied = packet()
    supplied['histories']['dog-a'] += [
        row(28, ''), row(29, 'NaN'), row(30, 'Infinity'),
        row(25, '4.1', source_track='raw'), row(26, '4.2', distance_m=450),
        row(27, '4.3', date='2026-10-06'), row(24, '4.4', date='2026-10-07'),
        row(23, '4.5', date='not-a-date'),
    ]
    first = build_speed_features(supplied)['runners'][0]
    assert first['median_first_sectional'] == pytest.approx(5.2)
    assert first['selected_dates'] == ['2026-09-22', '2026-09-15', '2026-09-08']
    assert first['exclusions'] == {
        'MISSING_FIRST_SECTIONAL': 1, 'INVALID_FIRST_SECTIONAL': 2,
        'CONTEXT_MISMATCH': 2, 'NOT_PRIOR_DATE': 2, 'INVALID_HISTORY_DATE': 1,
    }


@pytest.mark.parametrize('value', [None, '-', '--', 'N/A', 'DNF', 0, -1, True, float('nan'), float('inf')])
def test_three_usable_starts_required_and_unusable_values_remain_explicit(value):
    supplied = packet()
    supplied['histories']['dog-b'][0]['first_sectional'] = value
    second = build_speed_features(supplied)['runners'][1]
    assert second['status'] == 'INSUFFICIENT_COMPARABLE_HISTORY'
    assert second['usable_prior_dates'] == 2
    assert second['median_first_sectional'] is None
    assert sum(second['exclusions'].values()) == 1


def test_identical_repeated_observation_is_one_start():
    supplied = packet()
    supplied['histories']['dog-b'].append(row(22, 5.80))
    second = build_speed_features(supplied)['runners'][1]
    assert second['usable_prior_dates'] == 3
    assert second['median_first_sectional'] == pytest.approx(5.6)
    assert second['exclusions'] == {'DUPLICATE_OBSERVATION': 1}


@pytest.mark.parametrize('change', [
    {'first_sectional': '4.0'}, {'first_sectional': '-'},
    {'source_track': 'OTHER'}, {'distance_m': 450},
])
def test_conflicting_same_date_excludes_all_interpretations(change):
    supplied = packet()
    supplied['histories']['dog-b'].append({**row(22, '5.8'), **change})
    second = build_speed_features(supplied)['runners'][1]
    assert second['status'] == 'INSUFFICIENT_COMPARABLE_HISTORY'
    assert second['usable_prior_dates'] == 2
    assert second['exclusions'] == {'CONFLICTING_PRIOR_DATE': 2}


def test_known_target_layout_mismatch_excludes_without_pooling():
    supplied = packet()
    supplied['target'].update(layout_id='new-course', layout_era='era-2')
    supplied['histories']['dog-b'][0]['layout_id'] = 'old-course'
    result = build_speed_features(supplied)
    assert result['runners'][1]['exclusions'] == {'KNOWN_LAYOUT_MISMATCH': 1}
    assert result['runners'][1]['status'] == 'INSUFFICIENT_COMPARABLE_HISTORY'


@pytest.mark.parametrize('dimension', ['layout_id', 'layout_era'])
def test_unknown_target_does_not_permit_conflicting_known_selected_contexts(dimension):
    supplied = packet()
    supplied['histories']['dog-b'][0][dimension] = 'first'
    supplied['histories']['dog-b'][1][dimension] = 'second'
    second = build_speed_features(supplied)['runners'][1]
    assert second['status'] == 'KNOWN_LAYOUT_CONFLICT'
    assert second['median_first_sectional'] is None


def test_different_known_layouts_across_runners_withhold_only_field_comparison():
    supplied = packet()
    for runner_id, histories in supplied['histories'].items():
        for history in histories:
            history['layout_id'] = runner_id + '-layout'
    result = build_speed_features(supplied)
    assert result['supported_runner_count'] == 2
    assert all(runner['median_first_sectional'] is not None for runner in result['runners'])
    assert all(runner['field_median_gap'] is None for runner in result['runners'])
    assert result['full_field_supported'] is False
    assert result['field_blocker'] == 'KNOWN_LAYOUT_CONFLICT'


@pytest.mark.parametrize('captured,cutoff,reason', [
    ('2026-10-06T08:00:00Z', '2026-10-06T08:00:00Z', 'CAPTURE_NOT_BEFORE_CUTOFF'),
    ('2026-10-06T08:00:01Z', '2026-10-06T08:00:00Z', 'CAPTURE_NOT_BEFORE_CUTOFF'),
    ('2026-10-06T07:00:00', '2026-10-06T08:00:00Z', 'TIMESTAMP_INVALID'),
    ('2026-10-06T07:00:00Z', 'not-a-date', 'TIMESTAMP_INVALID'),
])
def test_capture_cutoff_rejects_shared_input(captured, cutoff, reason):
    supplied = packet()
    supplied['captured_at'], supplied['target']['cutoff'] = captured, cutoff
    with pytest.raises(FeatureRejected, match=f'^{reason}$'):
        build_speed_features(supplied)


def test_offset_capture_and_no_original_publication_timestamp_are_supported():
    supplied = packet()
    supplied['captured_at'] = '2026-10-06T18:50:00+11:00'
    assert build_speed_features(supplied)['full_field_supported'] is True


@pytest.mark.parametrize('defect', ['empty_roster', 'duplicate_runner', 'unknown_runner',
                                    'missing_context', 'unexpected_field', 'invalid_layout'])
def test_shared_schema_and_roster_fail_closed_without_echoing_values(defect):
    supplied = packet()
    if defect == 'empty_roster': supplied['roster'] = []
    elif defect == 'duplicate_runner': supplied['roster'].append('dog-a')
    elif defect == 'unknown_runner': supplied['histories']['private-runner-not-in-roster'] = []
    elif defect == 'missing_context': del supplied['target']['source_track']
    elif defect == 'unexpected_field': supplied['histories']['dog-a'][0]['outcome'] = 'private-value'
    else: supplied['histories']['dog-a'][0]['layout_id'] = {'not': 'a label'}
    with pytest.raises(FeatureRejected, match='^(SCHEMA_INVALID|ROSTER_INVALID)$'):
        build_speed_features(supplied)


def test_row_context_invalidity_is_explicit_not_a_default_context():
    supplied = packet()
    supplied['histories']['dog-b'][0]['distance_m'] = None
    second = build_speed_features(supplied)['runners'][1]
    assert second['exclusions'] == {'INVALID_HISTORY_CONTEXT': 1}
    assert second['median_first_sectional'] is None


def test_finite_inputs_do_not_overflow_even_field_median():
    supplied = packet()
    for histories in supplied['histories'].values():
        for history in histories:
            history['first_sectional'] = 1e308
    result = build_speed_features(supplied)
    assert result['field_median_first_sectional'] == 1e308
    assert all(math.isfinite(runner['field_median_gap']) for runner in result['runners'])


def test_input_order_does_not_select_different_dates_or_mutate_caller():
    supplied = packet()
    original = deepcopy(supplied)
    expected = build_speed_features(supplied)
    assert supplied == original
    for histories in supplied['histories'].values():
        histories.reverse()
    assert build_speed_features(supplied) == expected


def test_historical_dates_cannot_postdate_captured_availability_in_cutoff_timezone():
    supplied = packet()
    supplied['target']['cutoff'] = '2026-10-06T19:00:00+11:00'
    supplied['captured_at'] = '2026-09-22T12:30:00Z'  # September 22 in cutoff timezone.
    supplied['histories']['dog-a'].append(row(23, '4.0'))
    first = build_speed_features(supplied)['runners'][0]
    assert first['exclusions'] == {'HISTORY_AFTER_CAPTURE_DATE': 1}
    assert first['selected_dates'][0] == '2026-09-22'
    supplied['target']['date'] = '2026-10-07'
    with pytest.raises(FeatureRejected, match='^TARGET_DATE_AFTER_CUTOFF$'):
        build_speed_features(supplied)


def test_earlier_meeting_date_is_allowed_after_midnight_cutoff():
    supplied = packet()
    supplied['target']['cutoff'] = '2026-10-07T00:01:00+11:00'
    supplied['captured_at'] = '2026-10-06T23:50:00+11:00'
    assert build_speed_features(supplied)['full_field_supported'] is True


def test_retained_whitelist_fingerprint_preserves_conflicts_in_other_source_fields():
    supplied = packet()
    supplied['histories']['dog-b'][1]['observation_fingerprint'] = 'a' * 64
    supplied['histories']['dog-b'].append(row(22, '5.8', observation_fingerprint='b' * 64))
    second = build_speed_features(supplied)['runners'][1]
    assert second['status'] == 'INSUFFICIENT_COMPARABLE_HISTORY'
    assert second['exclusions'] == {'CONFLICTING_PRIOR_DATE': 2}
