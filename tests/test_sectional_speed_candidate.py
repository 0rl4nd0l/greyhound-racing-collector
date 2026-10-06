"""Fabricated tests only: the public pure seam owns the calculation contract."""
from copy import deepcopy
import hashlib
import json
import math

import pytest

from race_collection.sectional_speed_candidate import CandidateRejected, build_sectional_candidate


def observation(identity='dog:a', day='2026-09-20', value=5.0, track='A', distance=500,
                available='2026-10-02T18:00:00+10:00', **extra):
    row = {
        'observation_id': f'{identity}:{day}:{track}:{distance}',
        'event_id': f'{day}:{track}:{distance}', 'runner_identity_id': identity,
        'date': day, 'available_at': available, 'source_track': track,
        'canonical_track': track, 'distance_m': distance, 'first_sectional': value,
        'observation_fingerprint': hashlib.sha256(str(value).encode()).hexdigest(),
        'source_bindings': [{'csv_sha256': 'a' * 64, 'row': f'{identity}:{day}'}],
        'event_identity_kind': 'RUNNER_DATE_CONTEXT_PROXY',
    }
    row.update(extra)
    return row


def packet(own=None, others=None):
    return {
        'target': {'race_id': 'race:target', 'date': '2026-10-03', 'cutoff': '2026-10-03T18:00:00+10:00'},
        'roster': [roster('dog:a'), roster('dog:missing')],
        'observations': ([observation()] if own is None else own) + (
            [observation(f'dog:{i}', value=float(4 + i)) for i in range(5)] if others is None else others),
    }


def roster(identity):
    return {'runner_id': f'slot:{identity}', 'identity_id': identity,
            'identity_available_at': '2026-10-02T18:00:00+10:00'}


def first(built):
    return built['runners'][0]


def test_exact_independent_small_example_and_incomplete_field_retained():
    built = build_sectional_candidate(packet())
    actual = first(built)
    assert built['supported_runner_count'] == 1
    assert actual['speed_estimate'] == pytest.approx((6 - 5) / 1.4826 / 4)
    assert actual['shrinkage_weight'] == 0.25
    assert actual['supported_history_count'] == 1
    assert built['runners'][1]['speed_estimate'] == 0
    bench = built['benchmarks'][0]
    assert bench['centre'] == 6
    assert bench['mad'] == 1
    assert bench['spread'] == 1.4826
    assert len(bench['members']) == 5
    assert all(row['runner_identity_id'] != 'dog:a' for row in bench['members'])
    assert actual['selected_observations'][0]['source_bindings']
    assert actual['selected_observations'][0]['benchmark_id'] == bench['benchmark_id']


def test_two_histories_receive_support_and_count_shrinkage():
    own = [observation(value=4), observation(day='2026-09-21', value=5)]
    actual = first(build_sectional_candidate(packet(own=own)))
    assert actual['supported_history_count'] == 2
    assert actual['shrinkage_weight'] == 0.4
    assert actual['speed_estimate'] == pytest.approx(1.5 / 1.4826 * 0.4)


def test_latest_five_supported_histories_selected():
    own = [observation(day=f'2026-09-{day}', value=5) for day in range(10, 17)]
    actual = first(build_sectional_candidate(packet(own=own)))
    assert actual['usable_history_count'] == 7
    assert actual['supported_history_count'] == 5
    assert [row['date'] for row in actual['selected_observations']] == [f'2026-09-{day}' for day in range(16, 11, -1)]
    assert actual['shrinkage_weight'] == 5 / 8


def test_cross_context_transfer_normalizes_each_context_separately():
    own = [observation(value=5), observation(day='2026-09-21', value=10, track='B', distance=600)]
    others = [observation(f'dog:{i}', value=float(4 + i)) for i in range(5)]
    others += [observation(f'dog:{i}', day='2026-09-21', value=float(8 + i * 2), track='B', distance=600) for i in range(5)]
    built = build_sectional_candidate(packet(own=own, others=others))
    assert first(built)['speed_estimate'] == pytest.approx(1 / 1.4826 * 0.4)
    assert len(built['benchmarks']) == 2
    assert 'CROSS_CONTEXT_STANDARDIZED_SECTIONAL_TRANSFER_EXPLORATORY' in built['assumptions']


def test_benchmarks_balance_runner_contributions_not_observation_counts():
    original = packet()
    expected = first(build_sectional_candidate(original))['speed_estimate']
    original['observations'].extend(observation('dog:0', day=f'2026-09-{day:02d}', value=4) for day in range(1, 20))
    built = build_sectional_candidate(original)
    assert first(built)['speed_estimate'] == expected
    assert built['benchmarks'][0]['other_runner_count'] == 5
    assert built['benchmarks'][0]['observation_count'] == 24


def test_own_history_never_enters_own_benchmark():
    own = [observation(day=f'2026-09-{day}', value=100) for day in range(10, 17)]
    built = build_sectional_candidate(packet(own=own))
    assert built['benchmarks'][0]['centre'] == 6
    assert all(member['runner_identity_id'] != 'dog:a' for member in built['benchmarks'][0]['members'])
    assert first(built)['speed_estimate'] == -3 * 5 / 8


def test_exact_copies_do_not_increase_support_or_benchmark_weight():
    data = packet()
    expected = build_sectional_candidate(data)
    duplicate = deepcopy(data['observations'][0])
    duplicate['observation_id'] = 'second-card-copy'
    duplicate['source_bindings'] = [{'csv_sha256': 'b' * 64, 'row': 2}]
    data['observations'].append(duplicate)
    built = build_sectional_candidate(data)
    assert first(built)['speed_estimate'] == first(expected)['speed_estimate']
    assert first(built)['supported_history_count'] == 1
    assert len(first(built)['selected_observations'][0]['source_bindings']) == 2
    assert built['population_exclusions']['DUPLICATE_COPY'] == 1


@pytest.mark.parametrize('change', [
    {'first_sectional': 5.1}, {'observation_fingerprint': 'c' * 64},
    {'canonical_track': 'B', 'alias_evidence': 'verified-retained-ref'},
    {'event_id': 'another-event'}, {'layout_id': 'layout:B'},
])
def test_conflicting_same_runner_date_is_excluded(change):
    data = packet()
    duplicate = {**data['observations'][0], **change}
    data['observations'].append(duplicate)
    built = build_sectional_candidate(data)
    assert first(built)['speed_estimate'] == 0
    assert built['population_exclusions']['CONFLICTING_RUNNER_DATE'] == 2


def test_same_event_cannot_appear_as_two_dates():
    data = packet()
    data['observations'].append({**data['observations'][0], 'date': '2026-09-21'})
    built = build_sectional_candidate(data)
    assert first(built)['speed_estimate'] == 0
    assert built['population_exclusions']['CONFLICTING_EVENT_DATES'] == 2


@pytest.mark.parametrize('available', ['2026-10-03T18:00:00+10:00', '2026-10-04T01:00:00+10:00'])
def test_future_conflicting_copy_cannot_poison_earlier_features(available):
    data = packet()
    expected = first(build_sectional_candidate(data))['speed_estimate']
    data['observations'].append(observation(value=100, available=available))
    built = build_sectional_candidate(data)
    assert first(built)['speed_estimate'] == expected
    assert built['population_exclusions']['EVIDENCE_NOT_BEFORE_CUTOFF'] == 1


@pytest.mark.parametrize('day', ['2026-10-03', '2026-10-04'])
def test_same_day_and_future_history_are_excluded(day):
    built = build_sectional_candidate(packet(own=[observation(day=day)]))
    assert first(built)['speed_estimate'] == 0
    assert built['population_exclusions']['NOT_PRIOR_DATE'] == 1


def test_history_after_evidence_date_is_excluded():
    built = build_sectional_candidate(packet(own=[observation(day='2026-10-02', available='2026-10-01T18:00:00+10:00')]))
    assert first(built)['speed_estimate'] == 0
    assert built['population_exclusions']['HISTORY_AFTER_EVIDENCE_DATE'] == 1


def test_availability_uses_timezone_aware_instant_comparison():
    own = [observation(available='2026-10-03T07:59:59+00:00')]
    assert first(build_sectional_candidate(packet(own=own)))['status'] == 'SUPPORTED'
    own[0]['available_at'] = '2026-10-03T08:00:00+00:00'
    assert first(build_sectional_candidate(packet(own=own)))['status'] == 'NO_SUPPORTED_HISTORIES'


def test_benchmark_update_cannot_use_future_evidence():
    data = packet(others=[observation(f'dog:{i}', value=float(4 + i)) for i in range(4)])
    data['observations'].append(observation('dog:4', value=8, available='2026-10-04T00:00:00+10:00'))
    built = build_sectional_candidate(data)
    assert first(built)['speed_estimate'] == 0
    assert built['benchmarks'][0]['other_runner_count'] == 4


@pytest.mark.parametrize('value,reason', [
    ('-', 'MISSING_FIRST_SECTIONAL'), ('', 'MISSING_FIRST_SECTIONAL'),
    (None, 'MISSING_FIRST_SECTIONAL'), (0, 'INVALID_FIRST_SECTIONAL'),
    (-1, 'INVALID_FIRST_SECTIONAL'), (True, 'INVALID_FIRST_SECTIONAL'),
    (float('nan'), 'INVALID_FIRST_SECTIONAL'), (float('inf'), 'INVALID_FIRST_SECTIONAL'),
    ('nonsense', 'INVALID_FIRST_SECTIONAL'),
])
def test_missing_and_invalid_are_not_slow(value, reason):
    built = build_sectional_candidate(packet(own=[observation(value=value)]))
    assert first(built)['speed_estimate'] == 0
    assert first(built)['supported_history_count'] == 0
    assert built['population_exclusions'][reason] == 1
    json.dumps(built, allow_nan=False)


def test_zero_mad_has_no_unprespecified_fallback():
    others = [observation(f'dog:{i}', value=5) for i in range(5)]
    built = build_sectional_candidate(packet(others=others))
    assert first(built)['speed_estimate'] == 0
    assert built['benchmarks'][0]['status'] == 'ZERO_OR_NONFINITE_SPREAD'


def test_known_unknown_layouts_stay_separate():
    data = packet()
    data['observations'][0]['layout_id'] = 'new-layout'
    built = build_sectional_candidate(data)
    assert first(built)['speed_estimate'] == 0
    assert built['benchmarks'][0]['other_runner_count'] == 0


def test_verified_alias_can_contribute_but_unverified_alias_rejected():
    data = packet()
    data['observations'][0]['source_track'] = 'Long Track A'
    with pytest.raises(CandidateRejected, match='UNVERIFIED_ALIAS'):
        build_sectional_candidate(data)
    data['observations'][0]['alias_evidence'] = 'retained-same-event-binding'
    built = build_sectional_candidate(data)
    assert first(built)['status'] == 'SUPPORTED'
    assert first(built)['selected_observations'][0]['alias_evidence'] == ['retained-same-event-binding']


def test_missing_identity_never_joins_by_name():
    data = packet()
    data['roster'][0].update(identity_id=None, identity_available_at=None)
    actual = first(build_sectional_candidate(data))
    assert actual['status'] == 'NO_VERIFIED_IDENTITY'
    assert actual['speed_estimate'] == 0


def test_identity_discovered_later_does_not_backdate_join():
    data = packet()
    data['roster'][0]['identity_available_at'] = data['target']['cutoff']
    actual = first(build_sectional_candidate(data))
    assert actual['status'] == 'IDENTITY_NOT_BEFORE_CUTOFF'
    assert actual['speed_estimate'] == 0


@pytest.mark.parametrize('value', [0.00001, 1e200])
def test_extreme_sectionals_clipped_before_shrinkage(value):
    built = build_sectional_candidate(packet(own=[observation(value=value)]))
    assert math.isfinite(first(built)['speed_estimate'])
    assert abs(first(built)['speed_estimate']) <= 3 / 4


def test_all_unsupported_is_baseline_only():
    built = build_sectional_candidate(packet(own=[], others=[]))
    assert built['status'] == 'BASELINE_ONLY'
    assert all(row['speed_estimate'] == 0 for row in built['runners'])


def test_input_is_unchanged_and_output_order_deterministic():
    data = packet()
    saved = deepcopy(data)
    one = build_sectional_candidate(data)
    assert data == saved
    data['observations'].reverse()
    assert build_sectional_candidate(data) == one


@pytest.mark.parametrize('mutation,reason', [
    (lambda data: data['roster'].append(deepcopy(data['roster'][0])), 'DUPLICATE_ROSTER_RUNNER'),
    (lambda data: data['roster'][1].update(identity_id='dog:a'), 'DUPLICATE_ROSTER_IDENTITY'),
    (lambda data: data['target'].update(cutoff='2026-10-03T18:00:00'), 'TIMESTAMP_INVALID'),
    (lambda data: data['observations'][0].update(layout_id=[]), 'LAYOUT_INVALID'),
    (lambda data: data['observations'][0].update(source_bindings=[]), 'SOURCE_BINDINGS_INVALID'),
])
def test_shared_identity_or_schema_errors_fail_closed(mutation, reason):
    data = packet()
    mutation(data)
    with pytest.raises(CandidateRejected, match=reason):
        build_sectional_candidate(data)
