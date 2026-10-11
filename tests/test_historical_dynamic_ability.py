from copy import deepcopy

import pytest

from race_collection.historical_dynamic_ability import representations


def race(day, key, order, context=('A', 500)):
    return {'date': day, 'key': key, 'context': context,
            'runners': [{'box': i, 'native_id': dog, 'finish': finish}
                        for i, (dog, finish) in enumerate(order, 1)]}


def test_future_results_and_identities_cannot_change_earlier_predictions():
    early = [race('2025-01-01', 'a', [('x', 1), ('y', 2)]),
             race('2025-01-03', 'b', [('x', 2), ('y', 1)])]
    later = race('2025-01-04', 'c', [('x', 2), ('z', 1)])
    before = representations(early)[0]
    after = representations(early + [later])[0]
    for arm in before:
        assert before[arm] == {k: after[arm][k] for k in before[arm]}


def test_same_date_results_do_not_update_another_race():
    events = [race('2025-01-01', 'a', [('x', 1), ('y', 2)]),
              race('2025-01-01', 'b', [('x', 2), ('z', 1)])]
    values, audit, _ = representations(events)
    assert values['dynamic'][('b', 1)]['ability'] is None
    assert audit['update_excluded_identity_races'] == 2


def test_opposition_strength_changes_dynamic_but_not_recency_performance():
    events = [race('2025-01-01', 'a', [('strong', 1), ('weak', 2)]),
              race('2025-01-02', 'b', [('x', 1), ('strong', 2)]),
              race('2025-01-02', 'c', [('y', 1), ('weak', 2)]),
              race('2025-01-03', 'd', [('x', 1), ('y', 2)])]
    values, _, _ = representations(events)
    assert values['recency'][('d', 1)]['ability'] == values['recency'][('d', 2)]['ability']
    assert values['dynamic'][('d', 1)]['ability'] > values['dynamic'][('d', 2)]['ability']


def test_sparse_support_is_shrunk_and_missing_is_distinct():
    values, _, _ = representations([
        race('2025-01-01', 'a', [('x', 1), ('y', 2)]),
        race('2025-02-01', 'b', [('x', 1), ('new', 2)])])
    experienced = values['dynamic'][('b', 1)]
    sparse = values['dynamic'][('b', 2)]
    assert 0 < experienced['ability'] < .25
    assert 0 < experienced['support'] < 1
    assert experienced['uncertainty'] > 0
    assert sparse['ability'] is None and sparse['support'] == 0


def test_target_labels_and_unknown_context_do_not_enter_current_features():
    events = [race('2025-01-01', 'a', [('x', 1), ('y', 2)]),
              race('2025-01-02', 'b', [('x', 1), ('y', 2)], None)]
    changed = deepcopy(events)
    for runner in changed[-1]['runners']:
        runner['finish'] = 3 - runner['finish']
    a, b = representations(events)[0], representations(changed)[0]
    assert a == b
    assert a['dynamic'][('b', 1)]['context_residual'] is None
    assert a['dynamic'][('b', 1)]['context_support'] is None


def test_incomplete_order_is_rejected():
    with pytest.raises(ValueError, match='COMPLETE_UNIQUE'):
        representations([race('2025-01-01', 'a', [('x', 1), ('y', 1)])])


def test_same_date_race_input_order_invariant():
    events = [race('2025-01-01', 'a', [('x', 1), ('y', 2)]),
              race('2025-01-01', 'b', [('z', 2), ('q', 1)]),
              race('2025-01-02', 'c', [('x', 1), ('z', 2)])]
    assert representations(events)[0] == representations(events[::-1])[0]


def test_future_identity_resolution_never_backfills_unqualified_prior_event():
    events = [race('2025-01-01', 'a', [(None, 1), ('y', 2)]),
              race('2025-01-02', 'b', [('x', 1), ('y', 2)])]
    values, audit, _ = representations(events)
    assert values['dynamic'][('b', 1)]['support'] == 0
    assert values['dynamic'][('b', 2)]['support'] == 0
    assert audit['update_excluded_identity_races'] == 1


def test_context_residual_does_not_transfer_to_other_track_distance():
    events = [race('2025-01-01', 'a', [('x', 1), ('y', 2)]),
              race('2025-01-02', 'b', [('x', 1), ('y', 2)], ('B', 500))]
    values = representations(events)[0]['dynamic'][('b', 1)]
    assert values['ability'] > 0
    assert values['context_residual'] == 0
    assert values['context_support'] == 0
