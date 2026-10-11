import math
import pytest

from race_collection.research_correction_diagnostics import summarize


def race(key, market, form, winner=0):
    return {'key': key, 'date': '2025-09-' + key, 'track': 'A',
            'conditions': {'history': 'missing' if key == '01' else 'supported'},
            'runners': [{'y': int(i == winner), 'probabilities': {'market': m, 'form': f}}
                        for i, (m, f) in enumerate(zip(market, form))]}


def test_exhaustive_unchanged_gains_losses_and_ties():
    races = [race('01', [.6, .4], [.7, .3]), race('02', [.6, .4], [.4, .6]),
             race('03', [.4, .6], [.6, .4]), race('04', [.6, .4], [.5, .5]),
             race('05', [.6, .4], [.6, .4])]
    result = summarize(races, reference='market', model='form')
    assert len(result['paired_races']) == 5
    assert result['overall']['helped'] == 2
    assert result['overall']['hurt'] == 2
    assert result['overall']['equal'] == 1
    assert result['overall']['top_credit_delta'] == -.1
    assert sum(s['races'] for s in result['strata'] if s['dimension'] == 'selection') == 5
    assert len(result['leave_one_date_out']) == 5
    expected = sum(math.log(r['runners'][0]['probabilities']['market'] /
                            r['runners'][0]['probabilities']['form']) for r in races) / 5
    assert result['overall']['log_loss_delta'] == pytest.approx(expected)


def test_rejects_changed_fields_disguised_as_probabilities_and_duplicate_races():
    bad = race('01', [.6, .4], [.6, .3])
    with pytest.raises(ValueError, match='sum to one'):
        summarize([bad], reference='market', model='form')
    good = race('01', [.6, .4], [.7, .3])
    with pytest.raises(ValueError, match='duplicate race'):
        summarize([good, good], reference='market', model='form')


def test_rejects_nonbinary_labels_even_when_they_sum_to_one():
    invalid = race('01', [.5, .3, .2], [.4, .4, .2])
    for runner, label in zip(invalid['runners'], [1, 1, -1]):
        runner['y'] = label
    with pytest.raises(ValueError, match='binary'):
        summarize([invalid], reference='market', model='form')
