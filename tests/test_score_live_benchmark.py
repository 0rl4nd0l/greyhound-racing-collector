import copy
import importlib.util
import math
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('live_score', Path(__file__).parents[1]/'scripts/score_live_benchmark.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def record():
    return dict(race_id='native:1', date='2026-07-17', model_version='sha256:one', prediction_id='sealed1',
                allocation_status='AUTHORISED_NONRESERVED', forecast_type='ORIGINAL_SEALED_LIVE',
                field_status='EXACT_UNCHANGED', verified=True, quote_at='2026-07-17T01:00:00+00:00',
                cutoff_at='2026-07-17T01:00:01+00:00', predicted_at='2026-07-17T01:00:02+00:00',
                sealed_at='2026-07-17T01:00:03+00:00', jump_at='2026-07-17T01:02:00+00:00',
                result_at='2026-07-17T01:03:00+00:00',
                runners=[dict(runner_id='a', box=1, probability=.8, decimal_odds=2., winner=1),
                         dict(runner_id='b', box=2, probability=.2, decimal_odds=2., winner=0)])


def test_hand_calculated_scores_and_empty_population():
    result = m.score(record())
    assert result['model_log_loss'] == pytest.approx(-math.log(.8))
    assert result['market_log_loss'] == pytest.approx(math.log(2))
    assert result['model_brier'] == pytest.approx(.08)
    assert result['market_brier'] == pytest.approx(.5)
    assert result['model_top_credit'] == 1
    assert result['market_top_credit'] == .5
    assert m.benchmark([])['status'] == 'NO_ELIGIBLE_DECISION_TIME_COMPARISONS'


@pytest.mark.parametrize('key,value,error', [
    ('allocation_status', 'RESERVED', 'scientific_allocation'),
    ('forecast_type', 'REPLAY', 'not_original'),
    ('field_status', 'SCRATCHED', 'changed_or_unresolved'),
    ('verified', False, 'bundle_not'),
    ('sealed_at', '2026-07-17T01:02:00+00:00', 'chronology'),
    ('quote_at', '2026-07-17T01:01:00+00:00', 'chronology'),
    ('cutoff_at', '2026-07-17T01:00:01', 'naive'),
    ('result_at', '2026-07-17T01:00:00+00:00', 'result_before'),
])
def test_reject_metadata_failures(key, value, error):
    r = record()
    r[key] = value
    with pytest.raises(ValueError, match=error):
        m.score(r)


@pytest.mark.parametrize('key,value,error', [
    ('runner_id', 'a', 'duplicate_or_incomplete'), ('box', 1, 'duplicate_box'),
    ('probability', float('nan'), 'invalid_probability'),
    ('probability', .5, 'sum_to_one'), ('decimal_odds', 1., 'invalid_win_odds'),
    ('winner', 1, 'ambiguous'), ('winner', .5, 'dead_heat'),
])
def test_reject_runner_failures(key, value, error):
    r = record()
    r['runners'][1][key] = value
    with pytest.raises(ValueError, match=error):
        m.score(r)


def test_zero_winner_probability_is_infinite_not_clipped():
    r = record()
    r['runners'][0]['probability'] = 0
    r['runners'][1]['probability'] = 1
    assert m.score(r)['model_log_loss'] == math.inf
    assert m.json_safe(m.score(r))['model_log_loss'] == 'Infinity'


def test_no_duplicate_primary_and_date_uncertainty():
    r = record()
    with pytest.raises(ValueError, match='multiple_primary'):
        m.benchmark([r, r])
    result = m.benchmark([r])['by_version']['sha256:one']
    assert result['uncertainty']['intervals'] is None
    r2 = copy.deepcopy(r)
    r2['race_id'] = 'native:2'
    r2['date'] = '2026-07-18'
    intervals = m.benchmark([r, r2])['by_version']['sha256:one']['uncertainty']['intervals']
    assert intervals['delta_log_loss'] == pytest.approx([math.log(.625)]*2)


def test_partial_field_requires_explicit_diagnostic_mode():
    r = record()
    r['field_status'] = 'RESULT_FIELD_PARTIAL'
    with pytest.raises(ValueError, match='changed_or_unresolved'):
        m.benchmark([r])
    result = m.benchmark([r], diagnostic=True)
    assert result['analysis_class'] == 'RESULT_FIELD_UNVERIFIED_DIAGNOSTIC'
    r['field_status'] = 'KNOWN_SCRATCH'
    with pytest.raises(ValueError, match='changed_or_unresolved'):
        m.benchmark([r], diagnostic=True)
