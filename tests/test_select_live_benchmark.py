import copy
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from select_live_benchmark import evaluate, freeze
spec = importlib.util.spec_from_file_location("score_fixture", Path(__file__).with_name("test_score_live_benchmark.py"))
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
record = fixture.record


def records():
    values = []
    for day in range(2):
        for i in range(4):
            r = copy.deepcopy(record())
            r['race_id'] = f'{day}:{i}'
            r['date'] = f'2026-07-{17+day}'
            r['runners'][0]['probability'] = .55+.1*i
            r['runners'][1]['probability'] = 1-r['runners'][0]['probability']
            values.append(r)
    return values


def test_thresholds_are_outcome_blind_and_passes_preserved():
    rows = records()
    plan = freeze(rows)
    changed = copy.deepcopy(rows)
    for r in changed:
        for runner in r['runners']:
            runner['winner'] = 1-runner['winner']
    assert freeze(changed) == plan
    result = evaluate(rows, plan)
    for comparison in result['comparisons']:
        assert len(comparison['selected_race_ids']) == len(comparison['market_confidence_race_ids'])
    assert len(result['dispositions']) == 4+4*12
    missing = [r for r in result['comparisons'] if r['selector'] == 'relevant_history' and r['target_coverage'] < 1]
    assert all(r['selected_races'] == 0 for r in missing)
    full = [r for r in result['comparisons'] if r['target_coverage'] == 1]
    assert all(r['selected_races'] == 4 for r in full)


def test_changed_threshold_rejected_and_single_date_unavailable():
    rows = records()
    plan = freeze(rows)
    plan['plans']['sha256:one']['rules'][0]['threshold'] = 0
    with pytest.raises(ValueError, match='plan_mismatch'):
        evaluate(rows, plan)
    assert evaluate(rows[:4], freeze(rows[:4]))['status'] == 'NO_EVALUABLE_SELECTORS'


def test_primary_selections_do_not_change_when_results_are_missing():
    from select_live_benchmark import census, score_forecast_selection
    rows = records()
    plan = freeze(rows)
    decisions = census(rows, plan)
    result = score_forecast_selection(decisions, rows[:4])
    assert all(c['selected_outcomes'] == 0 for c in result['comparisons'])
    assert result['comparisons'][0]['selected_race_ids'] == decisions['comparisons'][0]['selected_race_ids']
    assert result['comparisons'][0]['selected_missing_outcomes'] == decisions['comparisons'][0]['selected_races']
