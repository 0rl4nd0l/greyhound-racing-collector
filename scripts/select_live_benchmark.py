#!/usr/bin/env python3
"""Frozen, outcome-blind selectors; exploratory offline comparison only."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from score_live_benchmark import benchmark, json_safe, score

COVERAGES = (.1, .25, .5, 1.)
SELECTORS = ('model_confidence', 'relevant_history', 'price_disagreement')


def selector_value(record, name):
    if name == 'model_confidence':
        return max(r['probability'] for r in record['runners'])
    if name == 'price_disagreement':
        return max(r['probability'] * r['decimal_odds'] - 1 for r in record['runners'])
    value = record.get('pre_race_quality', {}).get('minimum_starts_same_distance')
    if value is not None and (not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0):
        raise ValueError('invalid_sealed_history_count')
    return value


def freeze(records):
    """Uses dates and predictors only. Never inspect winner outcomes here."""
    plans = {}
    for version in sorted({r['model_version'] for r in records}):
        subset = [r for r in records if r['model_version'] == version]
        dates = sorted({r['date'] for r in subset})
        if len(dates) < 2:
            plans[version] = {'status': 'UNAVAILABLE_FEWER_THAN_TWO_DATES', 'dates': dates}
            continue
        development = dates[:max(1, len(dates)//2)]
        evaluation = dates[len(development):]
        rules = []
        for selector in SELECTORS:
            values = [selector_value(r, selector) for r in subset if r['date'] in development]
            values = sorted(v for v in values if v is not None)
            for coverage in COVERAGES:
                threshold = None if coverage == 1 or not values else values[max(0, math.ceil((1-coverage)*len(values))-1)]
                rules.append(dict(selector=selector, target_coverage=coverage, threshold=threshold,
                                  status='AVAILABLE' if values or coverage == 1 else 'UNAVAILABLE_SEALED_QUALITY'))
        plans[version] = dict(status='FROZEN', development_dates=development, evaluation_dates=evaluation,
                             later_exposure='EXPLORATORY_UNTOUCHED_STATUS_NOT_ESTABLISHED', rules=rules)
    return dict(schema='live_selector_plan_v1', plans=plans)


def evaluate(records, plan, diagnostic=False):
    # Recompute outcome-free plan to detect edited scores/dates/thresholds.
    if freeze(records) != plan:
        raise ValueError('frozen_selector_plan_mismatch')
    comparisons, dispositions = [], []
    for version, vp in plan['plans'].items():
        subset = [r for r in records if r['model_version'] == version]
        if vp['status'] != 'FROZEN':
            dispositions.extend(dict(race_id=r['race_id'], model_version=version, disposition=vp['status']) for r in subset)
            continue
        later = [r for r in subset if r['date'] in vp['evaluation_dates']]
        for r in subset:
            if r['date'] in vp['development_dates']:
                dispositions.append(dict(race_id=r['race_id'], model_version=version, disposition='DEVELOPMENT_ONLY'))
        market_ranked = sorted(later, key=lambda r: (-max(score(r, diagnostic=diagnostic)['market_probabilities']), r['race_id']))
        for rule in vp['rules']:
            selected = []
            for r in later:
                value = selector_value(r, rule['selector'])
                if rule['target_coverage'] == 1:
                    disposition = 'SELECTED'
                elif rule['status'] != 'AVAILABLE' or value is None:
                    disposition = 'PASS_MISSING_SEALED_QUALITY'
                else:
                    disposition = 'SELECTED' if value >= rule['threshold'] else 'PASS'
                dispositions.append(dict(race_id=r['race_id'], model_version=version,
                                         selector=rule['selector'], target_coverage=rule['target_coverage'],
                                         value=value, threshold=rule['threshold'], disposition=disposition))
                if disposition == 'SELECTED':
                    selected.append(r)
            matched = market_ranked[:len(selected)]
            comparisons.append(dict(model_version=version, **rule,
                                    evaluation_races=len(later), selected_races=len(selected),
                                    achieved_coverage=len(selected)/len(later) if later else None,
                                    selected_race_ids=[r['race_id'] for r in selected],
                                    market_confidence_race_ids=[r['race_id'] for r in matched],
                                    selected=benchmark(selected, diagnostic=diagnostic), market_confidence=benchmark(matched, diagnostic=diagnostic)))
    return dict(schema='live_selector_results_v1', status='EXPLORATORY' if comparisons else 'NO_EVALUABLE_SELECTORS',
                comparisons=comparisons, dispositions=dispositions,
                limitation='Later untouched status not established. Batch market ranking is diagnostic, not an operating rule.')


def census(records, plan):
    """Preserve selections for every forecast, including missing-result races."""
    if freeze(records) != plan:
        raise ValueError('frozen_selector_plan_mismatch')
    dispositions, summaries = [], []
    for version, vp in plan['plans'].items():
        subset = [r for r in records if r['model_version'] == version]
        if vp['status'] != 'FROZEN':
            dispositions.extend(dict(race_id=r['race_id'], model_version=version, disposition=vp['status']) for r in subset)
            continue
        earlier = [r for r in subset if r['date'] in vp['development_dates']]
        later = [r for r in subset if r['date'] in vp['evaluation_dates']]
        dispositions.extend(dict(race_id=r['race_id'], model_version=version, disposition='DEVELOPMENT_ONLY') for r in earlier)
        def market_confidence(r):
            inverse = [1/runner['decimal_odds'] for runner in r['runners']]
            return max(inverse)/sum(inverse)
        ranked = sorted(later, key=lambda r: (-market_confidence(r), r['race_id']))
        for rule in vp['rules']:
            selected = []
            for r in later:
                value = selector_value(r, rule['selector'])
                if rule['target_coverage'] == 1:
                    state = 'SELECTED'
                elif rule['status'] != 'AVAILABLE' or value is None:
                    state = 'PASS_MISSING_SEALED_QUALITY'
                else:
                    state = 'SELECTED' if value >= rule['threshold'] else 'PASS'
                dispositions.append(dict(race_id=r['race_id'], model_version=version,
                                         selector=rule['selector'], target_coverage=rule['target_coverage'],
                                         value=value, threshold=rule['threshold'], disposition=state))
                if state == 'SELECTED':
                    selected.append(r['race_id'])
            summaries.append(dict(model_version=version, **rule, evaluation_races=len(later),
                                  selected_races=len(selected), achieved_coverage=len(selected)/len(later),
                                  selected_race_ids=selected, market_confidence_race_ids=[r['race_id'] for r in ranked[:len(selected)]]))
    return dict(schema='live_selector_forecast_census_v1', status='FORECAST_SELECTION_ONLY_NO_OUTCOMES',
                plans=plan['plans'], comparisons=summaries, dispositions=dispositions)


def score_forecast_selection(selection, outcome_records, diagnostic=False):
    """Apply result-independent selections without replacing missing outcomes."""
    lookup = {(r['race_id'], r['model_version']): r for r in outcome_records}
    comparisons = []
    for item in selection['comparisons']:
        version = item['model_version']
        selected = [lookup[(race, version)] for race in item['selected_race_ids'] if (race, version) in lookup]
        market = [lookup[(race, version)] for race in item['market_confidence_race_ids'] if (race, version) in lookup]
        comparisons.append(dict(item, selected_outcomes=len(selected), market_selector_outcomes=len(market),
                                selected_missing_outcomes=len(item['selected_race_ids'])-len(selected),
                                market_selector_missing_outcomes=len(item['market_confidence_race_ids'])-len(market),
                                selected=benchmark(selected, diagnostic=diagnostic),
                                market_confidence=benchmark(market, diagnostic=diagnostic)))
    return dict(schema='live_selector_result_independent_evaluation_v1',
                status='FROZEN_FORECAST_POPULATION_SELECTIONS', comparisons=comparisons,
                limitation='Selections and matched forecast coverage remain fixed when results are unavailable; available outcome counts can differ.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'evaluate', 'census'))
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = args.dataset.read_bytes()
    payload = json.loads(data)
    records = payload['records']
    diagnostic = payload.get('analysis_class') == 'RESULT_FIELD_UNVERIFIED_DIAGNOSTIC'
    digest = hashlib.sha256(data).hexdigest()
    if args.mode == 'freeze':
        plan = freeze(records)
        plan['dataset_sha256'] = digest
        args.plan.write_text(json.dumps(plan, indent=2, sort_keys=True)+'\n')
    else:
        plan = json.loads(args.plan.read_text())
        if plan.pop('dataset_sha256') != digest:
            raise ValueError('selector_dataset_hash_mismatch')
        result = census(records, plan) if args.mode == 'census' else evaluate(records, plan, diagnostic=diagnostic)
        result['analysis_class'] = payload.get('analysis_class', 'STRICT_DECISION_TIME')
        result['dataset_sha256'] = digest
        result['plan_sha256'] = hashlib.sha256(args.plan.read_bytes()).hexdigest()
        if args.output is None:
            parser.error('--output required for evaluate')
        args.output.write_text(json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
