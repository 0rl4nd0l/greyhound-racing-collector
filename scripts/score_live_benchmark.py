#!/usr/bin/env python3
"""Offline, fit-free scores for a manifest-authorised exact-field live benchmark."""
import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def stamp(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('naive_timestamp')
    return parsed


def validate(record, diagnostic=False):
    """Fail closed; the adapter must authenticate source bytes before this stage."""
    if record['allocation_status'] != 'AUTHORISED_NONRESERVED':
        raise ValueError('scientific_allocation_exclusion')
    if record['forecast_type'] != 'ORIGINAL_SEALED_LIVE':
        raise ValueError('not_original_live_forecast')
    if record['verified'] is not True:
        raise ValueError('bundle_not_verified')
    if record['field_status'] != 'EXACT_UNCHANGED' and not (diagnostic and record['field_status'] == 'RESULT_FIELD_PARTIAL'):
        raise ValueError('changed_or_unresolved_field')
    times = [stamp(record[k]) for k in ('quote_at', 'cutoff_at', 'predicted_at', 'sealed_at', 'jump_at')]
    if not all(a <= b for a, b in zip(times, times[1:])) or times[-2] >= times[-1]:
        raise ValueError('invalid_prejump_chronology')
    if stamp(record['result_at']) < times[-1]:
        raise ValueError('result_before_jump')
    runners = record['runners']
    if len(runners) < 2 or len({r['runner_id'] for r in runners}) != len(runners):
        raise ValueError('duplicate_or_incomplete_runner_identity')
    if len({r['box'] for r in runners}) != len(runners):
        raise ValueError('duplicate_box')
    for runner in runners:
        p, odds, y = runner['probability'], runner['decimal_odds'], runner['winner']
        if isinstance(p, bool) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('invalid_probability')
        if isinstance(odds, bool) or not math.isfinite(odds) or odds <= 1:
            raise ValueError('invalid_win_odds')
        if y not in (0, 1):
            raise ValueError('ambiguous_or_dead_heat_result')
    if not math.isclose(sum(r['probability'] for r in runners), 1, rel_tol=0, abs_tol=1e-6):
        raise ValueError('probabilities_do_not_sum_to_one')
    if sum(r['winner'] for r in runners) != 1:
        raise ValueError('ambiguous_or_dead_heat_result')


def top_credit(probabilities, targets):
    top = max(probabilities)
    indexes = [i for i, p in enumerate(probabilities) if p == top]
    return sum(targets[i] for i in indexes) / len(indexes), indexes


def score(record, diagnostic=False):
    validate(record, diagnostic=diagnostic)
    rows = record['runners']
    p = [r['probability'] for r in rows]
    inverse = [1 / r['decimal_odds'] for r in rows]
    overround = sum(inverse)
    q = [v / overround for v in inverse]
    y = [r['winner'] for r in rows]
    winner = y.index(1)
    pc, pt = top_credit(p, y)
    qc, qt = top_credit(q, y)
    pll = -math.log(p[winner]) if p[winner] else math.inf
    qll = -math.log(q[winner])
    pb = sum((a - b) ** 2 for a, b in zip(p, y))
    qb = sum((a - b) ** 2 for a, b in zip(q, y))
    return dict(race_id=record['race_id'], date=record['date'], prediction_id=record['prediction_id'],
                model_version=record['model_version'], runners=len(rows), model_log_loss=pll,
                market_log_loss=qll, delta_log_loss=pll-qll, model_brier=pb, market_brier=qb,
                delta_brier=pb-qb, model_top_credit=pc, market_top_credit=qc,
                delta_top_credit=pc-qc, top_sets_unchanged=pt == qt, overround=overround,
                overround_excess=overround-1,
                prediction_lead_seconds=(stamp(record['jump_at'])-stamp(record['predicted_at'])).total_seconds(),
                capture_age_seconds=(stamp(record['cutoff_at'])-stamp(record['quote_at'])).total_seconds(),
                provider_quote_age_seconds=record.get('provider_quote_age_seconds'),
                market_probabilities=q)


METRICS = ('model_log_loss', 'market_log_loss', 'delta_log_loss', 'model_brier', 'market_brier',
           'delta_brier', 'model_top_credit', 'market_top_credit', 'delta_top_credit')


def aggregate(scores):
    return {'races': len(scores), 'runners': sum(r['runners'] for r in scores),
            **{k: sum(r[k] for r in scores)/len(scores) if scores else None for k in METRICS}}


def calibration(records, scores, model):
    bins = [dict(lower=i/10, upper=(i+1)/10, count=0, probability_sum=0., outcome_sum=0.) for i in range(10)]
    for record, result in zip(records, scores):
        probabilities = ([r['probability'] for r in record['runners']] if model else result['market_probabilities'])
        for p, runner in zip(probabilities, record['runners']):
            cell = bins[min(int(p*10), 9)]
            cell['count'] += 1
            cell['probability_sum'] += p
            cell['outcome_sum'] += runner['winner']
    total = sum(cell['count'] for cell in bins)
    for cell in bins:
        n = cell['count']
        cell['mean_probability'] = cell['probability_sum']/n if n else None
        cell['observed_frequency'] = cell['outcome_sum']/n if n else None
    ece = sum(abs(c['probability_sum']-c['outcome_sum']) for c in bins)/total if total else None
    return {'weighting': 'runner_weighted', 'ece': ece, 'bins': bins}


def cluster_intervals(scores):
    groups = defaultdict(list)
    for r in scores:
        groups[r['date']].append(r)
    if len(groups) < 2:
        return {'status': 'NOT_ESTIMABLE_FEWER_THAN_TWO_DATES', 'intervals': None}
    groups = [groups[k] for k in sorted(groups)]
    rng = random.Random(20261011)
    keys = ('delta_log_loss', 'delta_brier', 'delta_top_credit')
    draws = {k: [] for k in keys}
    for _ in range(10000):
        sample = [r for _ in groups for r in rng.choice(groups)]
        for k in keys:
            draws[k].append(sum(r[k] for r in sample)/len(sample))
    intervals = {}
    for k, values in draws.items():
        values.sort()
        intervals[k] = [values[249], values[9749]]
    return {'status': 'DESCRIPTIVE_DATE_CLUSTER_PERCENTILE_95', 'seed': 20261011,
            'draws': 10000, 'dates': len(groups), 'intervals': intervals,
            'limitation': 'Few dates and repeated dogs; not independent confirmation.'}


def benchmark(records, diagnostic=False):
    keys = [(r['race_id'], r['model_version']) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError('multiple_primary_forecasts_for_race_and_version')
    scores = [score(r, diagnostic=diagnostic) for r in records]
    dates = sorted({r['date'] for r in scores})
    versions = sorted({r['model_version'] for r in scores})
    # Never pool differing versions into a single model-performance claim.
    by_version = {}
    for version in versions:
        indexes = [i for i, r in enumerate(scores) if r['model_version'] == version]
        subset, inputs = [scores[i] for i in indexes], [records[i] for i in indexes]
        vd = sorted({r['date'] for r in subset})
        by_version[version] = dict(aggregate(subset), dates=vd,
            timing={key: {'minimum': min(r[key] for r in subset), 'median': statistics.median(r[key] for r in subset), 'maximum': max(r[key] for r in subset)} for key in ('prediction_lead_seconds', 'capture_age_seconds')},
            provider_quote_age_known=sum(r['provider_quote_age_seconds'] is not None for r in subset),
            by_date={d: aggregate([r for r in subset if r['date'] == d]) for d in vd},
            leave_one_date_out={d: aggregate([r for r in subset if r['date'] != d]) for d in vd},
            uncertainty=cluster_intervals(subset),
            model_calibration=calibration(inputs, subset, True),
            market_calibration=calibration(inputs, subset, False),
            disagreements={str(same): dict(aggregate([r for r in subset if r['top_sets_unchanged'] == same])) for same in (True, False)},
            winner_probability_directions=dict(Counter('model_better' if r['delta_log_loss'] < 0 else 'market_better' if r['delta_log_loss'] > 0 else 'equal' for r in subset)))
    return {'schema': 'retained_live_benchmark_v1', 'race_model_records': len(scores),
            'unique_races': len({r['race_id'] for r in scores}), 'dates': dates,
            'model_versions': versions, 'by_version': by_version, 'per_race': scores,
            'status': 'SCORED' if scores else 'NO_ELIGIBLE_DECISION_TIME_COMPARISONS',
            'profitability_claim': False,
            'analysis_class': 'RESULT_FIELD_UNVERIFIED_DIAGNOSTIC' if diagnostic else 'STRICT_DECISION_TIME'}


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return 'Infinity' if value > 0 else '-Infinity'
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--diagnostic', action='store_true', help='Explicitly permit partial result field diagnostics; never strict evidence.')
    args = parser.parse_args()
    data = args.dataset.read_bytes()
    payload = json.loads(data)
    if args.diagnostic != (payload.get('analysis_class') == 'RESULT_FIELD_UNVERIFIED_DIAGNOSTIC'):
        raise ValueError('analysis_class_mismatch')
    result = benchmark(payload['records'], diagnostic=args.diagnostic)
    result['dataset_sha256'] = hashlib.sha256(data).hexdigest()
    result['protocol_sha256'] = payload['protocol_sha256']
    args.output.write_text(json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
