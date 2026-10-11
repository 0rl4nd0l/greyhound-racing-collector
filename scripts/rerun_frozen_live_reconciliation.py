#!/usr/bin/env python3
"""Rerun PR231 scores on reconciled original forecasts without refreezing selectors."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from score_live_benchmark import benchmark, json_safe
from select_live_benchmark import score_forecast_selection


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False)+'\n')


def paired_bounds(forecasts, records):
    """Conditional bounds: each missing race must retain its original field."""
    known = {(r['race_id'], r['model_version']): r for r in records}
    ranges = {'delta_log_loss': [0., 0.], 'delta_brier': [0., 0.], 'delta_top_credit': [0., 0.]}
    observed = 0
    for race in forecasts:
        rows = race['runners']
        p = [r['probability'] for r in rows]
        inverses = [1/r['decimal_odds'] for r in rows]
        q = [v/sum(inverses) for v in inverses]
        model_top, market_top = max(p), max(q)
        pm, qm = sum(v == model_top for v in p), sum(v == market_top for v in q)
        brier_constant = sum(a*a-b*b for a,b in zip(p,q))
        row = known.get((race['race_id'], race['model_version']))
        candidates = range(len(rows))
        if row is not None:
            labels = {r['runner_id']: r['winner'] for r in row['runners']}
            candidates = [i for i,r in enumerate(rows) if labels[r['runner_id']] == 1]
            if len(candidates) != 1:
                raise ValueError('winner_not_unique')
            observed += 1
        values = {
            'delta_log_loss': [math.log(q[i]/p[i]) if p[i] else math.inf for i in candidates],
            'delta_brier': [brier_constant-2*(p[i]-q[i]) for i in candidates],
            'delta_top_credit': [int(p[i] == model_top)/pm-int(q[i] == market_top)/qm for i in candidates],
        }
        for key, samples in values.items():
            ranges[key][0] += min(samples)
            ranges[key][1] += max(samples)
    n = len(forecasts)
    return {'races': n, 'known_outcomes': observed, 'missing_outcomes': n-observed,
            'conditional_mean_difference_bounds': {k: [a/n,b/n] if n else None for k,(a,b) in ranges.items()},
            'assumption': 'Every race must eventually prove its original field unchanged with one winner. Any diagnostic winning-box-to-original-runner mapping used as known must also prove correct. These conditional identification bounds are not confidence intervals and do not qualify missing fields.'}


def verify_originals(records, originals):
    by_race = {r['original']['race_id']: [] for r in originals}
    for item in originals:
        by_race[item['original']['race_id']].append(item['original'])
    for row in records:
        variant = row['model_version'].rsplit(':', 1)[1]
        matches = [r for r in by_race.get(row['race_id'], []) if r['record_key'] == row['prediction_id']]
        if len(matches) != 1 or variant not in ('full','half'):
            raise ValueError('original_forecast_identity_mismatch')
        original = matches[0]
        if row['model_version'] != original['model_sha256']+':'+variant:
            raise ValueError('original_model_identity_changed')
        if row['predicted_at'] != original['score_timestamp'] or row['jump_at'] != original['jump_timestamp']:
            raise ValueError('original_forecast_chronology_changed')
        expected = {r['runner_id']: r for r in original['predictions']}
        if set(expected) != {r['runner_id'] for r in row['runners']}:
            raise ValueError('original_runner_field_changed')
        for r in row['runners']:
            source = expected[r['runner_id']]
            if r['box'] != source['box_number'] or r['probability'] != source[variant+'_probability'] or r['decimal_odds'] != source['strict_win_odds']:
                raise ValueError('original_prediction_or_price_changed')


def verify_frozen_inputs(forecasts, plan, census, dataset_sha256, plan_sha256):
    if plan.get('dataset_sha256') != dataset_sha256 or census.get('dataset_sha256') != dataset_sha256:
        raise ValueError('frozen_dataset_hash_mismatch')
    if census.get('plan_sha256') != plan_sha256:
        raise ValueError('frozen_plan_hash_mismatch')
    if len(forecasts) != 210 or len({r['race_id'] for r in forecasts}) != 210:
        raise ValueError('frozen_population_count_or_identity_mismatch')


def run(old, reconciled, output, protocol):
    output.mkdir(parents=True, exist_ok=True)
    originals = read(old/'alignment/authenticated-originals.json')
    forecasts = read(old/'completed/forecast-dataset.json')['records']
    census_path = old/'completed/forecast-selector-results.json'
    plan_path = old/'completed/forecast-selector-plan.json'
    census = read(census_path)
    plans = read(plan_path)
    verify_frozen_inputs(forecasts, plans, census, digest(old/'completed/forecast-dataset.json'), digest(plan_path))
    # Preserve frozen bytes, rather than regenerating thresholds or cohort ranks.
    (output/'forecast-dataset.json').write_bytes((old/'completed/forecast-dataset.json').read_bytes())
    (output/'forecast-selector-plan.FROZEN.json').write_bytes(plan_path.read_bytes())
    (output/'forecast-selector-dispositions.FROZEN.json').write_bytes(census_path.read_bytes())
    expected = {r['race_id']: r['prediction_id'] for r in forecasts}
    results = {}
    for name in ('strict','diagnostic'):
        source = reconciled/f'{name}-records.json'
        records = read(source)
        if not {r['race_id'] for r in records} <= set(expected):
            raise ValueError('reconciliation_outside_210')
        if any(r['prediction_id'] != expected[r['race_id']] for r in records):
            raise ValueError('frozen_primary_forecast_changed')
        if name == 'strict' and any(r.get('official_runner_identity_observed') is not True for r in records):
            raise ValueError('strict_official_runner_identity_not_observed')
        verify_originals(records, originals)
        payload = dict(schema='retained_live_benchmark_dataset_v1',
                       analysis_class='STRICT_DECISION_TIME' if name == 'strict' else 'RESULT_FIELD_UNVERIFIED_DIAGNOSTIC',
                       protocol_sha256=digest(protocol), source_path=str(source), source_sha256=digest(source), records=records)
        dataset = output/f'{name}-dataset.json'
        write(dataset,payload)
        score = benchmark(records, diagnostic=name == 'diagnostic')
        score.update(dataset_sha256=digest(dataset), protocol_sha256=digest(protocol))
        write(output/f'{name}-scorecard.json',score)
        selectors = score_forecast_selection(census, records, diagnostic=name == 'diagnostic')
        selectors.update(frozen_plan_sha256=digest(plan_path), frozen_dispositions_sha256=digest(census_path))
        write(output/f'{name}-selector-evaluation.json',selectors)
        full = [r for r in records if r['model_version'].endswith(':full')]
        sensitivity = {'all_210':paired_bounds(forecasts, full), 'selected_groups':[]}
        for group in census['comparisons']:
            selected_ids = set(group['selected_race_ids'])
            market_ids = set(group['market_confidence_race_ids'])
            sensitivity['selected_groups'].append(dict(selector=group['selector'],target_coverage=group['target_coverage'],
                selected=paired_bounds([r for r in forecasts if r['race_id'] in selected_ids],full),
                market_confidence=paired_bounds([r for r in forecasts if r['race_id'] in market_ids],full)))
        write(output/f'{name}-missing-outcome-sensitivity.json',sensitivity)
        results[name] = len({r['race_id'] for r in records})
    plan_dates = next(iter(plans['plans'].values()))
    summary = dict(schema='live_reconciliation_rerun_v1',original_forecasts=210, populations=results,
                   frozen_plan_sha256=digest(plan_path), frozen_dispositions_sha256=digest(census_path),
                   development_dates=plan_dates['development_dates'], evaluation_dates=plan_dates['evaluation_dates'],
                   model_fits=0, calibration_fits=0, selector_threshold_changes=0)
    write(output/'summary.json',summary)
    write(output/'manifest.json',{'artifacts':{p.name:digest(p) for p in sorted(output.glob('*.json')) if p.name != 'manifest.json'}})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-evidence',required=True,type=Path)
    parser.add_argument('--reconciled-records',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    protocol=Path(__file__).resolve().parents[1]/'docs/research/live_benchmark_protocol_20261011.md'
    print(json.dumps(run(args.original_evidence,args.reconciled_records,args.output_dir,protocol)))


if __name__ == '__main__':
    main()
