"""Reconstruct the fixed authorised 82-card graph without opening target labels."""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import resource
import time

from race_collection import prospective_speed_inputs as inputs
from race_collection import sectional_speed_oracle as oracle
from race_collection.prospective_speed_runtime import encoded, put_new
from race_collection.retained_card_timing_coverage import Reader


def scalar_baseline(rows, model):
    """Independent scalar arithmetic, without the production numpy scorer."""
    prep = model['prep']
    vectors = []
    for row in rows:
        values = [row['features'][name] for name in prep['names']]
        filled = [prep['median'][i] if value is None else value for i, value in enumerate(values)]
        combined = filled + [int(value is None) for value in values]
        vectors.append([(v - prep['mean'][i]) / prep['scale'][i] for i, v in enumerate(combined)])
    centers = [math.fsum(v[i] for v in vectors) / len(vectors) for i in range(len(vectors[0]))]
    logits = [math.log(row['market']) + .35 * math.tanh(math.fsum(
        (v - centers[i]) * model['beta'][i] for i, v in enumerate(vector)) / .35)
        for row, vector in zip(rows, vectors)]
    weights = [math.exp(v - max(logits)) for v in logits]
    return [v / math.fsum(weights) for v in weights]


def run(jobs_reference, output):
    started = time.perf_counter()
    reader = Reader()
    jobs = reader.json(jobs_reference)
    if len(jobs) != 82:
        raise ValueError('FIXED_REPLAY_POPULATION_CHANGED')
    jobs = [reader.json(ref) for ref in jobs]
    authority = reader.json(jobs[0]['execution_authority'])
    if (authority['maximum_cards'] != 82 or authority['target_results_opened'] != 0
            or authority['provider_requests'] != 0 or authority['result_requests'] != 0):
        raise ValueError('REPLAY_AUTHORITY_INVALID')
    seed = reader.json(jobs[0]['history_inventory'])
    if len(seed['cards']) != 82 or any(j['history_inventory'] != jobs[0]['history_inventory'] for j in jobs):
        raise ValueError('REPLAY_HISTORY_POPULATION_CHANGED')
    history_started = time.perf_counter()
    observations = []
    for card in seed['cards']:
        _, rows, _ = inputs.retained.construct_member(Reader(), card['member'], card['original'])
        observations.extend(rows)
    history_seconds = time.perf_counter() - history_started
    output = Path(output)
    output.mkdir(mode=0o700)
    put_new(output / 'started.json', {'at': datetime.now(timezone.utc).isoformat(),
        'jobs': jobs_reference, 'mode': 'RETAINED_RECONSTRUCTION_NOT_PROSPECTIVE'})
    packets, features, records = [], [], []
    maximum_error = 0
    for index, job in enumerate(jobs):
        before = time.perf_counter()
        try:
            result = inputs.forecast(Reader(), job['member'], job['original'], job['model'],
                forecast_at=job['forecast_at'], prior_observations=observations)
            result['speed_packet']['target']['observation_pool_scope'] = 'ALL_SOURCE_CARDS'
            model = Reader().json(job['model'])['base16']
            baseline = scalar_baseline(result['baseline_rows'], model)
            discrepancy = max(abs(a-b['baseline']) for a, b in zip(baseline, result['predictions']))
            if discrepancy > 1e-12:
                raise ValueError('INDEPENDENT_BASELINE_MISMATCH')
            maximum_error = max(maximum_error, discrepancy)
            oracle.verify_adjustment([r['baseline'] for r in result['predictions']],
                [r['speed_estimate'] for r in result['speed_features']['runners']], .1,
                [r['baseline_plus_speed'] for r in result['predictions']])
            packets.append(result['speed_packet']); features.append(result['speed_features'])
            ref = put_new(output / f'forecast-{index:03}.json', result)
            records.append({'race_id': result['race_id'], 'status': 'RECONSTRUCTED', 'output': ref,
                'runners': len(result['predictions']),
                'supported': sum(r['status'] == 'SUPPORTED' for r in result['speed_features']['runners']),
                'wall_seconds': time.perf_counter() - before, 'storage_bytes': len(encoded(result)) + 1})
        except Exception as error:
            records.append({'race_id': job['member']['race_id'], 'status': 'REPLAY_FAILED',
                'error_type': type(error).__name__, 'reason': str(error)[:150]})
        put_new(output / f'disposition-{index:03}.json', records[-1])
    verification = oracle.verify_sample(packets, features, Reader()) if len(packets) == 82 else {
        'status': 'NOT_RUN_INCOMPLETE_SOURCE_POPULATION'}
    put_new(output / 'raw-cell-verification.json', verification)
    summary = {'status': 'COMPLETE' if len(packets) == 82 else 'COMPLETE_WITH_FAILURES',
        'mode': 'RETAINED_RECONSTRUCTION_NOT_PROSPECTIVE', 'records': records,
        'source_cards': 82, 'successful_forecasts': len(packets),
        'runner_appearances': sum(r.get('runners', 0) for r in records),
        'supported_appearances': sum(r.get('supported', 0) for r in records),
        'races_with_support': sum(r.get('supported', 0) > 0 for r in records),
        'fully_supported_races': sum(r.get('supported', -1) == r.get('runners', -2) for r in records),
        'independent_baseline_max_absolute_error': maximum_error,
        'history_authentication_seconds': history_seconds,
        'total_wall_seconds': time.perf_counter() - started,
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'forecast_storage_bytes': sum(r.get('storage_bytes', 0) for r in records),
        'provider_requests': 0, 'result_requests': 0, 'target_results_opened': 0,
        'fresh_prospective_forecasts': 0, 'predictive_performance_evaluated': False}
    put_new(output / 'summary.json', summary)
    return {k:v for k,v in summary.items() if k != 'records'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', required=True)
    parser.add_argument('--jobs-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(run({'path': args.jobs, 'sha256': args.jobs_sha256}, args.output))
