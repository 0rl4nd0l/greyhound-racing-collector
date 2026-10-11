#!/usr/bin/env python3
"""Assemble and score audited retained projections offline; no acquisition or fits."""
import argparse
import hashlib
import json
from pathlib import Path

from score_live_benchmark import benchmark, json_safe
from select_live_benchmark import census, evaluate, freeze, score_forecast_selection


def write(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    root, output = args.evidence_root, args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    protocol = repo/'docs/research/live_benchmark_protocol_20261011.md'
    selector_protocol = repo/'docs/research/live_selector_protocol_20261011.md'
    protocol_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
    selector_hash = hashlib.sha256(selector_protocol.read_bytes()).hexdigest()
    datasets = {}
    for name, kind in [('strict', 'STRICT_DECISION_TIME'), ('diagnostic', 'RESULT_FIELD_UNVERIFIED_DIAGNOSTIC'), ('forecast', 'AUTHENTICATED_FORECASTS_RESULT_INDEPENDENT'), ('manual-forecast', 'MANUAL_INCOMPLETE_SEAL_NO_RESULTS')]:
        source = root/'alignment'/f'{name}-records.json'
        data = dict(schema='retained_live_benchmark_dataset_v1', analysis_class=kind,
                    protocol_sha256=protocol_hash, source_path=str(source),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), records=json.loads(source.read_text()))
        target = output/f'{name}-dataset.json'
        write(target, data)
        datasets[name] = (data, hashlib.sha256(target.read_bytes()).hexdigest())
    # Complete original all-race scoring before freezing/executing selectors.
    for name in ('strict', 'diagnostic'):
        data, digest = datasets[name]
        result = benchmark(data['records'], diagnostic=name == 'diagnostic')
        result.update(dataset_sha256=digest, protocol_sha256=protocol_hash)
        write(output/f'{name}-scorecard.json', result)
    for name, (data, digest) in datasets.items():
        plan = freeze(data['records'])
        plan_path = output/f'{name}-selector-plan.json'
        write(plan_path, dict(plan, dataset_sha256=digest, selector_protocol_sha256=selector_hash))
        # Re-read sealed thresholds before performance calculation; no fitting.
        sealed = json.loads(plan_path.read_text())
        sealed.pop('dataset_sha256')
        sealed.pop('selector_protocol_sha256')
        result = (census(data['records'], sealed) if name == 'forecast'
                  else evaluate(data['records'], sealed, diagnostic=name == 'diagnostic'))
        if name == 'diagnostic':
            result['selector_population_warning'] = 'EXPLORATORY_RESULT_AVAILABILITY_CONDITIONED_SUBSET_NOT_PRIMARY_SELECTOR_TEST'
        result.update(dataset_sha256=digest, analysis_class=data['analysis_class'],
                      plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest())
        write(output/f'{name}-selector-results.json', result)
    forecast_selection = json.loads((output/'forecast-selector-results.json').read_text())
    for name in ('strict', 'diagnostic'):
        result = score_forecast_selection(forecast_selection, datasets[name][0]['records'], diagnostic=name == 'diagnostic')
        result['forecast_selection_sha256'] = hashlib.sha256((output/'forecast-selector-results.json').read_bytes()).hexdigest()
        write(output/f'primary-{name}-selector-evaluation.json', result)
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob('*.json')) if p.name != 'manifest.json'}
    write(output/'manifest.json', {'schema': 'live_benchmark_output_manifest_v1', 'artifacts': manifest,
                                 'protocol_sha256': protocol_hash, 'selector_protocol_sha256': selector_hash})
    print(json.dumps({'output_dir': str(output), 'strict_races': len(datasets['strict'][0]['records']),
                      'diagnostic_race_model_records': len(datasets['diagnostic'][0]['records']),
                      'forecast_records': len(datasets['forecast'][0]['records'])}))


if __name__ == '__main__':
    main()
