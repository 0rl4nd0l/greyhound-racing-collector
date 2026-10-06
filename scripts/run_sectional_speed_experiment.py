"""Private local orchestration; source/authority containment belongs to root launcher."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from race_collection import sectional_development_inputs as development
from race_collection import sectional_speed_candidate as candidate
from race_collection import sectional_speed_evaluation as evaluation
from race_collection import sectional_speed_oracle as oracle
from race_collection import speed_candidate_inputs as october
from race_collection.sectional_experiment_io import CheckedReader, PrivateOutput


def short(ref):
    return {k: ref[k] for k in ('path', 'sha256')}


def feature_counts(outputs):
    return {'races': len(outputs), 'runner_appearances': sum(o['runner_count'] for o in outputs),
        'supported_runner_appearances': sum(o['supported_runner_count'] for o in outputs),
        'races_with_any_support': sum(o['supported_runner_count'] > 0 for o in outputs),
        'fully_supported_races': sum(o['supported_runner_count'] == o['runner_count'] for o in outputs),
        'supported_history_counts': dict(Counter(str(r['supported_history_count']) for o in outputs for r in o['runners']))}


def feature_stage(scope, reader, output, accounting):
    retained = october.load_inputs(short(scope['october_manifest']), reader)
    legacy = development.load_inputs(scope['legacy'], reader)
    input_refs = {'october': output.put('october-inputs.private.json', retained),
                  'legacy': output.put('legacy-inputs.private.json', legacy)}
    summaries, inventories = {}, {}
    for name, source in [('october', retained), ('legacy', legacy)]:
        packets = [{**p, 'observations': source['observations']} for p in source['packets']] if name == 'october' else source['packets']
        results, refs, single_card = [], [], []
        for packet in packets:
            rid = packet['target']['race_id']
            accounting['active'] = rid
            result = candidate.build_sectional_candidate(packet)
            ref = output.put(name+'-'+hashlib.sha256(rid.encode()).hexdigest()[:24]+'.private.json', result)
            results.append(result)
            refs.append({'race_id': rid, 'artifact': ref})
            if name == 'october':
                local = [row for row in source['observations'] if any(
                    binding['source_race_id'] == rid for binding in row['source_bindings'])]
                single_card.append(candidate.build_sectional_candidate({**packet, 'observations': local}))
            accounting['completed'].append(rid)
            accounting['unattempted'].remove(rid)
            accounting['active'] = None
            reader.check()
        verification = oracle.verify_sample(packets, results, reader)
        verify_ref = output.put(name+'-raw-cell-verification.json', verification)
        summaries[name] = {'final': feature_counts(results), 'raw_cell_verification': verify_ref,
            'source_audit': source['audit'], 'single_card_normalized': feature_counts(single_card) if single_card else None}
        inventories[name] = {'input': input_refs[name], 'records': refs}
    strict = reader.json(scope['strict_summary'])
    summaries['october']['strict_reference'] = {k: strict[k] for k in
        ('cards', 'target_runner_slots', 'supported_runner_slots', 'full_field_supported_cards')}
    summaries['october']['after_verified_aliases'] = summaries['october']['strict_reference']
    summaries['october']['alias_change'] = 'ZERO_PROVEN_ALIASES_NO_POOLING'
    inventory_ref = output.put('feature-inventory.json', inventories)
    reader.check()
    summary = {'status': 'COMPLETE_PRIVATE_SECTIONAL_CANDIDATE_FEATURES', 'cohorts': summaries,
        'inventory': inventory_ref, 'reads': reader.reads, 'read_bytes': reader.bytes,
        'output_bytes_before_summary': output.bytes, 'provider_requests': 0, 'target_labels_decoded': 0,
        'models_fitted': 0, 'performance_evaluation': False}
    output.put('summary.json', summary)
    return {'status': summary['status'], 'coverage': {k: v['final'] for k, v in summaries.items()}}


def _development_records(scope, reader, protocol, output):
    members = {m['race_id'] for m in protocol['members']}
    rows = development.evaluation_rows(scope['legacy'], reader, members)
    model = reader.json(scope['legacy']['baseline_model'])['base16']
    enriched_raw = reader.read(scope['legacy']['enriched'])
    # Enriched dataset has the same allowed development population. Check its
    # identities independently before decoding only the 177 admitted later rows.
    development.identity_rows(enriched_raw, reader.json(scope['legacy']['protected']))
    enriched = {(r['race_id'], r['box']): r for line in enriched_raw.decode().splitlines()
                if development.scalar(line, 'race_id') in members for r in [json.loads(line)]}
    for row in rows:
        other = enriched[row['race_id'], row['box']]
        if (row['dog_token'] != other['dog_token'] or any(
                row['features'].get(name) != other['features'].get(name) for name in model['prep']['names'])):
            raise ValueError('BASE16_INPUT_PARITY_FAILED')
    reproduced = development.reproduce_base16(rows, model)
    prior_raw = reader.read(scope['legacy']['baseline_predictions'])
    prior = {}
    for line in prior_raw.decode().splitlines():
        rid = development.scalar(line, 'race_id')
        if rid not in members:
            raise ValueError('BASELINE_PREDICTION_POPULATION_CHANGED')
        row = json.loads(line)
        if row['outer'] == 'period1':
            prior[row['race_id'], row['box']] = row
    differences = []
    for row, probability in zip(rows, reproduced):
        if row['race_date'] in protocol['splits']['training']:
            original = prior[row['race_id'], row['box']]
            if original['dog_token'] != row['dog_token']:
                raise ValueError('BASELINE_REPRODUCTION_ROSTER')
            differences.append(abs(probability-original['predictions']['refit_base16']))
    if len(differences) != 626 or max(differences) > 1e-12:
        raise ValueError('FIRST_PERIOD_BASELINE_REPRODUCTION_FAILED')
    frozen_forecasts = output.put('fixed-baseline-forecasts.private.json', {
        'model': scope['legacy']['baseline_model'], 'probabilities': reproduced,
        'runner_keys': [[r['race_id'], r['box'], r['dog_token']] for r in rows]})
    proof = output.put('baseline-reproduction.json', {'status': 'REPRODUCED_ORIGINAL_FIRST_PERIOD',
        'original_forecasts_checked': len(differences), 'original_races_checked': 86,
        'max_absolute_error': max(differences), 'base16_feature_parity_runner_entries': len(rows),
        'fixed_model_used_for_all_later_races': True,
        'later_original_models_intentionally_not_used': 'KEEP_BASELINE_TRAINED_BEFORE_SPEED_TRAINING',
        'historical_reference_class': 'EXISTING_DEVELOPMENT_BASE16_METHOD_NOT_LIVE_INSTALLED_ARTIFACT',
        'model': scope['legacy']['baseline_model'], 'forecasts': frozen_forecasts})
    inventory = reader.json(scope['feature_inventory'])['legacy']
    packet = reader.json(inventory['input'])
    targets = {p['target']['race_id']: p for p in packet['packets']}
    refs = {r['race_id']: r['artifact'] for r in inventory['records']}
    grouped = defaultdict(list)
    for row, probability in zip(rows, reproduced):
        grouped[row['race_id']].append((row, probability))
    records = []
    for rid, pairs in sorted(grouped.items()):
        f = reader.json(refs[rid])
        roster = targets[rid]['roster']
        by_box = {r['box_number']: r for r in f['runners']}
        own = [by_box[row['box']] for row, _ in pairs]
        baseline = [p for _, p in pairs]
        records.append({'race_id': rid, 'race_date': pairs[0][0]['race_date'],
            'cutoff': targets[rid]['target']['cutoff'], 'runner_ids': [r['runner_id'] for r in own],
            'market_odds': [r['odds'] for r, _ in pairs],
            'stored_market_probabilities': [r['market'] for r, _ in pairs],
            'stored_baseline_probabilities': baseline, 'reproduced_baseline_probabilities': list(baseline),
            'speed_estimates': [r['speed_estimate'] for r in own],
            'speed_supported': [r['status'] == 'SUPPORTED' for r in own],
            'label_status': 'FULL_ORDER_WIN_ELIGIBLE', 'outcome': [r['y'] for r, _ in pairs],
            'source_bindings': {'forecast': short(frozen_forecasts), 'reproduction': short(proof),
                'features': short(refs[rid]), 'label': short(scope['legacy']['development'])}})
    return records, proof


def evaluation_stage(scope, reader, output, accounting):
    protocol = reader.json(scope['protocol'])
    # Completed raw-cell checks are required before decoding development labels.
    for ref in scope['raw_cell_checks']:
        if reader.json(ref)['status'] != 'VERIFIED':
            raise ValueError('RAW_CELL_VERIFICATION_REQUIRED')
    records, proof = _development_records(scope, reader, protocol, output)
    output.put('evaluation-inputs.private.json', records)
    result = evaluation.evaluate_experiment(records, protocol)
    if result['status'] != 'COMPLETE_RETROSPECTIVE_EXPLORATORY_EVALUATION':
        raise ValueError('EVALUATION_INCOMPLETE')
    for record in result['records']:
        oracle.verify_adjustment(record['stored_baseline_probabilities'], record['speed_estimates'],
                                 record['coefficient'], record['probabilities']['baseline_speed'])
    result_ref = output.put('evaluation.private.json', result)
    reader.check()
    summary = {'status': result['status'], 'result': result_ref, 'baseline_reproduction': proof,
        'accounting': result['accounting'], 'reads': reader.reads, 'read_bytes': reader.bytes,
        'probability_oracle_checked_races': len(result['records']), 'provider_requests': 0,
        'production_changes': False, 'outcomes_public': False}
    output.put('summary.json', summary)
    return {k: summary[k] for k in ('status', 'accounting', 'provider_requests', 'production_changes')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--scope')
    parser.add_argument('--stage', choices=['features', 'evaluation'])
    parser.add_argument('--output')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'status': 'DEFAULT_OFF'}))
        return 0
    if not all((args.scope, args.stage, args.output)):
        parser.error('execute requires scope, stage and new output')
    scope = json.loads(Path(args.scope).read_bytes())
    reader = CheckedReader(scope['allowed_files'], scope['limits'])
    output = PrivateOutput(args.output, scope['limits']['max_output_bytes'])
    accounting = {'completed': [], 'active': None,
                  'unattempted': list(scope['expected_race_ids'])}
    try:
        result = (feature_stage if args.stage == 'features' else evaluation_stage)(scope, reader, output, accounting)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception:
        # Tracebacks remain in root's private stderr; no partial success summary.
        output.failure('EXECUTION_FAILURE_SEE_PRIVATE_STDERR', **accounting)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
