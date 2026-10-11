#!/usr/bin/env python3
"""Three predeclared offline fits; exclusive outputs and complete fit ledger."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import minimize

from race_collection import historical_dynamic_ability as ability
from race_collection import historical_model_experiment as prior
from race_collection import historical_ranking_experiment as ranking
from race_collection import historical_relative_support as support

DEFAULT_CONTEXT = Path('/mnt/tenn-nvme2/tenn/greyhound-historical-next-comparison-20261010-evidence/context-package-01')
CONTEXT_MANIFEST_SHA = '414766581055965d8c648013a7006141c9fee43d6f69adcdb1740a79d73fad21'
DEFAULT_BASELINE = Path('/mnt/tenn-nvme2/tenn/greyhound-historical-improvement-execution-20261010-evidence/run-01')
SPLITS = ('train', 'development', 'later')
FEATURE_NAMES = list(support.BASELINE65) + ['state_' + f for f in ability.FIELDS]
REPLACED_FEATURES = ('recent_finish_mean_3', 'retained_finish_mean', 'retained_win_rate',
                     'retained_top3_rate', 'retained_same_venue_win_rate',
                     'retained_exact_distance_win_rate')


def feature_names(mode):
    baseline = [name for name in support.BASELINE65
                if mode == 'addition' or name not in REPLACED_FEATURES]
    return baseline + ['state_' + field for field in ability.FIELDS]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_checked(path, sha):
    raw = path.read_bytes()
    if digest(raw) != sha:
        raise ValueError('INPUT_HASH_MISMATCH:' + str(path))
    return raw


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def jsonl(path, rows):
    with path.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')


def load(context):
    hashes = json.loads(read_checked(context / 'artifacts.sha256.json', CONTEXT_MANIFEST_SHA))
    ref = json.loads(read_checked(context / 'input_reference.json', hashes['input_reference.json']))
    corpus = ref['corpus']['manifest']
    # Access scope is decoded before any authorized target-body file.
    scope = json.loads(read_checked(Path(corpus['path']), corpus['sha256']))
    if scope['split_cutoffs'] != {'train_end': '2025-08-31', 'validation_end': '2025-10-31'}:
        raise ValueError('ACCESS_SCOPE_CHANGED')
    allowed = set(scope['allowed_source_race_keys']) - set(scope['excluded_source_race_keys'])
    summary = json.loads(read_checked(context / 'summary.json', hashes['summary.json']))
    groups = {}
    for split, expected in zip(SPLITS, (720, 656, 975)):
        groups[split] = [json.loads(line) for line in read_checked(context / (split + '.jsonl'), hashes[split + '.jsonl']).splitlines()]
        if len(groups[split]) != expected:
            raise ValueError('FIXED_POPULATION_CHANGED')
        for group in groups[split]:
            meta = group['metadata']
            day = meta['race_date']
            expected_split = 'train' if day <= '2025-08-31' else 'development' if day <= '2025-09-08' else 'later'
            if expected_split != split or meta['source_race_key'] not in allowed or day > '2025-10-01':
                raise ValueError('OUT_OF_SCOPE_TARGET')
    archive = [json.loads(line) for line in read_checked(context / 'retained_context_evidence.jsonl', hashes['retained_context_evidence.jsonl']).splitlines()]
    index = {(r['source_race_key'], r['guide_box']): r for r in archive}
    expected = {(g['metadata']['source_race_key'], r['metadata']['guide_box']) for gs in groups.values() for g in gs for r in g['runners']}
    if set(index) != expected or len(index) != len(archive):
        raise ValueError('IDENTITY_ARCHIVE_POPULATION_MISMATCH')
    events = []
    for split in SPLITS:
        for group in groups[split]:
            meta = group['metadata']
            venue = meta['source_race_key'].split(' - ')[1]
            runners, distances = [], set()
            for runner in group['runners']:
                box = runner['metadata']['guide_box']
                item = index[(meta['source_race_key'], box)]
                native = item['native_dog_id'] if item['identity_qualification'] == 'NATIVE_ID_AND_TOKEN_UNAMBIGUOUS' else None
                distances.add(item['static_target_context']['distance_m'])
                runners.append({'box': box, 'native_id': native, 'finish': runner['target']['finish_position']})
            if len(distances) != 1:
                raise ValueError('INCONSISTENT_TARGET_CONTEXT')
            distance = next(iter(distances))
            events.append({'date': meta['race_date'], 'key': meta['source_race_key'], 'partition': split,
                           'venue': venue, 'distance_m': distance,
                           'context': (venue, distance) if distance is not None else None, 'runners': runners})
    return groups, events, {'context_manifest_sha256': CONTEXT_MANIFEST_SHA, 'context_input_reference': ref,
                            'opened_target_files': [str(context / (s + '.jsonl')) for s in SPLITS],
                            'test_labels_opened': False, 'new_recovered_population_used': False,
                            'counts': {s: len(groups[s]) for s in SPLITS}, 'upstream_summary': summary}


def rows_for(groups, states):
    rows = ranking._rows(groups, labels=True)
    for row in rows:
        row['features'].update({'state_' + key: value for key, value in states[(row['race_id'], row['guide_box'])].items()})
    return rows


def fit(rows, names=FEATURE_NAMES):
    prep = prior.research.prefit(rows, names)
    x = prior.research.transform(rows, prep)
    fitted = minimize(ranking._objective(x, rows, .5), np.zeros(x.shape[1]), jac=True,
                      method='L-BFGS-B', options=ranking.OPTIMIZER)
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError('OPTIMIZER_FAILED:' + str(fitted.message))
    return {'prep': prep, 'beta': fitted.x, 'offset': False, 'cap': None, 'iterations': fitted.nit}


def comparison(rows, candidate, reference):
    starts, ends = prior.research.group(rows)
    delta = prior.research.scores(rows, candidate)[0] - prior.research.scores(rows, reference)[0]
    dates = np.array([rows[i]['race_date'] for i in starts])
    venues = np.array([rows[i]['race_id'].split(' - ')[1] for i in starts])
    days = sorted(set(dates))
    blocks = [np.flatnonzero(dates == day) for day in days]
    rng = np.random.default_rng(20261011)
    draws = [float(delta[np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])].mean()) for _ in range(2000)]
    return {'delta_log_loss': float(delta.mean()), 'date_bootstrap_95': np.quantile(draws, [.025, .975]).tolist(),
            'dates': [{'date': str(day), 'races': int((dates == day).sum()), 'delta_log_loss': float(delta[dates == day].mean()),
                       'leave_out_delta_log_loss': float(delta[dates != day].mean())} for day in days],
            'tracks': [{'venue': str(v), 'races': int((venues == v).sum()), 'delta_log_loss': float(delta[venues == v].mean())} for v in sorted(set(venues))]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--context', type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--feature-mode', choices=('addition', 'replacement'), default='addition')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    protocol = {'schema': 'dynamic_ability_v1', 'arms': list(ability.ARMS), 'predictor_fits': 3,
                'features': feature_names(args.feature_mode), 'feature_mode': args.feature_mode,
                'replaced_features': list(REPLACED_FEATURES) if args.feature_mode == 'replacement' else [],
                'half_life_days': ability.HALF_LIFE_DAYS,
                'global_prior_starts': ability.GLOBAL_PRIOR_STARTS, 'context_prior_starts': ability.CONTEXT_PRIOR_STARTS,
                'downstream': 'same hybrid50 regularized linear conditional-softmax,720 training races',
                'outcome_update': 'all qualified admitted races strictly before target DATE; never same date',
                'representation': 'centered fraction of opponents beaten; dynamic adds mean opponents prior ratings; shrunk exact track/distance residual',
                'tuning': 'none', 'coefficient_training_end': '2025-08-31',
                'development_dates': ['2025-09-01', '2025-09-08'], 'later_dates': ['2025-09-09', '2025-10-01'],
                'preprocessing': 'medians,missing flags,means,scales fitted only720 training races',
                'evaluation_status': 'previously_exposed_development', 'primary_metric': 'paired race log loss',
                'bootstrap': {'draws': 2000, 'seed': 20261011, 'unit': 'whole date'},
                'stopping': 'three fixed converged fits; no tuning expansion after results; retain every intent/failure',
                'context_caveat': 'static distance and final field reconstructed from results; no decision-time witness',
                'identity_caveat': 'as-of context archive native-ID qualification; no later identity backfill',
                'test_labels_opened': False, 'promotion_eligible': False,
                'source_hashes': {str(p): digest(p.read_bytes()) for p in (Path(__file__), Path(ability.__file__), Path(ranking.__file__), Path(prior.research.__file__))}}
    write(out / 'protocol.json', protocol)
    groups, events, provenance = load(args.context)
    states, audit, lineage = ability.representations(events)
    write(out / 'input_provenance.json', provenance)
    write(out / 'state_audit.json', audit)
    jsonl(out / 'state_update_lineage.jsonl', lineage)
    exported = []
    for event in events:
        for runner in event['runners']:
            key = (event['key'], runner['box'])
            rivals = [states['dynamic'][(event['key'], r['box'])] for r in event['runners'] if r['box'] != runner['box']]
            exported.append({'source_race_key': key[0], 'guide_box': key[1], 'native_dog_id': runner['native_id'],
                             'race_date': event['date'], 'partition': event['partition'], 'venue': event['venue'], 'distance_m': event['distance_m'],
                             'states': {arm: states[arm][key] for arm in ability.ARMS},
                             'opponents_prior_mean': sum((s['ability'] or 0.) + (s['context_residual'] or 0.) for s in rivals) / len(rivals),
                             'opponents_with_prior_support': sum(s['support'] > 0 for s in rivals)})
    jsonl(out / 'runner_states.jsonl', exported)
    jsonl(out / 'membership.jsonl', [{k: e[k] for k in ('key', 'date', 'partition', 'venue', 'distance_m')} for e in events])
    if args.prepare_only:
        print(json.dumps({'state': 'PREPARED', 'audit': audit}))
        return
    predictions = {split: {} for split in SPLITS}
    models = {}
    for arm in ability.ARMS:
        train = rows_for(groups['train'], states[arm])
        write(out / ('fit_intent_' + arm + '.json'), {'arm': arm, 'training_races': 720, 'protocol_sha256': digest((out / 'protocol.json').read_bytes())})
        try:
            model = fit(train, feature_names(args.feature_mode))
        except Exception as exc:
            write(out / ('fit_failure_' + arm + '.json'), {'error': repr(exc)})
            raise
        record = prior._model_record(model)
        record['fit_method'] = {'winner_weight': .5, 'optimizer': ranking.OPTIMIZER}
        models[arm] = record
        write(out / ('model_' + arm + '.json'), record)
        for split in SPLITS:
            # Strip target fields before predicting; state updates already froze
            # each feature vector at its own prior-date information boundary.
            rows = rows_for(groups[split], states[arm])
            unlabeled = [{k: v for k, v in r.items() if k not in ('y', 'finish_position')} for r in rows]
            predictions[split][arm] = prior.research.predict_linear(unlabeled, model)
        print(json.dumps({'completed_arm': arm, 'iterations': model['iterations']}), flush=True)
    baseline_hashes = json.loads((args.baseline / 'artifacts.sha256.json').read_text())
    # Frozen predictions are replayed as a comparator; no fourth predictor fit.
    for split in SPLITS:
        file = split + '_predictions.jsonl'
        if file not in baseline_hashes:
            continue
        saved = [json.loads(line) for line in read_checked(args.baseline / file, baseline_hashes[file]).splitlines()]
        rows = rows_for(groups[split], states['average'])
        lookup = {(r['source_race_key'], r['guide_box']): r for r in saved}
        if set(lookup) != {(r['source_race_key'], r['guide_box']) for r in rows}:
            raise ValueError('FROZEN_BASELINE_MEMBERSHIP_CHANGED')
        name = 'matched65'
        print('baseline_probability_names=' + ','.join(saved[0]['probabilities']), flush=True)
        candidates = [n for n in saved[0]['probabilities'] if 'matched' in n and '65' in n]
        if len(candidates) != 1:
            raise ValueError('FROZEN_BASELINE_NAME_AMBIGUOUS')
        predictions[split]['saved_baseline65'] = np.array([lookup[(r['source_race_key'], r['guide_box'])]['probabilities'][candidates[0]] for r in rows])
    for split in SPLITS:
        rows = rows_for(groups[split], states['average'])
        jsonl(out / (split + '_predictions.jsonl'), prior._predictions(rows, predictions[split]))
    write(out / 'prediction_hashes.json', {s: digest((out / (s + '_predictions.jsonl')).read_bytes()) for s in SPLITS})
    summary = {'state': 'COMPLETE', 'fits_completed': len(models), 'metrics': {}, 'contrasts': {},
               'test_labels_opened': False, 'promotion_eligible': False, 'evaluation_status': 'exploratory_development'}
    for split in SPLITS:
        rows = rows_for(groups[split], states['average'])
        summary['metrics'][split] = {arm: prior.research.metrics(rows, p) for arm, p in predictions[split].items()}
        if split == 'train':
            continue
        summary['contrasts'][split] = {}
        pairs = [('recency', 'average'), ('dynamic', 'recency'), ('dynamic', 'average')]
        if 'saved_baseline65' in predictions[split]:
            pairs += [(arm, 'saved_baseline65') for arm in ability.ARMS]
        for candidate, reference in pairs:
            summary['contrasts'][split][candidate + '_minus_' + reference] = comparison(rows, predictions[split][candidate], predictions[split][reference])
        race_scores = []
        scores = {a: prior.research.scores(rows, p) for a, p in predictions[split].items()}
        for index, (start, end) in enumerate(zip(*prior.research.group(rows))):
            race_scores.append({'source_race_key': rows[start]['race_id'], 'race_date': rows[start]['race_date'],
                                'field_size': int(end - start), 'scores': {a: {'log_loss': float(s[0][index]), 'brier': float(s[1][index]), 'top1': float(s[2][index])} for a, s in scores.items()}})
        jsonl(out / (split + '_race_scores.jsonl'), race_scores)
    write(out / 'summary.json', summary)
    write(out / 'artifacts.sha256.json', {p.name: digest(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps({'state': 'COMPLETE', 'metrics': {s: {a: {k: m[k] for k in ('races', 'log_loss', 'brier', 'top1')} for a, m in summary['metrics'][s].items()} for s in ('development', 'later')}}), flush=True)


if __name__ == '__main__':
    main()
