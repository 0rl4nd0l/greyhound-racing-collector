"""Six fixed, fully receipted chronological fits on admitted earlier-card history.

This is an exploratory depth contrast, not a production-history replay. The
protocol is fixed before fitting; every failed attempt is retained separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

from scripts import offline_systematic_search as framework
from scripts.offline_form_packet import FEATURES
from scripts.explain_market_residual import ROOT, FOUNDATION, SOURCE, load_scope, gate_lines, scalar, sha, write, losses
from scripts.development_form_fit import save

PROTOCOL = ROOT / 'docs/research/history_support_20260929_depth_protocol.json'
PROTOCOL_SHA256 = '20f1245937e5915d1f8b6624f83cacdd2a4fff6a35150ba0716330edc7a286b3'
HISTORY = Path('/home/l4nd0/greyhound-history-support-output-20260929/asof_history_v1')
METHODS = ('market', 'short', 'richer')


def admitted_pairs(payload, allowed, original):
    """Resolve all identities and complete fields before feature decoding."""
    lines = payload.decode().splitlines()
    keys = [(scalar(line, 'race_id'), scalar(line, 'box')) for line in lines]
    if len(set(keys)) != len(keys) or any(key not in allowed for key in keys):
        raise ValueError('unadmitted or duplicate history identity')
    for rid in {key[0] for key in keys}:
        if {key for key in keys if key[0] == rid} != {key for key in allowed if key[0] == rid}:
            raise ValueError('incomplete history field')
    pairs = {key: json.loads(line) for key, line in zip(keys, lines)}
    for key, pair in pairs.items():
        for name in ('short_features', 'richer_features'):
            if set(pair[name]) != set(FEATURES):
                raise ValueError('history feature contract mismatch')
            if any(value is not None and not np.isfinite(value) for value in pair[name].values()):
                raise ValueError('nonfinite history feature')
        if pair['short_features'] != {f: original[key]['features'][f] for f in FEATURES}:
            raise ValueError('short features changed')
    return pairs


def paired_summary(races, draws=3000, seed=20260929):
    dates = sorted({r['race_date'] for r in races})
    if len(dates) < 2:
        raise ValueError('date uncertainty requires two dates')
    groups = [[r for r in races if r['race_date'] == day] for day in dates]
    rng = np.random.default_rng(seed)
    weights = np.array([np.bincount(draw, minlength=len(dates)) for draw in rng.integers(0, len(dates), (draws, len(dates)))])
    denominator = weights @ np.array([len(group) for group in groups])
    def aggregate(rows):
        return {'races': len(rows), 'runners': sum(r['runners'] for r in rows), 'dates': len({r['race_date'] for r in rows}),
                'scores': {method: {metric: float(np.mean([r[method][metric] for r in rows])) for metric in ('ll', 'brier', 'accuracy')} for method in METHODS}}
    contrasts = []
    for method, baseline in (('richer', 'short'), ('short', 'market'), ('richer', 'market')):
        for metric in ('ll', 'brier'):
            diffs = np.array([r[method][metric] - r[baseline][metric] for r in races])
            totals = np.array([sum(r[method][metric] - r[baseline][metric] for r in group) for group in groups])
            bootstrap = weights @ totals / denominator
            ordered = sorted(races, key=lambda r: r[method][metric] - r[baseline][metric])
            difference = lambda r: r[method][metric] - r[baseline][metric]
            contrasts.append({'method': method, 'baseline': baseline, 'metric': metric, 'mean_difference': float(diffs.mean()),
                'pointwise95': np.quantile(bootstrap, [.025, .975]).tolist(),
                'improved_races': int((diffs < 0).sum()), 'harmed_races': int((diffs > 0).sum()),
                'improved_dates': int((totals < 0).sum()), 'harmed_dates': int((totals > 0).sum()),
                'leave_one_date_out': {day: float(np.mean([difference(r) for r in races if r['race_date'] != day])) for day in dates},
                'best_five': [{'race_id': r['race_id'], 'difference': difference(r)} for r in ordered[:5]],
                'worst_five': [{'race_id': r['race_id'], 'difference': difference(r)} for r in ordered[-5:]],
                'mean_without_best_five': float(np.mean([difference(r) for r in ordered[5:]])),
                'mean_without_worst_five': float(np.mean([difference(r) for r in ordered[:-5]]))})
    return {'overall': aggregate(races), 'paired': contrasts,
            'periods': {fold: aggregate([r for r in races if r['outer'] == fold]) for fold in sorted({r['outer'] for r in races})},
            'dates': {day: aggregate(group) for day, group in zip(dates, groups)},
            'history_groups': {name: aggregate(group) for name, flag in [('enriched', True), ('unchanged', False)] if (group := [r for r in races if bool(r['enriched_runners']) == flag])},
            'uncertainty': {'draws': draws, 'seed': seed, 'method': 'date cluster percentile bootstrap; exploratory pointwise intervals; no multiple-search correction'}}


def fit_receipt(rows, evaluation, out, admission, contract, environment, ledger, method, fold):
    out.mkdir(parents=True, exist_ok=False)
    attempt = framework.Ledger(out / 'attempt.jsonl')
    attempt.append('START', method=method, outer=fold['name'], new_fit=True)
    ledger.append('FIT_STARTED', method=method, outer=fold['name'])
    try:
        if not rows or any(r['race_date'] >= fold['test_start'] for r in rows):
            raise ValueError('training chronology violation')
        if any(not fold['test_start'] <= r['race_date'] <= fold['test_end'] for r in evaluation):
            raise ValueError('evaluation chronology violation')
        if {r['race_id'] for r in rows} & {r['race_id'] for r in evaluation}:
            raise ValueError('race overlap')
        framework.validate(rows, np.array([r['market'] for r in rows]))
        pins = {name: save(out / name, value) for name, value in (
            ('training_inputs.json', rows), ('evaluation_inputs.json', evaluation),
            ('admission.json', admission), ('feature_contract.json', contract))}
        model = framework.linear_fit(rows, list(FEATURES), 1.0)
        probabilities = framework.predict(evaluation, model)
        framework.validate(evaluation, probabilities)
        receipt = {'identity': 'NEW_DIAGNOSTIC_FIT_NOT_ORIGINAL', 'method': method, 'outer': fold,
            'model': framework.serialize_model(model), 'input_sha256': pins, 'environment': environment,
            'training_membership': [{k: r[k] for k in ('race_id', 'race_date', 'box', 'dog_token')} for r in rows],
            'evaluation_membership': [{k: r[k] for k in ('race_id', 'race_date', 'box', 'dog_token')} for r in evaluation],
            'code_sha256': {str(p): sha(p.read_bytes()) for p in [Path(__file__), Path(framework.__file__), ROOT / 'scripts/offline_form_packet.py', ROOT / 'scripts/build_form_only_v1_packet.py']}}
        receipt_hash = save(out / 'receipt.json', receipt)
        save(out / 'evaluation_probabilities.json', probabilities.tolist())
        attempt.append('COMPLETE', receipt_sha256=receipt_hash)
        ledger.append('FIT_COMPLETE', method=method, outer=fold['name'], receipt_sha256=receipt_hash)
        return probabilities
    except Exception as exc:
        attempt.append('FAILED', error_type=type(exc).__name__, error=str(exc))
        raise


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    ledger = framework.Ledger(out / 'trial_ledger.jsonl')
    ledger.append('START', planned_fits=6, methods=['short', 'richer'], tuning=False)
    try:
        protocol_bytes = PROTOCOL.read_bytes()
        if sha(protocol_bytes) != PROTOCOL_SHA256:
            raise ValueError('frozen protocol changed')
        protocol = json.loads(protocol_bytes)
        write(out / 'protocol.json', protocol)
        allowed, pins = load_scope()
        ledger.append('ADMISSION_PASSED', runners=len(allowed), protected_decodes=0)
        expected = {**json.loads((ROOT / 'docs/research/market_explanation_20260929_inputs.json').read_text()), **protocol['input_pins'], **pins}
        def read(path):
            payload = path.read_bytes()
            if sha(payload) != expected[str(path.resolve())]:
                raise ValueError('pinned source changed: ' + str(path))
            pins[str(path.resolve())] = sha(payload)
            return payload
        for path in protocol['input_pins']:
            read(Path(path))
        data = gate_lines(read(FOUNDATION / 'development.jsonl'), allowed)
        original = {(r['race_id'], r['box']): r for r in data}
        pairs = admitted_pairs(read(HISTORY / 'paired_features.jsonl'), allowed, original)
        reference = gate_lines(read(SOURCE / 'search_v1/outer_predictions.jsonl'), allowed)
        reference_by_key = {(r['race_id'], r['box']): r for r in reference}
        eligible = [r for r in data if (r['race_id'], r['box']) in pairs]
        # No selective exclusion is needed in this fixed reconstruction.
        if len(eligible) != 2360 or len({r['race_id'] for r in eligible}) != 331:
            raise ValueError('fixed qualified reconstruction population changed')
        contract = {'version': 'asof_card_depth_canonical16_v1', 'feature_order': list(FEATURES),
            'formula_identity': 'unchanged scripts.offline_form_packet and build_form_only_v1_packet.feature_row',
            'history': 'short: target admitted card; richer: union with earlier-captured admitted cards, exact same-day normalized dedup, cap20; canonical dog token linkage only',
            'context': 'exact integer distance; canonical grade label, no jurisdiction equivalence claim; original known-finish denominators and zero/null rules held fixed',
            'preprocessing': 'training medians/allmissing0, missingness indicators, training standardization, within-race centering',
            'residual': 'L2=1; market log offset; 0.35*tanh(z/0.35); fullstrength; whole-field softmax',
            'naming': 'career means retained history; no full-career coverage claim'}
        environment = {'python': sys.version, 'executable': sys.executable, 'executable_sha256': sha(Path(sys.executable).read_bytes()),
            'platform': platform.platform(), 'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn')},
            'thread_settings': {k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS')}}
        admission = {'source_hashes': pins, 'history_membership': json.loads(read(HISTORY / 'membership.json')),
            'exclusions': json.loads(read(HISTORY / 'summary.json'))['exclusions'], 'allowed_races': 331, 'allowed_runners': 2360}
        save(out / 'environment.json', environment)
        save(out / 'feature_contract.json', contract)
        save(out / 'paired_features.json', list(pairs.values()))
        predictions, races, max_error = [], [], 0.0
        for fold in framework.OUTER:
            train = [r for r in eligible if r['race_date'] < fold['test_start']]
            test = [r for r in eligible if fold['test_start'] <= r['race_date'] <= fold['test_end']]
            probs = {'market': np.array([r['market'] for r in test])}
            for method in ('short', 'richer'):
                def with_features(rows):
                    return [{**r, 'features': pairs[r['race_id'], r['box']][method + '_features']} for r in rows]
                probs[method] = fit_receipt(with_features(train), with_features(test), out / (fold['name'] + '_' + method), admission, contract, environment, ledger, method, fold)
            for i, r in enumerate(test):
                old = reference_by_key[r['race_id'], r['box']]
                if old['outer'] != fold['name'] or any(old[k] != r[k] for k in ('y', 'market', 'capture', 'jump')):
                    raise ValueError('original evaluation identity or inputs changed')
                max_error = max(max_error, abs(probs['short'][i] - old['predictions']['refit_base16']))
                predictions.append({k: r[k] for k in ('race_id', 'race_date', 'box', 'dog_token', 'y', 'market')} | {
                    'outer': fold['name'], 'probabilities': {m: float(p[i]) for m, p in probs.items()},
                    'short_count': pairs[r['race_id'], r['box']]['short_count'], 'richer_count': pairs[r['race_id'], r['box']]['richer_count']})
            boundaries, sizes = framework.starts(test)
            for start, size in zip(boundaries, sizes):
                rows = test[start:start + size]
                races.append({'race_id': rows[0]['race_id'], 'race_date': rows[0]['race_date'], 'outer': fold['name'], 'runners': int(size),
                    'enriched_runners': sum(pairs[r['race_id'], r['box']]['richer_count'] > pairs[r['race_id'], r['box']]['short_count'] for r in rows),
                    **{m: losses([r['y'] for r in rows], p[start:start + size]) for m, p in probs.items()}})
        if len(races) != 177 or len(predictions) != 1251 or max_error > 1e-10:
            raise ValueError('original evaluation coverage or short refit replay failed: ' + str(max_error))
        save(out / 'evaluation_predictions.json', predictions)
        ledger.append('ALL_PREDICTIONS_SEALED', new_fits=6, prediction_sha256=sha((out / 'evaluation_predictions.json').read_bytes()))
        result = paired_summary(races, **{'draws': protocol['uncertainty']['date_cluster_draws'], 'seed': protocol['uncertainty']['seed']})
        result['short_vs_original_maximum_probability_error'] = max_error
        result['coverage'] = {'original_races': 331, 'eligible_races': 331, 'original_evaluation_races': 177, 'evaluated_races': 177, 'evaluated_runners': 1251,
            'enriched_evaluation_races': sum(bool(r['enriched_runners']) for r in races), 'enriched_evaluation_runners': sum(r['enriched_runners'] for r in races)}
        write(out / 'race_metrics.json', races)
        write(out / 'summary.json', result)
        write(out / 'input_hashes.json', pins)
        ledger.append('COMPLETE', new_fits=6, summary_sha256=sha((out / 'summary.json').read_bytes()))
        return result
    except Exception as exc:
        ledger.append('FAILED', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.out)['overall'], sort_keys=True))
