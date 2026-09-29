"""Finite offline support shrinkage using previously retained chronological predictions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import sys

import numpy as np

from scripts.offline_form_packet import FEATURES
from scripts.offline_systematic_search import OUTER, Ledger
from scripts.explain_market_residual import (
    ROOT, SOURCE, FOUNDATION, load_scope, gate_lines, sha, write, losses, decomposition,
)

PROTOCOL = ROOT / 'docs/research/history_support_20260929_protocol.json'
PROTOCOL_SHA256 = '15ea074b35cefd8dad001340e8e1901c0cb973b0f8b58ef58f38e3ca94aa11e0'
MODELS = ('market', 'full', 'half', 'adaptive')


def adjust(market, full, alpha):
    """Race-wide scaling is invariant to the unretained residual normalizer."""
    market, full = np.asarray(market, float), np.asarray(full, float)
    if market.ndim != 1 or full.shape != market.shape or not 0 <= alpha <= 1:
        raise ValueError('invalid whole-field shrinkage')
    for p in (market, full):
        if len(p) < 2 or not np.isfinite(p).all() or (p <= 0).any() or not np.isclose(p.sum(), 1, atol=1e-10, rtol=0):
            raise ValueError('invalid whole-field distribution')
    logits = np.log(market) + alpha * (np.log(full) - np.log(market))
    result = np.exp(logits - logits.max())
    return result / result.sum()


def race_support(rows):
    """Outcome-free, prespecified support; unavailable context lowers confidence."""
    support = []
    for row in rows:
        f = row['features']
        n = f['prior_start_count']
        n = 0.0 if n is None else float(n)
        if not np.isfinite(n) or n < 0:
            raise ValueError('invalid history count')
        contexts = []
        for count, rate in (('starts_same_venue', 'win_rate_same_venue'),
                            ('starts_same_distance', 'win_rate_same_distance'),
                            ('same_grade_start_count', 'same_grade_win_rate')):
            value = f[count]
            if value is None or not np.isfinite(value) or f[rate] is None or not np.isfinite(f[rate]):
                value = 0.0
            if not np.isfinite(value) or not 0 <= value <= n:
                raise ValueError('invalid context count')
            contexts.append(min(value, 3) / 3)
        known = sum(f[name] is not None and np.isfinite(f[name]) for name in FEATURES) / len(FEATURES)
        support.append(min(n, 5) / 5 * float(np.mean(contexts)) * known)
    if not support:
        raise ValueError('empty race')
    return float(np.mean(support))


def grouped(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['race_id']].append(row)
    return [sorted(group, key=lambda r: r['box']) for _, group in sorted(groups.items())]


def select_parameter(validation, parameters):
    """Accept earlier validation races only; no evaluation rows are an input."""
    trials = []
    for parameter in parameters:
        scores = []
        for rows in grouped(validation):
            support = race_support(rows)
            alpha = support / (support + parameter)
            prediction = adjust([r['market'] for r in rows], [r['full'] for r in rows], alpha)
            scores.append(losses([r['y'] for r in rows], prediction)['ll'])
        trials.append({'lambda': parameter, 'log_loss': float(np.mean(scores)), 'races': len(scores)})
    chosen = min(trials, key=lambda trial: (trial['log_loss'], -trial['lambda']))['lambda']
    return chosen, trials


def metric_summary(races, draws=3000, seed=20260929):
    dates = sorted({r['race_date'] for r in races})
    if len(dates) < 2:
        raise ValueError('date uncertainty requires at least two dates')
    date_rows = [[r for r in races if r['race_date'] == day] for day in dates]
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(dates), size=(draws, len(dates)))
    weights = np.array([np.bincount(c, minlength=len(dates)) for c in choices])
    denominator = weights @ np.array([len(rr) for rr in date_rows])
    comparisons, standardized = [], []
    for baseline in MODELS[:-1]:
        for metric in ('ll', 'brier'):
            differences = np.array([r['adaptive'][metric] - r[baseline][metric] for r in races])
            mean = float(differences.mean())
            totals = np.array([sum(r['adaptive'][metric] - r[baseline][metric] for r in rr) for rr in date_rows])
            sampled = weights @ totals / denominator
            se = float(sampled.std(ddof=1))
            if se > 1e-15:
                standardized.append(np.abs((sampled - mean) / se))
            comparisons.append({'baseline': baseline, 'metric': metric,
                                'difference_adaptive_minus_baseline': mean,
                                'pointwise95': np.quantile(sampled, [.025, .975]).tolist(),
                                'bootstrap_se': se,
                                'leave_one_date_out': {day: float(np.mean([r['adaptive'][metric] - r[baseline][metric] for r in races if r['race_date'] != day])) for day in dates},
                                'improved_races': int((differences < 0).sum()),
                                'harmed_races': int((differences > 0).sum()),
                                'improved_dates': int((totals < 0).sum()),
                                'harmed_dates': int((totals > 0).sum())})
    critical = float(np.quantile(np.max(standardized, axis=0), .95)) if standardized else 0.0
    for comparison in comparisons:
        mean = comparison['difference_adaptive_minus_baseline']
        width = critical * comparison['bootstrap_se']
        comparison['simultaneous95'] = [mean - width, mean + width]
    def aggregate(rr):
        return {'races': len(rr), 'dates': len({r['race_date'] for r in rr}),
                'scores': {model: {metric: float(np.mean([r[model][metric] for r in rr])) for metric in ('ll', 'brier', 'accuracy')} for model in MODELS}}
    influence = {}
    for baseline in MODELS[:-1]:
        ordered = sorted(races, key=lambda r: r['adaptive']['ll'] - r[baseline]['ll'])
        difference = lambda r: r['adaptive']['ll'] - r[baseline]['ll']
        influence[baseline] = {
            'best_five': [{'race_id': r['race_id'], 'difference': difference(r)} for r in ordered[:5]],
            'worst_five': [{'race_id': r['race_id'], 'difference': difference(r)} for r in ordered[-5:]],
            'mean_without_best_five': float(np.mean([difference(r) for r in ordered[5:]])),
            'mean_without_worst_five': float(np.mean([difference(r) for r in ordered[:-5]])),
            'net_difference_sum': sum(difference(r) for r in races),
            'best_five_difference_sum': sum(difference(r) for r in ordered[:5]),
        }
    return {'overall': aggregate(races), 'paired': comparisons,
            'periods': {key: aggregate([r for r in races if r['outer'] == key]) for key in sorted({r['outer'] for r in races})},
            'dates': {day: aggregate(rr) for day, rr in zip(dates, date_rows)},
            'support_groups': {name: aggregate(rr) for name, low, high in [('low', 0, 1/3), ('middle', 1/3, 2/3), ('high', 2/3, 1.00001)] if (rr := [r for r in races if low <= r['support'] < high])},
            'influence': influence, 'uncertainty': {'draws': draws, 'seed': seed, 'contrasts': 6, 'simultaneous_critical': critical,
            'status': 'exploratory; does not correct all prior experimentation'}}


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    ledger = Ledger(out / 'trial_ledger.jsonl')
    ledger.append('START', new_fits=0, attempted_parameters=[.25, 1.0])
    try:
        protocol_bytes = PROTOCOL.read_bytes()
        if sha(protocol_bytes) != PROTOCOL_SHA256:
            raise ValueError('prespecified protocol changed')
        protocol = json.loads(protocol_bytes)
        write(out / 'protocol.json', protocol)
        allowed, pins = load_scope()
        ledger.append('ADMISSION_PASSED', races=331, runners=len(allowed), protected_decodes=0)
        expected = {**json.loads((ROOT / 'docs/research/market_explanation_20260929_inputs.json').read_text()),
                    **protocol['input_pins'], **pins}
        def read(path):
            data = path.read_bytes()
            if sha(data) != expected[str(path)]:
                raise ValueError('pinned input changed: ' + str(path))
            pins[str(path)] = sha(data)
            return data
        data = gate_lines(read(FOUNDATION / 'development.jsonl'), allowed)
        data.sort(key=lambda r: (r['race_date'], r['race_id'], r['box']))
        by_key = {(r['race_id'], r['box']): r for r in data}
        old_ledger = json.loads('[' + ','.join(read(Path('docs/research/offline_systematic_evidence/experiment_ledger.jsonl')).decode().splitlines()) + ']')
        trials = {r['outer']: r for r in old_ledger if r['event'] == 'VALIDATION_TRIAL' and r.get('name') == 'base16'}
        prior_events = Counter(r['event'] for r in old_ledger)
        write(out / 'novelty.json', {'prior_event_counts': dict(prior_events),
              'prior_questions': sorted({r.get('question', '') for r in old_ledger}),
              'prior_fixed_strengths': [.25, .5, 1],
              'distinction': 'one outcome-free race-wide support multiplier applied after fitting; prior interactions change fitted inputs, missing indicators are learned coefficients, fixed half uses .5 for every race, prior rule gates select predictions rather than normalize a support-scaled whole field',
              'exact_method_found_in_reviewed_ledger': False})
        selections, validation_records = [], []
        # Freeze every lambda from earlier OOF data before opening outer predictions.
        for outer in OUTER:
            oid = outer['name']
            identity_bytes = read(SOURCE / f'search_v1/{oid}_inner_identities.json')
            identities = json.loads(identity_bytes)  # This pinned file contains only identities.
            expected_keys = [(r['race_id'], r['box']) for end, start, stop in outer['inner'] for r in data if start <= r['race_date'] <= stop]
            keys = [(r['race_id'], r['box']) for r in identities]
            if keys != expected_keys or len(set(keys)) != len(keys):
                raise ValueError('inner prediction identity/order mismatch')
            if any(k not in allowed or by_key[k]['race_date'] != ident['race_date'] or ident['race_date'] >= outer['test_start'] for k, ident in zip(keys, identities)):
                raise ValueError('inner chronology/admission failure')
            trial = trials[oid]
            payload = read(SOURCE / 'search_v1' / trial['prediction_path'])
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                probability = archive['probability']
            if probability.shape != (len(keys),):
                raise ValueError('inner probability shape mismatch')
            validation = [{**by_key[k], 'full': float(p)} for k, p in zip(keys, probability)]
            full_score = np.mean([losses([r['y'] for r in rr], [r['full'] for r in rr])['ll'] for rr in grouped(validation)])
            if abs(full_score - trial['log_loss']) > 1e-12:
                raise ValueError('saved validation score mismatch')
            chosen, evaluated = select_parameter(validation, protocol['parameters'])
            for result in evaluated:
                ledger.append('VALIDATION_TRIAL', outer=oid, **result)
            selection = {'outer': oid, 'lambda': chosen, 'trials': evaluated,
                         'validation_first_date': min(r['race_date'] for r in validation),
                         'validation_last_date': max(r['race_date'] for r in validation),
                         'evaluation_first_date': outer['test_start'],
                         'inner_splits': outer['inner'],
                         'saved_full_log_loss': float(full_score)}
            selections.append(selection)
            ledger.append('SELECTION_FROZEN', **selection)
            for rr in grouped(validation):
                support = race_support(rr)
                candidate_predictions = {
                    str(parameter): adjust([r['market'] for r in rr], [r['full'] for r in rr],
                                          support / (support + parameter))
                    for parameter in protocol['parameters']}
                for index, r in enumerate(rr):
                    validation_records.append({**r, 'outer': oid, 'support': support,
                        'candidate_predictions': {key: float(p[index]) for key, p in candidate_predictions.items()}})
        write(out / 'selections.json', selections)
        outer_rows = gate_lines(read(SOURCE / 'search_v1/outer_predictions.jsonl'), allowed)
        if len(outer_rows) != 1251 or len({r['race_id'] for r in outer_rows}) != 177:
            raise ValueError('evaluation population changed')
        models = {outer['name']: json.loads(read(SOURCE / f"search_v1/{outer['name']}_models.json"))['base16'] for outer in OUTER}
        races, predictions = [], []
        error = 0.0
        for rr in grouped(outer_rows):
            oid = rr[0]['outer']
            fold = next(o for o in OUTER if o['name'] == oid)
            if any(r['outer'] != oid or not fold['test_start'] <= r['race_date'] <= fold['test_end'] for r in rr):
                raise ValueError('outer chronology mismatch')
            for r in rr:
                original = by_key[r['race_id'], r['box']]
                if any(r['features'][name] != original['features'][name] for name in FEATURES) or any(r[k] != original[k] for k in ('y', 'market', 'odds', 'capture', 'jump')):
                    raise ValueError('outer/foundation values changed')
            market = np.array([r['market'] for r in rr])
            full = np.array([r['predictions']['refit_base16'] for r in rr])
            half = np.array([r['predictions']['refit_half'] for r in rr])
            for calculated, retained in [(decomposition(rr, models[oid])['p'], full), (adjust(market, full, .5), half)]:
                error = max(error, float(np.max(np.abs(calculated - retained))))
            if error > 1e-12:
                raise ValueError('unchanged baseline replay failed')
            support = race_support(rr)
            parameter = next(s['lambda'] for s in selections if s['outer'] == oid)
            alpha = support / (support + parameter)
            probs = {'market': market, 'full': full, 'half': half, 'adaptive': adjust(market, full, alpha)}
            race = {'race_id': rr[0]['race_id'], 'race_date': rr[0]['race_date'], 'outer': oid,
                    'runners': len(rr), 'support': support, 'alpha': alpha,
                    **{name: losses([r['y'] for r in rr], p) for name, p in probs.items()}}
            races.append(race)
            for i, r in enumerate(rr):
                predictions.append({**r, 'support': support, 'alpha': alpha,
                                    'comparison_predictions': {name: float(p[i]) for name, p in probs.items()}})
        # Every prediction and choice is retained before aggregate later scoring.
        for name, rows in [('validation_inputs_predictions.jsonl', validation_records), ('evaluation_predictions.jsonl', predictions), ('development_inputs.jsonl', data)]:
            with (out / name).open('x') as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        write(out / 'retained_outer_models.json', {oid: {'model': model, 'identity': 'original_retained_base16_from_PR193',
              'training_membership': [{k: r[k] for k in ('race_id', 'race_date', 'box', 'dog_token')} for r in data if r['race_date'] <= next(o['val_end'] for o in OUTER if o['name'] == oid)]} for oid, model in models.items()})
        ledger.append('ALL_OUTER_PREDICTIONS_SEALED', runners=len(predictions), races=len(races), new_fits=0,
                      prediction_sha256=sha((out / 'evaluation_predictions.jsonl').read_bytes()))
        summary = metric_summary(races, protocol['uncertainty']['date_cluster_draws'], protocol['uncertainty']['seed'])
        summary['baseline_maximum_replay_error'] = error
        write(out / 'race_metrics.json', races)
        write(out / 'summary.json', summary)
        for path in (Path(__file__), PROTOCOL, ROOT / 'scripts/offline_systematic_search.py', ROOT / 'scripts/explain_market_residual.py'):
            pins[str(path)] = sha(path.read_bytes())
        write(out / 'input_hashes.json', pins)
        write(out / 'environment.json', {'python': sys.version, 'executable': sys.executable,
              'executable_sha256': sha(Path(sys.executable).read_bytes()), 'platform': platform.platform(),
              'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy')},
              'new_fits': 0, 'preprocessing': 'unchanged original outer receipts; inner decisions use retained OOF predictions, no missing original model reconstructed'})
        ledger.append('COMPLETE', new_fits=0, validation_variants=6, evaluation_procedures=4,
                      summary_sha256=sha((out / 'summary.json').read_bytes()))
        return summary
    except Exception as exc:
        ledger.append('FAILED', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.out)
    print(json.dumps({'overall': result['overall'], 'paired': result['paired']}, indent=2))
