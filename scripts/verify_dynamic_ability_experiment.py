#!/usr/bin/env python3
"""No-fit numerical replay and independent metric arithmetic for saved trials."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scripts import run_dynamic_ability_experiment as experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    hashes = json.loads((run / 'artifacts.sha256.json').read_text())
    for name, sha in hashes.items():
        experiment.read_checked(run / name, sha)
    groups, events, _ = experiment.load(experiment.DEFAULT_CONTEXT)
    target = {(g['metadata']['source_race_key'], r['metadata']['guide_box']): int(r['target']['is_winner'])
              for races in groups.values() for g in races for r in g['runners']}
    states = {}
    for line in (run / 'runner_states.jsonl').read_text().splitlines():
        row = json.loads(line)
        states[(row['source_race_key'], row['guide_box'])] = row['states']
    summary = json.loads((run / 'summary.json').read_text())
    replay, metrics, row_counts = {}, {}, {}
    for split in experiment.SPLITS:
        saved = [json.loads(line) for line in (run / (split + '_predictions.jsonl')).read_text().splitlines()]
        row_counts[split] = len(saved)
        metrics[split], replay[split] = {}, {}
        for arm in experiment.ability.ARMS:
            model = json.loads((run / ('model_' + arm + '.json')).read_text())
            rows = experiment.rows_for(groups[split], {key: s[arm] for key, s in states.items()})
            names = model['feature_names']
            prep = model['preprocessing']
            matrix = np.array([[row['features'][name] if row['features'][name] is not None else np.nan for name in names] for row in rows])
            matrix = np.c_[np.where(np.isfinite(matrix), matrix, prep['medians']), ~np.isfinite(matrix)]
            matrix = (matrix - prep['means']) / prep['standard_deviations']
            groups_index = {}
            for i, row in enumerate(rows):
                assert (row['source_race_key'], row['guide_box']) == (saved[i]['source_race_key'], saved[i]['guide_box'])
                groups_index.setdefault(row['source_race_key'], []).append(i)
            predicted = np.empty(len(rows))
            for indices in groups_index.values():
                part = matrix[indices]
                logits = (part - part.mean(axis=0)) @ np.array(model['coefficients'])
                p = np.exp(logits - logits.max())
                predicted[indices] = p / p.sum()
            error = float(max(abs(predicted[i] - row['probabilities'][arm]) for i, row in enumerate(saved)))
            assert error < 1e-12
            replay[split][arm] = error
        for arm in saved[0]['probabilities']:
            buckets = {}
            for row in saved:
                buckets.setdefault(row['source_race_key'], []).append(row)
            logloss, brier, correct = [], [], []
            for rows in buckets.values():
                p = [r['probabilities'][arm] for r in rows]
                y = [target[(r['source_race_key'], r['guide_box'])] for r in rows]
                assert abs(sum(p) - 1) < 1e-12 and all(0 < v < 1 for v in p) and sum(y) == 1
                logloss.append(-math.log(p[y.index(1)]))
                brier.append(sum((a-b)**2 for a, b in zip(p, y)))
                selected = [i for i, v in enumerate(p) if v == max(p)]
                correct.append(sum(y[i] for i in selected) / len(selected))
            measured = {'log_loss': sum(logloss)/len(logloss), 'brier': sum(brier)/len(brier), 'top1': sum(correct)/len(correct)}
            assert all(abs(v - summary['metrics'][split][arm][k]) < 1e-12 for k, v in measured.items())
            metrics[split][arm] = measured
    baseline = experiment.DEFAULT_BASELINE / 'artifacts.sha256.json'
    experiment.write(args.output, {'state': 'VERIFIED', 'new_fits': 0, 'test_labels_opened': False,
                                   'verified_artifacts': len(hashes), 'max_probability_replay_error': replay,
                                   'independent_arithmetic_metrics': metrics, 'runner_counts': row_counts,
                                   'frozen_baseline_artifact_manifest': {'path': str(baseline), 'sha256': experiment.digest(baseline.read_bytes())},
                                   'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps({'state': 'VERIFIED', 'new_fits': 0, 'max_error': max(e for s in replay.values() for e in s.values())}))


if __name__ == '__main__':
    main()
