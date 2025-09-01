#!/usr/bin/env python3
import os
import sys
import json
import glob
from collections import defaultdict, Counter
from datetime import datetime


def load_predictions(file_path: str):
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError:
                # skip broken lines
                continue
    return records


def latest_predictions_file(base_dir: str) -> str:
    files = sorted(glob.glob(os.path.join(base_dir, 'walk_forward_predictions_*.jsonl')))
    if not files:
        raise FileNotFoundError(f"No predictions JSONL found in {base_dir}")
    return files[-1]


def summarize(records, high_conf_thresholds=(0.6, 0.5, 0.4)):
    total = len(records)
    scorable = [r for r in records if r.get('actual_winner') is not None]
    unscorable = total - len(scorable)

    top1_acc = sum(1 for r in scorable if r.get('correct')) / len(scorable) if scorable else 0.0

    high_conf = {}
    for thr in high_conf_thresholds:
        subset = [r for r in scorable if float(r.get('predicted_prob', 0.0)) >= thr]
        wrong = [r for r in subset if not r.get('correct')]
        high_conf[thr] = {
            'count': len(subset),
            'wrong': len(wrong),
            'wrong_rate': (len(wrong) / len(subset)) if subset else 0.0,
        }

    # Calibration bins (finer in 0.05 steps up to 0.35, then 0.35+)
    bins = [0.0, 0.05, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30, 0.35, 1.01]
    bin_labels = [f"[{bins[i]:.3f},{bins[i+1]:.3f})" for i in range(len(bins)-1)]
    bin_counts = Counter()
    bin_correct = Counter()
    bin_avg_prob = defaultdict(float)

    for r in scorable:
        p = float(r.get('predicted_prob', 0.0))
        # find bin
        b_idx = None
        for i in range(len(bins)-1):
            if bins[i] <= p < bins[i+1]:
                b_idx = i
                break
        if b_idx is None:
            continue
        label = bin_labels[b_idx]
        bin_counts[label] += 1
        if r.get('correct'):
            bin_correct[label] += 1
        bin_avg_prob[label] += p

    calib_rows = []
    for label in bin_labels:
        n = bin_counts[label]
        if n == 0:
            continue
        avg_p = bin_avg_prob[label] / n
        acc = bin_correct[label] / n
        calib_rows.append({'bin': label, 'n': n, 'avg_pred': avg_p, 'emp_acc': acc})

    # Field size patterns
    def fs_bucket(fs):
        try:
            fs = int(fs)
        except Exception:
            return 'unknown'
        if fs <= 1:
            return '1'
        if fs == 2:
            return '2'
        if fs == 3:
            return '3'
        if 4 <= fs <= 6:
            return '4-6'
        if 7 <= fs <= 8:
            return '7-8'
        return '9+'

    fs_buckets = defaultdict(lambda: {'n': 0, 'acc': 0})
    for r in scorable:
        b = fs_bucket(r.get('field_size'))
        fs_buckets[b]['n'] += 1
        fs_buckets[b]['acc'] += 1 if r.get('correct') else 0

    fs_rows = []
    for b, stats in sorted(fs_buckets.items(), key=lambda x: x[0]):
        n = stats['n']
        acc = (stats['acc'] / n) if n else 0.0
        fs_rows.append({'field_size_bucket': b, 'n': n, 'top1_acc': acc})

    # Track code patterns (prefix of race_id)
    track_stats = defaultdict(lambda: {'n': 0, 'acc': 0})
    for r in scorable:
        rid = str(r.get('race_id', ''))
        tcode = rid.split('_', 1)[0] if '_' in rid else rid
        track_stats[tcode]['n'] += 1
        track_stats[tcode]['acc'] += 1 if r.get('correct') else 0

    track_rows = []
    for t, stats in sorted(track_stats.items(), key=lambda kv: -kv[1]['n'])[:10]:
        n = stats['n']
        acc = stats['acc'] / n if n else 0.0
        track_rows.append({'track': t, 'n': n, 'top1_acc': acc})

    return {
        'total': total,
        'scorable': len(scorable),
        'unscorable': unscorable,
        'top1_acc_scorable': top1_acc,
        'high_conf': high_conf,
        'calibration': calib_rows,
        'field_size': fs_rows,
        'tracks_top10': track_rows,
    }


def main():
    base_dir = os.path.join(os.getcwd(), 'predictions', 'backtests', 'walk_forward')
    file_path = sys.argv[1] if len(sys.argv) > 1 else latest_predictions_file(base_dir)
    print(f"Analyzing predictions file: {file_path}")
    records = load_predictions(file_path)
    result = summarize(records)

    # Pretty print
    print("\nSUMMARY:")
    print(f"  Total predictions: {result['total']}")
    print(f"  Scorable (actual_winner present): {result['scorable']}")
    print(f"  Unscorable (actual_winner missing): {result['unscorable']}")
    print(f"  Top-1 accuracy (scorable only): {result['top1_acc_scorable']:.3f}")

    print("\nHigh-confidence miss rates:")
    for thr, stats in result['high_conf'].items():
        print(f"  prob >= {thr:.2f}: count={stats['count']}, wrong={stats['wrong']}, wrong_rate={stats['wrong_rate']:.3f}")

    print("\nCalibration (bin, n, avg_pred, emp_acc):")
    for row in result['calibration']:
        print(f"  {row['bin']:>15}  n={row['n']:4d}  avg_pred={row['avg_pred']:.3f}  emp_acc={row['emp_acc']:.3f}")

    print("\nField size buckets (n, top1_acc):")
    for row in result['field_size']:
        print(f"  {row['field_size_bucket']:>4}: n={row['n']:4d}  acc={row['top1_acc']:.3f}")

    print("\nTop tracks by volume (n, top1_acc):")
    for row in result['tracks_top10']:
        print(f"  {row['track']:<6} n={row['n']:4d}  acc={row['top1_acc']:.3f}")


if __name__ == '__main__':
    main()
