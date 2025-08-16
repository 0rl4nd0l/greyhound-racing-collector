import argparse
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Per project rules, use pipe-delimited files. This script aggregates one or more .psv files.
# Expected columns per row (dog-level):
#   race_id | dog_clean_name | win_prob_norm | predicted_rank | is_winner
# Optional columns: place_prob_norm, odds, actual_place

REQUIRED_COLUMNS = ['race_id', 'dog_clean_name', 'win_prob_norm', 'predicted_rank', 'is_winner']


def load_frames(paths):
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p, sep='|')
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                print(f"Skipping {p}: missing columns {missing}")
                continue
            frames.append(df)
        except Exception as e:
            print(f"Failed to read {p}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def top1_accuracy(df: pd.DataFrame) -> float:
    if df.empty:
        return float('nan')
    # Determine predicted winner per race by max win_prob_norm, tie-break by lowest rank
    pred_top = df.sort_values(['race_id', 'win_prob_norm', 'predicted_rank'], ascending=[True, False, True]) \
                .groupby('race_id').head(1)
    return float((pred_top['is_winner'] == 1).mean())


def precision_recall(df: pd.DataFrame, threshold: float = 0.5):
    if df.empty:
        return float('nan'), float('nan')
    y_true = (df['is_winner'] == 1).astype(int)
    y_pred = (df['win_prob_norm'] >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else float('nan')
    recall = tp / (tp + fn) if (tp + fn) else float('nan')
    return precision, recall


def brier_score(df: pd.DataFrame) -> float:
    if df.empty:
        return float('nan')
    y_true = (df['is_winner'] == 1).astype(int)
    p = df['win_prob_norm'].astype(float).clip(0, 1)
    return float(np.mean((p - y_true) ** 2))


def log_loss_metric(df: pd.DataFrame) -> float:
    if df.empty:
        return float('nan')
    eps = 1e-15
    y_true = (df['is_winner'] == 1).astype(int)
    p = df['win_prob_norm'].astype(float).clip(eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def reliability_bins(df: pd.DataFrame, bins: int = 10):
    if df.empty:
        return []
    df = df.copy()
    df['bin'] = pd.cut(df['win_prob_norm'], bins=bins, labels=False, include_lowest=True)
    out = []
    for b, g in df.groupby('bin'):
        if len(g) == 0:
            continue
        mean_p = float(g['win_prob_norm'].mean())
        emp_rate = float((g['is_winner'] == 1).mean())
        out.append({'bin': int(b), 'avg_pred': mean_p, 'emp_rate': emp_rate, 'n': int(len(g))})
    return out


def main():
    parser = argparse.ArgumentParser(description='Analyze MLSystemV4 prediction accuracy and calibration')
    parser.add_argument('--glob', dest='pattern', default='predictions/*.psv', help='Glob for prediction files')
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"No files matched pattern: {args.pattern}")
        return

    df = load_frames(files)
    if df.empty:
        print("No valid data loaded. Ensure files contain required columns:", REQUIRED_COLUMNS)
        return

    # Compute metrics
    acc = top1_accuracy(df)
    prec, rec = precision_recall(df)
    brier = brier_score(df)
    ll = log_loss_metric(df)
    calib = reliability_bins(df)

    report = {
        'files': files,
        'num_rows': int(len(df)),
        'num_races': int(df['race_id'].nunique()),
        'metrics': {
            'top1_accuracy': acc,
            'precision_winner_threshold_0_5': prec,
            'recall_winner_threshold_0_5': rec,
            'brier_score': brier,
            'log_loss': ll
        },
        'calibration_bins': calib
    }

    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

