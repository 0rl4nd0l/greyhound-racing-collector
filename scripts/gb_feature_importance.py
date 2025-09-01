#!/usr/bin/env python3
import os
import sys
from datetime import datetime

import numpy as np

# Ensure we can import the trainer
sys.path.append(os.path.dirname(__file__))
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from ml_backtesting_trainer import MLBacktestingTrainer


def main():
    months_back = int(os.getenv("GB_MONTHS_BACK", "12"))
    trainer = MLBacktestingTrainer()

    # Load data and build dataset
    hist_df = trainer.load_historical_race_data(months_back=months_back)
    if hist_df is None or len(hist_df) < 200:
        print(
            f"Insufficient historical data ({0 if hist_df is None else len(hist_df)})"
        )
        return

    enhanced_df = trainer.create_enhanced_features(hist_df)
    ml_df, feature_columns = trainer.prepare_ml_dataset(enhanced_df)

    # Sort by time and split
    ml_df = ml_df.sort_values("race_date")
    split = int(0.8 * len(ml_df))
    train_df = ml_df.iloc[:split]
    test_df = ml_df.iloc[split:]

    X_train = train_df[feature_columns].values
    y_train = train_df["is_winner"].values
    X_test = test_df[feature_columns].values
    y_test = test_df["is_winner"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Use the best GB params found in optimization
    params = dict(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model = GradientBoostingClassifier(**params)
    model.fit(X_train_s, y_train)

    # Feature importances
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        order = np.argsort(imps)[::-1]
        print("Top 20 feature importances (Gradient Boosting):")
        for i in order[:20]:
            print(f"  {feature_columns[i]:<28} {imps[i]:.4f}")

    # Calibration diagnostics on test set
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test_s)[:, 1]
    else:
        # decision_function fallback normalized to 0-1
        scores = model.decision_function(X_test_s)
        vmin, vmax = np.min(scores), np.max(scores)
        proba = (scores - vmin) / (vmax - vmin + 1e-12)

    # Brier and log loss
    try:
        brier = brier_score_loss(y_test, proba)
    except Exception:
        brier = float("nan")
    try:
        ll = log_loss(y_test, proba, labels=[0, 1])
    except Exception:
        ll = float("nan")
    print(f"\nCalibration (test set): Brier={brier:.4f}  LogLoss={ll:.4f}")

    # Reliability table
    bins = [
        0.0,
        0.05,
        0.10,
        0.125,
        0.15,
        0.175,
        0.20,
        0.225,
        0.25,
        0.275,
        0.30,
        0.35,
        1.01,
    ]
    print("\nReliability (bin, n, avg_pred, emp_acc):")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (proba >= lo) & (proba < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        avg_p = float(proba[mask].mean())
        emp_acc = float(y_test[mask].mean()) if n > 0 else 0.0
        print(
            f"  [{lo:.3f},{hi:.3f})  n={n:4d}  avg_pred={avg_p:.3f}  emp_acc={emp_acc:.3f}"
        )


if __name__ == "__main__":
    main()
