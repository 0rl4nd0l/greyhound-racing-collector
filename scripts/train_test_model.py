#!/usr/bin/env python3
"""
Train a lightweight model for testing performance guardrails.
"""
import json
import os
import sys
import time

import joblib
import numpy as np

sys.path.insert(0, ".")

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


def main():
    # Guard: synthetic training is disabled by default. Enable for dev only.
    if os.getenv("ALLOW_SYNTHETIC_TEST_MODEL", "0").lower() not in ("1", "true", "yes"):
        print(
            "This script uses synthetic data and is disabled by default. Set ALLOW_SYNTHETIC_TEST_MODEL=1 to run.",
            file=sys.stderr,
        )
        sys.exit(2)
    # Create dummy training data (dev only) - binary classification for greyhound racing
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

    # Train lightweight model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    start_time = time.time()
    model.fit(X, y)
    training_time = time.time() - start_time

    # Test prediction latency
    test_start = time.time()
    y_pred_prob_subset = model.predict_proba(X[:100])
    prediction_latency = (time.time() - test_start) / 100  # Average per prediction

    # Calculate AUC for binary classification using full dataset
    y_pred_prob = model.predict_proba(X)
    auc = roc_auc_score(y, y_pred_prob[:, 1])

    # Save model
    joblib.dump(model, "test_model.pkl")

    # Save metrics
    metrics = {
        "auc": float(auc),
        "prediction_latency": float(prediction_latency),
        "training_time": float(training_time),
    }

    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model trained successfully:")
    print(f"AUC: {auc:.4f}")
    print(f"Prediction latency: {prediction_latency:.4f}s")
    print(f"Training time: {training_time:.2f}s")


if __name__ == "__main__":
    main()
