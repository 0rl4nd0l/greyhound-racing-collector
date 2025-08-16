#!/usr/bin/env python3
"""
Train a lightweight model for testing performance guardrails.
"""
import sys
import os
import time
import json
import numpy as np
import joblib

sys.path.insert(0, ".")

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

def main():
    # Create dummy training data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, random_state=42)
    
    # Train lightweight model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    start_time = time.time()
    model.fit(X, y)
    training_time = time.time() - start_time
    
    # Test prediction latency
    test_start = time.time()
    y_pred_prob = model.predict_proba(X[:100])
    prediction_latency = (time.time() - test_start) / 100  # Average per prediction
    
    # Calculate AUC (using one-vs-rest for multiclass)
    y_bin = label_binarize(y, classes=[0, 1, 2])
    auc = roc_auc_score(y_bin, y_pred_prob, multi_class="ovr")
    
    # Save model
    joblib.dump(model, "test_model.pkl")
    
    # Save metrics
    metrics = {
        "auc": float(auc),
        "prediction_latency": float(prediction_latency),
        "training_time": float(training_time)
    }
    
    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model trained successfully:")
    print(f"AUC: {auc:.4f}")
    print(f"Prediction latency: {prediction_latency:.4f}s")
    print(f"Training time: {training_time:.2f}s")

if __name__ == "__main__":
    main()
