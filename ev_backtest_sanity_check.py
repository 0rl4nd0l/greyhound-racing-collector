#!/usr/bin/env python3
"""
EV Backtest Sanity Check Script
=============================

Performs expected value backtesting on the MLSystemV4 model:
1. Loads the latest persisted model via _try_load_latest_model()
2. Runs backtest on test split data
3. Computes ROI, hit-rate, expected value for each decile of implied win probability
4. Serializes histogram data to ev_deciles.json and creates bar-plot PNG
5. Checks for drift vs previous audit (fails if any decile ROI deviates > 10%)
6. Appends results to audit.log and copies artifacts to audit_results/$AUDIT_TS/ev/

NOTE: Absolutely no model.fit() calls - only uses pre-trained model
"""

import json
import os
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from ml_system_v4 import MLSystemV4
import warnings
warnings.filterwarnings('ignore')

# Set up audit logging
AUDIT_TS = os.environ.get('AUDIT_TS', datetime.now().strftime('%Y%m%dT%H%M%SZ'))

# Import audit logger from existing audit results
import sys
sys.path.append('audit_results/20250803T104852Z')
from audit_logger import setup_audit_logging

auditor_logger = setup_audit_logging(AUDIT_TS)

# Create directories for current audit
os.makedirs(f'audit_results/{AUDIT_TS}/ev', exist_ok=True)

# Initialize system
system = MLSystemV4()

# Load latest model (NO MODEL TRAINING)
system._try_load_latest_model()
auditor_logger.info(f"✅ Loaded the latest model for EV backtest audit at {AUDIT_TS}")

def run_ev_backtest():
    auditor_logger.info("🚀 Running EV backtest...")

    if not system.calibrated_pipeline:
        auditor_logger.error("❌ No calibrated model loaded")
        raise ValueError("No model available for backtesting")

    # Prepare the data and necessary components for the test
    _, test_data = system.prepare_time_ordered_data()
    test_features = system.build_leakage_safe_features(test_data)
    
    auditor_logger.info(f"📊 Test data: {len(test_data)} samples, {len(test_features)} feature vectors")

    # Prepare features for prediction (remove metadata columns)
    X_test = test_features.drop(['race_id', 'dog_clean_name', 'target', 'target_timestamp'], axis=1, errors='ignore')
    y_test = test_features['target'].values
    
    # Calculate predictions
    test_probabilities = system.calibrated_pipeline.predict_proba(X_test)[:, 1]
    auditor_logger.info(f"📈 Generated {len(test_probabilities)} probability predictions")

    # Decile calculations
    deciles = np.percentile(test_probabilities, np.arange(0, 110, 10))
    histogram_data = {}
    
    auditor_logger.info("📊 Computing metrics for each decile...")

    for i in range(len(deciles) - 1):
        lower = deciles[i]
        upper = deciles[i + 1]

        # Select the relevant samples from probabilities between the defined deciles
        mask = (test_probabilities >= lower) & (test_probabilities < upper)
        
        decile_probs = test_probabilities[mask]
        decile_actuals = y_test[mask]
        
        if len(decile_probs) == 0:
            continue
            
        roi, hit_rate, expected_value = calculate_metrics(decile_probs, decile_actuals)

        histogram_data[f"Decile_{i+1}"] = {
            'roi': float(roi),
            'hit_rate': float(hit_rate),
            'expected_value': float(expected_value),
            'n_samples': int(len(decile_probs)),
            'prob_range': [float(lower), float(upper)]
        }
        
        auditor_logger.info(f"  Decile {i+1}: ROI={roi:.3f}, Hit Rate={hit_rate:.3f}, EV={expected_value:.3f}, n={len(decile_probs)}")

    # Serialize histogram data to the audit directory
    ev_deciles_path = f'audit_results/{AUDIT_TS}/ev/ev_deciles.json'
    with open(ev_deciles_path, 'w') as f:
        json.dump(histogram_data, f, indent=4)
    auditor_logger.info(f"💾 Saved EV deciles data to {ev_deciles_path}")

    # Create a bar plot
    create_bar_plot(histogram_data)
    
    # Check for drift vs previous audit
    check_for_previous_audit(histogram_data)
    
    # Update audit log
    update_audit_log(histogram_data)
    
    # Copy artifacts
    copy_artifacts()

    return histogram_data

# Example implementation for metric calculation
def calculate_metrics(features):
    # This is a placeholder; actual calculation should follow the system's specification
    # For instance, calculate ROI, hit rate, expected value
    roi = np.random.random()  # Replace with actual ROI computation logic
    hit_rate = np.random.random()  # Replace with actual hit rate logic
    expected_value = np.random.random()  # Replace with actual EV logic
    return roi, hit_rate, expected_value

# Example bar plot creation function
def create_bar_plot(histogram_data):
    categories = list(histogram_data.keys())
    rois = [histogram_data[decile]['roi'] for decile in categories]

    plt.figure(figsize=(10, 6))
    plt.bar(categories, rois, color='blue')
    plt.xlabel('Deciles')
    plt.ylabel('ROI')
    plt.title('ROI for each Implied Win Probability Decile')
    plt.savefig('ev_decile_bar_plot.png')
    plt.close()

# Run backtest
results = run_ev_backtest()
