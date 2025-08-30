#!/usr/bin/env python3
"""
Train and register a V4-compatible Gradient Boosting model using leakage-safe features.

- Uses MLSystemV4 to prepare time-ordered data and build leakage-safe features
- Builds a sklearn pipeline with ColumnTransformer + OneHotEncoder + GradientBoostingClassifier
- Wraps in CalibratedClassifierCV (isotonic)
- Evaluates metrics and registers the calibrated pipeline in the Model Registry as V4_GradientBoosting (STAGING)

Usage:
  python scripts/train_register_v4_gb.py

Notes:
- Requires access to greyhound_racing_data.db
- If no data is available, exits gracefully
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss

from ml_system_v4 import MLSystemV4
from model_registry import get_model_registry

CONTRACT_PATH = Path("docs/model_contracts/V4_ExtraTrees_20250819.json")


def load_contract_features() -> list[str] | None:
    try:
        with open(CONTRACT_PATH, "r") as f:
            data = json.load(f)
        feats = data.get("features")
        if isinstance(feats, list) and feats:
            return feats
    except Exception:
        pass
    return None


def main() -> int:
    system = MLSystemV4()

    # Prepare data
    train_data, test_data = system.prepare_time_ordered_data()
    if train_data is None or test_data is None or train_data.empty or test_data.empty:
        print(json.dumps({
            "success": False,
            "error": "No data available for training (greyhound_racing_data.db missing or empty)",
        }))
        return 0

    # Build leakage-safe features
    train_features = system.build_leakage_safe_features(train_data)
    test_features = system.build_leakage_safe_features(test_data)

    if train_features is None or test_features is None or train_features.empty or test_features.empty:
        print(json.dumps({
            "success": False,
            "error": "Feature building returned empty result",
        }))
        return 0

    # Prepare X/y
    X_train = train_features.drop(['race_id', 'dog_clean_name', 'target', 'target_timestamp'], axis=1, errors='ignore')
    y_train = train_features['target']
    X_test = test_features.drop(['race_id', 'dog_clean_name', 'target', 'target_timestamp'], axis=1, errors='ignore')
    y_test = test_features['target']

    # Detect categorical/numeric columns (mirror MLSystemV4)
    categorical_columns = [c for c in X_train.columns if c in ['venue', 'grade', 'track_condition', 'weather', 'trainer_name']]
    exclude_columns = ['race_date', 'race_time']  # exclude non-numeric date/time strings
    numerical_columns = [
        c for c in X_train.columns
        if c not in categorical_columns + exclude_columns and pd.api.types.is_numeric_dtype(X_train[c])
    ]

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            (
                'num',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                ]),
                numerical_columns,
            ),
            (
                'cat',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ]),
                categorical_columns,
            ),
        ],
        remainder='drop',
    )
    try:
        preprocessor.set_output(transform='pandas')
    except Exception:
        pass

    # Base model and calibrated wrapper
    base = GradientBoostingClassifier(random_state=42)
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', base),
    ])
    calibrated = CalibratedClassifierCV(pipe, method='isotonic', cv=3)

    # Fit
    calibrated.fit(X_train, y_train)

    # Evaluate
    proba_tr = calibrated.predict_proba(X_train)[:, 1]
    proba_te = calibrated.predict_proba(X_test)[:, 1]

    # Compute top-1 winner hits on test split (per race)
    top1_hits = 0
    races_eval = 0
    top1_rate = 0.0
    try:
        if 'race_id' in test_features.columns:
            df_eval = pd.DataFrame({
                'race_id': test_features['race_id'].values,
                'y': y_test.values,
                'p': proba_te,
            })
            grouped = df_eval.groupby('race_id', sort=False)
            races_eval = int(grouped.ngroups)
            if races_eval > 0:
                # For each race, check if the top-probability dog actually won
                hits_series = grouped.apply(lambda g: int(g.loc[g['p'].idxmax(), 'y'] == 1)).astype(int)
                top1_hits = int(hits_series.sum())
                top1_rate = float(top1_hits / races_eval)
    except Exception:
        pass

    metrics = {
        'train_accuracy': float(accuracy_score(y_train, (proba_tr > 0.5).astype(int))),
        'test_accuracy': float(accuracy_score(y_test, (proba_te > 0.5).astype(int))),
        'train_auc': float(roc_auc_score(y_train, proba_tr)),
        'test_auc': float(roc_auc_score(y_test, proba_te)),
        'train_brier': float(brier_score_loss(y_train, proba_tr)),
        'test_brier': float(brier_score_loss(y_test, proba_te)),
        'trained_at': datetime.now().isoformat(),
        'top1_hits': int(top1_hits),
        'top1_rate': float(top1_rate),
        'races_evaluated': int(races_eval),
    }

    # Prepare registry metadata
    perf = {
        'accuracy': metrics['test_accuracy'],
        'auc': metrics['test_auc'],
        'f1_score': 0.0,
        'precision': 0.0,
        'recall': 0.0,
    }
    training_info = {
        'training_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'validation_method': 'temporal_split',
        'cv_scores': [],
        'is_ensemble': False,
        'ensemble_components': [],
        'data_quality_score': 0.5,
        'inference_time_ms': 0.0,
        'prediction_type': 'win',
        # Winner-hit metrics for registry selection by correct_winners
        'correct_winners': int(top1_hits),
        'races_evaluated': int(races_eval),
        'top1_rate': float(top1_rate),
    }

    # Feature names: prefer V4 contract for compatibility
    contract_features = load_contract_features()
    feature_names = contract_features if contract_features else list(X_train.columns)

    # Register
    registry = get_model_registry()
    model_id = registry.register_model(
        model_obj=calibrated,
        scaler_obj=FunctionTransformer(validate=False),
        model_name='V4_GradientBoosting',
        model_type='CalibratedPipeline',
        performance_metrics=perf,
        training_info=training_info,
        feature_names=feature_names,
        hyperparameters={'calibration_method': 'isotonic', 'base_model': 'GradientBoostingClassifier'},
        notes='V4_STAGING: GradientBoosting trained on leakage-safe features',
    )

    print(json.dumps({
        'success': True,
        'model_id': model_id,
        'metrics': metrics,
    }))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

