#!/usr/bin/env python3
"""
Register the latest MLSystemV4 artifact into the Model Registry.
Creates a trivial scaler artifact so the registry loader can operate uniformly.
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from pathlib import Path as _P

import joblib

# Ensure project root is on sys.path
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

import os

from sklearn.preprocessing import FunctionTransformer

from model_registry import get_model_registry
from scripts.db_guard import db_guard

MODELS_DIR = Path("ml_models_v4")


def main():
    if not MODELS_DIR.exists():
        print(
            json.dumps({"success": False, "error": "ml_models_v4 directory not found"})
        )
        return

    model_files = sorted(
        MODELS_DIR.glob("ml_model_v4_*.joblib"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not model_files:
        print(json.dumps({"success": False, "error": "no v4 model artifacts found"}))
        return

    latest = model_files[0]
    data = joblib.load(latest)

    calibrated = data.get("calibrated_pipeline")
    feature_columns = data.get("feature_columns", [])
    model_info = data.get("model_info", {})

    # Prepare a trivial scaler (identity) so registry can load both model and scaler
    scaler = FunctionTransformer(validate=False)

    # Metrics and training info
    perf = {
        "accuracy": float(model_info.get("test_accuracy") or 0.0),
        "auc": float(model_info.get("test_auc") or 0.5),
        "f1_score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    training_info = {
        "training_samples": int(model_info.get("n_train_samples") or 0),
        "test_samples": int(model_info.get("n_test_samples") or 0),
        "validation_method": "temporal_split",
        "cv_scores": [],
        "is_ensemble": False,
        "ensemble_components": [],
        "data_quality_score": 0.5,
        "inference_time_ms": 0.0,
        "prediction_type": "win",
    }

    registry = get_model_registry()
    model_name = "V4_ExtraTrees"
    model_type = "CalibratedPipeline"

    # Guarded registration (pre-backup, post-validate)
    # Model registration is a write operation, use staging DB
    db_path = (
        os.getenv("STAGING_DB_PATH")
        or os.getenv("GREYHOUND_DB_PATH")
        or os.getenv("DATABASE_PATH")
        or "greyhound_racing_data_stage.db"
    )
    with db_guard(db_path=db_path, label="register_latest_v4_model") as guard:
        guard.expect_table_growth("ml_model_registry", min_delta=0)
        model_id = registry.register_model(
            model_obj=calibrated,
            scaler_obj=scaler,
            model_name=model_name,
            model_type=model_type,
            performance_metrics=perf,
            training_info=training_info,
            feature_names=feature_columns,
            hyperparameters={
                "trained_at": model_info.get("trained_at", datetime.now().isoformat()),
                "calibration_method": model_info.get("calibration_method", "isotonic"),
            },
            notes="Auto-registered from ml_models_v4 artifact",
        )

    print(
        json.dumps(
            {"success": True, "registered_model_id": model_id, "artifact": str(latest)}
        )
    )


if __name__ == "__main__":
    main()
