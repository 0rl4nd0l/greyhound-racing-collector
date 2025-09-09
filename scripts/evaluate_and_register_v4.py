#!/usr/bin/env python3
"""
Evaluate latest V4 model artifact and register it with real metrics.

- Loads latest ml_models_v4/ml_model_v4_*.joblib
- Builds leakage-safe features via MLSystemV4 on a time-ordered test split
- Computes accuracy, ROC AUC, Brier, and per-race top1_rate
- Registers calibrated pipeline in Model Registry with identity scaler
- Prints a single JSON line with the registration result
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
import sys

# Ensure project root on sys.path so imports like ml_system_v4 resolve when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.preprocessing import FunctionTransformer

from ml_system_v4 import MLSystemV4
from model_registry import get_model_registry


MODELS_DIR = Path("ml_models_v4")


def _latest_artifact() -> Path | None:
    if not MODELS_DIR.exists():
        return None
    files = sorted(MODELS_DIR.glob("ml_model_v4_*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _compute_top1(test_features: pd.DataFrame, y_true: pd.Series, y_prob: np.ndarray) -> tuple[int, int, float]:
    try:
        if "race_id" not in test_features.columns:
            return 0, 0, 0.0
        df_eval = pd.DataFrame({
            "race_id": test_features["race_id"].values,
            "y": y_true.values,
            "p": y_prob,
        })
        grouped = df_eval.groupby("race_id", sort=False)
        races_eval = int(grouped.ngroups)
        if races_eval == 0:
            return 0, 0, 0.0
        hits_series = grouped.apply(lambda g: int(g.loc[g["p"].idxmax(), "y"] == 1)).astype(int)
        top1_hits = int(hits_series.sum())
        top1_rate = float(top1_hits / races_eval) if races_eval > 0 else 0.0
        return top1_hits, races_eval, top1_rate
    except Exception:
        return 0, 0, 0.0


def main() -> int:
    out = {
        "success": False,
        "error": None,
        "artifact": None,
        "registered_model_id": None,
        "metrics": {},
    }
    try:
        latest = _latest_artifact()
        if not latest:
            out["error"] = "no V4 artifacts found"
            print(json.dumps(out))
            return 0
        out["artifact"] = str(latest)

        data = joblib.load(latest)
        calibrated = data.get("calibrated_pipeline")
        feature_columns = list(data.get("feature_columns", []) or [])
        model_info = dict(data.get("model_info", {}) or {})
        if calibrated is None:
            out["error"] = "artifact missing calibrated_pipeline"
            print(json.dumps(out))
            return 0

        # Prepare time-ordered splits and leakage-safe features
        system = MLSystemV4()
        try:
            raw_train_df, raw_test_df = system.prepare_time_ordered_data()
        except TypeError:
            raw_train_df, raw_test_df = system.prepare_time_ordered_data()
        if raw_train_df is None or raw_test_df is None or raw_train_df.empty or raw_test_df.empty:
            out["error"] = "prepare_time_ordered_data returned empty splits"
            print(json.dumps(out))
            return 0

        train_features = system.build_leakage_safe_features(raw_train_df)
        test_features = system.build_leakage_safe_features(raw_test_df)
        if (
            train_features is None
            or test_features is None
            or train_features.empty
            or test_features.empty
        ):
            out["error"] = "Feature building produced empty frames"
            print(json.dumps(out))
            return 0

        # Build matrices (mirror conventions across scripts)
        drop_cols = ["race_id", "dog_clean_name", "target", "target_timestamp"]
        # Pre-inject a few known optional categoricals that pipelines sometimes expect
        for col in ("trainer_name",):
            if col not in test_features.columns:
                test_features[col] = ""
        X_train = train_features.drop([c for c in drop_cols if c in train_features.columns], axis=1)
        y_train = train_features["target"].astype(int)
        X_test = test_features.drop([c for c in drop_cols if c in test_features.columns], axis=1)
        y_test = test_features["target"].astype(int)

        # Ensure any required columns expected by the stored preprocessor (and saved feature_columns) exist
        try:
            pre = None
            required_num: set[str] = set()
            required_cat: set[str] = set()
            # CalibratedClassifierCV exposes the fitted pipeline as base_estimator_
            if hasattr(calibrated, "base_estimator_"):
                pre = calibrated.base_estimator_.named_steps.get("preprocessor")
            elif hasattr(calibrated, "named_steps"):
                pre = calibrated.named_steps.get("preprocessor")
            if pre is not None and hasattr(pre, "transformers_"):
                for name, _trans, cols in pre.transformers_:
                    try:
                        cols_list = list(cols) if isinstance(cols, (list, tuple, set)) else []
                    except Exception:
                        cols_list = []
                    if name == "num":
                        required_num |= set(cols_list)
                    elif name == "cat":
                        required_cat |= set(cols_list)
                # Also consider saved feature_columns from the artifact
                saved_feats = set(feature_columns or [])
                # Heuristic: if a saved feature appears as string-like (likely categorical), treat as cat when unknown
                for col in saved_feats:
                    if col not in required_num and col not in required_cat:
                        required_cat.add(col)
                # Add missing categorical columns as empty strings
                for col in sorted(required_cat):
                    if col not in X_test.columns:
                        X_test[col] = ""
                # Add missing numerical columns as NaN
                for col in sorted(required_num):
                    if col not in X_test.columns:
                        X_test[col] = np.nan
                # Reorder columns to put required ones first where possible
                prefer = list(required_num) + list(required_cat)
                remain = [c for c in X_test.columns if c not in prefer]
                X_test = X_test[prefer + remain] if prefer else X_test
        except Exception:
            # Best effort: continue even if we can't introspect
            pass

        # Predict with stored calibrated pipeline
        try:
            y_pred = calibrated.predict(X_test)
        except Exception:
            # Fallback to threshold if predict() unavailable
            y_pred = (calibrated.predict_proba(X_test)[:, 1] > 0.5).astype(int)
        y_prob = (
            calibrated.predict_proba(X_test)[:, 1]
            if hasattr(calibrated, "predict_proba")
            else y_pred.astype(float)
        )

        # Metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "auc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.5,
            "brier": float(brier_score_loss(y_test, y_prob)),
            "f1": float(f1_score(y_test, y_pred)) if len(np.unique(y_test)) > 1 else 0.0,
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "evaluated_at": datetime.now().isoformat(),
        }
        top1_hits, races_eval, top1_rate = _compute_top1(test_features, y_test, y_prob)
        metrics.update({
            "top1_hits": int(top1_hits),
            "races_evaluated": int(races_eval),
            "top1_rate": float(top1_rate),
        })

        # Register with Model Registry
        registry = get_model_registry()
        model_name = "V4_ExtraTrees" if "ExtraTrees" in str(model_info.get("model_type", "")) else "V4_Model"
        model_type = "CalibratedPipeline"
        perf_payload = {
            "accuracy": metrics["accuracy"],
            "auc": metrics["auc"],
            "f1_score": metrics.get("f1", 0.0),
            "precision": 0.0,
            "recall": 0.0,
            # winner-hit metrics supported by registry
            "correct_winners": int(top1_hits),
            "races_evaluated": int(races_eval),
            "top1_rate": float(top1_rate),
        }
        training_info = {
            "training_samples": metrics["training_samples"],
            "test_samples": metrics["test_samples"],
            "validation_method": "temporal_split",
            "cv_scores": [],
            "is_ensemble": False,
            "ensemble_components": [],
            "data_quality_score": 0.5,
            "inference_time_ms": 0.0,
            "prediction_type": "win",
            "correct_winners": int(top1_hits),
            "races_evaluated": int(races_eval),
            "top1_rate": float(top1_rate),
        }
        model_id = registry.register_model(
            model_obj=calibrated,
            scaler_obj=FunctionTransformer(validate=False),
            model_name=model_name,
            model_type=model_type,
            performance_metrics=perf_payload,
            training_info=training_info,
            feature_names=feature_columns if feature_columns else list(X_train.columns),
            hyperparameters={
                "trained_at": model_info.get("trained_at"),
                "calibration_method": model_info.get("calibration_method", "isotonic"),
            },
            notes=f"Evaluated from artifact {latest.name}",
        )

        out.update({
            "success": True,
            "registered_model_id": model_id,
            "metrics": metrics,
        })
        print(json.dumps(out))
        return 0

    except Exception as e:
        out["success"] = False
        out["error"] = str(e)
        print(json.dumps(out))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

