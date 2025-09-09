#!/usr/bin/env python3
"""
Backfill top1_rate and correct_winners for models that lack them in the Model Registry.

- Loads current registry index
- For each active model with top1_rate == 0, loads the model artifact and evaluates on
  MLSystemV4 test split to compute per-race top1 stats
- Updates registry index and per-model metadata JSON

Environment:
- V4_MAX_RACES: optional int to cap races for faster evaluation (recommended)

Outputs a JSON summary to stdout.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Ensure project root for imports
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_system_v4 import MLSystemV4
from model_registry import ModelMetadata, get_model_registry
# Try to import optimized training pipeline for feature parity evaluation
try:
    from scripts.train_optimized_v4 import OptimizedTrainingPipeline  # type: ignore
except Exception:
    OptimizedTrainingPipeline = None  # type: ignore


def _compute_top1(test_df: pd.DataFrame, y_true: pd.Series, y_prob: np.ndarray) -> Tuple[int, int, float]:
    if "race_id" not in test_df.columns:
        return 0, 0, 0.0
    df = pd.DataFrame({"race_id": test_df["race_id"].values, "y": y_true.values, "p": y_prob})
    grouped = df.groupby("race_id", sort=False)
    races_eval = int(grouped.ngroups)
    if races_eval == 0:
        return 0, 0, 0.0
    hits_series = grouped.apply(lambda g: int(g.loc[g["p"].idxmax(), "y"] == 1)).astype(int)
    hits = int(hits_series.sum())
    rate = float(hits / races_eval)
    return hits, races_eval, rate


def evaluate_model_top1(model: Any, feature_names: Optional[list[str]] = None) -> Optional[Dict[str, Any]]:
    try:
        sys_v4 = MLSystemV4()
        # Honor V4_MAX_RACES if set (ml_system honors internally)
        try:
            train_df, test_df = sys_v4.prepare_time_ordered_data()
        except TypeError:
            train_df, test_df = sys_v4.prepare_time_ordered_data()
        if train_df is None or test_df is None or test_df.empty:
            return None
        drop_cols = ["race_id", "dog_clean_name", "target", "target_timestamp"]
        X_test = test_df.drop([c for c in drop_cols if c in test_df.columns], axis=1)
        y_test = test_df["target"].astype(int)
        # Attempt to introspect preprocessor to derive required columns
        required_num: set[str] = set()
        required_cat: set[str] = set()
        try:
            pre = None
            if hasattr(model, "base_estimator_"):
                pre = getattr(model.base_estimator_, "named_steps", {}).get("preprocessor")
            elif hasattr(model, "named_steps"):
                pre = model.named_steps.get("preprocessor")
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
        except Exception:
            pass
        # Also include saved feature names if provided
        if feature_names:
            for col in feature_names:
                if col not in required_num and col not in required_cat:
                    required_cat.add(col)
        # Backfill columns
        for col in sorted(required_cat):
            if col not in X_test.columns:
                X_test[col] = ""
            X_test[col] = X_test[col].astype(str)
        for col in sorted(required_num):
            if col not in X_test.columns:
                X_test[col] = np.nan
        # Minimal defaults for common categoricals
        for col, default in {
            "venue": "UNKNOWN",
            "grade": "5",
            "track_condition": "Good",
            "weather": "Fine",
            "trainer_name": "Unknown",
        }.items():
            if col not in X_test.columns:
                X_test[col] = default
            X_test[col] = X_test[col].astype(str)
        # Probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y = model.predict(X_test)
            y_prob = y.astype(float) if isinstance(y, np.ndarray) else np.asarray(y, dtype=float)
        hits, races, rate = _compute_top1(test_df, y_test, y_prob)
        return {"correct_winners": hits, "races_evaluated": races, "top1_rate": rate}
    except Exception:
        return None


def main() -> int:
    reg = get_model_registry()
    updated = []
    skipped = []

    def _evaluate_optimized(model_obj: Any) -> Optional[Dict[str, Any]]:
        if OptimizedTrainingPipeline is None:
            return None
        try:
            # Use env or defaults for DB paths
            analytics_db = os.getenv("ANALYTICS_DB_PATH") or os.getenv("GREYHOUND_DB_PATH") or "greyhound_racing_data.db"
            staging_db = os.getenv("STAGING_DB_PATH") or os.getenv("GREYHOUND_DB_PATH") or "greyhound_racing_data_stage.db"
            pipe = OptimizedTrainingPipeline(analytics_db, staging_db)
            raw_train, raw_test = pipe.ml_system.prepare_time_ordered_data()
            if raw_train is None or raw_test is None or raw_test.empty:
                return None
            train_feats, test_feats = pipe.prepare_enhanced_features(raw_train, raw_test)
            if test_feats is None or test_feats.empty:
                return None
            drop_cols = ["race_id", "dog_clean_name", "target", "target_timestamp"]
            X_test = test_feats.drop([c for c in drop_cols if c in test_feats.columns], axis=1)
            y_test = test_feats["target"].astype(int)
            # Probabilities
            if hasattr(model_obj, "predict_proba"):
                y_prob = model_obj.predict_proba(X_test)[:, 1]
            else:
                y = model_obj.predict(X_test)
                y_prob = y.astype(float) if isinstance(y, np.ndarray) else np.asarray(y, dtype=float)
            hits, races, rate = _compute_top1(test_feats, y_test, y_prob)
            return {"correct_winners": hits, "races_evaluated": races, "top1_rate": rate}
        except Exception:
            return None
    for model_id, data in list(getattr(reg, "model_index", {}).items()):
        if not isinstance(data, dict) or not data.get("is_active", True):
            continue
        try:
            md = ModelMetadata(**data)
        except Exception:
            skipped.append({"model_id": model_id, "reason": "metadata_error"})
            continue
        # Skip if already has a non-zero top1_rate
        if (md.top1_rate or 0.0) > 0:
            continue
        # Load artifact
        model_path = Path(md.model_file_path)
        if not model_path.exists():
            skipped.append({"model_id": model_id, "reason": "missing_artifact"})
            continue
        try:
            model_obj = joblib.load(model_path)
        except Exception:
            skipped.append({"model_id": model_id, "reason": "load_error"})
            continue
        # Evaluate (try MLSystemV4 features first, then optimized pipeline)
        res = evaluate_model_top1(model_obj, feature_names=md.feature_names)
        if not res:
            # Heuristic: optimized models use 'V4_Optimized' name prefix
            if (md.model_name or "").lower().startswith("v4_optimized"):
                res = _evaluate_optimized(model_obj)
        if not res:
            skipped.append({"model_id": model_id, "reason": "eval_failed"})
            continue
        # Update index
        try:
            reg.model_index[model_id]["correct_winners"] = int(res["correct_winners"])  # type: ignore
            reg.model_index[model_id]["races_evaluated"] = int(res["races_evaluated"])  # type: ignore
            reg.model_index[model_id]["top1_rate"] = float(res["top1_rate"])  # type: ignore
            # Update per-model metadata JSON
            meta_path = reg.metadata_dir / f"{model_id}_metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                except Exception:
                    meta = {}
                meta.update({
                    "correct_winners": int(res["correct_winners"]),
                    "races_evaluated": int(res["races_evaluated"]),
                    "top1_rate": float(res["top1_rate"]),
                })
                meta_path.write_text(json.dumps(meta, indent=2))
            updated.append({"model_id": model_id, **res})
        except Exception:
            skipped.append({"model_id": model_id, "reason": "update_failed"})
            continue
    # Save registry and refresh best per current policy
    try:
        reg._save_registry()  # type: ignore[attr-defined]
        # Re-apply current policy to update symlinks
        policy = (reg.config or {}).get("best_selection_metric", "top1_rate")
        reg.set_best_selection_policy(policy)
        reg.auto_promote_best_by_metric(policy, prediction_type="win")
    except Exception:
        pass
    print(json.dumps({"success": True, "updated": updated, "skipped": skipped}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

