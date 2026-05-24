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


def evaluate_model_top1(model: Any, feature_names: Optional[list[str]] = None, scaler_path: Optional[str] = None, scaler_obj: Optional[Any] = None, debug_log: Optional[list] = None) -> Optional[Dict[str, Any]]:
    try:
        sys_v4 = MLSystemV4()
        # Honor V4_MAX_RACES if set (ml_system honors internally)
        try:
            train_df, test_df = sys_v4.prepare_time_ordered_data()
        except TypeError:
            train_df, test_df = sys_v4.prepare_time_ordered_data()
        if train_df is None or test_df is None or test_df.empty:
            if debug_log is not None:
                debug_log.append({"phase": "load_data", "error": "empty_or_none"})
            return None
        # Build leakage-safe features to ensure 'target' exists and columns align
        try:
            train_features = sys_v4.build_leakage_safe_features(train_df)
            test_features = sys_v4.build_leakage_safe_features(test_df)
        except Exception as e:
            if debug_log is not None:
                debug_log.append({"phase": "build_leakage_safe_features", "error": str(e)})
            return None
        if train_features is None or test_features is None or test_features.empty:
            if debug_log is not None:
                debug_log.append({"phase": "build_leakage_safe_features", "error": "empty_or_none"})
            return None
        drop_cols = ["race_id", "dog_clean_name", "target", "target_timestamp"]
        X_test = test_features.drop([c for c in drop_cols if c in test_features.columns], axis=1)
        y_test = test_features["target"].astype(int)
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
        except Exception as e:
            if debug_log is not None:
                debug_log.append({"phase": "introspect_model_preprocessor", "error": str(e)})
        # Also introspect scaler_obj when available
        try:
            pre2 = scaler_obj
            # Direct ColumnTransformer
            if pre2 is not None and hasattr(pre2, "transformers_"):
                for name, _trans, cols in pre2.transformers_:
                    try:
                        cols_list = list(cols) if isinstance(cols, (list, tuple, set)) else []
                    except Exception:
                        cols_list = []
                    if name == "num":
                        required_num |= set(cols_list)
                    elif name == "cat":
                        required_cat |= set(cols_list)
            # Pipeline wrapper
            if pre2 is not None and hasattr(pre2, "named_steps"):
                try:
                    for step_name, step_obj in pre2.named_steps.items():
                        if hasattr(step_obj, "transformers_"):
                            for name, _trans, cols in step_obj.transformers_:
                                try:
                                    cols_list = list(cols) if isinstance(cols, (list, tuple, set)) else []
                                except Exception:
                                    cols_list = []
                                if name == "num":
                                    required_num |= set(cols_list)
                                elif name == "cat":
                                    required_cat |= set(cols_list)
                except Exception as _e:
                    if debug_log is not None:
                        debug_log.append({"phase": "introspect_scaler_pipeline", "error": str(_e)})
            # Also inspect scaler_path artifact
            if scaler_path:
                try:
                    from pathlib import Path as _Path
                    if _Path(scaler_path).exists():
                        pre3 = joblib.load(scaler_path)
                        if hasattr(pre3, "transformers_"):
                            for name, _trans, cols in pre3.transformers_:
                                try:
                                    cols_list = list(cols) if isinstance(cols, (list, tuple, set)) else []
                                except Exception:
                                    cols_list = []
                                if name == "num":
                                    required_num |= set(cols_list)
                                elif name == "cat":
                                    required_cat |= set(cols_list)
                        if hasattr(pre3, "named_steps"):
                            for step_name, step_obj in pre3.named_steps.items():
                                if hasattr(step_obj, "transformers_"):
                                    for name, _trans, cols in step_obj.transformers_:
                                        try:
                                            cols_list = list(cols) if isinstance(cols, (list, tuple, set)) else []
                                        except Exception:
                                            cols_list = []
                                        if name == "num":
                                            required_num |= set(cols_list)
                                        elif name == "cat":
                                            required_cat |= set(cols_list)
                except Exception as _e:
                    if debug_log is not None:
                        debug_log.append({"phase": "introspect_scaler_path", "error": str(_e)})
            if debug_log is not None:
                debug_log.append({
                    "phase": "introspect_scaler_obj",
                    "num_cols": sorted(list(required_num))[:50],
                    "cat_cols": sorted(list(required_cat))[:50]
                })
        except Exception as e:
            if debug_log is not None:
                debug_log.append({"phase": "introspect_scaler_obj", "error": str(e)})
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
        # Normalize boolean-like strings across all object dtype columns first
        try:
            obj_cols = [c for c in X_test.columns if X_test[c].dtype == object]
            for col in obj_cols:
                try:
                    X_test[col] = (
                        X_test[col]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .replace({"true": "1", "false": "0"})
                    )
                except Exception:
                    pass
            if debug_log is not None:
                debug_log.append({"phase": "boolean_normalization", "columns": obj_cols[:50]})
        except Exception as e:
            if debug_log is not None:
                debug_log.append({"phase": "boolean_normalization", "error": str(e)})
        # Heuristic numeric coercion: only coerce columns that are > 95% numeric after boolean normalization
        try:
            candidate_cols = [c for c in X_test.columns if X_test[c].dtype == object]
            coerced = []
            bad_values = {}
            for col in candidate_cols:
                tmp = pd.to_numeric(X_test[col], errors="coerce")
                non_na_ratio = float(tmp.notna().mean()) if len(tmp) > 0 else 0.0
                if non_na_ratio >= 0.95:
                    # Record a small sample of previously non-numeric values
                    mask_bad = tmp.isna() & X_test[col].notna()
                    if bool(mask_bad.any()):
                        bad_values[col] = list(X_test.loc[mask_bad, col].astype(str).unique()[:5])
                    X_test[col] = tmp
                    coerced.append(col)
            if debug_log is not None:
                entry = {"phase": "heuristic_numeric_coercion", "columns": coerced[:50]}
                if bad_values:
                    entry["bad_values"] = bad_values
                debug_log.append(entry)
        except Exception as e:
            if debug_log is not None:
                debug_log.append({"phase": "heuristic_numeric_coercion", "error": str(e)})
        # Coerce numeric columns using MLSystemV4's configured numerical_columns + preprocessor introspection
        try:
            sys_nums = set(getattr(sys_v4, "numerical_columns", []) or [])
            inferred_nums = set(required_num)
            num_cols = (sys_nums | inferred_nums) & set(X_test.columns)
            cleaned = []
            bad_values = {}
            for col in list(num_cols):
                if X_test[col].dtype == object:
                    # Normalize boolean-like strings to numeric
                    try:
                        X_test[col] = (
                            X_test[col]
                            .astype(str)
                            .str.strip()
                            .str.lower()
                            .replace({"true": "1", "false": "0"})
                        )
                    except Exception:
                        pass
                # Find non-numeric samples
                try:
                    tmp = pd.to_numeric(X_test[col], errors="coerce")
                    non_numeric_mask = tmp.isna() & X_test[col].notna()
                    if bool(non_numeric_mask.any()):
                        # capture a small sample of offending values
                        samples = list(X_test.loc[non_numeric_mask, col].astype(str).unique()[:5])
                        bad_values[col] = samples
                    X_test[col] = tmp
                except Exception:
                    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
                cleaned.append(col)
            if debug_log is not None:
                dbg_entry = {"phase": "numeric_coercion", "columns": sorted(list(num_cols))}
                if bad_values:
                    dbg_entry["bad_values"] = bad_values
                debug_log.append(dbg_entry)
        except Exception as e:
            if debug_log is not None:
                debug_log.append({"phase": "numeric_coercion", "error": str(e)})
        # If feature_names are provided, ensure all present and reorder columns accordingly
        if feature_names:
            for col in feature_names:
                if col not in X_test.columns:
                    # Default fill: try to keep type stable (strings for cats, NaN for nums if known)
                    X_test[col] = "" if col in required_cat else np.nan
            try:
                X_test = X_test[[c for c in feature_names]]
            except Exception as e:
                if debug_log is not None:
                    debug_log.append({"phase": "reorder_to_feature_names", "error": str(e)})

        def _predict_with_optional_scaler(mdl: Any, X: pd.DataFrame) -> np.ndarray:
            # Try direct
            if hasattr(mdl, "predict_proba"):
                try:
                    return mdl.predict_proba(X)[:, 1]
                except Exception as e:
                    if debug_log is not None:
                        debug_log.append({"phase": "direct_predict_proba", "error": str(e), "shape": [len(X), len(X.columns)]})
            # Try with provided scaler/preprocessor object
            if scaler_obj is not None and hasattr(scaler_obj, "transform"):
                try:
                    Xtx = scaler_obj.transform(X)
                    if hasattr(mdl, "predict_proba"):
                        return mdl.predict_proba(Xtx)[:, 1]
                    y = mdl.predict(Xtx)
                    return y.astype(float) if isinstance(y, np.ndarray) else np.asarray(y, dtype=float)
                except Exception as e:
                    if debug_log is not None:
                        debug_log.append({"phase": "scaler_obj_transform", "error": str(e)})
            # Try with scaler/preprocessor from path
            if scaler_path:
                try:
                    from pathlib import Path as _Path
                    if _Path(scaler_path).exists():
                        preproc = joblib.load(scaler_path)
                        if hasattr(preproc, "transform"):
                            Xtx = preproc.transform(X)
                            if hasattr(mdl, "predict_proba"):
                                return mdl.predict_proba(Xtx)[:, 1]
                            y = mdl.predict(Xtx)
                            return y.astype(float) if isinstance(y, np.ndarray) else np.asarray(y, dtype=float)
                except Exception as e:
                    if debug_log is not None:
                        debug_log.append({"phase": "scaler_path_transform", "error": str(e), "scaler_path": scaler_path})
            # Fallback: plain predict
            try:
                y = mdl.predict(X)
                return y.astype(float) if isinstance(y, np.ndarray) else np.asarray(y, dtype=float)
            except Exception as e:
                if debug_log is not None:
                    debug_log.append({"phase": "direct_predict", "error": str(e)})
                raise

        # Probabilities
        y_prob = _predict_with_optional_scaler(model, X_test)
        hits, races, rate = _compute_top1(test_df, y_test, y_prob)
        return {"correct_winners": hits, "races_evaluated": races, "top1_rate": rate}
    except Exception as e:
        if debug_log is not None:
            debug_log.append({"phase": "evaluate_model_top1", "error": str(e)})
        return None

        # Probabilities
        y_prob = _predict_with_optional_scaler(model, X_test)
        hits, races, rate = _compute_top1(test_df, y_test, y_prob)
        return {"correct_winners": hits, "races_evaluated": races, "top1_rate": rate}
    except Exception:
        return None


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Backfill top1_rate/correct_winners for Model Registry models")
    parser.add_argument("-f", "--force", action="store_true", help="Re-evaluate even if a model already has a non-zero top1_rate")
    parser.add_argument("-m", "--model-id", dest="model_id", default=None, help="Limit backfill to a single model_id")
    parser.add_argument("--prediction-type", default="win", help="Prediction type to promote under (default: win)")
    parser.add_argument(
        "--policy",
        default=None,
        help="Override selection policy after backfill (default: current registry policy)",
    )
    args = parser.parse_args(argv)

    reg = get_model_registry()
    updated = []
    skipped = []

    def _evaluate_optimized(model_obj: Any, debug_log: Optional[list] = None) -> Optional[Dict[str, Any]]:
        if OptimizedTrainingPipeline is None:
            if debug_log is not None:
                debug_log.append({"phase": "optimized_pipeline", "error": "not_available"})
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
        except Exception as e:
            if debug_log is not None:
                debug_log.append({"phase": "optimized_pipeline_eval", "error": str(e)})
            return None

    debug_bundle: Dict[str, Any] = {}
    for model_id, data in list(getattr(reg, "model_index", {}).items()):
        if not isinstance(data, dict) or not data.get("is_active", True):
            continue
        if args.model_id and model_id != args.model_id:
            continue
        try:
            md = ModelMetadata(**data)
        except Exception:
            skipped.append({"model_id": model_id, "reason": "metadata_error"})
            continue
        # Skip if already has a non-zero top1_rate (unless forcing)
        if not args.force and (md.top1_rate or 0.0) > 0:
            continue
        # Load artifacts (prefer registry-aware loader for paired model+scaler)
        model_obj = None
        scaler_obj = None
        dbg: list = []
        try:
            got = reg.get_model_by_id(model_id)
            if isinstance(got, tuple) and len(got) >= 2:
                model_obj, scaler_obj = got[0], got[1]
        except Exception as e:
            if dbg is not None:
                dbg.append({"phase": "registry_get_model_by_id", "error": str(e)})
            model_obj = None
        if model_obj is None:
            model_path = Path(md.model_file_path)
            if not model_path.exists():
                skipped.append({"model_id": model_id, "reason": "missing_artifact"})
                continue
            try:
                model_obj = joblib.load(model_path)
            except Exception as e:
                dbg.append({"phase": "artifact_load_error", "error": str(e)})
                skipped.append({"model_id": model_id, "reason": "load_error", "debug": dbg})
                continue
        # Evaluate (try MLSystemV4 features first, then optimized pipeline)
        res = evaluate_model_top1(
            model_obj,
            feature_names=md.feature_names,
            scaler_path=getattr(md, "scaler_file_path", None),
            scaler_obj=scaler_obj,
            debug_log=dbg,
        )
        if not res:
            # Heuristic: optimized models use 'V4_Optimized' name prefix
            if (md.model_name or "").lower().startswith("v4_optimized"):
                res = _evaluate_optimized(model_obj, debug_log=dbg)
        if not res:
            skipped.append({"model_id": model_id, "reason": "eval_failed", "debug": dbg})
            debug_bundle[model_id] = dbg
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

    # Save registry and refresh best per current (or requested) policy
    try:
        reg._save_registry()  # type: ignore[attr-defined]
        # Apply requested or existing policy to update symlinks
        policy = args.policy or (reg.config or {}).get("best_selection_metric", "top1_rate")
        reg.set_best_selection_policy(policy)
        reg.auto_promote_best_by_metric(policy, prediction_type=args.prediction_type)
    except Exception:
        pass

    print(
        json.dumps(
            {
                "success": True,
                "updated": updated,
                "skipped": skipped,
                "forced": bool(args.force),
                "filtered_model_id": args.model_id or None,
                "policy": policy,
                "debug": debug_bundle,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

