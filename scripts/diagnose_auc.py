#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUC Diagnostics for MLSystemV4 (advanced implementation)
- Evaluates metrics under different calibration strategies
- Supports ExtraTrees and XGBoost with optional tuning to maximize ROC AUC
- Streams informative progress logs for the UI

Environment defaults:
  V4_DIAG_MODELS=et,xgb
  V4_DIAG_CALS=raw,sigmoid,isotonic
  V4_DIAG_TUNE=0|1
  V4_DIAG_TUNE_ITER=20
  V4_DIAG_TUNE_CV=3
  V4_MAX_RACES=0 (unlimited)
"""

from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import argparse

import numpy as np
import pandas as pd
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Optional xgboost, guard import
try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False

from ml_system_v4 import MLSystemV4


def _find_calibrate_script() -> Path | None:
    """Locate the calibrate_model.py script.
    Search order (first existing path wins):
      1) Env CALIBRATE_MODEL_PATH
      2) scripts/calibrate_model.py (sibling to this file)
      3) scripts/ml/calibrate_model.py
      4) calibrate_model.py at repo root
    Returns a Path if found, else None.
    """
    try:
        # Env override
        env_path = os.environ.get("CALIBRATE_MODEL_PATH")
        if env_path:
            p = Path(env_path)
            if p.exists():
                return p
        here = Path(__file__).resolve().parent
        candidates = [
            here / "calibrate_model.py",
            here / "ml" / "calibrate_model.py",
            Path.cwd() / "scripts" / "calibrate_model.py",
            Path.cwd() / "scripts" / "ml" / "calibrate_model.py",
            Path.cwd() / "calibrate_model.py",
        ]
        for c in candidates:
            try:
                if c.exists():
                    return c
            except Exception:
                continue
        return None
    except Exception:
        return None


OUT_DIR = Path("logs/diagnostics_auc")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUT_DIR / TS
RUN_DIR.mkdir(parents=True, exist_ok=True)


def topk_hit_rate(y_true, proba, groups, k=1):
    df = pd.DataFrame({"y": y_true, "p": proba, "g": groups})
    hits = []
    for _, gdf in df.groupby("g"):
        topk = gdf.sort_values("p", ascending=False).head(k)
        hits.append(int(topk["y"].max() == 1))
    return float(np.mean(hits)) if hits else np.nan


def build_preprocessor(X: pd.DataFrame, categorical_cols):
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
        ]
    )
    return ColumnTransformer(
        [
            ("cat", cat_pipe, categorical_cols),
            ("num", num_pipe, numeric_cols),
        ]
    )


def evaluate_variant(
    name,
    base_clf,
    calibrator: str | None,
    X_train,
    y_train,
    X_test,
    y_test,
    groups_test,
    categorical_cols,
    plots_prefix,
):
    pre = build_preprocessor(X_train, categorical_cols)
    if calibrator is None or calibrator == "raw":
        model = Pipeline([("prep", pre), ("clf", base_clf)])
        model.fit(X_train, y_train)
        prob_test = model.predict_proba(X_test)[:, 1]
    else:
        wrapped = Pipeline([("prep", pre), ("clf", base_clf)])
        model = CalibratedClassifierCV(
            wrapped, method=("sigmoid" if calibrator == "sigmoid" else "isotonic"), cv=5
        )
        model.fit(X_train, y_train)
        prob_test = model.predict_proba(X_test)[:, 1]

    try:
        roc_auc = roc_auc_score(y_test, prob_test)
    except Exception:
        roc_auc = np.nan
    try:
        pr_auc = average_precision_score(y_test, prob_test)
    except Exception:
        pr_auc = np.nan
    brier = brier_score_loss(y_test, prob_test)
    acc = accuracy_score(y_test, (prob_test >= 0.5).astype(int))
    top1 = topk_hit_rate(y_test, prob_test, groups_test, k=1)
    top3 = topk_hit_rate(y_test, prob_test, groups_test, k=3)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        RocCurveDisplay.from_predictions(y_test, prob_test, ax=ax[0])
        ax[0].set_title(f"ROC {name}")
        PrecisionRecallDisplay.from_predictions(y_test, prob_test, ax=ax[1])
        ax[1].set_title(f"PR {name}")
        fig.tight_layout()
        fig.savefig(RUN_DIR / f"{plots_prefix}_{name}_roc_pr.png", dpi=140)
        plt.close(fig)
        prob_true, prob_pred = calibration_curve(y_test, prob_test, n_bins=10)
        plt.figure(figsize=(4.5, 4))
        plt.plot(prob_pred, prob_true, marker="o", label=name)
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("Mean predicted prob")
        plt.ylabel("Fraction of positives")
        plt.title(f"Calibration {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RUN_DIR / f"{plots_prefix}_{name}_calibration.png", dpi=140)
        plt.close()
    except Exception:
        pass

    return {
        "name": name,
        "roc_auc": float(roc_auc) if roc_auc == roc_auc else None,
        "pr_auc": float(pr_auc) if pr_auc == pr_auc else None,
        "brier": float(brier),
        "accuracy": float(acc),
        "top1": float(top1) if top1 == top1 else None,
        "top3": float(top3) if top3 == top3 else None,
    }


def _build_features_for_split(raw_df: pd.DataFrame, sys: MLSystemV4) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()
    feats = []
    for rid, g in raw_df.groupby("race_id"):
        try:
            f = sys.build_features_for_race_with_cache(g, rid)
            if f is not None and not f.empty:
                if "race_id" not in f.columns:
                    f["race_id"] = rid
                feats.append(f)
        except Exception:
            continue
    return pd.concat(feats, ignore_index=True) if feats else pd.DataFrame()


def _decode_to_str(v):
    if isinstance(v, (bytes, bytearray)):
        for enc in ("utf-8", "latin-1"):
            try:
                return v.decode(enc, errors="ignore")
            except Exception:
                continue
        return str(v)
    return v


def _coerce_categoricals(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    if not categorical_cols:
        return df
    df2 = df.copy()
    cols = [c for c in categorical_cols if c in df2.columns]
    if cols:
        df2[cols] = df2[cols].applymap(_decode_to_str).astype(str)
    return df2


def _tune_model_et(
    X_train, y_train, categorical_cols, random_state=42, n_iter=20, cv=3
):
    pre = build_preprocessor(X_train, categorical_cols)
    base = ExtraTreesClassifier(
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1,
    )
    pipe = Pipeline(
        [
            ("prep", pre),
            ("clf", base),
        ]
    )
    param_dist = {
        "clf__n_estimators": sp_randint(200, 800),
        "clf__max_depth": sp_randint(6, 30),
        "clf__min_samples_leaf": sp_randint(1, 6),
        "clf__max_features": sp_uniform(0.3, 0.7),
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=int(n_iter) if n_iter else 20,
        cv=int(cv) if cv else 3,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def _tune_model_xgb(
    X_train, y_train, categorical_cols, random_state=42, n_iter=20, cv=3
):
    if not HAS_XGB:
        return None
    pre = build_preprocessor(X_train, categorical_cols)
    base = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        tree_method="hist",
        random_state=random_state,
        eval_metric="logloss",
    )
    pipe = Pipeline(
        [
            ("prep", pre),
            ("clf", base),
        ]
    )
    param_dist = {
        "clf__n_estimators": sp_randint(200, 800),
        "clf__max_depth": sp_randint(3, 10),
        "clf__learning_rate": sp_uniform(0.01, 0.2),
        "clf__subsample": sp_uniform(0.6, 0.4),
        "clf__colsample_bytree": sp_uniform(0.6, 0.4),
        "clf__reg_lambda": sp_uniform(0.0, 2.0),
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=int(n_iter) if n_iter else 20,
        cv=int(cv) if cv else 3,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


# ---- CLI helpers and argument parsing with env defaults ----


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).lower() in ("1", "true", "yes", "on")


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _parse_args():
    parser = argparse.ArgumentParser(description="AUC Diagnostics for MLSystemV4")
    parser.add_argument(
        "--models",
        type=str,
        default=_env_str("V4_DIAG_MODELS", "et,xgb"),
        help="Comma-separated: et,xgb",
    )
    parser.add_argument(
        "--calibrations",
        type=str,
        default=_env_str("V4_DIAG_CALS", "raw,sigmoid,isotonic"),
        help="Comma-separated: raw,sigmoid,isotonic",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        default=_env_bool("V4_DIAG_TUNE", False),
        help="Run hyperparameter search to maximize ROC AUC (env V4_DIAG_TUNE)",
    )
    parser.add_argument(
        "--tune-iter",
        type=int,
        default=_env_int("V4_DIAG_TUNE_ITER", 20),
        help="RandomizedSearch iterations per model (env V4_DIAG_TUNE_ITER)",
    )
    parser.add_argument(
        "--tune-cv",
        type=int,
        default=_env_int("V4_DIAG_TUNE_CV", 3),
        help="CV folds for tuning (env V4_DIAG_TUNE_CV)",
    )
    parser.add_argument(
        "--max-races",
        type=int,
        default=_env_int("V4_MAX_RACES", 0),
        help="Cap number of races to evaluate (env V4_MAX_RACES)",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    # Promotion flags
    default_auto = os.environ.get("V4_DIAG_AUTOPROMOTE", "1").lower() not in (
        "0",
        "false",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--auto-promote",
        dest="auto_promote",
        action="store_true",
        default=default_auto,
        help="Enable auto-promotion of best model (env V4_DIAG_AUTOPROMOTE)",
    )
    group.add_argument(
        "--no-promote",
        dest="auto_promote",
        action="store_false",
        help="Disable auto-promotion",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    print("🔎 Running AUC diagnostics ...")
    print(
        f"[diag] Using models={args.models}, calibrations={args.calibrations}, tune={'on' if args.tune else 'off'}, tune_iter={args.tune_iter}, tune_cv={args.tune_cv}, max_races={args.max_races or 'unlimited'}"
    )

    sys_v4 = MLSystemV4()
    if hasattr(sys_v4, "prepare_time_ordered_data"):
        try:
            raw_train_df, raw_test_df = sys_v4.prepare_time_ordered_data(
                max_races=args.max_races or None
            )
        except TypeError:
            raw_train_df, raw_test_df = sys_v4.prepare_time_ordered_data()
    else:
        raw_train_df, raw_test_df = sys_v4.prepare_time_ordered_data()

    if raw_train_df.empty or raw_test_df.empty:
        print("❌ No data to evaluate")
        return 1

    print("[diag] Building features (train/test) ...")
    train_df = _build_features_for_split(raw_train_df, sys_v4)
    test_df = _build_features_for_split(raw_test_df, sys_v4)
    if train_df.empty or test_df.empty:
        print("❌ Failed to build features for diagnostics")
        return 1

    if "target" not in train_df.columns or "target" not in test_df.columns:
        print("❌ Target column missing after feature build")
        return 1

    y_train = train_df["target"].astype(int).to_numpy()
    y_test = test_df["target"].astype(int).to_numpy()
    groups_test = test_df.get("race_id", pd.Series(np.arange(len(y_test))))

    drop_cols = ["target", "race_id", "dog_clean_name", "target_timestamp"]
    X_train = train_df.drop([c for c in drop_cols if c in train_df.columns], axis=1)
    X_test = test_df.drop([c for c in drop_cols if c in test_df.columns], axis=1)

    categorical_cols = [c for c in X_train.columns if X_train[c].dtype == object]
    X_train = _coerce_categoricals(X_train, categorical_cols)
    X_test = _coerce_categoricals(X_test, categorical_cols)

    results = []

    selected_models = [
        m.strip().lower() for m in (args.models.split(",") if args.models else [])
    ]
    selected_cals = [
        c.strip().lower()
        for c in (args.calibrations.split(",") if args.calibrations else [])
    ]

    tuned_et = None
    tuned_xgb = None
    if args.tune:
        if "et" in selected_models:
            print("⚙️ Tuning ExtraTrees for ROC AUC ...")
            tuned_et = _tune_model_et(
                X_train,
                y_train,
                categorical_cols,
                random_state=args.random_state,
                n_iter=args.tune_iter,
                cv=args.tune_cv,
            )
        if HAS_XGB and "xgb" in selected_models:
            print("⚙️ Tuning XGBoost for ROC AUC ...")
            tuned_xgb = _tune_model_xgb(
                X_train,
                y_train,
                categorical_cols,
                random_state=args.random_state,
                n_iter=args.tune_iter,
                cv=args.tune_cv,
            )

    if "et" in selected_models:
        if tuned_et is not None:
            try:
                tuned_clf = tuned_et.named_steps.get("clf")
            except Exception:
                tuned_clf = None
            et_base = (
                ExtraTreesClassifier(**{**tuned_clf.get_params()})
                if tuned_clf is not None
                else ExtraTreesClassifier(
                    n_estimators=500,
                    max_depth=15,
                    min_samples_leaf=3,
                    n_jobs=-1,
                    random_state=args.random_state,
                )
            )
        else:
            et_base = ExtraTreesClassifier(
                n_estimators=500,
                max_depth=15,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=args.random_state,
            )
        for cal in selected_cals:
            print(f"[diag] Evaluating ET ({cal}) ...")
            r = evaluate_variant(
                f"ExtraTrees__{cal}",
                et_base,
                (None if cal == "raw" else cal),
                X_train,
                y_train,
                X_test,
                y_test,
                groups_test,
                categorical_cols,
                plots_prefix="et",
            )
            results.append(r)
        et_bal = ExtraTreesClassifier(
            n_estimators=et_base.get_params().get("n_estimators", 500),
            max_depth=et_base.get_params().get("max_depth", 15),
            min_samples_leaf=et_base.get_params().get("min_samples_leaf", 3),
            n_jobs=-1,
            random_state=args.random_state,
            class_weight="balanced",
        )
        if "sigmoid" in selected_cals:
            print("[diag] Evaluating ET (balanced + sigmoid) ...")
            r = evaluate_variant(
                "ExtraTrees__balanced_sigmoid",
                et_bal,
                "sigmoid",
                X_train,
                y_train,
                X_test,
                y_test,
                groups_test,
                categorical_cols,
                plots_prefix="et_bal",
            )
            results.append(r)

    if HAS_XGB and "xgb" in selected_models:
        if tuned_xgb is not None:
            try:
                tuned_clf = tuned_xgb.named_steps.get("clf")
            except Exception:
                tuned_clf = None
            xgb = (
                XGBClassifier(**{**tuned_clf.get_params()})
                if tuned_clf is not None
                else XGBClassifier(
                    n_estimators=600,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    n_jobs=-1,
                    tree_method="hist",
                    random_state=args.random_state,
                    eval_metric="logloss",
                )
            )
        else:
            xgb = XGBClassifier(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                n_jobs=-1,
                tree_method="hist",
                random_state=args.random_state,
                eval_metric="logloss",
            )
        for cal in selected_cals:
            print(f"[diag] Evaluating XGB ({cal}) ...")
            r = evaluate_variant(
                f"XGBoost__{cal}",
                xgb,
                (None if cal == "raw" else cal),
                X_train,
                y_train,
                X_test,
                y_test,
                groups_test,
                categorical_cols,
                plots_prefix="xgb",
            )
            results.append(r)

    res_df = pd.DataFrame(results)
    res_path = RUN_DIR / "auc_diagnostics_summary.csv"
    res_df.to_csv(res_path, index=False)

    print("\n📊 Summary:\n", res_df.to_string(index=False))

    # Select best configuration by highest ROC AUC (fallback to PR AUC if NaNs)
    best_idx = None
    if not res_df.empty:
        try:
            # Prefer rows with non-null roc_auc; break ties by pr_auc then top1
            scored = res_df.copy()
            scored["roc_auc_fill"] = scored["roc_auc"].fillna(-1)
            scored["pr_auc_fill"] = scored["pr_auc"].fillna(-1)
            scored["top1_fill"] = scored["top1"].fillna(-1)
            best_idx = scored.sort_values(
                ["roc_auc_fill", "pr_auc_fill", "top1_fill"],
                ascending=[False, False, False],
            ).index[0]
        except Exception:
            best_idx = res_df.index[0]

    best_config = None
    if best_idx is not None:
        best_row = res_df.loc[best_idx]
        best_name = str(best_row.get("name", ""))
        # Parse model and calibration from name pattern "Model__cal"
        model_key = None
        calibration = None
        try:
            if "__" in best_name:
                parts = best_name.split("__", 1)
                model_key = parts[0].lower()
                calibration = parts[1].lower()
            else:
                model_key = best_name.lower()
                calibration = "raw"
        except Exception:
            model_key = None
            calibration = None
        best_config = {
            "name": best_name,
            "model": model_key,
            "calibration": calibration,
            "metrics": {
                "roc_auc": best_row.get("roc_auc"),
                "pr_auc": best_row.get("pr_auc"),
                "brier": best_row.get("brier"),
                "accuracy": best_row.get("accuracy"),
                "top1": best_row.get("top1"),
                "top3": best_row.get("top3"),
            },
            "timestamp": datetime.now().isoformat(),
        }
        try:
            import json

            with open(RUN_DIR / "best_config.json", "w") as f:
                json.dump(best_config, f, indent=2)
            print(f"[diag] Best config saved: {best_config['name']}")
        except Exception as e:
            print(f"[diag] Failed to save best_config.json: {e}")

        # Attempt automatic model update via calibration script
        try:
            if not args.auto_promote:
                print(
                    "[diag] Auto-promotion disabled by flag/env; skipping promotion step"
                )
            else:
                script_path = _find_calibrate_script()
                supported = (
                    model_key in ("extratrees", "xgboost", "et", "xgb") and calibration
                )
                if script_path and supported:
                    # Normalize model flag values
                    model_flag = (
                        "et"
                        if model_key in ("extratrees", "et")
                        else ("xgb" if model_key in ("xgboost", "xgb") else model_key)
                    )
                    print(
                        f"[diag] ⬆️  Promoting best model via {script_path} (model={model_flag}, cal={calibration})"
                    )
                    import json
                    import subprocess

                    cmd = [
                        sys.executable,
                        str(script_path),
                        "--model",
                        model_flag,
                        "--calibration",
                        calibration,
                        "--promote",
                    ]
                    if args.max_races and int(args.max_races) > 0:
                        cmd.extend(["--max-races", str(int(args.max_races))])
                    env = os.environ.copy()
                    env.setdefault("PYTHONPATH", str(Path.cwd()))
                    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
                    model_id = None
                    if proc.returncode == 0:
                        print(
                            "[diag] ✅ Model calibration/promotion completed successfully."
                        )
                        # Try parse JSON from stdout for model metadata
                        try:
                            # Look for the last JSON object in stdout
                            for line in reversed(proc.stdout.splitlines()):
                                line = line.strip()
                                if line.startswith("{") and line.endswith("}"):
                                    meta = json.loads(line)
                                    model_id = meta.get("model_id") or meta.get("id")
                                    break
                        except Exception:
                            pass
                    else:
                        print(
                            "[diag] ⚠️ Calibration script returned non-zero exit status."
                        )
                        if proc.stderr:
                            print(proc.stderr[-400:])
                    # Write audit entry
                    try:
                        audit = {
                            "timestamp": datetime.now().isoformat(),
                            "module": "model_promotion",
                            "severity": "INFO" if proc.returncode == 0 else "ERROR",
                            "event": (
                                "model_promoted"
                                if proc.returncode == 0
                                else "model_promotion_failed"
                            ),
                            "message": f"Promoted {model_flag} with {calibration}",
                            "details": {
                                "model": model_flag,
                                "calibration": calibration,
                                "model_id": model_id,
                                "roc_auc": best_config["metrics"].get("roc_auc"),
                                "pr_auc": best_config["metrics"].get("pr_auc"),
                            },
                        }
                        log_path = Path("logs") / "system_log.jsonl"
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(audit) + "\n")
                        print(
                            "[diag] 📗 Promotion audit written to logs/system_log.jsonl"
                        )
                    except Exception as e:
                        print(f"[diag] Failed to write promotion audit: {e}")
                else:
                    reason = []
                    if not script_path:
                        reason.append(
                            "calibrate_model.py not found; set CALIBRATE_MODEL_PATH or place it under scripts/ or scripts/ml/"
                        )
                    if not supported:
                        reason.append(
                            f"unsupported model/calibration: model={model_key}, cal={calibration}"
                        )
                    msg = "; ".join(reason) if reason else "unknown reason"
                    print(f"[diag] Skipping auto-promotion ({msg})")
        except Exception as e:
            print(f"[diag] Auto-promotion step failed: {e}")

    print(f"\n🗂️ Outputs saved to: {RUN_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
