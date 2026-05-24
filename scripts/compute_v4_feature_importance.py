#!/usr/bin/env python3
"""
Compute permutation feature importance for the current MLSystemV4 model.
Saves:
- artifacts/eval/feature_importance.json
- artifacts/eval/feature_importance_summary.md

This samples the test split from MLSystemV4.prepare_time_ordered_data(), builds
leakage-safe features, aligns to the model's expected input columns, and computes
permutation importance on the fitted pipeline.
"""
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from ml_system_v4 import MLSystemV4


def _get_expected_columns_from_pipeline(cp):
    expected_cols = []
    try:
        pre = cp.base_estimator_.named_steps.get("preprocessor")
        pre_num_cols, pre_cat_cols = [], []
        if pre and hasattr(pre, "transformers_"):
            for name, _trans, cols in pre.transformers_:
                if name == "num":
                    try:
                        pre_num_cols = list(cols) if isinstance(cols, (list, tuple, set)) else []
                    except Exception:
                        pre_num_cols = []
                elif name == "cat":
                    try:
                        pre_cat_cols = list(cols) if isinstance(cols, (list, tuple, set)) else []
                    except Exception:
                        pre_cat_cols = []
        expected_cols = (pre_num_cols or []) + (pre_cat_cols or [])
        if not expected_cols:
            # Try calibrated clones
            try:
                calibrators = getattr(cp, "calibrated_classifiers_", []) or []
                for cal in calibrators:
                    base_est = getattr(cal, "base_estimator", None)
                    if base_est is not None and hasattr(base_est, "named_steps"):
                        pre2 = base_est.named_steps.get("preprocessor")
                        if pre2 and hasattr(pre2, "transformers_"):
                            pre_num_cols, pre_cat_cols = [], []
                            for name, _trans, cols in pre2.transformers_:
                                if name == "num":
                                    pre_num_cols = list(cols) if isinstance(cols, (list, tuple, set)) else []
                                elif name == "cat":
                                    pre_cat_cols = list(cols) if isinstance(cols, (list, tuple, set)) else []
                            derived = (pre_num_cols or []) + (pre_cat_cols or [])
                            if derived:
                                expected_cols = derived
                                break
            except Exception:
                pass
    except Exception:
        expected_cols = []
    return list(expected_cols or [])


def main():
    out_dir = Path("artifacts/eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = os.getenv("GREYHOUND_DB_PATH") or "greyhound_racing_data.db"
    system = MLSystemV4(db_path)
    cp = getattr(system, "calibrated_pipeline", None)
    if cp is None:
        print("No calibrated pipeline loaded; aborting feature importance computation.")
        return 2

    # Prepare train/test data (time-ordered)
    train_df, test_df = system.prepare_time_ordered_data()
    if test_df is None or len(test_df) == 0:
        print("Empty test data; aborting.")
        return 2

    # Build leakage-safe features for test
    feats = system.build_leakage_safe_features(test_df)
    if feats is None or len(feats) == 0:
        print("No features built for test data; aborting.")
        return 2

    # Prepare X and y
    y = pd.to_numeric(feats.get("target"), errors="coerce").fillna(0).astype(int)
    X = feats.drop(columns=[c for c in ["race_id", "dog_clean_name", "target", "target_timestamp"] if c in feats.columns], errors="ignore").copy()

    # Align to expected columns from the fitted preprocessor
    expected_cols = _get_expected_columns_from_pipeline(cp)
    if not expected_cols:
        # fallback to training-time feature_columns
        expected_cols = list(getattr(system, "feature_columns", []) or [])

    # Ensure we include known TGR features if model was trained with them
    known_tgr = [
        "tgr_total_races",
        "tgr_recent_races",
        "tgr_avg_finish_position",
        "tgr_best_finish_position",
        "tgr_win_rate",
        "tgr_place_rate",
        "tgr_consistency",
        "tgr_form_trend",
        "tgr_recent_avg_position",
        "tgr_recent_best_position",
        "tgr_preferred_distance",
        "tgr_preferred_distance_avg",
        "tgr_preferred_distance_races",
        "tgr_venues_raced",
        "tgr_days_since_last_race",
        "tgr_last_race_position",
        "tgr_has_comments",
        "tgr_sentiment_score",
    ]
    # Extend expected columns without duplicates
    expected_cols = list(dict.fromkeys(list(expected_cols) + known_tgr))

    # Ensure categorical defaults
    cat_defaults = {
        "venue": "UNKNOWN",
        "grade": "5",
        "track_condition": "Good",
        "weather": "Fine",
        "trainer_name": "Unknown",
    }
    for col, default in cat_defaults.items():
        if col in X.columns:
            X[col] = X[col].replace({0: default, "0": default}).astype(str)

    # Ensure all expected columns exist with numeric defaults where needed
    missing_cols = [c for c in expected_cols if c not in X.columns]
    for col in missing_cols:
        X[col] = 0.0
    # Final strict column order
    X = X.reindex(columns=expected_cols, fill_value=0.0)

    # Sampling for speed
    n = len(X)
    sample_n = min(2000, n)
    if sample_n < n:
        idx = np.linspace(0, n - 1, sample_n).astype(int)
        X = X.iloc[idx]
        y = y.iloc[idx]

    # Compute permutation importance
    result = permutation_importance(cp, X, y, n_repeats=3, random_state=42, n_jobs=-1)
    importances = result.importances_mean

    importance_list = [
        {"feature": f, "importance": float(im)} for f, im in zip(X.columns.tolist(), importances)
    ]
    # Sort desc
    importance_list.sort(key=lambda d: d["importance"], reverse=True)

    # Persist JSON
    json_path = out_dir / "feature_importance.json"
    json_path.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "db_path": str(db_path),
        "samples": int(len(X)),
        "features": importance_list,
    }, indent=2))

    # Persist summary markdown
    md_lines = ["# Feature Importance Summary", "", f"Samples: {len(X)}", ""]
    md_lines.append("## Top 20 Features")
    for i, item in enumerate(importance_list[:20], 1):
        md_lines.append(f"{i}. {item['feature']}: {item['importance']:.6f}")

    # Highlight underused features of interest
    interested = ["venue", "weather", "first_sectional", "win_time", "pir_rating"]
    md_lines.append("\n## Underused Features (of interest)")
    for name in interested:
        val = next((x["importance"] for x in importance_list if x["feature"] == name), None)
        if val is None:
            md_lines.append(f"- {name}: not present in model inputs")
        else:
            md_lines.append(f"- {name}: importance {val:.6f}")

    (out_dir / "feature_importance_summary.md").write_text("\n".join(md_lines) + "\n")
    print(f"Saved {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

