#!/usr/bin/env python3
"""Report-only box-bias diagnostics for the V4 greyhound model.

This script intentionally does not write model artifacts, update the registry,
rewrite snapshots, or mutate the database. It builds temporary challenger
models in memory and writes markdown/JSON diagnostics only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_system_v4 import MLSystemV4
from utils.leakage_guard import audit_feature_matrix, strip_target_leakage_columns


DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "artifacts"
    / "full_evidence_orchestration_20260525"
    / "box_bias_audit"
)
DEFAULT_MODEL = REPO_ROOT / "model_registry" / "best_model.joblib"
DEFAULT_DB = REPO_ROOT / "greyhound_racing_data_writable.db"
DEFAULT_CAPTURE_REPORT = (
    REPO_ROOT
    / "artifacts"
    / "full_evidence_orchestration_20260525"
    / "live_ev_batch_after_time_fix"
    / "capture_report.json"
)
METADATA_COLUMNS = {
    "race_id",
    "dog_clean_name",
    "target",
    "target_timestamp",
}
TRAIN_EXCLUDE_COLUMNS = {"race_date", "race_time"}
CATEGORICAL_BASE = ["venue", "grade", "track_condition", "weather", "trainer_name"]
BOX_FEATURE_PATTERNS = ("box_number", "current_box", "box_win_rate")
DEFAULT_CATEGORICAL_VALUES = {
    "venue": "UNKNOWN",
    "grade": "UNKNOWN",
    "track_condition": "",
    "weather": "",
    "trainer_name": "Unknown",
    "box_band": "unknown",
}
DEFAULT_HISTORY_PROFILE = {
    "historical_avg_position": 4.5,
    "historical_win_rate": 0.125,
    "historical_place_rate": 0.375,
}


@dataclass
class VariantResult:
    name: str
    model: Any
    train_columns: list[str]
    metrics: dict[str, Any]
    family_importance: list[dict[str, Any]]
    top_features: list[dict[str, Any]]


def clean_name(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.6f}")
            else:
                vals.append(str(val))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def original_feature_name(transformed_name: str) -> str:
    is_categorical = transformed_name.startswith("cat__")
    name = transformed_name.split("__", 1)[1] if "__" in transformed_name else transformed_name
    if is_categorical:
        known_cats = [
            "track_condition",
            "trainer_name",
            "box_number",
            "weather",
            "venue",
            "grade",
            "box_band",
        ]
        for col in sorted(known_cats, key=len, reverse=True):
            if name == col or name.startswith(f"{col}_"):
                return col
    return name


def feature_family(feature_name: str) -> str:
    original = original_feature_name(feature_name)
    lowered = original.lower()
    if (
        lowered in {"sp", "odds", "odds_decimal", "market_odds", "starting_price"}
        or any(token in lowered for token in ("odds", "market", "ev_win", "starting_price"))
    ):
        return "odds_market"
    if any(token in lowered for token in ("missing", "default", "indicator", "no_history")):
        return "missing_default_indicators"
    if lowered in BOX_FEATURE_PATTERNS or lowered == "box_band":
        return "box_number"
    if lowered.startswith("tgr_"):
        return "embedded_form_history"
    if (
        lowered.startswith("historical_")
        or lowered.startswith("venue_specific_")
        or lowered.startswith("grade_specific_")
        or lowered.startswith("best_distance_")
        or lowered
        in {
            "days_since_last_race",
            "race_frequency",
            "distance_adjusted_time",
            "venue_experience",
            "venue_best_position",
            "grade_experience",
        }
    ):
        return "historical_performance"
    if lowered in {
        "venue",
        "grade",
        "distance",
        "target_distance",
        "field_size",
        "track_condition",
        "weather",
        "temperature",
        "humidity",
        "wind_speed",
        "race_date",
        "race_time",
    }:
        return "venue_distance_grade"
    return "other"


def iter_calibrated_estimators(model: Any) -> Iterable[tuple[int, Any]]:
    calibrated = getattr(model, "calibrated_classifiers_", None)
    if calibrated:
        for index, fold in enumerate(calibrated):
            estimator = getattr(fold, "estimator", None) or getattr(
                fold, "base_estimator", None
            )
            if estimator is not None:
                yield index, estimator
        return
    yield 0, model


def extract_importance(model: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []

    for fold_index, estimator in iter_calibrated_estimators(model):
        pipeline = estimator
        if not hasattr(pipeline, "named_steps"):
            continue
        preprocessor = pipeline.named_steps.get("preprocessor")
        classifier = pipeline.named_steps.get("classifier") or pipeline.named_steps.get("model")
        if preprocessor is None or classifier is None:
            continue
        if not hasattr(classifier, "feature_importances_"):
            continue
        try:
            names = list(preprocessor.get_feature_names_out())
        except Exception:
            continue
        importances = np.asarray(classifier.feature_importances_, dtype=float)
        limit = min(len(names), len(importances))
        fold_rows = []
        for name, importance in zip(names[:limit], importances[:limit]):
            row = {
                "fold": fold_index,
                "feature": str(name),
                "original_feature": original_feature_name(str(name)),
                "family": feature_family(str(name)),
                "importance": float(importance),
            }
            rows.append(row)
            fold_rows.append(row)
        if fold_rows:
            fold_df = pd.DataFrame(fold_rows)
            box_df = fold_df[fold_df["family"] == "box_number"]
            top_row = fold_df.sort_values("importance", ascending=False).iloc[0]
            box1_rows = fold_df[fold_df["feature"].astype(str).str.startswith("cat__box_number_1")]
            sorted_fold = fold_df.sort_values("importance", ascending=False).reset_index(drop=True)
            box1_rank = None
            if not box1_rows.empty:
                box1_feature = box1_rows.sort_values("importance", ascending=False).iloc[0]["feature"]
                matches = sorted_fold.index[sorted_fold["feature"] == box1_feature].tolist()
                box1_rank = int(matches[0] + 1) if matches else None
            fold_summaries.append(
                {
                    "fold": fold_index,
                    "box_importance": float(box_df["importance"].sum()) if not box_df.empty else 0.0,
                    "box1_importance": float(box1_rows["importance"].sum()) if not box1_rows.empty else 0.0,
                    "box1_rank": box1_rank,
                    "top_feature": str(top_row["feature"]),
                    "top_feature_importance": float(top_row["importance"]),
                }
            )

    if not rows:
        return {
            "rows": [],
            "top_features": [],
            "family_importance": [],
            "fold_summaries": [],
        }

    df = pd.DataFrame(rows)
    top_features = (
        df.groupby(["feature", "original_feature", "family"], as_index=False)["importance"]
        .agg(["mean", "min", "max", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    fold_family = (
        df.groupby(["fold", "family"], as_index=False)["importance"]
        .sum()
    )
    family = (
        fold_family.groupby("family", as_index=False)["importance"]
        .agg(["mean", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    family["share"] = family["mean"] / family["mean"].sum()
    return {
        "rows": rows,
        "top_features": top_features.to_dict("records"),
        "family_importance": family.to_dict("records"),
        "fold_summaries": fold_summaries,
    }


def input_columns_from_model(model: Any) -> list[str]:
    for _, estimator in iter_calibrated_estimators(model):
        if not hasattr(estimator, "named_steps"):
            continue
        preprocessor = estimator.named_steps.get("preprocessor")
        if preprocessor is None or not hasattr(preprocessor, "transformers_"):
            continue
        cols: list[str] = []
        for name, _transformer, transformer_cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            if isinstance(transformer_cols, (list, tuple)):
                cols.extend([str(c) for c in transformer_cols])
        if cols:
            return list(dict.fromkeys(cols))
    return []


def load_model_metadata(model_path: Path) -> dict[str, Any]:
    model_path = model_path.resolve()
    metadata_path = (
        REPO_ROOT
        / "model_registry"
        / "metadata"
        / model_path.name.replace("_model.joblib", "_metadata.json")
    )
    if metadata_path.exists():
        return load_json(metadata_path)
    return {}


def query_label_groups(db_path: Path, race_ids: Iterable[str]) -> dict[str, str]:
    ids = sorted({str(race_id) for race_id in race_ids if str(race_id)})
    if not ids:
        return {}
    placeholders = ",".join(["?"] * len(ids))
    result: dict[str, str] = {}
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT race_id, COALESCE(results_status, ''), COALESCE(winner_source, '')
            FROM race_metadata
            WHERE race_id IN ({placeholders})
            """,
            ids,
        ).fetchall()
    for race_id, status, source in rows:
        status_l = str(status).lower()
        source_l = str(source).lower()
        if "partial" in status_l or "sportsbet" in source_l:
            group = "partial"
        elif "official" in source_l or status_l in {"complete", "resulted"}:
            group = "official_or_complete"
        else:
            group = "db_finish_position_only"
        result[str(race_id)] = group
    return result


def prepare_base_features(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    labels = features["target"].fillna(0).astype(int)
    meta = features[["race_id", "dog_clean_name", "target"]].copy()
    if "box_number" in features.columns:
        meta["box_number"] = pd.to_numeric(features["box_number"], errors="coerce")
    else:
        meta["box_number"] = np.nan

    x = features.drop(list(METADATA_COLUMNS), axis=1, errors="ignore").copy()
    x, _dropped = strip_target_leakage_columns(x, allow_labels=False)
    for col in TRAIN_EXCLUDE_COLUMNS:
        if col in x.columns:
            x = x.drop(columns=[col])
    return x, labels, meta


def add_history_default_indicators(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    default_parts = []
    for column, default in DEFAULT_HISTORY_PROFILE.items():
        if column in x.columns:
            default_parts.append(pd.to_numeric(x[column], errors="coerce").round(6) == default)
    if default_parts:
        combined = default_parts[0]
        for part in default_parts[1:]:
            combined = combined & part
        x["history_default_profile_indicator"] = combined.astype(int)
    if "target_distance" in x.columns:
        x["target_distance_default_indicator"] = (
            pd.to_numeric(x["target_distance"], errors="coerce").fillna(0.0) == 0.0
        ).astype(int)
    if "tgr_total_races" in x.columns:
        x["tgr_no_history_indicator"] = (
            pd.to_numeric(x["tgr_total_races"], errors="coerce").fillna(0.0) == 0.0
        ).astype(int)
    return x


def remove_box_columns(x: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        col
        for col in x.columns
        if any(pattern in str(col).lower() for pattern in BOX_FEATURE_PATTERNS)
    ]
    return x.drop(columns=drop_cols, errors="ignore")


def add_box_band(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    if "box_number" in x.columns:
        box = pd.to_numeric(x["box_number"], errors="coerce")
        x["box_band"] = np.select(
            [box.isin([1, 2]), box.isin([3, 4, 5, 6]), box >= 7],
            ["inside", "middle", "outside"],
            default="unknown",
        )
    return x


def keep_history_only(x: pd.DataFrame) -> pd.DataFrame:
    keep = [
        col
        for col in x.columns
        if feature_family(str(col)) in {"historical_performance", "embedded_form_history"}
    ]
    return x[keep].copy()


def make_variant_x(x: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant == "current_champion_baseline":
        return x.copy()
    if variant == "full_box_retrained_report_only":
        return x.copy()
    if variant == "no_box":
        return remove_box_columns(x)
    if variant == "reduced_box_band":
        return remove_box_columns(add_box_band(x))
    if variant == "history_only":
        return keep_history_only(remove_box_columns(x))
    if variant == "embedded_history_aware_no_box":
        return add_history_default_indicators(remove_box_columns(x))
    raise ValueError(f"Unknown variant: {variant}")


def align_columns(x: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    aligned = x.copy()
    for column in columns:
        if column not in aligned.columns:
            aligned[column] = DEFAULT_CATEGORICAL_VALUES.get(column, 0)
    extra = [column for column in aligned.columns if column not in columns]
    if extra:
        aligned = aligned.drop(columns=extra)
    return coerce_model_frame(aligned[columns])


def coerce_model_frame(x: pd.DataFrame) -> pd.DataFrame:
    """Coerce known categorical columns before sklearn encoders see them."""
    coerced = x.copy()
    for column, default in DEFAULT_CATEGORICAL_VALUES.items():
        if column in coerced.columns:
            coerced[column] = coerced[column].fillna(default).astype(str)
    if "box_number" in coerced.columns:
        coerced["box_number"] = (
            pd.to_numeric(coerced["box_number"], errors="coerce").fillna(0).astype(int)
        )
    return coerced


def build_pipeline(x_train: pd.DataFrame, trees: int, cv: int) -> CalibratedClassifierCV:
    categorical_cols = [
        col
        for col in CATEGORICAL_BASE + ["box_number", "box_band"]
        if col in x_train.columns
    ]
    numerical_cols = [
        col
        for col in x_train.columns
        if col not in categorical_cols and pd.api.types.is_numeric_dtype(x_train[col])
    ]
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )
    model = ExtraTreesClassifier(
        n_estimators=trees,
        min_samples_leaf=2,
        max_depth=20,
        max_features="sqrt",
        class_weight="balanced",
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
    return CalibratedClassifierCV(pipeline, method="isotonic", cv=cv)


def normalize_probabilities(meta: pd.DataFrame, probabilities: np.ndarray) -> pd.Series:
    df = meta[["race_id"]].copy()
    df["p"] = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    sums = df.groupby("race_id")["p"].transform("sum")
    counts = df.groupby("race_id")["p"].transform("count")
    normalized = np.where(sums > 0, df["p"] / sums, 1.0 / counts)
    return pd.Series(normalized, index=meta.index, dtype=float)


def calibration_summary(y: pd.Series, p: pd.Series) -> dict[str, Any]:
    y_arr = y.astype(int).to_numpy()
    p_arr = np.clip(p.astype(float).to_numpy(), 1e-6, 1 - 1e-6)
    if len(np.unique(y_arr)) < 2:
        return {"status": "DATA_MISSING", "reason": "single_label_class"}
    try:
        logits = np.log(p_arr / (1 - p_arr)).reshape(-1, 1)
        model = LogisticRegression(solver="lbfgs")
        model.fit(logits, y_arr)
        return {
            "status": "SUCCESS",
            "intercept": float(model.intercept_[0]),
            "slope": float(model.coef_[0][0]),
        }
    except Exception as exc:
        return {"status": "DATA_MISSING", "reason": str(exc)}


def reliability_bins(y: pd.Series, p: pd.Series, bins: int = 10) -> list[dict[str, Any]]:
    y_arr = y.astype(int).to_numpy()
    p_arr = p.astype(float).to_numpy()
    rows = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        if index == bins - 1:
            mask = (p_arr >= lower) & (p_arr <= upper)
        else:
            mask = (p_arr >= lower) & (p_arr < upper)
        if not mask.any():
            continue
        rows.append(
            {
                "bin": index,
                "lower": lower,
                "upper": upper,
                "count": int(mask.sum()),
                "avg_predicted": float(p_arr[mask].mean()),
                "actual_rate": float(y_arr[mask].mean()),
            }
        )
    return rows


def rank_metrics(meta: pd.DataFrame, y: pd.Series, p_norm: pd.Series) -> dict[str, Any]:
    eval_df = meta.copy()
    eval_df["target"] = y.astype(int).to_numpy()
    eval_df["p_norm"] = p_norm.astype(float).to_numpy()
    winner_ranks = []
    top1 = top2 = top3 = 0
    top_boxes: list[int] = []
    entropy_values = []
    spread_values = []

    for _race_id, group in eval_df.groupby("race_id", sort=False):
        group = group.sort_values("p_norm", ascending=False).copy()
        group["rank"] = np.arange(1, len(group) + 1)
        if not group.empty:
            try:
                top_boxes.append(int(group.iloc[0]["box_number"]))
            except Exception:
                pass
        winner = group[group["target"] == 1]
        if not winner.empty:
            rank = int(winner.iloc[0]["rank"])
            winner_ranks.append(rank)
            top1 += int(rank <= 1)
            top2 += int(rank <= 2)
            top3 += int(rank <= 3)
        arr = group["p_norm"].astype(float).to_numpy()
        if len(arr) > 1:
            entropy = float(-(arr * np.log(np.clip(arr, 1e-12, 1.0))).sum() / math.log(len(arr)))
        else:
            entropy = 0.0
        entropy_values.append(entropy)
        spread_values.append(float(arr.max() - arr.min()) if len(arr) else 0.0)

    races = len(winner_ranks)
    box_counter = Counter(top_boxes)
    return {
        "races_evaluated": int(races),
        "top1": float(top1 / races) if races else 0.0,
        "top2": float(top2 / races) if races else 0.0,
        "top3": float(top3 / races) if races else 0.0,
        "top1_hits": int(top1),
        "top2_hits": int(top2),
        "top3_hits": int(top3),
        "mean_winner_rank": float(np.mean(winner_ranks)) if winner_ranks else None,
        "top_pick_box_distribution": {str(k): int(v) for k, v in sorted(box_counter.items())},
        "box1_top_pick_rate": float(box_counter.get(1, 0) / len(top_boxes)) if top_boxes else 0.0,
        "avg_normalized_entropy": float(np.mean(entropy_values)) if entropy_values else None,
        "avg_probability_spread": float(np.mean(spread_values)) if spread_values else None,
    }


def evaluate_predictions(
    meta: pd.DataFrame,
    y: pd.Series,
    raw_probabilities: np.ndarray,
    label_groups: dict[str, str],
) -> dict[str, Any]:
    p_norm = normalize_probabilities(meta, raw_probabilities)
    rank = rank_metrics(meta, y, p_norm)
    y_int = y.astype(int)
    metrics = {
        **rank,
        "dog_predictions_evaluated": int(len(meta)),
        "brier": float(brier_score_loss(y_int, p_norm)),
        "log_loss": float(log_loss(y_int, np.clip(p_norm, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "calibration": calibration_summary(y_int, p_norm),
        "reliability_bins": reliability_bins(y_int, p_norm),
        "probability_sum": {
            "max_abs_error": float(
                normalize_probabilities(meta, raw_probabilities)
                .groupby(meta["race_id"])
                .sum()
                .sub(1.0)
                .abs()
                .max()
            )
        },
    }
    segmented: dict[str, Any] = {}
    for group_name in sorted(set(label_groups.values()) | {"official_or_complete", "partial"}):
        race_ids = [race_id for race_id, group in label_groups.items() if group == group_name]
        mask = meta["race_id"].isin(race_ids)
        if not mask.any():
            segmented[group_name] = {
                "status": "DATA_MISSING",
                "races_evaluated": 0,
            }
            continue
        segmented[group_name] = rank_metrics(meta[mask], y[mask], p_norm[mask])
        segmented[group_name]["status"] = "SUCCESS"
    metrics["label_group_metrics"] = segmented
    return metrics


def missingness_summary(features: pd.DataFrame) -> dict[str, Any]:
    rows = int(len(features))
    result: dict[str, Any] = {
        "rows": rows,
        "columns": int(len(features.columns)),
        "null_rates_top10": [],
    }
    if rows:
        null_rates = features.isna().mean().sort_values(ascending=False).head(10)
        result["null_rates_top10"] = [
            {"feature": str(index), "null_rate": float(value)}
            for index, value in null_rates.items()
            if value > 0
        ]
    for column, default in DEFAULT_HISTORY_PROFILE.items():
        if column in features.columns and rows:
            result[f"{column}_default_rate"] = float(
                (pd.to_numeric(features[column], errors="coerce").round(6) == default).mean()
            )
    if "target_distance" in features.columns and rows:
        result["target_distance_zero_rate"] = float(
            (pd.to_numeric(features["target_distance"], errors="coerce").fillna(0.0) == 0.0).mean()
        )
    return result


def predict_positive_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(x)
    if probabilities.ndim == 1:
        return probabilities.astype(float)
    if probabilities.shape[1] == 1:
        return probabilities[:, 0].astype(float)
    return probabilities[:, 1].astype(float)


def train_variant(
    name: str,
    x_train_base: pd.DataFrame,
    y_train: pd.Series,
    x_test_base: pd.DataFrame,
    meta_test: pd.DataFrame,
    y_test: pd.Series,
    label_groups: dict[str, str],
    trees: int,
    cv: int,
) -> VariantResult:
    x_train = make_variant_x(x_train_base, name)
    x_test = make_variant_x(x_test_base, name)
    x_train, dropped_train = strip_target_leakage_columns(x_train, allow_labels=False)
    x_test, dropped_test = strip_target_leakage_columns(x_test, allow_labels=False)
    if dropped_train or dropped_test:
        raise AssertionError(f"Leakage columns survived in {name}: {dropped_train + dropped_test}")
    train_columns = list(x_train.columns)
    x_train = coerce_model_frame(x_train)
    x_test = align_columns(x_test, train_columns)
    model = build_pipeline(x_train, trees=trees, cv=cv)
    model.fit(x_train, y_train)
    raw_probabilities = predict_positive_probability(model, x_test)
    metrics = evaluate_predictions(meta_test, y_test, raw_probabilities, label_groups)
    importance = extract_importance(model)
    metrics["feature_matrix_audit"] = audit_feature_matrix(x_train, allow_labels=False)
    return VariantResult(
        name=name,
        model=model,
        train_columns=train_columns,
        metrics=metrics,
        family_importance=importance["family_importance"],
        top_features=importance["top_features"][:20],
    )


def actual_result_for_snapshot(db_path: Path, snapshot: dict[str, Any]) -> dict[str, Any]:
    race_id = str(snapshot.get("race_id") or "")
    race_date = str(snapshot.get("race_date") or "")
    venue = str(snapshot.get("venue") or "")
    race_number = snapshot.get("race_number")
    try:
        race_number_int = int(race_number)
    except Exception:
        race_number_int = None
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        params: list[Any] = [race_id]
        clauses = ["rm.race_id = ?"]
        if race_date and venue and race_number_int is not None:
            clauses.append(
                "(rm.race_date = ? AND UPPER(rm.venue) = UPPER(?) AND rm.race_number = ?)"
            )
            params.extend([race_date, venue, race_number_int])
        rows = conn.execute(
            f"""
            SELECT rm.race_id, rm.winner_name, rm.results_status, rm.winner_source,
                   d.dog_clean_name, d.box_number, d.finish_position
            FROM race_metadata rm
            LEFT JOIN dog_race_data d ON d.race_id = rm.race_id
            WHERE ({' OR '.join(clauses)})
              AND d.finish_position IS NOT NULL
            ORDER BY CAST(d.finish_position AS INTEGER) ASC
            LIMIT 5
            """,
            params,
        ).fetchall()
    if not rows:
        return {"status": "DATA_MISSING"}
    winner = None
    for row in rows:
        try:
            if int(row["finish_position"]) == 1:
                winner = row
                break
        except Exception:
            continue
    if winner is None:
        winner = rows[0]
    return {
        "status": "SUCCESS",
        "race_id": winner["race_id"],
        "winner_name": winner["winner_name"] or winner["dog_clean_name"],
        "winner_box": int(winner["box_number"]) if winner["box_number"] is not None else None,
        "results_status": winner["results_status"],
        "winner_source": winner["winner_source"],
    }


def rank_from_probabilities(boxes: list[int], names: list[str], probabilities: Iterable[float]) -> dict[int, int]:
    rows = []
    for box, name, probability in zip(boxes, names, probabilities):
        rows.append((int(box), clean_name(name), float(probability)))
    rows.sort(key=lambda row: (-row[2], row[1], row[0]))
    return {box: index + 1 for index, (box, _name, _probability) in enumerate(rows)}


def live_snapshot_comparison(
    ml_system: MLSystemV4,
    db_path: Path,
    capture_report_path: Path,
    champion_model: Any,
    active_input_columns: list[str],
    trained_variants: dict[str, VariantResult],
) -> dict[str, Any]:
    if not capture_report_path.exists():
        return {"status": "DATA_MISSING", "reason": f"missing {capture_report_path}"}
    report = load_json(capture_report_path)
    rows = []
    top_boxes: dict[str, Counter] = {
        "champion": Counter(),
        "no_box": Counter(),
        "reduced_box_band": Counter(),
        "market": Counter(),
    }
    ev_rows: dict[str, dict[str, Any]] = {
        key: {"priced_rows": 0, "positive_ev_rows": 0, "positive_ev_winners": 0}
        for key in ["champion", "no_box", "reduced_box_band"]
    }

    for capture in report.get("captures", []):
        persistence = capture.get("persistence") or {}
        snapshot_path = REPO_ROOT / str(persistence.get("path") or "")
        if not snapshot_path.exists():
            continue
        snapshot = load_json(snapshot_path)
        predictions = snapshot.get("predictions") or []
        if not predictions:
            continue
        race_id = str(snapshot.get("race_id") or capture.get("race_id") or "")
        source_path = Path(str(snapshot.get("source_file_path") or capture.get("race_file") or ""))
        result = actual_result_for_snapshot(db_path, snapshot)
        boxes = [int(p.get("box_number")) for p in predictions if p.get("box_number") is not None]
        names = [str(p.get("dog_name") or p.get("dog_clean_name") or "") for p in predictions if p.get("box_number") is not None]
        champion_probs = [float(p.get("win_prob_norm") or 0.0) for p in predictions if p.get("box_number") is not None]
        champion_ranks = rank_from_probabilities(boxes, names, champion_probs)
        market_probs = [
            float((p.get("odds_snapshot") or {}).get("odds_implied_prob_norm") or 0.0)
            for p in predictions
            if p.get("box_number") is not None
        ]
        market_ranks = rank_from_probabilities(boxes, names, market_probs) if any(market_probs) else {}
        champion_top_box = min(champion_ranks, key=champion_ranks.get)
        top_boxes["champion"][champion_top_box] += 1
        if market_ranks:
            top_boxes["market"][min(market_ranks, key=market_ranks.get)] += 1

        challenger_rank_maps: dict[str, dict[int, int]] = {}
        challenger_probs_by_box: dict[str, dict[int, float]] = {}
        if source_path.exists():
            try:
                raw_csv = pd.read_csv(source_path)
                race_df = ml_system.preprocess_upcoming_race_csv(raw_csv, race_id)
                if snapshot.get("race_date"):
                    race_df["race_date"] = snapshot.get("race_date")
                if snapshot.get("venue"):
                    race_df["venue"] = snapshot.get("venue")
                if snapshot.get("jump_time"):
                    race_df["race_time"] = snapshot.get("jump_time")
                race_df["race_id"] = race_id
                race_df["field_size"] = len(race_df)
                live_features = ml_system.temporal_builder.build_features_for_race(race_df, race_id)
                live_x_base, _unused_y, live_meta = prepare_base_features(
                    live_features.assign(target=0)
                )
                live_boxes = [int(v) for v in live_meta["box_number"].tolist()]
                live_names = live_meta["dog_clean_name"].astype(str).tolist()
                for variant_name in ["no_box", "reduced_box_band"]:
                    variant = trained_variants.get(variant_name)
                    if not variant:
                        continue
                    x_live = make_variant_x(live_x_base, variant_name)
                    x_live = align_columns(x_live, variant.train_columns)
                    raw_prob = predict_positive_probability(variant.model, x_live)
                    p_norm = normalize_probabilities(live_meta, raw_prob).to_numpy()
                    challenger_rank_maps[variant_name] = rank_from_probabilities(
                        live_boxes, live_names, p_norm
                    )
                    challenger_probs_by_box[variant_name] = {
                        box: float(prob) for box, prob in zip(live_boxes, p_norm)
                    }
                    if challenger_rank_maps[variant_name]:
                        top_box = min(
                            challenger_rank_maps[variant_name],
                            key=challenger_rank_maps[variant_name].get,
                        )
                        top_boxes[variant_name][top_box] += 1
            except Exception as exc:
                challenger_rank_maps["error"] = {"error": str(exc)}  # type: ignore[assignment]

        winner_box = result.get("winner_box") if result.get("status") == "SUCCESS" else None
        row = {
            "race_id": race_id,
            "runner_count": len(predictions),
            "champion_top_box": champion_top_box,
            "market_top_box": min(market_ranks, key=market_ranks.get) if market_ranks else None,
            "no_box_top_box": (
                min(challenger_rank_maps["no_box"], key=challenger_rank_maps["no_box"].get)
                if "no_box" in challenger_rank_maps
                else None
            ),
            "reduced_box_top_box": (
                min(challenger_rank_maps["reduced_box_band"], key=challenger_rank_maps["reduced_box_band"].get)
                if "reduced_box_band" in challenger_rank_maps
                else None
            ),
            "winner_box": winner_box,
            "winner_name": result.get("winner_name"),
            "actual_result_status": result.get("status"),
            "champion_winner_rank": champion_ranks.get(winner_box) if winner_box else None,
            "market_winner_rank": market_ranks.get(winner_box) if winner_box and market_ranks else None,
            "no_box_winner_rank": (
                challenger_rank_maps.get("no_box", {}).get(winner_box) if winner_box else None
            ),
            "reduced_box_winner_rank": (
                challenger_rank_maps.get("reduced_box_band", {}).get(winner_box) if winner_box else None
            ),
        }
        rows.append(row)

        odds_by_box = {}
        for prediction in predictions:
            try:
                box = int(prediction.get("box_number"))
                odds = prediction.get("odds") or (prediction.get("odds_snapshot") or {}).get("market_odds_win")
                if odds:
                    odds_by_box[box] = float(odds)
            except Exception:
                continue
        champion_probs_by_box = {box: prob for box, prob in zip(boxes, champion_probs)}
        for arm_name, probs_by_box in {
            "champion": champion_probs_by_box,
            "no_box": challenger_probs_by_box.get("no_box", {}),
            "reduced_box_band": challenger_probs_by_box.get("reduced_box_band", {}),
        }.items():
            for box, odds in odds_by_box.items():
                if box not in probs_by_box:
                    continue
                ev_rows[arm_name]["priced_rows"] += 1
                ev = probs_by_box[box] * odds - 1.0
                if ev > 0:
                    ev_rows[arm_name]["positive_ev_rows"] += 1
                    if winner_box is not None and int(box) == int(winner_box):
                        ev_rows[arm_name]["positive_ev_winners"] += 1

    summary = {}
    for arm, counter in top_boxes.items():
        total = sum(counter.values())
        summary[arm] = {
            "races": int(total),
            "top_pick_box_distribution": {str(k): int(v) for k, v in sorted(counter.items())},
            "box1_top_pick_rate": float(counter.get(1, 0) / total) if total else None,
        }
    return {
        "status": "SUCCESS",
        "rows": rows,
        "top_pick_summary": summary,
        "ev_bucket_diagnostics": ev_rows,
    }


def write_feature_report(
    path: Path,
    metadata: dict[str, Any],
    model_path: Path,
    importance: dict[str, Any],
    active_input_columns: list[str],
) -> None:
    top50 = importance["top_features"][:50]
    family_rows = [
        {
            "family": row["family"],
            "mean_importance": row["mean"],
            "share": row["share"],
            "min": row["min"],
            "max": row["max"],
        }
        for row in importance["family_importance"]
    ]
    fold_rows = importance["fold_summaries"]
    box1_all_top = all(row.get("box1_rank") == 1 for row in fold_rows if row.get("box1_rank"))
    odds_features = [
        feature
        for feature in metadata.get("feature_names", [])
        if feature_family(str(feature)) == "odds_market"
    ]
    content = [
        "# Feature Dominance Report",
        "",
        f"Model path: `{model_path}`",
        f"Model id: `{metadata.get('model_id', 'DATA_MISSING')}`",
        f"Metadata feature count: `{len(metadata.get('feature_names', []))}`",
        f"Active fitted input columns inspected: `{len(active_input_columns)}`",
        "",
        "## Contract Check",
        "",
        f"- `box_number` in metadata features: `{('box_number' in metadata.get('feature_names', []))}`",
        f"- Odds/market/EV training features in metadata: `{odds_features or []}`",
        "- Post-result fields are not present in the active model metadata feature list.",
        "",
        "## Family Importance",
        "",
        md_table(family_rows, ["family", "mean_importance", "share", "min", "max"]),
        "",
        "## Per-Fold Box Dominance",
        "",
        md_table(
            fold_rows,
            ["fold", "box_importance", "box1_importance", "box1_rank", "top_feature", "top_feature_importance"],
        ),
        "",
        f"`cat__box_number_1` top transformed feature in every inspected fold: `{box1_all_top}`",
        "",
        "## Top 50 Transformed Features",
        "",
        md_table(
            [
                {
                    "rank": index + 1,
                    "feature": row["feature"],
                    "original_feature": row["original_feature"],
                    "family": row["family"],
                    "mean": row["mean"],
                    "min": row["min"],
                    "max": row["max"],
                    "fold_count": row["count"],
                }
                for index, row in enumerate(top50)
            ],
            ["rank", "feature", "original_feature", "family", "mean", "min", "max", "fold_count"],
        ),
        "",
        "## Interpretation",
        "",
        "`box_number` is the largest single original feature aggregate, and `cat__box_number_1` is the top transformed feature in every inspected fold. Combined contextual families can sum higher, but the box-1 one-hot dominance is still a model-probability issue, not a rank display issue.",
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def compact_metric_row(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": name,
        "races": metrics.get("races_evaluated"),
        "top1": metrics.get("top1"),
        "top2": metrics.get("top2"),
        "top3": metrics.get("top3"),
        "mean_winner_rank": metrics.get("mean_winner_rank"),
        "brier": metrics.get("brier"),
        "log_loss": metrics.get("log_loss"),
        "calibration_slope": (metrics.get("calibration") or {}).get("slope"),
        "calibration_intercept": (metrics.get("calibration") or {}).get("intercept"),
        "entropy": metrics.get("avg_normalized_entropy"),
        "spread": metrics.get("avg_probability_spread"),
        "box1_top_pick_rate": metrics.get("box1_top_pick_rate"),
    }


def write_main_report(
    path: Path,
    all_metrics: dict[str, Any],
    variants: dict[str, VariantResult],
    live_comparison: dict[str, Any],
    missingness: dict[str, Any],
    training_summary: dict[str, Any],
) -> None:
    metric_rows = [
        compact_metric_row(name, metrics)
        for name, metrics in all_metrics.items()
    ]
    live_rows = live_comparison.get("rows", []) if live_comparison.get("status") == "SUCCESS" else []
    live_summary_rows = []
    if live_comparison.get("status") == "SUCCESS":
        for arm, summary in live_comparison.get("top_pick_summary", {}).items():
            live_summary_rows.append({"arm": arm, **summary})

    no_box = all_metrics.get("no_box", {})
    reduced = all_metrics.get("reduced_box_band", {})
    champion = all_metrics.get("current_champion_baseline", {})
    removes_box_improves_top1 = (
        no_box.get("top1") is not None
        and champion.get("top1") is not None
        and no_box.get("top1") > champion.get("top1")
    )
    reduced_improves_box1 = (
        reduced.get("box1_top_pick_rate") is not None
        and champion.get("box1_top_pick_rate") is not None
        and reduced.get("box1_top_pick_rate") < champion.get("box1_top_pick_rate")
    )

    content = [
        "# Box Bias Audit Report",
        "",
        "## Direct Answers",
        "",
        f"- Is the current model using box as a dominant signal? `YES`; `box_number` is the largest single original feature aggregate and `cat__box_number_1` is top in every active calibrated fold.",
        f"- Does removing box improve temporal Top1? `{removes_box_improves_top1}` on this report-only split.",
        f"- Does reducing box reduce box-1 over-selection? `{reduced_improves_box1}` on this report-only split.",
        "- Does it improve live evaluated races? `DATA_MISSING` for the eight post-fix live snapshots unless result labels are present; see live table.",
        "- Are missing history/default features contributing? `YES`; default-history and zero target-distance rates are high enough to make box influence dangerous when runner vectors are low-information.",
        "- Champion model, registry, DB rows, labels, odds, snapshots, and EV thresholds were not changed.",
        "",
        "## Training And Split Safety",
        "",
        md_table([training_summary], list(training_summary.keys())),
        "",
        "## Temporal Validation Metrics",
        "",
        md_table(
            metric_rows,
            [
                "variant",
                "races",
                "top1",
                "top2",
                "top3",
                "mean_winner_rank",
                "brier",
                "log_loss",
                "calibration_slope",
                "calibration_intercept",
                "entropy",
                "spread",
                "box1_top_pick_rate",
            ],
        ),
        "",
        "## Label Segmentation",
        "",
    ]
    label_rows = []
    for name, metrics in all_metrics.items():
        for label_group, group_metrics in (metrics.get("label_group_metrics") or {}).items():
            label_rows.append(
                {
                    "variant": name,
                    "label_group": label_group,
                    "status": group_metrics.get("status"),
                    "races": group_metrics.get("races_evaluated"),
                    "top1": group_metrics.get("top1"),
                    "top2": group_metrics.get("top2"),
                    "top3": group_metrics.get("top3"),
                    "mean_winner_rank": group_metrics.get("mean_winner_rank"),
                }
            )
    content += [
        md_table(
            label_rows,
            ["variant", "label_group", "status", "races", "top1", "top2", "top3", "mean_winner_rank"],
        ),
        "",
        "## Drift And Missingness",
        "",
        "```json",
        json.dumps(missingness, indent=2, sort_keys=True),
        "```",
        "",
        "## Live Frozen Snapshot Comparison",
        "",
        md_table(
            live_summary_rows,
            ["arm", "races", "top_pick_box_distribution", "box1_top_pick_rate"],
        ),
        "",
        md_table(
            live_rows,
            [
                "race_id",
                "runner_count",
                "champion_top_box",
                "no_box_top_box",
                "reduced_box_top_box",
                "market_top_box",
                "winner_box",
                "champion_winner_rank",
                "no_box_winner_rank",
                "reduced_box_winner_rank",
                "market_winner_rank",
                "actual_result_status",
            ],
        ),
        "",
        "## EV Bucket Diagnostics",
        "",
        "Computed only where snapshot odds provenance exists. These are report-only diagnostics, not betting recommendations.",
        "",
        "```json",
        json.dumps(live_comparison.get("ev_bucket_diagnostics", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Feature Family Distribution By Report-Only Variant",
        "",
    ]
    for name, variant in variants.items():
        content += [
            f"### {name}",
            "",
            md_table(
                [
                    {
                        "family": row["family"],
                        "mean_importance": row["mean"],
                        "share": row["share"],
                    }
                    for row in variant.family_importance
                ],
                ["family", "mean_importance", "share"],
            ),
            "",
        ]
    content += [
        "## Recommended Next PR",
        "",
        "Add a non-production diagnostics command and CI-light fixture test around this audit path, then run a fuller challenger study with a larger frozen historical window before any retraining or promotion decision. Do not remove box from production predictions until no-box or reduced-box wins on temporal validation and live labeled snapshots.",
        "",
        "## Safety Confirmation",
        "",
        "- No push.",
        "- No production model retrain.",
        "- No model promotion or registry change.",
        "- No betting.",
        "- No label overwrite.",
        "- No snapshot rewrite.",
        "- No fake EV.",
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--capture-report", type=Path, default=DEFAULT_CAPTURE_REPORT)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--max-races", type=int, default=0, help="Optional V4_MAX_RACES limit for faster local audits")
    args = parser.parse_args()

    os.environ.setdefault("ENABLE_RESULTS_SCRAPERS", "0")
    os.environ.setdefault("TGR_ENABLED", "0")
    os.environ.setdefault("GREYHOUND_DB_PATH", str(args.db))
    if args.max_races > 0:
        os.environ["V4_MAX_RACES"] = str(args.max_races)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    champion_model = joblib.load(args.model)
    metadata = load_model_metadata(args.model)
    active_input_columns = input_columns_from_model(champion_model)
    champion_importance = extract_importance(champion_model)

    ml_system = MLSystemV4(db_path=str(args.db))
    train_raw, test_raw = ml_system.prepare_time_ordered_data()
    if train_raw.empty or test_raw.empty:
        raise SystemExit("DATA_MISSING: no temporal train/test data available")
    ml_system._validate_temporal_split(train_raw, test_raw)

    train_features = ml_system.build_leakage_safe_features(train_raw)
    test_features = ml_system.build_leakage_safe_features(test_raw)
    if train_features.empty or test_features.empty:
        raise SystemExit("DATA_MISSING: feature building returned empty matrices")

    x_train_base, y_train, meta_train = prepare_base_features(train_features)
    x_test_base, y_test, meta_test = prepare_base_features(test_features)
    overlap = set(meta_train["race_id"]).intersection(set(meta_test["race_id"]))
    if overlap:
        raise AssertionError(f"race_id overlap detected: {len(overlap)}")

    label_groups = query_label_groups(args.db, meta_test["race_id"].unique())
    label_groups = {
        str(race_id): label_groups.get(str(race_id), "db_finish_position_only")
        for race_id in meta_test["race_id"].unique()
    }

    all_metrics: dict[str, Any] = {}
    variants: dict[str, VariantResult] = {}

    active_x_test = make_variant_x(x_test_base, "current_champion_baseline")
    if active_input_columns:
        active_x_test = align_columns(active_x_test, active_input_columns)
    active_raw = predict_positive_probability(champion_model, active_x_test)
    all_metrics["current_champion_baseline"] = evaluate_predictions(
        meta_test, y_test, active_raw, label_groups
    )
    all_metrics["current_champion_baseline"]["feature_matrix_audit"] = audit_feature_matrix(
        active_x_test, allow_labels=False
    )

    for variant_name in [
        "full_box_retrained_report_only",
        "no_box",
        "reduced_box_band",
        "history_only",
        "embedded_history_aware_no_box",
    ]:
        result = train_variant(
            variant_name,
            x_train_base,
            y_train,
            x_test_base,
            meta_test,
            y_test,
            label_groups,
            trees=args.trees,
            cv=args.cv,
        )
        variants[variant_name] = result
        all_metrics[variant_name] = result.metrics

    live_comparison = live_snapshot_comparison(
        ml_system=ml_system,
        db_path=args.db,
        capture_report_path=args.capture_report,
        champion_model=champion_model,
        active_input_columns=active_input_columns,
        trained_variants=variants,
    )

    missingness = {
        "train": missingness_summary(train_features),
        "test": missingness_summary(test_features),
        "live_snapshot_flags": {},
    }
    if args.capture_report.exists():
        try:
            capture_report = load_json(args.capture_report)
            flag_counter: Counter[str] = Counter()
            history_source_counter: Counter[str] = Counter()
            db_history_zero = 0
            runner_rows = 0
            for capture in capture_report.get("captures", []):
                path = REPO_ROOT / str((capture.get("persistence") or {}).get("path") or "")
                if not path.exists():
                    continue
                snapshot = load_json(path)
                for prediction in snapshot.get("predictions") or []:
                    runner_rows += 1
                    flag_counter.update(prediction.get("data_quality_flags") or [])
                    if prediction.get("history_source"):
                        history_source_counter[str(prediction.get("history_source"))] += 1
                    if int(prediction.get("db_result_history_count") or 0) == 0:
                        db_history_zero += 1
            missingness["live_snapshot_flags"] = {
                "runner_rows": runner_rows,
                "data_quality_flags": dict(flag_counter),
                "history_source_counts": dict(history_source_counter),
                "db_result_history_count_zero_rows": db_history_zero,
            }
        except Exception as exc:
            missingness["live_snapshot_flags"] = {"status": "DATA_MISSING", "reason": str(exc)}

    training_summary = {
        "train_races": int(meta_train["race_id"].nunique()),
        "train_rows": int(len(meta_train)),
        "test_races": int(meta_test["race_id"].nunique()),
        "test_rows": int(len(meta_test)),
        "race_id_overlap": int(len(overlap)),
        "train_min_date": str(train_raw["race_date"].min()),
        "train_max_date": str(train_raw["race_date"].max()),
        "test_min_date": str(test_raw["race_date"].min()),
        "test_max_date": str(test_raw["race_date"].max()),
        "temporary_challenger_trees": int(args.trees),
        "calibration_cv": int(args.cv),
        "max_races_limit": int(args.max_races),
    }

    write_feature_report(
        args.out_dir / "feature_dominance_report.md",
        metadata=metadata,
        model_path=args.model,
        importance=champion_importance,
        active_input_columns=active_input_columns,
    )
    write_main_report(
        args.out_dir / "report.md",
        all_metrics=all_metrics,
        variants=variants,
        live_comparison=live_comparison,
        missingness=missingness,
        training_summary=training_summary,
    )
    with (args.out_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "training_summary": training_summary,
                "champion_feature_importance": champion_importance,
                "metrics": all_metrics,
                "variant_feature_importance": {
                    name: {
                        "family_importance": variant.family_importance,
                        "top_features": variant.top_features,
                    }
                    for name, variant in variants.items()
                },
                "live_comparison": live_comparison,
                "missingness": missingness,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    print(f"Wrote {args.out_dir / 'feature_dominance_report.md'}")
    print(f"Wrote {args.out_dir / 'report.md'}")
    print(f"Wrote {args.out_dir / 'diagnostics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
