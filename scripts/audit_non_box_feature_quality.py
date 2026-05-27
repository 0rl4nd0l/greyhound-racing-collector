#!/usr/bin/env python3
"""Report-only non-box feature quality audit for frozen prediction snapshots.

The script reads existing snapshot JSON files and, when source CSVs are
available, reconstructs the active model input surface for diagnostics only. It
does not retrain, promote, write labels, rewrite snapshots, or mutate the DB.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SNAPSHOT_ROOT = REPO_ROOT / "artifacts" / "prediction_snapshots"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts"
    / "full_evidence_orchestration_20260525"
    / "non_box_feature_quality_audit"
)
DEFAULT_MODEL = REPO_ROOT / "model_registry" / "best_model.joblib"
DEFAULT_DB = REPO_ROOT / "greyhound_racing_data_writable.db"

CATEGORICAL_COLUMNS = {"venue", "grade", "track_condition", "weather", "trainer_name"}
BOX_FEATURE_PATTERNS = ("box_number", "current_box", "box_win_rate")
DEFAULT_NUMERIC_VALUES = {
    "historical_avg_position": 4.5,
    "historical_best_position": 4.0,
    "historical_win_rate": 0.125,
    "historical_place_rate": 0.375,
    "historical_form_trend": 0.0,
    "historical_avg_time": 30.0,
    "historical_best_time": 29.0,
    "historical_time_consistency": 2.0,
    "venue_specific_avg_position": 4.5,
    "venue_specific_win_rate": 0.125,
    "venue_experience": 0.0,
    "venue_best_position": 4.0,
    "grade_specific_avg_position": 4.5,
    "grade_specific_win_rate": 0.125,
    "grade_experience": 0.0,
    "days_since_last_race": 30.0,
    "race_frequency": 2.0,
    "best_distance_avg_position": 4.5,
    "best_distance_win_rate": 0.125,
    "distance_adjusted_time": 0.0,
    "target_distance": 0.0,
}
TGR_COLUMNS = [
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
REPORT_ONLY_EMBEDDED_COLUMNS = [
    "embedded_history_race_count",
    "embedded_history_recent_count",
    "embedded_history_avg_finish",
    "embedded_history_best_finish",
    "embedded_history_win_rate",
    "embedded_history_place_rate",
    "embedded_history_avg_time",
    "embedded_history_best_time",
    "embedded_history_recent_avg_time",
    "embedded_history_same_track_count",
    "embedded_history_same_distance_band_count",
    "embedded_history_recency_days_min",
    "embedded_history_recency_days_max",
    "embedded_history_recency_days_mean",
]


def clean_name(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    body = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join(body)


def default_snapshot_dir() -> Path:
    dated_dirs = [
        path
        for path in DEFAULT_SNAPSHOT_ROOT.iterdir()
        if path.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.name)
    ]
    if not dated_dirs:
        return DEFAULT_SNAPSHOT_ROOT
    return sorted(dated_dirs, key=lambda p: p.name)[-1]


def iter_snapshot_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    for path in paths:
        if path.is_dir():
            result.extend(
                p
                for p in path.rglob("*.json")
                if "evaluation" not in p.name and "result" not in p.name
            )
        elif path.is_file():
            result.append(path)
    return sorted(dict.fromkeys(result))


def load_model_metadata(model_path: Path, metadata_path: Path | None) -> dict[str, Any]:
    if metadata_path:
        return load_json(metadata_path)
    resolved = model_path.resolve()
    candidate = (
        REPO_ROOT
        / "model_registry"
        / "metadata"
        / resolved.name.replace("_model.joblib", "_metadata.json")
    )
    if candidate.exists():
        return load_json(candidate)
    best = REPO_ROOT / "model_registry" / "best_metadata.json"
    if best.exists():
        return load_json(best)
    return {}


def iter_calibrated_estimators(model: Any):
    calibrated = getattr(model, "calibrated_classifiers_", None)
    if calibrated:
        for fold in calibrated:
            estimator = getattr(fold, "estimator", None) or getattr(
                fold, "base_estimator", None
            )
            if estimator is not None:
                yield estimator
        return
    base = getattr(model, "base_estimator_", None)
    if base is not None:
        yield base
    yield model


def input_columns_from_model(model: Any) -> list[str]:
    for estimator in iter_calibrated_estimators(model):
        if not hasattr(estimator, "named_steps"):
            continue
        preprocessor = estimator.named_steps.get("preprocessor")
        if preprocessor is None or not hasattr(preprocessor, "transformers_"):
            continue
        columns: list[str] = []
        for name, _transformer, transformer_cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            if isinstance(transformer_cols, (list, tuple)):
                columns.extend([str(c) for c in transformer_cols])
        if columns:
            return list(dict.fromkeys(columns))
    return []


def model_feature_contract(
    model_path: Path,
    metadata_path: Path | None,
    no_model_load: bool,
) -> tuple[list[str], dict[str, Any]]:
    metadata = load_model_metadata(model_path, metadata_path)
    features = [str(v) for v in metadata.get("feature_names", [])]
    if features or no_model_load:
        return features, metadata
    import joblib

    model = joblib.load(model_path)
    features = input_columns_from_model(model)
    return features, metadata


def feature_family(column: str) -> str:
    lowered = str(column).lower()
    if any(token in lowered for token in ("box_number", "current_box", "box_win_rate")):
        return "box"
    if lowered.startswith("tgr_"):
        return "tgr"
    if (
        lowered.startswith("historical_")
        or lowered.startswith("venue_specific_")
        or lowered.startswith("grade_specific_")
        or lowered.startswith("best_distance_")
        or lowered in {"days_since_last_race", "race_frequency", "target_distance"}
    ):
        return "history"
    if lowered in {"venue", "grade", "distance", "field_size", "track_condition", "weather"}:
        return "target_context"
    return "other"


def stable_race_key(snapshot: dict[str, Any]) -> str:
    return "|".join(
        [
            str(snapshot.get("race_date") or ""),
            str(snapshot.get("venue") or ""),
            str(snapshot.get("race_number") or ""),
        ]
    )


def prediction_name(prediction: dict[str, Any]) -> str:
    return str(
        prediction.get("dog_clean_name")
        or prediction.get("dog_name")
        or prediction.get("name")
        or ""
    )


def normalized_value(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "NA"
    except Exception:
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.6f}"
    text = str(value).strip()
    if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        try:
            return f"{float(text):.6f}"
        except Exception:
            return text
    return text.upper()


def vector_signature(values: list[Any]) -> str:
    payload = "\x1f".join(normalized_value(value) for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def numeric_default_features(feature_row: pd.Series, model_columns: list[str]) -> list[str]:
    defaults: list[str] = []
    for column in model_columns:
        if column not in feature_row.index:
            continue
        value = pd.to_numeric(pd.Series([feature_row.get(column)]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        if column in DEFAULT_NUMERIC_VALUES and abs(float(value) - DEFAULT_NUMERIC_VALUES[column]) < 1e-9:
            defaults.append(column)
        elif column in TGR_COLUMNS and abs(float(value)) < 1e-9:
            defaults.append(column)
    return defaults


def default_flags(row: dict[str, Any], feature_row: pd.Series, model_columns: list[str]) -> list[str]:
    flags: list[str] = []
    if row.get("distance_source") == "default_missing_target":
        flags.append("default_distance_source")
    if row.get("grade_source") == "default_missing_target":
        flags.append("default_grade_source")
    if "target_distance" in model_columns:
        target_distance = pd.to_numeric(
            pd.Series([feature_row.get("target_distance")]), errors="coerce"
        ).fillna(0.0).iloc[0]
        if float(target_distance) == 0.0:
            flags.append("target_distance_zero")
    tgr_cols = [col for col in model_columns if col.startswith("tgr_")]
    if tgr_cols:
        tgr_sum = float(
            pd.to_numeric(feature_row.reindex(tgr_cols), errors="coerce").fillna(0.0).sum()
        )
        if abs(tgr_sum) < 1e-12:
            flags.append("tgr_all_zero")
    history_defaults = {
        col
        for col in DEFAULT_NUMERIC_VALUES
        if col.startswith("historical_") and col in model_columns
    }
    row_defaults = set(numeric_default_features(feature_row, model_columns))
    if history_defaults and len(history_defaults.intersection(row_defaults)) >= 3:
        flags.append("history_profile_mostly_default")
    return flags


class FeatureReconstructor:
    def __init__(self, db_path: Path, model_columns: list[str]) -> None:
        os.environ.setdefault("ENABLE_RESULTS_SCRAPERS", "0")
        os.environ.setdefault("INGEST_EMBEDDED_HISTORY_ON_PREDICT", "0")
        os.environ.setdefault("TGR_FEATURES_ENABLED", "0")
        os.environ.setdefault("GREYHOUND_DB_PATH", str(db_path))
        from prediction_pipeline_v4 import PredictionPipelineV4
        from utils.leakage_guard import strip_target_leakage_columns

        self.pipeline = PredictionPipelineV4(db_path=str(db_path))
        self.strip_target_leakage_columns = strip_target_leakage_columns
        self.model_columns = model_columns

    def reconstruct(
        self, snapshot: dict[str, Any], source_file_path: Path
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
        raw = pd.read_csv(source_file_path, sep=None, engine="python", encoding="utf-8-sig")
        mapped = self.pipeline._map_csv_to_v4_format(raw, str(source_file_path))
        mapped, _dropped = self.strip_target_leakage_columns(mapped, allow_labels=False)
        mapped = self.pipeline._annotate_history_provenance(mapped)
        race_id = str(snapshot.get("race_id") or source_file_path.stem)
        features = self.pipeline.ml_system_v4.temporal_builder.build_features_for_race(
            mapped.copy(), race_id
        )
        x_original = features.drop(
            ["race_id", "dog_clean_name", "target", "target_timestamp", "race_time"],
            axis=1,
            errors="ignore",
        )
        x_original, _dropped = self.strip_target_leakage_columns(
            x_original, allow_labels=False
        )
        missing = [col for col in self.model_columns if col not in x_original.columns]
        extra = [col for col in x_original.columns if col not in self.model_columns]
        x = x_original.reindex(columns=self.model_columns, fill_value=0.0)
        for cat_col in CATEGORICAL_COLUMNS:
            if cat_col in x.columns:
                default = "UNKNOWN" if cat_col in {"venue", "grade"} else "Unknown"
                x[cat_col] = (
                    x[cat_col]
                    .replace({0: default, "0": default})
                    .fillna(default)
                    .astype(str)
                )
        for column in x.columns:
            if column in CATEGORICAL_COLUMNS:
                continue
            if x[column].dtype == object:
                x[column] = (
                    x[column]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .replace({"true": "1", "false": "0"})
                )
            x[column] = pd.to_numeric(x[column], errors="coerce").fillna(0.0)
        if "dog_clean_name" in features.columns:
            x["dog_clean_name"] = features["dog_clean_name"].values
        return mapped, x, missing, extra


def row_for_prediction(
    prediction: dict[str, Any],
    snapshot: dict[str, Any],
    snapshot_path: Path,
    source_file_path: str,
    mapped_by_name: dict[str, pd.Series],
    mapped_by_box: dict[int, pd.Series],
    features_by_name: dict[str, pd.Series],
    features_by_box: dict[int, pd.Series],
    model_columns: list[str],
    missing_columns: list[str],
    extra_columns: list[str],
) -> dict[str, Any]:
    name = prediction_name(prediction)
    box = prediction.get("box_number")
    try:
        box_int = int(box)
    except Exception:
        box_int = 0
    name_key = clean_name(name)
    mapped_row = mapped_by_name.get(name_key)
    if mapped_row is None:
        mapped_row = mapped_by_box.get(box_int)
    if mapped_row is None:
        mapped_row = pd.Series(dtype=object)
    feature_row = features_by_name.get(name_key)
    if feature_row is None:
        feature_row = features_by_box.get(box_int)
    if feature_row is None:
        feature_row = pd.Series(index=model_columns, dtype=object)

    predicted_rank = prediction.get("predicted_rank")
    try:
        is_top_pick = int(predicted_rank) == 1
    except Exception:
        is_top_pick = False
    embedded_count = (
        mapped_row.get("embedded_history_race_count")
        if "embedded_history_race_count" in mapped_row.index
        else mapped_row.get("csv_historical_races", prediction.get("csv_historical_races", 0))
    )
    row: dict[str, Any] = {
        "race_id": snapshot.get("race_id"),
        "stable_race_key": stable_race_key(snapshot),
        "snapshot_path": str(snapshot_path.relative_to(REPO_ROOT) if snapshot_path.is_relative_to(REPO_ROOT) else snapshot_path),
        "source_file_path": source_file_path,
        "dog_name": name,
        "box_number": box_int,
        "is_box_1": int(box_int == 1),
        "predicted_rank": predicted_rank,
        "is_top_pick": int(is_top_pick),
        "win_prob_norm": prediction.get("win_prob_norm"),
        "win_prob_raw": prediction.get("win_prob_raw"),
        "history_source": mapped_row.get("history_source", prediction.get("history_source")),
        "history_match_status": mapped_row.get(
            "history_match_status", prediction.get("history_match_status")
        ),
        "db_history_match_status": mapped_row.get(
            "db_history_match_status", prediction.get("db_history_match_status")
        ),
        "db_result_history_count": mapped_row.get(
            "db_result_history_count", prediction.get("db_result_history_count", 0)
        ),
        "embedded_csv_history_count": embedded_count,
        "csv_prefixed_history_rows": mapped_row.get(
            "csv_prefixed_history_rows", prediction.get("csv_prefixed_history_rows", 0)
        ),
        "csv_blank_history_rows": mapped_row.get(
            "csv_blank_history_rows", prediction.get("csv_blank_history_rows", 0)
        ),
        "csv_historical_sources": mapped_row.get(
            "csv_historical_sources", prediction.get("csv_historical_sources", "")
        ),
        "distance_source": mapped_row.get(
            "distance_source", prediction.get("distance_source", "DATA_MISSING")
        ),
        "grade_source": mapped_row.get(
            "grade_source", prediction.get("grade_source", "DATA_MISSING")
        ),
        "metadata_is_leakage_safe": bool(
            mapped_row.get(
                "metadata_is_leakage_safe",
                prediction.get("metadata_is_leakage_safe", False),
            )
        ),
        "rejected_metadata_sources": ";".join(
            str(v)
            for v in (
                mapped_row.get("rejected_metadata_sources")
                or prediction.get("rejected_metadata_sources")
                or []
            )
        ),
        "missing_expected_model_columns": ";".join(missing_columns),
        "extra_reconstructed_columns": ";".join(extra_columns),
    }

    for column in REPORT_ONLY_EMBEDDED_COLUMNS:
        row[column] = mapped_row.get(column, prediction.get(column))
    row["embedded_history_sources"] = mapped_row.get(
        "embedded_history_sources", prediction.get("embedded_history_sources", "")
    )

    numeric_model_cols = [col for col in model_columns if col not in CATEGORICAL_COLUMNS]
    numeric_values = pd.to_numeric(
        feature_row.reindex(numeric_model_cols), errors="coerce"
    )
    default_features = numeric_default_features(feature_row, model_columns)
    flags = default_flags(row, feature_row, model_columns)
    non_box_cols = [
        col
        for col in model_columns
        if not any(pattern in col.lower() for pattern in BOX_FEATURE_PATTERNS)
    ]
    history_cols = [
        col for col in non_box_cols if feature_family(col) in {"history", "tgr"}
    ]
    row["non_null_numeric_feature_count"] = int(numeric_values.notna().sum())
    row["zero_numeric_feature_count"] = int(
        (numeric_values.fillna(0.0).astype(float) == 0.0).sum()
    )
    row["default_numeric_feature_count"] = len(default_features)
    row["default_numeric_features"] = ";".join(default_features)
    row["default_missing_feature_flags"] = ";".join(flags)
    row["historical_avg_position"] = feature_row.get("historical_avg_position")
    row["historical_win_rate"] = feature_row.get("historical_win_rate")
    row["historical_place_rate"] = feature_row.get("historical_place_rate")
    row["historical_avg_time"] = feature_row.get("historical_avg_time")
    row["days_since_last_race"] = feature_row.get("days_since_last_race")
    row["target_distance_feature"] = feature_row.get("target_distance")
    row["distance_feature"] = feature_row.get("distance")
    row["field_size_feature"] = feature_row.get("field_size")
    row["tgr_total_races"] = feature_row.get("tgr_total_races", 0)
    row["__non_box_cols"] = non_box_cols
    row["__history_cols"] = history_cols
    row["__non_box_vector"] = [feature_row.get(col) for col in non_box_cols]
    row["__history_vector"] = [feature_row.get(col) for col in history_cols]
    row["non_box_model_vector_signature"] = vector_signature(row["__non_box_vector"])
    row["history_feature_vector_signature"] = vector_signature(row["__history_vector"])
    return row


def add_similarity_metrics(rows: list[dict[str, Any]]) -> None:
    rows_by_race: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_race[str(row.get("race_id") or "")].append(row)

    for race_rows in rows_by_race.values():
        non_box_counts = Counter(row["non_box_model_vector_signature"] for row in race_rows)
        history_counts = Counter(row["history_feature_vector_signature"] for row in race_rows)
        non_box_cols = race_rows[0].get("__non_box_cols", [])
        history_cols = race_rows[0].get("__history_cols", [])
        constant_non_box = _constant_feature_count(race_rows, "__non_box_vector", non_box_cols)
        constant_history = _constant_feature_count(race_rows, "__history_vector", history_cols)
        for row in race_rows:
            equal_shares = [
                _equal_share(row["__non_box_vector"], peer["__non_box_vector"])
                for peer in race_rows
                if peer is not row
            ]
            row["non_box_model_vector_duplicate_count_within_race"] = int(
                non_box_counts[row["non_box_model_vector_signature"]]
            )
            row["history_feature_vector_duplicate_count_within_race"] = int(
                history_counts[row["history_feature_vector_signature"]]
            )
            row["non_box_unique_vectors_in_race"] = int(len(non_box_counts))
            row["history_unique_vectors_in_race"] = int(len(history_counts))
            row["non_box_equal_share_to_most_similar_peer"] = (
                max(equal_shares) if equal_shares else 0.0
            )
            row["non_box_near_duplicate_peer_count_ge_80pct_equal"] = int(
                sum(1 for share in equal_shares if share >= 0.8)
            )
            row["non_box_near_duplicate_peer_count_ge_90pct_equal"] = int(
                sum(1 for share in equal_shares if share >= 0.9)
            )
            row["non_box_constant_feature_count_within_race"] = constant_non_box
            row["non_box_constant_feature_share_within_race"] = (
                constant_non_box / len(non_box_cols) if non_box_cols else 0.0
            )
            row["history_constant_feature_count_within_race"] = constant_history
            row["history_constant_feature_share_within_race"] = (
                constant_history / len(history_cols) if history_cols else 0.0
            )


def _equal_share(left: list[Any], right: list[Any]) -> float:
    if not left:
        return 0.0
    equal = sum(1 for a, b in zip(left, right) if normalized_value(a) == normalized_value(b))
    return float(equal) / float(len(left))


def _constant_feature_count(
    rows: list[dict[str, Any]], vector_key: str, columns: list[str]
) -> int:
    if not rows or not columns:
        return 0
    count = 0
    for index, _column in enumerate(columns):
        values = {normalized_value(row[vector_key][index]) for row in rows}
        if len(values) <= 1:
            count += 1
    return count


def build_audit_rows(
    snapshot_paths: list[Path],
    model_columns: list[str],
    db_path: Path,
    reconstruct: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    reconstructor = FeatureReconstructor(db_path, model_columns) if reconstruct else None

    for snapshot_path in snapshot_paths:
        try:
            snapshot = load_json(snapshot_path)
        except Exception as exc:
            errors.append(f"{snapshot_path}:load_error:{exc}")
            continue
        predictions = snapshot.get("predictions") or []
        if not isinstance(predictions, list) or not predictions:
            continue

        source_value = str(snapshot.get("source_file_path") or "")
        source_path = Path(source_value)
        if source_value and not source_path.is_absolute():
            source_path = REPO_ROOT / source_path
        mapped = pd.DataFrame()
        features = pd.DataFrame()
        missing_columns: list[str] = []
        extra_columns: list[str] = []
        if reconstructor is not None:
            if source_value and source_path.exists():
                try:
                    mapped, features, missing_columns, extra_columns = reconstructor.reconstruct(
                        snapshot, source_path
                    )
                except Exception as exc:
                    errors.append(f"{snapshot_path}:reconstruct_error:{exc}")
            else:
                errors.append(f"{snapshot_path}:source_file_missing:{source_value}")

        mapped_by_name: dict[str, pd.Series] = {}
        mapped_by_box: dict[int, pd.Series] = {}
        features_by_name: dict[str, pd.Series] = {}
        features_by_box: dict[int, pd.Series] = {}
        for _, mapped_row in mapped.iterrows():
            mapped_by_name[clean_name(mapped_row.get("dog_clean_name"))] = mapped_row
            try:
                mapped_by_box[int(mapped_row.get("box_number"))] = mapped_row
            except Exception:
                pass
        for _, feature_row in features.iterrows():
            features_by_name[clean_name(feature_row.get("dog_clean_name"))] = feature_row
            try:
                features_by_box[int(mapped_by_name[clean_name(feature_row.get("dog_clean_name"))].get("box_number"))] = feature_row
            except Exception:
                pass

        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            rows.append(
                row_for_prediction(
                    prediction,
                    snapshot,
                    snapshot_path,
                    source_value,
                    mapped_by_name,
                    mapped_by_box,
                    features_by_name,
                    features_by_box,
                    model_columns,
                    missing_columns,
                    extra_columns,
                )
            )

    add_similarity_metrics(rows)
    return rows, errors


def summarize(rows: list[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    runner_rows = len(rows)
    race_ids = {str(row.get("race_id") or "") for row in rows}
    distance_counter = Counter(str(row.get("distance_source") or "DATA_MISSING") for row in rows)
    grade_counter = Counter(str(row.get("grade_source") or "DATA_MISSING") for row in rows)
    history_counter = Counter(str(row.get("history_source") or "DATA_MISSING") for row in rows)
    rejected_metadata_counter = Counter()
    for row in rows:
        for item in str(row.get("rejected_metadata_sources") or "").split(";"):
            if item:
                rejected_metadata_counter[item] += 1
    top_pick_box_counter = Counter(
        str(row.get("box_number")) for row in rows if int(row.get("is_top_pick") or 0) == 1
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runner_rows": runner_rows,
        "races": len(race_ids),
        "distance_source_counts": dict(distance_counter),
        "grade_source_counts": dict(grade_counter),
        "rejected_metadata_source_counts": dict(rejected_metadata_counter),
        "history_source_counts": dict(history_counter),
        "db_result_history_zero_rows": int(
            sum(int(row.get("db_result_history_count") or 0) == 0 for row in rows)
        ),
        "embedded_history_rows": int(
            sum(float(row.get("embedded_csv_history_count") or 0) > 0 for row in rows)
        ),
        "near_duplicate_rows_ge80pct_equal_peer": int(
            sum(
                int(row.get("non_box_near_duplicate_peer_count_ge_80pct_equal") or 0) > 0
                for row in rows
            )
        ),
        "near_duplicate_rows_ge90pct_equal_peer": int(
            sum(
                int(row.get("non_box_near_duplicate_peer_count_ge_90pct_equal") or 0) > 0
                for row in rows
            )
        ),
        "exact_non_box_duplicate_rows": int(
            sum(
                int(row.get("non_box_model_vector_duplicate_count_within_race") or 0) > 1
                for row in rows
            )
        ),
        "mean_most_similar_non_box_equal_share": _mean(
            row.get("non_box_equal_share_to_most_similar_peer") for row in rows
        ),
        "mean_constant_non_box_feature_share": _mean(
            row.get("non_box_constant_feature_share_within_race") for row in rows
        ),
        "top_pick_box_distribution": dict(top_pick_box_counter),
        "box1_top_pick_races": int(top_pick_box_counter.get("1", 0)),
        "source_errors": errors,
        "embedded_history_richness": embedded_history_richness(rows),
    }


def embedded_history_richness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mean_race_count": _mean(row.get("embedded_history_race_count") for row in rows),
        "mean_recent_count": _mean(row.get("embedded_history_recent_count") for row in rows),
        "mean_same_track_count": _mean(
            row.get("embedded_history_same_track_count") for row in rows
        ),
        "mean_same_distance_band_count": _mean(
            row.get("embedded_history_same_distance_band_count") for row in rows
        ),
        "rows_with_recency_days": int(
            sum(row.get("embedded_history_recency_days_mean") not in (None, "", "nan") for row in rows)
        ),
        "mean_avg_finish": _mean(row.get("embedded_history_avg_finish") for row in rows),
        "mean_win_rate": _mean(row.get("embedded_history_win_rate") for row in rows),
        "mean_place_rate": _mean(row.get("embedded_history_place_rate") for row in rows),
    }


def _mean(values) -> float | None:
    nums = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if nums.empty:
        return None
    return float(nums.mean())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    public_rows = [
        {key: value for key, value in row.items() if not key.startswith("__")}
        for row in rows
    ]
    fieldnames: list[str] = []
    for row in public_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(public_rows)


def load_baseline_summary(output_dir: Path, baseline_csv: Path | None) -> dict[str, Any] | None:
    candidates = []
    if baseline_csv:
        candidates.append(baseline_csv)
    candidates.append(
        output_dir.parent
        / "non_box_feature_quality_audit"
        / "live_feature_missingness.csv"
    )
    for candidate in candidates:
        if candidate.exists():
            try:
                df = pd.read_csv(candidate)
                return {
                    "runner_rows": int(len(df)),
                    "distance_source_counts": df.get("distance_source", pd.Series(dtype=str)).fillna("DATA_MISSING").value_counts().to_dict(),
                    "grade_source_counts": df.get("grade_source", pd.Series(dtype=str)).fillna("DATA_MISSING").value_counts().to_dict(),
                    "near_duplicate_rows_ge80pct_equal_peer": int(
                        (
                            pd.to_numeric(
                                df.get(
                                    "non_box_near_duplicate_peer_count_ge_80pct_equal",
                                    pd.Series(dtype=float),
                                ),
                                errors="coerce",
                            ).fillna(0)
                            > 0
                        ).sum()
                    ),
                    "top_pick_box_distribution": df[df.get("is_top_pick", pd.Series(dtype=int)).fillna(0).astype(int) == 1].get("box_number", pd.Series(dtype=str)).astype(str).value_counts().to_dict(),
                    "embedded_history_richness": {
                        "mean_race_count": _mean(
                            df.get("embedded_history_race_count", df.get("embedded_csv_history_count", pd.Series(dtype=float)))
                        ),
                        "mean_recent_count": _mean(
                            df.get("embedded_history_recent_count", pd.Series(dtype=float))
                        ),
                        "mean_same_track_count": _mean(
                            df.get("embedded_history_same_track_count", pd.Series(dtype=float))
                        ),
                    },
                }
            except Exception:
                continue
    return None


def race_level_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("race_id") or "")].append(row)
    result = []
    for race_id, race_rows in sorted(grouped.items()):
        top_pick = next((row for row in race_rows if int(row.get("is_top_pick") or 0) == 1), {})
        result.append(
            {
                "race_id": race_id,
                "runner_count": len(race_rows),
                "top_pick_box": top_pick.get("box_number", "DATA_MISSING"),
                "rows_with_ge80pct_equal_peer": sum(
                    int(row.get("non_box_near_duplicate_peer_count_ge_80pct_equal") or 0) > 0
                    for row in race_rows
                ),
                "mean_most_similar_equal_share": _mean(
                    row.get("non_box_equal_share_to_most_similar_peer")
                    for row in race_rows
                ),
                "all_distance_source_default": int(
                    all(row.get("distance_source") == "default_missing_target" for row in race_rows)
                ),
                "all_grade_source_default": int(
                    all(row.get("grade_source") == "default_missing_target" for row in race_rows)
                ),
            }
        )
    return result


def write_report(
    path: Path,
    summary: dict[str, Any],
    baseline: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    snapshot_arg: list[Path],
    csv_path: Path,
    json_path: Path,
) -> None:
    before = baseline or {"status": "DATA_MISSING"}
    after = summary
    richness = after.get("embedded_history_richness") or {}
    distance_sources = after.get("distance_source_counts") or {}
    grade_sources = after.get("grade_source_counts") or {}
    default_distance_rows = int(distance_sources.get("default_missing_target", 0) or 0)
    default_grade_rows = int(grade_sources.get("default_missing_target", 0) or 0)
    safe_distance_rows = max(0, int(after.get("runner_rows") or 0) - default_distance_rows)
    safe_grade_rows = max(0, int(after.get("runner_rows") or 0) - default_grade_rows)
    content = [
        "# Non-Box Feature Quality Audit",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Scope",
        "",
        f"- Snapshot input: `{', '.join(str(p) for p in snapshot_arg)}`",
        f"- Runner rows audited: `{summary['runner_rows']}` across `{summary['races']}` races.",
        f"- Runner CSV: `{csv_path}`",
        f"- JSON summary: `{json_path}`",
        "- Mode: report-only; no retraining, model promotion, DB writes, betting, label writes, or snapshot rewrites.",
        "- Baseline comparison uses the previous audit CSV when present; compare row counts before interpreting deltas.",
        "",
        "## Before/After Target Metadata",
        "",
        "```json",
        json.dumps(
            {
                "before": {
                    "runner_rows": before.get("runner_rows"),
                    "distance_source_counts": before.get("distance_source_counts"),
                    "grade_source_counts": before.get("grade_source_counts"),
                },
                "after": {
                    "runner_rows": after.get("runner_rows"),
                    "distance_source_counts": after.get("distance_source_counts"),
                    "grade_source_counts": after.get("grade_source_counts"),
                },
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Target Metadata Capture Status",
        "",
        f"- Safe target distance rows in audited snapshots: `{safe_distance_rows}/{summary['runner_rows']}`.",
        f"- Safe target grade rows in audited snapshots: `{safe_grade_rows}/{summary['runner_rows']}`.",
        "- Existing snapshots are not rewritten; these counts reflect only explicit target metadata already available in each snapshot source CSV/sidecar.",
        "",
        "## Embedded History Richness",
        "",
        "```json",
        json.dumps(
            {
                "before": before.get("embedded_history_richness"),
                "after": richness,
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Vector Similarity And Box Distribution",
        "",
        "```json",
        json.dumps(
            {
                "before": {
                    "near_duplicate_rows_ge80pct_equal_peer": before.get(
                        "near_duplicate_rows_ge80pct_equal_peer"
                    ),
                    "top_pick_box_distribution": before.get("top_pick_box_distribution"),
                },
                "after": {
                    "near_duplicate_rows_ge80pct_equal_peer": after.get(
                        "near_duplicate_rows_ge80pct_equal_peer"
                    ),
                    "near_duplicate_rows_ge90pct_equal_peer": after.get(
                        "near_duplicate_rows_ge90pct_equal_peer"
                    ),
                    "mean_most_similar_non_box_equal_share": after.get(
                        "mean_most_similar_non_box_equal_share"
                    ),
                    "top_pick_box_distribution": after.get("top_pick_box_distribution"),
                },
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Race-Level Findings",
        "",
        md_table(
            race_level_rows(rows),
            [
                "race_id",
                "runner_count",
                "top_pick_box",
                "rows_with_ge80pct_equal_peer",
                "mean_most_similar_equal_share",
                "all_distance_source_default",
                "all_grade_source_default",
            ],
        ),
        "",
        "## Remaining Missing/Default Drivers",
        "",
    ]
    flag_counter = Counter()
    for row in rows:
        for flag in str(row.get("default_missing_feature_flags") or "").split(";"):
            if flag:
                flag_counter[flag] += 1
    content.extend(
        [
            md_table(
                [
                    {"flag": flag, "rows": count, "row_rate": count / max(1, summary["runner_rows"])}
                    for flag, count in flag_counter.most_common()
                ],
                ["flag", "rows", "row_rate"],
            ),
            "",
            "## Rejected Metadata Sources",
            "",
            "```json",
            json.dumps(
                summary.get("rejected_metadata_source_counts") or {},
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Production Prediction Change",
            "",
            "Production predictions changed: `NO_BY_CONSTRUCTION`. This script reads existing snapshots and source CSVs for diagnostics and does not call the prediction writer, retrain, promote, or rewrite any snapshot/model artifact.",
            "",
            "## Recommended Next PR",
            "",
            "Run a report-only challenger retrain using the improved non-box diagnostic surface after a larger frozen, result-free snapshot corpus has safe target distance/grade and richer embedded-history coverage.",
            "",
            "## Safety Confirmation",
            "",
            "- No push.",
            "- No retraining.",
            "- No model promotion or registry change.",
            "- No betting.",
            "- No label overwrite.",
            "- No snapshot rewrite.",
            "- No fake EV.",
            "- No post-result fields used as target metadata.",
        ]
    )
    if summary.get("source_errors"):
        content.extend(["", "## Source Errors", "", "```json", json.dumps(summary["source_errors"], indent=2), "```"])
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, nargs="*", default=None)
    parser.add_argument("--output-dir", "--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-metadata", type=Path, default=None)
    parser.add_argument("--baseline-csv", type=Path, default=None)
    parser.add_argument(
        "--no-reconstruct",
        action="store_true",
        help="Use snapshot prediction fields only; intended for light fixture tests.",
    )
    args = parser.parse_args()

    snapshot_args = args.snapshots or [default_snapshot_dir()]
    snapshot_paths = iter_snapshot_paths(snapshot_args)
    if not snapshot_paths:
        raise SystemExit("DATA_MISSING: no snapshot JSON files found")

    model_columns, metadata = model_feature_contract(
        args.model,
        args.model_metadata,
        no_model_load=args.no_reconstruct,
    )
    if not model_columns:
        model_columns = [
            "venue",
            "grade",
            "distance",
            "field_size",
            "box_number",
            "historical_avg_position",
            "historical_win_rate",
            "historical_place_rate",
            "target_distance",
            *TGR_COLUMNS,
        ]

    rows, errors = build_audit_rows(
        snapshot_paths=snapshot_paths,
        model_columns=model_columns,
        db_path=args.db,
        reconstruct=not args.no_reconstruct,
    )
    if not rows:
        raise SystemExit("DATA_MISSING: no runner rows could be audited")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "live_feature_missingness.csv"
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "report.md"
    missingness_report_path = args.output_dir / "live_feature_missingness_report.md"

    summary = summarize(rows, errors)
    summary["model_id"] = metadata.get("model_id")
    summary["model_feature_count"] = len(model_columns)
    summary["snapshots_audited"] = len(snapshot_paths)

    write_csv(csv_path, rows)
    write_json(summary_path, summary)
    baseline = load_baseline_summary(args.output_dir, args.baseline_csv)
    write_report(report_path, summary, baseline, rows, snapshot_args, csv_path, summary_path)
    write_report(
        missingness_report_path,
        summary,
        baseline,
        rows,
        snapshot_args,
        csv_path,
        summary_path,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
