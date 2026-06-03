#!/usr/bin/env python3
"""Report-only retest for reconstructed pre-race history features.

This helper reads the clean official holdout and the reconstructed history
feature packet, fits only in-memory challenger scorers, and writes diagnostic
artifacts under the requested report directory. It never writes labels,
snapshots, manifests, production model files, model registry entries, odds, EV,
or betting outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accuracy_program.evaluation import (
    blend_probabilities,
    market_implied_probabilities,
    score_predictions,
    validate_feature_columns,
    validate_temporal_holdout,
)


SCHEMA_VERSION = "history_feature_challenger_retest_v1"
DEFAULT_PACKET_DIR = Path(
    "artifacts/full_evidence_orchestration_20260525/"
    "clean_history_feature_packet_20260602"
)
DEFAULT_CLEAN_DATASET = Path(
    "artifacts/full_evidence_orchestration_20260525/"
    "isolated_challenger_box_bias_study_20260602/clean_official_dataset.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/full_evidence_orchestration_20260525/"
    "history_feature_challenger_retest_20260602"
)
PROTECTED_PREFIXES = (
    "artifacts/prediction_snapshots",
    "model_registry",
    "docs/model_registry",
    "ml_models_v4",
    "advanced_models",
)
MIN_ODDS_RACES = 10
HGB_PARAMS = {
    "max_iter": 80,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 10,
    "learning_rate": 0.05,
    "l2_regularization": 0.1,
    "random_state": 42,
}

HISTORY_NUMERIC_FEATURES = (
    "prior_start_count",
    "days_since_last_start",
    "recent_finish_mean_3",
    "recent_finish_mean_5",
    "recent_finish_best_5",
    "recent_win_rate_5",
    "recent_place_rate_5",
    "recent_avg_margin_5",
    "recent_avg_time_5",
    "starts_same_distance",
    "prior_same_distance_start_count",
    "best_time_same_distance",
    "avg_time_same_distance",
    "median_time_same_distance",
    "recent_best_time_same_distance_5",
    "recent_avg_time_same_distance_5",
    "days_since_last_same_distance_start",
    "win_rate_same_distance",
    "place_rate_same_distance",
    "same_distance_venue_start_count",
    "same_distance_venue_best_time",
    "starts_same_venue",
    "win_rate_same_venue",
    "grade_change_indicator",
    "last_start_days",
    "last_start_weight",
    "recent_avg_weight_5",
    "recent_avg_sectional_1st_5",
    "db_prior_start_count",
    "csv_staging_prior_start_count",
    "embedded_form_prior_start_count",
)
GRADE_CONTEXT_NUMERIC_FEATURES = (
    "same_grade_start_count",
    "same_grade_win_rate",
    "same_grade_place_rate",
    "grade_change_indicator",
    "grade_strength_delta",
    "same_distance_same_grade_start_count",
    "same_distance_same_grade_best_time",
    "same_distance_same_grade_avg_time",
)
NO_BOX_CONTEXT_FEATURES = (
    "field_size",
    "target_distance_numeric",
)
REDUCED_BOX_FEATURES = (
    "box_band_inside",
    "box_band_middle",
    "box_band_outside",
)
POWER_ALPHA_GRID = (0.25, 0.35, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return None
        return parsed
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    parsed = _safe_float(value)
    return int(parsed) if parsed is not None else None


def _numeric_distance(value: Any) -> float | None:
    if value in (None, ""):
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit() or ch == ".")
    return _safe_float(digits)


def _box_band(box_number: Any) -> str:
    box = _safe_int(box_number)
    if box in (1, 2):
        return "inside"
    if box in (3, 4, 5, 6):
        return "middle"
    if box is not None and box >= 7:
        return "outside"
    return "unknown"


def _group_key(row: Mapping[str, Any]) -> str:
    return str(row.get("snapshot_instance_id") or row.get("race_id") or "DATA_MISSING")


def _race_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return len({str(row.get("race_id")) for row in rows if row.get("race_id")})


def _group_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(dict(row))
    return dict(grouped)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def assert_output_dir_safe(output_dir: Path, repo_root: Path = ROOT) -> Path:
    resolved = output_dir.resolve()
    root = repo_root.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    relative_text = relative.as_posix()
    for prefix in PROTECTED_PREFIXES:
        if relative_text == prefix or relative_text.startswith(prefix + "/"):
            raise ValueError(f"output_dir_protected:{prefix}")
    required_parent = "artifacts/full_evidence_orchestration_20260525"
    if not relative_text.startswith(required_parent + "/"):
        raise ValueError(f"output_dir_must_be_under:{required_parent}")
    return resolved


def _join_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    dog = str(row.get("dog_name") or row.get("normalized_dog_name") or "").strip().lower()
    return (
        str(row.get("snapshot_instance_id") or ""),
        dog,
        str(row.get("box_number") or ""),
    )


def _prepare_rows(
    clean_rows: Sequence[Mapping[str, Any]],
    packet_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clean_by_key = {_join_key(row): dict(row) for row in clean_rows}
    joined: list[dict[str, Any]] = []
    missing_clean_keys: list[tuple[str, str, str]] = []

    for packet in packet_rows:
        clean = clean_by_key.get(_join_key(packet))
        if clean is None:
            missing_clean_keys.append(_join_key(packet))
            continue
        row = dict(clean)
        row.update(packet)
        row["actual_win"] = int(clean.get("actual_win") or 0)
        row["finish_position"] = clean.get("finish_position")
        row["label_quality"] = clean.get("label_quality")
        row["result_detail_quality"] = clean.get("result_detail_quality")
        row["odds_win"] = clean.get("odds_win")
        row["data_quality_flags"] = clean.get("data_quality_flags")
        row["target_distance"] = clean.get("target_distance") or clean.get("distance")
        row["target_grade"] = clean.get("target_grade")
        joined.append(row)

    field_sizes = {key: len(value) for key, value in _group_rows(joined).items()}
    for row in joined:
        field_size = field_sizes[_group_key(row)]
        distance = (
            row.get("target_distance_safe")
            or row.get("target_distance")
            or row.get("distance")
        )
        band = _box_band(row.get("box_number"))
        row["field_size"] = field_size
        row["target_distance_numeric"] = _numeric_distance(distance)
        row["box_band_inside"] = 1.0 if band == "inside" else 0.0
        row["box_band_middle"] = 1.0 if band == "middle" else 0.0
        row["box_band_outside"] = 1.0 if band == "outside" else 0.0

    allowed_label_values = {"official_or_complete_result"}
    allowed_detail_values = {"finish_position"}
    excluded_label_rows = [
        row
        for row in joined
        if str(row.get("label_quality")) not in allowed_label_values
        or str(row.get("result_detail_quality")) not in allowed_detail_values
    ]

    audit = {
        "clean_rows": len(clean_rows),
        "packet_rows": len(packet_rows),
        "joined_rows": len(joined),
        "missing_clean_key_count": len(missing_clean_keys),
        "missing_clean_key_examples": missing_clean_keys[:10],
        "excluded_label_rows": len(excluded_label_rows),
        "join_status": (
            "PASS"
            if len(joined) == len(packet_rows) == len(clean_rows)
            and not missing_clean_keys
            and not excluded_label_rows
            else "FAIL"
        ),
    }
    return joined, audit


def _select_features(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[str],
    *,
    min_present: int = 10,
) -> tuple[list[str], dict[str, Any]]:
    selected: list[str] = []
    details: dict[str, Any] = {}
    for column in candidates:
        values = [_safe_float(row.get(column)) for row in rows]
        present = [value for value in values if value is not None]
        unique = {round(value, 12) for value in present}
        if len(present) < min_present:
            status = "EXCLUDED_TRAIN_COVERAGE_TOO_LOW"
        elif len(unique) < 2:
            status = "EXCLUDED_TRAIN_ZERO_VARIANCE"
        else:
            status = "SELECTED"
            selected.append(column)
        details[column] = {
            "status": status,
            "train_present_rows": len(present),
            "train_present_pct": len(present) / len(rows) if rows else None,
            "train_unique_values": len(unique),
        }
    return selected, details


def _feature_matrix(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> np.ndarray:
    matrix: list[list[float]] = []
    for row in rows:
        matrix.append(
            [
                np.nan if _safe_float(row.get(column)) is None else float(row.get(column))
                for column in columns
            ]
        )
    return np.array(matrix, dtype=float)


def _sample_weights(labels: np.ndarray) -> np.ndarray:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    total = len(labels)
    if positives <= 0 or negatives <= 0:
        return np.ones(total, dtype=float)
    return np.where(
        labels == 1,
        total / (2.0 * positives),
        total / (2.0 * negatives),
    )


def _normalize_by_group(
    rows: Sequence[Mapping[str, Any]],
    raw_scores: Sequence[float],
    *,
    output_key: str,
) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    indexes_by_group: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indexes_by_group[_group_key(row)].append(index)
    for indexes in indexes_by_group.values():
        scores = [max(1e-12, float(raw_scores[index])) for index in indexes]
        total = sum(scores)
        for index, score in zip(indexes, scores):
            output[index][output_key] = score / total if total else 1.0 / len(indexes)
    return output


def fit_hgb_variant(
    *,
    name: str,
    train_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
    candidate_features: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected, details = _select_features(train_rows, candidate_features)
    forbidden = validate_feature_columns(selected)
    if forbidden:
        raise ValueError(f"forbidden_feature_columns:{forbidden}")
    if len(selected) < 2:
        raise ValueError("insufficient_selected_features")
    labels = np.array([int(row.get("actual_win") or 0) for row in train_rows], dtype=int)
    if len(set(labels.tolist())) < 2:
        raise ValueError("train_labels_single_class")

    model = HistGradientBoostingClassifier(**HGB_PARAMS)
    model.fit(
        _feature_matrix(train_rows, selected),
        labels,
        sample_weight=_sample_weights(labels),
    )
    train_raw = model.predict_proba(_feature_matrix(train_rows, selected))[:, 1]
    eval_raw = model.predict_proba(_feature_matrix(eval_rows, selected))[:, 1]
    train_scored = _normalize_by_group(train_rows, train_raw, output_key="study_prob")
    eval_scored = _normalize_by_group(eval_rows, eval_raw, output_key="study_prob")
    training = {
        "variant": name,
        "model_family": "HistGradientBoostingClassifier",
        "model_params": HGB_PARAMS,
        "feature_columns": selected,
        "feature_selection": details,
        "nan_policy": "native_missing_value_handling; missing history remains NaN, not filled with fake values",
        "forbidden_feature_columns": forbidden,
        "model_artifact_written": False,
        "registry_mutation_allowed": False,
        "production_training": False,
    }
    return train_scored, eval_scored, training


def _power_normalize_by_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    input_key: str,
    output_key: str,
) -> list[dict[str, Any]]:
    raw_scores = []
    for row in rows:
        probability = _safe_float(row.get(input_key))
        if probability is None or probability < 0:
            raise ValueError(f"{input_key}_invalid")
        raw_scores.append(max(1e-12, probability) ** float(alpha))
    return _normalize_by_group(rows, raw_scores, output_key=output_key)


def _selection_key(rows: Sequence[Mapping[str, Any]], probability_key: str) -> tuple[float, float, float]:
    metrics = score_predictions(
        rows,
        probability_key=probability_key,
        race_id_key="snapshot_instance_id",
    )
    return (
        _safe_float(metrics.get("log_loss")) or float("inf"),
        _safe_float(metrics.get("brier")) or float("inf"),
        -(_safe_float(metrics.get("top1")) or 0.0),
    )


def tune_power_alpha(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    input_key: str,
) -> dict[str, Any]:
    attempts = []
    best_alpha: float | None = None
    best_key: tuple[float, float, float] | None = None
    for alpha in POWER_ALPHA_GRID:
        candidate = _power_normalize_by_group(
            train_rows,
            alpha=alpha,
            input_key=input_key,
            output_key="calibrated_prob",
        )
        metrics = score_predictions(
            candidate,
            probability_key="calibrated_prob",
            race_id_key="snapshot_instance_id",
        )
        key = _selection_key(candidate, "calibrated_prob")
        attempts.append(
            {
                "alpha": alpha,
                "log_loss": metrics.get("log_loss"),
                "brier": metrics.get("brier"),
                "top1": metrics.get("top1"),
                "top3": metrics.get("top3"),
            }
        )
        if best_key is None or key < best_key:
            best_key = key
            best_alpha = float(alpha)
    if best_alpha is None:
        raise ValueError("alpha_tuning_failed")
    return {
        "selected_alpha": best_alpha,
        "selection_metric": "train_log_loss_then_brier_then_top1",
        "attempts": attempts,
        "ranking_preserving": True,
        "model_artifact_written": False,
        "registry_mutation_allowed": False,
    }


def _clone_probability(
    rows: Sequence[Mapping[str, Any]],
    *,
    input_key: str,
    output_key: str = "study_prob",
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        item = dict(row)
        item[output_key] = _safe_float(row.get(input_key))
        output.append(item)
    return output


def _top_and_winner_boxes(
    rows: Sequence[Mapping[str, Any]],
    *,
    probability_key: str,
) -> dict[str, Any]:
    top_pick_boxes: Counter[str] = Counter()
    winner_boxes: Counter[str] = Counter()
    for race_rows in _group_rows(rows).values():
        ranked = sorted(
            race_rows,
            key=lambda row: _safe_float(row.get(probability_key)) or 0.0,
            reverse=True,
        )
        if not ranked:
            continue
        top_pick_boxes[str(ranked[0].get("box_number") or "DATA_MISSING")] += 1
        for row in ranked:
            if int(row.get("actual_win") or 0) == 1:
                winner_boxes[str(row.get("box_number") or "DATA_MISSING")] += 1
                break
    total = sum(top_pick_boxes.values())
    return {
        "top_pick_box_distribution": dict(sorted(top_pick_boxes.items())),
        "winner_box_distribution": dict(sorted(winner_boxes.items())),
        "box1_top_pick_share": (
            float(top_pick_boxes.get("1", 0) / total) if total else None
        ),
    }


def _per_box_performance(
    rows: Sequence[Mapping[str, Any]],
    *,
    probability_key: str,
) -> dict[str, Any]:
    top_boxes = Counter()
    for race_rows in _group_rows(rows).values():
        ranked = sorted(
            race_rows,
            key=lambda row: _safe_float(row.get(probability_key)) or 0.0,
            reverse=True,
        )
        if ranked:
            top_boxes[str(ranked[0].get("box_number") or "DATA_MISSING")] += 1

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("box_number") or "DATA_MISSING")].append(row)

    out: dict[str, Any] = {}
    race_count = len(_group_rows(rows))
    for box, box_rows in sorted(grouped.items()):
        pairs = [
            (_safe_float(row.get(probability_key)), int(row.get("actual_win") or 0))
            for row in box_rows
            if _safe_float(row.get(probability_key)) is not None
        ]
        out[box] = {
            "runner_rows": len(box_rows),
            "winner_count": sum(int(row.get("actual_win") or 0) for row in box_rows),
            "actual_win_rate": (
                sum(int(row.get("actual_win") or 0) for row in box_rows) / len(box_rows)
                if box_rows
                else None
            ),
            "top_pick_count": top_boxes.get(box, 0),
            "top_pick_share": top_boxes.get(box, 0) / race_count if race_count else None,
            "avg_probability": (
                float(np.mean([p for p, _ in pairs])) if pairs else None
            ),
            "brier": (
                float(np.mean([(p - y) ** 2 for p, y in pairs])) if pairs else None
            ),
        }
    return out


def _complete_odds_groups(rows: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    groups = []
    for race_rows in _group_rows(rows).values():
        if race_rows and all(_safe_float(row.get("odds_win")) is not None for row in race_rows):
            groups.append([dict(row) for row in race_rows])
    return groups


def _ev_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    probability_key: str,
) -> dict[str, Any]:
    complete_groups = _complete_odds_groups(rows)
    eligible_rows = [row for group in complete_groups for row in group]
    if len(complete_groups) < MIN_ODDS_RACES:
        status = "BLOCKED_UNDERPOWERED"
    else:
        status = "RUN"
    positives = []
    all_ev = []
    for row in eligible_rows:
        probability = _safe_float(row.get(probability_key))
        odds = _safe_float(row.get("odds_win"))
        if probability is None or odds is None or odds <= 1.0:
            continue
        ev = probability * odds - 1.0
        actual = int(row.get("actual_win") or 0)
        roi = odds - 1.0 if actual == 1 else -1.0
        all_ev.append({"ev": ev, "actual_win": actual, "roi": roi})
        if ev > 0:
            positives.append({"ev": ev, "actual_win": actual, "roi": roi})
    return {
        "status": status,
        "complete_valid_odds_races": len(complete_groups),
        "eligible_runner_count": len(eligible_rows),
        "minimum_races_for_claim": MIN_ODDS_RACES,
        "positive_ev_count": len(positives),
        "average_ev": float(np.mean([row["ev"] for row in all_ev])) if all_ev else None,
        "positive_ev_hit_rate": (
            float(np.mean([row["actual_win"] for row in positives])) if positives else None
        ),
        "positive_ev_realized_roi": (
            float(np.mean([row["roi"] for row in positives])) if positives else None
        ),
        "note": "report-only only; underpowered odds samples are not EV-edge evidence",
    }


def _arm_result(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    probability_key: str,
    scope: str,
    training: Mapping[str, Any] | None = None,
    status: str = "RUN",
) -> dict[str, Any]:
    metrics = score_predictions(
        rows,
        probability_key=probability_key,
        race_id_key="snapshot_instance_id",
    )
    return {
        "status": status,
        "variant": name,
        "scope": scope,
        "race_count": _race_count(rows),
        "snapshot_instance_count": len(_group_rows(rows)),
        "runner_row_count": len(rows),
        "metrics": metrics,
        "box_bias": _top_and_winner_boxes(rows, probability_key=probability_key),
        "per_box_performance": _per_box_performance(rows, probability_key=probability_key),
        "ev_report_only": _ev_diagnostics(rows, probability_key=probability_key),
        "training": dict(training or {}),
    }


def _blocked_variant(name: str, status: str, reason: str) -> dict[str, Any]:
    return {
        "status": status,
        "variant": name,
        "blocker": reason,
        "promotion_allowed": False,
        "registry_mutation_allowed": False,
        "model_artifact_written": False,
    }


def _market_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for race_rows in _complete_odds_groups(rows):
        odds = {
            str(row.get("dog_name")): float(row.get("odds_win"))
            for row in race_rows
            if _safe_float(row.get("odds_win")) is not None
        }
        market = market_implied_probabilities(odds)
        for row in race_rows:
            name = str(row.get("dog_name"))
            if name in market:
                item = dict(row)
                item["study_prob"] = market[name]
                output.append(item)
    return output


def _blend_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_weight: float,
    model_probability_key: str,
) -> list[dict[str, Any]]:
    output = []
    for race_rows in _complete_odds_groups(rows):
        odds = {
            str(row.get("dog_name")): float(row.get("odds_win"))
            for row in race_rows
            if _safe_float(row.get("odds_win")) is not None
        }
        market = market_implied_probabilities(odds)
        model = {
            str(row.get("dog_name")): float(row.get(model_probability_key))
            for row in race_rows
            if _safe_float(row.get(model_probability_key)) is not None
        }
        probabilities = blend_probabilities(model, market, model_weight=model_weight)
        for row in race_rows:
            name = str(row.get("dog_name"))
            if name in probabilities:
                item = dict(row)
                item["study_prob"] = probabilities[name]
                output.append(item)
    return output


def _bootstrap_delta(
    baseline_rows: Sequence[Mapping[str, Any]],
    variant_rows: Sequence[Mapping[str, Any]],
    *,
    probability_key: str,
    samples: int = 500,
) -> dict[str, Any]:
    baseline_groups = _group_rows(baseline_rows)
    variant_groups = _group_rows(variant_rows)
    keys = sorted(set(baseline_groups) & set(variant_groups))
    if len(keys) < 2:
        return {"status": "DATA_MISSING", "reason": "too_few_paired_races"}

    rng = np.random.default_rng(42)
    deltas: dict[str, list[float]] = defaultdict(list)
    for draw_index in range(samples):
        selected = rng.choice(keys, size=len(keys), replace=True)
        baseline_sample: list[dict[str, Any]] = []
        variant_sample: list[dict[str, Any]] = []
        for sample_index, key in enumerate(selected):
            bootstrap_key = f"{key}::bootstrap::{draw_index}:{sample_index}"
            for row in baseline_groups[str(key)]:
                item = dict(row)
                item["bootstrap_group"] = bootstrap_key
                baseline_sample.append(item)
            for row in variant_groups[str(key)]:
                item = dict(row)
                item["bootstrap_group"] = bootstrap_key
                variant_sample.append(item)
        base_metrics = score_predictions(
            baseline_sample,
            probability_key="study_prob",
            race_id_key="bootstrap_group",
        )
        variant_metrics = score_predictions(
            variant_sample,
            probability_key=probability_key,
            race_id_key="bootstrap_group",
        )
        for metric in ("top1", "top2", "top3", "mean_winner_rank", "brier", "log_loss"):
            base_value = _safe_float(base_metrics.get(metric))
            variant_value = _safe_float(variant_metrics.get(metric))
            if base_value is not None and variant_value is not None:
                deltas[metric].append(variant_value - base_value)

    summary: dict[str, Any] = {
        "status": "RUN",
        "method": f"paired_race_bootstrap_{samples}_seed_42",
        "paired_snapshot_instances": len(keys),
        "delta_direction": (
            "positive improves top1/top2/top3; negative improves "
            "mean_winner_rank/brier/log_loss"
        ),
    }
    for metric, values in deltas.items():
        summary[metric] = {
            "mean_delta": float(np.mean(values)),
            "lower_95": float(np.percentile(values, 2.5)),
            "upper_95": float(np.percentile(values, 97.5)),
        }
    return summary


def _feature_coverage(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for column in columns:
        present = sum(_safe_float(row.get(column)) is not None for row in rows)
        out[column] = {
            "present_rows": present,
            "present_pct": present / len(rows) if rows else None,
        }
    return out


def _git_output(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception as exc:
        return f"DATA_MISSING:{type(exc).__name__}:{exc}"


def _write_comparison_tsv(path: Path, variants: Mapping[str, Any]) -> None:
    columns = [
        "variant",
        "status",
        "scope",
        "races",
        "runner_rows",
        "top1",
        "top2",
        "top3",
        "mean_winner_rank",
        "brier",
        "log_loss",
        "calibration_slope",
        "calibration_intercept",
        "prob_sum_max_abs_error",
        "box1_top_pick_share",
        "blocker",
    ]
    lines = ["\t".join(columns)]
    for name, result in variants.items():
        metrics = result.get("metrics") or {}
        calibration = metrics.get("calibration") or {}
        probability_sum = metrics.get("probability_sum") or {}
        box_bias = result.get("box_bias") or {}
        row = {
            "variant": name,
            "status": result.get("status"),
            "scope": result.get("scope", "DATA_MISSING"),
            "races": metrics.get("races_evaluated", "DATA_MISSING"),
            "runner_rows": metrics.get("dog_predictions_evaluated", "DATA_MISSING"),
            "top1": metrics.get("top1", "DATA_MISSING"),
            "top2": metrics.get("top2", "DATA_MISSING"),
            "top3": metrics.get("top3", "DATA_MISSING"),
            "mean_winner_rank": metrics.get("mean_winner_rank", "DATA_MISSING"),
            "brier": metrics.get("brier", "DATA_MISSING"),
            "log_loss": metrics.get("log_loss", "DATA_MISSING"),
            "calibration_slope": calibration.get("slope", "DATA_MISSING"),
            "calibration_intercept": calibration.get("intercept", "DATA_MISSING"),
            "prob_sum_max_abs_error": probability_sum.get("max_abs_error", "DATA_MISSING"),
            "box1_top_pick_share": box_bias.get("box1_top_pick_share", "DATA_MISSING"),
            "blocker": result.get("blocker", ""),
        }
        lines.append("\t".join(str(row[column]) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    parsed = _safe_float(value)
    return "DATA_MISSING" if parsed is None else f"{parsed:.4f}"


def _md_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    output = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(output)


def _recommendation(
    *,
    join_audit: Mapping[str, Any],
    leakage_status: str,
    champion: Mapping[str, Any],
    variants: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> str:
    if join_audit.get("join_status") != "PASS" or leakage_status != "PASS":
        return "FEATURE_JOIN_BLOCKED_NEEDS_DATA_REPAIR"

    champion_metrics = champion.get("metrics") or {}
    champion_box1 = _safe_float((champion.get("box_bias") or {}).get("box1_top_pick_share"))
    champion_top3 = _safe_float(champion_metrics.get("top3"))
    champion_rank = _safe_float(champion_metrics.get("mean_winner_rank"))
    champion_brier = _safe_float(champion_metrics.get("brier"))

    for name in (
        "history_only_hgb",
        "grade_context_hgb",
        "no_box_history_hgb",
        "reduced_box_band_history_hgb",
        "reduced_box_band_grade_context_hgb",
        "calibrated_history_only_hgb",
        "calibrated_grade_context_hgb",
        "calibrated_no_box_history_hgb",
        "calibrated_reduced_box_band_history_hgb",
        "calibrated_reduced_box_band_grade_context_hgb",
    ):
        result = variants.get(name) or {}
        if result.get("status") != "RUN":
            continue
        metrics = result.get("metrics") or {}
        box1 = _safe_float((result.get("box_bias") or {}).get("box1_top_pick_share"))
        top3 = _safe_float(metrics.get("top3"))
        rank = _safe_float(metrics.get("mean_winner_rank"))
        brier = _safe_float(metrics.get("brier"))
        if box1 is None or champion_box1 is None:
            continue
        box_reduced = box1 <= min(0.5, champion_box1 - 0.2)
        ranking_not_worse = (
            top3 is not None
            and champion_top3 is not None
            and top3 >= champion_top3
            and rank is not None
            and champion_rank is not None
            and rank <= champion_rank
        )
        calibration_not_worse = (
            brier is not None and champion_brier is not None and brier <= champion_brier
        )
        delta = bootstrap.get(name) or {}
        top3_ci = ((delta.get("top3") or {}).get("lower_95"), (delta.get("top3") or {}).get("upper_95"))
        rank_ci = (
            (delta.get("mean_winner_rank") or {}).get("lower_95"),
            (delta.get("mean_winner_rank") or {}).get("upper_95"),
        )
        meaningful = (
            top3_ci[0] is not None
            and top3_ci[1] is not None
            and rank_ci[0] is not None
            and rank_ci[1] is not None
            and (top3_ci[0] > 0 or rank_ci[1] < 0)
        )
        if box_reduced and ranking_not_worse and calibration_not_worse:
            return (
                "HISTORY_FEATURES_PROMISING_NEEDS_CONTROLLED_TRAINING_STUDY"
                if meaningful
                else "NO_PROMOTION_MORE_DATA_NEEDED"
            )
    return "HISTORY_FEATURES_DO_NOT_FIX_BOX_BIAS"


def _write_report(
    path: Path,
    *,
    recommendation: str,
    data_inventory: Mapping[str, Any],
    packet_provenance: Mapping[str, Any],
    leakage: Mapping[str, Any],
    variants: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    endpoint_health: Mapping[str, Any],
    sqlite_quick_check: str,
    active_processes: str,
    commands_run: Sequence[str],
) -> None:
    comparison_rows = []
    for name, result in variants.items():
        metrics = result.get("metrics") or {}
        calibration = metrics.get("calibration") or {}
        box_bias = result.get("box_bias") or {}
        comparison_rows.append(
            {
                "variant": name,
                "status": result.get("status"),
                "races": metrics.get("races_evaluated", "DATA_MISSING"),
                "top1": _fmt(metrics.get("top1")),
                "top2": _fmt(metrics.get("top2")),
                "top3": _fmt(metrics.get("top3")),
                "mean_rank": _fmt(metrics.get("mean_winner_rank")),
                "brier": _fmt(metrics.get("brier")),
                "log_loss": _fmt(metrics.get("log_loss")),
                "slope": _fmt(calibration.get("slope")),
                "box1_share": _fmt(box_bias.get("box1_top_pick_share")),
                "blocker": result.get("blocker", ""),
            }
        )

    champion = variants.get("champion_current_baseline", {})
    champion_box = champion.get("box_bias") or {}
    content = [
        "# History Feature Challenger Retest",
        "",
        "## Executive Summary",
        "",
        f"Final recommendation: `{recommendation}`.",
        "",
        "This was a report-only retest over the reused clean official holdout and reconstructed pre-race history packet. No production model was trained, promoted, registered, or used for betting.",
        "",
        "## Data Used",
        "",
        f"- Clean races: `{data_inventory.get('clean_races')}`",
        f"- Clean snapshot instances: `{data_inventory.get('clean_snapshot_instances')}`",
        f"- Clean runner rows: `{data_inventory.get('clean_runner_rows')}`",
        f"- Primary split: `{data_inventory.get('primary_split')}`",
        f"- Train rows/instances: `{data_inventory.get('train_rows')}` / `{data_inventory.get('train_snapshot_instances')}`",
        f"- Eval rows/instances: `{data_inventory.get('eval_rows')}` / `{data_inventory.get('eval_snapshot_instances')}`",
        f"- Exclusions: `{data_inventory.get('exclusions')}`",
        "",
        "## Feature Packet Provenance",
        "",
        f"- Packet report: `{packet_provenance.get('packet_report')}`",
        f"- Packet CSV: `{packet_provenance.get('packet_csv')}`",
        f"- Feature coverage summary: `{packet_provenance.get('coverage_summary')}`",
        f"- Join audit: `{data_inventory.get('join_audit')}`",
        f"- Selected-feature policy: train features with low coverage or zero variance are excluded and recorded; missing feature values remain `NaN` for native missing-value handling, not fake defaults.",
        "",
        "## Leakage Audit",
        "",
        f"- Status: `{leakage.get('status')}`",
        f"- Packet leakage status: `{leakage.get('packet_leakage_status')}`",
        f"- Temporal holdout: `{leakage.get('temporal_holdout')}`",
        f"- Forbidden feature columns by variant: `{leakage.get('forbidden_feature_columns_by_variant')}`",
        "",
        "## Champion Baseline",
        "",
        f"- Rolling eval top-pick boxes: `{champion_box.get('top_pick_box_distribution')}`",
        f"- Rolling eval winner boxes: `{champion_box.get('winner_box_distribution')}`",
        f"- Rolling eval box-1 top-pick share: `{_fmt(champion_box.get('box1_top_pick_share'))}`",
        "",
        "## Challenger Comparison Table",
        "",
        _md_table(
            comparison_rows,
            [
                "variant",
                "status",
                "races",
                "top1",
                "top2",
                "top3",
                "mean_rank",
                "brier",
                "log_loss",
                "slope",
                "box1_share",
                "blocker",
            ],
        ),
        "",
        "## Box-Bias Diagnostics",
        "",
        "- Box-bias production gate remains red and was not weakened.",
        "- Full per-variant top-pick, winner-box, and per-box diagnostics are in `box_bias_diagnostics.json`.",
        "",
        "## Calibration Diagnostics",
        "",
        "- Calibration slope/intercept, Brier, log loss, and reliability bins are in `challenger_metrics.json` and `calibration_diagnostics.json`.",
        "- Paired race-bootstrap deltas are in `statistical_significance.json`.",
        f"- Bootstrap summary: `{bootstrap}`",
        "",
        "## EV Report-Only Diagnostics",
        "",
        "- Odds-derived variants are marked `BLOCKED_UNDERPOWERED` unless at least 10 complete valid-odds races are available.",
        "- EV diagnostics are report-only only and are not evidence of an EV edge.",
        "- Details are in `ev_diagnostics_report_only.json`.",
        "",
        "## Endpoint And SQLite Health",
        "",
        f"- Endpoint health: `{endpoint_health}`",
        f"- SQLite quick_check: `{sqlite_quick_check}`",
        f"- Active capture/ingest/promotion/model-registry processes: `{active_processes or 'none found'}`",
        "",
        "## Commands Run",
        "",
        *[f"- `{command}`" for command in commands_run],
        "",
        "## Changed Files",
        "",
        "- `scripts/run_history_feature_challenger_retest.py`",
        "- `tests/test_run_history_feature_challenger_retest.py`",
        "- `artifacts/full_evidence_orchestration_20260525/history_feature_challenger_retest_20260602/`",
        "",
        "## No-Mutation Confirmation",
        "",
        "- No production writes, live result-ingest writes, result label writes, snapshot writes or rewrites, manifest append, model registry mutation, production retrain, production model file changes, model promotion, betting, fake odds, fake EV, mock racing data, `--persist`, `--capture-live-odds`, `--allow-unverified-runner-set`, or `APPROVE_RESULT_LABEL_WRITE` were used.",
        "- The known box-bias regression gate remains intact and red.",
        "",
        "## Recommendation",
        "",
        f"`{recommendation}`.",
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def run_retest(
    *,
    clean_dataset: Path,
    packet_dir: Path,
    output_dir: Path,
    sqlite_quick_check: str,
    endpoint_health: Mapping[str, Any],
    active_processes: str,
    commands_run: Sequence[str],
) -> dict[str, Any]:
    output_dir = assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    packet_csv = packet_dir / "pre_race_history_feature_packet.csv"
    clean_rows = _load_jsonl(clean_dataset)
    packet_rows = _load_csv(packet_csv)
    joined_rows, join_audit = _prepare_rows(clean_rows, packet_rows)
    train_rows = [row for row in joined_rows if row.get("packet") == "historical"]
    eval_rows = [row for row in joined_rows if row.get("packet") == "rolling"]
    temporal = validate_temporal_holdout(train_rows, eval_rows)
    packet_leakage = json.loads((packet_dir / "feature_leakage_audit.json").read_text())
    packet_coverage = json.loads(
        (packet_dir / "feature_coverage_after_reconstruction.json").read_text()
    )

    variants: dict[str, Any] = {}
    baseline_eval = _clone_probability(eval_rows, input_key="win_prob_norm")
    row_outputs: dict[str, list[dict[str, Any]]] = {
        "champion_current_baseline": baseline_eval,
    }
    variants["champion_current_baseline"] = _arm_result(
        "champion_current_baseline",
        baseline_eval,
        probability_key="study_prob",
        scope="rolling_clean_official_eval",
    )

    feature_sets = {
        "history_only_hgb": HISTORY_NUMERIC_FEATURES,
        "grade_context_hgb": (*HISTORY_NUMERIC_FEATURES, *GRADE_CONTEXT_NUMERIC_FEATURES),
        "no_box_history_hgb": (*HISTORY_NUMERIC_FEATURES, *NO_BOX_CONTEXT_FEATURES),
        "reduced_box_band_history_hgb": (
            *HISTORY_NUMERIC_FEATURES,
            *NO_BOX_CONTEXT_FEATURES,
            *REDUCED_BOX_FEATURES,
        ),
        "reduced_box_band_grade_context_hgb": (
            *HISTORY_NUMERIC_FEATURES,
            *GRADE_CONTEXT_NUMERIC_FEATURES,
            *NO_BOX_CONTEXT_FEATURES,
            *REDUCED_BOX_FEATURES,
        ),
    }
    train_scored_by_variant: dict[str, list[dict[str, Any]]] = {}
    eval_scored_by_variant: dict[str, list[dict[str, Any]]] = {}
    for name, candidates in feature_sets.items():
        try:
            train_scored, eval_scored, training = fit_hgb_variant(
                name=name,
                train_rows=train_rows,
                eval_rows=eval_rows,
                candidate_features=candidates,
            )
            train_scored_by_variant[name] = train_scored
            eval_scored_by_variant[name] = eval_scored
            row_outputs[name] = eval_scored
            variants[name] = _arm_result(
                name,
                eval_scored,
                probability_key="study_prob",
                scope="historical_train_to_rolling_eval",
                training=training,
            )
        except Exception as exc:
            variants[name] = _blocked_variant(
                name,
                "NOT_RUN",
                f"history variant failed:{type(exc).__name__}:{exc}",
            )

    try:
        alpha_info = tune_power_alpha(train_rows, input_key="win_prob_norm")
        calibrated = _power_normalize_by_group(
            eval_rows,
            alpha=float(alpha_info["selected_alpha"]),
            input_key="win_prob_norm",
            output_key="study_prob",
        )
        row_outputs["calibrated_champion_power"] = calibrated
        variants["calibrated_champion_power"] = _arm_result(
            "calibrated_champion_power",
            calibrated,
            probability_key="study_prob",
            scope="historical_train_to_rolling_eval",
            training=alpha_info,
        )
    except Exception as exc:
        variants["calibrated_champion_power"] = _blocked_variant(
            "calibrated_champion_power",
            "NOT_RUN",
            f"champion calibration failed:{type(exc).__name__}:{exc}",
        )

    for base_name in (
        "history_only_hgb",
        "grade_context_hgb",
        "no_box_history_hgb",
        "reduced_box_band_history_hgb",
        "reduced_box_band_grade_context_hgb",
    ):
        calibrated_name = f"calibrated_{base_name}"
        if variants.get(base_name, {}).get("status") != "RUN":
            variants[calibrated_name] = _blocked_variant(
                calibrated_name,
                "NOT_RUN",
                f"base variant not runnable:{base_name}",
            )
            continue
        try:
            alpha_info = tune_power_alpha(
                train_scored_by_variant[base_name],
                input_key="study_prob",
            )
            eval_base_rows = eval_scored_by_variant.get(base_name)
            if not eval_base_rows:
                raise ValueError("base_eval_rows_missing")
            calibrated = _power_normalize_by_group(
                eval_base_rows,
                alpha=float(alpha_info["selected_alpha"]),
                input_key="study_prob",
                output_key="study_prob",
            )
            row_outputs[calibrated_name] = calibrated
        except Exception:
            base_rows = None
        else:
            base_rows = calibrated
        if base_rows is None:
            variants[calibrated_name] = _blocked_variant(
                calibrated_name,
                "NOT_RUN",
                "history calibration failed:base eval rows missing or alpha tuning failed",
            )
        else:
            variants[calibrated_name] = _arm_result(
                calibrated_name,
                base_rows,
                probability_key="study_prob",
                scope="historical_train_to_rolling_eval",
                training=alpha_info,
            )

    market_rows = _market_rows(eval_rows)
    if len(_complete_odds_groups(eval_rows)) < MIN_ODDS_RACES:
        variants["market_implied"] = _blocked_variant(
            "market_implied",
            "BLOCKED_UNDERPOWERED",
            f"complete valid-odds eval races={len(_complete_odds_groups(eval_rows))}; minimum={MIN_ODDS_RACES}",
        )
        if market_rows:
            variants["market_implied"]["diagnostic_sample"] = _arm_result(
                "market_implied",
                market_rows,
                probability_key="study_prob",
                scope="complete_valid_odds_sample_only",
                status="BLOCKED_UNDERPOWERED",
            )
    else:
        row_outputs["market_implied"] = market_rows
        variants["market_implied"] = _arm_result(
            "market_implied",
            market_rows,
            probability_key="study_prob",
            scope="complete_valid_odds_eval",
        )

    blend_rows = _blend_rows(
        eval_rows,
        model_weight=0.5,
        model_probability_key="win_prob_norm",
    )
    if len(_complete_odds_groups(eval_rows)) < MIN_ODDS_RACES:
        variants["simple_blend_50"] = _blocked_variant(
            "simple_blend_50",
            "BLOCKED_UNDERPOWERED",
            f"complete valid-odds eval races={len(_complete_odds_groups(eval_rows))}; minimum={MIN_ODDS_RACES}",
        )
        if blend_rows:
            variants["simple_blend_50"]["diagnostic_sample"] = _arm_result(
                "simple_blend_50",
                blend_rows,
                probability_key="study_prob",
                scope="complete_valid_odds_sample_only",
                status="BLOCKED_UNDERPOWERED",
                training={"model_weight": 0.5},
            )
    else:
        row_outputs["simple_blend_50"] = blend_rows
        variants["simple_blend_50"] = _arm_result(
            "simple_blend_50",
            blend_rows,
            probability_key="study_prob",
            scope="complete_valid_odds_eval",
            training={"model_weight": 0.5},
        )
    variants["learned_blend"] = _blocked_variant(
        "learned_blend",
        "BLOCKED_UNDERPOWERED",
        f"complete valid-odds train races={len(_complete_odds_groups(train_rows))}; minimum={MIN_ODDS_RACES}",
    )
    variants["calibrated_blend"] = _blocked_variant(
        "calibrated_blend",
        "BLOCKED_UNDERPOWERED",
        f"complete valid-odds train races={len(_complete_odds_groups(train_rows))}; minimum={MIN_ODDS_RACES}",
    )

    bootstrap = {}
    for name, rows_for_bootstrap in row_outputs.items():
        if name == "champion_current_baseline":
            bootstrap[name] = _bootstrap_delta(
                baseline_eval,
                rows_for_bootstrap,
                probability_key="study_prob",
            )
            continue
        bootstrap[name] = _bootstrap_delta(
            baseline_eval,
            rows_for_bootstrap,
            probability_key="study_prob",
        )

    leakage = {
        "schema_version": "history_feature_challenger_leakage_audit_v1",
        "status": "PASS"
        if temporal.ok
        and packet_leakage.get("status") == "PASS"
        and join_audit.get("join_status") == "PASS"
        else "FAIL",
        "packet_leakage_status": packet_leakage.get("status"),
        "temporal_holdout": {
            "ok": temporal.ok,
            "train_max_date": temporal.train_max_date,
            "test_min_date": temporal.test_min_date,
            "race_id_overlap": temporal.race_id_overlap,
            "violations": temporal.violations,
        },
        "forbidden_feature_columns_by_variant": {
            name: (result.get("training") or {}).get("forbidden_feature_columns")
            for name, result in variants.items()
            if result.get("status") == "RUN"
        },
        "target_outcome_columns_used_as_features": [],
        "production_training": False,
        "model_artifact_written": False,
        "registry_mutation": False,
    }

    data_inventory = {
        "schema_version": "history_feature_challenger_data_inventory_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_head": _git_output(["rev-parse", "--short=12", "HEAD"]),
        "git_branch": _git_output(["branch", "--show-current"]),
        "clean_dataset": str(clean_dataset),
        "packet_csv": str(packet_csv),
        "clean_races": _race_count(joined_rows),
        "clean_snapshot_instances": len(_group_rows(joined_rows)),
        "clean_runner_rows": len(joined_rows),
        "primary_split": "historical_packet_train_to_rolling_packet_eval",
        "train_rows": len(train_rows),
        "train_snapshot_instances": len(_group_rows(train_rows)),
        "eval_rows": len(eval_rows),
        "eval_snapshot_instances": len(_group_rows(eval_rows)),
        "exclusions": {
            "row_exclusion_count": 0,
            "reason": "none; exact clean packet join used",
        },
        "join_audit": join_audit,
        "complete_valid_odds_eval_races": len(_complete_odds_groups(eval_rows)),
        "complete_valid_odds_train_races": len(_complete_odds_groups(train_rows)),
        "feature_coverage_recomputed": {
            "train": _feature_coverage(train_rows, HISTORY_NUMERIC_FEATURES),
            "eval": _feature_coverage(eval_rows, HISTORY_NUMERIC_FEATURES),
            "grade_context_train": _feature_coverage(train_rows, GRADE_CONTEXT_NUMERIC_FEATURES),
            "grade_context_eval": _feature_coverage(eval_rows, GRADE_CONTEXT_NUMERIC_FEATURES),
        },
    }
    packet_provenance = {
        "packet_report": str(packet_dir / "report.md"),
        "packet_csv": str(packet_csv),
        "coverage_summary": packet_coverage.get("summary"),
        "leakage_audit": str(packet_dir / "feature_leakage_audit.json"),
        "dictionary": str(packet_dir / "pre_race_history_feature_dictionary.json"),
    }

    recommendation = _recommendation(
        join_audit=join_audit,
        leakage_status=str(leakage.get("status")),
        champion=variants["champion_current_baseline"],
        variants=variants,
        bootstrap=bootstrap,
    )

    write_json(output_dir / "data_inventory.json", data_inventory)
    write_json(output_dir / "packet_provenance.json", packet_provenance)
    write_json(output_dir / "leakage_audit.json", leakage)
    write_json(output_dir / "challenger_metrics.json", variants)
    write_json(
        output_dir / "box_bias_diagnostics.json",
        {
            name: result.get("box_bias")
            for name, result in variants.items()
            if isinstance(result, Mapping)
        },
    )
    write_json(
        output_dir / "calibration_diagnostics.json",
        {
            name: (result.get("metrics") or {}).get("calibration")
            for name, result in variants.items()
            if result.get("status") == "RUN"
        },
    )
    write_json(
        output_dir / "ev_diagnostics_report_only.json",
        {
            name: result.get("ev_report_only") or result.get("diagnostic_sample", {}).get("ev_report_only")
            for name, result in variants.items()
        },
    )
    write_json(output_dir / "statistical_significance.json", bootstrap)
    _write_comparison_tsv(output_dir / "comparison_table.tsv", variants)
    _write_report(
        output_dir / "report.md",
        recommendation=recommendation,
        data_inventory=data_inventory,
        packet_provenance=packet_provenance,
        leakage=leakage,
        variants=variants,
        bootstrap=bootstrap,
        endpoint_health=endpoint_health,
        sqlite_quick_check=sqlite_quick_check,
        active_processes=active_processes,
        commands_run=commands_run,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "SUCCESS" if recommendation != "FEATURE_JOIN_BLOCKED_NEEDS_DATA_REPAIR" else "BLOCKED",
        "recommendation": recommendation,
        "output_dir": str(output_dir),
        "clean_runner_rows": len(joined_rows),
        "eval_snapshot_instances": len(_group_rows(eval_rows)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dataset", default=str(DEFAULT_CLEAN_DATASET))
    parser.add_argument("--packet-dir", default=str(DEFAULT_PACKET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sqlite-quick-check", default="DATA_MISSING")
    parser.add_argument("--endpoint-health", default="{}")
    parser.add_argument("--active-processes", default="")
    parser.add_argument("--command", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    if os.environ.get("APPROVE_RESULT_LABEL_WRITE"):
        raise SystemExit("refusing_to_run_with_APPROVE_RESULT_LABEL_WRITE_set")
    args = build_parser().parse_args(argv)
    endpoint_health = json.loads(args.endpoint_health)
    result = run_retest(
        clean_dataset=Path(args.clean_dataset),
        packet_dir=Path(args.packet_dir),
        output_dir=Path(args.output_dir),
        sqlite_quick_check=str(args.sqlite_quick_check),
        endpoint_health=endpoint_health,
        active_processes=str(args.active_processes),
        commands_run=list(args.command),
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0 if result.get("status") == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
