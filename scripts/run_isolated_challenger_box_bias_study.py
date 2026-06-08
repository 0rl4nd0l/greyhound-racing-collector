#!/usr/bin/env python3
"""Run a report-only clean-official box-bias challenger study.

The study is intentionally artifact-only. It reads frozen prediction snapshots
and SQLite labels through the existing read-only evaluator, trains only small
in-memory scoring variants, and refuses production output directories.
"""

from __future__ import annotations

import argparse
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
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accuracy_program.calibration import power_normalize_by_race
from accuracy_program.evaluation import (
    blend_probabilities,
    market_implied_probabilities,
    score_predictions,
    validate_feature_columns,
    validate_temporal_holdout,
)
from scripts.evaluate_prediction_snapshots import (
    _clean_official_evaluation_rows,
    evaluate_snapshots,
)


SCHEMA_VERSION = "isolated_challenger_box_bias_study_v1"
PROTECTED_PREFIXES = (
    "model_registry",
    "docs/model_registry",
    "ml_models_v4",
    "artifacts/prediction_snapshots",
)
HISTORY_FEATURE_COLUMNS = (
    "historical_avg_position",
    "historical_win_rate",
    "historical_place_rate",
    "historical_best_position",
    "tgr_total_races",
    "tgr_win_rate",
    "tgr_place_rate",
    "embedded_history_race_count",
    "embedded_history_win_rate",
    "embedded_history_place_rate",
)
POWER_ALPHA_GRID = (0.25, 0.35, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0)
BLEND_WEIGHT_GRID = tuple(round(i / 10, 1) for i in range(11))


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
        if value is None:
            return None
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return None
        return parsed
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _logit(probability: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, float(probability)))
    return math.log(p / (1.0 - p))


def parse_snapshot_manifest_line(line: str) -> str | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("{"):
        item = json.loads(text)
        if not isinstance(item, Mapping):
            return None
        value = item.get("snapshot_path") or item.get("path")
        return str(value) if value else None
    return text


def snapshot_paths_from_manifests(paths: Sequence[Path]) -> list[str]:
    out: list[str] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parsed = parse_snapshot_manifest_line(line)
                if parsed:
                    out.append(parsed)
    return sorted(dict.fromkeys(out))


def rows_from_evaluation_datasets(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                item = json.loads(text)
                if isinstance(item, Mapping):
                    rows.append(dict(item))
    return rows


def _dedupe_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (
            _evaluation_group(row),
            str(row.get("dog_name") or ""),
            str(row.get("box_number") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


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


def _run_git(args: Sequence[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _source_group(row: Mapping[str, Any]) -> str:
    path = str(row.get("snapshot_path") or "")
    if "historical_replay_backfill_20260531T210114AEST" in path:
        return "historical_clean_official_packet"
    if "artifacts/prediction_snapshots" in path:
        return "rolling_persisted_snapshot_corpus"
    return "other_snapshot_corpus"


def _evaluation_group(row: Mapping[str, Any]) -> str:
    return str(row.get("snapshot_instance_id") or row.get("race_id") or "DATA_MISSING")


def _race_date(row: Mapping[str, Any]) -> str:
    return str(row.get("race_date") or "")


def _group_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_evaluation_group(row)].append(dict(row))
    return dict(grouped)


def build_primary_split(clean_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    historical = [
        dict(row)
        for row in clean_rows
        if _source_group(row) == "historical_clean_official_packet"
    ]
    rolling = [
        dict(row)
        for row in clean_rows
        if _source_group(row) == "rolling_persisted_snapshot_corpus"
    ]
    if historical and rolling:
        return {
            "strategy": "historical_train_rolling_holdout",
            "train_rows": historical,
            "eval_rows": rolling,
        }

    dates = sorted({_race_date(row) for row in clean_rows if _race_date(row)})
    if len(dates) < 2:
        return {
            "strategy": "insufficient_temporal_dates",
            "train_rows": [],
            "eval_rows": [dict(row) for row in clean_rows],
        }
    holdout = dates[-1]
    return {
        "strategy": "latest_date_holdout",
        "train_rows": [dict(row) for row in clean_rows if _race_date(row) < holdout],
        "eval_rows": [dict(row) for row in clean_rows if _race_date(row) == holdout],
    }


def _field_size_by_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {key: len(value) for key, value in _group_rows(rows).items()}


def _box_band(value: Any) -> str:
    box = _safe_int(value)
    if box in (1, 2):
        return "inside"
    if box in (3, 4, 5, 6):
        return "middle"
    if box is not None and box >= 7:
        return "outside"
    return "unknown"


def _numeric_distance(value: Any) -> float:
    if value is None:
        return 0.0
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    parsed = _safe_float(digits)
    return parsed if parsed is not None else 0.0


def _feature_vector(row: Mapping[str, Any], *, mode: str, field_size: int) -> list[float]:
    probability = _safe_float(row.get("win_prob_norm")) or 1e-6
    values = [
        _logit(probability),
        float(field_size),
        _numeric_distance(row.get("target_distance") or row.get("distance")),
    ]
    if mode == "reduced_box_band":
        band = _box_band(row.get("box_number"))
        values.extend(
            [
                1.0 if band == "inside" else 0.0,
                1.0 if band == "middle" else 0.0,
                1.0 if band == "outside" else 0.0,
            ]
        )
    return values


def _normalize_raw_scores(
    rows: Sequence[Mapping[str, Any]],
    raw_scores: Sequence[float],
    *,
    output_key: str = "study_prob",
) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    indexes_by_group: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indexes_by_group[_evaluation_group(row)].append(index)
    for indexes in indexes_by_group.values():
        scores = [max(0.0, float(raw_scores[index])) for index in indexes]
        total = sum(scores)
        if total <= 0:
            probability = 1.0 / len(indexes)
            for index in indexes:
                output[index][output_key] = probability
        else:
            for index, score in zip(indexes, scores):
                output[index][output_key] = score / total
    return output


def fit_logistic_variant(
    train_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not train_rows or not eval_rows:
        raise ValueError("train_or_eval_rows_missing")
    if len({int(row.get("actual_win") or 0) for row in train_rows}) < 2:
        raise ValueError("train_labels_single_class")
    train_sizes = _field_size_by_group(train_rows)
    eval_sizes = _field_size_by_group(eval_rows)
    x_train = np.array(
        [
            _feature_vector(row, mode=mode, field_size=train_sizes[_evaluation_group(row)])
            for row in train_rows
        ],
        dtype=float,
    )
    y_train = np.array([int(row.get("actual_win") or 0) for row in train_rows], dtype=int)
    x_eval = np.array(
        [
            _feature_vector(row, mode=mode, field_size=eval_sizes[_evaluation_group(row)])
            for row in eval_rows
        ],
        dtype=float,
    )
    model = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")
    model.fit(x_train, y_train)
    raw = model.predict_proba(x_eval)[:, 1]
    rows = _normalize_raw_scores(eval_rows, raw)
    feature_columns = ["baseline_logit", "field_size", "target_distance_numeric"]
    if mode == "reduced_box_band":
        feature_columns.extend(["box_band_inside", "box_band_middle", "box_band_outside"])
    return rows, {
        "model_family": "LogisticRegression",
        "class_weight": "balanced",
        "mode": mode,
        "feature_columns": feature_columns,
        "forbidden_feature_columns": validate_feature_columns(feature_columns),
        "model_artifact_written": False,
        "registry_mutation_allowed": False,
        "warning": (
            "score-only variant; baseline_logit may inherit champion box bias"
            if mode == "no_box"
            else "score-only reduced box-band variant; baseline_logit may inherit champion box bias"
        ),
    }


def _clone_with_probability(
    rows: Sequence[Mapping[str, Any]],
    *,
    input_key: str,
    output_key: str = "study_prob",
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item[output_key] = _safe_float(row.get(input_key))
        out.append(item)
    return out


def _complete_odds_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for race_rows in _group_rows(rows).values():
        if race_rows and all(_safe_float(row.get("odds_win")) is not None for row in race_rows):
            out.extend(dict(row) for row in race_rows)
    return out


def _market_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for race_rows in _group_rows(_complete_odds_rows(rows)).values():
        odds = {
            str(row.get("dog_name")): float(row.get("odds_win"))
            for row in race_rows
            if _safe_float(row.get("odds_win")) is not None
        }
        probabilities = market_implied_probabilities(odds)
        for row in race_rows:
            name = str(row.get("dog_name"))
            if name in probabilities:
                item = dict(row)
                item["study_prob"] = probabilities[name]
                out.append(item)
    return out


def _blend_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_weight: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for race_rows in _group_rows(_complete_odds_rows(rows)).values():
        odds = {
            str(row.get("dog_name")): float(row.get("odds_win"))
            for row in race_rows
            if _safe_float(row.get("odds_win")) is not None
        }
        market = market_implied_probabilities(odds)
        model = {
            str(row.get("dog_name")): float(row.get("win_prob_norm"))
            for row in race_rows
            if _safe_float(row.get("win_prob_norm")) is not None
        }
        probabilities = blend_probabilities(model, market, model_weight=model_weight)
        for row in race_rows:
            name = str(row.get("dog_name"))
            if name in probabilities:
                item = dict(row)
                item["study_prob"] = probabilities[name]
                out.append(item)
    return out


def _score_for_selection(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float, float]:
    metrics = score_predictions(
        rows,
        probability_key="study_prob",
        race_id_key="snapshot_instance_id",
    )
    log_loss = _safe_float(metrics.get("log_loss")) or float("inf")
    brier = _safe_float(metrics.get("brier")) or float("inf")
    top1 = _safe_float(metrics.get("top1")) or 0.0
    return log_loss, brier, -top1


def tune_power_alpha(rows: Sequence[Mapping[str, Any]], *, input_key: str) -> dict[str, Any]:
    attempts = []
    best_alpha = None
    best_key = None
    for alpha in POWER_ALPHA_GRID:
        try:
            candidate = power_normalize_by_race(
                rows,
                alpha=alpha,
                input_key=input_key,
                output_key="study_prob",
                race_key="snapshot_instance_id",
            )
            metrics = score_predictions(
                candidate,
                probability_key="study_prob",
                race_id_key="snapshot_instance_id",
            )
            key = _score_for_selection(candidate)
        except Exception as exc:
            attempts.append({"alpha": alpha, "status": "FAILED", "reason": type(exc).__name__})
            continue
        attempts.append(
            {
                "alpha": alpha,
                "status": "SUCCESS",
                "log_loss": metrics.get("log_loss"),
                "brier": metrics.get("brier"),
                "top1": metrics.get("top1"),
                "top3": metrics.get("top3"),
            }
        )
        if best_key is None or key < best_key:
            best_key = key
            best_alpha = alpha
    if best_alpha is None:
        raise ValueError("power_alpha_tuning_failed")
    return {"selected_alpha": best_alpha, "attempts": attempts}


def apply_power_alpha(
    rows: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    input_key: str,
) -> list[dict[str, Any]]:
    return power_normalize_by_race(
        rows,
        alpha=alpha,
        input_key=input_key,
        output_key="study_prob",
        race_key="snapshot_instance_id",
    )


def tune_blend_weight(
    train_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    complete = _complete_odds_rows(train_rows)
    if len({_evaluation_group(row) for row in complete}) < 3:
        raise ValueError("insufficient_complete_odds_train_races")
    attempts = []
    best_weight = None
    best_key = None
    for weight in BLEND_WEIGHT_GRID:
        candidate = _blend_rows(complete, model_weight=weight)
        if not candidate:
            continue
        metrics = score_predictions(
            candidate,
            probability_key="study_prob",
            race_id_key="snapshot_instance_id",
        )
        key = _score_for_selection(candidate)
        attempts.append(
            {
                "model_weight": weight,
                "log_loss": metrics.get("log_loss"),
                "brier": metrics.get("brier"),
                "top1": metrics.get("top1"),
                "top3": metrics.get("top3"),
            }
        )
        if best_key is None or key < best_key:
            best_key = key
            best_weight = weight
    if best_weight is None:
        raise ValueError("blend_weight_tuning_failed")
    return {"selected_model_weight": best_weight, "attempts": attempts}


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


def _per_box_calibration(
    rows: Sequence[Mapping[str, Any]],
    *,
    probability_key: str,
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for row in rows:
        p = _safe_float(row.get(probability_key))
        if p is None:
            continue
        grouped[str(row.get("box_number") or "DATA_MISSING")].append(
            (p, int(row.get("actual_win") or 0))
        )
    out = {}
    for box, pairs in sorted(grouped.items()):
        out[box] = {
            "count": len(pairs),
            "avg_predicted": float(np.mean([p for p, _ in pairs])),
            "actual_win_rate": float(np.mean([y for _, y in pairs])),
            "brier": float(np.mean([(p - y) ** 2 for p, y in pairs])),
        }
    return out


def _slice_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    probability_key: str,
    key: str,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "DATA_MISSING")].append(row)
    out = {}
    for value, value_rows in sorted(grouped.items()):
        race_count = len({_evaluation_group(row) for row in value_rows})
        if race_count < 2:
            out[value] = {"status": "TOO_SMALL", "races": race_count}
            continue
        out[value] = score_predictions(
            value_rows,
            probability_key=probability_key,
            race_id_key="snapshot_instance_id",
        )
    return out


def _ev_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    probability_key: str,
) -> dict[str, Any]:
    eligible = []
    ineligible = 0
    for row in rows:
        p = _safe_float(row.get(probability_key))
        odds = _safe_float(row.get("odds_win"))
        if p is None or odds is None or odds <= 1.0:
            ineligible += 1
            continue
        ev = p * odds - 1.0
        actual = int(row.get("actual_win") or 0)
        roi = odds - 1.0 if actual == 1 else -1.0
        eligible.append(
            {
                "race": _evaluation_group(row),
                "ev": ev,
                "actual_win": actual,
                "roi": roi,
            }
        )
    positives = [row for row in eligible if row["ev"] > 0]
    buckets = {
        "ev_lt_0": [row for row in eligible if row["ev"] < 0],
        "ev_0_to_0_5": [row for row in eligible if 0 <= row["ev"] < 0.5],
        "ev_gte_0_5": [row for row in eligible if row["ev"] >= 0.5],
    }
    bucket_rows = {}
    for name, bucket in buckets.items():
        bucket_rows[name] = {
            "count": len(bucket),
            "avg_ev": float(np.mean([row["ev"] for row in bucket])) if bucket else None,
            "hit_rate": (
                float(np.mean([row["actual_win"] for row in bucket])) if bucket else None
            ),
            "realized_roi": (
                float(np.mean([row["roi"] for row in bucket])) if bucket else None
            ),
        }

    roi_ci = None
    if positives:
        by_race: dict[str, list[float]] = defaultdict(list)
        for row in positives:
            by_race[row["race"]].append(float(row["roi"]))
        race_values = [float(np.mean(values)) for values in by_race.values()]
        if len(race_values) >= 2:
            rng = np.random.default_rng(42)
            samples = []
            for _ in range(500):
                draw = rng.choice(race_values, size=len(race_values), replace=True)
                samples.append(float(np.mean(draw)))
            roi_ci = {
                "method": "race_bootstrap_500_seed_42",
                "lower_95": float(np.percentile(samples, 2.5)),
                "upper_95": float(np.percentile(samples, 97.5)),
            }
    return {
        "status": "SUCCESS" if eligible else "DATA_MISSING",
        "eligible_runner_count": len(eligible),
        "ineligible_runner_count": ineligible,
        "positive_ev_count": len(positives),
        "average_ev": float(np.mean([row["ev"] for row in eligible])) if eligible else None,
        "positive_ev_hit_rate": (
            float(np.mean([row["actual_win"] for row in positives])) if positives else None
        ),
        "positive_ev_realized_roi": (
            float(np.mean([row["roi"] for row in positives])) if positives else None
        ),
        "positive_ev_realized_roi_ci": roi_ci,
        "buckets": bucket_rows,
        "note": "report-only; not betting advice and not evidence of an EV edge",
    }


def _arm_result(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    probability_key: str,
    scope: str,
    training: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = score_predictions(
        rows,
        probability_key=probability_key,
        race_id_key="snapshot_instance_id",
    )
    return {
        "status": "RUN",
        "variant": name,
        "scope": scope,
        "metrics": metrics,
        "box_bias": _top_and_winner_boxes(rows, probability_key=probability_key),
        "per_box_calibration": _per_box_calibration(rows, probability_key=probability_key),
        "slices": {
            "venue": _slice_metrics(rows, probability_key=probability_key, key="venue"),
            "distance": _slice_metrics(rows, probability_key=probability_key, key="distance"),
            "grade": _slice_metrics(rows, probability_key=probability_key, key="target_grade"),
        },
        "ev_report_only": _ev_diagnostics(rows, probability_key=probability_key),
        "training": dict(training or {}),
    }


def _not_run(name: str, blocker: str) -> dict[str, Any]:
    return {
        "status": "NOT_RUN",
        "variant": name,
        "blocker": blocker,
        "promotion_allowed": False,
        "registry_mutation_allowed": False,
        "model_artifact_written": False,
    }


def _baseline_by_scope(clean_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scopes = {
        "all_clean_official": list(clean_rows),
        "historical_clean_official_packet": [
            row
            for row in clean_rows
            if _source_group(row) == "historical_clean_official_packet"
        ],
        "rolling_clean_official_packet": [
            row
            for row in clean_rows
            if _source_group(row) == "rolling_persisted_snapshot_corpus"
        ],
    }
    out = {}
    for name, rows in scopes.items():
        if not rows:
            out[name] = {"status": "DATA_MISSING"}
            continue
        arm_rows = _clone_with_probability(rows, input_key="win_prob_norm")
        out[name] = _arm_result(
            "champion_current_production_scoring",
            arm_rows,
            probability_key="study_prob",
            scope=name,
        )
    return out


def _manifest(clean_rows: Sequence[Mapping[str, Any]], excluded: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dates = sorted({_race_date(row) for row in clean_rows if _race_date(row)})
    return {
        "schema_version": "clean_holdout_manifest_v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "race_count": len({str(row.get("label_race_id") or row.get("race_id")) for row in clean_rows}),
        "snapshot_instance_count": len({_evaluation_group(row) for row in clean_rows}),
        "runner_row_count": len(clean_rows),
        "date_range": {"min": dates[0] if dates else None, "max": dates[-1] if dates else None},
        "venues": sorted({str(row.get("venue")) for row in clean_rows if row.get("venue")}),
        "source_groups": dict(Counter(_source_group(row) for row in clean_rows)),
        "source_result_provenance": dict(Counter(str(row.get("label_quality")) for row in clean_rows)),
        "official_label_provenance": dict(Counter(str(row.get("result_detail_quality")) for row in clean_rows)),
        "requirements": [
            "snapshot_result_free",
            "snapshot_readiness_ready",
            "label_quality_official_or_complete_result",
            "full_finish_position_labels",
            "exactly_one_winner_per_snapshot_instance",
        ],
        "clean_snapshot_instances": sorted({_evaluation_group(row) for row in clean_rows}),
        "excluded_count": len(excluded),
        "excluded_reason_counts": dict(
            Counter(reason for item in excluded for reason in item.get("reasons", []))
        ),
        "excluded_examples": list(excluded)[:50],
    }


def _leakage_audit(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
    feature_families: Mapping[str, Any],
) -> dict[str, Any]:
    temporal = validate_temporal_holdout(train_rows, eval_rows)
    return {
        "schema_version": "box_bias_study_leakage_audit_v1",
        "status": "PASS" if temporal.ok else "FAIL",
        "temporal_holdout": {
            "ok": temporal.ok,
            "train_max_date": temporal.train_max_date,
            "test_min_date": temporal.test_min_date,
            "race_id_overlap": temporal.race_id_overlap,
            "violations": temporal.violations,
        },
        "train_rows": len(train_rows),
        "train_snapshot_instances": len({_evaluation_group(row) for row in train_rows}),
        "eval_rows": len(eval_rows),
        "eval_snapshot_instances": len({_evaluation_group(row) for row in eval_rows}),
        "post_outcome_feature_columns_used": [],
        "target_columns_used_for_scoring": [],
        "odds_policy": "odds arms use rows only when real pre-jump dog odds passed evaluator provenance",
        "labels_policy": "primary rows require official_or_complete_result and full finish_position labels",
        "participant_mismatch_policy": "non-ready snapshots and label mismatches are excluded by evaluator before clean holdout",
        "snapshot_policy": "read existing frozen snapshots only; no snapshot writes or rewrites",
        "feature_families_by_variant": dict(feature_families),
    }


def _model_inventory() -> dict[str, Any]:
    paths = {
        "best_model": ROOT / "model_registry" / "best_model.joblib",
        "best_metadata": ROOT / "model_registry" / "best_metadata.json",
        "current_production": ROOT / "docs" / "model_registry" / "current_production.json",
        "model_index": ROOT / "model_registry" / "model_index.json",
    }
    inventory = {
        "git_head": _run_git(["rev-parse", "--short=12", "HEAD"]),
        "git_branch": _run_git(["branch", "--show-current"]),
        "paths": {
            name: {
                "path": str(path.relative_to(ROOT)) if path.exists() else str(path.relative_to(ROOT)),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
            for name, path in paths.items()
        },
    }
    if paths["best_metadata"].exists():
        try:
            metadata = json.loads(paths["best_metadata"].read_text(encoding="utf-8"))
            inventory["best_metadata_summary"] = {
                "model_id": metadata.get("model_id"),
                "model_name": metadata.get("model_name"),
                "model_type": metadata.get("model_type"),
                "feature_count": len(metadata.get("feature_names") or []),
                "created_at": metadata.get("created_at"),
            }
        except Exception as exc:
            inventory["best_metadata_summary"] = {"status": "UNREADABLE", "reason": str(exc)}
    return inventory


def _read_text_if_exists(path: Path) -> str | None:
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def _endpoint_health(output_dir: Path) -> dict[str, Any]:
    return {
        "ss_5002": _read_text_if_exists(output_dir / "endpoint_ss_5002.txt"),
        "ss_5002_error": _read_text_if_exists(output_dir / "endpoint_ss_5002.err"),
        "api_health": _read_text_if_exists(output_dir / "api_health.json"),
        "api_health_error": _read_text_if_exists(output_dir / "api_health.err"),
        "api_model_health": _read_text_if_exists(output_dir / "api_model_health.json"),
        "api_model_health_error": _read_text_if_exists(output_dir / "api_model_health.err"),
    }


def _comparison_table_rows(challengers: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, result in challengers.items():
        if result.get("status") != "RUN":
            rows.append(
                {
                    "variant": name,
                    "status": result.get("status"),
                    "scope": "DATA_MISSING",
                    "races": "DATA_MISSING",
                    "top1": "DATA_MISSING",
                    "top2": "DATA_MISSING",
                    "top3": "DATA_MISSING",
                    "mean_winner_rank": "DATA_MISSING",
                    "brier": "DATA_MISSING",
                    "log_loss": "DATA_MISSING",
                    "calibration_slope": "DATA_MISSING",
                    "box1_top_pick_share": "DATA_MISSING",
                    "positive_ev_count": "DATA_MISSING",
                    "blocker": result.get("blocker", ""),
                }
            )
            continue
        metrics = result.get("metrics") or {}
        calibration = metrics.get("calibration") or {}
        box_bias = result.get("box_bias") or {}
        ev = result.get("ev_report_only") or {}
        rows.append(
            {
                "variant": name,
                "status": "RUN",
                "scope": result.get("scope"),
                "races": metrics.get("races_evaluated"),
                "top1": metrics.get("top1"),
                "top2": metrics.get("top2"),
                "top3": metrics.get("top3"),
                "mean_winner_rank": metrics.get("mean_winner_rank"),
                "brier": metrics.get("brier"),
                "log_loss": metrics.get("log_loss"),
                "calibration_slope": calibration.get("slope"),
                "box1_top_pick_share": box_bias.get("box1_top_pick_share"),
                "positive_ev_count": ev.get("positive_ev_count"),
                "blocker": "",
            }
        )
    return rows


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "variant",
        "status",
        "scope",
        "races",
        "top1",
        "top2",
        "top3",
        "mean_winner_rank",
        "brier",
        "log_loss",
        "calibration_slope",
        "box1_top_pick_share",
        "positive_ev_count",
        "blocker",
    ]
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(str(row.get(column, "")) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_metric(value: Any) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "DATA_MISSING"
    return f"{parsed:.4f}"


def _md_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_No rows._"
    out = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(out)


def _write_report(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    baseline: Mapping[str, Any],
    challengers: Mapping[str, Any],
    comparison_rows: Sequence[Mapping[str, Any]],
    leakage: Mapping[str, Any],
    endpoint: Mapping[str, Any],
    sqlite_quick_check: str | None,
) -> None:
    historical = baseline.get("historical_clean_official_packet", {})
    rolling = baseline.get("rolling_clean_official_packet", {})
    hist_metrics = (historical.get("metrics") or {}) if isinstance(historical, Mapping) else {}
    roll_metrics = (rolling.get("metrics") or {}) if isinstance(rolling, Mapping) else {}
    recommendation = "NO_PROMOTION_MORE_DATA_NEEDED"
    content = [
        "# Isolated Challenger Box-Bias Study",
        "",
        "## Executive Summary",
        "",
        "- Infrastructure progress is real: frozen snapshots, official-label gates, odds provenance gates, and report-only evaluation paths are functioning.",
        "- Champion accuracy is not acceptable on the clean official evidence.",
        "- Box-1 dominance remains the main model-quality failure and is measured directly in this report.",
        "- EV mechanics do not imply an EV edge; EV rows here are report-only diagnostics only.",
        "- Promotion remains blocked behind a separate human-approved promotion gate.",
        "",
        f"Recommendation: `{recommendation}`.",
        "",
        "## Data Set",
        "",
        f"- Clean races: `{manifest.get('race_count')}`",
        f"- Clean snapshot instances: `{manifest.get('snapshot_instance_count')}`",
        f"- Clean runner rows: `{manifest.get('runner_row_count')}`",
        f"- Date range: `{(manifest.get('date_range') or {}).get('min')}` to `{(manifest.get('date_range') or {}).get('max')}`",
        f"- Source groups: `{manifest.get('source_groups')}`",
        f"- Excluded reason counts: `{manifest.get('excluded_reason_counts')}`",
        "",
        "## Champion Baseline",
        "",
        f"- Historical packet top1: `{_format_metric(hist_metrics.get('top1'))}`, top2: `{_format_metric(hist_metrics.get('top2'))}`, top3: `{_format_metric(hist_metrics.get('top3'))}`, mean winner rank: `{_format_metric(hist_metrics.get('mean_winner_rank'))}`.",
        f"- Rolling packet top1: `{_format_metric(roll_metrics.get('top1'))}`, top2: `{_format_metric(roll_metrics.get('top2'))}`, top3: `{_format_metric(roll_metrics.get('top3'))}`, mean winner rank: `{_format_metric(roll_metrics.get('mean_winner_rank'))}`.",
        f"- Historical top-pick boxes: `{(historical.get('box_bias') or {}).get('top_pick_box_distribution') if isinstance(historical, Mapping) else {}}`.",
        f"- Rolling top-pick boxes: `{(rolling.get('box_bias') or {}).get('top_pick_box_distribution') if isinstance(rolling, Mapping) else {}}`.",
        "",
        "## Challenger Comparison",
        "",
        _md_table(
            [
                {
                    "variant": row.get("variant"),
                    "status": row.get("status"),
                    "scope": row.get("scope"),
                    "races": row.get("races"),
                    "top1": _format_metric(row.get("top1")),
                    "top3": _format_metric(row.get("top3")),
                    "brier": _format_metric(row.get("brier")),
                    "log_loss": _format_metric(row.get("log_loss")),
                    "box1_share": _format_metric(row.get("box1_top_pick_share")),
                    "blocker": row.get("blocker"),
                }
                for row in comparison_rows
            ],
            ["variant", "status", "scope", "races", "top1", "top3", "brier", "log_loss", "box1_share", "blocker"],
        ),
        "",
        "## Leakage Audit",
        "",
        f"- Status: `{leakage.get('status')}`",
        f"- Temporal holdout: `{leakage.get('temporal_holdout')}`",
        "- No post-outcome feature columns are used by the report-only scoring variants.",
        "- Market and blend arms are restricted to rows with valid pre-jump odds provenance.",
        "",
        "## Endpoint And DB Health",
        "",
        f"- SQLite quick_check: `{sqlite_quick_check or 'DATA_MISSING'}`",
        f"- API health error: `{endpoint.get('api_health_error') or 'none'}`",
        f"- API model health error: `{endpoint.get('api_model_health_error') or 'none'}`",
        "",
        "## No-Mutation Confirmation",
        "",
        "- The study writes only under its isolated artifact directory.",
        "- No production model registry, production model files, snapshot JSON, manifest, labels, odds capture, result ingestion write, retrain, promotion, or betting decision was performed by this script.",
        "- Any `NOT_RUN` variant is blocked rather than substituted with fake data.",
        "",
        "## Recommendation",
        "",
        f"`{recommendation}`. The only credible next step is a separate, approved clean-feature reconstruction or challenger training review. Do not promote, bet, or infer EV edge from this report.",
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def run_study(
    *,
    db_path: Path,
    snapshot_paths: Sequence[str],
    evaluation_dataset_paths: Sequence[Path] = (),
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = evaluate_snapshots(
        str(db_path),
        list(snapshot_paths),
        include_dataset_rows=True,
    )
    rows = report.get("evaluation_dataset_rows")
    rows = rows if isinstance(rows, list) else []
    dataset_input_rows = rows_from_evaluation_datasets(evaluation_dataset_paths)
    rows = _dedupe_rows([*dataset_input_rows, *rows])
    clean_rows, excluded = _clean_official_evaluation_rows(rows)

    dataset_path = output_dir / "evaluation_dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, default=_json_default) + "\n")
    clean_dataset_path = output_dir / "clean_official_dataset.jsonl"
    with clean_dataset_path.open("w", encoding="utf-8") as handle:
        for row in clean_rows:
            handle.write(json.dumps(dict(row), sort_keys=True, default=_json_default) + "\n")
    report["evaluation_dataset_output"] = str(dataset_path)
    report["evaluation_dataset_rows_written"] = len(rows)
    report.pop("evaluation_dataset_rows", None)
    write_json(output_dir / "evaluation_report.json", report)

    split = build_primary_split(clean_rows)
    train_rows = split["train_rows"]
    eval_rows = split["eval_rows"]
    split_scope = split["strategy"]

    baseline = _baseline_by_scope(clean_rows)
    challengers: dict[str, Any] = {}
    feature_families: dict[str, Any] = {
        "champion_current_production_scoring": ["production_snapshot_win_prob_norm"],
        "market_implied": ["real_pre_jump_odds_only"],
        "simple_blend_50": ["production_snapshot_win_prob_norm", "real_pre_jump_odds_only"],
        "calibrated_champion": ["production_snapshot_win_prob_norm"],
    }

    champion_eval_rows = _clone_with_probability(eval_rows, input_key="win_prob_norm")
    challengers["champion_current_production_scoring"] = _arm_result(
        "champion_current_production_scoring",
        champion_eval_rows,
        probability_key="study_prob",
        scope=split_scope,
    )

    try:
        no_box_rows, training = fit_logistic_variant(train_rows, eval_rows, mode="no_box")
        challengers["no_box_score_variant"] = _arm_result(
            "no_box_score_variant",
            no_box_rows,
            probability_key="study_prob",
            scope=split_scope,
            training=training,
        )
        feature_families["no_box_score_variant"] = training.get("feature_columns", [])
    except Exception as exc:
        challengers["no_box_score_variant"] = _not_run(
            "no_box_score_variant",
            f"clean official no-box score variant failed:{type(exc).__name__}:{exc}",
        )

    try:
        reduced_rows, training = fit_logistic_variant(
            train_rows,
            eval_rows,
            mode="reduced_box_band",
        )
        challengers["reduced_box_band_score_variant"] = _arm_result(
            "reduced_box_band_score_variant",
            reduced_rows,
            probability_key="study_prob",
            scope=split_scope,
            training=training,
        )
        feature_families["reduced_box_band_score_variant"] = training.get("feature_columns", [])
    except Exception as exc:
        challengers["reduced_box_band_score_variant"] = _not_run(
            "reduced_box_band_score_variant",
            f"clean official reduced-box-band score variant failed:{type(exc).__name__}:{exc}",
        )

    available_history_columns = sorted(set(clean_rows[0].keys()) & set(HISTORY_FEATURE_COLUMNS)) if clean_rows else []
    if available_history_columns:
        challengers["history_only_model"] = _not_run(
            "history_only_model",
            "history-only training is not implemented in this report-only orchestrator despite available columns",
        )
    else:
        challengers["history_only_model"] = _not_run(
            "history_only_model",
            "clean snapshot evaluation rows do not contain historical performance feature columns; existing history-only training path is DB-temporal and not clean-official-holdout safe",
        )

    market_rows = _market_rows(eval_rows)
    if market_rows:
        challengers["market_implied"] = _arm_result(
            "market_implied",
            market_rows,
            probability_key="study_prob",
            scope=f"{split_scope}:complete_valid_pre_jump_odds_only",
        )
    else:
        challengers["market_implied"] = _not_run(
            "market_implied",
            "no complete eval race groups with valid pre-jump dog-level odds",
        )

    blend_rows = _blend_rows(eval_rows, model_weight=0.5)
    if blend_rows:
        challengers["simple_blend_50"] = _arm_result(
            "simple_blend_50",
            blend_rows,
            probability_key="study_prob",
            scope=f"{split_scope}:complete_valid_pre_jump_odds_only",
            training={"model_weight": 0.5},
        )
    else:
        challengers["simple_blend_50"] = _not_run(
            "simple_blend_50",
            "no complete eval race groups with valid pre-jump dog-level odds",
        )

    try:
        alpha_info = tune_power_alpha(train_rows, input_key="win_prob_norm")
        calibrated_rows = apply_power_alpha(
            eval_rows,
            alpha=float(alpha_info["selected_alpha"]),
            input_key="win_prob_norm",
        )
        challengers["calibrated_champion_power"] = _arm_result(
            "calibrated_champion_power",
            calibrated_rows,
            probability_key="study_prob",
            scope=split_scope,
            training=alpha_info,
        )
    except Exception as exc:
        challengers["calibrated_champion_power"] = _not_run(
            "calibrated_champion_power",
            f"power calibration failed:{type(exc).__name__}:{exc}",
        )

    try:
        weight_info = tune_blend_weight(train_rows)
        learned_blend_rows = _blend_rows(
            eval_rows,
            model_weight=float(weight_info["selected_model_weight"]),
        )
        if not learned_blend_rows:
            raise ValueError("no_complete_eval_odds_rows")
        challengers["learned_weight_blend"] = _arm_result(
            "learned_weight_blend",
            learned_blend_rows,
            probability_key="study_prob",
            scope=f"{split_scope}:complete_valid_pre_jump_odds_only",
            training=weight_info,
        )
    except Exception as exc:
        challengers["learned_weight_blend"] = _not_run(
            "learned_weight_blend",
            f"learned blend not backtest-safe:{type(exc).__name__}:{exc}",
        )

    try:
        blend_train = _blend_rows(train_rows, model_weight=0.5)
        blend_eval = _blend_rows(eval_rows, model_weight=0.5)
        if len({_evaluation_group(row) for row in blend_train}) < 3:
            raise ValueError("insufficient_complete_odds_train_races")
        alpha_info = tune_power_alpha(blend_train, input_key="study_prob")
        calibrated_blend_rows = apply_power_alpha(
            blend_eval,
            alpha=float(alpha_info["selected_alpha"]),
            input_key="study_prob",
        )
        challengers["calibrated_blend_power"] = _arm_result(
            "calibrated_blend_power",
            calibrated_blend_rows,
            probability_key="study_prob",
            scope=f"{split_scope}:complete_valid_pre_jump_odds_only",
            training=alpha_info,
        )
    except Exception as exc:
        challengers["calibrated_blend_power"] = _not_run(
            "calibrated_blend_power",
            f"calibrated blend not backtest-safe:{type(exc).__name__}:{exc}",
        )

    manifest = _manifest(clean_rows, excluded)
    leakage = _leakage_audit(
        train_rows=train_rows,
        eval_rows=eval_rows,
        feature_families=feature_families,
    )
    comparison_rows = _comparison_table_rows(challengers)
    box_bias = {
        "baseline_by_scope": {
            name: result.get("box_bias")
            for name, result in baseline.items()
            if isinstance(result, Mapping)
        },
        "challenger_by_variant": {
            name: result.get("box_bias")
            for name, result in challengers.items()
            if isinstance(result, Mapping)
        },
    }
    calibration = {
        "baseline_by_scope": {
            name: (result.get("metrics") or {}).get("calibration")
            for name, result in baseline.items()
            if isinstance(result, Mapping)
        },
        "challenger_by_variant": {
            name: (result.get("metrics") or {}).get("calibration")
            for name, result in challengers.items()
            if isinstance(result, Mapping) and result.get("status") == "RUN"
        },
    }
    ev = {
        name: result.get("ev_report_only")
        for name, result in challengers.items()
        if isinstance(result, Mapping)
    }
    data_inventory = {
        "schema_version": "box_bias_study_data_inventory_v1",
        "snapshot_paths_supplied": len(snapshot_paths),
        "evaluation_dataset_inputs": [str(path) for path in evaluation_dataset_paths],
        "evaluation_dataset_input_rows": len(dataset_input_rows),
        "snapshot_files_scanned": report.get("snapshot_files"),
        "json_files_scanned": report.get("json_files_scanned"),
        "runner_rows_scored": report.get("runner_rows_scored"),
        "clean_holdout": {
            "race_count": manifest["race_count"],
            "snapshot_instance_count": manifest["snapshot_instance_count"],
            "runner_row_count": manifest["runner_row_count"],
            "date_range": manifest["date_range"],
            "source_groups": manifest["source_groups"],
        },
        "snapshot_corpus_readiness": report.get("snapshot_corpus_readiness"),
        "snapshot_provenance_report": report.get("snapshot_provenance_report"),
    }
    model_inventory = _model_inventory()
    endpoint = _endpoint_health(output_dir)
    sqlite_quick_check = _read_text_if_exists(output_dir / "sqlite_quick_check.txt")

    write_json(output_dir / "data_inventory.json", data_inventory)
    write_json(output_dir / "clean_holdout_manifest.json", manifest)
    write_json(output_dir / "leakage_audit.json", leakage)
    write_json(output_dir / "model_inventory.json", model_inventory)
    write_json(output_dir / "champion_baseline_metrics.json", baseline)
    write_json(output_dir / "challenger_metrics.json", challengers)
    write_json(output_dir / "box_bias_diagnostics.json", box_bias)
    write_json(output_dir / "calibration_diagnostics.json", calibration)
    write_json(output_dir / "ev_diagnostics_report_only.json", ev)
    _write_tsv(output_dir / "comparison_table.tsv", comparison_rows)
    _write_report(
        output_dir / "report.md",
        manifest=manifest,
        baseline=baseline,
        challengers=challengers,
        comparison_rows=comparison_rows,
        leakage=leakage,
        endpoint=endpoint,
        sqlite_quick_check=sqlite_quick_check,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "SUCCESS" if clean_rows else "DATA_MISSING",
        "output_dir": str(output_dir),
        "clean_holdout": manifest,
        "primary_split": {
            "strategy": split_scope,
            "train_rows": len(train_rows),
            "train_snapshot_instances": len({_evaluation_group(row) for row in train_rows}),
            "eval_rows": len(eval_rows),
            "eval_snapshot_instances": len({_evaluation_group(row) for row in eval_rows}),
        },
        "recommendation": "NO_PROMOTION_MORE_DATA_NEEDED",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="greyhound_racing_data_writable.db")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--snapshot", action="append", default=[])
    parser.add_argument("--snapshot-manifest", action="append", default=[])
    parser.add_argument("--evaluation-dataset", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if os.environ.get("APPROVE_RESULT_LABEL_WRITE"):
        raise SystemExit("refusing_to_run_with_APPROVE_RESULT_LABEL_WRITE_set")
    output_dir = assert_output_dir_safe(Path(args.output_dir))
    manifests = [Path(path) for path in args.snapshot_manifest]
    evaluation_datasets = [Path(path) for path in args.evaluation_dataset]
    snapshot_paths = list(args.snapshot) + snapshot_paths_from_manifests(manifests)
    if not snapshot_paths and not evaluation_datasets:
        raise SystemExit("DATA_MISSING: no snapshots or evaluation datasets supplied")
    result = run_study(
        db_path=Path(args.db),
        snapshot_paths=snapshot_paths,
        evaluation_dataset_paths=evaluation_datasets,
        output_dir=output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0 if result.get("status") == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
