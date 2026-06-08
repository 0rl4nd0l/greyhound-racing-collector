#!/usr/bin/env python3
"""Report-only challenger review for frozen snapshot evaluation rows.

This script fits challenger models in memory only. It writes a JSON review
artifact and never saves a model, registers a model, promotes a model, or bets.
"""

from __future__ import annotations

import argparse
import json
import math
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

from accuracy_program.evaluation import (
    score_predictions,
    validate_feature_columns,
    validate_temporal_holdout,
)
from accuracy_program.calibration import power_normalize_by_race


SCHEMA_VERSION = "snapshot_challenger_review_v1"
DEFAULT_HOLDOUT_DATE_COUNT = 1
FEATURE_COLUMNS = (
    "baseline_win_prob",
    "baseline_logit",
    "box_number",
    "box_share",
    "field_size",
)
POWER_ALPHA_GRID = (
    0.25,
    0.35,
    0.5,
    0.65,
    0.8,
    1.0,
    1.25,
    1.5,
    2.0,
)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return default
        return parsed
    except Exception:
        return default


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            item = json.loads(raw)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _clean_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    for row in rows:
        probability = _safe_float(row.get("win_prob_norm"))
        actual = row.get("actual_win")
        if probability is None:
            continue
        if str(row.get("label_quality") or "") != "official_or_complete_result":
            continue
        if str(row.get("result_detail_quality") or "") != "finish_position":
            continue
        if actual not in (0, 1, "0", "1"):
            continue
        if not row.get("race_id") or not row.get("race_date"):
            continue
        item = dict(row)
        item["win_prob_norm"] = max(1e-9, min(1.0 - 1e-9, float(probability)))
        item["actual_win"] = int(actual)
        clean.append(item)
    return clean


def _group_by_race(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["race_id"])].append(dict(row))
    return dict(grouped)


def _add_field_size(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_by_race(rows)
    out: list[dict[str, Any]] = []
    for race_rows in grouped.values():
        field_size = len(race_rows)
        for row in race_rows:
            item = dict(row)
            item["field_size"] = field_size
            out.append(item)
    return out


def _temporal_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    holdout_date_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    dates = sorted({str(row["race_date"]) for row in rows if row.get("race_date")})
    if len(dates) < 2:
        return [], [], dates
    count = max(1, min(int(holdout_date_count), len(dates) - 1))
    holdout_dates = set(dates[-count:])
    train_rows = [dict(row) for row in rows if str(row["race_date"]) not in holdout_dates]
    test_rows = [dict(row) for row in rows if str(row["race_date"]) in holdout_dates]
    return train_rows, test_rows, sorted(holdout_dates)


def _row_features(row: Mapping[str, Any]) -> list[float]:
    p = max(1e-9, min(1.0 - 1e-9, float(row["win_prob_norm"])))
    box = _safe_float(row.get("box_number"), 0.0) or 0.0
    field_size = max(1.0, _safe_float(row.get("field_size"), 1.0) or 1.0)
    return [
        p,
        math.log(p / (1.0 - p)),
        box,
        box / field_size,
        field_size,
    ]


def _normalize_by_race(
    rows: Sequence[Mapping[str, Any]],
    raw_scores: Sequence[float],
    *,
    output_key: str,
) -> list[dict[str, Any]]:
    grouped_indexes: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped_indexes[str(row["race_id"])].append(index)
    output = [dict(row) for row in rows]
    for indexes in grouped_indexes.values():
        scores = [max(0.0, float(raw_scores[index])) for index in indexes]
        total = sum(scores)
        if total <= 0:
            prob = 1.0 / len(indexes)
            for index in indexes:
                output[index][output_key] = prob
        else:
            for index, score in zip(indexes, scores):
                output[index][output_key] = score / total
    return output


def _uniform_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_by_race(rows)
    out: list[dict[str, Any]] = []
    for race_rows in grouped.values():
        probability = 1.0 / len(race_rows)
        for row in race_rows:
            item = dict(row)
            item["uniform_prob"] = probability
            out.append(item)
    return out


def _power_calibrated_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    output_key: str,
) -> list[dict[str, Any]]:
    return power_normalize_by_race(
        rows,
        alpha=alpha,
        input_key="win_prob_norm",
        output_key=output_key,
    )


def _tune_power_alpha(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    alphas: Sequence[float] = POWER_ALPHA_GRID,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    best_alpha = 1.0
    best_metrics: dict[str, Any] = {}
    best_key: tuple[float, float, float] | None = None
    for alpha in alphas:
        candidate_rows = _power_calibrated_rows(
            train_rows,
            alpha=alpha,
            output_key="power_calibrated_prob",
        )
        metrics = _score(candidate_rows, "power_calibrated_prob")
        log_loss = _safe_float(metrics.get("log_loss"), float("inf")) or float("inf")
        brier = _safe_float(metrics.get("brier"), float("inf")) or float("inf")
        top1 = _safe_float(metrics.get("top1"), 0.0) or 0.0
        key = (log_loss, brier, -top1)
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
            best_metrics = metrics
    return best_alpha, best_metrics, attempts


def _box_prior_rows(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    box_counts: Counter[int] = Counter()
    box_wins: Counter[int] = Counter()
    for row in train_rows:
        box = int(row.get("box_number") or 0)
        if box <= 0:
            continue
        box_counts[box] += 1
        box_wins[box] += int(row.get("actual_win") or 0)
    overall_rate = (
        sum(box_wins.values()) / sum(box_counts.values()) if box_counts else 0.125
    )
    raw = []
    for row in test_rows:
        box = int(row.get("box_number") or 0)
        wins = box_wins.get(box, 0)
        count = box_counts.get(box, 0)
        raw.append((wins + overall_rate) / (count + 1.0))
    return _normalize_by_race(test_rows, raw, output_key="box_prior_prob")


def _challenger_rows(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    x_train = np.array([_row_features(row) for row in train_rows], dtype=float)
    y_train = np.array([int(row["actual_win"]) for row in train_rows], dtype=int)
    model = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")
    model.fit(x_train, y_train)
    raw = model.predict_proba(
        np.array([_row_features(row) for row in test_rows], dtype=float)
    )[:, 1]
    rows = _normalize_by_race(test_rows, raw, output_key="challenger_prob")
    return rows, {
        "model_family": "LogisticRegression",
        "class_weight": "balanced",
        "feature_columns": list(FEATURE_COLUMNS),
        "model_artifact_written": False,
        "registry_mutation_allowed": False,
    }


def _score(rows: Sequence[Mapping[str, Any]], probability_key: str) -> dict[str, Any]:
    return score_predictions(rows, probability_key=probability_key)


def _comparison(challenger: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    def lower(metric: str) -> bool | None:
        c = _safe_float(challenger.get(metric))
        b = _safe_float(baseline.get(metric))
        return None if c is None or b is None else c < b

    def higher(metric: str) -> bool | None:
        c = _safe_float(challenger.get(metric))
        b = _safe_float(baseline.get(metric))
        return None if c is None or b is None else c > b

    return {
        "log_loss_improved": lower("log_loss"),
        "brier_improved": lower("brier"),
        "top1_improved": higher("top1"),
        "top3_improved": higher("top3"),
    }


def _best_by_metric(arms: Mapping[str, Mapping[str, Any]], metric: str) -> str | None:
    best_name: str | None = None
    best_value: float | None = None
    for name, metrics in arms.items():
        value = _safe_float(metrics.get(metric))
        if value is None:
            continue
        if best_value is None or value < best_value:
            best_name = name
            best_value = value
    return best_name


def _metric_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "races_evaluated": metrics.get("races_evaluated"),
        "dog_predictions_evaluated": metrics.get("dog_predictions_evaluated"),
        "top1": metrics.get("top1"),
        "top2": metrics.get("top2"),
        "top3": metrics.get("top3"),
        "log_loss": metrics.get("log_loss"),
        "brier": metrics.get("brier"),
        "mean_winner_rank": metrics.get("mean_winner_rank"),
    }


def _evaluate_split_arms(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    failures: list[str] = []
    baseline_rows = [dict(row) for row in test_rows]
    uniform_rows = _uniform_rows(test_rows)
    box_rows = _box_prior_rows(train_rows, test_rows)
    power_alpha, power_train_metrics, power_attempts = _tune_power_alpha(train_rows)
    power_rows = _power_calibrated_rows(
        test_rows,
        alpha=power_alpha,
        output_key="power_calibrated_prob",
    )
    try:
        challenger_rows, challenger_training = _challenger_rows(
            train_rows,
            test_rows,
        )
    except Exception as exc:
        return {}, None, [f"challenger_training_failed:{type(exc).__name__}"]

    arms = {
        "baseline_model": _score(baseline_rows, "win_prob_norm"),
        "uniform": _score(uniform_rows, "uniform_prob"),
        "box_prior": _score(box_rows, "box_prior_prob"),
        "power_calibrated_baseline": _score(
            power_rows,
            "power_calibrated_prob",
        ),
        "logistic_numeric_challenger": _score(
            challenger_rows,
            "challenger_prob",
        ),
    }
    challenger_training["power_calibration"] = {
        "selected_alpha": power_alpha,
        "selection_metric": "train_log_loss_then_brier",
        "train_metrics": {
            "log_loss": power_train_metrics.get("log_loss"),
            "brier": power_train_metrics.get("brier"),
            "top1": power_train_metrics.get("top1"),
            "top3": power_train_metrics.get("top3"),
        },
        "attempts": power_attempts,
        "ranking_preserving": power_alpha > 0,
        "model_artifact_written": False,
        "registry_mutation_allowed": False,
    }
    return arms, challenger_training, failures


def _rolling_stability_review(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_split_count: int = 2,
) -> dict[str, Any]:
    dates = sorted({str(row["race_date"]) for row in rows if row.get("race_date")})
    splits: list[dict[str, Any]] = []
    for holdout_date in dates[1:]:
        train_rows = [dict(row) for row in rows if str(row["race_date"]) < holdout_date]
        test_rows = [dict(row) for row in rows if str(row["race_date"]) == holdout_date]
        split_check = validate_temporal_holdout(train_rows, test_rows)
        split_failures = [
            f"temporal_holdout:{item}" for item in split_check.violations
        ]
        if len({row.get("race_id") for row in train_rows}) < 10:
            split_failures.append("small_train_race_count")
        if len({row.get("race_id") for row in test_rows}) < 10:
            split_failures.append("small_holdout_race_count")
        arms: dict[str, Any] = {}
        training: dict[str, Any] | None = None
        if not split_failures:
            arms, training, arm_failures = _evaluate_split_arms(train_rows, test_rows)
            split_failures.extend(arm_failures)
        baseline = arms.get("baseline_model") or {}
        power = arms.get("power_calibrated_baseline") or {}
        comparison = _comparison(power, baseline) if arms else {}
        ranking_preserved = (
            power.get("top1") == baseline.get("top1")
            and power.get("top2") == baseline.get("top2")
            and power.get("top3") == baseline.get("top3")
            and power.get("mean_winner_rank") == baseline.get("mean_winner_rank")
        )
        splits.append(
            {
                "holdout_date": holdout_date,
                "status": "SUCCESS" if not split_failures else "NOT_READY",
                "failures": split_failures,
                "train_rows": len(train_rows),
                "train_races": len({row.get("race_id") for row in train_rows}),
                "test_rows": len(test_rows),
                "test_races": len({row.get("race_id") for row in test_rows}),
                "temporal_holdout_check": {
                    "ok": split_check.ok,
                    "train_max_date": split_check.train_max_date,
                    "test_min_date": split_check.test_min_date,
                    "race_id_overlap": split_check.race_id_overlap,
                    "violations": split_check.violations,
                },
                "power_calibration": (
                    (training or {}).get("power_calibration")
                    if isinstance(training, dict)
                    else None
                ),
                "baseline_model": _metric_summary(baseline),
                "power_calibrated_baseline": _metric_summary(power),
                "comparison_to_baseline": comparison,
                "ranking_preserved": ranking_preserved if arms else False,
            }
        )

    successful_splits = [split for split in splits if split["status"] == "SUCCESS"]
    failed_split_count = len(splits) - len(successful_splits)
    all_log_loss_improved = bool(successful_splits) and all(
        (split["comparison_to_baseline"] or {}).get("log_loss_improved") is True
        for split in successful_splits
    )
    all_brier_improved = bool(successful_splits) and all(
        (split["comparison_to_baseline"] or {}).get("brier_improved") is True
        for split in successful_splits
    )
    all_ranking_preserved = bool(successful_splits) and all(
        split.get("ranking_preserved") is True for split in successful_splits
    )
    enough_splits = len(successful_splits) >= minimum_split_count
    stable = (
        enough_splits
        and failed_split_count == 0
        and all_log_loss_improved
        and all_brier_improved
        and all_ranking_preserved
    )
    return {
        "schema_version": "snapshot_challenger_stability_v1",
        "candidate_arm": "power_calibrated_baseline",
        "status": "STABLE_REPORT_ONLY" if stable else "NOT_STABLE",
        "minimum_split_count": minimum_split_count,
        "split_count": len(successful_splits),
        "failed_split_count": failed_split_count,
        "all_log_loss_improved": all_log_loss_improved,
        "all_brier_improved": all_brier_improved,
        "all_ranking_preserved": all_ranking_preserved,
        "promotion_allowed": False,
        "reason": (
            "stability review is report-only; promotion still requires an "
            "explicit approved deployment path and broader held-out evidence"
        ),
        "splits": splits,
    }


def build_review(
    *,
    dataset_path: Path,
    holdout_date_count: int = DEFAULT_HOLDOUT_DATE_COUNT,
) -> dict[str, Any]:
    rows = _load_jsonl(dataset_path)
    clean_rows = _add_field_size(_clean_rows(rows))
    train_rows, test_rows, holdout_dates = _temporal_split(
        clean_rows,
        holdout_date_count=holdout_date_count,
    )
    feature_violations = validate_feature_columns(FEATURE_COLUMNS)
    split_check = validate_temporal_holdout(train_rows, test_rows)

    failures: list[str] = []
    warnings: list[str] = []
    if not rows:
        failures.append("dataset_empty")
    if len(clean_rows) <= 0:
        failures.append("clean_official_rows_zero")
    if feature_violations:
        failures.append("forbidden_feature_columns_present")
    if not split_check.ok:
        failures.extend(f"temporal_holdout:{item}" for item in split_check.violations)
    if len({row.get("race_id") for row in train_rows}) < 10:
        warnings.append("small_train_race_count")
    if len({row.get("race_id") for row in test_rows}) < 10:
        warnings.append("small_holdout_race_count")
    if len({row.get("actual_win") for row in train_rows}) < 2:
        failures.append("train_labels_single_class")

    arms: dict[str, Any] = {}
    challenger_training: dict[str, Any] | None = None
    if not failures:
        arms, challenger_training, arm_failures = _evaluate_split_arms(
            train_rows,
            test_rows,
        )
        failures.extend(arm_failures)

    comparison = {}
    if arms:
        comparison = _comparison(
            arms["logistic_numeric_challenger"],
            arms["baseline_model"],
        )
        comparison["by_arm"] = {
            name: _comparison(metrics, arms["baseline_model"])
            for name, metrics in arms.items()
            if name != "baseline_model"
        }
        comparison["best_log_loss_arm"] = _best_by_metric(arms, "log_loss")
        comparison["best_brier_arm"] = _best_by_metric(arms, "brier")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "status": "SUCCESS" if not failures else "NOT_READY",
        "failures": failures,
        "warnings": warnings,
        "source_evidence": {
            "evaluation_dataset": str(dataset_path.resolve()),
            "rows_loaded": len(rows),
            "clean_official_rows": len(clean_rows),
            "clean_official_races": len({row.get("race_id") for row in clean_rows}),
        },
        "feature_safety": {
            "feature_columns": list(FEATURE_COLUMNS),
            "forbidden_feature_columns": feature_violations,
            "target_columns_used_only_for_training_and_scoring": [
                "actual_win",
                "finish_position",
            ],
        },
        "temporal_holdout": {
            "holdout_date_count": int(holdout_date_count),
            "holdout_dates": holdout_dates,
            "train_rows": len(train_rows),
            "train_races": len({row.get("race_id") for row in train_rows}),
            "test_rows": len(test_rows),
            "test_races": len({row.get("race_id") for row in test_rows}),
            "check": {
                "ok": split_check.ok,
                "train_max_date": split_check.train_max_date,
                "test_min_date": split_check.test_min_date,
                "race_id_overlap": split_check.race_id_overlap,
                "violations": split_check.violations,
            },
        },
        "arms": arms,
        "comparison_to_baseline": comparison,
        "stability_review": _rolling_stability_review(clean_rows),
        "challenger_training": challenger_training,
        "promotion_control": {
            "action_taken": "none",
            "model_artifact_written": False,
            "registry_mutation_allowed": False,
            "promotion_allowed": False,
            "required_gate": "APPROVE_MODEL_PROMOTION",
            "reason": (
                "report-only challenger review; promotion still requires a "
                "separate approved promotion gate and stronger held-out evidence"
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Evaluation dataset JSONL")
    parser.add_argument("--output", required=True, help="Review JSON output path")
    parser.add_argument(
        "--holdout-date-count",
        type=int,
        default=DEFAULT_HOLDOUT_DATE_COUNT,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_review(
        dataset_path=Path(args.dataset),
        holdout_date_count=args.holdout_date_count,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["status"] == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
