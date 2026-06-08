"""Probability calibration helpers for report-only and gated runtime paths."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def power_normalize_by_race(
    rows: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    input_key: str = "win_prob_norm",
    output_key: str,
    race_key: str = "race_id",
    min_probability: float = 1e-12,
) -> list[dict[str, Any]]:
    """Apply positive-alpha power normalization independently per race.

    The transform is additive: input rows are copied, and the calibrated
    probability is written to ``output_key``. It intentionally depends only on
    model probabilities and race grouping, not labels, odds, or results.
    """

    parsed_alpha = _finite_float(alpha)
    if parsed_alpha is None or parsed_alpha <= 0:
        raise ValueError("alpha_must_be_positive_finite")
    parsed_floor = _finite_float(min_probability)
    if parsed_floor is None or parsed_floor <= 0:
        raise ValueError("min_probability_must_be_positive_finite")
    if not input_key:
        raise ValueError("input_key_missing")
    if not output_key:
        raise ValueError("output_key_missing")
    if not race_key:
        raise ValueError("race_key_missing")
    if not rows:
        raise ValueError("rows_missing")

    output: list[dict[str, Any]] = []
    grouped_indexes: dict[str, list[int]] = defaultdict(list)
    original_probability_totals: dict[str, float] = defaultdict(float)
    raw_scores: list[float] = []

    for index, row in enumerate(rows):
        race_id = row.get(race_key)
        if race_id in (None, ""):
            raise ValueError(f"{race_key}_missing")
        probability = _finite_float(row.get(input_key))
        if probability is None:
            raise ValueError(f"{input_key}_invalid")
        if probability < 0:
            raise ValueError(f"{input_key}_negative")

        race_id_text = str(race_id)
        output.append(dict(row))
        grouped_indexes[race_id_text].append(index)
        original_probability_totals[race_id_text] += probability
        raw_scores.append(max(parsed_floor, probability) ** parsed_alpha)

    for race_id, indexes in grouped_indexes.items():
        if original_probability_totals[race_id] <= 0:
            raise ValueError(f"{input_key}_race_total_nonpositive")
        total = sum(raw_scores[index] for index in indexes)
        if total <= 0:
            raise ValueError(f"{output_key}_race_total_nonpositive")
        for index in indexes:
            output[index][output_key] = raw_scores[index] / total

    return output


def power_normalize_prediction_group(
    predictions: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    input_key: str = "win_prob_norm",
    output_key: str = "calibrated_win_prob_report_only",
    output_rank_key: str = "calibrated_predicted_rank_report_only",
) -> list[dict[str, Any]]:
    """Apply report-only power calibration to one race's prediction rows."""

    temporary_race_key = "__calibration_race_id"
    prepared: list[dict[str, Any]] = []
    for row in predictions:
        if temporary_race_key in row:
            raise ValueError("reserved_calibration_key_present")
        item = dict(row)
        item[temporary_race_key] = "__single_prediction_group__"
        prepared.append(item)

    calibrated = power_normalize_by_race(
        prepared,
        alpha=alpha,
        input_key=input_key,
        output_key=output_key,
        race_key=temporary_race_key,
    )
    ranked_indexes = sorted(
        range(len(calibrated)),
        key=lambda index: (-float(calibrated[index][output_key]), index),
    )
    for rank, index in enumerate(ranked_indexes, start=1):
        calibrated[index].pop(temporary_race_key, None)
        calibrated[index][output_rank_key] = rank
    return calibrated
