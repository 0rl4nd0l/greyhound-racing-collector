"""Leakage-safe prediction evaluation primitives."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FORBIDDEN_FEATURE_COLUMNS = {
    "actual_results",
    "actual_winner",
    "beaten_margin",
    "finish_position",
    "margin",
    "official_result",
    "official_results",
    "placing",
    "race_result",
    "race_results",
    "result",
    "result_status",
    "results_status",
    "scraped_finish_position",
    "scraped_raw_result",
    "target_finish_position",
    "winner",
    "winner_margin",
    "winner_name",
    "winner_odds",
    "winning_time",
}


@dataclass(frozen=True)
class TemporalHoldoutCheck:
    ok: bool
    train_max_date: str | None
    test_min_date: str | None
    race_id_overlap: list[str]
    violations: list[str]


def _norm_column(name: str) -> str:
    return str(name).strip().lower()


def validate_feature_columns(columns: Iterable[str]) -> list[str]:
    """Return forbidden post-result/target columns present in feature inputs."""

    present = {_norm_column(col) for col in columns}
    return sorted(present & FORBIDDEN_FEATURE_COLUMNS)


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw[:10]).date()
    except ValueError:
        return None


def validate_temporal_holdout(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    *,
    race_id_key: str = "race_id",
    date_key: str = "race_date",
) -> TemporalHoldoutCheck:
    """Validate no race overlap and test dates strictly after train dates."""

    train_ids = {str(row.get(race_id_key)) for row in train_rows if row.get(race_id_key)}
    test_ids = {str(row.get(race_id_key)) for row in test_rows if row.get(race_id_key)}
    overlap = sorted(train_ids & test_ids)
    train_dates = [_parse_date(row.get(date_key)) for row in train_rows]
    test_dates = [_parse_date(row.get(date_key)) for row in test_rows]
    train_dates = [d for d in train_dates if d is not None]
    test_dates = [d for d in test_dates if d is not None]

    train_max = max(train_dates) if train_dates else None
    test_min = min(test_dates) if test_dates else None
    violations: list[str] = []
    if overlap:
        violations.append("race_id_overlap")
    if train_max is None or test_min is None:
        violations.append("missing_temporal_dates")
    elif test_min <= train_max:
        violations.append("test_not_strictly_after_train")

    return TemporalHoldoutCheck(
        ok=not violations,
        train_max_date=train_max.isoformat() if train_max else None,
        test_min_date=test_min.isoformat() if test_min else None,
        race_id_overlap=overlap,
        violations=violations,
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


def _race_groups(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        race_id = row.get("race_id")
        if race_id is None:
            continue
        groups[str(race_id)].append(row)
    return groups


def _brier(pairs: list[tuple[float, int]]) -> float | None:
    if not pairs:
        return None
    return float(np.mean([(p - y) ** 2 for p, y in pairs]))


def _log_loss_by_race(groups: dict[str, list[Mapping[str, Any]]], probability_key: str, actual_key: str) -> float | None:
    losses: list[float] = []
    eps = 1e-12
    for rows in groups.values():
        winner_probs = [
            _safe_float(row.get(probability_key))
            for row in rows
            if int(row.get(actual_key) or 0) == 1
        ]
        winner_probs = [p for p in winner_probs if p is not None]
        if not winner_probs:
            continue
        losses.append(-math.log(max(eps, min(1.0, winner_probs[0]))))
    return float(np.mean(losses)) if losses else None


def _calibration(pairs: list[tuple[float, int]]) -> dict[str, float | None]:
    if len(pairs) < 10 or len({y for _, y in pairs}) < 2:
        return {"slope": None, "intercept": None}
    probs = np.array([max(1e-6, min(1.0 - 1e-6, p)) for p, _ in pairs], dtype=float)
    y = np.array([y for _, y in pairs], dtype=int)
    x = np.log(probs / (1.0 - probs)).reshape(-1, 1)
    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(solver="lbfgs")
        model.fit(x, y)
        return {
            "slope": float(model.coef_[0][0]),
            "intercept": float(model.intercept_[0]),
        }
    except Exception:
        return {"slope": None, "intercept": None}


def _reliability_bins(
    pairs: list[tuple[float, int]],
    *,
    bins: int = 10,
) -> list[dict[str, float | int]]:
    bucketed: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for prob, actual in pairs:
        idx = max(0, min(bins - 1, int(prob * bins)))
        bucketed[idx].append((prob, actual))
    out = []
    for idx, bucket in enumerate(bucketed):
        if not bucket:
            continue
        out.append(
            {
                "bin": idx,
                "lower": idx / bins,
                "upper": (idx + 1) / bins,
                "count": len(bucket),
                "avg_predicted": float(np.mean([p for p, _ in bucket])),
                "actual_rate": float(np.mean([y for _, y in bucket])),
            }
        )
    return out


def _roi_by_ev_decile(
    rows: list[Mapping[str, Any]],
    *,
    probability_key: str,
    actual_key: str,
    odds_key: str,
) -> list[dict[str, float | int]]:
    eligible = []
    for row in rows:
        p = _safe_float(row.get(probability_key))
        odds = _safe_float(row.get(odds_key))
        if p is None or odds is None or odds <= 1:
            continue
        ev = p * odds - 1.0
        actual = int(row.get(actual_key) or 0)
        roi = odds - 1.0 if actual == 1 else -1.0
        eligible.append((ev, roi))
    if not eligible:
        return []
    eligible.sort(key=lambda item: item[0])
    out = []
    chunks = np.array_split(np.array(eligible, dtype=float), min(10, len(eligible)))
    for idx, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        out.append(
            {
                "decile": idx + 1,
                "count": int(len(chunk)),
                "ev_min": float(np.min(chunk[:, 0])),
                "ev_max": float(np.max(chunk[:, 0])),
                "avg_ev": float(np.mean(chunk[:, 0])),
                "realized_roi": float(np.mean(chunk[:, 1])),
            }
        )
    return out


def score_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    probability_key: str = "win_prob_norm",
    actual_key: str = "actual_win",
    odds_key: str = "odds_win",
) -> dict[str, Any]:
    """Score already frozen dog-level predictions with result labels."""

    groups = _race_groups(rows)
    top_hits = {1: 0, 2: 0, 3: 0}
    scored_races = 0
    prob_sum_errors: list[float] = []
    pairs: list[tuple[float, int]] = []
    winner_ranks: list[int] = []

    for race_rows in groups.values():
        ranked = sorted(
            race_rows,
            key=lambda row: _safe_float(row.get(probability_key)) or 0.0,
            reverse=True,
        )
        if not ranked:
            continue
        winners = {
            str(row.get("dog_name") or row.get("dog_clean_name") or row.get("box_number"))
            for row in ranked
            if int(row.get(actual_key) or 0) == 1
        }
        if not winners:
            continue
        scored_races += 1
        for idx, row in enumerate(ranked, start=1):
            runner_key = str(row.get("dog_name") or row.get("dog_clean_name") or row.get("box_number"))
            if runner_key in winners:
                winner_ranks.append(idx)
                break
        for k in (1, 2, 3):
            top_names = {
                str(row.get("dog_name") or row.get("dog_clean_name") or row.get("box_number"))
                for row in ranked[:k]
            }
            top_hits[k] += int(bool(winners & top_names))
        probs = [_safe_float(row.get(probability_key)) for row in ranked]
        probs = [p for p in probs if p is not None]
        if probs:
            prob_sum_errors.append(abs(sum(probs) - 1.0))
        for row in ranked:
            p = _safe_float(row.get(probability_key))
            if p is not None:
                pairs.append((p, int(row.get(actual_key) or 0)))

    rows_list = list(rows)
    return {
        "races_evaluated": scored_races,
        "dog_predictions_evaluated": len(pairs),
        "top1": top_hits[1] / scored_races if scored_races else None,
        "top2": top_hits[2] / scored_races if scored_races else None,
        "top3": top_hits[3] / scored_races if scored_races else None,
        "winner_ranks": winner_ranks,
        "winner_rank_counts": {
            str(rank): winner_ranks.count(rank) for rank in sorted(set(winner_ranks))
        },
        "mean_winner_rank": float(np.mean(winner_ranks)) if winner_ranks else None,
        "brier": _brier(pairs),
        "log_loss": _log_loss_by_race(groups, probability_key, actual_key),
        "calibration": _calibration(pairs),
        "reliability_bins": _reliability_bins(pairs),
        "roi_ev_by_decile": _roi_by_ev_decile(
            rows_list,
            probability_key=probability_key,
            actual_key=actual_key,
            odds_key=odds_key,
        ),
        "probability_sum": {
            "max_abs_error": max(prob_sum_errors) if prob_sum_errors else None,
            "mean_abs_error": float(np.mean(prob_sum_errors)) if prob_sum_errors else None,
        },
    }


def market_implied_probabilities(odds_by_runner: Mapping[str, float]) -> dict[str, float]:
    implied = {
        runner: 1.0 / float(odds)
        for runner, odds in odds_by_runner.items()
        if odds is not None and float(odds) > 1.0
    }
    total = sum(implied.values())
    if total <= 0:
        return {}
    return {runner: value / total for runner, value in implied.items()}


def blend_probabilities(
    model_probs: Mapping[str, float],
    market_probs: Mapping[str, float],
    *,
    model_weight: float,
) -> dict[str, float]:
    """Return a normalized simple blend for experiment reporting only."""

    weight = max(0.0, min(1.0, float(model_weight)))
    runners = set(model_probs) | set(market_probs)
    blended = {
        runner: weight * float(model_probs.get(runner, 0.0))
        + (1.0 - weight) * float(market_probs.get(runner, 0.0))
        for runner in runners
    }
    total = sum(max(0.0, value) for value in blended.values())
    if total <= 0:
        return {}
    return {runner: max(0.0, value) / total for runner, value in blended.items()}
