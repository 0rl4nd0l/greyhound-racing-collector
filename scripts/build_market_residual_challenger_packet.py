#!/usr/bin/env python3
"""Build a report-only market residual challenger packet.

This consumes the dog-level market residual runner matrix emitted by
build_rolling_model_comparison_packet.py. It tests a predeclared conditional
grid that anchors to market probabilities and only allows raw Stage 2 to move
probability when the market favourite is weak.

It writes artifacts only. It does not train a production model, promote,
mutate registries, update pointers, write DB labels/odds, emit EV, place bets,
rewrite snapshots/manifests, or enable TGR.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from scripts.join_forward_shadow_results import logistic_calibration_review  # noqa: E402


DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = (
    "artifacts/full_evidence_orchestration_20260525/"
    "market_residual_challenger_"
)
REPORT_FILE = "market_residual_challenger_report.json"
SUMMARY_FILE = "SUMMARY.md"
FOLD_SUMMARY_CSV = "cross_validated_fold_summary.csv"
RACE_PREDICTIONS_CSV = "cross_validated_race_predictions.csv"
FINAL_READY = "MARKET_RESIDUAL_CHALLENGER_REVIEW_READY"
FINAL_COLLECTING = "MARKET_RESIDUAL_CHALLENGER_COLLECTING"
MIN_RACES_FOR_REVIEW = 100
DEFAULT_FOLDS = 5
DEFAULT_MIN_TRAIN_RACES = 50
GRID_MARKET_WEIGHTS = (0.5, 0.75, 0.9, 0.95)
GRID_MARKET_FAVOURITE_THRESHOLDS = (1.5, 2.0, 4.0, 8.0)
NO_WRITE_GUARANTEES = {
    "training_production_model": False,
    "production_promotion": False,
    "registry_mutation": False,
    "production_pointer_update": False,
    "active_model_replacement": False,
    "model_artifact_overwrite": False,
    "db_write": False,
    "label_write": False,
    "odds_write": False,
    "betting_or_ev_action": False,
    "snapshot_rewrite": False,
    "manifest_rewrite": False,
    "tgr_enabled": False,
}


def now_id(now: datetime | None = None) -> str:
    return (now or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return os.path.relpath(path.resolve(), ROOT.resolve())
    except ValueError:
        return str(path)


def assert_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative = logical.absolute().relative_to(ROOT.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_market_residual_challenger:{relative}")
    return logical.absolute()


def unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = Path(f"{base}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"output_dir_collision_exhausted:{base}")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_manifest(output_dir: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        files[relpath(path) or str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "schema_version": "market_residual_challenger_output_manifest_v1",
        "output_dir": relpath(output_dir),
        "files": files,
    }


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def finite_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return None


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_scores(raw_scores: Sequence[float | None]) -> list[float] | None:
    if not raw_scores or any(score is None for score in raw_scores):
        return None
    scores = [float(score) for score in raw_scores if score is not None]
    if any(score < 0 or not math.isfinite(score) for score in scores):
        return None
    total = sum(scores)
    if total <= 0:
        return None
    return [score / total for score in scores]


def candidate_key(weight: float, threshold: float) -> str:
    weight_key = str(int(round(weight * 100)))
    threshold_key = str(threshold).replace(".", "_")
    return f"market_anchor_stage2_uncalibrated_blend{weight_key}_fav_gt_{threshold_key}"


def load_matrix(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def collect_races(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    skipped = Counter()
    for row in rows:
        race_id = str(row.get("race_id") or "").strip()
        if not race_id:
            skipped["row_missing_race_id"] += 1
            continue
        grouped.setdefault(race_id, []).append(dict(row))

    races: list[dict[str, Any]] = []
    for race_id, race_rows in grouped.items():
        winner_count = sum(1 for row in race_rows if parse_bool(row.get("is_winner")))
        if winner_count != 1:
            skipped["race_winner_count_not_one"] += 1
            continue
        required_scores = (
            "market_probability",
            "candidate_probability",
            "stage2_shadow_uncalibrated_probability_norm",
        )
        if any(
            any(finite_float(row.get(key)) is None for row in race_rows)
            for key in required_scores
        ):
            skipped["race_missing_required_probability"] += 1
            continue
        race_rows = sorted(
            race_rows,
            key=lambda row: (
                finite_int(row.get("box_number")) or 999,
                str(row.get("dog_name") or ""),
            ),
        )
        first = race_rows[0]
        races.append(
            {
                "race_id": race_id,
                "race_date": first.get("race_date"),
                "venue": first.get("venue"),
                "race_number": finite_int(first.get("race_number")),
                "rows": race_rows,
            }
        )
    races.sort(
        key=lambda race: (
            str(race.get("race_date") or ""),
            str(race.get("venue") or ""),
            finite_int(race.get("race_number")) or 999,
            str(race.get("race_id") or ""),
        )
    )
    return races, {
        "input_rows": len(rows),
        "input_races": len(grouped),
        "accepted_races": len(races),
        "skipped_counts": dict(sorted(skipped.items())),
    }


def base_scores(rows: Sequence[Mapping[str, Any]], key: str) -> list[float] | None:
    return normalize_scores([finite_float(row.get(key)) for row in rows])


def market_scores(rows: Sequence[Mapping[str, Any]]) -> list[float] | None:
    return base_scores(rows, "market_probability")


def current_non_market_scores(rows: Sequence[Mapping[str, Any]]) -> list[float] | None:
    return base_scores(rows, "candidate_probability")


def stage2_uncalibrated_scores(rows: Sequence[Mapping[str, Any]]) -> list[float] | None:
    return base_scores(rows, "stage2_shadow_uncalibrated_probability_norm")


def race_market_favourite_odds(rows: Sequence[Mapping[str, Any]]) -> float | None:
    order_scores = market_scores(rows)
    if order_scores is None:
        return None
    order = ranking_order(rows, order_scores)
    if not order:
        return None
    return finite_float(rows[order[0]].get("odds_decimal"))


def conditional_residual_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    market_weight: float,
    favourite_threshold: float,
) -> list[float] | None:
    market = market_scores(rows)
    stage2 = stage2_uncalibrated_scores(rows)
    if market is None or stage2 is None:
        return None
    favourite_odds = race_market_favourite_odds(rows)
    if favourite_odds is None:
        return None
    if favourite_odds <= favourite_threshold:
        return market
    return normalize_scores(
        [
            (market_weight * market_score) + ((1.0 - market_weight) * stage2_score)
            for market_score, stage2_score in zip(market, stage2, strict=True)
        ]
    )


def candidate_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "candidate_key": "current_best_non_market_from_matrix",
            "family": "current_best_non_market",
            "score_function": current_non_market_scores,
        }
    ]
    for weight in GRID_MARKET_WEIGHTS:
        for threshold in GRID_MARKET_FAVOURITE_THRESHOLDS:
            specs.append(
                {
                    "candidate_key": candidate_key(weight, threshold),
                    "family": "conditional_market_residual_grid",
                    "market_weight": weight,
                    "market_favourite_odds_gt": threshold,
                    "score_function": (
                        lambda rows,
                        weight=weight,
                        threshold=threshold: conditional_residual_scores(
                            rows,
                            market_weight=weight,
                            favourite_threshold=threshold,
                        )
                    ),
                }
            )
    return specs


def ranking_order(rows: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> list[int]:
    return sorted(
        range(len(rows)),
        key=lambda index: (
            -scores[index],
            finite_int(rows[index].get("box_number")) or 999,
            str(rows[index].get("dog_name") or ""),
        ),
    )


def evaluate_scored_races(scored_races: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rank_hits_top1 = 0
    rank_hits_top3 = 0
    winner_ranks: list[int] = []
    brier_values: list[float] = []
    logloss_values: list[float] = []
    probability_sum_errors: list[float] = []
    box1_top_picks = 0
    calibration_labels: list[int] = []
    calibration_probabilities: list[float] = []
    skipped = Counter()

    for scored in scored_races:
        race = scored.get("race")
        scores = scored.get("scores")
        if not isinstance(race, Mapping) or not isinstance(scores, list):
            skipped["invalid_scored_race"] += 1
            continue
        rows = list(race.get("rows") or [])
        if len(scores) != len(rows):
            skipped["score_length_mismatch"] += 1
            continue
        winner_indexes = [
            index for index, row in enumerate(rows) if parse_bool(row.get("is_winner"))
        ]
        if len(winner_indexes) != 1:
            skipped["race_winner_count_not_one"] += 1
            continue
        winner_index = winner_indexes[0]
        order = ranking_order(rows, scores)
        winner_rank = order.index(winner_index) + 1
        winner_ranks.append(winner_rank)
        rank_hits_top1 += int(winner_rank == 1)
        rank_hits_top3 += int(winner_rank <= 3)
        brier_values.append(
            sum(
                (scores[index] - (1.0 if index == winner_index else 0.0)) ** 2
                for index in range(len(rows))
            )
        )
        logloss_values.append(-math.log(max(scores[winner_index], 1e-15)))
        probability_sum_errors.append(abs(sum(scores) - 1.0))
        top_pick = rows[order[0]]
        box1_top_picks += int(finite_int(top_pick.get("box_number")) == 1)
        for index, _row in enumerate(rows):
            calibration_labels.append(1 if index == winner_index else 0)
            calibration_probabilities.append(scores[index])

    race_count = len(winner_ranks)
    if race_count == 0:
        return {
            "status": "NO_EVALUABLE_RACES",
            "race_count": 0,
            "blockers": ["no_evaluable_races"],
            "skipped_counts": dict(sorted(skipped.items())),
        }
    return {
        "status": "EVALUATED",
        "race_count": race_count,
        "top1": rank_hits_top1 / race_count,
        "top3": rank_hits_top3 / race_count,
        "mean_winner_rank": sum(winner_ranks) / race_count,
        "brier": sum(brier_values) / race_count,
        "logloss": sum(logloss_values) / race_count,
        "box1_top_pick_share": box1_top_picks / race_count,
        "probability_sum_max_error": max(probability_sum_errors),
        "calibration_slope_intercept": logistic_calibration_review(
            calibration_labels,
            calibration_probabilities,
        ),
        "skipped_counts": dict(sorted(skipped.items())),
        "blockers": [],
    }


def score_races(
    races: Sequence[Mapping[str, Any]],
    score_function: Callable[[Sequence[Mapping[str, Any]]], list[float] | None],
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for race in races:
        rows = list(race.get("rows") or [])
        scores = score_function(rows)
        if scores is None or len(scores) != len(rows):
            continue
        scored.append({"race": race, "scores": scores})
    return scored


def evaluate_candidate(
    races: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    score_function: Callable[[Sequence[Mapping[str, Any]]], list[float] | None] = spec[
        "score_function"
    ]
    metrics = evaluate_scored_races(score_races(races, score_function))
    metrics.update(
        {
            "candidate_key": spec.get("candidate_key"),
            "family": spec.get("family"),
            "market_weight": spec.get("market_weight"),
            "market_favourite_odds_gt": spec.get("market_favourite_odds_gt"),
        }
    )
    return metrics


def candidate_sort_key(metrics: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        finite_float(metrics.get("top1")) or -1.0,
        finite_float(metrics.get("top3")) or -1.0,
        -(finite_float(metrics.get("mean_winner_rank")) or 999.0),
        -(finite_float(metrics.get("brier")) or 999.0),
        -(finite_float(metrics.get("logloss")) or 999.0),
    )


def metric_delta(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    key: str,
) -> float | None:
    left = finite_float(baseline.get(key))
    right = finite_float(candidate.get(key))
    if left is None or right is None:
        return None
    return right - left


def metric_deltas(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, float | None]:
    return {
        key: metric_delta(baseline, candidate, key)
        for key in (
            "top1",
            "top3",
            "mean_winner_rank",
            "brier",
            "logloss",
            "box1_top_pick_share",
        )
    }


def split_folds(races: Sequence[Mapping[str, Any]], fold_count: int) -> list[list[Mapping[str, Any]]]:
    fold_count = max(1, min(fold_count, len(races)))
    folds: list[list[Mapping[str, Any]]] = [[] for _ in range(fold_count)]
    for index, race in enumerate(races):
        folds[index % fold_count].append(race)
    return folds


def cross_validated_residual_review(
    races: Sequence[Mapping[str, Any]],
    *,
    fold_count: int,
    min_train_races: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    specs = candidate_specs()
    folds = split_folds(races, fold_count)
    fold_summaries: list[dict[str, Any]] = []
    scored_test_races: list[dict[str, Any]] = []
    race_prediction_rows: list[dict[str, Any]] = []

    for fold_index, test_races in enumerate(folds):
        test_ids = {race.get("race_id") for race in test_races}
        train_races = [race for race in races if race.get("race_id") not in test_ids]
        if len(train_races) < min_train_races:
            fold_summaries.append(
                {
                    "fold": fold_index,
                    "status": "SKIPPED_TRAIN_RACE_COUNT_BELOW_MINIMUM",
                    "train_races": len(train_races),
                    "test_races": len(test_races),
                    "selected_candidate_key": None,
                }
            )
            continue
        train_metrics = [evaluate_candidate(train_races, spec) for spec in specs]
        evaluated_train = [
            item for item in train_metrics if item.get("status") == "EVALUATED"
        ]
        if not evaluated_train:
            fold_summaries.append(
                {
                    "fold": fold_index,
                    "status": "SKIPPED_NO_EVALUATED_TRAIN_CANDIDATES",
                    "train_races": len(train_races),
                    "test_races": len(test_races),
                    "selected_candidate_key": None,
                }
            )
            continue
        selected_metrics = max(evaluated_train, key=candidate_sort_key)
        selected_key = str(selected_metrics.get("candidate_key"))
        selected_spec = next(
            spec for spec in specs if str(spec.get("candidate_key")) == selected_key
        )
        test_scored = score_races(test_races, selected_spec["score_function"])
        scored_test_races.extend(
            {
                "race": item["race"],
                "scores": item["scores"],
                "fold": fold_index,
                "selected_candidate_key": selected_key,
            }
            for item in test_scored
        )
        fold_test_metrics = evaluate_scored_races(test_scored)
        fold_summaries.append(
            {
                "fold": fold_index,
                "status": "EVALUATED",
                "train_races": len(train_races),
                "test_races": len(test_races),
                "selected_candidate_key": selected_key,
                "selected_family": selected_spec.get("family"),
                "selected_market_weight": selected_spec.get("market_weight"),
                "selected_market_favourite_odds_gt": selected_spec.get(
                    "market_favourite_odds_gt"
                ),
                "train_top1": selected_metrics.get("top1"),
                "train_top3": selected_metrics.get("top3"),
                "train_logloss": selected_metrics.get("logloss"),
                "test_top1": fold_test_metrics.get("top1"),
                "test_top3": fold_test_metrics.get("top3"),
                "test_logloss": fold_test_metrics.get("logloss"),
            }
        )
        race_prediction_rows.extend(
            race_prediction_rows_for_fold(
                test_scored,
                fold_index=fold_index,
                selected_candidate_key=selected_key,
            )
        )

    metrics = evaluate_scored_races(scored_test_races)
    metrics["candidate_key"] = "cross_validated_market_residual_challenger"
    metrics["family"] = "train_fold_selected_conditional_market_residual"
    metrics["fold_count"] = fold_count
    metrics["evaluated_fold_count"] = sum(
        1 for item in fold_summaries if item.get("status") == "EVALUATED"
    )
    return metrics, fold_summaries, race_prediction_rows


def race_prediction_rows_for_fold(
    scored_races: Sequence[Mapping[str, Any]],
    *,
    fold_index: int,
    selected_candidate_key: str,
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for scored in scored_races:
        race = scored["race"]
        rows = list(race.get("rows") or [])
        scores = list(scored["scores"])
        market = market_scores(rows)
        if market is None:
            continue
        winner_indexes = [
            index for index, row in enumerate(rows) if parse_bool(row.get("is_winner"))
        ]
        if len(winner_indexes) != 1:
            continue
        winner_index = winner_indexes[0]
        challenger_order = ranking_order(rows, scores)
        market_order = ranking_order(rows, market)
        challenger_rank = challenger_order.index(winner_index) + 1
        market_rank = market_order.index(winner_index) + 1
        challenger_logloss = -math.log(max(scores[winner_index], 1e-15))
        market_logloss = -math.log(max(market[winner_index], 1e-15))
        winner = rows[winner_index]
        rows_out.append(
            {
                "fold": fold_index,
                "selected_candidate_key": selected_candidate_key,
                "race_id": race.get("race_id"),
                "race_date": race.get("race_date"),
                "venue": race.get("venue"),
                "race_number": race.get("race_number"),
                "winner_dog_name": winner.get("dog_name"),
                "winner_box_number": finite_int(winner.get("box_number")),
                "winner_odds_decimal": finite_float(winner.get("odds_decimal")),
                "challenger_winner_rank": challenger_rank,
                "market_winner_rank": market_rank,
                "challenger_winner_probability": scores[winner_index],
                "market_winner_probability": market[winner_index],
                "challenger_logloss": challenger_logloss,
                "market_logloss": market_logloss,
                "challenger_minus_market_logloss": challenger_logloss - market_logloss,
            }
        )
    return rows_out


def promotion_gate(
    *,
    market_metrics: Mapping[str, Any],
    challenger_metrics: Mapping[str, Any],
    min_races_for_review: int,
) -> dict[str, Any]:
    deltas = metric_deltas(market_metrics, challenger_metrics)
    blockers = ["report_only_residual_challenger_not_promotion_eligible"]
    if (challenger_metrics.get("race_count") or 0) < min_races_for_review:
        blockers.append("sample_race_count_below_review_floor")
    if (deltas.get("top1") or 0.0) <= 0:
        blockers.append("top1_not_above_market")
    if (deltas.get("top3") or 0.0) < 0:
        blockers.append("top3_below_market")
    if (deltas.get("mean_winner_rank") or 999.0) > 0:
        blockers.append("mean_winner_rank_worse_than_market")
    if (deltas.get("brier") or 999.0) > 0:
        blockers.append("brier_worse_than_market")
    if (deltas.get("logloss") or 999.0) > 0:
        blockers.append("logloss_worse_than_market")
    return {
        "promotion_ready": False,
        "would_clear_metric_gates": blockers == [
            "report_only_residual_challenger_not_promotion_eligible"
        ],
        "candidate_minus_market": deltas,
        "blockers": blockers,
    }


def write_fold_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "fold",
        "status",
        "train_races",
        "test_races",
        "selected_candidate_key",
        "selected_family",
        "selected_market_weight",
        "selected_market_favourite_odds_gt",
        "train_top1",
        "train_top3",
        "train_logloss",
        "test_top1",
        "test_top3",
        "test_logloss",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_race_predictions_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "fold",
        "selected_candidate_key",
        "race_id",
        "race_date",
        "venue",
        "race_number",
        "winner_dog_name",
        "winner_box_number",
        "winner_odds_decimal",
        "challenger_winner_rank",
        "market_winner_rank",
        "challenger_winner_probability",
        "market_winner_probability",
        "challenger_logloss",
        "market_logloss",
        "challenger_minus_market_logloss",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def summary_markdown(report: Mapping[str, Any]) -> str:
    gate = report.get("promotion_gate") or {}
    return "\n".join(
        [
            "# Market Residual Challenger Review",
            "",
            f"Final status: `{report.get('final_status')}`",
            "",
            f"- Matrix rows: `{report.get('matrix_row_count')}`",
            f"- Accepted races: `{report.get('accepted_race_count')}` / `{report.get('minimum_races_for_review')}`",
            f"- Fold count: `{report.get('fold_count')}`",
            f"- Evaluated folds: `{report.get('evaluated_fold_count')}`",
            f"- Market top1: `{(report.get('market_metrics') or {}).get('top1')}`",
            f"- Challenger top1: `{(report.get('challenger_metrics') or {}).get('top1')}`",
            f"- Candidate minus market: `{gate.get('candidate_minus_market')}`",
            f"- Promotion ready: `{gate.get('promotion_ready')}`",
            f"- Promotion blockers: `{gate.get('blockers')}`",
            "",
            "No production training, promotion, registry mutation, pointer update, DB write, label write, odds write, betting/EV action, snapshot rewrite, manifest rewrite, or TGR enablement was performed.",
            "",
        ]
    )


def build_packet(
    *,
    runner_matrix_csv: Path,
    output_dir: Path,
    fold_count: int = DEFAULT_FOLDS,
    min_train_races: int = DEFAULT_MIN_TRAIN_RACES,
    min_races_for_review: int = MIN_RACES_FOR_REVIEW,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    output_dir = unique_dir(assert_output_dir_safe(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)

    matrix_rows = load_matrix(runner_matrix_csv)
    races, collection = collect_races(matrix_rows)
    market_metrics = evaluate_candidate(
        races,
        {
            "candidate_key": "market_only_implied",
            "family": "market_only",
            "score_function": market_scores,
        },
    )
    current_non_market_metrics = evaluate_candidate(
        races,
        {
            "candidate_key": "current_best_non_market_from_matrix",
            "family": "current_best_non_market",
            "score_function": current_non_market_scores,
        },
    )
    challenger_metrics, fold_rows, prediction_rows = cross_validated_residual_review(
        races,
        fold_count=fold_count,
        min_train_races=min_train_races,
    )
    gate = promotion_gate(
        market_metrics=market_metrics,
        challenger_metrics=challenger_metrics,
        min_races_for_review=min_races_for_review,
    )
    blockers: list[str] = []
    if len(races) < min_races_for_review:
        blockers.append("accepted_race_count_below_review_floor")
    if challenger_metrics.get("status") != "EVALUATED":
        blockers.append("cross_validated_challenger_not_evaluated")

    report = {
        "schema_version": "market_residual_challenger_report_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": FINAL_READY if not blockers else FINAL_COLLECTING,
        "output_dir": relpath(output_dir),
        "runner_matrix_csv": relpath(runner_matrix_csv),
        "fold_summary_csv": relpath(output_dir / FOLD_SUMMARY_CSV),
        "race_predictions_csv": relpath(output_dir / RACE_PREDICTIONS_CSV),
        "matrix_row_count": len(matrix_rows),
        "accepted_race_count": len(races),
        "minimum_races_for_review": min_races_for_review,
        "fold_count": fold_count,
        "min_train_races": min_train_races,
        "evaluated_fold_count": challenger_metrics.get("evaluated_fold_count"),
        "collection": collection,
        "candidate_grid": [
            {
                "candidate_key": spec.get("candidate_key"),
                "family": spec.get("family"),
                "market_weight": spec.get("market_weight"),
                "market_favourite_odds_gt": spec.get("market_favourite_odds_gt"),
            }
            for spec in candidate_specs()
        ],
        "market_metrics": market_metrics,
        "current_non_market_metrics": current_non_market_metrics,
        "challenger_metrics": challenger_metrics,
        "promotion_gate": gate,
        "blockers": blockers,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }
    write_json(output_dir / REPORT_FILE, report)
    write_fold_summary_csv(output_dir / FOLD_SUMMARY_CSV, fold_rows)
    write_race_predictions_csv(output_dir / RACE_PREDICTIONS_CSV, prediction_rows)
    write_text(output_dir / SUMMARY_FILE, summary_markdown(report))
    write_text(output_dir / "final_status.txt", str(report["final_status"]) + "\n")
    write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-matrix-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fold-count", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--min-train-races", type=int, default=DEFAULT_MIN_TRAIN_RACES)
    parser.add_argument("--min-races-for-review", type=int, default=MIN_RACES_FOR_REVIEW)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    generated_at = datetime.now().astimezone()
    output_dir = (
        args.output_dir
        or DEFAULT_EVIDENCE_ROOT / f"market_residual_challenger_{now_id(generated_at)}"
    )
    report = build_packet(
        runner_matrix_csv=args.runner_matrix_csv,
        output_dir=output_dir,
        fold_count=args.fold_count,
        min_train_races=args.min_train_races,
        min_races_for_review=args.min_races_for_review,
        generated_at=generated_at,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
