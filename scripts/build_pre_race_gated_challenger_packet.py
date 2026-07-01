#!/usr/bin/env python3
"""Build a report-only pre-race gated challenger packet.

This consumes the runner-level market residual matrix from the rolling model
comparison packet. It tests a predeclared grid of pre-race gates that only
allows non-market scores to move market probabilities in selected race regimes.

It writes artifacts only. It does not train a production model, promote, mutate
registries, update pointers, write DB labels/odds, emit EV, place bets, rewrite
snapshots/manifests, or enable TGR.
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

from scripts.build_market_residual_challenger_packet import (  # noqa: E402
    collect_races,
    current_non_market_scores,
    evaluate_candidate,
    evaluate_scored_races,
    finite_float,
    finite_int,
    load_matrix,
    market_scores,
    metric_deltas,
    normalize_scores,
    parse_bool,
    ranking_order,
    score_races,
    split_folds,
    stage2_uncalibrated_scores,
)
from utils.report_output_dir_guard import assert_prefixed_report_output_dir  # noqa: E402


DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = (
    "artifacts/full_evidence_orchestration_20260525/"
    "pre_race_gated_challenger_"
)
OUTPUT_ARTIFACT_PREFIX = "pre_race_gated_challenger_"
REPORT_FILE = "pre_race_gated_challenger_report.json"
FOLD_SUMMARY_CSV = "cross_validated_fold_summary.csv"
RACE_PREDICTIONS_CSV = "cross_validated_race_predictions.csv"
CANDIDATE_METRICS_CSV = "candidate_metrics.csv"
RANK_FIRST_HYPOTHESIS_METRICS_CSV = "rank_first_hypothesis_candidate_metrics.csv"
SUMMARY_FILE = "SUMMARY.md"
FINAL_READY = "PRE_RACE_GATED_CHALLENGER_REVIEW_READY"
FINAL_COLLECTING = "PRE_RACE_GATED_CHALLENGER_COLLECTING"
MIN_RACES_FOR_REVIEW = 100
DEFAULT_FOLDS = 5
DEFAULT_MIN_TRAIN_RACES = 50
GRID_RUNNER_COUNTS = (4, 5, 7, 8)
GRID_MARKET_FAVOURITE_GT = (2.0, 4.0)
GRID_MARKET_WEIGHTS = (0.5, 0.75, 0.9)
PREDECLARED_RESIDUAL_CANDIDATE_KEY = (
    "market_favourite_gt_4_0__raw_stage2_market_blend_75"
)
PREDECLARED_RESIDUAL_TRIGGER_FLOOR = 10
RANK_FIRST_HYPOTHESIS_TRIGGER_FLOOR = 10
SUPPORTED_RANK_FIRST_HYPOTHESIS_DIMENSIONS = {
    "venue",
    "runner_count",
    "market_favourite_odds_band",
    "market_favourite_odds_group",
    "stage2_uncalibrated_agrees_market_top",
    "current_candidate_agrees_market_top",
}
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


def assert_output_dir_safe(
    output_dir: Path,
    *,
    evidence_root: Path | None = None,
) -> Path:
    return assert_prefixed_report_output_dir(
        output_dir,
        repo_root=ROOT,
        repo_prefix=OUTPUT_PREFIX,
        artifact_prefix=OUTPUT_ARTIFACT_PREFIX,
        prefix_error="output_dir_must_be_pre_race_gated_challenger",
        evidence_root=evidence_root,
    )


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
        "schema_version": "pre_race_gated_challenger_output_manifest_v1",
        "output_dir": relpath(output_dir),
        "files": files,
    }


def top_pick(rows: Sequence[Mapping[str, Any]], rank_key: str, flag_key: str | None = None) -> Mapping[str, Any] | None:
    if flag_key:
        flagged = [row for row in rows if parse_bool(row.get(flag_key))]
        if flagged:
            return sorted(
                flagged,
                key=lambda row: (
                    finite_int(row.get("box_number")) or 999,
                    str(row.get("dog_name") or ""),
                ),
            )[0]
    ranked = [row for row in rows if finite_int(row.get(rank_key)) is not None]
    if not ranked:
        return None
    return sorted(
        ranked,
        key=lambda row: (
            finite_int(row.get(rank_key)) or 999,
            finite_int(row.get("box_number")) or 999,
            str(row.get("dog_name") or ""),
        ),
    )[0]


def market_favourite_odds(rows: Sequence[Mapping[str, Any]]) -> float | None:
    direct = finite_float((rows[0] if rows else {}).get("market_favourite_odds_decimal"))
    if direct is not None:
        return direct
    market_top = top_pick(rows, "market_rank", "market_top_pick")
    if market_top is None:
        return None
    return finite_float(market_top.get("odds_decimal"))


def market_favourite_odds_group(rows: Sequence[Mapping[str, Any]]) -> str:
    odds = market_favourite_odds(rows)
    if odds is None:
        return "market_favourite_missing"
    if odds <= 2.0:
        return "market_favourite_lte_2"
    if odds <= 4.0:
        return "market_favourite_2_4"
    if odds <= 8.0:
        return "market_favourite_4_8"
    return "market_favourite_gt_8"


def runner_count(rows: Sequence[Mapping[str, Any]]) -> int:
    direct = finite_int((rows[0] if rows else {}).get("runner_count"))
    return direct or len(rows)


def top_box(rows: Sequence[Mapping[str, Any]], rank_key: str, flag_key: str | None = None) -> int | None:
    pick = top_pick(rows, rank_key, flag_key)
    if pick is None:
        return None
    return finite_int(pick.get("box_number"))


def current_agrees_market_top(rows: Sequence[Mapping[str, Any]]) -> bool:
    market_box = top_box(rows, "market_rank", "market_top_pick")
    current_box = top_box(rows, "candidate_rank", "candidate_top_pick")
    return market_box is not None and market_box == current_box


def stage2_uncalibrated_agrees_market_top(rows: Sequence[Mapping[str, Any]]) -> bool:
    market_box = top_box(rows, "market_rank", "market_top_pick")
    stage2_box = top_box(rows, "stage2_shadow_uncalibrated_rank")
    return market_box is not None and market_box == stage2_box


def safe_key_part(value: Any) -> str:
    text = str(value).strip().lower()
    return "".join(char if char.isalnum() else "_" for char in text).strip("_") or "blank"


def race_dimension_value(rows: Sequence[Mapping[str, Any]], dimension: str) -> Any:
    first = rows[0] if rows else {}
    if dimension == "runner_count":
        return runner_count(rows)
    if dimension == "market_favourite_odds_group":
        return str(first.get("market_favourite_odds_group") or market_favourite_odds_group(rows))
    if dimension == "market_favourite_odds_band":
        return str(first.get("market_favourite_odds_band") or market_favourite_odds_group(rows))
    if dimension == "stage2_uncalibrated_agrees_market_top":
        return stage2_uncalibrated_agrees_market_top(rows)
    if dimension == "current_candidate_agrees_market_top":
        return current_agrees_market_top(rows)
    return first.get(dimension)


def dimension_value_matches(actual: Any, expected: Any, dimension: str) -> bool:
    if dimension == "runner_count":
        actual_int = finite_int(actual)
        expected_int = finite_int(expected)
        return actual_int is not None and actual_int == expected_int
    if isinstance(actual, bool) or isinstance(expected, bool):
        return parse_bool(actual) == parse_bool(expected)
    return str(actual).strip() == str(expected).strip()


def gate_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "gate_key": "stage2_uncalibrated_agrees_market_top",
            "gate_family": "top_pick_agreement",
            "description": "Activate when raw Stage 2 and market have the same top pick.",
            "gate_function": stage2_uncalibrated_agrees_market_top,
        },
        {
            "gate_key": "current_candidate_agrees_market_top",
            "gate_family": "top_pick_agreement",
            "description": "Activate when current candidate and market have the same top pick.",
            "gate_function": current_agrees_market_top,
        },
    ]
    for count in GRID_RUNNER_COUNTS:
        specs.append(
            {
                "gate_key": f"runner_count_eq_{count}",
                "gate_family": "runner_count",
                "runner_count": count,
                "description": f"Activate when runner_count == {count}.",
                "gate_function": lambda rows, count=count: runner_count(rows) == count,
            }
        )
    specs.append(
        {
            "gate_key": "runner_count_in_4_5_7_8",
            "gate_family": "runner_count",
            "runner_count_set": list(GRID_RUNNER_COUNTS),
            "description": "Activate when runner_count is one of 4, 5, 7, 8.",
            "gate_function": lambda rows: runner_count(rows) in set(GRID_RUNNER_COUNTS),
        }
    )
    for threshold in GRID_MARKET_FAVOURITE_GT:
        specs.append(
            {
                "gate_key": f"market_favourite_gt_{str(threshold).replace('.', '_')}",
                "gate_family": "market_favourite_odds",
                "market_favourite_odds_gt": threshold,
                "description": f"Activate when market favourite odds > {threshold}.",
                "gate_function": (
                    lambda rows, threshold=threshold: (
                        (market_favourite_odds(rows) or 0.0) > threshold
                    )
                ),
            }
        )
    return specs


def blended_stage2_scores(rows: Sequence[Mapping[str, Any]], *, market_weight: float) -> list[float] | None:
    market = market_scores(rows)
    stage2 = stage2_uncalibrated_scores(rows)
    if market is None or stage2 is None:
        return None
    return normalize_scores(
        [
            (market_weight * market_score) + ((1.0 - market_weight) * stage2_score)
            for market_score, stage2_score in zip(market, stage2, strict=True)
        ]
    )


def score_mode_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "score_mode": "current_candidate",
            "market_weight": None,
            "score_function": current_non_market_scores,
        },
        {
            "score_mode": "raw_stage2_uncalibrated",
            "market_weight": None,
            "score_function": stage2_uncalibrated_scores,
        },
    ]
    for weight in GRID_MARKET_WEIGHTS:
        specs.append(
            {
                "score_mode": f"raw_stage2_market_blend_{int(round(weight * 100))}",
                "market_weight": weight,
                "score_function": (
                    lambda rows, weight=weight: blended_stage2_scores(
                        rows,
                        market_weight=weight,
                    )
                ),
            }
        )
    return specs


def gated_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    gate_function: Callable[[Sequence[Mapping[str, Any]]], bool],
    score_function: Callable[[Sequence[Mapping[str, Any]]], list[float] | None],
) -> list[float] | None:
    market = market_scores(rows)
    if market is None:
        return None
    if not gate_function(rows):
        return market
    scores = score_function(rows)
    return scores if scores is not None else market


def candidate_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for gate in gate_specs():
        for score in score_mode_specs():
            key = f"{gate['gate_key']}__{score['score_mode']}"
            specs.append(
                {
                    "candidate_key": key,
                    "family": "pre_race_gated_non_market",
                    "gate_key": gate["gate_key"],
                    "gate_family": gate["gate_family"],
                    "gate_description": gate["description"],
                    "score_mode": score["score_mode"],
                    "market_weight": score["market_weight"],
                    "runner_count": gate.get("runner_count"),
                    "runner_count_set": gate.get("runner_count_set"),
                    "market_favourite_odds_gt": gate.get("market_favourite_odds_gt"),
                    "score_function": (
                        lambda rows,
                        gate_function=gate["gate_function"],
                        score_function=score["score_function"]: gated_scores(
                            rows,
                            gate_function=gate_function,
                            score_function=score_function,
                        )
                    ),
                    "gate_function": gate["gate_function"],
                }
            )
    return specs


def selection_sort_key(metrics: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        finite_float(metrics.get("top1")) or -1.0,
        finite_float(metrics.get("top3")) or -1.0,
        -(finite_float(metrics.get("mean_winner_rank")) or 999.0),
        -(finite_float(metrics.get("brier")) or 999.0),
        -(finite_float(metrics.get("logloss")) or 999.0),
    )


def gate_triggered_count(
    races: Sequence[Mapping[str, Any]],
    gate_function: Callable[[Sequence[Mapping[str, Any]]], bool],
) -> int:
    return sum(1 for race in races if gate_function(list(race.get("rows") or [])))


def evaluate_spec(races: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]) -> dict[str, Any]:
    metrics = evaluate_candidate(races, spec)
    metrics.update(
        {
            "gate_key": spec.get("gate_key"),
            "gate_family": spec.get("gate_family"),
            "gate_description": spec.get("gate_description"),
            "score_mode": spec.get("score_mode"),
            "market_weight": spec.get("market_weight"),
            "runner_count": spec.get("runner_count"),
            "runner_count_set": spec.get("runner_count_set"),
            "market_favourite_odds_gt": spec.get("market_favourite_odds_gt"),
            "hypothesis_dimension": spec.get("hypothesis_dimension"),
            "hypothesis_dimension_value": spec.get("hypothesis_dimension_value"),
            "hypothesis_source_race_count": spec.get("hypothesis_source_race_count"),
            "gate_triggered_race_count": gate_triggered_count(
                races,
                spec["gate_function"],
            ),
        }
    )
    return metrics


def load_rank_first_hypothesis_gate_specs(path: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path is None:
        return [], {
            "schema_version": "rank_first_hypothesis_source_v1",
            "source_path": None,
            "source_status": "NOT_PROVIDED",
            "loaded_hypothesis_count": 0,
            "supported_hypothesis_count": 0,
            "unsupported_hypotheses": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    regimes = payload.get("pre_race_rank_first_help_regimes") or []
    specs: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    for index, regime in enumerate(regimes):
        dimension = str(regime.get("dimension") or "")
        dimension_value = regime.get("dimension_value")
        if dimension not in SUPPORTED_RANK_FIRST_HYPOTHESIS_DIMENSIONS:
            unsupported.append(
                {
                    "index": index,
                    "dimension": dimension,
                    "dimension_value": dimension_value,
                    "reason": "unsupported_or_not_pre_race_gate_dimension",
                }
            )
            continue
        gate_key = (
            f"rank_first_hypothesis_{safe_key_part(dimension)}_"
            f"eq_{safe_key_part(dimension_value)}"
        )
        specs.append(
            {
                "candidate_key": f"{gate_key}__raw_stage2_uncalibrated",
                "family": "rank_first_hypothesis_report_only",
                "gate_key": gate_key,
                "gate_family": "rank_first_hypothesis",
                "gate_description": (
                    "Report-only rank-first hypothesis gate from residual regime "
                    f"audit: {dimension} == {dimension_value}."
                ),
                "score_mode": "raw_stage2_uncalibrated",
                "market_weight": None,
                "hypothesis_dimension": dimension,
                "hypothesis_dimension_value": dimension_value,
                "hypothesis_source_race_count": regime.get("race_count"),
                "hypothesis_source": dict(regime),
                "score_function": (
                    lambda rows,
                    dimension=dimension,
                    dimension_value=dimension_value: gated_scores(
                        rows,
                        gate_function=(
                            lambda gate_rows,
                            dimension=dimension,
                            dimension_value=dimension_value: dimension_value_matches(
                                race_dimension_value(gate_rows, dimension),
                                dimension_value,
                                dimension,
                            )
                        ),
                        score_function=stage2_uncalibrated_scores,
                    )
                ),
                "gate_function": (
                    lambda rows,
                    dimension=dimension,
                    dimension_value=dimension_value: dimension_value_matches(
                        race_dimension_value(rows, dimension),
                        dimension_value,
                        dimension,
                    )
                ),
            }
        )
    return specs, {
        "schema_version": "rank_first_hypothesis_source_v1",
        "source_path": relpath(path),
        "source_status": "LOADED",
        "loaded_hypothesis_count": len(regimes),
        "supported_hypothesis_count": len(specs),
        "unsupported_hypotheses": unsupported,
    }


def rank_first_hypothesis_gate_review(
    *,
    races: Sequence[Mapping[str, Any]],
    market_metrics: Mapping[str, Any],
    hypotheses_path: Path | None,
    trigger_floor: int = RANK_FIRST_HYPOTHESIS_TRIGGER_FLOOR,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    specs, source = load_rank_first_hypothesis_gate_specs(hypotheses_path)
    blockers = [
        "rank_first_hypothesis_review_report_only",
        "requires_fresh_future_out_of_sample_packet",
    ]
    if hypotheses_path is None:
        blockers.append("rank_first_hypotheses_json_not_provided")
        return {
            "schema_version": "rank_first_hypothesis_gate_review_v1",
            "status": "RANK_FIRST_HYPOTHESIS_REVIEW_NOT_PROVIDED",
            "promotion_eligible": False,
            "source": source,
            "candidate_count": 0,
            "evaluated_candidate_count": 0,
            "best_candidate_key": None,
            "minimum_triggered_races_for_directional_read": trigger_floor,
            "best_candidate": {},
            "best_candidate_minus_market": {},
            "directional_read_ready": False,
            "blockers": blockers,
        }, []
    if not specs:
        blockers.append("no_supported_rank_first_hypotheses")
        return {
            "schema_version": "rank_first_hypothesis_gate_review_v1",
            "status": "RANK_FIRST_HYPOTHESIS_REVIEW_NO_SUPPORTED_HYPOTHESES",
            "promotion_eligible": False,
            "source": source,
            "candidate_count": 0,
            "evaluated_candidate_count": 0,
            "best_candidate_key": None,
            "minimum_triggered_races_for_directional_read": trigger_floor,
            "best_candidate": {},
            "best_candidate_minus_market": {},
            "directional_read_ready": False,
            "blockers": blockers,
        }, []

    candidate_metrics: list[dict[str, Any]] = []
    for spec in specs:
        metrics = evaluate_spec(races, spec)
        metrics["candidate_minus_market"] = metric_deltas(market_metrics, metrics)
        candidate_metrics.append(metrics)
    evaluated = [item for item in candidate_metrics if item.get("status") == "EVALUATED"]
    best = max(evaluated, key=selection_sort_key, default={})
    best_deltas = dict(best.get("candidate_minus_market") or {})
    best_trigger_count = int(best.get("gate_triggered_race_count") or 0)
    if not best:
        blockers.append("no_evaluated_rank_first_hypothesis_candidates")
    if best and best_trigger_count < trigger_floor:
        blockers.append("best_triggered_race_count_below_directional_floor")
    if best and (best_deltas.get("top1") or 0.0) <= 0:
        blockers.append("best_top1_not_above_market")
    if best and (best_deltas.get("top3") or 0.0) < 0:
        blockers.append("best_top3_below_market")
    if best and (best_deltas.get("mean_winner_rank") or 999.0) > 0:
        blockers.append("best_mean_winner_rank_worse_than_market")
    if best and (best_deltas.get("brier") or 999.0) > 0:
        blockers.append("best_brier_worse_than_market")
    if best and (best_deltas.get("logloss") or 999.0) > 0:
        blockers.append("best_logloss_worse_than_market")
    directional_blockers = [
        item
        for item in blockers
        if item
        not in {
            "rank_first_hypothesis_review_report_only",
            "requires_fresh_future_out_of_sample_packet",
        }
    ]
    return {
        "schema_version": "rank_first_hypothesis_gate_review_v1",
        "status": (
            "RANK_FIRST_HYPOTHESIS_REVIEW_READY"
            if evaluated
            else "RANK_FIRST_HYPOTHESIS_REVIEW_COLLECTING"
        ),
        "promotion_eligible": False,
        "source": source,
        "candidate_count": len(specs),
        "evaluated_candidate_count": len(evaluated),
        "best_candidate_key": best.get("candidate_key"),
        "minimum_triggered_races_for_directional_read": trigger_floor,
        "best_candidate": dict(best),
        "best_candidate_minus_market": best_deltas,
        "directional_read_ready": bool(best and not directional_blockers),
        "blockers": blockers,
    }, candidate_metrics


def race_prediction_rows_for_fold(
    scored_races: Sequence[Mapping[str, Any]],
    *,
    fold_index: int,
    selected_spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    gate_function: Callable[[Sequence[Mapping[str, Any]]], bool] = selected_spec[
        "gate_function"
    ]
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
                "selected_candidate_key": selected_spec.get("candidate_key"),
                "gate_key": selected_spec.get("gate_key"),
                "gate_family": selected_spec.get("gate_family"),
                "score_mode": selected_spec.get("score_mode"),
                "gate_triggered": bool(gate_function(rows)),
                "race_id": race.get("race_id"),
                "race_date": race.get("race_date"),
                "venue": race.get("venue"),
                "race_number": race.get("race_number"),
                "runner_count": runner_count(rows),
                "market_favourite_odds_decimal": market_favourite_odds(rows),
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


def cross_validated_gated_review(
    races: Sequence[Mapping[str, Any]],
    *,
    fold_count: int,
    min_train_races: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    specs = candidate_specs()
    folds = split_folds(races, fold_count)
    fold_summaries: list[dict[str, Any]] = []
    scored_test_races: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

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
                }
            )
            continue
        train_metrics = [evaluate_spec(train_races, spec) for spec in specs]
        evaluated = [item for item in train_metrics if item.get("status") == "EVALUATED"]
        if not evaluated:
            fold_summaries.append(
                {
                    "fold": fold_index,
                    "status": "SKIPPED_NO_EVALUATED_TRAIN_CANDIDATES",
                    "train_races": len(train_races),
                    "test_races": len(test_races),
                }
            )
            continue
        selected = max(evaluated, key=selection_sort_key)
        selected_spec = next(
            spec for spec in specs if spec.get("candidate_key") == selected.get("candidate_key")
        )
        test_scored = score_races(test_races, selected_spec["score_function"])
        scored_test_races.extend(
            {
                "race": item["race"],
                "scores": item["scores"],
                "fold": fold_index,
                "selected_candidate_key": selected_spec.get("candidate_key"),
            }
            for item in test_scored
        )
        test_metrics = evaluate_scored_races(test_scored)
        fold_summaries.append(
            {
                "fold": fold_index,
                "status": "EVALUATED",
                "train_races": len(train_races),
                "test_races": len(test_races),
                "selected_candidate_key": selected_spec.get("candidate_key"),
                "gate_key": selected_spec.get("gate_key"),
                "gate_family": selected_spec.get("gate_family"),
                "score_mode": selected_spec.get("score_mode"),
                "market_weight": selected_spec.get("market_weight"),
                "train_gate_triggered_race_count": selected.get("gate_triggered_race_count"),
                "test_gate_triggered_race_count": gate_triggered_count(
                    test_races,
                    selected_spec["gate_function"],
                ),
                "train_top1": selected.get("top1"),
                "train_top3": selected.get("top3"),
                "train_mean_winner_rank": selected.get("mean_winner_rank"),
                "train_brier": selected.get("brier"),
                "train_logloss": selected.get("logloss"),
                "test_top1": test_metrics.get("top1"),
                "test_top3": test_metrics.get("top3"),
                "test_mean_winner_rank": test_metrics.get("mean_winner_rank"),
                "test_brier": test_metrics.get("brier"),
                "test_logloss": test_metrics.get("logloss"),
            }
        )
        prediction_rows.extend(
            race_prediction_rows_for_fold(
                test_scored,
                fold_index=fold_index,
                selected_spec=selected_spec,
            )
        )

    metrics = evaluate_scored_races(scored_test_races)
    metrics["candidate_key"] = "cross_validated_pre_race_gated_challenger"
    metrics["family"] = "train_fold_selected_pre_race_gate"
    metrics["fold_count"] = fold_count
    metrics["evaluated_fold_count"] = sum(
        1 for item in fold_summaries if item.get("status") == "EVALUATED"
    )
    metrics["gate_triggered_test_race_count"] = sum(
        1 for row in prediction_rows if row.get("gate_triggered") is True
    )
    return metrics, fold_summaries, prediction_rows


def promotion_gate(
    *,
    market_metrics: Mapping[str, Any],
    challenger_metrics: Mapping[str, Any],
    min_races_for_review: int,
) -> dict[str, Any]:
    deltas = metric_deltas(market_metrics, challenger_metrics)
    blockers = [
        "report_only_pre_race_gated_challenger_not_promotion_eligible",
        "requires_fresh_future_out_of_sample_packet",
    ]
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
        "would_clear_metric_gates": blockers
        == [
            "report_only_pre_race_gated_challenger_not_promotion_eligible",
            "requires_fresh_future_out_of_sample_packet",
        ],
        "candidate_minus_market": deltas,
        "blockers": blockers,
    }


def predeclared_residual_candidate_review(
    *,
    market_metrics: Mapping[str, Any],
    candidate_metrics: Sequence[Mapping[str, Any]],
    candidate_key: str = PREDECLARED_RESIDUAL_CANDIDATE_KEY,
    trigger_floor: int = PREDECLARED_RESIDUAL_TRIGGER_FLOOR,
) -> dict[str, Any]:
    candidate = next(
        (item for item in candidate_metrics if item.get("candidate_key") == candidate_key),
        None,
    )
    blockers: list[str] = ["predeclared_residual_candidate_report_only"]
    if candidate is None:
        return {
            "schema_version": "predeclared_residual_candidate_review_v1",
            "candidate_key": candidate_key,
            "status": "PREDECLARED_RESIDUAL_CANDIDATE_MISSING",
            "promotion_eligible": False,
            "triggered_race_count": 0,
            "minimum_triggered_races_for_directional_read": trigger_floor,
            "candidate_minus_market": {},
            "blockers": blockers + ["predeclared_residual_candidate_missing"],
        }

    trigger_count = int(candidate.get("gate_triggered_race_count") or 0)
    deltas = metric_deltas(market_metrics, candidate)
    if candidate.get("status") != "EVALUATED":
        blockers.append("predeclared_residual_candidate_not_evaluated")
    if trigger_count < trigger_floor:
        blockers.append("triggered_race_count_below_directional_floor")
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

    metric_blockers = [
        item
        for item in blockers
        if item
        not in {
            "predeclared_residual_candidate_report_only",
            "triggered_race_count_below_directional_floor",
        }
    ]
    return {
        "schema_version": "predeclared_residual_candidate_review_v1",
        "candidate_key": candidate_key,
        "status": (
            "PREDECLARED_RESIDUAL_CANDIDATE_REVIEWABLE"
            if trigger_count >= trigger_floor
            else "PREDECLARED_RESIDUAL_CANDIDATE_COLLECTING"
        ),
        "promotion_eligible": False,
        "directional_read_ready": (
            candidate.get("status") == "EVALUATED"
            and trigger_count >= trigger_floor
            and not metric_blockers
        ),
        "triggered_race_count": trigger_count,
        "minimum_triggered_races_for_directional_read": trigger_floor,
        "gate_key": candidate.get("gate_key"),
        "score_mode": candidate.get("score_mode"),
        "market_weight": candidate.get("market_weight"),
        "market_favourite_odds_gt": candidate.get("market_favourite_odds_gt"),
        "metrics": dict(candidate),
        "candidate_minus_market": deltas,
        "blockers": blockers,
    }


def write_fold_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "fold",
        "status",
        "train_races",
        "test_races",
        "selected_candidate_key",
        "gate_key",
        "gate_family",
        "score_mode",
        "market_weight",
        "train_gate_triggered_race_count",
        "test_gate_triggered_race_count",
        "train_top1",
        "train_top3",
        "train_mean_winner_rank",
        "train_brier",
        "train_logloss",
        "test_top1",
        "test_top3",
        "test_mean_winner_rank",
        "test_brier",
        "test_logloss",
    ]
    write_csv(path, rows, fields)


def write_race_predictions_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "fold",
        "selected_candidate_key",
        "gate_key",
        "gate_family",
        "score_mode",
        "gate_triggered",
        "race_id",
        "race_date",
        "venue",
        "race_number",
        "runner_count",
        "market_favourite_odds_decimal",
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
    write_csv(path, rows, fields)


def write_candidate_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "candidate_key",
        "gate_key",
        "gate_family",
        "score_mode",
        "market_weight",
        "runner_count",
        "runner_count_set",
        "market_favourite_odds_gt",
        "gate_triggered_race_count",
        "status",
        "race_count",
        "top1",
        "top3",
        "mean_winner_rank",
        "brier",
        "logloss",
        "box1_top_pick_share",
    ]
    write_csv(path, rows, fields)


def write_rank_first_hypothesis_metrics_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    fields = [
        "candidate_key",
        "hypothesis_dimension",
        "hypothesis_dimension_value",
        "hypothesis_source_race_count",
        "gate_key",
        "score_mode",
        "gate_triggered_race_count",
        "status",
        "race_count",
        "top1",
        "top3",
        "mean_winner_rank",
        "brier",
        "logloss",
        "box1_top_pick_share",
        "candidate_minus_market",
    ]
    write_csv(path, rows, fields)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def summary_markdown(report: Mapping[str, Any]) -> str:
    gate = report.get("promotion_gate") or {}
    rank_first = report.get("rank_first_hypothesis_gate_review") or {}
    return "\n".join(
        [
            "# Pre-Race Gated Challenger Review",
            "",
            f"Final status: `{report.get('final_status')}`",
            "",
            f"- Matrix rows: `{report.get('matrix_row_count')}`",
            f"- Accepted races: `{report.get('accepted_race_count')}` / `{report.get('minimum_races_for_review')}`",
            f"- Candidate grid count: `{report.get('candidate_grid_count')}`",
            f"- Evaluated folds: `{report.get('evaluated_fold_count')}`",
            f"- Predeclared residual candidate: `{(report.get('predeclared_residual_candidate') or {}).get('candidate_key')}`",
            f"- Predeclared residual triggered races: `{(report.get('predeclared_residual_candidate') or {}).get('triggered_race_count')}` / `{(report.get('predeclared_residual_candidate') or {}).get('minimum_triggered_races_for_directional_read')}`",
            f"- Rank-first hypothesis review: `{rank_first.get('status')}`",
            f"- Rank-first hypothesis candidates: `{rank_first.get('evaluated_candidate_count')}` / `{rank_first.get('candidate_count')}`",
            f"- Rank-first best candidate: `{rank_first.get('best_candidate_key')}`",
            f"- Rank-first best triggered races: `{(rank_first.get('best_candidate') or {}).get('gate_triggered_race_count')}` / `{rank_first.get('minimum_triggered_races_for_directional_read')}`",
            f"- Rank-first best candidate minus market: `{rank_first.get('best_candidate_minus_market')}`",
            f"- Market top1: `{(report.get('market_metrics') or {}).get('top1')}`",
            f"- Gated challenger top1: `{(report.get('challenger_metrics') or {}).get('top1')}`",
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
    evidence_root: Path | None = None,
    fold_count: int = DEFAULT_FOLDS,
    min_train_races: int = DEFAULT_MIN_TRAIN_RACES,
    min_races_for_review: int = MIN_RACES_FOR_REVIEW,
    rank_first_hypotheses_json: Path | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    output_dir = unique_dir(assert_output_dir_safe(output_dir, evidence_root=evidence_root))
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
    current_metrics = evaluate_candidate(
        races,
        {
            "candidate_key": "current_best_non_market_from_matrix",
            "family": "current_best_non_market",
            "score_function": current_non_market_scores,
        },
    )
    full_candidate_metrics = [evaluate_spec(races, spec) for spec in candidate_specs()]
    predeclared_residual_candidate = predeclared_residual_candidate_review(
        market_metrics=market_metrics,
        candidate_metrics=full_candidate_metrics,
    )
    rank_first_review, rank_first_candidate_metrics = rank_first_hypothesis_gate_review(
        races=races,
        market_metrics=market_metrics,
        hypotheses_path=rank_first_hypotheses_json,
    )
    best_full_sample_candidate = max(
        [item for item in full_candidate_metrics if item.get("status") == "EVALUATED"],
        key=selection_sort_key,
        default={},
    )
    challenger_metrics, fold_rows, prediction_rows = cross_validated_gated_review(
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
        "schema_version": "pre_race_gated_challenger_report_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": FINAL_READY if not blockers else FINAL_COLLECTING,
        "output_dir": relpath(output_dir),
        "runner_matrix_csv": relpath(runner_matrix_csv),
        "fold_summary_csv": relpath(output_dir / FOLD_SUMMARY_CSV),
        "race_predictions_csv": relpath(output_dir / RACE_PREDICTIONS_CSV),
        "candidate_metrics_csv": relpath(output_dir / CANDIDATE_METRICS_CSV),
        "rank_first_hypothesis_metrics_csv": relpath(
            output_dir / RANK_FIRST_HYPOTHESIS_METRICS_CSV
        ),
        "matrix_row_count": len(matrix_rows),
        "accepted_race_count": len(races),
        "minimum_races_for_review": min_races_for_review,
        "fold_count": fold_count,
        "min_train_races": min_train_races,
        "candidate_grid_count": len(candidate_specs()),
        "evaluated_fold_count": challenger_metrics.get("evaluated_fold_count"),
        "collection": collection,
        "market_metrics": market_metrics,
        "current_non_market_metrics": current_metrics,
        "best_full_sample_gated_candidate": best_full_sample_candidate,
        "predeclared_residual_candidate": predeclared_residual_candidate,
        "rank_first_hypothesis_gate_review": rank_first_review,
        "challenger_metrics": challenger_metrics,
        "promotion_gate": gate,
        "blockers": blockers,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }
    write_json(output_dir / REPORT_FILE, report)
    write_fold_summary_csv(output_dir / FOLD_SUMMARY_CSV, fold_rows)
    write_race_predictions_csv(output_dir / RACE_PREDICTIONS_CSV, prediction_rows)
    write_candidate_metrics_csv(output_dir / CANDIDATE_METRICS_CSV, full_candidate_metrics)
    write_rank_first_hypothesis_metrics_csv(
        output_dir / RANK_FIRST_HYPOTHESIS_METRICS_CSV,
        rank_first_candidate_metrics,
    )
    write_text(output_dir / SUMMARY_FILE, summary_markdown(report))
    write_text(output_dir / "final_status.txt", str(report["final_status"]) + "\n")
    write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-matrix-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--fold-count", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--min-train-races", type=int, default=DEFAULT_MIN_TRAIN_RACES)
    parser.add_argument("--min-races-for-review", type=int, default=MIN_RACES_FOR_REVIEW)
    parser.add_argument(
        "--rank-first-hypotheses-json",
        "--next-hypotheses-json",
        dest="rank_first_hypotheses_json",
        type=Path,
        help="Optional next_hypotheses.json from the market residual regime audit.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    generated_at = datetime.now().astimezone()
    output_dir = (
        args.output_dir
        or args.evidence_root / f"pre_race_gated_challenger_{now_id(generated_at)}"
    )
    report = build_packet(
        runner_matrix_csv=args.runner_matrix_csv,
        output_dir=output_dir,
        evidence_root=args.evidence_root,
        fold_count=args.fold_count,
        min_train_races=args.min_train_races,
        min_races_for_review=args.min_races_for_review,
        rank_first_hypotheses_json=args.rank_first_hypotheses_json,
        generated_at=generated_at,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
