#!/usr/bin/env python3
"""Run a report-only challenger evaluation on the cumulative runner matrix.

This script consumes the adapter packet produced by
``build_cumulative_runner_matrix_challenger_packet.py``. It reads existing
rolling artifacts only and emits report-local evidence. It does not train,
write databases, capture odds/results, mutate registries, promote models, or
place bets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SCHEMA_VERSION = "cumulative_runner_matrix_challenger_evaluation_v1"

NO_WRITE_GUARANTEES = {
    "live_db_write": False,
    "official_result_capture": False,
    "live_odds_capture": False,
    "model_fit": False,
    "model_artifact_write": False,
    "registry_mutation": False,
    "promotion": False,
    "betting": False,
    "tgr_enabled": False,
}

PROTECTED_OUTPUT_PREFIXES = (
    "artifacts/prediction_snapshots",
    "model_registry",
    "docs/model_registry",
    "ml_models_v4",
    "advanced_models",
)

BASE_PROBABILITY_COLUMNS = {
    "market_only_implied": "market_probability",
    "primary_shadow": "primary_shadow_probability_norm",
    "stage2_shadow": "stage2_shadow_probability_norm",
    "stage2_shadow_uncalibrated": "stage2_shadow_uncalibrated_probability_norm",
}

BASE_FAMILIES = {
    "market_only_implied": "market_only",
    "primary_shadow": "primary_shadow",
    "stage2_shadow": "stage2_shadow",
    "stage2_shadow_uncalibrated": "stage2_raw_rf",
}

REQUESTED_ALIAS_KEYS = {
    "stage2_uncalibrated": "stage2_shadow_uncalibrated",
}

POWER_PATTERN = re.compile(
    r"^(primary_shadow|stage2_shadow|stage2_shadow_uncalibrated)"
    r"_power_gamma_([0-9]+(?:_[0-9]+)?)$"
)
BLEND_PATTERN = re.compile(
    r"^(stage2|stage2_uncalibrated)_market_blend_([0-9]+)$"
)

REPORT_ONLY_BLOCKERS = (
    "report_only_no_training_challenger_not_promotion_eligible",
    "requires_fresh_future_out_of_sample_packet",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_not_object:{path}")
    return payload


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y"}


def _race_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        race_id = str(row.get("race_id") or "")
        if race_id:
            grouped[race_id].append(dict(row))
    return dict(grouped)


def _assert_output_dir_safe(output_dir: Path, repo_root: Path | None = None) -> Path:
    resolved = output_dir.resolve()
    repo_root = (repo_root or Path.cwd()).resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError:
        return resolved

    relative_text = relative.as_posix()
    for prefix in PROTECTED_OUTPUT_PREFIXES:
        if relative_text == prefix or relative_text.startswith(prefix + "/"):
            raise ValueError(f"output_dir_protected:{prefix}")
    return resolved


def _winner_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    winners = [
        row
        for row in rows
        if _safe_int(row.get("finish_position")) == 1 or _truthy(row.get("is_winner"))
    ]
    if len(winners) != 1:
        return None
    return winners[0]


def _normalize_scores(scores: Sequence[float | None]) -> list[float | None]:
    if any(score is None or score < 0 for score in scores):
        return [None for _ in scores]
    total = sum(score for score in scores if score is not None)
    if total <= 0:
        return [None for _ in scores]
    return [score / total if score is not None else None for score in scores]


def _column_scores(rows: Sequence[Mapping[str, Any]], column: str) -> list[float | None]:
    return [_safe_float(row.get(column)) for row in rows]


def _power_scores(
    rows: Sequence[Mapping[str, Any]],
    column: str,
    gamma: float,
) -> list[float | None]:
    raw_scores = _column_scores(rows, column)
    powered = [
        score**gamma if score is not None and score >= 0 else None
        for score in raw_scores
    ]
    return _normalize_scores(powered)


def _blend_scores(
    rows: Sequence[Mapping[str, Any]],
    model_column: str,
    market_weight: float,
) -> list[float | None]:
    market_scores = _column_scores(rows, "market_probability")
    model_scores = _column_scores(rows, model_column)
    blended: list[float | None] = []
    model_weight = 1.0 - market_weight
    for market_score, model_score in zip(market_scores, model_scores):
        if market_score is None or model_score is None:
            blended.append(None)
        else:
            blended.append((market_weight * market_score) + (model_weight * model_score))
    return _normalize_scores(blended)


def _candidate_definition(
    candidate_key: str,
) -> tuple[str, Callable[[Sequence[Mapping[str, Any]]], list[float | None]]] | None:
    if candidate_key in BASE_PROBABILITY_COLUMNS:
        column = BASE_PROBABILITY_COLUMNS[candidate_key]
        family = BASE_FAMILIES[candidate_key]
        return family, lambda rows, column=column: _normalize_scores(
            _column_scores(rows, column)
        )

    power_match = POWER_PATTERN.match(candidate_key)
    if power_match:
        base_key = power_match.group(1)
        gamma = float(power_match.group(2).replace("_", "."))
        column = BASE_PROBABILITY_COLUMNS[base_key]
        family = f"{BASE_FAMILIES[base_key]}_power"
        return family, lambda rows, column=column, gamma=gamma: _power_scores(
            rows, column, gamma
        )

    blend_match = BLEND_PATTERN.match(candidate_key)
    if blend_match:
        model_key = blend_match.group(1)
        market_weight = int(blend_match.group(2)) / 100.0
        if model_key == "stage2":
            column = "stage2_shadow_probability_norm"
            family = "stage2_market_blend"
        else:
            column = "stage2_shadow_uncalibrated_probability_norm"
            family = "stage2_raw_rf_odds_augmented_blend"
        return family, lambda rows, column=column, market_weight=market_weight: (
            _blend_scores(rows, column, market_weight)
        )

    return None


def _ranked_rows(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[float | None],
) -> list[tuple[int, Mapping[str, Any], float]]:
    scored = [
        (index, row, score)
        for index, (row, score) in enumerate(zip(rows, scores))
        if score is not None
    ]
    scored.sort(
        key=lambda item: (
            -item[2],
            _safe_int(item[1].get("box_number")) or 999,
            str(item[1].get("dog_name") or ""),
            item[0],
        )
    )
    return [(rank, row, score) for rank, (_, row, score) in enumerate(scored, start=1)]


def _evaluate_candidate(
    *,
    candidate_key: str,
    family: str,
    grouped_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    score_fn: Callable[[Sequence[Mapping[str, Any]]], list[float | None]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    race_count = 0
    top1 = 0
    top3 = 0
    winner_rank_sum = 0.0
    brier_sum = 0.0
    logloss_sum = 0.0
    box1_top_pick_count = 0
    probability_sum_max_error = 0.0
    skipped: Counter[str] = Counter()
    per_race_rows: list[dict[str, Any]] = []

    for race_id in sorted(grouped_rows):
        race_rows = list(grouped_rows[race_id])
        winner = _winner_row(race_rows)
        if winner is None:
            skipped["missing_or_ambiguous_winner"] += 1
            continue

        scores = score_fn(race_rows)
        if len(scores) != len(race_rows) or any(score is None for score in scores):
            skipped["missing_candidate_probability"] += 1
            continue

        probability_sum = sum(score for score in scores if score is not None)
        if probability_sum <= 0:
            skipped["zero_probability_sum"] += 1
            continue
        probability_sum_max_error = max(probability_sum_max_error, abs(1.0 - probability_sum))

        ranked = _ranked_rows(race_rows, scores)
        if len(ranked) != len(race_rows):
            skipped["ranked_runner_count_mismatch"] += 1
            continue

        winner_index = next(
            (
                index
                for index, row in enumerate(race_rows)
                if row is winner
            ),
            None,
        )
        if winner_index is None:
            skipped["winner_index_missing"] += 1
            continue

        winner_rank = next(rank for rank, row, _ in ranked if row is winner)
        winner_probability = scores[winner_index] or 0.0
        top_pick = ranked[0][1]
        top_pick_score = ranked[0][2]
        box1_top_pick_count += 1 if _safe_int(top_pick.get("box_number")) == 1 else 0
        race_count += 1
        top1 += 1 if winner_rank == 1 else 0
        top3 += 1 if winner_rank <= 3 else 0
        winner_rank_sum += winner_rank
        brier_sum += sum(
            (score - (1.0 if row is winner else 0.0)) ** 2
            for row, score in zip(race_rows, scores)
            if score is not None
        )
        logloss_sum += -math.log(max(winner_probability, 1e-15))

        per_race_rows.append(
            {
                "candidate_key": candidate_key,
                "race_id": race_id,
                "race_date": winner.get("race_date"),
                "venue": winner.get("venue"),
                "race_number": winner.get("race_number"),
                "runner_count": len(race_rows),
                "winner_dog": winner.get("dog_name"),
                "winner_box": winner.get("box_number"),
                "winner_probability": winner_probability,
                "winner_rank": winner_rank,
                "top_pick_dog": top_pick.get("dog_name"),
                "top_pick_box": top_pick.get("box_number"),
                "top_pick_probability": top_pick_score,
            }
        )

    if race_count == 0:
        metrics = {
            "candidate_key": candidate_key,
            "family": family,
            "status": "DATA_MISSING",
            "blockers": ["no_evaluable_races"],
            "race_count": 0,
            "skipped_race_counts": dict(sorted(skipped.items())),
        }
        return metrics, per_race_rows

    metrics = {
        "candidate_key": candidate_key,
        "family": family,
        "status": "EVALUATED",
        "blockers": [],
        "race_count": race_count,
        "top1": top1 / race_count,
        "top3": top3 / race_count,
        "mean_winner_rank": winner_rank_sum / race_count,
        "brier": brier_sum / race_count,
        "logloss": logloss_sum / race_count,
        "box1_top_pick_share": box1_top_pick_count / race_count,
        "probability_sum_max_error_joined_races": probability_sum_max_error,
        "skipped_race_counts": dict(sorted(skipped.items())),
    }
    return metrics, per_race_rows


def _candidate_minus(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in (
        "top1",
        "top3",
        "mean_winner_rank",
        "brier",
        "logloss",
        "box1_top_pick_share",
    ):
        candidate_value = _safe_float(candidate.get(key))
        baseline_value = _safe_float(baseline.get(key))
        output[key] = (
            candidate_value - baseline_value
            if candidate_value is not None and baseline_value is not None
            else None
        )
    return output


def _metric_sort_key(metrics: Mapping[str, Any]) -> tuple[float, float, float, float, float, str]:
    return (
        -float(metrics.get("top1") or 0.0),
        -float(metrics.get("top3") or 0.0),
        float(metrics.get("mean_winner_rank") or 999.0),
        float(metrics.get("brier") or 999.0),
        float(metrics.get("logloss") or 999.0),
        str(metrics.get("candidate_key") or ""),
    )


def _market_safety_gate(
    *,
    candidate: Mapping[str, Any],
    market: Mapping[str, Any],
    minimum_races_for_review: int,
) -> dict[str, Any]:
    delta = _candidate_minus(candidate, market)
    candidate_race_count = int(candidate.get("race_count") or 0)
    blockers = list(REPORT_ONLY_BLOCKERS)
    if candidate_race_count < minimum_races_for_review:
        blockers.append("sample_race_count_below_review_floor")
    if (delta.get("top1") or 0.0) <= 0.0:
        blockers.append("top1_not_above_market")
    if (delta.get("top3") or 0.0) < 0.0:
        blockers.append("top3_below_market")
    if (delta.get("mean_winner_rank") or 0.0) > 0.0:
        blockers.append("mean_winner_rank_worse_than_market")
    if (delta.get("brier") or 0.0) > 0.0:
        blockers.append("brier_worse_than_market")
    if (delta.get("logloss") or 0.0) > 0.0:
        blockers.append("logloss_worse_than_market")

    metric_gate_blockers = [
        blocker for blocker in blockers if blocker not in REPORT_ONLY_BLOCKERS
    ]
    return {
        "promotion_ready": False,
        "would_clear_metric_gates": not metric_gate_blockers,
        "minimum_races_for_review": minimum_races_for_review,
        "candidate_minus_market": delta,
        "blockers": blockers,
    }


def _comparison_rows(
    candidate_metrics: Mapping[str, Mapping[str, Any]],
    market_key: str,
) -> list[dict[str, Any]]:
    market = candidate_metrics[market_key]
    rows = []
    for key, metrics in sorted(candidate_metrics.items(), key=lambda item: _metric_sort_key(item[1])):
        delta = _candidate_minus(metrics, market)
        rows.append(
            {
                "candidate_key": key,
                "family": metrics.get("family"),
                "status": metrics.get("status"),
                "race_count": metrics.get("race_count"),
                "top1": metrics.get("top1"),
                "top1_minus_market": delta.get("top1"),
                "top3": metrics.get("top3"),
                "top3_minus_market": delta.get("top3"),
                "mean_winner_rank": metrics.get("mean_winner_rank"),
                "mean_winner_rank_minus_market": delta.get("mean_winner_rank"),
                "brier": metrics.get("brier"),
                "brier_minus_market": delta.get("brier"),
                "logloss": metrics.get("logloss"),
                "logloss_minus_market": delta.get("logloss"),
                "box1_top_pick_share": metrics.get("box1_top_pick_share"),
            }
        )
    return rows


def _declared_candidate_keys(rolling_report: Mapping[str, Any]) -> list[str]:
    candidates = rolling_report.get("candidate_metrics_by_key")
    if not isinstance(candidates, dict):
        return []
    return sorted(str(key) for key in candidates)


def _score_candidates(
    declared_keys: Sequence[str],
    grouped_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, str],
    list[str],
    dict[str, list[dict[str, Any]]],
]:
    scored: dict[str, dict[str, Any]] = {}
    unsupported: list[str] = []
    aliases = {
        alias: target
        for alias, target in REQUESTED_ALIAS_KEYS.items()
        if target in declared_keys
    }
    per_race_by_candidate: dict[str, list[dict[str, Any]]] = {}

    for candidate_key in declared_keys:
        definition = _candidate_definition(candidate_key)
        if definition is None:
            unsupported.append(candidate_key)
            continue
        family, score_fn = definition
        metrics, per_race = _evaluate_candidate(
            candidate_key=candidate_key,
            family=family,
            grouped_rows=grouped_rows,
            score_fn=score_fn,
        )
        scored[candidate_key] = metrics
        per_race_by_candidate[candidate_key] = per_race

    return scored, aliases, unsupported, per_race_by_candidate


def _load_race_ids_from_tsv(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle, delimiter="\t")]
    return {str(row.get("race_id") or "") for row in rows if row.get("race_id")}


def _old_input_guard(
    *,
    current_race_ids: set[str],
    old_result_path: Path | None,
) -> dict[str, Any]:
    if old_result_path is None:
        return {
            "status": "NOT_PROVIDED",
            "reuse_for_current_market_safety": False,
            "note": "No recovered-input challenger result was supplied.",
        }
    if not old_result_path.exists():
        return {
            "status": "DATA_MISSING",
            "reuse_for_current_market_safety": False,
            "old_result_path": str(old_result_path),
            "blockers": ["old_result_path_missing"],
        }

    old_race_ids = _load_race_ids_from_tsv(old_result_path)
    overlap = sorted(current_race_ids & old_race_ids)
    if not overlap:
        return {
            "status": "OLD_INPUT_DIAGNOSTIC",
            "reuse_for_current_market_safety": False,
            "old_result_path": str(old_result_path),
            "old_race_count": len(old_race_ids),
            "current_race_count": len(current_race_ids),
            "overlap_race_count": 0,
            "note": (
                "Recovered-input challenger results have zero race overlap with "
                "the current cumulative runner matrix and are diagnostic only."
            ),
        }

    return {
        "status": "OVERLAP_PRESENT_REVIEW_REQUIRED",
        "reuse_for_current_market_safety": False,
        "old_result_path": str(old_result_path),
        "old_race_count": len(old_race_ids),
        "current_race_count": len(current_race_ids),
        "overlap_race_count": len(overlap),
        "overlap_examples": overlap[:10],
        "blockers": ["old_result_overlap_requires_manual_lineage_review"],
    }


def _market_favourite_odds(rows: Sequence[Mapping[str, Any]]) -> float | None:
    odds_values = [
        _safe_float(row.get("odds_decimal"))
        for row in rows
        if _safe_float(row.get("odds_decimal")) is not None
    ]
    return min(odds_values) if odds_values else None


def _slice_grouped_rows(
    grouped_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    predicate: Callable[[Sequence[Mapping[str, Any]]], bool],
) -> dict[str, Sequence[Mapping[str, Any]]]:
    return {
        race_id: rows
        for race_id, rows in grouped_rows.items()
        if predicate(rows)
    }


def _residual_slice_diagnostics(
    *,
    grouped_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_key: str,
    candidate_definition: tuple[str, Callable[[Sequence[Mapping[str, Any]]], list[float | None]]],
    market_definition: tuple[str, Callable[[Sequence[Mapping[str, Any]]], list[float | None]]],
) -> list[dict[str, Any]]:
    slices = [
        (
            "market_favourite_odds_gt_2",
            lambda rows: (_market_favourite_odds(rows) or 0.0) > 2.0,
        ),
        (
            "market_favourite_odds_gt_3",
            lambda rows: (_market_favourite_odds(rows) or 0.0) > 3.0,
        ),
        (
            "market_favourite_odds_gt_4",
            lambda rows: (_market_favourite_odds(rows) or 0.0) > 4.0,
        ),
    ]
    diagnostics: list[dict[str, Any]] = []
    minimum_slice_races = 10
    for slice_key, predicate in slices:
        sliced = _slice_grouped_rows(grouped_rows, predicate)
        candidate_family, candidate_score_fn = candidate_definition
        market_family, market_score_fn = market_definition
        candidate_metrics, _ = _evaluate_candidate(
            candidate_key=candidate_key,
            family=candidate_family,
            grouped_rows=sliced,
            score_fn=candidate_score_fn,
        )
        market_metrics, _ = _evaluate_candidate(
            candidate_key="market_only_implied",
            family=market_family,
            grouped_rows=sliced,
            score_fn=market_score_fn,
        )
        race_count = int(candidate_metrics.get("race_count") or 0)
        blockers = []
        if race_count < minimum_slice_races:
            blockers.append("slice_race_count_below_directional_floor")
        diagnostics.append(
            {
                "schema_version": "cumulative_runner_matrix_residual_slice_v1",
                "status": "DIAGNOSTIC_ONLY",
                "slice_key": slice_key,
                "selected_candidate_key": candidate_key,
                "race_count": race_count,
                "minimum_slice_races_for_directional_read": minimum_slice_races,
                "blockers": blockers,
                "candidate_minus_market": _candidate_minus(
                    candidate_metrics, market_metrics
                ),
                "no_ev_or_betting_claim": True,
            }
        )
    return diagnostics


def _readme_text(report: Mapping[str, Any]) -> str:
    gate = report["market_safety_rank_first_gate"]
    best = report["best_non_market_candidate"]
    market = report["market_metrics"]
    delta = gate["candidate_minus_market"]
    blockers = "\n".join(f"- `{blocker}`" for blocker in gate["blockers"])
    return f"""# Cumulative Runner Matrix Challenger Evaluation

Generated: {report["generated_at"]}

## Status

- Final status: `{report["final_status"]}`
- Market candidate: `{report["market_candidate_key"]}`
- Best non-market candidate: `{best["candidate_key"]}`
- Current cumulative races: `{market["race_count"]}`
- Promotion ready: `false`

## Market-Safety Gate

Candidate minus market:

- top1: `{delta["top1"]}`
- top3: `{delta["top3"]}`
- mean winner rank: `{delta["mean_winner_rank"]}`
- brier: `{delta["brier"]}`
- logloss: `{delta["logloss"]}`

Blockers:

{blockers}

## Old Input Guard

- Status: `{report["old_recovered_input_guard"]["status"]}`
- Reuse for current market-safety: `{report["old_recovered_input_guard"]["reuse_for_current_market_safety"]}`

## Boundaries

This evaluator is report-only. It did not train, promote, mutate registries,
write live DBs, capture official results, capture live odds, enable TGR, or
perform EV/betting actions.
"""


def run_evaluation(
    *,
    adapter_packet_path: Path,
    output_dir: Path,
    old_result_per_race_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    output_dir = _assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_packet = _load_json(adapter_packet_path)
    if adapter_packet.get("schema_version") != "cumulative_runner_matrix_challenger_packet_v1":
        raise ValueError("adapter_packet_schema_mismatch")
    if adapter_packet.get("status") != "READY_FOR_REPORT_ONLY_CHALLENGER":
        raise ValueError("adapter_packet_not_ready_for_report_only_challenger")
    if adapter_packet.get("input_surface") != "current_cumulative_rolling_runner_matrix":
        raise ValueError("adapter_packet_not_current_cumulative_surface")

    paths = adapter_packet.get("paths") or {}
    rolling_report_path = Path(str(paths.get("rolling_report") or ""))
    runner_matrix_path = Path(str(paths.get("runner_matrix") or ""))
    if not rolling_report_path.exists():
        raise FileNotFoundError(f"rolling_report_missing:{rolling_report_path}")
    if not runner_matrix_path.exists():
        raise FileNotFoundError(f"runner_matrix_missing:{runner_matrix_path}")

    rolling_report = _load_json(rolling_report_path)
    runner_rows = _load_csv(runner_matrix_path)
    grouped = _race_rows(runner_rows)
    declared_keys = _declared_candidate_keys(rolling_report)
    if "market_only_implied" not in declared_keys:
        raise ValueError("rolling_report_missing_market_candidate")

    (
        candidate_metrics,
        aliases,
        unsupported_keys,
        per_race_by_candidate,
    ) = _score_candidates(declared_keys, grouped)
    if "market_only_implied" not in candidate_metrics:
        raise ValueError("market_candidate_not_evaluable")
    if len(candidate_metrics) <= 1:
        raise ValueError("no_non_market_candidate_evaluable")

    market_metrics = candidate_metrics["market_only_implied"]
    non_market_metrics = [
        metrics
        for key, metrics in candidate_metrics.items()
        if key != "market_only_implied" and metrics.get("status") == "EVALUATED"
    ]
    if not non_market_metrics:
        raise ValueError("no_evaluated_non_market_candidates")
    best_non_market = sorted(non_market_metrics, key=_metric_sort_key)[0]
    best_non_market_key = str(best_non_market["candidate_key"])
    per_race_rows = (
        per_race_by_candidate.get("market_only_implied", [])
        + per_race_by_candidate.get(best_non_market_key, [])
    )
    minimum_races_for_review = int(
        rolling_report.get("minimum_races_for_review")
        or adapter_packet.get("rolling_context", {}).get("minimum_races_for_review")
        or 100
    )
    gate = _market_safety_gate(
        candidate=best_non_market,
        market=market_metrics,
        minimum_races_for_review=minimum_races_for_review,
    )
    final_status = (
        "WOULD_CLEAR_MARKET_SAFETY_METRIC_GATES_REPORT_ONLY"
        if gate["would_clear_metric_gates"]
        else "BLOCKED_KEEP_BASELINE"
    )
    state = "DONE_WITH_RISK" if final_status == "BLOCKED_KEEP_BASELINE" else "DONE"

    current_race_ids = set(grouped)
    old_guard = _old_input_guard(
        current_race_ids=current_race_ids,
        old_result_path=old_result_per_race_path,
    )

    market_definition = _candidate_definition("market_only_implied")
    best_definition = _candidate_definition(str(best_non_market["candidate_key"]))
    residual_slices = []
    if market_definition is not None and best_definition is not None:
        residual_slices = _residual_slice_diagnostics(
            grouped_rows=grouped,
            candidate_key=best_non_market_key,
            candidate_definition=best_definition,
            market_definition=market_definition,
        )

    comparison = _comparison_rows(candidate_metrics, "market_only_implied")
    candidate_metrics_path = output_dir / "candidate_metrics.json"
    comparison_path = output_dir / "candidate_comparison.tsv"
    per_race_path = output_dir / "per_race_top_picks.tsv"
    residual_path = output_dir / "residual_slice_diagnostics.json"
    report_path = output_dir / "CUMULATIVE_RUNNER_MATRIX_MARKET_SAFETY_REPORT.json"
    manifest_path = output_dir / "output_manifest.json"
    readme_path = output_dir / "README.md"

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now.isoformat(),
        "state": state,
        "final_status": final_status,
        "input_surface": "current_cumulative_rolling_runner_matrix",
        "adapter_packet_path": str(adapter_packet_path),
        "source_rolling_report_path": str(rolling_report_path),
        "source_runner_matrix_path": str(runner_matrix_path),
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        "market_candidate_key": "market_only_implied",
        "candidate_count_declared": len(declared_keys),
        "candidate_count_evaluated": len(candidate_metrics),
        "unsupported_declared_candidate_keys": unsupported_keys,
        "candidate_aliases": aliases,
        "market_metrics": market_metrics,
        "best_non_market_candidate": best_non_market,
        "best_non_market_minus_market": _candidate_minus(best_non_market, market_metrics),
        "market_safety_rank_first_gate": gate,
        "old_recovered_input_guard": old_guard,
        "residual_slice_diagnostics": residual_slices,
        "adapter_counts": adapter_packet.get("counts"),
        "adapter_readiness": adapter_packet.get("readiness"),
        "rolling_context": adapter_packet.get("rolling_context"),
        "output_paths": {
            "report": str(report_path),
            "candidate_metrics": str(candidate_metrics_path),
            "candidate_comparison": str(comparison_path),
            "per_race_top_picks": str(per_race_path),
            "residual_slice_diagnostics": str(residual_path),
            "output_manifest": str(manifest_path),
            "readme": str(readme_path),
        },
        "stop_state_reason": (
            "Market-safety/rank-first gate still fails; baseline must remain."
            if final_status == "BLOCKED_KEEP_BASELINE"
            else "Metric gates would clear, but promotion remains forbidden in report-only mode."
        ),
    }

    manifest = {
        "schema_version": "cumulative_runner_matrix_challenger_output_manifest_v1",
        "generated_at": now.isoformat(),
        "final_status": final_status,
        "files": report["output_paths"],
    }

    _write_json(candidate_metrics_path, candidate_metrics)
    _write_tsv(comparison_path, comparison)
    _write_tsv(per_race_path, per_race_rows)
    _write_json(residual_path, residual_slices)
    _write_json(report_path, report)
    _write_json(manifest_path, manifest)
    readme_path.write_text(_readme_text(report), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-packet", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--old-result-per-race",
        type=Path,
        help=(
            "Optional recovered-input per-race result file. When supplied, "
            "zero overlap with the current cumulative sample is labeled "
            "OLD_INPUT_DIAGNOSTIC."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_evaluation(
        adapter_packet_path=args.adapter_packet,
        output_dir=args.output_dir,
        old_result_per_race_path=args.old_result_per_race,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
