#!/usr/bin/env python3
"""Train report-only calibration challengers from exact-joined shadow results.

This worker is intentionally detached from production. It reads deduplicated
forward-shadow result joins, fits only additive probability-calibration
parameters on exact official joins, and writes challenger artifacts under the
evidence directory. It never mutates the model registry, DB labels, production
models, prediction snapshots, EV output, or betting output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from accuracy_program.calibration import power_normalize_by_race  # noqa: E402
from scripts.aggregate_forward_shadow_results import (  # noqa: E402
    DEFAULT_EVIDENCE_ROOT,
    grouped_joined_rows,
    read_jsonl,
    result_join_dirs,
    selected_unique_joined_races,
)
from scripts.join_forward_shadow_results import (  # noqa: E402
    clip_probability,
    logistic_calibration_review,
    probability_reliability_bins,
)


OUTPUT_PREFIX = (
    "artifacts/full_evidence_orchestration_20260525/"
    "forward_shadow_challenger_calibration_"
)
DEFAULT_PROTECTED_PATHS = (
    ROOT / "greyhound_racing_data.db",
    ROOT / "greyhound_racing_data_writable.db",
    ROOT / "model_registry/best_metadata.json",
    ROOT / "docs/model_contracts/v4_feature_contract.json",
    ROOT / "artifacts/prediction_snapshots/manifest.jsonl",
)
DEFAULT_ALPHA_GRID = (
    0.40,
    0.50,
    0.60,
    0.75,
    0.90,
    1.00,
    1.10,
    1.25,
    1.50,
    1.75,
    2.00,
    2.40,
    3.00,
)
BASELINE_ALPHA = 1.0

FINAL_READY = "CHALLENGER_CALIBRATION_REPORT_ONLY_READY_FOR_REVIEW"
FINAL_BLOCKED = "CHALLENGER_CALIBRATION_BLOCKED_KEEP_BASELINE"
FINAL_NO_SAFE_JOINS = "CHALLENGER_CALIBRATION_NO_SAFE_JOINED_RACES"


@dataclass(frozen=True)
class ChallengerThresholds:
    min_total_safe_joined_races: int = 100
    min_train_races: int = 40
    min_eval_races: int = 20
    max_probability_sum_error: float = 1e-6
    metric_tolerance: float = 0.0
    train_fraction: float = 0.8


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return os.path.relpath(path.resolve(), ROOT.resolve())
    except ValueError:
        return str(path)


def now_id(now: datetime | None = None) -> str:
    return (now or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes(paths: Sequence[Path] = DEFAULT_PROTECTED_PATHS) -> dict[str, str | None]:
    return {relpath(path) or str(path): sha256_file(path) for path in paths}


def assert_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative = logical.absolute().relative_to(ROOT.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(
            f"output_dir_must_be_forward_shadow_challenger_calibration_artifact:{relative}"
        )
    return logical.absolute()


def unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = Path(f"{base}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"output_dir_collision_exhausted:{base}")


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def race_key(row: Mapping[str, Any]) -> str:
    return str(row.get("race_id") or "").strip()


def race_sort_key(rows: Sequence[Mapping[str, Any]]) -> tuple[str, str, int, str]:
    first = rows[0] if rows else {}
    race_number = first.get("race_number")
    try:
        parsed_number = int(race_number)
    except (TypeError, ValueError):
        parsed_number = 999
    return (
        str(first.get("race_date") or ""),
        str(first.get("jump_datetime") or ""),
        parsed_number,
        race_key(first),
    )


def validate_exact_joined_race(
    race_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    input_probability_key: str,
    max_probability_sum_error: float,
) -> tuple[list[dict[str, Any]] | None, list[str]]:
    reasons: list[str] = []
    if not rows:
        return None, ["race_rows_missing"]
    winners = [row for row in rows if row.get("is_winner") is True]
    if len(winners) != 1:
        reasons.append("winner_row_count_not_exactly_one")
    boxes = [row.get("box") for row in rows]
    if len(set(boxes)) != len(boxes):
        reasons.append("duplicate_box_in_joined_rows")
    for row in rows:
        if race_key(row) != race_id:
            reasons.append("mixed_race_id_in_group")
            break
        if row.get("identity_match_status") != "exact_box_and_normalized_name":
            reasons.append("non_exact_identity_match_status")
            break
        probability = finite_float(row.get(input_probability_key))
        if probability is None:
            reasons.append(f"{input_probability_key}_missing_or_invalid")
            break
        if probability < 0:
            reasons.append(f"{input_probability_key}_negative")
            break
        if bool(row.get("tgr_enabled")):
            reasons.append("tgr_enabled_joined_row")
            break
    probability_sum = sum(
        finite_float(row.get(input_probability_key)) or 0.0 for row in rows
    )
    if abs(probability_sum - 1.0) > max_probability_sum_error:
        reasons.append("probability_sum_error_exceeds_threshold")
    if reasons:
        return None, list(dict.fromkeys(reasons))

    cleaned: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item[input_probability_key] = float(item[input_probability_key])
        item["is_winner"] = bool(item["is_winner"])
        cleaned.append(item)
    return cleaned, []


def collect_candidate_races(
    *,
    evidence_root: Path,
    input_probability_key: str,
    max_probability_sum_error: float,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]], list[str]]:
    join_dirs = result_join_dirs(evidence_root)
    selected, duplicates = selected_unique_joined_races(join_dirs)
    safe_races: list[list[dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []
    for key, rows in sorted(selected.items()):
        cleaned, reasons = validate_exact_joined_race(
            key,
            rows,
            input_probability_key=input_probability_key,
            max_probability_sum_error=max_probability_sum_error,
        )
        if cleaned is None:
            rejected.append(
                {
                    "race_id": key,
                    "reasons": reasons,
                    "row_count": len(rows),
                }
            )
            continue
        safe_races.append(cleaned)
    safe_races.sort(key=race_sort_key)
    return safe_races, rejected, [str(item.get("race_id")) for item in duplicates]


def flatten(races: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    return [dict(row) for rows in races for row in rows]


def metric_cohort_fields(races: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    race_ids = [race_key(rows[0]) for rows in races if rows]
    race_ids_hash = hashlib.sha256(
        json.dumps(race_ids, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return {
        "evaluation_cohort_id": f"forward_shadow_challenger_eval:{race_ids_hash}",
        "metric_cohort_id": f"forward_shadow_challenger_eval:{race_ids_hash}",
        "safe_joined_race_ids_hash": race_ids_hash,
        "safe_joined_race_ids_count": len(race_ids),
    }


def activation_metric_payload(
    metrics: Mapping[str, Any],
    *,
    metric_role: str,
    cohort_fields: Mapping[str, Any],
    source_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(metrics)
    payload.update(cohort_fields)
    payload.update(source_fields or {})
    payload.update(
        {
            "schema_version": "forward_shadow_activation_metrics_v1",
            "metric_role": metric_role,
        }
    )
    return payload


def top_pick(rows: Sequence[Mapping[str, Any]], probability_key: str) -> Mapping[str, Any]:
    return sorted(
        rows,
        key=lambda row: (-float(row[probability_key]), int(row.get("box") or 999), str(row.get("dog_name") or "")),
    )[0]


def winner(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    winners = [row for row in rows if row.get("is_winner") is True]
    if len(winners) != 1:
        raise ValueError("winner_row_count_not_exactly_one")
    return winners[0]


def rank_of_winner(rows: Sequence[Mapping[str, Any]], probability_key: str) -> int:
    ranked = sorted(
        rows,
        key=lambda row: (-float(row[probability_key]), int(row.get("box") or 999), str(row.get("dog_name") or "")),
    )
    for index, row in enumerate(ranked, start=1):
        if row.get("is_winner") is True:
            return index
    raise ValueError("winner_missing_from_ranked_rows")


def metric_summary(races: Sequence[Sequence[Mapping[str, Any]]], probability_key: str) -> dict[str, Any]:
    if not races:
        return {
            "safe_joined_race_count": 0,
            "safe_joined_runner_count": 0,
            "top1": None,
            "top3": None,
            "mean_winner_rank": None,
            "brier": None,
            "logloss": None,
            "probability_sum_max_error_joined_races": None,
            "box1_top_pick_share": None,
            "reliability_bins": [],
            "calibration_slope_intercept": {
                "status": "no_safe_joined_rows",
                "slope": None,
                "intercept": None,
            },
        }

    labels: list[int] = []
    probabilities: list[float] = []
    winner_ranks: list[int] = []
    top1_count = 0
    top3_count = 0
    top_pick_boxes: Counter[str] = Counter()
    probability_sum_errors: list[float] = []
    logloss_values: list[float] = []

    for rows in races:
        race_probabilities = [float(row[probability_key]) for row in rows]
        probability_sum_errors.append(abs(sum(race_probabilities) - 1.0))
        top = top_pick(rows, probability_key)
        win = winner(rows)
        winner_rank = rank_of_winner(rows, probability_key)
        winner_ranks.append(winner_rank)
        if bool(top.get("is_winner")):
            top1_count += 1
        if winner_rank <= 3:
            top3_count += 1
        top_pick_boxes[str(top.get("box"))] += 1
        logloss_values.append(-math.log(clip_probability(float(win[probability_key]))))
        for row in rows:
            labels.append(1 if row.get("is_winner") else 0)
            probabilities.append(float(row[probability_key]))

    brier = sum((probability - label) ** 2 for label, probability in zip(labels, probabilities)) / len(labels)
    return {
        "safe_joined_race_count": len(races),
        "safe_joined_runner_count": len(labels),
        "top1": top1_count / len(races),
        "top3": top3_count / len(races),
        "winner_ranks": winner_ranks,
        "mean_winner_rank": sum(winner_ranks) / len(winner_ranks),
        "brier": brier,
        "logloss": sum(logloss_values) / len(logloss_values),
        "probability_sum_max_error_joined_races": max(probability_sum_errors),
        "box1_top_pick_share": top_pick_boxes.get("1", 0) / len(races),
        "top_pick_box_distribution": dict(sorted(top_pick_boxes.items())),
        "reliability_bins": probability_reliability_bins(labels, probabilities),
        "calibration_slope_intercept": logistic_calibration_review(labels, probabilities),
    }


def apply_alpha(
    races: Sequence[Sequence[Mapping[str, Any]]],
    *,
    alpha: float,
    input_probability_key: str,
    output_probability_key: str,
) -> list[list[dict[str, Any]]]:
    rows = flatten(races)
    calibrated = power_normalize_by_race(
        rows,
        alpha=alpha,
        input_key=input_probability_key,
        output_key=output_probability_key,
        race_key="race_id",
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in calibrated:
        grouped[race_key(row)].append(row)
    return [grouped[race_key(rows[0])] for rows in races]


def alpha_metric_grid(
    races: Sequence[Sequence[Mapping[str, Any]]],
    *,
    alpha_grid: Sequence[float],
    input_probability_key: str,
) -> list[dict[str, Any]]:
    output_key = "challenger_probability"
    results = []
    for alpha in alpha_grid:
        calibrated = apply_alpha(
            races,
            alpha=alpha,
            input_probability_key=input_probability_key,
            output_probability_key=output_key,
        )
        metrics = metric_summary(calibrated, output_key)
        results.append(
            {
                "alpha": alpha,
                "metrics": {
                    key: value
                    for key, value in metrics.items()
                    if key
                    not in {
                        "reliability_bins",
                        "calibration_slope_intercept",
                        "winner_ranks",
                        "top_pick_box_distribution",
                    }
                },
            }
        )
    return sorted(
        results,
        key=lambda row: (
            float(row["metrics"]["logloss"]),
            float(row["metrics"]["brier"]),
            abs(float(row["alpha"]) - BASELINE_ALPHA),
        ),
    )


def split_races(
    races: Sequence[Sequence[Mapping[str, Any]]],
    *,
    train_fraction: float,
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    if not races:
        return [], []
    if not (0 < train_fraction < 1):
        raise ValueError("train_fraction_must_be_between_0_and_1")
    split_index = int(len(races) * train_fraction)
    split_index = min(max(split_index, 1), len(races) - 1)
    return [list(rows) for rows in races[:split_index]], [list(rows) for rows in races[split_index:]]


def activation_blockers(
    *,
    total_races: int,
    train_races: int,
    eval_races: int,
    baseline_eval_metrics: Mapping[str, Any],
    candidate_eval_metrics: Mapping[str, Any],
    thresholds: ChallengerThresholds,
    protected_paths_unchanged: bool,
) -> list[str]:
    blockers: list[str] = []
    if total_races <= 0:
        blockers.append("no_safe_exact_joined_races")
    if total_races < thresholds.min_total_safe_joined_races:
        blockers.append("safe_joined_race_count_below_min_total")
    if train_races < thresholds.min_train_races:
        blockers.append("train_race_count_below_min")
    if eval_races < thresholds.min_eval_races:
        blockers.append("eval_race_count_below_min")
    if not protected_paths_unchanged:
        blockers.append("protected_paths_changed")
    max_error = candidate_eval_metrics.get("probability_sum_max_error_joined_races")
    if max_error is None or float(max_error) > thresholds.max_probability_sum_error:
        blockers.append("candidate_probability_sum_error_failed")

    comparisons = (
        ("top1", "higher_or_equal"),
        ("top3", "higher_or_equal"),
        ("mean_winner_rank", "lower_or_equal"),
        ("brier", "lower_or_equal"),
        ("logloss", "lower_or_equal"),
    )
    for key, direction in comparisons:
        baseline_value = baseline_eval_metrics.get(key)
        candidate_value = candidate_eval_metrics.get(key)
        if baseline_value is None or candidate_value is None:
            blockers.append(f"metric_missing:{key}")
            continue
        baseline = float(baseline_value)
        candidate = float(candidate_value)
        if direction == "higher_or_equal" and candidate + thresholds.metric_tolerance < baseline:
            blockers.append(f"metric_regressed:{key}")
        if direction == "lower_or_equal" and candidate > baseline + thresholds.metric_tolerance:
            blockers.append(f"metric_regressed:{key}")
    return list(dict.fromkeys(blockers))


def report_only_prediction_rows(
    races: Sequence[Sequence[Mapping[str, Any]]],
    *,
    alpha: float,
    input_probability_key: str,
) -> list[dict[str, Any]]:
    output_key = "challenger_calibrated_probability_report_only"
    calibrated = apply_alpha(
        races,
        alpha=alpha,
        input_probability_key=input_probability_key,
        output_probability_key=output_key,
    )
    rows = []
    for race_rows in calibrated:
        ranked = sorted(
            race_rows,
            key=lambda row: (-float(row[output_key]), int(row.get("box") or 999), str(row.get("dog_name") or "")),
        )
        rank_by_identity = {
            (race_key(row), row.get("box"), row.get("dog_name")): index
            for index, row in enumerate(ranked, start=1)
        }
        for row in race_rows:
            item = {
                "race_id": row.get("race_id"),
                "race_date": row.get("race_date"),
                "venue": row.get("venue"),
                "race_number": row.get("race_number"),
                "box": row.get("box"),
                "dog_name": row.get("dog_name"),
                "is_winner": bool(row.get("is_winner")),
                input_probability_key: row.get(input_probability_key),
                output_key: row.get(output_key),
                "challenger_predicted_rank_report_only": rank_by_identity[
                    (race_key(row), row.get("box"), row.get("dog_name"))
                ],
                "identity_match_status": row.get("identity_match_status"),
            }
            rows.append(item)
    return rows


def build_report(
    *,
    evidence_root: Path,
    input_probability_key: str,
    alpha_grid: Sequence[float],
    thresholds: ChallengerThresholds,
    generated_at: datetime | None = None,
    protected_before: Mapping[str, str | None] | None = None,
    protected_after: Mapping[str, str | None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    generated_at = generated_at or datetime.now().astimezone()
    protected_before = dict(protected_before or protected_hashes())
    safe_races, rejected_races, duplicate_race_ids = collect_candidate_races(
        evidence_root=evidence_root,
        input_probability_key=input_probability_key,
        max_probability_sum_error=thresholds.max_probability_sum_error,
    )
    train_races, eval_races = split_races(
        safe_races,
        train_fraction=thresholds.train_fraction,
    ) if len(safe_races) >= 2 else (safe_races, [])

    if not safe_races:
        protected_after = dict(protected_after or protected_hashes())
        report = {
            "schema_version": "forward_shadow_challenger_calibration_v1",
            "generated_at": generated_at.isoformat(),
            "final_status": FINAL_NO_SAFE_JOINS,
            "input_probability_key": input_probability_key,
            "safe_exact_joined_race_count": 0,
            "safe_exact_joined_runner_count": 0,
            "rejected_joined_races": rejected_races,
            "activation_blockers": ["no_safe_exact_joined_races"],
            "protected_hashes_before": protected_before,
            "protected_hashes_after": protected_after,
            "protected_paths_unchanged": protected_before == protected_after,
            "no_write_guarantees": no_write_guarantees(),
        }
        return report, []

    training_grid = alpha_metric_grid(
        train_races,
        alpha_grid=alpha_grid,
        input_probability_key=input_probability_key,
    )
    best_alpha = float(training_grid[0]["alpha"])
    baseline_train = metric_summary(
        apply_alpha(
            train_races,
            alpha=BASELINE_ALPHA,
            input_probability_key=input_probability_key,
            output_probability_key="baseline_probability",
        ),
        "baseline_probability",
    )
    baseline_eval = metric_summary(
        apply_alpha(
            eval_races,
            alpha=BASELINE_ALPHA,
            input_probability_key=input_probability_key,
            output_probability_key="baseline_probability",
        ),
        "baseline_probability",
    )
    candidate_train = metric_summary(
        apply_alpha(
            train_races,
            alpha=best_alpha,
            input_probability_key=input_probability_key,
            output_probability_key="candidate_probability",
        ),
        "candidate_probability",
    )
    candidate_eval_races = apply_alpha(
        eval_races,
        alpha=best_alpha,
        input_probability_key=input_probability_key,
        output_probability_key="candidate_probability",
    )
    eval_cohort_fields = metric_cohort_fields(eval_races)
    baseline_eval = activation_metric_payload(
        baseline_eval,
        metric_role="baseline_eval",
        cohort_fields=eval_cohort_fields,
        source_fields={
            "source_safe_exact_joined_race_count": len(safe_races),
            "source_safe_exact_joined_runner_count": len(flatten(safe_races)),
            "source_train_race_count": len(train_races),
            "source_eval_race_count": len(eval_races),
            "source_generated_at": generated_at.isoformat(),
        },
    )
    candidate_eval = activation_metric_payload(
        metric_summary(candidate_eval_races, "candidate_probability"),
        metric_role="candidate_eval",
        cohort_fields=eval_cohort_fields,
        source_fields={
            "source_safe_exact_joined_race_count": len(safe_races),
            "source_safe_exact_joined_runner_count": len(flatten(safe_races)),
            "source_train_race_count": len(train_races),
            "source_eval_race_count": len(eval_races),
            "source_generated_at": generated_at.isoformat(),
        },
    )
    protected_after = dict(protected_after or protected_hashes())
    blockers = activation_blockers(
        total_races=len(safe_races),
        train_races=len(train_races),
        eval_races=len(eval_races),
        baseline_eval_metrics=baseline_eval,
        candidate_eval_metrics=candidate_eval,
        thresholds=thresholds,
        protected_paths_unchanged=protected_before == protected_after,
    )
    final_status = FINAL_BLOCKED if blockers else FINAL_READY
    report = {
        "schema_version": "forward_shadow_challenger_calibration_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": final_status,
        "input_probability_key": input_probability_key,
        "calibration_family": "per_race_power_normalization",
        "candidate_alpha": best_alpha,
        "baseline_alpha": BASELINE_ALPHA,
        "alpha_selection_metric": "train_logloss_then_train_brier",
        "alpha_grid_results_train": training_grid,
        "safe_exact_joined_race_count": len(safe_races),
        "safe_exact_joined_runner_count": len(flatten(safe_races)),
        "train_race_count": len(train_races),
        "eval_race_count": len(eval_races),
        "split_policy": {
            "method": "chronological_by_race_date_jump_datetime_race_number_race_id",
            "train_fraction": thresholds.train_fraction,
        },
        "baseline_train_metrics": baseline_train,
        "candidate_train_metrics": candidate_train,
        "baseline_eval_metrics": baseline_eval,
        "candidate_eval_metrics": candidate_eval,
        "activation_blockers": blockers,
        "thresholds": asdict(thresholds),
        "rejected_joined_races": rejected_races,
        "duplicate_joined_race_ids_seen": duplicate_race_ids,
        "source_join_artifact_selection_policy": (
            "latest_result_join_per_source_shadow_run_then_latest_join_artifact_per_unique_race"
        ),
        "report_only": True,
        "production_activation_allowed": False,
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "protected_paths_unchanged": protected_before == protected_after,
        "no_write_guarantees": no_write_guarantees(),
    }
    return report, report_only_prediction_rows(
        safe_races,
        alpha=best_alpha,
        input_probability_key=input_probability_key,
    )


def no_write_guarantees() -> dict[str, bool]:
    return {
        "production_promotion": False,
        "registry_mutation": False,
        "production_pointer_update": False,
        "active_model_replacement": False,
        "production_prediction_write": False,
        "db_write": False,
        "label_write": False,
        "canonical_schema_mutation": False,
        "tgr_enabled": False,
        "betting_or_ev_output": False,
    }


def build_summary(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Forward Shadow Challenger Calibration",
            "",
            f"- Final status: `{report.get('final_status')}`",
            f"- Safe exact joined races: `{report.get('safe_exact_joined_race_count')}`",
            f"- Train races: `{report.get('train_race_count')}`",
            f"- Eval races: `{report.get('eval_race_count')}`",
            f"- Candidate alpha: `{report.get('candidate_alpha')}`",
            f"- Activation blockers: `{len(report.get('activation_blockers') or [])}`",
            f"- Protected paths unchanged: `{report.get('protected_paths_unchanged')}`",
            "",
            "This is report-only. No production model, registry, DB, label, snapshot, EV, or betting output was written.",
            "",
        ]
    )


def run_challenger_calibration(
    *,
    evidence_root: Path = DEFAULT_EVIDENCE_ROOT,
    output_dir: Path | None = None,
    input_probability_key: str = "shadow_rf_calibrated_probability",
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
    thresholds: ChallengerThresholds = ChallengerThresholds(),
) -> dict[str, Any]:
    generated_at = datetime.now().astimezone()
    output_dir = output_dir or evidence_root / f"forward_shadow_challenger_calibration_{now_id(generated_at)}"
    output_dir = unique_dir(assert_output_dir_safe(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    protected_before = protected_hashes()
    report, prediction_rows = build_report(
        evidence_root=evidence_root,
        input_probability_key=input_probability_key,
        alpha_grid=alpha_grid,
        thresholds=thresholds,
        generated_at=generated_at,
        protected_before=protected_before,
    )
    write_json(output_dir / "challenger_calibration_report.json", report)
    baseline_activation_metrics = dict(report.get("baseline_eval_metrics") or {})
    candidate_activation_metrics = dict(report.get("candidate_eval_metrics") or {})
    source_report = relpath(output_dir / "challenger_calibration_report.json")
    baseline_activation_metrics["source_report"] = source_report
    candidate_activation_metrics["source_report"] = source_report
    baseline_activation_metrics["source_final_status"] = report.get("final_status")
    candidate_activation_metrics["source_final_status"] = report.get("final_status")
    baseline_activation_metrics["source_activation_blockers"] = report.get("activation_blockers") or []
    candidate_activation_metrics["source_activation_blockers"] = report.get("activation_blockers") or []
    write_json(output_dir / "baseline_eval_metrics_for_activation.json", baseline_activation_metrics)
    write_json(output_dir / "candidate_eval_metrics_for_activation.json", candidate_activation_metrics)
    write_json(output_dir / "challenger_activation_gate.json", {
        "schema_version": "forward_shadow_challenger_activation_gate_v1",
        "final_status": report["final_status"],
        "activation_blockers": report.get("activation_blockers") or [],
        "production_activation_allowed": False,
        "report_only": True,
    })
    write_jsonl(output_dir / "challenger_predictions_report_only.jsonl", prediction_rows)
    write_text(output_dir / "SUMMARY.md", build_summary(report))
    write_text(output_dir / "final_status.txt", str(report["final_status"]) + "\n")
    return {
        "output_dir": relpath(output_dir),
        "final_status": report["final_status"],
        "safe_exact_joined_race_count": report.get("safe_exact_joined_race_count"),
        "train_race_count": report.get("train_race_count"),
        "eval_race_count": report.get("eval_race_count"),
        "candidate_alpha": report.get("candidate_alpha"),
        "activation_blockers": report.get("activation_blockers") or [],
        "protected_paths_unchanged": report.get("protected_paths_unchanged"),
    }


def parse_alpha_grid(text: str | None) -> tuple[float, ...]:
    if not text:
        return DEFAULT_ALPHA_GRID
    values = []
    for part in text.split(","):
        value = finite_float(part.strip())
        if value is None or value <= 0:
            raise ValueError("alpha_grid_values_must_be_positive_finite")
        values.append(value)
    if not values:
        raise ValueError("alpha_grid_missing_values")
    return tuple(dict.fromkeys(values))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--input-probability-key", default="shadow_rf_calibrated_probability")
    parser.add_argument("--alpha-grid")
    parser.add_argument("--min-total-safe-joined-races", type=int, default=100)
    parser.add_argument("--min-train-races", type=int, default=40)
    parser.add_argument("--min-eval-races", type=int, default=20)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--metric-tolerance", type=float, default=0.0)
    parser.add_argument("--max-probability-sum-error", type=float, default=1e-6)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    thresholds = ChallengerThresholds(
        min_total_safe_joined_races=args.min_total_safe_joined_races,
        min_train_races=args.min_train_races,
        min_eval_races=args.min_eval_races,
        max_probability_sum_error=args.max_probability_sum_error,
        metric_tolerance=args.metric_tolerance,
        train_fraction=args.train_fraction,
    )
    result = run_challenger_calibration(
        evidence_root=args.evidence_root,
        output_dir=args.output_dir,
        input_probability_key=args.input_probability_key,
        alpha_grid=parse_alpha_grid(args.alpha_grid),
        thresholds=thresholds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
