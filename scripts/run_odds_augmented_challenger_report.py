#!/usr/bin/env python3
"""Run report-only odds-augmented challenger comparisons after the odds gate.

This script requires an already-ready odds research gate and exact joined
forward-shadow results. It never writes model artifacts, registry pointers, DB
rows, labels, production predictions, betting advice, stakes, or actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from scripts.join_forward_shadow_results import (  # noqa: E402
    clip_probability,
    logistic_calibration_review,
    probability_reliability_bins,
)


OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/odds_augmented_challenger_"
ODDS_RESEARCH_READY_REPORT_ONLY = "ODDS_RESEARCH_READY_REPORT_ONLY"
ODDS_AUGMENTED_MODEL_BLOCKED = "ODDS_AUGMENTED_MODEL_BLOCKED"
ODDS_AUGMENTED_MODEL_READY_FOR_PR_REVIEW = "ODDS_AUGMENTED_MODEL_READY_FOR_PR_REVIEW"
MIN_COMPLETE_VALID_PREJUMP_ODDS_RACES = 100
DEFAULT_PROTECTED_PATHS = (
    ROOT / "greyhound_racing_data.db",
    ROOT / "greyhound_racing_data_writable.db",
    ROOT / "model_registry/best_metadata.json",
    ROOT / "docs/model_contracts/v4_feature_contract.json",
    ROOT / "artifacts/prediction_snapshots/manifest.jsonl",
)


def now_id(now: datetime | None = None) -> str:
    return (now or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return os.path.relpath(path.resolve(), ROOT.resolve())
    except ValueError:
        return str(path)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("json_root_not_object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def input_jsonl_path(path: Path, default_name: str) -> Path:
    return path / default_name if path.is_dir() else path


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
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
        raise ValueError(f"output_dir_must_be_odds_augmented_challenger_artifact:{relative}")
    return logical.absolute()


def unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = Path(f"{base}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"output_dir_collision_exhausted:{base}")


def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())


def identity_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(row.get("race_id") or "").strip(),
        int(row.get("box") or row.get("box_number") or 0),
        normalize_name(row.get("dog_name") or row.get("dog_clean_name")),
    )


def odds_decimal(row: Mapping[str, Any]) -> float | None:
    snapshot = row.get("odds_snapshot") if isinstance(row.get("odds_snapshot"), Mapping) else {}
    value = snapshot.get("market_odds_win") or row.get("odds_decimal") or row.get("odds")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 1.0 and math.isfinite(parsed) else None


def grouped_by_race(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("race_id") or "")].append(row)
    return dict(grouped)


def normalize_probability_key(
    rows: Sequence[Mapping[str, Any]],
    source_key: str,
    output_key: str,
) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    for race_rows in grouped_by_race(output).values():
        total = 0.0
        values: list[float] = []
        for row in race_rows:
            try:
                value = max(float(row.get(source_key) or 0.0), 0.0)
            except (TypeError, ValueError):
                value = 0.0
            values.append(value)
            total += value
        if total <= 0:
            normalized = [1.0 / len(race_rows) for _ in race_rows]
        else:
            normalized = [value / total for value in values]
        for row, probability in zip(race_rows, normalized):
            row[output_key] = probability
    return output


def with_candidate_probabilities(
    joined_rows: Sequence[Mapping[str, Any]],
    odds_rows_by_key: Mapping[tuple[str, int, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    rejected = []
    for row in joined_rows:
        key = identity_key(row)
        odds_row = odds_rows_by_key.get(key)
        if odds_row is None:
            rejected.append({"race_id": row.get("race_id"), "box": row.get("box"), "reason": "missing_valid_odds_row"})
            continue
        decimal = odds_decimal(odds_row)
        if decimal is None or odds_row.get("odds_match_status") != "valid_pre_jump_dog_odds":
            rejected.append({"race_id": row.get("race_id"), "box": row.get("box"), "reason": "invalid_odds_row"})
            continue
        base_probability = float(row.get("shadow_rf_calibrated_probability") or 0.0)
        rows.append(
            {
                **dict(row),
                "odds_decimal": decimal,
                "market_implied_raw": 1.0 / decimal,
                "stage2_no_odds_probability": base_probability,
            }
        )
    rows = normalize_probability_key(rows, "stage2_no_odds_probability", "stage2_no_odds_probability")
    rows = normalize_probability_key(rows, "market_implied_raw", "market_only_probability")
    for row in rows:
        row["odds_augmented_probability_raw"] = (
            0.5 * float(row["stage2_no_odds_probability"])
            + 0.5 * float(row["market_only_probability"])
        )
        row["probability_blend_candidate_raw"] = (
            0.65 * float(row["stage2_no_odds_probability"])
            + 0.35 * float(row["market_only_probability"])
        )
    rows = normalize_probability_key(rows, "odds_augmented_probability_raw", "odds_augmented_probability")
    rows = normalize_probability_key(rows, "probability_blend_candidate_raw", "probability_blend_candidate_probability")
    return rows, rejected


def metrics_for_probability(rows: Sequence[Mapping[str, Any]], probability_key: str) -> dict[str, Any]:
    grouped = grouped_by_race(rows)
    safe_races = []
    labels: list[int] = []
    probabilities: list[float] = []
    brier_values: list[float] = []
    top_boxes: Counter[str] = Counter()
    winner_ranks: list[int] = []
    logloss_values: list[float] = []
    probability_sum_errors: list[float] = []
    for race_id, race_rows in grouped.items():
        ordered = sorted(
            race_rows,
            key=lambda row: (-float(row.get(probability_key) or 0.0), int(row.get("box") or 999)),
        )
        if not ordered:
            continue
        winners = [row for row in ordered if row.get("is_winner") is True]
        if len(winners) != 1:
            continue
        winner = winners[0]
        winner_rank = ordered.index(winner) + 1
        top_pick = ordered[0]
        top_boxes[str(top_pick.get("box"))] += 1
        safe_races.append(
            {
                "race_id": race_id,
                "top_pick_box": top_pick.get("box"),
                "top_pick_won": top_pick.get("is_winner") is True,
                "winner_predicted_rank": winner_rank,
                "winner_in_top3": winner_rank <= 3,
            }
        )
        winner_ranks.append(winner_rank)
        probability_sum_errors.append(
            abs(sum(float(row.get(probability_key) or 0.0) for row in race_rows) - 1.0)
        )
        logloss_values.append(-math.log(clip_probability(float(winner.get(probability_key) or 0.0))))
        for row in race_rows:
            label = 1 if row.get("is_winner") is True else 0
            probability = clip_probability(float(row.get(probability_key) or 0.0))
            labels.append(label)
            probabilities.append(probability)
            brier_values.append((probability - label) ** 2)
    race_count = len(safe_races)
    return {
        "race_count": race_count,
        "safe_joined_race_count": race_count,
        "safe_joined_runner_count": len(rows),
        "top1": sum(1 for race in safe_races if race["top_pick_won"]) / race_count if race_count else None,
        "top3": sum(1 for race in safe_races if race["winner_in_top3"]) / race_count if race_count else None,
        "mean_winner_rank": sum(winner_ranks) / len(winner_ranks) if winner_ranks else None,
        "brier": sum(brier_values) / len(brier_values) if brier_values else None,
        "logloss": sum(logloss_values) / len(logloss_values) if logloss_values else None,
        "probability_sum_max_error_joined_races": max(probability_sum_errors) if probability_sum_errors else None,
        "calibration_slope_intercept": logistic_calibration_review(labels, probabilities) if labels else {},
        "reliability_bins": probability_reliability_bins(labels, probabilities) if labels else [],
        "box1_top_pick_share": top_boxes.get("1", 0) / race_count if race_count else None,
        "top_pick_box_distribution": dict(sorted(top_boxes.items())),
    }


def metric_value(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def int_value(metrics: Mapping[str, Any], key: str) -> int:
    try:
        return int(metrics.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def calibration_distance(metrics: Mapping[str, Any]) -> float | None:
    calibration = metrics.get("calibration_slope_intercept")
    if not isinstance(calibration, Mapping):
        return None
    slope = metric_value(calibration, "slope")
    intercept = metric_value(calibration, "intercept")
    if slope is None or intercept is None:
        return None
    return abs(slope - 1.0) + abs(intercept)


def candidate_blockers(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    min_top1_delta: float,
    max_box1_top_pick_share: float,
    max_probability_sum_error: float,
) -> list[str]:
    blockers: list[str] = []
    comparisons = (
        ("top1", "higher", min_top1_delta),
        ("top3", "higher", 0.0),
        ("mean_winner_rank", "lower", 0.0),
        ("brier", "lower", 0.0),
        ("logloss", "lower", 0.0),
    )
    for key, direction, required_delta in comparisons:
        base = metric_value(baseline, key)
        cand = metric_value(candidate, key)
        if base is None or cand is None:
            blockers.append(f"metric_missing:{key}")
        elif direction == "higher" and cand - base < required_delta:
            blockers.append(f"metric_regressed:{key}" if required_delta == 0 else f"{key}_delta_below_min")
        elif direction == "lower" and cand - base > required_delta:
            blockers.append(f"metric_regressed:{key}")
    base_cal = calibration_distance(baseline)
    cand_cal = calibration_distance(candidate)
    if base_cal is None or cand_cal is None:
        blockers.append("metric_missing:calibration_slope_intercept")
    elif cand_cal > base_cal:
        blockers.append("metric_regressed:calibration_slope_intercept")
    box1 = metric_value(candidate, "box1_top_pick_share")
    if box1 is None:
        blockers.append("metric_missing:box1_top_pick_share")
    elif box1 > max_box1_top_pick_share:
        blockers.append("candidate_box1_top_pick_share_above_max")
    sum_error = metric_value(candidate, "probability_sum_max_error_joined_races")
    if sum_error is None or sum_error > max_probability_sum_error:
        blockers.append("candidate_probability_sum_error_failed")
    return list(dict.fromkeys(blockers))


def ev_diagnostics(rows: Sequence[Mapping[str, Any]], probability_key: str) -> dict[str, Any]:
    values = [
        float(row[probability_key]) * float(row["odds_decimal"]) - 1.0
        for row in rows
        if row.get(probability_key) is not None and row.get("odds_decimal") is not None
    ]
    return {
        "schema_version": "report_only_ev_diagnostics_v1",
        "status": "EV_DIAGNOSTICS_REPORT_ONLY",
        "probability_key": probability_key,
        "ev_rows": len(values),
        "positive_ev_rows": sum(1 for value in values if value > 0),
        "negative_ev_rows": sum(1 for value in values if value < 0),
        "mean_ev": sum(values) / len(values) if values else None,
        "min_ev": min(values) if values else None,
        "max_ev": max(values) if values else None,
        "betting_advice": False,
        "stakes": False,
        "betting_action_allowed": False,
        "promotion_signal": False,
        "ev_can_override_accuracy_gate": False,
    }


def no_write_guarantees() -> dict[str, bool]:
    return {
        "production_promotion": False,
        "registry_mutation": False,
        "production_pointer_update": False,
        "active_model_replacement": False,
        "model_artifact_write": False,
        "production_prediction_write": False,
        "db_write": False,
        "label_write": False,
        "snapshot_rewrite": False,
        "tgr_enabled": False,
        "betting_action": False,
    }


def build_report(
    *,
    joined_rows: Sequence[Mapping[str, Any]],
    odds_rows: Sequence[Mapping[str, Any]],
    odds_gate_report: Mapping[str, Any],
    generated_at: datetime | None = None,
    min_top1_delta: float = 0.0,
    max_box1_top_pick_share: float = 0.35,
    max_probability_sum_error: float = 1e-6,
    protected_before: Mapping[str, str | None] | None = None,
    protected_after: Mapping[str, str | None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    generated_at = generated_at or datetime.now().astimezone()
    protected_before = dict(protected_before or protected_hashes())
    protected_after = dict(protected_after or protected_hashes())
    blockers: list[str] = []
    if odds_gate_report.get("status") != ODDS_RESEARCH_READY_REPORT_ONLY:
        blockers.append("odds_research_gate_not_ready")
    if int_value(odds_gate_report, "complete_valid_prejump_odds_races") < MIN_COMPLETE_VALID_PREJUMP_ODDS_RACES:
        blockers.append("odds_research_gate_complete_valid_races_below_min")
    if int_value(odds_gate_report, "source_url_rows_missing") > 0:
        blockers.append("odds_research_gate_source_url_coverage_not_100_pct")
    source_url_coverage = metric_value(odds_gate_report, "source_url_coverage_pct")
    if source_url_coverage is not None and source_url_coverage < 100.0:
        blockers.append("odds_research_gate_source_url_coverage_not_100_pct")
    odds_by_key = {
        identity_key(row): row
        for row in odds_rows
        if row.get("odds_match_status") == "valid_pre_jump_dog_odds"
    }
    comparison_rows, rejected_rows = with_candidate_probabilities(joined_rows, odds_by_key)
    if rejected_rows:
        blockers.append("joined_rows_missing_valid_prejump_odds")
    if not comparison_rows:
        blockers.append("no_comparable_joined_rows")

    candidates = {
        "stage2_no_odds_challenger": metrics_for_probability(
            comparison_rows,
            "stage2_no_odds_probability",
        ),
        "market_only_implied_probability_baseline": metrics_for_probability(
            comparison_rows,
            "market_only_probability",
        ),
        "odds_augmented_challenger": metrics_for_probability(
            comparison_rows,
            "odds_augmented_probability",
        ),
        "probability_blend_calibration_candidate": metrics_for_probability(
            comparison_rows,
            "probability_blend_candidate_probability",
        ),
    }
    baseline = candidates["stage2_no_odds_challenger"]
    candidate_gates: dict[str, dict[str, Any]] = {}
    for name, metrics in candidates.items():
        if name == "stage2_no_odds_challenger":
            continue
        gate_blockers = candidate_blockers(
            baseline,
            metrics,
            min_top1_delta=min_top1_delta,
            max_box1_top_pick_share=max_box1_top_pick_share,
            max_probability_sum_error=max_probability_sum_error,
        )
        candidate_gates[name] = {
            "status": "PASS" if not gate_blockers else "BLOCKED",
            "blockers": gate_blockers,
        }
    passing = [
        (name, candidates[name])
        for name, gate in candidate_gates.items()
        if gate["status"] == "PASS"
    ]
    passing.sort(
        key=lambda item: (
            metric_value(item[1], "top1") or -1.0,
            metric_value(item[1], "top3") or -1.0,
            -(metric_value(item[1], "mean_winner_rank") or 999.0),
            -(metric_value(item[1], "logloss") or 999.0),
        ),
        reverse=True,
    )
    best_name = passing[0][0] if passing else None
    if not passing:
        blockers.append("no_odds_candidate_passed_rank_accuracy_guardrails")
    if protected_before != protected_after:
        blockers.append("protected_paths_changed")
    final_status = ODDS_AUGMENTED_MODEL_READY_FOR_PR_REVIEW if not blockers else ODDS_AUGMENTED_MODEL_BLOCKED
    ev = ev_diagnostics(
        comparison_rows,
        {
            "odds_augmented_challenger": "odds_augmented_probability",
            "probability_blend_calibration_candidate": "probability_blend_candidate_probability",
            "market_only_implied_probability_baseline": "market_only_probability",
        }.get(best_name or "", "stage2_no_odds_probability"),
    )
    report = {
        "schema_version": "odds_augmented_challenger_report_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": final_status,
        "odds_research_gate_status": odds_gate_report.get("status"),
        "joined_runner_rows": len(joined_rows),
        "comparable_runner_rows": len(comparison_rows),
        "rejected_joined_rows": rejected_rows,
        "activation_blockers": list(dict.fromkeys(blockers)),
        "baseline_metrics": baseline,
        "candidate_metrics": candidates.get(best_name or "", {}),
        "best_rank_accuracy_candidate": best_name,
        "model_comparisons": candidates,
        "candidate_gates": candidate_gates,
        "metrics_required": [
            "top1",
            "top3",
            "mean_winner_rank",
            "brier",
            "logloss",
            "calibration_slope_intercept",
            "box1_top_pick_share",
            "probability_sum_error",
        ],
        "promotion_boundary": {
            "promotion_pr_allowed": final_status == ODDS_AUGMENTED_MODEL_READY_FOR_PR_REVIEW,
            "direct_registry_mutation_allowed": False,
            "production_pointer_update_allowed": False,
            "odds_can_override_failed_accuracy_gate": False,
            "ev_can_override_failed_accuracy_gate": False,
        },
        "odds_used_for_shadow_scoring": False,
        "ev_output_allowed": True,
        "betting_action_allowed": False,
        "report_only": True,
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "protected_paths_unchanged": protected_before == protected_after,
        "no_write_guarantees": no_write_guarantees(),
    }
    return report, ev


def build_summary(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Odds-Augmented Challenger Report",
            "",
            f"- Final status: `{report.get('final_status')}`",
            f"- Comparable runner rows: `{report.get('comparable_runner_rows')}`",
            f"- Best candidate: `{report.get('best_rank_accuracy_candidate')}`",
            f"- Activation blockers: `{report.get('activation_blockers')}`",
            "",
            "Report-only. No betting advice, stakes, actions, model artifacts, registry writes, DB writes, labels, snapshots, production predictions, TGR, or production pointer updates were written.",
            "",
        ]
    )


def run_odds_augmented_report(
    *,
    joined_predictions: Path,
    odds_snapshot: Path,
    odds_gate_report_path: Path,
    output_dir: Path | None = None,
    min_top1_delta: float = 0.0,
    max_box1_top_pick_share: float = 0.35,
    max_probability_sum_error: float = 1e-6,
) -> dict[str, Any]:
    generated_at = datetime.now().astimezone()
    output_dir = output_dir or ROOT / "artifacts/full_evidence_orchestration_20260525" / f"odds_augmented_challenger_{now_id(generated_at)}"
    output_dir = unique_dir(assert_output_dir_safe(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    joined_path = input_jsonl_path(joined_predictions, "joined_shadow_predictions.jsonl")
    odds_path = input_jsonl_path(odds_snapshot, "shadow_odds_snapshot.jsonl")
    protected_before = protected_hashes()
    report, ev = build_report(
        joined_rows=read_jsonl(joined_path),
        odds_rows=read_jsonl(odds_path),
        odds_gate_report=load_json(odds_gate_report_path),
        generated_at=generated_at,
        min_top1_delta=min_top1_delta,
        max_box1_top_pick_share=max_box1_top_pick_share,
        max_probability_sum_error=max_probability_sum_error,
        protected_before=protected_before,
    )
    write_json(output_dir / "odds_augmented_challenger_report.json", report)
    write_json(output_dir / "report_only_ev_diagnostics.json", ev)
    write_text(output_dir / "SUMMARY.md", build_summary(report))
    write_text(output_dir / "final_status.txt", str(report["final_status"]) + "\n")
    return {
        "output_dir": relpath(output_dir),
        "final_status": report["final_status"],
        "best_rank_accuracy_candidate": report["best_rank_accuracy_candidate"],
        "activation_blockers": report["activation_blockers"],
        "protected_paths_unchanged": report["protected_paths_unchanged"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joined-predictions", required=True, type=Path)
    parser.add_argument("--odds-snapshot", required=True, type=Path)
    parser.add_argument("--odds-gate-report", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-top1-delta", type=float, default=0.0)
    parser.add_argument("--max-box1-top-pick-share", type=float, default=0.35)
    parser.add_argument("--max-probability-sum-error", type=float, default=1e-6)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_odds_augmented_report(
        joined_predictions=args.joined_predictions,
        odds_snapshot=args.odds_snapshot,
        odds_gate_report_path=args.odds_gate_report,
        output_dir=args.output_dir,
        min_top1_delta=args.min_top1_delta,
        max_box1_top_pick_share=args.max_box1_top_pick_share,
        max_probability_sum_error=args.max_probability_sum_error,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
