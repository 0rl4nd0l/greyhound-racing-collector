#!/usr/bin/env python3
"""Build a report-only time-split pre-race gated challenger packet.

This consumes the runner-level market residual matrix from a rolling comparison
packet. It trains gate selection on earlier race dates and evaluates the
selected gate on later race dates. The purpose is to test whether the pre-race
gated challenger lead survives chronological validation.

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
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from scripts.build_market_residual_challenger_packet import (  # noqa: E402
    collect_races,
    evaluate_candidate,
    evaluate_scored_races,
    finite_float,
    load_matrix,
    market_scores,
    metric_deltas,
    score_races,
)
from scripts.build_pre_race_gated_challenger_packet import (  # noqa: E402
    candidate_specs,
    evaluate_spec,
    gate_triggered_count,
    race_prediction_rows_for_fold,
    selection_sort_key,
)


DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = (
    "artifacts/full_evidence_orchestration_20260525/"
    "time_split_gated_challenger_"
)
REPORT_FILE = "time_split_gated_challenger_report.json"
SPLIT_SUMMARY_CSV = "time_split_summary.csv"
RACE_PREDICTIONS_CSV = "time_split_race_predictions.csv"
SUMMARY_FILE = "SUMMARY.md"
FINAL_READY = "TIME_SPLIT_GATED_CHALLENGER_REVIEW_READY"
FINAL_COLLECTING = "TIME_SPLIT_GATED_CHALLENGER_COLLECTING"
MIN_RACES_FOR_REVIEW = 100
DEFAULT_MIN_TRAIN_RACES = 20
DEFAULT_MIN_TEST_RACES = 5
DEFAULT_MIN_TRAIN_GATE_TRIGGERS = 1
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
        raise ValueError(f"output_dir_must_be_time_split_gated_challenger:{relative}")
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
        "schema_version": "time_split_gated_challenger_output_manifest_v1",
        "output_dir": relpath(output_dir),
        "files": files,
    }


def race_date(race: Mapping[str, Any]) -> str:
    return str(race.get("race_date") or "")


def group_dates(races: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for race in races:
        grouped.setdefault(race_date(race), []).append(race)
    return grouped


def time_split_review(
    races: Sequence[Mapping[str, Any]],
    *,
    min_train_races: int,
    min_test_races: int,
    min_train_gate_triggers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    specs = candidate_specs()
    dates = sorted(date for date in group_dates(races) if date)
    split_summaries: list[dict[str, Any]] = []
    scored_test_races: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for split_index, test_date in enumerate(dates[1:], start=1):
        train_dates = [date for date in dates if date < test_date]
        train_races = [race for race in races if race_date(race) in set(train_dates)]
        test_races = [race for race in races if race_date(race) == test_date]
        if len(train_races) < min_train_races:
            split_summaries.append(
                {
                    "split": split_index,
                    "status": "SKIPPED_TRAIN_RACE_COUNT_BELOW_MINIMUM",
                    "train_dates": ",".join(train_dates),
                    "test_date": test_date,
                    "train_races": len(train_races),
                    "test_races": len(test_races),
                }
            )
            continue
        if len(test_races) < min_test_races:
            split_summaries.append(
                {
                    "split": split_index,
                    "status": "SKIPPED_TEST_RACE_COUNT_BELOW_MINIMUM",
                    "train_dates": ",".join(train_dates),
                    "test_date": test_date,
                    "train_races": len(train_races),
                    "test_races": len(test_races),
                }
            )
            continue
        train_metrics = [evaluate_spec(train_races, spec) for spec in specs]
        evaluated = [
            item
            for item in train_metrics
            if item.get("status") == "EVALUATED"
            and (item.get("gate_triggered_race_count") or 0) >= min_train_gate_triggers
        ]
        if not evaluated:
            split_summaries.append(
                {
                    "split": split_index,
                    "status": "SKIPPED_NO_EVALUATED_TRAIN_CANDIDATES",
                    "train_dates": ",".join(train_dates),
                    "test_date": test_date,
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
                "split": split_index,
                "test_date": test_date,
                "selected_candidate_key": selected_spec.get("candidate_key"),
            }
            for item in test_scored
        )
        test_metrics = evaluate_scored_races(test_scored)
        split_summaries.append(
            {
                "split": split_index,
                "status": "EVALUATED",
                "train_dates": ",".join(train_dates),
                "test_date": test_date,
                "train_races": len(train_races),
                "test_races": len(test_races),
                "selected_candidate_key": selected_spec.get("candidate_key"),
                "gate_key": selected_spec.get("gate_key"),
                "gate_family": selected_spec.get("gate_family"),
                "score_mode": selected_spec.get("score_mode"),
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
        for row in race_prediction_rows_for_fold(
            test_scored,
            fold_index=split_index,
            selected_spec=selected_spec,
        ):
            row["split"] = split_index
            row["train_dates"] = ",".join(train_dates)
            row["test_date"] = test_date
            prediction_rows.append(row)

    metrics = evaluate_scored_races(scored_test_races)
    metrics["candidate_key"] = "time_split_pre_race_gated_challenger"
    metrics["family"] = "chronological_train_past_dates_test_next_date"
    metrics["evaluated_split_count"] = sum(
        1 for item in split_summaries if item.get("status") == "EVALUATED"
    )
    metrics["gate_triggered_test_race_count"] = sum(
        1 for row in prediction_rows if row.get("gate_triggered") is True
    )
    metrics["date_count"] = len(dates)
    return metrics, split_summaries, prediction_rows


def promotion_gate(
    *,
    market_metrics: Mapping[str, Any],
    challenger_metrics: Mapping[str, Any],
    min_races_for_review: int,
) -> dict[str, Any]:
    deltas = metric_deltas(market_metrics, challenger_metrics)
    blockers = [
        "report_only_time_split_gated_challenger_not_promotion_eligible",
        "requires_fresh_future_out_of_sample_packet",
    ]
    if (challenger_metrics.get("race_count") or 0) < min_races_for_review:
        blockers.append("time_split_test_race_count_below_review_floor")
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
            "report_only_time_split_gated_challenger_not_promotion_eligible",
            "requires_fresh_future_out_of_sample_packet",
        ],
        "candidate_minus_market": deltas,
        "blockers": blockers,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_split_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "split",
        "status",
        "train_dates",
        "test_date",
        "train_races",
        "test_races",
        "selected_candidate_key",
        "gate_key",
        "gate_family",
        "score_mode",
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
        "split",
        "train_dates",
        "test_date",
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


def summary_markdown(report: Mapping[str, Any]) -> str:
    gate = report.get("promotion_gate") or {}
    return "\n".join(
        [
            "# Time-Split Gated Challenger Review",
            "",
            f"Final status: `{report.get('final_status')}`",
            "",
            f"- Matrix rows: `{report.get('matrix_row_count')}`",
            f"- Accepted races: `{report.get('accepted_race_count')}`",
            f"- Race dates: `{report.get('race_dates')}`",
            f"- Evaluated splits: `{report.get('evaluated_split_count')}`",
            f"- Time-split test races: `{(report.get('time_split_metrics') or {}).get('race_count')}`",
            f"- Market top1 on same test races: `{(report.get('market_metrics_on_time_split_test_races') or {}).get('top1')}`",
            f"- Challenger top1: `{(report.get('time_split_metrics') or {}).get('top1')}`",
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
    min_train_races: int = DEFAULT_MIN_TRAIN_RACES,
    min_test_races: int = DEFAULT_MIN_TEST_RACES,
    min_train_gate_triggers: int = DEFAULT_MIN_TRAIN_GATE_TRIGGERS,
    min_races_for_review: int = MIN_RACES_FOR_REVIEW,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    output_dir = unique_dir(assert_output_dir_safe(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)

    matrix_rows = load_matrix(runner_matrix_csv)
    races, collection = collect_races(matrix_rows)
    race_dates = sorted(set(race_date(race) for race in races if race_date(race)))
    full_market_metrics = evaluate_candidate(
        races,
        {
            "candidate_key": "market_only_implied",
            "family": "market_only",
            "score_function": market_scores,
        },
    )
    time_split_metrics, split_rows, prediction_rows = time_split_review(
        races,
        min_train_races=min_train_races,
        min_test_races=min_test_races,
        min_train_gate_triggers=min_train_gate_triggers,
    )
    test_race_ids = {str(row.get("race_id")) for row in prediction_rows}
    test_races = [race for race in races if str(race.get("race_id")) in test_race_ids]
    market_test_metrics = evaluate_candidate(
        test_races,
        {
            "candidate_key": "market_only_implied_on_time_split_test_races",
            "family": "market_only",
            "score_function": market_scores,
        },
    )
    gate = promotion_gate(
        market_metrics=market_test_metrics,
        challenger_metrics=time_split_metrics,
        min_races_for_review=min_races_for_review,
    )

    blockers: list[str] = []
    if len(race_dates) < 2:
        blockers.append("fewer_than_two_race_dates")
    if time_split_metrics.get("status") != "EVALUATED":
        blockers.append("time_split_challenger_not_evaluated")

    report = {
        "schema_version": "time_split_gated_challenger_report_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": FINAL_READY if not blockers else FINAL_COLLECTING,
        "output_dir": relpath(output_dir),
        "runner_matrix_csv": relpath(runner_matrix_csv),
        "time_split_summary_csv": relpath(output_dir / SPLIT_SUMMARY_CSV),
        "time_split_race_predictions_csv": relpath(output_dir / RACE_PREDICTIONS_CSV),
        "matrix_row_count": len(matrix_rows),
        "accepted_race_count": len(races),
        "race_dates": race_dates,
        "minimum_races_for_review": min_races_for_review,
        "min_train_races": min_train_races,
        "min_test_races": min_test_races,
        "min_train_gate_triggers": min_train_gate_triggers,
        "evaluated_split_count": time_split_metrics.get("evaluated_split_count"),
        "collection": collection,
        "full_sample_market_metrics": full_market_metrics,
        "market_metrics_on_time_split_test_races": market_test_metrics,
        "time_split_metrics": time_split_metrics,
        "promotion_gate": gate,
        "blockers": blockers,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }
    write_json(output_dir / REPORT_FILE, report)
    write_split_summary_csv(output_dir / SPLIT_SUMMARY_CSV, split_rows)
    write_race_predictions_csv(output_dir / RACE_PREDICTIONS_CSV, prediction_rows)
    write_text(output_dir / SUMMARY_FILE, summary_markdown(report))
    write_text(output_dir / "final_status.txt", str(report["final_status"]) + "\n")
    write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-matrix-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-train-races", type=int, default=DEFAULT_MIN_TRAIN_RACES)
    parser.add_argument("--min-test-races", type=int, default=DEFAULT_MIN_TEST_RACES)
    parser.add_argument(
        "--min-train-gate-triggers",
        type=int,
        default=DEFAULT_MIN_TRAIN_GATE_TRIGGERS,
    )
    parser.add_argument("--min-races-for-review", type=int, default=MIN_RACES_FOR_REVIEW)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    generated_at = datetime.now().astimezone()
    output_dir = (
        args.output_dir
        or DEFAULT_EVIDENCE_ROOT / f"time_split_gated_challenger_{now_id(generated_at)}"
    )
    report = build_packet(
        runner_matrix_csv=args.runner_matrix_csv,
        output_dir=output_dir,
        min_train_races=args.min_train_races,
        min_test_races=args.min_test_races,
        min_train_gate_triggers=args.min_train_gate_triggers,
        min_races_for_review=args.min_races_for_review,
        generated_at=generated_at,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
