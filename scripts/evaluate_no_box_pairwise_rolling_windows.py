#!/usr/bin/env python3
"""Evaluate no-box pairwise ranking over multiple rolling temporal windows.

This is a report-only companion to evaluate_no_box_pairwise_ranking_smoke. It
uses fixed rolling train/eval windows over earlier races and can reserve the
last races as untouched holdout evidence. It never persists or promotes a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_no_box_pairwise_ranking_smoke import (
    FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL,
    WRITES_PERFORMED,
    _assert_output_dir_safe,
    _group_by_race,
    _load_json,
    _load_jsonl,
    _predict_race,
    _race_sort_key,
    _ranking_metrics,
    _source_packet_rejection,
    _train_pairwise_weights,
    _validate_rows,
)


SCHEMA_VERSION = "no_box_pairwise_rolling_windows_v1"
PREDICTION_FIELDS = [
    "window_id",
    "model",
    "race_id",
    "race_date",
    "venue",
    "race_number",
    "dog_name_key",
    "dog_name",
    "score",
    "predicted_rank",
    "actual_win",
    "candidate_kind",
    "field_scope",
    "field_complete_for_ranking",
    "train_race_count",
    "feature_count",
]


def _ordered_races(rows: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped = _group_by_race(rows)
    return [
        list(race_rows)
        for _, race_rows in sorted(grouped.items(), key=lambda item: _race_sort_key(item[1]))
    ]


def _race_ref(race_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = race_rows[0] if race_rows else {}
    return {
        "race_id": first.get("race_id"),
        "race_date": first.get("race_date"),
        "venue": first.get("venue"),
        "race_number": first.get("race_number"),
        "row_count": len(race_rows),
        "field_complete_for_ranking": all(
            row.get("field_complete_for_ranking") is True for row in race_rows
        ),
    }


def _window_predictions(
    *,
    ordered: Sequence[Sequence[Mapping[str, Any]]],
    feature_columns: Sequence[str],
    train_races: int,
    eval_races: int,
    step_races: int,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    max_start = len(ordered) - train_races - eval_races
    if max_start < 0:
        return [], []
    for window_number, train_start in enumerate(range(0, max_start + 1, step_races), start=1):
        train_slice = list(ordered[train_start : train_start + train_races])
        eval_slice = list(
            ordered[train_start + train_races : train_start + train_races + eval_races]
        )
        weights, means, scales, training_summary = _train_pairwise_weights(
            train_slice,
            feature_columns,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        window_id = f"window_{window_number:02d}"
        window_predictions = []
        for race_rows in eval_slice:
            race_predictions = _predict_race(
                race_rows=race_rows,
                feature_columns=feature_columns,
                weights=weights,
                means=means,
                scales=scales,
                train_race_count=len(train_slice),
            )
            for row in race_predictions:
                row["window_id"] = window_id
            window_predictions.extend(race_predictions)
        metrics = _ranking_metrics(window_predictions)
        predictions.extend(window_predictions)
        windows.append(
            {
                "window_id": window_id,
                "train_start_index": train_start,
                "train_end_index": train_start + train_races - 1,
                "eval_start_index": train_start + train_races,
                "eval_end_index": train_start + train_races + eval_races - 1,
                "train_races": [_race_ref(race_rows) for race_rows in train_slice],
                "eval_races": [_race_ref(race_rows) for race_rows in eval_slice],
                "training_summary": training_summary,
                "metrics": metrics,
            }
        )
    return predictions, windows


def _window_metric_summary(windows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    top1 = [
        float((window.get("metrics") or {}).get("top1_accuracy"))
        for window in windows
        if (window.get("metrics") or {}).get("top1_accuracy") is not None
    ]
    top3 = [
        float((window.get("metrics") or {}).get("top3_hit_rate"))
        for window in windows
        if (window.get("metrics") or {}).get("top3_hit_rate") is not None
    ]
    return {
        "window_count": len(windows),
        "top1_min": min(top1) if top1 else None,
        "top1_max": max(top1) if top1 else None,
        "top1_mean_across_windows": sum(top1) / len(top1) if top1 else None,
        "top1_range": (max(top1) - min(top1)) if top1 else None,
        "top3_min": min(top3) if top3 else None,
        "top3_max": max(top3) if top3 else None,
        "top3_mean_across_windows": sum(top3) / len(top3) if top3 else None,
        "top3_range": (max(top3) - min(top3)) if top3 else None,
    }


def evaluate_rolling_windows(
    *,
    feature_join_packet: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    feature_join_packet_path: str | None = None,
    rows_path: str | None = None,
    expected_races: int | None = None,
    train_races: int = 10,
    eval_races: int = 5,
    step_races: int = 5,
    reserve_final_races: int = 5,
    min_windows: int = 2,
    epochs: int = 15,
    learning_rate: float = 0.05,
    l2: float = 0.001,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validation = _validate_rows(
        feature_join_packet=feature_join_packet,
        rows=rows,
        expected_races=expected_races,
    )
    ordered_all = _ordered_races(rows)
    reserve_count = max(0, int(reserve_final_races))
    if reserve_count:
        reserved_races = ordered_all[-reserve_count:]
        rolling_races = ordered_all[:-reserve_count]
    else:
        reserved_races = []
        rolling_races = ordered_all
    source_status, source_reason = _source_packet_rejection(feature_join_packet)
    status = "REPORT_ONLY_PAIRWISE_ROLLING_WINDOWS_EVALUATED"
    predictions: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    aggregate_metrics: dict[str, Any] | None = None
    metric_summary: dict[str, Any] | None = None
    if source_status:
        status = source_status.replace("PAIRWISE_RANKING", "PAIRWISE_ROLLING_WINDOWS")
    elif validation["status"] != "PASS":
        status = "REPORT_ONLY_PAIRWISE_ROLLING_WINDOWS_FAILED_CONTRACT"
    elif not validation["usable_feature_columns"]:
        status = "REPORT_ONLY_PAIRWISE_ROLLING_WINDOWS_INSUFFICIENT_FEATURES"
    elif train_races <= 0 or eval_races <= 0 or step_races <= 0:
        status = "REPORT_ONLY_PAIRWISE_ROLLING_WINDOWS_FAILED_CONTRACT"
        validation["failures"].append("rolling_window_parameters_must_be_positive")
        validation["status"] = "FAIL"
    elif len(rolling_races) < train_races + eval_races:
        status = "REPORT_ONLY_PAIRWISE_ROLLING_WINDOWS_INSUFFICIENT_DATA"
    else:
        predictions, windows = _window_predictions(
            ordered=rolling_races,
            feature_columns=validation["usable_feature_columns"],
            train_races=train_races,
            eval_races=eval_races,
            step_races=step_races,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        aggregate_metrics = _ranking_metrics(predictions)
        metric_summary = _window_metric_summary(windows)
        if len(windows) < min_windows:
            status = "REPORT_ONLY_PAIRWISE_ROLLING_WINDOWS_INSUFFICIENT_DATA"

    complete_field_count = validation["complete_field_races"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "report_only": True,
        "writes_performed": dict(WRITES_PERFORMED),
        "feature_join_packet": feature_join_packet_path,
        "rows_jsonl": rows_path,
        "source_packet": {
            "schema_version": feature_join_packet.get("schema_version"),
            "status": feature_join_packet.get("status"),
            "history_db_fill_policy": (feature_join_packet.get("summary") or {}).get(
                "history_db_fill_policy"
            ),
            "label_proxy_audit_status": (
                (feature_join_packet.get("summary") or {}).get("label_proxy_audit") or {}
            ).get("status"),
            "rejection_reason": source_reason,
        },
        "validation": validation,
        "rolling_window_policy": {
            "race_sort_key": ["race_date", "race_id"],
            "train_races": train_races,
            "eval_races": eval_races,
            "step_races": step_races,
            "min_windows": min_windows,
            "available_races": len(ordered_all),
            "rolling_candidate_races": len(rolling_races),
            "reserved_final_races": len(reserved_races),
            "reserved_race_refs": [_race_ref(race_rows) for race_rows in reserved_races],
            "reserved_races_predicted": False,
            "second_holdout_files_written": False,
            "note": "Reserved races are excluded from rolling fit/eval; this script does not touch frozen second-holdout prediction files.",
        },
        "ranking_model": {
            "model_key": "report_local_pairwise_logistic_ranker",
            "report_local_pairwise_fit_performed": bool(predictions),
            "model_persistence_performed": False,
            "model_training_status": "REPORT_LOCAL_TRANSIENT_FIT_ONLY_NO_MODEL_PERSISTENCE",
            "probability_output": "not_applicable",
            "feature_columns": validation["usable_feature_columns"],
        },
        "aggregate_metrics": aggregate_metrics,
        "window_metric_summary": metric_summary,
        "windows": windows,
        "race_grouped_complete_field_gate": {
            "status": (
                "READY_FOR_RACE_GROUPED_RANKING_EXPERIMENT"
                if complete_field_count >= 100
                else "SKIPPED_INSUFFICIENT_COMPLETE_FIELD_RACES"
            ),
            "complete_field_races": complete_field_count,
            "required_complete_field_races": 100,
            "note": "Rolling actual-win windows can be diagnostic while complete-field ranking evidence remains underpowered.",
        },
        "minimums": {
            "rolling_temporal_actual_win_races": 50,
            "race_grouped_ranking_complete_field_races": 100,
            "rolling_window_min_windows": min_windows,
        },
        "sample_size_status": (
            "UNDERPOWERED_BELOW_50_ACTUAL_WIN_RACES"
            if validation["race_count"] < 50
            else "MEETS_MIN_ROLLING_TEMPORAL_ACTUAL_WIN_RACES"
        ),
        "forbidden_without_explicit_approval": list(FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL),
        "recommended_next_action": (
            "do_not_use_history_db_enriched_metrics_until_label_proxy_risk_is_reviewed"
            if "REJECTED_LEAKAGE_RISK" in status
            else "expand_official_safe_labels_then_repeat_rolling_temporal_no_box_ranking"
        ),
    }
    return report, predictions


def write_outputs(output_dir: Path, report: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]]) -> None:
    output_dir = _assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "no_box_pairwise_rolling_windows_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "no_box_pairwise_rolling_window_predictions.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output_dir / "no_box_pairwise_rolling_window_predictions.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for row in predictions:
            writer.writerow({field: row.get(field) for field in PREDICTION_FIELDS})
    validation = report.get("validation") or {}
    policy = report.get("rolling_window_policy") or {}
    aggregate = report.get("aggregate_metrics") or {}
    window_summary = report.get("window_metric_summary") or {}
    complete_gate = report.get("race_grouped_complete_field_gate") or {}
    lines = [
        "# No-Box Pairwise Rolling Windows",
        "",
        f"Status: `{report.get('status')}`.",
        "",
        "No DB writes, label writes, metadata writes, official fetches, snapshot or manifest mutations, model persistence, production model training, registry updates, promotions, TGR enablement, betting decisions, or EV actions occurred.",
        "",
        "## Contract",
        "",
        f"- Validation status: `{validation.get('status')}`",
        f"- Races: `{validation.get('race_count')}`",
        f"- Rows: `{validation.get('row_count')}`",
        f"- Usable features: `{validation.get('usable_feature_count')}`",
        f"- Complete-field gate: `{complete_gate.get('status')}`",
        f"- Sample-size status: `{report.get('sample_size_status')}`",
        "",
        "## Rolling Policy",
        "",
        f"- Train races per window: `{policy.get('train_races')}`",
        f"- Eval races per window: `{policy.get('eval_races')}`",
        f"- Step races: `{policy.get('step_races')}`",
        f"- Reserved final races: `{policy.get('reserved_final_races')}`",
        f"- Reserved races predicted: `{policy.get('reserved_races_predicted')}`",
        "",
        "## Metrics",
        "",
        f"- Windows: `{window_summary.get('window_count')}`",
        f"- Aggregate evaluated races: `{aggregate.get('race_count')}`",
        f"- Aggregate Top1: `{aggregate.get('top1_accuracy')}`",
        f"- Aggregate Top3: `{aggregate.get('top3_hit_rate')}`",
        f"- Top1 range across windows: `{window_summary.get('top1_range')}`",
        f"- Top3 range across windows: `{window_summary.get('top3_range')}`",
        "",
        "## Next",
        "",
        str(report.get("recommended_next_action")),
        "",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-join-packet", required=True)
    parser.add_argument("--rows-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-races", type=int)
    parser.add_argument("--train-races", type=int, default=10)
    parser.add_argument("--eval-races", type=int, default=5)
    parser.add_argument("--step-races", type=int, default=5)
    parser.add_argument("--reserve-final-races", type=int, default=5)
    parser.add_argument("--min-windows", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packet_path = Path(args.feature_join_packet).expanduser().resolve()
    rows_path = Path(args.rows_jsonl).expanduser().resolve()
    report, predictions = evaluate_rolling_windows(
        feature_join_packet=_load_json(packet_path),
        rows=_load_jsonl(rows_path),
        feature_join_packet_path=str(packet_path),
        rows_path=str(rows_path),
        expected_races=args.expected_races,
        train_races=args.train_races,
        eval_races=args.eval_races,
        step_races=args.step_races,
        reserve_final_races=args.reserve_final_races,
        min_windows=args.min_windows,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    write_outputs(Path(args.output_dir), report, predictions)
    print(
        json.dumps(
            {
                "status": report["status"],
                "validation": report["validation"],
                "rolling_window_policy": report["rolling_window_policy"],
                "aggregate_metrics": report["aggregate_metrics"],
                "window_metric_summary": report["window_metric_summary"],
                "sample_size_status": report["sample_size_status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["status"] in {
        "REPORT_ONLY_PAIRWISE_ROLLING_WINDOWS_FAILED_CONTRACT",
        "REPORT_ONLY_PAIRWISE_ROLLING_WINDOWS_REJECTED_LEAKAGE_RISK",
        "REPORT_ONLY_PAIRWISE_ROLLING_WINDOWS_REJECTED_SOURCE_PACKET_NOT_READY",
    }:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
