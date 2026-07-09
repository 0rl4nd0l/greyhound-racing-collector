#!/usr/bin/env python3
"""Build stratified error analysis for no-box actual-win smoke predictions.

This is report-only analysis over prediction artifacts. It does not write
labels, mutate databases, train models, promote models, enable TGR, or create
betting/EV actions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/"
SCHEMA_VERSION = "no_box_actual_win_stratified_error_analysis_v1"
WRITES_PERFORMED = {
    "db_write": False,
    "label_write": False,
    "metadata_write": False,
    "official_fetch": False,
    "snapshot_mutation": False,
    "manifest_mutation": False,
    "model_training": False,
    "registry_mutation": False,
    "promotion": False,
    "tgr_enablement": False,
    "betting_decision": False,
    "ev_action": False,
}
FORBIDDEN_ROW_FIELDS = {
    "box_number",
    "official_box_number",
    "db_box_number",
    "finish_position",
    "official_finish_position",
    "db_finish_position",
    "db_result_position",
    "result_position",
    "placing",
    "scraped_finish_position",
}
DIMENSIONS = [
    "venue",
    "field_size",
    "winner_rank",
    "winner_rank_bucket",
    "candidate_kind",
    "field_scope",
    "field_complete_for_ranking",
    "feature_join_status",
]
UNAVAILABLE_DIMENSIONS = {
    "distance": "DATA_MISSING: no distance field is carried in the no-box actual-win smoke rows",
    "box": "DATA_MISSING: box fields are intentionally excluded by the no-box contract",
    "source_bucket": "DATA_MISSING: source bucket is not carried into smoke prediction rows",
}
CSV_FIELDS = [
    "baseline",
    "dimension",
    "value",
    "race_count",
    "top1_accuracy",
    "top3_hit_rate",
    "mean_winner_rank",
    "mean_winner_probability",
    "brier",
    "top1_miss_count",
    "top3_miss_count",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"jsonl_row_not_object:{line_number}")
            rows.append(row)
    return rows


def _repo_relative_text(path: Path, root: Path | None = None) -> str:
    root = root or ROOT
    logical = path.expanduser()
    if not logical.is_absolute():
        logical = root / logical
    try:
        return logical.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
    except ValueError as exc:
        raise ValueError(f"output_dir_must_be_inside_repo:{logical}") from exc


def _assert_output_dir_safe(output_dir: Path) -> Path:
    relative = _repo_relative_text(output_dir)
    if not relative.startswith(ALLOWED_OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_under_artifacts:{relative}")
    return output_dir


def _safe_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _winner_rank_bucket(rank: int) -> str:
    if rank <= 1:
        return "rank_1"
    if rank == 2:
        return "rank_2"
    if rank == 3:
        return "rank_3"
    return "rank_4_plus"


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    failures = []
    for index, row in enumerate(rows, start=1):
        forbidden = sorted(set(row).intersection(FORBIDDEN_ROW_FIELDS))
        if forbidden:
            failures.append(f"forbidden_row_fields:{index}:{','.join(forbidden)}")
        for flag in (
            "box_features_allowed",
            "finish_order_labels_allowed",
            "top3_labels_allowed",
            "official_safe_label_candidate",
            "label_write_approved",
        ):
            if flag in row and row.get(flag) is not False:
                failures.append(f"row_flag_not_false:{index}:{flag}")
    return failures


def _race_records(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    failures = _validate_rows(rows)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        baseline = str(row.get("baseline") or "")
        race_id = str(row.get("race_id") or "")
        if not baseline or not race_id:
            failures.append("row_missing_baseline_or_race_id")
            continue
        grouped[(baseline, race_id)].append(row)

    races = []
    for (baseline, race_id), race_rows in grouped.items():
        winners = [row for row in race_rows if int(row.get("actual_win") or 0) == 1]
        if len(winners) != 1:
            failures.append(f"winner_count_not_one:{baseline}:{race_id}:{len(winners)}")
            continue
        winner = winners[0]
        winner_rank = int(_safe_float(winner.get("predicted_rank")))
        brier = sum(
            (_safe_float(row.get("probability")) - float(int(row.get("actual_win") or 0))) ** 2
            for row in race_rows
        ) / len(race_rows)
        races.append(
            {
                "baseline": baseline,
                "race_id": race_id,
                "venue": str(winner.get("venue") or "DATA_MISSING"),
                "field_size": len(race_rows),
                "winner_rank": winner_rank,
                "winner_rank_bucket": _winner_rank_bucket(winner_rank),
                "candidate_kind": str(winner.get("candidate_kind") or "DATA_MISSING"),
                "field_scope": str(winner.get("field_scope") or "DATA_MISSING"),
                "field_complete_for_ranking": str(
                    winner.get("field_complete_for_ranking") is True
                ),
                "feature_join_status": str(winner.get("feature_join_status") or "DATA_MISSING"),
                "winner_probability": _safe_float(winner.get("probability")),
                "top1_hit": winner_rank == 1,
                "top3_hit": winner_rank <= 3,
                "brier": brier,
            }
        )
    return races, failures


def _summarize_group(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    race_count = len(records)
    if race_count == 0:
        return {}
    top1_hits = sum(1 for record in records if record.get("top1_hit") is True)
    top3_hits = sum(1 for record in records if record.get("top3_hit") is True)
    miss_examples = [
        str(record.get("race_id"))
        for record in records
        if record.get("top1_hit") is not True
    ][:10]
    return {
        "race_count": race_count,
        "top1_accuracy": top1_hits / race_count,
        "top3_hit_rate": top3_hits / race_count,
        "mean_winner_rank": sum(float(record.get("winner_rank") or 0) for record in records) / race_count,
        "mean_winner_probability": (
            sum(float(record.get("winner_probability") or 0) for record in records) / race_count
        ),
        "brier": sum(float(record.get("brier") or 0) for record in records) / race_count,
        "top1_miss_count": race_count - top1_hits,
        "top3_miss_count": race_count - top3_hits,
        "top1_miss_examples": miss_examples,
    }


def build_stratified_error_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    predictions_path: str | None = None,
) -> dict[str, Any]:
    race_records, failures = _race_records(rows)
    by_baseline: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in race_records:
        by_baseline[str(record.get("baseline"))].append(record)

    baselines = {}
    flat_rows = []
    for baseline, records in sorted(by_baseline.items()):
        dimension_summaries = {}
        for dimension in DIMENSIONS:
            values: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for record in records:
                values[str(record.get(dimension) or "DATA_MISSING")].append(record)
            dimension_summaries[dimension] = {}
            for value, grouped_records in sorted(values.items()):
                summary = _summarize_group(grouped_records)
                dimension_summaries[dimension][value] = summary
                flat_rows.append(
                    {
                        "baseline": baseline,
                        "dimension": dimension,
                        "value": value,
                        **{field: summary.get(field) for field in CSV_FIELDS[3:]},
                    }
                )
        baselines[baseline] = {
            "overall": _summarize_group(records),
            "dimensions": dimension_summaries,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": "REPORT_ONLY_STRATIFIED_ERROR_ANALYSIS"
        if not failures
        else "REPORT_ONLY_STRATIFIED_ERROR_ANALYSIS_WITH_FAILURES",
        "failures": failures,
        "report_only": True,
        "write_ready": False,
        "label_write_approved": False,
        "model_training_performed": False,
        "model_promotion_allowed": False,
        "writes_performed": dict(WRITES_PERFORMED),
        "predictions_path": predictions_path,
        "dimensions": list(DIMENSIONS),
        "unavailable_dimensions": dict(UNAVAILABLE_DIMENSIONS),
        "summary": {
            "baseline_count": len(baselines),
            "race_records": len(race_records),
            "prediction_rows": len(rows),
        },
        "baselines": baselines,
        "csv_rows": flat_rows,
    }


def write_outputs(output_dir: Path, analysis: Mapping[str, Any]) -> None:
    output_dir = _assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = list(analysis.get("csv_rows") or [])
    json_payload = {key: value for key, value in analysis.items() if key != "csv_rows"}
    (output_dir / "no_box_smoke_stratified_error_analysis.json").write_text(
        json.dumps(json_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "no_box_smoke_stratified_error_analysis.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    (output_dir / "SUMMARY.md").write_text(
        "\n".join(
            [
                "# No-Box Smoke Stratified Error Analysis",
                "",
                f"- Status: `{analysis.get('status')}`",
                f"- Prediction rows: `{analysis.get('summary', {}).get('prediction_rows')}`",
                f"- Race records: `{analysis.get('summary', {}).get('race_records')}`",
                f"- Baselines: `{analysis.get('summary', {}).get('baseline_count')}`",
                "- Unavailable dimensions: `distance`, `box`, `source_bucket`",
                "",
                "No labels, DB rows, registries, model pointers, TGR settings, or betting/EV actions were changed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    predictions_path = Path(args.predictions_jsonl).expanduser().resolve()
    analysis = build_stratified_error_analysis(
        _load_jsonl(predictions_path),
        predictions_path=str(predictions_path),
    )
    write_outputs(Path(args.output_dir), analysis)
    print(json.dumps(analysis["summary"], indent=2, sort_keys=True))
    return 1 if analysis["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
