#!/usr/bin/env python3
"""Build stratified error analysis for no-box rolling pairwise predictions.

This is report-only analysis. Prediction rows remain no-box model inputs. When
a DB path is provided, the script opens it in read-only/query-only mode only to
add post-hoc analysis dimensions such as target distance and actual winner box.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/"
SCHEMA_VERSION = "no_box_pairwise_rolling_stratified_error_analysis_v1"
WRITES_PERFORMED = {
    "db_write": False,
    "label_write": False,
    "metadata_write": False,
    "official_fetch": False,
    "snapshot_mutation": False,
    "manifest_mutation": False,
    "model_training": False,
    "model_persistence": False,
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
    "window_id",
    "venue",
    "distance_bucket",
    "distance",
    "winner_box",
    "winner_box_bucket",
    "source_bucket",
    "field_size",
    "winner_rank",
    "winner_rank_bucket",
    "candidate_kind",
    "field_scope",
    "field_complete_for_ranking",
    "feature_join_status",
    "history_feature_join_status",
]
CSV_FIELDS = [
    "model",
    "dimension",
    "value",
    "race_count",
    "top1_accuracy",
    "top3_hit_rate",
    "mean_winner_rank",
    "top1_miss_count",
    "top3_miss_count",
    "top1_miss_examples",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("json_root_not_object")
    return payload


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


def _repo_output_path(path: Path, root: Path | None = None) -> tuple[Path, str]:
    root = root or ROOT
    logical = path.expanduser()
    if not logical.is_absolute():
        logical = root / logical
    resolved = logical.resolve(strict=False)
    try:
        relative = resolved.relative_to(root.resolve(strict=False)).as_posix()
    except ValueError as exc:
        raise ValueError(f"output_dir_must_be_inside_repo:{logical}") from exc
    return resolved, relative


def _repo_relative_text(path: Path, root: Path | None = None) -> str:
    return _repo_output_path(path, root)[1]


def _assert_output_dir_safe(output_dir: Path) -> Path:
    resolved, relative = _repo_output_path(output_dir)
    if not relative.startswith(ALLOWED_OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_under_artifacts:{relative}")
    return resolved


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    parsed = _safe_float(value)
    return int(parsed) if parsed is not None else None


def _name_key(value: Any) -> str:
    text = re.sub(r"^\s*\d{1,2}\s*[\.\):-]\s*", "", str(value or "").strip())
    text = text.replace('"', "").replace("'", "").replace("`", "")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    return re.sub(r"\s+", " ", text)


def _distance_bucket(distance: Any) -> str:
    parsed = _safe_float(distance)
    if parsed is None:
        return "DATA_MISSING"
    if parsed < 400:
        return "sprint_lt_400"
    if parsed < 500:
        return "sprint_400_499"
    if parsed < 600:
        return "middle_500_599"
    return "staying_600_plus"


def _box_bucket(box: Any) -> str:
    parsed = _safe_int(box)
    if parsed is None:
        return "DATA_MISSING"
    if parsed <= 2:
        return "inside_1_2"
    if parsed <= 5:
        return "middle_3_5"
    return "outside_6_plus"


def _winner_rank_bucket(rank: int) -> str:
    if rank <= 1:
        return "rank_1"
    if rank == 2:
        return "rank_2"
    if rank == 3:
        return "rank_3"
    return "rank_4_plus"


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    resolved = db_path.expanduser().resolve()
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _race_db_metadata(
    conn: sqlite3.Connection,
    race_ids: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not race_ids:
        return {}, {"quick_check": None, "race_metadata_rows": 0, "dog_race_data_rows": 0}
    quick_check = conn.execute("PRAGMA quick_check").fetchone()
    placeholders = ",".join("?" for _ in race_ids)
    race_meta = {
        str(row["race_id"]): dict(row)
        for row in conn.execute(
            f"""
            SELECT race_id, distance, grade, winner_name, winner_source, results_status
            FROM race_metadata
            WHERE race_id IN ({placeholders})
            """,
            list(race_ids),
        )
    }
    dog_rows_by_race: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dog_count = 0
    for row in conn.execute(
        f"""
        SELECT race_id, dog_name, box_number, data_source
        FROM dog_race_data
        WHERE race_id IN ({placeholders})
        """,
        list(race_ids),
    ):
        dog_count += 1
        dog_rows_by_race[str(row["race_id"])].append(dict(row))
    metadata: dict[str, dict[str, Any]] = {}
    for race_id in race_ids:
        meta = dict(race_meta.get(race_id) or {})
        metadata[race_id] = {
            "distance": meta.get("distance"),
            "grade": meta.get("grade"),
            "winner_source": meta.get("winner_source"),
            "results_status": meta.get("results_status"),
            "dog_rows": dog_rows_by_race.get(race_id, []),
        }
    return metadata, {
        "quick_check": quick_check[0] if quick_check else None,
        "race_metadata_rows": len(race_meta),
        "dog_race_data_rows": dog_count,
    }


def _validate_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    failures = []
    for index, row in enumerate(rows, start=1):
        forbidden = sorted(set(row).intersection(FORBIDDEN_ROW_FIELDS))
        if forbidden:
            failures.append(f"forbidden_prediction_row_fields:{index}:{','.join(forbidden)}")
        for flag in (
            "box_features_allowed",
            "finish_order_labels_allowed",
            "top3_labels_allowed",
            "official_safe_label_candidate",
            "label_write_approved",
        ):
            if flag in row and row.get(flag) is not False:
                failures.append(f"prediction_row_flag_not_false:{index}:{flag}")
    return failures


def _winner_db_enrichment(
    *,
    winner: Mapping[str, Any],
    race_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    winner_key = _name_key(winner.get("dog_name_key") or winner.get("dog_name"))
    matched_rows = [
        row
        for row in race_metadata.get("dog_rows", [])
        if _name_key(row.get("dog_name")) == winner_key
    ]
    matched = matched_rows[0] if len(matched_rows) == 1 else {}
    winner_source = race_metadata.get("winner_source") or matched.get("data_source")
    return {
        "distance": race_metadata.get("distance"),
        "distance_bucket": _distance_bucket(race_metadata.get("distance")),
        "winner_box": (
            str(_safe_int(matched.get("box_number")))
            if _safe_int(matched.get("box_number")) is not None
            else "DATA_MISSING"
        ),
        "winner_box_bucket": _box_bucket(matched.get("box_number")),
        "source_bucket": str(winner_source or "DATA_MISSING"),
        "db_winner_name_match_count": len(matched_rows),
        "db_enrichment_status": (
            "MATCHED_WINNER_ROW"
            if len(matched_rows) == 1
            else "WINNER_ROW_NOT_MATCHED"
            if not matched_rows
            else "WINNER_ROW_AMBIGUOUS"
        ),
    }


def _race_records(
    *,
    rows: Sequence[Mapping[str, Any]],
    race_metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    failures = _validate_prediction_rows(rows)
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        model = str(row.get("model") or "")
        window_id = str(row.get("window_id") or "DATA_MISSING")
        race_id = str(row.get("race_id") or "")
        if not model or not race_id:
            failures.append("prediction_row_missing_model_or_race_id")
            continue
        grouped[(model, window_id, race_id)].append(row)

    records = []
    for (model, window_id, race_id), race_rows in grouped.items():
        winners = [row for row in race_rows if int(row.get("actual_win") or 0) == 1]
        if len(winners) != 1:
            failures.append(f"winner_count_not_one:{model}:{window_id}:{race_id}:{len(winners)}")
            continue
        winner = winners[0]
        winner_rank = int(_safe_int(winner.get("predicted_rank")) or 999999)
        enrichment = _winner_db_enrichment(
            winner=winner,
            race_metadata=race_metadata.get(race_id) or {},
        )
        record = {
            "model": model,
            "window_id": window_id,
            "race_id": race_id,
            "race_date": winner.get("race_date"),
            "venue": str(winner.get("venue") or "DATA_MISSING"),
            "field_size": len(race_rows),
            "winner_rank": winner_rank,
            "winner_rank_bucket": _winner_rank_bucket(winner_rank),
            "candidate_kind": str(winner.get("candidate_kind") or "DATA_MISSING"),
            "field_scope": str(winner.get("field_scope") or "DATA_MISSING"),
            "field_complete_for_ranking": str(winner.get("field_complete_for_ranking") is True),
            "feature_join_status": str(winner.get("feature_join_status") or "DATA_MISSING"),
            "history_feature_join_status": str(
                winner.get("history_feature_join_status") or "DATA_MISSING"
            ),
            "top1_hit": winner_rank == 1,
            "top3_hit": winner_rank <= 3,
            **enrichment,
        }
        records.append(record)
    return records, failures


def _summarize_group(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    race_count = len(records)
    if not race_count:
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
        "mean_winner_rank": (
            sum(float(record.get("winner_rank") or 0) for record in records) / race_count
        ),
        "top1_miss_count": race_count - top1_hits,
        "top3_miss_count": race_count - top3_hits,
        "top1_miss_examples": miss_examples,
    }


def build_stratified_error_analysis(
    *,
    rolling_report: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
    predictions_path: str | None = None,
    rolling_report_path: str | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    race_ids = sorted({str(row.get("race_id") or "") for row in prediction_rows if row.get("race_id")})
    db_summary: dict[str, Any] = {
        "db_path": str(db_path.expanduser().resolve()) if db_path else None,
        "read_only": bool(db_path),
        "query_only": bool(db_path),
        "quick_check": None,
        "race_metadata_rows": 0,
        "dog_race_data_rows": 0,
    }
    race_metadata: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    if db_path:
        with _connect_read_only(db_path) as conn:
            race_metadata, db_summary_update = _race_db_metadata(conn, race_ids)
            db_summary.update(db_summary_update)
            if db_summary.get("quick_check") != "ok":
                failures.append("db_quick_check_failed")

    race_records, row_failures = _race_records(
        rows=prediction_rows,
        race_metadata=race_metadata,
    )
    failures.extend(row_failures)
    missing_dimension_counts = {
        "distance": sum(1 for record in race_records if record.get("distance_bucket") == "DATA_MISSING"),
        "winner_box": sum(1 for record in race_records if record.get("winner_box") == "DATA_MISSING"),
        "source_bucket": sum(1 for record in race_records if record.get("source_bucket") == "DATA_MISSING"),
    }
    by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in race_records:
        by_model[str(record.get("model"))].append(record)

    models = {}
    flat_rows = []
    for model, model_records in sorted(by_model.items()):
        dimension_summaries = {}
        for dimension in DIMENSIONS:
            values: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for record in model_records:
                values[str(record.get(dimension) or "DATA_MISSING")].append(record)
            dimension_summaries[dimension] = {}
            for value, grouped_records in sorted(values.items()):
                summary = _summarize_group(grouped_records)
                dimension_summaries[dimension][value] = summary
                flat_rows.append(
                    {
                        "model": model,
                        "dimension": dimension,
                        "value": value,
                        "top1_miss_examples": "|".join(
                            str(item) for item in summary.get("top1_miss_examples") or []
                        ),
                        **{field: summary.get(field) for field in CSV_FIELDS[3:-1]},
                    }
                )
        models[model] = {
            "overall": _summarize_group(model_records),
            "dimensions": dimension_summaries,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": (
            "REPORT_ONLY_PAIRWISE_ROLLING_STRATIFIED_ERROR_ANALYSIS"
            if not failures
            else "REPORT_ONLY_PAIRWISE_ROLLING_STRATIFIED_ERROR_ANALYSIS_WITH_FAILURES"
        ),
        "failures": failures,
        "report_only": True,
        "write_ready": False,
        "label_write_approved": False,
        "model_training_performed": False,
        "model_promotion_allowed": False,
        "writes_performed": dict(WRITES_PERFORMED),
        "rolling_report_path": rolling_report_path,
        "predictions_path": predictions_path,
        "source_report_status": rolling_report.get("status"),
        "source_sample_size_status": rolling_report.get("sample_size_status"),
        "source_reserved_final_races": (rolling_report.get("rolling_window_policy") or {}).get(
            "reserved_final_races"
        ),
        "source_reserved_races_predicted": (rolling_report.get("rolling_window_policy") or {}).get(
            "reserved_races_predicted"
        ),
        "db_enrichment": db_summary,
        "dimensions": list(DIMENSIONS),
        "missing_dimension_counts": missing_dimension_counts,
        "summary": {
            "model_count": len(models),
            "race_records": len(race_records),
            "prediction_rows": len(prediction_rows),
            "unique_races": len({str(record.get("race_id")) for record in race_records}),
            "window_count": len({str(record.get("window_id")) for record in race_records}),
        },
        "models": models,
        "race_records": race_records,
        "csv_rows": flat_rows,
    }


def write_outputs(output_dir: Path, analysis: Mapping[str, Any]) -> None:
    output_dir = _assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = list(analysis.get("csv_rows") or [])
    json_payload = {key: value for key, value in analysis.items() if key != "csv_rows"}
    (output_dir / "no_box_pairwise_rolling_stratified_error_analysis.json").write_text(
        json.dumps(json_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "no_box_pairwise_rolling_stratified_error_analysis.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    summary = analysis.get("summary") or {}
    missing = analysis.get("missing_dimension_counts") or {}
    lines = [
        "# No-Box Pairwise Rolling Stratified Error Analysis",
        "",
        f"Status: `{analysis.get('status')}`.",
        "",
        "No labels, DB rows, registries, model pointers, TGR settings, or betting/EV actions were changed.",
        "",
        "## Summary",
        "",
        f"- Prediction rows: `{summary.get('prediction_rows')}`",
        f"- Race records: `{summary.get('race_records')}`",
        f"- Windows: `{summary.get('window_count')}`",
        f"- DB quick check: `{(analysis.get('db_enrichment') or {}).get('quick_check')}`",
        f"- Missing distance records: `{missing.get('distance')}`",
        f"- Missing winner-box records: `{missing.get('winner_box')}`",
        f"- Missing source-bucket records: `{missing.get('source_bucket')}`",
        "",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rolling-report", required=True)
    parser.add_argument("--predictions-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--db")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report_path = Path(args.rolling_report).expanduser().resolve()
    predictions_path = Path(args.predictions_jsonl).expanduser().resolve()
    analysis = build_stratified_error_analysis(
        rolling_report=_load_json(report_path),
        prediction_rows=_load_jsonl(predictions_path),
        rolling_report_path=str(report_path),
        predictions_path=str(predictions_path),
        db_path=Path(args.db) if args.db else None,
    )
    write_outputs(Path(args.output_dir), analysis)
    print(json.dumps({"status": analysis["status"], "summary": analysis["summary"], "missing_dimension_counts": analysis["missing_dimension_counts"]}, indent=2, sort_keys=True))
    return 1 if analysis["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
