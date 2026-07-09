#!/usr/bin/env python3
"""Refresh no-box downstream diagnostics from rolling predictions and features.

This is report-only analysis for the greyhound accuracy packet. It rebuilds the
failure surface, tied-score triage, insufficient-variance gate, feature coverage
error matrix, and priority digest from supplied prediction/feature artifacts.
It opens SQLite only in read-only/query-only mode for post-hoc dimensions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_OUTPUT_PREFIX = Path("artifacts/full_evidence_orchestration_20260525")
WRITES_PERFORMED = {
    "db_write": False,
    "label_write": False,
    "metadata_write": False,
    "official_fetch": False,
    "snapshot_or_manifest_mutation": False,
    "model_training": False,
    "model_persistence": False,
    "registry_update": False,
    "promotion": False,
    "github_write": False,
    "tgr_enablement": False,
    "betting_decision": False,
    "ev_action": False,
    "source_pdf_write": False,
    "dataset_regeneration": False,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _assert_output_dir_safe(output_dir: Path) -> Path:
    root = ROOT.expanduser().resolve()
    candidate = output_dir.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"output_dir_must_be_inside_repo:{resolved}") from exc
    if not (relative == ALLOWED_OUTPUT_PREFIX or ALLOWED_OUTPUT_PREFIX in relative.parents):
        raise ValueError(f"output_dir_must_be_under_artifacts:{relative}")
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("json_root_not_object")
    return payload


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_int(value: Any) -> int | None:
    parsed = _safe_float(value)
    return int(parsed) if parsed is not None else None


def _safe_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _name_key(value: Any) -> str:
    text = re.sub(r"^\s*\d{1,2}\s*[\.\):-]\s*", "", str(value or "").strip())
    text = text.replace('"', "").replace("'", "").replace("`", "")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    return re.sub(r"\s+", " ", text)


def _bucket_field_size(size: Any) -> str:
    parsed = _safe_int(size) or 0
    if parsed <= 5:
        return "field_le_5"
    if parsed == 6:
        return "field_6"
    if parsed == 7:
        return "field_7"
    return "field_8_plus"


def _bucket_box(box: Any) -> str:
    parsed = _safe_int(box)
    if parsed is None:
        return "DATA_MISSING"
    if parsed <= 2:
        return "inside_1_2"
    if parsed <= 5:
        return "middle_3_5"
    return "outside_6_plus"


def _bucket_race_number(number: Any) -> str:
    parsed = _safe_int(number) or 0
    if parsed <= 3:
        return "early_1_3"
    if parsed <= 7:
        return "middle_4_7"
    return "late_8_plus"


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    resolved = db_path.expanduser().resolve()
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _db_metadata(db_path: Path, race_ids: Sequence[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    if not race_ids:
        return metadata, {"quick_check": None, "race_metadata_rows": 0, "dog_race_data_rows": 0}
    placeholders = ",".join("?" for _ in race_ids)
    with _connect_read_only(db_path) as conn:
        quick_check = conn.execute("PRAGMA quick_check").fetchone()[0]
        race_meta = {
            str(row["race_id"]): dict(row)
            for row in conn.execute(
                f"select race_id, distance, grade, winner_source, results_status from race_metadata where race_id in ({placeholders})",
                list(race_ids),
            )
        }
        dog_rows_by_race: dict[str, list[dict[str, Any]]] = defaultdict(list)
        dog_count = 0
        for row in conn.execute(
            f"select race_id, dog_name, box_number, data_source from dog_race_data where race_id in ({placeholders})",
            list(race_ids),
        ):
            dog_count += 1
            dog_rows_by_race[str(row["race_id"])].append(dict(row))
    for race_id in race_ids:
        meta = race_meta.get(race_id) or {}
        metadata[race_id] = {
            "distance": meta.get("distance"),
            "grade": meta.get("grade"),
            "winner_source": meta.get("winner_source"),
            "results_status": meta.get("results_status"),
            "dog_rows": dog_rows_by_race.get(race_id, []),
        }
    return metadata, {
        "quick_check": quick_check,
        "race_metadata_rows": len(race_meta),
        "dog_race_data_rows": dog_count,
    }


def _feature_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(
        key
        for key in {field for row in rows for field in row if field.startswith("feature_")}
        if key != "feature_join_status"
    )


def _nonnull_feature_count(row: Mapping[str, Any], feature_columns: Sequence[str]) -> int:
    return sum(1 for key in feature_columns if _safe_float(row.get(key)) is not None)


def _feature_vector(row: Mapping[str, Any], feature_columns: Sequence[str]) -> tuple[Any, ...]:
    return tuple(_safe_float(row.get(key)) for key in feature_columns)


def _family_values(row: Mapping[str, Any], family: str) -> list[float]:
    family_prefixes = {
        "recent_form": ["feature_recent_", "feature_career_"],
        "same_distance": ["feature_starts_same_distance", "feature_prior_same_distance", "feature_win_rate_same_distance", "feature_place_rate_same_distance", "feature_best_time_same_distance", "feature_avg_time_same_distance", "feature_median_time_same_distance", "feature_same_distance_"],
        "venue_history": ["feature_starts_same_venue", "feature_win_rate_same_venue", "feature_place_rate_same_venue", "feature_best_time_same_venue", "feature_avg_time_same_venue"],
        "grade_movement": ["feature_grade_", "feature_same_grade_", "feature_last_start_grade", "feature_recent_grade"],
        "days_since_start": ["feature_days_since_last_start"],
        "time_sectional_weight": ["feature_recent_avg_time", "feature_recent_best_time", "feature_recent_time_std", "feature_last_start_weight", "feature_recent_avg_weight", "feature_weight_delta", "feature_last_start_sectional", "feature_recent_avg_sectional", "feature_recent_best_sectional", "feature_recent_sectional_std", "feature_sectional_time_delta"],
    }[family]
    values = []
    for key, value in row.items():
        if any(key.startswith(prefix) for prefix in family_prefixes):
            parsed = _safe_float(value)
            if parsed is not None:
                values.append(parsed)
    return values


def _family_available(row: Mapping[str, Any], family: str) -> bool:
    return bool(_family_values(row, family))


def _family_positive(row: Mapping[str, Any], family: str) -> bool:
    return any(value != 0 for value in _family_values(row, family))


def _summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"races": 0, "top1_hits": 0, "top1_rate": None, "top3_hits": 0, "top3_rate": None, "mean_winner_rank": None, "winner_rank_counts": {}}
    top1_hits = sum(1 for row in records if _safe_bool(row.get("top1_hit")))
    top3_hits = sum(1 for row in records if _safe_bool(row.get("top3_hit")))
    ranks = [_safe_int(row.get("winner_predicted_rank") or row.get("winner_rank")) or 0 for row in records]
    return {
        "races": len(records),
        "top1_hits": top1_hits,
        "top1_rate": top1_hits / len(records),
        "top3_hits": top3_hits,
        "top3_rate": top3_hits / len(records),
        "mean_winner_rank": sum(ranks) / len(ranks),
        "winner_rank_counts": dict(sorted(Counter(str(rank) for rank in ranks).items())),
    }


def _breakdowns(records: Sequence[Mapping[str, Any]], dimensions: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for dimension in dimensions:
        buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            buckets[str(record.get(dimension) or "DATA_MISSING")].append(record)
        for bucket, bucket_records in sorted(buckets.items()):
            summary = _summarize_records(bucket_records)
            rows.append({"dimension": dimension, "bucket": bucket, **summary})
    return rows


def _build_race_records(
    *,
    predictions: Sequence[Mapping[str, Any]],
    features: Sequence[Mapping[str, Any]],
    db_metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    feature_by_key = {
        (str(row.get("race_id") or ""), _name_key(row.get("dog_name_key") or row.get("dog_name"))): dict(row)
        for row in features
    }
    feature_columns = _feature_columns(features)
    grouped_predictions: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped_predictions[(str(row.get("window_id") or ""), str(row.get("race_id") or ""))].append(row)

    race_records: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    for (window_id, race_id), race_predictions in sorted(grouped_predictions.items()):
        winners = [row for row in race_predictions if _safe_int(row.get("actual_win")) == 1]
        if len(winners) != 1:
            failures.append(f"winner_count_not_one:{window_id}:{race_id}:{len(winners)}")
            continue
        winner = winners[0]
        top_pred = sorted(race_predictions, key=lambda row: _safe_int(row.get("predicted_rank")) or 999999)[0]
        race_features = [
            feature_by_key.get((race_id, _name_key(row.get("dog_name_key") or row.get("dog_name"))), {})
            for row in race_predictions
        ]
        winner_feature = feature_by_key.get((race_id, _name_key(winner.get("dog_name_key") or winner.get("dog_name"))), {})
        top_feature = feature_by_key.get((race_id, _name_key(top_pred.get("dog_name_key") or top_pred.get("dog_name"))), {})
        scores = [_safe_float(row.get("score")) or 0.0 for row in race_predictions]
        unique_scores = sorted(set(scores))
        score_range = max(scores) - min(scores) if scores else 0.0
        score_bucket = "all_scores_tied" if len(unique_scores) <= 1 else "scores_distinguish_field"
        unique_vectors = { _feature_vector(row, feature_columns) for row in race_features }
        nonnull_counts = [_nonnull_feature_count(row, feature_columns) for row in race_features]
        history_value_counts = [
            _safe_int(row.get("global_prior_history_values_filled") or row.get("history_feature_values_filled")) or 0
            for row in race_features
        ]
        global_counts = [_safe_int(row.get("global_prior_history_count")) or 0 for row in race_features]
        field_global_prior_runners = sum(1 for value in global_counts if value > 0)
        field_history_value_rows = sum(1 for value in history_value_counts if value > 0)
        field_history_values = sum(history_value_counts)
        winner_global_count = _safe_int(winner_feature.get("global_prior_history_count")) or 0
        winner_history_values = _safe_int(winner_feature.get("global_prior_history_values_filled") or winner_feature.get("history_feature_values_filled")) or 0
        top_global_count = _safe_int(top_feature.get("global_prior_history_count")) or 0
        top_history_values = _safe_int(top_feature.get("global_prior_history_values_filled") or top_feature.get("history_feature_values_filled")) or 0
        if winner_global_count > 0:
            history_bucket = "winner_has_global_prior_history"
        elif field_global_prior_runners > 0:
            history_bucket = "field_has_global_prior_history_but_winner_missing"
        else:
            history_bucket = "no_global_prior_history_in_field"

        metadata = db_metadata.get(race_id) or {}
        dog_rows = metadata.get("dog_rows") or []
        winner_key = _name_key(winner.get("dog_name_key") or winner.get("dog_name"))
        top_key = _name_key(top_pred.get("dog_name_key") or top_pred.get("dog_name"))
        winner_db = next((row for row in dog_rows if _name_key(row.get("dog_name")) == winner_key), {})
        top_db = next((row for row in dog_rows if _name_key(row.get("dog_name")) == top_key), {})
        winner_box = _safe_int(winner_db.get("box_number"))
        top_box = _safe_int(top_db.get("box_number"))
        distance_present = _safe_float(metadata.get("distance")) is not None
        grade_present = bool(str(metadata.get("grade") or "").strip())
        winner_rank = _safe_int(winner.get("predicted_rank")) or 999999
        field_size = len(race_predictions)
        gate_reasons = []
        if len(unique_vectors) <= 1:
            gate_reasons.append("identical_within_race_feature_vectors")
        if score_bucket == "all_scores_tied":
            gate_reasons.append("all_prediction_scores_tied")
        if field_history_values == 0:
            gate_reasons.append("no_history_feature_values_filled")
        if sum(global_counts) == 0:
            gate_reasons.append("no_global_prior_history_count")
        if score_bucket == "all_scores_tied" and len(unique_vectors) <= 1:
            gate_status = "EXCLUDE_INSUFFICIENT_FEATURE_VARIANCE_AND_TIED_SCORES"
            metric_bucket = "insufficient_signal_excluded"
            meaningful = False
        elif score_bucket == "all_scores_tied":
            gate_status = "EXCLUDE_TIED_SCORES_DESPITE_FEATURE_VARIANCE"
            metric_bucket = "insufficient_signal_excluded"
            meaningful = False
        else:
            gate_status = "WARN_NO_HISTORY_SIGNAL_BUT_FEATURES_DISTINGUISH"
            metric_bucket = "meaningful_with_warning"
            meaningful = True

        base = {
            "race_id": race_id,
            "window_id": window_id,
            "race_date": winner.get("race_date"),
            "venue": winner.get("venue"),
            "race_number": winner.get("race_number"),
            "field_size": field_size,
            "field_size_bucket": _bucket_field_size(field_size),
            "source_bucket": winner.get("field_scope") or "DATA_MISSING",
            "feature_count": winner.get("feature_count"),
            "winner_name": winner.get("dog_name"),
            "winner_box": winner_box if winner_box is not None else "DATA_MISSING",
            "winner_box_bucket": _bucket_box(winner_box),
            "winner_predicted_rank": winner_rank,
            "top1_hit": winner_rank == 1,
            "top3_hit": winner_rank <= 3,
            "top_predicted_dog_name": top_pred.get("dog_name"),
            "top_predicted_box": top_box if top_box is not None else "DATA_MISSING",
            "race_number_bucket": _bucket_race_number(winner.get("race_number")),
            "score_unique_count": len(unique_scores),
            "score_range": score_range,
            "score_bucket": score_bucket,
            "distance_present": distance_present,
            "grade_present": grade_present,
            "history_rows_with_global_prior": field_global_prior_runners,
            "history_rows_with_any_filled_features": field_history_value_rows,
            "winner_global_prior_history_count": winner_global_count,
            "history_coverage_bucket": history_bucket,
        }
        race_records.append(base)

        family_stats: dict[str, Any] = {}
        for family in ("recent_form", "same_distance", "venue_history", "grade_movement", "days_since_start", "time_sectional_weight"):
            available = [_family_available(row, family) for row in race_features]
            positive = [_family_positive(row, family) for row in race_features]
            family_stats[f"{family}_field_rows_available"] = sum(1 for value in available if value)
            family_stats[f"{family}_field_rows_positive"] = sum(1 for value in positive if value)
            family_stats[f"winner_{family}_available"] = _family_available(winner_feature, family)
            family_stats[f"winner_{family}_positive"] = _family_positive(winner_feature, family)
            family_stats[f"top_pred_{family}_available"] = _family_available(top_feature, family)
            family_stats[f"top_pred_{family}_positive"] = _family_positive(top_feature, family)

        if not meaningful:
            root_cause = "insufficient_feature_variance_or_tied_scores"
        elif field_global_prior_runners == 0:
            root_cause = "no_field_prior_history"
        elif winner_global_count == 0:
            root_cause = "winner_missing_prior_history_but_field_has_history"
        elif family_stats["same_distance_field_rows_available"] == 0:
            root_cause = "same_distance_features_absent_for_field"
        else:
            root_cause = "model_ranker_miss_with_available_history"

        matrix_rows.append(
            {
                "race_id": race_id,
                "window_id": window_id,
                "race_date": winner.get("race_date"),
                "venue": winner.get("venue"),
                "race_number": winner.get("race_number"),
                "field_size": field_size,
                "winner_name": winner.get("dog_name"),
                "winner_rank": winner_rank,
                "top_pred_name": top_pred.get("dog_name"),
                "top1_hit": winner_rank == 1,
                "top3_hit": winner_rank <= 3,
                "score_bucket": score_bucket,
                "top_scores_tied": score_bucket == "all_scores_tied",
                "score_gap_top2": (
                    float(sorted(scores, reverse=True)[0] - sorted(scores, reverse=True)[1])
                    if len(scores) >= 2
                    else None
                ),
                "meaningful_ranking_signal": meaningful,
                "metric_bucket": metric_bucket,
                "gate_status": gate_status,
                "history_coverage_bucket": history_bucket,
                "distance_present": distance_present,
                "grade_present": grade_present,
                "field_global_prior_runners": field_global_prior_runners,
                "field_history_value_rows": field_history_value_rows,
                "field_history_values": field_history_values,
                "winner_global_prior_count": winner_global_count,
                "winner_history_values": winner_history_values,
                "top_pred_global_prior_count": top_global_count,
                "top_pred_history_values": top_history_values,
                "root_cause_bucket": root_cause,
                **family_stats,
                "winner_history_present": winner_global_count > 0,
                "winner_recent_form_present": family_stats["winner_recent_form_available"],
                "winner_same_distance_present": family_stats["winner_same_distance_available"],
                "winner_grade_movement_present": family_stats["winner_grade_movement_available"],
                "winner_time_sectional_weight_present": family_stats["winner_time_sectional_weight_available"],
                "unique_feature_vectors_within_race": len(unique_vectors),
                "min_nonnull_feature_columns_per_runner": min(nonnull_counts) if nonnull_counts else 0,
                "max_nonnull_feature_columns_per_runner": max(nonnull_counts) if nonnull_counts else 0,
                "total_history_feature_values_filled": field_history_values,
                "total_global_prior_history_count": sum(global_counts),
                "gate_reasons": ";".join(gate_reasons),
            }
        )
    return race_records, matrix_rows, failures


def _tied_rows(matrix_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in matrix_rows:
        reasons = []
        if _safe_int(row.get("total_global_prior_history_count")) == 0:
            reasons.append("no_global_prior_history_for_any_runner")
        if _safe_int(row.get("unique_feature_vectors_within_race")) <= 1:
            reasons.append("feature_vectors_identical_within_race")
        if row.get("score_bucket") == "all_scores_tied" and _safe_int(row.get("unique_feature_vectors_within_race")) > 1:
            reasons.append("score_tie_despite_some_feature_variance")
        out = {
            "race_id": row.get("race_id"),
            "window_id": row.get("window_id"),
            "race_date": row.get("race_date"),
            "venue": row.get("venue"),
            "race_number": row.get("race_number"),
            "field_size": row.get("field_size"),
            "winner_name": row.get("winner_name"),
            "winner_predicted_rank": row.get("winner_rank"),
            "top1_hit": row.get("top1_hit"),
            "top3_hit": row.get("top3_hit"),
            "score_bucket": row.get("score_bucket"),
            "history_coverage_bucket": row.get("history_coverage_bucket"),
            "feature_rows": row.get("field_size"),
            "unique_feature_vectors_within_race": row.get("unique_feature_vectors_within_race"),
            "min_nonnull_feature_columns_per_runner": row.get("min_nonnull_feature_columns_per_runner"),
            "max_nonnull_feature_columns_per_runner": row.get("max_nonnull_feature_columns_per_runner"),
            "total_history_feature_values_filled": row.get("total_history_feature_values_filled"),
            "total_global_prior_history_count": row.get("total_global_prior_history_count"),
            "triage_reason": ";".join(reasons) if reasons else "scores_distinguish_field",
        }
        rows.append(out)
    return rows


def _gate_rows(matrix_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "race_id": row.get("race_id"),
            "window_id": row.get("window_id"),
            "race_date": row.get("race_date"),
            "venue": row.get("venue"),
            "race_number": row.get("race_number"),
            "field_size": row.get("field_size"),
            "winner_name": row.get("winner_name"),
            "winner_predicted_rank": row.get("winner_rank"),
            "top1_hit": row.get("top1_hit"),
            "top3_hit": row.get("top3_hit"),
            "score_bucket": row.get("score_bucket"),
            "unique_feature_vectors_within_race": row.get("unique_feature_vectors_within_race"),
            "total_history_feature_values_filled": row.get("total_history_feature_values_filled"),
            "total_global_prior_history_count": row.get("total_global_prior_history_count"),
            "history_coverage_bucket": row.get("history_coverage_bucket"),
            "distance_present": row.get("distance_present"),
            "grade_present": row.get("grade_present"),
            "gate_status": row.get("gate_status"),
            "metric_bucket": row.get("metric_bucket"),
            "meaningful_ranking_signal": row.get("meaningful_ranking_signal"),
            "gate_reasons": row.get("gate_reasons"),
        }
        for row in matrix_rows
    ]


def _priority_rows(matrix_rows: Sequence[Mapping[str, Any]], stratified_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets = _breakdowns(matrix_rows, ["venue", "field_size", "history_coverage_bucket", "score_bucket", "winner_box_bucket", "winner_rank"])
    priority = [
        {
            "source": "failure_surface",
            "dimension": row["dimension"],
            "value": row["bucket"],
            "race_count": row["races"],
            "top1_miss_count": row["races"] - row["top1_hits"],
            "top3_miss_count": row["races"] - row["top3_hits"],
            "top1_accuracy": row["top1_rate"],
            "top3_hit_rate": row["top3_rate"],
            "mean_winner_rank": row["mean_winner_rank"],
        }
        for row in buckets
        if int(row["races"] or 0) >= 5
    ]
    for row in stratified_rows:
        race_count = _safe_int(row.get("race_count")) or 0
        if race_count >= 5 and (_safe_int(row.get("top1_miss_count")) or 0) > 0:
            priority.append(
                {
                    "source": "stratified_error_analysis",
                    "dimension": row.get("dimension"),
                    "value": row.get("value"),
                    "race_count": race_count,
                    "top1_miss_count": _safe_int(row.get("top1_miss_count")) or 0,
                    "top3_miss_count": _safe_int(row.get("top3_miss_count")) or 0,
                    "top1_accuracy": _safe_float(row.get("top1_accuracy")),
                    "top3_hit_rate": _safe_float(row.get("top3_hit_rate")),
                    "mean_winner_rank": _safe_float(row.get("mean_winner_rank")),
                }
            )
    priority.sort(key=lambda row: (-(row["top1_miss_count"] or 0), -(row["race_count"] or 0), str(row["dimension"]), str(row["value"])))
    return priority[:50]


def _write_summary(path: Path, title: str, status: str, bullets: Sequence[str]) -> None:
    lines = [
        f"# {title}",
        "",
        f"Status: `{status}`.",
        "",
        "No official fetches, DB writes, label writes, metadata writes, model persistence, registry/GitHub writes, EV actions, or betting actions occurred.",
        "",
        *bullets,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_outputs(
    *,
    predictions_csv: Path,
    feature_rows_csv: Path,
    rolling_report_json: Path,
    feature_join_json: Path,
    stratified_csv: Path | None,
    stratified_json: Path | None,
    db_path: Path,
    output_root: Path,
    tag: str,
) -> dict[str, Any]:
    output_root = _assert_output_dir_safe(output_root)
    predictions = _load_csv(predictions_csv)
    features = _load_csv(feature_rows_csv)
    rolling_report = _load_json(rolling_report_json)
    feature_join = _load_json(feature_join_json)
    stratified_rows = _load_csv(stratified_csv) if stratified_csv and stratified_csv.exists() else []
    race_ids = sorted({str(row.get("race_id") or "") for row in predictions if row.get("race_id")})
    metadata, db_summary = _db_metadata(db_path, race_ids)
    race_records, matrix_rows, failures = _build_race_records(
        predictions=predictions,
        features=features,
        db_metadata=metadata,
    )
    if failures:
        raise ValueError(";".join(failures[:10]))

    aggregate = {
        "evaluated_races": len(race_records),
        "evaluated_prediction_rows": len(predictions),
        "windows": len({str(row.get("window_id")) for row in race_records}),
        "top1": _summarize_records(race_records)["top1_rate"],
        "top3": _summarize_records(race_records)["top3_rate"],
        "mean_winner_rank": _summarize_records(race_records)["mean_winner_rank"],
        "winner_rank_counts": _summarize_records(race_records)["winner_rank_counts"],
        "score_bucket_counts": dict(sorted(Counter(str(row.get("score_bucket")) for row in race_records).items())),
        "history_coverage_bucket_counts": dict(sorted(Counter(str(row.get("history_coverage_bucket")) for row in race_records).items())),
        "distance_present_races": sum(1 for row in race_records if row.get("distance_present") is True),
        "grade_present_races": sum(1 for row in race_records if row.get("grade_present") is True),
        "missing_winner_box_matches": sum(1 for row in race_records if row.get("winner_box") == "DATA_MISSING"),
    }

    breakdown_dimensions = ["venue", "field_size_bucket", "field_size", "winner_box_bucket", "winner_box", "race_number_bucket", "source_bucket", "score_bucket", "history_coverage_bucket", "distance_present", "grade_present"]
    breakdown_rows = _breakdowns(race_records, breakdown_dimensions)
    weak_rows = [row for row in breakdown_rows if int(row.get("races") or 0) >= 8 and (row.get("top1_rate") or 0) <= aggregate["top1"]]
    strong_rows = [row for row in breakdown_rows if int(row.get("races") or 0) >= 8 and (row.get("top3_rate") or 0) >= aggregate["top3"]]
    weak_rows.sort(key=lambda row: (row.get("top1_rate") or 0, -(row.get("races") or 0)))
    strong_rows.sort(key=lambda row: (-(row.get("top3_rate") or 0), -(row.get("races") or 0)))

    output_dirs = {
        "failure_surface": output_root / f"no_box_pairwise_rolling_failure_surface_{tag}",
        "tied_score": output_root / f"no_box_pairwise_tied_score_triage_{tag}",
        "variance_gate": output_root / f"insufficient_feature_variance_gate_{tag}",
        "error_matrix": output_root / f"no_box_current_214_feature_coverage_error_matrix_{tag}",
        "priority_digest": output_root / f"stratified_error_priority_digest_{tag}",
    }
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    failure_report = {
        "schema_version": "no_box_pairwise_rolling_failure_surface_v1",
        "status": "REPORT_ONLY_PAIRWISE_ROLLING_FAILURE_SURFACE_COMPLETE",
        "generated_at": utc_now(),
        "inputs": {
            "predictions_csv": str(predictions_csv),
            "feature_rows_csv": str(feature_rows_csv),
            "rolling_report_json": str(rolling_report_json),
            "feature_join_json": str(feature_join_json),
        },
        "db_path": str(db_path.expanduser().resolve()),
        "db_summary": db_summary,
        "writes_performed": dict(WRITES_PERFORMED),
        "aggregate": aggregate,
        "breakdown_dimensions": breakdown_dimensions,
        "weak_buckets_min_8_races": weak_rows[:20],
        "strong_top3_buckets_min_8_races": strong_rows[:20],
        "blockers": [
            "Top1 remains random-level despite suffix-normalized global prior-history recovery.",
            "Distance and grade breakdowns remain unavailable until target metadata is collected.",
            "No-box packet still correctly avoids box features; winner-box is post-hoc analysis only.",
        ],
        "outputs": {
            "report_json": str(output_dirs["failure_surface"] / "failure_surface_report.json"),
            "race_level_csv": str(output_dirs["failure_surface"] / "race_level_failure_surface.csv"),
            "breakdowns_csv": str(output_dirs["failure_surface"] / "failure_surface_breakdowns.csv"),
        },
    }
    _write_json(output_dirs["failure_surface"] / "failure_surface_report.json", failure_report)
    _write_csv(output_dirs["failure_surface"] / "race_level_failure_surface.csv", race_records, list(race_records[0].keys()))
    _write_csv(output_dirs["failure_surface"] / "failure_surface_breakdowns.csv", breakdown_rows, ["dimension", "bucket", "races", "top1_hits", "top1_rate", "top3_hits", "top3_rate", "mean_winner_rank", "winner_rank_counts"])
    _write_summary(
        output_dirs["failure_surface"] / "SUMMARY.md",
        "No-Box Pairwise Rolling Failure Surface",
        failure_report["status"],
        [
            f"- Evaluated races: `{aggregate['evaluated_races']}`",
            f"- Top1: `{aggregate['top1']}`",
            f"- Top3: `{aggregate['top3']}`",
            f"- Score buckets: `{aggregate['score_bucket_counts']}`",
            f"- History buckets: `{aggregate['history_coverage_bucket_counts']}`",
        ],
    )

    tied_all_rows = _tied_rows(matrix_rows)
    tied_rows = [row for row in tied_all_rows if row.get("score_bucket") == "all_scores_tied"]
    tied_report = {
        "schema_version": "no_box_pairwise_tied_score_triage_v1",
        "status": "REPORT_ONLY_TIED_SCORE_TRIAGE_COMPLETE",
        "generated_at": utc_now(),
        "inputs": {
            "feature_rows_csv": str(feature_rows_csv),
            "race_level_failure_surface_csv": str(output_dirs["failure_surface"] / "race_level_failure_surface.csv"),
        },
        "writes_performed": dict(WRITES_PERFORMED),
        "aggregate": {
            "evaluated_races": len(tied_all_rows),
            "tied_score_races": len(tied_rows),
            "non_tied_score_races": len(tied_all_rows) - len(tied_rows),
            "tied_reason_counts": dict(sorted(Counter(reason for row in tied_rows for reason in str(row.get("triage_reason") or "").split(";") if reason).items())),
            "tied_history_coverage_counts": dict(sorted(Counter(str(row.get("history_coverage_bucket")) for row in tied_rows).items())),
            "tied_unique_feature_vector_counts": dict(sorted(Counter(str(row.get("unique_feature_vectors_within_race")) for row in tied_rows).items())),
            "tied_max_nonnull_feature_column_counts": dict(sorted(Counter(str(row.get("max_nonnull_feature_columns_per_runner")) for row in tied_rows).items())),
            "tied_top1": _summarize_records(tied_rows)["top1_rate"],
            "tied_top3": _summarize_records(tied_rows)["top3_rate"],
            "non_tied_top1": _summarize_records([row for row in tied_all_rows if row.get("score_bucket") != "all_scores_tied"])["top1_rate"],
            "non_tied_top3": _summarize_records([row for row in tied_all_rows if row.get("score_bucket") != "all_scores_tied"])["top3_rate"],
        },
        "interpretation": [
            "Tied-score races identify where ranking output is not meaningfully model-driven.",
            "No-global-prior tied races remain a feature-coverage issue rather than a model-family issue.",
        ],
    }
    tied_fields = ["race_id", "window_id", "race_date", "venue", "race_number", "field_size", "winner_name", "winner_predicted_rank", "top1_hit", "top3_hit", "score_bucket", "history_coverage_bucket", "feature_rows", "unique_feature_vectors_within_race", "min_nonnull_feature_columns_per_runner", "max_nonnull_feature_columns_per_runner", "total_history_feature_values_filled", "total_global_prior_history_count", "triage_reason"]
    _write_json(output_dirs["tied_score"] / "tied_score_triage_report.json", tied_report)
    _write_csv(output_dirs["tied_score"] / "all_race_tied_score_triage.csv", tied_all_rows, tied_fields)
    _write_csv(output_dirs["tied_score"] / "tied_score_races.csv", tied_rows, tied_fields)
    _write_summary(output_dirs["tied_score"] / "SUMMARY.md", "No-Box Pairwise Tied-Score Triage", tied_report["status"], [f"- Tied races: `{len(tied_rows)}`", f"- Tied reason counts: `{tied_report['aggregate']['tied_reason_counts']}`"])

    gate_rows = _gate_rows(matrix_rows)
    gate_status_rows = _breakdowns(gate_rows, ["gate_status"])
    metric_rows = [{"bucket": "all_evaluated_races", **_summarize_records(gate_rows)}]
    metric_rows.extend(
        {"bucket": bucket, **_summarize_records(rows)}
        for bucket, rows in sorted(defaultdict(list, {k: [r for r in gate_rows if r.get("metric_bucket") == k] for k in sorted({r.get("metric_bucket") for r in gate_rows})}).items())
    )
    variance_report = {
        "schema_version": "insufficient_feature_variance_gate_v1",
        "status": "REPORT_ONLY_INSUFFICIENT_FEATURE_VARIANCE_GATE_COMPLETE",
        "generated_at": utc_now(),
        "inputs": {
            "race_level_failure_surface_csv": str(output_dirs["failure_surface"] / "race_level_failure_surface.csv"),
            "all_race_tied_score_triage_csv": str(output_dirs["tied_score"] / "all_race_tied_score_triage.csv"),
        },
        "writes_performed": dict(WRITES_PERFORMED),
        "gate_definition": {
            "exclude_if_all_prediction_scores_tied": True,
            "exclude_if_unique_feature_vectors_within_race_lte": 1,
            "warn_if_no_history_feature_values_filled": True,
            "reason": "Separate races with no dog-specific feature variance or tied scores from meaningful learned rankings.",
        },
        "aggregate": {
            "evaluated_races": len(gate_rows),
            "excluded_insufficient_signal_races": sum(1 for row in gate_rows if not _safe_bool(row.get("meaningful_ranking_signal"))),
            "meaningful_ranking_signal_races": sum(1 for row in gate_rows if _safe_bool(row.get("meaningful_ranking_signal"))),
            "gate_status_counts": dict(sorted(Counter(str(row.get("gate_status")) for row in gate_rows).items())),
            "metric_bucket_counts": dict(sorted(Counter(str(row.get("metric_bucket")) for row in gate_rows).items())),
            "distance_present_races": sum(1 for row in gate_rows if _safe_bool(row.get("distance_present"))),
            "grade_present_races": sum(1 for row in gate_rows if _safe_bool(row.get("grade_present"))),
        },
        "gate_status_summaries": gate_status_rows,
        "metric_summaries": metric_rows,
        "interpretation": [
            "Headline Top1/Top3 should be reported with the insufficient-signal split.",
            "Distance and grade remain unavailable for evaluated races.",
        ],
    }
    gate_fields = ["race_id", "window_id", "race_date", "venue", "race_number", "field_size", "winner_name", "winner_predicted_rank", "top1_hit", "top3_hit", "score_bucket", "unique_feature_vectors_within_race", "total_history_feature_values_filled", "total_global_prior_history_count", "history_coverage_bucket", "distance_present", "grade_present", "gate_status", "metric_bucket", "meaningful_ranking_signal", "gate_reasons"]
    _write_json(output_dirs["variance_gate"] / "feature_variance_gate_report.json", variance_report)
    _write_csv(output_dirs["variance_gate"] / "race_feature_variance_gate.csv", gate_rows, gate_fields)
    _write_csv(output_dirs["variance_gate"] / "feature_variance_gate_metric_summary.csv", metric_rows, ["bucket", "races", "top1_hits", "top1_rate", "top3_hits", "top3_rate", "mean_winner_rank", "winner_rank_counts"])
    _write_summary(output_dirs["variance_gate"] / "SUMMARY.md", "Insufficient Feature Variance Gate", variance_report["status"], [f"- Gate status counts: `{variance_report['aggregate']['gate_status_counts']}`", f"- Meaningful signal races: `{variance_report['aggregate']['meaningful_ranking_signal_races']}`"])

    root_counts = dict(sorted(Counter(str(row.get("root_cause_bucket")) for row in matrix_rows).items()))
    matrix_report = {
        "status": "REPORT_ONLY_CURRENT_214_FEATURE_COVERAGE_ERROR_MATRIX_COMPLETE",
        "generated_at_utc": utc_now(),
        "inputs": {
            "prediction_csv": str(predictions_csv),
            "feature_csv": str(feature_rows_csv),
            "rolling_report_json": str(rolling_report_json),
            "feature_join_json": str(feature_join_json),
            "variance_gate_csv": str(output_dirs["variance_gate"] / "race_feature_variance_gate.csv"),
            "variance_gate_json": str(output_dirs["variance_gate"] / "feature_variance_gate_report.json"),
            "protected_db": str(db_path.expanduser().resolve()),
        },
        "writes_performed": {
            "db_write": False,
            "label_write": False,
            "metadata_write": False,
            "official_fetch": False,
            "snapshot_or_manifest_mutation": False,
            "model_training_or_persistence": False,
            "registry_update": False,
            "github_write": False,
            "ev_or_betting_action": False,
        },
        "summary": {
            "evaluated_races": len(matrix_rows),
            "prediction_rows": len(predictions),
            "validation_status": (rolling_report.get("validation") or {}).get("status"),
            "usable_feature_count": (rolling_report.get("validation") or {}).get("usable_feature_count"),
            "top1_accuracy": round(aggregate["top1"], 6),
            "top3_rate": round(aggregate["top3"], 6),
            "top1_hits": sum(1 for row in matrix_rows if _safe_bool(row.get("top1_hit"))),
            "top1_misses": sum(1 for row in matrix_rows if not _safe_bool(row.get("top1_hit"))),
            "top3_hits": sum(1 for row in matrix_rows if _safe_bool(row.get("top3_hit"))),
            "top3_misses": sum(1 for row in matrix_rows if not _safe_bool(row.get("top3_hit"))),
            "top1_expected_random": (rolling_report.get("aggregate_metrics") or {}).get("expected_random_top1"),
            "top3_expected_random": (rolling_report.get("aggregate_metrics") or {}).get("expected_random_top3"),
            "distance_present_races": aggregate["distance_present_races"],
            "grade_present_races": aggregate["grade_present_races"],
            "same_distance_available_for_winner_races": sum(1 for row in matrix_rows if _safe_bool(row.get("winner_same_distance_available"))),
            "grade_movement_available_for_winner_races": sum(1 for row in matrix_rows if _safe_bool(row.get("winner_grade_movement_available"))),
            "winner_has_prior_history_races": sum(1 for row in matrix_rows if _safe_int(row.get("winner_global_prior_count")) and _safe_int(row.get("winner_global_prior_count")) > 0),
            "winner_missing_prior_history_races": sum(1 for row in matrix_rows if (_safe_int(row.get("winner_global_prior_count")) or 0) == 0),
            "meaningful_signal_races": variance_report["aggregate"]["meaningful_ranking_signal_races"],
            "insufficient_signal_races": variance_report["aggregate"]["excluded_insufficient_signal_races"],
            "root_cause_counts": root_counts,
            "race_matrix_join_failures": [],
        },
        "interpretation": {
            "metadata_blocker": "Distance and grade are absent for all evaluated races, leaving same-distance and target-grade movement features unavailable.",
            "history_blocker": "Winner missing-prior-history races identify where source coverage or identity recovery can help.",
            "modeling_blocker": "Top1 remains near random after suffix-normalized history recovery.",
        },
    }
    matrix_fields = list(matrix_rows[0].keys())
    bucket_summary = [
        {"root_cause_bucket": bucket, "race_count": count}
        for bucket, count in root_counts.items()
    ]
    top1_priority = sorted(
        [row for row in matrix_rows if not _safe_bool(row.get("top1_hit"))],
        key=lambda row: (str(row.get("root_cause_bucket")), str(row.get("race_date")), str(row.get("race_id"))),
    )[:100]
    _write_json(output_dirs["error_matrix"] / "feature_coverage_error_matrix_report.json", matrix_report)
    _write_csv(output_dirs["error_matrix"] / "race_feature_coverage_error_matrix.csv", matrix_rows, matrix_fields)
    _write_csv(output_dirs["error_matrix"] / "feature_coverage_bucket_summary.csv", bucket_summary, ["root_cause_bucket", "race_count"])
    _write_csv(output_dirs["error_matrix"] / "top1_miss_priority_buckets.csv", top1_priority, matrix_fields)
    _write_summary(output_dirs["error_matrix"] / "SUMMARY.md", "Current 214 Feature Coverage Error Matrix", matrix_report["status"], [f"- Top1: `{matrix_report['summary']['top1_accuracy']}`", f"- Top3: `{matrix_report['summary']['top3_rate']}`", f"- Root causes: `{root_counts}`"])

    stratified_json_data = _load_json(stratified_json) if stratified_json and stratified_json.exists() else {}
    priority_rows = _priority_rows(matrix_rows, stratified_rows)
    digest = {
        "schema_version": "current_214_stratified_error_priority_digest_v1",
        "status": "REPORT_ONLY_STRATIFIED_ERROR_PRIORITY_DIGEST_COMPLETE",
        "generated_at_utc": utc_now(),
        "inputs": {
            "stratified_error_analysis_csv": str(stratified_csv) if stratified_csv else None,
            "stratified_error_analysis_json": str(stratified_json) if stratified_json else None,
            "feature_coverage_error_matrix_csv": str(output_dirs["error_matrix"] / "race_feature_coverage_error_matrix.csv"),
            "feature_coverage_error_matrix_report_json": str(output_dirs["error_matrix"] / "feature_coverage_error_matrix_report.json"),
            "feature_coverage_bucket_summary_csv": str(output_dirs["error_matrix"] / "feature_coverage_bucket_summary.csv"),
        },
        "writes_performed": dict(WRITES_PERFORMED),
        "summary": {
            "evaluated_races": matrix_report["summary"]["evaluated_races"],
            "prediction_rows": matrix_report["summary"]["prediction_rows"],
            "top1_accuracy": matrix_report["summary"]["top1_accuracy"],
            "top1_hits": matrix_report["summary"]["top1_hits"],
            "top1_misses": matrix_report["summary"]["top1_misses"],
            "top3_rate": matrix_report["summary"]["top3_rate"],
            "top3_hits": matrix_report["summary"]["top3_hits"],
            "top3_misses": matrix_report["summary"]["top3_misses"],
            "mean_winner_rank": round(float(aggregate["mean_winner_rank"] or 0), 6),
            "missing_distance_records": (stratified_json_data.get("missing_dimension_counts") or {}).get("distance", aggregate["evaluated_races"]),
            "missing_winner_box_records": (stratified_json_data.get("missing_dimension_counts") or {}).get("winner_box", 0),
            "missing_source_bucket_records": (stratified_json_data.get("missing_dimension_counts") or {}).get("source_bucket", 0),
            "priority_row_count": len(priority_rows),
        },
        "requested_dimension_status": {
            "venue": "AVAILABLE",
            "distance": "DATA_MISSING",
            "field_size": "AVAILABLE",
            "box": "AVAILABLE_POST_HOC_WINNER_BOX_ONLY",
            "source_bucket": "AVAILABLE_SINGLE_BUCKET",
            "winner_rank": "AVAILABLE",
        },
        "priority_rows": priority_rows,
        "top_findings": [
            "Distance stratification remains DATA_MISSING until target metadata is collected.",
            "Top1 remains near random despite suffix-normalized global-prior recovery.",
            "Field-size and history-coverage buckets still explain large miss surfaces.",
        ],
        "next_action_map": {
            "metadata": "approve_or_provide_source_bound_target_distance_and_grade",
            "history": "continue_prior_history_source_coverage_and_identity_recovery",
            "model": "avoid_promotion_until_feature_coverage_and_temporal_results_improve",
        },
    }
    _write_json(output_dirs["priority_digest"] / "stratified_error_priority_digest_report.json", digest)
    _write_csv(output_dirs["priority_digest"] / "stratified_error_priority_buckets.csv", priority_rows, ["source", "dimension", "value", "race_count", "top1_miss_count", "top3_miss_count", "top1_accuracy", "top3_hit_rate", "mean_winner_rank"])
    _write_summary(output_dirs["priority_digest"] / "SUMMARY.md", "Stratified Error Priority Digest", digest["status"], [f"- Priority rows: `{len(priority_rows)}`", f"- Top1 misses: `{digest['summary']['top1_misses']}`", f"- Requested dimension status: `{digest['requested_dimension_status']}`"])
    return {"status": "REPORT_ONLY_DOWNSTREAM_DIAGNOSTICS_COMPLETE", "output_dirs": {key: str(value) for key, value in output_dirs.items()}, "aggregate": aggregate, "matrix_summary": matrix_report["summary"], "priority_summary": digest["summary"], "db_summary": db_summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", required=True, type=Path)
    parser.add_argument("--feature-rows-csv", required=True, type=Path)
    parser.add_argument("--rolling-report-json", required=True, type=Path)
    parser.add_argument("--feature-join-json", required=True, type=Path)
    parser.add_argument("--stratified-csv", type=Path)
    parser.add_argument("--stratified-json", type=Path)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = build_outputs(
        predictions_csv=args.predictions_csv,
        feature_rows_csv=args.feature_rows_csv,
        rolling_report_json=args.rolling_report_json,
        feature_join_json=args.feature_join_json,
        stratified_csv=args.stratified_csv,
        stratified_json=args.stratified_json,
        db_path=args.db,
        output_root=args.output_root,
        tag=args.tag,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
