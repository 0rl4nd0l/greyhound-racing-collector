#!/usr/bin/env python3
"""Audit dog-form feature-family coverage for no-box ranking packets.

This report-only helper consumes the current no-box dog-form feature rows and,
optionally, rolling ranking predictions. It does not read or write the DB,
fetch official sources, write labels, train or persist models, mutate snapshots
or manifests, update registries, promote models, enable TGR, or make EV/betting
decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/"
DEFAULT_PACKET_ROOT = (
    ROOT
    / "artifacts"
    / "full_evidence_orchestration_20260525"
    / "expanded_historical_shadow_evaluation_20260609T_accuracy_improvement_packet_v23"
)
DEFAULT_FEATURE_PACKET_DIR = (
    DEFAULT_PACKET_ROOT
    / "no_box_actual_win_dog_form_feature_join_combined_37_history_db_masked_no_outcome_proxy_correct_db"
)
DEFAULT_ROLLING_DIR = (
    DEFAULT_PACKET_ROOT
    / "no_box_pairwise_rolling_windows_combined_37_history_db_masked_no_outcome_proxy_correct_db"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PACKET_ROOT
    / "no_box_dog_form_feature_coverage_combined_37_history_db_masked_no_outcome_proxy_correct_db"
)
SCHEMA_VERSION = "no_box_dog_form_feature_coverage_audit_v1"
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
FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL = [
    "write_official_safe_labels",
    "mutate_db",
    "regenerate_canonical_dataset",
    "train_or_promote_model",
    "update_registry",
    "enable_tgr",
    "betting_or_ev_action",
]
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
REQUIRED_ROW_FIELDS = {
    "race_id",
    "dog_name_key",
    "dog_name",
    "actual_win",
    "box_features_allowed",
    "finish_order_labels_allowed",
    "top3_labels_allowed",
    "label_write_approved",
}
FAMILY_ORDER = [
    "recent_win_place",
    "same_distance",
    "venue_history",
    "grade_movement",
    "recency",
    "finish_trend_excluded",
    "time_trend",
    "career_history",
    "prior_experience",
    "weight_sectional",
    "other_dog_form",
]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("json_root_not_object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _safe_int(value: Any) -> int | None:
    parsed = _safe_float(value)
    return int(parsed) if parsed is not None else None


def _mean(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def _feature_reason_forbidden(feature_key: str) -> str | None:
    name = feature_key.removeprefix("feature_")
    if "box" in name:
        return "box_feature"
    if name == "race_number":
        return "race_number_feature"
    if name in {"target_day_of_week", "target_month"} or name.startswith("target_"):
        return "target_calendar_or_target_feature"
    if name == "field_size":
        return "field_size_feature"
    if "finish" in name or "placing" in name or "result_position" in name:
        return "finish_or_result_proxy_feature"
    return None


def feature_family(feature_key: str) -> str:
    name = feature_key.removeprefix("feature_")
    if name in {"recent_win_rate_5", "recent_place_rate_5"}:
        return "recent_win_place"
    if "same_distance" in name or name.endswith("_same_distance"):
        return "same_distance"
    if "same_venue" in name or "distance_venue" in name:
        return "venue_history"
    if "grade" in name:
        return "grade_movement"
    if name.startswith("days_since_"):
        return "recency"
    if "finish" in name:
        return "finish_trend_excluded"
    if "time" in name:
        return "time_trend"
    if name.startswith("career_"):
        return "career_history"
    if name.startswith("prior_"):
        return "prior_experience"
    if "weight" in name or "sectional" in name:
        return "weight_sectional"
    return "other_dog_form"


def _feature_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({key for row in rows for key in row if key.startswith("feature_")})


def _group_by_race(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("race_id") or "")].append(dict(row))
    return dict(grouped)


def _is_present(value: Any) -> bool:
    return _safe_float(value) is not None


def _row_feature_share(row: Mapping[str, Any], columns: Sequence[str]) -> float | None:
    if not columns:
        return None
    return sum(1 for column in columns if _is_present(row.get(column))) / len(columns)


def _column_stats(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    usable_features: set[str],
) -> list[dict[str, Any]]:
    row_count = len(rows)
    race_count = len(_group_by_race(rows))
    prediction_row_count = len(predictions)
    stats: list[dict[str, Any]] = []
    for column in _feature_columns(rows):
        values = [_safe_float(row.get(column)) for row in rows]
        present_values = [value for value in values if value is not None]
        present_rows = len(present_values)
        present_races = sum(
            1 for race_rows in _group_by_race(rows).values() if any(_is_present(row.get(column)) for row in race_rows)
        )
        winner_rows = [row for row in rows if _safe_int(row.get("actual_win")) == 1]
        winner_present_rows = sum(1 for row in winner_rows if _is_present(row.get(column)))
        prediction_present_rows = sum(1 for row in predictions if _is_present(row.get(column)))
        distinct_values = sorted({value for value in present_values})
        stats.append(
            {
                "feature": column,
                "family": feature_family(column),
                "ranker_usable": column in usable_features,
                "excluded_reason": _feature_reason_forbidden(column),
                "row_count": row_count,
                "present_rows": present_rows,
                "row_coverage": _rate(present_rows, row_count),
                "race_count": race_count,
                "present_races": present_races,
                "race_coverage": _rate(present_races, race_count),
                "winner_rows": len(winner_rows),
                "winner_present_rows": winner_present_rows,
                "winner_coverage": _rate(winner_present_rows, len(winner_rows)),
                "prediction_rows": prediction_row_count,
                "prediction_present_rows": prediction_present_rows,
                "prediction_row_coverage": _rate(
                    prediction_present_rows, prediction_row_count
                ),
                "zero_rows": sum(1 for value in present_values if value == 0.0),
                "zero_rate_of_present": _rate(
                    sum(1 for value in present_values if value == 0.0), present_rows
                ),
                "distinct_non_null_values": len(distinct_values),
                "flat_or_all_null": present_rows == 0 or len(distinct_values) <= 1,
            }
        )
    return stats


def _winner_rows_by_predicted_race(
    predictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for race_id, race_rows in sorted(_group_by_race(predictions).items()):
        winners = [row for row in race_rows if _safe_int(row.get("actual_win")) == 1]
        if len(winners) != 1:
            continue
        winner = dict(winners[0])
        top_pick = min(
            race_rows,
            key=lambda row: (
                _safe_int(row.get("predicted_rank")) or 9999,
                str(row.get("dog_name_key") or ""),
            ),
        )
        winner["winner_rank"] = _safe_int(winner.get("predicted_rank"))
        winner["top1_hit"] = (
            _safe_int(top_pick.get("predicted_rank")) == 1
            and _safe_int(top_pick.get("actual_win")) == 1
        )
        winner["top_pick_dog_name"] = top_pick.get("dog_name")
        winner["top_pick_dog_name_key"] = top_pick.get("dog_name_key")
        winner["race_id"] = race_id
        return_row = {
            "race_id": race_id,
            "race_date": winner.get("race_date"),
            "venue": winner.get("venue"),
            "race_number": winner.get("race_number"),
            "winner_dog_name": winner.get("dog_name"),
            "winner_dog_name_key": winner.get("dog_name_key"),
            "winner_rank": winner.get("winner_rank"),
            "top1_hit": winner.get("top1_hit"),
            "top_pick_dog_name": winner.get("top_pick_dog_name"),
            "top_pick_dog_name_key": winner.get("top_pick_dog_name_key"),
        }
        return_row.update(
            {
                key: value
                for key, value in winner.items()
                if key.startswith("feature_")
            }
        )
        result.append(return_row)
    return result


def _family_stats(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    column_stats: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    features_by_family: dict[str, list[str]] = defaultdict(list)
    usable_by_family: dict[str, list[str]] = defaultdict(list)
    for stat in column_stats:
        feature = str(stat["feature"])
        family = str(stat["family"])
        features_by_family[family].append(feature)
        if stat.get("ranker_usable") is True:
            usable_by_family[family].append(feature)

    winner_rows = _winner_rows_by_predicted_race(predictions)
    top1_hits = [row for row in winner_rows if row.get("top1_hit") is True]
    top1_misses = [row for row in winner_rows if row.get("top1_hit") is False]
    results: list[dict[str, Any]] = []
    for family in FAMILY_ORDER + sorted(set(features_by_family) - set(FAMILY_ORDER)):
        columns = sorted(features_by_family.get(family, []))
        if not columns:
            continue
        usable_columns = sorted(usable_by_family.get(family, []))
        row_any_present = sum(
            1 for row in rows if any(_is_present(row.get(column)) for column in columns)
        )
        row_all_present = sum(
            1 for row in rows if all(_is_present(row.get(column)) for column in columns)
        )
        winner_source_rows = [row for row in rows if _safe_int(row.get("actual_win")) == 1]
        winner_any_present = sum(
            1
            for row in winner_source_rows
            if any(_is_present(row.get(column)) for column in columns)
        )
        race_any_present = sum(
            1
            for race_rows in _group_by_race(rows).values()
            if any(
                _is_present(row.get(column))
                for row in race_rows
                for column in columns
            )
        )
        hit_shares = [
            share
            for share in (_row_feature_share(row, columns) for row in top1_hits)
            if share is not None
        ]
        miss_shares = [
            share
            for share in (_row_feature_share(row, columns) for row in top1_misses)
            if share is not None
        ]
        usable_hit_shares = [
            share
            for share in (_row_feature_share(row, usable_columns) for row in top1_hits)
            if share is not None
        ]
        usable_miss_shares = [
            share
            for share in (_row_feature_share(row, usable_columns) for row in top1_misses)
            if share is not None
        ]
        mean_hit = _mean(hit_shares)
        mean_miss = _mean(miss_shares)
        usable_mean_hit = _mean(usable_hit_shares)
        usable_mean_miss = _mean(usable_miss_shares)
        family_column_stats = [stat for stat in column_stats if stat.get("family") == family]
        results.append(
            {
                "family": family,
                "feature_count": len(columns),
                "ranker_usable_feature_count": len(usable_columns),
                "excluded_feature_count": len(columns) - len(usable_columns),
                "row_count": len(rows),
                "row_any_present_count": row_any_present,
                "row_any_present_rate": _rate(row_any_present, len(rows)),
                "row_all_present_count": row_all_present,
                "row_all_present_rate": _rate(row_all_present, len(rows)),
                "race_count": len(_group_by_race(rows)),
                "race_any_present_count": race_any_present,
                "race_any_present_rate": _rate(race_any_present, len(_group_by_race(rows))),
                "winner_rows": len(winner_source_rows),
                "winner_any_present_count": winner_any_present,
                "winner_any_present_rate": _rate(winner_any_present, len(winner_source_rows)),
                "mean_column_row_coverage": _mean(
                    [
                        float(stat["row_coverage"])
                        for stat in family_column_stats
                        if stat.get("row_coverage") is not None
                    ]
                ),
                "flat_or_all_null_feature_count": sum(
                    1 for stat in family_column_stats if stat.get("flat_or_all_null") is True
                ),
                "top1_evaluated_races": len(winner_rows),
                "top1_hit_races": len(top1_hits),
                "top1_miss_races": len(top1_misses),
                "winner_feature_share_on_top1_hits": mean_hit,
                "winner_feature_share_on_top1_misses": mean_miss,
                "winner_feature_share_miss_minus_hit": (
                    mean_miss - mean_hit
                    if mean_miss is not None and mean_hit is not None
                    else None
                ),
                "winner_usable_feature_share_on_top1_hits": usable_mean_hit,
                "winner_usable_feature_share_on_top1_misses": usable_mean_miss,
                "winner_usable_feature_share_miss_minus_hit": (
                    usable_mean_miss - usable_mean_hit
                    if usable_mean_miss is not None and usable_mean_hit is not None
                    else None
                ),
                "features": columns,
                "ranker_usable_features": usable_columns,
            }
        )
    return results


def _example_rows(
    predictions: Sequence[Mapping[str, Any]],
    family_stats: Sequence[Mapping[str, Any]],
    limit: int = 20,
) -> list[dict[str, Any]]:
    focus_families = [
        str(row["family"])
        for row in sorted(
            family_stats,
            key=lambda row: (
                row.get("winner_usable_feature_share_miss_minus_hit") is None,
                row.get("winner_usable_feature_share_miss_minus_hit") or 0,
            ),
        )
        if row.get("top1_miss_races")
    ][:5]
    winner_rows = _winner_rows_by_predicted_race(predictions)
    examples: list[dict[str, Any]] = []
    for winner in winner_rows:
        if winner.get("top1_hit") is not False:
            continue
        item = {
            "race_id": winner.get("race_id"),
            "race_date": winner.get("race_date"),
            "venue": winner.get("venue"),
            "race_number": winner.get("race_number"),
            "winner_dog_name": winner.get("winner_dog_name"),
            "winner_rank": winner.get("winner_rank"),
            "top_pick_dog_name": winner.get("top_pick_dog_name"),
        }
        for family in focus_families:
            columns = [
                feature
                for row in family_stats
                if row.get("family") == family
                for feature in row.get("ranker_usable_features", [])
            ]
            item[f"{family}_winner_usable_share"] = _row_feature_share(winner, columns)
        examples.append(item)
        if len(examples) >= limit:
            break
    return examples


def _validation(
    feature_join_packet: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    expected_races: int | None,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    grouped = _group_by_race(rows)
    if expected_races is not None and len(grouped) != expected_races:
        failures.append(f"expected_races_mismatch:{expected_races}:{len(grouped)}")
    if feature_join_packet.get("report_only") is not True:
        failures.append("feature_join_packet_report_only_not_true")
    writes = feature_join_packet.get("writes_performed") or {}
    for key in ("db_write", "label_write", "model_training", "registry_mutation", "promotion"):
        if writes.get(key) is not False:
            failures.append(f"feature_join_packet_{key}_not_false")
    packet_summary = feature_join_packet.get("summary") or {}
    label_proxy_audit = packet_summary.get("label_proxy_audit") or {}
    if label_proxy_audit.get("status") == "POTENTIAL_LABEL_PROXY":
        warnings.append("feature_join_packet_label_proxy_audit_potential_proxy")

    for index, row in enumerate(rows, start=1):
        missing = sorted(field for field in REQUIRED_ROW_FIELDS if field not in row)
        if missing:
            failures.append(f"row_{index}_missing_required_fields:{','.join(missing)}")
        forbidden = sorted(FORBIDDEN_ROW_FIELDS & set(row))
        if forbidden:
            failures.append(f"row_{index}_forbidden_fields_present:{','.join(forbidden)}")
        if row.get("box_features_allowed") is not False:
            failures.append(f"row_{index}_box_features_allowed_not_false")
        if row.get("finish_order_labels_allowed") is not False:
            failures.append(f"row_{index}_finish_order_labels_allowed_not_false")
        if row.get("top3_labels_allowed") is not False:
            failures.append(f"row_{index}_top3_labels_allowed_not_false")
        if row.get("label_write_approved") is not False:
            failures.append(f"row_{index}_label_write_approved_not_false")
        if _safe_int(row.get("actual_win")) not in (0, 1):
            failures.append(f"row_{index}_actual_win_not_binary")

    for race_id, race_rows in grouped.items():
        positive_count = sum(_safe_int(row.get("actual_win")) or 0 for row in race_rows)
        if positive_count != 1:
            failures.append(f"race_{race_id}_actual_win_positive_count:{positive_count}")

    numeric_features = [
        feature
        for feature in _feature_columns(rows)
        if any(_safe_float(row.get(feature)) is not None for row in rows)
    ]
    hard_forbidden = {
        feature: reason
        for feature in numeric_features
        if (reason := _feature_reason_forbidden(feature))
        and reason != "finish_or_result_proxy_feature"
    }
    if hard_forbidden:
        failures.append(
            "forbidden_numeric_features_present:"
            + ",".join(
                f"{feature}:{reason}" for feature, reason in sorted(hard_forbidden.items())
            )
        )
    return {
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "warnings": warnings,
        "race_count": len(grouped),
        "row_count": len(rows),
        "complete_field_races": sum(
            1
            for race_rows in grouped.values()
            if all(row.get("field_complete_for_ranking") is True for row in race_rows)
        ),
        "feature_column_count": len(_feature_columns(rows)),
        "numeric_feature_column_count": len(numeric_features),
    }


def analyze_feature_coverage(
    *,
    feature_join_packet: Mapping[str, Any],
    feature_rows: Sequence[Mapping[str, Any]],
    rolling_report: Mapping[str, Any] | None = None,
    rolling_predictions: Sequence[Mapping[str, Any]] | None = None,
    feature_join_packet_path: str | None = None,
    feature_rows_path: str | None = None,
    rolling_report_path: str | None = None,
    rolling_predictions_path: str | None = None,
    expected_races: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rolling_predictions = list(rolling_predictions or [])
    validation = _validation(feature_join_packet, feature_rows, expected_races)
    usable_features = set(
        ((rolling_report or {}).get("validation") or {}).get("usable_feature_columns") or []
    )
    if not usable_features:
        usable_features = {
            feature
            for feature in _feature_columns(feature_rows)
            if _feature_reason_forbidden(feature) is None
            and any(_safe_float(row.get(feature)) is not None for row in feature_rows)
        }
    column_rows = _column_stats(feature_rows, rolling_predictions, usable_features)
    family_rows = _family_stats(feature_rows, rolling_predictions, column_rows)
    miss_examples = _example_rows(rolling_predictions, family_rows)
    low_coverage_families = [
        row["family"]
        for row in family_rows
        if (row.get("mean_column_row_coverage") is not None)
        and float(row["mean_column_row_coverage"]) < 0.50
    ]
    no_usable_families = [
        row["family"]
        for row in family_rows
        if int(row.get("ranker_usable_feature_count") or 0) == 0
    ]
    status = "REPORT_ONLY_DOG_FORM_FEATURE_COVERAGE_AUDIT_COMPLETE"
    packet_status = str(feature_join_packet.get("status") or "")
    if validation["status"] != "PASS":
        status = "REPORT_ONLY_DOG_FORM_FEATURE_COVERAGE_AUDIT_FAILED_CONTRACT"
    elif "LEAKAGE_RISK" in packet_status:
        status = "REPORT_ONLY_DOG_FORM_FEATURE_COVERAGE_AUDIT_REJECTED_LEAKAGE_RISK"
    elif not rolling_predictions:
        status = "REPORT_ONLY_DOG_FORM_FEATURE_COVERAGE_AUDIT_NO_ROLLING_PREDICTIONS"

    rolling_metrics = (rolling_report or {}).get("aggregate_metrics") or {}
    evaluated_races = len(_winner_rows_by_predicted_race(rolling_predictions))
    top1_misses = sum(
        1
        for row in _winner_rows_by_predicted_race(rolling_predictions)
        if row.get("top1_hit") is False
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "report_only": True,
        "writes_performed": dict(WRITES_PERFORMED),
        "safe_to_write_now": False,
        "label_write_approved": False,
        "model_promotion_allowed": False,
        "feature_join_packet": feature_join_packet_path,
        "feature_rows_jsonl": feature_rows_path,
        "rolling_report": rolling_report_path,
        "rolling_predictions_jsonl": rolling_predictions_path,
        "source_packet": {
            "schema_version": feature_join_packet.get("schema_version"),
            "status": feature_join_packet.get("status"),
            "history_db_fill_policy": (feature_join_packet.get("summary") or {}).get(
                "history_db_fill_policy"
            ),
            "label_proxy_audit_status": (
                (feature_join_packet.get("summary") or {}).get("label_proxy_audit") or {}
            ).get("status"),
        },
        "validation": validation,
        "summary": {
            "race_count": validation["race_count"],
            "row_count": validation["row_count"],
            "complete_field_races": validation["complete_field_races"],
            "feature_column_count": validation["feature_column_count"],
            "ranker_usable_feature_count": len(usable_features),
            "rolling_prediction_rows": len(rolling_predictions),
            "rolling_evaluated_races": evaluated_races,
            "rolling_top1_misses": top1_misses,
            "rolling_top1_hits": evaluated_races - top1_misses if evaluated_races else 0,
            "rolling_top1_accuracy": rolling_metrics.get("top1_accuracy"),
            "rolling_top3_hit_rate": rolling_metrics.get("top3_hit_rate"),
            "rolling_mean_winner_rank": rolling_metrics.get("mean_winner_rank"),
            "low_coverage_families_below_50pct_mean_column_coverage": low_coverage_families,
            "families_without_current_ranker_usable_features": no_usable_families,
            "sample_size_status": (
                "UNDERPOWERED_BELOW_50_ACTUAL_WIN_RACES"
                if validation["race_count"] < 50
                else "MEETS_MIN_ROLLING_TEMPORAL_ACTUAL_WIN_RACES"
            ),
            "complete_field_status": (
                "UNDERPOWERED_BELOW_100_COMPLETE_FIELD_RACES"
                if validation["complete_field_races"] < 100
                else "MEETS_COMPLETE_FIELD_RANKING_GATE"
            ),
        },
        "objective_progress": {
            "pre_race_history_feature_repair": "MEASURED_BY_FAMILY_COVERAGE_NOT_CLAIMED_READY",
            "race_grouped_ranking": "USES_EXISTING_REPORT_ONLY_PAIRWISE_ROLLING_OUTPUTS",
            "weak_heuristic_ablation": "NO_BOX_PACKET_CONFIRMED_BY_CONTRACT",
            "stratified_error_analysis": "ADDS_TOP1_MISS_FEATURE_FAMILY_VIEW",
            "official_safe_label_expansion": "STILL_REQUIRED",
            "rolling_temporal_validation": "CONSUMES_EXISTING_ROLLING_WINDOWS_WITH_RESERVED_FINAL_RACES",
        },
        "blockers": [
            "actual_win_race_count_below_50" if validation["race_count"] < 50 else None,
            "complete_field_race_count_below_100"
            if validation["complete_field_races"] < 100
            else None,
            "finish_trend_features_excluded_from_current_ranker_contract"
            if "finish_trend_excluded" in no_usable_families
            else None,
            "low_mean_family_column_coverage:" + ",".join(low_coverage_families)
            if low_coverage_families
            else None,
        ],
        "forbidden_without_explicit_approval": list(FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL),
        "recommended_next_action": (
            "expand_official_safe_labels_then_repeat_family_coverage_and_rolling_ranker"
            if validation["race_count"] < 50
            else "repair_low_coverage_dog_form_families_then_repeat_rolling_ranker"
        ),
    }
    report["blockers"] = [item for item in report["blockers"] if item]
    return report, family_rows, column_rows, miss_examples


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        ";".join(str(item) for item in value)
                        if isinstance(value, list)
                        else value
                    )
                    for key, value in row.items()
                }
            )


def write_outputs(
    output_dir: Path,
    report: Mapping[str, Any],
    family_rows: Sequence[Mapping[str, Any]],
    column_rows: Sequence[Mapping[str, Any]],
    miss_examples: Sequence[Mapping[str, Any]],
) -> None:
    output_dir = _assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "dog_form_feature_coverage_report.json", report)
    _write_csv(output_dir / "dog_form_feature_family_coverage.csv", family_rows)
    _write_csv(output_dir / "dog_form_feature_column_coverage.csv", column_rows)
    _write_csv(output_dir / "dog_form_top1_miss_feature_examples.csv", miss_examples)
    summary = report.get("summary") or {}
    lines = [
        "# No-Box Dog-Form Feature Coverage",
        "",
        f"Status: `{report.get('status')}`.",
        "",
        "No DB writes, label writes, metadata writes, official fetches, snapshot or manifest mutations, model training or persistence, registry updates, promotions, TGR enablement, betting decisions, or EV actions occurred.",
        "",
        "## Counts",
        "",
        f"- Races: `{summary.get('race_count')}`",
        f"- Rows: `{summary.get('row_count')}`",
        f"- Complete-field races: `{summary.get('complete_field_races')}`",
        f"- Feature columns: `{summary.get('feature_column_count')}`",
        f"- Ranker-usable feature columns: `{summary.get('ranker_usable_feature_count')}`",
        f"- Rolling evaluated races: `{summary.get('rolling_evaluated_races')}`",
        f"- Rolling Top1 misses: `{summary.get('rolling_top1_misses')}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = report.get("blockers") or []
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- _None recorded._")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `dog_form_feature_coverage_report.json`",
            "- `dog_form_feature_family_coverage.csv`",
            "- `dog_form_feature_column_coverage.csv`",
            "- `dog_form_top1_miss_feature_examples.csv`",
        ]
    )
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-join-packet",
        type=Path,
        default=DEFAULT_FEATURE_PACKET_DIR / "no_box_actual_win_feature_join_packet.json",
    )
    parser.add_argument(
        "--feature-rows",
        type=Path,
        default=DEFAULT_FEATURE_PACKET_DIR / "no_box_actual_win_feature_rows.jsonl",
    )
    parser.add_argument(
        "--rolling-report",
        type=Path,
        default=DEFAULT_ROLLING_DIR / "no_box_pairwise_rolling_windows_report.json",
    )
    parser.add_argument(
        "--rolling-predictions",
        type=Path,
        default=DEFAULT_ROLLING_DIR / "no_box_pairwise_rolling_window_predictions.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-races", type=int, default=None)
    args = parser.parse_args()

    rolling_report = _load_json(args.rolling_report) if args.rolling_report.exists() else None
    rolling_predictions = (
        _load_jsonl(args.rolling_predictions) if args.rolling_predictions.exists() else []
    )
    report, family_rows, column_rows, miss_examples = analyze_feature_coverage(
        feature_join_packet=_load_json(args.feature_join_packet),
        feature_rows=_load_jsonl(args.feature_rows),
        rolling_report=rolling_report,
        rolling_predictions=rolling_predictions,
        feature_join_packet_path=str(args.feature_join_packet),
        feature_rows_path=str(args.feature_rows),
        rolling_report_path=str(args.rolling_report) if args.rolling_report.exists() else None,
        rolling_predictions_path=str(args.rolling_predictions)
        if args.rolling_predictions.exists()
        else None,
        expected_races=args.expected_races,
    )
    write_outputs(args.output_dir, report, family_rows, column_rows, miss_examples)
    print(json.dumps({"status": report["status"], "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
