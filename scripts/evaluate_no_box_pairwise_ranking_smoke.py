#!/usr/bin/env python3
"""Evaluate a report-only no-box pairwise ranking smoke packet.

This script consumes rows from build_no_box_actual_win_feature_join_packet. It
fits only transient, report-local pairwise comparators over earlier races and
scores later races by Top1/Top3 winner rank. It does not persist, promote, or
train any production model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/"
SCHEMA_VERSION = "no_box_pairwise_ranking_smoke_v1"
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
PREDICTION_FIELDS = [
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


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


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


def _race_sort_key(rows: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    first = rows[0] if rows else {}
    return (str(first.get("race_date") or ""), str(first.get("race_id") or ""))


def _dog_sort_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (str(row.get("dog_name_key") or ""), str(row.get("dog_name") or ""))


def _group_by_race(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("race_id") or "")].append(dict(row))
    return dict(grouped)


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


def _numeric_feature_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key.startswith("feature_") and _safe_float(value) is not None
        }
    )


def _validate_rows(
    *,
    feature_join_packet: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    expected_races: int | None,
) -> dict[str, Any]:
    failures = []
    warnings = []
    grouped = _group_by_race(rows)
    if expected_races is not None and len(grouped) != expected_races:
        failures.append(f"expected_races_mismatch:{expected_races}:{len(grouped)}")

    packet_summary = feature_join_packet.get("summary") or {}
    label_proxy_audit = packet_summary.get("label_proxy_audit") or {}
    if feature_join_packet.get("report_only") is not True:
        failures.append("feature_join_packet_report_only_not_true")
    if (feature_join_packet.get("writes_performed") or {}).get("label_write") is not False:
        failures.append("feature_join_packet_label_write_not_false")
    if label_proxy_audit.get("status") == "POTENTIAL_LABEL_PROXY":
        warnings.append("feature_join_packet_label_proxy_audit_potential_proxy")

    candidate_kind_counts: Counter[str] = Counter()
    venue_counts: Counter[str] = Counter()
    race_field_sizes: dict[str, int] = {}
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
        candidate_kind_counts[str(row.get("candidate_kind") or "UNKNOWN")] += 1
        venue_counts[str(row.get("venue") or "UNKNOWN")] += 1

    for race_id, race_rows in grouped.items():
        positive_count = sum(int(row.get("actual_win") or 0) for row in race_rows)
        race_field_sizes[race_id] = len(race_rows)
        if positive_count != 1:
            failures.append(f"race_{race_id}_actual_win_positive_count:{positive_count}")

    numeric_features = _numeric_feature_columns(rows)
    forbidden_feature_reasons = {
        feature: _feature_reason_forbidden(feature)
        for feature in numeric_features
        if _feature_reason_forbidden(feature)
    }
    hard_forbidden_feature_reasons = {
        feature: reason
        for feature, reason in forbidden_feature_reasons.items()
        if reason != "finish_or_result_proxy_feature"
    }
    soft_excluded_feature_reasons = {
        feature: reason
        for feature, reason in forbidden_feature_reasons.items()
        if reason == "finish_or_result_proxy_feature"
    }
    if hard_forbidden_feature_reasons:
        failures.append(
            "forbidden_numeric_features_present:"
            + ",".join(
                f"{feature}:{reason}"
                for feature, reason in sorted(hard_forbidden_feature_reasons.items())
            )
        )
    if soft_excluded_feature_reasons:
        warnings.append(
            "excluded_finish_or_result_proxy_features:"
            + ",".join(sorted(soft_excluded_feature_reasons))
        )
    all_feature_keys = sorted(
        {key for row in rows for key in row if key.startswith("feature_")}
    )
    all_null_forbidden_features = [
        feature
        for feature in all_feature_keys
        if feature not in numeric_features and _feature_reason_forbidden(feature)
    ]
    if all_null_forbidden_features:
        warnings.append(
            "all_null_forbidden_feature_columns_not_used:"
            + ",".join(all_null_forbidden_features)
        )
    usable_features = [
        feature
        for feature in numeric_features
        if feature not in forbidden_feature_reasons
    ]
    complete_field_races = sum(
        1
        for race_rows in grouped.values()
        if all(row.get("field_complete_for_ranking") is True for row in race_rows)
    )
    return {
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "warnings": warnings,
        "race_count": len(grouped),
        "row_count": len(rows),
        "candidate_kind_counts": dict(sorted(candidate_kind_counts.items())),
        "venue_counts": dict(sorted(venue_counts.items())),
        "field_size_counts": dict(sorted(Counter(race_field_sizes.values()).items())),
        "complete_field_races": complete_field_races,
        "partial_field_races": len(grouped) - complete_field_races,
        "numeric_feature_columns_present": numeric_features,
        "usable_feature_columns": usable_features,
        "usable_feature_count": len(usable_features),
        "excluded_numeric_feature_reasons": forbidden_feature_reasons,
        "hard_forbidden_numeric_feature_reasons": hard_forbidden_feature_reasons,
        "all_null_forbidden_feature_columns": all_null_forbidden_features,
    }


def _source_packet_rejection(feature_join_packet: Mapping[str, Any]) -> tuple[str | None, str | None]:
    status = str(feature_join_packet.get("status") or "")
    if status == "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_LEAKAGE_RISK":
        return (
            "REPORT_ONLY_PAIRWISE_RANKING_REJECTED_LEAKAGE_RISK",
            "feature_join_packet_status_leakage_risk",
        )
    if status and status != "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_READY":
        return (
            "REPORT_ONLY_PAIRWISE_RANKING_REJECTED_SOURCE_PACKET_NOT_READY",
            f"feature_join_packet_status_not_ready:{status}",
        )
    return None, None


def _standardization(
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for feature in feature_columns:
        values = [
            value
            for value in (_safe_float(row.get(feature)) for row in rows)
            if value is not None
        ]
        if not values:
            means[feature] = 0.0
            scales[feature] = 1.0
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        scale = variance ** 0.5
        means[feature] = mean
        scales[feature] = scale if scale > 1e-12 else 1.0
    return means, scales


def _row_vector(
    row: Mapping[str, Any],
    feature_columns: Sequence[str],
    means: Mapping[str, float],
    scales: Mapping[str, float],
) -> list[float]:
    vector = []
    for feature in feature_columns:
        value = _safe_float(row.get(feature))
        if value is None:
            vector.append(0.0)
        else:
            vector.append((value - means[feature]) / scales[feature])
    return vector


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=False))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _train_pairwise_weights(
    train_race_rows: Sequence[Sequence[Mapping[str, Any]]],
    feature_columns: Sequence[str],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[list[float], dict[str, float], dict[str, float], dict[str, Any]]:
    flat_train_rows = [row for race_rows in train_race_rows for row in race_rows]
    means, scales = _standardization(flat_train_rows, feature_columns)
    weights = [0.0 for _ in feature_columns]
    pair_count = 0
    for _ in range(max(1, epochs)):
        for race_rows in train_race_rows:
            ordered = sorted(race_rows, key=_dog_sort_key)
            winners = [row for row in ordered if int(row.get("actual_win") or 0) == 1]
            if len(winners) != 1:
                continue
            winner_vector = _row_vector(winners[0], feature_columns, means, scales)
            for loser in ordered:
                if int(loser.get("actual_win") or 0) == 1:
                    continue
                loser_vector = _row_vector(loser, feature_columns, means, scales)
                diff = [winner - loser for winner, loser in zip(winner_vector, loser_vector, strict=False)]
                prediction = _sigmoid(_dot(weights, diff))
                error = 1.0 - prediction
                for index, value in enumerate(diff):
                    weights[index] += learning_rate * (error * value - l2 * weights[index])
                pair_count += 1
    training_summary = {
        "train_row_count": len(flat_train_rows),
        "train_race_count": len(train_race_rows),
        "pairwise_comparison_updates": pair_count,
        "epochs": max(1, epochs),
        "learning_rate": learning_rate,
        "l2": l2,
    }
    return weights, means, scales, training_summary


def _predict_race(
    *,
    race_rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    weights: Sequence[float],
    means: Mapping[str, float],
    scales: Mapping[str, float],
    train_race_count: int,
) -> list[dict[str, Any]]:
    scored = []
    for row in race_rows:
        vector = _row_vector(row, feature_columns, means, scales)
        scored.append(
            {
                **dict(row),
                "model": "report_local_pairwise_logistic_ranker",
                "score": _dot(weights, vector),
                "train_race_count": train_race_count,
                "feature_count": len(feature_columns),
            }
        )
    scored.sort(key=lambda row: (-float(row["score"]), _dog_sort_key(row)))
    for rank, row in enumerate(scored, start=1):
        row["predicted_rank"] = rank
    return scored


def _ranking_metrics(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped = _group_by_race(predictions)
    per_race = []
    for race_id, race_rows in sorted(grouped.items(), key=lambda item: _race_sort_key(item[1])):
        ordered = sorted(race_rows, key=lambda row: int(row.get("predicted_rank") or 999999))
        winners = [row for row in ordered if int(row.get("actual_win") or 0) == 1]
        winner_rank = int(winners[0].get("predicted_rank")) if winners else None
        field_size = len(ordered)
        per_race.append(
            {
                "race_id": race_id,
                "race_date": ordered[0].get("race_date") if ordered else None,
                "field_size": field_size,
                "winner_rank": winner_rank,
                "top1_hit": winner_rank == 1,
                "top3_hit": winner_rank is not None and winner_rank <= min(3, field_size),
                "field_complete_for_ranking": all(
                    row.get("field_complete_for_ranking") is True for row in ordered
                ),
            }
        )
    race_count = len(per_race)
    top1_hits = sum(1 for row in per_race if row["top1_hit"])
    top3_hits = sum(1 for row in per_race if row["top3_hit"])
    random_top1 = [1.0 / row["field_size"] for row in per_race if row["field_size"]]
    random_top3 = [
        min(3, row["field_size"]) / row["field_size"]
        for row in per_race
        if row["field_size"]
    ]
    winner_ranks = [row["winner_rank"] for row in per_race if row["winner_rank"] is not None]
    return {
        "race_count": race_count,
        "row_count": len(predictions),
        "top1_accuracy": top1_hits / race_count if race_count else None,
        "top3_hit_rate": top3_hits / race_count if race_count else None,
        "mean_winner_rank": sum(winner_ranks) / len(winner_ranks) if winner_ranks else None,
        "expected_random_top1": sum(random_top1) / len(random_top1) if random_top1 else None,
        "expected_random_top3": sum(random_top3) / len(random_top3) if random_top3 else None,
        "probability_metrics": "not_applicable_pairwise_ranking_scores_only",
        "per_race": per_race,
    }


def _temporal_pairwise_predictions(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    feature_columns: Sequence[str],
    *,
    min_train_races: int,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered_races = [
        list(race_rows)
        for _, race_rows in sorted(grouped.items(), key=lambda item: _race_sort_key(item[1]))
    ]
    predictions: list[dict[str, Any]] = []
    training_windows: list[dict[str, Any]] = []
    for race_index in range(min_train_races, len(ordered_races)):
        train_races = ordered_races[:race_index]
        eval_race = ordered_races[race_index]
        weights, means, scales, training_summary = _train_pairwise_weights(
            train_races,
            feature_columns,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        race_predictions = _predict_race(
            race_rows=eval_race,
            feature_columns=feature_columns,
            weights=weights,
            means=means,
            scales=scales,
            train_race_count=len(train_races),
        )
        predictions.extend(race_predictions)
        training_windows.append(
            {
                "eval_race_id": str(eval_race[0].get("race_id") or ""),
                "eval_race_date": str(eval_race[0].get("race_date") or ""),
                **training_summary,
            }
        )
    return predictions, training_windows


def evaluate_pairwise_ranking(
    *,
    feature_join_packet: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    feature_join_packet_path: str | None = None,
    rows_path: str | None = None,
    expected_races: int | None = None,
    min_train_races: int = 10,
    min_eval_races: int = 5,
    epochs: int = 15,
    learning_rate: float = 0.05,
    l2: float = 0.001,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validation = _validate_rows(
        feature_join_packet=feature_join_packet,
        rows=rows,
        expected_races=expected_races,
    )
    grouped = _group_by_race(rows)
    source_status, source_reason = _source_packet_rejection(feature_join_packet)
    status = "REPORT_ONLY_PAIRWISE_RANKING_EVALUATED"
    predictions: list[dict[str, Any]] = []
    training_windows: list[dict[str, Any]] = []
    metrics: dict[str, Any] | None = None
    if source_status:
        status = source_status
    elif validation["status"] != "PASS":
        status = "REPORT_ONLY_PAIRWISE_RANKING_FAILED_CONTRACT"
    elif not validation["usable_feature_columns"]:
        status = "REPORT_ONLY_PAIRWISE_RANKING_INSUFFICIENT_FEATURES"
    elif validation["race_count"] < min_train_races + min_eval_races:
        status = "REPORT_ONLY_PAIRWISE_RANKING_INSUFFICIENT_DATA"
    else:
        predictions, training_windows = _temporal_pairwise_predictions(
            grouped,
            validation["usable_feature_columns"],
            min_train_races=min_train_races,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        metrics = _ranking_metrics(predictions)
        if (metrics.get("race_count") or 0) < min_eval_races:
            status = "REPORT_ONLY_PAIRWISE_RANKING_INSUFFICIENT_DATA"

    complete_field_count = validation["complete_field_races"]
    recommended_next_action = "collect_more_complete_field_races_before_ranking_claims"
    if status == "REPORT_ONLY_PAIRWISE_RANKING_REJECTED_LEAKAGE_RISK":
        recommended_next_action = "do_not_use_history_db_enriched_metrics_until_label_proxy_risk_is_reviewed"
    elif status == "REPORT_ONLY_PAIRWISE_RANKING_EVALUATED":
        recommended_next_action = "expand_official_safe_labels_then_repeat_rolling_temporal_no_box_ranking"

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
        "metrics": metrics,
        "ranking_model": {
            "model_key": "report_local_pairwise_logistic_ranker",
            "report_local_pairwise_fit_performed": bool(predictions),
            "model_persistence_performed": False,
            "model_training_status": "REPORT_LOCAL_TRANSIENT_FIT_ONLY_NO_MODEL_PERSISTENCE",
            "probability_output": "not_applicable",
            "feature_columns": validation["usable_feature_columns"],
            "training_windows": training_windows,
        },
        "temporal_split": {
            "race_sort_key": ["race_date", "race_id"],
            "min_train_races": min_train_races,
            "min_eval_races": min_eval_races,
            "warmup_skipped_races": min(min_train_races, validation["race_count"]),
            "evaluated_races": metrics.get("race_count") if metrics else 0,
        },
        "race_grouped_complete_field_gate": {
            "status": (
                "READY_FOR_RACE_GROUPED_RANKING_EXPERIMENT"
                if complete_field_count >= 100
                else "SKIPPED_INSUFFICIENT_COMPLETE_FIELD_RACES"
            ),
            "complete_field_races": complete_field_count,
            "required_complete_field_races": 100,
            "note": "This smoke can rank actual-win rows, including partial-field rows, but it is not a production ranking-ready gate.",
        },
        "minimums": {
            "rolling_temporal_actual_win_races": 50,
            "race_grouped_ranking_complete_field_races": 100,
            "pairwise_smoke_min_train_races": min_train_races,
            "pairwise_smoke_min_eval_races": min_eval_races,
        },
        "forbidden_without_explicit_approval": list(FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL),
        "recommended_next_action": recommended_next_action,
    }
    return report, predictions


def write_outputs(output_dir: Path, report: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]]) -> None:
    output_dir = _assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "no_box_pairwise_ranking_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "no_box_pairwise_ranking_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output_dir / "no_box_pairwise_ranking_predictions.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for row in predictions:
            writer.writerow({field: row.get(field) for field in PREDICTION_FIELDS})
    metrics = report.get("metrics") or {}
    validation = report.get("validation") or {}
    complete_gate = report.get("race_grouped_complete_field_gate") or {}
    summary = [
        "# No-Box Pairwise Ranking Smoke",
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
        f"- Complete-field races: `{validation.get('complete_field_races')}`",
        f"- Partial-field races: `{validation.get('partial_field_races')}`",
        f"- Complete-field ranking gate: `{complete_gate.get('status')}`",
        "",
        "## Metrics",
        "",
        f"- Evaluated races: `{metrics.get('race_count')}`",
        f"- Top1: `{metrics.get('top1_accuracy')}`",
        f"- Top3: `{metrics.get('top3_hit_rate')}`",
        f"- Mean winner rank: `{metrics.get('mean_winner_rank')}`",
        f"- Probability metrics: `{metrics.get('probability_metrics')}`",
        "",
        "## Next",
        "",
        str(report.get("recommended_next_action")),
        "",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-join-packet", required=True)
    parser.add_argument("--rows-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-races", type=int)
    parser.add_argument("--min-train-races", type=int, default=10)
    parser.add_argument("--min-eval-races", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packet_path = Path(args.feature_join_packet).expanduser().resolve()
    rows_path = Path(args.rows_jsonl).expanduser().resolve()
    report, predictions = evaluate_pairwise_ranking(
        feature_join_packet=_load_json(packet_path),
        rows=_load_jsonl(rows_path),
        feature_join_packet_path=str(packet_path),
        rows_path=str(rows_path),
        expected_races=args.expected_races,
        min_train_races=args.min_train_races,
        min_eval_races=args.min_eval_races,
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
                "metrics": report["metrics"],
                "race_grouped_complete_field_gate": report["race_grouped_complete_field_gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["status"] in {
        "REPORT_ONLY_PAIRWISE_RANKING_FAILED_CONTRACT",
        "REPORT_ONLY_PAIRWISE_RANKING_REJECTED_LEAKAGE_RISK",
        "REPORT_ONLY_PAIRWISE_RANKING_REJECTED_SOURCE_PACKET_NOT_READY",
    }:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
