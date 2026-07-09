#!/usr/bin/env python3
"""Triage same-distance evidence on no-box pairwise Top1 misses.

This report-only helper consumes existing rolling predictions and dog-form
coverage output. It writes diagnostic artifacts only; it does not read or write
the DB, fetch official sources, write labels, train or persist models, mutate
snapshots or manifests, update registries, promote models, enable TGR, or make
EV/betting decisions.
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
DEFAULT_PACKET_ROOT = (
    ROOT
    / "artifacts"
    / "full_evidence_orchestration_20260525"
    / "expanded_historical_shadow_evaluation_20260609T_accuracy_improvement_packet_v23"
)
DEFAULT_ROLLING_DIR = (
    DEFAULT_PACKET_ROOT
    / "no_box_pairwise_rolling_windows_combined_37_history_db_masked_no_outcome_proxy_correct_db"
)
DEFAULT_COVERAGE_DIR = (
    DEFAULT_PACKET_ROOT
    / "no_box_dog_form_feature_coverage_combined_37_history_db_masked_no_outcome_proxy_correct_db"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PACKET_ROOT
    / "no_box_same_distance_miss_triage_combined_37_history_db_masked_no_outcome_proxy_correct_db"
)
SCHEMA_VERSION = "no_box_same_distance_miss_triage_v1"
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
REQUIRED_PREDICTION_FIELDS = {
    "race_id",
    "dog_name_key",
    "dog_name",
    "actual_win",
    "predicted_rank",
    "box_features_allowed",
    "finish_order_labels_allowed",
    "top3_labels_allowed",
    "label_write_approved",
}


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


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
    return parsed if math.isfinite(parsed) else None


def _safe_int(value: Any) -> int | None:
    parsed = _safe_float(value)
    return int(parsed) if parsed is not None else None


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def _mean(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _is_present(value: Any) -> bool:
    return _safe_float(value) is not None


def _group_by_race(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("race_id") or "")].append(dict(row))
    return dict(grouped)


def _is_same_distance_feature(feature: str) -> bool:
    name = feature.removeprefix("feature_")
    return "same_distance" in name or name.endswith("_same_distance")


def _same_distance_features(
    predictions: Sequence[Mapping[str, Any]],
    column_coverage_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    from_coverage = [
        str(row.get("feature"))
        for row in column_coverage_rows
        if str(row.get("family") or "") == "same_distance" and row.get("feature")
    ]
    if from_coverage:
        return sorted(dict.fromkeys(from_coverage))
    return sorted(
        {
            key
            for row in predictions
            for key in row
            if key.startswith("feature_") and _is_same_distance_feature(key)
        }
    )


def _validation(
    rolling_report: Mapping[str, Any],
    coverage_report: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    expected_eval_races: int | None,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    if rolling_report.get("report_only") is not True:
        failures.append("rolling_report_report_only_not_true")
    if coverage_report.get("report_only") is not True:
        failures.append("coverage_report_report_only_not_true")
    for label, report in (("rolling_report", rolling_report), ("coverage_report", coverage_report)):
        writes = report.get("writes_performed") or {}
        for key in ("db_write", "label_write", "model_training", "registry_mutation", "promotion"):
            if writes.get(key) is not False:
                failures.append(f"{label}_{key}_not_false")

    grouped = _group_by_race(predictions)
    if expected_eval_races is not None and len(grouped) != expected_eval_races:
        failures.append(f"expected_eval_races_mismatch:{expected_eval_races}:{len(grouped)}")
    for index, row in enumerate(predictions, start=1):
        missing = sorted(field for field in REQUIRED_PREDICTION_FIELDS if field not in row)
        if missing:
            failures.append(f"prediction_{index}_missing_required_fields:{','.join(missing)}")
        forbidden = sorted(FORBIDDEN_ROW_FIELDS & set(row))
        if forbidden:
            failures.append(f"prediction_{index}_forbidden_fields_present:{','.join(forbidden)}")
        if row.get("box_features_allowed") is not False:
            failures.append(f"prediction_{index}_box_features_allowed_not_false")
        if row.get("finish_order_labels_allowed") is not False:
            failures.append(f"prediction_{index}_finish_order_labels_allowed_not_false")
        if row.get("top3_labels_allowed") is not False:
            failures.append(f"prediction_{index}_top3_labels_allowed_not_false")
        if row.get("label_write_approved") is not False:
            failures.append(f"prediction_{index}_label_write_approved_not_false")
    for race_id, race_rows in grouped.items():
        positive_count = sum(_safe_int(row.get("actual_win")) or 0 for row in race_rows)
        rank_one_count = sum(
            1 for row in race_rows if _safe_int(row.get("predicted_rank")) == 1
        )
        if positive_count != 1:
            failures.append(f"race_{race_id}_actual_win_positive_count:{positive_count}")
        if rank_one_count != 1:
            failures.append(f"race_{race_id}_predicted_rank_one_count:{rank_one_count}")
    if not predictions:
        warnings.append("no_predictions")
    return {
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "warnings": warnings,
        "prediction_rows": len(predictions),
        "evaluated_races": len(grouped),
    }


def _feature_presence(row: Mapping[str, Any], features: Sequence[str]) -> dict[str, Any]:
    present = [feature for feature in features if _is_present(row.get(feature))]
    zero = [
        feature
        for feature in present
        if (_safe_float(row.get(feature)) is not None and _safe_float(row.get(feature)) == 0.0)
    ]
    return {
        "present_count": len(present),
        "present_rate": _rate(len(present), len(features)),
        "zero_count": len(zero),
        "zero_rate_of_present": _rate(len(zero), len(present)),
    }


def _winner_and_top_pick(race_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    winners = [row for row in race_rows if _safe_int(row.get("actual_win")) == 1]
    if len(winners) != 1:
        return None
    top_pick = min(
        race_rows,
        key=lambda row: (
            _safe_int(row.get("predicted_rank")) or 9999,
            str(row.get("dog_name_key") or ""),
        ),
    )
    return dict(winners[0]), dict(top_pick)


def classify_same_distance_miss(
    *,
    winner_present_count: int,
    top_pick_present_count: int,
    feature_count: int,
    top1_hit: bool,
) -> str:
    if top1_hit:
        return "top1_hit"
    sparse_threshold = max(1, int(feature_count * 0.35))
    if winner_present_count <= sparse_threshold and top_pick_present_count <= sparse_threshold:
        return "both_sparse_same_distance"
    if winner_present_count <= sparse_threshold and top_pick_present_count > winner_present_count:
        return "winner_sparse_top_pick_richer"
    if top_pick_present_count > winner_present_count:
        return "top_pick_richer_same_distance"
    if winner_present_count > top_pick_present_count:
        return "winner_richer_but_ranked_lower"
    return "same_distance_present_but_ranked_other"


def _race_triage_rows(
    predictions: Sequence[Mapping[str, Any]],
    same_distance_features: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for race_id, race_rows in sorted(
        _group_by_race(predictions).items(),
        key=lambda item: (
            str(item[1][0].get("race_date") or ""),
            str(item[0]),
        ),
    ):
        pair = _winner_and_top_pick(race_rows)
        if pair is None:
            continue
        winner, top_pick = pair
        top1_hit = _safe_int(top_pick.get("actual_win")) == 1
        winner_presence = _feature_presence(winner, same_distance_features)
        top_pick_presence = _feature_presence(top_pick, same_distance_features)
        row = {
            "race_id": race_id,
            "race_date": winner.get("race_date"),
            "venue": winner.get("venue"),
            "race_number": winner.get("race_number"),
            "top1_hit": top1_hit,
            "winner_dog_name": winner.get("dog_name"),
            "winner_rank": _safe_int(winner.get("predicted_rank")),
            "top_pick_dog_name": top_pick.get("dog_name"),
            "same_distance_feature_count": len(same_distance_features),
            "winner_same_distance_present_count": winner_presence["present_count"],
            "winner_same_distance_present_rate": winner_presence["present_rate"],
            "top_pick_same_distance_present_count": top_pick_presence["present_count"],
            "top_pick_same_distance_present_rate": top_pick_presence["present_rate"],
            "winner_minus_top_pick_present_count": (
                winner_presence["present_count"] - top_pick_presence["present_count"]
            ),
            "winner_same_distance_zero_count": winner_presence["zero_count"],
            "top_pick_same_distance_zero_count": top_pick_presence["zero_count"],
            "winner_starts_same_distance": winner.get("feature_starts_same_distance"),
            "top_pick_starts_same_distance": top_pick.get("feature_starts_same_distance"),
            "winner_same_distance_same_grade_start_count": winner.get(
                "feature_same_distance_same_grade_start_count"
            ),
            "top_pick_same_distance_same_grade_start_count": top_pick.get(
                "feature_same_distance_same_grade_start_count"
            ),
            "winner_recent_avg_time_same_distance_5": winner.get(
                "feature_recent_avg_time_same_distance_5"
            ),
            "top_pick_recent_avg_time_same_distance_5": top_pick.get(
                "feature_recent_avg_time_same_distance_5"
            ),
            "winner_avg_time_same_distance": winner.get("feature_avg_time_same_distance"),
            "top_pick_avg_time_same_distance": top_pick.get("feature_avg_time_same_distance"),
        }
        row["miss_class"] = classify_same_distance_miss(
            winner_present_count=int(row["winner_same_distance_present_count"]),
            top_pick_present_count=int(row["top_pick_same_distance_present_count"]),
            feature_count=len(same_distance_features),
            top1_hit=top1_hit,
        )
        winner_avg = _safe_float(row["winner_avg_time_same_distance"])
        top_pick_avg = _safe_float(row["top_pick_avg_time_same_distance"])
        row["winner_avg_time_minus_top_pick"] = (
            winner_avg - top_pick_avg
            if winner_avg is not None and top_pick_avg is not None
            else None
        )
        rows.append(row)
    return rows


def _feature_triage_rows(
    same_distance_features: Sequence[str],
    column_coverage_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_feature = {str(row.get("feature")): row for row in column_coverage_rows}
    rows: list[dict[str, Any]] = []
    for feature in same_distance_features:
        source = by_feature.get(feature, {})
        row_coverage = _safe_float(source.get("row_coverage"))
        winner_coverage = _safe_float(source.get("winner_coverage"))
        zero_rate = _safe_float(source.get("zero_rate_of_present"))
        flat = _safe_bool(source.get("flat_or_all_null"))
        action = "keep_for_next_rolling_check"
        if flat is True or (winner_coverage is not None and winner_coverage == 0.0):
            action = "quarantine_or_drop_until_real_winner_coverage"
        elif row_coverage is not None and row_coverage < 0.50:
            action = "repair_source_coverage_before_retraining"
        elif zero_rate is not None and zero_rate > 0.90:
            action = "review_zero_dominance_before_weighting"
        rows.append(
            {
                "feature": feature,
                "row_coverage": row_coverage,
                "winner_coverage": winner_coverage,
                "prediction_row_coverage": _safe_float(source.get("prediction_row_coverage")),
                "zero_rate_of_present": zero_rate,
                "distinct_non_null_values": _safe_int(source.get("distinct_non_null_values")),
                "flat_or_all_null": flat,
                "triage_action": action,
            }
        )
    return rows


def analyze_same_distance_misses(
    *,
    rolling_report: Mapping[str, Any],
    coverage_report: Mapping[str, Any],
    column_coverage_rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    rolling_report_path: str | None = None,
    coverage_report_path: str | None = None,
    column_coverage_path: str | None = None,
    predictions_path: str | None = None,
    expected_eval_races: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    validation = _validation(
        rolling_report=rolling_report,
        coverage_report=coverage_report,
        predictions=predictions,
        expected_eval_races=expected_eval_races,
    )
    same_distance_features = _same_distance_features(predictions, column_coverage_rows)
    race_rows = _race_triage_rows(predictions, same_distance_features)
    feature_rows = _feature_triage_rows(same_distance_features, column_coverage_rows)
    misses = [row for row in race_rows if row["top1_hit"] is False]
    hits = [row for row in race_rows if row["top1_hit"] is True]
    miss_class_counts = Counter(str(row["miss_class"]) for row in misses)
    feature_action_counts = Counter(str(row["triage_action"]) for row in feature_rows)
    status = "REPORT_ONLY_SAME_DISTANCE_MISS_TRIAGE_COMPLETE"
    if validation["status"] != "PASS":
        status = "REPORT_ONLY_SAME_DISTANCE_MISS_TRIAGE_FAILED_CONTRACT"
    elif not same_distance_features:
        status = "REPORT_ONLY_SAME_DISTANCE_MISS_TRIAGE_NO_SAME_DISTANCE_FEATURES"
    summary = {
        "evaluated_races": len(race_rows),
        "prediction_rows": len(predictions),
        "top1_hits": len(hits),
        "top1_misses": len(misses),
        "top1_accuracy": _rate(len(hits), len(race_rows)),
        "same_distance_feature_count": len(same_distance_features),
        "winner_same_distance_present_rate_on_hits": _mean(
            [
                float(row["winner_same_distance_present_rate"])
                for row in hits
                if row.get("winner_same_distance_present_rate") is not None
            ]
        ),
        "winner_same_distance_present_rate_on_misses": _mean(
            [
                float(row["winner_same_distance_present_rate"])
                for row in misses
                if row.get("winner_same_distance_present_rate") is not None
            ]
        ),
        "miss_class_counts": dict(sorted(miss_class_counts.items())),
        "feature_triage_action_counts": dict(sorted(feature_action_counts.items())),
        "quarantine_candidate_features": [
            row["feature"]
            for row in feature_rows
            if row["triage_action"] == "quarantine_or_drop_until_real_winner_coverage"
        ],
        "repair_candidate_features": [
            row["feature"]
            for row in feature_rows
            if row["triage_action"] == "repair_source_coverage_before_retraining"
        ],
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "report_only": True,
        "writes_performed": dict(WRITES_PERFORMED),
        "safe_to_write_now": False,
        "label_write_approved": False,
        "model_promotion_allowed": False,
        "rolling_report": rolling_report_path,
        "coverage_report": coverage_report_path,
        "column_coverage_csv": column_coverage_path,
        "rolling_predictions_jsonl": predictions_path,
        "validation": validation,
        "source_metrics": {
            "rolling_status": rolling_report.get("status"),
            "coverage_status": coverage_report.get("status"),
            "rolling_sample_size_status": rolling_report.get("sample_size_status"),
            "coverage_sample_size_status": (coverage_report.get("summary") or {}).get(
                "sample_size_status"
            ),
            "coverage_complete_field_status": (coverage_report.get("summary") or {}).get(
                "complete_field_status"
            ),
        },
        "summary": summary,
        "blockers": [
            "same_distance_win_place_rate_features_have_no_winner_coverage"
            if any(
                feature
                in {
                    "feature_win_rate_same_distance",
                    "feature_place_rate_same_distance",
                }
                for feature in summary["quarantine_candidate_features"]
            )
            else None,
            "same_distance_time_coverage_still_below_retraining_ready_threshold"
            if summary["repair_candidate_features"]
            else None,
            "actual_win_race_count_below_50"
            if validation["evaluated_races"] < 50
            else None,
        ],
        "objective_progress": {
            "pre_race_history_feature_repair": "IDENTIFIES_SAME_DISTANCE_COLUMNS_TO_REPAIR_OR_QUARANTINE",
            "stratified_error_analysis": "ADDS_TOP1_MISS_CLASSIFICATION_BY_SAME_DISTANCE_EVIDENCE",
            "rolling_temporal_validation": "USES_EXISTING_ROLLING_WINDOWS_ONLY",
            "official_safe_label_expansion": "STILL_REQUIRED",
        },
        "forbidden_without_explicit_approval": list(FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL),
        "recommended_next_action": (
            "quarantine_or_drop_same_distance_win_place_rate_features_until_real_winner_coverage_then_expand_labels_and_rerun"
            if summary["quarantine_candidate_features"]
            else "expand_official_safe_labels_and_repeat_same_distance_triage"
        ),
    }
    report["blockers"] = [item for item in report["blockers"] if item]
    return report, race_rows, feature_rows


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
            writer.writerow(row)


def write_outputs(
    output_dir: Path,
    report: Mapping[str, Any],
    race_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
) -> None:
    output_dir = _assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "same_distance_miss_triage_report.json", report)
    _write_csv(output_dir / "same_distance_top1_miss_triage.csv", race_rows)
    _write_csv(output_dir / "same_distance_feature_triage.csv", feature_rows)
    summary = report.get("summary") or {}
    lines = [
        "# No-Box Same-Distance Miss Triage",
        "",
        f"Status: `{report.get('status')}`.",
        "",
        "No DB writes, label writes, metadata writes, official fetches, snapshot or manifest mutations, model training or persistence, registry updates, promotions, TGR enablement, betting decisions, or EV actions occurred.",
        "",
        "## Counts",
        "",
        f"- Evaluated races: `{summary.get('evaluated_races')}`",
        f"- Top1 hits: `{summary.get('top1_hits')}`",
        f"- Top1 misses: `{summary.get('top1_misses')}`",
        f"- Same-distance features: `{summary.get('same_distance_feature_count')}`",
        "",
        "## Miss Classes",
        "",
    ]
    for key, value in (summary.get("miss_class_counts") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Feature Actions", ""])
    for key, value in (summary.get("feature_triage_action_counts") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `same_distance_miss_triage_report.json`",
            "- `same_distance_top1_miss_triage.csv`",
            "- `same_distance_feature_triage.csv`",
        ]
    )
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--coverage-report",
        type=Path,
        default=DEFAULT_COVERAGE_DIR / "dog_form_feature_coverage_report.json",
    )
    parser.add_argument(
        "--column-coverage",
        type=Path,
        default=DEFAULT_COVERAGE_DIR / "dog_form_feature_column_coverage.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-eval-races", type=int, default=None)
    args = parser.parse_args()

    report, race_rows, feature_rows = analyze_same_distance_misses(
        rolling_report=_load_json(args.rolling_report),
        coverage_report=_load_json(args.coverage_report),
        column_coverage_rows=_load_csv(args.column_coverage),
        predictions=_load_jsonl(args.rolling_predictions),
        rolling_report_path=str(args.rolling_report),
        coverage_report_path=str(args.coverage_report),
        column_coverage_path=str(args.column_coverage),
        predictions_path=str(args.rolling_predictions),
        expected_eval_races=args.expected_eval_races,
    )
    write_outputs(args.output_dir, report, race_rows, feature_rows)
    print(json.dumps({"status": report["status"], "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
