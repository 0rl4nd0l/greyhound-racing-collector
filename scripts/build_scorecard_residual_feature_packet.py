#!/usr/bin/env python3
"""Build a report-only feature audit for high-market residual scorecard races.

The packet focuses on races where the market assigned the eventual winner a
strong probability, the model assigned a weak probability, and the model top box
disagreed with the market top box. It compares source feature coverage for the
model-top, market-top, and winner runners without training, promotion, DB writes,
labels, EV output, registry mutation, snapshot rewrites, or daemon control.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/" "scorecard_residual_feature_"
REPORT_FILE = "scorecard_residual_feature_report.json"
ROLE_SUMMARY_CSV = "residual_feature_role_summary.csv"
FAMILY_COMPARISON_CSV = "residual_feature_family_comparison.csv"
RACE_DETAIL_CSV = "residual_race_feature_rows.csv"
SUMMARY_MD = "SUMMARY.md"
FINAL_READY = "SCORECARD_RESIDUAL_FEATURE_AUDIT_READY"
FINAL_DATA_MISSING = "SCORECARD_RESIDUAL_FEATURE_AUDIT_DATA_MISSING"

NO_WRITE_GUARANTEES = {
    "training": False,
    "production_promotion": False,
    "registry_mutation": False,
    "production_pointer_update": False,
    "active_model_replacement": False,
    "db_write": False,
    "label_write": False,
    "odds_write": False,
    "official_result_write": False,
    "daemon_control": False,
    "betting_or_ev_action": False,
    "snapshot_rewrite": False,
    "manifest_rewrite": False,
}

FEATURE_FAMILIES = {
    "weather_track": [
        "weather",
        "weather_source_backed",
        "track_condition",
        "track_condition_source_backed",
        "race_time_minutes_since_midnight",
    ],
    "expert_form": [
        "expert_form_metadata_from_sidecar",
        "expert_form_career_starts",
        "expert_form_career_wins",
        "expert_form_win_percent",
        "expert_form_place_percent",
        "expert_form_current_box_starts",
        "expert_form_current_box_wins",
        "expert_form_track_distance_starts",
        "expert_form_track_distance_wins",
        "expert_form_track_distance_best_time",
    ],
    "career_stats": [
        "prior_start_count",
        "career_win_rate",
        "career_place_rate",
        "career_avg_finish",
        "career_best_finish",
        "career_avg_time",
        "career_best_time",
        "career_time_std",
    ],
    "same_distance_time": [
        "starts_same_distance",
        "prior_same_distance_start_count",
        "best_time_same_distance",
        "avg_time_same_distance",
        "median_time_same_distance",
        "recent_best_time_same_distance_5",
        "recent_avg_time_same_distance_5",
        "days_since_last_same_distance_start",
        "win_rate_same_distance",
        "place_rate_same_distance",
    ],
    "same_venue_time": [
        "starts_same_venue",
        "win_rate_same_venue",
        "place_rate_same_venue",
        "best_time_same_venue",
        "avg_time_same_venue",
    ],
    "same_distance_same_grade": [
        "same_distance_same_grade_start_count",
        "same_distance_same_grade_best_time",
        "same_distance_same_grade_avg_time",
        "same_distance_same_grade_history_status",
        "same_distance_same_grade_prior_history_rows_used",
    ],
    "sectional_weight": [
        "last_start_weight",
        "recent_avg_weight_5",
        "weight_delta_last_to_recent",
        "last_start_sectional_1st",
        "recent_avg_sectional_1st_5",
        "recent_best_sectional_1st_5",
        "recent_sectional_std_5",
        "sectional_time_delta_recent",
        "sectional_missing_rate_5",
    ],
    "history_rates": [
        "recent_finish_mean_3",
        "recent_finish_mean_5",
        "recent_finish_best_5",
        "recent_win_rate_5",
        "recent_place_rate_5",
        "recent_avg_margin_5",
        "recent_avg_time_5",
        "recent_best_time_5",
        "recent_time_std_5",
    ],
    "target_context": [
        "field_size",
        "target_distance_safe",
        "target_distance_source_is_safe",
        "target_grade_safe",
        "target_grade_provenance_safe",
        "grade_change_indicator",
        "grade_change_direction",
        "grade_strength_delta",
    ],
    "venue_distance": [
        "venue",
        "race_number",
        "target_month",
        "target_day_of_week",
        "target_distance_band_sprint",
        "target_distance_band_middle",
        "target_distance_band_staying",
    ],
    "box_draw": [
        "box_number",
        "box_band_inside",
        "box_band_middle",
        "box_band_outside",
    ],
}

ROLES = ("model_top", "market_top", "winner")


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
    root = ROOT.expanduser().resolve(strict=False)
    logical = output_dir.expanduser()
    if not logical.is_absolute():
        logical = root / logical
    resolved = logical.resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_scorecard_residual_feature:{relative}")
    return resolved


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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_file(path: Path) -> str:
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
        "schema_version": "scorecard_residual_feature_output_manifest_v1",
        "output_dir": relpath(output_dir),
        "files": files,
    }


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_feature_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"feature_rows_json_root_not_list:{path}")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def finite_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return None


def box_text(value: Any) -> str:
    parsed = finite_int(value)
    return str(parsed) if parsed is not None else str(value or "")


def is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def is_nondefault(value: Any) -> bool:
    if not is_present(value):
        return False
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    if isinstance(value, str):
        text = value.strip().lower()
        return text not in {"0", "0.0", "false", "none", "not_populated", "unknown"}
    return True


def field_values(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> list[Any]:
    values: list[Any] = []
    for row in rows:
        for field in fields:
            values.append(row.get(field))
    return values


def race_feature_path(scorecard_row: Mapping[str, Any]) -> Path | None:
    source = str(scorecard_row.get("winner_prediction_source_path") or "")
    if not source:
        return None
    return Path(source).parent / "shadow_feature_rows.json"


def residual_filter(row: Mapping[str, Any]) -> bool:
    market_winner_probability = finite_float(row.get("market_winner_probability")) or 0.0
    model_winner_probability = finite_float(row.get("model_winner_probability")) or 0.0
    return (
        market_winner_probability >= 0.30
        and model_winner_probability < 0.20
        and box_text(row.get("model_top_box")) != box_text(row.get("market_top_box"))
    )


def load_features_by_path(
    paths: Iterable[Path | None],
) -> tuple[dict[Path, dict[tuple[str, str], dict[str, Any]]], list[dict[str, Any]]]:
    loaded: dict[Path, dict[tuple[str, str], dict[str, Any]]] = {}
    errors: list[dict[str, Any]] = []
    for path in sorted({item for item in paths if item is not None}):
        if path in loaded:
            continue
        if not path.exists():
            errors.append({"path": str(path), "error": "feature_rows_missing"})
            loaded[path] = {}
            continue
        try:
            rows = load_feature_rows(path)
        except Exception as exc:
            errors.append({"path": str(path), "error": f"{type(exc).__name__}:{exc}"})
            loaded[path] = {}
            continue
        keyed: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            race_id = str(row.get("race_id") or "")
            box = box_text(row.get("box_number"))
            if race_id and box:
                keyed.setdefault((race_id, box), row)
        loaded[path] = keyed
    return loaded, errors


def summarize_role_family(
    rows: Sequence[Mapping[str, Any]],
    *,
    role: str,
    family: str,
    fields: Sequence[str],
) -> dict[str, Any]:
    joined_rows = [row for row in rows if isinstance(row.get(role), Mapping)]
    feature_rows = [row[role] for row in joined_rows]
    values = field_values(feature_rows, fields)
    present_values = [value for value in values if is_present(value)]
    nondefault_values = [value for value in values if is_nondefault(value)]
    present_by_field = {
        field: sum(1 for row in joined_rows if is_present(row.get(field))) for field in fields
    }
    nondefault_by_field = {
        field: sum(1 for row in joined_rows if is_nondefault(row.get(field))) for field in fields
    }
    rows_with_any_present = sum(
        1 for row in feature_rows if any(is_present(row.get(field)) for field in fields)
    )
    rows_with_any_nondefault = sum(
        1 for row in feature_rows if any(is_nondefault(row.get(field)) for field in fields)
    )
    return {
        "role": role,
        "feature_family": family,
        "field_count": len(fields),
        "residual_race_count": len(rows),
        "joined_row_count": len(joined_rows),
        "rows_with_any_present": rows_with_any_present,
        "rows_with_any_nondefault": rows_with_any_nondefault,
        "any_present_rate": rows_with_any_present / len(joined_rows) if joined_rows else None,
        "any_nondefault_rate": rows_with_any_nondefault / len(joined_rows) if joined_rows else None,
        "present_value_count": len(present_values),
        "nondefault_value_count": len(nondefault_values),
        "present_by_field_json": json.dumps(present_by_field, sort_keys=True),
        "nondefault_by_field_json": json.dumps(nondefault_by_field, sort_keys=True),
    }


def family_comparisons(role_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_family_role = {
        (row["feature_family"], row["role"]): row
        for row in role_rows
        if row.get("feature_family") and row.get("role")
    }
    comparisons: list[dict[str, Any]] = []
    for family in FEATURE_FAMILIES:
        model = by_family_role.get((family, "model_top"), {})
        market = by_family_role.get((family, "market_top"), {})
        winner = by_family_role.get((family, "winner"), {})
        model_nondefault = finite_float(model.get("any_nondefault_rate"))
        market_nondefault = finite_float(market.get("any_nondefault_rate"))
        winner_nondefault = finite_float(winner.get("any_nondefault_rate"))
        model_present = finite_float(model.get("any_present_rate"))
        market_present = finite_float(market.get("any_present_rate"))
        winner_present = finite_float(winner.get("any_present_rate"))
        comparisons.append(
            {
                "feature_family": family,
                "model_any_present_rate": model_present,
                "market_any_present_rate": market_present,
                "winner_any_present_rate": winner_present,
                "model_minus_market_any_present_rate": (
                    model_present - market_present
                    if model_present is not None and market_present is not None
                    else None
                ),
                "model_minus_winner_any_present_rate": (
                    model_present - winner_present
                    if model_present is not None and winner_present is not None
                    else None
                ),
                "model_any_nondefault_rate": model_nondefault,
                "market_any_nondefault_rate": market_nondefault,
                "winner_any_nondefault_rate": winner_nondefault,
                "model_minus_market_any_nondefault_rate": (
                    model_nondefault - market_nondefault
                    if model_nondefault is not None and market_nondefault is not None
                    else None
                ),
                "model_minus_winner_any_nondefault_rate": (
                    model_nondefault - winner_nondefault
                    if model_nondefault is not None and winner_nondefault is not None
                    else None
                ),
            }
        )
    return comparisons


def role_race_detail(
    row: Mapping[str, Any], roles: Mapping[str, Mapping[str, Any] | None]
) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "race_id": row.get("race_id"),
        "race_date": row.get("race_date"),
        "venue": row.get("venue"),
        "race_number": row.get("race_number"),
        "runner_count": row.get("runner_count"),
        "winner_box": row.get("winner_box"),
        "model_top_box": row.get("model_top_box"),
        "market_top_box": row.get("market_top_box"),
        "model_winner_probability": row.get("model_winner_probability"),
        "market_winner_probability": row.get("market_winner_probability"),
    }
    for role, feature_row in roles.items():
        detail[f"{role}_feature_joined"] = feature_row is not None
        detail[f"{role}_dog_name"] = feature_row.get("dog_name") if feature_row else None
        for family, fields in FEATURE_FAMILIES.items():
            if feature_row is None:
                present_count = 0
                nondefault_count = 0
            else:
                present_count = sum(1 for field in fields if is_present(feature_row.get(field)))
                nondefault_count = sum(
                    1 for field in fields if is_nondefault(feature_row.get(field))
                )
            detail[f"{role}_{family}_present_field_count"] = present_count
            detail[f"{role}_{family}_nondefault_field_count"] = nondefault_count
    return detail


def recommendation(
    *,
    residual_race_count: int,
    all_roles_joined_race_count: int,
    comparisons: Sequence[Mapping[str, Any]],
) -> str:
    if residual_race_count == 0:
        return "DATA_MISSING_NO_HIGH_MARKET_RESIDUAL_SLICE"
    join_rate = all_roles_joined_race_count / residual_race_count
    if join_rate < 0.95:
        return "FEATURE_JOIN_GAP_KEEP_MARKET_AS_BASELINE_COLLECT_AND_REPAIR_DATA"
    non_box_families = [row for row in comparisons if row.get("feature_family") != "box_draw"]
    model_not_better = all(
        float(row.get("model_minus_market_any_nondefault_rate") or 0.0) <= 0.05
        and float(row.get("model_minus_winner_any_nondefault_rate") or 0.0) <= 0.05
        for row in non_box_families
    )
    if model_not_better:
        return "FEATURE_COVERAGE_BLOCKER_KEEP_MARKET_AS_BASELINE_COLLECT_AND_REPAIR_DATA"
    return "RUN_REDUCED_BOX_OR_RESIDUAL_REPORT_ONLY_CHALLENGER_REVIEW"


def summary_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Scorecard Residual Feature Audit",
            "",
            f"Final status: `{report.get('final_status')}`",
            f"Recommended decision: `{report.get('recommended_decision')}`",
            f"Residual race count: `{report.get('residual_race_count')}`",
            f"All-role joined races: `{report.get('all_roles_joined_race_count')}`",
            "",
            "## Filter",
            "",
            "`market_winner_probability >= 0.30 AND model_winner_probability < 0.20 AND model_top_box != market_top_box`",
            "",
            "## Family Comparisons",
            "",
            "```json",
            json.dumps(report.get("family_comparisons") or [], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


ROLE_SUMMARY_FIELDS = [
    "role",
    "feature_family",
    "field_count",
    "residual_race_count",
    "joined_row_count",
    "rows_with_any_present",
    "rows_with_any_nondefault",
    "any_present_rate",
    "any_nondefault_rate",
    "present_value_count",
    "nondefault_value_count",
    "present_by_field_json",
    "nondefault_by_field_json",
]

COMPARISON_FIELDS = [
    "feature_family",
    "model_any_present_rate",
    "market_any_present_rate",
    "winner_any_present_rate",
    "model_minus_market_any_present_rate",
    "model_minus_winner_any_present_rate",
    "model_any_nondefault_rate",
    "market_any_nondefault_rate",
    "winner_any_nondefault_rate",
    "model_minus_market_any_nondefault_rate",
    "model_minus_winner_any_nondefault_rate",
]


def build_packet(
    *,
    scorecard_csv: Path,
    output_dir: Path,
    min_market_winner_probability: float = 0.30,
    max_model_winner_probability: float = 0.20,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    del min_market_winner_probability, max_model_winner_probability
    output_dir = unique_dir(assert_output_dir_safe(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    generated_at = generated_at or datetime.now().astimezone()

    scorecard_rows = load_csv(scorecard_csv)
    residual_rows = [row for row in scorecard_rows if residual_filter(row)]
    feature_paths = [race_feature_path(row) for row in residual_rows]
    features_by_path, feature_load_errors = load_features_by_path(feature_paths)

    joined_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    missing_join_counts = Counter()
    for score_row in residual_rows:
        path = race_feature_path(score_row)
        keyed = features_by_path.get(path or Path(), {})
        race_id = str(score_row.get("race_id") or "")
        roles: dict[str, Mapping[str, Any] | None] = {
            "model_top": keyed.get((race_id, box_text(score_row.get("model_top_box")))),
            "market_top": keyed.get((race_id, box_text(score_row.get("market_top_box")))),
            "winner": keyed.get((race_id, box_text(score_row.get("winner_box")))),
        }
        for role, feature_row in roles.items():
            if feature_row is None:
                missing_join_counts[role] += 1
        joined_row = {"scorecard_row": score_row, **roles}
        joined_rows.append(joined_row)
        detail = role_race_detail(score_row, roles)
        detail["feature_rows_path"] = str(path) if path else None
        detail["all_roles_joined"] = all(value is not None for value in roles.values())
        detail_rows.append(detail)

    role_summaries: list[dict[str, Any]] = []
    for role in ROLES:
        for family, fields in FEATURE_FAMILIES.items():
            role_summaries.append(
                summarize_role_family(
                    joined_rows,
                    role=role,
                    family=family,
                    fields=fields,
                )
            )
    comparisons = family_comparisons(role_summaries)
    all_roles_joined_race_count = sum(1 for row in detail_rows if row.get("all_roles_joined"))
    final_status = FINAL_READY if residual_rows else FINAL_DATA_MISSING
    report = {
        "schema_version": "scorecard_residual_feature_audit_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": final_status,
        "recommended_decision": recommendation(
            residual_race_count=len(residual_rows),
            all_roles_joined_race_count=all_roles_joined_race_count,
            comparisons=comparisons,
        ),
        "filter": {
            "market_winner_probability_gte": 0.30,
            "model_winner_probability_lt": 0.20,
            "requires_model_market_top_box_disagreement": True,
        },
        "scorecard_csv": relpath(scorecard_csv),
        "output_dir": relpath(output_dir),
        "scorecard_race_count": len(scorecard_rows),
        "residual_race_count": len(residual_rows),
        "feature_path_count": len({path for path in feature_paths if path is not None}),
        "feature_load_error_count": len(feature_load_errors),
        "feature_load_errors": feature_load_errors[:25],
        "all_roles_joined_race_count": all_roles_joined_race_count,
        "all_roles_joined_race_rate": (
            all_roles_joined_race_count / len(residual_rows) if residual_rows else None
        ),
        "missing_join_counts": dict(sorted(missing_join_counts.items())),
        "role_summaries": role_summaries,
        "family_comparisons": comparisons,
        "sample_residual_races": detail_rows[:25],
        "role_summary_csv": relpath(output_dir / ROLE_SUMMARY_CSV),
        "family_comparison_csv": relpath(output_dir / FAMILY_COMPARISON_CSV),
        "race_detail_csv": relpath(output_dir / RACE_DETAIL_CSV),
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }
    detail_fields = sorted({key for row in detail_rows for key in row})
    write_csv(output_dir / ROLE_SUMMARY_CSV, role_summaries, ROLE_SUMMARY_FIELDS)
    write_csv(output_dir / FAMILY_COMPARISON_CSV, comparisons, COMPARISON_FIELDS)
    write_csv(output_dir / RACE_DETAIL_CSV, detail_rows, detail_fields)
    write_json(output_dir / REPORT_FILE, report)
    write_text(output_dir / SUMMARY_MD, summary_markdown(report))
    write_text(output_dir / "final_status.txt", final_status + "\n")
    write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or (ROOT / f"{OUTPUT_PREFIX}{now_id()}_report_only")
    report = build_packet(
        scorecard_csv=args.scorecard_csv,
        output_dir=output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
