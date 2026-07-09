#!/usr/bin/env python3
"""Build a report-only weather/track and feature-utility root-cause packet.

The packet answers one bounded question: are source-safe weather/track values
available at enough coverage to justify a later report-only ablation, or is
source repair still the next blocker?

It does not train models, promote models, mutate registries, write labels, write
DB rows, emit EV, emit betting output, or control daemons.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from utils.csv_metadata import load_safe_weather_track_metadata  # noqa: E402


SCHEMA_VERSION = "weather_track_feature_utility_root_cause_v1"
OUTPUT_PREFIX = (
    "artifacts/full_evidence_orchestration_20260525/"
    "weather_track_feature_utility_root_cause_"
)
DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
DEFAULT_DB = ROOT / "greyhound_racing_data.db"
MIN_SAFE_WEATHER_TRACK_RACES = 20
MIN_SAFE_WEATHER_TRACK_RUNNER_ROW_PCT = 0.10
READY_TRAIN_HOLDOUT_COVERAGE_PCT = 0.80

FINAL_SOURCE_REPAIR = "WEATHER_TRACK_SOURCE_REPAIR_NEXT"
FINAL_READY_ABLATION = "READY_FOR_REPORT_ONLY_ABLATION"
FINAL_KEEP_COLLECTING = "KEEP_COLLECTING_ONLY_DATA_MISSING"
FINAL_BLOCKED_PROTECTED = "BLOCKED_PROTECTED_PATH_MUTATION"

NO_WRITE_GUARANTEES = {
    "report_only": True,
    "training_run": False,
    "model_artifact_write": False,
    "production_promotion": False,
    "registry_mutation": False,
    "production_pointer_update": False,
    "production_prediction_write": False,
    "db_write": False,
    "label_write": False,
    "ev_output": False,
    "betting_output": False,
    "daemon_control": False,
    "canonical_schema_mutation": False,
}

DEFAULT_PROTECTED_PATHS = (
    ROOT / "greyhound_racing_data.db",
    ROOT / "greyhound_racing_data_writable.db",
    ROOT / "model_registry/best_metadata.json",
    ROOT / "model_registry/current_production.json",
    ROOT / "docs/model_contracts/v4_feature_contract.json",
    ROOT / "artifacts/prediction_snapshots/manifest.jsonl",
    ROOT / "predictions",
)

FEATURE_FAMILIES = {
    "weather_track": (
        "weather",
        "track_condition",
        "race_time_minutes_since_midnight",
    ),
    "same_distance_time": (
        "best_time_same_distance",
        "avg_time_same_distance",
        "median_time_same_distance",
        "recent_avg_time_same_distance_5",
        "prior_same_distance_start_count",
    ),
    "same_venue_time": (
        "best_time_same_venue",
        "avg_time_same_venue",
        "place_rate_same_venue",
    ),
    "same_distance_same_grade": (
        "same_distance_same_grade_start_count",
        "same_distance_same_grade_best_time",
        "same_distance_same_grade_avg_time",
    ),
    "sectional_weight": (
        "last_start_sectional_1st",
        "last_start_weight",
        "recent_avg_sectional_1st_5",
        "recent_best_sectional_1st_5",
        "recent_avg_weight_5",
    ),
    "history_rates": (
        "career_win_rate",
        "career_place_rate",
        "prior_start_count",
        "place_rate_same_distance",
        "career_avg_finish",
        "career_best_time",
    ),
    "target_context": (
        "box_number",
        "field_size",
        "target_distance_safe",
        "target_grade_safe",
        "grade_change_indicator",
        "grade_change_direction",
        "grade_strength_delta",
    ),
    "venue_distance": (
        "race_id",
        "race_date",
        "venue",
        "race_number",
        "race_time",
    ),
}

DEFAULT_SOURCE_ROOT_HINTS = (
    ROOT / "artifacts",
    ROOT / "artifacts/full_evidence_orchestration_20260525",
    ROOT.parent
    / "greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525",
    ROOT.parent / "greyhound_racing_collector/artifacts/full_evidence_orchestration_20260525",
)


def now_id(value: datetime | None = None) -> str:
    return (value or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return os.path.relpath(path.resolve(), ROOT.resolve())
    except ValueError:
        return str(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def sha256_file(path: Path) -> str | None:
    if not path.exists() or path.is_dir():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_state(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {
            "type": "file",
            "exists": True,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if path.is_dir():
        entries: list[dict[str, Any]] = []
        for item in sorted(path.rglob("*"), key=lambda candidate: candidate.as_posix()):
            if item.is_file():
                entries.append(
                    {
                        "type": "file",
                        "path": relpath(item),
                        "bytes": item.stat().st_size,
                        "sha256": sha256_file(item),
                    }
                )
        digest = hashlib.sha256(
            "\n".join(json.dumps(entry, sort_keys=True) for entry in entries).encode(
                "utf-8"
            )
        ).hexdigest()
        return {
            "type": "directory",
            "exists": True,
            "file_count": len(entries),
            "listing_sha256": digest,
        }
    return {"type": "missing", "exists": False}


def protected_path_states(paths: Sequence[Path] | None = None) -> dict[str, dict[str, Any]]:
    return {relpath(path) or str(path): path_state(path) for path in (paths or DEFAULT_PROTECTED_PATHS)}


def assert_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative = logical.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(
            f"output_dir_must_be_weather_track_feature_utility_artifact:{relative}"
        )
    return logical


def is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        stripped = value.strip()
        return stripped != "" and stripped.casefold() not in {"none", "null", "nan", "n/a"}
    return True


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def pct(count: int, total: int) -> float | None:
    return count / total if total else None


def read_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path is None or not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, Mapping):
                rows.append(dict(value))
    return rows


def read_feature_rows(path: Path | None) -> list[dict[str, Any]]:
    value = read_json(path)
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    if isinstance(value, Mapping) and isinstance(value.get("rows"), list):
        return [dict(row) for row in value["rows"] if isinstance(row, Mapping)]
    return []


def read_csv_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    seen: set[Path] = set()
    output: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        output.append(path)
    return output


def existing_roots(paths: Sequence[Path] | None = None) -> list[Path]:
    candidates = paths if paths is not None else DEFAULT_SOURCE_ROOT_HINTS
    return dedupe_paths([path for path in candidates if path.exists()])


def csv_path_for_sidecar(sidecar_path: Path) -> Path:
    text = str(sidecar_path)
    suffix = ".metadata.json"
    if text.endswith(suffix):
        return Path(text[: -len(suffix)])
    return sidecar_path.with_suffix("")


def race_id_from_path(path: Path) -> str:
    name = path.name
    for suffix in (".metadata.json", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name[:-4] if name.endswith(".csv") else name


def sidecar_payload(sidecar_path: Path) -> Mapping[str, Any]:
    value = read_json(sidecar_path)
    return value if isinstance(value, Mapping) else {}


def sidecar_runner_count(payload: Mapping[str, Any]) -> int | None:
    for key in (
        "expert_form_runner_count",
        "runner_count",
        "accepted_runner_count",
        "field_size",
    ):
        value = payload.get(key)
        try:
            if value not in (None, ""):
                return int(value)
        except (TypeError, ValueError):
            pass
    completeness = payload.get("runner_completeness")
    if isinstance(completeness, Mapping):
        for key in ("accepted_runner_count", "canonical_runner_count", "runner_count"):
            try:
                value = completeness.get(key)
                if value not in (None, ""):
                    return int(value)
            except (TypeError, ValueError):
                pass
    final_set = payload.get("canonical_runner_set")
    if isinstance(final_set, Mapping) and isinstance(final_set.get("final_runner_boxes"), list):
        return len(final_set["final_runner_boxes"])
    return None


def discover_weather_track_source_rows(roots: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        for sidecar_path in sorted(root.rglob("*.csv.metadata.json")):
            resolved = sidecar_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            csv_path = csv_path_for_sidecar(sidecar_path)
            payload = sidecar_payload(sidecar_path)
            safe = load_safe_weather_track_metadata(csv_path)
            weather_present = is_present(safe.get("weather"))
            track_present = is_present(safe.get("track_condition"))
            accepted = bool(safe.get("weather_track_metadata_is_leakage_safe")) and (
                weather_present or track_present
            )
            rejected = list(safe.get("rejected_weather_track_metadata_sources") or [])
            rows.append(
                {
                    "race_id": race_id_from_path(csv_path),
                    "sidecar_path": relpath(sidecar_path),
                    "csv_path": relpath(csv_path),
                    "csv_exists": csv_path.exists(),
                    "runner_count": sidecar_runner_count(payload),
                    "status": "ACCEPTED" if accepted else "REJECTED",
                    "weather_present": weather_present,
                    "track_condition_present": track_present,
                    "both_weather_track_present": weather_present and track_present,
                    "weather": safe.get("weather"),
                    "track_condition": safe.get("track_condition"),
                    "metadata_is_leakage_safe": bool(payload.get("metadata_is_leakage_safe")),
                    "weather_track_metadata_is_leakage_safe": bool(
                        safe.get("weather_track_metadata_is_leakage_safe")
                    ),
                    "weather_track_metadata_source": safe.get("weather_track_metadata_source"),
                    "weather_track_metadata_source_url": json.dumps(
                        safe.get("weather_track_metadata_source_url"), sort_keys=True
                    )
                    if isinstance(safe.get("weather_track_metadata_source_url"), Mapping)
                    else safe.get("weather_track_metadata_source_url"),
                    "metadata_captured_at": safe.get("metadata_captured_at"),
                    "race_date": safe.get("race_date"),
                    "race_time": safe.get("race_time"),
                    "rejected_reasons": ";".join(sorted(set(rejected))),
                }
            )
    return rows


def discover_feature_row_paths(roots: Sequence[Path]) -> list[Path]:
    candidates: list[Path] = []
    allowed_markers = (
        "daily_race_ingest_shadow_",
        "shadow_autopilot_v1_",
        "shadow_evaluation",
        "live_uv_",
        "expert_form_shadow_feature_row_backfill_",
    )
    for root in roots:
        for path in root.rglob("shadow_feature_rows.json"):
            text = path.as_posix()
            if any(marker in text for marker in allowed_markers):
                candidates.append(path)
    return dedupe_paths(sorted(candidates, key=lambda item: item.as_posix()))


def discover_policy_paths(roots: Sequence[Path]) -> list[Path]:
    candidates: list[Path] = []
    for root in roots:
        for name in ("active_feature_policy_report.json", "shadow_manifest.json"):
            candidates.extend(root.rglob(name))
    return dedupe_paths(sorted(candidates, key=lambda item: item.as_posix()))


def load_feature_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in paths:
        for row in read_feature_rows(path):
            copy = dict(row)
            copy["_source_path"] = relpath(path)
            output.append(copy)
    return output


def policy_summary(paths: Sequence[Path]) -> dict[str, Any]:
    inactive: set[str] = set()
    source_paths = []
    active_counts = []
    for path in paths:
        payload = read_json(path)
        if not isinstance(payload, Mapping):
            continue
        source_paths.append(relpath(path))
        inactive.update(
            str(item)
            for item in payload.get("inactive_features_due_to_train_all_missing") or []
        )
        if payload.get("active_feature_count") is not None:
            active_counts.append(payload.get("active_feature_count"))
    return {
        "source_paths": source_paths,
        "inactive_features_due_to_train_all_missing": sorted(inactive),
        "latest_active_feature_count": active_counts[-1] if active_counts else None,
    }


def row_weather_track_acceptance(row: Mapping[str, Any]) -> dict[str, bool]:
    metadata_safe = row.get("metadata_is_leakage_safe") is True
    sidecar_safe = row.get("weather_track_metadata_from_sidecar") is True or row.get(
        "weather_track_metadata_is_leakage_safe"
    ) is True
    weather = is_present(row.get("weather"))
    track = is_present(row.get("track_condition"))
    weather_backed = row.get("weather_source_backed") is True or (
        sidecar_safe and weather and row.get("weather_track_metadata_source")
    )
    track_backed = row.get("track_condition_source_backed") is True or (
        sidecar_safe and track and row.get("weather_track_metadata_source")
    )
    weather_ok = bool(metadata_safe and sidecar_safe and weather and weather_backed)
    track_ok = bool(metadata_safe and sidecar_safe and track and track_backed)
    return {
        "weather": weather_ok,
        "track_condition": track_ok,
        "any": weather_ok or track_ok,
        "both": weather_ok and track_ok,
    }


def split_name(row: Mapping[str, Any]) -> str:
    for key in ("split", "dataset_split", "shadow_split", "train_holdout_split"):
        value = row.get(key)
        if is_present(value):
            text = str(value).strip().casefold()
            if "train" in text:
                return "train"
            if "hold" in text or "test" in text or "eval" in text:
                return "holdout"
    return "unknown"


def weather_track_feature_coverage_rows(feature_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    total = len(feature_rows)
    by_split_total = Counter(split_name(row) for row in feature_rows)
    accepted = [row_weather_track_acceptance(row) for row in feature_rows]
    rows = []
    for feature in ("weather", "track_condition"):
        present = sum(is_present(row.get(feature)) for row in feature_rows)
        accepted_present = sum(item[feature] for item in accepted)
        unique_values = sorted(
            {
                str(row.get(feature)).strip()
                for row in feature_rows
                if is_present(row.get(feature)) and row_weather_track_acceptance(row)[feature]
            }
        )
        values = [
            str(row.get(feature)).strip()
            for row in feature_rows
            if is_present(row.get(feature)) and row_weather_track_acceptance(row)[feature]
        ]
        dominant = Counter(values).most_common(1)[0][1] / len(values) if values else None
        row = {
            "feature": feature,
            "feature_rows": total,
            "raw_present_rows": present,
            "raw_present_pct": pct(present, total),
            "accepted_source_backed_rows": accepted_present,
            "accepted_source_backed_pct": pct(accepted_present, total),
            "unique_accepted_values": len(unique_values),
            "default_dominance_pct": dominant,
            "status": "ACCEPTED_COVERAGE"
            if accepted_present
            else "DATA_MISSING_OR_REJECTED",
        }
        for split in ("train", "holdout", "unknown"):
            split_rows = [
                index
                for index, source_row in enumerate(feature_rows)
                if split_name(source_row) == split
            ]
            split_total = by_split_total.get(split, 0)
            split_accepted = sum(accepted[index][feature] for index in split_rows)
            row[f"{split}_rows"] = split_total
            row[f"{split}_accepted_rows"] = split_accepted
            row[f"{split}_accepted_pct"] = pct(split_accepted, split_total)
        rows.append(row)

    both_rows = sum(item["both"] for item in accepted)
    any_rows = sum(item["any"] for item in accepted)
    both_races = {
        str(row.get("race_id") or "")
        for row, item in zip(feature_rows, accepted)
        if item["both"] and row.get("race_id")
    }
    any_races = {
        str(row.get("race_id") or "")
        for row, item in zip(feature_rows, accepted)
        if item["any"] and row.get("race_id")
    }
    rows.append(
        {
            "feature": "weather_track_both",
            "feature_rows": total,
            "raw_present_rows": sum(
                is_present(row.get("weather")) and is_present(row.get("track_condition"))
                for row in feature_rows
            ),
            "raw_present_pct": pct(
                sum(
                    is_present(row.get("weather")) and is_present(row.get("track_condition"))
                    for row in feature_rows
                ),
                total,
            ),
            "accepted_source_backed_rows": both_rows,
            "accepted_source_backed_pct": pct(both_rows, total),
            "accepted_source_backed_races": len(both_races),
            "unique_accepted_values": None,
            "default_dominance_pct": None,
            "status": "ACCEPTED_COVERAGE" if both_rows else "DATA_MISSING_OR_REJECTED",
        }
    )
    rows.append(
        {
            "feature": "weather_track_any",
            "feature_rows": total,
            "raw_present_rows": sum(
                is_present(row.get("weather")) or is_present(row.get("track_condition"))
                for row in feature_rows
            ),
            "raw_present_pct": pct(
                sum(
                    is_present(row.get("weather")) or is_present(row.get("track_condition"))
                    for row in feature_rows
                ),
                total,
            ),
            "accepted_source_backed_rows": any_rows,
            "accepted_source_backed_pct": pct(any_rows, total),
            "accepted_source_backed_races": len(any_races),
            "unique_accepted_values": None,
            "default_dominance_pct": None,
            "status": "ACCEPTED_COVERAGE" if any_rows else "DATA_MISSING_OR_REJECTED",
        }
    )
    return rows


def source_coverage_summary(source_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in source_rows if row.get("status") == "ACCEPTED"]
    both = [row for row in accepted if row.get("both_weather_track_present")]
    any_safe = [
        row
        for row in accepted
        if row.get("weather_present") or row.get("track_condition_present")
    ]
    total_runner_rows = sum(int(row.get("runner_count") or 0) for row in source_rows)
    any_runner_rows = sum(int(row.get("runner_count") or 0) for row in any_safe)
    both_runner_rows = sum(int(row.get("runner_count") or 0) for row in both)
    return {
        "sidecar_rows_scanned": len(source_rows),
        "accepted_sidecar_rows": len(accepted),
        "accepted_any_weather_track_races": len({row.get("race_id") for row in any_safe}),
        "accepted_both_weather_track_races": len({row.get("race_id") for row in both}),
        "accepted_any_weather_track_runner_rows": any_runner_rows,
        "accepted_both_weather_track_runner_rows": both_runner_rows,
        "total_sidecar_runner_rows": total_runner_rows,
        "accepted_any_weather_track_runner_row_pct": pct(any_runner_rows, total_runner_rows),
        "accepted_both_weather_track_runner_row_pct": pct(both_runner_rows, total_runner_rows),
        "rejected_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in source_rows
                    for reason in str(row.get("rejected_reasons") or "").split(";")
                    if reason
                ).items()
            )
        ),
    }


def extract_importance_map(roots: Sequence[Path]) -> tuple[dict[str, float], list[str]]:
    paths: list[str] = []
    importances: dict[str, float] = {}
    candidate_names = (
        "shadow_training_report.json",
        "feature_importance.json",
        "feature_importances.json",
    )
    for root in roots:
        for name in candidate_names:
            for path in root.rglob(name):
                payload = read_json(path)
                mapping = importance_payload_to_map(payload)
                if not mapping:
                    continue
                paths.append(relpath(path) or str(path))
                importances = mapping
    return importances, paths


def importance_payload_to_map(payload: Any) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    candidates = (
        payload.get("feature_importance"),
        payload.get("feature_importances"),
        payload.get("importances"),
    )
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return {
                str(key): float(value)
                for key, value in candidate.items()
                if safe_float(value) is not None
            }
        if isinstance(candidate, list):
            output: dict[str, float] = {}
            for item in candidate:
                if not isinstance(item, Mapping):
                    continue
                feature = item.get("feature") or item.get("name")
                value = safe_float(item.get("importance") or item.get("value"))
                if feature and value is not None:
                    output[str(feature)] = value
            if output:
                return output
    return {}


def feature_family_utility_rows(
    feature_rows: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    importances: Mapping[str, float],
) -> list[dict[str, Any]]:
    inactive = set(policy.get("inactive_features_due_to_train_all_missing") or [])
    total_importance = sum(abs(value) for value in importances.values())
    top20 = {
        feature
        for feature, _value in sorted(
            importances.items(), key=lambda item: abs(item[1]), reverse=True
        )[:20]
    }
    rows = []
    total_rows = len(feature_rows)
    for family, fields in FEATURE_FAMILIES.items():
        present_values: list[str] = []
        present_by_field: dict[str, int] = {}
        rows_with_any = 0
        for feature_row in feature_rows:
            if any(is_present(feature_row.get(field)) for field in fields):
                rows_with_any += 1
        for field in fields:
            values = [row.get(field) for row in feature_rows if is_present(row.get(field))]
            present_by_field[field] = len(values)
            present_values.extend(str(value) for value in values)
        dominant = (
            Counter(present_values).most_common(1)[0][1] / len(present_values)
            if present_values
            else None
        )
        family_importance = sum(abs(importances.get(field, 0.0)) for field in fields)
        rows.append(
            {
                "feature_family": family,
                "field_count": len(fields),
                "rows_with_any_present": rows_with_any,
                "total_feature_rows": total_rows,
                "coverage_pct": pct(rows_with_any, total_rows),
                "unique_present_values": len(set(present_values)),
                "default_dominance_pct": dominant,
                "importance_share": family_importance / total_importance
                if total_importance
                else None,
                "top20_importance_features": ";".join(
                    field for field in fields if field in top20
                ),
                "inactive_train_all_missing_fields": ";".join(
                    field for field in fields if field in inactive
                ),
                "present_by_field_json": json.dumps(present_by_field, sort_keys=True),
                "utility_status": "DATA_MISSING_IMPORTANCE"
                if not importances
                else "REVIEW_ONLY",
            }
        )
    return rows


def inspect_optional_db(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {"status": "DATA_MISSING", "reason": "db_missing", "path": relpath(db_path)}
    size = db_path.stat().st_size
    if size == 0:
        return {
            "status": "DATA_MISSING",
            "reason": "db_zero_bytes",
            "path": relpath(db_path),
            "bytes": size,
        }
    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        tables = [str(row["name"]) for row in table_rows]
        counts: dict[str, int | str] = {}
        for table in tables:
            try:
                counts[table] = int(
                    connection.execute(f'SELECT COUNT(*) AS n FROM "{table}"').fetchone()["n"]
                )
            except sqlite3.Error as exc:
                counts[table] = f"count_failed:{type(exc).__name__}"
        connection.close()
    except sqlite3.Error as exc:
        return {
            "status": "DATA_MISSING",
            "reason": f"db_unreadable:{type(exc).__name__}",
            "path": relpath(db_path),
            "bytes": size,
        }
    if not tables:
        return {
            "status": "DATA_MISSING",
            "reason": "db_tableless",
            "path": relpath(db_path),
            "bytes": size,
        }
    return {
        "status": "OPTIONAL_EVIDENCE_ONLY",
        "reason": "db_present_but_not_accepted_without_prejump_collection_timestamp",
        "path": relpath(db_path),
        "bytes": size,
        "tables": tables,
        "table_counts": counts,
    }


def train_holdout_weather_track_coverage(
    coverage_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_feature = {str(row.get("feature")): row for row in coverage_rows}
    both = by_feature.get("weather_track_both", {})
    weather = by_feature.get("weather", {})
    track = by_feature.get("track_condition", {})
    train_pct = both.get("train_accepted_pct")
    holdout_pct = both.get("holdout_accepted_pct")
    return {
        "weather_train_accepted_pct": weather.get("train_accepted_pct"),
        "weather_holdout_accepted_pct": weather.get("holdout_accepted_pct"),
        "track_train_accepted_pct": track.get("train_accepted_pct"),
        "track_holdout_accepted_pct": track.get("holdout_accepted_pct"),
        "both_train_accepted_pct": train_pct,
        "both_holdout_accepted_pct": holdout_pct,
        "split_evidence_available": train_pct is not None and holdout_pct is not None,
        "ready_train_holdout_coverage": (
            train_pct is not None
            and holdout_pct is not None
            and train_pct >= READY_TRAIN_HOLDOUT_COVERAGE_PCT
            and holdout_pct >= READY_TRAIN_HOLDOUT_COVERAGE_PCT
        ),
    }


def decide(
    *,
    source_summary: Mapping[str, Any],
    feature_coverage: Sequence[Mapping[str, Any]],
    family_utility: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    both_races = int(source_summary.get("accepted_both_weather_track_races") or 0)
    both_pct = source_summary.get("accepted_both_weather_track_runner_row_pct")
    if both_pct is None:
        both_pct = 0.0
    if both_races < MIN_SAFE_WEATHER_TRACK_RACES or both_pct < MIN_SAFE_WEATHER_TRACK_RUNNER_ROW_PCT:
        return {
            "final_status": FINAL_SOURCE_REPAIR,
            "ablation_status": "NOT_RUN_SOURCE_COVERAGE_LOW",
            "reason": (
                "source-safe rows with both weather and track_condition are below "
                "the predeclared race or runner-row coverage gate"
            ),
        }
    split = train_holdout_weather_track_coverage(feature_coverage)
    weather_family = next(
        (row for row in family_utility if row.get("feature_family") == "weather_track"),
        {},
    )
    nonflat = (
        int(weather_family.get("unique_present_values") or 0) >= 3
        and (weather_family.get("default_dominance_pct") is None
             or float(weather_family.get("default_dominance_pct") or 0.0) < 0.95)
    )
    if split["ready_train_holdout_coverage"] and nonflat:
        return {
            "final_status": FINAL_READY_ABLATION,
            "ablation_status": "APPROVED_TO_RUN_REPORT_ONLY_ABLATION",
            "reason": "source-safe train and holdout coverage are meaningful and non-flat",
        }
    return {
        "final_status": FINAL_KEEP_COLLECTING,
        "ablation_status": "NOT_RUN_TRAIN_HOLDOUT_OR_UTILITY_EVIDENCE_MISSING",
        "reason": (
            "source-safe weather/track evidence exists, but train/holdout "
            "coverage or non-flat utility evidence is not sufficient"
        ),
    }


def failure_attribution_rows(
    *,
    decision: Mapping[str, Any],
    source_summary: Mapping[str, Any],
    policy: Mapping[str, Any],
    db_status: Mapping[str, Any],
    feature_coverage: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    split = train_holdout_weather_track_coverage(feature_coverage)
    inactive = set(policy.get("inactive_features_due_to_train_all_missing") or [])
    rows = [
        {
            "rank": 1,
            "failure_area": "weather_track_source_coverage",
            "status": decision.get("final_status"),
            "evidence": (
                f"accepted_both_races={source_summary.get('accepted_both_weather_track_races')}; "
                f"accepted_both_runner_pct={source_summary.get('accepted_both_weather_track_runner_row_pct')}"
            ),
            "next_action": "repair source capture before ablation"
            if decision.get("final_status") == FINAL_SOURCE_REPAIR
            else "continue to train/holdout utility review",
        },
        {
            "rank": 2,
            "failure_area": "train_all_missing_quarantine_policy",
            "status": "BLOCKED" if {"weather", "track_condition"} & inactive else "CLEAR_OR_DATA_MISSING",
            "evidence": ";".join(sorted(inactive)),
            "next_action": "do not activate inactive features without report-only gate pass",
        },
        {
            "rank": 3,
            "failure_area": "train_holdout_weather_track_coverage",
            "status": "READY" if split["ready_train_holdout_coverage"] else "DATA_MISSING_OR_LOW",
            "evidence": json.dumps(split, sort_keys=True),
            "next_action": "run report-only ablation only after train and holdout both clear 80 percent",
        },
        {
            "rank": 4,
            "failure_area": "optional_db_weather_track_evidence",
            "status": db_status.get("status"),
            "evidence": db_status.get("reason"),
            "next_action": "DB rows remain optional unless pre-jump collection timestamp is proven",
        },
    ]
    return rows


def source_roots_report(roots: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": relpath(path),
            "absolute_path": str(path.resolve()),
            "exists": path.exists(),
        }
        for path in roots
    ]


def build_packet(
    *,
    artifact_roots: Sequence[Path] | None = None,
    db_path: Path = DEFAULT_DB,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    roots = existing_roots(artifact_roots)
    source_rows = discover_weather_track_source_rows(roots)
    feature_paths = discover_feature_row_paths(roots)
    feature_rows = load_feature_rows(feature_paths)
    policy = policy_summary(discover_policy_paths(roots))
    importances, importance_paths = extract_importance_map(roots)
    source_summary = source_coverage_summary(source_rows)
    feature_coverage = weather_track_feature_coverage_rows(feature_rows)
    family_utility = feature_family_utility_rows(feature_rows, policy, importances)
    db_status = inspect_optional_db(db_path)
    decision = decide(
        source_summary=source_summary,
        feature_coverage=feature_coverage,
        family_utility=family_utility,
    )
    failures = failure_attribution_rows(
        decision=decision,
        source_summary=source_summary,
        policy=policy,
        db_status=db_status,
        feature_coverage=feature_coverage,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_roots": source_roots_report(roots),
        "feature_row_paths": [relpath(path) for path in feature_paths],
        "feature_row_file_count": len(feature_paths),
        "feature_row_count": len(feature_rows),
        "importance_paths": importance_paths,
        "source_coverage_summary": source_summary,
        "train_holdout_weather_track_coverage": train_holdout_weather_track_coverage(
            feature_coverage
        ),
        "policy_summary": policy,
        "optional_db_status": db_status,
        "decision": decision,
        "ablation": {
            "status": decision.get("ablation_status"),
            "production_mutation": False,
            "training_run": False,
            "reason": decision.get("reason"),
        },
        "no_write_guarantees": NO_WRITE_GUARANTEES,
    }
    ledgers = {
        "weather_track_source_coverage": list(source_rows),
        "weather_track_feature_coverage": list(feature_coverage),
        "weather_track_leakage_ledger": list(source_rows),
        "feature_family_utility": list(family_utility),
        "failure_attribution": failures,
    }
    return report, ledgers


def output_manifest(output_dir: Path) -> list[dict[str, Any]]:
    return [
        {"path": relpath(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(output_dir.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
    ]


def build_summary(report: Mapping[str, Any]) -> str:
    decision = report.get("decision") or {}
    source = report.get("source_coverage_summary") or {}
    split = report.get("train_holdout_weather_track_coverage") or {}
    return "\n".join(
        [
            "# Weather/Track Feature Utility Root Cause",
            "",
            f"- Final status: `{decision.get('final_status')}`",
            f"- Ablation status: `{decision.get('ablation_status')}`",
            f"- Reason: `{decision.get('reason')}`",
            f"- Feature rows scanned: `{report.get('feature_row_count')}`",
            f"- Sidecars scanned: `{source.get('sidecar_rows_scanned')}`",
            f"- Accepted both-weather-track races: `{source.get('accepted_both_weather_track_races')}`",
            f"- Accepted both-weather-track runner-row pct: `{source.get('accepted_both_weather_track_runner_row_pct')}`",
            f"- Train/holdout split evidence available: `{split.get('split_evidence_available')}`",
            f"- Protected paths unchanged: `{report.get('protected_paths_unchanged')}`",
            "",
            "No production promotion, registry mutation, DB writes, label writes, schema mutation, EV output, betting output, daemon control, or model training was performed.",
            "",
        ]
    )


def build_board_outputs(report: Mapping[str, Any]) -> dict[str, str]:
    decision = report.get("decision") or {}
    source = report.get("source_coverage_summary") or {}
    final_status = str(decision.get("final_status"))
    if final_status == FINAL_READY_ABLATION:
        board_decision = "proceed"
        next_goal = (
            "Run a report-only weather/track ablation on predeclared, labeled "
            "train/holdout rows with no production mutation."
        )
        minority = "none_found: coverage and non-flat gates passed"
    elif final_status == FINAL_SOURCE_REPAIR:
        board_decision = "revise_plan"
        next_goal = (
            "Repair and widen source-safe weather/track capture before any "
            "ablation."
        )
        minority = (
            "Ablation now would test source absence, not feature utility; "
            "accepted both-weather-track coverage is below gate."
        )
    else:
        board_decision = "park"
        next_goal = "Keep collecting source-safe rows until train/holdout evidence exists."
        minority = "DATA_MISSING train/holdout coverage prevents a defensible ablation."
    board_md = "\n".join(
        [
            "# Weather/Track Review Board",
            "",
            "## Decision",
            "",
            f"- Board decision: `{board_decision}`",
            f"- Packet final status: `{final_status}`",
            f"- Reason: `{decision.get('reason')}`",
            "",
            "## Perspectives",
            "",
            "- Architect: keep report-only boundaries; do not activate inactive features without train/holdout coverage.",
            "- Skeptic/red-team: source-safe both weather+track coverage is the hard gate; partial weather-only rows are not enough.",
            "- Product/value: ablation is only valuable if it can distinguish signal from missing source plumbing.",
            "- Validation/test: require protected hashes unchanged and focused unit tests around leakage and coverage gates.",
            "- Repo hygiene/git guard: tracked greyhound code is the active lane; unrelated Tenn dirty branch is out of scope.",
            "- Domain: Sportsbet/open-meteo sidecars are acceptable only with pre-jump timestamps and non-result URLs.",
            "",
            "## Evidence",
            "",
            f"- Sidecars scanned: `{source.get('sidecar_rows_scanned')}`",
            f"- Accepted both-weather-track races: `{source.get('accepted_both_weather_track_races')}`",
            f"- Accepted both-weather-track runner-row pct: `{source.get('accepted_both_weather_track_runner_row_pct')}`",
            "",
        ]
    )
    board_decision_json = {
        "schema_version": "tenn_review_board_decision_v1",
        "decision": board_decision,
        "packet_final_status": final_status,
        "reason": decision.get("reason"),
        "minority_objection": minority,
        "ledger_sources_checked": [
            "current greyhound git status",
            "local artifact roots",
            "external autonomous accuracy artifact root when present",
            "memory duplicate-work summary",
        ],
        "duplicate_work_classification": "SUPERSEDED_IGNORE",
        "matching_candidates": [
            "prediction_accuracy_system_audit packets cover broad system audit but not dedicated weather/track decision packet",
            "expert_form_schema_trial_ablation covers expert_form fields only",
        ],
        "duplicate_work_decision": "new dedicated packet is justified; ablation only if packet gate passes",
        "next_goal": next_goal,
    }
    next_goal_md = "\n".join(["# Next Goal", "", next_goal, ""])
    return {
        "BOARD.md": board_md,
        "BOARD_DECISION.json": json.dumps(board_decision_json, indent=2, sort_keys=True)
        + "\n",
        "NEXT_GOAL.md": next_goal_md,
    }


def run_packet(
    *,
    output_dir: Path | None = None,
    artifact_roots: Sequence[Path] | None = None,
    db_path: Path = DEFAULT_DB,
    write_board: bool = True,
) -> dict[str, Any]:
    output_dir = output_dir or DEFAULT_EVIDENCE_ROOT / f"weather_track_feature_utility_root_cause_{now_id()}_report_only"
    output_dir = assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    protected_before = protected_path_states()
    report, ledgers = build_packet(artifact_roots=artifact_roots, db_path=db_path)
    protected_after = protected_path_states()
    report["protected_paths_before"] = protected_before
    report["protected_paths_after"] = protected_after
    report["protected_paths_unchanged"] = protected_before == protected_after
    if not report["protected_paths_unchanged"]:
        report["decision"] = {
            "final_status": FINAL_BLOCKED_PROTECTED,
            "ablation_status": "NOT_RUN_PROTECTED_PATH_MUTATION",
            "reason": "protected path hashes changed during report-only packet build",
        }
    write_csv(
        output_dir / "weather_track_source_coverage.csv",
        ledgers["weather_track_source_coverage"],
        [
            "race_id",
            "sidecar_path",
            "csv_path",
            "csv_exists",
            "runner_count",
            "status",
            "weather_present",
            "track_condition_present",
            "both_weather_track_present",
            "weather",
            "track_condition",
            "metadata_is_leakage_safe",
            "weather_track_metadata_is_leakage_safe",
            "weather_track_metadata_source",
            "weather_track_metadata_source_url",
            "metadata_captured_at",
            "race_date",
            "race_time",
            "rejected_reasons",
        ],
    )
    write_csv(
        output_dir / "weather_track_feature_coverage.csv",
        ledgers["weather_track_feature_coverage"],
        [
            "feature",
            "feature_rows",
            "raw_present_rows",
            "raw_present_pct",
            "accepted_source_backed_rows",
            "accepted_source_backed_pct",
            "accepted_source_backed_races",
            "unique_accepted_values",
            "default_dominance_pct",
            "status",
            "train_rows",
            "train_accepted_rows",
            "train_accepted_pct",
            "holdout_rows",
            "holdout_accepted_rows",
            "holdout_accepted_pct",
            "unknown_rows",
            "unknown_accepted_rows",
            "unknown_accepted_pct",
        ],
    )
    write_csv(
        output_dir / "weather_track_leakage_ledger.csv",
        ledgers["weather_track_leakage_ledger"],
        [
            "race_id",
            "sidecar_path",
            "status",
            "metadata_is_leakage_safe",
            "weather_track_metadata_is_leakage_safe",
            "weather_track_metadata_source",
            "weather_track_metadata_source_url",
            "metadata_captured_at",
            "race_date",
            "race_time",
            "rejected_reasons",
        ],
    )
    write_csv(
        output_dir / "feature_family_utility.csv",
        ledgers["feature_family_utility"],
        [
            "feature_family",
            "field_count",
            "rows_with_any_present",
            "total_feature_rows",
            "coverage_pct",
            "unique_present_values",
            "default_dominance_pct",
            "importance_share",
            "top20_importance_features",
            "inactive_train_all_missing_fields",
            "present_by_field_json",
            "utility_status",
        ],
    )
    write_csv(
        output_dir / "failure_attribution.csv",
        ledgers["failure_attribution"],
        ["rank", "failure_area", "status", "evidence", "next_action"],
    )
    write_json(
        output_dir / "weather_track_feature_utility_root_cause_report.json",
        report,
    )
    if write_board:
        for name, text in build_board_outputs(report).items():
            write_text(output_dir / name, text)
    write_text(output_dir / "SUMMARY.md", build_summary(report))
    write_text(output_dir / "final_status.txt", str(report["decision"]["final_status"]) + "\n")
    write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
    return {
        "output_dir": relpath(output_dir),
        "final_status": report["decision"]["final_status"],
        "ablation_status": report["decision"]["ablation_status"],
        "feature_row_count": report["feature_row_count"],
        "accepted_both_weather_track_races": report["source_coverage_summary"][
            "accepted_both_weather_track_races"
        ],
        "protected_paths_unchanged": report["protected_paths_unchanged"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--artifact-root", action="append", type=Path, default=[])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--no-board", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    roots = args.artifact_root or None
    result = run_packet(
        output_dir=args.output_dir,
        artifact_roots=roots,
        db_path=args.db,
        write_board=not args.no_board,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
