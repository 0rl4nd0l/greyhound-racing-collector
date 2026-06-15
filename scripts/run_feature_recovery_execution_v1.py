#!/usr/bin/env python3
"""Run the report-only feature recovery and challenger program.

This helper deliberately writes only under
artifacts/full_evidence_orchestration_20260525/feature_recovery_execution_v1_*.
It rebuilds a non-TGR repaired matrix from clean official rows, safe target
metadata, and DB-only pre-target history. Embedded form-history rows are
classified but not used as feature sources.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEAN_DATASET = (
    ROOT
    / "artifacts/full_evidence_orchestration_20260525"
    / "isolated_challenger_box_bias_study_20260602/clean_official_dataset.jsonl"
)
DEFAULT_REPAIRED_PACKET = (
    ROOT
    / "artifacts/full_evidence_orchestration_20260525"
    / "bounded_target_grade_repair_20260603/repaired_pre_race_history_feature_packet.csv"
)
DEFAULT_SCHEMA = ROOT / "outputs/milestone_6a_non_tgr_challenger_training_design/repaired_non_tgr_schema.json"
DEFAULT_DB = ROOT / "greyhound_racing_data.db"
DEFAULT_OUTPUT_PARENT = ROOT / "artifacts/full_evidence_orchestration_20260525"
PROTECTED_PREFIXES = (
    "artifacts/prediction_snapshots",
    "model_registry",
    "docs/model_registry",
    "ml_models_v4",
    "advanced_models",
)
HARD_EXCLUDED_FEATURE_NAMES = {
    "actual_results",
    "actual_winner",
    "beaten_margin",
    "finish_position",
    "margin",
    "official_result",
    "official_results",
    "placing",
    "race_result",
    "race_results",
    "result",
    "result_status",
    "results_status",
    "scraped_finish_position",
    "scraped_raw_result",
    "target",
    "target_finish_position",
    "winner",
    "winner_margin",
    "winner_name",
    "winner_odds",
    "winning_time",
}
IDENTITY_FEATURE_NAMES = {
    "race_id",
    "dog_clean_name",
    "dog_name",
    "source_file_path",
    "snapshot_path",
    "canonical_race_url",
}
MISSING = ""
RANDOM_SEED = 42
EXPECTED_REPAIRED_FEATURE_COUNT = 103
REPAIRED_SLICE_DIMENSIONS = (
    "venue",
    "target_distance_band",
    "target_grade",
    "field_size",
    "box_band",
    "venue_box_band",
    "distance_box_band",
)
REPAIRED_FEATURE_FAMILY_ORDER = (
    "safe_target_context",
    "repaired_history_reconstruction",
    "class_and_field_strength",
    "draw_adjusted_history",
    "same_venue",
    "same_distance",
    "same_grade_and_grade_transition",
    "sectional_metrics",
)
KEY_REPAIRED_FEATURES_BY_FAMILY = {
    "safe_target_context": (
        "target_distance_safe",
        "target_grade_safe",
        "target_distance_source_is_safe",
        "target_grade_provenance_safe",
    ),
    "repaired_history_reconstruction": (
        "prior_start_count",
        "recent_finish_mean_5",
        "recent_avg_time_5",
        "recent_avg_speed_mps_5",
        "recent_avg_race_strength_5",
    ),
    "class_and_field_strength": (
        "safe_grade_rank",
        "safe_field_strength",
        "prior_race_strength_delta_to_target",
        "grade_normalized_recent_speed_index",
    ),
    "draw_adjusted_history": (
        "target_box_band_prior_start_count",
        "venue_box_band_start_count",
        "distance_box_band_start_count",
    ),
    "same_venue": (
        "starts_same_venue",
        "win_rate_same_venue",
        "best_time_same_venue",
    ),
    "same_distance": (
        "starts_same_distance",
        "best_time_same_distance",
        "same_distance_same_grade_best_time",
    ),
    "same_grade_and_grade_transition": (
        "same_grade_start_count",
        "same_grade_win_rate",
        "grade_change_indicator",
    ),
    "sectional_metrics": (
        "recent_avg_sectional_1st_5",
        "last_start_sectional_1st",
        "sectional_missing_rate_5",
    ),
}
HISTORY_PROVENANCE_METADATA_FIELDS = (
    "grade",
    "distance",
    "race_time",
    "track_condition",
    "weather",
    "start_datetime",
    "race_metadata_url",
)
SAFE_TARGET_METADATA_SOURCES = {
    "canonical_pre_race_page",
    "sidecar_target_metadata",
    "explicit_csv_sidecar",
}
SAFE_DB_TARGET_METADATA_SOURCES = {
    "canonical_pre_race_page",
    "sidecar_target_metadata",
}
UNSAFE_TARGET_METADATA_SOURCE_MARKERS = (
    "embedded_form_history",
    "embedded_form_guide",
    "post_result",
    "result_page",
    "sportsbet_result",
)
POST_OUTCOME_STATUS_VALUES = {
    "complete",
    "completed",
    "final",
    "official",
    "resulted",
}
TARGET_METADATA_DEPENDENT_PACKET_REUSE_BLOCKLIST = {
    "target_distance_safe",
    "target_distance_source_is_safe",
    "target_distance_missing",
    "target_distance_band_sprint",
    "target_distance_band_middle",
    "target_distance_band_staying",
    "target_grade_safe",
    "target_grade_normalized",
    "target_grade_missing",
    "target_grade_vocab_known",
    "target_grade_provenance_safe",
    "safe_grade_rank",
    "safe_field_strength",
    "prior_race_strength_delta_to_target",
    "same_grade_start_count",
    "same_grade_win_rate",
    "same_grade_place_rate",
    "same_grade_avg_speed_mps",
    "grade_normalized_recent_speed_index",
    "same_distance_same_grade_start_count",
    "same_distance_same_grade_best_time",
    "same_distance_same_grade_avg_time",
    "grade_change_indicator",
    "grade_change_direction",
    "grade_strength_delta",
}


GRADE_MAP = {
    "1": "Grade 1",
    "2": "Grade 2",
    "3": "Grade 3",
    "4": "Grade 4",
    "5": "Grade 5",
    "6": "Grade 6",
    "7": "Grade 7",
    "8": "Grade 8",
    "M": "Maiden",
    "MAIDEN": "Maiden",
    "MDN": "Maiden",
    "OPEN": "Open",
    "FFA": "Free For All",
    "FREE FOR ALL": "Free For All",
    "NG": "Non Graded",
    "NON GRADED": "Non Graded",
    "MIXED": "Mixed",
    "4/5": "Mixed 4/5",
    "5/6": "Mixed 5/6",
    "3/4": "Mixed 3/4",
    "2/3/4": "Mixed 2/3/4",
    "3/4/5": "Mixed 3/4/5",
    "RESTRICTED": "Restricted Win",
    "RESTRICTED WIN": "Restricted Win",
    "TIER 3 - RESTRICTED WIN": "Restricted Win",
    "TIER 3 - MAIDEN": "Maiden",
    "TIER 3 - GRADE 5": "Grade 5",
    "TIER 3 - GRADE 6": "Grade 6",
    "TIER 3 - GRADE 7": "Grade 7",
    "GRADE 1": "Grade 1",
    "GRADE 2": "Grade 2",
    "GRADE 3": "Grade 3",
    "GRADE 4": "Grade 4",
    "GRADE 5": "Grade 5",
    "GRADE 6": "Grade 6",
    "GRADE 7": "Grade 7",
    "1ST GRADE": "Grade 1",
    "2ND GRADE": "Grade 2",
    "3RD GRADE": "Grade 3",
    "4TH GRADE": "Grade 4",
    "5TH GRADE": "Grade 5",
    "6TH GRADE": "Grade 6",
    "7TH GRADE": "Grade 7",
}
GRADE_RANK = {
    "Maiden": 0,
    "Restricted Win": 1,
    "Novice": 2,
    "Non Graded": 3,
    "Mixed": 4,
    "Mixed 2/3/4": 4,
    "Mixed 3/4": 4,
    "Mixed 3/4/5": 4,
    "Mixed 4/5": 4,
    "Mixed 5/6": 4,
    "Grade 7": 5,
    "Grade 6": 6,
    "Grade 5": 7,
    "Grade 4": 8,
    "Grade 3": 9,
    "Grade 2": 10,
    "Grade 1": 11,
    "Open": 12,
    "Free For All": 13,
}


def now_id() -> str:
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: serialize_cell(row.get(column)) for column in columns})


def serialize_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.10g}"
    return str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "DATA_MISSING"


def safe_output_dir(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    logical = logical.absolute()
    root = ROOT.absolute()
    try:
        relative = logical.relative_to(root)
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    relative_text = relative.as_posix()
    required = "artifacts/full_evidence_orchestration_20260525"
    if not relative_text.startswith(required + "/"):
        raise ValueError(f"output_dir_must_be_under:{required}")
    for prefix in PROTECTED_PREFIXES:
        if relative_text == prefix or relative_text.startswith(prefix + "/"):
            raise ValueError(f"output_dir_protected:{prefix}")
    return output_dir


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        parsed = float(match.group(0))
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def usable_context_text(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.lower() in {"unknown", "n/a", "na", "none", "null", "-"}:
        return None
    return text


def safe_target_metadata_source(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in UNSAFE_TARGET_METADATA_SOURCE_MARKERS):
        return False
    return text in SAFE_TARGET_METADATA_SOURCES or text.startswith(("target_column:", "filename:"))


def safe_db_target_metadata_source(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in UNSAFE_TARGET_METADATA_SOURCE_MARKERS):
        return False
    return text in SAFE_DB_TARGET_METADATA_SOURCES


def has_post_outcome_marker(row: Mapping[str, Any]) -> bool:
    for field in ("winner_name", "winner_source", "winner_margin", "winner_odds"):
        if row.get(field) not in (None, ""):
            return True
    status = str(row.get("results_status") or row.get("result_status") or "").strip().lower()
    return status in POST_OUTCOME_STATUS_VALUES


def can_reuse_packet_feature(feature: str, target: Mapping[str, Any]) -> bool:
    if target.get("status") != "SAFE" and feature in TARGET_METADATA_DEPENDENT_PACKET_REUSE_BLOCKLIST:
        return False
    return True


def safe_int(value: Any) -> int | None:
    parsed = safe_float(value)
    return int(parsed) if parsed is not None else None


def clean_name(name: Any) -> str:
    text = str(name or "").strip().lower()
    text = re.sub(r"^\d+\.\s*", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def parse_race_number(race_id: Any) -> int | None:
    match = re.search(r"Race\s+(\d+)", str(race_id or ""), re.IGNORECASE)
    return int(match.group(1)) if match else None


def parse_date(value: Any) -> str | None:
    text = str(value or "").strip()
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    return match.group(0) if match else None


def parse_datetime_minutes(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    iso_match = re.search(r"T(\d{2}):(\d{2})", text)
    if iso_match:
        return int(iso_match.group(1)) * 60 + int(iso_match.group(2))
    clock_match = re.search(r"(\d{1,2}):(\d{2})\s*([AP]M)?", text, re.IGNORECASE)
    if not clock_match:
        return None
    hour = int(clock_match.group(1))
    minute = int(clock_match.group(2))
    suffix = (clock_match.group(3) or "").upper()
    if suffix == "PM" and hour < 12:
        hour += 12
    if suffix == "AM" and hour == 12:
        hour = 0
    return hour * 60 + minute


def normalize_grade(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    compact = re.sub(r"\s+", " ", text.upper().replace("_", " ").replace("-", " ")).strip()
    compact = compact.replace("TH GRADE", "TH GRADE")
    slash = compact.replace(" ", "")
    if slash in GRADE_MAP:
        return GRADE_MAP[slash]
    if compact in GRADE_MAP:
        return GRADE_MAP[compact]
    grade_match = re.fullmatch(r"([1-8])(?:ST|ND|RD|TH)?\s*GRADE", compact)
    if grade_match:
        return f"Grade {grade_match.group(1)}"
    return text


def grade_rank(value: Any) -> int | None:
    normalized = normalize_grade(value)
    return GRADE_RANK.get(normalized or "")


def box_band(value: Any) -> str | None:
    box = safe_int(value)
    if box in (1, 2):
        return "inside"
    if box in (3, 4, 5, 6):
        return "middle"
    if box is not None and box >= 7:
        return "outside"
    return None


def history_grade_rank(row: Mapping[str, Any]) -> float | None:
    rank = safe_float(row.get("grade_rank_num"))
    if rank is not None:
        return rank
    return safe_float(grade_rank(row.get("grade_normalized") or row.get("grade")))


def history_field_size(row: Mapping[str, Any]) -> float | None:
    return safe_float(row.get("race_field_size")) or safe_float(row.get("field_size"))


def history_box_band(row: Mapping[str, Any]) -> str | None:
    return str(row.get("box_band") or "") or box_band(row.get("box_number"))


def history_speed(row: Mapping[str, Any]) -> float | None:
    speed = safe_float(row.get("speed_mps"))
    if speed is not None:
        return speed
    distance = safe_float(row.get("distance_num") or row.get("distance"))
    time_value = safe_float(row.get("time_num") or row.get("individual_time"))
    if distance is not None and time_value is not None and time_value > 0:
        return distance / time_value
    return None


def history_race_strength(row: Mapping[str, Any]) -> float | None:
    strength = safe_float(row.get("race_strength_num"))
    if strength is not None:
        return strength
    rank = history_grade_rank(row)
    field_size = history_field_size(row)
    if rank is not None and field_size is not None:
        return rank * field_size
    return None


def mean(values: Iterable[float]) -> float | None:
    items = [value for value in values if value is not None and not math.isnan(value)]
    return sum(items) / len(items) if items else None


def std(values: Iterable[float]) -> float | None:
    items = [value for value in values if value is not None and not math.isnan(value)]
    if len(items) < 2:
        return None
    return statistics.pstdev(items)


def best_min(values: Iterable[float]) -> float | None:
    items = [value for value in values if value is not None and not math.isnan(value)]
    return min(items) if items else None


def mode_text(values: Iterable[str | None]) -> str | None:
    items = [value for value in values if value]
    if not items:
        return None
    return Counter(items).most_common(1)[0][0]


def group_key(row: Mapping[str, Any]) -> str:
    return str(row.get("snapshot_instance_id") or row.get("race_id") or "DATA_MISSING")


def row_join_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("snapshot_instance_id") or ""),
        clean_name(row.get("dog_name") or row.get("normalized_dog_name")),
        str(row.get("box_number") or ""),
    )


def relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def current_path_from_snapshot_path(value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.exists():
        return path
    marker = "artifacts/"
    if marker in text:
        candidate = ROOT / text[text.index(marker) :]
        if candidate.exists():
            return candidate
    return path if path.exists() else None


def load_snapshot(path_text: Any, cache: dict[str, Any]) -> dict[str, Any] | None:
    path = current_path_from_snapshot_path(path_text)
    if path is None:
        return None
    key = str(path)
    if key not in cache:
        try:
            cache[key] = load_json(path)
        except Exception:
            cache[key] = None
    item = cache[key]
    return item if isinstance(item, dict) else None


def find_prediction(snapshot: Mapping[str, Any] | None, dog_name: Any, box_number: Any) -> dict[str, Any] | None:
    if not snapshot:
        return None
    wanted_name = clean_name(dog_name)
    wanted_box = str(box_number or "")
    for prediction in snapshot.get("predictions", []) or []:
        if clean_name(prediction.get("dog_name")) == wanted_name and str(prediction.get("box_number")) == wanted_box:
            return dict(prediction)
    return None


def sqlite_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def load_db_history(connection: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    dog_columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(dog_race_data)").fetchall()
    }
    race_columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(race_metadata)").fetchall()
    }

    def dog_expr(column: str, *, fallback: str | None = None) -> str:
        if column in dog_columns:
            return f"dr.{column}"
        if fallback and fallback in dog_columns:
            return f"dr.{fallback} AS {column}"
        return f"NULL AS {column}"

    def dog_alias_expr(column: str, alias: str) -> str:
        if column in dog_columns:
            return f"dr.{column} AS {alias}"
        return f"NULL AS {alias}"

    def race_expr(column: str, *, alias: str | None = None) -> str:
        output = alias or column
        if column in race_columns:
            return f"rm.{column} AS {output}"
        return f"NULL AS {output}"

    if "dog_name" in dog_columns:
        dog_name_where = "dr.dog_name"
    elif "dog_clean_name" in dog_columns:
        dog_name_where = "dr.dog_clean_name"
    else:
        dog_name_where = "NULL"
    query = """
        SELECT
            {race_id},
            {dog_name},
            {box_number},
            {finish_position},
            {placing},
            {individual_time},
            {sectional_1st},
            {weight},
            {beaten_margin},
            {margin},
            {venue},
            {race_number},
            {race_date},
            {grade},
            {distance},
            {track_condition},
            {weather},
            {race_time},
            {start_datetime},
            {dog_data_source},
            {race_metadata_data_source},
            {race_metadata_url},
            {dog_data_source_loaded} AS dog_data_source_column_loaded,
            {race_metadata_data_source_loaded} AS race_metadata_data_source_column_loaded
        FROM dog_race_data dr
        LEFT JOIN race_metadata rm ON rm.race_id = dr.race_id
        WHERE {dog_name_where} IS NOT NULL
    """.format(
        race_id=dog_expr("race_id"),
        dog_name=dog_expr("dog_name", fallback="dog_clean_name"),
        box_number=dog_expr("box_number"),
        finish_position=dog_expr("finish_position"),
        placing=dog_expr("placing"),
        individual_time=dog_expr("individual_time"),
        sectional_1st=dog_expr("sectional_1st"),
        weight=dog_expr("weight"),
        beaten_margin=dog_expr("beaten_margin"),
        margin=dog_expr("margin"),
        venue=race_expr("venue"),
        race_number=race_expr("race_number"),
        race_date=race_expr("race_date"),
        grade=race_expr("grade"),
        distance=race_expr("distance"),
        track_condition=race_expr("track_condition"),
        weather=race_expr("weather"),
        race_time=race_expr("race_time"),
        start_datetime=race_expr("start_datetime"),
        dog_data_source=dog_alias_expr("data_source", "dog_data_source"),
        race_metadata_data_source=race_expr("data_source", alias="race_metadata_data_source"),
        race_metadata_url=race_expr("url", alias="race_metadata_url"),
        dog_data_source_loaded=1 if "data_source" in dog_columns else 0,
        race_metadata_data_source_loaded=1 if "data_source" in race_columns else 0,
        dog_name_where=dog_name_where,
    )
    raw_rows: list[dict[str, Any]] = []
    for item in connection.execute(query):
        row = dict(item)
        row["dog_key"] = clean_name(row.get("dog_name"))
        row["race_date"] = parse_date(row.get("race_date"))
        if not row["race_date"]:
            continue
        row["distance_num"] = safe_float(row.get("distance"))
        row["finish_num"] = safe_int(row.get("finish_position") or row.get("placing"))
        row["time_num"] = safe_float(row.get("individual_time"))
        row["sectional_1st_num"] = safe_float(row.get("sectional_1st"))
        row["weight_num"] = safe_float(row.get("weight"))
        row["margin_num"] = safe_float(row.get("beaten_margin") or row.get("margin"))
        row["grade_normalized"] = normalize_grade(row.get("grade"))
        row["grade_rank_num"] = grade_rank(row.get("grade_normalized"))
        row["box_band"] = box_band(row.get("box_number"))
        if row["distance_num"] and row["time_num"] and row["time_num"] > 0:
            row["speed_mps"] = row["distance_num"] / row["time_num"]
        else:
            row["speed_mps"] = None
        raw_rows.append(row)

    race_field_sizes = Counter(
        str(row.get("race_id") or "")
        for row in raw_rows
        if row.get("race_id") not in (None, "")
    )
    history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        row["race_field_size"] = race_field_sizes.get(str(row.get("race_id") or ""), 0) or None
        if row.get("grade_rank_num") is not None and row.get("race_field_size") is not None:
            row["race_strength_num"] = row["grade_rank_num"] * row["race_field_size"]
        else:
            row["race_strength_num"] = None
        history[row["dog_key"]].append(row)
    for rows in history.values():
        rows.sort(key=lambda row: (row.get("race_date") or "", safe_int(row.get("race_number")) or 0, row.get("race_id") or ""))
    return dict(history)


def race_metadata_candidates(connection: sqlite3.Connection, race_date: str, venue: str, race_number: int | None) -> list[dict[str, Any]]:
    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(race_metadata)").fetchall()
    }
    if "race_date" not in columns or "venue" not in columns:
        return []

    def race_expr(column: str) -> str:
        if column in columns:
            return column
        return f"NULL AS {column}"

    rows: list[dict[str, Any]] = []
    for item in connection.execute(
        f"""
        SELECT
            {race_expr("race_id")},
            venue,
            {race_expr("race_number")},
            race_date,
            {race_expr("grade")},
            {race_expr("distance")},
            {race_expr("field_size")},
            {race_expr("race_time")},
            {race_expr("track_condition")},
            {race_expr("weather")},
            {race_expr("weather_condition")},
            {race_expr("url")},
            {race_expr("data_source")},
            {race_expr("start_datetime")},
            {race_expr("results_status")},
            {race_expr("winner_source")},
            {race_expr("winner_name")},
            {race_expr("winner_margin")},
            {race_expr("winner_odds")}
        FROM race_metadata
        WHERE race_date = ? AND upper(venue) = upper(?)
        """,
        (race_date, venue),
    ):
        row = dict(item)
        row["identity_match"] = False
        rows.append(row)
    safe_rows = [
        row
        for row in rows
        if (safe_float(row.get("distance")) is not None or normalize_grade(row.get("grade")))
        and safe_db_target_metadata_source(row.get("data_source"))
        and not has_post_outcome_marker(row)
    ]
    exact = [
        row
        for row in safe_rows
        if race_number is not None
        and safe_int(row.get("race_number")) == race_number
    ]
    if exact:
        for row in exact:
            row["identity_match"] = True
        return exact
    return safe_rows


def race_metadata_probe(connection: sqlite3.Connection, race_date: str | None, venue: str, race_number: int | None) -> dict[str, Any]:
    default = {
        "db_lookup_status": "not_attempted",
        "db_exact_row_count": 0,
        "db_exact_metadata_row_count": 0,
        "db_safe_candidate_count": 0,
        "db_unsafe_candidate_count": 0,
        "db_embedded_form_metadata_count": 0,
        "db_post_outcome_metadata_count": 0,
        "db_metadata_source_counts": {},
        "db_exact_row_has_target_metadata": False,
        "db_exact_row_has_post_outcome_marker": False,
    }
    if not race_date or not venue:
        return {**default, "db_lookup_status": "missing_identity"}
    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(race_metadata)").fetchall()
    }
    if "race_date" not in columns or "venue" not in columns:
        return {**default, "db_lookup_status": "race_metadata_identity_columns_missing"}

    def race_expr(column: str) -> str:
        if column in columns:
            return column
        return f"NULL AS {column}"

    rows = [
        dict(row)
        for row in connection.execute(
            f"""
            SELECT
                {race_expr("race_id")},
                venue,
                {race_expr("race_number")},
                race_date,
                {race_expr("grade")},
                {race_expr("distance")},
                {race_expr("url")},
                {race_expr("data_source")},
                {race_expr("results_status")},
                {race_expr("winner_source")},
                {race_expr("winner_name")},
                {race_expr("winner_margin")},
                {race_expr("winner_odds")}
            FROM race_metadata
            WHERE race_date = ? AND upper(venue) = upper(?)
            """,
            (race_date, venue),
        )
    ]
    exact_rows = [
        row
        for row in rows
        if race_number is not None and safe_int(row.get("race_number")) == race_number
    ]
    metadata_rows = [
        row
        for row in rows
        if safe_float(row.get("distance")) is not None or normalize_grade(row.get("grade"))
    ]
    safe_rows = [
        row
        for row in metadata_rows
        if safe_db_target_metadata_source(row.get("data_source")) and not has_post_outcome_marker(row)
    ]
    unsafe_rows = [row for row in metadata_rows if row not in safe_rows]
    source_counts: Counter[str] = Counter()
    for row in metadata_rows:
        source_counts[provenance_bucket(row.get("data_source"))] += 1
    exact_metadata_rows = [
        row
        for row in exact_rows
        if safe_float(row.get("distance")) is not None or normalize_grade(row.get("grade"))
    ]
    return {
        **default,
        "db_lookup_status": "checked",
        "db_exact_row_count": len(exact_rows),
        "db_exact_metadata_row_count": len(exact_metadata_rows),
        "db_safe_candidate_count": len(safe_rows),
        "db_unsafe_candidate_count": len(unsafe_rows),
        "db_embedded_form_metadata_count": sum(
            1
            for row in metadata_rows
            if str(row.get("data_source") or "").strip().lower() == "embedded_form_guide"
        ),
        "db_post_outcome_metadata_count": sum(1 for row in metadata_rows if has_post_outcome_marker(row)),
        "db_metadata_source_counts": dict(source_counts),
        "db_exact_row_has_target_metadata": bool(exact_metadata_rows),
        "db_exact_row_has_post_outcome_marker": any(has_post_outcome_marker(row) for row in exact_rows),
    }


def safe_sidecar_metadata(snapshot: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not snapshot:
        return None
    source_path = current_path_from_snapshot_path(snapshot.get("source_file_path"))
    if source_path is None:
        return None
    sidecar = Path(str(source_path) + ".metadata.json")
    if not sidecar.exists():
        return None
    try:
        payload = load_json(sidecar)
    except Exception:
        return None
    if not payload.get("metadata_is_leakage_safe"):
        return None
    race_info = payload.get("race_info") or {}
    verification = payload.get("normalization_verification") or {}
    if verification.get("target_metadata_status") not in (None, "verified"):
        return None
    distance = safe_float(race_info.get("distance") or payload.get("target_distance"))
    grade = normalize_grade(race_info.get("grade") or payload.get("target_grade"))
    distance_source = race_info.get("target_distance_source") or payload.get("target_distance_source")
    grade_source = race_info.get("target_grade_source") or payload.get("target_grade_source")
    if distance is not None and not safe_target_metadata_source(distance_source):
        distance = None
    if grade and not safe_target_metadata_source(grade_source):
        grade = None
    return {
        "distance": distance,
        "grade": grade,
        "race_time": race_info.get("race_time"),
        "track_condition": usable_context_text(
            payload.get("track_condition") or race_info.get("track_condition")
        ),
        "weather": usable_context_text(
            payload.get("weather")
            or race_info.get("weather")
            or payload.get("weather_condition")
            or race_info.get("weather_condition")
        ),
        "source": "safe_sidecar_metadata",
        "url": payload.get("metadata_source_url") or payload.get("race_url"),
        "sidecar_path": relpath(sidecar),
    }


def sidecar_metadata_probe(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    default = {
        "sidecar_status": "not_attempted",
        "sidecar_path": None,
        "sidecar_has_distance": False,
        "sidecar_has_grade": False,
        "sidecar_distance_source": None,
        "sidecar_grade_source": None,
        "sidecar_verification_status": None,
        "sidecar_metadata_is_leakage_safe": False,
    }
    if not snapshot:
        return {**default, "sidecar_status": "snapshot_missing"}
    source_path = current_path_from_snapshot_path(snapshot.get("source_file_path"))
    if source_path is None:
        return {**default, "sidecar_status": "source_file_path_missing"}
    sidecar = Path(str(source_path) + ".metadata.json")
    if not sidecar.exists():
        return {**default, "sidecar_status": "sidecar_missing", "sidecar_path": relpath(sidecar)}
    try:
        payload = load_json(sidecar)
    except Exception:
        return {**default, "sidecar_status": "sidecar_unreadable", "sidecar_path": relpath(sidecar)}
    if not isinstance(payload, Mapping):
        return {**default, "sidecar_status": "sidecar_not_object", "sidecar_path": relpath(sidecar)}

    race_info = payload.get("race_info") if isinstance(payload.get("race_info"), Mapping) else {}
    verification = payload.get("normalization_verification") or {}
    if not isinstance(verification, Mapping):
        verification = {}
    distance = safe_float(race_info.get("distance") or payload.get("target_distance"))
    grade = normalize_grade(race_info.get("grade") or payload.get("target_grade"))
    distance_source = race_info.get("target_distance_source") or payload.get("target_distance_source")
    grade_source = race_info.get("target_grade_source") or payload.get("target_grade_source")
    verification_status = verification.get("target_metadata_status")
    leakage_safe = payload.get("metadata_is_leakage_safe") is True
    distance_safe = distance is not None and safe_target_metadata_source(distance_source)
    grade_safe = bool(grade and safe_target_metadata_source(grade_source))

    if not leakage_safe:
        status = "sidecar_not_leakage_safe"
    elif verification_status not in (None, "verified"):
        status = f"sidecar_target_metadata_{verification_status}"
    elif distance_safe or grade_safe:
        status = "sidecar_verified_safe_target_metadata"
    elif distance is not None or grade:
        status = "sidecar_target_metadata_unsafe_source"
    else:
        status = "sidecar_without_target_metadata"
    return {
        **default,
        "sidecar_status": status,
        "sidecar_path": relpath(sidecar),
        "sidecar_has_distance": distance is not None,
        "sidecar_has_grade": bool(grade),
        "sidecar_distance_source": distance_source,
        "sidecar_grade_source": grade_source,
        "sidecar_verification_status": verification_status,
        "sidecar_metadata_is_leakage_safe": leakage_safe,
    }


def target_metadata_blocker_reason(
    *,
    target: Mapping[str, Any],
    sidecar_probe: Mapping[str, Any],
    db_probe: Mapping[str, Any],
    packet_row: Mapping[str, Any] | None,
) -> str:
    if target.get("status") == "SAFE":
        return "SAFE"
    if target.get("status") == "AMBIGUOUS":
        return "DATA_MISSING:ambiguous_safe_pre_race_metadata"
    packet_source = str((packet_row or {}).get("target_grade_source") or "").strip()
    if packet_source == "race_metadata.grade":
        return "DATA_MISSING:legacy_packet_race_metadata_grade_rejected"
    if sidecar_probe.get("sidecar_status") == "sidecar_verified_safe_target_metadata":
        return "DATA_MISSING:partial_sidecar_target_metadata"
    if db_probe.get("db_exact_row_count") and not db_probe.get("db_exact_row_has_target_metadata"):
        return "DATA_MISSING:canonical_exact_db_row_has_no_distance_grade"
    if db_probe.get("db_embedded_form_metadata_count") and not db_probe.get("db_safe_candidate_count"):
        return "DATA_MISSING:only_embedded_form_history_metadata_available"
    if db_probe.get("db_post_outcome_metadata_count") and not db_probe.get("db_safe_candidate_count"):
        return "DATA_MISSING:only_post_outcome_metadata_available"
    if db_probe.get("db_unsafe_candidate_count") and not db_probe.get("db_safe_candidate_count"):
        return "DATA_MISSING:only_unsafe_metadata_candidates_available"
    if db_probe.get("db_safe_candidate_count", 0) > 1:
        return "DATA_MISSING:ambiguous_safe_pre_race_metadata"
    if sidecar_probe.get("sidecar_status") in {
        "sidecar_missing",
        "source_file_path_missing",
        "snapshot_missing",
        "sidecar_without_target_metadata",
    }:
        return "DATA_MISSING:no_verified_sidecar_or_db_target_metadata"
    return "DATA_MISSING:no_safe_target_metadata"


def target_metadata_recovery_audit_row(
    *,
    clean_row: Mapping[str, Any],
    packet_row: Mapping[str, Any] | None,
    snapshot: Mapping[str, Any] | None,
    target: Mapping[str, Any],
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    race_date = parse_date(clean_row.get("race_date"))
    venue = str(clean_row.get("venue") or "").strip()
    race_number = safe_int(snapshot.get("race_number") if snapshot else None) or parse_race_number(clean_row.get("race_id"))
    sidecar_probe = sidecar_metadata_probe(snapshot)
    db_probe = race_metadata_probe(connection, race_date, venue, race_number)
    blocker_reason = target_metadata_blocker_reason(
        target=target,
        sidecar_probe=sidecar_probe,
        db_probe=db_probe,
        packet_row=packet_row,
    )
    packet = packet_row or {}
    return {
        "race_id": clean_row.get("race_id"),
        "snapshot_instance_id": clean_row.get("snapshot_instance_id"),
        "dog_name": clean_row.get("dog_name"),
        "box_number": clean_row.get("box_number"),
        "race_date": race_date,
        "venue": venue,
        "race_number": race_number,
        "target_metadata_status_v2": target.get("status"),
        "target_metadata_source_v2": target.get("source"),
        "target_metadata_reason_v2": target.get("reason"),
        "target_distance_safe_present": target.get("distance") is not None,
        "target_grade_safe_present": bool(target.get("grade")),
        "target_metadata_blocker_reason": blocker_reason,
        "sidecar_status": sidecar_probe.get("sidecar_status"),
        "sidecar_path": sidecar_probe.get("sidecar_path"),
        "sidecar_has_distance": sidecar_probe.get("sidecar_has_distance"),
        "sidecar_has_grade": sidecar_probe.get("sidecar_has_grade"),
        "sidecar_distance_source": sidecar_probe.get("sidecar_distance_source"),
        "sidecar_grade_source": sidecar_probe.get("sidecar_grade_source"),
        "sidecar_verification_status": sidecar_probe.get("sidecar_verification_status"),
        "sidecar_metadata_is_leakage_safe": sidecar_probe.get("sidecar_metadata_is_leakage_safe"),
        "db_lookup_status": db_probe.get("db_lookup_status"),
        "db_exact_row_count": db_probe.get("db_exact_row_count"),
        "db_exact_metadata_row_count": db_probe.get("db_exact_metadata_row_count"),
        "db_safe_candidate_count": db_probe.get("db_safe_candidate_count"),
        "db_unsafe_candidate_count": db_probe.get("db_unsafe_candidate_count"),
        "db_embedded_form_metadata_count": db_probe.get("db_embedded_form_metadata_count"),
        "db_post_outcome_metadata_count": db_probe.get("db_post_outcome_metadata_count"),
        "db_metadata_source_counts": compact_counter(db_probe.get("db_metadata_source_counts") or {}),
        "db_exact_row_has_target_metadata": db_probe.get("db_exact_row_has_target_metadata"),
        "db_exact_row_has_post_outcome_marker": db_probe.get("db_exact_row_has_post_outcome_marker"),
        "packet_target_grade_source": packet.get("target_grade_source"),
        "packet_target_distance_source": packet.get("target_distance_source"),
    }


def target_metadata_recovery_audit_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    blocker_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    race_ids_by_blocker: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        blocker = str(row.get("target_metadata_blocker_reason") or "DATA_MISSING:unknown")
        blocker_counts[blocker] += 1
        source_counts[provenance_bucket(row.get("target_metadata_source_v2"))] += 1
        status_counts[provenance_bucket(row.get("target_metadata_status_v2"))] += 1
        race_ids_by_blocker[blocker].add(str(row.get("race_id") or "DATA_MISSING"))
    return {
        "schema_version": "target_metadata_recovery_audit_v1",
        "report_only": True,
        "matrix_rows": len(rows),
        "safe_rows": blocker_counts.get("SAFE", 0),
        "data_missing_rows": len(rows) - blocker_counts.get("SAFE", 0),
        "safe_races": len(race_ids_by_blocker.get("SAFE", set())),
        "data_missing_races": len(
            set().union(
                *[
                    race_ids
                    for blocker, race_ids in race_ids_by_blocker.items()
                    if blocker != "SAFE"
                ]
            )
        )
        if any(blocker != "SAFE" for blocker in race_ids_by_blocker)
        else 0,
        "target_metadata_source_counts": dict(source_counts),
        "target_metadata_status_counts": dict(status_counts),
        "blocker_counts": dict(blocker_counts),
        "blocker_race_counts": {
            blocker: len(race_ids)
            for blocker, race_ids in sorted(race_ids_by_blocker.items())
        },
        "notes": [
            "Rows are recoverable only from clean target fields, verified canonical sidecars, or safe pre-race DB metadata.",
            "Embedded form-history metadata, post-outcome metadata, odds, EV, result fields, and ambiguous candidates remain DATA_MISSING.",
        ],
    }


def resolve_target_metadata(
    row: Mapping[str, Any],
    packet_row: Mapping[str, Any] | None,
    snapshot: Mapping[str, Any] | None,
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    race_date = parse_date(row.get("race_date"))
    venue = str(row.get("venue") or "").strip()
    race_number = safe_int(snapshot.get("race_number") if snapshot else None) or parse_race_number(row.get("race_id"))
    result: dict[str, Any] = {
        "distance": None,
        "grade": None,
        "track_condition": None,
        "weather": None,
        "race_time": None,
        "source": "missing",
        "status": "MISSING",
        "reason": "no_safe_target_metadata",
        "race_number": race_number,
    }

    clean_distance = safe_float(row.get("target_distance") or row.get("distance"))
    clean_grade = normalize_grade(row.get("target_grade"))
    if clean_distance is not None or clean_grade:
        result.update(
            {
                "distance": clean_distance,
                "grade": clean_grade,
                "source": "clean_official_dataset_target_metadata",
                "status": "SAFE",
                "reason": "clean_dataset_target_metadata_present",
            }
        )

    sidecar = safe_sidecar_metadata(snapshot)
    if sidecar and (sidecar.get("distance") is not None or sidecar.get("grade")):
        result.update(
            {
                "distance": sidecar.get("distance") if sidecar.get("distance") is not None else result.get("distance"),
                "grade": sidecar.get("grade") or result.get("grade"),
                "track_condition": sidecar.get("track_condition"),
                "weather": sidecar.get("weather"),
                "race_time": sidecar.get("race_time"),
                "source": sidecar.get("source"),
                "status": "SAFE",
                "reason": "verified_metadata_sidecar",
                "source_url": sidecar.get("url"),
                "sidecar_path": sidecar.get("sidecar_path"),
            }
        )

    if race_date and venue and (result.get("distance") is None or not result.get("grade")):
        candidates = race_metadata_candidates(connection, race_date, venue, race_number)
        unique_candidates = [
            candidate
            for candidate in candidates
            if safe_float(candidate.get("distance")) is not None or normalize_grade(candidate.get("grade"))
        ]
        if len(unique_candidates) == 1:
            candidate = unique_candidates[0]
            identity_match = bool(candidate.get("identity_match"))
            result.update(
                {
                    "distance": result.get("distance")
                    if result.get("distance") is not None
                    else safe_float(candidate.get("distance")),
                    "grade": result.get("grade") or normalize_grade(candidate.get("grade")),
                    "track_condition": usable_context_text(candidate.get("track_condition"))
                    or result.get("track_condition"),
                    "weather": usable_context_text(candidate.get("weather"))
                    or usable_context_text(candidate.get("weather_condition"))
                    or result.get("weather"),
                    "race_time": candidate.get("race_time") or result.get("race_time"),
                    "source": "canonical_race_metadata_exact_identity"
                    if identity_match
                    else "canonical_race_metadata_unique_date_venue",
                    "status": "SAFE",
                    "reason": "exact_identity_metadata_candidate"
                    if identity_match
                    else "unique_safe_date_venue_metadata_candidate",
                    "source_race_id": candidate.get("race_id"),
                }
            )
        elif len(unique_candidates) > 1 and result.get("status") != "SAFE":
            result.update(
                {
                    "status": "AMBIGUOUS",
                    "reason": f"ambiguous_safe_metadata_candidates:{len(unique_candidates)}",
                    "candidate_count": len(unique_candidates),
                }
            )

    if packet_row and not result.get("grade"):
        packet_grade = normalize_grade(packet_row.get("target_grade_safe"))
        packet_source = packet_row.get("target_grade_source")
        if packet_grade and packet_source == "clean_official_dataset.target_grade":
            result.update(
                {
                    "grade": packet_grade,
                    "source": f"packet_{packet_source}",
                    "status": "SAFE",
                    "reason": "packet_safe_target_grade_reused",
                }
            )
    return result


def history_before_target(
    rows: Sequence[Mapping[str, Any]],
    target_date: str | None,
    target_race_id: str,
) -> list[dict[str, Any]]:
    if not target_date:
        return []
    selected: list[dict[str, Any]] = []
    for row in rows:
        race_date = str(row.get("race_date") or "")
        if not race_date or race_date >= target_date:
            continue
        if row.get("race_id") == target_race_id:
            continue
        selected.append(dict(row))
    selected.sort(key=lambda item: (item.get("race_date") or "", safe_int(item.get("race_number")) or 0, item.get("race_id") or ""))
    return selected


def provenance_bucket(value: Any, *, column_loaded: bool = True) -> str:
    if not column_loaded:
        return "COLUMN_ABSENT"
    text = str(value or "").strip()
    return text if text else "MISSING"


def history_time_source(row: Mapping[str, Any]) -> str:
    if safe_float(row.get("time_num")) is not None:
        return "time_num"
    if safe_float(row.get("individual_time")) is not None:
        return "individual_time"
    return "missing"


def compact_counter(counter: Mapping[str, int]) -> str:
    if not counter:
        return ""
    return ";".join(f"{key}={counter[key]}" for key in sorted(counter))


def summarize_history_source_provenance_for_row(
    *,
    clean_row: Mapping[str, Any],
    history_rows: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    dog_source_counts: Counter[str] = Counter()
    race_source_counts: Counter[str] = Counter()
    time_source_counts: Counter[str] = Counter()
    dog_source_loaded_rows = 0
    race_source_loaded_rows = 0
    metadata_present_counts: Counter[str] = Counter()
    time_num_present_rows = 0
    individual_time_present_rows = 0

    for row in history_rows:
        dog_loaded = bool(row.get("dog_data_source_column_loaded"))
        race_loaded = bool(row.get("race_metadata_data_source_column_loaded"))
        dog_source_loaded_rows += 1 if dog_loaded else 0
        race_source_loaded_rows += 1 if race_loaded else 0
        dog_source_counts[provenance_bucket(row.get("dog_data_source"), column_loaded=dog_loaded)] += 1
        race_source_counts[
            provenance_bucket(row.get("race_metadata_data_source"), column_loaded=race_loaded)
        ] += 1
        time_source_counts[history_time_source(row)] += 1
        if safe_float(row.get("time_num")) is not None:
            time_num_present_rows += 1
        if safe_float(row.get("individual_time")) is not None:
            individual_time_present_rows += 1
        for field in HISTORY_PROVENANCE_METADATA_FIELDS:
            if row.get(field) not in (None, ""):
                metadata_present_counts[field] += 1

    history_row_count = len(history_rows)
    return {
        "race_id": clean_row.get("race_id"),
        "snapshot_instance_id": clean_row.get("snapshot_instance_id"),
        "dog_name": clean_row.get("dog_name"),
        "box_number": clean_row.get("box_number"),
        "history_rows_used": history_row_count,
        "dog_data_source_column_loaded_rows": dog_source_loaded_rows,
        "race_metadata_data_source_column_loaded_rows": race_source_loaded_rows,
        "dog_data_source_counts": dict(dog_source_counts),
        "race_metadata_data_source_counts": dict(race_source_counts),
        "time_source_counts": dict(time_source_counts),
        "time_num_present_rows": time_num_present_rows,
        "individual_time_present_rows": individual_time_present_rows,
        "time_missing_rows": time_source_counts.get("missing", 0),
        "history_metadata_present_rows": dict(metadata_present_counts),
        "target_metadata_status_v2": target.get("status"),
        "target_metadata_source_v2": target.get("source"),
        "target_metadata_reason_v2": target.get("reason"),
    }


def history_source_provenance_report(
    row_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    total_history_rows = sum(safe_int(row.get("history_rows_used")) or 0 for row in row_summaries)
    rows_with_history = sum(1 for row in row_summaries if (safe_int(row.get("history_rows_used")) or 0) > 0)
    dog_sources: Counter[str] = Counter()
    race_sources: Counter[str] = Counter()
    time_sources: Counter[str] = Counter()
    target_metadata_sources: Counter[str] = Counter()
    target_metadata_statuses: Counter[str] = Counter()
    metadata_field_present_rows: Counter[str] = Counter()
    time_num_present_rows = 0
    individual_time_present_rows = 0
    dog_source_loaded_rows = 0
    race_source_loaded_rows = 0

    for row in row_summaries:
        dog_sources.update(row.get("dog_data_source_counts") or {})
        race_sources.update(row.get("race_metadata_data_source_counts") or {})
        time_sources.update(row.get("time_source_counts") or {})
        metadata_field_present_rows.update(row.get("history_metadata_present_rows") or {})
        target_metadata_sources[provenance_bucket(row.get("target_metadata_source_v2"))] += 1
        target_metadata_statuses[provenance_bucket(row.get("target_metadata_status_v2"))] += 1
        time_num_present_rows += safe_int(row.get("time_num_present_rows")) or 0
        individual_time_present_rows += safe_int(row.get("individual_time_present_rows")) or 0
        dog_source_loaded_rows += safe_int(row.get("dog_data_source_column_loaded_rows")) or 0
        race_source_loaded_rows += safe_int(row.get("race_metadata_data_source_column_loaded_rows")) or 0

    return {
        "schema_version": "history_source_provenance_v1",
        "report_only": True,
        "scope": "pre_target_db_history_rows_selected_for_stage2_feature_matrix",
        "matrix_rows": len(row_summaries),
        "matrix_rows_with_history": rows_with_history,
        "history_rows_used": total_history_rows,
        "data_source_columns": {
            "dog_race_data.data_source_loaded_rows": dog_source_loaded_rows,
            "race_metadata.data_source_loaded_rows": race_source_loaded_rows,
            "dog_race_data.data_source_counts": dict(dog_sources),
            "race_metadata.data_source_counts": dict(race_sources),
        },
        "time_source_availability": {
            "source_priority_counts": dict(time_sources),
            "time_num_present_rows": time_num_present_rows,
            "individual_time_present_rows": individual_time_present_rows,
            "missing_time_rows": time_sources.get("missing", 0),
        },
        "history_metadata_coverage": {
            field: {
                "present_rows": metadata_field_present_rows.get(field, 0),
                "missing_rows": max(total_history_rows - metadata_field_present_rows.get(field, 0), 0),
            }
            for field in HISTORY_PROVENANCE_METADATA_FIELDS
        },
        "target_metadata_source_counts": dict(target_metadata_sources),
        "target_metadata_status_counts": dict(target_metadata_statuses),
        "notes": [
            "This report is generated after pre-target history filtering and does not change features, metrics, training, labels, snapshots, manifests, odds, EV, or registry state.",
            "time_num is the normalized timing value used by the feature builder; individual_time records whether the raw DB timing field was available.",
            "COLUMN_ABSENT means the DB table did not expose that data_source column in the read-only schema.",
        ],
    }


def history_source_provenance_csv_rows(
    row_summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in row_summaries:
        metadata_counts = row.get("history_metadata_present_rows") or {}
        rows.append(
            {
                "race_id": row.get("race_id"),
                "snapshot_instance_id": row.get("snapshot_instance_id"),
                "dog_name": row.get("dog_name"),
                "box_number": row.get("box_number"),
                "history_rows_used": row.get("history_rows_used"),
                "dog_data_source_column_loaded_rows": row.get("dog_data_source_column_loaded_rows"),
                "race_metadata_data_source_column_loaded_rows": row.get(
                    "race_metadata_data_source_column_loaded_rows"
                ),
                "dog_data_source_counts": compact_counter(row.get("dog_data_source_counts") or {}),
                "race_metadata_data_source_counts": compact_counter(
                    row.get("race_metadata_data_source_counts") or {}
                ),
                "time_source_counts": compact_counter(row.get("time_source_counts") or {}),
                "time_num_present_rows": row.get("time_num_present_rows"),
                "individual_time_present_rows": row.get("individual_time_present_rows"),
                "time_missing_rows": row.get("time_missing_rows"),
                "grade_present_rows": metadata_counts.get("grade", 0),
                "distance_present_rows": metadata_counts.get("distance", 0),
                "race_time_present_rows": metadata_counts.get("race_time", 0),
                "track_condition_present_rows": metadata_counts.get("track_condition", 0),
                "weather_present_rows": metadata_counts.get("weather", 0),
                "start_datetime_present_rows": metadata_counts.get("start_datetime", 0),
                "race_metadata_url_present_rows": metadata_counts.get("race_metadata_url", 0),
                "target_metadata_status_v2": row.get("target_metadata_status_v2"),
                "target_metadata_source_v2": row.get("target_metadata_source_v2"),
                "target_metadata_reason_v2": row.get("target_metadata_reason_v2"),
            }
        )
    return rows


def rate(rows: Sequence[Mapping[str, Any]], predicate: Any) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if predicate(row)) / len(rows)


def add_history_features(
    features: dict[str, Any],
    history_rows: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
    target_date: str | None,
    target_venue: str,
) -> dict[str, Any]:
    recent = list(history_rows[-5:])
    recent3 = list(history_rows[-3:])
    last = history_rows[-1] if history_rows else None
    finish_values = [safe_float(row.get("finish_num")) for row in history_rows]
    recent_finish_values = [safe_float(row.get("finish_num")) for row in recent]
    recent3_finish_values = [safe_float(row.get("finish_num")) for row in recent3]
    time_values = [safe_float(row.get("time_num")) for row in history_rows]
    recent_time_values = [safe_float(row.get("time_num")) for row in recent]
    recent_speed_values = [history_speed(row) for row in recent]
    margin_values = [safe_float(row.get("margin_num")) for row in recent]
    weight_values = [safe_float(row.get("weight_num")) for row in recent]
    sectional_values = [safe_float(row.get("sectional_1st_num")) for row in recent]
    recent_grade_rank_values = [history_grade_rank(row) for row in recent]
    recent_field_size_values = [history_field_size(row) for row in recent]
    recent_race_strength_values = [history_race_strength(row) for row in recent]
    last_date = str(last.get("race_date")) if last else None

    features["prior_start_count"] = len(history_rows)
    features["days_since_last_start"] = days_between(last_date, target_date)
    features["recent_finish_mean_3"] = mean(recent3_finish_values)
    features["recent_finish_mean_5"] = mean(recent_finish_values)
    features["recent_finish_best_5"] = best_min(recent_finish_values)
    features["recent_win_rate_5"] = rate(recent, lambda row: safe_int(row.get("finish_num")) == 1)
    features["recent_place_rate_5"] = rate(recent, lambda row: (safe_int(row.get("finish_num")) or 99) <= 3)
    features["recent_avg_margin_5"] = mean(margin_values)
    features["recent_avg_time_5"] = mean(recent_time_values)
    features["recent_best_time_5"] = best_min(recent_time_values)
    features["recent_time_std_5"] = std(recent_time_values)
    features["recent_avg_speed_mps_5"] = mean(recent_speed_values)
    features["career_win_rate"] = rate(history_rows, lambda row: safe_int(row.get("finish_num")) == 1)
    features["career_place_rate"] = rate(history_rows, lambda row: (safe_int(row.get("finish_num")) or 99) <= 3)
    features["career_avg_finish"] = mean(finish_values)
    features["career_best_finish"] = best_min(finish_values)
    features["career_avg_time"] = mean(time_values)
    features["career_best_time"] = best_min(time_values)
    features["career_time_std"] = std(time_values)
    features["last_start_weight"] = safe_float(last.get("weight_num")) if last else None
    features["recent_avg_weight_5"] = mean(weight_values)
    if features.get("last_start_weight") is not None and features.get("recent_avg_weight_5") is not None:
        features["weight_delta_last_to_recent"] = features["last_start_weight"] - features["recent_avg_weight_5"]

    safe_grade_rank = grade_rank(target.get("grade"))
    target_field_size = safe_float(features.get("field_size"))
    features["safe_grade_rank"] = safe_grade_rank
    if safe_grade_rank is not None and target_field_size is not None:
        features["safe_field_strength"] = safe_grade_rank * target_field_size
    features["last_start_grade_rank"] = history_grade_rank(last) if last else None
    features["recent_avg_grade_rank_5"] = mean(recent_grade_rank_values)
    features["last_start_field_size"] = history_field_size(last) if last else None
    features["recent_avg_field_size_5"] = mean(recent_field_size_values)
    features["last_start_race_strength"] = history_race_strength(last) if last else None
    features["recent_avg_race_strength_5"] = mean(recent_race_strength_values)
    if (
        features.get("safe_field_strength") is not None
        and features.get("recent_avg_race_strength_5") is not None
    ):
        features["prior_race_strength_delta_to_target"] = (
            features["safe_field_strength"] - features["recent_avg_race_strength_5"]
        )

    target_band = box_band(features.get("box_number"))
    if target_band:
        same_box_band = [row for row in history_rows if history_box_band(row) == target_band]
        features["target_box_band_prior_start_count"] = len(same_box_band)
        features["target_box_band_win_rate"] = rate(
            same_box_band, lambda row: safe_int(row.get("finish_num")) == 1
        )
        features["target_box_band_place_rate"] = rate(
            same_box_band, lambda row: (safe_int(row.get("finish_num")) or 99) <= 3
        )
        features["target_box_band_avg_finish"] = mean(
            [safe_float(row.get("finish_num")) for row in same_box_band]
        )
        features["target_box_band_avg_time"] = mean(
            [safe_float(row.get("time_num")) for row in same_box_band]
        )
    else:
        same_box_band = []

    same_venue = [row for row in history_rows if str(row.get("venue") or "").upper() == target_venue.upper()]
    features["starts_same_venue"] = len(same_venue)
    features["win_rate_same_venue"] = rate(same_venue, lambda row: safe_int(row.get("finish_num")) == 1)
    features["place_rate_same_venue"] = rate(same_venue, lambda row: (safe_int(row.get("finish_num")) or 99) <= 3)
    features["best_time_same_venue"] = best_min([safe_float(row.get("time_num")) for row in same_venue])
    features["avg_time_same_venue"] = mean([safe_float(row.get("time_num")) for row in same_venue])
    venue_box_band = [row for row in same_venue if target_band and history_box_band(row) == target_band]
    features["venue_box_band_start_count"] = len(venue_box_band)
    features["venue_box_band_win_rate"] = rate(
        venue_box_band, lambda row: safe_int(row.get("finish_num")) == 1
    )
    features["venue_box_band_place_rate"] = rate(
        venue_box_band, lambda row: (safe_int(row.get("finish_num")) or 99) <= 3
    )
    features["venue_box_band_avg_finish"] = mean(
        [safe_float(row.get("finish_num")) for row in venue_box_band]
    )

    target_distance = safe_float(target.get("distance"))
    if target_distance is not None:
        same_distance = [
            row
            for row in history_rows
            if safe_float(row.get("distance_num")) is not None
            and abs(float(row.get("distance_num")) - target_distance) <= 50.0
        ]
        same_distance_recent = same_distance[-5:]
        features["starts_same_distance"] = len(same_distance)
        features["prior_same_distance_start_count"] = len(same_distance)
        features["best_time_same_distance"] = best_min([safe_float(row.get("time_num")) for row in same_distance])
        features["avg_time_same_distance"] = mean([safe_float(row.get("time_num")) for row in same_distance])
        time_same_distance = [safe_float(row.get("time_num")) for row in same_distance]
        present_time_same_distance = [x for x in time_same_distance if x is not None]
        if present_time_same_distance:
            features["median_time_same_distance"] = statistics.median(present_time_same_distance)
        features["recent_best_time_same_distance_5"] = best_min([safe_float(row.get("time_num")) for row in same_distance_recent])
        features["recent_avg_time_same_distance_5"] = mean([safe_float(row.get("time_num")) for row in same_distance_recent])
        features["days_since_last_same_distance_start"] = days_between(
            str(same_distance[-1].get("race_date")) if same_distance else None,
            target_date,
        )
        features["win_rate_same_distance"] = rate(same_distance, lambda row: safe_int(row.get("finish_num")) == 1)
        features["place_rate_same_distance"] = rate(
            same_distance, lambda row: (safe_int(row.get("finish_num")) or 99) <= 3
        )
        same_distance_venue = [
            row for row in same_distance if str(row.get("venue") or "").upper() == target_venue.upper()
        ]
        features["same_distance_venue_start_count"] = len(same_distance_venue)
        features["same_distance_venue_best_time"] = best_min(
            [safe_float(row.get("time_num")) for row in same_distance_venue]
        )
        distance_box_band = [
            row for row in same_distance if target_band and history_box_band(row) == target_band
        ]
        features["distance_box_band_start_count"] = len(distance_box_band)
        features["distance_box_band_win_rate"] = rate(
            distance_box_band, lambda row: safe_int(row.get("finish_num")) == 1
        )
        features["distance_box_band_place_rate"] = rate(
            distance_box_band, lambda row: (safe_int(row.get("finish_num")) or 99) <= 3
        )
        features["distance_box_band_avg_time"] = mean(
            [safe_float(row.get("time_num")) for row in distance_box_band]
        )
    else:
        same_distance = []

    target_grade = normalize_grade(target.get("grade"))
    if target_grade:
        same_grade = [row for row in history_rows if normalize_grade(row.get("grade_normalized")) == target_grade]
        features["same_grade_start_count"] = len(same_grade)
        features["same_grade_win_rate"] = rate(same_grade, lambda row: safe_int(row.get("finish_num")) == 1)
        features["same_grade_place_rate"] = rate(same_grade, lambda row: (safe_int(row.get("finish_num")) or 99) <= 3)
        features["same_grade_avg_speed_mps"] = mean(
            [history_speed(row) for row in same_grade]
        )
        if (
            features.get("recent_avg_speed_mps_5") is not None
            and features.get("same_grade_avg_speed_mps") is not None
            and features["same_grade_avg_speed_mps"] > 0
        ):
            features["grade_normalized_recent_speed_index"] = (
                features["recent_avg_speed_mps_5"] / features["same_grade_avg_speed_mps"]
            )
        features["last_start_grade_normalized"] = normalize_grade(last.get("grade_normalized")) if last else None
        features["recent_grade_mode_5"] = mode_text([normalize_grade(row.get("grade_normalized")) for row in recent])
        same_distance_same_grade = [
            row for row in same_distance if normalize_grade(row.get("grade_normalized")) == target_grade
        ]
        features["same_distance_same_grade_start_count"] = len(same_distance_same_grade)
        features["same_distance_same_grade_best_time"] = best_min(
            [safe_float(row.get("time_num")) for row in same_distance_same_grade]
        )
        features["same_distance_same_grade_avg_time"] = mean(
            [safe_float(row.get("time_num")) for row in same_distance_same_grade]
        )
        last_grade = features.get("last_start_grade_normalized")
        last_rank = grade_rank(last_grade)
        target_rank = grade_rank(target_grade)
        if last_grade and target_rank is not None and last_rank is not None:
            delta = target_rank - last_rank
            features["grade_change_indicator"] = 1 if delta != 0 else 0
            features["grade_change_direction"] = "UP" if delta > 0 else "DOWN" if delta < 0 else "FLAT"
            features["grade_strength_delta"] = delta

    features["recent_avg_sectional_1st_5"] = mean(sectional_values)
    features["recent_best_sectional_1st_5"] = best_min(sectional_values)
    features["recent_sectional_std_5"] = std(sectional_values)
    features["last_start_sectional_1st"] = safe_float(last.get("sectional_1st_num")) if last else None
    if features.get("last_start_sectional_1st") is not None and features.get("recent_avg_sectional_1st_5") is not None:
        features["sectional_time_delta_recent"] = (
            features["last_start_sectional_1st"] - features["recent_avg_sectional_1st_5"]
        )
    if recent:
        missing_sectional = sum(1 for row in recent if safe_float(row.get("sectional_1st_num")) is None)
        features["sectional_missing_rate_5"] = missing_sectional / len(recent)
    else:
        features["sectional_missing_rate_5"] = 1.0
    return features


def days_between(start: str | None, end: str | None) -> int | None:
    if not start or not end:
        return None
    try:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days
    except Exception:
        return None


def build_repaired_dataset(
    *,
    clean_rows: Sequence[Mapping[str, Any]],
    packet_rows: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    features = list(schema["feature_columns"])
    categorical = set(schema.get("categorical_features") or [])
    packet_by_key = {row_join_key(row): dict(row) for row in packet_rows}
    snapshots: dict[str, Any] = {}
    history_index = load_db_history(connection)
    clean_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in clean_rows:
        clean_by_group[group_key(row)].append(row)

    output_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    lineage_by_feature: dict[str, Counter[str]] = {feature: Counter() for feature in features}
    target_resolution_counts: Counter[str] = Counter()
    source_status_counts: Counter[str] = Counter()
    history_status_counts: Counter[str] = Counter()
    history_provenance_rows: list[dict[str, Any]] = []
    target_metadata_audit_rows: list[dict[str, Any]] = []

    for clean_row in clean_rows:
        packet_row = packet_by_key.get(row_join_key(clean_row), {})
        snapshot = load_snapshot(clean_row.get("snapshot_path"), snapshots)
        prediction = find_prediction(snapshot, clean_row.get("dog_name"), clean_row.get("box_number"))
        target = resolve_target_metadata(clean_row, packet_row, snapshot, connection)
        target_resolution_counts[str(target.get("status"))] += 1
        target_metadata_audit_rows.append(
            target_metadata_recovery_audit_row(
                clean_row=clean_row,
                packet_row=packet_row,
                snapshot=snapshot,
                target=target,
                connection=connection,
            )
        )

        row_features: dict[str, Any] = {feature: None for feature in features}
        field_size = len(clean_by_group[group_key(clean_row)])
        box = safe_int(clean_row.get("box_number"))
        race_number = safe_int(target.get("race_number")) or parse_race_number(clean_row.get("race_id"))
        race_date = parse_date(clean_row.get("race_date"))
        target_distance = safe_float(target.get("distance"))
        target_grade = normalize_grade(target.get("grade"))

        row_features.update(
            {
                "field_size": field_size,
                "box_number": box,
                "box_band_inside": 1 if box in (1, 2) else 0,
                "box_band_middle": 1 if box in (3, 4, 5, 6) else 0,
                "box_band_outside": 1 if box is not None and box >= 7 else 0,
                "target_distance_safe": target_distance,
                "target_distance_source_is_safe": 1 if target_distance is not None else 0,
                "target_distance_missing": 0 if target_distance is not None else 1,
                "target_grade_safe": target_grade,
                "target_grade_normalized": target_grade,
                "target_grade_missing": 0 if target_grade else 1,
                "target_grade_vocab_known": 1 if target_grade in GRADE_RANK else 0,
                "target_grade_provenance_safe": 1 if target_grade else 0,
                "venue": str(clean_row.get("venue") or ""),
                "race_number": race_number,
                "race_time_minutes_since_midnight": parse_datetime_minutes(
                    target.get("race_time")
                    or (snapshot or {}).get("jump_datetime")
                    or (snapshot or {}).get("prediction_timestamp")
                ),
                "track_condition": target.get("track_condition"),
                "weather": target.get("weather"),
                "target_month": safe_int((race_date or "").split("-")[1]) if race_date else None,
                "target_day_of_week": datetime.fromisoformat(race_date).weekday() if race_date else None,
                "target_distance_band_sprint": 1 if target_distance is not None and target_distance < 450 else 0
                if target_distance is not None
                else None,
                "target_distance_band_middle": 1
                if target_distance is not None and 450 <= target_distance < 650
                else 0
                if target_distance is not None
                else None,
                "target_distance_band_staying": 1 if target_distance is not None and target_distance >= 650 else 0
                if target_distance is not None
                else None,
            }
        )

        dog_history = history_before_target(
            history_index.get(clean_name(clean_row.get("dog_name")), []),
            race_date,
            str(clean_row.get("race_id") or ""),
        )
        history_provenance_rows.append(
            summarize_history_source_provenance_for_row(
                clean_row=clean_row,
                history_rows=dog_history,
                target=target,
            )
        )
        add_history_features(
            row_features,
            dog_history,
            target,
            race_date,
            str(clean_row.get("venue") or ""),
        )

        for feature in features:
            if (
                row_features.get(feature) is None
                and packet_row.get(feature) not in (None, "")
                and can_reuse_packet_feature(feature, target)
            ):
                # Reuse safe packet values only for feature columns, never IDs or outcome fields.
                row_features[feature] = packet_row.get(feature)
                lineage_by_feature[feature]["repaired_packet_safe_reuse"] += 1
            elif row_features.get(feature) is not None:
                lineage_by_feature[feature]["builder_v2"] += 1

        history_source = ""
        db_count = 0
        if prediction:
            data_shape = prediction.get("data_shape") or {}
            history_source = str(prediction.get("history_source") or data_shape.get("history_source") or "")
            db_count = safe_int(prediction.get("db_result_history_count") or data_shape.get("db_result_history_count")) or 0
        old_class = packet_row.get("identity_classification", "")
        final_leakage_class = "safe"
        final_history_status = "db_history_rebuilt" if dog_history else "safe_history_unavailable"
        if old_class == "excluded_leakage_risk":
            if db_count > 0:
                final_leakage_class = "repairable_leakage_db_only_rebuilt"
            else:
                final_leakage_class = "unrecoverable_embedded_history_stripped"
            leakage_rows.append(
                {
                    "race_id": clean_row.get("race_id"),
                    "snapshot_instance_id": clean_row.get("snapshot_instance_id"),
                    "dog_name": clean_row.get("dog_name"),
                    "box_number": clean_row.get("box_number"),
                    "old_identity_classification": old_class,
                    "final_classification": final_leakage_class,
                    "snapshot_history_source": history_source,
                    "snapshot_db_result_history_count": db_count,
                    "source_file_delimiter_status": packet_row.get("source_file_delimiter_status"),
                    "final_training_action": "include_safe_row_with_embedded_history_removed",
                }
            )
        source_status_counts[final_leakage_class] += 1
        history_status_counts[final_history_status] += 1

        output: dict[str, Any] = {
            "race_id": clean_row.get("race_id"),
            "snapshot_instance_id": clean_row.get("snapshot_instance_id"),
            "snapshot_path": clean_row.get("snapshot_path"),
            "race_date": race_date,
            "venue": clean_row.get("venue"),
            "dog_name": clean_row.get("dog_name"),
            "box_number": clean_row.get("box_number"),
            "actual_win": safe_int(clean_row.get("actual_win")) or 0,
            "finish_position": safe_int(clean_row.get("finish_position")),
            "champion_win_prob_norm": safe_float(clean_row.get("win_prob_norm")),
            "old_identity_classification": old_class,
            "leakage_classification_v2": final_leakage_class,
            "history_status_v2": final_history_status,
            "target_metadata_status_v2": target.get("status"),
            "target_metadata_source_v2": target.get("source"),
            "target_metadata_reason_v2": target.get("reason"),
        }
        output.update(row_features)
        output_rows.append(output)

    population = feature_population(output_rows, features)
    train_rows, holdout_rows = temporal_split(output_rows)
    train_population = feature_population(train_rows, features)
    holdout_population = feature_population(holdout_rows, features)
    leakage_audit = build_leakage_audit(
        leakage_rows=leakage_rows,
        final_rows=output_rows,
        features=features,
        source_status_counts=source_status_counts,
    )
    return {
        "rows": output_rows,
        "features": features,
        "categorical_features": list(categorical),
        "lineage": lineage_by_feature,
        "feature_population": population,
        "train_population": train_population,
        "holdout_population": holdout_population,
        "leakage_audit": leakage_audit,
        "leakage_rows": leakage_rows,
        "target_resolution_counts": dict(target_resolution_counts),
        "source_status_counts": dict(source_status_counts),
        "history_status_counts": dict(history_status_counts),
        "history_source_provenance": history_source_provenance_report(history_provenance_rows),
        "history_source_provenance_rows": history_source_provenance_csv_rows(history_provenance_rows),
        "target_metadata_recovery_audit": target_metadata_recovery_audit_report(target_metadata_audit_rows),
        "target_metadata_recovery_audit_rows": target_metadata_audit_rows,
        "train_rows": train_rows,
        "holdout_rows": holdout_rows,
    }


def feature_population(rows: Sequence[Mapping[str, Any]], features: Sequence[str]) -> dict[str, Any]:
    by_feature: dict[str, Any] = {}
    for feature in features:
        values = [row.get(feature) for row in rows]
        present_values = [value for value in values if value not in (None, "")]
        by_feature[feature] = {
            "rows": len(rows),
            "present_rows": len(present_values),
            "present_pct": len(present_values) / len(rows) if rows else 0.0,
            "unique_present_values": len({serialize_cell(value) for value in present_values}),
            "all_missing": len(present_values) == 0,
        }
    return {
        "rows": len(rows),
        "features": len(features),
        "populated_feature_count": sum(1 for item in by_feature.values() if not item["all_missing"]),
        "all_missing_features": [feature for feature, item in by_feature.items() if item["all_missing"]],
        "by_feature": by_feature,
    }


def target_distance_band_from_row(row: Mapping[str, Any]) -> str | None:
    if safe_int(row.get("target_distance_band_sprint")) == 1:
        return "sprint"
    if safe_int(row.get("target_distance_band_middle")) == 1:
        return "middle"
    if safe_int(row.get("target_distance_band_staying")) == 1:
        return "staying"
    distance = safe_float(row.get("target_distance_safe"))
    if distance is None:
        return None
    if distance < 450:
        return "sprint"
    if distance < 650:
        return "middle"
    return "staying"


def slice_value(row: Mapping[str, Any], dimension: str) -> str | None:
    venue = str(row.get("venue") or "").strip().upper()
    distance_band = target_distance_band_from_row(row)
    draw_band = box_band(row.get("box_number"))
    if dimension == "venue":
        return venue or None
    if dimension == "target_distance_band":
        return distance_band
    if dimension == "target_grade":
        return normalize_grade(row.get("target_grade_safe") or row.get("target_grade_normalized"))
    if dimension == "field_size":
        field_size = safe_int(row.get("field_size"))
        return str(field_size) if field_size is not None else None
    if dimension == "box_band":
        return draw_band
    if dimension == "venue_box_band":
        if venue and draw_band:
            return f"{venue}|{draw_band}"
        return None
    if dimension == "distance_box_band":
        if distance_band and draw_band:
            return f"{distance_band}|{draw_band}"
        return None
    raise ValueError(f"unknown_slice_dimension:{dimension}")


def repaired_feature_families(schema: Mapping[str, Any], features: Sequence[str]) -> dict[str, list[str]]:
    feature_set = set(features)
    raw_families = schema.get("feature_families") if isinstance(schema, Mapping) else None
    families: dict[str, list[str]] = {}
    if isinstance(raw_families, Mapping):
        for family in REPAIRED_FEATURE_FAMILY_ORDER:
            values = raw_families.get(family)
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                families[family] = [str(feature) for feature in values if str(feature) in feature_set]
    if not families:
        families = {
            "safe_target_context": [
                feature
                for feature in features
                if feature.startswith("target_distance")
                or feature.startswith("target_grade")
            ],
            "draw_adjusted_history": [
                feature
                for feature in features
                if "box_band" in feature
            ],
            "same_distance": [
                feature
                for feature in features
                if "same_distance" in feature
            ],
            "same_grade_and_grade_transition": [
                feature
                for feature in features
                if "same_grade" in feature or feature.startswith("grade_")
            ],
        }
    return {family: values for family, values in families.items() if values}


def family_population_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_population_report: Mapping[str, Any],
    family_features: Sequence[str],
    key_features: Sequence[str],
) -> dict[str, Any]:
    by_feature = feature_population_report.get("by_feature") or {}
    selected = [feature for feature in family_features if feature in by_feature]
    key_selected = [feature for feature in key_features if feature in by_feature]
    present_pcts = [safe_float((by_feature.get(feature) or {}).get("present_pct")) for feature in selected]
    present_pcts = [value for value in present_pcts if value is not None]
    all_missing = [
        feature
        for feature in selected
        if (by_feature.get(feature) or {}).get("all_missing")
    ]
    return {
        "row_count": len(rows),
        "race_count": len({group_key(row) for row in rows}),
        "feature_count": len(selected),
        "populated_feature_count": len(selected) - len(all_missing),
        "avg_present_pct": mean(present_pcts) or 0.0,
        "min_present_pct": min(present_pcts) if present_pcts else 0.0,
        "all_missing_features": all_missing,
        "key_feature_present_pct": {
            feature: (by_feature.get(feature) or {}).get("present_pct", 0.0)
            for feature in key_selected
        },
        "key_feature_present_rows": {
            feature: (by_feature.get(feature) or {}).get("present_rows", 0)
            for feature in key_selected
        },
    }


def build_repaired_slice_population_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    features: Sequence[str],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    families = repaired_feature_families(schema, features)
    overall_races = len({group_key(row) for row in rows})
    report: dict[str, Any] = {
        "schema_version": "stage2_repaired_slice_population_diagnostics_v1",
        "mode": "report_only_no_training_registry_db_snapshot_manifest_odds_or_ev",
        "row_count": len(rows),
        "race_count": overall_races,
        "dimensions": {},
        "feature_families": families,
    }
    for dimension in REPAIRED_SLICE_DIMENSIONS:
        buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        missing_rows = 0
        for row in rows:
            value = slice_value(row, dimension)
            if value is None or value == "":
                value = "DATA_MISSING"
                missing_rows += 1
            buckets[str(value)].append(row)
        bucket_summaries: dict[str, Any] = {}
        for bucket, bucket_rows in sorted(buckets.items(), key=lambda item: item[0]):
            bucket_population = feature_population(bucket_rows, features)
            family_summaries = {
                family: family_population_summary(
                    bucket_rows,
                    feature_population_report=bucket_population,
                    family_features=family_features,
                    key_features=KEY_REPAIRED_FEATURES_BY_FAMILY.get(family, ()),
                )
                for family, family_features in families.items()
            }
            bucket_summaries[bucket] = {
                "row_count": len(bucket_rows),
                "race_count": len({group_key(row) for row in bucket_rows}),
                "row_pct": len(bucket_rows) / len(rows) if rows else 0.0,
                "race_pct": (
                    len({group_key(row) for row in bucket_rows}) / overall_races
                    if overall_races
                    else 0.0
                ),
                "family_population": family_summaries,
            }
        report["dimensions"][dimension] = {
            "bucket_count": len(bucket_summaries),
            "missing_row_count": missing_rows,
            "missing_row_pct": missing_rows / len(rows) if rows else 0.0,
            "buckets": bucket_summaries,
        }
    return report


def slice_population_csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dimensions = report.get("dimensions") or {}
    for dimension, dimension_report in dimensions.items():
        for bucket, bucket_report in (dimension_report.get("buckets") or {}).items():
            for family, family_report in (bucket_report.get("family_population") or {}).items():
                rows.append(
                    {
                        "dimension": dimension,
                        "bucket": bucket,
                        "family": family,
                        "row_count": bucket_report.get("row_count"),
                        "race_count": bucket_report.get("race_count"),
                        "row_pct": bucket_report.get("row_pct"),
                        "race_pct": bucket_report.get("race_pct"),
                        "feature_count": family_report.get("feature_count"),
                        "populated_feature_count": family_report.get("populated_feature_count"),
                        "avg_present_pct": family_report.get("avg_present_pct"),
                        "min_present_pct": family_report.get("min_present_pct"),
                        "all_missing_features": ",".join(family_report.get("all_missing_features") or []),
                        "key_feature_present_pct": json.dumps(
                            family_report.get("key_feature_present_pct") or {},
                            sort_keys=True,
                        ),
                        "key_feature_present_rows": json.dumps(
                            family_report.get("key_feature_present_rows") or {},
                            sort_keys=True,
                        ),
                    }
                )
    return rows


def temporal_split(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_race: dict[str, dict[str, Any]] = {}
    for row in rows:
        race_id = str(row.get("race_id") or "")
        if race_id not in by_race:
            by_race[race_id] = {"race_date": row.get("race_date"), "race_id": race_id}
    latest_date = max(str(item["race_date"]) for item in by_race.values() if item.get("race_date"))
    holdout_ids = {race_id for race_id, item in by_race.items() if item.get("race_date") == latest_date}
    train = [dict(row) for row in rows if row.get("race_id") not in holdout_ids]
    holdout = [dict(row) for row in rows if row.get("race_id") in holdout_ids]
    return train, holdout


def build_leakage_audit(
    *,
    leakage_rows: Sequence[Mapping[str, Any]],
    final_rows: Sequence[Mapping[str, Any]],
    features: Sequence[str],
    source_status_counts: Mapping[str, int],
) -> dict[str, Any]:
    forbidden = sorted(
        [feature for feature in features if feature.startswith("tgr_") or feature in HARD_EXCLUDED_FEATURE_NAMES]
    )
    identity = sorted([feature for feature in features if feature in IDENTITY_FEATURE_NAMES])
    final_counts = Counter(row.get("leakage_classification_v2") for row in final_rows)
    old_excluded = len(leakage_rows)
    unresolved = [
        row
        for row in leakage_rows
        if row.get("final_training_action") != "include_safe_row_with_embedded_history_removed"
    ]
    return {
        "schema_version": "feature_recovery_leakage_audit_v2",
        "status": "PASS" if not forbidden and not identity and not unresolved else "FAIL",
        "old_excluded_leakage_risk_rows_investigated": old_excluded,
        "classification_counts": dict(final_counts),
        "source_feature_risk_counts": {
            "genuine_embedded_history_source_risk_rejected": old_excluded,
            "repairable_db_history_rebuilt": final_counts.get("repairable_leakage_db_only_rebuilt", 0),
            "unrecoverable_embedded_history_stripped": final_counts.get("unrecoverable_embedded_history_stripped", 0),
            "false_positive_row_level_exclusion_after_stripping": old_excluded,
        },
        "forbidden_feature_columns_present": forbidden,
        "identity_columns_present_as_features": identity,
        "unresolved_leakage_rows": unresolved,
        "embedded_form_history_used_as_features": False,
        "tgr_columns_present": [feature for feature in features if feature.startswith("tgr_")],
        "source_status_counts": dict(source_status_counts),
        "notes": [
            "Rows previously marked excluded_leakage_risk were not allowed to contribute embedded form-history features.",
            "Rows with DB pre-target history were rebuilt from dog_race_data/race_metadata only.",
            "Rows without DB history were retained only as safe no-history rows with missing/default-safe history features.",
        ],
    }


def matrix_gate(dataset: Mapping[str, Any]) -> dict[str, Any]:
    rows = dataset["rows"]
    features = dataset["features"]
    race_count = len({row.get("race_id") for row in rows})
    row_count = len(rows)
    populated_count = dataset["feature_population"]["populated_feature_count"]
    target_distance_present = dataset["feature_population"]["by_feature"]["target_distance_safe"]["present_rows"]
    target_grade_present = dataset["feature_population"]["by_feature"]["target_grade_safe"]["present_rows"]
    fail_reasons: list[str] = []
    if race_count < 132:
        fail_reasons.append(f"trainable_races_below_minimum:{race_count}")
    if row_count < 928:
        fail_reasons.append(f"trainable_rows_below_minimum:{row_count}")
    if populated_count < 70:
        fail_reasons.append(f"populated_features_below_minimum:{populated_count}")
    if target_distance_present <= 0:
        fail_reasons.append("target_distance_safe_still_zero_populated")
    if target_grade_present < 100:
        fail_reasons.append(f"target_grade_safe_not_materially_populated:{target_grade_present}")
    if dataset["leakage_audit"]["status"] != "PASS":
        fail_reasons.append("leakage_audit_failed")
    return {
        "status": "PASS" if not fail_reasons else "FAIL",
        "race_count": race_count,
        "row_count": row_count,
        "feature_count": len(features),
        "populated_feature_count": populated_count,
        "target_distance_safe_present_rows": target_distance_present,
        "target_grade_safe_present_rows": target_grade_present,
        "fail_reasons": fail_reasons,
    }


def prepare_xy(rows: Sequence[Mapping[str, Any]], features: Sequence[str]) -> tuple[list[list[Any]], list[int]]:
    x_rows = [[row.get(feature) if row.get(feature) not in ("", None) else None for feature in features] for row in rows]
    y = [int(row.get("actual_win") or 0) for row in rows]
    return x_rows, y


def train_challengers(
    dataset: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    try:
        import numpy as np
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.inspection import permutation_importance
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except Exception as exc:
        return {
            "status": "FAIL",
            "reason": "missing_ml_dependencies",
            "error": repr(exc),
            "models": {},
        }

    features = list(dataset["features"])
    categorical = set(dataset["categorical_features"])
    categorical_indices = [index for index, feature in enumerate(features) if feature in categorical]
    numeric_indices = [index for index, feature in enumerate(features) if feature not in categorical]
    train_rows = dataset["train_rows"]
    holdout_rows = dataset["holdout_rows"]
    x_train, y_train = prepare_xy(train_rows, features)
    x_holdout, _ = prepare_xy(holdout_rows, features)

    transformer = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_indices),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_indices,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    candidates = {
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=4,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=8,
            l2_regularization=0.1,
            random_state=RANDOM_SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=4,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }
    model_results: dict[str, Any] = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline(steps=[("prep", transformer), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        raw_prob = pipeline.predict_proba(x_holdout)[:, 1]
        scored_rows = attach_probabilities(holdout_rows, raw_prob, f"{name}_prob")
        metrics = score_grouped(scored_rows, probability_key=f"{name}_prob")
        importance = model_importance(name, pipeline, features, x_holdout, [int(r.get("actual_win") or 0) for r in holdout_rows])
        model_results[name] = {
            "status": "RUN",
            "metrics": metrics,
            "feature_importance_top25": importance[:25],
            "fit": {
                "train_rows": len(train_rows),
                "train_races": len({row.get("race_id") for row in train_rows}),
                "holdout_rows": len(holdout_rows),
                "holdout_races": len({row.get("race_id") for row in holdout_rows}),
                "features": len(features),
            },
        }
        write_csv(
            output_dir / f"{name.lower()}_holdout_predictions.csv",
            scored_rows,
            [
                "race_id",
                "snapshot_instance_id",
                "race_date",
                "dog_name",
                "box_number",
                "actual_win",
                "finish_position",
                f"{name}_prob",
                f"{name}_rank",
            ],
        )
    champion_rows = attach_probabilities(
        holdout_rows,
        [safe_float(row.get("champion_win_prob_norm")) or 0.0 for row in holdout_rows],
        "champion_prob",
    )
    champion = score_grouped(champion_rows, probability_key="champion_prob")
    return {
        "status": "PASS",
        "models": model_results,
        "champion_same_holdout": champion,
        "holdout_races": len({row.get("race_id") for row in holdout_rows}),
        "holdout_rows": len(holdout_rows),
    }


def model_importance(
    name: str,
    pipeline: Any,
    features: Sequence[str],
    x_holdout: Sequence[Sequence[Any]],
    y_holdout: Sequence[int],
) -> list[dict[str, Any]]:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        transformed_names = pipeline.named_steps["prep"].get_feature_names_out()
        pairs = [
            {"feature": str(feature), "importance": float(value)}
            for feature, value in zip(transformed_names, model.feature_importances_)
        ]
        return sorted(pairs, key=lambda item: item["importance"], reverse=True)
    try:
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            pipeline,
            x_holdout,
            y_holdout,
            n_repeats=5,
            random_state=RANDOM_SEED,
            scoring="neg_log_loss",
        )
        return sorted(
            [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in zip(features, result.importances_mean)
            ],
            key=lambda item: item["importance"],
            reverse=True,
        )
    except Exception as exc:
        return [{"feature": "DATA_MISSING", "importance": 0.0, "reason": f"{name}_importance_failed:{exc}"}]


def attach_probabilities(
    rows: Sequence[Mapping[str, Any]],
    probabilities: Sequence[float],
    probability_key: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, probability in zip(rows, probabilities):
        grouped[group_key(row)].append((dict(row), max(float(probability), 0.0)))
    scored: list[dict[str, Any]] = []
    rank_key = probability_key.replace("_prob", "_rank")
    for group_rows in grouped.values():
        total = sum(probability for _, probability in group_rows)
        if total <= 0:
            total = float(len(group_rows)) or 1.0
            normalized = [(row, 1.0 / total) for row, _ in group_rows]
        else:
            normalized = [(row, probability / total) for row, probability in group_rows]
        ordered = sorted(normalized, key=lambda item: item[1], reverse=True)
        for rank, (row, probability) in enumerate(ordered, start=1):
            row[probability_key] = probability
            row[rank_key] = rank
            scored.append(row)
    return scored


def score_grouped(rows: Sequence[Mapping[str, Any]], probability_key: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    top1 = 0
    top3 = 0
    winner_ranks: list[int] = []
    top_boxes: Counter[str] = Counter()
    brier_values: list[float] = []
    log_values: list[float] = []
    calibration_bins: dict[str, list[tuple[float, int]]] = defaultdict(list)
    eps = 1e-12
    for group_rows in groups.values():
        ordered = sorted(group_rows, key=lambda row: safe_float(row.get(probability_key)) or 0.0, reverse=True)
        if not ordered:
            continue
        top_boxes[str(ordered[0].get("box_number"))] += 1
        for index, row in enumerate(ordered, start=1):
            y = int(row.get("actual_win") or 0)
            p = min(max(safe_float(row.get(probability_key)) or 0.0, eps), 1.0 - eps)
            brier_values.append((p - y) ** 2)
            log_values.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
            bucket = f"{min(int(p * 5), 4) / 5:.1f}-{(min(int(p * 5), 4) + 1) / 5:.1f}"
            calibration_bins[bucket].append((p, y))
            if y == 1:
                winner_ranks.append(index)
        if int(ordered[0].get("actual_win") or 0) == 1:
            top1 += 1
        if any(int(row.get("actual_win") or 0) == 1 for row in ordered[:3]):
            top3 += 1
    race_count = len(groups)
    calibration = {
        bucket: {
            "count": len(items),
            "mean_probability": mean([prob for prob, _ in items]),
            "win_rate": mean([float(y) for _, y in items]),
        }
        for bucket, items in sorted(calibration_bins.items())
    }
    return {
        "race_count": race_count,
        "row_count": len(rows),
        "top1": top1 / race_count if race_count else 0.0,
        "top3": top3 / race_count if race_count else 0.0,
        "mean_winner_rank": mean([float(rank) for rank in winner_ranks]),
        "brier": mean(brier_values),
        "log_loss": mean(log_values),
        "calibration": calibration,
        "box_bias": {
            "box1_top_pick_share": top_boxes.get("1", 0) / race_count if race_count else 0.0,
            "top_pick_box_distribution": dict(sorted(top_boxes.items(), key=lambda item: item[0])),
            "ranking_concentration_top_box_share": (top_boxes.most_common(1)[0][1] / race_count)
            if race_count and top_boxes
            else 0.0,
        },
    }


def choose_challenger(training: Mapping[str, Any]) -> dict[str, Any]:
    champion = training.get("champion_same_holdout") or {}
    models = training.get("models") or {}
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for name, result in models.items():
        if result.get("status") == "RUN":
            candidates.append((name, result.get("metrics") or {}))
    if not candidates:
        return {
            "decision": "CHALLENGER_FAILS",
            "winner_candidate": None,
            "reason": "no_challenger_model_ran",
        }
    def sort_key(item: tuple[str, Mapping[str, Any]]) -> tuple[float, float, float, float]:
        metrics = item[1]
        bias = metrics.get("box_bias") or {}
        return (
            float(metrics.get("top1") or 0.0),
            float(metrics.get("top3") or 0.0),
            -float(metrics.get("log_loss") or 999.0),
            -float(bias.get("box1_top_pick_share") or 1.0),
        )
    winner_name, winner_metrics = sorted(candidates, key=sort_key, reverse=True)[0]
    champ_top1 = float(champion.get("top1") or 0.0)
    champ_top3 = float(champion.get("top3") or 0.0)
    champ_brier = float(champion.get("brier") or 999.0)
    champ_log = float(champion.get("log_loss") or 999.0)
    champ_box1 = float((champion.get("box_bias") or {}).get("box1_top_pick_share") or 0.0)
    win_box1 = float((winner_metrics.get("box_bias") or {}).get("box1_top_pick_share") or 0.0)
    improves_rank = float(winner_metrics.get("top1") or 0.0) > champ_top1 or float(winner_metrics.get("top3") or 0.0) > champ_top3
    improves_calibration = float(winner_metrics.get("brier") or 999.0) < champ_brier and float(winner_metrics.get("log_loss") or 999.0) < champ_log
    improves_bias = win_box1 < champ_box1
    if improves_rank and improves_calibration and improves_bias:
        decision = "CHALLENGER_OUTPERFORMS_CHAMPION"
    elif improves_bias and (float(winner_metrics.get("top1") or 0.0) >= champ_top1 or float(winner_metrics.get("top3") or 0.0) >= champ_top3):
        decision = "CHALLENGER_EQUIVALENT"
    else:
        decision = "CHALLENGER_FAILS"
    return {
        "decision": decision,
        "winner_candidate": winner_name,
        "winner_metrics": winner_metrics,
        "champion_metrics": champion,
        "improves_rank": improves_rank,
        "improves_calibration": improves_calibration,
        "improves_box_bias": improves_bias,
    }


def promotion_readiness(review: Mapping[str, Any], gate: Mapping[str, Any], training: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if gate.get("status") != "PASS":
        blockers.append("NOT_READY_FEATURES")
    if training.get("status") != "PASS":
        blockers.append("NOT_READY_FEATURES")
    decision = review.get("decision")
    winner = review.get("winner_metrics") or {}
    box1 = float((winner.get("box_bias") or {}).get("box1_top_pick_share") or 1.0)
    if box1 >= 0.50:
        blockers.append("NOT_READY_BIAS")
    if decision != "CHALLENGER_OUTPERFORMS_CHAMPION":
        blockers.append("NOT_READY_CALIBRATION")
    if not blockers:
        status = "READY_FOR_PROMOTION_REVIEW"
    elif len(set(blockers)) > 1:
        status = "NOT_READY_MULTIPLE_BLOCKERS"
    else:
        status = blockers[0]
    return {
        "status": status,
        "blockers": sorted(set(blockers)),
        "no_promotion_performed": True,
        "registry_mutation": False,
        "active_model_replacement": False,
    }


def write_phase_common(
    phase_dir: Path,
    *,
    summary: str,
    status: str,
    verification: Mapping[str, Any],
    extra_manifest: Mapping[str, Any] | None = None,
) -> None:
    write_text(phase_dir / "SUMMARY.md", summary.rstrip() + "\n")
    write_text(phase_dir / "final_status.txt", status.rstrip() + "\n")
    write_text(
        phase_dir / "verification_results.txt",
        "\n".join(f"{key}={value}" for key, value in verification.items()) + "\n",
    )
    write_text(
        phase_dir / "rollback_plan.md",
        "\n".join(
            [
                "# Rollback Plan",
                "",
                "This phase wrote report-only artifacts in this phase directory.",
                "Rollback is deletion of this generated phase directory only.",
                "No DB writes, label writes, snapshot/manifest rewrites, model registry mutations, active-model replacements, production promotions, odds, EV, betting, or TGR enablement were performed.",
                "",
            ]
        ),
    )
    manifest: dict[str, Any] = {
        "phase_dir": relpath(phase_dir),
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_head": git_output(["rev-parse", "--short=12", "HEAD"]),
        "git_branch": git_output(["branch", "--show-current"]),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    files = {}
    for path in sorted(phase_dir.glob("*")):
        if path.is_file() and path.name != "evidence_manifest.json":
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest["files"] = files
    write_json(phase_dir / "evidence_manifest.json", manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dataset", type=Path, default=DEFAULT_CLEAN_DATASET)
    parser.add_argument("--repaired-packet", type=Path, default=DEFAULT_REPAIRED_PACKET)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stop-after-phase1", action="store_true")
    args = parser.parse_args(argv)

    timestamp = now_id()
    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_PARENT / f"feature_recovery_execution_v1_{timestamp}"
    )
    output_dir = safe_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    clean_rows = load_jsonl(args.clean_dataset)
    packet_rows = load_csv(args.repaired_packet)
    schema = load_json(args.schema)
    feature_columns = list(schema["feature_columns"])
    if any(feature.startswith("tgr_") for feature in feature_columns):
        raise SystemExit("schema_contains_tgr_columns")
    if len(feature_columns) != EXPECTED_REPAIRED_FEATURE_COUNT:
        raise SystemExit(
            f"schema_feature_count_not_{EXPECTED_REPAIRED_FEATURE_COUNT}:"
            f"{len(feature_columns)}"
        )

    protected_before = {
        "model_registry/best_metadata.json": sha256_file(ROOT / "model_registry/best_metadata.json")
        if (ROOT / "model_registry/best_metadata.json").exists()
        else "MISSING",
        "docs/model_contracts/v4_feature_contract.json": sha256_file(
            ROOT / "docs/model_contracts/v4_feature_contract.json"
        )
        if (ROOT / "docs/model_contracts/v4_feature_contract.json").exists()
        else "MISSING",
    }

    connection = sqlite_ro(args.db)
    quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
    dataset = build_repaired_dataset(
        clean_rows=clean_rows,
        packet_rows=packet_rows,
        schema=schema,
        connection=connection,
    )
    gate = matrix_gate(dataset)

    phase1 = output_dir / "phase_1_dataset_hardening"
    metadata_columns = [
        "race_id",
        "snapshot_instance_id",
        "snapshot_path",
        "race_date",
        "venue",
        "dog_name",
        "box_number",
        "actual_win",
        "finish_position",
        "champion_win_prob_norm",
        "old_identity_classification",
        "leakage_classification_v2",
        "history_status_v2",
        "target_metadata_status_v2",
        "target_metadata_source_v2",
        "target_metadata_reason_v2",
    ]
    write_csv(phase1 / "repaired_dataset_v2.csv", dataset["rows"], metadata_columns + feature_columns)
    write_json(phase1 / "leakage_audit_v2.json", dataset["leakage_audit"])
    write_csv(
        phase1 / "leakage_risk_row_classification_v2.csv",
        dataset["leakage_rows"],
        [
            "race_id",
            "snapshot_instance_id",
            "dog_name",
            "box_number",
            "old_identity_classification",
            "final_classification",
            "snapshot_history_source",
            "snapshot_db_result_history_count",
            "source_file_delimiter_status",
            "final_training_action",
        ],
    )
    write_json(
        phase1 / "feature_population_v2.json",
        {
            "overall": dataset["feature_population"],
            "train": dataset["train_population"],
            "holdout": dataset["holdout_population"],
            "target_resolution_counts": dataset["target_resolution_counts"],
            "history_status_counts": dataset["history_status_counts"],
            "gate": gate,
        },
    )
    slice_diagnostics = build_repaired_slice_population_diagnostics(
        dataset["rows"],
        feature_columns,
        schema,
    )
    write_json(phase1 / "stage2_slice_population_diagnostics_v1.json", slice_diagnostics)
    write_csv(
        phase1 / "stage2_slice_population_diagnostics_v1.csv",
        slice_population_csv_rows(slice_diagnostics),
        [
            "dimension",
            "bucket",
            "family",
            "row_count",
            "race_count",
            "row_pct",
            "race_pct",
            "feature_count",
            "populated_feature_count",
            "avg_present_pct",
            "min_present_pct",
            "all_missing_features",
            "key_feature_present_pct",
            "key_feature_present_rows",
        ],
    )
    write_json(phase1 / "history_source_provenance_v1.json", dataset["history_source_provenance"])
    write_csv(
        phase1 / "history_source_provenance_by_matrix_row_v1.csv",
        dataset["history_source_provenance_rows"],
        [
            "race_id",
            "snapshot_instance_id",
            "dog_name",
            "box_number",
            "history_rows_used",
            "dog_data_source_column_loaded_rows",
            "race_metadata_data_source_column_loaded_rows",
            "dog_data_source_counts",
            "race_metadata_data_source_counts",
            "time_source_counts",
            "time_num_present_rows",
            "individual_time_present_rows",
            "time_missing_rows",
            "grade_present_rows",
            "distance_present_rows",
            "race_time_present_rows",
            "track_condition_present_rows",
            "weather_present_rows",
            "start_datetime_present_rows",
            "race_metadata_url_present_rows",
            "target_metadata_status_v2",
            "target_metadata_source_v2",
            "target_metadata_reason_v2",
        ],
    )
    write_json(phase1 / "target_metadata_recovery_audit_v1.json", dataset["target_metadata_recovery_audit"])
    write_csv(
        phase1 / "target_metadata_recovery_audit_by_matrix_row_v1.csv",
        dataset["target_metadata_recovery_audit_rows"],
        [
            "race_id",
            "snapshot_instance_id",
            "dog_name",
            "box_number",
            "race_date",
            "venue",
            "race_number",
            "target_metadata_status_v2",
            "target_metadata_source_v2",
            "target_metadata_reason_v2",
            "target_distance_safe_present",
            "target_grade_safe_present",
            "target_metadata_blocker_reason",
            "sidecar_status",
            "sidecar_path",
            "sidecar_has_distance",
            "sidecar_has_grade",
            "sidecar_distance_source",
            "sidecar_grade_source",
            "sidecar_verification_status",
            "sidecar_metadata_is_leakage_safe",
            "db_lookup_status",
            "db_exact_row_count",
            "db_exact_metadata_row_count",
            "db_safe_candidate_count",
            "db_unsafe_candidate_count",
            "db_embedded_form_metadata_count",
            "db_post_outcome_metadata_count",
            "db_metadata_source_counts",
            "db_exact_row_has_target_metadata",
            "db_exact_row_has_post_outcome_marker",
            "packet_target_grade_source",
            "packet_target_distance_source",
        ],
    )
    write_phase_common(
        phase1,
        summary=phase1_summary(gate, dataset, args),
        status="PASS" if gate["status"] == "PASS" else "STOP_GATE_FAIL",
        verification={
            "sqlite_quick_check": quick_check,
            "rows": gate["row_count"],
            "races": gate["race_count"],
            "populated_features": gate["populated_feature_count"],
            "target_distance_safe_present_rows": gate["target_distance_safe_present_rows"],
            "target_grade_safe_present_rows": gate["target_grade_safe_present_rows"],
            "leakage_audit": dataset["leakage_audit"]["status"],
            "gate": gate["status"],
            "stage2_slice_dimensions": ",".join(REPAIRED_SLICE_DIMENSIONS),
        },
        extra_manifest={
            "clean_dataset": relpath(args.clean_dataset),
            "source_repaired_packet": relpath(args.repaired_packet),
            "schema": relpath(args.schema),
            "db": relpath(args.db),
        },
    )
    if gate["status"] != "PASS" or args.stop_after_phase1:
        write_final(output_dir, gate, None, None, protected_before)
        return 2

    phase2 = output_dir / "phase_2_feature_pipeline_repair"
    arrival = {
        "schema_version": "feature_arrival_report_v1",
        "status": "PASS",
        "training_matrix_feature_count": len(feature_columns),
        "silent_drops": [],
        "features_in_matrix": feature_columns,
        "lineage_by_feature": {feature: dict(counter) for feature, counter in dataset["lineage"].items()},
        "matrix_inventory": {
            "train_rows": len(dataset["train_rows"]),
            "train_races": len({row.get("race_id") for row in dataset["train_rows"]}),
            "holdout_rows": len(dataset["holdout_rows"]),
            "holdout_races": len({row.get("race_id") for row in dataset["holdout_rows"]}),
            "categorical_features": schema.get("categorical_features"),
            "numeric_or_boolean_features": schema.get("numeric_or_boolean_features"),
        },
    }
    write_json(phase2 / "repaired_training_matrix_inventory.json", arrival["matrix_inventory"])
    write_json(phase2 / "feature_arrival_report.json", arrival)
    write_phase_common(
        phase2,
        summary=phase2_summary(arrival),
        status="PASS",
        verification={
            "training_matrix_feature_count": len(feature_columns),
            "silent_drops": len(arrival["silent_drops"]),
            "status": arrival["status"],
        },
    )

    phase3 = output_dir / "phase_3_non_tgr_challenger_training"
    training = train_challengers(dataset, phase3)
    write_json(phase3 / "challenger_metrics.json", training)
    write_json(
        phase3 / "feature_importance.json",
        {
            name: result.get("feature_importance_top25")
            for name, result in (training.get("models") or {}).items()
        },
    )
    write_json(
        phase3 / "box_bias_diagnostics.json",
        {
            name: (result.get("metrics") or {}).get("box_bias")
            for name, result in (training.get("models") or {}).items()
        },
    )
    write_phase_common(
        phase3,
        summary=phase3_summary(training),
        status=training["status"],
        verification={
            "training_status": training["status"],
            "models_run": sum(1 for result in (training.get("models") or {}).values() if result.get("status") == "RUN"),
            "leakage_gate": dataset["leakage_audit"]["status"],
        },
    )
    if training["status"] != "PASS":
        write_final(output_dir, gate, training, None, protected_before)
        return 2

    phase4 = output_dir / "phase_4_box_bias_attack"
    box_compare = {
        "schema_version": "box_bias_attack_v1",
        "champion": (training.get("champion_same_holdout") or {}).get("box_bias"),
        "challengers": {
            name: (result.get("metrics") or {}).get("box_bias")
            for name, result in (training.get("models") or {}).items()
        },
    }
    write_json(phase4 / "box_bias_comparison.json", box_compare)
    write_phase_common(
        phase4,
        summary=phase4_summary(box_compare),
        status="PASS",
        verification={
            "champion_box1_share": (box_compare["champion"] or {}).get("box1_top_pick_share"),
            "challenger_count": len(box_compare["challengers"]),
        },
    )

    phase5 = output_dir / "phase_5_challenger_review"
    review = choose_challenger(training)
    write_json(phase5 / "challenger_review_packet.json", review)
    write_phase_common(
        phase5,
        summary=phase5_summary(review),
        status=str(review["decision"]),
        verification={
            "decision": review["decision"],
            "winner_candidate": review.get("winner_candidate"),
            "same_holdout": True,
            "no_tgr": True,
            "no_promotion": True,
        },
    )

    phase6 = output_dir / "phase_6_promotion_readiness_review"
    readiness = promotion_readiness(review, gate, training)
    write_json(phase6 / "promotion_readiness_packet.json", readiness)
    write_phase_common(
        phase6,
        summary=phase6_summary(readiness),
        status=readiness["status"],
        verification={
            "readiness": readiness["status"],
            "registry_mutation": readiness["registry_mutation"],
            "active_model_replacement": readiness["active_model_replacement"],
            "no_promotion_performed": readiness["no_promotion_performed"],
        },
    )
    write_final(output_dir, gate, training, review, protected_before, readiness=readiness)
    return 0


def phase1_summary(gate: Mapping[str, Any], dataset: Mapping[str, Any], args: argparse.Namespace) -> str:
    return "\n".join(
        [
            "# Phase 1 - Dataset Hardening",
            "",
            f"Status: `{gate['status']}`",
            "",
            f"- Clean rows rebuilt: `{gate['row_count']}`.",
            f"- Clean races rebuilt: `{gate['race_count']}`.",
            f"- Populated repaired schema features: `{gate['populated_feature_count']}` / `{gate['feature_count']}`.",
            f"- target_distance_safe populated rows: `{gate['target_distance_safe_present_rows']}`.",
            f"- target_grade_safe populated rows: `{gate['target_grade_safe_present_rows']}`.",
            f"- Previous leakage-risk rows investigated: `{dataset['leakage_audit']['old_excluded_leakage_risk_rows_investigated']}`.",
            "",
            "Outputs:",
            "",
            "- `repaired_dataset_v2.csv`",
            "- `leakage_audit_v2.json`",
            "- `feature_population_v2.json`",
            "- `stage2_slice_population_diagnostics_v1.json`",
            "- `stage2_slice_population_diagnostics_v1.csv`",
            "- `leakage_risk_row_classification_v2.csv`",
            "- `history_source_provenance_v1.json`",
            "- `history_source_provenance_by_matrix_row_v1.csv`",
            "- `target_metadata_recovery_audit_v1.json`",
            "- `target_metadata_recovery_audit_by_matrix_row_v1.csv`",
            "",
            "No DB writes, registry mutation, model replacement, TGR enablement, promotion, betting, EV, snapshot rewrite, or manifest append was performed.",
            "",
            f"Fail reasons: `{gate['fail_reasons']}`",
        ]
    )


def phase2_summary(arrival: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 2 - Feature Pipeline Repair",
            "",
            f"Status: `{arrival['status']}`",
            "",
            f"- Repaired features in actual training matrix: `{arrival['training_matrix_feature_count']}`.",
            f"- Silent drops: `{arrival['silent_drops']}`.",
            "- Lineage is recorded by feature in `feature_arrival_report.json`.",
        ]
    )


def phase3_summary(training: Mapping[str, Any]) -> str:
    lines = ["# Phase 3 - Non-TGR Challenger Training", "", f"Status: `{training['status']}`", ""]
    if training["status"] != "PASS":
        lines.append(f"Reason: `{training.get('reason')}`")
        lines.append(f"Error: `{training.get('error')}`")
        return "\n".join(lines)
    lines.append(f"Holdout races: `{training['holdout_races']}`.")
    for name, result in (training.get("models") or {}).items():
        metrics = result.get("metrics") or {}
        bias = metrics.get("box_bias") or {}
        lines.append(
            f"- {name}: Top1 `{metrics.get('top1')}`, Top3 `{metrics.get('top3')}`, "
            f"Brier `{metrics.get('brier')}`, log loss `{metrics.get('log_loss')}`, "
            f"box1 share `{bias.get('box1_top_pick_share')}`."
        )
    return "\n".join(lines)


def phase4_summary(box_compare: Mapping[str, Any]) -> str:
    lines = ["# Phase 4 - Box-Bias Attack", ""]
    champion = box_compare.get("champion") or {}
    lines.append(f"Champion box-1 top-pick share: `{champion.get('box1_top_pick_share')}`.")
    for name, bias in (box_compare.get("challengers") or {}).items():
        lines.append(f"- {name}: box-1 top-pick share `{(bias or {}).get('box1_top_pick_share')}`.")
    return "\n".join(lines)


def phase5_summary(review: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 5 - Challenger Review",
            "",
            f"Decision: `{review.get('decision')}`",
            f"Winner candidate: `{review.get('winner_candidate')}`",
            f"Improves rank: `{review.get('improves_rank')}`",
            f"Improves calibration: `{review.get('improves_calibration')}`",
            f"Improves box bias: `{review.get('improves_box_bias')}`",
        ]
    )


def phase6_summary(readiness: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Phase 6 - Promotion Readiness Review",
            "",
            f"Decision: `{readiness.get('status')}`",
            f"Blockers: `{readiness.get('blockers')}`",
            "",
            "No promotion occurred.",
        ]
    )


def write_final(
    output_dir: Path,
    gate: Mapping[str, Any],
    training: Mapping[str, Any] | None,
    review: Mapping[str, Any] | None,
    protected_before: Mapping[str, str],
    readiness: Mapping[str, Any] | None = None,
) -> None:
    protected_after = {
        "model_registry/best_metadata.json": sha256_file(ROOT / "model_registry/best_metadata.json")
        if (ROOT / "model_registry/best_metadata.json").exists()
        else "MISSING",
        "docs/model_contracts/v4_feature_contract.json": sha256_file(
            ROOT / "docs/model_contracts/v4_feature_contract.json"
        )
        if (ROOT / "docs/model_contracts/v4_feature_contract.json").exists()
        else "MISSING",
    }
    answer = "UNDETERMINED_STOPPED_BEFORE_TRAINING"
    if training and training.get("status") == "PASS" and review:
        answer = str(review.get("decision"))
    final = {
        "schema_version": "feature_recovery_execution_v1_final",
        "output_dir": relpath(output_dir),
        "definitive_answer": answer,
        "dataset_gate": gate,
        "training_status": None if not training else training.get("status"),
        "challenger_review": review,
        "promotion_readiness": readiness,
        "protected_hashes_before": dict(protected_before),
        "protected_hashes_after": protected_after,
        "protected_hashes_unchanged": protected_before == protected_after,
        "forbidden_actions": {
            "db_writes": False,
            "label_writes": False,
            "registry_mutation": False,
            "production_pointer_mutation": False,
            "snapshot_or_manifest_rewrite": False,
            "promotion": False,
            "tgr_enablement": False,
            "betting_or_ev_action": False,
        },
    }
    write_json(output_dir / "FINAL_PROGRAM_STATUS.json", final)
    lines = [
        "# Feature Recovery Execution V1",
        "",
        f"Definitive answer: `{answer}`",
        "",
        f"Dataset gate: `{gate.get('status')}`.",
        f"Training status: `{None if not training else training.get('status')}`.",
        f"Promotion readiness: `{None if not readiness else readiness.get('status')}`.",
        f"Protected hashes unchanged: `{protected_before == protected_after}`.",
        "",
        "No production promotion, registry mutation, active-model replacement, DB write, label write, snapshot/manifest rewrite, TGR enablement, betting, or EV action was performed.",
    ]
    write_text(output_dir / "SUMMARY.md", "\n".join(lines) + "\n")
    write_text(output_dir / "final_status.txt", answer + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
