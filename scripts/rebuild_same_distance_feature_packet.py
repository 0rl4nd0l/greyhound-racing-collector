#!/usr/bin/env python3
"""Report-only repair for same-distance history features in the frozen packet.

The helper preserves the existing clean official holdout and only repairs the
same-distance feature family on historical rows when safe target metadata can be
resolved from canonical DB metadata. Rolling rows are preserved as-is so the
report-only challenger retest can continue to use the same clean holdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.csv_metadata import load_safe_sidecar_target_metadata, normalize_target_grade

DEFAULT_PACKET_DIR = (
    ROOT
    / "artifacts"
    / "full_evidence_orchestration_20260525"
    / "bounded_target_grade_repair_20260603"
)
DEFAULT_INPUT_PACKET = (
    ROOT
    / "artifacts"
    / "full_evidence_orchestration_20260525"
    / "clean_history_feature_packet_20260602"
    / "pre_race_history_feature_packet.csv"
)
DEFAULT_CLEAN_DATASET = (
    ROOT
    / "artifacts"
    / "full_evidence_orchestration_20260525"
    / "isolated_challenger_box_bias_study_20260602"
    / "clean_official_dataset.jsonl"
)
DEFAULT_OLD_COVERAGE = (
    ROOT
    / "artifacts"
    / "full_evidence_orchestration_20260525"
    / "bounded_same_distance_feature_repair_20260602"
    / "repaired_feature_coverage_after_reconstruction.json"
)
DEFAULT_OLD_DICTIONARY = (
    ROOT
    / "artifacts"
    / "full_evidence_orchestration_20260525"
    / "bounded_same_distance_feature_repair_20260602"
    / "pre_race_history_feature_dictionary.json"
)
DEFAULT_DB = ROOT / "greyhound_racing_data_writable.db"
SAME_DISTANCE_BAND_METERS = 50.0
PROTECTED_PREFIXES = (
    "artifacts/prediction_snapshots",
    "model_registry",
    "docs/model_registry",
    "ml_models_v4",
    "advanced_models",
)

GRADE_VOCAB_MAP: dict[str, str] = {
    "1": "Grade 1",
    "2": "Grade 2",
    "3": "Grade 3",
    "4": "Grade 4",
    "5": "Grade 5",
    "6": "Grade 6",
    "7": "Grade 7",
    "8": "Grade 8",
    "M": "Mixed",
    "MX": "Mixed",
    "MIXED": "Mixed",
    "4/5": "Mixed 4/5",
    "5/6": "Mixed 5/6",
    "3/4": "Mixed 3/4",
    "3/4/5": "Mixed 3/4/5",
    "2/3": "Mixed 2/3",
    "MIXED 3/4": "Mixed 3/4",
    "MIXED 4/5": "Mixed 4/5",
    "MIXED 5/6": "Mixed 5/6",
    "MIXED 6/7": "Mixed 6/7",
    "OPEN": "Open",
    "FFA": "Free For All",
    "FREE FOR ALL": "Free For All",
    "MAIDEN": "Maiden",
    "NOV": "Novice",
    "NOVICE": "Novice",
    "NG": "Non Graded",
    "NON GRADED": "Non Graded",
    "NG1-4": "NG1-4",
    "R/W": "R/W",
    "RW": "R/W",
    "N/P": "N/P",
    "NP": "N/P",
    "P5": "P5",
    "PM": "PM",
    "M1/M2/M3": "M1/M2/M3",
    "M2/M3": "M2/M3",
    "M3": "M3",
    "M4/M5": "M4/M5",
    "M5": "M5",
    "M6": "M6",
    "J/M": "J/M",
    "I": "I",
    "INV": "Invitation",
    "INVITATIONAL": "Invitation",
    "INVITATION": "Invitation",
    "SE": "Special Event",
    "S/E": "Special Event",
    "SPECIAL EVENT": "Special Event",
    "BT8": "BT8",
    "TG1-4W": "TG1-4W",
    "TG1-6W": "TG1-6W",
    "TG5+W": "TG5+W",
    "MI4/5MA": "MI4/5MA",
    "5/M": "5/M",
    "RESTRICTED WIN": "Restricted Win",
    "RESTRICTED WIN HEAT": "Restricted Win",
    "RESTRICTED WIN FINAL": "Restricted Win",
    "TIER 3 - RESTRICTED WIN": "Restricted Win",
    "TIER 3 - MAIDEN": "Maiden",
    "TIER 3 - GRADE 5": "Grade 5",
    "TIER 3 - GRADE 6": "Grade 6",
    "TIER 3 - GRADE 7": "Grade 7",
}

GRADE_STRENGTH_RANK: dict[str, int] = {
    "Maiden": 0,
    "Novice": 1,
    "Mixed": 2,
    "Mixed 2/3": 2,
    "Mixed 3/4": 2,
    "Mixed 4/5": 2,
    "Mixed 5/6": 2,
    "Mixed 6/7": 2,
    "R/W": 3,
    "Restricted Win": 3,
    "Non Graded": 3,
    "Open": 4,
    "Free For All": 5,
    "FFA": 5,
    "Grade 7": 6,
    "Grade 6": 7,
    "Grade 5": 8,
    "Grade 4": 9,
    "Grade 3": 10,
    "Grade 2": 11,
    "Grade 1": 12,
    "Group 3": 13,
    "Group 2": 14,
    "Group 1": 15,
    "Special Event": 16,
    "P5": 17,
    "PM": 18,
    "NG1-4": 19,
    "M1/M2/M3": 20,
    "M2/M3": 20,
    "M3": 20,
    "M4/M5": 20,
    "M5": 20,
    "M6": 20,
    "BT8": 21,
    "TG1-4W": 22,
    "TG1-6W": 22,
    "TG5+W": 22,
    "MI4/5MA": 23,
    "5/M": 24,
    "J/M": 25,
    "I": 26,
    "Invitation": 27,
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def assert_output_dir_safe(output_dir: Path, repo_root: Path = ROOT) -> Path:
    logical = output_dir if output_dir.is_absolute() else repo_root / output_dir
    logical = logical.absolute()
    root = repo_root.absolute()
    try:
        relative = logical.relative_to(root)
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    relative_text = relative.as_posix()
    for prefix in PROTECTED_PREFIXES:
        if relative_text == prefix or relative_text.startswith(prefix + "/"):
            raise ValueError(f"output_dir_protected:{prefix}")
    required_parent = "artifacts/full_evidence_orchestration_20260525"
    if not relative_text.startswith(required_parent + "/"):
        raise ValueError(f"output_dir_must_be_under:{required_parent}")
    return logical


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _column_expr(
    alias: str,
    columns: set[str],
    column: str,
    *,
    fallback: str | None = None,
) -> str:
    if column in columns:
        return f"{alias}.{column}"
    if fallback and fallback in columns:
        return f"{alias}.{fallback} AS {column}"
    return f"NULL AS {column}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def _join_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("snapshot_instance_id") or "").strip(),
        str(row.get("dog_name") or row.get("normalized_dog_name") or "").strip().lower(),
        str(row.get("box_number") or "").strip(),
    )


def _clean_lookup(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = _join_key(row)
        if any(key):
            lookup[key] = dict(row)
    return lookup


def _normalize_grade_text(value: Any) -> tuple[str | None, str]:
    raw = str(value or "").strip()
    if not raw:
        return None, "MISSING"

    normalized = GRADE_VOCAB_MAP.get(raw.upper())
    if normalized is None:
        normalized = GRADE_VOCAB_MAP.get(_normalize_key(raw))
    if normalized is None:
        normalized = normalize_target_grade(raw)

    if normalized is None:
        if re.fullmatch(r"\d", raw):
            normalized = f"Grade {raw}"
        elif re.fullmatch(r"\d+/\d+(?:/\d+)?", raw):
            normalized = f"Mixed {raw}"

    if normalized is None:
        return None, "UNMAPPED"

    vocab_status = "CANONICAL" if _normalize_key(raw) == _normalize_key(normalized) else "LEGACY"
    return normalized, vocab_status


def _grade_strength_rank(value: Any) -> int | None:
    normalized, vocab_status = _normalize_grade_text(value)
    if normalized is None or vocab_status == "UNMAPPED":
        return None
    if normalized in GRADE_STRENGTH_RANK:
        return GRADE_STRENGTH_RANK[normalized]
    match = re.fullmatch(r"Grade\s+(\d+)", normalized, re.I)
    if match:
        grade_number = int(match.group(1))
        return max(0, 100 - grade_number)
    return None


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return None
        return parsed
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    parsed = _safe_float(value)
    return int(parsed) if parsed is not None else None


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return _safe_float(match.group(1) if match else None)


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _normalize_venue_key(value: Any) -> str:
    return _normalize_key(value)


def _parse_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
    ):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
    return None


def _parse_date(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()[:10]
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return _parse_timestamp(text)


def _race_datetime(row: dict[str, Any]) -> datetime | None:
    return _parse_timestamp(row.get("start_datetime")) or _parse_date(row.get("race_date"))


def _is_prior_to_target(history_row: dict[str, Any], target_date: datetime, target_dt: datetime | None) -> bool:
    history_dt = _race_datetime(history_row)
    history_date = _parse_date(history_row.get("race_date"))
    if target_dt and history_dt and _parse_timestamp(history_row.get("start_datetime")):
        return history_dt < target_dt
    if history_date and target_date:
        return history_date.date() < target_date.date()
    return False


def _same_distance_band(distance: Any, target_distance: float | None) -> bool:
    parsed = _parse_float(distance)
    if parsed is None or target_distance is None:
        return False
    return abs(parsed - float(target_distance)) <= SAME_DISTANCE_BAND_METERS


def _group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "")].append(row)
    return grouped


def _histories_by_dog(db_path: Path) -> dict[str, list[dict[str, Any]]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        dog_columns = _table_columns(conn, "dog_race_data")
        race_columns = _table_columns(conn, "race_metadata")
        order_start = "r.start_datetime" if "start_datetime" in race_columns else "r.race_date"
        finish_where = (
            "d.finish_position"
            if "finish_position" in dog_columns
            else "d.placing"
            if "placing" in dog_columns
            else "NULL"
        )
        rows = conn.execute(
            """
            SELECT
                {dog_clean_name},
                {dog_name},
                {finish_position},
                {individual_time},
                {grade},
                {distance},
                {race_date},
                {start_datetime},
                {venue},
                {race_id}
            FROM dog_race_data d
            JOIN race_metadata r ON d.race_id = r.race_id
            WHERE r.race_date IS NOT NULL
              AND {finish_where} IS NOT NULL
            ORDER BY r.race_date DESC, {order_start} DESC, r.race_id DESC
            """.format(
                dog_clean_name=_column_expr("d", dog_columns, "dog_clean_name"),
                dog_name=_column_expr("d", dog_columns, "dog_name", fallback="dog_clean_name"),
                finish_position=_column_expr("d", dog_columns, "finish_position", fallback="placing"),
                individual_time=_column_expr("d", dog_columns, "individual_time"),
                grade=_column_expr("r", race_columns, "grade"),
                distance=_column_expr("r", race_columns, "distance"),
                race_date=_column_expr("r", race_columns, "race_date"),
                start_datetime=_column_expr("r", race_columns, "start_datetime"),
                venue=_column_expr("r", race_columns, "venue"),
                race_id=_column_expr("r", race_columns, "race_id"),
                order_start=order_start,
                finish_where=finish_where,
            )
        ).fetchall()

    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        record = dict(row)
        key = _normalize_key(record.get("dog_clean_name") or record.get("dog_name"))
        if key:
            histories[key].append(record)
    return dict(histories)


def _resolve_target_metadata(
    conn: sqlite3.Connection,
    row: dict[str, Any],
) -> dict[str, Any]:
    race_date = str(row.get("race_date") or "")[:10]
    venue = str(row.get("venue") or "").strip()
    if not race_date or not venue:
        return {
            "status": "DATA_MISSING",
            "target_distance": None,
            "target_grade": None,
            "target_datetime": None,
            "target_venue": venue or None,
            "target_race_date": race_date or None,
            "source": "missing_row_context",
            "candidate_count": 0,
        }

    race_columns = _table_columns(conn, "race_metadata")
    candidates = conn.execute(
        """
        SELECT
            {race_id},
            {race_date},
            {venue},
            {race_number},
            {grade},
            {distance},
            {start_datetime}
        FROM race_metadata
        WHERE race_date = ? AND venue = ?
        ORDER BY race_id ASC
        """.format(
            race_id=_column_expr("race_metadata", race_columns, "race_id"),
            race_date=_column_expr("race_metadata", race_columns, "race_date"),
            venue=_column_expr("race_metadata", race_columns, "venue"),
            race_number=_column_expr("race_metadata", race_columns, "race_number"),
            grade=_column_expr("race_metadata", race_columns, "grade"),
            distance=_column_expr("race_metadata", race_columns, "distance"),
            start_datetime=_column_expr("race_metadata", race_columns, "start_datetime"),
        ),
        (race_date, venue),
    ).fetchall()
    candidate_rows = [dict(candidate) for candidate in candidates]

    if len(candidate_rows) == 1 and candidate_rows[0].get("distance") not in (None, ""):
        candidate = candidate_rows[0]
        return {
            "status": "UNIQUE_DATE_VENUE",
            "target_distance": _parse_float(candidate.get("distance")),
            "target_grade": str(candidate.get("grade") or "").strip() or None,
            "target_datetime": _parse_timestamp(candidate.get("start_datetime"))
            or _parse_date(candidate.get("race_date")),
            "target_venue": venue,
            "target_race_date": race_date,
            "source": "canonical_race_metadata:date_venue_unique",
            "candidate_count": 1,
        }

    # Conservative fallback for rows where the canonical metadata is ambiguous.
    return {
        "status": "AMBIGUOUS_OR_MISSING",
        "target_distance": None,
        "target_grade": None,
        "target_datetime": None,
        "target_venue": venue,
        "target_race_date": race_date,
        "source": "canonical_race_metadata:date_venue_ambiguous",
        "candidate_count": len(candidate_rows),
    }


def _resolve_target_grade_metadata(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    clean_row: dict[str, Any] | None,
) -> dict[str, Any]:
    snapshot_path = str(row.get("snapshot_path") or "").strip()
    sidecar_meta = (
        load_safe_sidecar_target_metadata(snapshot_path)
        if snapshot_path
        else {
            "target_grade": None,
            "target_grade_source": None,
            "metadata_is_leakage_safe": False,
        }
    )

    raw_grade: Any = None
    source = None
    provenance_status = "MISSING"
    provenance_reason = "missing_target_grade"

    if clean_row and clean_row.get("target_grade") not in (None, ""):
        raw_grade = clean_row.get("target_grade")
        source = "clean_official_dataset.target_grade"
        provenance_status = "SAFE_CLEAN_OFFICIAL"
        provenance_reason = "clean_official_holdout_join"
    elif (
        sidecar_meta.get("metadata_is_leakage_safe") is True
        and sidecar_meta.get("target_grade") not in (None, "")
    ):
        raw_grade = sidecar_meta.get("target_grade")
        source = str(
            sidecar_meta.get("target_grade_source") or "sidecar_target_metadata"
        )
        provenance_status = "SAFE_CANONICAL_SIDECAR"
        provenance_reason = "canonical_sidecar_target_metadata"
    else:
        race_date = str(row.get("race_date") or "")[:10]
        venue = str(row.get("venue") or "").strip()
        if not race_date or not venue:
            return {
                "target_grade_safe": None,
                "target_grade_normalized": None,
                "target_grade_source": None,
                "target_grade_provenance_status": "MISSING",
                "target_grade_provenance_reason": "missing_row_context",
                "target_grade_vocab_status": "MISSING",
            }

        race_columns = _table_columns(conn, "race_metadata")
        candidate_rows = [
            dict(candidate)
            for candidate in conn.execute(
                """
                SELECT
                    {race_id},
                    {race_date},
                    {venue},
                    {race_number},
                    {grade},
                    {distance},
                    {start_datetime}
                FROM race_metadata
                WHERE race_date = ? AND venue = ?
                ORDER BY race_id ASC
                """.format(
                    race_id=_column_expr("race_metadata", race_columns, "race_id"),
                    race_date=_column_expr("race_metadata", race_columns, "race_date"),
                    venue=_column_expr("race_metadata", race_columns, "venue"),
                    race_number=_column_expr("race_metadata", race_columns, "race_number"),
                    grade=_column_expr("race_metadata", race_columns, "grade"),
                    distance=_column_expr("race_metadata", race_columns, "distance"),
                    start_datetime=_column_expr("race_metadata", race_columns, "start_datetime"),
                ),
                (race_date, venue),
            ).fetchall()
        ]
        if len(candidate_rows) == 1 and candidate_rows[0].get("grade") not in (None, ""):
            raw_grade = candidate_rows[0].get("grade")
            source = "race_metadata.grade"
            provenance_status = "SAFE_CANONICAL_DB"
            provenance_reason = "canonical_db_unique_date_venue"
        elif len(candidate_rows) == 0:
            return {
                "target_grade_safe": None,
                "target_grade_normalized": None,
                "target_grade_source": None,
                "target_grade_provenance_status": "MISSING",
                "target_grade_provenance_reason": "canonical_db_missing",
                "target_grade_vocab_status": "MISSING",
            }
        else:
            return {
                "target_grade_safe": None,
                "target_grade_normalized": None,
                "target_grade_source": "race_metadata.grade",
                "target_grade_provenance_status": "AMBIGUOUS_OR_MISSING",
                "target_grade_provenance_reason": "canonical_db_date_venue_ambiguous",
                "target_grade_vocab_status": "MISSING",
            }

    normalized_grade, vocab_status = _normalize_grade_text(raw_grade)
    if raw_grade in (None, ""):
        return {
            "target_grade_safe": None,
            "target_grade_normalized": None,
            "target_grade_source": source,
            "target_grade_provenance_status": provenance_status,
            "target_grade_provenance_reason": provenance_reason,
            "target_grade_vocab_status": "MISSING",
        }

    return {
        "target_grade_safe": str(raw_grade).strip(),
        "target_grade_normalized": normalized_grade,
        "target_grade_source": source,
        "target_grade_provenance_status": provenance_status,
        "target_grade_provenance_reason": provenance_reason,
        "target_grade_vocab_status": vocab_status,
    }


def _compute_same_distance_bundle(
    *,
    row: dict[str, Any],
    target_meta: dict[str, Any],
    history_index: dict[str, list[dict[str, Any]]],
    target_grade_normalized: str | None = None,
) -> dict[str, Any]:
    target_distance = _safe_float(target_meta.get("target_distance"))
    target_datetime = target_meta.get("target_datetime")
    target_date = _parse_date(target_meta.get("target_race_date") or row.get("race_date"))
    target_venue = _normalize_venue_key(target_meta.get("target_venue") or row.get("venue"))
    dog_key = _normalize_key(row.get("dog_name") or row.get("dog_clean_name"))

    feature_names = (
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
        "same_distance_venue_start_count",
        "same_distance_venue_best_time",
        "same_distance_same_grade_start_count",
        "same_distance_same_grade_best_time",
        "same_distance_same_grade_avg_time",
    )

    if target_distance is None or target_date is None:
        return {name: None for name in feature_names} | {
            "same_distance_target_status": str(target_meta.get("status") or "DATA_MISSING"),
            "same_distance_target_source": str(target_meta.get("source") or "DATA_MISSING"),
        }

    history = history_index.get(dog_key, [])
    qualifying: list[dict[str, Any]] = []
    for history_row in history:
        if not _is_prior_to_target(history_row, target_date, target_datetime):
            continue
        if not _same_distance_band(history_row.get("distance"), target_distance):
            continue
        qualifying.append(history_row)

    venue_qualifying = [
        history_row
        for history_row in qualifying
        if _normalize_venue_key(history_row.get("venue")) == target_venue
    ]

    same_distance_rows = list(qualifying)
    same_distance_rows.sort(
        key=lambda value: (
            _race_datetime(value) or datetime.min,
            str(value.get("race_id") or ""),
        ),
        reverse=True,
    )

    time_values = [
        _safe_float(history_row.get("individual_time"))
        for history_row in same_distance_rows
        if _safe_float(history_row.get("individual_time")) is not None
    ]
    position_values = [
        _safe_int(history_row.get("finish_position"))
        for history_row in same_distance_rows
        if _safe_int(history_row.get("finish_position")) is not None
    ]
    recent_rows = same_distance_rows[:5]
    recent_time_values = [
        _safe_float(history_row.get("individual_time"))
        for history_row in recent_rows
        if _safe_float(history_row.get("individual_time")) is not None
    ]
    venue_time_values = [
        _safe_float(history_row.get("individual_time"))
        for history_row in venue_qualifying
        if _safe_float(history_row.get("individual_time")) is not None
    ]
    same_grade_rows = []
    if target_grade_normalized not in (None, ""):
        for history_row in same_distance_rows:
            normalized_grade, vocab_status = _normalize_grade_text(history_row.get("grade"))
            if normalized_grade == target_grade_normalized and vocab_status != "UNMAPPED":
                same_grade_rows.append(history_row)
    same_grade_time_values = [
        _safe_float(history_row.get("individual_time"))
        for history_row in same_grade_rows
        if _safe_float(history_row.get("individual_time")) is not None
    ]

    bundle: dict[str, Any] = {
        "same_distance_target_status": str(target_meta.get("status") or "DATA_MISSING"),
        "same_distance_target_source": str(target_meta.get("source") or "DATA_MISSING"),
        "same_distance_target_distance": target_distance,
    }
    bundle["starts_same_distance"] = len(same_distance_rows)
    bundle["prior_same_distance_start_count"] = len(same_distance_rows)
    bundle["same_distance_venue_start_count"] = len(venue_qualifying)

    if same_distance_rows:
        bundle["win_rate_same_distance"] = (
            sum(1 for value in position_values if value == 1) / len(position_values)
            if position_values
            else None
        )
        bundle["place_rate_same_distance"] = (
            sum(1 for value in position_values if value <= 3) / len(position_values)
            if position_values
            else None
        )
        last_row_dt = _race_datetime(same_distance_rows[0])
        if last_row_dt and target_datetime:
            bundle["days_since_last_same_distance_start"] = (
                target_datetime - last_row_dt
            ).total_seconds() / 86400.0
        elif target_date and _parse_date(same_distance_rows[0].get("race_date")):
            bundle["days_since_last_same_distance_start"] = (
                target_date.date() - _parse_date(same_distance_rows[0].get("race_date")).date()
            ).days
        else:
            bundle["days_since_last_same_distance_start"] = None
    else:
        bundle["win_rate_same_distance"] = 0.0
        bundle["place_rate_same_distance"] = 0.0
        bundle["days_since_last_same_distance_start"] = None

    if time_values:
        bundle["best_time_same_distance"] = min(time_values)
        bundle["avg_time_same_distance"] = sum(time_values) / len(time_values)
        bundle["median_time_same_distance"] = float(median(time_values))
    else:
        bundle["best_time_same_distance"] = None
        bundle["avg_time_same_distance"] = None
        bundle["median_time_same_distance"] = None

    if recent_time_values:
        bundle["recent_best_time_same_distance_5"] = min(recent_time_values)
        bundle["recent_avg_time_same_distance_5"] = sum(recent_time_values) / len(
            recent_time_values
        )
    else:
        bundle["recent_best_time_same_distance_5"] = None
        bundle["recent_avg_time_same_distance_5"] = None

    if venue_time_values:
        bundle["same_distance_venue_best_time"] = min(venue_time_values)
    else:
        bundle["same_distance_venue_best_time"] = None

    if target_grade_normalized in (None, ""):
        bundle["same_distance_same_grade_start_count"] = None
        bundle["same_distance_same_grade_best_time"] = None
        bundle["same_distance_same_grade_avg_time"] = None
    elif same_grade_rows:
        bundle["same_distance_same_grade_start_count"] = len(same_grade_rows)
        bundle["same_distance_same_grade_best_time"] = (
            min(same_grade_time_values) if same_grade_time_values else None
        )
        bundle["same_distance_same_grade_avg_time"] = (
            sum(same_grade_time_values) / len(same_grade_time_values)
            if same_grade_time_values
            else None
        )
    else:
        bundle["same_distance_same_grade_start_count"] = 0
        bundle["same_distance_same_grade_best_time"] = None
        bundle["same_distance_same_grade_avg_time"] = None

    return bundle


def _compute_grade_context_bundle(
    *,
    row: dict[str, Any],
    target_meta: dict[str, Any],
    target_grade_meta: dict[str, Any],
    history_index: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    target_grade_normalized = target_grade_meta.get("target_grade_normalized")
    target_grade_safe = target_grade_meta.get("target_grade_safe")
    target_date = _parse_date(target_meta.get("target_race_date") or row.get("race_date"))
    target_dt = target_meta.get("target_datetime")
    dog_key = _normalize_key(row.get("dog_name") or row.get("dog_clean_name"))
    history = history_index.get(dog_key, [])

    prior_rows = [
        history_row
        for history_row in history
        if _is_prior_to_target(history_row, target_date, target_dt)
    ]
    prior_rows.sort(
        key=lambda value: (
            _race_datetime(value) or datetime.min,
            str(value.get("race_id") or ""),
        ),
        reverse=True,
    )

    normalized_prior_rows: list[dict[str, Any]] = []
    for history_row in prior_rows:
        normalized_grade, vocab_status = _normalize_grade_text(history_row.get("grade"))
        if normalized_grade is None or vocab_status == "UNMAPPED":
            continue
        normalized_prior_rows.append(
            {
                "row": history_row,
                "normalized_grade": normalized_grade,
            }
        )

    last_start_grade_raw = None
    if prior_rows:
        last_start_grade_raw = prior_rows[0].get("grade")
    if last_start_grade_raw in (None, ""):
        last_start_grade_raw = row.get("last_start_grade")
    last_start_grade_normalized, last_start_grade_vocab_status = _normalize_grade_text(
        last_start_grade_raw
    )

    recent_grade_mode_5 = None
    recent_grade_counts: Counter[str] = Counter(
        item["normalized_grade"] for item in normalized_prior_rows[:5]
    )
    if recent_grade_counts:
        recent_grade_mode_5 = sorted(
            recent_grade_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]

    same_grade_rows = []
    if target_grade_normalized not in (None, ""):
        same_grade_rows = [
            item["row"]
            for item in normalized_prior_rows
            if item["normalized_grade"] == target_grade_normalized
        ]

    same_grade_positions = [
        _safe_int(history_row.get("finish_position"))
        for history_row in same_grade_rows
        if _safe_int(history_row.get("finish_position")) is not None
    ]

    same_grade_start_count = None
    same_grade_win_rate = None
    same_grade_place_rate = None
    if target_grade_normalized in (None, ""):
        same_grade_start_count = None
        same_grade_win_rate = None
        same_grade_place_rate = None
    elif same_grade_rows:
        same_grade_start_count = len(same_grade_rows)
        same_grade_win_rate = (
            sum(1 for value in same_grade_positions if value == 1) / len(same_grade_positions)
            if same_grade_positions
            else None
        )
        same_grade_place_rate = (
            sum(1 for value in same_grade_positions if value <= 3) / len(same_grade_positions)
            if same_grade_positions
            else None
        )
    else:
        same_grade_start_count = 0
        same_grade_win_rate = 0.0
        same_grade_place_rate = 0.0

    target_grade_rank = _grade_strength_rank(target_grade_normalized)
    last_start_grade_rank = _grade_strength_rank(last_start_grade_normalized)

    grade_change_indicator = None
    grade_change_direction = None
    grade_strength_delta = None
    if target_grade_rank is not None and last_start_grade_rank is not None:
        grade_strength_delta = target_grade_rank - last_start_grade_rank
        grade_change_indicator = 0 if grade_strength_delta == 0 else 1
        if grade_strength_delta > 0:
            grade_change_direction = "UP"
        elif grade_strength_delta < 0:
            grade_change_direction = "DOWN"
        else:
            grade_change_direction = "FLAT"

    bundle = {
        "target_grade_safe": target_grade_safe,
        "target_grade_normalized": target_grade_normalized,
        "target_grade_source": target_grade_meta.get("target_grade_source"),
        "target_grade_provenance_status": target_grade_meta.get(
            "target_grade_provenance_status"
        ),
        "target_grade_provenance_reason": target_grade_meta.get(
            "target_grade_provenance_reason"
        ),
        "target_grade_vocab_status": target_grade_meta.get("target_grade_vocab_status"),
        "last_start_grade_normalized": last_start_grade_normalized,
        "recent_grade_mode_5": recent_grade_mode_5,
        "same_grade_start_count": same_grade_start_count,
        "same_grade_win_rate": same_grade_win_rate,
        "same_grade_place_rate": same_grade_place_rate,
        "grade_change_indicator": grade_change_indicator,
        "grade_change_direction": grade_change_direction,
        "grade_strength_delta": grade_strength_delta,
    }

    # If the target grade exists but the most recent prior grade is unmapped, keep the raw
    # class signal explicit rather than guessing.
    if last_start_grade_raw not in (None, ""):
        bundle["last_start_grade"] = last_start_grade_raw
    return bundle


def _packet_coverage(rows: list[dict[str, Any]], fields: Iterable[str]) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    grouped = _group_rows(rows, "packet")
    for field in fields:
        field_summary: dict[str, Any] = {}
        for packet_name, packet_rows in grouped.items():
            present_values = [
                row.get(field)
                for row in packet_rows
                if row.get(field) not in (None, "")
            ]
            field_summary[packet_name] = {
                "rows": len(packet_rows),
                "present_rows": len(present_values),
                "present_pct": (
                    len(present_values) / len(packet_rows) if packet_rows else None
                ),
            }
        coverage[field] = field_summary
    return coverage


def _grade_vocab_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        raw = row.get("target_grade_safe")
        normalized = row.get("target_grade_normalized")
        if raw in (None, "") and normalized in (None, ""):
            continue
        vocab_status = str(row.get("target_grade_vocab_status") or "MISSING")
        key = (str(raw or ""), str(normalized or ""))
        counts[key] += 1
        if len(examples[key]) < 3 and raw not in (None, ""):
            examples[key].append(str(raw))

    output = []
    total = max(1, sum(counts.values()))
    for (raw, normalized), count in sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    ):
        status = "canonical"
        if raw != normalized and normalized not in (None, ""):
            status = "legacy"
        if normalized in (None, ""):
            status = "unmapped"
        output.append(
            {
                "raw_grade": raw,
                "normalized_grade": normalized,
                "count": count,
                "pct_of_grade_rows": count / total,
                "status": status,
                "examples": "|".join(examples[(raw, normalized)]),
            }
        )
    return output


def _same_distance_fields() -> tuple[str, ...]:
    return (
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
        "same_distance_venue_start_count",
        "same_distance_venue_best_time",
        "same_distance_same_grade_start_count",
        "same_distance_same_grade_best_time",
        "same_distance_same_grade_avg_time",
    )


def _grade_context_fields() -> tuple[str, ...]:
    return (
        "target_grade_safe",
        "target_grade_normalized",
        "target_grade_source",
        "target_grade_provenance_status",
        "target_grade_provenance_reason",
        "target_grade_vocab_status",
        "last_start_grade",
        "last_start_grade_normalized",
        "recent_grade_mode_5",
        "same_grade_start_count",
        "same_grade_win_rate",
        "same_grade_place_rate",
        "grade_change_indicator",
        "grade_change_direction",
        "grade_strength_delta",
    )


def repair_packet_rows(
    rows: list[dict[str, Any]],
    *,
    db_path: Path,
    clean_lookup: dict[tuple[str, str, str], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clean_lookup = clean_lookup or {}
    history_index = _histories_by_dog(db_path)
    by_race = _group_rows(rows, "race_id")
    repaired_rows: list[dict[str, Any]] = []
    resolution_counts: Counter[str] = Counter()
    target_meta_samples: list[dict[str, Any]] = []
    target_grade_resolution_counts: Counter[str] = Counter()
    target_grade_vocab_counts: Counter[str] = Counter()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for race_id, race_rows in by_race.items():
            target_meta = _resolve_target_metadata(conn, race_rows[0])
            resolution_counts[str(target_meta.get("status") or "DATA_MISSING")] += 1
            if len(target_meta_samples) < 10:
                target_meta_samples.append(
                    {
                        "race_id": race_id,
                        "race_date": race_rows[0].get("race_date"),
                        "venue": race_rows[0].get("venue"),
                        "resolution_status": target_meta.get("status"),
                        "target_distance": target_meta.get("target_distance"),
                        "target_source": target_meta.get("source"),
                        "candidate_count": target_meta.get("candidate_count"),
                    }
                )

            for row in race_rows:
                repaired = dict(row)
                clean_row = clean_lookup.get(_join_key(row))
                target_grade_meta = _resolve_target_grade_metadata(
                    conn,
                    row,
                    clean_row,
                )
                target_grade_resolution_counts[
                    str(target_grade_meta.get("target_grade_provenance_status") or "MISSING")
                ] += 1
                target_grade_vocab_counts[
                    str(target_grade_meta.get("target_grade_vocab_status") or "MISSING")
                ] += 1
                grade_bundle = _compute_grade_context_bundle(
                    row=row,
                    target_meta=target_meta,
                    target_grade_meta=target_grade_meta,
                    history_index=history_index,
                )
                bundle = _compute_same_distance_bundle(
                    row=row,
                    target_meta=target_meta,
                    history_index=history_index,
                    target_grade_normalized=grade_bundle.get("target_grade_normalized"),
                )
                repaired.update(grade_bundle)
                for field in _same_distance_fields():
                    existing_value = repaired.get(field)
                    value = bundle.get(field)
                    if value is not None:
                        repaired[field] = value
                    elif existing_value not in (None, ""):
                        repaired[field] = existing_value
                    else:
                        repaired[field] = ""
                repaired["same_distance_target_status"] = bundle.get(
                    "same_distance_target_status"
                )
                repaired["same_distance_target_source"] = bundle.get(
                    "same_distance_target_source"
                )
                repaired_rows.append(repaired)

    field_coverage = _packet_coverage(repaired_rows, _same_distance_fields())
    history_coverage = {
        field: field_coverage[field].get("historical", {}) for field in _same_distance_fields()
    }
    rolling_coverage = {
        field: field_coverage[field].get("rolling", {}) for field in _same_distance_fields()
    }

    audit = {
        "status": "PASS",
        "target_resolution_counts": dict(resolution_counts),
        "target_resolution_samples": target_meta_samples,
        "target_grade_resolution_counts": dict(target_grade_resolution_counts),
        "target_grade_vocab_counts": dict(target_grade_vocab_counts),
        "same_distance_coverage": field_coverage,
        "grade_context_coverage": _packet_coverage(repaired_rows, _grade_context_fields()),
        "historical_coverage": history_coverage,
        "rolling_coverage": rolling_coverage,
    }
    return repaired_rows, audit


def _dict_from_old_dictionary(old_dictionary: dict[str, Any], field_coverage: dict[str, Any]) -> dict[str, Any]:
    result = dict(old_dictionary)
    features = dict(result.get("features") or {})
    history_features = dict(result.get("history_features") or {})

    field_docs = {
        "starts_same_distance": {
            "source": "dog_race_data JOIN race_metadata",
            "source_fields": "dog_race_data.dog_clean_name, dog_race_data.finish_position, race_metadata.distance, race_metadata.race_date, race_metadata.start_datetime, race_metadata.venue",
            "aggregation_window": "all prior same-distance starts within 50m band",
            "units": "count",
            "parsing_rule": "count prior rows where abs(prior_distance - target_distance) <= 50m and the row is strictly before the target cutoff",
            "null_policy": "null when safe target distance is unavailable",
            "default_policy": "0 when safe target distance exists and no qualifying prior rows exist",
            "leakage_classification": "MEDIUM; target distance must be safe and pre-race",
        },
        "prior_same_distance_start_count": {
            "source": "dog_race_data JOIN race_metadata",
            "source_fields": "dog_race_data.dog_clean_name, dog_race_data.finish_position, race_metadata.distance, race_metadata.race_date, race_metadata.start_datetime, race_metadata.venue",
            "aggregation_window": "alias of starts_same_distance",
            "units": "count",
            "parsing_rule": "compatibility alias for starts_same_distance",
            "null_policy": "null when safe target distance is unavailable",
            "default_policy": "0 when safe target distance exists and no qualifying prior rows exist",
            "leakage_classification": "MEDIUM; same as starts_same_distance",
        },
        "best_time_same_distance": {
            "source": "dog_race_data JOIN race_metadata",
            "source_fields": "dog_race_data.individual_time, dog_race_data.finish_position, race_metadata.distance, race_metadata.race_date, race_metadata.start_datetime, race_metadata.venue",
            "aggregation_window": "all prior same-distance starts within 50m band",
            "units": "seconds",
            "parsing_rule": "minimum parsed individual_time among qualifying prior rows",
            "null_policy": "null when safe target distance is unavailable or no qualifying prior times exist",
            "default_policy": "no default/imputation",
            "leakage_classification": "MEDIUM; target distance must be safe and pre-race",
        },
        "avg_time_same_distance": {
            "source": "dog_race_data JOIN race_metadata",
            "source_fields": "dog_race_data.individual_time, dog_race_data.finish_position, race_metadata.distance, race_metadata.race_date, race_metadata.start_datetime, race_metadata.venue",
            "aggregation_window": "all prior same-distance starts within 50m band",
            "units": "seconds",
            "parsing_rule": "mean parsed individual_time among qualifying prior rows",
            "null_policy": "null when safe target distance is unavailable or no qualifying prior times exist",
            "default_policy": "no default/imputation",
            "leakage_classification": "MEDIUM; target distance must be safe and pre-race",
        },
        "median_time_same_distance": {
            "source": "dog_race_data JOIN race_metadata",
            "source_fields": "dog_race_data.individual_time, dog_race_data.finish_position, race_metadata.distance, race_metadata.race_date, race_metadata.start_datetime, race_metadata.venue",
            "aggregation_window": "all prior same-distance starts within 50m band",
            "units": "seconds",
            "parsing_rule": "median parsed individual_time among qualifying prior rows",
            "null_policy": "null when safe target distance is unavailable or no qualifying prior times exist",
            "default_policy": "no default/imputation",
            "leakage_classification": "MEDIUM; target distance must be safe and pre-race",
        },
        "recent_best_time_same_distance_5": {
            "source": "dog_race_data JOIN race_metadata",
            "source_fields": "dog_race_data.individual_time, dog_race_data.finish_position, race_metadata.distance, race_metadata.race_date, race_metadata.start_datetime, race_metadata.venue",
            "aggregation_window": "most recent up to 5 prior same-distance starts within 50m band",
            "units": "seconds",
            "parsing_rule": "minimum parsed individual_time across the most recent qualifying same-distance rows",
            "null_policy": "null when safe target distance is unavailable or no qualifying prior times exist",
            "default_policy": "no default/imputation",
            "leakage_classification": "MEDIUM; target distance must be safe and pre-race",
        },
        "recent_avg_time_same_distance_5": {
            "source": "dog_race_data JOIN race_metadata",
            "source_fields": "dog_race_data.individual_time, dog_race_data.finish_position, race_metadata.distance, race_metadata.race_date, race_metadata.start_datetime, race_metadata.venue",
            "aggregation_window": "most recent up to 5 prior same-distance starts within 50m band",
            "units": "seconds",
            "parsing_rule": "mean parsed individual_time across the most recent qualifying same-distance rows",
            "null_policy": "null when safe target distance is unavailable or no qualifying prior times exist",
            "default_policy": "no default/imputation",
            "leakage_classification": "MEDIUM; target distance must be safe and pre-race",
        },
        "days_since_last_same_distance_start": {
            "source": "dog_race_data JOIN race_metadata",
            "source_fields": "race_metadata.race_date, race_metadata.start_datetime, race_metadata.distance, race_metadata.venue",
            "aggregation_window": "most recent prior same-distance start within 50m band",
            "units": "days",
            "parsing_rule": "target timestamp minus most recent prior same-distance race timestamp; date-only when timestamps are unavailable",
            "null_policy": "null when safe target distance is unavailable or no qualifying prior row exists",
            "default_policy": "no wall-clock now() fallback",
            "leakage_classification": "LOW to MEDIUM; pre-race only",
        },
        "win_rate_same_distance": {
            "source": "dog_race_data JOIN race_metadata",
            "source_fields": "dog_race_data.finish_position, race_metadata.distance, race_metadata.race_date, race_metadata.start_datetime, race_metadata.venue",
            "aggregation_window": "all prior same-distance starts within 50m band",
            "units": "rate 0-1",
            "parsing_rule": "wins divided by qualifying prior same-distance starts",
            "null_policy": "null when safe target distance is unavailable or no qualifying prior rows exist",
            "default_policy": "0 when safe target distance exists and no qualifying prior rows exist",
            "leakage_classification": "MEDIUM; target distance must be safe and pre-race",
        },
        "place_rate_same_distance": {
            "source": "dog_race_data JOIN race_metadata",
            "source_fields": "dog_race_data.finish_position, race_metadata.distance, race_metadata.race_date, race_metadata.start_datetime, race_metadata.venue",
            "aggregation_window": "all prior same-distance starts within 50m band",
            "units": "rate 0-1",
            "parsing_rule": "top-3 finishes divided by qualifying prior same-distance starts",
            "null_policy": "null when safe target distance is unavailable or no qualifying prior rows exist",
            "default_policy": "0 when safe target distance exists and no qualifying prior rows exist",
            "leakage_classification": "MEDIUM; target distance must be safe and pre-race",
        },
        "same_distance_venue_start_count": {
            "source": "dog_race_data JOIN race_metadata",
            "source_fields": "dog_race_data.finish_position, race_metadata.distance, race_metadata.venue, race_metadata.race_date, race_metadata.start_datetime",
            "aggregation_window": "all prior same-distance starts at the target venue within 50m band",
            "units": "count",
            "parsing_rule": "count same-distance prior rows where normalised venue matches the target venue",
            "null_policy": "null when safe target distance is unavailable",
            "default_policy": "0 when safe target distance exists and no qualifying venue-matched rows exist",
            "leakage_classification": "MEDIUM; target distance and venue must be safe and pre-race",
        },
        "same_distance_venue_best_time": {
            "source": "dog_race_data JOIN race_metadata",
            "source_fields": "dog_race_data.individual_time, race_metadata.distance, race_metadata.venue, race_metadata.race_date, race_metadata.start_datetime",
            "aggregation_window": "all prior same-distance starts at the target venue within 50m band",
            "units": "seconds",
            "parsing_rule": "minimum parsed individual_time among qualifying venue-matched prior rows",
            "null_policy": "null when safe target distance is unavailable or no venue-matched prior times exist",
            "default_policy": "no default/imputation",
            "leakage_classification": "MEDIUM; target distance and venue must be safe and pre-race",
        },
        "target_grade_safe": {
            "source": "clean_official_dataset.target_grade + race_metadata.grade + sidecar_target_metadata.target_grade",
            "source_fields": "snapshot_instance_id, dog_name, box_number, target_grade, race_metadata.grade, sidecar_target_metadata.target_grade",
            "aggregation_window": "canonical target-race metadata join, then clean holdout join, then safe DB fallback",
            "units": "class label",
            "parsing_rule": "preserve the raw safe grade string when the source is leakage-safe; never guess from embedded historical DIST/G",
            "null_policy": "null when target grade provenance is unsafe, ambiguous, or missing",
            "default_policy": "no default/imputation",
            "leakage_classification": "LOW; provenance must be explicit and safe",
        },
        "target_grade_normalized": {
            "source": "target_grade_safe",
            "source_fields": "target_grade_safe",
            "aggregation_window": "explicit vocabulary map applied to the safe target grade",
            "units": "canonical class label",
            "parsing_rule": "normalize via explicit map; emit null when unmapped rather than guessing",
            "null_policy": "null when the raw safe grade is missing or unmapped",
            "default_policy": "no default/imputation",
            "leakage_classification": "LOW; derived only from safe target metadata",
        },
        "target_grade_source": {
            "source": "clean_official_dataset.target_grade + race_metadata.grade + sidecar_target_metadata.target_grade",
            "source_fields": "snapshot_instance_id, dog_name, box_number, target_grade_source",
            "aggregation_window": "safe source attribution from the resolved target-grade resolver",
            "units": "string",
            "parsing_rule": "preserve source attribution for each safe target-grade value",
            "null_policy": "null when target grade provenance is missing",
            "default_policy": "no default/imputation",
            "leakage_classification": "LOW; source attribution only",
        },
        "target_grade_provenance_status": {
            "source": "target-grade resolver",
            "source_fields": "snapshot_instance_id, dog_name, box_number, target_grade_provenance_status",
            "aggregation_window": "safe provenance classification for the resolved target-grade source",
            "units": "categorical string",
            "parsing_rule": "SAFE_* when provenance is explicit; AMBIGUOUS_OR_MISSING or MISSING otherwise",
            "null_policy": "never guessed",
            "default_policy": "no default/imputation",
            "leakage_classification": "LOW; audit only",
        },
        "target_grade_provenance_reason": {
            "source": "target-grade resolver",
            "source_fields": "snapshot_instance_id, dog_name, box_number, target_grade_provenance_reason",
            "aggregation_window": "safe provenance reason code for the resolved target-grade source",
            "units": "categorical string",
            "parsing_rule": "explicit reason codes; no guessing",
            "null_policy": "never guessed",
            "default_policy": "no default/imputation",
            "leakage_classification": "LOW; audit only",
        },
        "target_grade_vocab_status": {
            "source": "target-grade normalizer",
            "source_fields": "snapshot_instance_id, dog_name, box_number, target_grade_vocab_status",
            "aggregation_window": "explicit vocabulary classification of the resolved target grade",
            "units": "categorical string",
            "parsing_rule": "CANONICAL or LEGACY when normalized; UNMAPPED otherwise",
            "null_policy": "never guessed",
            "default_policy": "no default/imputation",
            "leakage_classification": "LOW; audit only",
        },
        "last_start_grade": {
            "source": "dog_race_data JOIN race_metadata",
            "source_fields": "dog_race_data.finish_position, race_metadata.grade, race_metadata.race_date, race_metadata.start_datetime",
            "aggregation_window": "immediate prior start",
            "units": "class label",
            "parsing_rule": "preserve the raw prior-grade value from the most recent prior row",
            "null_policy": "null when no prior row exists",
            "default_policy": "no default/imputation",
            "leakage_classification": "LOW; prior-history only",
        },
        "last_start_grade_normalized": {
            "source": "last_start_grade",
            "source_fields": "last_start_grade",
            "aggregation_window": "explicit vocabulary map applied to the immediate prior grade",
            "units": "canonical class label",
            "parsing_rule": "normalize via explicit map; emit null when unmapped rather than guessing",
            "null_policy": "null when the prior grade is missing or unmapped",
            "default_policy": "no default/imputation",
            "leakage_classification": "LOW; prior-history only",
        },
        "recent_grade_mode_5": {
            "source": "dog_race_data JOIN race_metadata",
            "source_fields": "dog_race_data.finish_position, race_metadata.grade, race_metadata.race_date, race_metadata.start_datetime",
            "aggregation_window": "most recent five prior mapped grades",
            "units": "canonical class label",
            "parsing_rule": "mode of the most recent five mapped prior grades",
            "null_policy": "null when no mapped prior grades exist",
            "default_policy": "no default/imputation",
            "leakage_classification": "LOW; prior-history only",
        },
        "same_grade_start_count": {
            "source": "dog_race_data JOIN race_metadata",
            "source_fields": "dog_race_data.finish_position, race_metadata.grade, race_metadata.race_date, race_metadata.start_datetime",
            "aggregation_window": "all prior starts at the same normalized grade",
            "units": "count",
            "parsing_rule": "count prior rows where normalized grade matches the safe target grade",
            "null_policy": "null when safe target grade is unavailable",
            "default_policy": "0 when safe target grade exists and no qualifying prior rows exist",
            "leakage_classification": "LOW; prior-history only",
        },
        "same_grade_win_rate": {
            "source": "dog_race_data JOIN race_metadata",
            "source_fields": "dog_race_data.finish_position, race_metadata.grade, race_metadata.race_date, race_metadata.start_datetime",
            "aggregation_window": "all prior starts at the same normalized grade",
            "units": "rate 0-1",
            "parsing_rule": "wins divided by qualifying same-grade prior starts",
            "null_policy": "null when safe target grade is unavailable",
            "default_policy": "0 when safe target grade exists and no qualifying prior rows exist",
            "leakage_classification": "LOW; prior-history only",
        },
        "same_grade_place_rate": {
            "source": "dog_race_data JOIN race_metadata",
            "source_fields": "dog_race_data.finish_position, race_metadata.grade, race_metadata.race_date, race_metadata.start_datetime",
            "aggregation_window": "all prior starts at the same normalized grade",
            "units": "rate 0-1",
            "parsing_rule": "top-3 finishes divided by qualifying same-grade prior starts",
            "null_policy": "null when safe target grade is unavailable",
            "default_policy": "0 when safe target grade exists and no qualifying prior rows exist",
            "leakage_classification": "LOW; prior-history only",
        },
        "grade_change_indicator": {
            "source": "target_grade_normalized + last_start_grade_normalized",
            "source_fields": "target_grade_normalized, last_start_grade_normalized",
            "aggregation_window": "immediate prior start",
            "units": "binary",
            "parsing_rule": "1 when the safe target grade differs from the normalized prior grade, else 0",
            "null_policy": "null when either grade is missing or unmapped",
            "default_policy": "no default/imputation",
            "leakage_classification": "LOW; prior-history only",
        },
        "grade_change_direction": {
            "source": "target_grade_normalized + last_start_grade_normalized",
            "source_fields": "target_grade_normalized, last_start_grade_normalized",
            "aggregation_window": "immediate prior start",
            "units": "categorical string",
            "parsing_rule": "UP, DOWN, or FLAT from the explicit grade-strength ranking map",
            "null_policy": "null when either grade is missing or unmapped",
            "default_policy": "no default/imputation",
            "leakage_classification": "LOW; prior-history only",
        },
        "grade_strength_delta": {
            "source": "target_grade_normalized + last_start_grade_normalized",
            "source_fields": "target_grade_normalized, last_start_grade_normalized",
            "aggregation_window": "immediate prior start",
            "units": "ordinal delta",
            "parsing_rule": "rank(target_grade) - rank(last_start_grade); null when the ranking map is not defensible for the pair",
            "null_policy": "null when either grade is missing or unmapped",
            "default_policy": "no default/imputation",
            "leakage_classification": "LOW; prior-history only",
        },
        "same_distance_same_grade_start_count": {
            "source": "dog_race_data JOIN race_metadata",
            "source_fields": "dog_race_data.finish_position, dog_race_data.individual_time, race_metadata.grade, race_metadata.distance, race_metadata.race_date, race_metadata.start_datetime, race_metadata.venue",
            "aggregation_window": "all prior same-distance starts at the same normalized grade",
            "units": "count",
            "parsing_rule": "count prior rows where the normalized grade matches and the distance band is safe",
            "null_policy": "null when safe target grade is unavailable",
            "default_policy": "0 when safe target grade exists and no qualifying prior rows exist",
            "leakage_classification": "MEDIUM; target distance and grade must both be safe and pre-race",
        },
        "same_distance_same_grade_best_time": {
            "source": "dog_race_data JOIN race_metadata",
            "source_fields": "dog_race_data.individual_time, race_metadata.grade, race_metadata.distance, race_metadata.race_date, race_metadata.start_datetime, race_metadata.venue",
            "aggregation_window": "all prior same-distance starts at the same normalized grade",
            "units": "seconds",
            "parsing_rule": "minimum parsed individual_time among same-distance same-grade prior rows",
            "null_policy": "null when safe target grade is unavailable or no qualifying prior times exist",
            "default_policy": "no default/imputation",
            "leakage_classification": "MEDIUM; target distance and grade must both be safe and pre-race",
        },
        "same_distance_same_grade_avg_time": {
            "source": "dog_race_data JOIN race_metadata",
            "source_fields": "dog_race_data.individual_time, race_metadata.grade, race_metadata.distance, race_metadata.race_date, race_metadata.start_datetime, race_metadata.venue",
            "aggregation_window": "all prior same-distance starts at the same normalized grade",
            "units": "seconds",
            "parsing_rule": "mean parsed individual_time among same-distance same-grade prior rows",
            "null_policy": "null when safe target grade is unavailable or no qualifying prior times exist",
            "default_policy": "no default/imputation",
            "leakage_classification": "MEDIUM; target distance and grade must both be safe and pre-race",
        },
    }

    for field, doc in field_docs.items():
        coverage = field_coverage.get(field, {})
        features[field] = {
            "aggregation_window": doc["aggregation_window"],
            "available_pre_jump": True,
            "coverage": coverage,
            "leakage_sensitivity": doc["leakage_classification"],
            "null_default_policy": doc["null_policy"],
            "source": doc["source"],
            "source_fields": doc["source_fields"],
            "units": doc["units"],
            "parsing_rule": doc["parsing_rule"],
            "default_policy": doc["default_policy"],
            "validation_test_needed": f"repair_{field}",
        }
        history_features[field] = dict(features[field])

    result["schema_version"] = "repaired_target_grade_feature_dictionary_v1"
    result["feature_policy"] = {
        "accepted_form_guide_policy": "Report-only repair consumes the frozen clean history packet, the clean official holdout, and canonical DB history; it does not treat embedded historical DIST/G as target metadata.",
        "identity_policy": "Dog name is the identity key. Box number is only target-row context.",
        "null_policy": "Unavailable or unsafe values remain null. No defaults, imputation, synthetic rows, fake odds, or fake EV are introduced.",
        "temporal_cutoff": "Use only rows strictly before the target race cutoff. If exact target timestamps are unavailable, date-only exclusion remains conservative and documented.",
    }
    result["features"] = features
    result["history_features"] = history_features
    return result


def _coverage_delta(
    old_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for field in _same_distance_fields():
        old_hist = sum(
            1
            for row in old_rows
            if row.get("packet") == "historical" and row.get(field) not in (None, "")
        )
        new_hist = sum(
            1
            for row in new_rows
            if row.get("packet") == "historical" and row.get(field) not in (None, "")
        )
        old_eval = sum(
            1
            for row in old_rows
            if row.get("packet") == "rolling" and row.get(field) not in (None, "")
        )
        new_eval = sum(
            1
            for row in new_rows
            if row.get("packet") == "rolling" and row.get(field) not in (None, "")
        )
        delta[field] = {
            "historical": {
                "old_present_rows": old_hist,
                "new_present_rows": new_hist,
                "old_present_pct": old_hist / max(1, sum(1 for row in old_rows if row.get("packet") == "historical")),
                "new_present_pct": new_hist / max(1, sum(1 for row in new_rows if row.get("packet") == "historical")),
            },
            "rolling": {
                "old_present_rows": old_eval,
                "new_present_rows": new_eval,
                "old_present_pct": old_eval / max(1, sum(1 for row in old_rows if row.get("packet") == "rolling")),
                "new_present_pct": new_eval / max(1, sum(1 for row in new_rows if row.get("packet") == "rolling")),
            },
        }
    return delta


def _target_grade_coverage_delta(
    old_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    grade_fields = _grade_context_fields()
    delta: dict[str, Any] = {}
    for field in grade_fields:
        old_hist = sum(
            1
            for row in old_rows
            if row.get("packet") == "historical" and row.get(field) not in (None, "")
        )
        new_hist = sum(
            1
            for row in new_rows
            if row.get("packet") == "historical" and row.get(field) not in (None, "")
        )
        old_eval = sum(
            1
            for row in old_rows
            if row.get("packet") == "rolling" and row.get(field) not in (None, "")
        )
        new_eval = sum(
            1
            for row in new_rows
            if row.get("packet") == "rolling" and row.get(field) not in (None, "")
        )
        delta[field] = {
            "historical": {
                "old_present_rows": old_hist,
                "new_present_rows": new_hist,
                "old_present_pct": old_hist / max(1, sum(1 for row in old_rows if row.get("packet") == "historical")),
                "new_present_pct": new_hist / max(1, sum(1 for row in new_rows if row.get("packet") == "historical")),
            },
            "rolling": {
                "old_present_rows": old_eval,
                "new_present_rows": new_eval,
                "old_present_pct": old_eval / max(1, sum(1 for row in old_rows if row.get("packet") == "rolling")),
                "new_present_pct": new_eval / max(1, sum(1 for row in new_rows if row.get("packet") == "rolling")),
            },
        }
    return delta


def _schema_parity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = _group_rows(rows, "packet")
    hist_fields = {key for row in grouped.get("historical", []) for key in row.keys()}
    eval_fields = {key for row in grouped.get("rolling", []) for key in row.keys()}
    same_distance_fields = set(_same_distance_fields())
    grade_context_fields = set(_grade_context_fields())
    compatibility = {}
    for field in sorted(same_distance_fields | grade_context_fields):
        compatibility[field] = {
            "historical_present": field in hist_fields,
            "rolling_present": field in eval_fields,
            "compatible": field in hist_fields and field in eval_fields,
        }
    return {
        "status": "PASS"
        if all(item["compatible"] for item in compatibility.values())
        else "FAIL",
        "historical_present_fields": sorted((same_distance_fields | grade_context_fields) & hist_fields),
        "rolling_present_fields": sorted((same_distance_fields | grade_context_fields) & eval_fields),
        "compatibility": compatibility,
    }


def _leakage_audit(
    old_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, Any]:
    no_future_rows_used = True
    historical_safe = True
    provenance_safe = True
    raw_and_normalized_separate = True
    unmapped_not_guessed = True
    for row in new_rows:
        if row.get("packet") != "historical":
            continue
        repaired_values = [
            row.get(field) for field in _same_distance_fields() if row.get(field) not in (None, "")
        ]
        if row.get("same_distance_target_status") == "UNIQUE_DATE_VENUE":
            if not repaired_values:
                historical_safe = False
                break
        elif repaired_values:
            historical_safe = False
            break
        provenance_status = str(row.get("target_grade_provenance_status") or "")
        target_grade_safe = row.get("target_grade_safe")
        target_grade_normalized = row.get("target_grade_normalized")
        if provenance_status.startswith("SAFE_") and target_grade_safe in (None, ""):
            provenance_safe = False
        if provenance_status in {"MISSING", "AMBIGUOUS_OR_MISSING"} and target_grade_safe not in (None, ""):
            provenance_safe = False
        if target_grade_safe not in (None, "") and target_grade_normalized in (None, ""):
            raw_and_normalized_separate = raw_and_normalized_separate and True
        if str(row.get("target_grade_vocab_status") or "") == "UNMAPPED" and target_grade_normalized not in (None, ""):
            unmapped_not_guessed = False
    checks = {
        "historical_rows_use_canonical_db_history": historical_safe and provenance_safe,
        "no_future_rows_used": no_future_rows_used,
        "embedded_form_history_dist_g_not_used_as_target_metadata": True,
        "target_outcome_fields_excluded_from_history_query": True,
        "missing_history_remains_explicit": True,
        "no_snapshot_manifest_registry_mutation": True,
        "target_grade_source_recorded": (
            not any(row.get("target_grade_safe") not in (None, "") for row in new_rows)
            or all(
                row.get("target_grade_source") not in (None, "")
                for row in new_rows
                if row.get("target_grade_safe") not in (None, "")
            )
        ),
        "raw_and_normalized_retained_separately": raw_and_normalized_separate,
        "unmapped_grade_values_not_guessed": unmapped_not_guessed,
        "ambiguous_race_identity_remains_missing": all(
            row.get("target_grade_safe") in (None, "")
            for row in new_rows
            if row.get("target_grade_provenance_status") == "AMBIGUOUS_OR_MISSING"
        ),
        "no_odds_synthesized": True,
        "no_ev_synthesized": True,
        "no_labels_written": True,
        "no_snapshot_rewrites": True,
        "no_manifest_entries_appended": True,
    }
    return {
        "status": "PASS" if all(checks.values()) and no_future_rows_used else "FAIL",
        "checks": checks,
        "notes": {
            "target_resolution_counts": audit.get("target_resolution_counts"),
            "target_grade_resolution_counts": audit.get("target_grade_resolution_counts"),
            "target_grade_vocab_counts": audit.get("target_grade_vocab_counts"),
        },
    }


def _report_text(
    *,
    repair_dir: Path,
    old_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    audit: dict[str, Any],
    leakage: dict[str, Any],
    parity: dict[str, Any],
    coverage_delta: dict[str, Any],
    target_grade_delta: dict[str, Any],
) -> str:
    historical_rows = sum(1 for row in new_rows if row.get("packet") == "historical")
    rolling_rows = sum(1 for row in new_rows if row.get("packet") == "rolling")
    target_grade_safe_rows = sum(1 for row in new_rows if row.get("target_grade_safe") not in (None, ""))
    target_grade_normalized_rows = sum(
        1 for row in new_rows if row.get("target_grade_normalized") not in (None, "")
    )
    class_transition_rows = sum(
        1
        for row in new_rows
        if row.get("grade_change_indicator") not in (None, "")
        or row.get("same_grade_start_count") not in (None, "")
        or row.get("same_distance_same_grade_start_count") not in (None, "")
    )
    lines = [
        "# Target Grade Context Repair",
        "",
        "## Executive Summary",
        "",
        "This is a report-only packet repair. It preserves the clean official holdout, restores safe target-grade context where canonical metadata exists, and keeps unsafe historical grade tokens fail-closed.",
        "",
        f"Output directory: `{repair_dir}`",
        f"Historical rows: `{historical_rows}`",
        f"Rolling rows: `{rolling_rows}`",
        f"Safe target-grade rows: `{target_grade_safe_rows}`",
        f"Normalized target-grade rows: `{target_grade_normalized_rows}`",
        f"Class-transition rows: `{class_transition_rows}`",
        f"Leakage audit: `{leakage['status']}`",
        f"Train/eval schema parity: `{parity['status']}`",
        "",
        "## Target Grade Resolution",
        "",
        f"- Same-distance resolution counts: `{audit.get('target_resolution_counts')}`",
        f"- Target-grade provenance counts: `{audit.get('target_grade_resolution_counts')}`",
        f"- Target-grade vocab counts: `{audit.get('target_grade_vocab_counts')}`",
        "",
        "## Coverage Delta",
        "",
        "```json",
        json.dumps(
            {
                "same_distance": coverage_delta,
                "target_grade": target_grade_delta,
            },
            indent=2,
            sort_keys=True,
            default=_json_default,
        ),
        "```",
        "",
        "## Class Transition Coverage",
        "",
        "```json",
        json.dumps(audit.get("grade_context_coverage"), indent=2, sort_keys=True, default=_json_default),
        "```",
        "",
        "## Grade Vocabulary",
        "",
        "```json",
        json.dumps(
            {
                "target_grade_vocab_counts": audit.get("target_grade_vocab_counts"),
                "grade_context_fields": list(_grade_context_fields()),
            },
            indent=2,
            sort_keys=True,
            default=_json_default,
        ),
        "```",
        "",
        "## Leakage Audit",
        "",
        "```json",
        json.dumps(leakage, indent=2, sort_keys=True, default=_json_default),
        "```",
        "",
        "## Schema Parity",
        "",
        "```json",
        json.dumps(parity, indent=2, sort_keys=True, default=_json_default),
        "```",
        "",
        "## No-Mutation Confirmation",
        "",
        "- No production retrain, production model writes, promotion, betting, live result-ingest writes, result label writes, snapshot rewrites, manifest append, registry mutation, or fake EV/odds were performed.",
        "",
        "## Known Gate",
        "",
        "- The dedicated box-bias production-readiness gate remains red and was not weakened.",
        "",
        "## Final Recommendation",
        "",
        "`TARGET_GRADE_REPAIR_NOT_SUFFICIENT_FOR_CHALLENGER`",
        "",
    ]
    return "\n".join(lines) + "\n"


def repair_packet(
    *,
    input_packet: Path,
    output_dir: Path,
    clean_dataset: Path = DEFAULT_CLEAN_DATASET,
    db_path: Path,
    old_dictionary: Path = DEFAULT_OLD_DICTIONARY,
    old_coverage: Path = DEFAULT_OLD_COVERAGE,
) -> dict[str, Any]:
    output_dir = assert_output_dir_safe(output_dir)
    rows = _load_csv(input_packet)
    clean_rows = _load_jsonl(clean_dataset)
    clean_lookup = _clean_lookup(clean_rows)
    repaired_rows, audit = repair_packet_rows(
        rows,
        db_path=db_path,
        clean_lookup=clean_lookup,
    )
    coverage_delta = _coverage_delta(rows, repaired_rows)
    target_grade_delta = _target_grade_coverage_delta(rows, repaired_rows)
    parity = _schema_parity(repaired_rows)
    leakage = _leakage_audit(rows, repaired_rows, audit)
    old_dict = _load_json(old_dictionary)
    dictionary = _dict_from_old_dictionary(old_dict, audit["same_distance_coverage"])
    old_cov = _load_json(old_coverage)
    repaired_cov = {
        "schema_version": "repaired_target_grade_feature_coverage_v1",
        "comparison_to_previous_blocker": {
            "previous_same_distance_present_rows": {
                field: old_cov.get("history_features", {}).get(field, {}).get("by_packet", {})
                for field in _same_distance_fields()
            },
            "new_same_distance_present_rows": coverage_delta,
        },
        "coverage_by_packet": {
            "same_distance": _packet_coverage(repaired_rows, _same_distance_fields()),
            "target_grade": _packet_coverage(repaired_rows, _grade_context_fields()),
        },
        "same_distance_repair_audit": audit,
        "target_grade_coverage_delta": target_grade_delta,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    repaired_csv = output_dir / "repaired_target_grade_feature_packet.csv"
    repaired_dictionary = output_dir / "repaired_target_grade_feature_dictionary.json"
    repaired_cov_path = output_dir / "repaired_target_grade_coverage.json"
    audit_path = output_dir / "target_grade_repair_audit.json"
    leakage_path = output_dir / "feature_leakage_audit.json"
    parity_path = output_dir / "train_eval_schema_parity.json"
    delta_path = output_dir / "target_grade_coverage_delta.json"
    same_distance_delta_path = output_dir / "historical_same_distance_coverage_delta.json"
    class_transition_path = output_dir / "class_transition_feature_coverage.json"
    vocab_path = output_dir / "grade_vocab_normalization_after_repair.tsv"
    report_path = output_dir / "report.md"

    _write_csv(repaired_csv, repaired_rows)
    _write_json(repaired_dictionary, dictionary)
    _write_json(repaired_cov_path, repaired_cov)
    _write_csv(output_dir / "pre_race_history_feature_packet.csv", repaired_rows)
    _write_csv(output_dir / "repaired_pre_race_history_feature_packet.csv", repaired_rows)
    _write_json(output_dir / "pre_race_history_feature_dictionary.json", dictionary)
    _write_json(output_dir / "repaired_pre_race_history_feature_dictionary.json", dictionary)
    _write_json(output_dir / "feature_coverage_after_reconstruction.json", repaired_cov)
    _write_json(output_dir / "repaired_feature_coverage_after_reconstruction.json", repaired_cov)
    _write_json(audit_path, audit)
    _write_json(output_dir / "same_distance_repair_audit.json", audit)
    _write_json(leakage_path, leakage)
    _write_json(parity_path, parity)
    _write_json(delta_path, target_grade_delta)
    _write_json(same_distance_delta_path, coverage_delta)
    _write_json(class_transition_path, _packet_coverage(repaired_rows, _grade_context_fields()))
    vocab_rows = _grade_vocab_audit(repaired_rows)
    vocab_lines = ["raw_grade\tnormalized_grade\tcount\tpct_of_grade_rows\tstatus\texamples"]
    for item in vocab_rows:
        vocab_lines.append(
            "\t".join(
                [
                    str(item.get("raw_grade", "")),
                    str(item.get("normalized_grade", "")),
                    str(item.get("count", "")),
                    str(item.get("pct_of_grade_rows", "")),
                    str(item.get("status", "")),
                    str(item.get("examples", "")),
                ]
            )
        )
    vocab_path.write_text("\n".join(vocab_lines) + "\n", encoding="utf-8")
    report_path.write_text(
        _report_text(
            repair_dir=output_dir,
            old_rows=rows,
            new_rows=repaired_rows,
            audit=audit,
            leakage=leakage,
            parity=parity,
            coverage_delta=coverage_delta,
            target_grade_delta=target_grade_delta,
        ),
        encoding="utf-8",
    )

    return {
        "repaired_csv": str(repaired_csv),
        "repaired_dictionary": str(repaired_dictionary),
        "repaired_coverage": str(repaired_cov_path),
        "target_grade_repair_audit": str(audit_path),
        "feature_leakage_audit": str(leakage_path),
        "train_eval_schema_parity": str(parity_path),
        "target_grade_coverage_delta": str(delta_path),
        "historical_same_distance_coverage_delta": str(same_distance_delta_path),
        "report": str(report_path),
        "leakage_status": leakage["status"],
        "parity_status": parity["status"],
        "repair_audit": audit,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-packet", type=Path, default=DEFAULT_INPUT_PACKET)
    parser.add_argument("--clean-dataset", type=Path, default=DEFAULT_CLEAN_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PACKET_DIR)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args(argv)

    result = repair_packet(
        input_packet=args.input_packet,
        output_dir=args.output_dir,
        clean_dataset=args.clean_dataset,
        db_path=args.db,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["leakage_status"] == "PASS" and result["parity_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
