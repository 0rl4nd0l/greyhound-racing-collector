#!/usr/bin/env python3
"""Build report-only official-vs-DB runner identity reconciliation packets."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "official_identity_reconciliation_v1"
FULL_SCHEMA_VERSION = "official_identity_reconciliation_full_v1"
LOOKUP_SCHEMA_VERSION = "official_reverify_lookup_dry_run_v1"

PRIMARY_BUCKETS = (
    "exact_identity_and_position_match",
    "exact_identity_position_diff_safe_review",
    "box_identity_drift",
    "name_set_mismatch",
    "missing_db_rows",
    "missing_metadata",
    "missing_official_names",
    "parser_failure",
    "official_lookup_failure",
    "duplicate_or_ambiguous_db_identity",
    "review_required_other",
)

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
    "betting_decision": False,
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("json_root_not_object")
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _canonical_race_id(result: Mapping[str, Any]) -> str:
    lookup_key = _mapping(result.get("lookup_key"))
    venue = str(lookup_key.get("venue") or "").strip().upper().replace(" ", "_")
    race_date = str(lookup_key.get("race_date") or "").strip()
    race_number = _safe_int(lookup_key.get("race_number")) or 0
    return f"{venue}_{race_date}_{race_number}" if venue and race_date and race_number else ""


def _race_id_variants(result: Mapping[str, Any]) -> list[str]:
    lookup_key = _mapping(result.get("lookup_key"))
    venue = str(lookup_key.get("venue") or "").strip().upper().replace(" ", "_")
    race_date = str(lookup_key.get("race_date") or "").strip()
    race_number = _safe_int(lookup_key.get("race_number")) or 0
    variants = {
        _canonical_race_id(result),
        f"Race {race_number} - {venue} - {race_date}" if venue and race_date and race_number else "",
    }
    return sorted(item for item in variants if item)


def _clean_display_name(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(r"^\s*\d{1,2}\s*[\.\):-]\s*", "", text)
    text = re.sub(r"\s+\d{1,2}\.\d{2}\s+T:\s+.*$", "", text)
    text = re.sub(r"\s+T:\s+.*$", "", text)
    return text.strip()


def _name_key(value: Any) -> str:
    text = _clean_display_name(value).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    resolved = db_path.expanduser().resolve()
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    if not _table_exists(conn, table):
        return []
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _resolve_race_id(conn: sqlite3.Connection, result: Mapping[str, Any]) -> str:
    variants = _race_id_variants(result)
    for race_id in variants:
        row = conn.execute(
            "SELECT 1 FROM dog_race_data WHERE race_id = ? LIMIT 1",
            (race_id,),
        ).fetchone()
        if row is not None:
            return race_id
    return variants[0] if variants else _canonical_race_id(result)


def _db_rows(conn: sqlite3.Connection, race_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT dog_name, box_number, finish_position, placing,
               scraped_finish_position, data_source
        FROM dog_race_data
        WHERE race_id = ?
        ORDER BY box_number
        """,
        (race_id,),
    ).fetchall()
    return [
        {
            "dog_name": row["dog_name"],
            "dog_name_clean": _clean_display_name(row["dog_name"]),
            "dog_name_key": _name_key(row["dog_name"]),
            "box_number": _safe_int(row["box_number"]),
            "finish_position": _safe_int(row["finish_position"]),
            "placing": _safe_int(row["placing"]),
            "scraped_finish_position": row["scraped_finish_position"],
            "data_source": row["data_source"],
        }
        for row in rows
    ]


def _official_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_rows = _list(result.get("official_runner_rows")) or _list(result.get("positions"))
    rows = []
    for item in source_rows:
        row = _mapping(item)
        finish_position = _safe_int(row.get("finish_position"))
        if finish_position is None:
            continue
        box_number = _safe_int(row.get("box_number"))
        dog_name = _clean_display_name(row.get("dog_name"))
        rows.append(
            {
                "dog_name": dog_name or None,
                "dog_name_key": _name_key(dog_name),
                "box_number": box_number,
                "finish_position": finish_position,
            }
        )
    return sorted(rows, key=lambda row: (row["finish_position"], row["box_number"] or 99))


def _reconcile_result(conn: sqlite3.Connection, result: Mapping[str, Any]) -> dict[str, Any]:
    race_id = _resolve_race_id(conn, result)
    official_rows = _official_rows(result)
    db_rows = _db_rows(conn, race_id)
    if not db_rows:
        return {
            "race_id": race_id,
            "legacy_race_id": result.get("legacy_race_id"),
            "lookup_key": result.get("lookup_key"),
            "status": "DB_ROWS_MISSING",
            "official_rows": official_rows,
            "db_rows": [],
            "matches_by_name": [],
            "all_official_names_found_in_db": False,
            "box_mismatch_count": 0,
            "finish_position_mismatch_count": 0,
        }
    if not official_rows or any(not row["dog_name_key"] for row in official_rows):
        return {
            "race_id": race_id,
            "legacy_race_id": result.get("legacy_race_id"),
            "lookup_key": result.get("lookup_key"),
            "status": "OFFICIAL_NAMES_MISSING",
            "official_rows": official_rows,
            "db_rows": db_rows,
            "matches_by_name": [],
            "all_official_names_found_in_db": False,
            "box_mismatch_count": 0,
            "finish_position_mismatch_count": 0,
        }

    db_by_name = {row["dog_name_key"]: row for row in db_rows if row["dog_name_key"]}
    official_keys = [row["dog_name_key"] for row in official_rows]
    db_keys = [row["dog_name_key"] for row in db_rows if row["dog_name_key"]]
    all_names_found = set(official_keys) == set(db_keys) and len(official_keys) == len(db_keys)
    matches = []
    box_mismatch_count = 0
    finish_mismatch_count = 0
    for official in official_rows:
        db_row = db_by_name.get(official["dog_name_key"])
        if db_row is None:
            matches.append(
                {
                    "official_dog_name": official["dog_name"],
                    "db_dog_name": None,
                    "official_box_number": official["box_number"],
                    "db_box_number": None,
                    "official_finish_position": official["finish_position"],
                    "db_finish_position": None,
                    "box_matches": False,
                    "finish_position_matches": False,
                }
            )
            box_mismatch_count += 1
            finish_mismatch_count += 1
            continue
        box_matches = official["box_number"] == db_row["box_number"]
        finish_matches = official["finish_position"] == db_row["finish_position"]
        if not box_matches:
            box_mismatch_count += 1
        if not finish_matches:
            finish_mismatch_count += 1
        matches.append(
            {
                "official_dog_name": official["dog_name"],
                "db_dog_name": db_row["dog_name"],
                "official_box_number": official["box_number"],
                "db_box_number": db_row["box_number"],
                "official_finish_position": official["finish_position"],
                "db_finish_position": db_row["finish_position"],
                "box_matches": box_matches,
                "finish_position_matches": finish_matches,
            }
        )

    if not all_names_found:
        status = "NAME_SET_MISMATCH"
    elif box_mismatch_count:
        status = "BOX_IDENTITY_DRIFT"
    elif finish_mismatch_count:
        status = "POSITION_DRIFT"
    else:
        status = "EXACT_IDENTITY_AND_POSITION_MATCH"

    return {
        "race_id": race_id,
        "legacy_race_id": result.get("legacy_race_id"),
        "lookup_key": result.get("lookup_key"),
        "status": status,
        "official_rows": official_rows,
        "db_rows": db_rows,
        "matches_by_name": matches,
        "all_official_names_found_in_db": all_names_found,
        "box_mismatch_count": box_mismatch_count,
        "finish_position_mismatch_count": finish_mismatch_count,
    }


def _select_results(lookup: Mapping[str, Any], candidate_scope: str) -> list[Mapping[str, Any]]:
    results = [row for row in _list(lookup.get("results")) if isinstance(row, Mapping)]
    if candidate_scope == "all":
        return results
    if candidate_scope == "label-write-ready":
        return [row for row in results if row.get("label_write_ready") is True]
    if candidate_scope == "parse-ready":
        return [row for row in results if row.get("result_parse_ready") is True]
    raise ValueError(f"unknown_candidate_scope:{candidate_scope}")


def _file_state(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.expanduser().resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _safe_count(conn: sqlite3.Connection, table: str) -> int | None:
    if not _table_exists(conn, table):
        return None
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _safe_max(conn: sqlite3.Connection, table: str, column: str) -> Any:
    if column not in _table_columns(conn, table):
        return None
    return conn.execute(f"SELECT MAX({column}) FROM {table}").fetchone()[0]


def _db_health_snapshot(conn: sqlite3.Connection, db_path: Path) -> dict[str, Any]:
    tables = ["race_metadata", "dog_race_data", "dogs", "race_results"]
    table_counts = {table: _safe_count(conn, table) for table in tables if _table_exists(conn, table)}
    official_label_counts: dict[str, Any] = {
        "official_races": None,
        "official_dog_rows": None,
    }
    dog_data_source_counts: list[dict[str, Any]] = []
    if _table_exists(conn, "dog_race_data") and "data_source" in _table_columns(conn, "dog_race_data"):
        official_label_counts["official_dog_rows"] = int(
            conn.execute(
                "SELECT COUNT(*) FROM dog_race_data WHERE data_source = 'thedogs_official'"
            ).fetchone()[0]
        )
        official_label_counts["official_races"] = int(
            conn.execute(
                "SELECT COUNT(DISTINCT race_id) FROM dog_race_data WHERE data_source = 'thedogs_official'"
            ).fetchone()[0]
        )
        dog_data_source_counts = [
            dict(row)
            for row in conn.execute(
                """
                SELECT
                    COALESCE(data_source, 'NULL') AS data_source,
                    COUNT(*) AS dog_rows,
                    COUNT(DISTINCT race_id) AS races
                FROM dog_race_data
                GROUP BY COALESCE(data_source, 'NULL')
                ORDER BY dog_rows DESC
                """
            ).fetchall()
        ]
    metadata_winner_source_counts: list[dict[str, Any]] = []
    if _table_exists(conn, "race_metadata") and "winner_source" in _table_columns(conn, "race_metadata"):
        metadata_winner_source_counts = [
            dict(row)
            for row in conn.execute(
                """
                SELECT
                    COALESCE(winner_source, 'NULL') AS winner_source,
                    COUNT(*) AS races
                FROM race_metadata
                GROUP BY COALESCE(winner_source, 'NULL')
                ORDER BY races DESC
                """
            ).fetchall()
        ]
    label_like_counts = {"dog_rows": None, "races": None}
    if _table_exists(conn, "dog_race_data"):
        label_like = conn.execute(
            """
            SELECT COUNT(*) AS dog_rows, COUNT(DISTINCT race_id) AS races
            FROM dog_race_data
            WHERE finish_position IS NOT NULL
               OR placing IS NOT NULL
               OR scraped_finish_position IS NOT NULL
            """
        ).fetchone()
        label_like_counts = {
            "dog_rows": int(label_like["dog_rows"] or 0),
            "races": int(label_like["races"] or 0),
        }
    return {
        "quick_check": conn.execute("PRAGMA quick_check").fetchone()[0],
        "query_only": conn.execute("PRAGMA query_only").fetchone()[0],
        "file_state": _file_state(db_path),
        "table_counts": table_counts,
        "official_label_counts": official_label_counts,
        "dog_data_source_counts": dog_data_source_counts,
        "metadata_winner_source_counts": metadata_winner_source_counts,
        "label_like_counts": label_like_counts,
        "max_timestamps": {
            "dog_race_data.extraction_timestamp": _safe_max(
                conn, "dog_race_data", "extraction_timestamp"
            ),
            "race_metadata.extraction_timestamp": _safe_max(
                conn, "race_metadata", "extraction_timestamp"
            ),
            "race_metadata.last_scraped_at": _safe_max(conn, "race_metadata", "last_scraped_at"),
        },
    }


def _metadata_rows(conn: sqlite3.Connection, result: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not _table_exists(conn, "race_metadata"):
        return []
    columns = _table_columns(conn, "race_metadata")
    select_columns = [
        column
        for column in (
            "race_id",
            "venue",
            "race_number",
            "race_date",
            "results_status",
            "winner_name",
            "winner_source",
            "grade",
            "distance",
            "race_time",
            "url",
            "sportsbet_url",
            "data_source",
            "race_status",
        )
        if column in columns
    ]
    if not select_columns:
        return []

    lookup_key = _mapping(result.get("lookup_key"))
    venue = str(lookup_key.get("venue") or "").strip().upper().replace(" ", "_")
    race_date = str(lookup_key.get("race_date") or "").strip()
    race_number = _safe_int(lookup_key.get("race_number")) or 0
    variants = _race_id_variants(result)
    conditions: list[str] = []
    params: list[Any] = []
    if variants and "race_id" in columns:
        conditions.append("race_id IN ({})".format(",".join("?" for _ in variants)))
        params.extend(variants)
    if {"race_date", "race_number", "venue"}.issubset(columns) and venue and race_date and race_number:
        conditions.append(
            "(race_date = ? AND CAST(race_number AS INTEGER) = ? "
            "AND UPPER(REPLACE(COALESCE(venue, ''), ' ', '_')) = ?)"
        )
        params.extend([race_date, race_number, venue])
    if not conditions:
        return []

    sql = (
        "SELECT "
        + ", ".join(select_columns)
        + " FROM race_metadata WHERE "
        + " OR ".join(conditions)
        + " ORDER BY race_id"
    )
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def _metadata_status(metadata_rows: list[Mapping[str, Any]]) -> str:
    if not metadata_rows:
        return "missing"
    if len(metadata_rows) > 1:
        return "ambiguous"
    row = metadata_rows[0]
    required = ("race_id", "venue", "race_number", "race_date")
    return "complete" if all(row.get(key) not in (None, "") for key in required) else "incomplete"


def _first_race_id_with_dog_rows(
    conn: sqlite3.Connection, result: Mapping[str, Any], metadata_rows: list[Mapping[str, Any]]
) -> str:
    candidates = [
        str(row.get("race_id"))
        for row in metadata_rows
        if row.get("race_id") not in (None, "")
    ] + _race_id_variants(result)
    if not _table_exists(conn, "dog_race_data"):
        return candidates[0] if candidates else _canonical_race_id(result)
    for race_id in candidates:
        row = conn.execute(
            "SELECT 1 FROM dog_race_data WHERE race_id = ? LIMIT 1",
            (race_id,),
        ).fetchone()
        if row is not None:
            return race_id
    return candidates[0] if candidates else _canonical_race_id(result)


def _db_rows_full(conn: sqlite3.Connection, race_id: str) -> list[dict[str, Any]]:
    if not _table_exists(conn, "dog_race_data"):
        return []
    columns = _table_columns(conn, "dog_race_data")
    select_columns = [
        column
        for column in (
            "race_id",
            "dog_name",
            "dog_clean_name",
            "box_number",
            "finish_position",
            "placing",
            "scraped_finish_position",
            "data_source",
            "extraction_timestamp",
            "form_guide_json",
            "historical_records",
            "data_quality_note",
        )
        if column in columns
    ]
    if not select_columns:
        return []
    order_by = "CAST(box_number AS INTEGER), dog_name" if "box_number" in columns else "rowid"
    rows = conn.execute(
        f"SELECT {', '.join(select_columns)} FROM dog_race_data WHERE race_id = ? ORDER BY {order_by}",
        (race_id,),
    ).fetchall()
    payload = []
    for row in rows:
        raw = dict(row)
        dog_name = raw.get("dog_name") or raw.get("dog_clean_name")
        result_position = (
            _safe_int(raw.get("finish_position"))
            if _safe_int(raw.get("finish_position")) is not None
            else _safe_int(raw.get("placing"))
        )
        if result_position is None:
            result_position = _safe_int(raw.get("scraped_finish_position"))
        payload.append(
            {
                **raw,
                "dog_name_clean": _clean_display_name(dog_name),
                "dog_name_key": _name_key(dog_name),
                "box_number": _safe_int(raw.get("box_number")),
                "finish_position": _safe_int(raw.get("finish_position")),
                "placing": _safe_int(raw.get("placing")),
                "result_position": result_position,
            }
        )
    return payload


def _duplicates(values: Iterable[str]) -> list[str]:
    counts = Counter(value for value in values if value)
    return sorted(value for value, count in counts.items() if count > 1)


def _comparison_payload(
    official_rows: list[dict[str, Any]],
    db_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    official_key_counts = Counter(row["dog_name_key"] for row in official_rows if row["dog_name_key"])
    db_key_counts = Counter(row["dog_name_key"] for row in db_rows if row["dog_name_key"])
    official_set = sorted(official_key_counts)
    db_set = sorted(db_key_counts)
    official_not_in_db = sorted(set(official_set) - set(db_set))
    db_not_in_official = sorted(set(db_set) - set(official_set))

    duplicate_official = _duplicates(row["dog_name_key"] for row in official_rows)
    duplicate_db = _duplicates(row["dog_name_key"] for row in db_rows)
    can_match_by_name = not duplicate_official and not duplicate_db
    official_by_name = {row["dog_name_key"]: row for row in official_rows if row["dog_name_key"]}
    db_by_name = {row["dog_name_key"]: row for row in db_rows if row["dog_name_key"]}
    overlap = sorted(set(official_by_name) & set(db_by_name))

    matches_by_name: list[dict[str, Any]] = []
    box_mismatches: list[dict[str, Any]] = []
    position_mismatches: list[dict[str, Any]] = []
    if can_match_by_name:
        for key in overlap:
            official = official_by_name[key]
            db_row = db_by_name[key]
            box_matches = official.get("box_number") == db_row.get("box_number")
            position_matches = official.get("finish_position") == db_row.get("result_position")
            match = {
                "dog_name_key": key,
                "official_dog_name": official.get("dog_name"),
                "db_dog_name": db_row.get("dog_name"),
                "official_box_number": official.get("box_number"),
                "db_box_number": db_row.get("box_number"),
                "official_finish_position": official.get("finish_position"),
                "db_result_position": db_row.get("result_position"),
                "box_matches": box_matches,
                "finish_position_matches": position_matches,
            }
            matches_by_name.append(match)
            if not box_matches:
                box_mismatches.append(match)
            if not position_matches:
                position_mismatches.append(match)

    official_boxes = sorted(row.get("box_number") for row in official_rows if row.get("box_number") is not None)
    db_boxes = sorted(row.get("box_number") for row in db_rows if row.get("box_number") is not None)
    name_result = (
        "exact_match"
        if official_key_counts == db_key_counts and not duplicate_official and not duplicate_db
        else "duplicate_or_ambiguous"
        if duplicate_official or duplicate_db
        else "mismatch"
    )
    box_result = (
        "exact_identity_box_match"
        if name_result == "exact_match" and not box_mismatches
        else "box_identity_drift"
        if box_mismatches
        else "box_set_mismatch"
        if official_boxes != db_boxes
        else "not_comparable"
    )
    position_result = (
        "exact_position_match"
        if name_result == "exact_match" and not position_mismatches
        else "position_diff"
        if position_mismatches
        else "not_comparable"
    )
    return (
        {
            "result": name_result,
            "official_name_keys": official_set,
            "db_name_keys": db_set,
            "official_not_in_db": official_not_in_db,
            "db_not_in_official": db_not_in_official,
            "duplicate_official_name_keys": duplicate_official,
            "duplicate_db_name_keys": duplicate_db,
            "overlap_name_keys": overlap,
        },
        {
            "result": box_result,
            "official_box_numbers": official_boxes,
            "db_box_numbers": db_boxes,
            "box_mismatch_count": len(box_mismatches),
            "box_mismatches_by_name": box_mismatches,
        },
        {
            "result": position_result,
            "position_mismatch_count": len(position_mismatches),
            "position_mismatches_by_name": position_mismatches,
        },
        matches_by_name,
    )


def _primary_bucket(
    *,
    result: Mapping[str, Any],
    official_rows: list[dict[str, Any]],
    db_rows: list[dict[str, Any]],
    metadata_status: str,
    name_comparison: Mapping[str, Any],
    box_comparison: Mapping[str, Any],
    position_comparison: Mapping[str, Any],
) -> str:
    lookup_status = str(result.get("lookup_status") or "")
    skip_reasons = [str(reason) for reason in _list(result.get("skip_reasons"))]
    if result.get("result_parse_ready") is not True:
        parser_reason = any(
            "position" in reason or "strict" in reason or "parser" in reason
            for reason in skip_reasons
        )
        return "parser_failure" if parser_reason else "official_lookup_failure"
    if not official_rows or any(not row.get("dog_name_key") for row in official_rows):
        return "missing_official_names"
    if lookup_status and lookup_status != "OFFICIAL_RESULT_PARSED":
        return "official_lookup_failure"
    if not db_rows:
        return "missing_db_rows"
    if (
        name_comparison.get("duplicate_db_name_keys")
        or name_comparison.get("duplicate_official_name_keys")
        or metadata_status == "ambiguous"
    ):
        return "duplicate_or_ambiguous_db_identity"
    if box_comparison.get("result") == "box_identity_drift":
        return "box_identity_drift"
    if name_comparison.get("result") != "exact_match":
        return "name_set_mismatch"
    if metadata_status != "complete":
        return "missing_metadata"
    if position_comparison.get("result") == "position_diff":
        return "exact_identity_position_diff_safe_review"
    if position_comparison.get("result") == "exact_position_match":
        return "exact_identity_and_position_match"
    return "review_required_other"


def _write_safety_status(
    *,
    primary_bucket: str,
    metadata_status: str,
    name_comparison: Mapping[str, Any],
    box_comparison: Mapping[str, Any],
    position_comparison: Mapping[str, Any],
    official_rows: list[dict[str, Any]],
    db_rows: list[dict[str, Any]],
    skip_reasons: list[str],
) -> str:
    names_present = bool(official_rows) and all(row.get("dog_name_key") for row in official_rows)
    db_names_present = bool(db_rows) and all(row.get("dog_name_key") for row in db_rows)
    dog_rows_complete = (
        len(official_rows) == len(db_rows)
        and name_comparison.get("result") == "exact_match"
        and box_comparison.get("result") == "exact_identity_box_match"
    )
    safe_common = (
        names_present
        and db_names_present
        and metadata_status == "complete"
        and dog_rows_complete
        and not skip_reasons
    )
    if safe_common and position_comparison.get("result") == "position_diff":
        return "safe_review_candidate"
    if safe_common and position_comparison.get("result") == "exact_position_match":
        return "safe_no_write_candidate"
    if primary_bucket == "box_identity_drift":
        return "unsafe_box_identity_drift"
    if primary_bucket == "name_set_mismatch":
        return "unsafe_name_set_mismatch"
    if primary_bucket in {"missing_db_rows", "missing_metadata", "missing_official_names"}:
        return f"blocked_{primary_bucket}"
    if primary_bucket in {"official_lookup_failure", "parser_failure"}:
        return f"blocked_{primary_bucket}"
    if primary_bucket == "duplicate_or_ambiguous_db_identity":
        return "blocked_duplicate_or_ambiguous_identity"
    return "review_required"


def _recommended_action(primary_bucket: str, write_safety_status: str) -> str:
    if write_safety_status == "safe_review_candidate":
        return "No-write rehearsal for this correction lane only, then human review, backup, and explicit approval before any label update."
    if write_safety_status == "safe_no_write_candidate":
        return "No correction needed; retain as audit evidence only."
    if primary_bucket == "box_identity_drift":
        return "Do not position-correct; investigate runner identity and box mapping as unsafe identity drift."
    if primary_bucket == "name_set_mismatch":
        return "Do not write; reconcile official and DB runner identity sets first."
    if primary_bucket == "missing_db_rows":
        return "Do not write labels; repair or seed missing DB runner rows only under a separate approved plan."
    if primary_bucket == "missing_metadata":
        return "Do not write labels; repair missing or incomplete race metadata under a separate approved plan."
    if primary_bucket == "missing_official_names":
        return "Do not write labels; rerun or repair official parser evidence so official names are present."
    if primary_bucket == "official_lookup_failure":
        return "Retry official lookup or inspect source availability; no DB mutation."
    if primary_bucket == "parser_failure":
        return "Inspect parser failure against official HTML; no DB mutation."
    return "Manual review required; no DB mutation."


def _reconcile_result_full(conn: sqlite3.Connection, result: Mapping[str, Any]) -> dict[str, Any]:
    metadata_rows = _metadata_rows(conn, result)
    metadata_status = _metadata_status(metadata_rows)
    race_id = _first_race_id_with_dog_rows(conn, result, metadata_rows)
    official_rows = _official_rows(result)
    db_rows = _db_rows_full(conn, race_id)
    name_comparison, box_comparison, position_comparison, matches_by_name = _comparison_payload(
        official_rows, db_rows
    )
    skip_reasons = [str(reason) for reason in _list(result.get("skip_reasons"))]
    primary_bucket = _primary_bucket(
        result=result,
        official_rows=official_rows,
        db_rows=db_rows,
        metadata_status=metadata_status,
        name_comparison=name_comparison,
        box_comparison=box_comparison,
        position_comparison=position_comparison,
    )
    reason_codes = set(skip_reasons)
    if result.get("label_write_ready") is not True:
        reason_codes.add("lookup_not_label_write_ready_in_current_packet")
    if metadata_status != "complete":
        reason_codes.add(f"metadata_{metadata_status}")
    if not db_rows:
        reason_codes.add("db_dog_rows_missing")
    if not official_rows:
        reason_codes.add("official_runner_rows_missing")
    if any(not row.get("dog_name_key") for row in official_rows):
        reason_codes.add("official_names_missing")
    if any(not row.get("dog_name_key") for row in db_rows):
        reason_codes.add("db_names_missing")
    if name_comparison.get("result") != "exact_match":
        reason_codes.add(f"name_set_{name_comparison.get('result')}")
    if box_comparison.get("result") == "box_identity_drift":
        reason_codes.add("box_identity_drift")
    if position_comparison.get("result") == "position_diff":
        reason_codes.add("position_diff")
    if metadata_status == "ambiguous" or name_comparison.get("duplicate_db_name_keys"):
        reason_codes.add("duplicate_or_ambiguous_db_identity")

    write_safety_status = _write_safety_status(
        primary_bucket=primary_bucket,
        metadata_status=metadata_status,
        name_comparison=name_comparison,
        box_comparison=box_comparison,
        position_comparison=position_comparison,
        official_rows=official_rows,
        db_rows=db_rows,
        skip_reasons=skip_reasons,
    )
    normalized_official = name_comparison["official_name_keys"]
    normalized_db = name_comparison["db_name_keys"]
    return {
        "race_id": race_id,
        "legacy_race_id": result.get("legacy_race_id"),
        "lookup_key": result.get("lookup_key"),
        "official_source_url": result.get("source_url"),
        "official_source_identifier": result.get("source_url") or result.get("legacy_race_id"),
        "official_lookup_status": result.get("lookup_status"),
        "official_result_parse_ready": result.get("result_parse_ready") is True,
        "lookup_label_write_ready": result.get("label_write_ready") is True,
        "lookup_skip_reasons": skip_reasons,
        "db_metadata_status": metadata_status,
        "db_metadata_rows": metadata_rows,
        "db_runner_rows": db_rows,
        "official_runner_rows": official_rows,
        "normalized_official_name_set": normalized_official,
        "normalized_db_name_set": normalized_db,
        "name_set_comparison": name_comparison,
        "box_comparison": box_comparison,
        "position_comparison": position_comparison,
        "matches_by_name": matches_by_name,
        "missing_official_names": [
            row
            for row in official_rows
            if not row.get("dog_name") or not row.get("dog_name_key")
        ],
        "missing_db_names": [
            row for row in db_rows if not row.get("dog_name") or not row.get("dog_name_key")
        ],
        "primary_bucket": primary_bucket,
        "secondary_reason_codes": sorted(reason_codes),
        "write_safety_status": write_safety_status,
        "recommended_next_action": _recommended_action(primary_bucket, write_safety_status),
        "tgr_used": False,
        "leakage_indicators": [],
        "fuzzy_identity_match_used": False,
    }


def _status_counts(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    bucket_counts = Counter(str(row.get("primary_bucket") or "DATA_MISSING") for row in records)
    write_safety_counts = Counter(
        str(row.get("write_safety_status") or "DATA_MISSING") for row in records
    )
    reason_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {bucket: [] for bucket in PRIMARY_BUCKETS}
    for record in records:
        bucket = str(record.get("primary_bucket") or "DATA_MISSING")
        if len(examples.setdefault(bucket, [])) < 8:
            examples[bucket].append(str(record.get("race_id") or record.get("legacy_race_id")))
        for reason in record.get("secondary_reason_codes") or []:
            reason_counts[str(reason)] += 1
    return {
        "bucket_counts": {bucket: int(bucket_counts.get(bucket, 0)) for bucket in PRIMARY_BUCKETS},
        "write_safety_status_counts": dict(sorted(write_safety_counts.items())),
        "examples_by_bucket": examples,
        "top_reason_codes": dict(reason_counts.most_common(25)),
        "safe_correction_lane_candidates": sum(
            1 for row in records if row.get("write_safety_status") == "safe_review_candidate"
        ),
        "safe_no_write_candidates": sum(
            1 for row in records if row.get("write_safety_status") == "safe_no_write_candidate"
        ),
        "blocked_by_box_identity_drift": bucket_counts.get("box_identity_drift", 0),
        "blocked_by_missing_db_rows_or_metadata": bucket_counts.get("missing_db_rows", 0)
        + bucket_counts.get("missing_metadata", 0),
        "blocked_by_missing_official_names_or_parser_failure": bucket_counts.get(
            "missing_official_names", 0
        )
        + bucket_counts.get("parser_failure", 0)
        + bucket_counts.get("official_lookup_failure", 0),
    }


def _docs_checked(repo_root: Path) -> list[dict[str, Any]]:
    paths = [
        repo_root / "ACTIVE_SCRIPTS_GUIDE.md",
        repo_root / "docs" / "ACTIVE_SCRIPTS_GUIDE.md",
        repo_root / "docs" / "VERIFY_AND_ROLLBACK_LIBRARY.md",
        repo_root / "docs" / "FORM_GUIDE_SPEC.md",
        repo_root / "docs" / "DATA_DICTIONARY.md",
        repo_root / "docs" / "ACCEPTANCE_CRITERIA.md",
    ]
    return [
        {
            "path": str(path),
            "exists": path.exists(),
            "note": "checked" if path.exists() else "not present in this checkout",
        }
        for path in paths
    ]


def _full_summary(
    *,
    records: list[Mapping[str, Any]],
    lookup: Mapping[str, Any],
    expected_count: int | None,
    source_artifacts: list[str],
    db_health: Mapping[str, Any],
) -> dict[str, Any]:
    counts = _status_counts(records)
    ap_matches = [
        record
        for record in records
        if record.get("race_id") == "AP_K_2025-07-21_10"
        or str(record.get("legacy_race_id") or "").lower() == "ap_k_2025-07-21_10"
    ]
    ap_assertion = {
        "race_id": "AP_K_2025-07-21_10",
        "present": bool(ap_matches),
        "primary_bucket": ap_matches[0].get("primary_bucket") if ap_matches else None,
        "status": "PASS"
        if ap_matches and ap_matches[0].get("primary_bucket") == "box_identity_drift"
        else "WARNING_NOT_PRESENT"
        if not ap_matches
        else "FAIL_NOT_BOX_IDENTITY_DRIFT",
    }
    return {
        "records_accounted_for": len(records),
        "expected_candidate_count": expected_count,
        "candidate_count_matches_expected": expected_count is None or len(records) == expected_count,
        "lookup_results_seen": len(_list(lookup.get("results"))),
        "lookup_summary": lookup.get("summary"),
        "source_artifact_paths": source_artifacts,
        "official_runner_rows_present_count": sum(
            1 for row in _list(lookup.get("results")) if row.get("official_runner_rows")
        ),
        "positions_with_clean_dog_name_count": sum(
            1
            for row in _list(lookup.get("results"))
            if any(_mapping(item).get("dog_name") for item in _list(row.get("positions")))
        ),
        **counts,
        "ap_k_2025_07_21_10_assertion": ap_assertion,
        "db_quick_check_before": _mapping(db_health.get("before")).get("quick_check"),
        "db_quick_check_after": _mapping(db_health.get("after")).get("quick_check"),
        "db_row_counts_unchanged": db_health.get("row_counts_unchanged"),
        "no_writes_performed": True,
        "tgr_used": False,
        "predictions_remain_diagnostic": True,
        "promotion_ready": False,
        "final_recommendation": (
            "Do not train, do not promote, and do not use TGR. Next step is a no-write "
            "rehearsal for the safest correction lane only; label application requires "
            "review, backup, and explicit approval."
        ),
    }


def build_full_identity_reconciliation_packet(
    *,
    lookup_packet_path: Path,
    db_path: Path,
    candidate_scope: str = "all",
    expected_count: int | None = None,
    source_artifacts: list[str] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    lookup_resolved = lookup_packet_path.expanduser().resolve()
    db_resolved = db_path.expanduser().resolve()
    lookup = _load_json(lookup_resolved)
    failures = []
    if lookup.get("schema_version") != LOOKUP_SCHEMA_VERSION:
        failures.append("lookup_schema_mismatch")
    writes = _mapping(lookup.get("writes_performed"))
    if any(key != "official_fetch" and value is not False for key, value in writes.items()):
        failures.append("lookup_has_forbidden_write_flags")

    selected = _select_results(lookup, candidate_scope)
    records: list[dict[str, Any]] = []
    db_health: dict[str, Any] = {}
    with _connect_read_only(db_resolved) as conn:
        db_health["before"] = _db_health_snapshot(conn, db_resolved)
        if db_health["before"]["quick_check"] != "ok":
            failures.append("db_quick_check_failed_before")
        records = [_reconcile_result_full(conn, result) for result in selected]
        db_health["after"] = _db_health_snapshot(conn, db_resolved)
        if db_health["after"]["quick_check"] != "ok":
            failures.append("db_quick_check_failed_after")

    before_counts = _mapping(_mapping(db_health.get("before")).get("table_counts"))
    after_counts = _mapping(_mapping(db_health.get("after")).get("table_counts"))
    before_times = _mapping(_mapping(db_health.get("before")).get("max_timestamps"))
    after_times = _mapping(_mapping(db_health.get("after")).get("max_timestamps"))
    db_health["row_count_changes"] = {
        key: {"before": before_counts.get(key), "after": after_counts.get(key)}
        for key in sorted(set(before_counts) | set(after_counts))
        if before_counts.get(key) != after_counts.get(key)
    }
    db_health["timestamp_changes"] = {
        key: {"before": before_times.get(key), "after": after_times.get(key)}
        for key in sorted(set(before_times) | set(after_times))
        if before_times.get(key) != after_times.get(key)
    }
    db_health["row_counts_unchanged"] = not db_health["row_count_changes"]
    db_health["timestamps_unchanged"] = not db_health["timestamp_changes"]

    source_paths = [
        str(lookup_resolved),
        str(db_resolved),
        *(source_artifacts or []),
    ]
    summary = _full_summary(
        records=records,
        lookup=lookup,
        expected_count=expected_count,
        source_artifacts=source_paths,
        db_health=db_health,
    )
    if expected_count is not None and len(records) != expected_count:
        failures.append(f"candidate_count_mismatch:expected_{expected_count}:actual_{len(records)}")
    ap_status = _mapping(summary.get("ap_k_2025_07_21_10_assertion")).get("status")
    if ap_status == "FAIL_NOT_BOX_IDENTITY_DRIFT":
        failures.append("ap_k_2025_07_21_10_not_box_identity_drift")

    root = repo_root or Path.cwd()
    status = "REPORT_ONLY_NAME_AWARE_RECONCILED" if not failures else "REPORT_ONLY_WITH_FAILURES"
    return {
        "schema_version": FULL_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": status,
        "failures": failures,
        "candidate_scope": candidate_scope,
        "source_evidence": {
            "lookup_packet": str(lookup_resolved),
            "db": str(db_resolved),
            "source_artifact_paths": source_paths,
            "docs_checked": _docs_checked(root),
        },
        "normalization_policy": {
            "trim_whitespace": True,
            "collapse_repeated_spaces": True,
            "normalize_case": "lower",
            "punctuation_policy": "uses existing project-compatible non-alphanumeric separator normalization; no fuzzy matching is proof of identity",
            "fuzzy_matching_used": False,
        },
        "constraints": {
            "db_writes": False,
            "label_application": False,
            "model_training": False,
            "model_promotion": False,
            "registry_mutation": False,
            "manifest_mutation": False,
            "snapshot_mutation": False,
            "betting_decision": False,
            "ev_claim": False,
            "tgr_used": False,
        },
        "writes_performed": dict(WRITES_PERFORMED),
        "db_health": db_health,
        "summary": summary,
        "records": records,
    }


def _write_report(path: Path, packet: Mapping[str, Any]) -> None:
    summary = _mapping(packet.get("summary"))
    lines = [
        "# Official Identity Reconciliation",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB writes, label writes, metadata writes, official fetches, snapshot mutations, manifest mutations, model training, model-registry state changes, promotions, betting decisions, or expected-value assertions were performed.",
        "",
        "## Summary",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in summary.items())
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_full_summary(path: Path, packet: Mapping[str, Any]) -> None:
    summary = _mapping(packet.get("summary"))
    bucket_counts = _mapping(summary.get("bucket_counts"))
    write_counts = _mapping(summary.get("write_safety_status_counts"))
    ap = _mapping(summary.get("ap_k_2025_07_21_10_assertion"))
    db_health = _mapping(packet.get("db_health"))
    before = _mapping(db_health.get("before"))
    after = _mapping(db_health.get("after"))
    source_paths = _list(summary.get("source_artifact_paths"))
    lines = [
        "# Name-Aware Official Identity Reconciliation",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB writes, label application, metadata writes, model training, model promotion, model-registry state change, manifest mutation, snapshot mutation, betting decision, expected-value assertion, or TGR usage occurred.",
        "",
        "Predictions remain diagnostic only and are not promotion-ready.",
        "",
        "## Source Artifacts",
        "",
    ]
    lines.extend(f"- `{path}`" for path in source_paths)
    lines.extend(
        [
            "",
            "## Candidate Accounting",
            "",
            f"- Records accounted for: `{summary.get('records_accounted_for')}`",
            f"- Expected candidate count: `{summary.get('expected_candidate_count')}`",
            f"- Candidate count matches expected: `{summary.get('candidate_count_matches_expected')}`",
            f"- Official runner rows present in lookup packet: `{summary.get('official_runner_rows_present_count')}`",
            f"- Positions carrying cleaned dog names: `{summary.get('positions_with_clean_dog_name_count')}`",
            "",
            "## Bucket Counts",
            "",
        ]
    )
    lines.extend(f"- {bucket}: `{bucket_counts.get(bucket, 0)}`" for bucket in PRIMARY_BUCKETS)
    lines.extend(
        [
            "",
            "## Write Safety Counts",
            "",
        ]
    )
    lines.extend(f"- {key}: `{value}`" for key, value in write_counts.items())
    lines.extend(
        [
            "",
            "## Key Gates",
            "",
            f"- Safe correction lane candidates: `{summary.get('safe_correction_lane_candidates')}`",
            f"- Safe no-write candidates: `{summary.get('safe_no_write_candidates')}`",
            f"- Still blocked by box identity drift: `{summary.get('blocked_by_box_identity_drift')}`",
            f"- Blocked by missing DB rows/metadata: `{summary.get('blocked_by_missing_db_rows_or_metadata')}`",
            f"- Blocked by missing official names/parser/lookup failure: `{summary.get('blocked_by_missing_official_names_or_parser_failure')}`",
            f"- AP_K_2025-07-21_10 assertion: `{ap.get('status')}` bucket=`{ap.get('primary_bucket')}` present=`{ap.get('present')}`",
            "",
            "## DB Health",
            "",
            f"- PRAGMA quick_check before: `{before.get('quick_check')}`",
            f"- PRAGMA quick_check after: `{after.get('quick_check')}`",
            f"- Row counts unchanged: `{db_health.get('row_counts_unchanged')}`",
            f"- Write timestamps unchanged where checked: `{db_health.get('timestamps_unchanged')}`",
            f"- Official label rows/races before: `{_mapping(before.get('official_label_counts'))}`",
            f"- Official label rows/races after: `{_mapping(after.get('official_label_counts'))}`",
            f"- dog_race_data source counts before: `{_list(before.get('dog_data_source_counts'))}`",
            f"- race_metadata winner source counts before: `{_list(before.get('metadata_winner_source_counts'))}`",
            "",
            "## Recommendation",
            "",
            "Do not train yet. Do not promote yet. Do not use TGR yet. The next step should be a no-write rehearsal for the safest correction lane only. Label application should happen only after review, backup, and explicit approval.",
            "",
            "Clean official label volume remains below the promotion gate; prediction accuracy and box-bias remain the immediate blockers.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_full_outputs(output_dir: Path, packet: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _list(packet.get("records"))
    safe_records = [
        record
        for record in records
        if _mapping(record).get("write_safety_status") == "safe_review_candidate"
    ]
    unsafe_records = [
        record
        for record in records
        if _mapping(record).get("write_safety_status") != "safe_review_candidate"
    ]
    bucket_counts = _mapping(_mapping(packet.get("summary")).get("bucket_counts"))
    counts_payload = {
        "schema_version": "official_identity_reconciliation_bucket_counts_v1",
        "generated_at": packet.get("generated_at"),
        "bucket_counts": bucket_counts,
        "write_safety_status_counts": _mapping(
            _mapping(packet.get("summary")).get("write_safety_status_counts")
        ),
        "examples_by_bucket": _mapping(_mapping(packet.get("summary")).get("examples_by_bucket")),
        "top_reason_codes": _mapping(_mapping(packet.get("summary")).get("top_reason_codes")),
        "safe_correction_lane_candidates": _mapping(packet.get("summary")).get(
            "safe_correction_lane_candidates"
        ),
        "blocked_by_box_identity_drift": _mapping(packet.get("summary")).get(
            "blocked_by_box_identity_drift"
        ),
        "blocked_by_missing_db_rows_or_metadata": _mapping(packet.get("summary")).get(
            "blocked_by_missing_db_rows_or_metadata"
        ),
        "blocked_by_missing_official_names_or_parser_failure": _mapping(
            packet.get("summary")
        ).get("blocked_by_missing_official_names_or_parser_failure"),
    }
    (output_dir / "official_identity_reconciliation_full.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_full_summary(output_dir / "SUMMARY.md", packet)
    (output_dir / "safe_correction_lane_candidates.json").write_text(
        json.dumps(
            {
                "schema_version": "safe_correction_lane_candidates_v1",
                "generated_at": packet.get("generated_at"),
                "summary": {"count": len(safe_records)},
                "writes_performed": dict(WRITES_PERFORMED),
                "records": safe_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "unsafe_or_review_required.json").write_text(
        json.dumps(
            {
                "schema_version": "unsafe_or_review_required_v1",
                "generated_at": packet.get("generated_at"),
                "summary": {"count": len(unsafe_records)},
                "writes_performed": dict(WRITES_PERFORMED),
                "records": unsafe_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "bucket_counts.json").write_text(
        json.dumps(counts_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "db_health.json").write_text(
        json.dumps(packet.get("db_health"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_identity_reconciliation_packet(
    *,
    lookup_packet_path: Path,
    db_path: Path,
) -> dict[str, Any]:
    lookup_resolved = lookup_packet_path.expanduser().resolve()
    db_resolved = db_path.expanduser().resolve()
    lookup = _load_json(lookup_resolved)
    failures = []
    if lookup.get("schema_version") != LOOKUP_SCHEMA_VERSION:
        failures.append("lookup_schema_mismatch")
    writes = _mapping(lookup.get("writes_performed"))
    if any(key != "official_fetch" and value is not False for key, value in writes.items()):
        failures.append("lookup_has_forbidden_write_flags")

    with _connect_read_only(db_resolved) as conn:
        quick_check = conn.execute("PRAGMA quick_check").fetchone()
        if not quick_check or quick_check[0] != "ok":
            failures.append("db_quick_check_failed")
        races = [
            _reconcile_result(conn, result)
            for result in _list(lookup.get("results"))
            if _mapping(result).get("result_parse_ready") is True
        ]

    status_counts = Counter(str(race.get("status") or "DATA_MISSING") for race in races)
    summary = {
        "lookup_results_seen": len(_list(lookup.get("results"))),
        "identity_ready_count": len(races),
        "exact_identity_and_position_match_count": status_counts[
            "EXACT_IDENTITY_AND_POSITION_MATCH"
        ],
        "same_names_different_boxes_count": status_counts["BOX_IDENTITY_DRIFT"],
        "name_set_mismatch_count": status_counts["NAME_SET_MISMATCH"],
        "db_rows_missing_count": status_counts["DB_ROWS_MISSING"],
        "official_names_missing_count": status_counts["OFFICIAL_NAMES_MISSING"],
    }
    packet_status = (
        "NOT_READY" if failures
        else "REPORT_ONLY_IDENTITY_RECONCILED"
        if summary["identity_ready_count"] > 0
        and summary["identity_ready_count"] == summary["exact_identity_and_position_match_count"]
        else "REPORT_ONLY_IDENTITY_RECONCILIATION_REQUIRED"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": packet_status,
        "failures": failures,
        "source_evidence": {
            "lookup_packet": str(lookup_resolved),
            "db": str(db_resolved),
        },
        "summary": summary,
        "writes_performed": dict(WRITES_PERFORMED),
        "races": races,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookup-packet", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--output")
    parser.add_argument("--report")
    parser.add_argument("--full-output-dir")
    parser.add_argument(
        "--candidate-scope",
        choices=("parse-ready", "label-write-ready", "all"),
        default="parse-ready",
    )
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--source-artifact", action="append", default=[])
    parser.add_argument("--repo-root", default=str(Path.cwd()))
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.full_output_dir:
        packet = build_full_identity_reconciliation_packet(
            lookup_packet_path=Path(args.lookup_packet),
            db_path=Path(args.db),
            candidate_scope=args.candidate_scope,
            expected_count=args.expected_count,
            source_artifacts=list(args.source_artifact or []),
            repo_root=Path(args.repo_root),
        )
        _write_full_outputs(Path(args.full_output_dir), packet)
        print(json.dumps({"status": packet["status"], "summary": packet["summary"]}, indent=2))
        return 0 if not packet["failures"] else 2

    if not args.output or not args.report:
        raise SystemExit("--output and --report are required unless --full-output-dir is used")
    packet = build_identity_reconciliation_packet(
        lookup_packet_path=Path(args.lookup_packet),
        db_path=Path(args.db),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    _write_report(report, packet)
    print(json.dumps({"status": packet["status"], "summary": packet["summary"]}, indent=2))
    return 0 if not packet["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
