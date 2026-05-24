"""Read-only dog-level odds coverage analysis."""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

_DOG_PREFIX_RE = re.compile(r"^\s*\d{1,2}\s*[\.\):-]\s*")


def _open_readonly(db_path: str | os.PathLike[str]) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def normalize_dog_name(value: Any) -> str:
    """Return the deterministic dog-name key used for coverage-only joins."""

    raw = str(value or "").strip()
    for before, after in (
        ("\u201c", ""),
        ("\u201d", ""),
        ("\u2018", ""),
        ("\u2019", ""),
        ('"', ""),
        ("'", ""),
        ("`", ""),
        ("\u00a0", " "),
    ):
        raw = raw.replace(before, after)
    raw = _DOG_PREFIX_RE.sub("", raw)
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


def normalize_venue(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _register_sqlite_functions(conn: sqlite3.Connection) -> None:
    conn.create_function("norm_dog", 1, normalize_dog_name)
    conn.create_function("norm_venue", 1, normalize_venue)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _dict_row(row: sqlite3.Row | None) -> dict[str, Any]:
    return dict(row) if row is not None else {}


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0] or 0) if row else 0


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    for candidate in (raw, raw.replace("Z", "+00:00")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw[:19], fmt)
        except ValueError:
            continue
    return None


def _age_hours(now_dt: datetime, timestamp: datetime) -> float:
    compare_now = now_dt
    if timestamp.tzinfo is not None and compare_now.tzinfo is None:
        compare_now = compare_now.replace(tzinfo=timestamp.tzinfo)
    elif timestamp.tzinfo is None and compare_now.tzinfo is not None:
        compare_now = compare_now.replace(tzinfo=None)
    return (compare_now - timestamp).total_seconds() / 3600.0


def _timestamp_quality(
    rows: list[sqlite3.Row],
    *,
    now: datetime,
    stale_after_hours: float,
) -> dict[str, Any]:
    null_rows = 0
    invalid_rows = 0
    stale_rows = 0
    ages: list[float] = []
    for row in rows:
        raw = row["timestamp"]
        if raw is None or str(raw).strip() == "":
            null_rows += 1
            continue
        parsed = _parse_timestamp(raw)
        if parsed is None:
            invalid_rows += 1
            continue
        age = _age_hours(now, parsed)
        ages.append(age)
        if age > stale_after_hours:
            stale_rows += 1

    age_stats = {
        "min": round(min(ages), 6) if ages else None,
        "max": round(max(ages), 6) if ages else None,
        "avg": round(sum(ages) / len(ages), 6) if ages else None,
    }
    return {
        "rows_checked": len(rows),
        "timestamped_rows": len(ages),
        "null_timestamp_rows": null_rows,
        "invalid_timestamp_rows": invalid_rows,
        "stale_rows": stale_rows,
        "stale_after_hours": stale_after_hours,
        "age_hours_at_report_time": age_stats,
    }


def _current_win_where(current_only: bool) -> str:
    current_clause = "AND (is_current = 1 OR is_current IS NULL)" if current_only else ""
    return f"""
        lower(coalesce(market_type, 'win')) = 'win'
        AND odds_decimal IS NOT NULL
        AND odds_decimal > 1
        {current_clause}
    """


def analyze_odds_coverage(
    db_path: str | os.PathLike[str],
    *,
    current_only: bool = True,
    stale_after_hours: float = 6.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return dog-level odds coverage metrics from a SQLite DB.

    The DB is opened in SQLite read-only mode with query_only enabled. The
    function never scrapes and never writes.
    """

    with _open_readonly(db_path) as conn:
        _register_sqlite_functions(conn)
        tables = {
            table: _table_exists(conn, table)
            for table in ("live_odds", "odds_history", "race_metadata", "dog_race_data")
        }
        if not all(tables.values()):
            return {
                "db_path": str(db_path),
                "current_only": current_only,
                "tables": tables,
                "error": "required table missing",
            }

        win_where = _current_win_where(current_only)
        counts = {
            "live_odds_rows": _scalar(conn, "SELECT COUNT(*) FROM live_odds"),
            "odds_history_rows": _scalar(conn, "SELECT COUNT(*) FROM odds_history"),
            "live_odds_races": _scalar(conn, "SELECT COUNT(DISTINCT race_id) FROM live_odds"),
            "odds_history_races": _scalar(conn, "SELECT COUNT(DISTINCT race_id) FROM odds_history"),
            "dog_rows": _scalar(conn, "SELECT COUNT(*) FROM dog_race_data"),
            "race_rows": _scalar(conn, "SELECT COUNT(*) FROM race_metadata"),
        }

        win_counts = _dict_row(
            conn.execute(
                f"""
                SELECT
                    COUNT(*) AS dog_level_win_odds_rows,
                    COUNT(DISTINCT race_id) AS races_with_dog_level_win_odds
                FROM live_odds
                WHERE {win_where}
                """
            ).fetchone()
        )
        odds_history_dog_counts = _dict_row(
            conn.execute(
                """
                SELECT
                    COUNT(*) AS odds_history_dog_level_rows,
                    COUNT(DISTINCT race_id) AS odds_history_dog_level_races
                FROM odds_history
                WHERE odds_decimal IS NOT NULL
                  AND odds_decimal > 1
                  AND trim(coalesce(dog_clean_name, '')) <> ''
                """
            ).fetchone()
        )

        match_counts = _dict_row(
            conn.execute(
                f"""
                WITH win AS (
                    SELECT *,
                           norm_dog(coalesce(dog_clean_name, dog_name)) AS dog_key,
                           CASE WHEN trim(coalesce(dog_clean_name, dog_name, '')) <> ''
                                THEN 1 ELSE 0 END AS has_name,
                           CASE WHEN box_number IS NOT NULL THEN 1 ELSE 0 END AS has_box
                    FROM live_odds
                    WHERE {win_where}
                ),
                field AS (
                    SELECT id, race_id, box_number,
                           norm_dog(coalesce(dog_clean_name, dog_name)) AS dog_key
                    FROM dog_race_data
                ),
                summary AS (
                    SELECT
                        win.id,
                        win.has_name,
                        win.has_box,
                        COUNT(DISTINCT rm.race_id) AS metadata_race_id_matches,
                        COUNT(DISTINCT f.id) AS dog_rows_for_race_id,
                        COUNT(DISTINCT CASE
                            WHEN win.dog_key <> '' AND f.dog_key = win.dog_key
                            THEN f.id END) AS race_id_name_candidates,
                        COUNT(DISTINCT CASE
                            WHEN win.box_number IS NOT NULL AND f.box_number = win.box_number
                            THEN f.id END) AS race_id_box_candidates,
                        COUNT(DISTINCT CASE
                            WHEN win.dog_key <> ''
                             AND win.box_number IS NOT NULL
                             AND f.dog_key = win.dog_key
                             AND f.box_number = win.box_number
                            THEN f.id END) AS race_id_box_name_candidates
                    FROM win
                    LEFT JOIN field f ON f.race_id = win.race_id
                    LEFT JOIN race_metadata rm ON rm.race_id = win.race_id
                    GROUP BY win.id, win.has_name, win.has_box
                ),
                m_vdr_name AS (
                    SELECT win.id
                    FROM win
                    JOIN race_metadata rm
                      ON norm_venue(rm.venue) = norm_venue(win.venue)
                     AND rm.race_date = win.race_date
                     AND CAST(rm.race_number AS INTEGER) = CAST(win.race_number AS INTEGER)
                    JOIN field f
                      ON f.race_id = rm.race_id
                     AND f.dog_key = win.dog_key
                    WHERE win.dog_key <> ''
                ),
                m_vdr_box AS (
                    SELECT win.id
                    FROM win
                    JOIN race_metadata rm
                      ON norm_venue(rm.venue) = norm_venue(win.venue)
                     AND rm.race_date = win.race_date
                     AND CAST(rm.race_number AS INTEGER) = CAST(win.race_number AS INTEGER)
                    JOIN field f
                      ON f.race_id = rm.race_id
                     AND f.box_number = win.box_number
                    WHERE win.box_number IS NOT NULL
                )
                SELECT
                    SUM(CASE WHEN metadata_race_id_matches > 0 THEN 1 ELSE 0 END)
                        AS race_id_metadata_matches,
                    SUM(CASE WHEN dog_rows_for_race_id > 0 THEN 1 ELSE 0 END)
                        AS race_id_field_matches,
                    SUM(CASE WHEN race_id_name_candidates > 0 THEN 1 ELSE 0 END)
                        AS race_id_name_matches,
                    SUM(CASE WHEN race_id_box_candidates > 0 THEN 1 ELSE 0 END)
                        AS race_id_box_matches,
                    SUM(CASE WHEN race_id_box_name_candidates > 0 THEN 1 ELSE 0 END)
                        AS race_id_box_name_matches,
                    SUM(CASE
                        WHEN has_name = 1 AND has_box = 1
                         AND race_id_box_name_candidates = 1
                        THEN 1 ELSE 0 END) AS safe_race_id_box_name_matches,
                    SUM(CASE
                        WHEN has_name = 1 AND has_box = 0
                         AND race_id_name_candidates = 1
                        THEN 1 ELSE 0 END) AS safe_race_id_name_only_matches,
                    SUM(CASE
                        WHEN has_name = 0 AND has_box = 1
                         AND race_id_box_candidates = 1
                        THEN 1 ELSE 0 END) AS safe_race_id_box_only_matches,
                    SUM(CASE
                        WHEN race_id_box_name_candidates > 1 THEN 1 ELSE 0 END)
                        AS race_id_box_name_ambiguous_rows,
                    SUM(CASE
                        WHEN race_id_name_candidates > 1 THEN 1 ELSE 0 END)
                        AS race_id_name_ambiguous_rows,
                    SUM(CASE
                        WHEN race_id_box_candidates > 1 THEN 1 ELSE 0 END)
                        AS race_id_box_ambiguous_rows,
                    (SELECT COUNT(DISTINCT id) FROM m_vdr_name) AS venue_date_race_name_matches,
                    (SELECT COUNT(DISTINCT id) FROM m_vdr_box) AS venue_date_race_box_matches
                FROM summary
                """
            ).fetchone()
        )
        safe_match_counts = _dict_row(
            conn.execute(
                f"""
                WITH win AS (
                    SELECT *,
                           norm_dog(coalesce(dog_clean_name, dog_name)) AS dog_key,
                           CASE WHEN trim(coalesce(dog_clean_name, dog_name, '')) <> ''
                                THEN 1 ELSE 0 END AS has_name,
                           CASE WHEN box_number IS NOT NULL THEN 1 ELSE 0 END AS has_box
                    FROM live_odds
                    WHERE {win_where}
                ),
                field AS (
                    SELECT id, race_id, box_number,
                           norm_dog(coalesce(dog_clean_name, dog_name)) AS dog_key
                    FROM dog_race_data
                ),
                summary AS (
                    SELECT
                        win.id,
                        win.has_name,
                        win.has_box,
                        COUNT(DISTINCT CASE
                            WHEN win.dog_key <> '' AND f.dog_key = win.dog_key
                            THEN f.id END) AS name_candidates,
                        COUNT(DISTINCT CASE
                            WHEN win.box_number IS NOT NULL AND f.box_number = win.box_number
                            THEN f.id END) AS box_candidates,
                        COUNT(DISTINCT CASE
                            WHEN win.dog_key <> ''
                             AND win.box_number IS NOT NULL
                             AND f.dog_key = win.dog_key
                             AND f.box_number = win.box_number
                            THEN f.id END) AS strict_candidates
                    FROM win
                    LEFT JOIN field f ON f.race_id = win.race_id
                    GROUP BY win.id, win.has_name, win.has_box
                )
                SELECT
                    SUM(CASE
                        WHEN has_name = 1 AND has_box = 1 AND strict_candidates = 1
                        THEN 1 ELSE 0 END) AS safe_direct_identity_matches,
                    SUM(CASE
                        WHEN has_name = 1 AND has_box = 1
                         AND strict_candidates = 0
                         AND name_candidates > 0
                         AND box_candidates > 0
                        THEN 1 ELSE 0 END) AS dog_name_box_conflict_rows,
                    SUM(CASE
                        WHEN has_name = 1 AND has_box = 1 AND strict_candidates > 1
                        THEN 1 ELSE 0 END) AS ambiguous_strict_identity_rows
                FROM summary
                """
            ).fetchone()
        )

        missing_reasons = [
            dict(row)
            for row in conn.execute(
                f"""
                WITH win AS (
                    SELECT *,
                           norm_dog(coalesce(dog_clean_name, dog_name)) AS dog_key,
                           CASE WHEN trim(coalesce(dog_clean_name, dog_name, '')) <> ''
                                THEN 1 ELSE 0 END AS has_name,
                           CASE WHEN box_number IS NOT NULL THEN 1 ELSE 0 END AS has_box
                    FROM live_odds
                    WHERE {win_where}
                ),
                field AS (
                    SELECT id, race_id, box_number,
                           norm_dog(coalesce(dog_clean_name, dog_name)) AS dog_key
                    FROM dog_race_data
                ),
                summary AS (
                    SELECT
                        win.id,
                        win.race_id,
                        win.has_name,
                        win.has_box,
                        COUNT(DISTINCT rm.race_id) AS metadata_race_id_matches,
                        COUNT(DISTINCT f.id) AS dog_rows_for_race_id,
                        COUNT(DISTINCT CASE
                            WHEN win.dog_key <> '' AND f.dog_key = win.dog_key
                            THEN f.id END) AS name_candidates,
                        COUNT(DISTINCT CASE
                            WHEN win.box_number IS NOT NULL AND f.box_number = win.box_number
                            THEN f.id END) AS box_candidates,
                        COUNT(DISTINCT CASE
                            WHEN win.dog_key <> ''
                             AND win.box_number IS NOT NULL
                             AND f.dog_key = win.dog_key
                             AND f.box_number = win.box_number
                            THEN f.id END) AS strict_candidates
                    FROM win
                    LEFT JOIN field f ON f.race_id = win.race_id
                    LEFT JOIN race_metadata rm ON rm.race_id = win.race_id
                    GROUP BY win.id, win.race_id, win.has_name, win.has_box
                ),
                safe AS (
                    SELECT id
                    FROM summary
                    WHERE (has_name = 1 AND has_box = 1 AND strict_candidates = 1)
                       OR (has_name = 1 AND has_box = 0 AND name_candidates = 1)
                       OR (has_name = 0 AND has_box = 1 AND box_candidates = 1)
                )
                SELECT
                    CASE
                        WHEN metadata_race_id_matches = 0
                            THEN 'no_race_metadata_race_id'
                        WHEN dog_rows_for_race_id = 0
                            THEN 'no_dog_rows_race_id'
                        WHEN has_box = 0 AND has_name = 0
                            THEN 'no_box_or_dog_name'
                        WHEN strict_candidates > 1
                            THEN 'ambiguous_race_id_box_name'
                        WHEN has_name = 1 AND name_candidates > 1
                            THEN 'ambiguous_race_id_name'
                        WHEN has_box = 1 AND box_candidates > 1
                            THEN 'ambiguous_race_id_box'
                        WHEN has_name = 1 AND has_box = 1
                         AND strict_candidates = 0
                         AND name_candidates > 0
                         AND box_candidates > 0
                            THEN 'dog_name_box_conflict'
                        ELSE 'dog_match_failed'
                    END AS missing_reason,
                    COUNT(*) AS rows,
                    COUNT(DISTINCT race_id) AS races
                FROM summary
                WHERE id NOT IN (SELECT id FROM safe)
                GROUP BY missing_reason
                ORDER BY rows DESC, missing_reason ASC
                """
            ).fetchall()
        ]

        race_coverage = _dict_row(
            conn.execute(
                f"""
                WITH win AS (
                    SELECT *
                    FROM live_odds
                    WHERE {win_where}
                ),
                race_field AS (
                    SELECT race_id, COUNT(*) AS dog_rows
                    FROM dog_race_data
                    GROUP BY race_id
                ),
                race_odds AS (
                    SELECT race_id, COUNT(*) AS odds_rows
                    FROM win
                    GROUP BY race_id
                )
                SELECT
                    COUNT(*) AS odds_races,
                    SUM(CASE WHEN rf.dog_rows IS NOT NULL THEN 1 ELSE 0 END)
                        AS races_with_field_by_race_id,
                    ROUND(AVG(CASE WHEN rf.dog_rows IS NOT NULL
                        THEN 1.0 * ro.odds_rows / rf.dog_rows END), 6)
                        AS avg_row_coverage_by_race_id,
                    SUM(CASE WHEN ro.odds_rows >= COALESCE(rf.dog_rows, 9999)
                        THEN 1 ELSE 0 END) AS races_full_or_more_by_race_id
                FROM race_odds ro
                LEFT JOIN race_field rf ON rf.race_id = ro.race_id
                """
            ).fetchone()
        )

        timestamp_bounds = _dict_row(
            conn.execute(
                f"""
                SELECT MIN(timestamp) AS min_timestamp, MAX(timestamp) AS max_timestamp
                FROM live_odds
                WHERE {win_where}
                """
            ).fetchone()
        )
        current_rows = conn.execute(
            f"""
            SELECT id, timestamp
            FROM live_odds
            WHERE {win_where}
            """
        ).fetchall()
        now_dt = now or datetime.now()
        live_timestamp_quality = _timestamp_quality(
            current_rows,
            now=now_dt,
            stale_after_hours=stale_after_hours,
        )
        history_timestamp_rows = conn.execute(
            """
            SELECT id, timestamp
            FROM odds_history
            WHERE odds_decimal IS NOT NULL
              AND odds_decimal > 1
              AND trim(coalesce(dog_clean_name, '')) <> ''
            """
        ).fetchall()
        history_timestamp_quality = _timestamp_quality(
            history_timestamp_rows,
            now=now_dt,
            stale_after_hours=stale_after_hours,
        )

        late_risk = _dict_row(
            conn.execute(
                f"""
                WITH win AS (
                    SELECT *
                    FROM live_odds
                    WHERE {win_where}
                )
                SELECT
                    SUM(CASE
                        WHEN rm.start_datetime IS NOT NULL
                         AND win.timestamp IS NOT NULL
                         AND datetime(win.timestamp) > datetime(rm.start_datetime)
                        THEN 1 ELSE 0 END) AS rows_after_start_datetime,
                    SUM(CASE
                        WHEN rm.start_datetime IS NOT NULL
                        THEN 1 ELSE 0 END) AS rows_with_start_datetime
                FROM win
                LEFT JOIN race_metadata rm ON rm.race_id = win.race_id
                """
            ).fetchone()
        )
        source_counts = {
            "live_odds": [
                dict(row)
                for row in conn.execute(
                    f"""
                    SELECT coalesce(source, 'unknown') AS source, COUNT(*) AS rows
                    FROM live_odds
                    WHERE {win_where}
                    GROUP BY coalesce(source, 'unknown')
                    ORDER BY rows DESC, source ASC
                    """
                ).fetchall()
            ],
            "odds_history": [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT coalesce(source, 'unknown') AS source, COUNT(*) AS rows
                    FROM odds_history
                    WHERE odds_decimal IS NOT NULL
                      AND odds_decimal > 1
                      AND trim(coalesce(dog_clean_name, '')) <> ''
                    GROUP BY coalesce(source, 'unknown')
                    ORDER BY rows DESC, source ASC
                    """
                ).fetchall()
            ],
        }
        vdr_mismatch_cases = [
            dict(row)
            for row in conn.execute(
                f"""
                WITH win AS (
                    SELECT *
                    FROM live_odds
                    WHERE {win_where}
                ),
                cases AS (
                    SELECT
                        'race_id_metadata_fields_mismatch' AS mismatch_type,
                        win.id AS odds_id,
                        win.race_id AS odds_race_id,
                        win.venue AS odds_venue,
                        win.race_date AS odds_race_date,
                        win.race_number AS odds_race_number,
                        rm.race_id AS metadata_race_id,
                        rm.venue AS metadata_venue,
                        rm.race_date AS metadata_race_date,
                        rm.race_number AS metadata_race_number
                    FROM win
                    JOIN race_metadata rm ON rm.race_id = win.race_id
                    WHERE (
                           win.venue IS NOT NULL
                       AND rm.venue IS NOT NULL
                       AND norm_venue(win.venue) <> norm_venue(rm.venue)
                    ) OR (
                           win.race_date IS NOT NULL
                       AND rm.race_date IS NOT NULL
                       AND win.race_date <> rm.race_date
                    ) OR (
                           win.race_number IS NOT NULL
                       AND rm.race_number IS NOT NULL
                       AND CAST(win.race_number AS INTEGER)
                           <> CAST(rm.race_number AS INTEGER)
                    )

                    UNION ALL

                    SELECT
                        'venue_date_race_resolves_different_race_id' AS mismatch_type,
                        win.id AS odds_id,
                        win.race_id AS odds_race_id,
                        win.venue AS odds_venue,
                        win.race_date AS odds_race_date,
                        win.race_number AS odds_race_number,
                        rm.race_id AS metadata_race_id,
                        rm.venue AS metadata_venue,
                        rm.race_date AS metadata_race_date,
                        rm.race_number AS metadata_race_number
                    FROM win
                    JOIN race_metadata rm
                      ON norm_venue(rm.venue) = norm_venue(win.venue)
                     AND rm.race_date = win.race_date
                     AND CAST(rm.race_number AS INTEGER)
                         = CAST(win.race_number AS INTEGER)
                    WHERE win.race_id IS NULL OR rm.race_id <> win.race_id
                )
                SELECT *
                FROM cases
                ORDER BY mismatch_type, odds_id
                LIMIT 25
                """
            ).fetchall()
        ]
        vdr_mismatch_counts = [
            dict(row)
            for row in conn.execute(
                f"""
                WITH win AS (
                    SELECT *
                    FROM live_odds
                    WHERE {win_where}
                ),
                cases AS (
                    SELECT 'race_id_metadata_fields_mismatch' AS mismatch_type, win.id
                    FROM win
                    JOIN race_metadata rm ON rm.race_id = win.race_id
                    WHERE (
                           win.venue IS NOT NULL
                       AND rm.venue IS NOT NULL
                       AND norm_venue(win.venue) <> norm_venue(rm.venue)
                    ) OR (
                           win.race_date IS NOT NULL
                       AND rm.race_date IS NOT NULL
                       AND win.race_date <> rm.race_date
                    ) OR (
                           win.race_number IS NOT NULL
                       AND rm.race_number IS NOT NULL
                       AND CAST(win.race_number AS INTEGER)
                           <> CAST(rm.race_number AS INTEGER)
                    )

                    UNION ALL

                    SELECT 'venue_date_race_resolves_different_race_id' AS mismatch_type,
                           win.id
                    FROM win
                    JOIN race_metadata rm
                      ON norm_venue(rm.venue) = norm_venue(win.venue)
                     AND rm.race_date = win.race_date
                     AND CAST(rm.race_number AS INTEGER)
                         = CAST(win.race_number AS INTEGER)
                    WHERE win.race_id IS NULL OR rm.race_id <> win.race_id
                )
                SELECT mismatch_type, COUNT(DISTINCT id) AS rows
                FROM cases
                GROUP BY mismatch_type
                ORDER BY mismatch_type
                """
            ).fetchall()
        ]

    dog_level_win_rows = int(win_counts.get("dog_level_win_odds_rows") or 0)
    matched_by_name = int(match_counts.get("race_id_name_matches") or 0)
    matched_by_box = int(match_counts.get("race_id_box_matches") or 0)
    matched_strict = int(match_counts.get("race_id_box_name_matches") or 0)
    safe_direct = int(safe_match_counts.get("safe_direct_identity_matches") or 0)
    best_direct_match = max(matched_by_name, matched_by_box)
    return {
        "db_path": str(db_path),
        "current_only": current_only,
        "tables": tables,
        "counts": {**counts, **win_counts, **odds_history_dog_counts},
        "match_counts": match_counts,
        "safe_match_counts": safe_match_counts,
        "missing_reasons": missing_reasons,
        "venue_date_race_mismatches": {
            "counts": vdr_mismatch_counts,
            "sample_cases": vdr_mismatch_cases,
        },
        "race_coverage": race_coverage,
        "stale_late_risks": {
            **timestamp_bounds,
            "stale_after_hours": stale_after_hours,
            "stale_current_win_rows": live_timestamp_quality["stale_rows"],
            **late_risk,
        },
        "timestamp_quality": {
            "report_reference_time": now_dt.isoformat(),
            "live_odds_current_win": live_timestamp_quality,
            "odds_history_dog_level": history_timestamp_quality,
        },
        "source_provenance": source_counts,
        "coverage_rates": {
            "race_id_metadata_match_rate": (
                int(match_counts.get("race_id_metadata_matches") or 0)
                / dog_level_win_rows
                if dog_level_win_rows
                else None
            ),
            "race_id_field_match_rate": (
                int(match_counts.get("race_id_field_matches") or 0) / dog_level_win_rows
                if dog_level_win_rows
                else None
            ),
            "race_id_name_match_rate": (
                matched_by_name / dog_level_win_rows if dog_level_win_rows else None
            ),
            "race_id_box_match_rate": (
                matched_by_box / dog_level_win_rows if dog_level_win_rows else None
            ),
            "race_id_box_name_match_rate": (
                matched_strict / dog_level_win_rows if dog_level_win_rows else None
            ),
            "safe_direct_identity_match_rate": (
                safe_direct / dog_level_win_rows if dog_level_win_rows else None
            ),
            "best_direct_match_rate": (
                best_direct_match / dog_level_win_rows if dog_level_win_rows else None
            ),
        },
    }
