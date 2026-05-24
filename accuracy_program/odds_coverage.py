"""Read-only dog-level odds coverage analysis."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


def _open_readonly(db_path: str | os.PathLike[str]) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


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

        match_counts = _dict_row(
            conn.execute(
                f"""
                WITH win AS (
                    SELECT *
                    FROM live_odds
                    WHERE {win_where}
                ),
                m_rid_name AS (
                    SELECT win.id
                    FROM win
                    JOIN dog_race_data d
                      ON d.race_id = win.race_id
                     AND lower(trim(coalesce(d.dog_clean_name, d.dog_name))) =
                         lower(trim(coalesce(win.dog_clean_name, win.dog_name)))
                ),
                m_rid_box AS (
                    SELECT win.id
                    FROM win
                    JOIN dog_race_data d
                      ON d.race_id = win.race_id
                     AND d.box_number = win.box_number
                ),
                m_vdr_name AS (
                    SELECT win.id
                    FROM win
                    JOIN race_metadata rm
                      ON lower(trim(rm.venue)) = lower(trim(win.venue))
                     AND rm.race_date = win.race_date
                     AND rm.race_number = win.race_number
                    JOIN dog_race_data d
                      ON d.race_id = rm.race_id
                     AND lower(trim(coalesce(d.dog_clean_name, d.dog_name))) =
                         lower(trim(coalesce(win.dog_clean_name, win.dog_name)))
                ),
                m_vdr_box AS (
                    SELECT win.id
                    FROM win
                    JOIN race_metadata rm
                      ON lower(trim(rm.venue)) = lower(trim(win.venue))
                     AND rm.race_date = win.race_date
                     AND rm.race_number = win.race_number
                    JOIN dog_race_data d
                      ON d.race_id = rm.race_id
                     AND d.box_number = win.box_number
                )
                SELECT
                    (SELECT COUNT(DISTINCT id) FROM m_rid_name) AS race_id_name_matches,
                    (SELECT COUNT(DISTINCT id) FROM m_rid_box) AS race_id_box_matches,
                    (SELECT COUNT(DISTINCT id) FROM m_vdr_name) AS venue_date_race_name_matches,
                    (SELECT COUNT(DISTINCT id) FROM m_vdr_box) AS venue_date_race_box_matches
                """
            ).fetchone()
        )

        missing_reasons = [
            dict(row)
            for row in conn.execute(
                f"""
                WITH win AS (
                    SELECT *
                    FROM live_odds
                    WHERE {win_where}
                ),
                matched AS (
                    SELECT DISTINCT win.id
                    FROM win
                    JOIN dog_race_data d
                      ON d.race_id = win.race_id
                     AND (
                            lower(trim(coalesce(d.dog_clean_name, d.dog_name))) =
                            lower(trim(coalesce(win.dog_clean_name, win.dog_name)))
                         OR d.box_number = win.box_number
                     )
                ),
                rm AS (SELECT race_id FROM race_metadata),
                dr AS (SELECT DISTINCT race_id FROM dog_race_data)
                SELECT
                    CASE
                        WHEN win.race_id NOT IN (SELECT race_id FROM rm)
                            THEN 'no_race_metadata_race_id'
                        WHEN win.race_id NOT IN (SELECT race_id FROM dr)
                            THEN 'no_dog_rows_race_id'
                        WHEN win.box_number IS NULL
                         AND trim(coalesce(win.dog_clean_name, win.dog_name, '')) = ''
                            THEN 'no_box_or_dog_name'
                        ELSE 'dog_match_failed'
                    END AS missing_reason,
                    COUNT(*) AS rows,
                    COUNT(DISTINCT win.race_id) AS races
                FROM win
                WHERE win.id NOT IN (SELECT id FROM matched)
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
            SELECT timestamp
            FROM live_odds
            WHERE {win_where}
            """
        ).fetchall()

        now_dt = now or datetime.now()
        stale_rows = 0
        for row in current_rows:
            parsed = _parse_timestamp(row["timestamp"])
            if parsed is None:
                continue
            compare_now = now_dt
            if parsed.tzinfo is not None and compare_now.tzinfo is None:
                compare_now = compare_now.replace(tzinfo=parsed.tzinfo)
            age_hours = (compare_now - parsed).total_seconds() / 3600.0
            if age_hours > stale_after_hours:
                stale_rows += 1

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

    dog_level_win_rows = int(win_counts.get("dog_level_win_odds_rows") or 0)
    matched_by_name = int(match_counts.get("race_id_name_matches") or 0)
    matched_by_box = int(match_counts.get("race_id_box_matches") or 0)
    best_direct_match = max(matched_by_name, matched_by_box)
    return {
        "db_path": str(db_path),
        "current_only": current_only,
        "tables": tables,
        "counts": {**counts, **win_counts},
        "match_counts": match_counts,
        "missing_reasons": missing_reasons,
        "race_coverage": race_coverage,
        "stale_late_risks": {
            **timestamp_bounds,
            "stale_after_hours": stale_after_hours,
            "stale_current_win_rows": stale_rows,
            **late_risk,
        },
        "coverage_rates": {
            "race_id_name_match_rate": (
                matched_by_name / dog_level_win_rows if dog_level_win_rows else None
            ),
            "race_id_box_match_rate": (
                matched_by_box / dog_level_win_rows if dog_level_win_rows else None
            ),
            "best_direct_match_rate": (
                best_direct_match / dog_level_win_rows if dog_level_win_rows else None
            ),
        },
    }
