"""Read-only dog-level odds coverage analysis."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from accuracy_program.snapshots import (
    assert_no_result_fields,
    classify_odds_snapshot_for_ev,
)

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


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


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


def _snapshot_files(paths: Iterable[str | os.PathLike[str]]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("**/*.json")))
        elif path.is_file():
            files.append(path)
    return files


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def summarize_read_only_odds_coverage_report(
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Normalize daemon odds coverage into one report-only readiness shape."""

    report = report or {}
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else report
    dog_level_rows = int(summary.get("dog_level_win_odds_rows") or 0)
    source_url_missing = int(summary.get("source_url_rows_missing") or 0)
    stale_rows = int(summary.get("stale_current_win_rows") or 0)
    blocker_counts: dict[str, int] = {}
    if not report:
        blocker_counts["odds_coverage_report_missing"] = 1
    if dog_level_rows <= 0:
        blocker_counts["no_dog_level_win_odds_rows"] = 1
    if source_url_missing > 0:
        blocker_counts["missing_source_url_rows"] = source_url_missing
    if stale_rows > 0:
        blocker_counts["stale_current_win_rows"] = stale_rows

    if not report:
        next_action = "WAIT_FOR_DAEMON_ODDS_COVERAGE_DIAGNOSTIC"
    elif dog_level_rows <= 0:
        next_action = "COLLECT_EXACT_DOG_LEVEL_WIN_ODDS_REPORT_ONLY"
    elif source_url_missing > 0 and stale_rows > 0:
        next_action = "CAPTURE_FRESH_DOG_LEVEL_ODDS_WITH_SOURCE_URLS"
    elif source_url_missing > 0:
        next_action = "CAPTURE_ODDS_SOURCE_URL_PROVENANCE"
    elif stale_rows > 0:
        next_action = "REFRESH_DOG_LEVEL_ODDS_WITHIN_TTL"
    else:
        next_action = "READY_FOR_REPORT_ONLY_ODDS_SNAPSHOT_JOIN_NO_EV_ACTION"

    return {
        "status": summary.get("status"),
        "mode": summary.get("mode"),
        "readiness_status": (
            "ODDS_COVERAGE_READY_REPORT_ONLY_EV_DISABLED"
            if not blocker_counts
            else "ODDS_COVERAGE_BLOCKED_REPORT_ONLY_EV_DISABLED"
        ),
        "blocker_counts": blocker_counts,
        "next_action": next_action,
        "dog_level_win_odds_rows": dog_level_rows,
        "live_odds_rows": int(summary.get("live_odds_rows") or 0),
        "live_odds_races": int(summary.get("live_odds_races") or 0),
        "odds_history_rows": int(summary.get("odds_history_rows") or 0),
        "races_with_dog_level_win_odds": int(
            summary.get("races_with_dog_level_win_odds") or 0
        ),
        "safe_direct_identity_matches": int(
            summary.get("safe_direct_identity_matches") or 0
        ),
        "safe_direct_identity_match_rate": summary.get("safe_direct_identity_match_rate"),
        "source_url_rows_checked": int(summary.get("source_url_rows_checked") or 0),
        "source_url_rows_missing": source_url_missing,
        "stale_current_win_rows": stale_rows,
        "stale_after_hours": summary.get("stale_after_hours"),
        "source_provenance": summary.get("source_provenance") or {},
        "odds_capture_performed": bool(summary.get("odds_capture_performed")),
        "odds_used_for_shadow_scoring": bool(summary.get("odds_used_for_shadow_scoring")),
        "shadow_model_input": bool(summary.get("shadow_model_input")),
        "db_write": bool(summary.get("db_write")),
        "ev_action": bool(summary.get("ev_action")),
        "betting_action": bool(summary.get("betting_action")),
    }


def _runner_null_ev_reason(
    runner: Mapping[str, Any],
    odds_eligibility: Mapping[str, Any],
) -> str:
    if runner.get("ev_win") is not None:
        return "ev_win_present"
    return str(
        runner.get("odds_exclusion_reason")
        or odds_eligibility.get("odds_exclusion_reason")
        or odds_eligibility.get("odds_match_status")
        or "unknown"
    )


def analyze_snapshot_odds_coverage(
    snapshot_paths: Iterable[str | os.PathLike[str]],
) -> dict[str, Any]:
    """Return leakage-safe odds/EV diagnostics from result-free snapshots."""

    files = _snapshot_files(snapshot_paths)
    if not files:
        return {
            "status": "DATA_MISSING",
            "reason": "no_snapshot_files_found",
            "snapshot_files": 0,
            "runner_rows": 0,
            "null_ev_reason_rows": [],
        }

    rejected: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    provenance_counts: Counter[str] = Counter()
    match_method_counts: Counter[str] = Counter()
    null_ev_counts: Counter[str] = Counter()
    race_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for path in files:
        try:
            snapshot = json.loads(path.read_text(encoding="utf-8"))
            assert_no_result_fields(snapshot)
        except Exception as exc:
            rejected.append({"path": str(path), "reason": str(exc)})
            continue

        race_id = str(snapshot.get("race_id") or "")
        prediction_timestamp = snapshot.get("prediction_timestamp")
        freeze_timestamp = snapshot.get("feature_freeze_timestamp")
        jump_time = snapshot.get("jump_datetime") or snapshot.get("jump_time")
        for runner in snapshot.get("predictions") or []:
            if not isinstance(runner, Mapping):
                continue
            odds_snapshot = (
                runner.get("odds_snapshot")
                if isinstance(runner.get("odds_snapshot"), Mapping)
                else {}
            )
            eligibility = classify_odds_snapshot_for_ev(
                runner,
                odds_snapshot,
                snapshot_race_id=race_id,
            )
            status = str(
                runner.get("odds_match_status") or eligibility.get("odds_match_status") or "unknown"
            )
            exclusion = str(
                runner.get("odds_exclusion_reason")
                or eligibility.get("odds_exclusion_reason")
                or "none"
            )
            provenance_status = str(
                runner.get("odds_provenance_status")
                or eligibility.get("odds_provenance_status")
                or "unknown"
            )
            method = str(
                runner.get("odds_match_method")
                or eligibility.get("odds_match_method")
                or "DATA_MISSING"
            )
            null_ev_reason = _runner_null_ev_reason(runner, eligibility)
            record = {
                "snapshot_path": str(path),
                "race_id": race_id,
                "dog_name": runner.get("dog_name") or runner.get("dog_clean_name"),
                "box_number": runner.get("box_number"),
                "odds_decimal": runner.get("odds") or odds_snapshot.get("market_odds_win"),
                "ev_win": runner.get("ev_win"),
                "odds_match_status": status,
                "odds_match_method": method,
                "odds_exclusion_reason": exclusion,
                "odds_provenance_status": provenance_status,
                "null_ev_reason": null_ev_reason,
                "odds_timestamp": odds_snapshot.get("odds_timestamp")
                or runner.get("odds_timestamp"),
                "prediction_timestamp": prediction_timestamp,
                "feature_freeze_timestamp": freeze_timestamp,
                "jump_time": jump_time,
                "odds_age_seconds_at_prediction": odds_snapshot.get(
                    "odds_age_seconds_at_prediction"
                ),
                "odds_captured_before_prediction": odds_snapshot.get(
                    "odds_captured_before_prediction"
                ),
                "odds_captured_before_feature_freeze": odds_snapshot.get(
                    "odds_captured_before_feature_freeze"
                ),
                "odds_captured_before_jump": odds_snapshot.get("odds_captured_before_jump"),
                "odds_stale_at_prediction": odds_snapshot.get("odds_stale_at_prediction"),
                "odds_source": runner.get("odds_source")
                or (
                    odds_snapshot.get("odds_provenance", {}).get("source")
                    if isinstance(odds_snapshot.get("odds_provenance"), Mapping)
                    else None
                ),
                "odds_source_url": (
                    odds_snapshot.get("odds_provenance", {}).get("source_url")
                    if isinstance(odds_snapshot.get("odds_provenance"), Mapping)
                    else None
                ),
            }
            rows.append(record)
            race_rows[race_id].append(record)
            status_counts[status] += 1
            exclusion_counts[exclusion] += 1
            provenance_counts[provenance_status] += 1
            match_method_counts[method] += 1
            null_ev_counts[null_ev_reason] += 1

    complete_valid_races = 0
    partial_valid_races = 0
    no_valid_races = 0
    for runners in race_rows.values():
        valid = [row for row in runners if row["odds_match_status"] == "valid_pre_jump_dog_odds"]
        if valid and len(valid) == len(runners):
            complete_valid_races += 1
        elif valid:
            partial_valid_races += 1
        else:
            no_valid_races += 1

    valid_rows = status_counts.get("valid_pre_jump_dog_odds", 0)
    return {
        "status": "SUCCESS" if not rejected else "PARTIAL",
        "snapshot_files": len(files),
        "snapshots_rejected": len(rejected),
        "rejected_snapshots": rejected,
        "runner_rows": len(rows),
        "valid_pre_jump_dog_odds_rows": valid_rows,
        "ev_eligibility_rows": valid_rows,
        "ev_win_non_null_rows": sum(1 for row in rows if row.get("ev_win") is not None),
        "rows_with_null_ev": sum(1 for row in rows if row.get("ev_win") is None),
        "odds_coverage_rate": valid_rows / len(rows) if rows else None,
        "races": len(race_rows),
        "races_with_complete_valid_odds": complete_valid_races,
        "races_with_partial_valid_odds": partial_valid_races,
        "races_with_no_valid_odds": no_valid_races,
        "odds_match_status_distribution": _counter_dict(status_counts),
        "odds_exclusion_reason_distribution": _counter_dict(exclusion_counts),
        "odds_provenance_status_distribution": _counter_dict(provenance_counts),
        "odds_match_method_distribution": _counter_dict(match_method_counts),
        "null_ev_reason_distribution": _counter_dict(null_ev_counts),
        "stale_odds_rows": status_counts.get("stale_beyond_ttl", 0),
        "missing_timestamp_rows": status_counts.get("missing_timestamp", 0),
        "timestamp_after_prediction_rows": status_counts.get("timestamp_after_prediction", 0),
        "timestamp_after_feature_freeze_rows": status_counts.get(
            "timestamp_after_feature_freeze", 0
        ),
        "timestamp_after_jump_rows": status_counts.get("timestamp_after_jump", 0),
        "null_ev_reason_rows": rows,
    }


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
        live_odds_columns = _table_columns(conn, "live_odds")
        if "source_url" in live_odds_columns:
            source_url_quality = _dict_row(
                conn.execute(
                    f"""
                    SELECT
                        COUNT(*) AS rows_checked,
                        SUM(CASE
                            WHEN trim(coalesce(source_url, '')) <> ''
                            THEN 1 ELSE 0 END) AS rows_with_source_url,
                        SUM(CASE
                            WHEN trim(coalesce(source_url, '')) = ''
                            THEN 1 ELSE 0 END) AS rows_missing_source_url,
                        SUM(CASE
                            WHEN lower(coalesce(source_url, '')) LIKE '%result%'
                              OR lower(coalesce(source_url, '')) LIKE '%dividend%'
                              OR lower(coalesce(source_url, '')) LIKE '%payout%'
                              OR lower(coalesce(source_url, '')) LIKE '%starting-price%'
                              OR lower(coalesce(source_url, '')) LIKE '%starting_price%'
                              OR lower(coalesce(source_url, '')) LIKE '%startingprice%'
                              OR lower(coalesce(source_url, '')) LIKE '%/sp/%'
                              OR lower(coalesce(source_url, '')) LIKE '%/sp?%'
                              OR lower(coalesce(source_url, '')) LIKE '%/sp#%'
                              OR lower(coalesce(source_url, '')) LIKE '%/sp'
                              OR lower(coalesce(source_url, '')) LIKE '%=sp%'
                              OR lower(coalesce(source_url, '')) LIKE '%?sp%'
                              OR lower(coalesce(source_url, '')) LIKE '%&sp%'
                            THEN 1 ELSE 0 END) AS post_race_source_url_rows
                    FROM live_odds
                    WHERE {win_where}
                    """
                ).fetchone()
            )
        else:
            source_url_quality = {
                "rows_checked": int(win_counts.get("dog_level_win_odds_rows") or 0),
                "rows_with_source_url": 0,
                "rows_missing_source_url": int(win_counts.get("dog_level_win_odds_rows") or 0),
                "post_race_source_url_rows": 0,
                "source_url_column_present": False,
            }
        source_url_quality.setdefault(
            "source_url_column_present", "source_url" in live_odds_columns
        )
        for key in (
            "rows_checked",
            "rows_with_source_url",
            "rows_missing_source_url",
            "post_race_source_url_rows",
        ):
            if source_url_quality.get(key) is None:
                source_url_quality[key] = 0
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
        "source_url_quality": source_url_quality,
        "coverage_rates": {
            "race_id_metadata_match_rate": (
                int(match_counts.get("race_id_metadata_matches") or 0) / dog_level_win_rows
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
