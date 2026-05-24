#!/usr/bin/env python3
"""
Report prediction coverage and upcoming metadata counts.

- Computes last-N-hours predictions count, join counts with race_metadata and live_odds (is_current=1)
- Reports upcoming window (today..+2d) metadata counts and distinct race_ids

Usage:
  python scripts/report_prediction_coverage.py --hours 24 [--db ./greyhound_racing_data_writable.db] [--save docs/analysis/prediction_coverage_report.json]

DB resolution order when --db not provided:
  $DATABASE_PATH > $STAGING_DB_PATH > $GREYHOUND_DB_PATH > ./greyhound_racing_data.db
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# Prefer routed read-only open if available
try:
    from scripts.db_utils import open_sqlite_readonly  # type: ignore
except Exception:
    def open_sqlite_readonly(db_path: str | None = None) -> sqlite3.Connection:
        path = db_path or os.getenv("DATABASE_PATH") or os.getenv("STAGING_DB_PATH") or os.getenv("GREYHOUND_DB_PATH") or "./greyhound_racing_data.db"
        uri = f"file:{Path(path).resolve()}?mode=ro"
        return sqlite3.connect(uri, uri=True)


def resolve_db(cli_db: str | None) -> str:
    if cli_db:
        return cli_db
    for env in ("DATABASE_PATH", "STAGING_DB_PATH", "GREYHOUND_DB_PATH"):
        v = os.getenv(env)
        if v:
            return v
    return "./greyhound_racing_data.db"


def run_report(db_path: str, hours: int) -> dict:
    out: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db_path": str(Path(db_path).resolve()),
        "hours": int(hours),
        "predictions": {},
        "predictions_latest": {},
        "odds_coverage_by_venue": [],
        "upcoming_metadata": {},
    }
    with open_sqlite_readonly(db_path) as conn:
        cur = conn.cursor()
        # Coverage metrics for last N hours (raw predictions)
        cur.execute(
            """
            WITH pN AS (
              SELECT * FROM predictions WHERE timestamp >= datetime('now', ?)
            )
            SELECT
              (SELECT COUNT(*) FROM pN) AS preds_count,
              (SELECT COUNT(*) FROM pN p JOIN race_metadata r ON p.race_id=r.race_id) AS preds_join_rm,
              (SELECT COUNT(*) FROM pN p JOIN live_odds l ON p.race_id=l.race_id WHERE l.is_current=1) AS preds_join_odds
            """,
            (f"-{hours} hours",),
        )
        row = cur.fetchone() or (0, 0, 0)
        preds_count, preds_join_rm, preds_join_odds = row
        out["predictions"] = {
            "total": int(preds_count),
            "join_race_metadata": int(preds_join_rm),
            "join_live_odds_current": int(preds_join_odds),
        }
        # Coverage metrics using predictions_latest view if present
        try:
            cur.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM predictions_latest) AS total_latest,
                  (SELECT COUNT(*) FROM predictions_latest p JOIN race_metadata r ON p.race_id=r.race_id) AS latest_join_rm,
                  (SELECT COUNT(*) FROM predictions_latest p JOIN live_odds l ON p.race_id=l.race_id WHERE l.is_current=1) AS latest_join_odds
                """
            )
            row2 = cur.fetchone() or (0, 0, 0)
            out["predictions_latest"] = {
                "total": int(row2[0] or 0),
                "join_race_metadata": int(row2[1] or 0),
                "join_live_odds_current": int(row2[2] or 0),
            }
        except Exception:
            out["predictions_latest"] = {"available": False}
        # Per-venue odds coverage (current odds joined to predictions_latest)
        try:
            cur.execute(
                """
                SELECT COALESCE(l.venue, r.venue) AS venue,
                       COUNT(*) AS matches
                FROM predictions_latest p
                JOIN live_odds l ON p.race_id = l.race_id AND l.is_current=1
                LEFT JOIN race_metadata r ON p.race_id = r.race_id
                GROUP BY COALESCE(l.venue, r.venue)
                ORDER BY matches DESC
                LIMIT 25
                """
            )
            out["odds_coverage_by_venue"] = [
                {"venue": (v or "UNKNOWN"), "matches": int(n or 0)} for (v, n) in cur.fetchall()
            ]
        except Exception:
            out["odds_coverage_by_venue"] = []
        # Upcoming window today..+2d
        cur.execute(
            """
            SELECT race_date, COUNT(*)
            FROM race_metadata
            WHERE race_date BETWEEN date('now','localtime') AND date('now','localtime','+2 day')
            GROUP BY race_date
            ORDER BY race_date
            """
        )
        upcoming_counts = {r: c for (r, c) in cur.fetchall()}
        cur.execute(
            """
            SELECT COUNT(DISTINCT race_id)
            FROM race_metadata
            WHERE race_date BETWEEN date('now','localtime') AND date('now','localtime','+2 day')
            """
        )
        distinct_ids = cur.fetchone()[0] if cur.fetchone is not None else 0
        out["upcoming_metadata"] = {
            "by_date": upcoming_counts,
            "distinct_race_ids": int(distinct_ids),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Prediction coverage reporter")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--db", dest="db_path", default=None)
    ap.add_argument("--save", dest="save_path", default=None)
    ns = ap.parse_args()

    db_path = resolve_db(ns.db_path)
    report = run_report(db_path, ns.hours)

    print(json.dumps(report, indent=2, sort_keys=True))

    if ns.save_path:
        out_path = Path(ns.save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Saved report to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

