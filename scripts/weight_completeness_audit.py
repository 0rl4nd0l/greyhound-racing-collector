#!/usr/bin/env python3
"""
Compute daily weight completeness over a recent window and write a CSV report.

- Connects read-only to GREYHOUND_DB_PATH (falls back to greyhound_racing_data_writable.db or greyhound_racing_data.db)
- Aggregates last N days (default 30) by race_date
- Writes to logs/data_quality/weight_completeness_daily.csv (overwrites each run)
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

REPORT_DIR = Path("logs") / "data_quality"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE = REPORT_DIR / "weight_completeness_daily.csv"


def resolve_db_path() -> str:
    env = (
        os.getenv("GREYHOUND_DB_PATH")
        or os.getenv("ANALYTICS_DB_PATH")
        or os.getenv("STAGING_DB_PATH")
        or "greyhound_racing_data_writable.db"
    )
    if not os.path.exists(env):
        # fallback to analytics DB if available
        if os.path.exists("greyhound_racing_data.db"):
            return "greyhound_racing_data.db"
    return env


def compute_report(days_back: int = 30) -> None:
    db_path = resolve_db_path()
    # Read-only connection
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        try:
            conn.execute("PRAGMA query_only=ON")
            conn.execute("PRAGMA foreign_keys=ON")
        except Exception:
            pass
        # Use Python-side date bounds for robustness
        end_dt = datetime.utcnow().date()
        start_dt = end_dt - timedelta(days=days_back)
        sql = (
            "SELECT rm.race_date AS race_date, "
            "COUNT(*) AS dogs, "
            "SUM(CASE WHEN drd.weight IS NOT NULL THEN 1 ELSE 0 END) AS weights_non_null "
            "FROM dog_race_data drd "
            "JOIN race_metadata rm ON drd.race_id = rm.race_id "
            "GROUP BY rm.race_date ORDER BY rm.race_date DESC"
        )
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()

    # Filter and write CSV
    with REPORT_FILE.open("w", encoding="utf-8") as f:
        f.write("date,total_dogs,weights_non_null,completeness,timestamp,db\n")
        for race_date, dogs, non_null in rows:
            try:
                # try parse ISO-like, else skip non-ISO rows
                d = datetime.fromisoformat(str(race_date).split(" ")[0]).date()
            except Exception:
                # attempt dayfirst parse fallback
                try:
                    from dateutil import parser as _p  # optional fallback

                    d = _p.parse(str(race_date), dayfirst=True).date()
                except Exception:
                    continue
            if not (start_dt <= d <= end_dt):
                continue
            dogs = int(dogs or 0)
            non_null = int(non_null or 0)
            comp = (non_null / dogs) if dogs > 0 else 0.0
            f.write(
                f"{d.isoformat()},{dogs},{non_null},{comp:.4f},{datetime.utcnow().isoformat()}Z,{db_path}\n"
            )

    print(f"✅ Wrote daily weight completeness to {REPORT_FILE}")


if __name__ == "__main__":
    try:
        days = int(os.getenv("WEIGHT_AUDIT_DAYS", "30"))
    except Exception:
        days = 30
    compute_report(days_back=days)
