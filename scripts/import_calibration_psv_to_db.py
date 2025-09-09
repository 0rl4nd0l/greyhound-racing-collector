#!/usr/bin/env python3
"""
Import calibration PSV exported by ml_backtesting_trainer.py into the SQLite predictions table.

- Creates table predictions if it does not exist, with columns expected by ProbabilityCalibrator:
  race_id TEXT, dog_name TEXT, raw_win_prob REAL, raw_place_prob REAL NULL,
  actual_win INTEGER, actual_place INTEGER, prediction_date TEXT

Usage:
  .venv/bin/python scripts/import_calibration_psv_to_db.py --psv predictions/backtests/walk_forward/walk_forward_dog_predictions_*.psv \
    --db greyhound_racing_data.db

Notes:
- actual_place is derived from dog_race_data.finish_position <= 3 when available; otherwise falls back to 1 if actual_win==1 else 0.
- raw_place_prob is left NULL (calibration can proceed on win-only data).
"""
from __future__ import annotations

import argparse
import glob
import os
import sqlite3
from pathlib import Path

import pandas as pd


def ensure_predictions_table(conn: sqlite3.Connection) -> None:
    # Create a self-contained predictions table in the target DB for calibrator consumption
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            race_id TEXT,
            dog_name TEXT,
            raw_win_prob REAL,
            raw_place_prob REAL,
            actual_win INTEGER,
            actual_place INTEGER,
            prediction_date TEXT
        )
        """
    )
    conn.commit()


def lookup_finish_position(conn: sqlite3.Connection, race_id: str, dog_name: str) -> int | None:
    # If dog_race_data table does not exist in this DB, skip lookup
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='dog_race_data' LIMIT 1")
        if not cur.fetchone():
            return None
    except Exception:
        return None
    # Try multiple name fields; normalize to lower for robustness
    q = (
        "SELECT finish_position FROM dog_race_data WHERE race_id=? AND ("
        "LOWER(COALESCE(dog_clean_name, '')) = LOWER(?) OR LOWER(COALESCE(dog_name, '')) = LOWER(?)"
        ") LIMIT 1"
    )
    cur = conn.execute(q, (race_id, dog_name, dog_name))
    row = cur.fetchone()
    if not row:
        return None
    try:
        return int(str(row[0]).strip())
    except Exception:
        return None


def import_psv(psv_paths: list[str], db_path: str) -> dict:
    # Open writable connection but avoid WAL files by using default journal
    conn = sqlite3.connect(db_path)
    try:
        ensure_predictions_table(conn)
        total_rows = 0
        inserted = 0
        for pattern in psv_paths:
            for p in glob.glob(pattern):
                df = pd.read_csv(p, sep="|")
                # Required columns check
                required = {"race_id", "dog_clean_name", "raw_win_prob", "actual_win", "race_date"}
                missing = required - set(df.columns)
                if missing:
                    print(f"Skipping {p}: missing columns {sorted(missing)}")
                    continue
                total_rows += len(df)
                # Insert row-by-row to allow DB lookups for actual_place
                for _, row in df.iterrows():
                    race_id = str(row["race_id"]) if pd.notna(row["race_id"]) else None
                    dog_name = str(row["dog_clean_name"]).strip() if pd.notna(row["dog_clean_name"]) else None
                    raw_win_prob = float(row["raw_win_prob"]) if pd.notna(row["raw_win_prob"]) else None
                    actual_win = int(row["actual_win"]) if pd.notna(row["actual_win"]) else None
                    prediction_date = str(row["race_date"]) if pd.notna(row["race_date"]) else None
                    if not race_id or not dog_name or raw_win_prob is None or actual_win is None:
                        continue
                    fp = lookup_finish_position(conn, race_id, dog_name)
                    if fp is not None and fp > 0:
                        actual_place = 1 if fp <= 3 else 0
                    else:
                        actual_place = 1 if actual_win == 1 else 0
                    conn.execute(
                        "INSERT INTO predictions (race_id, dog_name, raw_win_prob, raw_place_prob, actual_win, actual_place, prediction_date)"
                        " VALUES (?, ?, ?, NULL, ?, ?, ?)",
                        (race_id, dog_name, raw_win_prob, actual_win, actual_place, prediction_date),
                    )
                    inserted += 1
        conn.commit()
        return {"total": total_rows, "inserted": inserted}
    finally:
        conn.close()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psv", action="append", required=True, help="Glob pattern(s) for PSV files to import")
    ap.add_argument("--db", default="greyhound_racing_data.db", help="Path to SQLite DB")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    res = import_psv(args.psv, args.db)
    print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

