#!/usr/bin/env python3
"""
Export DB-backed historical rows for specified dogs into a CSV.

Usage:
  python scripts/export_dog_history.py \
    --db /path/to/greyhound_racing_data_writable.db \
    --dogs "Hes A Profit,Elegant Zephyr,Zaidee Diva,Sebon Rocks,Searing Lass,Nitro Bolt" \
    --out exports/db_history_Race_10_-_AP_K_-_2025-09-09.csv

Notes:
- Exports rows where race_metadata.race_date IS NOT NULL and finish_position IS NOT NULL.
- Joins dog_race_data with race_metadata and includes key fields for review.
"""
from __future__ import annotations

import argparse
import csv
import os
import sqlite3
from typing import List


def export_history(db_path: str, dogs: List[str], out_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        dogs_upper = [str(d).strip().upper() for d in dogs if str(d).strip()]
        if not dogs_upper:
            raise ValueError("No valid dog names provided")
        placeholders = ",".join(["?"] * len(dogs_upper))
        query = (
            "SELECT "
            " d.dog_clean_name, d.race_id, r.race_date, r.race_time, r.venue, r.grade, r.distance,"
            " d.finish_position, d.individual_time, d.margin, d.weight, d.starting_price, d.data_source "
            "FROM dog_race_data d LEFT JOIN race_metadata r ON d.race_id = r.race_id "
            "WHERE r.race_date IS NOT NULL AND d.finish_position IS NOT NULL "
            f"AND UPPER(d.dog_clean_name) IN ({placeholders}) "  # nosec B608: only qmark placeholders are injected; values parameterized; no identifiers constructed
            "ORDER BY d.dog_clean_name, r.race_date DESC, r.race_time DESC"
        )
        cur.execute(query, dogs_upper)
        rows = cur.fetchall()
        headers = [desc[0] for desc in cur.description]
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            w.writerow(headers)
            w.writerows(rows)
        return len(rows)
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description="Export DB-backed dog history to CSV")
    ap.add_argument("--db", required=True, help="Path to SQLite DB")
    ap.add_argument(
        "--dogs",
        required=True,
        help="Comma-separated list of dog names",
    )
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    dogs = [d.strip() for d in args.dogs.split(",") if d.strip()]
    count = export_history(args.db, dogs, args.out)
    print(f"✅ Exported {count} rows to {args.out}")


if __name__ == "__main__":
    main()
