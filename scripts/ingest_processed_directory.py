#!/usr/bin/env python3
"""
Batch staged ingestion from a directory of CSVs.

Walks a root (default: processed/) and ingests each CSV via the staged pipeline:
- Parse metadata and dogs with ingestion.staging_writer
- Upsert into race_metadata and dog_race_data with ON CONFLICT guards

Usage:
  python scripts/ingest_processed_directory.py --root processed --db greyhound_racing_data.db

Notes:
- Skips any path segment named 'excluded'
- Safe to re-run; upserts are idempotent and uniqueness constraints apply
- For single-file ingest, use scripts/ingest_csv_history.py
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path
from typing import List
from scripts.db_utils import open_sqlite_writable

from ingestion.staging_writer import parse_race_csv_for_staging
from scripts.ingest_csv_history import (
    ensure_staging_tables,
    upsert_race_metadata,
    upsert_dogs,
    pick_db_path,
)


def iter_csvs(root: Path) -> List[Path]:
    for p in root.rglob("*.csv"):
        parts = set(map(str.lower, p.parts))
        if "excluded" in parts:
            continue
        yield p


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch staged ingestion from directory")
    ap.add_argument("--root", default="processed", help="Root directory containing CSVs (default: processed)")
    ap.add_argument("--db", default=None, help="Path to SQLite DB (defaults to env or greyhound_racing_data.db)")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of files to process (0 = no limit)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"❌ Root directory not found: {root}")
        return 2

    db_path = pick_db_path(args.db)

    files = list(iter_csvs(root))
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    if not files:
        print(f"ℹ️ No CSV files found under {root}")
        return 0

    conn = open_sqlite_writable(db_path)
    try:
        ensure_staging_tables(conn)
    finally:
        conn.close()

    ok = 0
    fail = 0
    total_dogs = 0

    for i, f in enumerate(files, 1):
        try:
            meta, dogs = parse_race_csv_for_staging(str(f))
            total_dogs += len(dogs)
            conn = open_sqlite_writable(db_path)
            try:
                upsert_race_metadata(conn, meta, field_size=len(dogs))
                upsert_dogs(conn, dogs)
            finally:
                conn.close()
            ok += 1
        except Exception as e:
            print(f"❌ {f}: {e}")
            fail += 1

        if i % 200 == 0:
            print(f"Progress: {i}/{len(files)} processed (ok={ok}, fail={fail}, dogs={total_dogs})")

    print(f"✅ DONE: files_ok={ok}, files_failed={fail}, dogs={total_dogs}, root={root}, db={db_path}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

