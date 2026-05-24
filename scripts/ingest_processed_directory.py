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

from ingestion.staging_writer import parse_race_csv_for_staging
from scripts.db_utils import open_sqlite_writable
from scripts.ingest_csv_history import (
    ensure_staging_tables,
    pick_db_path,
    upsert_dogs,
    upsert_race_metadata,
)
from datetime import datetime

# Reuse data-quality logging target
from pathlib import Path as _Path
DQ_DIR = _Path("logs") / "data_quality"
DQ_DIR.mkdir(parents=True, exist_ok=True)
DQ_FILE = DQ_DIR / "weight_completeness.csv"


def _compute_and_log_weight_completeness_batch(dogs, meta, src_path, db_path):
    try:
        total = len(dogs) if dogs is not None else 0
        non_null = 0
        for d in (dogs or []):
            w = d.get("weight")
            if w is None:
                continue
            s = str(w).strip()
            if s and s.lower() not in {"nan", "none", "null"}:
                try:
                    float(s)
                    non_null += 1
                except Exception:
                    pass
        frac = (non_null / total) if total > 0 else 0.0
        threshold_env = os.getenv("WEIGHT_ALERT_THRESHOLD") if "os" in globals() else None
        try:
            threshold = float(threshold_env) if threshold_env else 0.50
        except Exception:
            threshold = 0.50
        new_file = not DQ_FILE.exists()
        with DQ_FILE.open("a", encoding="utf-8") as f:
            if new_file:
                f.write(
                    "timestamp,file,race_id,total_dogs,weights_non_null,completeness,threshold,db\n"
                )
            f.write(
                f"{datetime.utcnow().isoformat()}Z,{_Path(src_path).name},{meta.race_id},{total},{non_null},{frac:.4f},{threshold:.2f},{db_path}\n"
            )
        if frac < threshold:
            print(
                f"⚠️ Weight completeness low for {_Path(src_path).name} (race_id={meta.race_id}): {non_null}/{total} = {frac:.1%} (< {threshold:.0%})"
            )
    except Exception as _e:
        print(f"⚠️ Weight completeness check failed: {_e}")


def iter_csvs(root: Path) -> List[Path]:
    for p in root.rglob("*.csv"):
        parts = set(map(str.lower, p.parts))
        if "excluded" in parts:
            continue
        yield p


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch staged ingestion from directory")
    ap.add_argument(
        "--root",
        default="processed",
        help="Root directory containing CSVs (default: processed)",
    )
    ap.add_argument(
        "--db",
        default=None,
        help="Path to SQLite DB (defaults to env or greyhound_racing_data.db)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of files to process (0 = no limit)",
    )
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
            # Data-quality alert per file
            try:
                _compute_and_log_weight_completeness_batch(dogs, meta, str(f), db_path)
            except Exception:
                pass
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
            print(
                f"Progress: {i}/{len(files)} processed (ok={ok}, fail={fail}, dogs={total_dogs})"
            )

    print(
        f"✅ DONE: files_ok={ok}, files_failed={fail}, dogs={total_dogs}, root={root}, db={db_path}"
    )
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
