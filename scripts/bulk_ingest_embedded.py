#!/usr/bin/env python3
"""
Ingest embedded history into dog_race_data and minimal race_metadata for all CSVs
in the upcoming_races_temp directory.

Usage:
  python scripts/bulk_ingest_embedded.py --db /path/to/writable.db [--dir upcoming_races_temp]
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Tuple

from scripts.ingest_embedded_form_history import upsert_embedded_history_and_meta


def run_bulk(db_path: str, dir_path: str) -> Tuple[int, int, int, int]:
    files = sorted(glob.glob(os.path.join(dir_path, "*.csv")))
    inserted_total = 0
    skipped_total = 0
    errors = 0
    processed = 0
    print(f"Found {len(files)} CSVs in {dir_path}")
    for f in files:
        try:
            stats = upsert_embedded_history_and_meta(db_path, f)
            ins = int(stats.get("inserted", 0) or 0)
            skp = int(stats.get("skipped", 0) or 0)
            inserted_total += ins
            skipped_total += skp
            processed += 1
            print(f"{os.path.basename(f)} -> inserted={ins} skipped={skp}")
        except Exception as e:
            errors += 1
            print(f"ERROR {f}: {e}")
    return processed, inserted_total, skipped_total, errors


def main():
    ap = argparse.ArgumentParser(description="Bulk ingest embedded history for all CSVs in a directory")
    ap.add_argument("--db", required=True, help="Path to writable SQLite DB")
    ap.add_argument("--dir", default="upcoming_races_temp", help="Directory of CSVs")
    args = ap.parse_args()
    processed, ins, skp, errs = run_bulk(args.db, args.dir)
    print(f"SUMMARY: files={processed} inserted={ins} skipped={skp} errors={errs}")


if __name__ == "__main__":
    main()
