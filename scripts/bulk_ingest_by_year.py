#!/usr/bin/env python3
"""
Bulk-ingest historical CSVs by detecting the true race_date from content and filtering by year(s).

- Uses ingestion.staging_writer.parse_race_csv_for_staging to parse each CSV
- Filters by meta.race_date year (e.g., 2023/2024)
- Performs a single guarded DB session (one backup + integrity check) for the whole batch
- Upserts into canonical tables via scripts.ingest_csv_history helpers
- Logs per-file outcomes to logs/ingest_legacy/ingest_{years}_{ts}.jsonl

Usage:
    python scripts/bulk_ingest_by_year.py --years 2023 2024 --db greyhound_racing_data_writable.db \
        --roots archive processed

Notes:
- Archive-first policy: scan archive/ before processed/
- Idempotent: upserts avoid duplicates and preserve existing non-null values via COALESCE
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

from ingestion.staging_writer import parse_race_csv_for_staging
from scripts.db_guard import db_guard
from scripts.db_utils import open_sqlite_writable
from scripts.ingest_csv_history import (
    ensure_staging_tables,
    upsert_race_metadata,
    upsert_dogs,
)

LOG_ROOT = Path("logs") / "ingest_legacy"
LOG_ROOT.mkdir(parents=True, exist_ok=True)


def iter_csv_files(roots: List[str]) -> Iterable[Path]:
    for root in roots:
        base = Path(root)
        if not base.exists() or not base.is_dir():
            continue
        for p in base.rglob("*.csv"):
            # Skip obvious non-form CSVs if any future patterns emerge (keep permissive for now)
            yield p


def parse_and_filter(csv_path: Path, years: set[int]) -> Tuple[bool, dict]:
    """Return (keep, result_json). keep indicates whether file's race_date year is in `years`.
    result_json contains file, status, race_id (if parsed), year, dogs_count, and error if any.
    """
    rec = {
        "file": str(csv_path),
        "status": "parsed",
        "race_id": None,
        "year": None,
        "dogs": 0,
        "error": None,
    }
    try:
        meta, dogs = parse_race_csv_for_staging(str(csv_path))
        rec["race_id"] = meta.race_id
        rec["dogs"] = len(dogs)
        y = int(meta.race_date.split("-")[0])
        rec["year"] = y
        keep = y in years
        rec["status"] = "keep" if keep else "skip_year"
        return keep, rec | {"meta": {
            "race_date": meta.race_date,
            "venue": meta.venue,
            "race_number": meta.race_number,
        }} | {"dogs_preview": [d.get("dog_clean_name") for d in dogs[:2]]}
    except Exception as e:
        rec["status"] = "error"
        rec["error"] = str(e)
        return False, rec


def bulk_ingest(years: List[int], db_path: str, roots: List[str]) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    years_key = "-".join(str(y) for y in sorted(years))
    log_path = LOG_ROOT / f"ingest_{years_key}_{ts}.jsonl"

    years_set = set(years)

    # Pre-parse to determine candidates (and gather meta) before DB writes
    candidates: List[Tuple[dict, list]] = []
    parsed_count = 0
    kept_count = 0

    with log_path.open("a", encoding="utf-8") as logf:
        for csv_path in iter_csv_files(roots):
            keep, info = parse_and_filter(csv_path, years_set)
            parsed_count += 1
            logf.write(json.dumps(info, ensure_ascii=False) + "\n")
            if keep:
                kept_count += 1
                # Re-parse for staging dogs; parse_and_filter already parsed but we didn't keep dogs list
                try:
                    meta, dogs = parse_race_csv_for_staging(str(csv_path))
                    candidates.append((meta.__dict__, dogs))
                except Exception as e:
                    logf.write(json.dumps({
                        "file": str(csv_path),
                        "status": "error_reparse",
                        "error": str(e),
                    }, ensure_ascii=False) + "\n")

    # One guarded DB session for all upserts
    with db_guard(db_path=db_path, label=f"bulk_ingest_{years_key}") as guard:
        guard.expect_table_growth("race_metadata", min_delta=0)
        guard.expect_table_growth("dog_race_data", min_delta=0)

        conn = open_sqlite_writable(db_path)
        try:
            ensure_staging_tables(conn)
            with log_path.open("a", encoding="utf-8") as logf:
                for meta_dict, dogs in candidates:
                    # meta_dict keys: race_id, venue, race_number, race_date, etc. Convert to RaceMeta-like for upsert
                    class _M:
                        def __init__(self, d):
                            self.race_id = d.get("race_id")
                            self.venue = d.get("venue")
                            self.race_number = d.get("race_number")
                            self.race_date = d.get("race_date")
                            self.race_name = d.get("race_name")
                            self.grade = d.get("grade")
                            self.distance = d.get("distance")
                    _meta = _M(meta_dict)
                    try:
                        upsert_race_metadata(conn, _meta, field_size=len(dogs))
                        upsert_dogs(conn, dogs)
                        logf.write(json.dumps({
                            "race_id": _meta.race_id,
                            "status": "upsert_ok",
                            "dogs": len(dogs),
                        }, ensure_ascii=False) + "\n")
                    except Exception as e:
                        logf.write(json.dumps({
                            "race_id": _meta.race_id,
                            "status": "upsert_error",
                            "error": str(e),
                        }, ensure_ascii=False) + "\n")
        finally:
            conn.close()

    print(f"Parsed files: {parsed_count}, kept by year({years_key}): {kept_count}. Log: {log_path}")
    return log_path


def main():
    ap = argparse.ArgumentParser(description="Bulk-ingest historical CSVs filtered by year(s)")
    ap.add_argument("--years", nargs="+", type=int, required=True, help="Years to include, e.g. 2023 2024")
    ap.add_argument("--db", required=False, default="greyhound_racing_data_writable.db", help="SQLite DB path to write to")
    ap.add_argument("--roots", nargs="+", required=False, default=["archive", "processed"], help="Root directories to scan for CSVs")
    args = ap.parse_args()

    # Stable environment hints for lower noise in logs
    os.environ.setdefault("WEIGHT_ALERT_THRESHOLD", "0.50")

    bulk_ingest(args.years, args.db, args.roots)


if __name__ == "__main__":
    main()
