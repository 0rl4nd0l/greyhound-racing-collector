#!/usr/bin/env python3
"""
TGR Backfill for a Specific Race
================================

Backfills enhanced TGR data (dog form history, performance summaries, insights)
for all dogs in a specified race_id, writing into the enhanced TGR tables.

Usage:
  python scripts/tgr_backfill_for_race.py --db databases/comprehensive_greyhound_data.db --race-id ap_k_2025-02-18_2 [--rate-limit 2.5] [--no-cache]

Notes:
- Respects EnhancedTGRCollector configuration and rate limiting.
- Requires TGR scraper components to be available.
- Intended to be run before re-predicting a historical race so that TGR features
  are built from real data rather than defaults.
"""

import argparse
import os
import sqlite3
import sys
from typing import List
from scripts.db_utils import open_sqlite_writable

# Ensure project root on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from enhanced_tgr_collector import EnhancedTGRCollector


def get_dogs_for_race(db_path: str, race_id: str) -> List[str]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT DISTINCT dog_clean_name
            FROM dog_race_data
            WHERE race_id = ? AND dog_clean_name IS NOT NULL AND TRIM(dog_clean_name) != ''
            ORDER BY box_number ASC
            """,
            [race_id],
        )
        rows = cur.fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Backfill TGR enhanced data for a specific race")
    parser.add_argument("--db", dest="db_path", required=True, help="Path to SQLite DB")
    parser.add_argument("--race-id", dest="race_id", required=True, help="Race ID to backfill")
    parser.add_argument("--rate-limit", dest="rate_limit", default="2.5", help="TGR scraper rate limit (seconds)")
    parser.add_argument("--no-cache", dest="no_cache", action="store_true", help="Disable scraper cache for fresh fetch")
    args = parser.parse_args()

    db_path = args.db_path
    race_id = args.race_id

    # Configure scraper behavior via env for EnhancedTGRCollector
    os.environ.setdefault("TGR_RATE_LIMIT", str(args.rate_limit))
    if args.no_cache:
        os.environ["TGR_DISABLE_CACHE"] = "1"
    else:
        os.environ["TGR_DISABLE_CACHE"] = "0"

    # Resolve GREYHOUND_DB_PATH for downstream components that auto-detect DB
    os.environ.setdefault("GREYHOUND_DB_PATH", db_path)

    dogs = get_dogs_for_race(db_path, race_id)
    if not dogs:
        print(f"❌ No dogs found for race_id={race_id} in DB {db_path}")
        sys.exit(2)

    print(f"🐕 Dogs in race {race_id}: {len(dogs)}")
    for i, d in enumerate(dogs, 1):
        print(f"  {i}. {d}")

    collector = EnhancedTGRCollector(db_path=db_path)
    results = collector.collect_comprehensive_dog_data(dogs)

    # Print summary
    print("\n📊 Backfill Summary:")
    print(f"  Dogs processed: {results.get('dogs_processed', 0)}")
    print(f"  Form entries saved: {results.get('total_entries', 0)}")
    print(f"  Expert insights saved: {results.get('total_insights', 0)}")

    errs = results.get("errors") or []
    if errs:
        print(f"  Errors: {len(errs)} (showing up to 5):")
        for e in errs[:5]:
            print(f"   - {e}")

    # Basic DB verify: count enhanced form rows for these dogs
    try:
        conn = open_sqlite_writable(db_path)
        cur = conn.cursor()
        q_marks = ",".join(["?"] * len(dogs))
        cur.execute(
            f"SELECT COUNT(*) FROM tgr_enhanced_dog_form WHERE UPPER(dog_name) IN ({q_marks})",
            [d.upper() for d in dogs],
        )
        cnt = cur.fetchone()[0]
        print(f"  tgr_enhanced_dog_form rows for race dogs: {cnt}")
    except Exception as e:
        print(f"  ⚠️ Post-verify failed: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    print("\n✅ Backfill completed (see logs for details)")


if __name__ == "__main__":
    main()

