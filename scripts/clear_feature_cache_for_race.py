#!/usr/bin/env python3
"""
Clear feature cache for a specific race (ML System V4)
=====================================================

Removes cached feature files under .cache/features_v4 for a given race_id.
This ensures the next prediction rebuilds features (including fresh TGR features)
from the updated database instead of using stale cached artifacts.

Usage:
  python scripts/clear_feature_cache_for_race.py --race-id ap_k_2025-02-18_2

Notes:
- Only deletes files matching the sanitized race_id prefix "<safe_race>__".
- Safe sanitization mirrors MLSystemV4._cache_paths safe filename behavior.
"""

import argparse
import os
from pathlib import Path


def sanitize_race_id(race_id: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(race_id))[:80]


def main():
    parser = argparse.ArgumentParser(description="Clear ML V4 feature cache for a race")
    parser.add_argument("--race-id", dest="race_id", required=True, help="Race ID to clear cache for")
    parser.add_argument("--cache-dir", dest="cache_dir", default=".cache/features_v4", help="Features cache directory")
    args = parser.parse_args()

    safe_race = sanitize_race_id(args.race_id)
    cache_dir = Path(args.cache_dir)

    if not cache_dir.exists():
        print(f"ℹ️ Cache directory not found: {cache_dir}")
        return

    pattern = f"{safe_race}__"
    removed = 0
    for p in cache_dir.glob(f"{pattern}*"):
        try:
            p.unlink()
            removed += 1
        except Exception as e:
            print(f"⚠️ Failed to remove {p}: {e}")

    print(f"✅ Removed {removed} cache files for race_id={args.race_id}")


if __name__ == "__main__":
    main()

