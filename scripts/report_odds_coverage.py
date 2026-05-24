#!/usr/bin/env python3
"""Report dog-level odds coverage without scraping or writing to the DB."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accuracy_program.odds_coverage import analyze_odds_coverage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default="greyhound_racing_data.db",
        help="SQLite DB path to inspect read-only",
    )
    parser.add_argument(
        "--all-odds",
        action="store_true",
        help="Include non-current live_odds rows in dog-level win coverage",
    )
    parser.add_argument(
        "--stale-after-hours",
        type=float,
        default=6.0,
        help="TTL used to flag stale odds timestamps",
    )
    parser.add_argument(
        "--now",
        help="Optional ISO timestamp to use as the report reference time",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    report_now = datetime.fromisoformat(args.now) if args.now else None
    metrics = analyze_odds_coverage(
        args.db,
        current_only=not args.all_odds,
        stale_after_hours=args.stale_after_hours,
        now=report_now,
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
