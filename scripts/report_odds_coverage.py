#!/usr/bin/env python3
"""Report dog-level odds coverage without scraping or writing to the DB."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accuracy_program.odds_coverage import (
    analyze_odds_coverage,
    analyze_snapshot_odds_coverage,
)


def _write_null_ev_csv(metrics: dict, output: str) -> None:
    rows = []
    snapshot_report = metrics.get("snapshot_odds_coverage")
    if isinstance(snapshot_report, dict):
        rows = list(snapshot_report.get("null_ev_reason_rows") or [])
    elif isinstance(metrics, dict):
        rows = list(metrics.get("null_ev_reason_rows") or [])

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "snapshot_path",
        "race_id",
        "dog_name",
        "box_number",
        "odds_decimal",
        "ev_win",
        "odds_match_status",
        "odds_match_method",
        "odds_exclusion_reason",
        "odds_provenance_status",
        "null_ev_reason",
        "odds_timestamp",
        "prediction_timestamp",
        "feature_freeze_timestamp",
        "jump_time",
        "odds_age_seconds_at_prediction",
        "odds_captured_before_prediction",
        "odds_captured_before_feature_freeze",
        "odds_captured_before_jump",
        "odds_stale_at_prediction",
        "odds_source",
        "odds_source_url",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


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
    parser.add_argument(
        "--snapshot-dir",
        action="append",
        default=[],
        help="Snapshot JSON file or directory to inspect for odds/EV provenance",
    )
    parser.add_argument(
        "--null-ev-csv",
        help="Optional CSV path for per-runner null EV reason diagnostics",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    report_now = datetime.fromisoformat(args.now) if args.now else None
    db_metrics = analyze_odds_coverage(
        args.db,
        current_only=not args.all_odds,
        stale_after_hours=args.stale_after_hours,
        now=report_now,
    )
    if args.snapshot_dir:
        metrics = {
            "db_odds_coverage": db_metrics,
            "snapshot_odds_coverage": analyze_snapshot_odds_coverage(
                args.snapshot_dir
            ),
        }
    else:
        metrics = db_metrics
    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    if args.null_ev_csv:
        _write_null_ev_csv(metrics, args.null_ev_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
