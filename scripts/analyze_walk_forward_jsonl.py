#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sqlite3
from collections import defaultdict, Counter
from datetime import datetime
from typing import Optional, Dict

# Utility: open SQLite in read-only/query-only

def open_sqlite_readonly(path: str) -> Optional[sqlite3.Connection]:
    try:
        db_uri = f"file:{os.path.abspath(path)}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True)
        try:
            conn.execute("PRAGMA query_only=ON")
            conn.execute("PRAGMA foreign_keys=ON")
        except Exception:
            pass
        return conn
    except Exception:
        return None


def find_latest_predictions_file(root: str) -> Optional[str]:
    pattern = os.path.join(root, "predictions", "backtests", "walk_forward", "walk_forward_predictions_*.jsonl")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort()
    return files[-1]


def load_race_metadata_map(db_path: str) -> Dict[str, Dict[str, str]]:
    """Return mapping race_id -> { 'venue': ..., 'date': ... } (best-effort)."""
    result = {}
    conn = open_sqlite_readonly(db_path)
    if conn is None:
        return result
    try:
        # Try common column names
        cursor = conn.cursor()
        # Introspect columns
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(race_metadata)").fetchall()}
        race_id_col = "race_id" if "race_id" in cols else None
        venue_col = "venue" if "venue" in cols else ("track" if "track" in cols else ("course" if "course" in cols else None))
        date_col = "race_date" if "race_date" in cols else ("date" if "date" in cols else None)
        if race_id_col is None:
            return result
        sel = [race_id_col]
        if venue_col:
            sel.append(venue_col)
        if date_col:
            sel.append(date_col)
        query = f"SELECT {', '.join(sel)} FROM race_metadata"
        for row in cursor.execute(query):
            rid = str(row[0])
            venue = str(row[1]) if venue_col and len(row) > 1 and row[1] is not None else None
            date_val = str(row[2]) if date_col and len(row) > 2 and row[2] is not None else None
            result[rid] = {"venue": venue, "race_date": date_val}
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return result


def month_bucket(date_str: str) -> str:
    try:
        d = datetime.fromisoformat(date_str)
    except Exception:
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            return "unknown"
    return d.strftime("%Y-%m")


def main():
    parser = argparse.ArgumentParser(description="Analyze walk-forward JSONL predictions")
    parser.add_argument("--file", dest="file", default=None, help="Path to predictions JSONL. If omitted, the latest is used.")
    parser.add_argument("--threshold", dest="threshold", type=float, default=0.6, help="High-confidence threshold for misses")
    parser.add_argument("--db", dest="db", default=None, help="SQLite DB path for venue enrichment (defaults to $ANALYTICS_DB_PATH or greyhound_racing_data.db)")
    args = parser.parse_args()

    root = os.getcwd()
    file_path = args.file or find_latest_predictions_file(root)
    if not file_path or not os.path.exists(file_path):
        print(f"❌ Predictions file not found. Looked at: {file_path}")
        return 1

    # Resolve DB path: prefer env ANALYTICS_DB_PATH, then GREYHOUND_DB_PATH, then default
    db_path = args.db or os.getenv("ANALYTICS_DB_PATH") or os.getenv("GREYHOUND_DB_PATH") or "greyhound_racing_data.db"

    # Enrich race_id -> venue map
    race_meta_map = load_race_metadata_map(db_path)

    total = 0
    scorable = 0
    correct = 0

    # Calibration bins (0.0-0.1 ... 0.9-1.0)
    bins = [0] * 10
    bins_correct = [0] * 10
    bins_sum_p = [0.0] * 10

    # High-confidence misses (HCM)
    hcm_total = 0
    hcm_by_field = Counter()
    hcm_by_month = Counter()
    hcm_by_venue = Counter()
    hcm_examples = []  # keep top N by predicted_prob
    HCM_EX_LIMIT = 25

    def get_venue_for(rid: str, rec: dict) -> Optional[str]:
        # Prefer venue in record if present
        v = rec.get("venue")
        if v:
            return v
        meta = race_meta_map.get(str(rid))
        if meta:
            return meta.get("venue")
        return None

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            total += 1

            p = rec.get("predicted_prob")
            corr = bool(rec.get("correct", False))
            actual_winner = rec.get("actual_winner")
            is_scorable = rec.get("scorable")
            if is_scorable is None:
                is_scorable = actual_winner is not None and str(actual_winner).strip() not in ("", "N/A")

            if p is None:
                continue

            # Calibration uses scorable only
            b = int(max(0, min(9, int(p * 10))))
            if is_scorable:
                scorable += 1
                bins[b] += 1
                bins_sum_p[b] += float(p)
                if corr:
                    correct += 1
                    bins_correct[b] += 1

            # High-confidence misses
            if is_scorable and (not corr) and float(p) >= args.threshold:
                hcm_total += 1
                field_size = rec.get("field_size")
                if field_size is not None:
                    try:
                        hcm_by_field[int(field_size)] += 1
                    except Exception:
                        pass
                rdate = rec.get("race_date")
                if rdate:
                    hcm_by_month[month_bucket(str(rdate))] += 1
                venue = get_venue_for(rec.get("race_id"), rec)
                if venue:
                    hcm_by_venue[venue] += 1
                # retain top examples
                hcm_examples.append({
                    "race_id": rec.get("race_id"),
                    "race_date": rec.get("race_date"),
                    "predicted_top": rec.get("predicted_top"),
                    "predicted_prob": float(p),
                    "actual_winner": actual_winner,
                    "field_size": rec.get("field_size"),
                    "venue": venue,
                    "odds_top": rec.get("odds_top"),
                    "expected_value_top": rec.get("expected_value_top"),
                })

    # Sort examples by predicted_prob desc and keep top N
    hcm_examples.sort(key=lambda x: x["predicted_prob"], reverse=True)
    hcm_examples = hcm_examples[:HCM_EX_LIMIT]

    print("\n=== Predictions Analysis ===")
    print(f"File: {file_path}")
    print(f"DB (for venue enrichment): {args.db} {'(found)' if race_meta_map else '(not used/found)'}")
    print(f"Total predictions: {total}")
    print(f"Scorable predictions: {scorable}")
    if scorable > 0:
        print(f"Top-1 accuracy (scorable): {correct / scorable:.3f}")

    # Calibration table
    print("\n-- Calibration (scorable only; bin width=0.1) --")
    print("Bin    N      AvgPred   ObsWinRate   Diff")
    for i in range(10):
        n = bins[i]
        if n == 0:
            avg_p = 0.0
            obs = 0.0
        else:
            avg_p = bins_sum_p[i] / n
            obs = bins_correct[i] / n
        diff = obs - avg_p
        print(f"{i/10:.1f}-{(i+1)/10:.1f}  {n:6d}  {avg_p:7.3f}    {obs:10.3f}   {diff:6.3f}")

    # High-confidence misses summary
    print(f"\n-- High-Confidence Misses (p >= {args.threshold}) --")
    print(f"Count: {hcm_total}")

    if hcm_total > 0:
        # Field size distribution
        print("\nBy Field Size:")
        for fs, cnt in sorted(hcm_by_field.items()):
            pct = cnt / hcm_total * 100
            print(f"  {fs:2d}: {cnt:4d} ({pct:5.1f}%)")

        # Month distribution
        print("\nBy Month:")
        for m, cnt in sorted(hcm_by_month.items()):
            pct = cnt / hcm_total * 100
            print(f"  {m}: {cnt:4d} ({pct:5.1f}%)")

        # Venue distribution (top 10)
        if hcm_by_venue:
            print("\nBy Venue (Top 10):")
            for venue, cnt in hcm_by_venue.most_common(10):
                pct = cnt / hcm_total * 100
                print(f"  {venue}: {cnt:4d} ({pct:5.1f}%)")

        # Examples
        print("\nTop Examples (highest predicted_prob but incorrect):")
        for ex in hcm_examples:
            print(
                f"  [{ex['race_date']}] race_id={ex['race_id']} venue={ex.get('venue') or 'unknown'} "
                f"p={ex['predicted_prob']:.3f} top={ex['predicted_top']} actual={ex['actual_winner']} "
                f"field_size={ex.get('field_size')} EV={ex.get('expected_value_top')} odds={ex.get('odds_top')}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

