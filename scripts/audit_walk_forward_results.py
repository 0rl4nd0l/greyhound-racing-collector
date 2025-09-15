#!/usr/bin/env python3
"""
Audit walk-forward predictions for field size mismatches and underperforming slices.

Outputs a brief report to stdout.
"""
import json
import os
import re
import sqlite3
from collections import defaultdict

SUMMARY_PATH = "predictions/backtests/walk_forward/walk_forward_summary_20250915_200052.json"
PREDS_PATH = "predictions/backtests/walk_forward/walk_forward_predictions_20250915_200052.jsonl"

DB = None
for cand in ("greyhound_racing_data_writable.db", "greyhound_racing_data.db"):
    if os.path.isfile(cand):
        DB = cand
        break

re_dist = re.compile(r"^(\d{2,4})m$", re.I)

def get_dist(rid: str):
    for tok in rid.split("_"):
        m = re_dist.match(tok)
        if m:
            return int(m.group(1))
    return None


def load_preds(path):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def db_field_sizes(db):
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT race_id, field_size FROM race_metadata").fetchall()
    conn.close()
    out = {}
    for r in rows:
        rid = r["race_id"]
        fs = r["field_size"]
        try:
            fs = int(fs)
        except Exception:
            try:
                fs = int(str(fs).strip())
            except Exception:
                continue
        out[rid] = fs
    return out


def main():
    preds = load_preds(PREDS_PATH)
    db_sizes = db_field_sizes(DB) if DB else {}

    # Field size mismatches
    mismatches = []
    small = []
    for r in preds:
        rid = r["race_id"]
        pf = r.get("field_size")
        if pf is not None:
            try:
                pf = int(pf)
            except Exception:
                continue
            if pf <= 3:
                small.append(rid)
        df = db_sizes.get(rid)
        if pf is not None and df is not None and pf != df:
            mismatches.append((rid, pf, df, df - pf))

    mismatches.sort(key=lambda x: x[3], reverse=True)

    print("FIELD SIZE AUDIT")
    print(f"Total predictions: {len(preds)}")
    print(f"With mismatched field_size vs DB: {len(mismatches)}")
    print("Top 10 largest deficits (DB - predicted):")
    for rid, pf, df, delta in mismatches[:10]:
        print(f"  {rid}: predicted={pf} db={df} delta={delta}")

    print(f"\nRaces with predicted field_size <= 3: {len(small)}")
    for rid in small[:10]:
        print("  ", rid)

    # Slice performance by venue/distance (all races)
    def slice_stats(get_key, min_n=50):
        stats = defaultdict(lambda: {"n": 0, "top1": 0, "topk": 0})
        for r in preds:
            k = get_key(r)
            if k is None:
                continue
            stats[k]["n"] += 1
            stats[k]["top1"] += 1 if r.get("correct") else 0
            stats[k]["topk"] += 1 if r.get("top_k_hit") else 0
        rows = []
        for k, v in stats.items():
            n = v["n"]
            if n >= min_n:
                rows.append((k, n, v["top1"]/n, v["topk"]/n))
        rows.sort(key=lambda x: x[2], reverse=True)
        top = rows[:10]
        bottom = rows[-10:]
        return top, bottom

    top_v, bot_v = slice_stats(lambda r: r["race_id"].split("_", 1)[0])
    top_d, bot_d = slice_stats(lambda r: get_dist(r["race_id"]))

    print("\nVENUE PERFORMANCE (top-1 acc, min_n=50)")
    print("Top:")
    for k, n, a, h in top_v:
        print(f"  {k:20s} n={n:4d} top1={a:.3f} topk={h:.3f}")
    print("Bottom:")
    for k, n, a, h in bot_v:
        print(f"  {k:20s} n={n:4d} top1={a:.3f} topk={h:.3f}")

    print("\nDISTANCE PERFORMANCE (top-1 acc, min_n=50)")
    print("Top:")
    for k, n, a, h in top_d:
        print(f"  {k:4d} n={n:4d} top1={a:.3f} topk={h:.3f}")
    print("Bottom:")
    for k, n, a, h in bot_d:
        print(f"  {k:4d} n={n:4d} top1={a:.3f} topk={h:.3f}")

if __name__ == "__main__":
    main()
