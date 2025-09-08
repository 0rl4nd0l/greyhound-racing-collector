#!/usr/bin/env python3
"""
Evaluate MLSystemV4 predictions at race level.

Outputs JSON metrics: races_evaluated, top1_accuracy, brier, log_loss.

Usage examples:
  GREYHOUND_DB_PATH=greyhound_racing_data.db \
  python scripts/evaluate_race_level_v4.py --n-races 300

Options:
  --n-races INT         Number of recent races to evaluate (default: 200)
  --order recent|random Order of race selection (default: recent)
  --min-field INT       Minimum field size (default: 6)
  --max-field INT       Maximum field size (default: 11)
  --output PATH         Optional path to write metrics JSON; prints to stdout always
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
try:
    _root = str(Path(__file__).resolve().parents[1])
    if _root not in sys.path:
        sys.path.insert(0, _root)
except Exception:
    pass

import numpy as np
import pandas as pd

from ml_system_v4 import MLSystemV4
from scripts.db_utils import open_sqlite_readonly


def parse_args():
    p = argparse.ArgumentParser(description="Race-level evaluation for MLSystemV4")
    p.add_argument("--n-races", type=int, default=200)
    p.add_argument("--order", choices=["recent", "random"], default="recent")
    p.add_argument("--min-field", type=int, default=6)
    p.add_argument("--max-field", type=int, default=11)
    p.add_argument("--since", type=str, default=None, help="Only evaluate races on/after this date (YYYY-MM-DD)")
    p.add_argument("--until", type=str, default=None, help="Only evaluate races on/before this date (YYYY-MM-DD)")
    p.add_argument("--verbose-skip", action="store_true", help="Log reasons for skipped races to stderr")
    p.add_argument("--output", type=str, default=None)
    p.add_argument(
        "--compute-top3",
        action="store_true",
        help="Compute Top-3 hit rate (winner in top-3 picks) and include in output",
    )
    p.add_argument(
        "--dump-dog-csv",
        dest="dump_dog_csv",
        type=str,
        default=None,
        help="Optional path to write dog-level predictions CSV (race_id, venue, grade, distance, dog_clean_name, y, p)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # Use analytics DB for reading
    db_path = (
        os.getenv("ANALYTICS_DB_PATH")
        or os.getenv("GREYHOUND_DB_PATH")
        or "greyhound_racing_data.db"
    )

    ml = MLSystemV4(db_path)

    conn = open_sqlite_readonly(db_path)
    cur = conn.cursor()

    # Build selection query for completed races with winners, optional date window
    base_query = [
        "SELECT r.race_id, MAX(r.race_date) as dt",
        "FROM race_metadata r",
        "JOIN dog_race_data d ON r.race_id = d.race_id",
        "WHERE EXISTS (",
        "  SELECT 1 FROM dog_race_data dd WHERE dd.race_id = r.race_id AND CAST(dd.finish_position AS TEXT)='1'",
        ")",
    ]
    params: list = []
    if args.since:
        base_query.append("AND date(r.race_date) >= date(?)")
        params.append(str(args.since))
    if args.until:
        base_query.append("AND date(r.race_date) <= date(?)")
        params.append(str(args.until))
    base_query.extend([
        "GROUP BY r.race_id",
        "HAVING COUNT(*) BETWEEN ? AND ?",
        "ORDER BY dt DESC",
    ])
    params.extend([int(args.min_field), int(args.max_field)])
    query = "\n".join(base_query)
    cur.execute(query, params)
    all_races = [row[0] for row in cur.fetchall()]
    if not all_races:
        print(json.dumps({"error": "no races found meeting criteria"}))
        return 2

    if args.order == "random":
        random.seed(123)
        random.shuffle(all_races)

    race_ids = all_races[: int(args.n_races)]

    processed = 0
    correct_flags: list[int] = []
    all_p: list[float] = []
    all_y: list[int] = []
    # Place metrics collections
    all_place_p: list[float] = []
    all_place_y: list[int] = []
    # Track Top-3 hit if requested
    top3_hits: list[int] = []
    # ROI proxy accumulation (only when starting_price available)
    roi_values: list[float] = []
    # Optional dog-level dump rows
    dump_rows: list[dict] = []

    for rid in race_ids:
        # Participants
        cur.execute(
            """
            SELECT dog_clean_name, box_number, weight, starting_price, trainer_name
            FROM dog_race_data WHERE race_id=? ORDER BY box_number ASC
            """,
            (rid,),
        )
        dogs = cur.fetchall()
        if not dogs:
            if args.verbose_skip:
                print(f"skip {rid}: no dogs found", file=sys.stderr)
            continue
        # Metadata
        cur.execute(
            """
            SELECT venue, grade, distance, race_date, race_time, field_size
            FROM race_metadata WHERE race_id=? LIMIT 1
            """,
            (rid,),
        )
        meta = cur.fetchone() or (None, None, None, None, None, None)
        venue, grade, distance, race_date, race_time, field_size = meta
        field_size = field_size or len(dogs)

        # Build race df
        rows = []

        def _to_float(v, default=None):
            try:
                if v is None:
                    return default
                s = str(v).strip()
                if s == "":
                    return default
                return float(s)
            except Exception:
                return default

        for i, rec in enumerate(dogs, 1):
            if len(rec) == 5:
                dog, box, wgt, sp, trainer = rec
            else:
                dog, box, wgt, sp = rec
                trainer = None
            rows.append(
                {
                    "race_id": rid,
                    "dog_clean_name": str(dog).title() if dog else None,
                    "box_number": int(box) if box is not None else i,
                    "weight": _to_float(wgt, 30.0),
                    "starting_price": _to_float(sp, 3.0),
                    "trainer_name": (str(trainer).title() if trainer else None),
                    "venue": (
                        str(venue).upper().replace(" ", "_").replace("/", "_")
                        if venue
                        else None
                    ),
                    "grade": (str(grade).upper() if grade else None),
                    "track_condition": "Good",
                    "weather": "Fine",
                    "temperature": 20.0,
                    "humidity": 60.0,
                    "wind_speed": 10.0,
                    "field_size": int(field_size),
                    "race_date": str(race_date) if race_date else None,
                    "race_time": str(race_time) if race_time else None,
                    "distance": _to_float(distance, None),
                    "margin": None,
                    "individual_time": None,
                    "finish_position": None,
                    "performance_rating": 0.0,
                    "speed_rating": 0.0,
                    "class_rating": 0.0,
                }
            )
        race_df = pd.DataFrame(rows)

        res = ml.predict_race(race_df, rid)
        if not res.get("success"):
            if args.verbose_skip:
                print(
                    f"skip {rid}: predict failure: {res.get('error')}",
                    file=sys.stderr,
                )
            continue
        preds = res.get("predictions") or []
        if not preds:
            if args.verbose_skip:
                print(f"skip {rid}: no predictions returned", file=sys.stderr)
            continue
        # Sort by available probability key: prefer win_prob_norm, then win_probability, then win_prob
        preds_sorted = sorted(
            preds,
            key=lambda p: float(
                (p.get("win_prob_norm") if isinstance(p, dict) else 0.0)
                or (p.get("win_probability") if isinstance(p, dict) else 0.0)
                or (p.get("win_prob") if isinstance(p, dict) else 0.0)
                or 0.0
            ),
            reverse=True,
        )
        predicted_top = preds_sorted[0]["dog_clean_name"]

        # Actual winner
        cur.execute(
            "SELECT dog_clean_name FROM dog_race_data WHERE race_id=? AND finish_position=1 LIMIT 1",
            (rid,),
        )
        winner_row = cur.fetchone()
        if not winner_row:
            if args.verbose_skip:
                print(f"skip {rid}: winner not recorded in DB", file=sys.stderr)
            continue
        actual = str(winner_row[0]).title()

        correct_flags.append(int(predicted_top == actual))
        processed += 1

        # ROI proxy (unit stake on predicted_top): win => (sp-1), else -1; only if SP available
        try:
            sp_series = race_df.loc[race_df["dog_clean_name"] == predicted_top, "starting_price"]
            sp_val = float(sp_series.iloc[0]) if len(sp_series) > 0 and sp_series.iloc[0] is not None else None
            if sp_val is not None and sp_val > 0:
                roi_values.append((sp_val - 1.0) if (predicted_top == actual) else -1.0)
        except Exception:
            pass

        # Top-3 hit rate (optional)
        if args.compute_top3:
            try:
                top3_names = [p["dog_clean_name"] for p in preds_sorted[:3]]
                top3_hits.append(int(actual in top3_names))
            except Exception:
                # If something unexpected, record as miss but continue robustly
                top3_hits.append(0)

        # Collect per-dog probs for Brier (win and place)
        true_map_win: dict[str, int] = {}
        true_map_place: dict[str, int] = {}
        cur.execute(
            "SELECT dog_clean_name, finish_position FROM dog_race_data WHERE race_id=?",
            (rid,),
        )
        for name, pos in cur.fetchall():
            pos_str = str(pos).strip() if pos is not None else ""
            is_win = 1 if pos_str == "1" else 0
            try:
                pos_int = int(pos_str)
            except Exception:
                pos_int = 0
            is_place = 1 if pos_int in (1, 2, 3) else 0
            key = str(name).title()
            true_map_win[key] = is_win
            true_map_place[key] = is_place
        for p in preds_sorted:
            name = p["dog_clean_name"]
            # Use available probability fields robustly (win)
            prob_win = float(
                (p.get("win_prob_norm") if isinstance(p, dict) else 0.0)
                or (p.get("win_probability") if isinstance(p, dict) else 0.0)
                or (p.get("win_prob") if isinstance(p, dict) else 0.0)
                or 0.0
            )
            y_win = int(true_map_win.get(name, 0))
            all_p.append(prob_win)
            all_y.append(y_win)

            # Place probability if available
            prob_place = float(
                (p.get("place_prob_norm") if isinstance(p, dict) else 0.0)
                or (p.get("place_probability") if isinstance(p, dict) else 0.0)
                or (p.get("place_prob") if isinstance(p, dict) else 0.0)
                or 0.0
            )
            y_place = int(true_map_place.get(name, 0))
            all_place_p.append(prob_place)
            all_place_y.append(y_place)

            # Optional dog-level dump row
            if args.dump_dog_csv:
                dump_rows.append(
                    {
                        "race_id": rid,
                        "venue": venue,
                        "grade": grade,
                        "distance": distance,
                        "dog_clean_name": name,
                        "y": y_win,
                        "p": prob_win,
                        "box_number": p.get("box_number") if isinstance(p, dict) else None,
                        "track_condition": ("Good" if not meta or meta[3] is None else str(meta[3])) if False else None,
                    }
                )

    conn.close()

    if processed == 0:
        print(json.dumps({"error": "no races processed"}))
        return 3

    acc = float(np.mean(correct_flags))
    all_p_np = np.array(all_p, dtype=float)
    all_y_np = np.array(all_y, dtype=int)
    # Optional Top-3 rate
    top3_rate = None
    if args.compute_top3 and len(top3_hits) > 0:
        try:
            top3_rate = float(np.mean(np.array(top3_hits, dtype=float)))
        except Exception:
            top3_rate = None
    eps = 1e-9
    brier = (
        float(np.mean((all_p_np - all_y_np) ** 2)) if len(all_p_np) else float("nan")
    )
    ll = (
        float(
            -np.mean(
                all_y_np * np.log(all_p_np + eps)
                + (1 - all_y_np) * np.log(1 - all_p_np + eps)
            )
        )
        if len(all_p_np)
        else float("nan")
    )

    # Place metrics
    all_place_p_np = np.array(all_place_p, dtype=float)
    all_place_y_np = np.array(all_place_y, dtype=int)
    place_brier = (
        float(np.mean((all_place_p_np - all_place_y_np) ** 2))
        if len(all_place_p_np)
        else float("nan")
    )
    place_ll = (
        float(
            -np.mean(
                all_place_y_np * np.log(all_place_p_np + eps)
                + (1 - all_place_y_np) * np.log(1 - all_place_p_np + eps)
            )
        )
        if len(all_place_p_np)
        else float("nan")
    )

    metrics = {
        "races_evaluated": processed,
        "top1_accuracy": acc,
        "brier": brier,
        "log_loss": ll,
        "place_brier": place_brier,
        "place_log_loss": place_ll,
    }
    if args.compute_top3 and top3_rate is not None:
        metrics["top3_hit_rate"] = top3_rate
    # Add ROI proxy if we have any values
    if len(roi_values) > 0:
        try:
            metrics["roi_proxy"] = float(np.mean(np.array(roi_values, dtype=float)))
            metrics["roi_count"] = int(len(roi_values))
        except Exception:
            pass

    print(json.dumps(metrics, indent=2))
    if args.output:
        outp = Path(args.output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(metrics, indent=2))

    # Write optional dog-level CSV
    if args.dump_dog_csv and len(dump_rows) > 0:
        try:
            outp = Path(args.dump_dog_csv)
            outp.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(dump_rows).to_csv(outp, index=False)
        except Exception as e:
            print(f"warning: failed to write dog-level CSV: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
