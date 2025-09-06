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
from pathlib import Path

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
    p.add_argument("--output", type=str, default=None)
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

    cur.execute(
        """
        SELECT r.race_id, MAX(r.race_date) as dt
        FROM race_metadata r
        JOIN dog_race_data d ON r.race_id = d.race_id
        GROUP BY r.race_id
        HAVING COUNT(*) BETWEEN ? AND ?
        ORDER BY dt DESC
        """,
        [int(args.min_field), int(args.max_field)],
    )
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

    for rid in race_ids:
        # Participants
        cur.execute(
            """
            SELECT dog_clean_name, box_number, weight, starting_price
            FROM dog_race_data WHERE race_id=? ORDER BY box_number ASC
            """,
            (rid,),
        )
        dogs = cur.fetchall()
        if not dogs:
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

        for i, (dog, box, wgt, sp) in enumerate(dogs, 1):
            rows.append(
                {
                    "race_id": rid,
                    "dog_clean_name": str(dog).title() if dog else None,
                    "box_number": int(box) if box is not None else i,
                    "weight": _to_float(wgt, 30.0),
                    "starting_price": _to_float(sp, 3.0),
                    "trainer_name": None,
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
            continue
        preds = res.get("predictions") or []
        if not preds:
            continue
        preds_sorted = sorted(
            preds, key=lambda p: float(p.get("win_prob_norm") or 0.0), reverse=True
        )
        predicted_top = preds_sorted[0]["dog_clean_name"]

        # Actual winner
        cur.execute(
            "SELECT dog_clean_name FROM dog_race_data WHERE race_id=? AND finish_position=1 LIMIT 1",
            (rid,),
        )
        winner_row = cur.fetchone()
        if not winner_row:
            continue
        actual = str(winner_row[0]).title()

        correct_flags.append(int(predicted_top == actual))
        processed += 1

        # Collect per-dog probs for Brier
        true_map: dict[str, int] = {}
        cur.execute(
            "SELECT dog_clean_name, finish_position FROM dog_race_data WHERE race_id=?",
            (rid,),
        )
        for name, pos in cur.fetchall():
            is_win = 1 if str(pos).strip() == "1" else 0
            true_map[str(name).title()] = is_win
        for p in preds_sorted:
            name = p["dog_clean_name"]
            prob = float(p.get("win_prob_norm") or 0.0)
            y = int(true_map.get(name, 0))
            all_p.append(prob)
            all_y.append(y)

    conn.close()

    if processed == 0:
        print(json.dumps({"error": "no races processed"}))
        return 3

    acc = float(np.mean(correct_flags))
    all_p_np = np.array(all_p, dtype=float)
    all_y_np = np.array(all_y, dtype=int)
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

    metrics = {
        "races_evaluated": processed,
        "top1_accuracy": acc,
        "brier": brier,
        "log_loss": ll,
    }

    print(json.dumps(metrics, indent=2))
    if args.output:
        outp = Path(args.output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
