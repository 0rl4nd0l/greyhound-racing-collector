#!/usr/bin/env python3
"""
Predict a historical race (ML System V4) using DB data
=====================================================

Loads participants and race metadata from the database by race_id, builds
ML System V4 features (with optional TGR enhancement), and prints results.

Usage:
  GREYHOUND_DB_PATH=databases/comprehensive_greyhound_data.db \
  TGR_ENABLED=1 \
  python scripts/predict_historical_race_v4.py --db databases/comprehensive_greyhound_data.db --race-id ap_k_2025-02-18_2

Flags:
  --tgr 1|0           Enable/disable TGR integration at runtime (default: env TGR_ENABLED)
"""

import argparse
import os
import sqlite3
import sys
from typing import Any, Dict, List

import pandas as pd

from scripts.db_utils import open_sqlite_readonly

# Ensure project root on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ml_system_v4 import MLSystemV4


def fetch_race_df(db_path: str, race_id: str) -> pd.DataFrame:
    conn = open_sqlite_readonly(db_path)
    cur = conn.cursor()
    try:
        # Participants
        cur.execute(
            """
            SELECT dog_clean_name, box_number, weight, starting_price
            FROM dog_race_data
            WHERE race_id = ?
            ORDER BY box_number ASC
            """,
            [race_id],
        )
        dogs = cur.fetchall()
        if not dogs:
            raise RuntimeError(f"No participants found for race_id={race_id}")

        # Race metadata
        cur.execute(
            """
            SELECT venue, grade, distance, race_date, race_time, field_size
            FROM race_metadata
            WHERE race_id = ?
            LIMIT 1
            """,
            [race_id],
        )
        meta = cur.fetchone()

        if not meta:
            venue, grade, distance, race_date, race_time, field_size = (
                None,
                None,
                None,
                None,
                None,
                None,
            )
        else:
            venue, grade, distance, race_date, race_time, field_size = meta

        if field_size is None:
            field_size = len(dogs)

        # Helper for safe numeric conversion
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

        rows: List[Dict[str, Any]] = []
        for i, (dog, box, wgt, sp) in enumerate(dogs, 1):
            rows.append(
                {
                    "race_id": race_id,
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

        df = pd.DataFrame(rows)
        return df
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Predict a historical race using ML System V4"
    )
    parser.add_argument("--db", dest="db_path", required=True, help="Path to SQLite DB")
    parser.add_argument(
        "--race-id", dest="race_id", required=True, help="Race ID to predict"
    )
    parser.add_argument(
        "--tgr",
        dest="tgr",
        choices=["0", "1"],
        default=None,
        help="Enable (1) or disable (0) TGR at runtime",
    )
    args = parser.parse_args()

    # Ensure environment for DB detection
    os.environ.setdefault("GREYHOUND_DB_PATH", args.db_path)

    # Build race DataFrame
    race_df = fetch_race_df(args.db_path, args.race_id)

    # Initialize ML V4
    ml = MLSystemV4(args.db_path)

    # TGR toggle by CLI overrides env (TGR_ENABLED)
    tgr_env = os.getenv("TGR_ENABLED")
    tgr_enabled = None
    if args.tgr is not None:
        tgr_enabled = args.tgr == "1"
    elif tgr_env is not None:
        tgr_enabled = tgr_env.strip() in ("1", "true", "yes", "True", "TRUE")

    if tgr_enabled is not None and hasattr(ml, "set_tgr_enabled"):
        ml.set_tgr_enabled(tgr_enabled)

    # Predict
    result = ml.predict_race(race_df, args.race_id)

    if not result.get("success"):
        print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

    preds = (
        result.get("predictions", []) or result.get("enhanced_predictions", []) or []
    )

    # Display
    print(f"\n✅ Prediction successful for {args.race_id}")
    print(
        "Rank | Box | Dog Name                    | Win Prob | Confidence | TGR races | TGR win% | TGR avg pos"
    )
    print("-" * 106)

    # Sort by normalized prob if available
    preds_sorted = sorted(
        preds,
        key=lambda p: float(p.get("win_prob_norm") or p.get("win_probability") or 0.0),
        reverse=True,
    )

    def _fmt(v, default="-"):
        try:
            if v is None:
                return default
            return f"{float(v):.3f}"
        except Exception:
            return str(v)

    for i, p in enumerate(preds_sorted, 1):
        name = p.get("dog_clean_name") or p.get("dog_name") or "Unknown"
        box = p.get("box_number") or "?"
        prob = p.get("win_prob_norm") or p.get("win_probability") or 0.0
        conf = p.get("confidence") or p.get("confidence_level") or 0.0
        tgr_total = p.get("tgr_total_races")
        tgr_win_rate = p.get("tgr_win_rate")
        tgr_avg_pos = p.get("tgr_avg_finish_position")
        print(
            f"{i:4} | {box:3} | {str(name)[:26].ljust(26)} | {float(prob):7.3f} | {_fmt(conf):10} | "
            f"{tgr_total if tgr_total is not None else '-':>9} | {_fmt(tgr_win_rate):8} | {_fmt(tgr_avg_pos):11}"
        )

    print("\n📎 Additional info:")
    if result.get("fallback_used"):
        print(f"  Fallback used: {result.get('fallback_reason')}")
    if result.get("model_info"):
        print(f"  Model info: {result.get('model_info')}")


if __name__ == "__main__":
    main()
