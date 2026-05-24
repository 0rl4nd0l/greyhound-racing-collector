#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

import pandas as pd
import logging

# Configure logging so MLSystemV4 INFO logs are visible
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Ensure local imports work when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_system_v4 import MLSystemV4  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Debug MLSystemV4 confidence signals for a single race CSV"
    )
    parser.add_argument("--race-csv", required=True, help="Path to the race CSV file")
    parser.add_argument(
        "--race-id",
        default=None,
        help="Race identifier string (e.g., 'Race 12 - GUNNEDAH - 2025-09-03'). Defaults to CSV stem",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable V4_CONFIDENCE_DEBUG gated logging while running",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Set V4_CONFIDENCE_DEBUG_FILTER to only log for matching race-id substrings",
    )

    args = parser.parse_args()

    if args.debug:
        os.environ["V4_CONFIDENCE_DEBUG"] = "1"
    if args.filter:
        os.environ["V4_CONFIDENCE_DEBUG_FILTER"] = args.filter

    csv_path = Path(args.race_csv).expanduser().resolve()
    if not csv_path.exists():
        print(f"Error: race CSV not found at {csv_path}", file=sys.stderr)
        sys.exit(2)

    race_id = args.race_id or csv_path.stem

    df = pd.read_csv(csv_path)

    system = MLSystemV4()

    # Attempt to map columns using built-in preprocessor for upcoming races
    try:
        processed_df = system.preprocess_upcoming_race_csv(df, race_id)
    except Exception:
        processed_df = df

    result = system.predict_race(processed_df, race_id)

    if not result or not result.get("success", False):
        print("Prediction failed:")
        print(result)
        sys.exit(1)

    preds = result.get("predictions", [])

    # Print compact per-runner table
    print("dog_name,box,win_prob_raw,win_prob_norm,confidence,confidence_label")
    for p in preds:
        dog = p.get("dog_clean_name") or p.get("dog_name")
        box = p.get("box_number")
        raw = p.get("win_prob_raw")
        norm = p.get("win_prob_norm", p.get("win_probability"))
        conf = p.get("confidence")
        label = p.get("confidence_label")
        try:
            raw_f = f"{float(raw):.6f}" if raw is not None else ""
        except Exception:
            raw_f = ""
        try:
            norm_f = f"{float(norm):.6f}" if norm is not None else ""
        except Exception:
            norm_f = ""
        try:
            conf_f = f"{float(conf):.6f}" if conf is not None else ""
        except Exception:
            conf_f = ""
        print(f"{dog},{box},{raw_f},{norm_f},{conf_f},{label}")

    # Compute race-level signals for convenience
    try:
        import numpy as np
        import math

        probs = [float(p.get("win_prob_norm", p.get("win_probability", 0.0))) for p in preds]
        n = len(probs)
        if n > 0:
            p_sorted = sorted(probs, reverse=True)
            p1 = p_sorted[0]
            p2 = p_sorted[1] if n > 1 else 0.0
            H = 0.0
            eps = 1e-12
            H = -sum([pr * math.log(max(pr, eps)) for pr in probs]) / math.log(max(n, 2))
            print(f"race_signals: p1={p1:.4f}, p2={p2:.4f}, H={H:.4f}, n_dogs={n}")
    except Exception:
        pass


if __name__ == "__main__":
    main()

