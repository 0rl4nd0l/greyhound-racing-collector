#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Simulate anomalous place EV predictions to trigger auto-disable")
    parser.add_argument("--races", type=int, default=6, help="Number of synthetic races to write")
    args = parser.parse_args()

    out_dir = os.path.join("predictions")
    os.makedirs(out_dir, exist_ok=True)

    now = datetime.now()
    for i in range(args.races):
        race_id = f"SIM_RACE_{now.strftime('%Y%m%d')}_{i+1}"
        preds = []
        # Create 8 runners with skewed place probs and negative EVs
        for j in range(8):
            place_prob = 0.99 if j == 0 else 0.001
            preds.append({
                "dog_name": f"DOG_{j+1}",
                "win_prob_norm": 0.125,
                "place_prob_norm": place_prob,
                "ev_win": -0.05,
                "ev_place": -0.25,
                "predicted_rank": j+1,
            })
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        out_path = os.path.join(out_dir, f"{race_id}_{ts}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "success": True,
                "race_id": race_id,
                "predictions": preds,
                "prediction_timestamp": ts,
            }, f, indent=2)
        # Make timestamps distinct
        import time as _t
        _t.sleep(0.2)

    print(f"Wrote {args.races} synthetic prediction files in {out_dir}")


if __name__ == "__main__":
    main()
