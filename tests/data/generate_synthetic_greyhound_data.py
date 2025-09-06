import argparse
import hashlib
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_block(
    start_time: datetime, n_races: int, rng: random.Random, np_rng: np.random.Generator
):
    rows_hist = []
    rows_race = []
    for i in range(n_races):
        race_id = f"R{start_time.strftime('%Y%m%d')}-{i+1:03d}"
        race_time = start_time + timedelta(minutes=i * 7)
        # latent skill per dog
        dog_skills = np_rng.normal(0.0, 1.0, 10)
        winner_ix = int(np.argmax(dog_skills + np_rng.normal(0.0, 0.2, 10)))
        for d in range(10):
            dog_id = f"D{race_id}-{d+1:02d}"
            dog_name = f"Dog_{race_id}_{d+1:02d}"
            weight = float(25 + np_rng.normal(0, 2))
            age = int(np.clip(np_rng.normal(36, 6), 18, 96))
            recent1 = int(np.clip(np_rng.integers(1, 9), 1, 8))
            recent2 = int(np.clip(np_rng.integers(1, 9), 1, 8))
            recent3 = int(np.clip(np_rng.integers(1, 9), 1, 8))
            split = float(np.clip(np_rng.normal(5.5, 0.3), 4.5, 6.8))
            track = rng.choice(["WENT", "SAND", "MEAD", "BEND"])  # venue code
            grade = rng.choice(["A1", "A2", "A3", "B1", "B2"])
            box = d + 1
            rows_hist.append(
                {
                    "race_id": race_id,
                    "dog_id": dog_id,
                    "dog_name": dog_name,
                    "age_months": age,
                    "weight_kg": weight,
                    "recent_finish1": recent1,
                    "recent_finish2": recent2,
                    "recent_finish3": recent3,
                    "avg_split_time": split,
                    "track_code": track,
                    "grade": grade,
                    "box": box,
                    "race_datetime": race_time.isoformat(),
                }
            )
        winner_dog_id = f"D{race_id}-{winner_ix+1:02d}"
        rows_race.append(
            {
                "race_id": race_id,
                "race_datetime": race_time.isoformat(),
                "weather": rng.choice(["Fine", "Rain", "Overcast"]),
                "track_condition": rng.choice(["Fast", "Good", "Slow"]),
                "winning_time": float(np.clip(np_rng.normal(29.9, 0.6), 28.0, 31.5)),
                "winner_dog_id": winner_dog_id,
            }
        )
    return pd.DataFrame(rows_hist), pd.DataFrame(rows_race)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", required=True)
    ap.add_argument("--races-train", type=int, default=40)
    ap.add_argument("--races-test", type=int, default=15)
    ap.add_argument("--format", choices=["normalized"], default="normalized")
    args = ap.parse_args()

    out_dir = Path(args.out)
    hist_dir = out_dir / "historical"
    race_dir = out_dir / "race_data"
    hist_dir.mkdir(parents=True, exist_ok=True)
    race_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    train_start = datetime(2020, 1, 1)
    test_start = datetime(2022, 1, 1)

    hist_tr, race_tr = generate_block(train_start, args.races_train, rng, np_rng)
    hist_te, race_te = generate_block(test_start, args.races_test, rng, np_rng)

    p1 = hist_dir / "train_form_guide.csv"
    p2 = hist_dir / "test_form_guide.csv"
    p3 = race_dir / "train_race_results.csv"
    p4 = race_dir / "test_race_results.csv"

    hist_tr.to_csv(p1, index=False)
    hist_te.to_csv(p2, index=False)
    race_tr.to_csv(p3, index=False)
    race_te.to_csv(p4, index=False)

    hashes = {p.name: sha256(p) for p in [p1, p2, p3, p4]}
    with open(out_dir / ".hashes.json", "w") as f:
        json.dump(hashes, f, indent=2)

    print("Wrote:", p1, p2, p3, p4)


if __name__ == "__main__":
    main()
