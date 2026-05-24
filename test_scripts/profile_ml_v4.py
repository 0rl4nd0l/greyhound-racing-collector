import os
import sys
import time

import memory_profiler

# Ensure project root is on sys.path when running from test_scripts
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from ml_system_v4 import MLSystemV4


@memory_profiler.profile
def profile_system():
    system = MLSystemV4()

    # Construct a small mock upcoming race for prediction
    race_id = "Race 1 - DAPT - 2025-08-04"
    mock_race = pd.DataFrame(
        {
            "dog_clean_name": [f"Dog {i}" for i in range(1, 7)],
            "box_number": list(range(1, 7)),
            "weight": [30.5, 29.8, 31.2, 28.9, 32.0, 30.1],
            "distance": [500] * 6,
            "venue": ["DAPT"] * 6,
            "grade": ["5"] * 6,
            "track_condition": ["Good"] * 6,
            "weather": ["Fine"] * 6,
            "race_date": ["2025-08-04"] * 6,
            "race_time": ["12:00"] * 6,
            "historical_avg_position": [3.2, 4.1, 2.8, 5.0, 3.8, 4.5],
            "historical_win_rate": [0.12, 0.08, 0.15, 0.05, 0.10, 0.07],
            "venue_specific_avg_position": [3.0, 4.0, 2.9, 5.1, 3.7, 4.4],
            "days_since_last_race": [10, 14, 7, 21, 12, 9],
        }
    )

    start = time.time()
    result = system.predict_race(mock_race, race_id)
    end = time.time()

    print(f"Execution time: {end - start:.2f} seconds")
    print(
        f"Success: {result.get('success')}, Dogs: {len(result.get('predictions', []))}"
    )
    return result


if __name__ == "__main__":
    profile_system()
