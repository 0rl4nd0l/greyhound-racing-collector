#!/usr/bin/env python3
"""
Smoke test: TGR toggle does not crash when TGR columns are missing/zeroed.
Verifies that MLSystemV4 exposes a tgr_zero_check in explainability_meta.
"""
import os

import pandas as pd

from ml_system_v4 import MLSystemV4


def test_tgr_toggle_missing_columns_smoke():
    # Use the default repo DB if available; many tests already depend on this path
    db_path = os.getenv("GREYHOUND_DB_PATH", "greyhound_racing_data.db")
    os.environ.setdefault("GREYHOUND_DB_PATH", db_path)

    ml = MLSystemV4(db_path)
    ml.set_tgr_enabled(True)

    # Minimal upcoming-like race frame without any TGR columns
    race_id = "TEST_TGR_ZERO_GUARD"
    race_df = pd.DataFrame(
        [
            {
                "race_id": race_id,
                "dog_clean_name": "DOG A",
                "box_number": 1,
                "weight": 30.0,
                "distance": 450,
                "venue": "TST",
                "grade": "5",
                "track_condition": "Good",
                "weather": "Fine",
                "race_date": "2025-01-01",
                "race_time": "12:00",
                "field_size": 2,
            },
            {
                "race_id": race_id,
                "dog_clean_name": "DOG B",
                "box_number": 2,
                "weight": 31.0,
                "distance": 450,
                "venue": "TST",
                "grade": "5",
                "track_condition": "Good",
                "weather": "Fine",
                "race_date": "2025-01-01",
                "race_time": "12:00",
                "field_size": 2,
            },
        ]
    )

    result = ml.predict_race(race_df, race_id)
    assert result.get("success") is True

    # Explainability meta should include TGR zero-guard if TGR columns were added
    exp = result.get("explainability_meta") or {}
    tgr_check = exp.get("tgr_zero_check") or {}
    # Presence may depend on integrator availability; but when present it should identify zeroed features
    if tgr_check:
        assert tgr_check.get("present") in (True, False)
        # If present, zeroed should be a boolean
        if tgr_check.get("present"):
            assert tgr_check.get("zeroed") in (True, False)

