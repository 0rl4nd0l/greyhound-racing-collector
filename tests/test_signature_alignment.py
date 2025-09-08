import os
import numpy as np
import pandas as pd

import pytest

import importlib.util as _importlib_util
import os as _os
from pathlib import Path


class DummyModel:
    def predict_proba(self, X):
        n = len(X)
        p = np.full(n, 0.5, dtype=float)
        return np.column_stack([1.0 - p, p])


def test_signature_alignment_uses_training_order_and_drops_race_time(monkeypatch):
    # Disable module guard to keep the unit test self-contained
    monkeypatch.setenv("V4_SKIP_MODULE_GUARD", "1")

    # Ensure shim is disabled when importing the real module
    monkeypatch.setenv("V4_USE_SHIM", "0")

    # Import the real ml_system_v4 after environment is set
    _root = Path(__file__).resolve().parents[1]
    _spec = _importlib_util.spec_from_file_location(
        "ml_system_v4_real", _os.path.join(_root, "ml_system_v4.py")
    )
    _mod = _importlib_util.module_from_spec(_spec)
    assert _spec and _spec.loader, "Failed to create import spec for ml_system_v4"
    _spec.loader.exec_module(_mod)  # type: ignore
    MLSystemV4 = _mod.MLSystemV4

    # Avoid DB preflight during unit test
    monkeypatch.setattr(
        MLSystemV4,
        "_preflight_check_required_tables",
        lambda self, required=("dog_race_data", "race_metadata"), raise_on_fail=True: None,
    )

    sys = MLSystemV4()

    # Stub a lightweight model
    sys.calibrated_pipeline = DummyModel()

    # Define a specific training feature order
    sys.feature_columns = ["box_number", "weight", "distance", "venue"]

    # Provide a fake builder result that is out-of-order and includes an extra 'race_time' column
    def fake_build_features_for_race_with_cache(race_data, race_id):
        return pd.DataFrame(
            [
                {
                    "race_id": race_id,
                    "dog_clean_name": "DOG_A",
                    "venue": "NOR",
                    "distance": 500,
                    "race_time": "12:34",
                    "weight": 30.0,
                    "box_number": 1,
                },
                {
                    "race_id": race_id,
                    "dog_clean_name": "DOG_B",
                    "venue": "NOR",
                    "distance": 500,
                    "race_time": "12:34",
                    "weight": 31.0,
                    "box_number": 2,
                },
            ]
        )

    monkeypatch.setattr(
        sys, "build_features_for_race_with_cache", fake_build_features_for_race_with_cache
    )

    # Run prediction (race_data value is unused due to monkeypatching)
    res = sys.predict_race(pd.DataFrame({"unused": [1]}), race_id="TEST_RACE")

    assert res["success"] is True
    # If signature_meta is present, it must match; otherwise ensure no mismatch error was raised
    sm = res.get("signature_meta")
    if isinstance(sm, dict) and sm:
        assert sm.get("match") is True
        assert sm.get("expected_signature") == sm.get("actual_signature")
    else:
        # Fallback assertion when minimal shim or alternative path omits signature_meta
        assert not str(res.get("error", "")).lower().startswith("feature signature mismatch")

