import json
import os

import pytest

# Import the Flask app
from app import app

CSV_KEYS = [
    "csv_historical_races",
    "csv_win_rate",
    "csv_place_rate",
    "csv_avg_finish_position",
    "csv_best_finish_position",
    "csv_recent_form",
    "csv_avg_time",
    "csv_best_time",
]


def _pick_any_csv():
    # Prefer project upcoming_races dir
    dirs = [
        os.path.join(os.getcwd(), "upcoming_races"),
        os.path.join(os.getcwd(), "data", "upcoming_races"),
    ]
    for d in dirs:
        if os.path.isdir(d):
            for name in os.listdir(d):
                if name.endswith(".csv") and not name.startswith("."):
                    return name
    return None


def _extract_predictions(container):
    # Walk up to two levels of 'prediction' nesting to find the actual payload
    d = container
    for _ in range(2):
        if isinstance(d, dict) and isinstance(d.get("prediction"), dict):
            d = d["prediction"]
        else:
            break
    for key in ("predictions", "enhanced_predictions"):
        val = d.get(key) if isinstance(d, dict) else None
        if isinstance(val, list):
            return val
    # Fallback: some degraded responses return 'prediction_details'
    pd = container.get("prediction_details") if isinstance(container, dict) else None
    if isinstance(pd, dict):
        for key in ("predictions", "enhanced_predictions"):
            val = pd.get(key)
            if isinstance(val, list):
                return val
    return None


def _latest_prediction_json_path():
    pred_dir = os.path.join(os.getcwd(), "predictions")
    if not os.path.isdir(pred_dir):
        return None
    latest_path = None
    latest_mtime = 0
    for name in os.listdir(pred_dir):
        if not name.endswith(".json"):
            continue
        p = os.path.join(pred_dir, name)
        try:
            mtime = os.path.getmtime(p)
        except Exception:
            continue
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = p
    return latest_path


def test_enhanced_prediction_csv_enrichment_present_and_persisted():
    race_filename = _pick_any_csv()
    if not race_filename:
        pytest.skip("No CSV files found in upcoming_races")

    client = app.test_client()

    payload = {"race_filename": race_filename}
    resp = client.post("/api/predict_single_race_enhanced", json=payload)
    assert resp.status_code == 200

    data = resp.get_json(silent=True)
    assert data and data.get("success") is True

    preds = _extract_predictions(data)
    if isinstance(preds, list) and len(preds) > 0:
        # Verify csv_* enrichment in API response
        for p in preds:
            for k in CSV_KEYS:
                assert k in p, f"Missing {k} in response prediction"
                assert p[k] is not None, f"{k} is None in response prediction"
    else:
        # If response shape varies in some environments, fall back to verifying the persisted artifact only
        print(
            "Note: No predictions found in API response payload; validating persisted JSON only."
        )

    # Verify enrichment persisted in saved JSON
    latest = _latest_prediction_json_path()
    assert latest is not None, "No prediction JSON files found after call"

    with open(latest, "r", encoding="utf-8") as f:
        saved = json.load(f)

    saved_preds = _extract_predictions(saved)
    assert isinstance(saved_preds, list) and len(saved_preds) > 0

    for p in saved_preds:
        for k in CSV_KEYS:
            assert k in p, f"Missing {k} in saved prediction"
            assert p[k] is not None, f"{k} is None in saved prediction"
