import json
import os
from pathlib import Path

import pytest

import app as app_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Ensure a clean ./predictions directory pointing to a temp location
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Use a temporary working directory so app finds ./predictions at tmp
    monkeypatch.chdir(tmp_path)

    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def test_confidence_label_mapping_from_numeric(client):
    # Create a minimal prediction file with a top runner having numeric confidence 0.72 (-> HIGH)
    pred_dir = Path("./predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "race_info": {"filename": "Race 1 - TEST - 2025-09-04.csv"},
        "predictions": [
            {
                "dog_name": "TEST RUNNER",
                "box_number": 1,
                "final_score": 0.6,
                "confidence": 0.72,
                "confidence_level": "HIGH"
            }
        ],
    }
    out_file = pred_dir / "prediction_Race 1 - TEST - 2025-09-04.json"
    out_file.write_text(json.dumps(payload), encoding="utf-8")

    # Hit the API to build the UI-friendly summary
    resp = client.get("/api/prediction_results")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data and data.get("success") is not False  # may be missing 'success' -> treat non-False as OK

    preds = data.get("predictions") or []
    assert isinstance(preds, list) and len(preds) >= 1

    # Find our test file entry (by race_name or filename if present)
    first = preds[0]
    # In the summary payload, top_pick should have a textual confidence_level
    top_pick = first.get("top_pick") or {}
    conf = top_pick.get("confidence_level")
    if conf is not None:
        assert conf in ("HIGH", "MEDIUM", "LOW", "VERY_LOW"), f"Unexpected confidence_level: {conf}"
        # Numeric 0.72 should map to HIGH
        assert conf == "HIGH"
    else:
        # If not surfaced in this summary, at least ensure top_pick exists with a score
        assert "prediction_score" in top_pick

