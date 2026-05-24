import json
import os
from pathlib import Path

import pytest

import app as app_module


class _FakeEnhancedService:
    def predict_race_file_enhanced(self, race_file_path: str, tgr_enabled=None):
        # Return a minimal enhanced payload with recommendations so UI sees them
        return {
            "success": True,
            "race_id": Path(race_file_path).stem,
            "predictions": [
                {
                    "dog_name": "DOG A",
                    "box_number": 1,
                    "win_prob": 0.20,
                    "csv_win_rate": 0.05,
                    "csv_avg_finish_position": 5.2,
                },
                {"dog_name": "DOG B", "box_number": 2, "win_prob": 0.19},
            ],
            "recommendations": [
                "CRITICAL: Competitive race – weak favorite (small lead, poor recent win/place history)"
            ],
            "prediction_methods_used": ["ml_system", "sp_tiebreaker"],
        }


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Set up a temp upcoming dir with a simple CSV to satisfy the endpoint's enhancement step
    upc = tmp_path / "upcoming"
    upc.mkdir(parents=True, exist_ok=True)

    csv_path = upc / "Race 1 - TST - 2025-09-04.csv"
    csv_path.write_text(
        "Dog Name,BOX,WGT,SP\n1. Alpha,1,30,3.5\n\n2. Beta,2,31,4.0\n",
        encoding="utf-8",
    )

    # Point the app UPCOMING_DIR to our temp directory
    app_module.UPCOMING_DIR = str(upc)
    app_module.app.config["TESTING"] = True

    # Ensure EnhancedPredictionService is considered available and monkeypatch the global instance
    monkeypatch.setattr(app_module, "ENHANCED_PREDICTION_SERVICE_AVAILABLE", True)
    monkeypatch.setattr(app_module, "enhanced_prediction_service", _FakeEnhancedService())

    with app_module.app.test_client() as c:
        yield c, csv_path.name


def test_enhanced_endpoint_returns_recommendations_by_default(client):
    c, race_filename = client

    # Call the enhanced endpoint with the race filename
    resp = c.post(
        "/api/predict_single_race_enhanced",
        data=json.dumps({"race_filename": race_filename}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()

    # The top-level response should include the enhanced prediction payload
    assert data.get("success") is True
    pred = data.get("prediction") or {}
    recs = pred.get("recommendations") or data.get("recommendations") or []
    assert isinstance(recs, list) and len(recs) > 0
    assert any("weak favorite" in r.lower() for r in recs)

    # The endpoint should annotate predictor_used where possible
    assert data.get("predictor_used") in ("EnhancedPredictionService", None)

