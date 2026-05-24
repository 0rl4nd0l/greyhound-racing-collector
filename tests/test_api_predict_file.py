import app as app_module
from utils.race_lifecycle import RaceLifecycle


def test_predict_file_top_pick_order_by_win_prob(monkeypatch):
    # Always resolve to a dummy path
    monkeypatch.setattr(
        app_module, "resolve_race_file_path", lambda fn: "/tmp/dummy.csv", raising=True
    )
    # Avoid CSV meta enrichment side-effects
    monkeypatch.setattr(
        app_module, "enhance_prediction_with_csv_meta", lambda pr, p: pr, raising=True
    )

    def fake_run_prediction(path: str):
        # Out-of-order probabilities to ensure endpoint sorts by win_prob desc
        return {
            "success": True,
            "predictions": [
                {"dog_name": "Alpha", "win_prob": 0.10},
                {"dog_name": "Bravo", "win_prob": 0.30},
                {"dog_name": "Charlie", "win_prob": 0.20},
            ],
        }

    monkeypatch.setattr(
        app_module, "run_prediction_for_race_file", fake_run_prediction, raising=True
    )

    client = app_module.app.test_client()
    resp = client.post(
        "/api/predict_file", json={"race_file": "Race 4 - DARW - 2025-08-24.csv"}
    )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("success") is True
    assert data.get("resolved_path") == "/tmp/dummy.csv"

    computed = data.get("computed") or {}
    assert computed.get("top_pick", {}).get("name") == "Bravo"
    readiness = data.get("bet_readiness") or {}
    assert readiness.get("schema_version") == "bet_readiness_v1"
    assert data.get("abstain") is True
    assert "missing_live_odds" in data.get("abstain_reasons", [])
    assert (
        data["prediction_result"]["bet_readiness"]["status"]
        == "prediction_available_not_bet_qualified"
    )
    assert "prediction_snapshot" in data["prediction_result"]
    assert data.get("prediction_snapshot_status") == "created"

    top3 = computed.get("top3") or []
    names = [e.get("name") for e in top3]
    assert names == ["Bravo", "Charlie", "Alpha"]

    probs = [e.get("win_prob") for e in top3]
    assert probs == sorted(probs, reverse=True)


def test_predict_file_readiness_snapshot_preserves_prediction_values(monkeypatch):
    monkeypatch.setattr(
        app_module, "resolve_race_file_path", lambda fn: "/tmp/dummy.csv", raising=True
    )
    monkeypatch.setattr(
        app_module, "enhance_prediction_with_csv_meta", lambda pr, p: pr, raising=True
    )
    monkeypatch.setattr(
        app_module,
        "_classify_file_lifecycle",
        lambda *_args, **_kwargs: RaceLifecycle(
            status="upcoming_not_jumped",
            status_reason="jump_time_after_now_no_result",
        ),
        raising=True,
    )

    def fake_run_prediction(path: str):
        return {
            "success": True,
            "race_id": "RACE_1",
            "model_version": "model-v1",
            "predictions": [
                {
                    "dog_name": "Alpha",
                    "win_prob_norm": 0.6,
                    "predicted_rank": 1,
                    "odds_win": 2.0,
                    "ev_win": 0.2,
                },
                {
                    "dog_name": "Bravo",
                    "win_prob_norm": 0.4,
                    "predicted_rank": 2,
                    "ev_win": None,
                },
            ],
        }

    monkeypatch.setattr(
        app_module, "run_prediction_for_race_file", fake_run_prediction, raising=True
    )

    client = app_module.app.test_client()
    resp = client.post("/api/predict_file", json={"race_file": "Any.csv"})

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["lifecycle_status"] == "upcoming_not_jumped"
    result = data["prediction_result"]
    assert [p["predicted_rank"] for p in result["predictions"]] == [1, 2]
    assert [p["win_prob_norm"] for p in result["predictions"]] == [0.6, 0.4]
    assert result["predictions"][0]["ev_win"] == 0.2
    assert result["predictions"][1]["ev_win"] is None
    snapshot = result["prediction_snapshot"]
    assert snapshot["is_pre_jump_snapshot"] is True
    assert "finish_position" not in str(snapshot)


def test_predict_file_rejects_snapshot_result_field_leakage(monkeypatch):
    monkeypatch.setattr(
        app_module, "resolve_race_file_path", lambda fn: "/tmp/dummy.csv", raising=True
    )
    monkeypatch.setattr(
        app_module, "enhance_prediction_with_csv_meta", lambda pr, p: pr, raising=True
    )

    def fake_run_prediction(path: str):
        return {
            "success": True,
            "race_id": "RACE_1",
            "predictions": [
                {
                    "dog_name": "Alpha",
                    "win_prob_norm": 1.0,
                    "predicted_rank": 1,
                    "finish_position": 1,
                }
            ],
        }

    monkeypatch.setattr(
        app_module, "run_prediction_for_race_file", fake_run_prediction, raising=True
    )

    client = app_module.app.test_client()
    resp = client.post("/api/predict_file", json={"race_file": "Any.csv"})

    assert resp.status_code == 500
    data = resp.get_json()
    assert data["success"] is False
    assert data["error"] == "prediction snapshot rejected"
    assert "finish_position" in data["reason"]


def test_predict_file_percentage_conversion(monkeypatch):
    # Always resolve to a dummy path
    monkeypatch.setattr(
        app_module, "resolve_race_file_path", lambda fn: "/tmp/dummy.csv", raising=True
    )
    # Avoid CSV meta enrichment side-effects
    monkeypatch.setattr(
        app_module, "enhance_prediction_with_csv_meta", lambda pr, p: pr, raising=True
    )

    def fake_run_prediction_pct(path: str):
        # Provide win probabilities as percentages; endpoint should normalize to 0-1
        return {
            "success": True,
            "predictions": [
                {"dog_name": "DogA", "win_probability": 55.0},
                {"dog_name": "DogB", "win_probability": 30.0},
                {"dog_name": "DogC", "win_probability": 15.0},
            ],
        }

    monkeypatch.setattr(
        app_module,
        "run_prediction_for_race_file",
        fake_run_prediction_pct,
        raising=True,
    )

    client = app_module.app.test_client()
    resp = client.post("/api/predict_file", json={"race_file": "Any.csv"})

    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("success") is True

    comp = data.get("computed") or {}
    top3 = comp.get("top3") or []
    names = [e.get("name") for e in top3]
    assert names == ["DogA", "DogB", "DogC"]

    probs = [e.get("win_prob") for e in top3]
    assert all(0.0 <= float(p) <= 1.0 for p in probs)
    assert abs(float(probs[0]) - 0.55) < 1e-6


def test_predict_file_not_found_404(monkeypatch):
    # Simulate unresolved path
    monkeypatch.setattr(
        app_module, "resolve_race_file_path", lambda fn: None, raising=True
    )
    client = app_module.app.test_client()
    resp = client.post("/api/predict_file", json={"race_file": "no_such_file.csv"})
    assert resp.status_code == 404
    data = resp.get_json()
    assert data.get("success") is False
    assert "not found" in (data.get("error") or "").lower()


def test_predict_file_empty_predictions_degrades_gracefully(monkeypatch):
    # Resolve to a dummy path and return empty predictions
    monkeypatch.setattr(
        app_module, "resolve_race_file_path", lambda fn: "/tmp/dummy.csv", raising=True
    )
    monkeypatch.setattr(
        app_module, "enhance_prediction_with_csv_meta", lambda pr, p: pr, raising=True
    )

    def fake_run_prediction_empty(path: str):
        return {"success": True, "predictions": []}  # no runners

    monkeypatch.setattr(
        app_module,
        "run_prediction_for_race_file",
        fake_run_prediction_empty,
        raising=True,
    )

    client = app_module.app.test_client()
    resp = client.post("/api/predict_file", json={"race_file": "empty.csv"})

    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("success") is True
    # No computed top picks when predictions are empty
    assert ("computed" not in data) or (not data.get("computed"))
