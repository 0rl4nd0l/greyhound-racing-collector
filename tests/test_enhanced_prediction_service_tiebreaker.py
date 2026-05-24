import os
import copy
import pandas as pd
import pytest

from enhanced_prediction_service import EnhancedPredictionService


def make_pred(name, prob, sp=None, box=1):
    p = {
        "dog_name": name,
        "win_prob": float(prob),
        "win_prob_norm": float(prob),
        "win_probability": float(prob),
        "final_score": float(prob),
        "box_number": box,
    }
    if sp is not None:
        p["starting_price"] = sp
    return p


def test_sp_tiebreaker_applies_when_near_tie(monkeypatch):
    # Configure thresholds
    monkeypatch.setenv("TIEBREAKER_SP_ENABLED", "1")
    monkeypatch.setenv("TIEBREAKER_MARGIN_THRESH", "0.03")  # near tie if < 3%
    monkeypatch.setenv("TIEBREAKER_BUMP", "0.01")

    svc = EnhancedPredictionService()

    # Build a synthetic prediction_result with a near tie between top-2
    preds = [
        make_pred("DOG A", 0.200, sp=5.0, box=1),  # slightly shorter probability
        make_pred("DOG B", 0.195, sp=3.0, box=2),  # lower SP -> should be nudged above
        make_pred("DOG C", 0.160, sp=8.0, box=3),
    ]
    result = {"success": True, "predictions": preds, "race_id": "TEST-RACE-001"}

    # Race data DataFrame fallback with starting_price present
    df = pd.DataFrame({
        "dog_clean_name": ["DOG A", "DOG B", "DOG C"],
        "starting_price": [5.0, 3.0, 8.0],
    })

    # Act
    svc._apply_sp_tiebreaker(result, race_data=df, market_odds=None)

    # Assert ordering changed: DOG B should be first due to lower SP
    ordered = result.get("predictions")
    assert ordered[0]["dog_name"].upper() == "DOG B"
    assert ordered[0]["predicted_rank"] == 1
    assert result.get("tiebreaker_meta", {}).get("applied") is True


def test_sp_tiebreaker_no_apply_when_margin_clear(monkeypatch):
    monkeypatch.setenv("TIEBREAKER_SP_ENABLED", "1")
    monkeypatch.setenv("TIEBREAKER_MARGIN_THRESH", "0.01")  # strict
    svc = EnhancedPredictionService()

    preds = [
        make_pred("DOG A", 0.220, sp=5.0, box=1),
        make_pred("DOG B", 0.200, sp=3.0, box=2),
    ]
    result = {"success": True, "predictions": preds, "race_id": "TEST-RACE-002"}

    svc._apply_sp_tiebreaker(result, race_data=None, market_odds=None)
    ordered = result.get("predictions")
    # Margin 0.02 >= 0.01 -> no tie-breaker
    assert ordered[0]["dog_name"].upper() == "DOG A"
    assert result.get("tiebreaker_meta", {}).get("applied") in (None, False)

