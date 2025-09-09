import pytest
from enhanced_prediction_service import EnhancedPredictionService


def test_weak_favorite_advisory_flagged():
    svc = EnhancedPredictionService()

    # Build predictions with a near tie top-2 and poor CSV history for the top pick
    preds = [
        {
            "dog_name": "TOP",
            "win_prob": 0.200,  # top
            "win_prob_norm": 0.200,
            "win_probability": 0.200,
            "final_score": 0.200,
            "csv_win_rate": 0.05,
            "csv_avg_finish_position": 5.1,
        },
        {
            "dog_name": "SECOND",
            "win_prob": 0.191,  # margin 0.009 < 0.05 -> weak favorite
            "win_prob_norm": 0.191,
            "win_probability": 0.191,
            "final_score": 0.191,
        },
        {
            "dog_name": "OTHER",
            "win_prob": 0.160,
            "win_prob_norm": 0.160,
            "win_probability": 0.160,
            "final_score": 0.160,
        },
    ]

    # Quality metrics computed from predictions (service helper)
    quality = svc._calculate_prediction_quality(preds)

    recs = svc._generate_prediction_recommendations(preds, quality)
    assert any(
        "weak favorite" in str(r).lower() for r in recs
    ), f"Expected weak-favorite advisory in recommendations, got: {recs}"

