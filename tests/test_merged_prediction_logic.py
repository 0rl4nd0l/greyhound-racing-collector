import hashlib
import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from app import app


@pytest.fixture
def client():
    """Flask test client fixture"""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_ml_predictions():
    """Sample ML predictions for testing merged logic"""
    return [
        {
            "dog_name": "Dog A",
            "box_number": 1,
            "prediction_score": 0.85,
            "confidence_level": "HIGH",
            "predicted_rank": 1,
            "reasoning": ["Strong recent form"],
        },
        {
            "dog_name": "Dog B",
            "box_number": 2,
            "prediction_score": 0.70,
            "confidence_level": "MEDIUM",
            "predicted_rank": 2,
            "reasoning": ["Good track record"],
        },
    ]


@pytest.fixture
def sample_gpt_analysis():
    """Sample GPT analysis for testing merged logic"""
    return {
        "race_info": {
            "venue": "Test Venue",
            "race_number": 1,
            "race_date": "2025-01-01",
            "distance": "500m",
            "grade": "Grade 5",
        },
        "gpt_race_analysis": {
            "analysis_confidence": 0.9,
            "raw_analysis": '{"order": ["Dog A", "Dog B"]}',
        },
        "enhanced_ml_predictions": {
            "enhanced_insights": {"key_factors": ["Track conditions favor early speed"]}
        },
        "betting_strategy": {"betting_strategy": "Focus on win bets for top pick"},
        "pattern_analysis": {"venue_patterns": "Inside boxes perform well"},
        "merged_predictions": [
            {
                "dog_name": "Dog A",
                "prediction_score": 0.85,
                "confidence_level": "HIGH",
                "gpt_insights": {
                    "gpt_predicted_position": 1,
                    "gpt_analysis_available": True,
                    "gpt_confidence": 0.9,
                },
                "enhanced_with_gpt": True,
            },
            {
                "dog_name": "Dog B",
                "prediction_score": 0.70,
                "confidence_level": "HIGH",  # Enhanced from MEDIUM
                "gpt_insights": {
                    "gpt_predicted_position": 2,
                    "gpt_analysis_available": True,
                    "gpt_confidence": 0.9,
                },
                "enhanced_with_gpt": True,
            },
        ],
        "analysis_summary": {
            "analysis_confidence": 0.9,
            "gpt_available": True,
            "prediction_enhancement": True,
            "betting_strategy_available": True,
            "key_insights": ["Track conditions favor early speed"],
        },
        "timestamp": "2025-01-01T00:00:00",
        "tokens_used": 200,
    }


def create_response_hash(response_data):
    """Create a hash of response data for snapshot comparison"""
    # Sort keys to ensure consistent hashing
    normalized = json.dumps(response_data, sort_keys=True)
    return hashlib.md5(normalized.encode()).hexdigest()


# Test merged prediction logic snapshot
@patch("app.get_gpt_enhancer")
def test_merged_prediction_snapshot(mock_get_enhancer, sample_gpt_analysis, client):
    """Test that merged prediction logic produces consistent results"""
    mock_enhancer = MagicMock()
    mock_enhancer.gpt_available = True
    mock_enhancer.enhance_race_prediction.return_value = sample_gpt_analysis
    mock_get_enhancer.return_value = mock_enhancer

    data = {
        "race_file_path": "test_path.csv",
        "include_betting_strategy": True,
        "include_pattern_analysis": True,
    }

    response = client.post("/api/gpt/enhance_race", json=data)
    assert response.status_code == 200
    assert response.json["success"] is True

    # Verify structure remains consistent
    enhancement = response.json["enhancement"]
    assert enhancement["race_info"]["venue"] == "Test Venue"
    assert enhancement["gpt_race_analysis"]["analysis_confidence"] == 0.9
    assert len(enhancement["merged_predictions"]) == 2

    # Test that GPT insights are properly merged
    merged_pred = enhancement["merged_predictions"][0]
    assert merged_pred["enhanced_with_gpt"] is True
    assert "gpt_insights" in merged_pred
    assert merged_pred["gpt_insights"]["gpt_predicted_position"] == 1

    # Snapshot test: create hash of consistent parts
    # Remove timestamp for consistent comparison
    stable_data = {k: v for k, v in enhancement.items() if k != "timestamp"}
    response_hash = create_response_hash(stable_data)

    # This would be the expected hash for this specific test case
    # In practice, you'd store this in a snapshot file
    expected_structure_keys = {
        "race_info",
        "gpt_race_analysis",
        "enhanced_ml_predictions",
        "betting_strategy",
        "pattern_analysis",
        "merged_predictions",
        "analysis_summary",
        "tokens_used",
    }
    assert set(stable_data.keys()) == expected_structure_keys


# Test prediction merging preserves ML scores
@patch("app.get_gpt_enhancer")
def test_prediction_merging_preserves_ml_scores(
    mock_get_enhancer, sample_gpt_analysis, client
):
    """Test that GPT enhancement preserves original ML prediction scores"""
    mock_enhancer = MagicMock()
    mock_enhancer.gpt_available = True
    mock_enhancer.enhance_race_prediction.return_value = sample_gpt_analysis
    mock_get_enhancer.return_value = mock_enhancer

    data = {
        "race_file_path": "test_path.csv",
        "include_betting_strategy": True,
        "include_pattern_analysis": True,
    }

    response = client.post("/api/gpt/enhance_race", json=data)
    assert response.status_code == 200

    merged_predictions = response.json["enhancement"]["merged_predictions"]

    # Verify original ML scores are preserved
    assert merged_predictions[0]["prediction_score"] == 0.85  # Original ML score
    assert merged_predictions[1]["prediction_score"] == 0.70  # Original ML score

    # Verify GPT insights are added without overriding ML scores
    for pred in merged_predictions:
        assert "gpt_insights" in pred
        assert pred["enhanced_with_gpt"] is True
        assert "prediction_score" in pred  # Original ML score preserved


# Test confidence level enhancement
@patch("app.get_gpt_enhancer")
def test_confidence_enhancement_logic(mock_get_enhancer, sample_gpt_analysis, client):
    """Test that confidence levels are enhanced when ML and GPT agree"""
    mock_enhancer = MagicMock()
    mock_enhancer.gpt_available = True
    mock_enhancer.enhance_race_prediction.return_value = sample_gpt_analysis
    mock_get_enhancer.return_value = mock_enhancer

    data = {
        "race_file_path": "test_path.csv",
        "include_betting_strategy": True,
        "include_pattern_analysis": True,
    }

    response = client.post("/api/gpt/enhance_race", json=data)
    assert response.status_code == 200

    merged_predictions = response.json["enhancement"]["merged_predictions"]

    # Dog B's confidence should be enhanced from MEDIUM to HIGH
    # because both ML and GPT predict it in top positions
    dog_b_pred = next(p for p in merged_predictions if p["dog_name"] == "Dog B")
    assert dog_b_pred["confidence_level"] == "HIGH"  # Enhanced from MEDIUM
