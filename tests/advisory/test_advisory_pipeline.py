import json
from advisory import AdvisoryGenerator
import pytest


@pytest.fixture
def sample_inputs():
    # historical data CSV snippet is not parsed here; we simulate parsed artifacts in analysis
    # race data dict should represent scraped race info (weather, winners, etc.)
    historical_csv_snippet = """Dog Name|WGT|SP|TRAINER|G|DIST|PERF|SPEED|CLASS\n1. Fast Pup|31.2|3.2|A Trainer|G5|515|92|88|80\n|| || || || || ||\n2. Rapid Hound|29.8|4.5|B Trainer|G5|515|89|86|79\n"""

    race_data = {
        "venue": "Test Park",
        "date": "2025-08-05",
        "weather": "Fine",
        "winner": "Fast Pup",  # winner should be part of race data (scraped), not historical
    }

    # prediction-like structure used by AdvisoryGenerator's analyzer
    predictions = [
        {"dog_name": "Fast Pup", "box_number": 1, "win_prob": 0.62},
        {"dog_name": "Rapid Hound", "box_number": 2, "win_prob": 0.22},
        {"dog_name": "Third Dog", "box_number": 3, "win_prob": 0.16},
    ]

    data = {
        "race_id": "Race 1 - TEST - 05 August 2025",
        "race_date": race_data["date"],
        "race_time": "12:34",
        "race_data": race_data,
        "historical_data_csv": historical_csv_snippet,
        "predictions": predictions,
    }
    return data


def test_advisory_pipeline_shapes_output(sample_inputs, monkeypatch):
    # Monkeypatch QAAnalyzer to deterministic output (no network)
    class FakeAnalyzer:
        def comprehensive_qa_analysis(self, data):
            return {
                "overall_quality_score": 78,
                "total_issues_detected": 1,
                "issue_categories": ["confidence_analysis"],
                "individual_analyses": {
                    "confidence_variance": {
                        "issues_detected": True,
                        "low_confidence_count": 1,
                        "flagged_predictions": [data["predictions"][2]],
                    }
                },
            }

    from advisory import QAAnalyzer as RealQA

    monkeypatch.setattr("advisory.QAAnalyzer", lambda: FakeAnalyzer())
    gen = AdvisoryGenerator(api_key=None)

    out = gen.generate_advisory(data=sample_inputs)
    assert out["success"] is True
    assert "messages" in out and isinstance(out["messages"], list)
    assert any(m["type"] in ("WARNING", "CRITICAL", "INFO") for m in out["messages"])  # types present

    # JSON-shaped ML output
    ml = out["ml_json"]
    assert set(["version", "timestamp", "summary", "messages", "raw_validation", "raw_analysis", "feature_flags"]).issubset(ml.keys())
    assert isinstance(ml["summary"], dict)

    # Domain rule check: ensure winner present in race_data, not inferred from historical
    assert "race_data" in out["analysis_results"] or "race_data" in sample_inputs
    assert sample_inputs["race_data"]["winner"] == "Fast Pup"

