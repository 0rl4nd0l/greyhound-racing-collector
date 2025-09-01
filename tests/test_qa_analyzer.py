#!/usr/bin/env python3
"""
Test script for QA Analyzer - Demonstrates all functionality
==========================================================

This script tests all the QA Analyzer functions with various test cases
to show how it detects different types of issues.
"""

import json

from qa_analyzer import QAAnalyzer


def test_low_confidence_predictions():
    """Test low confidence detection."""
    print("🔍 Test 1: Low Confidence Predictions")

    qa_analyzer = QAAnalyzer(confidence_threshold=0.4)

    test_predictions = {
        "race_id": "test_low_confidence",
        "race_date": "2025-08-04",
        "predictions": [
            {"dog_name": "High Confidence Dog", "box_number": 1, "win_prob": 0.65},
            {"dog_name": "Medium Confidence Dog", "box_number": 2, "win_prob": 0.35},
            {"dog_name": "Low Confidence Dog 1", "box_number": 3, "win_prob": 0.15},
            {"dog_name": "Low Confidence Dog 2", "box_number": 4, "win_prob": 0.10},
            {"dog_name": "Very Low Confidence Dog", "box_number": 5, "win_prob": 0.05},
        ],
    }

    result = qa_analyzer.analyze_low_confidence_and_variance(test_predictions)
    print(f"  - Low confidence count: {result['low_confidence_count']}/5")
    print(f"  - Low variance flag: {result['low_variance_flag']}")
    print(f"  - Mean confidence: {result['statistics']['mean_confidence']:.3f}")
    print(f"  - Issues detected: {result['issues_detected']}")
    print()


def test_class_imbalance():
    """Test class imbalance detection."""
    print("🔍 Test 2: Class Imbalance Detection")

    qa_analyzer = QAAnalyzer()

    # Test extreme imbalance
    extreme_imbalance = {
        "race_id": "test_extreme_imbalance",
        "race_date": "2025-08-04",
        "predictions": [
            {"dog_name": "Dominant Dog", "box_number": 1, "win_prob": 0.85},
            {"dog_name": "Weak Dog 1", "box_number": 2, "win_prob": 0.05},
            {"dog_name": "Weak Dog 2", "box_number": 3, "win_prob": 0.05},
            {"dog_name": "Weak Dog 3", "box_number": 4, "win_prob": 0.05},
        ],
    }

    result1 = qa_analyzer.check_class_imbalance(extreme_imbalance)
    print("  - Extreme imbalance test:")
    print(f"    * Normalized entropy: {result1['normalized_entropy']:.3f}")
    print(
        f"    * Dominant probability: {result1['dominant_runner']['probability']:.3f}"
    )
    print(
        f"    * Extreme imbalance flag: {result1['imbalance_flags']['extreme_imbalance']}"
    )
    print(f"    * Issues detected: {result1['issues_detected']}")

    # Test balanced distribution
    balanced = {
        "race_id": "test_balanced",
        "race_date": "2025-08-04",
        "predictions": [
            {"dog_name": "Dog 1", "box_number": 1, "win_prob": 0.20},
            {"dog_name": "Dog 2", "box_number": 2, "win_prob": 0.20},
            {"dog_name": "Dog 3", "box_number": 3, "win_prob": 0.20},
            {"dog_name": "Dog 4", "box_number": 4, "win_prob": 0.20},
            {"dog_name": "Dog 5", "box_number": 5, "win_prob": 0.20},
        ],
    }

    result2 = qa_analyzer.check_class_imbalance(balanced)
    print("  - Balanced distribution test:")
    print(f"    * Normalized entropy: {result2['normalized_entropy']:.3f}")
    print(f"    * Issues detected: {result2['issues_detected']}")
    print()


def test_leakage_detection():
    """Test leakage and date drift detection."""
    print("🔍 Test 3: Leakage and Date Drift Detection")

    qa_analyzer = QAAnalyzer()

    # Test future date (too far)
    future_race = {
        "race_id": "test_future_race",
        "race_date": "2025-08-20",  # More than 7 days in future
        "race_time": "14:30",
        "extraction_time": "2025-08-04T12:00:00",
        "predictions": [{"dog_name": "Test Dog", "box_number": 1, "win_prob": 0.5}],
    }

    result1 = qa_analyzer.detect_leakage_and_date_drift(future_race)
    print("  - Future race test:")
    print(f"    * Future race flag: {result1['flags']['future_race_flag']}")
    print(f"    * Errors: {len(result1['errors'])}")
    print(f"    * Issues detected: {result1['issues_detected']}")

    # Test temporal inconsistency
    temporal_issue = {
        "race_id": "test_temporal_issue",
        "race_date": "2025-08-04",
        "race_time": "14:30",
        "extraction_time": "2025-08-04T18:00:00",  # After race time
        "predictions": [{"dog_name": "Test Dog", "box_number": 1, "win_prob": 0.5}],
    }

    result2 = qa_analyzer.detect_leakage_and_date_drift(temporal_issue)
    print("  - Temporal inconsistency test:")
    print(
        f"    * Temporal inconsistency flag: {result2['flags']['temporal_inconsistency_flag']}"
    )
    print(f"    * Errors: {len(result2['errors'])}")

    # Test potential leakage
    leakage_data = {
        "race_id": "test_leakage",
        "race_date": "2025-08-04",
        "race_time": "14:30",
        "extraction_time": "2025-08-04T12:00:00",
        "predictions": [
            {
                "dog_name": "Winner Dog",
                "box_number": 1,
                "win_prob": 0.5,
                "finish_position": 1,
            },  # Leakage!
            {
                "dog_name": "Loser Dog",
                "box_number": 2,
                "win_prob": 0.5,
                "actual_time": 31.2,
            },  # Leakage!
        ],
    }

    result3 = qa_analyzer.detect_leakage_and_date_drift(leakage_data)
    print("  - Potential leakage test:")
    print(f"    * Potential leakage flag: {result3['flags']['potential_leakage']}")
    print(f"    * Leakage indicators: {len(result3['leakage_indicators'])}")
    print(f"    * Issues detected: {result3['issues_detected']}")
    print()


def test_calibration_with_outcomes():
    """Test calibration drift with actual outcomes."""
    print("🔍 Test 4: Calibration Drift with Actual Outcomes")

    qa_analyzer = QAAnalyzer()

    test_predictions = {
        "race_id": "test_calibration",
        "race_date": "2025-08-04",
        "predictions": [
            {"dog_name": "Dog 1", "box_number": 1, "win_prob": 0.6},
            {"dog_name": "Dog 2", "box_number": 2, "win_prob": 0.3},
            {"dog_name": "Dog 3", "box_number": 3, "win_prob": 0.1},
        ],
    }

    # Simulate actual outcomes (Dog 1 won)
    actual_outcomes = [1, 0, 0]

    result = qa_analyzer.analyze_calibration_drift(test_predictions, actual_outcomes)
    print(f"  - Current Brier score: {result.get('current_brier_score', 'N/A')}")
    print(f"  - Calibration slope: {result.get('calibration_slope', 'N/A')}")
    print(
        f"  - Well calibrated: {result.get('calibration_quality', {}).get('well_calibrated', 'N/A')}"
    )
    print(f"  - Issues detected: {result['issues_detected']}")
    print()


def test_comprehensive_analysis():
    """Test comprehensive QA analysis."""
    print("🔍 Test 5: Comprehensive QA Analysis")

    qa_analyzer = QAAnalyzer(confidence_threshold=0.25)

    # Mix of issues
    mixed_issues = {
        "race_id": "test_comprehensive",
        "race_date": "2025-08-15",  # Future but within limit
        "race_time": "14:30",
        "extraction_time": "2025-08-04T12:00:00",
        "predictions": [
            {
                "dog_name": "Dominant Dog",
                "box_number": 1,
                "win_prob": 0.82,
            },  # High probability
            {
                "dog_name": "Weak Dog 1",
                "box_number": 2,
                "win_prob": 0.08,
            },  # Low confidence
            {
                "dog_name": "Weak Dog 2",
                "box_number": 3,
                "win_prob": 0.06,
            },  # Low confidence
            {
                "dog_name": "Weak Dog 3",
                "box_number": 4,
                "win_prob": 0.04,
            },  # Low confidence
        ],
    }

    result = qa_analyzer.comprehensive_qa_analysis(mixed_issues)
    print(f"  - Overall quality score: {result['overall_quality_score']}/100")
    print(f"  - Quality grade: {result['summary']['quality_grade']}")
    print(f"  - Total issues detected: {result['total_issues_detected']}")
    print(f"  - Issue categories: {result['issue_categories']}")
    print(f"  - Processing time: {result['processing_time_ms']:.1f}ms")

    print("\n📝 Recommendations:")
    for i, rec in enumerate(result["summary"]["recommendations"], 1):
        print(f"    {i}. {rec}")
    print()


def main():
    """Run all QA Analyzer tests."""
    print("🧪 QA Analyzer Test Suite")
    print("=" * 50)
    print()

    test_low_confidence_predictions()
    test_class_imbalance()
    test_leakage_detection()
    test_calibration_with_outcomes()
    test_comprehensive_analysis()

    print("✅ All tests completed!")
    print("📝 Detailed logs available in: logs/qa/qa.jsonl")

    # Show last few log entries
    print("\n📊 Recent QA Log Entries:")
    try:
        with open("logs/qa/qa.jsonl", "r") as f:
            lines = f.readlines()
            for line in lines[-3:]:
                entry = json.loads(line)
                print(
                    f"  [{entry['level']}] {entry['analysis_type']}: {entry['outcome']} "
                    f"(race: {entry['race_id']})"
                )
    except Exception as e:
        print(f"  Could not read log file: {e}")


if __name__ == "__main__":
    main()
