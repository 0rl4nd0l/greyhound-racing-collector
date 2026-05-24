from unittest.mock import patch

import pandas as pd

from drift_monitor import DriftMonitor

# Sample reference and current data for drift detection tests
reference_data = pd.DataFrame(
    {
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
    }
)

current_data = pd.DataFrame(
    {
        "feature1": [1.1, 2.1, 3.1],
        "feature2": [3.9, 5.1, 6.2],
    }
)


def test_drift_detection_with_missing_evidently():
    """Test drift detection with missing evidently library using monkeypatch."""
    with patch("drift_monitor.EVIDENTLY_AVAILABLE", False):
        monitor = DriftMonitor(reference_data)
        result = monitor.check_for_drift(current_data)

        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "drift_detected" in result
        assert "feature_drift" in result
        assert isinstance(result["feature_drift"], dict)
        assert "summary" in result
        assert isinstance(result["summary"], dict)

        # Example checks for drift specifics
        for feature, drift_info in result.get("feature_drift", {}).items():
            assert "psi" in drift_info  # PSI should be calculated

        # Ensure feature_drift exists with keys
        assert all(key in result["feature_drift"] for key in ["feature1", "feature2"])
