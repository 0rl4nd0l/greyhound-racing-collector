import numpy as np


class V3DistanceStatsFeatures:
    """Feature group for distance-based performance statistics and trends."""

    def __init__(self):
        self.version = "v3.0"

    def create_features(self, data):
        """Creates distance-based features from input data."""
        features = {}
        distances = data.get("distances", [])

        if not distances:
            return self.get_default_features()

        # Distance averages and trends
        features["avg_distance"] = np.mean(distances)
        features["max_distance"] = np.max(distances)

        # Improve prediction power
        # Calculate distance variance
        features["distance_variance"] = np.var(distances)

        return features

    def get_default_features(self):
        """Return default values for distance-based features."""
        return {
            "avg_distance": 500,  # Default average distance
            "max_distance": 520,  # Default max distance
            "distance_variance": 0,
        }
