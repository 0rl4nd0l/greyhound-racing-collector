import numpy as np


class V3BoxPositionFeatures:
    """Feature group for box position and starting advantages."""

    def __init__(self):
        self.version = "v3.0"

    def create_features(self, data):
        """Creates box position-based features from input data."""
        features = {}
        box_number = data.get("box_number", 4)
        box_stats = data.get("box_stats", {})

        # Box position advantage (inside boxes have advantage)
        if box_number <= 3:
            features["box_advantage"] = 1  # Inside advantage
        elif box_number >= 6:
            features["box_advantage"] = -1  # Outside disadvantage
        else:
            features["box_advantage"] = 0  # Neutral

        # Historical performance from this box
        if box_stats and str(box_number) in box_stats:
            box_data = box_stats[str(box_number)]
            features["box_win_rate"] = box_data.get("win_rate", 0.125)
            features["box_avg_position"] = box_data.get("avg_position", 4.5)
        else:
            features["box_win_rate"] = 0.125
            features["box_avg_position"] = 4.5

        # Box position normalized (1-8 -> 0-1)
        features["box_position_normalized"] = (box_number - 1) / 7

        return features

    def get_default_features(self):
        """Return default values for box position features."""
        return {
            "box_advantage": 0,
            "box_win_rate": 0.125,
            "box_avg_position": 4.5,
            "box_position_normalized": 0.5,
        }
