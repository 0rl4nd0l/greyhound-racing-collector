import numpy as np

class V3RecentFormFeatures:
    """Feature group for recent performance and form analysis."""

    def __init__(self):
        self.version = "v3.0"

    def create_features(self, data):
        """Creates recent form-based features from input data."""
        features = {}
        recent_form = data.get('recent_form', [])

        if not recent_form:
            return self.get_default_features()

        # Calculate recent form trend
        if len(recent_form) >= 3:
            x = np.arange(len(recent_form))
            slope = np.polyfit(x, recent_form, 1)[0]
            features['form_trend_slope'] = -slope  # Negative slope = improving
        else:
            features['form_trend_slope'] = 0

        # Weighted recent performance
        weights = np.exp(-0.1 * np.arange(len(recent_form)))  # Exponential decay
        features['weighted_recent_position'] = np.average(recent_form, weights=weights)

        return features

    def get_default_features(self):
        """Return default values for recent form-based features."""
        return {
            'form_trend_slope': 0,
            'weighted_recent_position': 4.0
        }
