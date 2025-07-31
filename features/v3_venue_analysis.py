import numpy as np

class V3VenueAnalysisFeatures:
    """Feature group for venue-specific performance patterns."""

    def __init__(self):
        self.version = "v3.0"

    def create_features(self, data):
        """Creates venue-specific features from input data."""
        features = {}
        venue_stats = data.get('venue_stats', {})
        current_venue = data.get('current_venue', '')

        if not venue_stats or not current_venue:
            return self.get_default_features()

        # Venue-specific win rate and experience
        venue_data = venue_stats.get(current_venue, {})
        features['venue_win_rate'] = venue_data.get('win_rate', 0.125)
        features['venue_avg_position'] = venue_data.get('avg_position', 4.5)
        features['venue_experience'] = min(1.0, venue_data.get('races', 0) / 10)

        # Venue adaptability (performance across different venues)
        if len(venue_stats) > 1:
            positions = [v.get('avg_position', 4.5) for v in venue_stats.values()]
            features['venue_adaptability'] = 1 - (np.std(positions) / 4.0)  # Lower std = better adaptability
        else:
            features['venue_adaptability'] = 0.5

        return features

    def get_default_features(self):
        """Return default values for venue-specific features."""
        return {
            'venue_win_rate': 0.125,
            'venue_avg_position': 4.5,
            'venue_experience': 0,
            'venue_adaptability': 0.5
        }
