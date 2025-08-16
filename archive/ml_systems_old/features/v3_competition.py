import numpy as np


class V3CompetitionFeatures:
    """Feature group for competition level and field analysis."""

    def __init__(self):
        self.version = "v3.0"

    def create_features(self, data):
        """Creates competition-based features from input data."""
        features = {}
        field_size = data.get("field_size", 8)
        grade = data.get("grade", "Grade 5")
        competitor_ratings = data.get("competitor_ratings", [])

        # Field size impact
        features["field_size"] = field_size
        features["field_size_normalized"] = (
            field_size / 12
        )  # Normalize to typical range

        # Grade level (higher grades = stronger competition)
        grade_mapping = {
            "Group 1": 10,
            "Group 2": 9,
            "Group 3": 8,
            "Grade 1": 7,
            "Grade 2": 6,
            "Grade 3": 5,
            "Grade 4": 4,
            "Grade 5": 3,
            "Maiden": 2,
            "Novice": 1,
        }
        features["grade_level"] = grade_mapping.get(grade, 3) / 10

        # Competition strength based on competitor ratings
        if competitor_ratings:
            features["avg_competitor_rating"] = np.mean(competitor_ratings)
            features["strongest_competitor"] = np.max(competitor_ratings)
            features["competition_variance"] = np.var(competitor_ratings)
        else:
            features["avg_competitor_rating"] = 0.5
            features["strongest_competitor"] = 0.5
            features["competition_variance"] = 0.1

        # Competitive advantage (how dog compares to field)
        dog_rating = data.get("dog_rating", 0.5)
        if competitor_ratings:
            features["rating_advantage"] = dog_rating - np.mean(competitor_ratings)
            features["rating_rank"] = sum(
                r < dog_rating for r in competitor_ratings
            ) / len(competitor_ratings)
        else:
            features["rating_advantage"] = 0
            features["rating_rank"] = 0.5

        return features

    def get_default_features(self):
        """Return default values for competition features."""
        return {
            "field_size": 8,
            "field_size_normalized": 0.67,
            "grade_level": 0.3,
            "avg_competitor_rating": 0.5,
            "strongest_competitor": 0.5,
            "competition_variance": 0.1,
            "rating_advantage": 0,
            "rating_rank": 0.5,
        }
