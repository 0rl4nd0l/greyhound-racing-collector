import numpy as np


class V3TrainerFeatures:
    """Feature group for trainer and ownership effects."""

    def __init__(self):
        self.version = "v3.0"

    def create_features(self, data):
        """Creates trainer-based features from input data."""
        features = {}
        trainer = data.get("trainer", "")
        trainer_stats = data.get("trainer_stats", {})

        # Trainer performance
        if trainer and trainer in trainer_stats:
            trainer_data = trainer_stats[trainer]
            features["trainer_win_rate"] = trainer_data.get("win_rate", 0.125)
            features["trainer_avg_position"] = trainer_data.get("avg_position", 4.5)
            features["trainer_experience"] = min(
                1.0, trainer_data.get("total_races", 0) / 100
            )

            # Trainer recent form (last 30 days)
            recent_performance = trainer_data.get("recent_performance", {})
            features["trainer_recent_win_rate"] = recent_performance.get(
                "win_rate", 0.125
            )
            features["trainer_recent_strike_rate"] = recent_performance.get(
                "strike_rate", 0.3
            )
        else:
            features["trainer_win_rate"] = 0.125
            features["trainer_avg_position"] = 4.5
            features["trainer_experience"] = 0
            features["trainer_recent_win_rate"] = 0.125
            features["trainer_recent_strike_rate"] = 0.3

        # Trainer-dog combination
        dog_name = data.get("dog_name", "")
        if trainer and dog_name:
            combination_key = f"{trainer}_{dog_name}"
            combination_stats = data.get("trainer_dog_combinations", {}).get(
                combination_key, {}
            )
            features["trainer_dog_win_rate"] = combination_stats.get("win_rate", 0.125)
            features["trainer_dog_races"] = min(
                1.0, combination_stats.get("races", 0) / 20
            )
        else:
            features["trainer_dog_win_rate"] = 0.125
            features["trainer_dog_races"] = 0

        # Trainer effectiveness score
        features["trainer_effectiveness"] = np.mean(
            [
                features["trainer_win_rate"] * 8,  # Convert to 0-1 scale
                1
                - (features["trainer_avg_position"] - 1)
                / 7,  # Convert position to score
                features["trainer_experience"],
            ]
        )

        return features

    def get_default_features(self):
        """Return default values for trainer features."""
        return {
            "trainer_win_rate": 0.125,
            "trainer_avg_position": 4.5,
            "trainer_experience": 0,
            "trainer_recent_win_rate": 0.125,
            "trainer_recent_strike_rate": 0.3,
            "trainer_dog_win_rate": 0.125,
            "trainer_dog_races": 0,
            "trainer_effectiveness": 0.3,
        }
