import numpy as np


class V3WeatherTrackFeatures:
    """Feature group for weather and track condition effects."""

    def __init__(self):
        self.version = "v3.0"

    def create_features(self, data):
        """Creates weather and track condition features from input data."""
        features = {}
        track_condition = data.get("track_condition", "Good")
        temperature = data.get("temperature", 20)
        humidity = data.get("humidity", 50)
        wind_speed = data.get("wind_speed", 5)
        track_performance = data.get("track_condition_performance", {})

        # Track condition suitability
        if track_condition in track_performance:
            condition_data = track_performance[track_condition]
            if isinstance(condition_data, dict) and condition_data.get("races", 0) >= 2:
                features["track_condition_win_rate"] = condition_data.get(
                    "win_rate", 0.125
                )
                features["track_condition_avg_position"] = condition_data.get(
                    "avg_position", 4.5
                )
                features["track_condition_experience"] = min(
                    1.0, condition_data.get("races", 0) / 10
                )
            else:
                features["track_condition_win_rate"] = 0.1  # Slight penalty for no data
                features["track_condition_avg_position"] = 4.7
                features["track_condition_experience"] = 0
        else:
            features["track_condition_win_rate"] = 0.08  # Penalty for unknown condition
            features["track_condition_avg_position"] = 5.0
            features["track_condition_experience"] = 0

        # Weather suitability
        # Optimal temperature range (15-25°C)
        if temperature:
            temp_score = 1 - abs(temperature - 20) / 20
            features["temperature_suitability"] = max(0, min(1, temp_score))
        else:
            features["temperature_suitability"] = 0.5

        # Humidity impact (moderate humidity preferred)
        if humidity:
            humidity_score = 1 - abs(humidity - 60) / 60
            features["humidity_suitability"] = max(0, min(1, humidity_score))
        else:
            features["humidity_suitability"] = 0.5

        # Wind impact (lower wind preferred)
        if wind_speed:
            features["wind_impact"] = max(
                0, 1 - wind_speed / 20
            )  # Wind over 20km/h has significant impact
        else:
            features["wind_impact"] = 0.8  # Assume light wind

        # Overall weather score
        features["weather_composite"] = np.mean(
            [
                features["temperature_suitability"],
                features["humidity_suitability"],
                features["wind_impact"],
            ]
        )

        return features

    def get_default_features(self):
        """Return default values for weather and track features."""
        return {
            "track_condition_win_rate": 0.125,
            "track_condition_avg_position": 4.5,
            "track_condition_experience": 0,
            "temperature_suitability": 0.5,
            "humidity_suitability": 0.5,
            "wind_impact": 0.8,
            "weather_composite": 0.6,
        }
