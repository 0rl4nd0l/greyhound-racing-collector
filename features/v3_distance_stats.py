class V3DistanceStatsFeatures:
    """V3 Distance-based performance statistics and trends"""

    version = "3.0.0"

    def create_features(self, dog_stats: dict) -> dict:
        avg_time = _safe_float(dog_stats.get("avg_time"), 30.0)
        best_time = _safe_float(dog_stats.get("best_time"), avg_time)
        # Simple speed proxy: faster time => higher rating (normalize around 30s)
        try:
            speed_rating = max(0.0, min(100.0, (30.0 / max(1e-6, avg_time)) * 80))
        except Exception:
            speed_rating = 50.0
        return {
            "v3_distance_avg_time": float(avg_time),
            "v3_distance_speed_rating": float(speed_rating),
        }


def _safe_float(v, default=0.0):
    try:
        f = float(v)
        if f != f:  # NaN
            return default
        return f
    except Exception:
        return default

