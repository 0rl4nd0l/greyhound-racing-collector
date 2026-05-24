class V3CompetitionFeatures:
    """V3 Competition level and field analysis"""

    version = "3.0.0"

    def create_features(self, dog_stats: dict) -> dict:
        # Simple competition strength proxy: more races => more exposure
        races = dog_stats.get("races_count") or 0
        try:
            strength = min(1.0, float(races) / 20.0)
        except Exception:
            strength = 0.5
        return {"v3_competition_strength": float(strength)}

