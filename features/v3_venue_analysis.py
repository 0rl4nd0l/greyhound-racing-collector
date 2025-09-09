class V3VenueAnalysisFeatures:
    """V3 Venue-specific performance patterns"""

    version = "3.0.0"

    def create_features(self, dog_stats: dict) -> dict:
        venue_stats = dog_stats.get("venue_stats") or {}
        # Heuristic: home advantage proxy — lower avg_position => better
        adv = 0.0
        try:
            for venue, stats in venue_stats.items():
                avg_pos = _safe_float(stats.get("avg_position"), 4.0)
                adv = max(adv, max(0.0, (4.0 - avg_pos) / 4.0))
        except Exception:
            adv = 0.0
        return {"v3_venue_home_advantage": float(adv)}


def _safe_float(v, default=0.0):
    try:
        f = float(v)
        if f != f:
            return default
        return f
    except Exception:
        return default
