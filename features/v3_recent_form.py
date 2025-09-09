class V3RecentFormFeatures:
    """V3 Recent performance and form analysis"""

    version = "3.0.0"

    def create_features(self, dog_stats: dict) -> dict:
        form = dog_stats.get("recent_form") or []
        form = [int(x) for x in form if _is_int_like(x)]
        win_rate = 0.0
        trend = 0.0
        if form:
            wins = sum(1 for x in form if x == 1)
            win_rate = wins / len(form)
            if len(form) >= 3:
                x = list(range(len(form)))
                try:
                    # Simple slope: decreasing position (better) -> positive trend
                    n = len(form)
                    x_mean = sum(x) / n
                    y_mean = sum(form) / n
                    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, form))
                    den = sum((xi - x_mean) ** 2 for xi in x) or 1.0
                    slope = num / den
                    trend = max(-1.0, min(1.0, -slope / 10.0))
                except Exception:
                    trend = 0.0
        return {
            "v3_recent_form_trend": float(trend),
            "v3_recent_win_rate": float(win_rate),
        }


def _is_int_like(v) -> bool:
    try:
        int(v)
        return True
    except Exception:
        return False

