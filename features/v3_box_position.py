class V3BoxPositionFeatures:
    """V3 Box position and starting advantages"""

    version = "3.0.0"

    def create_features(self, dog_stats: dict) -> dict:
        box_positions = dog_stats.get("box_positions") or {}
        # Advantage proxy: favor inner boxes (1..4) if counts higher
        try:
            inner = sum(int(box_positions.get(i, 0)) for i in range(1, 5))
            outer = sum(int(box_positions.get(i, 0)) for i in range(5, 9))
            total = inner + outer or 1
            advantage = (inner - outer) / total
        except Exception:
            advantage = 0.0
        return {"v3_box_position_advantage": float(advantage)}

