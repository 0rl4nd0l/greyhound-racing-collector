class V3TrainerFeatures:
    """V3 Trainer and ownership effects"""

    version = "3.0.0"

    def create_features(self, dog_stats: dict) -> dict:
        trainer = dog_stats.get("trainer_stats") or {}
        win_rate = trainer.get("win_rate")
        try:
            sr = max(0.0, min(1.0, float(win_rate))) if win_rate is not None else 0.0
        except Exception:
            sr = 0.0
        return {"v3_trainer_success_rate": float(sr)}

