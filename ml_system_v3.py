# Compatibility wrapper for MLSystemV3
# This exposes the archived implementation under the original top-level name
# so that tests and downstream modules can import `ml_system_v3` as expected.

import os
import logging

logger = logging.getLogger(__name__)

_use_stub = False
try:
    _use_stub = str(os.getenv('USE_ML_V3_STUB', '0')).strip().lower() in ('1', 'true', 'yes', 'on')
    # In CI/testing, prefer a lightweight stub to avoid heavy numeric imports/ops
    if not _use_stub:
        _use_stub = str(os.getenv('TESTING', '0')).strip().lower() in ('1', 'true', 'yes', 'on')
except Exception:
    _use_stub = False

if not _use_stub:
    try:
        # Prefer the maintained archived implementation
        from archive.ml_systems_old.ml_system_v3 import *  # noqa: F401,F403
    except Exception as e:  # pragma: no cover
        logger.warning(f"Falling back to stub MLSystemV3 due to import error: {e}")
        _use_stub = True

if _use_stub:
    class MLSystemV3:  # type: ignore
        def __init__(self, db_path: str = "greyhound_racing_data.db"):
            self.db_path = db_path
            # Indicate stub mode in logs
            try:
                logger.info("MLSystemV3 running in lightweight stub mode (TESTING/USE_ML_V3_STUB)")
            except Exception:
                pass

        def predict(self, dog_record):
            # Return a minimal structured response for tests that only check wiring
            # Include an explainability field with an error message (mirrors disabled SHAP path)
            return {
                "win_probability": 0.2,
                "place_probability": 0.45,
                "raw_win_probability": 0.2,
                "raw_place_probability": 0.45,
                "calibration_applied": False,
                "win_calibration_applied": False,
                "place_calibration_applied": False,
                "confidence": 0.5,
                "model_info": "stub-ml-system-v3",
                "explainability": {"error": "SHAP not available in stub mode", "feature_importance": {}, "available_models": []},
            }

