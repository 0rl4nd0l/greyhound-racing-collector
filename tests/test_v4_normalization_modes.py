import os

import numpy as np

from ml_system_v4 import MLSystemV4


def test_v4_normalization_mode_simple():
    os.environ["V4_NORMALIZATION_MODE"] = "simple"
    try:
        sys = MLSystemV4()
        arr = np.array([0.2, 0.3, 0.5])
        norm = sys._group_normalize_probabilities(arr)
        assert np.isclose(norm.sum(), 1.0, atol=1e-8)
        # In simple mode, ratios are preserved
        assert norm[2] > norm[1] > norm[0]
    finally:
        os.environ.pop("V4_NORMALIZATION_MODE", None)
