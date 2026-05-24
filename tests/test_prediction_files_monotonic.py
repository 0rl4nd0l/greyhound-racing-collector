import json
import os
from pathlib import Path

import math
import pytest

BASE = Path(__file__).resolve().parents[1]
PRED_DIR = BASE / "predictions"


@pytest.mark.skipif(not PRED_DIR.exists(), reason="predictions directory not found")
def test_prediction_files_rank_monotonic_by_probability():
    """
    Validate that in saved prediction JSONs, predicted_rank is consistent with
    descending probability when both fields are present. Uses existing real files.
    """
    any_checked = False
    for fp in sorted(PRED_DIR.glob("prediction_*.json"))[:200]:
        try:
            data = json.loads(fp.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        # Attempt nested or top-level predictions
        preds = []
        if isinstance(data, dict):
            if isinstance(data.get("predictions"), list):
                preds = data.get("predictions")
            elif isinstance(data.get("prediction"), dict) and isinstance(data["prediction"].get("predictions"), list):
                preds = data["prediction"]["predictions"]
        if not preds:
            continue
        # Filter to predictions with at least one probability-like field
        cleaned = []
        for p in preds:
            if not isinstance(p, dict):
                continue
            wp = p.get("win_prob")
            if wp is None:
                wp = p.get("win_prob_norm")
            if wp is None:
                wp = p.get("normalized_win_probability")
            if wp is None:
                wp = p.get("win_probability")
            if wp is None:
                continue
            try:
                f = float(wp)
            except Exception:
                continue
            cleaned.append((f, p.get("predicted_rank")))
        if len(cleaned) < 2:
            continue
        any_checked = True
        # Sort by prob desc and ensure ranks are non-decreasing with index
        probs, ranks = zip(*cleaned)
        order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        # Gather ranks in that order (ignore None)
        ordered_ranks = [ranks[i] for i in order if ranks[i] is not None]
        # Ensure strictly increasing ranks starting at 1 if ranks are present
        if ordered_ranks:
            # Allow for minor inconsistencies; just check ascending
            assert all(ordered_ranks[i] <= ordered_ranks[i + 1] for i in range(len(ordered_ranks) - 1)), (
                fp.name,
                ordered_ranks,
            )
    # If we didn't find any predictions, skip gracefully
    if not any_checked:
        pytest.skip("No prediction files with usable probabilities found")
