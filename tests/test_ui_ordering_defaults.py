import os
from pathlib import Path

# These tests assert that the UI defaults are probability-first and support win_prob_norm.
# They are light-weight, text-level checks that avoid heavy predictor/runtime dependencies.

BASE = Path(__file__).resolve().parents[1]

JS_PRED_BUTTONS = BASE / "static" / "js" / "prediction-buttons.js"
JS_INTERACTIVE = BASE / "static" / "js" / "interactive-races.js"


def test_prediction_buttons_default_order_is_win_prob():
    src = JS_PRED_BUTTONS.read_text(encoding="utf-8", errors="ignore")
    # Default sort mode should be win_prob now
    assert "window.predOrderingMode" in src
    assert "'win_prob'" in src or '"win_prob"' in src


def test_prediction_score_extractor_supports_win_prob_norm():
    src = JS_PRED_BUTTONS.read_text(encoding="utf-8", errors="ignore")
    # Ensure win_prob_norm is included as a fallback in score extraction
    assert "win_prob_norm" in src


def test_interactive_races_uses_win_prob_norm_in_sorting():
    src = JS_INTERACTIVE.read_text(encoding="utf-8", errors="ignore")
    # Ensure the summary sort path recognizes win_prob_norm
    assert "win_prob_norm" in src

