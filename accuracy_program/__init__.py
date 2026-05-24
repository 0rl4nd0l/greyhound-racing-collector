"""Leakage-safe prediction accuracy helpers.

These modules are intentionally additive: they measure, snapshot, and evaluate
predictions without changing model scoring or ranking behavior.
"""

from .bet_readiness import apply_bet_readiness_gates
from .evaluation import score_predictions, validate_feature_columns
from .odds_coverage import analyze_odds_coverage
from .snapshots import build_prediction_snapshot

__all__ = [
    "analyze_odds_coverage",
    "apply_bet_readiness_gates",
    "build_prediction_snapshot",
    "score_predictions",
    "validate_feature_columns",
]
