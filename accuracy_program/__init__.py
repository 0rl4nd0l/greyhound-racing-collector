"""Leakage-safe prediction accuracy helpers.

These modules are intentionally additive: they measure, snapshot, and evaluate
predictions without changing model scoring or ranking behavior.
"""

__all__ = [
    "analyze_odds_coverage",
    "apply_bet_readiness_gates",
    "build_prediction_snapshot",
    "score_predictions",
    "validate_feature_columns",
]


def __getattr__(name):
    if name == "apply_bet_readiness_gates":
        from .bet_readiness import apply_bet_readiness_gates

        return apply_bet_readiness_gates
    if name in {"score_predictions", "validate_feature_columns"}:
        from .evaluation import score_predictions, validate_feature_columns

        return {
            "score_predictions": score_predictions,
            "validate_feature_columns": validate_feature_columns,
        }[name]
    if name == "analyze_odds_coverage":
        from .odds_coverage import analyze_odds_coverage

        return analyze_odds_coverage
    if name == "build_prediction_snapshot":
        from .snapshots import build_prediction_snapshot

        return build_prediction_snapshot
    raise AttributeError(name)
