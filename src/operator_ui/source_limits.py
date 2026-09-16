"""Finite repository-v1 JSON budgets shared by package generation and startup."""
from types import MappingProxyType

CONTROL_BYTES = 256 * 1024
REPORT_BYTES = 512 * 1024
# Odds reports retain pre-jump HTTP provenance, including base64 response bodies.
# The source-specific operating envelope is justified in
# docs/notes/r3_odds_report_budget_20260916.md; it is not an unbounded fallback.
ODDS_REPORT_BYTES = 2 * 1024 * 1024
LIVE_SOURCE_MAX_STRING_BYTES = MappingProxyType({
    "odds_report": 128 * 1024,
    "odds_refresh": 128 * 1024,
})

# Collector reports can exceed the small-control budget. Keep their limit
# bounded and identical when sealing, bootstrapping and reading evidence.
LIVE_SOURCE_MAX_BYTES = MappingProxyType({
    "full_state": REPORT_BYTES,
    "full_report": REPORT_BYTES,
    "odds_state": CONTROL_BYTES,
    "odds_report": ODDS_REPORT_BYTES,
    "odds_refresh": ODDS_REPORT_BYTES,
    "corpus_report": CONTROL_BYTES,
    "corpus_manifest": CONTROL_BYTES,
    "deployment_manifest": CONTROL_BYTES,
    "model_catalog": CONTROL_BYTES,
})
