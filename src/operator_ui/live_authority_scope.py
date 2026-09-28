"""Exact source sets for full and operational-only deployment authority."""
from .source_limits import LIVE_SOURCE_MAX_BYTES

FULL_AUTHORITY = "operator_ui_live_authority_v1"
OPERATIONAL_AUTHORITY = "operator_ui_operational_authority_v1"
RAW_KEYS = frozenset({
    "corpus_inventory_csv", "corpus_inventory_jsonl", "corpus_scorecard_csv",
    "corpus_scorecard_jsonl", "corpus_report_bytes", "corpus_summary",
    "corpus_final_status", "model_latest_config", "model_latest_schema",
    "model_latest_artifact", "model_latest_manifest", "model_baseline_config",
    "model_baseline_schema",
})


def source_keys(schema):
    """Reject unknown scopes; no optional or discovered source locators."""
    if schema not in (FULL_AUTHORITY, OPERATIONAL_AUTHORITY):
        raise ValueError("unknown live authority scope")
    json_keys = frozenset(LIVE_SOURCE_MAX_BYTES)
    if schema == OPERATIONAL_AUTHORITY:
        return (frozenset(k for k in json_keys if not k.startswith("corpus_")),
                frozenset(k for k in RAW_KEYS if not k.startswith("corpus_")))
    return json_keys, RAW_KEYS
