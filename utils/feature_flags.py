#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Dict, Tuple

try:
    import yaml
except Exception:
    yaml = None

_FEATURE_FLAGS_PATH = Path("config/feature_flags.yaml")

DEFAULT_FLAGS = {
    "ENABLE_PLACE_ODDS_INTEGRATION": False,
    "AUTO_DISABLE_PLACE_THRESHOLD": 5,
    # Allow inference for future race dates (prediction-only override). Default is False.
    # Can also be controlled via ALLOW_FUTURE_RACE_DATES env var.
    "allow_future_race_dates": False,
}

def _bool_env(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")

def load_flags() -> Tuple[Dict[str, object], Dict[str, str]]:
    """Load feature flags from YAML then overlay environment variables.

    Returns: (flags, sources) where sources maps name->"env"|"yaml"|"default".
    """
    flags = DEFAULT_FLAGS.copy()
    sources = {k: "default" for k in flags}

    # Load YAML
    if _FEATURE_FLAGS_PATH.exists() and yaml is not None:
        try:
            content = yaml.safe_load(_FEATURE_FLAGS_PATH.read_text()) or {}
            for k, v in content.items():
                if k in flags:
                    flags[k] = v
                    sources[k] = "yaml"
        except Exception:
            pass

    # Overlay env
    if os.getenv("ENABLE_PLACE_ODDS_INTEGRATION") is not None:
        flags["ENABLE_PLACE_ODDS_INTEGRATION"] = _bool_env(
            os.getenv("ENABLE_PLACE_ODDS_INTEGRATION", "false")
        )
        sources["ENABLE_PLACE_ODDS_INTEGRATION"] = "env"

    if os.getenv("AUTO_DISABLE_PLACE_THRESHOLD") is not None:
        try:
            flags["AUTO_DISABLE_PLACE_THRESHOLD"] = int(
                os.getenv("AUTO_DISABLE_PLACE_THRESHOLD", str(flags["AUTO_DISABLE_PLACE_THRESHOLD"]))
            )
            sources["AUTO_DISABLE_PLACE_THRESHOLD"] = "env"
        except Exception:
            pass

    # Overlay ALLOW_FUTURE_RACE_DATES env var -> allow_future_race_dates flag
    if os.getenv("ALLOW_FUTURE_RACE_DATES") is not None:
        flags["allow_future_race_dates"] = _bool_env(
            os.getenv("ALLOW_FUTURE_RACE_DATES", "false")
        )
        sources["allow_future_race_dates"] = "env"

    return flags, sources

def persist_flags(updated: Dict[str, object]) -> None:
    """Persist provided flags to YAML atomically.
    Only writes keys present in DEFAULT_FLAGS.
    """
    if yaml is None:
        return
    current = {}
    if _FEATURE_FLAGS_PATH.exists():
        try:
            current = yaml.safe_load(_FEATURE_FLAGS_PATH.read_text()) or {}
        except Exception:
            current = {}
    for k in DEFAULT_FLAGS.keys():
        if k in updated:
            current[k] = updated[k]
    tmp = _FEATURE_FLAGS_PATH.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(current, sort_keys=True))
    os.replace(tmp, _FEATURE_FLAGS_PATH)

def get_flag(name: str, default=None):
    flags, _ = load_flags()
    return flags.get(name, default)
