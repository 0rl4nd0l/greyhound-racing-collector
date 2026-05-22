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
    "ENABLE_AUTO_SCRAPE_ODDS": False,
    "AUTO_SCRAPE_ODDS_DOM_FALLBACK_MAX_CARDS": 3,
    "ENABLE_PLACE_ODDS_INTEGRATION": False,
    "AUTO_DISABLE_PLACE_THRESHOLD": 5,
    # Allow inference for future race dates (prediction-only override). Default is False.
    # Can also be controlled via ALLOW_FUTURE_RACE_DATES env var.
    "allow_future_race_dates": False,
}

def _bool_env(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")

def _bounded_int(v, default: int, min_value: int, max_value: int) -> int:
    try:
        value = int(v)
    except Exception:
        value = int(default)
    return max(min_value, min(max_value, value))

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

    if os.getenv("ENABLE_AUTO_SCRAPE_ODDS") is not None:
        flags["ENABLE_AUTO_SCRAPE_ODDS"] = _bool_env(
            os.getenv("ENABLE_AUTO_SCRAPE_ODDS", "false")
        )
        sources["ENABLE_AUTO_SCRAPE_ODDS"] = "env"

    if os.getenv("AUTO_SCRAPE_ODDS_DOM_FALLBACK_MAX_CARDS") is not None:
        flags["AUTO_SCRAPE_ODDS_DOM_FALLBACK_MAX_CARDS"] = _bounded_int(
            os.getenv("AUTO_SCRAPE_ODDS_DOM_FALLBACK_MAX_CARDS"),
            flags["AUTO_SCRAPE_ODDS_DOM_FALLBACK_MAX_CARDS"],
            0,
            10,
        )
        sources["AUTO_SCRAPE_ODDS_DOM_FALLBACK_MAX_CARDS"] = "env"

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

    flags["AUTO_SCRAPE_ODDS_DOM_FALLBACK_MAX_CARDS"] = _bounded_int(
        flags["AUTO_SCRAPE_ODDS_DOM_FALLBACK_MAX_CARDS"],
        DEFAULT_FLAGS["AUTO_SCRAPE_ODDS_DOM_FALLBACK_MAX_CARDS"],
        0,
        10,
    )
    for name in (
        "ENABLE_AUTO_SCRAPE_ODDS",
        "ENABLE_PLACE_ODDS_INTEGRATION",
        "allow_future_race_dates",
    ):
        flags[name] = _bool_env(flags[name])

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

def auto_scrape_odds_enabled() -> bool:
    return _bool_env(get_flag("ENABLE_AUTO_SCRAPE_ODDS", False))

def auto_scrape_dom_fallback_limit() -> int:
    return _bounded_int(
        get_flag(
            "AUTO_SCRAPE_ODDS_DOM_FALLBACK_MAX_CARDS",
            DEFAULT_FLAGS["AUTO_SCRAPE_ODDS_DOM_FALLBACK_MAX_CARDS"],
        ),
        DEFAULT_FLAGS["AUTO_SCRAPE_ODDS_DOM_FALLBACK_MAX_CARDS"],
        0,
        10,
    )
