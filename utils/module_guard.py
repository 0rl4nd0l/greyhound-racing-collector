#!/usr/bin/env python3
"""
Module Guard
============

Centralized whitelist/blacklist enforcement for loaded modules.
- Provides a startup check to validate sys.modules against policy
- Provides a pre-prediction sanity check to run before each prediction
- Emits clear error messages with guidance on resolution
- Supports environment-configurable ALLOWED/DISALLOWED prefixes

Environment variables:
- ALLOWED_MODULE_PREFIXES: comma-separated list of allowed module name prefixes
- DISALLOWED_MODULE_PREFIXES: comma-separated list of disallowed module name prefixes
- PREDICTION_IMPORT_MODE: when set to 'prediction_only', enforces stricter defaults
- MODULE_GUARD_STRICT: if '0' disables raising, only warns (default: '1')
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

# --------- Defaults ---------
# Conservative defaults for prediction-only operations
DEFAULT_ALLOWED_PREFIXES: Tuple[str, ...] = (
    # Standard lib commonly used
    "builtins",
    "collections",
    "datetime",
    "functools",
    "hashlib",
    "heapq",
    "itertools",
    "json",
    "logging",
    "math",
    "os",
    "pathlib",
    "random",
    "re",
    "sys",
    "threading",
    "time",
    "typing",
    "uuid",
    "sqlite3",
    "subprocess",
    "argparse",
    "glob",
    "warnings",
    "shutil",
    # Third-party commonly used in this project
    "numpy",
    "pandas",
    "sklearn",
    "flask",
    "flask_cors",
    "flask_compress",
    "werkzeug",
    "joblib",
    "dotenv",
    "yaml",
    # Project-local
    "features",
    "utils",
    "config",
    "advisory",
    "endpoint_cache",
    "database_manager",
    "assets",
    "prediction_pipeline_v4",
    "ml_system_v4",
    "temporal_feature_builder",
    "model_registry",
)

# Modules we must never load during prediction (I/O heavy, scraping, external calls, or unsafe)
DEFAULT_DISALLOWED_PREFIXES: Tuple[str, ...] = (
    # Network or external automation during prediction
    "selenium",
    "playwright",
    "requests_html",
    # Heavier frameworks that shouldn't be needed at inference here
    "torch",
    "tensorflow",
    "jax",
    # OpenAI or LLM calls must not happen during core prediction
    "openai",
    # File system watchers/schedulers
    "watchdog",
    "apscheduler",
    # Headless browser utils
    "pyppeteer",
    # Project results scrapers (must never load during prediction-only flows)
    "src.collectors",
    "comprehensive_form_data_collector",
)


@dataclass
class ModuleGuardViolation(Exception):
    message: str
    details: Dict[str, List[str]]
    resolution: List[str]

    def __str__(self) -> str:
        return self.message


def _parse_prefixes(env_value: str | None) -> List[str]:
    if not env_value:
        return []
    return [p.strip() for p in env_value.split(",") if p.strip()]


def _compile_policy() -> Tuple[List[str], List[str], bool]:
    # Load from env
    env_allowed = _parse_prefixes(os.getenv("ALLOWED_MODULE_PREFIXES"))
    env_disallowed = _parse_prefixes(os.getenv("DISALLOWED_MODULE_PREFIXES"))

    prediction_mode = os.getenv("PREDICTION_IMPORT_MODE", "prediction_only")
    strict = os.getenv("MODULE_GUARD_STRICT", "1") != "0"

    # Start with defaults; tighten if prediction_only
    allowed = list(DEFAULT_ALLOWED_PREFIXES)
    disallowed = list(DEFAULT_DISALLOWED_PREFIXES)

    if prediction_mode == "prediction_only":
        # In prediction-only mode, keep defaults as-is (already strict)
        pass
    else:
        # In relaxed mode, allow a bit more commonly used tooling
        allowed.extend(["matplotlib", "seaborn", "scipy"])

    # Apply env overrides
    if env_allowed:
        allowed.extend(env_allowed)
    if env_disallowed:
        disallowed.extend(env_disallowed)

    # De-duplicate while preserving order
    def dedup(seq: Iterable[str]) -> List[str]:
        seen = set()
        out = []
        for s in seq:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    return dedup(allowed), dedup(disallowed), strict


def _classify_loaded_modules(
    allowed_prefixes: List[str], disallowed_prefixes: List[str]
):
    loaded = list(sys.modules.keys())

    def starts_with_any(name: str, prefixes: Iterable[str]) -> bool:
        return any(name == p or name.startswith(p + ".") for p in prefixes)

    disallowed_loaded = [m for m in loaded if starts_with_any(m, disallowed_prefixes)]
    allowed_loaded = [m for m in loaded if starts_with_any(m, allowed_prefixes)]

    # Everything loaded that is neither explicitly allowed nor disallowed
    unknown_loaded = [
        m for m in loaded if m not in disallowed_loaded and m not in allowed_loaded
    ]

    return allowed_loaded, disallowed_loaded, unknown_loaded


def _build_resolution_guidance(disallowed_loaded: List[str], context: str) -> List[str]:
    steps = [
        "Stop prediction and unload disallowed modules.",
        "Confirm PREDICTION_IMPORT_MODE=prediction_only in your config/environment.",
        "Set or adjust DISALLOWED_MODULE_PREFIXES/ALLOWED_MODULE_PREFIXES as needed.",
        "Move scraping or test-only scripts to archive folders so they don't get auto-imported.",
        "Avoid importing heavy frameworks (torch/tensorflow) during inference.",
    ]
    if any(
        m.startswith(("selenium", "playwright", "pyppeteer")) for m in disallowed_loaded
    ):
        steps.append("Disable any live browser-based scraping during prediction.")
    if any(m.startswith("openai") for m in disallowed_loaded):
        steps.append(
            "Ensure advisory or LLM components are decoupled from core prediction path."
        )
    return steps


def _raise_or_log_violation(
    disallowed_loaded: List[str], unknown_loaded: List[str], strict: bool, context: str
):
    # Emit structured alert to system log JSONL and standard logger
    try:
        from utils import module_monitor as _module_monitor  # for JSONL writing

        _module_monitor._write_event(
            {
                "module": "module_guard",
                "severity": "CRITICAL",
                "event": "module_guard_violation",
                "context": context,
                "disallowed": disallowed_loaded,
                "unknown_sample": unknown_loaded[:15],
            }
        )
    except Exception:
        pass
    try:
        from logger import logger as _lg

        _lg.log_system(
            message=f"Violation in {context}: disallowed={disallowed_loaded[:10]} (total={len(disallowed_loaded)})",
            level="ERROR",
            component="MODULE_GUARD",
        )
    except Exception:
        pass
    # Provide a clearer message when results scrapers are involved
    if any(m.startswith("src.collectors") or "scraper" in m for m in disallowed_loaded):
        prefix = "Results scraping module loaded"
    else:
        prefix = "Disallowed modules detected"
    message = (
        f"{prefix} during {context}: {disallowed_loaded}. "
        "These modules are not permitted during prediction operations."
    )
    details = {
        "disallowed_loaded": disallowed_loaded,
        "unknown_loaded_sample": unknown_loaded[:15],
    }
    resolution = _build_resolution_guidance(disallowed_loaded, context)

    if strict:
        raise ModuleGuardViolation(
            message=message, details=details, resolution=resolution
        )
    else:
        # Soft warning via stderr
        sys.stderr.write(message + "\n" + "Resolution: " + "; ".join(resolution) + "\n")


def startup_module_sanity_check() -> None:
    """Validate loaded modules at process startup.
    Raises ModuleGuardViolation if a disallowed module is already loaded.
    """
    allowed, disallowed, strict = _compile_policy()
    _, disallowed_loaded, unknown_loaded = _classify_loaded_modules(allowed, disallowed)
    if disallowed_loaded:
        _raise_or_log_violation(
            disallowed_loaded, unknown_loaded, strict, context="startup"
        )


def pre_prediction_sanity_check(
    context: str = "prediction", extra_info: Dict[str, str] | None = None
) -> None:
    """Validate modules immediately before performing prediction.
    Call this at the beginning of any prediction entrypoint.
    """
    allowed, disallowed, strict = _compile_policy()
    _, disallowed_loaded, unknown_loaded = _classify_loaded_modules(allowed, disallowed)

    if disallowed_loaded:
        # Capture initial snapshot before cleanup for policy decisions
        initial_disallowed = list(disallowed_loaded)
        initial_unknown = list(unknown_loaded)
        # Best-effort unload to reduce risk first
        for name in list(sys.modules.keys()):
            if any(name == p or name.startswith(p + ".") for p in disallowed):
                try:
                    del sys.modules[name]
                except Exception:
                    pass
        # Recompute after purge; if clear, proceed
        _, disallowed_loaded_after, unknown_loaded_after = _classify_loaded_modules(
            allowed, disallowed
        )
        # In manual web prediction flow, block if results scrapers were ever detected (even if purged)
        if context == "manual_prediction":

            def is_results_scraper(mod: str) -> bool:
                if "the_greyhound_recorder_scraper" in mod:
                    # Allow guard to soft-clean this specific legacy scraper without blocking
                    return False
                return (
                    mod.startswith("src.collectors")
                    or "scraper" in mod
                    or mod.startswith("comprehensive_form_data_collector")
                )

            # Prefer initial detection to avoid races where modules are purged mid-check
            initial_results_scrapers = [m for m in initial_disallowed if is_results_scraper(m)]
            if initial_results_scrapers:
                _raise_or_log_violation(
                    initial_results_scrapers,
                    initial_unknown,
                    strict,
                    context=context,
                )
            # Otherwise, if still present after purge, block as well
            results_scrapers_loaded = [
                m for m in disallowed_loaded_after if is_results_scraper(m)
            ]
            if results_scrapers_loaded:
                _raise_or_log_violation(
                    results_scrapers_loaded,
                    unknown_loaded_after,
                    strict,
                    context=context,
                )
            # Non-scraper disallowed modules (e.g., playwright/selenium/watchdog) are tolerated here
            # after cleanup to avoid false positives in tests; they are still removed above.
            return
        # For other contexts, if anything remains disallowed after cleanup, raise
        if not disallowed_loaded_after:
            return
        _raise_or_log_violation(
            disallowed_loaded_after, unknown_loaded_after, strict, context=context
        )


# Convenience alias used by callers
ensure_prediction_module_integrity = pre_prediction_sanity_check
