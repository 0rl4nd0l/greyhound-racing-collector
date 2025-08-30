#!/usr/bin/env python3
"""
Module Monitor
==============

Utilities to snapshot, classify, and log loaded Python modules at:
- Process startup
- Per-request boundaries (Flask)
- Around prediction operations

Outputs structured JSON lines to logs/system_log.jsonl to match the
workflow logging policy. Also provides simple performance metric helpers.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Where to write structured system logs (JSONL)
SYSTEM_LOG_PATH = Path("logs") / "system_log.jsonl"
SYSTEM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# In-memory snapshots
_STARTUP_SNAPSHOT: Set[str] | None = None
_LAST_SNAPSHOT: Set[str] | None = None


def _write_event(payload: Dict) -> None:
    try:
        payload.setdefault("timestamp", datetime.now().isoformat())
        with SYSTEM_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        # Best-effort only
        pass


def _classify(mod_names: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Classify module names into (allowed, disallowed, unknown) via utils.module_guard."""
    try:
        from utils import module_guard  # Local import to avoid import-time side effects
        allowed, disallowed, _strict = module_guard._compile_policy()

        def starts_with_any(name: str, prefixes: List[str]) -> bool:
            return any(name == p or name.startswith(p + ".") for p in prefixes)

        disallowed_loaded = [m for m in mod_names if starts_with_any(m, disallowed)]
        allowed_loaded = [m for m in mod_names if starts_with_any(m, allowed)]
        unknown_loaded = [m for m in mod_names if m not in disallowed_loaded and m not in allowed_loaded]
        return allowed_loaded, disallowed_loaded, unknown_loaded
    except Exception:
        # If guard unavailable, treat all as unknown
        return [], [], mod_names


def take_snapshot() -> Set[str]:
    return set(sys.modules.keys())


def log_startup_modules(extra: Dict | None = None) -> None:
    global _STARTUP_SNAPSHOT, _LAST_SNAPSHOT
    _STARTUP_SNAPSHOT = take_snapshot()
    _LAST_SNAPSHOT = set(_STARTUP_SNAPSHOT)
    allowed, disallowed, unknown = _classify(sorted(_STARTUP_SNAPSHOT))
    _write_event({
        "module": "module_monitor",
        "severity": "INFO",
        "event": "module_snapshot",
        "context": "startup",
        "counts": {
            "total": len(_STARTUP_SNAPSHOT),
            "allowed": len(allowed),
            "disallowed": len(disallowed),
            "unknown": len(unknown),
        },
        "samples": {
            "disallowed": disallowed[:20],
            "unknown": unknown[:20],
        },
        **(extra or {}),
    })


def log_request_modules(request_path: str, method: str = "GET", context: str = "request") -> None:
    """Log newly loaded modules since the previous snapshot. Emits alert if disallowed are added
    and the request appears to be a prediction flow (heuristic by path).
    """
    global _LAST_SNAPSHOT
    current = take_snapshot()
    prev = _LAST_SNAPSHOT or set()
    added = sorted(list(current - prev))
    _LAST_SNAPSHOT = set(current)

    if not added:
        _write_event({
            "module": "module_monitor",
            "severity": "DEBUG",
            "event": "module_delta",
            "context": context,
            "request": {"path": request_path, "method": method},
            "added_count": 0,
        })
        return

    allowed, disallowed, unknown = _classify(added)
    payload = {
        "module": "module_monitor",
        "severity": "INFO",
        "event": "module_delta",
        "context": context,
        "request": {"path": request_path, "method": method},
        "added_count": len(added),
        "counts": {
            "allowed": len(allowed),
            "disallowed": len(disallowed),
            "unknown": len(unknown),
        },
        "samples": {
            "allowed": allowed[:15],
            "disallowed": disallowed[:15],
            "unknown": unknown[:15],
        },
    }
    _write_event(payload)

    # Heuristic: if this looks like a prediction operation, escalate on disallowed
    looks_like_prediction = any(k in request_path for k in ["predict", "api/predict", "predict_page"]) or context.startswith("prediction")
    if disallowed and looks_like_prediction:
        _write_event({
            "module": "module_monitor",
            "severity": "CRITICAL",
            "event": "unexpected_module_load",
            "context": context,
            "request": {"path": request_path, "method": method},
            "message": "Disallowed modules loaded during prediction",
            "disallowed": disallowed,
            "unknown_sample": unknown[:15],
        })


def time_block(metric_name: str, context: str = "request"):
    """Context manager to time a code block and log the metric."""
    class _Timer:
        def __enter__(self):
            self.t0 = time.time()
            return self
        def __exit__(self, exc_type, exc, tb):
            duration_ms = (time.time() - self.t0) * 1000.0
            _write_event({
                "module": "module_monitor",
                "severity": "INFO",
                "event": "perf_metric",
                "context": context,
                "metric": metric_name,
                "duration_ms": round(duration_ms, 2),
            })
    return _Timer()
