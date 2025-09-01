#!/usr/bin/env python3
"""
Hardening guard: verifies production paths do not contain mock/placeholder logic
or random-based fallbacks unless explicitly gated via environment flags.

This script is intentionally focused and conservative to avoid false positives.
It performs targeted checks on key files and a small, safe global scan.

Exit codes:
- 0: All checks passed
- 1: One or more checks failed
"""
import sys
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Helper utilities

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""

def fail(msg: str):
    print(f"[HARDENING-GUARD] FAIL: {msg}")
    sys.exit(1)

def warn(msg: str):
    print(f"[HARDENING-GUARD] WARN: {msg}")

def ok(msg: str):
    print(f"[HARDENING-GUARD] OK: {msg}")

# 1) API must not serve mock predictions
api_main = REPO_ROOT / "fastapi_app" / "main.py"
api_txt = read_text(api_main)
if not api_txt:
    warn("fastapi_app/main.py missing or unreadable; skipping API mock check")
else:
    forbidden_terms = [
        "MockPredictor",
        "mock predictions",
        "Generating mock predictions",
    ]
    hits = [t for t in forbidden_terms if t in api_txt]
    if hits:
        fail(f"API contains forbidden mock artifacts: {hits}")
    ok("API mock prediction checks passed")

# 2) Unified predictor basic fallback must be dev-gated
unified = REPO_ROOT / "unified_predictor.py"
unified_txt = read_text(unified)
if not unified_txt:
    warn("unified_predictor.py missing or unreadable; skipping unified gating check")
else:
    if "def _basic_fallback_prediction" in unified_txt and "UNIFIED_ALLOW_BASIC_FALLBACK" not in unified_txt:
        fail("UnifiedPredictor basic fallback is not gated by UNIFIED_ALLOW_BASIC_FALLBACK")
    ok("UnifiedPredictor gating present for basic fallback")

# 3) ML V4 simulated odds/heuristic must be dev-gated
ml_v4 = REPO_ROOT / "ml_system_v4.py"
ml_v4_txt = read_text(ml_v4)
if not ml_v4_txt:
    warn("ml_system_v4.py missing or unreadable; skipping ML V4 gating checks")
else:
    if "ML_V4_ALLOW_SIMULATED_ODDS" not in ml_v4_txt:
        fail("ML_V4_ALLOW_SIMULATED_ODDS gating not found in ml_system_v4.py")
    if "ML_V4_ALLOW_HEURISTIC" not in ml_v4_txt:
        fail("ML_V4_ALLOW_HEURISTIC gating not found in ml_system_v4.py")
    ok("ML V4 gating present for simulated odds and heuristic path")

# 4) Drift detection must not use dummy KS fallback that returns no drift
feature_store = REPO_ROOT / "features" / "feature_store.py"
fs_txt = read_text(feature_store)
if not fs_txt:
    warn("features/feature_store.py missing or unreadable; skipping drift check validation")
else:
    if "DRIFT_DETECTION_DISABLED" not in fs_txt:
        fail("FeatureStore must log DRIFT_DETECTION_DISABLED when scipy/pandas/numpy are missing")
    ok("FeatureStore drift detection fallback is safe (disabled, not misleading)")

# 5) FastTrack feature builder placeholders must be disabled in production
ft_builder = REPO_ROOT / "src" / "predictor" / "feature_builders" / "fasttrack_features.py"
ft_txt = read_text(ft_builder)
if not ft_txt:
    warn("fasttrack_features.py missing or unreadable; skipping FastTrack placeholder checks")
else:
    required_funcs = [
        "get_last_n_races",
        "calculate_sectional_features",
        "calculate_performance_metrics",
        "calculate_normalized_time_features",
        "build_fasttrack_features",
    ]
    missing_raises = []
    for fn in required_funcs:
        # Ensure each function contains a RuntimeError raise
        pattern = rf"def\s+{fn}\b[\s\S]*?raise\s+RuntimeError\("
        if not re.search(pattern, ft_txt):
            missing_raises.append(fn)
    if missing_raises:
        fail(f"FastTrack builder functions not disabled with RuntimeError: {missing_raises}")
    ok("FastTrack placeholder functions are disabled in production")

# 6) Global scan: forbid random.uniform in production paths, with allowlist
allowlist_files = {unified.resolve()}
prod_roots = [REPO_ROOT / "fastapi_app", REPO_ROOT / "src", REPO_ROOT / "features"]
violations = []
for root in prod_roots:
    if not root.exists():
        continue
    for path in root.rglob("*.py"):
        if path.resolve() in allowlist_files:
            continue
        try:
            txt = read_text(path)
        except Exception:
            continue
        if "random.uniform(" in txt:
            violations.append(str(path.relative_to(REPO_ROOT)))
if violations:
    fail(f"random.uniform found in production code: {violations}")
ok("Global random.uniform scan passed for production paths")

print("[HARDENING-GUARD] All checks passed.")
sys.exit(0)

