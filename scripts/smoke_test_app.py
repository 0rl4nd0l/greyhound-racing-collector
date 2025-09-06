#!/usr/bin/env python3
"""
Lightweight smoke test for the Flask app using test_client.
- Runs with safe env flags to avoid scraping or heavy initializations.
- Verifies key JSON endpoints return 200 and expected keys.
"""
from __future__ import annotations

import os
import sys

# Safe environment flags
os.environ.setdefault("TESTING", "1")
os.environ.setdefault("ENABLE_LIVE_SCRAPING", "0")
os.environ.setdefault("ENABLE_RESULTS_SCRAPERS", "0")
# Ensure consistent DB path resolution if custom envs exist
os.environ.setdefault("DATABASE_PATH", "greyhound_racing_data.db")

# Import app
try:
    sys.path.insert(0, os.getcwd())
    import app as app_module

    app = app_module.app
except Exception as e:
    print(f"[smoke] ERROR: Failed to import app: {e}")
    sys.exit(1)

results = []


def check(path: str, expect_json: bool = True, expect_keys: list[str] | None = None):
    try:
        with app.test_client() as c:
            rv = c.get(path)
            status = rv.status_code
            ok = status == 200
            payload = None
            if expect_json:
                try:
                    payload = rv.get_json(silent=True)
                    ok = ok and isinstance(payload, dict)
                except Exception:
                    ok = False
            if expect_keys and isinstance(payload, dict):
                for k in expect_keys:
                    ok = ok and (k in payload)
            results.append(
                {
                    "path": path,
                    "status": status,
                    "ok": ok,
                    "keys": expect_keys or [],
                    "payload_sample": payload,
                }
            )
            print(f"[smoke] GET {path} -> {status} | ok={ok}")
            return ok
    except Exception as e:
        results.append({"path": path, "status": None, "ok": False, "error": str(e)})
        print(f"[smoke] ERROR: {path}: {e}")
        return False


# Core API endpoints to probe
ok = True
ok &= check("/api/model_health", expect_json=True, expect_keys=["ready"])
ok &= check("/api/diagnostics/summary", expect_json=True, expect_keys=["success"])
ok &= check("/api/tgr/settings", expect_json=True, expect_keys=["success"])

# Exit code reflects overall success
if not ok:
    print("[smoke] One or more checks failed")
    sys.exit(2)

print("[smoke] All checks passed")
sys.exit(0)
