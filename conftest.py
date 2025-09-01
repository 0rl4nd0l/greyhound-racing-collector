# Root-level pytest configuration to avoid downstream stubs overriding real modules.
# Ensure environment is test-safe and import the real ml_system_v4 before tests/ conftest can stub it.
import os

# Mirror key env relaxations used by tests to keep imports light
os.environ.setdefault("DISABLE_STARTUP_GUARD", "1")
os.environ.setdefault("PREDICTION_IMPORT_MODE", "prediction_only")
os.environ.setdefault("MODULE_GUARD_STRICT", "1")

import importlib.util

# Force-load the real ml_system_v4 from project path into sys.modules before tests' conftest stubs it
import sys

proj_path = os.path.join(os.path.dirname(__file__), "ml_system_v4.py")
if os.path.exists(proj_path):
    try:
        spec = importlib.util.spec_from_file_location("ml_system_v4", proj_path)
        real_mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(real_mod)
        sys.modules["ml_system_v4"] = real_mod
    except Exception:
        # Fall back silently; tests may still stub if this fails
        pass
