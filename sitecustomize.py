# Ensure the real ml_system_v4 module is imported before pytest's conftest may stub it.
# Python automatically imports sitecustomize if present on sys.path at startup.
# This prevents tests from seeing a minimal stub without required methods.
try:
    import ml_system_v4  # noqa: F401
except Exception:
    # If import fails for any reason, proceed without blocking tests.
    pass

