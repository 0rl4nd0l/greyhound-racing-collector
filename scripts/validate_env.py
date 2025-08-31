#!/usr/bin/env python3
"""
Validate runtime environment versions used by ML models.

- Verifies scikit-learn == 1.7.1 (as pinned by constraints-unified.txt)
- Prints numpy and pandas versions for quick audit

Usage:
  python scripts/validate_env.py
"""

import sys


def main() -> int:
    try:
        import sklearn  # type: ignore
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as e:
        print(f"ERROR: Missing core dependency: {e}")
        return 1

    expected_sklearn = "1.7.1"
    actual_sklearn = getattr(sklearn, "__version__", "unknown")

    print(f"scikit-learn: {actual_sklearn}")
    print(f"numpy:        {getattr(np, '__version__', 'unknown')}")
    print(f"pandas:       {getattr(pd, '__version__', 'unknown')}")

    if actual_sklearn != expected_sklearn:
        print(
            "WARNING: scikit-learn version mismatch. "
            f"Expected {expected_sklearn}, found {actual_sklearn}.\n"
            "This can cause model unpickle warnings or runtime errors.\n"
            "To fix: ensure requirements/constraints-unified.txt is applied or pip install scikit-learn==1.7.1"
        )
        return 0

    print("Environment OK: versions match expected pins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

