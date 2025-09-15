#!/usr/bin/env python3
# Archived (dev-only): generate_mock_data.py
# This script has been moved under archive_unused_scripts/dev/ to avoid accidental use in production.
# It generates synthetic data and must never be used for live predictions.

from archive_unused_scripts.dev.generate_mock_data import main  # type: ignore

if __name__ == "__main__":
    import sys
    print("This script has been archived. Run: python archive_unused_scripts/dev/generate_mock_data.py", file=sys.stderr)
    sys.exit(2)
