#!/usr/bin/env python3
"""
Remove problematic symlinks in ./upcoming_races created during normalization.
Keeps originals untouched. Criteria for removal:
- entry is a symlink AND
  - name contains "__" OR "UNKNOWN" OR "__R_" OR date "1970-01-01"
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPCOMING = ROOT / "upcoming_races"


def main():
    removed = []
    if not UPCOMING.exists():
        print("{}")
        return
    for p in UPCOMING.iterdir():
        try:
            if p.is_symlink():
                name = p.name
                if (
                    ("__" in name)
                    or ("UNKNOWN" in name)
                    or ("__R_" in name)
                    or ("1970-01-01" in name)
                ):
                    p.unlink(missing_ok=True)
                    removed.append(name)
        except Exception:
            continue
    print({"removed": len(removed)})


if __name__ == "__main__":
    main()
