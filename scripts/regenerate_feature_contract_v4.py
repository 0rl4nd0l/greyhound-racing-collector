#!/usr/bin/env python3
"""
Regenerate V4 feature contract file.
Writes docs/model_contracts/v4_feature_contract.json
"""
from __future__ import annotations
import json
from pathlib import Path
from ml_system_v4 import MLSystemV4


def main() -> int:
    sys = MLSystemV4()
    res = sys.regenerate_feature_contract()
    print(json.dumps(res, indent=2))
    return 0 if res.get("success") else 2

if __name__ == "__main__":
    raise SystemExit(main())

