#!/usr/bin/env python3
"""
Mark a specific model_id as best in the Model Registry and update symlinks.
Usage:
  PYTHONPATH=. python scripts/mark_registry_best.py --model-id <MODEL_ID>
"""
import argparse
import json
from pathlib import Path

from model_registry import ModelMetadata, get_model_registry
from scripts.db_guard import db_guard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    args = ap.parse_args()

    reg = get_model_registry()
    idx = getattr(reg, "model_index", {})
    from os import getenv as _ge

    db_path = (
        _ge("GREYHOUND_DB_PATH") or _ge("DATABASE_PATH") or "greyhound_racing_data.db"
    )
    mid = args.model_id
    if mid not in idx or not isinstance(idx[mid], dict):
        print(json.dumps({"success": False, "error": f"model_id '{mid}' not found"}))
        return

    # Reset all best flags and mark the requested one
    for k in idx:
        if isinstance(idx[k], dict):
            idx[k]["is_best"] = k == mid
            idx[k]["is_active"] = True

    # Persist and update symlinks
    try:
        with db_guard(db_path=db_path, label="mark_registry_best") as guard:
            guard.expect_table_growth("ml_model_registry", min_delta=0)
            # Access protected methods intentionally for maintenance
            reg._save_registry()  # type: ignore
            md = ModelMetadata(**idx[mid])
            reg._create_best_model_symlinks(md)  # type: ignore
        print(json.dumps({"success": True, "marked_best": mid}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))


if __name__ == "__main__":
    main()
