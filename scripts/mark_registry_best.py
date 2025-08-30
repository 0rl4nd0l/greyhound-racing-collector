#!/usr/bin/env python3
"""
Mark a specific model_id as best in the Model Registry and update symlinks.
Usage:
  PYTHONPATH=. python scripts/mark_registry_best.py --model-id <MODEL_ID>
"""
import argparse
import json
from pathlib import Path
from model_registry import get_model_registry, ModelMetadata

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-id', required=True)
    args = ap.parse_args()

    reg = get_model_registry()
    idx = getattr(reg, 'model_index', {})
    mid = args.model_id
    if mid not in idx or not isinstance(idx[mid], dict):
        print(json.dumps({"success": False, "error": f"model_id '{mid}' not found"}))
        return

    # Reset all best flags and mark the requested one
    for k in idx:
        if isinstance(idx[k], dict):
            idx[k]['is_best'] = (k == mid)
            idx[k]['is_active'] = True

    # Persist and update symlinks
    try:
        # Access protected methods intentionally for maintenance
        reg._save_registry()  # type: ignore
        md = ModelMetadata(**idx[mid])
        reg._create_best_model_symlinks(md)  # type: ignore
        print(json.dumps({"success": True, "marked_best": mid}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

if __name__ == '__main__':
    main()

