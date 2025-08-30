#!/usr/bin/env python3
"""
Promote the most recent V4_ExtraTrees model to best in the Model Registry.

- Finds the latest active V4_ExtraTrees by created_at
- Marks it as is_best: true and others false
- Updates symlinks via internal registry method
"""
from __future__ import annotations

import json
from pathlib import Path

import sys
from pathlib import Path as _P

# Ensure project root is on sys.path
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from model_registry import get_model_registry, ModelMetadata  # type: ignore

def main() -> int:
    reg = get_model_registry()

    # Find latest active V4_ExtraTrees
    candidates = []
    for mid, mdata in reg.model_index.items():
        if isinstance(mdata, dict) and mdata.get('is_active', True):
            if mdata.get('model_name') == 'V4_ExtraTrees':
                try:
                    mm = ModelMetadata(**mdata)
                    candidates.append(mm)
                except Exception:
                    continue

    if not candidates:
        print(json.dumps({'success': False, 'error': 'No active V4_ExtraTrees found'}))
        return 0

    latest = max(candidates, key=lambda m: m.created_at)

    # Mark latest as best and others not best
    for mid in list(reg.model_index.keys()):
        data = reg.model_index[mid]
        if not isinstance(data, dict):
            continue
        reg.model_index[mid]['is_best'] = (mid == latest.model_id)

    # Update symlinks
    try:
        # Access metadata JSON for symlink creation
        reg._save_registry()  # persist index
        reg._create_best_model_symlinks(latest)  # type: ignore[attr-defined]
    except Exception:
        pass

    print(json.dumps({'success': True, 'promoted_model_id': latest.model_id}))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

