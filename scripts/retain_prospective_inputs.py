#!/usr/bin/env python3
"""One input-only retention attempt from a separately approved exact inventory."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from race_collection.prospective_input_retention import RetentionRejected, failure_code, retain_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', required=True, type=Path)
    parser.add_argument('--inventory-sha256', required=True)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--generate-features', action='store_true')
    parser.add_argument('--max-bundle-bytes', type=int)
    args = parser.parse_args()
    try:
        raw = args.inventory.read_bytes()
        if hashlib.sha256(raw).hexdigest() != args.inventory_sha256:
            raise RetentionRejected('INVENTORY_HASH_MISMATCH')
        plan = json.loads(raw)
        if set(plan) != {'schema_version', 'race_id', 'runner_names', 'observed_at',
                         'prediction_cutoff', 'jump_at', 'history_source', 'files'}:
            raise RetentionRejected('INVENTORY_SCHEMA_INVALID')
        if plan['schema_version'] != 'prospective_input_inventory_v1':
            raise RetentionRejected('INVENTORY_SCHEMA_INVALID')
        retain_inputs(
            destination=args.destination,
            race_id=plan['race_id'], runner_names=plan['runner_names'],
            observed_at=datetime.fromisoformat(plan['observed_at']),
            prediction_cutoff=datetime.fromisoformat(plan['prediction_cutoff']),
            jump_at=datetime.fromisoformat(plan['jump_at']),
            history_source=Path(plan['history_source']),
            files={role: (Path(spec['path']), spec['sha256']) for role, spec in plan['files'].items()},
            generate_features=args.generate_features,
            max_bundle_bytes=args.max_bundle_bytes,
        )
    except Exception as error:
        # No source values, SQL, exception details, historical rows or labels.
        reason = failure_code(error)
        print(json.dumps({'status': 'INPUT_RETENTION_FAILED', 'reason': reason, 'predictions_generated': False}))
        return 1
    print(json.dumps({'status': 'INPUTS_RETAINED_NOT_QUALIFIED', 'predictions_generated': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
