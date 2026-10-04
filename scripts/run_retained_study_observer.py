"""One root-scheduled, finite, outcome-blind retained-study observation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args(argv)
    os.umask(0o077)
    from race_collection.retained_study_observer import checked, observe
    try:
        cfg = checked({'path': str(args.config), 'sha256': args.config_sha256})
        result = observe(cfg, now=datetime.now(timezone.utc))
    except Exception as exc:
        # Arbitrary exception text can contain input data; emit only its class.
        print(json.dumps({'status': 'OBSERVER_HOLD', 'error_class': type(exc).__name__}))
        return 78
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get('complete_scan') or result['status'] == 'NOT_EFFECTIVE' else 75


if __name__ == '__main__':
    raise SystemExit(main())
