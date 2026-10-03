"""Installed standing daily collector entrypoint."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--config-sha256')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--discover', nargs=3, metavar=('CONTRACT', 'INVENTORY', 'RECEIPT'))
    args = parser.parse_args()
    from race_collection.persistent_collector import discover, load_config, run
    if args.discover:
        discover(*(Path(p) for p in args.discover))
        return 0
    if not args.config or not args.config_sha256:
        parser.error('an exact installed configuration is required')
    if args.preflight:
        load_config(args.config, args.config_sha256)
        print('PERSISTENT_CONFIGURATION_VERIFIED_NO_NETWORK')
        return 0
    try:
        run(args.config, args.config_sha256)
    except Exception as exc:
        print('PERSISTENT_HOLD: '+type(exc).__name__, file=sys.stderr)
        return 78
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
