"""Explicit, bounded offline retained-history qualification; default OFF."""
import argparse
import json
from pathlib import Path

from race_collection.retained_speed_history import audit_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'status': 'DEFAULT_OFF', 'provider_requests': 0}))
        return 0
    if not all((args.manifest, args.manifest_sha256, args.output)):
        parser.error('execution requires exact manifest path/hash and a new output')
    try:
        result = audit_manifest({'path': str(args.manifest), 'sha256': args.manifest_sha256})
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write('\n')
    except Exception:
        print(json.dumps({'status': 'AUDIT_REJECTED', 'provider_requests': 0}))
        return 2
    print(json.dumps({'status': result['status'], 'scope_counts': result['scope_counts']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
