"""Default-OFF entrypoint for retained pre-jump timing presence counts."""
import argparse
import json

from race_collection.retained_card_timing_coverage import CoverageRejected, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--manifest')
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--output')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'status': 'DEFAULT_OFF', 'provider_requests': 0, 'result_requests': 0}))
        return 0
    if not all((args.manifest, args.manifest_sha256, args.output)):
        parser.error('execute requires exact manifest, SHA256 and isolated output')
    try:
        print(json.dumps(run({'path': args.manifest, 'sha256': args.manifest_sha256}, args.output), sort_keys=True))
        return 0
    except CoverageRejected:
        print(json.dumps({'status': 'FAILED_NO_SUCCESSFUL_COVERAGE'}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
