"""Default-off, private historical first-sectional feature construction."""
import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--manifest')
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({'status': 'DEFAULT_OFF', 'provider_requests': 0, 'result_requests': 0}))
        return 0
    if not (args.manifest and args.manifest_sha256 and args.output):
        parser.error('execute requires the exact fixed manifest, SHA256 and exclusive output')
    from race_collection.retained_speed_features import run
    try:
        result = run({'path': args.manifest, 'sha256': args.manifest_sha256}, args.output)
    except Exception:
        print(json.dumps({'status': 'FAILED_NO_SUCCESSFUL_SPEED_FEATURES'}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
