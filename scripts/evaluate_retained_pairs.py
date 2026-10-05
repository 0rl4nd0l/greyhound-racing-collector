"""Root-only private paired evaluation, default OFF; stdout contains no metrics."""
import argparse
import json
from race_collection.retained_paired_evaluation import run_paired_evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--authority')
    parser.add_argument('--authority-sha256')
    args = parser.parse_args(argv)
    if args.execute and not (args.authority and args.authority_sha256):
        parser.error('Exact authority path and SHA256 required.')
    try:
        ref = {'path': args.authority, 'sha256': args.authority_sha256} if args.execute else None
        print(json.dumps(run_paired_evaluation(ref, execute=args.execute), sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({'status': 'PAIRED_EVALUATION_FAILED', 'provider_requests': 0, 'result_requests': 0}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
