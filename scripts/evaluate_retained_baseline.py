"""Root-launched local baseline, OFF unless exact evaluation authority is supplied."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from race_collection.retained_baseline_evaluation import freeze_membership, run_baseline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--freeze-membership', action='store_true')
    for name in ('membership', 'authority', 'protocol', 'journal', 'out'):
        parser.add_argument('--'+name, type=Path)
    for name in ('membership-sha256', 'authority-sha256', 'protocol-sha256', 'through-date'):
        parser.add_argument('--'+name)
    args = parser.parse_args(argv)
    if args.execute and args.freeze_membership:
        parser.error('Freeze membership before requesting evaluation authority.')
    if not args.execute and not args.freeze_membership:
        print(json.dumps({'status': 'DEFAULT_OFF', 'provider_requests': 0, 'result_reads': 0}))
        return 0
    required = ('protocol', 'protocol_sha256', 'journal', 'through_date', 'out') if args.freeze_membership else ('membership', 'membership_sha256', 'authority', 'authority_sha256')
    if any(getattr(args, key) is None for key in required):
        parser.error('Exact paths and hashes are required for the selected operation.')
    try:
        if args.freeze_membership:
            ref = freeze_membership({'path': str(args.protocol), 'sha256': args.protocol_sha256},
                args.journal, through_date=args.through_date, output=args.out, now=datetime.now(timezone.utc))
            print(json.dumps({'status': 'MEMBERSHIP_PROPOSED_NOT_AUTHORIZED', 'membership': ref}))
        else:
            result = run_baseline({'path': str(args.membership), 'sha256': args.membership_sha256},
                {'path': str(args.authority), 'sha256': args.authority_sha256}, execute=True)
            print(json.dumps(result))  # Counts and hashes only; never metric values.
        return 0
    except Exception:
        print(json.dumps({'status': 'BASELINE_OPERATION_FAILED', 'provider_requests': 0}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
