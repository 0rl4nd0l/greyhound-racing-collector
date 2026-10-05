#!/usr/bin/env python3
"""Default-off inventory of original commitments; never derive a pair."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.predictor.controlled_retained_inputs import BoundedReader, audit_membership, canonical


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-audit',action='store_true')
    parser.add_argument('--authority',type=Path)
    parser.add_argument('--authority-sha256')
    parser.add_argument('--membership',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if not args.execute_audit:
        print(json.dumps({'status':'CONTROLLED_INPUT_AUDIT_DISABLED'}))
        return 0
    try:
        if not all((args.authority,args.authority_sha256,args.membership,args.output)):
            raise ValueError('audit_arguments_required')
        authority=json.loads(BoundedReader().read(args.authority,args.authority_sha256))
        if (authority['schema_version']!='offline_research_continuation_task_v1'
                or 'read-only retained pre-race evidence qualification and opaque hashes' not in authority['authorized_now']
                or any(authority[key]!=0 for key in ('provider_requests','result_requests','protected_result_reads','performance_metrics'))
                or authority['preserve_original_artifacts'] is not True
                or args.membership!=Path(authority['membership_path'])/'membership.json'
                or not args.output.is_absolute() or args.output.resolve()!=args.output
                or not args.output.is_relative_to(Path(authority['outputs_root']))):
            raise ValueError('audit_authority_scope_changed')
        if args.output.exists():
            raise ValueError('audit_output_already_exists')
        result=audit_membership(args.membership)
        result['task_authority']={'path':str(args.authority),'sha256':args.authority_sha256}
        # The inventory is written only after every member was accounted for.
        with args.output.open('xb') as stream:
            os.fchmod(stream.fileno(),0o600)
            stream.write(canonical(result));stream.flush();os.fsync(stream.fileno())
        print(json.dumps({'status':'CONTROLLED_INPUT_COMMITMENT_AUDIT_COMPLETE',
            'denominator':result['denominator'],'categories':result['failure_categories'],
            'controlled_pairs_derived':0}))
        return 0
    except Exception as exc:
        # No input, forecast values, raw path-bearing exception or partial success.
        print(json.dumps({'status':'CONTROLLED_INPUT_AUDIT_FAILED','exception_type':type(exc).__name__}))
        return 2


if __name__=='__main__':
    raise SystemExit(main())
