"""Explicit offline entrypoints; no runtime hook, acquisition or scheduler."""
from pathlib import Path
import argparse
import json

from race_collection.development_examples import seal, join_result, verify_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['seal','join','verify'])
    parser.add_argument('--access',type=Path,required=True)
    parser.add_argument('--access-sha256',required=True)
    parser.add_argument('--race-id',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--result-authority',type=Path)
    parser.add_argument('--result-authority-sha256')
    args=parser.parse_args()
    if args.action=='seal':
        value=seal(args.access,args.access_sha256,args.race_id,args.output)
    elif args.action=='verify':
        value=verify_package(args.access,args.access_sha256,args.race_id,args.output)
    else:
        if args.result_authority is None or args.result_authority_sha256 is None:
            parser.error('join requires the exact result authority and digest')
        value=join_result(args.access,args.access_sha256,args.race_id,args.output,
                          args.result_authority,args.result_authority_sha256)
    print(json.dumps(value,sort_keys=True))


if __name__=='__main__':
    main()
