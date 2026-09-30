#!/usr/bin/env python3
"""Private finite result worker; inspect performs no acquisition or label reads."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from race_collection.development_examples import DevelopmentRejected
from race_collection.development_pilot_results import inspect_queue, run_cycle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('inspect','cycle'))
    parser.add_argument('--config',required=True)
    parser.add_argument('--config-sha256',required=True)
    args=parser.parse_args()
    try:
        result=(inspect_queue if args.action=='inspect' else run_cycle)(args.config,args.config_sha256)
    except DevelopmentRejected as error:
        print(json.dumps({'status':'REJECTED','reason':str(error)}))
        return 2
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
