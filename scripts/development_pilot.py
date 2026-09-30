#!/usr/bin/env python3
"""Installed, hash-pinned pilot command. No implicit activation or acquisition."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from race_collection.development_pilot import load_config, run_session, status


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--config-sha256',required=True)
    parser.add_argument('action',choices=['status','tick','run-session'])
    args=parser.parse_args()
    config=load_config(args.config,args.config_sha256)
    result=status(config) if args.action in {'status','tick'} else run_session(config)
    print(json.dumps(result,sort_keys=True))
    return 0 if result['status'] in {'NO_SLOT_DUE','SLOT_DUE','COMPLETE','SKIPPED_LATE_PREPARATION'} else 2


if __name__=='__main__':raise SystemExit(main())
