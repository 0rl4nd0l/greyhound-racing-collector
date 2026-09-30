"""One reproducible launcher for the entirely synthetic exported pilot proof."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allocation',type=Path,required=True,help='Read-only approved allocation/control metadata')
    args=parser.parse_args()
    from scripts.check_freshness_service import deny_network
    deny_network()
    root=Path(__file__).resolve().parents[2]
    output=args.output.resolve();output.mkdir(mode=0o700,parents=True,exist_ok=False)
    clock=output/'synthetic-clock.json'
    clock.write_text(json.dumps({'synthetic':True,'at':'2026-10-03T12:40:00+10:00','monotonic':time.monotonic()}))
    env={**os.environ,'TZ':'Australia/Melbourne',
        'PYTHONPATH':str(root/'tests/fixtures/development_pilot_transport')+os.pathsep+str(root),
        'SYNTHETIC_DEVELOPMENT_CLOCK':str(clock)}
    with (output/'run.log').open('w') as log:
        result=subprocess.run([sys.executable,'-B','-m','tests.fixtures.development_pilot_case',
            '--output',str(output/'case'),'--allocation',str(args.allocation.resolve())],
            cwd=root,env=env,stdout=log,stderr=log,timeout=240)
    summary={'synthetic':True,'status':'PASS' if result.returncode==0 else 'FAIL',
        'returncode':result.returncode,'log':str(output/'run.log'),
        'demonstration':str(output/'case/demonstration.json'),'real_provider_operations':0,'kernel_network':'denied'}
    (output/'package-proof.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,sort_keys=True))
    return result.returncode


if __name__=='__main__':raise SystemExit(main())
