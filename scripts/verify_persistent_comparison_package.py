"""Export the exact commit, deny networking, run real prediction/result children."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from scripts.verify_frozen_comparison_package import run as baseline_package
from src.predictor.on_demand import canonical_bytes


def run(output,python):
    start=time.monotonic();usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    baseline_package(output,python,repetitions=1)
    source=output/'source'
    env=dict(os.environ,PYTHONPATH=str(source),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    tests=['tests/test_persistent_comparison.py','tests/test_persistent_schedule.py','tests/test_persistent_approval.py',
           'tests/test_prepare_comparison_deployment.py','tests/test_check_comparison_deployment.py','tests/test_check_comparison_health.py']
    code='from scripts.check_freshness_service import deny_network; deny_network(); import pytest; raise SystemExit(pytest.main('+repr(['--noconftest','--basetemp='+str(output/'process-cases'),'-o','addopts=','-p','no:cacheprovider','-q',*tests])+'))'
    proc=subprocess.run([str(python),'-B','-c',code],cwd=source,env=env,capture_output=True,text=True,timeout=240)
    (output/'persistent-tests.log').write_text(proc.stdout+proc.stderr)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    result={'status':'PASS' if proc.returncode==0 else 'FAIL','source_commit':json.loads((output/'package_identity.json').read_bytes())['source_commit'],
        'elapsed_seconds':time.monotonic()-start,'cpu_seconds':after.ru_utime+after.ru_stime-usage.ru_utime-usage.ru_stime,
        'max_child_rss_kib':after.ru_maxrss,'network':'kernel-denied in parent and inherited children',
        'tests':tests,'log_sha256':hashlib.sha256((output/'persistent-tests.log').read_bytes()).hexdigest(),
        'actual_future_predictions':0,'actual_official_results':0,'service_installation':False,
        'substitution':'synthetic HTTP bytes and clocks only for collector processes; systemd control mocked in scheduler boundary tests'}
    (output/'persistent-verification.json').write_bytes(canonical_bytes(result))
    print(json.dumps(result))
    if proc.returncode:raise RuntimeError('packaged_persistence_verification_failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--python',type=Path,default=Path(sys.executable));a=p.parse_args();run(a.output,a.python)
