"""Actual CLI demonstration with invented HTTP/browser/result evidence only."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def run(output):
    from scripts.check_freshness_service import deny_network
    deny_network()  # Kernel filter inherited by all exported collector children.
    import pytest
    from tests.test_operational_prediction_packaged import test_packaged_capture_retention_frozen_prediction
    from tests.fixtures.development_pipeline import prepare_access,prepare_result
    output.mkdir(parents=True,exist_ok=False)
    seed=output/'synthetic-collector';seed.mkdir()
    started=time.monotonic()
    with pytest.MonkeyPatch.context() as patch:
        test_packaged_capture_retention_frozen_prediction(seed,patch,False,'murray')
    capture_seconds=time.monotonic()-started
    rid,access,pin=prepare_access(seed,output/'controls')
    example=output/'example'
    def cli(action,*extra):
        start=time.monotonic()
        command=[sys.executable,'-B','-m','scripts.development_examples',action,'--access',str(access),
            '--access-sha256',pin,'--race-id',rid,'--output',str(example),*extra]
        proc=subprocess.run(command,capture_output=True,text=True,timeout=45)
        (output/(action+'.log')).write_text(proc.stdout+proc.stderr)
        if proc.returncode:raise RuntimeError('synthetic_'+action+'_failed')
        return {'elapsed_seconds':time.monotonic()-start,'result':json.loads(proc.stdout)}
    sealed=cli('seal')
    authority,authority_pin=prepare_result(access.parent,example,rid)
    joined=cli('join','--result-authority',str(authority),'--result-authority-sha256',authority_pin)
    verified=cli('verify')
    before=(example/'example.json').read_bytes()
    repeated=cli('join','--result-authority',str(authority),'--result-authority-sha256',authority_pin)
    if (example/'example.json').read_bytes()!=before:raise RuntimeError('restart_changed_example')
    stats={'status':'SYNTHETIC_COMPLETE_OFFLINE_ASSEMBLY','synthetic':True,'race_id':rid,
        'kernel_network':'denied_parent_and_inherited_children','provider_operations_real':0,
        'protected_target_decodes':0,'development_fits':0,'collector_fixture_seconds':capture_seconds,
        'seal':sealed,'join':joined,'verify':verified,'idempotent_restart':True,
        'example_bytes':sum(p.stat().st_size for p in example.rglob('*') if p.is_file()),
        'retained_bundle_bytes':sum(p.stat().st_size for p in seed.glob('campaign/operational-predictions/races/*/retention/*/bundle/**/*') if p.is_file()),
        'max_child_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        'interpreter':sys.executable,'elapsed_seconds':time.monotonic()-started}
    (output/'demonstration.json').write_text(json.dumps(stats,indent=2,sort_keys=True)+'\n')
    print(json.dumps(stats,sort_keys=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
