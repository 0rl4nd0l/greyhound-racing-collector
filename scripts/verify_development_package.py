"""Export this committed source and demonstrate the default-off CLI offline."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


def run(output,python):
    root=Path(__file__).resolve().parents[1]
    output.mkdir(parents=True,exist_ok=False)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    tracked=subprocess.check_output(['git','ls-tree','-r','--name-only',commit],cwd=root,text=True).splitlines()
    names=[n for n in tracked if n.endswith('.py') or n.startswith(('configs/','config/')) and n.endswith('.json')
        or n in ('artifacts/frozen_models/market_form_residual_v1/model.json','artifacts/frozen_models/market_form_residual_v1/manifest.json','accuracy_program/repaired_non_tgr_schema.json')]
    raw=subprocess.check_output(['git','archive',commit,*names],cwd=root)
    # The collector's package builder verifies git HEAD itself. Preserve real
    # exact-commit Git metadata; a plain source tar is insufficient for that path.
    source=output/'source'
    subprocess.check_call(['git','worktree','add','--detach',str(source),commit],cwd=root)
    env=dict(os.environ,PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    proc=subprocess.run([str(python),'-B','-m','tests.fixtures.development_case','--output',str(output/'demonstration')],
        cwd=source,env=env,capture_output=True,text=True,timeout=180)
    (output/'execution.log').write_text(proc.stdout+proc.stderr)
    report={'status':'PASS' if proc.returncode==0 else 'FAIL','source_commit':commit,
        'archive_sha256':hashlib.sha256(raw).hexdigest(),'python':str(python),
        'python_sha256':hashlib.sha256(python.read_bytes()).hexdigest(),'synthetic':True,
        'installed':False,'activated':False,'output':str(output),
        'source_representation':'isolated_detached_exact_commit_worktree'}
    (output/'package_identity.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,sort_keys=True))
    if proc.returncode:raise RuntimeError('exported_development_demonstration_failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--python',type=Path,default=Path(sys.executable));a=p.parse_args();run(a.output,a.python)
