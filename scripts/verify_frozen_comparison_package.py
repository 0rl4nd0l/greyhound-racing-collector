"""Export committed sources and execute real retained prediction subprocesses.

Only fabricated source evidence is used. Each child installs an inherited kernel
network deny filter before retention/publication/worker execution. No services.
"""
import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
from datetime import datetime,timezone

from src.predictor.on_demand import canonical_bytes

ROOT=Path(__file__).resolve().parents[1]


def run(output,python,repetitions=3):
    if not 1<=repetitions<=3: raise ValueError('bounded_repetitions_required')
    output.mkdir(parents=True,exist_ok=False)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    tracked=subprocess.check_output(['git','ls-tree','-r','--name-only',commit],cwd=ROOT,text=True).splitlines()
    names=[name for name in tracked if (name.endswith('.py') and not name.startswith('docs/')) or
        name.startswith('artifacts/research_comparison/frozen_20260924/') or
        name in ('artifacts/frozen_models/market_form_residual_v1/model.json','artifacts/frozen_models/market_form_residual_v1/manifest.json','accuracy_program/repaired_non_tgr_schema.json') or
        (name.startswith(('configs/','config/')) and name.endswith('.json'))]
    raw=subprocess.check_output(['git','archive',commit,*names],cwd=ROOT)
    source=output/'source';source.mkdir()
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive: archive.extractall(source,filter='data')
    identities={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names}
    env=dict(os.environ,PYTHONPATH=str(source),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    probe="import json,platform,importlib.metadata as m;print(json.dumps({'python':platform.python_version(),'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))"
    environment=json.loads(subprocess.check_output([str(python),'-B','-c',probe],env=env))
    (output/'package_identity.json').write_bytes(canonical_bytes({'source_commit':commit,'files':identities,
        'archive_sha256':hashlib.sha256(raw).hexdigest(),'python':str(python),'python_sha256':hashlib.sha256(python.read_bytes()).hexdigest(),
        'environment':environment,'created_at':datetime.now(timezone.utc).isoformat()}))
    records=[]
    for i in range(repetitions):
        trial=output/f'trial-{i+1}'
        proc=subprocess.run([str(python),'-B','-m','tests.fixtures.frozen_comparison_case','--output',str(trial),*(['--comparison-first'] if i%2 else [])],
            cwd=source,env=env,capture_output=True,text=True,timeout=90)
        (output/f'trial-{i+1}.log').write_text(proc.stdout+proc.stderr)
        if proc.returncode: raise RuntimeError(f'packaged_trial_{i+1}_failed')
        baseline=json.loads((trial/'baseline-execution.json').read_bytes());compared=json.loads((trial/'predictions-execution.json').read_bytes())
        if baseline['phase']!=compared['phase'] or compared['phase']!='PREDICTION_READY': raise RuntimeError('packaged_prediction_failed')
        admission=next((trial/'comparison-programme').glob('*/attempts/*/admission.json'))
        verify=subprocess.run([str(python),'-B','-c',
            'from scripts.check_freshness_service import deny_network; deny_network(); import runpy; runpy.run_module("scripts.verify_frozen_comparison",run_name="__main__")',
            '--output-root',str(trial/'predictions'),'--admission',str(admission)],cwd=source,env=env,capture_output=True,text=True,timeout=30)
        (output/f'trial-{i+1}-verification.log').write_text(verify.stdout+verify.stderr)
        if verify.returncode: raise RuntimeError('packaged_replay_failed')
        value=json.loads(verify.stdout)
        if not value['eligible_common_race'] or value['future_race_evidence']: raise RuntimeError('synthetic_evidence_class_failed')
        def bundle(name):
            index=json.loads((trial/name/'prediction_bundle_index_v1.json').read_bytes())
            return trial/name/index['entries'][0]['directory']
        if json.loads((bundle('baseline')/'result.json').read_bytes())['prediction']!=json.loads((bundle('predictions')/'result.json').read_bytes())['prediction']:
            raise RuntimeError('production_probabilities_changed')
        summary=json.loads((bundle('predictions')/'comparison/summary.json').read_bytes())
        records.append({'trial':i+1,'baseline':baseline,'comparison':compared,'comparison_work':summary,
            'all_four_reproduced':True,'production_prediction_identical':True,'future_race_evidence':False})
        (output/'executed_trials.json').write_bytes(canonical_bytes(records))
    print(json.dumps({'source_commit':commit,'output':str(output),'trials':len(records),'future_race_evidence':False}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--python',type=Path,default=Path(sys.executable));p.add_argument('--repetitions',type=int,default=3)
    a=p.parse_args();run(a.output,a.python,a.repetitions)
