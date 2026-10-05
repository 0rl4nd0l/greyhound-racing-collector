#!/usr/bin/env python3
"""Default-off private native-v2 derivation; never evaluate outcomes."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def replay_worker():
    """Fresh interpreter: only the authenticated original package may supply code."""
    try:
        raw=sys.stdin.buffer.read(2*1024*1024+1)
        if len(raw)>2*1024*1024:
            raise ValueError('worker_input_bound')
        request=json.loads(raw)
        source=Path(request['source_root'])
        if not source.is_absolute() or source.resolve()!=source:
            raise ValueError('worker_source_path')
        current=Path(__file__).resolve().parents[1]
        sys.path=[str(source)]+[p for p in sys.path if p and
            Path(p).resolve() not in {current,current/'scripts',Path.cwd().resolve()}]
        runtime=request['runtime_identity'];lock=request['environment_lock']
        distributions=sorted(({'name':d.metadata['Name'],'version':d.version,
            'record_sha256':hashlib.sha256((d.read_text('RECORD') or '').encode()).hexdigest()}
            for d in importlib.metadata.distributions()),
            key=lambda r:(r['name'],r['version'],r['record_sha256']))
        if (runtime['executable']!=sys.executable or runtime['version']!=sys.version
                or runtime['prefix']!=sys.prefix or runtime['distributions']!=distributions
                or lock['python']!=platform.python_version()
                or any(importlib.metadata.version(name)!=version for name,version in lock['packages'].items())):
            raise ValueError('worker_computational_runtime_changed')
        import scripts.predict_market_form_residual as producer
        import src.predictor.market_form_residual as scorer
        for module in (producer,scorer):
            if not Path(module.__file__).resolve().is_relative_to(source):
                raise ValueError('worker_mixed_source_import')
        from datetime import datetime
        artifact=producer.score_from_artifacts(race_id=request['race_id'],
            score_timestamp=datetime.fromisoformat(request['historical_validation_anchor']),
            **{key+'_path':Path(value) for key,value in request['replay_paths'].items()})
        # Deliberately discard the temporary v3 record commitments and timestamp.
        response={key:artifact[key] for key in ('race_id','jump_timestamp','model_sha256',
            'manifest_sha256','effective_state_sha256','variants','predictions','input_hashes',
            'feature_freeze_timestamp','odds_capture_timestamp','odds_append_timestamp')}
        response['worker_status']='REPLAYED_WITH_HISTORICAL_VALIDATION_ANCHOR'
        response['loaded_source_files']={}
        for module in tuple(sys.modules.values()):
            name=getattr(module,'__file__',None)
            if name and Path(name).resolve().is_relative_to(source):
                path=Path(name).resolve()
                if path.suffix=='.py':
                    response['loaded_source_files'][str(path.relative_to(source))]=hashlib.sha256(path.read_bytes()).hexdigest()
        sys.stdout.buffer.write(encoded(response))
        return 0
    except Exception as exc:
        # Parent retains only a static category. No input values or exception text.
        allowed={'worker_input_bound','worker_source_path','worker_computational_runtime_changed',
            'worker_mixed_source_import','feature_generator_implementation_hash_mismatch',
            'source_timestamp_order_invalid','manual_score_not_prejump'}
        sys.stdout.buffer.write(encoded({'worker_status':'REPLAY_REJECTED','exception_type':type(exc).__name__,
            'failure_category':str(exc) if str(exc) in allowed else 'original_producer_rejected'}))
        return 2


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--authority',type=Path)
    parser.add_argument('--authority-sha256')
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({'status':'RETROSPECTIVE_NATIVE_V2_DISABLED'}));return 0
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from src.predictor.retrospective_native_v2 import run
    return run(args.authority,args.authority_sha256)


if __name__=='__main__':
    raise SystemExit(replay_worker() if sys.argv[1:]==['--worker'] else main())
