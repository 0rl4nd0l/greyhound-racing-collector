"""Copy and hash the approved result DB at closure, without decoding outcomes.

No providers, metrics, evaluation authority creation, or scientific activation.
The result owner runs this only after collection closes and its writer is idle.
"""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import argparse,hashlib,json,os
from src.predictor.future_comparison import load_plan,checked,stamp,put


def seal(binding_path, output, *, now=None):
    now=now or datetime.now(timezone.utc);os.umask(0o077)
    binding=json.loads(binding_path.read_bytes());plan,_=load_plan(Path(binding['plan']),binding['plan_sha256'])
    authority=json.loads(checked(Path(binding['authority']),binding['authority_sha256']))
    cutoff=stamp(plan['ends_at'])+timedelta(days=14)
    if (plan['status']!='AUTHORIZED' or authority.get('status')!='AUTHORIZED_MACHINE_RESULT_RETENTION'
            or authority.get('plan_sha256')!=binding['plan_sha256'] or not authority.get('owner')
            or not authority.get('authority_reference') or authority.get('human_outcome_access') is not False
            or now<cutoff):raise ValueError('closure_not_authorized_or_not_due')
    database=Path(authority['result_database'])
    if not database.is_absolute() or database.is_symlink():raise ValueError('closure_database_unsafe')
    sidecars=[Path(str(database)+s) for s in ('-wal','-shm','-journal')]
    if any(p.exists() or p.is_symlink() for p in sidecars):raise ValueError('closure_writer_not_quiescent')
    before=database.stat();raw=database.read_bytes();after=database.stat()
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if identity(before)!=identity(after) or any(p.exists() for p in sidecars):raise ValueError('closure_database_changed')
    output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'official-results.sqlite3'
    with snapshot.open('xb') as stream:stream.write(raw)
    snapshot.chmod(0o400)
    receipt={'status':'RESULT_CLOSURE_SEALED_NOT_EVALUATED','plan_sha256':binding['plan_sha256'],
        'closure_cutoff':cutoff.isoformat(),'sealed_at':now.isoformat(),'result_database':str(snapshot.absolute()),
        'result_database_sha256':hashlib.sha256(raw).hexdigest(),'target_values_decoded':False,
        'result_authority_sha256':binding['authority_sha256']}
    put(output/'closure.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(seal(a.binding,a.output)))
