"""Replay sealed forecasts only with explicitly approved producing packages."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from contextlib import redirect_stdout, redirect_stderr

if __package__:
    from .readonly_json import read
else:
    from readonly_json import read


def digest(value):
    return hashlib.sha256((json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()).hexdigest()


def validate_packages(entries):
    if not isinstance(entries,list) or not 1 <= len(entries) <= 32:
        raise ValueError('invalid_approved_producer_bound')
    identities=set()
    for entry in entries:
        if set(entry) != {'plan','source_root','commit','tree','source_identity_sha256','configuration_sha256'}:
            raise ValueError('invalid_approved_producer')
        if set(entry['plan']) != {'path','sha256'}:
            raise ValueError('invalid_approved_native_plan')
        for name in ('commit','tree','source_identity_sha256','configuration_sha256'):
            if not isinstance(entry[name],str) or not re.fullmatch('[0-9a-f]{'+str(40 if name in ('commit','tree') else 64)+'}',entry[name]):
                raise ValueError('invalid_approved_producer_hash')
        if not re.fullmatch('[0-9a-f]{64}',entry['plan']['sha256']):
            raise ValueError('invalid_approved_native_hash')
        for value in (entry['source_root'],entry['plan']['path']):
            path=Path(value)
            if not path.is_absolute() or path.resolve()!=path:
                raise ValueError('unsafe_approved_producer_path')
        identity=(entry['plan']['path'],entry['plan']['sha256'])
        if identity in identities:
            raise ValueError('duplicate_approved_producer')
        identities.add(identity)
    return entries


def verify_source(entry):
    source=Path(entry['source_root'])
    identity=read(source/'SOURCE_IDENTITY.json')
    if digest(identity)!=entry['source_identity_sha256'] or any(identity[name]!=entry[name] for name in ('commit','tree')):
        raise ValueError('approved_producer_identity_changed')
    files=identity['files']
    if not isinstance(files,dict) or not 1 <= len(files) <= 4096:
        raise ValueError('approved_producer_file_bound')
    for name,expected in files.items():
        path=source/name
        if Path(name).is_absolute() or not path.is_relative_to(source) or path.resolve()!=path or not re.fullmatch('[0-9a-f]{64}',expected):
            raise ValueError('unsafe_approved_producer_file')
        with path.open('rb') as handle:
            hasher=hashlib.sha256()
            while chunk:=handle.read(1024*1024):
                hasher.update(chunk)
        if hasher.hexdigest()!=expected:
            raise ValueError('approved_producer_file_changed')
    # Imported code cannot come from an extra module outside the pinned archive.
    for path in source.rglob('*'):
        if path.is_symlink():
            raise ValueError('unapproved_producer_symlink')
        if path.is_file() and path.suffix in ('.py','.pyc','.so') and path.relative_to(source).as_posix() not in files:
            raise ValueError('unapproved_producer_module')
    return identity


def select_package(binding,receipt,output):
    entries=validate_packages(binding['producer_packages'])
    matches=[entry for entry in entries if entry['plan']==receipt['plan']]
    if len(matches)!=1:
        raise ValueError('unapproved_producer_package')
    entry=matches[0]
    plan_path=Path(entry['plan']['path'])
    if not plan_path.is_relative_to(output) or not Path(entry['source_root']).is_relative_to(output):
        raise ValueError('approved_producer_outside_package')
    plan=read(plan_path,entry['plan']['sha256'])
    if receipt['configuration_sha256']!=entry['configuration_sha256'] or any(plan[name]!=entry[name] for name in ('source_root','commit','tree','source_identity_sha256')):
        raise ValueError('approved_producer_plan_changed')
    verify_source(entry)
    return entry


def attempt_package(binding,comparison,admission_path,allocation):
    """Bind the sealed admission's retained publication to one approved producer."""
    admission=read(admission_path)
    completion=read(admission_path.with_name('completion.json'))
    bundles=Path(allocation['prediction_root'])/'bundles'
    directory=bundles/completion['bundle_entry']['directory']
    if directory.resolve()!=directory or not directory.is_relative_to(bundles):
        raise ValueError('unsafe_producer_bundle')
    manifest=read(directory/'bundle_manifest.json',completion['bundle_entry']['manifest_sha256'])
    request=read(directory/'request.json',manifest['files']['request.json']['sha256'])
    if request['job_id']!=admission['job_id']:
        raise ValueError('producer_request_admission_changed')
    provenance=request['operational_index_provenance']
    run_id=provenance['run_id']
    if not re.fullmatch('[A-Za-z0-9_+.-]{1,128}',run_id):
        raise ValueError('invalid_producer_run_id')
    implementation=read(directory/'features/sealed/implementation_file_manifest.json',
        manifest['files']['features/sealed/implementation_file_manifest.json']['sha256'])
    matches=[]
    for entry in validate_packages(binding['producer_packages']):
        native=read(entry['plan']['path'],entry['plan']['sha256'])
        if native.get('frozen_comparison')!=comparison:
            continue
        evidence=Path(native['evidence_root'])
        package=Path(entry['source_root']).parent
        if evidence.resolve()!=evidence or not evidence.is_relative_to(package):
            raise ValueError('unsafe_producer_evidence')
        report=evidence/('shadow_autopilot_v1_'+run_id+'_phase_0')/'current_race_index_publish.json'
        if not report.exists():
            continue
        publication=read(report,provenance['report_sha256'])
        refresh=Path(publication['source_refresh_report_path'])
        if refresh.resolve()!=refresh or not refresh.is_relative_to(evidence):
            raise ValueError('unsafe_producer_refresh')
        read(refresh,provenance['source_refresh_sha256'])
        if (publication['run_id']!=run_id or publication['packet_sha256']!=provenance['packet_sha256']
                or publication['source_refresh_report_sha256']!=provenance['source_refresh_sha256']
                or publication['source_refresh_report_path']!=str(refresh)):
            raise ValueError('producer_publication_changed')
        identity=verify_source(entry)
        hashes=implementation['implementation_file_hashes']
        if set(hashes)!=set(implementation['implementation_files']) or not hashes or any(identity['files'].get(name)!=sha for name,sha in hashes.items()):
            raise ValueError('producer_implementation_changed')
        # Approval includes the immutable native plan, not just compatible files.
        select_package(binding,{'plan':entry['plan'],'configuration_sha256':entry['configuration_sha256']},package)
        matches.append(entry)
    if len(matches)!=1:
        raise ValueError('producer_identity_missing_or_ambiguous')
    return matches[0]


def replay_group(binding,entry,comparison,selected,*,remaining,deadline):
    seconds=30 if deadline is None else max(.1,min(30,deadline-time.monotonic()))
    request={'entry':entry,'receipt':{'plan':entry['plan'],'configuration_sha256':entry['configuration_sha256']},
        'output':str(Path(entry['source_root']).parent),'comparison':comparison,'remaining':remaining,'seconds':seconds,'selected':selected}
    result=subprocess.run([binding['python'],'-B',str(Path(__file__).resolve()),'--worker'],
        input=json.dumps(request).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        timeout=seconds+2,check=True,cwd=entry['source_root'],
        env={'PATH':os.defpath,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','TRACE_MALLOC':'0'})
    if len(result.stdout)>4*1024*1024:
        raise ValueError('producer_response_bound')
    verify_source(entry)
    value,limited,used=json.loads(result.stdout)
    for category in ('forecasts','failed_forecasts'):
        for forecast in value[category]:
            forecast['producer']={name:entry[name] for name in ('plan','source_root','commit','tree','source_identity_sha256','configuration_sha256')}
    return value,limited,used


def replay(binding,receipt,output,comparison,*,remaining=255,deadline=None):
    select_package(binding,receipt,output)
    from src.predictor.future_comparison import load_plan
    from race_collection.persistent_comparison import validate_persistent_plan
    plan,_=load_plan(Path(comparison['path']),comparison['sha256'])
    allocation=validate_persistent_plan(plan)
    admissions=sorted((Path(plan['programme_root'])/comparison['sha256']/'attempts').glob('*/admission.json'))
    if len(admissions)>min(allocation['max_capture_attempts'],255):
        raise ValueError('producer_admission_bound')
    groups={};value={'forecasts':[],'failed_forecasts':[],'forecast_errors':[]};used=0;limited=False
    for path in admissions:
        if used>=remaining or (deadline is not None and time.monotonic()>=deadline):
            limited=True;break
        used+=1
        if not path.with_name('completion.json').exists():
            continue
        try:
            entry=attempt_package(binding,comparison,path,allocation)
            group=groups.setdefault(entry['plan']['sha256'],(entry,[]))
            group[1].append(str(path))
        except Exception:
            value['forecast_errors'].append({'attempt':path.parent.name,'reason':'Sealed comparison producer could not be verified.'})
    for entry,selected in groups.values():
        if deadline is not None and time.monotonic()>=deadline:
            limited=True;break
        try:
            checked,partial,_=replay_group(binding,entry,comparison,selected,remaining=len(selected),deadline=deadline)
            limited |= partial
            for category in value:value[category].extend(checked[category])
        except Exception:
            value['forecast_errors'].append({'reason':'Sealed comparison producer replay could not be verified.'})
    return value,limited,used


def worker(request):
    entry=select_package({'producer_packages':[request['entry']]},request['receipt'],Path(request['output']))
    if not os.statvfs(entry['source_root']).f_flag & os.ST_RDONLY:
        raise ValueError('producer_worker_not_readonly')
    native=read(entry['plan']['path'],entry['plan']['sha256'])
    if native['frozen_comparison']!=request['comparison']:
        raise ValueError('producer_worker_comparison_changed')
    # UI replay orchestration is trusted separately from the producing package.
    from persistent_collector import verified_attempts
    sys.path.insert(0,entry['source_root'])
    from src.predictor.future_comparison import load_plan
    from race_collection.persistent_comparison import validate_persistent_plan
    plan,_=load_plan(Path(request['comparison']['path']),request['comparison']['sha256'])
    allocation=validate_persistent_plan(plan)
    value=verified_attempts(request['comparison']['sha256'],plan,allocation,
        remaining=request['remaining'],deadline=time.monotonic()+request['seconds'],selected=request['selected'])
    verify_source(entry)
    return value


if __name__=='__main__':
    raw=sys.stdin.buffer.read(65537)
    if len(raw)>65536:
        raise ValueError('producer_request_bound')
    with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
        value=worker(json.loads(raw))
    print(json.dumps(value,sort_keys=True,allow_nan=False))
