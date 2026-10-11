"""Outcome-free retained stage-status census; never opens result files."""
import json,collections,re,hashlib
from pathlib import Path
import argparse
parser = argparse.ArgumentParser(description="Read-only retained prediction inventory; no outcome files opened.")
parser.add_argument("--access-boundary", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=False)
OUT=args.output_dir
E=Path('/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525')
a=json.loads(args.access_boundary.read_text());aliases=a['venue_aliases'];reserved=set(a['canonical_protected_identity_keys']);rows=[];stages=[];refs=[]
def key(r):
 m=re.fullmatch(r'Race (\d+) - (.+) - (\d{4}-\d{2}-\d{2})',r);n,v,date=m.groups();v=v.strip().upper();v=aliases.get(v,aliases.get(v.replace(' ','_'),v.replace(' ','_')));return f'{date}|{v}|{n}'
for day in range(17,23):
 for p in sorted(E.glob(f'shadow_autopilot_daemonization_v1_202607{day}*/early_residual_shadow_status.json')):
  raw=p.read_bytes();d=json.loads(raw);assert d['outcomes_read'] is False
  refs.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest()})
  stages.append({'path':str(p),'status':d['status'],'race_count':d['race_count'],'blockers':d.get('plan',{}).get('blockers',[])})
  for r in d.get('races',[]):
   k=key(r['race_id'])
   failure_reason=None;failure_ref=None
   if k not in reserved and r['status']=='BLOCKED':
    q=Path(r.get('score_step',{}).get('stderr_path',''))
    if q.is_file():
     b=q.read_bytes();failure_ref={'path':str(q),'sha256':hashlib.sha256(b).hexdigest()};refs.append(failure_ref)
     try:
      reason=json.loads(b)
      if set(reason)<= {'reason','status'}:failure_reason=reason.get('reason')
     except (ValueError,TypeError):pass
   rows.append({'path':str(p),'race_id':r['race_id'],'canonical_race_key':k,'reserved':k in reserved,'status':r['status'],'blocker':r.get('blocker'),'failure_reason':failure_reason,'failure_reference':failure_ref,'feature_started_at':r.get('feature_step',{}).get('started_at'),'score_finished_at':r.get('score_step',{}).get('finished_at')})
byrace=collections.defaultdict(list)
for r in rows:byrace[r['canonical_race_key']].append(r)
races=[]
for k,rs in sorted(byrace.items()):
 states=collections.Counter(r['status'] for r in rs);races.append({'canonical_race_key':k,'race_ids':sorted(set(x['race_id'] for x in rs)),'reserved':rs[0]['reserved'],'attempt_records':len(rs),'status_counts':dict(states),'any_append':states['APPENDED']>0,'blockers':dict(collections.Counter(x['blocker'] for x in rs if x['blocker']))})
summary={'stage_files':len(stages),'stage_statuses':dict(collections.Counter(x['status'] for x in stages)),'attempt_records':len(rows),'attempt_statuses':dict(collections.Counter(x['status'] for x in rows)),'unique_canonical_races':len(races),'nonreserved_unique_races':sum(not x['reserved'] for x in races),'nonreserved_races_with_append':sum(not x['reserved'] and x['any_append'] for x in races),'reserved_unique_races':sum(x['reserved'] for x in races),'nonreserved_without_append':sum(not x['reserved'] and not x['any_append'] for x in races),'nonreserved_failure_reasons':dict(collections.Counter(r['failure_reason'] or 'UNRESOLVED_FAILURE_REASON' for r in rows if not r['reserved'] and r['status']=='BLOCKED')),'limitation':'Attempted-stage denominator only. Skipped-no-new-capture stages do not establish unobserved racing calendar or promise to predict.'}
(OUT/'july-operational-coverage.json').write_text(json.dumps({'summary':summary,'races':races,'attempts':rows,'stages':stages,'sources':refs},indent=2)+'\n'); print(json.dumps(summary,indent=2))
