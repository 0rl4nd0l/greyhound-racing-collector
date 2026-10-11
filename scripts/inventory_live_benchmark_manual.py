import json,hashlib,re,collections,sqlite3
from pathlib import Path
import argparse
parser = argparse.ArgumentParser(description="Read-only retained prediction inventory; no outcome files opened.")
parser.add_argument("--access-boundary", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=False)
B=Path('/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0');O=args.output_dir;a=json.loads(args.access_boundary.read_text());aliases=a['venue_aliases'];protected=set(a['canonical_protected_identity_keys'])
patterns=['greyhound*/artifacts/on_demand_prediction_runs/*/request.json','greyhound*/output/*/request.json','greyhound*/output/*/*/request.json','greyhound*/isolated_output/*/request.json','greyhound*/prediction_*/request.json','greyhound*/artifacts/manual_prediction*/**/request.json']
ps=sorted(set(p for pat in patterns for p in B.glob(pat)));rows=[]
def ref(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
for p in ps:
 q=p.with_name('bundle_manifest.json');manifest=json.loads(q.read_text()) if q.exists() else {};d=json.loads(p.read_text());m=re.fullmatch(r'Race (\d+) - (.+) - (\d{4}-\d{2}-\d{2})',d.get('race_id',''));key=None
 if m:
  n,v,date=m.groups();v=v.upper();v=aliases.get(v,aliases.get(v.replace(' ','_'),v.replace(' ','_')));key=f'{date}|{v}|{n}'
 excluded=not key or key in protected or any(date>=w['start'] and (w['end'] is None or date<=w['end']) for w in a['protected_windows']);row={'request':ref(p),'manifest':ref(q) if q.exists() else None,'race_id':d.get('race_id'),'canonical_race_key':key,'requested_at':d.get('request_timestamp'),'access_disposition':'EXCLUDED_SCIENTIFIC_ALLOCATION' if excluded else 'AUTHORIZED_NONRESERVED_RETAINED_OPERATIONAL'}
 if not excluded:
  r=p.with_name('result.json');z=json.loads(r.read_text()) if r.exists() else {'status':'MISSING_RESULT_ARTIFACT'};row.update({'prediction_result':ref(r) if r.exists() else None,'status':z.get('status'),'blocker_codes':[x.get('code') for x in z.get('blockers',[])],'manifest_files_valid':all((p.parent/k).exists() and hashlib.sha256((p.parent/k).read_bytes()).hexdigest()==v['sha256'] for k,v in manifest.get('files',{}).items())})
 rows.append(row)
p=B/'greyhound-operator-ui-r3-repair-operations-20260811/jobs.sqlite3';c=sqlite3.connect(f'file:{p}?mode=ro',uri=True);jobs=[]
for job,at,race in c.execute("select job_id,created_at,json_extract(input_json,'$.race_id') from jobs where created_at<'2026-08-18'"):
 events=c.execute('select phase,status,reason,event_at from job_events where job_id=? order by sequence',(job,)).fetchall();jobs.append({'job_id':job,'created_at':at,'race_id':race,'source':str(p),'events':[dict(zip(['phase','status','reason','event_at'],e)) for e in events]})
d={'patterns':patterns,'base':str(B),'requests':rows,'early_r3_jobs':jobs,'summary':{'requests':len(rows),'dispositions':dict(collections.Counter(r['access_disposition'] for r in rows)),'authorized_statuses':dict(collections.Counter(r['status'] for r in rows if 'status' in r)),'early_r3_jobs':len(jobs)}};(O/'manual-census.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d['summary'],indent=2));print([(r['race_id'],r.get('status'),r['request']['path'])for r in rows if r.get('status') in ['OK','SUCCESS','PREDICTION_READY']])
