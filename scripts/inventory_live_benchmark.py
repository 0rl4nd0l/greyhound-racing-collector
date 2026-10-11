"""Metadata-only named operational population census; no outcome files opened."""
import collections, datetime, hashlib, json, re
from pathlib import Path
import argparse
parser = argparse.ArgumentParser(description="Read-only retained prediction inventory; no outcome files opened.")
parser.add_argument("--access-boundary", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=False)
BASE=Path('/mnt/tenn-nvme2/tenn')
OUT=args.output_dir
PREFIXES=['greyhound-persistent-engineering-20261003','greyhound-persistent-engineering-20261003-02','greyhound-persistent-scope-successor-20261010','greyhound-persistent-comparison-20261005','greyhound-development-pilot-20261003','greyhound-incident-engineering-20261001','greyhound-incident-engineering-20261002']
PREFIXES+= [p.name for p in BASE.glob('greyhound-live-engineering-202610*') if p.is_dir()]
boundary=json.loads(args.access_boundary.read_text());aliases=boundary['venue_aliases'];refs=[]
def read(path):
 b=path.read_bytes(); refs.append({'path':str(path),'sha256':hashlib.sha256(b).hexdigest()});return json.loads(b)
def canonical(race):
 m=re.fullmatch(r'Race (\d+) - (.+) - (\d{4}-\d{2}-\d{2})',race or '')
 if not m:return None
 n,v,d=m.groups();v=v.strip().upper();v=aliases.get(v,aliases.get(v.replace(' ','_'),v.replace(' ','_')));return f'{d}|{v}|{n}'
records=[];roots=[];opportunities=[]
for name in sorted(set(PREFIXES)):
 root=BASE/name
 for pred in [root/'predictions',root/'operational-predictions']:
  if not pred.exists():continue
  identities=sorted(set(pred.glob('races/*/identity.json'))|set(pred.glob('days/*/races/*/identity.json')))
  status=collections.Counter()
  for p in identities:
   d=read(p);t=p.with_name('terminal.json');terminal=read(t) if t.exists() else {};key=canonical(d.get('race_id'));s=terminal.get('status','UNRESOLVED_NO_TERMINAL');status[s]+=1
   date=key[:10] if key else None
   reasons=[]
   if key in boundary['canonical_protected_identity_keys']:reasons.append('RESERVED_EXACT_MEMBERSHIP')
   for w in boundary['protected_windows']:
    if date and date>=w['start'] and (w['end'] is None or date<=w['end']):reasons.append('RESERVED_DATE_WINDOW')
   records.append({'root':str(pred),'identity_path':str(p),'race_id':d.get('race_id'),'canonical_race_key':key,'status':s,'started_at':d.get('started_at'),'completed_at':terminal.get('completed_at'),'job_id':terminal.get('job_id'),'seconds_to_jump_at_verification':terminal.get('seconds_to_jump_at_verification'),'terminal_path':str(t) if t.exists() else None,'access_disposition':'EXCLUDED_SCIENTIFIC_ALLOCATION' if reasons else 'UNASSESSED','exclusion_reasons':reasons})
  roots.append({'root':str(pred),'identity_records':len(identities),'terminal_status_counts':dict(status)})
 runtime=root/'runtime'
 for p in sorted(runtime.glob('days/*/*/opportunities.json')):
  d=read(p)
  for row in d.get('opportunities',[]):opportunities.append({'source':str(p),'root':str(root),'observed_at':d.get('observed_at'),**row})
summary={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Named persistent/incident/live engineering metadata roots; not universal project census','records':len(records),'unique_canonical_races':len({x['canonical_race_key'] for x in records}),'statuses':dict(collections.Counter(x['status'] for x in records)),'scientific_dispositions':dict(collections.Counter(x['access_disposition'] for x in records)),'by_date':{},'roots':roots,'opportunity_snapshot_rows':len(opportunities),'opportunity_unique_urls':len({x['url'] for x in opportunities}),'no_outcomes_decoded':True,'no_result_score_computed':True}
for day in sorted({x['canonical_race_key'][:10] for x in records if x['canonical_race_key']}):
 r=[x for x in records if x['canonical_race_key'] and x['canonical_race_key'].startswith(day)]; summary['by_date'][day]={'records':len(r),'unique_races':len({x['canonical_race_key'] for x in r}),'status_counts':dict(collections.Counter(x['status'] for x in r))}
for name,obj in [('operational-census.json',{'summary':summary,'records':records}),('opportunity-snapshots.json',{'rows':opportunities}),('census-source-references.json',refs)]: (OUT/name).write_text(json.dumps(obj,indent=2)+'\n')
print(json.dumps(summary,indent=2))
