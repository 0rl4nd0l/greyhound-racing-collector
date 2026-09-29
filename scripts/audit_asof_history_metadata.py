"""Support-only receipts from the pinned completed history reconstruction.

No fitting or history-value decoding. Duplicate counts are identifiable here
because the completed reconstruction has zero conflicts and never hits cap20.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
from scripts.explain_market_residual import load_scope, gate_lines, FOUNDATION, sha, write

PAIRED_SHA='44e7a0f6013ae1acaa84d453a3c9460205f805740ab109af794a8dd6a9b5e63e'


def run(root, output):
    allowed,pins=load_scope()
    raw=(FOUNDATION/'development.jsonl').read_bytes()
    if sha(raw)!=pins[str(FOUNDATION/'development.jsonl')]:raise ValueError('development identity')
    identities={(r['race_id'],r['box']):r['dog_token'] for r in gate_lines(raw,allowed)}
    payload=(root/'paired_features.jsonl').read_bytes()
    if sha(payload)!=PAIRED_SHA:raise ValueError('paired feature identity')
    rows=[json.loads(line) for line in payload.splitlines()]
    by_runner={(r['race_id'],identities[r['race_id'],r['box']]):r for r in rows}
    if len(by_runner)!=len(rows):raise ValueError('within-field runner token collision')
    summary=json.loads((root/'summary.json').read_bytes())
    if summary['exclusions'] or any(r['richer_count']>=20 for r in rows):raise ValueError('duplicate count requires nonbinding cap and no conflicts')
    observed=0
    for r in rows:
        token=identities[r['race_id'],r['box']]
        prior=[by_runner[rid,token] for rid in r['prior_card_races']]
        if any(datetime.fromisoformat(p['card_capture'])>=datetime.fromisoformat(r['card_capture']) for p in prior):raise ValueError('cutoff order')
        observed+=r['short_count']+sum(p['short_count'] for p in prior)
    unique=sum(r['richer_count'] for r in rows)
    result={'paired_features_sha256':PAIRED_SHA,'script_sha256':sha(Path(__file__).read_bytes()),
        'reconstruction_script_sha256':sha((Path(__file__).parent/'reconstruct_asof_card_history.py').read_bytes()),
        'reservation_and_development_pins':pins,'metadata_source':str(root),
        'original_short_observations':sum(r['short_count'] for r in rows),
        'union_observations_considered':observed,'exact_duplicate_observations_removed':observed-unique,
        'unique_retained_observations':unique,'cap20_rows_removed':0,'conflicting_target_fields':0,
        'distinct_runner_tokens':len(set(identities.values())),'within_field_token_collisions':0}
    write(output,result)
    print(json.dumps({k:v for k,v in result.items() if isinstance(v,int)},sort_keys=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reconstruction',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.reconstruction,a.output)
