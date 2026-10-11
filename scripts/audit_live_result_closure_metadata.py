#!/usr/bin/env python3
"""Intersect named private closure identities with the frozen authorised population.

Read identity/state columns only. Never decode closure bodies, results or labels
outside the population. A match is a lead for a separate authorised investigation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3


RACE_ID = re.compile(rb'"race_id"\s*:\s*("(?:[^"\\]|\\.)*")')


def scalar_identities(path):
    return sorted({json.loads(m.group(1)) for m in RACE_ID.finditer(path.read_bytes())})


def audit(base, population):
    allowed = {r['race_id'] for r in json.loads(population.read_text())['records']}
    evidence = base/'offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525'
    items = []
    patterns = [
        'greyhound-persistent-comparison-20261005/results*/queue.sqlite3',
        'greyhound-persistent-comparison-20261005/results*/official-results.sqlite3',
        'greyhound-live-engineering-20261002*/results/*/queue.sqlite3',
        'greyhound-live-engineering-20261002*/results/*/official-results.sqlite3',
    ]
    for path in sorted({p for pattern in patterns for p in base.glob(pattern)}):
        with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True) as con:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            identities, states = set(), Counter()
            if 'jobs' in tables:
                for race, state in con.execute('SELECT race, state FROM jobs'):
                    identities.add(race)
                    states[state] += 1
            if 'autonomous_official_result_evidence_races' in tables:
                identities.update(r[0] for r in con.execute('SELECT race_id FROM autonomous_official_result_evidence_races'))
            items.append(dict(path=str(path), kind='identity_only_sqlite_snapshot',
                              identities=sorted(identities), states=dict(states),
                              matches=sorted(identities & allowed)))
    metadata = set((evidence/'forward_prediction_journal_v1_private_20260825').glob('*/prediction_provenance_v1.json'))
    metadata.update((evidence/'single_prediction_canary_private_20260825T1540').glob('canary_provenance_v1.json'))
    corpus = evidence/'forward_sealed_corpus_official_result_first_observation_v1'
    # Named provenance and request-stage records only; no closure/result bodies.
    metadata.update(corpus.glob('races/*/*prejump*.json'))
    metadata.update(corpus.glob('races/*/official-requests/*/response-stage.json'))
    for path in sorted(metadata):
        identities = set(scalar_identities(path))
        items.append(dict(path=str(path), kind='identity_scalar_only_metadata',
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          identities=sorted(identities), matches=sorted(identities & allowed)))
    all_ids = {race for item in items for race in item['identities']}
    return dict(schema='named_private_closure_identity_audit_v1',
                population_path=str(population), population_sha256=hashlib.sha256(population.read_bytes()).hexdigest(),
                scope='Named historical closure lanes only; not a whole-host archive search. No outcome columns or closure bodies decoded.',
                snapshot_caveat='SQLite identity/state projections may evolve if their original operational owner writes new records.',
                metadata_files=sum(i['kind']=='identity_scalar_only_metadata' for i in items),
                databases=sum(i['kind']=='identity_only_sqlite_snapshot' for i in items),
                distinct_identities=len(all_ids), matches=sorted(all_ids & allowed), items=items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--population', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.base, args.population)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'items'}))


if __name__ == '__main__':
    main()
