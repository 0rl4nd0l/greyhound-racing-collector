#!/usr/bin/env python3
"""Read-only exact-allowlist official result projection; never reads other outcomes."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def project(membership_path, database):
    membership = json.loads(membership_path.read_text())
    allowed = {r['race_id'] for r in membership['records'] if r['access_disposition'] == 'AUTHORIZED_NONRESERVED_RETAINED_OPERATIONAL'}
    con = sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True)
    con.execute('PRAGMA query_only=ON')
    con.execute('BEGIN')  # One consistent read snapshot across race and runner projections.
    con.row_factory = sqlite3.Row
    records, missing = [], []
    for race_id in sorted(allowed):
        races = [dict(r) for r in con.execute('SELECT race_id,race_date,venue,race_number,source,source_url,status,position_count,participant_count,participant_source,captured_at,source_artifact_dir,row_json FROM autonomous_official_result_evidence_races WHERE race_id=?', (race_id,))]
        if not races:
            missing.append(race_id)
            continue
        runners = [dict(r) for r in con.execute('SELECT race_id,source,source_url,box_number,dog_name,finish_position,is_winner,captured_at,source_artifact_dir,row_json FROM autonomous_official_result_evidence_runners WHERE race_id=?', (race_id,))]
        records.append(dict(race_id=race_id, races=races, runners=runners))
    con.close()
    return dict(schema='authorised_official_result_projection_v1', membership_path=str(membership_path),
                membership_sha256=hashlib.sha256(membership_path.read_bytes()).hexdigest(), database=str(database),
                query='Exact race_id parameter for authorised nonreserved membership only; read-only SQLite',
                allowed_races=len(allowed), result_races=len(records), records=records, missing_race_ids=missing)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--membership', required=True, type=Path)
    p.add_argument('--database', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    result = project(args.membership, args.database)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k: result[k] for k in ('allowed_races', 'result_races')}))


if __name__ == '__main__':
    main()
