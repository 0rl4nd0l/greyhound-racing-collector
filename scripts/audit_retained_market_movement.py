"""Outcome-blind qualification of two pinned retained market surfaces.

No database or provider access. Scope is resolved before any source row decode.
The inventory never turns legacy WIN labels or capture times into qualification.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.explain_market_residual import FOUNDATION, load_scope

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = Path('/home/l4nd0/greyhound/artifacts/sportsbet_win_market_surface_audit_20260815_report_only/canonical_win_sidecar.jsonl')
EXTRACT = Path('/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector/artifacts/prejump_market_structure_experiment_20260815_report_only/sportsbet_source_extract.jsonl')
PINS = {
    CANONICAL: '880ae93680e56991fa2c9eb316732cbc71bc7ff713525efcf83750ceace4493d',
    EXTRACT: '5bd61de48887245e71c89e3af7722bebc2ddbe0ea127c5b41b46e13d02f1ccef',
}
PROTOCOL = ROOT/'docs/research/market_movement_20260929_protocol.md'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def scalar(line, field):
    matches = re.findall(r'"'+re.escape(field)+r'"\s*:\s*("(?:[^"\\]|\\.)*"|[0-9]+)', line)
    if not matches:
        raise ValueError('missing identity/scalar: '+field)
    values = [json.loads(v) for v in matches]
    if any(v != values[0] for v in values):
        raise ValueError('conflicting identity/scalar: '+field)
    return values[0]


def scoped_rows(payload, scope):
    """Decode full row only after exact race and runner-box membership checks."""
    admitted = {race for race, _ in scope}
    for line in payload.decode().splitlines():
        race = scalar(line, 'race_id')
        if race not in admitted:
            continue
        box = scalar(line, 'box_number')
        if (race, box) not in scope:
            raise ValueError('unadmitted runner before full row decode')
        yield json.loads(line)


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError('timestamp has no timezone')
    return parsed


def timing_pair(times, jump):
    """Inventory only: a capture timestamp is not proof of availability."""
    cutoff = jump-timedelta(minutes=2)
    late = [t for t in times if cutoff-timedelta(minutes=8) <= t <= cutoff]
    if not late:
        return None
    late = max(late)
    early = [t for t in times if jump-timedelta(minutes=30) <= t <= late-timedelta(minutes=2)]
    return (min(early), late) if early else None


def field_complete(rows, boxes):
    actual = [r['box_number'] for r in rows]
    return len(actual) == len(set(actual)) and set(actual) == set(boxes)


def audit(output):
    output.mkdir(parents=True, exist_ok=False)
    ledger = output/'trial_ledger.jsonl'
    def event(kind, **kw):
        with ledger.open('a') as handle:
            handle.write(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(), event=kind, **kw), sort_keys=True)+'\n')
    event('START', protocol_sha256=digest(PROTOCOL.read_bytes()), code_sha256=digest(Path(__file__).read_bytes()), fits=0, performance_trials=0)
    try:
        scope, inputs = load_scope()
        for source in (Path(__file__), Path(__file__).with_name('explain_market_residual.py'), PROTOCOL):
            inputs[str(source)] = digest(source.read_bytes())
        event('SCOPE_VERIFIED', races=len({r for r, b in scope}), runners=len(scope), target_labels_decoded=0)
        sources = {}
        for path, pin in PINS.items():
            payload = path.read_bytes()
            if digest(payload) != pin:
                raise ValueError('source hash drift: '+str(path))
            inputs[str(path)] = pin
            sources[path] = list(scoped_rows(payload, scope))
        # The scope gate has already hash-verified this file. Read only the two
        # pre-race scalar fields needed for timing, never labels or histories.
        jumps = {}
        admitted = {r for r,b in scope}
        for line in (FOUNDATION/'development.jsonl').read_text().splitlines():
            rid = scalar(line, 'race_id')
            if rid not in admitted:
                raise ValueError('development identity changed after scope verification')
            jump = timestamp(scalar(line, 'jump'))
            if rid in jumps and jumps[rid] != jump:
                raise ValueError('inconsistent scheduled jump')
            jumps[rid] = jump
        canonical = collections.defaultdict(list)
        extracted = collections.defaultdict(list)
        for row in sources[CANONICAL]: canonical[row['race_id']].append(row)
        for row in sources[EXTRACT]: extracted[row['race_id']].append(row)
        known_ids = {r['source_row_id']: r for r in sources[CANONICAL]}
        schema = {str(p): sorted(set().union(*(r.keys() for r in rr))) for p, rr in sources.items()}
        records = []
        for rid in sorted(jumps):
            boxes = {b for r,b in scope if r == rid}
            cc = canonical[rid]
            if not field_complete(cc, boxes):
                raise ValueError('canonical roster mismatch')
            if any(re.sub('[^A-Z0-9]', '', r['dog_name'].upper()) != scope[(rid,r['box_number'])] for r in cc):
                raise ValueError('canonical runner identity mismatch')
            if any(not r['race_qualified'] or r['source'] != 'sportsbet' or r['canonical_win_odds'] != r['paired_win_odds'] for r in cc):
                raise ValueError('canonical WIN proof mismatch')
            ctimes = {timestamp(r['capture_timestamp']) for r in cc}
            if len(ctimes) != 1:
                raise ValueError('non-atomic canonical snapshot')
            groups = collections.defaultdict(list)
            for row in extracted[rid]: groups[row['capture_timestamp']].append(row)
            complete = {timestamp(t): rows for t,rows in groups.items() if field_complete(rows, boxes)}
            # Cross-binding exact source row IDs establishes only rows actually
            # present in the canonical sidecar, never their earlier neighbours.
            bound = []
            for row in extracted[rid]:
                match = known_ids.get(row['id'])
                if match:
                    if any(row[k] != match[k] for k in ('race_id','box_number','capture_timestamp','source','source_url')):
                        raise ValueError('source ID identity conflict')
                    bound.append(row['id'])
            union_times = ctimes | set(complete)
            pair = timing_pair(union_times, jumps[rid])
            qualified_complete = [t for t,rows in complete.items() if all(r['id'] in bound for r in rows)]
            records.append(dict(
                race_id=rid, race_date=rid[-10:], runners=len(boxes), canonical_snapshot_count=len(ctimes),
                canonical_capture_lead_minutes=(jumps[rid]-next(iter(ctimes))).total_seconds()/60,
                extract_runner_rows=len(extracted[rid]), extract_snapshot_count=len(groups),
                extract_complete_box_sets=len(complete), extract_incomplete_box_sets=len(groups)-len(complete),
                exact_canonical_bound_extract_rows=len(bound), canonical_bound_extract_complete_snapshots=len(qualified_complete),
                union_distinct_capture_times=len(union_times), timing_only_candidate_pair=pair is not None,
                timing_only_early=pair[0].isoformat() if pair else None,
                timing_only_late=pair[1].isoformat() if pair else None,
                independent_availability_evidence=False, explicit_scratching_history=False,
                qualified_movement=False,
                exclusions=['independent_availability_timestamp_absent','explicit_scratching_and_market_status_absent']+
                    (['extract_absent'] if not extracted[rid] else ['earlier_WIN_reconstruction_and_runner_identity_unproved'])+
                    ([] if pair else ['no_complete_box_timing_pair_in_fixed_window']),
            ))
        def stage(label, predicate):
            rr = [r for r in records if predicate(r)]
            return dict(stage=label, races=len(rr), dates=len({r['race_date'] for r in rr}), runners=sum(r['runners'] for r in rr))
        summary = dict(
            verdict='INSUFFICIENT_QUALIFIED_DATA', fits=0, performance_trials=0, decoded_target_labels=0,
            protected_target_rows_decoded=0, database_opens=0, provider_requests=0,
            stages=[stage('admitted_development_population', lambda r: True),
                    stage('canonical_single_complete_corrected_WIN_snapshot', lambda r: True),
                    stage('fixed_window_extract_present', lambda r: r['extract_runner_rows']>0),
                    stage('two_or_more_complete_box_snapshots_in_extract', lambda r: r['extract_complete_box_sets']>=2),
                    stage('timing_only_pair_in_union_not_qualified', lambda r: r['timing_only_candidate_pair']),
                    stage('two_canonical_bound_WIN_snapshots', lambda r: r['canonical_bound_extract_complete_snapshots']>=2),
                    stage('qualified_movement_pair', lambda r: r['qualified_movement'])],
            canonical_rows=len(sources[CANONICAL]), extract_rows=len(sources[EXTRACT]),
            extract_rows_bound_to_canonical_WIN=sum(r['exact_canonical_bound_extract_rows'] for r in records),
            exclusion_counts=dict(collections.Counter(x for r in records for x in r['exclusions'])),
            source_fields=schema, inputs=inputs,
            limitation='Finite audit of the two evidence-linked retained surfaces; no claim that every local byte lacks other snapshots. Raw database reconstruction is not attempted because independent availability and scratching evidence are already absent.',
        )
        for name, value in [('qualification.json', summary), ('race_qualification.json', records)]:
            (output/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
        event('COMPLETE', verdict=summary['verdict'], fits=0, performance_trials=0, qualified_races=0)
        return summary
    except Exception as exc:
        event('FAILED', error=repr(exc), fits=0, performance_trials=0)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.output), indent=2))
