"""Outcome-blind, pinned feasibility audit. No provider, DB or model access."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import re
from datetime import date, datetime, timezone
from pathlib import Path

from scripts import offline_form_packet as form

ROOT = Path(__file__).resolve().parents[1]
FOUNDATION = Path('/home/l4nd0/greyhound-offline-systematic-output-20260924/foundation')
REVIEW = Path('/home/l4nd0/greyhound-comparison-october1-20260928/docs/research/future_comparison_20260928_evidence/reservation_review.json')
INCIDENT = Path('/home/l4nd0/greyhound-offline-prediction-20260924/docs/research/offline_20260924_access_incident.json')
PINS = {
    'development.jsonl': 'c58bd59bc0d52666d31812dccd60981de7f2f03b64862f98ddf9f88ef60cec46',
    'protected_records.json': '9fddce8fa70ea96c4d7a6c33ef61594ddeb47e254cd08bb87555b2d94fc4da88',
    'dataset_assessment.json': 'b0737c5b1fcf27ba34333ea824bbc363ab4d4a5c9986bc6b04b453ed81e16bea',
    'form_provenance.json': 'e7a99b20ba682f87e3f990e7b10566677408d061e5c154b05e65db0a6087059d',
}
INCIDENT_PIN = '724b4bf449aaa17f3f1419b579bf38ea4e836a776a6b1366339c44f821eb5f58'
REVIEW_PIN = '50c2ccd5fe13e2ae4c2102812ef76e81ea42de474acff4766597958b13ba19d9'
AMBIGUOUS = {'QOT', 'RICH', 'MURR'}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def scalar(line, field):
    """Only scalar identities are decoded; complete outcome records never are."""
    matches = list(re.finditer(r'"' + re.escape(field) + r'"\s*:\s*("(?:[^"\\]|\\.)*"|[0-9]+)', line))
    if not matches:
        raise ValueError('missing/ambiguous identity: ' + field)
    values = [json.loads(match[1]) for match in matches]
    # August identity manifests repeat the same race ID inside provenance.
    # Decode only those identity scalars; a conflicting nested identity fails.
    if any(value != values[0] for value in values):
        raise ValueError('missing/ambiguous identity: ' + field)
    return values[0]


def key(rid):
    m = re.fullmatch(r'Race (\d+) - (.+) - (\d{4}-\d{2}-\d{2})', rid)
    if not m:
        raise ValueError('unsupported race identity')
    return f'{m[3]}|{m[2]}|{int(m[1])}'


def identities(payload, denied, incident):
    rows = []
    for line in payload.decode().splitlines():
        rid = scalar(line, 'race_id')
        day = scalar(line, 'race_date')
        if key(rid) in denied or rid in incident or not '2026-06-10' <= day <= '2026-07-09' or key(rid)[:10] != day:
            raise ValueError('ineligible identity before record/card decode')
        rows.append({f: scalar(line, f) for f in ('race_id', 'race_date', 'dog_token', 'box')})
    pairs = [(r['race_id'], r['box']) for r in rows]
    if len(pairs) != len(set(pairs)) or any(not 1 <= r['box'] <= 8 for r in rows):
        raise ValueError('invalid or duplicate target box')
    return rows


def parse_number(value):
    raw = '' if value is None else str(value).strip()
    if not raw:
        return 'missing', None
    try:
        number = float(raw)
    except ValueError:
        return 'malformed', None
    if not math.isfinite(number):
        return 'nonfinite', None
    if number <= 0:
        return 'nonpositive', None
    return 'positive', number


def neighbours(box, boxes):
    left = max((b for b in boxes if b < box), default=None)
    right = min((b for b in boxes if b > box), default=None)
    return [{'box': b, 'gap': abs(box-b)} for b in (left, right) if b is not None]


def freeze_sample(race_ids):
    by_venue = collections.defaultdict(list)
    for rid in race_ids:
        by_venue[rid.split(' - ')[1]].append(rid)
    return sorted(min(ids, key=lambda rid: (digest(('early-speed-20260928-v1|' + rid).encode()), rid)) for ids in by_venue.values())


def retained_raw_header(metadata):
    """Only for preselected, admitted, hash-verified pre-race sidecars."""
    name, pin = metadata.get('raw_export_path'), metadata.get('raw_content_sha256')
    result = {'path': name, 'sha256': pin}
    if not name or not pin:
        return {**result, 'status': 'missing_path_or_hash_do_not_decode'}
    path = Path(name)
    if not path.is_file():
        return {**result, 'status': 'retained_path_absent'}
    payload = path.read_bytes()
    if digest(payload) != pin:
        return {**result, 'status': 'raw_hash_mismatch_do_not_decode'}
    first = payload.decode('utf-8', errors='replace').splitlines()[0]
    header = next(csv.reader([first], delimiter='|' if first.count('|') > first.count(',') else ','))
    return {**result, 'status': 'hash_verified_header_only', 'bytes': len(payload), 'header': header}


def audit(out):
    out.mkdir(parents=True, exist_ok=False)
    ledger = out/'trial_ledger.jsonl'

    def event(kind, **details):
        with ledger.open('a') as handle:
            handle.write(json.dumps({'time_utc': datetime.now(timezone.utc).isoformat(), 'event': kind, **details}, sort_keys=True) + '\n')

    executed = Path(__file__).read_bytes()
    (out/'audit_executed.py').write_bytes(executed)
    event('START', fits=0, target_outcomes_decoded=0, code_sha256=digest(executed))
    try:
        inputs = {}

        def verified(path, pin):
            value = form._verified(path, pin)
            inputs[str(path)] = {'sha256': pin, 'bytes': len(value)}
            return value

        # Restrictions before any mixed development file, then check original pins.
        deny = json.loads(verified(FOUNDATION/'protected_records.json', PINS['protected_records.json']))['records']
        incident = json.loads(verified(INCIDENT, INCIDENT_PIN))
        review = json.loads(verified(REVIEW, REVIEW_PIN))
        assessment = json.loads(verified(FOUNDATION/'dataset_assessment.json', PINS['dataset_assessment.json']))
        original = {}
        for name, pin in assessment['inputs'].items():
            payload = verified(Path(name), pin)
            if name.endswith('pre-outcome-manifest-v2.json'):
                original.update({r['race_key']: 'closed114' for r in json.loads(payload)['candidates']})
            elif name.endswith('out_of_time_races.csv'):
                original.update({key(r['race_id']): 'form_only' for r in csv.DictReader(payload.decode().splitlines())})
            elif name.endswith('frozen_august_odds_cohort.jsonl'):
                original.update({key(scalar(line, 'race_id')): 'august' for line in payload.decode().splitlines()})
            # Canonical market files are byte-hashed only; no row decoding.
        if set(original) != set(deny) or min(k[:10] for k in deny) <= '2026-07-09':
            raise ValueError('reservation union changed or overlaps earlier histories')
        if not all(key(rid) in deny for rid in incident['identities']):
            raise ValueError('incident identity absent from deny union')
        verified(Path(form.canonical.__file__), form.CANONICAL_BUILDER_SHA256)
        rows = identities(verified(FOUNDATION/'development.jsonl', PINS['development.jsonl']), deny, incident['identities'])
        grouped = collections.defaultdict(list)
        for row in rows:
            grouped[row['race_id']].append(row)
        if len(grouped) != 331 or len(rows) != 2360:
            raise ValueError('development population changed')
        provenance = json.loads(verified(FOUNDATION/'form_provenance.json', PINS['form_provenance.json']))
        sources = {s['race_id']: s for s in provenance['sources']}
        sample = freeze_sample(grouped)
        protocol = ROOT/'docs/research/early_speed_20260928_protocol.md'
        write(out/'audit_sample.json', {'rule': 'minimum salted SHA256 per literal target venue', 'race_ids': sample,
                                      'protocol_sha256': digest(protocol.read_bytes()), 'before_card_reads': True})
        write(out/'eligible_identities.json', rows)
        event('SAMPLE_FROZEN', races=len(sample), population_races=331, population_runners=2360,
              protocol_sha256=digest(protocol.read_bytes()), sample_sha256=digest((out/'audit_sample.json').read_bytes()))
        totals = collections.Counter()
        fields = {f: collections.Counter() for f in ('1 SEC', 'PIR', 'TIME', 'PLC')}
        hist_headers = collections.Counter()
        runner_counts = collections.Counter()
        race_counts = collections.Counter()
        by_venue = collections.defaultdict(collections.Counter)
        by_date = collections.defaultdict(collections.Counter)
        by_distance = collections.defaultdict(collections.Counter)
        histories = []
        history_dates = []
        race_audits = []
        failures = []
        raw_exports = []
        # No new semantic source has been supplied; numeric coverage is an upper bound.
        for rid, rr in sorted(grouped.items()):
            source = sources[rid]
            try:
                metadata = json.loads(verified(Path(source['card_sidecar_path']), source['card_sidecar_sha256']))
                capture = form.canonical.capture_timestamp(metadata, require_timezone=True)
                jump = form.canonical.sidecar_jump_timestamp(metadata, rid)
                if jump is None or (jump-capture).total_seconds() < 3600:
                    raise ValueError('source T-60 timing failed')
                if metadata.get('metadata_is_leakage_safe') is not True or metadata['runner_completeness']['status'] != 'COMPLETE':
                    raise ValueError('unsafe/incomplete sidecar')
                card = verified(Path(source['card_source_path']), source['card_source_sha256'])
                if metadata['content_sha256'] != digest(card) or metadata['content_length'] != len(card):
                    raise ValueError('source card binding')
                roster = sorted((r['box'], r['dog_token']) for r in rr)
                if roster != form.canonical.sidecar_roster(metadata, source=rid) or roster != form.canonical.parse_card_target_roster_bytes(card, source=rid):
                    raise ValueError('target roster differs from pre-race evidence')
                venue, _, _, _ = form.canonical.target_metadata({'metadata': metadata}, rid)
                distance = form._metres(metadata.get('target_distance') or metadata.get('race_info', {}).get('distance'))
                blocks = form.canonical.parse_form_blocks_bytes(card, source=rid)
                if rid in sample:
                    raw_exports.append({'race_id': rid, **retained_raw_header(metadata)})
                day = rr[0]['race_date']
                race_count = collections.Counter()
                runner_audits = []
                for runner in sorted(rr, key=lambda r: r['box']):
                    raw = blocks[runner['dog_token']]
                    # No historical value is admitted unless its date is independently safe.
                    for h in raw:
                        if not h['DATE'] < min(day, capture.date().isoformat(), min(k[:10] for k in deny)):
                            raise ValueError('history is not strictly before capture/target/protection')
                        date.fromisoformat(h['DATE'])
                    accepted, rejected = form.canonical.accepted_history(raw, date.fromisoformat(day))
                    if rejected or len(accepted) != len(raw):
                        raise ValueError('canonical history membership ambiguous')
                    rc = collections.Counter()
                    for h in raw:
                        hist_headers.update(h.keys())
                        history_dates.append(h['DATE'])
                        totals['history_observations'] += 1
                        same = venue not in AMBIGUOUS and form.canonical.canonical_venue(h.get('TRACK')) == venue and form._metres(h.get('DIST')) == distance
                        for field in fields:
                            state, value = parse_number(h.get(field))
                            fields[field][state] += 1
                            if field == '1 SEC' and state == 'positive':
                                rc['numeric_section'] += 1
                                if same:
                                    rc['same_layout_distance_numeric_section'] += 1
                        pir = str(h.get('PIR') or '').strip()
                        if re.fullmatch('[1-8]', pir):
                            totals['single_digit_pir'] += 1
                            if form.canonical.safe_float(pir) == form.canonical.safe_float(h.get('PLC')):
                                totals['single_digit_pir_equals_finish'] += 1
                        elif re.fullmatch('[1-8]{2,}', pir):
                            totals['multi_digit_1_to_8_pir'] += 1
                        elif pir:
                            totals['other_pir'] += 1
                        else:
                            totals['blank_pir'] += 1
                        if rid in sample:
                            histories.append({'target_race': rid, 'target_box': runner['box'], 'dog_token': runner['dog_token'],
                                              'target_venue': venue, 'target_distance': distance, 'card_capture': capture.isoformat(),
                                              'raw': {f: h.get(f) for f in ('DATE', 'TRACK', 'DIST', 'BOX', '1 SEC', 'PIR', 'PLC', 'TIME')},
                                              'same_unambiguous_layout_distance': same, 'semantic_early_call_qualified': False})
                    for metric in ('numeric_section', 'same_layout_distance_numeric_section'):
                        for threshold in (1, 3):
                            name = f'{metric}_ge{threshold}'
                            if rc[metric] >= threshold:
                                runner_counts[name] += 1
                                race_count[name] += 1
                    ns = neighbours(runner['box'], [r['box'] for r in rr])
                    if any(n['gap'] > 1 for n in ns):
                        race_count['runners_with_vacancy_gap'] += 1
                        runner_counts['with_vacancy_gap'] += 1
                    runner_audits.append({'box': runner['box'], 'dog_token': runner['dog_token'], 'history_count': len(raw),
                                          **dict(rc), 'nearest_occupied': ns})
                record = {'race_id': rid, 'race_date': day, 'venue': venue, 'source_venue': rid.split(' - ')[1],
                          'distance': distance, 'runners': len(rr), 'capture': capture.isoformat(), 'jump': jump.isoformat(),
                          'unoccupied_boxes': sorted(set(range(1, 9)) - {r['box'] for r in rr}),
                          'roster_basis': 'hash-bound pre-race card and sidecar; no independently timestamped scratch ledger',
                          'runner_audit': runner_audits, 'semantic_qualified': False}
                for metric in ('numeric_section', 'same_layout_distance_numeric_section'):
                    for threshold in (1, 3):
                        name = f'{metric}_ge{threshold}'
                        record['complete_'+name] = race_count[name] == len(rr)
                        if record['complete_'+name]:
                            race_counts[name] += 1
                race_counts['with_unoccupied_boxes'] += bool(record['unoccupied_boxes'])
                race_counts['with_internal_vacancy_gap'] += race_count['runners_with_vacancy_gap'] > 0
                for dimension, label in ((by_venue, venue), (by_date, day), (by_distance, str(distance))):
                    dimension[label]['races'] += 1
                    dimension[label]['runners'] += len(rr)
                    dimension[label]['complete_numeric_ge1'] += record['complete_same_layout_distance_numeric_section_ge1']
                    dimension[label]['complete_numeric_ge3'] += record['complete_same_layout_distance_numeric_section_ge3']
                race_audits.append(record)
            except (ValueError, KeyError, OSError) as exc:
                failures.append({'race_id': rid, 'reason': str(exc)})
                event('CARD_ACCESS_PATH_STOPPED', race_id=rid, reason=str(exc))
        composition = {}
        for threshold in (1, 3):
            eligible = [r for r in race_audits if r[f'complete_same_layout_distance_numeric_section_ge{threshold}']]
            periods = {}
            for start, end in [('2026-06-10', '2026-06-23'), ('2026-06-24', '2026-06-30'),
                               ('2026-07-01', '2026-07-02'), ('2026-07-03', '2026-07-09')]:
                period = [r for r in eligible if start <= r['race_date'] <= end]
                periods[start+'..'+end] = {'races': len(period), 'runners': sum(r['runners'] for r in period),
                                          'dates': len({r['race_date'] for r in period})}
            composition[str(threshold)] = {'races': len(eligible), 'runners': sum(r['runners'] for r in eligible),
                                           'dates': len({r['race_date'] for r in eligible}), 'periods': periods,
                                           'venues': dict(collections.Counter(r['venue'] for r in eligible))}
        result = {'status': 'PARTIALLY_TESTABLE_MEASUREMENT_AUDIT_ONLY',
                  'population': {'races': len(grouped), 'runners': len(rows), 'dates': len({r['race_date'] for r in rows}),
                                 'evaluated_races_prior_study': sum(r['race_date'] >= '2026-06-24' for r in race_audits),
                                 'audited_races': len(race_audits), 'sample_races': len(sample)},
                  'totals': dict(totals), 'field_parse_states': fields, 'runner_numeric_coverage': runner_counts,
                  'complete_race_numeric_coverage': race_counts, 'by_venue': by_venue, 'by_date': by_date,
                  'numeric_subset_composition': composition,
                  'by_distance': by_distance, 'history_headers': hist_headers,
                  'history_date_range': [min(history_dates), max(history_dates)] if history_dates else [],
                  'qualified_races': 0, 'qualified_runners': 0, 'fits': 0, 'target_outcomes_decoded': 0,
                  'failed_card_paths': failures,
                  'semantic_gate': 'NO_INDEPENDENT_MEASUREMENT_POINT_UNITS_RUNNER_VS_LEADER_OR_STYLE_CONTRACT',
                  'interpretation': 'Positive numeric section coverage is an upper bound, not validated early speed. No model test performed.',
                  'protected_keys': len(deny), 'prior_incident_races': len(incident['identities']),
                  'reservation_review_status': review['status']}
        write(out/'coverage.json', result)
        write(out/'race_audit.json', race_audits)
        write(out/'fixed_sample_rows.json', histories)
        write(out/'retained_raw_exports.json', raw_exports)
        write(out/'input_identities.json', inputs)
        write(out/'source_identities.json', {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in
                                           (Path(__file__), Path(form.__file__), Path(form.canonical.__file__), protocol)})
        event('PRIMARY_NOT_RUN', reason=result['semantic_gate'], qualified_races=0, fits=0, variants=0)
        event('COMPLETE', fits=0, target_outcomes_decoded=0, coverage_sha256=digest((out/'coverage.json').read_bytes()))
        print(json.dumps({k: result[k] for k in ('status', 'population', 'totals', 'field_parse_states', 'runner_numeric_coverage',
                                               'complete_race_numeric_coverage', 'qualified_races', 'failed_card_paths')}, indent=2))
    except Exception as exc:
        event('FAILED', exception=type(exc).__name__, reason=str(exc), fits=0)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    audit(parser.parse_args().out)
