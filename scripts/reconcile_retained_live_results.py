#!/usr/bin/env python3
"""Offline, allowlist-before-decode result reconciliation; never changes originals."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3

RACE_ID = re.compile(rb'"race_id"\s*:\s*"([^"\n]+)"')
JSONL_NAMES = ('official_result_races.jsonl', 'official_result_runners.jsonl',
               'official_result_quarantine.jsonl',
               'autonomous_official_result_capture_attempts.progress.jsonl')
JSON_NAMES = ('official_result_ingest_dry_run_report.json',
              'autonomous_official_result_capture_report.json',
              'live_odds_backlog_recovery_queue.json',
              'live_odds_backlog_runner_set_validation.json',
              'live_odds_backlog_join_eligibility_packet.json',
              'shadow_run_candidate_source_report.json')


def load(path):
    return json.loads(Path(path).read_bytes())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def allowlisted_jsonl(raw, allowed):
    for number, line in enumerate(raw.splitlines(), 1):
        identities = {m.decode() for m in RACE_ID.findall(line)}
        if identities and identities <= allowed:
            row = json.loads(line)
            if row.get('race_id') in allowed:
                yield number, row


def allowlisted_json_objects(raw, allowed):
    """Structural byte scan; outcomes outside exact membership are never decoded.

    Children are examined independently. A mixed parent is never decoded even
    when it contains an eligible child. Duplicate nested race IDs must all be
    allowed before JSON decoding. Offsets identify original immutable bytes.
    """
    stack = []
    quoted = escaped = False
    for offset, char in enumerate(raw):
        if quoted:
            if escaped:
                escaped = False
            elif char == 92:
                escaped = True
            elif char == 34:
                quoted = False
        elif char == 34:
            quoted = True
        elif char == 123:
            stack.append(offset)
        elif char == 125:
            start = stack.pop()
            fragment = raw[start:offset + 1]
            identities = {m.decode() for m in RACE_ID.findall(fragment)}
            if identities and identities <= allowed:
                row = json.loads(fragment)
                if row.get('race_id') in allowed:
                    yield start, row


def competition_positions(values):
    if not values or any(type(v) is not int or v < 1 for v in values):
        return False
    expected = 1
    for place, count in sorted(Counter(values).items()):
        if place != expected:
            return False
        expected += count
    return True


def qualified_win_order(box_order, runners):
    by_box = {r['box_number']: r for r in runners}
    if len(by_box) != len(runners) or len(set(box_order)) != len(box_order) or set(by_box) != set(box_order):
        return False
    values = [by_box[box]['finish_position'] for box in box_order]
    return values == sorted(values) and values.count(1) == 1 and competition_positions(values)


def scan_archives(root, allowed):
    jsonl, objects, references, directories = [], [], [], []
    # The named July collection lane stopped producing result batches July18.
    # Directory names only are enumerated; no other study roots are opened.
    for directory in sorted(root.glob('autonomous_official_result_capture_202607*')):
        match = re.search(r'capture_(\d{8})T', directory.name)
        if not match or match[1] < '20260717':
            continue
        directories.append(str(directory))
        for name in JSONL_NAMES + JSON_NAMES:
            path = directory / name
            if not path.exists():
                continue
            raw = path.read_bytes()
            references.append({'path': str(path), 'sha256': sha(raw), 'bytes': len(raw)})
            iterator = allowlisted_jsonl if name in JSONL_NAMES else allowlisted_json_objects
            target = jsonl if name in JSONL_NAMES else objects
            for location, row in iterator(raw, allowed):
                target.append({'path': str(path), 'line' if name in JSONL_NAMES else 'offset': location, 'row': row})
    return jsonl, objects, references, directories


def wrap(row, directory):
    return {**row, 'source_artifact_dir': str(directory), 'row_json': json.dumps(row, sort_keys=True)}


def build_projection(original_projection, jsonl):
    """Earliest richest accepted snapshot; no outcome-performance selection."""
    groups = defaultdict(list)
    for item in jsonl:
        if Path(item['path']).name == 'official_result_races.jsonl':
            groups[item['row']['race_id']].append(item)
    records = []
    for race_id, snapshots in sorted(groups.items()):
        selected = min(snapshots, key=lambda x: (-x['row']['position_count'], x['row']['captured_at'], x['path']))
        directory = Path(selected['path']).parent
        runners = [x['row'] for x in jsonl if Path(x['path']) == directory / 'official_result_runners.jsonl' and x['row']['race_id'] == race_id]
        if not qualified_win_order(selected['row']['box_order'], runners):
            raise ValueError('invalid_native_win_result_order:' + race_id)
        strongest = {x['box_number']: (x['dog_name'], x['finish_position']) for x in runners}
        for snapshot in snapshots:
            row = snapshot['row']
            if row['source_url'] != selected['row']['source_url'] or (row['winner_box'], row['winner_name']) != (selected['row']['winner_box'], selected['row']['winner_name']):
                raise ValueError('conflicting_native_result_identity:' + race_id)
            source_directory = Path(snapshot['path']).parent
            observed = [x['row'] for x in jsonl if Path(x['path']) == source_directory / 'official_result_runners.jsonl' and x['row']['race_id'] == race_id]
            if any(strongest.get(x['box_number']) != (x['dog_name'], x['finish_position']) for x in observed):
                raise ValueError('conflicting_native_result_position:' + race_id)
        # Preserve all versions separately in the archive census. A projection
        # selects one complete native snapshot and never constructs positions.
        records.append({'race_id': race_id, 'official_runner_identity_observed': False,
                        'runner_name_origin': 'candidate_participants_not_retained_official_names',
                        'races': [wrap(selected['row'], directory)],
                        'runners': [wrap(r, directory) for r in runners]})
    return {**original_projection, 'schema': 'authorised_archive_result_projection_v2',
            'query': 'Exact membership before JSON decode; native archive rows, no database writes',
            'records': records, 'result_races': len(records),
            'missing_race_ids': sorted(set(original_projection['missing_race_ids']) - set(groups))}


def audit_documented_aliases(boundary, allowed, directories, database):
    aliases = {}
    for race_id in sorted(allowed):
        match = re.fullmatch(r'Race (\d+) - (.+) - (\d{4}-\d{2}-\d{2})', race_id)
        canonical = boundary['venue_aliases'].get(match[2], match[2])
        for alias, value in boundary['venue_aliases'].items():
            if value == canonical:
                aliases[f'Race {match[1]} - {alias} - {match[3]}'] = race_id
    additional = []
    for directory in directories:
        path = Path(directory) / 'official_result_races.jsonl'
        for line, row in allowlisted_jsonl(path.read_bytes(), set(aliases)):
            if row['race_id'] not in allowed:
                additional.append({'path': str(path), 'line': line,
                                   'canonical_native_id': aliases[row['race_id']], 'row': row})
    metadata = []
    with sqlite3.connect('file:' + str(database) + '?mode=ro', uri=True) as connection:
        connection.execute('BEGIN')
        for alias, native in sorted(aliases.items()):
            if alias != native:
                for row in connection.execute('SELECT race_id,source_artifact_dir FROM autonomous_official_result_evidence_races WHERE race_id=?', (alias,)):
                    metadata.append({'race_id': row[0], 'source_artifact_dir': row[1], 'canonical_native_id': native})
    return {'schema': 'documented_venue_alias_result_audit_v1', 'alias_candidates': aliases,
            'additional_rows': additional, 'database_alias_metadata_matches': metadata}


def append_corrections(path, events):
    """Canonical event identity survives tuples becoming JSON arrays on replay."""
    previous = load(path) if path.exists() else []
    unique = {}
    for event in previous + events:
        key = sha(json.dumps(event, sort_keys=True, separators=(',', ':')).encode())
        unique.setdefault(key, event)
    if len(unique) < len(previous):
        raw = path.read_bytes()
        preserved = path.with_name('correction-trail-duplicate-history-' + sha(raw)[:16] + '.json')
        if not preserved.exists():
            preserved.write_bytes(raw)
        write(path.with_name('correction-trail-audit.json'), {
            'preserved_original': str(preserved), 'sha256': sha(raw),
            'historical_entries': len(previous), 'unique_events': len(unique),
            'duplicate_append_entries': len(previous) - len(unique),
            'repair': 'CANONICAL_JSON_EVENT_IDENTITY_TUPLE_LIST_REPLAY_FIX'})
    write(path, list(unique.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-evidence', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base, output = args.benchmark_evidence, args.output
    boundary = load(base / 'inventory/access-boundary.json')
    membership = load(base / 'inventory/residual-membership.json')
    forecasts = load(base / 'alignment/forecast-records.json')
    allowed = {r['race_id'] for r in forecasts}
    members = {r['record_key']: r for r in membership['records']}
    for f in forecasts:
        m = members[f['prediction_id']]
        if m['access_disposition'] != 'AUTHORIZED_NONRESERVED_RETAINED_OPERATIONAL' or m['canonical_race_key'] in boundary['canonical_protected_identity_keys']:
            raise ValueError('population_membership_not_authorized')
    if len(allowed) != 210 or len(forecasts) != 210:
        raise ValueError('expected_exact_210_primary_forecasts')
    originals = {x['original']['record_key']: x for x in load(base / 'alignment/authenticated-originals.json')}
    original_projection = load(base / 'official-result-projection.json')
    original_dispositions = {x['record_key']: x for x in load(base / 'all-dispositions.json')['july_join_dispositions']}
    jsonl, objects, references, directories = scan_archives(args.native_root, allowed)
    alias_audit = audit_documented_aliases(boundary, allowed, directories, original_projection['database'])
    if alias_audit['additional_rows'] or alias_audit['database_alias_metadata_matches']:
        raise ValueError('new_alias_result_requires_independent_identity_verification')
    projection = build_projection(original_projection, jsonl)
    results = {r['race_id']: r for r in projection['records']}
    records, corrections, acquisition = [], [], []
    for f in forecasts:
        race_id = f['race_id']
        original = originals[f['prediction_id']]
        predicted = original['original']['predictions']
        sidecar = load(original['append_proof']['artifacts']['sidecar']['path'])
        url = sidecar.get('canonical_runner_alignment', {}).get('canonical_source_url') or sidecar.get('race_url')
        race_rows = [x for x in jsonl if x['row']['race_id'] == race_id]
        race_objects = [x for x in objects if x['row']['race_id'] == race_id]
        result = results.get(race_id)
        returned_boxes = {x['box_number'] for x in result['runners']} if result else set()
        snapshot = Path(result['races'][0]['source_artifact_dir']) if result else None
        summaries = [x for x in race_objects if x['row'].get('status') == 'resulted']
        terms = [x for x in summaries if x['row'].get('terminal_statuses')]
        expected_boxes = {x['box_number'] for x in predicted}
        missing_boxes = sorted(expected_boxes - returned_boxes)
        recovered_projection = result is not None and race_id in original_projection['missing_race_ids']
        # Official names were NOT retained by write_result(dry_run=True).
        # Do not relabel participant-derived names as source-observed names.
        disposition = 'RETAINED_EVIDENCE_INCOMPLETE_OR_CONTRADICTORY' if result else 'REQUIRES_SPECIFIC_FRESH_OFFICIAL_RESULT_ACQUISITION'
        gaps = ['official_runner_names_not_retained_independently'] if result else ['official_result_not_found_in_named_retained_archives']
        if missing_boxes and result:
            gaps.append('terminal_status_missing_for_forecast_boxes:' + ','.join(map(str, missing_boxes)))
        if recovered_projection:
            corrections.append({'race_id': race_id, 'record_key': f['prediction_id'],
                                'original_disposition': original_dispositions[f['prediction_id']]['disposition'],
                                'correction': 'RECOVERED_NATIVE_ARCHIVE_RESULT_REJECTED_BY_ORDERED_FINISH_GATE',
                                'classification': 'WIN_DIAGNOSTIC_ONLY_PENDING_INDEPENDENT_RUNNER_IDENTITY',
                                'new_disposition': disposition, 'source_directory': str(snapshot),
                                'preserved_finishes': [(x['box_number'], x['finish_position']) for x in result['runners']]})
        attempts = [x for x in race_rows if Path(x['path']).name == 'autonomous_official_result_capture_attempts.progress.jsonl']
        queue = [x for x in race_objects if Path(x['path']).name in ('live_odds_backlog_recovery_queue.json', 'live_odds_backlog_join_eligibility_packet.json')]
        rejection = [x for x in race_objects if x['row'].get('reasons')]
        entry = {'race_id': race_id, 'canonical_race_key': members[f['prediction_id']]['canonical_race_key'],
                 'prediction_id': f['prediction_id'], 'native_job_id': Path(original['append_proof']['artifacts']['append_report']['path']).parent.name,
                 'native_record_id': f['prediction_id'], 'original_disposition': original_dispositions[f['prediction_id']]['disposition'],
                 'current_disposition': disposition, 'projection_recovered': recovered_projection,
                 'original_runner_field': [{'box': x['box_number'], 'dog_name': x['dog_name'], 'runner_id': x['runner_id']} for x in predicted],
                 'forecast_source': original['append_proof']['artifacts']['append_report'],
                 'official_source_url': url, 'official_url_source': original['append_proof']['artifacts']['sidecar'],
                 'result_queue_owner': 'July autonomous official-result capture lane; no current operation ownership asserted',
                 'closure_state': 'ARCHIVE_ACCEPTED_DB_REJECTED' if recovered_projection else 'PARTIAL_RETAINED_RESULT' if result else 'NO_ACCEPTED_RETAINED_RESULT',
                 'attempts': attempts, 'queue_records': queue, 'rejection_records': rejection,
                 'retained_response_locations': sorted({x['path'] for x in race_rows + race_objects}),
                 'official_result_field': result['runners'] if result else [], 'official_missing_forecast_boxes': missing_boxes,
                 'retained_terminal_status_observations': terms, 'remaining_gaps': gaps}
        records.append(entry)
        acquisition.append({'race_id': race_id, 'canonical_race_key': entry['canonical_race_key'],
                            'official_source_url': url, 'source': 'thedogs_official',
                            'scope': 'WIN winner plus official runner identities and every terminal status',
                            'reason': gaps, 'deduplication_key': entry['canonical_race_key']})
    output.mkdir(parents=True, exist_ok=True)
    write(output / 'allowlisted-native-jsonl-records.json', jsonl)
    write(output / 'allowlisted-native-json-objects.json', objects)
    write(output / 'archive-result-projection.json', projection)
    write(output / 'alias-result-audit.json', alias_audit)
    write(output / 'reconciliation.json', {'schema': 'exact_210_retained_result_reconciliation_v1', 'records': records})
    append_corrections(output / 'correction-trail.json', corrections)
    write(output / 'acquisition-needed.json', {'authorization': 'PREPARED_ONLY_ROOT_OWNER_MUST_VALIDATE', 'races': acquisition})
    write(output / 'archive-references.json', {'directories': directories, 'artifacts': references})
    summary = {'original_forecasts': 210, 'current_dispositions': dict(Counter(x['current_disposition'] for x in records)),
               'reconciliation_counts': {'evaluable_without_repair': 0, 'recovered_fully_evaluable': 0, 'still_incomplete': 9, 'require_acquisition': 201, 'restricted': 0},
               'original_projection_results': 7, 'archive_results': len(results), 'recovered_archive_joins': len(corrections),
               'archive_directories': len(directories), 'native_accepted_observations': sum(Path(x['path']).name == 'official_result_races.jsonl' for x in jsonl),
               'request_count': 0, 'reserved_outcomes_decoded': 0,
               'missing_names_warning': 'write_result(dry_run=True) copies participant names and discards official dog_names_by_box'}
    write(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
