#!/usr/bin/env python3
"""Project explicitly authorized retained HTML responses; never fetch or alter forecasts."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, urlparse

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ingest_results_for_date import (  # noqa: E402
    parse_thedogs_result_html_runner_rows, remap_promoted_reserve_runner_rows,
)

NONFINISH = {'DNF', 'FELL', 'DISQ'}
SCRATCH = {'SCR', 'L/SCR', 'LSCR'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def write_bytes_once(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('derived_artifact_already_exists_with_different_bytes:' + str(path))
        return
    with path.open('xb') as stream:
        stream.write(raw)


def write(path, value):
    write_bytes_once(path, (json.dumps(value, indent=2, sort_keys=True) + '\n').encode())


def token(value):
    return re.sub('[^A-Z0-9]', '', str(value or '').upper())


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('naive_acquisition_timestamp')
    return result


def official_url_identity(url):
    parsed = urlparse(url)
    if any(value.casefold() != 'false' for value in parse_qs(parsed.query).get('trial', [])):
        raise ValueError('official_url_trial_scope_mismatch')
    match = re.match(r'^/racing/([^/]+)/(\d{4}-\d{2}-\d{2})/(\d+)(?:/|$)', parsed.path)
    if parsed.scheme != 'https' or parsed.hostname not in {'www.thedogs.com.au', 'thedogs.com.au'} or not match:
        raise ValueError('invalid_official_race_url')
    # The race-title/results suffix and trial=false query are navigation, not
    # identity. The venue slug, source date and official race number are exact.
    return match[1], match[2], int(match[3])


def parse_official_html(raw, expected_url):
    from bs4 import BeautifulSoup
    markup = raw.decode('utf-8', errors='strict')
    soup = BeautifulSoup(markup, 'html.parser')
    identities = []
    for element in soup.select('link[rel="canonical"], meta[property="og:url"]'):
        value = element.get('href') or element.get('content')
        if value:
            identities.append(official_url_identity(value))
    expected = official_url_identity(expected_url)
    if not identities or any(identity != expected for identity in identities):
        raise ValueError('source_declared_race_identity_missing_or_mismatched')
    tables = soup.select('table.race-runners--result')
    if len(tables) != 1:
        raise ValueError('official_result_table_missing_or_ambiguous')
    rows = parse_thedogs_result_html_runner_rows(str(tables[0]))
    if not rows:
        raise ValueError('official_result_rows_missing')
    if len(rows) != len(tables[0].select('tr.race-runner')):
        raise ValueError('official_result_runner_row_not_parsed')
    return rows, {'source_declared_identities': [list(x) for x in identities],
                  'identity_semantics': 'exact_official_venue_slug_date_race_number'}


def validate_official_field(original, native_rows):
    participants = [{'box_number': r['box_number'], 'dog_name': r['dog_name']}
                    for r in original['predictions']]
    mapping = remap_promoted_reserve_runner_rows(native_rows, participants)
    if mapping['rejected_remappings']:
        raise ValueError('official_reserve_mapping_rejected')
    rows = mapping['rows']
    if len({r['box_number'] for r in rows}) != len(rows):
        raise ValueError('duplicate_official_box')
    original_by_box = {r['box_number']: r for r in original['predictions']}
    active, reserves = [], []
    for row in rows:
        box, name, status = row['box_number'], row.get('dog_name'), row.get('status')
        if not name:
            raise ValueError('official_runner_name_missing')
        finish = row.get('finish_position')
        if box not in original_by_box:
            if status in SCRATCH and finish is None and box in {9, 10}:
                reserves.append(row)
                continue
            raise ValueError('official_runner_outside_forecast_field')
        if token(name) != token(original_by_box[box]['dog_name']):
            raise ValueError('official_runner_name_changed_at_box:' + str(box))
        if status in SCRATCH:
            raise ValueError('forecast_runner_scratched:' + str(box))
        if finish is None and status not in NONFINISH:
            raise ValueError('official_runner_terminal_status_missing:' + str(box))
        if finish is not None and (type(finish) is not int or finish < 1 or status):
            raise ValueError('contradictory_official_finish_status')
        active.append(row)
    if {r['box_number'] for r in active} != set(original_by_box):
        raise ValueError('official_field_incomplete')
    positions = [r['finish_position'] for r in active if r['finish_position'] is not None]
    if positions.count(1) != 1:
        raise ValueError('official_win_target_ambiguous')
    expected = 1
    for place, count in sorted(Counter(positions).items()):
        if place != expected:
            raise ValueError('official_positions_not_competition_ranked')
        expected += count
    return active, reserves, mapping


def project_one(item, original, expected_url, directory):
    metadata_path, response_path = Path(item['metadata_path']), Path(item['response_path'])
    metadata_raw = metadata_path.read_bytes()
    metadata = json.loads(metadata_raw)
    expected_identity = official_url_identity(expected_url)
    if metadata.get('http_status') != 200:
        raise ValueError('response_not_http_200')
    if metadata.get('requested_url') != expected_url or metadata.get('final_url') != metadata.get('requested_url'):
        raise ValueError('response_url_not_exact_authorized_request_or_redirect')
    for key in ('requested_url', 'final_url'):
        if official_url_identity(metadata[key]) != expected_identity:
            raise ValueError('response_route_identity_mismatch')
    acquired = timestamp(metadata['acquired_at'])
    if acquired <= timestamp(original['jump_timestamp']):
        raise ValueError('result_response_not_after_jump')
    raw = response_path.read_bytes()
    if digest(raw) != metadata['response_sha256']:
        raise ValueError('response_hash_mismatch')
    native_rows, identity = parse_official_html(raw, expected_url)
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / 'parsed_official_response.json', {'native_official_rows': native_rows,
          'source_identity': identity, 'response_path': str(response_path),
          'response_sha256': digest(raw), 'metadata_path': str(metadata_path),
          'metadata_sha256': digest(metadata_raw)})
    active, reserves, mapping = validate_official_field(original, native_rows)
    finished = sorted((r for r in active if r['finish_position'] is not None),
                      key=lambda r: (r['finish_position'], r['box_number']))
    winner = next(r for r in active if r['finish_position'] == 1)
    race_id = original['race_id']
    native = re.fullmatch(r'Race (\d+) - (.+) - (\d{4}-\d{2}-\d{2})', race_id)
    if expected_identity[1:] != (native[3], int(native[1])):
        raise ValueError('source_url_native_race_identity_mismatch')
    imported_parser_path = Path(sys.modules[parse_thedogs_result_html_runner_rows.__module__].__file__).resolve()
    provenance = {'imported_parser_path': str(imported_parser_path),
                  'imported_parser_sha256': digest(imported_parser_path.read_bytes()),
                  'response_path': str(response_path), 'response_sha256': digest(raw),
                  'metadata_path': str(metadata_path), 'metadata_sha256': digest(metadata_raw),
                  'parser_path': str(Path(__file__).resolve()),
                  'parser_sha256': digest(Path(__file__).read_bytes())}
    race = {'schema_version': 'acquired_official_win_race_v1', 'race_id': race_id,
            'race_date': native[3], 'race_number': int(native[1]), 'venue': native[2],
            'source': 'thedogs_official', 'source_url': metadata['final_url'],
            'captured_at': metadata['acquired_at'], 'status': 'resulted',
            'winner_box': winner['box_number'], 'winner_name': winner['dog_name'],
            'box_order': [r['box_number'] for r in finished],
            'position_count': len(finished), 'participant_count': len(active),
            'participant_source': 'official_response_html',
            'official_terminal_statuses': {str(r['box_number']): r['status'] for r in active if r['status']},
            'native_official_rows': native_rows, 'unused_scratched_reserves': reserves,
            'reserve_mapping': mapping, 'source_identity': identity, 'response_provenance': provenance}
    runners = [{'schema_version': 'acquired_official_win_runner_v1', 'race_id': race_id,
                'source': 'thedogs_official', 'source_url': metadata['final_url'],
                'captured_at': metadata['acquired_at'], 'box_number': r['box_number'],
                'dog_name': r['dog_name'], 'finish_position': r['finish_position'],
                'terminal_status': r['status'] or 'FINISHED',
                'is_winner': int(r['finish_position'] == 1),
                'original_source_box': r.get('original_box_number', r['box_number']),
                'official_runner_identity_observed': True} for r in active]
    directory.mkdir(parents=True, exist_ok=True)
    write_bytes_once(directory / 'official_result_races.jsonl', (json.dumps(race, sort_keys=True) + '\n').encode())
    write_bytes_once(directory / 'official_result_runners.jsonl', ''.join(json.dumps(r, sort_keys=True) + '\n' for r in runners).encode())
    def wrap(row):
        return {**row, 'source_artifact_dir': str(directory), 'row_json': json.dumps(row, sort_keys=True)}
    return {'race_id': race_id, 'official_runner_identity_observed': True,
            'runner_name_origin': 'SOURCE_NATIVE_OFFICIAL_HTML',
            'response_provenance': provenance,
            'races': [wrap(race)], 'runners': [wrap(r) for r in runners]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acquisition-manifest', type=Path, required=True)
    parser.add_argument('--benchmark-evidence', type=Path, required=True)
    parser.add_argument('--reconciliation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.benchmark_evidence
    # Membership/access are decoded BEFORE either result manifest or response.
    boundary = load(base / 'inventory/access-boundary.json')
    membership = load(base / 'inventory/residual-membership.json')
    forecasts = load(base / 'alignment/forecast-records.json')
    by_key = {r['record_key']: r for r in membership['records']}
    allowed = {}
    for forecast in forecasts:
        member = by_key[forecast['prediction_id']]
        if member['access_disposition'] != 'AUTHORIZED_NONRESERVED_RETAINED_OPERATIONAL' or member['canonical_race_key'] in boundary['canonical_protected_identity_keys']:
            raise ValueError('population_member_not_authorized')
        allowed[forecast['race_id']] = forecast['prediction_id']
    if len(allowed) != 210:
        raise ValueError('expected_exact_210_population')
    originals = {r['original']['record_key']: r['original'] for r in load(base / 'alignment/authenticated-originals.json')}
    expected = {r['race_id']: r['official_source_url'] for r in load(args.reconciliation)['records']}
    manifest = load(args.acquisition_manifest)
    items = manifest if isinstance(manifest, list) else manifest['records']
    if any(item['race_id'] not in allowed for item in items):
        raise ValueError('response_manifest_outside_authorized_population')
    if len({item['race_id'] for item in items}) != len(items):
        raise ValueError('duplicate_official_race_response')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('output_directory_must_be_new_or_empty')
    args.output.mkdir(parents=True, exist_ok=True)
    records, dispositions = [], []
    for item in items:
        rid = item['race_id']
        try:
            projection = project_one(item, originals[allowed[rid]], expected[rid], args.output / 'native' / allowed[rid])
        except (KeyError, ValueError, OSError) as error:
            dispositions.append({'race_id': rid, 'disposition': 'RESPONSE_NOT_QUALIFIED', 'reason': str(error),
                                 'response_reference': item, 'parsed_response_path': str(args.output / 'native' / allowed[rid] / 'parsed_official_response.json')})
            continue
        records.append(projection)
        dispositions.append({'race_id': rid, 'disposition': 'EXACT_OFFICIAL_WIN_FIELD', 'response_reference': item})
    for rid in sorted(set(allowed) - {x['race_id'] for x in items}):
        dispositions.append({'race_id': rid, 'disposition': 'NOT_ACQUIRED'})
    projection = {'schema': 'authorized_acquired_official_result_projection_v1',
                  'membership_path': str(base / 'inventory/residual-membership.json'),
                  'membership_sha256': digest((base / 'inventory/residual-membership.json').read_bytes()),
                  'allowed_races': 210, 'result_races': len(records), 'records': records,
                  'missing_race_ids': sorted(set(allowed) - {r['race_id'] for r in records})}
    write(args.output / 'acquired-result-projection.json', projection)
    write(args.output / 'acquired-result-dispositions.json', dispositions)
    print(json.dumps({'qualified': len(records), 'dispositions': dict(Counter(r['disposition'] for r in dispositions))}))


if __name__ == '__main__':
    main()
