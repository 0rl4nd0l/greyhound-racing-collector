#!/usr/bin/env python3
"""Offline READY-only exact title fallback bound to an externally frozen policy.

The canonical projector and reviewed-held-response V2 stay unchanged. This
reader preserves their field validation by importing the pinned base validator.
"""
from datetime import date
from pathlib import Path
import json
import re
import sys

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import project_acquired_live_results as base
from scripts.project_acquired_live_results import (
    digest, timestamp, official_url_identity, validate_official_field,
    parse_thedogs_result_html_runner_rows, write, write_bytes_once,
)

IDENTITY_RULE = 'frozen_exact_source_title_venue_date_race_v3'
MONTHS = {name: i for i, name in enumerate(('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'), 1)}


def parse_official_html(raw, expected_url, original, policy, policy_reference):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(raw.decode('utf-8', errors='strict'), 'html.parser')
    expected = official_url_identity(expected_url)
    identities = []
    for element in soup.select('link[rel="canonical"], meta[property="og:url"]'):
        value = element.get('href') if element.name == 'link' else element.get('content')
        if not value:
            raise ValueError('source_declared_race_identity_empty')
        identity = official_url_identity(value)
        if identity != expected:
            raise ValueError('source_declared_race_identity_missing_or_mismatched')
        identities.append(identity)
    titles = soup.find_all('title')
    if len(titles) != 1:
        raise ValueError('source_title_missing_or_ambiguous')
    title = titles[0].get_text().strip()
    match = re.fullmatch(r'thedogs - (.+) ([1-9][0-9]?) ([A-Za-z]+) ([0-9]{4}) Race ([1-9][0-9]*)', title)
    if not match or match[3] not in MONTHS:
        raise ValueError('source_title_identity_unsupported')
    native = re.fullmatch(r'Race (\d+) - (.+) - (\d{4}-\d{2}-\d{2})', original['race_id'])
    if not native:
        raise ValueError('source_title_native_race_identity_invalid')
    matching = [row for row in policy['mappings'] if row['native_venue'] == native[2]
                and row['official_url_slug'] == expected[0]
                and match[1] in row['proposed_exact_title_venues']]
    if len(matching) != 1:
        raise ValueError('source_title_display_slug_native_not_exact_policy_member')
    title_date = date(int(match[4]), MONTHS[match[3]], int(match[2])).isoformat()
    title_identity = (expected[0], title_date, int(match[5]))
    if title_identity != expected or expected[1:] != (native[3], int(native[1])):
        raise ValueError('source_title_date_or_race_number_mismatch')
    tables = soup.select('table.race-runners--result')
    if len(tables) != 1:
        raise ValueError('official_result_table_missing_or_ambiguous')
    rows = parse_thedogs_result_html_runner_rows(str(tables[0]))
    if not rows:
        raise ValueError('official_result_rows_missing')
    if len(rows) != len(tables[0].select('tr.race-runner')):
        raise ValueError('official_result_runner_row_not_parsed')
    return rows, {'source_declared_identities': [list(x) for x in identities],
                  'source_title': title, 'source_title_identity': list(title_identity),
                  'source_title_native_venue': native[2], 'identity_semantics': IDENTITY_RULE,
                  'projection_policy_reference': policy_reference}


def project_one(item, original, expected_url, directory, policy, policy_reference):
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
    native_rows, identity = parse_official_html(raw, expected_url, original, policy, policy_reference)
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
    provenance = {'reused_base_projector_path': str(Path(base.__file__).resolve()),
                  'reused_base_projector_sha256': digest(Path(base.__file__).read_bytes()),
                  'projection_policy_reference': policy_reference,
                  'imported_parser_path': str(imported_parser_path),
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

