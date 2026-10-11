#!/usr/bin/env python3
"""Offline source-declared nontrial navigation; all prior readers unchanged."""
from urllib.parse import urlparse
from datetime import date
from pathlib import Path
import json
import re
import sys

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import project_acquired_live_results as base
from scripts import project_policy_live_results_v3 as title_projector
from scripts import project_structured_live_results_v4 as name_projector
from scripts.project_acquired_live_results import (
    digest, timestamp, official_url_identity, validate_official_field,
    parse_thedogs_result_html_runner_rows, write, write_bytes_once,
)

from scripts import project_navigation_live_results_v5 as navigation_projector
from scripts import project_prerace_field_live_results_v6 as field_projector
structured_name_rows = navigation_projector.structured_name_rows


def parse_official_html(raw, expected_url, original, title_policy, structural_identity_policy_reference, policy_reference, nontrial_navigation_policy_reference):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(raw.decode('utf-8', errors='strict'), 'html.parser')
    if soup.select('link[rel="canonical"], meta[property="og:url"]'):
        raise ValueError('source_navigation_declared_identity_must_be_absent')
    expected = official_url_identity(expected_url)
    native = re.fullmatch(r'Race (\d+) - (.+) - (\d{4}-\d{2}-\d{2})', original['race_id'])
    if not native or expected[1:] != (native[3], int(native[1])):
        raise ValueError('source_navigation_native_identity_mismatch')
    source = urlparse(expected_url)
    origin = source.scheme + '://' + source.netloc
    if (source.scheme != 'https' or source.netloc not in {'www.thedogs.com.au', 'thedogs.com.au'}
            or expected_url not in {origin + source.path, origin + source.path + '?trial=false'}):
        raise ValueError('source_nontrial_authorized_url_not_exact_supported_spelling')
    if not any((m['native_venue'], m['official_url_slug']) == (native[2], expected[0]) for m in title_policy['mappings']):
        raise ValueError('source_nontrial_native_slug_mapping_missing')
    nontrial_url = origin + source.path + '?trial=false'
    def exact_route(href, target):
        parsed = urlparse(href)
        if not parsed.scheme and not parsed.netloc and href.startswith('/racing/'):
            resolved = origin + href
        elif parsed.scheme == 'https' and parsed.netloc == source.netloc:
            resolved = href
        else:
            raise ValueError('source_navigation_href_foreign_or_malformed')
        if resolved != target:
            raise ValueError('source_navigation_href_not_exact_approved_route')
        return href
    def unique(selector):
        found = soup.select(selector)
        if len(found) != 1:
            raise ValueError('source_navigation_anchor_missing_or_ambiguous:' + selector)
        return found[0]
    race = unique('a.race-box--active')
    race_href = exact_route(race.get('href', ''), nontrial_url)
    race_labels = race.select('.race-box__number')
    if race_labels and (len(race_labels) != 1 or race_labels[0].get_text(strip=True) != 'R' + str(expected[2])):
        raise ValueError('source_navigation_active_race_number_mismatch')
    result_nodes = [n for n in soup.select('.page__layout .tabs-wrapper a.tabs__item')
                    if ' '.join(n.get_text(' ', strip=True).split()) == 'RESULTS']
    if len(result_nodes) != 1:
        raise ValueError('source_nontrial_results_anchor_missing_or_ambiguous')
    results = result_nodes[0]
    active_tabs = soup.select('.page__layout .tabs-wrapper a.tabs__item--active')
    if active_tabs and (len(active_tabs) != 1 or active_tabs[0] is not results):
        raise ValueError('source_nontrial_active_tab_contradicts_results')
    results_href = exact_route(results.get('href', ''), nontrial_url)
    meeting = unique('a.dropdown__menu__meeting--active')
    meeting_url = origin + '/racing/' + expected[0] + '/' + expected[1] + '?trial=false'
    meeting_href = exact_route(meeting.get('href', ''), meeting_url)
    titles = soup.find_all('title')
    if len(titles) != 1:
        raise ValueError('source_navigation_title_missing_or_ambiguous')
    title = titles[0].get_text().strip()
    match = re.fullmatch(r'thedogs - (.+) ([1-9][0-9]?) ([A-Za-z]+) ([0-9]{4}) Race ([1-9][0-9]*)', title)
    if not match or match[3] not in title_projector.MONTHS:
        raise ValueError('source_navigation_title_format_unsupported')
    try:
        title_date = date(int(match[4]), title_projector.MONTHS[match[3]], int(match[2])).isoformat()
    except ValueError as error:
        raise ValueError('source_navigation_title_date_invalid') from error
    if (title_date, int(match[5])) != expected[1:]:
        raise ValueError('source_navigation_title_date_or_number_mismatch')
    for mapping in title_policy['mappings']:
        if match[1].strip() in mapping['proposed_exact_title_venues'] and (mapping['native_venue'], mapping['official_url_slug']) != (native[2], expected[0]):
            raise ValueError('source_navigation_known_title_venue_contradiction')
    tables = soup.select('table.race-runners--result')
    if len(tables) != 1:
        raise ValueError('official_result_table_missing_or_ambiguous')
    rows = parse_thedogs_result_html_runner_rows(str(tables[0]))
    if not rows or len(rows) != len(tables[0].select('tr.race-runner')):
        raise ValueError('official_result_runner_row_not_parsed')
    identity = {'identity_semantics': 'exact_source_nontrial_navigation_v7',
                'source_declared_identities': [], 'source_title': title,
                'source_title_identity': [expected[0], title_date, int(match[5])],
                'source_title_native_venue': native[2],
                'active_race_href': race_href, 'results_href': results_href,
                'results_anchor_classes': results.get('class', []),
                'source_title_display_venue': match[1].strip(),
                'nontrial_navigation_policy_reference': nontrial_navigation_policy_reference,
                'active_meeting_href': meeting_href, 'active_race_number': expected[2],
                'structural_identity_policy_reference': structural_identity_policy_reference,
                'projection_policy_reference': policy_reference}
    return rows, identity


def project_one(item, wrapper, expected_url, directory, structural_identity_policy_reference, title_policy, policy_reference, nontrial_navigation_policy_reference, runner_name_policy_reference=None, name_repair_trail=None):
    original = wrapper["original"]
    pre_race_proof = field_projector.verify_prerace_field_proof(wrapper, expected_url)
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
    native_rows, identity = parse_official_html(raw, expected_url, original, title_policy, structural_identity_policy_reference, policy_reference, nontrial_navigation_policy_reference)
    identity["pre_race_identity_proof"] = pre_race_proof
    name_witnesses = []
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / 'parsed_official_response.json', {'native_official_rows': native_rows,
          'source_identity': identity, 'structured_runner_name_witnesses': name_witnesses, 'response_path': str(response_path),
          'response_sha256': digest(raw), 'metadata_path': str(metadata_path),
          'metadata_sha256': digest(metadata_raw)})
    try:
        active, reserves, mapping = validate_official_field(original, native_rows)
    except ValueError as error:
        if not str(error).startswith('official_runner_name_changed_at_box:') or runner_name_policy_reference is None:
            raise
        from bs4 import BeautifulSoup
        if name_repair_trail is None:
            name_repair_trail = {}
        name_repair_trail.update(status='ATTEMPTED', original_parser_failure=str(error), runner_name_policy_reference=runner_name_policy_reference)
        try:
            native_rows, name_witnesses = structured_name_rows(BeautifulSoup(raw.decode('utf-8', errors='strict'), 'html.parser'), native_rows)
            write(directory / 'structured_name_response.json', {'native_official_rows': native_rows,
                  'source_identity': identity, 'structured_runner_name_witnesses': name_witnesses,
                  'response_path': str(response_path), 'response_sha256': digest(raw)})
            active, reserves, mapping = validate_official_field(original, native_rows)
        except (ValueError, KeyError, OSError) as fallback_error:
            name_repair_trail.update(status='REJECTED', fallback_failure=str(fallback_error))
            raise
        name_repair_trail['status'] = 'QUALIFIED'

    finished = sorted((r for r in active if r['finish_position'] is not None),
                      key=lambda r: (r['finish_position'], r['box_number']))
    winner = next(r for r in active if r['finish_position'] == 1)
    race_id = original['race_id']
    native = re.fullmatch(r'Race (\d+) - (.+) - (\d{4}-\d{2}-\d{2})', race_id)
    if expected_identity[1:] != (native[3], int(native[1])):
        raise ValueError('source_url_native_race_identity_mismatch')
    imported_parser_path = Path(sys.modules[parse_thedogs_result_html_runner_rows.__module__].__file__).resolve()
    provenance = {'nontrial_navigation_policy_reference': nontrial_navigation_policy_reference,
                  'pre_race_identity_sidecar_reference': pre_race_proof['sidecar_reference'],
                  'reused_prerace_projector_path': str(Path(field_projector.__file__).resolve()),
                  'reused_prerace_projector_sha256': digest(Path(field_projector.__file__).read_bytes()),
                  'reused_navigation_projector_path': str(Path(navigation_projector.__file__).resolve()),
                  'reused_navigation_projector_sha256': digest(Path(navigation_projector.__file__).read_bytes()),
                  'reused_base_projector_path': str(Path(base.__file__).resolve()),
                  'reused_base_projector_sha256': digest(Path(base.__file__).read_bytes()),
                  'structural_identity_policy_reference': structural_identity_policy_reference,
                  'projection_policy_reference': policy_reference,
                  'reused_name_projector_path': str(Path(name_projector.__file__).resolve()),
                  'reused_name_projector_sha256': digest(Path(name_projector.__file__).read_bytes()),
                  'runner_name_policy_reference': runner_name_policy_reference if name_witnesses else None,
                  'reused_title_projector_path': str(Path(title_projector.__file__).resolve()),
                  'reused_title_projector_sha256': digest(Path(title_projector.__file__).read_bytes()),
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
            'native_official_rows': native_rows, 'structured_runner_name_witnesses': name_witnesses, 'unused_scratched_reserves': reserves,
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

