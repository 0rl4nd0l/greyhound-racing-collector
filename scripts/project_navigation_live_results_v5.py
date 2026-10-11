#!/usr/bin/env python3
"""Offline exact active-route race identity; earlier readers and policies unchanged."""
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

def structured_name_rows(soup, rows):
    """Exact V4 extraction block; identity is independently verified by V5."""
    elements = soup.select('table.race-runners--result tr.race-runner')
    witnesses = []
    for row, element in zip(rows, elements):
        cells = element.select('td.race-runners__name')
        containers = element.select('td.race-runners__name .race-runners__name__dog')
        if len(cells) != 1 or len(containers) != 1:
            raise ValueError('structured_dog_container_missing_or_ambiguous')
        anchors = containers[0].select('a')
        if len(anchors) != 1:
            raise ValueError('structured_dog_anchor_missing_or_ambiguous')
        anchor = anchors[0]
        href = anchor.get('href', '')
        parsed = urlparse(href)
        relative = not parsed.scheme and not parsed.netloc and href.startswith('/dogs/')
        absolute = parsed.scheme == 'https' and parsed.netloc in {'www.thedogs.com.au', 'thedogs.com.au'}
        if not (relative or absolute) or parsed.query or parsed.fragment:
            raise ValueError('structured_dog_profile_foreign_or_malformed')
        match = re.fullmatch(r'/dogs/([1-9][0-9]*)/([^/\s]+)', parsed.path)
        if not match:
            raise ValueError('structured_dog_profile_foreign_or_malformed')
        dog_id = match[1]
        blackbooks = cells[0].select('blackbook-dog[data-dog-id]')
        ids = [x.get('data-dog-id') for x in blackbooks]
        if ids and (len(ids) != 1 or ids[0] != dog_id):
            raise ValueError('structured_dog_blackbook_identity_mismatch')
        name = anchor.get_text(' ', strip=True)
        if not name:
            raise ValueError('structured_dog_anchor_name_missing')
        rugs = element.select('td.race-runners__box [name]')
        rug_ids = [int(x['name'].split('_')[1]) for x in rugs if re.fullmatch(r'rug_[0-9]{1,2}', x['name'])]
        if rug_ids != [row['box_number']]:
            raise ValueError('structured_dog_source_box_missing_or_ambiguous')
        notes = containers[0].select('.race-runners__name__box')
        if len(notes) > 1:
            raise ValueError('structured_dog_box_note_ambiguous')
        note = notes[0].get_text(' ', strip=True) if notes else ''
        if note and not re.fullmatch(r'\(from box [1-8]\)', note, flags=re.I):
            raise ValueError('structured_dog_box_note_unsupported')
        time_nodes = containers[0].select('.race-runners__name__time')
        if len(time_nodes) > 1:
            raise ValueError('structured_dog_time_annotation_ambiguous')
        witness = {'source_box': row['box_number'], 'source_dog_href': href, 'source_dog_id': dog_id,
                   'source_blackbook_ids': ids, 'source_anchor_name': name,
                   'source_box_note': note, 'original_flattened_name': cells[0].get_text(' ', strip=True),
                   'original_parser_name': row.get('dog_name'),
                   'source_name_cell_html': str(cells[0]),
                   'separate_time_annotations': [x.get_text(' ', strip=True) for x in time_nodes]}
        witnesses.append(witness)
        row['dog_name'] = name + (' ' + note if note else '')
    return rows, witnesses


def parse_official_html(raw, expected_url, original, title_policy, structural_identity_policy_reference, policy_reference):
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
    race_href = exact_route(race.get('href', ''), expected_url)
    race_labels = race.select('.race-box__number')
    if race_labels and (len(race_labels) != 1 or race_labels[0].get_text(strip=True) != 'R' + str(expected[2])):
        raise ValueError('source_navigation_active_race_number_mismatch')
    results = unique('.page__layout .tabs-wrapper a.tabs__item--active')
    if results.get_text(' ', strip=True) != 'RESULTS':
        raise ValueError('source_navigation_active_tab_not_results')
    results_href = exact_route(results.get('href', ''), expected_url)
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
        if match[1] in mapping['proposed_exact_title_venues'] and (mapping['native_venue'], mapping['official_url_slug']) != (native[2], expected[0]):
            raise ValueError('source_navigation_known_title_venue_contradiction')
    tables = soup.select('table.race-runners--result')
    if len(tables) != 1:
        raise ValueError('official_result_table_missing_or_ambiguous')
    rows = parse_thedogs_result_html_runner_rows(str(tables[0]))
    if not rows or len(rows) != len(tables[0].select('tr.race-runner')):
        raise ValueError('official_result_runner_row_not_parsed')
    identity = {'identity_semantics': 'exact_active_official_race_navigation_v5',
                'source_declared_identities': [], 'source_title': title,
                'source_title_identity': [expected[0], title_date, int(match[5])],
                'source_title_native_venue': native[2],
                'active_race_href': race_href, 'active_results_href': results_href,
                'active_meeting_href': meeting_href, 'active_race_number': expected[2],
                'structural_identity_policy_reference': structural_identity_policy_reference,
                'projection_policy_reference': policy_reference}
    return rows, identity


def project_one(item, original, expected_url, directory, structural_identity_policy_reference, title_policy, policy_reference, runner_name_policy_reference=None, name_repair_trail=None):
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
    native_rows, identity = parse_official_html(raw, expected_url, original, title_policy, structural_identity_policy_reference, policy_reference)
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
    provenance = {'reused_base_projector_path': str(Path(base.__file__).resolve()),
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

