#!/usr/bin/env python3
"""Offline source-row reconciliation against independently sealed pre-race fields."""
import copy
import json
import re
import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import project_acquired_live_results as base
from scripts import project_policy_live_results_v3 as title_projector
from scripts import project_structured_live_results_v4 as name_projector
from scripts import project_navigation_live_results_v5 as navigation_projector
from scripts.project_acquired_live_results import digest, timestamp, official_url_identity, validate_official_field, parse_thedogs_result_html_runner_rows, write, write_bytes_once


def field_keys(rows):
    keys = [(r['box_number'], base.token(r['dog_name'])) for r in rows]
    if len(keys) != len(set(keys)) or len({k[0] for k in keys}) != len(keys):
        raise ValueError('pre_race_field_duplicate_identity')
    return set(keys)


def verify_prerace_field_proof(wrapper, expected_url):
    original = wrapper['original']
    reference = wrapper['append_proof']['artifacts']['sidecar']
    raw = Path(reference['path']).read_bytes()
    if (digest(raw) != reference['sha256']
            or wrapper['append_proof']['native_prediction']['input_hashes']['sidecar_sha256'] != reference['sha256']):
        raise ValueError('pre_race_sidecar_hash_mismatch')
    sidecar = json.loads(raw)
    alignment = sidecar['canonical_runner_alignment']
    pre = sidecar['prejump_shadow_metadata']
    if (pre.get('status') != 'PASS' or pre.get('metadata_is_leakage_safe') is not True
            or alignment.get('status') != 'aligned' or alignment.get('canonical_runner_set_status') != 'available'
            or pre.get('fail_reasons', []) or alignment.get('missing_canonical_participants', [])
            or alignment.get('duplicate_source_runner_names', [])
            or alignment.get('canonical_runner_count') != len(original['predictions'])
            or alignment.get('prediction_runner_count') != len(original['predictions'])):
        raise ValueError('pre_race_field_source_not_authenticated_complete')
    if pre['source_url'] != expected_url or alignment['canonical_source_url'] != expected_url:
        raise ValueError('pre_race_field_source_url_mismatch')
    if field_keys(pre['runner_box_name_list']) != field_keys(original['predictions']):
        raise ValueError('pre_race_field_not_original_forecast_field')
    win = wrapper['win_field']
    captured, scored, sealed, jumped = map(timestamp, (pre['metadata_captured_at'], original['score_timestamp'], wrapper['append_proof']['sealed_at'], original['jump_timestamp']))
    if (not captured <= scored <= sealed < jumped or win.get('status') != 'EXACT_COMPLETE_WIN_FIELD'
            or not timestamp(win['capture_at']) <= timestamp(win['available_at']) <= scored):
        raise ValueError('pre_race_field_chronology_or_complete_market_invalid')
    dropped = alignment['dropped_participants']
    if len([(r['box_number'], base.token(r['dog_name'])) for r in dropped]) != len({(r['box_number'], base.token(r['dog_name'])) for r in dropped}):
        raise ValueError('pre_race_dropped_identity_duplicate')
    return {'sidecar_reference': reference, 'pre_race_active_field': pre['runner_box_name_list'],
            'dropped_participants': dropped, 'remapped_participants': alignment['remapped_participants'],
            'metadata_captured_at': pre['metadata_captured_at'], 'forecast_scored_at': original['score_timestamp'],
            'forecast_sealed_at': wrapper['append_proof']['sealed_at'], 'complete_win_field': win}


def derive_original_field_rows(original, native_rows, proof):
    """No inferred result/field: every removal and promotion needs earlier proof."""
    original_keys = field_keys(original['predictions'])
    original_names = {key[1] for key in original_keys}
    dropped = {(r['box_number'], base.token(r['dog_name'])) for r in proof['dropped_participants']}
    active, excluded, remapped = [], [], []
    for native in native_rows:
        row = dict(native)
        box, name = row['box_number'], row['dog_name']
        key = (box, base.token(name))
        if row.get('status') in base.SCRATCH and row.get('finish_position') is None:
            if key in original_keys:
                raise ValueError('forecast_runner_scratched:' + str(box))
            if key in dropped:
                if key[1] in original_names:
                    raise ValueError('pre_race_exclusion_overlaps_original_dog_identity')
                excluded.append(dict(row))
                continue
        note = re.fullmatch(r'(.+?)\s+\(from box ([1-8])\)', name, flags=re.I)
        if note:
            target, clean_name = int(note[2]), note[1]
            matches = [r for r in proof['remapped_participants']
                       if r['source_box_number'] == box and r['original_box_number'] == box
                       and r['final_box_number'] == target and base.token(r['dog_name']) == base.token(clean_name)]
            if box not in {9, 10} or len(matches) != 1 or (target, base.token(clean_name)) not in original_keys:
                raise ValueError('pre_race_reserve_continuity_unproved')
            row.update(box_number=target, dog_name=clean_name, original_box_number=box,
                       reserve_box_remap_source='SEALED_PRE_RACE_AND_EXPLICIT_OFFICIAL_FROM_BOX')
            remapped.append({'source_box_number': box, 'final_box_number': target,
                             'dog_name': clean_name, 'source_name_with_box_note': name,
                             'pre_race_mapping': matches[0]})
        active.append(row)
    if len({r['box_number'] for r in active}) != len(active):
        raise ValueError('pre_race_derived_duplicate_box')
    return active, excluded, remapped


def parse_identity(raw, expected_url, original, title_policy, title_ref, structural_ref):
    try:
        return base.parse_official_html(raw, expected_url)
    except ValueError as error:
        if str(error) != 'source_declared_race_identity_missing_or_mismatched' or title_policy is None:
            raise
    try:
        return title_projector.parse_official_html(raw, expected_url, original, title_policy, title_ref)
    except ValueError as error:
        if str(error) != 'source_title_display_slug_native_not_exact_policy_member' or structural_ref is None:
            raise
    return navigation_projector.parse_official_html(raw, expected_url, original, title_policy, structural_ref, title_ref)


def project_one(item, wrapper, expected_url, directory, title_policy, policy_reference, runner_name_policy_reference, structural_identity_policy_reference, pre_race_field_policy_reference):
    original = wrapper['original']
    proof = verify_prerace_field_proof(wrapper, expected_url)
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
    raw_native_rows, identity = parse_identity(raw, expected_url, original, title_policy, policy_reference, structural_identity_policy_reference)
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(raw.decode('utf-8', errors='strict'), 'html.parser')
    # A field repair cannot bypass a malformed declaration ignored by older code.
    for declaration in soup.select('link[rel="canonical"], meta[property="og:url"]'):
        value = declaration.get('href') if declaration.name == 'link' else declaration.get('content')
        if not value or official_url_identity(value) != expected_identity:
            raise ValueError('source_declared_race_identity_empty_or_mismatched')
    native_rows, name_witnesses = navigation_projector.structured_name_rows(soup, copy.deepcopy(raw_native_rows))
    derived_rows, excluded, remapped = derive_original_field_rows(original, native_rows, proof)
    field_projection = {'sidecar_reference': proof['sidecar_reference'],
                        'pre_race_active_field': proof['pre_race_active_field'],
                        'excluded_source_rows': excluded, 'remapped_source_rows': remapped,
                        'derived_original_field_rows': derived_rows,
                        'pre_race_proof': proof}
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / 'parsed_official_response.json', {'native_official_rows': native_rows,
          'raw_native_official_rows': raw_native_rows, 'source_identity': identity,
          'structured_runner_name_witnesses': name_witnesses,
          'pre_race_field_projection': field_projection, 'response_path': str(response_path),
          'response_sha256': digest(raw), 'metadata_path': str(metadata_path),
          'metadata_sha256': digest(metadata_raw)})
    active, reserves, mapping = validate_official_field(original, derived_rows)
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
                  'structural_identity_policy_reference': structural_identity_policy_reference if identity['identity_semantics'] == 'exact_active_official_race_navigation_v5' else None,
                  'pre_race_field_policy_reference': pre_race_field_policy_reference,
                  'sidecar_reference': proof['sidecar_reference'],
                  'source_name_extraction_policy_reference': runner_name_policy_reference,
                  'reused_navigation_projector_path': str(Path(navigation_projector.__file__).resolve()),
                  'reused_navigation_projector_sha256': digest(Path(navigation_projector.__file__).read_bytes()),
                  'projection_policy_reference': policy_reference,
                  'reused_name_projector_path': str(Path(name_projector.__file__).resolve()),
                  'reused_name_projector_sha256': digest(Path(name_projector.__file__).read_bytes()),
                  'runner_name_policy_reference': None,
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
            'native_official_rows': native_rows, 'raw_native_official_rows': raw_native_rows,
            'pre_race_field_projection': field_projection, 'structured_runner_name_witnesses': name_witnesses, 'unused_scratched_reserves': reserves,
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

