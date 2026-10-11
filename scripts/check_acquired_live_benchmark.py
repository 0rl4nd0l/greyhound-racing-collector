#!/usr/bin/env python3
"""Independent acquired-result audit; no scorer, projector or acquisition imports."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import date, datetime
import json
from pathlib import Path
import re
from urllib.parse import urlsplit, parse_qs

try:
    from . import check_live_reconciliation as independent
except ImportError:
    import check_live_reconciliation as independent

read, digest = independent.read, independent.digest
V3_IDENTITY = 'frozen_exact_source_title_venue_date_race_v3'
V5_IDENTITY = 'exact_active_official_race_navigation_v5'
V7_IDENTITY = 'exact_source_nontrial_navigation_v7'
V7_FAILURES = {'source_navigation_href_not_exact_approved_route',
               'source_navigation_anchor_missing_or_ambiguous:.page__layout .tabs-wrapper a.tabs__item--active'}


def require(test, reason):
    if not test:
        raise ValueError(reason)


def load_projection_policy(path, expected_sha256, forecasts, expected_urls=None):
    """Authenticate an externally pinned policy and population before body reads."""
    path = Path(path)
    require(isinstance(expected_sha256, str) and re.fullmatch('[0-9a-f]{64}', expected_sha256)
            and digest(path) == expected_sha256, 'projection policy hash mismatch')
    policy = read(path)
    require(policy.get('schema') == 'frozen_offline_source_title_projection_policy_v3'
            and policy.get('status') == 'FROZEN_BEFORE_ADDITIONAL_RESULT_DECODING'
            and policy.get('field_policy_unchanged') is True
            and policy.get('source_hold_preserved') is True
            and policy.get('new_runtime_or_provider_authority') is False,
            'projection policy not frozen with unchanged scope')
    reference = policy['population_reference']
    require(digest(reference['path']) == reference['sha256'], 'projection population hash mismatch')
    population = read(reference['path'])['records']
    expected = {r['race_id']: r['prediction_id'] for r in forecasts}
    admitted = {r['race_id']: r['prediction_id'] for r in population}
    require(len(population) == len(admitted) == len(expected) == len(forecasts) == 210
            and admitted == expected and policy.get('population_races') == 210,
            'projection policy population differs from frozen forecasts')
    mappings = policy['mappings']
    native_races = [re.fullmatch(r'Race ([1-9][0-9]*) - (.+) - ([0-9]{4}-[0-9]{2}-[0-9]{2})', rid)
                    for rid in expected]
    require(all(native_races) and {match[2] for match in native_races}
            == {row['native_venue'] for row in mappings}, 'projection policy native mapping outside population')
    if expected_urls is not None:
        expected_pairs = {(match[2], url_identity(expected_urls[match[0]])[0]) for match in native_races}
        require(expected_pairs == {(row['native_venue'], row['official_url_slug']) for row in mappings},
                'projection policy exact native slug mapping outside population')
    triples = [(display, row['official_url_slug'], row['native_venue'])
               for row in mappings for display in row['proposed_exact_title_venues']]
    require(len({r['native_venue'] for r in mappings}) == len(mappings) == policy['native_venues']
            and len(triples) == len(set(triples)) == policy['exact_display_entries']
            and all(isinstance(value, str) and value and value == value.strip()
                    for triple in triples for value in triple),
            'projection policy mapping duplicate or malformed')
    return {'document': policy, 'reference': {'path': str(path.resolve()), 'sha256': expected_sha256},
            'population': admitted, 'triples': frozenset(triples)}


def load_runner_name_policy(path, expected_sha256, forecasts, projection_policy=None):
    """Authenticate the independently reviewed structural repair before bodies."""
    path = Path(path)
    require(isinstance(expected_sha256, str) and re.fullmatch('[0-9a-f]{64}', expected_sha256)
            and digest(path) == expected_sha256, 'runner name policy hash mismatch')
    policy = read(path)
    require(policy.get('schema') == 'frozen_offline_structured_dog_name_policy_v4'
            and policy.get('status') == 'FROZEN_BEFORE_CORRECTED_METRICS'
            and policy.get('field_and_scoring_policy_unchanged') is True
            and policy.get('newly_corrected_metrics_inspected') is False
            and policy.get('no_new_provider_or_runtime_authority') is True,
            'runner name policy not frozen with unchanged scope')
    terminal = policy.get('terminal_policy', {})
    require(terminal.get('accepted_vocabulary_unchanged') is True
            and terminal.get('no_runner_removal_or_renormalization') is True
            and all(str(terminal.get(code, '')).startswith('UNMAPPED;') for code in ('F', 'T', '-')),
            'runner name policy changed terminal rules')
    require(all(policy.get('version_and_scope', {}).get(key) is True for key in (
        'base_v2_v3_projector_bytes_preserved', 'external_policy_hash_required',
        'no_model_fit_calibration_selector_threshold_change', 'no_performance_based_rule',
        'no_provider_or_runtime_authority', 'separate_v4_adapter_required')),
        'runner name policy changed version or evaluation scope')
    refs = [policy['population_reference'], policy['independent_review_reference'],
            policy['source_title_policy_reference']] + policy['prior_parser_references']
    for reference in refs:
        require(digest(reference['path']) == reference['sha256'], 'runner name policy linked hash mismatch')
    require(projection_policy is None or policy['source_title_policy_reference'] == projection_policy['reference'],
            'runner name policy title policy mismatch')
    population = read(policy['population_reference']['path'])['records']
    expected = {r['race_id']: r['prediction_id'] for r in forecasts}
    admitted = {r['race_id']: r['prediction_id'] for r in population}
    require(len(population) == len(admitted) == len(expected) == len(forecasts) == 210
            and admitted == expected and policy.get('population_races') == 210,
            'runner name policy population differs from frozen forecasts')
    return {'document': policy, 'reference': {'path': str(path.resolve()), 'sha256': expected_sha256},
            'population': admitted}


def load_structural_identity_policy(path, expected_sha256, forecasts, projection_policy, runner_name_policy):
    path = Path(path)
    require(isinstance(expected_sha256, str) and re.fullmatch('[0-9a-f]{64}', expected_sha256)
            and digest(path) == expected_sha256, 'structural identity policy hash mismatch')
    policy = read(path)
    require(policy.get('schema') == 'frozen_offline_active_route_identity_policy_v5'
            and policy.get('status') == 'FROZEN_BEFORE_CORRECTED_METRICS'
            and policy.get('field_and_scoring_policy_unchanged') is True
            and policy.get('newly_corrected_metrics_inspected') is False
            and policy.get('no_model_fit_calibration_selector_threshold_change') is True
            and policy.get('no_new_provider_or_runtime_authority') is True,
            'structural identity policy not frozen with unchanged scope')
    require(projection_policy is not None and runner_name_policy is not None
            and policy['source_title_policy_reference'] == projection_policy['reference']
            and policy['runner_name_policy_reference'] == runner_name_policy['reference'],
            'structural identity linked external policy mismatch')
    require(policy.get('selectors') == {'active_meeting': 'a.dropdown__menu__meeting--active',
            'active_race': 'a.race-box--active',
            'active_result_tab': '.page__layout .tabs-wrapper a.tabs__item--active'},
            'structural identity selector scope changed')
    refs = [policy['population_reference'], policy['independent_review_reference'],
            policy['source_title_policy_reference'], policy['runner_name_policy_reference']] + policy['prior_parser_references']
    for reference in refs:
        require(digest(reference['path']) == reference['sha256'], 'structural identity linked hash mismatch')
    population = read(policy['population_reference']['path'])['records']
    expected = {r['race_id']: r['prediction_id'] for r in forecasts}
    admitted = {r['race_id']: r['prediction_id'] for r in population}
    require(len(population) == len(admitted) == len(expected) == len(forecasts) == 210
            and admitted == expected and policy.get('population_races') == 210,
            'structural identity population differs from frozen forecasts')
    return {'document': policy, 'reference': {'path': str(path.resolve()), 'sha256': expected_sha256},
            'population': admitted}


def structured_source_name(cell, box):
    """Read the source's dog identity, keeping annotation text separate."""
    containers = cell.select('.race-runners__name__dog')
    require(len(containers) == 1, 'structured dog container ambiguous')
    anchors = containers[0].find_all('a')
    require(len(anchors) == 1, 'structured dog anchor ambiguous')
    anchor = anchors[0]
    href = anchor.get('href', '')
    route = urlsplit(href)
    match = re.fullmatch(r'/dogs/([1-9][0-9]*)/([^/\s]+)', route.path)
    require(match is not None and not route.query and not route.fragment
            and ((not route.scheme and not route.netloc and href.startswith('/dogs/'))
                 or (route.scheme == 'https' and route.netloc in {'www.thedogs.com.au', 'thedogs.com.au'})),
            'structured dog profile route invalid')
    name = ' '.join(anchor.stripped_strings)
    require(bool(name), 'structured dog name missing')
    identities = cell.select('blackbook-dog[data-dog-id]')
    require(len(identities) <= 1 and all(n['data-dog-id'] == match[1] for n in identities),
            'structured dog id contradictory')
    times = containers[0].select('.race-runners__name__time')
    notes = containers[0].select('.race-runners__name__box')
    require(len(times) <= 1 and len(notes) <= 1, 'structured dog annotation ambiguous')
    note = notes[0].get_text(' ', strip=True) if notes else ''
    require(not note or re.fullmatch(r'\(\s*from\s+box\s+[1-8]\s*\)', note, re.I),
            'structured reserve note unsupported')
    native_name = name + (' ' + note if note else '')
    original_name = ' '.join(cell.get_text(' ', strip=True).split())
    original_name = re.sub(r'^\d{1,2}\s*[.\):\-]\s*', '', original_name)
    original_name = re.split(r'\s+(?:\d{1,2}\.\d{2}\s+)?T:\s+', original_name)[0]
    return native_name, {'source_box': box, 'source_anchor_name': name, 'source_dog_href': href,
                         'source_dog_id': match[1], 'source_blackbook_ids': [n['data-dog-id'] for n in identities],
                         'separate_time_annotations': [n.get_text(' ', strip=True) for n in times],
                         'source_box_note': note, 'original_flattened_name': cell.get_text(' ', strip=True),
                         'original_parser_name': original_name,
                         'source_name_cell_html': str(cell)}


def load_pre_race_field_policy(path, expected_sha256, forecasts, projection_policy,
                               runner_name_policy, structural_identity_policy):
    path = Path(path)
    require(isinstance(expected_sha256, str) and re.fullmatch('[0-9a-f]{64}', expected_sha256)
            and digest(path) == expected_sha256, 'pre-race field policy hash mismatch')
    policy = read(path)
    require(policy.get('schema') == 'frozen_offline_pre_race_field_policy_v6'
            and policy.get('status') == 'FROZEN_BEFORE_CORRECTED_METRICS'
            and policy.get('field_and_scoring_policy_unchanged') is True
            and policy.get('newly_corrected_metrics_inspected') is False
            and policy.get('no_model_fit_calibration_selector_threshold_change') is True
            and policy.get('no_new_provider_or_runtime_authority') is True,
            'pre-race field policy changed scope')
    for key, dep in [('source_title_policy_reference', projection_policy),
                     ('runner_name_policy_reference', runner_name_policy),
                     ('structural_identity_policy_reference', structural_identity_policy)]:
        require(dep is not None and policy[key] == dep['reference'], 'pre-race field policy dependency mismatch')
    refs = [policy[k] for k in ('population_reference', 'authenticated_originals_reference',
            'independent_review_reference', 'source_title_policy_reference',
            'runner_name_policy_reference', 'structural_identity_policy_reference')] + policy['prior_parser_references']
    for ref in refs:
        require(digest(ref['path']) == ref['sha256'], 'pre-race field linked hash mismatch')
    review = read(policy['independent_review_reference']['path'])
    require(policy.get('criteria') == review.get('criteria')
            and review.get('status') == 'PASS_CRITERIA_BEFORE_CORRECTED_METRICS', 'pre-race field criteria changed')
    population = read(policy['population_reference']['path'])['records']
    expected = {r['race_id']: r['prediction_id'] for r in forecasts}
    admitted = {r['race_id']: r['prediction_id'] for r in population}
    require(len(population) == len(admitted) == len(expected) == len(forecasts) == 210
            and admitted == expected and policy.get('population_races') == 210,
            'pre-race field population differs from frozen forecasts')
    original_rows = read(policy['authenticated_originals_reference']['path'])
    originals = {r['original']['record_key']: r for r in original_rows}
    require(len(originals) == len(original_rows) and set(expected.values()) <= set(originals)
            and all(originals[pid]['original']['race_id'] == rid for rid, pid in expected.items()),
            'pre-race field original membership mismatch')
    # The authenticated archive also retains two nonprimary repeat forecasts.
    # They cannot become eligible through this repair: select only frozen IDs.
    originals = {pid: originals[pid] for pid in expected.values()}
    return {'document': policy, 'reference': {'path': str(path.resolve()), 'sha256': expected_sha256},
            'population': admitted, 'originals': originals}


def check_pre_race_field_proof(original, expected_url=None):
    """Authenticate original field without consulting the official outcome."""
    require(isinstance(original, dict) and not original.get('append_errors'), 'pre-race append evidence invalid')
    proof, record = original['append_proof'], original['original']
    ref = proof['artifacts']['sidecar']
    require(digest(ref['path']) == ref['sha256'] == proof['native_prediction']['input_hashes']['sidecar_sha256'],
            'pre-race sidecar not bound to sealed input')
    sidecar = read(ref['path'])
    meta, alignment = sidecar['prejump_shadow_metadata'], sidecar['canonical_runner_alignment']
    require(meta.get('status') == 'PASS' and meta.get('metadata_is_leakage_safe') is True
            and meta.get('fail_reasons') == [] and alignment.get('status') == 'aligned'
            and alignment.get('canonical_runner_set_status') == 'available'
            and alignment.get('duplicate_source_runner_names') == []
            and alignment.get('missing_canonical_participants') == [], 'pre-race field metadata invalid')
    active = meta['runner_box_name_list']
    expected = {(r['box_number'], token(r['dog_name'])) for r in record['predictions']}
    observed = {(r['box_number'], token(r['dog_name'])) for r in active}
    require(len(active) == len(observed) == len(expected) == len(record['predictions'])
            and len({box for box, _ in observed}) == len(active)
            and observed == expected, 'pre-race active field differs from sealed original')
    if expected_url is not None:
        require(meta.get('source_url') == alignment.get('canonical_source_url') == expected_url
                and alignment.get('canonical_runner_count') == alignment.get('prediction_runner_count') == len(active),
                'pre-race source URL or active count mismatch')
    win = original['win_field']
    require(win.get('status') == 'EXACT_COMPLETE_WIN_FIELD', 'pre-race WIN field incomplete')
    def at(value):
        instant = datetime.fromisoformat(value.replace('Z', '+00:00'))
        require(instant.tzinfo is not None, 'pre-race timestamp naive')
        return instant
    require(at(meta['metadata_captured_at']) <= at(record['score_timestamp']) <= at(proof['sealed_at'])
            < at(record['jump_timestamp'])
            and at(win['capture_at']) <= at(win['available_at']) <= at(record['score_timestamp']),
            'pre-race field chronology invalid')
    return {'sidecar_reference': ref, 'pre_race_active_field': active,
            'dropped_participants': alignment['dropped_participants'],
            'remapped_participants': alignment['remapped_participants'],
            'metadata_captured_at': meta['metadata_captured_at'], 'forecast_scored_at': record['score_timestamp'],
            'forecast_sealed_at': proof['sealed_at'], 'complete_win_field': win}


def check_pre_race_field_derivation(rows, proof):
    active = {(r['box_number'], token(r['dog_name'])) for r in proof['pre_race_active_field']}
    names = {name for _, name in active}
    dropped = {(r['box_number'], token(r['dog_name'])) for r in proof['dropped_participants']}
    require(len(dropped) == len(proof['dropped_participants']), 'pre-race dropped identity duplicated')
    derived, excluded, remapped = [], [], []
    for source in rows:
        row = dict(source)
        key = row['box_number'], token(row['dog_name'])
        if row['status'] in {'SCR', 'L/SCR', 'LSCR'} and row['finish_position'] is None:
            require(key not in active, 'pre-race repair cannot remove original scratched runner')
            if key in dropped:
                require(key[1] not in names, 'pre-race exclusion overlaps original identity')
                excluded.append(row)
                continue
        note = re.fullmatch(r'(.+?)\s+\(from box ([1-8])\)', row['dog_name'], re.I)
        if note:
            source_box, target, name = row['box_number'], int(note[2]), note[1]
            matches = [r for r in proof['remapped_participants']
                       if r['source_box_number'] == r['original_box_number'] == source_box
                       and r['final_box_number'] == target and token(r['dog_name']) == token(name)]
            require(source_box in (9, 10) and len(matches) == 1 and (target, token(name)) in active,
                    'pre-race reserve continuity unproved')
            remapped.append({'source_box_number': source_box, 'final_box_number': target,
                             'dog_name': name, 'source_name_with_box_note': row['dog_name'],
                             'pre_race_mapping': matches[0]})
            row.update(box_number=target, dog_name=name, original_box_number=source_box,
                       reserve_box_remap_source='SEALED_PRE_RACE_AND_EXPLICIT_OFFICIAL_FROM_BOX')
        derived.append(row)
    require(len({r['box_number'] for r in derived}) == len(derived), 'pre-race derived boxes duplicated')
    return {'sidecar_reference': proof['sidecar_reference'], 'pre_race_active_field': proof['pre_race_active_field'],
            'excluded_source_rows': excluded, 'remapped_source_rows': remapped,
            'derived_original_field_rows': derived, 'pre_race_proof': proof}


def load_nontrial_navigation_policy(path, expected_sha256, forecasts, title, name, structural, field):
    path = Path(path)
    require(isinstance(expected_sha256, str) and re.fullmatch('[0-9a-f]{64}', expected_sha256)
            and digest(path) == expected_sha256, 'nontrial policy hash mismatch')
    doc = read(path)
    require(doc.get('schema') == 'frozen_offline_nontrial_navigation_policy_v7'
            and doc.get('status') == 'FROZEN_BEFORE_CORRECTED_METRICS'
            and doc.get('newly_corrected_metrics_inspected') is False
            and all(doc.get(k) is True for k in ('field_and_scoring_policy_unchanged',
                'no_new_provider_or_runtime_authority', 'no_model_fit_calibration_selector_threshold_change')),
            'nontrial policy not frozen unchanged scope')
    keys = ('source_title_policy_reference', 'runner_name_policy_reference',
            'structural_identity_policy_reference', 'pre_race_field_policy_reference')
    for key, policy in zip(keys, (title, name, structural, field)):
        require(policy is not None and doc[key] == policy['reference'], 'nontrial policy dependency mismatch')
    refs = [doc[k] for k in keys + ('population_reference', 'independent_review_reference')] + doc['prior_parser_references']
    for reference in refs:
        require(digest(reference['path']) == reference['sha256'], 'nontrial policy linked hash mismatch')
    review = read(doc['independent_review_reference']['path'])
    require(review.get('status') == 'PASS_CRITERIA_BEFORE_CORRECTED_METRICS'
            and review.get('criteria') == doc.get('criteria'), 'nontrial frozen criteria mismatch')
    population = read(doc['population_reference']['path'])['records']
    expected = {r['race_id']: r['prediction_id'] for r in forecasts}
    admitted = {r['race_id']: r['prediction_id'] for r in population}
    require(len(population) == len(admitted) == len(forecasts) == len(expected) == doc.get('population_races') == 210
            and expected == admitted == field['population'], 'nontrial population mismatch')
    return {'reference': {'path': str(path.resolve()), 'sha256': expected_sha256},
            'document': doc, 'population': admitted, 'originals': field['originals']}


def check_census(forecasts, dispositions, strict, diagnostic, projections):
    expected = {r['race_id']: r['prediction_id'] for r in forecasts}
    require(len(expected) == len(forecasts), 'duplicate forecast census')
    require(len(dispositions) == len(expected) and {r['race_id'] for r in dispositions} == set(expected),
            'integration disposition census omitted duplicated or added race')
    ids = []
    for name, records in [('strict', strict), ('diagnostic', diagnostic)]:
        pairs = {(r['race_id'], r['model_version']) for r in records}
        require(len(pairs) == len(records), 'duplicate race model record')
        races = {r['race_id'] for r in records}
        require(races <= set(expected), 'outcome outside forecast census')
        for rid in races:
            group = [r for r in records if r['race_id'] == rid]
            require({r['model_version'].rsplit(':', 1)[-1] for r in group} == {'full', 'half'} and len(group) == 2,
                    'missing or extra frozen forecast variant')
        for r in records:
            require(r['prediction_id'] == expected[r['race_id']], 'primary prediction changed')
            require((r.get('official_runner_identity_observed') is True) == (name == 'strict'),
                    'source identity classification mismatch')
        ids.append(races)
    require(not ids[0] & ids[1], 'race appears in both strict and diagnostic')
    selected = {r['race_id']: r for r in projections}
    require(len(selected) == len(projections) and set(selected) == ids[0] | ids[1],
            'integrated projection census mismatch')
    for d in dispositions:
        require((d['selected_projection'] is not None) == (d['race_id'] in selected),
                'selected projection disposition mismatch')
        if d['race_id'] in ids[0]:
            reviewed = d['disposition'] in {'REVIEWED_OFFLINE_RESULT_SUPERSEDES_PRIOR_DIAGNOSTIC',
                                          'REVIEWED_OFFLINE_STRICT_RESULT'}
            require(d['selected_projection'] == 'ACQUIRED' and
                    (d['disposition'].startswith('NEW_STRICT_RESULT') or reviewed),
                    'strict result lacks acquired disposition')
            if reviewed:
                projection = selected[d['race_id']]
                require(projection.get('original_transport_hold_preserved') is True
                        and projection.get('reviewed_reclassification', {}).get('race_id') == d['race_id'],
                        'reviewed result lacks preserved hold and correction')
                observations = d.get('completion_observations', [])
                require(len(observations) == 1 and observations[0]['source_hold'] is True
                        and observations[0]['charged_attempts'] == observations[0]['provider_requests'] == 1
                        and observations[0]['status'] == 'TERMINAL_SOURCE_HOLD_NO_RETRY',
                        'reviewed result changed original terminal accounting')
            projection = selected[d['race_id']]
            if projection.get('projection_policy_fallback'):
                observations = d.get('completion_observations', [])
                require(not reviewed and len(observations) == 1
                        and observations[0]['status'] == 'RETAINED_RAW_RESULT_IDENTITY_UNVERIFIED'
                        and observations[0]['source_hold'] is False
                        and observations[0]['charged_attempts'] == observations[0]['provider_requests'] == 1,
                        'V3 fallback requires original READY completion without hold')
            if projection.get('runner_name_policy_fallback'):
                observations = d.get('completion_observations', [])
                require(not reviewed and len(observations) == 1
                        and observations[0]['status'] == 'RETAINED_RAW_RESULT_IDENTITY_UNVERIFIED'
                        and observations[0]['source_hold'] is False
                        and observations[0]['charged_attempts'] == observations[0]['provider_requests'] == 1,
                        'V4 fallback requires original READY completion without hold')
            if projection.get('structural_identity_policy_fallback'):
                observations = d.get('completion_observations', [])
                require(not reviewed and len(observations) == 1
                        and observations[0]['status'] == 'RETAINED_RAW_RESULT_IDENTITY_UNVERIFIED'
                        and observations[0]['source_hold'] is False
                        and observations[0]['charged_attempts'] == observations[0]['provider_requests'] == 1,
                        'V5 fallback requires original READY completion without hold')
    for r in strict + diagnostic:
        require(r['result_evidence'] == selected[r['race_id']], 'scored evidence differs from selected projection')
    return {'status': 'PASS', 'forecasts': len(expected), 'strict_races': len(ids[0]),
            'diagnostic_races': len(ids[1]), 'without_usable_result': len(expected)-len(ids[0])-len(ids[1]),
            'dispositions': dict(Counter(r['disposition'] for r in dispositions))}


def token(value):
    return re.sub('[^A-Z0-9]', '', value.upper())


def source_rows(raw, structured_names=False):
    """Independently extract every result-table row; never silently skip one."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(raw, 'html.parser')
    require(len(soup.select('table.race-runners--result')) == 1,
            'source result table missing or ambiguous')
    rows = []
    for node in soup.select('table.race-runners--result tr.race-runner'):
        if structured_names:
            require(all(len(node.select('td.race-runners__' + name)) == 1
                        for name in ('box', 'name', 'finish-position')), 'structured source runner cell ambiguous')
        cells = [node.select_one('td.race-runners__' + name) for name in ('box', 'name', 'finish-position')]
        require(all(c is not None for c in cells), 'source runner cell missing')
        rugs = cells[0].find_all(attrs={'name': re.compile(r'^rug_[0-9]{1,2}$')})
        require(len(rugs) == 1, 'source box ambiguous')
        box = int(rugs[0]['name'][4:])
        name = ' '.join(' '.join(cells[1].stripped_strings).split())
        name = re.sub(r'^\d{1,2}\s*[.\):\-]\s*', '', name)
        name = re.split(r'\s+(?:\d{1,2}\.\d{2}\s+)?T:\s+', name)[0]
        if structured_names:
            name, _ = structured_source_name(cells[1], box)
        status = ' '.join(cells[2].stripped_strings).upper()
        match = re.fullmatch(r'([1-9]|10)(ST|ND|RD|TH)', status)
        require(bool(name) and (match or status in {'SCR', 'L/SCR', 'LSCR', 'DNF', 'FELL', 'DISQ'}),
                'source name or terminal status missing')
        rows.append({'box_number': box, 'dog_name': name, 'finish_position': int(match[1]) if match else None,
                     'status': None if match else status})
    require(rows and len({r['box_number'] for r in rows}) == len(rows), 'empty or duplicate source field')
    return soup, rows


def url_identity(url):
    p = urlsplit(url)
    parts = p.path.rstrip('/').split('/')
    require(p.scheme == 'https' and p.netloc == 'www.thedogs.com.au' and not p.fragment
            and len(parts) >= 5 and parts[1] == 'racing', 'unrecognized official URL')
    require(all(v == 'false' for v in parse_qs(p.query).get('trial', [])), 'source trial mismatch')
    return tuple(parts[2:5])


def check_active_route_identity(soup, expected_url, race_id, projection_policy):
    """Exact event routes from active source controls, without display aliases."""
    require(projection_policy is not None and race_id in projection_policy['population'],
            'V5 race outside frozen title policy population')
    require(not soup.select('link[rel="canonical"], meta[property="og:url"]'),
            'V5 fallback cannot replace present source declarations')
    expected = url_identity(expected_url)
    native = re.fullmatch(r'Race ([1-9][0-9]*) - (.+) - ([0-9]{4}-[0-9]{2}-[0-9]{2})', race_id)
    require(native is not None and expected[1:] == (native[3], native[1]),
            'V5 native date or race mismatch')
    selectors = {'active_race_href': 'a.race-box--active',
                 'active_results_href': '.page__layout .tabs-wrapper a.tabs__item--active',
                 'active_meeting_href': 'a.dropdown__menu__meeting--active'}
    witness = {}
    for key, css in selectors.items():
        nodes = soup.select(css)
        require(len(nodes) == 1, 'V5 active source control missing or ambiguous: ' + key)
        href = nodes[0].get('href', '')
        route = urlsplit(href)
        require((not route.scheme and not route.netloc and href.startswith('/racing/'))
                or (route.scheme == 'https' and route.netloc == 'www.thedogs.com.au'),
                'V5 active source route malformed or foreign')
        resolved = 'https://www.thedogs.com.au' + href if not route.scheme else href
        wanted = ('https://www.thedogs.com.au/racing/' + expected[0] + '/' + expected[1] + '?trial=false'
                  if key == 'active_meeting_href' else expected_url)
        require(resolved == wanted, 'V5 active source route differs from authorized identity')
        if key == 'active_results_href':
            require(nodes[0].get_text(' ', strip=True) == 'RESULTS', 'V5 active tab is not results')
        if key == 'active_race_href':
            labels = nodes[0].select('.race-box__number')
            require(not labels or (len(labels) == 1 and labels[0].get_text(strip=True) == 'R' + expected[2]),
                    'V5 active race label contradicts identity')
        witness[key] = href
    titles = soup.find_all('title')
    require(len(titles) == 1, 'V5 source title missing or ambiguous')
    title = titles[0].get_text().strip()
    match = re.fullmatch(r'thedogs - (.+) ([1-9][0-9]?) ([A-Za-z]+) ([0-9]{4}) Race ([1-9][0-9]*)', title)
    months = ('January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December')
    require(match is not None and match[3] in months, 'V5 title syntax unsupported')
    source_date = date(int(match[4]), months.index(match[3])+1, int(match[2])).isoformat()
    require((source_date, match[5]) == expected[1:], 'V5 source title date or race contradiction')
    known = {(slug, venue) for display, slug, venue in projection_policy['triples'] if display == match[1]}
    require(not known or known == {(expected[0], native[2])}, 'V5 known source title venue contradiction')
    witness.update(active_race_number=int(native[1]), source_title=title,
                   source_title_identity=[expected[0], source_date, int(native[1])],
                   source_title_native_venue=native[2], source_declared_identities=[])
    return witness


def check_declared_identity(soup, expected_url, required_rule=None, *, projection_policy=None, race_id=None):
    """Independent exact native-title witness; no projector/parser imports."""
    expected = url_identity(expected_url)
    nodes = soup.select('link[rel="canonical"], meta[property="og:url"]')
    declared = [n.get('href') if n.name == 'link' else n.get('content') for n in nodes]
    require(all(isinstance(u, str) and u and url_identity(u) == expected for u in declared),
            'source declared race identity mismatch')
    titles = soup.find_all('title')
    if required_rule == V3_IDENTITY:
        require(projection_policy is not None, 'V3 identity requires external projection policy')
        require(race_id in projection_policy['population'], 'V3 race outside frozen policy population')
        require(not declared, 'V3 fallback requires missing source declarations')
        require(len(titles) == 1, 'source title missing or ambiguous')
        native = re.fullmatch(r'Race ([1-9][0-9]*) - (.+) - ([0-9]{4}-[0-9]{2}-[0-9]{2})', race_id)
        require(native is not None and expected[1:] == (native[3], native[1]),
                'V3 native race date or number mismatch')
        months = ('January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December')
        title = re.fullmatch(r'thedogs - (.+) ([1-9][0-9]?) ([A-Za-z]+) ([0-9]{4}) Race ([1-9][0-9]*)',
                             titles[0].get_text().strip())
        require(title is not None and title[3] in months, 'V3 source title identity unsupported')
        title_date = date(int(title[4]), months.index(title[3])+1, int(title[2])).isoformat()
        require((title[1], expected[0], native[2]) in projection_policy['triples']
                and (title_date, title[5]) == expected[1:], 'V3 exact title policy identity mismatch')
        return V3_IDENTITY
    # Preserve the existing canonical path for other source layouts. The new
    # adapter explicitly declares the title rule, which must verify both.
    if declared and required_rule != 'exact_source_title_venue_date_race_v1':
        return 'canonical_or_og_url'
    require(len(titles) == 1, 'source title missing or ambiguous')
    months = ('January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December')
    match = re.fullmatch(r'thedogs - Healesville ([1-9][0-9]?) ([A-Za-z]+) ([0-9]{4}) Race ([1-9][0-9]*)',
                         titles[0].get_text().strip())
    require(match is not None and match[2] in months, 'source title identity unsupported')
    native_date = date(int(match[3]), months.index(match[2])+1, int(match[1])).isoformat()
    require(('healesville', native_date, str(int(match[4]))) == expected,
            'source title identity mismatch')
    return 'exact_source_title_venue_date_race_v1'


def check_nontrial_route_identity(soup, expected_url, race_id, title_policy):
    """Source-declared nontrial controls; never arbitrary URL normalization."""
    require(title_policy is not None and race_id in title_policy['population'], 'V7 missing exact scope')
    route = urlsplit(expected_url)
    require(route.scheme == 'https' and route.netloc == 'www.thedogs.com.au'
            and route.query in ('', 'trial=false') and not route.fragment
            and expected_url == 'https://www.thedogs.com.au' + route.path + ('?' + route.query if route.query else ''),
            'V7 authorized URL malformed or nontrial scope unsupported')
    expected = url_identity(expected_url)
    native = re.fullmatch(r'Race ([1-9][0-9]*) - (.+) - ([0-9]{4}-[0-9]{2}-[0-9]{2})', race_id)
    require(native is not None and expected[1:] == (native[3], native[1])
            and (expected[0], native[2]) in {(slug, venue) for _, slug, venue in title_policy['triples']},
            'V7 native exact route mapping mismatch')
    require(not soup.select('link[rel="canonical"], meta[property="og:url"]'), 'V7 cannot replace source declarations')
    def single(css):
        nodes = soup.select(css)
        require(len(nodes) == 1, 'V7 source node missing or duplicate')
        return nodes[0]
    def exact_href(node, target):
        href = node.get('href', '')
        require(href == target or href == 'https://www.thedogs.com.au' + target,
                'V7 source route not exact explicit nontrial')
        return href
    race = single('a.race-box--active')
    wanted = route.path + '?trial=false'
    race_href = exact_href(race, wanted)
    labels = race.select('.race-box__number')
    require(not labels or (len(labels) == 1 and labels[0].get_text(strip=True) == 'R' + expected[2]),
            'V7 source race label contradiction')
    results = [node for node in soup.select('.page__layout .tabs-wrapper a.tabs__item')
               if node.get_text(' ', strip=True) == 'RESULTS']
    require(len(results) == 1, 'V7 results node missing or duplicate')
    active = soup.select('.page__layout .tabs-wrapper a.tabs__item--active')
    require(not active or (len(active) == 1 and active[0] is results[0]), 'V7 active tab contradiction')
    results_href = exact_href(results[0], wanted)
    meeting_href = exact_href(single('a.dropdown__menu__meeting--active'),
                             '/racing/' + expected[0] + '/' + expected[1] + '?trial=false')
    titles = soup.find_all('title')
    require(len(titles) == 1, 'V7 title missing or duplicate')
    title = titles[0].get_text().strip()
    match = re.fullmatch(r'thedogs - (.+) ([1-9][0-9]?) ([A-Za-z]+) ([0-9]{4}) Race ([1-9][0-9]*)', title)
    months = ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
              'September', 'October', 'November', 'December')
    require(match is not None and match[3] in months, 'V7 title malformed')
    source_date = date(int(match[4]), months.index(match[3])+1, int(match[2])).isoformat()
    require((source_date, match[5]) == expected[1:], 'V7 title date or race contradiction')
    known = {(slug, venue) for display, slug, venue in title_policy['triples'] if display == match[1].strip()}
    require(not known or known == {(expected[0], native[2])}, 'V7 known title venue contradiction')
    require(len(soup.select('table.race-runners--result')) == 1, 'V7 result table missing or ambiguous')
    return {'active_race_href': race_href, 'results_href': results_href,
            'results_anchor_classes': results[0].get('class', []), 'active_meeting_href': meeting_href,
            'active_race_number': int(native[1]), 'source_title': title,
            'source_title_identity': [expected[0], source_date, int(native[1])],
            'source_title_native_venue': native[2], 'source_declared_identities': [],
            'source_title_display_venue': match[1].strip()}


def check_source_identity(record, expected_url, projection_policy=None, runner_name_policy=None,
                          structural_identity_policy=None, pre_race_field_policy=None,
                          nontrial_navigation_policy=None):
    require(record.get('official_runner_identity_observed') is True, 'strict source identity missing')
    evidence = record['result_evidence']
    require(evidence.get('runner_name_origin') == 'SOURCE_NATIVE_OFFICIAL_HTML', 'circular source runner names')
    require(len(evidence['races']) == 1, 'ambiguous source race')
    source_identity = evidence['races'][0].get('source_identity', {})
    required_rule = source_identity.get('identity_semantics')
    fallback = evidence.get('projection_policy_fallback')
    name_fallback = evidence.get('runner_name_policy_fallback')
    structural_fallback = evidence.get('structural_identity_policy_fallback')
    field_fallback = evidence.get('pre_race_field_policy_fallback')
    nontrial_fallback = evidence.get('nontrial_navigation_policy_fallback')
    identity_preproof = None
    if required_rule == V7_IDENTITY or nontrial_fallback:
        require(required_rule == V7_IDENTITY and nontrial_navigation_policy is not None,
                'V7 requires external nontrial policy')
        require(nontrial_navigation_policy['population'].get(record['race_id']) == record['prediction_id'],
                'V7 result outside frozen population')
        require(isinstance(nontrial_fallback, dict) and nontrial_fallback.get('status') == 'QUALIFIED'
                and nontrial_fallback.get('original_parser_failure') in V7_FAILURES
                and nontrial_fallback.get('nontrial_navigation_policy_reference') == nontrial_navigation_policy['reference']
                and source_identity.get('nontrial_navigation_policy_reference') == nontrial_navigation_policy['reference'],
                'V7 fallback trigger or policy mismatch')
        require(not evidence.get('reviewed_reclassification') and not field_fallback
                and evidence.get('original_transport_hold_preserved') is not True,
                'V7 cannot supersede held receipt or silently repair field')
        identity_preproof = check_pre_race_field_proof(nontrial_navigation_policy['originals'][record['prediction_id']], expected_url)
        require(source_identity.get('pre_race_identity_proof') == identity_preproof,
                'V7 pre-race identity proof differs from sealed source')
    field_proof = None
    if field_fallback:
        require(pre_race_field_policy is not None and runner_name_policy is not None,
                'V6 requires external pre-race field policy')
        require(pre_race_field_policy['population'].get(record['race_id']) == record['prediction_id'],
                'V6 outside original forecast population')
        require(field_fallback.get('status') == 'QUALIFIED'
                and field_fallback.get('pre_race_field_policy_reference') == pre_race_field_policy['reference']
                and str(field_fallback.get('original_parser_failure', '')).startswith((
                    'official_runner_name_changed_at_box:', 'official_runner_outside_forecast_field')),
                'V6 fallback policy or original failure mismatch')
        require(not evidence.get('reviewed_reclassification')
                and evidence.get('original_transport_hold_preserved') is not True,
                'V6 cannot supersede held response')
        field_proof = check_pre_race_field_proof(pre_race_field_policy['originals'][record['prediction_id']], expected_url)
    if required_rule in {V5_IDENTITY, V7_IDENTITY} or structural_fallback:
        require(required_rule in {V5_IDENTITY, V7_IDENTITY} and structural_identity_policy is not None,
                'V5 result requires external structural identity policy')
        require(record['race_id'] in structural_identity_policy['population']
                and record['prediction_id'] == structural_identity_policy['population'][record['race_id']],
                'V5 result outside frozen policy population')
        require(isinstance(structural_fallback, dict) and (structural_fallback.get('status') == 'QUALIFIED'
                or (field_fallback and structural_fallback.get('status') == 'REJECTED'
                    and structural_fallback.get('fallback_failure') == field_fallback.get('original_parser_failure'))
                or (nontrial_fallback and structural_fallback.get('status') == 'REJECTED'
                    and structural_fallback.get('fallback_failure') == nontrial_fallback.get('original_parser_failure')))
                and structural_fallback.get('original_parser_failure') == 'source_title_display_slug_native_not_exact_policy_member'
                and structural_fallback.get('structural_identity_policy_reference') == structural_identity_policy['reference']
                and source_identity.get('structural_identity_policy_reference') == structural_identity_policy['reference'],
                'V5 result structural policy reference or original failure mismatch')
        require(not evidence.get('reviewed_reclassification')
                and evidence.get('original_transport_hold_preserved') is not True,
                'V5 fallback cannot supersede held receipt')
    if name_fallback:
        require(runner_name_policy is not None, 'V4 result requires external runner name policy')
        require(record['race_id'] in runner_name_policy['population']
                and record['prediction_id'] == runner_name_policy['population'][record['race_id']],
                'V4 result outside frozen policy population')
        require(isinstance(name_fallback, dict) and (name_fallback.get('status') == 'QUALIFIED'
                or (field_fallback and name_fallback.get('status') == 'REJECTED'
                    and name_fallback.get('fallback_failure') == field_fallback.get('original_parser_failure')))
                and str(name_fallback.get('original_parser_failure', '')).startswith('official_runner_name_changed_at_box:')
                and name_fallback.get('runner_name_policy_reference') == runner_name_policy['reference'],
                'V4 result runner name policy reference mismatch')
        require(not evidence.get('reviewed_reclassification')
                and evidence.get('original_transport_hold_preserved') is not True,
                'V4 fallback cannot supersede held receipt')
    if required_rule in {V3_IDENTITY, V5_IDENTITY, V7_IDENTITY} or fallback:
        require(required_rule in {V3_IDENTITY, V5_IDENTITY, V7_IDENTITY} and projection_policy is not None,
                'V3 result lacks externally pinned projection policy')
        require(record['race_id'] in projection_policy['population']
                and record['prediction_id'] == projection_policy['population'][record['race_id']],
                'V3 result outside frozen policy population')
        accepted_title_trail = isinstance(fallback, dict) and (fallback.get('status') == 'QUALIFIED'
            or (name_fallback and fallback.get('status') == 'REJECTED'
                and fallback.get('fallback_failure') == name_fallback.get('original_parser_failure'))
            or (structural_fallback and fallback.get('status') == 'REJECTED'
                and fallback.get('fallback_failure') == structural_fallback.get('original_parser_failure'))
            or (field_fallback and fallback.get('status') == 'REJECTED'
                and fallback.get('fallback_failure') == field_fallback.get('original_parser_failure')))
        require(accepted_title_trail
                and fallback.get('original_parser_failure') == 'source_declared_race_identity_missing_or_mismatched'
                and fallback.get('projection_policy_reference') == projection_policy['reference']
                and source_identity.get('projection_policy_reference') == projection_policy['reference'],
                'V3 result projection policy reference mismatch')
        require(not evidence.get('reviewed_reclassification')
                and evidence.get('original_transport_hold_preserved') is not True,
                'V3 fallback cannot supersede held receipt')
    provenance = evidence['response_provenance']
    require(bool(nontrial_fallback) == bool(provenance.get('nontrial_navigation_policy_reference')),
            'V7 policy provenance lacks fallback')
    if nontrial_fallback:
        require(provenance.get('nontrial_navigation_policy_reference') == nontrial_navigation_policy['reference']
                and provenance.get('pre_race_identity_sidecar_reference') == identity_preproof['sidecar_reference'],
                'V7 source provenance differs from sealed identity')
    if structural_fallback:
        require(provenance.get('structural_identity_policy_reference') == structural_identity_policy['reference'],
                'V5 response provenance policy reference mismatch')
    require(bool(structural_fallback) == bool(provenance.get('structural_identity_policy_reference')),
            'V5 structural identity evidence lacks fallback')
    if name_fallback and not field_fallback:
        require(provenance.get('runner_name_policy_reference') == runner_name_policy['reference'],
                'V4 response provenance policy reference mismatch')
    if field_fallback:
        require(provenance.get('pre_race_field_policy_reference') == pre_race_field_policy['reference']
                and provenance.get('source_name_extraction_policy_reference') == runner_name_policy['reference']
                and provenance.get('sidecar_reference') == field_proof['sidecar_reference']
                and provenance.get('runner_name_policy_reference') is None,
                'V6 provenance field or name dependency mismatch')
    else:
        require(not provenance.get('pre_race_field_policy_reference'), 'V6 policy lacks fallback')
        require(bool(name_fallback) == bool(provenance.get('runner_name_policy_reference'))
                == bool(evidence['races'][0].get('structured_runner_name_witnesses')),
                'V4 structural name evidence missing fallback or policy')
    if required_rule in {V3_IDENTITY, V5_IDENTITY, V7_IDENTITY}:
        require(provenance.get('projection_policy_reference') == projection_policy['reference'],
                'V3 response provenance policy reference mismatch')
    for prefix in ('response', 'metadata'):
        require(digest(provenance[prefix + '_path']) == provenance[prefix + '_sha256'], 'source response hash changed')
    for ref in record['result_artifacts']:
        require(digest(ref['path']) == ref['sha256'], 'result witness hash changed')
    metadata = read(provenance['metadata_path'])
    require(metadata['response_sha256'] == provenance['response_sha256'] and metadata['http_status'] == 200
            and metadata['requested_url'] == metadata['final_url'] == expected_url, 'source metadata mismatch')
    # Membership and scope are checked by run before this function opens a body.
    raw = Path(provenance['response_path']).read_bytes()
    soup, rows = source_rows(raw, structured_names=bool(name_fallback or field_fallback))
    if name_fallback or field_fallback:
        witnesses = [structured_source_name(node.select_one('td.race-runners__name'), row['box_number'])[1]
                     for node, row in zip(soup.select('table.race-runners--result tr.race-runner'), rows)]
        require(witnesses == evidence['races'][0]['structured_runner_name_witnesses'],
                'V4 projected dog identity witnesses differ from independent source')
    if required_rule == V7_IDENTITY:
        witness = check_nontrial_route_identity(soup, metadata['requested_url'], record['race_id'], projection_policy)
        require(all(source_identity.get(k) == v for k, v in witness.items()),
                'V7 projected identity differs from independently checked source')
        identity_rule = V7_IDENTITY
    elif required_rule == V5_IDENTITY:
        witness = check_active_route_identity(soup, metadata['requested_url'], record['race_id'], projection_policy)
        require(all(source_identity.get(k) == v for k, v in witness.items()),
                'V5 projected identity differs from independent active source routes')
        identity_rule = V5_IDENTITY
    else:
        identity_rule = check_declared_identity(soup, metadata['requested_url'],
            required_rule, projection_policy=projection_policy, race_id=record['race_id'])
    if identity_rule == V3_IDENTITY:
        native = re.fullmatch(r'Race ([1-9][0-9]*) - (.+) - ([0-9]{4}-[0-9]{2}-[0-9]{2})', record['race_id'])
        expected = url_identity(expected_url)
        require(source_identity.get('source_title') == soup.find('title').get_text().strip()
                and source_identity.get('source_title_identity') == [expected[0], expected[1], int(expected[2])]
                and source_identity.get('source_title_native_venue') == native[2]
                and source_identity.get('source_declared_identities') == [],
                'V3 projected source identity differs from independent title')
    require(len(evidence['races']) == 1 and evidence['races'][0]['native_official_rows'] == rows,
            'native source rows differ from independently parsed HTML')
    field_projection = None
    if field_fallback:
        field_projection = check_pre_race_field_derivation(rows, field_proof)
        require(field_projection == evidence['races'][0].get('pre_race_field_projection'),
                'V6 derived field differs from independent sealed-proof derivation')
        require(source_rows(raw)[1] == evidence['races'][0].get('raw_native_official_rows'),
                'V6 original native rows changed')
    by_source = {r['box_number']: r for r in rows}
    used, mapped = set(), set()
    for runner in evidence['runners']:
        source_box = runner['original_source_box']
        require(source_box not in used, 'source runner reused')
        used.add(source_box)
        src = by_source[source_box]
        name = src['dog_name']
        if source_box != runner['box_number']:
            match = re.search(r'\(\s*from\s+box\s+(\d+)\s*\)\s*$', name, re.I)
            require(source_box in (9, 10) and match and int(match[1]) == runner['box_number']
                    and (src['finish_position'] is not None or field_fallback), 'source reserve continuity not explicit')
            if field_fallback:
                require(any(r['source_box_number'] == source_box and r['final_box_number'] == runner['box_number']
                            for r in field_projection['remapped_source_rows']), 'V6 reserve mapping lacks pre-race proof')
            name = re.sub(r'\s+\d{1,2}\.\d{2}\s*$', '', name[:match.start()].strip())
            name = re.sub(r'\s+NBT\s*$', '', name, flags=re.I)
            mapped.add(runner['box_number'])
        require(token(name) == token(runner['dog_name']) and src['finish_position'] == runner['finish_position']
                and (src['status'] or 'FINISHED') == runner['terminal_status'], 'source runner identity or terminal disposition changed')
        require(runner['is_winner'] == int(src['finish_position'] == 1), 'source winner changed')
    for box in set(by_source)-used:
        src = by_source[box]
        require(src['status'] in {'SCR', 'L/SCR', 'LSCR'} and src['finish_position'] is None
                and (box in (9, 10) or box in mapped
                     or (field_projection and src in field_projection['excluded_source_rows'])),
                'source runner silently dropped')
    return {'status': 'PASS', 'race_id': record['race_id'], 'source_runners': len(rows),
            'forecast_runners': len(evidence['runners']), 'identity_rule': identity_rule}


def run(baseline, integration, completed, projection_policy_path=None, projection_policy_sha256=None,
        runner_name_policy_path=None, runner_name_policy_sha256=None,
        structural_identity_policy_path=None, structural_identity_policy_sha256=None,
        pre_race_field_policy_path=None, pre_race_field_policy_sha256=None,
        nontrial_navigation_policy_path=None, nontrial_navigation_policy_sha256=None):
    # This audits allocation before decoding any acquired result body, then all
    # immutable forecasts, metrics, calibration, dates, intervals and selectors.
    result = independent.run(baseline, completed)
    for old_name, new_name in [('forecast-dataset.json', 'forecast-dataset.json'), ('forecast-selector-plan.json', 'forecast-selector-plan.FROZEN.json'), ('forecast-selector-results.json', 'forecast-selector-dispositions.FROZEN.json')]:
        require(digest(baseline / 'completed' / old_name) == digest(completed / new_name), 'frozen artifact bytes changed')
    provenance = read(integration / 'provenance.json')
    for ref in provenance['inputs']:
        require(digest(ref['path']) == ref['sha256'], 'integration input hash changed')
    expected_urls = {r['race_id']: r['official_source_url'] for r in read(provenance['inputs'][1]['path'])['records']}
    protocol_hash = read(baseline / 'completed/strict-dataset.json')['protocol_sha256']
    forecasts = read(completed / 'forecast-dataset.json')['records']
    require((projection_policy_path is None) == (projection_policy_sha256 is None),
            'projection policy path and hash must be provided together')
    policy = (load_projection_policy(projection_policy_path, projection_policy_sha256, forecasts, expected_urls)
              if projection_policy_path is not None else None)
    require((runner_name_policy_path is None) == (runner_name_policy_sha256 is None),
            'runner name policy path and hash must be provided together')
    name_policy = (load_runner_name_policy(runner_name_policy_path, runner_name_policy_sha256, forecasts, policy)
                   if runner_name_policy_path is not None else None)
    require((structural_identity_policy_path is None) == (structural_identity_policy_sha256 is None),
            'structural identity policy path and hash must be provided together')
    identity_policy = (load_structural_identity_policy(structural_identity_policy_path,
                        structural_identity_policy_sha256, forecasts, policy, name_policy)
                       if structural_identity_policy_path is not None else None)
    require((pre_race_field_policy_path is None) == (pre_race_field_policy_sha256 is None),
            'pre-race field policy path and hash must be provided together')
    field_policy = (load_pre_race_field_policy(pre_race_field_policy_path, pre_race_field_policy_sha256,
                    forecasts, policy, name_policy, identity_policy)
                    if pre_race_field_policy_path is not None else None)
    require((nontrial_navigation_policy_path is None) == (nontrial_navigation_policy_sha256 is None),
            'nontrial policy path and hash must be supplied together')
    nontrial_policy = (load_nontrial_navigation_policy(nontrial_navigation_policy_path,
        nontrial_navigation_policy_sha256, forecasts, policy, name_policy, identity_policy, field_policy)
        if nontrial_navigation_policy_path is not None else None)
    records = {kind: read(completed / f'{kind}-dataset.json')['records'] for kind in ('strict', 'diagnostic')}
    for kind, rows in records.items():
        dataset = read(completed / f'{kind}-dataset.json')
        claim = read(completed / f'{kind}-scorecard.json')
        require(dataset['protocol_sha256'] == claim['protocol_sha256'] == protocol_hash, 'frozen protocol changed')
        source = integration / f'{kind}-records.json'
        require(dataset['source_sha256'] == digest(source) and rows == read(source), 'rerun integration binding mismatch')
        require(len({(r['race_id'], r['model_version']) for r in rows}) == len(rows), 'duplicate scored record')
    result['integration_census'] = check_census(forecasts, read(integration / 'integration-dispositions.json'),
        records['strict'], records['diagnostic'], read(integration / 'integrated-result-projection.json')['records'])
    result['source_identity'] = [check_source_identity(r, expected_urls[r['race_id']], policy, name_policy, identity_policy, field_policy, nontrial_policy)
                                 for r in records['strict'] if r['model_version'].endswith(':full')]
    if policy is not None:
        result['projection_policy_reference'] = policy['reference']
    if name_policy is not None:
        result['runner_name_policy_reference'] = name_policy['reference']
    if identity_policy is not None:
        result['structural_identity_policy_reference'] = identity_policy['reference']
    if field_policy is not None:
        result['pre_race_field_policy_reference'] = field_policy['reference']
    if nontrial_policy is not None:
        result['nontrial_navigation_policy_reference'] = nontrial_policy['reference']
    for kind in records:
        for row in records[kind]:
            for ref in row['result_artifacts']:
                require(digest(ref['path']) == ref['sha256'], 'result artifact changed')
    result['scope'] = 'Independent metrics and integrated membership/source audit; runtime admission and consumption require separate owner receipt review.'
    result['integration_artifacts'] = {str(p): digest(p) for p in sorted(integration.glob('*.json'))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-root', 'integration', 'completed', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--projection-policy', type=Path)
    parser.add_argument('--projection-policy-sha256')
    parser.add_argument('--runner-name-policy', type=Path)
    parser.add_argument('--runner-name-policy-sha256')
    parser.add_argument('--structural-identity-policy', type=Path)
    parser.add_argument('--structural-identity-policy-sha256')
    parser.add_argument('--pre-race-field-policy', type=Path)
    parser.add_argument('--pre-race-field-policy-sha256')
    parser.add_argument('--nontrial-navigation-policy', type=Path)
    parser.add_argument('--nontrial-navigation-policy-sha256')
    args = parser.parse_args()
    result = run(args.baseline_root, args.integration, args.completed,
                 args.projection_policy, args.projection_policy_sha256,
                 args.runner_name_policy, args.runner_name_policy_sha256,
                 args.structural_identity_policy, args.structural_identity_policy_sha256,
                 args.pre_race_field_policy, args.pre_race_field_policy_sha256,
                 args.nontrial_navigation_policy, args.nontrial_navigation_policy_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
