"""Default-off, values-free qualification of one retained pre-race history pair.

This adapter does not create model features or read result databases. Event links
are opaque source observations, never a claimed cross-provider race identity.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

ORIGIN = 'https://www.thedogs.com.au'
ROLES = ('standard', 'expert', 'runner', 'runner_expert')
FIELDS = {'TIME': 'finish_time', 'WIN': 'race_finish_time', 'BON': 'best_of_night_time',
          '1 SEC': 'first_sectional_time'}
MISSING = {'', '-', '—', '–', 'N/A'}
MAX_BODY = 4 * 1024 * 1024
MAX_CONTROL = 128 * 1024


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _token(value):
    return _hash(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def _read(ref, limit):
    _require(set(ref) == {'path', 'sha256'}, 'reference_shape')
    path = Path(ref['path'])
    _require(path.is_absolute() and path.resolve() == path and path.is_file(), 'reference_path')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    _require(len(raw) <= limit and _hash(raw) == ref['sha256'], 'reference_hash_or_limit')
    return raw


def _time(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    _require(parsed.utcoffset() is not None, 'timestamp_offset_missing')
    return parsed


def _url(value):
    parsed = urlsplit(value)
    _require(parsed.scheme == 'https' and parsed.netloc == 'www.thedogs.com.au'
             and not parsed.fragment and not parsed.username, 'source_url')
    return value


def _surface(item, role, jump):
    _require(set(item) == {'url', 'receipt', 'body'}, 'surface_shape')
    receipt = json.loads(_read(item['receipt'], MAX_CONTROL))
    raw = _read(item['body'], MAX_BODY)
    url = _url(item['url'])
    if role == 'standard':
        _require(receipt.get('schema_version') == 'thedogs_primary_race_page_evidence_v1', 'primary_schema')
        _require(receipt.get('requested_url') == receipt.get('final_url') == url
                 and receipt.get('status_code') == 200
                 and receipt.get('body_sha256') == item['body']['sha256']
                 and receipt.get('content_length') == len(raw), 'primary_transport')
        start, end = _time(receipt['request_start_utc']), _time(receipt['request_end_utc'])
        headers = receipt.get('headers', {})
        local_failure = False
    else:
        _require(receipt.get('kind') == role and receipt.get('method') == 'GET'
                 and receipt.get('url') == receipt.get('final_url') == url
                 and receipt.get('http_status') == 200 and receipt.get('body') == item['body']
                 and receipt.get('body_bytes') == len(raw) and receipt.get('redirects') is False
                 and receipt.get('retries') == 0, 'inspection_transport')
        _require(set(receipt.get('retry_headers', {})) <= {'date'}, 'source_retry_guidance')
        local_failure = receipt.get('status') == 'FAILED_PRESERVED'
        _require(receipt.get('status') == 'RETAINED_HTTP_200' or
                 (role == 'expert' and local_failure and receipt.get('error_type') == 'TypeError'),
                 'inspection_terminal')
        start, end = _time(receipt['started_at']), _time(receipt['ended_at'])
        headers = {'content-type': receipt.get('content_type', ''), **receipt.get('retry_headers', {})}
    headers = {str(k).lower(): v for k, v in headers.items()}
    _require(str(headers.get('content-type', '')).lower().startswith('text/html')
             and not ({'retry-after', 'ratelimit-reset', 'x-ratelimit-reset'} & set(headers)), 'source_hold_or_type')
    _require(start <= end < jump, 'surface_not_prejump')
    return BeautifulSoup(raw, 'html.parser'), start, end, local_failure


def _ids(node, attr):
    return {str(n[attr]) for n in [node, *node.select('[' + attr + ']')] if n.has_attr(attr)}


def _parent_binding(soups, runner_url, expert_runner_url):
    entry = re.fullmatch(r'https://www\.thedogs\.com\.au/dogs/runner/([1-9][0-9]*)', runner_url)
    _require(entry is not None and expert_runner_url == runner_url + '/expert-form', 'runner_route')
    entry = entry.group(1)
    groups = [n for n in soups['standard'].select('tbody[data-content-url]')
              if urljoin(ORIGIN, n['data-content-url']) == runner_url]
    _require(len(groups) == 1 and _ids(groups[0], 'data-runner-id') == {entry}, 'standard_entry_binding')
    profiles = _ids(groups[0], 'data-dog-id')
    _require(len(profiles) == 1 and all(re.fullmatch(r'[1-9][0-9]*', p) for p in profiles), 'standard_profile_binding')
    loaders = [n for n in soups['expert'].select('table-loader[data-src]')
               if urljoin(ORIGIN, n['data-src']) == expert_runner_url]
    _require(len(loaders) == 1, 'expert_route_binding')
    block = loaders[0].find_previous(class_='layout--sidebar--expert')
    _require(block is not None and _ids(block, 'data-runner-id') == {entry}
             and _ids(block, 'data-dog-id') == profiles, 'expert_parent_binding')
    _require(_ids(soups['runner'], 'data-dog-id') == profiles, 'runner_profile_binding')
    # Expert details have no native attributes in the retained sample; any
    # supplied identity must still agree. Exact parent-route HTTP proof binds it.
    for role in ('runner', 'runner_expert'):
        observed_entries = _ids(soups[role], 'data-runner-id')
        observed_profiles = _ids(soups[role], 'data-dog-id')
        _require(not observed_entries or observed_entries == {entry}, 'detail_entry_conflict')
        _require(not observed_profiles or observed_profiles == profiles, 'detail_profile_conflict')
    return _token({'namespace': 'thedogs_parent_entry_profile', 'entry': entry, 'profile': sorted(profiles)})


def _definitions(expert):
    observed = {}
    for select in expert.select('select'):
        options = select.select('option')
        if not any(o.get_text(' ', strip=True) == '1 SEC' for o in options):
            continue
        for option in options:
            label = option.get_text(' ', strip=True)
            if label in FIELDS:
                observed.setdefault(label, set()).add(option.get('value'))
    return {label: ('SOURCE_SORT_KEY_CONFIRMED' if observed.get(label) == {key}
                    else 'SOURCE_SORT_KEY_MISSING_OR_CONFLICTING') for label, key in FIELDS.items()}


def _history(soup, role, jump):
    required = {'DATE', 'TRACK', 'DIST', *FIELDS, 'PIR'}
    tables = []
    for candidate in soup.select('table.runner-form' if role == 'runner' else 'table.race-runners--expert'):
        labels = [h.get_text(' ', strip=True).upper() for h in candidate.select('thead th')]
        if required.issubset(labels):
            tables.append((candidate, labels))
    _require(len(tables) == 1, 'history_table_cardinality')
    table, headers = tables[0]
    _require(all(headers.count(label) == 1 for label in required), 'history_schema')
    allrows = table.select('tbody > tr')
    _require(len(allrows) <= 100, 'history_row_limit')
    rows, excluded, duplicate = {}, Counter(), 0
    present, numeric = Counter(), Counter()
    rendered = 0
    for row in allrows:
        cells = row.find_all('td', recursive=False)
        if len(cells) != len(headers):
            excluded['NON_HISTORY_OR_INCOMPLETE_ROW'] += 1
            continue
        rendered += 1
        bylabel = dict(zip(headers, cells))
        datecell = bylabel['DATE']
        epochs = datecell.select('[data-timestamp]')
        links = {urljoin(ORIGIN, a['href']) for a in datecell.select('a[href]')}
        if len(epochs) != 1 or len(links) != 1:
            excluded['EVENT_IDENTITY_INCOMPLETE'] += 1
            continue
        event = next(iter(links))
        try:
            _url(event)
            epoch = float(epochs[0]['data-timestamp'])
            date = datetime.fromtimestamp(epoch, timezone.utc)
            _require(date < jump, 'history_not_before_target')
        except (ValueError, TypeError, OverflowError):
            excluded['EVENT_DATE_OR_URL_INVALID'] += 1
            continue
        track = bylabel['TRACK'].get_text(' ', strip=True)
        distance = bylabel['DIST'].get_text(' ', strip=True)
        if not track or not distance:
            excluded['TRACK_OR_DISTANCE_MISSING'] += 1
            continue
        key = _token({'namespace': 'thedogs_exact_event_url_and_date', 'url': event, 'epoch': epoch})
        fields = {label: bylabel[label].get_text(' ', strip=True) for label in (*FIELDS, 'PIR')}
        observation = {'track': track, 'distance': distance, 'fields': fields,
                       'last_win': 'runner-form__last-win' in row.get('class', [])}
        if key in rows:
            prior = rows[key]
            same = all(prior[k] == observation[k] for k in ('track', 'distance', 'fields'))
            if same and (prior['last_win'] or observation['last_win']):
                duplicate += 1
                continue
            excluded['DUPLICATE_EVENT_CONFLICT_OR_UNMARKED'] += 1
            # Keep neither interpretation qualified on contradictory evidence.
            prior['conflict'] = True
            continue
        rows[key] = observation
    qualified = {k: v for k, v in rows.items() if not v.get('conflict')}
    for value in qualified.values():
        for label, text in value['fields'].items():
            present[label] += text not in MISSING
            numeric[label] += bool(re.fullmatch(r'\d+(?:\.\d+)?', text))
    return qualified, {'rendered_rows': rendered, 'unique_identity_rows': len(qualified),
        'deduplicated_last_win_rows': duplicate, 'exclusion_categories': dict(excluded),
        'field_presence': {label: {'present': present[label], 'numeric': numeric[label],
                                 'missing': len(qualified) - present[label]} for label in (*FIELDS, 'PIR')},
        'show_more_present': bool(soup.select('[data-runner-show-more]')),
        'history_completeness': 'PAGINATION_OR_HISTORY_EXTENT_UNVERIFIED',
        'event_identity': 'SOURCE_URL_DATE_OBSERVATION_ONLY',
        'historical_runner_namespace': 'UNKNOWN_NOT_MAPPED_FROM_PARENT_ENTRY',
        'history_publication_time': 'UNKNOWN_CAPTURE_IS_UPPER_BOUND_ONLY',
        'pir_source_class': ('IN_RUNNING_PLACES_CLASS_PRESENT' if table.select('.runner-form__in-running-places')
                             else 'SOURCE_CLASS_UNVERIFIED')}


def audit_manifest(reference):
    manifest = json.loads(_read(reference, MAX_CONTROL))
    _require(set(manifest) == {'schema_version', 'max_cards', 'max_runners', 'target_jump', 'surfaces'}
             and manifest.get('schema_version') == 'retained_speed_history_scope_v1'
             and manifest.get('max_cards') == manifest.get('max_runners') == 1
             and set(manifest['surfaces']) == set(ROLES), 'bounded_scope')
    jump = _time(manifest['target_jump'])
    surfaces = {role: _surface(manifest['surfaces'][role], role, jump) for role in ROLES}
    soups = {role: value[0] for role, value in surfaces.items()}
    urls = {role: manifest['surfaces'][role]['url'] for role in ROLES}
    standard = urlsplit(urls['standard'])
    _require(standard.path.startswith('/racing/') and
             urls['expert'] == ORIGIN + standard.path.rstrip('/') + '/expert-form', 'same_parent_card')
    _require(surfaces['standard'][2] <= surfaces['runner'][1]
             and surfaces['expert'][2] <= surfaces['runner_expert'][1], 'parent_after_detail')
    binding = _parent_binding(soups, urls['runner'], urls['runner_expert'])
    normal, normal_projection = _history(soups['runner'], 'runner', jump)
    expert, expert_projection = _history(soups['runner_expert'], 'runner_expert', jump)
    shared = normal.keys() & expert.keys()
    return {'schema_version': 'retained_speed_history_qualification_v1',
        'status': 'STRUCTURE_AUDITED_SPEED_SEMANTICS_UNQUALIFIED', 'scope': reference,
        'source_bindings': manifest['surfaces'], 'parent_identity_token': binding,
        'scope_counts': {'cards': 1, 'runners': 1, 'surfaces': 4},
        'source_label_definitions': _definitions(soups['expert']),
        'normal': normal_projection, 'expert': expert_projection,
        'shared_event_observations': len(shared), 'normal_only_observations': len(normal.keys() - expert.keys()),
        'expert_only_observations': len(expert.keys() - normal.keys()),
        'equal_raw_track_labels': sum(normal[k]['track'] == expert[k]['track'] for k in shared),
        'equal_raw_distance_labels': sum(normal[k]['distance'] == expert[k]['distance'] for k in shared),
        'track_namespace': 'UNVERIFIED_NO_CROSS_SURFACE_CANONICALIZATION',
        'distance_units': 'UNVERIFIED_NO_UNIT_INFERENCE',
        'physical_sectional_clock': 'UNKNOWN', 'pir_call_positions': 'UNKNOWN',
        'best_of_night_publication_basis': 'UNKNOWN', 'usable_early_speed_events': 0,
        'preserved_expert_local_failure': surfaces['expert'][3],
        'provider_requests': 0, 'result_reads': 0, 'features_or_models_changed': False,
        'numeric_or_outcome_values_emitted': False}
