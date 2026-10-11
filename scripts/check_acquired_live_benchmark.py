#!/usr/bin/env python3
"""Independent acquired-result audit; no scorer, projector or acquisition imports."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import re
from urllib.parse import urlsplit, parse_qs

try:
    from . import check_live_reconciliation as independent
except ImportError:
    import check_live_reconciliation as independent

read, digest = independent.read, independent.digest


def require(test, reason):
    if not test:
        raise ValueError(reason)


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
    for r in strict + diagnostic:
        require(r['result_evidence'] == selected[r['race_id']], 'scored evidence differs from selected projection')
    return {'status': 'PASS', 'forecasts': len(expected), 'strict_races': len(ids[0]),
            'diagnostic_races': len(ids[1]), 'without_usable_result': len(expected)-len(ids[0])-len(ids[1]),
            'dispositions': dict(Counter(r['disposition'] for r in dispositions))}


def token(value):
    return re.sub('[^A-Z0-9]', '', value.upper())


def source_rows(raw):
    """Independently extract every result-table row; never silently skip one."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(raw, 'html.parser')
    require(len(soup.select('table.race-runners--result')) == 1,
            'source result table missing or ambiguous')
    rows = []
    for node in soup.select('table.race-runners--result tr.race-runner'):
        cells = [node.select_one('td.race-runners__' + name) for name in ('box', 'name', 'finish-position')]
        require(all(c is not None for c in cells), 'source runner cell missing')
        rugs = cells[0].find_all(attrs={'name': re.compile(r'^rug_[0-9]{1,2}$')})
        require(len(rugs) == 1, 'source box ambiguous')
        box = int(rugs[0]['name'][4:])
        name = ' '.join(' '.join(cells[1].stripped_strings).split())
        name = re.sub(r'^\d{1,2}\s*[.\):\-]\s*', '', name)
        name = re.split(r'\s+(?:\d{1,2}\.\d{2}\s+)?T:\s+', name)[0]
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


def check_declared_identity(soup, expected_url, required_rule=None):
    """Independent exact native-title witness; no projector/parser imports."""
    expected = url_identity(expected_url)
    nodes = soup.select('link[rel="canonical"], meta[property="og:url"]')
    declared = [n.get('href') if n.name == 'link' else n.get('content') for n in nodes]
    require(all(isinstance(u, str) and u and url_identity(u) == expected for u in declared),
            'source declared race identity mismatch')
    titles = soup.find_all('title')
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


def check_source_identity(record, expected_url):
    require(record.get('official_runner_identity_observed') is True, 'strict source identity missing')
    evidence = record['result_evidence']
    require(evidence.get('runner_name_origin') == 'SOURCE_NATIVE_OFFICIAL_HTML', 'circular source runner names')
    provenance = evidence['response_provenance']
    for prefix in ('response', 'metadata'):
        require(digest(provenance[prefix + '_path']) == provenance[prefix + '_sha256'], 'source response hash changed')
    for ref in record['result_artifacts']:
        require(digest(ref['path']) == ref['sha256'], 'result witness hash changed')
    metadata = read(provenance['metadata_path'])
    require(metadata['response_sha256'] == provenance['response_sha256'] and metadata['http_status'] == 200
            and metadata['requested_url'] == metadata['final_url'] == expected_url, 'source metadata mismatch')
    # Membership and scope are checked by run before this function opens a body.
    soup, rows = source_rows(Path(provenance['response_path']).read_bytes())
    require(len(evidence['races']) == 1, 'ambiguous source race')
    identity_rule = check_declared_identity(soup, metadata['requested_url'],
        evidence['races'][0].get('source_identity', {}).get('identity_semantics'))
    require(len(evidence['races']) == 1 and evidence['races'][0]['native_official_rows'] == rows,
            'native source rows differ from independently parsed HTML')
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
                    and src['finish_position'] is not None, 'source reserve continuity not explicit')
            name = re.sub(r'\s+\d{1,2}\.\d{2}\s*$', '', name[:match.start()].strip())
            name = re.sub(r'\s+NBT\s*$', '', name, flags=re.I)
            mapped.add(runner['box_number'])
        require(token(name) == token(runner['dog_name']) and src['finish_position'] == runner['finish_position']
                and (src['status'] or 'FINISHED') == runner['terminal_status'], 'source runner identity or terminal disposition changed')
        require(runner['is_winner'] == int(src['finish_position'] == 1), 'source winner changed')
    for box in set(by_source)-used:
        src = by_source[box]
        require(src['status'] in {'SCR', 'L/SCR', 'LSCR'} and src['finish_position'] is None
                and (box in (9, 10) or box in mapped), 'source runner silently dropped')
    return {'status': 'PASS', 'race_id': record['race_id'], 'source_runners': len(rows),
            'forecast_runners': len(evidence['runners']), 'identity_rule': identity_rule}


def run(baseline, integration, completed):
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
    result['source_identity'] = [check_source_identity(r, expected_urls[r['race_id']]) for r in records['strict'] if r['model_version'].endswith(':full')]
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
    args = parser.parse_args()
    result = run(args.baseline_root, args.integration, args.completed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
