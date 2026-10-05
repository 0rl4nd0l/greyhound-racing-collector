"""Retain and authenticate a race-local primary/odds roster contradiction.

This evidence can only exclude a race. It never supplies a usable runner set.
"""
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

SCHEMA = 'thedogs_native_roster_rejection_v1'
REASON = 'expected_native_runner_set_mismatch'
MAX_BODY = 16 * 1024 * 1024
MAX_RECEIPT = 48 * 1024 * 1024


def _directory(root, *, create=False):
    root = Path(root)
    if not root.is_absolute() or root.resolve() != root or not root.is_dir():
        raise ValueError('native_roster_rejection_root_unsafe')
    directory = root
    for name in ('source_evidence', 'native_roster_rejections'):
        directory = directory/name
        if create: directory.mkdir(exist_ok=True)
        if directory.resolve() != directory or not directory.is_dir():
            raise ValueError('native_roster_rejection_path_unsafe')
    return directory


def _receipt(response):
    from scripts.capture_thedogs_market_history import _response_receipt, bounded_receipt_headers
    value = _response_receipt(response, include_body=True)
    from utils.http_client import source_retry_headers
    value['headers'] = {**bounded_receipt_headers(response.headers), **source_retry_headers(response.headers)}
    # Retain no cookies or authorization; these requests use public source headers.
    value.pop('request_headers', None)
    return value


def persist_native_roster_rejection(root, race_page, odds, expected_boxes, jump):
    from scripts.capture_thedogs_market_history import (
        _write_or_verify_immutable, canonical_json_bytes, iso_utc, parse_source_runners,
    )
    if any(not 0 < len(r.body) <= MAX_BODY for r in (race_page, odds)):
        raise ValueError('native_roster_rejection_body_limit')
    observed = sorted(r.native_runner_id for r in parse_source_runners(odds.body) if r.active)
    value = {'schema_version': SCHEMA, 'disposition': 'EXCLUDED_NATIVE_ROSTER_CONTRADICTION',
        'reason': REASON, 'race_url': race_page.requested_url, 'jump': iso_utc(jump),
        'expected_active_runner_boxes': dict(sorted(expected_boxes.items())),
        'observed_active_runner_ids': observed,
        'primary': _receipt(race_page), 'odds': _receipt(odds)}
    raw = canonical_json_bytes(value)
    digest = hashlib.sha256(raw).hexdigest()
    path = _directory(root, create=True)/(digest+'.json')
    _write_or_verify_immutable(path, raw, collision='native_roster_rejection_collision')
    return {'schema_version': SCHEMA, 'path': str(path), 'sha256': digest}


def verify_native_roster_rejection(root, reference, candidate, *, primary_sha256=None):
    """Replay exact pre-jump source pages; reject unknown transport/identity faults."""
    from scripts.capture_thedogs_market_history import (
        _stored_response, _expected_native_runner_box_map, parse_source_runners,
        parse_jump_from_source, parse_timestamp, exact_odds_identity,
    )
    from utils.http_client import source_retry_headers
    from utils.runner_completeness import extract_canonical_runner_set_from_html
    if set(reference) != {'schema_version', 'path', 'sha256'} or reference['schema_version'] != SCHEMA:
        raise ValueError('native_roster_rejection_reference_invalid')
    path = Path(reference['path'])
    if (path.parent != _directory(root) or path.resolve() != path or not path.is_file()
            or path.stat().st_nlink != 1 or not 0 < path.stat().st_size <= MAX_RECEIPT
            or path.name != reference['sha256']+'.json'):
        raise ValueError('native_roster_rejection_path_unsafe')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise ValueError('native_roster_rejection_hash_changed')
    value = json.loads(raw)
    url = candidate['race_url']
    parsed = urlparse(url)
    canonical_url = parsed._replace(query='', fragment='').geturl().rstrip('/')
    odds_url = exact_odds_identity(canonical_url+'/odds')['odds_url']
    jump = parse_timestamp(candidate['jump_datetime'], field='candidate_jump')
    if (value['schema_version'] != SCHEMA or value['reason'] != REASON
            or value['disposition'] != 'EXCLUDED_NATIVE_ROSTER_CONTRADICTION'
            or value['race_url'] != url or parse_timestamp(value['jump'],field='jump') != jump):
        raise ValueError('native_roster_rejection_identity_changed')
    responses = []
    for role, exact in [('primary', url), ('odds', odds_url)]:
        item = value[role]
        if (type(item['body_bytes']) is not int or not 0 < item['body_bytes'] <= MAX_BODY
                or set(source_retry_headers(item['headers'])) - {'date'}):
            raise ValueError('native_roster_rejection_transport_hold')
        responses.append(_stored_response(item, field=role, exact_url=exact,
            content_type_prefix='text/html', require_body=True))
    primary, odds = responses
    if (not primary.request_start_utc <= primary.request_end_utc <= odds.request_start_utc <= odds.request_end_utc < jump
            or (odds.request_end_utc-primary.request_end_utc).total_seconds() > 1200
            or parse_jump_from_source(primary.body) != jump
            or (primary_sha256 is not None and value['primary']['body_sha256'] != primary_sha256)):
        raise ValueError('native_roster_rejection_time_or_primary_changed')
    canonical = extract_canonical_runner_set_from_html(primary.body.decode('utf-8'),
        source_url=url, expected_race_number=int(candidate['race_number']),
        extraction_timestamp=primary.request_end_utc.isoformat())
    if canonical['canonical_runner_set_status'] != 'available':
        raise ValueError('native_roster_rejection_primary_unavailable')
    expected = _expected_native_runner_box_map([(r['source_native_runner_id'],r['box_number'])
        for r in canonical['final_runner_participants']])
    observed = sorted(r.native_runner_id for r in parse_source_runners(odds.body) if r.active)
    if (expected != value['expected_active_runner_boxes']
            or observed != value['observed_active_runner_ids'] or set(observed) == set(expected)):
        raise ValueError('native_roster_rejection_not_reproduced')
    return {'primary_body_sha256': value['primary']['body_sha256'],
            'expected_active_runner_boxes': expected, 'observed_active_runner_ids': observed}
