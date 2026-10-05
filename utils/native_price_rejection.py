"""Private diagnostic retention for an already rejected scratch/price conflict.

A reference from this module never qualifies a race or authorizes another request.
"""
import hashlib
import json
import os
import re
import stat
from pathlib import Path

SCHEMA = 'thedogs_native_price_rejection_v1'
REASON = 'scratched_runner_has_active_price'
DISPOSITION = 'REJECTED_DIAGNOSTIC_ONLY_NOT_ADMISSIBLE'
MAX_BODY = 16 * 1024 * 1024
MAX_RECEIPT = 48 * 1024 * 1024


def _directory(root, *, create=False):
    root = Path(root)
    if not root.is_absolute() or root.resolve() != root or not root.is_dir():
        raise ValueError('native_price_rejection_root_unsafe')
    parent = root / 'source_evidence'
    if create:
        parent.mkdir(exist_ok=True)
    if parent.resolve() != parent or not parent.is_dir():
        raise ValueError('native_price_rejection_path_unsafe')
    directory = parent / 'native_price_rejections'
    if create:
        directory.mkdir(mode=0o700, exist_ok=True)
    if (directory.resolve() != directory or not directory.is_dir()
            or directory.stat().st_uid != os.geteuid()
            or stat.S_IMODE(directory.stat().st_mode) != 0o700):
        raise ValueError('native_price_rejection_private_directory_required')
    return directory


def _response(response, *, include_body):
    from scripts.capture_thedogs_market_history import _response_receipt, bounded_receipt_headers
    from utils.http_client import source_retry_headers
    value = _response_receipt(response, include_body=include_body)
    value['headers'] = {**bounded_receipt_headers(response.headers), **source_retry_headers(response.headers)}
    value.pop('request_headers', None)
    return value


def _read(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as handle:
        before = os.fstat(handle.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_uid != os.geteuid() or stat.S_IMODE(before.st_mode) != 0o400
                or not 0 < before.st_size <= MAX_RECEIPT):
            raise ValueError('native_price_rejection_file_unsafe')
        raw = handle.read(MAX_RECEIPT + 1)
        after = os.fstat(handle.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mode, s.st_mtime_ns, s.st_ctime_ns)
    if path.resolve() != path or identity(before) != identity(after) or identity(after) != identity(path.stat()):
        raise ValueError('native_price_rejection_file_changed')
    return raw


def persist_native_price_rejection(root, primary, odds, api, expected_boxes, jump):
    """Persist only responses already obtained; caller re-raises the original error."""
    from scripts.capture_thedogs_market_history import canonical_json_bytes, iso_utc
    if any(not 0 < len(response.body) <= MAX_BODY for response in (primary, odds, api)):
        raise ValueError('native_price_rejection_body_limit')
    value = {'schema_version': SCHEMA, 'reason': REASON, 'disposition': DISPOSITION,
             'race_url': primary.requested_url, 'jump': iso_utc(jump),
             'expected_active_runner_boxes': dict(sorted(expected_boxes.items())),
             'primary': _response(primary, include_body=False),
             'odds': _response(odds, include_body=True), 'api': _response(api, include_body=True),
             'additional_requests': 0}
    raw = canonical_json_bytes(value)
    if len(raw) > MAX_RECEIPT:
        raise ValueError('native_price_rejection_receipt_limit')
    digest = hashlib.sha256(raw).hexdigest()
    path = _directory(root, create=True) / (digest + '.json')
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    except FileExistsError:
        if _read(path) != raw:
            raise ValueError('native_price_rejection_collision') from None
    else:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        # Leave any failed partial write as consumed evidence; never overwrite it.
        if _read(path) != raw:
            raise ValueError('native_price_rejection_write_changed')
    return {'schema_version': SCHEMA, 'path': str(path), 'sha256': digest}


def inspect_native_price_rejection(root, reference):
    """Authenticate and replay the typed rejection, returning categories only."""
    from scripts.capture_thedogs_market_history import (
        CaptureError, _api_url, _stored_response, normalize_api_snapshot, parse_source_runners,
    )
    if (set(reference) != {'schema_version', 'path', 'sha256'} or reference['schema_version'] != SCHEMA
            or re.fullmatch('[a-f0-9]{64}', str(reference['sha256'])) is None):
        raise ValueError('native_price_rejection_reference_invalid')
    path = Path(reference['path'])
    if path.parent != _directory(root) or path.name != reference['sha256'] + '.json':
        raise ValueError('native_price_rejection_path_unsafe')
    raw = _read(path)
    if hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise ValueError('native_price_rejection_hash_changed')
    value = json.loads(raw)
    if (value['schema_version'] != SCHEMA or value['reason'] != REASON
            or value['disposition'] != DISPOSITION or value['additional_requests'] != 0):
        raise ValueError('native_price_rejection_schema_changed')
    for role in ('odds', 'api'):
        if type(value[role]['body_bytes']) is not int or not 0 < value[role]['body_bytes'] <= MAX_BODY:
            raise ValueError('native_price_rejection_body_limit')
    odds = _stored_response(value['odds'], field='odds', exact_url=value['race_url'].split('?')[0].rstrip('/')+'/odds',
                            content_type_prefix='text/html', require_body=True)
    runners = parse_source_runners(odds.body)
    if {r.native_runner_id for r in runners if r.active} != set(value['expected_active_runner_boxes']):
        raise ValueError('native_price_rejection_roster_changed')
    api = _stored_response(value['api'], field='api', exact_url=_api_url(runners),
                           content_type_prefix='application/json', require_body=True)
    try:
        normalize_api_snapshot(json.loads(api.body.decode('utf-8')), runners)
    except CaptureError as error:
        if str(error) != REASON:
            raise ValueError('native_price_rejection_reason_changed') from None
    else:
        raise ValueError('native_price_rejection_not_reproduced')
    return {'schema_version': SCHEMA, 'status': 'REJECTION_REPRODUCED_NOT_ADMISSIBLE',
            'reason': REASON, 'odds_body_sha256': value['odds']['body_sha256'],
            'api_body_sha256': value['api']['body_sha256'], 'additional_requests': 0}
