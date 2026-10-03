"""Retain and replay completed race-local schedule rejections; admit no race."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

from bs4 import BeautifulSoup
from scripts.capture_thedogs_market_history import (
    CaptureError, _response_receipt, _stored_response, _write_new_immutable,
)
from utils.csv_metadata import canonical_thedogs_race_identity
from utils.race_schedule_time import extract_formatted_timing, scheduled_jump_datetime

MAX_EVIDENCE_BYTES = 16 * 1024 * 1024


def _valid(candidate, evidence):
    identity = canonical_thedogs_race_identity(candidate.get('race_url') or candidate.get('url'))
    if (identity is None or evidence.get('schema_version') != 'canonical_schedule_change_rejection_v1'
            or evidence.get('race_url') != (candidate.get('race_url') or candidate.get('url'))
            or evidence.get('race_date') != identity['race_date']
            or str(evidence.get('race_number')) != str(identity['race_number'])
            or str(candidate.get('race_number')) != str(identity['race_number'])
            or candidate.get('date') != identity['race_date']):
        return False
    old = scheduled_jump_datetime(candidate)
    if old is None or evidence.get('discovery_jump') != candidate.get('scheduled_jump_datetime'):
        return False
    if candidate.get('jump_datetime') is not None:
        alias = datetime.fromisoformat(candidate['jump_datetime'])
        if alias.utcoffset() is None or alias != old:
            return False
    response = _stored_response(evidence.get('race_page_http'), field='race_page_http',
        exact_url=evidence['race_url'], content_type_prefix='text/html', require_body=True)
    if any(response.headers.get(key) for key in ('retry-after','x-ratelimit-reset','ratelimit-reset')):
        return False
    timing = extract_formatted_timing(BeautifulSoup(response.body.decode('utf-8'), 'html.parser'))
    new = scheduled_jump_datetime({'date':identity['race_date'], **timing})
    claimed = datetime.fromisoformat(evidence['canonical_jump'])
    return (new is not None and claimed.utcoffset() is not None and claimed == new
        and old != new and old > response.request_end_utc and new > response.request_end_utc)


def retain_schedule_change(*, race_url, hint, canonical_info, response, artifact_root):
    """Called only after the existing canonical download detected disagreement."""
    identity = canonical_thedogs_race_identity(race_url)
    if (identity is None or not isinstance(hint, dict)
            or canonical_info.get('date') != identity['race_date']
            or str(canonical_info.get('race_number')) != str(identity['race_number'])):
        return None
    evidence = {'schema_version':'canonical_schedule_change_rejection_v1',
        'race_url':race_url,'race_date':identity['race_date'],'race_number':identity['race_number'],
        'discovery_jump':hint.get('scheduled_jump_datetime'),
        'canonical_jump':canonical_info.get('scheduled_jump_datetime'),
        'race_page_http':_response_receipt(response, include_body=True)}
    try:
        if not _valid(hint, evidence):
            return None
    except (CaptureError, KeyError, TypeError, ValueError, OverflowError):
        return None
    raw = json.dumps(evidence,sort_keys=True,separators=(',',':')).encode()
    if len(raw) > MAX_EVIDENCE_BYTES:
        return None
    root = Path(artifact_root)
    if not root.is_absolute() or root.resolve() != root:
        return None
    sha = hashlib.sha256(raw).hexdigest()
    path = root/'schedule-rejections'/(sha+'.json')
    _write_new_immutable(path,raw)
    return {'path':str(path),'sha256':sha}


def complete_schedule_change_rejection(candidate, result, root):
    """A bare reason, missing acquisition, changed bytes or denial remains fatal."""
    try:
        if (result.get('success') is not False
                or result.get('error') != 'discovery_canonical_jump_changed'):
            return False
        ref=result['schedule_change_evidence']
        if not isinstance(ref,dict) or set(ref) != {'path','sha256'}:
            return False
        directory=Path(root);path=Path(ref['path'])
        if (not directory.is_absolute() or directory.resolve()!=directory
                or not path.is_absolute() or not path.is_relative_to(directory)
                or any(p.is_symlink() for p in (path,*path.parents))):
            return False
        with path.open('rb') as stream:
            raw=stream.read(MAX_EVIDENCE_BYTES+1)
        if len(raw)>MAX_EVIDENCE_BYTES or hashlib.sha256(raw).hexdigest()!=ref['sha256']:
            return False
        return _valid(candidate,json.loads(raw))
    except (CaptureError, KeyError, TypeError, ValueError, AttributeError, OSError, OverflowError):
        return False
