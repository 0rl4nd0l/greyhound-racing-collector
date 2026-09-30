"""Verified finite development authority and separately charged source leases."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

DATES = ['2026-10-03', '2026-10-04', '2026-10-10', '2026-10-11']
ALLOCATION = 'development-single-snapshot-20261003-v1'
CAPS = dict(max_capture_attempts=24, max_attempts_per_date=6,
            max_logical_requests=24000, max_logical_requests_per_date=6000,
            max_live_seconds=28800, max_live_seconds_per_date=7200,
            max_source_operations=192, max_source_operations_per_date=48,
            max_result_operations=72, max_result_logical_requests=720)


def load_development_authority(ref):
    """Read only an immutable identity/control record, never target data."""
    try:
        path = Path(ref['path'])
        if (not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents))
                or not path.is_file() or path.stat().st_size > 65536):
            raise ValueError()
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != ref['sha256']:
            raise ValueError()
        value = json.loads(raw)
        if (value.get('schema_version') != 'collector_development_pilot_authority_v1'
                or value.get('status') != 'AUTHORIZED_DEVELOPMENT_PILOT'
                or value.get('allocation_id') != ALLOCATION
                or value.get('dates') != DATES
                or not value.get('authority_reference') or not value.get('campaign_id')
                or not Path(value['state_root']).is_absolute()
                or value.get('result_closure_at') != '2026-10-25T12:00:00+11:00'
                or any(type(value.get(k)) is not int or value[k] != v for k, v in CAPS.items())
                or any(not re.fullmatch('[0-9a-f]{64}', value.get(k, ''))
                       for k in ('allocation_sha256', 'prior_effective_authorization_sha256'))):
            raise ValueError()
        return value
    except (KeyError, TypeError, ValueError, OSError):
        raise ValueError('invalid_development_source_authority') from None


def validate_development_lease(row):
    authority = load_development_authority(row['development_authority'])
    day = row['development_slot']
    zone = ZoneInfo('Australia/Melbourne')
    begin = datetime.fromtimestamp(row['authorized_at'], zone)
    end = datetime.fromtimestamp(row['expires_at'], zone)
    if (day not in DATES or begin.date().isoformat() != day or end.date().isoformat() != day
            or not begin.replace(hour=12, minute=40, second=0, microsecond=0) <= begin < end
            or end > begin.replace(hour=14, minute=40, second=0, microsecond=0)
            or row.get('development_allocation_id') != ALLOCATION
            or row.get('development_authority_sha256') != row['development_authority']['sha256']
            or row.get('reference') != authority['authority_reference'] + ':slot:' + day
            or row.get('prior_phase') != 'OPEN' or row.get('max_operations') != 48
            or 'engineering_authority' in row or 'cooldown_revision' in row):
        raise ValueError('invalid_development_source_lease')
    return authority


def development_source_usage(value, baseline_count):
    """Subtract only authenticated pilot operations; retain global consumption."""
    operations = value.get('operations', [])
    authorizations = value.get('diagnostic_authorizations', [])
    recognized = set()
    days = set()
    authorities = set()
    for position, row in enumerate(authorizations):
        if 'development_authority' not in row:
            continue
        validate_development_lease(row)
        day = row['development_slot']
        if day in days:
            raise ValueError('development_slot_consumed')
        days.add(day)
        authorities.add(row['development_authority_sha256'])
        start = row['operation_start']
        end = authorizations[position+1]['operation_start'] if position+1 < len(authorizations) else len(operations)
        if (type(start) is not int or type(end) is not int
                or not 0 <= start <= end <= len(operations) or end-start > 48
                or len(authorities) != 1):
            raise ValueError('invalid_development_source_accounting')
        for index in range(start, end):
            operation = operations[index]
            if (operation.get('development_authority_sha256') != row['development_authority_sha256']
                    or operation.get('development_slot') != day
                    or not row['authorized_at'] <= operation['at'] < row['expires_at']):
                raise ValueError('invalid_development_source_accounting')
            recognized.add(index)
    if any(('development_authority_sha256' in row or 'development_slot' in row)
           and index not in recognized for index, row in enumerate(operations)):
        raise ValueError('invalid_development_source_accounting')
    return sum(index >= baseline_count for index in recognized)
