"""Standing engineering authority and immutable daily accounting; no live access."""
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

CAP_KEYS = frozenset({'max_python_requests', 'max_browser_navigations',
    'max_capture_attempts', 'max_source_operations', 'max_result_requests'})
TAG_KEYS = frozenset({'persistent_allocation', 'persistent_allocation_sha256',
                      'persistent_allocation_id'})


def stamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError('persistent_timestamp_ambiguous')
    return parsed.astimezone(timezone.utc)


def checked(ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('persistent_reference_invalid')
    path = Path(ref['path'])
    if not path.is_absolute() or path.resolve() != path or not path.is_file() or path.stat().st_size > 262144:
        raise ValueError('persistent_reference_unsafe')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('persistent_reference_changed')
    return json.loads(raw)


def _root(value):
    path = Path(value)
    if not path.is_absolute() or path.resolve() != path or path == Path('/'):
        raise ValueError('persistent_root_unsafe')
    return path


def _overlap(a, b):
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def _caps(value, maximum=None):
    if (not isinstance(value, dict) or set(value) != CAP_KEYS
            or any(type(v) is not int or v < (0 if k == 'max_result_requests' else 1)
                   for k, v in value.items()) or value['max_result_requests'] != 0
            or maximum is not None and any(value[k] > maximum[k] for k in CAP_KEYS)):
        raise ValueError('persistent_caps_invalid')
    return value


def load_standing_authority(ref):
    value = checked(ref)
    if (value.get('schema_version') != 'collector_persistent_operation_authority_v1'
            or value.get('status') != 'AUTHORIZED_PERSISTENT_ENGINEERING'
            or any(not isinstance(value.get(k), str) or not value[k].strip()
                   for k in ('operation_id', 'campaign_id', 'authority_reference'))
            or value.get('timezone') != 'Australia/Melbourne'
            or value.get('engineering_only') is not True
            or any(value.get(k) is not False for k in ('study_enrolment', 'performance_evaluation',
                'human_outcome_access', 'local_request_caps_are_provider_permission'))
            or stamp(value['issued_at']) > stamp(value['starts_at'])):
        raise ValueError('persistent_standing_authority_invalid')
    _caps(value['daily_caps'])
    study = checked(value['study_plan'])
    checked(value['candidate_registry'])
    if study.get('status') != 'AUTHORIZED' or study.get('candidate_registry') != value['candidate_registry']:
        raise ValueError('persistent_study_binding_invalid')
    protected = value.get('protected_roots')
    if not isinstance(protected, list) or not protected:
        raise ValueError('persistent_protected_roots_required')
    roots = [_root(value[k]) for k in ('state_root', 'prediction_root')]
    protected = [_root(p) for p in protected]
    protected += [_root(study['programme_root'])]
    protected += [_root(p) for p in study.get('prediction_output_roots', [])]
    if _overlap(*roots) or any(_overlap(p, q) for p in roots for q in protected):
        raise ValueError('persistent_root_overlaps_protected_population')
    return value


def allocation_identity(standing_ref, racing_date):
    if date.fromisoformat(racing_date).isoformat() != racing_date:
        raise ValueError('persistent_racing_date_invalid')
    return standing_ref['sha256'] + ':' + racing_date


def load_persistent_allocation(ref):
    value = checked(ref)
    standing = load_standing_authority(value['standing_authority'])
    zone = ZoneInfo(standing['timezone'])
    start, end, cleanup, issued = (stamp(value[k]) for k in ('starts_at', 'ends_at', 'cleanup_by', 'issued_at'))
    racing_date = value['racing_date']
    if (value.get('schema_version') != 'collector_persistent_daily_allocation_v1'
            or value.get('status') != 'AUTHORIZED_PERSISTENT_DAILY_ENGINEERING'
            or value.get('allocation_id') != allocation_identity(value['standing_authority'], racing_date)
            or start.astimezone(zone).date().isoformat() != racing_date
            or not stamp(standing['issued_at']) <= issued <= start < end <= cleanup
            or start < stamp(standing['starts_at'])
            or not 0 < (cleanup - start).total_seconds() <= 26 * 3600
            or (cleanup - end).total_seconds() > 1860):
        raise ValueError('persistent_daily_allocation_invalid')
    for key in ('state_root', 'prediction_root'):
        if _root(value[key]) != _root(standing[key]) / 'days' / racing_date:
            raise ValueError('persistent_daily_root_invalid')
    caps = _caps(value['caps'], standing['daily_caps'])
    return {**value, **caps, 'campaign_id': standing['campaign_id'],
        'authority_reference': standing['authority_reference'], 'study_plan': standing['study_plan'],
        'candidate_registry': standing['candidate_registry'],
        'max_logical_requests': caps['max_python_requests'] + caps['max_browser_navigations'],
        'max_live_seconds': (cleanup - start).total_seconds()}


def persistent_usage(ledger, campaign_id, *, allocation_sha256=None):
    """Subtract only complete authenticated daily allocations from study totals."""
    totals = dict(capture_attempts=0, live_seconds=0, python=0, browser=0, prediction=0, results=0, logical_requests=0)
    allocations = {}; identities = {}; per_allocation = {}

    def validate(row):
        ref = row.get('persistent_allocation')
        if not isinstance(ref, dict):
            raise ValueError('persistent_consumption_missing_reference')
        sha = ref.get('sha256')
        if sha not in allocations:
            allocations[sha] = load_persistent_allocation(ref)
        allocation = allocations[sha]
        identity = allocation['allocation_id']
        if (allocation['campaign_id'] != campaign_id or row.get('persistent_allocation_sha256') != sha
                or row.get('persistent_allocation_id') != identity
                or any(k in row for k in ('engineering_authority', 'incident_authority', 'development_authority_sha256'))
                or identities.setdefault(identity, ref) != ref):
            raise ValueError('persistent_consumption_binding_invalid')
        usage = per_allocation.setdefault(sha, dict(capture_attempts=0, live_seconds=0, python=0, browser=0, results=0))
        return allocation, usage, sha

    for row in ledger['attempts']:
        if not TAG_KEYS.intersection(row):
            continue
        allocation, usage, sha = validate(row)
        if not stamp(allocation['starts_at']) <= stamp(row['consumed_at']) < stamp(allocation['ends_at']):
            raise ValueError('persistent_capture_time_invalid')
        usage['capture_attempts'] += 1
    for row in ledger['launches'].values():
        if not TAG_KEYS.intersection(row):
            continue
        allocation, usage, sha = validate(row)
        charged = row['charged_seconds']
        start = stamp(row['started_at']).timestamp()
        end = row['deadline_epoch']
        if (type(charged) not in (int, float) or not 0 <= charged <= allocation['max_live_seconds']
                or not stamp(allocation['starts_at']).timestamp() <= start < end <= stamp(allocation['cleanup_by']).timestamp()
                or not charged <= end - start):
            raise ValueError('persistent_launch_time_invalid')
        usage['live_seconds'] += charged
    for identity, row in ledger.get('persistent_operation_request_usage', {}).items():
        allocation, usage, sha = validate(row)
        counts = row.get('counts')
        if (identity != allocation['allocation_id'] or not isinstance(counts, dict)
                or set(counts) != {'python', 'browser', 'results'}
                or any(type(v) is not int or v < 0 for v in counts.values())):
            raise ValueError('persistent_request_consumption_invalid')
        for k in counts:
            usage[k] += counts[k]
    for sha, usage in per_allocation.items():
        a = allocations[sha]
        caps = dict(capture_attempts=a['max_capture_attempts'], live_seconds=a['max_live_seconds'],
                    python=a['max_python_requests'], browser=a['max_browser_navigations'], results=0)
        if any(usage[k] > caps[k] for k in caps):
            raise ValueError('persistent_consumption_exceeds_allocation')
        if allocation_sha256 is None or sha == allocation_sha256:
            for k in usage:
                totals[k] += usage[k]
    totals['prediction'] = totals['python'] + totals['browser']
    totals['logical_requests'] = totals['prediction'] + totals['results']
    if totals['logical_requests'] > ledger.get('logical_requests', 0):
        raise ValueError('persistent_consumption_exceeds_global_accounting')
    return totals


def validate_persistent_lease(row):
    allocation = load_persistent_allocation(row['persistent_allocation'])
    if (row.get('persistent_allocation_sha256') != row['persistent_allocation']['sha256']
            or row.get('persistent_allocation_id') != allocation['allocation_id']
            or row.get('reference') != allocation['authority_reference'] + ':day:' + allocation['allocation_id']
            or row.get('prior_phase') != 'OPEN'
            or type(row.get('max_operations')) is not int
            or row['max_operations'] != allocation['max_source_operations']
            or not stamp(allocation['starts_at']).timestamp() <= row['authorized_at'] < row['expires_at'] <= stamp(allocation['ends_at']).timestamp()
            or any(k in row for k in ('engineering_authority', 'development_authority', 'incident_authority', 'cooldown_revision'))):
        raise ValueError('persistent_source_lease_invalid')
    return allocation


def persistent_source_usage(value, baseline_count):
    operations = value.get('operations', [])
    grants = value.get('diagnostic_authorizations', [])
    recognized = set(); seen = set()
    for index, row in enumerate(grants):
        if not TAG_KEYS.intersection(row):
            continue
        allocation = validate_persistent_lease(row)
        if allocation['allocation_id'] in seen:
            raise ValueError('persistent_source_day_consumed')
        seen.add(allocation['allocation_id'])
        begin = row['operation_start']
        end = grants[index + 1]['operation_start'] if index + 1 < len(grants) else len(operations)
        if (type(begin) is not int or type(end) is not int or not 0 <= begin <= end <= len(operations)
                or end - begin > row['max_operations']):
            raise ValueError('persistent_source_accounting_invalid')
        for i in range(begin, end):
            operation = operations[i]
            if (any(operation.get(k) != row[k] for k in ('persistent_allocation_sha256', 'persistent_allocation_id'))
                    or not row['authorized_at'] <= operation['at'] < row['expires_at']):
                raise ValueError('persistent_source_accounting_invalid')
            recognized.add(i)
    if any(TAG_KEYS.intersection(row) and i not in recognized for i, row in enumerate(operations)):
        raise ValueError('persistent_source_accounting_invalid')
    return sum(i >= baseline_count for i in recognized)
