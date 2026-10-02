"""Finite dated incident authority; no provider or outcome access."""
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

CAPS = {'max_capture_attempts_per_window':24,
        'max_prediction_logical_requests_per_window':16000,
        'max_source_operations_per_window':192,
        'max_result_requests_per_window':72,
        'max_result_operations_per_window':72}
OCTOBER2_CAPS = {**CAPS, 'max_capture_attempts_per_window':32,
                 'max_prediction_logical_requests_per_window':24064,
                 'max_result_requests_per_window':96, 'max_result_operations_per_window':96}
DISPOSITION = 'AUTHORIZED_NON_EVALUATIVE_REUSE_EXCLUDING_ADMITTED_STUDY_IDENTITIES'


def stamp(value):
    result = datetime.fromisoformat(value)
    if result.utcoffset() is None:
        raise ValueError('incident_timestamp_ambiguous')
    return result


def checked(ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('incident_reference_invalid')
    path = Path(ref['path'])
    if (not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents))
            or not path.is_file() or path.stat().st_size > 262144):
        raise ValueError('incident_reference_unsafe')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('incident_reference_changed')
    return json.loads(raw)


def load_incident_authority(ref):
    """Authenticate explicit non-evaluative reuse, unchanged reservations/models."""
    try:
        value = checked(ref)
        october2 = value.get('schema_version') == 'collector_incident_engineering_authority_20261002_v1'
        caps = OCTOBER2_CAPS if october2 else CAPS
        day = '2026-10-02' if october2 else '2026-10-01'
        if (value['schema_version'] not in {'collector_incident_engineering_authority_v1',
                                            'collector_incident_engineering_authority_20261002_v1'}
                or value['status'] != 'AUTHORIZED_INCIDENT_ENGINEERING'
                or not all(isinstance(value[k],str) and value[k].strip()
                           for k in ('incident_id','authority_reference','campaign_id'))
                or value['engineering_only'] is not True
                or any(value[k] is not False for k in
                       ('study_enrolment','performance_evaluation','human_outcome_access'))
                or value['reservation_disposition'] != DISPOSITION
                or any(type(value[k]) is not int or value[k] != cap for k,cap in caps.items())):
            raise ValueError()
        if october2:
            if (type(value.get('max_python_requests_per_window')) is not int
                    or type(value.get('max_browser_navigations_per_window')) is not int
                    or value.get('max_python_requests_per_window') != 24000
                    or value.get('max_browser_navigations_per_window') != 64
                    or value.get('second_window_requires_demonstrated_correction') is not True
                    or not isinstance(value.get('limits_basis'), dict) or not value['limits_basis']):
                raise ValueError()
        zone = ZoneInfo('Australia/Melbourne')
        issued = stamp(value['issued_at'])
        stop = stamp(value['collection_stop_at'])
        cleanup = stamp(value['cleanup_deadline'])
        deadline = stamp(value['result_deadline'])
        if (stop != stamp(day+'T21:00:00+10:00')
                or cleanup != stamp(day+'T21:30:00+10:00')
                or not cleanup <= deadline <= stamp('2026-10-04T12:00:00+11:00' if october2 else '2026-10-02T12:00:00+10:00')
                or issued.astimezone(zone).date().isoformat() != day):
            raise ValueError()
        slots = value['slots']
        if not isinstance(slots,list) or len(slots) != 2:
            raise ValueError()
        previous = None
        for number, slot in enumerate(slots, 1):
            start,end,closed = (stamp(slot[k]) for k in ('starts_at','ends_at','cleanup_by'))
            if (set(slot) != {'id','starts_at','ends_at','cleanup_by'} or slot['id'] != f'{number:03d}'
                    or not issued < start < end <= stop or end-start != timedelta(minutes=90)
                    or not end <= closed <= min(end+timedelta(seconds=1860),cleanup)
                    or start.astimezone(zone).date().isoformat() != day
                    or previous is not None and start < previous):
                raise ValueError()
            previous = closed
        study = checked(value['study_plan'])
        study_results = checked(value['study_result_authority'])
        allocation = checked(value['study_allocation'])
        weekend = checked(value['weekend_allocation'])
        pilot = checked(value['weekend_authority'])
        checked(value['candidate_registry'])
        if (study.get('status') != 'AUTHORIZED'
                or study_results.get('status') != 'AUTHORIZED_MACHINE_RESULT_RETENTION'
                or study_results.get('plan_sha256') != value['study_plan']['sha256']
                or allocation.get('status') != 'AUTHORIZED_EXCLUSIVE_ALLOCATION'
                or weekend.get('status') != 'AUTHORIZED'
                or pilot.get('allocation_sha256') != value['weekend_allocation']['sha256']
                or study.get('candidate_registry') != value['candidate_registry']):
            raise ValueError()
        roots = [Path(value[k]) for k in ('state_root','prediction_root','result_root')]
        protected = [Path(study['programme_root']),
                     Path(study_results['result_database']).parent,
                     *[Path(p) for p in study['prediction_output_roots']]]
        protected.extend(Path(weekend[k]) for k in ('state_root','prediction_root','result_root') if weekend.get(k))
        protected.extend(Path(pilot[k]) for k in ('state_root','prediction_root','result_root') if pilot.get(k))
        if not pilot.get('state_root') or not pilot.get('prediction_root'):
            raise ValueError()
        for path in roots:
            if not path.is_absolute() or path.resolve() != path:
                raise ValueError()
            if any(path == other or path.is_relative_to(other) or other.is_relative_to(path)
                   for other in protected):
                raise ValueError()
        if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
               for i,a in enumerate(roots) for b in roots[i+1:]):
            raise ValueError()
        return value
    except (KeyError, TypeError, ValueError, OSError):
        raise ValueError('invalid_incident_authority') from None


def incident_slot(authority, slot_id):
    rows = [s for s in authority['slots'] if s['id'] == slot_id]
    if len(rows) != 1:
        raise ValueError('invalid_incident_slot')
    return rows[0]


def incident_usage(value, campaign_id, *, authority_sha256=None, slot=None):
    """Subtract only hash-authenticated charges; never refund global counters."""
    totals = dict(capture_attempts=0, live_seconds=0, prediction=0, results=0, logical_requests=0)
    def validate(row):
        ref = row['incident_authority']
        authority = load_incident_authority(ref)
        if (authority['campaign_id'] != campaign_id
                or row.get('incident_authority_sha256') != ref['sha256']
                or row.get('incident_id') != authority['incident_id']
                or any(k in row for k in ('engineering_authority','development_authority_sha256'))):
            raise ValueError('invalid_incident_consumption')
        incident_slot(authority,row['incident_slot'])
        return ((authority_sha256 is None or ref['sha256']==authority_sha256)
                and (slot is None or row['incident_slot']==slot))
    def tagged(row):
        # Historical incident_adjustment_reference is unrelated, already charged
        # consumption. Only this profile's explicit authority tags select it.
        return bool({'incident_authority','incident_authority_sha256','incident_id','incident_slot'} & row.keys())
    for row in value.get('attempts',[]):
        if tagged(row) and validate(row): totals['capture_attempts'] += 1
    for row in value.get('launches',{}).values():
        if tagged(row) and validate(row): totals['live_seconds'] += row['charged_seconds']
    for key,row in value.get('incident_request_usage',{}).items():
        matches = validate(row)
        authority = load_incident_authority(row['incident_authority'])
        if (key != row['incident_authority_sha256']+':'+row['incident_slot']
                or set(row['counts']) != {'prediction','results'}
                or any(type(v) is not int or not 0<=v<=authority['max_prediction_logical_requests_per_window' if k=='prediction' else 'max_result_requests_per_window'] for k,v in row['counts'].items())):
            raise ValueError('invalid_incident_request_consumption')
        if matches:
            for kind in ('prediction','results'):totals[kind] += row['counts'][kind]
    totals['logical_requests'] = totals['prediction']+totals['results']
    return totals


def validate_incident_lease(row):
    authority = load_incident_authority(row['incident_authority'])
    slot = incident_slot(authority,row['incident_slot'])
    kind = row.get('incident_kind')
    start,end = row['authorized_at'],row['expires_at']
    bound = stamp(slot['ends_at'] if kind == 'prediction' else authority['result_deadline']).timestamp()
    limit = authority['max_source_operations_per_window' if kind == 'prediction' else 'max_result_operations_per_window']
    reference = authority['authority_reference']+':slot:'+slot['id']+('' if kind=='prediction' else ':results')
    if (kind not in {'prediction','results'} or row.get('prior_phase') != 'OPEN'
            or row.get('incident_authority_sha256') != row['incident_authority']['sha256']
            or row.get('incident_id') != authority['incident_id']
            or row.get('reference') != reference or row.get('max_operations') != limit
            or not stamp(authority['issued_at']).timestamp() <= start < end <= bound
            or start < stamp(slot['starts_at']).timestamp()-600
            or any(k in row for k in ('engineering_authority','development_authority','cooldown_revision'))):
        raise ValueError('invalid_incident_source_lease')
    return authority


def incident_source_usage(value, baseline_count):
    operations = value.get('operations',[])
    allocations = value.get('diagnostic_authorizations',[])
    recognized = set(); seen = set()
    for index,row in enumerate(allocations):
        if 'incident_authority' not in row:continue
        validate_incident_lease(row)
        identity = (row['incident_authority_sha256'],row['incident_slot'],row['incident_kind'])
        if identity in seen:raise ValueError('incident_source_slot_consumed')
        seen.add(identity)
        begin = row['operation_start']
        end = allocations[index+1]['operation_start'] if index+1<len(allocations) else len(operations)
        if type(begin) is not int or type(end) is not int or not 0<=begin<=end<=len(operations) or end-begin>row['max_operations']:
            raise ValueError('invalid_incident_source_accounting')
        for i in range(begin,end):
            op=operations[i]
            if (any(op.get(k)!=row[k] for k in ('incident_authority_sha256','incident_slot','incident_kind'))
                    or not row['authorized_at']<=op['at']<row['expires_at']):
                raise ValueError('invalid_incident_source_accounting')
            recognized.add(i)
    if any({'incident_authority','incident_authority_sha256','incident_id','incident_slot','incident_kind'} & row.keys()
           and i not in recognized for i,row in enumerate(operations)):
        raise ValueError('invalid_incident_source_accounting')
    return sum(i>=baseline_count for i in recognized)
