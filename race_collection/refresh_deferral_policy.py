"""Finite scheduled outage accounting; never grants requests or source access."""
from pathlib import Path
from datetime import datetime

from race_collection.persistent_authority import checked, load_persistent_allocation
from race_collection.persistent_capacity import calculate_capacity


def policy_for(scope):
    reference = scope.get('persistent_allocation')
    if reference is None:
        return None
    allocation = load_persistent_allocation(reference)
    capacity_ref = allocation.get('limits_basis')
    capacity = checked(capacity_ref)
    calculation = capacity['calculation']
    expected = calculate_capacity(allocation['starts_at'], allocation['ends_at'],
        selected_limit=calculation['selected_limit'], race_baseline=calculation['race_baseline'])
    if (Path(capacity_ref['path']) != Path(reference['path']).parent/'capacity.json'
            or capacity != expected or capacity['caps'] != allocation['caps']):
        raise ValueError('persistent_refresh_capacity_unverified')
    limit = min(allocation['caps']['max_source_operations'],
        calculation['planned_refreshes'] + calculation['source_operation_recovery_allowance'])
    return dict(schema_version='persistent_scheduled_refresh_deferral_policy_v1',
        allocation=reference, capacity=capacity_ref, standing_authority=allocation['standing_authority'],
        maximum_failed_cycles=limit, request_retries_added=0,
        accounting='cumulative_failed_cycles_preserved', provider_permission_extended=False)


def limit_for(scope):
    policy = policy_for(scope)
    return policy['maximum_failed_cycles'] if policy else 2


def record_fields(classified, scope):
    policy = policy_for(scope)
    if policy is None:
        return dict(classified)
    return {**classified, 'maximum_failed_cycles': policy['maximum_failed_cycles'],
        'refresh_deferral_policy': policy, 'allocation_sha256': policy['allocation']['sha256']}


def verify_record(value, classified, *, allocation_sha=None):
    """Replay old classification without rewriting its original policy fields."""
    if classified is None or any(value.get(k) != v for k, v in classified.items()
                                 if k != 'maximum_failed_cycles'):
        raise ValueError('refresh_deferral_classification_changed')
    policy = value.get('refresh_deferral_policy')
    if policy is None:
        limit = classified['maximum_failed_cycles']
    else:
        expected = policy_for({'persistent_allocation': policy.get('allocation')})
        if (expected is None or policy != expected
                or value.get('allocation_sha256') != policy['allocation']['sha256']
                or allocation_sha is not None and policy['allocation']['sha256'] != allocation_sha):
            raise ValueError('persistent_refresh_policy_changed')
        limit = policy['maximum_failed_cycles']
    if (value.get('maximum_failed_cycles') != limit
            or type(value.get('failed_cycle_count')) is not int
            or not 1 <= value['failed_cycle_count'] <= limit):
        raise ValueError('refresh_deferral_count_unverified')
    return value


def index_allows_scheduled_wait(read_view, current, *, persistent):
    """Readiness is never granted here; strict provenance failures still escape."""
    from race_collection.synchronous_manual_capture import CaptureOneRejected
    try:
        view = read_view()
    except CaptureOneRejected as exc:
        if persistent and exc.code in {'CURRENT_INDEX_UNAVAILABLE', 'CURRENT_INDEX_STALE', 'DISCOVERY_TIMEOUT'}:
            return True
        raise
    observed = datetime.fromisoformat(view.source_generated_at)
    if observed.utcoffset() is None or current.utcoffset() is None:
        return False
    age = (current - observed).total_seconds()
    return age >= 0 if persistent else 0 <= age < 270
