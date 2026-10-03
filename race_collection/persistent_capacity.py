"""Pure, finite bootstrap capacity calculation; grants no provider permission."""
from datetime import datetime, timezone
import math


def _instant(value):
    instant = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(instant, datetime) or instant.utcoffset() is None:
        raise ValueError('persistent_capacity_requires_aware_instants')
    return instant.astimezone(timezone.utc)


def calculate_capacity(start, end, selected_limit=16, race_baseline=204):
    """Bound a daily allocation before discovery using all-active workload.

    Inventory reuse must already be installed: every selected refresh is bounded
    at seven Python requests per selected race plus one shared source snapshot.
    Inventory acquisition is independently costed at its observed 192 envelope.
    Runtime request counters remain authoritative if actual demand exceeds this
    sizing; exhausting a cap never permits a partial successful publication.
    """
    start, end = _instant(start), _instant(end)
    seconds = (end - start).total_seconds()
    if not 0 < seconds <= 26 * 3600:
        raise ValueError('persistent_capacity_duration_out_of_range')
    if (type(selected_limit) is not int or not 1 <= selected_limit <= 16
            or type(race_baseline) is not int or race_baseline <= 0):
        raise ValueError('persistent_capacity_workload_invalid')
    # Full and minute lanes may replace one another. Counting both is deliberate
    # bootstrap headroom until the actual inventory determines active intervals.
    minute_refreshes = math.ceil(seconds / 60)
    full_refreshes = math.ceil(seconds / 900)
    refreshes = minute_refreshes + full_refreshes + 4
    inventory_scans = math.ceil(seconds / 900) + 2
    capture_attempts = (race_baseline * 125 + 99) // 100
    selected_refresh_requests = 7 * selected_limit + 1
    selected_requests = refreshes * selected_refresh_requests
    discovery_requests = inventory_scans * 192
    capture_requests = 20 * capture_attempts
    python_subtotal = selected_requests + discovery_requests + capture_requests + 4000
    # Integer percentage arithmetic keeps large finite budgets reproducible.
    python_with_headroom = (python_subtotal * 115 + 99) // 100
    python_cap = ((python_with_headroom + 499) // 500) * 500
    source_subtotal = refreshes + capture_attempts + 20
    source_with_headroom = (source_subtotal * 115 + 99) // 100
    source_cap = ((source_with_headroom + 31) // 32) * 32
    caps = dict(max_python_requests=python_cap,
        max_browser_navigations=2 * capture_attempts,
        max_capture_attempts=capture_attempts,
        max_source_operations=source_cap, max_result_requests=0)
    calculation = dict(schema_version='persistent_capacity_calculation_v1',
        starts_at=start.isoformat(), ends_at=end.isoformat(), utc_scope_seconds=seconds,
        bootstrap_inventory_known=False, bootstrap_all_active=True,
        race_baseline=race_baseline, largest_retained_inventory=154,
        capture_headroom_percent=25, selected_limit=selected_limit,
        selected_race_python_requests=7, shared_snapshot_python_requests=1,
        selected_refresh_python_requests=selected_refresh_requests,
        minute_refreshes=minute_refreshes, full_refreshes=full_refreshes,
        setup_drain_refreshes=4, planned_refreshes=refreshes,
        inventory_cadence_seconds_active=900, inventory_cadence_seconds_idle=1800,
        inventory_scan_headroom=2, planned_inventory_scans=inventory_scans,
        discovery_python_requests_per_scan=192,
        selected_refresh_python_total=selected_requests, inventory_python_total=discovery_requests,
        capture_python_per_attempt=20, capture_python_total=capture_requests,
        permitted_recovery_python_allowance=4000, python_subtotal=python_subtotal,
        request_headroom_percent=15, python_with_headroom=python_with_headroom,
        python_rounding_multiple=500, source_operation_subtotal=source_subtotal,
        source_operation_recovery_allowance=20, source_with_headroom=source_with_headroom,
        source_rounding_multiple=32, browser_navigations_per_capture=2,
        combined_prediction_logical_requests=python_cap+2*capture_attempts,
        local_allowance_establishes_provider_permission=False,
        result_access_authorized=False, after_inventory_caps_mutable=False,
        limitation='Measured request envelope, not a provider maximum; actual dispatch caps and source controls remain binding.')
    return {'caps': caps, 'calculation': calculation}


def audit_inventory_capacity(capacity, jump_instants):
    """Audit a complete dated inventory; never amend its issued allocation.

    The caller owns completeness and identity checks and supplies each distinct
    engineering opportunity once. Timing outside the allocation requires a
    visible hold, even when the race has not entered the capture horizon yet.
    """
    calc, caps = capacity['calculation'], capacity['caps']
    start, end = _instant(calc['starts_at']), _instant(calc['ends_at'])
    jumps = [_instant(value) for value in jump_instants]
    future = [jump for jump in jumps if jump >= start]
    after_end = sum(jump >= end for jump in future)
    reasons = []
    if after_end:
        reasons.append('KNOWN_RACE_OUTSIDE_ALLOCATION')
    if len(future) > caps['max_capture_attempts']:
        reasons.append('KNOWN_RACES_EXCEED_CAPTURE_CAP')
    # Union of active one-hour lead-ins, clipped to allocation bounds.
    from datetime import timedelta
    intervals = sorted((max(start,jump-timedelta(hours=1)),min(end,jump))
                       for jump in future if max(start,jump-timedelta(hours=1)) < min(end,jump))
    merged = []
    for begin, finish in intervals:
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0],max(merged[-1][1],finish))
        else:
            merged.append((begin,finish))
    active = sum((finish-begin).total_seconds() for begin,finish in merged)
    idle = max(0,calc['utc_scope_seconds']-active)
    return dict(status='HOLD' if reasons else 'INVENTORY_FITS_FINITE_BOOTSTRAP',
        reasons=reasons, opportunity_count=len(jumps), remaining_opportunities=len(future),
        already_past_at_allocation_start=len(jumps)-len(future), races_at_or_after_scope_end=after_end,
        active_interval_seconds=active, idle_interval_seconds=idle,
        recommended_inventory_scans=math.ceil(active/900)+math.ceil(idle/1800)+2,
        issued_caps_unchanged=True, result_access_authorized=False)
