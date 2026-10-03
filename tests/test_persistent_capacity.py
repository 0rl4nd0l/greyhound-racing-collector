"""Finite daily sizing stays distinct from provider and result authority."""
import copy
import pytest
from race_collection.persistent_capacity import calculate_capacity, audit_inventory_capacity


def test_selected_limit16_uses113_not_four_race29():
    value=calculate_capacity('2026-10-03T09:00:00+10:00','2026-10-03T21:00:00+10:00')
    c=value['calculation'];caps=value['caps']
    assert c['selected_refresh_python_requests']==113
    assert c['planned_refreshes']==772
    assert c['planned_inventory_scans']==50
    assert caps==dict(max_python_requests=122000,max_browser_navigations=510,
                     max_capture_attempts=255,max_source_operations=1216,max_result_requests=0)
    assert c['combined_prediction_logical_requests']==122510
    assert c['local_allowance_establishes_provider_permission'] is False
    assert c['result_access_authorized'] is False


def test_utc_duration_accounts_for_spring_dst():
    value=calculate_capacity('2026-10-04T00:00:00+10:00','2026-10-05T00:00:00+11:00')
    assert value['calculation']['utc_scope_seconds']==23*3600
    assert value['calculation']['minute_refreshes']==1380


def test_utc_duration_accounts_for_fall_dst():
    value=calculate_capacity('2027-04-04T00:00:00+11:00','2027-04-05T00:00:00+10:00')
    assert value['calculation']['utc_scope_seconds']==25*3600


@pytest.mark.parametrize('start,end',[
    ('2026-10-03T00:00:00','2026-10-04T00:00:00+10:00'),
    ('2026-10-03T00:00:00+10:00','2026-10-03T00:00:00+10:00'),
    ('2026-10-03T00:00:00+10:00','2026-10-04T03:01:00+11:00')])
def test_invalid_or_over26h_duration_rejected(start,end):
    with pytest.raises(ValueError):calculate_capacity(start,end)


@pytest.mark.parametrize('selected,races',[(0,204),(17,204),(True,204),(16,0),(16,float('inf'))])
def test_nonfinite_or_unsupported_workload_rejected(selected,races):
    with pytest.raises(ValueError):calculate_capacity('2026-10-03T00:00:00+10:00','2026-10-03T01:00:00+10:00',selected,races)


def test_late_race_triggers_hold_without_modifying_caps():
    value=calculate_capacity('2026-10-03T09:00:00+10:00','2026-10-04T01:20:00+10:00');before=copy.deepcopy(value)
    audit=audit_inventory_capacity(value,['2026-10-04T00:13:00+10:00','2026-10-04T01:21:00+10:00'])
    assert audit['status']=='HOLD' and audit['reasons']==['KNOWN_RACE_OUTSIDE_ALLOCATION']
    assert value==before


def test_timing_inventory_union_handles_empty_overlap_and_overnight():
    value=calculate_capacity('2026-10-03T09:00:00+10:00','2026-10-04T01:20:00+10:00')
    assert audit_inventory_capacity(value,[])['status']=='INVENTORY_FITS_FINITE_BOOTSTRAP'
    a=audit_inventory_capacity(value,['2026-10-03T08:30:00+10:00',
        '2026-10-03T10:00:00+10:00','2026-10-03T10:30:00+10:00','2026-10-04T00:13:00+10:00'])
    assert a['active_interval_seconds']==150*60
    assert a['already_past_at_allocation_start']==1
    assert a['remaining_opportunities']==3
    assert a['issued_caps_unchanged'] is True


def test_inventory_excess_cannot_silently_expand_capture_allowance():
    value=calculate_capacity('2026-10-03T09:00:00+10:00','2026-10-03T21:00:00+10:00',race_baseline=2)
    a=audit_inventory_capacity(value,['2026-10-03T10:00:00+10:00']*4)
    assert a['status']=='HOLD' and a['reasons']==['KNOWN_RACES_EXCEED_CAPTURE_CAP']
