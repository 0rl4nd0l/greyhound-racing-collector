"""Prepare one immutable daily native package; no scheduler or provider calls.

The coordinator holds the campaign owner lock. This module briefly acquires the
native collector lock to reconcile prior consumption; it never steals ownership,
starts a campaign lease, grants source access, or changes an installed unit.
"""
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from race_collection.live_freshness_contract import (
    AttemptAllowance, FreshnessContract, create_once, digest, verify_source_package,
)
from race_collection.freshness_attempt_reconciliation import reconcile
from race_collection.persistent_authority import checked, load_persistent_allocation, load_standing_authority, stamp
from race_collection.persistent_capacity import calculate_capacity
from race_collection.synchronous_manual_capture import acquire_collector_lock_no_steal, release_owned_collector_lock
from scripts.check_freshness_runtime import verify_runtime
from scripts.prepare_freshness_rehearsal import prepare
from scripts.run_freshness_rehearsal import execution_contract


def _ref(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _configuration_identity(cfg):
    # Normalize the documented path interface without silently dropping any
    # coordinator binding. A changed config cannot adopt a prepared old day.
    value = {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}
    return digest(value)


def _verified_result(receipt_path, cfg, standing_ref, racing_date):
    receipt = json.loads(receipt_path.read_bytes())
    if (receipt.get('schema_version') != 'persistent_native_preparation_v1'
            or receipt.get('status') != 'PREPARED_NOT_STARTED'
            or receipt.get('configuration_sha256') != _configuration_identity(cfg)
            or receipt.get('standing_authority') != standing_ref
            or receipt.get('racing_date') != racing_date):
        raise ValueError('persistent_native_preparation_binding_changed')
    documents = {key: checked(receipt[key]) for key in
                 ('allocation', 'comparison', 'plan', 'contract', 'reconciliation', 'scope')}
    allocation = load_persistent_allocation(receipt['allocation'])
    plan, contract = documents['plan'], documents['contract']
    if (allocation['standing_authority'] != standing_ref or allocation['racing_date'] != racing_date
            or plan.get('persistent_allocation') != receipt['allocation']
            or plan.get('frozen_comparison') != receipt['comparison']
            or plan.get('racing_date') != racing_date
            or contract != documents['scope']
            or contract != execution_contract(plan, documents['reconciliation'])
            or Path(receipt['plan']['path']).parent != Path(receipt['output'])
            or receipt['contract']['path'] != str(Path(receipt['output'])/'contract.json')):
        raise ValueError('persistent_native_package_binding_changed')
    identity = verify_source_package(plan['source_root'], plan['source_identity_sha256'])
    if identity.get('commit') != plan['commit']:
        raise ValueError('persistent_native_source_commit_changed')
    if cfg.get('source_commit') and cfg['source_commit'] != plan['commit']:
        raise ValueError('persistent_native_installed_source_changed')
    if hashlib.sha256(Path(plan['python']).resolve().read_bytes()).hexdigest() != plan['python_sha256']:
        raise ValueError('persistent_native_python_changed')
    verify_runtime(plan)
    if hashlib.sha256((Path(receipt['output'])/'source.tar').read_bytes()).hexdigest() != plan['source_archive_sha256']:
        raise ValueError('persistent_native_source_archive_changed')
    for name, expected in plan['unit_sha256'].items():
        if hashlib.sha256((Path(receipt['output'])/'units'/name).read_bytes()).hexdigest() != expected:
            raise ValueError('persistent_native_packaged_unit_changed')
    scope = FreshnessContract(contract)
    if receipt['scope']['path'] != str(scope.session/'scope.json') or receipt['reconciliation']['path'] != str(scope.session/'reconciliation.json'):
        raise ValueError('persistent_native_scope_path_changed')
    # Re-read accounting validation only. initialize() is intentionally not
    # called on restart, even when the day has stopped or exhausted its budget.
    AttemptAllowance(scope)._accounting()
    return {'output': receipt['output'], 'plan': plan, 'contract': contract,
        'allocation_ref': receipt['allocation'], 'plan_path': receipt['plan']['path'],
        'contract_path': receipt['contract']['path'], 'receipt_ref': _ref(receipt_path)}


def prepare_day(cfg, standing_ref, racing_date, now):
    """Bootstrap today once, or return the exact verified existing package.

    New preparations leave a durable marker before any allocation/package write.
    A partial preparation is held for explicit diagnosis, never deleted/retried.
    The end bound is a finite bootstrap bound, not a claim that all future races
    fit: the coordinator must audit every actual inventory against it.
    """
    standing = load_standing_authority(standing_ref)
    zone = ZoneInfo('Australia/Melbourne')
    current = stamp(now) if isinstance(now, str) else now
    if not isinstance(current, datetime) or current.utcoffset() is None:
        raise ValueError('persistent_native_time_ambiguous')
    current = current.astimezone(timezone.utc)
    day = date.fromisoformat(racing_date)
    if day.isoformat() != racing_date:
        raise ValueError('persistent_native_source_date_invalid')
    dayroot = Path(standing['state_root'])/'days'/racing_date
    receipt_path = dayroot/'native-prepared.json'
    if receipt_path.exists():
        # Prior source-date work may still own an after-midnight interval.
        # Return its exact receipts for resume/drain; never invent today's copy.
        return _verified_result(receipt_path, cfg, standing_ref, racing_date)
    if current.astimezone(zone).date() != day:
        raise ValueError('persistent_native_source_date_not_today')
    if dayroot.exists():
        raise ValueError('persistent_native_preparation_incomplete_preserved')
    start = current + timedelta(seconds=120)
    if start.astimezone(zone).date() != day:
        raise ValueError('persistent_native_start_crosses_source_date')
    next_date = day + timedelta(days=1)
    end = min(datetime.combine(next_date,time(1,20),tzinfo=zone).astimezone(timezone.utc), start+timedelta(hours=25))
    if not start < end:
        raise ValueError('persistent_native_no_collection_interval')
    cleanup = end + timedelta(seconds=1860)
    capacity = calculate_capacity(start,end)
    if any(capacity['caps'][key] > standing['daily_caps'][key] for key in capacity['caps']):
        raise ValueError('persistent_native_standing_capacity_insufficient')
    # A unique day/authority basename is also the native rehearsal/session key.
    package = dayroot / ('native-'+racing_date+'-'+standing_ref['sha256'][:12])
    dayroot.mkdir(parents=True,exist_ok=False,mode=0o700)
    create_once(dayroot/'preparation-started.json',dict(
        status='PREPARING_CONSUMED_NO_AUTOMATIC_RETRY',at=current.isoformat(),
        standing_authority=standing_ref,racing_date=racing_date,
        configuration_sha256=_configuration_identity(cfg)))
    create_once(dayroot/'capacity.json',capacity)
    allocation = dict(schema_version='collector_persistent_daily_allocation_v1',
        status='AUTHORIZED_PERSISTENT_DAILY_ENGINEERING', standing_authority=standing_ref,
        allocation_id=standing_ref['sha256']+':'+racing_date, racing_date=racing_date,
        issued_at=current.isoformat(),starts_at=start.isoformat(),ends_at=end.isoformat(),
        cleanup_by=cleanup.isoformat(),state_root=str(dayroot),
        prediction_root=str(Path(standing['prediction_root'])/'days'/racing_date),
        caps=capacity['caps'],limits_basis=_ref(dayroot/'capacity.json'))
    create_once(dayroot/'allocation.json',allocation)
    allocation_ref=_ref(dayroot/'allocation.json')
    load_persistent_allocation(allocation_ref)
    study=checked(standing['study_plan'])
    comparison={**study,'status':'AUTHORIZED_ENGINEERING','persistent_allocation':allocation_ref,
        'authority_reference':standing['authority_reference'],'activated_at':current.isoformat(),
        'starts_at':start.isoformat(),'ends_at':cleanup.isoformat(),
        'programme_root':str(dayroot/'admission'),
        'prediction_output_roots':[str(Path(allocation['prediction_root'])/'bundles')],
        'performance_evaluation':False,'study_enrolment':False,'race_ids':None}
    create_once(dayroot/'comparison.json',comparison)
    # Existing native preparation resolves this setting from the environment.
    # Confine the adjustment to preparation; never change a service environment.
    prior_source=os.environ.get('GREYHOUND_SPORTSBET_ACCESS_STATE')
    os.environ['GREYHOUND_SPORTSBET_ACCESS_STATE']=str(cfg['source_state'])
    try:
        result=prepare(output=package,start=start,python=Path(cfg['python']),
            db=Path(cfg['history_database']),lock=Path(cfg['lock_path']),
            reconciliation_roots=cfg['reconciliation_roots'],installed_dir=Path(cfg['installed_dir']),
            campaign_root=Path(cfg['campaign_root']),operational_predictions=True,
            comparison_plan=dayroot/'comparison.json',prediction_root=Path(allocation['prediction_root']),
            persistent_allocation=allocation_ref)
    finally:
        if prior_source is None:os.environ.pop('GREYHOUND_SPORTSBET_ACCESS_STATE',None)
        else:os.environ['GREYHOUND_SPORTSBET_ACCESS_STATE']=prior_source
    plan=checked({'path':result['plan'],'sha256':result['plan_sha256']})
    owned=acquire_collector_lock_no_steal(Path(cfg['lock_path']),run_id=plan['rehearsal_id'],
        output_dir=package,phase='persistent_daily_reconciliation')
    try:
        accounting=reconcile(roots=cfg['reconciliation_roots'],db_path=Path(cfg['history_database']),
            source_date=racing_date,lock_path=Path(cfg['lock_path']),owner_run_id=plan['rehearsal_id'])
        contract=execution_contract(plan,accounting)
        scope=FreshnessContract(contract)
        AttemptAllowance(scope).initialize(accounting)
        create_once(package/'contract.json',contract)
    finally:
        release_owned_collector_lock(owned)
    receipt=dict(schema_version='persistent_native_preparation_v1',status='PREPARED_NOT_STARTED',
        at=current.isoformat(),standing_authority=standing_ref,racing_date=racing_date,
        configuration_sha256=_configuration_identity(cfg),output=str(package),
        allocation=allocation_ref,comparison=_ref(dayroot/'comparison.json'),
        plan=_ref(Path(result['plan'])),contract=_ref(package/'contract.json'),
        reconciliation=_ref(scope.session/'reconciliation.json'),scope=_ref(scope.session/'scope.json'))
    create_once(receipt_path,receipt)
    return _verified_result(receipt_path,cfg,standing_ref,racing_date)
