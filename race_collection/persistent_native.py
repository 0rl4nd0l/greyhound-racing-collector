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


def _checked_reconciliation(ref):
    """Historical evidence inventories are larger than authority documents.

    Keep the small authority limit unchanged. This separate read still pins
    every byte and subsequently exercises native reconciliation validation.
    """
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('persistent_reconciliation_reference_invalid')
    path = Path(ref['path'])
    if (not path.is_absolute() or path.resolve() != path or not path.is_file()
            or path.stat().st_size > 64 * 1024 * 1024):
        raise ValueError('persistent_reconciliation_reference_unsafe')
    raw = path.read_bytes()
    if len(raw) > 64 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('persistent_reconciliation_changed')
    return json.loads(raw)


def _verified_result(receipt_path, cfg, standing_ref, racing_date):
    receipt = json.loads(receipt_path.read_bytes())
    if (receipt.get('schema_version') != 'persistent_native_preparation_v1'
            or receipt.get('status') != 'PREPARED_NOT_STARTED'
            or receipt.get('configuration_sha256') != _configuration_identity(cfg)
            or receipt.get('standing_authority') != standing_ref
            or receipt.get('racing_date') != racing_date):
        raise ValueError('persistent_native_preparation_binding_changed')
    documents = {key: checked(receipt[key]) for key in
                 ('allocation', 'comparison', 'plan', 'contract', 'scope')}
    documents['reconciliation'] = _checked_reconciliation(receipt['reconciliation'])
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
    if cfg.get('recovery_selection'):
        selection = checked(cfg['recovery_selection'])
        if selection.get('racing_date') == racing_date:
            return prepare_recovery(cfg, standing_ref, racing_date, now)
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


def recovery_baseline(prepared, cfg):
    """Read-only snapshot for a reviewed, same-allocation recovery selection.

    Call after the owner has verified cleanup and closed the old campaign lease.
    This does not close a lease, forgive a failure or grant any allowance.
    """
    from race_collection.persistent_authority import persistent_usage
    from race_collection.operational_prediction import require_completed_lifetimes
    from utils.sportsbet_access import SportsbetAccess
    scope = FreshnessContract(prepared['contract'])
    output = Path(prepared['output'])
    require_completed_lifetimes(output, scope.campaign)
    owner = json.loads((output/'persistent-owner-state.json').read_bytes())
    if any(row.get('returncode') is None for row in owner['dispatches']):
        raise ValueError('persistent_recovery_unfinished_dispatch')
    ledger = json.loads((Path(cfg['campaign_root'])/'ledger.json').read_bytes())
    launch = ledger['launches'][prepared['plan']['rehearsal_id']]
    if not launch.get('closed_at') or launch.get('persistent_allocation') != prepared['allocation_ref']:
        raise ValueError('persistent_recovery_prior_lease_not_closed')
    if ledger.get('source_holds') or Path(cfg['lock_path']).exists():
        raise ValueError('persistent_recovery_shared_hold_or_owner')
    source = SportsbetAccess(cfg['source_state']).read()
    grant = source.get('diagnostic_authority') or {}
    if (source['phase'] != 'OPEN' or source['active'] is not None
            or source['access_basis']['status'] != 'permitted'
            or grant.get('persistent_allocation') != prepared['allocation_ref']):
        raise ValueError('persistent_recovery_source_not_idle_or_bound')
    used = persistent_usage(ledger, scope.campaign.value['campaign_id'],
        allocation_sha256=prepared['allocation_ref']['sha256'])
    # All previous attempt records and source operations remain prefix-bound.
    records = []
    for directory in (Path(prepared['plan']['prediction_root'])/'races').glob('*'):
        for name in ('identity.json', 'job.json', 'terminal.json'):
            if (directory/name).is_file(): records.append(_ref(directory/name))
    return dict(closed_launch=launch, usage=used, source_grant_sha256=digest(grant),
        prior_owner_state=_ref(output/'persistent-owner-state.json'),
        preserved_prediction_records=sorted(records, key=lambda r:r['path']),
        source_operation_count=len(source.get('operations', [])), source_operations_sha256=digest(source.get('operations', [])),
        attempt_count=len(ledger['attempts']), attempts_sha256=digest(ledger['attempts']))




def _checked_recovery_refresh_report(ref):
    """Bounded acquisition metadata, distinct from the 256 KiB authority reader."""
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('persistent_recovery_report_reference_invalid')
    path = Path(ref['path'])
    limit = 4 * 1024 * 1024
    if (not path.is_absolute() or path.resolve() != path or not path.is_file()
            or path.stat().st_size > limit):
        raise ValueError('persistent_recovery_report_reference_unsafe')
    raw = path.read_bytes()
    if len(raw) > limit or hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('persistent_recovery_report_changed')
    return json.loads(raw)


def _reviewed_refresh_failure(selection, prior, halt, stop):
    """Authenticate explicit prospective repair; old refresh remains failed.

    A bare schedule-change error cannot prove that a race-local exclusion was
    safe. Only a separately reviewed selection may restart a corrected package;
    the old terminal, checkpoint and incomplete publication remain immutable.
    """
    review = checked(selection['reviewed_failure'])
    if (review.get('schema_version') != 'persistent_reviewed_refresh_failure_v1'
            or review.get('disposition') != 'PROSPECTIVE_CORRECTION_OLD_FAILURE_UNRESOLVED'
            or not review.get('authority_reference')
            or review.get('source_commit') != selection['source_commit']
            or review.get('cleanup') != selection['cleanup']
            or halt.get('reason') != 'persistent_native_terminal_failure:LIVE_PHASE_FAILED'
            or stop.get('reason') != 'LIVE_PHASE_FAILED'):
        raise ValueError('persistent_recovery_refresh_review_invalid')
    terminal, lifecycle, checkpoint, phase = (
        checked(review[key]) for key in
        ('service_terminal','service_lifecycle','checkpoint','phase_result'))
    report = _checked_recovery_refresh_report(review['refresh_report'])
    invocation = terminal.get('invocation_id', '')
    evidence = Path(prior['plan']['evidence_root'])
    runtime = evidence/'shadow_autopilot_daemon_runtime'
    cycle = Path(terminal['output_dir'])
    state = checked(selection['baseline']['prior_owner_state'])
    failed = [row for row in state['dispatches'] if row.get('native_disposition') == 'FAILED_OR_UNVERIFIED']
    if (len(invocation) != 32 or any(c not in '0123456789abcdef' for c in invocation)
            or len(failed) != 1 or failed[0].get('invocation_id') != invocation
            or failed[0].get('returncode') != 2
            or terminal.get('allocation_sha256') != prior['allocation_ref']['sha256']
            or terminal.get('status') != 'FAILED' or terminal.get('runtime_action') != 'LIVE_PHASE_FAILED'
            or terminal.get('final_verdict') != 'NEEDS_MORE_AUTOMATION'
            or lifecycle.get('invocation_id') != invocation or lifecycle.get('status') != 'COMPLETE'
            or lifecycle.get('returncode') != 2 or lifecycle.get('children_reaped') is not True
            or lifecycle.get('interrupted', False) is not False
            or Path(review['service_terminal']['path']) != runtime/'service-terminals'/(invocation+'.json')
            or Path(review['service_lifecycle']['path']) != runtime/'service-lifecycles'/(invocation+'.json')
            or cycle.parent != evidence or cycle.name != 'shadow_autopilot_daemonization_v1_'+terminal['run_id']
            or Path(review['checkpoint']['path']) != cycle/'phase-checkpoint.json'
            or checkpoint.get('schema_version') != 'collector_live_phase_checkpoint_v1'
            or checkpoint.get('status') != 'LIVE_PHASE_FAILED'
            or checkpoint.get('cycle_id') != terminal['run_id'] or checkpoint.get('output_dir') != str(cycle)):
        raise ValueError('persistent_recovery_refresh_terminal_unverified')
    phases = checkpoint['phases']
    if (len(phases) != 1 or phases[0].get('number') != 0 or phases[0].get('kind') != 'refresh'
            or phases[0].get('status') != 'COMPLETE' or phases[0].get('budget_exceeded') is not False
            or {'path':phases[0]['result_path'], 'sha256':phases[0]['result_sha256']} != review['phase_result']
            or Path(review['phase_result']['path']) != cycle/'phase-0-result.json'
            or phase.get('status') != 'FAIL' or phase.get('collection_phase') != 'refresh'
            or phase.get('final_verdict') != 'COLLECTION_PHASE_BLOCKED'
            or phase.get('run_id') != terminal['run_id']+'_phase_0'
            or Path(phase['output_dir']) != evidence/('shadow_autopilot_v1_'+phase['run_id'])
            or Path(review['refresh_report']['path']) != Path(phase['output_dir'])/'odds_capture_refresh_report.json'
            or phase.get('current_race_index_publish', {}).get('status') != 'REJECTED'
            or phase['current_race_index_publish'].get('reason') != 'CURRENT_INDEX_SOURCE_INVALID'
            or phase['current_race_index_publish'].get('source_refresh_report_path') != review['refresh_report']['path']
            or report.get('status') != 'ACQUISITION_INCOMPLETE'
            or report.get('reason') != 'unisolated_selected_race_acquisition_failure'
            or not any(row.get('success') is False and row.get('result', {}).get('success') is False
                and row['result'].get('error') == 'discovery_canonical_jump_changed'
                for row in report.get('downloads', []))):
        raise ValueError('persistent_recovery_refresh_checkpoint_unverified')


def prepare_recovery(cfg, standing_ref, racing_date, now):
    """Select one explicitly reviewed successor package under the SAME grant.

    The root coordinator must hold campaign owner.lock during preparation, as
    for prepare_day. A unique durable marker forbids retrying partial package
    preparation. Selection is configuration-pinned; no stopped scope is reopened.
    """
    reference = cfg['recovery_selection']
    selection = checked(reference)
    old_cfg = checked(selection['prior_configuration'])
    checked(selection['prior_preparation'])
    if (selection.get('schema_version') != 'persistent_recovery_selection_v1'
            or selection.get('status') != 'AUTHORIZED_SAME_ALLOCATION_RECOVERY'
            or not selection.get('authority_reference')
            or selection.get('racing_date') != racing_date
            or selection.get('source_commit') != cfg.get('source_commit')
            or cfg.get('source_commit') == old_cfg.get('source_commit')
            or {k:v for k,v in cfg.items() if k not in ('source_commit','recovery_selection')}
                != {k:v for k,v in old_cfg.items() if k not in ('source_commit','recovery_selection')}):
        raise ValueError('persistent_recovery_selection_invalid')
    prior = _verified_result(Path(selection['prior_preparation']['path']), old_cfg, standing_ref, racing_date)
    cleanup = checked(selection['cleanup'])
    prior_output = Path(prior['output'])
    halt = checked(cleanup['halt'])
    stop = checked(selection['prior_stop'])
    old_scope = FreshnessContract(prior['contract'])
    if (cleanup.get('status') != 'FAILED_DRAINED_LEASE_CLOSED'
            or cleanup.get('allocation') != prior['allocation_ref']
            or cleanup.get('package') != str(prior_output)
            or cleanup.get('plan') != _ref(Path(prior['plan_path']))
            or any(cleanup.get(k) is not True for k in ('all_dispatches_reaped',
                'prediction_lifetimes_complete','collector_lock_absent','consumption_preserved','failed_attempt_preserved'))
            or cleanup.get('source_active') is not False
            or cleanup['halt']['path'] != str(prior_output/'HALT.json')
            or selection['prior_stop']['path'] != str(old_scope.session/'STOP.json')
            ):
        raise ValueError('persistent_recovery_cleanup_unverified')
    if selection.get('reviewed_failure') is not None:
        _reviewed_refresh_failure(selection, prior, halt, stop)
    elif (halt.get('reason') != 'operational_prediction_failed_preserved_consumption'
            or stop.get('reason') != 'PERSISTENT_OWNER_FAILURE'):
        raise ValueError('persistent_recovery_cleanup_unverified')
    allocation = load_persistent_allocation(prior['allocation_ref'])
    dayroot = Path(allocation['state_root'])
    recovery_root = dayroot/'recoveries'/reference['sha256']
    receipt_path = recovery_root/'native-prepared.json'
    baseline = selection['baseline']
    live = recovery_baseline(prior, cfg)
    if cleanup['closed_launch'] != baseline['closed_launch'] or live['closed_launch'] != baseline['closed_launch']:
        raise ValueError('persistent_recovery_closed_lease_changed')
    source = json.loads(Path(cfg['source_state']).read_bytes())
    ledger = json.loads((Path(cfg['campaign_root'])/'ledger.json').read_bytes())
    checked(baseline['prior_owner_state'])
    for reference_to_preserve in baseline['preserved_prediction_records']:
        checked(reference_to_preserve)
    if (live['source_grant_sha256'] != baseline['source_grant_sha256']
            or digest(source.get('operations', [])[:baseline['source_operation_count']]) != baseline['source_operations_sha256']
            or digest(ledger['attempts'][:baseline['attempt_count']]) != baseline['attempts_sha256']
            or any(live['usage'][k] < baseline['usage'][k] for k in
                ('python','browser','results','capture_attempts','live_seconds'))):
        raise ValueError('persistent_recovery_consumption_changed')
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_bytes())
        if receipt.get('recovery_selection') != reference:
            raise ValueError('persistent_recovery_receipt_changed')
        return _verified_result(receipt_path, cfg, standing_ref, racing_date)
    if live != baseline or recovery_root.exists():
        raise ValueError('persistent_recovery_partial_or_baseline_changed')
    current = stamp(now) if isinstance(now, str) else now
    if not stamp(allocation['starts_at']) <= current < stamp(allocation['ends_at'])-timedelta(seconds=600):
        raise ValueError('persistent_recovery_window_unavailable')
    if any(not row.get('closed_at') for row in ledger['launches'].values()):
        raise ValueError('persistent_recovery_campaign_owner_active')
    recovery_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    create_once(recovery_root/'preparation-started.json', dict(
        status='RECOVERY_PREPARING_CONSUMED_NO_AUTOMATIC_RETRY',
        at=current.isoformat(), recovery_selection=reference))
    root_health = Path(load_standing_authority(standing_ref)['state_root'])/'health.json'
    if root_health.exists():
        create_once(recovery_root/'prior-root-health.json', json.loads(root_health.read_bytes()))
    package = recovery_root/('native-recovery-'+racing_date+'-'+reference['sha256'][:12])
    comparison = json.loads(Path(selection['prior_preparation']['path']).read_bytes())['comparison']
    prior_source = os.environ.get('GREYHOUND_SPORTSBET_ACCESS_STATE')
    os.environ['GREYHOUND_SPORTSBET_ACCESS_STATE'] = str(cfg['source_state'])
    try:
        result = prepare(output=package, start=stamp(allocation['starts_at']), python=Path(cfg['python']),
            db=Path(cfg['history_database']), lock=Path(cfg['lock_path']),
            reconciliation_roots=cfg['reconciliation_roots'], installed_dir=Path(cfg['installed_dir']),
            campaign_root=Path(cfg['campaign_root']), operational_predictions=True,
            comparison_plan=Path(comparison['path']), prediction_root=Path(allocation['prediction_root']),
            persistent_allocation=prior['allocation_ref'])
    finally:
        if prior_source is None: os.environ.pop('GREYHOUND_SPORTSBET_ACCESS_STATE', None)
        else: os.environ['GREYHOUND_SPORTSBET_ACCESS_STATE'] = prior_source
    plan = checked({'path':result['plan'], 'sha256':result['plan_sha256']})
    owned = acquire_collector_lock_no_steal(Path(cfg['lock_path']), run_id=plan['rehearsal_id'],
        output_dir=package, phase='persistent_recovery_reconciliation')
    try:
        accounting = reconcile(roots=cfg['reconciliation_roots'], db_path=Path(cfg['history_database']),
            source_date=racing_date, lock_path=Path(cfg['lock_path']), owner_run_id=plan['rehearsal_id'])
        contract = execution_contract(plan, accounting)
        scope = FreshnessContract(contract)
        AttemptAllowance(scope).initialize(accounting)
        create_once(package/'contract.json', contract)
    finally:
        release_owned_collector_lock(owned)
    # Preserve cadence, cumulative lane counts and every old dispatch in its
    # original file. The shared prediction roots already exclude failed races.
    old_state = json.loads((prior_output/'persistent-owner-state.json').read_bytes())
    create_once(package/'persistent-owner-state.json', dict(schema_version='persistent_owner_state_v1',
        dispatches=[], inventory=None, completed_lanes=old_state['completed_lanes'],
        deferred_lanes=old_state.get('deferred_lanes', {'full':0,'odds':0}),
        refresh_failures=[], next_due_at=old_state.get('next_due_at', {}),
        prior_owner_state=_ref(prior_output/'persistent-owner-state.json'), recovery_selection=reference))
    receipt = dict(schema_version='persistent_native_preparation_v1', status='PREPARED_NOT_STARTED',
        at=current.isoformat(), standing_authority=standing_ref, racing_date=racing_date,
        configuration_sha256=_configuration_identity(cfg), output=str(package),
        allocation=prior['allocation_ref'], comparison=comparison, plan=_ref(Path(result['plan'])),
        contract=_ref(package/'contract.json'), reconciliation=_ref(scope.session/'reconciliation.json'),
        scope=_ref(scope.session/'scope.json'), recovery_selection=reference,
        prior_root_health=(_ref(recovery_root/'prior-root-health.json')
            if (recovery_root/'prior-root-health.json').exists() else None))
    create_once(receipt_path, receipt)
    return _verified_result(receipt_path, cfg, standing_ref, racing_date)
