"""One restartable owner of daily discovery and the native acquisition lanes.

This module supervises existing capture, verification and prediction entrypoints.
It never resets a campaign, repairs a source hold, or enrols engineering in study.
"""
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
from zoneinfo import ZoneInfo

from race_collection.live_freshness_contract import create_once, FreshnessContract, AttemptAllowance
from race_collection.live_phase_checkpoint import atomic_json
from race_collection.persistent_authority import checked, load_standing_authority, stamp

ZONE = ZoneInfo('Australia/Melbourne')


def now():
    return datetime.now(timezone.utc)


def reference(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def publish_owner_health(root, cfg, daily, status=None):
    """Publish the actual package's state after a completed tick or drain."""
    path = daily.output/'persistent-health.json'
    value = json.loads(path.read_bytes()) if path.exists() else {}
    if status is not None:
        value.update(status=status, children=[])
    atomic_json(Path(root)/'health.json', {**value, 'at': now().isoformat(),
        'output': str(daily.output), 'preparation': daily.prepared['receipt_ref'],
        'source_commit': cfg['source_commit'], 'recovery_selection': cfg.get('recovery_selection')})


def inventory_schedule(value, current):
    """Keep every discovered opportunity, including unresolved or past races."""
    from utils.race_schedule_time import scheduled_jump_datetime
    from scripts.refresh_prejump_upcoming import _parse_race_jump_datetime
    rows = []
    for race in value['races']:
        jump = scheduled_jump_datetime(race) if race.get('scheduled_jump_datetime') else _parse_race_jump_datetime(race, now=current)
        if jump is None:
            disposition = 'EXCLUDED_UNRESOLVED_JUMP'
        elif jump <= current:
            disposition = 'ALREADY_PAST'
        elif jump-current <= timedelta(hours=1):
            disposition = 'ACTIVE_LEAD_IN'
        else:
            disposition = 'WAITING_FOR_LEAD_IN'
        rows.append({'url': race['url'], 'source_date': value['source_date'],
                     'jump': jump.isoformat() if jump else None, 'disposition': disposition})
    return rows


def verify_captures(output, plan, scope):
    """Use the same native receipt verification as the accepted session runner."""
    from race_collection.operational_prediction import classify_unready_capture
    from race_collection.manual_prediction_collector_request import ManualPredictionCollectorProtocol
    from scripts.run_freshness_rehearsal import verify_claim_receipt
    evidence = Path(plan['evidence_root'])
    for claim in AttemptAllowance(scope).claims():
        verification = output/'capture-verifications'/(claim.parent.name+'.json')
        rejection = output/'capture-rejections'/(claim.parent.name+'.json')
        terminal = claim.with_suffix('.terminal.json')
        if verification.exists() or rejection.exists() or not terminal.exists():
            continue
        result = json.loads(terminal.read_bytes())['result']
        unready = classify_unready_capture(claim, result, evidence, Path(plan['source_root']))
        if unready:
            create_once(rejection, unready)
            continue
        item = json.loads(claim.read_bytes())['item']
        handoff = ManualPredictionCollectorProtocol(evidence/'manual_prediction_collector_requests_v1').discover_collector_exact_handoff(
            race_id=item['race_id'], current_time=now(), max_age_seconds=300)
        if handoff is None:
            raise ValueError('native_capture_receipt_unavailable')
        verify_claim_receipt(claim, handoff, evidence, Path(plan['source_root']))
        create_once(verification, dict(status='NATIVE_HANDOFF_VERIFIED', race_id=item['race_id'],
            capture_window_minutes=item['capture_window_minutes'],
            capture_attempt_sha256=handoff['capture_attempt_sha256'], observed_at=now().isoformat(),
            prediction_started=False))


def record_native_terminal(args, result):
    """Retain only the native command verdict for this authenticated invocation."""
    invocation = os.environ.get("GREYHOUND_SERVICE_INVOCATION")
    allocation_sha = os.environ.get("GREYHOUND_PERSISTENT_ALLOCATION_SHA256")
    if not invocation or not allocation_sha:
        return
    if len(invocation) != 32 or any(c not in "0123456789abcdef" for c in invocation):
        raise ValueError("persistent_native_invocation_invalid")
    scope = FreshnessContract.load(args.live_freshness_contract)
    binding = scope.value.get("persistent_allocation")
    if not binding or binding['sha256'] != allocation_sha:
        raise ValueError("persistent_native_terminal_authority_mismatch")
    create_once(Path(args.evidence_root)/'shadow_autopilot_daemon_runtime'/'service-terminals'/(invocation+'.json'), {
        'invocation_id': invocation, 'allocation_sha256': allocation_sha,
        'status': result.get('status'), 'runtime_action': result.get('runtime_action'),
        'final_verdict': result.get('final_verdict'), 'run_id': result.get('run_id'),
        'output_dir': result.get('output_dir'), 'at': now().isoformat(),
    })


def classify_native_dispatch(evidence, record, allocation_sha, *, output=None):
    """A successful process exit alone never establishes native lane progress."""
    runtime = Path(evidence)/'shadow_autopilot_daemon_runtime'
    invocation = record['invocation_id']
    lifecycle_path = runtime/'service-lifecycles'/(invocation+'.json')
    terminal_path = runtime/'service-terminals'/(invocation+'.json')
    if not lifecycle_path.is_file() or not terminal_path.is_file():
        raise ValueError('persistent_native_terminal_missing')
    lifecycle = json.loads(lifecycle_path.read_bytes())
    terminal = json.loads(terminal_path.read_bytes())
    if (lifecycle.get('invocation_id') != invocation or lifecycle.get('children_reaped') is not True
            or lifecycle.get('status') != 'COMPLETE' or lifecycle.get('returncode') != record['returncode']
            or terminal.get('invocation_id') != invocation or terminal.get('allocation_sha256') != allocation_sha):
        raise ValueError('persistent_native_terminal_identity_or_lifetime_invalid')
    action, status = terminal.get('runtime_action'), terminal.get('status')
    if (action == 'LIVE_COLLECTION_COMPLETE' and status in {'READY', 'DAEMON_READY'}
            and terminal.get('final_verdict') == 'DAEMON_READY' and record['returncode'] == 0):
        return {'disposition': 'COMPLETED', 'runtime_action': action,
                'terminal': reference(terminal_path), 'lifecycle': reference(lifecycle_path)}
    deferred = {'DEFERRED_LOCK_HELD', 'DEFERRED_FULL_LOCK_HANDOFF',
                'LIVE_CYCLE_OWNER_ACTIVE', 'LIVE_CYCLE_FULL_WAITER_ACTIVE'}
    if (action in deferred and status in {'SKIPPED_LOCK_HELD', 'SKIPPED_FULL_DAEMON_LOCK_HANDOFF'}
            and record['returncode'] in (0, 2)):
        return {'disposition': 'DEFERRED', 'runtime_action': action,
                'terminal': reference(terminal_path), 'lifecycle': reference(lifecycle_path)}
    if (output is not None and action == 'LIVE_PHASE_FAILED' and status == 'FAILED'
            and terminal.get('final_verdict') == 'NEEDS_MORE_AUTOMATION'
            and record['returncode'] == 2 and lifecycle.get('interrupted', False) is False):
        run_id = terminal.get('run_id')
        import re
        if isinstance(run_id, str) and re.fullmatch(r'[A-Za-z0-9_+.-]+', run_id):
            retained = Path(output)/'refresh-deferrals'/(run_id+'.json')
            if (retained.is_file() and Path(terminal.get('output_dir', '')) ==
                    Path(evidence)/('shadow_autopilot_daemonization_v1_'+run_id)):
                ref = reference(retained)
                _verified_refresh_deferral(ref, evidence, allocation_sha)
                return {'disposition': 'REFRESH_OUTAGE', 'runtime_action': action,
                        'terminal': reference(terminal_path), 'lifecycle': reference(lifecycle_path),
                        'refresh_deferral': ref}
    raise ValueError('persistent_native_terminal_failure:'+str(action))



def _verified_refresh_deferral(ref, evidence, allocation_sha):
    from race_collection.live_freshness_contract import classify_refresh_outage
    value = checked(ref)
    classified = classify_refresh_outage(value.get('source_evidence_root', evidence), value.get('run_id'))
    if (classified is None or any(value.get(key) != item for key, item in classified.items())
            or value.get('allocation_sha256', allocation_sha) != allocation_sha
            or type(value.get('failed_cycle_count')) is not int
            or not 1 <= value['failed_cycle_count'] <= 2
            or stamp(value['observed_at']) > now()
            or Path(ref['path']).name != value['run_id']+'.json'):
        raise ValueError('persistent_refresh_deferral_unverified')
    return value


def refresh_outage_pending(output, plan, references, allocation_sha):
    """A prior fresh index is retained history, never proof an outage recovered."""
    paths = list((Path(output)/'refresh-deferrals').glob('*.json'))
    if len(paths) > 2 or len(references) > 2:
        raise ValueError('persistent_refresh_outage_limit_exceeded')
    if not paths and not references:
        return False
    known = []
    for ref in references:
        if Path(ref['path']).parent != Path(output)/'refresh-deferrals':
            raise ValueError('persistent_refresh_deferral_path_invalid')
        known.append(_verified_refresh_deferral(ref, plan['evidence_root'], allocation_sha))
    if len({item['run_id'] for item in known}) != len(known):
        raise ValueError('persistent_refresh_deferral_duplicate')
    # The wrapper may have durably recorded a failure before its child exits.
    # Keep dispatch blocked until the owner can verify that terminal lifecycle.
    if {str(path) for path in paths} != {ref['path'] for ref in references}:
        return True
    from race_collection.synchronous_manual_capture import bounded_current_race_index, CaptureOneRejected
    evidence = Path(plan['evidence_root'])
    try:
        view = bounded_current_race_index(current_time=now(), timeout_seconds=5,
            index_path=evidence/'shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json',
            evidence_root=evidence, max_age_seconds=270, return_verified_view=True)
        source_at = stamp(view.source_generated_at)
        # VerifiedView intentionally preserves stale historical publications;
        # callers must enforce their own admission age even with max_age set.
        age = (now()-source_at).total_seconds()
        return not (0 <= age < 270 and source_at > max(stamp(item['observed_at']) for item in known))
    except CaptureOneRejected as exc:
        if exc.code in {'CURRENT_INDEX_UNAVAILABLE', 'CURRENT_INDEX_STALE', 'DISCOVERY_TIMEOUT'}:
            return True
        raise


def restore_due_times(state):
    """Retain cadence across restarts, including previously consumed launches."""
    due = state.setdefault('next_due_at', {})
    for lane, seconds in (('full', 900), ('odds', 60)):
        starts = [stamp(row['started_at']) for row in state['dispatches'] if row['lane'] == lane]
        derived = max(starts)+timedelta(seconds=seconds) if starts else None
        if lane in due:
            retained = stamp(due[lane])
            if derived is not None and retained < derived:
                raise ValueError('persistent_lane_due_time_regressed')
        elif derived is not None:
            due[lane] = derived.isoformat()
    return due


def discover(contract_path, inventory_path, receipt_path):
    """Child entrypoint: established TheDogs transport with native request guard."""
    from race_collection.live_execution import configure_profile_execution
    from race_collection.live_freshness_contract import install_request_guard
    from race_collection.daily_race_inventory import discover_daily_inventory
    configure_profile_execution(contract_path)
    scope = FreshnessContract.load(contract_path)
    restore = install_request_guard(scope)
    try:
        from upcoming_race_browser import UpcomingRaceBrowser
        browser = UpcomingRaceBrowser()
        ref = discover_daily_inventory(browser, source_date=scope.value['source_date'], path=inventory_path)
        create_once(receipt_path, {'inventory': ref, 'completed_at': now().isoformat()})
    finally:
        restore()


def load_config(path, expected_sha256):
    cfg = checked({'path': str(Path(path).resolve(strict=True)), 'sha256': expected_sha256})
    standing = load_standing_authority(cfg['standing_authority'])
    if cfg.get('status') != 'AUTHORIZED_PERSISTENT_COLLECTOR':
        raise ValueError('persistent_collector_not_authorized')
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root, text=True)
    if commit != cfg['source_commit'] or dirty:
        raise ValueError('persistent_installed_source_changed')
    for key in ('python', 'history_database', 'lock_path', 'installed_dir', 'campaign_root', 'source_state'):
        if not Path(cfg[key]).is_absolute():
            raise ValueError('persistent_config_path_not_absolute')
    if cfg['campaign_id'] != standing['campaign_id']:
        raise ValueError('persistent_campaign_config_mismatch')
    from race_collection.persistent_storage import check_mount
    check_mount(cfg['storage_mount'], Path(standing['state_root']))
    check_mount(cfg['storage_mount'], Path(standing['prediction_root']))
    return cfg, standing


class DailyOwner:
    """A day's processes and immutable consumption survive ordinary restarts."""
    def __init__(self, cfg, prepared):
        from race_collection.operational_prediction import Supervisor
        self.cfg, self.prepared = cfg, prepared
        self.output, self.plan = Path(prepared['output']), prepared['plan']
        self.scope = FreshnessContract(prepared['contract'])
        self.campaign = self.scope.campaign
        self.children = {}
        self.inventory = None
        self.predictions = Supervisor(self.output, self.plan, self.scope)
        self.state_path = self.output/'persistent-owner-state.json'
        self.state = json.loads(self.state_path.read_bytes()) if self.state_path.exists() else {
            'schema_version': 'persistent_owner_state_v1', 'dispatches': [], 'inventory': None,
            'completed_lanes': {'full': 0, 'odds': 0}, 'refresh_failures': []}
        restore_due_times(self.state)
        self.state.setdefault('deferred_lanes', {'full': 0, 'odds': 0})
        # A stale OS PID alone never proves that children completed.
        for dispatch in self.state['dispatches']:
            if dispatch.get('returncode') is None:
                raise ValueError('persistent_interrupted_dispatch_requires_reconciliation')
        if (self.scope.session/'STOP.json').exists() or (self.output/'HALT.json').exists():
            raise ValueError('persistent_day_stopped')
        from race_collection.operational_prediction import require_completed_lifetimes
        require_completed_lifetimes(self.output, self.campaign)

    def save(self):
        self.state['updated_at'] = now().isoformat()
        atomic_json(self.state_path, self.state)

    def activate(self):
        from utils.sportsbet_access import SportsbetAccess
        allocation = self.campaign.persistent
        self.campaign.begin(self.plan['rehearsal_id'], now=now(), deadline=stamp(allocation['cleanup_by']))
        source = SportsbetAccess(self.cfg['source_state'])
        value = source.read()
        existing = value.get('diagnostic_authority') or {}
        if existing.get('persistent_allocation') != self.prepared['allocation_ref']:
            source.authorize_diagnostic(reference=allocation['authority_reference']+':day:'+allocation['allocation_id'],
                expected_sha256=hashlib.sha256(Path(self.cfg['source_state']).read_bytes()).hexdigest(),
                expires_at=stamp(allocation['ends_at']).timestamp(), max_operations=allocation['max_source_operations'],
                rationale='Standing daily engineering authority; finite measured workload; source controls unchanged',
                persistent_allocation=self.prepared['allocation_ref'])
        source.check_admission(allow_active=False)
        self.state['active_at'] = now().isoformat()
        self.save()

    def launch(self, lane, command, cwd, environment, receipt=None):
        if lane in self.children:
            return
        identity = uuid.uuid4().hex
        record = {'lane': lane, 'invocation_id': identity, 'started_at': now().isoformat(),
                  'returncode': None, 'receipt': str(receipt) if receipt else None,
                  'deadline_at': (now()+timedelta(seconds=180 if lane=='inventory' else 1200)).isoformat()}
        self.state['dispatches'].append(record)
        if lane in ('full', 'odds'):
            seconds = 900 if lane == 'full' else 60
            self.state['next_due_at'][lane] = (stamp(record['started_at'])+timedelta(seconds=seconds)).isoformat()
        self.save()  # consume dispatch before process creation
        logpath = self.output/'owner-logs'/(identity+'.log')
        logpath.parent.mkdir(exist_ok=True)
        log = logpath.open('xb')
        environment = {**os.environ, **environment, 'INVOCATION_ID': identity, 'PYTHONDONTWRITEBYTECODE': '1',
                       'GREYHOUND_SPORTSBET_ACCESS_STATE': str(self.cfg['source_state'])}
        child = subprocess.Popen(command, cwd=cwd, env=environment, stdout=log, stderr=log)
        self.children[lane] = (child, log, record)

    def poll(self, predictions=True):
        failure = None
        for lane, (child, log, record) in list(self.children.items()):
            result = child.poll()
            if result is None:
                if now() >= stamp(record['deadline_at']) and not record.get('timeout_signal_sent_at'):
                    self.scope.stop('PERSISTENT_CHILD_DEADLINE')
                    record['timeout_signal_sent_at'] = now().isoformat()
                    self.save()
                    # The native wrapper forwards SIGINT and reaps its owned
                    # descendants. The installed owner cgroup is the final
                    # bounded containment if that proof cannot be obtained.
                    child.send_signal(signal.SIGINT)
                    failure = 'persistent_child_deadline'
                continue
            log.close()
            record.update(returncode=result, completed_at=now().isoformat())
            del self.children[lane]
            self.save()
            if lane == 'inventory':
                if result != 0:
                    failure = 'persistent_native_inventory_failed'
                    continue
                receipt = json.loads(Path(record['receipt']).read_bytes())
                self.state['inventory'] = receipt['inventory']
                self.save()
            else:
                try:
                    verdict = classify_native_dispatch(self.plan['evidence_root'], record,
                        self.prepared['allocation_ref']['sha256'], output=self.output)
                except Exception as exc:
                    record['native_disposition'] = 'FAILED_OR_UNVERIFIED'
                    failure = str(exc)
                    self.save()
                    continue
                record['native_verification'] = verdict
                record['native_disposition'] = verdict['disposition']
                key = 'completed_lanes' if verdict['disposition'] == 'COMPLETED' else 'deferred_lanes'
                self.state[key][lane] += 1
                if verdict['disposition'] == 'REFRESH_OUTAGE':
                    ref = verdict['refresh_deferral']
                    if ref not in self.state['refresh_failures']:
                        self.state['refresh_failures'].append(ref)
                self.save()
        if failure:
            raise ValueError(failure)
        if predictions:
            if (self.scope.session/'STOP.json').exists():
                raise ValueError('persistent_native_scope_stopped')
            verify_captures(self.output, self.plan, self.scope)
            self.predictions.tick(allow_dispatch=not self.outage_pending())

    def outage_pending(self):
        # A native lane owns publication until its complete lifecycle is reaped.
        # Reading the mutable index during its atomic replacement correctly
        # triggers the strict path-swap guard. Wait at the owner boundary;
        # never catch or weaken that integrity rejection.
        publishing = any(lane in self.children for lane in ('full', 'odds'))
        pending = publishing or refresh_outage_pending(self.output, self.plan,
            self.state['refresh_failures'], self.prepared['allocation_ref']['sha256'])
        self.state['forecast_admission_reason'] = (
            'CURRENT_INDEX_REFRESH_IN_PROGRESS' if publishing
            else 'UPSTREAM_TEMPORARY_UNAVAILABLE' if pending else None)
        self.state['forecast_admission_ready'] = not pending
        self.save()
        return pending

    def tick(self):
        from race_collection.daily_race_inventory import load_daily_inventory
        from race_collection.persistent_capacity import audit_inventory_capacity
        from scripts.check_freshness_service import service_command
        self.poll()
        current = now()
        if (self.scope.session/'STOP.json').exists():
            raise ValueError('persistent_native_scope_stopped')
        if current >= self.scope.end:
            return 'DAY_ENDED'
        ref = self.state.get('inventory')
        inventory = None
        if ref:
            raw = json.loads(Path(ref['path']).read_bytes())
            age = (current-stamp(raw['observed_at'])).total_seconds()
            if age <= 1800:
                inventory = load_daily_inventory(**ref, source_date=self.scope.value['source_date'], now=current, max_age_seconds=1800)
        opportunities = inventory_schedule(inventory, current) if inventory else []
        active = any(row['disposition']=='ACTIVE_LEAD_IN' for row in opportunities)
        cadence = 900 if active else 1800
        if inventory and (current-stamp(inventory['observed_at'])).total_seconds() >= cadence-60:
            inventory = None
        if inventory is None and 'inventory' not in self.children:
            if (self.scope.end-current).total_seconds() > 180:
                ident = uuid.uuid4().hex
                target = self.output/'inventories'/(ident+'.json')
                receipt = self.output/'inventories'/(ident+'.receipt.json')
                self.launch('inventory', [self.plan['python'], '-B', '-m', 'scripts.run_persistent_collector',
                    '--discover', str(self.output/'contract.json'), str(target), str(receipt)],
                    self.plan['source_root'], {}, receipt)
        if inventory:
            capacity = json.loads((Path(self.campaign.persistent['state_root'])/'capacity.json').read_bytes())
            audit = audit_inventory_capacity(capacity, [row['jump'] for row in opportunities if row['jump']])
            atomic_json(self.output/'opportunities.json', {'observed_at': current.isoformat(), 'inventory': ref,
                'capacity': audit, 'opportunities': opportunities})
            if audit['status']=='HOLD':
                raise ValueError('persistent_inventory_exceeds_issued_scope')
            # Recheck active-date rollover only after a complete fresh census.
            if current.astimezone(ZONE).date().isoformat() > self.scope.value['source_date']:
                future = [stamp(row['jump']) for row in opportunities if row['jump']]
                if future and max(future)+timedelta(minutes=10) < current and not any(row['jump'] is None for row in opportunities):
                    return 'DAY_ENDED'
            if active and (current-stamp(inventory['observed_at'])).total_seconds() <= 840:
                for lane, unit in (('full', 'shadow-autopilot.service'),
                                   ('odds', 'shadow-autopilot-odds-capture.service')):
                    due = self.state['next_due_at'].get(lane)
                    # Prediction workers read the mutable index after retaining
                    # history. Keep the ownership exclusion bidirectional.
                    if (self.predictions.child is not None or (due and current < stamp(due))
                            or lane in self.children or (self.scope.end-current).total_seconds() <= 600):
                        continue
                    command, cwd, env = service_command(self.output/'units'/unit)
                    command += ['--discovery-inventory', ref['path'], '--discovery-inventory-sha256', ref['sha256'],
                                '--discovery-inventory-source-date', self.scope.value['source_date']]
                    self.launch(lane, command, cwd, env)
        pending = self.outage_pending()
        atomic_json(self.output/'persistent-health.json', {'at': current.isoformat(),
            'reason': self.state['forecast_admission_reason'],
            'forecast_admission_ready': not pending, 'refresh_failures': self.state['refresh_failures'],
            'status': 'HOLD' if pending else 'ACTIVE_COLLECTION' if active and inventory else 'DISCOVERING' if not inventory else 'WAITING_FOR_RACE',
            'source_date': self.scope.value['source_date'], 'inventory': ref,
            'children': list(self.children), 'completed_lanes': self.state['completed_lanes'],
            'deferred_lanes': self.state['deferred_lanes'], 'next_due_at': self.state['next_due_at'],
            'opportunities': len(opportunities), 'source_commit': self.plan['commit'],
            'scientific_admission': 'UNCHANGED', 'result_access': False})
        return 'RUNNING'

    def drain(self, close=False):
        deadline = min(stamp(self.campaign.persistent['cleanup_by']), now()+timedelta(seconds=1860))
        failure = None
        while self.children:
            try:
                self.poll(predictions=False)
            except Exception as exc:
                failure = exc
            if now() >= deadline:
                raise RuntimeError('persistent_drain_deadline_unknown_children')
            time.sleep(1)
        self.predictions.drain()
        from race_collection.operational_prediction import require_completed_lifetimes
        require_completed_lifetimes(self.output, self.campaign)
        if Path(self.plan['lock_path']).exists():
            raise ValueError('persistent_collector_lock_not_released')
        from utils.sportsbet_access import SportsbetAccess
        if SportsbetAccess(self.cfg['source_state']).read()['active'] is not None:
            raise ValueError('persistent_source_operation_not_released')
        if failure:
            raise failure
        if close:
            with self.campaign.ledger() as ledger:
                launched = self.plan['rehearsal_id'] in ledger['launches']
            if launched:
                self.campaign.close(self.plan['rehearsal_id'], now=now())
            elif now() < self.scope.end:
                raise ValueError('persistent_unstarted_day_not_expired')
            create_once(self.output/'day-closed.json', {'at': now().isoformat(),
                'status': 'DRAINED' if launched else 'EXPIRED_UNSTARTED',
                'consumption_preserved': True, 'source_date': self.scope.value['source_date']})
        else:
            self.state['restartable_pause_at'] = now().isoformat()
            self.save()


def _safe_capture_diagnostics(error):
    from race_collection.synchronous_manual_capture import CaptureOneRejected
    if not isinstance(error, CaptureOneRejected):
        return {}
    return {'capture_rejection': {'code': error.code, **{
        key: value for key, value in error.details.items()
        if key in ('path', 'reason') and isinstance(value, str) and len(value) <= 4096}}}


def run(config_path, config_sha256):
    cfg, standing = load_config(config_path, config_sha256)
    root = Path(standing['state_root'])
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    owner = (Path(cfg['campaign_root'])/'owner.lock').open('a')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopping = False
    def stop(_sig, _frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    daily = None
    pointer = root/'current-day.json'
    try:
        while not stopping:
            if daily is None:
                from race_collection.persistent_native import prepare_day
                day = now().astimezone(ZONE).date().isoformat()
                if pointer.exists():
                    prior = json.loads(pointer.read_bytes())
                    if not (Path(prior['output'])/'day-closed.json').exists():
                        day = prior['racing_date']
                    elif prior['racing_date'] == day:
                        # A day ended at its finite bound. Never reopen its
                        # consumed membership while waiting for the next date.
                        time.sleep(2)
                        continue
                prepared = prepare_day(cfg, cfg['standing_authority'], day, now())
                atomic_json(pointer, {'racing_date': day, 'output': prepared['output'],
                    'preparation': prepared['receipt_ref']})
                daily = DailyOwner(cfg, prepared)
                while now() < daily.scope.start and not stopping:
                    time.sleep(1)
                if stopping: break
                if now() >= daily.scope.end:
                    # A retained expired day may be reconciled and closed, but
                    # its acquisition/source authority must never be reopened.
                    daily.drain(close=True)
                    publish_owner_health(root, cfg, daily, 'DAY_ENDED')
                    daily = None
                    continue
                daily.activate()
            if daily.tick() == 'DAY_ENDED':
                daily.drain(close=True)
                publish_owner_health(root, cfg, daily, 'DAY_ENDED')
                daily = None
                continue
            publish_owner_health(root, cfg, daily)
            time.sleep(2)
        if daily:
            daily.drain(close=False)
            publish_owner_health(root, cfg, daily, 'PAUSED')
    except Exception as exc:
        if daily:
            create_once(daily.output/('failure-'+uuid.uuid4().hex+'.json'), {
                'at': now().isoformat(), 'failure_class': type(exc).__name__, 'reason': str(exc),
                **_safe_capture_diagnostics(exc)})
            daily.scope.stop('PERSISTENT_OWNER_FAILURE')
            try: daily.drain(close=False)
            except Exception as cleanup_error:
                create_once(daily.output/('cleanup-unverified-'+uuid.uuid4().hex+'.json'), {
                    'at': now().isoformat(), 'reason': str(cleanup_error),
                    'campaign_lease_retained': True, 'source_lease_not_manually_cleared': True})
            if not (daily.output/'HALT.json').exists():
                create_once(daily.output/'HALT.json', {'at': now().isoformat(), 'reason': str(exc),
                    **_safe_capture_diagnostics(exc)})
        atomic_json(root/'health.json', {'at': now().isoformat(), 'status': 'HOLD', 'reason': str(exc),
            'output': str(daily.output) if daily else None,
            'preparation': daily.prepared['receipt_ref'] if daily else None,
            'source_commit': cfg['source_commit'], 'recovery_selection': cfg.get('recovery_selection')})
        raise
    finally:
        owner.close()
