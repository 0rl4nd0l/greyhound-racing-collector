"""Finite result batches in verified gaps of the existing collector.

The speed calculation never calls this module. Only the separately installed
result controller can drain and resume the pinned collector, outside every
known one-hour lead-in. A durable intent and recovery entrypoint survive a
controller restart. No result value is printed or used to choose a batch.
"""
from datetime import timedelta
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from race_collection import prospective_speed_runtime as runtime
from race_collection import prospective_speed_results as results
from race_collection.prospective_speed_deadline import bounded
from race_collection.retained_card_timing_coverage import Reader, instant

UNIT = 'greyhound-persistent-collector.service'


def require(value, reason):
    if not value:
        raise ValueError(reason)


def safe_gap(health, inventory, now):
    """Full retained census, not an optimistic absence of current forecasts."""
    require(health.get('status') == 'WAITING_FOR_RACE' and health.get('children') == [],
            'WAIT_FOR_IDLE_COLLECTOR')
    require(0 <= (now-instant(health['at'])).total_seconds() <= 90, 'COLLECTOR_HEALTH_STALE')
    require(0 <= (now-instant(inventory['observed_at'])).total_seconds() <= 1800,
            'INVENTORY_STALE')
    from race_collection.daily_race_inventory import _validate
    from zoneinfo import ZoneInfo
    _validate(inventory, now.astimezone(ZoneInfo('Australia/Melbourne')).date().isoformat())
    jumps = []
    for row in inventory['races']:
        require(row.get('scheduled_jump_datetime'), 'INVENTORY_JUMP_UNKNOWN')
        jump = instant(row['scheduled_jump_datetime'])
        if jump > now:
            jumps.append(jump)
    require(not jumps or min(jumps)-now > timedelta(minutes=70), 'COLLECTION_LEAD_IN_ACTIVE')
    require((now+timedelta(minutes=10)).astimezone(ZoneInfo('Australia/Melbourne')).date()
        == now.astimezone(ZoneInfo('Australia/Melbourne')).date(), 'RACING_DAY_ROLLOVER_GAP_UNKNOWN')
    return min(jumps) if jumps else None


class Host:
    def __init__(self, config):
        self.config = config
        self.collector = Reader().json(config['collector_config'])
        self.standing = Reader().json(self.collector['standing_authority'])
        self.producer = Path(self.standing['state_root'])

    def service(self):
        Reader().read(self.config['collector_unit'])
        command = ['systemctl', '--user', 'show', UNIT]
        for name in ('ActiveState', 'SubState', 'MainPID', 'ControlGroup', 'ExecStart', 'WorkingDirectory', 'FragmentPath'):
            command += ['-p', name]
        raw = subprocess.check_output(command, text=True, timeout=20)
        state = dict(line.split('=', 1) for line in raw.splitlines())
        ref = self.config['collector_config']
        require(ref['path'] in state['ExecStart'] and ref['sha256'] in state['ExecStart']
            and Path(state['FragmentPath']).resolve() == Path(self.config['collector_unit']['path']),
            'COLLECTOR_SERVICE_CHANGED')
        source = Path(state['WorkingDirectory'])
        require(source == source.resolve() and str(source) == self.config['collector_source_root'],
                'COLLECTOR_SOURCE_PATH_CHANGED')
        require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
            == self.collector['source_commit'], 'COLLECTOR_COMMIT_CHANGED')
        require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
            cwd=source, text=True).strip(), 'COLLECTOR_SOURCE_DIRTY')
        return state

    def view(self):
        health = Reader().json(runtime.reference(self.producer/'health.json'))
        pointer = Reader().json(runtime.reference(self.producer/'current-day.json'))
        preparation = Reader().json(pointer['preparation'])
        require(health.get('preparation') == pointer['preparation']
            and health.get('output') == preparation['output'] == pointer['output']
            and health.get('source_commit') == self.collector['source_commit'], 'COLLECTOR_HEALTH_BINDING_CHANGED')
        output = Path(preparation['output'])
        require(output.is_relative_to(self.producer) and not (output/'HALT.json').exists(),
                'COLLECTOR_HAS_HALT')
        return health, preparation, output

    def source_open(self):
        from utils.sportsbet_access import SportsbetAccess
        source = SportsbetAccess(self.collector['source_state']).read()
        ledger = Reader().json(runtime.reference(Path(self.collector['campaign_root'])/'ledger.json'))
        require(source['active'] is None and source['phase'] == 'OPEN'
            and source['access_basis']['status'] == 'permitted'
            and source['not_before'] <= runtime.utc_now().timestamp()
            and not ledger.get('source_holds') and not Path(self.collector['lock_path']).exists(),
            'SOURCE_OR_COLLECTOR_NOT_QUIET')

    def ready(self):
        state = self.service()
        require(state['ActiveState'] == 'active' and state['SubState'] == 'running'
            and state['MainPID'] != '0', 'COLLECTOR_NOT_RUNNING')
        health, preparation, output = self.view()
        inventory = Reader().json(health['inventory'])
        require(inventory['source_date'] == preparation['racing_date'], 'INVENTORY_PRODUCER_DATE_CHANGED')
        next_jump = safe_gap(health, inventory, runtime.utc_now())
        owner = Reader().json(runtime.reference(output/'persistent-owner-state.json'))
        self.lifetimes(preparation, output, owner)
        self.source_open()
        safe_gap(health, inventory, runtime.utc_now())
        return {'service': state, 'preparation': health['preparation'], 'health_at': health['at'],
                'inventory': health['inventory'], 'next_jump_at': next_jump.isoformat() if next_jump else None}

    def stop(self):
        subprocess.run(['systemctl', '--user', 'stop', '--no-block', UNIT],
                       check=True, capture_output=True, timeout=20)
        end = time.monotonic()+90
        while time.monotonic() < end:
            state = self.service()
            if state['ActiveState'] == 'inactive' and state['MainPID'] == '0' and not state['ControlGroup']:
                return
            time.sleep(1)
        raise ValueError('COLLECTOR_DRAIN_NOT_YET_VERIFIED')

    def quiet(self, intent):
        state = self.service()
        require(state['ActiveState'] == 'inactive' and state['SubState'] == 'dead'
            and state['MainPID'] == '0' and not state['ControlGroup'], 'COLLECTOR_NOT_DRAINED')
        health, preparation, output = self.view()
        owner = Reader().json(runtime.reference(output/'persistent-owner-state.json'))
        require(health['status'] == 'PAUSED' and health['children'] == []
            and health['preparation'] == intent['before']['preparation']
            and instant(owner['restartable_pause_at']) >= instant(intent['at'])
            and all(row.get('returncode') is not None for row in owner['dispatches']),
            'NATIVE_PAUSE_UNVERIFIED')
        self.lifetimes(preparation, output, owner)
        self.source_open()
        with open(Path(self.collector['campaign_root'])/'owner.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return {'health': runtime.reference(self.producer/'health.json'),
                'owner': runtime.reference(output/'persistent-owner-state.json')}

    def lifetimes(self, preparation, output, owner):
        """Check retained child ownership before stopping, and again after drain."""
        plan = Reader().json(preparation['plan'])
        native = Path(plan['evidence_root'])/'shadow_autopilot_daemon_runtime'
        for row in owner['dispatches']:
            require(row.get('returncode') is not None and row.get('completed_at'), 'CHILD_LIFETIME_PENDING')
            if row['lane'] == 'inventory':
                # Failed but reaped discovery attempts remain immutable history.
                # The current complete census is verified separately in ready().
                continue
            else:
                lifecycle = Reader().json(runtime.reference(native/'service-lifecycles'/(row['invocation_id']+'.json')))
                require(lifecycle.get('status') == 'COMPLETE' and lifecycle.get('children_reaped') is True
                    and lifecycle.get('returncode') == row['returncode'], 'NATIVE_LIFETIME_UNVERIFIED')
        # The original source owns lifetime semantics; use it in an isolated,
        # source-only interpreter rather than importing today's research copy.
        code = ('import json,sys;from pathlib import Path;'
            'from race_collection.freshness_campaign import Campaign;'
            'from race_collection.operational_prediction import require_completed_lifetimes;'
            'p=json.loads(Path(sys.argv[1]).read_bytes());'
            'require_completed_lifetimes(Path(p["output"]),Campaign(sys.argv[2],persistent_allocation=p["allocation"]))')
        subprocess.run(['bwrap', '--die-with-parent', '--unshare-net', '--ro-bind', '/', '/',
            '--tmpfs', '/tmp', '--proc', '/proc', '--dev', '/dev', '--chdir', self.config['collector_source_root'],
            '--setenv', 'PYTHONPATH', self.config['collector_source_root'], '--setenv', 'PYTHONDONTWRITEBYTECODE', '1',
            self.collector['python'], '-B', '-c', code, self._preparation_path(preparation), self.collector['campaign_root']],
            check=True, capture_output=True, timeout=30)

    def _preparation_path(self, preparation):
        pointer = Reader().json(runtime.reference(self.producer/'current-day.json'))
        require(Reader().json(pointer['preparation']) == preparation, 'PRODUCER_CHANGED_DURING_CHECK')
        return pointer['preparation']['path']

    def resume(self, intent):
        state = self.service()
        limit = time.monotonic()+90
        while state['ActiveState'] == 'deactivating' and time.monotonic() < limit:
            time.sleep(1)
            state = self.service()
        if state['ActiveState'] == 'active' and state['SubState'] == 'running':
            health, _, _ = self.view()
            require(health['status'] in {'WAITING_FOR_RACE', 'ACTIVE_COLLECTION', 'DISCOVERING', 'DAY_ENDED'}
                and 0 <= (runtime.utc_now()-instant(health['at'])).total_seconds() <= 90,
                    'RESUMED_HEALTH_STALE')
            return 'ALREADY_RUNNING'
        self.quiet(intent)
        started_at = runtime.utc_now()
        subprocess.run(['systemctl', '--user', 'start', UNIT], check=True, capture_output=True, timeout=30)
        limit = time.monotonic()+60
        while time.monotonic() < limit:
            state = self.service()
            require(state['ActiveState'] in {'active', 'activating'}, 'RESUME_NOT_RUNNING')
            health, _, _ = self.view()
            if (state['ActiveState'] == 'active' and state['MainPID'] != '0'
                    and health['status'] in {'WAITING_FOR_RACE', 'ACTIVE_COLLECTION', 'DISCOVERING', 'DAY_ENDED'}
                    and health['preparation'] == intent['before']['preparation']
                    and started_at <= instant(health['at']) <= runtime.utc_now()):
                return 'RESUMED'
            time.sleep(1)
        raise ValueError('RESUME_FRESH_HEALTH_UNVERIFIED')


def cycle(reference, *, recover_only=False, host_factory=Host):
    config = Reader().json(reference)
    require(config.get('schema_version') == 'prospective_speed_result_controller_v1'
        and config.get('status') == 'AUTHORIZED_EXISTING_DEVELOPMENT_QUIET_GAPS', 'CONTROLLER_NOT_AUTHORIZED')
    bridge, plan, _ = results.load_config(config['bridge'])
    root = Path(config['state_root'])
    host = host_factory(config)
    with runtime.exclusive(root):
        # Every intent is durable before a stop. A prior batch is never repeated
        # until its existing collector ownership is restored or explicitly held.
        for directory in sorted(root.glob('batch-*')):
            if (directory/'complete.json').exists():
                continue
            intent_path = directory/'intent.json'
            require(intent_path.exists(), 'CONTROLLER_PARTIAL_INTENT_REQUIRES_REVIEW')
            intent = Reader().json(runtime.reference(intent_path))
            require(intent['config'] == reference, 'CONTROLLER_RESTART_CONFIG_CHANGED')
            status = host.resume(intent)
            runtime.put_new(directory/'complete.json', {'status': 'RECOVERED_'+status, 'at': runtime.utc_now().isoformat()})
        if recover_only:
            return {'status': 'RECOVERY_COMPLETE'}
        if runtime.utc_now() < instant('2026-10-10T12:50:00+11:00'):
            return {'status': 'BEFORE_DEVELOPMENT_HORIZON'}
        # Nomination is source-only. Outcomes remain unopened until exclusive
        # acquisition ownership and an exact selected native seal are proved.
        results.prepare_queue(config['bridge'])
        due = results.inspect_queue(config['bridge'])
        if not due['selected_due']:
            return {'status': 'NO_SELECTED_RESULT_DUE'}
        if not due['transport_permitted_by_time_and_budget']:
            for _ in range(12):
                if not results.inspect_queue(config['bridge'])['selected_due']:
                    break
                results.run_cycle(config['bridge'])
            return {'status': 'TERMINAL_RESULT_ACCOUNTING'}
        try:
            before = host.ready()
        except (ValueError, BlockingIOError) as error:
            return {'status': 'WAIT_FOR_VERIFIED_COLLECTION_GAP', 'reason': str(error)}
        directory = root/('batch-'+uuid.uuid4().hex)
        directory.mkdir(mode=0o700)
        intent = {'at': runtime.utc_now().isoformat(), 'config': reference, 'before': before}
        runtime.put_new(directory/'intent.json', intent)
        try:
            host.stop()
            runtime.put_new(directory/'quiet.json', host.quiet(intent))
            end = time.monotonic()+300
            for _ in range(12):
                if time.monotonic() > end-65 or not results.inspect_queue(config['bridge'])['selected_due']:
                    break
                next_jump = intent['before'].get('next_jump_at')
                if next_jump and instant(next_jump)-runtime.utc_now() <= timedelta(minutes=63):
                    break
                with bounded(60):
                    verdict = results.run_cycle(config['bridge'])
                runtime.put_new(directory/('cycle-'+uuid.uuid4().hex+'.json'), verdict)
                if verdict.get('status') in {'SOURCE_STOP', 'WAITING_FOR_EXISTING_SOURCE_OWNER', 'WORKER_BUSY'}:
                    break
        finally:
            resumed = host.resume(intent)
            runtime.put_new(directory/'complete.json', {'status': resumed, 'at': runtime.utc_now().isoformat()})
        return {'status': 'RESULT_BATCH_COMPLETE_COLLECTOR_RUNNING'}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--recover-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(cycle({'path': args.config, 'sha256': args.config_sha256}, recover_only=args.recover_only), sort_keys=True))


if __name__ == '__main__':
    main()
