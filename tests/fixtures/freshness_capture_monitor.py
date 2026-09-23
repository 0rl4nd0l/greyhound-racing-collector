"""Real packaged monitor/restorer surrounding fabricated-transport service commands."""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

package, action = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(package / 'source'))
from scripts.check_freshness_service import deny_network
deny_network()
from scripts import run_freshness_rehearsal as supervisor
from scripts.prepare_freshness_rehearsal import UNITS
from race_collection.live_freshness_contract import AttemptAllowance, FreshnessContract

plan = json.loads((package / 'plan.json').read_bytes())


class Control:
    active = {name: True for name in supervisor.TIMERS}

    def show(self, name):
        return dict(ActiveState='active' if self.active.get(name, name == 'greyhound-operator-ui-r3.service') else 'inactive',
                    SubState='dead', MainPID='123' if name == 'greyhound-operator-ui-r3.service' else '0',
                    DropInPaths='', WorkingDirectory=plan['source_root'],
                    ExecMainStartTimestampMonotonic='0', ExecMainExitTimestampMonotonic='0')

    def idle(self):
        return True

    def command(self, command, *args):
        if command in {'start', 'stop'}:
            self.active[args[0]] = command == 'start'
        elif command == 'is-enabled':
            return 'enabled\n'
        elif command == 'show':
            return 'LastTriggerUSecMonotonic=0\nNextElapseUSecMonotonic=0\nNextElapseUSecRealtime=0\nActiveState=active\n'
        elif command != 'daemon-reload':
            raise AssertionError(command)
        return ''


control = Control()
if action == 'startup':
    supervisor.snapshot(package, plan, control)
    for name in UNITS:
        (Path(plan['installed_dir']) / name).write_bytes((package / 'units' / name).read_bytes())
    sample = supervisor.sample(plan, package, control)
    assert sample['index_status'] == 'UNAVAILABLE/DATA_MISSING', sample
    assert sample['collector_status'] == 'UNAVAILABLE/DATA_MISSING', sample
else:
    sample = supervisor.sample(plan, package, control)
    assert sample['index_status'] == 'AVAILABLE/FRESH', sample
    allowance = AttemptAllowance(FreshnessContract.load(package / 'contract.json'))
    claims = allowance.claims()
    # Controlled restoration clock waits out the original consumed windows;
    # systemd file restoration and hash/activity checks are the real implementation.
    closes = [AttemptAllowance.check_window(json.loads(p.read_bytes())['item'],
                now=datetime.fromisoformat(json.loads(p.read_bytes())['reserved_at'])) for p in claims]
    stamp = max(closes) + timedelta(seconds=1)
    supervisor.now = lambda: stamp
    supervisor.restore(package, plan, control)
    assert json.loads((package / 'restored.json').read_bytes())['hashes'] == plan['baseline_unit_sha256']
(package / ('monitor-' + action + '.json')).write_text(json.dumps(sample, indent=2))
