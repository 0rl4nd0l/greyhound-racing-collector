#!/usr/bin/env python3
"""Read-only persistent deployment preflight/status; no provider or database access."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

COLLECTORS = ('shadow-autopilot.service', 'shadow-autopilot-odds-capture.service')
TIMERS = ('shadow-autopilot.timer', 'shadow-autopilot-odds-capture.timer')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def show(name):
    raw = subprocess.check_output(['systemctl', '--user', 'show', name,
        '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID', '-p', 'ControlPID',
        '-p', 'UnitFileState', '-p', 'FragmentPath', '-p', 'DropInPaths', '-p', 'ControlGroup'], text=True, timeout=5)
    return dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)


def inspect(package, expected, *, preflight=False, installed=False):
    manifest_path = package / 'deployment.json'
    if sha(manifest_path) != expected:
        raise ValueError('deployment_manifest_changed')
    manifest = read(manifest_path)
    if manifest['schema_version'] != 'persistent_comparison_deployment_v1':
        raise ValueError('unknown_deployment_schema')
    checks = {'source_archive': sha(package / 'source.tar') == manifest['source_archive_sha256'],
              'baseline': sha(package / 'baseline.json') == manifest['baseline_sha256'],
              'python': sha(Path(manifest['python']).resolve()) == manifest['python_sha256']}
    for name, expected_hash in manifest.get('baseline_unit_sha256', {}).items():
        checks['retained_baseline:' + name] = sha(package / 'baseline-units' / name) == expected_hash
    source = Path(manifest['source_root'])
    checks['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() == manifest['source_commit']
    checks['source_clean'] = not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=source, text=True).strip()
    for name, expected_hash in manifest['unit_sha256'].items():
        checks['staged:' + name] = sha(package / 'units' / name) == expected_hash
    mount = Path(manifest['filesystem']['target'])
    fs = json.loads(subprocess.check_output(['findmnt', '--json', '--output', 'UUID,FSTYPE,TARGET', '--target', str(mount)], text=True))['filesystems'][0]
    checks['dedicated_mount'] = fs == manifest['filesystem'] and mount.is_mount()
    free = shutil.disk_usage(mount).free
    checks['free_space'] = free >= (100 if preflight else 10) * 2**30
    baseline = read(package / 'baseline.json')
    unit_states = {}
    for name, row in baseline['units'].items():
        current = show(name); unit_states[name] = current
        if preflight:
            checks['baseline_unit:' + name] = sha(Path(row['path'])) == row['sha256']
            checks['no_dropins:' + name] = not current['DropInPaths']
            if name in COLLECTORS:
                checks['quiescent:' + name] = current['ActiveState'] in {'inactive', 'failed'} and current['MainPID'] == '0' and current['ControlPID'] == '0'
                group = current.get('ControlGroup')
                checks['empty_cgroup:' + name] = not group or not any(p.read_text().strip() for p in (Path('/sys/fs/cgroup') / group.lstrip('/')).glob('**/cgroup.procs'))
            elif name in TIMERS:
                checks['held:' + name] = current['ActiveState'] == 'inactive' and current['UnitFileState'] == 'disabled'
    checks['r3_binding'] = sha(Path(baseline['r3_binding'])) == baseline['r3_binding_sha256']
    for name, expected_hash in manifest['unit_sha256'].items():
        current = show(name); unit_states[name] = current
        if installed:
            path = Path(current.get('FragmentPath', ''))
            checks['installed:' + name] = path.is_file() and sha(path) == expected_hash and not current['DropInPaths']
        elif preflight:
            checks['new_unit_absent:' + name] = not current.get('FragmentPath')
    gate = read(Path(baseline['source']['path']))
    campaign = Path(manifest['campaign_root'])
    ledger = read(campaign / 'ledger.json')
    if preflight:
        checks['collector_lock_released'] = not Path(baseline['collector_lock']).exists()
        checks['source_quiescent'] = gate['active'] is None
        checks['source_open'] = gate['phase'] == 'OPEN'
        checks['campaign_no_open_lease'] = not any(not row.get('closed_at') for row in ledger['launches'].values())
        checks['campaign_no_source_hold'] = not ledger.get('source_holds')
        with (campaign / 'owner.lock').open('rb') as owner:
            try:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                checks['campaign_owner_released'] = True
            except BlockingIOError:
                checks['campaign_owner_released'] = False
    authority = {}
    for name in ('schedule_config', 'result_binding'):
        path = Path(manifest[name]); authority[name] = {'present': path.is_file()}
        if name == 'schedule_config' and path.is_file():
            authority[name]['status'] = read(path).get('status')
    return {'status': 'CHECKS_PASS' if all(checks.values()) else 'CHECKS_FAILED',
        'mode': 'PREFLIGHT' if preflight else 'STATUS', 'source_commit': manifest['source_commit'],
        'checks': checks, 'findings': [k for k, v in checks.items() if not v],
        'units': unit_states, 'volume_free_bytes': free, 'authority': authority,
        'source': {'phase': gate['phase'], 'active': gate['active'] is not None,
                   'operations': len(gate['operations']), 'denials': len(gate['denials'])},
        'campaign': {'attempts': len(ledger['attempts']), 'requests': ledger['logical_requests'],
                     'open_leases': sum(not row.get('closed_at') for row in ledger['launches'].values()),
                     'source_holds': len(ledger.get('source_holds', []))},
        'outcomes_released': False, 'live_actions': False,
        'note': 'Artifact/runtime checks are not activation authority or evidence of live prediction/result success.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--installed', action='store_true')
    args = parser.parse_args()
    result = inspect(args.package, args.manifest_sha256, preflight=args.preflight, installed=args.installed)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['status'] == 'CHECKS_PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
