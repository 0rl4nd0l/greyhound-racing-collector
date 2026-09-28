#!/usr/bin/env python3
"""Stage pinned persistent-service units and a manifest; never install or activate."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

COMPONENTS = {
    'schedule': ('scripts.run_comparison_schedule', '--config', 5, 9000, 2400),
    'results': ('scripts.run_comparison_result_queue', '--binding', 20, 360, 30),
    'health': ('scripts.check_comparison_health', '--schedule-config', 5, 60, 15),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unit_path(path):
    text = str(path)
    if not path.is_absolute() or any(c.isspace() or c in '%"\'\\' for c in text):
        raise ValueError('unit_path_must_be_absolute_without_expansion_or_whitespace')
    return text


def units(*, source, python, schedule_config, result_binding, mount):
    source, python, mount = map(unit_path, (source, python, mount))
    paths = {'schedule': unit_path(schedule_config), 'results': unit_path(result_binding),
             'health': unit_path(schedule_config)}
    output = {}
    for name, (module, flag, minutes, runtime, stop) in COMPONENTS.items():
        stem = 'greyhound-comparison-' + name
        arguments = f'{flag} {paths[name]}'
        if name == 'health':
            arguments += f' --result-binding {unit_path(result_binding)} --output {unit_path(schedule_config.parent / "monitor/health.json")}'
            arguments += f' --notification-config {unit_path(schedule_config.parent / "notification.APPROVED.json")}'
        conditions = '' if name == 'health' else f'ConditionPathIsMountPoint={mount}\nConditionPathExists={paths[name]}\n'
        output[stem + '.service'] = f'''[Unit]
Description=Greyhound persistent comparison {name}
{conditions}
[Service]
Type=exec
WorkingDirectory={source}
Environment=PYTHONPATH={source}
Environment=PYTHONDONTWRITEBYTECODE=1
Environment=OPENBLAS_NUM_THREADS=1
Environment=OMP_NUM_THREADS=1
Environment=MKL_NUM_THREADS=1
ExecStart={python} -B -m {module} {arguments}
UMask=0077
Nice=10
MemoryMax=1G
NoNewPrivileges=yes
RuntimeMaxSec={runtime}
TimeoutStopSec={stop}
KillMode=mixed
Restart=no
StandardOutput=journal
StandardError=journal
SyslogIdentifier={stem}
'''
        output[stem + '.timer'] = f'''[Unit]
Description=Check due Greyhound comparison {name} work

[Timer]
OnBootSec=2min
OnCalendar=*:0/{minutes}
Persistent=true
AccuracySec=15s
Unit={stem}.service

[Install]
WantedBy=timers.target
'''
    return output


def prepare(*, source, commit, python, baseline, schedule_config, result_binding, campaign_root, mount, output):
    source = source.resolve(strict=True)
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if actual != commit or len(commit) != 40:
        raise ValueError('source_commit_changed')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=source, text=True).strip():
        raise ValueError('tracked_source_dirty')
    for module, *_ in COMPONENTS.values():
        if not (source / (module.replace('.', '/') + '.py')).is_file():
            raise ValueError('persistent_component_missing:' + module)
    identity = json.loads(baseline.read_bytes())
    if identity['tested_runtime_commit'] != '8b5552c78222f9a0b5dfa1fa7bb30003bb375922':
        raise ValueError('unexpected_tested_baseline')
    if subprocess.run(['git', 'merge-base', '--is-ancestor', identity['tested_runtime_commit'], commit], cwd=source).returncode:
        raise ValueError('tested_baseline_not_integrated')
    mount = mount.resolve(strict=True)
    fs = subprocess.check_output(['findmnt', '--json', '--output', 'UUID,FSTYPE,TARGET', '--target', str(mount)], text=True)
    filesystem = json.loads(fs)['filesystems'][0]
    if filesystem['target'] != str(mount) or not filesystem['uuid']:
        raise ValueError('dedicated_mount_identity_missing')
    output.mkdir(parents=True, exist_ok=False)
    generated = output / 'units'; generated.mkdir()
    for name, text in units(source=source, python=python, schedule_config=schedule_config,
                            result_binding=result_binding, mount=mount).items():
        (generated / name).write_text(text)
    (output / 'baseline.json').write_bytes(baseline.read_bytes())
    baseline_units = output / 'baseline-units'
    baseline_units.mkdir()
    for name, row in identity['units'].items():
        original = Path(row['path'])
        if digest(original) != row['sha256']:
            raise ValueError('baseline_unit_changed:' + name)
        (baseline_units / name).write_bytes(original.read_bytes())
    with (output / 'source.tar').open('xb') as stream:
        subprocess.run(['git', 'archive', '--format=tar', commit], cwd=source, stdout=stream, check=True)
    manifest = {
        'schema_version': 'persistent_comparison_deployment_v1',
        'status': 'PREPARED_NOT_AUTHORIZED', 'recorded_at': datetime.now(timezone.utc).isoformat(),
        'source_root': str(source), 'source_commit': commit,
        'python': str(python), 'python_sha256': digest(python.resolve(strict=True)),
        'schedule_config': str(schedule_config), 'result_binding': str(result_binding),
        'campaign_root': str(campaign_root.resolve(strict=True)),
        'monitor_output': str(schedule_config.parent / 'monitor/health.json'),
        'filesystem': filesystem,
        'baseline_sha256': digest(output / 'baseline.json'),
        'baseline_unit_sha256': {p.name: digest(p) for p in sorted(baseline_units.iterdir())},
        'source_archive_sha256': digest(output / 'source.tar'),
        'unit_sha256': {p.name: digest(p) for p in sorted(generated.iterdir())},
        'live_actions': False, 'target_results_accessed': False,
    }
    with (output / 'deployment.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True); stream.write('\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'python', 'baseline', 'schedule-config', 'result-binding', 'campaign-root', 'mount', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
