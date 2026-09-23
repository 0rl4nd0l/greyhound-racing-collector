"""Stage both existing capture services together; never install or activate them.

The installed units must match the existing generators apart from their source
checkout. Keep lane-specific settings and exact rollback bytes. Both staged
retention units use one absolute configuration path and one collector lock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import shadow_autopilot_daemon as daemon

SERVICES = (daemon.SERVICE_NAME, daemon.ODDS_CAPTURE_SERVICE_NAME)
TIMERS = (daemon.TIMER_NAME, daemon.ODDS_CAPTURE_TIMER_NAME)


def service_command(unit: str) -> list[str]:
    commands = [line.removeprefix("ExecStart=") for line in unit.splitlines()
                if line.startswith("ExecStart=")]
    if len(commands) != 1:
        raise ValueError("one ExecStart required")
    return shlex.split(commands[0].replace("%%", "%"))


def prepare_services(*, installed_dir: Path, output_dir: Path, repo_path: Path,
                     retention_config: Path) -> dict:
    repo_path = repo_path.resolve()
    retention_config = retention_config.absolute()
    if output_dir.exists():
        raise ValueError("output directory must be new")
    originals = {name: (installed_dir / name).read_text() for name in SERVICES + TIMERS}
    rendered = {}
    lane_args = []
    for name, subcommand in zip(SERVICES, ("run-once", "run-odds-capture-once")):
        original = originals[name]
        command = service_command(original)
        args = daemon.parse_args(command[2:])
        if args.command != subcommand or args.input_retention_config is not None:
            raise ValueError("both installed lanes must be default-off")
        old_repo = Path(command[1]).parent.parent
        if Path(command[1]) != old_repo / "scripts/shadow_autopilot_daemon.py":
            raise ValueError("unsupported service entrypoint")
        common = dict(repo_path=repo_path, python_path=Path(command[0]),
                      timeout_seconds=args.timeout_seconds, evidence_root=args.evidence_root,
                      db_path=args.db, lock_path=args.lock_path, state_path=args.state_path,
                      forward_corpus_root=args.forward_corpus_root,
                      forward_baseline_config=args.forward_baseline_config)
        if subcommand == "run-once":
            pause = [line.removeprefix("ConditionPathExists=!") for line in original.splitlines()
                     if line.startswith("ConditionPathExists=!")]
            if len(pause) > 1:
                raise ValueError("unsupported pause conditions")
            common.update(shadow_model=args.shadow_model,
                          odds_capture_state_path=args.odds_capture_state_path,
                          pause_path=Path(pause[0]) if pause else None)
            renderer = daemon.service_file_text
        else:
            common.update(refresh_limit=args.refresh_limit)
            renderer = daemon.odds_capture_service_file_text
        disabled = renderer(**common)
        expected = original.replace(f"WorkingDirectory={old_repo}\n", f"WorkingDirectory={repo_path}\n")
        expected = expected.replace(f"{old_repo}/scripts/shadow_autopilot_daemon.py",
                                    f"{repo_path}/scripts/shadow_autopilot_daemon.py")
        if disabled != expected:
            raise ValueError("installed service differs from supported generator; review required")
        enabled = renderer(**common, input_retention_config=retention_config)
        rendered[name] = {"default-off": disabled, "retention-configured": enabled}
        lane_args.append((command[0], args))
    full_python, full = lane_args[0]
    odds_python, odds = lane_args[1]
    if (full_python != odds_python or full.db != odds.db or full.lock_path is None
            or full.lock_path != odds.lock_path or full.evidence_root != odds.evidence_root
            or full.odds_capture_state_path != odds.state_path):
        raise ValueError("capture lanes must share interpreter, DB, lock, evidence and odds state")

    # All checks precede any output write; no installed file is changed.
    output_dir.mkdir(parents=True)
    for label in ("default-off", "retention-configured", "rollback"):
        directory = output_dir / label
        directory.mkdir()
        for name in SERVICES + TIMERS:
            content = originals[name] if label == "rollback" or name in TIMERS else rendered[name][label]
            (directory / name).write_text(content)
    report = {
        "status": "PAIRED_SERVICES_PREPARED_NOT_INSTALLED",
        "services": list(SERVICES), "timers": list(TIMERS),
        "source_path": str(repo_path), "shared_retention_config": str(retention_config),
        "shared_lock": str(full.lock_path), "db": str(full.db),
        "retention_config_created": False, "timers_changed": False,
        "files": {str(p.relative_to(output_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(output_dir.glob("*/*"))},
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--retention-config", type=Path, required=True)
    args = parser.parse_args(argv)
    report = prepare_services(installed_dir=args.installed_dir, output_dir=args.output_dir,
                              repo_path=args.repo_path, retention_config=args.retention_config)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
