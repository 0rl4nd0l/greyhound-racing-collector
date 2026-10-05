"""Observe the collector or run its one dedicated Codex recovery worker."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('command', choices=('check','run-incident','exercise'))
    args = parser.parse_args()
    from race_collection.codex_recovery_watch import check, load_config, run_incident
    try:
        cfg = load_config(args.config)
        result = check(cfg) if args.command == 'check' else run_incident(cfg, exercise=args.command == 'exercise')
    except Exception as exc:
        print(json.dumps({'status':'WATCH_ERROR','failure_class':type(exc).__name__}))
        return 1
    print(json.dumps(result,sort_keys=True))
    return 1 if result['status'] in ('SPAWN_FAILED','RUNNER_FAILED','AGENT_FAILED','RECOVERY_UNVERIFIED') else 0


if __name__ == '__main__':
    raise SystemExit(main())
