#!/usr/bin/env python3
"""
CI promotion gate for ML System V4.

- Runs race-level evaluation with current DB
- Enforces thresholds controlled by env vars (with safe defaults)
- Exits non-zero if thresholds are not met (CI-friendly)

Env (with defaults):
- V4_DISABLE_ACCURACY_OPTIMIZER=1 (should be 1; gate expects core V4 path)
- V4_NORMALIZATION_MODE=simple
- BRIER_MAX=0.13
- LOGLOSS_MAX=0.42
- TOP1_MIN=0.27
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def run_eval(n_races: int = 200, order: str = "recent") -> dict:
    env = os.environ.copy()
    env.setdefault("V4_DISABLE_ACCURACY_OPTIMIZER", "1")
    env.setdefault("V4_NORMALIZATION_MODE", "simple")
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "evaluate_race_level_v4.py"),
        "--n-races",
        str(n_races),
        "--order",
        order,
    ]
    proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)
    try:
        data = json.loads(proc.stdout.strip())
        return data
    except Exception:
        print(proc.stdout)
        print("Could not parse evaluator JSON from stdout", file=sys.stderr)
        raise SystemExit(2)


def main() -> int:
    # Thresholds (defaults tuned to current baseline with small slack)
    brier_max = float(os.getenv("BRIER_MAX", "0.13"))
    logloss_max = float(os.getenv("LOGLOSS_MAX", "0.42"))
    top1_min = float(os.getenv("TOP1_MIN", "0.27"))

    metrics = run_eval()
    print(json.dumps({"gate_metrics": metrics}, indent=2))

    acc = float(metrics.get("top1_accuracy", 0.0))
    brier = float(metrics.get("brier", 1.0))
    logloss = float(metrics.get("log_loss", 10.0))

    failed = []
    if brier > brier_max:
        failed.append(f"Brier {brier:.6f} > {brier_max:.6f}")
    if logloss > logloss_max:
        failed.append(f"LogLoss {logloss:.6f} > {logloss_max:.6f}")
    if acc < top1_min:
        failed.append(f"Top1 {acc:.6f} < {top1_min:.6f}")

    if failed:
        print("PROMOTION GATE: FAIL", file=sys.stderr)
        for f in failed:
            print(" - ", f, file=sys.stderr)
        return 3

    print("PROMOTION GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

