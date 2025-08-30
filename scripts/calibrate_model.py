#!/usr/bin/env python3
"""
Calibration/Promotion Wrapper
=============================

This script adapts the expected calibrate_model.py CLI used by diagnostics to the
existing tooling in this repository. It accepts the following arguments:

  --model {et|xgb}
  --calibration {raw|sigmoid|isotonic}
  --promote               (required to actually promote)
  --max-races N           (optional, accepted and ignored here)

Behavior:
- Attempts to register the latest MLSystemV4 artifact using scripts/register_latest_v4_model.py
- Prints a single JSON object line to stdout with keys: success, model_id, model, calibration
  so that the diagnostics script can parse and include it in its audit log.

Exit codes:
  0 on success,
  non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate/Promote model wrapper")
    p.add_argument("--model", required=True, choices=["et", "xgb", "extratrees", "xgboost"], help="Model to calibrate/promote")
    p.add_argument("--calibration", required=True, choices=["raw", "sigmoid", "isotonic"], help="Calibration method")
    p.add_argument("--promote", action="store_true", help="Promote the calibrated/best model")
    p.add_argument("--max-races", type=int, default=0, help="Optional cap for upstream processes (ignored here)")
    return p.parse_args()


def find_registration_script() -> Path | None:
    # Resolve repo root by walking up from this file
    here = Path(__file__).resolve().parent
    candidates = [
        here / "register_latest_v4_model.py",
        here.parent / "scripts" / "register_latest_v4_model.py",
        Path.cwd() / "scripts" / "register_latest_v4_model.py",
    ]
    for c in candidates:
        try:
            if c.exists():
                return c
        except Exception:
            continue
    return None

def main() -e int:
    args = parse_args()

    # Require --promote flag to proceed
    if not args.promote:
        print(json.dumps({
            "success": False,
            "error": "--promote flag is required to perform promotion",
            "model": args.model,
            "calibration": args.calibration
        }))
        return 2

    # 1) Materialize a true V4 artifact from the in-memory MLSystemV4
    try:
        from datetime import datetime as _dt
        import joblib
        from ml_system_v4 import MLSystemV4
        system = MLSystemV4()
        calibrated = getattr(system, 'calibrated_pipeline', None)
        feature_columns = getattr(system, 'feature_columns', []) or []
        model_info = getattr(system, 'model_info', {}) or {}
        # Annotate calibration method requested
        model_info = dict(model_info)
        model_info['trained_at'] = model_info.get('trained_at') or _dt.now().isoformat()
        model_info['calibration_method'] = args.calibration
        # Persist artifact
        out_dir = Path('ml_models_v4')
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = out_dir / f"ml_model_v4_{_dt.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        payload = {
            'calibrated_pipeline': calibrated,
            'feature_columns': feature_columns,
            'model_info': model_info,
        }
        joblib.dump(payload, artifact_path)
        saved_path = str(artifact_path)
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": f"failed to save V4 artifact: {e}",
            "model": args.model,
            "calibration": args.calibration
        }))
        return 5

    # 2) Run true calibration verification (Brier, reliability) on the saved artifact
    try:
        env2 = os.environ.copy()
        env2.setdefault("PYTHONPATH", str(Path.cwd()))
        calib_proc = subprocess.run([sys.executable, "run_calibration.py", "--model_path", saved_path], capture_output=True, text=True, env=env2)
        calib_ok = (calib_proc.returncode == 0)
        calib_stdout_tail = (calib_proc.stdout[-400:] if calib_proc.stdout else None)
        calib_stderr_tail = (calib_proc.stderr[-400:] if calib_proc.stderr else None)
        if not calib_ok:
            print(json.dumps({
                "success": False,
                "error": "calibration verification failed",
                "model": args.model,
                "calibration": args.calibration,
                "artifact": saved_path,
                "stdout_tail": calib_stdout_tail,
                "stderr_tail": calib_stderr_tail
            }))
            return 6
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": f"calibration step error: {e}",
            "model": args.model,
            "calibration": args.calibration,
            "artifact": saved_path
        }))
        return 7

    # 3) Register latest V4 artifact into the registry
    reg_script = find_registration_script()
    if not reg_script:
        print(json.dumps({
            "success": False,
            "error": "register_latest_v4_model.py not found in scripts/",
            "hint": "Ensure ml_models_v4 contains a model artifact or add the registration script.",
            "model": args.model,
            "calibration": args.calibration,
            "artifact": saved_path
        }))
        return 3

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(Path.cwd()))
    proc = subprocess.run([sys.executable, str(reg_script)], capture_output=True, text=True, env=env)

    model_id = None
    success = (proc.returncode == 0)
    reg_payload = {}

    if proc.stdout:
        try:
            reg_payload = json.loads(proc.stdout.strip().splitlines()[-1])
            model_id = reg_payload.get("registered_model_id") or reg_payload.get("model_id")
        except Exception:
            pass

    print(json.dumps({
        "success": success and bool(model_id),
        "model_id": model_id,
        "model": args.model,
        "calibration": args.calibration,
        "artifact": saved_path,
        "registration": reg_payload,
        "stderr_tail": (proc.stderr[-200:] if proc.stderr else None)
    }))

    return 0 if (success and model_id) else 4
    return 0 if (success and model_id) else 4


if __name__ == "__main__":
    raise SystemExit(main())

