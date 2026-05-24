#!/usr/bin/env python3
"""
Manual drift check for recent vs older window using DriftMonitor.
Saves: audit_results/<ts>/drift/manual_drift_report.json
"""
import json
import os
from pathlib import Path
from datetime import datetime

import pandas as pd

from scripts.ml_backtesting_trainer import MLBacktestingTrainer
from drift_monitor import DriftMonitor


def main():
    # Windows can be customized via env
    ref_n = int(os.getenv("DRIFT_REF_N", "1000"))
    cur_n = int(os.getenv("DRIFT_CUR_N", "200"))

    trainer = MLBacktestingTrainer()
    df = trainer.load_historical_race_data(months_back=12)
    if df is None or len(df) == 0:
        print("No historical data available for drift check.")
        return 2

    # Ensure date column is datetime
    if "race_date" in df.columns:
        df = df.sort_values("race_date")
    else:
        print("race_date column missing; aborting.")
        return 2

    # Build reference and current windows
    if len(df) < (ref_n + cur_n):
        # scale down proportionally
        half = max(50, len(df) // 2)
        ref_n = half
        cur_n = len(df) - half

    ref_df = df.iloc[:ref_n].copy()
    cur_df = df.iloc[-cur_n:].copy()

    monitor = DriftMonitor(reference_data=ref_df)
    results = monitor.check_for_drift(cur_df)

    ts = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(f"audit_results/{ts}/drift")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "manual_drift_report.json"

    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

