#!/usr/bin/env python3
"""
Feature coverage checker against v4 feature contract.

Usage:
  python scripts/check_features_against_contract.py /path/to/race.csv [--preprocess] [--build-features] [--race-id "Race 1 - ABC - 1970-01-01"]

Modes:
- Raw: compares CSV columns directly to contract all_feature_columns.
- --preprocess: runs MLSystemV4.preprocess_upcoming_race_csv before comparison.
- --build-features: builds temporal features and compares feature matrix columns (metadata dropped).

Notes:
- The contract's authoritative expected inputs are taken from all_feature_columns.
- If both numerical_columns and categorical_columns are empty in the contract, this script still works.
"""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

# Ensure repository root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_contract(path: Path) -> dict:
    data = json.loads(path.read_text())
    return data


def compare_columns(expected, df, label: str):
    present = [c for c in expected if c in df.columns]
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    print(f"[{label}] rows={len(df):,}, cols={len(df.columns)}")
    print(f"  present={len(present)}/{len(expected)}")
    print(f"  missing={len(missing)} -> {missing}")
    extra_preview = extra[:25]
    suffix = " ..." if len(extra) > 25 else ""
    print(f"  extra={len(extra)} -> {extra_preview}{suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=str, help="Path to race CSV")
    ap.add_argument("--preprocess", action="store_true", help="Run upcoming race preprocessing")
    ap.add_argument("--build-features", action="store_true", help="Build temporal features for comparison")
    ap.add_argument("--race-id", type=str, default="Race 1 - TEST - 1970-01-01", help="Race ID string for preprocessing/feature building")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    csv_path = (root / args.csv).resolve() if not Path(args.csv).is_absolute() else Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(2)

    contract_path = root / "docs" / "model_contracts" / "v4_feature_contract.json"
    if not contract_path.exists():
        print(f"Contract not found: {contract_path}")
        sys.exit(3)

    contract = load_contract(contract_path)
    expected = contract.get("all_feature_columns") or []
    print(f"Contract all_feature_columns: {len(expected)}")

    df = pd.read_csv(csv_path)
    compare_columns(expected, df, label="raw")

    if args.preprocess or args.build_features:
        from ml_system_v4 import MLSystemV4

        sys_v4 = MLSystemV4()
        proc = sys_v4.preprocess_upcoming_race_csv(df, args.race_id)
        compare_columns(expected, proc, label="preprocessed")

        if args.build_features:
            feats = sys_v4.build_features_for_race_with_cache(proc, args.race_id)
            if feats is None or feats.empty:
                print("Feature building returned empty result.")
                sys.exit(4)
            meta_drop = ["race_id", "dog_clean_name", "target", "target_timestamp"]
            X = feats.drop(columns=[c for c in meta_drop if c in feats.columns], errors="ignore")
            compare_columns(expected, X, label="built_features")
            print("First 12 feature columns:", X.columns[:12].tolist())


if __name__ == "__main__":
    main()

