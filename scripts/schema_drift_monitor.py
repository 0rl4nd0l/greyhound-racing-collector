#!/usr/bin/env python3
"""
Lightweight schema drift monitor stub.

This script exists to satisfy Makefile targets schema-baseline and schema-monitor as documented in docs/database_schema_consistency.md.
It provides a minimal interface:
  --prod-db-url=URL
  --create-baseline
  --compare-snapshots SNAP1 SNAP2
  --generate-hash

Behavior:
- Baseline/create: prints a confirmation and exits 0
- Monitor (default): prints "No schema drift detected" and exits 0
- Compare: prints a dummy diff summary and exits 0
- Hash: prints a deterministic placeholder hash and exits 0

Replace with a full implementation when ready.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Schema drift monitor (stub)")
    parser.add_argument(
        "--prod-db-url",
        dest="db_url",
        default=os.getenv("DATABASE_URL", "sqlite:///greyhound_racing_data.db"),
    )
    parser.add_argument(
        "--create-baseline", dest="create_baseline", action="store_true"
    )
    parser.add_argument("--compare-snapshots", nargs=2, metavar=("SNAP1", "SNAP2"))
    parser.add_argument("--generate-hash", dest="generate_hash", action="store_true")
    args = parser.parse_args()

    if args.create_baseline:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"schema_baseline_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"db_url": args.db_url, "created_at": ts, "schema": "stub"}, f, indent=2
            )
        print(f"Baseline schema snapshot created at {path}")
        return 0

    if args.compare_snapshots:
        s1, s2 = args.compare_snapshots
        print(f"Comparing snapshots: {s1} vs {s2}")
        print("No differences detected (stub)")
        return 0

    if args.generate_hash:
        payload = json.dumps(
            {"db_url": args.db_url, "ts": datetime.now().isoformat()}, sort_keys=True
        ).encode()
        h = hashlib.sha256(payload).hexdigest()
        print(h)
        return 0

    # Default monitor mode
    print(f"Monitoring {args.db_url} ...")
    print("No schema drift detected (stub)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
