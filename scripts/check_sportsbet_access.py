#!/usr/bin/env python3
"""Network-free systemd admission condition for both collector lanes."""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.sportsbet_access import SportsbetAccess, SportsbetAccessBlocked


def main():
    try:
        contract = os.environ.get('GREYHOUND_LIVE_FRESHNESS_CONTRACT')
        if contract:
            from race_collection.live_freshness_contract import FreshnessContract
            FreshnessContract.load(Path(contract))
        # Both services acquire the collector lock before any provider access.
        # An active OPEN operation may finish while this service waits there.
        SportsbetAccess().check_admission(allow_active=True)
    except (SportsbetAccessBlocked, ValueError, OSError) as error:
        print(str(error))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
