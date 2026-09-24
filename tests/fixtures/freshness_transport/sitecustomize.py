"""Fabricated external transports ONLY; no planner, identity, launch or storage mocks."""

import json
import os
from pathlib import Path
import sys
import time

if os.environ.get("FRESHNESS_FABRICATED_SOURCE"):
    import freshness_transport

    freshness_transport.install()

if os.environ.get("GREYHOUND_SHARED_SNAPSHOT_FIXTURE"):
    from snapshot_transport import install
    install()
