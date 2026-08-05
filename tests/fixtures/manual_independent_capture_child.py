"""Outcome-free process fixture for GHU-051 executor tests."""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import time


def _success(source_timestamp: str, race_sha: str, *, mode: str = "success") -> int:
    profile = os.environ["MANUAL_CAPTURE_PROFILE"]
    run_dir = os.environ["MANUAL_CAPTURE_RUN_DIR"]
    if not os.path.isabs(profile) or not os.path.isabs(run_dir):
        return 4
    source = b"box,dog,decimal_odds\n1,Alpha Dog,2.5\n2,Beta Dog,3.75\n"
    content_class = "prejump_form"
    content_type = "text/csv; charset=utf-8"
    if mode == "outcome-json":
        source = b'{"winners":[1],"runners":[]}\n'
        content_class = "prejump_sidecar"
        content_type = "application/json"
    elif mode == "outcome-csv":
        source = b"box,dog,decimal_odds,placings\n1,Alpha Dog,2.5,1\n"
    elif mode == "outcome-html":
        source = b'<div class="winners">Alpha Dog</div>\n'
        content_class = "prejump_race_source"
        content_type = "text/html"
    value = {
        "schema_version": "manual_independent_capture_child_fixture_v2",
        "requested_race_url": os.environ["MANUAL_CAPTURE_EXACT_URL"],
        "race_identity_sha256": race_sha,
        "runners": [
            {
                "box_number": 1,
                "display_name": "Alpha Dog",
                "identity": "ALPHA DOG",
                "source_native_runner_id": "dog-1",
                "decimal_odds": 2.5,
            },
            {
                "box_number": 2,
                "display_name": "Beta Dog",
                "identity": "BETA DOG",
                "source_native_runner_id": "dog-2",
                "decimal_odds": 3.75,
            },
        ],
        "source": {
            "content_class": content_class,
            "source_timestamp": source_timestamp,
            "final_url": os.environ["MANUAL_CAPTURE_EXACT_URL"],
            "status_code": 200,
            "content_type": content_type,
            "bytes_base64": base64.b64encode(source).decode("ascii"),
        },
    }
    sys.stdout.write(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "success",
            "outcome-json",
            "outcome-csv",
            "outcome-html",
            "malformed",
            "hang",
            "ignore-term",
            "invalid-bytes-descendant",
        ),
    )
    parser.add_argument("--source-timestamp", required=True)
    parser.add_argument("--race-sha", required=True)
    args = parser.parse_args()
    if args.mode == "success":
        return _success(args.source_timestamp, args.race_sha)
    if args.mode.startswith("outcome-"):
        return _success(args.source_timestamp, args.race_sha, mode=args.mode)
    if args.mode == "malformed":
        sys.stdout.write("not-json")
        return 0
    if args.mode in {"ignore-term", "invalid-bytes-descendant"}:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        subprocess.Popen(
            [sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    if args.mode == "invalid-bytes-descendant":
        sys.stdout.buffer.write(b"\xffnot-utf8")
        sys.stdout.buffer.flush()
        return 0
    time.sleep(60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
