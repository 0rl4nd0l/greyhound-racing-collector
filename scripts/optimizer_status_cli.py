#!/usr/bin/env python3
"""
optimizer_status_cli.py

Lightweight CLI to query the optimizer status endpoint.

Usage examples:
  # Default localhost:5002
  python3 scripts/optimizer_status_cli.py

  # Summary-only view (single-line fields)
  python3 scripts/optimizer_status_cli.py --summary

  # Custom host/port
  python3 scripts/optimizer_status_cli.py --host 127.0.0.1 --port 5002

  # Override full URL
  python3 scripts/optimizer_status_cli.py --url http://localhost:5002/api/optimizer/status

Environment:
  - PORT or DEFAULT_PORT can be used to determine default port if not provided (falls back to 5002)
"""
import argparse
import json
import os
import sys
import urllib.request


def fetch_status(url: str, timeout: float = 5.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read()
            return json.loads(body.decode("utf-8", errors="ignore"))
    except Exception as e:
        print(f"Error: failed to fetch {url}: {e}", file=sys.stderr)
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser(
        description="Query the optimizer status endpoint and print JSON or a summary.",
    )
    parser.add_argument(
        "--host", default="localhost", help="API host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", os.environ.get("DEFAULT_PORT", "5002"))),
        help="API port (default: env PORT/DEFAULT_PORT or 5002)",
    )
    parser.add_argument(
        "--path",
        default="/api/optimizer/status",
        help="API path (default: /api/optimizer/status)",
    )
    parser.add_argument(
        "--url", help="Override full URL (takes precedence over host/port/path)"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a concise summary instead of full JSON",
    )
    args = parser.parse_args()

    url = args.url or f"http://{args.host}:{args.port}{args.path}"
    data = fetch_status(url)

    if args.summary:
        last = (data or {}).get("last_prediction") or {}
        mids = last.get("model_ids_used") or []
        print(f"service_available: {data.get('service_available')}")
        print(f"ensemble_mode: {data.get('ensemble_mode')}")
        print(f"registry_best_id: {data.get('registry_best_id')}")
        print(f"last_race_id: {last.get('race_id')}")
        print(f"predictions_count: {last.get('predictions_count')}")
        print(f"ensemble_models: {last.get('ensemble_models')}")
        print(f"last_model_ids_used: {', '.join(mids) if mids else '-'}")
        print(f"timestamp: {data.get('timestamp')}")
        return

    print(json.dumps(data, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
