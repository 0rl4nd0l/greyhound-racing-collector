#!/usr/bin/env python3
"""
Export per-model drift metrics to model_registry/metrics/<model_id>_drift.json

Usage:
  python scripts/export_model_drift.py --model-id MODEL_ID [--score 0.35] [--history 10] [--window-hours 24]
  python scripts/export_model_drift.py --model-id MODEL_ID --from-file path/to/drift.json

Notes:
- If --from-file is provided, it should contain {"drift_results": {...}} or the script will wrap content into that structure.
- Without --from-file, the script will generate a synthetic history around --score.
- The API /api/model/monitoring/drift will pick up this file automatically when Details modal passes model_id.
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys


def load_input_drift(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "drift_results" in data:
            return data
        # Wrap arbitrary payload
        return {"drift_results": data}
    except Exception as e:
        raise SystemExit(f"Failed to load drift input file: {e}")


def generate_drift(score: float, history_points: int, window_hours: int) -> dict:
    now = datetime.now()
    base = max(0.01, min(0.99, float(score)))
    hist = []
    for i in range(history_points):
        t = now - timedelta(hours=(history_points - 1 - i) * max(1, window_hours // max(1, history_points - 1)))
        # small deterministic oscillation
        offset = ((i % 2) - 0.5) * 0.05
        s = round(max(0.01, min(0.99, base + offset)), 3)
        hist.append({"date": t.isoformat(), "drift_score": s})
    return {
        "drift_results": {
            "drift_detected": base > 0.5,
            "drift_score": round(base, 3),
            "history": hist,
        }
    }


def export_drift(model_id: str, payload: dict) -> Path:
    out_dir = Path("model_registry") / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_id}_drift.json"
    try:
        payload = dict(payload or {})
        # Annotate metadata
        payload["model_id"] = model_id
        payload["generated_at"] = datetime.now().isoformat()
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out_path
    except Exception as e:
        raise SystemExit(f"Failed to write drift file: {e}")


def main():
    p = argparse.ArgumentParser(description="Export per-model drift metrics")
    p.add_argument("--model-id", required=True, help="Registry model_id")
    p.add_argument("--from-file", help="Path to input JSON containing drift_results")
    p.add_argument("--score", type=float, default=0.35, help="Base drift score (0..1)")
    p.add_argument("--history", type=int, default=10, help="Number of history points")
    p.add_argument("--window-hours", type=int, default=24, help="Window hours for synthetic history spacing")
    args = p.parse_args()

    if args.from_file:
        in_path = Path(args.from_file)
        if not in_path.exists():
            raise SystemExit(f"Input file not found: {in_path}")
        payload = load_input_drift(in_path)
        # attach window_hours hint
        payload.setdefault("window_hours", args.window_hours)
    else:
        payload = generate_drift(score=args.score, history_points=args.history, window_hours=args.window_hours)
        payload["window_hours"] = args.window_hours

    out = export_drift(args.model_id, payload)
    print(str(out))


if __name__ == "__main__":
    main()

