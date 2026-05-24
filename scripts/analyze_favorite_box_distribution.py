#!/usr/bin/env python3
"""
Analyze favorite-by-box distribution from saved prediction JSONs.

Usage:
  python scripts/analyze_favorite_box_distribution.py \
      --paths predictions predictions/backtests \
      --glob "*.json" "*.jsonl" \
      --limit 0

Notes:
- Accepts both single-JSON files and JSON-lines files.
- Tries to find the top pick per race from keys:
  win_prob, win_probability, win_prob_norm, final_score, prediction_score, confidence
- Looks for predictions under:
  - prediction.enhanced_predictions
  - prediction.predictions
  - enhanced_predictions
  - predictions
- Prints counts and percentages per box_number.
- Also prints average top probability.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from typing import Iterable, Iterator, Tuple


def iter_files(paths: list[str], patterns: list[str]) -> Iterator[str]:
    if not paths:
        paths = ["predictions"]
    if not patterns:
        patterns = ["*.json", "*.jsonl"]
    seen: set[str] = set()
    for base in paths:
        for pat in patterns:
            for p in glob.glob(os.path.join(base, pat)):
                ap = os.path.abspath(p)
                if ap not in seen:
                    seen.add(ap)
                    yield ap


def _top_prob(p: dict) -> float:
    for k in ("win_prob", "win_probability", "win_prob_norm", "final_score", "prediction_score", "confidence"):
        v = p.get(k)
        if v is None:
            continue
        try:
            f = float(v)
            # Scale down if it looks like a percentage > 1.5
            if f > 1.5:
                f = f / 100.0
            return max(0.0, min(1.0, f))
        except Exception:
            continue
    return 0.0


def _extract_pred_list(obj: dict) -> list[dict]:
    if not isinstance(obj, dict):
        return []
    # Prefer nested under prediction
    pred = obj.get("prediction") if isinstance(obj.get("prediction"), dict) else None
    if pred:
        for key in ("enhanced_predictions", "predictions"):
            if isinstance(pred.get(key), list):
                return [x for x in pred.get(key) if isinstance(x, dict)]
    # Fallback to top-level
    for key in ("enhanced_predictions", "predictions"):
        if isinstance(obj.get(key), list):
            return [x for x in obj.get(key) if isinstance(x, dict)]
    return []


def _iter_json_records(path: str) -> Iterator[dict]:
    # Try to load single JSON
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read(1024)
            # Quick heuristic for JSONL (multiple lines starting with '{')
            if "\n{" in txt:
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue
                return
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            yield obj
            return
    except Exception:
        return


def analyze(paths: list[str], patterns: list[str], limit: int = 0) -> Tuple[Counter, int, float]:
    counts = Counter()
    total = 0
    sum_top_prob = 0.0
    for i, file_path in enumerate(iter_files(paths, patterns), 1):
        if limit and i > limit:
            break
        for rec in _iter_json_records(file_path):
            preds = _extract_pred_list(rec)
            if not preds:
                continue
            # Choose top by highest win prob
            try:
                top = max(preds, key=_top_prob)
            except ValueError:
                continue
            box = top.get("box_number") or top.get("box")
            try:
                box_int = int(box)
            except Exception:
                # Skip if cannot determine box
                continue
            counts[box_int] += 1
            total += 1
            sum_top_prob += _top_prob(top)
    avg_top_prob = (sum_top_prob / total) if total > 0 else 0.0
    return counts, total, avg_top_prob


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze favorite-by-box distribution from saved prediction JSONs")
    ap.add_argument("--paths", nargs="*", default=["predictions"], help="Directories to scan (default: predictions)")
    ap.add_argument("--glob", dest="patterns", nargs="*", default=["*.json", "*.jsonl"], help="Filename patterns (default: *.json *.jsonl)")
    ap.add_argument("--limit", type=int, default=0, help="Max number of files to scan (0 = no limit)")
    args = ap.parse_args()

    counts, total, avg_top_prob = analyze(args.paths, args.patterns, args.limit)

    print("\nFavorite-by-box distribution")
    print("==========================")
    if total == 0:
        print("No favorites found in the provided files.")
        return 0

    for box in sorted(counts.keys()):
        c = counts[box]
        pct = (c / total) * 100.0
        print(f"Box {box}: {c} ({pct:.1f}%)")

    box1 = counts.get(1, 0)
    pct1 = (box1 / total) * 100.0
    print(f"\nTotal races analyzed: {total}")
    print(f"Box 1 share: {box1} ({pct1:.1f}%)")
    print(f"Average top probability: {avg_top_prob:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

