#!/usr/bin/env python3
"""
Evaluate baseline vs candidate metrics and emit SUCCESS or ROLLBACK based on thresholds.

Usage:
  python scripts/evaluate_accuracy_thresholds.py \
    --baseline artifacts/eval/backtest_win.json \
    --candidate artifacts/eval/backtest_win_optimizer.json \
    --place artifacts/eval/backtest_place.json \
    --calibration calibration_results.json \
    --output artifacts/eval/accuracy_report.md

Decision:
- Calibration must pass thresholds (win and, if present, place):
  - Win:   Brier ≤ 0.18 and |slope-1| ≤ 0.2
  - Place: Brier ≤ 0.20 and |slope-1| ≤ 0.25
- AND any improvement holds:
  - top1_accuracy +≥ 0.02 absolute OR
  - top3_hit_rate +≥ 0.03 absolute OR
  - log_loss −≥ 5% relative
Outputs:
- Writes the report to --output (markdown)
- Writes artifacts/eval/status.txt with either SUCCESS or ROLLBACK
- Exits 0 on SUCCESS, 2 on ROLLBACK
"""
import argparse
import json
import math
from pathlib import Path


def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def within(v, lo, hi) -> bool:
    try:
        return v is not None and (lo <= float(v) <= hi)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--place", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)
    place = load_json(args.place)
    calib = load_json(args.calibration)

    # Extract metrics (robustly)
    b_top1 = float(baseline.get("top1_accuracy", 0.0) or 0.0)
    c_top1 = float(candidate.get("top1_accuracy", 0.0) or 0.0)
    b_ll = float(baseline.get("log_loss", math.nan))
    c_ll = float(candidate.get("log_loss", math.nan))

    # Place (Top-3) metrics: baseline from place.json; candidate from candidate JSON if present
    b_top3 = place.get("top3_hit_rate")
    if b_top3 is not None:
        b_top3 = float(b_top3)
    c_top3 = candidate.get("top3_hit_rate")
    if c_top3 is not None:
        c_top3 = float(c_top3)

    # Improvements
    top1_delta = c_top1 - b_top1
    top3_delta = None if (b_top3 is None or c_top3 is None) else (c_top3 - b_top3)
    ll_rel = None
    if (not math.isnan(b_ll)) and (not math.isnan(c_ll)) and b_ll > 0:
        ll_rel = (b_ll - c_ll) / b_ll  # positive means better (lower log loss)

    # Calibration thresholds
    win_brier = calib.get("win_brier_score")
    place_brier = calib.get("place_brier_score")
    slope_win = calib.get("reliability_slope_win")
    slope_place = calib.get("reliability_slope_place")

    win_ok = (win_brier is not None and float(win_brier) <= 0.18) and (
        slope_win is not None and abs(float(slope_win) - 1.0) <= 0.2
    )
    place_ok = True  # default to True if place metrics are absent
    if place_brier is not None and slope_place is not None:
        place_ok = (float(place_brier) <= 0.20) and (abs(float(slope_place) - 1.0) <= 0.25)

    calib_ok = win_ok and place_ok

    # Any improvement condition
    any_improve = False
    reasons = []
    if top1_delta >= 0.02:
        any_improve = True
        reasons.append(f"Win top1 +{top1_delta:.3f} (≥ 0.020)")
    if (top3_delta is not None) and (top3_delta >= 0.03):
        any_improve = True
        reasons.append(f"Place top3 +{top3_delta:.3f} (≥ 0.030)")
    if (ll_rel is not None) and (ll_rel >= 0.05):
        any_improve = True
        reasons.append(f"Log loss −{ll_rel*100:.2f}% (≥ 5.00%)")

    status = "SUCCESS" if (calib_ok and any_improve) else "ROLLBACK"

    # Compose report
    lines = []
    lines.append("# Accuracy Threshold Evaluation\n")
    lines.append("## Inputs\n")
    lines.append(f"- Baseline: {args.baseline}")
    lines.append(f"- Candidate: {args.candidate}")
    lines.append(f"- Place: {args.place}")
    lines.append(f"- Calibration: {args.calibration}\n")

    lines.append("## Metrics\n")
    lines.append(f"- Baseline Top-1: {b_top1:.3f}")
    lines.append(f"- Candidate Top-1: {c_top1:.3f} (Δ {top1_delta:+.3f})")
    if b_top3 is not None:
        lines.append(f"- Baseline Top-3: {b_top3:.3f}")
    if c_top3 is not None:
        lines.append(f"- Candidate Top-3: {c_top3:.3f}{'' if top3_delta is None else f' (Δ {top3_delta:+.3f})'}")
    if (not math.isnan(b_ll)) and (not math.isnan(c_ll)):
        lines.append(f"- Baseline LogLoss: {b_ll:.6f}")
        lines.append(f"- Candidate LogLoss: {c_ll:.6f}{'' if ll_rel is None else f' (−{ll_rel*100:.2f}%)'}")

    lines.append("\n## Calibration\n")
    lines.append(f"- Win: Brier={win_brier}, slope={slope_win}")
    lines.append(f"- Place: Brier={place_brier}, slope={slope_place}")
    lines.append(f"- Calibration Pass: {'YES' if calib_ok else 'NO'}\n")

    lines.append("## Decision\n")
    if status == "SUCCESS":
        lines.append("- Status: SUCCESS")
        if reasons:
            lines.append("- Reasons:")
            for r in reasons:
                lines.append(f"  - {r}")
    else:
        lines.append("- Status: ROLLBACK")
        lines.append("- Reasons:")
        if not calib_ok:
            lines.append("  - Calibration thresholds not satisfied")
        if not any_improve:
            lines.append("  - No qualifying improvement (top1 +≥0.02, top3 +≥0.03, or log-loss −≥5%)")

    # Write report
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(lines) + "\n")

    # Write status file
    status_file = Path("artifacts/eval/status.txt")
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(status + "\n")

    print(status)
    if status == "SUCCESS":
        raise SystemExit(0)
    raise SystemExit(2)


if __name__ == "__main__":
    main()

