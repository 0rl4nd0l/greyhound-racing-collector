#!/usr/bin/env python3
"""
CI gate for MLSystemV4: include favorite-by-box guardrail metric
- Loads recent prediction JSONs from an evaluation folder
- Computes Box 1 share among favorites
- Fails promotion if share exceeds configured threshold
"""
import os
import sys
import json
from pathlib import Path

EVAL_DIR = Path(os.environ.get("PREDICTIONS_EVAL_DIR", "predictions"))
BOX1_MAX_SHARE = float(os.environ.get("BOX1_MAX_SHARE", "0.50"))
MIN_FILES = int(os.environ.get("EVAL_MIN_FILES", "50"))


def _top_prob(p: dict) -> float:
    for k in ("win_prob", "win_probability", "win_prob_norm", "final_score", "prediction_score", "confidence"):
        v = p.get(k)
        if v is None:
            continue
        try:
            f = float(v)
            if f > 1.5:
                f = f / 100.0
            return max(0.0, min(1.0, f))
        except Exception:
            continue
    return 0.0


def _extract_preds(obj: dict):
    if not isinstance(obj, dict):
        return []
    pred = obj.get("prediction") if isinstance(obj.get("prediction"), dict) else None
    if pred:
        for key in ("enhanced_predictions", "predictions"):
            if isinstance(pred.get(key), list):
                return [x for x in pred.get(key) if isinstance(x, dict)]
    for key in ("enhanced_predictions", "predictions"):
        if isinstance(obj.get(key), list):
            return [x for x in obj.get(key) if isinstance(x, dict)]
    return []


def main():
    if not EVAL_DIR.exists():
        print(f"EVAL_DIR not found: {EVAL_DIR}")
        sys.exit(0)  # no-op in environments without eval data
    files = sorted([p for p in EVAL_DIR.glob("*.json") if p.is_file()])
    if len(files) < MIN_FILES:
        print(f"Not enough eval files ({len(files)} < {MIN_FILES}); skipping gate")
        sys.exit(0)

    box1 = 0
    total = 0
    for fp in files[:500]:
        try:
            data = json.loads(fp.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        preds = _extract_preds(data)
        if not preds:
            continue
        try:
            top = max(preds, key=_top_prob)
        except ValueError:
            continue
        box = top.get("box_number") or top.get("box")
        try:
            box = int(box)
        except Exception:
            continue
        total += 1
        if box == 1:
            box1 += 1

    if total < MIN_FILES:
        print(f"Insufficient usable eval files after parsing: {total}")
        sys.exit(0)

    share = box1 / total if total else 0.0
    print(f"Box1 favorite share: {share:.2%} over {total} files (threshold {BOX1_MAX_SHARE:.0%})")
    if share > BOX1_MAX_SHARE:
        print("❌ Gate failed: excessive Box 1 dominance")
        sys.exit(2)
    print("✅ Gate passed")
    sys.exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CI promotion gate for ML System V4.

- Runs race-level evaluation with current DB
- Enforces thresholds controlled by env vars (with safe defaults)
- Exits non-zero if thresholds are not met (CI-friendly)

Env (with defaults):
- V4_DISABLE_ACCURACY_OPTIMIZER=1 (should be 1; gate expects core V4 path)
- V4_NORMALIZATION_MODE=simple
- BRIER_MAX=0.13
- LOGLOSS_MAX=0.42
- TOP1_MIN=0.27
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def run_eval(n_races: int = 200, order: str = "recent") -> dict:
    env = os.environ.copy()
    env.setdefault("V4_DISABLE_ACCURACY_OPTIMIZER", "1")
    env.setdefault("V4_NORMALIZATION_MODE", "simple")
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "evaluate_race_level_v4.py"),
        "--n-races",
        str(n_races),
        "--order",
        order,
    ]
    proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)
    try:
        data = json.loads(proc.stdout.strip())
        return data
    except Exception:
        print(proc.stdout)
        print("Could not parse evaluator JSON from stdout", file=sys.stderr)
        raise SystemExit(2)


def main() -> int:
    # Thresholds (defaults tuned to current baseline with small slack)
    brier_max = float(os.getenv("BRIER_MAX", "0.13"))
    logloss_max = float(os.getenv("LOGLOSS_MAX", "0.42"))
    top1_min = float(os.getenv("TOP1_MIN", "0.27"))

    metrics = run_eval()
    print(json.dumps({"gate_metrics": metrics}, indent=2))

    acc = float(metrics.get("top1_accuracy", 0.0))
    brier = float(metrics.get("brier", 1.0))
    logloss = float(metrics.get("log_loss", 10.0))

    failed = []
    if brier > brier_max:
        failed.append(f"Brier {brier:.6f} > {brier_max:.6f}")
    if logloss > logloss_max:
        failed.append(f"LogLoss {logloss:.6f} > {logloss_max:.6f}")
    if acc < top1_min:
        failed.append(f"Top1 {acc:.6f} < {top1_min:.6f}")

    if failed:
        print("PROMOTION GATE: FAIL", file=sys.stderr)
        for f in failed:
            print(" - ", f, file=sys.stderr)
        return 3

    print("PROMOTION GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

