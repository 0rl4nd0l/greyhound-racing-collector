import json
import os
from pathlib import Path

import pytest

EVAL_DIR = Path(os.environ.get("PREDICTIONS_EVAL_DIR", "predictions"))
BOX1_MAX_SHARE = float(os.environ.get("BOX1_MAX_SHARE", "0.50"))  # 50% default threshold
MIN_FILES = int(os.environ.get("EVAL_MIN_FILES", "20"))


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


@pytest.mark.skipif(not EVAL_DIR.exists(), reason="evaluation directory not found")
def test_favorite_box1_share_under_threshold():
    files = sorted([p for p in EVAL_DIR.glob("*.json") if p.is_file()])
    if len(files) < MIN_FILES:
        pytest.skip(f"not enough files to evaluate (have {len(files)}, need {MIN_FILES})")

    box1 = 0
    total = 0
    for fp in files[:200]:
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
        pytest.skip(f"insufficient usable files after parsing (have {total}, need {MIN_FILES})")

    share = box1 / total if total else 0.0
    assert share <= BOX1_MAX_SHARE, (
        f"Box 1 favorites share too high: {share:.2%} > {BOX1_MAX_SHARE:.0%} over {total} files."
    )