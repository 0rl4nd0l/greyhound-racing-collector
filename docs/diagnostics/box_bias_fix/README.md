# Box-1 Bias Fix Diagnostic

This note documents the investigation and the initial remediation applied to eliminate the UI/ordering bias that surfaced Box 1 as the top pick too often.

## Summary
- Root cause: probability/rank gaps in responses combined with a UI default sort of `predicted_rank` caused the first runner (often Box 1) to render as “Top Pick”, even when probabilities were non‑uniform.
- Fixes applied:
  - Backend (adapters and unified predictor): always emit `predicted_rank` and probability aliases (`win_prob`, `win_prob_norm`, `normalized_win_probability`, `win_probability`).
  - Frontend (Interactive Races + Prediction Buttons): default sort is now probability-first; score extractor recognizes `win_prob_norm`.

## Baseline (before re-running predictions)
A scan over existing prediction files in `predictions/` (historical) shows:

```
Box 1 share: 2441 / 2939 = 83.1%
Average top probability: 0.308
```

These are historical artefacts produced prior to the fixes and reflect the bias we are addressing.

## How to produce a fresh, post-fix snapshot
1) Archive old predictions so we analyze only new outputs:
   - mkdir -p predictions/archive
   - mv predictions/prediction_*.json predictions/archive/ 2>/dev/null || true

2) Generate new predictions (ensure UPCOMING_RACES_DIR contains current CSVs):
   - make deps
   - export UPCOMING_RACES_DIR=${UPCOMING_RACES_DIR:-./upcoming_races_temp}
   - Run predictions via the UI (Download + Predict) or with your usual CLI.

3) Re-run the analyzer to compute the fresh distribution:
   - ./.venv/bin/python scripts/analyze_favorite_box_distribution.py --paths predictions --glob "*.json" "*.jsonl" --limit 0

4) Expected outcome
   - Box 1 share should no longer be artificially dominant due to ordering; distribution should reflect true model strengths. The precise share depends on the race mix, but it should be far below ~80%.

## Tests
- Added tests to assert UI defaults and probability recognition:
  - tests/test_ui_ordering_defaults.py
    - Default sort is probability-based
    - Recognizes `win_prob_norm` in extractors
- A backend-level regression test over saved predictions was added:
  - tests/test_prediction_files_monotonic.py
  - Note: Running the full test suite currently imports app.py via tests/conftest.py and may fail if that module parses incorrectly in your environment; fix pending. The test itself only reads JSON predictions and checks rank monotonicity.

## Next actions
- Reproduce new predictions and commit a post-fix distribution to this folder for comparison (e.g., `after_distribution.txt`).
- If degraded/synthetic paths are encountered frequently, consider adding a seeded uniform jitter to degraded responses so no input-order bias can ever leak into the UI, even on failure.

