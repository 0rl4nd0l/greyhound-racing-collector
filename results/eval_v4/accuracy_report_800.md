# Accuracy Threshold Evaluation

## Inputs

- Baseline: /Users/test/Desktop/greyhound_racing_collector/results/eval_v4/baseline_800.json
- Candidate: /Users/test/Desktop/greyhound_racing_collector/results/eval_v4/candidate_800.json
- Place: /Users/test/Desktop/greyhound_racing_collector/results/eval_v4/baseline_800.json
- Calibration: /Users/test/Desktop/greyhound_racing_collector/calibration_results.json

## Metrics

- Baseline Top-1: 0.350
- Candidate Top-1: 0.350 (Δ +0.000)
- Baseline Top-3: 0.680
- Candidate Top-3: 0.680 (Δ +0.000)
- Baseline LogLoss: 0.403702
- Candidate LogLoss: 0.402614 (−0.27%)

## Calibration

- Win: Brier=0.03668897903913648, slope=0.8327848902116839
- Place: Brier=0.15806722740303375, slope=1.0352240559985773
- Calibration Pass: YES

## Decision

- Status: ROLLBACK
- Reasons:
  - No qualifying improvement (top1 +≥0.02, top3 +≥0.03, or log-loss −≥5%)
