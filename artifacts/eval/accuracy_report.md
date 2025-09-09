# Accuracy Threshold Evaluation

## Inputs

- Baseline: /Users/test/Desktop/greyhound_racing_collector/artifacts/eval/backtest_win.json
- Candidate: /Users/test/Desktop/greyhound_racing_collector/artifacts/eval/backtest_win_optimizer.json
- Place: /Users/test/Desktop/greyhound_racing_collector/artifacts/eval/backtest_place.json
- Calibration: /Users/test/Desktop/greyhound_racing_collector/calibration_results.json

## Metrics

- Baseline Top-1: 0.320
- Candidate Top-1: 0.320 (Δ +0.000)
- Baseline Top-3: 0.700
- Candidate Top-3: 0.700 (Δ +0.000)
- Baseline LogLoss: 0.400271
- Candidate LogLoss: 0.399618 (−0.16%)

## Calibration

- Win: Brier=0.04254640135661972, slope=0.8727622806511044
- Place: Brier=0.15786663773975132, slope=0.9291836141437234
- Calibration Pass: YES

## Decision

- Status: ROLLBACK
- Reasons:
  - No qualifying improvement (top1 +≥0.02, top3 +≥0.03, or log-loss −≥5%)
