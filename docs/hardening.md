# Production Hardening and Safety Defaults

This repository enforces a strict “no fabricated outputs” policy in production paths. All mock, placeholder, or random-based logic is either removed or explicitly gated behind development-only environment flags. API endpoints must fail fast (HTTP 503) when predictors are unavailable rather than returning fabricated predictions.

Key changes and defaults:

- API no-mock guarantee
  - The FastAPI endpoint /api/predict_single_race_enhanced will not generate mock predictions. If predictors are unavailable, it returns HTTP 503 with a clear message.

- Dev-only fallbacks (all default OFF)
  - UNIFIED_ALLOW_BASIC_FALLBACK: Enables basic randomized fallback in unified_predictor.py. Default: 0
  - ML_V4_ALLOW_HEURISTIC: Enables single-dog heuristic prediction in ml_system_v4.py when no model is loaded. Default: 0
  - ML_V4_ALLOW_SIMULATED_ODDS: Allows simulated odds during EV threshold learning in ml_system_v4.py. Default: 0
  - TGR_ALLOW_PLACEHOLDER: Enables placeholder race insights in TheGreyhoundRecorder scraper. Default: 0
  - ALLOW_SYNTHETIC_TEST_MODEL: Allows scripts/train_test_model.py to run with synthetic data. Default: 0

- Drift detection safety
  - If SciPy/pandas/numpy are missing, drift detection is disabled and logged as DRIFT_DETECTION_DISABLED rather than returning a false “no drift” signal.

- FastTrack feature builder placeholders
  - All placeholder feature-builder functions raise RuntimeError by default to prevent fabricated features sneaking into production paths.

CI/pre-commit guard

- The hardening guard runs on push/PR via .github/workflows/hardening.yml and locally via pre-commit.
- It checks:
  - No mock artifacts in FastAPI main.
  - Dev gating present for UnifiedPredictor basic fallback and ML V4 heuristics/simulated odds.
  - Drift detection uses a safe disabled-mode fallback.
  - FastTrack builder placeholders are disabled.
  - random.uniform is not present in production paths (except explicitly allowlisted dev fallback in unified_predictor.py).

How to enable a dev-only fallback locally

- Export the appropriate environment variable before running a script or server. Examples:
  - macOS/Linux
    - export UNIFIED_ALLOW_BASIC_FALLBACK=1
    - export ML_V4_ALLOW_HEURISTIC=1
    - export ML_V4_ALLOW_SIMULATED_ODDS=1
    - export TGR_ALLOW_PLACEHOLDER=1
    - export ALLOW_SYNTHETIC_TEST_MODEL=1
  - Windows PowerShell
    - $env:UNIFIED_ALLOW_BASIC_FALLBACK='1'
    - $env:ML_V4_ALLOW_HEURISTIC='1'

FAQs

- Why not keep a universal random fallback?
  - Fabricated outputs can poison downstream systems and mislead users. The safe default is to fail fast with a 503 and a clear message.

- How do I run the hardening guard locally?
  - python scripts/ci/check_hardening_guards.py
  - Or install pre-commit and run pre-commit run --all-files

- Will this break tests that depend on dev fallbacks?
  - Tests that require fallbacks should set the corresponding env vars in their setup/fixtures. The production default remains OFF.

