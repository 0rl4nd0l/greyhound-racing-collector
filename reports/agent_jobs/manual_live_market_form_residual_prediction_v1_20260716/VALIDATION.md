# Validation

- Prototype focused suite: 44 passed.
- Ruff: passed.
- `py_compile`: passed.
- `git diff --check`: passed.
- V2 task-card validation: passed.
- V2 diff allowlist: passed.
- Independent review: completed with four release-blocking findings.

These green checks validate the prototype's internal behavior only. They do
not establish feature-source parity, so they are not a readiness claim.

Read-only source proof:

- Exact R2 feature rows: 8.
- Feature freeze: `2026-07-16T18:18:32.550267+10:00`.
- Feature packet SHA-256:
  `0ebecaf980665545aa8c19d1a4b1ef976bd069049d42f7f6ebde0f3b29a36b62`.
- Shadow manifest SHA-256:
  `7096c534de850f3ee12ef7dab8a133d38990cefff22e22bb0cfc9ece19c85e4b`.
- Implementation manifest SHA-256:
  `9822a77a4d69a72c8b7b2e7d234538b6207b99530b3b717fb9cb31f64929a651`.

## Runtime Functionality Proof

- Intended output: one exact feature-packet-bound frozen residual prediction.
- Live output location: stdout only; no persistent live output was authorized.
- Pre-run max timestamp or count: 0 persisted manual residual predictions.
- Post-run max timestamp or count: 0 persisted manual residual predictions.
- Rows/files inserted or updated after run start: 0 runtime rows and 0 runtime files.
- Readiness/gate status: blocked before publication by feature-source parity review.
- Exact command/query used: `uv run --no-project --with 'numpy==1.26.4' --with pytest python -m pytest -q --noconftest tests/test_predict_market_form_residual.py tests/test_market_form_residual.py`.
- Result: `PARTIAL`.
- Remaining blocker: replace form-only recomputation with the exact sealed system feature packet and validate immutable input binding.
