# Validation

## Code and contract validation

- Task-card validation: passed; scope fingerprint
  `5f6b64a02bdc64f18f09b06a39ff56634be0aca138f5d8ca864377aed5391057`.
- RED/GREEN: the new lightweight flag, batch scorer, feature-batch plan, and
  fetch-session contract failed before implementation and passed afterward.
- Affected suite: `302 passed in 39.66s` across autopilot, daemon, feature
  evaluation, and autonomous capture tests.
- Residual scorer compatibility: `47 passed in 12.66s`.
- Review-edge regressions: `3 passed in 33.73s` for failed-fetch setup
  accounting, evidence-root report containment, and generated service policy.
- Python compile for all five changed runtime modules: passed.
- CI-critical Flake8 selection `E9,F63,F7,F82`: passed with zero findings.
- `git diff --check`: passed.
- Existing repository-wide Black baseline: not clean on the touched legacy
  modules; no bulk whole-file reformat was applied because it would widen the
  approved diff. This is not a CI-critical gate in the repository workflow.
- Code review: no remaining critical findings, warnings, or suggestions.

## Preserved contracts

- Lightweight mode calls zero full dashboard, cumulative-history, drift,
  readiness, join, aggregate, daily-status, or unified-dataset builders.
- Full/default mode remains unchanged and its existing tests pass.
- The feature batch records one model load and one read-only SQLite history
  load, retains isolated race output directories, and continues after a
  per-race failure.
- Browser reuse records setup and restart counts; timeout and exception paths
  reset before continuing.
- Early residual scoring remains before shared-lock release.
- Generated and static odds service contracts use best-effort I/O priority 7
  with CPU and I/O weights. Static service SHA-256:
  `7bed9a5a1b9eecca21dd9072e2aaff70ea272373e8a5a612283a0d6def4d09dc`.

## Runtime boundary

Validation used synthetic fixtures and repository tests only. No production
database, runtime artifacts, live browser, installed unit, timer, lock, model
pointer, or daemon was inspected or changed. See `RUNTIME_PROOF.md`.
