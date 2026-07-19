## Summary

- add an explicit lightweight odds-only child that stops after refresh,
  strict capture, and residual handoff instead of rebuilding global reporting
  artifacts every near-minute cycle
- load the frozen feature model and read-only SQLite history once per race
  batch while keeping isolated, fail-closed per-race outputs
- reuse one Sportsbet browser session per bounded capture cycle and reset it
  after a timeout or driver exception
- make the odds-only unit's CPU and best-effort I/O priorities durable

## Why

The odds-only action was paying for the full reporting tail, reopening and
rescanning feature history for each race, and repeatedly starting Selenium.
Those operations create avoidable disk pressure and host latency even though
the fast lane only needs refresh, capture, provenance handoff, and early
residual scoring.

## Functional and safety contract

The 15-minute full daemon retains its complete dashboard, cumulative history,
drift, readiness, join, aggregate, daily status, and unified evidence outputs.
The change also preserves the 16-race cap, timer cadence, capture windows,
source fallback, WIN and PLACE validation, runner and URL checks, append-only
odds writes, hash-bound per-race provenance, and early residual scoring before
shared-lock release.

This is a draft stacked on PR #48. It should be retargeted only after PR #48
lands. It does not deploy or change the live runtime, so observed PC latency
improvement remains a separate runtime-proof gate.

## Validation

- 302 affected autopilot, daemon, feature-evaluation, and capture tests passed
- 47 residual-prediction compatibility tests passed
- Python compile passed for all changed runtime modules
- CI-critical Flake8 selection `E9,F63,F7,F82` passed with zero findings
- `git diff --check`, V2 allowed-file, report-artifact, and closeout checks passed
- final code review found no remaining critical findings, warnings, or suggestions
