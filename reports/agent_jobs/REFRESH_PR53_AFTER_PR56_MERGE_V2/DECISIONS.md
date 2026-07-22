# Decisions

1. Reused the exact eight-file implementation commit from PR #53 rather than
   cherry-picking its 42-commit branch history.
2. Kept all PR #56-owned handoff, scorer, refresh, browser, CSV metadata, tests,
   and documentation at master and adapted the on-demand seam to them.
3. Rejected stale dependency imports and did not import old reports, task cards,
   scorer/writer copies, services, timers, daemon/evaluation code, outcomes, or
   production writers.
4. Kept receipt reuse read-only and direct capture isolated, fixed-window gated,
   no-steal, and bound to the selected database root's canonical daemon lock.
5. Sealed only pre-target-date history and selected full outcome-bearing rows
   only after race IDs passed the cutoff classification.
6. Contained legacy relative import-time writes in transient bundle scratch and
   documented `uv run --no-project` to prevent repository lockfile creation.
7. Classified the live proof as `DATA_MISSING`; no retry or capture was made.
