# Runtime proof

## Runtime Functionality Proof

- Intended output: one exact-race, outcome-free, normalized full/half prediction
  on canonical stdout after strict target refresh and capture.
- Live output location: `/tmp/gh-priority-live.stdout` during this bounded run;
  the durable redacted result is `optional-manual-proof.json`.
- Pre-run max timestamp or count: zero `live_odds` rows for the exact target.
- Post-run max timestamp or count: zero `live_odds` rows for the exact target.
- Rows/files inserted or updated after run start: zero canonical rows and zero
  surviving feature or prediction files.
- Readiness/gate status: `BLOCKED_RUNNER_IDENTITY`.
- Exact command/query used: `scripts/run_priority_race_prediction.py --race-id
  'Race 1 - BAL - 2026-07-18' --execute-collection
  --allow-auto-scrape-odds --max-wait-seconds 600` with the canonical DB and
  shared lock paths.
- Result: `PARTIAL`.
- Remaining blocker: Sportsbet supplied no source URL and no accepted WIN or
  PLACE runner rows for the exact target.

At 2026-07-18 15:51 AEST, the plan-only command resolved Ballarat R1 with a
16:57 AEST jump and a T-60 target of 15:57. Stdout contained exactly one
canonical JSON line and reported `PLAN_ONLY`, `persisted:false`, and
`result_access:false`.

At T-60, one authorized execution was started with a 600-second maximum lock
wait and the canonical shared lock. The installed daemon was naturally active;
it was not interrupted. The command later acquired the lock, performed the
target-only refresh, and stopped at strict capture validation. Its sole stdout
object reported `BLOCKED_RUNNER_IDENTITY`, zero accepted runner rows, missing
Sportsbet source URL, `persisted:false`, and `result_access:false`.

Post-run checks showed the shared lock absent and zero canonical `live_odds`
rows for `Race 1 - BAL - 2026-07-18`. No feature packet survived the temporary
directory and no prediction was emitted or persisted.
