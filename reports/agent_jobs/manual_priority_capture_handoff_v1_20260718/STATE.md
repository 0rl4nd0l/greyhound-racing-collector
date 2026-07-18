# State

## Before

The manual command could only wait for or acquire the shared writer lock. When
the autonomous daemon appended the exact target receipt while retaining that
lock, the manual command could not consume the completed capture.

## After

- Finalized autonomous receipts are discoverable by exact race and current
  T-60/T-30/T-10/T-2 window.
- Producer final-marker, plan, report, URL, runner/box, timestamps, fetch
  provenance, append-only counts, WIN/PLACE values, and exact SQLite group are
  fail-closed gates.
- Accepted report/plan/form/sidecar bytes are read once, hashed, and staged in a
  private temporary directory.
- Reuse does not acquire, release, steal, or bypass the daemon lock and does not
  refresh, scrape, or append.
- `--require-autonomous-handoff` makes receipt-only execution incapable of
  falling through to the direct writer path.
- PR #48 and its runtime remain unchanged and read-only.

## Stop state

`BLOCKED_TASK_CONTRACT`

The optional live score was not run because the existing feature builder reads
historical result columns outside this card's exact production-data allowlist.
No target result or outcome was queried.

Resume only under a separate owner-approved V2 card that explicitly permits
the necessary query-only historical feature read with target-race exclusion,
or supplies an independently authorized presealed outcome-free feature packet,
and only if a new exact receipt remains pre-jump. Do not reuse this claim.

