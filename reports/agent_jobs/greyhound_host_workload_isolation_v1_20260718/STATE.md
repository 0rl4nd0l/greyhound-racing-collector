# State

result: WORKING

## Before

- Canonical base: `origin/master` at
  `c1dfd464cf6ecfb2034f96ac1a8d3ea58d4e6afa`.
- The established incident evidence showed that heavy offline scans lacked a
  durable exact-root and physical-I/O admission boundary.
- The user systemd manager exposed only `memory pids` controllers, so a user
  transient scope could not prove `io.max` on this host.
- Launch checkout
  `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector`
  remained on its existing branch and commit and was not repurposed.

## After

- Task branch: `codex/host-workload-isolation-v1-20260718` in the clean sibling
  worktree `greyhound-host-workload-isolation-v1-20260718`.
- `scripts/run_bounded_offline.py` supplies exact-root validation, fixed command
  vocabulary, one worker, default exclusions, bounded timeout/throughput/IOPS,
  low priority, read-only mounting, interruption cleanup, and fail-closed
  cgroup read-back.
- Docker 29.1.3 uses cgroup v2 with the systemd driver on this host and applied
  `259:2 rbps=1048576 riops=16` during the representative minimum-limit run.
- Eighteen focused tests passed. Unsafe roots returned 2. The controlled scan
  completed with the expected no-match exit 1 and left no task container.
- Two natural odds-capture executions completed successfully during the bounded
  scan. No completed full cycle occurred in the bounded observation window.
- No live unit, timer, service, collector process, SQLite database, production
  evidence, model, secret, or unrelated Docker workload was mutated.

## Control-plane state

- V2 task card and shared registry claim: active during validation.
- Portable guard: `ALLOW_NEW_SCOPE`; code/report work permitted.
- Decision ledger: PASS, 48 entries.
- Task ledger: `DATA_MISSING` for both committed and live locations; fallback
  task-card/report/worktree/branch search found no duplicate scope.
- Documentation impact: `DOCS_UPDATED` in `AGENTS.md` and
  `docs/development/bounded_offline_workloads.md`.
