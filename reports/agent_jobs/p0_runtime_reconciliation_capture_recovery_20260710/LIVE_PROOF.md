# Paired-Market Live Proof

result: WORKING

## Reviewed Runtime

- PR: `#41`
- Commit: `314091604fc245185638cbf30be07ed7241301d9`
- Runtime: `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-runtime-p0-20260710`
- DB used for all append-capable proof:
  `/tmp/greyhound-pr41-manual-gate-31409160/greyhound_racing_data.db`

## Manual Odds-Only Gate

Artifact:

`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525/shadow_autopilot_daemonization_v1_pr41_manual_gate_31409160/odds_capture_only_daemon_report.json`

- `final_status=ODDS_CAPTURE_ONLY_READY`
- `ready_count=8`
- `validation_pass_count=8`
- `blocked_attempt_count=0`
- `inserted_live_odds_rows=118` on the isolated copy only
- lock released by owner

## Consecutive Full-Daemon Cycles

Cycle 1 artifact:

`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525/shadow_autopilot_daemonization_v1_pr41_full_cycle1_31409160/daemon_run_report.json`

- exit 0
- `odds_coverage_status=SUCCESS`
- `autonomous_live_odds_capture_ready_count=13`
- `autonomous_live_odds_capture_inserted_rows=142` on the isolated copy only
- `protected_paths_unchanged=true`
- final state `DAEMON_READY_NEEDS_DEPLOYMENT` because timers remained disabled
- lock released

Cycle 2 artifact:

`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525/shadow_autopilot_daemonization_v1_pr41_full_cycle2_31409160/daemon_run_report.json`

- exit 0
- `odds_coverage_status=SUCCESS`
- `autonomous_live_odds_capture_ready_count=10`
- `autonomous_live_odds_capture_inserted_rows=84` on the isolated copy only
- `protected_paths_unchanged=true`
- final state `DAEMON_READY_NEEDS_DEPLOYMENT` because timers remained disabled
- lock released

Transient expected-vs-source runner identity mismatches were blocked with zero
append. They were not missing-market failures and this lane did not weaken or
modify the exact runner gate.

## Production Boundary

The production hashes remained unchanged before and after all three runs:

- main: `470e97b83b02bc8070277945c062052572ce209a58d1d5bacb0f24076cedd61b`
- writable: `61b9ee76a52068435ef3c96528bbdbd9d4498180f6b055ab0e828a7f3559436e`
- stage: `7af475c57e63f2ad69cac2c2281c8a59d06bc073e1ef5e722729dc9f1cfbe6f1`

Both timers stayed disabled/inactive. The full and odds-only services stayed
inactive; the odds-only unit retains its earlier failed result but was not
started during this proof.
