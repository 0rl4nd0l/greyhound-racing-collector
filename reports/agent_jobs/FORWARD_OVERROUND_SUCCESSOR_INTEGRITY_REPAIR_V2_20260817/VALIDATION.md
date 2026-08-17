# Validation

## Passing

- Five targeted tests failed on exact parent source before repair: post-jump
  prediction admission, pre-staged future result admission, rejection replay
  conflict, malformed candidate exception/silent exclusion, and out-of-order
  fatal state without final/consumed receipts.
- Exact runtime-interpreter stdlib suite: 22 tests passed in 137.830 seconds.
  It includes six N=1000 finalization fault boundaries and all requested
  temporal, replay, malformed, membership, result contamination, finish-order,
  activation drift, and terminal restart regressions.
- State-machine adjacent suite: 17 tests passed through an in-memory shim for
  its single `pytest.raises` use; no dependency was installed.
- Exact runtime-interpreter `py_compile`: runtime, state machine, finalizer, and
  both focused test files passed.
- Prospective two-phase synthetic run: 1,000 prediction receipts sealed at
  23:45 for a 00:00 jump; no result existed in that phase; 1,000 results with
  00:05 source capture were observed at 00:06; one finalization request and one
  paired-score commit were recorded.
- Frozen protocol/model/preprocessing/development-protocol/scorer hashes are
  exact and unchanged.
- `systemd-analyze verify` exited successfully for the successor service/timer.
  Only unrelated host netplan permission and snapd `RestartMode` warnings were
  emitted.
- Installed successor unit state is not-found/inactive/dead; both successor
  cohort paths and activation receipts are absent.
- Task-card validation, live decision-ledger validation, semantic scope
  admission, JSON parsing, checksum verification, and `git diff --check` pass.

## Unavailable or separate

- The exact runtime interpreter has no pytest module. The supported unittest
  suite and no-install state-machine harness were used instead.
- Ruff and Black are unavailable in the existing environment; neither was
  installed. `py_compile`, focused suites, JSON parsing, checksum verification,
  and whitespace validation provide the supported local checks.
- Exact candidate-head GitHub CI and review-thread state are publication-time
  external evidence and are rechecked after the containing commit is pushed;
  they are not self-asserted inside this commit.
