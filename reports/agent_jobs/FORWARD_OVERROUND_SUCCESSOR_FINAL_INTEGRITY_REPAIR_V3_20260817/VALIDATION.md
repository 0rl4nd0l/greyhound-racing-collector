# Validation

## Passing

- Exact blocked-head probes reproduced all four reported defects before repair.
- Exact runtime-interpreter stdlib suite: 27 tests passed in 207.836 seconds.
  It includes seven finalization restart boundaries, two new N=1000
  crash/conflict cases, temporal admission, immutable membership, finish-order,
  admission drift, rejection replay, and deterministic terminal regressions.
- State-machine adjacent suite: 20 tests passed through a no-install in-memory
  shim for `pytest.raises`.
- Exact runtime-interpreter `py_compile` passed for runtime, state machine,
  finalizer, and both focused test files.
- Disposable synthetic two-phase N=1000 proof sealed 1,000 predictions with
  zero results, then admitted exactly 1,000 post-jump results and recorded one
  finalization request and one paired-score commit.
- Crash after metric calculation followed by sealed-receipt corruption
  terminalized with no `METRICS.json`; post-score metrics hash drift before
  consumption also terminalized with no metrics and remained restart-stable.
- Frozen protocol/model/preprocessing/development-protocol/scorer hashes are
  exact and unchanged.
- `systemd-analyze verify` exited zero. It emitted only unrelated host netplan
  permission and snapd `RestartMode` warnings.
- Installed successor unit state is not-found/inactive/dead; cohort and
  activation paths are absent.

## Environment limitations

- Neither the exact runtime interpreter nor system Python contains pytest.
  Nothing was installed; the supported unittest entry point and no-install
  state-machine harness were used.
- Ruff and Black are unavailable and were not installed. Compile checks,
  focused suites, checksum verification, semantic controls, and
  `git diff --check` are the supported checks.
- Exact PR-head CI is verified after publication because GitHub cannot test an
  unpushed commit.
