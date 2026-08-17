# Validation

## Passing

- Exact old-head probes at `4481d2cb5c07da0b6091c94085b78c1599acfef4`
  reproduced stale-time admission, exposed partial final publication, corrupt
  commitment retention, PR 137 cohort overlap, and repeated scorer execution.
- Exact runtime-interpreter stdlib suite: 35 tests passed in 236.696 seconds.
- Stable six-boundary N=1000 crash matrix: passed in 129.738 seconds.
- Disposable two-phase N=1000 proof: passed in 20.810 seconds. It sealed
  exactly 1,000 predictions with zero results, admitted exactly 1,000 results
  after jump, and produced one durable score commitment.
- State-machine adjacent suite: 21 tests passed with a no-install in-memory
  shim for `pytest.raises`.
- Adversarial validation covers partial temporary writes; corrupt, incomplete,
  cross-hash-consistent but schema-invalid, journal-inconsistent, and
  verdict-inconsistent terminal evidence; fatal conflicts after metrics; and
  valid commits beside stray corrupt sentinels.
- `py_compile`, protocol JSON parsing, PR 137 checksum verification, semantic
  file checksums, and `git diff --check` passed.
- `systemd-analyze verify` exited zero. It emitted only unrelated host netplan
  permission and snapd `RestartMode` warnings.
- The successor service and timer are not-found/inactive/dead. Their installed
  unit files, the October successor cohort root, and ACTIVATION are absent.
- Semantic Run Control V2 was not activated or mutated; no task card, claim,
  ledger, decision candidate, release receipt, or `.tenn` state was created by
  this repair.

## Environment limitations

- Neither the exact runtime interpreter nor system Python contains pytest.
  Nothing was installed; the supported unittest entry point and no-install
  state-machine harness were used.
- GitHub CI can validate the exact PR head only after publication. Its final
  state is recorded in the PR body and handoff, not retroactively in this
  semantic-freeze-bound evidence file.
