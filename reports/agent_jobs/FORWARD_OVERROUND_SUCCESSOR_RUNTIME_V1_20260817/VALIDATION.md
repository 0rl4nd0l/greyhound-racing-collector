# Validation

## Passing

- Frozen successor protocol SHA-256: exact.
- Frozen model, preprocessing, development protocol, scorer contract, and
  25-entry source artifact manifest: exact.
- V2 block/final-report/consumed hashes and 9/6/null-metrics terminal state:
  exact.
- Exact runtime-interpreter `py_compile`: state machine, runtime, finalizer,
  and runtime tests passed.
- Nine standard-library runtime tests passed in 5.789 seconds in the exact
  runtime interpreter. They cover absent activation, future activation,
  pre-seal timing rejection, admission pause/reviewed resume, fatal finalizer
  drift, immutable packet conflict, tampered sealed receipt, and the exact
  1,000-race path.
- Fifteen prepared state-machine tests passed in the exact runtime interpreter
  through an in-memory compatibility shim for their single `pytest.raises`
  use. The test file and environment were not changed.
- Synthetic exact-N proof: 1,000 predictions, 1,000 results, one finalization
  request, one paired-score commit, 20,000 race bootstraps, 20,000 date-cluster
  bootstraps, and five 200-race blocks.
- JSON parsing: successor protocol and all readiness JSON artifacts passed.
- Systemd unit static verification exited successfully; calendar normalization
  passed as `*-*-* *:00/2:00`.
- SHA256SUMS and repository whitespace/diff validation passed.
- Code review: no open critical, warning, or suggestion findings.
- Semantic V2 task-card, diff, artifact, runtime-closeout, decision-candidate,
  live-ledger, and release checks passed. The decision was appended exactly
  once, the active claim was removed, and authoritative status is `released`.

## Unavailable or pre-existing

- The exact runtime interpreter has no `pytest` module; direct pytest commands
  fail before collection with `No module named pytest`. No dependency was
  installed.
- The exact runtime interpreter has no Ruff or Black module; both checks are
  unavailable. No formatter or linter was installed.
- The architecture-check skill's five expected `.cursor/rules/` files are not
  present in this repository. The implementation touches none of the skill's
  embedding, vector-store, RAG, UUID, metric, dimension, or SQLite invariants.
- `systemd-analyze verify` emitted unrelated host warnings for a permission-
  denied netplan unit and an unsupported `RestartMode` key in snapd; the
  successor unit/timer themselves produced no diagnostic and the command
  exited successfully.
- Login-shell startup reports missing `/tmp/cargo-codex-v2/env`; it is
  pre-existing and unrelated.
