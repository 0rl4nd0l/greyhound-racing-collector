# Validation

result: WORKING

## Focused automated checks

- `python3 -m unittest tests.test_run_bounded_offline`
  - PASS: 18 tests
  - Covers absolute/root containment, symlink resolution, Docker mount
    delimiters, bounded numeric arguments, empty search patterns, exclusion
    defaults and opt-ins, immutable image identity, single-worker construction,
    sequential hashes, read-only/low-priority/hard-limit construction, timeout
    cleanup, interruption cleanup, and hard-limit failure propagation.
- `python3 -m py_compile scripts/run_bounded_offline.py tests/test_run_bounded_offline.py`
  - PASS
- `git diff --check`
  - PASS

## Host enforcement checks

- systemd: `249`; cgroup v2 root contains `io`.
- User manager controller set: `memory pids`; user scopes cannot prove a hard
  I/O ceiling here.
- Docker: server `29.1.3`, cgroup version `2`, driver `systemd`.
- Local image only: immutable
  `sha256:cc9071bd161080c1a543f3023b7d0db905b497e6ae757fe078227803bc7e4dc8`;
  the wrapper uses `--pull never`.
- Mounted root source: `/dev/nvme0n1p1`; enforced parent device:
  `/dev/nvme0n1` (`259:2`).
- Mounted `rg` is statically linked, so no additional host library tree is
  exposed.
- Live cgroup read-back during the controlled scan:
  - `io.max`: `259:2 rbps=1048576 wbps=max riops=16 wiops=max`
  - `io.weight`: `default 1`
  - `cpu.weight`: `8`
  - one container PID; Docker reported one task

The child bootstrap exits 78 before `exec` when hard limits, weights, or idle
I/O priority do not match. The focused hard-limit test verifies that return 78
is propagated without an unconstrained fallback.

## Integration smokes

- Root `/mnt` and the common Greyhound worktree parent were both rejected with
  exit 2 as ancestors of the wrapper repository.
- Sequential hash of the exact task card returned
  `1ea67adbe94adacb8c13809d044d00b6edf6d7f31e7ecc212bead7f90e0469a0`.
- Test discovery under the exact `tests` root returned
  `/scan/test_run_bounded_offline.py` without importing tests.
- Representative report search returned the expected no-match exit 1, rather
  than timeout or hard-limit failure, and no matching task container remained.

## Independent review

The code-review pass found no critical issues. Its initial warnings covered
archive/large-data exclusions, empty search patterns, mutable image tags, mount
delimiter revalidation, and I/O-priority read-back. All were fixed before this
validation; the follow-up review/test pass reported no deferred items.

## Collection safety

During the representative scan, natural odds-capture services starting at
17:13:38 and 17:14:04 AEST both exited one second later with `Result=success`
and `ExecMainStatus=0`. The odds timer remained `active/waiting`. No completed
full cycle appeared before the bounded observation ended. No service action or
database access was performed.
