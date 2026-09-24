# Greyhound host workload isolation V1

result: WORKING

The repository now owns a fail-closed wrapper for broad read-only agent and
offline filesystem work. It admits one exact root into one transient child,
uses a single `rg` worker, excludes irrelevant/heavy paths by default, and
verifies cgroup-v2 `io.max` plus low CPU/I/O priority before starting the
requested operation.

The implementation is intentionally limited to `rg`, file listing, sequential
SHA-256 hashing, and test-file discovery. It does not expose an arbitrary shell
escape. Normal narrow source-local searches remain direct and the active
collector is not cgrouped, reconfigured, restarted, paused, or inspected through
its database.

See `VALIDATION.md` for focused verification and `MEASUREMENT.md` for the
controlled scan and passive odds-cycle evidence.

## Runtime-proof fields

- intended output: a bounded, read-only offline scan with hard physical-device
  read limits and unchanged live collection
- live output location: transient Docker cgroup under
  `/sys/fs/cgroup/system.slice/docker-*.scope`; repository evidence is in this
  report directory
- pre-run max timestamp or count: 0 task-owned transient containers before the
  representative run
- post-run max timestamp or count: 0 task-owned transient containers after the
  representative run
- rows/files inserted or updated after run start: 0 production rows and 0
  production files; only task allowlisted repository/report files changed
- readiness/gate status: focused tests, host support, hard-limit read-back,
  bounded measurement, collection safety, and cleanup gates passed
- exact command/query used: `python3 scripts/run_bounded_offline.py --root
  /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector/reports/agent_jobs
  --timeout-seconds 60 --read-mib-per-sec 1 --read-iops 16
  --include-generated rg __bounded_validation_no_match_20260718__ --glob '*.md'
  --glob '*.json'`
- result: WORKING
- remaining blocker: none for the target transition; convention bypass and
  page-cache reads remain residual risks described below

## Residual risk and next structural option

The wrapper is a repository convention, not an OS-wide admission controller;
a caller with Docker access can still bypass it. Physical-read limits do not
limit page-cache hits, and intentionally throttled work can raise global I/O PSI
while its single task waits even when interactive commands remain responsive.

If convention-level isolation proves insufficient, the next structural option
is an administrator-owned systemd system service/template or narrow broker that
delegates the `io` controller and admits only allowlisted exact-root workloads.
If physical contention persists after that, place cold archives on a separate
physical device in a separately authorised task; no archive movement occurred
here.
