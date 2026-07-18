# Bounded Offline Workloads

Use `scripts/run_bounded_offline.py` for read-heavy offline work such as broad
searches, bulk hashes, report archaeology, or test discovery. It constrains only
the transient workload; it does not move the whole Codex/app server or any live
Greyhound service into a constrained cgroup.

The wrapper accepts one absolute, existing `--root`. Choose one exact worktree
or a narrower subtree, and run sibling roots serially. Do not use shell globs
such as `greyhound-*` to create an archive-wide command.

## Safe examples

Search Markdown under one exact documentation root. Searches are fixed-string
and case-sensitive unless `--regex` or `--ignore-case` is requested:

```bash
python3 scripts/run_bounded_offline.py \
  --root "$PWD/docs" \
  rg 'capture_timestamp' --glob '*.md'
```

Search one exact report bundle. Generated output is excluded by default, so
place `--include-generated` before the subcommand when report archaeology is
intentional:

```bash
python3 scripts/run_bounded_offline.py \
  --root "$PWD/reports/agent_jobs/one_exact_job" \
  --include-generated \
  rg 'DATA_MISSING' --glob '*.md' --glob '*.json'
```

Hash Markdown files sequentially under one exact root:

```bash
python3 scripts/run_bounded_offline.py \
  --root "$PWD/docs/agent_tasks" \
  hash --glob '*.md'
```

List test modules without importing or executing them:

```bash
python3 scripts/run_bounded_offline.py \
  --root "$PWD/tests" \
  tests
```

A narrow interactive lookup in a known source directory does not require the
wrapper. For example, `rg -n 'def main' scripts/run_bounded_offline.py` is not a
broad filesystem scan.

## Constraints and exclusions

Defaults are one worker, a 300-second timeout, 8 MiB/s device reads, and 64 read
IOPS. Operator values are bounded to at most 3600 seconds, 16 MiB/s, and 128
read IOPS. The transient container also uses idle I/O scheduling, nice level 15,
low CPU and I/O cgroup weights, one CPU, a read-only root filesystem, no network,
and a read-only bind mount for the exact requested root.

The default file selection excludes:

- VCS metadata and worktree directories;
- archives and backups;
- caches, virtual environments, and dependency trees;
- reports, artifacts, logs, build products, and other generated output;
- data, datasets, model/prediction output, SQLite, CSV, Parquet, compressed
  archives, and other large-data formats.

An exclusion group may be included explicitly with `--include-vcs`,
`--include-worktrees`, `--include-archives`, `--include-caches`,
`--include-environments`, `--include-generated`, or `--include-large-data`.
Place inclusion flags before `rg`, `files`, `hash`, or `tests`. Inclusion changes
file selection only; it never relaxes root containment, worker count, timeout,
priority, or hard I/O limits.

## Fail-closed host requirements

This host implementation uses the local Docker daemon because its systemd
cgroup-v2 hierarchy can enforce `io.max`; the user systemd manager cannot do so
on this host. Docker must report cgroup v2 with the systemd driver, the local
`alpine:3.20` image must already exist, and `rg` must be statically linked. The
wrapper never pulls an image. It resolves the image to an immutable image ID and
discovers the block device backing the bounded root.

Inside the transient container, a bootstrap guard reads back and verifies the
expected device `rbps` and `riops`, low I/O and CPU weights, and idle I/O
priority before executing the requested operation. Missing or mismatched hard
limits exit 78 without running the scan. Invalid or unsafe configuration exits
2, timeout exits 124, and interruption exits 130 after stopping and removing the
transient container.

If the host no longer supports these checks, do not bypass the wrapper. Use a
narrow direct lookup only when it is genuinely interactive and source-local;
otherwise stop and repair or replace the hard-limit execution surface first.
