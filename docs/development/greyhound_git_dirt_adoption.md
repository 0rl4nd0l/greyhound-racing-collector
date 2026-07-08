# Greyhound Git Dirt Adoption Policy

Last verified: 2026-07-08
Current source branch: `feature/dog-odds-snapshot-readiness`

This note prevents drift during Greyhound git cleanup sessions. The default
cleanup order is source adoption first, artifact cleanup second.

## Operating Order

1. Run Tenn git guard and record the current branch, HEAD, upstream, dirty
   files, registry status, and ledger status.
2. Inventory sibling worktrees read-only. Do not mutate sibling worktrees while
   deciding whether their code should be adopted.
3. Adopt loose source/test changes one narrow lane at a time into the current
   review branch.
4. Run focused validation for the touched files. If repo-wide pytest collection
   is blocked by environment-only dependencies, record the blocker and run an
   isolated focused test path.
5. Commit only the intended source/test/docs files. Verify the staged file list
   and `git diff --cached --check` before commit.
6. Write a report under `reports/agent_jobs/<job_id>/README.md`.
7. Only after a lane is adopted, superseded, or explicitly parked should cleanup
   consider worktree/artifact deletion or archival.

## Hard Boundaries

- Do not clean, delete, stash, reset, or mutate sibling worktrees during
  adopt-review.
- Do not mutate runtime services, systemd units, live daemon state, production
  DBs, model registry pointers, prediction snapshots, labels, odds capture,
  EV/betting outputs, or generated artifact evidence roots without explicit
  authorization for that exact action.
- Preserve unrelated current-checkout dirt as boundary evidence. Do not absorb
  `.gitignore` edits, daemon evidence artifacts, or generated report outputs
  into adoption commits unless the current task explicitly owns them.
- Keep strict-odds/provenance and live-runtime lanes parked when another agent
  owns those surfaces or when daemon-safe authorization is missing.

## Adoption Criteria

Adopt a loose code lane when all of these are true:

- The files are source/test/docs, not generated runtime evidence.
- The lane is report-only or otherwise preserves current repo behavior.
- The implementation is narrow enough to review in one commit.
- Focused tests can run, or an isolated equivalent can run when repo-wide test
  collection is blocked by unrelated dependencies.
- Output paths are guarded so report helpers cannot write into protected
  runtime/model/registry surfaces or existing input packet directories.

Park a lane when it is odds-heavy, runtime-heavy, service-facing, broad enough
to hide multiple behaviors, or interdependent with a currently active sibling
agent lane.

## Current Drift Lesson

During the July 8, 2026 remediation pass, the session initially drifted toward
cleanup/status framing while the real owner intent was to drain valuable loose
code from sibling worktrees. Future sessions should treat "clean up git" in
this repo as "review and adopt loose code improvements first" unless the owner
explicitly authorizes artifact/worktree deletion.
