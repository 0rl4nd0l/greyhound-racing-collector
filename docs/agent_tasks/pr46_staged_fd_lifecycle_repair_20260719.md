---
job_id: pr46_staged_fd_lifecycle_repair_20260719
lane: Evaluation
supporting_lanes:
  - Repo Hygiene
  - Reporting
owner: Codex
allowed_files:
  - docs/agent_tasks/pr46_staged_fd_lifecycle_repair_20260719.md
  - src/predictor/market_form_residual.py
  - tests/test_market_form_residual.py
  - reports/agent_jobs/pr46_staged_fd_lifecycle_repair_20260719/README.md
  - reports/agent_jobs/pr46_staged_fd_lifecycle_repair_20260719/STATE.md
  - reports/agent_jobs/pr46_staged_fd_lifecycle_repair_20260719/DECISIONS.md
  - reports/agent_jobs/pr46_staged_fd_lifecycle_repair_20260719/VALIDATION.md
  - reports/agent_jobs/pr46_staged_fd_lifecycle_repair_20260719/CODE_REVIEW.json
  - reports/agent_jobs/pr46_staged_fd_lifecycle_repair_20260719/RUN_OUTCOME.json
  - reports/agent_jobs/pr46_staged_fd_lifecycle_repair_20260719/DECISION_ENTRY.json
  - reports/agent_jobs/pr46_staged_fd_lifecycle_repair_20260719/status.json
  - reports/agent_jobs/pr46_staged_fd_lifecycle_repair_20260719/validation.json
  - reports/agent_jobs/pr46_staged_fd_lifecycle_repair_20260719/guard-preflight.json
  - reports/agent_jobs/pr46_staged_fd_lifecycle_repair_20260719/descriptor-evidence.json
  - reports/agent_jobs/pr46_staged_fd_lifecycle_repair_20260719/WAIT_RESULT.json
approval_required: true
approval_source: The owner's 2026-07-19 request explicitly authorizes at most one narrow descendant commit and normal push to existing draft PR 46 from exact head 624dde3067edda1bd045573e8bec5c9d749c6836 to repair the independently confirmed staged-descriptor lifecycle defect.
allow_unapproved_safe_extension: false
timeout_seconds: 14400
output_dir: reports/agent_jobs/pr46_staged_fd_lifecycle_repair_20260719
mutation_mode: safe_extension
production_data_access: false
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: pr46_staged_fd_lifecycle_repair_20260719
proof_question: Can append_shadow_record retain explicit raw staged-descriptor ownership until managed-file transfer, close it exactly once on fchmod or fdopen failure, and preserve every existing atomic-publication, retry, history, permission, cleanup, deterministic, and artifact contract?
hypothesis_id: pr46_staged_fd_exception_safe_ownership_v1
program_track: offline_development
entry_state: exact_pr46_624dde30_draft_unmerged_blocked_on_fchmod_and_fdopen_staged_descriptor_leaks
target_transition: exact_pr46_one_descendant_exception_safe_staged_descriptor_lifecycle_ready_for_fresh_independent_review
exit_predicate: At most one normal commit whose sole parent is 624dde3067edda1bd045573e8bec5c9d749c6836 is pushed to existing PR 46 only if deterministic Linux descriptor probes prove no leak, no double-close, no reused-descriptor closure, exact precommit target preservation and retry semantics; the expanded focused suite, 133 resource and lock tests, Ruff, compile, artifact hashes, deterministic regeneration, exact diff review, and GitHub CI all pass; and PR 46 remains open, draft, and unmerged.
source_class: owner_exact_head_repair_goal_plus_independent_fchmod_fdopen_failure_evidence
dataset_version: greyhound_pr46_master_c1dfd464_head_624dde30_staged_fd_lifecycle_20260719
evidence_hash: sha256:596ac84f853eeb6c1b5940ba581613777fab2b108a1aea4aedbc95e7ae2e2ff8
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: PR 46 remains open, draft, and unmerged at exact head 624dde3067edda1bd045573e8bec5c9d749c6836, master remains c1dfd464cf6ecfb2034f96ac1a8d3ea58d4e6afa, and no active claim is implementing the same descriptor-lifecycle repair.
docs_impact: DOCS_UPDATED
docs_checked:
  - AGENTS.md
  - docs/agent_tasks/pr46_atomic_shadow_append_repair_20260719.md
  - src/predictor/market_form_residual.py
  - tests/test_market_form_residual.py
docs_changed:
  - docs/agent_tasks/pr46_staged_fd_lifecycle_repair_20260719.md
docs_followup: NONE
reason: The new task card records the exception-safe ownership invariant; the public result and atomic-publication semantics remain unchanged, so no operator documentation changes are required.
---

# PR 46 staged descriptor lifecycle repair

Continue only existing draft PR 46 from exact head
`624dde3067edda1bd045573e8bec5c9d749c6836`. Publish at most one normal
descendant commit; do not rebase, force-push, create another PR, mark ready,
merge, deploy, adapt runtime callers, or migrate data.

Own the raw same-directory staged descriptor explicitly from `os.open` until
successful transfer to a managed file object. Close it exactly once on every
failure before transfer, and after transfer allow only the managed object to
close it. Add deterministic Linux regressions for `fchmod` and `fdopen`
failure, repeated-failure descriptor growth, cleanup failure with a retained
pathname but closed descriptor, successful and existing failure paths, and
descriptor-number reuse without double-close or unrelated-descriptor closure.

Preserve the sidecar lock, full-history validation, staged flush/fsync, atomic
replace commit point, directory-sync classification, `APPENDED`,
`EXACT_REPLAY`, `COMMIT_STATE_UNKNOWN`, permissions, cleanup recovery,
deterministic serialization, fixed outputs, and frozen model/manifest hashes.
Use only temporary synthetic JSONL fixtures. Do not access production SQLite,
live history, outcomes, fitting, runtime services, timers, PR 47, or PR 48.

Task routing: `large`, high-reasoning primary model, no worker delegation
because the ownership algorithm, failure injection, ancestry, and release are
one tightly coupled correctness lane.
