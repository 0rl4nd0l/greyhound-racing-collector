---
job_id: pr46_unambiguous_staged_fd_ownership_repair_v2_20260719
lane: Evaluation
supporting_lanes:
  - Repo Hygiene
  - Reporting
owner: Codex
allowed_files:
  - docs/agent_tasks/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719.md
  - src/predictor/market_form_residual.py
  - tests/test_market_form_residual.py
  - reports/agent_jobs/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719/README.md
  - reports/agent_jobs/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719/STATE.md
  - reports/agent_jobs/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719/DECISIONS.md
  - reports/agent_jobs/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719/VALIDATION.md
  - reports/agent_jobs/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719/CODE_REVIEW.json
  - reports/agent_jobs/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719/INDEPENDENT_REVIEW.json
  - reports/agent_jobs/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719/RUN_OUTCOME.json
  - reports/agent_jobs/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719/DECISION_ENTRY.json
  - reports/agent_jobs/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719/status.json
  - reports/agent_jobs/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719/validation.json
  - reports/agent_jobs/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719/guard-preflight.json
  - reports/agent_jobs/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719/red-repro.json
  - reports/agent_jobs/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719/green-evidence.json
  - reports/agent_jobs/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719/deterministic-evidence.json
  - reports/agent_jobs/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719/frozen-hashes.json
  - reports/agent_jobs/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719/diff-check.json
  - reports/agent_jobs/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719/github-checks.json
  - reports/agent_jobs/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719/final-refs.json
  - reports/agent_jobs/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719/commands.log
  - reports/agent_jobs/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719/WAIT_RESULT.json
approval_required: true
approval_source: The owner's 2026-07-19 PR46_UNAMBIGUOUS_STAGED_FD_OWNERSHIP_REPAIR_V2 goal explicitly authorizes one bounded normal descendant commit and fast-forward push to existing draft PR 46 from exact head c75f6746394be9c18f479030e4d2e4dd2156956d, while forbidding merge and any PR 53 modification.
allow_unapproved_safe_extension: false
timeout_seconds: 14400
output_dir: reports/agent_jobs/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719
mutation_mode: safe_extension
production_data_access: false
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: pr46_unambiguous_staged_fd_ownership_repair_v2_20260719
proof_question: Can append_shadow_record retain unambiguous raw ownership through every staged-wrapper failure, close each staged descriptor at most once, preserve the primary failure when cleanup also fails, and retain all existing transaction and deterministic contracts?
hypothesis_id: pr46_unambiguous_staged_fd_ownership_and_primary_precedence_v2
program_track: offline_development
entry_state: exact_pr46_c75f6746_draft_unmerged_blocked_on_ambiguous_closefd_true_constructor_failure_and_exception_precedence
target_transition: exact_pr46_one_descendant_unambiguous_staged_fd_ownership_ready_for_fresh_exact_head_review
exit_predicate: Exactly one normal commit whose sole parent is c75f6746394be9c18f479030e4d2e4dd2156956d is fast-forward pushed to existing PR 46 only if consume-then-raise, immediate descriptor reuse, EBADF masking, fchmod-plus-close, wrapper, write, flush, fsync, publication, cleanup, target-state, retry, descriptor-growth, deterministic-output, frozen-hash, diff, V2, independent-review, and GitHub gates pass; PR 46 remains open draft unmerged; and PR 53 remains exactly untouched.
source_class: owner_exact_head_repair_goal_plus_live_refs_plus_fresh_independent_partial_fdopen_and_close_precedence_reproduction
dataset_version: greyhound_pr46_master_c1dfd464_head_c75f6746_parent_624dde30_pr53_2deb5aec_unambiguous_staged_fd_ownership_v2_20260719
evidence_hash: sha256:29c676ee65bd3db0bd4927b3268b12ce8fae2f0b7749ecbbff6be4606aaf56ce
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: PR 46 remains open draft unmerged at exact head c75f6746394be9c18f479030e4d2e4dd2156956d, master remains c1dfd464cf6ecfb2034f96ac1a8d3ea58d4e6afa, PR 53 remains exactly 2deb5aec454fb9314a22f30a30169aa05b2261c5, and no active claim overlaps this repair.
docs_impact: DOCS_UPDATED
docs_checked:
  - AGENTS.md
  - docs/semantic_anti_loop_control_v2.md
  - docs/agent_tasks/pr46_staged_fd_lifecycle_repair_20260719.md
  - docs/agent_tasks/pr46_fresh_exact_head_acceptance_v1_20260719.md
  - src/predictor/market_form_residual.py
  - tests/test_market_form_residual.py
docs_changed:
  - docs/agent_tasks/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719.md
docs_followup: NONE
reason: The task card records the new constructor-failure ownership and primary-exception invariants; public writer results and operator/runtime behavior remain unchanged.
---

# PR 46 unambiguous staged FD ownership repair V2

Continue only existing draft PR 46 from exact head
`c75f6746394be9c18f479030e4d2e4dd2156956d`. Publish exactly one normal
descendant commit only after every gate passes. Do not rebase, force-push,
create another PR, mark ready, merge, deploy, activate, refresh PR 53, adapt
runtime callers, or migrate data.

Regression adjudication is `NEW_FAILURE_CLASS` with `TEST_GAP`: the prior
repair fixed ordinary pre-transfer leaks, but a `closefd=True` constructor can
consume the descriptor before raising and make outer cleanup ownership
unknowable. Use a non-owning wrapper or direct raw-descriptor I/O so the writer
retains explicit raw ownership until its own exactly-once close. Preserve the
primary failure if close or staged-path cleanup also fails; if close is the
only failure, preserve `shadow_output_write_failed`.

Add deterministic Linux regressions for constructor failure before and after
consumption, immediate descriptor-number reuse, EBADF and combined close
failure precedence, exactly-once/no-retry close behavior, repeated attempts,
wrapper/write/flush/fsync/publication faults, target absent/present,
permissions, retained/deleted stage cleanup, exact bytes/rows, and `APPENDED`
then `EXACT_REPLAY`. Preserve sidecar locking, full-history validation,
canonical serialization, staged flush/fsync, atomic replace, directory-fsync
classification, retry, `COMMIT_STATE_UNKNOWN`, fixed predictions, and frozen
hashes.

Use only temporary synthetic JSONL and fixed frozen fixtures. Do not access
production SQLite, live history, outcomes, fitting, services, timers, runtime,
deployment, activation, prediction generation, or betting actions.

Task routing: `large`, high-reasoning primary model. Read-only workers may
scout live refs, overlap, and independently review the final diff; all scope,
implementation, release, and publication decisions remain with the primary
orchestrator.
