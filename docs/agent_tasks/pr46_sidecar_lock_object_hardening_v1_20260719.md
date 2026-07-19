---
job_id: pr46_sidecar_lock_object_hardening_v1_20260719
lane: Evaluation
supporting_lanes:
  - Repo Hygiene
  - Reporting
owner: Codex
allowed_files:
  - docs/agent_tasks/pr46_sidecar_lock_object_hardening_v1_20260719.md
  - src/predictor/market_form_residual.py
  - tests/test_market_form_residual.py
  - reports/agent_jobs/pr46_sidecar_lock_object_hardening_v1_20260719/README.md
  - reports/agent_jobs/pr46_sidecar_lock_object_hardening_v1_20260719/STATE.md
  - reports/agent_jobs/pr46_sidecar_lock_object_hardening_v1_20260719/DECISIONS.md
  - reports/agent_jobs/pr46_sidecar_lock_object_hardening_v1_20260719/VALIDATION.md
  - reports/agent_jobs/pr46_sidecar_lock_object_hardening_v1_20260719/CODE_REVIEW.json
  - reports/agent_jobs/pr46_sidecar_lock_object_hardening_v1_20260719/RUN_OUTCOME.json
  - reports/agent_jobs/pr46_sidecar_lock_object_hardening_v1_20260719/DECISION_ENTRY.json
  - reports/agent_jobs/pr46_sidecar_lock_object_hardening_v1_20260719/status.json
  - reports/agent_jobs/pr46_sidecar_lock_object_hardening_v1_20260719/validation.json
  - reports/agent_jobs/pr46_sidecar_lock_object_hardening_v1_20260719/guard-preflight.json
  - reports/agent_jobs/pr46_sidecar_lock_object_hardening_v1_20260719/guard-final.json
  - reports/agent_jobs/pr46_sidecar_lock_object_hardening_v1_20260719/red-repro.json
  - reports/agent_jobs/pr46_sidecar_lock_object_hardening_v1_20260719/green-evidence.json
  - reports/agent_jobs/pr46_sidecar_lock_object_hardening_v1_20260719/diff-check.json
  - reports/agent_jobs/pr46_sidecar_lock_object_hardening_v1_20260719/github-checks.json
  - reports/agent_jobs/pr46_sidecar_lock_object_hardening_v1_20260719/final-refs.json
  - reports/agent_jobs/pr46_sidecar_lock_object_hardening_v1_20260719/WAIT_RESULT.json
  - reports/agent_jobs/pr46_sidecar_lock_object_hardening_v1_20260719/commands.log
approval_required: true
approval_source: The owner's 2026-07-19 explicit confirmation authorizes one bounded PR 46 descendant commit and normal push from exact head 649518236d335326b726b8d3cd81bd12660dae0f containing only sidecar-lock object hardening, regressions, and required V2 task/report metadata; PR 46 must remain draft and unmerged and PR 53 must remain untouched.
allow_unapproved_safe_extension: false
timeout_seconds: 14400
output_dir: reports/agent_jobs/pr46_sidecar_lock_object_hardening_v1_20260719
mutation_mode: safe_extension
production_data_access: false
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: pr46_sidecar_lock_object_hardening_v1_20260719
proof_question: Can exact PR 46 head 649518236d335326b726b8d3cd81bd12660dae0f receive one bounded descendant that rejects symlink and non-regular persistent sidecar lock objects before transaction entry, preserves ordinary regular-lock serialization and every existing descriptor/history/publication invariant, and remains safe for a new independent whole-PR review?
hypothesis_id: pr46_persistent_sidecar_lock_object_admission_and_single_inode_serialization_v1
program_track: offline_development
entry_state: exact_pr46_64951823_draft_unmerged_blocked_on_symlinkable_sidecar_lock_lost_acknowledged_append
target_transition: exact_pr46_one_descendant_rejects_substituted_lock_objects_ready_for_fresh_whole_pr_review
exit_predicate: At most one normal commit whose sole parent is 649518236d335326b726b8d3cd81bd12660dae0f is pushed to the existing PR 46 branch only if a public append fixture is red before the fix and green after it, symlink and non-regular lock objects fail closed without target or staged-file mutation, ordinary same/distinct record concurrency still uses one persistent lock inode, the full scorer/writer and resource/collector-lock suites plus Ruff, compile, hardening, deterministic, artifact, diff, V2 and GitHub gates pass, PR 46 remains draft and unmerged, and PR 53 remains exactly untouched at 2deb5aec454fb9314a22f30a30169aa05b2261c5.
source_class: owner_authorized_exact_head_repair_plus_fresh_pr46_64951823_symlink_lock_split_reproduction
dataset_version: greyhound_pr46_master_c1dfd464_head_64951823_pr53_2deb5aec_sidecar_lock_object_hardening_v1_20260719
evidence_hash: sha256:7b093ccc89794c2fc1933653e400a906be52ff9e46570b8d952b3fdccc88624a
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: Live master remains c1dfd464cf6ecfb2034f96ac1a8d3ea58d4e6afa, PR 46 remains open draft unmerged at 649518236d335326b726b8d3cd81bd12660dae0f, PR 53 remains open draft unmerged at 2deb5aec454fb9314a22f30a30169aa05b2261c5, the current worktree is clean apart from this task's allowlisted files, and no overlapping active claim exists.
docs_impact: DOCS_UPDATED
docs_checked:
  - src/predictor/market_form_residual.py
  - tests/test_market_form_residual.py
  - docs/agent_tasks/pr46_atomic_shadow_append_repair_20260719.md
  - docs/agent_tasks/pr46_unambiguous_staged_fd_ownership_repair_v2_20260719.md
docs_changed:
  - docs/agent_tasks/pr46_sidecar_lock_object_hardening_v1_20260719.md
  - src/predictor/market_form_residual.py
docs_followup: NONE
reason: The persistent-lock admission behavior and fail-closed filesystem-object boundary change; the source API documentation and this exact task contract document the change without widening runtime or operator scope.
---

# PR 46 persistent sidecar lock-object hardening V1

Use a test-first vertical slice. First add one public `append_shadow_record`
regression that deterministically reproduces the reviewed symlink split and
fails at exact head `64951823`. Then make the smallest lock-open change that
rejects a final-component symlink and validates the opened descriptor as a
regular file. Add the next non-regular-object case only after the first slice
is green. Preserve the public append return values, history format, frozen
artifacts, staged-FD lifecycle, publication point, and durability semantics.

Task routing: `large`, high reasoning, no worker delegation because lock-object
admission, deterministic concurrency reproduction, and exact-head publication
form one coupled correctness lane. No refit, database or outcome access,
runtime/service/timer mutation, deployment, activation, prediction, betting,
merge, ready-state transition, PR creation, force-push, rebase, or PR 53 change
is authorized.
