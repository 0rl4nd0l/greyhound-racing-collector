---
job_id: MANUAL_REQUEST_CAPTURE_ONLY_NO_SHADOW_V1_20260730
title: Remove accidental shadow-model dependency from capture-only manual requests
lane: Query Orchestration
supporting_lanes:
  - Runtime Safety
  - Provenance
  - Repo Hygiene
owner: Codex
approval_required: true
approval_source: "Owner /goal request on 2026-07-30 authorizes bounded repository implementation, validation, review, commit, push, focused PR creation, exact-head CI wait, and merge under existing repository authority; deployment and live execution are not authorized."
allow_unapproved_safe_extension: false
timeout_seconds: 28800
mutation_mode: safe_extension
base: origin/master
production_data_access: false
production_data_boundary: "No live attempt, deployment, service/timer action, canonical DB/history mutation, result access, training, model change, promotion, EV, staking, or betting."
github_mutation_allowed: true
git_history_mutation_allowed: true
live_service_mutation_allowed: false
closeout_scope: repo_only
docs_impact: DOCS_NOT_REQUIRED
task_tier: medium
recommended_model: high_reasoning
actual_model: gpt-5
worker_model_allowed: false
worker_decision_limit: "No delegated implementation; the receipt dependency decision, exact diff, and merge gate remain with the primary agent."
escalation_needed: false
output_dir: reports/agent_jobs/MANUAL_REQUEST_CAPTURE_ONLY_NO_SHADOW_V1_20260730
allowed_files:
  - docs/agent_tasks/manual_request_capture_only_no_shadow_v1_20260730.md
  - scripts/shadow_autopilot_v1.py
  - tests/test_shadow_autopilot_v1.py
  - reports/agent_jobs/MANUAL_REQUEST_CAPTURE_ONLY_NO_SHADOW_V1_20260730/README.md
  - reports/agent_jobs/MANUAL_REQUEST_CAPTURE_ONLY_NO_SHADOW_V1_20260730/STATE.md
  - reports/agent_jobs/MANUAL_REQUEST_CAPTURE_ONLY_NO_SHADOW_V1_20260730/DECISIONS.md
  - reports/agent_jobs/MANUAL_REQUEST_CAPTURE_ONLY_NO_SHADOW_V1_20260730/VALIDATION.md
  - reports/agent_jobs/MANUAL_REQUEST_CAPTURE_ONLY_NO_SHADOW_V1_20260730/REVIEW.md
  - reports/agent_jobs/MANUAL_REQUEST_CAPTURE_ONLY_NO_SHADOW_V1_20260730/CODE_REVIEW.json
  - reports/agent_jobs/MANUAL_REQUEST_CAPTURE_ONLY_NO_SHADOW_V1_20260730/COMMANDS.md
  - reports/agent_jobs/MANUAL_REQUEST_CAPTURE_ONLY_NO_SHADOW_V1_20260730/guard-preflight.json
  - reports/agent_jobs/MANUAL_REQUEST_CAPTURE_ONLY_NO_SHADOW_V1_20260730/status.json
  - reports/agent_jobs/MANUAL_REQUEST_CAPTURE_ONLY_NO_SHADOW_V1_20260730/RUN_OUTCOME.json
  - reports/agent_jobs/MANUAL_REQUEST_CAPTURE_ONLY_NO_SHADOW_V1_20260730/DECISION_ENTRY.json
  - reports/agent_jobs/MANUAL_REQUEST_CAPTURE_ONLY_NO_SHADOW_V1_20260730/pr-body.md
  - reports/agent_jobs/MANUAL_REQUEST_CAPTURE_ONLY_NO_SHADOW_V1_20260730/WAIT_RESULT.json
  - reports/agent_jobs/MANUAL_REQUEST_CAPTURE_ONLY_NO_SHADOW_V1_20260730/WAIT_RESULT.json.log
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: MANUAL_REQUEST_CAPTURE_ONLY_NO_SHADOW_V1_20260730
proof_question: Can a scheduled odds-only collector running with --skip-shadow-run and no shadow model service one claimed manual request through exact-race capture and a sealed terminal response without changing capture authority, while genuine shadow runs still require a model?
hypothesis_id: manual_request_capture_only_no_shadow_v1_exact_master
program_track: offline_development
entry_state: owner_authorized_clean_exact_origin_master_53f372ed_tree_80769c1b_after_pr76_pr77_and_warragul_r4_capture_proof
target_transition: one_narrow_pr_removes_only_the_accidental_manual_request_to_shadow_scoring_dependency_and_adds_capture_only_and_full_shadow_regressions
exit_predicate: Exact origin/master and tree are verified; receipt construction is proven independent of shadow scoring; the diff remains inside allowed_files; a focused red-green regression proves one claimed capture-only request can finalize without a shadow model while a genuine shadow run still fails without one; classifier-selected tests, fatal Ruff, py_compile, git diff check, exact-diff review, and exact-head CI pass; the focused PR is merged without deployment or live execution.
source_class: exact_origin_master_53f372ed_tree_80769c1b_plus_pr76_pr77_lineage_and_warragul_r4_live_proof_bundle
dataset_version: manual_request_capture_only_no_shadow_v1_repository_synthetic_validation_20260730
evidence_hash: sha256:e759aff96aa424a10fa3b1838ef6246df6f12400050cf436ea1e169a66e3f5eb
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: Resume only while origin/master remains 53f372ed27a4eb19601e1719495723881f0bf983 with tree 80769c1bb18d9a3465fe9b9c93de6584e682cd1c; the clean task worktree remains owned by this job; and no active claim or open PR overlaps this exact target transition.
---

# Capture-only manual request shadow dependency repair

Remove only the accidental requirement that a claimed manual collector request
forces shadow scoring in the scheduled odds-only path. Preserve request
priority, exact-race capture, sealed receipt publication, one-attempt semantics,
capture authority, and the normal missing-model failure for genuine shadow
runs.

## Hard stops

- Stop if the sealed receipt consumes shadow scores or model output.
- No arbitrary model wiring or protocol redesign.
- No files outside `allowed_files`.
- No deployment, runtime action, live attempt, data mutation, result access,
  training, promotion, EV, staking, or betting.
