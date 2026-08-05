---
job_id: GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_V1_20260805
title: Implement GHU-051 isolated manual capture executor
lane: Query Orchestration
supporting_lanes:
  - Provenance
  - Reporting
owner: Codex
approval_required: true
approval_source: >-
  The owner's 2026-08-05 /goal explicitly authorizes the bounded repository
  implementation, validation, independent exact-diff review, dedicated commit
  and branch, focused pull request, exact-head CI, and merge only after the
  existing gates pass.
allow_unapproved_safe_extension: false
timeout_seconds: 43200
mutation_mode: safe_extension
base: e4f3699986237aad265b34e77d06d536f6046ee4
production_data_access: false
production_data_boundary: >-
  Fixture and OS-process tests only. No live source or browser launch, SQLite,
  canonical database/history/live_odds, autonomous lock/state/process,
  forward corpus, collector request/state, result evidence, service/timer,
  Phase 7, model training/promotion, EV, staking, betting, or deployment access.
github_mutation_allowed: true
git_history_mutation_allowed: true
live_service_mutation_allowed: false
closeout_scope: repo_and_publish
docs_impact: DOCS_REQUIRED
task_tier: large
recommended_model: high_reasoning
actual_model: gpt-5
why_this_model: >-
  Process-group cancellation, fail-closed artifact construction, path and lock
  isolation, and exact-head delivery require careful cross-cutting correctness.
worker_model_allowed: true
worker_decision_limit: >-
  One independent agent may perform the requested read-only exact-diff review;
  implementation, scope, GitHub, merge, and final decisions remain with the
  primary agent.
escalation_needed: false
output_dir: reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_V1_20260805
allowed_files:
  - docs/agent_tasks/ghu_051_isolated_manual_capture_executor_v1_20260805.md
  - docs/manual_independent_capture_v1.md
  - src/predictor/manual_independent_capture_executor.py
  - tests/fixtures/manual_independent_capture_child.py
  - tests/test_manual_independent_capture_executor.py
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_V1_20260805/README.md
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_V1_20260805/VALIDATION.md
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_V1_20260805/REVIEW.md
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_V1_20260805/RUN_OUTCOME.json
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_V1_20260805/DECISION_ENTRY.json
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_V1_20260805/status.json
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_V1_20260805/guard-preflight.json
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_V1_20260805/guard-final.json
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_V1_20260805/pr-body.md
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_V1_20260805/github-checks.json
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_V1_20260805/final-refs.json
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_V1_20260805/release-receipt.json
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_V1_20260805
proof_question: >-
  Can one exact TheDogs race fixture be captured through a manual-only,
  single-attempt child process with a dedicated lock, fixed isolated profile
  and run root, immediate pre-launch margin check, monotonic cancellation and
  timeout, proven TERM/KILL process-group cleanup, and a GHU-050-valid terminal
  artifact without reading or changing autonomous or canonical state?
hypothesis_id: ghu_051_fixture_proven_isolated_manual_capture_executor_v1
program_track: offline_development
entry_state: >-
  Origin master and the clean task worktree are exactly
  e4f3699986237aad265b34e77d06d536f6046ee4 with tree
  ac9cdde82d3a0ede953e36c9a87d9afd216c5826; GHU-050 is merged through PR 106
  and commit 9a638963ec78c772ec7b19c20961010b37c6cea3; no GHU-051 issue, PR, active
  claim, or duplicate implementation was found.
target_transition: >-
  one focused merged pull request adds only a fixture-proven isolated manual
  executor, its process fixture and tests, the required documentation, and V2
  task/report metadata while preserving all autonomous callers and state
exit_predicate: >-
  The final exact diff accepts one canonical TheDogs URL and bound identity,
  uses only the GHU-050 manual root/profile/lock, launches at most one child in
  a new session after an immediate margin recheck, handles busy/cancel/timeout
  with fail-closed TERM/KILL group cleanup proof, emits a terminal artifact
  accepted by the GHU-050 validator, rejects malformed/unsafe output and paths,
  preserves autonomous sentinel bytes and metadata, passes focused tests and
  fatal Ruff/py_compile/diff checks, receives an independent ACCEPT review,
  passes the honestly classified final suite and exact-head GitHub CI, and is
  merged without runtime, data, deployment, model, Phase 7, or betting action.
source_class: >-
  exact_origin_master_e4f3699_tree_ac9cdde_plus_merged_ghu050_pr106_contract_and_refreshed_owner_ghu051_goal
dataset_version: ghu_051_fixture_process_contract_20260805
evidence_hash: sha256:cb8c5f7e1c28a48cb340462f53059f9a5cfc38c49e758a5e92f4660e673a7ab2
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: >-
  Continue only while origin/master and the task base remain exactly
  e4f3699986237aad265b34e77d06d536f6046ee4, the task worktree contains only
  allowlisted changes, no non-stale overlapping claim or PR appears, browser
  authority remains fixture-only and manual-root-bound, and process cleanup is
  provable; stop on any canonical/autonomous dependency, cleanup uncertainty,
  scope conflict, failed gate, or weakened GHU-050 rule.
---

# GHU-051 isolated manual capture executor

Implement only the research-only, non-canonical, fixture-proven executor. The
manual seam receives no database locator or persistence-capable object. It does
not import or call autonomous capture APIs, inspect shared state, launch a live
browser, score a race, seal GHU-052 evidence, or expose UI/runtime/deployment.

The refreshed exact-master scope replaces the planning ticket's suggested
shared browser refactor with a new manual-only process boundary. This is the
smallest path that can inject the fixed profile/output controls and prove
single-attempt process cleanup without changing autonomous callers.
