---
job_id: P7_MODEL_02_FORWARD_SEALED_CORPUS_CAPTURE_CLEAN_20260728
title: Prospectively capture and close authentic Phase 7 training evidence
lane: Provenance
supporting_lanes:
  - Architecture
  - Evaluation
  - Repo Hygiene
owner: Codex
approval_required: true
approval_source: "Owner /goal request on 2026-07-28 authorizes bounded repository implementation, validation, commit, push, and one draft PR; the owner then authorized a clean sibling."
allow_unapproved_safe_extension: false
timeout_seconds: 28800
mutation_mode: safe_extension
base: origin/master
production_data_access: false
production_data_boundary: "No live capture, historical reconstruction, database mutation, service/timer action, training, calibration, model export, bundle registration, prediction, promotion, betting, deployment, activation, merge, or other runtime action."
github_mutation_allowed: true
git_history_mutation_allowed: true
live_service_mutation_allowed: false
closeout_scope: repo_only
docs_impact: DOCS_REQUIRED
task_tier: large
recommended_model: high_reasoning
actual_model: gpt-5
worker_model_allowed: false
worker_decision_limit: "No subagents or delegated implementation; final provenance and publication decisions remain with the primary agent."
escalation_needed: false
output_dir: reports/agent_jobs/P7_MODEL_02_FORWARD_SEALED_CORPUS_CAPTURE_CLEAN_20260728
allowed_files:
  - docs/FORWARD_SEALED_CORPUS_CAPTURE.md
  - docs/CANONICAL_RACE_FORECASTING_PHASE5.md
  - docs/agent_tasks/p7_model_02_forward_sealed_corpus_capture_clean_20260728.md
  - race_collection/forward_sealed_corpus.py
  - race_collection/source_admission.py
  - scripts/collect_forward_sealed_corpus.py
  - tests/race_collection/test_forward_sealed_corpus.py
  - tests/race_collection/test_phase7_source_admission.py
  - reports/agent_jobs/P7_MODEL_02_FORWARD_SEALED_CORPUS_CAPTURE_CLEAN_20260728/README.md
  - reports/agent_jobs/P7_MODEL_02_FORWARD_SEALED_CORPUS_CAPTURE_CLEAN_20260728/STATE.md
  - reports/agent_jobs/P7_MODEL_02_FORWARD_SEALED_CORPUS_CAPTURE_CLEAN_20260728/VALIDATION.md
  - reports/agent_jobs/P7_MODEL_02_FORWARD_SEALED_CORPUS_CAPTURE_CLEAN_20260728/REVIEW.md
  - reports/agent_jobs/P7_MODEL_02_FORWARD_SEALED_CORPUS_CAPTURE_CLEAN_20260728/CODE_REVIEW.json
  - reports/agent_jobs/P7_MODEL_02_FORWARD_SEALED_CORPUS_CAPTURE_CLEAN_20260728/guard-preflight.json
  - reports/agent_jobs/P7_MODEL_02_FORWARD_SEALED_CORPUS_CAPTURE_CLEAN_20260728/status.json
  - reports/agent_jobs/P7_MODEL_02_FORWARD_SEALED_CORPUS_CAPTURE_CLEAN_20260728/RUN_OUTCOME.json
  - reports/agent_jobs/P7_MODEL_02_FORWARD_SEALED_CORPUS_CAPTURE_CLEAN_20260728/DECISION_ENTRY.json
  - reports/agent_jobs/P7_MODEL_02_FORWARD_SEALED_CORPUS_CAPTURE_CLEAN_20260728/pr-body.md
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: P7_MODEL_02_FORWARD_SEALED_CORPUS_CAPTURE_CLEAN_20260728
proof_question: Can the repository prospectively bind immutable pre-jump raw source bytes, production sealed features, and later official-result bytes into deterministic forward-sealed training packages accepted by source admission, without exposing outcomes to feature derivation or performing live/runtime/model actions?
hypothesis_id: phase7_forward_sealed_corpus_capture_clean_exact_base
program_track: offline_development
entry_state: owner_authorized_clean_exact_origin_master_9544d5ce_after_pr68_source_admission_merge
target_transition: one_clean_draft_pr_adds_a_repository_only_forward_sealed_capture_and_closure_pipeline_with_synthetic_admission_proof_and_fail_closed_real_source_timestamp_boundary
exit_predicate: The exact origin/master base and tree are verified; the diff remains inside allowed_files; synthetic end-to-end closure passes source admission; ordering, leakage, identity, runner, hash, crash recovery, idempotence, duplicate closure, deterministic reorder, focused Phase 4/5/6, source-admission, format, compile, schema, and diff checks pass; a fresh read-only review has no critical or warning findings; one exact commit is pushed and one draft PR is opened; no live capture, runtime/data/model/training/deployment/activation/merge action occurs.
source_class: exact_origin_master_9544d5ce_tree_46d6a4e8_plus_owner_contract_and_merged_pr68_source_admission
dataset_version: p7_model_02_repository_synthetic_validation_clean_20260728
evidence_hash: sha256:4d7c5be001bb75c7dbe226e1562c3c196a81379911327ff58d20c86350bf75c8
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: Resume only while origin/master remains 9544d5ce383df0c5bedb7180d890d64fa55ad661 or any movement is disjoint from the allowed product files; the clean worktree remains task-owned; and no active claim, branch, worktree, issue, or PR overlaps this target transition.
---

# P7 Model 02 forward-sealed corpus capture

Implement the smallest repository-supported, append-only prospective recorder
over the existing Race Collection Service ordering, source adapters,
content-addressed artifact store, collector schedules and locks, and atomic
publication patterns.

Pre-jump capture must bind canonical source URLs, source-native race and runner
identities, immutable raw bytes, normalized sealed evidence, production feature
schema and missingness policy, deterministic feature bytes, and prospective
timestamps. Official result capture occurs only after jump and must preserve
raw bytes, aligned source-native identities, collector observation time, and a
source-declared publication timestamp when available. TheDogs' absent
publication timestamp must remain an explicit non-closure state, never an
inferred value.

Only a complete, hash-consistent A+B+C record may publish immutable
`historical-training-example-v1` bytes and a deterministic package accepted by
`race_collection.source_admission`. The bounded CLI executes at most one
supplied canonical collection iteration or reports status. It must not install
or start a service or timer.

## Hard stops

- No live capture or external source access in this run.
- No database, service, timer, label, model, training, prediction, calibration,
  registration, promotion, betting, deployment, activation, merge, or runtime
  action.
- No historical reconstruction or timestamp inference.
- No parallel scraper or widening beyond the declared files.
