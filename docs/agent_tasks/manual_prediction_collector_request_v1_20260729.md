---
job_id: MANUAL_PREDICTION_COLLECTOR_REQUEST_V1_20260729
title: Add an exact-race manual predictor request handoff to the sole collector
lane: Query Orchestration
supporting_lanes:
  - Runtime Safety
  - Provenance
  - Repo Hygiene
owner: Codex
approval_required: true
approval_source: "Owner /goal request on 2026-07-29 authorizes bounded repository implementation, validation, commit, push, and one draft PR; no live capture, runtime action, deployment, merge, or prediction is authorized."
allow_unapproved_safe_extension: false
timeout_seconds: 28800
mutation_mode: safe_extension
base: origin/master
production_data_access: false
production_data_boundary: "No live prediction or capture, canonical database/history write, service/timer change, training, result access, promotion, deployment, activation, EV/staking, betting, branch-protection change, or merge."
github_mutation_allowed: true
git_history_mutation_allowed: true
live_service_mutation_allowed: false
closeout_scope: repo_only
docs_impact: DOCS_REQUIRED
task_tier: large
recommended_model: high_reasoning
actual_model: gpt-5
worker_model_allowed: false
worker_decision_limit: "No subagents or delegated implementation; the coupled predictor/collector protocol and final authority decision remain with the primary agent."
escalation_needed: false
output_dir: reports/agent_jobs/MANUAL_PREDICTION_COLLECTOR_REQUEST_V1_20260729
allowed_files:
  - configs/prediction/manual-default.json
  - configs/prediction/market-only.json
  - configs/prediction/schemas/market_form_residual_v1.schema.json
  - configs/prediction/schemas/market_only_v1.schema.json
  - docs/on_demand_race_prediction.md
  - docs/agent_tasks/manual_prediction_collector_request_v1_20260729.md
  - race_collection/manual_prediction_collector_request.py
  - scripts/autonomous_live_odds_capture.py
  - scripts/predict_race_now.py
  - scripts/refresh_prejump_upcoming.py
  - scripts/shadow_autopilot_v1.py
  - tests/race_collection/test_manual_prediction_collector_request.py
  - tests/test_autonomous_live_odds_capture.py
  - tests/test_predict_race_now.py
  - tests/test_prejump_prediction_loop.py
  - tests/test_shadow_autopilot_v1.py
  - reports/agent_jobs/MANUAL_PREDICTION_COLLECTOR_REQUEST_V1_20260729/README.md
  - reports/agent_jobs/MANUAL_PREDICTION_COLLECTOR_REQUEST_V1_20260729/STATE.md
  - reports/agent_jobs/MANUAL_PREDICTION_COLLECTOR_REQUEST_V1_20260729/DECISIONS.md
  - reports/agent_jobs/MANUAL_PREDICTION_COLLECTOR_REQUEST_V1_20260729/VALIDATION.md
  - reports/agent_jobs/MANUAL_PREDICTION_COLLECTOR_REQUEST_V1_20260729/REVIEW.md
  - reports/agent_jobs/MANUAL_PREDICTION_COLLECTOR_REQUEST_V1_20260729/CODE_REVIEW.json
  - reports/agent_jobs/MANUAL_PREDICTION_COLLECTOR_REQUEST_V1_20260729/COMMANDS.md
  - reports/agent_jobs/MANUAL_PREDICTION_COLLECTOR_REQUEST_V1_20260729/guard-preflight.json
  - reports/agent_jobs/MANUAL_PREDICTION_COLLECTOR_REQUEST_V1_20260729/status.json
  - reports/agent_jobs/MANUAL_PREDICTION_COLLECTOR_REQUEST_V1_20260729/RUN_OUTCOME.json
  - reports/agent_jobs/MANUAL_PREDICTION_COLLECTOR_REQUEST_V1_20260729/DECISION_ENTRY.json
  - reports/agent_jobs/MANUAL_PREDICTION_COLLECTOR_REQUEST_V1_20260729/pr-body.md
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: MANUAL_PREDICTION_COLLECTOR_REQUEST_V1_20260729
proof_question: Can the manual predictor reuse an exact valid receipt or publish one atomic expiring exact-race request that the already-running sole collector claims at a safe boundary, attempts once, answers exactly once, and the predictor consumes exactly once before entering its existing validation and scoring path?
hypothesis_id: manual_prediction_collector_request_v1_exact_master
program_track: offline_development
entry_state: owner_authorized_clean_exact_origin_master_d5774979_tree_d77ff4d3_after_pr66_pr69_pr70
target_transition: one_clean_draft_pr_adds_a_filesystem_backed_exact_race_request_claim_attempt_response_receipt_consume_protocol_between_the_manual_predictor_and_the_existing_scheduled_collector
exit_predicate: The exact origin/master base and tree are verified; the diff remains inside allowed_files; existing receipt bypass, synthetic request through collector claim and sealed receipt to one predictor continuation, duplicate, replay, expiry, post-jump, identity mismatch, active-phase deferral, crash recovery, hash drift, and consume-once cases pass; focused and nearby tests, Ruff, format, py_compile, schemas, git diff check, and fresh read-only authority review pass; one exact commit is pushed and one draft PR is opened; no live capture, runtime/data/model/training/result/deployment/activation/merge action occurs.
source_class: exact_origin_master_d5774979_tree_d77ff4d3_plus_owner_contract_issue50_and_pr66_pr69_pr70_ancestry
dataset_version: manual_prediction_collector_request_v1_repository_synthetic_validation_20260729
evidence_hash: sha256:22a88e437eed4e0adb69895f43160946fcc19916f8d4b734c0376791f893a735
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: Resume only while origin/master remains d5774979ed075ed5ef129b3556632011069a7d4b with tree d77ff4d36955937164534cfcfe4a7421c595805f; the clean worktree remains task-owned; and no active claim or open PR overlaps this exact target transition.
---

# Manual prediction collector request V1

Implement the smallest append-only filesystem protocol that lets the existing
manual predictor ask the already-running scheduled collector for one exact-race
capture without acquiring, stealing, replacing, or bypassing the shared
collector lock.

The predictor must reuse an existing exact valid receipt before request
publication. The collector may claim only at a safe boundary, may begin at most
one capture attempt, and must publish exactly one supported terminal response.
The predictor must wait for a finite deadline, consume a ready response once,
and continue through the existing exact receipt validation and scoring path.

## Hard stops

- No second collector, parallel browser, daemon API, database protocol, or
  scheduler rewrite.
- No live capture, live prediction, result access, canonical database/history
  mutation, service/timer action, training, model change, promotion, deployment,
  activation, EV/staking, betting, branch-protection change, or merge.
- No weakening of PR #66 exact-race filtering, PR #69 shared-lock safety,
  isolated bundles, append-only odds history, or one-attempt/no-retry behavior.
- No files outside `allowed_files`.
