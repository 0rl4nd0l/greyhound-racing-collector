---
job_id: MANUAL_LIVE_PREDICTION_RUNNER_TIMING_R4_7BA3AF86_20260818
title: Exclude late runner sources and finish one sealed manual prediction
lane: Query Orchestration
supporting_lanes: [Provenance, Reporting]
owner: Codex
approval_required: true
approval_source: >-
  The owner's active 2026-08-18 /goal authorizes reproduced readiness repairs,
  tests, review, publication, merge, exact generated deployment, natural cycles,
  and guarded one-at-a-time prediction POSTs until one valid sealed On-demand
  Forecast succeeds. It forbids predictive-semantic change, frozen-test
  alteration, result-aware repair, promotion, ROI/edge or betting claims,
  wagering, blind retry, provenance relaxation, and direct canonical DB writes.
allow_unapproved_safe_extension: false
timeout_seconds: 43200
mutation_mode: safe_extension
allow_audit_code_changes: false
base: 7ba3af86aaaad337ce1319ce9e18c3504cde20e3
production_data_access: false
production_data_boundary: >-
  Read installed services and immutable evidence. Existing collector cycles may
  perform only established append-only writes. Each freshly admitted prediction
  job may write its isolated job/audit/request/receipt/attempt/bundle. No direct
  canonical DB write, training, scoring, model/config/pointer mutation, result
  use, lock deletion, unit hand edit, experiment activation, or betting action.
owner_db_append_only_approval: true
github_mutation_allowed: true
git_history_mutation_allowed: true
live_service_mutation_allowed: true
closeout_scope: repo_and_publish
docs_impact: DOCS_REQUIRED
task_tier: critical
recommended_model: high_reasoning
actual_model: gpt-5
why_this_model: >-
  The repair must keep producer timing eligibility identical to the strict
  publisher without weakening provenance, then continue through live sealing.
worker_model_allowed: true
worker_decision_limit: >-
  Independent reviewers are read-only. The primary agent retains implementation,
  Git, deployment, runtime, POST, and final authority.
escalation_needed: false
output_dir: reports/agent_jobs/MANUAL_LIVE_PREDICTION_RUNNER_TIMING_R4_7BA3AF86_20260818
allowed_files:
  - docs/agent_tasks/manual_live_prediction_runner_timing_r4_7ba3af86_20260818.md
  - docs/operator_ui_v1/MANUAL_LIVE_PREDICTION_RUNBOOK.md
  - race_collection/synchronous_manual_capture.py
  - scripts/refresh_prejump_upcoming.py
  - tests/test_prejump_prediction_loop.py
  - tests/race_collection/test_synchronous_manual_capture.py
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_RUNNER_TIMING_R4_7BA3AF86_20260818/README.md
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_RUNNER_TIMING_R4_7BA3AF86_20260818/STATE.md
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_RUNNER_TIMING_R4_7BA3AF86_20260818/DECISIONS.md
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_RUNNER_TIMING_R4_7BA3AF86_20260818/VALIDATION.md
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_RUNNER_TIMING_R4_7BA3AF86_20260818/REVIEW.md
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_RUNNER_TIMING_R4_7BA3AF86_20260818/CODE_REVIEW.json
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_RUNNER_TIMING_R4_7BA3AF86_20260818/RUN_OUTCOME.json
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_RUNNER_TIMING_R4_7BA3AF86_20260818/DECISION_ENTRY.json
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_RUNNER_TIMING_R4_7BA3AF86_20260818/status.json
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_RUNNER_TIMING_R4_7BA3AF86_20260818/guard-preflight.json
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_RUNNER_TIMING_R4_7BA3AF86_20260818/guard-final.json
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_RUNNER_TIMING_R4_7BA3AF86_20260818/pr-body.md
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_RUNNER_TIMING_R4_7BA3AF86_20260818/github-checks.json
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_RUNNER_TIMING_R4_7BA3AF86_20260818/final-refs.json
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_RUNNER_TIMING_R4_7BA3AF86_20260818/deployment-evidence.json
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_RUNNER_TIMING_R4_7BA3AF86_20260818/live-attempts.json
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_RUNNER_TIMING_R4_7BA3AF86_20260818/SHA256SUMS
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_RUNNER_TIMING_R4_7BA3AF86_20260818/release-receipt.json
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: MANUAL_LIVE_PREDICTION_RUNNER_TIMING_R4_7BA3AF86_20260818
proof_question: >-
  Can runner-source timing be included in the exact per-race eligibility set so
  a post-jump observation is excluded, later safe races publish consistently,
  and exact deployed master completes one valid sealed manual prediction?
hypothesis_id: runner_observation_timing_subset_and_live_prediction_r4
program_track: prospective_readiness
entry_state: >-
  PR #140 merged as master 7ba3af86/tree 5bc2cbf3 and was exactly deployed.
  Natural run 20260818T194102+1000_odds_capture emitted SUCCESS with three
  metadata-eligible races and one exclusion, but publication rejected because
  DARW race 3 runner metadata was captured at 19:42:06 for a 19:42:00 jump.
  WRGL race 4 and TWN race 4 sealed individually. No prediction POST occurred.
target_transition: >-
  One reviewed clean PR binds runner-source observation timing into the exact
  current-index subset; exact merged master is generated/deployed; a natural
  consistent current index is verified; and guarded distinct jobs stop at the
  first valid sealed prediction with probabilities.
exit_predicate: >-
  Tests prove pre-jump inclusion, post-jump exclusion, stale observation
  exclusion, mixed and zero-eligible behavior, exact publisher recomputation,
  and prior stale/hash contradictions; review/CI pass; exact merge/deployment
  and natural index identities match; one guarded POST succeeds and all sealed
  provenance/hash/probability checks pass; issue #135/runbook/V2 closeout finish.
source_class: exact_origin_master_7ba3af86_plus_hash_bound_runner_timing_rejection_20260818
dataset_version: manual_live_prediction_runner_timing_runtime_r4_20260818
evidence_hash: sha256:f6f4b8a955f2ea70a9e8f4aa1b5a47afc1deb2067b7f35ae2403c99a0b211aef
capabilities: [READ, REPORT_WRITE, CODE_EDIT, PUBLISH, RUNTIME_CHANGE]
resume_only_if: >-
  Continue only while identities are verifiable, the lock is never bypassed,
  predictive semantics/frozen experiments stay unchanged, every admitted race
  has a fresh pre-jump runner source, deployment stays generator-owned, and
  every further POST has fresh readiness plus a distinct repair or new race.
---

# Runner-source timing repair and live proof

This task owns the reproduced post-jump runner-source eligibility omission and
continues issue #135 without weakening the publisher or predictive semantics.
