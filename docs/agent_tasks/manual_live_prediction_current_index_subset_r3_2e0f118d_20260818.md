---
job_id: MANUAL_LIVE_PREDICTION_CURRENT_INDEX_SUBSET_R3_2E0F118D_20260818
title: Publish only safe races from mixed refresh coverage and finish live proof
lane: Query Orchestration
supporting_lanes: [Provenance, Reporting]
owner: Codex
approval_required: true
approval_source: >-
  The owner's active 2026-08-18 /goal authorizes the smallest current-index
  repair, tests, review, GitHub publication and merge, exact generated runtime
  deployment, natural cycles, and guarded one-at-a-time prediction POSTs until
  one valid sealed On-demand Forecast succeeds. It forbids predictive-semantic
  change, frozen-test alteration, result-aware repair, promotion, ROI/edge or
  betting claims, wagering, blind retry, provenance relaxation, and direct
  canonical database mutation.
allow_unapproved_safe_extension: false
timeout_seconds: 43200
mutation_mode: safe_extension
allow_audit_code_changes: false
base: 2e0f118d02ec35270d2a04c89df54578342da792
production_data_access: false
production_data_boundary: >-
  Read installed generated services, immutable collector evidence, sealed
  reports/publications, identities, operation stores, and the canonical racing
  database. Existing collector-owned cycles retain only established append-only
  writes. Each freshly admitted prediction job may write its isolated job,
  audit, request, receipt, attempt, and sealed bundle. No direct canonical DB
  write, training, scoring, model/config/pointer mutation, result use, lock
  deletion, service-file hand edit, experiment activation, or betting action.
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
  The repair crosses a hash-bound producer/publisher contract, an odds-capture
  shared-data boundary, generated deployment, live collection, and sealed job
  verification.
worker_model_allowed: true
worker_decision_limit: >-
  Independent read-only reviewers may inspect the exact diff. The primary agent
  retains implementation, Git, deployment, runtime, POST, and final authority.
escalation_needed: false
output_dir: reports/agent_jobs/MANUAL_LIVE_PREDICTION_CURRENT_INDEX_SUBSET_R3_2E0F118D_20260818
allowed_files:
  - docs/agent_tasks/manual_live_prediction_current_index_subset_r3_2e0f118d_20260818.md
  - docs/operator_ui_v1/MANUAL_LIVE_PREDICTION_RUNBOOK.md
  - race_collection/synchronous_manual_capture.py
  - scripts/refresh_prejump_upcoming.py
  - tests/test_prejump_prediction_loop.py
  - tests/race_collection/test_synchronous_manual_capture.py
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_CURRENT_INDEX_SUBSET_R3_2E0F118D_20260818/README.md
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_CURRENT_INDEX_SUBSET_R3_2E0F118D_20260818/STATE.md
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_CURRENT_INDEX_SUBSET_R3_2E0F118D_20260818/DECISIONS.md
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_CURRENT_INDEX_SUBSET_R3_2E0F118D_20260818/VALIDATION.md
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_CURRENT_INDEX_SUBSET_R3_2E0F118D_20260818/REVIEW.md
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_CURRENT_INDEX_SUBSET_R3_2E0F118D_20260818/CODE_REVIEW.json
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_CURRENT_INDEX_SUBSET_R3_2E0F118D_20260818/RUN_OUTCOME.json
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_CURRENT_INDEX_SUBSET_R3_2E0F118D_20260818/DECISION_ENTRY.json
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_CURRENT_INDEX_SUBSET_R3_2E0F118D_20260818/status.json
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_CURRENT_INDEX_SUBSET_R3_2E0F118D_20260818/guard-preflight.json
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_CURRENT_INDEX_SUBSET_R3_2E0F118D_20260818/guard-final.json
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_CURRENT_INDEX_SUBSET_R3_2E0F118D_20260818/pr-body.md
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_CURRENT_INDEX_SUBSET_R3_2E0F118D_20260818/github-checks.json
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_CURRENT_INDEX_SUBSET_R3_2E0F118D_20260818/final-refs.json
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_CURRENT_INDEX_SUBSET_R3_2E0F118D_20260818/deployment-evidence.json
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_CURRENT_INDEX_SUBSET_R3_2E0F118D_20260818/live-attempts.json
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_CURRENT_INDEX_SUBSET_R3_2E0F118D_20260818/SHA256SUMS
  - reports/agent_jobs/MANUAL_LIVE_PREDICTION_CURRENT_INDEX_SUBSET_R3_2E0F118D_20260818/release-receipt.json
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: MANUAL_LIVE_PREDICTION_CURRENT_INDEX_SUBSET_R3_2E0F118D_20260818
proof_question: >-
  Can the refresh producer expose a bounded safe current-index subset without
  removing candidates needed by odds capture, so mixed coverage publishes only
  complete races, zero-complete coverage remains rejected, and exact deployed
  master can produce one valid sealed manual prediction?
hypothesis_id: explicit_safe_current_index_subset_and_live_prediction_r3
program_track: prospective_readiness
entry_state: >-
  PR #139 merged as exact master 2e0f118d, tree bd3d6aac, with all CI green. Its
  exact deployed collector completed a natural cycle, appended 14 live-odds
  rows, and had two fully safe plus two incomplete selected races. The refresh
  globally emitted METADATA_COVERAGE_INCOMPLETE and the strict publisher
  correctly rejected it. selected_races is also the odds-capture fallback
  input, so it must remain intact. No prediction POST has occurred.
target_transition: >-
  A reviewed clean PR introduces a distinct bounded current-index subset while
  retaining all odds candidates; exact merged master is generated and deployed;
  a fresh natural internally consistent current index is sealed; and guarded
  distinct jobs stop at the first valid sealed prediction with probabilities.
exit_predicate: >-
  Tests prove all-complete SUCCESS, mixed subset exclusion, zero-complete
  rejection, odds-candidate preservation, stale/hash conflicts, contradictory
  publication rejection, and replay; focused checks, independent review, and
  exact-head CI pass; merge/deployment identities match; fresh natural index
  and each guarded POST satisfy provenance; one terminal success passes bundle,
  request, receipt, model, config, schema, race, runner, timing, hash, and
  probability checks; issue #135 and the runbook are updated; V2 closeout and
  registry release succeed.
source_class: exact_origin_master_2e0f118d_plus_hash_bound_mixed_coverage_runtime_report_20260818
dataset_version: manual_live_prediction_current_index_subset_runtime_r3_20260818
evidence_hash: sha256:1b77a7d2674ea44cc82ad7b560df47d3f40bb7812cda05fb8f9592ef5dfe3ea1
capabilities: [READ, REPORT_WRITE, CODE_EDIT, PUBLISH, RUNTIME_CHANGE]
resume_only_if: >-
  Continue only while identities remain verifiable, the canonical lock is never
  bypassed, predictive semantics and frozen experiments stay unchanged, only
  complete races enter the current index, all selected races remain available
  to odds capture, deployment stays generator-owned, and every further POST has
  fresh readiness plus a distinct repair or genuinely new race after a
  pre-POST availability stop.
---

# Safe current-index subset and live proof

This task owns the producer/publisher contract repair reproduced by the first
exact-2e0f118d natural cycle and continues issue #135 through one valid sealed
On-demand Forecast. It may not weaken publication or predictive semantics.
