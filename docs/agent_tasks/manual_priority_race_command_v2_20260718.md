---
job_id: manual_priority_race_command_v2_20260718
title: Standalone lock-aware named-race priority collection and prediction command
lane: Query Orchestration
supporting_lanes:
  - Provenance
  - Runtime proof
  - Testing
owner: Codex
approval_required: true
approval_source: The owner's 2026-07-18 /goal explicitly authorizes one clean current-origin/master implementation lane for issue 50, exact adoption of the current reviewed PR 46 and PR 47 heads, append-only target collection when explicitly executed, one optional outcome-free pre-jump proof, local validation and closeout, and no deployment, merge, or activation.
allow_unapproved_safe_extension: false
timeout_seconds: 21600
output_dir: reports/agent_jobs/manual_priority_race_command_v2_20260718
mutation_mode: safe_extension
production_data_access: false
production_data_boundary: No outcome, result, label, model, threshold, promotion, betting, deployment, service, timer, or general production-data mutation. The only conditionally authorized canonical write is append-only strict WIN and PLACE live_odds through the reviewed collector path for the exact selected pre-jump race, under the shared lock, fixed windows, and explicit double execution gate.
owner_db_append_only_approval: true
github_mutation_allowed: false
git_history_mutation_allowed: true
live_service_mutation_allowed: false
allowed_files:
  - docs/agent_tasks/manual_priority_race_command_v2_20260718.md
  - artifacts/frozen_models/market_form_residual_v1/manifest.json
  - artifacts/frozen_models/market_form_residual_v1/model.json
  - docs/agent_tasks/frozen_residual_exact_head_integration_v1_20260718.md
  - docs/agent_tasks/manual_live_market_form_residual_prediction_v1_20260716.md
  - docs/agent_tasks/manual_live_market_form_residual_prediction_v2_20260716.md
  - docs/agent_tasks/pr46_operational_integrity_v2_20260718.md
  - docs/agent_tasks/pr46_whole_history_consistency_repair_20260718.md
  - docs/agent_tasks/prospective_market_form_residual_cohort_v2_20260716.md
  - docs/agent_tasks/race_first_market_form_residual_prediction_v3_20260716.md
  - docs/agent_tasks/race_grade_metadata_transport_v1_20260718.md
  - docs/manual_live_market_form_residual_prediction.md
  - docs/manual_priority_race_prediction.md
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/CODE_REVIEW.md
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/DECISIONS.md
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/DECISION_ENTRY.json
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/PR_REVIEW.md
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/RUN_OUTCOME.json
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/STATE.md
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/VALIDATION.md
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/compile.log
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/diff-check.json
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/lint.log
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/pytest.log
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/status.json
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/validation.json
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/CODE_REVIEW.md
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/DECISION_ENTRY.json
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/PR_REVIEW.md
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/README.md
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/RUN_OUTCOME.json
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/STATE.md
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/VALIDATION.md
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/artifact-manifest.sha256
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/compile.log
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/deterministic_proof.json
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/diff-check.json
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/fit_population.json
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/lint.log
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/pr45_gate.json
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/pytest.log
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/status.json
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/validation.json
  - scripts/predict_market_form_residual.py
  - scripts/refresh_prejump_upcoming.py
  - scripts/run_priority_race_prediction.py
  - scripts/run_shadow_non_tgr_rf_evaluation.py
  - scripts/shadow_autopilot_v1.py
  - src/predictor/market_form_residual.py
  - tests/test_csv_download_hardening.py
  - tests/test_market_form_residual.py
  - tests/test_predict_market_form_residual.py
  - tests/test_prejump_prediction_loop.py
  - tests/test_run_priority_race_prediction.py
  - tests/test_run_shadow_non_tgr_rf_evaluation.py
  - tests/test_shadow_autopilot_v1.py
  - tests/test_upcoming_race_time_mapping.py
  - upcoming_race_browser.py
  - utils/csv_metadata.py
  - reports/agent_jobs/manual_priority_race_command_v2_20260718/README.md
  - reports/agent_jobs/manual_priority_race_command_v2_20260718/STATE.md
  - reports/agent_jobs/manual_priority_race_command_v2_20260718/DECISIONS.md
  - reports/agent_jobs/manual_priority_race_command_v2_20260718/VALIDATION.md
  - reports/agent_jobs/manual_priority_race_command_v2_20260718/CODE_REVIEW.json
  - reports/agent_jobs/manual_priority_race_command_v2_20260718/ADVERSARIAL_REVIEW.md
  - reports/agent_jobs/manual_priority_race_command_v2_20260718/RUNTIME_PROOF.md
  - reports/agent_jobs/manual_priority_race_command_v2_20260718/RUN_OUTCOME.json
  - reports/agent_jobs/manual_priority_race_command_v2_20260718/DECISION_ENTRY.json
  - reports/agent_jobs/manual_priority_race_command_v2_20260718/status.json
  - reports/agent_jobs/manual_priority_race_command_v2_20260718/validation.json
  - reports/agent_jobs/manual_priority_race_command_v2_20260718/diff-check.json
  - reports/agent_jobs/manual_priority_race_command_v2_20260718/ancestry.json
  - reports/agent_jobs/manual_priority_race_command_v2_20260718/integrity.json
  - reports/agent_jobs/manual_priority_race_command_v2_20260718/determinism.json
  - reports/agent_jobs/manual_priority_race_command_v2_20260718/legacy-runtime.json
  - reports/agent_jobs/manual_priority_race_command_v2_20260718/optional-manual-proof.json
  - reports/agent_jobs/manual_priority_race_command_v2_20260718/WAIT_RESULT.json
  - reports/agent_jobs/manual_priority_race_command_v2_20260718/WAIT_RESULT.json.log
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: manual_priority_race_command_v2_20260718
proof_question: Can one standalone command safely plan and explicitly execute target-only named-race refresh, strict fixed-window WIN and PLACE capture, fresh sealed feature generation, and deterministic non-persisted full and half residual prediction while preserving shared-lock ownership, runner identity, provenance, append-only idempotency, PR 45 ancestry, exact reviewed PR 46 and PR 47 adoption, and the read-only PR 48 legacy runtime boundary?
hypothesis_id: standalone_lock_aware_named_race_priority_command_v2
program_track: prospective_readiness
entry_state: issue_50_open_master_contains_pr45_reviewed_pr46_2c595d27_and_pr47_0ae5937c_are_green_siblings_and_no_supported_target_only_operator_command_exists
target_transition: standalone_named_race_priority_command_proven_awaiting_separate_activation
exit_predicate: A clean branch from origin/master c1dfd464 preserves PR 45 head aa35fa70 exactly once, adopts independently reviewed PR 46 head 2c595d27 and PR 47 head 0ae5937c exactly once, implements a plan-only-default target-only command with explicit double-gated collection, bounded lock and fixed-window waits, strict runner provenance append-only idempotency gates, fresh outcome-free feature sealing, deterministic normalized non-persisted full and half stdout, focused and full validation, and independent adversarial review; PR 48 head f776bfd1, its worktree, service, timer, lock ownership, and runtime outputs remain unchanged; and the run ends in one owner-authorized stop state.
source_class: live_issue_50_plus_current_origin_master_and_exact_reviewed_pr45_pr46_pr47_heads_with_read_only_pr48_runtime
dataset_version: master_c1dfd464_pr45_aa35fa70_pr46_2c595d27_pr47_0ae5937c_pr48_f776bfd1_issue50_20260718
evidence_hash: sha256:bdcd21ac8b6ad89df09db8d18bec61dcc5db2b77779f96fbd081d1173e7a20b0
capabilities:
  - READ
  - REPORT_WRITE
  - DATASET_MATERIALIZE
  - CODE_EDIT
  - CANONICAL_DB_WRITE
  - PUBLISH
resume_only_if: Resume only while origin/master remains c1dfd464cf6ecfb2034f96ac1a8d3ea58d4e6afa, the exact reviewed live PR 46 and PR 47 heads remain 2c595d27ac748d3df8e4031d5491c76606c5be89 and 0ae5937cde87131c714fb7383c58ce13e3cfbc06, PR 46 exact-head CI run 29631150452 remains successful, the change stays inside this allowlist and issue 50 contract, and PR 48 remains read-only. Stop on ancestry drift, contract drift, active-lane overlap, outcome access, model or threshold mutation, deployment, service or timer change, betting, promotion, cohort cutoff, GitHub mutation, merge, or any write outside exact isolated evidence plus append-only live_odds for one explicitly executed target.
docs_impact: DOCS_UPDATED
docs_checked:
  - AGENTS.md
  - docs/manual_live_market_form_residual_prediction.md
  - issue 50
docs_changed:
  - docs/manual_priority_race_prediction.md
docs_followup: A separate owner-approved activation lane must decide PR 48 disposition and document installed runtime only after exact-head review and fresh runtime proof.
reason: Issue 50 has a source-proven standalone tracer-bullet gap. The owner has authorized implementation and an optional single outcome-free pre-jump execution, while keeping runtime activation, PR 48 mutation, outcomes, models, promotion, betting, deployment, and merge out of scope.
---

# Manual priority race command V2

Implement and prove the smallest standalone exact-race operator command. The
default must be plan-only and write-free. Collection requires explicit
execution plus the existing explicit Sportsbet scrape approval, holds the
shared daemon lock only while working, preserves T-60/T-30/T-10/T-2 windows,
and admits only one exact race. Feature sealing may read only history strictly
before the target jump and must not persist the final full/half prediction.

The card permits one optional live command only if an exact race is still
pre-jump after all static gates pass. That command may write isolated refresh,
capture, and feature evidence plus append-only strict WIN and PLACE `live_odds`
rows for that race. It may not read or write target outcomes, ingest results,
change a model, threshold, cohort, unit, service, timer, runtime worktree, or
deployment, or mutate GitHub.

PR #48 is read-only legacy runtime. Its recommended later disposition is report
evidence only, never an action in this lane.
