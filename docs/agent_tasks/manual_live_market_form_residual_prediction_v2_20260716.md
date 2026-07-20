---
job_id: manual_live_market_form_residual_prediction_v2_20260716
lane: Evaluation
supporting_lanes:
  - Provenance
  - Testing
owner: Codex
allowed_files:
  - docs/agent_tasks/manual_live_market_form_residual_prediction_v2_20260716.md
  - docs/manual_live_market_form_residual_prediction.md
  - scripts/predict_market_form_residual.py
  - tests/test_predict_market_form_residual.py
  - artifacts/frozen_models/market_form_residual_v1/model.json
  - artifacts/frozen_models/market_form_residual_v1/manifest.json
  - src/predictor/market_form_residual.py
  - tests/test_market_form_residual.py
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v2_20260716/STATE.md
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v2_20260716/DECISIONS.md
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v2_20260716/VALIDATION.md
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v2_20260716/CODE_REVIEW.md
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v2_20260716/PR_REVIEW.md
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v2_20260716/RUN_OUTCOME.json
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v2_20260716/DECISION_ENTRY.json
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v2_20260716/status.json
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v2_20260716/validation.json
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v2_20260716/pytest.log
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v2_20260716/lint.log
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v2_20260716/compile.log
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v2_20260716/diff-check.json
approval_required: true
approval_source: The owner's 2026-07-16 instructions, "use the system" and "Okay proceed with this", authorize the manual prediction path using already-materialized system evidence; they do not authorize database, network, runtime, service, activation, deployment, promotion, betting or merge mutations.
allow_unapproved_safe_extension: false
timeout_seconds: 10800
output_dir: reports/agent_jobs/manual_live_market_form_residual_prediction_v2_20260716
mutation_mode: safe_extension
production_data_access: false
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: manual_live_market_form_residual_prediction_v2_20260716
proof_question: Can an operator obtain one deterministic outcome-free frozen residual prediction from an exact hash-bound pre-jump system feature packet and strict Sportsbet capture with one command, without recomputing features or accessing a database, network, service or activation path?
hypothesis_id: manual_live_frozen_residual_prediction_exact_feature_packet_v2
program_track: offline_development
entry_state: exact_prejump_system_feature_packet_identified_manual_cli_not_yet_ready
target_transition: manual_prejump_frozen_residual_prediction_cli_from_exact_system_feature_packet_ready_not_activated
exit_predicate: A documented CLI reads each input once into immutable bytes; validates an exact system shadow-feature packet against its adjacent shadow and implementation manifests, verified TheDogs target sidecar, complete strict pre-jump Sportsbet capture and frozen artifacts; prints normalized full and half rankings; and performs no database, network, runtime, service, persistence, deployment, promotion, betting, merge or activation mutation.
source_class: sealed_prejump_system_shadow_feature_rows_with_hash_binding_manifests_verified_thedogs_sidecar_and_strict_prejump_sportsbet_capture
dataset_version: manual_prediction_cli_exact_system_feature_packet_v2_no_outcomes
evidence_hash: sha256:0ebecaf980665545aa8c19d1a4b1ef976bd069049d42f7f6ebde0f3b29a36b62
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: Automatic feature generation, network fetching, scheduler integration, database reads or writes, service activation, deployment, promotion, betting and PR merge remain separate transitions requiring a new exact owner-approved card; this command consumes already-materialized evidence only.
docs_impact: DOCS_UPDATED
docs_checked:
  - AGENTS.md
  - docs/agent_tasks/manual_live_market_form_residual_prediction_v1_20260716.md
  - docs/agent_tasks/prospective_market_form_residual_cohort_v2_20260716.md
  - src/predictor/market_form_residual.py
  - scripts/run_shadow_non_tgr_rf_evaluation.py
  - scripts/autonomous_live_odds_capture.py
docs_changed:
  - docs/manual_live_market_form_residual_prediction.md
docs_followup: Automatic exact-event feature and odds acquisition remains separately gated.
reason: Independent review proved form-only feature reconstruction is not frozen-contract parity. This revised task replaces recomputation with the exact outcome-free system feature packet already materialized before Sandown R2, while retaining the frozen model and all non-activation boundaries.
---

# Manual live market-form residual prediction v2

Implement an explicit-input, stdout-only command over the frozen residual
loader/scorer. The feature source must be an already-materialized
`shadow_feature_rows.json` packet whose raw bytes are hash-bound by its adjacent
`shadow_manifest.json` and `implementation_file_manifest.json`. The command
must never reconstruct model features from the form CSV.

Use the adjacent TheDogs form sidecar only to bind exact target identity,
runner completeness and jump time. Use a strict autonomous Sportsbet accepted
capture for odds and scratches. Read every artifact exactly once into immutable
bytes and hash the bytes actually parsed and scored.

Fail closed on packet/manifest/path/hash/schema drift; race, source CSV,
runner, box or name mismatch; missing or extra features; outcome-like fields;
unsafe/post-result URLs; ambiguous capture; any missing required validation
field; or timestamps that do not prove
`metadata <= feature freeze <= fetch <= append <= score < jump`.

No output path, model override, feature recomputation, database, network,
service, automatic persistence, runtime, deployment, promotion, betting, merge
or activation option is allowed.

The Sandown R2 proof packet is read-only evidence at SHA-256
`0ebecaf980665545aa8c19d1a4b1ef976bd069049d42f7f6ebde0f3b29a36b62`,
with feature freeze `2026-07-16T18:18:32.550267+10:00`; it was materialized
before the strict odds capture and the 18:58 jump.
