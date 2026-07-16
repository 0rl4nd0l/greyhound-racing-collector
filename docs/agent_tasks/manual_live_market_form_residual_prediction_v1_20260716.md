---
job_id: manual_live_market_form_residual_prediction_v1_20260716
lane: Evaluation
supporting_lanes:
  - Provenance
  - Testing
owner: Codex
allowed_files:
  - docs/agent_tasks/manual_live_market_form_residual_prediction_v1_20260716.md
  - docs/manual_live_market_form_residual_prediction.md
  - scripts/predict_market_form_residual.py
  - tests/test_predict_market_form_residual.py
  - artifacts/frozen_models/market_form_residual_v1/model.json
  - artifacts/frozen_models/market_form_residual_v1/manifest.json
  - docs/agent_tasks/prospective_market_form_residual_cohort_v2_20260716.md
  - src/predictor/market_form_residual.py
  - tests/test_market_form_residual.py
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
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/STATE.md
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/DECISIONS.md
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/VALIDATION.md
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/CODE_REVIEW.md
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/PR_REVIEW.md
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/RUN_OUTCOME.json
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/DECISION_ENTRY.json
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/status.json
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/validation.json
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/pytest.log
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/lint.log
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/compile.log
  - reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716/diff-check.json
approval_required: true
approval_source: The owner's 2026-07-16 instruction, "Okay proceed with this", explicitly authorizes the previously proposed one-command manual prediction path and a live Sandown R2 prediction.
allow_unapproved_safe_extension: false
timeout_seconds: 10800
output_dir: reports/agent_jobs/manual_live_market_form_residual_prediction_v1_20260716
mutation_mode: safe_extension
production_data_access: false
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: manual_live_market_form_residual_prediction_v1_20260716
proof_question: Can an operator obtain one deterministic outcome-free frozen residual prediction from explicit verified pre-jump form and strict Sportsbet capture artifacts with one command, without database, service, deployment, promotion or betting mutation?
hypothesis_id: manual_live_frozen_residual_prediction_cli_v1
program_track: offline_development
entry_state: frozen_market_form_residual_model_and_shadow_scorer_ready_awaiting_separate_activation
target_transition: manual_prejump_frozen_residual_prediction_cli_ready_not_activated
exit_predicate: A documented CLI loads the exact frozen model and explicit pre-jump form, sidecar and capture artifacts; validates race identity, runner completeness, timing, hashes and outcomes absence; prints normalized full and half rankings deterministically; and performs no database, runtime, service, deployment, promotion, betting or automatic persistence mutation.
source_class: explicit_verified_thedogs_form_sidecar_and_strict_prejump_sportsbet_capture_artifacts
dataset_version: manual_prediction_cli_inputs_v1_no_outcomes
evidence_hash: sha256:73c6889318ee375ce512267866527b79c839f5a6e9435aa2166814b83219bd64
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: Automatic network fetching, scheduler integration, database reads or writes, service activation, deployment, promotion, betting, and PR merge remain separate transitions requiring an exact owner-approved card; this command consumes explicit pre-jump artifacts only.
docs_impact: DOCS_UPDATED
docs_checked:
  - AGENTS.md
  - docs/agent_tasks/prospective_market_form_residual_cohort_v2_20260716.md
  - src/predictor/market_form_residual.py
  - scripts/autonomous_live_odds_capture.py
docs_changed:
  - docs/manual_live_market_form_residual_prediction.md
docs_followup: Automatic exact-event network acquisition remains separately gated.
reason: This is a narrow operator interface over the already frozen scorer. It composes the merged resource-isolation PR 45 ancestry with draft frozen-model PR 46 ancestry, but does not activate either daemon or touch the production database.
---

# Manual live market-form residual prediction v1

Implement one explicit-input, outcome-free command over the frozen residual
loader/scorer. Inputs are a verified pre-jump TheDogs form CSV and sidecar plus
an autonomous Sportsbet capture report or attempts JSONL containing a complete,
strict, pre-jump WIN market for the same race.

The command must fail closed on missing or duplicate runners, identity or box
mismatch, unsafe form metadata, absent capture provenance, post-jump timing,
invalid odds, outcome-like fields, artifact hash/schema drift, or non-normalized
probabilities. It may print JSON to stdout only. It must not automatically
append shadow output, inspect results, open any database, fetch the network,
change runtime or services, deploy, promote, bet, merge a PR, or activate the
model.

The branch must retain ancestry from merged PR #45 head
`aa35fa70fc49199acde09f5561b521ddb00d45aa` and draft frozen-model PR #46 head
`106fbc09c6d9e4943365c2c1034b09575031ec2e`.
