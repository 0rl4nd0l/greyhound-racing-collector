---
job_id: prospective_market_form_residual_cohort_v2_20260716
lane: Evaluation
supporting_lanes:
  - Provenance
  - Reporting
owner: Codex
allowed_files:
  - docs/agent_tasks/prospective_market_form_residual_cohort_v2_20260716.md
  - src/predictor/market_form_residual.py
  - tests/test_market_form_residual.py
  - artifacts/frozen_models/market_form_residual_v1/model.json
  - artifacts/frozen_models/market_form_residual_v1/manifest.json
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/README.md
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/STATE.md
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/VALIDATION.md
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/CODE_REVIEW.md
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/PR_REVIEW.md
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/RUN_OUTCOME.json
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/DECISION_ENTRY.json
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/status.json
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/validation.json
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/fit_population.json
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/deterministic_proof.json
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/pr45_gate.json
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/diff-check.json
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/pytest.log
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/lint.log
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/compile.log
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/artifact-manifest.sha256
approval_required: true
approval_source: The owner's 2026-07-16 fresh /goal explicitly authorizes amending this original card, one final deterministic historical fit, model persistence, minimal canonical shadow scorer code and tests, commit/push, and one draft PR; it explicitly forbids activation and all runtime or production-data mutation.
allow_unapproved_safe_extension: false
timeout_seconds: 14400
output_dir: reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716
mutation_mode: safe_extension
production_data_access: false
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: prospective_market_form_residual_cohort_v2_20260716
proof_question: Can the already frozen market-form residual candidate contract be materialized once, deterministically, from the complete eligible historical Tier A population through 2026-07-09 and loaded and scored fail-closed without activating any live path or inspecting prospective outcomes?
hypothesis_id: frozen_market_form_residual_materialization_v1
program_track: offline_development
entry_state: blocked_contract_defect_no_cohort_registered_market_baseline_retained
target_transition: deterministic_frozen_market_form_residual_model_materialized_shadow_scorer_ready
exit_predicate: Exactly one shared base model and preprocessing state are fit on the predeclared 678-race Tier A population, persisted with complete identity and hash provenance, reproduced deterministically, and exercised by a minimal fail-closed full/half shadow scorer; no prospective cutoff, outcome access, activation, deployment, promotion, merge, database access, service change, or production pointer change occurs.
source_class: frozen_tier_a_strict_prejump_sportsbet_win_point_in_time_form_and_official_result_evidence_through_20260709
dataset_version: historical_win_forward_snapshot_eval_20260715_strict_timing_feature_provenance_678
evidence_hash: sha256:aeb1758953e48addd0717c9e1cac1c8f5e0c95c338120d10bc5788895825978f
capabilities:
  - READ
  - REPORT_WRITE
  - RESEARCH_FIT
  - MODEL_PERSIST
  - CODE_EDIT
  - PUBLISH
resume_only_if: "Runtime activation is a separate owner-approved task whose branch is a descendant of both this frozen-model PR and PR #45 resource-isolation head aa35fa70fc49199acde09f5561b521ddb00d45aa; the shared collector lock is idle and a new exact activation card authorizes runtime paths."
docs_impact: DOCS_UPDATED
docs_checked:
  - AGENTS.md
  - docs/semantic_anti_loop_control_v2.md
  - reports/agent_jobs/historical_eligibility_refresh_v2_20260716/README.md
  - reports/agent_jobs/prospective_market_form_residual_challenger_20260716/candidate_definition.json
  - reports/agent_jobs/prospective_market_form_residual_challenger_20260716/evaluate_market_form_residual.py
docs_changed:
  - docs/agent_tasks/prospective_market_form_residual_cohort_v2_20260716.md
docs_followup: NONE
reason: "This amended card repairs the recorded contract defect for one deterministic offline fit, model persistence, minimal scorer implementation, tests and one draft PR only. Production DB access, canonical or copied DB writes, prospective outcome access, cohort registration, runtime or unit changes, deployment, activation, promotion, betting, merge and PR #45 mutation remain forbidden."
---

# Prospective market-form residual cohort v2

This is the original card amended in place under the owner's 2026-07-16 goal.
It authorizes one offline materialization transition before any prospective
cohort exists. The historical OOF windows remain exhausted: do not rerun
selection, reopen folds, inspect prospective outcomes, or assign a cohort
cutoff.

The only fit population is the complete frozen Tier A population already
specified by the July 16 evaluator's final-fit branch: 678 complete races and
4,752 exact runners with race dates no later than 2026-07-09. The fit must
preserve the exact candidate algorithm, 16-feature order, median imputation
plus missing indicators, training-population mean/std scaling, within-race
centering, fixed market offset, ridge L2 1.0, residual cap 0.35, optimizer
configuration, and zero initialization. Full and half outputs must derive from
that one shared base state at strengths 1.0 and 0.5. No sweep, tuning,
threshold change, alternate population, alternate serialization, or
outcome-informed choice is permitted.

Before fitting, persist the complete included race and runner identities and
the 140 frozen historical exclusions with reason codes. The canonical model
and manifest must record coefficients, preprocessing, feature types and
missing-value behavior, algorithm/config/seeds, dependency versions, source
and candidate hashes, artifact hashes, schema/loading/scoring contracts and a
fixed prediction fixture. A same-fit replay may only prove byte identity or
semantic equivalence; it must not create or compare an alternative.

The canonical scorer may only load this artifact, validate exact feature and
runner alignment, derive normalized full/half probabilities, and write an
append-only shadow record contract. It must reject outcome fields, partial or
duplicate runner sets, invalid odds/features/provenance, hash/schema drift,
non-finite values, and conflicting duplicate output. It must not hook into a
collector, runtime, database, model registry, service, timer, prediction
pointer, promotion path, betting path, or PR #45 branch.

Production remains `KEEP_BASELINE / market-only implied probability`.
