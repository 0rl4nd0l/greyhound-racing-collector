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
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/README.md
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/STATE.md
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/VALIDATION.md
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/CODE_REVIEW.md
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/PR_REVIEW.md
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/RUN_OUTCOME.json
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/DECISION_ENTRY.json
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/status.json
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/validation.json
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/pr45_gate.json
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/diff-check.json
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/pytest.log
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/lint.log
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/compile.log
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/artifact-manifest.sha256
approval_required: true
approval_source: The owner's 2026-07-16 PR 46 repair /goal explicitly authorizes amending this original card, one bounded loader/scorer and append-writer repair with regressions and required V2 metadata, one normal commit and push to the existing draft PR 46 branch descending from exact head 106fbc09c6d9e4943365c2c1034b09575031ec2e; it explicitly forbids artifact changes, refit, outcome or production DB access, activation, runtime mutation, promotion and merge.
allow_unapproved_safe_extension: false
timeout_seconds: 14400
output_dir: reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716
mutation_mode: safe_extension
production_data_access: false
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: prospective_market_form_residual_cohort_v2_20260716
proof_question: Can the source-proven post-load in-memory mutation class be closed on exact PR 46 head 106fbc09 without changing frozen artifact bytes, so every prediction and accepted append is cryptographically bound to the verified effective score state while deterministic scoring and append idempotency remain unchanged?
hypothesis_id: frozen_market_form_residual_effective_state_integrity_repair_v1
program_track: offline_development
entry_state: pr45_merged_pr46_blocked_on_source_proven_effective_state_identity_mutation
target_transition: pr46_effective_state_repair_ready_for_independent_merge_review
exit_predicate: Exact PR 46 head 106fbc09 has one normal descendant commit that deep-freezes or encapsulates every score-affecting value, verifies a canonical effective-state hash before scoring and append acceptance, recomputes writer identity from verified state, rejects post-load and score-to-write mutation and forged identity fields, preserves deterministic fixed-fixture scoring and append idempotency, leaves model and manifest bytes exact, passes the post-PR45 integration and required V2 gates, and remains draft, unmerged and unactivated.
source_class: frozen_tier_a_strict_prejump_sportsbet_win_point_in_time_form_and_official_result_evidence_through_20260709
dataset_version: pr46_106fbc09_post_pr45_c1dfd464_effective_state_mutation_probe_20260716
evidence_hash: sha256:80e2ba4c2606062adcc40091e3cd22f9e3d1924582a86e40c538a4f16b1b4dba
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: "Independent merge review may begin only on the exact pushed repair head after all exact-head checks pass; merge, activation, deployment, promotion, runtime, service, unit, timer, production DB, outcome and artifact mutations remain separately forbidden."
docs_impact: DOCS_UPDATED
docs_checked:
  - AGENTS.md
  - docs/semantic_anti_loop_control_v2.md
  - reports/agent_jobs/greyhound_pr45_pr46_integration_20260716/PR46_REVIEW.md
  - reports/agent_jobs/greyhound_pr45_pr46_integration_20260716/BOARD_DECISION.json
  - src/predictor/market_form_residual.py
  - tests/test_market_form_residual.py
docs_changed:
  - docs/agent_tasks/prospective_market_form_residual_cohort_v2_20260716.md
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/README.md
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/STATE.md
  - reports/agent_jobs/prospective_market_form_residual_cohort_v2_20260716/VALIDATION.md
docs_followup: NONE
reason: "The source-proven reopen condition authorizes only the class-level effective-state identity repair on the existing exact PR 46 branch. Artifact, fit, population, candidate, feature, weight, outcome, database, runtime, activation, promotion, merge and new issue/branch/PR changes remain forbidden."
---

# Prospective market-form residual cohort v2

This is the original card amended in place under the owner's 2026-07-16 PR 46
repair goal. The integration review proved that post-load nested mutation can
change predictions while the cached artifact hashes and record identity stay
unchanged, satisfying the prior decision's source-proven scorer-contract reopen
condition. This card authorizes only that class-level repair on the existing
exact PR branch. Do not rerun selection, reopen folds, refit, inspect prospective
outcomes, assign a cohort cutoff, or alter either frozen artifact.

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
append-only shadow record contract. All score-affecting state must be deeply
immutable or encapsulated; score and append must re-derive and verify its
canonical effective-state hash against the original frozen contract. The
writer must recompute record identity from verified state and reject caller
forgery or score-to-write mutation. It must continue to reject outcome fields,
partial or duplicate runner sets, invalid odds/features/provenance, hash/schema
drift, non-finite values, and conflicting duplicate output. It must not hook
into a collector, runtime, database, model registry, service, timer, prediction
pointer, promotion path, betting path, or PR #45 branch.

Production remains `KEEP_BASELINE / market-only implied probability`.
