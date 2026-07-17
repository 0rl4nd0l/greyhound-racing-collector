---
job_id: race_first_market_form_residual_prediction_v3_20260716
lane: Evaluation
supporting_lanes:
  - Provenance
  - Testing
owner: Codex
allowed_files:
  - docs/agent_tasks/race_first_market_form_residual_prediction_v3_20260716.md
  - docs/manual_live_market_form_residual_prediction.md
  - scripts/predict_market_form_residual.py
  - tests/test_predict_market_form_residual.py
  - reports/agent_jobs/race_first_market_form_residual_prediction_v3_20260716/STATE.md
  - reports/agent_jobs/race_first_market_form_residual_prediction_v3_20260716/DECISIONS.md
  - reports/agent_jobs/race_first_market_form_residual_prediction_v3_20260716/VALIDATION.md
  - reports/agent_jobs/race_first_market_form_residual_prediction_v3_20260716/CODE_REVIEW.md
  - reports/agent_jobs/race_first_market_form_residual_prediction_v3_20260716/PR_REVIEW.md
  - reports/agent_jobs/race_first_market_form_residual_prediction_v3_20260716/RUN_OUTCOME.json
  - reports/agent_jobs/race_first_market_form_residual_prediction_v3_20260716/DECISION_ENTRY.json
  - reports/agent_jobs/race_first_market_form_residual_prediction_v3_20260716/status.json
  - reports/agent_jobs/race_first_market_form_residual_prediction_v3_20260716/validation.json
  - reports/agent_jobs/race_first_market_form_residual_prediction_v3_20260716/pytest.log
  - reports/agent_jobs/race_first_market_form_residual_prediction_v3_20260716/lint.log
  - reports/agent_jobs/race_first_market_form_residual_prediction_v3_20260716/compile.log
  - reports/agent_jobs/race_first_market_form_residual_prediction_v3_20260716/determinism.log
  - reports/agent_jobs/race_first_market_form_residual_prediction_v3_20260716/diff-check.json
approval_required: true
approval_source: The owner's 2026-07-17 instruction, "okay. lets fix the frozen prediction method", authorizes the narrow source-proven venue and grade input-contract repair on the existing draft PR 47 lineage while retaining every prior non-activation boundary.
allow_unapproved_safe_extension: false
timeout_seconds: 10800
output_dir: reports/agent_jobs/race_first_market_form_residual_prediction_v3_20260716
mutation_mode: safe_extension
production_data_access: false
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: race_first_market_form_residual_prediction_v3_20260716
proof_question: Can the frozen residual CLI accept the repository's exact canonical hyphenated venue identity and the source-proven Restricted/Restricted Win grade equivalence while continuing to reject malformed venues, unknown or genuinely different grades, ambiguous races and every existing provenance, completeness, timing and outcome violation?
hypothesis_id: frozen_residual_known_input_alias_contract_repair_v4
program_track: offline_development
entry_state: race_first_frozen_prediction_ready_but_source_proven_hyphenated_venue_and_restricted_grade_packets_fail_before_scoring
target_transition: frozen_residual_known_venue_and_grade_contract_repaired_ready_not_activated
exit_predicate: Sealed outcome-free regressions reproduce target_venue_invalid for a canonical hyphenated venue and feature_row_target_grade_mismatch for Restricted versus Restricted Win before the fix; the repaired explicit and race-first paths accept only those source-proven forms, emit identical deterministic normalized full and half predictions, reject malformed venue tokens plus unknown or genuinely different grades, preserve every existing hash, runner, scratch, URL, timing, ambiguity and no-outcome gate, and perform no model, database, network, runtime, deployment, promotion, betting, merge or activation mutation.
source_class: strict_prejump_sportsbet_capture_plus_verified_thedogs_form_and_hash_bound_shadow_feature_packet_no_outcomes
dataset_version: frozen_residual_input_contract_regressions_20260717_no_outcomes
evidence_hash: sha256:ae061ccaf9938b0d17a8d757a75f6ccf33c135f22c0dc3274df6f18fa68b3ebb
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: A new exact pre-jump packet proves another materially different scorer-contract defect, or a separate exact owner-approved task authorizes runtime activation. Runtime or service/unit/timer changes, direct production or copied database access, deployment, model promotion, betting, prospective outcome inspection and PR merge remain forbidden. Later activation must retain merged PR 45 resource-isolation ancestry, the repaired PR 46 effective-state ancestry, and this task's reviewed head.
docs_impact: DOCS_UPDATED
docs_checked:
  - AGENTS.md
  - https://github.com/0rl4nd0l/greyhound-racing-collector/issues/49
  - docs/agent_tasks/prospective_market_form_residual_cohort_v2_20260716.md
  - docs/agent_tasks/manual_live_market_form_residual_prediction_v2_20260716.md
  - scripts/predict_market_form_residual.py
docs_changed:
  - docs/manual_live_market_form_residual_prediction.md
docs_followup: Live activation and runtime proof remain separately gated.
reason: Two exact outcome-free pre-jump packets satisfy the released V3 decision's reopen condition. Ladbrokes Q1 Lakeside R3 is rejected because the CLI permits only underscore venue tokens even though the repository's canonical venue contract preserves hyphens, and Healesville R5 is rejected because the sidecar's Restricted label and feature packet's Restricted Win label are a source-proven equivalence. This card repairs only those CLI comparison boundaries without altering the frozen fit, artifacts, features, strengths, normalization, seeds, thresholds, generator, runtime or activation state.
---

# Race-first market-form residual prediction v3

This original race-first card is amended in place for the source-proven V4
input-contract repair authorized on 2026-07-17. The already-released V3 handoff
remains complete. This continuation changes only the manual scorer's handling
of repository-canonical venue tokens and one documented grade equivalence.

Accept canonical venue strings made from uppercase letters, digits,
underscores and internal hyphens without rewriting the identity bound into the
sidecar, feature packet, capture or race ID. Apply one deterministic grade
canonicalization contract at comparison boundaries: `Restricted` and
`Restricted Win` are equivalent. Empty, unknown, ambiguous and genuinely
different grade values must continue to fail closed.

Regression tests must exercise the full sealed-packet explicit and race-first
paths, including the Sportsbet URL validator, and prove that deterministic full
and half probabilities are unchanged for the existing fixture. Preserve every
hash, provenance, path containment, runner completeness, scratch, timestamp,
URL, ambiguity and no-outcome guard.

Do not fit or alter the model or artifacts; change features, preprocessing,
strengths, normalization, thresholds or seeds; inspect outcomes; access a
database or network from this task; touch services, units, timers or runtime
state; deploy, activate, promote, bet or merge.
