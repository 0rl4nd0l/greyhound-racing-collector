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
  - scripts/shadow_autopilot_v1.py
  - tests/test_shadow_autopilot_v1.py
  - scripts/run_shadow_non_tgr_rf_evaluation.py
  - tests/test_run_shadow_non_tgr_rf_evaluation.py
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
approval_source: The owner's 2026-07-16 instruction, "Okay proceed to fixing the residual model issue so we can begin predicting asap", authorizes the narrow missing-feature handoff and race-first frozen prediction interface while retaining the previously stated non-activation boundaries.
allow_unapproved_safe_extension: false
timeout_seconds: 10800
output_dir: reports/agent_jobs/race_first_market_form_residual_prediction_v3_20260716
mutation_mode: safe_extension
production_data_access: false
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: race_first_market_form_residual_prediction_v3_20260716
proof_question: Does a complete strict pre-jump capture from the bounded odds-refresh lane reach exact feature generation in the same full cycle, produce a generator-hash-bound packet accepted by the frozen residual scorer, and remain invocable race-first without database, network, service or activation mutation by this task?
hypothesis_id: captured_race_exact_feature_handoff_and_race_first_frozen_scoring_v3
program_track: offline_development
entry_state: strict_capture_complete_but_exact_feature_packet_missing_for_later_bounded_races
target_transition: captured_race_exact_feature_handoff_and_race_first_frozen_prediction_ready_not_activated
exit_predicate: Focused tests prove that a successfully captured race omitted by the bounded primary refresh is added once to the same-cycle daily feature inputs with a fresh pre-jump cutoff; new feature packets bind exact implementation-file hashes and legacy reviewed packets remain loadable; the stdout-only scorer can resolve one unambiguous race from outcome-free evidence and produces deterministic normalized full and half predictions; no production database, network, service, unit, timer, runtime, deployment, promotion, betting, merge or activation mutation is performed.
source_class: strict_prejump_sportsbet_capture_plus_verified_thedogs_form_and_hash_bound_shadow_feature_packet_no_outcomes
dataset_version: race_first_residual_handoff_v3_no_outcomes
evidence_hash: sha256:346ae155caf087f9e9e440f7922777430e60849fda981829cd7bc4a465aa918d
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: Runtime or service/unit/timer changes, direct production or copied database access, live deployment, activation, model promotion, betting, prospective outcome inspection and PR merge remain forbidden and require a separate exact owner-approved task. Later activation must descend from merged PR 45 head aa35fa70fc49199acde09f5561b521ddb00d45aa and this task's reviewed head.
docs_impact: DOCS_UPDATED
docs_checked:
  - AGENTS.md
  - docs/agent_tasks/prospective_market_form_residual_cohort_v2_20260716.md
  - docs/agent_tasks/manual_live_market_form_residual_prediction_v2_20260716.md
  - docs/agent_tasks/greyhound_resource_isolation_20260716.md
  - scripts/shadow_autopilot_v1.py
  - scripts/daily_race_ingest_shadow_orchestrator.py
  - scripts/run_shadow_non_tgr_rf_evaluation.py
  - scripts/predict_market_form_residual.py
docs_changed:
  - docs/manual_live_market_form_residual_prediction.md
docs_followup: Live activation and runtime proof remain separately gated.
reason: The failure is upstream of the frozen model. The full cycle refreshes a broader bounded set for strict odds capture but passes only the smaller primary refresh directory to feature generation, so later captured races have no exact packet. New packets are also rejected by a scorer pinned only to the historical PR 45 commit. This card repairs those two contract seams and the race-first invocation without altering the frozen fit, features, strengths, normalization, seeds, thresholds, units or runtime.
---

# Race-first market-form residual prediction v3

Repair only the same-cycle handoff from a successful strict odds capture to the
existing shadow feature generator. Preserve the primary refresh limit and the
PR #45 resource-isolation behavior. Add only successful capture-plan form
inputs that are not already represented by the primary input set, and evaluate
the daily pre-jump cutoff immediately before feature generation so a race that
jumped during capture cannot enter scoring.

Bind newly generated feature packets to SHA-256 hashes of the declared
implementation files. The manual scorer must accept the existing exact PR #45
legacy packet and new packets only when every declared implementation hash
matches the local reviewed file. It must fail closed on absent, extra or
mismatched identity data.

Provide a race-first discovery option over explicitly supplied evidence roots.
Discovery may inspect only outcome-free form sidecars, shadow feature packets
and strict capture reports. It must select exactly one internally consistent
pre-jump packet/capture set or fail closed. Explicit-path invocation remains
supported. The command writes canonical JSON to stdout only.

Do not fit or alter the model; change features, preprocessing, strengths,
normalization, thresholds or seeds; inspect outcomes; access a database or
network from this task; touch services, units, timers or runtime state; deploy,
activate, promote, bet or merge.
