---
job_id: form_only_v1_acquisition_foundation_20260718
title: Build the acquisition-only FORM_ONLY_V1 foundation
lane: Provenance
supporting_lanes:
  - Data Engineering
  - Testing
  - Reporting
owner: Codex
approval_required: true
approval_source: The owner's 2026-07-18 /goal explicitly authorizes one clean current-origin/master acquisition-only implementation lane, deterministic materialization of odds-free development and outcome-unopened input-only packets, local validation, and at most one narrow draft PR without altering PRs 46 through 48.
allow_unapproved_safe_extension: false
timeout_seconds: 21600
output_dir: reports/agent_jobs/form_only_v1_acquisition_foundation_20260718
mutation_mode: safe_extension
production_data_access: false
production_data_boundary: Read only immutable raw pre-race evidence and frozen pre-2026-07-10 label sources. Do not open any post-2026-07-09 outcome, read or write a production database, backfill, mutate runtime capture, or materialize outside the task report bundle.
github_mutation_allowed: true
git_history_mutation_allowed: true
live_service_mutation_allowed: false
allowed_files:
  - docs/agent_tasks/form_only_v1_acquisition_foundation_20260718.md
  - docs/form_only_v1_acquisition.md
  - scripts/build_form_only_v1_packet.py
  - tests/test_build_form_only_v1_packet.py
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/README.md
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/STATE.md
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/VALIDATION.md
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/CODE_REVIEW.md
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/RUN_OUTCOME.json
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/DECISION_ENTRY.json
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/status.json
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/validation.json
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/guard-preflight.json
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/duplicate-preflight.json
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/commands.log
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/feature_contract.json
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/development_manifest.json
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/development_races.csv
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/development_runners.csv
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/development_features.csv
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/development_exclusions.csv
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/development_source_inventory.csv
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/overlap_reconciliation.csv
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/reconciliation_summary.json
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/market_coverage.json
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/out_of_time_manifest.json
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/out_of_time_races.csv
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/out_of_time_runners.csv
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/out_of_time_exclusions.csv
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/out_of_time_source_inventory.csv
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/deterministic_regeneration.json
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/artifact-manifest.sha256
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/diff-check.json
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/compile.log
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/lint.log
  - reports/agent_jobs/form_only_v1_acquisition_foundation_20260718/pytest.log
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: form_only_v1_acquisition_foundation_20260718
proof_question: Can one canonical raw-history builder deterministically materialize an odds-free pre-race FORM_ONLY_V1 development packet and a separate outcome-unopened out-of-time input manifest, reconcile every known 530-runner builder discrepancy, and prove zero target or post-target history, target outcomes, market fields, post-jump corrections, or dog-identity features?
hypothesis_id: canonical_raw_asof_form_only_v1_acquisition_v1
program_track: offline_development
entry_state: no_canonical_raw_asof_form_packet_builder_builders_disagree_on_530_overlap_runners_market_only_remains_safe_baseline
target_transition: deterministic_hash_bound_form_only_v1_inputs_ready_for_separately_authorized_experiment
exit_predicate: Candidate counts and authoritative source identity are reproduced; every included and excluded race and runner is frozen with source hashes, bytes, capture and jump times, and label provenance; the 530-row legacy overlap reconciliation is disclosed only as a separately hashed NON_AUTHORITATIVE_DIAGNOSTIC bundle and cannot affect trainer artifacts; development and Jul 11 through Aug 9 input-only packets rebuild byte-identically; trust-domain linkability, leakage, forbidden-field, ambiguity, and malformed-input guards pass; post-2026-07-09 outcomes remain unopened; and no model, database, runtime, service, timer, activation, betting, or PR 46 through 48 mutation occurs.
source_class: immutable_raw_contemporaneous_pre_race_cards_plus_raw_prior_history_and_frozen_pre_2026_07_10_label_sources
dataset_version: form_only_v1_acquisition_cutoff_20260709_out_of_time_input_window_20260711_20260809
evidence_hash: sha256:bf36b16d70b3afe524c24f14aadb7ea92087f956c7f6c34b44f0b035bab6f751
capabilities:
  - READ
  - REPORT_WRITE
  - DATASET_MATERIALIZE
  - CODE_EDIT
  - PUBLISH
resume_only_if: Resume only while origin/master remains the verified base or advances without overlapping FORM_ONLY_V1 work, no active claim or PR owns the same paths, raw inputs remain byte-identical or changes are explicitly rebound, post-2026-07-09 outcomes stay unopened, and the lane remains acquisition-only. Stop on unresolved reconciliation, missing source/capture/jump/label provenance, reconstructed out-of-time cards, forbidden odds/outcome/identity fields, source drift, runtime or production access, or overlap with PRs 46 through 48.
docs_impact: DOCS_UPDATED
docs_checked:
  - docs/semantic_anti_loop_control_v2.md
  - docs/development/greyhound_git_dirt_adoption.md
  - docs/architecture/ml_pipeline.md
docs_changed:
  - docs/form_only_v1_acquisition.md
docs_followup: Model fitting, train-only vocabulary fitting, metric evaluation, market comparison, and the separate T-60 DATA_MISSING experiment require a new owner-approved task after this packet is sealed.
reason: The canonical as-of feature semantics, packet schema, provenance and leakage boundaries are durable behavior and must be documented with the implementation.
---

# FORM_ONLY_V1 acquisition foundation

Build exactly one canonical odds-free point-in-time feature packet from raw
pre-race cards and raw prior history. Reconcile prior builder outputs only as
diagnostic evidence; never choose one builder as truth.

## Hard boundaries

- Never open or infer any post-2026-07-09 target outcome.
- Never fit, calibrate, blend, evaluate, promote, activate, serve, or bet.
- Never read or write a production database, backfill a card, or reconstruct a
  non-contemporaneous out-of-time input.
- Never include odds, OPEN, LOW, HIGH, SP, target outcomes, post-jump
  corrections, dog identity as a feature, target history, or post-target
  history.
- Keep official race-page labels and TheDogs published-history labels as
  distinct provenance classes; never upgrade published history to Tier A.
- Keep PRs 46, 47, and 48, runtime services, timers, locks, and model artifacts
  read-only and outside the diff.
