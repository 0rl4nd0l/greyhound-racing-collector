---
job_id: race_grade_metadata_transport_v1_20260718
lane: Provenance
supporting_lanes:
  - Runtime
  - Testing
owner: Codex
allowed_files:
  - docs/agent_tasks/race_grade_metadata_transport_v1_20260718.md
  - upcoming_race_browser.py
  - scripts/refresh_prejump_upcoming.py
  - scripts/predict_market_form_residual.py
  - utils/csv_metadata.py
  - tests/test_csv_download_hardening.py
  - tests/test_prejump_prediction_loop.py
  - tests/test_predict_market_form_residual.py
  - tests/test_upcoming_race_time_mapping.py
  - reports/agent_jobs/race_grade_metadata_transport_v1_20260718/STATE.md
  - reports/agent_jobs/race_grade_metadata_transport_v1_20260718/DECISIONS.md
  - reports/agent_jobs/race_grade_metadata_transport_v1_20260718/VALIDATION.md
  - reports/agent_jobs/race_grade_metadata_transport_v1_20260718/CODE_REVIEW.md
  - reports/agent_jobs/race_grade_metadata_transport_v1_20260718/PR_REVIEW.md
  - reports/agent_jobs/race_grade_metadata_transport_v1_20260718/RUN_OUTCOME.json
  - reports/agent_jobs/race_grade_metadata_transport_v1_20260718/DECISION_ENTRY.json
  - reports/agent_jobs/race_grade_metadata_transport_v1_20260718/status.json
  - reports/agent_jobs/race_grade_metadata_transport_v1_20260718/validation.json
  - reports/agent_jobs/race_grade_metadata_transport_v1_20260718/pytest.log
  - reports/agent_jobs/race_grade_metadata_transport_v1_20260718/lint.log
  - reports/agent_jobs/race_grade_metadata_transport_v1_20260718/compile.log
  - reports/agent_jobs/race_grade_metadata_transport_v1_20260718/determinism.log
  - reports/agent_jobs/race_grade_metadata_transport_v1_20260718/diff-check.json
approval_required: true
approval_source: The owner's 2026-07-18 instruction, "Proceed", follows the exact proposed repair to transport source-proven meeting-card grade metadata into the existing downloader and expose its fail-closed quarantine reason through the one-command frozen predictor. It does not authorize live runtime mutation or retrospective scoring.
allow_unapproved_safe_extension: false
timeout_seconds: 10800
output_dir: reports/agent_jobs/race_grade_metadata_transport_v1_20260718
mutation_mode: safe_extension
production_data_access: false
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: race_grade_metadata_transport_v1_20260718
proof_question: Can the existing pre-jump refresh carry a recognized grade from the exact TheDogs meeting-card race into the exact canonical race-page download, while rejecting mismatched or unknown hints, and can the offline one-command frozen predictor report the resulting outcome-free quarantine reason instead of the opaque race_feature_packet_not_found error?
hypothesis_id: exact_race_grade_transport_and_quarantine_diagnostic_v1
program_track: offline_development
entry_state: exact_race_runner_and_odds_capture_rejected_before_feature_materialization_when_race_page_omits_grade_and_manual_predictor_hides_the_source_quarantine
target_transition: source_proven_exact_race_grade_transport_and_precise_outcome_free_quarantine_diagnostic_ready_not_activated
exit_predicate: Fixture-only regressions prove refresh passes the selected race context to the downloader; the downloader admits only a recognized grade whose canonical TheDogs URL, race date, venue identity and race number match the requested page; accepted metadata records source the grade to the exact meeting-card race; page-header and exact-URL structured metadata remain fallback authorities; missing, unknown, mismatched or conflicting hints remain quarantined; the offline frozen predictor reports a bounded exact-race missing-grade quarantine reason when no sealed feature packet exists; alias handling includes MAND and MANDURAH without weakening identity; existing packet scoring remains deterministic; and no database, network execution, service, unit, timer, runtime, deployment, activation, promotion, betting, outcome inspection, model, artifact, threshold, feature, normalization or seed mutation occurs.
source_class: exact_thedogs_meeting_card_race_context_plus_exact_canonical_race_page_and_outcome_free_prejump_refresh_quarantine_report
dataset_version: mandurah_r10_prejump_missing_grade_diagnostic_20260717_no_outcomes_plus_sealed_fixtures
evidence_hash: sha256:d469b0d3c4e6cd3a5855c5d7d83e5f5290ac544d9272a7b74ed8fe1fe856abb9
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: A fixture or fresh pre-jump no-outcome race proves a materially different metadata or discovery defect. Direct production or copied database access, prospective or retrospective outcome inspection for the target race, network execution during validation, collector restart, runtime/service/unit/timer change, deployment, activation, promotion, betting, model or artifact mutation, threshold or feature change and PR merge remain forbidden. PR 47 stays draft and unmerged; PR 45 resource-isolation ancestry and PR 46 effective-state integrity remain prerequisites for any later activation.
docs_impact: DOCS_CHECKED_NO_CHANGE
docs_checked:
  - AGENTS.md
  - docs/agent_tasks/race_first_market_form_residual_prediction_v3_20260716.md
  - docs/agent_tasks/manual_live_market_form_residual_prediction_v2_20260716.md
  - docs/manual_live_market_form_residual_prediction.md
  - upcoming_race_browser.py
  - scripts/refresh_prejump_upcoming.py
docs_changed: []
docs_followup: Later activation documentation remains separately gated because this task changes repository source and fixtures only.
reason: The 2026-07-17 Mandurah R10 pre-jump refresh proved an exact canonical URL, exact race-time mapping and complete six-runner alignment but quarantined the canonical TheDogs export solely as target_metadata_not_verified:missing_target_grade. The selected meeting-card race object is the nearest source-proven context, but refresh currently calls download_race_csv with only its URL and discards that context. The manual scorer then reports only race_feature_packet_not_found because quarantined races never materialize a feature packet. This task repairs those two narrow boundaries without inferring grade from historical runner form, odds, sponsor text, outcomes or venue.
---

# Exact race grade transport and frozen-prediction quarantine diagnostic

Carry the already-selected TheDogs meeting-card race object into
`download_race_csv` as a read-only hint. The downloader may use only an
explicit grade normalized by the repository's closed target-grade vocabulary,
and only after the hint's canonical race URL, date, venue identity and race
number all bind to the requested canonical page. Record the accepted grade's
provenance as `thedogs_meeting_card_exact_race`. Never infer grade from the
historical `G` columns, runner form, sponsor or race name, odds, venue or
outcomes. Preserve exact page-header and structured-data extraction as the
fallback. Unknown, missing, identity-mismatched and materially conflicting
metadata must fail closed and retain quarantine behavior.

When race-first packet discovery finds no sealed feature packet, inspect only
bounded, outcome-free pre-jump refresh reports under the already-authorized
evidence roots. Match the exact race identity and emit only a closed,
normalized quarantine reason such as
`race_feature_packet_quarantined:missing_target_grade`; do not expose raw
exception text or make a quarantined artifact scoreable. Existing sealed
packet discovery, hash binding, timing, completeness and deterministic scoring
must remain unchanged.

Validation is fixture-only and offline. Do not open the production database,
fetch the past Mandurah page, inspect its result, run or restart collectors,
touch installed services/units/timers, deploy, activate, promote, bet or merge.
