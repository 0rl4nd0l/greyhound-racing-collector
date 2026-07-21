---
job_id: strict_win_nonready_identity_validation_repair_v1_20260721
lane: Evaluation
supporting_lanes:
  - Provenance
  - Repo Hygiene
owner: Codex
allowed_files:
  - docs/agent_tasks/strict_win_nonready_identity_validation_repair_v1_20260721.md
  - scripts/strict_win_odds_fixture_capture.py
  - tests/test_strict_win_odds_fixture_capture.py
  - reports/agent_jobs/strict_win_nonready_identity_validation_repair_v1_20260721/README.md
  - reports/agent_jobs/strict_win_nonready_identity_validation_repair_v1_20260721/STATE.md
  - reports/agent_jobs/strict_win_nonready_identity_validation_repair_v1_20260721/DECISIONS.md
  - reports/agent_jobs/strict_win_nonready_identity_validation_repair_v1_20260721/COMMANDS.md
  - reports/agent_jobs/strict_win_nonready_identity_validation_repair_v1_20260721/VALIDATION.md
  - reports/agent_jobs/strict_win_nonready_identity_validation_repair_v1_20260721/REVIEW.md
  - reports/agent_jobs/strict_win_nonready_identity_validation_repair_v1_20260721/REGRESSION_ADJUDICATION.md
  - reports/agent_jobs/strict_win_nonready_identity_validation_repair_v1_20260721/HASHES.json
  - reports/agent_jobs/strict_win_nonready_identity_validation_repair_v1_20260721/CODE_REVIEW.json
  - reports/agent_jobs/strict_win_nonready_identity_validation_repair_v1_20260721/guard-preflight.json
  - reports/agent_jobs/strict_win_nonready_identity_validation_repair_v1_20260721/status.json
  - reports/agent_jobs/strict_win_nonready_identity_validation_repair_v1_20260721/RUN_OUTCOME.json
  - reports/agent_jobs/strict_win_nonready_identity_validation_repair_v1_20260721/DECISION_ENTRY.json
approval_required: true
approval_source: Owner explicitly authorized this narrow successor repair to rejected local candidate 36c13b12d61e83de892c79475d8fbf4b6ff3ffb5 on 2026-07-21.
allow_unapproved_safe_extension: false
timeout_seconds: 14400
output_dir: reports/agent_jobs/strict_win_nonready_identity_validation_repair_v1_20260721
mutation_mode: safe_extension
production_data_access: false
closeout_scope: repo_only
github_mutation_allowed: false
live_service_mutation_allowed: false
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: strict_win_nonready_identity_validation_repair_v1_20260721
proof_question: Does the strict-WIN capture-plan consumer validate and uniquely normalize every canonical or guarded-legacy race identity before status filtering while preserving valid zero-ready behavior and all five PR 57 integrity contracts?
hypothesis_id: strict_win_nonready_identity_validation_gap_36c13b12
program_track: offline_development
entry_state: owner_authorized_test_gap_narrow_fix_on_exact_rejected_local_candidate_36c13b12_and_reviewed_base_95ec4fb
target_transition: one_clean_local_successor_commit_to_36c13b12_repairs_plan_wide_identity_validation_and_is_ready_for_fresh_independent_local_review
exit_predicate: Exactly one normal clean local successor commit to 36c13b12d61e83de892c79475d8fbf4b6ff3ffb5 contains only this task card and the two authorized implementation files; independent probes fail before and pass after; requested Python 3.11 and 3.13 compilation, Ruff, focused and five-file tests, sealing, replay, manifest, hash, direct producer-consumer, determinism, review, and V2 closeout gates pass; no GitHub, runtime, ingestion, database, captured-data, model, prediction, activation, EV, betting, merge, deployment, PR 56, producer, or direct-consumer mutation occurs.
source_class: exact_rejected_candidate_36c13b12_parent_95ec4fb_plus_released_independent_rejection_artifacts_and_fresh_local_reproduction
dataset_version: strict_win_nonready_identity_validation_repair_20260721
evidence_hash: sha256:5ae8abc43ef7fe9f1ad78b8c52209246f0b102546bcc085df90c177174ef733d
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: Resume only while the worktree remains a clean task-owned successor lane at 36c13b12d61e83de892c79475d8fbf4b6ff3ffb5 before repair, origin/master is 95ec4fb430484a1c15f99cd187f0ab77715f9791 or its movement is disjoint from the strict-WIN consumer, focused tests, producer, and direct consumers, and no active claim overlaps the two authorized product files.
---

# Strict-WIN non-READY identity validation repair

Repair only the independently confirmed plan-wide identity-validation gap in
the local strict-WIN capture-plan compatibility candidate. Validate and
normalize canonical `races`/`race_id` and guarded legacy
`items`/`canonical_race_identity` records before status-based eligibility;
reject malformed, missing, conflicting, duplicate, colliding, or contradictory
identities on `READY_TO_CAPTURE`, `BLOCKED`, and `NO_DUE_WINDOW` records.

The product diff is limited to
`scripts/strict_win_odds_fixture_capture.py` and
`tests/test_strict_win_odds_fixture_capture.py`. The producer, its direct
consumers, PR #56, GitHub, runtime, ingestion, databases, captured data, model,
prediction, activation, deployment, EV, and betting surfaces are read-only or
`NOT_RUN`.
