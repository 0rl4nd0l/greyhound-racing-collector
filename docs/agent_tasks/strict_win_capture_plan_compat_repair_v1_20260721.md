---
job_id: strict_win_capture_plan_compat_repair_v1_20260721
lane: Evaluation
supporting_lanes:
  - Provenance
  - Repo Hygiene
owner: Codex
allowed_files:
  - docs/agent_tasks/strict_win_capture_plan_compat_repair_v1_20260721.md
  - scripts/strict_win_odds_fixture_capture.py
  - tests/test_strict_win_odds_fixture_capture.py
  - reports/agent_jobs/strict_win_capture_plan_compat_repair_v1_20260721/README.md
  - reports/agent_jobs/strict_win_capture_plan_compat_repair_v1_20260721/STATE.md
  - reports/agent_jobs/strict_win_capture_plan_compat_repair_v1_20260721/DECISIONS.md
  - reports/agent_jobs/strict_win_capture_plan_compat_repair_v1_20260721/COMMANDS.md
  - reports/agent_jobs/strict_win_capture_plan_compat_repair_v1_20260721/VALIDATION.md
  - reports/agent_jobs/strict_win_capture_plan_compat_repair_v1_20260721/REVIEW.md
  - reports/agent_jobs/strict_win_capture_plan_compat_repair_v1_20260721/REGRESSION_ADJUDICATION.md
  - reports/agent_jobs/strict_win_capture_plan_compat_repair_v1_20260721/HASHES.json
  - reports/agent_jobs/strict_win_capture_plan_compat_repair_v1_20260721/CODE_REVIEW.json
  - reports/agent_jobs/strict_win_capture_plan_compat_repair_v1_20260721/guard-preflight.json
  - reports/agent_jobs/strict_win_capture_plan_compat_repair_v1_20260721/status.json
  - reports/agent_jobs/strict_win_capture_plan_compat_repair_v1_20260721/RUN_OUTCOME.json
  - reports/agent_jobs/strict_win_capture_plan_compat_repair_v1_20260721/DECISION_ENTRY.json
approval_required: true
approval_source: Owner explicitly authorized the post-merge PR 57 strict-WIN capture-plan compatibility repair in the active goal on 2026-07-21.
allow_unapproved_safe_extension: false
timeout_seconds: 14400
output_dir: reports/agent_jobs/strict_win_capture_plan_compat_repair_v1_20260721
mutation_mode: safe_extension
production_data_access: false
closeout_scope: repo_only
github_mutation_allowed: false
live_service_mutation_allowed: false
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: strict_win_capture_plan_compat_repair_v1_20260721
proof_question: Does the strict-WIN consumer accept the exact autonomous_live_odds_capture_plan_v1 races/race_id producer contract while failing closed on count, identity, duplication, conflict, and mixed-schema inconsistencies and preserving all five PR 57 integrity contracts?
hypothesis_id: strict_win_pr57_capture_plan_races_race_id_compatibility_regression
program_track: offline_development
entry_state: owner_authorized_true_regression_repair_on_exact_origin_master_merge_95ec4fb_with_review_comment_3618983087
target_transition: one_clean_local_descendant_commit_repairs_races_race_id_compatibility_with_direct_producer_consumer_regression_and_preserved_integrity_contracts
exit_predicate: Exactly one clean local descendant commit of 95ec4fb430484a1c15f99cd187f0ab77715f9791 contains only the authorized task card, strict-WIN script, and focused test; the new producer-to-consumer regression fails on the merged parent and passes on the repair; all requested Python 3.11 and 3.13 compilation, Ruff, focused and related tests, replay, deterministic identity, direct-consumer, review, and V2 closeout gates pass; no GitHub, runtime, ingestion, database, captured-data, model, prediction, activation, EV, betting, merge, deployment, or other-worktree mutation occurs.
source_class: exact_origin_master_95ec4fb_pr57_merge_plus_review_comment_3618983087_and_released_owner_acceptance_validation_evidence
dataset_version: strict_win_pr57_merge_95ec4fb_capture_plan_contract_20260721
evidence_hash: sha256:63c628dcaa2ec10745b169f91316eabc26fd3604185a2772fddc01f8aa12f521
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: Resume only while origin/master remains 95ec4fb430484a1c15f99cd187f0ab77715f9791 or any movement is disjoint from scripts/strict_win_odds_fixture_capture.py, tests/test_strict_win_odds_fixture_capture.py, scripts/autonomous_live_odds_capture.py, and their direct consumers; the isolated worktree remains task-owned; and no active claim overlaps the authorized product files.
---

# Strict-WIN capture-plan compatibility repair

Repair the confirmed post-merge PR #57 schema mismatch at the strict-WIN
consumer boundary. Consume the canonical producer's `races` rows and `race_id`
identity, normalize each accepted row once, preserve count and race-set
consistency, and fail closed on malformed, duplicate, conflicting, discarded,
or contradictory mixed-schema records.

The product diff is limited to
`scripts/strict_win_odds_fixture_capture.py` and
`tests/test_strict_win_odds_fixture_capture.py`. The autonomous producer and
PRs #53/#56 remain read-only owner boundaries. All work is repository-only;
runtime, ingestion, databases, captured data, models, prediction, activation,
deployment, EV, betting, push, PR mutation, and merge are `NOT_RUN`.
