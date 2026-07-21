---
job_id: strict_win_fixture_salvage_e66ce_v5_20260720
lane: Evaluation
supporting_lanes:
  - Provenance
  - Repo Hygiene
owner: Codex
allowed_files:
  - docs/agent_tasks/strict_win_fixture_salvage_e66ce_v5_20260720.md
  - scripts/strict_win_odds_fixture_capture.py
  - tests/test_strict_win_odds_fixture_capture.py
  - reports/agent_jobs/strict_win_fixture_salvage_e66ce_v5_20260720/README.md
  - reports/agent_jobs/strict_win_fixture_salvage_e66ce_v5_20260720/STATE.md
  - reports/agent_jobs/strict_win_fixture_salvage_e66ce_v5_20260720/DECISIONS.md
  - reports/agent_jobs/strict_win_fixture_salvage_e66ce_v5_20260720/COMMANDS.md
  - reports/agent_jobs/strict_win_fixture_salvage_e66ce_v5_20260720/VALIDATION.md
  - reports/agent_jobs/strict_win_fixture_salvage_e66ce_v5_20260720/FINDINGS_TO_FIX_MATRIX.md
  - reports/agent_jobs/strict_win_fixture_salvage_e66ce_v5_20260720/BASE_STATE.json
  - reports/agent_jobs/strict_win_fixture_salvage_e66ce_v5_20260720/SALVAGE_COMPARISON.patch
  - reports/agent_jobs/strict_win_fixture_salvage_e66ce_v5_20260720/SALVAGE_COMPARISON.sha256
  - reports/agent_jobs/strict_win_fixture_salvage_e66ce_v5_20260720/HASHES.json
  - reports/agent_jobs/strict_win_fixture_salvage_e66ce_v5_20260720/CODE_REVIEW.json
  - reports/agent_jobs/strict_win_fixture_salvage_e66ce_v5_20260720/guard-preflight.json
  - reports/agent_jobs/strict_win_fixture_salvage_e66ce_v5_20260720/status.json
  - reports/agent_jobs/strict_win_fixture_salvage_e66ce_v5_20260720/RUN_OUTCOME.json
  - reports/agent_jobs/strict_win_fixture_salvage_e66ce_v5_20260720/DECISION_ENTRY.json
approval_required: true
approval_source: Owner explicitly authorized the exact-base strict-WIN salvage lane on 2026-07-20.
allow_unapproved_safe_extension: false
timeout_seconds: 14400
output_dir: reports/agent_jobs/strict_win_fixture_salvage_e66ce_v5_20260720
mutation_mode: safe_extension
production_data_access: false
closeout_scope: repo_only
github_mutation_allowed: false
live_service_mutation_allowed: false
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: strict_win_fixture_salvage_e66ce_v5_20260720
proof_question: Does the salvaged strict WIN fixture builder and replay validator bind all derivation to one immutable byte buffer per file, recursively exclude outcomes, exactly reconcile one Sportsbet WIN race, reject ambiguous or malformed candidates, and enforce a closed manifest inventory on master commit e66ce849 including merged PR 55?
hypothesis_id: strict_win_fixture_accepted_failure_classes_e66ce_v5
program_track: offline_development
entry_state: owner_authorized_salvage_of_preserved_two_file_draft_from_55559f_onto_exact_master_e66ce849_with_pr55_merged
target_transition: one_clean_local_commit_parented_by_e66ce849_repairs_all_five_accepted_strict_win_fixture_failures
exit_predicate: Exactly one clean local commit parented by e66ce84982173a3a473db0d5f8e7655327014ff9 contains only the two authorized product files plus task and closeout evidence; all requested Python 3.11 and 3.13, deterministic identity, manifest/hash, regression, valid-fixture, PR 55 consumer, review, and V2 closeout gates pass; no GitHub, runtime, ingestion, database, captured-data, model, prediction, activation, betting, merge, or other-worktree mutation occurs.
source_class: owner_authorized_combined_comparison_from_98e363dd_to_preserved_worktree_55559f_plus_two_file_draft_applied_to_exact_master_e66ce849
dataset_version: strict_win_fixture_master_e66ce_pr55_merged_preserved_55559f_draft_20260720
evidence_hash: sha256:4190c50f43ad1389fb1ad129b029ad29281817bf66dc08584e214785825389cb
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: Resume only while the preserved worktree remains unmodified at tracked HEAD 55559f489836133365a28fdac4e84e2829e8b4d8 with the same two-file draft, PR 53 remains unchanged, no active claim overlaps the two product files, and any post-start master movement is disjoint from both files and their direct consumers.
---

# Strict WIN fixture salvage on master e66ce849

Salvage and review the preserved two-file strict-WIN draft by constructing one
combined comparison from old base `98e363dd9cc9950ac5d05f4d533df3f5e06f138e`
to the preserved working-tree contents, then applying only the resulting two
product files to exact base `e66ce84982173a3a473db0d5f8e7655327014ff9`.

The preserved worktree is read-only. PRs 53 and 56 are owner boundaries. PR 55
is merged base content and its direct consumers must remain intact and pass.
Runtime, ingestion, databases, captured data, models, prediction, activation,
betting, merge, push, and all GitHub mutation are `NOT_RUN`.
