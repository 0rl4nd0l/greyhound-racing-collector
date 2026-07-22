---
job_id: REFRESH_PR53_AFTER_PR56_MERGE_V2
title: Refresh the PR 53 on-demand predictor after PR 56 merge
lane: Query Orchestration
supporting_lanes:
  - Provenance
  - Runtime safety
  - Testing
owner: Codex
approval_required: true
approval_source: The owner's 2026-07-22 /goal explicitly authorizes an isolated V2 worktree from exact master 585052ba7271f3a7e357dd5b69aec7f661591938, a minimal transplant from stale draft PR 53 at 2deb5aec454fb9314a22f30a30169aa05b2261c5, validation, normal push, and one clean draft successor PR. It forbids rebuilding, merging, deploying, activation, betting, production persistence, production database writes, and mutation of PRs 47, 48, 52, 53, or 54.
allow_unapproved_safe_extension: false
timeout_seconds: 43200
output_dir: reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2
mutation_mode: safe_extension
production_data_access: false
production_data_boundary: Implementation and validation use repository fixtures, temporary isolated bundles, and read-only source inspection. An optional live proof may only read an eligible pre-jump race and write an isolated non-persisting bundle; it may not read the target outcome, write any database or production path, alter locks, or mutate services, timers, daemons, models, registries outside the V2 claim, or existing PRs.
owner_db_append_only_approval: false
github_mutation_allowed: true
git_history_mutation_allowed: true
live_service_mutation_allowed: false
allowed_files:
  - docs/agent_tasks/refresh_pr53_after_pr56_merge_v2_20260722.md
  - scripts/predict_race_now.py
  - src/predictor/on_demand.py
  - configs/prediction/manual-default.json
  - configs/prediction/market-only.json
  - configs/prediction/schemas/market_only_v1.schema.json
  - configs/prediction/schemas/market_form_residual_v1.schema.json
  - tests/test_predict_race_now.py
  - docs/on_demand_race_prediction.md
  - reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2/README.md
  - reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2/STATE.md
  - reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2/DECISIONS.md
  - reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2/VALIDATION.md
  - reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2/CODE_REVIEW.md
  - reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2/RUN_OUTCOME.json
  - reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2/DECISION_ENTRY.json
  - reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2/status.json
  - reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2/guard-preflight.json
  - reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2/inventory.json
  - reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2/ancestry.json
  - reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2/fixture-proofs.json
  - reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2/live-proof.json
  - reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2/diff-check.json
  - reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2/pr-body.md
  - reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2/remote-pr.json
  - reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2/check-runs.json
  - reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2/focused-pytest.log
  - reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2/combined-pytest.log
  - reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2/ruff-check.log
  - reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2/ruff-format.log
  - reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2/compile.log
  - reports/agent_jobs/REFRESH_PR53_AFTER_PR56_MERGE_V2/commands.log
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: REFRESH_PR53_AFTER_PR56_MERGE_V2
proof_question: Can the eight-file on-demand predictor delta from stale draft PR 53 be transplanted onto exact master after PR 56 while consuming master's race identity, grade, jump-time, runner, provenance, record-V3, and effective-state-V2 contracts and preserving one immediate research-only command with deterministic finite config selection, isolated sealed outputs, and zero production writes?
hypothesis_id: refresh_pr53_on_demand_predictor_after_pr56_v2
program_track: offline_development
entry_state: master_585052ba_contains_pr56_handoff_and_stale_pr53_2deb5aec_remains_open_draft_unmerged_with_obsolete_pr46_block
target_transition: one_clean_draft_successor_transplants_only_the_required_on_demand_predictor_delta_onto_master_585052ba
exit_predicate: Exact master and stale PR 53 identities remain unchanged; the eight-file PR 53 implementation delta is classified and transplanted without copying stale master-owned handoff, scorer, writer, daemon, evaluation, task, or report files; the command lists configs and runs one named race immediately with explicit config; market-only and market-form-residual modes, deterministic replay, verified receipt reuse, isolated capture, tamper and mismatch rejection, target-outcome exclusion, concurrency cleanup, and zero unintended writes pass; current PR 56 integration regressions, Ruff, compile, diff and V2 guards pass; one normal-push draft successor is opened; no old PR, live service, timer, daemon, production database, production model, betting, deployment, activation, or target outcome is mutated.
source_class: exact_master_585052ba_plus_stale_pr53_2deb5aec_eight_file_delta_plus_fixture_only_and_optional_read_only_prejump_validation
dataset_version: master_585052ba7271f3a7e357dd5b69aec7f661591938_pr53_2deb5aec454fb9314a22f30a30169aa05b2261c5_20260722
evidence_hash: sha256:23cfa8c2e09749974011308b5f594678d6f466063deaa454398752e00c96db73
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: Resume only while origin/master remains 585052ba7271f3a7e357dd5b69aec7f661591938, PR 53 remains open, draft, unmerged, and exactly 2deb5aec454fb9314a22f30a30169aa05b2261c5, no non-stale active claim overlaps the exact allowed paths, and all implementation and publication changes remain inside this allowlist. Stop on ref, state, ancestry, identity, evidence, allowlist, validation, production-boundary, or existing-PR drift.
docs_impact: DOCS_UPDATE_REQUIRED
docs_checked:
  - AGENTS.md
  - docs/on_demand_race_prediction.md at PR 53
  - scripts/predict_race_now.py at PR 53
  - scripts/predict_market_form_residual.py at master
docs_changed:
  - docs/on_demand_race_prediction.md
docs_followup: none
reason: The successor must document the immediate repo-root command, finite config selection, receipt reuse, isolated capture and bundle behavior, PR 56 integration, and non-interference boundaries.
---

# Refresh PR 53 after PR 56 merge V2

Transplant only the verified eight-file on-demand predictor delta from stale
draft PR 53 onto exact master after PR 56. Consume all master-owned scorer,
handoff, capture, provenance, and browser paths in place. Publish only one new
draft successor after validation; leave every old PR and all production/runtime
surfaces untouched.
