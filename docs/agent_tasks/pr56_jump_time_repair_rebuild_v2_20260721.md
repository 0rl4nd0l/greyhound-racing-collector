---
job_id: PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721
title: Rebuild and publish PR 56 jump-time agreement repair
lane: Provenance
supporting_lanes:
  - Testing
  - Repo Hygiene
  - Reporting
owner: Codex
approval_required: true
approval_source: >-
  The owner's 2026-07-21 /goal REBUILD_AND_PUBLISH_PR56_JUMP_TIME_REPAIR
  authorizes a fresh worktree, one ordered-parent integration commit from
  accepted PR 56 head 3858527fcd63f7235f918ecc4ed2b8e2c4dc58fd and fetched
  master fadcd19a83b6c8b2f26a2344431546ff2016ff1d, one actual scorer repair
  child, one exact force-with-lease replacement of wrong remote head
  77019c1881ea9d27bb6b3ad26c74f660f0e722d9, exact-head CI inspection, V2
  closeout, and claim release.
allow_unapproved_safe_extension: false
timeout_seconds: 43200
output_dir: reports/agent_jobs/PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721
mutation_mode: safe_extension
production_data_access: false
production_data_boundary: >-
  Source code, focused tests, tracked frozen artifacts for byte comparison,
  outcome-free temporary fixtures, report artifacts, and the exact PR 56
  branch ref only. No live races, outcomes, refit, production writes, services,
  deployment, betting, model promotion, PR metadata/readiness/merge changes,
  or other-PR mutation.
github_mutation_allowed: true
git_history_mutation_allowed: true
live_service_mutation_allowed: false
closeout_scope: repo_and_publish
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721
proof_question: >-
  Does the integrated PR 56 manual scorer determine supplied jump-time aliases
  by key presence, independently parse every reachable full-datetime and
  time-only value, normalize time-only values to the agreed race date and
  timezone, require all canonical instants to agree before the pre-jump gate,
  and reject invalid or conflicting inputs before score_race or any side effect?
hypothesis_id: pr56_manual_scorer_jump_time_presence_and_agreement_rebuild_v2_20260721
program_track: offline_development
entry_state: >-
  Draft PR 56 is open, draft, clean/mergeable and unmerged at wrong remote head
  77019c1881ea9d27bb6b3ad26c74f660f0e722d9, whose sole parent is accepted
  head 3858527fcd63f7235f918ecc4ed2b8e2c4dc58fd and whose delta is limited to
  two daily-race orchestrator files. Fetched master is
  fadcd19a83b6c8b2f26a2344431546ff2016ff1d; its drift from 95ec4fb is the six
  disjoint control-plane and strict-WIN paths declared below. Candidate
  7b77049d is abandoned and contains only a task card.
target_transition: >-
  An ordered-parent integration commit preserves accepted PR 56 behavior while
  incorporating fetched master, and one real repair child fixes only the manual
  scorer jump-time agreement seam plus focused tests and required V2 artifacts;
  that repair child replaces exact wrong remote head 77019c once and leaves PR
  56 draft, open, unmerged and ready only for focused independent review.
exit_predicate: >-
  The exact 6:58 PM versus 6:51 PM at 6:52 PM failure is reproduced through
  scripts/predict_market_form_residual.py before repair; all reachable manual
  CLI full-datetime and time-only jump fields are inventoried; supplied values
  are detected by key existence and independently validated; invalid, empty,
  null, boolean, malformed, conflicting, reversed-conflict, equivalent 12/24
  hour, timezone-equivalent and absent-alias cases are covered; every rejection
  proves score_calls zero and no output, history or database effects; focused
  scorer and strict-WIN suites, Ruff check/format, compile, diff/task/no-write
  guards, frozen hashes and canonical-output SHA pass; formal code review is
  clean; exactly the integration commit plus repair child exist; the wrong
  orchestrator delta is absent; remote head is re-proved as 77019c immediately
  before one force-with-lease; exact-head CI passes; V2 release succeeds; and PR
  56 remains draft, open and unmerged.
source_class: >-
  exact_remote_pr56_wrong_head_77019c_plus_accepted_head_3858527f_plus_fetched_master_fadcd19_plus_owner_supplied_three_alias_reproduction_and_reachable_manual_scorer_paths
dataset_version: pr56_3858527f_master_fadcd19_jump_time_repair_rebuild_v2_20260721
evidence_hash: sha256:811d22e09f294f8a89d1e2c214d58df9b4aa003ed4f2fa7eb833d806d912820a
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: >-
  Before publication, continue only while fetched origin/master is exactly
  fadcd19a83b6c8b2f26a2344431546ff2016ff1d or has moved solely through paths
  disjoint from scripts/predict_market_form_residual.py and
  tests/test_predict_market_form_residual.py, remote PR 56 head remains exactly
  77019c1881ea9d27bb6b3ad26c74f660f0e722d9, PR 56 remains open, draft,
  mergeable and unmerged, no overlapping claim or collaborator appears, and no
  relevant path overlap appears. Never weaken or retry a failed lease. After
  publication, continue only while remote PR 56 head is the exact locally
  validated repair child and the PR remains open, draft and unmerged.
docs_impact: DOCS_NOT_REQUIRED
docs_checked:
  - AGENTS.md
  - docs/manual_live_market_form_residual_prediction.md
docs_changed: []
docs_followup: none
reason: >-
  Existing documentation already requires fail-closed temporal identity and
  read-only behavior; the repair enforces that contract for reachable manual
  CLI aliases without changing operator syntax, schema, or steps.
task_tier: critical
recommended_model: high_reasoning
actual_model: Codex GPT-5
why_this_model: >-
  The code seam is narrow, but temporal normalization, ordered-parent merge,
  exact force-with-lease publication, and V2 closeout require strict reasoning.
worker_model_allowed: false
worker_decision_limit: >-
  No delegation; repository instructions make subagents opt-in and this is one
  tightly coupled scorer/test, topology, validation and publication lane.
escalation_needed: false
allowed_files:
  - .codex/hooks.json
  - AGENTS.md
  - docs/agent_tasks/strict_win_capture_plan_compat_repair_v1_20260721.md
  - docs/agent_tasks/strict_win_nonready_identity_validation_repair_v1_20260721.md
  - scripts/strict_win_odds_fixture_capture.py
  - tests/test_strict_win_odds_fixture_capture.py
  - docs/agent_tasks/pr56_jump_time_repair_rebuild_v2_20260721.md
  - scripts/predict_market_form_residual.py
  - tests/test_predict_market_form_residual.py
  - reports/agent_jobs/PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721/README.md
  - reports/agent_jobs/PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721/STATE.md
  - reports/agent_jobs/PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721/DECISIONS.md
  - reports/agent_jobs/PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721/REPRODUCTION.md
  - reports/agent_jobs/PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721/FIELD_MATRIX.md
  - reports/agent_jobs/PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721/VALIDATION.md
  - reports/agent_jobs/PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721/COMMANDS.md
  - reports/agent_jobs/PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721/CODE_REVIEW.json
  - reports/agent_jobs/PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721/HASH_PROOF.json
  - reports/agent_jobs/PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721/RUN_OUTCOME.json
  - reports/agent_jobs/PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721/DECISION_ENTRY.json
  - reports/agent_jobs/PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721/status.json
  - reports/agent_jobs/PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721/guard-preflight.json
  - reports/agent_jobs/PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721/guard-final.json
  - reports/agent_jobs/PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721/diff-check.json
  - reports/agent_jobs/PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721/remote-pr.json
  - reports/agent_jobs/PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721/github-checks.json
  - reports/agent_jobs/PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721/final-refs.json
  - reports/agent_jobs/PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721/WAIT_RESULT.json
  - reports/agent_jobs/PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721/WAIT_RESULT.json.log
  - reports/agent_jobs/PR56_JUMP_TIME_REPAIR_REBUILD_V2_20260721/release-receipt.json
---

# Rebuild PR 56 jump-time agreement repair

Integrate fetched master with accepted PR 56 behavior using ordered parents,
then make one real repair child at the manual scorer seam. Exclude the wrong
daily-race orchestrator delta, publish exactly once under the specified lease,
and leave the PR draft and unmerged for focused independent review.
