---
job_id: PR56_RACE_IDENTITY_AGREEMENT_REPAIR_V2_20260720
title: Repair duplicate race identity agreement in draft PR 56
lane: Provenance
supporting_lanes:
  - Testing
  - Repo Hygiene
  - Reporting
owner: Codex
approval_required: true
approval_source: >-
  The owner's 2026-07-20 /goal REPAIR_PR56_RACE_IDENTITY_AGREEMENT and
  subsequent explicit approval authorize one normal merge child of exact PR 56
  head 0470edabf1a4fdf85922c88e327899e3621bcca4 with current master
  e66ce84982173a3a473db0d5f8e7655327014ff9 as second parent. The original goal
  governs conflict resolution: retain PR 56's read-only, no-history-writer CLI
  boundary while leaving master's scorer/writer implementation untouched.
allow_unapproved_safe_extension: false
timeout_seconds: 43200
output_dir: reports/agent_jobs/PR56_RACE_IDENTITY_AGREEMENT_REPAIR_V2_20260720
mutation_mode: safe_extension
production_data_access: false
production_data_boundary: >-
  Source code, focused tests, tracked frozen artifacts for byte comparison, and
  outcome-free temporary fixtures only. No outcomes, fitting, production
  SQLite/history, migration, services, timers, deployment, activation, EV,
  betting, live-race attempts, or PR 51 sealed trainer/control data.
github_mutation_allowed: true
git_history_mutation_allowed: true
live_service_mutation_allowed: false
closeout_scope: report_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: PR56_RACE_IDENTITY_AGREEMENT_REPAIR_V2_20260720
proof_question: >-
  Does one merge child reconcile master and PR 56 while retaining the PR 56
  no-history-writer boundary and requiring independently parsed duplicate date,
  race number, and authoritative distance fields to normalize and agree?
hypothesis_id: pr56_duplicate_race_identity_agreement_no_writer_v2_20260720
program_track: offline_development
entry_state: >-
  Draft PR 56 is open and unmerged at 0470edab on merge base 98e363dd; master
  advanced through merged PR 55 to e66ce849, leaving PR 56 one ahead, three
  behind, and conflict-dirty on the scorer and tests. Exact review reproductions
  prove race_info-only date and race-number conflicts are silently accepted.
target_transition: >-
  one normal merge child retains PR 56's read-only CLI, rejects duplicate
  identity disagreement or invalid one-sided values without side effects,
  passes all requested validation, and leaves PR 56 draft, open, unmerged and
  CI-green.
exit_predicate: >-
  Focused regressions prove date and race-number conflict in both directions,
  invalid one-sided values, normalized equivalence, canonical output stability,
  and rejection no-write; adjacent venue and grade remain fail-closed and
  duplicate distance is repaired by the same agreement rule; record V3,
  effective-state V2, canonical JSON/runner order, provenance/hash binding,
  outcome quarantine, PR 51 isolation, read-only operation and no history-writer
  calls remain intact; requested focused, manual-handoff, collector/grade, PR 46,
  feature/resource, Ruff, compile, diff, task, no-write, deterministic-output
  and frozen-hash checks pass; exact refs are rechecked; exactly one normal merge
  child is pushed to the existing PR 56 branch; all GitHub checks finish green;
  PR 56 remains draft and unmerged.
source_class: >-
  exact_remote_pr56_head_0470edab_plus_current_master_e66ce849_plus_owner_review_reproductions_plus_no_writer_contract_correction
dataset_version: pr56_0470edab_master_e66ce849_duplicate_identity_no_writer_v2_20260720
evidence_hash: sha256:cdab403ae37486b8206f2daa8bd79a7d461ee437a9af72eacfa8871b805f53e2
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: >-
  Before publication, continue only while origin/master is exactly
  e66ce84982173a3a473db0d5f8e7655327014ff9 and remote PR 56 is exactly
  0470edabf1a4fdf85922c88e327899e3621bcca4, open, draft and unmerged. After
  publication, the remote head must be exactly the one locally validated merge
  child with those two parents. Abort on further ref drift, disallowed-path need,
  failed unchanged-master comparison, validation/review failure, force-push need,
  history-writer call, or any outcome, production, runtime, PR 51, deployment or
  activation access.
docs_impact: DOCS_NOT_REQUIRED
docs_checked:
  - AGENTS.md
  - docs/manual_live_market_form_residual_prediction.md
docs_changed: []
docs_followup: none
reason: >-
  Existing PR 56 documentation already requires exact race identity and
  fail-closed read-only behavior; this correction enforces that contract for
  duplicate artifact fields without changing syntax or operator steps.
task_tier: large
recommended_model: high_reasoning
actual_model: Codex GPT-5
why_this_model: >-
  The narrow validation change must resolve a same-file merge conflict without
  importing a disallowed writer caller and must pass exact-head publication.
worker_model_allowed: false
worker_decision_limit: >-
  No delegation; repository instructions make subagents opt-in and this repair
  remains within one tightly coupled scorer/test seam.
escalation_needed: false
allowed_files:
  - docs/agent_tasks/pr56_race_identity_agreement_repair_v2_20260720.md
  - scripts/predict_market_form_residual.py
  - tests/test_predict_market_form_residual.py
  - reports/agent_jobs/PR56_RACE_IDENTITY_AGREEMENT_REPAIR_V2_20260720/README.md
  - reports/agent_jobs/PR56_RACE_IDENTITY_AGREEMENT_REPAIR_V2_20260720/STATE.md
  - reports/agent_jobs/PR56_RACE_IDENTITY_AGREEMENT_REPAIR_V2_20260720/VALIDATION.md
  - reports/agent_jobs/PR56_RACE_IDENTITY_AGREEMENT_REPAIR_V2_20260720/CODE_REVIEW.json
  - reports/agent_jobs/PR56_RACE_IDENTITY_AGREEMENT_REPAIR_V2_20260720/RUN_OUTCOME.json
  - reports/agent_jobs/PR56_RACE_IDENTITY_AGREEMENT_REPAIR_V2_20260720/DECISION_ENTRY.json
  - reports/agent_jobs/PR56_RACE_IDENTITY_AGREEMENT_REPAIR_V2_20260720/status.json
  - reports/agent_jobs/PR56_RACE_IDENTITY_AGREEMENT_REPAIR_V2_20260720/guard-preflight.json
  - reports/agent_jobs/PR56_RACE_IDENTITY_AGREEMENT_REPAIR_V2_20260720/guard-final.json
  - reports/agent_jobs/PR56_RACE_IDENTITY_AGREEMENT_REPAIR_V2_20260720/diff-check.json
  - reports/agent_jobs/PR56_RACE_IDENTITY_AGREEMENT_REPAIR_V2_20260720/remote-pr.json
  - reports/agent_jobs/PR56_RACE_IDENTITY_AGREEMENT_REPAIR_V2_20260720/github-checks.json
  - reports/agent_jobs/PR56_RACE_IDENTITY_AGREEMENT_REPAIR_V2_20260720/final-refs.json
  - reports/agent_jobs/PR56_RACE_IDENTITY_AGREEMENT_REPAIR_V2_20260720/release-receipt.json
---

# Repair PR 56 duplicate race identity agreement

Reconcile current master into draft PR 56 in one normal merge child while
retaining PR 56's read-only, no-history-writer CLI. Independently parse every
supplied duplicate date, race number, and authoritative distance field, then
fail closed unless their normalized values agree. Preserve all other behavior.
