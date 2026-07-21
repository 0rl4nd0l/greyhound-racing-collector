---
job_id: PR56_TIME_ONLY_SECONDS_REPAIR_V1_20260721
title: Repair supported time-only seconds handling in PR 56 manual scorer
lane: Provenance
supporting_lanes:
  - Testing
  - Repo Hygiene
  - Reporting
owner: Codex
approval_required: true
approval_source: >-
  The owner's 2026-07-21 instruction "Proceed with repair" authorizes the
  smallest local source and regression-test repair for the independently
  reproduced supported HH:MM:SS time-only alias rejection at exact PR 56 head
  f6e5a14fb55fc9a5b44902f8e5163be4c7f43b16. It does not authorize commit,
  push, PR metadata/readiness changes, or merge.
allow_unapproved_safe_extension: false
timeout_seconds: 43200
output_dir: reports/agent_jobs/PR56_TIME_ONLY_SECONDS_REPAIR_V1_20260721
mutation_mode: safe_extension
production_data_access: false
production_data_boundary: >-
  Exact-head source, focused outcome-free temporary fixtures, tests, frozen
  artifact byte comparisons, and local report evidence only. No live races,
  outcomes, refit, production database/history writes, services, timers,
  deployment, activation, EV, betting, model promotion, or other-PR mutation.
github_mutation_allowed: false
git_history_mutation_allowed: false
live_service_mutation_allowed: false
closeout_scope: report_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: PR56_TIME_ONLY_SECONDS_REPAIR_V1_20260721
proof_question: >-
  Does the manual scorer accept contract-supported matching HH:MM:SS time-only
  jump aliases, preserve their seconds when normalizing against the agreed race
  date and Melbourne timezone, and continue to reject conflicting or invalid
  aliases before score_race or any side effect?
hypothesis_id: pr56_time_only_seconds_supported_contract_repair_v1_20260721
program_track: offline_development
entry_state: >-
  Draft PR 56 is open, draft, clean and unmerged at exact head f6e5a14f on
  exact base fadcd19a with five green checks. Independent review proved the
  existing pre-jump sidecar contract accepts 18:58:30 while the manual scorer
  raises jump_time_invalid before scoring because its time-only format list
  omits %H:%M:%S.
target_transition: >-
  One minimal uncommitted local repair adds the already-supported %H:%M:%S
  format at the manual scorer seam and one focused regression proves matching
  seconds-resolution aliases score normally while disagreement remains
  fail-closed with no side effects. PR 56 remains unchanged, draft and
  unmerged for separately authorized publication and independent review.
exit_predicate: >-
  The exact supported-path case is red before repair and green after repair;
  focused scorer and strict-WIN suites, Ruff check/format, compile, diff and
  no-write guards pass; frozen model, manifest and fit-population bytes remain
  identical to master; canonical output remains byte-identical to f6e5 at the
  same fixture path; the final diff is limited to this task card, scorer and
  focused test plus allowed report evidence; live head/base/state are unchanged;
  and no GitHub, runtime, database, history, model or service mutation occurs.
source_class: >-
  exact_remote_pr56_head_f6e5a14_plus_supported_prejump_sidecar_contract_hhmmss_reproduction_and_owner_repair_instruction
dataset_version: pr56_f6e5a14_time_only_seconds_repair_v1_20260721
evidence_hash: sha256:58833ec21dc51c0cca0d7cbc41261f229d15056e3baa6aa8149b39edab8444b8
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
resume_only_if: >-
  Continue only while origin/master is exactly
  fadcd19a83b6c8b2f26a2344431546ff2016ff1d, remote PR 56 head is exactly
  f6e5a14fb55fc9a5b44902f8e5163be4c7f43b16, PR 56 remains open, draft,
  clean/mergeable and unmerged, no non-stale overlapping claim exists, and the
  repair needs only the declared files. Stop on drift, disallowed-path need,
  validation failure, or any publish/runtime/data boundary.
docs_impact: DOCS_NOT_REQUIRED
docs_checked:
  - AGENTS.md
  - docs/manual_live_market_form_residual_prediction.md
docs_changed: []
docs_followup: none
reason: >-
  The existing operator contract already permits supported timestamp forms and
  requires fail-closed temporal agreement; this aligns one parser format with
  the established sidecar contract without changing syntax, schema or steps.
task_tier: medium
recommended_model: standard_coding
actual_model: Codex GPT-5
why_this_model: >-
  The code change is one parser-format addition plus a focused regression, but
  exact-head provenance and no-side-effect validation remain material.
worker_model_allowed: false
worker_decision_limit: >-
  No delegation; the repair and regression share one narrow two-file seam.
escalation_needed: false
allowed_files:
  - docs/agent_tasks/pr56_time_only_seconds_repair_v1_20260721.md
  - scripts/predict_market_form_residual.py
  - tests/test_predict_market_form_residual.py
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_REPAIR_V1_20260721/STATE.md
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_REPAIR_V1_20260721/DECISIONS.md
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_REPAIR_V1_20260721/VALIDATION.md
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_REPAIR_V1_20260721/RUN_OUTCOME.json
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_REPAIR_V1_20260721/DECISION_ENTRY.json
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_REPAIR_V1_20260721/status.json
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_REPAIR_V1_20260721/guard-preflight.json
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_REPAIR_V1_20260721/guard-final.json
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_REPAIR_V1_20260721/diff-check.json
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_REPAIR_V1_20260721/release-receipt.json
---

# Repair PR 56 supported time-only seconds handling

Align the manual scorer's time-only alias parser with the existing pre-jump
sidecar contract by accepting seconds-resolution `HH:MM:SS` values. Preserve
all existing key-presence, independent validation, canonical-instant agreement,
pre-jump ordering and no-write behavior.
