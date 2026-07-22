---
job_id: REPAIR_PR63_NONFINITE_CONFIG_V1
title: Repair PR 63 non-finite JSON config acceptance
lane: Query Orchestration
supporting_lanes:
  - Runtime safety
  - Testing
  - Repo Hygiene
owner: Codex
approval_required: true
approval_source: >-
  The owner's 2026-07-22 /goal explicitly authorizes the smallest two-file
  repair for the independently reproduced non-finite config blocker at draft
  PR 63. After the original exact head changed, the owner's follow-up "its
  okay" explicitly authorized retargeting only this same repair to exact head
  728415ae2d66a5ba861983af7f76aa32a21a061f. The authorization includes focused
  validation, normal push to the existing PR branch, exact-head check waiting,
  normal task evidence, one decision append, and claim release. It forbids
  rebuilding or replacing the PR, force-push, merge, deploy, live prediction,
  runtime/data mutation, and adjacent hardening.
allow_unapproved_safe_extension: false
timeout_seconds: 43200
output_dir: reports/agent_jobs/REPAIR_PR63_NONFINITE_CONFIG_V1
mutation_mode: safe_extension
production_data_access: false
production_data_boundary: >-
  Exact-head source, canonical temporary JSON fixtures, focused tests, and
  report evidence only. No live races, network capture, production data or
  outputs, history/database writes, services, timers, daemons, models, EV,
  betting, deployment, activation, or merge.
github_mutation_allowed: true
git_history_mutation_allowed: true
live_service_mutation_allowed: false
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: REPAIR_PR63_NONFINITE_CONFIG_V1
proof_question: >-
  Does rejecting Python's non-standard NaN, Infinity, and -Infinity JSON
  constants at PR 63's config decoding boundary preserve all finite canonical
  config behavior while ensuring invalid config exits before lock waiting,
  capture, scoring, runtime mutation, or output creation?
hypothesis_id: pr63_nonfinite_json_constants_bypass_bounded_lock_wait_v1
program_track: offline_development
entry_state: >-
  origin/master is exact 585052ba7271f3a7e357dd5b69aec7f661591938;
  draft PR 63 is exact head 728415ae2d66a5ba861983af7f76aa32a21a061f,
  a direct child of independently reviewed head
  6a326ba564a20eb74c71ed89a5d9da095d04d3ef; the intervening packet-sealing
  commit does not modify src/predictor/on_demand.py. Independent review
  reproduced canonical lock_wait_seconds NaN acceptance and an unbounded
  busy-lock sleep path at the parent while all other bounded checks passed.
target_transition: >-
  one minimal in-place PR 63 repair rejects all non-finite JSON numeric
  constants through the existing invalid-config contract and a focused
  regression proves zero downstream side effects, without changing valid
  prediction modes or deterministic replay.
exit_predicate: >-
  The exact NaN reproduction is red before and green after; NaN, Infinity and
  -Infinity are rejected before sleep, scoring, capture/runtime mutation, or
  output writes; the focused PR 63 suite, smallest relevant PR 56 scorer and
  record regressions, both valid prediction modes, deterministic replay, Ruff,
  Python 3.11 compile, diff check and V2 guards pass; final diff is confined to
  the declared repair, focused regression, task card and normal report bundle;
  one normal fast-forward push updates existing draft PR 63; exact-head GitHub
  checks complete; no merge, live prediction, production/runtime/data action,
  model claim, market edge, profitability or production-readiness claim occurs.
source_class: >-
  exact_pr63_head_728415ae_direct_child_of_reviewed_6a326ba_plus_independent_review_603ccdc7_nonfinite_config_reproduction_fixture_only
dataset_version: >-
  pr63_728415ae_nonfinite_config_repair_v1_20260722
evidence_hash: sha256:f40a83ab68fa6936d2d48114bd0c04be336f08d4c86bd8841fd0ae8e156cd1b7
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: >-
  Continue only while origin/master remains exact
  585052ba7271f3a7e357dd5b69aec7f661591938, PR 63 remains open, draft,
  unmerged and at exact head 728415ae2d66a5ba861983af7f76aa32a21a061f
  until this run's own publication, no equivalent non-stale ownership or unsafe
  overlapping publication lane exists, and all changes stay inside this
  allowlist. Stop on ref, state, ownership, allowlist, validation, runtime/data,
  or publication drift.
docs_impact: DOCS_NOT_REQUIRED
docs_checked:
  - AGENTS.md
  - docs/on_demand_race_prediction.md
docs_changed: []
docs_followup: none
reason: >-
  Existing documentation already requires finite bounded config selection and
  fail-closed invalid config; this repairs the decoder to enforce that stated
  contract without changing supported syntax or operator steps.
task_tier: medium
recommended_model: standard_coding
actual_model: Codex GPT-5
why_this_model: >-
  The product delta is a standard-library decoder hook plus one focused test,
  while exact-head V2 publication and no-side-effect proof require careful
  coordination.
worker_model_allowed: false
worker_decision_limit: >-
  No delegation; the repair and regression share one narrow two-file seam and
  repository instructions make subagents opt-in.
escalation_needed: false
allowed_files:
  - docs/agent_tasks/repair_pr63_nonfinite_config_v1_20260722.md
  - src/predictor/on_demand.py
  - tests/test_predict_race_now.py
  - reports/agent_jobs/REPAIR_PR63_NONFINITE_CONFIG_V1/README.md
  - reports/agent_jobs/REPAIR_PR63_NONFINITE_CONFIG_V1/STATE.md
  - reports/agent_jobs/REPAIR_PR63_NONFINITE_CONFIG_V1/DECISIONS.md
  - reports/agent_jobs/REPAIR_PR63_NONFINITE_CONFIG_V1/VALIDATION.md
  - reports/agent_jobs/REPAIR_PR63_NONFINITE_CONFIG_V1/CODE_REVIEW.json
  - reports/agent_jobs/REPAIR_PR63_NONFINITE_CONFIG_V1/RUN_OUTCOME.json
  - reports/agent_jobs/REPAIR_PR63_NONFINITE_CONFIG_V1/DECISION_ENTRY.json
  - reports/agent_jobs/REPAIR_PR63_NONFINITE_CONFIG_V1/status.json
  - reports/agent_jobs/REPAIR_PR63_NONFINITE_CONFIG_V1/guard-preflight.json
  - reports/agent_jobs/REPAIR_PR63_NONFINITE_CONFIG_V1/guard-final.json
  - reports/agent_jobs/REPAIR_PR63_NONFINITE_CONFIG_V1/diff-check.json
  - reports/agent_jobs/REPAIR_PR63_NONFINITE_CONFIG_V1/reproduction-before.json
  - reports/agent_jobs/REPAIR_PR63_NONFINITE_CONFIG_V1/reproduction-after.json
  - reports/agent_jobs/REPAIR_PR63_NONFINITE_CONFIG_V1/remote-pr.json
  - reports/agent_jobs/REPAIR_PR63_NONFINITE_CONFIG_V1/check-runs.json
  - reports/agent_jobs/REPAIR_PR63_NONFINITE_CONFIG_V1/focused-pytest.log
  - reports/agent_jobs/REPAIR_PR63_NONFINITE_CONFIG_V1/pr56-regressions.log
  - reports/agent_jobs/REPAIR_PR63_NONFINITE_CONFIG_V1/ruff-check.log
  - reports/agent_jobs/REPAIR_PR63_NONFINITE_CONFIG_V1/ruff-format.log
  - reports/agent_jobs/REPAIR_PR63_NONFINITE_CONFIG_V1/compile.log
  - reports/agent_jobs/REPAIR_PR63_NONFINITE_CONFIG_V1/commands.log
  - reports/agent_jobs/REPAIR_PR63_NONFINITE_CONFIG_V1/release-receipt.json
---

# Repair PR 63 non-finite config acceptance

Reject Python-specific non-finite JSON numeric constants at the config decoding
boundary, then prove the existing invalid-config contract prevents all lock,
capture, scoring, mutation, and output behavior. Preserve every valid finite
configuration path and leave draft PR 63 unmerged.
