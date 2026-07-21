---
job_id: PR56_TIME_ONLY_SECONDS_PUBLICATION_V1_20260722
title: Publish and independently review PR 56 time-only seconds repair
lane: Provenance
supporting_lanes:
  - Testing
  - Repo Hygiene
  - Reporting
owner: Codex
approval_required: true
approval_source: >-
  The owner's 2026-07-22 "Approve" follows the stated request for approval to
  commit and publish the exact locally validated PR 56 time-only-seconds repair,
  wait for exact-head CI, and perform a fresh independent review. It authorizes
  one local commit and one normal fast-forward push to the existing PR 56 head
  branch. It does not authorize force-push, PR metadata/readiness changes,
  merge, runtime, data, model, service, deployment, EV or betting actions.
allow_unapproved_safe_extension: false
timeout_seconds: 43200
output_dir: reports/agent_jobs/PR56_TIME_ONLY_SECONDS_PUBLICATION_V1_20260722
mutation_mode: safe_extension
production_data_access: false
production_data_boundary: >-
  Exact reviewed source and tests, task cards, outcome-free temporary fixtures,
  frozen artifact byte comparisons, report evidence, and the existing PR 56
  branch ref only. No live races, outcomes, refit, production database/history
  writes, services, timers, deployment, activation, EV, betting, model promotion,
  adjacent-PR mutation, PR readiness change, or merge.
github_mutation_allowed: true
git_history_mutation_allowed: true
live_service_mutation_allowed: false
closeout_scope: repo_and_publish
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: PR56_TIME_ONLY_SECONDS_PUBLICATION_V1_20260722
proof_question: >-
  Can the exact locally validated HH:MM:SS manual jump-time repair be committed
  as one child of f6e5a14, normally fast-forwarded to existing draft PR 56,
  pass exact-head CI, and pass a fresh independent read-only review without
  changing any other PR, runtime, data, frozen artifact, or workflow boundary?
hypothesis_id: pr56_time_only_seconds_exact_candidate_publication_v1_20260722
program_track: offline_development
entry_state: >-
  Origin master is exactly fadcd19a83b6c8b2f26a2344431546ff2016ff1d and
  draft PR 56 is open, clean/mergeable and unmerged at exact head
  f6e5a14fb55fc9a5b44902f8e5163be4c7f43b16. The released repair task has one
  validated uncommitted source-line addition, two focused test additions, and
  its task card. Its full scorer and strict-WIN suites, static guards, frozen
  hashes, canonical output SHA, and formal review passed. The only active
  registry claim is a stale, disjoint PR 51 audit claim.
target_transition: >-
  One normal child of f6e5a14 records only both task cards, the one-line scorer
  repair and focused regressions; one non-force push advances only the existing
  PR 56 branch to that child; exact-head CI and a fresh detached-worktree review
  pass; and PR 56 remains open, draft and unmerged for separate merge approval.
exit_predicate: >-
  The pre-commit diff exactly matches the released repair plus this publication
  card; focused scorer and strict-WIN suites, Ruff, compile, diff, task-card and
  no-write guards pass; the commit has sole parent f6e5a14 and contains no
  unrelated paths; master, remote head and PR state are rechecked immediately
  before a normal fast-forward push; all exact-head CI jobs finish green; a new
  detached worktree reproduces matching and conflicting seconds behavior and
  verifies hashes, canonical output, topology, side-effect ordering and PR 51/
  frozen boundaries; the claim releases once; and no force, metadata, readiness,
  merge, runtime, database, history, model, service or other-PR mutation occurs.
source_class: >-
  exact_remote_pr56_head_f6e5a14_plus_released_local_time_only_seconds_repair_and_owner_publication_approval
dataset_version: pr56_f6e5a14_time_only_seconds_publication_v1_20260722
evidence_hash: sha256:5bf83032e3d2d9f555c9e3e33c1a75100a7b53a83919ecd73c1ed69a7f761912
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: >-
  Before publication, continue only while origin/master is exactly
  fadcd19a83b6c8b2f26a2344431546ff2016ff1d, remote PR 56 head is exactly
  f6e5a14fb55fc9a5b44902f8e5163be4c7f43b16, PR 56 remains open, draft,
  clean/mergeable and unmerged, the candidate diff remains exact, and no
  non-stale overlapping claim exists. Never force or retry a failed/non-fast-
  forward push. After publication, continue only while the remote PR head is
  the exact local child and the PR remains open, draft and unmerged. Stop on
  any drift, validation/review defect, disallowed-path need, or boundary breach.
docs_impact: DOCS_NOT_REQUIRED
docs_checked:
  - AGENTS.md
  - docs/manual_live_market_form_residual_prediction.md
docs_changed: []
docs_followup: none
reason: >-
  The approved source change aligns the manual scorer with the existing
  supported timestamp contract; this card records only the separately approved
  publication and exact-head review transition.
task_tier: medium
recommended_model: standard_coding
actual_model: Codex GPT-5
why_this_model: >-
  The code change is minimal, while exact-head publication and provenance gates
  require careful state verification.
worker_model_allowed: false
worker_decision_limit: >-
  No delegation; repository instructions make subagents opt-in and this is one
  tightly coupled publication and review lane.
escalation_needed: false
allowed_files:
  - docs/agent_tasks/pr56_time_only_seconds_repair_v1_20260721.md
  - docs/agent_tasks/pr56_time_only_seconds_publication_v1_20260722.md
  - scripts/predict_market_form_residual.py
  - tests/test_predict_market_form_residual.py
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_PUBLICATION_V1_20260722/STATE.md
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_PUBLICATION_V1_20260722/DECISIONS.md
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_PUBLICATION_V1_20260722/VALIDATION.md
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_PUBLICATION_V1_20260722/REVIEW.md
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_PUBLICATION_V1_20260722/RUN_OUTCOME.json
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_PUBLICATION_V1_20260722/DECISION_ENTRY.json
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_PUBLICATION_V1_20260722/status.json
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_PUBLICATION_V1_20260722/guard-preflight.json
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_PUBLICATION_V1_20260722/guard-final.json
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_PUBLICATION_V1_20260722/diff-check.json
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_PUBLICATION_V1_20260722/remote-pr.json
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_PUBLICATION_V1_20260722/github-checks.json
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_PUBLICATION_V1_20260722/final-refs.json
  - reports/agent_jobs/PR56_TIME_ONLY_SECONDS_PUBLICATION_V1_20260722/release-receipt.json
---

# Publish PR 56 time-only seconds repair

Commit and normally publish the exact released local repair to the existing
draft PR 56 branch. Wait for exact-head CI, then repeat the focused review in a
fresh detached worktree. Leave the PR draft and unmerged for separate approval.
