---
job_id: GHU_050_MANUAL_INDEPENDENT_CAPTURE_V1_20260804
title: Specify manual-independent-capture-v1
lane: Provenance
supporting_lanes:
  - Runtime Safety
  - Testing
  - Repo Hygiene
owner: Codex
approval_required: true
approval_source: "Owner /goal request on 2026-08-04 authorizes GHU-050-only repository implementation, validation, exact-diff review, commit, push, focused PR, required exact-head CI, review corrections, and merge under existing authority."
allow_unapproved_safe_extension: false
timeout_seconds: 28800
mutation_mode: safe_extension
base: c20932008edaa02f602733253165f2cd7845a2a3
base_tree: c4b5fc900e1a347c6fe0c889d3b300c7df8d2922
branch: agent/ghu-050-manual-independent-contract
production_data_access: false
production_data_boundary: "No browser, network, capture, scoring, canonical database/history/live_odds, forward corpus, collector request/state, result evidence, shared lock, service, timer, deployment, training, promotion, EV, staking, or betting access or mutation."
github_mutation_allowed: true
git_history_mutation_allowed: true
live_service_mutation_allowed: false
closeout_scope: repo_and_pr
docs_impact: DOCS_REQUIRED
task_tier: large
recommended_model: high_reasoning
actual_model: gpt-5
worker_model_allowed: false
worker_decision_limit: "No delegated implementation; the schema, authority, timing, identity, and failure contracts are one coupled review surface."
escalation_needed: false
output_dir: reports/agent_jobs/GHU_050_MANUAL_INDEPENDENT_CAPTURE_V1_20260804
allowed_files:
  - .github/forecasting-paths.ini
  - .github/workflows/forecasting-tests.yml
  - configs/prediction/manual-independent-capture-v1/config.schema.json
  - configs/prediction/manual-independent-capture-v1/example-config.json
  - configs/prediction/manual-independent-capture-v1/terminal-artifact.schema.json
  - docs/agent_tasks/ghu_050_manual_independent_capture_v1_20260804.md
  - docs/manual_independent_capture_v1.md
  - src/predictor/manual_independent_capture.py
  - tests/ci/test_forecasting_change_classifier.py
  - tests/test_manual_independent_capture.py
  - reports/agent_jobs/GHU_050_MANUAL_INDEPENDENT_CAPTURE_V1_20260804/README.md
---

# GHU-050 manual independent capture contract

Define the versioned, deterministic, fail-closed authority, configuration,
artifact, terminal-failure, provenance, timing, replay, cancellation, and
Phase 7 exclusion contracts for a future exact-race manual research lane.

## Exit predicate

- Exact base and tree remain the values above and the task worktree contains
  only allowlisted changes.
- The contract requires one exact canonical TheDogs race URL and stable race
  identity, one manual run and one attempt, an isolated manual root/profile/
  lock, complete hashes and source byte/time provenance, hard deadlines and
  cancellation, deterministic terminal states, and explicit non-canonical /
  Phase 7 exclusion flags.
- Focused positive/negative schema and invariant tests reject unknown or
  missing fields, unsafe/overlapping paths, hash and identity drift, replay,
  late/conflicting artifacts, outcome leakage, canonical claims, and Phase 7
  eligibility.
- Ticket-specified checks, fatal Ruff, `py_compile`, `git diff --check`, the
  repository classifier, and only its required broader tier pass.
- Exact-diff review has no blocking findings; one focused PR reaches required
  exact-head CI and is merged only under the owner's existing authority.

## Hard stops

- No GHU-051 executor, browser/process/network code, capture, scoring, UI,
  service, timer, deployment, live race, database/corpus/result access,
  training, promotion, EV, staking, or betting.
- No change to existing autonomous capture, collector-request, canonical
  bundle, or Phase 7 contracts.
- No file outside `allowed_files` and no use of the dirty launch checkout for
  implementation.

