---
job_id: GHU_052_IMMUTABLE_MANUAL_EVIDENCE_SEALING_V1_20260805
title: Implement GHU-052 immutable atomic manual evidence sealing
lane: Provenance
supporting_lanes:
  - Query Orchestration
  - Reporting
owner: Codex
approval_required: true
approval_source: >-
  The owner's 2026-08-05 /goal explicitly authorizes GHU-052-only repository
  implementation, validation, independent exact-diff review, a dedicated
  commit and branch, focused pull request, exact-head CI, and merge only after
  acceptance.
allow_unapproved_safe_extension: false
timeout_seconds: 43200
mutation_mode: safe_extension
base: 47e76063cfa14d697a4f4805f75aeaf9d597762e
production_data_access: false
production_data_boundary: >-
  Fixture artifacts and same-filesystem temporary directories only. No live
  source or browser launch, SQLite, canonical database/history/live_odds,
  autonomous lock/state/process, collector protocol, forward corpus, result
  evidence, service/timer, Phase 7, scoring, training/promotion, EV, staking,
  betting, or deployment access.
github_mutation_allowed: true
git_history_mutation_allowed: true
live_service_mutation_allowed: false
closeout_scope: repo_and_publish
docs_impact: DOCS_REQUIRED
task_tier: large
recommended_model: high_reasoning
actual_model: gpt-5
why_this_model: >-
  Atomic directory publication, interruption recovery, deterministic replay,
  tamper detection, and provenance/leakage closure require careful multi-file
  correctness while retaining the exact GHU-050 and GHU-051 authority bounds.
worker_model_allowed: true
worker_decision_limit: >-
  One independent agent may perform the requested read-only exact-diff review;
  implementation, scope, publishing, merge, and final decisions remain with
  the primary agent.
escalation_needed: false
output_dir: reports/agent_jobs/GHU_052_IMMUTABLE_MANUAL_EVIDENCE_SEALING_V1_20260805
allowed_files:
  - docs/agent_tasks/ghu_052_immutable_manual_evidence_sealing_v1_20260805.md
  - docs/manual_independent_capture_v1.md
  - configs/prediction/manual-independent-capture-v1/evidence-bundle.schema.json
  - configs/prediction/manual-independent-capture-v1/evidence-manifest.schema.json
  - src/predictor/manual_independent_capture_executor.py
  - src/predictor/manual_independent_capture_sealer.py
  - tests/fixtures/manual_independent_capture_child.py
  - tests/test_manual_independent_capture_executor.py
  - tests/test_manual_independent_capture_sealer.py
  - .github/forecasting-paths.ini
  - .github/workflows/forecasting-tests.yml
  - scripts/ci/run_full_forecasting.py
  - tests/ci/test_forecasting_change_classifier.py
  - docs/forecasting_ci_tiers.md
  - reports/agent_jobs/GHU_052_IMMUTABLE_MANUAL_EVIDENCE_SEALING_V1_20260805/README.md
  - reports/agent_jobs/GHU_052_IMMUTABLE_MANUAL_EVIDENCE_SEALING_V1_20260805/VALIDATION.md
  - reports/agent_jobs/GHU_052_IMMUTABLE_MANUAL_EVIDENCE_SEALING_V1_20260805/REVIEW.md
  - reports/agent_jobs/GHU_052_IMMUTABLE_MANUAL_EVIDENCE_SEALING_V1_20260805/RUN_OUTCOME.json
  - reports/agent_jobs/GHU_052_IMMUTABLE_MANUAL_EVIDENCE_SEALING_V1_20260805/DECISION_ENTRY.json
  - reports/agent_jobs/GHU_052_IMMUTABLE_MANUAL_EVIDENCE_SEALING_V1_20260805/status.json
  - reports/agent_jobs/GHU_052_IMMUTABLE_MANUAL_EVIDENCE_SEALING_V1_20260805/guard-preflight.json
  - reports/agent_jobs/GHU_052_IMMUTABLE_MANUAL_EVIDENCE_SEALING_V1_20260805/guard-final.json
  - reports/agent_jobs/GHU_052_IMMUTABLE_MANUAL_EVIDENCE_SEALING_V1_20260805/pr-body.md
  - reports/agent_jobs/GHU_052_IMMUTABLE_MANUAL_EVIDENCE_SEALING_V1_20260805/github-checks.json
  - reports/agent_jobs/GHU_052_IMMUTABLE_MANUAL_EVIDENCE_SEALING_V1_20260805/final-refs.json
  - reports/agent_jobs/GHU_052_IMMUTABLE_MANUAL_EVIDENCE_SEALING_V1_20260805/release-receipt.json
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: GHU_052_IMMUTABLE_MANUAL_EVIDENCE_SEALING_V1_20260805
proof_question: >-
  Can one terminal-success GHU-051 fixture capture with confirmed cleanup be
  transformed into one deterministic, tamper-evident, atomically published
  manual research bundle whose exact race, runner order, timing, source bytes,
  HTTP metadata, odds, config/schema/implementation identities, and exclusions
  verify without canonical or autonomous authority?
hypothesis_id: ghu_052_fixture_proven_immutable_atomic_manual_evidence_v1
program_track: offline_development
entry_state: >-
  Origin master and the clean task worktree are exactly
  47e76063cfa14d697a4f4805f75aeaf9d597762e with tree
  5cc7625500e0d84979de365e5155b45ef28df6af; that commit is merged PR 117 for
  GHU-051, no GHU-052 issue, PR, branch, worktree, or overlapping active claim
  exists, and the planning-pack GHU-052 scope has been read for refresh.
target_transition: >-
  one focused merged pull request adds only a fixture-proven versioned sealer,
  schemas, read-only verifier, minimal executor provenance binding, focused
  tests, documentation, CI inclusion, and required V2 report metadata
exit_predicate: >-
  The final exact diff seals only terminal CAPTURE_READY output with confirmed
  cleanup; publishes canonical bytes through a same-filesystem fsynced staging
  directory and atomic rename; verifies exact replay and rejects mismatch,
  lateness, cancellation, leakage, unsafe paths/symlinks, partial publication,
  interruption, concurrency, stale temp artifacts, and tampering; leaves
  autonomous sentinels unchanged; passes focused GHU-050/051/052/schema/atomic
  tests plus fatal Ruff, py_compile, diff checks, independent ACCEPT review,
  honestly classified exact-head CI, and merge without canonical, runtime,
  browser, scoring, Phase 7, model, or betting action.
source_class: >-
  exact_origin_master_47e76063_tree_5cc76255_merged_ghu050_pr106_and_ghu051_pr117_plus_refreshed_owner_ghu052_goal_and_planning_ticket
dataset_version: ghu_052_fixture_atomic_sealing_contract_20260805
evidence_hash: sha256:c7b88e75f591380068a37989003e6b85dc8d6cd53dd05c8a3d32723b69d203bd
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: >-
  Continue only while origin/master and the task base remain exactly
  47e76063cfa14d697a4f4805f75aeaf9d597762e, the task worktree contains only
  allowlisted changes, no non-stale overlapping claim or PR appears, source and
  odds provenance remain unambiguous, cleanup is confirmed, publication is
  same-filesystem and atomic, and no GHU-050/051 authority rule is weakened;
  stop on canonical access, result/outcome material, non-atomic publication,
  provenance ambiguity, cleanup uncertainty, scope conflict, or failed gate.
---

# GHU-052 immutable manual evidence sealing

Implement only the research-only, non-canonical, Phase-7-excluded seal and
read-only verifier for a completed GHU-051 fixture capture. The refreshed scope
uses the accepted GHU-051 terminal artifact and isolated run root as the sole
producer boundary. It does not reuse canonical receipts, databases, forward
corpus, scoring, browser, service, or collector paths and does not begin
GHU-053.
