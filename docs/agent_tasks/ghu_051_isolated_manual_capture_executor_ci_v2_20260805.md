---
job_id: GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_CI_V2_20260805
title: Complete GHU-051 executor with exact-head CI inclusion
lane: Query Orchestration
supporting_lanes:
  - Provenance
  - Reporting
owner: Codex
approval_required: true
approval_source: >-
  The owner's 2026-08-05 /goal explicitly authorizes the bounded GHU-051
  implementation, validation, independent exact-diff review, dedicated commit
  and branch, focused pull request, exact-head CI, and gated merge.
allow_unapproved_safe_extension: false
timeout_seconds: 43200
mutation_mode: safe_extension
base: e4f3699986237aad265b34e77d06d536f6046ee4
production_data_access: false
production_data_boundary: >-
  Fixture and OS-process tests only. No live source or browser launch, SQLite,
  canonical database/history/live_odds, autonomous lock/state/process, forward
  corpus, collector request/state, result evidence, service/timer, Phase 7,
  model training/promotion, EV, staking, betting, or deployment access.
github_mutation_allowed: true
git_history_mutation_allowed: true
live_service_mutation_allowed: false
closeout_scope: repo_and_publish
docs_impact: DOCS_REQUIRED
task_tier: large
recommended_model: high_reasoning
actual_model: gpt-5
why_this_model: >-
  Process-group cleanup and authority isolation require careful proof, while
  the final diff must also preserve the repository's exact-head CI contract.
worker_model_allowed: true
worker_decision_limit: >-
  One independent agent may perform the requested read-only exact-diff review;
  implementation, scope, publishing, merge, and final decisions remain with
  the primary agent.
escalation_needed: false
output_dir: reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_CI_V2_20260805
allowed_files:
  - docs/agent_tasks/ghu_051_isolated_manual_capture_executor_v1_20260805.md
  - docs/agent_tasks/ghu_051_isolated_manual_capture_executor_ci_v2_20260805.md
  - docs/manual_independent_capture_v1.md
  - src/predictor/manual_independent_capture_executor.py
  - tests/fixtures/manual_independent_capture_child.py
  - tests/test_manual_independent_capture_executor.py
  - .github/forecasting-paths.ini
  - .github/workflows/forecasting-tests.yml
  - scripts/ci/run_full_forecasting.py
  - tests/ci/test_forecasting_change_classifier.py
  - docs/forecasting_ci_tiers.md
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_CI_V2_20260805/README.md
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_CI_V2_20260805/VALIDATION.md
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_CI_V2_20260805/REVIEW.md
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_CI_V2_20260805/RUN_OUTCOME.json
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_CI_V2_20260805/DECISION_ENTRY.json
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_CI_V2_20260805/status.json
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_CI_V2_20260805/guard-preflight.json
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_CI_V2_20260805/guard-final.json
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_CI_V2_20260805/pr-body.md
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_CI_V2_20260805/github-checks.json
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_CI_V2_20260805/final-refs.json
  - reports/agent_jobs/GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_CI_V2_20260805/release-receipt.json
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: GHU_051_ISOLATED_MANUAL_CAPTURE_EXECUTOR_CI_V2_20260805
proof_question: >-
  Can one exact TheDogs race fixture be captured through a manual-only,
  single-attempt child process with a dedicated lock, fixed isolated profile
  and run root, immediate pre-launch margin check, monotonic cancellation and
  timeout, proven TERM/KILL process-group cleanup, and a GHU-050-valid terminal
  artifact without reading or changing autonomous or canonical state?
hypothesis_id: ghu_051_fixture_proven_isolated_manual_capture_executor_ci_v2
program_track: offline_development
entry_state: >-
  Origin master, the clean task HEAD, and their trees remain exactly
  e4f3699986237aad265b34e77d06d536f6046ee4 and
  ac9cdde82d3a0ede953e36c9a87d9afd216c5826. The predecessor V2 milestone
  released ADVANCED after proving the fixture-only seam feasible and finding
  that exact-head CI explicitly enumerates manual tests.
target_transition: >-
  one exact-base GHU-051 successor completes the fixture executor plus explicit
  manual/full forecasting CI inclusion, final validation, independent review,
  focused PR, exact-head CI, and conditional merge
exit_predicate: >-
  The exact diff implements the fixture-only executor, includes its test in the
  manual and full forecasting commands, classifies the product-plus-CI diff
  honestly, passes focused GHU-050/GHU-051 and affected autonomous/process
  regressions plus fatal Ruff, py_compile, and diff checks, receives independent
  ACCEPT review, passes exact-head GitHub CI, and merges without touching live,
  autonomous, canonical, outcome, model, deployment, Phase 7, or betting state.
source_class: >-
  exact_origin_master_e4f3699_tree_ac9cdde_plus_ghu050_pr106_contract_plus_ghu051_scope_refresh_decision_v1
dataset_version: ghu_051_fixture_process_and_ci_contract_20260805
evidence_hash: sha256:941de8628cb7cacd600f16fd18320bcba37d6694f4f457b19f19e3f128ccd563
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: >-
  Continue only while origin/master and the task base remain exact, the diff is
  allowlisted, no non-stale overlapping claim or PR appears, browser authority
  remains fixture-only and manual-root-bound, and process cleanup is provable;
  stop on any canonical/autonomous dependency, cleanup uncertainty, conflict,
  failed gate, or weakened GHU-050 rule.
---

# GHU-051 executor and exact-head CI inclusion

Complete the predecessor's fixture-only manual process seam and add only the
minimal CI routing needed for the repository's enumerated manual and full
forecasting checks to execute its tests. Shared browser and autonomous capture
code remain unchanged. The strongest permitted claim is fixture-proven isolated
executor; this does not authorize a live implementation or GHU-052 evidence.
