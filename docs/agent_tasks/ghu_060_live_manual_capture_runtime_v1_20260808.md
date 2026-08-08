---
job_id: GHU_060_LIVE_MANUAL_CAPTURE_RUNTIME_V1_20260808
title: Implement the bounded live manual capture runtime for GHU-051/GHU-052
lane: Query Orchestration
supporting_lanes:
  - Provenance
  - Runtime Safety
  - Reporting
owner: Codex
approval_required: true
approval_source: >-
  The owner's 2026-08-08 /goal explicitly authorizes the bounded repository
  implementation, focused validation, independent review, dedicated commit and
  review-ready PR; it forbids live attempts, deployment, activation, service
  lifecycle changes, lock manipulation, merge, and canonical/runtime mutation.
allow_unapproved_safe_extension: false
timeout_seconds: 43200
mutation_mode: safe_extension
base: 1c937b53491787f1e54b16d235f7536af48c3c85
base_tree: 78ffa1301b064136f688697ac6881e700232b0ee
production_data_access: false
production_data_boundary: >-
  Controlled local fixtures and mocked browser/process tests only. No live
  browser or source attempt, network execution, SQLite, canonical database or
  history, result evidence, autonomous lock/profile/state/process, Phase 7,
  service installation/activation/restart, model/scoring change, betting,
  promotion, or merge.
github_mutation_allowed: true
git_history_mutation_allowed: true
live_service_mutation_allowed: false
closeout_scope: repo_and_publish
docs_impact: DOCS_REQUIRED
task_tier: large
recommended_model: high_reasoning
actual_model: gpt-5
why_this_model: >-
  Browser isolation, exact-race provenance, process cleanup, parent validation,
  deployment binding, and immutable evidence compatibility cross several safety
  surfaces.
worker_model_allowed: false
worker_decision_limit: >-
  No delegated implementation; the primary agent owns scope, integration,
  review, GitHub publication, and final claims.
escalation_needed: false
output_dir: reports/agent_jobs/GHU_060_LIVE_MANUAL_CAPTURE_RUNTIME_V1_20260808
allowed_files:
  - .github/forecasting-paths.ini
  - .github/workflows/forecasting-tests.yml
  - configs/prediction/manual-independent-capture-v1/evidence-bundle.schema.json
  - docs/agent_tasks/ghu_060_live_manual_capture_runtime_v1_20260808.md
  - docs/manual_independent_capture_v1.md
  - ops/systemd/manual-research-api.service.in
  - src/predictor/manual_independent_capture_executor.py
  - src/predictor/manual_independent_capture_sealer.py
  - src/predictor/manual_live_capture.py
  - src/predictor/manual_live_capture_child.py
  - src/predictor/manual_research_deployment.py
  - src/predictor/manual_research_worker.py
  - tests/test_manual_independent_capture_executor.py
  - tests/test_manual_independent_capture_sealer.py
  - tests/test_manual_live_capture.py
  - tests/test_manual_research_deployment.py
  - reports/agent_jobs/GHU_060_LIVE_MANUAL_CAPTURE_RUNTIME_V1_20260808/README.md
  - reports/agent_jobs/GHU_060_LIVE_MANUAL_CAPTURE_RUNTIME_V1_20260808/STATE.md
  - reports/agent_jobs/GHU_060_LIVE_MANUAL_CAPTURE_RUNTIME_V1_20260808/DECISIONS.md
  - reports/agent_jobs/GHU_060_LIVE_MANUAL_CAPTURE_RUNTIME_V1_20260808/VALIDATION.md
  - reports/agent_jobs/GHU_060_LIVE_MANUAL_CAPTURE_RUNTIME_V1_20260808/REVIEW.md
  - reports/agent_jobs/GHU_060_LIVE_MANUAL_CAPTURE_RUNTIME_V1_20260808/CODE_REVIEW.json
  - reports/agent_jobs/GHU_060_LIVE_MANUAL_CAPTURE_RUNTIME_V1_20260808/RUN_OUTCOME.json
  - reports/agent_jobs/GHU_060_LIVE_MANUAL_CAPTURE_RUNTIME_V1_20260808/guard-preflight.json
  - reports/agent_jobs/GHU_060_LIVE_MANUAL_CAPTURE_RUNTIME_V1_20260808/guard-final.json
  - reports/agent_jobs/GHU_060_LIVE_MANUAL_CAPTURE_RUNTIME_V1_20260808/pr-body.md
  - reports/agent_jobs/GHU_060_LIVE_MANUAL_CAPTURE_RUNTIME_V1_20260808/final-refs.json
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: GHU_060_LIVE_MANUAL_CAPTURE_RUNTIME_V1_20260808
proof_question: >-
  Can one explicit exact race and bound expected runner set traverse the
  existing GHU-051 executor through a dedicated manual browser profile and
  single child attempt, emit the unchanged child envelope, and pass unchanged
  GHU-052 sealing semantics using a versioned live source identity, while all
  redirects, challenges, malformed data, mismatches, invalid odds, timing
  failures, retries, discovery, substitution, and protected-path access fail
  closed?
hypothesis_id: ghu_060_bounded_live_manual_child_fixture_proven
program_track: offline_development
entry_state: >-
  GHU-059 stopped fail-closed at the runtime gate: merged GHU-051 remains
  fixture-only, no live child or default CLI exists, and no safe manual runtime
  is installed; canonical origin/master is bound to the declared base/tree.
target_transition: >-
  One focused review-ready change adds only the bounded live child, explicit
  executor runner binding, live evidence parser identity, default-off package
  binding, focused tests, classifier routing, and architecture documentation.
exit_predicate: >-
  Exact master identity remains bound; the live adapter has no discovery,
  fallback, retry, result access, or autonomous dependency; controlled fixtures
  prove success and every required fail-closed case; existing fixture tests and
  GHU-052 sealing remain valid; focused classifier/test/compile/diff/review
  gates pass; no live attempt, install, activation, restart, lock mutation,
  canonical/Phase 7/scoring/model change, merge, or deployment occurs.
source_class: exact_origin_master_1c937b53_tree_78ffa130_plus_ghu059_runtime_blocker
dataset_version: ghu_060_live_manual_child_controlled_fixture_v1
evidence_hash: sha256:f5754710052f9432a5252495a528c41fbe945db24f88e93bc9e4c3bc1e94853e
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: >-
  Continue only while origin/master and the clean task worktree remain at the
  declared base, the diff stays inside allowed_files, the implementation uses
  the existing GHU-051 executor and GHU-052 validator boundaries, no live or
  autonomous authority is crossed, and all cleanup/provenance checks pass.
---

# GHU-060 bounded live manual capture runtime

This implementation closes the GHU-059 runtime gap without creating a second
capture contract. The live child is a parent-launched Playwright process that
uses only the exact URL, race ID, dedicated profile, and run directory supplied
by GHU-051. It extracts only the pre-jump runner/strict WIN-odds surface and
returns the existing exact child envelope. The parent remains authoritative for
identity, runner binding, odds, timestamps, source bytes, timeout, and process
group cleanup.

The default-off GHU-056 package records the reviewed live entrypoints and their
hashes but does not enable or start them. The bounded CLI requires explicit
race and expected-runner inputs; it has no discovery or substitution mode.

Acceptance is controlled-fixture proof only. No real browser/source attempt is
part of this task.
