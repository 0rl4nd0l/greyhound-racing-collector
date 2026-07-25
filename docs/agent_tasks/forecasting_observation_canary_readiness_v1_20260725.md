---
job_id: FORECASTING_OBSERVATION_CANARY_READINESS_V1_20260725
title: Forecasting observation canary repository readiness
lane: Provenance
supporting_lanes:
  - Architecture
  - Provenance
  - Evaluation
owner: Codex
approval_required: true
approval_source: "Owner /goal request on 2026-07-25 authorizes repository implementation, validation, push, and one draft PR only."
allow_unapproved_safe_extension: false
timeout_seconds: 21600
mutation_mode: safe_extension
base: origin/master
production_data_access: false
production_data_boundary: "No live service, installed unit, timer, runtime process, legacy database, production operations database, model pointer, prediction, training, promotion, betting, or live artifact mutation."
github_mutation_allowed: true
git_history_mutation_allowed: true
live_service_mutation_allowed: false
docs_impact: DOCS_REQUIRED
task_tier: large
recommended_model: high_reasoning
actual_model: gpt-5
worker_model_allowed: true
worker_decision_limit: "One fresh read-only exact-diff review; no edits, GitHub writes, runtime access, or final publication decision."
escalation_needed: false
output_dir: reports/agent_jobs/FORECASTING_OBSERVATION_CANARY_READINESS_V1_20260725
allowed_files:
  - bin/race-collection-operator
  - config/race_collection_runtime_input.schema.json
  - docs/CANONICAL_RACE_FORECASTING_PHASE7.md
  - docs/FORECASTING_OBSERVATION_CANARY.md
  - docs/agent_tasks/forecasting_observation_canary_readiness_v1_20260725.md
  - race_collection/operational.py
  - race_collection/operator.py
  - race_collection/recovery.py
  - race_collection/runtime_adapters.py
  - race_collection/service.py
  - tests/race_collection/test_operator.py
  - tests/race_collection/test_phase7_operational.py
  - tests/race_collection/test_phase7_runtime_adapter.py
  - reports/agent_jobs/FORECASTING_OBSERVATION_CANARY_READINESS_V1_20260725/README.md
  - reports/agent_jobs/FORECASTING_OBSERVATION_CANARY_READINESS_V1_20260725/STATE.md
  - reports/agent_jobs/FORECASTING_OBSERVATION_CANARY_READINESS_V1_20260725/DECISIONS.md
  - reports/agent_jobs/FORECASTING_OBSERVATION_CANARY_READINESS_V1_20260725/VALIDATION.md
  - reports/agent_jobs/FORECASTING_OBSERVATION_CANARY_READINESS_V1_20260725/REVIEW.md
---

# Forecasting observation canary repository readiness

## Objective

Add only the repository interfaces needed for a future, separately authorized,
result-blind observation canary while preserving the legacy database and live
odds-capture authority.

## Acceptance boundary

- One explicit-path, noninteractive operator CLI delegates to existing
  migration, immutable-registration, cutover, rollback, backup, and restore
  authority APIs.
- The CLI rejects the supplied legacy database as an operations database and
  accepts only a fresh or already-valid separate schema-29 operations store.
- A result-blind runtime input can execute the exact ordered prefix through
  deferred prediction and then stop with receipts 1 through 5.
- Canonical champion and challenger registrations are authenticated before any
  cycle work.
- Generated user service content binds an explicitly verified Python 3.11
  executable, uses `default.target`, retains release identity checks, and emits
  no timer.
- Focused and current-master regression gates pass; a fresh read-only reviewer
  accepts the exact diff before draft-PR publication.

## Hard stops

- No live runtime, service-manager, timer, database, model, prediction,
  training, promotion, betting, or deployment mutation.
- No migrations against the legacy database.
- No ad hoc persistence outside existing authority APIs.
- No merge.
