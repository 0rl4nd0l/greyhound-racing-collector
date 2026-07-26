---
job_id: OBSERVATION_AUTHORITY_MODE_BOUNDARY_V1_20260726
title: Bind observation authority to the result-blind runtime cycle
lane: Provenance
supporting_lanes:
  - Architecture
  - Provenance
owner: Codex
approval_required: true
approval_source: "Owner /goal request on 2026-07-26 authorizes the bounded repair, validation, commit, push, and one draft PR."
allow_unapproved_safe_extension: false
timeout_seconds: 21600
mutation_mode: safe_extension
base: origin/master
production_data_access: false
production_data_boundary: "No deployment, activation, service, timer, runtime input, database, model, prediction, training, promotion, betting, or live-data mutation."
github_mutation_allowed: true
git_history_mutation_allowed: true
live_service_mutation_allowed: false
docs_impact: DOCS_REQUIRED
task_tier: large
recommended_model: high_reasoning
actual_model: gpt-5
worker_model_allowed: false
worker_decision_limit: "No delegated implementation or owner-boundary decision."
escalation_needed: false
output_dir: reports/agent_jobs/OBSERVATION_AUTHORITY_MODE_BOUNDARY_V1_20260726
allowed_files:
  - config/race_collection_runtime_input.schema.json
  - docs/CANONICAL_RACE_FORECASTING_PHASE7.md
  - docs/FORECASTING_OBSERVATION_CANARY.md
  - docs/agent_tasks/observation_authority_mode_boundary_v1_20260726.md
  - race_collection/runtime_adapters.py
  - race_collection/service.py
  - tests/race_collection/test_phase7_runtime_adapter.py
  - reports/agent_jobs/OBSERVATION_AUTHORITY_MODE_BOUNDARY_V1_20260726/README.md
  - reports/agent_jobs/OBSERVATION_AUTHORITY_MODE_BOUNDARY_V1_20260726/STATE.md
  - reports/agent_jobs/OBSERVATION_AUTHORITY_MODE_BOUNDARY_V1_20260726/DECISIONS.md
  - reports/agent_jobs/OBSERVATION_AUTHORITY_MODE_BOUNDARY_V1_20260726/VALIDATION.md
  - reports/agent_jobs/OBSERVATION_AUTHORITY_MODE_BOUNDARY_V1_20260726/REVIEW.md
---

# Observation authority mode boundary

## Objective

Make observation release authority mechanically compatible only with an
explicit `result-blind-observation-v1` runtime cycle ending at deferred
prediction, without changing separately authorised complete-cycle behavior.

## Acceptance boundary

- Observation authority rejects omitted, complete, unknown, conflicting, or
  internally inconsistent runtime modes.
- Composition binds durable release authority to the adapter's explicit mode
  and immutable cycle terminal phase.
- Observation cycles compose and plan only the five phases through deferred
  prediction.
- Full release authority preserves the valid complete nine-phase cycle.
- Existing immutable release, bundle, operations-DB, recovery, service, and
  Python 3.11 contracts remain unchanged.
- Focused validation and one fresh read-only exact-diff review pass before
  draft-PR publication.

## Hard stops

- No deployment, activation, service-manager, runtime input, database, model,
  prediction, training, promotion, betting, or live-data action.
- No weakening of full-operation release behavior beyond this authority
  boundary.
- No merge.
