---
job_id: FORWARD_OVERROUND_SUCCESSOR_RUNTIME_V1_20260817
title: Implement and freeze the prepared forward overround successor runtime
lane: Evaluation
supporting_lanes:
  - Provenance
  - Runtime Safety
  - Reporting
owner: Codex
approval_required: true
approval_source: >-
  The owner's active /goal on 2026-08-17 explicitly authorizes implementation,
  synthetic validation, and readiness freezing of the already-reviewed
  successor protocol, while prohibiting activation, live collection, cohort
  creation, V2 mutation or scoring, model or protocol changes, canonical DB
  writes, ROI or betting analysis, and unrelated cleanup.
allow_unapproved_safe_extension: false
timeout_seconds: 14400
mutation_mode: safe_extension
base: 24521a25687887d77bacd6202d471e864e8f986a
production_data_access: false
production_data_boundary: >-
  Read only the immutable V2 terminal status and frozen development asset
  identities. All prediction, result, and finalization proof is synthetic and
  uses temporary test directories. Do not create a successor cohort root,
  activation receipt, installed unit, live request, prediction, result, or
  canonical database row.
live_service_mutation_allowed: false
github_mutation_allowed: false
git_history_mutation_allowed: false
closeout_scope: repo_only
output_dir: reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_RUNTIME_V1_20260817
allowed_files:
  - docs/agent_tasks/forward_overround_successor_runtime_v1_20260817.md
  - docs/forward_overround_successor_runtime.md
  - scripts/forward_overround_successor_runtime.py
  - scripts/finalize_forward_overround_successor.py
  - ops/systemd/forward-overround-successor.service
  - ops/systemd/forward-overround-successor.timer
  - tests/test_forward_overround_successor_runtime.py
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_RUNTIME_V1_20260817/README.md
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_RUNTIME_V1_20260817/STATE.md
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_RUNTIME_V1_20260817/VALIDATION.md
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_RUNTIME_V1_20260817/REVIEW.md
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_RUNTIME_V1_20260817/CODE_REVIEW.md
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_RUNTIME_V1_20260817/RUNTIME_MANIFEST.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_RUNTIME_V1_20260817/DEPLOYMENT_READINESS.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_RUNTIME_V1_20260817/SYNTHETIC_E2E.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_RUNTIME_V1_20260817/SHA256SUMS
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_RUNTIME_V1_20260817/guard-preflight.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_RUNTIME_V1_20260817/guard-final.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_RUNTIME_V1_20260817/RUN_OUTCOME.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_RUNTIME_V1_20260817/DECISION_ENTRY.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_RUNTIME_V1_20260817/release-receipt.json
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: FORWARD_OVERROUND_SUCCESSOR_RUNTIME_V1_20260817
proof_question: >-
  Can the exact prepared protocol be implemented as a disabled fail-closed
  collector, sealer, finalizer, and service unit whose deterministic synthetic
  path reaches exactly one paired-scoring action at 1,000 immutable members,
  while reviewed capture/unit drift can pause and resume and all frozen or
  sealed-evidence drift remains terminal?
hypothesis_id: frozen_overround_allocation_generalizes_prospectively_v1
program_track: prospective_readiness
entry_state: >-
  The reviewed successor protocol is PREPARED_NOT_AUTHORIZED at commit
  24521a25687887d77bacd6202d471e864e8f986a with SHA-256
  4978163d1dd9c0e4ced5eb1d4cb9425d3994379d8c617fb3306a489b838073be;
  V2 is terminal BLOCKED_FORWARD_EVIDENCE with nine predictions, six results,
  no metrics, and immutable evidence; no successor runtime or scheduler exists.
target_transition: >-
  The exact protocol has a hash-frozen tested runtime/finalizer/unit package and
  synthetic empty-to-fixed-N proof, while no cohort, activation receipt,
  installed or enabled timer, live request, prediction, result, or database
  mutation exists; terminal readiness is READY_FOR_ACTIVATION_AUTHORIZATION.
exit_predicate: >-
  Protocol and frozen asset hashes verify; state-machine tests cover admission
  pause and reviewed resume, fatal drift and evidence paths, immutable
  membership, result closure, and one exact-N scoring action; a synthetic
  end-to-end run reaches deterministic finalization; implementation, finalizer,
  service, and timer hashes are frozen; focused and adjacent tests, compile,
  JSON/schema, checksum, unit static validation, lint/format where available,
  and diff checks pass or exact pre-existing/unavailable failures are recorded;
  live checks prove the successor cohort and activation receipt absent and its
  unit/timer disabled and inactive.
source_class: frozen_protocol_and_synthetic_successor_runtime_validation
dataset_version: forward_overround_successor_runtime_v1_20260817
evidence_hash: sha256:4978163d1dd9c0e4ced5eb1d4cb9425d3994379d8c617fb3306a489b838073be
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
resume_only_if: >-
  Resume only while the protocol and frozen model, preprocessing, scorer, and
  V2 terminal hashes remain exact; V2 and its known outcomes remain unused;
  no successor cohort, activation receipt, installed or enabled unit, live
  request, prediction, result, or canonical DB write exists; and the repository
  diff remains inside allowed_files.
docs_impact: DOCS_REQUIRED
docs_checked:
  - AGENTS.md
  - docs/forward_overround_successor_protocol.md
  - docs/race_evidence_inventory.md
  - docs/semantic_anti_loop_control_v2.md
---

# Forward overround successor runtime preparation

Implement and validate only the repository-local prepared runtime package.
All evidence writes in validation must use disposable temporary directories.
Do not install, enable, start, or invoke the service against a live source; do
not create an activation receipt or successor cohort; do not alter or score V2.
