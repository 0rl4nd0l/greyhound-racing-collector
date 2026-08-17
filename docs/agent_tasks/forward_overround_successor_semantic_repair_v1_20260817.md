---
job_id: FORWARD_OVERROUND_SUCCESSOR_SEMANTIC_REPAIR_V1_20260817
title: Repair and re-freeze three successor runtime semantic defects
lane: Evaluation
supporting_lanes:
  - Provenance
  - Runtime Safety
  - Testing
  - Reporting
owner: Codex
approval_required: true
approval_source: >-
  The owner's active /goal on 2026-08-17 explicitly authorizes only the three
  confirmed successor-runtime repairs: restart-idempotent finalization,
  continuous activation-receipt byte binding, and complete unique integer
  official finish positions. It also authorizes targeted regression and fault
  injection testing, exact-N synthetic revalidation, re-freezing required
  evidence, one clean candidate commit, normal branch publication, and a draft
  PR if no blocking finding remains. It forbids activation, deployment, live
  collection, cohort creation, V2 mutation, model or protocol changes,
  canonical database writes, ROI or betting work, and unrelated changes.
allow_unapproved_safe_extension: false
timeout_seconds: 14400
mutation_mode: safe_extension
base: 3283e8d42c13dd494b6f81b48edae1aa60fad683
production_data_access: false
production_data_boundary: >-
  Read only the immutable V2 terminal status, frozen development asset
  identities, absent successor cohort and activation paths, and installed unit
  state. All prediction, result, crash, restart, and finalization writes use
  disposable temporary directories. Do not create a live successor cohort,
  activation receipt, installed unit, request, prediction, result, or database
  row.
live_service_mutation_allowed: false
github_mutation_allowed: true
git_history_mutation_allowed: true
closeout_scope: repo_and_publish
output_dir: reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_SEMANTIC_REPAIR_V1_20260817
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: FORWARD_OVERROUND_SUCCESSOR_SEMANTIC_REPAIR_V1_20260817
proof_question: >-
  Can only the three confirmed defects be repaired so every finalization write
  boundary is restart-idempotent, activation evidence drift always fails
  closed, and accepted official results contain one complete unique integer
  finish position per accepted runner, while the exact protocol/model assets
  remain unchanged and the disabled successor stays inactive?
hypothesis_id: frozen_overround_allocation_generalizes_prospectively_v1
program_track: prospective_readiness
entry_state: >-
  Clean local commit 3283e8d42c13dd494b6f81b48edae1aa60fad683,
  tree a10d927f892bbd9dfad8eaaf0ba9569765944ba3, contains the inactive
  successor candidate. Publication review found exactly three blocking
  semantic defects. V2 remains BLOCKED_FORWARD_EVIDENCE with nine predictions,
  six results, null metrics, and immutable evidence.
target_transition: >-
  One clean child candidate commit contains only the three repairs, regression
  tests, and required re-frozen evidence/readiness updates; all required checks
  pass; a normal-push branch and draft PR identify the exact candidate; and the
  successor is READY_FOR_ACTIVATION_REVIEW but remains inactive and undeployed.
exit_predicate: >-
  Each defect is reproduced by a targeted pre-fix test and passes after repair;
  crash/restart at every finalization write boundary eventually produces one
  deterministic final report and consumed receipt with one score commit;
  activation drift after initialization and malformed, duplicate, missing, or
  non-integer finish positions fail closed; protocol and model assets are
  unchanged; synthetic empty-to-1000-to-1000 reaches exactly one paired score;
  focused and adjacent tests, py_compile, JSON/checksum, systemd static checks,
  repository-native lint/format where available, semantic controls, review,
  and git diff checks pass or exact pre-existing/unavailable failures are
  recorded; no blocking finding remains; successor cohort/activation and
  installed/enabled/active units remain absent; and a draft PR is published
  without merge, deployment, activation, live collection, V2 or DB mutation.
source_class: frozen_protocol_semantic_repair_and_synthetic_revalidation
dataset_version: forward_overround_successor_semantic_repair_v1_20260817
evidence_hash: sha256:4978163d1dd9c0e4ced5eb1d4cb9425d3994379d8c617fb3306a489b838073be
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: >-
  Resume only while base commit/tree, protocol and frozen model,
  preprocessing, scorer, and V2 terminal hashes remain exact; no successor
  cohort, activation receipt, installed/enabled/active unit, live request,
  prediction, result, or canonical DB write exists; and all mutations remain
  within allowed_files. Stop for any fourth semantic defect or required
  protocol/model change.
docs_impact: DOCS_REQUIRED
docs_checked:
  - AGENTS.md
  - CONTEXT.md
  - ARCHITECTURE.md
  - docs/forward_overround_successor_protocol.md
  - docs/forward_overround_successor_runtime.md
  - docs/race_evidence_inventory.md
  - docs/semantic_anti_loop_control_v2.md
allowed_files:
  - docs/agent_tasks/forward_overround_successor_semantic_repair_v1_20260817.md
  - docs/forward_overround_successor_runtime.md
  - scripts/forward_overround_successor_runtime.py
  - scripts/forward_overround_successor_state_machine.py
  - scripts/finalize_forward_overround_successor.py
  - tests/test_forward_overround_successor_runtime.py
  - tests/test_forward_overround_successor_state_machine.py
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_SEMANTIC_REPAIR_V1_20260817/README.md
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_SEMANTIC_REPAIR_V1_20260817/STATE.md
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_SEMANTIC_REPAIR_V1_20260817/VALIDATION.md
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_SEMANTIC_REPAIR_V1_20260817/REVIEW.md
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_SEMANTIC_REPAIR_V1_20260817/CODE_REVIEW.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_SEMANTIC_REPAIR_V1_20260817/DEFECT_REPRODUCTION.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_SEMANTIC_REPAIR_V1_20260817/RUNTIME_MANIFEST.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_SEMANTIC_REPAIR_V1_20260817/DEPLOYMENT_READINESS.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_SEMANTIC_REPAIR_V1_20260817/SYNTHETIC_E2E.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_SEMANTIC_REPAIR_V1_20260817/SHA256SUMS
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_SEMANTIC_REPAIR_V1_20260817/RUN_OUTCOME.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_SEMANTIC_REPAIR_V1_20260817/DECISION_ENTRY.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_SEMANTIC_REPAIR_V1_20260817/status.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_SEMANTIC_REPAIR_V1_20260817/guard-preflight.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_SEMANTIC_REPAIR_V1_20260817/guard-final.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_SEMANTIC_REPAIR_V1_20260817/diff-check.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_SEMANTIC_REPAIR_V1_20260817/final-refs.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_SEMANTIC_REPAIR_V1_20260817/release-receipt.json
---

# Forward overround successor semantic repair

Repair only the three confirmed semantic defects and validate all behavior in
disposable state. Preserve every frozen protocol/model identity and leave the
successor inactive. Publish only a clean draft candidate with no blocking
finding; never deploy or activate it.
