---
job_id: FORWARD_OVERROUND_SUCCESSOR_FINAL_INTEGRITY_REPAIR_V3_20260817
title: Repair final PR 136 prospective evidence integrity blockers
lane: Evaluation
supporting_lanes:
  - Provenance
  - Runtime Safety
  - Testing
  - Reporting
owner: Codex
approval_required: true
approval_source: >-
  The owner's active /goal on 2026-08-17 explicitly authorizes repair of four
  independently reproduced PR 136 evidence-integrity blockers, test-first
  validation, append-only exact-head semantic re-freezing, and one normal
  update to the existing draft PR branch. It forbids merge, deployment,
  activation, cohort creation, live collection, V2 mutation, canonical
  database writes, model or predictive-protocol changes, ROI or betting work,
  dependency installation, and unrelated refactoring.
allow_unapproved_safe_extension: false
timeout_seconds: 14400
mutation_mode: safe_extension
base: 86d5eff3b765176c2c36fca96cfeceee6c1127b5
production_data_access: false
production_data_boundary: >-
  Read only frozen identities, inactive successor state, PR metadata, prior
  review evidence, and installed unit state. All candidate, result, rejection,
  crash, restart, drift, and finalization writes use disposable temporary
  directories. Do not create a live successor cohort, activation receipt,
  installed unit, live request, prediction, result, or database row.
live_service_mutation_allowed: false
github_mutation_allowed: true
git_history_mutation_allowed: true
closeout_scope: repo_and_publish
output_dir: reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_FINAL_INTEGRITY_REPAIR_V3_20260817
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: FORWARD_OVERROUND_SUCCESSOR_FINAL_INTEGRITY_REPAIR_V3_20260817
proof_question: >-
  Can PR 136 enforce prediction-only collection until immutable N=1000,
  permanently tombstone rejected races, bind every executable evidence
  transition byte, and prevent any valid metrics artifact from surviving a
  precommit fatal conflict while keeping the frozen predictive design and
  inactive boundary exact?
hypothesis_id: frozen_overround_allocation_generalizes_prospectively_v1
program_track: prospective_readiness
entry_state: >-
  Draft PR 136 exact head 86d5eff3b765176c2c36fca96cfeceee6c1127b5,
  tree 00107a307a7759ac0cf070a323fa99d498cdc5de, is inactive and
  preserves protocol SHA-256
  4978163d1dd9c0e4ced5eb1d4cb9425d3994379d8c617fb3306a489b838073be,
  but independent review reproduced early result admission, rejected-race
  resurrection, unbound state-machine drift, and surviving precommit metrics.
target_transition: >-
  One reviewed descendant on the existing draft PR branch contains only the
  four integrity repairs, regressions, documentation, and fresh append-only
  freeze evidence; all supported checks pass; the frozen predictive design is
  unchanged; and PR 136 is READY_FOR_INDEPENDENT_REVIEW, not merged, deployed,
  or activated.
exit_predicate: >-
  Pre-fix probes reproduce all four defects; runtime and state machine reject
  any result presence before exactly 1000 predictions, rejected race IDs can
  never become members, activation and admission bind runtime/state-machine/
  finalizer/unit bytes on every nonterminal invocation, and any fatal conflict
  before terminal publication removes or invalidates uncommitted metrics before
  deterministic no-metrics closure. Focused and adjacent regressions, N=1000
  two-phase fault injection, py_compile, JSON/checksum, systemd static checks,
  semantic controls, code review, and git diff checks pass; protocol/model/
  preprocessing/scorer hashes remain exact; successor cohort, activation, and
  installed/enabled/active units remain absent; and the draft PR is updated
  without merge or activation.
source_class: frozen_protocol_integrity_repair_and_synthetic_revalidation
dataset_version: forward_overround_successor_final_integrity_repair_v3_20260817
evidence_hash: sha256:4978163d1dd9c0e4ced5eb1d4cb9425d3994379d8c617fb3306a489b838073be
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: >-
  Resume only while PR base/head, protocol and frozen model, preprocessing,
  scorer, and V2 terminal hashes remain exact; no successor cohort, activation
  receipt, installed/enabled/active unit, live request, prediction, result, or
  canonical database write exists; and every mutation remains in allowed_files.
  Stop for any required hypothesis, model, feature, baseline, N, activation
  boundary, predictive-protocol, or live-data change.
docs_impact: DOCS_REQUIRED
docs_checked:
  - AGENTS.md
  - docs/forward_overround_successor_protocol.md
  - docs/forward_overround_successor_runtime.md
  - docs/semantic_anti_loop_control_v2.md
allowed_files:
  - docs/agent_tasks/forward_overround_successor_final_integrity_repair_v3_20260817.md
  - docs/forward_overround_successor_runtime.md
  - scripts/forward_overround_successor_runtime.py
  - scripts/forward_overround_successor_state_machine.py
  - tests/test_forward_overround_successor_runtime.py
  - tests/test_forward_overround_successor_state_machine.py
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_FINAL_INTEGRITY_REPAIR_V3_20260817/README.md
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_FINAL_INTEGRITY_REPAIR_V3_20260817/STATE.md
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_FINAL_INTEGRITY_REPAIR_V3_20260817/VALIDATION.md
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_FINAL_INTEGRITY_REPAIR_V3_20260817/REVIEW.md
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_FINAL_INTEGRITY_REPAIR_V3_20260817/CODE_REVIEW.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_FINAL_INTEGRITY_REPAIR_V3_20260817/DEFECT_REPRODUCTION.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_FINAL_INTEGRITY_REPAIR_V3_20260817/RUNTIME_MANIFEST.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_FINAL_INTEGRITY_REPAIR_V3_20260817/DEPLOYMENT_READINESS.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_FINAL_INTEGRITY_REPAIR_V3_20260817/SYNTHETIC_E2E.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_FINAL_INTEGRITY_REPAIR_V3_20260817/PREDECESSOR_STATUS.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_FINAL_INTEGRITY_REPAIR_V3_20260817/SHA256SUMS
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_FINAL_INTEGRITY_REPAIR_V3_20260817/RUN_OUTCOME.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_FINAL_INTEGRITY_REPAIR_V3_20260817/DECISION_ENTRY.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_FINAL_INTEGRITY_REPAIR_V3_20260817/status.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_FINAL_INTEGRITY_REPAIR_V3_20260817/guard-preflight.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_FINAL_INTEGRITY_REPAIR_V3_20260817/guard-final.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_FINAL_INTEGRITY_REPAIR_V3_20260817/diff-check.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_FINAL_INTEGRITY_REPAIR_V3_20260817/final-refs.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_FINAL_INTEGRITY_REPAIR_V3_20260817/release-receipt.json
---

# Forward overround successor final integrity repair

Repair only the four independently reproduced evidence-integrity blockers.
Preserve the exact frozen predictive experiment and leave the successor wholly
inactive. Publication is limited to the existing draft PR branch and still
requires a fresh independent review.
