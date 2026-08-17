---
job_id: FORWARD_OVERROUND_V2_CLOSEOUT_SUCCESSOR_SPEC_V1_20260817
title: Close blocked forward overround V2 and predeclare a reachable successor
lane: Evaluation
supporting_lanes:
  - Provenance
  - Runtime Safety
  - Reporting
owner: Codex
approval_required: true
approval_source: >-
  The owner's active /goal on 2026-08-17 explicitly authorizes deterministic
  fail-closed V2 terminal closeout with zero metrics and preparation, but not
  launch, of a successor fixed-N forward protocol. It forbids V2 mutation or
  retrospective scoring, successor collection, model changes, scheduler
  activation, database writes, ROI or betting claims, and unrelated cleanup.
allow_unapproved_safe_extension: false
timeout_seconds: 14400
mutation_mode: safe_extension
base: origin/master
production_data_access: false
production_data_boundary: >-
  Read the immutable forward-overround V2 cohort and installed unit state. Run
  only the frozen V2 finalizer, which may append FINAL_REPORT.json and
  CONSUMED.json under the existing cohort root. Do not change any pre-existing
  cohort byte, database, service, timer, model, protocol, scorer, prediction,
  result, promotion state, EV, or betting state. Successor validation is
  synthetic and repository-local.
live_service_mutation_allowed: false
github_mutation_allowed: false
git_history_mutation_allowed: true
closeout_scope: repo_and_runtime
output_dir: reports/agent_jobs/FORWARD_OVERROUND_V2_CLOSEOUT_SUCCESSOR_SPEC_V1_20260817
allowed_files:
  - configs/prediction/forward_overround_successor_v1_protocol.json
  - docs/forward_overround_successor_protocol.md
  - docs/agent_tasks/forward_overround_v2_closeout_successor_spec_v1_20260817.md
  - scripts/forward_overround_successor_state_machine.py
  - tests/test_forward_overround_successor_state_machine.py
  - reports/agent_jobs/FORWARD_OVERROUND_V2_CLOSEOUT_SUCCESSOR_SPEC_V1_20260817/README.md
  - reports/agent_jobs/FORWARD_OVERROUND_V2_CLOSEOUT_SUCCESSOR_SPEC_V1_20260817/STATE.md
  - reports/agent_jobs/FORWARD_OVERROUND_V2_CLOSEOUT_SUCCESSOR_SPEC_V1_20260817/VALIDATION.md
  - reports/agent_jobs/FORWARD_OVERROUND_V2_CLOSEOUT_SUCCESSOR_SPEC_V1_20260817/REVIEW.md
  - reports/agent_jobs/FORWARD_OVERROUND_V2_CLOSEOUT_SUCCESSOR_SPEC_V1_20260817/CODE_REVIEW.md
  - reports/agent_jobs/FORWARD_OVERROUND_V2_CLOSEOUT_SUCCESSOR_SPEC_V1_20260817/guard-preflight.json
  - reports/agent_jobs/FORWARD_OVERROUND_V2_CLOSEOUT_SUCCESSOR_SPEC_V1_20260817/guard-final.json
  - reports/agent_jobs/FORWARD_OVERROUND_V2_CLOSEOUT_SUCCESSOR_SPEC_V1_20260817/RUN_OUTCOME.json
  - reports/agent_jobs/FORWARD_OVERROUND_V2_CLOSEOUT_SUCCESSOR_SPEC_V1_20260817/DECISION_ENTRY.json
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: FORWARD_OVERROUND_V2_CLOSEOUT_SUCCESSOR_SPEC_V1_20260817
proof_question: >-
  Can the blocked V2 experiment be terminally closed without metrics or any
  mutation of its prior evidence, while a successor protocol proves a fixed-N
  paired scoring path that survives resolved pre-seal code-drift admission and
  never starts collection without separate authority?
hypothesis_id: frozen_overround_allocation_generalizes_prospectively_v1
program_track: prospective_readiness
entry_state: >-
  V2 has nine immutable prediction receipts, six approved result receipts, an
  immutable installed_capture_code_hash_drift block, no final report, no
  consumed marker, and a disabled inactive forward timer. Its frozen finalizer
  checks the block before sample size, results, or metrics.
target_transition: >-
  V2 has one deterministic BLOCKED_FORWARD_EVIDENCE terminal report with null
  metrics and a matching consumed receipt while all prior cohort bytes remain
  unchanged; a repository-only successor protocol and tested deterministic
  state machine are READY_FOR_AUTHORIZATION but no successor collection or
  scheduler starts.
exit_predicate: >-
  Fresh hashes prove every pre-existing V2 protocol, block, prediction, and
  result byte unchanged; V2 FINAL_REPORT.json is BLOCKED_FORWARD_EVIDENCE with
  null metrics and CONSUMED.json binds it; successor fixed-N rules explicitly
  distinguish temporary pre-seal admission pauses from fatal sealed-evidence
  violations; synthetic tests reach paired scoring after reviewed code-drift
  re-admission, prove fatal no-metrics closure and restart idempotence; focused
  tests, JSON checks, compile and diff checks pass; systemd proves no successor
  unit exists or starts and the V2 timer remains disabled and inactive.
source_class: immutable_forward_overround_v2_block_and_repository_synthetic_successor_validation
dataset_version: forward_overround_v2_terminal_closeout_successor_protocol_v1_20260817
evidence_hash: sha256:2a26391aeba078e543ce3275015925d4036502a3e22758bb34fa10244a92b1a5
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - DATASET_MATERIALIZE
resume_only_if: >-
  Resume only while V2 pre-existing cohort hashes remain exact, the forward V2
  timer remains disabled and inactive, no successor collection has started,
  known V2 outcomes are excluded from model or configuration selection, and
  the repository diff remains inside allowed_files.
docs_impact: DOCS_REQUIRED
docs_checked:
  - AGENTS.md
  - docs/race_evidence_inventory.md
  - docs/semantic_anti_loop_control_v2.md
---

# Forward overround V2 closeout and successor protocol

Append only the terminal artifacts produced by the frozen V2 finalizer. Build
the smallest successor protocol and synthetic state machine needed to prove a
reachable fixed-N paired evaluation. Do not install, enable, start, or invoke a
successor collector, timer, service, production database path, or live source.
