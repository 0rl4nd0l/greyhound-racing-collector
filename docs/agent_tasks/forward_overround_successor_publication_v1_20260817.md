---
job_id: FORWARD_OVERROUND_SUCCESSOR_PUBLICATION_V1_20260817
title: Publish the frozen forward overround successor runtime
lane: Provenance
supporting_lanes:
  - Runtime Safety
  - Testing
  - Reporting
owner: Codex
approval_required: true
approval_source: >-
  The owner's active /goal on 2026-08-17 explicitly authorizes committing and
  publishing the already-validated forward-overround successor implementation
  without changing its semantics or activating it. It forbids redesign,
  protocol/model/config changes, refit, live collection, cohort or activation
  creation, unit installation or enablement, V2 mutation, canonical database
  writes, ROI or betting work, merge, and unrelated cleanup.
allow_unapproved_safe_extension: false
timeout_seconds: 14400
output_dir: reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_PUBLICATION_V1_20260817
mutation_mode: safe_extension
base: 24521a25687887d77bacd6202d471e864e8f986a
production_data_access: false
production_data_boundary: >-
  Read-only verification of the released implementation evidence, frozen
  assets, immutable V2 terminal state, absent successor cohort and activation,
  and installed-unit state. Validation may write only disposable temporary
  files and this task's report directory.
github_mutation_allowed: true
git_history_mutation_allowed: true
live_service_mutation_allowed: false
closeout_scope: repo_and_publish
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: FORWARD_OVERROUND_SUCCESSOR_PUBLICATION_V1_20260817
proof_question: >-
  Can the byte-exact released successor implementation and required readiness
  evidence be anchored in one clean child commit of 24521a25, published as a
  draft PR, and revalidated at exact HEAD while the successor remains wholly
  inactive and V2 remains immutable?
hypothesis_id: frozen_overround_allocation_generalizes_prospectively_v1
program_track: prospective_readiness
entry_state: >-
  Commit 24521a25687887d77bacd6202d471e864e8f986a contains the reviewed
  protocol and pure state machine. Seven released implementation files remain
  uncommitted with hashes frozen by the released runtime evidence bundle. No
  successor cohort, activation, installed unit, prediction, result, live
  request, or canonical database write exists.
target_transition: >-
  One clean child commit records the exact reviewed implementation, task cards,
  and required readiness evidence; the branch and draft PR are published;
  committed runtime, finalizer, state-machine, service, timer, and protocol
  blobs match the frozen hashes; focused validation passes at exact HEAD; and
  the terminal claim is READY_FOR_ACTIVATION_REVIEW, not activated.
exit_predicate: >-
  Pre-commit scope and hashes equal the released bundle; the candidate commit
  has sole parent 24521a25687887d77bacd6202d471e864e8f986a and no unrelated
  paths; exact committed blob bytes match every frozen runtime identity;
  protocol/checksum/JSON, focused tests, py_compile, systemd static checks, and
  diff checks pass; the branch and draft PR point to the exact candidate HEAD;
  successor cohort and ACTIVATION.json remain absent, successor units remain
  not installed/not enabled/inactive, prediction/result counts remain zero,
  V2 remains 9/6/null-metrics immutable, and no merge or activation occurs.
source_class: >-
  exact_released_forward_overround_successor_runtime_bundle_and_owner_publication_authority
dataset_version: forward_overround_successor_publication_v1_20260817
evidence_hash: sha256:4978163d1dd9c0e4ced5eb1d4cb9425d3994379d8c617fb3306a489b838073be
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: >-
  Continue only while the base is exact, the candidate path set is exact, all
  frozen source and unit hashes match, origin has not advanced incompatibly,
  no overlapping active claim or remote branch exists, V2 remains immutable,
  and the successor remains absent and inactive. Never force-push or retry a
  failed/non-fast-forward push; stop for semantic or hash drift, unrelated path
  inclusion, activation/runtime mutation, or any need to alter the experiment.
docs_impact: DOCS_REQUIRED
docs_checked:
  - AGENTS.md
  - docs/forward_overround_successor_protocol.md
  - docs/forward_overround_successor_runtime.md
  - docs/race_evidence_inventory.md
  - docs/semantic_anti_loop_control_v2.md
allowed_files:
  - docs/agent_tasks/forward_overround_successor_runtime_v1_20260817.md
  - docs/agent_tasks/forward_overround_successor_publication_v1_20260817.md
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
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_PUBLICATION_V1_20260817/STATE.md
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_PUBLICATION_V1_20260817/VALIDATION.md
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_PUBLICATION_V1_20260817/REVIEW.md
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_PUBLICATION_V1_20260817/RUN_OUTCOME.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_PUBLICATION_V1_20260817/DECISION_ENTRY.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_PUBLICATION_V1_20260817/status.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_PUBLICATION_V1_20260817/guard-preflight.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_PUBLICATION_V1_20260817/guard-final.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_PUBLICATION_V1_20260817/diff-check.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_PUBLICATION_V1_20260817/final-refs.json
  - reports/agent_jobs/FORWARD_OVERROUND_SUCCESSOR_PUBLICATION_V1_20260817/release-receipt.json
---

# Forward overround successor publication

Publish only the exact released successor implementation and its required
readiness evidence. Leave the draft PR unmerged and the successor inactive.
