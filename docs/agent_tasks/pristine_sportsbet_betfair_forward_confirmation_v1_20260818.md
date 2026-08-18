---
job_id: PRISTINE_SPORTSBET_BETFAIR_FORWARD_CONFIRMATION_V1_20260818
title: Replace compromised PR137 confirmation with pristine forward cohort
lane: Evaluation
supporting_lanes:
  - Provenance
  - Runtime
  - Repo Hygiene
owner: Codex
approval_required: true
approval_source: The owner's 2026-08-18 /goal explicitly authorizes preserving
  the frozen PR137 95 percent Betfair scheduled-off plus 5 percent corrected
  Sportsbet candidate while implementing, publishing, and minimally activating
  a replacement outcome-quarantined forward cohort.
allow_unapproved_safe_extension: false
timeout_seconds: 43200
mutation_mode: safe_extension
allow_audit_code_changes: true
production_data_access: false
production_data_boundary: Label-blind prospective prediction and market inputs
  for eligible races from no earlier than 2026-08-20 through 2026-09-30 only.
  No race outcome, result-bearing field, result artifact, result database row,
  interim metric, October cohort, betting, promotion, or model change access.
live_service_mutation_allowed: true
closeout_scope: repo_and_runtime
output_dir: reports/agent_jobs/PRISTINE_SPORTSBET_BETFAIR_FORWARD_CONFIRMATION_V1_20260818
allowed_files:
  - docs/agent_tasks/pristine_sportsbet_betfair_forward_confirmation_v1_20260818.md
  - docs/sportsbet_betfair_forward_consensus_protocol.md
  - scripts/evaluate_frozen_sportsbet_betfair_forward.py
  - tests/test_evaluate_frozen_sportsbet_betfair_forward.py
  - artifacts/sportsbet_betfair_pristine_forward_confirmation_20260818/frozen_consensus_rule.json
  - artifacts/sportsbet_betfair_pristine_forward_confirmation_20260818/protocol.json
  - artifacts/sportsbet_betfair_pristine_forward_confirmation_20260818/cohort_manifest.json
  - artifacts/sportsbet_betfair_pristine_forward_confirmation_20260818/SHA256SUMS
  - artifacts/sportsbet_betfair_pristine_forward_confirmation_20260818/STATUS.md
  - reports/agent_jobs/PRISTINE_SPORTSBET_BETFAIR_FORWARD_CONFIRMATION_V1_20260818/README.md
  - reports/agent_jobs/PRISTINE_SPORTSBET_BETFAIR_FORWARD_CONFIRMATION_V1_20260818/STATE.md
  - reports/agent_jobs/PRISTINE_SPORTSBET_BETFAIR_FORWARD_CONFIRMATION_V1_20260818/RUN_OUTCOME.json
  - reports/agent_jobs/PRISTINE_SPORTSBET_BETFAIR_FORWARD_CONFIRMATION_V1_20260818/DECISION_ENTRY.json
  - reports/agent_jobs/PRISTINE_SPORTSBET_BETFAIR_FORWARD_CONFIRMATION_V1_20260818/VALIDATION.md
  - reports/agent_jobs/PRISTINE_SPORTSBET_BETFAIR_FORWARD_CONFIRMATION_V1_20260818/commands.log
  - reports/agent_jobs/PRISTINE_SPORTSBET_BETFAIR_FORWARD_CONFIRMATION_V1_20260818/status.json
  - reports/agent_jobs/PRISTINE_SPORTSBET_BETFAIR_FORWARD_CONFIRMATION_V1_20260818/release-receipt.json
evidence_hash: sha256:7a83adb4ebc4c793e8255a4ad07b99f7eb67b7a125deb6d3c79c7a2fd4b97688
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: PRISTINE_SPORTSBET_BETFAIR_FORWARD_CONFIRMATION_V1_20260818
proof_question: Can the exact PR137 candidate be rebound to a future,
  label-blind replacement cohort with fail-closed outcome quarantine and no
  scoring before 2026-09-30 plus explicit external authorization?
hypothesis_id: sportsbet_betfair_pristine_forward_confirmation_v1
program_track: prospective_readiness
entry_state: pr137_candidate_frozen_but_confirmation_window_compromised_by_outcome_rendering
target_transition: pristine_replacement_forward_collection_ready_without_outcome_access_or_scoring
exit_predicate: Current master is verified; candidate bytes and semantics match
  PR137 exactly; the replacement start is no earlier than 2026-08-20 Melbourne
  and remains future at freeze; population is label-blind and reproducible;
  adversarial tests prove result access and premature scoring fail closed;
  checksums and deterministic replay pass; PR137 is only marked compromised in
  new metadata and its artifacts remain byte-identical; source changes complete
  reviewed merge; only the replacement collector required for predictions is
  activated; and no score exists.
source_class: current_origin_master_plus_immutable_pr137_freeze
dataset_version: prospective_unpopulated_replacement_cohort_20260820_20260930
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - MODEL_PERSIST
  - PUBLISH
  - RUNTIME_CHANGE
resume_only_if: Resume only while origin/master, PR137 artifact hashes,
  replacement artifact hashes, zero-outcome-access proof, zero-score proof,
  branch/PR state, and runtime activation state remain exact.
docs_impact: DOCS_UPDATED
docs_checked:
  - AGENTS.md
  - docs/semantic_anti_loop_control_v2.md
  - docs/sportsbet_betfair_forward_consensus_protocol.md
---

# Pristine Sportsbet Betfair forward confirmation

Preserve the frozen PR137 95/5 candidate exactly. Create a replacement cohort
whose eligible jump window starts no earlier than 2026-08-20 Australia/Melbourne
and ends before 2026-10-01. During freeze and collection, permit only label-blind
race identity, prediction, and market inputs. Fail closed before any parser,
diagnostic, report, or scorer can read or render an outcome-bearing row.

Scoring requires both a time strictly after the cohort end and a separate,
explicit external authorization artifact. Produce no interim metrics. Record
PR137 as `COMPROMISED_FOR_PRISTINE_CONFIRMATION` only in new replacement
metadata; do not modify or further inspect its frozen artifacts or outcomes.

Publish source changes through a reviewed PR and merge only after exact-head
validation. Runtime authority is limited to the replacement forward collection
path necessary to accumulate predictions; no result collection, scoring,
October activation, model change, betting, or promotion is authorized.
