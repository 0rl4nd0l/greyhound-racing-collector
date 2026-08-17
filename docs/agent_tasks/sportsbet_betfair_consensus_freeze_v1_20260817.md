---
job_id: SPORTSBET_BETFAIR_CONSENSUS_FREEZE_V1_20260817
title: Freeze Sportsbet plus Betfair scheduled-off consensus candidate
lane: Reporting
supporting_lanes:
  - Evaluation
  - Provenance
  - Reporting
owner: Codex
approval_required: true
approval_source: The owner's 2026-08-17 /goal explicitly authorizes one bounded
  corrected Sportsbet WIN plus Betfair scheduled-off convex-weight development
  fit, validation comparison, deterministic rule freeze, and untouched future
  eligibility protocol for 2026-08-18 through 2026-09-30.
allow_unapproved_safe_extension: false
timeout_seconds: 21600
mutation_mode: safe_extension
allow_audit_code_changes: true
production_data_access: false
production_data_boundary: Read-only use of the already-audited strict joined
  June and July report-only surface in the source checkout. No canonical
  database, runtime, service, timer, registry, prediction, betting,
  future-label, or future-result access.
live_service_mutation_allowed: false
closeout_scope: repo_only
output_dir: reports/agent_jobs/SPORTSBET_BETFAIR_CONSENSUS_FREEZE_V1_20260817
allowed_files:
  - docs/agent_tasks/sportsbet_betfair_consensus_freeze_v1_20260817.md
  - scripts/build_sportsbet_betfair_consensus_freeze.py
  - tests/test_build_sportsbet_betfair_consensus_freeze.py
  - artifacts/sportsbet_betfair_consensus_freeze_20260817_report_only/REPORT.md
  - artifacts/sportsbet_betfair_consensus_freeze_20260817_report_only/SHA256SUMS
  - artifacts/sportsbet_betfair_consensus_freeze_20260817_report_only/development_population.jsonl
  - artifacts/sportsbet_betfair_consensus_freeze_20260817_report_only/frozen_consensus_rule.json
  - artifacts/sportsbet_betfair_consensus_freeze_20260817_report_only/future_eligibility_protocol.json
  - artifacts/sportsbet_betfair_consensus_freeze_20260817_report_only/input_manifest.json
  - artifacts/sportsbet_betfair_consensus_freeze_20260817_report_only/protocol.json
  - artifacts/sportsbet_betfair_consensus_freeze_20260817_report_only/replay_fixture.json
  - artifacts/sportsbet_betfair_consensus_freeze_20260817_report_only/report.json
  - artifacts/sportsbet_betfair_consensus_freeze_20260817_report_only/report.schema.json
  - reports/agent_jobs/SPORTSBET_BETFAIR_CONSENSUS_FREEZE_V1_20260817/README.md
  - reports/agent_jobs/SPORTSBET_BETFAIR_CONSENSUS_FREEZE_V1_20260817/STATE.md
  - reports/agent_jobs/SPORTSBET_BETFAIR_CONSENSUS_FREEZE_V1_20260817/RUN_OUTCOME.json
  - reports/agent_jobs/SPORTSBET_BETFAIR_CONSENSUS_FREEZE_V1_20260817/DECISION_ENTRY.json
  - reports/agent_jobs/SPORTSBET_BETFAIR_CONSENSUS_FREEZE_V1_20260817/VALIDATION.md
  - reports/agent_jobs/SPORTSBET_BETFAIR_CONSENSUS_FREEZE_V1_20260817/commands.log
  - reports/agent_jobs/SPORTSBET_BETFAIR_CONSENSUS_FREEZE_V1_20260817/status.json
  - reports/agent_jobs/SPORTSBET_BETFAIR_CONSENSUS_FREEZE_V1_20260817/release-receipt.json
evidence_hash: sha256:86fabb05556160e555f076322eb8786b6166e369a6a8ec57d475c0e4a06e67f7
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: SPORTSBET_BETFAIR_CONSENSUS_FREEZE_V1_20260817
proof_question: Does a predeclared bounded convex combination of normalized
  corrected Sportsbet WIN and normalized Betfair scheduled-off probabilities
  improve paired multiclass race log loss on the frozen July validation slice?
hypothesis_id: sportsbet_betfair_scheduled_off_consensus_v1
program_track: offline_development
entry_state: trustworthy strict joined June and July report-only surface exists;
  no consensus rule or future evaluation protocol has been frozen
target_transition: deterministic development-screening rule and untouched
  2026-08-18 through 2026-09-30 prospective eligibility protocol frozen
exit_predicate: Exact inputs and eligible races are verified; one predeclared
  bounded weight rule is selected deterministically from fit and validation;
  paired validation metrics and meeting-date-cluster uncertainty are reported;
  scorer, config, protocol, replay, and checksums pass; and no August 2026
  outcome or future cohort row is read or scored.
source_class: audited_strict_corrected_sportsbet_win_plus_betfair_scheduled_off
dataset_version: sportsbet_betfair_strict_intersection_20260817_sha256_86fabb05
capabilities:
  - READ
  - REPORT_WRITE
  - DATASET_MATERIALIZE
  - RESEARCH_FIT
  - CODE_EDIT
  - MODEL_PERSIST
resume_only_if: Resume only against the exact declared source hashes and frozen
  protocol. Any input drift, timing ambiguity, outcome leakage, post-hoc
  exclusion, or need for BSP as a predictor requires a new owner-authorized run.
docs_impact: DOCS_UPDATED
docs_checked:
  - AGENTS.md
  - docs/semantic_anti_loop_control_v2.md
  - /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector-ci-routing-fix/docs/betfair_anz_greyhound_historical_csv_primary_source_audit_20260817.md
  - /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector-ci-routing-fix/artifacts/betfair_historical_surface_20260817_report_only/REPORT.md
---

# Sportsbet plus Betfair scheduled-off consensus freeze

This task may fit and persist only the predeclared convex consensus rule. The
fit slice is 2026-06-10 through 2026-06-30 and the validation slice is
2026-07-01 through 2026-07-18, both restricted to the already-audited strict
joined surface. Races on 2026-08-01 and 2026-08-02 are excluded before any
metric or selection step and must not be read by the evaluator.

The only allowed predictors are normalized corrected Sportsbet WIN
probability and normalized Betfair `BEST_AVAIL_BACK_AT_SCHEDULED_OFF` implied
probability. BSP, actual-off time, post-jump fields, names as identity, partial
runner sets, reserve or box mismatches, future labels, and future results are
forbidden. The prospective interval is 2026-08-18 through 2026-09-30; this run
freezes its eligibility and scoring protocol but does not ingest or score it.

No promotion, deployment, canonical database write, registry mutation, live
runtime change, service/timer change, betting output, EV claim, or publication
is authorized.
