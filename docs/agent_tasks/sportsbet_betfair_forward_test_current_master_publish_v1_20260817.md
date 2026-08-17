---
job_id: SPORTSBET_BETFAIR_FORWARD_TEST_CURRENT_MASTER_PUBLISH_V1_20260817
title: Publish frozen Sportsbet Betfair forward test from current master
lane: Reporting
supporting_lanes:
  - Evaluation
  - Provenance
  - Repo Hygiene
owner: Codex
approval_required: true
approval_source: The owner's 2026-08-17 /goal explicitly authorizes a clean
  commit and draft PR publishing the already-frozen 95 percent Betfair
  scheduled-off plus 5 percent corrected Sportsbet WIN rule and untouched
  2026-08-18 through 2026-09-30 protocol without future scoring.
allow_unapproved_safe_extension: false
timeout_seconds: 21600
mutation_mode: safe_extension
allow_audit_code_changes: true
production_data_access: false
production_data_boundary: Hash verification of frozen June and July evidence
  only. No 2026-08-18 or later predictor, label, result, cohort, metric,
  canonical database, runtime, service, timer, betting or live API access.
live_service_mutation_allowed: false
closeout_scope: repo_only
output_dir: reports/agent_jobs/SPORTSBET_BETFAIR_FORWARD_TEST_CURRENT_MASTER_PUBLISH_V1_20260817
allowed_files:
  - docs/agent_tasks/sportsbet_betfair_consensus_freeze_v1_20260817.md
  - docs/agent_tasks/sportsbet_betfair_forward_test_freeze_publish_v1_20260817.md
  - docs/agent_tasks/sportsbet_betfair_forward_test_current_master_publish_v1_20260817.md
  - docs/sportsbet_betfair_forward_consensus_protocol.md
  - scripts/build_sportsbet_betfair_consensus_freeze.py
  - scripts/evaluate_frozen_sportsbet_betfair_forward.py
  - tests/test_build_sportsbet_betfair_consensus_freeze.py
  - tests/test_evaluate_frozen_sportsbet_betfair_forward.py
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
  - reports/agent_jobs/SPORTSBET_BETFAIR_FORWARD_TEST_CURRENT_MASTER_PUBLISH_V1_20260817/README.md
  - reports/agent_jobs/SPORTSBET_BETFAIR_FORWARD_TEST_CURRENT_MASTER_PUBLISH_V1_20260817/STATE.md
  - reports/agent_jobs/SPORTSBET_BETFAIR_FORWARD_TEST_CURRENT_MASTER_PUBLISH_V1_20260817/RUN_OUTCOME.json
  - reports/agent_jobs/SPORTSBET_BETFAIR_FORWARD_TEST_CURRENT_MASTER_PUBLISH_V1_20260817/DECISION_ENTRY.json
  - reports/agent_jobs/SPORTSBET_BETFAIR_FORWARD_TEST_CURRENT_MASTER_PUBLISH_V1_20260817/VALIDATION.md
  - reports/agent_jobs/SPORTSBET_BETFAIR_FORWARD_TEST_CURRENT_MASTER_PUBLISH_V1_20260817/commands.log
  - reports/agent_jobs/SPORTSBET_BETFAIR_FORWARD_TEST_CURRENT_MASTER_PUBLISH_V1_20260817/status.json
  - reports/agent_jobs/SPORTSBET_BETFAIR_FORWARD_TEST_CURRENT_MASTER_PUBLISH_V1_20260817/release-receipt.json
evidence_hash: sha256:86fabb05556160e555f076322eb8786b6166e369a6a8ec57d475c0e4a06e67f7
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: SPORTSBET_BETFAIR_FORWARD_TEST_CURRENT_MASTER_PUBLISH_V1_20260817
proof_question: Can the validated local freeze package be transplanted as an
  exact focused diff onto current origin/master, pass exact-head validation,
  and be published as one draft PR with future scores absent?
hypothesis_id: sportsbet_betfair_forward_consensus_current_master_publication_v1
program_track: offline_development
entry_state: validated_package_commit_5418f02e_is_parked_on_a_divergent_source_base
target_transition: exact_frozen_consensus_package_in_one_clean_current_master_draft_pr_without_future_scoring
exit_predicate: The final branch is based on exact origin/master
  2f82901d7df6927de56958307324840021a4db6a; its diff contains only the frozen
  package and this card; hashes, focused tests, compilation, JSON/schema,
  checksums, deterministic replay, reviews and whitespace pass at exact head;
  future scores remain absent; and one draft PR targets master.
source_class: validated_local_package_commit_5418f02e_plus_current_origin_master
dataset_version: sportsbet_betfair_strict_intersection_20260817_sha256_86fabb05
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - MODEL_PERSIST
  - PUBLISH
resume_only_if: Resume only while origin/master, source package commit,
  declared hashes, branch/PR state and zero-future-score proof remain exact.
docs_impact: DOCS_UPDATED
docs_checked:
  - AGENTS.md
  - docs/semantic_anti_loop_control_v2.md
  - docs/agent_tasks/sportsbet_betfair_consensus_freeze_v1_20260817.md
---

# Current-master publication of the frozen forward consensus test

Transplant only the validated package from
`5418f02e05048fbf18c2199f29dfe4047d953dc0` onto exact current
`origin/master` `2f82901d7df6927de56958307324840021a4db6a`. Omit the parked publication
attempt card from the final diff and retain this current-master card instead.

Do not alter the frozen 95/5 rule, development evidence, protocol or hashes.
Do not open, inventory, materialize or score any real 2026-08-18 or later
predictor, label, result or cohort source. Synthetic contract fixtures only.
Publish exactly one draft PR to master; do not merge or mark it ready.
