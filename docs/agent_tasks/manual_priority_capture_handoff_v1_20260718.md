---
job_id: manual_priority_capture_handoff_v1_20260718
title: Reuse one verified autonomous capture in the manual priority race command
lane: Provenance
supporting_lanes:
  - Query Orchestration
  - Runtime proof
  - Testing
owner: Codex
approval_required: true
approval_source: After reviewing the 2026-07-18 Ballarat manual-command lock outcome and the missing autonomous-capture handoff in lay terms, the owner said proceed. This authorizes the smallest local implementation, validation, and one optional outcome-free pre-jump read-only receipt proof; it does not authorize runtime, GitHub, deployment, model, outcome, or canonical database mutation.
allow_unapproved_safe_extension: false
timeout_seconds: 21600
output_dir: reports/agent_jobs/manual_priority_capture_handoff_v1_20260718
mutation_mode: safe_extension
production_data_access: false
production_data_boundary: Product changes and tests use isolated fixtures. An optional live proof may read only one exact pre-jump autonomous capture report, its paired plan/form/sidecar, and the exact matching live_odds rows through SQLite query-only mode. It may not query outcomes or results, write the canonical database, persist a prediction, scrape independently, alter a lock, or mutate any runtime artifact.
owner_db_append_only_approval: false
github_mutation_allowed: false
git_history_mutation_allowed: true
live_service_mutation_allowed: false
allowed_files:
  - docs/agent_tasks/manual_priority_capture_handoff_v1_20260718.md
  - scripts/run_priority_race_prediction.py
  - tests/test_run_priority_race_prediction.py
  - docs/manual_priority_race_prediction.md
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/README.md
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/STATE.md
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/DECISIONS.md
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/VALIDATION.md
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/ARCHITECTURE_REVIEW.md
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/DUPLICATE_SEARCH.md
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/BOARD.md
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/BOARD_DECISION.json
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/CODE_REVIEW.json
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/ADVERSARIAL_REVIEW.md
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/RUNTIME_PROOF.md
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/RUN_OUTCOME.json
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/DECISION_ENTRY.json
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/status.json
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/validation.json
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/diff-check.json
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/git-guard.json
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/live-verification.json
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/integrity.json
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/determinism.json
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/provenance.json
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/append-only.json
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/runtime-boundary.json
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/optional-manual-proof.json
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/WAIT_RESULT.json
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/WAIT_RESULT.json.log
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/focused-pytest.log
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/full-pytest.log
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/ruff.log
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/compile.log
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/determinism.log
  - reports/agent_jobs/manual_priority_capture_handoff_v1_20260718/diff-check.log
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: manual_priority_capture_handoff_v1_20260718
proof_question: Can the standalone manual priority command consume exactly one immutable autonomous fixed-window capture, cross-check it against the matching append-only WIN and PLACE live_odds rows, seal fresh features, and emit one non-persisted outcome-free prediction without acquiring or bypassing the daemon writer lock?
hypothesis_id: verified_autonomous_capture_manual_priority_handoff_v1
program_track: prospective_readiness
entry_state: issue_50_command_is_statically_proven_but_the_first_manual_execution_stopped_waiting_for_a_continuously_owned_daemon_lock_while_the_daemon_successfully_appended_the_target_capture
target_transition: manual_priority_command_can_reuse_one_exact_verified_autonomous_capture_without_writer_lock_contention
exit_predicate: Starting from baseline 5c235643 on preserved PR 45 ancestry with exact PR 46 head 2c595d27 and PR 47 head 0ae5937c adopted once, the manual command accepts only an explicit or boundedly discovered unique autonomous report for the exact target and currently due fixed window; verifies report, plan, runner, provenance, timing, WIN and PLACE rows against one query-only SQLite snapshot; snapshots the accepted bytes into its ephemeral scoring directory; regenerates sealed features; emits deterministic non-persisted full and half stdout; never acquires or releases the writer lock on the reuse path; preserves all existing direct-capture behavior; passes focused/full validation and independent review; and leaves PR 48, services, timers, locks, outcomes, models, thresholds, betting, deployment, merge, and GitHub unchanged.
source_class: issue_50_plus_manual_ballarat_t30_lock_runtime_evidence_plus_current_origin_and_exact_reviewed_pr_heads
dataset_version: master_c1dfd464_baseline_5c235643_pr45_aa35fa70_pr46_2c595d27_pr47_0ae5937c_pr48_f776bfd1_issue50_20260718
evidence_hash: sha256:b32dc0c5a8d90d1297e203e95e1f75bab45a5c196709093efd72576890b36cdf
capabilities:
  - READ
  - REPORT_WRITE
  - DATASET_MATERIALIZE
  - CODE_EDIT
  - PUBLISH
resume_only_if: Resume only while origin/master remains c1dfd464cf6ecfb2034f96ac1a8d3ea58d4e6afa, live PR 46 and PR 47 heads remain 2c595d27ac748d3df8e4031d5491c76606c5be89 and 0ae5937cde87131c714fb7383c58ce13e3cfbc06 with successful checks, the implementation base remains 5c2356431a96659a7cd68edd5b94b50a5877de84, the change stays inside this exact allowlist, and PR 48 remains read-only. Stop on ancestry or contract drift, active overlap, any outcome access, canonical DB write, new scrape on the receipt path, model or threshold mutation, service or timer change, lock mutation or bypass, prediction persistence, GitHub mutation, deployment, merge, betting, promotion, or cohort cutoff.
docs_impact: DOCS_UPDATE_REQUIRED
docs_checked:
  - AGENTS.md
  - ARCHITECTURE.md
  - docs/manual_priority_race_prediction.md
  - issue 50
docs_changed:
  - docs/manual_priority_race_prediction.md
docs_followup: A separate owner-approved activation lane must decide whether to deploy the proven standalone command and how to supersede or rebase only the still-needed PR 48 runtime hook.
reason: "The initial live manual proof exposed one narrow orchestration gap: the daemon can append a valid target capture while its broader cycle keeps the shared lock, but the standalone command cannot consume that completed append. A verified read-only receipt path removes duplicate work without changing writer ownership or runtime."
---

# Manual priority autonomous-capture handoff V1

Implement only the read-only handoff between an already completed autonomous
fixed-window capture and the standalone manual priority scorer. The handoff must
read the original report once, bind it to its sibling plan and exact query-only
WIN and PLACE database rows, copy only the accepted source bytes into a private
temporary scoring directory, regenerate fresh features, and print the existing
non-persisted full and half prediction payload.

The reuse path must not acquire, release, delete, steal, or bypass the shared
writer lock. It must not scrape or append. If no exact valid current-window
receipt exists, the existing bounded lock/direct-capture behavior remains in
force and fails closed.

One optional live proof is authorized only for a different future race that is
still pre-jump, has an exact complete autonomous receipt, and requires no
canonical write. Do not read the completed Ballarat outcome or any result
surface.
