---
job_id: p0_runtime_reconciliation_capture_recovery_20260710
title: P0 runtime reconciliation and WIN plus PLACE capture recovery
lane: Evaluation
supporting_lanes:
  - Provenance
  - Data Quality
  - Evaluation
  - Reporting
owner: Codex
approval_required: true
approval_source: "Owner approved the decision-complete P0 plan and requested implementation on 2026-07-10."
allow_unapproved_safe_extension: false
timeout_seconds: 21600
output_dir: reports/agent_jobs/p0_runtime_reconciliation_capture_recovery_20260710
mutation_mode: safe_extension
production_data_access: false
production_data_boundary: "No direct DB writes; owner-approved append-only production mutations may occur only through the existing capture services."
github_mutation_allowed: true
git_history_mutation_allowed: true
live_service_mutation_allowed: true
owner_db_append_only_approval: true
allowed_files:
  - docs/agent_tasks/p0_runtime_reconciliation_capture_recovery_20260710.md
  - scripts/autonomous_live_odds_capture.py
  - tests/test_autonomous_live_odds_capture.py
  - ops/systemd/shadow-autopilot.service
  - ops/systemd/shadow-autopilot-odds-capture.service
  - docs/race_evidence_inventory.md
  - reports/agent_jobs/p0_runtime_reconciliation_capture_recovery_20260710/README.md
  - reports/agent_jobs/p0_runtime_reconciliation_capture_recovery_20260710/STATE.md
  - reports/agent_jobs/p0_runtime_reconciliation_capture_recovery_20260710/VALIDATION.md
  - reports/agent_jobs/p0_runtime_reconciliation_capture_recovery_20260710/PR_REVIEW.md
  - reports/agent_jobs/p0_runtime_reconciliation_capture_recovery_20260710/DEPLOYMENT_MANIFEST.md
  - reports/agent_jobs/p0_runtime_reconciliation_capture_recovery_20260710/RUNTIME_CHANGE_CLASSIFICATION.md
  - reports/agent_jobs/p0_runtime_reconciliation_capture_recovery_20260710/LIVE_PROOF.md
  - reports/agent_jobs/p0_runtime_reconciliation_capture_recovery_20260710/NEXT_GOAL.md
  - reports/agent_jobs/p0_runtime_reconciliation_capture_recovery_20260710/guard-preflight.json
  - reports/agent_jobs/p0_runtime_reconciliation_capture_recovery_20260710/test-results.txt
  - reports/agent_jobs/p0_runtime_reconciliation_capture_recovery_20260710/code-review.json
  - reports/agent_jobs/p0_runtime_reconciliation_capture_recovery_20260710/db-before.sha3
  - reports/agent_jobs/p0_runtime_reconciliation_capture_recovery_20260710/db-after.sha3
  - reports/agent_jobs/p0_runtime_reconciliation_capture_recovery_20260710/db-before-counts.tsv
  - reports/agent_jobs/p0_runtime_reconciliation_capture_recovery_20260710/db-after-counts.tsv
  - reports/agent_jobs/p0_runtime_reconciliation_capture_recovery_20260710/systemd-before.txt
  - reports/agent_jobs/p0_runtime_reconciliation_capture_recovery_20260710/systemd-after.txt
  - reports/agent_jobs/p0_runtime_reconciliation_capture_recovery_20260710/runtime-backup-manifest.sha256
docs_impact: DOCS_UPDATED
docs_checked:
  - AGENTS.md
  - docs/race_evidence_inventory.md
docs_changed:
  - docs/race_evidence_inventory.md
docs_followup: NONE
reason: "WIN plus PLACE completion semantics and the authoritative runtime discovery/deployment procedure are operator-visible contracts."
task_tier: critical
recommended_model: high_reasoning
actual_model: Codex GPT-5
why_this_model: "The lane reconciles dirty runtime evidence, append-only production data, generated systemd units, live scheduling and GitHub release state."
worker_model_allowed: false
worker_decision_limit: "No worker delegation; the primary agent owns code, validation, deployment and live proof."
escalation_needed: false
---

# P0 Runtime Reconciliation And Capture Recovery

## Objective

Restore one reproducible clean runtime from current `origin/master`, recover the
merged-but-regressed WIN plus PLACE capture contract, and prove two consecutive
healthy scheduled cycles without rewriting historical data or destroying the
dirty launch/runtime evidence surfaces.

## Approved Scope

- Preserve the launch checkout and legacy dirty runtime without cleanup.
- Reconstruct only the PR #37 WIN plus PLACE semantics on current master.
- Validate current provenance, official-result retry, planner fallback, output
  guards and service generation rather than reimplementing them.
- Commit, push, open a PR, deploy its reviewed exact head before merge, operate
  the two user services/timers, and merge only after live proof.
- Back up and inspect the configured production SQLite database. Permit only
  append-only writes through existing capture paths to `live_odds`,
  `autonomous_official_result_evidence_races`, and
  `autonomous_official_result_evidence_runners`.

## Hard Stops

- No destructive cleanup, reset, rebase, stash, lock deletion, DB restore,
  schema migration, historical rewrite, source expansion, training, promotion,
  EV, staking or betting.
- Stop on unexpected table mutation, live PID ownership, installed/generated
  unit mismatch, dirty release worktree, unreviewed PR head, or failed manual
  service smoke.
- Do not repoint services to the dirty legacy runtime after new code executes.
- Do not widen product edits beyond the allowlist.

## Validation

- Tenn Git Guard and task-card contract validation.
- Focused capture, provenance, result, daemon, orchestrator and feature-gate
  tests in an ephemeral validation environment.
- Generator smoke, `systemd-analyze verify`, compile checks and diff checks.
- Code-reviewer audit with no unresolved critical or warning findings.
- Pre/post SQLite integrity, table counts and per-table hashes.
- Manual odds-only and full-daemon success, then two consecutive scheduled full
  cycles and at least two successful odds-only timer firings.
- Installed/generated unit equality, clean release worktree, released lock and
  exact deployed/merged tree equality.
