---
job_id: issue30_daemon_official_result_retry_v1_20260630
title: Issue 30 daemon official-result retry candidate retention
lane: Evaluation
supporting_lanes:
  - Reporting
owner: Codex
approval_required: true
approval_source: "User replied 'approve', then 'allow db too', then approved proceed for Git history/GitHub PR mutation on 2026-06-30."
allow_unapproved_safe_extension: false
timeout_seconds: 10800
output_dir: reports/agent_jobs/issue30_daemon_official_result_retry_v1_20260630
mutation_mode: safe_extension
production_data_access: false
github_mutation_allowed: true
git_history_mutation_allowed: true
owner_db_append_only_approval: true
allowed_files:
  - docs/agent_tasks/issue30_daemon_official_result_retry_v1_20260630.md
  - reports/agent_jobs/issue30_daemon_official_result_retry_v1_20260630/README.md
  - reports/agent_jobs/issue30_daemon_official_result_retry_v1_20260630/STATE.md
  - reports/agent_jobs/issue30_daemon_official_result_retry_v1_20260630/VALIDATION.md
  - reports/agent_jobs/issue30_daemon_official_result_retry_v1_20260630/PR_REVIEW.md
  - reports/agent_jobs/issue30_daemon_official_result_retry_v1_20260630/backlog_ranking_proof.md
  - reports/agent_jobs/issue30_daemon_official_result_retry_v1_20260630/NEXT_GOAL.md
  - reports/agent_jobs/issue30_daemon_official_result_retry_v1_20260630/diff-check.json
  - scripts/autonomous_official_result_capture.py
  - tests/test_autonomous_official_result_capture.py
docs_impact: DOCS_NOT_REQUIRED
docs_checked:
  - docs/agent_tasks/issue30_daemon_official_result_retry_v1_20260630.md
docs_changed: []
docs_followup: NONE
reason: "Narrow bug fix to existing official-result capture candidate retention; no operator command or public workflow contract is changed."
task_tier: medium
recommended_model: "standard coding model"
actual_model: "Codex GPT-5"
why_this_model: "The fix is a focused daemon candidate-selection bug with DB-read proof and narrow tests."
worker_model_allowed: false
worker_decision_limit: "No worker delegation; orchestrator owns source, validation, and DB boundary."
escalation_needed: false
---

# Issue 30 Daemon Official-Result Retry Candidate Retention

## Objective

Fix GitHub issue #30 by preserving post-jump current-source races in the
official-result capture retry candidate set when the source-backed live-odds
backlog is larger than the daemon candidate limit.

## Scope

Allowed:

- Reproduce the issue #29 backlog-ranking proof with read-only DB queries.
- Edit only `scripts/autonomous_official_result_capture.py` and
  `tests/test_autonomous_official_result_capture.py`.
- Add focused regression coverage showing source-backed current-source races
  are retained for retry even when they fall after the nominal backlog limit.
- Run read-only DB validation and focused tests.
- If code and identity/source gates validate, run append-only official-result
  evidence DB validation/ingest only through the existing official-result
  evidence path and only for the issue #29 exact 15-race packet.
- If exact append-only official-result evidence is already present, validate it
  read-only and do not duplicate append rows.
- Commit and open a draft PR for the narrow issue #30 code/test/task-card
  surface after owner approval.

Forbidden:

- No runtime/service/config, canonical labels, model, promotion, training,
  EV, betting, snapshot, manifest, merge, rebase, reset, stash, clean, branch
  deletion, or worktree deletion mutation.
- No GitHub mutation beyond opening a draft PR for this branch and no Git
  history mutation beyond the one approved local commit/push for this task.
- No broad DB mutation. DB approval is append-only official-result evidence
  for validated issue #29/current-source candidates only.
- No DB append while the shared daemon lock reports `write_allowed=false`.
- No weakening of official source, runner identity, box-set, or result
  completeness gates.
- No use of raw DB totals, stale daemon activity, or old readiness packets as
  current-source recovery proof.

## Validation

- `python3 /home/l4nd0/.agents/skills/tenn-git-guard/scripts/tenn_git_guard.py preflight --repo-root . --topic "GitHub issue #30 daemon official-result capture issue #29 exact 15 current-source packet" --json`
- `python3 /home/l4nd0/tenn-control-plane-task-ledger-status-refresh-v1-20260623/scripts/agent_job_contract.py validate docs/agent_tasks/issue30_daemon_official_result_retry_v1_20260630.md`
- Read-only issue #29 backlog-ranking proof query.
- `python3 -m pytest tests/test_autonomous_official_result_capture.py -q`
- `python3 /home/l4nd0/tenn-control-plane-task-ledger-status-refresh-v1-20260623/scripts/agent_job_contract.py check-diff docs/agent_tasks/issue30_daemon_official_result_retry_v1_20260630.md --repo-root .`
