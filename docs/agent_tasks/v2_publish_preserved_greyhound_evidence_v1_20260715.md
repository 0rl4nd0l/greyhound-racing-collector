---
job_id: v2_publish_preserved_greyhound_evidence_v1_20260715
lane: Reporting
supporting_lanes:
  - Repo Hygiene
owner: Codex
allowed_files:
  - docs/agent_tasks/v2_publish_preserved_greyhound_evidence_v1_20260715.md
  - docs/agent_tasks/v2_preserve_greyhound_untracked_evidence_v1_20260715.md
  - .playwright-mcp/page-2026-07-09T09-37-09-835Z.yml
  - artifacts/full_evidence_orchestration_20260525/daemon_orchestrator_decision_packet_20260616T172233+1000/README.md
  - artifacts/full_evidence_orchestration_20260525/daemon_orchestrator_decision_packet_20260616T172233+1000/verification_results.txt
  - reports/agent_jobs/v2_publish_preserved_greyhound_evidence_v1_20260715/STATE.md
  - reports/agent_jobs/v2_publish_preserved_greyhound_evidence_v1_20260715/VALIDATION.md
  - reports/agent_jobs/v2_publish_preserved_greyhound_evidence_v1_20260715/RUN_OUTCOME.json
  - reports/agent_jobs/v2_publish_preserved_greyhound_evidence_v1_20260715/DECISION_ENTRY.json
  - reports/agent_jobs/v2_publish_preserved_greyhound_evidence_v1_20260715/status.json
approval_required: true
allow_unapproved_safe_extension: false
timeout_seconds: 3600
output_dir: reports/agent_jobs/v2_publish_preserved_greyhound_evidence_v1_20260715
mutation_mode: safe_extension
production_data_access: false
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: publish_preserved_greyhound_evidence_20260715
proof_question: Can the approved preservation commit be transplanted unchanged onto current Greyhound master and published as one focused draft PR without touching stale, dirty, seed, data, unit, service, or runtime surfaces?
hypothesis_id: current_master_evidence_publication_v1
program_track: offline_development
entry_state: local_evidence_commit_on_stale_diverged_branch
target_transition: exact_preserved_evidence_available_in_current_master_draft_pr
exit_predicate: A clean current-master branch contains this publication card and the exact four-file diff from e23951cf5ee2a61ae57034273d3464dd1e51828f, the branch is pushed, and one focused draft PR targets master.
source_class: owner_approved_local_git_commit
dataset_version: greyhound_master_d3c27ce21900_source_e23951cf5ee2
evidence_hash: sha256:b0014977d71094a687df9f2e5f0703ffb7867f1b930030d1c5974ee53cce010d
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: Greyhound origin/master, source commit e23951cf5ee2a61ae57034273d3464dd1e51828f, remote publication branch, matching PR state, or repository cleanliness changes.
---

# Publish preserved Greyhound evidence

Publish the already reviewed evidence preservation commit from a clean sibling
worktree based on current `origin/master`.

## Authorized sequence

1. Validate and commit this task card as the task-card-only bootstrap commit.
2. Claim the task and run the portable post-claim guard and identity checks.
3. Cherry-pick only `e23951cf5ee2a61ae57034273d3464dd1e51828f`.
4. Verify the resulting diff contains only the four source-commit files plus
   this publication task card.
5. Run contract, hash, diff, and closeout checks.
6. Push `agent/preserve-greyhound-evidence` and open exactly one draft PR to
   `master`.

## Hard boundaries

- Do not modify the four files supplied by the preservation commit.
- Do not merge the PR or mark it ready for review.
- Do not alter the stale source checkout or any dirty or seed checkout.
- Do not touch databases, dependencies, units, services, timers, installed
  hooks, live runtime, production data, models, or decision-ledger seeds.
- Do not restore or reactivate the abandoned Greyhound global dispatcher.
- Do not rebase, reset, stash, clean, force-push, delete, or remove worktrees.

## Docs impact

`DOCS_NOT_REQUIRED`: publication preserves existing evidence and task contracts;
it changes no workflow, command, behavior, schema, interface, or safety rule.
