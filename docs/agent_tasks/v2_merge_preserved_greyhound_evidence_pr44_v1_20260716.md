---
job_id: v2_merge_preserved_greyhound_evidence_pr44_v1_20260716
lane: Reporting
supporting_lanes:
  - Repo Hygiene
owner: Codex
allowed_files:
  - docs/agent_tasks/v2_merge_preserved_greyhound_evidence_pr44_v1_20260716.md
  - reports/agent_jobs/v2_merge_preserved_greyhound_evidence_pr44_v1_20260716/STATE.md
  - reports/agent_jobs/v2_merge_preserved_greyhound_evidence_pr44_v1_20260716/VALIDATION.md
  - reports/agent_jobs/v2_merge_preserved_greyhound_evidence_pr44_v1_20260716/RUN_OUTCOME.json
  - reports/agent_jobs/v2_merge_preserved_greyhound_evidence_pr44_v1_20260716/DECISION_ENTRY.json
  - reports/agent_jobs/v2_merge_preserved_greyhound_evidence_pr44_v1_20260716/status.json
approval_required: true
allow_unapproved_safe_extension: false
timeout_seconds: 7200
output_dir: reports/agent_jobs/v2_merge_preserved_greyhound_evidence_pr44_v1_20260716
mutation_mode: safe_extension
production_data_access: false
closeout_scope: control_plane_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: merge_preserved_greyhound_evidence_pr44_20260716
proof_question: Can owner-approved draft PR 44 be made ready and merged by a normal merge commit after a fresh exact-head and all-checks-pass audit without changing its preserved evidence or any runtime, data, service, unit, seed, dirty, or stale surface?
hypothesis_id: exact_green_pr44_owner_approved_merge_v1
program_track: offline_development
entry_state: exact_green_mergeable_draft_pr44_at_9f9b1fc3
target_transition: preserved_greyhound_evidence_merged_to_master_with_exact_head_ancestry
exit_predicate: The final PR 44 head consists of the previously green 9f9b1fc3f344de06e93bc1872b5ba1213f18c5a3 evidence head plus only this merge card, all required checks pass on that final head, PR 44 is marked ready and merged by a normal merge commit, and live master contains the final PR head as an ancestor.
source_class: owner_approved_green_github_pull_request
dataset_version: greyhound_pr44_base_d3c27ce2_head_9f9b1fc3_20260716
evidence_hash: sha256:c6a41735dd6f278f1baba0f35e166a4840df82e9ae0df1d4fc5dbb9c73b96556
capabilities:
  - READ
  - REPORT_WRITE
  - PUBLISH
resume_only_if: PR 44 head, base, draft state, mergeability, required checks, source blob identity, owner approval, origin/master, registry state, or worktree cleanliness changes.
---

# Merge preserved Greyhound evidence PR 44

Close the already published evidence-preservation lane after the owner's
explicit 2026-07-16 approval to mark PR #44 ready and merge it.

## Authorized sequence

1. Commit only this merge-authorization card and validate it.
2. Claim the V2 task and rerun the portable guard and identity checks.
3. Push only the task-card commit to the existing PR branch.
4. Wait for every required check on the final PR head to pass.
5. Reverify the live PR head, base, file scope, mergeability, and check results.
6. Mark PR #44 ready and merge it with a normal merge commit.
7. Verify live `master` contains the exact final PR head as an ancestor, write
   the ignored closeout bundle, and release the claim.

## Hard boundaries

- Do not edit any previously published evidence or task card.
- Do not squash, rebase, force-push, auto-merge, delete the branch, or remove a
  worktree.
- Do not touch databases, dependencies, installed hooks, units, services,
  timers, live runtime, production data, models, decision-ledger seeds, dirty
  checkouts, stale checkouts, or unrelated GitHub issues and PRs.
- Do not restore or reactivate the abandoned Greyhound global dispatcher.
- Stop before merge if the final head, base, scope, mergeability, checks, owner
  approval, or exact source evidence identity changes unexpectedly.

## Docs impact

`DOCS_NOT_REQUIRED`: this task records approval and merges an evidence-only PR;
it changes no product behavior, command, schema, workflow, interface, or safety
boundary.
