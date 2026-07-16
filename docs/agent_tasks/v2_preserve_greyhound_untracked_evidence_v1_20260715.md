---
job_id: v2_preserve_greyhound_untracked_evidence_v1_20260715
lane: Reporting
supporting_lanes:
  - Repo Hygiene
owner: Codex
allowed_files:
  - docs/agent_tasks/v2_preserve_greyhound_untracked_evidence_v1_20260715.md
  - .playwright-mcp/page-2026-07-09T09-37-09-835Z.yml
  - artifacts/full_evidence_orchestration_20260525/daemon_orchestrator_decision_packet_20260616T172233+1000/README.md
  - artifacts/full_evidence_orchestration_20260525/daemon_orchestrator_decision_packet_20260616T172233+1000/verification_results.txt
  - reports/agent_jobs/v2_preserve_greyhound_untracked_evidence_v1_20260715/STATE.md
  - reports/agent_jobs/v2_preserve_greyhound_untracked_evidence_v1_20260715/VALIDATION.md
  - reports/agent_jobs/v2_preserve_greyhound_untracked_evidence_v1_20260715/RUN_OUTCOME.json
  - reports/agent_jobs/v2_preserve_greyhound_untracked_evidence_v1_20260715/DECISION_ENTRY.json
  - reports/agent_jobs/v2_preserve_greyhound_untracked_evidence_v1_20260715/status.json
approval_required: true
allow_unapproved_safe_extension: false
timeout_seconds: 3600
output_dir: reports/agent_jobs/v2_preserve_greyhound_untracked_evidence_v1_20260715
mutation_mode: safe_extension
production_data_access: false
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: preserve_greyhound_untracked_evidence_20260715
proof_question: Can the owner-approved three untracked Greyhound evidence files be preserved in one local commit without altering their bytes or any database, unit, runtime, seed, or unrelated checkout state?
hypothesis_id: exact_hash_preservation_commit_v1
program_track: offline_development
entry_state: three_owner_approved_untracked_evidence_files
target_transition: exact_three_file_evidence_preserved_in_local_git_history
exit_predicate: One local commit contains this task card and the exact three reviewed evidence files at their recorded hashes, with no other path changed.
source_class: owner_approved_existing_greyhound_evidence
dataset_version: greyhound_head_158ea4affc31
evidence_hash: sha256:de71e151c42eb97aed84dabd81fb2abac5fcd46c4d49f4cc13d843048fca848e
capabilities:
  - READ
  - REPORT_WRITE
  - PUBLISH
resume_only_if: The three file hashes, Greyhound HEAD, owner authorization, or repository dirt changes before commit.
---

# Preserve existing Greyhound untracked evidence

Commit the three owner-approved untracked evidence files without changing their
contents.

## Bound evidence

- `.playwright-mcp/page-2026-07-09T09-37-09-835Z.yml`:
  `sha256:6afea606eda0b554a543b503bdc88a52f64c68d084ead50233d5d9b72f8b7f56`
- `README.md`:
  `sha256:de71e151c42eb97aed84dabd81fb2abac5fcd46c4d49f4cc13d843048fca848e`
- `verification_results.txt`:
  `sha256:4d61c0f1ab6d8f4b823ae836bbdc12829804af72a473ce4df77d09be1788488b`

## Hard boundaries

- Do not edit the three evidence files.
- Do not restore or reactivate the abandoned Greyhound global dispatcher.
- Do not touch databases, units, services, timers, runtime state, seed or dirty
  checkouts, dependencies, GitHub, or unrelated artifacts.
- Do not push, merge, rebase, reset, stash, clean, delete, or remove worktrees.

## Required checks

- Validate and claim this V2 task card.
- Run portable Git guard and identity checks.
- Verify the three hashes before staging and after commit.
- Stage only this task card and the three bound evidence files.
- Review the staged diff and commit locally once.

## Docs impact

`DOCS_NOT_REQUIRED`: exact preservation of existing evidence does not change
workflow, runtime, schema, commands, or safety behavior.
