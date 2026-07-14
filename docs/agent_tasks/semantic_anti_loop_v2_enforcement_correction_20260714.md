---
job_id: semantic_anti_loop_v2_enforcement_correction_20260714
lane: Reporting
owner: Codex
allowed_files:
  - .codex/hooks.json
  - AGENTS.md
  - docs/agent_tasks/semantic_anti_loop_v2_enforcement_correction_20260714.md
  - docs/semantic_anti_loop_control_v2.md
  - tests/test_build_semantic_anti_loop_seed.py
  - tests/test_semantic_anti_loop_control_v2.py
  - reports/agent_jobs/semantic_anti_loop_v2_enforcement_correction_20260714/CODE_REVIEW.md
  - reports/agent_jobs/semantic_anti_loop_v2_enforcement_correction_20260714/DECISION_ENTRY.json
  - reports/agent_jobs/semantic_anti_loop_v2_enforcement_correction_20260714/DECISIONS.md
  - reports/agent_jobs/semantic_anti_loop_v2_enforcement_correction_20260714/PR_REVIEW.md
  - reports/agent_jobs/semantic_anti_loop_v2_enforcement_correction_20260714/RUN_OUTCOME.json
  - reports/agent_jobs/semantic_anti_loop_v2_enforcement_correction_20260714/STATE.md
  - reports/agent_jobs/semantic_anti_loop_v2_enforcement_correction_20260714/VALIDATION.md
  - reports/agent_jobs/semantic_anti_loop_v2_enforcement_correction_20260714/diff-check.json
  - reports/agent_jobs/semantic_anti_loop_v2_enforcement_correction_20260714/status.json
  - reports/agent_jobs/semantic_anti_loop_v2_enforcement_correction_20260714/validation.json
approval_required: true
timeout_seconds: 14400
output_dir: reports/agent_jobs/semantic_anti_loop_v2_enforcement_correction_20260714
mutation_mode: safe_extension
production_data_access: false
closeout_scope: control_plane_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: greyhound_semantic_anti_loop_v2_pilot_enforcement
proof_question: Does the merged Tenn V2 control plane mechanically enforce Greyhound's pilot without duplicating its four stable seed decisions or blocking legitimate offline research?
hypothesis_id: merged_tenn_v2_hook_release_and_first_five_correction_v1
program_track: offline_development
entry_state: pilot_merged_but_v2_requirement_not_mechanically_enforced
target_transition: greyhound_pilot_enforced_with_validated_seed_reuse_and_first_five_review
exit_predicate: Both Greyhound hooks require V2 through the approved portable skill, locked release semantics are documented and tested, the four stable seeds validate without replay, the first-five review records zero false duplicate blocks, and focused end-to-end controls pass.
source_class: merged_tenn_control_plane_and_greyhound_pilot_registry_snapshot
dataset_version: tenn_af1b33eb_greyhound_40f56646_ledger14_seed_v1
evidence_hash: sha256:4e9ff7a1ccfc5ab285a6f72beb1cf9b41744be25bfefa595fd68aceb312adb53
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: The merged Tenn control plane, Greyhound master, stable seed manifest, shared ledger identity, or focused pilot enforcement evidence changes after a blocked closeout.
---

# Greyhound Semantic Anti-Loop V2 enforcement correction

Correct the merged Greyhound pilot from a clean current-master worktree after
Tenn PR #509 and the approved portable-skill sync. Require V2 mechanically on
both pre-tool and terminal hooks, route through the synced `$HOME/.codex`
portable guard, document release-owned decision publication, validate rather
than replay the four approved seed decisions, and preserve the reviewed
first-five result.

This task is control-plane-only. It authorizes one reviewed commit, branch push,
pull request, and merge for the exact listed files. It does not authorize model,
database, ledger-seed replay, registry-pointer, timer, service, installed
runtime, or production-data mutation.
