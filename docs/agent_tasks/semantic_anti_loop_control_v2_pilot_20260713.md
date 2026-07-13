---
job_id: semantic_anti_loop_control_v2_pilot_20260713
lane: Reporting
owner: Codex
allowed_files:
  - .codex/config.toml
  - .codex/hooks.json
  - .gitignore
  - AGENTS.md
  - docs/agent_decisions/greyhound_semantic_anti_loop_seed_v1.jsonl
  - docs/agent_tasks/semantic_anti_loop_control_v2_pilot_20260713.md
  - docs/semantic_anti_loop_control_v2.md
  - scripts/build_semantic_anti_loop_seed.py
  - tests/test_build_semantic_anti_loop_seed.py
  - reports/agent_jobs/semantic_anti_loop_control_v2_pilot_20260713/CODE_REVIEW.md
  - reports/agent_jobs/semantic_anti_loop_control_v2_pilot_20260713/DECISION_ENTRY.json
  - reports/agent_jobs/semantic_anti_loop_control_v2_pilot_20260713/PR_REVIEW.md
  - reports/agent_jobs/semantic_anti_loop_control_v2_pilot_20260713/RUN_OUTCOME.json
  - reports/agent_jobs/semantic_anti_loop_control_v2_pilot_20260713/SEED_EVIDENCE.md
  - reports/agent_jobs/semantic_anti_loop_control_v2_pilot_20260713/STATE.md
  - reports/agent_jobs/semantic_anti_loop_control_v2_pilot_20260713/VALIDATION.md
  - reports/agent_jobs/semantic_anti_loop_control_v2_pilot_20260713/diff-check.json
  - reports/agent_jobs/semantic_anti_loop_control_v2_pilot_20260713/status.json
  - reports/agent_jobs/semantic_anti_loop_control_v2_pilot_20260713/validation.json
approval_required: true
timeout_seconds: 14400
output_dir: reports/agent_jobs/semantic_anti_loop_control_v2_pilot_20260713
mutation_mode: safe_extension
production_data_access: false
closeout_scope: control_plane_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: semantic_anti_loop_control_v2_pilot
proof_question: Can Greyhound enforce Tenn V2 semantic reuse and meaningful closeout while preserving strict research, persistence, promotion, and runtime boundaries?
hypothesis_id: greyhound_instructions_hooks_and_stable_seed_v1
program_track: offline_development
entry_state: non_trivial_runs_without_semantic_reuse_gate
target_transition: greyhound_semantic_anti_loop_v2_pilot_validated_for_merge
exit_predicate: Repo instructions and hooks require V2, the deterministic four-decision seed is reviewed and validated for merge, focused tests pass, and post-merge seed append plus the first-five-run observation gate remain explicitly pending.
source_class: greyhound_control_plane_source
dataset_version: master_224dc2dddace
evidence_hash: sha256:a8cf759561d91c26f2c11c8bfa4d18ea7bc6ffe6efec1a096cf769cee7135ac7
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: Greyhound canonical source, Tenn V2 control semantics, or one of the four stable seed artifacts changes after a blocked closeout.
---

# Semantic Anti-Loop Control V2 Greyhound Pilot

Validate the merged Tenn V2 control pilot for Greyhound non-trivial work and
prepare only the four owner-approved stable decisions for post-merge append.
This task must not mutate models, databases, registry pointers, timers,
services, deployments, production data, or live runtime state.
