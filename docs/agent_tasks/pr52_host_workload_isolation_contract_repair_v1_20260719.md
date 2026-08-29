---
job_id: PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1
title: Repair PR52 hard-limit verification, recursive mount containment, and cleanup failure handling
lane: Query Orchestration
supporting_lanes:
  - Query Orchestration
  - Runtime Proof
owner: Codex
approval_required: true
approval_source: Owner said proceed after the exact-head PR52 adversarial review prescribed one bounded descendant repair and fresh whole-PR review.
allow_unapproved_safe_extension: false
timeout_seconds: 10800
output_dir: reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1
mutation_mode: safe_extension
production_data_access: false
production_data_boundary: Tests and validation use mocks, temporary files, and read-only host metadata only. No production database, evidence payload, live collector, service, timer, installed unit, model, archive, or unrelated Docker workload may be read or changed.
github_mutation_allowed: true
github_mutation_boundary: Fast-forward push one normal descendant to the existing PR52 head branch and update its description only. Do not create another PR, mark ready, merge, close, or modify another PR.
live_service_mutation_allowed: false
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1
proof_question: Does one bounded descendant make PR52 token-exact for io.max, exclude recursively mounted devices, preserve finite timeout and interruption semantics through Docker cleanup failures, and bound host-support probes without weakening the existing read-only wrapper?
hypothesis_id: pr52_fail_closed_limit_mount_cleanup_failure_classes_v1
program_track: offline_development
entry_state: draft PR52 exact head 1af0a240 is merge-clean and green but an independent review reproduced three critical fail-open contract classes and one unbounded host-probe warning
target_transition: one reviewed descendant closes all exact-head PR52 blocker classes and is published to the same draft PR for fresh whole-PR acceptance
exit_predicate: RED/GREEN tests prove exact nested-key limit parsing, non-recursive Docker 29 root binding, cleanup-failure settlement and exit semantics, and bounded host probes; focused suites pass under Python 3.11 and 3.13; format, lint, compile, diff, merge, docs, V2, and GitHub gates pass; PR52 remains draft/open/unmerged.
source_class: exact_pr52_head_1af0a240_plus_independent_board_and_deterministic_adversarial_probes
dataset_version: pr52_1af0a240_board_f4b5d857_code_review_02bb2943_20260719
evidence_hash: sha256:f4b5d857ccd9aa95f94f22fa8bee22ab8a06069678f2d3cb0eeb0c45a87129ee
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: Stop if PR52 head changes from 1af0a24064821864d609adb59744cfb43f69a4b4 before push, master changes from c1dfd464cf6ecfb2034f96ac1a8d3ea58d4e6afa in a way that invalidates the repair, another non-stale claim overlaps an allowed path, the clean repair worktree develops unrelated dirt, a production or runtime mutation becomes necessary, validation fails outside the named repair classes, or publication would require non-fast-forward or force push.
docs_impact: DOCS_UPDATED
docs_checked:
  - AGENTS.md
  - docs/development/bounded_offline_workloads.md
docs_changed:
  - docs/development/bounded_offline_workloads.md
docs_followup: None if the repaired fail-closed and cleanup semantics are documented exactly.
reason: The repair changes enforcement and failure semantics for the documented bounded-offline command.
task_tier: large
recommended_model: high_reasoning
actual_model: Codex GPT-5
why_this_model: The change couples Docker 29 bind topology, cgroup-v2 nested-key verification, subprocess timeouts, signal/cleanup semantics, and exact PR publication.
worker_model_allowed: false
worker_decision_limit: No worker owns the coupled code, tests, docs, publication, or final decision.
escalation_needed: false
allowed_files:
  - docs/agent_tasks/pr52_host_workload_isolation_contract_repair_v1_20260719.md
  - scripts/run_bounded_offline.py
  - tests/test_run_bounded_offline.py
  - docs/development/bounded_offline_workloads.md
  - reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1/README.md
  - reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1/STATE.md
  - reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1/DECISIONS.md
  - reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1/VALIDATION.md
  - reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1/CODE_FIXER.json
  - reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1/CODE_REVIEW.json
  - reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1/RUN_OUTCOME.json
  - reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1/DECISION_ENTRY.json
  - reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1/status.json
  - reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1/validation.json
  - reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1/guard-preflight.json
  - reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1/guard-final.json
  - reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1/diff-check.json
  - reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1/red-tests.json
  - reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1/adversarial-proof.json
  - reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1/remote-pr.json
  - reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1/github-checks.json
  - reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1/final-refs.json
  - reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1/pr-body.md
  - reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1/WAIT_RESULT.json
  - reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1/WAIT_RESULT.json.log
  - reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1/release-receipt.json
  - reports/agent_jobs/PR52_HOST_WORKLOAD_ISOLATION_CONTRACT_REPAIR_V1/commands.log
---

# PR52 host workload isolation contract repair V1

Repair only the exact failure classes reproduced by the independent review of
PR #52 head `1af0a24064821864d609adb59744cfb43f69a4b4`:

1. compare complete `io.max` `rbps` and `riops` tokens rather than substrings;
2. exclude recursive submounts from the Docker 29 scanned-root bind;
3. settle the Docker client and preserve timeout/interruption results even when
   stop or remove fails;
4. bound host-support probes and convert timeout/OS failures into fail-closed
   configuration errors;
5. format the touched Python files and document the repaired semantics.

Add failure-class regression tests before changing the implementation. Preserve
the command vocabulary, exact-root checks, read-only/no-network/capability
boundaries, exclusion defaults, immutable image identity, single worker,
physical-device discovery, priority verification, and no-unconstrained-fallback
semantics. Do not start a Docker container or touch live collection during this
repo-only repair.
