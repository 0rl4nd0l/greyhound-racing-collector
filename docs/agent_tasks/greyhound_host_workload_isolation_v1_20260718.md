---
job_id: greyhound_host_workload_isolation_v1_20260718
title: Bound heavy read-only Greyhound agent and offline filesystem workloads
lane: Query Orchestration
owner: Codex
approval_required: true
approval_source: Owner authorised implementation, validation, commit, push, and a draft PR in goal IMPLEMENT_GREYHOUND_HOST_WORKLOAD_ISOLATION_V1 on 2026-07-18.
timeout_seconds: 10800
output_dir: reports/agent_jobs/greyhound_host_workload_isolation_v1_20260718
mutation_mode: safe_extension
production_data_access: false
production_data_boundary: The task may observe process, cgroup, PSI, diskstats, and user-service metadata read-only; it must not read or write the active SQLite database or production evidence payloads.
github_mutation_allowed: true
live_service_mutation_allowed: false
closeout_scope: repo_only
control_contract_version: 2
project_id: greyhound_racing_collector
claim_id: greyhound_host_workload_isolation_v1_20260718
proof_question: Can broad read-only Greyhound agent and offline filesystem work be admitted only through an exact-root, single-worker, exclusion-aware, time-bounded, low-priority child with a verified cgroup-v2 physical-read limit while live collection remains untouched?
hypothesis_id: docker_cgroup_v2_exact_root_offline_wrapper_v1
program_track: offline_development
entry_state: verified_unbounded_offline_scan_incident_with_user_systemd_io_controller_not_delegated
target_transition: fail_closed_bounded_offline_wrapper_validated_under_unchanged_live_collection
exit_predicate: A focused wrapper and tests reject unbounded use, prove command and exclusion construction plus cleanup and hard-limit failure, run one controlled measurement through a cgroup with visible io.max, update agent guidance, and land as one reviewed draft PR without collector or data mutation.
source_class: verified_host_process_kernel_device_systemd_docker_and_repository_evidence
dataset_version: canonical_c1dfd464_systemd249_docker29_cgroup2_20260718
evidence_hash: sha256:3038b59f84b49352e6bac5da4956e26505c7ead2c36e1887ea4e7ee7bdbeb1ae
capabilities:
  - READ
  - REPORT_WRITE
  - CODE_EDIT
  - PUBLISH
resume_only_if: Canonical HEAD, task-card admission, Docker cgroup-v2 hard-limit proof, live collection safety, focused validation, or draft-PR state changes before closeout.
allowed_files:
  - docs/agent_tasks/greyhound_host_workload_isolation_v1_20260718.md
  - scripts/run_bounded_offline.py
  - tests/test_run_bounded_offline.py
  - docs/development/bounded_offline_workloads.md
  - AGENTS.md
  - reports/agent_jobs/greyhound_host_workload_isolation_v1_20260718/README.md
  - reports/agent_jobs/greyhound_host_workload_isolation_v1_20260718/STATE.md
  - reports/agent_jobs/greyhound_host_workload_isolation_v1_20260718/VALIDATION.md
  - reports/agent_jobs/greyhound_host_workload_isolation_v1_20260718/MEASUREMENT.md
  - reports/agent_jobs/greyhound_host_workload_isolation_v1_20260718/measurement.json
  - reports/agent_jobs/greyhound_host_workload_isolation_v1_20260718/PR_REVIEW.md
  - reports/agent_jobs/greyhound_host_workload_isolation_v1_20260718/RUN_OUTCOME.json
  - reports/agent_jobs/greyhound_host_workload_isolation_v1_20260718/DECISION_ENTRY.json
  - reports/agent_jobs/greyhound_host_workload_isolation_v1_20260718/status.json
  - reports/agent_jobs/greyhound_host_workload_isolation_v1_20260718/guard-preflight.json
  - reports/agent_jobs/greyhound_host_workload_isolation_v1_20260718/commands.log
---

# Greyhound host workload isolation V1

Implement one repository-owned wrapper for heavy read-only agent and offline
filesystem work. The wrapper must require one exact existing root, mount it
read-only into a transient child, default to one worker, exclude irrelevant and
large trees, enforce a finite timeout, lower CPU and I/O priority, and refuse to
run unless the child can verify its cgroup-v2 physical-read ceiling.

Use the completed incident explanation only as established evidence. Do not
repeat diagnosis, run an uncontrolled sibling-worktree scan, or modify the live
collector, active SQLite database, production evidence, installed units,
timers, secrets, models, or unrelated Docker workloads.

The task may create and remove only its own short-lived validation containers.
It must use the existing local image without pulling and must leave no task
container behind after normal exit, timeout, failure, or interruption.
