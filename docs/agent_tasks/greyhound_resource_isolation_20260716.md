---
job_id: greyhound_resource_isolation_20260716
title: Reduce full-daemon resource impact without reducing prospective odds capture
lane: Reporting
owner: Codex
approval_required: true
approval_source: Owner authorised bounded user-service, timer, and batch-limit deployment on 2026-07-16.
timeout_seconds: 7200
output_dir: reports/agent_jobs/greyhound_resource_isolation_20260716
mutation_mode: safe_extension
production_data_access: false
production_data_boundary: Existing append-only odds and results collection may continue; no direct DB writes or schema changes are performed by this task.
github_mutation_allowed: true
live_service_mutation_allowed: true
allowed_files:
  - docs/agent_tasks/greyhound_resource_isolation_20260716.md
  - scripts/shadow_autopilot_daemon.py
  - tests/test_shadow_autopilot_daemon.py
  - ops/systemd/shadow-autopilot.service
  - ops/systemd/shadow-autopilot.timer
  - ops/systemd/shadow-autopilot-odds-capture.service
  - ops/systemd/shadow-autopilot-odds-capture.timer
  - docs/race_evidence_inventory.md
  - reports/agent_jobs/greyhound_resource_isolation_20260716/README.md
  - reports/agent_jobs/greyhound_resource_isolation_20260716/RUN_OUTCOME.json
---

# Greyhound resource isolation

Deploy only the generated user-service/timer changes that make full daemon
cycles completion-aware, lower their CPU/I/O priority, bound non-imminent
refresh/result work, and retain the minutely odds-only collector.

Hard boundaries: do not interrupt the running daemon, change llama-server,
write or migrate the SQLite database directly, alter schemas, train/promote
models, or modify historical evidence. Back up installed units before copying
generated units, then deploy only after the current lock owner exits normally.
