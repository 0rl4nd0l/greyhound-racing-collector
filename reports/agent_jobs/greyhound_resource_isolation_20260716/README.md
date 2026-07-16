# Greyhound resource isolation — RUNNING

Objective: lower heavy full-daemon CPU and I/O contention while retaining the
minutely append-only odds-only collection lane.

Initial evidence: full-daemon PID 1503518 held the shared lock; its HWM was
199852 kB. Host swap was about 9.8 GiB used, with llama-server occupying about
8.4 GiB RSS. SQLite `quick_check` and `integrity_check` both returned `ok`.

The legacy feature checkout was stale relative to master, so implementation is
in clean worktree `greyhound-resource-isolation-20260716` from `origin/master`.

## Generator review

```json
{
  "status": "SUCCESS",
  "work_log": {
    "assumptions": ["The full daemon is the heavy lane; odds-only capacity is time-sensitive."],
    "sources_used": ["git diff", "installed user units", "systemd unit syntax verification"],
    "files_read": ["scripts/shadow_autopilot_daemon.py", "ops/systemd/*.service", "ops/systemd/shadow-autopilot.timer", "tests/test_shadow_autopilot_daemon.py"],
    "files_modified": ["scripts/shadow_autopilot_daemon.py", "ops/systemd/*.service", "ops/systemd/shadow-autopilot.timer", "tests/test_shadow_autopilot_daemon.py", "docs/race_evidence_inventory.md"],
    "validation_checks": ["133 focused daemon tests passed", "py_compile passed", "git diff --check passed"]
  },
  "result": {"critical": [], "warnings": [], "suggestions": []}
}
```

## Deployment and validation

- Outcome: `PARTIAL_IMPROVEMENT` — the controls are installed and effective in
  systemd, but the next resource-capped full cycle is scheduled for 16:17:27
  AEST and was not force-started or observed.
- Deployed runtime code commit: `9395798f452e8c21dd28a59494e5fee11bf3f84d`.
- Backup: `/home/l4nd0/.config/systemd/user/greyhound-resource-isolation-backup-20260716T1607+1000`.
- Generated and installed equality: byte-for-byte `cmp` passed for both
  services and both timers after copy and `systemctl --user daemon-reload`.
- Full timer: `OnUnitInactiveSec=15min`; after the full daemon completed at
  16:02:26 AEST, systemd scheduled the next start for 16:17:27 AEST.
- Full controls: `Nice=10`, `CPUWeight=20`, `IOWeight=20`,
  `IOSchedulingClass=idle`; `MemoryHigh` remains `infinity`.
- Full budgets: refresh 16 -> 6; rejoin remains 8; result backlog 32 -> 8 and
  backlog shadow runs 64 -> 16. The full lane still retains its 16-race
  autonomous odds capacity.
- Odds-only: timer enabled and active; its near-minutely calendar and 16-race
  refresh/capture budget are unchanged, with best-effort I/O retained.
- Pause switch: creating `pause-heavy-scheduling` at the documented shared
  runtime path prevents only future full-daemon starts. It does not interrupt a
  running cycle or disable odds-only capture.
- Lock lifecycle: the preceding full service completed successfully and
  released its lock. The currently active odds-only process owns the same JSON
  lock, as designed; no second full service is active.
- SQLite: read-only `quick_check=ok`, `integrity_check=ok`, 101 schema objects,
  schema SHA-256 `49e3...9417de6e`. No schema-changing code or command ran.
- Resource context: the prior full-daemon HWM was 199852 kB, so no `MemoryHigh`
  was added. Host swap was about 9.8 GiB used and llama-server was about 8.4
  GiB RSS; llama-server was not changed.
- Tests: `133 passed` for `tests/test_shadow_autopilot_daemon.py` under an
  ephemeral `uv` environment; `py_compile` and `git diff --check` passed.
- Docs impact: `DOCS_UPDATED` in `docs/race_evidence_inventory.md` for cadence,
  controls, odds preservation, and the operator pause switch.
