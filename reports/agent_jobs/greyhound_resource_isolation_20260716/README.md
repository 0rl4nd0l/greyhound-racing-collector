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
