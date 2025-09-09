# CPU Diagnostics and Git Swarm Remediation Plan

Intent
- Re-run CPU diagnostics, identify any runaway git/Warp processes, and apply safe mitigations to prevent a recurrence while preserving repository functionality.

Environment
- OS: macOS (Darwin), shell: zsh
- Repo: /Users/test/Desktop/greyhound_racing_collector
- Guidance: See WARP.md (turn off Warp Agent Mode > Codebase context auto-indexing; renice +19 as needed)
- Existing tooling: scripts/prevent_git_swarm.sh, GIT_SWARM_PREVENTION.md, enhanced .gitignore

Plan (phased)
A. Baseline snapshot (read-only, to /tmp)
- uptime; top -l 1 -o cpu; vm_stat; ps auxww | grep -E "git (diff|status|ls-files|rev-parse)"
- Save to SNAPDIR=$(mktemp -d /tmp/ghr_diag_XXXXXX)

B. Attribute and triage
- Trace PPID chains for git diff processes to confirm Warp parent
- Deprioritize Warp: renice +19 -p <warp_pid>
- Kill only git children of Warp (TERM then KILL)

C. Apply mitigations per WARP.md
- Disable Warp Agent Mode codebase auto-indexing for this repo and restart Warp
- Optional: reduce prompt-induced git status for this repo (p10k/starship/oh-my-zsh local config)
- Optional repo-local tweak: git config --local status.showUntrackedFiles no

D. Validate repo functionality
- Python venv setup; pip install deps; python scripts/monitor_system_health.py --detailed; pytest -q

E. Post-mitigation snapshot
- Repeat A, confirm no new git diff swarm; record CPU stable

F. Rollback
- Steps to undo prompt/warp local settings and repo-local git config

Notes
- Do not commit diagnostics; keep under /tmp
- Prefer existing docs and scripts; avoid creating new files in the repo

