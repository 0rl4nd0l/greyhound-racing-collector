#!/usr/bin/env bash
# Non-destructive, read-only CPU check + safe remediation checklist
# Usage: bash scripts/cpu_check_again.sh

set -euo pipefail
OS="$(uname -s)"
SNAPDIR="$(mktemp -d /tmp/ghr_diag_XXXXXX)"
echo "Snapshot dir: $SNAPDIR"

echo "== Baseline ==" | tee "$SNAPDIR/baseline.log"
date '+%F %T' | tee -a "$SNAPDIR/baseline.log"
uptime | tee -a "$SNAPDIR/baseline.log"
if [[ "$OS" == "Darwin" ]]; then
  top -l 1 -o cpu > "$SNAPDIR/top.txt"
  vm_stat > "$SNAPDIR/vm_stat.txt"
else
  top -b -n1 -o %CPU > "$SNAPDIR/top.txt"
  free -m > "$SNAPDIR/memory.txt"
fi

# Capture git/warp processes (read-only)
ps auxww | grep -E "git (diff|status|ls-files|rev-parse)" | grep -v grep > "$SNAPDIR/git_processes.txt" || true
pgrep -fal "(WarpTerminal|warp)" > "$SNAPDIR/warp_pids.txt" || true

# Output quick summary
GIT_COUNT=$(wc -l < "$SNAPDIR/git_processes.txt" || echo 0)
echo "Git processes: $GIT_COUNT"
if [[ "$GIT_COUNT" -gt 0 ]]; then
  echo "Top few git processes:"; head -n 10 "$SNAPDIR/git_processes.txt" || true
fi

echo "== Next steps (manual, non-destructive) =="
echo "1) Consider deprioritizing Warp (WARP.md): renice +19 -p <warp_pid>"
echo "2) If many 'git diff' under Warp, terminate only those children (TERM then KILL)"
echo "3) Disable Warp Agent Mode > Codebase context auto-indexing for this repo and restart Warp"
echo "4) Optionally reduce prompt-induced VCS checks for this repo (see docs/CPU_REMEDIATION_PLAN.md)"
echo "5) Re-run: ./scripts/prevent_git_swarm.sh (read-only unless you confirm cleanup)"
echo "6) Re-check: uptime; and inspect $SNAPDIR/top.txt"

