# Decisions

1. Make lightweight behavior explicit through
   `--odds-capture-lightweight`; do not infer it from skip flags or weaken the
   default full-autopilot contract.
2. Stop the lightweight child only after candidate refresh, strict capture,
   and residual handoff artifacts are written. Defer global reporting builders
   to the existing 15-minute full daemon.
3. Batch only shared setup: load the frozen model and read-only SQLite history
   once, but retain one isolated feature directory and result per race. One
   failed race must not suppress later races.
4. Reuse one Sportsbet browser within a bounded capture cycle. Reset it after
   timeout or exception, and retain all existing pre-fetch, URL, race identity,
   runner-set, WIN, PLACE, pre-append, and append-only checks per race.
5. Keep early residual feature generation and scoring inside the odds-only lock
   owner before lock release.
6. Preserve the timer, capture windows, 16-race cap, full daemon path, existing
   single-race CLI, and append-only odds functionality.
7. Use `Nice=10`, `CPUWeight=20`, `IOWeight=20`,
   `IOSchedulingClass=best-effort`, and `IOSchedulingPriority=7` for the
   time-sensitive odds-only unit. Idle I/O scheduling is not used.
8. Do not deploy or claim live latency improvement in this repo-only lane.
