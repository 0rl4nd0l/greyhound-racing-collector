# State

- State before: `live odds-only cycles run the full reporting tail and repeat database and browser setup per captured race`
- State after: `one reviewed stacked implementation provides a separate lightweight child contract, one history load and browser session per batch, and durable low-priority service controls without changing capture semantics`
- Outcome: `ADVANCED`
- Status: `DONE_WITH_RISK`
- Base: PR #48 exact head `f776bfd142b1e8acd3befca330eee36f490402ed`
- Production data inspected or mutated: no
- Installed service or timer changed: no
- Live daemon stopped, started, reloaded, or restarted: no
- Capture windows, timer frequency, and 16-race cap changed: no
- Outcomes, training, promotion, activation, betting, push, merge, or GitHub mutation: no

The stacked implementation adds an explicit lightweight odds child that exits
after candidate refresh, strict odds capture, and residual handoff. The
15-minute full daemon retains its complete dashboard, cumulative history,
drift, readiness, daily status, join, and aggregate contract.

Early residual feature construction now loads the frozen model and read-only
SQLite history once for the batch, while preserving isolated per-race outputs,
hash-bound inputs, fail-closed scoring, and completion before shared-lock
release. Sportsbet capture similarly reuses one browser session within the
bounded cycle and resets that session after a timeout or driver exception.

The generated odds-only unit retains `Nice=10` and adds durable CPU and I/O
weights plus best-effort priority 7. No unit was installed, so the observed PC
latency improvement remains a separate runtime-proof gate.
