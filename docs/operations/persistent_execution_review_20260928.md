# Persistent execution integration review

Initial fixed candidate: `6f8d7562` against tested baseline `8b5552c7`.
The review focused on persistent additions relative to `8f85f850`; inherited
research analyses and target-result artifacts were not opened or re-evaluated.
Both review axes ran independently. No provider, real database or runtime writes.

## Standards

1. **P2 — Closure publication can leave unresolved jobs permanently nonterminal.**
   `scripts/run_comparison_result_queue.py` publishes the closure directory before
   committing `DEADLINE_UNRESOLVED` states and the closure event. A crash between
   those operations reaches the existing-directory branch on restart, which
   returns `CLOSURE_SEALED` without reconciling the queue. This conflicts with
   ADR-0004's transactional workflow-state requirement and ADR-0008's explicit
   terminal lifecycle. Make existing-closure recovery idempotently reconcile
   queue state/event; test interruption after publication but before commit.
2. **Documented architecture deviation — JSON/filesystem claims are authoritative
   scheduler state.** Configuration identity, slot directories and terminal JSON
   files determine admission and recovery, separately from result-queue SQLite.
   ADR-0004 requires one transactional workflow store and says JSON reports are
   never authoritative. A scoped documented exception with crash-safety evidence
   resolves this mismatch without a broad redesign. This finding alone does not
   establish duplicate dispatch: exclusive claims and the scheduler lock provide
   meaningful protection.

No additional actionable smell-baseline findings.

## Spec

1. **P1 — Every scheduled preparation passes an unserializable value.**
   The scheduler passes `Path(cfg['reconciliation_roots'])` into `prepare()`,
   which stores it directly in the canonical JSON plan. The CLI correctly loads
   the referenced JSON first. The bug consumes the slot/source allocation but
   prevents launch, violating the required existing-supervisor integration and
   operation without an open Codex session.
2. **P1 — Programme capture storage is incompatible with the existing contract.**
   Preparation moves `capture.sqlite3` beneath the programme `prediction_root`,
   but `FreshnessContract` still requires the old campaign prediction path.
   Every programme session consequently fails
   `operational_database_separation_required`. Carry the authenticated storage
   binding through the actual supervisor/contract seam; preserve separation.
3. **P2 — Cumulative elapsed-time accounting changes for existing campaigns.**
   `Campaign.close()` caps recorded elapsed time at the initial reservation,
   including when no programme is configured. Cleanup overrun is hidden relative
   to tested behavior. Preserve actual elapsed time; any different charging
   policy requires an explicit prospective scope and retained actual duration.

Initial totals: Standards 2 findings (worst: crash-inconsistent closure state);
Spec 3 findings (worst: two unconditional persistent-launch blockers).

## Additional operational integration findings

- Unique programme/slot package names, fixed 13:00 slot admission, final in-flight
  cleanup timeout and restoration after a pre-terminal crash needed explicit
  wiring; a mocked scheduler test did not establish the real package boundary.
- Separate prediction/result request budgets must reserve owed-result capacity.
- Fresh startup must accept an absent, never-created prediction store; disappearance
  after prior work must remain an integrity failure.
- Result work must defer shared-owner contention before charging a request.
- Preparation needs an up-to-date, hash-bound reconciliation-root inventory.
  New owned packages join later inventories prospectively; old roots are retained.
- Missing mount, dirty source and modified prepared authority must fail before
  admission or approval materialization. Monitoring is structural only.

Fixes and final exported verification are recorded in the sealed deployment
receipt and final handoff. Initial findings remain evidence, not a release claim.
