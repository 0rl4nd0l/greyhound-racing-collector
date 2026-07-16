# Decisions

1. Run frozen residual scoring inside the existing odds-only lock owner,
   immediately after a successful PR #47 handoff and before lock release.
2. Reuse the full service's exact configured Stage-2 feature model. Do not
   discover, select, fit, tune, or compare another model.
3. Keep feature-history database access on the reviewed SQLite `mode=ro`
   command and persist only the idempotent append-only outcome-free shadow
   JSONL.
4. Preserve the existing minutely timer calendar, capture offsets, PR #45
   resource isolation, market-only production model, frozen residual
   parameters, full/half strengths, normalization, and thresholds.
5. Do not retroactively execute the scorer for Mandurah Race 12 after jump.
   Use its sealed handoff only to prove that the repaired plan is now `READY`.
6. Park the first-live-append proof until the next genuine eligible capture at
   2026-07-17 09:55 AEST. Do not fake time, bypass the lock, or manufacture a
   race to satisfy the gate.

Canonical integration order remains PR #47 first, followed by stacked draft
PR #48 retargeted to `master`. PR #45 is already merged and is an ancestor of
both heads; its resource-isolation settings must remain present.
