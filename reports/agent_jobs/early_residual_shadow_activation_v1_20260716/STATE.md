# State

- State before: `reviewed_pr47_handoff_not_live_and_external_prediction_starved_by_continuous_lock_handoff`
- State after: `early_residual_stage_installed_model_pin_repaired_first_live_append_waiting_for_capture_window`
- Outcome: `ADVANCED`
- Production model state: unchanged market-only baseline
- Frozen residual artifact: unchanged
- Prospective outcomes inspected: no
- Cohort cutoff assigned: no
- Service stop, start, or restart: no
- Timer frequency or capture-window change: no
- Deployment, promotion, betting, or merge: no

The early residual scorer now runs inside the odds-only process after a
successful strict pre-jump capture and before that process releases the shared
lock. The stage consumes the reviewed PR #47 handoff, generates hash-bound
features through the existing SQLite `mode=ro` path, and idempotently appends
one outcome-free frozen full/half residual record.

The first live capture at 23:08 exposed one fail-closed configuration defect:
the odds-only generated unit did not receive the Stage-2 feature-model path
already pinned in the full service. Commit
`cbbe78a2103da18a381263a9a2874ce02f243fbf` wires that exact model through the
CLI, generator, and early plan. Its SHA-256 is
`d7e9ff35b383a0e6400bcb67bcf6df374e4c0bfe6c974f32d1c9f057876e471d`.

The repaired unit was installed at 23:21:53 AEST and systemd metadata was
reloaded without interrupting a process. The ordinary 23:22 invocation loaded
the new unit and exited `WAITING_FOR_FUTURE_WINDOW`; the feed had rolled to
tomorrow and reports the next real capture target at 2026-07-17 09:55 AEST.
The first outcome-free JSONL append therefore remains an observational gate,
not a claimed success.

A read-only replay of only the repaired plan builder against the exact 23:08
handoff changed its status from `BLOCKED(feature_model_missing)` to `READY`
for `Race 12 - MAND - 2026-07-16`. The score command was not executed after
jump and no prospective outcome was read.
