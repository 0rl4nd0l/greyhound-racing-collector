# Manual priority capture handoff V1

The standalone named-race command can now consume one finalized autonomous
fixed-window capture without taking the daemon writer lock. It waits for the
producer's final marker, binds the exact plan/report/form/sidecar bytes to the
exact WIN and PLACE SQLite rows in query-only mode, stages the accepted bytes
privately, and retains the original direct-capture fallback unless
`--require-autonomous-handoff` is selected.

Implementation commit: `2c41b1df3c216b8cb75a788725853cde456cc4a9` on
`codex/manual-priority-capture-handoff-v1-20260718`, based on
`5c2356431a96659a7cd68edd5b94b50a5877de84`.

One real pre-jump Gunnedah receipt was verified read-only against 16 exact odds
rows. The prediction itself was not run. Independent review showed that fresh
feature sealing would read historical `finish_position` and `placing` columns,
which this card's production boundary does not authorize. Terminal stop:
`BLOCKED_TASK_CONTRACT`.

