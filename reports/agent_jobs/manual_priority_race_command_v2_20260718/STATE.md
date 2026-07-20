# State

## Before

Issue 50 was open. `origin/master` contained PR 45 resource isolation, while
reviewed PR 46 and PR 47 were unmerged sibling heads. There was no standalone
exact-race operator command.

## After

The clean lane preserves PR 45 and merges the exact reviewed PR 46 and PR 47
heads once each. It provides a plan-only-default command with exact-race
selection, target-only refresh, explicit double-gated collection, bounded
shared-lock waiting, T-60/T-30/T-10/T-2 capture windows, strict source and
runner validation, append-only/idempotent capture, ephemeral fresh feature
sealing, and canonical non-persisted full/half prediction output.

The live plan resolved `Race 1 - BAL - 2026-07-18`. The one permitted execution
waited for the existing daemon lock and then failed closed because Sportsbet
returned no source URL and zero accepted WIN/PLACE runner rows. The command
reported `BLOCKED_RUNNER_IDENTITY`; the canonical database contains zero
`live_odds` rows for that race. No prediction was produced or persisted.

Owner-facing stop: `BLOCKED`.
