# Decisions

- Preserve the dirty launch checkout and implement from exact canonical in a
  clean sibling worktree.
- Keep migrations 0001 through 0029 unchanged; protect the legacy DB at the
  operator boundary and prove a fresh separate DB reaches exactly schema 29.
- Reuse existing operational, evaluation, and recovery authorities rather than
  write persistence SQL in the operator CLI.
- Represent result-blind observation as an immutable runtime-input mode that
  plans the existing closed order but executes only the prefix ending at
  deferred prediction.
- Authenticate the complete declared champion/challenger cohort at adapter
  construction, before receipt 1.
- Keep runtime proof explicitly `DATA_MISSING`; repository tests do not prove a
  live canary.
