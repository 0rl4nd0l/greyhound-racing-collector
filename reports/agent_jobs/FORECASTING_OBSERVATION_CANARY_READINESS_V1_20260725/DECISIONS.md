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
- Apply the fresh review's fail-closed repairs before publication: bind exact
  configuration bytes, reject hard-link aliases, authenticate release bundle
  contracts before receipt 1, keep activation time authority internal, and bind
  recovery replay to the current snapshot, inventory, and replica root.
- Treat the first complete regression run as non-green because its final
  resume-through-main test returned fail-closed code 69 once, despite that exact
  test subsequently passing 21/21 focused repetitions and the exact
  preceding-file sequence.
- Stop the second complete-suite attempt at the owner's direction for time. Its
  approximately 72% failure-free progress is partial evidence only, not a
  complete passing-suite result.
- Publish only a draft PR with authoritative full-suite GitHub CI confirmation
  as a mandatory pre-merge gate. Do not claim merge or runtime readiness.
