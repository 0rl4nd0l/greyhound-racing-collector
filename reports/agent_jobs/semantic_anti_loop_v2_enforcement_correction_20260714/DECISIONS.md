# Decisions

- Require V2 mechanically on both Greyhound `PreToolUse` and `Stop` hooks.
- Use the approved `$HOME/.codex` portable guard, not the stale physical
  `$HOME/.agents` copy.
- Keep task state separate from decision state. A claimed run writes an outcome
  and one candidate; normal registry release alone publishes it under lock.
- Reserve standalone ledger append for an explicitly authorized unclaimed
  seed. Validate the four existing Greyhound seeds without replay.
- Keep prospective evidence blocks transition-specific and preserve offline
  research fitting unless an explicit dependency says otherwise.
- Record `FIRST_FIVE_REVIEW_PASSED` from five distinct scopes; preserve the two
  legacy duplicate chains as append-only evidence.
- Treat broader adoption as a separate review, not an automatic consequence of
  the Greyhound result.
- Do not create a continuation goal for this completed transition.
