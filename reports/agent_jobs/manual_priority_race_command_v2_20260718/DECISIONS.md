# Decisions

- Keep plan-only as the default and require both `--execute-collection` and
  `--allow-auto-scrape-odds` for collection.
- Reuse the reviewed daemon lock and odds collector; do not create a second
  lock or capture implementation.
- Refresh collection time after lock acquisition so a bounded wait cannot use
  a stale pre-window timestamp.
- Route dependency diagnostics to stderr and reserve stdout for one canonical
  JSON object.
- Treat the live Sportsbet source/runner failure as a blocker. Do not retry,
  relax identity checks, or synthesize odds.
- Leave PR 48, its worktree, service, and timer unchanged. Recommend a separate
  owner-approved lane later decide whether the legacy activation branch should
  be retired or superseded after this command's eventual live proof.
