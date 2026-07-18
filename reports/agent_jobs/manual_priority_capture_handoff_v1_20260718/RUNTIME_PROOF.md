# Runtime proof

Only a read-only receipt compatibility probe was performed.

- Evidence root: retained autonomous runtime artifacts.
- Candidate at probe time: `Race 4 - GUNNEDAH - 2026-07-18`, still pre-jump at
  the recorded 2026-07-18 17:33:12 +10:00 probe time.
- Capture: T-60, producer run
  `20260718T171200+1000_daemon_autopilot`.
- Exact SQLite rows: 16 (complete WIN and PLACE group).
- Exact row hash after lossless normalization:
  `562b4ceb8c23710e1d2339d91bc6178eb422494fa59cff97c7a51db7c5e20fad`.
- Consistency claim: `HASH_SEALED_DB_BOUND_AT_USE_TIME`.
- Historical authentication: false.
- SQLite access: URI `mode=ro`, `PRAGMA query_only=ON`, one snapshot.

No scoring command was run because feature sealing would exceed the card's
production-data boundary. No lock, service, timer, PR #48 file, database row,
prediction artifact, or target outcome was touched.

Fresh GitHub verification retained origin/master
`c1dfd464cf6ecfb2034f96ac1a8d3ea58d4e6afa`; PR #46 head
`2c595d27ac748d3df8e4031d5491c76606c5be89` and PR #47 head
`0ae5937cde87131c714fb7383c58ce13e3cfbc06` remain open, draft, clean, with
five successful checks each. PR #48 remains open, draft, dirty/conflicted at
`f776bfd142b1e8acd3befca330eee36f490402ed` and read-only.

## Runtime Functionality Proof

- Intended output: one non-persisted normalized full/half prediction from one
  exact finalized autonomous receipt without writer-lock ownership.
- Live output location: stdout only; no prediction artifact.
- Pre-run max timestamp or count: 16 exact receipt WIN/PLACE rows.
- Post-run max timestamp or count: 16 exact receipt WIN/PLACE rows at the
  query-only receipt probe; the score itself was not started.
- Rows/files inserted or updated after run start: zero production rows or
  runtime files; temporary test-generated ignored files were removed.
- Readiness/gate status: `BLOCKED_TASK_CONTRACT` before feature sealing.
- Exact command/query used: `discover_capture_handoff` against the retained
  finalized evidence root and canonical SQLite URI `mode=ro`, with
  `PRAGMA query_only=ON`, for the exact race/window group.
- Result: `PARTIAL`.
- Remaining blocker: the existing feature builder reads historical result
  columns outside this card's production-data allowlist.
