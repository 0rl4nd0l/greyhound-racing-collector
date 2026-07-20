# Decisions

1. Reuse only a producer-finalized report, never progress JSONL or a skip-only
   report.
2. Discover plans first, require one unique latest exact target/window, and
   block fallback past a newer finalized invalid target candidate.
3. Bind both WIN and PLACE report values losslessly to one SQLite query-only
   snapshot using exact IEEE-754 value tokens.
4. Treat the evidence claim as `HASH_SEALED_DB_BOUND_AT_USE_TIME`; do not claim
   independent historical authentication.
5. Stage plan, report, capture-time form, and adjacent sidecar bytes privately
   and verify all four hashes before feature sealing.
6. Recheck the due fixed window before feature sealing and immediately before
   scoring.
7. Provide `--require-autonomous-handoff` for a proof mode that can never take
   the writer lock or scrape/append.
8. Do not run the optional live score under this card: the existing feature
   loader reads historical result columns beyond the authorized live boundary.
9. Recommend a later owner-approved disposition lane for PR #48 after this
   primitive is reviewed; do not alter PR #48, its worktree, service, or timer
   here.

