# Official result identity contract

This path labels no race by name alone. Its canonical pre-jump identity is the
single preserved Sportsbet win-odds tuple:

`(race_id, race_date, venue, race_number, Sportsbet source URL, latest capture timestamp, complete box+runner set)`.

An official result is accepted only when all of these gates pass:

1. Every win-odds row is Sportsbet evidence from a declared autonomous pre-jump
   capture, with a source URL, timezone-aware timestamp before the scheduled
   start, valid decimal price, and mutually consistent race date, venue and
   number.
2. The textual `race_id` components exactly equal those stored components.
3. Venue maps through the explicit repository TheDogs mapping, or its preserved
   venue slug exactly equals the Sportsbet URL slug. No fuzzy alias is used.
4. The HTTPS TheDogs URL path encodes exactly that venue slug, date and race
   number. Redirects and multiple candidate URLs are rejected.
5. The official table contains one named runner per box and a complete,
   contiguous finish order. Terminal/partial results are rejected.
6. Official and latest pre-jump Sportsbet runner sets agree exactly on both box
   and normalized full name. Names are normalized only after box identity is
   fixed; collisions on either side are rejected.

The raw HTTP response is stored byte-for-byte under its SHA-256. The evidence
rows retain that hash and artifact path, fetch timestamp, official and
Sportsbet URLs, and a hash of all sealed odds identity rows. Inserts target only
the append-only `autonomous_official_result_evidence_*` tables. An identical
re-run is a no-op only after the complete stored race and runner bundle and its
provenance are reverified. Conflicting, incomplete, or orphaned existing
evidence fails closed and is never altered. New race and runner rows commit
together or roll back together.

`scripts/append_august_official_results.py` is dry-run by default. `--execute`
only authorizes the evidence-table append; it never writes `race_metadata`,
`dog_race_data`, odds, features, labels, models, or runtime/service state.
Offline `--raw-html` use also requires the preserved official `--source-url`
and a timezone-aware `--fetched-at`; filesystem timestamps are not treated as
official fetch provenance.
