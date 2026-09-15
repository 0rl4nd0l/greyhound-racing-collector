# R3 source-alias readiness repair — 2026-09-15

Scope: approved narrow offline repair at the public
`ResultAcquisitionReadiness.require` seam. The defect was a literal
`race_id` join between the caller JobInput and current collector precursor
rows, rejecting source-proven aliases such as SHEP/SHEPPARTON and
QOT/LADBROKES-Q-STRAIGHT.

The repair uses the existing strict
`utils.race_identity_equivalence.race_identity_equivalent` seam for precursor
prediction and feature rows. It requires caller/evidence structured identity
(date, race number, configured venue identity) and the exact canonical source
URL identity. Current-index ownership remains an exact `job_input.race_id`
match; no index alias expansion was added. Existing native runner identity,
runner-set hash, URL/date/race-number, participant/jump, freshness, bounded
read, immutable-source, outcome-blind, and fail-closed guards remain in place.
Mixed matched race IDs remain ambiguous and reject readiness. No extra source
reads, fallback paths, timeout relaxation, retries, production-data reads,
model/scorer/protocol changes, or live operations were performed.

Validation used only the pinned Python and synthetic fixtures at the public
seam. Full commands and results are retained in `repair-evidence/`.

- Alias regression plus cross-venue rejection: 3 passed, 33 deselected.
- Existing coordinator/readiness matrix: 36 passed.
- Affected bounded-alias integration guard: 1 passed, 42 deselected.
- `git diff --check`: passed.

The initial RED run was captured before production correction in
`repair-evidence/20260915-red-readiness-alias.txt`; its full terminal trace was
emitted interactively but was not separately redirected to a file.
