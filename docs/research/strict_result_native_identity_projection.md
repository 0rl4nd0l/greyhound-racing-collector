# Strict native result identity projection

The full-order result serializer previously discarded native race-entry IDs,
including when earlier checks proved the entry-to-dog-profile bridge. This
preparation preserves that bridge without changing legacy closure acceptance,
queue state, result requests, evaluation authority, or historical evidence.

`frozen_result_participants` attaches a proof projection only after its existing
sealed metadata, native race/odds transport, active field and same-row profile
checks pass. The projection retains the metadata, native evidence and primary
race-page hashes, source URL, and exact entry/profile/box/name mapping, including
the original reserve box when present. Missing legacy evidence stays missing.

The official HTML parser retains a SHA-256 of the parsed Unicode markup encoded
as UTF-8. This is **not** asserted to be a raw HTTP response receipt or proof of
source access. Existing callers still authenticate source, transport, admission,
timestamp and completeness. Historical revalidation must independently bind
the exact authorized raw request/response/body and original sealed bundle; this
change does not perform or authorize that work.

`comparison_native_identity_projection` is a pure, non-mutating bridge join.
It requires complete unique entry/profile identity, consistent pre-jump proof,
an official parsed-markup hash, matching official profiles at effective boxes,
and the existing name, scratch and reserve checks. It emits no IDs for an
incomplete bridge. Known non-finisher handling remains its separate existing
path; this full-order projection does not invent finishing places.

Both live dry-run ingestion and retained reconciliation use the projection.
Their final race and runner rows carry `IDENTITY_VERIFIED` or explicit
`IDENTITY_INCOMPLETE` with a reason. Only verified rows retain native entry ID,
dog-profile ID and the source-proof projection. Append-only evidence tables
already preserve the complete row JSON, so no database migration is required.

This code is offline preparation. It neither repairs the historical 73 excluded
records nor makes an earlier CLOSED state sufficient for native-ID evaluation.
Original claims, partial private metrics, result artifacts and quarantines must
remain unchanged. No new evaluation or automatic retry is authorized.
