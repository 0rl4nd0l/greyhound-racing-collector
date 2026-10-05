# Prospective native identity retention recovery

The reviewed recovery preserves the old FAILED refresh, rejected index, consumed
requests, quarantines, prediction claims and deferrals. It does not classify the
unavailable old odds body as a safe exclusion or create an outage retry.

`persistent_reviewed_native_identity_retention_gap_v1` with disposition
`PROSPECTIVE_NATIVE_IDENTITY_RETENTION_CORRECTION` keeps the existing recovery
review fields and adds two hash-bound references: `diagnosis` (the original
`native_identity_failure_diagnosis_v1`) and `retention_gap_proof`.

The proof schema `persistent_native_identity_retention_gap_proof_v1` requires:

- `source_commit`: exact reviewed successor, distinct from `original_source_commit`.
- `diagnosis`, `invocation_id`, `refresh_report`, `failed_phase_number: 0`.
- `affected_races`: exact failed-download order of `{race_id, race_url, jump}`.
- `worker_request_timings`: every selected worker in original order, each with
  `{race_url, request_timing: {path, sha256}}`.
- `missing_odds_body: true`, `mismatch_worker_odds_api_requested: false`,
  `current_index_published: false`, `request_retries_added: 0`,
  `old_failure_disposition: FAILED_UNRESOLVED`.

The guard rechecks common terminal, lifecycle, checkpoint, report, publication,
cleanup and consumption bindings. Every HTTP request must have one matching GET
start/end pair with HTTP 200. Exactly one failed worker must have the exact
`native_identity_evidence_rejected:expected_native_runner_set_mismatch` failure.
Its transport must end at the exact odds page, with no odds API call or retained
native identity evidence. The absent old body remains a reviewed archival fact,
not an invented reconstruction. Other failed workers must have the exact known
canonical missing-participant rejection with native identity available.
All affected jumps must be before preparation's current timestamp. The original
failed report and all prior files remain immutable; only a new same-allocation
package can be prepared after separate root review and installation.

The actual Oct5 owner journal is 306,255 bytes, above the small authority
reader's 256 KiB ceiling. Recovery now uses a separate 4 MiB evidence ceiling
for hash-bound owner journals. Path, hash and all semantic checks remain; the
small authority-file ceiling does not change.
