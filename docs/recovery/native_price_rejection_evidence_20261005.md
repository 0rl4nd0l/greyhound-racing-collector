# Retain rejected native scratch/price evidence

The October5 SHEP3, NOR9 and LCTN1 exclusions record
`scratched_runner_has_active_price`, but their failed odds-page/API responses
were not retained. Their primary pages alone cannot prove whether the separate
odds page/API contradicted each other or a parser associated rows incorrectly.
The original failures stay excluded and cannot be reconstructed by a new fetch.

This prospective change catches only that exact typed error after the existing
response, identity, time-order and JSON checks. It saves the already received
odds HTML and API bytes, bounded safe HTTP receipts, primary-page receipt/hash,
expected active IDs/boxes and recorded jump. It re-raises the same original
error. It does not retry, fetch, infer an identity, admit a race or change request
accounting. Other errors, including denials, retain existing handling.

The content-addressed receipt lives in the worker's
`source_evidence/native_price_rejections` directory (0700); its immutable file
is0400. Each received body is bounded16MiB and the complete receipt48MiB.
Headers are allowlisted; cookies, authorization and request headers are omitted.
Only a schema/path/SHA reference enters the normalization report. A failed
retention attempt adds the fixed `FAILED_NO_REJECTION_EVIDENCE` status while
preserving the original rejection; a partial failed write is not overwritten.

`inspect_native_price_rejection()` is an offline diagnostic reader. It checks
safe path, privacy, size/hash/body bindings and replays the same typed error;
its scalar verdict is `REJECTION_REPRODUCED_NOT_ADMISSIBLE`. It is never called
by the admission verifier. The stored primary receipt is not a substitute for
the separately retained primary body. This inspector does not certify full
pre-jump provenance or qualify source clocks; those remain separate checks.

Focused synthetic tests cover the real identity-capture seam, exact two response
bytes, no added requests, private permissions, immutable repeat, header redaction,
body/receipt bounds, path/symlink/hardlink/hash mutation, denial and unknown error
precedence, and unchanged rejected index publication. The baseline test fails
because no rejection reference exists; the repaired path preserves it. No
historical response bodies were manufactured, no real source requests were made,
and all nine feature-generator files and frozen models remain unchanged.

Installation is separate and requires root's coordinated quiet opportunity.
