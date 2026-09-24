# Odds.com.au receipt integration contract

Offline source review, 2026-09-24, baseline
`5e32c21e9930c78b977f8e2ee474ceeae212a5c8` (PR #189). This review read code,
instructions and synthetic test source only: no network requests, databases,
historical payloads, results or runtime state. It makes no claim about the
Odds.com.au provider contract; the bounded observation report owns that evidence.

## Verdict

The current full receipt pipeline cannot honestly accept Odds.com.au acquisition
without explicit provenance changes. The low-level normalized receipt function
can represent its URL, but that alone neither validates the provider nor proves
collector/R3 admission. An Odds.com.au response carrying a Sportsbet price is
still Odds.com.au acquisition, not direct Sportsbet evidence.

Do not substitute a Sportsbet URL, label structured boxes as DOM evidence, or
set persisted `source="sportsbet"` to make existing validators pass. Preserve
Sportsbet's source STOP; it is neither reset nor spent by this offline review.

## Current boundaries

| Seam | Inspected source | Consequence |
| --- | --- | --- |
| Acquisition validation | `scripts/autonomous_live_odds_capture.py:1582` (`validate_fetched_odds`), `:1372` (`is_sportsbet_source_url`) | Odds.com.au URL rejected as `sportsbet_source_url_not_sportsbet`. Race number is checked when provided; an adapter must additionally prove event/date/venue mapping. |
| Runner identity | Same file `:1438` (`normalize_fetched_row`), `:1496` (`validate_fetched_market_rows`) | Box/name identities must match the canonical active set. Accepted box provenance is `explicit_dom` or `runner_text`; structured source identity needs its own honest type. Missing, extra, duplicate and priced scratched runners fail. |
| Persisted provenance | `sportsbet_odds_integrator.py:1205` (`append_pre_jump_odds_snapshot`), especially `:1339` | Append hardcodes `source="sportsbet"`. Stored fields have no independent acquisition-source/bookmaker-origin pair. This path cannot be used unchanged. |
| PLACE terms and time | `scripts/autonomous_live_odds_capture.py:2499` (`append_validated_capture`) | PLACE `topN` is fixed at three; both markets are stamped with append-time `current_time`. Neither proves provider terms nor response observation time. |
| Normalized receipt | `src/predictor/on_demand.py:1113` (`normalize_validation_receipt`) | Requires PASS and finite decimal WIN/PLACE prices greater than one, equal identity sets, boxes 1–10 and at least two runners. Copies `source_url` and caller `source_kind`, but discards extra source metadata and provider IDs from rows. It does not independently prove complete provider coverage, suspension, freshness or place terms. |
| Handoff | Same file `:1184` (`receipt_from_handoff`) | Uses `append_timestamp` as `captured_at`; hardcodes `source_kind="verified_autonomous_receipt"`; requires exactly one matching APPENDED/PASS attempt and matching report/form/sidecar hashes. Here source_kind describes the verification path, not the acquisition host. |
| Exact receipt schema | Same file `:605–676`; `race_collection/synchronous_manual_capture.py:2277–2299` | Exact receipt/handoff key sets and timestamp equality are enforced. Adding acquisition fields or redefining capture time requires deliberate compatible schema changes and actual verifier tests. |
| R3 source admission | `src/predictor/receipt_preflight.py:55` and `:210` | Exact Sportsbet HTTPS race-page URL and venue/race match required. An Odds.com.au receipt fails this admission even if normalization succeeds. |
| Research trust | `accuracy_program/odds_provenance.py:18`, `:511` | Trusted source set is Sportsbet-only. Leave this unchanged: alternative acquisition does not automatically become eligible for existing research cohorts. |

## Smallest honest integration, conditional on a verified response

1. Add an explicit default-off acquisition selector within the existing capture
   path. Keep original reservations, finite request accounting, time gates and
   publisher; do not create another scheduler or silently fall back between
   providers. `execute_capture_plan` at `autonomous_live_odds_capture.py:2710–2787`
   consumes before fetch, rechecks expiry, and binds the original reservation.
   Source-specific denial must stop its operation; Sportsbet's durable STOP must
   remain untouched.
2. Validate a complete single-bookmaker snapshot before producing accepted rows.
   Carry `acquisition_source="odds.com.au"`, its exact source-page URL, and a
   separately validated `price_origin_bookmaker` (plus provider bookmaker ID).
   Do not infer origin from URL or select different bookmakers per runner or
   market. Preserve event/market/runner IDs in hash-bound evidence and map each
   runner to the existing canonical box/name identity.
3. Add an explicit source-aware validation branch, retaining all current runner
   and two-market requirements. Use source-neutral structured box provenance.
   Absence is not scratching; stale last-good prices are not current prices.
   Suspended, closed, partial, unknown-state or mixed-version data must remain
   unavailable until source evidence supports a precise alternative.
4. Preserve original local response observation time separately from append and
   receipt emission times, with request time and monotonic timing where captured.
   Keep provider timestamp nullable unless observed and semantically understood.
   Carry proven PLACE terms; do not inherit the existing three-place default.
5. Extend existing persistence/report/receipt provenance explicitly, with a
   versioned contract if new exact-schema fields are necessary. Persist
   acquisition source and bookmaker origin separately and ensure normalization
   and handoff do not discard them. Refuse unsupported schemas instead of
   silently omitting fields because an old table lacks columns.
6. Test the existing publisher and receipt verifier end to end with isolated
   synthetic storage before claiming receipt integration. R3 source admission
   and research trust are separate decisions; leave research cohorts unchanged
   and report R3 admission as unsupported unless explicitly implemented and
   validated within the agreed engineering scope.

An offline parser or inspection projection can be useful before these changes,
but must not advertise itself as a production-compatible odds adapter.

## Meaningful offline checks

Reuse `validate_fetched_odds`, `normalize_validation_receipt`,
`receipt_from_handoff`, and exact receipt admission as appropriate. A direct
call to normalization with a fabricated PASS bypasses source validation and
does not establish integration. PR #189 already demonstrates this distinction
in `tests/test_sportsbet_response_inspection.py:293`: schema inspection output
is rejected by the actual receipt normalizer.

Relevant synthetic seams are `tests/test_autonomous_live_odds_capture.py`
(complete markets, scratches, missing/extra runners and invalid box evidence),
`tests/test_capture_reservation_expiry.py` (expiry and reservation identity),
`tests/test_predict_race_now.py:308,877,2202` (handoff normalization), and
`tests/race_collection/test_manual_prediction_collector_request.py` (publisher
and collector protocol). Add source/origin spoofing, mixed bookmakers,
observation-vs-append age, missing PLACE terms, denied/partial responses and
wrong source URL rejection where justified by the observed response contract.

This source review did not execute tests. No live capability, savings,
reliability or receipt acceptance is established by it.
