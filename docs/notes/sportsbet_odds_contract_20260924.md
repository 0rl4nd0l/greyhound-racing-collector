# Sportsbet odds acquisition: existing capture and receipt contract

Date: 2026-09-24. Scope: source inspection of the isolated engineering checkout,
with synthetic test source inspection only. No provider operations, historical
race payload reads, databases, outcome inspection, or runtime changes were used
for this contract review. Line references describe the inspected baseline and
may move as the integration changes. Endpoint discovery and recorder results
are documented separately by the engineering task.

## Finding

The current capture path does not consume structured WIN/PLACE responses.
It discovers a race through browser links, then extracts rendered runner rows.
The only structured Sportsbet data paths found in the inspected capture and
metadata modules are NextEvents metadata and JSON-LD race discovery. Neither
establishes a fixed WIN/PLACE response contract.

An odds adapter cannot yet be justified by this code evidence alone. A passive
recorder is a useful bounded candidate; a production adapter additionally needs
an observed response with event/market/runner identity, state and complete-price
semantics. In particular, merely constructing accepted rows from an invented
JSON shape would prove fixture compatibility, not provider compatibility.

## Current acquisition

- `odds_auto_integrator.py:428-552`: `fetch_odds_for_target_race` creates the
  guarded browser, visits the greyhound landing page, waits five seconds,
  chooses an exact venue/race link or resolves it from a meeting, and delegates
  to `get_race_odds_from_page`. The result has `success`, counts, `race_info`,
  WIN `odds_data`, and PLACE rows nested in `race_info.odds_data_place`.
  Source recovery is accepted only through the supplied validation callback.
- `sportsbet_odds_integrator.py:1430-1624`: race-page navigation, document-ready
  wait (up to ten seconds), a further five-second sleep, then runner-card,
  separate-button, table and generic DOM extraction strategies. Where needed,
  a market interaction is followed by another paired-row extraction.
- `sportsbet_odds_integrator.py:218-273`: paired fixed prices come from the
  final two decimal strings before an `EW` control in one runner row. Both
  must exceed one, and PLACE must not exceed WIN. This is rendered-row
  evidence, not a network market identifier.
- `sportsbet_odds_integrator.py:768-833`: JSON-LD reads SportsEvent name, URL
  and start date; it initializes empty `odds_data` for later page scraping.
- `utils/prejump_sportsbet.py:26-28,219-283`: guarded GET to
  `https://www.sportsbet.com.au/apigw/sportsbook-racing/Sportsbook/Racing/NextEvents`
  supplies one metadata snapshot. Its local observation timestamp and hash
  must not be described as provider odds timestamps.
- `utils/sportsbet_browser.py:70-151`: existing CDP response instrumentation
  handles HTTP status and denial coordination; it does not extract odds
  response bodies. It owns its observer connection and fails closed when
  observation is lost. A second listener must not replace that behavior.

Discovery is still a separate operation and source of overhead. Reading a
race-page response earlier does not remove landing/meeting discovery or prove
that NextEvents provides sufficient discovery coverage.

## Contract an adapter must preserve

| Concern | Existing requirement and source | Structured-response implication |
| --- | --- | --- |
| Runner identity | `autonomous_live_odds_capture.py:1438-1579`: named runner plus box, compared as normalized `(box, name)` against expected runners | Provider runner ID alone is insufficient; preserve the canonical cross-source mapping and retain provider ID as additional evidence. |
| Box provenance | `autonomous_live_odds_capture.py:54,1452,1839-1842`: only `explicit_dom` and `runner_text` accepted | Add an honest, tested structured provenance type if justified. Never relabel a JSON field as DOM/text evidence. |
| Both markets | `autonomous_live_odds_capture.py:52,1582-1654`: complete WIN and PLACE sets required | Do not downgrade to WIN-only or synthesize PLACE odds from WIN. |
| Coverage | `autonomous_live_odds_capture.py:1496-1579`: rejects duplicates, missing active runners, unexpected runners and priced scratches | Require a complete snapshot or a proven complete stream state before invoking normal validation. |
| Scratchings | `autonomous_live_odds_capture.py:1684-1731`: explicit scratch flags/statuses remove expected runners from active set | A newly scratched provider runner needs reconciliation against the canonical expected set; absence alone is not scratch evidence. |
| Price validity | `on_demand.py:1127-1171`: decimal odds finite and greater than one; boxes 1–10; nonempty names; WIN/PLACE identity sets equal | Reject suspension, closed markets, absent prices and nonfinite numbers before creating accepted rows. No last-good-price fill. |
| Minimum field | `on_demand.py:1103-1110`: at least two unique runner identities for receipt hash | An empty or single-runner result is not a usable receipt. |
| Source URL | `receipt_preflight.py:55-83`: HTTPS Sportsbet race-page URL, supported path, matching venue/race, no query or fragment | Keep the accepted page URL as source identity and record the API route separately. Substituting an API URL breaks admission. |
| Receipt age | `on_demand.py:1184-1229`: timezone-aware capture timestamp, nonnegative age within caller limit, exactly one APPENDED/PASS attempt, matching hashes | Original observation time must remain distinguishable from parsing, append and emission times. |
| No outcomes | `on_demand.py:1203-1205`: source report rejected if it contains outcome fields | Keep only bounded, allowlisted pre-race response projections; do not attach unfiltered network bodies to reports. |
| Reservations | `autonomous_live_odds_capture.py:2710-2758`: existing reservation required before fetch; durable allowance starts before acquisition | Adapter stays inside the authorized fetch. No independent requests, second state store or recovery fallback. |
| Expiry | `autonomous_live_odds_capture.py:2769-2790`: re-evaluate time and bind original reservation before append | A faster response cannot justify changing the reserved race/window or appending after expiry. |
| Publication | `autonomous_live_odds_capture.py:2823-2883`: normal append and exact receipt publication precede retention | Reuse this chain instead of creating an alternate receipt publisher. |

Two existing limitations need explicit integration handling rather than silent
adaptation:

1. `append_validated_capture` currently passes append-time `current_time` into
   both market timestamps (`autonomous_live_odds_capture.py:2499-2567`), and
   `receipt_from_handoff` reads `append_timestamp` as capture time
   (`on_demand.py:1188`). A structured response may arrive substantially earlier.
   Preserve its actual local observation time and propagate it deliberately;
   do not call append time a provider timestamp or re-date stale prices.
2. PLACE persistence currently uses `DEFAULT_PLACE_TOPN = 3`
   (`autonomous_live_odds_capture.py:53,2528`); rendered paired extraction also
   assigns three (`sportsbet_odds_integrator.py:1592,1609`). That is not evidence
   of provider place terms. A response with different terms must fail closed
   until terms can be carried faithfully through persistence and receipts.

Event IDs, market IDs, stable runner IDs, suspension/closure state, provider
timestamps, update sequences and reconnect semantics are not established by
these accepted-row fields. The adapter must validate those facts from actual
source evidence before flattening data into the existing receipt format.

## Reusable validation seams

Use `validate_fetched_odds(plan, fetch_result)` followed by
`normalize_validation_receipt(...)` in network-denied adapter tests; successful
JSON decoding alone is insufficient. For complete receipt handoff tests, also
use `receipt_from_handoff`, which verifies freshness, unique accepted attempt
and source hashes. Keep acquisition expiry/identity tests at
`execute_capture_plan`/`AttemptAllowance` rather than mirroring parser logic.

Existing synthetic test examples suitable for extension are:

- `tests/test_autonomous_live_odds_capture.py:1337`: both markets persisted;
  `:1952`: missing PLACE rejected; `:2001`: reserve runner mapped to final box;
  `:2099`: explicit scratched runner may be absent; `:2190`: partial WIN while
  PLACE complete; `:2330`: extra identity mismatch; `:2380`: priced scratch
  rejected; `:2541`: ambiguous box provenance rejected.
- `tests/test_sportsbet_odds_safety.py:578,689,718`: paired-render extraction
  and safe recognition of explicit WIN/PLACE row semantics.
- `tests/test_capture_reservation_expiry.py:23,44`: expiry before consumption
  and exclusive native window boundaries.
- `tests/test_sportsbet_access.py:9,82,223`: durable denial, pre-browser gate
  and draining denial before navigation.

These are source-inspected test seams, not newly executed validation results.
No new adapter or provider contract has been validated by this document.

## Timing and integration consequences

Source code establishes two five-second waits in a normal landing-plus-race
capture and a race-page readiness wait. Those are configured waits, not measured
latency savings; the response could arrive before, during or after them.
Measure browser startup, navigation, usable response arrival, rendered
extraction completion, parsing and receipt verification separately in the same
authorized observation. A passive listener adds no logical provider operation,
but normal browser requests and the existing source operation remain charged.

The narrow eventual seam is the acquisition portion of
`SportsbetOddsIntegrator.get_race_odds_from_page`, reached through the existing
`fetch_odds_for_target_race` path. Integrating there potentially touches
`sportsbet_odds_integrator.py`, `odds_auto_integrator.py`,
`utils/sportsbet_browser.py` and `scripts/autonomous_live_odds_capture.py`, which
overlap collector coordination work. Source provenance, capture-time and PLACE
terms changes require tests before receipt acceptance. Leave selection
default-off, no unaccounted fallback, and use the main agent's explicit handoff
before any live attachment or separately reserved page load.
