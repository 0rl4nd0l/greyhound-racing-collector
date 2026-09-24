# Sportsbet response inspection: isolated handoff

## Decision and evidence boundary

No odds-acquisition adapter is justified yet. Current rendered-page extraction
is the only implemented fixed WIN/PLACE route supported by the inspected code.
Browser-response extraction is the next experiment to prepare, because an
already admitted browser can expose its normal responses without another
provider operation. Direct HTTP remains unproven; this is not a finding that no
API exists. No endpoint enumeration, HTTP replay, browser launch or provider
operation was performed by this task.

Verified GitHub heads on 24 September 2026:

- [PR #184](https://github.com/0rl4nd0l/greyhound-racing-collector/pull/184):
  `fe3d984a13a7446ca0af301dd736fed20bab0b96`.
- [PR #187](https://github.com/0rl4nd0l/greyhound-racing-collector/pull/187):
  `87a3d33928896b141e9b5137bcf098ea73c03cf2`, this branch's base.
- [PR #185](https://github.com/0rl4nd0l/greyhound-racing-collector/pull/185):
  `5f5ef82999675ae89c2100ed32f8abbe7d1669d8`.

Source inspection covered the current capture modules, shared browser and HTTP
controls, receipt code, and synthetic tests. Filename-only inspection of tracked
fixtures/samples and the existing campaign evidence found no identified Sportsbet
pre-race odds-response body or HAR. The tracked FastTrack HAR was not opened.
This is a scoped evidence gap, not an exhaustive assertion about all retained data.
Historical race payloads, results, account material and databases were not read.

The live source state was read without acquiring/mutating its operational lock:
`phase=STOP`, `recovery_attempts=1`. The main repair agent confirmed through the
existing Codex thread channel (`01a0d132-20ed-7e53-a42d-46ee66ec1526`) that a
second supervised launch encountered HTTP 429 at 13:21:25.139392 AEST and that
**no live handoff or further provider calls are authorized**. Its reported
campaign totals were 1,035 logical requests, 3/12 captures, 8/512 source operations;
these are the main agent's report, not measurements made by this task. Do not
infer the denied resource from those observations. No consumed allowance was
reset, reopened or replaced here.

## Exact known route versus missing contract

`utils/prejump_sportsbet.py` already sends GET to:

`https://www.sportsbet.com.au/apigw/sportsbook-racing/Sportsbook/Racing/NextEvents`

Non-secret parameters: `racingFilters=HR_DOMESTIC,HR_INTERNATIONAL,GH_DOMESTIC,GH_INTERNATIONAL,HA_DOMESTIC,HA_INTERNATIONAL`
and `groupByFilters=true`. This supports structured discovery/metadata only.
The current metadata extractor uses event identity, competition/race, start time
and track condition. It does not establish any WIN/PLACE market contract.

| Needed odds fact | Evidence established here |
| --- | --- |
| Exact odds route, method, parameters | Unknown; the test `/Events?eventId=123` route is fabricated, not a discovered endpoint. |
| Snapshot versus incremental delivery | Unknown; websocket frame counts are not stream parsing. |
| Event, market, runner identifiers and mapping | Unknown for odds. Metadata event IDs do not prove odds-market mapping. |
| Fixed WIN/PLACE, terms, active/scratched runners | Unknown for structured odds. Existing PLACE top-three default is not provider evidence. |
| Suspended/closed market and missing-price semantics | Unknown. Never carry forward last prices to fill missing data. |
| Decimal representation, timestamps, sequences | Unknown. Recorder timestamps are local callback observations only. |
| Complete coverage, ordering and reconnect behavior | Unknown. Schema shapes cannot establish completeness or a simultaneous snapshot. |

See [the source contract review](sportsbet_odds_contract_20260924.md) for actual
receipt requirements, including truthful box provenance, canonical page URL,
observation-time propagation, PLACE terms, exact runner sets and window identity.
No validators, receipt fields, models, research protocols or cohorts were changed.

## Implemented candidate: passive, default-off schema recorder

`utils/sportsbet_response_inspection.py` consumes the existing browser's CDP
`Network.requestWillBeSent`, `responseReceived`, `loadingFinished` and
`loadingFailed` events. It counts websocket frames without reading their payload.
It records local request/response/completion times and navigation generations.
The only changed existing module is `utils/sportsbet_browser.py`: an optional
`response_inspection` argument tees events after the existing denial event is
queued. No caller enables this argument by default. Denial monitoring remains
independent; recorder parsing failure stops only the optional recorder.

After normal rendered extraction, an explicit call to
`driver.sportsbet_inspect_response_shapes()` can request at most four **already
buffered local CDP bodies** through `Network.getResponseBody`. This is not HTTP
replay. Each command uses the existing two-second CDP timeout and source gate
checks. No body reads occur through this method after browser closure or while
the source gate is held. This method never accepts source recovery, publishes a
receipt, schedules work, reserves an attempt or writes shared state.

The projection intentionally retains **no scalar body values** (even numbers
might be finishing placements), unknown property names, headers, cookies,
request bodies, arbitrary query values, errors or raw bodies. Recognized field
names and scalar types are retained; unknown fields have unnamed shapes. Paths
use a reviewed token allowlist; unknown segments are redacted and explicitly
marked. A redacted route is not sufficient evidence of an executable endpoint.
Non-JSON and base64 bodies are not inspected. The full JSON is transiently parsed
in memory; only the projection is returned. The raw-body limit is checked after
CDP delivery, so it is not a hard browser/transport allocation bound.

Bounds: 64 requests, four body reads, 1 MB accepted body text, 2,048 projected
nodes, depth 12, 16 sampled array elements, 16 timing markers, and 50 seconds or
the caller's window expiry, whichever occurs first. Bounds and exclusions are
explicit. Prior-navigation, partial, redirected/reused-ID, failed, cached,
service-worker, oversized and non-200 responses cannot qualify for body
inspection. These exclusions are conservative research choices, not claims that
such provider delivery modes are unusable. No stream accumulator or price merger
exists. The actual receipt normalizer rejects the resulting schema-only report.

## Minimal main-agent integration steps (prepared, not executed)

1. Review/cherry-pick this isolated commit into the next offline integration
   candidate. Reconcile the small shared-file change in
   `utils/sportsbet_browser.py` with PR #188's denial instrumentation; never
   replace the main agent's denial/provenance changes with this older base.
2. Add an explicit recorder argument at the existing `setup_driver` call to
   `create_sportsbet_driver`. Construct it immediately before browser startup
   with `expires_at=min(reserved_window_end, jump_time)` from the unchanged
   canonical reservation. Persist the main campaign/run/reservation/race/window
   identifiers alongside its report using the existing evidence writer.
   Do not infer a race binding from response route tokens.
3. Mark normal extraction completion and call the inspection method before the
   existing driver closes, only if the unchanged lifecycle budget has room.
   Retain `recorder.report()` on failure as finite evidence. Local inspection can
   add up to four CDP timeouts; do not extend the acquisition window to fit it.
   Perform the same offline exported-entrypoint checks required by the campaign.
4. Only after an explicit main-agent pre-run handoff and admission through its
   existing campaign lock, reservation, request guard and source gate, attach to
   one already authorized future-race capture. No separate launcher is supplied.
   STOP currently prevents this step. The original observation authority does
   not authorize clearing the renewed hold.
5. Review the safe route/shape evidence. If it exposes a usable candidate, define
   a schema-specific restricted pre-race extractor before retaining any values.
   Unknown shapes and redacted route segments remain missing evidence. A single
   schema-only observation may be insufficient; do not perform another load or
   endpoint request under this task's exhausted/held authority to fill the gap.
6. Only then build the narrow default-off odds adapter and run the actual parser,
   capture validator and receipt verifier on recorded safe fixtures. Cover exact
   identity, scratches, missing/absent prices, suspension, partial/stale/order,
   reconnect, acquisition expiry, source denial and durable accounting. Keep the
   API route as supplemental evidence and the canonical race page URL in the
   existing receipt identity. No silent DOM/HTTP retry or fallback.

Example wiring fragment **inside the existing authorized capture**, not a live
command or a new execution path:

```python
recorder = ResponseInspection(expires_at=min(reserved_window_end, jump_time))
driver = create_sportsbet_driver(existing_factory, response_inspection=recorder,
                                **existing_browser_options)
# Existing capture controls and ordinary extraction run here, unchanged.
inspection = driver.sportsbet_inspect_response_shapes()
# Existing evidence writer binds inspection to the exact reservation.
```

## Measurement and validation

| Quantity | Result |
| --- | --- |
| Browser startup/navigation live latency | Not measured; markers prepared. |
| Usable fixed-odds response arrival | Not established; response callbacks alone do not mean usable odds. |
| Current rendered extraction completion | Not measured live; completion marker prepared. |
| Proposed odds parsing and receipt verification | No provider adapter yet; no improvement measurement. |
| Configured existing waits | Two five-second waits plus document-ready wait in the inspected landing/race path; not estimated savings. |
| This task's provider operations / campaign consumption | **0 / 0**. No browser was launched against Sportsbet. |
| Synthetic guarded-browser accounting | One existing browser operation with recorder enabled; local body inspection adds zero operations. |

There is no measured live latency or reliability improvement, and no same-race
price comparison. Discovery remains an independent bottleneck; this recorder
and any future faster race-price parser do not solve NextEvents/refresh workload.

Validation used `/home/l4nd0/greyhound_racing_collector/.venv/bin/python` in
`unshare --user --map-root-user --net` (outbound-denied Linux network namespace):

```sh
unshare --user --map-root-user --net \
  /home/l4nd0/greyhound_racing_collector/.venv/bin/python -m pytest \
  --noconftest -o addopts= -p no:cacheprovider -q \
  tests/test_sportsbet_response_inspection.py tests/test_sportsbet_access.py \
  tests/test_capture_reservation_expiry.py tests/test_autonomous_live_odds_capture.py
```

**95 passed in 1.74 seconds**. The new fixture events/body are explicitly
fabricated in `tests/test_sportsbet_response_inspection.py`; the asynchronous CDP
transport is `tests/fixtures/freshness_transport/fake_cdp.py`. Tests exercise the
actual guarded driver, durable source policy and receipt normalizer. The existing
capture suite checks complete rendered captures, runner mismatches, scratches,
partial markets and expiry; it does not prove a structured provider contract.

The first test launch using global `conftest.py` failed during collection because
this environment lacks `flask_compress`; no dependencies were changed. These
focused tests do not require the unrelated Flask app, so the final command uses
`--noconftest`. An intermediate fixture failure exposed missing test-only source
policy initialization (2 failed, 36 passed); initialization was corrected in the
synthetic fixture. A later order-guard run found a test assertion matching a decimal substring in a local timestamp (1 failed, 94 passed); the assertion now checks only the body projection. The production policy and source state were not changed.

Full passing log:
`/home/l4nd0/greyhound-response-observer-evidence-20260924/verified-tests.log`.
The isolated checkout is `/home/l4nd0/greyhound-response-observer-20260924`, branch
`research/sportsbet-response-observer-20260924`. No merge, deployment, service
restart, main-package modification or provider execution is part of this handoff.


Exact validation environment: CPython 3.11.15, pytest 8.4.1, Selenium 4.34.2,
requests 2.34.2; Linux 6.8.0-138-generic x86_64, glibc 2.35. Interpreter resolves
to `/home/l4nd0/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11`.
No dependencies were installed. Fixture/evidence SHA-256:

- `tests/test_sportsbet_response_inspection.py`:
  `e120c12ed633c6db0984783aba849355a009037edbab9f92caf2298dfcbe1041`
- `tests/fixtures/freshness_transport/fake_cdp.py`:
  `60c738d90760f4f27fec1e6fc9f6b648b4cd31fe3e114db9e40529e67b669b88`
- `verified-tests.log`:
  `fc1d25e10ccda4a423861859606df075baa307b21a4d590e1b87431d50980e28`

The main agent's operational engineering notes already describe conservative
capture-fetch-start timing in its newer PR #188 candidate. Preserve that work
when integrating; the companion contract review describes the pinned #187
baseline, not a claim that #188 still stamps only append time. Neither timestamp
substitutes for a future adapter's actual response observation time.
