# Odds.com.au alternative acquisition: bounded investigation

## User-authorized headed follow-up, 14:08 AEST

The user reported ordinary Chrome/Edge access works and explicitly authorized
trying the different browser method. One additional bounded page load was made
with **headed Chrome**, the same executable/host, a fresh profile, and unchanged
host/redirect/request limits. No user-agent spoofing, proxy change, stealth flags,
saved account profile or challenge solving was used. Chrome's ordinary user-agent
was verified on `about:blank` with outbound networking denied before the attempt:
`Chrome/146.0.0.0` replaced the headless run's `HeadlessChrome/146.0.0.0`.
`navigator.webdriver` remained true; this remains an automated browser, not the
user's existing session. No desktop display was available, so an official Ubuntu
Xvfb package was downloaded (866 kB) and extracted locally under the isolated
evidence directory. No system packages or production services were changed.
The owned virtual display and browser were closed after the attempt.

At **14:08:22.772210 AEST** the same public page returned **HTTP 202**, with
`server: CloudFront`, `x-cache: LambdaGeneratedResponse from cloudfront`, and
HTML content type. Its next attempted resource was
`https://challenges.cloudflare.com/turnstile/v0/api.js` (query values discarded).
**Our Odds-only host allowlist blocked that script before transport.** No
Cloudflare denial was observed and the verification flow was not completed.
The HTTP 202 body's contents were not read or retained. The evidence is
consistent with a browser-verification step; it does not establish the cause of
the earlier 403, successful normal-browser access, or a usable odds response.
Time also changed between attempts, so headless mode alone is not proven causal.

This follow-up admitted **one Odds.com.au GET**, blocked one script, and made
**zero Cloudflare, Sportsbet or bookmaker requests**. Cumulative site consumption
across the two separately authorized observations is **two Odds.com.au GETs**;
no data endpoint, redirect, odds capture or recovery operation was admitted.
Sportsbet's canonical source-state SHA-256 was byte-identical before and after:
`e20281969c94eacc1f46cb0f2d3565cc226396d059a8ad26a693cdd238f7e4d9`.
The separate Sportsbet recovery owner was notified; this task holds no Sportsbet
execution authority and has no remaining planned provider traffic.

The immutable original receipt is
[headed observation](../../tests/fixtures/odds_com_au_headed_observation_20260924.json),
SHA-256 `4d6f2381f6df7b96fe495da1dfd11ac5250bdc93346a471a71030834531416c2`.
It records `stop: null`: the guard correctly blocked the cross-host script but
had not classified a Turnstile request as a terminal challenge. That reporting
and admission gap was fixed **offline afterward**: such a request now sets
`challenge_request`, blocks all subsequent admissions, and never fetches the
challenge. The original observation was not rewritten or rerun.

The final tooling adds opt-in `--headed`, records local browser identity and a
small allowlist of non-secret diagnostic response headers, and preserves every
existing request/source boundary. **33 network-denied tests passed in 0.65s**,
including headed denial and HTTP 202 followed by blocked Turnstile and same-site
requests. Log: `/home/l4nd0/greyhound-odds-com-au-evidence-20260924/headed-final-tests.log`,
SHA-256 `605db210d3fe0bcabbe08f50d8bda44bda853c4866c7f0dcd664f3fb171482f8`.
The earlier counts, hashes and environment below describe the initial observation
and initial committed candidate. No odds adapter or receipt relaxation is justified
by either observation. A future access test must explicitly account for normal
verification dependencies; this attempt's host filter means it was not a full
unrestricted normal-browser comparison.

## Initial observation and verdict

**Not established as an integrable odds source on the evidence obtained.** The
first ordinary public page request returned HTTP 403, so inspection stopped
without reading its body, requesting another race or probing an endpoint. This
single denial does not establish that Odds.com.au has no structured API or that
normal access is unavailable in every context. It establishes that this
particular authorized browser observation could not proceed.

No odds adapter is justified. The reviewable implementation is a bounded manual
inspection probe reusing PR #189's value-free JSON shape projector, with offline
tests and this handoff. Nothing calls it from the production collection path.

## Actual network observation and accounting

Starting URL supplied by the user:
<https://www.odds.com.au/greyhounds/gosford-20260924/central-coast-locksmiths-race-4/>

| Observation | Retained evidence |
| --- | --- |
| Method | GET |
| Request observed | 2026-09-24 03:49:59.960609 UTC / 13:49:59.960609 AEST |
| Response observed | 2026-09-24 03:50:00.002173 UTC / 13:50:00.002173 AEST |
| Status | HTTP 403 |
| Retry-After header | Absent; this does not authorize retry |
| Body | Not read or retained |
| Odds.com.au document requests admitted | 1 |
| Subresources, data endpoints, redirects, websockets admitted | 0 |
| Sportsbet or other bookmaker requests | 0 |
| Additional race loads, retries, recovery operations | 0 |
| Sportsbet campaign consumption or state mutation | 0 |

The race's start/finished state could not be established. No outcomes were read,
and no alternative upcoming page was fetched after the denial. The response
callback occurred 0.053519 seconds after the probe's browser-context setup point;
that is denial-observation timing, not usable-odds latency or a performance gain.
The response's exact denying infrastructure/cause is not established.

The retained, sanitized observation is checked in at
[tests/fixtures/odds_com_au_denied_page_20260924.json](../../tests/fixtures/odds_com_au_denied_page_20260924.json).
Its SHA-256 is
`1128482f02a634ac4bc2a59dc7a31121970b5491a911a89e765d40cc2235f4a1`.
The original is
`/home/l4nd0/greyhound-odds-com-au-evidence-20260924/observation-1.json`.
It contains no raw response body, cookies, authorization/account data or headers.
This is a separate **Odds.com.au 403**, not another Sportsbet 429. The main agent
was notified through its existing thread channel and independently confirmed the
retained request count and unchanged Sportsbet source/campaign state.

## What remains unknown

| Required contract | Evidence status |
| --- | --- |
| Structured race/runner/market endpoint, method and non-secret parameters | No data response reached; only the supplied page GET was observed. |
| Exact race/date/venue mapping and stable runner/box IDs | Unknown. |
| Complete fixed WIN and PLACE from one bookmaker | Unknown; no bookmaker or odds values observed. |
| Bookmaker identity versus comparison site's acquisition identity | Semantics to preserve, not demonstrated source fields. |
| Scratchings and complete active-runner coverage | Unknown; missing runners cannot be treated as scratched. |
| Suspension, closure, absent prices and PLACE terms | Unknown; no default terms may be invented. |
| Provider timestamps, version/sequence, update interval | Unknown. Local request/response times are available only for the denial. |
| Snapshot versus deltas, reconnects, stale/mixed versions | Unknown. |
| Capture reliability and latency improvement | Not demonstrated or estimated. |

No best-odds aggregation or market mixing was attempted. Faster odds extraction
would not itself address the existing discovery/refresh bottleneck.

## Probe implementation and limits

[`scripts/inspect_odds_com_au_public.py`](../../scripts/inspect_odds_com_au_public.py)
uses a fresh Playwright browser context with the installed Google Chrome. It
sets request interception before navigation, allows Odds.com.au hosts only,
blocks redirects, secondary documents, explicit result routes and all websockets,
and disables service workers. No saved profile, identity/proxy rotation,
privileged request replay, challenge bypass or bookmaker click-through is used.

The bounds are one page navigation, at most 60 admitted same-site requests and
30 seconds of request admission. The document timeout is 20 seconds, followed
by a short finite observation period. HTTP 401/403/429 and recognized HTML
challenge indicators stop further admission. The operation intent and request
admissions are written before transport; completed response observations are
separate. Reusing an existing output path is rejected. There is no automatic
retry or alternate endpoint/race fallback. HTTP routing alone does not cover
websocket handshakes, so the separate websocket guard is essential.

For a successful future authorized page, the tool would project recognized JSON
field names/types with PR #189's `response_shape`, and retain only restricted
schedule/state metadata from inline JSON or normal JSON responses. It suppresses
runner names, prices, outcomes and raw payloads. Thus it is an initial schema
probe, **not an odds parser or evidence of a simultaneous complete market**.
Its conservative cross-host/websocket blocks can prevent normal site delivery;
that limitation must be reported rather than called provider incapability.
No such blocked dependencies occurred in the actual single-request denial.

The final probe received offline formatting/refactoring/tests after the recorded
observation. It was not rerun live. The observation is not a successful live
validation of the final candidate or of future response parsing. Do not rerun
this command or change the output path to bypass the retained denial; any
future source access needs a separately established authorization/access basis.

## Receipt integration consequences

The [offline source-contract review](odds_com_au_receipt_contract_20260924.md)
traces the current collector, persistence, publisher and exact R3 verifier.
They currently include Sportsbet-only URL admission, Sportsbet-labelled box
provenance and persisted source, top-three PLACE defaults, and receipt schemas
that do not preserve the required acquisition/origin pair. The receipt
normalizer's generic source_kind parameter is not full pipeline support.

If future safe evidence establishes viability, the minimum integration is:

1. Add one default-off acquisition selector in the existing reserved capture
   path. Reuse its window identity, accounting, expiry checks and receipt
   publication. Source-specific admission must leave Sportsbet's STOP intact;
   no separate scheduler, database or silent fallback.
2. Select one identified bookmaker with complete fixed WIN **and** PLACE for
   the same active runners and compatible observation/version. Reject mixed
   best-price composites, incomplete markets, unclear suspension and unknown
   terms. Define stream initialization/order/reconnect handling if applicable.
3. Bind `acquisition_source=odds.com.au`, the Odds.com.au page/data route, and
   `price_origin_bookmaker`/bookmaker ID independently. Preserve provider IDs,
   original local observation time and provider time only when actually supplied.
   A Sportsbet-origin price acquired from Odds.com.au is not direct Sportsbet
   acquisition and must never be relabelled to pass existing source validators.
4. Extend the existing provenance/report/receipt contract deliberately, test
   actual persistence and exact receipt admission, and retain the existing
   completeness and expiry requirements. The main repair already has newer
   capture-time changes; reconcile those instead of replacing them with this
   branch's older #189 baseline.
5. Keep research trust, cohorts and model inputs unchanged. Operational support
   for another acquisition source is not automatic research eligibility.

No implementation of those conditional integration changes was made, because
there is no observed source response to implement against.

## Offline validation and reproducibility

**31 tests passed in 0.70 seconds**, with outbound networking denied:

```sh
unshare --user --map-root-user --net \
  /home/l4nd0/greyhound_racing_collector/.venv/bin/python -m pytest \
  --noconftest -o addopts= -p no:cacheprovider -q \
  tests/test_odds_com_au_inspection.py \
  tests/test_sportsbet_response_inspection.py
```

The new tests drive the actual probe through fabricated browser callbacks and
exercise host restrictions, redirect/result/websocket rejection, denial without
body reading, request cap, challenges, schedule projection and output reuse.
The actual `validate_fetched_odds` rejects an honest Odds.com.au URL despite
complete synthetic WIN/PLACE identities and a Sportsbet bookmaker label; the
actual receipt normalizer then rejects that failed validation. No validator was
weakened to manufacture a passing alternative-source receipt. All successful
fixture prices are fabricated, not observed Odds.com.au capability.

Validation environment: CPython 3.11.15, pytest 8.4.1, Playwright 1.54.0 on Linux;
no dependencies were installed. Evidence log:
`/home/l4nd0/greyhound-odds-com-au-evidence-20260924/final-tests.log`.
SHA-256:
`4d97fad67c83f5447b9fe58773136e310ad4682f990e5b74767302d2aff8ef1d`.
New test file SHA-256:
`7ee3d6a1078636eba9912abe58a8d98d71ce8c6f5478cf0584afc4b79c8bcc46`.

Isolated worktree: `/home/l4nd0/greyhound-odds-com-au-20260924`.
Branch: `research/odds-com-au-acquisition-20260924`.
Base: PR #189 commit `5e32c21e9930c78b977f8e2ee474ceeae212a5c8`.
Only new probe, tests, sanitized evidence and documentation are added. Main
integration needs PR #189's projector (already cherry-picked by the main agent);
there are no modified shared production files in this change. No deployment,
service restart, production write, betting, results acquisition or research
change was performed.
