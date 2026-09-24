# Sportsbet open-source implementation verdict and integration handoff

Inspected 24 September 2026, isolated branch `research/sportsbet-open-source-20260924`,
base `b9d2db2cd1158c366d7e6de4620552b050b850cc` (PR #188). Zero Sportsbet requests,
no browser attachment, no runtime/database/service changes. External repositories
were cloned for static source review only; no installation or entrypoints executed.
See [external implementation evidence](sportsbet_external_implementations_20260924.md)
for pinned commits, licenses, exact racing schema claims and client defects.

## Ranked verdict

1. **sportsdata-mcp racing endpoint definitions: best validation candidate, not a
   ready adapter.** AllRacing could supply a reusable daily discovery snapshot and
   Racecard could supply one-event paired prices. Its actual implementation is a
   generic HTTP dispatcher, not a complete greyhound market validator. Horse
   examples, spec/documentation contradictions, no greyhound fixture, ambiguous
   fixed-price classification and batch wrapping prevent receipt-producing reuse.
2. **Our existing guarded acquisition and receipt chain: reuse as the integration
   host.** PRs #184/#187/#188/#189 already own access, discovery sharing,
   reservations, validation, retention and observation. The useful implementation
   in this branch extends #189's existing value-free recorder with the documented
   racing route/field names. No caller is enabled and no provider values retained.
3. **bensharkey3/sports-odds-scraper: small transport/parsing illustration only.**
   Its AFL sports markets do not establish racing coverage, complete runner fields,
   fixed WIN/PLACE terms, source coordination or timestamps. Do not adapt by merely
   swapping an event ID.
4. **Legacy EventScraper: unsuitable restoration target.** Its identifiable
   upstream implements sports SportCard/market-group requests, not Racing/Racecard.
   It also loses IDs above seven digits and performs uncoordinated nested requests.

**No production odds adapter is justified by this evidence.** A synthetic parser
that invents greyhound box, fixed-market code or batch envelope semantics would
only validate our assumptions. The tested recorder extension is reusable code
that makes the next authorized observation more informative while preserving
all existing receipt requirements.

## EventScraper history and actual endpoints

The local dependency is vendored source, not a pinned package dependency:
`hybrid_odds_scraper.py` imports `event_scraper.EventScraper`; that imports
`sportsbook_factory`, which imports ten `sportsbook_implementations` modules.
The package is absent in this clean worktree and the user's starting checkout;
`.gitignore` explicitly excludes it. Requirements do not restore it.

Pinned history:

- [78f9c97c](https://github.com/0rl4nd0l/greyhound-racing-collector/commit/78f9c97c921668c6f52d08e26ca34e22d803ae3a)
  introduced the hybrid wrapper on 27 July 2025. It accepts any nonempty API
  DataFrame as success, retries then falls back to Selenium. This is not evidence
  of complete greyhound paired markets.
- [dd0db759](https://github.com/0rl4nd0l/greyhound-racing-collector/commit/dd0db75902f5dd7c6326ff7dfb01d44b5c1f0ceb)
  introduced EventScraper, its factory and `test_sportsbook_integration.py` that
  day. The latter is a live demonstration with a Sale race URL and conditional
  success print statements, not an assertion-bearing fixture or saved success.
  The Selenium SportsbetOddsIntegrator also existed in this commit: history does
  **not** establish a later migration caused by a verified API failure.
- [bb1ab999](https://github.com/0rl4nd0l/greyhound-racing-collector/commit/bb1ab99910b2ab8edb4d19cd9fc1562dd0b5fa66)
  already excludes `/sportsbook_implementations/` on 27 July 2025.
- [2f7b6148](https://github.com/0rl4nd0l/greyhound-racing-collector/commit/2f7b6148091d6ebcb955664b9a5adeb49f2c51b6)
  adds opt-in pre-jump live capture, 26 May 2026;
  [580de0a3](https://github.com/0rl4nd0l/greyhound-racing-collector/commit/580de0a33415d9bd81e7df4129b26b66d2dc74d2)
  binds explicit runner boxes;
  [31409160](https://github.com/0rl4nd0l/greyhound-racing-collector/commit/314091604fc245185638cbf30be07ed7241301d9)
  repairs paired market extraction, 10 July 2026. Current capture calls
  `fetch_odds_for_target_race` and `SportsbetOddsIntegrator.get_race_odds_from_page`,
  not HybridOddsScraper.

The upstream identified by its class/factory structure is
[declanwalpole/sportsbook-odds-scraper at 8e791fc31a3b118936dd2f2b02e1919aaa4daeb3](https://github.com/declanwalpole/sportsbook-odds-scraper/tree/8e791fc31a3b118936dd2f2b02e1919aaa4daeb3).
The factory differs only in formatting/import ordering. This is strong lineage
support, not proof that an untracked historical local implementation was identical.
Its [Sportsbet class](https://github.com/declanwalpole/sportsbook-odds-scraper/blob/8e791fc31a3b118936dd2f2b02e1919aaa4daeb3/sportsbook_implementations/sportsbet.py)
was last changed at `d67af9e197e12b1486c5a3c656e65653a70a2987` (30 August 2023):

```
/apigw/sportsbook-sports/Sportsbook/Sports/Events/{id}/SportCard
 ?displayWinnersPriceMkt=true&includeLiveMarketGroupings=true&includeCollection=true
/apigw/sportsbook-sports/Sportsbook/Sports/Events/{id}/MarketGroupings/{group}/Markets
```

It slices exactly seven URL characters for event ID, reads `displayName` and
`marketGrouping`, requests every group with bare `requests.get` without timeout,
then flattens `selections[].price.winPrice`. No racing discovery, scratching/box
mapping, PLACE price or place-term handling exists. No tests/fixtures or license
file were found at the pinned upstream tree. We copied no implementation code.
Local EventScraper's outer request now uses the coordinated session, but restoring
this upstream class would leave its nested bare requests outside that session.
Hybrid import also registers process-killing cleanup; we did not import/run it.

**Historical conclusion:** no successful greyhound API capture is established by
inspected source/history. The demo URL and success log template do not prove one;
absence of the ignored implementation prevents a universal claim that none ever
happened. Existing rendered collection has an independently developed, hardened
capture contract; no historical incident proving why all API collection was
abandoned was found. No target race outcomes or DB histories were inspected.

## Comparison with concurrent repair

| Work | Pinned head inspected | Reuse / boundary |
| --- | --- | --- |
| [#184](https://github.com/0rl4nd0l/greyhound-racing-collector/pull/184) | `fe3d984a13a7446ca0af301dd736fed20bab0b96` | Durable source operation, denial history, campaign consumption and reservation expiry. New route needs route-specific usable-data acceptance, not HTTP 200. |
| [#187](https://github.com/0rl4nd0l/greyhound-racing-collector/pull/187) | `87a3d33928896b141e9b5137bcf098ea73c03cf2` | Shares one NextEvents snapshot across refresh workers. AllRacing must replace/extend this snapshot seam; it must not create another polling loop. |
| [#188](https://github.com/0rl4nd0l/greyhound-racing-collector/pull/188) | `b9d2db2cd1158c366d7e6de4620552b050b850cc` | Real capture to retained inputs to frozen prediction chain; separate operational DB and unchanged canonical receipt validation. |
| [#189](https://github.com/0rl4nd0l/greyhound-racing-collector/pull/189) | `5e32c21e9930c78b977f8e2ee474ceeae212a5c8` | Passive bounded recorder integrated in #188 as `eb366c38`. Our extension names racing routes/fields while retaining value suppression. |

Main recovery thread `01a0d190-dfee-7b52-912c-a923f58b5633` confirmed ownership
and received these hypotheses. Its candidate `88b5c8d5` attaches the recorder at
capture entry and retains failure observations; it requested our constants/tests
extension for cherry-pick. We did not alter its checkout or access state.
Odds.com.au investigator `01a0d174-4b1a-70d3-8654-0fc7c46a15f0` confirmed a separate
Odds-only investigation/PR #190; it grants no Sportsbet traffic authority. No
Odds.com.au source work was duplicated here. Main's diagnostic resumption
authorization supersedes treating the old recovery counter as a permanent ban;
this branch neither resets that counter nor implements a parallel access gate.

## Exact minimal live validation for the owner

**First use the already planned, admitted browser operation.** Do not add another
navigation or request merely to test this hypothesis. With the extended recorder,
check whether that operation receives `Events/{id}/Racecard`, a contextual card,
or a different route; measure local request/response/completion timestamps.
The recorder preserves only field names/types: it cannot establish actual event,
runner, market codes, price values, completeness or correct box semantics.

If separate direct JSON diagnostic ownership is explicitly handed off through the
existing coordinator, the minimal staged plan is:

1. Select exactly one future greyhound already identified by the existing current
   snapshot and canonical mapping. Confirm event ID, venue alias, race number,
   timezone-aware jump time and full expected active `(box,name)` set. Never use
   demonstration IDs or guess a page slug.
2. Admit **one GET**, with no retries/redirects/fallback, to
   `https://www.sportsbet.com.au/apigw/sportsbook-racing/Sportsbook/Racing/Events/{exact_event_id}/Racecard`.
   No `selectionNames`, account context or credentials. Existing source operation,
   budget, shared ownership and renewal-denial stop apply. A new racecard validator
   must be supplied inside that same operation before recovery acceptance;
   `SourceCoordinatedSession` currently recognizes NextEvents recovery only.
   Do not nest a second operation or report recovery from status alone.
3. Bound body size/time and process only a pre-race allowlist in memory; reject
   nonfuture/non-greyhound/in-play/settled/inactive event or market, unknown price
   classifications, incomplete expected identities, ambiguous box mapping, duplicate
   IDs, priced scratches and missing/nonfinite prices. Establish `numPlaces` and
   PLACE availability from observed provider facts. Do not automatically treat
   horse `drawNumber`, `L`, or the label `Win or Place` as sufficient proof.
   Raw responses contain result/form/history fields: do not print or persist them
   as research inputs, and stop if restricted processing cannot be maintained.
4. Record request/response observation interval, cache headers/age if supplied,
   cache/service-worker delivery, exact allowlisted source identities and hash of
   the approved projection. Parsing time and append time are not provider time.
   Missing source clock/cache facts remain unknown. A 200 with missing data is a
   failed diagnostic, not permission to try another endpoint.
5. Only after single-card semantics are resolved, separately authorize **one**
   `AllRacing/{YYYY-MM-DD}` discovery read for the relevant Melbourne calendar date.
   Require explicit greyhound type, exact event/meeting identities and UTC seconds;
   compare future coverage to the existing snapshot, retaining its observation
   timestamp/hash. Do not infer full-day completeness from one returned section.
6. Batch is a later optional test, not part of the first diagnostic: exactly two
   already identified future IDs to `Events/MultipleRacecards?eventIds={id1},{id2}`.
   Compare requested/returned ID sets, reject duplicates, and represent absent/error
   members explicitly. Do not accept a partial batch as complete, refetch missing
   IDs automatically, or reuse the MCP 60-second cache.

These are ordered diagnostic gates, not authorization for a three-endpoint sweep.
On renewed denial, stop immediately and preserve counters/backoff/denial evidence.
A successful observation still needs network-denied recorded-fixture integration
before a production adapter is justified.

## Adapter acceptance and measured versus projected improvement

Reuse the acquisition seam documented in
[sportsbet_odds_contract_20260924.md](sportsbet_odds_contract_20260924.md).
An eventual default-off adapter must carry provider IDs as extra provenance while
preserving canonical mapping; introduce honest structured box provenance rather
than `explicit_dom`; preserve original observation time; propagate actual place
terms through persistence (current code defaults to three). Run the ordinary
`validate_fetched_odds`, normalization, append, receipt and retained-input chain
inside the original reservation and recheck expiry before append. No independent
scheduler, fallback source, receipt writer or predictor path.

Measured here: zero provider operations, zero live latency/coverage measurements.
The original integration baseline passed 99 focused tests in 2.48 seconds in a
network-denied namespace. Recorder extension commit `c39efe88` passed **104 tests in 2.08 seconds** using:

```sh
unshare --user --map-root-user --net \
  /home/l4nd0/greyhound_racing_collector/.venv/bin/python -m pytest \
  --noconftest -o addopts= -p no:cacheprovider -q \
  tests/test_sportsbet_response_inspection.py tests/test_sportsbet_access.py \
  tests/test_capture_reservation_expiry.py tests/test_autonomous_live_odds_capture.py
```

`git diff --check` passed. Five new cases prove documented racing route names
remain recognizable while target IDs, dates and filters stay redacted, and
paired-price field types remain recognizable without revealing prices,
status values, outcomes, form or credentials.
Fixtures are fabricated and do not establish provider compatibility.

Projected only: daily discovery reuse could reduce repeated discovery; one exact
Racecard read could avoid browser startup and the two configured five-second waits;
batching N racecards could reduce N logical fetches to one if completeness and
freshness are verified. None proves a ten-second speedup, provider permission,
full-day coverage or sustainable live capacity. #187 already removes repeated
NextEvents reads within a refresh; do not count those savings again.

## Follow-up: owner-recorded 14:40 browser attempt, local inspection only

The main owner supplied a recorder from candidate `e07ab93e`. This investigation
made no provider request or browser attachment. Read-only JSON inspection used:

```
/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector/artifacts/full_evidence_orchestration_20260525/shadow_autopilot_daemon_runtime/live-freshness-attempts-v1/rehearsal-1607afa2e341f7dd41321aceb2a1d45d125a84047273fc6f86bfccab0dbf3ed8/captures/5cefded090bc4ffa9408983734116b9a/capture-reservation.requests.responses.json
SHA256 56cf1f11736de1ea3106a2e0488d2e48e0b67473cdc17a50c6762d197043661d
```

Directly verified metadata:

- 53 network response records, nine gateway response records, zero dropped
  requests/responses, **zero body reads**, zero observed WebSocket frames.
- Three Document responses: HTTP 200 at local elapsed 1.3436s, HTTP 200 at
  8.7015s, HTTP 429 at 19.6449s. The owner identifies the last as
  `/{uuid}/{uuid}/fp`; this recorder retains only its path hash
  `292b2011ef38576baca1e65126565729016d25423ad75da24f7009e11a6882b0`.
  It records Date `Thu, 24 Sep 2026 04:41:14 GMT`, no Retry-After.
- First NextEvents response: HTTP 200, noncached, 9,609 encoded bytes. Second:
  HTTP 200, **from_cache=true**, zero encoded bytes, no service-worker flag.
  These are not two independently fresh upstream observations.
- Browser `146.0.7680.153`, driver `146.0.7680.165`; count-only snapshot has nine
  runner elements and zero price elements. These selectors are not market
  validators. The owner independently reports actual generic-button extraction
  of five WIN and five paired PLACE rows before denial; this investigation did
  not inspect those values or establish receipt acceptance.

**What `/fp` means:** a fingerprint/challenge document is a plausible hypothesis,
not established vendor attribution. Neither inspected local implementation nor
pinned external Sportsbet source documents this path. `Document` is a browser
resource classification, not proof that it was the top-level document, an iframe,
a required pricing resource, or dispensable telemetry. The recorder lacks
frameId/loaderId/initiator/frame-parent relationships and redirect chains needed
to resolve that distinction. A suffix and two UUID-shaped segments cannot justify
ignoring the denial or changing its ownership/backoff treatment. The owner retains
shared STOP and sole provider ownership.

**Embedded odds:** plausible but unproven. No Racecard route is visible in the
recorded responses, but the observation is bounded, same-Sportsbet-host scoped,
and not a page/stream completeness inventory. With zero retained body shapes,
absence of a Racecard fetch does not distinguish server-rendered HTML, embedded
hydration state, another transport/host, preexisting cache or stream delivery.
Zero observed WebSocket frames likewise cannot prove no stream exists. The
owner-reported paired DOM rows establish that prices reached the rendered page;
they do not identify the transport or prove complete provider market semantics.
The current JSON-LD parser (`sportsbet_odds_integrator.py`,
`extract_race_info_from_json_ld`) explicitly initializes empty odds and extracts
only event metadata. Its separate page-source regex extracts race number only.
Neither is an embedded-price implementation.

Useful offline next step for the main repair owner: replace obsolete readiness
selectors with a predicate built on the existing paired-row extraction contract,
then exercise delayed/partial/scratched/suspended fixtures. Preserve denial STOP,
reservation consumption and final validation even when DOM rows arrive first.
No measured latency saving follows from the configured waits alone. If a later
observation is authorized, bounded frame/request attribution and a value-free
inventory of embedded script field names would answer the transport question
without retaining raw HTML, outcomes, credentials or scripts. Do not replay or
suppress `/fp`, attach to a live browser, or take another provider request for
this follow-up. No new adapter is justified by this recorder.
