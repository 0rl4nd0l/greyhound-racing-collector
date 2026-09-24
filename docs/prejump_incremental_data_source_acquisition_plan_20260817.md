# Pre-jump incremental data-source acquisition plan

Date: 2026-08-17

Status: report-only source-feasibility research; no collector, account automation, model fit, wager, or broad collection is authorised

Decision baseline: corrected Sportsbet WIN remains the comparator

## Executive decision

The highest-value genuinely different source is a **licensed Betfair Exchange order-book feed**. It can add a second market's back/lay prices, spread, displayed depth, traded volume and source-timestamped price evolution rather than another transformation of Sportsbet WIN. However, the standard no-betting path cannot presently deliver valid fixed-window evidence: Betfair explicitly says Live-key read-only use is not permitted, while a Delayed key can delay REST prices by 1–180 seconds and conflates delayed Stream updates to one message every three minutes. The terminal state is therefore `BETFAIR_LIVE_PILOT_BLOCKED_ACCESS_POLICY`, pending a written Australian no-betting live-data permission.

No other bookmaker page should be scraped. Repository inspection performed in the parallel local-inventory lane establishes that current TheDogs `/api/runners/odds` evidence is already Ladbrokes-priced, so direct Ladbrokes WIN is not new; only separately sourced Ladbrokes metadata such as bet/hold percentages, source-timestamped fluctuations or explicit scratching time could survive a distinctness screen. TAB has a first-party real-time Web Services API, but access is discretionary and its personal licence restricts storage and derivative use; it becomes the second market-data candidate only after written commercial/research permission and production schema access. GRV FastTrack is the clearest official source found for explicitly timed late scratchings and several richer dog/trainer fields, but its terms prohibit storing Race Information unless GRV approves the user or grants a separate written agreement. Those are acquisition-negotiation candidates, not currently authorised collectors.

The recommendation is therefore at most two **conditional** acquisition pilots, neither executable under the presently documented standard access terms:

1. Betfair Exchange five-window order-book capture, only after written approval for live, read-only/no-betting data and persistent research storage.
2. TAB Studio five-window fixed-WIN/tote/field-change capture, only after written research/commercial data rights and production endpoint/schema access.

BOM nearest-station observations remain the best immediately documented low-cost non-market source, but are ranked third because their likely incremental information content over a strong market is lower. They should be collected only if the market-data permission gates fail or as a later covariate-quality lane, not used to displace the two higher-information acquisition targets.

Neither pilot supports a market-edge, profitability, model-quality, or promotion claim. A 10–20-race pilot tests acquisition reliability, identity, timing and missingness only.

## Repository context checked

The local inventory was read before external source research. No live race-coverage claim is made from a single run or packet.

- [`docs/race_evidence_inventory.md`](race_evidence_inventory.md), [`scripts/autonomous_live_odds_capture.py`](../scripts/autonomous_live_odds_capture.py), and [`sportsbet_odds_integrator.py`](../sportsbet_odds_integrator.py) establish the existing strict pre-jump Sportsbet WIN lane and its append-only persistence boundary.
- [`scripts/capture_thedogs_market_history.py`](../scripts/capture_thedogs_market_history.py), [`scripts/audit_thedogs_published_market_history.py`](../scripts/audit_thedogs_published_market_history.py), and [`docs/thedogs_market_history_capture.md`](thedogs_market_history_capture.md) establish the five-window TheDogs lane, immutable raw-plus-receipt design, and the current source-explicit Ladbrokes provider. Direct Ladbrokes WIN was therefore screened out as duplicate information.
- [`utils/prejump_weather.py`](../utils/prejump_weather.py) and [`utils/prejump_sportsbet.py`](../utils/prejump_sportsbet.py) establish that Open-Meteo forecast weather and Sportsbet-declared track condition already exist. BOM is ranked only for observed weather that is different from those fields.
- [`utils/expert_form_metadata.py`](../utils/expert_form_metadata.py) confirms existing TheDogs expert-form metadata, so trainer/sectional candidates were screened for genuinely different events or provenance rather than repackaged form.
- [`src/collectors/fasttrack_scraper.py`](../src/collectors/fasttrack_scraper.py) and [`docs/FASTTRACK_INTEGRATION_GUIDE.md`](FASTTRACK_INTEGRATION_GUIDE.md) were reviewed as legacy FastTrack work. The guide's proxy/CAPTCHA-bypass approach is outside this goal's access rules and must not be reused. No approved Betfair or TAB collector was found in the inspected repository paths.

This work added only this report. It did not call a live provider, capture a race, write the runtime DB, alter a service, or implement a collector. Pre-existing working-tree changes were preserved.

## Evidence standard and rejection rule

Feasibility claims below use provider-owned documentation or provider-operated live surfaces. A public page is not treated as an API contract. A source is rejected from implementation if any of the following remains true:

- runner/race identity cannot be bridged one-to-one using provider-native IDs and a complete runner set;
- the payload can cross scheduled or actual jump without an explicit fail-closed `in_play`/time check;
- the only time is an undated current value or a post-race publication time;
- access would require scraping circumvention, private mobile endpoints, anti-bot bypass, credential sharing, or use outside published/licensed terms;
- raw bytes plus a receipt cannot be stored lawfully and immutably.

## Ranked feasibility table

Scores are implementation priorities, not predicted effect sizes. `H/M/L` are relative judgements from the cited source facts and the present acquisition goal.

| Rank | Candidate | Distinct fields beyond Sportsbet WIN | Native identity and time semantics | Availability, auth, cost and coverage | Principal risk / legal constraint | Incremental value | Effort | Decision |
|---:|---|---|---|---|---|---|---:|---|
| 1 | **Betfair Exchange API + licensed historical data** | Best back/lay ladders, spread, available size/depth, total matched/traded volume, market status and price history | Provider `eventId`, `marketId`, `selectionId`; market start time; Stream `pt` publish time and `clk`; capture receipt supplies receive UTC/monotonic time | Official API supports market reads; Historical Data covers Australia/New Zealand from October 2016 in provider-timestamped Stream format. Australian greyhound live presence/liquidity still need measurement. Delayed key is free but prices may lag 1–180 seconds and delayed Stream is three-minute conflated; Live-key read-only use is not permitted. Generic and Australian fee pages differ, while Australia also documents an API access charge up to A$200/month net of qualifying commission; obtain a written AU quote rather than assuming an activation cost. ([Betting API](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687158/Betting%2BAPI), [historical coverage](https://support.developer.betfair.com/hc/en-us/articles/360002407732-What-data-is-provided-by-the-Historical-Data-service), [application keys](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687105/Application%20Keys), [read-only policy](https://support.developer.betfair.com/hc/en-us/articles/25033076334748-What-is-read-only-Betfair-API-access), [Australian charges](https://www.betfair.com.au/AUS_NZL/aboutUs/Betfair.Charges/)) | Website scraping is prohibited. The no-betting live pilot needs written Betfair Australia approval plus storage/use rights; delayed data cannot prove strict five-window evidence. ([Australian terms](https://www.betfair.com.au/AUS_NZL/aboutUs/Terms.and.Conditions/)) | **H**: independent supply/demand, disagreement, spread and liquidity can exist even when midpoint resembles Sportsbet | M | **Pilot 1 conditional; currently BLOCKED** |
| 2 | **TAB Studio Web Services** | Independent fixed WIN, tote approximates and pools, scratchings, substitutes, weather and track condition | First-party API is described as real-time across NSW/VIC/QLD TAB jurisdictions; production IDs, update timestamps and endpoint schema are not public in the evidence reviewed | Access is application-based, depends on account tier/history and Tabcorp discretion; commercial/development access needs approvals. Public results retain tote results/pools for 12 months and final fixed odds for five months, but that is post-race history, not pre-jump snapshots. Production cost/rate limits are quote/approval-only. ([API access page](https://help.tab.com.au/troubleshooting-support/applying-for-tab-studio-api-access), [fixed WIN/Place surface](https://help.tab.com.au/betting-on-racing/tab-fixed-odds-win-and-place-bet), [results retention](https://help.tab.com.au/betting-on-racing/accessing-racing-results)) | Personal terms restrict storage to temporary personal/non-commercial use and restrict derivative works; exact production endpoint/schema/rate limit/cost is unavailable before approval. ([Web Services terms](https://help.tab.com.au/troubleshooting-support/tab-web-services-terms-of-use)) | **H if licensed**: an independent bookmaker/tote consensus, pool state and explicit field changes; may still be highly correlated with Sportsbet | M | **Pilot 2 conditional; currently BLOCKED** |
| 3 | **BOM 10-minute observations** | Measured temperature, humidity/dew point, wind direction/speed/gust, pressure, recent rain and observed weather/cloud where available | State product code plus predeclared station/WMO ID; observation contains local and UTC time; immutable acquisition receipt adds fetch time | Anonymous `/anon/gen/fwo/` products are free for non-commercial use, are issued every ten minutes, and cover all states/territories through documented product codes; no anonymous-service availability/rate SLA is promised. The current file is overwritten, so prospective capture is required. Separate individual-station products retain 72 hours and climate archives are available separately. ([10-minute observations guide](https://www.bom.gov.au/catalogue/Observations-XML.pdf), [72-hour guide](https://www.bom.gov.au/catalogue/72_hr_historical_obs.pdf), [Weather Data Services](https://www.bom.gov.au/catalogue/data-feeds.shtml), [station directory](https://www.bom.gov.au/climate/data/stations/)) | Nearest station may be distant or not track-representative; fields vary by station/message type; anonymous service has no availability guarantee. Anonymous terms do not authorise commercial redistribution. ([anonymous FTP guide](https://www.bom.gov.au/catalogue/Bureau_of_Meteorology_Anonymous_FTP_Service_user_guide.pdf)) | **M-L**: realised weather and short-horizon change are physically different from a market price and from forecast weather, but likely effects may be small | L | **Fallback / later pilot** |
| 4 | **GRV FastTrack official field, scratching and dog-history surfaces (Victoria)** | Late-scratching time/reason, reserve allocation; official boxes, weights, trainer, trainer-history dates, trials, racing offences, grades, first split/PIR and starting-speed comments | Numeric meeting IDs in `/RaceField/ViewRaces/{meetingId}` and `/Meeting/Scratchings/{meetingId}`; numeric dog ID in `/Dog/Form?id={dogId}`; the scratching row can contain a late-scratching clock time and reason. ([example field](https://fasttrack.grv.org.au/RaceField/ViewRaces/1289861089), [example scratchings](https://fasttrack.grv.org.au/Meeting/Scratchings/1289861089), [FastTrack guide](https://grvaueprdfasttrackstr03.blob.core.windows.net/webcontent/documents/GRV%20FastTrack%20Client%20Portal%20User%20Guide%20Full%20Release.pdf)) | Public browsing and historical form/results exist and a free account exists, but no versioned historical pre-jump snapshot/API/rate SLA or research price is published; coverage is Victorian. FastTrack warns that pages can update late. | GRV says Race Information may not be downloaded/stored unless the user is approved and other uses require a separate written agreement. Do not collect until GRV gives permission that covers immutable research storage. ([GRV terms](https://fasttrack.grv.org.au/Home/TermsandConditions)) | **M**: explicit late-field events and trainer changes are novel; much historical first-split/PIR content overlaps already-tested data | M | **BLOCKED; negotiate rights, do not scrape** |
| 5 | **BetMakers CoreAPI / RaceOdds / Racelab** | Race fields/results, dynamic prices, tote data; vendor advertises ratings, speed maps, sectionals and performance factors | GraphQL/WS production endpoint is customer-issued and OAuth2 client-credentialed; test endpoint is `https://graphql.coreapi.gcpintau.tbm.sh/query`; UAT is limited to Queensland Gallop/Greyhound/Harness meetings and VIC tote prices. UAT responses are capped at 4 MB. ([CoreAPI introduction](https://docs.betmakers.com/docs/core-api/introduction/index.html), [authentication](https://docs.betmakers.com/docs/core-api/auth/index.html)) | Enterprise/contact sales; public pages do not state production price, rate limits, historical retention, or a greyhound field-level coverage SLA. RaceOdds says it supplies real-time official data and dynamic pricing across racing codes. ([fixed-odds product](https://betmakers.com/solutions/fixed-odds), [data products](https://betmakers.com/solutions/data)) | Sportsbet is a stated BetMakers partner, so content/prices may share upstream information; uniqueness must be contractually and empirically established. | **M**: potentially rich and reliable, but likely expensive and some duplication risk | H | **Acquisition inquiry only** |
| 6 | **Racing and Sports greyhound feed/content** | Greyhound speed maps, sectionals, statistics, scratchings and track label on the consumer form surface | The live form pages expose race numbers and displayed content, but the reviewed first-party material does not document a stable API, provider-native dog IDs, source update timestamps, historical delivery contract, cost or rate limits. ([greyhound form](https://www.racingandsports.com.au/form-guide/greyhound/australia), [company/services](https://www.racingandsports.com.au/about-us)) | Provider sells B2B data/content, but a greyhound API contract and schema require direct commercial inquiry. | Without a documented identity/timestamp contract, public-page collection fails the stop rule. Speed-map/sectional content may also overlap already-tested race-shape features. | **M-L** until schema proof | H | **Reject current surface; inquiry only** |
| 7 | **Other consumer bookmaker pages/apps (including Ladbrokes/Neds/TopSport)** | Another fixed WIN opinion could create consensus/dispersion features; Ladbrokes bet/hold percentages, source-timestamped fluctuations or `scr_time` could be novel if licensed and source-explicit | No first-party, approved, stable race/runner API with durable IDs and provider timestamps was established in the official material reviewed | Consumer platforms show prices, but live access, retention rights, rate limits and historical access are not documented as a research feed. Direct Ladbrokes WIN is already represented by the current TheDogs odds provider and is not new. | Do not reverse engineer private app calls, automate an account, or bypass bot controls. Any Ladbrokes pilot must first prove field-level distinctness from TheDogs and obtain data rights. | **L for direct Ladbrokes WIN; potentially M for unique licensed metadata** | H | **Reject unless provider supplies a written API/data licence and distinctness proof** |

## What is genuinely incremental

### Exchange microstructure, not another quoted favourite

Sportsbet WIN is a single bookmaker's quoted probability surface. A Betfair order book can contribute disagreement (`exchange_midpoint - sportsbet_probability`), transaction cost (back/lay spread), displayed conviction (price-level available size), side imbalance, and accumulated matching activity. Those variables can differ while the best quoted price is unchanged. This is the strongest plausible new information family, but it is only a hypothesis until prospectively tested.

### Independent bookmaker/tote consensus, but only through a licensed feed

TAB fixed WIN and tote state could identify cross-market disagreement, consensus dispersion and public-money movement. TAB explicitly says its pages show fixed odds and tote approximates side-by-side and its Web Services API provides real-time multi-jurisdiction wagering information. However, the personal API licence is not suitable for a durable shared research corpus. The correct next action is permission/schema acquisition, not scraping the web application. ([TAB fixed odds](https://help.tab.com.au/betting-on-racing/tab-fixed-odds-win-and-place-bet), [TAB API access](https://help.tab.com.au/troubleshooting-support/applying-for-tab-studio-api-access))

### Explicit field changes

The GRV scratching surface can state `Late Scr`, a clock time, reason, and reserve allocation against a numeric meeting ID; an official example displays all four. This is stronger provenance than inferring a scratching from runner-set disappearance. It is also Victorian-only and its terms block persistent research collection without approval. ([GRV example](https://fasttrack.grv.org.au/Meeting/Scratchings/1289861089), [GRV terms](https://fasttrack.grv.org.au/Home/TermsandConditions))

TAB also publishes scratchings and says its information service supplies scratching, weather and track conditions before the first race, but that is a meeting/day surface rather than proof of source-declared change timestamps. ([TAB phone/information service](https://help.tab.com.au/betting-on-racing/phone-betting-with-tab))

No reviewed primary source proves that Betfair, TAB, GRV or any consumer bookmaker is universally the *fastest* publisher. Betfair runner status/removal data can show an Exchange-side field change, but no official evidence found guarantees a scratching reason or replacement linkage. GRV is the most explicit observed surface, not a proven latency winner.

### Observed weather, not another forecast

BOM observations carry an observation time in local time and UTC, and the ten-minute state products update at `x:00, x:10, ... x:50`. They can add realised rain, gust, humidity and temperature changes near the track. The source explicitly warns that station message types and update frequencies differ, so a predeclared station map and freshness limit are mandatory. ([BOM observations guide](https://www.bom.gov.au/catalogue/Observations-XML.pdf), [latest-observations explanation](https://www.bom.gov.au/catalogue/observations/about.shtml))

BOM does **not** establish the greyhound racing surface condition. Track condition must remain a separate provider field with its own source timestamp; do not infer it from rainfall.

No genuinely new track-condition feed passed all gates. TAB may expose a declared condition through approved Web Services, but its exact source/update timestamp is unverified until the production schema is obtained; GRV public pages do not supply a licensed, versioned pre-jump condition feed in the evidence reviewed.

### Richer dog/trainer information

FastTrack documents dog-level form fields including first split, positions in running, starting-speed comments and trainer history; its portal also exposes trials and racing-offence tabs. The novel components relative to already-tested raw PIR/sectionals are trainer-change recency, trial events, and explicit administrative/availability changes—not another aggregation of the same splits. ([FastTrack guide](https://grvaueprdfasttrackstr03.blob.core.windows.net/webcontent/documents/GRV%20FastTrack%20Client%20Portal%20User%20Guide%20Full%20Release.pdf), [dog form example](https://fasttrack.grv.org.au/Dog/Form?id=1067557860))

This source is not acquisition-ready because the GRV storage/licensing boundary is explicit. Racing and Sports claims greyhound speed maps and sectionals but provides no reviewed public API identity/timestamp contract. BetMakers advertises API-delivered sectional/pace data, but greyhound-specific production coverage, cost and independence from Sportsbet must be established in a written proposal before implementation.

## Pilot 1 — Betfair Exchange five-window microstructure

### Exact hypothesis

At the same frozen pre-jump windows, Betfair exchange microstructure contains race-level information not already present in corrected Sportsbet WIN: specifically, a predeclared set of exchange midpoint divergence, back/lay spread, displayed-depth imbalance and matched-liquidity features will later improve out-of-sample probabilistic accuracy versus Sportsbet alone on the exact same accepted races. The 20-race acquisition pilot does **not** test that predictive hypothesis; it tests whether the necessary evidence can be captured without identity or timing ambiguity.

### Official surfaces and fields

This design is **conditional and not executable with the standard read-only key policy**. Use only a written-approved Betfair Exchange API path, never the consumer site or private BFF endpoints.

- Betting JSON-RPC endpoint: `https://api.betfair.com/exchange/betting/json-rpc/v1`; operations `SportsAPING/v1.0/listMarketCatalogue` and `SportsAPING/v1.0/listMarketBook`. ([official Betting API](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687158/Betting%2BAPI))
- Market discovery/catalogue: request Australian greyhound `WIN` markets and retain `event`, `marketStartTime`, `marketName`, `marketId`, runner metadata and `selectionId`. ([catalogue](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687517))
- Fixed-window book: retain market/runner status, in-play state, requested back/lay ladders and sizes, last traded price and matched/available fields allowed by the approved key/projection. `lastMatchTime` is the latest execution time, not a snapshot-publication timestamp; REST evidence therefore requires local request/receive timing. `spread` is derived locally as best lay minus best back and is not a provider field. ([market book](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687510/listMarketBook), [types](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687465))
- Optional continuous evidence only under the same written live read-only exception: TLS Stream endpoint `stream-api.betfair.com:443`, with `EX_BEST_OFFERS`, `EX_TRADED`, `EX_TRADED_VOL`, `EX_LTP` and `EX_MARKET_DEF`; retain provider publish time `pt`, clock `clk`, initial image and every delta. ([Stream API](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687396/Exchange%2BStream%2BAPI), [historical/Stream specification](https://historicdata.betfair.com/Betfair-Historical-Data-Feed-Specification.pdf))
- Engineering comparison only: official Historical Data, which Betfair describes as time-stamped Exchange price/market data for purchase and download. It must not be used to tune the prospective pilot definitions. ([developer overview](https://developer.betfair.com.au/))

Australian non-interactive authentication requires an account, App Key and session token; the documented certificate login endpoint is `https://identitysso-cert.betfair.com.au/api/certlogin`. Tokens/credentials must never enter evidence. ([AU/NZ bot login](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687915/Non-Interactive%2Bbot%2Blogin), [Australian API guide](https://www.betfair.com.au/hub/automation/betting-api/))

The Delayed key is not an evidentiary fallback: official documentation says delayed REST snapshots may vary from 1–180 seconds, runner `totalMatched`/`EX_ALL_OFFERS` are unavailable at that tier, and delayed Stream is forced to a three-minute conflation. That cannot establish `T-120/T-60/T-30/T-10/T-2` market state. A delayed-key transport/identity canary must be labelled `NON_EVIDENTIARY`, cannot be scored, and cannot be counted toward the reliability gate. ([application keys](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687105/Application%20Keys), [Stream access](https://support.developer.betfair.com/hc/en-us/articles/115003887871-How-do-I-get-access-to-the-Stream-API), [read-only policy](https://support.developer.betfair.com/hc/en-us/articles/25033076334748-What-is-read-only-Betfair-API-access))

### Native identity bridge

1. Persist Betfair `eventId`, `marketId` and every `selectionId` from catalogue.
2. Build a candidate join on race date, canonical venue, race number/name, advertised start and trap/runner metadata.
3. Accept only one Betfair market whose complete active runner set matches the current canonical TheDogs runner set one-to-one, including reserve/scratching state. Names may support diagnosis but cannot be the sole identity key.
4. Write an immutable bridge sidecar containing both provider-native IDs, canonical race ID, normalized comparison fields, exact runner-set hashes and the rejection reason for every non-unique candidate.
5. If Betfair does not expose an adequate trap/runner identifier for a race, reject it; do not infer from odds order or neighbouring races.

### Capture schedule and temporal guard

Only after written live read-only approval, schedule exactly five observations at `T-120`, `T-60`, `T-30`, `T-10` and `T-2` minutes relative to the frozen advertised start, aligning with the existing TheDogs prospective windows. At every request:

- record scheduled window, advertised start, request-start UTC/monotonic time, response-finish UTC/monotonic time and HTTP/server metadata;
- require market status open and `inPlay == false` (or the exact schema equivalent);
- reject responses received at/after the earliest evidenced actual jump or with any in-play transition;
- retain every blocked/missing attempt; never replace the race or fabricate a window;
- freeze race selection before the first window and do not add races based on observed liquidity.

### Immutable raw + receipt design

For each attempt write once under a new race/window directory:

- exact raw response bytes (or exact Stream frames for an authorised streaming variant);
- canonical request body excluding secrets, endpoint/operation name, app-key identifier hash, schema/client version and HTTP response metadata;
- SHA-256 of raw and request, byte count, capture timestamps, provider market/selection IDs, runner-set hash, status/in-play flags and parser outcome;
- bridge sidecar SHA-256 and canonical TheDogs race/runner-set identifiers;
- a manifest that hashes all files and refuses pre-existing output paths.

Credentials, session tokens and full app keys must never enter raw packets, logs or receipts.

### 20-race reliability gate

Freeze 20 otherwise eligible Australian greyhound races before collection, spanning at least three venues and more than one meeting date where Exchange markets exist. Pass only if:

- 100% of accepted races have an unambiguous one-to-one market and complete runner bridge;
- at least 95/100 planned race-window snapshots are accepted, and every missing/blocked attempt remains in the denominator;
- 100% of accepted snapshots have provider-native IDs, complete active runner sets, raw hash/receipt agreement, and no in-play/post-jump contamination;
- two-sided-book availability at `T-10` and `T-2` is reported empirically for every active runner; do not exclude thin races or require liquidity for capture validity;
- zero delayed/conflated-key observations, HTTP 503s, Stream `clk` gaps or slow-consumer states are accepted as fixed-window evidence;
- liquidity availability/missingness is reported by race and window without excluding thin races after inspection;
- deterministic replay reproduces every accepted normalized row and every rejection.

Any licence ambiguity, credential misuse, market ambiguity, or post-jump row is **BLOCKING**, irrespective of numeric completion rate.

### Expected storage

Cap the pilot at 1 GiB compressed/raw and report observed bytes per race/window rather than asserting an unverified precise size. Store outside canonical DB/runtime paths during reliability testing.

### Failure modes

- no Exchange market or insufficient Australian greyhound coverage;
- thin/one-sided order book, especially at early windows;
- advertised start drift and accidental in-play transition;
- reserve/scratching changes create runner-set disagreement;
- ambiguous meeting/race naming or missing trap metadata;
- delayed/conflated app-key data mistaken for live fixed-window evidence;
- API throttling or session expiry;
- licence does not permit persistent storage or intended derivative research.

All fail closed. The official limit is at most five `listMarketBook` calls per second per `marketId`; this narrow pilot stays below that bound. No consumer-page fallback, proxy, private endpoint, race substitution or post-jump repair is allowed. ([request limits](https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687478/Market%2BData%2BRequest%2BLimits))

### Later modelling experiment enabled

After a separately authorised fixed-N corpus is complete, freeze features and test a nested comparison on identical races:

- A: corrected Sportsbet WIN probability only;
- B: A plus Betfair midpoint divergence, log spread, predeclared depth imbalance, log matched/available liquidity, and fixed-window changes/missingness indicators.

Use a strict time split and paired race-level Brier/log-loss/ranking deltas with meeting-cluster uncertainty. Report effect estimates and uncertainty; do not claim profitability, EV or robust incremental signal from the reliability pilot.

## Pilot 2 — TAB Studio independent fixed-WIN/tote/field-change feed

### Exact hypothesis

At identical pre-jump windows, a licensed TAB feed contains independent market information absent from corrected Sportsbet WIN: cross-book probability disagreement, fixed-WIN consensus/dispersion, tote approximate/pool movement and explicit field-state changes. The later modelling hypothesis is that a frozen subset of those features improves out-of-sample proper scoring loss on the exact same races. The 20-race acquisition pilot tests access, identity, timestamp semantics, completeness and field distinctness only.

### Official surfaces and required contract

The documented access surface is TAB Studio at `https://studio.tab.com.au/`. TAB says the Web Services API supplies trusted real-time NSW TAB, VIC TAB/SuperTAB and QLD TAB (including NT/TAS/SA) wagering information. The public evidence does not expose production operation URLs, schema, field timestamps, rate limits or pricing; those must be obtained from the approved Studio documentation and frozen before implementation. ([API access](https://help.tab.com.au/troubleshooting-support/applying-for-tab-studio-api-access))

The access request must explicitly cover persistent report-only research storage and request these fields where licensed:

- provider meeting, race, market and runner/selection IDs;
- code, jurisdiction, venue, race number, advertised start and market status;
- unboosted fixed WIN price and status per active runner;
- tote WIN approximate, pool total/available pool state and source update time where supplied;
- scratching/late-scratching status, substitute/reserve allocation and effective time where supplied;
- source-declared weather and track condition plus provider update time where supplied;
- response/as-of timestamp and any sequence/version field.

TAB's public help confirms fixed odds and tote approximates are displayed side-by-side, its information service supplies scratching/weather/track condition, and historical results include pool totals, scratchings, substitutes and final prices. These prove field families exist on official TAB services, not that Studio exposes every field or a source timestamp. ([fixed WIN/Place](https://help.tab.com.au/betting-on-racing/tab-fixed-odds-win-and-place-bet), [information service](https://help.tab.com.au/betting-on-racing/phone-betting-with-tab), [results retention](https://help.tab.com.au/betting-on-racing/accessing-racing-results))

No collector should be written until the exact approved production endpoint/operation names, schema and licence receipt are available. The consumer racing website and undocumented browser/mobile endpoints are not fallbacks.

### Native identity and distinctness bridge

1. Persist every TAB provider-native meeting/race/runner/market ID from the approved fixture endpoint.
2. Candidate-match on code, jurisdiction, date, canonical venue, race number, advertised start and complete box/rug runner set.
3. Accept only a unique one-to-one TAB race and runner bridge to canonical TheDogs IDs. Name-only, odds-order or neighbouring-race joins are prohibited.
4. Write an immutable bridge sidecar with both native ID sets, complete runner-set hashes, normalized comparison fields, effective-box state and every rejection.
5. Prove source distinctness before admission: TAB fixed WIN must be a TAB-labelled feed, not a republished Sportsbet or Ladbrokes price. Direct Ladbrokes fixed WIN fails because current TheDogs market-history evidence already identifies Ladbrokes as provider. If a feed cannot declare provider at field level, reject it.

### Capture schedule and temporal guard

After the rights/schema gate, freeze 20 races and capture at `T-120`, `T-60`, `T-30`, `T-10` and `T-2` relative to frozen advertised start. At each attempt:

- record scheduled cutoff, request-start and response-finish UTC/monotonic time, server/response timestamp and provider as-of/version fields;
- require a strictly pre-jump/open market state and reject any response whose provider or receipt timing can cross the earliest evidenced actual jump;
- preserve prices, runner status, scratch/substitute state, tote approximates/pools and declared meeting conditions exactly as returned;
- retain missing and rejected attempts in the denominator; never substitute a race or infer a provider timestamp from receipt time.

### Immutable raw + receipt design

For each attempt, create a fresh write-once race/window directory containing:

- exact response bytes and SHA-256;
- canonical request excluding secrets, approved endpoint/operation, API/client/schema version and HTTP metadata;
- request/receive UTC and monotonic times, provider as-of/version, response status and byte count;
- provider meeting/race/runner/market IDs, canonical bridge hash and complete runner-set hash;
- field-level provider labels and presence bitmap for fixed WIN, tote/pool, scratching/substitution and meeting conditions;
- parser outcome, timing decision, rejection reason and manifest hashing all files.

Account credentials, API secrets, bet-slip/account data and personal information must never be captured. The pilot is read-only and must not call betting operations.

### 20-race reliability gate

Freeze 20 Australian greyhound races before the first capture, spanning at least three venues, more than one meeting date and every TAB jurisdiction actually licensed. Pass only if:

- 100% of accepted races have unique provider-native race/runner IDs and complete runner-set agreement;
- at least 95/100 planned race-window snapshots are accepted, with all failures retained;
- at least 18/20 races expose source-explicit TAB fixed WIN at `T-10` and `T-2` for every active runner;
- tote/pool availability, update age and missingness are reported empirically rather than made a pass prerequisite;
- 100% of accepted snapshots have raw/receipt hash agreement, field-level provider identity, verifiable pre-jump timing and no account/bet mutation;
- scratch/substitute transitions replay exactly when present, and zero ambiguous/provider-unknown rows enter the normalized corpus;
- deterministic replay reproduces every accepted row and rejection.

### Expected storage

Cap the 20-race pilot at 250 MB raw/receipts and report observed bytes per race/window. This is a conservative engineering cap, not a provider size guarantee. Keep the pilot outside canonical DB/runtime paths.

### Failure modes

- personal/commercial access application denied or rights do not permit durable research storage/derivatives;
- production schema lacks stable native runner IDs or source as-of/update timestamps;
- feed/provider label is absent or prices are republished Sportsbet/Ladbrokes rather than distinct TAB data;
- tote pool/approximate fields are final-only or updated too slowly for fixed windows;
- market suspension, advertised-start drift or post-jump contamination;
- scratches/reserves create runner-set disagreement;
- jurisdictional gaps, throttling, credential expiry or terms/schema drift.

All fail closed. Do not replace Studio with public-page scraping, account automation or a private app endpoint.

### Later modelling experiment enabled

After a separately authorised fixed-N future corpus, freeze a nested same-race comparison:

- A: corrected Sportsbet WIN probability only;
- B: A plus TAB normalized fixed-WIN probability, Sportsbet–TAB divergence, cross-book dispersion, predeclared tote approximate/pool level/change features, and explicit missingness/field-change flags.

Use a strict time split and paired race-level Brier/log-loss/ranking deltas with meeting-cluster uncertainty. Keep prices unboosted and compare the same runner set. No ROI, EV, betting or profitability claim is permitted.

## BLOCKING, IMPORTANT and OPTIONAL issues

### BLOCKING

- **Betfair live read-only policy:** Betfair explicitly says Live-key read-only data collection is not permitted and directs read-only users to Delayed keys. Delayed REST/Stream data cannot prove the five fixed windows. Obtain a written Betfair Australia exception/approved no-betting live-data product plus persistent research-storage rights before any evidentiary capture. Do not rely on consumer-page access; scraping is prohibited. Terminal state: `BETFAIR_LIVE_PILOT_BLOCKED_ACCESS_POLICY`. ([read-only policy](https://support.developer.betfair.com/hc/en-us/articles/25033076334748-What-is-read-only-Betfair-API-access), [terms](https://www.betfair.com.au/AUS_NZL/aboutUs/Terms.and.Conditions/))
- **Betfair identity:** no race enters the corpus without unique `marketId` plus complete `selectionId` runner-set alignment to canonical TheDogs identity.
- **TAB/GRV rights:** TAB personal Web Services and GRV FastTrack do not currently authorise the proposed durable shared corpus. TAB also withholds the production endpoint/schema/rate contract until approval. No collector until written permission and schema/field rights exist. ([TAB terms](https://help.tab.com.au/troubleshooting-support/tab-web-services-terms-of-use), [GRV terms](https://fasttrack.grv.org.au/Home/TermsandConditions))
- **Ladbrokes distinctness:** direct Ladbrokes fixed WIN duplicates the provider already found in current TheDogs market-history capture. No direct-price pilot; only source-explicit unique metadata may proceed after a field-level distinctness and rights gate.
- **Pre-jump integrity:** any in-play/post-jump or time-ambiguous payload is rejected, not repaired.

### IMPORTANT

- If Betfair grants the exception, measure greyhound market coverage, runner metadata, two-sided depth and liquidity prospectively before estimating modelling value. A Delayed-key canary may test transport/identity only and must remain `NON_EVIDENTIARY`.
- Treat provider publication/observation time and local receipt time as separate fields. A receipt proves when this system observed bytes; it does not invent a provider publication time.
- Freeze actual-jump authority and scheduled-start drift handling before either pilot.
- Keep BOM weather separate from track condition and keep GRV/TAB source classes separate from TheDogs and Sportsbet.
- If commercial/public use is contemplated, use BOM Registered User Services/licensing rather than assuming anonymous FTP rights or continuity. ([Weather Data Services](https://www.bom.gov.au/catalogue/data-feeds.shtml))

### OPTIONAL

- Request written proposals from TAB Studio, GRV data licensing and BetMakers with sample greyhound schemas, native IDs, timestamps, rate limits, historical retention rights, coverage SLA and price. This is procurement evidence, not collector implementation.
- Purchase a narrowly filtered Betfair Historical Data sample only to validate parser/replay mechanics after the licence gate; do not use it to reopen feature definitions or select races.
- If both market-data permissions remain blocked, a later BOM nearest-station pilot can use the documented ten-minute `IDX60920.xml` state products with UTC observation time, a frozen station map and raw+receipt capture. It must remain weather-only and never infer track condition. ([10-minute observations](https://www.bom.gov.au/catalogue/Observations-XML.pdf))
- Revisit trainer-transfer/trial features only if GRV or another rights-holder supplies a licensed native-ID feed. Do not pursue further first-split/PIR variants merely because another display republishes them.

## Strongest claims permitted

1. Betfair's official developer program specifies two-sided Exchange data and provider-timestamped Stream/history, making it the strongest potentially incremental source. The standard no-betting live collector is nevertheless blocked because Live-key read-only use is not permitted and Delayed data cannot prove the fixed windows. It does **not** prove Australian greyhound coverage, sufficient liquidity, incremental predictive signal or profitability.
2. TAB has a first-party real-time wagering API and potentially valuable independent fixed/tote information, but personal-use storage/derivative restrictions and unavailable production schema block immediate implementation.
3. BOM officially publishes ten-minute, UTC-tagged state observation products with measured weather fields; it is a feasible low-cost fallback within applicable use terms, but it does **not** prove track condition or predictive value.
4. GRV FastTrack visibly provides explicit late-scratching times/reasons and stable numeric meeting/dog identifiers on Victorian official surfaces, but its terms prevent treating public display as permission for an immutable research corpus.
5. Direct Ladbrokes fixed WIN is not genuinely new because current TheDogs capture already identifies Ladbrokes as provider; only separately licensed, source-explicit metadata absent from the existing payload could qualify.
6. No researched source supports a market-edge, betting-value, model-promotion or profitability claim.

## One next implementation goal

**Resolve the Betfair Australia access gate and, only if it succeeds, implement the isolated report-only 20-race five-window reliability pilot specified above.** Written confirmation must cover a live, read-only/no-betting API or data product; permitted fields, retention and derivatives; authentication/key tier; Australian fees and rate limits; and permission to retain raw-plus-receipt evidence. The bounded implementation must stop at catalogue/market-book capture, native-ID bridge, immutable receipts and deterministic replay outside runtime/DB paths.

Do not implement the collector until the access gate succeeds. If Betfair refuses, preserve `BETFAIR_LIVE_PILOT_BLOCKED_ACCESS_POLICY` and pursue the TAB Studio rights/schema request; do not substitute delayed Betfair data or consumer-page scraping.
