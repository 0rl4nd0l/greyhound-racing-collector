# Betfair ANZ greyhound historical CSV primary-source audit

Research snapshot: 2026-08-17 (Australia/Melbourne). This note uses only
first-party Betfair sources. It identifies the official free files and their
published semantics; it does not download the experiment corpus, certify any
join, or support a modelling or predictive-edge claim.

## Source decision

The correct source for the requested surface is the Betfair Australia and New
Zealand Automation Hub [Historic Data Listing](https://betfair-datascientists.github.io/data/dataListing/).
Betfair describes this curated lane as CSV, free to download without login,
limited to Australian and New Zealand markets, and containing runner-level
market snapshots including BSP/result and best available prices and market
overrounds at scheduled race start. The listing itself says that the site is
run by Betfair Australia and New Zealand and provides only markets taking place
in those countries. ([source overview](https://betfair-datascientists.github.io/modelling/dataSources/),
[listing and disclaimer](https://betfair-datascientists.github.io/data/dataListing/))

This lane is distinct from:

- the [Historical Stream site](https://historicdata.betfair.com.au/), whose
  files are JSON in compressed TAR archives and whose Basic, Advanced and Pro
  plans differ in update frequency and content; and
- the UK [PROMO BSP files](https://promo.betfair.com/betfairsp/prices), which
  Betfair says use GBP, exclude New Zealand, use UTC dates, and incorrectly
  label `market_id` as `event_id`.

Betfair Australia directs ANZ customers to contact
`automation@betfair.com.au` before purchasing Stream history. Basic Stream
history is described as free one-minute data without volume, but it is not the
curated free CSV product requested here. ([official historical-data overview](https://www.betfair.com.au/hub/automation/historical-betting-data/),
[official pricing-source comparison](https://betfair-datascientists.github.io/modelling/dataSources/))

## Published files and date packaging

The live official listing publishes annual greyhound ZIP archives for 2020
through 2025 and monthly CSVs for 2026 through July at the research snapshot:

| Coverage label | Official asset |
| --- | --- |
| 2020 | [`ANZ_Greyhounds_2020.zip`](https://betfair-datascientists.github.io/data/assets/ANZ_Greyhounds_2020.zip) |
| 2021 | [`ANZ_Greyhounds_2021.zip`](https://betfair-datascientists.github.io/data/assets/ANZ_Greyhounds_2021.zip) |
| 2022 | [`ANZ_Greyhounds_2022.zip`](https://betfair-datascientists.github.io/data/assets/ANZ_Greyhounds_2022.zip) |
| 2023 | [`ANZ_Greyhounds_2023.zip`](https://betfair-datascientists.github.io/data/assets/ANZ_Greyhounds_2023.zip) |
| 2024 | [`ANZ_Greyhounds_2024.zip`](https://betfair-datascientists.github.io/data/assets/ANZ_Greyhounds_2024.zip) |
| 2025 | [`ANZ_Greyhounds_2025.zip`](https://betfair-datascientists.github.io/data/assets/ANZ_Greyhounds_2025.zip) |
| 2026-01 | [`ANZ_Greyhounds_2026_01.csv`](https://betfair-datascientists.github.io/data/assets/ANZ_Greyhounds_2026_01.csv) |
| 2026-02 | [`ANZ_Greyhounds_2026_02.csv`](https://betfair-datascientists.github.io/data/assets/ANZ_Greyhounds_2026_02.csv) |
| 2026-03 | [`ANZ_Greyhounds_2026_03.csv`](https://betfair-datascientists.github.io/data/assets/ANZ_Greyhounds_2026_03.csv) |
| 2026-04 | [`ANZ_Greyhounds_2026_04.csv`](https://betfair-datascientists.github.io/data/assets/ANZ_Greyhounds_2026_04.csv) |
| 2026-05 | [`ANZ_Greyhounds_2026_05.csv`](https://betfair-datascientists.github.io/data/assets/ANZ_Greyhounds_2026_05.csv) |
| 2026-06 | [`ANZ_Greyhounds_2026_06.csv`](https://betfair-datascientists.github.io/data/assets/ANZ_Greyhounds_2026_06.csv) |
| 2026-07 | [`ANZ_Greyhounds_2026_07.csv`](https://betfair-datascientists.github.io/data/assets/ANZ_Greyhounds_2026_07.csv) |

There is a first-party documentation inconsistency: the current listing exposes
2020-2022 archives, while the general source overview says the ANZ monthly CSV
lane dates back to January 2023. Acquisition must therefore verify the actual
date range inside every required file and bind it by filename, URL, retrieval
time, byte size and SHA-256; prose coverage must not substitute for file
evidence. ([listing](https://betfair-datascientists.github.io/data/dataListing/),
[source overview](https://betfair-datascientists.github.io/modelling/dataSources/))

## Published CSV schema

The directly inspected official July 2026 CSV publishes this 40-column header:

```text
LOCAL_MEETING_DATE,SCHEDULED_RACE_TIME,ACTUAL_OFF_TIME,TRACK,STATE_CODE,RACE_NO,WIN_MARKET_ID,WIN_MARKET_NAME,PLACE_MARKET_ID,RACING_TYPE,DISTANCE,RACE_TYPE,SELECTION_ID,TAB_NUMBER,SELECTION_NAME,WIN_RESULT,WIN_BSP,PLACE_RESULT,PLACE_BSP,WIN_BSP_VOLUME,WIN_PREPLAY_MAX_PRICE_TAKEN,WIN_PREPLAY_MIN_PRICE_TAKEN,WIN_PREPLAY_LAST_PRICE_TAKEN,WIN_PREPLAY_WEIGHTED_AVERAGE_PRICE_TAKEN,WIN_PREPLAY_VOLUME,WIN_INPLAY_MAX_PRICE_TAKEN,WIN_INPLAY_MIN_PRICE_TAKEN,WIN_LAST_PRICE_TAKEN,WIN_INPLAY_WEIGHTED_AVERAGE_PRICE_TAKEN,WIN_INPLAY_VOLUME,PLACE_BSP_VOLUME,PLACE_MAX_PRICE_TAKEN,PLACE_MIN_PRICE_TAKEN,PLACE_LAST_PRICE_TAKEN,PLACE_WEIGHTED_AVERAGE_PRICE_TAKEN,PLACE_PREPLAY_VOLUME,BEST_AVAIL_BACK_AT_SCHEDULED_OFF,BEST_AVAIL_LAY_AT_SCHEDULED_OFF,BACK_MARKET_PERCENTAGE_AT_SCHEDULED_OFF,LAY_MARKET_PERCENTAGE_AT_SCHEDULED_OFF
```

Source: [`ANZ_Greyhounds_2026_07.csv`](https://betfair-datascientists.github.io/data/assets/ANZ_Greyhounds_2026_07.csv).
Schema equality across all required annual/monthly inputs must be tested rather
than assumed.

The file represents `WIN_MARKET_ID` as a numeric suffix such as `259600715`,
not as the API-style string `1.259600715`. Preserve the published value. Any
prefix addition must be an explicit, reversible derived field, never an
in-place rewrite. Betfair's ANZ API guide says catalogue and book data are
joined on common market ID and selection ID values. ([official ANZ API guide](https://www.betfair.com.au/hub/automation/betting-api/))

## Semantics that first-party evidence supports

### Race, runner and result

- `LOCAL_MEETING_DATE`, `SCHEDULED_RACE_TIME` and `ACTUAL_OFF_TIME` are three
  separate published fields. The two times are time-only strings with no UTC
  offset in the file.
- `TRACK`, `STATE_CODE`, `RACE_NO`, `WIN_MARKET_ID` and `WIN_MARKET_NAME`
  provide race/market metadata. The file does not publish an event ID.
- `SELECTION_ID` is the native market selection identifier. Betfair recommends
  market ID plus selection ID for joins between catalogue and book objects.
  ([official ANZ API guide](https://www.betfair.com.au/hub/automation/betting-api/))
- `WIN_RESULT` carries settlement-style `WINNER`/`LOSER` values, not a complete
  finishing position. Betfair documents those statuses as the runner status
  available after a market is settled. ([official result-status guidance](https://support.developer.betfair.com/hc/en-us/articles/115003887731-How-can-I-view-or-display-results-after-the-event))
- The file has no explicit scratch flag, reserve flag, removal timestamp,
  `removalDate`, market status, void reason, or final physical-box field.

### Scheduled-off quote, matched prices and BSP

- `BEST_AVAIL_BACK_AT_SCHEDULED_OFF` is a scheduled-start available quote. In
  its own worked example, Betfair converts it to `1 / price`, groups it by
  `WIN_MARKET_ID` to form the back market percentage, and calls it the best
  available price at the scheduled off. It is not BSP and is not an
  actual-off snapshot. ([official scheduled-off example](https://betfair-datascientists.github.io/wagering/2ndPlaceVoid/))
- `WIN_BSP` remains a separate field. Betfair describes BSP as being formed at
  the start of the event from SP backer/layer demand and eligible unmatched
  Exchange bets. A late non-runner can cause a calculated BSP to be revised.
  ([BSP explainer](https://www.betfair.com.au/hub/education/betfair-basics/betfair-starting-price-bsp/),
  [Exchange rules](https://www.betfair.com.au/AUS_NZL/aboutUs/Rules.and.Regulations/))
- The pre-play `*_PRICE_TAKEN` and `*_VOLUME` fields are aggregated matched
  market measures. They must not be substituted for the scheduled-off
  available quote. The official source overview distinguishes max/min,
  weighted-average and volume fields from BSP/result and from best available
  prices at scheduled start. ([source overview](https://betfair-datascientists.github.io/modelling/dataSources/))
- Betfair says BSP is offered on Australian greyhound win and place markets,
  but on New Zealand greyhounds only on win markets. Missing New Zealand place
  BSP is therefore not, by itself, an ingestion error. ([BSP availability](https://www.betfair.com.au/hub/education/betfair-basics/betfair-starting-price-bsp/))

### Known quality conditions

Betfair's own Topaz tutorial documents all of the following for these curated
greyhound results CSVs:

- duplicate rows are known to occur;
- a missing `WIN_BSP` may be associated with a very late scratching that was
  not removed before the race, or with some New Zealand races where BSP was
  unavailable due to lack of vision; and
- `BEST_AVAIL_BACK_AT_SCHEDULED_OFF` can be missing even when BSP exists if a
  late scratching caused existing market bets to be voided and BSP required
  manual reconciliation.

These are possible causes, not deterministic classifications for a null row.
The audit must retain the raw row and emit an explicit exclusion reason rather
than silently backfill a price or infer a scratch. ([official Topaz tutorial](https://betfair-datascientists.github.io/modelling/topazTutorial/))

## Fail-closed ambiguity register

### BLOCKING for identity or timing claims

1. **`TAB_NUMBER` is not proven to be the final physical box.** Betfair's own
   tutorial joins the CSV `TAB_NUMBER` to Topaz `rugNumber`, while Topaz keeps
   `boxNumber` as a different field. Betfair also says market description can
   contain greyhound reserve-runner box clarifications. Treat `TAB_NUMBER` as
   the published TAB/rug/trap identity only. Do not assert an effective box or
   accept a runner join when reserve/replacement state could make rug and box
   diverge. ([Topaz join example](https://betfair-datascientists.github.io/modelling/topazTutorial/),
   [ANZ API guide](https://www.betfair.com.au/hub/automation/betting-api/))
2. **The published times are timezone-naive.** `LOCAL_MEETING_DATE` suggests a
   local-date convention, but no first-party data dictionary found in this
   audit defines the timezone, daylight-saving rule, or cross-midnight
   convention. Preserve the raw strings. A derived instant needs an explicit
   track/jurisdiction timezone policy and must fail closed on ambiguity.
3. **`ACTUAL_OFF_TIME` derivation is undocumented.** The file names the field,
   but the inspected first-party sources do not define its clock source,
   precision, or whether it is official jump, Exchange suspension, or
   in-play-transition time. It supports only a raw provider-field claim, not a
   stronger actual-jump provenance claim.
4. **The scheduled-off quote boundary is not fully specified.** Betfair calls
   it the best available price at scheduled off, but the inspected sources do
   not say whether the snapshot contains virtual prices, its precise
   publication/sampling boundary, or a minimum available size. It is a quote,
   not proof that a useful stake could have been matched.

### IMPORTANT for deterministic ingestion

1. Do not use normalized runner name as identity. Use race metadata plus the
   native `WIN_MARKET_ID`/`SELECTION_ID` where a source-bound bridge exists and
   require one-to-one runner-set agreement. Names are consistency diagnostics
   only.
2. Detect exact duplicate rows separately from conflicting duplicates. Exact
   duplicates may be deterministically collapsed with counts retained;
   conflicts on the proposed race/runner key must be rejected as ambiguous.
3. Keep scheduled-off back, scheduled-off lay, matched pre-play aggregates,
   BSP and actual-off time as separate raw concepts. The CSV provides no
   price-at-actual-off field.
4. A blank price or result cannot by itself identify a scratch, reserve, void,
   no-race, or data error because the source lacks the corresponding status
   fields.

### OPTIONAL clarification from Betfair

Ask `automation@betfair.com.au` for the curated-file data dictionary covering:

- timezone and construction of scheduled and actual off;
- the authoritative meaning of `TAB_NUMBER` under reserves/replacements;
- `WIN_MARKET_ID` prefix convention;
- virtual-price and liquidity treatment of `BEST_AVAIL_*_AT_SCHEDULED_OFF`;
- duplicate-generation rules and result/null status mapping; and
- retention, derivative and redistribution rights for a report-only research
  corpus.

## Access and licensing boundary

The listing explicitly makes the files free public downloads and attaches a
disclaimer: Betfair gives no warranty of accuracy or completeness and use is at
the downloader's risk. It does not grant an unrestricted redistribution or
commercial-use licence. Betfair's general terms reserve rights in price data
and prohibit commercial use without prior written consent. Local report-only
acquisition must therefore not be broadened into publication, redistribution,
commercial use, or a claim of unrestricted reuse rights. ([listing disclaimer](https://betfair-datascientists.github.io/data/dataListing/),
[Betfair ANZ terms](https://www.betfair.com.au/AUS_NZL/aboutUs/Terms.and.Conditions/))

## Research verdict for the parent audit

The free ANZ CSV lane is the correct official acquisition source and exposes
the market fields required for a report-only overlap audit. It is not yet an
identity-safe joined surface. The parent audit should proceed only with frozen
raw files and deterministic exclusions, while keeping these claim boundaries:

- `TAB_NUMBER` is TAB/rug/trap identity, not proven effective physical box;
- `BEST_AVAIL_BACK_AT_SCHEDULED_OFF` and `WIN_BSP` are distinct;
- `ACTUAL_OFF_TIME` is a separate raw, timezone-naive field, not a price; and
- reserve/scratch, timezone, conflicting-duplicate or runner-set ambiguity
  remains fail-closed.

No source reviewed here establishes predictive improvement, profitability, or
an edge over Sportsbet.
