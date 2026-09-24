# Sportsbet external implementations — 2026-09-24

Static source inspection only. No provider requests, installs, entrypoints, target outcomes or credentials. Sources cloned by the coordinating agent were inspected as files. Local EventScraper history and PR comparison belong to the coordinating report.

## Pinned sources and verdict

1. **sportsdata-mcp: useful endpoint candidates, not a validated greyhound acquisition client.** Inspected commit `598b4947ddd3b0f15ca74350e77e13f25e53ccdf`. Its documentation claims anonymous public endpoint verification on 2026-05-25, but the racing examples are horses. Its executable specifications contradict several documented response shapes and do not enforce a response contract. No inspected Sportsbet greyhound racecard or batch fixture proves coverage. [Pinned repository](https://github.com/DanielTomaro13/sportsdata-mcp/tree/598b4947ddd3b0f15ca74350e77e13f25e53ccdf).
2. **sports-odds-scraper: actual direct JSON AFL implementation, unsuitable as a racing parser.** Inspected commit `1af1752cf2f8e3ae57675cd9fad174cbdfb8c32b`. Its sports service endpoints and singular price object are distinct from documented racing endpoints and `prices[]`. [Pinned repository](https://github.com/bensharkey3/sports-odds-scraper/tree/1af1752cf2f8e3ae57675cd9fad174cbdfb8c32b).

Neither is Sportsbet-owned documentation or proves current anonymous greyhound access. These sources justify preparing a tightly scoped capture, not enabling an adapter against invented fixtures. Unknown greyhound box/reserve semantics, fixed market discriminators and batch wrapper remain material.

## Racing endpoints and exact claimed shape

Base: `https://www.sportsbet.com.au/apigw/sportsbook-racing/Sportsbook/Racing/`.

| GET route | Documented shape | Unresolved |
|---|---|---|
| `AllRacing/{YYYY-MM-DD}` | `dates[].sections[].meetings[].events[]`; section `raceType`; meeting `id`, `name`, `classId`; event `id`, `raceNumber`, `startTime`, `type`, `category`, `statusCode`, `bettingStatus`, `httpLink` | Expanded example is horse; greyhound complete coverage, future-date horizon and schedule freshness are unproven. |
| `Events/{eventId}/Racecard` | Bare event object; `id`, `competitionId`, `competitionName`, `raceNumber`, `startTime`, `classId`, `type`, status/settlement fields, `markets[].selections[].prices[]` | Expanded runner is horse; no greyhound wire fixture, effective box/reserve rules or field completeness proof. |
| `Events/MultipleRacecards?eventIds=<CSV>` | A collection of racecards; spec binds CSV query | Documentation explicitly says wrapper varies; no exact envelope, maximum batch size, atomicity, order or missing-member semantics. |

[Discovery documentation](https://github.com/DanielTomaro13/sportsdata-mcp/blob/598b4947ddd3b0f15ca74350e77e13f25e53ccdf/documentation/Sportsbet.md#L337-L417), [single Racecard](https://github.com/DanielTomaro13/sportsdata-mcp/blob/598b4947ddd3b0f15ca74350e77e13f25e53ccdf/documentation/Sportsbet.md#L487-L587), [batch](https://github.com/DanielTomaro13/sportsdata-mcp/blob/598b4947ddd3b0f15ca74350e77e13f25e53ccdf/documentation/Sportsbet.md#L626-L644).

Class IDs 4 and 112 are described as greyhound classes, sourced from a results-class sample, not a captured greyhound racecard. `raceType` claims horse/harness/greyhound. Timestamps are documented as integer Unix seconds UTC; URL dates are Australian local racing dates. Demand actual type plus exact mapped event/meeting/date/race/start identity; class ID alone is insufficient. [Conventions](https://github.com/DanielTomaro13/sportsdata-mcp/blob/598b4947ddd3b0f15ca74350e77e13f25e53ccdf/documentation/Sportsbet.md#L103-L226).

## Markets, runner identities, restrictions

Documented market fields include `id`, `name`, `marketType`, `marketSort`, `availablePriceTypes`, `livePriceAvailable`, `statusCode`, `numPlaces`, `eachwayAvailable`, `placeAvailable`, `isDisplayed`. Runner fields include selection `id`, `runnerNumber`, `drawNumber`, `isOut`, `statusCode` and `prices[]`. The sample has MDP, TMD and L; its L entry contains winPrice/placePrice plus fractional numerators/denominators. Promo prices live separately in `powerPlayPricing`. The docs associate L with current fixed racing prices, but do not establish a comprehensive authoritative fixed/tote/SP taxonomy. No starting-price code is conclusively pinned by these examples. Do not select the first price, synthesize missing place terms or assume horse draw semantics are greyhound effective boxes. [Fields](https://github.com/DanielTomaro13/sportsdata-mcp/blob/598b4947ddd3b0f15ca74350e77e13f25e53ccdf/documentation/Sportsbet.md#L533-L587).

Check PRICED/OFF, active A, scratches, settlement flags and future start independently. Account for each scratched runner and require complete identities and required WIN/PLACE fields for every remaining active runner. Market name alone is insufficient to exclude promotions or other market variants. Selection IDs must survive normalization.

Raw Racecard can contain `results`, `exoticResults`, runner `result`, `shortForm`, `statistics`, `recentOddsFluctuations`, settlement flags and tips. WithContext adds surrounding events. Thus even a single-race request potentially exposes outcomes/history. Do not print or persist raw bodies as research inputs. Use a bounded pre-race allowlist and reject a resulted target before prediction admission; raw retention requires a separately established restricted path. No history/results/context endpoint is needed for the minimal validation. [Payload keys](https://github.com/DanielTomaro13/sportsdata-mcp/blob/598b4947ddd3b0f15ca74350e77e13f25e53ccdf/documentation/Sportsbet.md#L514-L587).

## Executable MCP specification differs from documentation

- AllRacing response hint uses `meetings`, `eventId`, `advertisedStartTime`, whereas documentation uses `dates.sections.meetings.events`, `id`, `startTime`.
- Racecard hint uses `event` plus singular selection `price`, whereas documentation uses a bare event plus selection `prices[]`.
- `selectionNames` is boolean in the spec, described as including full names; documentation says it is a string runner-name filter. Omit it to avoid partial fields.
- Batch hint says `racecards`; documentation does not fix the wrapper.

[Executable spec](https://github.com/DanielTomaro13/sportsdata-mcp/blob/598b4947ddd3b0f15ca74350e77e13f25e53ccdf/src/sportsdata_mcp/specs/sportsbet.yaml#L193-L259). The actual registry calls generic `request_json` then optional generic projections; these racing entries configure no racing normalization or completeness checks. `response_hint` is text, not validation. [Registry](https://github.com/DanielTomaro13/sportsdata-mcp/blob/598b4947ddd3b0f15ca74350e77e13f25e53ccdf/src/sportsdata_mcp/registry.py#L387-L412).

Default GET cache TTL is 60 seconds. Single Racecard sets `never_cache: true`; MultipleRacecards and WithContext do not. Cache hits return old decoded data without a fresh observation envelope. Upstream Age/Date/cache metadata is not propagated to our receipts. The generic HTTP client has its own token bucket and configurable retries; defaults are zero retries and empty retry statuses. Do not import that independent coordination mechanism. [TTL](https://github.com/DanielTomaro13/sportsdata-mcp/blob/598b4947ddd3b0f15ca74350e77e13f25e53ccdf/src/sportsdata_mcp/config.py#L19), [cache](https://github.com/DanielTomaro13/sportsdata-mcp/blob/598b4947ddd3b0f15ca74350e77e13f25e53ccdf/src/sportsdata_mcp/http_client.py#L364-L404), [retry defaults](https://github.com/DanielTomaro13/sportsdata-mcp/blob/598b4947ddd3b0f15ca74350e77e13f25e53ccdf/src/sportsdata_mcp/spec.py#L199-L200).

History: initial documentation commit 52f72ae dates to 2026-05-31; [5863dc9](https://github.com/DanielTomaro13/sportsdata-mcp/commit/5863dc9) on 2026-08-31 fixes cached live racing prices. Later account updates do not prove current greyhound reads.

Inspected integration tests verify registration and a live AllRacing response only as `dict`, converting gateway failures to xfail. No assertion of greyhound coverage, single-race odds completeness or requested/returned batch identity equality. Test search found cache-policy checks, not a Sportsbet greyhound racecard response fixture. [Tests](https://github.com/DanielTomaro13/sportsdata-mcp/blob/598b4947ddd3b0f15ca74350e77e13f25e53ccdf/tests/integration/test_sportsbet.py#L44-L114).

## AFL source implementation

`get_afl_events` GETs `/apigw/sportsbook-sports/Sportsbook/Sports/Competitions/4165/Events`, filtering MTCH plus two participants. `get_h2h_market` GETs `/Events/{id}/Markets` and selects first HH market. Calls use requests, 15-second timeout and raise_for_status. These are actual acquisition methods, but the sports service is distinct. [Code](https://github.com/bensharkey3/sports-odds-scraper/blob/1af1752cf2f8e3ae57675cd9fad174cbdfb8c32b/src/handler.py#L17-L55), [methods](https://github.com/bensharkey3/sports-odds-scraper/blob/1af1752cf2f8e3ae57675cd9fad174cbdfb8c32b/src/handler.py#L92-L111).

`parse_odds` sorts and truncates to two selections, reads `price.winPrice`, discards selection IDs, accepts missing prices as null and missing startTime as epoch zero. It stores status without enforcing open/suspended semantics, and has no places/scratches/price-code checks. Tests use hand-built OPEN/Active sports fixtures. [Parser](https://github.com/bensharkey3/sports-odds-scraper/blob/1af1752cf2f8e3ae57675cd9fad174cbdfb8c32b/src/handler.py#L114-L137), [tests](https://github.com/bensharkey3/sports-odds-scraper/blob/1af1752cf2f8e3ae57675cd9fad174cbdfb8c32b/tests/test_handler.py#L23-L73).

The loop sleeps 0.5 seconds between event requests, catches errors and continues, uploading surviving rows. No shared denial gate or batch completeness validation. The run timestamp precedes collection rather than recording individual response times; AFL rows omit it, although it names output objects. Module import constructs boto3 clients. [Loop](https://github.com/bensharkey3/sports-odds-scraper/blob/1af1752cf2f8e3ae57675cd9fad174cbdfb8c32b/src/handler.py#L813-L865).

## Licensing

sportsdata-mcp contains an [MIT LICENSE](https://github.com/DanielTomaro13/sportsdata-mcp/blob/598b4947ddd3b0f15ca74350e77e13f25e53ccdf/LICENSE), permitting software reuse with notices. That is not provider-data permission. No LICENSE/COPYING file or license/copyright declaration was found in sports-odds-scraper's tracked files/readable source. Treat the latter as an observed design reference; do not infer a reuse grant from public visibility.

## Minimal handoff to main provider owner

The main owner's instrumented browser baseline/PR #189 response recorder takes precedence. Reuse any already retained suitable response. No independent provider requests are authorized to this source-review agent.

The exact staged plan is in [the integration handoff](sportsbet_open_source_handoff_20260924.md#exact-minimal-live-validation-for-the-owner): first reuse the owner's already planned browser observation; if explicitly handed off, one exact single Racecard from existing mapping precedes any separately authorized AllRacing validation. Batch validation is deferred. These are conditional gates, not an endpoint sweep. Preserve all denial counters, backoff, reservation accounting and restricted-data boundaries. No source operations were executed here.

**Measured improvement: none.** No provider traffic occurred. Projected benefit only: one discovery request per date/approved refresh and one Racecard per capture; an evidenced batch could reduce N event requests to one. Neither runtime latency, coverage, current freshness nor successful receipt/capture rate has been measured here.
