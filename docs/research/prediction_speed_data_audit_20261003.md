# Speed, timing and neighbouring-runner data audit — 3 October 2026

**Disposition: qualified input semantics and coverage are insufficient for a new speed/pace experiment. Box occupancy is structurally available; occupancy alone does not measure pace pressure or interference.** A controlled comparison of existing verified model inputs is a separate, nearer-term preparation task.

This audit is preparation only. Source code is pinned to `a6f861ea9277d215f51bd61d7d0b361506bd3b02`. Historical evidence is the existing September 28 semantic/coverage audit at `3e75da9f850a027d759d83460b7f8b089fff9576`. No new provider access, target-result decoding, raw historical-record scan, fitting, tuning, implementation or runtime change was performed. No predictive performance conclusion is drawn.

## Feasibility table

| Input | What is supported | What is unknown / blocks use | Current decision |
|---|---|---|---|
| TheDogs CSV `1 SEC` | Local mapping retains the numeric field as `first_sectional`. Historical audit found positive values and missing entries. | Source-owned units, clock origin, physical timing call, runner versus leader, equipment/layout era, unique historical event identity and publication timing. | Numeric availability only; not qualified early speed. |
| CSV `PIR` | Audited single-digit values equalled historical finishing place; multi-digit sequences exist. | First-call meaning, call order/locations and special-code definitions. Local prose conflicts about the abbreviation. | Do not derive early position or direction from it. |
| `TIME`, `WIN`, `BON` | Historical audit found positive `TIME`; ingestion has separate `WIN` and `BON` mappings. | Exact runner/winner/best-of-meeting meanings, time units/precision, comparable conditions and as-of adjustment reference. Local `BON` mapping is called `bonus`; that name is not a provider definition. | Do not pool or substitute these fields; no adjusted-speed feature yet. |
| Best time / best first split | Parser stores best time plus date and a best-split number. | Best split has no dated event in this extraction; source timing definitions and the best-value update history are absent. | Descriptive metadata, not three comparable prior splits or proof of old decision-time availability. |
| FastTrack `split1` / comments | Adapter maps supplied `split1`, `run_home`, `pir`, `comment` into storage. | Versioned measurement definition, physical call, authorized eligible records and publication time. | Mapping proves plumbing, not source semantics or live availability. |
| Effective boxes / nearest occupied neighbours | Frozen candidate input includes verified literal box; retained roster allows geometric nearest occupied box and gap calculation. | Timely scratch/reserve changes, full-field qualified pace, and independent movement/interference observations. | Geometry is possible after as-of roster verification; pressure/style remains unknown. |

Primary mapping evidence: [CSV fields](../../csv_ingestion.py:170), [FastTrack adapter](../../src/collectors/adapters/fasttrack_adapter.py:80), [best-time/split parser](../../utils/expert_form_metadata.py:107), [candidate roster/box binding](../../src/predictor/comparison_candidates.py:81). Source-definition limitations are established by the prior [semantic audit](/home/l4nd0/greyhound-early-speed-neighbours-20260928/docs/research/early_speed_20260928_prior_work.md:100) and [measurement table](/home/l4nd0/greyhound-early-speed-neighbours-20260928/docs/research/early_speed_20260928_results.md:77). Neither local field names nor generic timing definitions from another source authenticate this export. No retained source-owned definition sufficient to pass those gates was established in these inspected materials; this is not a claim about every archive or source.

The current frozen form methods select 16 summary form features; one candidate adds scalar box. Sectionals, adjusted elapsed times and neighbour interactions are not selected. A scalar box coefficient also differs from categorical or layout-specific box effects. [Inputs](../../src/predictor/market_form_residual.py:38), [recipes](../../src/predictor/comparison_candidates.py:11), [scalar scorer](../../src/predictor/comparison_candidates.py:119).

## Historical coverage is an upper bound, not current availability

The existing audit covered **331 races, 2,360 runner-targets and 27 dates**, June 10–July 8. Its semantic sample was fixed by salted race-identity hash before card reads: 39 races, one per literal venue, containing 1,344 repeated historical observations. Full-population aggregates below include repeated prior starts, so they are not counts of independent timing events. We reused the report; we did not reopen that population. [Population and sample](/home/l4nd0/greyhound-early-speed-neighbours-20260928/docs/research/early_speed_20260928_results.md:45).

| Existing aggregate | Count / denominator |
|---|---:|
| Positive `1 SEC` / missing `1 SEC` | 7,838 / 3,617 of 11,455 repeated observations |
| Positive `TIME` | 11,455 of 11,455 repeated observations |
| Single-digit `PIR` matching historical finish | 6,919 of 6,919 |
| At least one positive sectional anywhere | 2,143 / 2,360 runner-targets; 246 / 331 complete fields |
| Three positive sectionals anywhere | 1,663 / 2,360; 118 / 331 fields |
| One positive sectional, same nonambiguous layout/distance | 738 / 2,360; 24 / 331 fields |
| Three positive sectionals, same nonambiguous layout/distance | 290 / 2,360; 7 / 331 fields |
| Fully verified early-call measurement and identity | **0 / 2,360; 0 / 331 fields** |

Source: [measurement and coverage tables](/home/l4nd0/greyhound-early-speed-neighbours-20260928/docs/research/early_speed_20260928_results.md:77). The seven numeric fields span only four dates. Ambiguous QOT/RICH/MURR layout aliases were excluded; even an unambiguous venue/distance label does not establish equal timing points or conditions. Historical date/track/distance/box corroboration lacks a native historical race/runner join in the audited 17-column export. All 39 sampled original exports matched their retained hashes and had the same limited headers; duplicating those CSVs would not add definitions.

The previous audit also found 169 fields with an unoccupied box, 163 with an internal gap and 482 runners whose nearest occupied neighbour was more than one box away. These describe the retained roster, not the cause of absence or subsequent scratch/reserve state. [Roster findings](/home/l4nd0/greyhound-early-speed-neighbours-20260928/docs/research/early_speed_20260928_results.md:85).

**Today's availability remains unmeasured by this audit.** Root's metadata-only check found a complete 90-race discovery inventory timestamped 05:48:32 UTC and zero verified captures for the new profile at the time of that check; the inventory timestamp is not the audit time. That is a dated operational observation, not a current timing-coverage census: discovery contains scheduling/venue fields, not runner sectionals. No previous population count should be attached to today's collection. Pre-jump manifests, receipts, complete rosters and history roles are supported by the retention code, but code support is not evidence of usable records. [Retention contract](../../src/predictor/retained_inputs.py:180).

## Prior work and fixed gates

Already explored constructions include same-venue/distance time and sectional summaries, field ranks/gaps, faster box±1 neighbours, and box/market pressure interactions. The later unexecuted construction used nearest **occupied** neighbours and retained vacancy gaps. This audit adds no renamed duplicate recipe and uses no previous predictive scores. [Recipe identities](/home/l4nd0/greyhound-early-speed-neighbours-20260928/docs/research/early_speed_20260928_results.md:19).

That neighbour protocol requires source-qualified runner-specific early times, three earlier comparable starts for every runner, a complete as-of field, at least 50 training races on five dates, 30 test races on five dates and representation in every fixed period. Those are feasibility floors, not power justification. Definitions and population both failed previously. Missing values must remain unknown; no finish/place proxy, slow-start imputation, threshold relaxation or post-result roster reconstruction. [Exact gates](/home/l4nd0/greyhound-early-speed-neighbours-20260928/docs/research/early_speed_20260928_protocol.md:41).

## Next bounded metadata census, before any new record access

No new census was run here. First write an immutable manifest containing:

- Exact current engineering cohort and exclusion/reservation identities; no automatic inclusion of scientific or weekend-pilot membership.
- The active profile `/mnt/tenn-nvme2/tenn/greyhound-persistent-engineering-20261003-02`, its pinned configuration, exact current-day allocation path, and **explicitly enumerated** retained manifest/completion/form-metadata/identity-receipt paths and hashes. Resolve pointers once, bind their bytes and reject traversal outside those paths; no recursive archive scan.
- Allowed projections: schema/parser version; source/layout/distance/definition IDs; native event/runner IDs; effective box and roster identity/hash; occurrence/publication/capture/seal timestamps; presence/missing-reason flags for timing and history roles; declared accepted/rejected-history counts. Exclude odds values, finish/winner/place, target results, historical row bodies and model outputs. If a metadata document lacks the necessary declaration, report unknown; do not infer it by opening a raw card.
- Exact census deadline and finite file/byte ceiling derived from the enumerated manifest; reject additional files. Report all eligible/excluded/missing records and failed identity/as-of checks.

A later values-level coverage audit needs a separately reviewed reservation-cleared sample and date-first projection contract before historical fields are decoded. It is not implicitly included in the metadata census.

## Minimum prospective retention and separate cost units

Before retaining a new timing family, establish a versioned source definition: units, clock origin, physical call, individual/leader/winner meaning, layout/distance/era, precision and special codes. Each authorized observation then needs native race/runner IDs, occurrence/publication/capture times, definition ID, original field/value, parsed value or explicit missing reason, immutable raw hash and parser version. Keep the decision-time active roster and substitution events separately bound to that observation. Adjustment references must be strictly available before the decision; a later condition estimate cannot be backdated. Style would need its own independently observed segment/direction/observer/timestamp contract. [Existing acquisition specification](/home/l4nd0/greyhound-early-speed-neighbours-20260928/docs/research/early_speed_20260928_results.md:147).

| Cost unit | This audit | Prospective proposal before activation |
|---|---:|---|
| Python provider requests | 0 | Count each required definition/record fetch separately; zero additional fetches if exact authorized retained bytes suffice. |
| Browser navigations | 0 | Count separately from Python; do not switch transport to bypass a denial. |
| Source operations | 0 | Account against the source's own operation ledger; do not infer this count from requests. |
| Capture attempts | 0 | Reuse authorized capture only if scope and retained payload suffice; every new attempt needs its own allocation and remains consumed on failure. |
| Result requests | 0 | Not needed for this semantic audit; any official-result endpoint use requires separate result authority and accounting even if intended to learn timing semantics. |
| Local bytes / CPU / records | Code and existing report reads only; no timed benchmark | Size the explicit metadata manifest and measure processing cost before setting a finite census ceiling; do not invent provider allowance from local capacity. |

No universal numeric collection allowance can be justified until the source route, eligible membership, refresh frequency and required fields are specified. Prefer extending retention of already-authorized, hash-bound bytes after review; keep any new discovery, refresh, retry/recovery and result work separately costed with explicit headroom and existing denial handling. Current frozen models and live collection configuration remain unchanged.
