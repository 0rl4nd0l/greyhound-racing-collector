# TheDogs pre-race form: four retained surfaces

The detailed runner routes contain usable additional history structure. They do **not** yet establish a usable early-pace feature. This audit inspected one current pre-race card and one selected runner; it is not a feed-wide coverage claim. No models, features, runtime settings or original evidence were changed.

## Exact scope and availability

The selected Temora Race 8 was scheduled for 2026-10-04 20:42 AEDT. The saved normal card was captured at 20:25:48.887650. Root acquired its expert card at approximately 20:28, then the selected normal and expert runner details at 20:36:04.465547 and 20:36:08.660200. All four responses are exact-URL HTTP 200 bodies, hash-bound to receipts and captured before the scheduled jump. The expert parent fetch had a local post-retention parsing error; its original failure receipt remains unchanged, and independent offline parsing succeeded.

Root consumed three new Python/source requests for this investigation, reusing the normal card. This audit agent made zero network requests. The selected routes came from actual page attributes, not guessed URLs. No target result pages or protected result databases were opened. Historical rows were processed only to project field names, availability counts and outcome-blind identity/date structure; no historical timing, odds, placement, winner or forecast values are reported.

| Surface | Observed structure | Additional availability and limitation |
|---|---|---|
| Normal race card | Eight runner groups; entry IDs, profile IDs and standard lazy-detail references | `Av 1 SEC` column exists, but all eight entries are missing in this sample. An earlier separate two-card audit found numeric presence in 10/12 entries; these samples must not be pooled into a current feed estimate. |
| Expert race card | Eight runner blocks, 32 summary cells, 16 box/distance summary tables; eight empty history loaders | The current expert parser extracts best time/date and best first split for 8/8 runners here. Full per-start histories are not embedded in those loaders. |
| Normal runner detail | Six tables including history, box/distance summaries and best-track-time tables; normal history has 17 columns | Six rendered history rows represent five unique exact date-plus-event-link keys. An extra row has the source class `runner-form__last-win`; counting six separate starts would overcount. A show-more control exists, but further pagination completeness is not established. |
| Expert runner detail | One history table with 16 columns | Five rendered history rows share all five exact date-plus-event-link keys with normal detail. This sample contains no additional distinct history event beyond normal detail. |

All five shared event keys also have identical distance text. Track display text differs between normal and expert (six versus four characters); raw track strings are not interchangeable. A combined raw date/link/track/distance join therefore deliberately fails. Any adapter must normalize track namespaces through an independently verified mapping while retaining the exact event URL and date; this audit does not silently substitute labels. All history date epochs are earlier than the target jump, but these dates are not original publication timestamps or proof of unlimited historical completeness.

## Fields and source-owned labels

Both detail tables expose placement, box, weight, distance, date, track, grade, TIME, WIN, BON, 1 SEC, margin, counterpart-name, PIR, starting-price and video columns; normal detail additionally has a result-link column. We inspected only names and structural/presence projections, not protected outcome values.

The expert parent sort control provides direct source-owned key mappings:

| Label | Source key |
|---|---|
| TIME | `finish_time` |
| WIN | `race_finish_time` |
| BON | `best_of_night_time` |
| 1 SEC | `first_sectional_time` |
| DATE | `actual_start_time` |
| PLC / BOX / WEIGHT | `finish_position` / `run_box` / `weight` |
| DIST / TRACK / GRADE | `race_distance` / `track` / `grade` |
| MGN / SP | `winner_margin` / `starting_price` |

PIR cells use `runner-form__in-running-places`. This corrects the earlier local interpretation of PIR as a generic “rating” at the schema-label level. BON likewise maps to a best-of-night field; the earlier “bonus” interpretation was incorrect. Neither class names nor sort keys establish exact call positions, runner-versus-leader clock semantics, units, timing equipment, era/layout comparability, or how missing values and averaging are defined.

TIME, WIN and BON are numeric in all six normal rows and all five expert rows. **1 SEC is missing in every one of those rendered rows**; early-pace coverage is not demonstrated by this sample. PIR is nonmissing in all rows, but is not accepted as an early-position measure. The selected history rows had no timing-cell or header tooltip definitions. All four inspected bodies have six script elements and no standalone `application/json` or `application/ld+json` script; this is not a claim about arbitrary external JavaScript or every site endpoint.

## Identity and current parser gaps

The standard parent entry ID is bound to its profile ID and exact `/dogs/runner/{entry-id}` URL. The expert parent independently binds that same entry to `/dogs/runner/{entry-id}/expert-form`. Both actual HTTP receipts preserve those exact URLs. Normal detail contains one native profile attribute matching the parent. Expert detail contains no native profile or entry attribute; its identity proof therefore depends on retaining the verified parent entry-to-route binding and response receipt, not inferring identity from display names.

The current `utils/expert_form_metadata.py` parser extracts summary fields, not full lazy history tables, source sort keys, native entry/profile identifiers or lazy URLs. The inspected native browser path does not acquire these runner-detail histories as part of this metadata parser. Existing standard/API identity evidence elsewhere in collection remains separate. This audit did not alter that pipeline.

## Concrete next work and unresolved requirements

1. A bounded, default-off retained-history adapter is feasible: retain exact parent/route/receipt hashes; extract history schema and event/date/track/distance identity; deduplicate repeated normal rows; account for show-more completeness; preserve missing fields. It must not invent an early split or treat raw track labels as canonical identity.
2. Source documentation is still needed for the physical definition of `first_sectional_time`, the averaging population and missing-value treatment behind `Av 1 SEC`, PIR call encoding, and the comparison/publication basis of best-of-night and elapsed times. No inference from names qualifies these features.
3. Establish prospective coverage across prespecified venues/layouts using verified pre-race snapshots, separate from this one-runner inspection. Existing elapsed-time presence is promising input availability, not evidence of predictive benefit. Training, feature activation and scoring are outside this audit.

The separate GRV RaceData/Isolynx documentation lead is retained in `speed-primary-documentation-followup-20261004.json`. It describes another measurement source; it does not retroactively authenticate TheDogs fields. No contact or API acquisition was performed here.

## Evidence

All paths below are beneath `/home/l4nd0/greyhound-recovery-20261003/persistent-operation/`:

- `thedogs-live-form-inspection-20261004/selection.json`, `EXPERT-COMPARISON.json`, `paired-schema-inventory.json`, `expert-fields-projection.json`, `current-parser-presence.json`.
- `thedogs-runner-inspection-20261004/selection-addendum.json`, `0-receipt.json`, `1-receipt.json`, `runner-schema-projection.json`, `history-coverage-projection.json`, `history-identity-projection.json`, `event-join-projection.json`, `event-namespace-projection.json`.
- Normal-detail body SHA: `a9750a21093058426ed316dd5c691dc27b59954e6ffd703ee624ac4badaa0364`.
- Expert-detail body SHA: `6ecbee5c0a9392dad1df10619fe7074169ba2fa348b30d1f376a3474ea4ae440`.

Projections ran offline with kernel networking disabled, production evidence mounted read-only, and only the new audit directory writable. The source snapshot audited was `8f33f1eaa86ec8af1815ed353b3c69d95107ed98`; the summary-parser file SHA is `185b840a29d8fb31d37266aa491bc2707a35e23ea0b3fd6389fef85980f8cf36`.
