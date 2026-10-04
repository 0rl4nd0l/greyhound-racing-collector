# Speed-data feasibility: retained definitions and code contract

**Decision: the retained evidence supports a repeatable offline plumbing audit, but does not yet support an early-speed or adjusted-time feature.** The new validator is implemented and tested; it does not fetch records, modify collectors, interpret unqualified numeric timing values, or qualify a model. Current engineering races have not been counted as having usable timing history merely because their four forecasts verified.

## What is now executable

Run `python -m scripts.audit_speed_source_contract` from this checkout. The script reads exactly five code/definition files (76,780 bytes in the retained run, ceiling 1 MB per file), using Python AST without importing the scraper, adapter or database modules. The [retained output](speed_source_contract_20261004.json) gives exact hashes, field mappings, missing adapter keys and seven unqualified definitions. Exit 0 means the audit executed; `SOURCE_SEMANTICS_NOT_QUALIFIED` means no timing use is qualified. An edited “verified” boolean cannot grant qualification. Dynamic key expressions and missing/duplicate field declarations require review.

The code inspects literal assignments and reads, **not arbitrary program dataflow**. It deliberately reports `adapter_keys_without_parser_literal_assignment`, not runtime coverage. It does not establish whether this legacy FastTrack adapter is used by the installed collector. Nine focused tests cover the actual retained code, fabricated mapping changes, definition claims, dynamic keys and bounded reads; all passed in 0.06 s with kernel networking denied and production filesystem read-only. No target or historical race records were opened.

[Implementation](../../scripts/audit_speed_source_contract.py:100), [focused tests](../../tests/test_speed_source_contract.py:17). The [compatibility receipt](speed_source_current_code_compatibility_20261004.json) verifies five source/document files byte-identical to reviewed collector `8f33f1eaa86ec8af1815ed353b3c69d95107ed98`; the expert timing parser AST is also identical. Its surrounding file has a reviewed target-jump datetime change, which supplies no new sectional definition. This is source compatibility, not a current runtime-health claim.

## Concrete findings

| Field/path | Established from retained primary code | Consequence |
|---|---|---|
| FastTrack parser → adapter | `_parse_race` assigns box, name, finish-position and race-time keys. The adapter expects `split1`, `run_home`, `pir`, `comment`, `margin`, `sp`, none assigned by that parser. | A nullable storage column does not prove acquisition. Fixing extraction requires a source-defined format first. This is result-page parser code, not a demonstrated pre-race route. |
| TheDogs `1 SEC` | CSV aliases map it to `first_sectional`. | Still unknown: whose clock, units, start trigger, physical call and applicable layout/distance/era. |
| `TIME`, `WIN`, `BON` | Local aliases label TIME individual/race time, WIN winner time, BON `bonus`; a different local FastTrack note calls BON best-of-night. | Do not interchange these fields or use a later meeting reference as an as-of adjustment. Local descriptions are not provider definitions. |
| `PIR` | Local documents variously say Performance Index Rating, Points in running and Position In Running. | No early-position interpretation. The prior equality finding is limited to its audited historical sample. |
| Best first split | Parser stores `best_first_split`, while only best total time gets a date. | A best value is neither three comparable prior starts nor a timestamped historical event. |
| Native event identity | FastTrack parser can extract race ID, but adapter resolves the base race by name and dog by name. | No demonstrated unique cross-source historical timing join. Do not repair with fuzzy matching. |

Primary references: [parser](../../src/collectors/fasttrack_scraper.py:311), [adapter](../../src/collectors/adapters/fasttrack_adapter.py:44), [CSV mappings](../../csv_ingestion.py:162), [local field map](../../docs/fasttrack_field_map.md:39), [local schema](../../docs/schema_diff_fasttrack.md:97), [best-split parser](../../utils/expert_form_metadata.py:107). The local documents' backup origin and unresolved source definitions are retained in the [seven-field matrix](prediction_source_definition_matrix_20261003.json); none is newly authenticated here.

## Coverage remains explicitly dated

The September 28 audit's June 10–July 8 population was 331 races / 2,360 runner-targets. It reported 7,838 positive `1 SEC` values among 11,455 repeated history observations; only seven complete fields met the numeric three-start/same-layout-distance upper bound, and **zero** met verified measurement plus identity. These are reused report aggregates, not independent timing-event counts and not October feed coverage. The numeric audit was not rerun. [Original measurement and coverage report](/home/l4nd0/greyhound-early-speed-neighbours-20260928/docs/research/early_speed_20260928_results.md:77).

The original neighbour hypothesis requires three qualified comparable starts per runner, a complete as-of field and fixed date/race coverage. Existing work already tested sectional ranks/gaps, adjacent-box pressure and related summaries; renaming those recipes is not a new experiment. Nearest occupied neighbours preserving vacancy gaps was the distinct unexecuted construction. Literal box geometry is implementable from a verified roster, but does not establish pace, movement or interference. [Prior protocol](/home/l4nd0/greyhound-early-speed-neighbours-20260928/docs/research/early_speed_20260928_protocol.md:41), [prior-work inventory](prediction_speed_data_audit_20261003.md).

## Smallest external requirement and next implementation boundary

Root can seek a **versioned official field/export dictionary**, using an already verified source route and separately accounting any requests. No new lookup was performed by this agent. The exact questions are:

1. For TheDogs `1 SEC`, is the value this runner's or the leader's? What are the unit, clock start, physical call, layout/distance/era and missing/special codes? How is its native historical event and runner identified?
2. For TheDogs `TIME`, `WIN`, `BON`, identify each clock/reference and whether it is measured or derived. For BON, when is the meeting reference first published, can later races change it, and what precise layout/distance grouping applies?
3. For each source's PIR, provide the call-position/code dictionary and version scope; for FastTrack Split1/run-home, provide the source-specific definition and exact export/form schema. Generic racing education about a leader's split does not authenticate a runner's CSV value.
4. For Best 1st Split, identify the selected event, selection window and publication/update time. For all timing observations, specify native event/runner IDs and publication/revision semantics.

After a definition is independently verified, the next bounded work is an explicit outcome-blind pre-race manifest with source/definition/layout/native identity and occurrence/publication/capture/seal times, followed by a values-free coverage projection of that exact membership. A parser can then be tested against a retained source fixture with the defined fields. Do not first patch the scraper to guess selectors or repurpose total time/PIR as speed. No current membership, frozen feature, source control, result access or allowance is amended by this audit.

All additional Python provider requests, browser navigations, source operations, capture attempts, result requests, model fits and runtime changes: **zero**. Existing reports, original forecasts, source holds and exclusions remain untouched.
