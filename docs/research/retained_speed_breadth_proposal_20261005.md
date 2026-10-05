# Retained speed evidence breadth proposal — 5 October 2026

## Decision

A broader **retained-input coverage audit is feasible without new requests**. Freeze the existing 82 October 1–3 forecast-race members, including all eight result quarantines, and measure timing-field presence only. This is an input audit, not outcome evaluation, proof of early-speed measurement, or a new scientific cohort. The original four-surface audit remains valid for its one runner and five events; it does not establish wider coverage.

The exact [proposed manifest](/home/l4nd0/greyhound-recovery-20261003/persistent-operation/research-continuation-20261005/speed-breadth-proposal/manifest.PROPOSED.json) has SHA256 `922456735d4117092fbec50333d68bbb8743008fc2c73b1fc8efa48529148223`. It is a fixed file-selection proposal, not executable authority. The [metadata inventory](/home/l4nd0/greyhound-recovery-20261003/persistent-operation/research-continuation-20261005/speed-breadth-proposal/inventory.json) has SHA256 `6cdc2724ad11dda83584cb45ad6397277ce89fab88ff534e8bf321959691e638`. Its builder is [inventory_metadata.py](/home/l4nd0/greyhound-recovery-20261003/persistent-operation/research-continuation-20261005/speed-breadth-proposal/inventory_metadata.py); it reads CSV headers only, hashes bodies opaquely, and inspects allowlisted pre-race metadata. No forecast probabilities, official result bodies, databases, model binaries, timing-row values or performance metrics were opened or decoded.

## Measured input breadth

There are **82 cards, 583 target-runner slots, 20 venue labels, 26 target distances and 35 venue/distance groups**. Runner slots are not unique lifetime dogs. Dates are October 1:20, October 2:37, October 3:25. The denominator is the existing forecast cohort; it is selected for previously admitted forecasts and cannot estimate coverage across all offered or rejected races.

All 82 have an exact hash-matching accepted CSV, raw export, sidecar, primary page and HTTP200 receipt captured before the target jump. Receipt race keys and source URLs agree. All 82 accepted CSV headers contain `TIME`, `WIN`, `BON`, `1 SEC` and `PIR`. **No nonmissing-field count has been measured in this inventory.** All 82 sidecars retain expert summary metadata; their declared expert source hashes alone do not prove the expert HTML body is still available. No raw expert/lazy body was located by an exact reference in this fixed worker metadata. This inventory did not search unrelated runtime trees or infer body retention from summary values.

| Venue | Target distance | Cards | Runner slots |
|---|---:|---:|---:|
| AP_K | 530m | 2 | 12 |
| AP_K | 595m | 1 | 7 |
| BEN | 425m | 1 | 8 |
| BEN | 500m | 2 | 12 |
| CANN | 520m | 3 | 20 |
| CANN | 601m | 2 | 12 |
| DUBBO | 318m | 2 | 16 |
| DUBBO | 400m | 1 | 6 |
| GEE | 400m | 2 | 16 |
| GEE | 460m | 1 | 6 |
| GRAF | 350m | 1 | 8 |
| GRDN | 272m | 3 | 24 |
| GRDN | 400m | 10 | 76 |
| GRDN | 515m | 1 | 7 |
| GUNN | 340m | 3 | 24 |
| HEA | 300m | 1 | 8 |
| HOBT | 461m | 1 | 7 |
| LADBROKES-Q1-LAKESIDE | 390m | 6 | 47 |
| LADBROKES-Q1-LAKESIDE | 457m | 3 | 17 |
| LADBROKES-Q1-LAKESIDE | 642m | 1 | 8 |
| LADBROKES-Q2-PARKLANDS | 520m | 3 | 20 |
| MAND | 400m | 3 | 20 |
| MAND | 488m | 2 | 11 |
| MEA | 525m | 2 | 15 |
| MURR | 300m | 1 | 8 |
| MURR | 395m | 3 | 16 |
| MURR | 455m | 2 | 11 |
| RICH | 320m | 2 | 16 |
| RICH | 401m | 3 | 23 |
| SAN | 515m | 3 | 21 |
| SAN | 715m | 1 | 6 |
| WAR | 450m | 1 | 8 |
| WPK | 520m | 5 | 39 |
| WRGL | 400m | 3 | 21 |
| WRGL | 460m | 1 | 7 |

The fixed input graph is **575 files / 19,094,748 bytes**, including membership, bundle manifests and admissions. There are no missing required references. This inventory verifies bytes and receipt timing; it is not a new full native forecast replay and makes no claim to repair an original quarantine.

## What the source definitions establish

The retained expert-form sorting controls label `TIME` as `finish_time`, `WIN` as `race_finish_time`, `BON` as `best_of_night_time` and `1 SEC` as `first_sectional_time`. The existing [four-surface source audit](thedogs_four_surface_data_audit_20261004.md) records those first-party HTML controls and exact response receipts; [the qualified adapter](../../race_collection/retained_speed_history.py) encodes them at lines18–19. They establish source labels, **not** common physical clocks, split distance/call, track-layout era, publication time or missing-value semantics across tracks.

`PIR` appears in the source's in-running-places cell; it is not an established early sectional or passing-order proxy. The same retained source audit documents differing raw track namespaces between normal and expert history, repeated last-win presentation, and unobserved pagination extent. Keep those qualifications. [The source-contract feasibility note](speed_source_feasibility_20261004.md) also separates legacy parser column names from actually acquired source fields.

The [retained primary-documentation follow-up](/home/l4nd0/greyhound-recovery-20261003/persistent-operation/speed-primary-documentation-followup-20261004.json) identifies GRV/Isolynx documentation as an alternative source, including its own timing distinctions and acquisition conditions. It does not supply a versioned dictionary for these TheDogs exports. No website was accessed for this proposal. Do not transfer another provider's timing semantics into this dataset.

## Smallest next implementation

Keep the current four-surface `audit_manifest()` unchanged: it requires one parent/runner relationship and parses runner-detail HTML, so feeding it 82 parent pages or CSVs would misrepresent its contract. The separate default-OFF **retained-card coverage scanner** implemented in `race_collection/retained_card_timing_coverage.py` takes this exact manifest and SHA, reusing its bounded reference/hash/receipt checks where applicable.

For every member, first authenticate membership → bundle/sidecar/CSV hashes and page receipt → target race/time. Preserve its original runner-set hash and target-runner count. Parse only the required historical identity/date/track/distance and timing-presence columns; leave placement, odds, margins and result fields uninterpreted and never emit them. Match target runners strictly through original metadata. Missing/ambiguous matches become explicit per-member or per-runner exclusions, never a name-only forced match or invented first-start history.

Report card and target-runner denominators, historical row counts, blank/placeholder/non-numeric/present counts for each named timing column, and whole-field coverage by raw venue/distance namespace. Report exact duplicate rows separately. Do not count date/track/distance guesses as verified distinct events or use current entry IDs as lifetime profile IDs. Cross-card event deduplication requires retained native profile/event bindings; otherwise report `DISTINCT_EVENT_IDENTITY_UNQUALIFIED`. Conflicting rows remain conflicts, not averaged observations. Parent summaries and historical observations remain separate surfaces; do not merge raw track labels or infer unobserved pagination.

Outputs contain counts, missingness categories, opaque identity hashes, source refs and failure enums only. No timing cell values, rankings, forecasts, private outcomes or aggregate performance. Every one of 82 members receives a disposition; eight result quarantines remain in this input-only denominator. Stop on manifest mutation, unsafe paths, shared receipt/hash failures or resource exhaustion and do not publish a successful partial index. A missing individual surface can be recorded only when the fixed manifest explicitly marks it missing.

## Finite local scope and acceptance

Two full bounded passes over the measured graph permit **1,150 file reads / 38,189,496 read bytes**, 8MiB per file, 300 seconds wall time and 4MiB output. The byte/read headroom is exactly twice measured input use; the wall bound is a conservative parsing watchdog, not a measured throughput claim. All inputs are pinned; no directory discovery or new member selection is allowed during execution. Use network denial and read-only production/source mounts with an isolated output directory. Source requests, result requests, model fits and evaluations are all **zero**.

Fabricated tests should cover blank/placeholder versus numeric presence, exact runner matches, ambiguous identities, missing columns, duplicate/conflicting observations, receipt after target jump, tampered bytes, all82 dispositions, finite exhaustion and output redaction. Then root can run the reviewed scanner once over this manifest. The resulting coverage counts can establish whether further source-definition work is worthwhile; they cannot establish predictive value or qualified early-speed measurements. No new collection or access permission is needed for this retained-input-only implementation within the existing research task.

## Review-ready implementation

The scanner reuses `parse_card_target_roster_bytes`, `canonical_roster` and `parse_form_blocks_bytes` from the existing native form packet builder, after projecting away uninterpreted columns. No original parser, model or feature file changes. It verifies all fixed file hashes, the membership/bundle/sidecar/receipt chains and pre-jump publication. Actual historical rows have **not** been scanned by the author; only fabricated fixtures using the exact retained17-column header have been parsed.

The focused28-case suite passed under kernel network denial with production and source read-only. It covers whole82 accounting including eight result quarantines, strict roster/schema failures, invalid/same-day/future dates, exact duplicates, ambiguous event keys, raw namespace mismatch, finite read/time limits, tamper/symlinks, default-OFF CLI, exclusive output and no source values in output. A test receipt outside source records the final pin.

Root execution after independent review uses the pinned Python under `bwrap --unshare-net --ro-bind / /`, with only a new empty output parent writable, from this exact checkout:

```sh
python -B -m scripts.audit_retained_card_timing_coverage --execute \
  --manifest /home/l4nd0/greyhound-recovery-20261003/persistent-operation/research-continuation-20261005/speed-breadth-proposal/manifest.PROPOSED.json \
  --manifest-sha256 922456735d4117092fbec50333d68bbb8743008fc2c73b1fc8efa48529148223 \
  --output /ABSOLUTE/NEW/ISOLATED/coverage-run01
```

The CLI prints completion, card/runner counts, output hash and `UNQUALIFIED` semantics only. Shared input failures produce `FAILED.json` without a successful `coverage.json`; retained originals remain untouched. Literal same-track/same-distance counts are an explicit conservative raw-key comparison, not verified same-layout or same-clock qualification.
