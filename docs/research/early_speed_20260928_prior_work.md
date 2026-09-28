# Early-speed neighbour hypothesis: prior work and semantics audit

Source/code-only audit, 28 September 2026. Base commit:
`956d8289b8be062e9fe5ccdf554b9a9462391c20`. This note reports existing
implementations and documentation; it neither reads raw target results nor
qualifies a new experimental population. The parent audit owns reservation,
source-availability and population verification.

## Verdict on novelty

**The broad hypothesis is already tested.** PR #193 included field-relative
historical sectional rank/gap, immediately adjacent faster-section runners,
inner-box × sectional rank, and favourite × adjacent pressure. An older local
audit additionally implemented comparable-neighbour pressure, lone leaders,
pressured favourites, and inside/outside pace imbalance. Renaming these
features would not create a new mechanism.

A defensibly different question is whether **directionally supported historical
running style changes the effect of a neighbour with comparable early speed**.
For example, inward movement from the outer neighbour is different evidence
from merely having a small numerical box difference. No verified source
definition or historical availability contract for such style was found in
the source files inspected here. This is an acquisition candidate, not a
currently established feature.

Vacancy-aware nearest-active neighbours, retaining the physical box gap, are
also a construction not found in these implementations. That distinction alone
does not establish a physical interference mechanism or validate comparing
sectional measurements. A box-two gap must not silently acquire the same
meaning as an occupied immediately adjacent box.

## Exact prior constructions

All relative paths below are in the pinned research worktree unless an absolute
path is provided. Line references refer to the inspected source versions.

| Level | Existing construction | Source |
|---|---|---|
| Individual historical form | Sixteen features: start/layoff counts, recent finish/win/place/margin, career finish/win/place, and same-venue/distance/grade starts and wins | `scripts/offline_prediction_research.py:17` |
| Individual elapsed times/sectionals | Positive `TIME` and `1 SEC` values; mean of three most recent available and best of five; same target venue and distance; QOT/RICH/MURR excluded as ambiguous layouts | `scripts/offline_systematic_features.py:28–29,55–85` |
| Relative current field | Section mean minus available-field median, divided by that median; fractional rank of available runners; best-section gap | `scripts/offline_systematic_features.py:96–102,156–157` |
| Adjacent runners | Present measured runners at absolute box difference exactly one; count with strictly smaller section mean; measured-neighbour count; own-minus-neighbour mean scaled by own section; faster-field fraction | `scripts/offline_systematic_features.py:160–165` |
| Box and market interactions | Inner box × section rank; maximum-market-probability indicator × count of faster adjacent measured runners | `scripts/offline_systematic_features.py:172–173` |
| Vulnerable-favourite rules | Unique favourite, model-minus-market below fixed thresholds, optionally at least one faster measured neighbour; qualification uses earlier OOF selections, dates and binary Brier | `scripts/offline_systematic_search.py:303–335` |
| Combined recipe | Base16 + sectionals + pace-pressure group; single group additions also tested | `scripts/offline_systematic_search.py:31–41` |

“Three most recent” means three most recent **available qualified positive
values**, not necessarily the last three starts: values are filtered before
slicing (`offline_systematic_features.py:60,68–84`). Missing sections do not
become slow times in this feature builder. The model uses training medians plus
missingness indicators, with training-only centring/scaling
(`offline_systematic_search.py:107–123`). Neighbour comparisons can nevertheless
use partial measured fields: a missing neighbour is excluded from the measured
list. Thus “no faster measured neighbour” is not verified absence of pressure.
The favourite-rule helper treats missing pressure as not selected
(`offline_systematic_search.py:313`), not as evidence of a clear path.

The field roster is checked against card and sidecar, without inferred scratch
or reserve removal (`scripts/offline_form_packet.py:136–144`). This establishes
the matched retained roster; it does not itself establish scratch events between
capture and the decision or between the decision and actual start.

The existing report records positive inner-validation log-loss deltas for
`add_sectionals` (+.008106, +.001876, +.001559), `add_pace_pressure`
(+.007644, +.000433, +.000279), `add_mechanism_interactions`
(+.006346, +.007658, +.007837) and `section_pressure`
(+.010830, +.002562, +.002225), relative to base16. These are prior overlapping
inner screens, not newly computed results or three independent replications
(`docs/research/offline_systematic_20260924_results.md:64–92`). They failed to
improve that screen; they do not disprove physical early-speed interaction.

## Older topology audit: additional duplication and an access boundary

Read-only source:
`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector-ci-routing-fix/scripts/audit_pace_topology_mechanism.py`.

Its complete-field gate requires unique boxes 1–8 and every runner to have a
pace estimate (`290–296`). It compares present box±1 starters, defines a
comparable-neighbour gap at a frozen q25 and high positive pressure at a q75,
then defines lone leader, pressured favourite, clear-path nonfavourite, and
inside/outside imbalance (`299–331,402–408`). Nearest occupied boxes beyond a
vacancy are not neighbours in that implementation. Box presence does not
demonstrate interference.

The audit's declared pace state uses exact track-distance expanding prior
moments, race-relative scores, prior opposition, 180-day decay and simultaneous
post-race updates (`390`). Its thresholds were outcome-blind distributions of
the complete frozen topology (`399`), rather than necessarily an earlier
training-only population for each later fold. It is not a drop-in implementation
of the present request's training-only preprocessing rule.

**Do not execute or inherit this older data path under present authority.**
It declares a June 10–July 18 population (`35–36,381`), overlapping currently
reserved dates. Its latent and native-history loaders refer to broader
outcome-bearing inputs. Only the Python source was used here; their arrays,
raw files and result reports were not opened. The older program's existence
establishes prior construction, not permission to access its data or a fresh
validated result.

## What source semantics are actually supported

| Candidate | What local code/documentation proves | What remains unverified |
|---|---|---|
| CSV `1 SEC` | Parser maps the literal header to `first_sectional`; numeric positivity is used by research | Source-owned units and identified physical timing point, timing-system changes, comparability across historical conditions |
| FastTrack `split1` | Adapter stores the supplied value as `split_1_time`; schema prose calls it individual first sectional | Neither adapter nor schema identifies metres/call location, provider definition version, or release time |
| `Best 1st Split` | Expert-form parser extracts the number following that literal label | No date of the split is captured by that extraction; career-best update timing and provenance cannot be inferred |
| Single-digit `PIR` | Prior coverage audit reports all 6,920 equal historical finishing position | Independent early position; interpreting the digit as first call would manufacture information |
| Multi-digit `PIR` | Prior audit documents multi-digit sequences, often using 1–8 | Call sequence, timing points, missing-call encoding, and consistent track conventions |
| `running_style` | A nullable database column exists | Populated observations, measurement definition, source/observer, historical timestamp and pre-target availability |
| Comment text | FastTrack adapter has a mapping for `comment` | A retained, eligible, earlier-race comment corpus and defensible directional-style coding |

Evidence: `csv_ingestion.py:182–186`;
`src/collectors/adapters/fasttrack_adapter.py:80–87`;
`docs/schema_diff_fasttrack.md:97–102`;
`utils/expert_form_metadata.py:107–114`; `models.py:279`;
`docs/research/offline_20260924_model_provenance.md:143–154`.
The synthetic test `tests/test_staging_writer_sectional_keys.py:4–15` verifies
header plumbing only. It cannot verify a measurement's physical definition.

Local PIR descriptions conflict: `docs/fasttrack_field_map.md:39` calls it
“Performance Index Rating”, while line 60 says “Points in running”; the schema
uses “Position In Running”. None identifies a first call. These are local
descriptions, not a verified provider definition.

The older native-history parser likewise does not solve this: in
`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-fast-nonfavourite-native-enrichment-20260818/scripts/recover_fast_nonfavourite_history_coverage.py:378–435`,
it requires sixteen cells and reads cell 10 into `first_sectional_seconds`.
That establishes the implementation mapping, not a physical call point or
cross-track comparability. Its CSV path reads `1 SEC` and writes the same named
field (`581–600`). Native race/dog identity and timestamps are useful provenance,
but a variable named `seconds` does not establish units by itself.

The prior 332-race coverage counts are superseded as a qualification denominator
by the corrected 331-race population (`offline_systematic_20260924_results.md:5–7`).
This note deliberately does not recycle the older 31 complete-field / seven
three-start-field counts as counts for the newly audited population. The parent
audit must supply current denominators after eligibility and layout exclusions.

## Minimum evidence before another interaction experiment

First inspect eligible retained pre-race HTML/CSV and independently retained
source documentation, with identity filtering before outcome-bearing decode.
The minimum gap is not another model family: it is a source-owned definition
of units and first-call location for each track/layout/distance/era, and either
verified directional style or a explicitly limited spatial-neighbour mechanism.

Retain measurement definition/version and effective dates; native runner and
historical race identity; track/layout/distance; exact history date/time; raw
field/header/value and missing reason; source capture time and immutable raw
hash; source publication time if available; and target decision-time roster
with effective boxes, reserves, vacancy and timestamped scratches. Distinguish
publication time from retrieval time. A later historical backfill cannot be
labelled demonstrably available at an earlier decision without separate evidence.

For directional style, require earlier-race observations at a specified segment,
explicit inward/outward/straight/unknown encoding, observer/source and extraction
version, and repeatability evidence. Do not derive style from the target result
or from a career prose label with no timestamp. Unknown remains unknown.

A default-off retention extension is justified only after eligible retained
material proves that these fields occur in an already authorized response.
Then retain the additional fields with receipts in that existing path, under
separate future population allocation; no parallel collector is needed. The
incremental request burden could be zero for fields already in captured pages,
but this audit has not measured payload sizes, current retained field coverage
or annotation effort. If definitions or video/style observations are absent,
estimate their acquisition/annotation burden in a separate authorized plan;
do not invent a numeric traffic estimate or contact providers here.

## Inspected identities and audit limitation

SHA-256:

| Source | Digest |
|---|---|
| `scripts/offline_systematic_features.py` | `b8cf9c8455c1f1ef1c61169520d8839176d906091cd3937c633c41f1c1fcd4b8` |
| `scripts/offline_systematic_search.py` | `d95828aec1bbc1c22bac4d4b7be5429bc92560f0c59a854ce328b2cc1040cec1` |
| `docs/research/offline_systematic_20260924_results.md` | `43077c5e0d83348df77a2a1813843111ce8354d47d7dcea4774ce46ab82332d9` |
| Older `scripts/audit_pace_topology_mechanism.py` at the absolute path above | `2b84220609d1b71acbd2c73f07cc3cf628c4d6f4900371b74c3ba2983a27531b` |

One initial text search over `docs/research` lacked the intended Markdown glob
and returned snippets from the already published prior experiment ledger:
aggregate fit/validation records and feature names. The scope was stopped and
disclosed immediately to the parent; later searches used explicit source/doc
paths or `.py`/`.md` filters. No raw runner outcome records, reserved data files,
databases or feature arrays were opened by this subtask. No provider requests,
model fits, services or shared-file edits were performed.
