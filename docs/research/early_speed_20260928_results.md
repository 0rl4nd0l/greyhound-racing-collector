# Early speed and neighbouring runners — feasibility result

**Verdict: partially testable.** The retained material supports an audit of
historical numeric sectionals and occupied-box geometry. It does **not** support
the proposed predictive test: **0 / 331 races are fully qualified** for verified
early-speed interactions. Even if every positive `1 SEC` value were accepted
without resolving its meaning, only **7 / 331 fields** have three comparable
observations for every runner; just **5 / 177** lie in the old evaluation periods.
No model was fitted, no favourite was selected, and no new performance result
or betting return was calculated. The hypothesis is not disproven.

**Recommendation: collect specified measurement evidence before investing in
another experiment.** Retain the narrower occupancy hypothesis as exploratory;
discard a rerun of the old sectional-pressure recipes. Do not change production
or the separately owned frozen four-model study.

## What is new, and what is duplicated

PR #193 at `956d8289b8be062e9fe5ccdf554b9a9462391c20` already tested:

| Evidence level | Previously tested |
|---|---|
| Individual form | Base16 finish/margin/win/place/experience and context features; recent form trends |
| Individual time/sectional | Mean of latest three available positive `TIME` / `1 SEC`; best of five; same venue/distance with ambiguous layouts excluded |
| Relative field | Sectional rank, median-relative advantage, best-sectional gap, faster-field fraction |
| Adjacent runners | Faster measured box±1 neighbours, measured-neighbour count, neighbour sectional gap |
| Box/market interaction | Inner box×sectional rank, favourite×adjacent pressure; a combined section-pressure recipe |
| Running style/interference | No independently verified directional style or observed interference |

The old sectional, pace-pressure and mechanism screens all worsened inner
validation log loss relative to base16 in all three overlapping screens. These
are **prior results**, not a new experiment or three independent replications.
An older topology script additionally defines lone leaders, comparable adjacent
runners and pressured favourites. Its loader extends through July 18 and was
**not executed or used as a data source**, because that intersects reservations.
See [the exact source audit](early_speed_20260928_prior_work.md).

The prespecified new construction uses the **nearest occupied box on either
side, retaining the number of empty boxes between**, and its interaction with
field-relative early advantage. Prior implementations use exact box±1 only.
This is a small, distinct occupancy association, not proof of interference.
Directional inward/outward running style would be a different, potentially
stronger hypothesis, but lacks supported retained observations.

## Access, population and audit sample

Live GitHub metadata verified #193's head above and #194's changed head
`036dc57fca8b54e9e05c4916c425639f588af7e9` (both open at inspection). The older
September 24 #194 head was not assumed current. This research branch is based
on #193 so its data readers and prior evidence remain reproducible; it neither
modifies nor merges into the operational or prospective-study branches.

Before cards, the audit verified the original closed114, form-only and August
reservation inputs, their **1,265-key union**, the current September 28 reservation
review and the **58-race / 393-label incident**. All incident identities remain
denied. The reservation review's future activation wording is historical
metadata here, not a claim about current operational status; **all targets after
July 9 remain outside this task regardless of later allocation changes**.

The exact development file hash is
`c58bd59bc0d52666d31812dccd60981de7f2f03b64862f98ddf9f88ef60cec46`.
Only identity scalars were decoded from it; target outcome fields were never
decoded. The corrected market matrix and sidecar were byte-hashed without row
decoding. Card reads were restricted to admitted identities; every card and
sidecar hash and complete roster was checked before use. Historical dates must
precede both source capture and target date, and the earliest protected date.
No prospective results, live DB, provider or private-result store was accessed.

The fixed sample was written **before opening card values**: one smallest salted
SHA256 race per literal target venue, **39 races / 1,344 repeated historical
observations**. Selection used race identity only. Full-population coverage used
**331 races / 2,360 runner-targets / 27 dates**, June 10–July 8. All 331 source
checks passed. The rejected TAREE June 13 race from the earlier 332-race foundation
was not reintroduced. The old evaluation denominator remains 177 races on 15
already inspected dates; none becomes fresh confirmation through this audit.

## Measurement findings

| Field/source | Observed meaning and availability | Limitation and disposition |
|---|---|---|
| Raw-card `1 SEC` | 7,838 positive numbers and 3,617 blanks in 11,455 repeated prior-start observations. No malformed/nonfinite/nonpositive values observed. Present in hash-bound pre-target cards. | Heading and parser mapping imply a sectional, but no retained source-owned units/call distance/start point, runner-versus-leader definition or era validity. Numeric presence is an upper bound, not qualified early speed. |
| Single-digit `PIR` | All **6,919 / 6,919** equal that historical row's `PLC`. | Not validated first-call position; excluded from early-speed construction. Equality is evidence against this use, not a universal claim about every provider's PIR. |
| Multi-digit `PIR` | 4,532 sequences comprised of 1–8; four other numeric codes. | Call locations/order and special codes unverified. No first character was relabelled as early position. |
| `TIME` | 11,455 positive historical elapsed-time entries. | Full-race time is not early speed; individual/winner and condition semantics need verification. No substitution. |
| Historical identity | Dog token inherited within its CSV block; historical date, track, distance, box and grade. Dates range 2022-11-20–2026-07-06, strictly earlier than target/capture. | No native historical race/runner ID in the 17-column export. Date/track/distance/box is corroboration, not a proven unique event identifier. No new cross-source join was attempted. |
| Current box occupancy | Complete pre-race card and sidecar roster matches the admitted market population. 169 fields have an unoccupied box; 163 have an internal gap; 482 runners have a nearest occupied neighbour more than one box away. | Empty means absent from that retained roster. No independent timestamped scratch/reserve ledger proves every subsequent decision-time change; absence does not identify the cause or prove interference. |
| Running style / comments | No style/comment column in any audited card header. Schema-only nullable style fields and adapter mappings exist elsewhere. | No authorized, timestamped directional-style observations established; unavailable. |
| Native HTML / FastTrack / best split | Prior source code maps native cell 10, FastTrack `split1`, and a best-split label. | Names/mappings do not establish physical call definitions; a career best lacks its historical timestamp. Broader retained native-history paths may contain reserved races and were left unopened. |

All 39 sampled **original raw exports** still exist and match their sidecar hashes.
Their headers are the same 17 fields as the normalized cards, with no additional
call dictionary, event IDs or style. Keeping another copy of these CSVs would
not fix the gap. This is a scoped retained-material audit, not a claim that no
useful evidence exists anywhere in every archive.

The sample's sectional numbers range from 2.15 to 16.59. For example, one
presampled runner has 300m historical entries of 2.56–2.69 at CASO and 6.68–6.79
at QST. These are raw numbers, **not verified seconds**. Equal race distance
does not establish equal call distance. New comparisons exclude QOT, RICH and
MURR alias families, which merge layouts. Even other matching venue/distance
labels do not prove timing-system/track-condition equivalence. No global or
future-derived adjustment was fitted. Missing entries remain unknown.

## Coverage and the prespecified stop

| Numeric availability only | Runner-targets / 2,360 | Complete fields / 331 |
|---|---:|---:|
| At least one positive sectional anywhere | 2,143 | 246 |
| At least three positive sectionals anywhere | 1,663 | 118 |
| At least one, same nonambiguous layout/distance | 738 | 24 |
| At least three, same nonambiguous layout/distance | 290 | 7 |
| Fully verified early-call measurement and identity | **0** | **0** |

The seven-field numeric upper bound contains 46 runners over four dates:
five AP_K races, one ROCK and one MAND. It is highly selected relative to the
331-race development population. Even the one-observation relaxation would
provide only 24 fields / 159 runners on 11 dates across eight canonical venues;
it was audited as coverage, **not tried as a model variant**.

| Fixed period | Three-observation numeric fields | Runners | Dates |
|---|---:|---:|---:|
| Initial training, June 10–23 | 2 | 14 | 2 |
| Evaluation, June 24–30 | 1 | 8 | 1 |
| Evaluation, July 1–2 | 0 | 0 | 0 |
| Evaluation, July 3–9 | 4 | 24 | 1 |

The [frozen protocol](early_speed_20260928_protocol.md) required supported
semantics, three comparable starts per runner, full-field coverage, at least
50 initial training races on five dates and 30 test races on five dates, with
representation in every period. Both semantics and population gates fail.
Therefore market/base16/base16-plus-interaction paired scores, date intervals
and vulnerable-favourite expected/actual wins are **not estimated**. Reporting
zero selected favourites would incorrectly imply the selection rule was run.

One hypothesis, zero supporting variants, **zero fits and zero predictive trials**.
This is a measurement/coverage failure, not a negative performance finding.
The complete ledger preserves two preparation failures: a checkout-local
incident path absent because the original file was retained outside tracked
Git, then rejection of repeated identical race IDs in the August provenance.
Both stopped before development cards were opened. Repairs pinned the existing
incident path and required all repeated identity scalars to agree; no protected
outcomes were decoded. The successful first audit and extended raw-header audit
are both retained. A helper's source search also accidentally returned prior
published aggregate experiment-ledger snippets; it stopped that search scope,
and the ledger discloses it. No new association was calculated.

## Minimum acquisition specification

Start with **source definitions and identity**, not more copies of the same
CSV. The likely owners are the source/export provider and the jurisdiction's
official timing records (TheDogs/official state results or FastTrack where
already authorized). These are source candidates, not a verified endpoint or
permission to contact them. Required evidence:

1. A versioned source definition for each track/layout/distance/era: units,
   clock origin, physical first-call location, individual versus leader timing,
   precision, missing/special codes and changes to timing equipment or layout.
2. Native dog and historical race identity, layout/distance, start time and
   timezone; original field/header/value; source publication time if available;
   immutable capture time and raw bytes/hash. Keep publication, occurrence and
   retrieval times distinct. A late backfill cannot prove availability at an
   old target decision.
3. Pre-decision active roster with effective boxes, vacancy/scratch/reserve
   status and event timestamps. Freeze the actual input roster used at decision;
   never reconstruct it from target finishers.
4. For a later style hypothesis only: independently observed inward/outward/
   straight/unknown movement at a specified segment in strictly earlier races,
   observer/source, repeatability check, and timestamp/version. Unknown stays
   unknown. Target result or untimed lifetime prose cannot supply style.

Proposed retention format: append-only JSONL observation records with
`source`, `native_race_id`, `native_runner_id`, `occurred_at`, `published_at`
(nullable with reason), `captured_at`, `track_layout_id`, `distance_m`,
`definition_id`, `call_point_m`, `units`, `raw_value`, `parsed_value`,
`missing_reason`, `raw_sha256`, and `parser_version`; a separate immutable
target decision/roster manifest references the observation hashes. Definitions
and as-of eligibility receipts are retained alongside raw inputs. Style, if
later supported, is a separate explicitly sourced observation. Historical
target outcomes remain in a separate access-controlled result store.

**Default-off proposal, not implemented:** extend the existing retained-input
bundle only if its owner first demonstrates that an already authorized captured
response contains missing identity/definition/roster fields. The sampled CSVs
do not. No parallel collector, study alteration or extra traffic is proposed
as an immediate action. A retention flag alone cannot manufacture semantics.

Burden estimate: a 100-field, separately allocated feasibility pilot with up to
eight runners and three prior starts needs at most **2,400 historical
runner-start slots before deduplication**, plus 100 roster receipts and one
definition per distinct timing context. Existing normalized cards total
1,048,624 bytes for 331 races (median 3,290 bytes); the same card volume for
100 fields is roughly 0.32 MB, **excluding** sidecars, HTML/video and new evidence.
Incremental provider requests can be zero only if those fields already occur
in authorized retained responses. Otherwise request/annotation burden is not
established; inventory eligible retained official records before proposing it.
No source traffic estimate is presented as measured.

Future performance work requires a **separately allocated** population outside
all current reservations, measurement qualification first, then a new frozen
chronological protocol and fixed endpoint. The present minimum 50 training /
30 evaluation races is a feasibility floor, not confirmation power. Plan for
substantially more independent dates than 15 and size the study from a declared
minimum meaningful paired-score effect and date-level variance before outcome
access. Do not add features or races to the existing four-model study. The
historical seven-field upper bound gives no basis to promise feasible yield.

## Reproduction and evidence

Worktree: `/home/l4nd0/greyhound-early-speed-neighbours-20260928`;
branch: `research/early-speed-neighbours-20260928`. No dependencies installed.
The audit is standard-library Python, one process at nice 10; the final run
took about two seconds and 80 MB peak RSS. No production/frozen artifact writes,
provider requests, service changes or protected target-outcome access occurred.

```bash
cd /home/l4nd0/greyhound-early-speed-neighbours-20260928
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_audit_early_speed_neighbours -v
# Choose a NEW output path; existing runs are deliberately refused.
PYTHONDONTWRITEBYTECODE=1 nice -n 10 python3 -m scripts.audit_early_speed_neighbours \
  --out /home/l4nd0/greyhound-early-speed-audit-output-20260928/reproduction01
```

The code requires exact pinned local files; it never downloads replacements.
[Input identities](early_speed_20260928_evidence/input_identities.json) pin every
used card/sidecar and original restriction input. [Source identities](early_speed_20260928_evidence/source_identities.json)
pin the executed code, canonical parser and frozen protocol. The
[sample](early_speed_20260928_evidence/audit_sample.json),
[coverage](early_speed_20260928_evidence/coverage.json),
[retained raw-header evidence](early_speed_20260928_evidence/retained_raw_exports.json),
[complete trial ledger](early_speed_20260928_evidence/complete_trial_ledger.jsonl)
and [execution record](early_speed_20260928_evidence/execution.json) accompany
this report. Full per-race occupancy and fixed-sample observations remain local
under `/home/l4nd0/greyhound-early-speed-audit-output-20260928/run04`, with exact
hashes in [the artifact inventory](early_speed_20260928_evidence/retained_artifact_identities.json).
Earlier failed and successful runs remain intact; the first two source snapshots
were reconstructed by reversing the recorded patches, explicitly identified as
such. The audit does not re-run the 277-fit search.
