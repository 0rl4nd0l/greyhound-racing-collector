# Sportsbet Paired-Market Source Diagnosis

result: WORKING

## Boundary

- Services and timers remained disabled.
- No production database was opened for write or modified.
- Exact expected-runner validation and paired-market append ordering were not
  changed.
- This follow-up changes only source extraction and its regression tests.

## Retained HEA Evidence

The retained 2026-07-10 Healesville Race 4/5/6 attempts prove two symptoms from
one source-layout mismatch:

- R4 and R5 returned eight WIN rows and zero PLACE rows, although every raw
  runner row ended with two decimal prices followed by `EW`.
- R6's first render returned four cross-wired WIN rows. Its second render
  returned eight runner rows whose raw text contained the complete paired
  prices.
- Exact runner validation blocked all three attempts and `inserted_rows` stayed
  zero.

The retained source URLs and a fresh browser inspection showed the same runner
card headings: `Win Fixed` followed by `Place Fixed`. Each runner row contains
both buttons. The old code extracted the first price, then clicked a generic
control containing `Place` and ran the same first-price extractor again. On
favourite rows the legacy selector could instead return the PLACE button as
WIN, explaining values such as Dinosaur Deano `1.30` while its raw row proves
WIN `2.90`, PLACE `1.30`.

## Bounded Fix

Parse the final two decimal prices before an exact `EW` marker only when the raw
text contains one runner header and its box agrees with the extracted box. Use
that source-proven pair for WIN and PLACE. If the first render does not contain
paired rows, retain the existing second render but accept it only when it also
contains source-proven pairs. A generic Place click is never treated as market
proof.

Replaying the retained rows through the new parser yields:

| Race | First render paired WIN/PLACE | Recovery render paired WIN/PLACE |
| --- | ---: | ---: |
| HEA R4 | 8 / 8 | 0 / 0 |
| HEA R5 | 8 / 8 | 0 / 0 |
| HEA R6 | 0 / 0 | 8 / 8 |

The manual pre-jump odds-only gate remains required after the reviewed commit is
published and deployed. Full-daemon and two-cycle proof must not resume until a
complete paired-market run succeeds, and the owner's current no-DB-mutation
boundary remains in force.
