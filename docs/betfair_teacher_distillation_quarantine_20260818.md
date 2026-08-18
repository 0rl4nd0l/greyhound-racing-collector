# Betfair-teacher orphan quarantine — 2026-08-18

<!-- markdownlint-disable MD013 -->

Verdict: **`QUARANTINED`**.

The orphaned `BETFAIR_TEACHER_SIGNAL_PROMISING` bundle is not canonical
research evidence and is not `CANDIDATE_READY_FOR_FORWARD_FREEZE`. Its sealed
calculations are internally reproducible, but the preserved execution history
does not satisfy the experiment's outcome-inaccessibility and pre-result freeze
requirements. Do not rerun, refit, retune, promote, deploy, score forward, or
use the bundle to weaken PR #142.

## Audited repository and evidence identities

The adoption audit fetched and verified `origin/master` before review:

- commit: `7d5931bbbc6d108a1f4dc44f960d236bc66720ba`
- tree: `1c644d84098220b4b97aa9a7a860174440fe9f74`

The orphan remains preserved byte-for-byte in the detached dirty evidence
workspace at HEAD/tree
`779761165637b709227d965f6c9be7e80706d23f` /
`a601f9c1a941c15dfeec4e300f7adbade5440bc2`. This quarantine record does not
copy, rewrite, or canonicalize its code or artifacts.

Exact orphan inputs and artifacts reviewed:

| Item | SHA-256 |
| --- | --- |
| `canonical_betfair_win_rows.jsonl` | `95ba227e61bf21f5e5896d78243b52b784cb5b1b498506deea9e6b60e471866a` |
| `sportsbet_betfair_joined_surface.jsonl` | `86fabb05556160e555f076322eb8786b6166e369a6a8ec57d475c0e4a06e67f7` |
| `source_manifest.json` | `14789e1767df6d3484a98b0242a53267546ece3a14fafe1d95cdc7ae1dfccdf4` |
| `protocol.json` | `b84122b005498d5cf9efd27a5078d38bed8eb1eace08edb7183b538befd1a12a` |
| `population.jsonl` | `7ad45e9ff7fac81f74b3ef09a9cbac6107a1f4545b11b9a030dd85eeb1ebf7af` |
| `oof_predictions.jsonl` | `219a7cf11205abb12b49ed45b18212a84ac824d002031d4123936b8c66c76982` |
| `exclusions.jsonl` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `report.json` | `1b57275a928b677d278e90945656369be3258a5ffb7d01c13bc6fcdd37937703` |
| `REPORT.md` | `0d322881f268d4551687c025c83ef2af0e695ca5faa74cfa5bcfe82ba9ddd9e5` |
| task card | `c19d7d50cecd1cac5e35107962543c052f802085668e5f29510439f40eeebdaa` |
| experiment script | `9b7ebc91237b3d6d42cd3c4815ab3aaea643aeefc38fed0af2e52b887c580165` |
| focused tests | `fa01b129d7576b70839992dbc4af1fb9fef5a6b603e8f49578a290eca3fb0f6b` |
| experiment narrative | `909300d21b71254a127816a440f6071f6bff3d62f14404d1727ecd3e1a88ccd2` |

Both artifact-local `SHA256SUMS` manifests passed. The official June and July
Betfair raw CSV hashes (`304085cec9dd7930c505f9b45d33835bdf1d2223dbb6f5c0723087c813114748`
and `f150a95b7ebd323d7626bb3653cae04a9a3165d04ebbf8c4611c22d2a647944f`)
and all three receipt hashes matched their source manifest. The corrected
Sportsbet canonical WIN matrix matched
`eb1783d4cc07e6980463a097c97fdac9f5370b08f493ca15addf768aa0b014b6`.

## What independently verified

The sealed population contains 1,008 races and 7,142 runners from 2026-06-10
through 2026-07-18. All 7,142 rows matched the strict joined surface exactly on
race/box identity, native Betfair market and selection IDs, scheduled timestamp,
provider actual-off provenance string, timing eligibility, Sportsbet source row
and matrix hash, and Betfair source rows and hashes. No 2026-08-18-or-later or
October row appears. Probability sums and race-centred teacher residuals agreed
within `4.45e-16`.

The 692-race / 4,864-runner OOF surface contains one prediction per race/box.
Its three validation folds contain 176, 211, and 305 races; train dates strictly
precede validation dates, validation races are unique, and train/validation race
overlap is zero. Independent population-only recomputation matched every
reported train-fold numeric mean, scale, categorical vocabulary count, and
unseen-validation count exactly. This verifies the recorded train-only
preprocessing metadata without refitting.

Independent metric recomputation from the sealed OOF probabilities produced:

| Gate | Sportsbet | Pseudo-Betfair | Delta | Meeting-date bootstrap 95% CI | Improved folds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Teacher soft-target cross-entropy | 1.453183262841 | 1.442520483301 | -0.010662779540 | [-0.012485090410, -0.008817044969] | 3/3 |
| Winner multiclass log loss | 1.422993772583 | 1.406826200498 | -0.016167572085 | [-0.026691063849, -0.005933558127] | 3/3 |

The fixed 5,000-draw, seed-20260818 meeting-date bootstrap reproduced both
reported intervals exactly. If considered only as arithmetic on the final OOF
file, both declared numerical decision rules pass.

## Why adoption fails

The task card and protocol required winner/result fields to remain inaccessible
until the frozen outcome-independent teacher gate passed. The preserved Codex
execution log at
`/home/l4nd0/.codex/sessions/2026/08/18/rollout-2026-08-18T21-34-33-01a014a6-d767-7612-aea4-b3ce5e34f043.jsonl`
proves a different sequence:

1. Ordinal 117 created the first experiment implementation. Its
   `build_population` loaded every `WIN_RESULT` into `winner_tokens` before the
   teacher fit or gate.
2. Ordinal 129 ran the fit and both evaluations and disclosed
   `BETFAIR_TEACHER_SIGNAL_PROMISING`.
3. Only after that disclosure, ordinal 151 changed the loader and winner-label
   flow to approximate the protocol's conditional-access requirement; ordinal
   157 ran the experiment again.
4. After the results were already known, ordinal 192 changed the materialized
   population provenance fields; ordinals 200 and 204 ran the experiment again.

The final loader is also not an access boundary: it opens the joined JSONL that
contains `win_result` and uses `json.loads(..., object_pairs_hook=...)` to discard
the field after JSON values have been decoded. This proves the label was omitted
from the returned feature dictionaries, but it does not prove the label was
inaccessible before the gate.

Betfair was also not target-only in the implemented inference inputs. The
declared categorical feature `distance` is populated by `_distance_lookup` from
`distance_raw` in `canonical_betfair_win_rows.jsonl`. No non-Betfair distance
source or forward mapping is frozen. Although distance itself is ordinary race
context rather than a price or result, this implementation depends on a
Betfair-derived inference feature and therefore fails the explicit target-only
contract.

The final feature list contains no Betfair probability, price, BSP, result, or
actual-off value, and code inspection found no winner label entering model fit
or inference. The recorded OOF metrics therefore remain an internally coherent
historical calculation. That does not cure the stronger predeclared requirement:
the first result was observed before the implementation satisfied the claimed
label gate, and final sealed bytes were produced only after post-result code
changes and repeated executions. Freeze order and outcome inaccessibility are
therefore unprovable and must fail closed.

## Findings

### BLOCKING

- Winner labels were loaded before the first teacher gate, contrary to the
  frozen protocol.
- The implementation and provenance materialization were changed after both
  teacher and winner results were known, then the experiment was rerun. The
  sealed bundle is not the result of one pre-result-frozen execution.
- The final parser filters `WIN_RESULT` after decoding a label-bearing source;
  it cannot establish the required inaccessible-until-gate boundary.
- The `distance` inference feature is loaded from the Betfair canonical rows,
  so Betfair is not teacher-target-only in the implemented scoring path.
- No persisted final model or coefficient bundle exists. The rolling OOF fits
  record only coefficient norms and preprocessing summaries, not coefficient
  vectors or categorical vocabularies. No full-development final-fit rule or
  selected fold model exists. A forward candidate would require prohibited new
  specification and refitting, independently blocking forward-ready status.

### IMPORTANT

- Missing retired-V2 claim, closeout, decision, and release records are control
  deficiencies, but they are not the quarantine reason by themselves. The
  substantive blockers above remain even if that bookkeeping is ignored.
- Betfair probability, price, and BSP were teacher-target or OOF-evaluation
  columns, not inference features; winner labels do not appear in the population
  or OOF files and code inspection found no label use in fitting. The separate
  Betfair-derived `distance` blocker above still applies.
- Source hashes, strict identity, timing provenance, development cutoff,
  normalization, fold uniqueness, train-only preprocessing metadata, reported
  metrics, confidence intervals, and numerical gate outcomes all verified.
- The strict source join used winner outcomes as post-join corroboration and
  would have excluded a result conflict. The realized population had zero such
  conflicts, so this did not change its rows, but source-population construction
  itself was not label-inaccessible.
- The teacher gate implementation checks improved-fold count and bootstrap
  upper bound but omits the protocol's explicit combined primary-delta check.
  Independent recomputation found the combined delta negative, so the exact
  predeclared rule still passes numerically despite this contract gap.
- The orphan `SHA256SUMS` binds only its six generated evidence files, not the
  task card, script, tests, or narrative. Their current hashes match session
  ordinal 212 and are listed above, but the orphan bundle did not durably bind
  those control files itself.

### OPTIONAL

- A materially new experiment could use a separately materialized, hash-bound
  teacher-only source that physically excludes all result fields, with complete
  code/protocol hashes frozen before its first and only fit. This quarantine
  grants no authority to create or run it.

## Claim boundary

Supported: these exact sealed bytes encode an internally consistent historical
calculation on strict June/July identities, and the reported arithmetic can be
recomputed from the sealed OOF rows and labels.

Unsupported: that the calculation was a genuinely outcome-gated,
pre-result-frozen experiment; canonical research evidence; a persisted or exact
pseudo-Betfair inference candidate; `CANDIDATE_READY_FOR_FORWARD_FREEZE`;
prospective accuracy; exact live Betfair semantics; economics, EV, ROI, betting
value, deployment, promotion, activation, or any weakening of PR #142.

Terminal handling: preserve the orphaned bytes under their hashes, retain this
record as the canonical disposition, and do not adopt or rerun the bundle.
