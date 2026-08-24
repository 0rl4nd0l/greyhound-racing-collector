# Historical Betfair teacher distillation — 2026-08-18

Decision: `BETFAIR_TEACHER_SIGNAL_PROMISING` on historical development OOF
evidence only.

## Repository and source boundary

- Verified `origin/master`: `d009ab12fddd75a55ab73c6f02052f0bd63195ff`.
- PR #142 blocking unsafe forward Betfair confirmation is present as
  `0f295566` and was not changed.
- Research checkout HEAD/tree:
  `779761165637b709227d965f6c9be7e80706d23f` /
  `a601f9c1a941c15dfeec4e300f7adbade5440bc2`.
- Strict historical population: 1,008 races / 7,142 runners, 2026-06-10
  through 2026-07-18. Every row retains race/box, Betfair market/selection,
  scheduled-time, source-row and source-hash provenance.
- Unique chronological OOF population: 692 races / 4,864 runners across 24
  validation meeting dates. The first chronological block is training-only.
- No 2026-08-18+ or October outcome was accessed.

## Frozen teacher and model

The teacher target is the within-race-centred residual
`log(normalized scheduled-off Betfair probability) - log(normalized corrected
Sportsbet probability)`. BSP, results, actual-off values, in-play values and
volume are not features.

One ridge specification was frozen before fitting: alpha 10, SVD solver,
train-fold-only numeric standardization and train-fold-only one-hot vocabularies.
Numeric features are Sportsbet normalized probability, log-probability, rank,
field size, raw overround, probability HHI and box. Categorical features are
venue and distance. Previously failed form/speed, fast-nonfavourite,
latent-ability and pace-topology mechanisms were not reopened. No model was
persisted.

Four contiguous meeting-date blocks produced three expanding chronological
folds:

| Fold | Train dates | Validation dates | Train races | OOF races |
| --- | --- | --- | ---: | ---: |
| 1 | 2026-06-10–2026-06-17 | 2026-06-18–2026-06-27 | 316 | 176 |
| 2 | 2026-06-10–2026-06-27 | 2026-06-28–2026-07-05 | 492 | 211 |
| 3 | 2026-06-10–2026-07-05 | 2026-07-06–2026-07-18 | 703 | 305 |

## Outcome-independent teacher gate

The pseudo-Betfair probabilities improved soft-target cross-entropy in all
three folds. Combined Sportsbet/pseudo cross-entropy was
`1.453183263 / 1.442520483`, delta `-0.010662780`. The fixed 5,000-draw,
seed-20260818 meeting-date bootstrap 95% CI was
`[-0.012485090, -0.008817045]`.

Probability MAE improved from `0.027332199` to `0.021128454`. Favourite
agreement improved from `0.878613` to `0.891618`; mean Spearman rank agreement
was slightly lower, `0.943117` to `0.940829`. The predeclared primary gate
therefore passed before winner evaluation.

## Gated winner comparison

Only after the teacher gate passed was `WIN_RESULT` loaded for evaluation.
Pseudo-Betfair improved paired multiclass winner log loss in all three folds.
Combined Sportsbet/pseudo log loss was `1.422993773 / 1.406826200`, delta
`-0.016167572`; the meeting-date bootstrap 95% CI was
`[-0.026691064, -0.005933558]`.

Actual scheduled-off Betfair log loss was `1.378690981` and the untouched
frozen `0.95 * Sportsbet + 0.05 * Betfair` diagnostic was `1.419596947`.
Neither diagnostic selected, tuned or altered the distilled model.

No economics or threshold search was run.

## Validation and claim boundary

- Six focused tests and `py_compile` passed.
- Independent probability normalization, identity, chronology and metric
  recomputation passed.
- Every JSON file parsed, all artifact-local checksums passed, and a full rerun
  was byte-identical.
- Protocol SHA-256: `b84122b005498d5cf9efd27a5078d38bed8eb1eace08edb7183b538befd1a12a`.
- Report SHA-256: `1b57275a928b677d278e90945656369be3258a5ffb7d01c13bc6fcdd37937703`.
- OOF predictions SHA-256:
  `219a7cf11205abb12b49ed45b18212a84ac824d002031d4123936b8c66c76982`.

Strongest supported claim: on this strict historical development population,
the single frozen Sportsbet-plus-context ridge correction predicted enough of
scheduled-off Betfair disagreement to improve both the outcome-independent
teacher target and winner log loss under the predeclared OOF gates.

Unsupported: deployability in production, prospective accuracy, reproduction
of exact Betfair scheduled-off semantics, profitability, ROI, EV, betting
value, promotion, activation, or weakening PR #142. This report does not
activate #137, persist a model, touch runtime, or score a forward cohort.

The repository's V2 registry was already retired on 2026-08-18, so the frozen
task card is a protocol/scope record only; no V2 claim or release receipt was
created or bypassed.
