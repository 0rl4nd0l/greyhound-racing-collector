# Dynamic ability: completed three-arm comparison, 11 October 2026

**Decision: reject this candidate as a replacement for the frozen 65-feature model.** Opponent adjustment modestly improves an otherwise identical recency representation, but all three added-state models predict substantially worse than the frozen baseline. No parameter search, follow-up fit, model promotion or live change followed this negative result. The result concerns this parsimonious sequential representation on this sparse corpus; it does not establish that every dynamic ability model is ineffective.

## Actual executed comparison

All models have exactly the same 720 training, 656 earlier development and 975 later diagnostic races; these are 5,204 / 4,635 / 7,016 runner appearances on 11 / 7 / 10 dates. The 975 races were already exposed development. The 169 reserved races were not opened. No October 11 recovered population was substituted.

| Representation | Development LL | Later LL | Later Brier | Later correct top picks | Later runner ECE |
|---|---:|---:|---:|---:|---:|
| Frozen matched 65-feature hybrid | 1.770609 | **1.750911** | **0.795881** | **285/975** | 0.009329 |
| Existing 65 + average state | 1.788669 | 1.830040 | 0.826254 | 253/975 | 0.021794 |
| Existing 65 + recency state | 1.787471 | 1.813733 | 0.819949 | 254/975 | 0.020479 |
| Existing 65 + dynamic state | 1.786595 | 1.811732 | 0.819159 | 260/975 | 0.019907 |

ECE is a descriptive fixed-width runner probability-bin error; full bin counts, means and outcome rates are saved. Race log loss is primary; Brier is the sum of squared multiclass errors per race. All 975 races and all 7,016 runners remain in every later comparison; no sparse-history race is removed. None of these metrics establishes profitability.

Later paired race log-loss differences, negative favouring the first model:

| Contrast | Difference | Descriptive 95% paired date-bootstrap interval |
|---|---:|---:|
| Recency minus average | -0.016307 | [-0.021103, -0.011808] |
| Dynamic minus recency | -0.002001 | [-0.002804, -0.001128] |
| Dynamic minus frozen 65 | +0.060821 | [+0.042750, +0.078217] |

The principal dynamic-versus-recency contrast isolates opponent adjustment: both arms use identical dates, complete prior-race histories, six state fields, decay, context residuals, shrinkage, preprocessing, classifier and penalty. Average versus recency isolates the fixed decay within the same added-state representation. Frozen 65 versus average measures the consequences of adding this particular state/history representation. This is an additive feature experiment; it does not claim that six features replaced all existing form summaries.

## Representation and information boundary

A completed admitted race provides each runner's centred pairwise finishing fraction, `1 - 2*(position-1)/(field_size-1)`. This uses qualified finishing order, without invented margins, intervals between positions, timing, sectionals or PIR semantics. Historical embedded starts do not provide complete native opponent fields and therefore do not generate synthetic races. Baseline original-guide features remain unchanged.

The average arm retains unweighted past race performances. The recency arm weights each performance by `2**(-age_days/30)`. The dynamic arm uses that same decay, but its performance observation adds the mean of its opponents' **then-current prior-date** ability plus exact track/distance residual. Global ability is a weighted average shrunk toward zero by three prior starts. The context residual pools performance minus the runner's own pre-event global ability at the exact track/distance, shrunk by five prior starts. Shared race-level track effects cancel in a ranking; the retained residual describes a runner's context deviation. No numerical grade hierarchy is asserted or used.

This is a small sequential ridge-style rank-score model, not a jointly smoothed latent-state posterior. It never retrospectively updates an earlier opponent estimate using a later performance. Context residuals are present in all three arms; the dynamic contrast does not separately identify a benefit from introducing context. The six downstream inputs are global ability, exact-context residual, latest performance minus current global ability, weighted support, weighted context support and a heuristic support/dispersion uncertainty score. The latter is `sqrt((1 + weighted performance dispersion)/(3 + weighted support))`, **not** a posterior standard deviation or calibrated credible interval.

All races on a date receive states before any outcomes on that date update history. A missing current native identity cannot inherit an identity discovered in a later card. Entire races with missing/ambiguous native identities or repeated same-date native dogs would be excluded from state updates, while remaining in prediction coverage; there were zero such exclusions here. Missing ability is null with support zero, rather than a bad rating. Known context without observations has zero shrunk residual and support zero; unknown distance has null context fields. Ratings update after prior development dates, but predictive coefficients and preprocessing remain frozen from the 720 training races.

The exact target field and static distance inherit retrospective official-result reconstruction. They are not independently witnessed decision-time rosters/context. Native IDs inherit each target's as-of qualified context archive; future identity resolution is not pooled backwards. The baseline also retains previously disclosed optional unqualified source-number timing descriptors. The present experiment introduces no new timing interpretation and makes no stronger pre-race availability claim.

## Sparse evidence and reversals

The rating can learn only from the 2,351 admitted complete races on 28 sparse dates. There are 7,839 distinct dogs with admitted updates; this is not a complete career graph.

| Split | Appearances | Any prior complete-race rating support | Exact track/distance support | Unknown target distance |
|---|---:|---:|---:|---:|
| Training | 5,204 | 1,119 (21.5%) | 623 | 319 |
| Earlier development | 4,635 | 2,470 (53.3%) | 1,188 | 233 |
| Later diagnostic | 7,016 | 5,427 (77.3%) | 3,126 | 123 |

The increase in support is a material distribution change. It is a plausible limitation of this sparse-corpus experiment, not a demonstrated causal explanation for the negative result or permission to tune until it improves. Existing baseline form has original-guide prior history even when these complete-race states are missing.

Dynamic versus recency improves on eight of ten later dates; it worsens on September 9 (+0.000426 LL, 105 races) and September 29 (+0.000175, 97 races). It worsens on 13 of 36 later tracks. Examples are Grafton (+0.011522, 12 races), Temora (+0.011347, 27) and Sale (+0.007575, 31); it helps Gosford (-0.007251, 33), Hobart (-0.010759, 15) and Gawler (-0.010800, 29). These are descriptive examples from the complete saved table, not selected betting rules. Several tracks reverse between earlier and later blocks, including Bulli, Geelong, Horsham, Murray Bridge, Northam, Temora and Warragul; other tracks reverse in the opposite direction.

Dropping each later date in turn keeps dynamic-minus-recency between -0.002294 and -0.001732, and dynamic-minus-frozen between +0.056781 and +0.067412. All date and track scores are retained, including losses and unchanged top selections. Only ten later dates support the 2,000-draw bootstrap. Cross-date repeated dogs, model-selection history, admission selection and retrospective source limitations are not resolved by these intervals. They are exploratory.

## Trials, validation and reproduction

Exactly three fixed fits converged: average 257, recency 243 and dynamic 259 L-BFGS-B iterations. All use the same hybrid objective: half mean winner negative log likelihood, half mean within-race strict-pair logistic loss, with `0.5*sum(beta**2)/N_train_races`. Training-only medians, missing flags, centring and scaling are saved. There are no temperature/calibration fits in this experiment. The frozen baseline is replayed from saved probabilities. Earlier development predictions are out of sample for coefficient training and can supply a separately evaluated restrained market correction.

Nine focused tests cover future/target outcome invariance, same-day race-order invariance, opponent strength, sparse shrinkage, missing identity, unknown/mismatched context and invalid finish orders. The no-fit verifier checks every artifact hash, independently reconstructs probabilities from saved coefficients/preprocessing, and recomputes log loss/Brier/top-selection arithmetic. Maximum probability difference is `6.11e-16`. This is a numerical self-check; independent methodology/implementation review belongs to the orchestrator's separate review.

Code and experiment source commit: `0abaf723`. Retained evidence root: `/mnt/tenn-nvme2/tenn/greyhound-dynamic-form-20261011-evidence/`.

- `preparation-01/`: retained no-fit preparation and its original source pins.
- `run-01/protocol.json`: compact frozen protocol and executing source hashes.
- `run-01/input_provenance.json`: hash-pinned authorised source manifests and exact opened target files.
- `run-01/membership.jsonl`: all 2,351 race memberships.
- `run-01/runner_states.jsonl`: all 16,855 prior-date states and diagnostics-only current opponent means.
- `run-01/state_update_lineage.jsonl`: qualified updates, then-current opponent ratings and source dates; later updates never backfill earlier states.
- `run-01/fit_intent_*.json`, `model_*.json`: all three attempts and learned preprocessing/coefficients; no scientific failures.
- `run-01/{train,development,later}_predictions.jsonl`: all model probabilities, pinned before scoring.
- `run-01/{development,later}_race_scores.jsonl`, `summary.json`: every paired score, date/track result, leave-one-date-out sensitivity and calibration bin.
- `run-01/artifacts.sha256.json`, `numerical-replay-01.json`: immutable output hashes and no-fit verification.

Reproduce with a **new** output directory; the runner refuses to overwrite previous evidence. On this host use the retained numerical environment and one thread:

```bash
PY=/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-operator-ui-r3-python311-20260803-9d58f340/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -B -m pytest --noconftest -q -o addopts= tests/test_historical_dynamic_ability.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 "$PY" -B scripts/run_dynamic_ability_experiment.py --output /new/exclusive/dynamic-run
OPENBLAS_NUM_THREADS=1 "$PY" -B scripts/verify_dynamic_ability_experiment.py /new/exclusive/dynamic-run --output /new/exclusive/numerical-replay.json
```

No provider requests, live owner changes, service modifications, protected-label access or deployments occurred. The smallest data improvement that would enable a stronger version of this particular experiment is an authorised contiguous earlier complete-race warm-up with native identities and qualified pre-race fields; this would reduce state cold-start and support-distribution drift. It is a data requirement, not an acquisition request made by this implementation. No new dynamic challenger is recommended from these results.
