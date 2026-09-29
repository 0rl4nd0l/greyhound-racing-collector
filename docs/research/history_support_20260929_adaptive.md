# Executed adaptive history-support shrinkage

**Result: no demonstrated improvement over the existing method.** On the same
177 chronological development evaluation races / 1,251 runners / 15 dates,
adaptive shrinkage slightly improves average scores over fixed half-strength,
but is worse than the unchanged full residual. Paired date-cluster intervals
include zero. Retain the existing method; do not advance this rule on these
results. This is a completed negative/inconclusive exploratory experiment,
not a reason to expand the search.

## Novelty and fixed rule

Reviewed #193's complete 1,000-event experiment ledger: 277 completed fits,
219 validation trials, 204 selection-rule evaluations. Its strengths
{.25,.5,1} were fixed across races; added uncertainty/history features changed
fitted coefficients and its rules selected runners. #197 decomposed saved
predictions and reported missingness groups, without adaptive fits. None of
those applies this prespecified support scalar to the whole race after fitting.
Missing indicators remain unchanged learned inputs in the base model; they are
not equivalent to this deterministic support multiplier.

The [protocol](history_support_20260929_protocol.json) was written and hash-bound
before any new adaptive evaluation. For runner i, set:

`q_i = min(retained_starts_i,5)/5 * mean(min(context_starts_ij,3)/3, j=venue,distance,grade) * observed_base16_fraction_i`.

An unavailable context count/rate contributes zero **support**, not a fabricated
zero feature value. Let `S_r = mean_i(q_i)` over the complete field,
`alpha_r = S_r / (S_r + lambda)`, and
`p_adaptive_i proportional to p_market_i * (p_full_i/p_market_i)^alpha_r`.

The sole candidate set is lambda `{.25,1}`. Minimize earlier OOF race log loss;
an exact tie chooses stronger shrinkage. No feature selection, threshold tuning,
support subgroup selection, lambda expansion or outer-result feedback occurs.
The scalar is race-wide: the unknown additive residual normalizer cancels.
This permits exact use of retained probabilities without reconstructing absent
inner model coefficients. Alpha .5 reproduces the existing fixed-half forecast;
alpha0 returns market and alpha1 returns full residual.

All three chronological periods select lambda **.25**. Validation race counts
are 101, 127 and 143; those populations overlap and are not independent repeats.
The original inner splits and outer dates are preserved. The 3 × 2 candidate
evaluations were sealed before opening retained outer predictions. All labels
are restricted to the existing admitted development population; earlier outer
dates can legitimately enter later validation. No new fits were needed.

## Proper scores and paired differences

Lower is better. Brier sums squared errors over each race, then averages races.
Top-pick accuracy uses fractional credit for ties and is secondary.

| Method | Log loss | Brier | Top-pick accuracy |
|---|---:|---:|---:|
| Same-time normalized WIN market | 1.412894 | .682928 | 45.20% |
| Unchanged full residual | 1.397606 | .677388 | 44.07% |
| Fixed half residual | 1.403559 | .679370 | 45.76% |
| Adaptive support residual | 1.401073 | .678414 | 45.76% |

Differences are **adaptive minus comparator**; negative favors adaptive.
3,000 paired calendar-date bootstrap draws, seed 20260929, preserve complete
race weighting within sampled dates. Simultaneous intervals cover six
LL/Brier contrasts, not the whole prior history of experimentation.

| Comparator | Δ log loss | Simultaneous 95% | Δ Brier | Simultaneous 95% |
|---|---:|---|---:|---|
| Market | −.011821 | [−.024009,+.000367] | −.004514 | [−.010951,+.001922] |
| Full | +.003466 | [−.003494,+.010426] | +.001026 | [−.002621,+.004674] |
| Half | −.002487 | [−.005704,+.000730] | −.000956 | [−.002620,+.000709] |

The primary adaptive-versus-half LL pointwise interval also includes zero:
[−.005268,+.000099]. Mean alpha is .64059 (range0–.79775), so some average
improvement over half may reflect a larger average adjustment rather than a
useful support relationship. This experiment does not isolate that explanation
from a fitted constant strength, and adding such a candidate after inspecting
these scores would be a new trial, not confirmation.

## Breadth and harm

| Evaluation period | Races / dates | Adaptive−half LL | Adaptive−full LL |
|---|---:|---:|---:|
| June 24–30 | 86 / 7 | −.002618 | +.001056 |
| July 1–2 | 16 / 2 | −.003587 | +.013144 |
| July 3–8 | 75 / 6 | −.002102 | +.004165 |

Adaptive improves LL over half on 96 races and harms 81; ten dates improve and
five worsen. Brier improves on 91 and harms 86. Removing any date retains a small
LL improvement over half (−.003267 to−.001936), but removing the best five races
reduces it to −.001127. Those five contribute 56.0% of the net LL improvement.
Worst harm versus half is +.05658 on WAR R7 June 29; best gain is −.06430 on
Capalaba R8 June 24. Full residual is better on average in every period, and
adaptive is worse than full on 99/177 races and 11/15 dates.

Prespecified support groups further weaken the simple sparse-history story:

| Race support | Races / dates | Adaptive−half LL | Adaptive−half Brier |
|---|---:|---:|---:|
| Low, S<1/3 | 38 / 13 | +.002409 | +.001026 |
| Middle, 1/3≤S<2/3 | 93 / 15 | −.000897 | +.000272 |
| High, S≥2/3 | 46 / 14 | −.009745 | −.005074 |

Low-support races deteriorate slightly relative to half and market; the pooled
gain versus half is concentrated in higher support. These are descriptive
groups fixed before scoring, not validated selection rules. Coefficient effects
and source support cannot identify causes of individual winners.

## Reproduction and artifacts

`scripts/history_support_shrinkage.py` resolves the original reservation union
and incident metadata before decoding the exact331-race foundation. It checks
saved inner-array hashes, runner order/membership, chronology and original OOF
log loss. Retained outer full/half probabilities replay against saved base16
preprocessing/coefficients. The market, labels and 16 features agree with the
foundation; all four methods use the same 177 complete fields.

Each output directory retains the protocol, six trial results and selections,
both candidate OOF predictions, exact admitted development inputs, evaluation
probabilities, all race/date/period/support scores, influence and uncertainty,
original saved outer models and derived training membership, input/code hashes,
environment identity and complete attempt ledger. Retained inner probabilities
are reused; absent original inner coefficients are not silently refitted.

```bash
cd /home/l4nd0/greyhound-history-support-20260929
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RESEARCH_PY=/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/.venv/bin/python
"$RESEARCH_PY" -m unittest tests.test_history_support_shrinkage -v
"$RESEARCH_PY" -m scripts.history_support_shrinkage --out /tmp/history-support-NEW
```

The initial successful attempt remains `history_support_20260929_adaptive_attempt1/`.
Final replay adds explicit retained predictions for both lambda variants and
treats nonfinite context as unavailable support; admitted values and metrics are
unchanged. No new parameter was added after results. The inherited development
population was inspected repeatedly, its latest-prejump quote inclusion is
retrospectively availability-selected, and the earlier protected-exposure
incident remains recorded. No fresh holdout or executable return is claimed.
