# Controlled retained-history depth experiment

Adding demonstrably earlier retained card history did **not** improve the fixed
residual method on the previously inspected development evaluation population.
Six new fits were executed, with no tuning or follow-up variants. The richer
route worsened mean race log loss by 0.002716 and whole-field Brier by 0.000913
against the original short-history route. This is a completed negative exploratory
experiment, not evidence that history depth never helps or a production DB replay.

## Qualification and fixed design

The [protocol](history_support_20260929_depth_protocol.json) was frozen before
fitting. Both methods used all 331 admitted races / 2,360 runners for their
chronological memberships, with zero history-conflict exclusions. Evaluation
covered the original 177 races / 1,251 runners / 15 dates (100% coverage), using
the unchanged three outer blocks. Training counts were respectively 154 races /
1,109 runners; 240 / 1,735; and 256 / 1,844. Training dates precede the respective
evaluation start; prior evaluation dates legitimately enter later training.

`short` uses each admitted target card. `richer` unions that card with normalized
history rows appearing in strictly earlier-captured admitted cards for the same
canonical dog token, capped at 20 starts. Each source retains its original
pre-jump admission/roster evidence. Prior rows also precede the target date;
same-day disagreements exclude a complete target field. This supplies an as-of
availability witness that a current database cannot supply by date alone. The
linkage still relies on canonical dog tokens and normalized history dates rather
than independent provider runner/history-event IDs. No current DB, operational
target result, or protected study outcome was accessed.

The reconstruction added 678 starts to 360 runners in 159 development races.
It changed 1,939 feature values across 10 columns; recent-five values did not
change. Evaluation enrichment affected 311 runners in 129 races. This is modest
retained-history extension, not a complete career. Of the 48 evaluation races
without direct added history, predictions can still change because preprocessing
and coefficients are learned from changed earlier training histories.

Both methods keep the original 16 canonical formulas, order, exact-distance
matching, grade-label handling and known-finish denominators. Neither switches to
production tolerance matching or to the v2 unavailable-context semantics. Each
fit learns its own training-only medians, missing indicators and standardization,
then uses identical race centering, L2=1, market log offset, `0.35*tanh(z/0.35)`
cap, full strength and whole-field normalization. Richer histories can separate
previously identical recent/retained columns; the resulting regularization
geometry is part of this fixed-definition depth intervention, not a separate
feature or hyperparameter experiment.

## Results

Lower scores are better. Brier is the sum over the whole field, then averaged
equally over races; accuracy uses fractional credit for tied top probabilities.

| Method | Race log loss | Whole-field Brier | Top-pick accuracy |
|---|---:|---:|---:|
| Same-time normalized WIN market | 1.412894 | 0.682928 | 45.20% |
| Short canonical16 residual | 1.397606 | 0.677388 | 44.07% |
| Richer canonical16 residual | 1.400323 | 0.678301 | 44.07% |

The paired richer-minus-short difference is **+0.002716 log loss**, with a
3,000-resample date-cluster percentile 95% interval **[-0.000592, +0.007000]**;
Brier is **+0.000913 [-0.000648, +0.002651]**. These are exploratory pointwise
intervals, with only 15 date clusters, and do not account for the substantial
previous experimentation on these same races. Market-relative gains remain
uncertain: richer-minus-market log loss is -0.012571 [-0.028768, +0.002824],
and Brier is -0.004628 [-0.013139, +0.004184].

| Original period | Races / dates | Richer − short log loss | Richer − short Brier |
|---|---:|---:|---:|
| June 24–30 | 86 / 7 | +0.003551 | +0.001608 |
| July 1–2 | 16 / 2 | -0.004208 | -0.001226 |
| July 3–9 | 75 / 6 | +0.003236 | +0.000574 |

Richer histories improved log loss in 73 races and harmed 104; 8 dates improved
and 7 worsened. The small middle period improved, while both larger periods
worsened. Leaving any one date out leaves a harmful mean log-loss difference
between +0.001134 and +0.003342; Brier remains between +0.000389 and +0.001287.
Removing the five largest harms still leaves +0.000985 log loss; removing the
five largest benefits leaves +0.004614. Added-history races average +0.003361
log-loss harm and unchanged-history races +0.000983. These are descriptive
categories fixed by input changes, not outcome-selected strategies.

Both helpful and harmful changes are retained. Largest log-loss benefit:
July 6 WAR R6, -0.074883; largest harm: July 5 CAPALABA R7, +0.113865.
These examples describe probability movement and scores, not causal reasons
for the observed winner. Complete per-race/date metrics and both influence tails
are in the [evidence summary](history_support_20260929_depth_evidence/summary.json).

## Reproducibility and validation

All six artifacts are explicitly **NEW_DIAGNOSTIC_FIT_NOT_ORIGINAL**. Each stores
the complete preprocessing and coefficients, ordered feature contract, training
and evaluation membership, input/admission hashes, environment and executable
identity, code hashes, probabilities and append-only attempt ledger. Full local
inputs and receipts reside at
`/home/l4nd0/greyhound-history-support-output-20260929/depth_attempt1`;
the [portable manifest](history_support_20260929_depth_evidence/local_artifact_manifest.json)
pins every file. Portable evidence includes all six complete model receipts,
evaluation predictions, race metrics, summary and ledger. Hashes identify these
sources; the admission and timing evidence above, not hashes alone, qualify them.

The three newly fitted short models reproduce every retained #193 outer
probability **exactly (maximum difference 0.0)**. Independent evaluation replay
from all six saved model receipts differs by at most **2.22e-16**, with no new
fits. Five targeted tests cover predecode identity rejection, complete fields,
duplicate rejection, unchanged baseline values, chronological failure retention,
and paired race/date uncertainty. Independent review is recorded in the overall
research review.

An initial shell attempt used an absent `python` alias and exited 127 before
protocol creation or data decoding; it is recorded in the protocol. The sole
actual experiment attempt completed all six fits. A later metadata-only
reconstruction edit temporarily differed from its frozen source pin; the exact
fitted source was restored and archived, its stricter successor retained
separately, and all final protocol/source pins were reverified. The stricter
source reproduced the paired feature bytes exactly. No results were refitted or
protocol choices changed in response to this provenance repair.

```bash
cd /home/l4nd0/greyhound-history-support-20260929
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RESEARCH_PY=/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/.venv/bin/python
"$RESEARCH_PY" -m unittest tests.test_history_depth_comparison -v
# Exact frozen inputs must still match; use a fresh destination, never overwrite.
"$RESEARCH_PY" -m scripts.run_history_depth_comparison --out /tmp/history-depth-NEW
```

Retain the existing method. This experiment supplies no reason to promote the
richer candidate or to amend October. A future history intervention should first
address a demonstrated coverage/definition defect with durable availability and
runner/history-event provenance; adding these few older retained starts alone
was not beneficial here.
