# Early-speed neighbour feasibility protocol — 28 September 2026

Frozen before opening new raw-card values or computing predictive associations.
Parent source is PR #193, `956d8289b8be062e9fe5ccdf554b9a9462391c20`.
This is a separate offline research branch; PR #194, operational collection,
private results and all prospective populations remain outside this task.

## Question and novelty gate

The broad proposal duplicates PR #193's sectional field gaps/ranks,
faster-box-neighbour counts, neighbour sectional gaps and favourite×pressure.
The narrower, potentially untested mechanism is **loss of a historical early
advantage when the nearest occupied runner on either side is comparably fast,
with empty-box separation represented explicitly**. This is an occupancy
association, not a claim of observed interference or running style.

Primary hypothesis: among runners with supported comparable early measurements,
relative early advantage contributes less when a nearest occupied neighbour has
comparable or better early speed; this may expose overestimated favourites.
No supporting variants are authorized in this run.

## Access and fixed audit sample

Use only the exact 331-race / 2,360-runner foundation from
`/home/l4nd0/greyhound-offline-systematic-output-20260924/foundation`.
Check source hashes, original July/form/August reservations, the current
September 28 reservation review and the 58-race access-incident record first.
Project only race/date/box/dog identity from the hash-pinned development JSONL;
never decode its complete records during feasibility. Reject denied keys and
all dates outside June 10–July 9 before opening a card. Verify every selected
card and sidecar hash before parsing. All historical dates must strictly precede
both target and card capture dates and precede the earliest protected date.

Choose one race per literal target venue by the smallest SHA256 of
`early-speed-20260928-v1|race_id`, tie-breaking by race ID. Freeze the resulting
sample identities before opening cards. Inspect all runners and all historical
rows of those sampled cards. Separately count coverage over all 331 races,
including missing, malformed, zero, negative and nonfinite fields. This sample
is for semantics only; no winner, favourite outcome or association selects it.

## Measurement and feasibility gates

Require a source-supported, runner-specific early call/time with documented
units, start/end measurement point, layout and distance; historical event identity
and pre-target availability; verified occupied boxes at decision time. A heading
`1 SEC`, a plausible positive number or a multi-digit `PIR` is insufficient.
PIR equalling finish is evidence against using that value as early position.
Name/date/track/distance corroboration without a unique historical race identifier
must be disclosed, not silently treated as a native event join. Style and actual
interference are unavailable unless independent pre-target evidence establishes
their meaning. No inferred styles, cross-layout alias pooling or target-result
reconstruction. No missing sectional becomes a slow start.

For candidate construction require at least three qualified earlier starts at the
exact target layout/distance per runner, all current field runners qualified,
and an as-of complete active roster. Mean the most recent three early times.
Faster means a lower time. Let `a_i=(field median-time_i)/field median`.
For nearest occupied boxes left/right, define `c_i` as the sum of
`1[time_neighbour <= time_i]/box_gap`; absent track-edge neighbours contribute
zero, unknown runner measurements disqualify the race. Features are `a_i`,
`c_i`, and `a_i*c_i`. Retain exact neighbour IDs/gaps/vacancies; gap weighting is
a prespecified statistical assumption, not measured collision probability.

Proceed only with supported semantics and >=50 qualified training races on >=5
dates before the first test period, >=30 total qualified test races on >=5 dates,
and at least one test race in each fixed period. These are practical lower bounds,
not a power claim. Otherwise stop modelling and report acquisition needs. Do not
relax coverage thresholds, substitute PIR/finish/speed proxies or run alternatives.

## Conditional experiment, fixed before performance

Compare normalized corrected WIN market, refitted established base16 residual,
and base16 plus the three above features on exactly the same qualified races.
Refit research copies only. Use the existing deterministic capped residual
`softmax(log p + .35*tanh(X beta/.35))`, mean race log loss plus
`0.5*||beta||²`, zero initialization and L-BFGS-B settings from PR #193.
No hyperparameter/feature selection. Training-only medians/missing indicators,
means and scales for base16; early features require complete measurements.
No learned track adjustment or cross-track normalization.

Whole-race date splits: training strictly before June 24, July 1 and July 3;
tests June 24–30, July 1–2 and July 3–9 respectively. Earlier evaluation dates
may join subsequent expanding training. Six fits maximum (two methods × three
periods). Stop on qualification/optimization failure; preserve failure and do
not tune. Report matched-subset composition against all 331 development races
and 177 previously inspected evaluation races.

Primary scores: paired race log loss and summed multiclass race Brier; report
each period, each date and pooled results. Paired date-block bootstrap, 2,000
replicates, seed 20260928; descriptive 95% intervals and leave-one-date-out
sensitivity, without claiming fresh confirmation from the 15 reused dates.

Vulnerable favourite rule: unique market favourite, `a_i>0`, `c_i>=1`, and
interaction model probability at least .01 below market. Fixed thresholds,
training-fitted model only, no threshold search. Evaluate every selected
favourite including winners; report all eligible/selected races and dates,
sum of market probabilities (expected wins), actual wins, actual-minus-market
calibration and paired binary Brier. Zero selections is a valid result.
No betting returns. Stop after this one experiment, whatever its sign.

## Coordination and accounting

All writes confined to the new worktree and new audit-output directory. Existing
worktrees, raw inputs, models, reservations and study ledgers are read-only.
The research skill delegates only a source/Markdown novelty audit to a helper,
with its own new note; the parent owns audit code and evidence. No other active
agent is exposed by this thread's agent registry. No runtime/provider locks are
claimed. Use one low-priority process, one numerical thread if fitting becomes
possible, no new services, requests to providers, external model APIs or spend.
Retain the protocol, exact source/input hashes, audit sample, every failure and
trial status. A feasibility stop records zero fits, not negative model evidence.
