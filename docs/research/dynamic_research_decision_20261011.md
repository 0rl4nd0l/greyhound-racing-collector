# Completed prediction research: decision and evidence, 11 October 2026

**No new challenger is recommended.** Retained target starting prices provide a much stronger retrospective benchmark than the form models. A small apparent market-plus-form improvement is mostly ordinary market calibration. Opponent-adjusted ability modestly improves the controlled recency representation, but both added and replacement versions are substantially worse than the existing frozen form model. This completes the bounded experiment with a negative decision; it is not a claim that every possible dynamic model is ineffective.

## Starting point and preserved commitments

The requested `bdef84e964e193705d670d5004570cedfce64117` is the executed historical source. Handoff prose was committed subsequently at `27fe8945`; this is our integration base. The October11 successor is `greyhound-fair-comparison-20261011/docs/research/fair_comparison_handoff_20261011.md`. Separate historical loader/recovery and four-fit walk-forward work were also found. Their changing populations were not substituted into these experiments.

Actual authorized membership remains **2,351 races /16,855 runner appearances /28 dates**:720 training (11dates),656 earlier development (7dates),975 exposed diagnostics (10dates). The975 remain exploratory. The original3,198-guide denominator also contains244 without result evidence,434 quarantined,169 withheld. Source-level quarantine selection remains a limitation even though our matched comparison loses no evaluation races.

The169 withheld targets were verified through allocation metadata only: five2026dates, May21/25/26 and June1/9, proposed split `test`, absent from the decoded authorized labels. Their outcomes were not opened and no fresh-exposure claim was inferred from the word “test.” Existing protected histories and reservations remain excluded. [Data and benchmark audit](retrospective_sp_benchmark_20261011.md).

Original guide history is user-attested pre-race evidence; hashes establish retained bytes. Final active starters and some static distance/grade context were reconstructed from results, without an independent pre-jump witness. All new ability states use complete authorized orders from **strictly earlier dates**. Embedded partial histories were not invented into full opponent fields. Same-day results and later opponent performance cannot revise earlier ratings. Native identity is qualified through the target's retained archive, without future identity backfill. New states omit grade, times, margins, sectionals and PIR because their extra interpretation was not qualified.

## Matched benchmark

The actual inventory has complete target SP for718/720 training,656/656 development and975/975 diagnostic races. Prices match exact native runner identities and complete admitted fields; no missing prices were fabricated. These are **post-result source-reported SP, used as a retrospective closing-price proxy**. Source/bookmaker, pre-jump observation timestamp, execution availability and costs are not established. Decision-time market benchmark coverage is therefore still zero, despite excellent reported-SP coverage.

All rows below use the same975 races and7,016 entrants. Lower log loss and Brier are better; Brier sums squared errors across a race before averaging races. Top credits split ties rather than arbitrarily selecting a favourite.

| Method | Log loss | Brier | Top credits /975 |
|---|---:|---:|---:|
| Normalized reported SP |1.514420|0.711114|411|
| Market-only calibration |1.503299|0.708087|411|
| Frozen matched hybrid65 |1.750911|0.795881|285|
| Calibrated tree95 |1.765262|0.799625|301|
| Market + frozen hybrid |1.502936|0.708089|405|
| Market + tree |1.503299|0.708087|411|
| Market + replacement recency |1.503204|0.708185|405|
| Market + replacement dynamic |1.503206|0.708174|406|

SP has34 tied-top races and400 uniquely correct choices;411 is fractional credit. All combination comparisons preserve that accounting. Full calibration bins, runner ECE, memberships, per-race scores and excluded training fields are retained.

Fixed Aug31-trained form models predict September development out of training. The first development date is meta warm-up; six expanding earlier-date folds score568 races. Final restrained corrections train on all656 and score975. Market calibration uses the same slope bounds and penalty as the two-coefficient form correction. Tree temperature, fitted on656 outcomes, is excluded from combiner inputs: the uncalibrated tree supplies the out-of-training signal. The calibrated tree is scored only as a standalone later comparator.

The hybrid correction's incremental LL versus calibrated market is−0.000363, descriptive whole-date95% interval[−0.001131,+0.000321]. Earlier chronological LL worsens from1.517552 to1.519092. Tree form weight is zero. Replacement dynamic gives−0.000093[−0.000992,+0.000751], loses its sign when individual dates are omitted, worsens Brier/top accuracy, and also worsens earlier LL to1.519405. Additive-state corrections were similarly negligible. **Reject these tested form corrections as new challengers.**

The installed incumbent cannot defensibly join this2025 comparison: its model training ends9July2026 and the exact installed DB/live feature route is unavailable for these historical targets. Nine installed model/configuration/source references were freshly hash-verified, including model `624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d`; reproducible installed identity does not establish historically eligible forecasts. No substitute probabilities were presented as production. [Complete benchmark and commands](retrospective_sp_benchmark_20261011.md).

## Dynamic ability and recency control

The small sequential rating uses centered fraction of opponents beaten, shrinkage toward neutral ability, fixed30-day decay, and a shrunk exact-track/distance residual. Dynamic updates additionally account for rivals' ratings frozen before that historical date. Six matched summary roles encode ability, context residual, recent change, support, context support and heuristic uncertainty. Missing ability remains missing, not a poor result. Uncertainty is a support/dispersion descriptor, not a calibrated posterior interval.

| Form representation | Earlier LL /656 | Later LL /975 | Later Brier | Correct /975 |
|---|---:|---:|---:|---:|
| Existing frozen65 |1.770609|1.750911|0.795881|285|
| Add uniform summaries |1.788669|1.830040|0.826254|253|
| Add recency summaries |1.787471|1.813733|0.819949|254|
| Add dynamic summaries |1.786595|1.811732|0.819159|260|
| Replace six means/rates with uniform summaries |1.788453|1.827099|0.825321|258|
| Replace with recency summaries |1.787756|1.813093|0.819936|258|
| Replace with dynamic summaries |1.786856|1.810945|0.819081|259|

Within each family the classifier, preprocessing, feature dimensions, populations, shrinkage and history coverage match. The replacement family has65inputs; the additive family71. Independent review identified that the initial addition design did not fully answer the requested replacement intervention. Exactly one fixed three-arm replacement amendment was recorded before those fits, after the additive results were known. This is transparent exploratory scope repair, with no alternative feature-removal sets or parameter search. All original negatives remain reported.

Replacement dynamic improves versus recency by−0.002148[−0.002989,−0.001246], but worsens versus the frozen baseline by+0.060034[+0.042558,+0.078457], and is worse on every later date. The modest opponent increment reverses on September9 and29. Complete-race state history exists for only21.5% of training appearances,53.3% earlier development and77.3% later; this strong support-distribution shift is a limitation, not grounds to silently exclude races. **Reject the tested dynamic candidate; retain no new challenger.** [Full definitions, results and reproduction](dynamic_ability_results_20261011.md).

## Where corrections help and hurt

Exhaustive accounting retains every race, including unchanged top selections, losses and lower-ranked probability changes. Frozen hybrid versus raw SP helps winner probability on290races and hurts on685. The restrained hybrid correction versus calibrated market helps509/hurts466. Replacement dynamic correction helps494/hurts481;936top sets stay unchanged and39change. Its apparent benefit is not confined to successful outsiders, and unchanged selections can still improve or damage log loss.

The following pre-recorded descriptive conditions use the market favourite as the outcome-independent anchor. They describe the replacement dynamic correction **relative to calibrated market**, not causal mechanisms or usable betting rules. Denominators and opposite effects are shown; every date/track and all analogous model contrasts are in the machine-readable diagnostic artifact.

| Condition | Races | Paired LL change |
|---|---:|---:|
| Improving recent performance |470|+0.000607|
| Declining recent performance |195|+0.000725|
| No complete-race state history |235|−0.002290|
| Stronger current opposition versus last event |94|−0.001045|
| Weaker current opposition |157|+0.001832|
| Observed exact-context support |437|+0.001392|
| No exact-context support |522|−0.001395|
| At least two decayed starts of support |96|+0.000953|
| Favourite probability above0.4 |363|−0.002178|
| Favourite probability0.2–0.4 |604|+0.001071|
| Six-runner field |171|+0.003637|
| Eight-runner field |532|−0.001266|
| Guide draw1–3 |475|+0.000679|
| Guide draw4–5 |198|+0.002640|
| Guide draw6–8 |297|−0.003100|

Guide draw is not a qualified early-pace interaction. The dynamic correction helps on five later dates and hurts on five: September11/12/17/18/30 are adverse. Among the largest track groups, it hurts AP_K (53races,+0.005488), MAND (46,+0.000399) and BAL (38,+0.000724), but helps RICH (44,−0.002966), WRGL (44,−0.004790) and LAU (36,−0.002692). These reversals are retained alongside all36tracks. No explanatory stratum is promoted into a rule. Missing-history gains do not show that missing history causes an edge. Ten dates, repeated model inspection and dependent runner appearances limit uncertainty statements; date resampling is descriptive, not independent confirmation.

## Opening contest and odds movement

The field audit changes an important assumption:1,224 retained races have identity-linked, explicitly labelled first-section box order, genuinely distinct from ambiguous CSV PIR. However, only2,385/16,855 target appearances have qualified prior-date same-track/distance calls. Requiring the complete target field and stable original primary boxes leaves **seven races: one training, one development, five later**; none has three prior calls for every runner. Physical call location, directional running style and independently captured decision-time vacancies remain unqualified. The broad speed/box-pressure recipe was already explored; nearest occupied neighbours across vacancy gaps is the remaining incremental idea. **Insufficient evidence for the requested interaction fit.** No missing rival was assigned zero pressure.

Movement inventory refreshed84references for28October10captures covering28different races: no repeated snapshots in that slice. A separately scoped older audit had291timestamped pairs but zero fully qualified fixed-WIN pairs; this is an older bounded result, not a claim that every project price is unavailable. No qualified authorized same-field movement population was established, so no movement fit was run. **Insufficient evidence.** Minimal next retention is two genuinely distinct source-observed snapshots, e.g.T−10 andT−2, with native race/dog/active-box identity, odds type/bookmaker, capture/publication times, source hashes and scratching changes. Reuse the existing owner's scheduled captures when they qualify; a copied old quote is not a second observation. Any extra source demand needs that owner's existing authorization or one bounded extension, never parallel research scraping. [Exact audit, exclusions and requirements](opening_contest_and_odds_inventory_20261011.md).

## Decision, missing evidence and fresh evaluation

| Hypothesis | Decision |
|---|---|
| Tested form adds useful information beyond calibrated reported-SP proxy |Reject tested corrections|
| Tested dynamic representation improves existing form forecast |Reject candidate|
| Opponent adjustment adds to the otherwise identical recency representation |Small exploratory increment; no challenger because baseline is substantially better|
| Nearby-runner opening contest helps |Insufficient qualified complete-field evidence|
| Form predicts later market movement |Insufficient qualified snapshot pairs|

No fresh evaluation is requested for a new challenger from this task. Existing frozen prospective studies keep their candidates, memberships, schedules, caps and analysis rules. If a later candidate is justified by new evidence, freeze its code, decision cutoff and matched incumbent/market-calibration contrasts before allocating new eligible dates; use full-field chronological capture, score paired race LL with date/week sensitivity, and set its sample/stopping rule before seeing outcomes. The169 withheld outcomes are neither opened nor repurposed by this recommendation.

Highest-value missing evidence, in order:

1. **Decision-time market, active roster and installed-feature snapshots together.** Enables the absent deployment-faithful market/incumbent comparison, which reportedSP cannot supply.
2. **Contiguous earlier complete-race histories with native identity.** Gives opponent ratings a proper warm-up and comparable support distributions instead of sparse28-date cold starts.
3. **Earlier native-linked ordinal calls plus decision-time draw/vacancy geometry for complete planned fields.** Reuse the1,224retained call fields; fill the exact missing histories before attempting neighbour interaction.
4. **Two genuinely separated qualified odds observations per eligible field.** Enables the movement diagnostic without putting later prices into an earlier forecast.

No profit calculation was made: executable decision prices, complete strategy accounting and applicable costs are absent.

## Execution, independent review and operations

Six predictor fits and49small scalar fits completed, with no failed optimizer trials or outcome-driven parameter searches. The initial three additive predictors and35scalar fits remain intact; scope repair adds exactly three replacement predictors and14scalar fits. Preparation-only runs, prior inventories and numerical replays are separate from fits. Failed test commands and repaired malformed-input tests do not silently become extra scientific trials.

Standards review found and repaired active-field, probability/label and input-hash validation gaps; all2,351source fields and32original metrics rechecked unchanged without fitting. Methodology review independently reproduced159artifact hashes,77,182held meta probabilities, training-only preprocessing and prior-date state recurrence. Thirty focused tests pass. [Standards review](dynamic_standards_review_20261011.md) and [methodology review](dynamic_method_review_20261011.md) remain separate. The root also replayed scores, reviewed scope and retained both positive and negative strata.

At current read-only inspection, collector PID2848247 is installed from `55f728a3`, UI PID407284, unchanged by this task. At15:14AEDT the collector metadata had21completed full lanes,295odds lanes,one odds deferral and forecast admission true; that is a timestamped operational observation, not scientific acceptance or a future freshness guarantee. The speed-pilot's terminal metadata records `COMPLETE_SINGLE_PLANNED_EVALUATION` onOctober9,12selected races and6retention accesses; its private results were not opened. Separate fair-comparison input readiness consumed one of five attempts, while earlier-start activation remained gated in its successor handoff. We made zero provider requests, service/runtime mutations, protected-outcome reads, merges, deployments, promotions or bets.

Canonical local evidence roots are `/mnt/tenn-nvme2/tenn/greyhound-dynamic-{form,benchmark,race-shape,research}-20261011-evidence`. See [portable integrated evidence](dynamic_research_evidence_20261011.json) for exact directories, output hashes and summaries. Full runner predictions, memberships/exclusions, states, source lineage, model coefficients, fit intents, date/track tables and numerical audit programs remain in those directories. Raw source/runner data are not embedded in the public draft. Commands in the component reports reproduce fits into fresh directories; numerical replay verifies existing fits without consuming new trials. The draft contains only this task's changes and explicitly depends on the historical modules at27fe8945, avoiding publication of hundreds of unrelated predecessor changes.
