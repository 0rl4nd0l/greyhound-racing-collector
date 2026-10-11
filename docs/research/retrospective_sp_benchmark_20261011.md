# Matched source-reported SP benchmark, 11 October 2026

The actual retained result pages contain target-race starting prices. This changes the available retrospective comparison: the earlier absence of a decision-time market benchmark did not mean all useful target prices were absent. On the same 975 exposed races the normalized reported-SP probabilities are substantially better than either frozen form predictor. Almost all apparent combination improvement over raw SP comes from market-only calibration; the tested form corrections do not establish added value.

## Eligibility and scientific boundary

The source population is unchanged: 720 training races on 11 dates, 656 earlier development races on seven dates, and 975 later diagnostic races on ten dates. Source feature and prediction artifact manifests were hash-verified before body access. Only exact admitted keys in the original result manifest's allowed train/validation splits caused body decoding. Each price was matched to the admitted runner by guide box and native dog identity, with exact raw-body hash agreement. No missing runner price was synthesized and no partially covered field was normalized.

Full-field reported SP exists for 718/720 training races, 656/656 development and 975/975 later races. Two training races were excluded for missing/invalid active-runner SP: SAL race9 on20August and DARW race7 on24August. Their source receipts and exclusions are retained. All evaluated models share all975 later fields; none gains coverage by omitting difficult races within this already-admitted clean-result population.

These are post-result source-reported dollar SP values, interpreted as conventional decimal prices and normalized by inverse price. They were retrieved in2026 for2025 targets. The source does not establish the bookmaker, an observation time before jump, executable availability, costs or a decision cutoff. Final starter membership and some static context also came from official results. Thus this is a **retrospective closing-price proxy diagnostic**, not a deployment-faithful market comparison or profitability test. Decision-time market coverage remains zero.

The installed incumbent is reproducible in its own retained runtime lineage, but not valid on this historical population: its training cutoff is9July2026, after the2025 targets, and its exact DB/live feature route is absent here. Fabricating an incumbent probability from a similarly named subset of65 features would not fix this. The root independently rechecked all nine installed model/config/source references against the October11 authority inventory; all matched. Incumbent paired coverage here is zero.

The169 reserved test allocations remain outside body access. Original corpus metadata places them on21,25,26May and1,9June2026; the original labels-access contract allows only train/validation. Retained acceptance and successor audit report them unopened; this run opened none. Those statements do not imply a new audit of every other process's access history.

## Frozen fitting scheme

`run_retrospective_sp_benchmark.py` uses saved hybrid65 and uncalibrated tree95 predictions from predictors trained through31August2025. All656 September1–8 predictions therefore precede meta-model outcome fitting. The saved tree temperature was itself fitted on those656 outcomes, so it is deliberately excluded from meta-model inputs and earlier prequential comparison. It remains a standalone comparator on975.

The fixed correction is `softmax(a log(p_SP) + b log(p_form))`, with `a` in[0.5,1.5], `b` in[0,0.5], and equal-race NLL plus`.01*((a-1)^2+b^2)`. The market-only control uses the identical slope and penalty with no form coefficient. No parameter search occurs. The first88-race development date is warm-up; six expanding earlier-date fits score the remaining568 races. Three final fits use all656 and score975. Total21 small optimizer executions; every intent, fit and date membership is retained. No predictor refitting occurred in this benchmark.

## Results

| Same975 races | Log loss | Brier | Fractional top-one accuracy |
|---|---:|---:|---:|
| Normalized reported SP |1.514420|0.711114|42.15%|
| Frozen hybrid65 |1.750911|0.795881|29.23%|
| Calibrated tree95 |1.765262|0.799625|30.87%|
| Market-only calibration |1.503299|0.708087|42.15%|
| Market + hybrid65 |1.502936|0.708089|41.54%|
| Market + tree95 |1.503299|0.708087|42.15%|

Market-only slope is1.204653. The hybrid combination learns slope1.189448 and form coefficient0.036783. Its incremental log-loss difference against calibrated market is−0.000363, descriptive date-cluster95% interval[−0.001131,+0.000321]. Leaving one date out gives[−0.000599,−0.000172]; the small mean is not proof of robust improvement. It helps winner probability in509 races and hurts in466. Its Brier score is slightly worse. Earlier prequential performance also worsens:1.519092 versus market calibration1.517552. The tree learns exactly zero form coefficient and reproduces the market control to numerical optimizer precision.

Reported SP has34 tied-top races,400 uniquely correct selections, and411 fractional winner credits. The hybrid combination has405 unique correct choices and no ties. Comparing405 directly with400 without disclosing fractional tie credit would misrepresent selection accuracy.

Decision: **reject the claim of demonstrated form value beyond this calibrated SP proxy** for these tested combinations. The tiny hybrid diagnostic gain does not justify promotion or another search. The ordinary market calibration effect is much larger but is also retrospective exploratory evidence, not a new market-beating betting method. Dynamic/recency correction results are appended separately when their fixed fits complete.

## Artifacts and reproduction

Evidence root: `/mnt/tenn-nvme2/tenn/greyhound-dynamic-benchmark-20261011-evidence/run-01`. `eligibility.json`, `memberships.json`, `exclusions.json`, and `matched_sp.jsonl` retain inventory/provenance; `protocol.json` preceded fitting; `fit-intent-*` and `fit-*` retain all21 trials. `predictions.jsonl` retains every runner probability and outcome in both exposed blocks. `race_scores.json`, `summary.json` and `paired_summary.json` retain per-race scores, calibration bins, per-date contrasts, shared date resampling and leave-one-date-out sensitivity. Independent arithmetic replay matched33,078 meta probabilities exactly without fitting.

Use the existing research interpreter and a new output directory; never overwrite a consumed run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-operator-ui-r3-python311-20260803-9d58f340/bin/python \
  scripts/run_retrospective_sp_benchmark.py --output /path/to/new-output --prepare --fit
OPENBLAS_NUM_THREADS=1 /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-operator-ui-r3-python311-20260803-9d58f340/bin/python \
  scripts/summarize_sp_pairing.py /path/to/new-output
```

Four focused synthetic tests verify identity/missing-price rejection, duplicate boxes, bounded fitting without future-label dependence, and race metrics/tie credit. No provider requests, protected outcomes, runtime changes or model promotion occurred.
