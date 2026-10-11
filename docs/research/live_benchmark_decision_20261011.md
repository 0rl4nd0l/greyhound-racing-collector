# Retained live prediction audit — 11 October 2026

**The installed model's improvement over its decision-time market remains unestablished.** This audit executed the benchmark, rather than substituting the 975-race retrospective starting-price study. It found **zero fully qualified comparisons** in the inspected nonreserved retained populations. Seven original live shadow forecasts support a separately labelled, incomplete-field diagnostic: the current installed artifact is slightly worse than the market on both probability losses. That tiny diagnostic cannot establish the performance of installed production.

## Population and evidence reconciliation

Allocation manifests were read before outcome access. Exact protected membership, the original 114 closed-study races, frozen prospective evaluations and all 169 withheld historical races remain excluded. Current operational labels do not release reserved outcomes. [Inventory and access audit](live_benchmark_inventory_20261011.md).

| Retained population | Accounting | Benchmark disposition |
|---|---:|---|
| July 17–22 residual journal |277 original records /275 races|65 records excluded by scientific allocation|
| Authorised original journal |212 records /210 races /6 dates|All 212 authenticate against original append/stdout, input hashes and complete pre-race WIN capture|
| Repeated forecast horizons |2 records|Earlier records preserved; deterministic latest original per race selected before results|
| Primary original forecasts |210 races|203 have no retained official result join; seven have partial result fields|
| Strict decision-time comparison |0 races|No loss, accuracy or calibration estimate is asserted|
| Partial-field diagnostics |7 races /56 runner appearances /2 dates|Original full and half probabilities; no runner removed or probability renormalized|
| Additional July 31 manual requests |23 authorised requests;3 READY|No retained official results; independent durable pre-jump completion witness also missing|
| Other early R3 requests |5 nonreserved jobs|All failed `PROCESS_OUTPUT_INVALID`|
| Named October operational roots |551 identities,539 READY,10 REJECTED,2 FAILED|Separately reserved; outcomes not opened|

The October census is a **04:38 UTC snapshot**, not a claim about a later mutable runtime. These are named artifact populations, not an exhaustive scan of every project file. Counts from distinct ledgers are not silently added into a single service denominator. All exclusions, duplicate horizons, unsuccessful requests and artifact references are retained in machine-readable files.

The seven official joins contain four or seven finishing runners, while **every original forecast contains eight**. Retained `participant_count=8` came from a shadow prediction roster; it is not an independent result-field witness. No retained terminal status establishes whether missing runners were scratched, substituted, DNF or simply omitted by capture. Treating them as established ordinary losing starters would overstate the evidence. The diagnostic keeps the entire original probability field and official winner but cannot establish an unchanged market contract. No later price, starting price or reconstructed active field was substituted.

A Traralgon result appears first as a four-runner prefix and later as seven compatible positions. Exact source identity, winner and overlapping positions permit a derived richer projection; both originals remain. This repairs a duplicate-snapshot join, **not** the missing eighth-runner evidence. Exact documented venue aliases also prevent three reserved forecasts escaping exclusion. [Alignment and correction trail](live_benchmark_alignment_20261011.md).

## Executed all-race diagnostic

Current installed artifact SHA-256 is `624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d`, with manifest `8537cbc3…` and configuration `f8a3c321…`. The July records used this frozen artifact contemporaneously in **shadow** mode; July authority retained the market baseline. They are genuine original forecasts, not new replayed predictions and not evidence that production selected the residual at those July cutoffs.

Each comparison uses all eight original runners and normalized inverse contemporaneous WIN odds. Scores below are **partial-result-field diagnostics only**. Race Brier is the sum of runner squared errors; lower losses are better. Exact tied maxima share top-selection credit.

| Forecast | Races | Log loss | Brier | Top credit |
|---|---:|---:|---:|---:|
| Contemporaneous normalized market |7|1.401722|0.619411|4/7|
| Current artifact, original full-strength shadow |7|1.413075|0.635906|4/7|
| Original half-strength shadow, same races |7|1.405818|0.627073|4/7|

Full-strength model minus market: **+0.011353 log loss, +0.016494 Brier**. Winner probability improves on three races and worsens on four. All seven top selections are unchanged, including three losses. Thus there are no omitted favourable or unfavourable changed-top races. Original full/half comparisons share exactly the same seven races; they are not 14 independent races.

| Source date | Races | Full minus market log loss | Full minus market Brier |
|---|---:|---:|---:|
| July 17 |5|+0.044058|+0.031968|
| July 18 |2|−0.070411|−0.022191|

Removing either date reverses the sign. The largest improvements and deteriorations are preserved per race, including Bendigo12 (−0.171489) and Traralgon9 (+0.159344). A descriptive whole-date 10,000-draw bootstrap gives log-loss difference interval **[−0.070411,+0.044058]**. With only two dates it largely restates the date contrast, not reliable generalization uncertainty; repeated dogs further limit independence.

Prediction lead ranges **20.73–54.42 minutes**, median41.95. Capture-start age at forecast computation ranges **154.99–353.71 seconds**, median224.53. Provider quote-publication age is unknown for every race. Original feature-freeze, capture, computation, durable append and actual result-attempt completion timestamps are retained separately. Acquisition age must not be described as market publication age.

Ten fixed runner probability bins are in the scorecard. Runner-weighted ECE is **0.060376 model /0.069227 market** on only56 runners. This noisy bin statistic does not overturn worse proper losses or establish reliable probabilities. No calibration was fitted or replaced.

## Selective prediction extension

Selectors were specified before performance inspection: maximum model win probability; minimum same-distance prior-start count across the sealed original field; and maximum `p × decimal_WIN_odds − 1`. The history scalar was resolved from retained pre-race feature names before scoring. High winning probability ranks confidence; reliability requires calibrated probabilities; an attractive price requires those probabilities to be credible relative to the offered odds. The last selector measures probability/price disagreement, not realised value or returns. [Frozen selector rules](live_selector_protocol_20261011.md).

**Primary selector thresholds use the result-independent 210-forecast population.** Earlier July17–19 supplies79 development races; thresholds then apply unchanged to131 July20–22 forecasts. Winner outcomes never enter threshold construction. Every original selected/pass disposition is retained. Each selector has a market-confidence comparison selecting exactly the same number of forecasts, with deterministic identity tie-breaking. This batch ranking is diagnostic, not an executable threshold rule.

| Selector | Selected at nominal10% |25%|50%|Full|
|---|---:|---:|---:|---:|
| Model confidence |9/131 (6.9%)|19/131 (14.5%)|42/131 (32.1%)|131/131|
| Relevant history |21/131 (16.0%)|35/131 (26.7%)|57/131 (43.5%)|131/131|
| Price disagreement |8/131 (6.1%)|32/131 (24.4%)|59/131 (45.0%)|131/131|

Discrete thresholds, ties and temporal distribution changes make actual coverage differ from nominal comparison points. **None of the 131 later forecasts has a retained matching official result**, so primary selected loss, calibration and accuracy remain unavailable at every coverage. Missing results do not change selected identities, count as losses, or trigger replacement selection. Three additional July31 manual forecasts remain a separate incomplete-evidence source, with explicit pass/unavailable dispositions; they do not change these frozen journal thresholds.

For completeness, a **secondary result-availability-conditioned diagnostic** uses the five July17 joined records to freeze thresholds and the two July18 joined records to calculate illustrative selected scores. It is not the primary selector test, not an untouched holdout, and inherits unresolved result fields. Full-strength outcomes:

| Selector | Later counts at10% /25% /50% /full | Paired LL difference where selected |
|---|---|---|
| Model confidence |0 /1 /1 /2|−0.032701 for the one selected race;−0.070411 for both|
| Relevant history |0 /2 /2 /2|−0.070411 for both|
| Price disagreement |0 /0 /0 /2|−0.070411 at full only|

Matched market-confidence selection picks the identical race sets in these tiny nonempty comparisons. The one selected race is won by both model and market top choices; both score1/2 at full coverage. Selected calibration/Brier/date tables and half-strength diagnostics are retained, but one later date cannot support date-cluster uncertainty. Higher selected accuracy here supplies no evidence of a distinctive selector advantage or betting edge. No threshold, subgroup or model was tuned to these outcomes.

## Operational coverage and decision

The July daemon ledger records **431 nonreserved attempted races**:210 with an appended forecast (**48.72%**) and221 without. The1,080 blocked attempts include repeats:369 target-grade mismatches,346 conflicting duplicates,145 invalid target venues,140 unsafe feature metadata and80 timestamp-order failures. They are not zero-probability forecasts. The seven result diagnostics cover **3.33% of appended forecasts /1.62% of attempted races**; strict comparison coverage is0%.

This is a denominator of actual recorded attempts, not every calendar race or every promised opportunity. Discovery-only October records and July skipped stages do not establish races missed through downtime, scheduling or capacity; those unobserved dispositions remain unresolved. [Full coverage ledger](live_benchmark_inventory_20261011.md).

1. **Valid decision-time population:** none established under all evidence and allocation gates; seven incomplete-field original-shadow diagnostics remain useful but weaker.
2. **Improvement:** inconclusive for installed production. The diagnostic point estimates worsen proper probability scores and reverse by date; no superiority claim is supported.
3. **Coverage:**210/431 July attempted races produced forecasts, but none closes the full benchmark. Recent539 READY forecasts remain reserved and unscored.
4. **Single highest-value next action:** complete result closure and retain an explicit terminal status for every originally forecast runner, linked to the original sealed field and result source. This resolves the observed203 missing joins and seven field ambiguities more directly than another model or selector search. The three manual forecasts additionally need durable completion evidence. No new acquisition or deployment was started.

## Reproduction, validation and deliverables

Canonical evidence root: `/mnt/tenn-nvme2/tenn/greyhound-live-benchmark-20261011-evidence`. `completed/` contains machine-readable strict/diagnostic runner datasets, per-race scores, calibration, date sensitivity, frozen selectors, every pass disposition, result-independent selector evaluations and SHA-256 manifest. `inventory/` contains access boundaries, every journal membership, operational/manual census, failed attempts and source references. `alignment/` contains original authentication, all exclusions and derived corrections. Official-result projections are exact-allowlist, read-only SQLite queries; original records are untouched.

From this checkout, reproduce all metrics and selectors without fits, provider requests or runtime access:

```bash
python3 scripts/run_retained_live_benchmark.py \
  --evidence-root /mnt/tenn-nvme2/tenn/greyhound-live-benchmark-20261011-evidence \
  --output-dir /tmp/greyhound-live-benchmark-replay
```

Reauthenticate original source joins before that numerical replay:

```bash
python3 scripts/live_benchmark_alignment.py \
  --membership /mnt/tenn-nvme2/tenn/greyhound-live-benchmark-20261011-evidence/inventory/residual-membership.json \
  --evidence-root /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525 \
  --results /mnt/tenn-nvme2/tenn/greyhound-live-benchmark-20261011-evidence/official-result-projection.json \
  --manual-census /mnt/tenn-nvme2/tenn/greyhound-live-benchmark-20261011-evidence/inventory/manual-census.json \
  --manual-results /mnt/tenn-nvme2/tenn/greyhound-live-benchmark-20261011-evidence/manual-official-result-projection.json \
  --output /tmp/greyhound-live-alignment-replay
```

The independent checker imports no scoring code and reproduces original membership, source hashes, probabilities, odds, runner/result identities, timestamps, losses, calibration, date bootstrap and selector accounting. [Independent scientific checks](live_benchmark_independent_review_20261011.md) and [Standards review](live_benchmark_standards_review_20261011.md) are separate. Review repaired consistent SQL read snapshots, even-sample timing medians and primary selector conditioning on result-independent forecasts. **54 focused tests pass**, and an independent numerical replay reproduced all16 output hashes. Exact source authentication and synthetic rejection cases are tested; missing evidence is not relabelled a software defect.

The draft contains research utilities, tests, reports and compact evidence references. Full authorised runner/source artifacts remain in the local evidence directory; reserved data are neither decoded nor published. Live collectors, UI, schedules, deployed models and scientific allocations are unchanged. No fitting, promotion, betting or profitability claim occurred.
