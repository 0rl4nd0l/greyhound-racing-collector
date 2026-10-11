# Retained live forecast reconciliation — PR #231 continuation

**Model advantage remains inconclusive.** Retained archives recover two result joins, but reveal a source-name retention defect affecting all nine joined races. None yet proves the complete decision-time comparison. The nine-race winning-box diagnostic is slightly worse than the market on both proper losses. No model or selector was fitted or changed.

## Reconciliation of the frozen population

| Mutually exclusive current disposition | Races |
|---|---:|
| Original forecasts |210|
| Evaluable without repair |0|
| Recovered and fully evaluable |0|
| Retained evidence incomplete or contradictory |9|
| Requires a specific fresh official-result acquisition |201|
| Restricted within this exact population |0|

The two newly recovered joins are **within the nine incomplete records**, not an additional population. The original 203 missing joins become 201. All original exclusions, failures, forecasts and allocation boundaries remain retained; zero restricted outcomes were borrowed.

The named July result lane contains 261 accepted observations of nine target races across 65 capture directories. Healesville July17 R7/R8 suffered 81 database rejections because losing runners tied for seventh/fifth. The offline WIN join now accepts valid competition-ranked losing ties while retaining ranks and rejecting winning ties, contradictions and invalid fields. The [qualification clarification](live_win_qualification_clarification_20261011.md) predates the new score inspection; original protocol bytes are unchanged.

A deeper defect prevents qualification: native result writing copied runner names from forecast participants and discarded the official source's name map. Matching those names is circular. Even the two complete eight-box results cannot independently establish runner continuity. The other seven also lack explicit terminal dispositions for one to four original runners. A full losing finishing order is unnecessary; independently identified starters and explicit terminal statuses are necessary to distinguish losses from scratches/substitutions. Exact gaps and original job, race, roster, queue, closure and source references are in [the reconciliation report](live_result_reconciliation_20261011.md) and every row of `results/reconciliation.json`.

Thirteen targets have queue metadata; fourteen have any matching archive metadata. The other 196 have no matching metadata in that named lane. This distinguishes absent retained evidence from proven noncapture; it does not claim all 196 were never collected anywhere. A separate identity-only cross-check of earlier named private closure lanes inspected 18 databases and 1,275 metadata files, covering 44 distinct identities and **zero target matches**. It read no protected outcome columns or closure bodies. This is a bounded named-archive investigation, not a whole-host raw-data search.

## Actual rerun

Strict scorecard: **zero races**, so no strict loss, calibration or accuracy estimate. Diagnostic scorecard: **nine races, 72 runner appearances, July17–18**, original artifact `624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d`. These are original July shadow forecasts of the installed artifact; the July operating policy retained the market baseline. They are not reconstructed predictions or proof of production use at those cutoffs.

The diagnostic maps an official winning box to the original forecast roster, conditionally assuming that identity is correct. Every original runner and probability remains present. Market probabilities are normalized inverse contemporaneous WIN odds; original odds and overround remain in the runner/per-race artifacts.

| Diagnostic forecast | Log loss | Brier | Top credit |
|---|---:|---:|---:|
| Contemporaneous market |1.406093|0.633952|5/9|
| Original full strength |1.409209|0.642702|5/9|
| Original half strength |1.406150|0.637803|5/9|
| Full minus market |+0.003116|+0.008750|0|

Full-strength winner probability improves in four races and worsens in five. All nine top selections are unchanged, including four losses. July17's seven races have paired log-loss difference **+0.024124**; July18's two have **−0.070411**. Removing either date reverses the sign. The frozen 10,000-draw date bootstrap gives **[−0.070411,+0.024124]**; two dates and repeated dogs make this descriptive, not credible confirmation of generalization.

Ten fixed-bin runner ECE is 0.063824 for full strength versus 0.077235 for market. These sparse bins cannot establish reliability or outweigh the proper-loss comparison. Lead time is 20.73–54.42 minutes, median41.95; capture-start age is 144.39–353.71 seconds, median182.55. Provider quote-publication age remains unknown in all nine. Capture age is not quote age. No calibration fitting occurred.

## Unchanged selective prediction

The byte-identical earlier-development plan uses 79 July17–19 forecasts and evaluates 131 July20–22 forecasts. Confidence, minimum same-distance history and maximum `p × odds − 1` thresholds remain frozen. High winning probability, probability reliability and attractive prices are separate questions; the last score is disagreement with price, not demonstrated value. These analyses remain exploratory under the original protocol.

| Selector | Nominal10% |25%|50%|Full|
|---|---:|---:|---:|---:|
| Model confidence |9 (6.9%)|19 (14.5%)|42 (32.1%)|131|
| Relevant history |21 (16.0%)|35 (26.7%)|57 (43.5%)|131|
| Price disagreement |8 (6.1%)|32 (24.4%)|59 (45.0%)|131|

All entries are selected counts out of131. Each has a market-confidence control of the same size and a model-versus-market comparison on exactly its selected identities. **Every selected and control group still has zero matched outcomes; its missing count equals its selected count.** No selected race was replaced. Every pass and development disposition is preserved. Both recovered joins fall in development. Existing result-conditioned exploratory thresholds were not re-estimated from the enlarged nine-race diagnostic.

Outcome availability is concentrated entirely in the earliest two dates; extrapolating complete cases would be unjustified. Enumerating every possible original winning runner gives full-population conditional mean log-loss-difference bounds **[−0.133615,+0.182852]** with no qualified labels. If all nine diagnostic mappings prove correct, bounds are **[−0.126587,+0.175306]**. Both signs remain possible. These are identification bounds, not confidence intervals, and require eventual proof of unchanged fields. Brier/top bounds and all 24 selected/control bounds are retained and independently reproduced.

## Operational failures and tested repair

There are **431 unique attempted races**, with 1,292 native attempts: 212 appended records on210 races, and1,080 blocked attempts. Thus **221 unique races never produced a forecast**, while425 blocked attempts occurred on already successful races.

| Exclusive unsuccessful-race cause | Races | Classification |
|---|---:|---|
| Equivalent grade spellings rejected |126|Demonstrated historical software defect|
| Hyphenated venue grammar rejected |49|Historical parser limitation; no downstream recovery claimed|
| Unsafe/incomplete metadata |45|Valid exclusion:34 missing mappings only,11 also source decode failures|
| Admission after jump |1|Valid late exclusion|

The offline grade adapter reproduces all369 failing attempts and passes the complete original feature validator with unchanged feature bytes after finite-vocabulary comparison correction. **122 races /316 attempts** also pass retained form, market and chronology checks. This is an input-readiness ceiling, not122 new live forecasts or guaranteed throughput. Tests preserve genuine disagreements, unsafe metadata, ambiguous grades and hash failures, and verify that one rejected race does not erase unrelated dispositions.

Read-only inspection confirms the installed source at `55f728a3` already contains grade canonicalization and the wider venue grammar. These historical findings do not justify redeploying an existing fix. [Failure diagnosis and exact accounting](live_forecast_failure_diagnosis_20261011.md).

Forecast coverage remains **210/431 =48.72%** of recorded attempted races. The nine diagnostics represent4.29% of forecasts and2.09% of attempts; strict comparison coverage is0%. Pre-attempt downtime and unrecorded opportunities remain outside this denominator. Missing forecasts are not zero-probability predictions.

## Decision and remaining action

The highest-value next action is **one scoped official-result closure pass for these exact210 races**, retaining independently sourced runner identities and terminal statuses. This can resolve201 absent joins and nine identity/field gaps, including all131 frozen later selector forecasts; success is not guaranteed. No model search, deployment or profitability claim is supported.

Prepared request: `acquisition-request.PREPARED.json`, bound to the210 exact identities/official URLs in `results/acquisition-needed.json`. Ceiling: **210 TheDogs public official-result GET attempts, one per identity, no automatic retry or alternate source**, concurrency1 and at least60seconds spacing, accepted-response reuse, raw-response/hash/name/status retention and append-only consumption. Existing holds, retry instructions and current-owner locks remain controlling. The operational owner must admit a safe result-purpose slot; no lock stealing, service change or study-budget reallocation is authorized by this proposal.

Additional result-purpose authority is required because the current standing engineering allocation has `max_result_requests=0`; the older result grant covers October1–January21, not July17–22. The retained verified public-HTML method is documented in the prepared request. No new provider request was made and no historical authority window was reopened. This is the single outstanding acquisition request after completing retained-evidence repairs and independent checks.

## Reproduction and evidence

Evidence root: `/mnt/tenn-nvme2/tenn/greyhound-live-reconciliation-20261011-evidence`. `artifact-references.json` hashes the deliverables. `results/reconciliation.json` contains all210 current and original dispositions; `failures/v2/` contains all431 races/1,292 attempts and derived corrections; `completed/` contains strict/diagnostic runner datasets, per-race scores, calibration, dates, unchanged selections/passes and missing-outcome bounds. Source archives remain local and unchanged.

```bash
python3 scripts/rerun_frozen_live_reconciliation.py \
  --original-evidence /mnt/tenn-nvme2/tenn/greyhound-live-benchmark-20261011-evidence \
  --reconciled-records /mnt/tenn-nvme2/tenn/greyhound-live-reconciliation-20261011-evidence/qualified \
  --output-dir /tmp/greyhound-reconciliation-metrics
python3 scripts/check_live_reconciliation.py \
  --baseline-root /mnt/tenn-nvme2/tenn/greyhound-live-benchmark-20261011-evidence \
  --completed /tmp/greyhound-reconciliation-metrics \
  --failure-root /mnt/tenn-nvme2/tenn/greyhound-live-reconciliation-20261011-evidence/failures/v2 \
  --result-root /mnt/tenn-nvme2/tenn/greyhound-live-reconciliation-20261011-evidence/results \
  --output /tmp/greyhound-reconciliation-check.json
```

Native archive replay commands are in the result/failure reports. The closure audit is reproducible with `audit_live_result_closure_metadata.py --base /mnt/tenn-nvme2/tenn --population <evidence>/population.FROZEN.json --output <new-file>`.

**77 focused tests and four subtests pass.** [Independent review](live_reconciliation_independent_review_20261011.md) authenticates membership, chronology, exact source rows, runner alignment, scores, calibration, date intervals, selector hashes, all conditional bounds and failure accounting. Separate integration review reproduces the private-closure intersection and tests frozen-input hash/uniqueness rejection. An intermediate duplicate correction append was detected and fixed idempotently; its exact four-entry history remains hashed, with two unique correction events and an explicit duplicate audit. Live operations, frozen studies, all consumption records and original forecasts remain unchanged.
