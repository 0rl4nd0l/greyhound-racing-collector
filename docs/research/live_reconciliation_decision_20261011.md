# Retained live forecast reconciliation — PR #231 continuation

**One race now supports a full decision-time comparison; model advantage remains inconclusive.** Offline repair recovered the first acquired official response after a false challenge classification and a source-title parser gap. Its original full/half shadow forecasts scored worse than the contemporaneous market. The actual July policy retained the market baseline, so this is no evidence of production improvement. No forecast, model, calibration or selector was changed.

[Acquisition status](live_result_acquisition_status_20261011.md): one request consumed,209 unattempted; the original recorded source hold remains preserved. The completed offline correction does not itself resume collection.

## Reconciliation of the frozen population

| Mutually exclusive current disposition | Races |
|---|---:|
| Original forecasts |210|
| Evaluable without repair |0|
| Recovered and fully evaluable |1|
| Retained evidence incomplete or contradictory |8|
| Requires a specific fresh official-result acquisition |201|
| Restricted within this exact population |0|

The two archive-recovered joins were within the original nine incomplete records. The newly retained response now qualifies one of those nine: Healesville July17 R7. Thus210 =0 without repair +1 recovered and evaluable +8 incomplete +201 requiring acquisition +0 restricted. The original203 missing joins remain reduced to201. All original exclusions, failures, forecasts and allocation boundaries remain retained; zero restricted outcomes were borrowed.

The named July result lane contains 261 accepted observations of nine target races across 65 capture directories. Healesville July17 R7/R8 suffered 81 database rejections because losing runners tied for seventh/fifth. The offline WIN join now accepts valid competition-ranked losing ties while retaining ranks and rejecting winning ties, contradictions and invalid fields. The [qualification clarification](live_win_qualification_clarification_20261011.md) predates the new score inspection; original protocol bytes are unchanged.

A deeper defect prevented qualification of the original nine: native result writing copied runner names from forecast participants and discarded the official source's name map. Matching those names is circular. The new response independently authenticates Healesville R7; the remaining eight still lack sufficient evidence. The other seven also lack explicit terminal dispositions for one to four original runners. A full losing finishing order is unnecessary; independently identified starters and explicit terminal statuses are necessary to distinguish losses from scratches/substitutions. Exact gaps and original job, race, roster, queue, closure and source references are in [the reconciliation report](live_result_reconciliation_20261011.md) and every row of `results/reconciliation.json`.

Thirteen targets have queue metadata; fourteen have any matching archive metadata. The other 196 have no matching metadata in that named lane. This distinguishes absent retained evidence from proven noncapture; it does not claim all 196 were never collected anywhere. A separate identity-only cross-check of earlier named private closure lanes inspected 18 databases and 1,275 metadata files, covering 44 distinct identities and **zero target matches**. It read no protected outcome columns or closure bodies. This is a bounded named-archive investigation, not a whole-host raw-data search.

## Actual rerun

The strict population is **one race, eight runners, July17**, original model artifact `624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d`. The native July policy was `KEEP_MARKET_BASELINE`: its retained baseline is the market comparator, with zero difference by construction. The genuinely sealed full/half forecasts below were shadow variants, not later reconstructions or separate evidence of production use.

| Strict forecast | Log loss | Brier | Top credit |
|---|---:|---:|---:|
| Contemporaneous market / retained policy baseline |1.658763|0.791786|0/1|
| Original full-strength shadow |1.737304|0.825579|0/1|
| Original half-strength shadow |1.697153|0.808562|0/1|
| Full minus market |+0.078541|+0.033792|0|
| Half minus market |+0.038390|+0.016775|0|

All original eight runners remain. The two source-listed unused reserves are explicitly scratched and retained in the evidence; neither was in the forecast field. The source title, exact requested/final URL, independently sourced runner names/boxes, complete terminal field and original forecast chronology pass independent verification. The review accepted the exact native-title identity rule before scoring; canonical/OG markup is not a frozen protocol requirement. Original transport failure and one consumed request are linked through an append-only derived correction, not rewritten as a successful transport receipt.

The forecast lead time was26.53minutes; capture age182.55seconds. Provider quote-publication age remains unknown. Complete original WIN odds and normalized inverse-odds probabilities are retained, with overround1.313203. The single top selection is unchanged and loses; the market assigns the winner more probability. Full-strength runner ECE0.128635 versus market0.131016 is a descriptive eight-runner bin calculation, not evidence of reliable calibration. There is **one date, so uncertainty is not estimable**; dropping it leaves no observations.

The remaining **eight weak diagnostics,64 runners, July17–18** remain separate. Full-strength log loss1.368197 versus market1.374509 gives−0.006312, but Brier0.619843 versus0.614223 worsens by+0.005619. Both top credits are5/8, all top selections unchanged, and winner probability improves in four races and worsens in four. July17's six races have log-loss difference+0.015054; July18's two have−0.070411. Removing either date reverses direction. The frozen date bootstrap [−0.070411,+0.015054] is descriptive only. Their source identity gaps remain unresolved.

The earlier nine-race diagnostic snapshot is preserved separately: full-strength log loss1.409209 versus market1.406093, Brier difference+0.008750. Promoting one race changes the strict/diagnostic partition, not any forecast or outcome value. The new strict race is not selected for favorable performance.

## Unchanged selective prediction

The byte-identical earlier-development plan uses 79 July17–19 forecasts and evaluates 131 July20–22 forecasts. Confidence, minimum same-distance history and maximum `p × odds − 1` thresholds remain frozen. High winning probability, probability reliability and attractive prices are separate questions; the last score is disagreement with price, not demonstrated value. These analyses remain exploratory under the original protocol.

| Selector | Nominal10% |25%|50%|Full|
|---|---:|---:|---:|---:|
| Model confidence |9 (6.9%)|19 (14.5%)|42 (32.1%)|131|
| Relevant history |21 (16.0%)|35 (26.7%)|57 (43.5%)|131|
| Price disagreement |8 (6.1%)|32 (24.4%)|59 (45.0%)|131|

All entries are selected counts out of131. Each has a market-confidence control of the same size and a model-versus-market comparison on exactly its selected identities. **Every selected and control group still has zero matched outcomes; its missing count equals its selected count.** No selected race was replaced. Every pass and development disposition is preserved. Both recovered joins fall in development. Existing result-conditioned exploratory thresholds were not re-estimated from the enlarged nine-race diagnostic.

Outcome availability is concentrated entirely in the earliest two dates; extrapolating complete cases would be unjustified. Enumerating every possible original winning runner gives full-population conditional mean log-loss-difference bounds **[−0.132854,+0.182195]**, conditioning on the one qualified label. If all eight remaining diagnostic mappings prove correct as well, the earlier nine-label bounds remain **[−0.126587,+0.175306]**. Both signs remain possible. These are identification bounds, not confidence intervals, and require eventual proof of unchanged fields. Brier/top bounds and all 24 selected/control bounds are retained and independently reproduced.

## Operational failures and tested repair

There are **431 unique attempted races**, with 1,292 native attempts: 212 appended records on210 races, and1,080 blocked attempts. Thus **221 unique races never produced a forecast**, while425 blocked attempts occurred on already successful races.

| Exclusive unsuccessful-race cause | Races | Classification |
|---|---:|---|
| Equivalent grade spellings rejected |126|Demonstrated historical software defect|
| Hyphenated venue grammar rejected |49|Historical parser limitation; no downstream recovery claimed|
| Unsafe/incomplete metadata |45|Valid exclusion:34 missing mappings only,11 also source decode failures|
| Admission after jump |1|Valid late exclusion|

The offline grade adapter reproduces all369 failing attempts and passes the complete original feature validator with unchanged feature bytes after finite-vocabulary comparison correction. **122 races /316 attempts** also pass retained form, market and chronology checks. This is an input-readiness ceiling, not122 new live forecasts or guaranteed throughput. Tests preserve genuine disagreements, unsafe metadata, ambiguous grades and hash failures, and verify that one rejected race does not erase unrelated dispositions.

Earlier read-only inspection confirmed that the then-installed source `55f728a3` already contained grade canonicalization and the wider venue grammar. These historical findings do not justify redeploying an existing fix. [Failure diagnosis and exact accounting](live_forecast_failure_diagnosis_20261011.md).

Forecast coverage remains **210/431 =48.72%** of recorded attempted races. The one strict race represents0.48% of forecasts and0.23% of attempts. Eight additional weak diagnostics represent3.81% of forecasts; they do not increase strict comparison coverage. Pre-attempt downtime and unrecorded opportunities remain outside this denominator. Missing forecasts are not zero-probability predictions.

## Decision and remaining action

The highest-value next action is **repair the demonstrated false challenge classifier and reconcile its hold through the operational owner**, then use the already approved remaining scope only if actual source controls permit it. Independently sourced runner identities and terminal statuses remain necessary for201 absent joins and eight remaining identity/field gaps, including all131 later selector forecasts. The first response is already recovered; its request remains consumed and must not be repeated. No model search or profitability claim is supported.

The paragraphs below preserve the original request rationale, subsequently approved and activated; they are not current approval blockers.

Prepared request: `acquisition-request.PREPARED.json`, bound to the210 exact identities/official URLs in `results/acquisition-needed.json`. Ceiling: **210 TheDogs public official-result GET attempts, one per identity, no automatic retry or alternate source**, concurrency1 and at least60seconds spacing, accepted-response reuse, raw-response/hash/name/status retention and append-only consumption. Existing holds, retry instructions and current-owner locks remain controlling. The operational owner must admit a safe result-purpose slot; no lock stealing, service change or study-budget reallocation is authorized by this proposal.

Additional result-purpose authority is required because the current standing engineering allocation has `max_result_requests=0`; the older result grant covers October1–January21, not July17–22. The retained verified public-HTML method is documented in the prepared request. No new provider request was made and no historical authority window was reopened. This is the single outstanding acquisition request after completing retained-evidence repairs and independent checks.

## Reproduction and evidence

Current recovered artifacts are under `acquisition/observed-first-completion-01/`: `reviewed-integration/` contains all210 dispositions and correction events; `reviewed-completed/` contains machine-readable strict/diagnostic datasets, scorecards, calibration, date sensitivity and unchanged selector/pass records. `reviewed-independent-check.json` and a separate root reproduction both pass. **139 focused tests pass**. Independent review checks exact membership, identity, chronology, unchanged field, consumption, score calculations and failure cases. The installed projector retains its original hash.

Reproduce the current actual benchmark into a fresh directory:

```bash
bash /mnt/tenn-nvme2/tenn/greyhound-live-reconciliation-20261011-evidence/acquisition/observed-first-completion-01/reproduce-reviewed.sh /tmp/july-reviewed-benchmark
```

The earlier retained-only reproduction and artifacts below remain historical evidence.

Evidence root: `/mnt/tenn-nvme2/tenn/greyhound-live-reconciliation-20261011-evidence`. `artifact-references.json` hashes the deliverables. `results/reconciliation.json` preserves the pre-acquisition210 dispositions; current dispositions are in `acquisition/observed-first-completion-01/reviewed-integration/integration-dispositions.json`; `failures/v2/` contains all431 races/1,292 attempts and derived corrections; `completed/` contains strict/diagnostic runner datasets, per-race scores, calibration, dates, unchanged selections/passes and missing-outcome bounds. Source archives remain local and unchanged.

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
