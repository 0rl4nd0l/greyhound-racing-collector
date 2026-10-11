# Retained July result reconciliation, 11 October 2026

The exact 210 original nonreserved forecasts now have **nine retained official-result joins**, including **two recovered archive joins** that the database projection missed. No forecast yet establishes the complete independent official runner identity needed for the frozen strict comparison. The two recovered joins add usable WIN diagnostics; they do not count as fully evaluable recoveries.

| Original forecasts | Evaluable without repair | Recovered and fully evaluable | Still incomplete | Require acquisition | Restricted |
|---:|---:|---:|---:|---:|---:|
|210|0|0|9|201|0|

The original 203 missing joins become 201 missing joins plus two recovered-but-incomplete records. The original seven incomplete records remain incomplete. These mutually exclusive current counts are separate from the two append-only correction events. Nothing was deleted from the original dispositions or consumption records.

## What was inspected

The exact PR #231 membership/access boundary was read before outcomes. A byte-level structural scanner selects only objects whose race IDs all belong to the 210-race allowlist, then decodes them. It inspected the named July result-capture lane's **65 July 17–18 directories**, ten known result, attempt, candidate and queue filenames per directory, retaining hashes and source offsets/line numbers. These contain **261 accepted race observations for nine unique races**, not 261 independent results. All overlapping runner positions and winners agree across the repeated snapshots. Selection uses the earliest richest native snapshot, independently of forecast performance.

An exact documented-venue-alias census found zero additional result rows in these archives or the result database. No fuzzy match was used. Thirteen forecast identities have retained queue metadata, nine have result-attempt records, and fourteen have any matching archive metadata, including pre-jump deferrals. The other 196 have no matching records in this named result lane. The directory inventory contains no raw response/HTML files or response subdirectories. Exact allowlisted failed-source diagnostics contain no retained source dog-name maps that can repair the nine identity gaps. The census does not claim to have searched every file on the host. Root-level investigation separately checks other named private closure populations without opening their reserved outcomes.

Each reconciliation row binds the original native daemon job directory, original record key, canonical race identity, complete original runner field, append receipt, queue records, attempt states, retained response locations and exact official URL from the pre-race metadata sidecar. The 210 URLs are all present and distinct. No provider request was made, no old window reopened, and no live queue was changed.

## Demonstrated repair

Healesville July 17 races 7 and 8 have all eight official finishing boxes. Their native source positions include losing ties: R7 `1,2,3,4,5,6,7,7`; R8 `1,2,3,4,5,5,7,8`. Both have one unambiguous winning box. The archive reports record **81 database rejections across these two races**, with `duplicate_finish_positions` and `finish_positions_not_contiguous`. Their results were retained in immutable native JSONL despite exclusion from the database evidence table.

The prior research join repeated that ordered-finish assumption. The offline repair accepts competition-ranked losing ties for the WIN target, preserves every original position, and still rejects winning ties, invalid rank gaps, duplicate boxes, mismatched source identities and contradictory snapshots. This is a WIN-specific clarification, not an ordered-finish settlement rule change. Live result ingestion remains unchanged.

A second demonstrated retention defect prevents strict qualification: `scripts/ingest_results_for_date.py:write_result(dry_run=True)` copies result runner names from `candidate.participants` and drops `SourceResult.dog_names_by_box`. Thus apparent name agreement with the forecast is circular. All nine derived result projections explicitly carry `official_runner_identity_observed=false`; benchmark rows carry `result_runner_identity_origin=CANDIDATE_PARTICIPANTS_NOT_SOURCE_VERIFIED`. Even R7/R8's complete box fields cannot become `EXACT_UNCHANGED` from this evidence. Explicit reserve 9/10 `SCR` observations are retained but are not used to invent proof of runner continuity.

The original seven missing-box gaps are:

| Race | Forecast boxes without retained official terminal disposition |
|---|---|
|Healesville July17 R9|2,4,5,7|
|Healesville July17 R11|1,2,3,7|
|Traralgon July17 R9|5|
|Geelong July17 R6|5|
|Bendigo July17 R12|8|
|Ballarat July18 R3|3,4,7,8|
|Ballarat July18 R5|1,2,3,7|

Full finishing order is unnecessary for WIN losses, calibration or accuracy. Independently observed identity and an explicit terminal status for every original runner would suffice, subject to the frozen scratch/substitution rules. The missing fields prevent confirming the same market contract; they do not prevent clearly labelled winner-box diagnostics. No losing runner is retrospectively removed or renormalized.

## Reproduction and remaining action

```bash
python3 scripts/reconcile_retained_live_results.py \
  --benchmark-evidence /mnt/tenn-nvme2/tenn/greyhound-live-benchmark-20261011-evidence \
  --native-root /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525 \
  --output /tmp/greyhound-retained-result-reconciliation
```

Canonical outputs are under `/mnt/tenn-nvme2/tenn/greyhound-live-reconciliation-20261011-evidence/results/`: `reconciliation.json`, `archive-result-projection.json`, `correction-trail.json`, `acquisition-needed.json`, both allowlisted native projections, and `archive-references.json`. The root orchestrator reruns authentication and the original frozen benchmark/selectors against this derived projection; the reconciliation utility does not fit or score models.

The exact smallest remaining result action is one official result response per unresolved identity, retaining source runner names and terminal statuses: 201 missing results plus nine identity/field completions. The prepared manifest deduplicates all 210 official identities and URLs. Any acquisition must use current source authorization and the root owner; this script does not fetch.

Tests first reproduced two failures: nonwinning ties rejected and circular names incorrectly qualifying a complete box field. Both now pass, alongside reserved-object decoding, nested scope, invalid positions and duplicate boxes. Fifteen focused reconciliation/alignment tests pass. Independent review confirmed that the source-name gap must remain disqualifying.

Independent review caught duplicate derived correction events caused by tuples becoming JSON arrays on replay. Canonical event identity now makes reruns idempotent. The four-entry intermediate trail is preserved byte-for-byte with its SHA-256 in a separate history artifact; `correction-trail-audit.json` identifies the two duplicate append entries, and the current trail has two unique events. Two regression tests cover idempotence and preserved duplicate history.
