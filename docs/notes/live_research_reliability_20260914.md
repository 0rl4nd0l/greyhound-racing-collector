# Live research reliability: 14 September 2026

This is a read-only runtime reconciliation and bounded source repair. No service,
timer, canonical database, model, frozen cohort, consumed attempt or retained
prediction was changed. No live prediction, capture, result fetch or research rescore was run.
Tests used disposable fixtures. Artifact verification read existing bytes only.

## Runtime truth

Observed between 16:37 and 17:00 Australia/Melbourne. Remote master remains
`fb2c545f0256829952e31e2cd65365212dc37e2e`. GitHub confirms #175 merged as
`a2249ca1` and #174 as `fb2c545f`. The main checkout remains on
`codex-x/august-official-result-writer-review`, HEAD `b5965fcf`, with four tracked
modifications and substantial untracked research. It was preserved. This repair
uses an isolated sibling based on the verified master.

Installed user units and `/proc` agree:

| Entry point | Actual source and configuration |
| --- | --- |
| Operator UI, PID 1133, localhost:5055 | `greyhound-runtime-master-live-20260825-a2249ca1`; frozen JSON `market_form_residual_v1`, `manual-default`, receipt-only; Python 3.11.15 / NumPy 1.26.4 |
| Full collector and odds-only collector | `greyhound-runtime-master-live-20260825-9f5c2409`; shared collector lock, canonical DB and retained evidence root |
| Full collector shadow scoring | Explicit June `shadow_randomforest_model.joblib`, SHA-256 `d7e9ff35b383a0e6400bcb67bcf6df374e4c0bfe6c974f32d1c9f057876e471d`; distinct from the R3 residual model |
| `predict_race_now.py` CLI | Defaults to `latest-research` / `manual-default`, but defaults to **auto** odds; receipt-only research must explicitly select `--odds-source receipt` |
| August private journal | Retained protocol and eleven provenance records; no recurring journal process or installed journal unit was found |

Both deployed source trees have no tracked modifications. The UI includes #175
but lacks #174 finalisation/restart recovery. The collector includes #172 alias
repair. Neither deployed tree contains this repair. The UI's legacy Flask
prediction routes are a separate fallback-based interface, not evidence of a
frozen R3 execution; their actual selected model was not exercised or certified.

The residual model SHA-256 is
`624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d`;
the unchanged config SHA-256 is
`f8a3c321dca12321a38a4d12a08f4f43461e1c1e73100eda871fd60252ed1820`.

## Operational stages and denominators

Evidence root:
`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525`.

The index is `shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json`.
At 16:43:13, packet `efb0e6a73fc4e6c84cf3506a506f2032dfa2ce1dbd8381970931571f4b40432e`
contained six races and was 1,636.68 seconds old. Its provenance verified;
receipt-only inspection found two READY, two missing receipts and two with
insufficient pre-jump margin. These six are index rows, not recorded journal
opportunities or attempted predictions. At 16:59:57, natural cycle
`20260914T164308+1000` had four index races, age 828.18 seconds: the actual
1,200-second predictor index gate passed and all four receipt preflights passed.
Neither observation submitted a job.

There were 17,515 collector-exact receipt JSON files at the first receipt scan,
with fresh September 14 writes. These are alias/window publications, **not**
17,515 independent races. One completed odds packet at 16:32 reported fourteen
appended odds rows, zero blocked attempts and window counts 8 captured / 3
missed / 5 pending, denominator sixteen windows in that packet only.

Authoritative journal:
`forward_prediction_journal_v1_private_20260825`, plus its referenced imported
canary bundle under `single_prediction_canary_private_20260825T1540`.

| Category | Count and meaning |
| --- | --- |
| Opportunities | 11 distinct retained race-provenance records; not all races collected since August |
| Attempts | 11 recorded prediction invocations, including the imported canary |
| Successful prediction bundles | 6: imported canary plus entries 2, 3, 5, 6, 7 |
| Permanent attempt failures | 5: entry 4 `RESIDUAL_SCORER_FAILED`; entries 8–11 `RECEIPT_UNAVAILABLE` |
| Scored research records | 1 existing `CANARY_RESULT_SCORED` evaluation; no new metric was computed |
| Successful predictions without journal closure | 5; August races awaiting closure work, not future races awaiting jump |

All eleven bundle indexes/manifests were checked with the canonical verifier.
Before this repair, the six successful bundles failed
`PREDICTION_BUNDLE_IDENTITY_MISMATCH`, field
`protocol.collector_exact_receipt.handoff`; the five blocked bundles verified.
With the candidate verifier, six READY and five BLOCKED bundles verify, without
rewriting them. The existing result record's canonical official-evidence hash
matches `07051346c26b1e277cbd70c6861ab44669a6f79eedc7e9d039544596bb64d361`.
This authenticates recorded evidence; it does not rescore or certify a new live
job. The five missing closure records do not establish that official results
are unavailable: a canonical-DB existence check was skipped when an active WAL
was detected.

The installed R3 operations store has a separate denominator: nine jobs and
nine claimed attempts, all terminal `PROCESS_OUTPUT_INVALID`, latest August 19.
The journal's job IDs are not nine additional successful UI jobs. The scheduled
forward-corpus observer is another population: its latest inspected report had
two closed examples and one `RESULT_PENDING`, rejected for inconsistent official
finish/status data. None of these counts should be added together.

The retained August 16 inventory was read as a historical cross-store reference,
not represented as current coverage. No broad inventory/scoring refresh was run.
Betfair confirmation still has an outcome-free scheduled-off source blocker in
its retained status, zero frozen population and no active collector found.
The October successor is separate, `PREPARED_NOT_AUTHORIZED`, and cannot activate
before October 1. Their outcome stores and interim metrics were not inspected.

## Cause and bounded repair

The earliest reproducible integration failure with usable source evidence is
the scheduled receipt handoff. The collector retains the source URL ending in
`?trial=false`. The predictor validates and stores a canonical race URL without
that query. Admission and sealed verification compared these representations as
raw strings. A scheduled-source integration fixture with the exact live query
failed `RECEIPT_INVALID`; the same fixture without the query passed.

Receipt preflight and sealed verification now compare precisely the canonical
URL and its `?trial=false` spelling. They still validate through the existing
strict TheDogs parser and retain exact source bytes/hashes. `trial=true`, extra
query parameters and a different race slug are not accepted as equivalents.
Race/date/venue/runner/WIN/timestamp and provenance checks remain in force.

An adjacent admission defect could consume attempts on expired indices:
`return_verified_view=True` intentionally authenticates stale packets for UI
display, but UI admission and worker revalidation omitted their own age gate.
HTTP integration reproduced a 1,201-second index receiving 202 and a job ID.
Worker integration reproduced an expired index consuming an attempt, including
expiry during receipt validation. Admission now checks the frozen 1,200-second
limit before creating a job and after receipt validation; the worker checks
again before claiming, using the clock after the final index read completes.
Independent review identified that last read-time boundary; a regression first
reproduced an attempt at age 1,200.2 seconds after a read began at 1,199.8 seconds.
Exactly 1,200 seconds remains admissible. No freshness
configuration or experiment rule was changed.

Scheduling remains a separate availability constraint. The full timer waits
fifteen minutes after completion; its 16:13 cycle took about fourteen minutes,
including a 115-second lock wait. Later odds ticks correctly skipped a held
full-collector lock. Odds-only refreshes do not publish the operational index.
These facts explain gaps but do not authorize a faster schedule or lock bypass.
Host inspection found about 23 GiB available RAM and 632 GiB free on the data
volume; initial load was 3.16. No evidence established memory exhaustion as the
primary cause. Five successful journal invocations took 2.72–2.81 seconds; the
imported canary took 27.78 seconds. Those historical timings are not a live SLA.

## Validation and remaining transition

Regression tests reproduce the old failures and check rejection before job or
attempt allocation, expiry during preflight, the exact age boundary, and full
scheduled-receipt prediction-to-verifier integration. Negative URL tests cover
trial/result queries, different race paths and rehashed contradictory receipts.
Final focused command, using the installed Python 3.11.15 environment:

```bash
python -m pytest -q -o addopts= --no-cov \
  tests/operator_ui/test_bootstrap.py \
  tests/operator_ui/test_prediction_worker.py \
  tests/test_predict_race_now.py \
  tests/test_forward_prediction_journal.py --tb=short
```

Result: **197 passed, 1 failed**, 181.08 seconds, with a dedicated disposable
`--basetemp`. Every new receipt and index-boundary regression passed. The one
failure is the existing stalled-pipe cleanup test's three-second wall-clock
limit: 3.206 seconds in the suite, 3.162 seconds isolated. Loading the exact
`fb2c545f` worker module into memory and running the same unchanged test also
failed at 3.160 seconds. This establishes a baseline-reproduced timing failure,
not a green suite; its bound was not relaxed.

Earlier baseline coverage passed 371 tests. A broader candidate run reached
419 passes and two failures before interruption after 17 minutes in unchanged
residual-model code. Its journal receipt failure did not recur in the final
sequential suite; the cleanup timing failure did. Full portability completion
is **not** claimed. No deployment gate is waived by these results.

## Standards

Independent standards review found no material violations.

## Spec

Independent spec review identified the final-read expiry gap described above;
the completion-clock repair and regression address it. Deployment and complete
future records remain outside the verified result.

Initial findings: Standards 0; Spec 1 (pre-attempt final-read freshness).

## Remaining operational transition

This is source implementation and offline verification only. No merge,
deployment, fresh live prediction or new result closure occurred. Deployment
must include current master's #174 as well as this repair. Use a clean accepted
master descendant, the unchanged frozen model/config and existing runtime
Python, then generate the R3 package through `src.operator_ui.deployment` with
a new contemporaneous live-authority observation. Follow ADRs 0009–0016: fixture
and portability replay, deployment-generator/R3 safety tests, review all four
package hashes and `systemd-analyze verify` before installation.

The operational transition is a reviewed R3 deployment followed by a bounded,
future-only receipt-backed acceptance through official-result closure. It needs
specific runtime authorization; this report does not authorize activation of
the private journal or a new recurring experiment. Preserve the installed
`a2249ca1` unit/package as rollback evidence; on a failed acceptance, disable
prediction admission through the generated feature gate and retain every new
job, attempt and bundle. Do not retry a consumed race. Unattended accumulation
still requires an explicitly activated owner of both admission and closure;
deploying a healthy web service alone does not create that stream.
