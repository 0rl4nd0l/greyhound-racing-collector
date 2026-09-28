# Inactive comparison handoff to collector and scientific owners

Research PR 194 includes the collector through
`deada7607e6aafd5be85c3cb6d910922c0a2e26e` (allocation repair
`23567cbec57f10472b093cfe0caf743c5a3ad63f`), preserving the September28 timer,
handoff, shutdown, receipt, accounting and R 3 changes. The exported comparison
proof pins integration `557b7ecb1350e3b85f4f0d8199351c50bcfbaf2c`.
Subsequent closure/preparation changes are recorded in the final integration
commit/report. **No deployment, activation, live provider call or service change
was performed by this track. Operational rollout is independent.**

Frozen files are `artifacts/research_comparison/frozen_20260924/{registry,
residual_box,residual_half}.json`; they and all production artifacts are unchanged.

| Identity | SHA-256 |
| --- | --- |
| Registry | `938f433133057591e4ebd17a1ceef4fb25da252b5e3d6aba903194eb54b96ca9` |
| Residual plus box | `51639ddada362b1a14110461ef258dbff852eb7b2797ea074cd85f22381a4d32` |
| Half-strength residual | `e827e8f29c756c8995de08e15f8a6c25ca92f0709693cb55ad6bbeb9e16716e0` |
| Production model | `624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d` |
| Production config | `f8a3c321dca12321a38a4d12a08f4f43461e1c1e73100eda871fd60252ed1820` |

The fixed candidates were fitted previously on 331 races/2,360 runners, June 10–
July 8; serialized coefficients, preprocessing and membership hashes are in the
registry. No refit occurred here. Candidates retain raw-card history; production
retains merged history. Do not substitute production feature rows for candidates.

## Ready now and still required

- Default-off four-way predictions via the existing retained worker/subprocess.
- Three exported off/on trials with kernel networking denied; exact replay,
  identical production probabilities, common field and market snapshot.
- Fixed real-input census:12/12 pass after a reviewed restricted projection;
  original production feature hashes reproduced, candidates execute.
- Optional authenticated result nomination through the **existing** result
  collector. Explicit machine-result authority required; ordinary operational
  jobs stay excluded. Synthetic nomination, ingestion and closure tests pass.
- Prepared allocation, bounded missing-result analysis and one-shot closure
  evaluator. No actual future-race performance evidence.

The scientific owner must approve the [specification](future_comparison_20260928_spec.md),
defer both inactive overlapping proposals, grant prospective earlier-history
scope, and appoint/authorize a result-retention owner. The operational owner has
not accepted that duty and its current campaign does not include results.
Before activation that result owner must demonstrate prompt acquisition/storage
under its finite provider budget and the existing source/lock rules. No research
code can supply provider authority or guarantee future availability.

## Prepared configuration; do not execute live under this task

The [prepared files](future_comparison_20260928_evidence/activation/plan.prepared.json)
are rejected by runtime. To refresh the unactivated allocation after readiness,
use the preparation runner; it writes review files only:

```bash
PYTHONPATH=. "$R3_PY" -B -m scripts.prepare_frozen_comparison_activation \
  --starts-at '2026-10-01T12:00:00+10:00' \
  --programme-root /home/l4nd0/greyhound-four-way-prospective-20261001 \
  --prediction-output-root /home/l4nd0/greyhound-collector-campaign-20260923/operational-predictions/bundles \
  --reservation-sha256 50c2ccd5fe13e2ae4c2102812ef76e81ea42de474acff4766597958b13ba19d9 \
  --output /absolute/new-review-directory
```

If approval is later, choose a new future start before activation. After the
single scientific approval, create a **new immutable** approved plan with its
actual activation timestamp before start, approval/allocation/history references,
refreshed reservation-review hash, and status `AUTHORIZED`. Freeze exact root,
112-day endpoint, artifact registry path/hash and source commit before collection.
Do not edit or reuse September 24/September 28 prepared files as evidence of approval.

Existing campaign preparation gets its usual owner-approved arguments plus:

```text
--campaign-root /home/l4nd0/greyhound-collector-campaign-20260923
--operational-predictions
--comparison-plan /absolute/approved-plan.json
--observation-minutes 90
--start-after-minutes <owner-selected-future-delay>
```

Keep the existing campaign/source ledger and cumulative attempts; do not reset
budgets, clear STOP, create a second collector, or transplant a research plan
into an in-flight session. Campaign output must match the plan's bound prediction
root. The research programme root is separate; past bundles in the shared
operational root cannot qualify without future, pre-outcome admission.
Existing preparation validates and passes the plan SHA to the real worker.
Rollback is to omit `--comparison-plan` from the **next normally prepared**
operational session, after the owner's normal shutdown. No production pointer
needs rollback. Preserve all seals, missing attempts and result queues; rollback
does not erase scientific membership or stop already owed result closure.

The result owner creates its restricted authority file with status
`AUTHORIZED_MACHINE_RESULT_RETENTION`, exact approved plan SHA, actual
`issued_at` at/before activation, named `owner`, approval and
`source_budget_reference`, bound absolute `result_database`, and `human_outcome_access:false`. A binding JSON names
absolute approved plan/authority paths and their exact SHA-256 values. Prepared
versions have null bindings and cannot run.

After approval only, use the existing result collector, with a dedicated private
result-evidence DB initialized by its owner (an empty SQLite file suffices; the
collector creates its own append-only evidence tables). Never bind the historical
training/production DB or reuse a mixed result source for this study:

```bash
umask 077
PYTHONPATH=. "$R3_PY" -B -m scripts.autonomous_official_result_capture \
  --r3-job-store /home/l4nd0/greyhound-collector-campaign-20260923/operational-predictions/jobs.sqlite3 \
  --r3-prediction-bundles /home/l4nd0/greyhound-collector-campaign-20260923/operational-predictions/bundles \
  --comparison-result-binding /absolute/approved-result-binding.json \
  --db /absolute/private-study/official-results.sqlite3 \
  --evidence-root /absolute/private-study \
  --output-dir /absolute/private-study/new-result-cycle \
  --execute-db-ingest --require-lock-free --lock-path "$EXISTING_COLLECTOR_LOCK" \
  > /absolute/private-study/new-result-cycle.log 2>&1
```

This command acquires results; even acquisition called “dry run” is **not** an
offline test. No instance was run on real data here. It must run under the result
owner's approved source account/budget and normal collector coordination. No
new timer/service is prepared by research. Full outputs/logs are machine-only;
release structural closure counts without winner identities or scores.

At closure the appointed owner uses the supplied byte-copy sealer after its
writer is quiescent (this command itself performs no result decoding):

```bash
PYTHONPATH=. "$R3_PY" -B -m scripts.seal_comparison_result_closure \
  --binding /absolute/approved-result-binding.json \
  --output /absolute/private-study/new-closure
```

It requires the fixed closure deadline and the approved machine-retention
binding, preserves the original DB, writes a read-only snapshot and a closure
hash receipt, and creates no evaluation permission.

At closure the appointed owner hashes an immutable result DB snapshot. The
later one-shot evaluation authority must bind that SHA, the plan SHA, exact
`closure_cutoff`, status `AUTHORIZED_ONE_SHOT_OUTCOMES` and approval reference.
The original evaluator invocation remains valid with this stronger v 2 receipt;
it refuses early, hashes the snapshot before/after joins, admits no late-captured
result, and preserves membership/closure failures. No interim performance runs.

## Reproduction and measured cost

Pinned interpreter:
`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-operator-ui-r3-python311-20260803-9d58f340/bin/python`.
Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.`.
Run `nice -n 10 "$R3_PY" -B -m scripts.verify_frozen_comparison_package
--output /absolute/new-proof --python "$R3_PY" --repetitions 3`.

Measured comparison work:29–32 ms CPU,40–44 ms elapsed. Whole-subprocess off/on
median difference 112 ms elapsed,51 ms CPU, about 3.0 MiB peak RSS; the elapsed range
was−11 to 115 ms across three trials, demonstrating scheduling noise. These are
synthetic packaged overhead measurements, not a live deadline guarantee.
Real candidate feature construction took 2.0–4.7 ms; restricted production replay
about 0.60–0.63 s, offline diagnostic cost rather than incremental live work.

Keep `docs/research/future_comparison_20260928_evidence/` with every sampled
disposition, projection repair attempt, reservation snapshot, planning scenario
and packaged identity. No unsuccessful feasibility attempt has been removed.
