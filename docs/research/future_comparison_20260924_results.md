# Frozen future comparison — executed preparation

**Runnable, default off, not activated. No new predictive advantage is claimed.**
The existing retained worker now seals market, unchanged production, residual
plus box and half-strength residual forecasts from one verified WIN snapshot.
Three exported-package synthetic comparisons succeeded with networking denied;
all four replayed, and production predictions were identical with comparison on
and off. **Actual future-race evidence: zero.**

The tested integration commit is
`9b3b38fdf231c7d8ec5dd7e2445844c769e43124`. It merges the collector owner's
committed `a15f5882` receipt-snapshot repair into the isolated research branch.
No provider calls, services, production configuration, original research files
or other worktrees were modified. The [handoff](future_comparison_20260924_handoff.md)
gives exact invocation; the [execution specification](future_comparison_20260924_spec.md)
and [prepared plan](future_comparison_evidence/prepared_plan.json) define the
single proposed, unallocated future evaluation.

## Frozen artifacts and reproduction

Registry SHA-256:
`938f433133057591e4ebd17a1ceef4fb25da252b5e3d6aba903194eb54b96ca9`.

| Method | Exact model artifact SHA-256 |
| --- | --- |
| residual plus box | `51639ddada362b1a14110461ef258dbff852eb7b2797ea074cd85f22381a4d32` |
| half-strength residual | `e827e8f29c756c8995de08e15f8a6c25ca92f0709693cb55ad6bbeb9e16716e0` |
| unchanged production | `624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d` |

Production configuration remains
`f8a3c321dca12321a38a4d12a08f4f43461e1c1e73100eda871fd60252ed1820`.
The [candidate registry](../../artifacts/research_comparison/frozen_20260924/registry.json)
pins training identities, preprocessing, feature definitions, coefficients,
source hashes, model configuration and environment. Experimental artifacts are
separate from the frozen production directory.

Exactly the two #193 recipes were fitted on **331 eligible races / 2,360 runners,
June 10–July 8, 2026**. All 2,360 feature vectors were reconstructed from their
qualified original raw cards with exact parity. Preprocessing uses fitted
medians, missing indicators, scaling and race centering; residual cap 0.35,
L2=1. Box adds literal verified box to the fixed 16 features at strength 1;
half-strength uses the same 16-feature fit at inference strength 0.5.
Research/standalone inference agree within 2.3e-16. Production's merged DB/card
feature route is retained; candidate raw-card semantics are explicit.

Initial fitting took 1.036 seconds CPU and elapsed; a source-provenance replay
following removal of unused imports/trailing whitespace took 1.009 seconds,
with **identical fitted coefficients and preprocessing**. Initial artifacts
remain in the isolated output root. Four total fits means the same two fits
twice, not four candidate variants. No candidate was selected from future data.
The fit interpreter is Python 3.11.15, NumPy 1.26.4, SciPy 1.16.1, scikit-learn
1.7.1; peak process RSS about 401 MiB including source reconstruction/imports.

Reproduce the fit with the pinned research interpreter and a new output path:

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/.venv/bin/python \
  -B -m scripts.freeze_future_comparison \
  --prepared /home/l4nd0/greyhound-offline-systematic-output-20260924/foundation \
  --out /absolute/new-candidate-freeze
```

The runner requires the exact development hash, protected manifest, original
source hashes and 331/2,360 population. It performs parity checks, not a new
backtest/search. Registry creation timestamps and measured runtime change on
reproduction; frozen model bytes reproduce for identical pinned sources/env.

## Actual packaged execution

[Package identity](future_comparison_evidence/package_identity.json) records
all exported source file hashes, committed source revision, interpreter hash
and installed environment. Archive SHA-256:
`841e0a5835cc46c67e1700270cd7e1f48261831c4cdff52332eaa7750f717d67`.
Runtime Python 3.11.15 uses the collector owner's pinned R3 interpreter (NumPy
1.26.4, SciPy 1.17.1, scikit-learn 1.9.0). Candidate inference performs no fitting.

The proof exports committed code, fabricates source evidence, runs the real
current-index/receipt publication and retention, deletes its private original
history DB, then executes **real WorkerConfig fixed argv → OS subprocess →
predict_race_now → retained feature generation → frozen scorer → existing bundle
verifier**. Kernel seccomp denies IPv4/IPv6 sockets and is inherited by children.
No provider transport is substituted into an actual request. The separate
comparison verifier re-extracts candidate features and replays all four outputs.
No source/feature/scorer mocks stand in for this path. Tests use three invented
runners, so this is not a broad live workload benchmark.

| Pair | Default-off subprocess | Comparison subprocess | Difference | Sampled peak RSS increment |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.6708 s | 1.6807 s | +0.0099 s | 3.20 MiB |
| 2 | 1.6224 s | 1.6783 s | +0.0560 s | 3.81 MiB |
| 3 | 1.6342 s | 1.6655 s | +0.0313 s | 8.24 MiB |

Execution order alternated on/off. Median instrumented comparison-only work was
**0.02794 s CPU / 0.04124 s elapsed**. Whole-subprocess CPU differences ranged
−0.0062 to +0.0334 seconds (startup/system noise); three pairs cannot estimate a
stable marginal cost precisely. Peak RSS was sampled through `/proc` every 10 ms
and may miss brief peaks. The score-phase high-water increment was zero, which
does not mean zero allocations. New-index denominator recording separately took
0.00783 s CPU / 0.00884 s elapsed; an unchanged-index check took 0.000077 s.
See [trial records](future_comparison_evidence/executed_trials.json) and
[schedule timing](future_comparison_evidence/schedule_overhead.json).

**187 focused tests passed in 43.82 seconds** after the collector repair merge;
two additional endpoint/outcome-authority guard tests subsequently passed.
These cover the new comparison, production CLI, retained consumer and worker
lifetime/default-off behavior. Negative checks prove late comparison rejection,
changed/incomplete retained input rejection, field/as-of mismatch rejection,
candidate hash failure without substitution, preflight-failure preservation,
consumed attempts, unchanged prior seals, bundle tamper rejection, protected
DATE projection before outcome-valued card decoding, unapproved-plan rejection,
endpoint/authority gating and fixed synthetic metrics. The real future terminal
result join has not been executed; only its metric and authority gates have
been tested, using no actual target outcomes.

All work ran sequentially, single numerical-library thread, nice 10. Host load
was inspected (roughly 0.5–1.3 during this track, about 27 GiB initially available).
The owner requested no heavy work during live observation; the only later fit
was a one-second exact provenance replay. No large sweep or parallel fitting ran.

## Preserved failures and historical exposure

The [execution ledger](future_comparison_evidence/execution_ledger.jsonl) preserves
initial fixture/interpreter failures, source-provenance replay, negative-test
runs, integration repair and exported proof. Raw proof bundles/logs remain in
`/home/l4nd0/greyhound-future-comparison-output-20260924/`.
Initial fixture repairs supplied required race date, native runner/source,
retention/raw-export, venue-alias and weather metadata. A late fixture initially
used seconds where discovery uses minute-resolution jumps. No production guard
was weakened to make a fixture pass.

A later shared-temporary-directory run had two receipt protocol failures. The
collector owner independently reproduced sibling publication invalidating an
exact receipt snapshot, and supplied `a15f5882`. Its repair is included in the
187-test and final packaged proofs. Prior failed logs remain alongside successful
ones; isolated tests also passed before the repair. This is not a claim that
omitting failed runs validates the earlier implementation.

The original [access incident](offline_20260924_access_incident.json) is copied
byte-for-byte from the old worktree because #193 had left it Git-ignored. The
58 reserved races / 393 decoded labels remain exposed historically. No shown
label-value path enters the eligible inputs, but protected timing aggregates
influenced the original T−2 design context. Exclusion does not undo either fact.
All original search/ledger records remain unchanged. The 177 inspected races
remain development; their only reuse here was transparent variance planning.
No protected or actual future target outcomes were opened in this preparation.

## Remaining dependency and one decision

Technical capability is ready. **Scientific population allocation and machine-only
history authority are not.** November is not automatically free: the October
overround successor is unbounded and the separate residual proposal must also
remain disjoint. Existing production/retention authority cannot be inferred from
this research plan. The comparison conservatively rejects shared protected
history instead of silently altering features; that may sharply reduce initial
coverage. The current live campaign remains outside this evaluation.

The concrete decision is whether to allocate and activate the proposed exclusive
**November 1–April 18, 24-week comparison**, with exact frozen models and T−2
rules, while deferring/conflict-clearing the other prospective proposals and
explicitly approving machine-only history processing. The prepared JSON rejects
execution until those authority references are supplied. No request to activate
or access outcomes has been executed here.

Historical date variability suggests a target of **119 contributing dates**
(about 1,405 races at old density) for a 0.01-nat log-loss half-width, using
factor-two variance inflation and adjustment for four candidate/comparator
contrasts. This is not promised power, especially against production, whose
future paired variance is unknown. The fixed endpoint will report shortfall
rather than resize based on outcomes. Missing result closure yields descriptive
complete-result findings only, never a silently reduced confirmatory cohort.
Keep the production baseline until new, properly authorized evidence supports
a later prospective decision; this preparation establishes no market advantage.
