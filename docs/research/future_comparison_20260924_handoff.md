# Frozen comparison integration handoff

Status: implemented offline; default off; no scientific activation. Historical
search #193 is preserved at `956d8289b8be062e9fe5ccdf554b9a9462391c20`.
This isolated branch merges collector prerequisites `729fb9fa`; the main owner
continues provider repairs independently. No provider, service or live-state
operation was performed by this track. Review/merge these opt-in changes into
the owner's current branch; do not roll back newer collector repairs.

The exact two artifacts and training provenance are in the
[registry](../../artifacts/research_comparison/frozen_20260924/registry.json).
Paths from repository root:

| Method | Artifact SHA-256 |
| --- | --- |
| residual plus box | `51639ddada362b1a14110461ef258dbff852eb7b2797ea074cd85f22381a4d32` |
| half residual | `e827e8f29c756c8995de08e15f8a6c25ca92f0709693cb55ad6bbeb9e16716e0` |
| unchanged production | `624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d` |

Both candidates fit the same eligible 331 races / 2,360 runners, June 10–July 8.
Preprocessing, feature list, coefficients, training membership and source hashes
are serialized; scalar inference agrees with the research implementation within
2.3e-16. Candidates are distinct from the deployed artifact. The 177 previously
examined evaluation races influenced development and are not a holdout.

After the exclusive allocation and activation described in the
[execution specification](future_comparison_20260924_spec.md), give the existing
campaign preparer its usual arguments plus:

```text
--operational-predictions --comparison-plan /absolute/path/approved-comparison-plan.json
```

No prepared plan can be passed successfully. The preparer validates status and
binds its exact SHA, exports the candidate files with its existing source
package, and adds `frozen_comparison: {path, sha256}` to the campaign plan.
The existing operational supervisor records observed-index denominators and
passes this binding through `WorkerConfig` to the actual prediction subprocess:

```text
--comparison-plan /absolute/path/approved-comparison-plan.json
--comparison-plan-sha256 <exact-approved-file-sha256>
```

The existing retained-input binding, receipt, current index, model and config
are still required. Do not invoke a different collector or pass an unretained
DB/form path. The comparison uses the same selected WIN receipt. Experimental
features intentionally follow #193's raw-card recipe, while production follows
its unchanged merged-history recipe. No fitted artifact is written into the
production model directory. Production's operational job stays operational;
comparison admission is separate and only exists under the approved plan.
Current live campaign predictions are not retrospectively research admissions.

Run this read-only verification on each completed comparison (no target labels):

```bash
PYTHONPATH=. "$R3_PY" -B -m scripts.verify_frozen_comparison \
  --output-root /absolute/prediction/bundles \
  --admission /absolute/programme/PLAN_SHA/attempts/RACE_SHA/admission.json \
  --expected-plan-sha256 PLAN_SHA
```

Archive comparison opportunities, dispatch/admission/completion files, bundles,
campaign capture/retention/terminal records and index coverage failures. A
consumed failure or missing completion stays missing. Do not retry a comparison,
select a later snapshot or overwrite a sealed prediction. Keep existing
collector coverage logs; an observed-index census cannot identify unseen races.

The one terminal evaluator is ready for later, separately authorized use:

```bash
PYTHONPATH=. "$R3_PY" -B -m scripts.evaluate_frozen_comparison \
  --plan /absolute/approved-plan.json --plan-sha256 PLAN_SHA \
  --authority /absolute/one-shot-outcome-authority.json --authority-sha256 AUTH_SHA \
  --result-database /absolute/collector-owned-official-results-snapshot.sqlite3 \
  --out /absolute/new-terminal-evaluation
```

It refuses before the fixed endpoint plus 14 days; never run it against a new
result source under this task's authority. It seals membership/exclusions before
joining exact admitted IDs via `OfficialResultSource`, preserves failed closure,
and uses date resampling with adjustment for both candidates against both
comparators. No interim metrics, fitting or production promotion.

Offline reproduction (no providers or services):

```bash
export PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R3_PY=/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-operator-ui-r3-python311-20260803-9d58f340/bin/python
nice -n 10 "$R3_PY" -B -m scripts.verify_frozen_comparison_package \
  --output /absolute/new-package-proof --python "$R3_PY" --repetitions 3
```

The helper exports committed source bytes, pins environment/file identities,
uses invented data, installs inherited kernel IPv4/IPv6 networking denial and
executes the real retained worker/subprocess path. It is engineering evidence,
not a future-race evaluation. Measured results and integration commit are in the
[execution report](future_comparison_20260924_results.md).
