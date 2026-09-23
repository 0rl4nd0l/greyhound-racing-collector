# Offline combined collector/R3 release preparation — 24 September 2026

Status: **prepared and tested offline; not merged, installed, armed or activated**.
This follow-up does not change the 24 September collector rehearsal.

## Exact sources and runtime

- PR #184: `fe3d984a13a7446ca0af301dd736fed20bab0b96` (unchanged).
- PR #185: `5f5ef82999675ae89c2100ed32f8abbe7d1669d8` (unchanged).
- Local combined baseline: `9011d614c1609147be0464810eda7173ed4bf7ab`.
- Tested packaging correction: `92f474ee8ce2f730e0292286bafc4ea2cc3a24b8`.
- Final package source commit/tree are recorded in `source-identity.json` and
  the generated repository binding. Later documentation changes do not change
  the tested Python implementation.

Local release root:
`/home/l4nd0/greyhound-r3-release-preparation-20260924`.
Its `source/` is an isolated Git worktree with code, configuration and UI source
materialized; real race/history/outcome artifacts were excluded. No installed
checkout represents this combined candidate. The source archive is for review;
its extraction alone cannot replace the generator's required clean Git identity.

The new `python/` environment copies the installed R3 interpreter's package set,
without changing either installed interpreter or downloading dependencies:
Python 3.11.15; **195 exact packages**, including requests 2.32.4 and
charset-normalizer 3.4.9. `pip check` passes. All **39,044** non-bytecode package
and executable files match the original pinned R3 environment. Package files are
independent copies; the existing pinned CPython base supplies the standard library.
`environment-lock.json`, `requirements.exact.txt` and `runtime-file-manifest.json`
record the identities. This is a local pinned runtime, not a portable wheelhouse.
Browser availability and live acquisition remain unproven.

A new synthetic proof passes the **full 195-package lock** through the real
retention subprocess and prediction consumer after deleting the invented original
history database. Substituting the campaign's requests 2.34.2 and charset-normalizer
3.5.0 is rejected before scoring. No environment or provenance check was relaxed.
The common environment therefore resolves the offline interoperability blocker;
it does not change the currently installed mismatch.

## Packaging defect fixed

The existing paired retention service packager rejected valid #184-generated
units when relocating their source, because its string substitutions omitted the
new source-bound Sportsbet `ExecCondition`. It also omitted #185's R3 options.

The packager now compares the complete original unit against the generator at
its original source path, then renders the destination source. It preserves
`--skip-shadow-run`, `--r3-job-store` and `--r3-prediction-bundles`. Missing or
modified access checks still fail before output. The new regression failed on
the original implementation and passes with the repair; exact rollback bytes
and both lanes' shared interpreter/DB/lock remain enforced.

## Concrete package, without live authority

- `collector-proposed/`: both services and unchanged timer bytes. Both services
  use the new common interpreter/source. Full collection has the R3 job and sealed
  bundle roots and skips shadow prediction. Existing forward-corpus/baseline and
  shadow-model bindings are preserved; these settings do not authorize starting
  their observers or accessing historical artifacts.
- `r3-disabled/`: actual deployment-generator output using existing persistent
  paths and external secret-file reference. Connected mode is zero. No live
  evidence, retained race binding or journal activation was fabricated.
- `source/var/operator_ui/generated/repository-v1.binding.json`: exact source,
  runtime, static artifact and persistent-root identities for the disabled package.
- `rollback/`: exact original four collector units, R3 service, R3 environment and
  repository binding. Secrets were not copied into the package.
- `retention-template-v2-NOT-AUTHORIZED/`: actual paired-packager proof, preserving
  proposed R3 settings. Its named retention configuration deliberately does not
  exist. These templates are **not an activation package**.
- `package-manifest.json`, `package-validation.json`, `systemd-verify.log` and
  `stage_package.py` / `validate_package.py`: reproducible local preparation and
  verification evidence. Scripts generate local files only; no installer or
  launch script is provided. Earlier superseded renderings remain labelled.

Independent standards review found no issue. Spec review identified omission of
installed forward-corpus bindings in the initial local rendering; final rendering
preserves those bindings and validates them against the originals. No protocol
membership or baseline was changed.

## Executed validation

All final tests use the separate common runtime, synthetic fixtures, the existing
network/data audit guard, no pytest plugin autoload, `--noconftest`, and
`-o addopts='' -p no:cacheprovider`:

- Deployment generator: **151 passed**, 36.86 seconds.
- Retained consumer, retention, bootstrap, journal, R3 official-result candidates
  and paired service packaging: **151 passed**, 37.78 seconds.
- Full-lock release consumer proof: **2 passed**, 2.71 seconds.
- Agent's separate targeted packager regression: **23 passed**, 7.74 seconds
  (included in the 151 above, not additional coverage).
- Actual staged package validation and systemd syntax verification: passed.

Logs are in the local release root. An initial source-only test attempt failed
because application source files were omitted; those files were materialized
from the same commit. A broader run using repository-wide pytest defaults was
**incomplete, exit 143**, with profiling/coverage harness failures and the then
unfixed packager errors. It has no pass total and is not acceptance evidence.
A focused deployment reproduction exposed profiling contamination; all 151
checks passed under the isolated harness. The large live-adapter suite was not
repeated; its partial run is not claimed as a pass. Prior overnight and GitHub
checks retain their original commits and are not presented as tests of this package.

## Next authorized transition

1. Keep today's #184 observation/package unchanged; its successful collection
   remains a separate acceptance case.
2. Review both PRs and this follow-up repair. The follow-up is stacked on #185
   and incorporates pinned #184; it must be reconciled/retargeted after those
   dependencies, not merged as an independent duplicate rollout.
3. After separate deployment approval, recheck installed hashes/quiescence,
   approve the permanent combined source/runtime and persistent paths, install
   paired units plus R3 bindings with timers still disabled, and retain exact
   rollback files. Regenerate bindings if any path, source or unit bytes change.
   No currently prepared artifact authorizes this transition.
4. Obtain separately authorized live evidence for the exact installed pair;
   generate enabled R3 authority only after those observations. The disabled
   binding cannot stand in for live authority. Preserve the 300-second R3 and
   65-second refresh boundaries.
5. Authorize one eligible future race's history/retention scope, obtain its
   accepted pre-cutoff retained bundle, bind its exact manifest while time remains,
   then separately authorize one journal claim and bounded official-result closure.
   The ordinary full collector retains reporting and result-observer stages;
   their scope must be included in that approval. This is not a new scheduler.

Proposal remains one complete authenticated retained observation for one eligible
race. Extra capture windows remain protocol-specific. No real-race prediction,
history/outcome read, live source access or model-performance claim was made.
