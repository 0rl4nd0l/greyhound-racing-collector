# Frozen market-form residual materialization

Final status: `FROZEN_MODEL_READY_AWAITING_ACTIVATION`

The original task card was repaired in place, transferred byte-for-byte into
this clean worktree, revalidated, and freshly claimed before substantive work.
The final-fit population was predeclared before fitting: all 678 frozen Tier A
races and 4,752 runners dated no later than 2026-07-09. The 140 frozen
historical exclusions are listed with their reason codes in
`fit_population.json`.

Exactly one shared race-conditional-logit base model and preprocessing state
was persisted. Full and half variants are deterministic derivations at
strengths 1.0 and 0.5; they are not separately fit models. Candidate
selection was not rerun, no fold was reopened, no threshold or algorithm was
changed, no prospective outcome was inspected, and no cohort cutoff was
assigned.

The canonical model SHA-256 is
`624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d`.
The manifest SHA-256 is
`8537cbc3d843d106a1fe48793ef01197454ef092c0244025fd65685636a42080`.
Two temporary same-fit verification executions reproduced the primary
parameters, preprocessing, optimizer result, fixed-fixture predictions, and
canonical model bytes. They created no alternative artifact or comparison.

The minimal canonical loader/scorer is fail-closed, canonicalizes complete
runner-set ordering, validates hashes/schema/features/provenance, derives
normalized full/half probabilities from one residual adjustment, rejects
outcome-bearing inputs, and exposes an append-only shadow-record writer. It is
not connected to a collector, database, runtime, unit, timer, service,
production pointer, deployment, promotion, or betting path.

## Runtime Functionality Proof

- Intended output: one offline frozen artifact plus a loadable append-only shadow scoring contract; no live output is authorized.
- Live output location: not created; activation and runtime integration are forbidden.
- Pre-run max timestamp or count: zero live shadow records because no live path was opened.
- Post-run max timestamp or count: zero live shadow records because no live path was opened.
- Rows/files inserted or updated after run start: zero production rows and zero runtime files; repository-only artifacts listed by the amended card were created.
- Readiness/gate status: `FROZEN_MODEL_READY_AWAITING_ACTIVATION`; production remains `KEEP_BASELINE / market-only implied probability`.
- Exact command/query used: artifact `sha256sum`, canonical loader fixed-fixture scoring, focused pytest, Ruff, and `py_compile`; no database query was executed.
- Result: `PARTIAL`
- Remaining blocker: live activation requires a separate owner-approved exact activation card, ancestry containing both this frozen-model PR and PR #45 head `aa35fa70fc49199acde09f5561b521ddb00d45aa` (or merged descendants), and an idle shared collector lock.

No production database was opened. No collector, service, unit, timer, runtime,
deployment, promotion, merge, or PR #45 branch was changed.
