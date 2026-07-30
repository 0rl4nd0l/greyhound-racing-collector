# Greyhound Operator UI V1 status

Observed 2026-07-30 during the uncommitted `GHU-010H` ledger-correction candidate:

| Field | State |
|---|---|
| Repository | `0rl4nd0l/greyhound-racing-collector` |
| Branch | Clean integration base; this uncommitted correction is being produced on isolated Codex X branch `codex-x/20260730T134724Z-13cf3a3b54-40caa5` |
| HEAD / tree | Exact accepted integration `13cf3a3b54a4a411465ac570e5ecb65b1669cdc3` / `cf80477be77676f4e8eec54a8aa23d2fd6917896` |
| Baseline cleanliness | clean |
| Upstream base | `origin/master` `51a5287dfc28c8d059df2768534498c4b6321230`, merged by `6f4fba42c45c73702efb017a21cbd284b44c1d04` |
| Current release / ticket | R1 / `GHU-010H` ledger correction is unaccepted in review; rejected `GHU-010G` is blocked; parent-accepted `GHU-010` and `GHU-010F` are accepted; `GHU-011` is ready and unassigned |
| Counts | 5 accepted tickets, 1 ready, 2 review, 5 blocked, 2 deferred, 24 planned (plus accepted audit milestone `GHU-000A`) |
| Assignment | `GHU-010H` transitioned legally `planned -> ready -> active -> review` on 2026-07-30 in run `20260730T134724Z-13cf3a3b54-40caa5`, child `46e2cfc9b75f3ff6170baa9263698df4`; `GHU-011` remains ready with no implementation run |
| Next safe action | Independent exact ledger review of `GHU-010H`, then assign one fresh bounded `GHU-011` fixture-dashboard implementer |
| Validation | Exact reviewed product/test bytes retained; prior focused pytest `2 passed` and Playwright `3 passed in 2.1s`; original broad suite remains failed at `24 failed, 518 passed, 40 subtests passed in 4527.96s`; stable diagnostic isolates validation-environment effects without relabeling that command passed |

`GHU-000` was accepted at original base
`9be52ecd589615b4ebd6212bd9595be761520b89`. `GHU-000A` accepted source-delta
audit run `20260730T092421Z-6f4fba42c4-baf67b`, session
`019fb257-217f-7d10-8409-b2a06a6bd20b`.

The accepted `GHU-001` contract is commit
`aa4b45a004b1f897fc5d4cd06b0a741be6cd2446`, parent
`6f4fba42c45c73702efb017a21cbd284b44c1d04`; file SHA-256
`b2b4af016b24dafa4f121f0415c0d948a4c0699c73586a235c6547fc3002512b`;
reviewed base-to-result diff SHA-256
`22df513812a6460fbae2f247a36bfca4f37536438d5eed6eabebda47b2774b30`;
implementer session `019fb267-ff74-71e3-bb91-2ed8d55be316`; independent reviewer
session `019fb26b-ae76-7411-91ae-b54d5514a137`.

Parent accepted correction `GHU-002A` after independent review. Its accepted
integration commit is `73f1e5d041f8d78ee0f48ce13e008f71c20090ca`,
parent `aa4b45a004b1f897fc5d4cd06b0a741be6cd2446`, tree
`d68116ba72b28f149707d7610821aea02cb35781`, and frozen/reviewed base-to-result
diff SHA-256
`34bbf4d1b6e6286b143d296183d35064909a614f8bef9dc90553c639b787fdd3`.
Implementer session: `019fb27d-891e-75f1-aea2-fac1565f1db4`. Independent
reviewer session/verdict:
`019fb285-4b56-7202-a298-206855e4c875` / `ACCEPT_GHU_002A`. Parent decision:
accepted the exact reviewed four-file delta, committed it mechanically, and
fast-forward integrated it into the clean canonical branch on 2026-07-30.
Accepted file SHA-256 values: `PLAN.md`
`6fba2ff2d028f0ae92ee85fe0acb556d4bd70fb19b5129db8b6a5a83828c5e0e`;
`TICKETS.md`
`9148ed8b3df8ac51d3e157ee1454ae9979d4113c79b3612186de640806c02e92`;
`STATUS.md`
`e77e671e786da138ee80f157e36ec91235508d12c9ff0921e29329e8465be208`;
`DECISIONS.md`
`107c05734e4d1ff780c136270afb2331eea69c32ed930ff9cbebf1347656c3bf`.

Supported now: the accepted repository/UI/runtime authority inventory, exact
source/evidence/authority contract, and persistent programme ledger.
Unsupported now: an accepted or deployed UI,
UI operations store, live dashboard/API, deployed identity, runtime health,
manual UI prediction, runtime prediction proof, corpus/model readiness,
market edge, training, promotion, EV, staking, betting, or public availability.

No push, PR, merge, deployment, runtime mutation, or live prediction has
occurred in this programme yet. Publication, PR, merge to the repository
default branch, deployment, and runtime proof are all `NOT_OCCURRED`.
`GHU-010` transitioned `active -> review` after freezing its fixture-only
candidate. `python3 -m py_compile app.py tests/test_operator_ui_shell.py`,
`node --check tests/playwright/operator-ui-shell.spec.js`, `git diff --check`,
stdlib HTML/CSS/route-AST validation, the explicit prototype-asset forbidden
content/mutation-hook scans, and the unchanged-root-template check passed.
The repository classifier selected `full_forecasting` because `app.py` and the
new Python test default to full. The candidate environment has no project
`.venv`, Flask, Jinja, pytest, local Node modules, Playwright, axe, or browser
harness, so focused Flask rendering, Chromium desktop/375px, axe, and the
classifier-selected broad suites did not execute and remain unproven. This is
a review candidate, not an accepted implementation. There is no authority or
runtime stop; acceptance requires validation in a provisioned environment.

The parent provisioned that environment and rejected the original browser
candidate because the generic legacy `after_request` HTML mutator injected
assets, a banner, scripts, and an inline handler into `/operator-ui/`.
`GHU-010A` is the smallest bounded correction: it preserves `GHU-010` in
review, reproduces its exact eight files, excludes only the operator namespace
from that mutation block, and extends the focused final-response assertion.
Focused pytest passed `2 passed`, and Python/Node syntax checks passed. The
single permitted Playwright command exited 1 before config or browser test
loading with `ENOENT` while creating
`/tmp/playwright-transform-cache-1000/20`. The private test server was stopped
immediately and no retry occurred. `GHU-010A` therefore transitioned
`active -> blocked`, not review. No ticket is accepted by this correction
implementer.

`GHU-010B` is the smallest follow-up correction to that preserved blocked
candidate. It changes only the Playwright reduced-motion setup and the two
ledgers: the test explicitly emulates reduced motion on its page before
navigation, proves the media query is active, and retains the computed style
assertions. It transitioned `planned -> ready -> active` after the accepted
`GHU-002` dependency was reverified and is assigned to run
`20260730T104152Z-73f1e5d041-f3a7f7`, child
`1980f622df32340e1887b42f494d557e`. Focused pytest passed `2 passed`; syntax,
diff, unchanged-root/product-identity, forbidden scans, and classifier checks
passed. The classifier selected `full_forecasting`; the broad suite was not
run. The one permitted Chromium-desktop invocation passed desktop and 375px.
The reduced-motion test proved its media query active and reached the retained
computed-style assertions, but Chromium returned the equivalent duration
serialization `1e-05s` where `0.01ms` was expected. The server was stopped
immediately and no retry occurred. `GHU-010B` therefore transitioned
`active -> blocked`, not review. Neither `GHU-010B` nor `GHU-010` is accepted.

`GHU-010C` is the smallest follow-up correction to the preserved blocked
`GHU-010B` candidate. It changes only the Playwright spec and the two ledgers:
computed animation and transition durations are parsed as numeric `ms` or `s`
values, normalized to milliseconds, and each must be at most `0.01ms`; the
active reduced-motion media-query and `scroll-behavior: auto` assertions remain.
It transitioned in order `planned -> ready -> active` after the accepted
`GHU-002` dependency was reverified and is assigned to run
`20260730T104856Z-73f1e5d041-46c798`, child
`3d827523ebdd701c6f2d65e937321714`. Focused pytest passed `2 passed`; syntax,
diff, root/identity, unchanged-root/product-identity, forbidden/isolation, and
classifier checks passed. The classifier selected `full_forecasting`; the
parent-owned broad suite was not run. The one private server process was
started on `127.0.0.1:5520`, but an incorrectly non-waiting readiness loop
exhausted while the app was still importing, and the trap stopped the process.
Chromium was never invoked and no retry occurred. `GHU-010C` therefore
transitioned `active -> blocked`, not review. No ticket is accepted by this
correction implementer.

`GHU-010D` reproduces the exact eight-file `GHU-010C` candidate and changes
only these two ledgers after reproduction. The six product/test files remain
byte-identical to `GHU-010C`. Parent validation passed focused pytest
(`2 passed`) and Playwright (`3 passed in 2.1s`), and the classifier selected
`full_forecasting`. The single broad suite exited 1 with `24 failed, 518
passed, 40 subtests passed in 4527.96s`. The failures cluster in validation
environment assumptions: Phase 7 runtime tests reject the launcher
`.state/runs` path through `_safe_operational_path`, and Phase 6 promotion
fixtures fail `data_domain_drift`. This is not a broad pass and does not accept
`GHU-010`.

The evidence-reconciliation run
`20260730T121336Z-73f1e5d041-2eec9d`, child
`5bc80e1862cf6053ea84699b9df5c8e5`, verified base commit
`73f1e5d041f8d78ee0f48ce13e008f71c20090ca` and tree
`d68116ba72b28f149707d7610821aea02cb35781`. Python and Node syntax,
`git diff --check`, forbidden/isolation, classifier, exact identity, and
allowed-path checks passed. Focused pytest was not provisioned in this run and
was not invoked; neither browser nor broad suite was rerun. A first static
batch did not start because of shell-quoting construction error (exit 2); the
corrected batch exited 0. Only after the static checks passed did `GHU-010D`
transition `active -> review`. `GHU-010A`, `GHU-010B`, and `GHU-010C` remain
blocked, while `GHU-010` remains unaccepted in review. The next action is
independent exact-delta review followed by the smallest stable-path/baseline
diagnostic correction.

Independent review of `GHU-010D` was completed in reviewer run
`20260730T122656Z-73f1e5d041-20a0f9`, session
`019fb309-f0b2-7e81-bcac-fa7d219e911b`, child
`70bb7e85bfd14d3400b7184ab019bf75`, with verdict
`ACCEPT_GHU_010D_FOR_DIAGNOSTIC`. Independent spec session
`019fb30a-b648-7ba2-936c-6e5f503ba1ac` agreed there was no deviation.

`GHU-010E` preserves the exact reviewed `GHU-010D` product/test bytes and
reconciles only these ledgers. A stable diagnostic ran only the 24 nodes from
the failed broad command. On the untouched base, all `24 passed in 709.55s`.
On the exact uncommitted candidate, `21 passed` and only 3 release-identity
nodes failed because the dirty worktree was correctly rejected, in `704.25s`.
The same exact eight-file candidate was frozen validation-only as commit
`eda152192e96f7f89ccfc4ab3e89d5965cbc4055`, tree
`ae2a32a5371f8fe73dc83d4e8daab5bcee9b37d6`; only those 3 nodes then passed
in `6.73s`. This isolates validation-environment effects and establishes no
candidate regression among those nodes. It does not turn the original broad
result—`24 failed, 518 passed, 40 subtests passed in 4527.96s`—into a pass.

The reviewer recorded one medium, non-blocking proof limitation: the focused
test does not instrument generic filesystem writes or pre-existing request
telemetry. Delivery also retains one low risk: the two ignored templates must
be force-staged with their exact verified bytes.

Parent exact-delta inspection rejected `GHU-010E` because the top-level
`GHU-010` Next safe action and Closeout evidence still stated that provisioned
Flask/Playwright coverage had not executed and must run before acceptance,
contradicting the later authoritative focused pytest `2 passed`, Playwright
`3 passed in 2.1s`, and stable diagnosis of the original 24 failures. This was
a ledger contradiction only; the six product/test files were accepted for
correction. `GHU-010E` therefore transitioned `review -> blocked`.

`GHU-010F` reproduced all eight `GHU-010E` candidate paths and changed only
these two ledgers relative to that reference. It transitioned in order
`planned -> ready -> active -> review` and is assigned to run
`20260730T131716Z-73f1e5d041-5989e5`, child
`2f5201106343bd3e03a2fe631d50d0ae`, session
`019fb339-6c63-7a70-a9a3-1b84f2d269b7`. The corrected top-level `GHU-010` entry
now records the current focused and stable-path evidence while preserving the
original attempt, blocked A/B/C history, D diagnostic review, and original
broad failure. Independent reviewer run
`20260730T133751Z-73f1e5d041-0d30ea`, session
`019fb33f-0f5a-7650-bc0c-1de8cc391704`, child
`dbdaf9f23daee7c986a0a59c61aea562`, returned `ACCEPT_GHU_010F` with no
blocking findings. Parent accepted `GHU-010F` and `GHU-010`, and mechanically
committed the exact reviewed eight-file delta on the clean integration branch
on 2026-07-30. Exact commit `13cf3a3b54a4a411465ac570e5ecb65b1669cdc3`,
parent `73f1e5d041f8d78ee0f48ce13e008f71c20090ca`, tree
`cf80477be77676f4e8eec54a8aa23d2fd6917896`, accepted cached diff SHA-256
`f6b25bc07f7f1a385154acdb87d1357399d4361c72b338abefc68fbbf2cd6cc8`.
The non-blocking medium limitation remains: focused coverage does not
instrument generic filesystem writes or pre-existing request telemetry. The
low delivery fact remains: both ignored templates were force-staged with
verified bytes.

`GHU-010G` was the rejected ledger-only closeout. After exact base verification,
it transitioned legally `planned -> ready -> active -> review` in run
`20260730T134255Z-13cf3a3b54-4c8070`, child
`e1c1f6cc7c4b53e89595c31dd6328bfb`. It correctly recorded the accepted
closeout above and the satisfied `GHU-010` dependency, so `GHU-011`
transitioned legally `planned -> ready` and remains unassigned. Parent diff
inspection found that `GHU-010G` also accidentally changed the existing
accepted `GHU-002` Claims supported line from its base wording to a false
cross-ticket fixture-shell claim. `GHU-010G` therefore transitioned
`review -> blocked`.

`GHU-010H` reproduces the correct `GHU-010G` closeout and restores the accepted
`GHU-002` Claims supported line byte-for-byte from base `13cf3a3b`. It
transitioned legally `planned -> ready -> active -> review` in run
`20260730T134724Z-13cf3a3b54-40caa5`, child
`46e2cfc9b75f3ff6170baa9263698df4`. It remains unaccepted in review pending
independent exact ledger review. Publication, PR, merge to the repository
default branch, deployment, and runtime proof remain `NOT_OCCURRED`.
