# Greyhound Operator UI V1 status

Observed 2026-08-01 at the current clean programme integration base (the R1
product checkpoint remains preserved in the history below):

| Field | State |
|---|---|
| Repository | `0rl4nd0l/greyhound-racing-collector` |
| Branch | No current branch claim; ledger state is durable independently of delivery mechanics |
| HEAD / tree | Base `dee082e954038c9ac4bf48d48bbe3901879310b8` / `302e144765d9bfd7ea3a7a1ef8a25e4fd3ab2c41`; current work is an uncommitted review candidate |
| Baseline cleanliness | clean |
| Historical/upstream source | Actual historical merge parent `51a5287d05c790e3855e5b74ce7117a29340135e`; later `origin/master` drift `f38a125f6364b8a60d17ae9c971b0ce172874eea`, tree `408a8adbfa2bd436132bc4d2c63e952aeb57c5a5`, parent `51a5287d05c790e3855e5b74ce7117a29340135e`; local merge `0b08966b31c15d8b459b9c6b60a48b19030a9ce4` |
| Current release / ticket | R2 and `GHU-030`/`GHU-031` are accepted/integrated. Coupled `GHU-032 + GHU-033 + GHU-034` is an unaccepted review candidate. R5 remains deferred. |
| Counts | 29 accepted tickets, 0 ready, 0 active, 3 review, 13 blocked, 2 deferred, 7 planned (plus accepted audit milestone `GHU-000A`) |
| Assignment | Fresh bounded implementer candidate for dependency-ordered `GHU-032 + GHU-033 + GHU-034`; implementation is complete pending independent review |
| Next safe action | Freeze and independently review the exact uncommitted R3 submission/polling/result candidate; parent retains acceptance and integration authority. |
| Validation | Focused R3/security/store/bootstrap/connected UI: `282 passed in 29.78s`. Required full Operator UI: `804 passed in 119.75s (0:01:59)`. |

`GHU-000B` audit run `20260730T172346Z-1bacc67937-3c5f6b`, session
`019fb40d-e8ed-7d40-8ab2-8ad2b156552c`, child
`4a4c8758a536447e5c001a3a80aa6caa`, returned
`GHU_000B_CORRECTION_REQUIRED`. This correction supersedes changed source
details only; it does not reopen accepted `GHU-001`.

Rejected `GHU-000B` implementer run
`20260730T173844Z-c77b3be5ad-37e474`, session
`019fb41b-abcb-7110-8b5d-1c6f7857758e`, child
`cb1c337f1c00368a51b58ba87b4ccdbe`, produced preserved checkpoint
`c550b81f111d0e053c1c3dd6014ef0f28b7638c1`, tree
`179300b2ddff681a52c9f7ae6fdffbf2c0137c15`, five-file diff SHA-256
`0ea14c1bafef9ca8917a87ac9a1836e4d733e5c36f83543c962a58601add2835`.
Parent rejected it for abbreviated evidence hashes, false missing-session
statements, incorrect predecessor failure classification, an accepted ticket
depending on a blocked ticket, redundant downstream dependencies, incorrect
new-ledger dates, and missing rejection/correction-ticket ledger state.
`GHU-000B` therefore transitioned `review -> blocked`.

Rejected `GHU-000B1` implementer run
`20260730T174818Z-c550b81f11-c6ce5a`, session
`019fb424-5860-7bc3-9b71-bd0e88b8880a`, child
`812c251abd99366b07fc1f9f02f82820`, produced checkpoint
`cc65dca19cd4bb9fa6b8c836dc843c0ba00bed7b`, tree
`009129fc2506a1f9d5d867177279c27c4956d113`, correction diff SHA-256
`fd2b93a33df15974173f9212449c7ce0c43e22fb303b6112e5d96e20bcd87363`.
Independent reviewer run `20260730T175213Z-cc65dca19c-10a89a`, session
`019fb427-fb80-7122-91ad-6cd2987b17c4`, child
`0f16abbb824ddff3ed7e63f6deba86bc`, returned `REJECT_GHU_000B1`.
The exact blockers were that `DEC-GHU-000B-FIXED-COLLECTOR-INDEX` still read as
operative without an append-only entry recording parent rejection of B,
independent rejection of B1, and B2 correction state, and that the decision
abbreviated accepted integration `c77b3be5ad4aa78b70a9ba89f25ee801d50f27c0`.
Parent rejected `GHU-000B1`; it transitioned `review -> blocked`.

Rejected `GHU-000B2` implementer run
`20260730T175638Z-cc65dca19c-c66202`, session
`019fb42d-217a-7fa0-966f-f36ddabd78d5`, child
`49e930e41c3a2fa72690df154d3b130c`, produced checkpoint
`3271721e5b19bc795f775a00a608c557f85b0112`, tree
`cc52608ffff88ce784c08da3236b55c82ec753fd`, binary diff SHA-256
`0893bde44d28fd9bf24795771947374dca82a3654d76dfdd979dbbb2f85fdc6d`.
Independent reviewer run `20260730T180236Z-3271721e5b-bed535`, session
`019fb431-732f-7dc2-ade5-45803721691c`, child
`fcd20a35022ea048ded40777f364b710`, returned `REJECT_GHU_000B2`. The sole
blocker was the exact stale `GHU-000B` current pointer directing independent
review of rejected `GHU-000B1`; all other review axes passed. Parent rejected
`GHU-000B2`; it transitioned `review -> blocked`.

`GHU-000B3` preserves B2's independently-passed fixed-index mapping and
authority correction and changes only that stale current pointer plus the
necessary rejection/closeout bookkeeping. It transitioned
`planned -> ready -> active -> review` in run
`20260730T180514Z-3271721e5b-a531a0`, session
`019fb433-d9c0-7b22-b36d-4a832833e4ce`, child
`50f7490271d4dd156e56032b72ddb9ab`. It remains unaccepted pending independent
exact-delta review and parent decision.

Rejected predecessor `GHU-000C` ran as
`20260730T172828Z-f38a125f63-770b6a`, session
`019fb412-36a3-7992-b249-0ba5e635c72c`, child
`9f5cd6a19933e2a044a2a3c23c88ec1c`. Checkpoint
`3e9f639dfff62ffddd85aa00bab3d5c6b475cdf6`, tree
`bdf39b69919e07bbfd5d8d330644b665aaa57fc7`, is not in integration history.
Parent focused validation was exactly `1 failed, 18 passed`: the post-read
`/proc/self/fd` `OSError` was caught by the broad handler and misclassified as
caller-missing code `CURRENT_INDEX_SOURCE_MISSING`, rather than the expected
`CURRENT_INDEX_PATH_UNSAFE` with reason `path_replaced`.

Accepted correction `GHU-000C1` ran as
`20260730T173227Z-3e9f639dff-b1c869`, session
`019fb415-e17b-7761-84b9-97a96a8fb58d`, child
`a1704231395179acca71854ce5ba7acb`, correction diff identity
`86567b5e32177ed0028940cb11158fee6e34a5f876f0903a4ef151458e1e59aa`.
Parent validation passed `19 passed in 0.86s`. Independent reviewer run
`20260730T173456Z-04c32c37fe-4018cf`, session
`019fb418-1b23-7fa1-985d-48f0219864e1`, child
`6132758c7ecce3222bbf0f2059b29c77`, returned `ACCEPT_GHU_000C1`. Parent
accepted final integration `c77b3be5ad4aa78b70a9ba89f25ee801d50f27c0`,
tree `fe5115435d18cbce6be055cf452acdba65518a76`, accepted staged diff identity
`26886b3fba57cce1369dc45794f563a5ba250fbeadb9f7585bdf3f731ddbe373`.
Linux `/proc/self/fd` and `O_NOFOLLOW` remain platform
limitations; the accepted work does not claim detection of absolute same-inode
concurrent content mutation.

At the historical `GHU-000C1` checkpoint, R1 history remained unchanged. In
particular, the `GHU-011L` selected suite was still recorded as running and had
no invented terminal result; the rejected F broad result remained rejected.
The distinct post-upstream `full_forecasting` selected broader gate remained
pending and was not the running pre-upstream `GHU-011L` suite. No `GHU-011`
acceptance was claimed at that time.

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

At the `GHU-002A` checkpoint, supported then were the accepted repository/UI/
runtime authority inventory, exact source/evidence/authority contract, and
persistent programme ledger. Unsupported at that checkpoint were an accepted
or deployed UI, UI operations store, live dashboard/API, deployed identity,
runtime health, manual UI prediction, runtime prediction proof, corpus/model
readiness, market edge, training, promotion, EV, staking, betting, or public
availability. This paragraph is historical and does not describe the later
accepted R1 fixture dashboard.

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
`GHU-002` Claims supported line byte-for-byte from base
`13cf3a3b54a4a411465ac570e5ecb65b1669cdc3`. It
transitioned legally `planned -> ready -> active -> review` in run
`20260730T134724Z-13cf3a3b54-40caa5`, child
`46e2cfc9b75f3ff6170baa9263698df4`. It remains unaccepted in review pending
independent exact ledger review. Publication, PR, merge to the repository
default branch, deployment, and runtime proof remain `NOT_OCCURRED`.

Accepted closeout supersedes the historical pending states above.
`GHU-000B3` implementer run `20260730T180514Z-3271721e5b-a531a0`, session
`019fb433-d9c0-7b22-b36d-4a832833e4ce`, was independently reviewed in run
`20260730T180925Z-44fe9a0875-a94ad1`, session
`019fb437-b8d1-7dd3-9e06-f4494603e9d7`, verdict `ACCEPT_GHU_000B3`, and
accepted at integration `6e0b0d99c296a4c984faf0775bab88f8689e66da`, tree
`bff53978cdfeea8f604404432e1d672cba95a692`. `GHU-010H` implementer run
`20260730T134724Z-13cf3a3b54-40caa5`, session
`019fb347-d93d-7e31-a20d-01d41fa1c7f6`, was independently reviewed in run
`20260730T135324Z-13cf3a3b54-ca20f6`, session
`019fb34d-4cd5-7dd2-b7be-d7a8d7c745ff`, verdict `ACCEPT_GHU_010H`, and
accepted at integration `1bacc679377f54433ea757f8cbf7045e3ce8526a`, tree
`dee68158b8455d898d60807bdc0ff41c8caf1f7f`.

`GHU-011` product evidence spans run `20260730T135824Z-1bacc67937-e184ab`, session
`019fb351-f550-7682-9a15-ba28e7f5e1e0`, and run
`20260730T181451Z-b836e769ae-738e8d`, session
`019fb43c-b9e8-7c30-823d-894c79e30695`. Independent review run
`20260730T181807Z-e10cff2931-e22d8c`, session
`019fb43f-a9bd-7363-851c-d6a392a44548`, returned `ACCEPT_GHU_011M`.
The exact product delta is base
`6e0b0d99c296a4c984faf0775bab88f8689e66da`, tree
`bff53978cdfeea8f604404432e1d672cba95a692`, to head/integration
`e10cff293141569b1a5a169dd05efc8109e3c603`, tree
`07f02fc46b88b47bf0ade8ee264505f8b47c7d91`, and changes exactly `app.py`,
`static/css/operator-ui.css`, `templates/operator_ui.html`,
`templates/operator_ui_components.html`,
`tests/playwright/operator-ui-shell.spec.js`, and
`tests/test_operator_ui_shell.py`. Its base-to-head binary diff SHA-256 is
`7641c45d98a26590e64fff71a50d9dc1b7639b03ee1e6c1fb05aedbecd58681b`;
the head file SHA-256 values in that order are
`7d948ad95c314c857b0379f5a2aae63587bf2dfb3a6adc2867f12ac82d23f7f2`,
`ce995078d9a7bffa8baea4e924f9a55e2dc272908a9574feb7bce21756e8a9b7`,
`961c8054076b04240285acd0a74ea955f8b4b1af99271ae782092c0c2d1c407d`,
`203956f714d6e4622ec4c21a1e32d2da62f6c9baf5af6a549297e750543dc17f`,
`6bc876014659a24ca072cad73e31d298b898445dc7d4e981e47835486ee0a9e4`,
and `45b8e96df32c64f2d389cbe64ba86d87e487cf28d5c1942d497fe0a921f9a24c`.

Focused command `/tmp/ghu010-validation-73f1e5d/bin/python -m pytest -q
tests/test_operator_ui_shell.py` exited 0 with `5 passed`. The private browser
test server command was `PORT=5002 FLASK_ENV=testing MODULE_GUARD_STRICT=0
PREDICTION_IMPORT_MODE=relaxed ENABLE_ENDPOINT_DROPDOWNS=1 TESTING=1
TRAINING_MAX_SECS=30 DISABLE_NAV_DROPDOWNS=1
/tmp/ghu010-validation-73f1e5d/bin/python app.py --host localhost --port 5002`;
it was stopped after validation. Browser command `NODE_PATH=/tmp/ghu010-node-73f1e5d/node_modules
PLAYWRIGHT_BROWSERS_PATH=/tmp/ghu010-playwright-browsers
/tmp/ghu010-node-73f1e5d/node_modules/.bin/playwright test
tests/playwright/operator-ui-shell.spec.js --config=playwright.config.js
--project=chromium-desktop --reporter=line --workers=1` exited 0 with
`3 passed`. Classifier command `python3
scripts/ci/classify_forecasting_changes.py --base
f38a125f6364b8a60d17ae9c971b0ce172874eea --head
e10cff293141569b1a5a169dd05efc8109e3c603` exited 0 and selected
`full_forecasting`.

The first broad command `TMPDIR=/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/operator-ui-validation-tmp/ghu011m-e10-full
uv run --no-project --with-requirements requirements/all.in python -m pytest
-q --noconftest tests/race_collection` exited 1 with `1 failed, 550 passed,
40 subtests passed in 4945.67s`. Isolated diagnostic checkpoint
`86f39d54d3949c7bc2b6f670c809a6e5dea5050d`, tree
`944336f6465bf4d23786d0fcf9ef894cd04565c2`, changes only
`tests/race_collection/test_phase7_runtime_adapter.py`; target/module/full
runs passed 1, 25, and 551 tests, with 40 subtests in the full run. It is
`EXCLUDED`, `UNMERGED`, and `RETAINED`, not discarded; root cause is `UNKNOWN`.
Gate review run
`20260730T222316Z-86f39d54d3-b5f20d`, session
`019fb520-39bf-7be0-863e-7c3b8ced34bf`, returned verdict A `ACCEPT`.
Parent accepted the exact product checkpoint at the head/integration identity
above. Publication, push, PR, default-branch merge, deploy, runtime mutation,
and live proof are all `NOT_OCCURRED`.

The R1 gate is closed. `GHU-012`, `GHU-013`, `GHU-014`, `GHU-015`, `GHU-016`,
and the corrected `GHU-016A` closeout are parent-accepted. The corrected
closeout is integrated at commit
`3505efb299d25320fd86a0bf76aef5bf953fb5a7`, tree
`567c913b7be46c4d5747c8bf74d2fb4df5d8f664`. Supported claims are limited to accepted R0
contracts, the fixed collector-owned current-race index, and the accepted R1
fixture dashboard. No push, PR, default-branch merge, deployment, runtime
mutation, live proof, deployed/live/authenticated dashboard, R2+ behavior,
prediction/runtime/deployment proof, training, promotion, EV, staking, betting,
or public claim occurred.

`GHU-016A` is the parent-accepted ledger-only R1 closeout correction. Its
corrected closeout is integrated at the commit/tree stated above and records
accepted `GHU-012`–`GHU-016` at product checkpoint
`51fe070ba2a0778bca0b0334c00cae9d75561952`, tree
`0c2b82a391b8dd0a1dcc525cf4203edc910843c1`, and seven-path binary diff
SHA-256 `3f74ec86de1ab68b0fbb1a13125efa6fa416e3cc1fdc9bfc658f97118d3dc135`.
Independent reviewer run `20260731T001624Z-51fe070ba2-937103`, child
`eda938ef8b7cf07d87eb37719e0929c2`, returned `PRODUCT ACCEPT` and `GATE
ACCEPT`; parent fast-forward integrated the exact checkpoint. Focused pytest
`5 passed in 0.50s`; Chromium `5 passed in 6.9s`; classifier
`full_forecasting`; syntax/diff checks passed. The timestamped broad run is
not a pass: `15 failed, 536 passed, 40 subtests passed in 5400.39s`, all 15
failures being timestamped-path `_safe_operational_path` rejections in
`test_phase7_runtime_adapter.py`; identical nodes passed `15 passed in 12.47s`
from the stable detached path. No UI/product assertion failed. Visual hashes:
desktop `fc181de498c5ae287343cfc6026ef4e0dafcb9bb51ac91a605af37fab8d06ba6`,
mobile `e841e7953ea63cfb278837511ded73a1298c78159a7f575f92fac4b00962896d`.

`GHU-016B` is a review correction record, not a new formal ticket. The rejected
`GHU-016A` ledger candidate/source `978c4c92514701453f7c8a3252ca33880352764b`,
tree `4914a4d73cb150440bf3e54a3c8b84234f22ec15`, with base-to-candidate
diff SHA-256
`38032b6f50966c924e01dce1ee143e73d0613f2a9413dfbe687d135a7f4ca022`,
is preserved. Independent reviewer run
`20260731T015703Z-978c4c9251-84738f`, session
`019fb5e3-d4a1-7541-811f-b330dc0ebdf4`, returned `PRODUCT REJECT` and `GATE
REJECT`. Its two blocking findings were stale current-action pointers in three
accepted tickets and a `GHU-020` title/outcome/acceptance/validation contract
that still claimed broad source adapters and display fields despite its
foundational-primitives-only scope. The corrected two-file closeout repaired
only those findings and was subsequently parent-accepted/integrated at
`3505efb299d25320fd86a0bf76aef5bf953fb5a7`.

The frozen UX contract is **RESEARCH ONLY — NOT FOR BETTING** and
**PROTOTYPE DATA**; navigation is Dashboard, Race, Lifecycle, Result,
Collector, Corpus, Models, System, Audit; advanced source/evidence controls
remain disclosure/details; mobile is stacked with contained/scrollable dense
tables and visible actions/warnings. IMPORTANT R2 follow-up: responsive
stacked-card or equivalently readable mobile treatment for dense runner/model/
governance tables while preserving exact identities. OPTIONAL: narrow the
operator path predicate to exact supported routes and replace positional
fixture tuples with named fields.

`GHU-020` and its rejected checkpoints remain preserved evidence, not accepted
history. `GHU-020B` is blocked at preserved HEAD
`182fc11dc995d114363e903e5e679aa78edd9602`, tree
`9fa01491c2d0f210c989fcc3166e1e6f75e65a4e`. Main reviewer session
`019fb65c-03ac-7720-b63c-88e18acca58d`, run
`20260731T040821Z-182fc11dc9-4738ef`, returned `REJECT_GHU_020B` for exactly
two blocking findings: no construction-to-observation/connect inode identity
for configured roots/components/files/databases, and arbitrary historical
supported-claim prose under `AVAILABLE/FRESH`.

`GHU-020C` owned only those two corrections and transitioned `planned -> ready
-> active -> review` from that exact clean preserved base. Candidate checkpoint
commit `1e436a461e51923d97b44929bcb198bc20535e0b`, tree
`1425b9d11a3ab192a31f63426f1145b7ea00759e`, exists. Main reviewer session
`019fb682-d853-76e2-9d6d-20835750d7bd`, run
`20260731T045046Z-1e436a461e-70062d`, returned `REJECT_GHU_020C` for one exact
blocker: the construction identities were checked only before fresh
`lstat`/open, so replacement in that gap could become a new internally
consistent accepted identity. `GHU-020C` is blocked preserved evidence.

`GHU-020D` owns only that verification/open-gap correction and transitioned
`planned -> ready -> active -> review -> accepted` from the exact clean
rejected `GHU-020C` checkpoint. It opens the configured root and each nested
component by descriptor, compares every opened descriptor to the retained
construction identity, holds the complete descriptor chain, and rechecks it
through file observation or SQLite connect. Candidate HEAD
`8b1ac4235d478a0ef62380bbf61a265731f4d3e4`, tree
`612e6a4ce27f9d3bf49274831b8f8e29121d5a12`, was independently accepted by
reviewer session `019fb68d-96c4-7cb3-9cfc-a50c4f49d5cc`, run
`20260731T050230Z-8b1ac4235d-ce8207`, verdict `ACCEPT_GHU_020D`. Parent
integrated the identical tree mechanically at integration HEAD
`c95b0467ded033bdb995da7941b44a11a04b22b7`, tree
`612e6a4ce27f9d3bf49274831b8f8e29121d5a12`, branch
`agent/operator-ui-programme-20260730`, without rejected checkpoint history.
Parent focused validation passed 62 tests. The exact frozen gate command was
`PYTHONDONTWRITEBYTECODE=1` with external `TMPDIR` and
`uv run --no-project --with-requirements requirements/all.in python -m pytest
-q --noconftest tests/race_collection`; it passed `551 passed, 40 subtests
passed in 6001.15s (1:40:01)`, exit 0. Exact `GHU-020D` correction diff
SHA-256 is
`cb3aa4bda38e069bb31800f286a4a3dadd0d2dabe5e11e30312cd8a55e5c1e13`;
accepted-base-to-final-product diff SHA-256 is
`10eae73d4490ee3fd52d722fb1ff8bc3b8a1a7969bfdca034b397ce28de17e1f`.
Push, PR, merge, deploy, publication, runtime/data mutation, and canonical
database/history access remain `NOT_OCCURRED`.

The prior `GHU-020E` ledger-only candidate is preserved rejected evidence. At exact clean
base `c95b0467ded033bdb995da7941b44a11a04b22b7` and candidate
`7462e42cd191adcbe1adc43ebe420b2c93226455`, its live-recomputed binary diff
SHA-256 is
`a0aa6cb799d98c0a5b6d27320ffa0328d35857e4499a95f33db63ef8583332da`;
candidate `STATUS.md` SHA-256 is
`cd11d85d32946a7f59d657225deeba569ec34823a4a52c3a07d26a8cca92978a`
and candidate `TICKETS.md` SHA-256 is
`315c54392da34320cd56f9fe4bb1bba9514fb931129d8f57d07b15eba3f3bc21`.
The historical `GHU-020C`/`GHU-020D` checkpoint objects are archived evidence
and are not claimed independently recomputable in this worktree. In
particular, historical D correction SHA-256
`cb3aa4bda38e069bb31800f286a4a3dadd0d2dabe5e11e30312cd8a55e5c1e13`
is archived evidence, while accepted-base-to-final-product diff SHA-256
`10eae73d4490ee3fd52d722fb1ff8bc3b8a1a7969bfdca034b397ce28de17e1f`
remains the accepted product evidence.

The non-formal `GHU-020F` candidate is rejected evidence at exact commit
`fd87bee4fb912154be33f5cd71e4505a220e323c`, tree
`e4c31a7ed695757d10c59d0f32c5f7f5cc692388`, from parent
`7462e42cd191adcbe1adc43ebe420b2c93226455`. Its parent-to-candidate binary
diff SHA-256 is
`40b1303fdd2821f776d4b08be10f2839f57c683b905d99159944f8af568dedb8`;
the final accepted-product-base `c95b0467ded033bdb995da7941b44a11a04b22b7`
to F binary diff SHA-256 is
`2510ece8dbebe136dbc7dfaea887727155879aab9b58499493f221cba7cab0d5`.
F file SHA-256 values are
`e033a570f65dd0ae041a6733c7e34efff82bb8513e9011e0e31489ad2e592faa`
for `STATUS.md` and
`73ca99aa024e580f1577ab87f9abbefccffea12b19c8e9ae43bd8826b1c7dcab`
for `TICKETS.md`. Main reviewer session
`019fb6fc-f83e-7902-b8e2-2e47f18d1957`, run
`20260731T070407Z-fd87bee4fb-450905`, returned `REJECT_GHU_020F` for exactly
three stale live Next safe action fields: accepted `GHU-016` still required
inspection/integration of already accepted `GHU-016A`; accepted `GHU-016A`
still required review of the superseded `GHU-020` candidate; and blocked
`GHU-020B` still required review/acceptance of already rejected `GHU-020C`.

`GHU-020G` is the non-formal smallest correction record for those three live
pointers and supersedes rejected F without recursive closeout, a formal ticket,
or any change to formal counts or the accepted status of corrected formal
`GHU-020E`. It preserves accepted R1/`GHU-016A`, accepted/integrated
`GHU-020D` foundation evidence, and all rejected predecessor/E/F evidence.
The reviewer/parent freezes final G diff and file hashes externally together
with the exact commit/tree; self-referential final G hashes are not embedded
here.
Accepted/integrated `GHU-020A` precedes accepted/integrated `GHU-021` and owns default-off connected-mode
authentication/authorization, secure configured secrets and session
expiry/rotation, CSRF, and the separate append-only UI operations/access-audit
store. Every authenticated operational GET appends and confirms audit before
disclosure; append failure returns deterministic non-operational error with no
evidence. The store is separate from the canonical racing DB and future
prediction-job DB. No public bind, operational POST, arbitrary path, shell,
service, lock, browser, canonical write, training, promotion, betting, or
runtime action is in scope.

`GHU-021` is accepted and integrated. Accepted child HEAD
`8db34fe53af252fdb6dd743b51d3531fb1f8b618` has tree
`af89578953145e5049bb9d2c70f3de150fad86ca`; parent integration commit is
`4a24218379d186d951f47d3fcf0d17d396d7d066` with that exact tree. The
programme correction chain is preserved. Final correction diff from
`f1f1bd96c60d690bb2a7247e79db4cffc6360594` has SHA-256
`d18f42206bfc4a103d4ae352f93f3963f54c62820de7cb5646dbb1860a67dafa`;
the full accepted four-path diff from
`d71857e232ce7371280f9e5c56c45be7b9f7f9e5` has SHA-256
`037adf12c9d37c0e96a66e65645392a06482378262ac8b496c0b662253b860ea`.
Parent authoritative focused gate passed `254 passed in 39.77s`. Final reviewer
run `20260731T100618Z-8db34fe53a-a6a1cc`, session
`019fb7a3-c1d4-7da1-946e-291651763774`, returned `ACCEPT_GHU_021`.
Classifier selected `full_forecasting` because paths default unknown-to-full;
the exact 551-test broad gate passed `551 passed in 1653.94s (0:27:33)`, exit
0. The validation caused no repository, product, runtime, or data mutation.
Push, PR, default-branch merge, deployment, runtime mutation, and live proof
remain `NOT_OCCURRED`.

Read-only audit run `20260731T095910Z-d71857e232-648124`, session
`019fb79d-352a-7c62-88a7-588974824079`, returned
`PREREQUISITE_REQUIRED`. The fixed packet is runnerless:
`collector_current_race_index_v1` and `_normalize_current_index_rows` retain
only race/time identity; refresh `race_window_record`/`selected_races` has no
sealed runner rows; daemon `current_race_index_publish` lacks packet SHA-256;
and predictor bounded discovery strips runners even though downstream request
construction can consume `participants`/`runners`. `GHU-022P` therefore owns
the smallest collector current-index v2 evolution before `GHU-022`. It does
not reopen `GHU-000` or `GHU-021`; R2 remains in progress and R5 deferred.

Read-only audit verdict `PREREQUISITE_REQUIRED` also found that current
`artifacts/on_demand_prediction_runs/prediction_<timestamp>_<12hex>` bundles
cannot safely back `GHU-023`. The producer writes canonical v1 result and
manifest files, and replay verification detects common changed/missing/added/
symlink cases, but there is no producer-owned bounded index or ordering,
strict unknown-field schemas, descriptor-safe fixed-root verifier, aggregate
logical identity, stable prediction/job identity, or complete terminal
race/model/config/time/stage contract. The result's absolute bundle path is
not UI authority. `GHU-023P` owns only the smallest producer/index/verifier v2
prerequisite. Legacy/unindexed v1 remains replay-compatible, catalog-ineligible
and unchanged. This does not reopen `GHU-021`; its exact 551-test broad gate
passed `551 passed in 1653.94s (0:27:33)`, exit 0.

Independent correction review further requires `GHU-023P` to serialize index
replacement with its one fixed producer-owned bounded lock, failing closed and
never stealing stale locks or manipulating a collector lock. Exact bundle
membership is the manifest regular-file set plus exactly the parent-directory
set derived from canonical manifest names, verified by bounded
descriptor-relative per-directory enumeration with no recursive tree walk.
Ticket state and counts are unchanged; no implementation validation is claimed.

R2 closeout supersedes the historical pending state above without rewriting its
rejection evidence. The initial `GHU-026` run
`20260731T175127Z-d5dcb1f846-a7be01`, session
`019fb94d-ab83-75a1-bae9-4eb635660147`, archive
`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/codex-x-run-archive/20260731T175127Z-d5dcb1f846-a7be01`,
produced historical shorthand candidate tree `6d7b5d...` and diff `b464...`.
Parent validation was `1 failed, 8 passed`: the candidate lacked an explicit
auth literal/provider-binding seam and browser coverage was incomplete.

Rejected `GHU-026C1` run `20260731T175955Z-7d581abb7f-63adaa`, session
`019fb955-6551-7cc3-a01f-8713152942e8`, archive
`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/codex-x-run-archive/20260731T175955Z-7d581abb7f-63adaa`,
produced historical shorthand candidate tree `122d169...` and binary diff
`f6a260...`. Python focused passed `13 passed` and API/security passed
`202 passed`, but parent browser validation was `12 passed, 2 failed` because
the page overflowed horizontally at 375px. Reviewer run
`20260731T180927Z-7d581abb7f-9345cb`, session
`019fb95e-165a-7ee3-81db-a8eac3b1a82d`, archive of that run ID, was
interrupted after the material browser failure and returned no acceptance
verdict.

Rejected `GHU-026C2` run `20260731T181619Z-7d581abb7f-7b2c3a`, session
`019fb964-77cc-7323-8a07-009d620d797a`, archive
`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/codex-x-run-archive/20260731T181619Z-7d581abb7f-7b2c3a`,
froze staged tree `d80e30e9fa535ad84643986f7a3eadcd2d4bca02` and staged binary
diff SHA-256
`866e7f015255d5d6afeef5c6a2295188ec70e12a235800f66a91a562a11bc6e1`.
Parent focused passed `13 passed`, API/security passed `253 passed`, and exact
browser passed `14 passed`. Independent reviewer run
`20260731T182606Z-7d581abb7f-912d4f`, session
`019fb96d-5ef8-7b41-8b5b-3adc9202a603`, archive of that run ID, returned
`REPAIR_REQUIRED`. Its sole remaining finding was that
`static/js/operator-ui-connected.js` fabricated `server_observed_at` from
browser time on the top-level offline catch while the evidence drawer labelled
it request-observed server evidence.

Accepted `GHU-026C3` implementer run
`20260731T182944Z-7d581abb7f-919530`, session
`019fb970-c2a7-79d0-8acb-ea04e99b9446`, archive
`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/codex-x-run-archive/20260731T182944Z-7d581abb7f-919530`,
used actual model `gpt-5.6-sol` and froze the exact nine-path candidate tree
`7e087f5f117a8278e3ae61016baa3c8c3adf2d06`, full binary patch SHA-256
`541fd821b79515982ed5372ba7581388176d12c9d71bb741095472d6a4cb34ab`.
It changed only the remaining correction atop C2: omit fabricated offline
`server_observed_at` and assert `request observed not supplied`; the other
seven files were byte-identical to C2. Implementer focused Python/API/security
passed `266 passed in 48.56s`; Python/JavaScript syntax and
`git diff --check` passed. Parent ran exactly
`npx playwright test --config=playwright.config.js tests/playwright/operator-ui-shell.spec.js --project=chromium-mobile --project=chromium-desktop`;
it passed `14 passed (29.0s)`.

Accepted independent `GHU-027C3` review run
`20260731T183753Z-7d581abb7f-c74749`, session
`019fb978-2391-7e50-b8e4-a38e61381870`, archive
`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/codex-x-run-archive/20260731T183753Z-7d581abb7f-c74749`,
used actual model `gpt-5.6-sol` and returned `ACCEPTED`. It independently
matched the base, all nine hashes, candidate tree/path scope, AST and
JavaScript syntax. Standards and specification axes had no material finding;
it relied on supplied 266-test and 14-browser-test evidence and launched no
browser.

Parent inspected and accepted the exact diff, then mechanically integrated it
as commit `0b34bdd10533676f9bf491c85259bc5342049652`, tree
`7e087f5f117a8278e3ae61016baa3c8c3adf2d06`. Parent focused validation passed
`266 passed in 46.90s`; the binary patch SHA-256 and every file SHA matched the
reviewed candidate. Supported now are the accepted/integrated R2 fixture
prototype plus authenticated GET-only evidence-backed dashboard, exact provider
binding, audit-before-disclosure, truthful stale/unavailable/offline/auth
states, and validated desktop/mobile/accessibility behavior at that integrated
tree. Unsupported remain deployed identity/runtime health until generated
deployment is merged, deployed and verified; live UI prediction until R3 and
bounded proof; and training, model promotion, EV, staking, betting,
profitability, public exposure, or outcomes not proven by evidence. No push,
PR, default-branch merge, publication, deployment, runtime/data mutation, live
proof, training, promotion, EV, staking, betting, or public exposure occurred.

R2 is accepted/integrated. R3 is next and in progress only through ready
`GHU-030`; `GHU-031` remains planned because it depends on `GHU-030`. One fresh
bounded implementer may own their dependency-ordered coupled tranche
atomically, implementing the job contract/store first and then the
fixed-argument worker in the same candidate. Independent review is required
before parent integration. R5 remains deferred.

R3 parent closeout supersedes only the stale current pointer immediately above.
Parent accepted and integrated `GHU-030` and `GHU-031` at commit
`dee082e954038c9ac4bf48d48bbe3901879310b8`, tree
`302e144765d9bfd7ea3a7a1ef8a25e4fd3ab2c41`. The accepted source candidate was
`88853fdd2a26d7b8b1b7b2c45a5900b87d8e9c5a` with the identical tree and
cumulative binary diff SHA-256
`3b6443dfb565829d35374ff56ea4656f1cae4483aa46451e388b762571efeb73`.
Implementer session `019fba2b-0df0-76c2-a55f-b6cdffbcc94c`; independent
reviewer session `019fba30-921d-7c13-a443-3ac0a79351ff`, verdict `ACCEPT`.
Targeted and full Operator UI validation passed `316` and `801` tests;
parent integration validation likewise passed targeted `316` and full `801`.

At that exact clean base, the coupled `GHU-032 + GHU-033 + GHU-034` candidate
transitioned dependency-order `planned -> ready -> active -> review`. It adds
an explicitly composed, default-off Level-2 JSON submission/read API, exact
server re-resolution seam, actor-scoped idempotency and rate limiting,
audit-confirmed durable transitions before launch, actor-isolated polling,
persisted timeline, and strict verified-only result disclosure. The connected
dashboard gains exact-race selection, guarded submit, refresh reconnect,
terminal blocker/result, and evidence views. Polling is used because the
repository has no SSE infrastructure. No replay command is displayed because
this tranche established no safe fixed copy-only command. Focused validation:
`/tmp/ghu010-validation-73f1e5d/bin/python -m pytest -q
tests/operator_ui/test_r3_api.py tests/operator_ui/test_security.py
tests/operator_ui/test_job_store.py tests/operator_ui/test_bootstrap.py
tests/operator_ui/test_connected_ui.py` passed `282 passed in 29.78s`.
The required single final command
`/tmp/ghu010-validation-73f1e5d/bin/python -m pytest -q tests/operator_ui`
passed `804 passed in 119.75s (0:01:59)`.
Independent review, parent acceptance/integration, commit, publication,
deployment, runtime/data mutation, live prediction/proof, browser execution,
collector action, training, promotion, EV, staking, betting, and outcomes:
`NOT_OCCURRED`.
