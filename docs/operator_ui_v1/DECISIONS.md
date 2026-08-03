# Greyhound Operator UI V1 decision log

Append-only: corrections add a superseding entry; they do not rewrite history.
Each entry records date, ID, context, decision, evidence, consequences, and
supersession.

## 2026-07-30 — DEC-GHU-000-AUTHORITY

- Context: R0 repository/UI/runtime audit.
- Decision: Authoritative UI is Flask/Jinja/static in `app.py`, `templates/`,
  and `static/`; FastAPI is secondary; `frontend/` and TGR are stale. No primary
  authn/authz/CSRF or generated UI service path is proven.
- Evidence: accepted `GHU-000`; exact source paths are in `CONTRACTS.md`.
- Consequences: extend the authoritative surface; fail closed on missing
  security/deployment capability.
- Supersession: none.

## 2026-07-30 — DEC-GHU-000A-CAPTURE-ONE

- Context: source delta after upstream PRs #79/#80.
- Decision: with no reusable receipt, the predictor synchronously starts zero
  or one collector-owned `capture-one` child. There is no interactive timer
  transport, retry, wait, lock steal, or race substitution; terminal handling
  is immediate. Contract budgets and ordering are authoritative.
- Evidence: accepted run `20260730T092421Z-6f4fba42c4-baf67b`, session
  `019fb257-217f-7d10-8409-b2a06a6bd20b`; `CONTRACTS.md`.
- Consequences: R3 tests bind synchronous records and preserve one-request/no-
  retry/stop-first-terminal.
- Supersession: refines DEC-GHU-000-AUTHORITY after #79/#80.

## 2026-07-30 — DEC-GHU-001-REJECTED-EVIDENCE

- Context: rejected contract candidates.
- Decision: rejected candidates remain provenance evidence and are superseded;
  they are not integrated product changes.
- Evidence: review history below.
- Consequences: corrections never silently overwrite or regress accepted work.
- Supersession: none.

## 2026-07-30 — DEC-GHU-001-ORIGINAL-REJECTED

- Context: original run `20260730T075157Z-9be52ecd58-7541d9`, implementer
  `019fb214-6373-7a32-8fae-3f67b1d4ab8c`.
- Decision: rejected for incorrect Level 1–4 authority mapping and vague
  freshness.
- Evidence: reviewer session `794f6fd7bead16cbbb06a94efa9cf4e5`;
  candidate/diff hashes `DATA_MISSING`.
- Consequences: no integration; smallest repair required.
- Supersession: superseded by the repaired successor.

## 2026-07-30 — DEC-GHU-001-REPAIR-REJECTED

- Context: repaired run `20260730T082719Z-9be52ecd58-5c31e1`.
- Decision: rejected for conflating the two collector lanes, omitting explicit
  R2 read-access audit, and leaving unused freshness term D.
- Evidence: implementer/reviewer sessions and hashes `DATA_MISSING`.
- Consequences: no integration; collector lanes and access audit became
  explicit contract requirements.
- Supersession: superseded by GHU-001D.

## 2026-07-30 — DEC-GHU-001D-SOURCE-STALE

- Context: GHU-001D was accepted before #79/#80.
- Decision: preserve it as accepted evidence, but supersede it as source-stale.
- Evidence: commit `6c19b1709e23b21c2b2b66e599e334745a6b1ff3`;
  other session/hash fields `DATA_MISSING`.
- Consequences: current integration must use the post-#79/#80 contract.
- Supersession: superseded by current GHU-001.

## 2026-07-30 — DEC-GHU-001E-LIVE-PROOF-REJECTED

- Context: run `20260730T093002Z-6f4fba42c4-dd989f`, implementer
  `019fb25c-66af-7d50-8b5f-3415a0dd5b42`.
- Decision: #79/#80 were correctly captured, but live proof wrongly required
  mutually exclusive branches; reviewer
  `019fb263-3127-7d03-b951-c55015acb7fb` rejected only that defect.
- Evidence: candidate SHA-256
  `baa3a2bff7e1d81930d9c76726f5455c556c23787415af0042a731c103a9970d`;
  diff SHA-256
  `39d98da03c067c552133aaaef92ed280b73c5e847495dc987a6710030f3e4e10`.
- Consequences: live proof must claim only its naturally reached branch.
- Supersession: superseded by GHU-001F.

## 2026-07-30 — DEC-GHU-001F-ACCEPTED

- Context: GHU-001F changed only the live-proof defect.
- Decision: independently accepted and integrated as current contract.
- Evidence: commit `aa4b45a004b1f897fc5d4cd06b0a741be6cd2446`, parent
  `6f4fba42c45c73702efb017a21cbd284b44c1d04`; file SHA-256
  `b2b4af016b24dafa4f121f0415c0d948a4c0699c73586a235c6547fc3002512b`;
  diff SHA-256
  `22df513812a6460fbae2f247a36bfca4f37536438d5eed6eabebda47b2774b30`;
  implementer `019fb267-ff74-71e3-bb91-2ed8d55be316`; reviewer
  `019fb26b-ae76-7411-91ae-b54d5514a137`.
- Consequences: this is current contract authority.
- Supersession: supersedes GHU-001D and GHU-001E.

## 2026-07-30 — DEC-GHU-PROGRAMME-AUTHORITY

- Context: execution governance.
- Decision: approved programme is authority; contracts, dependencies, and
  tickets may be refined autonomously inside it. Owner's unlimited bounded
  correction-ticket authority supersedes the seed's single-retry limit.
- Evidence: approved plan and current accepted contract.
- Consequences: ordinary review findings are engineering work, not owner
  escalation; parent retains acceptance/integration authority.
- Supersession: supersedes the seed's old procedural retry limit.

## 2026-07-30 — DEC-GHU-R3-PROOF-SPLIT

- Context: synthetic safety and live proof have different evidence scopes.
- Decision: GHU-035 proves fixture/synthetic success and every terminal blocker;
  GHU-037 runs one natural cycle and one exact live job and claims only the path
  naturally reached.
- Evidence: `CONTRACTS.md` release/proof gates.
- Consequences: live proof never induces mutually exclusive branches or retries.
- Supersession: incorporates the GHU-001E correction.

## 2026-07-30 — DEC-GHU-R5-DEFERRED

- Context: programme boundary.
- Decision: R5 training, experiment execution, model persistence/registration,
  activation, and promotion remain deliberately deferred.
- Evidence: approved programme and `CONTRACTS.md`.
- Consequences: R4 produces only non-executing specifications; future execution
  requires a new contract and separate authority.
- Supersession: none.

## 2026-07-30 — DEC-GHU-000C1-INDEX-READ-SAFETY

- Context: rejected `GHU-000C` checkpoint
  `3e9f639dfff62ffddd85aa00bab3d5c6b475cdf6` was not integrated after parent
  focused validation returned `1 failed, 18 passed`.
- Decision: accept only the bounded `GHU-000C1` correction that converts
  named-path/`/proc/self/fd` revalidation errors into deterministic
  `CURRENT_INDEX_PATH_UNSAFE`; do not claim portable non-Linux behavior or
  detection of same-inode concurrent content mutation.
- Evidence: implementer run `20260730T173227Z-3e9f639dff-b1c869`, session
  `019fb415-e17b-7761-84b9-97a96a8fb58d`, child
  `a1704231395179acca71854ce5ba7acb`; parent `19 passed in 0.86s`; reviewer run
  `20260730T173456Z-04c32c37fe-4018cf`, session
  `019fb418-1b23-7fa1-985d-48f0219864e1`, child
  `6132758c7ecce3222bbf0f2059b29c77`, verdict `ACCEPT_GHU_000C1`; final
  integration `c77b3be5ad4aa78b70a9ba89f25ee801d50f27c0`, tree
  `fe5115435d18cbce6be055cf452acdba65518a76`.
- Consequences: fixed index/source reads fail closed on the supported Linux
  descriptor/path replacement boundary; `/proc/self/fd` and `O_NOFOLLOW`
  limitations remain explicit.
- Supersession: supersedes rejected `GHU-000C`; does not erase it.

## 2026-07-30 — DEC-GHU-000B-FIXED-COLLECTOR-INDEX

- Context: audit run `20260730T172346Z-1bacc67937-3c5f6b`, session
  `019fb40d-e8ed-7d40-8ab2-8ad2b156552c`, child
  `4a4c8758a536447e5c001a3a80aa6caa`, verdict
  `GHU_000B_CORRECTION_REQUIRED`, found contract drift after upstream
  `f38a125f6364b8a60d17ae9c971b0ce172874eea`.
- Decision: the collector-owned fixed
  `shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json`
  packet and its `current_race_index_publish` plus sealed refresh-report
  path/SHA chain are authoritative for upcoming/predictor discovery. Predictor
  packet age may be at most 1200 seconds; UI selection separately remains at
  most 300 seconds and strictly pre-jump. Browser/UI has no packet/root/path/
  lock/browser/current-time input or discovery authority.
- Evidence: schema `collector_current_race_index_v1`, canonical bounded packet
  implementation in accepted integration
  `c77b3be5ad4aa78b70a9ba89f25ee801d50f27c0`, and the exact
  source-to-screen/ticket refinements in unaccepted `GHU-000B`.
- Consequences: `GHU-022` reads/adapts only the fixed chain; `GHU-031` uses
  server-owned allowlisted index/root argv; `GHU-035` covers all packet and
  publication failures plus no-path injection; `GHU-037` preserves packet,
  source, and publication hashes.
- Supersession: additively supersedes only source details changed after accepted
  `GHU-001`; it does not reopen or regress that accepted contract.

## 2026-07-30 — DEC-GHU-000B2-REJECTED-AUTHORITY-CORRECTION

- Context: Parent rejected `GHU-000B`, and independent review rejected its
  ledger correction `GHU-000B1`; `GHU-000B2` is the unaccepted review-state
  correction of their decision-ledger evidence.
- Decision: Neither rejected `GHU-000B` nor rejected `GHU-000B1`, including
  `DEC-GHU-000B-FIXED-COLLECTOR-INDEX`, ever became accepted authority. Pending
  parent acceptance of `GHU-000B2`, only the unchanged factual mapping continues
  into review: the collector-owned fixed
  `shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json`
  packet and its `current_race_index_publish` plus sealed refresh-report
  path/SHA chain are the documented source mapping; predictor packet age is at
  most 1200 seconds, UI selection remains separately at most 300 seconds and
  strictly pre-jump, and browser/UI receives no packet/root/path/lock/browser/
  current-time input or discovery authority.
- Evidence: rejected `GHU-000B` ticket and parent rejection evidence in
  `TICKETS.md` and `STATUS.md`; rejected `GHU-000B1` implementer run
  `20260730T174818Z-c550b81f11-c6ce5a`, session
  `019fb424-5860-7bc3-9b71-bd0e88b8880a`, child
  `812c251abd99366b07fc1f9f02f82820`, checkpoint
  `cc65dca19cd4bb9fa6b8c836dc843c0ba00bed7b`, tree
  `009129fc2506a1f9d5d867177279c27c4956d113`, correction diff SHA-256
  `fd2b93a33df15974173f9212449c7ce0c43e22fb303b6112e5d96e20bcd87363`;
  reviewer run `20260730T175213Z-cc65dca19c-10a89a`, session
  `019fb427-fb80-7122-91ad-6cd2987b17c4`, child
  `0f16abbb824ddff3ed7e63f6deba86bc`, verdict `REJECT_GHU_000B1`; current
  review ticket `GHU-000B2`, run `20260730T175638Z-cc65dca19c-c66202`,
  session `019fb42d-217a-7fa0-966f-f36ddabd78d5`, child
  `49e930e41c3a2fa72690df154d3b130c`.
- Consequences: B and B1 remain blocked historical evidence; B2 remains
  unaccepted in review. No acceptance, source/code change, runtime proof,
  prediction, capture, deployment, or terminal `GHU-011`/`GHU-011L` result is
  claimed.
- Supersession: supersedes the operative/authority reading of
  `DEC-GHU-000B-FIXED-COLLECTOR-INDEX` and links `GHU-000B`, `GHU-000B1`, and
  `GHU-000B2`; preserves that old entry append-only as rejected history and
  preserves only its unchanged factual fixed-index mapping for B2 review
  pending parent acceptance.

## 2026-07-30 — DEC-GHU-000B3-STALE-CURRENT-POINTER-CORRECTION

- Context: Independent reviewer run `20260730T180236Z-3271721e5b-bed535`,
  session `019fb431-732f-7dc2-ade5-45803721691c`, child
  `fcd20a35022ea048ded40777f364b710`, verdict `REJECT_GHU_000B2`, rejected
  `GHU-000B2` solely because the `GHU-000B` Next safe action still directed
  independent review of rejected `GHU-000B1`; all other review axes passed.
- Decision: `GHU-000B2` is blocked. `GHU-000B3` changes only that stale
  current pointer and the necessary rejection/closeout bookkeeping, while
  preserving B2's independently-passed fixed-index mapping and authority
  correction pending parent acceptance of B3.
- Evidence: B2 implementer run `20260730T175638Z-cc65dca19c-c66202`, session
  `019fb42d-217a-7fa0-966f-f36ddabd78d5`, child
  `49e930e41c3a2fa72690df154d3b130c`, checkpoint
  `3271721e5b19bc795f775a00a608c557f85b0112`, tree
  `cc52608ffff88ce784c08da3236b55c82ec753fd`, binary diff SHA-256
  `0893bde44d28fd9bf24795771947374dca82a3654d76dfdd979dbbb2f85fdc6d`;
  B3 run `20260730T180514Z-3271721e5b-a531a0`, session
  `019fb433-d9c0-7b22-b36d-4a832833e4ce`, child
  `50f7490271d4dd156e56032b72ddb9ab`.
- Consequences: No substantive source, product, code, test, runtime, data,
  fixed-index mapping, accepted-contract, or R1 history change is made. No
  acceptance, deployment/runtime proof, prediction/capture, or terminal
  `GHU-011`/`GHU-011L` result is claimed.
- Supersession: This append-only entry records B2's rejection and B3's narrow
  correction state; it does not delete or rewrite prior decision history.

## 2026-07-31 — DEC-GHU-R1-ACCEPTED-ATOMIC-FIXTURE-TRANCHE

- Context: Independent review verdicts for `GHU-000B3`, `GHU-010H`, and
  `GHU-011M`, followed by parent acceptance and the terminal R1 gate review.
- Decision: Accept `GHU-000B3`, `GHU-010H`, and `GHU-011`. An accepted
  delivery may be atomically bound by an independently reviewed exact frozen
  product/docs/test checkpoint and a subsequent independently reviewed,
  parent-integrated ledger-only closeout recording its identities, validation,
  verdict, and parent decision. Both require parent inspection and acceptance/
  integration; together they form the atomic evidence binding without requiring
  a self-referential single commit. The closeout changes only programme ledgers,
  changes no product/test bytes, weakens no evidence, preclaims no parent
  acceptance, and erases no rejected history. A successor cannot leave
  `planned` before that closeout is independently reviewed and parent-integrated;
  a successor `ready` state recorded in that reviewed closeout becomes durable
  and effective only when the parent integrates the closeout, so an unintegrated
  candidate neither preclaims acceptance nor mutates durable programme state.
  Ordinary knowable state transitions remain in the product delta. Close the
  R1 gate at product checkpoint
  `e10cff293141569b1a5a169dd05efc8109e3c603`, tree
  `07f02fc46b88b47bf0ade8ee264505f8b47c7d91`; and ready `GHU-012` through
  `GHU-015` as one atomic coupled fixture tranche. Each tranche ticket has
  accepted `GHU-011` as its sole prerequisite, no same-tranche ticket is
  another's prerequisite, and `GHU-015` acceptance requires all four tranche
  surfaces in the same frozen delta. `GHU-016` remains planned after
  `GHU-015`.
- Evidence: `ACCEPT_GHU_000B3`, reviewer run
  `20260730T180925Z-44fe9a0875-a94ad1`, session
  `019fb437-b8d1-7dd3-9e06-f4494603e9d7`, integration
  `6e0b0d99c296a4c984faf0775bab88f8689e66da`, tree
  `bff53978cdfeea8f604404432e1d672cba95a692`; `ACCEPT_GHU_010H`, reviewer
  run `20260730T135324Z-13cf3a3b54-ca20f6`, session
  `019fb34d-4cd5-7dd2-b7be-d7a8d7c745ff`, integration
  `1bacc679377f54433ea757f8cbf7045e3ce8526a`, tree
  `dee68158b8455d898d60807bdc0ff41c8caf1f7f`; `ACCEPT_GHU_011M`, reviewer
  run `20260730T181807Z-e10cff2931-e22d8c`, session
  `019fb43f-a9bd-7363-851c-d6a392a44548`; focused pytest `5 passed`,
  Playwright `3 passed`, classifier `full_forecasting`; gate reviewer run
  `20260730T222316Z-86f39d54d3-b5f20d`, session
  `019fb520-39bf-7be0-863e-7c3b8ced34bf`, verdict A `ACCEPT`. The first broad
  run exited 1 with `1 failed, 550 passed, 40 subtests passed`; diagnostic
  target/module/full passed 1, 25, and 551 tests with 40 subtests passing in
  the full run, while root cause remains `UNKNOWN`. Isolated diagnostic
  checkpoint `86f39d54d3949c7bc2b6f670c809a6e5dea5050d` is excluded and unmerged
  from integration and retained as isolated diagnostic evidence.
- Product identity: base `6e0b0d99c296a4c984faf0775bab88f8689e66da`, tree
  `bff53978cdfeea8f604404432e1d672cba95a692`, to head/integration
  `e10cff293141569b1a5a169dd05efc8109e3c603`, tree
  `07f02fc46b88b47bf0ade8ee264505f8b47c7d91`; exact paths `app.py`,
  `static/css/operator-ui.css`, `templates/operator_ui.html`,
  `templates/operator_ui_components.html`,
  `tests/playwright/operator-ui-shell.spec.js`, and
  `tests/test_operator_ui_shell.py`; binary diff SHA-256
  `7641c45d98a26590e64fff71a50d9dc1b7639b03ee1e6c1fb05aedbecd58681b`.
  Parent accepted this exact product checkpoint. Publication, push, PR,
  default-branch merge, deploy, runtime mutation, and live proof are
  `NOT_OCCURRED`.
- Consequences: One fresh bounded implementer owns atomic `GHU-012`–`GHU-015`,
  followed by independent exact-delta review. Supported claims remain limited
  to R0 contracts, the fixed collector-owned current-race index, and the R1
  fixture dashboard. No push, PR, default-branch merge, deployment, runtime
  mutation, live proof, deployed/live/authenticated dashboard, R2+ proof,
  prediction/runtime/deployment proof, training, promotion, EV, staking,
  betting, or public claim occurred.
- Supersession: Supersedes the pending/current-state reading of
  `DEC-GHU-000B3-STALE-CURRENT-POINTER-CORRECTION`; preserves all prior entries
  and detailed history.

## 2026-07-31 — DEC-GHU-016-UX-FREEZE

- Context: Accepted R1 product checkpoint and independent visual/UX review.
- Decision: Freeze fixture labels **RESEARCH ONLY — NOT FOR BETTING** and
  **PROTOTYPE DATA**; navigation is Dashboard, Race, Lifecycle, Result,
  Collector, Corpus, Models, System, Audit. Advanced source/evidence controls
  remain disclosure/details. Mobile remains stacked, with dense tables
  contained or scrollable and primary actions/warnings visible. No blocking
  workflow, truthfulness, security, identity, or evidence finding exists.
- Evidence: Integrated checkpoint `51fe070ba2a0778bca0b0334c00cae9d75561952`,
  tree `0c2b82a391b8dd0a1dcc525cf4203edc910843c1`; binary diff SHA-256
  `3f74ec86de1ab68b0fbb1a13125efa6fa416e3cc1fdc9bfc658f97118d3dc135`;
  reviewer run `20260731T001624Z-51fe070ba2-937103`, child
  `eda938ef8b7cf07d87eb37719e0929c2`, verdict `PRODUCT ACCEPT` / `GATE ACCEPT`.
  Focused pytest `5 passed in 0.50s`; Chromium desktop `5 passed in 6.9s`;
  classifier `full_forecasting`. Desktop screenshot SHA-256
  `fc181de498c5ae287343cfc6026ef4e0dafcb9bb51ac91a605af37fab8d06ba6`;
  mobile `e841e7953ea63cfb278837511ded73a1298c78159a7f575f92fac4b00962896d`.
- Consequences: IMPORTANT R2 follow-up is responsive stacked-card or
  equivalently readable mobile treatment for dense runner/model/governance
  tables while preserving exact identities. OPTIONAL polish is narrowing the
  operator path predicate to exact supported routes and replacing positional
  fixture tuples with named fields. GHU-020 owns only the foundational finite
  evidence envelope and primitives; GHU-020A precedes GHU-021 for auth/session/
  CSRF and separate append-only UI access audit.
- Supersession: None; append-only UX freeze decision.

## 2026-07-31 — DEC-GHU-021-ACCEPTED-INTEGRATION

- Context: Final independent review and parent integration of the versioned
  read-only operator API.
- Decision: Accept `GHU-021` at child HEAD
  `8db34fe53af252fdb6dd743b51d3531fb1f8b618`, tree
  `af89578953145e5049bb9d2c70f3de150fad86ca`, and integrate that exact tree at
  parent commit `4a24218379d186d951f47d3fcf0d17d396d7d066`. Preserve the programme
  correction chain and do not invent a terminal broad-gate result.
- Evidence: Final correction diff from
  `f1f1bd96c60d690bb2a7247e79db4cffc6360594` SHA-256
  `d18f42206bfc4a103d4ae352f93f3963f54c62820de7cb5646dbb1860a67dafa`;
  full accepted four-path diff from
  `d71857e232ce7371280f9e5c56c45be7b9f7f9e5` SHA-256
  `037adf12c9d37c0e96a66e65645392a06482378262ac8b496c0b662253b860ea`;
  parent focused gate `254 passed in 39.77s`; reviewer run
  `20260731T100618Z-8db34fe53a-a6a1cc`, session
  `019fb7a3-c1d4-7da1-946e-291651763774`, verdict `ACCEPT_GHU_021`.
  Classifier selected `full_forecasting` because paths default unknown-to-full;
  the exact 551-test broad gate subsequently passed `551 passed in 1653.94s
  (0:27:33)`, exit 0.
- Consequences: `GHU-021` remains closed and accepted. No push, PR,
  default-branch merge, deployment, runtime mutation, or live proof occurred.
- Supersession: None; append-only accepted integration record.

## 2026-07-31 — DEC-GHU-022P-RUNNER-SEALED-INDEX

- Context: Read-only audit run `20260731T095910Z-d71857e232-648124`, session
  `019fb79d-352a-7c62-88a7-588974824079`, verdict
  `PREREQUISITE_REQUIRED`, proved the fixed current-index packet is runnerless.
- Decision: Insert `GHU-022P` inside R2 and before `GHU-022`. Keep the sole
  fixed filename and bounded v1 predictor compatibility, but make v1
  catalog-ineligible. Collector-owned v2 must seal a canonical ordered final
  active runner set and its exact race/pre-race source/hash chain during the
  existing refresh/download flow; read/publication/UI paths may neither create
  nor guess that evidence.
- Evidence: `collector_current_race_index_v1` and
  `_normalize_current_index_rows` retain race/time identity only;
  `race_window_record`/`selected_races` contain no sealed runners;
  `current_race_index_publish` omits packet SHA-256; predictor request handling
  accepts `participants`/`runners`, while bounded discovery strips them.
  Existing canonical-alignment and runner-completeness helpers supply accepted
  leakage-safe box/name, final-active, scratch/ambiguity, and normalization
  evidence when the source actually contains it; no source-native ID is
  presumed.
- Consequences: `GHU-022` now depends on accepted/integrated `GHU-022P` and
  may adapt only a fully matched v2 `PUBLISHED` chain. R2 remains in progress;
  R5 remains deferred. `GHU-000` and `GHU-021` are not reopened. No new fetch,
  browser, scan, lock, caller path, retry, collector, canonical write, runtime
  action, training, promotion, EV, staking, or betting is authorized.
- Supersession: Additively refines only the current-index source contract for
  `GHU-022`; preserves v1 predictor behavior and all prior history.

## 2026-07-31 — DEC-GHU-023P-PRODUCER-SEALED-BUNDLE-INDEX

- Context: Accepted read-only audit verdict `PREREQUISITE_REQUIRED` found the
  current on-demand v1 bundle and replay verifier insufficient as a bounded UI
  catalogue/read contract. No further run or session identity is verified.
- Decision: Insert ready `GHU-023P` inside R2 before `GHU-023`. The producer
  owns one fixed-root atomic canonical index, one fixed producer-only bounded
  index-update lock, and strict v2 result/manifest/index schemas. Lock
  contention and stale locks fail closed without stealing, and this authority
  cannot touch any collector lock. Detail verification is bounded and
  descriptor-relative with held identities; exact bundle membership is the
  manifest regular-file set plus only its derived parent-directory set, with
  bounded per-directory enumeration and no recursive walk. Seal stable
  prediction and nullable job identity, complete
  terminal race/model/config/time/stage evidence, and a canonical aggregate
  logical hash over manifest payload/entries without requiring a self-hash.
- Evidence: Current bundles live beneath
  `artifacts/on_demand_prediction_runs` in
  `prediction_<timestamp>_<12hex>` directories. The producer emits
  `on_demand_race_prediction_v1` and
  `on_demand_prediction_bundle_manifest_v1`; its replay verifier detects
  changed/missing/added/symlink content but recursively walks caller-selected
  paths. The manifest excludes itself, results expose an absolute bundle path,
  market-only can truthfully lack a model artifact, and blocked bundles do not
  consistently seal exact identity, generated time or protocol/validation/
  scoring stage.
- Consequences: `GHU-023` depends on accepted/integrated `GHU-023P` and may
  adapt only fixed-index, independently verified v2 bundles. Old/unindexed v1
  bundles remain replay-compatible but UI catalog-ineligible and unchanged.
  Listing freshness never makes historical content current. No replay,
  outcome/result access, bundle rewrite, path input/scan, acquisition,
  service/browser/collector/runtime action or lock action beyond the fixed
  producer index-update lock, training, promotion, EV,
  staking or betting is authorized. `GHU-021` remains accepted; its broad gate
  passed `551 passed in 1653.94s (0:27:33)`, exit 0.
- Supersession: Additively refines only the prediction-bundle source contract
  for `GHU-023`; preserves all accepted history and current v1 replay evidence.

## 2026-07-31 — DEC-GHU-021V-BROAD-GATE-TERMINAL-GREEN

- Context: The already-running classifier-selected `GHU-021` broad gate
  reached a terminal result after `GHU-021` acceptance and integration.
- Decision: Record the terminal validation truth without reopening `GHU-021`
  or changing programme ticket counts, statuses, dependencies, or scope.
- Evidence: Exact command
  `PYTHONDONTWRITEBYTECODE=1 TMPDIR=/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/operator-ui-validation-tmp/ghu021-8db34-full uv run --no-project --with-requirements requirements/all.in python -m pytest -q --noconftest tests/race_collection`
  exited 0 with `551 passed in 1653.94s (0:27:33)`. The validation caused no
  repository, product, runtime, or data mutation.
- Consequences: `GHU-021` remains accepted and integrated. This is validation
  evidence only; product scope and all accepted ticket histories are unchanged.
- Supersession: Additively closes only the previously running exact `GHU-021`
  551-test broad-gate record.

## 2026-08-01 — DEC-GHU-R3-PERSISTED-POLLING

- Context: `GHU-032 + GHU-033 + GHU-034` require reconnectable job progress;
  the accepted application has no SSE or broker infrastructure.
- Decision: Use bounded same-origin polling of the accepted persisted job/event
  store. Disconnect and refresh only reread the actor-owned job; they never
  cancel, retry, requeue, substitute, or fabricate progress. Keep R3 default
  off until explicitly composed with exact server-owned resolver, store,
  launcher, and strict result-reader services.
- Evidence: Accepted `GHU-030`/`GHU-031` immutable store and one-attempt worker
  at commit `dee082e954038c9ac4bf48d48bbe3901879310b8`, tree
  `302e144765d9bfd7ea3a7a1ef8a25e4fd3ab2c41`; current unaccepted coupled R3
  review candidate and focused validation recorded in `STATUS.md`.
- Consequences: No SSE, broker, timer-derived progress, new process boundary,
  or rerun control is introduced. Ranked probabilities remain absent unless
  the accepted strict verifier has produced `PREDICTION_READY` and the
  configured verified-result reader returns the exact verified schema.
- Supersession: Additively implements the accepted GHU-033 polling fallback;
  it does not claim candidate acceptance, integration, deployment, or runtime
  proof.

## 2026-08-01 — DEC-GHU-032-034-REJECTED-CANDIDATE

- Context: Independent review of the original coupled `GHU-032`/`033`/`034` candidate.
- Decision: The candidate is rejected/blocked, not accepted. Correction is bounded to `GHU-032C1` through `GHU-034C1`.
- Evidence: commit `8262553eafc1dc3513efebe6f0f1d68cc58e044b`, tree `c2a31048fa043e28f1deeccfaa55b8dde9e0fd59`, reviewer session `019fba4a-d4de-7152-820e-504a013f7a85`.
- Consequences: Seven unresolved findings require correction: durable pre-POST intent; atomic launch and failure closure; full sealed-result binding; server-owned R3 capability; model-specific catalogs; default-off repository startup factory; and bounded non-overlapping reconnect.
- Supersession: Does not alter accepted `GHU-030`/`GHU-031` evidence. The correction remains unaccepted pending independent review and parent decision.

## 2026-08-01 — DEC-GHU-032C1-034C1-REJECTED

- Context: Independent review of C1 against the original rejected coupled R3 candidate.
- Decision: C1 is rejected/blocked and the correction proceeds only as `GHU-032C2` through `GHU-034C2`.
- Evidence: C1 commit `6a4e193a6677210b0dc8d406bb8c64df72230f86`, tree `7d5fd4da39963a01d82263fb2fbe2fb393c17744`, binary diff SHA-256 `e43fcdfc3c5b32026de241b18d07669d6599be5537390a64c123669f58508806`; implementer `019fba50-88d9-7ad2-b727-b0067f0481dc`; reviewer `019fba5e-ac63-7c21-b109-af800153b461`, verdict `REJECT/CORRECTION_REQUIRED`; focused `330 passed`, Operator UI `807 passed`, static gates passed; broader race-collection validation interrupted after an observed failure without a terminal count/pass.
- Consequences: Preserve accepted `GHU-030`/`GHU-031` evidence and both rejected predecessor deltas. `GHU-035` depends on accepted C2, not the blocked originals. No acceptance, integration, publication, deployment, or runtime proof is claimed.

## 2026-08-01 — DEC-GHU-032C2-PARENT-PRE-REVIEW-RETURN

- Context: Parent inspection returned C2 for five narrow repairs before independent review.
- Decision: Repair only GET-observed WAITING recovery, authenticated sealed-request binding, executable reconnect/accessibility coverage, fixed job-store object safety, and cleanup/ledger truth. Preserve the single terminal broad result as environment/path-gate evidence on rejected bytes; it is not an owner blocker and is not rerun.
- Evidence: Focused Python `564 passed`; executable Node state machine `7 passed`; updated full Operator UI `814 passed in 88.84s`; Python/Node syntax and `git diff --check` passed. The preserved broad invocation exited 2 after `1 failed, 389 passed in 650.55s` when the race-collection operator rejected the run-worktree basetemp, then ended with `KeyboardInterrupt`.
- Consequences: C2 may enter `REVIEW` only as an unaccepted candidate. Independent review and parent acceptance/integration remain pending; no deployment, runtime proof, live action, training, promotion, EV, staking, or betting is authorized or claimed.

## 2026-08-01 — DEC-GHU-032C2-034C2-REJECTED-C3-CORRECTION

- Context: Independent C2 review session `019fba90-d645-7361-a1f2-2ddf993b732b` returned five residual defects.
- Decision: Preserve original/C1 rejection and C2 candidate history; correct only those five defects as `GHU-032C3` through `GHU-034C3`.
- Consequences: C3 is an unaccepted review candidate. Parent retains acceptance, integration, publication, deployment, and runtime-proof authority.
## GHU-032C4 retained disclosure authority

Probability disclosure authority is the descriptor-retained sealed-v2 bundle alone. The bundle contains and independently binds the complete collector protocol chain and authenticated cutoff/history snapshot; mutable collector roots and the live database are not disclosure inputs.

The repository-v1 binding is output of the finite GHU-036 generator contract and binds generator identity, deployment identity, the checked-in profile digest, and fixed artifact hashes.

## 2026-08-01 — DEC-GHU-032C4-C5-REJECTED-C6-CORRECTION

- Context: Parent review rejected C5 only for a request-schema downgrade, incomplete coherent-snapshot and mutation evidence, incomplete generated-binding behavior, and connected-only untested validators.
- Decision: Preserve accepted C5 ownership/bootstrap/job-worker boundaries and correct only those residuals as C6. Require exact real protocol objects unconditionally, fixed bounded descriptor-retained snapshot enumeration, pure shared endpoint validators, and exact generated identity syntax.
- Evidence: Frozen C5 commit `90502a360a184a6ba9b301245c93b818494647cd`, tree `f16d12db919fd749b5118ea3b800db21eb2aaec4`, session `019fbacc-623e-7e71-bce6-ee79e646dec5`, delta SHA-256 `c3eec6356d23becdfadaa1e38c7c5c368c9514743d43d556c6438d2bc53ce168`. C6 validation is recorded in implementer closeout, not self-approval.
- Consequences: C6 remains an unaccepted local candidate pending independent parent inspection and review. No runtime action, deployment, outcome access, training, promotion, or wagering output is authorized or claimed.

## 2026-08-01 — DEC-GHU-032C6-REJECTED-C7-CORRECTION

- Context: Parent rejected C6 for missing independently attributable attacker-reseal evidence across sealed-v2 semantic relations and missing valid-but-wrong generated deployment identity behavior.
- Decision: Preserve C6 as rejected and correct only that evidence gap as GHU-032C7. Production changes are allowed only where a new mutation test first demonstrates a fail-open verifier or startup-binding defect.
- Consequences: C7 is an implementation candidate awaiting parent inspection and independent review. It is not accepted, integrated, published, deployed, runtime-proven, or live-proven.

## 2026-08-01 — DEC-GHU-032C7-REJECTED-C8-CORRECTION

- Context: Independent review session `019fbb00-37f9-7291-8741-8ca19878cf1a` found that the authentic 13-member history seal published by `seal_history_database` could not pass the reduced nine-member verifier schema.
- Decision: Preserve C7 as rejected for producer/verifier schema divergence and correct only that coherence defect as `GHU-032C8`.
- Consequences: C8 is a candidate awaiting parent inspection and independent review. C3-C7 security boundaries remain authoritative. No acceptance, integration, publication, deployment, runtime/data mutation, outcome access, training, promotion, EV, staking, or betting is claimed.

## 2026-08-03 — DEC-GHU-035-REJECTED-C1-CORRECTION

- Context: Parent rejected the GHU-035 three-path candidate from run `20260803T071516Z-3c92515a29-6015a7`, session `019fc67a-bc8c-7fe3-8ee3-e48607169c08`, child `f3da1a072215067e9f5f21a65db7d7bb`, constructed diff SHA-256 `dd047a5d1043b7472a6161389d89b79220d4c38d98ba4b11deee88d37470c5ca`.
- Decision: Preserve GHU-035 as rejected and correct only its false blocked-state bookkeeping as `GHU-035C1`. Unavailable child pytest is `DATA_MISSING`, not a product/programme blocker. The correction binds genuine lower-level emitted classifications and fixtures to one real API/store/worker integrated seam and remains `REVIEW`.
- Evidence: Accepted base commit `3c92515a29a566dd2136a5f954984f41236c0fdb`, tree `eba05242bf2cb665e43ed94331095130a51ff414`; C8 reviewer run `20260803T071114Z-3c92515a29-3e2573`, session `019fc676-aead-7fc0-8986-9a021dc38ca3`, verdict `ACCEPT`.
- Consequences: No production change, GHU-036, deployment, runtime/data/model mutation, collector/browser action, outcome access, training, promotion, EV, betting, acceptance, integration, staging, or commit is claimed.

## 2026-08-03 — DEC-GHU-035C1-REJECTED-C2-CORRECTION

- Context: Parent executable validation of C1 commit `49cb6e5328c9aaa9ad3787327a770e5c7a74744f`, tree `b68d34c9310981f582706fa18144b86be9521e28`, from implementer run `20260803T072530Z-3c92515a29-e984b8`, session `019fc683-afe7-7f52-92d8-ae11648ddacf`, ran `uv run --with-requirements requirements/all.in pytest -q tests/operator_ui/test_r3_e2e_safety.py` and returned exit 1 with `1 failed, 1 passed`; the integrated proof's first POST expected 202 and received 503.
- Decision: Preserve C1 as rejected evidence and correct only its fixture identity as `GHU-035C2`. The fixture supplied collector display identity `Race 5 - RICH - 2026-08-01` to the real Level-2 operation-audit seam instead of the required protocol `race-` identity, causing a fail-closed audit rejection before dispatch. No reachable production defect is established.
- Consequences: C2 is `REVIEW`, pending parent executable validation, freeze, and independent review. The 23 genuine fixture bindings, one submission/one invocation, duplicate reuse, exact worker blocker, and prohibition on false `PREDICTION_READY` remain unchanged. No production change, GHU-036, deployment, live action, runtime/data/model mutation, outcome access, training, promotion, EV, betting, staging, commit, push, or merge is authorized or claimed.
