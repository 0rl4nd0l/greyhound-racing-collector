# Greyhound Operator UI V1 ticket manifest

Schema: `greyhound-operator-ui-ticket-seed-v1` from the approved bundle. Status vocabulary is exactly `planned`, `ready`, `active`, `review`, `accepted`, `blocked`, and `deferred`. Accepted prerequisites mean parent-accepted integration, never child output alone. Legal transitions and closeout rules are defined in `PLAN.md`. Rejected candidates remain evidence and use the smallest correction ticket; they never overwrite or regress accepted prerequisites.

## GHU-000 — Authoritative repository, UI and runtime-surface audit

- Release: `R0`
- Priority: `P0`
- Status: `accepted`
- Dependencies: none
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: Fresh Codex X read-only explorer; parent Codex integrates findings
- Outcome: Proves which current web application, routes, assets, test runner, authentication, service generator and runtime artifacts are authoritative.
- Scope: Reverify remote master SHA/tree and working-tree cleanliness.; Read AGENTS.md plus current Greyhound project/evidence contracts.; Inventory active server entry points, templates/static assets or SPA packages, API routes, authentication, CSRF and browser-test infrastructure.; Trace manual predictor CLI, finite model/config catalog, collector request protocol, status packets and deployment generator.; Identify all existing components that can be reused and all stale/duplicate UI surfaces.
- Non-goals: No code edits, dependency changes, service actions, live predictions or data writes.; Do not choose a new frontend stack before the inventory is complete.
- Acceptance: One source-to-screen map names the exact current files and data owners.; One decision states which UI surface will be extended and why.; Exact available validation commands and CI classifier behavior are recorded.; Unknown or conflicting runtime/UI authority is reported as DATA_MISSING rather than guessed.
- Validation: Read-only Git/repository checks.; Route and asset enumeration.; Existing test discovery without broad test execution.
- Risks: Material risk: Stop if the authoritative UI or deployment path cannot be established, or if current master differs in a way that invalidates the supplied architecture. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop if the authoritative UI or deployment path cannot be established, or if current master differs in a way that invalidates the supplied architecture.
- Authority: Documentation/read-only repository evidence; parent accepts and integrates.
- Claims supported: The accepted, integrated outcome and its narrow evidence-backed claims.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Preserve accepted evidence; no further ticket action.
- Closeout evidence: Accepted integration evidence recorded in STATUS.md/DECISIONS.md.

## GHU-001 — Operator UI product, evidence and authority contract

- Release: `R0`
- Priority: `P0`
- Status: `accepted`
- Dependencies: `GHU-000`
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: Fresh Codex X documentation implementer
- Outcome: Freezes what the UI may display, what it may control, and which evidence makes every status truthful.
- Scope: Define operator personas and Levels 1–4 authority.; Define source, freshness, identity and claims requirements for every dashboard card.; Define permitted UI mutations and forbidden operational controls.; Define prototype, read-only, manual prediction and research-management release gates.; Define exact research-only and no-betting language.
- Non-goals: No visual implementation, API implementation, deployment or runtime action.; No decision to activate training or promotion.
- Acceptance: Every visible metric has a named authoritative source and freshness rule.; Every action has an explicit authority level and audit requirement.; Ambiguous/missing/stale states are specified.; The contract preserves race/runner/source/timestamp provenance and temporal separation.
- Validation: Review against the current equivalents named in `CONTRACTS.md` and the accepted source/evidence/authority contract.; Independent claims-boundary review.
- Risks: Material risk: Stop on unresolved authority over service control, canonical writes, outcome access, training or promotion. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop on unresolved authority over service control, canonical writes, outcome access, training or promotion.
- Authority: Documentation/read-only repository evidence; parent accepts and integrates.
- Claims supported: The accepted, integrated outcome and its narrow evidence-backed claims.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Preserve accepted evidence; no further ticket action.
- Closeout evidence: Accepted integration evidence recorded in STATUS.md/DECISIONS.md.

## GHU-002 — Ticket manifest, status ledger and decision log

- Release: `R0`
- Priority: `P0`
- Status: `accepted`
- Dependencies: `GHU-000`, `GHU-001`
- Model routing: GPT-5.6 Terra; fallback GPT-5.6 Luna
- Session role: Fresh Codex X mechanical implementer
- Outcome: Creates the persistent execution surface that lets parent Codex run the programme ticket by ticket without losing decisions or reopening completed work.
- Scope: Create repository-conventional PLAN, TICKETS, STATUS and DECISIONS artifacts.; Seed this plan's tickets with dependency, authority and release-gate metadata.; Add supported/unsupported claims and next-safe-action fields.; Define ticket state transitions and closeout requirements.
- Non-goals: No product code, UI code, runtime mutation or speculative follow-up tickets.
- Acceptance: Every seeded ticket has one outcome, scope, non-goals, acceptance, validation, risks and stop conditions.; Dependent tickets cannot start before accepted integration of prerequisites.; Status distinguishes planned, ready, active, review, accepted, blocked and deferred.
- Validation: Schema or lint validation used by existing task-card conventions.; Exact diff review.
- Risks: Material risk: Stop if the repository already has an authoritative backlog/control format that conflicts with the proposed files; adapt to it rather than creating a parallel system. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop if the repository already has an authoritative backlog/control format that conflicts with the proposed files; adapt to it rather than creating a parallel system.
- Authority: Documentation/read-only repository evidence; parent accepts and integrates.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Independently review and parent-integrate the corrected R1 ledger closeout, then proceed to `GHU-020`.
- Closeout evidence: Parent accepted correction `GHU-002A` after independent review. Accepted integration commit `73f1e5d041f8d78ee0f48ce13e008f71c20090ca`; parent `aa4b45a004b1f897fc5d4cd06b0a741be6cd2446`; tree `d68116ba72b28f149707d7610821aea02cb35781`; frozen/reviewed base-to-result diff SHA-256 `34bbf4d1b6e6286b143d296183d35064909a614f8bef9dc90553c639b787fdd3`; implementer session `019fb27d-891e-75f1-aea2-fac1565f1db4`; independent reviewer session `019fb285-4b56-7202-a298-206855e4c875`, verdict `ACCEPT_GHU_002A`. Parent decision: accepted the exact reviewed four-file delta, committed it mechanically, and fast-forward integrated it into the clean canonical branch on 2026-07-30. Accepted file SHA-256: `PLAN.md` `6fba2ff2d028f0ae92ee85fe0acb556d4bd70fb19b5129db8b6a5a83828c5e0e`; `TICKETS.md` `9148ed8b3df8ac51d3e157ee1454ae9979d4113c79b3612186de640806c02e92`; `STATUS.md` `e77e671e786da138ee80f157e36ec91235508d12c9ff0921e29329e8465be208`; `DECISIONS.md` `107c05734e4d1ff780c136270afb2331eea69c32ed930ff9cbebf1347656c3bf`. Publication, PR, merge to the repository default branch, deployment, and runtime proof: `NOT_OCCURRED`.

## GHU-000B — Correct collector-owned current-race-index source drift

- Release: `R0`
- Priority: `P0`
- Status: `blocked`
- Dependencies: `GHU-001`, `GHU-000C1`
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: Fresh bounded Codex X documentation implementer
- Outcome: Additively corrects source details changed by upstream `f38a125f6364b8a60d17ae9c971b0ce172874eea` without reopening accepted `GHU-001`.
- Scope: Correct the historical merge-parent identity.; Record later upstream source drift and local merge/current integration identities.; Make the fixed collector-owned current-race-index packet and publication/source chain authoritative for upcoming/predictor discovery.; Refine only materially affected tickets and ledgers.
- Non-goals: No product/code/test change, runtime action, acceptance, publication, merge, deployment, prediction, or broad programme change.
- Acceptance: Contract records schema `collector_current_race_index_v1`, canonical bytes, maximum 32 rows, source path/SHA, timezone-aware generation, maximum age 1200 seconds, exact TheDogs identity/uniqueness, publication statuses, and fail-closed `CURRENT_INDEX_*` handling.; P-UPCOMING-300-PREJUMP remains separately stricter.; UI cannot inject or display path/root/lock/browser/time inputs.; GHU-000C/C1 history and current identities are exact.
- Validation: Exact base/path checks, exhaustive stale-SHA/source scans, state/count/dependency/cross-reference checks, and `git diff --check` only.
- Risks: Material risk: accidentally reopening accepted GHU-001, weakening the 300-second/pre-jump UI rule, or granting browser/path authority.
- Stop conditions: Stop on product/code/test drift, unresolved identity, or need for runtime/external action.
- Authority: Documentation-only additive correction; parent owns review, acceptance, integration, publication, and deployment.
- Claims supported: Only the preserved rejected documentation checkpoint and its evidence-backed source mapping.
- Claims unsupported: Acceptance, terminal GHU-011 result, deployment/runtime proof, prediction, training, promotion, betting, or runtime/data mutation.
- Next safe action: Preserve this blocked predecessor and the accepted `GHU-000B3` correction evidence; no further predecessor action.
- Closeout evidence: Verified clean base `c77b3be5ad4aa78b70a9ba89f25ee801d50f27c0`, tree `fe5115435d18cbce6be055cf452acdba65518a76`. Transitioned `planned -> ready -> active -> review` in run `20260730T173844Z-c77b3be5ad-37e474`, session `019fb41b-abcb-7110-8b5d-1c6f7857758e`, child `cb1c337f1c00368a51b58ba87b4ccdbe`. Parent rejected the candidate and it transitioned `review -> blocked`; preserved checkpoint `c550b81f111d0e053c1c3dd6014ef0f28b7638c1`, tree `179300b2ddff681a52c9f7ae6fdffbf2c0137c15`, five-file diff SHA-256 `0ea14c1bafef9ca8917a87ac9a1836e4d733e5c36f83543c962a58601add2835`. Exact findings: abbreviated evidence hashes; false missing-session statements; incorrect predecessor failure classification; accepted `GHU-000C1` depending on blocked `GHU-000C`; redundant `GHU-000C1` dependencies in `GHU-035` and `GHU-037`; incorrect new-ledger dates; and missing rejection/correction-ticket ledger state. Audit provenance: run `20260730T172346Z-1bacc67937-3c5f6b`, session `019fb40d-e8ed-7d40-8ab2-8ad2b156552c`, child `4a4c8758a536447e5c001a3a80aa6caa`, verdict `GHU_000B_CORRECTION_REQUIRED`. Parent decision: `REJECTED`. Commit/push/PR/merge/deploy/runtime proof: `NOT_OCCURRED`.

## GHU-000B1 — Correct rejected GHU-000B ledger evidence

- Release: `R0`
- Priority: `P0`
- Status: `blocked`
- Dependencies: `GHU-001`, `GHU-000C1`
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: Fresh bounded Codex X ledger-correction implementer
- Outcome: Corrects only the three ledgers after parent rejection of `GHU-000B`.
- Scope: Correct `TICKETS.md`, `STATUS.md`, and `DECISIONS.md` evidence identities, failure wording, dependencies, dates, counts, transitions, and cross-references.
- Non-goals: No `PLAN.md`, `CONTRACTS.md`, product, code, test, runtime, data, database, lock, browser, service, prediction, capture, acceptance, commit, push, merge, deployment, or source-claim change.
- Acceptance: Exact parent findings are corrected without changing preserved contract bytes.; No accepted ticket depends on a blocked ticket.; Counts, transitions, dates, identities, and cross-references agree.
- Validation: Exact base/tree and allowed-path checks.; `PLAN.md` and `CONTRACTS.md` byte identity versus base.; Full-hash/no-ellipsis, failure-wording, dependency-graph, count, transition, cross-reference, and date-scope scans.; `git diff --check`.
- Risks: Material risk: altering the sound fixed-index contract substance or R1 running/failure history.
- Stop conditions: Stop on any need to edit outside the three allowed ledgers or to perform product/runtime/test/external action.
- Authority: Documentation-only ledger correction; parent owns review, acceptance, integration, publication, and deployment.
- Claims supported: Only this preserved rejected three-ledger correction and its exact evidence.
- Claims unsupported: Acceptance of `GHU-000B1`, acceptance of rejected `GHU-000B`, changes to `PLAN.md`, `CONTRACTS.md`, or source, terminal `GHU-011`/`GHU-011L` result, deployment/runtime proof, prediction, capture, training, promotion, betting, or runtime/data mutation.
- Next safe action: Preserve this blocked predecessor and the accepted `GHU-000B3` correction evidence; no further predecessor action.
- Closeout evidence: On programme date 2026-07-30, verified predecessor checkpoint/base `c550b81f111d0e053c1c3dd6014ef0f28b7638c1`, tree `179300b2ddff681a52c9f7ae6fdffbf2c0137c15`, and transitioned `planned -> ready -> active -> review` in implementer run `20260730T174818Z-c550b81f11-c6ce5a`, session `019fb424-5860-7bc3-9b71-bd0e88b8880a`, child `812c251abd99366b07fc1f9f02f82820`. Frozen checkpoint `cc65dca19cd4bb9fa6b8c836dc843c0ba00bed7b`, tree `009129fc2506a1f9d5d867177279c27c4956d113`, correction diff SHA-256 `fd2b93a33df15974173f9212449c7ce0c43e22fb303b6112e5d96e20bcd87363`. Independent reviewer run `20260730T175213Z-cc65dca19c-10a89a`, session `019fb427-fb80-7122-91ad-6cd2987b17c4`, child `0f16abbb824ddff3ed7e63f6deba86bc`, verdict `REJECT_GHU_000B1`, found two blockers: the existing `DEC-GHU-000B-FIXED-COLLECTOR-INDEX` still read as operative without an append-only entry recording rejection/correction state, and its accepted-integration identity was abbreviated. Parent rejected `GHU-000B1`; it transitioned `review -> blocked`. Commit/push/PR/merge/deploy/runtime proof: `NOT_OCCURRED`.

## GHU-000B2 — Correct rejected GHU-000B1 decision-ledger evidence

- Release: `R0`
- Priority: `P0`
- Status: `blocked`
- Dependencies: `GHU-001`, `GHU-000C1`
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: Fresh bounded Codex X ledger-correction implementer
- Outcome: Corrects only the three ledgers after parent rejection of `GHU-000B` and independent rejection of `GHU-000B1`.
- Scope: Correct only `TICKETS.md`, `STATUS.md`, and `DECISIONS.md` bookkeeping and decision supersession for the two reviewer findings.
- Non-goals: No `PLAN.md`, `CONTRACTS.md`, product, code, test, runtime, data, database, lock, browser, service, prediction, capture, acceptance, commit, push, merge, deployment, or source-claim change.
- Acceptance: `GHU-000B1` is blocked with exact implementer/reviewer evidence.; A dated append-only decision explicitly supersedes the operative reading of the rejected B/B1 entry while preserving only the unchanged factual fixed-index mapping pending parent acceptance.; The accepted integration identity is full-length.; Counts and current pointers agree.
- Validation: Exact base/tree and allowed-path checks.; `PLAN.md` and `CONTRACTS.md` byte identity versus base.; Append-only decision, full-hash, count, transition, dependency, cross-reference, and date scans.; `git diff --check`.
- Risks: Material risk: altering the sound fixed-index mapping, accepted `GHU-001`/`GHU-000C1` claims, or R1 running/failure history.
- Stop conditions: Stop on any need to edit outside the three allowed ledgers or to perform product/runtime/test/external action.
- Authority: Documentation-only ledger correction; parent owns review, acceptance, integration, publication, and deployment.
- Claims supported: Only this preserved rejected three-ledger correction and its exact evidence.
- Claims unsupported: Acceptance of `GHU-000B2`, `GHU-000B1`, or rejected `GHU-000B`; changes to `PLAN.md`, `CONTRACTS.md`, or source; terminal `GHU-011`/`GHU-011L` result; deployment/runtime proof; prediction, capture, training, promotion, betting, or runtime/data mutation.
- Next safe action: Preserve this blocked predecessor and the accepted `GHU-000B3` correction evidence; no further predecessor action.
- Closeout evidence: On programme date 2026-07-30, verified checkpoint/base `cc65dca19cd4bb9fa6b8c836dc843c0ba00bed7b`, tree `009129fc2506a1f9d5d867177279c27c4956d113`, and transitioned `planned -> ready -> active -> review` in implementer run `20260730T175638Z-cc65dca19c-c66202`, session `019fb42d-217a-7fa0-966f-f36ddabd78d5`, child `49e930e41c3a2fa72690df154d3b130c`. Frozen checkpoint `3271721e5b19bc795f775a00a608c557f85b0112`, tree `cc52608ffff88ce784c08da3236b55c82ec753fd`, binary diff SHA-256 `0893bde44d28fd9bf24795771947374dca82a3654d76dfdd979dbbb2f85fdc6d`. Independent reviewer run `20260730T180236Z-3271721e5b-bed535`, session `019fb431-732f-7dc2-ade5-45803721691c`, child `fcd20a35022ea048ded40777f364b710`, verdict `REJECT_GHU_000B2`: the sole blocker was the exact stale `GHU-000B` current pointer directing review of rejected `GHU-000B1`; all other review axes passed. Parent rejected `GHU-000B2`; it transitioned `review -> blocked`. Commit/push/PR/merge/deploy/runtime proof: `NOT_OCCURRED`.

## GHU-000B3 — Correct rejected GHU-000B2 current-pointer evidence

- Release: `R0`
- Priority: `P0`
- Status: `accepted`
- Dependencies: `GHU-001`, `GHU-000C1`
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: Fresh bounded Codex X ledger-correction implementer
- Outcome: Corrects the sole stale current pointer found during independent review of `GHU-000B2` and records its rejection/closeout evidence.
- Scope: Change only the `GHU-000B` current pointer and the necessary `TICKETS.md`, `STATUS.md`, and `DECISIONS.md` rejection/closeout bookkeeping.
- Non-goals: No change to preserved B2 fixed-index mapping or authority correction; no `PLAN.md`, `CONTRACTS.md`, product, code, test, runtime, data, database, lock, browser, service, prediction, capture, acceptance, commit, push, merge, deployment, or R1 terminal-result claim.
- Acceptance: Every current pointer reflects accepted `GHU-000B3` as appropriate.; `GHU-000B2` is blocked with exact implementer and reviewer evidence and its independently-passed mapping/authority correction is preserved in accepted B3.; Counts, transitions, dependencies, dates, and cross-references agree.
- Validation: Exact base/tree and allowed-path checks.; `PLAN.md` and `CONTRACTS.md` byte identity versus base.; Full-hash, no-stale-current-pointer, count, transition, dependency, cross-reference, date, append-only-decision, and no-supplied-evidence-placeholder scans.; `git diff --check`.
- Risks: Material risk: changing sound B2 substance, accepted `GHU-001`/`GHU-000C1` claims, or inventing a terminal R1 result.
- Stop conditions: Stop on any need to edit outside the three allowed ledgers or to perform product/runtime/test/external action.
- Authority: Documentation-only ledger correction; parent owns review, acceptance, integration, publication, and deployment.
- Claims supported: Only the accepted three-ledger current-pointer and rejection/closeout correction, and the narrow fixed collector-owned current-race-index mapping it makes operative.
- Claims unsupported: Acceptance of `GHU-000B2`, `GHU-000B1`, or rejected `GHU-000B`; changes to `PLAN.md`, `CONTRACTS.md`, product, code, tests, or source; deployment/runtime proof; prediction, capture, training, promotion, betting, or runtime/data mutation.
- Next safe action: Preserve accepted evidence; the R0 fixed collector-owned current-race-index correction is closed.
- Closeout evidence: On programme date 2026-07-30, verified checkpoint/base `3271721e5b19bc795f775a00a608c557f85b0112`, tree `cc52608ffff88ce784c08da3236b55c82ec753fd`, and transitioned `planned -> ready -> active -> review` in implementer run `20260730T180514Z-3271721e5b-a531a0`, session `019fb433-d9c0-7b22-b36d-4a832833e4ce`, child `50f7490271d4dd156e56032b72ddb9ab`. Independent reviewer run `20260730T180925Z-44fe9a0875-a94ad1`, session `019fb437-b8d1-7dd3-9e06-f4494603e9d7`, returned `ACCEPT_GHU_000B3`. Parent decision: accepted and integrated the exact correction as commit `6e0b0d99c296a4c984faf0775bab88f8689e66da`, tree `bff53978cdfeea8f604404432e1d672cba95a692`. Push/PR/default-branch merge/deploy/runtime proof: `NOT_OCCURRED`.

## GHU-000C — Harden collector index reads (rejected predecessor)

- Release: `R0`
- Priority: `P0`
- Status: `blocked`
- Dependencies: `GHU-001`
- Model routing: GPT-5.6 Sol
- Session role: Preserved predecessor implementation
- Outcome: Attempted descriptor-bound fixed-index reads before the accepted correction.
- Scope: Preserved historical candidate only.
- Non-goals: It is not integration history and grants no accepted behavior.
- Acceptance: Not met.
- Validation: Parent-focused result `1 failed, 18 passed`; the post-read `/proc/self/fd` `OSError` was caught by the broad handler and misclassified as caller-missing code `CURRENT_INDEX_SOURCE_MISSING`, rather than the expected `CURRENT_INDEX_PATH_UNSAFE` with reason `path_replaced`.
- Risks: Unsupported absolute claims about concurrent mutation.
- Stop conditions: Rejected candidate remains blocked.
- Authority: Historical evidence only.
- Claims supported: The recorded rejected checkpoint and exact observed failure.
- Claims unsupported: Integration, acceptance, or proof that same-inode concurrent content mutation is detected.
- Next safe action: Preserve; accepted successor GHU-000C1 is authoritative.
- Closeout evidence: Run `20260730T172828Z-f38a125f63-770b6a`, session `019fb412-36a3-7992-b249-0ba5e635c72c`, child `9f5cd6a19933e2a044a2a3c23c88ec1c`; checkpoint `3e9f639dfff62ffddd85aa00bab3d5c6b475cdf6`, tree `bdf39b69919e07bbfd5d8d330644b665aaa57fc7`; not in integration history.

## GHU-000C1 — Deterministic collector index path-replacement rejection

- Release: `R0`
- Priority: `P0`
- Status: `accepted`
- Dependencies: `GHU-001`
- Model routing: GPT-5.6 Sol
- Session role: Parent-accepted bounded correction
- Outcome: Converts post-open path-revalidation operating-system errors into deterministic fail-closed index rejection.
- Scope: Accepted prerequisite closeout only.
- Non-goals: No claim that portable `O_NOFOLLOW` exists outside supporting platforms or that same-inode concurrent content mutation is detected.
- Acceptance: Focused parent suite passes and independent review accepts the exact correction.
- Validation: Parent `19 passed in 0.86s`.
- Risks: Linux `/proc/self/fd` and `O_NOFOLLOW` are platform limitations; descriptor/path replacement checks do not support an absolute same-inode concurrent-mutation claim.
- Stop conditions: Preserve accepted scope and limitations.
- Authority: Accepted source prerequisite; no runtime/deployment claim.
- Claims supported: Exact accepted correction and deterministic path-replacement rejection.
- Claims unsupported: Portable non-Linux equivalence or same-inode mutation detection.
- Next safe action: Independently review and parent-integrate the corrected R1 ledger closeout, then proceed to `GHU-020`.
- Closeout evidence: Implementer run `20260730T173227Z-3e9f639dff-b1c869`, session `019fb415-e17b-7761-84b9-97a96a8fb58d`, child `a1704231395179acca71854ce5ba7acb`, correction diff identity `86567b5e32177ed0028940cb11158fee6e34a5f876f0903a4ef151458e1e59aa`; reviewer run `20260730T173456Z-04c32c37fe-4018cf`, session `019fb418-1b23-7fa1-985d-48f0219864e1`, child `6132758c7ecce3222bbf0f2059b29c77`, verdict `ACCEPT_GHU_000C1`. Parent accepted final integration `c77b3be5ad4aa78b70a9ba89f25ee801d50f27c0`, tree `fe5115435d18cbce6be055cf452acdba65518a76`, accepted staged diff identity `26886b3fba57cce1369dc45794f563a5ba250fbeadb9f7585bdf3f731ddbe373`.

## GHU-010 — Design tokens and responsive application shell

- Release: `R1`
- Priority: `P1`
- Status: `accepted`
- Dependencies: `GHU-002`
- Model routing: GPT-5.6 Terra; fallback GPT-5.6 Luna
- Session role: Fresh Codex X implementation session
- Outcome: Produces the dark operator-console shell from the approved mockup using the repository's existing UI stack.
- Scope: Implement typography, spacing, dark palette, status colors, panels, sidebar/header and responsive grid.; Add persistent PROTOTYPE DATA and RESEARCH ONLY indicators in fixture mode.; Create reusable status, freshness and evidence-reference components.; Keep desktop and phone layouts usable.
- Non-goals: No live adapters, prediction execution, new frontend stack or external design system unless already present.
- Acceptance: Shell matches the visual direction without copying illustrative counts as facts.; Keyboard focus, contrast and reduced-motion behavior are defined.; No inline operational commands or raw lock controls appear in operator mode.
- Validation: Existing frontend unit/lint/type checks.; Fixture render smoke at desktop and mobile widths.; Focused accessibility checks supported by the repository.
- Risks: Material risk: Stop before introducing a new framework or broad CSS rewrite not required by the authoritative UI. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop before introducing a new framework or broad CSS rewrite not required by the authoritative UI.
- Authority: Level 1 fixture UI only; no operational mutation.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Independently review and parent-integrate the corrected R1 ledger closeout, then proceed to `GHU-020`.
- Closeout evidence: On 2026-07-30, after the accepted `GHU-002` integration above was verified as candidate base commit/tree, transitioned in order `planned -> ready -> active` and assigned to fresh Codex X launcher run `20260730T101930Z-73f1e5d041-380183`, child identity `c03a66e899ab18c04d650cd106c1598e` (no separate Codex session UUID exposed). Base commit `73f1e5d041f8d78ee0f48ce13e008f71c20090ca`; base tree `d68116ba72b28f149707d7610821aea02cb35781`. After the candidate was frozen, transitioned `active -> review`. The original attempt's static checks passed, while its unprovisioned focused pytest and browser coverage did not execute; the preserved `GHU-010A`, `GHU-010B`, and `GHU-010C` blocked history and the independently accepted-for-diagnostic `GHU-010D` review below record the subsequent correction path. Authoritative current candidate evidence is focused pytest `2 passed` and Playwright `3 passed in 2.1s`. The original broad suite remains failed and is not relabeled passed: `24 failed, 518 passed, 40 subtests passed in 4527.96s`. Stable diagnostic of only those 24 nodes recorded untouched base `24 passed in 709.55s`; the exact uncommitted candidate `21 passed` plus 3 dirty-release-identity failures in `704.25s`; and those same 3 nodes passing in `6.73s` when the exact eight-file candidate was frozen validation-only as commit `eda152192e96f7f89ccfc4ab3e89d5965cbc4055`, tree `ae2a32a5371f8fe73dc83d4e8daab5bcee9b37d6`. This establishes no candidate regression among those nodes without claiming the original broad command passed. Parent accepted `GHU-010` and `GHU-010F` after exact independent review of the eight-file delta in run `20260730T133751Z-73f1e5d041-0d30ea`, session `019fb33f-0f5a-7650-bc0c-1de8cc391704`, child `dbdaf9f23daee7c986a0a59c61aea562`, verdict `ACCEPT_GHU_010F` with no blocking findings. Exact integrated commit `13cf3a3b54a4a411465ac570e5ecb65b1669cdc3`; parent `73f1e5d041f8d78ee0f48ce13e008f71c20090ca`; tree `cf80477be77676f4e8eec54a8aa23d2fd6917896`; accepted cached diff SHA-256 `f6b25bc07f7f1a385154acdb87d1357399d4361c72b338abefc68fbbf2cd6cc8`. Parent decision: accepted the exact reviewed eight-file delta and mechanically committed it on the clean integration branch on 2026-07-30. The medium limitation remains non-blocking: focused coverage does not instrument generic filesystem writes or pre-existing request telemetry. Low delivery fact: both ignored templates were force-staged with verified bytes. Publication, PR, merge to the repository default branch, deployment, and runtime proof: `NOT_OCCURRED`.

## GHU-010A — Isolate operator shell response assets

- Release: `R1`
- Priority: `P1`
- Status: `blocked`
- Dependencies: `GHU-002`
- Model routing: GPT-5.6 Terra; fallback GPT-5.6 Luna
- Session role: Fresh bounded Codex X correction implementer
- Outcome: Prevents the generic legacy HTML response mutator from altering the `/operator-ui/` namespace while preserving the rejected `GHU-010` shell candidate for renewed review.
- Scope: Reproduce the exact rejected eight-file candidate.; Exclude only `/operator-ui/` responses from the existing generic HTML asset/banner/script mutation block.; Extend the focused Flask response test to prove operator-owned assets and required shell content remain while legacy injections are absent.; Run the provisioned focused Flask and single Chromium validation.
- Non-goals: No shell, template, macro, palette, Playwright assertion, legacy-route, service, runtime, collector, browser-control, prediction, deployment, or adjacent UI redesign.
- Acceptance: The final operator response contains its own stylesheet and required fixture shell content.; It contains no legacy `style.css`, `a11y.js`, mode banner, E2E script, inline handler, or unrelated route injection.; Legacy request completion, logging, compression, caching, and route mutation behavior remain unchanged.; Focused Flask and Chromium/axe/mobile/focus/reduced-motion coverage pass.
- Validation: Provisioned focused pytest and Playwright Chromium desktop project.; Python and Node syntax checks.; Response/request isolation assertions.; `git diff --check`, unchanged `templates/index.html`, exact forbidden-content scan, and repository classifier.
- Risks: Material risk: A broader bypass could suppress legacy behavior or non-mutation response processing. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop on any required focused browser failure without retry, any changed legacy-route behavior, any modification outside the eight allowed paths, or any need to redesign the accepted shell.
- Authority: Level 1 fixture UI correction only; no operational mutation. Parent owns acceptance, commit, integration, publication, deployment, and runtime proof.
- Claims supported: Only that the frozen correction candidate isolates the operator response after the listed validation; `GHU-010` remains review-only pending independent review and parent acceptance.
- Claims unsupported: Acceptance, deployment, runtime proof, market edge, profitability, EV, staking, betting, public exposure, live prediction, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Parent must provide a writable Playwright temporary/cache location in a fresh bounded correction run; rerun the required Chromium project once there, then transition `active -> review` only if it passes. Keep `GHU-010` in review.
- Closeout evidence: On 2026-07-30, transitioned in order `planned -> ready -> active` after verifying accepted dependency `GHU-002` at base commit `73f1e5d041f8d78ee0f48ce13e008f71c20090ca`, tree `d68116ba72b28f149707d7610821aea02cb35781`; assigned to fresh Codex X launcher run `20260730T103343Z-73f1e5d041-b1335d`, child identity `841da0536f26ba759aa310e227aa327f` (no separate Codex session UUID exposed). This is the bounded correction to rejected `GHU-010`. Focused pytest passed `2 passed`; Python and Node syntax checks passed. The one permitted Playwright invocation exited 1 before config/test loading with `ENOENT: no such file or directory, mkdir '/tmp/playwright-transform-cache-1000/20'`; the private Flask process was stopped immediately and the browser test was not retried. Therefore transitioned `active -> blocked`, not review. Remaining validation and frozen hashes are reported in the implementer handoff. Independent reviewer/verdict and parent decision: `PENDING`. Commit/PR/merge/deploy/runtime proof: `NOT_OCCURRED`.

## GHU-010B — Deterministic reduced-motion browser emulation

- Release: `R1`
- Priority: `P1`
- Status: `blocked`
- Dependencies: `GHU-002`
- Model routing: GPT-5.6 Terra; fallback GPT-5.6 Luna
- Session role: Fresh bounded Codex X correction implementer
- Outcome: Makes the existing reduced-motion browser check deterministically activate `prefers-reduced-motion: reduce` before navigation while preserving the rejected `GHU-010A` product candidate for renewed review.
- Scope: Reproduce the exact rejected `GHU-010A` eight-file candidate.; In `tests/playwright/operator-ui-shell.spec.js` only, emulate reduced motion on the test page before navigation, assert the media query is active, remove the ineffective suite-level setting, and retain the computed scroll, animation, and transition assertions.; Update only this ticket and the status ledger.; Run the provisioned focused validation once.
- Non-goals: No product, Flask, template, CSS, Python-test, shell-behavior, legacy-route, service, runtime, collector, browser-control, prediction, deployment, broad-suite, or adjacent UI change.; No retry after a terminal browser-test result.
- Acceptance: The reduced-motion test explicitly emulates `reduce` before navigation and proves the corresponding media query is active.; Existing computed `scroll-behavior`, `animation-duration`, and `transition-duration` assertions remain.; All focused Python, syntax, static, isolation, forbidden-scan, classifier, and single Chromium operator-spec validation passes.; Only the Playwright spec and two ledgers differ from the preserved `GHU-010A` identities.
- Validation: Provisioned focused pytest.; Python and Node syntax checks.; One Playwright Chromium desktop invocation covering desktop, 375px, axe/focus, and reduced motion.; `git diff --check`, unchanged-root and exact preserved-product identity checks, forbidden scans, and repository classifier.
- Risks: Material risk: Browser media emulation could be applied too late or the correction could drift product behavior. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop on the first required focused failure without retry, any product-path drift from preserved `GHU-010A`, any modification outside the eight allowed paths, or any need to change product behavior.
- Authority: Level 1 fixture browser-test correction only; no operational mutation. Parent owns acceptance, commit, integration, publication, deployment, and runtime proof.
- Claims supported: Only that the frozen correction candidate deterministically validates the preserved reduced-motion product behavior after every listed focused check passes; `GHU-010` remains review-only pending independent review and parent acceptance.
- Claims unsupported: Acceptance, broad-suite success, deployment, runtime proof, market edge, profitability, EV, staking, betting, public exposure, live prediction, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Parent may authorize a fresh smallest correction that compares the equivalent computed duration serialization (`1e-05s`, or a parsed duration) without changing product CSS; preserve this failed run and do not retry it. Keep `GHU-010` in review.
- Closeout evidence: On 2026-07-30, after verifying accepted dependency `GHU-002` at base commit `73f1e5d041f8d78ee0f48ce13e008f71c20090ca`, tree `d68116ba72b28f149707d7610821aea02cb35781`, transitioned in order `planned -> ready -> active` and assigned to fresh Codex X launcher run `20260730T104152Z-73f1e5d041-f3a7f7`, child identity `1980f622df32340e1887b42f494d557e` (no separate Codex session UUID exposed). Focused pytest passed `2 passed`; Python and Node syntax, `git diff --check`, unchanged-root/product-identity checks, corrected forbidden scans, and classifier execution passed. The classifier selected `full_forecasting`; that broad suite was not run and is not claimed. The one permitted Playwright Chromium-desktop invocation ran three tests: desktop and 375px passed; reduced motion proved the media query active and `scroll-behavior: auto`, then failed because Chromium returned the equivalent computed duration serialization `1e-05s` where the retained assertion expected `0.01ms`. The private server was stopped immediately and the test was not retried. Therefore transitioned `active -> blocked`, not review. Independent reviewer/verdict and parent decision: `PENDING`. Commit/PR/merge/deploy/runtime proof: `NOT_OCCURRED`.

## GHU-010C — Normalize computed reduced-motion durations

- Release: `R1`
- Priority: `P1`
- Status: `blocked`
- Dependencies: `GHU-002`
- Model routing: GPT-5.6 Terra; fallback GPT-5.6 Luna
- Session role: Fresh bounded Codex X correction implementer
- Outcome: Validates the preserved reduced-motion behavior independently of Chromium's equivalent `ms` or `s` computed-duration serialization.
- Scope: Reproduce the exact blocked `GHU-010B` eight-file candidate.; In `tests/playwright/operator-ui-shell.spec.js` only, parse computed animation and transition duration lists in `ms` or `s`, normalize each value to milliseconds, and require every duration to be at most `0.01ms`.; Retain the active reduced-motion media-query and `scroll-behavior: auto` assertions.; Update only this ticket and the status ledger.; Run the provisioned focused validation once.
- Non-goals: No CSS, template, Flask, Python-test, product behavior, legacy-route, service, runtime, collector, browser-control, prediction, deployment, broad-suite, or adjacent UI change.; No retry after a terminal browser-test result.
- Acceptance: Computed animation and transition durations accept equivalent `ms` and `s` serialization and reject invalid, non-finite, or values above `0.01ms`.; The reduced-motion test still proves its media query is active and computed scroll behavior is `auto`.; All focused Python, syntax, static, isolation, forbidden-scan, classifier, and single Chromium operator-spec validation passes.; Only the Playwright spec and two ledgers differ from the preserved `GHU-010B` identities.
- Validation: Provisioned focused pytest.; Python and Node syntax checks.; One Playwright Chromium desktop invocation covering desktop, 375px, axe/focus, and reduced motion.; `git diff --check`, root/identity and exact preserved-product checks, forbidden scans, serialized source-diff hash, and repository classifier.
- Risks: Material risk: A permissive parser could accept a longer or invalid duration, or the correction could drift product behavior. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop on the first required focused failure without retry, any product-path drift from preserved `GHU-010B`, any modification outside the eight allowed paths, or any need to change product behavior.
- Authority: Level 1 fixture browser-test correction only; no operational mutation. Parent owns acceptance, commit, integration, publication, deployment, and runtime proof.
- Claims supported: Only that the frozen correction candidate validates the preserved reduced-motion product behavior after every listed focused check passes; `GHU-010` remains review-only pending independent review and parent acceptance.
- Claims unsupported: Acceptance, broad-suite success, deployment, runtime proof, market edge, profitability, EV, staking, betting, public exposure, live prediction, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Parent may authorize a fresh bounded validation correction with a readiness probe that genuinely waits for the provisioned app import; preserve this stopped server attempt and do not retry it. Keep `GHU-010` in review.
- Closeout evidence: On 2026-07-30, after verifying accepted dependency `GHU-002` at base commit `73f1e5d041f8d78ee0f48ce13e008f71c20090ca`, tree `d68116ba72b28f149707d7610821aea02cb35781`, transitioned in order `planned -> ready -> active` and assigned to fresh Codex X launcher run `20260730T104856Z-73f1e5d041-46c798`, child identity `3d827523ebdd701c6f2d65e937321714` (no separate Codex session UUID exposed). The five preserved product files matched the supplied `GHU-010B` SHA-256 identities, both ignored templates matched their supplied identities, only the Playwright spec and two ledgers differed from `GHU-010B`, focused pytest passed `2 passed`, Python and Node syntax passed, `git diff --check`, root/identity, unchanged-root-template, forbidden/isolation, and classifier checks passed. The classifier selected `full_forecasting`; the parent-owned broad suite was not run. One private test-server process was started on `127.0.0.1:5520`, but the readiness loop exhausted without genuinely waiting while the app was still importing; the trap stopped that process immediately. Playwright/Chromium was never invoked, no retry occurred, and required browser validation remains unproven. Therefore transitioned `active -> blocked`, not review. Independent reviewer/verdict and parent decision: `PENDING`. Commit/PR/merge/deploy/runtime proof: `NOT_OCCURRED`.

## GHU-010D — Reconcile validation evidence and correct the validation environment

- Release: `R1`
- Priority: `P1`
- Status: `blocked`
- Dependencies: `GHU-002`
- Model routing: GPT-5.6 Terra; fallback GPT-5.6 Luna
- Session role: Fresh bounded Codex X evidence-reconciliation implementer
- Outcome: Preserves the rejected diagnostic reconciliation and records its actual blocked, superseded disposition without changing its detailed evidence.
- Scope: Preserve the exact `GHU-010C` eight-file candidate evidence, parent focused/browser/classifier/broad results, exact failure clusters, and the later accepted shell correction chain.; Keep `GHU-010A`, `GHU-010B`, `GHU-010C`, and `GHU-010D` blocked.
- Non-goals: No product, Flask, CSS, template, Python-test, Playwright-test, runtime, service, data, lock, browser, collector, training, promotion, EV, betting, broad-suite rerun, browser rerun, acceptance, commit, integration, publication, merge, or deployment change.
- Acceptance: The six product/test SHA-256 identities exactly match the supplied `GHU-010C` candidate.; The ledger records parent focused pytest `2 passed`, Playwright `3 passed in 2.1s`, classifier `full_forecasting`, and the single broad-suite result `24 failed, 518 passed, 40 subtests passed in 4527.96s` with exit 1.; The observed failure clusters and later diagnostic evidence remain historical evidence only.; The ticket is blocked and superseded, with no broad-pass claim and no current-review claim.
- Validation: Exact HEAD/tree and supplied eight-file SHA-256 checks.; Python and Node syntax checks.; `git diff --check`.; forbidden/isolation scan.; repository classifier.; focused pytest once only if provisioned.; allowed-path and exact product/test identity audit.; No broad-suite or browser rerun.
- Risks: Material risk: Reconciliation could turn an environment-specific validation failure into a product claim, erase blocked candidate history, or imply acceptance. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop on identity drift, product/test byte drift, any path outside the eight-file candidate, any required static failure, or any need to rerun the broad suite or browser.
- Authority: Ledger-only Level 1 evidence reconciliation; no operational mutation. Parent owns acceptance, commit, integration, publication, deployment, and runtime proof.
- Claims supported: Only the preserved diagnostic evidence for the superseded candidate: recorded focused/browser passes and the original failed broad run in its two recorded validation-environment clusters.
- Claims unsupported: Readiness for review, broad-suite success, current `GHU-010` acceptance evidence, deployment, runtime proof, market edge, profitability, EV, staking, betting, public exposure, live prediction, training, promotion, and any runtime or data mutation.
- Next safe action: Preserve this blocked, superseded diagnostic predecessor and use the accepted `GHU-010H` closeout as current R1 shell evidence.
- Closeout evidence: On 2026-07-30, after verifying accepted dependency `GHU-002` at base commit `73f1e5d041f8d78ee0f48ce13e008f71c20090ca`, tree `d68116ba72b28f149707d7610821aea02cb35781`, transitioned in order `planned -> ready -> active` and was assigned to fresh Codex X launcher run `20260730T121336Z-73f1e5d041-2eec9d`, child identity `5bc80e1862cf6053ea84699b9df5c8e5` (no separate Codex session UUID exposed). The exact eight supplied `GHU-010C` files were reproduced; the six product/test files retained SHA-256 identities `fad5de42e5a8ef98cd6ee1eab5b12e98055d198c4bbaa133e9bdfc2f2442664c`, `fa1f2482849e66da3c6eefe999d0e691fe737792d0a052599c73a52adf069f8d`, `9d5c90efc8d5968ef4d893e8d82adf803685a2e36441694385b4ebc8787d7005`, `550cb1b946a9236d10346756b4065daa9de0e6c3e12c779ef2c1c13525ce3759`, `4a7869c759c77aec49210fab6f2f6cd03e3481e18096383a13fdc7bfb1d5c057`, and `753a5a1280fa28bcb072b3dc7b067b8e15fbfbf9c058998d343babcf0aaa47c2` in the scope order. Parent validation recorded focused pytest `2 passed`, Playwright `3 passed in 2.1s`, and classifier `full_forecasting`. Its single broad suite exited 1 with `24 failed, 518 passed, 40 subtests passed in 4527.96s`; Phase 7 runtime tests reject the launcher `.state/runs` path through `_safe_operational_path`, while Phase 6 promotion fixtures fail `data_domain_drift`. This run did not rerun pytest because it was not provisioned, and did not rerun the browser or broad suite. Python/Node syntax, `git diff --check`, forbidden/isolation, classifier, exact identity, and allowed-path static checks passed after one shell-quoting construction error (exit 2) prevented the first static batch from starting; the corrected batch exited 0. After those static checks, `GHU-010D` transitioned `active -> review`. It was subsequently blocked and superseded by the accepted shell correction chain recorded in `GHU-010H`; the proven parent disposition is `SUPERSEDED`. Commit/PR/default-branch merge/deploy/runtime proof for `GHU-010D`: `NOT_OCCURRED`.

## GHU-010E — Reconcile stable-path diagnostic evidence

- Release: `R1`
- Priority: `P1`
- Status: `blocked`
- Dependencies: `GHU-002`
- Model routing: GPT-5.6 Terra; fallback GPT-5.6 Luna
- Session role: Fresh bounded Codex X evidence-reconciliation implementer
- Outcome: Produces the final `GHU-010` candidate by preserving the independently reviewed `GHU-010D` product/test bytes and reconciling the stable-path diagnostic that isolates the prior broad-suite failures as validation-environment effects.
- Scope: Reproduce the exact reviewed `GHU-010D` six product/test files.; Update only `TICKETS.md` and `STATUS.md` with the independent `GHU-010D` verdict, stable diagnostic results, frozen validation-only commit/tree, proof limitation, delivery risk, and final-review boundary.; Preserve all rejection history and accepted prerequisites.
- Non-goals: No product, Flask, CSS, template, Python-test, Playwright-test, runtime, service, data, lock, browser, collector, training, promotion, EV, betting, test execution, acceptance, commit, integration, publication, merge, or deployment change.
- Acceptance: The six product/test SHA-256 identities exactly match the reviewed `GHU-010D` candidate.; The ledger retains the original broad-suite failure as `24 failed, 518 passed, 40 subtests passed in 4527.96s` and never relabels it passed.; The stable diagnostic records all 24 nodes passing on untouched base, 21 passing plus only 3 dirty-worktree release-identity failures on the exact uncommitted candidate, and those 3 passing on the same eight-file candidate frozen validation-only.; The ledger keeps `GHU-010` unaccepted and marks this candidate complete pending independent final review.
- Validation: Exact HEAD/tree, allowed-path, supplied six-file SHA-256, Python and Node syntax, and `git diff --check` checks only.; No pytest, browser, broad suite, service, or runtime invocation.
- Risks: Material risk: Reconciliation could erase the original broad failure, overstate the diagnostic, or omit the ignored-template delivery requirement. The reviewer medium proof limitation is non-blocking: the focused test does not instrument generic filesystem writes or pre-existing request telemetry. The low delivery risk is that ignored templates require force-staging of their exact bytes.
- Stop conditions: Stop on base/tree or product/test identity drift, evidence conflict, any path outside the eight allowed paths, any required static failure, or any need to change behavior or rerun tests.
- Authority: Ledger-only Level 1 evidence reconciliation; no operational mutation. Parent owns acceptance, commit, integration, publication, deployment, and runtime proof.
- Claims supported: Only that stable-path diagnostics isolate validation-environment effects and establish no candidate regression for the original 24 failed nodes, while the exact reviewed product/test bytes are preserved for final review.
- Claims unsupported: A pass for the original broad command, `GHU-010` acceptance, exhaustive proof of no generic filesystem writes or pre-existing request telemetry, deployment, runtime proof, market edge, profitability, EV, staking, betting, public exposure, live prediction, training, promotion, and any runtime or data mutation.
- Next safe action: Preserve this rejected candidate and proceed only through the smallest ledger correction `GHU-010F`; keep `GHU-010` unaccepted in review.
- Closeout evidence: On 2026-07-30, after verifying accepted dependency `GHU-002` at base commit `73f1e5d041f8d78ee0f48ce13e008f71c20090ca`, tree `d68116ba72b28f149707d7610821aea02cb35781`, transitioned in order `planned -> ready -> active -> review` and was assigned to fresh Codex X launcher run `20260730T125952Z-73f1e5d041-0a6b07`, child identity `d99ebf4b11ac7752e21424967ba727b4` (no separate Codex session UUID exposed). Reference `GHU-010D` run `20260730T121336Z-73f1e5d041-2eec9d`, child `5bc80e1862cf6053ea84699b9df5c8e5`, was independently reviewed in run `20260730T122656Z-73f1e5d041-20a0f9`, session `019fb309-f0b2-7e81-bcac-fa7d219e911b`, child `70bb7e85bfd14d3400b7184ab019bf75`, verdict `ACCEPT_GHU_010D_FOR_DIAGNOSTIC`; independent spec session `019fb30a-b648-7ba2-936c-6e5f503ba1ac` agreed there was no deviation. Previously recorded exact-byte validation remains focused pytest `2 passed` and Playwright `3 passed in 2.1s`. The original broad suite remains failed: `24 failed, 518 passed, 40 subtests passed in 4527.96s`. Stable diagnostic of only those 24 nodes: untouched base `24 passed in 709.55s`; exact uncommitted candidate `21 passed` and 3 release-identity nodes failed solely because the dirty worktree was correctly rejected, in `704.25s`; the same exact eight-file candidate was then frozen validation-only as commit `eda152192e96f7f89ccfc4ab3e89d5965cbc4055`, tree `ae2a32a5371f8fe73dc83d4e8daab5bcee9b37d6`, and only those 3 nodes passed in `6.73s`. This establishes no candidate regression among those nodes without claiming the original broad command passed. Reviewer medium proof limitation: focused coverage does not instrument generic filesystem writes or pre-existing request telemetry; this is a non-blocking follow-up only. Low delivery risk: both ignored templates require force-staging with their exact verified bytes. Parent exact-delta inspection rejected `GHU-010E` because the top-level `GHU-010` Next safe action and Closeout evidence still stated that provisioned Flask/Playwright coverage had not executed and must run before acceptance, contradicting the later authoritative focused pytest `2 passed`, Playwright `3 passed in 2.1s`, and stable diagnosis of the original 24 failures. This is a ledger contradiction only; the product/test bytes were accepted for correction. `GHU-010E` therefore transitioned `review -> blocked`. Parent acceptance, commit, integration, publication, PR, merge, deployment, and runtime proof: `NOT_OCCURRED`.

## GHU-010F — Correct top-level validation ledger

- Release: `R1`
- Priority: `P1`
- Status: `accepted`
- Dependencies: `GHU-002`
- Model routing: GPT-5.6 Terra; fallback GPT-5.6 Luna
- Session role: Fresh bounded Codex X ledger-correction implementer
- Outcome: Produces a consistent final `GHU-010` review candidate by correcting only the stale top-level validation ledger while preserving the accepted-for-correction `GHU-010E` product/test bytes and all validation history.
- Scope: Reproduce the exact eight `GHU-010E` candidate paths.; Change only `TICKETS.md` and `STATUS.md` relative to `GHU-010E`.; Replace the stale `GHU-010` Next safe action and Closeout evidence with authoritative current candidate evidence.; Record the parent rejection of `GHU-010E` and preserve all prior attempt, correction, review, failure, and diagnostic evidence.
- Non-goals: No product, Flask, CSS, template, Python-test, Playwright-test, runtime, service, data, lock, browser, collector, pytest, browser test, broad suite, install, commit, integration, publication, merge, deployment, training, promotion, EV, or betting change.
- Acceptance: The six product/test SHA-256 identities exactly match `GHU-010E`.; Relative to `GHU-010E`, only `TICKETS.md` and `STATUS.md` differ.; The top-level `GHU-010` entry records focused pytest `2 passed`, Playwright `3 passed in 2.1s`, the original broad failure unchanged, and the complete stable diagnostic without contradiction.; `GHU-010E` is blocked after parent rejection.; `GHU-010F` and `GHU-010` become accepted only after independent exact-delta review and parent integration.
- Validation: Exact base HEAD/tree, allowed-path, reference comparison, supplied six-file SHA-256, Python and Node syntax, and `git diff --check` checks only.; No pytest, browser, broad suite, service, or runtime invocation.
- Risks: Material risk: The correction could erase historical evidence, relabel the original broad failure passed, alter accepted product/test bytes, or preclaim acceptance. Both ignored templates retain the low delivery risk requiring force-staging with their exact verified bytes.
- Stop conditions: Stop on base/tree or product/test identity drift, evidence conflict, any path outside the eight allowed paths, any difference from `GHU-010E` outside the two ledgers, any required static failure, or any need to change behavior or rerun tests.
- Authority: Ledger-only Level 1 evidence reconciliation; no operational mutation. Parent owns acceptance, commit, integration, publication, deployment, and runtime proof.
- Claims supported: The accepted final ledger consistently records the established validation and diagnostic evidence while preserving the exact reviewed product/test bytes.
- Claims unsupported: A pass for the original broad command, deployment, runtime proof, market edge, profitability, EV, staking, betting, public exposure, live prediction, training, promotion, and any runtime or data mutation.
- Next safe action: Preserve accepted evidence; no further correction action.
- Closeout evidence: On 2026-07-30, after verifying accepted dependency `GHU-002` at base commit `73f1e5d041f8d78ee0f48ce13e008f71c20090ca`, tree `d68116ba72b28f149707d7610821aea02cb35781`, transitioned in order `planned -> ready -> active -> review` and was assigned to fresh Codex X launcher run `20260730T131716Z-73f1e5d041-5989e5`, session `019fb339-6c63-7a70-a9a3-1b84f2d269b7`, child identity `2f5201106343bd3e03a2fe631d50d0ae`. The exact eight `GHU-010E` candidate paths were reproduced; only these two ledgers differ from that reference. The six product/test files retain SHA-256 identities `fad5de42e5a8ef98cd6ee1eab5b12e98055d198c4bbaa133e9bdfc2f2442664c`, `fa1f2482849e66da3c6eefe999d0e691fe737792d0a052599c73a52adf069f8d`, `9d5c90efc8d5968ef4d893e8d82adf803685a2e36441694385b4ebc8787d7005`, `550cb1b946a9236d10346756b4065daa9de0e6c3e12c779ef2c1c13525ce3759`, `4a7869c759c77aec49210fab6f2f6cd03e3481e18096383a13fdc7bfb1d5c057`, and `753a5a1280fa28bcb072b3dc7b067b8e15fbfbf9c058998d343babcf0aaa47c2` in scope order. Independent reviewer run `20260730T133751Z-73f1e5d041-0d30ea`, session `019fb33f-0f5a-7650-bc0c-1de8cc391704`, child `dbdaf9f23daee7c986a0a59c61aea562`, returned `ACCEPT_GHU_010F` with no blocking findings. Parent accepted `GHU-010F` and `GHU-010`, then mechanically committed the exact reviewed eight-file delta on the clean integration branch on 2026-07-30: commit `13cf3a3b54a4a411465ac570e5ecb65b1669cdc3`; parent `73f1e5d041f8d78ee0f48ce13e008f71c20090ca`; tree `cf80477be77676f4e8eec54a8aa23d2fd6917896`; accepted cached diff SHA-256 `f6b25bc07f7f1a385154acdb87d1357399d4361c72b338abefc68fbbf2cd6cc8`. The medium limitation remains non-blocking: focused coverage does not instrument generic filesystem writes or pre-existing request telemetry. Low delivery fact: both ignored templates were force-staged with verified bytes. Publication, PR, merge to the repository default branch, deployment, and runtime proof: `NOT_OCCURRED`.

## GHU-010G — Close accepted shell ledger and ready dashboard work

- Release: `R1`
- Priority: `P1`
- Status: `blocked`
- Dependencies: `GHU-010F`
- Model routing: GPT-5.6 Terra; fallback GPT-5.6 Luna
- Session role: Fresh bounded Codex X ledger-closeout implementer
- Outcome: Records parent acceptance of `GHU-010` and `GHU-010F` and makes `GHU-011` ready.
- Scope: Update only `TICKETS.md` and `STATUS.md` with exact accepted identities, retained limitations, mechanical counts, and readiness handoff.
- Non-goals: No product, test, runtime, data, service, remote, publication, commit, merge, deployment, or `GHU-011` implementation action.
- Acceptance: `GHU-010` and `GHU-010F` are accepted with exact review/integration evidence.; `GHU-011` is ready and unassigned.; Counts and next safe action are consistent.
- Validation: Exact HEAD/tree, allowed-path audit, ledger consistency/counts, and `git diff --check` only.
- Risks: Material risk: Evidence loss, overstated delivery, or an implementation assignment would invalidate this correction.
- Stop conditions: Stop on evidence conflict, path drift, any need to alter product/test bytes, or validation failure.
- Authority: Ledger-only closeout and readiness transition; no operational mutation.
- Claims supported: Only the accepted ledger identities and readiness of `GHU-011`.
- Claims unsupported: `GHU-010G` acceptance, `GHU-011` implementation, publication, PR, default-branch merge, deployment, runtime proof, or operational claims.
- Next safe action: Preserve the rejected candidate as blocked evidence; use `GHU-010H` for the smallest ledger-only correction.
- Closeout evidence: On 2026-07-30, exact base commit/tree `13cf3a3b54a4a411465ac570e5ecb65b1669cdc3` / `cf80477be77676f4e8eec54a8aa23d2fd6917896` were verified, then `GHU-010G` transitioned legally `planned -> ready -> active -> review` in run `20260730T134255Z-13cf3a3b54-4c8070`, child `e1c1f6cc7c4b53e89595c31dd6328bfb`. Parent diff inspection found that `GHU-010G` accidentally changed the existing accepted `GHU-002` Claims supported line from its base wording to a false cross-ticket fixture-shell claim. Parent review therefore transitioned `GHU-010G` `review -> blocked`. Commit/PR/merge/deploy/runtime proof: `NOT_OCCURRED`.

## GHU-010H — Correct rejected shell ledger closeout

- Release: `R1`
- Priority: `P1`
- Status: `accepted`
- Dependencies: `GHU-010F`
- Model routing: GPT-5.6 Terra; fallback GPT-5.6 Luna
- Session role: Fresh bounded Codex X ledger-correction implementer
- Outcome: Preserves the correct `GHU-010G` closeout while restoring the accepted `GHU-002` claims boundary byte-for-byte.
- Scope: Reproduce the exact rejected `GHU-010G` ledgers.; Restore only the accepted base wording of the `GHU-002` Claims supported line.; Record the `GHU-010G` rejection and this correction bookkeeping in `TICKETS.md` and `STATUS.md`.
- Non-goals: No product, test, runtime, data, service, remote, publication, commit, merge, deployment, or `GHU-011` implementation action.
- Acceptance: The `GHU-002` Claims supported line exactly matches base `13cf3a3b54a4a411465ac570e5ecb65b1669cdc3`.; All correct `GHU-010G` evidence and readiness changes are preserved.; `GHU-010G` is blocked for the exact cross-ticket corruption.; `GHU-010H` is independently accepted and integrated.; Counts and next safe action are consistent.
- Validation: Exact HEAD/tree and rejected-reference hashes.; Reference comparison limited to the restored `GHU-002` line and G/H bookkeeping.; Allowed-path and mechanical-count checks.; Stale-claim scan and `git diff --check`.
- Risks: Material risk: Cross-ticket claims drift, evidence loss, or preclaimed correction acceptance invalidates this correction.
- Stop conditions: Stop on evidence conflict, path drift, any difference from `GHU-010G` outside the restored line and G/H bookkeeping, or validation failure.
- Authority: Ledger-only correction; no operational mutation.
- Claims supported: Only the accepted ledger correction restoring the `GHU-002` claims boundary while preserving the accepted R1 fixture shell evidence.
- Claims unsupported: Publication, PR, default-branch merge, deployment, runtime proof, or operational claims beyond the accepted fixture shell and dashboard.
- Next safe action: Preserve accepted evidence; the accepted `GHU-011` fixture dashboard closes the R1 gate.
- Closeout evidence: On 2026-07-30, exact base commit/tree `13cf3a3b54a4a411465ac570e5ecb65b1669cdc3` / `cf80477be77676f4e8eec54a8aa23d2fd6917896` and rejected `GHU-010G` ledger hashes were verified. `GHU-010H` transitioned legally `planned -> ready -> active -> review` and was assigned to implementer run `20260730T134724Z-13cf3a3b54-40caa5`, session `019fb347-d93d-7e31-a20d-01d41fa1c7f6`, child `46e2cfc9b75f3ff6170baa9263698df4`. Independent reviewer run `20260730T135324Z-13cf3a3b54-ca20f6`, session `019fb34d-4cd5-7dd2-b7be-d7a8d7c745ff`, returned `ACCEPT_GHU_010H`. Parent decision: accepted and integrated the exact ledger correction as commit `1bacc679377f54433ea757f8cbf7045e3ce8526a`, tree `dee68158b8455d898d60807bdc0ff41c8caf1f7f`. Push/PR/default-branch merge/deploy/runtime proof: `NOT_OCCURRED`.

## GHU-011 — Fixture-backed dashboard overview

- Release: `R1`
- Priority: `P1`
- Status: `accepted`
- Dependencies: `GHU-010`
- Model routing: GPT-5.6 Terra; fallback GPT-5.6 Luna
- Session role: Fresh Codex X implementation session
- Outcome: Makes the main dashboard navigable and reviewable before any live data is connected.
- Scope: Build manual-prediction launch card, collector summary, corpus funnel, model identity, recent predictions, system health and activity feed.; Use typed fixture states for healthy, stale, unavailable, waiting, running and blocked.; Show updated_at and evidence-source affordances on operational cards.
- Non-goals: No live endpoints, real counts, real PIDs, direct service status calls or prediction action.
- Acceptance: Every number is marked fixture/prototype.; Empty, stale and failed states are visually first-class.; Dashboard answers: what can I run, what is happening, what evidence is accumulating, and what is blocked?
- Validation: Component tests for all status variants.; Visual regression or screenshot tests if an existing harness supports them.
- Risks: Material risk: Stop if fixtures diverge from the status vocabulary defined in GHU-001. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop if fixtures diverge from the status vocabulary defined in GHU-001.
- Authority: Level 1 fixture UI only; no operational mutation.
- Claims supported: The accepted R1 fixture-backed dashboard only, with prototype/research-only labeling and fixture-state evidence.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Preserve the accepted R1 fixture checkpoint and parent-accepted/integrated `GHU-016A`, `GHU-020A`, and `GHU-021`; `GHU-022P` is next.
- Closeout evidence: On 2026-07-30, accepted dependency `GHU-010` was verified and `GHU-011` transitioned legally `planned -> ready`. Implementation evidence spans `GHU-011L` run `20260730T135824Z-1bacc67937-e184ab`, session `019fb351-f550-7682-9a15-ba28e7f5e1e0`, and `GHU-011M` run `20260730T181451Z-b836e769ae-738e8d`, session `019fb43c-b9e8-7c30-823d-894c79e30695`. Independent `GHU-011M` review run `20260730T181807Z-e10cff2931-e22d8c`, session `019fb43f-a9bd-7363-851c-d6a392a44548`, returned `ACCEPT_GHU_011M`. The exact product delta is base `6e0b0d99c296a4c984faf0775bab88f8689e66da`, tree `bff53978cdfeea8f604404432e1d672cba95a692`, to head/integration `e10cff293141569b1a5a169dd05efc8109e3c603`, tree `07f02fc46b88b47bf0ade8ee264505f8b47c7d91`, and changes exactly `app.py`; `static/css/operator-ui.css`; `templates/operator_ui.html`; `templates/operator_ui_components.html`; `tests/playwright/operator-ui-shell.spec.js`; `tests/test_operator_ui_shell.py`. Base-to-head binary diff SHA-256: `7641c45d98a26590e64fff71a50d9dc1b7639b03ee1e6c1fb05aedbecd58681b`. Per-file head SHA-256 values in that order: `7d948ad95c314c857b0379f5a2aae63587bf2dfb3a6adc2867f12ac82d23f7f2`; `ce995078d9a7bffa8baea4e924f9a55e2dc272908a9574feb7bce21756e8a9b7`; `961c8054076b04240285acd0a74ea955f8b4b1af99271ae782092c0c2d1c407d`; `203956f714d6e4622ec4c21a1e32d2da62f6c9baf5af6a549297e750543dc17f`; `6bc876014659a24ca072cad73e31d298b898445dc7d4e981e47835486ee0a9e4`; `45b8e96df32c64f2d389cbe64ba86d87e487cf28d5c1942d497fe0a921f9a24c`. Focused command `/tmp/ghu010-validation-73f1e5d/bin/python -m pytest -q tests/test_operator_ui_shell.py` exited 0 with `5 passed`. Private browser server command `PORT=5002 FLASK_ENV=testing MODULE_GUARD_STRICT=0 PREDICTION_IMPORT_MODE=relaxed ENABLE_ENDPOINT_DROPDOWNS=1 TESTING=1 TRAINING_MAX_SECS=30 DISABLE_NAV_DROPDOWNS=1 /tmp/ghu010-validation-73f1e5d/bin/python app.py --host localhost --port 5002` was stopped after validation. Browser command `NODE_PATH=/tmp/ghu010-node-73f1e5d/node_modules PLAYWRIGHT_BROWSERS_PATH=/tmp/ghu010-playwright-browsers /tmp/ghu010-node-73f1e5d/node_modules/.bin/playwright test tests/playwright/operator-ui-shell.spec.js --config=playwright.config.js --project=chromium-desktop --reporter=line --workers=1` exited 0 with `3 passed`. Classifier command `python3 scripts/ci/classify_forecasting_changes.py --base f38a125f6364b8a60d17ae9c971b0ce172874eea --head e10cff293141569b1a5a169dd05efc8109e3c603` exited 0 and selected `full_forecasting`. First broad command `TMPDIR=/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/operator-ui-validation-tmp/ghu011m-e10-full uv run --no-project --with-requirements requirements/all.in python -m pytest -q --noconftest tests/race_collection` exited 1 with `1 failed, 550 passed, 40 subtests passed in 4945.67s`. Isolated diagnostic checkpoint `86f39d54d3949c7bc2b6f670c809a6e5dea5050d`, tree `944336f6465bf4d23786d0fcf9ef894cd04565c2`, changes only `tests/race_collection/test_phase7_runtime_adapter.py`; target/module/full passed 1, 25, and 551 tests plus 40 subtests. It is `EXCLUDED`, `UNMERGED`, and `RETAINED`, not discarded; root cause is `UNKNOWN`. Gate reviewer run `20260730T222316Z-86f39d54d3-b5f20d`, session `019fb520-39bf-7be0-863e-7c3b8ced34bf`, returned verdict A `ACCEPT`. Parent accepted the exact product checkpoint at the head/integration identity above. Publication/push/PR/default-branch merge/deploy/runtime mutation/live proof: `NOT_OCCURRED`.

## GHU-012 — Exact upcoming-race picker prototype

- Release: `R1`
- Priority: `P1`
- Status: `accepted`
- Dependencies: `GHU-011`
- Model routing: GPT-5.6 Terra; fallback GPT-5.6 Luna
- Session role: Fresh Codex X implementation session
- Outcome: Replaces ambiguous free text with an exact date → meeting → race selection and runner confirmation flow.
- Scope: Build fixture date, meeting and race selectors.; Display canonical identity, source URL, jump, venue, race number, distance/grade when available, runners/boxes and runner-set hash.; Define disabled-button reasons for missing identity, jump, runners, supported window or configuration.; Submit an immutable server-issued race ID in the prototype contract.
- Non-goals: No schedule scraping, live race fetch, odds capture or manual free-text bypass.
- Acceptance: The user cannot proceed from an ambiguous or post-jump fixture.; Runner-set confirmation is visible before the action.; AEST is primary display time and the evidence drawer can expose UTC/source time.
- Validation: Interaction tests for valid, ambiguous, post-jump, missing-runner and stale-source fixtures.
- Risks: Material risk: Stop if exact race identity fields are not yet defined by GHU-001. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop if exact race identity fields are not yet defined by GHU-001.
- Authority: Level 1 fixture UI only; no operational mutation.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Preserve accepted R1 evidence and parent-accepted/integrated `GHU-016A`, `GHU-020A`, and `GHU-021`; `GHU-022P` is next.
- Closeout evidence: Accepted R1 checkpoint `51fe070ba2a0778bca0b0334c00cae9d75561952`, tree `0c2b82a391b8dd0a1dcc525cf4203edc910843c1`; reviewer child `eda938ef8b7cf07d87eb37719e0929c2` verdict `PRODUCT ACCEPT` / `GATE ACCEPT`; parent integrated exact checkpoint. Focused pytest `5 passed in 0.50s`; Chromium `5 passed in 6.9s`; classifier `full_forecasting`; broad and stable-node truth is recorded in `GHU-016`.

## GHU-013 — Prediction lifecycle and result prototype

- Release: `R1`
- Priority: `P1`
- Status: `accepted`
- Dependencies: `GHU-011`
- Model routing: GPT-5.6 Terra; fallback GPT-5.6 Luna
- Session role: Fresh Codex X implementation session
- Outcome: Shows the complete request-to-score lifecycle and a truthful result/blocker page using fixtures.
- Scope: Build confirmation dialog for one research-only invocation.; Build progress timeline for precheck, request, claim, attempt, receipt, validation and scoring.; Build terminal pages for PREDICTION_READY and every supported blocker.; Build probability ranking and evidence drawer without EV, staking or best-bet language.
- Non-goals: No live submission, subprocess, automatic retry, outcome display or profitability claim.
- Acceptance: Collector response blockers are distinguished from later validation/scoring blockers.; Refresh/reconnect behavior is represented in fixtures.; A successful result exposes model/config/source/bundle identities.
- Validation: Component and state-transition tests.; No forbidden copy such as 'best bet', 'edge' or 'profit' in operator result surfaces.
- Risks: Material risk: Stop if a status is not grounded in the protocol or product contract. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop if a status is not grounded in the protocol or product contract.
- Authority: Level 1 fixture UI only; no operational mutation.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Preserve accepted R1 evidence and parent-accepted/integrated `GHU-016A`, `GHU-020A`, and `GHU-021`; `GHU-022P` is next.
- Closeout evidence: Accepted R1 checkpoint `51fe070ba2a0778bca0b0334c00cae9d75561952` / `0c2b82a391b8dd0a1dcc525cf4203edc910843c1`; reviewer `PRODUCT ACCEPT` / `GATE ACCEPT`; parent integrated exact checkpoint. Evidence is shared with the atomic tranche and recorded in `GHU-016`.

## GHU-014 — Collector, corpus, models, system-health and audit prototype pages

- Release: `R1`
- Priority: `P1`
- Status: `accepted`
- Dependencies: `GHU-011`
- Model routing: GPT-5.6 Terra; fallback GPT-5.6 Luna
- Session role: Fresh Codex X implementation session
- Outcome: Completes the fixture prototype for monitoring and research governance.
- Scope: Collector page: phase, state age, lock owner, next action, recent captures and blockers.; Corpus page: observed → pre-jump sealed → results present → admitted → closed funnel and exclusions.; Models page: approved baseline, latest research, challengers, hashes, lineage and claims status.; System page: component freshness and deployed-source identity.; Audit page: illustrative append-only action records.
- Non-goals: No service-control buttons, training execution, model pointer updates or promotion action.
- Acceptance: Health is component-specific and freshness-aware.; Corpus readiness is not reduced to a raw DB count.; Models separate immutable identity from evidence quality and role.
- Validation: Fixture and rendering tests.; Claims-language review.
- Risks: Material risk: Stop before adding admin features not necessary for the manual prediction or evidence workflow. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop before adding admin features not necessary for the manual prediction or evidence workflow.
- Authority: Level 1 fixture UI only; no operational mutation.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Preserve accepted R1 evidence and parent-accepted/integrated `GHU-016A`, `GHU-020A`, and `GHU-021`; `GHU-022P` is next.
- Closeout evidence: Accepted R1 checkpoint `51fe070ba2a0778bca0b0334c00cae9d75561952` / `0c2b82a391b8dd0a1dcc525cf4203edc910843c1`; reviewer `PRODUCT ACCEPT` / `GATE ACCEPT`; parent integrated exact checkpoint. Evidence is shared with the atomic tranche and recorded in `GHU-016`.

## GHU-015 — Prototype navigation, accessibility and browser regression

- Release: `R1`
- Priority: `P1`
- Status: `accepted`
- Dependencies: `GHU-011`
- Model routing: GPT-5.6 Terra; fallback GPT-5.6 Luna
- Session role: Fresh Codex X implementation session
- Outcome: Turns the individual fixture screens into a coherent desktop/mobile prototype with repeatable regression coverage.
- Scope: Complete operator-mode navigation and advanced evidence drawers.; Add browser tests for the golden workflow.; Test keyboard navigation, focus, mobile layout, long hashes/paths and stale/error states.; Add print-friendly evidence view where supported.
- Non-goals: No live APIs or operational mutations.
- Acceptance: All four coupled-tranche surfaces are present in the same frozen delta: exact race picker (`GHU-012`), prediction lifecycle/result (`GHU-013`), monitoring/governance pages (`GHU-014`), and navigation/accessibility/browser regression (`GHU-015`).; Golden flow works: dashboard → race → confirm → lifecycle → result/evidence.; Phone width does not cut tables or hide primary actions.; Prototype can be demoed without verbal explanation.
- Validation: Repository-standard browser tests.; Focused accessibility and responsive checks.
- Risks: Material risk: Stop if adding a new browser-test framework would be disproportionate; use the existing harness or document the smallest justified addition. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop if adding a new browser-test framework would be disproportionate; use the existing harness or document the smallest justified addition.
- Authority: Level 1 fixture UI only; no operational mutation.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Preserve accepted R1 evidence and parent-accepted/integrated `GHU-016A`, `GHU-020A`, and `GHU-021`; `GHU-022P` is next.
- Closeout evidence: Accepted R1 checkpoint `51fe070ba2a0778bca0b0334c00cae9d75561952` / `0c2b82a391b8dd0a1dcc525cf4203edc910843c1`; reviewer `PRODUCT ACCEPT` / `GATE ACCEPT`; parent integrated exact checkpoint. Evidence is shared with the atomic tranche and recorded in `GHU-016`.

## GHU-016 — Prototype owner review and frozen UX contract

- Release: `R1`
- Priority: `P1`
- Status: `accepted`
- Dependencies: `GHU-015`
- Model routing: High (GPT-5.6 Sol); fallback Medium
- Session role: Current ChatGPT planning session or fresh independent review
- Outcome: Prevents live integration from cementing an unreviewed workflow.
- Scope: Review screenshots and golden workflow against the approved mockup and user priorities.; Record accepted labels, navigation, hidden advanced controls and mobile behavior.; Classify findings as BLOCKING, IMPORTANT or OPTIONAL.; Freeze the R1 UX contract.
- Non-goals: No implementation by the reviewer and no reopening unrelated architecture.
- Acceptance: Only material workflow/truthfulness issues block R2.; Optional polish is retained as follow-up and does not prevent progress.
- Validation: Visual review bundle plus exact tested commit identity.
- Risks: Material risk: Stop R2 only for a workflow that is misleading, unusable, insecure or inconsistent with evidence boundaries. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop R2 only for a workflow that is misleading, unusable, insecure or inconsistent with evidence boundaries.
- Authority: Level 1 fixture UI only; no operational mutation.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Preserve accepted R1 evidence and parent-accepted/integrated `GHU-016A`, `GHU-020A`, and `GHU-021`; `GHU-022P` is next.
- Closeout evidence: Accepted product checkpoint `51fe070ba2a0778bca0b0334c00cae9d75561952`, tree `0c2b82a391b8dd0a1dcc525cf4203edc910843c1`; reviewer run `20260731T001624Z-51fe070ba2-937103`, child `eda938ef8b7cf07d87eb37719e0929c2`, verdict `PRODUCT ACCEPT` and `GATE ACCEPT`; parent fast-forward integrated exact checkpoint. Focused pytest `5 passed in 0.50s`; Chromium desktop `5 passed in 6.9s`; classifier `full_forecasting`; seven-path binary diff SHA-256 `3f74ec86de1ab68b0fbb1a13125efa6fa416e3cc1fdc9bfc658f97118d3dc135`. Broad timestamped run is `15 failed, 536 passed, 40 subtests passed in 5400.39s`, all 15 path-safety failures; identical stable detached nodes passed `15 passed in 12.47s`. Visual review found no blocking issue; desktop/mobile screenshot hashes are recorded in STATUS/DECISIONS. No push/PR/default-branch merge/deploy/runtime proof occurred.

## GHU-016A — R1 acceptance and frozen UX ledger closeout correction

- Release: `R1`
- Priority: `P1`
- Status: `accepted`
- Dependencies: `GHU-012`, `GHU-013`, `GHU-014`, `GHU-015`, `GHU-016`
- Session role: Fresh bounded Codex X documentation correction implementer
- Outcome: Closes the accepted R1 evidence pair and frozen UX ledger without changing product/test bytes.
- Scope: Update only the four programme ledgers; preserve rejected history; record exact accepted R1 evidence and the narrow R2 dependency refinement.
- Non-goals: No product/test/runtime/service/database/network/Git remote action.
- Acceptance: Candidate records `planned -> ready -> active -> review`; GHU-012–016 are accepted with exact evidence and parent integration identity; GHU-020A is between GHU-020 and GHU-021; dependencies and current pointers are consistent.
- Validation: Exact base/head/tree, four-path allowlist, `git diff --check`, ticket/status/dependency/current-pointer/cross-reference consistency, stale-pointer scan, and repository classifier only.
- Authority: Ledger-only correction; parent review, acceptance, and integration are complete at the recorded identity.
- Claims supported: Parent-accepted corrected documentation closeout and recorded accepted R1 evidence.
- Claims unsupported: Deployment, runtime proof, prediction, training, promotion, betting, or public exposure.
- Next safe action: Preserve the parent-accepted R1 ledger closeout, accepted finite evidence/read-only foundation evidence, and accepted/integrated `GHU-020A` and `GHU-021`; `GHU-022P` is next.
- Closeout evidence: Parent-accepted and integrated corrected closeout at commit `3505efb299d25320fd86a0bf76aef5bf953fb5a7`, tree `567c913b7be46c4d5747c8bf74d2fb4df5d8f664`; this supersedes the rejected `978c4c92514701453f7c8a3252ca33880352764b` candidate while preserving its history.

## GHU-020 — Finite evidence and read-only foundations

- Release: `R2`
- Priority: `P2`
- Status: `blocked`
- Dependencies: `GHU-016`, `GHU-016A` (effective only after parent integration of the closeout)
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: Fresh Codex X implementation session
- Outcome: Provides only the finite evidence envelope, freshness/status vocabulary, server-owned allowlisted no-path reads, hashing/schema/fail-closed mapping, and a read-only DB helper.
- Scope: Own only the foundational finite evidence envelope, freshness/status vocabulary, server-owned allowlisted no-path observation/read primitives, hashing/schema/fail-closed mapping, and a read-only DB helper.; Do not duplicate downstream adapter ownership.
- Non-goals: No UI actions, service calls that mutate state, canonical writes or new evidence construction.
- Acceptance: The finite envelope and freshness/status vocabulary are deterministic.; Server-owned allowlisted reads accept no caller-supplied path and bind evidence to hashes and schemas.; Missing, stale, malformed, conflicting, or unavailable evidence fails closed.; The DB helper cannot perform writes.
- Validation: Focused unit tests prove finite-envelope bounds, freshness/status mapping, hashing/schema checks, and fail-closed states.; Negative tests prove that callers cannot supply arbitrary paths and that filesystem and DB helpers remain read-only.
- Risks: Material risk: Stop if an intended metric has no truthful authoritative source; remove it from the live UI rather than synthesize it. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop if an intended metric has no truthful authoritative source; remove it from the live UI rather than synthesize it.
- Authority: Authenticated Level 1 GET/read-only authority with mandatory access audit.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Preserve the rejected `GHU-020`/`GHU-020B`/`GHU-020C` checkpoints and accepted `GHU-020D`/`GHU-020E` evidence plus accepted/integrated `GHU-020A` and `GHU-021`; proceed to `GHU-022P`.
- Closeout evidence: Local candidate transition `ready -> active -> review -> blocked` from exact clean base `3505efb299d25320fd86a0bf76aef5bf953fb5a7`, tree `567c913b7be46c4d5747c8bf74d2fb4df5d8f664`. Candidate paths are `src/operator_ui/__init__.py`, `src/operator_ui/foundation.py`, `tests/operator_ui/test_foundation.py`, this ticket ledger, and `docs/operator_ui_v1/STATUS.md`. Its worker lacked pytest, so the worker's focused invocation exited before collection. Preserved rejected checkpoint `57256bc7f15b7311e3cffbcd1b5887e10084bfb6`, tree `ef2284ccef699477de55f6581572391335059bac`, subsequently received parent focused result `2 failed, 39 passed`; it is evidence, not accepted history. `GHU-020B` and `GHU-020C` are preserved rejected review evidence. The successor `GHU-020D` correction and finite foundation outcome are accepted at the exact evidence recorded below; this rejected predecessor remains blocked evidence.

## GHU-020B — Correct finite evidence and read-only path binding

- Release: `R2`
- Priority: `P1`
- Status: `blocked`
- Dependencies: `GHU-016`, `GHU-016A`, and preserved `GHU-020` rejected-checkpoint evidence
- Session role: Fresh bounded Codex X correction implementation session
- Outcome: Corrects integrity-state precedence, descriptor-bound allowlisted file/SQLite reads, SQLite attach/detach denial, and immutable-historical freshness semantics only.
- Scope: Repair only `src/operator_ui/foundation.py`, its focused test, and these precise correction ledgers; `src/operator_ui/__init__.py` only if required.
- Non-goals: No `GHU-020A` auth/audit, adapters, API/routes/UI/jobs/runtime/service/generator work, legacy changes, dependencies, arbitrary locator input, writes, browser, collector, locks, or product subprocess execution.
- Acceptance: Integrity failures deterministically map to `INVALID/INTEGRITY_FAILED`; file and SQLite opens remain bound to the canonical configured root/path across parent replacement; SQLite `ATTACH`/`DETACH` are denied; verified historical claims do not expire solely with age and cannot claim current health or present quality.
- Validation: Exact supplied interpreter focused pytest; in-memory compilation without pycache; `git diff --check`; exact allowlist/status checks; repository classifier.
- Authority: Bounded local implementation and validation only.
- Claims supported: Local correction evidence only until independent review and parent acceptance.
- Claims unsupported: Parent acceptance, accepted history, authentication/audit, current-health proof from historical evidence, deployment, runtime/data mutation, prediction, training, promotion, betting, or public exposure.
- Next safe action: Preserve rejected `GHU-020`/`GHU-020B`/`GHU-020C` predecessor and review evidence, accepted `GHU-020D`/formal `GHU-020E` evidence, and accepted/integrated `GHU-020A` and `GHU-021`; `GHU-022P` is next.
- Closeout evidence: Preserved rejected checkpoint HEAD `182fc11dc995d114363e903e5e679aa78edd9602`, tree `9fa01491c2d0f210c989fcc3166e1e6f75e65a4e`, is evidence, not accepted history. Main reviewer session `019fb65c-03ac-7720-b63c-88e18acca58d`, run `20260731T040821Z-182fc11dc9-4738ef`, returned `REJECT_GHU_020B` for exactly two blocking findings: construction retained no root/component/file/database inode identity against later same-path ordinary replacement; and immutable-historical policy accepted arbitrary supported-claim prose while returning `AVAILABLE/FRESH`. `GHU-020C` owns only these two corrections. Parent acceptance/integration, commit/PR/push/merge/deploy/publication/runtime mutation remain pending or `NOT_OCCURRED`.

## GHU-020C — Bind configured identities and finite historical claims

- Release: `R2`
- Priority: `P1`
- Status: `blocked`
- Dependencies: `GHU-016`, `GHU-016A`, plus preserved `GHU-020` and `GHU-020B` rejected-checkpoint/review evidence
- Session role: Fresh bounded Codex X correction implementation session
- Outcome: Retains configured root, component, file, and database identity from construction through observation/connect; replaces historical free prose with finite run- or slice-bound claims and fixed narrow rendering.
- Scope: Only `src/operator_ui/foundation.py`, `tests/operator_ui/test_foundation.py`, `docs/operator_ui_v1/STATUS.md`, and this ticket ledger; `src/operator_ui/__init__.py` only if a structured export is strictly required.
- Non-goals: No authentication, adapters, API, UI, jobs, services, dependencies, browser/collector use, locks, runtime/data mutation, arbitrary locators, merge, push, deploy, or publication.
- Acceptance: Same-path ordinary or symlink replacement of a configured root, nested component, file, or database fails closed after reader/helper construction; unchanged reads/connects pass; missing/unreadable evidence remains truthful and deterministic. Historical policy accepts only an explicit structured historical run or slice claim, renders only the contract-defined fixed narrow claim with displayed age, and rejects arbitrary/current-health/promotion-ready/representative-present-quality claims; future timestamps still fail closed.
- Validation: Exact focused pytest with an available interpreter; compilation without pycache; `git diff --check`; exact allowlist/status/count/cross-reference checks; repository classifier. The broad suite is parent-owned after freeze and is not run here.
- Authority: Bounded local implementation and validation only.
- Claims supported: Frozen local correction evidence only until independent review and parent acceptance.
- Claims unsupported: Acceptance or accepted history; authentication/adapters/API/UI/jobs/service work; current health, promotion readiness, representative present quality, deployment, runtime/data mutation, prediction, training, betting, or public exposure.
- Next safe action: Preserve this rejected checkpoint and exact review evidence; use the accepted `GHU-020D` correction and current ledger-only `GHU-020E` closeout.
- Closeout evidence: Transitioned `planned -> ready -> active -> review` from exact clean preserved rejected base HEAD `182fc11dc995d114363e903e5e679aa78edd9602`, tree `9fa01491c2d0f210c989fcc3166e1e6f75e65a4e`. Focused pytest was attempted with the available `/usr/bin/python3` but pytest is unavailable (`No module named pytest`); no dependency was added. No-pycache compilation and direct focused JSON/SQLite smoke scenarios passed. Candidate checkpoint commit HEAD `1e436a461e51923d97b44929bcb198bc20535e0b`, tree `1425b9d11a3ab192a31f63426f1145b7ea00759e`, exists. Main reviewer session `019fb682-d853-76e2-9d6d-20835750d7bd`, run `20260731T045046Z-1e436a461e-70062d`, returned `REJECT_GHU_020C` for one exact blocker: retained construction identities were verified only before fresh `lstat`/open, allowing a root/component/file/database replacement in that gap to become a new internally consistent accepted identity. Parent integration commit, push, merge, deploy, publication, and runtime/data mutation remain `NOT_OCCURRED`.

## GHU-020D — Close retained-identity verification/open gap

- Release: `R2`
- Priority: `P1`
- Status: `accepted`
- Dependencies: `GHU-016`, `GHU-016A`, plus preserved `GHU-020`, `GHU-020B`, and `GHU-020C` rejected-checkpoint/review evidence
- Session role: Fresh bounded Codex X correction implementation session
- Outcome: Binds every opened root, nested component, source file, and SQLite database descriptor to its retained construction identity through observation/connect.
- Scope: Only `src/operator_ui/foundation.py`, `tests/operator_ui/test_foundation.py`, `docs/operator_ui_v1/STATUS.md`, and this ticket ledger.
- Non-goals: No authentication, adapters, API, UI, jobs, services, dependencies, browser/collector use, locks, runtime/data mutation, canonical database/history access, arbitrary locators, commit, merge, push, deploy, or publication.
- Acceptance: Deterministic replacement of an ordinary same-path root, nested component, source file, or SQLite database after initial construction-binding verification but before or during open/connect fails closed; unchanged reads/connects pass. Opened descriptors for every retained component compare directly with retained construction identities and remain verified through observation/connect.
- Validation: Exact supplied-interpreter focused pytest with external TMPDIR; no-pycache compilation; `git diff --check`; exact allowlist/status/count/cross-reference checks; repository classifier. The rejected candidate broad run remains interrupted with no pass claim and is not rerun.
- Authority: Parent-accepted and integrated bounded foundation correction; no operational mutation.
- Claims supported: The accepted finite evidence/read-only foundation outcome and its exact validation/review/integration evidence only.
- Claims unsupported: Authentication/adapters/API/UI/jobs/service work; R2 completion, deployment, runtime/data mutation, canonical database/history access, prediction, training, promotion, betting, or public exposure.
- Next safe action: Preserve accepted `GHU-020D`, rejected prior `GHU-020E` candidate and non-formal `GHU-020F` evidence, the non-formal `GHU-020G` correction, and accepted/integrated `GHU-020A` and `GHU-021`; proceed to `GHU-022P`.
- Closeout evidence: Transitioned `planned -> ready -> active -> review -> accepted` from exact clean preserved rejected `GHU-020C` candidate HEAD `1e436a461e51923d97b44929bcb198bc20535e0b`, tree `1425b9d11a3ab192a31f63426f1145b7ea00759e`. Final candidate HEAD `8b1ac4235d478a0ef62380bbf61a265731f4d3e4`, tree `612e6a4ce27f9d3bf49274831b8f8e29121d5a12`. These historical `GHU-020C`/`GHU-020D` objects are archived evidence and are not claimed independently recomputable in this worktree. Historical D correction diff SHA-256 `cb3aa4bda38e069bb31800f286a4a3dadd0d2dabe5e11e30312cd8a55e5c1e13` is archived evidence; accepted-base-to-final-product diff SHA-256 `10eae73d4490ee3fd52d722fb1ff8bc3b8a1a7969bfdca034b397ce28de17e1f` remains accepted product evidence. Parent focused validation passed 62 tests. The exact frozen full-forecasting gate command used `PYTHONDONTWRITEBYTECODE=1`, external `TMPDIR`, and `uv run --no-project --with-requirements requirements/all.in python -m pytest -q --noconftest tests/race_collection`; result `551 passed, 40 subtests passed in 6001.15s (1:40:01)`, exit 0. Reviewer session `019fb68d-96c4-7cb3-9cfc-a50c4f49d5cc`, run `20260731T050230Z-8b1ac4235d-ce8207`, returned `ACCEPT_GHU_020D`. Parent mechanically integrated the identical product tree without rejected checkpoint history at HEAD `c95b0467ded033bdb995da7941b44a11a04b22b7`, tree `612e6a4ce27f9d3bf49274831b8f8e29121d5a12`, branch `agent/operator-ui-programme-20260730`. Push, PR, merge, deploy, publication, runtime/data mutation, and canonical database/history access are `NOT_OCCURRED`.

## GHU-020E — Accepted foundation ledger-only closeout

- Release: `R2`
- Priority: `P1`
- Status: `accepted`
- Dependencies: `GHU-020D`, `GHU-016A`
- Session role: Fresh bounded Codex X documentation implementer
- Outcome: Records the parent-accepted `GHU-020D` finite evidence/read-only foundation and exact integration evidence without changing product/test bytes.
- Scope: Update only `docs/operator_ui_v1/STATUS.md` and this ticket ledger; preserve rejected `GHU-020`, `GHU-020B`, and `GHU-020C` histories.
- Non-goals: No code, tests, templates, assets, dependencies, runtime, data, services, Git history, external state, commit, push, PR, merge, deploy, publication, collector/browser, or lock action.
- Acceptance: `GHU-020D` is accepted with exact focused, broad, reviewer, diff, and parent integration identities; counts, dependencies, current pointers, and next-safe-action pointers make `GHU-020A` the next ticket.
- Validation: Exact two-path allowlist, `git diff --check`, status counts, ticket uniqueness/dependencies/current pointers/stale-pointer scan, hashes/identities, and repository classifier only; no product tests or broad suite.
- Authority: Parent-accepted formal ledger closeout as corrected by non-formal `GHU-020G`; no implementer acceptance authority.
- Claims supported: The corrected parent-accepted ledger closeout, recorded parent-accepted `GHU-020D` foundation evidence, and exact rejected-candidate evidence only.
- Claims unsupported: R2 completion, runtime proof, deployment, prediction, training, promotion, betting, public exposure, or any runtime/data mutation.
- Next safe action: Preserve rejected prior `GHU-020E` and non-formal `GHU-020F` candidate evidence, the non-formal `GHU-020G` correction, and accepted/integrated `GHU-020A` and `GHU-021`; `GHU-022P` is the next ticket.
- Closeout evidence: Preserved rejected prior candidate at exact clean base `c95b0467ded033bdb995da7941b44a11a04b22b7` and candidate `7462e42cd191adcbe1adc43ebe420b2c93226455`. Live recomputation produced binary diff SHA-256 `a0aa6cb799d98c0a5b6d27320ffa0328d35857e4499a95f33db63ef8583332da`, candidate `STATUS.md` SHA-256 `cd11d85d32946a7f59d657225deeba569ec34823a4a52c3a07d26a8cca92978a`, and candidate `TICKETS.md` SHA-256 `315c54392da34320cd56f9fe4bb1bba9514fb931129d8f57d07b15eba3f3bc21`. Non-formal `GHU-020F` remains rejected evidence at exact commit `fd87bee4fb912154be33f5cd71e4505a220e323c`, tree `e4c31a7ed695757d10c59d0f32c5f7f5cc692388`, parent `7462e42cd191adcbe1adc43ebe420b2c93226455`; its parent-to-F binary diff SHA-256 is `40b1303fdd2821f776d4b08be10f2839f57c683b905d99159944f8af568dedb8`, accepted-product-base `c95b0467ded033bdb995da7941b44a11a04b22b7` to F binary diff SHA-256 is `2510ece8dbebe136dbc7dfaea887727155879aab9b58499493f221cba7cab0d5`, and F file SHA-256 values are `e033a570f65dd0ae041a6733c7e34efff82bb8513e9011e0e31489ad2e592faa` for `STATUS.md` and `73ca99aa024e580f1577ab87f9abbefccffea12b19c8e9ae43bd8826b1c7dcab` for `TICKETS.md`. Main reviewer session `019fb6fc-f83e-7902-b8e2-2e47f18d1957`, run `20260731T070407Z-fd87bee4fb-450905`, returned `REJECT_GHU_020F` for exactly three stale live Next safe action fields: accepted `GHU-016` still required inspection/integration of already accepted `GHU-016A`; accepted `GHU-016A` still required review of superseded `GHU-020`; blocked `GHU-020B` still required review/acceptance of already rejected `GHU-020C`. `GHU-020G` is the smallest non-formal correction/supersession record for those fields: it changes no formal counts or product, preserves accepted formal `GHU-020E`, accepted R1/`GHU-016A` and `GHU-020D` foundation evidence, and all rejected predecessor/E/F evidence, and creates no recursive closeout. Reviewer/parent freezes final G diff/file hashes externally with the exact commit/tree; self-referential final G hashes are not recorded in these files. Push, PR, merge, deploy, publication, runtime/data mutation, product-test execution, collector/browser use, and lock manipulation are `NOT_OCCURRED`.

## GHU-020A — Connected-mode authentication, session and UI audit prerequisite

- Release: `R2`
- Priority: `P1`
- Status: `accepted`
- Dependencies: `GHU-020E`, `GHU-016A`
- Session role: Fresh bounded Codex X security implementation session
- Outcome: Establishes the default-off connected-mode security and append-only UI operations audit boundary before any operational read disclosure.
- Scope: Add server-side Level-1 authentication/authorization; secure configured secrets and session expiry/rotation; CSRF for authentication forms and reusable CSRF enforcement for later mutation; a separate append-only UI operations/access-audit store. Every authenticated operational GET must append and confirm its audit event before disclosure; append failure returns a deterministic non-operational error with no evidence content. Keep the store separate from the canonical racing DB and future prediction-job DB.
- Non-goals: No public bind, operational POST, arbitrary path, shell, service, lock, browser, canonical DB write, training, promotion, betting, or runtime action.
- Acceptance: Connected mode is default-off; unauthenticated/unauthorized access fails closed; sessions expire/rotate securely; CSRF is reusable; audit append precedes every authenticated operational GET disclosure; append failure emits no evidence; store separation is proven.
- Validation: Focused auth/session/CSRF/hash-chain/append-failure/no-disclosure/separation tests.
- Authority: Authenticated Level-1 security boundary; parent owns acceptance and integration.
- Claims supported: Only the accepted security prerequisite after independent review and parent integration.
- Claims unsupported: Public exposure, mutation, runtime proof, prediction, training, promotion, betting, or canonical DB writes.
- Next safe action: Preserve accepted/integrated `GHU-020A` and `GHU-021`; implement and independently review `GHU-022P` before `GHU-022`.
- Closeout evidence: Accepted source candidate `9394b194c5ebc79e8383fdfc2b9271ffa4678bcc`, tree `408bd8bb416adc409a896a22a211ee77da54f169`; full delta SHA-256 `23bdbd1cecec16035c742b80de1fa7c12db04cd4480b9b5124359b6176353615`; independent reviewer session `019fb741-d824-7a93-8f98-e402de783fb4`, verdict `ACCEPT_GHU_020A`; validation `118 focused passed`, with `535` broad passes plus `16` environment-recovered focused passes (`551` total cases). Parent integrated the accepted tree at commit `d71857e232ce7371280f9e5c56c45be7b9f7f9e5`. Push, PR, default-branch merge, deployment, runtime mutation, and live proof: `NOT_OCCURRED`.

## GHU-021 — Versioned read-only operator API

- Release: `R2`
- Priority: `P2`
- Status: `accepted`
- Dependencies: `GHU-020E`, `GHU-020A`
- Dependency state: Satisfied; both prerequisites are accepted/integrated.
- Model routing: GPT-5.6 Terra; fallback GPT-5.6 Luna
- Session role: Fresh Codex X implementation session
- Outcome: Exposes narrow versioned endpoints for the prototype without allowing operational mutation.
- Scope: Add overview, upcoming races, race detail, recent predictions, prediction detail, collector, corpus, models, system and audit-read endpoints.; Validate response schemas and finite status vocabulary.; Include server time, source timestamps and stale flags.; Reuse existing application routing conventions.
- Non-goals: No POST endpoints, arbitrary paths, shell input, service control or public internet exposure.
- Acceptance: Unknown fields/statuses do not silently pass.; Responses distinguish empty, unavailable, stale and blocked.; No endpoint mutates canonical data or runtime.
- Validation: API schema/contract tests.; Read-only side-effect audit.
- Risks: Material risk: Stop rather than create duplicate API surfaces if an authoritative equivalent already exists; extend it compatibly. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop rather than create duplicate API surfaces if an authoritative equivalent already exists; extend it compatibly.
- Authority: Authenticated Level 1 GET/read-only authority with mandatory access audit.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Preserve accepted/integrated `GHU-021`; implement and independently review `GHU-022P` before `GHU-022`.
- Closeout evidence: Accepted child HEAD `8db34fe53af252fdb6dd743b51d3531fb1f8b618`, tree `af89578953145e5049bb9d2c70f3de150fad86ca`. The programme correction chain is preserved; final correction diff from `f1f1bd96c60d690bb2a7247e79db4cffc6360594` has SHA-256 `d18f42206bfc4a103d4ae352f93f3963f54c62820de7cb5646dbb1860a67dafa`, and the full accepted four-path diff from `d71857e232ce7371280f9e5c56c45be7b9f7f9e5` has SHA-256 `037adf12c9d37c0e96a66e65645392a06482378262ac8b496c0b662253b860ea`. Parent authoritative focused gate: `254 passed in 39.77s`. Final reviewer run `20260731T100618Z-8db34fe53a-a6a1cc`, session `019fb7a3-c1d4-7da1-946e-291651763774`, verdict `ACCEPT_GHU_021`. Parent integrated the exact accepted tree at commit `4a24218379d186d951f47d3fcf0d17d396d7d066`. Classifier selected `full_forecasting` because paths default unknown-to-full; the exact 551-test broad gate is `RUNNING` with no terminal result. Push, PR, default-branch merge, deployment, runtime mutation, and live proof: `NOT_OCCURRED`.

## GHU-022P — Runner-sealed collector current-index v2 prerequisite

- Release: `R2`
- Priority: `P2`
- Status: `ready`
- Dependencies: `GHU-020E`, `GHU-020A`, `GHU-021`, `GHU-000C1`
- Dependency state: Satisfied; every listed prerequisite is accepted/integrated.
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: Fresh bounded collector-contract implementation session
- Outcome: Evolves the sole fixed collector packet into a runner-sealed v2 that can truthfully support the exact upcoming-race UI catalog.
- Scope: Keep the sole server-configured filename `manual_prediction_current_race_index.json`; introduce `collector_current_race_index_v2` while retaining bounded legacy v1 reader compatibility for existing predictor discovery.; During the existing collector refresh/download flow only, derive one non-empty ordered unique final active runner set from accepted canonical-aligned, leakage-safe pre-race CSV/sidecar evidence and seal it into each race.; Preserve every validated v1 race/time field and bind runners, source chain, hashes, packet identity, and publication evidence.; Suggested implementation paths are limited to `scripts/refresh_prejump_upcoming.py`, `race_collection/synchronous_manual_capture.py`, `scripts/shadow_autopilot_daemon.py`, `scripts/predict_race_now.py`, and their focused tests; unrelated files are not pre-authorized.
- Non-goals: No UI adapter; no new fetch, browser, scan, lock, caller path, retry, second collector, canonical DB/history write, or runtime action.; Do not invent runner IDs or pretend any source field exists.; Legacy v1 remains predictor-compatible but is explicitly ineligible as a `GHU-022` UI catalog source.
- Acceptance: Each active runner has integer box, source display name, protocol-compatible normalized uppercase identity, and explicit `ACTIVE` scratch state; include a source-native runner ID only when the accepted source supplies it, otherwise record it explicitly unavailable and never guess.; Reject duplicate box or normalized identity, empty/partial/ambiguous runner sets, unknown scratch state, and noncanonical ordering.; Bind the runner set to exact race URL/date/venue/number/jump identity, named pre-race source URL/timestamp, source file locators beneath the configured evidence root, source byte hashes, and a deterministic runner-set SHA using the existing protocol contract where compatible.; Publication and bounded read safely verify the sealed refresh-report and runner-source chain and fail closed on missing, changed, tampered, stale, unsafe, or identity-mismatched evidence.; `current_race_index_publish` names v2 and exposes the exact packet SHA-256 plus source/runner hashes required to match the current odds-only daemon report; `PUBLISHED` is usable only when packet, publication evidence, source chain, and identities all match.; Preserve atomic canonical write, finite sizes/counts/deadline, v1 behavior, no-retry/no-second-collector boundaries, and predictor compatibility.
- Validation: Focused tests cover v1 migration compatibility; v2 happy path; canonical order/hash determinism; active-runner and exact-race binding; duplicate, partial, ambiguous, and scratch-state rejection; missing/changed/tampered/stale runner and refresh sources; packet/publication SHA mismatch; fixed-root safe reads including no symlink/no caller path; no shell/lock/browser/new fetch; and predictor compatibility.
- Risks: Material risk: a runnerless, partially sealed, source-invented, stale, or identity-divergent packet must fail closed and remain unusable by `GHU-022`.
- Stop conditions: Stop if accepted canonical-aligned leakage-safe pre-race evidence cannot supply the required final active runner set, or if implementation would require a new acquisition/UI path or authority outside the named collector flow.
- Authority: Collector-owned R2 prerequisite; parent review, acceptance, and integration required before `GHU-022`.
- Claims supported: After acceptance, only a verified runner-sealed current-index v2 publication for bounded predictor discovery and later read-only catalog adaptation.
- Claims unsupported: `GHU-022` completion, UI release, live/runtime proof, prediction, canonical writes, training, promotion, EV, staking, betting, or public exposure.
- Next safe action: Assign one fresh bounded implementer, then independently review and parent-integrate the exact v2 delta before `GHU-022`.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, focused commands/exits, reviewer identity/verdict, parent decision, and applicable integration fields.

## GHU-022 — Exact upcoming-race live catalog

- Release: `R2`
- Priority: `P2`
- Status: `planned`
- Dependencies: `GHU-020E`, `GHU-020A`, `GHU-021`, `GHU-022P`, `GHU-000C1`
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: Fresh Codex X implementation session
- Outcome: Populates the race picker from current source-safe upcoming-race evidence.
- Scope: Read/adapt only the fixed collector-owned `manual_prediction_current_race_index.json` packet when it is schema `collector_current_race_index_v2`, runner-sealed under `GHU-022P`, and matched by `current_race_index_publish` plus the sealed refresh/runner-source chain.; Return exact race IDs, TheDogs URL, date, venue, race number, jump and runner set.; Apply P-UPCOMING-300-PREJUMP after packet validation.; Never use legacy v1 as a catalog source, interpret refresh reports independently, browse, scan, fetch, scrape, lock, or start a browser.
- Non-goals: No live capture, schedule refresh, result access or guessing missing runners/times.
- Acceptance: Only exact supported races are selectable.; Ambiguous and post-jump races are excluded with reasons.; Runner identity and set are preserved.
- Validation: Identity, date, venue collision, runner-set and post-jump regressions.
- Risks: Material risk: Stop if live catalog construction would require the web request to acquire the collector lock or launch a browser. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop if live catalog construction would require the web request to acquire the collector lock or launch a browser.
- Authority: Authenticated Level 1 GET/read-only authority with mandatory access audit.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-023 — Prediction bundle and evidence read model

- Release: `R2`
- Priority: `P2`
- Status: `planned`
- Dependencies: `GHU-020E`, `GHU-020A`, `GHU-021`
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: Fresh Codex X implementation session
- Outcome: Lets the UI inspect prior manual prediction attempts and replay evidence without interpreting logs ad hoc.
- Scope: Index isolated prediction bundles through their canonical result and manifest.; Expose exact race/model/config identities, terminal status, probabilities when present and evidence references.; Verify manifest/hash consistency before showing a run as sealed.; Reject added/missing/changed bundle bytes in the view model.
- Non-goals: No replay execution, outcome join, result comparison or bundle rewriting.
- Acceptance: A run is either verified, blocked or unavailable—never partially trusted.; Protocol and later validation blockers remain distinct.; Raw private evidence remains access-controlled.
- Validation: Bundle tamper, missing-file, unknown-status and happy-path tests.
- Risks: Material risk: Stop if current bundle format cannot support a field; display unavailable and record the gap. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop if current bundle format cannot support a field; display unavailable and record the gap.
- Authority: Authenticated Level 1 GET/read-only authority with mandatory access audit.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-024 — Collector and system-status live adapters

- Release: `R2`
- Priority: `P2`
- Status: `planned`
- Dependencies: `GHU-020E`, `GHU-020A`, `GHU-021`
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: Fresh Codex X implementation session
- Outcome: Provides component-level operational truth without giving the web app service authority.
- Scope: Expose deployed source identity where current artifacts prove it.; Expose timer/service status through an approved read-only boundary.; Expose lock owner, phase, state age, next meaningful action and recent captures.; Define healthy/degraded/stale/unavailable rules.
- Non-goals: No start/stop/restart/kill, lock deletion, forced capture or systemd unit editing.
- Acceptance: A timer being enabled alone cannot produce HEALTHY.; Stale state cannot appear green.; Source/runtime divergence is visible.
- Validation: Status-age boundary tests.; Permission and side-effect audit.
- Risks: Material risk: Stop if status requires elevated or mutating access; use existing report artifacts instead. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop if status requires elevated or mutating access; use existing report artifacts instead.
- Authority: Authenticated Level 1 GET/read-only authority with mandatory access audit.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-025 — Corpus and model live adapters

- Release: `R2`
- Priority: `P2`
- Status: `planned`
- Dependencies: `GHU-020E`, `GHU-020A`, `GHU-021`
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: Fresh Codex X implementation session
- Outcome: Connects the corpus funnel and model cards to approved evidence rather than illustrative counts.
- Scope: Read canonical forward-corpus/admission reports and exclusion reasons.; Read finite resolved model/config catalog and immutable hashes.; Expose evidence window, generated timestamp and claims status.; Separate baseline/latest-research/challenger role from promotion readiness.
- Non-goals: No corpus construction, result capture, training, fitting, evaluation or model pointer mutation.
- Acceptance: Training-admissible means the admission contract passed, not that rows exist.; Model cards never claim market edge or production promotion without evidence.; Missing result-publication/closure evidence remains visible.
- Validation: Admission/exclusion and model-catalog contract tests.
- Risks: Material risk: Stop if no current closure-admissible result report exists; show the precise gap. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop if no current closure-admissible result report exists; show the precise gap.
- Authority: Authenticated Level 1 GET/read-only authority with mandatory access audit.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-026 — Live read-only UI integration

- Release: `R2`
- Priority: `P2`
- Status: `planned`
- Dependencies: `GHU-022`, `GHU-023`, `GHU-024`, `GHU-025`
- Model routing: GPT-5.6 Terra; fallback GPT-5.6 Luna
- Session role: Fresh Codex X implementation session
- Outcome: Replaces prototype fixtures with live, freshness-aware read models while retaining a fixture/demo mode.
- Scope: Connect dashboard and detail pages to versioned APIs.; Preserve explicit fixture mode for tests/demo.; Add loading, offline, stale, partial and unavailable states.; Show source/evidence drawers and last refresh.
- Non-goals: No mutation endpoints or prediction launch.
- Acceptance: Connected mode contains no invented operational values.; One broken adapter does not falsely mark the whole system healthy.; Fixture mode cannot be mistaken for live mode.
- Validation: Frontend contract tests and mocked adapter states.; Browser golden path in connected read-only mode.
- Risks: Material risk: Stop if the UI must bypass API contracts to read arbitrary files directly. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop if the UI must bypass API contracts to read arbitrary files directly.
- Authority: Authenticated Level 1 GET/read-only authority with mandatory access audit.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-027 — Read-only release security and acceptance

- Release: `R2`
- Priority: `P2`
- Status: `planned`
- Dependencies: `GHU-026`
- Model routing: Pro (GPT-5.6 Sol Pro); fallback Extra High
- Session role: Fresh independent review session
- Outcome: Proves the read-only dashboard is truthful, bounded and safe to deploy behind a feature flag.
- Scope: Review actual diff, API schemas, tests and side-effect evidence.; Test path traversal, arbitrary file access, stale-state truthfulness and data leakage.; Confirm no canonical writes or service control.; Classify findings BLOCKING, IMPORTANT or OPTIONAL.
- Non-goals: Reviewer does not implement fixes or broaden product scope.
- Acceptance: No blocking security, provenance, identity or truthfulness findings.; Exact reviewed delta is frozen for parent integration.; Supported/unsupported claims are explicit.
- Validation: Targeted security tests and exact-diff review.
- Risks: Material risk: Block only for reachable material risk; record cosmetic and low-probability issues as follow-up. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Block only for reachable material risk; record cosmetic and low-probability issues as follow-up.
- Authority: Authenticated Level 1 GET/read-only authority with mandatory access audit.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-030 — Persistent manual-prediction job and audit contract

- Release: `R3`
- Priority: `P3`
- Status: `planned`
- Dependencies: `GHU-027`
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: Fresh Codex X implementation session
- Outcome: Defines a separate durable control record for one UI-requested prediction without modifying canonical racing data.
- Scope: Define job ID, actor, idempotency key, exact race identity, model/config IDs, timestamps, phase, terminal result and evidence bundle reference.; Choose existing suitable persistence or a separate operations store after audit.; Define append-only audit events and recovery after web-process restart.; Define no-retry and one-subprocess invariants.
- Non-goals: No predictor invocation, canonical DB schema change, training or deployment.
- Acceptance: Canonical racing DB is not used as an ad hoc UI job store.; Duplicate idempotency key returns the existing job.; State transitions are finite and monotonic.
- Validation: Schema/state-machine/idempotency tests.; Crash/restart fixture tests.
- Risks: Material risk: Stop if the proposed persistence would share write authority with canonical evidence or weaken append-only guarantees. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop if the proposed persistence would share write authority with canonical evidence or weaken append-only guarantees.
- Authority: Level 2 only after all R3 gates; deployment/live proof require separate Level 4/explicit authority.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-031 — Fixed-argument manual prediction worker

- Release: `R3`
- Priority: `P3`
- Status: `planned`
- Dependencies: `GHU-030`, `GHU-000C1`
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: Fresh Codex X implementation session
- Outcome: Introduces a single-purpose local worker that invokes the approved predictor exactly once without shell or browser authority.
- Scope: Resolve server-side allowlisted race/model/config/odds inputs.; Build fixed argv using only the server-owned allowlisted current-index locator and evidence root; never use shell=True.; Capture canonical stdout/stderr and bind the resulting isolated bundle.; Persist PID/start/finish/exit/terminal status.; Never retry or substitute another race.
- Non-goals: No second collector/browser, direct capture, lock manipulation, current-time override, arbitrary paths or user-provided command fragments.
- Acceptance: One accepted job launches at most one predictor subprocess.; The predictor remains read-only against canonical history except for normal collector-owned append-only capture.; Worker restart cannot duplicate an already-started job.
- Validation: Subprocess argv and no-shell tests.; Duplicate/restart/timeout/terminal-status tests.; Forbidden-argument assertions.
- Risks: Material risk: Stop if the existing predictor lacks a stable noninteractive contract; repair that contract in a separate narrow ticket rather than wrapping logs heuristically. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop if the existing predictor lacks a stable noninteractive contract; repair that contract in a separate narrow ticket rather than wrapping logs heuristically.
- Authority: Level 2 only after all R3 gates; deployment/live proof require separate Level 4/explicit authority.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-032 — Prediction submission, validation, CSRF and idempotency

- Release: `R3`
- Priority: `P3`
- Status: `planned`
- Dependencies: `GHU-030`, `GHU-031`
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: Fresh Codex X implementation session
- Outcome: Lets the operator submit one exact race safely from the browser.
- Scope: Add a narrow mutation endpoint for race_id, finite model selector, finite config ID, odds source and idempotency key.; Re-resolve and validate race/jump/runner/model/config server-side.; Apply authentication, CSRF and rate/duplicate protection using existing conventions.; Reject unsupported, stale, ambiguous, post-jump or conflicting requests.
- Non-goals: No user paths, executable selection, lock path, output path, current time or arbitrary config JSON.
- Acceptance: Double-click, refresh and retransmit produce one job.; Validation errors are stable and user-readable.; An existing unexpired active request is not bypassed.
- Validation: API security, validation and idempotency tests.; Cross-request race/model confusion regressions.
- Risks: Material risk: Stop if current authentication/CSRF policy is absent or ambiguous; resolve security before exposing mutation. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop if current authentication/CSRF policy is absent or ambiguous; resolve security before exposing mutation.
- Authority: Level 2 only after all R3 gates; deployment/live proof require separate Level 4/explicit authority.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-033 — Prediction progress stream and reconnect

- Release: `R3`
- Priority: `P3`
- Status: `planned`
- Dependencies: `GHU-030`, `GHU-032`
- Model routing: GPT-5.6 Terra; fallback GPT-5.6 Luna
- Session role: Fresh Codex X implementation session
- Outcome: Shows persisted job progress without tying correctness to one long HTTP request.
- Scope: Expose persisted job state through SSE if existing infrastructure supports it, with polling fallback.; Map request, claim, attempt, receipt, validation and scoring events to the UI timeline.; Reconnect to the same job after browser refresh or network interruption.; Keep job store as source of truth.
- Non-goals: No automatic retry, event fabrication or direct tailing of arbitrary logs.
- Acceptance: Client disconnect does not cancel or duplicate the job.; Unknown/missing intermediate events do not create false success.; Terminal state is stable.
- Validation: Reconnect, duplicate listener, out-of-order and terminal-state tests.
- Risks: Material risk: Stop before introducing a new message broker unless existing process boundaries truly require one and the owner approves the added operational surface. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop before introducing a new message broker unless existing process boundaries truly require one and the owner approves the added operational surface.
- Authority: Level 2 only after all R3 gates; deployment/live proof require separate Level 4/explicit authority.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-034 — Live prediction result, evidence and offline replay controls

- Release: `R3`
- Priority: `P3`
- Status: `planned`
- Dependencies: `GHU-023`, `GHU-033`
- Model routing: GPT-5.6 Terra; fallback GPT-5.6 Luna
- Session role: Fresh Codex X implementation session
- Outcome: Presents a verified terminal result and lets the operator inspect or copy replay evidence.
- Scope: Show ranked probabilities for PREDICTION_READY.; Show first exact blocker for failures.; Expose request/claim/attempt/response/receipt/consume identities, hashes, temporal cutoff and bundle manifest.; Offer copy-only offline replay command where already supported.
- Non-goals: No live rerun button, post-race result comparison, EV, staking or betting recommendation.
- Acceptance: Result data is shown only after bundle verification.; Evidence drawer matches raw artifacts.; Replay is clearly offline verification and cannot trigger live acquisition.
- Validation: Verified/tampered/missing bundle frontend and API tests.; Claims-language scan.
- Risks: Material risk: Stop if evidence cannot be verified; display the blocker rather than partial probabilities. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop if evidence cannot be verified; display the blocker rather than partial probabilities.
- Authority: Level 2 only after all R3 gates; deployment/live proof require separate Level 4/explicit authority.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-035 — End-to-end manual prediction safety suite

- Release: `R3`
- Priority: `P3`
- Status: `planned`
- Dependencies: `GHU-031`, `GHU-032`, `GHU-033`, `GHU-034`
- Model routing: Pro (GPT-5.6 Sol Pro); fallback Extra High
- Session role: Fresh independent review and adversarial test session
- Outcome: Proves the UI cannot bypass the collector, duplicate an attempt, mutate protected data or overstate a result.
- Scope: Exercise success and every terminal blocker with synthetic/fixture protocol data.; Test current index missing, stale, unsafe, changed, invalid, noncanonical, oversized, and unbounded evidence; `SKIPPED`, `REJECTED`, and `PUBLISHED` publication states; and packet/root/path injection absence.; Test no-shell, no-path, no-lock, no-browser, no-current-time, no-retry and no-canonical-write invariants.; Test CSRF/auth/idempotency and cross-user/job isolation.; Review exact diff and validation evidence.
- Non-goals: No real race, deployment or live service action.
- Acceptance: One UI submission equals at most one predictor invocation.; Collector remains sole capture authority.; No blocking identity, timing, provenance, security or mutation finding.
- Validation: Classifier-selected focused suites; full forecasting only if required.; Independent exact-head review.
- Risks: Material risk: Any reachable path to duplicate capture, arbitrary command, protected write, outcome leakage or false PREDICTION_READY is blocking. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Any reachable path to duplicate capture, arbitrary command, protected write, outcome leakage or false PREDICTION_READY is blocking.
- Authority: Level 2 only after all R3 gates; deployment/live proof require separate Level 4/explicit authority.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-036 — Feature-flagged generated deployment package

- Release: `R3`
- Priority: `P3`
- Status: `planned`
- Dependencies: `GHU-035`
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: Fresh Codex X implementation session
- Outcome: Makes the accepted UI deployable and reversible through the repository-owned service/configuration path.
- Scope: Add feature flag and generated configuration using current conventions.; Document bind address, local/Tailscale access boundary, process layout and secrets handling.; Document enable, verify and rollback procedures.; Keep existing UI available until acceptance.
- Non-goals: No actual deployment, service restart, public exposure or live prediction.
- Acceptance: Deployment artifacts are generated from repository code.; Rollback disables the feature without deleting audit or prediction evidence.; Default remains safe if configuration is missing.
- Validation: Generator/config tests and static deployment review.
- Risks: Material risk: Stop before manual editing of generated user-service files or widening network exposure. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop before manual editing of generated user-service files or widening network exposure.
- Authority: Level 2 only after all R3 gates; deployment/live proof require separate Level 4/explicit authority.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-037 — Bounded deployed acceptance and one-race UI proof

- Release: `R3`
- Priority: `P3`
- Status: `planned`
- Dependencies: `GHU-036`
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: New Codex live-runtime session after explicit owner authorization
- Outcome: Establishes whether the deployed UI can complete one exact request → claim → response → consume → score chain.
- Scope: Deploy exact reviewed source through the generator.; Observe one natural timer cycle.; Select one suitable upcoming race.; Run exactly one UI prediction and preserve the fixed packet, source refresh report, `current_race_index_publish`, and all packet/source/publication hashes with UI, protocol, and bundle evidence.; Verify displayed state against raw artifacts.
- Non-goals: No second attempt, alternate race, training, promotion, service workaround, outcome access, EV, staking or betting.
- Acceptance: One click creates one job and one invocation.; UI timeline and terminal result match protocol evidence.; No prohibited mutation occurred.; Runtime-proven is claimed only if scoring completes with valid provenance.
- Validation: Exact runtime identity, hashes, protocol paths and bundle manifest.; Post-run no-retry/no-outcome/no-prohibited-mutation audit.
- Risks: Material risk: Requires a separate explicit live-action authorization. Stop after the first terminal result. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Requires a separate explicit live-action authorization. Stop after the first terminal result.
- Authority: Level 2 only after all R3 gates; deployment/live proof require separate Level 4/explicit authority.
- Claims supported: Only the exact naturally reached deployed path, and runtime prediction proof only after valid scoring provenance.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-040 — Forward-corpus readiness and exclusion drill-down

- Release: `R4`
- Priority: `P4`
- Status: `planned`
- Dependencies: `GHU-027`
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: Fresh Codex X implementation session
- Outcome: Lets the owner see exactly how close the prospective corpus is to a legitimate experiment.
- Scope: Show evidence window and funnel stages.; Drill into pending result evidence and exclusions.; Expose packet/report identity and top blockers.; Show readiness state without equating a count with training authority.
- Non-goals: No corpus mutation, result collection, historical reconstruction or training.
- Acceptance: Every count is reproducible from an approved report.; Result publication/closure gaps remain explicit.; No market-edge claim.
- Validation: Report-contract and exclusion-drilldown tests.
- Risks: Material risk: Stop if current evidence cannot support the requested aggregation. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop if current evidence cannot support the requested aggregation.
- Authority: Level 3 read and non-executing draft-spec authority only.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-041 — Model lineage and evidence comparison

- Release: `R4`
- Priority: `P4`
- Status: `planned`
- Dependencies: `GHU-025`
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: Fresh Codex X implementation session
- Outcome: Provides a truthful model catalog and comparison surface without activating any model.
- Scope: Show immutable artifact/config/manifest hashes and roles.; Show fit/evaluation window and qualifying sample when evidenced.; Show relevant market baseline metrics and uncertainty where available.; Expose permitted and prohibited claims.
- Non-goals: No model fitting, selector mutation, promotion or live deployment.
- Acceptance: Artifact identity and evidence quality are separate.; Small/exploratory samples cannot appear promotion-ready.; Missing evaluation evidence is visible.
- Validation: Model catalog and evidence-report contract tests.
- Risks: Material risk: Stop rather than infer evaluation metrics from incomplete artifacts. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop rather than infer evaluation metrics from incomplete artifacts.
- Authority: Level 3 read and non-executing draft-spec authority only.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-042 — Experiment specification builder

- Release: `R4`
- Priority: `P4`
- Status: `planned`
- Dependencies: `GHU-040`, `GHU-041`
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: Fresh Codex X implementation session
- Outcome: Lets the owner prepare a complete, reviewable Phase 7 experiment contract without training a model.
- Scope: Require hypothesis, baseline, frozen features/config, eligible population, exclusions, train/validation/test windows, leakage controls, metrics, uncertainty, minimum sample and strongest claim.; Resolve corpus/model identities server-side.; Produce a deterministic draft specification and review summary.; Record owner approval state separately from draft creation.
- Non-goals: No fitting, evaluation, artifact persistence, registration or promotion.
- Acceptance: Incomplete experiment contracts cannot be marked ready.; Forward/OOS boundaries and market baseline are explicit.; Draft is reproducible and immutable once approved.
- Validation: Schema, window-overlap, leakage and missing-field tests.
- Risks: Material risk: Stop if corpus closure or model input contracts are not yet honest enough to define the experiment. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop if corpus closure or model input contracts are not yet honest enough to define the experiment.
- Authority: Level 3 read and non-executing draft-spec authority only.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-043 — Authorised experiment execution gateway

- Release: `R5 deferred`
- Priority: `Deferred`
- Status: `deferred`
- Dependencies: `GHU-042`, separate training authority
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: New Codex experiment session
- Outcome: Eventually executes an explicitly authorised frozen experiment in isolation.
- Scope: Consume only an approved experiment specification and frozen corpus.; Fit candidate, evaluate against declared baseline and emit lineage/evidence artifacts.; Keep execution isolated from collector/manual prediction.
- Non-goals: Not authorised by this system plan.; No automatic activation, promotion, betting or production pointer change.
- Acceptance: Execution cannot begin without explicit owner authorization bound to one exact approved/frozen `GHU-042` specification, closure-admissible corpus/source manifest, code commit/tree, model/config/feature identities, train/validation/test windows, exclusions, seeds, and environment/dependency manifest.; Train/validation/test and genuine forward/OOS boundaries are temporally disjoint and leakage-audited, with no preclosure outcome leakage or same-day/undated unsafe history.; Execution remains isolated from collector, browser, locks, manual-prediction/UI runtime, canonical racing DB/history writes, model registry/pointer, deployment, activation, and promotion.; Output is one immutable, hash-complete experiment package with exact lineage, declared baseline/market comparison, metrics, uncertainty, minimum sample, segment/stability results, exclusions/failures, resource/process identity, and no automatic action.; These criteria specify a future gate only and grant no present authority.
- Validation: Schema, hash, and frozen-identity checks.; Window-overlap and leakage negative tests.; Deterministic seed and reproducibility checks.; Isolation and no-lock/no-browser/no-canonical-write/no-registry/no-runtime assertions.; Baseline, uncertainty, and stability checks.; Package tamper tests.; Fresh independent exact-package review.; These criteria specify a future gate only and grant no present authority.
- Risks: Material risk: Remain deferred until the evidence and authority gates are satisfied. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Remain deferred until the evidence and authority gates are satisfied.
- Authority: No present execution authority.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Remain deferred pending a new contract and separate authority.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-044 — Promotion review package, never automatic promotion

- Release: `R5 deferred`
- Priority: `Deferred`
- Status: `deferred`
- Dependencies: qualifying forward evaluation, separate owner authority
- Model routing: Pro (GPT-5.6 Sol Pro); fallback Extra High
- Session role: New independent claim-audit session
- Outcome: Eventually prepares a human-reviewable promotion recommendation with rollback and monitoring evidence.
- Scope: Assemble reproducible lineage, genuine forward/OOS evaluation, market comparison, uncertainty, segment stability, unresolved risks, rollback and monitoring.; Present approve/reject/defer review—not an automatic switch.
- Non-goals: No automatic model promotion, registry mutation, runtime change or betting.
- Acceptance: A package is eligible only under a future accepted replacement evidence/promotion contract and separate owner authority, with exact lineage from a frozen model/config/source/corpus through a genuine forward/OOS evaluation.; It includes declared baseline and market comparison, sample sufficiency, uncertainty/calibration, segment and temporal stability, exclusions, unresolved risks, rollback plan, monitoring thresholds/owners, deployment compatibility, and reproducible hashes.; Output is review-only approve/reject/defer material, with no registry/pointer/runtime/deployment mutation and never automatic promotion.; These criteria specify a future gate only and grant no present authority.
- Validation: Independent high-risk claim audit of exact package hashes/lineage, temporal and population separation, baseline/market comparison, uncertainty/calibration/sample sufficiency, segment/temporal stability, unresolved-risk completeness, rollback and monitoring testability, missing/stale/divergent/tamper rejection, and no automatic mutation.; These criteria specify a future gate only and grant no present authority.
- Risks: Material risk: Remain deferred while the default claims boundary prohibits promotion. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Remain deferred while the default claims boundary prohibits promotion.
- Authority: No present execution authority.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Remain deferred pending a new contract and separate authority.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.
