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
- Status: `active`
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
- Next safe action: Freeze and validate this exact four-file candidate, then independent review and parent exact-delta acceptance/integration; the later `GHU-010` bounded candidate must atomically record this ticket accepted with exact commit/tree/reviewer/parent-decision evidence and record `GHU-010`'s legal `ready -> active` transition before any `GHU-010` product work starts or lands.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-010 — Design tokens and responsive application shell

- Release: `R1`
- Priority: `P1`
- Status: `planned`
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
- Next safe action: Wait for parent-accepted integration of `GHU-002`; then the fresh bounded candidate must atomically record exact `GHU-002` commit/tree/reviewer/parent-decision evidence, record `GHU-010`'s legal `ready -> active` transition before product work, and record `active -> review` plus focused validation before freezing the same product-and-ledger delta.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-011 — Fixture-backed dashboard overview

- Release: `R1`
- Priority: `P1`
- Status: `planned`
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
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-012 — Exact upcoming-race picker prototype

- Release: `R1`
- Priority: `P1`
- Status: `planned`
- Dependencies: `GHU-010`
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
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-013 — Prediction lifecycle and result prototype

- Release: `R1`
- Priority: `P1`
- Status: `planned`
- Dependencies: `GHU-010`, `GHU-012`
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
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-014 — Collector, corpus, models, system-health and audit prototype pages

- Release: `R1`
- Priority: `P1`
- Status: `planned`
- Dependencies: `GHU-010`, `GHU-011`
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
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-015 — Prototype navigation, accessibility and browser regression

- Release: `R1`
- Priority: `P1`
- Status: `planned`
- Dependencies: `GHU-011`, `GHU-012`, `GHU-013`, `GHU-014`
- Model routing: GPT-5.6 Terra; fallback GPT-5.6 Luna
- Session role: Fresh Codex X implementation session
- Outcome: Turns the individual fixture screens into a coherent desktop/mobile prototype with repeatable regression coverage.
- Scope: Complete operator-mode navigation and advanced evidence drawers.; Add browser tests for the golden workflow.; Test keyboard navigation, focus, mobile layout, long hashes/paths and stale/error states.; Add print-friendly evidence view where supported.
- Non-goals: No live APIs or operational mutations.
- Acceptance: Golden flow works: dashboard → race → confirm → lifecycle → result/evidence.; Phone width does not cut tables or hide primary actions.; Prototype can be demoed without verbal explanation.
- Validation: Repository-standard browser tests.; Focused accessibility and responsive checks.
- Risks: Material risk: Stop if adding a new browser-test framework would be disproportionate; use the existing harness or document the smallest justified addition. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop if adding a new browser-test framework would be disproportionate; use the existing harness or document the smallest justified addition.
- Authority: Level 1 fixture UI only; no operational mutation.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-016 — Prototype owner review and frozen UX contract

- Release: `R1`
- Priority: `P1`
- Status: `planned`
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
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-020 — Read-only source adapters and freshness contract

- Release: `R2`
- Priority: `P2`
- Status: `planned`
- Dependencies: `GHU-016`
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: Fresh Codex X implementation session
- Outcome: Creates pure, testable adapters from current artifacts/services to truthful UI view models.
- Scope: Implement adapters for system version, collector state, prediction bundles, corpus status and model catalog.; Attach source path/identity, generated_at, observed_at and stale-after rules.; Map missing/malformed/conflicting evidence to unavailable or blocked states.; Keep database access read-only and avoid deriving training readiness from unapproved counts.
- Non-goals: No UI actions, service calls that mutate state, canonical writes or new evidence construction.
- Acceptance: Adapters are deterministic and pure where possible.; Every displayed field carries provenance/freshness metadata.; Malformed artifacts fail closed.
- Validation: Focused unit tests with valid, missing, stale, malformed and conflicting fixtures.; Read-only filesystem/DB assertions.
- Risks: Material risk: Stop if an intended metric has no truthful authoritative source; remove it from the live UI rather than synthesize it. Missing, stale, malformed, conflicting, or unavailable evidence must fail closed.
- Stop conditions: Stop if an intended metric has no truthful authoritative source; remove it from the live UI rather than synthesize it.
- Authority: Authenticated Level 1 GET/read-only authority with mandatory access audit.
- Claims supported: Only the ticket outcome after validation, independent review, and parent acceptance; no implementation claim while incomplete.
- Claims unsupported: Market edge, profitability, EV, staking, betting, public exposure, training, promotion, and any runtime or data mutation outside this ticket.
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-021 — Versioned read-only operator API

- Release: `R2`
- Priority: `P2`
- Status: `planned`
- Dependencies: `GHU-020`
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
- Next safe action: Wait until every listed prerequisite is accepted/integrated, then parent may mark ready and assign one fresh bounded implementer.
- Closeout evidence: PENDING: exact base/head/tree/path/diff identities, commands/exits, focused tests, reviewer identity/verdict, parent decision, and applicable commit/PR/merge/deploy/proof fields.

## GHU-022 — Exact upcoming-race live catalog

- Release: `R2`
- Priority: `P2`
- Status: `planned`
- Dependencies: `GHU-020`, `GHU-021`
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: Fresh Codex X implementation session
- Outcome: Populates the race picker from current source-safe upcoming-race evidence.
- Scope: Return exact race IDs, TheDogs URL, date, venue, race number, jump and runner set.; Expose source timestamp and ambiguity/exclusion reasons.; Never perform a browser scrape from a page request.; Use current source-safe cached/scheduled evidence only.
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
- Dependencies: `GHU-020`, `GHU-021`
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
- Dependencies: `GHU-020`, `GHU-021`
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
- Dependencies: `GHU-020`, `GHU-021`
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
- Dependencies: `GHU-030`
- Model routing: GPT-5.6 Sol; fallback GPT-5.6 Terra
- Session role: Fresh Codex X implementation session
- Outcome: Introduces a single-purpose local worker that invokes the approved predictor exactly once without shell or browser authority.
- Scope: Resolve server-side allowlisted race/model/config/odds inputs.; Build an argv list for the existing predictor; never use shell=True.; Capture canonical stdout/stderr and bind the resulting isolated bundle.; Persist PID/start/finish/exit/terminal status.; Never retry or substitute another race.
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
- Scope: Exercise success and every terminal blocker with synthetic/fixture protocol data.; Test no-shell, no-path, no-lock, no-browser, no-current-time, no-retry and no-canonical-write invariants.; Test CSRF/auth/idempotency and cross-user/job isolation.; Review exact diff and validation evidence.
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
- Scope: Deploy exact reviewed source through the generator.; Observe one natural timer cycle.; Select one suitable upcoming race.; Run exactly one UI prediction and preserve UI, protocol and bundle evidence.; Verify displayed state against raw artifacts.
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
