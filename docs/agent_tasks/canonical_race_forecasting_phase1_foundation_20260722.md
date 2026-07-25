---
job_id: canonical_race_forecasting_phase1_foundation_20260722
title: Canonical race forecasting Phase 1 foundation
lane: Architecture
supporting_lanes:
  - Provenance
  - Data Quality
  - Evaluation
  - Operations
owner: Codex X
approval_required: false
approval_source: "Owner approved the design during the 2026-07-22 grill-with-docs session."
allow_unapproved_safe_extension: false
timeout_seconds: 21600
mutation_mode: isolated_clone_only
production_data_access: false
production_data_boundary: "Canonical repository and ignored runtime artifacts are read-only references. Copy only task-required artifacts into the isolated clone; do not control services, write production databases, collect live data, or alter canonical runtime state."
github_mutation_allowed: false
git_history_mutation_allowed: true
live_service_mutation_allowed: false
docs_impact: DOCS_REQUIRED
task_tier: critical
recommended_model: high_reasoning
worker_model_allowed: false
escalation_needed: false
---

# Canonical Race Forecasting Phase 1 Foundation

## Objective

Establish the isolated, testable foundation for the canonical Race Collection
Service without changing deployed automation or production data. Reconcile the
actual canonical daemon and runtime artifact shapes against
`docs/CANONICAL_RACE_FORECASTING_SPEC.md`, then implement the minimum domain,
operations-state, and immutable-artifact seams needed by later phases.

## Session boundary

This ticket describes the full programme below, but the first new Codex X
session must complete **Phase 1 only**. It must not continue into Phase 2 in the
same session.

At Phase 1 completion, the agent must:

1. Validate and commit the isolated changes in tiny coherent commits.
2. Write a concise Phase 1 completion/remaining-work handoff.
3. Tell the owner: **Start a new Codex X session to continue with Phase 2.**
4. Provide the exact prompt for that next session, referencing this ticket and
   completed commit(s).

Every later phase follows the same rule: complete one bounded phase, stop, and
tell the owner to start a fresh Codex X session for the next phase.

## Required read-only discovery

Before editing, inspect from the isolated clone:

- `CONTEXT.md`
- `docs/CANONICAL_RACE_FORECASTING_SPEC.md`
- `docs/adr/0001-*.md` through `0008-*.md`
- `scripts/shadow_autopilot_daemon.py`
- `scripts/prejump_prediction_loop.py`
- `scripts/shadow_autopilot_v1.py`
- `automation_scheduler.py`
- `ops/systemd/*.service` and `ops/systemd/*.timer`
- `app.py`, `enhanced_prediction_service.py`, `prediction_pipeline_v4.py`,
  `ml_system_v4.py`, `model_registry.py`
- Evidence, identity, odds, result, manifest, and reconciliation utilities and
  their focused tests.

The owner has stated that the Codex X launcher may expose ignored models,
databases, and runtime state from the canonical repository as read-only inputs.
If needed, explicitly identify the smallest safe artifacts and copy them into
the isolated clone. Record source path, destination path, size, checksum, and
purpose. Never copy credentials, secrets, environment files, broad logs, or
unrelated state.

Do not assume checked-in systemd files identify the live runtime. Compare their
paths and contracts with the canonical read-only deployment evidence when that
evidence is safely available. Report unknowns rather than inventing them.

## Full programme

### Phase 1 — Domain and persistence foundation

Deliver:

- A focused package/module boundary for collection-domain types and operations.
- Typed immutable IDs and records for RacingDay, RaceId, DogId, Dog Run, Run
  Observation, evidence artifacts, operations, quarantine, and supersession.
- A forward-only race lifecycle with legal transition validation.
- A racing-day aggregate/barrier model.
- A separate SQLite operations store with WAL, explicit migrations, foreign
  keys, uniqueness constraints, idempotent operation IDs, and transactional
  repository methods.
- A content-addressed artifact-store interface and safe local implementation
  with atomic writes and checksum verification.
- No integration into the live daemon yet.

Required Phase 1 tests:

- Every legal and illegal lifecycle transition.
- Idempotent replay of operations.
- Prediction-before-result barrier.
- Supersession without backward mutation.
- Race/source alias uniqueness.
- Dog Run uniqueness by DogId and local racing date.
- Multiple Run Observations without duplicated starts/wins.
- Transaction rollback and concurrent reader behaviour.
- Artifact atomicity, checksum mismatch, duplicate content, and path safety.
- Migration from an empty database and repeat migration.

Phase 1 documentation:

- Operations schema and ownership.
- Artifact layout and checksum rules.
- Mapping from existing daemon concepts to new domain concepts.
- Explicit deferred decisions for Phase 2.

Phase 1 hard stops:

- No service/timer edits.
- No production DB writes or migrations.
- No network calls or live collection.
- No prediction/training behaviour changes.
- No copying broad canonical runtime directories.
- No speculative legacy deletion.
- No Phase 2 implementation.

### Phase 2 — Inventory, identity, collection, and sealing

- Add expected race-programme inventory and source adapters.
- Implement RaceId aliases and tiered DogId resolution.
- Ingest complete and provisional Dog Runs without duplication.
- Implement adaptive odds scheduling and append-only capture observations.
- Build raw/normalized Sealed Race Evidence packages.
- Enforce actual-jump or scheduled-minus-buffer freeze policy.
- Add field-level source authority and critical-conflict quarantine.
- Produce Racing Day reconciliation from authoritative operations state.
- Run observation-only against existing outputs; do not assume authority.

### Phase 3 — Deferred prediction and result barriers

Implemented and repair-gate validated on the isolated Phase 3 branch through
the cumulative independent review. Evidence and the strict Phase 4 boundary
are recorded in `docs/CANONICAL_RACE_COLLECTION_PHASE3.md`.

- Import the active model as an explicitly `legacy-origin` bundle after safe
  read-only artifact discovery and checksum capture.
- Pin bundle/release/policy per Racing Day.
- Run per-race deferred prediction after day closure from sealed inputs only.
- Commit or quarantine each prediction independently.
- Begin official result collection only after the day prediction barrier.
- Add bounded result retries and training-example joins.
- Keep on-demand forecasts separate from evaluation forecasts.

### Phase 4 — Canonical model bundle and serving path

- Implement the immutable bundle schema and one champion-pointer loader.
- Require prediction provenance fields from the specification.
- Implement pure, versioned feature derivation from sealed evidence.
- Route the canonical API through one service.
- Convert legacy endpoints to thin adapters.
- Remove production fallback, mock, SP tie-break, and GPT-rerank semantics from
  the canonical path without prematurely deleting compatibility surfaces.

### Phase 5 — Corpus and ordered-finish challenger

- Audit and deduplicate the Legacy Training Corpus.
- Build canonical forward Training Examples.
- Implement a coherent ordered-finish forecast contract.
- Train an initial Plackett-Luce-style challenger.
- Add win/top-N/exacta/trifecta derivation and ranking.
- Score only evaluation-eligible unambiguous outcomes initially.
- Keep wagering strategy reports separate and report-only.

### Phase 6 — Evaluation, drift, and promotion

- Implement paired ordered-finish negative log-likelihood comparison.
- Add calibration, rank, containment, exact-order, coverage, and slice metrics.
- Add long-horizon scorecard and short-horizon monitor.
- Implement versioned promotion gates, bootstrap evidence, nomination, atomic
  next-day promotion, provenance display, and rollback.
- Training remains separate; collector emits requests only.

### Phase 7 — Operational cutover and hardening

- Replace overlapping scheduling with the one Race Collection Service.
- Generate generic systemd units from versioned release configuration.
- Add shared administrative commands, alerts, scoped pauses, backup, and restore
  drills.
- Observation-only comparison, then two-day cutover with rollback.
- Fourteen-day probation before legacy retirement or automatic promotion.
- Remove or archive legacy authority only after evidence gates pass.

## Phase 1 implementation constraints

- Prefer deep modules with narrow interfaces over a new collection of helper
  scripts.
- Keep domain language aligned with `CONTEXT.md`.
- Preserve all unrelated user changes.
- Use explicit UTC instants plus official local racing dates and timezones.
- Store timestamps timezone-aware.
- Treat enum/string state values as persisted contracts.
- Make invalid states unrepresentable where practical and validated at storage
  boundaries.
- Never make filesystem reports a second source of truth.
- Do not import the Flask app, scrapers, ML systems, or existing monolithic
  orchestrators into the domain/persistence foundation.
- Keep SQLite-specific details behind the operations-store interface.
- Use repository-standard formatting, type checking, and tests.

## Validation

For Phase 1:

- Run all new focused tests.
- Run the repository's relevant existing manifest, lifecycle, daemon, and
  database tests to prove no regression.
- Run formatter/linter/type checks applicable to changed files.
- Run the broad test suite when feasible; report exact exclusions and reasons.
- Inspect the final diff for accidental runtime or production-path changes.
- Demonstrate an isolated lifecycle from discovery through training-example
  readiness using only ephemeral operations DB and artifacts.
- Demonstrate crash/retry idempotency and a rejected result-before-prediction
  transition.

## Phase 1 acceptance criteria

- Domain names match `CONTEXT.md` and do not depend on legacy filenames.
- Operations state has one transactional authority.
- Lifecycle transitions and day barriers are enforced, not merely documented.
- Operation replay cannot duplicate state or artifacts.
- Dog Run/Observation schema prevents duplicated starts while preserving every
  source observation.
- Artifacts are immutable, content-addressed, atomically written, and verified.
- Empty/repeated migrations and rollback behaviour are tested.
- Existing daemon behaviour is unchanged.
- Phase 2 unknowns and adapter seams are documented with evidence.
- Changes are organized into tiny coherent commits.
- The completion response explicitly stops and instructs the owner to start a
  new Codex X session for Phase 2.

## Suggested tiny commits

1. `docs: add canonical forecasting domain and architecture records`
2. `feat: add collection domain identities and lifecycle`
3. `feat: add transactional operations store and migrations`
4. `feat: add content-addressed evidence artifact store`
5. `test: prove lifecycle barriers idempotency and artifact integrity`
6. `docs: map legacy daemon concepts and hand off phase 2`

Adjust commit boundaries to the actual codebase, but do not combine unrelated
domain, storage, integration, and documentation changes.

## Next-session opening prompt

Use the following in a fresh Codex X session after this ticket and specification
have been deliberately integrated into the canonical repository:

> Implement Phase 1 only from
> `docs/agent_tasks/canonical_race_forecasting_phase1_foundation_20260722.md`.
> Read `docs/CANONICAL_RACE_FORECASTING_SPEC.md`, `CONTEXT.md`, and ADRs 0001–0008
> first. Inspect the canonical daemon and the smallest task-required ignored
> runtime artifacts as read-only evidence, copying only explicitly needed files
> into the isolated clone with checksums. Do not mutate services, production
> databases, canonical runtime state, or prediction behaviour. Build and test
> the domain, SQLite operations store, and content-addressed artifact foundation;
> commit in tiny coherent commits. Stop after Phase 1, tell me to start a new
> Codex X session for Phase 2, and provide that session's exact prompt.
