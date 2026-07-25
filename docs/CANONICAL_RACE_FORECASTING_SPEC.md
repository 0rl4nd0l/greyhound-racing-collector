# Canonical Race Forecasting and Collection Specification

## Purpose

Replace the overlapping prediction, training, artifact-loading, and unattended
automation paths with one auditable system that collects complete pre-jump
evidence, seals it immutably, computes reproducible deferred predictions before
labels are collected, and evolves models through trustworthy champion-versus-
challenger evaluation.

This specification governs the target architecture. Existing V3, V4, unified,
enhanced, shadow, and legacy automation paths are migration inputs, not competing
authorities.

## Outcomes

The system must provide:

1. One authoritative Race Collection Service.
2. One forward-only lifecycle per race and one barrier-controlled lifecycle per
   racing day.
3. Immutable raw and normalized evidence captured before jump.
4. Deferred snapshot predictions computed after day closure and before result
   access.
5. Separately stored on-demand forecasts for immediate pre-jump use.
6. One champion pointer resolving one immutable model bundle.
7. One coherent ordered-finish forecast from which win, top-N, exacta, trifecta,
   and ranking outputs are derived.
8. Long-horizon promotion and short-horizon drift monitoring with distinct
   authority.
9. Provenance-preserving Dog Run deduplication across full results and embedded
   form-guide histories.
10. Reconciliation, recovery, release, backup, and operator controls whose
    success can be proven.

## Non-goals

- The service does not place bets.
- The collection process does not train or promote models.
- A nightly scorecard does not tune mutable production configuration.
- Embedded per-dog history does not fabricate complete races.
- JSON status files and reports are not workflow authority.
- Legacy fallbacks, synthetic models, SP tie-breaking, and GPT reranking are not
  part of the canonical forecast.

## System boundary

### Race Collection Service

One scheduler owns:

- Expected race-programme discovery.
- Race-card and form-guide collection.
- Adaptive append-only odds capture.
- Lifecycle observation and day closure.
- Raw and normalized evidence sealing.
- Deferred prediction orchestration.
- Result collection after the prediction barrier.
- Training-example joins and reconciliation.
- Training requests, but not training execution.

The process supervisor starts and restarts this service. Independent timers may
not perform collection or advance workflow state. Manual tools and dashboards
must call the same application commands as the scheduler.

### Model Training Workflow

A separate workflow consumes immutable training examples and may create
immutable challenger bundles. It cannot modify the champion in the same run.

### Promotion Workflow

A separately auditable policy evaluates already registered immutable bundles.
It may atomically update the champion pointer only after all promotion gates
pass. A promotion becomes effective for the next racing day.

## Authoritative storage

Use a separate SQLite operations database in WAL mode for transactional state.
The existing racing database is a data source, not workflow authority. Keep the
storage interface narrow enough to permit a later PostgreSQL implementation.

Use content-addressed immutable artifact storage for:

- Raw race programmes, cards, form guides, and source responses.
- Every odds capture.
- Normalized evidence packages.
- Derivation outputs and reports.
- Deferred and on-demand forecasts.
- Official result observations.
- Training examples.
- Model bundles, scorecards, and promotion records.

Operations rows reference artifact checksums. Reports are projections and can be
regenerated.

## Identity model

### Race identity

Assign an immutable internal `RaceId` at discovery. Source race IDs, URLs,
filenames, and venue/date/race-number keys are aliases. Downstream code may not
derive a new race identity from a filename.

### Dog identity

Assign an immutable internal `DogId`. Prefer authoritative registration IDs and
source aliases. Name-only observations may receive a provisional identity when
unambiguous; ambiguous candidates are preserved in identity quarantine rather
than merged.

Identity decisions are append-only and versioned. Corrections supersede earlier
decisions. No source observation is deleted.

### Dog Run identity and embedded history

Under the domain invariant that one dog races at most once per local racing
date, identify a Dog Run by canonical DogId and local racing date.

Multiple sources may observe the same Dog Run. A complete official race entry is
authoritative. An embedded form-guide row may create a provisional Dog Run and
support dog-level features, but it cannot infer the complete field or finishing
order. When authoritative evidence arrives it supersedes, rather than
duplicates, the provisional record.

Conflicting observations remain attached to the same candidate fact for
reconciliation. Ambiguous matches are quarantined. Feature eligibility is
tiered:

- Authoritative identity: all eligible history features.
- High-confidence provisional identity: same-source dog-level history with
  provenance and coverage indicators.
- Ambiguous identity: observation preserved, identity-dependent features
  excluded.

## Racing-day and race lifecycles

The official programme's local racing date defines the Racing Day. Venue
timezone and UTC instants are retained separately; UTC truncation must not
derive the racing date.

The per-race lifecycle is:

```text
discovered
→ card_collected
→ collecting_odds
→ evidence_sealed
→ awaiting_day_close
→ prediction_pending
→ prediction_committed | prediction_quarantined
→ result_pending
→ result_collected | result_quarantined
→ training_example_ready | evaluation_ineligible
```

Transitions are durable, idempotent, and forward-only. Stable operation IDs
prevent duplicated work. Corrections create superseding records and never move
completed state backward.

A Racing Day closes when every expected race reaches a terminal lifecycle state
or a versioned hard-cutoff policy applies. Unresolved races enter quarantine
with explicit reasons.

The prediction batch completes when every eligible race commits an immutable
prediction or enters prediction quarantine. Only then may result collection
begin, and only committed predictions may receive labels. Result collection may
retry with bounded backoff; unresolved labels enter result quarantine at the
deadline.

## Collection and sealing

### Expected inventory

Use an independent programme inventory to establish expected venues and races.
Form-guide downloads populate those expected RaceIds. Reconciliation compares
expected and collected populations so an empty or partial scrape cannot appear
successful.

### Adaptive odds cadence

Initial policy:

- More than three hours to jump: every 30 minutes.
- One to three hours: every 10 minutes.
- Ten to 60 minutes: every five minutes.
- Final ten minutes: every minute.
- Stop when jump is confirmed.
- Back off retries without replacing the last valid observation.

Store every attempt and successful capture append-only with source, timestamps,
runner mapping, and checksum.

### Feature freeze

Prefer the latest fully validated odds snapshot strictly before authoritative
actual jump time. If actual jump cannot be proven, use the latest valid snapshot
before scheduled jump minus a conservative policy buffer. Quarantine when no
cutoff can be established or runner/box identity is ambiguous.

### Sealed Race Evidence

Each package retains permanently:

1. Original source bytes and source provenance.
2. Every odds observation and attempt.
3. Versioned normalized race, runner, lifecycle, and odds evidence.
4. Freeze timestamp and its authority.
5. Schema, normalization, and content checksums.

Old packages are never rewritten. New schema or feature versions produce new
derived views referencing the original evidence.

Field-specific authority replaces latest-write-wins. Critical conflicts in
identity, runner set, jump time, or result order block sealing or evaluation.

## Prediction products

### Deferred Snapshot Prediction

Computed once after the Racing Day closes, but before any result access, using
only Sealed Race Evidence and the model pinned when that Racing Day opened. It is
the authoritative forecast for evaluation.

### On-demand Forecast

A refreshable forecast computed before jump from current evidence for immediate
use. It uses the same forecast contract and pinned champion rules but has a
separate identity and store. It never satisfies the nightly barrier and never
enters champion evaluation.

### Mandatory provenance

Every prediction must contain:

- `champion_model_id`
- `artifact_checksum`
- `trained_through`
- `promotion_approved_at`
- `promotion_effective_from_racing_day`
- `promotion_record_id`
- `prediction_computed_at`
- `evidence_frozen_at`

Missing values fail prediction; production must not emit `unknown` placeholders.

## Feature contract

One deterministic, versioned derivation transforms Sealed Race Evidence into a
FeatureMatrix, FeatureContract, and DerivationReport. Training, deferred
prediction, challenger scoring, and replay call the same transformation.

It may not query mutable evidence databases, scrape sources, or silently repair
schemas. Contract inputs are classified as:

- Identity-critical: absence or ambiguity quarantines the race.
- Forecast-required: insufficient coverage quarantines the race.
- Optional: explicit missingness plus bundle-owned trained imputation.
- Inapplicable: deliberately absent and distinct from unknown.

No runtime zero filling or invented categorical defaults are allowed unless the
bundle contract declares and trains that exact behaviour.

## Model bundle and loader

One durable champion pointer resolves exactly one immutable bundle containing:

- Ordered-finish model or fixed ensemble.
- Feature derivation and forecast-contract versions.
- Feature schema and missingness policy.
- Training configuration and dependency manifest.
- Training-example identities and training cutoff.
- Calibration and evaluation metrics.
- Bundle and component checksums.

The loader must not select production models from environment precedence,
registry leaderboard, filesystem recency, or test/mock fallbacks. Legacy models
require explicit conversion. A legacy-origin incumbent may serve temporarily
with honest missing provenance, but cannot be permanent promotion evidence.

The forecast is authoritative. SP and GPT post-processing may not mutate ranks
or probabilities. Market evidence can annotate disagreement or feed a separately
versioned Wagering Strategy. Market-aware forecasting requires a trained and
evaluated challenger.

Prediction fails closed when the pinned bundle cannot score. Deferred failures
enter quarantine; on-demand failures return unavailable. V3, Unified, heuristic,
synthetic, and mock fallbacks are prohibited in production.

## Ordered finish forecast

The canonical model emits one coherent distribution over finishing orders. The
initial implementation should use a Plackett-Luce-style sequential distribution
over bundle-produced latent runner strengths. Derive from it:

- Win probability.
- Top-2 and top-3 marginal probability.
- Exacta and trifecta combination probability.
- Most likely orders.
- Runner ranking.

A future implementation may replace the internal algorithm while preserving the
forecast contract. A fixed ensemble is allowed only when its components and
weights are frozen inside one bundle.

## Evaluation and promotion

### Eligible outcomes

Initially score only official, provenance-bearing, unambiguous finishing orders.
Remove pre-seal scratches and renormalize. Quarantine post-seal scratches, dead
heats, abandoned results, order-changing disqualifications, and insufficient
result provenance until correct tied/partial-order scoring exists.

### Model quality

Primary score: mean ordered-finish negative log-likelihood on paired
evaluation-eligible races.

Guardrails and secondary measures:

- Win and top-3 calibration.
- Winner mean reciprocal rank.
- Top-3 containment.
- Exact top-2 and top-3 order accuracy.
- Coverage and abstention.
- Venue, distance, grade, and field-size slices.

Win/place strategy scorecards are initially supported when sealed odds exist.
Exotic hit metrics are forecast-only until reliable pre-jump exotic market
evidence exists. Wagering utility is separate from forecast quality and never
places real bets.

### Dual horizons

The Long-horizon Scorecard controls promotion. Initial policy target: at least
500 paired resolved races with minimum venue coverage. The Short-horizon Monitor
uses approximately the latest 100 resolved races and day-level views to detect
drift and nominate investigation or training. It cannot promote independently.

Promotion requires:

- A predefined practical reduction in long-horizon ordered-finish loss.
- Paired race-level bootstrap evidence supporting superiority.
- No failed calibration, coverage, or important-slice guardrail.
- No material short-horizon reversal.
- Minimum sample and coverage gates.
- Incumbent retention when evidence is inconclusive.

Short-horizon degradation across every model suggests data/domain drift. A
champion-only decline with stable challengers supports a model-change diagnosis.

Automatic promotion is permitted only for an already registered immutable
challenger. Training/tuning and promotion cannot occur in one run. Promotion
atomically changes the champion pointer for the next Racing Day, records the
complete evidence and rollback target, and preserves the prior champion.

## Training corpus and cadence

A canonical Training Example is immutable Sealed Race Evidence joined to one
official result for an evaluation-eligible race.

Audited legacy evidence may bootstrap challenger training, but cannot provide
authoritative promotion evaluation. Forward-sealed examples gradually replace
legacy dependence.

Score champion and challengers nightly. Retrain only after sufficient new
examples, on a slower schedule such as weekly, or after sustained drift. Tuning
uses expanding-window temporal validation and creates new immutable bundles.
The Race Collection Service emits requests but never executes training.

## Operations and reliability

### Reconciliation

A mandatory Racing Day report reconciles:

- Expected versus discovered races.
- Runner and box completeness.
- Odds attempts, successes, cadence gaps, and final valid snapshot.
- Seal checksums.
- Prediction commits and quarantines.
- Result states and provenance.
- Training-example joins.
- Champion/challenger coverage.
- Retries, supersessions, and failures.
- Active code, config, schema, policy, and model-bundle versions.

Unexplained count mismatches keep the day incomplete and block training and
promotion. Successful process exit is not proof of collection success.

### Alerts and repair

Individual quarantine appears in daily reporting. Source-wide outage, day
blocker, checksum failure, post-freeze contamination, result-before-prediction
attempt, or champion failure alerts immediately. Pause only affected downstream
phases while safe collection continues.

Administrative mutations require a shared command surface, actor, reason,
operation ID, and audit record. Direct database repair is prohibited.

### Release configuration

One typed versioned configuration defines paths, sources, schedules, and policy
versions. Secrets are externally referenced. Each immutable release manifest
binds code commit, configuration checksum, DB schema, artifact contract, and
supported bundle versions.

Merges build candidate releases. Deployment occurs at a Racing Day boundary,
rehearses migrations, performs health checks, atomically updates a stable release
pointer, and preserves rollback. Emergency deployment requires an audited
override. systemd units contain no dated worktree paths.

### Backup

Transactionally back up operations state after reconciliation and replicate
immutable artifacts by checksum to separate storage. Scheduled isolated restore
drills—not backup command success—prove recoverability.

## Migration

Implementation order:

1. Operations database, identities, lifecycle, and artifact interfaces.
2. Expected inventory, collection adapters, adaptive odds scheduling, sealing,
   and reconciliation.
3. Deferred prediction/result barriers using a converted legacy incumbent.
4. Canonical model bundle and loader; canonical endpoint adapters.
5. Training-example corpus and Dog Run migration.
6. Ordered-finish challenger and evaluation scorecards.
7. Training requests, long/short horizon policy, and gated promotion.
8. Operator/release/backup hardening and legacy retirement.

Cut over collection after two consecutive complete Racing Days with immediate
rollback and the old service disabled but preserved. Require a further fourteen
consecutive successful Racing Days before retiring it or enabling automatic
promotion. Critical failures reset or pause the applicable probation gate.

## Acceptance principles

- One race, dog, run, bundle, and workflow state has one authoritative identity.
- Every mutation is idempotent and auditable.
- Every evaluated forecast proves it could not access its result.
- Every probability and rank traces to one immutable bundle and evidence seal.
- Every exclusion is counted and explained.
- Every model change is prospective, reproducible, and reversible.
- Every release and backup can be verified and rolled back.

## Decisions

See `CONTEXT.md` and ADRs 0001 through 0008 in `docs/adr/`.
