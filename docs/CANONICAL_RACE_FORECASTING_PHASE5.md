# Canonical Race Forecasting — Phase 5 handoff

## Delivered scope

Phase 5 adds an append-only canonical forward Training Example relation in
migration `0012_phase5_training_corpus.sql`. One row authenticates exactly one
Phase 3 evaluation-eligible join, its prediction-bound exact normalized Sealed Race Evidence,
one collected official-result artifact, the Racing Day, the Phase 4 feature
matrix checksum, and the immutable joined artifact. Both Python and SQL reject
identity or relation forgery. Replay requires the same operation intent and
durable row. Result publication must be after the feature freeze and before the
join. Phase 3's positive integer official box order is mapped through a
one-to-one authoritative `fields.box` mapping to the sealed runner identities;
missing, duplicate, ambiguous, or extra boxes fail closed. The evidence artifact is scanned for forbidden result-derived feature
names and then passed through the same pure `derive_features` function used by
serving.

`audit_legacy_corpus` provides a deterministic, provenance-preserving audit of
historical Run Observations. It keys Dog Runs by DogId and local racing date,
deduplicates exact observations, selects authoritative evidence over
provisional evidence, records superseded observations, and quarantines
equal-authority value conflicts. Its manifest says `promotion_grade: false`:
legacy data may bootstrap a later training run but is never represented as
forward-sealed promotion evidence. No production corpus was available or
copied, so tests use small realistic synthetic records and demonstrate
capability, not production-corpus readiness.

`race_collection.ordered_finish` is the versioned
`plackett-luce-ordered-finish-v1` contract. It converts bundle-produced finite
latent runner strengths into one numerically stable sequential distribution.
Win, top-2, top-3, exacta, trifecta, most-likely orders, and deterministic
runner ranking are all sums or ordering of that same enumerated distribution.
The exact-enumeration implementation conservatively supports normal greyhound
fields up to eight runners and
fails closed beyond that bound rather than approximating silently. The
canonical service uses the ordered contract for new challengers while retaining
the exact Phase 4 `predict_proba` response path for the explicitly versioned
legacy contract. Dispatch is by exact contract identity: ordered bundles must
provide `latent_strengths`, and binary runner-win bundles cannot be coerced into
the ordered distribution.

`race_collection.training` fits a deterministic linear runner-strength model
by full-batch gradient ascent on Plackett–Luce ordered-finish likelihood. It
records its fixed seed, algorithm, cutoff derived from the latest bound
example, exact example/artifact/evidence/result/feature identities, corpus
identity, dependency/runtime requirements, native normalization calibration
description, and Racing-Day-grouped expanding-window temporal validation. The
complete nine-component bundle and manifest are content-addressed. Registration
uses the Phase 4 authority but deliberately creates no serving assignment,
champion pointer, or Racing Day assignment. Training therefore cannot affect
committed forecasts or future day pins.

Only official, provenance-bearing total orders are eligible. Post-seal
scratches, dead heats, abandonment, order-changing disqualifications,
incomplete provenance, runner-set disagreements, result/evidence temporal
violations, and result-derived features fail closed. Pre-seal scratches are
already absent from the sealed runner set and the remaining field is normalized
by the one ordered distribution. Wagering is a pure, separately versioned,
report-only projection; it does not place bets or enter forecast-quality
metadata.

## Production-readiness boundary

This phase proves the complete corpus/model capability with synthetic fixtures.
It does not claim that a production Legacy Training Corpus has been located,
audited, or accepted, and it does not commit a synthetic trained challenger as
a production candidate. Running the training workflow on production evidence
requires owner-provided immutable artifacts inside an authorized future run.
No network, external service, runtime database, model registry, champion,
release, service, timer, or canonical source was accessed or changed.

Phase 6 exclusively owns paired champion/challenger scorecards, promotion
thresholds and gates, drift policy, serving assignments, champion changes, and
next-day promotion. Phase 7 services and cutover remain untouched.

## Fresh Phase 6 prompt

> Work only on Phase 6 from the committed Phase 5 HEAD and the authoritative
> specification, task card, ADRs 0001–0008, and Phase 1–5 handoffs. Implement
> paired evaluation of already registered immutable champion and challenger
> bundles using evaluation-eligible forward examples; ordered-finish NLL,
> calibration, rank, containment, exact-order, coverage, and required slice
> metrics; distinct long-horizon promotion and short-horizon drift evidence;
> versioned promotion gates, bootstrap evidence, nomination, atomic next-Racing-
> Day assignment, complete provenance, incumbent retention, and rollback. Keep
> training separate from promotion and never train, tune, and promote in one
> run. Do not begin Phase 7 services, cutover, probation, or legacy retirement.
> Validate, independently review, commit coherent code/docs/tests, and stop with
> a fresh Phase 7 prompt.
