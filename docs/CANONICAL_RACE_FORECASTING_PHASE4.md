# Canonical Race Forecasting — Phase 4 handoff

Phase 4 adds one transactional model authority to the Race Collection Service
operations database. Migration `0011_canonical_model_bundle.sql` stores the
immutable bundle identity, one normalized row per required component, and the
immutable serving assignments, and the single bootstrap champion pointer.
Training and bundle facts do not contain promotion fields: a Phase 5 challenger
can register without an assignment and cannot serve. An assignment separately
carries complete approval/effective-day/record provenance and binds one exact
bundle checksum; Phase 4 only creates the explicit bootstrap authority, while
Phase 6 owns general promotion. Canonical-origin bundles are independent of
the Phase 3 legacy registry. A legacy-origin conversion instead carries an
explicit relational foreign key to its exact Phase 3 descriptor, and its model
checksum and size must match. Composite foreign keys authenticate the
pointer/bundle checksum relation. The tables are append-only; Phase 4 has no
general promotion or pointer replacement policy.

`race_collection.model_bundle` is the only registration and loading boundary.
The loader reads no model-selection environment variables, registries, ranked
metadata, or filesystem directory listings. For current on-demand use and
bootstrap day pinning it resolves the singleton pointer. For deferred prediction
and replay it authenticates the exact bundle and model checksum already carried
by Phase 3's immutable Racing Day request plus its unique immutable serving
assignment and never consults the live pointer.
Both entry points use the same verifier, which checks the exact manifest, every
component checksum and size, both contract
versions, feature schema, missingness policy, dependency versions, training
corpus identity, and Python runtime before deserializing the model. Any failure
is terminal for that request.

The Phase 3 incumbent remains honestly classified as a raw
`sklearn.calibration.CalibratedClassifierCV`. Its binary was not added to Git.
Conversion is quarantined because the observed source cannot prove the required
training cutoff, promotion record and dates, feature/missingness contract,
training corpus, dependency/runtime manifest, or calibration/evaluation
metadata. No placeholder is emitted. Consequently a deployment containing
only that legacy evidence has no canonical champion: deferred computation
enters Phase 3's per-race quarantine and on-demand requests return unavailable.

`race_collection.features.derive_features` is a pure byte-in/values-out
transformation over Phase 2's real normalized envelope
`{schema_version, normalization_version, race_id, fields, field_provenance,
freeze:{at,authority,odds_checksum}}`. Runner feature rows are read only from
the closed `runner_set`, `runner_identity`, and per-field mappings inside the
normalized `fields` object; a parallel top-level runner/freeze envelope is
rejected. It binds the sealed evidence, schema, and missingness checksums;
requires exact evidence and normalization versions; rejects unknown fields,
ambiguous or duplicated identities, missing required values, and undeclared
defaults; and distinguishes explicit optional imputation from inapplicability.
Training, deferred prediction, challenger scoring, and replay can call the same
function without a database or scraper.

`race_collection.forecast_service` owns scoring, normalization, deterministic
ranking, mandatory provenance, deferred integration, and on-demand failure
translation. The evidence freeze comes from the checksummed evidence document;
computation time comes from the injected service clock. HTTP clients cannot
assert either timestamp. The canonical Flask route is
`POST /api/canonical/forecast`. Every existing prediction POST route carrying a
sealed `evidence_checksum` is intercepted as a thin adapter to the identical
service result. SP and GPT inputs are ignored and cannot alter probabilities or
rank. Filename-based legacy compatibility remains present but is outside the
canonical path.

Phase 5 remains entirely deferred: this phase does not audit a historical
corpus, train an ordered-finish challenger, evaluate challengers, or define a
wagering strategy. The converted incumbent is not represented as more capable
or better-proven than the evidence supports.

## Fresh Phase 5 prompt

> Work only on Phase 5 from the committed Phase 4 HEAD and the authoritative
> programme specification, task card, ADRs 0001–0008, and Phase 1–4 handoffs.
> Audit and deduplicate the Legacy Training Corpus, build canonical immutable
> forward Training Examples, and implement/train/evaluate the initial coherent
> Plackett–Luce-style ordered-finish challenger behind the Phase 4 bundle,
> feature, and forecast contracts. Derive win/top-N/exacta/trifecta/ranking from
> the one ordered-finish distribution and score only evaluation-eligible,
> unambiguous outcomes. Keep wagering report-only and separate. Do not implement
> Phase 6 promotion policy or Phase 7 cutover. Validate, review, commit, and stop
> with a fresh Phase 6 prompt.
