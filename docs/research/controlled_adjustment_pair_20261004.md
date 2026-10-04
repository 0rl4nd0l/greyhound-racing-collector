# Controlled production adjustment derivation

This default-off module implements the first contrast in the October 3 controlled
comparison design: production's frozen base at strength 1.0 versus **that same
base** at 0.5. It is a retrospective development derivation, not a new original
pre-jump forecast, prospective admission or performance finding. No registry,
model, collector, sealed forecast, population membership or result queue changes.

The public seam is `src.predictor.controlled_adjustment_pair.derive_controlled_pair`.
It accepts one loaded frozen production parent, one original native shadow record,
one authenticated common binding and a pinned development plan. Execution is OFF
unless the caller explicitly supplies `execute=True`; that flag is not authority.
There is deliberately no service, provider, result reader, dataset selector,
training routine, CLI or automatic persistence hook.

The parent is `market_form_residual_v1`, model SHA256
`624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d`,
manifest SHA256 `8537cbc3d843d106a1fe48793ef01197454ef092c0244025fd65685636a42080`.
The module reuses `market_form_residual.score_race` unchanged, source SHA256
`50039cbc46f48d2f2d0dda8973e75dc73055872ff74b082821535060cc36b7f6`.
That scorer verifies its effective-state identity and computes both arms from
one centered, capped adjustment using the identical 16 features, missingness,
preprocessing, coefficients, market baseline and normalization. No hand-copied
prediction algorithm is introduced. Existing `residual_half` uses a different
history/feature/fitting recipe and is not this comparator.

`plan` has exactly these fields: schema_version `controlled_adjustment_plan_v1`,
parent_family, model_sha256, manifest_sha256, effective_state_sha256, race_id,
common_binding_sha256, strengths `[1.0,0.5]`, runtime_sha256 and scorer_source_sha256.
`binding` has exactly: schema_version `controlled_adjustment_common_binding_v1`,
race_id, original_record_sha256, feature_source_sha256, odds_source_sha256,
history_snapshot_sha256, retained_manifest_sha256, original_sealed_at and
original_source_commit. Canonical hashes are sorted compact JSON with no NaN and
one trailing newline. A record's existing native checksum is retained as well.

The caller must authenticate this plan and all external binding references from
an authorized cohort before passing them here. **A well-formed supplied hash is
not proof that history, a runtime binary or a seal was independently verified.**
The pure module performs no external reads and labels these checks as caller
requirements. It compares runner feature/odds-source bindings and exact original
record bytes, replays through the frozen scorer, and demands exact record equality.
The same common input digest and parent state identify both output candidates.

Original quote observations must be the same aware instant for all runners and
120–600 seconds before jump. Original durable seal metadata must be at least
120 seconds before jump and no earlier than the original score. `derived_at` is
the actual timestamp supplied by the root-owned caller and may not predate that
seal. The original score timestamp is used solely to replay the original
record's historical contract; the output separately records actual derivation
time and explicitly denies new pre-jump/scientific membership. There is no
fabricated now, relaxed native timestamp rule or late-forecast credit.

## Next actual execution step

Root first freezes exact reservation-cleared retrospective cohort membership and
a metadata readiness manifest. Reuse the October 3 technical readiness evidence
for 15 live07 bundles only as a lead: it did not decode or rehash opaque feature,
history and forecast content, independently resolve every reserve identity, or
create new paired forecasts. It is not blanket execution qualification.

A root-owned, separately reviewed loader must verify complete native input roles,
original source/runtime/model pins, independent verifier evidence and actual
history/feature/record hashes before any authorized protected forecast decoding.
A missing reconstructible native shadow record is an explicit exclusion; do not
invent its source timestamps or call a newly reconstructed record an original
seal. Run the exact frozen module in a network-denied, source-read-only process,
with actual `derived_at`, and persist each new development artifact exclusively
in a separate output root. Keep every eligibility failure and denominator; do not
select by outcome or overwrite original forecasts. No result access or analysis
is required to produce this pair. Any performance analysis remains a separately
bound root task with its own fixed endpoint/metrics/missingness decisions.

## Validation

Tests use fabricated race identities, features, odds and timestamps with the
unchanged frozen production parent. They check default-off behavior, the exact
2:1 log-odds adjustment relationship, normalization/cap, shared provenance,
deterministic replay, changed parent/strength/input/hash rejection, outcome-field
rejection, original seal and quote timing, and mutation-free original records.
No retained forecast was replayed, no outcome read and no model fitted.
