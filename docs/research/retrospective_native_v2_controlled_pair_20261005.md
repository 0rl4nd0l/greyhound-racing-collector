# Retrospective controlled pair from native v2 inputs

This is a separate development computation under
`native-v2-derivation-scope.json`, SHA
`f4a0464fdc11b80301fd542bd5ae4865c6a752dd5047fba3a0f54705077c5932`.
The original v3 checksum-reconstruction contract and its 82 missing-record
findings remain unchanged. This path neither fabricates an original record
nor replaces any original forecast.

The population is the exact, ordered 82-member October 1–3 membership, SHA
`782d6af9bcbe1dd84d1c2674de44f93bc6720cd77b3b7e0195bdcccc03dd4342`.
Result quarantine has no role in input selection. All members receive a pair,
an explicit exclusion, or an unattempted disposition after a shared failure.
No official result or performance evaluation enters this operation.

## Frozen numerical contract

The unchanged production parent model SHA is `624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d`;
its manifest SHA is `8537cbc3d843d106a1fe48793ef01197454ef092c0244025fd65685636a42080`.
The sole contrast is its existing strength **1.0 versus 0.5** from identical
features, coefficients, transforms and runtime. The separate original
experimental `residual_half` artifact is never read as the paired arm.

Original native v2 serialization stores the produced Python float directly in
canonical JSON. There is no probability rounding contract. The scorer's
existing 15-decimal residual-adjustment canonicalization remains inside the
unchanged scorer. Admission therefore requires **exact canonical equality**
of the complete reconstructed production rows—rank, box, dog name, identity,
native runner ID and probability—and their original prediction-output hash.
It separately checks the sealed comparison production rows. No epsilon,
rounding adjustment, alpha search or tolerance fitting is allowed. A one-ULP
difference rejects that race explicitly.

## Time and provenance

The original internal scoring timestamp is unavailable and remains `null`.
The recorded `comparison/production.json.completed_at` is authenticated and
used only as a **historical validation anchor**, the same convention used by
the existing native comparison verifier. It must remain after admission and
all inputs, no later than the original durable completion, and before the
native two-minute cutoff. It is never moved to make a replay pass.

The unchanged original producer may construct an ephemeral scoring record
internally. The worker discards its key, checksum and score timestamp. The new
artifact carries the actual derivation invocation time and labels its anchor
`RECORDED_PRODUCTION_COMPLETION_NOT_ORIGINAL_SCORE_TIME`. Its status is
`RETROSPECTIVE_COMPUTATION_NOT_ORIGINAL_PREJUMP_OR_SCIENTIFIC`.

`load_and_derive` authenticates the original admission, completion and plan,
then calls the existing descriptor-bound native bundle/input/protocol
verifier. It checks retained history seal metadata and exact DB-byte hashes;
it does not query the DB. The native verifier authenticates retained feature,
history, capture, receipt, model and parent-retention bindings. All identities
are joined through the native request roster; box/name alone cannot supply a
missing native identity.

The exact retained generator location selects one of ten independently pinned
original producing packages. The loader verifies that package's plan, complete
source identity map and every source-file hash, original archives and Python
binary. These checks are cached only inside this invocation; production and
source must remain mounted read-only throughout. Each race is replayed in a
fresh subprocess importing that original package, preserving both historical
feature-generator versions. The worker verifies the actual Python identity,
distribution version/RECORD census and retained environment lock; imported
original source files must match the authenticated map. A current source
checkout is never silently substituted for the original producer.

## Execution and accounting

`scripts/derive_retrospective_native_v2_pairs.py` is default off. Root must
issue a new `retrospective_native_v2_execution_authority_v1` with:

- Current issued/expiry timestamps, scope reference, fixed membership and
  original package-bindings reference.
- The reviewed source commit and exact five implementation-file hashes.
- The fixed `EXACT_NATIVE_V2_CANONICAL_FULL_ROWS_NO_ROUNDING_OR_TOLERANCE`
  serialization contract.
- Finite member/read/byte/wall/worker bounds and a new exclusive output path.

The metadata-measured proposal records application evidence reads plus native
verifier/producer passes. Interpreter imports, compiled-library loading and
distribution census reads are explicitly outside that evidence counter; each
worker and the complete execution have finite wall limits. Source-package
validation is charged once per exact package. No collection or result-request
allowance is used. Private outputs are capped at 256 KiB per pair and 1 MiB
per control file; 82 pairs plus claim, inventory and status fit the fixed
24 MiB total envelope.

Launch with the pinned Python, kernel networking denied and all source and
production paths read-only; mount only the fresh output parent writable:

```sh
python -B scripts/derive_retrospective_native_v2_pairs.py \
  --execute --authority /absolute/new-authority.json \
  --authority-sha256 EXACT_NEW_AUTHORITY_SHA256
```

The runner creates an exclusive claim before derivation. A local full-arm
mismatch remains an explicit exclusion while unrelated members continue.
Shared integrity, runtime, read-budget or deadline failures preserve the
claim, already-created private artifacts and remaining member dispositions;
they do not publish a successful inventory. Successful status is written
only after all 82 dispositions and their private artifact references are
durable. No previous claim, forecast, result quarantine, model or service is
modified.

Focused fabricated validation covers native producer/serializer replay, exact
row/ID matching, one-ULP rejection, anchor timing, parent identity, shared
inputs, runtime changes, the explicit source-worker interface, full cohort
accounting, expiry, budget exhaustion and immutable failed/consumed claims.
Passing these tests is implementation evidence; root's separately reviewed
real execution is still required to establish actual cohort qualification.
