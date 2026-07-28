# Forward-sealed corpus capture

`race_collection.forward_sealed_corpus` is the prospective collection boundary
for authentic Phase 7 training evidence. It does not fetch a race, schedule a
job, write a database, train a model, register a bundle, predict, promote, or
activate anything.

The current programme, pre-jump refresh, `EvidenceSealer`, odds capture, and
official-result collectors remain the source adapters and scheduling
authority. The forward collector consumes one immutable adapter output at a
time. The bounded CLI reuses the shadow-autopilot exclusive lock shape at a
corpus-local path; it does not install or start a service or timer.

## Evidence pillars

### A. Pre-jump source

`forward-source-capture-v1` binds:

- the canonical source URL and source-native race and runner identities;
- non-empty meeting and race metadata, racing date, and scheduled jump;
- immutable raw source bytes and `raw_source_checksum`;
- the existing canonical `race-evidence-v1` bytes produced by
  `EvidenceSealer`;
- source observation and feature-freeze timestamps before jump;
- `identity_authority: source-native` and `reconstructed: false`.

The sealed evidence must contain runner-set, identity, and runner-feature
provenance bound to the preserved raw source checksum. Raw bytes and normalized
evidence remain separate content-addressed objects. On first publication, the
collector also checks its own timezone-aware clock is at or after feature
freeze and still before jump. Exact receipt replay remains valid after jump,
but no new receipt can first appear late.

### B. Pre-jump features

The collector calls the production `derive_features` function with immutable
schema, missingness-policy, and sealed-evidence bytes. The feature schema must
declare `sealed-race-features-v1`; the missingness policy must exactly cover
every optional feature with finite values. The resulting canonical matrix
contains source-native runner IDs, ordered columns, and deterministic rows.

No result argument exists on the pre-jump API. Result-derived keys anywhere in
the source metadata or sealed evidence are rejected before feature derivation.

### C. Official result

`official-forward-result-v1` preserves:

- immutable raw official-result bytes and checksum;
- the canonical result URL, source-native race identity, aligned runner IDs and
  names, and complete official order;
- a source-declared publication timestamp and collector observation timestamp,
  both after jump;
- `identity_authority: source-native` and `reconstructed: false`.

The current TheDogs `SourceResult` contract exposes neither the exact response
bytes nor an authentic source publication timestamp. It therefore cannot be
wired to this closure path as-is. A later, separately authorized adapter change
must preserve those response bytes before normalization; this pipeline will not
reconstruct them from parsed records.

When an adapter does preserve exact result bytes but its source still does not
expose a publication timestamp, the iteration must use
`publication_timestamp_status: not-exposed-by-source` with
`result_published_at: null`. The collector preserves the raw bytes and
`result_observed_at`, returns `BLOCKED_RESULT_PUBLICATION_TIMESTAMP`, and does
not create normalized result, training-example, closure, or package evidence.
No timestamp is inferred from HTTP, page text, jump time, or collector time.

## Append-only state machine

```text
EMPTY
  -> PREJUMP_CAPTURED
      -> BLOCKED_RESULT_PUBLICATION_TIMESTAMP
      -> RESULT_CAPTURED
          -> CLOSED
```

An exact retry is an idempotent no-op. A different retry at a published stage,
runner drift, source-native identity mismatch, late capture, missing bytes,
unknown or naive timestamp, result-byte hash drift, or conflicting duplicate
closure fails closed. A blocked timestamp observation may close later only
when the same immutable result bytes receive an authentic source-declared
publication timestamp.

Objects use the repository `LocalArtifactStore`. Per-race stage receipts use
same-filesystem temporary files, `fsync`, and an exclusive hard-link publish;
they are never overwritten. A crash after object publication but before a
receipt leaves only harmless content-addressed objects, and an exact retry
completes the missing receipt.

## Closed package

Only a closed race has immutable `historical-training-example-v1` bytes with:

- `origin: forward-sealed-corpus-v1`;
- `forward_sealed: true`;
- raw source/result, normalized source/result, feature matrix, and source
  capture checksums;
- the complete prospective and post-jump timestamp chain.

`build_package()` sorts races by normalized race identity and emits the existing
canonical `historical-source-package-v1` /
`historical-source-manifest-v1` envelope. Input and artifact-map reordering
therefore produces byte-identical manifests. `race_collection.source_admission`
independently re-hashes every declared object, re-derives every feature matrix,
re-validates identities and temporal ordering, and admits a complete forward
package as `TRAINING_ADMISSIBLE`. Admission keeps
`promotion_evidence_eligible` and `production_readiness` false.

## One-iteration CLI

```bash
python3 scripts/collect_forward_sealed_corpus.py \
  --root /safe/append-only/corpus \
  --iteration /safe/adapter-iteration.json

python3 scripts/collect_forward_sealed_corpus.py \
  --root /safe/append-only/corpus \
  --status
```

The iteration object has exactly one `action`: `prejump` or `result`. Raw and
normalized bytes are passed as file paths generated by the existing adapters.
A successful result iteration closes the race and rebuilds the deterministic
package. The missing-publication decision exits non-zero after safely retaining
the observation.

This repository change is collection support only. Live invocation, service or
timer wiring, database writes, training, model lifecycle actions, deployment,
and activation each require separate post-merge authorization.
