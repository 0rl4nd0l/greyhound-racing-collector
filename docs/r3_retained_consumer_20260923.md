# Retained input consumer connection — 23 September 2026

## Finding and repair

At base `89d22067a737222a053e73e02726ba352eec08e3`, the existing
`predict_race_now.run_prediction` authenticates a collector receipt, then always
calls `seal_history_database(source=args.db)` and computes features at prediction
time. Nothing selects the earlier `ScheduledInputRetention` bundle. A successful
retention replay therefore did not prove that R3 would consume that history.

The synthetic baseline reproduction supplied a completed retained bundle,
deleted its original invented-history database, and executed the base predictor:
`HISTORY_DATABASE_UNAVAILABLE`, with zero fixture-scorer calls. The repaired
entrypoint produces a verified prediction under those same conditions.

The opt-in connection adds two explicit CLI arguments:

```
--retained-input-bundle /absolute/collector-evidence/claim/bundle
--retained-input-manifest-sha256 <exact manifest digest>
```

They require `--odds-source receipt`. Without either argument, existing behavior
is preserved. Supplying only one fails closed. No source acquisition, scheduler,
model, feature definition, research population, or frozen configuration changed.

The existing deployment generator accepts `--retained-input-bindings PATH`.
Its JSON contains a finite exact-race map:

```json
{
  "Race 1 - WAR - 2030-01-01": {
    "path": "/absolute/collector-evidence/claim/bundle",
    "manifest_sha256": "<64 lowercase hex characters>"
  }
}
```

This is an illustrative invented identity, not an activation configuration.
The generator embeds the map in the existing generated repository binding and
requires paths below its collector evidence root. This option does not enable
R3 or retention. Bootstrap passes the map to `WorkerConfig`; admission persists
the selected digest in the existing immutable `JobInput`. Old job identities
remain byte-compatible because the new field is omitted when absent. Missing
race entries, changed digests, and removal of the retained mode across restart
are rejected before claim. The retained choice is also present in the sealed
prediction request and checked against the stored job by the existing finalizer.

This is a bounded, explicit per-race connection. It does not discover or admit
future retained races automatically. A future release/configuration decision
must identify the accepted retained manifest before admitting its race; there
must still be time before the retained cutoff. Broader automatic selection is
outside this repair.

## What reaches the scorer

The adapter reads the supplied retained bundle's members once, validates their
manifest hashes, and copies only authenticated bytes into the isolated prediction
bundle. It never reads the `original_path` fields or the canonical history DB.

| Identity | Consumer enforcement |
| --- | --- |
| Race and runners | Exact current index and authenticated collector receipt still required; retained race/jump and receipt must agree; generated race/box/name projection must equal retained projection. |
| Observation and cutoff | Observed ≤ retention start ≤ completion ≤ seal ≤ consumption < retained prediction cutoff < jump; scheduled parent acceptance also precedes consumption/cutoff; completion of scoring must precede retained cutoff. Existing receipt age and pre-jump margins remain. |
| Source material | Retained normalized form, adjacent metadata and odds report must match the authenticated handoff byte-for-byte. Raw form, primary page and page receipt remain hash-bound inside the archived retention evidence. The scheduled producer already authenticated their source references. |
| WIN receipt | Retained exact collector receipt equals the snapshotted protocol member, including collector run and capture identities. A different newer receipt cannot silently replace it. |
| History | Copies retained runner-scoped `history.db`; verifies original source digest, target, cutoff and zero target/future materialization. Retention now preserves the original complete `history_seal.json` needed by the prediction verifier. Older bundles lacking that proof are rejected. |
| Generator | Required production generator files must exist in the source ZIP; every ZIP member equals the deployed source. Duplicate, incomplete and invalid source archives fail. Retained worker source also matches. |
| Environment | Python version and every package version declared in retained `environment_lock` must equal the consumer runtime. |
| Schemas/model/config | Retained feature schema equals the deployed repaired feature schema. Retained frozen model, model manifest and prediction config match the selected identities. The separate prediction configuration schema is still validated/pinned by ordinary R3 model/config admission. These two schemas are not interchangeable. |
| Features and missingness | Actual unchanged feature generator receives only the copied DB/form/metadata. Every one of the frozen 16 values, including nulls, and every race/runner identity must equal the retained canonical feature projection before invoking the scorer. |
| Prediction provenance | Original retention evidence is sealed as `retained_inputs.zip`; original retained history seal is the authenticated cutoff. Independent indexed-bundle verification joins the pinned retained manifest, copied history, forms, receipt, model/config and scored 16-feature projection. |

The immutable receipt's source members remain necessary for ordinary receipt
preflight. If those members are missing or altered, receipt validation can reject
the request. It cannot fall back to fresh mutable data. Removing or changing the
original canonical DB cannot change retained-mode consumption. This work does
not claim that every complete retention bundle is scientifically eligible:
existing feature/scorer validation remains decisive.

## Executed synthetic proof

`tests/test_retained_prediction_consumer.py` exercises the actual scheduled
receipt publisher/authenticator, scheduled retention worker entrypoint, history
sealer, retained feature generator, prediction entrypoint and indexed verifier.
Only the final probability scorer is a fixture. Its composed worker test also
executes `WorkerConfig` → `run_once` → predictor → `finalize_producer_bundle`,
then reopens the store and proves a second attempt is unavailable. That test
substitutes the OS process boundary with an in-process fixture calling the real
CLI parser/entrypoint; existing worker subprocess/lifetime tests remain in the
regression suite.

Cases cover removed original DB; absent evidence; byte changes; deliberately
rebound synthetic manifests with mismatched race, runner, source, receipt, model
and config; empty generator archive; incompatible environment; retained cutoff
before consumption and crossed during scoring; persisted digest across restart;
changed/removed worker binding; and a generated default-off package binding.
All data is invented. No real-race scorer or performance estimate was produced.

Executed with the existing collector `.venv` interpreter:

- Initial affected retention/worker/store subset: **260 passed**.
- Expanded predictor, store, worker, bootstrap, deployment, R3 API, scheduled
  corpus and retention regression set: **581 passed, 1 environment failure**.
  The failure was the existing connected-app startup subprocess lacking
  `flask_compress` in this interpreter; no dependencies were installed.
- Final consumer/binding regression set after the composed worker test:
  **23 passed in 13.86 seconds**.

Machine logs are outside the source tree in
`/home/l4nd0/greyhound-r3-integration-evidence-20260923/`:
`retained_consumer_baseline.log`, `retained_consumer_tests.log`, and the targeted
installed-R3 interpreter startup rerun `retained_consumer_r3_startup.log`.
The integrating handoff records that rerun and final package validation.

## Deployment prerequisite and limits

Both inspected pinned environments use Python 3.11.15, but the collector
interpreter has `requests 2.34.2` / `charset-normalizer 3.5.0`; installed R3 has
`requests 2.32.4` / `charset-normalizer 3.4.9`. The new exact environment check
correctly rejects that cross-environment handoff. Proposal: prepare and validate
a separate common pinned environment for both retained generation and R3 in a
future approved release. The campaign interpreter must remain unchanged.
Same-environment synthetic success is not deployed interoperability evidence.

Before a future approved acceptance: deploy this consumer plus the retention
history-seal producer change; select compatible pinned dependencies; authorize
retention/history access separately; obtain a complete pre-cutoff retained bundle
and fresh exact receipt; generate the finite retained manifest binding; activate
R3/journal only under its separate authorization. The installed service still
needs its ordinary current index, collector protocol and output/store bindings.
No activation, live capture, real-history access, installed binding changes,
service changes or campaign counter changes were performed by this workstream.
