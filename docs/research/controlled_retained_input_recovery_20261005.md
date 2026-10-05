# Retained inputs for the controlled adjustment comparison

The fixed October 1–3 development cohort remains **82 races**. Result
quarantine is not an input-selection rule. No pair, new forecast, result join,
performance metric, training run or live change was produced by this work.

## What the actual native packets retained

The earlier prerequisite note described the old race-first producer's embedded
artifact. The 82 actual native v2 packets retain output rows, feature/history
evidence, model files and native identity/timing seals. They do **not** retain
the original in-memory `market_form_residual_shadow_record_v3`, its record key
or checksum, or the original producer `score_timestamp`.

The machine-only inventory verifies each manifest and each inspected member
hash/size, then inspects non-model JSON members and bounded retained ZIP JSON.
Model test-fixture records do not count as original race records. It does not
query history databases or open result evidence. The fixed membership SHA is
`782d6af9bcbe1dd84d1c2674de44f93bc6720cd77b3b7e0195bdcccc03dd4342`.
The initial complete inventory is recorded under the research continuation
directory as `controlled-native-commitment-audit.json` (SHA
`0dcb77206094547b0ac920de04270110032b0464cca1e1c6369e6fa21b8761ba`).
It found zero original records and zero original commitment/time fields.

Therefore **0/82 meets the existing original-record prerequisite**; all 82
have `ORIGINAL_NATIVE_RECORD_COMMITMENT_AND_SCORE_TIMESTAMP_MISSING`. This
does not invalidate their original sealed forecasts or the completed private
baseline. It prevents claiming a checksum-exact original shadow record has
been reconstructed. The inventory is not a complete numerical input audit.

## Implementation and evidence boundary

`src/predictor/controlled_retained_inputs.py` adds three distinct operations:

- `audit_membership` preserves all 82 members and reports commitment presence
  and explicit failures. It is bounded to 512 MiB, 6,000 read operations and
  120 seconds; malformed or changed evidence aborts the inventory instead of
  publishing partial success. ZIP member reads count separately.
- `reconstruct_committed_record` is default off. Given an authenticated
  original producer artifact and exact original feature rows, it rebuilds the
  original computational record, requiring the production parent, exact
  record key/checksum, parity binding and deterministic native replay. It
  copies an existing original score timestamp and never invents one. Its
  native-producer fixture proves this seam; it is not evidence that any of
  the actual 82 can use it.
- `load_committed_original` is default off. A caller must supply an original
  artifact declared inside the exact native sealed bundle and an independently
  pinned execution binding. It calls the existing native bundle/input/protocol
  verifier, checks admission/completion identity, exact retained generator
  package location, original source identity map, current and original replay
  files, actual Python and distribution version/RECORD census, original model
  and artifact replay, and original pre-jump timing. Missing original evidence
  remains a failure. There is no fallback to current files or publication time.

The source binding schema is `controlled_original_execution_binding_v1` with
`bundle_manifest`, `original_package_plan`, `original_source_identity`,
`original_runtime_identity` (path/SHA references), `original_source_commit`,
`replay_source_hashes` and `replay_paths`. The latter maps the native producer's
eight inputs: form_csv, sidecar, feature_rows, feature_manifest,
implementation_manifest, capture, model, manifest. Required replay sources
are the producer's feature-generator set plus the producer, scorer, parity,
venue and identity helpers. The binding itself must be pinned by root's new
execution receipt. The old `on_demand_isolated_runtime` placeholder is not a
commit and is never converted into one.

Independent metadata work recovered ten genuine producing package commits,
source maps, runtime receipts and archive bindings. That removes a source
identification gap; it cannot manufacture the missing original record
commitment. Computational runtime verification is narrower than reproduction
of the old browser/service environment and is labelled accordingly.

The CLI `scripts/audit_controlled_retained_inputs.py` is default off and only
exposes the inventory. Its explicit audit mode requires the new task receipt,
its hash, exact fixed membership, and a new output inside the authorized
research directory. It emits counts/categories only. Use the pinned Python
under `bwrap --unshare-net --ro-bind / /`, mounting only the chosen output
directory writable. The source and retained production files remain read-only.
There is deliberately no real-cohort reconstruction or pair command.

## Separate decision needed for native v2

A useful alternative is a **new retrospective derivation from native v2
retained inputs**, preserving all 82 input qualifications. It would verify
the native seals, source/runtime/history/model and runner identities, replay
the full-strength calculation against the sealed original full-arm outputs,
and compute half strength from that same production parent and common inputs.
It would record the actual new derivation time and the original available
capture/seal times, explicitly leaving the unavailable original score time
unknown. It must not claim an original v3 checksum match, reuse the separately
frozen experimental half model, or acquire results. Root must approve and
review that different contract before implementation or execution. The
existing pure controlled-pair module is unchanged.
