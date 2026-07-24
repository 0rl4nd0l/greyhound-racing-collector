# Canonical Race Collection — Phase 3 handoff

This is an implementation handoff, not workflow authority. The operations
database and immutable artifacts remain authoritative.

## Delivered authority

Migration `0010_deferred_forecasting.sql` owns immutable legacy bundle and
release descriptors, one model/release/policy pin per Racing Day, deferred
prediction commits and quarantines, result attempts, training-example joins,
and on-demand forecasts. `race_collection.forecasting.ForecastingAuthority`
is the single mutation surface. Every call has an authenticated exact operation
intent. Replay is idempotent; reuse of an operation ID with different intent
is rejected.

A day pin must exist before the day closes. Replay-safe `begin_prediction()`
validates the exact sealed expected race, immutable pin, and closed day, then
records the sole legal `awaiting_day_close -> prediction_pending` edge.
Prediction snapshots the persisted seal, pin, bundle, and complete release
descriptor in a read transaction; injected `DeferredPredictor` computation
runs without a SQLite write lock. One immediate transaction then re-reads the
exact snapshot and atomically records either `prediction_committed` or a durable
per-race `prediction_quarantined` outcome. Ordinary load/model/feature/predict
exceptions take the quarantine path; process-control exceptions propagate.
The append-only terminal row persists the original snapshot, including the
`prediction_pending` race mutation token, closed-day instant, and complete
release descriptor. Exact, manual-quarantine, and concurrent replay all
authenticate that snapshot plus the caller's terminal outcome and timestamp.
There is intentionally no Phase 4 feature derivation or champion loader here.

Official result collection opens only for committed predictions and only after
every eligible expected race is prediction-committed or prediction-quarantined.
Failed result attempts are append-only and become terminal at the configured
attempt limit or deadline. Persisted policy `result-retry-v1` fixes a one-second
minimum backoff and enforces immutable limit/deadline, sequential attempts, and
one terminal result at both Python and SQLite boundaries. Only a committed deferred prediction plus a
collected result can create a training-example join. Ambiguous outcomes create
forward-only `evaluation_ineligible` joins. Prediction quarantine cannot have a
result attempt or join. On-demand forecasts have a separate table, do not alter
race lifecycle, and cannot satisfy the barrier or participate in a join.

## Legacy artifact inventory and selection truth

Read-only canonical evidence on 2026-07-23 showed:

- Registry `model_index.json` SHA-256
  `fbd4ec1e66d0dc6b4651893e25b3a53877b7ed51e157163bd492cea26ae496f0`
  marks `V4_ExtraTrees_ExtraTreesClassifier_Calibrated_20260329_212033`
  `is_best=true`.
- Its model is 87,072,784 bytes with SHA-256
  `60866c564253f774f96fd6f1c40368bc2aa0c8e477784a85917a41c30cbfd096`.
- Separate metadata is 2,731 bytes with SHA-256
  `21da78c9ced19005897164d79c284a340df20ea3a5fe2cd236929fe783d4b34f`.
  It lists 49 feature names and calls the file a calibrated ExtraTrees model.
- `best_model.joblib` and `best_metadata.json` resolve through canonical-checkout
  symlinks to those files. `best_scaler.joblib` names the corresponding scaler,
  but that target is absent. Metadata says a scaler path exists; the filesystem
  proves it does not.
- `pinned_override.json` names missing model
  `V4_ExtraTrees_ExtraTreesClassifier_Calibrated_20250910_154052`.
  `MLSystemV4._try_load_latest_model()` never reads this pin. Checked-in loader
  precedence is `V4_MODEL_PATH`, registry best, then filesystem fallback. No
  claim is made about an unobserved live-process environment override.
- The selected file is classified conservatively as a raw registry model. The
  registry loads it as the model and gets feature names from separate metadata.
  Trusted isolated inspection identified an
  `sklearn.calibration.CalibratedClassifierCV` object with `predict_proba`, not
  a dictionary envelope. No feature contract or training manifest accompanies
  it; full-envelope fields are therefore not fabricated.

Only this model, metadata, and index were copied to ignored `.phase3-artifacts/`
for local verification. The binary is not committed. The committed
`model_import/legacy_v4_manifest.json` records the inventory, and
`scripts/import_legacy_v4_bundle.py` deterministically verifies size/checksum
before unique-temporary-file atomic no-clobber publication without deserializing.
The importer holds no-follow source and destination-directory descriptors,
proves root confinement before creating a temporary file, and verifies the
temporary and any publication winner through the authoritative directory fd. Exact
destinations are `.phase3-artifacts/model_registry/models/V4_ExtraTrees_ExtraTreesClassifier_Calibrated_20260329_212033_model.joblib`,
`.phase3-artifacts/model_registry/metadata/V4_ExtraTrees_ExtraTreesClassifier_Calibrated_20260329_212033_metadata.json`,
and `.phase3-artifacts/model_registry/model_index.json`. The scaler is absent;
no feature-contract or training-manifest file was found. `artifacts/models/real_*`
was not used because it has a different envelope.

Isolated loader proof used the exact copied registry files, restricted the
in-memory copied index to the selected entry, rebound that entry to the copied
model path, cleared `V4_MODEL_PATH` and mock flags, and invoked the checked-in
`MLSystemV4._try_load_latest_model()` registry path. Result:
`sklearn.calibration.CalibratedClassifierCV`, `predict_proba=True`,
`source=model_registry`, exact selected model ID, and 49 feature columns. It did
not retrain or create a mock. A raw estimator remains invalid for the dict-only
`V4_MODEL_PATH` branch. Whether an unobserved live process has that variable set
remains unknown.

Repair validation on 2026-07-23 used the project-independent environment with
`PYTHONPATH=.` and `--noconftest -p no:cacheprovider`: focused Phase 3 tests
passed (`11 passed`), importer tests are included in the race-collection run,
and all `tests/race_collection` passed (`220 passed`). The reproducible import
command was `python3 scripts/import_legacy_v4_bundle.py --manifest
model_import/legacy_v4_manifest.json --destination-root .`; it verified all
three exact checksums above.

The reproducible loader command is `MPLCONFIGDIR=.phase3-artifacts/matplotlib
PYTHONPATH=. <project-independent-python> scripts/verify_legacy_v4_loader.py`;
the checked-in script asserts the selected ID and model checksum, raw estimator
type, `predict_proba`, registry source, absence of mock/retrain fallback, and 49
metadata features.

## Phase 4 boundary

Phase 4 must implement the canonical immutable bundle schema, pure versioned
feature derivation from sealed evidence, one champion-pointer loader, and the
canonical serving path. It must replace the temporary raw-registry descriptor
with honest mandatory prediction provenance. Phase 3 does not assert missing
`trained_through`, promotion, feature signature, EV threshold, or dependency
manifest facts.

Do not alter the Phase 3 barrier semantics: Phase 4 supplies computation behind
`DeferredPredictor`; it does not select by environment, registry rank, file
recency, mock, retraining, heuristic, V3, Unified, SP tie-break, or GPT rerank.

## Fresh-session Phase 4 prompt

> Complete Phase 4 only from
> `docs/agent_tasks/canonical_race_forecasting_phase1_foundation_20260722.md`,
> `docs/CANONICAL_RACE_FORECASTING_SPEC.md`, ADRs 0001–0008, and the committed
> Phase 1–3 handoffs/code. Implement the canonical immutable model bundle, pure
> versioned feature derivation from sealed evidence, one champion-pointer
> loader, mandatory prediction provenance, and canonical serving adapters.
> Integrate through Phase 3's `DeferredPredictor` seam without weakening its day
> pin, seal binding, quarantine, result barrier, retry, join, or on-demand
> separation. Do not select by environment, registry ranking, recency, fallback,
> mock, V3, Unified, heuristic, SP, or GPT. Inspect only the smallest safe
> canonical artifacts read-only; do not mutate services or runtime state. Run
> focused and relevant full validation, commit coherent source/docs/tests, stop
> before Phase 5, and provide the next fresh-session prompt.
