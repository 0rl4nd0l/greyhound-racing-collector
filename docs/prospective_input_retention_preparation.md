# Prospective input retention correction — prepared, not activated

The fixed September 16 metadata sample has five source-bound fields and exact
grades, and ten correct pre-jump WIN receipts. Its scheduled receipt schema does
not bind the complete database-plus-form history, generator, model and config.
A mutable database available months later cannot repair that capture-time gap.

The isolated `scripts/retain_prospective_inputs.py` runner adds an input-only
archive, reusing `seal_history_database` rather than invoking the prediction
pipeline. It saves exact hash-bound supplied source bytes and an immutable
history copy before an actual-clock prediction deadline. It records source and
sealed history hashes, source observation time, archive interval, cutoff, jump,
and all supplied artifact identities. Target, same-date and future DB rows are
excluded by the existing history policy. Raw histories never reach stdout.

This is a retention primitive, not automatic semantic qualification. It does
not prove source rights, race/runner alignment, form-history dates, grade
meaning, archive source closure, completeness of upstream collection, absence
of earlier protected targets, or training separation merely by hashing files.
The caller supplies an authenticated, approved inventory; no collector job,
timer, production option or current receipt schema changes. Existing files
are copied, not rewritten. No result access or scorer is called.

## Concrete invocation contract

After separate access approval, one exact inventory is frozen and passed as:

```
python scripts/retain_prospective_inputs.py \
  --inventory APPROVED_INVENTORY.json \
  --inventory-sha256 APPROVED_DIGEST \
  --destination NEW_ISOLATED_ATTEMPT_DIRECTORY
```

Inventory schema `prospective_input_inventory_v1` has exactly `schema_version`,
`race_id`, `runner_names`, `observed_at`, `prediction_cutoff`, `jump_at`,
`history_source`, and `files`. `files` maps each role in `REQUIRED_ROLES` to
`path` and `sha256`. Use aware timestamps, with jump in Australia/Melbourne
local time for the existing date cutoff. `observed_at` is the latest source
observation among the supplied predictor inputs. Both acquisition and complete
publication must precede `prediction_cutoff`, which itself precedes jump.

Required roles: normalized form, sidecar, raw form, primary page and receipt,
exact odds receipt and report, frozen model and manifest, configuration,
feature schema, complete generator source archive, and environment lock.
The generator archive must contain the entire reviewed source revision and
dependency inputs (not just a Git identifier); the environment lock must pin
Python and installed package versions. Archive original paths in the manifest
for replay mapping without rewriting source metadata. Later replay must use
an isolated filesystem mapping and sealed history, never current live paths.
Input-byte retention precedes feature generation; this runner does not certify
that later replay yet works. A retained packet is explicitly
`INPUTS_RETAINED_NOT_QUALIFIED`.

## Minimal rollout and verification scope

Publish this isolated patch for independent review first. Before making it a
recurring collector operation, authorize one future input-only acceptance:
one predetermined eligible race, one authenticated WIN capture, one fresh
database snapshot and supplied source inventory; no retries or substitutions.
Its population must be disjoint from protected studies. Permission for copying
and filtering earlier protected results as machine-only history must be
explicit; ordinary metadata inspection does not supply that permission.

Use the existing source/identity/WIN preflight and current-index freshness
gates. On that separately authorized acceptance, check raw form date/identity
and source lineage, generate features twice in an isolated filesystem with no
live database/network fallback, and compare all 16 values and missingness.
Keep feature/history payloads private. Do not score the model. A complete
packet requires those checks plus a source archive/environment replay before
recurrent retention can be called evaluation-ready. Failure terminates the
one-race acceptance without substitution.

Recurring integration, deployment/restart, model prediction sealing, target
outcome access and statistical evaluation are separate later scopes. This
patch does not silently activate any of them or amend any protocol. The
existing collector continues as installed; current records must not be
labelled qualified simply because this capability now exists offline.

## Focused synthetic validation

`tests/test_prospective_input_retention.py`: exact archived bytes survive later
source changes; earlier synthetic history survives while target/same-day/future
rows do not; repeated destinations, missing inputs, changed model bytes, and
late acquisition are rejected without a complete manifest. No historical or
current payload is used by the tests.
