# Forecasting observation-canary operator contract

This document describes repository capability only. It does not authorize or
prove installation, live execution, prediction, database migration, model
selection, training, promotion, betting, or deployment. Runtime status remains
`DATA_MISSING` until a separately authorized canary produces current,
source-bound evidence.

The observation canary is result-blind. Its immutable Racing Day plan retains
all nine command identities, but `mode: "result-blind-observation-v1"` permits
execution only through ordinal 5, `deferred_prediction`. The runtime input must
omit every race's `result` and `training_example` objects. The service must
produce exactly the contiguous receipt prefix 1--5 and must not read a result
checksum or execute result collection, joining, reconciliation, training
requests, evaluation, or promotion.

## Non-negotiable boundaries

- The legacy database is never an operations-database target. Every operator
  database command requires both absolute paths and rejects path, symlink, and
  hard-link aliases before opening the operations store.
- Programme migrations 0001--0029 apply only to a separately supplied
  operations database. A fresh successful migration has the exact version set
  1 through 29.
- Live odds capture remains authoritative. This capability installs no unit,
  creates no timer, owns no capture schedule, and authorizes no interruption,
  restart, lock removal, or database mutation in the live collector.
- Champion and challenger bundles must already be immutable canonical
  registrations. The champion must also be the exact canonical Racing Day
  assignment. The observation interface cannot invent, register, select,
  promote, or switch a model.
- Every mutating operator command requires one explicit operation ID and aware
  timestamp. Exact replay is permitted; changed intent under an existing
  operation ID is rejected.
- Direct SQL, implicit paths, environment-selected inputs, interactive prompts,
  and ad hoc persistence scripts are unsupported.

## Preconditions for any future canary

A fresh preflight must prove all of the following from current state:

1. The reviewed change is merged into the exact clean canonical checkout. The
   registered release manifest binds that commit, tree, service source,
   executable bytes and executable mode.
2. The supplied interpreter is the intended virtual-environment executable and
   reports exact Python 3.11. Unit generation rejects any other version.
3. The legacy database exists at its verified absolute path. The separately
   supplied operations path is not the same file, symlink target, or hard link,
   and its parent already exists.
4. The operations store is either new or already contains the exact schema
   chain 1--29. It is never inferred from filename or recency.
5. Canonical policy, configuration, candidate release and preserved legacy
   release documents are byte-canonical JSON and mutually consistent. The
   configuration names the exact operations database, artifact root, runtime
   adapter, runtime-input checksum and source allowlist.
6. The legacy rollback pointer is initialized and still has
   `authority=legacy` and `legacy_preserved=1`. Exactly one candidate has a
   current observation authorization.
7. The immutable runtime input is result-blind, contains no result or training
   example, and binds pre-existing programme, card/form, pre-jump odds,
   champion, challenger, day-assignment and artifact identities. Those inputs
   must be sourced without changing the live odds-capture owner.
8. No installed service, unit, timer, listener, lock, working directory or
   database path would collide with the live collector. The proposed unit has
   `WantedBy=default.target`, uses the verified Python 3.11 executable and has
   no timer.
9. A new isolated snapshot path and separate replica artifact root are
   available for the operations database. Restore validation targets only that
   snapshot and replica.

Any missing or ambiguous item is `STOP / DATA_MISSING`.

## Supported noninteractive interface

`bin/race-collection-operator` is the only repository operator entrypoint for
this contract. The path values below are examples of the required absolute
shape; an operator must replace them with preflight-proven literal paths before
separate authorization.

Create or migrate only the separate operations store:

```bash
/absolute/release/bin/race-collection-operator migrate \
  --operations-db /absolute/operations/race_collection_operations.sqlite3 \
  --legacy-db /absolute/legacy/greyhound_racing_data.db
```

Register immutable authority documents in dependency order:

```bash
/absolute/release/bin/race-collection-operator register-policy \
  --operations-db /absolute/operations/race_collection_operations.sqlite3 \
  --legacy-db /absolute/legacy/greyhound_racing_data.db \
  --artifacts-root /absolute/operations/artifacts \
  --document /absolute/authority/policy.json \
  --operation-id op_00000000000000000000000000000001 \
  --at 2026-07-26T00:00:00+00:00

/absolute/release/bin/race-collection-operator register-config \
  --operations-db /absolute/operations/race_collection_operations.sqlite3 \
  --legacy-db /absolute/legacy/greyhound_racing_data.db \
  --artifacts-root /absolute/operations/artifacts \
  --document /absolute/authority/configuration.json \
  --operation-id op_00000000000000000000000000000002 \
  --at 2026-07-26T00:00:01+00:00

/absolute/release/bin/race-collection-operator register-release \
  --operations-db /absolute/operations/race_collection_operations.sqlite3 \
  --legacy-db /absolute/legacy/greyhound_racing_data.db \
  --artifacts-root /absolute/operations/artifacts \
  --document /absolute/authority/legacy-release.json \
  --operation-id op_00000000000000000000000000000003 \
  --at 2026-07-26T00:00:02+00:00

/absolute/release/bin/race-collection-operator register-release \
  --operations-db /absolute/operations/race_collection_operations.sqlite3 \
  --legacy-db /absolute/legacy/greyhound_racing_data.db \
  --artifacts-root /absolute/operations/artifacts \
  --document /absolute/authority/candidate-release.json \
  --operation-id op_00000000000000000000000000000004 \
  --at 2026-07-26T00:00:03+00:00
```

Initialize the exact rollback authority and authorize observation:

```bash
/absolute/release/bin/race-collection-operator initialize-legacy \
  --operations-db /absolute/operations/race_collection_operations.sqlite3 \
  --legacy-db /absolute/legacy/greyhound_racing_data.db \
  --artifacts-root /absolute/operations/artifacts \
  --release-id legacy-release \
  --actor authorized-operator \
  --reason "record exact preserved legacy rollback authority" \
  --operation-id op_00000000000000000000000000000005 \
  --at 2026-07-26T00:00:04+00:00

/absolute/release/bin/race-collection-operator authorize-observation \
  --operations-db /absolute/operations/race_collection_operations.sqlite3 \
  --legacy-db /absolute/legacy/greyhound_racing_data.db \
  --artifacts-root /absolute/operations/artifacts \
  --release-id candidate-release \
  --actor authorized-operator \
  --reason "separately approved result-blind observation canary" \
  --operation-id op_00000000000000000000000000000006 \
  --at 2026-07-26T00:00:05+00:00
```

Generate—but do not install—the user-service content:

```bash
/absolute/release/bin/race-collection-operator generate-user-service \
  --release-document /absolute/authority/candidate-release.json \
  --configuration-document /absolute/authority/configuration.json \
  --config-path /absolute/authority/configuration.json \
  --python-executable /absolute/python311-environment/bin/python
```

The command prints canonical JSON containing one `race-collection.service`
unit. It performs no systemd write and emits no timer.

Create a backup and validate only an isolated restore:

```bash
/absolute/release/bin/race-collection-operator backup \
  --operations-db /absolute/operations/race_collection_operations.sqlite3 \
  --legacy-db /absolute/legacy/greyhound_racing_data.db \
  --artifacts-root /absolute/operations/artifacts \
  --backup-id backup-20260726 \
  --racing-day-id day_00000000000000000000000000000000 \
  --snapshot /absolute/isolated/backup-20260726.sqlite3 \
  --replica-root /absolute/isolated/replica \
  --operation-id op_00000000000000000000000000000007 \
  --at 2026-07-26T00:00:06+00:00

/absolute/release/bin/race-collection-operator validate-restore \
  --operations-db /absolute/operations/race_collection_operations.sqlite3 \
  --legacy-db /absolute/legacy/greyhound_racing_data.db \
  --artifacts-root /absolute/operations/artifacts \
  --backup-id backup-20260726 \
  --drill-id restore-20260726 \
  --snapshot /absolute/isolated/backup-20260726.sqlite3 \
  --replica-root /absolute/isolated/replica \
  --operation-id op_00000000000000000000000000000008 \
  --at 2026-07-26T00:00:07+00:00
```

An exact backup replay returns the recorded checksum only when the isolated
snapshot bytes still match. Restore validation never copies the snapshot over
either the operations database or the legacy database.

## Rollback boundaries

The result-blind observation canary does not switch the active release pointer.
Its normal rollback is therefore:

1. stop only the separately installed candidate process or unit, if a future
   authorization installed one;
2. revoke candidate observation authority with a new explicit operation ID;
3. retain the separate operations database and artifacts as append-only
   evidence; and
4. verify that the legacy pointer, live collector, live odds rows, locks,
   listeners and timers are unchanged.

Revocation is available as:

```bash
/absolute/release/bin/race-collection-operator revoke-observation \
  --operations-db /absolute/operations/race_collection_operations.sqlite3 \
  --legacy-db /absolute/legacy/greyhound_racing_data.db \
  --artifacts-root /absolute/operations/artifacts \
  --release-id candidate-release \
  --actor authorized-operator \
  --reason "stop result-blind observation authority" \
  --operation-id op_00000000000000000000000000000009 \
  --at 2026-07-26T00:00:08+00:00
```

`activate` is not an observation-canary command. It is exposed only for a later,
separately reviewed cutover after the existing two-complete-day and prospective
boundary gates pass. It atomically consumes observation authority and changes
the operations-store pointer; it does not touch the legacy database or stop the
legacy service. `rollback` reverses only that pointer to the initialized legacy
release. Neither command installs, starts, stops or restarts a service.

Stop and retain evidence if any result-blind cycle has a non-contiguous prefix,
an ordinal above 5, a result read, a result/join/training/evaluation/promotion
row, a release-identity mismatch, a model-registration mismatch, or any effect
on live odds capture. Do not repair those conditions by deleting rows,
rewinding authority, bypassing locks, or changing the legacy database.
