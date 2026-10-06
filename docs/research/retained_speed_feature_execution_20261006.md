# Retained historical first-sectional feature execution

The [adapter](/mnt/tenn-nvme2/tenn/greyhound-speed-features-20261006/race_collection/retained_speed_features.py) and [CLI](/mnt/tenn-nvme2/tenn/greyhound-speed-features-20261006/scripts/derive_retained_speed_features.py) compute private historical features for the entire fixed 82-card / 583-runner roster. This is the user's separately authorized working-semantics experiment: TheDogs `1 SEC` is treated as the individual runner's first sectional in seconds. `TIME` is understood as the individual total time but is not a feature in this construction. The older stricter hypothesis, zero-qualified disposition and presence-only audit remain unchanged.

Construction uses the [pure module](/mnt/tenn-nvme2/tenn/greyhound-speed-features-20261006/race_collection/historical_speed_features.py): the most recent three usable, nonconflicting, distinct prior dates at an exact matching track/distance key, median first sectional, and median absolute deviation. Supported individuals remain available when other runners lack history. A within-field median-relative time gap is emitted only for a wholly supported comparable field. These values are time summaries, not measured physical speed, model performance or predictive advantage. There is no age threshold, pooling, fitting, label read, par, grade adjustment, going correction or run-home feature.

The sole retained input manifest is `/home/l4nd0/greyhound-recovery-20261003/persistent-operation/research-continuation-20261005/speed-breadth-proposal/manifest.PROPOSED.json`, SHA256 `922456735d4117092fbec50333d68bbb8743008fc2c73b1fc8efa48529148223`. It binds the original membership SHA256 `782d6af9bcbe1dd84d1c2674de44f93bc6720cd77b3b7e0195bdcccc03dd4342`. The old manifest supplies immutable input references only; its presence-only purpose and expired execution receipt are not revived as authority to emit features. Root issues a fresh dated receipt and fixed launch intent externally before processing real values. The CLI has no added inner authority or claim framework.

## Authentication and actual source assumptions

The unchanged `retained_card_timing_coverage.verified_member` authenticates membership, original admission and bundle hashes, accepted/raw CSV hashes, target identity and distance, source-page receipt and URL, pre-jump sealing, complete native card roster and source blocks. The adapter then reuses its `projected_card`, `parse_card_target_roster_bytes` and `parse_form_blocks_bytes` interfaces. It performs no fuzzy dog join or cross-card runner linkage: each opaque identity is bound to this race, box and the already validated canonical roster token.

The target track key is the **literal retained admission venue**, while the historical key is the CSV `TRACK` string with surrounding whitespace removed. No venue alias table, case folding or claimed provider-certified canonical namespace is introduced. Distance uses the existing strict positive-integer-metres parser with an optional literal `m` suffix. Exact equality of these source keys is a working same-context assumption, not proof of identical historical layout, sectional endpoint or clock. The fixed retained CSV header has no layout/era columns. Unexpected columns or an explicit unbound layout/clock declaration fail rather than silently disappear; the pure module handles explicitly bound known contexts in its separate interface.

Each parsed observation also carries an opaque fingerprint of the existing timing audit's complete whitelisted projection (`DATE`, `TRACK`, `DIST`, `TIME`, `WIN`, `BON`, `1 SEC`, `PIR`). Thus identical duplicate renderings deduplicate, but conflicting observations on a date remain excluded even when their first sectionals happen to agree. The fingerprint passes no additional raw timing values into the pure interface. Placement and odds columns are removed by the unchanged projection and are never emitted.

Historical dates have day resolution. Same-day/future histories are excluded, as are histories after the retained availability date in the cutoff's original timezone. The adapter uses the **recorded admission `decision_at`** as cutoff and verifies its relation to admission/jump and the authenticated evidence; it does not invent a universal T-120 requirement. `original_published_complete_at` is a conservative bound by which the sealed historical card was available. It is not an original prediction timestamp or the exact CSV capture time. The separate primary-page receipt must also precede cutoff. All 82 actual admission/receipt/publication timing bindings passed a metadata-only check; no historical timing cells were decoded for that check.

## Finite execution and outputs

The unchanged `Reader` enforces SHA256, exact sizes, regular canonical paths, stable file identity, 1,150 reads, 38,189,496 bytes, an 8 MiB per-file maximum and 300 seconds. Exact manifest sizes predict **904 reads / 33,838,183 bytes**, leaving 4,351,313 bytes. This includes seven authenticated role reads plus rereads of accepted CSV, sidecar, admission and primary receipt for each member, along with manifest/membership. These measured input sizes are metadata, not an empirical feature result. Output is bounded to the original 4 MiB cap, with a 64 KiB failure-accounting reserve.

The output directory is created exclusively with mode0700; files use mode0600. `features.private.json` contains individual feature values, selected dates/ages and source bindings. `summary.json` contains counts, missingness/exclusion categories, per-race roster support, assumptions and the private artifact digest; it contains no timing values or rankings. Every member remains in original manifest order, including quarantines. The CLI prints only aggregate support counts and artifact digest.

A shared hash, schema, identity, resource or deadline failure aborts the entire construction. The successful summary is published last, after every member and private write succeeds. `FAILED.json` retains completed, actively failed and unattempted race identities without values. Pending/private bytes may remain after an output-stage failure, but no successful summary is published. An existing output directory is never reused or overwritten. Root preserves the failed attempt and any new explicit execution receipt rather than silently retrying.

```bash
/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-operator-ui-r3-python311-20260803-9d58f340/bin/python \
  -B -m scripts.derive_retained_speed_features
# DEFAULT_OFF; no supplied manifest is opened
```

After committing/reviewing this exact implementation, root uses its fresh receipt, network-denied read-only sandbox and 310-second external watchdog, with only the new private output parent writable. Inside that sandbox, from the reviewed checkout:

```bash
/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-operator-ui-r3-python311-20260803-9d58f340/bin/python \
  -B -m scripts.derive_retained_speed_features --execute \
  --manifest /home/l4nd0/greyhound-recovery-20261003/persistent-operation/research-continuation-20261005/speed-breadth-proposal/manifest.PROPOSED.json \
  --manifest-sha256 922456735d4117092fbec50333d68bbb8743008fc2c73b1fc8efa48529148223 \
  --output /ABSOLUTE/ROOT-BOUND/NEW/PRIVATE/OUTPUT
```

Validation: 10 fabricated adapter tests and 28 existing timing-coverage tests pass with the pinned Python, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, and `--noconftest`. They exercise actual authentication/parser plus feature computation for an invented 82/583 roster, private values versus value-free CLI/reporting, latest-three selection after whole-observation conflicts, incomplete-field behavior, pre-cutoff availability, consumed output paths, active-failure accounting, and read/output/deadline rejection. Only fixed identity pins are replaced for the fabricated graph; authentication and feature computation are not mocked. No real feature run occurred during implementation. Root owns subsequent values-private execution and reports measured construction support separately from these tests.
