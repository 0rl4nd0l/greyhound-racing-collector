# Market-form residual record contract

The frozen market-form residual scorer remains outcome-free, shadow-only, and
inactive by default. This document describes record construction and replay;
it does not authorize prediction activation, deployment, model refitting, or
production data access.

## Portable numerical output

Record schema `market_form_residual_shadow_record_v3` uses numerical
canonicalization contract
`market_form_residual_numeric_canonicalization_v1`. Scoring uses stable runner
and feature order, scalar binary64 operations, `math.fsum` for every reduction,
and the standard-library `log`, `exp`, and `tanh` functions. Before full/half
probability variants are derived, each finite residual adjustment is then
quantized to 15 decimal places with decimal round-half-even. Negative zero is
normalized to positive zero.

The boundary is part of effective-state schema
`market_form_residual_effective_state_v2`, so it changes the effective-state
hash and is covered by record checksum/key reconstruction. Market probability
and both probability variants are derived normally from the canonical residual
vector; no tolerance is used during sealing or replay.

Rounding alone was insufficient: 15 places still produced 15 complete-record
differences in an adversarial 3,000-case cross-runtime sweep, and 14 places
still produced one. The ordered scalar calculation removes the NumPy
wheel/runtime reduction difference before the representation rule is applied.
With both parts active, all 3,000 complete records matched at aggregate SHA
`34b92610bc5932f00f69a10985a8368bf7c1d31e96c58dcf1f3c6cd750d2ee06`.

Against the prior Python 3.11 calculation over the same sweep, maximum absolute
deltas were `1.3322676295501878e-15` for residual adjustment,
`2.7755575615628914e-16` for full probability, and
`2.220446049250313e-16` for half probability. All meaningful strict ranks and
winners were preserved. Seven sub-`3e-16` last-bit leaders became exact top
ties that still contain the prior leader; runner order, normalization, caps,
and explicit tie behavior remain deterministic.

## Representation migration

V1, v2, mixed, or incomplete history is not reinterpreted as v3. The writer
raises `history_migration_required` before append and leaves target bytes
unchanged. An operator must choose a separately reviewed output path or an
explicit migration process; this scorer never rewrites old records in place.

For identical frozen artifacts, runners, and provenance, Python 3.11.15 and
3.13.12 with NumPy 1.26.4 must emit the same complete canonical bytes, record
SHA, checksum, key, and effective-state hash. Each runtime must return
`EXACT_REPLAY` for a v3 history row emitted by the other.

The pinned digest and CI claim cover the repository's supported x86-64 Linux
runtime surface. A new operating system, architecture, Python build, or math
library is not silently assumed compatible: it must pass the same digest and
cross-replay gates before its records are treated as portable.

NumPy 1.26.4 has no published CPython 3.13 wheel. The focused CI therefore
builds that locked dependency from source on its first x86-64 Linux cache miss
and caches the reviewed wheel for later runs; this affects check latency, not
the scoring contract or production runtime.

## Preserved boundaries

- Frozen model, manifest, fit-population, coefficient, and exclusion bytes do
  not change.
- Outcomes and result-shaped fields remain prohibited.
- Exact provenance, runner-set, artifact, effective-state, checksum, record-key,
  whole-history, forgery, lock, descriptor, atomic-publication, and durability
  checks remain authoritative.
- Checksums detect corruption and inconsistent construction; they are not an
  external signature against a hostile actor controlling the filesystem.
- No record is a betting recommendation or permission to activate the model.
