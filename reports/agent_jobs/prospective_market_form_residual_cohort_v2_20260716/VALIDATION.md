# Validation

## Source-proven exploit and repair

Before implementation, the regression suite produced `14 failed, 21 passed`.
The standalone exploit proved `predictions_changed=True` while
`model_sha_unchanged=True`, `manifest_sha_unchanged=True`, and
`record_key_unchanged=True`; the first mutated append returned `APPENDED` and
the original score then conflicted.

After implementation:

- Direct nested mutation raises `TypeError`.
- Forced array mutation raises `ResidualContractError:effective_state_sha256_mismatch`.
- Forced mutation plus a forged new key raises `ResidualContractError:encapsulated_score_state_mismatch`.
- A mutated caller record raises `ResidualContractError:shadow_record_not_canonical_score`.
- Canonical append returns `APPENDED`, then `EXACT_REPLAY`.
- Fixed full and half fixture predictions remain exact.
- Fixed-fixture effective-state SHA-256 is `97da118363975ae63183a81b7d7773b7c6b7aff8377239703288a7c0f4bea95f`.

## Automated validation

- Focused scorer/writer suite: 37 passed.
- Resource and collector-lock regression suite: 133 passed.
- Ruff check: passed.
- Ruff format check: passed.
- `py_compile`: passed.
- `git diff --check`: passed.
- Clean post-PR45 integration simulation on master `c1dfd464cf6ecfb2034f96ac1a8d3ea58d4e6afa`: passed and merge state was aborted cleanly.
- Model SHA-256: `624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d`.
- Manifest SHA-256: `8537cbc3d843d106a1fe48793ef01197454ef092c0244025fd65685636a42080`.

The suite covers immutable nested state, non-aliasing read-only arrays,
effective-state verification, cached-hash and effective-key forgery, writer-side
canonical rescoring, caller-record forgery, input-order determinism, canonical
JSON history, append idempotency, malformed/conflicting history, fixed fixture
stability, provenance, runner-set integrity, and outcome rejection.

No production database or prospective outcome was read. No runtime output was
written and no service or timer was changed. The separately activated runtime
still uses the original two-argument writer API, so future deployment must
adapt that caller under separate authorization.
