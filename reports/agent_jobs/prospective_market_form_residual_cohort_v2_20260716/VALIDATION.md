# Validation

## Determinism and artifacts

- Model SHA-256: `624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d`
- Manifest SHA-256: `8537cbc3d843d106a1fe48793ef01197454ef092c0244025fd65685636a42080`
- Fit-population SHA-256: `145505fe20420ed457736774e62e20431d7da261378f88abeae364728a6223dc`
- Parameters identical across same-fit verification: pass
- Preprocessing identical: pass
- Optimizer result identical: pass
- Canonical model bytes identical: pass
- Fixed-fixture full and half predictions identical: pass
- Primary persisted model count: 1
- Alternative models created: 0

## Focused validation

- `uv run --no-project --with 'numpy==1.26.4' --with pytest python -m pytest -q --noconftest tests/test_market_form_residual.py`: 25 passed.
- `uv run --no-project --with ruff ruff check src/predictor/market_form_residual.py tests/test_market_form_residual.py`: passed.
- `python3 -m py_compile src/predictor/market_form_residual.py tests/test_market_form_residual.py`: passed.
- `git diff --check`: passed in independent review and is repeated at closeout.

`--noconftest` is intentional: repository-wide `conftest.py` imports Flask,
which is not needed by this isolated NumPy scorer test. The focused command
uses a pinned NumPy 1.26.4 environment matching the frozen artifact.

## Boundary validation

The scorer tests cover artifact loading and hash binding, canonical JSON,
exact algorithm/derivation enforcement, feature order, missing values,
normalization, full/half derivation, complete runner sets, input-order
determinism, fixed-fixture scoring, provenance timestamps and hashes,
outcome rejection, non-finite rejection, append-only idempotency, stable record
identity, and conflicting/malformed record rejection.

No runtime functionality claim is made: live output proof is intentionally
`PARTIAL` because activation was forbidden by the owner and the card.
