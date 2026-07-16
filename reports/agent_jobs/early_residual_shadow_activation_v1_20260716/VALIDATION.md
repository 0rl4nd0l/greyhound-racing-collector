# Validation

## Focused code validation

- Red proof: five new assertions failed before the model-path wiring existed.
- Focused suite: 185 passed across
  `tests/test_predict_market_form_residual.py` and
  `tests/test_shadow_autopilot_daemon.py`.
- Python compile: passed.
- Ruff: passed with repository-baseline F601 and F541 exclusions only.
- `git diff --check`: passed.
- V2 task-card validation and exact diff allowlist: passed.
- Generated systemd unit verification: passed; only unrelated host-unit
  warnings were emitted.
- Independent code review: no critical findings, warnings, or suggestions.

The regression test uses an explicitly configured Stage-2 model outside the
runtime evidence root and forces automatic model discovery to return `None`.
The plan still becomes `READY`, proving that the configured pin rather than an
outcome-informed or filesystem-selected alternative is used.

## Frozen identities

- Frozen residual model SHA-256:
  `624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d`
- Frozen residual manifest SHA-256:
  `8537cbc3d843d106a1fe48793ef01197454ef092c0244025fd65685636a42080`
- Stage-2 feature model SHA-256:
  `d7e9ff35b383a0e6400bcb67bcf6df374e4c0bfe6c974f32d1c9f057876e471d`
- Installed/generated odds service SHA-256:
  `8d798ce374495486c839c42a6687685c2acab8027d4fd73190bcf5b7bd50380e`
- Installed/generated odds timer SHA-256:
  `f359bc6327a5e1d3094844d58305a1eabc9304f5f8c7ceac7826285e0cdad1e1`
- Installed/generated full service SHA-256:
  `b17e866b5866480345d98dbc1f01dad4f097575afc735b1fbb77efebd9370767`
- Installed/generated full timer SHA-256:
  `258d67692a1284998e9579f0a863410f2ef1dbd72f378e84f25386694a050e28`

## Runtime boundary

The 23:08 capture legitimately appended 12 odds rows before the early stage
failed closed as `feature_model_missing`. The early stage did not reach its
read-only feature command. After the repair, scheduled waiting invocations did
not enter capture or feature scoring. The current database main-file SHA-256 is
`2a5d24f72433ad47d04cf91ae98dfbbebe9df56a8d817f89d63bed3ca82bcdbf`;
WAL activity belongs to the authorized capture lane and is not claimed as
unchanged. No SQL write path was added to the residual stage.

See `LIVE_PROOF.md` for the deliberately partial first-append proof.
