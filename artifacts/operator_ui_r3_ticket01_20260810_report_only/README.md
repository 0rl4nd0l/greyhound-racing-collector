# Operator UI R3 Ticket 01 authoritative-source proof

Status: `PASS_WITH_REPOSITORY_WIDE_COLLECTION_LIMITATION`

Observed at `2026-08-10T19:37:32+10:00` in a read-only runtime session.

## Authoritative source

- Isolated checkout: `/home/l4nd0/operator-ui-r3-ticket01-authoritative-HCOLll/source`
- Detached commit: `d343af94a57af80327dd41f18433f7466f86ca0d`
- Tree: `a04197ba455de2549f9c76adcd474d1feb520bd1`
- `git status --porcelain=v1`: empty
- Repository-owned contract files (SHA-256):
  - `configs/operator_ui/repository-v1.toml`: `eb6f2018a086e7543d4ae8a6705d6e2b3fb9972128cc33f102eab7dda6342aff`
  - `src/operator_ui/deployment.py`: `7584c80957460bdd1c81d292ff59d7dec2a54adf8a828bff52c2113b7fab25b0`
  - `src/operator_ui/prediction_worker.py`: `72fc69df95c409ecb83e1ebd9018e1e7b6826679eff27ad111889a2d38aee9d0`
  - `artifacts/frozen_models/market_form_residual_v1/model.json`: `624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d`
  - `artifacts/frozen_models/market_form_residual_v1/manifest.json`: `8537cbc3d843d106a1fe48793ef01197454ef092c0244025fd65685636a42080`
  - `configs/prediction/schemas/market_form_residual_v1.schema.json`: `89c1c758adb037ed1213ecd9bd29fd2cc0e470e86d06bf4654ccecefa140bdb2`
  - `docs/manual_live_market_form_residual_prediction.md`: `adf2c16a2d3638c170de7fd675715458081d7d1c17b17114ad41b615bad7ad5f`

## Frozen runtime and tests

- Python: `3.11.15`
- NumPy: `1.26.4`
- `<pinned-python> -m pytest -q tests/operator_ui/test_deployment_generator.py`:
  exit 0; 112 passed in 69.20s
- `<pinned-python> -m pytest -q tests/operator_ui/test_r3_e2e_safety.py`:
  exit 0; 2 passed in 15.35s
- `<pinned-python> -m pytest -q tests/test_market_form_residual.py tests/test_market_form_residual_portability.py`:
  exit 0; 114 passed in 943.62s
- `<pinned-python> -m pytest -q tests/test_predict_race_now.py::test_model_config_mismatch_and_alias_resolution_fail_or_resolve_exactly`:
  exit 0; 1 passed in 12.97s

`<pinned-python>` is
`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-operator-ui-r3-python311-20260803-9d58f340/bin/python`.

The one repository-wide `pytest -q` attempt did not execute tests: collection
was externally terminated after approximately 4,500 items. A collection-only
diagnostic was likewise terminated with exit 143. This is recorded as a broad
suite limitation; the ticket's declared highest seams above all passed.

## Selector and immutable identities

`latest-research` resolves to `market_form_residual_v1` through the checked-in
production resolver, with alias resolution required. The focused resolver test
above proves the exact mapping and rejects mismatched model/config inputs. The
repository pins:

- model SHA-256: `624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d`
- manifest SHA-256: `8537cbc3d843d106a1fe48793ef01197454ef092c0244025fd65685636a42080`
- schema SHA-256: `89c1c758adb037ed1213ecd9bd29fd2cc0e470e86d06bf4654ccecefa140bdb2`
- `manual-default` SHA-256: `f8a3c321dca12321a38a4d12a08f4f43461e1c1e73100eda871fd60252ed1820`

## Rejected active runtime retained as evidence

The installed `greyhound-operator-ui-r3.service` was observed as loaded,
active, and running with PID `1692390`. Its source identity is commit
`7b6143259b201242cb667c6b6c9056defbaec73d` and tree
`7f354484077eef0478c5487d9f829d666006a9e9`, so it remains disqualified. The
process working directory was
`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-operator-ui-r3-patch-deploy-20260810-f0b`.
It listened only on `127.0.0.1:5055`.

Read-only `systemctl --user show` linked that PID to an `ExecStart` using the
pinned Python above, module `src.operator_ui.deployment`, source root equal to
the process working directory, host `127.0.0.1`, and port `5055`. The installed
unit fragment was `/home/l4nd0/.config/systemd/user/greyhound-operator-ui-r3.service`.
The process environment declared the rejected commit/tree recorded above.

No service was edited, reloaded, restarted, stopped, or started. No runtime
database, external source, model registry, or remote repository was accessed
for mutation. The existing canonical working tree and its tracked and
untracked work were not used as the deployment source and were left intact.
