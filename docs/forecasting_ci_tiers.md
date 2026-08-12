# Forecasting CI tiers

The stable pull-request check is `tests-race-collection`. A classifier selects
one primary tier for the complete change set and separately reports whether the
CI contract changed. Unknown paths, destructive changes, shared surfaces, and
incompatible product-tier combinations are classified as `full_forecasting`,
but normal pull requests execute the bounded `pr_fast` fallback for that risk
tier. The complete suite is opt-in only.

| Tier | Intended paths | Validation command |
| --- | --- | --- |
| `ci_contract` | `.github/**`, `scripts/ci/**`, `tests/ci/**` | `uv run --no-project --with-requirements requirements/all.in --with PyYAML python scripts/ci/run_forecasting_ci_contract.py` |
| `manual_prediction` | isolated manual-capture contracts, atomic evidence sealing, and market-residual scoring | `uv run --no-project --with-requirements requirements/all.in --with jsonschema python -m pytest -q --noconftest tests/test_manual_independent_capture.py tests/test_manual_independent_capture_executor.py tests/test_manual_independent_capture_sealer.py tests/test_market_form_residual.py tests/test_market_form_residual_portability.py tests/test_predict_market_form_residual.py` |
| `official_results` | official-result fetch, parse, and label reporting | `uv run --no-project --with-requirements requirements/all.in python -m pytest -q tests/test_results_ingest_official_first.py tests/test_autonomous_official_result_capture.py tests/test_expert_form_official_result_labels_packet.py` |
| `forward_corpus` | scheduled producer, observer, admission, and closure | `uv run --no-project --with-requirements requirements/all.in python -m pytest -q --noconftest tests/race_collection/test_scheduled_forward_corpus.py tests/race_collection/test_forward_official_result_observer.py tests/race_collection/test_phase7_source_admission.py tests/race_collection/test_forward_sealed_corpus.py` |
| `operator_ui` | isolated foundation, job-store, worker, security, R3, and connected-UI paths | `uv run --no-project --with-requirements requirements/all.in python -m pytest -q tests/operator_ui/test_foundation.py tests/operator_ui/test_job_store.py tests/operator_ui/test_prediction_worker.py tests/operator_ui/test_security.py tests/operator_ui/test_r3_api.py tests/operator_ui/test_r3_e2e_safety.py tests/operator_ui/test_connected_ui.py` |
| `forecasting_core` | isolated phase-3 forecasting and phase-4 serving | `uv run --no-project --with-requirements requirements/all.in python -m pytest -q --noconftest tests/race_collection/test_phase3_forecasting.py tests/race_collection/test_phase4_model_serving.py` |
| `pr_fast` | bounded PR fallback when the classifier reports `full_forecasting` | five named smoke tests covering prediction, results, forward capture, Operator UI, and forecasting core |
| `full_forecasting` | explicit full-validation request; also the classifier's risk label for PR fallback | `uv run --no-project --with-requirements requirements/all.in --with PyYAML --with jsonschema python scripts/ci/run_full_forecasting.py` |
| `non_forecasting` | known unrelated documentation and presentation paths | no Forecasting test command |

Pure CI routing changes select `ci_contract`. When CI-contract paths accompany
exactly one focused product tier, the product tier remains primary and
`ci_contract_changed=true`: the stable job runs the CI-contract command first,
then that focused product suite. Each command, exit code, log hash, commit, and
tree is retained in the exact-head attestation. `ci_contract` validates the
classifier contracts and all workflow YAML files, then runs one named smoke from
every focused tier. It never invokes the complete `tests/race_collection`
directory.

The full runner executes the CI contract, the complete `tests/race_collection`
directory, and the top-level focused suites. It runs only on an explicit manual
workflow dispatch or on a PR labeled `ci:full-forecasting`. The scheduled
automatic full run has been removed.

Shared identity, timing, provenance, feature, schema, dependency, training,
evaluation, promotion, runtime-adapter, synchronous-capture, Operator UI
bootstrap/deployment/live-adapter, and on-demand collector-protocol paths remain
full-risk in the classifier. On a normal PR they use `pr_fast`; apply the
`ci:full-forecasting` label when the complete suite is wanted. Two or more
focused product tiers and new paths follow the same explicit opt-in rule.
