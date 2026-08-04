# Forecasting CI tiers

The stable pull-request check is `tests-race-collection`. A classifier selects
one tier for the complete change set; unknown paths, destructive changes, shared
surfaces, and incompatible tier combinations fail closed to
`full_forecasting`.

| Tier | Intended paths | Validation command |
| --- | --- | --- |
| `ci_contract` | `.github/**`, `scripts/ci/**`, `tests/ci/**` | `uv run --no-project --with-requirements requirements/all.in --with PyYAML python scripts/ci/run_forecasting_ci_contract.py` |
| `manual_prediction` | isolated manual-capture contracts and market-residual scoring | `uv run --no-project --with-requirements requirements/all.in python -m pytest -q --noconftest tests/test_manual_independent_capture.py tests/test_market_form_residual.py tests/test_market_form_residual_portability.py tests/test_predict_market_form_residual.py` |
| `official_results` | official-result fetch, parse, and label reporting | `uv run --no-project --with-requirements requirements/all.in python -m pytest -q tests/test_results_ingest_official_first.py tests/test_autonomous_official_result_capture.py tests/test_expert_form_official_result_labels_packet.py` |
| `forward_corpus` | scheduled producer, observer, admission, and closure | `uv run --no-project --with-requirements requirements/all.in python -m pytest -q --noconftest tests/race_collection/test_scheduled_forward_corpus.py tests/race_collection/test_forward_official_result_observer.py tests/race_collection/test_phase7_source_admission.py tests/race_collection/test_forward_sealed_corpus.py` |
| `operator_ui` | isolated foundation, job-store, worker, security, R3, and connected-UI paths | `uv run --no-project --with-requirements requirements/all.in python -m pytest -q tests/operator_ui/test_foundation.py tests/operator_ui/test_job_store.py tests/operator_ui/test_prediction_worker.py tests/operator_ui/test_security.py tests/operator_ui/test_r3_api.py tests/operator_ui/test_r3_e2e_safety.py tests/operator_ui/test_connected_ui.py` |
| `forecasting_core` | isolated phase-3 forecasting and phase-4 serving | `uv run --no-project --with-requirements requirements/all.in python -m pytest -q --noconftest tests/race_collection/test_phase3_forecasting.py tests/race_collection/test_phase4_model_serving.py` |
| `full_forecasting` | shared/high-risk, unknown, destructive, or unsafe mixed changes | `uv run --no-project --with-requirements requirements/all.in --with PyYAML python scripts/ci/run_full_forecasting.py` |
| `non_forecasting` | known unrelated documentation and presentation paths | no Forecasting test command |

`ci_contract` validates the classifier contracts and all workflow YAML files,
then runs one named smoke from every focused tier. It never invokes the complete
`tests/race_collection` directory.

The full runner executes the CI contract, the complete `tests/race_collection`
directory, and the top-level focused suites. It runs weekly, on every workflow dispatch, for PRs
labeled `ci:full-forecasting`, and whenever the classifier escalates risk.

Shared identity, timing, provenance, feature, schema, dependency, training,
evaluation, promotion, runtime-adapter, synchronous-capture, Operator UI
bootstrap/deployment/live-adapter, and on-demand collector-protocol paths remain
full. New paths are full until explicitly and safely assigned.
