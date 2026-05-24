# Contract Validation (V4)

This guide explains how to validate your MLSystemV4 feature contract against the currently loaded model, either via the REST API or via a Python utility script suitable for CI pipelines.

What is validated
- Feature signature matches (hash over model feature columns)
- Categorical and numerical column sets (missing/extra differences)

UI-based validation (recommended)

Prerequisites
- App is running (default PORT=5002)
- Open the ML Dashboard:
  - http://localhost:5002/ml-dashboard (also available as /ml_dashboard)

Steps
1) Open the “V4 Feature Contracts & Validation” panel

   ![Feature contracts panel](images/ml_dashboard_feature_contracts_overview.png)

2) Click “Refresh Contract” to regenerate the saved contract from the currently loaded model
3) Toggle “Strict” if you want HTTP 409 on mismatch (leave off to see diffs without failing)
4) Click “Validate” to run the check. The result shows:
   - Status: Matched / Mismatch (Strict) / Mismatched
   - Signature: Match/Mismatch with expected/current hashes
   - Differences: Categorical and Numerical → Missing/Extra

   ![Validation details](images/ml_dashboard_feature_contracts_violation.png)

5) Read the status badge:
   - Stable (green): matches the saved contract
   - Diffs (yellow): differences found in non-strict mode
   - Mismatch (red): strict check returned HTTP 409
6) Evaluation summary:
   - Use “Refresh Summary” or change the window to refresh the “Evaluation Summary” box

Note: If images are missing, capture screenshots of the ML Dashboard’s Feature Contracts panel and save them as:
- docs/images/ml_dashboard_feature_contracts_overview.png
- docs/images/ml_dashboard_feature_contracts_violation.png

Troubleshooting
- Strict mismatch (409): Click “Refresh Contract” then re-validate. If it still mismatches, align features or intentionally update the saved contract.
- “No contracts found”: Train/register a model first, or click “Refresh Contract”.
- Dashboard unreachable: Confirm PORT=5002 and try both routes (/ml-dashboard and /ml_dashboard).
- Prefer CLI/CI flows: See the sections below (curl and scripts/verify_feature_contract.py).

Endpoints used by the UI
- POST /api/v4/models/contracts/refresh
- GET /api/v4/models/contracts
- GET /api/v4/models/contracts/v4_feature_contract.json
- GET /api/v4/models/contracts/check?strict=1 (or without strict for a non-strict check)
- GET /api/v4/eval/summary/latest?window=...

API-based validation
- Refresh (regenerate) the contract from the current model:
  curl -sS -X POST http://localhost:5002/api/v4/models/contracts/refresh | jq .

- Get the current contract JSON:
  curl -sS http://localhost:5002/api/v4/models/contracts/v4_feature_contract.json | jq .

- Validate (non-strict, HTTP 200 always; returns matched flag and diff):
  curl -sS "http://localhost:5002/api/v4/models/contracts/check" | jq .

- Validate (strict, HTTP 409 on mismatch):
  curl -sS -i "http://localhost:5002/api/v4/models/contracts/check?strict=1"

Notes
- If your command contains a redacted secret (e.g. ******), replace it with {{SECRET_NAME}} and do not paste the secret inline.
- For secure usage, prefer exporting tokens into environment variables and referencing them from curl with an Authorization header.

Python/CI-based validation
Use the CI guard script (no server required):

- Refresh + strict validation (non-zero exit on mismatch):
  python3 scripts/verify_feature_contract.py --refresh --strict

- Strict validation only (do not regenerate):
  python3 scripts/verify_feature_contract.py --strict

- JSON output (for CI parsers):
  python3 scripts/verify_feature_contract.py --json --strict

API mode (CI against a running server)
- Strict validation via API:
  python3 scripts/verify_feature_contract.py --mode api --url http://localhost:5002 --strict

- Refresh via API then strict validation:
  python3 scripts/verify_feature_contract.py --mode api --url http://localhost:5002 --refresh --strict

Using in CI
- Add a pipeline step that fails on contract mismatches:

  - name: Validate feature contract
    run: |
      python3 scripts/verify_feature_contract.py --refresh --strict --json

Interpretation
- The script prints a diff block showing:
  - signature_match: whether the feature_signature hash matches
  - categorical missing/extra: columns missing/extra vs. saved contract
  - numerical missing/extra: columns missing/extra vs. saved contract
- Exit codes:
  - 0 on success/match (or non-strict diff)
  - 1 on strict mismatch or strict error

Troubleshooting
- "contract not found": If docs/model_contracts/v4_feature_contract.json is missing, use --refresh or the /refresh API to generate it.
- If running in minimal environments, some optional dependencies (like requests) may be unavailable; the script falls back to urllib for API mode.
- To compare specific versions, you can back up the contract file and diff against a newly generated file.

