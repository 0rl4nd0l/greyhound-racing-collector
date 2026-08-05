# GHU-054 manual research prediction adapter

`src.predictor.manual_research_cli` exposes one bounded, offline invocation of
the accepted GHU-053 scorer. It reads only caller-supplied regular files and
directories, then delegates sealed-evidence verification, form validation,
scoring, atomic publication, and output verification to GHU-052/GHU-053.

The CLI requires absolute paths for:

- the sealed GHU-052 bundle and its isolated run directory;
- canonical JSON files for `SealExpectations`, `SealingIdentity`, and
  `ResearchScoringIdentity` (their keys must exactly match the corresponding
  Python dataclasses);
- the embedded form, frozen model, model manifest, and scoring config;
- the caller-owned isolated output root.

The four `--*-sha256` arguments are required and are checked against the bytes
read from the supplied form, model, model manifest, and config. No input is
discovered, substituted, retried, or fetched.

Example:

```bash
python3 -m src.predictor.manual_research_cli \
  --sealed-bundle-dir /isolated/run/sealed-evidence/<bundle-id> \
  --run-dir /isolated/run \
  --evidence-expectations /isolated/args/evidence-expectations.json \
  --sealing-identity /isolated/args/sealing-identity.json \
  --embedded-form /isolated/args/form.json \
  --form-sha256 <sha256> \
  --model /isolated/args/model.json \
  --model-manifest /isolated/args/model-manifest.json \
  --model-sha256 <sha256> \
  --model-manifest-sha256 <sha256> \
  --config /isolated/args/config.json \
  --config-sha256 <sha256> \
  --scoring-identity /isolated/args/scoring-identity.json \
  --output-root /isolated/output
```

Success writes one canonical JSON envelope to stdout and returns exit code 0:

```json
{"schema_version":"manual_research_invocation_response_v1","status":"SUCCESS","bundle_id":"<bundle-id>","bundle_path":"/isolated/output/<bundle-id>","verification_status":"VERIFIED"}
```

Failure writes only the same envelope schema with `status: ERROR` and a stable
`error_code`, and returns exit code 2. The response never exposes prediction,
EV, staking, betting, outcome, result, or Phase-7 fields. This adapter does not
provide browser, network, capture, service, timer, database, canonical, live,
training, calibration, promotion, deployment, or UI authority.
