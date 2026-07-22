# Validation

- Focused: `35 passed` in `tests/test_predict_race_now.py`.
- Relevant regression set: `560 passed, 1 skipped, 4 deselected`.
- The four separated cases are two current-master weather assertions reproduced
  unchanged at exact master and two optional Flask dashboard tests whose host
  environment lacks Flask; none touches the successor delta.
- Ruff check: pass.
- Ruff format check: pass.
- Python 3.11 compile: pass through `uv run --python 3.11 --no-project`.
- `--list-configs`: pass, resolving market-form-residual and market-only modes
  with canonical config/schema/model hashes.
- Diff check: pass; the branch adds only the V2 task card, eight required
  implementation files, and this report bundle.
- Live receipt-only proof: exact Bendigo R5 resolved pre-jump; existing evidence
  was rejected as `RECEIPT_INVALID` because target-grade context was missing.
  The isolated result reports `production_persisted: false`; the sentinel
  database path was never created and direct capture was not attempted.

## Runtime Functionality Proof

- Intended output: one immediate research-only prediction for one exact named
  pre-jump race, written only as a canonical isolated bundle.
- Live output location: `/tmp/greyhound-pr53-successor-live-proof-20260722/prediction_20260722T145904596748+1000_8f670957eed5`.
- Pre-run max timestamp or count: sentinel database absent; production rows
  written by this run `0`.
- Post-run max timestamp or count: sentinel database absent; production rows
  written by this run `0`; isolated bundle files `5`.
- Rows/files inserted or updated after run start: `0` database rows, `0`
  production files, `5` isolated bundle files.
- Readiness/gate status: `DATA_MISSING` because the available receipt lacked the
  mandatory PR #56 target-grade context schema.
- Exact command/query used: `uv run --no-project scripts/predict_race_now.py --race "bendigo r5" --model market-only --config configs/prediction/market-only.json --odds-source receipt --db /tmp/greyhound-pr53-successor-no-db.sqlite --output-root /tmp/greyhound-pr53-successor-live-proof-20260722 --capture-evidence-root /mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525`.
- Result: `DATA_MISSING`.
- Remaining blocker: obtain a current sealed pre-jump packet whose sidecar
  carries the PR #56 target-grade context schema, then run the same receipt-only
  command for that eligible race. No retry is part of this draft publication.
