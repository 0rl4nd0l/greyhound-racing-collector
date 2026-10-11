# Original live forecast alignment audit

The retained July shadow stream contains 277 original predictions. The allocation audit admits 212 records (210 races) and excludes 65 before prediction-body decoding. All 212 admitted records authenticate against retained native append reports, original stdout, complete contemporaneous WIN captures and six original input-file hashes. The artifact is the model now installed in R3, SHA-256 `624bba020d24f93fac4d895a851195aed5d31cff2f35645d9253be1175cc694d`; its historical July role was shadow, not proof of July production deployment.

After outcome-blind latest-per-race selection, 203 races have no official result in the authorised exact-identity projection. Seven have an official winner but incomplete final fields: four have only four finishers, and three have seven finishers against eight original runners. Two older forecasts remain secondary horizons. **Zero races establish the complete unchanged final field required by the strict comparison.** The seven winner-joined races remain explicitly diagnostic, preserving all original runners and probabilities without renormalisation. Full and half residual variants were both genuinely written in each original record.

## Evidence and reproducible derivation

Evidence root `B` is `/mnt/tenn-nvme2/tenn/greyhound-live-benchmark-20261011-evidence`. Original live source root `E` is `/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-autonomous-accuracy-odds-v1-20260610/artifacts/full_evidence_orchestration_20260525`.

- `B/inventory/residual-membership.json` makes the allocation decision for every original line; `access-boundary.json` binds reserved race identities, exact venue aliases, reservation windows and the 169 withheld races. No reserved history row is decoded by the adapter.
- `E/market_form_residual_shadow_predictions_v1.jsonl`, SHA-256 `276ecd6fa2e87c4d395573374a59e2cdd8e724e16c1bc8d721c3b5866a9d0d27`, retains schema-v1 original forecasts. Native runner-set hashing uses sorted newline-separated runner IDs; native record identity uses canonical JSON plus newline. The adapter checks the original conventions instead of migrating or rescoring them.
- `B/alignment/authenticated-originals.json` binds every admitted line, record key, original probabilities and append proof. Native daemon `early_residual_shadow_status.json` reports `APPENDED`/`EXACT_REPLAY`, original stdout, zero exit status and completion before jump. The historical append implementation flushes and fsyncs before returning. This is retained execution evidence; it is not an external signature or independently witnessed clock.
- The native prediction command binds form CSV, sidecar, feature rows, feature manifest, implementation manifest and capture bytes. Every file matches its original `input_hashes`. WIN identity/box/odds exactly match the original prediction field; PLACE quotes are never substituted. Complete original fields are checked against `active_expected_runner_count` and native missing/extra/duplicate guards.
- `B/official-result-projection.json` comes from exact allowlisted SQLite queries. Every joined database `row_json` is authenticated against its original retained `official_result_races.jsonl` or `official_result_runners.jsonl` row. The adapter also checks projected columns agree with those retained rows. Each source-file reference carries SHA-256.
- `B/alignment/strict-records.json` is empty. `diagnostic-records.json` contains seven races for each of the two original variants. `forecast-records.json` contains all 210 primary full-strength forecasts without fabricated winners, including retained feature quality summaries. `original-exclusions.json` and `join-dispositions.json` preserve the full denominator.

```bash
python3 scripts/live_benchmark_alignment.py \
  --membership "$B/inventory/residual-membership.json" \
  --evidence-root "$E" --output "$B/alignment" \
  --results "$B/official-result-projection.json"
```

The root benchmark report supplies reproducible membership/result-projection generation and score calculation. This adapter neither fits nor regenerates a prediction, reads a provider, nor writes source records.

## Exact repair and remaining gaps

Traralgon R9 on July 17 has two retained official snapshots from the same native URL: a top-four result followed by seven finishers. Their winner and overlapping ordered boxes agree. The adapter coalesces this compatible prefix as an explicit derived correction and retains both references. It does not discard a conflicting result or infer an eighth runner. This repairs an overly strict one-row-only join, while leaving final-field eligibility unresolved.

Native result `captured_at` is a batch-generation timestamp, preceding actual per-race fetching. `result_at` uses retained successful attempt completion where present and labels its semantics. Original batch timestamps remain in the retained projection. Feature freeze, quote capture, forecast calculation and durable append completion are separate fields. Provider quote publication time is absent; capture age is not represented as provider quote age.

The three seven-finisher records omit Darra Topaz (Bendigo R12), Notorious Milo (Geelong R6) and Crisp (Traralgon R9), all July 17. Missing finishers cannot be classified as scratches, substitutions or nonfinishers from these rows alone. The four top-four records similarly cannot prove an unchanged complete field. Preserving an official winner permits a diagnostic score; it does not repair that missing evidence. No result-generated field or later price is used.

The separately indexed July 31 manual bundles for Murray Bridge R11, The Gardens R11 and Bendigo R1 are nonreserved and original ready forecasts of the same artifact. Their complete native bundle manifests, original runner probabilities, WIN receipts and capture fields authenticate. Their recorded calculation timestamps precede jump, but the old bundle schema supplies no durable-completion timestamp. The request timestamp and directory name describe invocation start and cannot substitute for completion. The exact result projection also finds no corresponding retained official results. `manual-forecast-records.json` preserves all three original forecasts with `sealed_at: null`, `verified: false` and both exclusions; `manual-qualification-dispositions.json` records the qualification. They do not enlarge either scorecard or silently enter the fully verified forecast census.

Reproduce this separate projection by adding `--manual-census "$B/inventory/manual-census.json" --manual-results "$B/manual-official-result-projection.json"` to the adapter command above. No stored historical database is decoded: the manifest checks those bytes by hash only.

## Checks

Seven focused synthetic tests cover native record identity corruption, changed market odds, post-jump prediction, duplicate runner boxes, partial final fields, result-projection tampering and conflicting result snapshots. Run with the project's pinned Python and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -o addopts= --noconftest -p no:cacheprovider tests/test_live_benchmark_alignment.py` (seven passed). Independent review separately verified the July 19 Healesville R4 capture, all six original input hashes, exact WIN field and pre-jump append chronology; its report is `B/review/independent-july19-source-sample.json`.
