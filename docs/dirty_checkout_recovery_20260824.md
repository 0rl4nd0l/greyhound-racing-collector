<!-- markdownlint-disable MD013 -->

# Dirty checkout recovery — 2026-08-24

Status: `PRESERVATION_ONLY_DRAFT`

This branch preserves the narrowly authorized unique material from the detached
dirty checkout at:

`/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector-ci-routing-fix`

It does not activate or normalize research code, install configuration, alter
promotion policy, regenerate evidence, deploy services, or reintroduce the
superseded TheDogs/current-index/forward-baseline implementations.

## Identity and authority

| Item                | Value                                                                       |
| ------------------- | --------------------------------------------------------------------------- |
| Recovery base       | `d41d3c710cf9493ba8ccd44ccfcede2a0264b527`                                  |
| Recovery base tree  | `a7f94ac70754a31d79e79a15346ff2b8c03e0d9b`                                  |
| Dirty checkout HEAD | `779761165637b709227d965f6c9be7e80706d23f`                                  |
| Dirty checkout tree | `a601f9c1a941c15dfeec4e300f7adbade5440bc2`                                  |
| Audit               | sibling `greyhound_racing_collector-ci-routing-fix-dirty-audit-20260824.md` |
| Recovery scope      | 17 `RECOVER_TO_FRESH_BRANCH` paths plus quarantined review material         |

The network `refs/heads/master` and local `origin/master` both resolved to the
recovery base immediately before the detached sibling worktree was created.

## Recovered paths

All 17 paths were absent from the recovery base and copied byte-for-byte to
their original relative locations. They remain report-only preservation
material; presence on this branch is not adoption or runtime authority.

| Original and recovered path                                                  | SHA-256                                                            | Provenance                                                                               |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| `docs/betfair_anz_greyhound_historical_csv_primary_source_audit_20260817.md` | `6445c6fded97cb7e83f0b5463a7cee39f0d150f02a04322abb231207f8777e89` | Official Betfair historical-surface source audit; terminal partial report-only evidence  |
| `docs/prejump_incremental_data_source_acquisition_plan_20260817.md`          | `2225fd81659d61108ec6a7e0822687bbe9938b27d6b5dd2b99d9fb718ef39eff` | Report-only source plan; no collection authority                                         |
| `scripts/audit_betfair_historical_surface.py`                                | `ff0e32bc3e91fe7c0ff41674addae6ab98c9dc0c53d862fc07a6a36a88756660` | Reproducibility code for the historical-source audit                                     |
| `scripts/audit_fast_nonfavourite_mechanism.py`                               | `d6f3fc9b5fc4591ef46c5ba13a39f1450b6720199ad0c1779ed8a4b2cfb52959` | Reproducibility code for a later-superseded coverage-blocked precursor                   |
| `scripts/audit_pace_topology_mechanism.py`                                   | `2b84220609d1b71acbd2c73f07cc3cf628c4d6f4900371b74c3ba2983a27531b` | Reproducibility code for `NO_PACE_TOPOLOGY_SIGNAL`                                       |
| `scripts/audit_sportsbet_win_market_surface.py`                              | `f4f4af166adc8bc2c3e282fc1d8ad985dc4e6c0295b53dd3df8cfa66a451f720` | Corrected Sportsbet WIN report-only audit                                                |
| `scripts/build_favourite_benchmark_report.py`                                | `e79d576e97522fdf6de7ba3f226969d61d6e93365c100ec723adcf7d0fbb37ab` | Favourite-benchmark report generator; not betting authority                              |
| `scripts/rerun_corrected_sportsbet_win_experiments.py`                       | `de8b11c220f6a6f48527d9ec4a76f4a22123e5922f0a44326100bc19d5978ff8` | Frozen corrected historical rerun implementation; not rerun here                         |
| `scripts/run_form_speed_market_residual_experiment.py`                       | `bfdc075b26a29bcafdd86a47886c77ca4c997734aca36b68214422bb5e89bcb1` | Reproducibility code for `NO_INCREMENTAL_FORM_SPEED_SIGNAL`; not run here                |
| `scripts/thedogs_effective_box_extension_state.py`                           | `91ed765f972f8b0d3deea84b7953c367fac88966f331c6e6e36f5d3dba147274` | Report-local resume/state helper; terminal `DATA_MISSING`, not live collection authority |
| `tests/test_audit_betfair_historical_surface.py`                             | `aadc678b3c0b61b2e48c98a7e53507b9712cfc387f90ebc243a7963fd90f3352` | Preservation test for the matching audit                                                 |
| `tests/test_audit_fast_nonfavourite_mechanism.py`                            | `eca8a256babb380d3599ef8fad98afcd648478c9ec13065fc7614073749f9af4` | Preservation test for the matching audit                                                 |
| `tests/test_audit_pace_topology_mechanism.py`                                | `730cdad3d7efeacbcf838982741fbbc414a10d315bac6943772e562abfddda9e` | Preservation test for the matching audit                                                 |
| `tests/test_audit_sportsbet_win_market_surface.py`                           | `85e05be306ff6f8c3fd982414997ac89e852fb24ccf00657a4d0a57cbfadbbbf` | Preservation test for the matching audit                                                 |
| `tests/test_rerun_corrected_sportsbet_win_experiments.py`                    | `effc7d89a281c249bada715cf279a87b3194c1cc250438fb56c8f0daa992efb8` | Preservation test for the frozen historical rerun                                        |
| `tests/test_run_form_speed_market_residual_experiment.py`                    | `8eadda73ec4dad64a964a84bb75f06f49f1072e3ba3c3514450bb5f564744762` | Preservation test for the negative residual experiment                                   |
| `tests/test_thedogs_effective_box_extension_state.py`                        | `798f38d231add55d354892095180ebb6aa2b5481066e6adb55866f257be5644b` | Preservation test for the report-local state helper                                      |

## Quarantine

Six additional files are copied under
`docs/dirty_checkout_quarantine_20260824/original_paths/`. Their original
relative paths are retained beneath that prefix, so none can shadow a canonical
runtime/configuration path.

The quarantine contains the four dirty `NEEDS_OWNER_REVIEW` items, the ignored
high-risk registry configuration, and the unique Betfair-teacher narrative that
the audit explicitly requires preserving as scientifically quarantined
diagnostic evidence.

See [the quarantine record](dirty_checkout_quarantine_20260824/README.md) and
its `SHA256SUMS`. Quarantined content must not be imported, installed, executed,
or treated as policy.

## Registry configuration disposition

Original path: `model_registry/registry_config.json`

SHA-256: `6fb519c3e1cbd7d7000ba2083d0dbf56d2c61f5849057d67fa670495bae4e47c`

Disposition: `GENERATED_DEFAULT_DEBRIS_QUARANTINED_INERT`

Evidence:

- The path is ignored by the repository-wide `*.json` rule and has no Git
  history.
- Its bytes match the default mapping emitted by `ModelRegistry._load_registry`
  and `_save_config` on current master, including
  `"min_races_for_promotion": 0`.
- Its creation timestamp is `2026-08-15 18:07:21.876923436 +1000`.
- Existing local logs record model-registry initialization at
  `18:07:21.823` and successful initialization with zero models at
  `18:07:21.879`.

This proves an application-generated default, not an intentional local policy
override. The unsafe default is not copied to `model_registry/`, is not loaded,
and is not changed by this branch.

## Explicit exclusions

The following were not copied:

- all 11 `SUPERSEDED` paths;
- `AGENTS.md`, classified safe to discard because an exact sibling duplicate
  exists;
- caches, standalone logs, and `uv.lock`;
- current-index, runtime, service, model, database, result, and betting state.

The audit's broader 61-path `KEEP` set and 247 ignored report/evidence files were
not part of the narrowed copy authorization, except for the quarantined
Betfair-teacher narrative. Their bytes and checksum manifests remain unchanged
in the dirty checkout.

## Integrity proof

Before recovery, the dirty checkout matched the audit on:

| Surface                       | Value                                                              |
| ----------------------------- | ------------------------------------------------------------------ |
| status digest                 | `22341deb0038bf27b3b2d2d921dce3fa67e8a1193e2b669a347f5c4c798dfc59` |
| 94-path dirty manifest digest | `6d6d49d84ab059f86938e1b270b5042d62f86ec12c14035be8dbb30e242c8f39` |
| index digest                  | `0e1b99b41474d2fb2585c66f19ace95778040381ef99ab497553b969c75ec07d` |
| HEAD control-file digest      | `d480b140a50a52ce0a9bac7ce71722bb7086eac455855ca46b4d1b9343db628e` |

The shared `git show-ref` digest had already changed from the audit's
`2211aaec...` to `64f08a4a...` before recovery started, while master and the
dirty checkout identity remained unchanged. Recovery therefore uses
`64f08a4a...` as its before-ref baseline and does not rewrite shared refs.

## Cleanup status

`NOT_YET_SAFE_TO_RESET_OR_DELETE`

The authorized recovery/quarantine set is durable on this branch, but the
broader audit `KEEP` evidence remains only in the dirty checkout under the
scope applied here. A separate cleanup task may now use this recovery PR and
the external audit, but must preserve or explicitly disposition those remaining
`KEEP` and ignored report/evidence bytes before destructive cleanup.
