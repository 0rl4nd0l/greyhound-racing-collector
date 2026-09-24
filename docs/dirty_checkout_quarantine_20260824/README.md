<!-- markdownlint-disable MD013 -->

# Dirty-checkout quarantine — 2026-08-24

Status: `INERT_OWNER_REVIEW_ONLY`

Files below are byte-for-byte copies from the audited dirty checkout. Each is
nested beneath `original_paths/` so it cannot occupy or override its original
runtime/documentation path.

Nothing in this directory is activated, normalized, executed, imported, or
treated as model/promotion policy.

| Original path                                                         | Quarantine path                                                                      | SHA-256                                                            | Classification and reason                                                                                  |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `MISSION.md`                                                          | `original_paths/MISSION.md`                                                          | `ec7f075dd867ac1c221aa2b04f90806909163e35f6d5d82761d1f3013b158466` | `GENERATED / NEEDS_OWNER_REVIEW`; teaching-bundle purpose is visible but ownership is unproven             |
| `RESOURCES.md`                                                        | `original_paths/RESOURCES.md`                                                        | `d44893f630da46f7c9a161e7b3fec6c3e38e8b28ab85356e01f41433d73cdbbc` | `GENERATED / NEEDS_OWNER_REVIEW`; teaching-bundle purpose is visible but ownership is unproven             |
| `assets/course.css`                                                   | `original_paths/assets/course.css`                                                   | `c21126a8bf88b2c89e1d9a6cd6d5c455a07e1a005593014588fd5acaf8c28447` | `GENERATED / NEEDS_OWNER_REVIEW`; optional presentation asset                                              |
| `docs/research/greyhound_prediction_feature_gap_research_20260819.md` | `original_paths/docs/research/greyhound_prediction_feature_gap_research_20260819.md` | `42e433e33d86c1910dc146bcbd1b3c5d448b78c752ca939b2d2cf6c956c8e19d` | `HIGH_RISK_UNIQUE`; substantive note, but no task card/commit proves owner                                 |
| `docs/betfair_teacher_distillation_20260818.md`                       | `original_paths/docs/betfair_teacher_distillation_20260818.md`                       | `909300d21b71254a127816a440f6071f6bff3d62f14404d1727ecd3e1a88ccd2` | `REPORT_ONLY / SCIENTIFIC_QUARANTINE`; historical diagnostics only, not a forward-scoreable model          |
| `model_registry/registry_config.json`                                 | `original_paths/model_registry/registry_config.json`                                 | `6fb519c3e1cbd7d7000ba2083d0dbf56d2c61f5849057d67fa670495bae4e47c` | `HIGH_RISK / GENERATED_DEFAULT_DEBRIS`; contains zero minimum-race default and must never become effective |

## Owner-review decisions

- The feature-gap note may be recovered later only after ownership, sources,
  and desired canonical location are reviewed.
- The teaching bundle may be kept or removed as one optional unit.
- The Betfair-teacher narrative must remain diagnostic/quarantined unless a
  separately authorized, outcome-inaccessible protocol is created from scratch.
- The registry configuration is stale generated debris. It should not be
  restored to its original path, installed, or used to change promotion policy.
