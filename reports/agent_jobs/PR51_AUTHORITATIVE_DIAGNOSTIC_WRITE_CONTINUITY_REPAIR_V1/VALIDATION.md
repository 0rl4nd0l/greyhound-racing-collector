# Validation

- compile: PASS
- Ruff: PASS
- FORM_ONLY_V1 Python 3.11: 177 passed
- FORM_ONLY_V1 Python 3.13: 177 passed
- continuity/isolation targeted selection: 93 passed on each 3.11 and 3.13
- terminal continuity probes: 15 passed; duplicate-fd/staged-fstat/close-error
  selection: 6 passed
- independent review: PASS; no critical findings or warnings
- focused coverage: 1,665/2,016 statements (82.59%); 610/876 branches (69.63%)
- deterministic authoritative+diagnostic and authoritative-only builds: PASS
- A/B authoritative diff: PASS
- physical inventories/declarations: 10/2/6/3, all regular single-link files
- contract hash verification: PASS for every declared artifact
- builder SHA-256: `cf20ffc0...11359` -> `0136e475...dd52b`
- reproducibility descriptor SHA-256 unchanged: `8fa8966e...89ad`
- authoritative aggregates unchanged: trainer `97967ab3...4e31`, control
  `1712d3d6...5462`, sealed `3b1284f3...691c`, diagnostic `bb6306bc...15b2`

Repository-wide Python 3.11 suite (fresh repair tree): 1,974 total; 1,842
passed; 61 failed; 50 skipped; 21 errors; 23.56% repository coverage. Exact
99be6c baseline: 1,953 total; 1,820 passed; 62 failed; 50 skipped; 21 errors.
Delta is +21 tests, +22 passes, -1 failure, and no error/skip transition; the
extra tests are this repair's continuity tests and no repair-attributable broad
regression was identified. Broad failures are pre-existing/environmental debt
(missing samples, unavailable Selenium/network, legacy fixtures and baseline
integration assumptions).
