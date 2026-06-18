# Weather/Track Review Board

## Decision

- Board decision: `park`
- Packet final status: `KEEP_COLLECTING_ONLY_DATA_MISSING`
- Reason: `source-safe weather/track evidence exists, but train/holdout coverage or non-flat utility evidence is not sufficient`

## Perspectives

- Architect: keep report-only boundaries; do not activate inactive features without train/holdout coverage.
- Skeptic/red-team: source-safe both weather+track coverage is the hard gate; partial weather-only rows are not enough.
- Product/value: ablation is only valuable if it can distinguish signal from missing source plumbing.
- Validation/test: require protected hashes unchanged and focused unit tests around leakage and coverage gates.
- Repo hygiene/git guard: tracked greyhound code is the active lane; unrelated Tenn dirty branch is out of scope.
- Domain: Sportsbet/open-meteo sidecars are acceptable only with pre-jump timestamps and non-result URLs.

## Evidence

- Sidecars scanned: `14769`
- Accepted both-weather-track races: `120`
- Accepted both-weather-track runner-row pct: `0.13795278852763104`
