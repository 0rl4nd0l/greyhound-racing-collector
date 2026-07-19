# State

- Before: receipt reuse and artifact scoring existed as separate dependency
  lanes, while the artifact scorer required prebuilt form, features, and odds.
- After: one command resolves one exact named pre-jump race, reuses a fresh
  verified receipt or performs an immediate isolated capture, seals historical
  rows before the target date, selects an exact model/config, writes a sealed
  research bundle, and replays it deterministically.
- Publication: implementation commit `76f17dbeec78a43e5493a8049ff84c47a13d3e8f`;
  draft PR pending creation by the authorized normal push step.
- Runtime: fixture-only. No live prediction, browser capture, daemon action,
  timer wait, service action, production database write, or shadow append ran.
- Dependency stop: PR #46 is `BLOCKED_DO_NOT_MERGE`; this draft must remain
  non-mergeable until that separate dependency is repaired and reviewed.
