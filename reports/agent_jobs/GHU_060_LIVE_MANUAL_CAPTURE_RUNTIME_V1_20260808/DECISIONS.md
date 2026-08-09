- Retained the GHU-051 parent executor as the sole process, lock, timeout,
  identity, runner, odds, timing, and artifact authority.
- Added only a versioned live child envelope and a parent-side expected runner
  binding; the fixture child protocol remains accepted unchanged.
- Added one live JSON media/parser identity to GHU-052 without changing its
  sealing, hash, timestamp, or outcome-rejection rules.
- Kept the GHU-056 package and service default-off; explicit race input is
  required and no service mode performs discovery or capture.
- Replaced the live child navigation blacklist with a pure allowlist policy:
  exact canonical race `GET` document navigation, or query-free `GET` static
  assets on the exact canonical host under `/assets/` with reviewed stylesheet,
  script, image, or font extensions.
- Did not admit XHR, fetch, websocket, event-stream, API, result-like,
  subframe, unknown, or unclassified requests; repository evidence did not
  prove a safe odds API endpoint.
- Preserved one exact `goto`, GHU-051 executor/process/timeout/runner binding,
  GHU-052 media/parser/sealing semantics, and production permission safety.
