- Retained the GHU-051 parent executor as the sole process, lock, timeout,
  identity, runner, odds, timing, and artifact authority.
- Added only a versioned live child envelope and a parent-side expected runner
  binding; the fixture child protocol remains accepted unchanged.
- Added one live JSON media/parser identity to GHU-052 without changing its
  sealing, hash, timestamp, or outcome-rejection rules.
- Kept the GHU-056 package and service default-off; explicit race input is
  required and no service mode performs discovery or capture.
