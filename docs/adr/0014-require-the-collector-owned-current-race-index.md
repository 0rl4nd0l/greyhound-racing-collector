# Require the collector-owned current-race index

A genuine report-only live manual prediction requires a fresh, verified collector v2 current-race index containing the selected pre-jump race. Its race identity, jump timestamp, and runner-set hash must agree with the sealed receipt. The index must not be manually edited, synthesized, copied into place, or bypassed; if publication is unavailable or stale, prediction waits or fails closed.

The operational index is published only by a completed primary collector
refresh and contains any non-empty bounded set of individually valid races.
Odds-only refreshes never publish it. Scientific cohort composition is a
separate downstream freeze gate: the forward baseline still requires exactly
20 races spanning at least three venues and two racing dates, but those
thresholds never determine operational-index validity.
