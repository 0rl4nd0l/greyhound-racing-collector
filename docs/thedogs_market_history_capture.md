# Prospective TheDogs market-history capture

This path records point-in-time TheDogs `/odds` evidence for future,
development-only research. It does not train, score, promote, emit EV, or place
bets. The source class is
`thedogs_prospective_point_in_time_market_history`; it is not Sportsbet
evidence and it is separate from historical published OPEN/LOW/HIGH summaries.

## Snapshot contract

Each plan names one exact HTTPS URL of the form:

`https://www.thedogs.com.au/racing/<venue>/<YYYY-MM-DD>/<race-number>/<race-name>/odds`

Queries, fragments, redirects, name-free numeric URLs, other hosts, and other
paths fail closed. The collector warms the date meeting page and the exact race
page, requires the plan's declared active native-ID set to match the page,
requests the exact `/odds` page once, extracts the identified runner IDs and
active/scratched field (excluding only a source-labelled vacant-box slot), and
requests TheDogs' native runner-odds JSON once.
There is no retry, discovery, race substitution, or provider substitution.

An accepted snapshot contains:

- externally recorded UTC request start/end times and strict
  `capture_end_utc < jump_timestamp`;
- the exact source/final URLs, status and selected HTTP response metadata;
- raw `/odds` HTML plus SHA-256;
- base64-preserved raw jump-page and runner-odds API bytes plus SHA-256;
- the exact jump-source URL, timestamp, native race identity, and source hash;
- every identified page runner's native ID, box, scratch state, and every
  active runner's current point-in-time fixed-WIN price;
- the provider declared in the source API, or `provider_unknown` if the source
  does not declare one; and
- one receipt-core hash and immutable read-only raw/receipt files.

The page's OPEN/LOW/HIGH values are extrema summaries. They are retained only
in the raw source and never counted as ordered or temporal observations.

## Effective boxes for reserves and replacements

Runner identity remains the numeric native ID from the exact odds page and the
same ID in the runner-odds API. A name, suffix, price, display order, or nearby
row never resolves a box.

The collector preserves four distinct box fields in each runner projection:

- `box` and `page_box`: the original page rug from `sprite-svg name="rug_N"`;
- `page_effective_box`: the optional explicit
  `race-runners__name__box` text `(from box N)`;
- `api_run_box`: the unchanged `run_box` value from the native runner's sole
  fixed-win API quote; and
- `effective_box`: the resolved active starting box, or `null` for an inactive
  runner.

For a normal active runner, page rug and API `run_box` must match. For a
replacement, the page must explicitly say `(from box N)` and that value must
match the same native runner's API `run_box`. Effective boxes must be unique
across active runners. A scratched original may retain the same API `run_box`
as its active replacement, but it has no active `effective_box`.

Missing or malformed explicit box text, multiple mappings, non-unique native
IDs, page/API disagreement, invalid API boxes, active/scratch price conflicts,
or duplicate active effective boxes fail closed. Raw HTML and the full native
API body remain receipt-bound, so the independent auditor recomputes the rule
instead of trusting the stored projection.

Existing raw/receipt pairs are validated and returned as an idempotent skip.
A partial pair, changed raw bytes, mismatched plan identity, or changed receipt
fails as conflicting evidence. The collector never overwrites a snapshot.

## Fixed windows and audit semantics

The only nominal windows are `T-120`, `T-60`, `T-30`, `T-10`, and `T-2`.
The default due interval is 30 seconds early through 90 seconds late. Actual
request and capture times are always recorded. A run outside that interval is
missed; it is not reassigned or interpolated.

The auditor counts one accepted complete snapshot as one temporal observation,
not one observation per runner. Duplicate references to identical evidence do
not increase depth. A race becomes trajectory-ready only when all five windows
have distinct, complete, strictly pre-jump snapshots with no conflict or
rejection.

Capture one due item:

```bash
python3 scripts/capture_thedogs_market_history.py plan.json \
  --output-dir reports/agent_jobs/<job>/capture/<race>/T-120
```

Audit an immutable manifest:

```bash
python3 scripts/audit_thedogs_market_history_snapshots.py manifest.json \
  --output-dir reports/agent_jobs/<job>/audit
```
