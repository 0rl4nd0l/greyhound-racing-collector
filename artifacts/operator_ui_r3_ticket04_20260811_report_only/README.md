# Operator UI R3 Ticket 04 live prediction report

Terminal status: `DATA_MISSING`

One read-only preflight was performed at `2026-08-11T16:16:00+10:00` against
the fixed collector-owned v2 current-race index. The deployed bounded verifier
rejected the evidence chain with `CURRENT_INDEX_REPORT_INVALID`. Therefore no
race was admitted and receipt matching was not attempted.

The deployed R3 submission contract also does not satisfy Ticket 04's
receipt-only requirement: it allowlists `odds_source_id=auto` and constructs an
`auto` job input, while Ticket 04 requires `odds_source=receipt`. No request was
submitted through that divergent contract.

The run stopped without retry, alternate race, prediction submission, capture,
external fetch, Ticket 04 database mutation, model change, EV, or betting. The
live canonical database changed size and mtime concurrently between final
read-only observations while the odds collector remained active; this workflow
never opened the database or initiated that activity. The separate R3 audit and
job stores remained byte-identical. No prediction or immutable result bundle
was produced.

Final observed service state:

- `greyhound-operator-ui-r3.service`: `active/running`, PID `2406787`.
- `shadow-autopilot.service`: `failed/failed`, PID `1398347`.
- `shadow-autopilot-odds-capture.service`: `activating/start`, PID `1426259`.

See `evidence.json` for the exact identities and retained evidence locations.
