# Bounded source operating policy — 23 September 2026

This applies only to the existing campaign, not unattended production. The access
interpretation is in collector_sportsbet_access_review_20260923.md. No published
Sportsbet numerical collection quota was located. These are local experimental
ceilings, not claims of a safe or provider-approved rate.

Retained launch-1 network-count.json separates 402 Python logical calls: Sportsbet
15, TheDogs 372, Open-Meteo 15. Two browser navigations complete the campaign's 404.
The earlier 429 occurred during browser collection, not proof that the 15 metadata
calls caused it. Normal legacy traffic and hidden prior retries are absent. CDP
request events overlap navigations and are not added to the logical total.

For a supervised trial, bound both lanes together to at most 10 Sportsbet Python
operations in any 60 seconds, one browser operation in any 60 seconds, and two
explicit navigations per browser operation. A maximum of 512 source operations
is allowed across this continuation, without resets on restart or another launch.
Exceeding a bound persists STOP rather than queueing or hiding a retry. The existing
exclusive source lock, fixed service deadlines and total campaign ceilings remain.

The Python trial ceiling accommodates at most two observed five-request refresh
batches in a minute (15 retained metadata calls over three refreshes). It is a
workload admission envelope derived from that observed implementation, not a
capacity inference. The browser ceiling funds one existing 50-second capture phase
plus its 10-second overhead budget per minute. Two navigations accommodate the
existing landing-to-race path; a route requiring more is an explicit failure.
512 bounds roughly 450 metadata operations over 90 nominal five-request minute
batches plus the remaining 11 captures, with finite headroom for full-lane refresh.
If actual demand differs, report the policy stop and coverage loss; do not raise
limits just to obtain a pass. No provider limit is being replaced by this policy.

The finite trial is the evidence needed to assess this bounded workload, not an
assertion that it is already sustainable. Record operations before transport with
kind/time in the existing locked state, retaining the complete operation list.
Python retries and automatic redirects remain disabled. Browser subrequests are
observed separately: these operation ceilings do not cap or fully measure them.
The independent denial observer stops shared source activity on denial. Provider
retry instructions remain controlling; a renewed denial consumes the existing
single recovery and ends automated resumption. Do not rotate identity or routes.

Recovery is usable only with route data: NextEvents must contain current,
identifiable greyhound metadata with usable track condition on the expected URL;
HTML, wrong shape, empty/stale data is inconclusive and leaves STOP. Browser
recovery must pass the existing capture runner/market/race validation before its
operation closes. HTTP 200 alone is not recovery. Append and receipt checks still
run separately and unchanged. No validation relaxation is implied.

Before live observation, exported entrypoints and real capture subprocess tests
must pass on the changed code. The 90-minute final-candidate requirement, both-lane
progress, natural eligible windows, 270-second observation target and unchanged
300-second R3 rejection remain. The old package/window is never repurposed.
