# Sportsbet resumption finding — 23 September 2026

Historical finding at e8e2833e. The subsequent [durable coordination repair](collector_sportsbet_coordination_20260923.md) supersedes its runtime state and next-action section; retained evidence below is unchanged.

Status: **BLOCKED; no acquisition resumed.** PR #184 remains draft and permanent
rollout is held. This continues campaign `collector-integration-20260923-01a0ccec`;
no new campaign, allowance, launch, or attempt was created.

## Retained retry guidance

The campaign's failed Taree R9 T-30 capture recorded Sportsbet HTTP 429, but the
then-installed browser accounting retained only host/status. It drained CDP
response events without saving their headers; the driver did not write a HAR or
network log. The retained metrics have no Retry-After/Date/rate-limit fields.
Twelve retained campaign operational log files contain no Retry-After or
rate-limit header entries. No alternate retained header record was found.

Whether the provider supplied retry guidance is **unknown**, not "no header was
sent". Neither an earliest retry time nor the restriction's scope can be recovered
from this evidence. No new provider request was made to investigate.

## Ordinary traffic after restoration

Installed commands still use
`/home/l4nd0/greyhound-retention-lanes-repair-20260917/source`.
The following sanitized evidence was extracted at 17:33 AEST from the five
ordinary runs that started between 17:07:01 and 17:22:02:

| Lane | Evidence of continued source contact |
| --- | --- |
| `shadow-autopilot-odds-capture.service` | Four distinct Sportsbet snapshot timestamps: 17:08:00, 17:14:17, 17:18:13, 17:22:44 AEST; seven logged completed race-page navigations |
| `shadow-autopilot.service` | Sportsbet snapshot timestamps 17:26:12 and 17:27:24 AEST from its two refresh phases; seven logged completed race-page navigations |

These are operational observations, not a complete request count. Browser page
loading does not prove HTTP 200, usable odds, or absence of a challenge. Later
metadata responses do not establish recovery of the previously limited request.

The full timer runs 15 minutes after inactivity; the odds timer covers 56 minutes
per hour, excluding minutes 02/17/32/47. Existing service execution and shared-lock
contention also constrain actual activations. The three separately named
`strict-v4-prospective`, `forward-overround-structure`, and
`forward-overround-structure-v2` services were inactive with MainPID 0. No other
Greyhound user timer was listed. This is not a host-wide packet inventory.

## Backoff and the corrected defect

The installed shared Python client uses urllib3 retries: total 2,
backoff_factor 0.1, status list 500/502/503/504. Retry-After handling is enabled by
default. Despite 429 being absent from that status list, urllib3 2.5.0 also retries
429 when Retry-After is present. It waits in that individual request; without
retry guidance this configuration does not select 429 for a status retry.

This state is per request/process, not a shared Sportsbet cooldown. Browser
navigations do not use this adapter. The service lock serializes lane ownership,
but records no source retry deadline. Timers and new processes do not inherit
an earlier process's delay or the campaign's durable STOP. No established
provider-permitted, header-independent Sportsbet resumption policy was found in
the relevant collector paths. API rate-limit documentation concerns this
application's inbound API, not Sportsbet access permission.

A localhost fixture demonstrated a real defect: the adapter swallowed a
429/Retry-After and returned a second response's 200 before the campaign guard
could stop. The repair surfaces 401/403/429 to callers without automatic adapter
status retries, preserving existing transient-server-error behavior. Campaign
Python/browser guards now retain bounded, allowlisted retry metadata and local
observation times, without URLs, cookies, credentials or bodies. The browser
retains up to 128 denial events so later guidance in the same already-received
batch is not discarded; overflow is explicit and prevents treating that evidence
as complete guidance.

Validation: 23 focused checks passed, including real localhost transport,
later-header retention, Python STOP enforcement, existing source metadata and
capture binding. The installed pinned interpreter separately returned the
original 429 after exactly one localhost request. No dependencies were installed.
Standards review had no findings; the Spec review's later-header-loss finding was
fixed and rechecked. These changes are offline only, not installed into ordinary
collectors or validated by a new live launch. They create no permission to retry.

## Accounting boundary

Campaign ledger remains **1/12 attempts, 404/48,000 logical requests,
720.899931/10,800 charged seconds**. The 404 comprises 372 TheDogs + 15 Sportsbet +
15 Open-Meteo Python Session calls and two browser navigation invocations.
Separately, CDP observed 64 Sportsbet-host request events and 128 other-host
events. Navigations can overlap CDP events; do not add these as independent
physical requests. Adapter retries, redirects, browser background/startup traffic,
buffered/missing events, ordinary collectors and other host users are not fully
counted by that ledger. Therefore 404 is neither Sportsbet traffic nor a host-wide
total. This investigation added zero provider requests and zero campaign attempts.

## Smallest justified next action

Obtain an applicable permitted retry/access policy or recover original provider
guidance from an independently retained response record. A new request to test
recovery is not justified. The unsent inquiry below is ready for authorized use.
Elapsed time and ordinary successful metadata calls alone do not supply permission.

If guidance becomes available, calculate the earliest retry from its actual
semantics; it is not a guarantee of recovery. Before any required quiet interval
or live continuation, explicitly pause **both** named collector timers and wait
for their existing workers to release the shared lock naturally. Preserve their
exact unit hashes and prior enabled/active states; do not stop R3, unrelated
services, or kill ordinary workers. Any quiet interval must account for the last
relevant traffic, not merely the campaign's 16:57 stop.

Before launch, reconcile how a new source denial affects restoration: blindly
restarting ordinary Sportsbet traffic would defeat a shared hold. Any necessary
continued pause must name these two triggers, preserve rollback, and follow the
established policy. No runtime change is made by this report. Once this dependency
is resolved, repin/retest the amended package, use the same ledger and consumed
windows, and select the earliest valid future opportunity under the original
11-attempt, 47,596-request, 10,079.100069-second remaining ceilings. The unchanged
90-minute final-candidate acceptance objective still applies.

### Provider inquiry draft — not sent

On 23 September 2026 at approximately 06:57 UTC, a browser request to
www.sportsbet.com.au received HTTP 429 during an automated pre-race collection
test. We stopped that test. Separate ordinary collection subsequently continued;
we do not have a complete request count or the original response headers. Could
you confirm whether this automated use is permitted and, if so, the applicable
rate/concurrency limits, cooldown scope and retry guidance after 429 when
Retry-After is unavailable? We will not treat an earliest retry time as guaranteed
recovery. We can supply sanitized operational details if requested.

Local evidence:
`/home/l4nd0/greyhound-collector-campaign-20260923/resumption-20260923/operational-evidence.json`
and `pinned-interpreter-retry-test.json`. Original campaign evidence and consumed
attempts remain unchanged.
