# Sportsbet access review — 23 September 2026

**Decision: the reviewed evidence supports a narrowly bounded operational basis
for one conditional, same-identity recovery; it does not establish sustainable
normal workload or a provider-issued data licence.** No actual requirement for
third-party clarification before that limited recovery was established. Missing
explicit automation language does not by itself justify an indefinite hold.
This report does not change the durable source gate; its owner must record the
scope, reasoning and evidence through the existing locked mechanism before use.

This is a bounded continuation of `collector-integration-20260923-01a0ccec`.
No event/odds endpoint was opened, live recovery requested, inquiry sent, source
switched, or campaign counter changed. Public documentation was researched using
the web tool. This report is not a refreshed observation of the earlier HTTP 429.

## Current first-party documentation

The [General Rules](https://helpcentre.sportsbet.com.au/hc/en-us/articles/115004802547-Sportsbet-Rules-Terms-Conditions)
were retrieved on 23 September 2026 and identify 11 February 2026 as their update
date. Relevant provisions, paraphrased:

| Provision | Scope and consequence |
| --- | --- |
| 1.2.5 | Ordinary website access accepts applicable terms. |
| 1.3.1 | Restricted-residency access/bypass restrictions. |
| 1.5.2, 1.6.14 | Other individuals' account access requires approval. |
| 1.9.1–1.9.2 | Storage restriction covers Victorian thoroughbred video, not greyhound metadata/odds. |
| 1.10.1, 1.10.9 | Product availability is discretionary; betting products target recreational members. |
| 1.19 | User-generated-content rules, not an odds-data licence. |

No blanket scraper prohibition, NextEvents contract, collection quota, or
odds-retention condition was located there.

**Interpretation:** ordinary access under 1.2.5 can reasonably encompass limited
internal factual observations, with local timestamps and integrity records,
provided the operation avoids account sharing, redistribution, video copying,
betting actions and circumvention. No located clause requires separate approval
merely because those observations are automated. This supports a bounded access
decision, not a guarantee about enforceability, undocumented endpoint support,
or indefinite collection rights.

The [official terms hub](https://helpcentre.sportsbet.com.au/hc/en-us/sections/19653152538253-Sportsbet-Terms-Conditions-Hub)
links General, Racing, Product & Features, and Feed terms. The
[Racing terms](https://helpcentre.sportsbet.com.au/hc/en-us/articles/360052539092-Sportsbet-Racing-Terms-Conditions)
describe betting/settlement rules; no collection or retention permission or
numerical API limit was located. Their minimum-bet obligations are not data-access
quotas. The [Product & Features terms](https://helpcentre.sportsbet.com.au/hc/en-us/articles/115013721108-Product-Features-Terms-and-Conditions)
body says 8 September 2026, although its introductory summary still says 20 August
2024; the body date is used here. No NextEvents contract was located there.
The [Feed terms](https://helpcentre.sportsbet.com.au/hc/en-us/articles/20973619526925-Sportsbet-Feed-Terms-Conditions)
concern Sportsbet's consumer Feed feature, not a published racing-data API licence.

One public-document request for
[robots.txt](https://www.sportsbet.com.au/robots.txt) returned the web tool's
`Internal Error`, without an origin status or content. No robots rules, permission,
denial, or recovery can be inferred. No alternate transport or identity was used
to obtain it. Search snippets from unrelated bookmakers, `sportsbet.io`, and
third-party scraping examples were excluded as authority.

## Retained evidence and interpretation

The scoped records were the campaign's
`coordination-20260923/ACCESS_BASIS.md`, `coordination-20260923/OUTCOME.md`, the
[coordination report](collector_sportsbet_coordination_20260923.md), the
[resumption report](collector_sportsbet_resumption_20260923.md),
`docs/data_source_audit.md`, and the caller checkout's
`docs/prejump_incremental_data_source_acquisition_plan_20260817.md`.
No signed agreement, provider correspondence granting or rejecting these exact
routes, or numerical Sportsbet collection limit was present in that scope.
Unrelated Betfair/TAB/GRV acquisition restrictions are not applied to Sportsbet.
This is not a search of private email or all files on the host.

The actual metadata route is
`https://www.sportsbet.com.au/apigw/sportsbook-racing/Sportsbook/Racing/NextEvents`,
as defined in `utils/prejump_sportsbet.py`. It is a direct JSON request, not merely
the ordinary act of reading a rendered page. Browser fixed-WIN capture and local
append-only persistence are existing collector behavior. The scope justified
here is those existing factual observations for internal operational use, with
no expansion into resale, redistribution, additional fields, prediction/model
work or authenticated account operations. Treating every undocumented endpoint
as requiring a provider letter would add a policy requirement not found in the
reviewed evidence. Public access success cannot decide rights or rate limits.

The retained 429 establishes a rate-limit response at that time. Its original
headers were not retained; it is unknown whether Retry-After was sent. It says
nothing conclusive about a permanent ban. The 30-minute fallback, two-hour fallback
cap, shared serialization, and single recovery allowance are local engineering
policy; they are not documented Sportsbet quotas. Even resolving the access
interpretation leaves normal-workload sustainability to be demonstrated.

No numerical source limit was found in the bounded official-document search.
There is consequently no sourced basis here for choosing a requests-per-minute
figure and calling it provider-approved. A finite recovery operation under the
existing one-recovery policy can be justified without asserting such a figure:
preserve all prior denial evidence, wait through the existing deadline, use the
same identity and intended route, require valid route data rather than HTTP 200,
and stop shared activity on renewed denial. This is a deliberately bounded
operational judgment supported by the reviewed access terms and the user's
campaign authority; the engineering fallback is not attributed to Sportsbet.

The mixed campaign request ceiling and one successful response cannot justify
sustained load. Before a 90-minute observation, separately bound the actual
combined workload, identify requests versus navigations versus observed browser
events, preserve visibility gaps, and ensure that the policy covers both lanes.
No particular safe numerical sustained rate is recommended by this review.

## Optional exact unsent clarification

If normal workload cannot be justified with bounded operating evidence, the
remaining provider question is its **aggregate operating/retry conditions**.
Long-term contractual assurance about routes and retention is also useful, but
is not made a prerequisite for the single bounded recovery here. A published
policy or applicable existing agreement can answer these questions; a bespoke
letter is not required. Do not revisit the same records without new evidence.

The official terms hub lists Customer Service on 1800 990 907 and live chat from
06:00 to midnight AEST. Ask that team to route this inquiry to the team responsible
for automated data access. No contact has been made.

> Subject: Clarify greyhound metadata/odds access, local retention and limits
>
> Please identify the applicable published policy or agreement for an existing
> Australian collector that (1) automatically reads your public
> `/apigw/sportsbook-racing/Sportsbook/Racing/NextEvents` JSON route for upcoming
> race identity/timing and available pre-race metadata, and (2) uses an automated
> browser to read displayed greyhound fixed-WIN prices. It retains timestamped
> observations and integrity receipts locally in an append-only operational
> record. This recovery campaign places no bets and does not redistribute data.
>
> Do those exact read-and-retain operations fall within permitted website use,
> require different access terms, or fall outside permitted use? Please identify
> any restrictions on fields, retention duration, and subsequent internal use.
> We are not requesting permission for public redistribution or new model use.
>
> If permitted, what aggregate rate, burst and concurrency limits apply across
> metadata requests and browser traffic, including browser subrequests? What is
> the documented retry/cooldown scope after HTTP 429 when response headers were
> not retained? A link to an applicable policy is sufficient.
>
> At approximately 06:57 UTC on 23 September 2026 a browser request received 429.
> That test stopped; separate ordinary collection subsequently continued until
> both scheduled collector triggers were paused at 07:39:21 UTC. Existing workers
> drained by 07:56:57 UTC. The original headers and complete wire request totals
> are unavailable. Both triggers remain paused. We will not change identity or
> route to avoid a restriction. Any permitted recovery would use the same identity,
> one bounded operation, retained provider instructions, and stop on renewed denial.

This inquiry asks for facts missing from the reviewed sources; it does not assert
that the terms prohibit collection. Sending it remains a separate user decision.
Permanent rollout remains unsupported until sustainable workload and the live
collection/90-minute acceptance requirements are demonstrated. The reviewed
terms do not themselves require a further approval loop for bounded recovery.
