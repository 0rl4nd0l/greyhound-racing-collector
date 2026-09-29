# Odds retention: acquisition change required for useful movement pairs

The installed operational route does **not** naturally acquire two qualified WIN observations separated by two minutes before its forecast. Its attempt ledger consumes a race across all capture windows. Saving more of the same bounded browser visit cannot produce the proposed T−10/T−4 pair. No retention hook, scheduler, runtime configuration or provider operation was added by this audit. The useful deliverable is the following acquisition and evidence contract for the operational owner; it requires a future allocation and explicit operational authorization.

This investigation reuses [PR #197's qualification result](market_movement_20260929_results.md) without rerunning its historical-pair audit, opening a database or decoding any new runner histories/outcomes. Its zero qualified pairs remain the historical finding, not a new result from this work. Source inspection, installed-unit inspection and reservation/control metadata are the evidence below.

## Source identity and current operational boundary

Read-only inspection on 29 September 2026 found:

| Surface | Verified identity |
|---|---|
| Research implementation checkout | `/home/l4nd0/greyhound-feature-quality-20260929`, base `bec84e8074dacf9c32330f040a67dc46919f3d88` |
| Installed user schedule/results unit WorkingDirectory and PYTHONPATH | `/home/l4nd0/greyhound-persistent-release-handoff-1cd7ccff` |
| Installed release HEAD | `1cd7ccff79b0fed33e5411b9bf1fb3f24b5ac280`; tracked Git status clean |
| Approved schedule `source_commit` | Same `1cd7ccff79b0fed33e5411b9bf1fb3f24b5ac280` |
| Approved control root | `/mnt/tenn-nvme2/tenn/greyhound-persistent-comparison-20261005/control-october1-v2-20260928` |
| Schedule SHA-256 | `d053131e7d259835c1e475dfb5d1ef981d714e3b1e370961941b0afef922be88` |
| Frozen comparison plan SHA-256 | `e0c71cba1618a05416a0087e85141803c32bd806b8faad36259a8a0d70808adb` |
| Exclusive allocation SHA-256 | `b708fa973aa972b8cd248b4b4d3269fa7aa16402755ee0fb84da5212db6822d1` |
| Reservation review SHA-256 | `4c1f859c8292fa76bb1a70809133f56fa534fc16c98d0371d9d36a5875b7030e` |

The code links below pin the **installed** source, because its paired-card readiness behavior is newer than the research branch. Unit presence and clean source are configuration observations, not proof of live predictive performance. The schedule validates its source commit/cleanliness and exact finite budget before operation: [run_comparison_schedule.py:23–47](https://github.com/0rl4nd0l/greyhound-racing-collector/blob/1cd7ccff79b0fed33e5411b9bf1fb3f24b5ac280/scripts/run_comparison_schedule.py#L23-L47). Live execution sets `GREYHOUND_SPORTSBET_RESPONSE_INSPECTION=1`, selecting the operational paired-readiness branch: [live_execution.py:74](https://github.com/0rl4nd0l/greyhound-racing-collector/blob/1cd7ccff79b0fed33e5411b9bf1fb3f24b5ac280/race_collection/live_execution.py#L74).

## What is acquired, retained and lost

| Stage | Actual behavior | Consequence for movement |
|---|---|---|
| Metadata refresh | `NextEvents` supplies pre-race event metadata; the module does not establish runner WIN prices. [utils/prejump_sportsbet.py:1–44](https://github.com/0rl4nd0l/greyhound-racing-collector/blob/1cd7ccff79b0fed33e5411b9bf1fb3f24b5ac280/utils/prejump_sportsbet.py#L1-L44) | Repeated discovery refreshes are not earlier WIN snapshots. |
| Race admission | Generic windows include T−60/T−30/T−10/T−2, but `AttemptAllowance.consumed` rejects **any prior alias-matched race attempt** across windows when `operational_predictions` is enabled. [live_freshness_contract.py:225–245](https://github.com/0rl4nd0l/greyhound-racing-collector/blob/1cd7ccff79b0fed33e5411b9bf1fb3f24b5ac280/race_collection/live_freshness_contract.py#L225-L245) | A successful T−10 capture, or a consumed failed attempt, is not followed by a new operational T−2 capture of that race. |
| Browser fetch | Visit landing page, resolve target, call `get_race_odds_from_page` once, return one `odds_data`/`race_info`, then close the driver. [odds_auto_integrator.py:489–625](https://github.com/0rl4nd0l/greyhound-racing-collector/blob/1cd7ccff79b0fed33e5411b9bf1fb3f24b5ac280/odds_auto_integrator.py#L489-L625) | No persistent per-race quote stream remains alive for a second point. |
| DOM readiness | Poll paired-card counts every ≤0.25 seconds for at most five seconds; stop at the first complete paired field. Start/end counts are diagnostic only. [sportsbet_odds_integrator.py:1653–1711](https://github.com/0rl4nd0l/greyhound-racing-collector/blob/1cd7ccff79b0fed33e5411b9bf1fb3f24b5ac280/sportsbet_odds_integrator.py#L1653-L1711) | Intermediate card text/prices are transient and discarded, but neither independent complete fields nor two-minute separation are demonstrated. Retaining these samples would not solve movement qualification. |
| WIN/PLACE extraction | Paired text yields simultaneous WIN and PLACE rows. These are **two markets at one observation**, not two WIN times. Operational mode disables the later PLACE-interaction fallback; in the generic non-operational fallback a second successful paired render replaces in-memory WIN rows. [sportsbet_odds_integrator.py:1559–1639](https://github.com/0rl4nd0l/greyhound-racing-collector/blob/1cd7ccff79b0fed33e5411b9bf1fb3f24b5ac280/sportsbet_odds_integrator.py#L1559-L1639) | Do not mistake market pairs or a legacy overwrite path for an existing operational movement stream. |
| Browser response inspection | Retains route metadata and value-free JSON shapes, deliberately suppressing scalar values/raw bodies; not an odds source. [sportsbet_response_inspection.py:1–5,91–145](https://github.com/0rl4nd0l/greyhound-racing-collector/blob/1cd7ccff79b0fed33e5411b9bf1fb3f24b5ac280/utils/sportsbet_response_inspection.py#L91-L145) | Route names or repeated responses do not establish multiple complete WIN observations. No new raw-response retention is authorized by the existing diagnostic recorder. |
| Validated append | PLACE then WIN rows are inserted, with separate commits, using the same caller-supplied timestamp; existing odds rows are not updated. Raw paired runner text and box provenance survive. [autonomous_live_odds_capture.py:2499–2597](https://github.com/0rl4nd0l/greyhound-racing-collector/blob/1cd7ccff79b0fed33e5411b9bf1fb3f24b5ac280/scripts/autonomous_live_odds_capture.py#L2499-L2597), [sportsbet_odds_integrator.py:1209–1379](https://github.com/0rl4nd0l/greyhound-racing-collector/blob/1cd7ccff79b0fed33e5411b9bf1fb3f24b5ac280/sportsbet_odds_integrator.py#L1209-L1379) | Odds rows are append-only; the current operational gap is not an UPDATE that erases an earlier qualified capture. Single-market partial persistence must not count as a completed receipt. |
| Timing | `append_time` is sampled before append and supplied as `capture_timestamp`; SQL commit occurs later. [autonomous_live_odds_capture.py:2761–2818](https://github.com/0rl4nd0l/greyhound-racing-collector/blob/1cd7ccff79b0fed33e5411b9bf1fb3f24b5ac280/scripts/autonomous_live_odds_capture.py#L2761-L2818) | The row timestamp alone proves neither exact DOM observation time nor durable availability time. Preserve the existing receipt chain, and add separate semantics for future snapshots. |

Retention of additional metadata around the **one** already acquired validated snapshot could improve future auditability without another provider request. That is distinct from obtaining a useful movement pair. No independently qualified second observation is exposed by the reviewed current path, so a speculative default-off hook was not implemented.

## Minimum comparable snapshot record

Each record must be immutable, content-addressed and linked to an append-only target/attempt manifest. Every missed, invalid, partial, late or denied observation needs a disposition, so future eligibility does not depend on successful capture or known winners.

| Group | Required fields and interpretation |
|---|---|
| Identity | Canonical race key; provider race, WIN market and runner IDs when actually supplied; original source URL; canonical mapping/version; native IDs absent must be explicit, not invented. Stable mapped name/box identities require an independently verified mapping if native selection IDs are unavailable. |
| Full field | Every provider roster member, including reserves/inactive runners; canonical runner, observed box, reserve/active/scratched/suspended flags with explicit unknown state; market open/suspended/closed state. Include source-effective and collector-observed times for changes if supplied. Missing price is not proof of scratching. |
| Prices | Decimal fixed WIN price, validity, paired market evidence sufficient to verify classification, complete active field, field hash and overround. Preserve PLACE text when it is the available evidence proving which member of a pair is WIN; no PLACE-to-WIN inference. |
| Clocks | Request start, source quote time if supplied, observation start/end, receipt time, post-commit/durable-publication acknowledgment time and monotonic sequence/clock identity. Distinguish collector clock from provider time. Never synthesize availability from mtime or append-start time. A later durable acknowledgment can serve as a conservative availability bound. |
| Schedule | Scheduled jump timestamp including UTC offset and named timezone; identity/hash/version of the discovery schedule; observed revision time and the fixed decision cutoff. Do not treat scheduled jump as actual off. A changed schedule is a new version, never silent backdating. |
| Provenance | Acquisition source and price-origin bookmaker separately; parser/schema version and source commit; source evidence bytes or exact relevant DOM fragments with hash; capture/reservation/receipt identities; storage hash, commit acknowledgment and qualification reasons. Retain only pre-race evidence needed for the market proof, without unrelated page/account data. |

Qualification needs unchanged, independently complete active fields at both observations; no silent runner intersection or scratch-induced renormalization. Unknown market/runner state fails pair qualification. The parser currently proves paired WIN/PLACE text and expected-field matching, but does not by itself demonstrate all native IDs, provider quote times or suspension-event semantics required above. Those are evidence requirements to validate during a future authorized acquisition, not fields to fill with guesses.

## Concrete proposal for a separately allocated future pilot

**Status: design only; not an amendment to October.** Reuse the existing finite collector supervisor and shared lock, with an explicitly versioned policy allowing exactly two separately consumed observation slots for pilot race identities. No new scheduler, parallel browser owner, retry allowance or change to the frozen scorer is proposed. Changing only a directory or bypassing `consumed` is not sufficient authority.

1. Target first observation around scheduled T−10 and second around T−4, durably available by T−2. Fix a two-minute minimum observation separation and the [existing protocol's](market_movement_20260929_protocol.md) T−30 early bound / eight-minute late-age bound. Record actual timing; T−2 capture opens too late to guarantee availability by T−2. T−4 currently falls inside the generic T−10 window, so the future pilot needs a **new observation-slot identity** within the existing supervisor, not reuse of a consumed T−10 reservation. Do not alter current window meanings or revive failed attempts.
2. Incremental cost per successfully paired race: one extra guarded browser operation, normally at least two extra navigations (landing plus target) with the present fetch function, plus all resulting subresource requests. Meeting resolution may require more navigations and can be blocked by the existing cap. The exact logical-request and latency costs are unknown until measured; one browser operation is not one HTTP request. Preserve `BrowserNetworkAccounting` and source denial STOP. [Browser admission/navigation budget](https://github.com/0rl4nd0l/greyhound-racing-collector/blob/1cd7ccff79b0fed33e5411b9bf1fb3f24b5ac280/utils/sportsbet_browser.py#L65-L81), [navigation cap](https://github.com/0rl4nd0l/greyhound-racing-collector/blob/1cd7ccff79b0fed33e5411b9bf1fb3f24b5ac280/utils/sportsbet_browser.py#L189-L203).
3. Current reserve admission requires 155 seconds remaining and fetch admission requires 50 seconds; these are admission requirements, **not measured lock occupancy**. [live_freshness_contract.py:266–269,401](https://github.com/0rl4nd0l/greyhound-racing-collector/blob/1cd7ccff79b0fed33e5411b9bf1fb3f24b5ac280/race_collection/live_freshness_contract.py#L266-L269). Budget one additional bounded collector cycle and its measured append/publication time per paired race. Release the shared lock between observations: holding a browser/lock from T−10 to T−4 would add six minutes of contention and is not the proposal. The existing 50-second fetch admission alone does not prove that a complete additional cycle fits before T−2.
4. Forecast coverage has priority. Admit pilot work only where the existing owner can demonstrate slack before all relevant capture/forecast deadlines; otherwise mark the extra slot missed. Do not reserve pilot work against October's 192 source operations/session or its capture allowance. With a fixed future pilot allowance of `A` one-observation attempts, at most `floor(A/2)` races can obtain two attempted points before discovery/failure costs; pair coverage can be lower. Queue interference, source denials and complete-field failures can reduce forecast coverage. Zero coverage impact has not been established.
5. Storage proposal (planning assumptions, not measured current footprints): cap each normalized record + relevant DOM evidence + receipts at 256 KiB, preserving an explicit oversize rejection; reserve up to 512 KiB per pair plus target/attempt manifest overhead, approximately 500 MiB for 1,000 pairs. No whole-page/video/network-body archive is required. Measure actual byte sizes and write/fsync durations in an offline fixture rehearsal before the owner chooses operational limits. Retaining only a hash would not be sufficient market evidence.
6. Offline acceptance before any pilot activation: synthesize two valid snapshots, duplicated IDs, reserve swaps, missing status, suspension, incomplete fields, ambiguous WIN/PLACE text, changed jump versions, late durability, interrupted append and consumed retries. Verify pair qualification/failure dispositions, source budgets, lock release, and identical ordinary forecasts with the hook disabled. Then obtain bounded **outcome-blind** operational coverage evidence; no performance fit until population and evaluation are separately frozen.

## Population reservation decision

The approved allocation and plan above reserve the four-way programme from `2026-10-01T12:00:00+10:00` through `2027-01-21T12:00:00+11:00`; the schedule has 80 slots. Existing historical reservation summaries cover July manifests, the compromised August predecessor and the August20–September30 replacement; those remain protected. The [29 September source checkout's reservation review](https://github.com/0rl4nd0l/greyhound-racing-collector/blob/1cd7ccff79b0fed33e5411b9bf1fb3f24b5ac280/docs/research/future_comparison_20260928_evidence/reservation_review.json) records the deferred overround/residual proposals and explicitly warns that November is not automatically unreserved.

The exact four successor activation paths in the approved `reservation-review.json` were checked again on 29 September: each remains absent and not a symlink. That bounded check does not discover every possible reservation; absence of those activations grants no new allocation. No suitably authorized unreserved **future** research population is established here.

The concrete owner decision needed is a new exclusive allocation after the current programme endpoint (or a separately proven disjoint population that changes no current inclusion rule), reconciled with all current reservations/deferred proposals, with explicit dates/venues/race eligibility, finite acquisition budget, history/label access policy and an independent chronological evaluation plan. Even after 21 January, independence must be decided rather than presumed. Freeze membership before labels; a separate directory or new study name cannot remove overlap.

For this task, improving retained-history meaning takes priority over a movement fit: it can be diagnosed within admitted existing inputs. The narrow later movement question remains whether **qualified pre-cutoff normalized WIN movement** adds to the same-time market on identical eligible races. First establish pair coverage and field/status semantics under a new allocation; no new model search is justified by the present evidence.

## Handoff and reproduction

Operational owner: retain current October source, models, allocation, budgets and services unchanged. Review the future two-slot policy and record contract above only after a new population decision. This audit makes no deployment request for the current programme. Findings are code/configuration evidence; no provider traffic, database opens, new historical-pair audit, fitted model or protected outcome decoding occurred.

Read-only reproduction of the key boundary (no runtime imports):

```bash
RELEASE=/home/l4nd0/greyhound-persistent-release-handoff-1cd7ccff
systemctl --user cat greyhound-comparison-schedule.service greyhound-comparison-results.service
git -C "$RELEASE" rev-parse HEAD
git -C "$RELEASE" status --porcelain --untracked-files=no
sed -n '225,269p' "$RELEASE/race_collection/live_freshness_contract.py"
sed -n '489,625p' "$RELEASE/odds_auto_integrator.py"
sed -n '1653,1711p' "$RELEASE/sportsbet_odds_integrator.py"
sed -n '2499,2597p' "$RELEASE/scripts/autonomous_live_odds_capture.py"
```

This document introduces no executable path, so no new retention tests or production suite were run. The task's separate development-feature changes carry their own focused tests.
