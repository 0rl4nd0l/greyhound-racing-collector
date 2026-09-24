# Sportsbet collection recovery, 24 September 2026

User authority: own investigation, isolated repairs, supervised source access,
temporary collector/R3 changes and real operational predictions. This explicitly
supersedes old internal attempt/recovery/window limits prospectively. Prior STOPs,
failures, consumed jobs and cumulative totals remain evidence. No betting, target
results, model changes, research evaluation or permanent deployment.

Baseline is PR188 commit b9d2db2c, which includes PR184/187/185 and PR189's recorder
plus ownership diagnostics. The active worktree is isolated. Root is the only
provider owner; use the existing campaign owner lock and shared source gate.

## Verified prior success

The latest retained legacy capture is Bendigo R13, 23 September, fetch
17:41:44.746393 and append 17:42:24.804957 AEST (40.058564 seconds). Eight exact
runners have eight WIN and eight PLACE rows. Current receipt verification against
sealed report/form/sidecar bytes passes at the historical capture clock plus one
second. This is historical verification, not a new capture. Source was the
installed retention release 64c71c568fbd3ef6fdc972e5fcb7640cc3b7c70c; its drivers.py
hash matches git. Receipt report SHA256 is
c01e70f4cadf0f786e4e2ed375bae084f1601ac20e371f245e86a4f6b27b55c8,
runner hash f08d04498460eaa43f4bc6c7c2719c518b05490c55cf77182b91d2655dcbfde7.

The success log identifies Chrome146.0.7680.153 and webdriver-manager selection.
Both installed interpreters currently report Python3.11.15, Selenium4.34.2,
webdriver-manager4.0.2, requests2.32.4. The experimental package pins Chrome
146.0.7680.153 and ChromeDriver146.0.7680.165 by hash. The historical log does not
attest the exact driver version or a full historical Python distribution snapshot;
present installed versions cannot fill that historical gap.

Both paths construct a new browser per race, load the greyhound landing page,
wait five seconds, select an exact venue/race link, load the race, wait for ready
state/dynamic content and extract rendered rows. The successful run made three
sequential complete captures (~35.5–40.1s each); full wire request counts were not
retained. Both lanes use the collector lock. No evidenced cross-agent breach.

Extraction code is unchanged. The experimental pinned driver removes the legacy
window-size/asset options and legacy automation-concealment/UA settings, adds
background suppression and independent CDP denial monitoring. Do not reintroduce
concealment or blindly restore packages. Launch differences alone are not causes.

Ranked falsifiable hypotheses:
1. New browser-wide guard converts an unrelated resource denial into capture
   failure: denied route/type and preceding data responses will distinguish it.
2. Browser configuration changes delivery: retain exact settings and vary one
   ordinary compatibility option only after evidence supports comparison.
3. Provider behavior changed: required document/odds denial on the unchanged
   packaged path supports a current external obstacle.
4. Traffic pressure: count ownership/operations/cadence; no historical wire trace
   means increased load cannot be assumed.

## Diagnostic policy and implementation

One explicit prospective source transition has a hash-bound prior state,
reference, rationale, expiry and finite operation allowance. Existing denials,
recovery_attempts=1, operations and not_before remain intact. Honor the later of
retained Retry-After and existing engineering cooldown. Any new denial stops the
sequence; no automatic recovery. Another sequence requires a materially different
question and new recorded justification after backoff.

The first comparison changes instrumentation only. Attach PR189's recorder in the
actual packaged operational capture, before navigation; retain its sanitized
report even when navigation fails. Record operation identity, categories,
response status, local timing and retry headers. Value-free JSON shape inspection
uses at most four already delivered bodies and performs no replay. Existing
identity, full-market and receipt validators remain authoritative. The guard
continues stopping on denial pending actual resource evidence.

Before execution record finite limits in the private authorization artifact:
at most 90 minutes observation, 31 minutes reserved cleanup, 128 source operations,
one browser operation/minute, ten Python source operations/minute, two explicit
navigations/capture, existing 50-second capture ceiling and zero automatic
retries. Cumulative campaign caps and all consumed identities remain counted;
if additional authority is needed it is an appended prospective amendment.
Limits fund the existing complete cycle and ensure cleanup while keeping source
traffic serial. They are engineering limits, never provider-approved capacity.

Tests use the pinned interpreter, network denial and fabricated data. The actual
entrypoint missing-recorder regression failed for Document/XHR/Image denials;
after wiring it passed. Prospective authority tests cover preserved STOP history,
cooldown, finite operation consumption and a renewed-denial stop. Packaged
capture/retention/frozen prediction remains required before any live candidate.

Private evidence root:
/home/l4nd0/greyhound-collector-campaign-20260923/recovery-20260924.
