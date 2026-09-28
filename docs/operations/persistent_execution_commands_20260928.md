# Exact persistent deployment and acceptance handoff

**PREPARED ONLY. Do not execute installation or activation without the consolidated
approval requested for this packet.** No future owner assignment is required:
user-systemd runs the scheduler, existing supervisor, private result queue and
structural monitor. Any authorized operational session can use these commands.

Runtime commit: `869fca1c66a6ee7c557facdb55e6f7592f2992cb`.
Tested collector baseline `8b5552c7` and its evidence are retained unchanged.
The two archives below are different exports of the same source commit; do not
substitute their hashes:

- Deployment: `/home/l4nd0/greyhound-collector-campaign-20260923/persistent-preparation-20260928/deployment-869fca1c`.
  Manifest SHA256 `79640db81b72dbdbeb5024d3f08882ca70d5e34f44add5c432929653f0ffefd5`.
  Full Git archive SHA256 `2162323182b84c09a22953bbcd4ba6b1d5736781282e20fbe89d944608097d34`.
- Inactive authority: `/home/l4nd0/greyhound-persistent-comparison-output-20260928/activation-final`.
  Prepared manifest SHA256 `d5abb20433db531a655909d9b671a8b73209b8e28aee1c5d2b4dd5b5aa776327`.
- Exported offline verification: `/home/l4nd0/greyhound-persistent-comparison-output-20260928/export-869fca1c/persistent-verification.json`.
  PASS, networking denied in parent and children, synthetic inputs only. Actual
  systemd dispatch remains the first scheduled live canary, not an offline claim.

## Initialize one authorized shell

These environment variables contain paths, not secrets. Set `GRC_APPROVAL_REF`
to the actual decision approving the allocation, machine-only history and result
retention, budgets, installation and activation. The earlier operational approval
does not supply this new scientific/persistent authority.

```bash
set -euo pipefail
export GRC_RELEASE=/home/l4nd0/greyhound-persistent-release-869fca1c
export GRC_PY=/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-operator-ui-r3-python311-20260803-9d58f340/bin/python
export GRC_PACKAGE=/home/l4nd0/greyhound-collector-campaign-20260923/persistent-preparation-20260928/deployment-869fca1c
export GRC_PREPARED=/home/l4nd0/greyhound-persistent-comparison-output-20260928/activation-final
export GRC_PROGRAMME=/mnt/tenn-nvme2/tenn/greyhound-persistent-comparison-20261005
export GRC_CONTROL="$GRC_PROGRAMME/control"
export GRC_MANIFEST=79640db81b72dbdbeb5024d3f08882ca70d5e34f44add5c432929653f0ffefd5
export GRC_PREPARED_MANIFEST=d5abb20433db531a655909d9b671a8b73209b8e28aee1c5d2b4dd5b5aa776327
cd "$GRC_RELEASE"
```

Read-only preflight is permitted before approval:

```bash
"$GRC_PY" -B -m scripts.check_comparison_deployment \
  --package "$GRC_PACKAGE" --manifest-sha256 "$GRC_MANIFEST" --preflight
systemd-analyze --user verify "$GRC_PACKAGE"/units/*.service "$GRC_PACKAGE"/units/*.timer
loginctl show-user l4nd0 -p Linger
```

Require `CHECKS_PASS`, valid unit syntax and `Linger=yes`. Missing approval files
are expected at this stage. A changed ledger, source state, unit, R3 binding,
interpreter or mount requires reconciliation and a new prospective packet; never
edit an already sealed packet or reset counters to make it pass.

## Materialize and install after approval

```bash
: "${GRC_APPROVAL_REF:?Set GRC_APPROVAL_REF to the actual consolidated approving decision}"
export GRC_APPROVAL_REF
"$GRC_PY" -B -m scripts.authorize_persistent_comparison \
  --prepared "$GRC_PREPARED" --prepared-manifest-sha256 "$GRC_PREPARED_MANIFEST" \
  --output "$GRC_CONTROL" \
  --approval-reference "$GRC_APPROVAL_REF" \
  --allocation-reference "$GRC_APPROVAL_REF" \
  --history-reference "$GRC_APPROVAL_REF" \
  --result-reference "$GRC_APPROVAL_REF"
```

That command creates separate immutable approved files; it changes no service,
source gate or campaign ledger. It rejects expired starts or changed preparation
inputs. Before publication, validate the approved interfaces and recheck current
quiescence. The following publication block appends the programme authority and
installs only the six new comparison units. It never replaces baseline units.

```bash
"$GRC_PY" -B - <<'PY'
from datetime import datetime, timezone
import fcntl, hashlib, json, math, os
from pathlib import Path
from scripts.check_comparison_deployment import inspect
from scripts.run_comparison_schedule import load_config
from src.predictor.comparison_result_runtime import load_runtime
from src.predictor.future_comparison import checked
from race_collection.freshness_campaign import Campaign
from race_collection.live_freshness_contract import create_once, digest
package = Path(os.environ['GRC_PACKAGE'])
control = Path(os.environ['GRC_CONTROL'])
check = inspect(package, os.environ['GRC_MANIFEST'], preflight=True)
if check['status'] != 'CHECKS_PASS': raise SystemExit(check['findings'])
cfg, _ = load_config(control / 'schedule.APPROVED.json')
if cfg['authority_reference'] != os.environ['GRC_APPROVAL_REF']: raise SystemExit('approval_reference_changed')
load_runtime(json.loads((control / 'result-binding.APPROVED.json').read_bytes()), now=datetime.now(timezone.utc))
programme = json.loads(checked(control / 'programme-authority.APPROVED.json', cfg['programme_authority_sha256']))
campaign = Campaign(cfg['campaign_root'])
manifest = json.loads((package / 'deployment.json').read_bytes())
unit_dir = Path(cfg['installed_dir'])
with (campaign.root / 'owner.lock').open('a') as owner:
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ledger = json.loads((campaign.root / 'ledger.json').read_bytes())
    current = {'capture_attempts': len(ledger['attempts']), 'logical_requests': ledger['logical_requests'],
               'live_seconds': math.ceil(sum(r['charged_seconds'] for r in ledger['launches'].values()))}
    if current != programme['initial_counters'] or ledger.get('source_holds') or any(not r.get('closed_at') for r in ledger['launches'].values()):
        raise SystemExit('campaign_changed_or_busy')
    if campaign.programme or digest(campaign.value) != programme['prior_effective_authorization_sha256']:
        raise SystemExit('programme_authority_changed')
    source = Path(cfg['source_state'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != cfg['source_baseline']['state_sha256'] or Path(cfg['lock_path']).exists():
        raise SystemExit('source_changed_or_collector_busy')
    if any((unit_dir / name).exists() for name in manifest['unit_sha256']): raise SystemExit('comparison_units_already_present')
    target = campaign.root / 'persistent-programme-authority.json'
    create_once(target, programme)
    target.chmod(0o400)
    for name in manifest['unit_sha256']:
        with (unit_dir / name).open('xb') as stream:
            stream.write((package / 'units' / name).read_bytes())
            stream.flush(); os.fsync(stream.fileno())
        (unit_dir / name).chmod(0o644)
    fd = os.open(unit_dir, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
PY
systemctl --user daemon-reload
"$GRC_PY" -B -m scripts.check_comparison_deployment \
  --package "$GRC_PACKAGE" --manifest-sha256 "$GRC_MANIFEST" --preflight --installed
systemctl --user enable --now greyhound-comparison-schedule.timer greyhound-comparison-results.timer greyhound-comparison-health.timer
```

If publication is interrupted, keep every partial file and do not run activation.
Compare files with the manifest and approval receipt, finish only missing exact
copies under the same owner lock, then repeat installed preflight. Never replace
an existing programme authority or reset a consumed slot. No package is inferred
authorized merely because some approved files exist.

## Monitoring and first-slot acceptance

```bash
systemctl --user list-timers 'greyhound-comparison-*'
"$GRC_PY" -B -m scripts.check_comparison_deployment \
  --package "$GRC_PACKAGE" --manifest-sha256 "$GRC_MANIFEST" --installed
"$GRC_PY" -B -m scripts.check_comparison_health \
  --schedule-config "$GRC_CONTROL/schedule.APPROVED.json" \
  --result-binding "$GRC_CONTROL/result-binding.APPROVED.json"
journalctl --user -u greyhound-comparison-schedule.service -u greyhound-comparison-results.service -u greyhound-comparison-health.service --since today --no-pager -n 30
```

Use only these structural reports and aggregate receipts. Never inspect private
result bodies, result databases, child result logs or comparative performance.
The scheduler prepares slot 001 on **5 October 2026, 12:50–12:55 Melbourne time**;
observation is **13:00–14:30**, with bounded cleanup afterward. Systemd owns this
work even when the client disconnects. A late/missed slot remains missed.

Acceptance requires the existing supervisor's successful session summary,
both collector lanes, at least one verified pre-jump prediction, successful
restoration, closed campaign lease, released collector lock and unchanged R3
binding. Record available-window and attempt denominators, failures, unavailable
intervals, original source age, preparation/scoring latency and lead times from
that session's operational artifacts. Reuse baseline 90-minute evidence; do not
schedule an extra observation merely to repeat it.

When results become due, require the private worker's structural `CLOSED` state
for an exact first-slot verified job. Before slot 002, the scheduler writes
`sessions/canary.json` only if that gate passes; the receipt contains counts and
plan hash, no outcomes. Missing canary, unhealthy result retention, >24-hour overdue
work, source hold or failed restoration prevents later admission. A quiet period
or no eligible race is not a successful canary and does not authorize a makeup slot.

For a minimal live restart check **after this approval**, stop and restart the
result service between requests, retaining before/after structural counters:

```bash
systemctl --user stop greyhound-comparison-results.service
systemctl --user start greyhound-comparison-results.service
"$GRC_PY" -B -m scripts.check_comparison_health \
  --schedule-config "$GRC_CONTROL/schedule.APPROVED.json" \
  --result-binding "$GRC_CONTROL/result-binding.APPROVED.json"
```

No request is owed solely because the service restarted. Due times, consumed
attempts and closed jobs must remain unchanged unless genuinely due work ran.
Controlled offline tests cover interruption and idempotency; planned shutdown
alone is not claimed to prove restart recovery. Arbitrary reboot/PID changes and
unclean kills may hold ownership for operator reconciliation.

## Exact pause and rollback

Pause new predictions while leaving already owed results and monitoring running:

```bash
systemctl --user disable --now greyhound-comparison-schedule.timer
mountpoint -q /mnt/tenn-nvme2
test -d "$GRC_PROGRAMME/sessions"
touch "$GRC_PROGRAMME/sessions/PAUSE_ADMISSIONS"
systemctl --user show greyhound-comparison-schedule.service -p ActiveState -p MainPID -p ControlGroup
```

Allow natural completion; if interruption is required, use
`systemctl --user stop greyhound-comparison-schedule.service` and allow its
2,400-second stop grace. Do not kill individual collector children. For a
same-boot incomplete slot, start the paused schedule service once; its existing
restore-only path runs before pause admission is checked:

```bash
systemctl --user start greyhound-comparison-schedule.service
```

If restoration reports an unknown lifetime, changed R3 PID or retained stale lock,
leave admission paused and preserve diagnostics. There is no approved automatic
lock deletion or synthetic reaping command. A focused ownership repair is then
required; the proposal does not promise unattended recovery through that case.

For a full emergency pause, stop the two remaining timers, then naturally stop
the result and monitor services. Owed results remain recorded and unresolved:

```bash
systemctl --user disable --now greyhound-comparison-results.timer greyhound-comparison-health.timer
systemctl --user stop greyhound-comparison-results.service greyhound-comparison-health.service
```

After all three services exit, verify baseline unit hashes, original R3 binding,
no live workers/lock/source owner and no open campaign lease using installed
preflight. Only then remove exact owned unit files:

```bash
"$GRC_PY" -B - <<'PY'
import hashlib, json, os, subprocess
from pathlib import Path
package = Path(os.environ['GRC_PACKAGE'])
from scripts.check_comparison_deployment import inspect
check = inspect(package, os.environ['GRC_MANIFEST'], preflight=True, installed=True)
# Source holds and the admission disk floor do not prohibit safe removal.
blocking = set(check['findings']) - {'source_open', 'campaign_no_source_hold', 'free_space'}
if blocking: raise SystemExit(sorted(blocking))
print(json.dumps(check, sort_keys=True))
manifest = json.loads((package / 'deployment.json').read_bytes())
unit_dir = Path.home() / '.config/systemd/user'
for name, expected in manifest['unit_sha256'].items():
    state = subprocess.check_output(['systemctl','--user','show',name,'-p','ActiveState','--value'], text=True).strip()
    if state not in {'inactive','failed'}: raise SystemExit('unit_not_quiescent:' + name)
    if hashlib.sha256((unit_dir / name).read_bytes()).hexdigest() != expected: raise SystemExit('unit_changed:' + name)
for name in manifest['unit_sha256']: (unit_dir / name).unlink()
PY
systemctl --user daemon-reload
```

The rollback block permits source-hold findings and the admission-only free-space
floor while requiring mount identity, every quiescence check and baseline hashes
to pass. Disk pressure must not prevent safe removal. It prints the retained hold receipt
and removes only inactive exact comparison units. Do not clear a source hold
to make rollback pass. Never enable legacy collector timers while held. Preserve the campaign
amendment, approved files, all programme evidence, private queue and source history.
Permanent model replacement, betting, merging, evaluation and human target-result
access are outside this proposal.
