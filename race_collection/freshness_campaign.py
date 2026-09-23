"""Durable cumulative engineering limits, independent of launch/package identity."""
import fcntl
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from race_collection.live_phase_checkpoint import atomic_json


class Campaign:
    def __init__(self, path):
        self.root = Path(path).resolve()
        self.value = json.loads((self.root / 'authorization.json').read_bytes())
        if (self.value.get('schema_version') != 'collector_engineering_campaign_v1'
                or self.value['max_capture_attempts'] != 12
                or self.value['max_logical_requests'] != 48000
                or self.value['max_live_seconds'] != 10800):
            raise ValueError('invalid_campaign_authorization')

    @contextmanager
    def ledger(self):
        with (self.root / 'ledger.lock').open('a') as mutex:
            fcntl.flock(mutex, fcntl.LOCK_EX)
            path = self.root / 'ledger.json'
            value = json.loads(path.read_bytes())
            if value['campaign_id'] != self.value['campaign_id']:
                raise ValueError('campaign_accounting_identity_changed')
            yield value
            atomic_json(path, value)

    def admit(self, launch, now):
        with self.ledger() as value:
            row = value['launches'].get(launch)
            if row is None or row.get('closed_at') or now.timestamp() >= row['deadline_epoch']:
                raise ValueError('campaign_live_lease_closed')

    def begin(self, launch, *, now, deadline):
        with self.ledger() as value:
            if launch in value['launches'] or any(not r.get('closed_at') for r in value['launches'].values()):
                raise ValueError('campaign_owner_or_launch_already_exists')
            used = sum(r['charged_seconds'] for r in value['launches'].values())
            charge = (deadline - now).total_seconds()
            if charge <= 0 or used + charge > self.value['max_live_seconds']:
                raise ValueError('campaign_live_time_exhausted')
            value['launches'][launch] = dict(started_at=now.isoformat(),
                deadline_epoch=deadline.timestamp(), charged_seconds=charge)

    def close(self, launch, *, now):
        # Only after verified restoration. Unclosed launches keep their full charge.
        with self.ledger() as value:
            row = value['launches'][launch]
            if not row.get('closed_at'):
                row['charged_seconds'] = max(0, (now - datetime.fromisoformat(row['started_at'])).total_seconds())
                row['closed_at'] = now.isoformat()

    def available(self):
        with self.ledger() as value:
            return len(value['attempts']) < self.value['max_capture_attempts']

    def consume(self, claim, item):
        with self.ledger() as value:
            if len(value['attempts']) >= self.value['max_capture_attempts']:
                raise ValueError('campaign_capture_allowance_consumed')
            aliases = set(item.get('race_id_aliases', [item['race_id']])) | {item['race_id']}
            if any(r['window'] == item['capture_window_minutes'] and aliases.intersection(r['aliases'])
                   for r in value['attempts']):
                raise ValueError('campaign_capture_window_consumed')
            value['attempts'].append(dict(claim=str(claim), race_id=item['race_id'],
                aliases=sorted(aliases), window=item['capture_window_minutes'],
                consumed_at=datetime.now(timezone.utc).isoformat()))

    def request(self):
        with self.ledger() as value:
            if value['logical_requests'] >= self.value['max_logical_requests']:
                raise ValueError('campaign_request_cap_exhausted')
            value['logical_requests'] += 1
