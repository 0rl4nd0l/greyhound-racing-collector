"""Durable cumulative engineering limits, independent of launch/package identity."""
import fcntl
import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from race_collection.live_phase_checkpoint import atomic_json


class Campaign:
    def __init__(self, path, *, engineering_authority=None, development_authority=None):
        if engineering_authority is not None and development_authority is not None:
            raise ValueError("conflicting_campaign_profiles")
        if engineering_authority is not None and (
                not isinstance(engineering_authority, str) or not engineering_authority.strip()):
            raise ValueError('explicit_engineering_authority_required')
        self.engineering_authority = engineering_authority
        self.root = Path(path).resolve()
        self.value = json.loads((self.root / 'authorization.json').read_bytes())
        if (self.value.get('schema_version') != 'collector_engineering_campaign_v1'
                or self.value['max_capture_attempts'] != 12
                or self.value['max_logical_requests'] != 48000
                or self.value['max_live_seconds'] != 10800):
            raise ValueError('invalid_campaign_authorization')
        amendment = self.root / 'prospective-authorization-amendment.json'
        if amendment.exists():
            extra = json.loads(amendment.read_bytes())
            if (extra.get('schema_version') != 'collector_engineering_amendment_v1'
                    or extra.get('campaign_id') != self.value['campaign_id']
                    or extra.get('prior_authorization_sha256') != hashlib.sha256(
                        (self.root / 'authorization.json').read_bytes()).hexdigest()
                    or not extra.get('authority_reference') or not extra.get('rationale')
                    or type(extra.get('max_capture_attempts')) is not int
                    or not 12 <= extra['max_capture_attempts'] <= 64
                    or type(extra.get('max_live_seconds', 10800)) is not int
                    or not 10800 <= extra.get('max_live_seconds', 10800) <= 21600):
                raise ValueError('invalid_prospective_campaign_amendment')
            # The original authority bytes and all ledger consumption remain.
            # A new package binds the effective authority, including this record.
            self.value = {**self.value, 'max_capture_attempts': extra['max_capture_attempts'],
                          'max_live_seconds': extra.get('max_live_seconds', 10800),
                          'prospective_amendment': extra}
        # New authorizations are immutable, ordered and bound to the entire
        # preceding effective authority. Historical files and charges stay put.
        from race_collection.live_freshness_contract import digest
        for number, path in enumerate(sorted((self.root / 'authorization-extensions').glob('*.json')), 1):
            extra = json.loads(path.read_bytes())
            limits = {'max_capture_attempts': 128, 'max_logical_requests': 96000,
                      'max_live_seconds': 43200}
            if (path.name != f'{number:04d}.json'
                    or extra.get('schema_version') != 'collector_engineering_extension_v1'
                    or extra.get('campaign_id') != self.value['campaign_id']
                    or extra.get('prior_effective_authorization_sha256') != digest(self.value)
                    or not extra.get('authority_reference') or not extra.get('rationale')
                    or any(type(extra.get(key)) is not int
                           or not self.value[key] <= extra[key] <= ceiling
                           for key, ceiling in limits.items())):
                raise ValueError('invalid_campaign_extension')
            self.value = {**self.value, **{key: extra[key] for key in limits},
                          'extensions': [*self.value.get('extensions', []), extra]}

        engineering_value = self.value
        programme = self.root / 'persistent-programme-authority.json'
        if programme.exists():
            extra = json.loads(programme.read_bytes())
            limits = {'max_capture_attempts': 1128, 'max_logical_requests': 1400000,
                      'max_live_seconds': 624000}
            if (extra.get('schema_version') != 'collector_persistent_programme_v1'
                    or extra.get('status') != 'AUTHORIZED_PERSISTENT_PROGRAMME'
                    or extra.get('campaign_id') != self.value['campaign_id']
                    or extra.get('prior_effective_authorization_sha256') != digest(self.value)
                    or not extra.get('authority_reference')
                    or not extra.get('programme_id')
                    or any(type(extra.get(k)) is not int or not self.value[k] <= extra[k] <= cap
                           for k, cap in limits.items())):
                raise ValueError('invalid_persistent_programme_authority')
            start = datetime.fromisoformat(extra['starts_at'])
            end = datetime.fromisoformat(extra['expires_at'])
            if start.tzinfo is None or end.tzinfo is None or not 0 < (end-start).total_seconds() <= 127*86400:
                raise ValueError('invalid_persistent_programme_window')
            initial=extra['initial_counters']
            if any(type(initial.get(k)) is not int or initial[k]<0 for k in ('capture_attempts','logical_requests','live_seconds')):
                raise ValueError('invalid_programme_initial_counters')
            self.programme = extra
            self.value = {**self.value, **{k: extra[k] for k in limits}, 'persistent_programme': extra}
            # Date-only prospective amendments preserve original authority and all
            # consumption. No budget, identity, model or routing change is allowed.
            for number, amendment in enumerate(sorted((self.root / 'programme-schedule-amendments').glob('*.json')), 1):
                row = json.loads(amendment.read_bytes())
                revised = row.get('programme', {})
                allowed = {'starts_at', 'expires_at'}
                issued = datetime.fromisoformat(row['issued_at'])
                start = datetime.fromisoformat(revised['starts_at'])
                end = datetime.fromisoformat(revised['expires_at'])
                if (amendment.name != f'{number:04d}.json'
                        or row.get('schema_version') != 'programme_schedule_amendment_v1'
                        or not row.get('authority_reference') or not row.get('empty_state_sha256')
                        or row.get('prior_programme_sha256') != digest(self.programme)
                        or set(revised) != set(self.programme)
                        or any(revised[k] != self.programme[k] for k in revised.keys() - allowed)
                        or issued.tzinfo is None or start.tzinfo is None or end.tzinfo is None
                        or not issued < start < end or not 125*86400 <= (end-start).total_seconds() <= 127*86400):
                    raise ValueError('invalid_programme_schedule_amendment')
                self.programme = revised
                self.value = {**self.value, 'persistent_programme': revised,
                              'programme_schedule_amendments': [*self.value.get('programme_schedule_amendments', []), row]}
        else:
            self.programme = None
        self.study_programme = self.programme
        self.development_authority = development_authority
        self.development = None
        self.pilot_authority = None
        pilot_path = self.root / 'development-pilot-authority.json'
        if pilot_path.exists():
            from race_collection.development_source_authority import load_development_authority
            pilot_ref = dict(path=str(pilot_path), sha256=hashlib.sha256(pilot_path.read_bytes()).hexdigest())
            self.pilot_authority = load_development_authority(pilot_ref)
            if (self.pilot_authority['campaign_id'] != self.value['campaign_id']
                    or self.pilot_authority['prior_effective_authorization_sha256'] != digest(self.value)):
                raise ValueError('development_campaign_binding_changed')
            self.pilot_ref = pilot_ref
        if development_authority is not None:
            if not self.pilot_authority or development_authority != self.pilot_ref:
                raise ValueError('development_authority_not_installed_in_campaign')
            self.development = self.pilot_authority
            self.value = {**self.value, **{key:self.development[key] for key in
                ('max_capture_attempts', 'max_logical_requests', 'max_live_seconds')},
                'development_authority': development_authority}
            self.programme = None
        if engineering_authority is not None:
            # Explicitly selected in the approved plan, never inferred from a
            # date or the existence of a programme. Retain the same root/locks.
            self.value = {**engineering_value,
                          'engineering_authority': engineering_authority,
                          'study_authority_sha256': digest(self.value)}
            self.programme = None

    @staticmethod
    def from_scope(value):
        if value.get('development_authority') is not None:
            predictions = value.get('operational_predictions', {})
            if (value.get('engineering_authority') is not None or value.get('frozen_comparison')
                    or not predictions or predictions.get('result_access') is not False):
                raise ValueError('development_requires_separate_operational_predictions')
            return Campaign(value['campaign_root'], development_authority=value['development_authority'])
        authority = value.get('engineering_authority')
        if authority is None:
            return Campaign(value['campaign_root'])
        predictions = value.get('operational_predictions', {})
        if (not predictions or predictions.get('result_access') is not False
                or predictions.get('research_activation') is not False
                or value.get('prediction_root') or value.get('frozen_comparison')):
            raise ValueError('engineering_requires_separate_operational_predictions')
        return Campaign(value['campaign_root'], engineering_authority=authority)

    def programme_usage(self, value):
        """Cumulative totals stay intact; explicitly charged engineering is separate."""
        initial = self.programme['initial_counters']
        pilot = self.development_usage(value)
        return {
            'capture_attempts': len(value['attempts']) - initial['capture_attempts']
                - sum(bool(r.get('engineering_authority')) for r in value['attempts']) - pilot['capture_attempts'],
            'logical_requests': value['logical_requests'] - initial['logical_requests']
                - value.get('preprogramme_engineering_requests', 0) - pilot['logical_requests'],
            'live_seconds': sum(r['charged_seconds'] for r in value['launches'].values()
                                if not r.get('engineering_authority')) - initial['live_seconds'] - pilot['live_seconds'],
        }

    def development_usage(self, value, day=None):
        rows = [*value['attempts'], *value['launches'].values()]
        tagged = [r for r in rows if 'development_authority_sha256' in r or 'development_slot' in r]
        usage = value.get('development_request_usage', {})
        if (tagged or usage) and not self.pilot_authority:
            raise ValueError('unrecognized_development_consumption')
        for row in tagged:
            if (row.get('development_authority_sha256') != self.pilot_ref['sha256']
                    or row.get('development_allocation_id') != self.pilot_authority['allocation_id']
                    or row.get('development_slot') not in self.pilot_authority['dates']):
                raise ValueError('invalid_development_consumption')
        if usage and value.get('development_request_authority_sha256') != self.pilot_ref['sha256']:
            raise ValueError('invalid_development_request_authority')
        for key, counts in usage.items():
            if (key not in self.pilot_authority['dates'] or set(counts) != {'prediction', 'results'}
                    or any(type(x) is not int or x < 0 for x in counts.values())):
                raise ValueError('invalid_development_request_consumption')
        matches = lambda r: bool(r.get('development_authority_sha256')) and (day is None or r['development_slot'] == day)
        return dict(capture_attempts=sum(matches(r) for r in value['attempts']),
            live_seconds=sum(r['charged_seconds'] for r in value['launches'].values() if matches(r)),
            prediction=sum(r['prediction'] for k,r in usage.items() if day is None or k == day),
            results=sum(r['results'] for k,r in usage.items() if day is None or k == day),
            logical_requests=sum(sum(r.values()) for k,r in usage.items() if day is None or k == day))

    def development_day(self, now=None, *, kind='prediction'):
        current = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo('Australia/Melbourne'))
        if kind == 'results':
            if not (datetime.fromisoformat('2026-10-03T13:00:00+10:00') <= current
                    < datetime.fromisoformat(self.development['result_closure_at'])):
                raise ValueError('development_results_window_closed')
            # Results are a separate cumulative allowance; assign to a fixed
            # accounting bucket, never a newly fabricated capture slot.
            return self.development['dates'][0]
        day = current.date().isoformat()
        if (day not in self.development['dates'] or not
                current.replace(hour=12, minute=40, second=0, microsecond=0) <= current <
                current.replace(hour=14, minute=40, second=0, microsecond=0)):
            raise ValueError('development_capture_window_closed')
        return day

    def development_tags(self, day):
        return dict(development_allocation_id=self.development['allocation_id'],
                    development_authority_sha256=self.development_authority['sha256'], development_slot=day)

    def require_development_member(self, item):
        from race_collection.development_pilot import selected_population
        day = self.development_day()
        rows = selected_population(Path(self.development['state_root']) / 'sessions' / day,
                                   self.development['allocation_sha256'])
        candidates = [r for r in rows if r['race_id'] == item['race_id']]
        identity = item.get('race_identity', {})
        if (len(candidates) != 1 or candidates[0]['url'] != identity.get('race_url')
                or candidates[0]['jump_at'] != identity.get('jump_datetime')
                or str(candidates[0]['source_native_race_id']) != str(identity.get('source_native_race_id'))):
            raise ValueError('development_race_not_frozen_member')
        if candidates[0]['runners'] != item.get('development_active_runners'):
            raise ValueError('development_selected_field_changed')
        current = datetime.now(timezone.utc).astimezone(ZoneInfo('Australia/Melbourne'))
        if not current.replace(hour=13,minute=0,second=0,microsecond=0) <= current < current.replace(hour=14,minute=30,second=0,microsecond=0):
            raise ValueError('development_win_capture_not_open')
        return day

    def check_programme_time(self):
        if self.development:
            self.development_day()
        if self.engineering_authority and self.study_programme:
            if datetime.now(timezone.utc) >= datetime.fromisoformat(self.study_programme['starts_at']):
                raise ValueError('engineering_window_overlaps_programme')
        if self.programme:
            now = datetime.now(timezone.utc)
            if not datetime.fromisoformat(self.programme['starts_at']) <= now < datetime.fromisoformat(self.programme['expires_at']):
                raise ValueError('persistent_programme_expired_or_not_started')

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
            if value.get('source_holds'):
                raise ValueError('campaign_source_hold')
            row = value['launches'].get(launch)
            if row is None or row.get('closed_at') or now.timestamp() >= row['deadline_epoch']:
                raise ValueError('campaign_live_lease_closed')

    def begin(self, launch, *, now, deadline):
        self.check_programme_time()
        if self.development:
            day = self.development_day(now)
            if deadline > now.astimezone(ZoneInfo('Australia/Melbourne')).replace(hour=14, minute=40, second=0, microsecond=0):
                raise ValueError('development_cleanup_deadline_exceeded')
        if (self.engineering_authority and self.study_programme
                and deadline >= datetime.fromisoformat(self.study_programme['starts_at'])):
            raise ValueError('engineering_window_overlaps_programme')
        with self.ledger() as value:
            if value.get('source_holds'):
                raise ValueError('campaign_source_hold')
            if launch in value['launches'] or any(not r.get('closed_at') for r in value['launches'].values()):
                raise ValueError('campaign_owner_or_launch_already_exists')
            used = (self.development_usage(value)['live_seconds'] if self.development else
                    sum(r['charged_seconds'] for r in value['launches'].values()) - self.development_usage(value)['live_seconds'])
            charge = (deadline - now).total_seconds()
            if (charge <= 0 or used + charge > self.value['max_live_seconds']
                    or self.programme and self.programme_usage(value)['live_seconds']+charge>580800):
                raise ValueError('campaign_live_time_exhausted')
            if self.development and (self.development_usage(value, day)['live_seconds'] + charge > 7200
                    or any(r.get('development_slot') == day for r in value['launches'].values())):
                raise ValueError('development_slot_consumed')
            value['launches'][launch] = dict(started_at=now.isoformat(),
                deadline_epoch=deadline.timestamp(), charged_seconds=charge)
            if self.development:
                value['launches'][launch].update(self.development_tags(day))
            if self.engineering_authority and self.study_programme:
                value['launches'][launch]['engineering_authority'] = self.engineering_authority

    def close(self, launch, *, now):
        # Only after verified restoration. Unclosed launches keep their full charge.
        with self.ledger() as value:
            row = value['launches'][launch]
            if not row.get('closed_at'):
                row['charged_seconds'] = max(0, (now - datetime.fromisoformat(row['started_at'])).total_seconds())
                row['closed_at'] = now.isoformat()

    def available(self):
        with self.ledger() as value:
            if self.development:
                return (self.development_usage(value)['capture_attempts'] < 24
                        and self.development_usage(value, self.development_day())['capture_attempts'] < 6)
            return (len(value['attempts']) - self.development_usage(value)['capture_attempts'] < self.value['max_capture_attempts'] and
                    (not self.programme or self.programme_usage(value)['capture_attempts']<1000))

    def consume(self, claim, item):
        self.check_programme_time()
        if self.development:
            day = self.require_development_member(item)
        with self.ledger() as value:
            if value.get('source_holds'):
                raise ValueError('campaign_source_hold')
            if self.development and (self.development_usage(value)['capture_attempts'] >= 24
                    or self.development_usage(value, day)['capture_attempts'] >= 6):
                raise ValueError('development_capture_allowance_consumed')
            if (not self.development and len(value['attempts']) - self.development_usage(value)['capture_attempts'] >= self.value['max_capture_attempts']
                    or self.programme and self.programme_usage(value)['capture_attempts']>=1000):
                raise ValueError('campaign_capture_allowance_consumed')
            aliases = set(item.get('race_id_aliases', [item['race_id']])) | {item['race_id']}
            if any(r['window'] == item['capture_window_minutes'] and aliases.intersection(r['aliases'])
                   for r in value['attempts']):
                raise ValueError('campaign_capture_window_consumed')
            value['attempts'].append(dict(claim=str(claim), race_id=item['race_id'],
                aliases=sorted(aliases), window=item['capture_window_minutes'], item=item,
                consumed_at=datetime.now(timezone.utc).isoformat()))
            if self.development:
                value['attempts'][-1].update(self.development_tags(day))
            if self.engineering_authority and self.study_programme:
                value['attempts'][-1]['engineering_authority'] = self.engineering_authority

    def development_request(self, kind):
        if kind not in {'prediction', 'results'}:
            raise ValueError('unknown_development_request_kind')
        day = self.development_day(kind=kind)
        with self.ledger() as value:
            if value.get('source_holds'):
                raise ValueError('campaign_source_hold')
            total, daily = self.development_usage(value), self.development_usage(value, day)
            if (total[kind] >= (24000 if kind == 'prediction' else 720)
                    or kind == 'prediction' and daily[kind] >= 6000):
                raise ValueError('development_request_cap_exhausted')
            value['development_request_authority_sha256'] = self.development_authority['sha256']
            value.setdefault('development_request_usage', {}).setdefault(day, {'prediction':0, 'results':0})[kind] += 1
            value['logical_requests'] += 1

    def request(self, *, kind='prediction'):
        if self.development:
            return self.development_request(kind)
        self.check_programme_time()
        if self.engineering_authority and kind != 'prediction':
            raise ValueError('engineering_results_forbidden')
        with self.ledger() as value:
            if value.get('source_holds'):
                raise ValueError('campaign_source_hold')
            if (value['logical_requests'] - self.development_usage(value)['logical_requests'] >= self.value['max_logical_requests']
                    or self.programme and self.programme_usage(value)['logical_requests']>=1304000):
                raise ValueError('campaign_request_cap_exhausted')
            if self.programme:
                ceilings={'prediction':1280000,'results':24000}
                if kind not in ceilings:raise ValueError('unknown_programme_request_kind')
                usage=value.setdefault('persistent_request_usage',{'prediction':0,'results':0})
                if usage[kind]>=ceilings[kind]:raise ValueError('programme_kind_request_cap_exhausted')
                usage[kind]+=1
            value['logical_requests'] += 1
            if self.engineering_authority and self.study_programme:
                value['preprogramme_engineering_requests'] = value.get('preprogramme_engineering_requests', 0) + 1

    def hold_source(self, observation):
        """A denial or retry instruction survives package/process replacement.

        Elapsed time alone never authorizes an unchanged retry. A subsequent
        recovery needs a separately reviewed prospective disposition.
        """
        with self.ledger() as value:
            value.setdefault('source_holds', []).append(observation)
