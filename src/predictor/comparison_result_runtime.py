"""Private, bounded transport for authenticated comparison members only.

Uses the collector's cumulative campaign and no-steal lock. No alternate source,
redirect, browser, automatic transport retry or public outcome reporting.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace
import uuid

from race_collection.freshness_campaign import Campaign
from race_collection.live_phase_checkpoint import atomic_json
from race_collection.synchronous_manual_capture import acquire_collector_lock_no_steal, release_owned_collector_lock
from src.predictor.comparison_result_scope import result_scope
from src.predictor.future_comparison import checked, stamp

ACTIVE = ContextVar('comparison_result_transport', default=None)
MAX_BODY = 1024 * 1024


def load_runtime(binding, *, now, allow_closure=False):
    from src.predictor.future_comparison import load_plan
    plan, _ = load_plan(Path(binding['plan']), binding['plan_sha256'])
    authority = json.loads(checked(Path(binding['authority']), binding['authority_sha256']))
    cfg = authority['runtime']
    from race_collection.incident_comparison import result_deadline
    deadline = result_deadline(plan)
    if plan['status'] == 'AUTHORIZED_ENGINEERING':
        from race_collection.incident_comparison import validate_incident_plan
        incident = validate_incident_plan(plan)
        if (any(cfg.get(key) != plan[key] for key in ('incident_authority', 'incident_slot'))
                or cfg['max_races'] != 24 or cfg['max_requests'] != 72
                or cfg['max_attempts_per_race'] != 3
                or Path(cfg['state_root']) != Path(incident['result_root']) / plan['incident_slot']
                or Path(cfg['prediction_bundles']) != Path(incident['prediction_root']) / 'bundles'
                or Path(cfg['job_store']) != Path(incident['prediction_root']) / 'jobs.sqlite3'):
            raise ValueError('incident_result_runtime_mismatch')
    # Closure has no acquisition or target decoding. It remains owed after expiry.
    scope_now = min(now, deadline) if allow_closure else now
    result_scope(binding, now=scope_now, prediction_bundles=Path(cfg['prediction_bundles']),
                 result_database=Path(authority['result_database']))
    if any(type(cfg.get(k)) is not int for k in ('max_races','max_requests','max_attempts_per_race','races_per_cycle','max_storage_bytes')):
        raise ValueError('result_budget_type_invalid')
    if (cfg.get('schema_version') != 'comparison_result_runtime_v1'
            or not 1 <= cfg['max_races'] <= 2000
            or not 1 <= cfg['max_requests'] <= 48000
            or not 1 <= cfg['max_attempts_per_race'] <= 24
            or not 1 <= cfg['races_per_cycle'] <= 8
            or not 2**30 <= cfg['max_storage_bytes'] <= 64 * 2**30
            or stamp(cfg['expires_at']) != deadline
            or any(not Path(cfg[k]).is_absolute() for k in
                   ('state_root', 'prediction_bundles', 'job_store', 'campaign_root', 'lock_path'))):
        raise ValueError('result_runtime_invalid')
    root = Path(cfg['state_root'])
    if root.resolve() != root or Path(authority['result_database']).parent != root:
        raise ValueError('result_storage_not_private_bound_root')
    from race_collection.persistent_storage import check_mount
    check_mount(cfg['storage_mount'], root)
    if (Path(authority['result_database']) == Path(cfg['job_store']) or root.is_relative_to(Path(cfg['prediction_bundles']))):
        raise ValueError('result_database_collision')
    return plan, authority, cfg


def storage_check(root, cfg):
    size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if size + 2 * MAX_BODY > cfg['max_storage_bytes'] or shutil.disk_usage(root).free < 2 * 2**30:
        raise ValueError('RESULT_DISK_PRESSURE')


def database(root):
    db = sqlite3.connect(root / 'queue.sqlite3', timeout=2)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA synchronous=FULL')
    db.executescript('''
      CREATE TABLE IF NOT EXISTS identity(binding TEXT PRIMARY KEY);
      CREATE TABLE IF NOT EXISTS jobs(race TEXT PRIMARY KEY, job TEXT UNIQUE, jump TEXT,
        state TEXT NOT NULL, due TEXT, attempts INTEGER NOT NULL DEFAULT 0);
      CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, at TEXT, race TEXT, status TEXT, artifact TEXT);
      CREATE TABLE IF NOT EXISTS requests(id INTEGER PRIMARY KEY, at TEXT, race TEXT, artifact TEXT);
    ''')
    return db


class Transport:
    def __init__(self, binding, cfg, root, output, campaign):
        self.binding, self.cfg, self.root = binding, cfg, root
        self.output, self.campaign = output, campaign
        self.allowed = set()
        self.race = None

    def select(self, candidate):
        self.race = candidate.race_id
        self.allowed = {candidate.canonical_thedogs_url + '?trial=false'}

    def get(self, session, url, **kwargs):
        now = datetime.now(timezone.utc)
        load_runtime(self.binding, now=now)
        if url not in self.allowed or not self.race:
            raise ValueError('result_url_not_exact_admitted_race')
        storage_check(self.root, self.cfg)
        from utils.sportsbet_access import SportsbetAccess
        if SportsbetAccess(self.cfg['source_state']).blocks_restoration():
            raise ValueError('RESULT_SHARED_SOURCE_HOLD')
        artifact = self.output / ('response-' + uuid.uuid4().hex)
        # Charge before transport; uncertain attempts remain consumed on a crash.
        with database(self.root) as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT count(*) FROM requests').fetchone()[0] >= self.cfg['max_requests']:
                raise ValueError('RESULT_REQUEST_BUDGET')
            row = db.execute('SELECT state FROM jobs WHERE race=?', (self.race,)).fetchone()
            if row is None or row['state'] != 'RUNNING':
                raise ValueError('result_queue_attempt_required')
            if db.execute('SELECT count(*) FROM requests WHERE race=?', (self.race,)).fetchone()[0] >= self.cfg['max_attempts_per_race']:
                raise ValueError('RESULT_RACE_REQUEST_BUDGET')
            self.campaign.request(kind='results')  # shared holds/counters; never reset or bypass
            db.execute('UPDATE jobs SET attempts=attempts+1 WHERE race=?', (self.race,))
            db.execute('INSERT INTO requests(at,race,artifact) VALUES(?,?,?)',
                       (now.isoformat(), self.race, str(artifact)))
        atomic_json(artifact.with_suffix('.request.json'), {'at': now.isoformat(), 'url': url})
        # requests.Session default adapter has zero retries. Redirects are evidence,
        # not permission to fetch an unadmitted page or discover other races.
        response = session.get(url, headers={**kwargs.get('headers', {}), 'Accept-Encoding': 'identity'},
                               timeout=(5, 20), allow_redirects=False, stream=True)
        try:
            from utils.http_client import source_retry_headers
            guidance = source_retry_headers(response.headers)
            observation = {'host': 'www.thedogs.com.au', 'status': response.status_code,
                           'observed_at': now.isoformat(), 'retry_headers': guidance}
            denied = response.status_code in (401, 403, 429) or any(
                k in guidance for k in ('retry-after', 'ratelimit-reset', 'x-ratelimit-reset'))
            if denied:
                self.campaign.hold_source(observation)
            body = response.raw.read(MAX_BODY + 1, decode_content=False)
            # Original bytes (including bounded rejected responses) stay private.
            with artifact.with_suffix('.body').open('xb') as stream:
                stream.write(body); stream.flush(); os.fsync(stream.fileno())
            atomic_json(artifact.with_suffix('.json'), {**observation,
                'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body),
                'final_url': response.url, 'content_type': response.headers.get('Content-Type')})
            if denied:
                atomic_json(self.output / 'transport-status.json', {'status': 'SOURCE_HOLD'})
                raise ValueError('RESULT_SOURCE_HOLD')
            if (len(body) > MAX_BODY or response.url != url or 300 <= response.status_code < 400
                    or response.headers.get('Content-Encoding', 'identity').lower() not in ('', 'identity')
                    or (response.status_code==200 and not response.headers.get('Content-Type','').lower().startswith('text/html'))):
                atomic_json(self.output / 'transport-status.json', {'status': 'QUARANTINED_ENVELOPE'})
                raise ValueError('RESULT_SOURCE_ENVELOPE')
            # Detect denial HTML too; a 200 challenge is not a pending result.
            from scripts.ingest_results_for_date import response_is_forbidden, title_from_html, rendered_text_from_html
            text = body.decode('utf-8', errors='strict')
            if response_is_forbidden(response.status_code, title_from_html(text), rendered_text_from_html(text)):
                self.campaign.hold_source({**observation, 'reason': 'html_denial'})
                atomic_json(self.output / 'transport-status.json', {'status': 'SOURCE_HOLD'})
                raise ValueError('RESULT_SOURCE_HOLD')
            return SimpleNamespace(text=text, status_code=response.status_code, url=url, close=lambda: None)
        finally:
            response.close()


@contextmanager
def collector_guard(binding, *, output, job_store, bundles, result_database):
    os.umask(0o077)
    _, authority, cfg = load_runtime(binding, now=datetime.now(timezone.utc))
    if (Path(cfg['job_store']) != job_store.absolute() or Path(cfg['prediction_bundles']) != bundles.absolute()
            or Path(authority['result_database']) != result_database.absolute()):
        raise ValueError('result_runtime_path_mismatch')
    root = Path(cfg['state_root']); root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if output.resolve().parent != root / 'attempts':
        raise ValueError('result_attempt_not_private')
    campaign = Campaign(cfg['campaign_root'], **{key: cfg[key] for key in
        ('incident_authority', 'incident_slot') if key in cfg})
    with (campaign.root / 'owner.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Existing owner blocks before any transport. No stale-lock deletion.
        owned = acquire_collector_lock_no_steal(Path(cfg['lock_path']), run_id=output.name,
                    output_dir=output, phase='comparison_result_retention')
        token = ACTIVE.set(Transport(binding, cfg, root, output, campaign))
        try:
            yield
        finally:
            ACTIVE.reset(token)
            release_owned_collector_lock(owned)
