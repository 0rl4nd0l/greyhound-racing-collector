"""Optional explicitly authorized structural webhook; no target data in payloads."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

from race_collection.live_phase_checkpoint import atomic_json


def deliver(value, config_path, state_path, *, now=None, post=None):
    if not config_path or not Path(config_path).exists():
        return 'LOCAL_ONLY_NO_DESTINATION'
    now = now or datetime.now(timezone.utc)
    cfg = json.loads(Path(config_path).read_bytes())
    if cfg.get('status') != 'AUTHORIZED_OPERATIONAL_ALERTS' or not cfg.get('authority_reference'):
        raise ValueError('notification_authority_missing')
    secret = Path(cfg['endpoint_file'])
    if not secret.is_absolute() or secret.stat().st_mode & 0o077:
        raise ValueError('notification_endpoint_must_be_private')
    endpoint = secret.read_text().strip()
    parsed = urlparse(endpoint)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('notification_https_required')
    # This deliberately never serializes the full monitor object.
    payload = {'service': 'greyhound-comparison', 'status': value['status'],
               'alerts': value.get('alerts', []), 'outcomes_released': False}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    previous = json.loads(state_path.read_bytes()) if state_path.exists() else {}
    if previous.get('fingerprint') == fingerprint and previous.get('delivered'):
        return 'DELIVERED_UNCHANGED'
    if previous.get('attempted_at') and now - datetime.fromisoformat(previous['attempted_at']) < timedelta(minutes=15):
        return 'DELIVERY_RETRY_PENDING'
    if value['status'] == 'HEALTHY' and not previous:
        return 'CONFIGURED_NO_ALERT_SENT'
    state = {'attempted_at': now.isoformat(), 'fingerprint': fingerprint, 'delivered': False}
    atomic_json(state_path, state)  # consume the attempt before transport
    try:
        if post is None:
            import requests
            post = requests.post
        response = post(endpoint, json=payload, timeout=(3, 5), allow_redirects=False, stream=True)
        try:
            state['delivered'] = 200 <= response.status_code < 300
        finally:
            response.close()
    except Exception:
        pass  # no endpoint, credentials or remote body in journal/monitor
    atomic_json(state_path, state)
    return 'DELIVERED' if state['delivered'] else 'DELIVERY_FAILED'
