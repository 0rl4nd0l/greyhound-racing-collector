"""Immutable daily discovery metadata, separate from freshly acquired race inputs."""
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit

SCHEMA = 'daily_race_inventory_v1'
MAX_BYTES = 16 * 1024 * 1024


class InventoryError(ValueError):
    """A pinned inventory cannot safely be used for selection."""


def _stamp(value):
    try:
        value = value if isinstance(value, datetime) else datetime.fromisoformat(value)
        if value.utcoffset() is None:
            raise ValueError
        return value
    except (TypeError, ValueError, AttributeError) as exc:
        raise InventoryError('INVENTORY_TIMESTAMP_INVALID') from exc


def _invalid_json_constant(_value):
    raise ValueError('nonfinite_json_number')


def _date(value):
    try:
        result = date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise InventoryError('INVENTORY_SOURCE_DATE_INVALID') from exc
    return result.isoformat()


def _validate(value, source_date):
    if not isinstance(value, dict) or value.get('schema_version') != SCHEMA:
        raise InventoryError('INVENTORY_SCHEMA_INVALID')
    if value.get('source_date') != _date(source_date):
        raise InventoryError('INVENTORY_SOURCE_DATE_MISMATCH')
    if value.get('status') != 'COMPLETE' or value.get('discovery_failures') != []:
        raise InventoryError('INVENTORY_INCOMPLETE_DISCOVERY')
    _stamp(value.get('observed_at'))
    races = value.get('races')
    if not isinstance(races, list) or not races:
        # An empty response needs explicit no-meeting evidence, which the legacy
        # browser does not supply. It cannot prove a complete empty racing day.
        raise InventoryError('INVENTORY_EMPTY_UNVERIFIED')
    if type(value.get('race_count')) is not int or value['race_count'] != len(races):
        raise InventoryError('INVENTORY_RACE_COUNT_MISMATCH')
    identities = set()
    for race in races:
        if not isinstance(race, dict) or race.get('date') != source_date:
            raise InventoryError('INVENTORY_RACE_SOURCE_DATE_MISMATCH')
        try:
            parsed = urlsplit(str(race.get('url', '')))
        except ValueError as exc:
            raise InventoryError('INVENTORY_RACE_IDENTITY_INVALID') from exc
        match = re.match(r'^/racing/([^/]+)/([^/]+)/(\d+)(?:/|$)', parsed.path, re.I)
        if (parsed.scheme != 'https' or parsed.hostname != 'www.thedogs.com.au'
                or parsed.username or parsed.password or parsed.fragment or not match):
            raise InventoryError('INVENTORY_RACE_IDENTITY_INVALID')
        venue, day, number = match.groups()
        if day != source_date or str(race.get('race_number')) != str(int(number)):
            raise InventoryError('INVENTORY_RACE_SOURCE_DATE_OR_NUMBER_MISMATCH')
        identity = (venue.lower(), day, int(number))
        if identity in identities:
            raise InventoryError('INVENTORY_DUPLICATE_RACE_IDENTITY')
        identities.add(identity)
        if race.get('scheduled_jump_datetime'):
            _stamp(race['scheduled_jump_datetime'])
        # Missing/unresolved times remain in the inventory; native selection
        # must exclude them, rather than silently dropping discovered races.
    return value


def write_daily_inventory(path, *, races, source_date, observed_at, discovery_failures=()):
    """Seal one complete discovery as a new immutable file; never overwrite it."""
    source_date = _date(source_date)
    value = {'schema_version': SCHEMA, 'status': 'COMPLETE',
             'source_date': source_date, 'observed_at': _stamp(observed_at).isoformat(),
             'races': list(races), 'discovery_failures': list(discovery_failures)}
    value['race_count'] = len(value['races'])
    _validate(value, source_date)
    try:
        body = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
    except (TypeError, ValueError) as exc:
        raise InventoryError('INVENTORY_ENCODING_INVALID') from exc
    if len(body) > MAX_BYTES:
        raise InventoryError('INVENTORY_TOO_LARGE')
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.inventory-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o400)
        os.link(temporary, path)  # create-only publication, including under races
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink()
    return {'path': str(path), 'sha256': hashlib.sha256(body).hexdigest()}


def load_daily_inventory(path, sha256, *, source_date, now, max_age_seconds=900):
    """Authenticate the exact complete inventory and enforce finite source age."""
    if (isinstance(max_age_seconds, bool) or not isinstance(max_age_seconds, (int, float))
            or not math.isfinite(max_age_seconds) or not 0 < max_age_seconds <= 1800):
        raise InventoryError('INVENTORY_MAX_AGE_INVALID')
    if not isinstance(sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', sha256):
        raise InventoryError('INVENTORY_HASH_INVALID')
    try:
        with Path(path).open('rb') as stream:
            body = stream.read(MAX_BYTES + 1)
    except OSError as exc:
        raise InventoryError('INVENTORY_UNREADABLE') from exc
    if len(body) > MAX_BYTES:
        raise InventoryError('INVENTORY_TOO_LARGE')
    if hashlib.sha256(body).hexdigest() != sha256:
        raise InventoryError('INVENTORY_HASH_MISMATCH')
    try:
        value = json.loads(body, parse_constant=_invalid_json_constant)
    except (ValueError, UnicodeError) as exc:
        raise InventoryError('INVENTORY_ENCODING_INVALID') from exc
    _validate(value, _date(source_date))
    age = (_stamp(now) - _stamp(value['observed_at'])).total_seconds()
    if age < 0:
        raise InventoryError('INVENTORY_FUTURE_OBSERVATION')
    if age > max_age_seconds:
        raise InventoryError('INVENTORY_STALE')
    return value


def discover_daily_inventory(browser, *, source_date, path, observed_at=None):
    """Use the caller's guarded source session for explicit all-day discovery.

    The caller owns request allowances, source admission and denial handling.
    Observation time starts before acquisition, conservatively preserving age.
    """
    source_date = _date(source_date)
    observed_at = _stamp(observed_at or datetime.now(timezone.utc))
    races = browser.get_races_for_date(date.fromisoformat(source_date))
    return write_daily_inventory(path, races=races, source_date=source_date,
                                 observed_at=observed_at,
                                 discovery_failures=getattr(browser, 'discovery_failures', []))


def add_inventory_arguments(parser):
    """Share the pinned inventory flags across native acquisition entrypoints."""
    parser.add_argument('--discovery-inventory', type=Path)
    parser.add_argument('--discovery-inventory-sha256')
    parser.add_argument('--discovery-inventory-source-date')
    parser.add_argument('--discovery-inventory-max-age-seconds', type=float, default=900)


def inventory_cli_args(args):
    """Forward a complete optional reference, without silently losing its pin."""
    path = getattr(args, 'discovery_inventory', None)
    sha = getattr(args, 'discovery_inventory_sha256', None)
    if not path and not sha:
        if getattr(args, 'discovery_inventory_source_date', None):
            raise InventoryError('INVENTORY_REFERENCE_INCOMPLETE')
        return []
    if not path or not sha:
        raise InventoryError('INVENTORY_REFERENCE_INCOMPLETE')
    result = ['--discovery-inventory', str(path), '--discovery-inventory-sha256', sha,
              '--discovery-inventory-max-age-seconds',
              str(getattr(args, 'discovery_inventory_max_age_seconds', 900))]
    if getattr(args, 'discovery_inventory_source_date', None):
        result += ['--discovery-inventory-source-date', args.discovery_inventory_source_date]
    return result
