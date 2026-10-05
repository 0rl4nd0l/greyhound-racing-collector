"""Bounded, outcome-blind timing presence census of a fixed retained-card manifest."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import time
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

from scripts.build_form_only_v1_packet import (
    canonical_roster, dog_token, parse_card_target_roster_bytes, parse_form_blocks_bytes,
)

FIELDS = ('TIME', 'WIN', 'BON', '1 SEC')
COLUMNS = ('Dog Name', 'DATE', 'TRACK', 'DIST', *FIELDS, 'PIR')
MISSING = {'', '-', '--', 'N/A', 'NA', 'NULL', 'NONE'}
MAX_READS = 1150
MAX_BYTES = 38_189_496
MAX_FILE = 8 * 1024 * 1024
MAX_SECONDS = 300
MAX_OUTPUT = 4 * 1024 * 1024
MEMBERS = 82
ROLES = ('bundle_manifest', 'admission', 'sidecar', 'accepted_csv', 'raw_export', 'primary_page', 'primary_receipt')


class CoverageRejected(ValueError):
    """A fixed safe category, never an arbitrary source payload."""


def require(condition, code):
    if not condition:
        raise CoverageRejected(code)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def instant(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'TIMESTAMP_UNAWARE')
    return parsed


class Reader:
    def __init__(self):
        self.started = time.monotonic()
        self.reads = self.bytes = 0

    def check(self):
        require(time.monotonic() - self.started <= MAX_SECONDS, 'WALL_LIMIT')

    def read(self, reference):
        self.check()
        path = Path(reference['path'])
        require(path.is_absolute() and path.resolve() == path, 'UNSAFE_PATH')
        require(re.fullmatch('[a-f0-9]{64}', reference['sha256']) is not None, 'HASH_FORMAT')
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, 'rb') as handle:
            before = os.fstat(handle.fileno())
            require(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_FILE, 'FILE_LIMIT')
            self.reads += 1
            self.bytes += before.st_size
            require(self.reads <= MAX_READS and self.bytes <= MAX_BYTES, 'READ_LIMIT')
            data = handle.read(MAX_FILE + 1)
            after = os.fstat(handle.fileno())
        identity = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        require(path.resolve() == path and identity(before) == identity(after) == identity(path.stat()), 'INPUT_CHANGED')
        require(len(data) == before.st_size and hashlib.sha256(data).hexdigest() == reference['sha256'], 'INPUT_HASH')
        require(reference.get('bytes', len(data)) == len(data), 'INPUT_SIZE')
        self.check()
        return data

    def json(self, reference):
        return json.loads(self.read(reference))


def projected_card(payload):
    """Preserve native block/roster parsing; remove uninterpreted columns first."""
    text = payload.decode('utf-8-sig', errors='strict')
    first = text.splitlines()[0] if text else ''
    delimiter = '|' if first.count('|') > first.count(',') else ','
    rows = csv.reader(io.StringIO(text), delimiter=delimiter)
    header = next(rows, [])
    require(len(header) == len(set(header)) and set(COLUMNS).issubset(header), 'CSV_SCHEMA')
    indices = [header.index(name) for name in COLUMNS]
    out = io.StringIO()
    writer = csv.writer(out, delimiter='|', lineterminator='\n')
    writer.writerow(COLUMNS)
    for row in rows:
        if not row:
            continue
        require(len(row) == len(header), 'CSV_ROW_WIDTH')
        writer.writerow([row[i] for i in indices])
    return out.getvalue().encode()


def number_presence(value):
    value = str(value or '').strip()
    if value.upper() in MISSING:
        return 'missing'
    try:
        number = float(value)
    except ValueError:
        return 'non_numeric'
    if not math.isfinite(number):
        return 'non_finite'
    return 'positive_numeric' if number > 0 else 'nonpositive_numeric'


def distance(value):
    match = re.fullmatch(r'\s*([1-9][0-9]*)\s*m?\s*', str(value))
    return int(match.group(1)) if match else None


def audit_card(payload, sidecar, member):
    projected = projected_card(payload)
    roster = parse_card_target_roster_bytes(projected, source='sealed retained card')
    complete = sidecar['runner_completeness_after_canonical_alignment']
    expected = canonical_roster(complete['participants'], box_key='box_number', name_key='dog_name', source='sealed sidecar')
    require(roster == expected and len(roster) == complete['runner_count'] == member['target_runner_slots'], 'TARGET_ROSTER_MISMATCH')
    blocks = parse_form_blocks_bytes(projected, source='sealed retained card')
    require(set(blocks) == {name for _, name in roster}, 'TARGET_BLOCK_MISMATCH')
    target_date = date.fromisoformat(member['source_race_date'])
    target_distance = distance(member['target_distance_raw'])
    require(target_distance is not None, 'TARGET_DISTANCE_INVALID')
    summaries = []
    for box, name in roster:
        reasons = Counter()
        candidates = []
        seen = set()
        groups = defaultdict(list)
        for row in blocks[name]:
            # DATE-only histories cannot establish ordering within target racing day.
            try:
                history_date = date.fromisoformat(row['DATE'].strip())
            except ValueError:
                reasons['INVALID_OR_MISSING_DATE'] += 1
                continue
            if history_date >= target_date:
                reasons['SAME_DAY_OR_LATER_EXCLUDED'] += 1
                continue
            track = row['TRACK'].strip()
            dist = distance(row['DIST'])
            if not track or dist is None:
                reasons['HISTORY_IDENTITY_INCOMPLETE'] += 1
                continue
            identity = (row['DATE'].strip(), track, dist)
            # Opaque fingerprints never make these source keys native event IDs.
            fingerprint = digest({k: row[k].strip() for k in COLUMNS if k != 'Dog Name'})
            if fingerprint in seen:
                reasons['EXACT_WHITELIST_DUPLICATE'] += 1
                continue
            seen.add(fingerprint)
            groups[identity].append((fingerprint, row, dist))
        for group in groups.values():
            if len(group) > 1:
                reasons['AMBIGUOUS_EVENT_KEY_CONFLICT'] += len(group)
            else:
                candidates.extend(group)
        presence = {f: Counter() for f in FIELDS}
        comparable = {f: 0 for f in FIELDS}
        pir = Counter()
        for _, row, dist in candidates:
            for field in FIELDS:
                status = number_presence(row[field])
                presence[field][status] += 1
                if status == 'positive_numeric' and dist == target_distance and row['TRACK'].strip() == member['venue']:
                    comparable[field] += 1
            pir['missing' if row['PIR'].strip().upper() in MISSING else 'present_unqualified'] += 1
        summaries.append({'target_runner_identity_sha256': digest([member['race_id'], box, name]),
                          'historical_rows_seen': len(blocks[name]), 'retained_prior_rows': len(candidates),
                          'row_dispositions': dict(reasons), 'field_presence': {k: dict(v) for k, v in presence.items()},
                          'same_raw_track_and_distance_positive_rows': comparable, 'pir_presence': dict(pir)})
    return {'race_id': member['race_id'], 'venue': member['venue'], 'target_distance_raw': member['target_distance_raw'],
            'status': 'INPUT_COVERAGE_MEASURED_DEFINITIONS_UNQUALIFIED', 'target_runner_slots': len(roster),
            'runner_set_sha256': member['runner_set_sha256'], 'runners': summaries,
            'whole_field_at_least_one_positive': {f: all(r['field_presence'][f].get('positive_numeric', 0) > 0 for r in summaries) for f in FIELDS},
            'whole_field_same_raw_track_distance_at_least_one_positive': {f: all(r['same_raw_track_and_distance_positive_rows'][f] > 0 for r in summaries) for f in FIELDS}}


def verified_member(reader, member, original):
    require(member['race_id'] == original['race_id'] and member['runner_set_sha256'] == original['runner_set_sha256']
            and member['jump_at'] == original['jump_at'], 'MEMBERSHIP_BINDING')
    for role in ('bundle_manifest', 'admission'):
        require(all(member[role][k] == original[role][k] for k in ('path', 'sha256')), 'MEMBERSHIP_REFERENCE')
    raw = {role: reader.read(member[role]) for role in ROLES}
    bundle = json.loads(raw['bundle_manifest'])
    base = Path(member['bundle_manifest']['path']).parent
    for role in ('sidecar', 'accepted_csv'):
        relative = str(Path(member[role]['path']).relative_to(base))
        require(bundle['files'][relative]['sha256'] == member[role]['sha256'], 'BUNDLE_ROLE_BINDING')
    admission = json.loads(raw['admission'])
    race = admission['race']
    require(race['race_id'] == member['race_id'] and race['race_date'] == member['source_race_date']
            and race['venue'] == member['venue'], 'RACE_BINDING')
    sidecar = json.loads(raw['sidecar'])
    require(str(sidecar['target_distance']) == member['target_distance_raw'], 'DISTANCE_BINDING')
    require(sidecar['content_sha256'] == member['accepted_csv']['sha256']
            and sidecar['raw_content_sha256'] == member['raw_export']['sha256'], 'CSV_BINDING')
    page = sidecar['primary_race_page_evidence']
    require(page['body_sha256'] == member['primary_page']['sha256']
            and page['receipt_sha256'] == member['primary_receipt']['sha256'], 'PRIMARY_BINDING')
    receipt = json.loads(raw['primary_receipt'])
    require(receipt['body_sha256'] == member['primary_page']['sha256'] and receipt['status_code'] == 200
            and receipt['race_discovery_key'] == member['race_id']
            and receipt['requested_url'].split('?')[0] == sidecar['race_url'].split('?')[0]
            and instant(receipt['capture_timestamp']) < instant(member['jump_at']), 'PRIMARY_RECEIPT_INVALID')
    require(instant(original['original_admitted_at']) <= instant(original['original_published_complete_at']) < instant(member['jump_at']), 'PREJUMP_SEAL_BINDING')
    return audit_card(raw['accepted_csv'], sidecar, member)


def audit_manifest(reference):
    reader = Reader()
    manifest = reader.json(reference)
    require(manifest['schema_version'] == 'retained_speed_breadth_metadata_manifest_v1'
            and manifest['selection'] == 'ALL_82_FIXED_MEMBERS_IRRESPECTIVE_OF_RESULT_STATUS', 'MANIFEST_SCHEMA')
    membership = reader.json(manifest['membership'])
    members = manifest['members']
    originals = {m['race_id']: m for m in membership['members']}
    require(len(members) == len(membership['members']) == len(originals) == MEMBERS
            and len({m['race_id'] for m in members}) == MEMBERS
            and set(originals) == {m['race_id'] for m in members}, 'DENOMINATOR_CHANGED')
    records = [verified_member(reader, m, originals[m['race_id']]) for m in members]
    reader.check()
    return {'schema_version': 'retained_card_timing_coverage_v1', 'status': 'COMPLETE_PRESENCE_ONLY',
            'manifest': reference, 'cards': len(records), 'target_runner_slots': sum(r['target_runner_slots'] for r in records),
            'measurement_semantics': 'UNQUALIFIED', 'native_event_identity': 'UNQUALIFIED',
            'same_track_comparison': 'LITERAL_RAW_NAMESPACE_ONLY_NOT_VERIFIED_LAYOUT_OR_CLOCK',
            'source_requests': 0, 'result_requests': 0, 'evaluations': 0, 'training': 0,
            'reads': reader.reads, 'read_bytes': reader.bytes, 'records': records}


def run(reference, output):
    output = Path(output)
    require(output.is_absolute() and output.parent.resolve() == output.parent, 'OUTPUT_UNSAFE')
    output.mkdir(mode=0o700, exist_ok=False)
    began = time.monotonic()
    try:
        result = audit_manifest(reference)
        payload = (json.dumps(result, indent=2, sort_keys=True) + '\n').encode()
        require(len(payload) <= MAX_OUTPUT and time.monotonic() - began <= MAX_SECONDS, 'OUTPUT_LIMIT')
        path = output / '.coverage.pending.json'
        with path.open('xb') as handle:
            os.chmod(path, 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        require(time.monotonic() - began <= MAX_SECONDS, 'WALL_LIMIT')
        os.replace(path, output / 'coverage.json')
        return {'status': result['status'], 'cards': result['cards'], 'target_runner_slots': result['target_runner_slots'],
                'output_sha256': hashlib.sha256(payload).hexdigest(), 'measurement_semantics': 'UNQUALIFIED'}
    except Exception as exc:
        safe = str(exc) if isinstance(exc, CoverageRejected) else 'UNCLASSIFIED_INPUT_FAILURE'
        (output / 'FAILED.json').write_text(json.dumps({'status': 'FAILED_NO_SUCCESSFUL_COVERAGE', 'reason': safe}) + '\n')
        raise CoverageRejected(safe) from None
