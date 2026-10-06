"""Authenticate retained cards and project outcome-free speed observations.

The interface returns one global observation bank and all original target
packets. It never reads a result reference, broadens the fixed 82-card graph,
joins dog names across cards, or assumes configured venue aliases are proof.
"""
from collections import Counter
import csv
from datetime import date
import hashlib
import io
import json
import re

from bs4 import BeautifulSoup

from race_collection import retained_card_timing_coverage as coverage
from race_collection import retained_speed_features as strict
from utils.runner_completeness import extract_canonical_runner_set_from_html
from utils.thedogs_runner_identity import (
    TheDogsProfileIdentityMissing, TheDogsRunnerIdentityError,
    extract_thedogs_profile_identity,
)

require = coverage.require


class SnapshotReader:
    """Read each hash-bound source once through the existing bounded reader."""
    def __init__(self, reader=None):
        self.reader = reader if reader is not None else coverage.Reader()
        self.cache = {}

    def read(self, reference):
        key = (reference['path'], reference['sha256'])
        self.check()
        if key not in self.cache:
            self.cache[key] = self.reader.read(reference)
        data = self.cache[key]
        require(reference.get('bytes', len(data)) == len(data), 'INPUT_SIZE')
        return data

    def json(self, reference):
        return json.loads(self.read(reference))

    def check(self):
        self.reader.check()


def _profiles(page, sidecar, roster, receipt, *, race_binding=None):
    """Same-row entry-to-profile bridge, after complete native field checking."""
    markup = page.decode('utf-8', errors='strict')
    soup = BeautifulSoup(markup, 'html.parser')
    canonical = extract_canonical_runner_set_from_html(markup,
        source_url=sidecar['race_url'], extraction_timestamp=receipt['capture_timestamp'])
    participants = sidecar['runner_completeness_after_canonical_alignment']['participants']
    expected = {(r['box_number'], coverage.dog_token(r['dog_name']),
        str(r.get('source_native_runner_id') or '')) for r in participants}
    active = canonical['final_runner_participants']
    actual = {(r['box_number'], coverage.dog_token(r['dog_name']),
        str(r.get('source_native_runner_id') or '')) for r in active}
    require(canonical['canonical_runner_set_status'] == 'available'
        and actual == expected and len(actual) == len(roster)
        and {(b, n) for b, n, _ in actual} == set(roster)
        and all(entry for _, _, entry in actual), 'TARGET_NATIVE_ROSTER_MISMATCH')
    native_race = str(sidecar.get('source_native_race_id') or '')
    require(re.fullmatch(r'[1-9][0-9]*', native_race) is not None, 'TARGET_SEALED_NATIVE_RACE_INVALID')
    if (not soup.select('[data-race-id]')
            and canonical['source_native_race_id'] is None
            and canonical['native_identity_reasons'] == ['source_native_race_id_missing']):
        # verified_member has already authenticated the original bundle and
        # sidecar, including this page/receipt pair. Legacy source HTML need
        # not repeat the event ID obtained by that sealed original capture.
        require(isinstance(race_binding, dict) and {'race_id', 'jump_at'} <= race_binding.keys(),
            'TARGET_RACE_RECEIPT_BINDING_MISSING')
        page_sha = hashlib.sha256(page).hexdigest()
        require(receipt.get('status_code') == 200
            and receipt.get('race_discovery_key') == race_binding['race_id']
            and receipt.get('body_sha256') == page_sha
            and sidecar.get('primary_race_page_evidence', {}).get('body_sha256') == page_sha
            and receipt.get('requested_url', '').split('?')[0] == sidecar['race_url'].split('?')[0]
            and coverage.instant(receipt['capture_timestamp']) < coverage.instant(race_binding['jump_at']),
            'TARGET_RACE_RECEIPT_BINDING_INVALID')
        method = 'SEALED_NATIVE_RACE_AND_BOUND_PREJUMP_PAGE_RECEIPT'
    else:
        require(canonical['native_identity_status'] == 'available'
            and str(canonical['source_native_race_id']) == native_race,
            'TARGET_NATIVE_RACE_MISMATCH')
        method = 'MATCHING_NATIVE_HTML_AND_SEALED_RACE_ID'
    rows = {}
    required_entries = {entry for _, _, entry in expected}
    for row in soup.select('tr.race-runner'):
        entries = {str(e.get('data-runner-id') or '').strip() for e in row.select('[data-runner-id]')}
        if len(entries) == 1:
            entry = next(iter(entries))
            if entry in required_entries:
                require(entry not in rows, 'TARGET_NATIVE_ENTRY_DUPLICATE')
                rows[entry] = row
    profiles = {}
    for box, name, entry in sorted(expected):
        require(entry in rows, 'TARGET_NATIVE_ENTRY_MISSING')
        try:
            dog = extract_thedogs_profile_identity(rows[entry], require_profile_link=False)
            profiles[(box, name)] = (entry, 'thedogs:dog:' + dog, 'VERIFIED_SAME_ROW_PROFILE')
        except TheDogsProfileIdentityMissing:
            profiles[(box, name)] = (entry, None, 'PROFILE_ID_MISSING')
        except TheDogsRunnerIdentityError:
            # A contradictory profile never becomes a name-only identity join.
            profiles[(box, name)] = (entry, None, 'PROFILE_ID_CONFLICT')
    known = [identity for _, identity, _ in profiles.values() if identity is not None]
    require(len(known) == len(set(known)), 'TARGET_DOG_PROFILE_DUPLICATE')
    return profiles, method


def construct_member(reader, member, original):
    """Return a target packet plus original observation copies and audit counts."""
    coverage.verified_member(reader, member, original)
    payload = reader.read(member['accepted_csv'])
    sidecar = reader.json(member['sidecar'])
    admission = reader.json(member['admission'])
    receipt = reader.json(member['primary_receipt'])
    page = reader.read(member['primary_page'])
    text = payload.decode('utf-8-sig', errors='strict')
    delimiter = '|' if text.splitlines()[0].count('|') > text.splitlines()[0].count(',') else ','
    require(next(csv.reader(io.StringIO(text), delimiter=delimiter)) == strict.HEADER
        and member['csv_header'] == strict.HEADER, 'UNSUPPORTED_RETAINED_COLUMNS')
    require(not any(sidecar.get(key) for key in ('layout_id', 'layout_era', 'track_layout',
        'target_layout_id', 'target_layout_era', 'clock_convention', 'sectional_endpoint')),
        'SOURCE_CONTEXT_METADATA_REQUIRES_EXPLICIT_BINDING')
    # This fixed source contract records the source racing calendar's offset
    # in the authenticated jump timestamp. A UTC decision serialization must
    # not shift the source calendar date used by date-only history filters.
    jump = coverage.instant(member['jump_at'])
    cutoff = coverage.instant(admission['decision_at']).astimezone(jump.tzinfo)
    available = coverage.instant(original['original_published_complete_at'])
    require(coverage.instant(original['original_admitted_at']) <= available < cutoff
        < jump
        and admission['admitted_at'] == original['original_admitted_at']
        and coverage.instant(receipt['capture_timestamp']) <= available,
        'FEATURE_CAPTURE_NOT_BEFORE_CUTOFF')
    projected = coverage.projected_card(payload)
    roster = coverage.parse_card_target_roster_bytes(projected, source='authenticated retained card')
    blocks = coverage.parse_form_blocks_bytes(projected, source='authenticated retained card')
    profiles, race_identity_method = _profiles(page, sidecar, roster, receipt,
        race_binding={'race_id': member['race_id'], 'jump_at': member['jump_at']})
    runners, observations = [], []
    reasons, contexts, sections = Counter(), Counter(), Counter()
    target_date = date.fromisoformat(member['source_race_date'])
    target_distance = coverage.distance(member['target_distance_raw'])
    require(target_distance is not None, 'TARGET_DISTANCE_INVALID')
    for box, name in roster:
        entry, identity, status = profiles[(box, name)]
        runners.append({'runner_id': entry, 'identity_id': identity,
            'identity_available_at': available.isoformat() if identity else None,
            'identity_status': status, 'box_number': box,
            'strict_runner_id': coverage.digest([member['race_id'], box, name])})
        for index, row in enumerate(blocks[name]):
            raw_track = row['TRACK'].strip()
            distance = coverage.distance(row['DIST'])
            contexts[(raw_track, distance)] += 1
            sections[coverage.number_presence(row['1 SEC'])] += 1
            try:
                day = date.fromisoformat(row['DATE'].strip())
            except ValueError:
                reasons['INVALID_OR_MISSING_DATE'] += 1
                continue
            if day >= target_date:
                reasons['SAME_DAY_OR_LATER_EXCLUDED'] += 1
                continue
            if day > available.astimezone(cutoff.tzinfo).date():
                reasons['OBSERVATION_AFTER_RETAINED_AVAILABILITY'] += 1
                continue
            if not raw_track or distance is None:
                reasons['HISTORY_CONTEXT_INCOMPLETE'] += 1
                continue
            if identity is None:
                reasons[status] += 1
                continue
            match = raw_track == member['venue'] and distance == target_distance
            reasons['RAW_TARGET_CONTEXT_MATCH' if match else 'RAW_TARGET_CONTEXT_MISMATCH'] += 1
            if distance != target_distance:
                reasons['VERIFIED_DISTANCE_DIFFERENCE'] += 1
            if raw_track != member['venue']:
                reasons['DISTINCT_TRACK_LABEL_EQUIVALENCE_UNRESOLVED'] += 1
            event = coverage.digest([day.isoformat(), raw_track, distance])
            fingerprint = coverage.digest({key: row[key].strip()
                for key in coverage.COLUMNS if key != 'Dog Name'})
            observations.append({'observation_id': coverage.digest([identity, event, fingerprint]),
                'event_id': event, 'event_identity_kind': 'RUNNER_DATE_CONTEXT_PROXY',
                'runner_identity_id': identity, 'date': day.isoformat(),
                'available_at': available.isoformat(), 'source_track': raw_track,
                'canonical_track': raw_track, 'distance_m': distance,
                'first_sectional': row['1 SEC'], 'observation_fingerprint': fingerprint,
                'source_bindings': [{'source_race_id': member['race_id'],
                    'accepted_csv': member['accepted_csv'], 'block_row_index': index,
                    'block_token': name,
                    'target_native_runner_id': entry, 'box_number': box,
                    'primary_page': member['primary_page'], 'primary_receipt': member['primary_receipt'],
                    'available_by': available.isoformat(), 'identity_available_by': available.isoformat()}]})
    packet = {'target': {'race_id': member['race_id'], 'date': member['source_race_date'],
        'cutoff': cutoff.isoformat(), 'source_track': member['venue'], 'distance_m': target_distance},
        'roster': runners}
    packet['target']['observation_pool_scope'] = 'ALL_SOURCE_CARDS'
    packet['target']['source_card'] = {'race_id': member['race_id'],
        'racing_date': member['source_race_date'], 'accepted_csv': member['accepted_csv'],
        'available_at': available.isoformat(), 'aliases': {}, 'binding_native_runner_id': True,
        'binding_base': {'source_race_id': member['race_id'], 'accepted_csv': member['accepted_csv'],
            'primary_page': member['primary_page'], 'primary_receipt': member['primary_receipt'],
            'available_by': available.isoformat(), 'identity_available_by': available.isoformat()},
        'roster': [{key: runner[key] for key in ('runner_id', 'identity_id', 'identity_available_at', 'box_number')}
            | {'block_token': name} for runner, (_, name) in zip(runners, roster)]}
    audit = {'race_id': member['race_id'], 'runner_count': len(runners),
        'race_identity_method': race_identity_method,
        'verified_profile_count': sum(r['identity_id'] is not None for r in runners),
        'profile_dispositions': dict(Counter(r['identity_status'] for r in runners)),
        'row_dispositions': dict(reasons), 'sectional_presence': dict(sections),
        'raw_contexts': [{'track': track, 'distance_m': dist, 'rows': count}
            for (track, dist), count in sorted(contexts.items(), key=lambda item: str(item[0]))],
        'observation_copies': len(observations), 'runner_set_sha256': member['runner_set_sha256'],
        'source': {role: member[role] for role in coverage.ROLES}}
    return packet, observations, audit


def load_inputs(reference, reader=None):
    """Load only the exact existing feature-authorized input population.

    There is intentionally no result-loader seam. No verified alias evidence
    exists in this population's manifest, so raw context labels stay distinct.
    Future aliases need a separately reviewed evidence-bound adapter version.
    """
    require(reference == {'path': strict.MANIFEST_PATH, 'sha256': strict.MANIFEST_SHA},
        'FEATURE_MANIFEST_NOT_FIXED')
    reader = SnapshotReader(reader)
    manifest = reader.json(reference)
    require(manifest['schema_version'] == 'retained_speed_breadth_metadata_manifest_v1'
        and manifest['selection'] == 'ALL_82_FIXED_MEMBERS_IRRESPECTIVE_OF_RESULT_STATUS'
        and manifest['membership']['sha256'] == strict.MEMBERSHIP_SHA, 'FEATURE_MANIFEST_INVALID')
    membership = reader.json(manifest['membership'])
    originals = {member['race_id']: member for member in membership['members']}
    members = manifest['members']
    require(len(members) == len(membership['members']) == len(originals) == strict.MEMBERS
        and len({m['race_id'] for m in members}) == strict.MEMBERS
        and set(originals) == {m['race_id'] for m in members}
        and sum(m['target_runner_slots'] for m in members) == strict.RUNNER_SLOTS,
        'FEATURE_DENOMINATOR_CHANGED')
    packets, observations, audits = [], [], []
    for member in members:
        packet, rows, audit = construct_member(reader, member, originals[member['race_id']])
        packets.append(packet)
        observations.extend(rows)
        audits.append(audit)
    reader.check()
    identities = {row['identity_id'] for packet in packets for row in packet['roster']
        if row['identity_id'] is not None}
    return {'schema_version': 'retained_speed_candidate_inputs_v1', 'manifest': reference,
        'membership': manifest['membership'], 'packets': packets, 'observations': observations,
        'audit': {'cards': len(packets), 'runner_appearances': sum(len(p['roster']) for p in packets),
            'unique_verified_dog_profiles': len(identities), 'records': audits,
            'verified_aliases': [], 'alias_disposition': 'NO_EXACT_SOURCE_ALIAS_PROOF_IN_FIXED_MANIFEST',
            'event_identity': 'CONSERVATIVE_DOG_DATE_CONTEXT_PROXY_NOT_NATIVE_EVENT_ID',
            'additional_retained_history_scope': 'EARLIER_CAPTURE_COPIES_WITHIN_EXACT_82_ONLY',
            'reads': reader.reader.reads, 'read_bytes': reader.reader.bytes,
            'provider_requests': 0, 'result_requests': 0, 'result_payloads_opened': 0}}
