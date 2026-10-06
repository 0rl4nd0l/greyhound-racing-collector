"""The pre-existing June/July development allocation, with original cell bindings.

Legacy cards establish distinct runners inside a field, not stable global dog
IDs. Their observation pools are therefore card-local; no name-only joins occur.
Labels are decoded only by the explicit separate evaluation loader.
"""
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
import json
import math
import re

from race_collection import retained_card_timing_coverage as coverage
from scripts import build_form_only_v1_packet as canonical

require = coverage.require
FIRST, LAST = '2026-06-10', '2026-07-08'


def scalar(line, field):
    values = [json.loads(m[1]) for m in re.finditer(
        r'"' + re.escape(field) + r'"\s*:\s*("(?:[^"\\]|\\.)*"|[0-9]+)', line)]
    require(bool(values) and all(v == values[0] for v in values), 'DEVELOPMENT_IDENTITY_INVALID')
    return values[0]


def identity_rows(payload, protection):
    """Project only identities before any complete labelled record is decoded."""
    denied = protection['records']
    require(min(key[:10] for key in denied) > LAST, 'PROTECTION_HISTORY_OVERLAP')
    result = []
    for line in payload.decode().splitlines():
        rid, day = scalar(line, 'race_id'), scalar(line, 'race_date')
        match = re.fullmatch(r'Race (\d+) - (.+) - (\d{4}-\d{2}-\d{2})', rid)
        require(match is not None and match[3] == day and FIRST <= day <= LAST,
                'DEVELOPMENT_DATE_UNALLOCATED')
        require(f'{day}|{match[2]}|{int(match[1])}' not in denied, 'PROTECTED_TARGET')
        box, token = scalar(line, 'box'), scalar(line, 'dog_token')
        require(type(box) is int and 1 <= box <= 8 and isinstance(token, str) and token,
                'DEVELOPMENT_ROSTER_INVALID')
        result.append({'race_id': rid, 'race_date': day, 'box': box, 'dog_token': token})
    require(len(result) == 2360 and len({r['race_id'] for r in result}) == 331
            and len({(r['race_id'], r['box']) for r in result}) == len(result),
            'DEVELOPMENT_POPULATION_CHANGED')
    return result


def load_inputs(scope, reader):
    protection = reader.json(scope['protected'])
    assessment = reader.json(scope['assessment'])
    require(assessment['development_sha256'] == scope['development']['sha256']
            and assessment['races'] == 331 and assessment['runners'] == 2360,
            'DEVELOPMENT_ASSESSMENT_CHANGED')
    identities = identity_rows(reader.read(scope['development']), protection)
    provenance = reader.json(scope['provenance'])
    sources = {s['race_id']: s for s in provenance['sources']}
    grouped = defaultdict(list)
    for row in identities:
        grouped[row['race_id']].append(row)
    packets, audits = [], []
    for rid, rows in sorted(grouped.items()):
        src = sources[rid]
        refs = {role: {'path': src[prefix + '_path'], 'sha256': src[prefix + '_sha256'],
                       'bytes': int(src[prefix + '_bytes'])}
                for role, prefix in [('accepted_csv', 'card_source'), ('sidecar', 'card_sidecar')]}
        metadata = reader.json(refs['sidecar'])
        payload = reader.read(refs['accepted_csv'])
        require(metadata['content_sha256'] == refs['accepted_csv']['sha256']
                and metadata['content_length'] == len(payload)
                and metadata.get('metadata_is_leakage_safe') is True
                and metadata['runner_completeness']['status'] == 'COMPLETE', 'LEGACY_CARD_BINDING')
        require(not any(metadata.get(key) for key in ('layout_id', 'layout_era', 'track_layout',
            'target_layout_id', 'target_layout_era', 'clock_convention', 'sectional_endpoint')),
            'SOURCE_CONTEXT_METADATA_REQUIRES_EXPLICIT_BINDING')
        capture = canonical.capture_timestamp(metadata, require_timezone=True)
        jump = canonical.sidecar_jump_timestamp(metadata, rid)
        require(jump is not None and (jump-capture).total_seconds() >= 3600, 'LEGACY_CAPTURE_TIMING')
        cutoff = jump - timedelta(minutes=2)
        projected = coverage.projected_card(payload)
        roster = canonical.parse_card_target_roster_bytes(projected, source='verified development card')
        expected = sorted((r['box'], r['dog_token']) for r in rows)
        require(sorted(roster) == expected == sorted(canonical.sidecar_roster(metadata,
                source='verified development sidecar')), 'LEGACY_ROSTER_MISMATCH')
        blocks = canonical.parse_form_blocks_bytes(projected, source='verified development card')
        target_distance = coverage.distance(metadata.get('target_distance') or metadata.get('race_info', {}).get('distance'))
        require(target_distance is not None, 'LEGACY_DISTANCE_MISSING')
        day = rows[0]['race_date']
        runners, observations = [], []
        counts = Counter()
        for box, token in roster:
            identity = 'retained-card:' + coverage.digest([refs['accepted_csv']['sha256'], box, token])
            runner = coverage.digest([rid, box, token])
            runners.append({'runner_id': runner, 'identity_id': identity,
                'identity_available_at': capture.isoformat(), 'box_number': box,
                'identity_status': 'VERIFIED_WITHIN_CARD_ONLY', 'strict_runner_id': runner})
            for index, row in enumerate(blocks[token]):
                raw_track, dist = row['TRACK'].strip(), coverage.distance(row['DIST'])
                try:
                    prior = date.fromisoformat(row['DATE'].strip())
                except ValueError:
                    counts['INVALID_HISTORY_DATE'] += 1
                    continue
                if prior >= date.fromisoformat(day):
                    counts['NOT_PRIOR_DATE'] += 1
                    continue
                if not raw_track or dist is None:
                    counts['INVALID_HISTORY_CONTEXT'] += 1
                    continue
                event = coverage.digest([prior.isoformat(), raw_track, dist])
                fingerprint = coverage.digest({k: row[k].strip() for k in coverage.COLUMNS if k != 'Dog Name'})
                observations.append({'observation_id': coverage.digest([identity, event, fingerprint]),
                    'event_id': event, 'event_identity_kind': 'RUNNER_DATE_CONTEXT_PROXY',
                    'runner_identity_id': identity, 'date': prior.isoformat(),
                    'available_at': capture.isoformat(), 'source_track': raw_track,
                    'canonical_track': raw_track, 'distance_m': dist,
                    'first_sectional': row['1 SEC'], 'observation_fingerprint': fingerprint,
                    'source_bindings': [{'source_race_id': rid, 'accepted_csv': refs['accepted_csv'],
                        'sidecar': refs['sidecar'], 'block_token': token, 'block_row_index': index,
                        'box_number': box, 'available_by': capture.isoformat(),
                        'identity_available_by': capture.isoformat()}]})
        packet = {'target': {'race_id': rid, 'date': day, 'cutoff': cutoff.isoformat(),
            'source_track': rid.split(' - ')[1], 'distance_m': target_distance},
            'roster': runners, 'observations': observations}
        packet['target']['observation_pool_scope'] = 'CARD_LOCAL'
        packet['target']['source_card'] = {'race_id': rid, 'racing_date': day,
            'accepted_csv': refs['accepted_csv'], 'available_at': capture.isoformat(),
            'aliases': {}, 'binding_native_runner_id': False,
            'binding_base': {'source_race_id': rid, 'accepted_csv': refs['accepted_csv'],
                'sidecar': refs['sidecar'], 'available_by': capture.isoformat(),
                'identity_available_by': capture.isoformat()},
            'roster': [{key: runner[key] for key in ('runner_id', 'identity_id', 'identity_available_at', 'box_number')}
                | {'block_token': token} for runner, (_, token) in zip(runners, roster)]}
        packets.append(packet)
        audits.append({'race_id': rid, 'runner_count': len(roster), 'source': refs,
            'row_exclusions': dict(counts), 'observation_copies': len(observations),
            'identity_scope': 'WITHIN_SINGLE_VERIFIED_CARD_NO_CROSS_CARD_JOIN'})
    return {'packets': packets, 'audit': {'cards': len(packets), 'runner_appearances': len(identities),
        'unique_global_dog_identities': None, 'identity_scope': 'WITHIN_CARD_ONLY',
        'verified_aliases': [], 'records': audits, 'target_labels_decoded': 0}}


def evaluation_rows(scope, reader, eligible_ids, *, on_label_decoded=None):
    """Caller must first authorize and freeze this exact development evaluation."""
    protection = reader.json(scope['protected'])
    payload = reader.read(scope['development'])
    identities = identity_rows(payload, protection)
    expected = {r['race_id'] for r in identities if r['race_date'] >= '2026-06-24'}
    require(expected == set(eligible_ids) and len(expected) == 177, 'EVALUATION_MEMBERSHIP_CHANGED')
    rows = []
    for line in payload.decode().splitlines():
        if scalar(line, 'race_id') in expected:
            row = json.loads(line)
            if on_label_decoded is not None:
                on_label_decoded(row['race_id'])
            rows.append(row)
    require(all(r['y'] in (0, 1) and math.isfinite(r['odds']) and r['odds'] > 1 for r in rows),
            'DEVELOPMENT_LABEL_OR_ODDS_INVALID')
    return sorted(rows, key=lambda r: (r['race_date'], r['race_id'], r['box']))


def reproduce_base16(rows, model):
    """Reproduce the retained first-period base16 coefficients without refitting."""
    import numpy as np
    require(model['kind'] == 'linear' and model['l2'] == 1.0 and model['prep']['center'] is True,
            'BASELINE_MODEL_CHANGED')
    prep = model['prep']
    values = np.asarray([[np.nan if r['features'].get(name) is None else r['features'][name]
                          for name in prep['names']] for r in rows], float)
    x = (np.c_[np.where(np.isfinite(values), values, np.asarray(prep['median'])),
               ~np.isfinite(values)] - np.asarray(prep['mean'])) / np.asarray(prep['scale'])
    starts = np.asarray([i for i, r in enumerate(rows) if i == 0 or rows[i-1]['race_id'] != r['race_id']])
    counts = np.diff(np.r_[starts, len(rows)])
    x -= np.repeat(np.add.reduceat(x, starts, axis=0) / counts[:, None], counts, axis=0)
    logits = np.log([r['market'] for r in rows]) + .35 * np.tanh(x @ np.asarray(model['beta']) / .35)
    out = np.empty(len(rows))
    for first, count in zip(starts, counts):
        segment = logits[first:first+count]
        shifted = np.exp(segment - np.max(segment))
        out[first:first+count] = shifted / np.sum(shifted)
    require(np.isfinite(out).all() and (out > 0).all(), 'BASELINE_PROBABILITY_INVALID')
    return out.tolist()
