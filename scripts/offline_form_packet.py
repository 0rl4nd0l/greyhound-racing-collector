"""Read-only, outcome-free raw-card features for the September offline study.

The original acquisition CSV has older semantics than the tracked v4 contract.
It is used for immutable eligibility and identity only; features are rebuilt
from hash-bound pre-race cards with the current canonical history parser.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

from scripts import build_form_only_v1_packet as canonical

SOURCE_ROOT = Path('/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound-form-only-v1-acquisition-20260718/reports/agent_jobs/form_only_v1_acquisition_foundation_20260718')
CANONICAL_BUILDER_SHA256 = '11b56970de0e53975a444d0ad066f7ea566ffc717757b074d5eb0d5f8865f086'
SOURCE_HASHES = {
    'development_features.csv': '195bc5174bdc9abe1755e2e1ca90c07f41927126e0fa0d27f632fb5926dfb568',
    'development_races.csv': 'be6b5c71430bbad4f47e8c3a79b9a38409154873b7e85fcc7fdc5f945f5cf348',
    'development_runners.csv': 'e209dea489932a97c15f1cdfe7554a24cdb6c2ecf4dfb8b6898a3b5c2c1adc01',
    'out_of_time_races.csv': '9ad6e0f2a89c2dac04f1c8f4be2dbea2a6ea7751fa3af486c490a71fb961c711',
}
FEATURES = (
    'prior_start_count', 'days_since_last_start', 'recent_finish_mean_3',
    'recent_finish_best_5', 'recent_win_rate_5', 'recent_place_rate_5',
    'recent_avg_margin_5', 'career_win_rate', 'career_place_rate',
    'career_avg_finish', 'starts_same_venue', 'win_rate_same_venue',
    'starts_same_distance', 'win_rate_same_distance', 'same_grade_start_count',
    'same_grade_win_rate',
)
ALIASES = {
    'recent_avg_margin_5': 'recent_margin_mean_5',
    'career_avg_finish': 'career_finish_mean',
    'starts_same_venue': 'same_venue_start_count',
    'win_rate_same_venue': 'same_venue_win_rate',
    'starts_same_distance': 'same_distance_start_count',
    'win_rate_same_distance': 'same_distance_win_rate',
}


def _verified(path: Path, digest: str, size: int | None = None) -> bytes:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f'source hash mismatch: {path}')
    if size is not None and len(payload) != size:
        raise ValueError(f'source byte length mismatch: {path}')
    return payload


def _csv(name: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(_verified(SOURCE_ROOT / name, SOURCE_HASHES[name]).decode())))


def _race_key(race_id: str) -> str:
    race, venue, day = race_id.split(' - ')
    return f'{day}|{canonical.canonical_venue(venue)}|{int(race.removeprefix("Race "))}'


def _number(value):
    return None if value in ('', None) else float(value)


def _metres(value):
    """Parse an explicit source distance, never infer it from prior starts."""
    if value in ('', None):
        return None
    match = re.fullmatch(r'([1-9][0-9]*)(?:\s*m)?', str(value).strip(), re.I)
    return int(match[1]) if match else None


def _weighted(history: list[dict], field: str):
    pairs = [(float(row[field]), 0.5 ** (i / 3)) for i, row in enumerate(history) if row[field] is not None]
    return sum(v * w for v, w in pairs) / sum(w for _, w in pairs) if pairs else None


def load_features(*, excluded_race_ids=(), excluded_race_keys=()):
    """Return (runner rows, audit metadata); never load labels or a database.

    Failures exclude an entire race and are reported, preserving paired runner
    populations. All input metadata are hash-bound; each card and sidecar is
    verified before parsing. Protected identities are rejected before raw reads.
    """
    _verified(Path(canonical.__file__), CANONICAL_BUILDER_SHA256)
    races = _csv('development_races.csv')
    runners = _csv('development_runners.csv')
    # Verify the old feature packet, but do not use its values for predictions.
    _verified(SOURCE_ROOT / 'development_features.csv', SOURCE_HASHES['development_features.csv'])
    reserved = _csv('out_of_time_races.csv')  # Identity-only; contains no outcomes.
    reserved_ids = {r['race_id'] for r in reserved}
    reserved_keys = {_race_key(r) for r in reserved_ids}
    denied_ids = set(excluded_race_ids) | reserved_ids
    denied_keys = set(excluded_race_keys) | reserved_keys
    by_race = defaultdict(list)
    for runner in runners:
        by_race[runner['race_id']].append(runner)
    result, exclusions, sources = [], [], []
    rejected_history = Counter()
    for race in races:
        rid = race['race_id']
        key = _race_key(rid)
        if rid in denied_ids or key in denied_keys:
            exclusions.append({'race_id': rid, 'reason': 'PROTECTED_IDENTITY'})
            continue
        try:
            target_date = date.fromisoformat(race['race_date'])
            if target_date > date(2026, 7, 9):
                raise ValueError('outside frozen development population')
            capture = datetime.fromisoformat(race['card_capture_timestamp'])
            jump = datetime.fromisoformat(race['jump_timestamp'])
            if capture.utcoffset() is None or jump.utcoffset() is None or (jump-capture).total_seconds() < 3600:
                raise ValueError('pre-race T-60 timing failed')
            card = _verified(Path(race['card_source_path']), race['card_source_sha256'], int(race['card_source_bytes']))
            sidecar_bytes = _verified(Path(race['card_sidecar_path']), race['card_sidecar_sha256'], int(race['card_sidecar_bytes']))
            metadata = json.loads(sidecar_bytes)
            if metadata.get('metadata_is_leakage_safe') is not True or metadata['runner_completeness']['status'] != 'COMPLETE':
                raise ValueError('unsafe or incomplete sidecar')
            if metadata['content_sha256'] != race['card_source_sha256'] or metadata['content_length'] != len(card):
                raise ValueError('sidecar card identity mismatch')
            card_roster = canonical.parse_card_target_roster_bytes(card, source=rid)
            sidecar_roster = canonical.sidecar_roster(metadata, source=rid)
            target_runners = by_race[rid]
            roster = sorted((int(r['box_number']), r['runner_id'].split('|dog:')[1]) for r in target_runners)
            if len(set(roster)) != len(roster) or len({box for box, _ in roster}) != len(roster):
                raise ValueError('duplicate target runner')
            # Strict complete population: no inferred scratches/reserve removal.
            if sorted(card_roster) != sorted(sidecar_roster) or sorted(sidecar_roster) != roster:
                raise ValueError('card sidecar target roster mismatch')
            venue, distance, grade, _ = canonical.target_metadata({'metadata': metadata}, rid)
            distance = _metres(metadata.get('target_distance') or metadata.get('race_info', {}).get('distance'))
            blocks = canonical.parse_form_blocks_bytes(card, source=rid)
            race_rows = []
            for runner in target_runners:
                box = int(runner['box_number'])
                token = runner['runner_id'].split('|dog:')[1]
                if token not in blocks:
                    raise ValueError('runner history block missing')
                history, rejected = canonical.accepted_history(blocks[token], target_date)
                rejected_history.update(reason for reason, _ in rejected)
                raw = canonical.feature_row(rid, target_date, venue, distance, grade, len(roster), box, token, history)
                features = {name: _number(raw.get(ALIASES.get(name, name))) for name in FEATURES}
                finishes = [h['finish'] for h in history[:5] if h['finish'] is not None]
                features['recent_finish_best_5'] = float(min(finishes)) if finishes else None
                for name in ('recent_finish_mean_5', 'career_margin_mean', 'history_missing', 'recency_missing', 'finish_missing', 'margin_missing'):
                    features[name] = _number(raw[name])
                features.update({
                    'ewma_finish_half_life_3_starts': _weighted(history, 'finish'),
                    'ewma_margin_half_life_3_starts': _weighted(history, 'margin'),
                    'box_number': float(box), 'target_distance_m': _number(distance),
                    'field_size': float(len(roster)),
                })
                assert all(v is None or math.isfinite(v) for v in features.values())
                race_rows.append({
                    'race_id': rid, 'race_key': key, 'race_date': target_date.isoformat(),
                    'box_number': box, 'dog_token': token, 'runner_id': runner['runner_id'],
                    'features': features, 'target_venue': venue, 'target_grade': grade,
                    'source_venue': rid.split(' - ')[1], 'card_capture_timestamp': capture.isoformat(),
                    'jump_timestamp': jump.isoformat(), 'card_source_sha256': race['card_source_sha256'],
                    'label_provenance_class': race['label_provenance_class'],
                    'history_latest_date': history[0]['date'].isoformat() if history else None,
                })
            result.extend(race_rows)
            sources.append({k: race[k] for k in ('race_id', 'card_source_path', 'card_source_sha256', 'card_source_bytes', 'card_sidecar_path', 'card_sidecar_sha256', 'card_sidecar_bytes')})
        except (ValueError, KeyError, FileNotFoundError) as exc:
            exclusions.append({'race_id': rid, 'reason': str(exc)})
    identities = [(r['race_id'], r['box_number']) for r in result]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate included race box identity')
    audit = {
        'source_root': str(SOURCE_ROOT), 'source_hashes': SOURCE_HASHES,
        'canonical_builder_sha256': hashlib.sha256(Path(canonical.__file__).read_bytes()).hexdigest(),
        'feature_semantics': 'current canonical raw-card history date<target, cap20; explicit integer-metre target distance including m suffix; residual 16 renamed plus fixed half-life3 EWMA',
        'old_feature_values_used': False, 'candidate_races': len(races), 'candidate_runners': len(runners),
        'included_races': len({r['race_id'] for r in result}), 'included_runners': len(result),
        'exclusions': exclusions, 'history_rejections': dict(rejected_history),
        'reserved_out_of_time_race_ids': sorted(reserved_ids), 'reserved_out_of_time_race_keys': sorted(reserved_keys),
        'explicit_excluded_race_ids': sorted(denied_ids), 'explicit_excluded_race_keys': sorted(denied_keys),
        'sources': sources,
        'missingness': {name: sum(r['features'][name] is None for r in result) for name in FEATURES},
    }
    return result, audit
