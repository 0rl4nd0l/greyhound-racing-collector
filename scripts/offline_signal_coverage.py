"""Coverage-only audit of raw prior history; never decode target outcomes."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from datetime import date
from pathlib import Path

from scripts.offline_form_packet import _metres, _verified, canonical


def audit(prepared: Path):
    targets = collections.defaultdict(set)
    identity_file = prepared / 'development.jsonl'
    for line in identity_file.open():
        values = {}
        for field in ('race_id', 'dog_token'):
            match = re.search(r'"' + field + r'":\s*("(?:[^"\\]|\\.)*")', line)
            if match is None:
                raise ValueError(f'missing identity {field}')
            values[field] = json.loads(match[1])
        targets[values['race_id']].add(values['dog_token'])
    provenance = json.loads((prepared / 'form_provenance.json').read_text())
    total = collections.Counter()
    pir_shapes = collections.Counter()
    runner_counts = collections.Counter()
    race_counts = collections.Counter()
    by_venue = collections.defaultdict(collections.Counter)
    for source in provenance['sources']:
        rid = source['race_id']
        if rid not in targets:
            continue
        target_date = date.fromisoformat(rid.rsplit(' - ', 1)[1])
        if target_date > date(2026, 7, 9):
            raise ValueError('outside development')
        card = _verified(Path(source['card_source_path']), source['card_source_sha256'])
        metadata = json.loads(_verified(Path(source['card_sidecar_path']), source['card_sidecar_sha256']))
        venue, _, _, _ = canonical.target_metadata({'metadata': metadata}, rid)
        distance = _metres(metadata.get('target_distance') or metadata.get('race_info', {}).get('distance'))
        blocks = canonical.parse_form_blocks_bytes(card, source=rid)
        full_field = collections.Counter()
        for token in sorted(targets[rid]):
            history, rejected = canonical.accepted_history(blocks[token], target_date)
            # Raw values can only be used when exact canonical row membership is
            # unambiguous; the current selected packet has no rejected rows.
            if rejected:
                raise ValueError(f'raw membership requires explicit reconciliation: {rid}')
            rows = sorted(blocks[token], key=lambda row: row['DATE'], reverse=True)
            assert len(history) == len(rows)
            count = collections.Counter()
            for row in rows:
                assert date.fromisoformat(row['DATE']) < target_date
                total['history_observations'] += 1
                by_venue[venue]['history_observations'] += 1
                pir = str(row.get('PIR') or '').strip()
                shape = 'blank' if not pir else 'single_digit' if re.fullmatch('[0-9]', pir) else 'multi_digit' if pir.isdigit() else 'other_code'
                pir_shapes[shape] += 1
                if pir:
                    count['pir_present'] += 1
                    total['pir_present'] += 1
                if re.fullmatch('[1-8]{2,}', pir):
                    count['pir_multi_valid'] += 1
                    total['pir_multi_valid'] += 1
                if shape == 'single_digit' and canonical.safe_int(pir) == canonical.safe_int(row.get('PLC')):
                    total['single_digit_pir_equals_finish'] += 1
                for field in ('1 SEC', 'TIME', 'MGN', 'PLC'):
                    value = canonical.safe_float(row.get(field))
                    if value is not None:
                        count[field] += 1
                        total[field] += 1
                        by_venue[venue][field] += 1
                        if field in ('1 SEC', 'TIME') and value > 0 and canonical.canonical_venue(row.get('TRACK')) == venue and _metres(row.get('DIST')) == distance:
                            count[field + '_same_venue_distance'] += 1
                            total[field + '_same_venue_distance'] += 1
            total['runner_targets'] += 1
            for field in ('pir_present', 'pir_multi_valid', '1 SEC', 'TIME', 'MGN', 'PLC', '1 SEC_same_venue_distance', 'TIME_same_venue_distance'):
                for threshold in (1, 3):
                    name = f'{field}_at_least_{threshold}'
                    if count[field] >= threshold:
                        runner_counts[name] += 1
                        full_field[name] += 1
            if count['MGN'] >= 3 and count['PLC'] >= 3:
                runner_counts['margin_and_finish_at_least_3'] += 1
        total['races'] += 1
        for field, count in full_field.items():
            if count == len(targets[rid]):
                race_counts[field] += 1
    return {
        'population': 'exact prepared eligible development runner identities; no target labels parsed',
        'development_sha256': hashlib.sha256(identity_file.read_bytes()).hexdigest(),
        'totals': dict(total), 'pir_shapes': dict(pir_shapes),
        'runner_target_coverage': dict(runner_counts),
        'complete_race_coverage': dict(race_counts),
        'source_target_venue_history_observations': dict(by_venue),
        'interpretation': 'PIR presence is not validated first-call position; variable code length and contradictory local definitions require source semantics. Sectional/time comparisons require same venue and distance. Counts are repeated history observations across target runners, not unique historical starts.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.prepared)
    with args.out.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_target_venue_history_observations'}, indent=2))
