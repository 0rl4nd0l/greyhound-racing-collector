"""Gated replay of admitted development cards; no provider/DB access or fits."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
import json
from pathlib import Path

import numpy as np

from scripts import offline_form_packet as form
from scripts.development_form_quality import build_record, CONTRACT, NAMES
from scripts.explain_market_residual import (
    ROOT, SOURCE, FOUNDATION, load_scope, gate_lines, sha, write, decomposition,
)


def admitted_sources(provenance, allowed):
    """Resolve all paths by permitted race identity before any card is opened."""
    ids = {rid for rid, _ in allowed}
    result = {}
    for source in provenance['sources']:
        rid = source['race_id']
        if rid not in ids:
            continue
        if rid in result:
            raise ValueError('duplicate admitted source')
        result[rid] = source
    if set(result) != ids:
        raise ValueError('missing admitted source')
    return result


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    ledger = out / 'attempt.jsonl'
    def event(name, **fields):
        with ledger.open('a') as handle:
            handle.write(json.dumps({'event': name, **fields}, sort_keys=True) + '\n')
    event('START', development_fits=0)
    try:
        allowed, pins = load_scope()
        event('ADMISSION_PASSED', runners=len(allowed), races=len({r for r, _ in allowed}))
        expected = json.loads((ROOT / 'docs/research/market_explanation_20260929_inputs.json').read_text())
        def read(path):
            payload = path.read_bytes()
            if sha(payload) != expected[str(path)]:
                raise ValueError('pinned source changed: ' + str(path))
            pins[str(path)] = sha(payload)
            return payload
        development_path = FOUNDATION / 'development.jsonl'
        development_bytes = development_path.read_bytes()
        if sha(development_bytes) != pins[str(development_path)]:
            raise ValueError('development bytes changed after admission')
        rows = gate_lines(development_bytes, allowed)
        evaluation = gate_lines(read(SOURCE / 'search_v1/outer_predictions.jsonl'), allowed)
        eval_ids = {(r['race_id'], r['box']) for r in evaluation}
        sources = admitted_sources(json.loads(read(FOUNDATION / 'form_provenance.json')), allowed)
        grouped = defaultdict(list)
        for row in rows:
            grouped[row['race_id']].append(row)
        evidence, metrics = [], {'development': [], 'evaluation': []}
        for rid, rr in sorted(grouped.items()):
            rr.sort(key=lambda r: r['box'])
            source = sources[rid]
            payloads = {}
            for kind in ('card_source', 'card_sidecar'):
                path = Path(source[kind + '_path'])
                payloads[kind] = form._verified(path, source[kind + '_sha256'], int(source[kind + '_bytes']))
                pins[str(path)] = source[kind + '_sha256']
            card = payloads['card_source']
            metadata = json.loads(payloads['card_sidecar'])
            if metadata.get('metadata_is_leakage_safe') is not True or metadata['runner_completeness']['status'] != 'COMPLETE':
                raise ValueError('unqualified card metadata')
            if metadata['content_sha256'] != sha(card) or metadata['content_length'] != len(card):
                raise ValueError('sidecar/card binding')
            capture = form.canonical.capture_timestamp(metadata, require_timezone=True)
            jump = form.canonical.sidecar_jump_timestamp(metadata, rid)
            if (jump - capture).total_seconds() < 3600:
                raise ValueError('T-60 card timing')
            roster = [(r['box'], r['dog_token']) for r in rr]
            if sorted(form.canonical.parse_card_target_roster_bytes(card, source=rid)) != roster or sorted(form.canonical.sidecar_roster(metadata, source=rid)) != roster:
                raise ValueError('admitted roster mismatch')
            blocks = form.canonical.parse_form_blocks_bytes(card, source=rid)
            target_date = date.fromisoformat(rr[0]['race_date'])
            venue, _, grade, _ = form.canonical.target_metadata({'metadata': metadata}, rid)
            distance = form._metres(metadata.get('target_distance') or metadata.get('race_info', {}).get('distance'))
            for row in rr:
                raw = blocks[row['dog_token']]
                history, rejected = form.canonical.accepted_history(raw, target_date)
                if any(h['date'] > capture.date() for h in history):
                    raise ValueError('history after card availability')
                old = form.canonical.feature_row(rid, target_date, venue, distance, grade,
                                                len(rr), row['box'], row['dog_token'], history)
                old_values = {name: form._number(old.get(form.ALIASES.get(name, name))) for name in form.FEATURES}
                finish = [h['finish'] for h in history[:5] if h['finish'] is not None]
                old_values['recent_finish_best_5'] = float(min(finish)) if finish else None
                if old_values != {name: row['features'][name] for name in form.FEATURES}:
                    raise ValueError('original feature replay mismatch')
                new = build_record(raw, target_date=target_date, venue=venue, distance=distance, grade=grade)
                changed = [name for name in form.FEATURES if old_values[name] != new['features'][NAMES[name]]]
                missing = [name for name in form.FEATURES if old_values[name] is None]
                nonzero_winner_margins = sum(h['finish'] == 1 and h['margin'] is not None and h['margin'] > 0 for h in history)
                item = {'race_id': rid, 'race_date': row['race_date'], 'box': row['box'],
                        'dog_token': row['dog_token'], 'evaluation': (rid, row['box']) in eval_ids,
                        'features': new['features'], 'quality': new['quality'], 'history': new['history'],
                        'changed_legacy_values': changed, 'legacy_missing': missing,
                        'positive_winner_margin_observations': nonzero_winner_margins,
                        'card_sha256': sha(card), 'target_context': {'venue': venue, 'distance_m': distance, 'grade': grade}}
                evidence.append(item)
                metrics['development'].append((row, item))
                if item['evaluation']:
                    metrics['evaluation'].append((row, item))
        summary = {}
        for population, pairs in metrics.items():
            summary[population] = {
                'races': len({r['race_id'] for r, _ in pairs}), 'runners': len(pairs),
                'history_length_runners': dict(Counter(i['history']['accepted_rows'] for _, i in pairs)),
                'history_rejections': dict(sum((Counter(i['history']['rejections']) for _, i in pairs), Counter())),
                'numeric_changed_races': len({r['race_id'] for r, i in pairs if i['changed_legacy_values']}),
                'numeric_changed_runners': sum(bool(i['changed_legacy_values']) for _, i in pairs),
                'numeric_changed_values': sum(len(i['changed_legacy_values']) for _, i in pairs),
                'renamed_columns': sum(old != new for old, new in NAMES.items()),
                'renamed_values': len(pairs) * sum(old != new for old, new in NAMES.items()),
                'equal_recent_retained_win': sum(r['features']['recent_win_rate_5'] == r['features']['career_win_rate'] for r, _ in pairs),
                'equal_recent_retained_top3': sum(r['features']['recent_place_rate_5'] == r['features']['career_place_rate'] for r, _ in pairs),
                'any_missing_races': len({r['race_id'] for r, i in pairs if i['legacy_missing']}),
                'any_missing_runners': sum(bool(i['legacy_missing']) for _, i in pairs),
                'positive_winner_margin_repeated_observations': sum(i['positive_winner_margin_observations'] for _, i in pairs),
                'per_feature': {name: {
                    'missing_runners': sum(r['features'][name] is None for r, _ in pairs),
                    'missing_races': len({r['race_id'] for r, _ in pairs if r['features'][name] is None}),
                    'zero_runners': sum(r['features'][name] == 0 for r, _ in pairs),
                    'quality_statuses': dict(Counter(i['quality'][NAMES[name]]['status'] for _, i in pairs)),
                } for name in form.FEATURES},
            }
        # Saved earlier fits only; no reconstructed model is described as original.
        eval_groups = defaultdict(list)
        for row in evaluation:
            eval_groups[row['race_id']].append(row)
        receipts = {fold: json.loads(read(SOURCE / f'search_v1/{fold}_models.json'))['base16']
                    for fold in {r['outer'] for r in evaluation}}
        errors = []
        missing_contribution_runners = 0
        for rr in eval_groups.values():
            rr.sort(key=lambda r: r['box'])
            model = receipts[rr[0]['outer']]
            dec = decomposition(rr, model)
            missing_contribution_runners += int(np.any(np.abs(dec['contributions'][:, 16:]) > 1e-15, axis=1).sum())
            for label, strength in [('refit_base16', 1), ('refit_half', .5)]:
                rebuilt = decomposition(rr, model, strength)['p']
                errors.append(float(np.max(np.abs(rebuilt - [r['predictions'][label] for r in rr]))))
        summary['replay'] = {'original_feature_values_checked': len(rows) * 16,
                             'saved_forecast_runners_per_model': len(evaluation),
                             'maximum_probability_error': max(errors),
                             'nonzero_missing_indicator_contribution_runners': missing_contribution_runners,
                             'development_fits': 0, 'protected_decodes': 0,
                             'installed_forecast_replay': 'not_attempted_no_protected_inputs_opened'}
        write(out / 'summary.json', summary)
        write(out / 'feature_contract.json', CONTRACT)
        with (out / 'runner_quality.jsonl').open('x') as handle:
            for item in evidence:
                handle.write(json.dumps(item, sort_keys=True, allow_nan=False) + '\n')
        for path in [Path(__file__), Path(form.__file__), Path(form.canonical.__file__),
                     ROOT / 'scripts/development_form_quality.py', ROOT / 'scripts/explain_market_residual.py']:
            pins[str(path)] = sha(path.read_bytes())
        write(out / 'input_hashes.json', pins)
        # Recheck every opened input at completion; completeness was checked above,
        # not inferred from these hashes.
        for path, digest in pins.items():
            if sha(Path(path).read_bytes()) != digest:
                raise ValueError('input changed during audit: ' + path)
        event('COMPLETE', development_fits=0, numeric_changes=summary['development']['numeric_changed_values'])
        return summary
    except Exception as exc:
        event('FAILED', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.out)
    print(json.dumps({key: {k: v for k, v in value.items() if k != 'per_feature'}
                      for key, value in result.items()}, indent=2, sort_keys=True))
