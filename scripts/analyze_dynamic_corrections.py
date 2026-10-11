"""Account for every authorized later-race correction, without fitting or selection."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from race_collection.research_correction_diagnostics import summarize


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def signed(value):
    return 'missing' if value is None else 'negative' if value < -.1 else 'positive' if value > .1 else 'flat'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--market', type=Path, required=True)
    parser.add_argument('--states', type=Path, required=True)
    parser.add_argument('--lineage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    states = {(r['source_race_key'], r['guide_box']): r for r in read_rows(args.states)}
    history = defaultdict(list)
    for row in read_rows(args.lineage):
        if row['status'] == 'PRIOR_DATE_UPDATE':
            history[row['native_id']].append(row)
    races = []
    for race in read_rows(args.market):
        if race['partition'] != 'later':
            continue
        runners = race['runners']
        # Lowest guide box breaks a market probability tie without looking at labels.
        favourite = min(runners, key=lambda r: (-r['probabilities']['reported_sp'], r['guide_box']))
        state = states[(race['source_race_key'], favourite['guide_box'])]
        if state['native_dog_id'] != favourite['native_dog_id'] or state['race_date'] != race['race_date']:
            raise ValueError('state identity or target date mismatch')
        values = state['states']['dynamic']
        previous = [r for r in history[state['native_dog_id']] if r['date'] < race['race_date']]
        previous.sort(key=lambda r: (r['date'], r['race_key']))
        opposition_change = (state['opponents_prior_mean'] - previous[-1]['opponent_prior_mean']
                             if previous else None)
        p, support, context = favourite['probabilities']['reported_sp'], values['support'], values['context_support']
        box = favourite['guide_box']
        conditions = {
            'market_favourite_form_change': signed(values['latest_change']),
            'market_favourite_opposition_change': signed(opposition_change),
            'market_favourite_context_effect': signed(values['context_residual']),
            'market_favourite_context_support': 'unknown' if context is None else 'none' if context == 0 else 'observed',
            'market_favourite_history_support': 'none' if support == 0 else 'below2' if support < 2 else 'at_least2',
            'market_favourite_probability': 'up_to0.2' if p <= .2 else '0.2_to0.4' if p <= .4 else 'above0.4',
            'market_favourite_guide_draw': 'inside1to3' if box <= 3 else 'middle4to5' if box <= 5 else 'outside6to8' if box <= 8 else 'reserve_unqualified',
            'field_size': str(len(runners)),
        }
        races.append({'key': race['source_race_key'], 'date': race['race_date'], 'track': race['track'],
                      'runners': runners, 'conditions': conditions})
    if len(races) != 975 or len({r['date'] for r in races}) != 10:
        raise ValueError('fixed975 race/10 date population changed')
    available = set.intersection(*(set(r['probabilities']) for race in races for r in race['runners']))
    comparisons = [('reported_sp', 'matched_hybrid65'), ('reported_sp', 'saved_tree95_temperature'),
                   ('reported_sp', 'market_calibrated'), ('market_calibrated', 'hybrid_sp_combo')]
    comparisons += [('market_calibrated', name) for name in ['recency_sp_combo', 'dynamic_sp_combo'] if name in available]
    records = []
    for reference, model in comparisons:
        result = summarize(races, reference=reference, model=model)
        filename = model + '_minus_' + reference + '.json'
        (args.output / filename).write_text(json.dumps(result, indent=2) + '\n')
        records.append({'file': filename, 'reference': reference, 'model': model, **result['overall']})
    manifest = {'status': 'EXPLORATORY_NO_RULE_SELECTION', 'population': 'all975_exposed_later_races',
                'condition_anchor': 'market favourite; lowest guide box breaks ties; no outcome-based strata',
                'sign_threshold': .1, 'support_threshold': 2, 'early_pace': 'unqualified; no invented proxy',
                'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [args.market, args.states, args.lineage]}, 'comparisons': records}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
