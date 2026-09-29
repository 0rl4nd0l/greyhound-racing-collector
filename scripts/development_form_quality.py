"""Versioned, offline-only descriptions of retained card history.

No production caller or provider access. Admission and source timing are the
caller's responsibility; the audit CLI supplies the existing identity gate.
"""
from __future__ import annotations

from collections import Counter
from datetime import date
import math

from scripts import offline_form_packet as legacy

VERSION = 'retained_card_form_v2'
NAMES = {
    'prior_start_count': 'retained_start_count',
    'days_since_last_start': 'days_since_last_retained_start',
    'recent_finish_mean_3': 'recent_finish_mean_3',
    'recent_finish_best_5': 'recent_finish_best_5',
    'recent_win_rate_5': 'recent_win_rate_5',
    'recent_place_rate_5': 'recent_top3_rate_5',
    'recent_avg_margin_5': 'recent_recorded_margin_mean_5',
    'career_win_rate': 'retained_win_rate',
    'career_place_rate': 'retained_top3_rate',
    'career_avg_finish': 'retained_finish_mean',
    'starts_same_venue': 'retained_same_venue_start_count',
    'win_rate_same_venue': 'retained_same_venue_win_rate',
    'starts_same_distance': 'retained_exact_distance_start_count',
    'win_rate_same_distance': 'retained_exact_distance_win_rate',
    'same_grade_start_count': 'retained_same_grade_label_start_count',
    'same_grade_win_rate': 'retained_same_grade_label_win_rate',
}
if tuple(NAMES) != tuple(legacy.FEATURES):
    raise ValueError('legacy feature order changed; versioned contract review required')

CONTRACT = {
    'version': VERSION, 'legacy_to_development_names': NAMES,
    'history': 'canonical accepted_history; strict earlier date; normalized dedup; cap20; newest first',
    'recent': 'first 3 or 5 retained starts, not last N calendar days',
    'rates': 'wins / observed numeric finishes; top3 means finish<=3, not bookmaker PLACE',
    'finish': 'recorded ordinal; dimensionless; smaller is better',
    'margin': 'numeric source MGN; units and winner/loser reference not independently established; recorded mean without sign conversion',
    'context': 'canonical venue, exact integer metres across venues, canonical grade label across jurisdictions; no ability equivalence',
    'unknown_context': 'null count/rate if target or any retained context is unclassifiable',
    'zero': 'no matches in inspected classifiable retained starts, or no recorded wins/top3 in known denominator',
    'completeness': 'unknown career coverage even if hashes and roster verify',
    'duplicates': 'retain recent and retained columns; no implicit regularization change',
    'numeric_validation': 'reject nonfinite/nonintegral numeric prior-date PLC/DIST before canonical int conversion; PLC 1..8, DIST positive; unavailable textual values stay missing',
    'preprocessing': 'training medians (all missing ->0), indicators, training standardization, within-race centering; legacy linear residual .35 tanh cap',
}


def build_record(raw_rows, *, target_date: date, venue, distance, grade):
    """Describe one admitted runner; never manufacture unavailable context."""
    raw_rows = list(raw_rows)
    for raw in raw_rows:
        try:
            history_date = date.fromisoformat(str(raw.get('DATE') or '').strip())
        except ValueError:
            continue  # Canonical parser records date rejection.
        if history_date >= target_date:
            continue
        for field, label in (('PLC', 'finish'), ('DIST', 'distance')):
            number = legacy.canonical.safe_float(raw.get(field))
            if number is not None and (
                not math.isfinite(number) or not number.is_integer() or number < 1
                or (field == 'PLC' and number > 8)
            ):
                raise ValueError('invalid recorded ' + label)
    history, rejected = legacy.canonical.accepted_history(raw_rows, target_date)
    for row in history:
        if row['finish'] is not None and not 1 <= row['finish'] <= 8:
            raise ValueError('invalid recorded finish')
        if row['margin'] is not None and not math.isfinite(row['margin']):
            raise ValueError('invalid recorded margin')
    venue = legacy.canonical.canonical_venue(venue)
    grade = legacy.canonical.canonical_grade(grade) if grade != '__MISSING__' else grade
    distance = legacy._metres(distance)
    old = legacy.canonical.feature_row('development', target_date, venue, distance,
                                       grade, 0, 1, 'development', history)
    values = {name: legacy._number(old.get(legacy.ALIASES.get(name, name)))
              for name in legacy.FEATURES}
    finishes = [h['finish'] for h in history[:5] if h['finish'] is not None]
    values['recent_finish_best_5'] = float(min(finishes)) if finishes else None
    features = {NAMES[name]: value for name, value in values.items()}
    quality = {}

    def describe(name, rows, field=None, status=None):
        denominator = sum(h[field] is not None for h in rows) if field else len(rows)
        if status is None:
            status = ('no_retained_history' if not history else
                      'no_matching_retained_starts' if not rows else
                      'no_recorded_values' if not denominator else 'observed')
        quality[NAMES[name]] = {'status': status, 'retained_starts': len(rows),
                               'observed_denominator': denominator}

    for name in legacy.FEATURES[:10]:
        rows = history[:3] if name.endswith('_3') else history[:5] if name.endswith('_5') else history
        if name == 'days_since_last_start':
            rows = history[:1]
        field = None if name in ('prior_start_count', 'days_since_last_start') else 'margin' if 'margin' in name else 'finish'
        describe(name, rows, field)
    for field, target, count, rate in (
        ('venue', venue, 'starts_same_venue', 'win_rate_same_venue'),
        ('distance', distance, 'starts_same_distance', 'win_rate_same_distance'),
        ('grade', grade, 'same_grade_start_count', 'same_grade_win_rate'),
    ):
        unknown = lambda value: value in (None, '', '__MISSING__')
        matching = [h for h in history if h[field] == target and not unknown(target)]
        status = ('target_context_unavailable' if unknown(target) else
                  'history_context_incomplete' if any(unknown(h[field]) for h in history) else
                  'no_retained_history' if not history else None)
        for name in (count, rate):
            if status:
                features[NAMES[name]] = None
            describe(name, matching, 'finish' if name == rate else None, status)
    return {'version': VERSION, 'features': features, 'quality': quality,
            'history': {'raw_rows': len(raw_rows), 'accepted_rows': len(history),
                        'rejections': dict(Counter(reason for reason, _ in rejected)),
                        'oldest_date': history[-1]['date'].isoformat() if history else None,
                        'latest_date': history[0]['date'].isoformat() if history else None,
                        'career_completeness': 'unknown',
                        'cap_reached': any(reason == 'HISTORY_CAP_20' for reason, _ in rejected)}}
