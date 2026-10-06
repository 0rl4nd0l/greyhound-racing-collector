import math

import pytest

from race_collection import sectional_development_inputs as inputs


def identities():
    # Deliberately invalid outcome syntax proves this path never JSON-decodes it.
    return '\n'.join(
        '{"race_id":"Race %d - TEST - 2026-06-10","race_date":"2026-06-10",'
        '"box":%d,"dog_token":"DOG%d","y":DO_NOT_DECODE}' % (race, box, box)
        for race in range(1, 332) for box in range(1, 9 if race <= 43 else 8)
    ).encode()


def test_identity_projection_never_decodes_labels():
    rows = inputs.identity_rows(identities(), {'records': {'2026-07-15|OTHER|1': 'reserved'}})
    assert len(rows) == 2360
    assert len({r['race_id'] for r in rows}) == 331


def test_changed_population_rejected_before_label_decode():
    with pytest.raises(ValueError, match='POPULATION_CHANGED'):
        inputs.identity_rows(identities().split(b'\n', 1)[1], {'records': {'2026-07-15|OTHER|1': 'reserved'}})


def test_protected_history_window_rejected():
    with pytest.raises(ValueError, match='PROTECTION_HISTORY_OVERLAP'):
        inputs.identity_rows(identities(), {'records': {'2026-06-10|TEST|1': 'reserved'}})


def test_baseline_reproduction_uses_saved_transform_and_centers_whole_race():
    model = {'kind': 'linear', 'l2': 1.0, 'prep': {'center': True, 'names': ['form'],
        'median': [0.0], 'mean': [0.0, 0.0], 'scale': [1.0, 1.0]}, 'beta': [0.2, 0.0]}
    rows = [{'race_id': 'a', 'features': {'form': 1.0}, 'market': 0.4},
            {'race_id': 'a', 'features': {'form': 3.0}, 'market': 0.6},
            {'race_id': 'b', 'features': {'form': None}, 'market': 0.5},
            {'race_id': 'b', 'features': {'form': 0.0}, 'market': 0.5}]
    p = inputs.reproduce_base16(rows, model)
    a = 0.4 * math.exp(0.35 * math.tanh(-0.2 / 0.35))
    b = 0.6 * math.exp(0.35 * math.tanh(0.2 / 0.35))
    assert p[:2] == pytest.approx([a/(a+b), b/(a+b)], abs=1e-14)
    assert p[2:] == [0.5, 0.5]
